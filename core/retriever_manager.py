from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import MultiQueryRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.stores import BaseStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever as LangchainParentDocumentRetriever

from backend import crud
from backend.database import SessionLocal
from .config_manager import config_manager
from .vector_store_manager import vector_store_manager
from .llm_manager import llm_manager

# --- Custom Docstore for ParentDocumentRetriever ---

class SQLDocStore(BaseStore[str, Document]):
    def __init__(self):
        self.db = SessionLocal()

    def __del__(self):
        self.db.close()

    def mget(self, keys: Sequence[str]) -> List[Optional[Document]]:
        db_docs = crud.mget_parent_documents(self.db, doc_ids=keys)
        # Create a dictionary for quick lookups
        doc_map = {doc.id: Document(page_content=doc.content) for doc in db_docs}
        return [doc_map.get(key) for key in keys]

    def mset(self, key_value_pairs: Sequence[Tuple[str, Document]]) -> None:
        # This docstore is read-only from the retriever's perspective.
        # The ingestion service handles writing to the database.
        pass

    def mdelete(self, keys: Sequence[str]) -> None:
        # Not implemented
        pass

    def yield_keys(self, *, prefix: Optional[str] = None) -> None:
        # Not implemented
        pass

# --- Abstract Base Class for Strategies ---

class RetrieverStrategy(ABC):
    @abstractmethod
    def get_retriever(self, user_id: int, dataset_ids: Optional[List[int]] = None) -> BaseRetriever:
        pass

# --- Concrete Strategy Implementations ---

class BasicRetrieverStrategy(RetrieverStrategy):
    def get_retriever(self, user_id: int, dataset_ids: Optional[List[int]] = None) -> BaseRetriever:
        config = config_manager.get_retriever_config("basic")
        collection_name = f"user_{user_id}"
        vector_store = vector_store_manager.get_vector_store(collection_name)

        filter = {"user_id": user_id}
        if dataset_ids:
            filter = {"$and": [
                {"user_id": user_id},
                {"dataset_id": {"$in": dataset_ids}}
            ]}

        return vector_store.as_retriever(
            search_kwargs={
                "k": config.parameters.get("k", 5),
                "filter": filter
            }
        )

class MultiQueryRetrieverStrategy(RetrieverStrategy):
    def get_retriever(self, user_id: int, dataset_ids: Optional[List[int]] = None) -> BaseRetriever:
        base_retriever = BasicRetrieverStrategy().get_retriever(user_id, dataset_ids)
        llm = llm_manager.get_llm()
        return MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm
        )

class ContextualCompressionRetrieverStrategy(RetrieverStrategy):
    def get_retriever(self, user_id: int, dataset_ids: Optional[List[int]] = None) -> BaseRetriever:
        base_retriever = BasicRetrieverStrategy().get_retriever(user_id, dataset_ids)
        llm = llm_manager.get_llm()
        compressor = LLMChainExtractor.from_llm(llm)
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )

class ParentDocumentRetrieverStrategy(RetrieverStrategy):
    def get_retriever(self, user_id: int, dataset_ids: Optional[List[int]] = None) -> BaseRetriever:
        collection_name = f"user_{user_id}"
        vector_store = vector_store_manager.get_vector_store(collection_name)
        store = SQLDocStore()
        
        config = config_manager.get_retriever_config("parent_document")
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=config.parameters.get("parent_chunk_size", 2000))
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=config.parameters.get("child_chunk_size", 400))

        return LangchainParentDocumentRetriever(
            vectorstore=vector_store,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )

# --- RetrieverManager --- 

class RetrieverManager:
    def __init__(self):
        self.strategies = {
            "basic": BasicRetrieverStrategy(),
            "multiquery": MultiQueryRetrieverStrategy(),
            "compression": ContextualCompressionRetrieverStrategy(),
            "parent_document": ParentDocumentRetrieverStrategy(),
        }

    def get_retriever(self, strategy_name: str, user_id: int, dataset_ids: Optional[List[int]] = None) -> BaseRetriever:
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown retriever strategy: {strategy_name}")
        
        strategy = self.strategies[strategy_name]
        return strategy.get_retriever(user_id, dataset_ids)

# Create a single, globally accessible instance
retriever_manager = RetrieverManager()
