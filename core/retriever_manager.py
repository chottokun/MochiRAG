from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain.chains import LLMChain
from langchain.retrievers import (ContextualCompressionRetriever,
                                  MultiQueryRetriever)
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import HydeRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.stores import BaseStore
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


class CustomHydeRetriever(HydeRetriever):
    """
    Custom HydeRetriever that supports filtering for multi-tenant vector stores.
    """
    search_kwargs: Dict[str, Any] = {}

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Generates a hypothetical document and retrieves relevant documents
        from the vector store, applying configured search filters.
        """
        if self.prompt_key is None:
            result = self.llm_chain.invoke(
                {self.llm_chain.input_keys[0]: query},
                config={"callbacks": run_manager.get_child()},
            )
            hypothetical_document = result[self.llm_chain.output_key]
        else:
            hypothetical_document = self.llm_chain.prompt.format(**{self.prompt_key: query})

        embedding = self.vectorstore.embeddings.embed_query(hypothetical_document)

        # Pass the stored search_kwargs to the search function
        return self.vectorstore.similarity_search_by_vector(
            embedding, **self.search_kwargs
        )


class HydeRetrieverStrategy(RetrieverStrategy):
    def get_retriever(self, user_id: int, dataset_ids: Optional[List[int]] = None) -> BaseRetriever:
        collection_name = f"user_{user_id}"
        vector_store = vector_store_manager.get_vector_store(collection_name)
        llm = llm_manager.get_llm()

        filter_dict = {"user_id": user_id}
        if dataset_ids:
            filter_dict = {"$and": [
                {"user_id": user_id},
                {"dataset_id": {"$in": dataset_ids}}
            ]}

        config = config_manager.get_retriever_config("hyde")
        search_kwargs = {
            "k": config.parameters.get("k", 5),
            "filter": filter_dict
        }

        template = """Please write a passage to answer the question.
Question: {question}
Passage:"""
        prompt = PromptTemplate.from_template(template)
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        return CustomHydeRetriever(
            vectorstore=vector_store,
            llm_chain=llm_chain,
            search_kwargs=search_kwargs,
        )


class StepBackRetriever(BaseRetriever):
    """
    Custom retriever that performs the step-back prompting technique.
    It wraps a standard retriever and uses an LLM to generate a more
    general question to improve document retrieval.
    """
    retriever: BaseRetriever
    question_gen_chain: Runnable

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Generates a step-back question and uses the underlying retriever
        to fetch documents based on it.
        """
        # Generate the step-back question
        step_back_query = self.question_gen_chain.invoke(
            {"question": query},
            config={"callbacks": run_manager.get_child()}
        )

        # Retrieve documents using the new question
        documents = self.retriever.get_relevant_documents(
            step_back_query,
            callbacks=run_manager.get_child()
        )
        return documents


class StepBackPromptingRetrieverStrategy(RetrieverStrategy):
    def get_retriever(self, user_id: int, dataset_ids: Optional[List[int]] = None) -> BaseRetriever:
        # Use the basic retriever as the underlying search mechanism
        base_retriever = BasicRetrieverStrategy().get_retriever(user_id, dataset_ids)
        llm = llm_manager.get_llm()

        # Prompt for generating the step-back question
        template = """You are an expert at world knowledge. I am going to ask you a question. Your job is to formulate a single, more general question that captures the essence of the original question. Frame the question from the perspective of a historian or a researcher.
Original question: {question}
Step-back question:"""
        prompt = PromptTemplate.from_template(template)

        # Chain to generate the question
        question_gen_chain = prompt | llm | StrOutputParser()

        return StepBackRetriever(
            retriever=base_retriever,
            question_gen_chain=question_gen_chain
        )


import importlib
import inspect
import logging

logger = logging.getLogger(__name__)

# --- RetrieverManager --- 

class RetrieverManager:
    def __init__(self):
        self.strategies = {}
        retriever_configs = config_manager.config.retrievers
        current_module = importlib.import_module(__name__)

        for name, config in retriever_configs.items():
            strategy_class_name = config.strategy_class
            if not strategy_class_name:
                continue

            try:
                StrategyClass = getattr(current_module, strategy_class_name)
                if inspect.isclass(StrategyClass) and issubclass(StrategyClass, RetrieverStrategy):
                    self.strategies[name] = StrategyClass()
            except AttributeError:
                logger.info(
                    f"Strategy class '{strategy_class_name}' not found in retriever_manager module, skipping. "
                    "This is expected for non-retriever strategies like DeepRAG."
                )

    def get_retriever(self, strategy_name: str, user_id: int, dataset_ids: Optional[List[int]] = None) -> BaseRetriever:
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown retriever strategy: {strategy_name}")
        
        strategy = self.strategies[strategy_name]
        return strategy.get_retriever(user_id, dataset_ids)

# Create a single, globally accessible instance
retriever_manager = RetrieverManager()
