from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json

from langchain.retrievers import EnsembleRetriever
from langchain.chains import LLMChain
from langchain.retrievers import (ContextualCompressionRetriever,
                                  MultiQueryRetriever)
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HypotheticalDocumentEmbedder
from langchain_chroma import Chroma
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
        if not dataset_ids:
            # If no specific datasets are requested, default to all documents for the user.
            collection_name = f"user_{user_id}"
            vector_store = vector_store_manager.get_vector_store(collection_name)
            return vector_store.as_retriever(search_kwargs={"filter": {"user_id": user_id}})

        retrievers = []
        config = config_manager.get_retriever_config("basic")
        search_k = config.parameters.get("k", 5)

        # 1. Separate personal and shared dataset IDs
        personal_ids = [ds_id for ds_id in dataset_ids if ds_id > 0]
        if personal_ids:
            personal_collection_name = f"user_{user_id}"
            personal_vector_store = vector_store_manager.get_vector_store(personal_collection_name)
            personal_filter = {"$and": [{"user_id": user_id}, {"dataset_id": {"$in": personal_ids}}]}
            retrievers.append(
                personal_vector_store.as_retriever(search_kwargs={"k": search_k, "filter": personal_filter})
            )

        # 2. Handle shared datasets
        shared_ids = [ds_id for ds_id in dataset_ids if ds_id < 0]
        if shared_ids:
            try:
                with open("shared_dbs.json", "r") as f:
                    shared_dbs_config = json.load(f)
                
                id_to_collection_map = {db["id"]: db["collection_name"] for db in shared_dbs_config}
                
                collection_to_ids_map = {}
                unmapped_ids = []
                for ds_id in shared_ids:
                    collection_name = id_to_collection_map.get(ds_id)
                    if collection_name:
                        if collection_name not in collection_to_ids_map:
                            collection_to_ids_map[collection_name] = []
                        collection_to_ids_map[collection_name].append(ds_id)
                    else:
                        unmapped_ids.append(ds_id)

                if unmapped_ids:
                    # Log a warning for dataset IDs not found in the config
                    # In a real app, you'd use a proper logger
                    print(f"Warning: The following shared dataset IDs were not found in shared_dbs.json: {unmapped_ids}")

                for collection_name, ids in collection_to_ids_map.items():
                    shared_vector_store = vector_store_manager.get_vector_store(collection_name)
                    shared_filter = {"dataset_id": {"$in": ids}}
                    retrievers.append(
                        shared_vector_store.as_retriever(search_kwargs={"k": search_k, "filter": shared_filter})
                    )

            except (FileNotFoundError, json.JSONDecodeError) as e:
                # Log a warning if the shared DBs config is missing or invalid
                print(f"Warning: Could not load or parse shared_dbs.json: {e}")

        # 3. Build the final retriever
        if not retrievers:
            # If no valid retrievers could be created, return one that finds nothing.
            # This is a safe fallback.
            empty_vs = vector_store_manager.get_vector_store(f"user_{user_id}") # Dummy collection
            return empty_vs.as_retriever(search_kwargs={"k": 0})
        elif len(retrievers) == 1:
            return retrievers[0]
        else:
            # Use EnsembleRetriever for multiple sources
            # The weights are distributed evenly.
            weights = [1.0 / len(retrievers)] * len(retrievers)
            return EnsembleRetriever(retrievers=retrievers, weights=weights)

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


class HydeRetrieverStrategy(RetrieverStrategy):
    def get_retriever(self, user_id: int, dataset_ids: Optional[List[int]] = None) -> BaseRetriever:
        config = config_manager.get_retriever_config("hyde")
        search_k = config.parameters.get("k", 5)
        llm = llm_manager.get_llm()
        template = """Please write a passage to answer the question.
Question: {question}
Passage:"""
        prompt = PromptTemplate.from_template(template)
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        # This method creates a HyDE-powered retriever for a given vector store
        def create_hyde_retriever(vector_store: Chroma, filter_dict: dict) -> BaseRetriever:
            base_embeddings = vector_store.embeddings
            hyde_embeddings = HypotheticalDocumentEmbedder(
                llm_chain=llm_chain,
                base_embeddings=base_embeddings,
            )
            # To avoid side effects, create a new Chroma instance with the new embedder
            # instead of modifying the one from the manager.
            temp_vector_store = Chroma(
                client=vector_store._client,
                collection_name=vector_store._collection.name,
                embedding_function=hyde_embeddings,
            )
            return temp_vector_store.as_retriever(
                search_kwargs={"k": search_k, "filter": filter_dict}
            )

        if not dataset_ids:
            # Default to all user documents
            vector_store = vector_store_manager.get_vector_store(f"user_{user_id}")
            return create_hyde_retriever(vector_store, {"user_id": user_id})

        retrievers = []

        # 1. Handle personal datasets
        personal_ids = [ds_id for ds_id in dataset_ids if ds_id > 0]
        if personal_ids:
            vector_store = vector_store_manager.get_vector_store(f"user_{user_id}")
            p_filter = {"$and": [{"user_id": user_id}, {"dataset_id": {"$in": personal_ids}}]}
            retrievers.append(create_hyde_retriever(vector_store, p_filter))

        # 2. Handle shared datasets
        shared_ids = [ds_id for ds_id in dataset_ids if ds_id < 0]
        if shared_ids:
            try:
                with open("shared_dbs.json", "r") as f:
                    shared_dbs_config = json.load(f)

                id_to_collection_map = {db["id"]: db["collection_name"] for db in shared_dbs_config}
                collection_to_ids_map = {}
                unmapped_ids = []
                for ds_id in shared_ids:
                    collection_name = id_to_collection_map.get(ds_id)
                    if collection_name:
                        if collection_name not in collection_to_ids_map:
                            collection_to_ids_map[collection_name] = []
                        collection_to_ids_map[collection_name].append(ds_id)
                    else:
                        unmapped_ids.append(ds_id)

                if unmapped_ids:
                    print(f"Warning: The following shared dataset IDs were not found in shared_dbs.json: {unmapped_ids}")

                for collection_name, ids in collection_to_ids_map.items():
                    vector_store = vector_store_manager.get_vector_store(collection_name)
                    s_filter = {"dataset_id": {"$in": ids}}
                    retrievers.append(create_hyde_retriever(vector_store, s_filter))

            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Warning: Could not load or parse shared_dbs.json: {e}")

        # 3. Build the final retriever
        if not retrievers:
            empty_vs = vector_store_manager.get_vector_store(f"user_{user_id}")
            return empty_vs.as_retriever(search_kwargs={"k": 0})
        elif len(retrievers) == 1:
            return retrievers[0]
        else:
            weights = [1.0 / len(retrievers)] * len(retrievers)
            return EnsembleRetriever(retrievers=retrievers, weights=weights)


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