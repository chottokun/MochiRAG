from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
import importlib
import inspect
import logging

from langchain.retrievers import EnsembleRetriever
from langchain.chains import LLMChain
from langchain.retrievers import (ContextualCompressionRetriever,
                                  MultiQueryRetriever)
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
from .prompts import DEFAULT_STEP_BACK_TEMPLATE, DEFAULT_ACE_TOPIC_TEMPLATE

logger = logging.getLogger(__name__)

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
        # Continue building retrievers when dataset_ids is provided
        retrievers: List[Any] = []
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
                    logger.warning(f"The following shared dataset IDs were not found in shared_dbs.json: {unmapped_ids}")

                for collection_name, ids in collection_to_ids_map.items():
                    shared_vector_store = vector_store_manager.get_vector_store(collection_name)
                    shared_filter = {"dataset_id": {"$in": ids}}
                    retrievers.append(
                        shared_vector_store.as_retriever(search_kwargs={"k": search_k, "filter": shared_filter})
                    )

            except (FileNotFoundError, json.JSONDecodeError) as e:
                # Log a warning if the shared DBs config is missing or invalid
                logger.warning(f"Could not load or parse shared_dbs.json: {e}")

        # 3. Build the final retriever
        if not retrievers:
            # If no valid retrievers could be created, return one that finds nothing.
            # This is a safe fallback.
            empty_vs = vector_store_manager.get_vector_store(f"user_{user_id}") # Dummy collection
            return empty_vs.as_retriever(search_kwargs={"k": 0})
        else:
            # Before returning, ensure retrievers are Runnable/BaseRetriever compatible
            normalized: List[BaseRetriever] = []
            for r in retrievers:
                if isinstance(r, Runnable):
                    normalized.append(r)
                else:
                    normalized.append(_BaseRetrieverAdapter(r))

            if len(normalized) == 1:
                return normalized[0]

            # Use EnsembleRetriever for multiple sources
            # The weights are distributed evenly.
            weights = [1.0 / len(normalized)] * len(normalized)
            return EnsembleRetriever(retrievers=normalized, weights=weights)

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


# NOTE: HyDE-related functionality (HypotheticalDocumentEmbedder / Hyde retriever)
# has been removed from the codebase due to compatibility issues with some
# runtime LangChain builds. If you need HyDE in future, reintroduce a robust
# implementation with careful version pinning or an adapter for Runnable types.


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


# Adapter to wrap objects (e.g., MagicMock from tests) so they satisfy
# langchain's BaseRetriever / Runnable type checks when used in EnsembleRetriever.
class _BaseRetrieverAdapter(BaseRetriever):
    def __init__(self, delegate: Any):
        # delegate is expected to have a `get_relevant_documents` method
        self._delegate = delegate

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        # Delegate to the underlying retriever (tests provide MagicMocks with this method)
        try:
            return self._delegate.get_relevant_documents(query, callbacks=run_manager.get_child())
        except TypeError:
            # Fallback if delegate expects different signature
            return self._delegate.get_relevant_documents(query)


class StepBackPromptingRetrieverStrategy(RetrieverStrategy):
    def get_retriever(self, user_id: int, dataset_ids: Optional[List[int]] = None) -> BaseRetriever:
        # Use the basic retriever as the underlying search mechanism
        base_retriever = BasicRetrieverStrategy().get_retriever(user_id, dataset_ids)
        llm = llm_manager.get_llm()

        # Prompt for generating the step-back question
        template = config_manager.get_prompt("step_back", default=DEFAULT_STEP_BACK_TEMPLATE)
        prompt = PromptTemplate.from_template(template)

        # Chain to generate the question
        question_gen_chain = prompt | llm | StrOutputParser()

        return StepBackRetriever(
            retriever=base_retriever,
            question_gen_chain=question_gen_chain
        )


class ACERetriever(BaseRetriever):
    """
    A retriever that combines results from a base retriever with
    insights from the EvolvedContext database.
    """
    base_retriever: BaseRetriever
    user_id: int
    topic_gen_chain: Runnable
    latest_topic: str = ""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # 1. Get standard documents from the base retriever
        base_docs = self.base_retriever.get_relevant_documents(
            query, callbacks=run_manager.get_child()
        )

        # 2. Generate a topic from the user's query and store it
        topic = self.topic_gen_chain.invoke(
            {"question": query},
            config={"callbacks": run_manager.get_child()}
        )
        self.latest_topic = topic.strip()
        logger.info(f"Generated topic '{topic}' for ACE retrieval.")

        # 3. Retrieve evolved contexts from the database
        db = SessionLocal()
        try:
            evolved_contexts = crud.get_evolved_contexts_by_topic(
                db, owner_id=self.user_id, topic=topic.strip()
            )
        finally:
            db.close()

        # 4. Convert them to LangChain Documents
        evolved_docs = []
        for context in evolved_contexts:
            doc = Document(
                page_content=context.content,
                metadata={
                    "source": "evolved_context",
                    "id": context.id,
                    "score": context.effectiveness_score,
                },
            )
            evolved_docs.append(doc)

        logger.info(f"Retrieved {len(evolved_docs)} documents from evolved contexts.")

        # 5. Combine and return the documents (evolved docs first for priority)
        return evolved_docs + base_docs


class ACERetrieverStrategy(RetrieverStrategy):
    def get_retriever(self, user_id: int, dataset_ids: Optional[List[int]] = None) -> BaseRetriever:
        base_retriever = BasicRetrieverStrategy().get_retriever(user_id, dataset_ids)
        llm = llm_manager.get_llm()

        # Prompt for generating a concise topic/keyword for DB lookup
        template = config_manager.get_prompt("ace_topic", default=DEFAULT_ACE_TOPIC_TEMPLATE)
        prompt = PromptTemplate.from_template(template)

        topic_gen_chain = prompt | llm | StrOutputParser()

        return ACERetriever(
            base_retriever=base_retriever,
            user_id=user_id,
            topic_gen_chain=topic_gen_chain
        )


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