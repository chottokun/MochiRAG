from typing import List, Optional, Any
from langchain_core.retrievers import BaseRetriever
from core.retriever_strategies.interface import RetrieverStrategyInterface
from core.embedding_manager import embedding_manager
from core.vector_store_manager import vector_store_manager
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever as LangchainParentDocumentRetriever
from core.document_processor import text_splitter as default_text_splitter
from langchain_chroma import Chroma
import logging

logger = logging.getLogger(__name__)

class ParentDocumentRetrieverStrategy(RetrieverStrategyInterface):
    def get_name(self) -> str:
        return "parent_document"

    def get_retriever(
        self,
        user_id: str,
        embedding_strategy_name: str,
        data_source_ids: Optional[List[str]] = None,
        dataset_ids: Optional[List[str]] = None,
        n_results: int = 3,
        **kwargs: Any
    ) -> BaseRetriever:
        logger.info("Attempting to initialize ParentDocumentRetriever.")
        if not default_text_splitter:
            logger.warning("Default text_splitter not available for ParentDocumentRetriever. Falling back to basic.")
            from core.retriever_strategies.basic import BasicRetrieverStrategy
            return BasicRetrieverStrategy().get_retriever(user_id, embedding_strategy_name, data_source_ids, dataset_ids, n_results)

        embedding_function = embedding_manager.get_embedding_model(embedding_strategy_name)

        vectorstore = Chroma(
            persist_directory=vector_store_manager.persist_directory,
            embedding_function=embedding_function,
        )
        docstore = InMemoryStore()

        logger.warning(
            "ParentDocumentRetriever created, but its effectiveness depends on "
            "how documents were added to the store (requires parent docs in docstore "
            "and child chunks in vectorstore, typically via this retriever's add_documents)."
            "Current setup might not fully support this. Querying child chunks directly."
        )

        retriever = LangchainParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            child_splitter=default_text_splitter,
            search_kwargs={"k": n_results}
        )
        return retriever