from typing import List, Optional, Any
from langchain_core.retrievers import BaseRetriever
from core.retriever_strategies.interface import RetrieverStrategyInterface
from core.embedding_manager import embedding_manager
from core.llm_manager import llm_manager
from core.vector_store_manager import vector_store_manager
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from core.retriever_strategies.basic import BasicRetrieverStrategy
import logging

logger = logging.getLogger(__name__)

class ContextualCompressionRetrieverStrategy(RetrieverStrategyInterface):
    def get_name(self) -> str:
        return "contextual_compression"

    def get_retriever(
        self,
        user_id: str,
        embedding_strategy_name: str,
        data_source_ids: Optional[List[str]] = None,
        dataset_ids: Optional[List[str]] = None,
        n_results: int = 5,
        **kwargs: Any
    ) -> BaseRetriever:
        llm_instance = llm_manager.get_llm()
        if not llm_instance:
            raise ValueError("LLM instance not available via LLMManager for ContextualCompressionRetriever.")

        base_retriever = BasicRetrieverStrategy().get_retriever(
            user_id, embedding_strategy_name, data_source_ids, dataset_ids, n_results
        )
        compressor = LLMChainExtractor.from_llm(llm_instance)
        return ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )