from typing import List, Optional, Any
from langchain_core.retrievers import BaseRetriever
from core.retriever_strategies.interface import RetrieverStrategyInterface
from core.embedding_manager import embedding_manager
from core.llm_manager import llm_manager
from core.vector_store_manager import vector_store_manager
from langchain.retrievers import MultiQueryRetriever
from core.retriever_strategies.basic import BasicRetrieverStrategy
from langchain_core.prompts import PromptTemplate
import logging

logger = logging.getLogger(__name__)

QUERY_GEN_PROMPT_TEMPLATE_STR = """You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from a vector
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of distance-based similarity search.
Provide these alternative questions separated by newlines.
Original question: {question}"""

try:
    query_gen_prompt = PromptTemplate(template=QUERY_GEN_PROMPT_TEMPLATE_STR, input_variables=["question"])
except ImportError:
    query_gen_prompt = None
    logger.warning("Could not import PromptTemplate for MultiQueryRetriever.")

class MultiQueryRetrieverStrategy(RetrieverStrategyInterface):
    def get_name(self) -> str:
        return "multi_query"

    def get_retriever(
        self,
        user_id: str,
        embedding_strategy_name: str,
        data_source_ids: Optional[List[str]] = None,
        dataset_ids: Optional[List[str]] = None,
        n_results: int = 3,
        **kwargs: Any
    ) -> BaseRetriever:
        llm_instance = llm_manager.get_llm()
        if not llm_instance:
            raise ValueError("LLM instance not available via LLMManager for MultiQueryRetriever.")
        if not query_gen_prompt:
            raise ValueError("Query generation prompt not available for MultiQueryRetriever.")

        base_retriever = BasicRetrieverStrategy().get_retriever(
            user_id, embedding_strategy_name, data_source_ids, dataset_ids, n_results
        )
        return MultiQueryRetriever.from_llm(
            retriever=base_retriever, llm=llm_instance, prompt=query_gen_prompt
        )