from typing import List, Optional, Any
from langchain_core.retrievers import BaseRetriever
from core.retriever_strategies.interface import RetrieverStrategyInterface
from core.embedding_manager import embedding_manager
from core.llm_manager import llm_manager
from core.vector_store_manager import vector_store_manager
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from core.retriever_strategies.basic import BasicRetrieverStrategy
import logging

logger = logging.getLogger(__name__)

DEEP_RAG_QUERY_DECOMPOSITION_PROMPT_TEMPLATE_STR = """
You are an expert at query decomposition. Your task is to break down a complex user question into simpler, \
atomic sub-questions that can be answered by a retrieval system.
Generate a list of 2 to 4 sub-questions. Each sub-question should be on a new line.

Original Question: {question}

Sub-questions:
"""

try:
    deep_rag_query_decomposition_prompt = PromptTemplate(
        template=DEEP_RAG_QUERY_DECOMPOSITION_PROMPT_TEMPLATE_STR,
        input_variables=["question"]
    )
except ImportError:
    deep_rag_query_decomposition_prompt = None
    logger.warning("Could not import PromptTemplate for DeepRag query decomposition.")

class DeepRagRetrieverStrategy(RetrieverStrategyInterface):
    def get_name(self) -> str:
        return "deep_rag"

    def _decompose_query(self, question: str, llm_instance: BaseLanguageModel) -> List[str]:
        if not deep_rag_query_decomposition_prompt:
            logger.error("Deep RAG query decomposition prompt is not available.")
            return [question]
        try:
            chain = deep_rag_query_decomposition_prompt | llm_instance | StrOutputParser()
            sub_queries_text = chain.invoke({"question": question})
            sub_queries = [q.strip() for q in sub_queries_text.split("\n") if q.strip()]
            logger.info(f"Decomposed question '{question}' into sub-queries: {sub_queries}")
            return sub_queries if sub_queries else [question]
        except Exception as e:
            logger.error(f"Error during query decomposition for DeepRag: {e}", exc_info=True)
            return [question]

    def get_retriever(
        self,
        user_id: str,
        embedding_strategy_name: str,
        data_source_ids: Optional[List[str]] = None,
        dataset_ids: Optional[List[str]] = None,
        n_results: int = 3,
        max_sub_queries: int = 3,
        **kwargs: Any
    ) -> BaseRetriever:
        llm_instance = llm_manager.get_llm()
        if not llm_instance:
            raise ValueError("LLM instance not available for DeepRagRetrieverStrategy query decomposition.")

        class DeepRagCustomRetriever(BaseRetriever):
            user_id: str
            embedding_strategy_name: str
            data_source_ids: Optional[List[str]]
            dataset_ids: Optional[List[str]]
            n_results_per_subquery: int
            max_sub_queries: int
            llm_for_decomposition: BaseLanguageModel

            def _decompose_query_internal(self, question: str) -> List[str]:
                if not deep_rag_query_decomposition_prompt:
                    logger.error("Deep RAG query decomposition prompt is not available.")
                    return [question]
                try:
                    chain = deep_rag_query_decomposition_prompt | self.llm_for_decomposition | StrOutputParser()
                    sub_queries_text = chain.invoke({"question": question})
                    sub_queries = [q.strip() for q in sub_queries_text.split("\n") if q.strip()]
                    logger.info(f"DeepRAG (internal): Decomposed question '{question}' into sub-queries: {sub_queries}")
                    return sub_queries if sub_queries else [question]
                except Exception as e:
                    logger.error(f"DeepRAG (internal): Error during query decomposition: {e}", exc_info=True)
                    return [question]

            def _get_relevant_documents(self, query: str, *, run_manager: Any) -> List[Document]:
                sub_queries = self._decompose_query_internal(query)

                all_retrieved_docs: List[Document] = []
                doc_ids = set()

                base_retriever_strategy = BasicRetrieverStrategy()
あ
                for sub_query in sub_queries[:self.max_sub_queries]:
                    logger.info(f"DeepRAG: Retrieving for sub-query: '{sub_query}'")
                    try:
                        retriever_for_subquery = base_retriever_strategy.get_retriever(
                            user_id=self.user_id,
                            embedding_strategy_name=self.embedding_strategy_name,
                            data_source_ids=self.data_source_ids,
                            dataset_ids=self.dataset_ids,
                            n_results=self.n_results_per_subquery
                        )
                        docs = retriever_for_subquery.invoke(sub_query)
                        for doc in docs:
                            doc_id = doc.metadata.get("data_source_id", "") + "_" + doc.page_content[:50]
                            if doc_id not in doc_ids:
                                all_retrieved_docs.append(doc)
                                doc_ids.add(doc_id)
                    except Exception as e:
                        logger.error(f"DeepRAG: Error retrieving for sub-query '{sub_query}': {e}", exc_info=True)

                logger.info(f"DeepRAG: Retrieved {len(all_retrieved_docs)} unique documents from {len(sub_queries)} sub-queries.")
                return all_retrieved_docs

        custom_retriever = DeepRagCustomRetriever(
            user_id=user_id,
            embedding_strategy_name=embedding_strategy_name,
            data_source_ids=data_source_ids,
            dataset_ids=dataset_ids,
            n_results_per_subquery=n_results,
            max_sub_queries=max_sub_queries,
            llm_for_decomposition=llm_instance
        )
        return custom_retriever