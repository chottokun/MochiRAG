import logging
from typing import List, Optional, Dict, Any, Literal
from abc import ABC, abstractmethod

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel

from langchain.retrievers import MultiQueryRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever as LangchainParentDocumentRetriever


try:
    from core.config_loader import load_strategy_config, StrategyConfigError
    from core.vector_store_manager import vector_store_manager
    from core.embedding_manager import embedding_manager
    from core.llm_manager import llm_manager
    from core.document_processor import text_splitter as default_text_splitter
except ImportError as e:
    raise ImportError(f"RetrieverManager failed to import core managers: {e}") from e


logger = logging.getLogger(__name__)


# RAG戦略の型定義 (core.rag_chain.py と共通化するのが望ましい)
RAG_STRATEGY_TYPE = Literal[
    "simple_rag",
    "advanced_rag",
    "deep_rag",
    "multi_query",
    "contextual_compression",
    "parent_document",
]
AVAILABLE_RAG_STRATEGIES = list(RAG_STRATEGY_TYPE.__args__)


# MultiQueryRetriever用プロンプト (core.rag_chain.py 等と共通化)
# TODO: これもconfigから読み込めるようにする
QUERY_GEN_PROMPT_TEMPLATE_STR = """You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from a vector
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of distance-based similarity search.
Provide these alternative questions separated by newlines.
Original question: {question}"""
try:
    from langchain_core.prompts import PromptTemplate
    query_gen_prompt = PromptTemplate(
        template=QUERY_GEN_PROMPT_TEMPLATE_STR, input_variables=["question"]
    )
except ImportError:
    query_gen_prompt = None  # type: ignore
    logger.warning("Could not import PromptTemplate for MultiQueryRetriever.")


# DeepRag用プロンプトテンプレート
DEEP_RAG_QUERY_DECOMPOSITION_PROMPT_TEMPLATE_STR = """
You are an expert at query decomposition. Your task is to break down a complex user question into simpler, \
atomic sub-questions that can be answered by a retrieval system.
Generate a list of 2 to 4 sub-questions. Each sub-question should be on a new line.

Original Question: {question}

Sub-questions:
"""
try:
    from langchain_core.prompts import PromptTemplate
    deep_rag_query_decomposition_prompt = PromptTemplate(
        template=DEEP_RAG_QUERY_DECOMPOSITION_PROMPT_TEMPLATE_STR,
        input_variables=["question"]
    )
except ImportError:
    deep_rag_query_decomposition_prompt = None  # type: ignore
    logger.warning("Could not import PromptTemplate for DeepRag query decomposition.")


class RetrieverStrategyInterface(ABC):
    """リトリーバー戦略のインターフェース"""

    @abstractmethod
    def get_retriever(
        self,
        user_id: str,
        embedding_strategy_name: str,
        data_source_ids: Optional[List[str]] = None,
        dataset_ids: Optional[List[str]] = None,
        n_results: int = 3,
        **kwargs: Any
    ) -> BaseRetriever:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass


class BasicRetrieverStrategy(RetrieverStrategyInterface):
    def get_name(self) -> str:
        return "basic"

    def get_retriever(
        self, user_id: str, embedding_strategy_name: str,
        data_source_ids: Optional[List[str]] = None,
        dataset_ids: Optional[List[str]] = None,
        n_results: int = 3, **kwargs: Any
    ) -> BaseRetriever:
        embedding_function = embedding_manager.get_embedding_model(
            embedding_strategy_name
        )

        try:
            from langchain_chroma import Chroma
        except ImportError:
            logger.error("Langchain_chroma not installed, cannot create Chroma client for BasicRetriever.")
            raise

        vectorstore = Chroma(
            persist_directory=vector_store_manager.persist_directory,
            embedding_function=embedding_function
        )

        filter_conditions: List[Dict[str, Any]] = [{"user_id": user_id}]
        if data_source_ids:
            if len(data_source_ids) == 1:
                filter_conditions.append({"data_source_id": data_source_ids[0]})
            elif len(data_source_ids) > 1:
                filter_conditions.append({"data_source_id": {"$in": data_source_ids}})

        if dataset_ids:
            if len(dataset_ids) == 1:
                filter_conditions.append({"dataset_id": dataset_ids[0]})
            elif len(dataset_ids) > 1:
                filter_conditions.append({"dataset_id": {"$in": dataset_ids}})

        final_filter: Optional[Dict[str, Any]] = None
        if filter_conditions:
            if len(filter_conditions) > 1:
                final_filter = {"$and": filter_conditions}
            elif len(filter_conditions) == 1:
                final_filter = filter_conditions[0]

        search_kwargs: Dict[str, Any] = {"k": n_results}
        if final_filter:
            search_kwargs["filter"] = final_filter

        return vectorstore.as_retriever(search_kwargs=search_kwargs)


class SimpleRagRetrieverStrategy(RetrieverStrategyInterface):
    def get_name(self) -> str:
        return "simple_rag"

    def get_retriever(
        self, user_id: str, embedding_strategy_name: str,
        data_source_ids: Optional[List[str]] = None,
        dataset_ids: Optional[List[str]] = None,
        n_results: int = 3, **kwargs: Any
    ) -> BaseRetriever:
        return BasicRetrieverStrategy().get_retriever(
            user_id, embedding_strategy_name, data_source_ids, dataset_ids, n_results
        )


class AdvancedRagRetrieverStrategy(RetrieverStrategyInterface):
    def get_name(self) -> str:
        return "advanced_rag"

    def get_retriever(
        self, user_id: str, embedding_strategy_name: str,
        data_source_ids: Optional[List[str]] = None,
        dataset_ids: Optional[List[str]] = None,
        n_results: int = 5, **kwargs: Any
    ) -> BaseRetriever:
        llm_instance = llm_manager.get_llm()
        if not llm_instance:
            raise ValueError("LLM instance not available via LLMManager for AdvancedRagRetriever.")

        base_retriever = BasicRetrieverStrategy().get_retriever(
            user_id, embedding_strategy_name, data_source_ids, dataset_ids, n_results
        )
        compressor = LLMChainExtractor.from_llm(llm_instance)
        return ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )


class MultiQueryRetrieverStrategy(RetrieverStrategyInterface):
    def get_name(self) -> str:
        return "multi_query"

    def get_retriever(
        self, user_id: str, embedding_strategy_name: str,
        data_source_ids: Optional[List[str]] = None,
        dataset_ids: Optional[List[str]] = None,
        n_results: int = 3, **kwargs: Any
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


class ContextualCompressionRetrieverStrategy(RetrieverStrategyInterface):
    def get_name(self) -> str:
        return "contextual_compression"

    def get_retriever(
        self, user_id: str, embedding_strategy_name: str,
        data_source_ids: Optional[List[str]] = None,
        dataset_ids: Optional[List[str]] = None,
        n_results: int = 5, **kwargs: Any
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


class ParentDocumentRetrieverStrategy(RetrieverStrategyInterface):
    def get_name(self) -> str:
        return "parent_document"

    def get_retriever(
        self, user_id: str, embedding_strategy_name: str,
        data_source_ids: Optional[List[str]] = None,
        dataset_ids: Optional[List[str]] = None,
        n_results: int = 3, **kwargs: Any
    ) -> BaseRetriever:
        logger.info("Attempting to initialize ParentDocumentRetriever.")
        if not default_text_splitter:
            logger.warning(
                "Default text_splitter not available for ParentDocumentRetriever. "
                "Falling back to basic."
            )
            return BasicRetrieverStrategy().get_retriever(
                user_id, embedding_strategy_name, data_source_ids, dataset_ids, n_results
            )

        embedding_function = embedding_manager.get_embedding_model(embedding_strategy_name)
        try:
            from langchain_chroma import Chroma
        except ImportError:
            logger.error("Langchain_chroma not installed, cannot create Chroma client for ParentDocumentRetriever.")
            raise

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


class DeepRagRetrieverStrategy(RetrieverStrategyInterface):
    def get_name(self) -> str:
        return "deep_rag"

    def _decompose_query(self, question: str, llm_instance: BaseLanguageModel) -> List[str]:
        """LLMを使って質問をサブクエリに分解する"""
        if not deep_rag_query_decomposition_prompt:
            logger.error("Deep RAG query decomposition prompt is not available.")
            return [question]

        try:
            from langchain_core.output_parsers import StrOutputParser
            chain = deep_rag_query_decomposition_prompt | llm_instance | StrOutputParser()
            sub_queries_text = chain.invoke({"question": question})

            sub_queries = [q.strip() for q in sub_queries_text.split("\n") if q.strip()]
            logger.info(f"Decomposed question '{question}' into sub-queries: {sub_queries}")
            return sub_queries if sub_queries else [question]
        except Exception as e:
            logger.error(f"Error during query decomposition for DeepRag: {e}", exc_info=True)
            return [question]

    def get_retriever(
        self, user_id: str, embedding_strategy_name: str,
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

            def _get_relevant_documents(self, query: str, *, run_manager: Any) -> List[Document]:
                sub_queries = self._decompose_query_internal(query)

                all_retrieved_docs: List[Document] = []
                doc_ids = set()

                base_retriever_strategy = BasicRetrieverStrategy()

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

                logger.info(
                    f"DeepRAG: Retrieved {len(all_retrieved_docs)} unique documents from "
                    f"{len(sub_queries)} sub-queries."
                )
                return all_retrieved_docs

            def _decompose_query_internal(self, question: str) -> List[str]:
                if not deep_rag_query_decomposition_prompt:
                    logger.error("Deep RAG query decomposition prompt is not available.")
                    return [question]
                try:
                    from langchain_core.output_parsers import StrOutputParser
                    chain = deep_rag_query_decomposition_prompt | self.llm_for_decomposition | StrOutputParser()
                    sub_queries_text = chain.invoke({"question": question})
                    sub_queries = [q.strip() for q in sub_queries_text.split("\n") if q.strip()]
                    logger.info(
                        f"DeepRAG (internal): Decomposed question '{question}' into sub-queries: "
                        f"{sub_queries}"
                    )
                    return sub_queries if sub_queries else [question]
                except Exception as e:
                    logger.error(f"DeepRAG (internal): Error during query decomposition: {e}", exc_info=True)
                    return [question]

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


class RetrieverManager:
    def __init__(self, config_path: Optional[str] = None):
        self.strategies: Dict[str, RetrieverStrategyInterface] = {}
        self.default_strategy_name: Optional[str] = None
        self._load_strategies_from_config(config_path)

    def _load_strategies_from_config(self, config_path: Optional[str] = None):
        try:
            config = load_strategy_config()
        except StrategyConfigError as e:
            logger.error(
                f"RetrieverManager: Failed to load strategy configuration: {e}. "
                "No strategies will be available.",
                exc_info=True
            )
            return

        rag_search_config = config.get("rag_search_strategies")
        if not isinstance(rag_search_config, dict):
            logger.warning(
                "RetrieverManager: 'rag_search_strategies' section not found or invalid in config. "
                "No strategies loaded."
            )
            return

        self.default_strategy_name = rag_search_config.get("default")
        available_configs = rag_search_config.get("available")

        if not isinstance(available_configs, list):
            logger.warning(
                "RetrieverManager: 'rag_search_strategies.available' section not found or not a list. "
                "No strategies loaded."
            )
            return

        logger.info(
            f"RetrieverManager: Loading RAG search strategies. Default: {self.default_strategy_name}, "
            f"Available configs: {available_configs}"
        )

        for strat_conf in available_configs:
            logger.debug(f"RetrieverManager: Processing strategy config: {strat_conf}")
            if not isinstance(strat_conf, dict):
                logger.warning(
                    f"RetrieverManager: Skipping invalid RAG search strategy config item (not a dict): "
                    f"{strat_conf}"
                )
                continue

            name = strat_conf.get("name")
            # 将来的には type フィールドを見て動的にクラスを選択する方がより柔軟

            if not name:
                logger.warning(f"RetrieverManager: Skipping RAG search strategy with no name: {strat_conf}")
                continue

            strat_type = strat_conf.get("type")
            if not strat_type:
                logger.warning(
                    f"RetrieverManager: RAG search strategy '{name}' has no 'type' defined in config. "
                    "Skipping."
                )
                continue

            strategy_instance: Optional[RetrieverStrategyInterface] = None
            try:
                if strat_type == "basic":
                    strategy_instance = BasicRetrieverStrategy()
                elif strat_type == "simple_rag":
                    strategy_instance = SimpleRagRetrieverStrategy()
                elif strat_type == "advanced_rag":
                    strategy_instance = AdvancedRagRetrieverStrategy()
                elif strat_type == "multi_query":
                    strategy_instance = MultiQueryRetrieverStrategy()
                elif strat_type == "contextual_compression":
                    strategy_instance = ContextualCompressionRetrieverStrategy()
                elif strat_type == "parent_document":
                    strategy_instance = ParentDocumentRetrieverStrategy()
                elif strat_type == "deep_rag":
                    strategy_instance = DeepRagRetrieverStrategy()
                else:
                    logger.warning(
                        f"RetrieverManager: Unsupported RAG search strategy type: '{strat_type}' "
                        f"for strategy '{name}'."
                    )
                    continue

                if strategy_instance:
                    self.strategies[name] = strategy_instance
                    logger.info(f"RetrieverManager: Registered RAG search strategy: {name} (type: {strat_type})")
            except Exception as e:
                logger.error(
                    f"RetrieverManager: Failed to initialize RAG search strategy '{name}': {e}",
                    exc_info=True
                )

        if self.default_strategy_name and self.default_strategy_name not in self.strategies:
            logger.warning(
                f"RetrieverManager: Default RAG search strategy '{self.default_strategy_name}' "
                "from config not found or failed to load. Default will be unset."
            )
            self.default_strategy_name = None

        if not self.default_strategy_name and self.strategies:
            self.default_strategy_name = list(self.strategies.keys())[0]
            logger.info(
                "RetrieverManager: No valid default RAG search strategy specified. "
                f"Using first available: '{self.default_strategy_name}'"
            )
        elif not self.strategies:
            logger.warning("RetrieverManager: No RAG search strategies were loaded or registered.")

    def get_retriever(
        self,
        user_id: str,
        embedding_strategy_name: str,
        name: Optional[RAG_STRATEGY_TYPE] = None,
        data_source_ids: Optional[List[str]] = None,
        dataset_ids: Optional[List[str]] = None,
        n_results: Optional[int] = None,
        max_sub_queries: Optional[int] = None,
        n_results_per_subquery: Optional[int] = None,
        **kwargs: Any
    ) -> BaseRetriever:
        target_name_str = name if name else self.default_strategy_name
        if not target_name_str:
            raise ValueError("No RAG search strategy name specified and no default is set.")

        strategy_instance = self.strategies.get(target_name_str)
        if not strategy_instance:
            logger.error(f"RAG search strategy '{target_name_str}' not implemented/registered.")
            if self.default_strategy_name and self.default_strategy_name in self.strategies:
                logger.warning(f"Falling back to default RAG search strategy: {self.default_strategy_name}")
                strategy_instance = self.strategies[self.default_strategy_name]
            else:
                raise ValueError(f"RAG search strategy '{target_name_str}' not found and no default available.")

        final_n_results = n_results if n_results is not None else 3

        if target_name_str == "deep_rag":
            if max_sub_queries is not None:
                kwargs["max_sub_queries"] = max_sub_queries
            if n_results_per_subquery is not None:
                kwargs["n_results_per_subquery"] = n_results_per_subquery

        return strategy_instance.get_retriever(
            user_id=user_id,
            embedding_strategy_name=embedding_strategy_name,
            data_source_ids=data_source_ids,
            dataset_ids=dataset_ids,
            n_results=final_n_results,
            **kwargs
        )

    def get_available_strategies(self) -> List[str]:
        return list(self.strategies.keys())


retriever_manager = RetrieverManager()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("Available RAG search strategies from manager:")
    for name in retriever_manager.get_available_strategies():
        print(f"- {name}")

    test_user = "retriever_mgr_test_user"

    default_emb_strat = (
        embedding_manager.get_available_strategies()[0]
        if embedding_manager.get_available_strategies()
        else "dummy_emb_strat"
    )

    if not retriever_manager.get_available_strategies():
        print("No RAG search strategies loaded, cannot run tests.")
    else:
        for strategy_name_str in retriever_manager.get_available_strategies():
            strategy_name: RAG_STRATEGY_TYPE = strategy_name_str  # type: ignore

            if strategy_name in ["multi_query", "contextual_compression"] and not llm_manager.get_llm():
                print(f"Skipping '{strategy_name}' test as LLM is not available.")
                continue

            print(f"\n--- Testing Retriever Strategy from Manager: {strategy_name} ---")
            try:
                retriever = retriever_manager.get_retriever(
                    name=strategy_name,
                    user_id=test_user,
                    embedding_strategy_name=default_emb_strat,
                )
                print(f"Successfully got retriever for strategy: {strategy_name} -> {type(retriever)}")

                test_query = "What is the MochiRAG system?"
                retrieved_docs = retriever.invoke(test_query)
                print(
                    f"Retrieved {len(retrieved_docs)} documents for query '{test_query[:30]}...' "
                    f"using {strategy_name}:"
                )
            except Exception as e:
                print(f"Error testing strategy '{strategy_name}': {e}", exc_info=True)

    print("\nRetrieverManager tests finished.")
