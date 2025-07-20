import logging
from typing import List, Optional, Dict, Any, Literal
from abc import ABC, abstractmethod

from langchain_core.retrievers import BaseRetriever

try:
    from core.config_loader import load_strategy_config, StrategyConfigError
    from core.embedding_manager import embedding_manager
    from core.vector_store_manager import vector_store_manager
    from core.llm_manager import llm_manager
    from core.document_processor import text_splitter as default_text_splitter
except ImportError as e:
    raise ImportError(f"RetrieverManager failed to import core managers: {e}") from e

import importlib

logger = logging.getLogger(__name__)

RAG_STRATEGY_TYPE = Literal["basic", "parent_document", "multi_query", "contextual_compression", "deep_rag"]
AVAILABLE_RAG_STRATEGIES = list(RAG_STRATEGY_TYPE.__args__)

class RetrieverStrategyInterface(ABC):
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

class RetrieverManager:
    def __init__(self, config_path: Optional[str] = None):
        self.strategies: Dict[str, RetrieverStrategyInterface] = {}
        self.default_strategy_name: Optional[str] = None
        self._load_strategies_from_config(config_path)

    def _load_strategies_from_config(self, config_path: Optional[str] = None):
        try:
            config = load_strategy_config()
        except StrategyConfigError as e:
            logger.error(f"RetrieverManager: Failed to load strategy configuration: {e}. No strategies will be available.", exc_info=True)
            return

        rag_search_config = config.get("rag_search_strategies")
        if not isinstance(rag_search_config, dict):
            logger.warning("RetrieverManager: 'rag_search_strategies' section not found or invalid in config. No strategies loaded.")
            return

        self.default_strategy_name = rag_search_config.get("default")
        available_configs = rag_search_config.get("available")

        if not isinstance(available_configs, list):
            logger.warning("RetrieverManager: 'rag_search_strategies.available' section not found or not a list. No strategies loaded.")
            return

        logger.info(f"RetrieverManager: Loading RAG search strategies. Default: {self.default_strategy_name}, Available configs: {available_configs}")

        for strat_conf in available_configs:
            logger.debug(f"RetrieverManager: Processing strategy config: {strat_conf}")
            if not isinstance(strat_conf, dict):
                logger.warning(f"RetrieverManager: Skipping invalid RAG search strategy config item (not a dict): {strat_conf}")
                continue

            name = strat_conf.get("name")
            strat_type = strat_conf.get("type")

            if not name or not strat_type:
                logger.warning(f"RetrieverManager: Skipping RAG search strategy with missing name or type: {strat_conf}")
                continue

            strategy_instance: Optional[RetrieverStrategyInterface] = None
            try:
                module_name = f"core.retriever_strategies.{strat_type}"
                module = importlib.import_module(module_name)
                class_name = "".join(word.capitalize() for word in strat_type.split("_")) + "RetrieverStrategy"
                strategy_class = getattr(module, class_name, None)
                if strategy_class is None:
                    logger.warning(f"RetrieverManager: Strategy class not found in module {module_name} for strategy '{name}'.")
                    continue
                strategy_instance = strategy_class()
                if strategy_instance is not None:
                    self.strategies[name] = strategy_instance
                    logger.info(f"RetrieverManager: Registered RAG search strategy: {name} (type: {strat_type})")
            except Exception as e:
                logger.error(f"RetrieverManager: Failed to initialize RAG search strategy '{name}': {e}", exc_info=True)

        if self.default_strategy_name and self.default_strategy_name not in self.strategies:
            logger.warning(f"RetrieverManager: Default RAG search strategy '{self.default_strategy_name}' from config not found or failed to load. Default will be unset.")
            self.default_strategy_name = None

        if not self.default_strategy_name and self.strategies:
            self.default_strategy_name = list(self.strategies.keys())[0]
            logger.info(f"RetrieverManager: No valid default RAG search strategy specified. Using first available: '{self.default_strategy_name}'")
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
