from abc import ABC, abstractmethod
from typing import List, Optional

from langchain_core.retrievers import BaseRetriever

from .config_manager import config_manager
from .vector_store_manager import vector_store_manager

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

        # Base filter for multi-tenancy
        filter = {"user_id": user_id}
        
        # If dataset_ids are provided, add them to the filter
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

# --- RetrieverManager --- 

class RetrieverManager:
    def __init__(self):
        self.strategies = {
            "basic": BasicRetrieverStrategy()
            # Future strategies like MultiQueryRetriever can be added here
        }

    def get_retriever(self, strategy_name: str, user_id: int, dataset_ids: Optional[List[int]] = None) -> BaseRetriever:
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown retriever strategy: {strategy_name}")
        
        strategy = self.strategies[strategy_name]
        return strategy.get_retriever(user_id, dataset_ids)

# Create a single, globally accessible instance
retriever_manager = RetrieverManager()
