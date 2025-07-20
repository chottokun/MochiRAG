from abc import ABC, abstractmethod
from typing import List, Optional, Any
from langchain_core.retrievers import BaseRetriever

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