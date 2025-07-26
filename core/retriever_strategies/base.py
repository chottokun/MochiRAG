from abc import ABC, abstractmethod
from typing import Dict, Any, List
from langchain_core.documents import Document

class BaseRAGStrategy(ABC):
    """
    全てのRAG戦略が継承すべき抽象基底クラス。
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def get_name(self) -> str:
        """
        戦略の名前を返す。
        """
        pass

    @abstractmethod
    def retrieve(self, user_id: str, question: str, data_source_ids: List[str], embedding_strategy_for_retrieval: str) -> List[Document]:
        """
        与えられた質問に基づいて関連ドキュメントを検索する。
        """
        pass

    @abstractmethod
    def generate_response(self, question: str, retrieved_documents: List[Document]) -> Dict[str, Any]:
        """
        検索されたドキュメントと質問に基づいて応答を生成する。
        """
        pass

    def execute(self, user_id: str, question: str, data_source_ids: List[str], embedding_strategy_for_retrieval: str) -> Dict[str, Any]:
        """
        RAG戦略の実行フローを定義する。
        """
        retrieved_documents = self.retrieve(user_id, question, data_source_ids, embedding_strategy_for_retrieval)
        response = self.generate_response(question, retrieved_documents)
        return response


