import logging
from typing import Dict, Type, Any, List

from .base import BaseRAGStrategy
from .simple_rag import SimpleRAGStrategy
from .advanced_rag import AdvancedRAGStrategy
from .deep_rag import DeepRAGStrategy

logger = logging.getLogger(__name__)

class RAGStrategyFactory:
    """
    RAG戦略のファクトリクラス。
    利用可能なRAG戦略を管理し、動的にインスタンスを生成する。
    """
    
    def __init__(self):
        self._strategies: Dict[str, Type[BaseRAGStrategy]] = {}
        self._register_default_strategies()

    def _register_default_strategies(self):
        """
        デフォルトのRAG戦略を登録する。
        """
        try:
            # SimpleRAGStrategy
            simple_strategy = SimpleRAGStrategy({})
            self.register_strategy(simple_strategy.get_name(), SimpleRAGStrategy)
            
            # AdvancedRAGStrategy
            advanced_strategy = AdvancedRAGStrategy({})
            self.register_strategy(advanced_strategy.get_name(), AdvancedRAGStrategy)
            
            # DeepRAGStrategy
            deep_strategy = DeepRAGStrategy({})
            self.register_strategy(deep_strategy.get_name(), DeepRAGStrategy)
            
            logger.info(f"Registered default RAG strategies: {list(self._strategies.keys())}")
            
        except Exception as e:
            logger.error(f"Error registering default RAG strategies: {e}")

    def register_strategy(self, name: str, strategy_cls: Type[BaseRAGStrategy]):
        """
        新しいRAG戦略を登録する。
        
        Args:
            name: 戦略の名前
            strategy_cls: 戦略クラス
        """
        if name in self._strategies:
            logger.warning(f"RAG strategy '{name}' is already registered. Overwriting.")
        
        self._strategies[name] = strategy_cls
        logger.info(f"Registered RAG strategy: {name}")

    def unregister_strategy(self, name: str):
        """
        RAG戦略の登録を解除する。
        
        Args:
            name: 戦略の名前
        """
        if name in self._strategies:
            del self._strategies[name]
            logger.info(f"Unregistered RAG strategy: {name}")
        else:
            logger.warning(f"RAG strategy '{name}' not found for unregistration")

    def get_strategy(self, name: str, config: Dict[str, Any]) -> BaseRAGStrategy:
        """
        指定された名前のRAG戦略インスタンスを取得する。
        
        Args:
            name: 戦略の名前
            config: 戦略の設定
            
        Returns:
            RAG戦略のインスタンス
            
        Raises:
            ValueError: 指定された戦略が見つからない場合
        """
        strategy_cls = self._strategies.get(name)
        if not strategy_cls:
            available_strategies = list(self._strategies.keys())
            raise ValueError(
                f"Unknown RAG strategy: '{name}'. "
                f"Available strategies are: {available_strategies}"
            )
        
        try:
            strategy_instance = strategy_cls(config)
            logger.info(f"Created RAG strategy instance: {name}")
            return strategy_instance
        except Exception as e:
            logger.error(f"Error creating RAG strategy instance '{name}': {e}")
            raise

    def get_available_strategies(self) -> List[str]:
        """
        利用可能なRAG戦略の名前一覧を取得する。
        
        Returns:
            戦略名のリスト
        """
        return list(self._strategies.keys())

    def is_strategy_available(self, name: str) -> bool:
        """
        指定された戦略が利用可能かチェックする。
        
        Args:
            name: 戦略の名前
            
        Returns:
            利用可能な場合True
        """
        return name in self._strategies

    def get_strategy_info(self) -> Dict[str, Dict[str, Any]]:
        """
        登録されている戦略の情報を取得する。
        
        Returns:
            戦略情報の辞書
        """
        info = {}
        for name, strategy_cls in self._strategies.items():
            try:
                # 一時的なインスタンスを作成して情報を取得
                temp_instance = strategy_cls({})
                info[name] = {
                    "class_name": strategy_cls.__name__,
                    "module": strategy_cls.__module__,
                    "description": strategy_cls.__doc__ or "No description available"
                }
            except Exception as e:
                info[name] = {
                    "class_name": strategy_cls.__name__,
                    "module": strategy_cls.__module__,
                    "description": f"Error getting info: {e}"
                }
        
        return info

# グローバルインスタンス
rag_strategy_factory = RAGStrategyFactory()

# 後方互換性のための定数
AVAILABLE_RAG_STRATEGIES = rag_strategy_factory.get_available_strategies()

def get_available_rag_strategies() -> List[str]:
    """
    利用可能なRAG戦略の一覧を取得する（後方互換性のための関数）。
    """
    return rag_strategy_factory.get_available_strategies()

def create_rag_strategy(strategy_name: str, config: Dict[str, Any]) -> BaseRAGStrategy:
    """
    RAG戦略のインスタンスを作成する（後方互換性のための関数）。
    """
    return rag_strategy_factory.get_strategy(strategy_name, config)

