from abc import ABC, abstractmethod
from typing import List, Any, Optional # Optional を追加
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings # OllamaEmbeddingsを正式にインポート

import logging # ロギング用

try:
    from core.config_loader import load_strategy_config, StrategyConfigError
except ImportError:
    # logger.warning("Could not import config_loader in embedding_manager. Using fallback defaults.")
    def load_strategy_config(): # フォールバック用のダミー関数
        return {
            "embedding_strategies": {
                "default": "sentence_transformer_all-MiniLM-L6-v2",
                "available": [
                    {"name": "sentence_transformer_all-MiniLM-L6-v2", "type": "sentence_transformer", "model_name": "all-MiniLM-L6-v2"}
                ]
            }
        }
    class StrategyConfigError(Exception): pass


logger = logging.getLogger(__name__)


class EmbeddingStrategy(ABC):
    """エンベディング戦略のインターフェース"""
    @abstractmethod
    def get_embedding_model(self) -> Embeddings:
        """LangChainのEmbeddings互換モデルを返す"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """戦略名を返す"""
        pass

class SentenceTransformerEmbedding(EmbeddingStrategy):
    """HuggingFace Sentence Transformersを利用したエンベディング戦略"""
    def __init__(self, model_name: str, **kwargs: Any): # configから渡されるmodel_nameを期待
        self.model_name = model_name
        self.config_kwargs = kwargs
        try:
            self._model = HuggingFaceEmbeddings(
                model_name=self.model_name,
                **self.config_kwargs # cache_folderなど他のパラメータを渡せるように
            )
            logger.info(f"SentenceTransformerEmbedding loaded for model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load HuggingFaceEmbeddings model {self.model_name}: {e}", exc_info=True)
            raise ValueError(f"Failed to load HuggingFaceEmbeddings model {self.model_name}: {e}") from e

    def get_embedding_model(self) -> Embeddings:
        return self._model

    def get_name(self) -> str:
        # 設定ファイルで指定されたnameを返すか、動的に生成するか。ここでは設定ファイル依存とする。
        # ただし、Manager側で登録時にconfigのnameを使うので、ここではmodel_nameベースでも良い。
        return f"sentence_transformer_{self.model_name.replace('/', '_')}"

class OllamaEmbeddingStrategy(EmbeddingStrategy):
    """Ollama経由でエンベディングモデルを利用する戦略"""
    def __init__(self, model_name: str, base_url: Optional[str] = None, **kwargs: Any):
        self.model_name = model_name
        self.base_url = base_url
        self.config_kwargs = kwargs
        try:
            params = {"model": self.model_name, **self.config_kwargs}
            if self.base_url:
                params["base_url"] = self.base_url
            self._model = OllamaEmbeddings(**params)
            logger.info(f"OllamaEmbeddingStrategy loaded for model: {self.model_name} (URL: {self.base_url or 'default'})")
        except Exception as e:
            logger.error(f"Failed to initialize OllamaEmbeddings model {self.model_name}: {e}", exc_info=True)
            raise ValueError(f"Failed to initialize OllamaEmbeddings model {self.model_name}: {e}") from e

    def get_embedding_model(self) -> Embeddings:
        return self._model

    def get_name(self) -> str:
        return f"ollama_embedding_{self.model_name}"


class EmbeddingManager:
    """エンベディング戦略を管理するクラス。設定ファイルから動的に戦略をロードする。"""
    def __init__(self, config_path: Optional[str] = None): # config_path はテスト用に残す
        self.strategies: Dict[str, EmbeddingStrategy] = {}
        self.default_strategy_name: Optional[str] = None
        self._load_strategies_from_config(config_path)

    def _load_strategies_from_config(self, config_path: Optional[str] = None):
        try:
            config = load_strategy_config()
        except StrategyConfigError as e:
            logger.error(f"EmbeddingManager: Failed to load strategy configuration: {e}. No strategies will be available.", exc_info=True)
            return # 設定が読めなければ戦略は空のまま

        embedding_config = config.get("embedding_strategies")
        if not isinstance(embedding_config, dict):
            logger.warning("EmbeddingManager: 'embedding_strategies' section not found or invalid in config. No strategies loaded.")
            return

        self.default_strategy_name = embedding_config.get("default")
        available_configs = embedding_config.get("available")

        if not isinstance(available_configs, list):
            logger.warning("EmbeddingManager: 'embedding_strategies.available' section not found or not a list. No strategies loaded.")
            return

        for strat_config in available_configs:
            if not isinstance(strat_config, dict):
                logger.warning(f"EmbeddingManager: Skipping invalid strategy config item (not a dict): {strat_config}")
                continue

            name = strat_config.get("name")
            strat_type = strat_config.get("type") # 'type' を 'strat_type' に変更してPythonの予約語と衝突回避
            model_name = strat_config.get("model_name")

            if not all([name, strat_type, model_name]):
                logger.warning(f"EmbeddingManager: Skipping incomplete embedding strategy config: {strat_config} (missing name, type, or model_name).")
                continue

            strategy_instance: Optional[EmbeddingStrategy] = None
            try:
                params = strat_config.get("params", {})
                if strat_type == "sentence_transformer":
                    strategy_instance = SentenceTransformerEmbedding(model_name=model_name, **params)
                elif strat_type == "ollama_embedding":
                    strategy_instance = OllamaEmbeddingStrategy(
                        model_name=model_name,
                        base_url=strat_config.get("base_url"), # configファイルから直接取得
                        **params
                    )
                else:
                    logger.warning(f"EmbeddingManager: Unsupported embedding strategy type: {strat_type} for strategy '{name}'")
                    continue

                if strategy_instance:
                    self.strategies[name] = strategy_instance
                    logger.info(f"EmbeddingManager: Successfully registered embedding strategy: {name}")

            except Exception as e:
                logger.error(f"EmbeddingManager: Failed to initialize or register embedding strategy '{name}': {e}", exc_info=True)

        if self.default_strategy_name and self.default_strategy_name not in self.strategies:
            logger.warning(f"EmbeddingManager: Default embedding strategy '{self.default_strategy_name}' not found in available strategies. Default will be unset.")
            self.default_strategy_name = None # 無効なデフォルトはクリア

        if not self.default_strategy_name and self.strategies:
            # 利用可能な戦略があれば、最初のものをデフォルトにする
            self.default_strategy_name = list(self.strategies.keys())[0]
            logger.info(f"EmbeddingManager: No valid default embedding strategy specified. Using first available: '{self.default_strategy_name}'")
        elif not self.strategies:
            logger.warning("EmbeddingManager: No embedding strategies were loaded or registered.")


    def get_strategy(self, name: Optional[str] = None) -> EmbeddingStrategy:
        target_name = name if name else self.default_strategy_name
        if not target_name:
            # このケースは、設定ファイルが全くないか、embedding_strategiesセクションがない場合に発生しうる
            raise StrategyConfigError("EmbeddingManager: No embedding strategy name provided and no default strategy is configured or available.")

        strategy = self.strategies.get(target_name)
        if not strategy:
            # default_strategy_name が設定されていても、その戦略がロードに失敗している場合など
            available_strats = list(self.strategies.keys())
            err_msg = f"EmbeddingManager: Embedding strategy '{target_name}' not found. Available strategies: {available_strats if available_strats else 'None'}."
            logger.error(err_msg)
            raise ValueError(err_msg)
        return strategy

    def get_available_strategies(self) -> List[str]:
        return list(self.strategies.keys())

    def get_embedding_model(self, name: Optional[str] = None) -> Embeddings:
        strategy = self.get_strategy(name)
        return strategy.get_embedding_model()

# グローバルなEmbeddingManagerインスタンス
embedding_manager = EmbeddingManager()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO) # __main__ で実行時のロギング設定
    print("Available embedding strategies from manager:")
    # マネージャーが設定ファイルからロードした戦略を表示
    for name in embedding_manager.get_available_strategies():
        print(f"- {name}")

    default_model_name = embedding_manager.default_strategy_name
    if default_model_name:
        print(f"\nTesting default strategy: {default_model_name}")
        try:
            model = embedding_manager.get_embedding_model() # デフォルトを取得
            print(f"Successfully got default model: {type(model)}")
            embeddings = model.embed_query("This is a test sentence for the default model.")
            print(f"Test embedding (first 5 dimensions): {embeddings[:5]}")
        except Exception as e:
            print(f"Error getting default embedding model '{default_model_name}': {e}", exc_info=True)
    else:
        print("\nNo default embedding strategy set or available.")

    # 設定ファイルに Ollama の設定があればテスト (Ollamaが起動している必要がある)
    ollama_strategy_name_in_config = "ollama_embedding_nomic-embed-text" # config/strategies.yaml で定義した名前に合わせる
    if ollama_strategy_name_in_config in embedding_manager.get_available_strategies():
        print(f"\nTesting Ollama strategy from config: {ollama_strategy_name_in_config}")
        try:
            ollama_model = embedding_manager.get_embedding_model(ollama_strategy_name_in_config)
            print(f"Successfully got Ollama model: {type(ollama_model)}")
            ollama_embeddings = ollama_model.embed_query("これはOllamaのテスト文です。")
            print(f"Ollama test embedding (first 5 dimensions): {ollama_embeddings[:5]}")
        except Exception as e:
            print(f"Error getting Ollama model '{ollama_strategy_name_in_config}'. Ensure Ollama is running and the model is pulled: {e}", exc_info=True)
    else:
        print(f"\nOllama strategy '{ollama_strategy_name_in_config}' not found in config or not loaded.")

    try:
        print(f"\nAttempting to get a non-existent strategy:")
        embedding_manager.get_embedding_model("non_existent_strategy_12345")
    except ValueError as e:
        print(f"Correctly caught error for non-existent strategy: {e}")
