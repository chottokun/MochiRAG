from abc import ABC, abstractmethod
from typing import List, Any
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
            # config_loader.py がグローバルな CONFIG_FILE_PATH を持つので、引数は基本不要
            # テスト用に config_path を渡せるようにしておくのは良いプラクティス
            # このローダーは dict を返す
            if config_path: # テスト用など、特定のコンフィグファイルを指したい場合
                from core.config_loader import load_strategy_config as load_specific_config, CONFIG_FILE_PATH as global_path
                original_path = global_path
                #一時的に差し替え
                # core.config_loader.CONFIG_FILE_PATH = Path(config_path)
                # config = load_specific_config()
                # core.config_loader.CONFIG_FILE_PATH = original_path #元に戻す
                # ちょっとトリッキーなので、素直にローダーに関数パスを渡すか、config_pathを直接使う
                # ここでは簡単化のため、config_pathは無視し、常にグローバルパスから読む
                # ただし、テスト容易性を考えると、設定dictを直接渡せるコンストラクタが良い
                pass

            config = load_strategy_config()
        except StrategyConfigError as e:
            logger.error(f"Failed to load embedding strategy configuration: {e}. No strategies will be available.", exc_info=True)
            return

        embedding_config = config.get("embedding_strategies", {})
        self.default_strategy_name = embedding_config.get("default")

        for strat_config in embedding_config.get("available", []):
            name = strat_config.get("name")
            type = strat_config.get("type")
            model_name = strat_config.get("model_name")

            if not all([name, type, model_name]):
                logger.warning(f"Skipping incomplete embedding strategy config: {strat_config}")
                continue

            strategy_instance: Optional[EmbeddingStrategy] = None
            try:
                if type == "sentence_transformer":
                    strategy_instance = SentenceTransformerEmbedding(model_name=model_name, **strat_config.get("params", {}))
                elif type == "ollama_embedding": # 設定ファイルと合わせる
                    strategy_instance = OllamaEmbeddingStrategy(
                        model_name=model_name,
                        base_url=strat_config.get("base_url"), # Optional
                        **strat_config.get("params", {})
                    )
                else:
                    logger.warning(f"Unsupported embedding strategy type: {type} for strategy '{name}'")
                    continue

                if strategy_instance:
                    # マネージャーに登録するキーは設定ファイルの `name` を使う
                    self.strategies[name] = strategy_instance
                    logger.info(f"Successfully registered embedding strategy: {name}")

            except Exception as e:
                logger.error(f"Failed to initialize or register embedding strategy '{name}': {e}", exc_info=True)

        if self.default_strategy_name and self.default_strategy_name not in self.strategies:
            logger.warning(f"Default embedding strategy '{self.default_strategy_name}' not found in available strategies. Manager might not function correctly.")
        elif not self.default_strategy_name and self.strategies:
            # デフォルトが指定されていなければ、利用可能な最初のものをデフォルトにする（フォールバック）
            self.default_strategy_name = list(self.strategies.keys())[0]
            logger.info(f"No default embedding strategy specified. Using first available: '{self.default_strategy_name}'")


    def get_strategy(self, name: Optional[str] = None) -> EmbeddingStrategy:
        target_name = name if name else self.default_strategy_name
        if not target_name:
            raise ValueError("No embedding strategy name provided and no default strategy is set.")

        strategy = self.strategies.get(target_name)
        if not strategy:
            logger.error(f"Embedding strategy '{target_name}' not found. Available: {list(self.strategies.keys())}")
            # フォールバックとして、もしデフォルト名が設定されていてそれが存在すればそれを返す
            if self.default_strategy_name and self.default_strategy_name in self.strategies:
                logger.warning(f"Falling back to default strategy: {self.default_strategy_name}")
                return self.strategies[self.default_strategy_name]
            raise ValueError(f"Embedding strategy '{target_name}' not found.")
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
