import logging
from typing import Dict, Any, Optional, Literal

from langchain_core.language_models import BaseLanguageModel
from langchain_ollama import ChatOllama
# from langchain_openai import ChatOpenAI # OpenAIなど他のLLMプロバイダーを利用する場合

try:
    from core.config_loader import load_strategy_config, StrategyConfigError
except ImportError:
    def load_strategy_config(): # フォールバック
        return {
            "llm_config": {
                "default_provider": "ollama_chat",
                "providers": [
                    {"name": "ollama_chat", "type": "ollama", "model": "gemma3:4b-it-qat", "temperature": 0}
                ]
            }
        }
    class StrategyConfigError(Exception): pass

logger = logging.getLogger(__name__)

LLMProviderType = Literal["ollama", "openai"] # 対応するLLMプロバイダーの型

class LLMManager:
    """LLMプロバイダーとモデル設定を管理し、LLMインスタンスを提供するクラス。"""
    def __init__(self, config_path: Optional[str] = None):
        self.llm_providers: Dict[str, BaseLanguageModel] = {}
        self.default_provider_name: Optional[str] = None
        self._load_llm_config(config_path)

    def _load_llm_config(self, config_path: Optional[str] = None):
        try:
            # config_loader.py がグローバルな CONFIG_FILE_PATH を持つので、引数は基本不要
            # テスト用に config_path を渡せるようにしておくのは良いプラクティス
            config = load_strategy_config()
        except StrategyConfigError as e:
            logger.error(f"Failed to load LLM configuration: {e}. No LLMs will be available.", exc_info=True)
            return

        llm_config_section = config.get("llm_config", {})
        self.default_provider_name = llm_config_section.get("default_provider")

        for provider_conf in llm_config_section.get("providers", []):
            name = provider_conf.get("name")
            provider_type = provider_conf.get("type")
            model_name = provider_conf.get("model")
            params = provider_conf.get("params", {}) # temperatureなど共通パラメータ

            if not all([name, provider_type, model_name]):
                logger.warning(f"Skipping incomplete LLM provider config: {provider_conf}")
                continue

            llm_instance: Optional[BaseLanguageModel] = None
            try:
                if provider_type == "ollama":
                    # temperature などの共通パラメータは **params で渡す
                    # base_url も params に含めることができる
                    llm_instance = ChatOllama(model=model_name, **params)
                    logger.info(f"Initialized Ollama LLM: {name} with model {model_name}")
                # elif provider_type == "openai":
                #     # api_key_env = provider_conf.get("api_key_env")
                #     # api_key = os.getenv(api_key_env) if api_key_env else None
                #     # if not api_key:
                #     #     logger.warning(f"OpenAI API key not found for provider {name}. Skipping.")
                #     #     continue
                #     # llm_instance = ChatOpenAI(model_name=model_name, openai_api_key=api_key, **params)
                #     # logger.info(f"Initialized OpenAI LLM: {name} with model {model_name}")
                #     pass
                else:
                    logger.warning(f"Unsupported LLM provider type: {provider_type} for provider '{name}'")
                    continue

                if llm_instance:
                    self.llm_providers[name] = llm_instance
            except Exception as e:
                logger.error(f"Failed to initialize LLM provider '{name}': {e}", exc_info=True)

        if self.default_provider_name and self.default_provider_name not in self.llm_providers:
            logger.warning(f"Default LLM provider '{self.default_provider_name}' not found or failed to initialize.")
        elif not self.default_provider_name and self.llm_providers:
            self.default_provider_name = list(self.llm_providers.keys())[0]
            logger.info(f"No default LLM provider specified. Using first available: '{self.default_provider_name}'")


    def get_llm(self, name: Optional[str] = None) -> BaseLanguageModel:
        target_name = name if name else self.default_provider_name
        if not target_name:
            raise ValueError("No LLM provider name specified and no default provider is set.")

        llm = self.llm_providers.get(target_name)
        if not llm:
            logger.error(f"LLM provider '{target_name}' not found. Available: {list(self.llm_providers.keys())}")
            if self.default_provider_name and self.default_provider_name in self.llm_providers:
                logger.warning(f"Falling back to default LLM provider: {self.default_provider_name}")
                return self.llm_providers[self.default_provider_name]
            raise ValueError(f"LLM provider '{target_name}' not found.")
        return llm

    def get_available_providers(self) -> List[str]:
        return list(self.llm_providers.keys())

# グローバルなLLMManagerインスタンス
llm_manager = LLMManager()

# core.rag_chain.py で使用していたグローバルな `llm` 変数を置き換える
# ただし、直接置き換えるのではなく、rag_chain.py 側で llm_manager.get_llm() を呼ぶようにする
# ここでは、下位互換性や他のモジュールでの直接参照のために一時的に設定するかもしれないが、
# 基本的には llm_manager を介したアクセスを推奨
llm = llm_manager.get_llm() # デフォルトプロバイダーのLLMを取得

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("Available LLM providers from manager:")
    for name in llm_manager.get_available_providers():
        print(f"- {name}")

    default_llm_name = llm_manager.default_provider_name
    if default_llm_name:
        print(f"\nTesting default LLM provider: {default_llm_name}")
        try:
            default_llm_instance = llm_manager.get_llm() # デフォルトを取得
            print(f"Successfully got default LLM: {type(default_llm_instance)}")
            if hasattr(default_llm_instance, 'invoke'):
                response = default_llm_instance.invoke("Hello, LLM!")
                print(f"Test response from default LLM: {response}")
            else:
                print("Default LLM instance does not have invoke method for simple test.")
        except Exception as e:
            print(f"Error getting or invoking default LLM provider '{default_llm_name}': {e}", exc_info=True)
    else:
        print("\nNo default LLM provider set or available.")

    # Ollamaプロバイダーが設定ファイルにあればテスト
    ollama_provider_name_in_config = "ollama_chat" # config/strategies.yaml で定義した名前
    if ollama_provider_name_in_config in llm_manager.get_available_providers():
        print(f"\nTesting Ollama provider from config: {ollama_provider_name_in_config}")
        try:
            ollama_llm = llm_manager.get_llm(ollama_provider_name_in_config)
            print(f"Successfully got Ollama LLM: {type(ollama_llm)}")
            response = ollama_llm.invoke("今日の天気は？") # 日本語でテスト
            print(f"Test response from Ollama LLM: {response}")
        except Exception as e:
            print(f"Error getting or invoking Ollama LLM '{ollama_provider_name_in_config}'. Ensure Ollama server is running: {e}", exc_info=True)
    else:
        print(f"\nOllama provider '{ollama_provider_name_in_config}' not found in config or not loaded.")

    try:
        print(f"\nAttempting to get a non-existent LLM provider:")
        llm_manager.get_llm("non_existent_llm_provider_123")
    except ValueError as e:
        print(f"Correctly caught error for non-existent LLM provider: {e}")
