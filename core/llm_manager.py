from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from typing import Dict, Optional

from .config_manager import config_manager


class LLMManager:
    """
    Manages the lifecycle of LLM clients.

    This is a singleton class to ensure that LLM instances are created only once
    and reused across the application.
    """
    _instance = None
    _llms: Dict[str, BaseChatModel] = {}

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(LLMManager, cls).__new__(cls)
        return cls._instance

    def get_llm(self, name: Optional[str] = None) -> BaseChatModel:
        """
        Retrieves a cached LLM client or instantiates a new one if not available.

        The configuration for the LLM is fetched from the ConfigManager.
        """
        if name is None:
            name = config_manager.get_default_llm_name()

        if name not in self._llms:
            print(f"LLM '{name}' not found in cache. Instantiating...")
            config = config_manager.get_llm_config(name)
            provider = getattr(config, 'provider', '').lower()

            if provider == 'ollama':
                self._llms[name] = ChatOllama(
                    model=config.model_name,
                    base_url=config.base_url
                )
            elif provider == 'openai':
                self._llms[name] = ChatOpenAI(
                    model=config.model_name,
                    api_key=config.api_key,
                    temperature=getattr(config, 'temperature', 0.7)
                )
            elif provider == 'azure':
                self._llms[name] = AzureChatOpenAI(
                    azure_deployment=config.deployment_name,
                    azure_endpoint=config.azure_endpoint,
                    api_version=config.api_version,
                    api_key=config.api_key,
                    temperature=getattr(config, 'temperature', 0.7)
                )
            elif provider == 'gemini':
                self._llms[name] = ChatGoogleGenerativeAI(
                    model=config.model_name,
                    google_api_key=config.api_key,
                    temperature=getattr(config, 'temperature', 0.7)
                )
            else:
                raise ValueError(f"Unsupported LLM provider: '{config.provider}'")

            print(f"LLM '{name}' ({provider}) instantiated and cached.")

        return self._llms[name]

# Create a single, globally accessible instance
llm_manager = LLMManager()