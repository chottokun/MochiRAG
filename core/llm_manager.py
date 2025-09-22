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

    def get_llm(self, role: Optional[str] = None) -> BaseChatModel:
        """
        Retrieves a cached LLM client or instantiates a new one if not available.
        The configuration for the LLM is fetched based on its role.
        """
        if role is None:
            role = config_manager.get_default_llm_role()

        config = config_manager.get_llm_config_by_role(role)
        
        # Use the specific model_name from the provider config for caching
        cache_key = config.model_name if config.model_name else role

        if cache_key not in self._llms:
            print(f"LLM for role '{role}' (model: '{cache_key}') not found in cache. Instantiating...")
            provider = getattr(config, 'provider', '').lower()

            if provider == 'ollama':
                self._llms[cache_key] = ChatOllama(
                    model=config.model_name,
                    base_url=config.base_url
                )
            elif provider == 'openai':
                self._llms[cache_key] = ChatOpenAI(
                    model=config.model_name,
                    api_key=config.api_key,
                    temperature=getattr(config, 'temperature', 0.7)
                )
            elif provider == 'azure':
                self._llms[cache_key] = AzureChatOpenAI(
                    azure_deployment=config.deployment_name,
                    azure_endpoint=config.azure_endpoint,
                    api_version=config.api_version,
                    api_key=config.api_key,
                    temperature=getattr(config, 'temperature', 0.7)
                )
            elif provider == 'gemini':
                self._llms[cache_key] = ChatGoogleGenerativeAI(
                    model=config.model_name,
                    google_api_key=config.api_key,
                    temperature=getattr(config, 'temperature', 0.7)
                )
            else:
                raise ValueError(f"Unsupported LLM provider: '{config.provider}'")

            print(f"LLM '{cache_key}' ({provider}) instantiated and cached.")

        return self._llms[cache_key]

# Create a single, globally accessible instance
llm_manager = LLMManager()