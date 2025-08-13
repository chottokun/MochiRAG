from typing import Dict, Any
from langchain_core.language_models import BaseLanguageModel
from langchain_ollama.llms import OllamaLLM

from core.config_manager import config_manager

class LLMManager:
    """
    Manages the lifecycle of Language Model (LLM) instances.

    This class is responsible for instantiating LLM clients based on configuration
    and caching them for reuse.
    """
    _instances: Dict[str, BaseLanguageModel] = {}

    def get_llm(self, name: str) -> BaseLanguageModel:
        """
        Retrieves an LLM instance by its configuration name.

        If the instance has already been created, it returns the cached instance.
        Otherwise, it creates a new one based on the configuration.

        Args:
            name: The name of the LLM configuration to use.

        Returns:
            An instance of a class that inherits from BaseLanguageModel.

        Raises:
            ValueError: If the specified LLM provider is not supported.
        """
        if name in self._instances:
            return self._instances[name]

        llm_config = config_manager.get_llm_config(name)
        provider = llm_config.get("provider")

        llm_instance = None
        if provider == "ollama":
            llm_instance = OllamaLLM(
                model=llm_config.get("model"),
                base_url=llm_config.get("base_url"),
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        self._instances[name] = llm_instance
        return llm_instance

# Default instance for singleton-like access
llm_manager = LLMManager()
