from langchain_ollama import ChatOllama
from typing import Dict, Optional

from .config_manager import config_manager

class LLMManager:
    _instance = None
    _llms: Dict[str, ChatOllama] = {}

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(LLMManager, cls).__new__(cls)
        return cls._instance

    def get_llm(self, name: Optional[str] = None) -> ChatOllama:
        """Instantiates an LLM client, caching it for reuse."""
        if name is None:
            name = config_manager.get_default_llm_name()
        if name not in self._llms:
            print(f"Instantiating LLM: {name}")
            config = config_manager.get_llm_config(name)
            
            if config.provider == 'ollama':
                self._llms[name] = ChatOllama(
                    model=config.model_name,
                    base_url=config.base_url
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {config.provider}")
            print("LLM instantiated.")

        return self._llms[name]

# Create a single, globally accessible instance
llm_manager = LLMManager()