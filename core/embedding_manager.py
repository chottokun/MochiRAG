from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from typing import Dict, Union

from .config_manager import config_manager

class EmbeddingManager:
    _instance = None
    _embedding_models: Dict[str, Union[HuggingFaceEmbeddings, OllamaEmbeddings]] = {}

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(EmbeddingManager, cls).__new__(cls)
        return cls._instance

    def get_embedding_model(self, name: str = "all-MiniLM-L6-v2") -> Union[HuggingFaceEmbeddings, OllamaEmbeddings]:
        """Loads a HuggingFace or Ollama embedding model, caching it for reuse."""
        if name not in self._embedding_models:
            print(f"Loading embedding model: {name}")
            config = config_manager.get_embedding_config(name)
            
            if config.provider == 'huggingface':
                self._embedding_models[name] = HuggingFaceEmbeddings(
                    model_name=config.model_name
                )
            elif config.provider == 'ollama':
                self._embedding_models[name] = OllamaEmbeddings(
                    model=config.model_name,
                    base_url=config.base_url
                )
            else:
                raise ValueError(f"Unsupported embedding provider: {config.provider}")
            print("Embedding model loaded.")
        
        return self._embedding_models[name]

# Create a single, globally accessible instance
embedding_manager = EmbeddingManager()