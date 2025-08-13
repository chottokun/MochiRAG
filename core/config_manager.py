import yaml
from pydantic import BaseModel, Field
from typing import Dict, Any

# --- Pydantic Models for Configuration Validation ---

class EmbeddingConfig(BaseModel):
    provider: str
    model_name: str

class LLMConfig(BaseModel):
    provider: str
    model_name: str
    base_url: str

class RetrieverStrategyConfig(BaseModel):
    strategy_class: str
    description: str
    parameters: Dict[str, Any]

class AppConfig(BaseModel):
    embeddings: Dict[str, EmbeddingConfig]
    llms: Dict[str, LLMConfig]
    retrievers: Dict[str, RetrieverStrategyConfig]

# --- ConfigManager Singleton ---

class ConfigManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_path: str = "config/strategies.yaml"):
        # The check prevents re-initialization on subsequent calls
        if not hasattr(self, 'is_initialized'):
            with open(config_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
            self.config = AppConfig(**yaml_data)
            self.is_initialized = True
    def get_default_llm_name(self) -> str:
        if not self.config.llms:
            raise ValueError("No LLM configurations found in strategies.yaml")
        return next(iter(self.config.llms))


    def get_llm_config(self, name: str) -> LLMConfig:
        if name not in self.config.llms:
            raise ValueError(f"LLM configuration '{name}' not found in strategies.yaml")
        return self.config.llms[name]

    def get_embedding_config(self, name: str) -> EmbeddingConfig:
        if name not in self.config.embeddings:
            raise ValueError(f"Embedding configuration '{name}' not found in strategies.yaml")
        return self.config.embeddings[name]

    def get_retriever_config(self, name: str) -> RetrieverStrategyConfig:
        if name not in self.config.retrievers:
            raise ValueError(f"Retriever configuration '{name}' not found in strategies.yaml")
        return self.config.retrievers[name]

# Create a single, globally accessible instance of the ConfigManager
config_manager = ConfigManager()