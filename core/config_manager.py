import yaml
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from pathlib import Path
import os

# --- Pydantic Models for Configuration Validation ---

class EmbeddingConfig(BaseModel):
    provider: str
    model_name: str
    base_url: str | None = None

class LLMConfig(BaseModel):
    provider: str
    model_name: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    deployment_name: Optional[str] = None
    azure_endpoint: Optional[str] = None
    api_version: Optional[str] = None
    temperature: Optional[float] = 0.7

class VectorStoreConfig(BaseModel):
    provider: str
    mode: str
    host: Optional[str] = None
    port: Optional[int] = None
    path: Optional[str] = None

class RetrieverStrategyConfig(BaseModel):
    strategy_class: str
    description: str
    parameters: Dict[str, Any]

class AppConfig(BaseModel):
    vector_store: VectorStoreConfig
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

    def __init__(self, config_path: str = None):
        # The check prevents re-initialization on subsequent calls
        if not hasattr(self, 'is_initialized'):
            if config_path is None:
                # Build a path relative to this file to find the config
                base_path = Path(os.path.dirname(os.path.abspath(__file__))).parent
                config_path = base_path / "config" / "strategies.yaml"
            with open(config_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
            # Allow environment variables to override vector_store host/port
            # This makes it easy to run inside Docker where the Chroma service
            # is available by service name (e.g. 'chroma') and uses container port 8000.
            chroma_host = os.getenv('CHROMA_HOST')
            chroma_port = os.getenv('CHROMA_PORT')
            if chroma_host or chroma_port:
                if 'vector_store' not in yaml_data or yaml_data['vector_store'] is None:
                    yaml_data['vector_store'] = {}
                if chroma_host:
                    yaml_data['vector_store']['host'] = chroma_host
                if chroma_port:
                    try:
                        yaml_data['vector_store']['port'] = int(chroma_port)
                    except ValueError:
                        # If port is not an int, keep the original YAML value (will raise later if invalid)
                        pass

            # Allow overriding Ollama base URL from environment (useful when Ollama
            # is running on the Docker host and reachable via the host gateway).
            ollama_base = os.getenv('OLLAMA_BASE_URL')
            if ollama_base:
                # Update any embeddings that use the 'ollama' provider
                if 'embeddings' in yaml_data and yaml_data['embeddings']:
                    for k, v in yaml_data['embeddings'].items():
                        if isinstance(v, dict) and v.get('provider') == 'ollama':
                            v['base_url'] = ollama_base
                # Update any llms that use the 'ollama' provider
                if 'llms' in yaml_data and yaml_data['llms']:
                    for k, v in yaml_data['llms'].items():
                        if isinstance(v, dict) and v.get('provider') == 'ollama':
                            v['base_url'] = ollama_base

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

    def get_vector_store_config(self) -> VectorStoreConfig:
        if not self.config.vector_store:
            raise ValueError("Vector store configuration not found in strategies.yaml")
        return self.config.vector_store

# Create a single, globally accessible instance of the ConfigManager
config_manager = ConfigManager()