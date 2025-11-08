import yaml
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from pathlib import Path
import os
import re
from dotenv import load_dotenv

# Load .env file from the project root
load_dotenv()

# --- Pydantic Models for Configuration Validation ---

class EmbeddingConfig(BaseModel):
    provider: str
    model_name: str
    base_url: str | None = None

class LLMProviderConfig(BaseModel):
    provider: str
    model_name: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    deployment_name: Optional[str] = None
    azure_endpoint: Optional[str] = None
    api_version: Optional[str] = None
    temperature: Optional[float] = 0.7

class LLMStructureConfig(BaseModel):
    roles: Dict[str, str]
    providers: Dict[str, LLMProviderConfig]

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
    llms: LLMStructureConfig
    retrievers: Dict[str, RetrieverStrategyConfig]
    prompts: Optional[Dict[str, str]] = Field(default_factory=dict)

# --- ConfigManager Singleton ---

class ConfigManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_path: str = None):
        if hasattr(self, 'is_initialized'):
            return

        if config_path is None:
            base_path = Path(os.path.dirname(os.path.abspath(__file__))).parent
            config_path = base_path / "config" / "strategies.yaml"

        with open(config_path, 'r') as f:
            raw_yaml_content = f.read()

        # Substitute environment variables of the form ${VAR_NAME}
        env_var_pattern = re.compile(r'\$\{(.*?)\}')
        def env_var_replacer(match):
            var_name = match.group(1)
            return os.getenv(var_name, '')
        
        substituted_yaml_content = env_var_pattern.sub(env_var_replacer, raw_yaml_content)
        yaml_data = yaml.safe_load(substituted_yaml_content)

        # Allow environment variables to override vector_store host/port
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
                except (ValueError, TypeError):
                    pass

        # Allow overriding Ollama base URL from environment
        ollama_base = os.getenv('OLLAMA_BASE_URL')
        if ollama_base:
            if 'embeddings' in yaml_data and yaml_data['embeddings']:
                for k, v in yaml_data['embeddings'].items():
                    if isinstance(v, dict) and v.get('provider') == 'ollama':
                        v['base_url'] = ollama_base
            if 'llms' in yaml_data and 'providers' in yaml_data['llms']:
                for k, v in yaml_data['llms']['providers'].items():
                    if isinstance(v, dict) and v.get('provider') == 'ollama':
                        v['base_url'] = ollama_base

        # Allow environment variables to override the default LLM provider and model for the "main" role
        llm_provider = os.getenv('LLM_PROVIDER')
        llm_model_name = os.getenv('LLM_MODEL_NAME')
        if llm_provider and llm_model_name:
            env_provider_key = "env_default_llm"
            if 'llms' not in yaml_data:
                yaml_data['llms'] = {'roles': {}, 'providers': {}}
            if 'providers' not in yaml_data['llms']:
                yaml_data['llms']['providers'] = {}
            if 'roles' not in yaml_data['llms']:
                yaml_data['llms']['roles'] = {}

            provider_config = {
                'provider': llm_provider.lower(),
                'model_name': llm_model_name
            }

            provider = llm_provider.lower()
            if provider == 'openai':
                provider_config['api_key'] = os.getenv('OPENAI_API_KEY', '')
            elif provider == 'azure':
                provider_config['api_key'] = os.getenv('AZURE_OPENAI_API_KEY', '')
                provider_config['azure_endpoint'] = os.getenv('AZURE_OPENAI_ENDPOINT', '')
            elif provider == 'gemini':
                provider_config['api_key'] = os.getenv('GOOGLE_API_KEY', '')
            elif provider == 'ollama':
                provider_config['base_url'] = os.getenv('OLLAMA_BASE_URL', '')

            yaml_data['llms']['providers'][env_provider_key] = provider_config
            yaml_data['llms']['roles']['main'] = env_provider_key

        self.config = AppConfig(**yaml_data)
        self.is_initialized = True

    def get_default_llm_role(self) -> str:
        if not self.config.llms.roles:
            raise ValueError("No LLM roles found in strategies.yaml")
        return "main" # Default to the 'main' role

    def get_llm_config_by_role(self, role: str) -> LLMProviderConfig:
        if role not in self.config.llms.roles:
            raise ValueError(f"LLM role '{role}' not found in strategies.yaml")
        
        provider_name = self.config.llms.roles[role]
        
        if provider_name not in self.config.llms.providers:
            raise ValueError(f"LLM provider configuration '{provider_name}' for role '{role}' not found in strategies.yaml")
            
        return self.config.llms.providers[provider_name]

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

    def get_prompt(self, name: str, default: str = "") -> str:
        """
        Retrieves a prompt template from the configuration.

        Args:
            name (str): The name of the prompt to retrieve.
            default (str): A fallback default prompt template if the requested
                           one is not found in the config file.

        Returns:
            str: The prompt template.
        """
        return self.config.prompts.get(name, default)

    def get_chunk_size(self) -> Optional[int]:
        """Reads CHUNK_SIZE from environment variables."""
        size = os.getenv("CHUNK_SIZE")
        if size:
            try:
                return int(size)
            except (ValueError, TypeError):
                return None
        return None

    def get_chunk_overlap(self) -> Optional[int]:
        """Reads CHUNK_OVERLAP from environment variables."""
        overlap = os.getenv("CHUNK_OVERLAP")
        if overlap:
            try:
                return int(overlap)
            except (ValueError, TypeError):
                return None
        return None

# Create a single, globally accessible instance of the ConfigManager
config_manager = ConfigManager()
