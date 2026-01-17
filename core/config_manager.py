import yaml
import os
import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# --- Pydantic Models for Configuration Validation ---

class EmbeddingConfig(BaseModel):
    provider: str
    model_name: str
    base_url: Optional[str] = None

class LLMProviderConfig(BaseModel):
    provider: str
    model_name: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    deployment_name: Optional[str] = None
    azure_endpoint: Optional[str] = None
    api_version: Optional[str] = None
    temperature: float = 0.7

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
    parameters: Dict[str, Any] = Field(default_factory=dict)

class AppConfig(BaseModel):
    vector_store: VectorStoreConfig
    embeddings: Dict[str, EmbeddingConfig]
    llms: LLMStructureConfig
    retrievers: Dict[str, RetrieverStrategyConfig]
    prompts: Dict[str, str] = Field(default_factory=dict)

# --- Settings class for Environment Variables ---

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    secret_key: str = "default_secret_key"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    chroma_host: Optional[str] = None
    chroma_port: Optional[int] = None
    ollama_base_url: Optional[str] = None

    # Provider keys
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    azure_openai_api_key: Optional[str] = None

settings = Settings()

# --- ConfigManager Singleton ---

class ConfigManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_path: Optional[str] = None):
        if hasattr(self, 'is_initialized'):
            return

        if config_path is None:
            base_path = Path(os.path.dirname(os.path.abspath(__file__))).parent
            config_path = str(base_path / "config" / "strategies.yaml")

        self._load_config(config_path)
        self.is_initialized = True

    def _load_config(self, config_path: str):
        try:
            with open(config_path, 'r') as f:
                raw_yaml_content = f.read()
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            raise

        # Substitute environment variables of the form ${VAR_NAME}
        env_var_pattern = re.compile(r'\$\{(.*?)\}')
        def env_var_replacer(match):
            var_name = match.group(1)
            return os.getenv(var_name, '')
        
        substituted_yaml_content = env_var_pattern.sub(env_var_replacer, raw_yaml_content)
        yaml_data = yaml.safe_load(substituted_yaml_content)

        # Apply Overrides from Settings
        self._apply_overrides(yaml_data)

        self.config = AppConfig(**yaml_data)
        logger.info(f"Configuration loaded from {config_path}")

    def _apply_overrides(self, yaml_data: Dict[str, Any]):
        if settings.chroma_host:
            yaml_data.setdefault('vector_store', {})['host'] = settings.chroma_host
        if settings.chroma_port:
            yaml_data.setdefault('vector_store', {})['port'] = settings.chroma_port

        if settings.ollama_base_url:
            # Override Ollama base URL in embeddings
            for emb in yaml_data.get('embeddings', {}).values():
                if isinstance(emb, dict) and emb.get('provider') == 'ollama':
                    emb['base_url'] = settings.ollama_base_url

            # Override Ollama base URL in LLMs
            for prov in yaml_data.get('llms', {}).get('providers', {}).values():
                if isinstance(prov, dict) and prov.get('provider') == 'ollama':
                    prov['base_url'] = settings.ollama_base_url

        # Apply API keys from settings if missing in YAML
        for prov_name, prov in yaml_data.get('llms', {}).get('providers', {}).items():
            if not isinstance(prov, dict): continue

            if prov.get('provider') == 'openai' and not prov.get('api_key'):
                prov['api_key'] = settings.openai_api_key
            elif prov.get('provider') == 'gemini' and not prov.get('api_key'):
                prov['api_key'] = settings.google_api_key
            elif prov.get('provider') == 'azure' and not prov.get('api_key'):
                prov['api_key'] = settings.azure_openai_api_key

    def get_default_llm_role(self) -> str:
        return "main"

    def get_llm_config_by_role(self, role: str) -> LLMProviderConfig:
        if role not in self.config.llms.roles:
            raise ValueError(f"LLM role '{role}' not found in configuration")
        
        provider_name = self.config.llms.roles[role]
        if provider_name not in self.config.llms.providers:
            raise ValueError(f"LLM provider '{provider_name}' for role '{role}' not found")
            
        return self.config.llms.providers[provider_name]

    def get_embedding_config(self, name: str) -> EmbeddingConfig:
        if name not in self.config.embeddings:
            raise ValueError(f"Embedding configuration '{name}' not found")
        return self.config.embeddings[name]

    def get_retriever_config(self, name: str) -> RetrieverStrategyConfig:
        if name not in self.config.retrievers:
            raise ValueError(f"Retriever configuration '{name}' not found")
        return self.config.retrievers[name]

    def get_vector_store_config(self) -> VectorStoreConfig:
        return self.config.vector_store

    def get_prompt(self, name: str, default: str = "") -> str:
        return self.config.prompts.get(name, default)

config_manager = ConfigManager()
