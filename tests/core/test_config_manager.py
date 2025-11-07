import os
import pytest
from unittest.mock import patch
from core.config_manager import ConfigManager

# Force a reload of the config manager for testing purposes
def reload_config_manager():
    # A trick to force re-initialization of the singleton
    if hasattr(ConfigManager._instance, 'is_initialized'):
        del ConfigManager._instance.is_initialized
    return ConfigManager()

@pytest.fixture(autouse=True)
def clear_env_vars():
    """Fixture to clear relevant env vars before and after each test."""
    vars_to_clear = [
        "CHUNK_SIZE", "CHUNK_OVERLAP", "LLM_PROVIDER", "LLM_MODEL_NAME",
        "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "GOOGLE_API_KEY"
    ]
    original_values = {var: os.environ.get(var) for var in vars_to_clear}

    # Clear vars before the test
    for var in vars_to_clear:
        if var in os.environ:
            del os.environ[var]

    yield

    # Restore original values after the test
    for var, value in original_values.items():
        if value is not None:
            os.environ[var] = value
        elif var in os.environ:
            del os.environ[var]

def test_chunk_size_override():
    """
    Tests if CHUNK_SIZE and CHUNK_OVERLAP env vars override the default config.
    """
    test_chunk_size = "1234"
    test_chunk_overlap = "567"

    with patch.dict(os.environ, {"CHUNK_SIZE": test_chunk_size, "CHUNK_OVERLAP": test_chunk_overlap}):
        config_manager = reload_config_manager()
        retriever_config = config_manager.get_retriever_config('basic')

        assert retriever_config.parameters['chunk_size'] == int(test_chunk_size)
        assert retriever_config.parameters['chunk_overlap'] == int(test_chunk_overlap)

def test_llm_provider_override_openai():
    """
    Tests if LLM_PROVIDER and LLM_MODEL_NAME env vars can configure OpenAI.
    """
    test_provider = "openai"
    test_model = "gpt-4-test"
    test_api_key = "sk-testkey123"

    with patch.dict(os.environ, {
        "LLM_PROVIDER": test_provider,
        "LLM_MODEL_NAME": test_model,
        "OPENAI_API_KEY": test_api_key
    }):
        config_manager = reload_config_manager()
        llm_config = config_manager.get_llm_config_by_role('main')

        assert llm_config.provider == test_provider
        assert llm_config.model_name == test_model
        assert llm_config.api_key == test_api_key

def test_llm_provider_override_ollama():
    """
    Tests if LLM_PROVIDER and LLM_MODEL_NAME env vars can configure Ollama.
    """
    test_provider = "ollama"
    test_model = "llama3-test"
    test_base_url = "http://localhost:ollama"

    with patch.dict(os.environ, {
        "LLM_PROVIDER": test_provider,
        "LLM_MODEL_NAME": test_model,
        "OLLAMA_BASE_URL": test_base_url
    }):
        config_manager = reload_config_manager()
        llm_config = config_manager.get_llm_config_by_role('main')

        assert llm_config.provider == test_provider
        assert llm_config.model_name == test_model
        assert llm_config.base_url == test_base_url
