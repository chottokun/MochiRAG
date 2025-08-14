import pytest
from unittest.mock import patch, MagicMock

# --- Fixtures ---
@pytest.fixture(autouse=True)
def clear_llm_manager_cache():
    """A fixture to automatically clear the LLMManager cache before each test."""
    # We must import here to avoid circular dependency issues at module load time
    from core.llm_manager import LLMManager
    LLMManager._llms.clear()


# --- Mock Configuration ---
MOCK_OLLAMA_CONFIG = {
    "provider": "ollama",
    "model": "test-model",
    "base_url": "http://test-host:11434"
}

# --- Test Cases ---

@patch("core.llm_manager.config_manager")
@patch("core.llm_manager.OllamaLLM")
def test_llm_manager_creates_ollama_instance(mock_ollama_class, mock_config_mgr):
    """
    Tests if LLMManager correctly initializes a LangChain Ollama instance
    based on the configuration.
    """
    # --- Arrange (準備) ---
    mock_config_mgr.get_llm_config.return_value = MOCK_OLLAMA_CONFIG
    mock_llm_instance = MagicMock()
    mock_ollama_class.return_value = mock_llm_instance

    from core.llm_manager import LLMManager
    llm_manager = LLMManager()

    # --- Act (実行) ---
    llm = llm_manager.get_llm("test_ollama")

    # --- Assert (検証) ---
    mock_config_mgr.get_llm_config.assert_called_once_with("test_ollama")
    mock_ollama_class.assert_called_once_with(
        model=MOCK_OLLAMA_CONFIG["model"],
        base_url=MOCK_OLLAMA_CONFIG["base_url"]
    )
    assert llm == mock_llm_instance

@patch("core.llm_manager.config_manager")
def test_llm_manager_raises_error_for_unknown_provider(mock_config_mgr):
    """
    Tests if LLMManager raises a ValueError for an unsupported provider.
    """
    # --- Arrange (準備) ---
    mock_config_mgr.get_llm_config.return_value = {"provider": "unknown_provider"}

    from core.llm_manager import LLMManager
    llm_manager = LLMManager()

    # --- Act & Assert ---
    with pytest.raises(ValueError, match="Unsupported LLM provider: unknown_provider"):
        llm_manager.get_llm("test_unknown")

@patch("core.llm_manager.config_manager")
def test_llm_manager_caches_instances(mock_config_mgr):
    """
    Tests if LLMManager caches and reuses LLM instances.
    """
    # --- Arrange (準備) ---
    mock_config_mgr.get_llm_config.return_value = MOCK_OLLAMA_CONFIG

    # This time, let the original Ollama class be called, but we don't care about the instance
    with patch("core.llm_manager.OllamaLLM") as mock_ollama_class:
        mock_ollama_class.return_value = MagicMock()

        from core.llm_manager import LLMManager
        llm_manager = LLMManager()

        # --- Act (実行) ---
        llm1 = llm_manager.get_llm("test_ollama")
        llm2 = llm_manager.get_llm("test_ollama") # Call it again

        # --- Assert (検証) ---
        assert llm1 is llm2 # Should be the exact same object
        mock_config_mgr.get_llm_config.assert_called_once_with("test_ollama") # Config should only be fetched once
        mock_ollama_class.assert_called_once() # Ollama should only be instantiated once
