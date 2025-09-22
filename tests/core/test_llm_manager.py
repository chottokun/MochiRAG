import pytest
from unittest.mock import patch, MagicMock

# --- Fixtures ---
@pytest.fixture(autouse=True)
def clear_llm_manager_cache():
    """A fixture to automatically clear the LLMManager cache before each test."""
    # We must import here to avoid circular dependency issues at module load time
    from core.llm_manager import LLMManager
    LLMManager._llms.clear()


# --- Test Cases ---

@patch("core.llm_manager.config_manager")
@patch("core.llm_manager.ChatOllama")
def test_llm_manager_creates_ollama_instance(mock_chat_ollama_class, mock_config_mgr):
    """
    Tests if LLMManager correctly initializes a LangChain Ollama instance
    based on the configuration.
    """
    # --- Arrange (準備) ---
    # Mock the config object to support attribute access
    mock_config = MagicMock()
    mock_config.provider = "ollama"
    mock_config.model_name = "test-model"
    mock_config.base_url = "http://test-host:11434"
    mock_config_mgr.get_llm_config_by_role.return_value = mock_config

    mock_llm_instance = MagicMock()
    mock_chat_ollama_class.return_value = mock_llm_instance

    from core.llm_manager import LLMManager
    llm_manager = LLMManager()

    # --- Act (実行) ---
    llm = llm_manager.get_llm(role="test_ollama")

    # --- Assert (検証) ---
    mock_config_mgr.get_llm_config_by_role.assert_called_once_with("test_ollama")
    mock_chat_ollama_class.assert_called_once_with(
        model=mock_config.model_name,
        base_url=mock_config.base_url
    )
    assert llm == mock_llm_instance

@patch("core.llm_manager.config_manager")
def test_llm_manager_raises_error_for_unknown_provider(mock_config_mgr):
    """
    Tests if LLMManager raises a ValueError for an unsupported provider.
    """
    # --- Arrange (準備) ---
    # Mock the config object to support attribute access
    mock_config = MagicMock()
    mock_config.provider = "unknown_provider"
    mock_config_mgr.get_llm_config_by_role.return_value = mock_config

    from core.llm_manager import LLMManager
    llm_manager = LLMManager()

    # --- Act & Assert ---
    with pytest.raises(ValueError, match="Unsupported LLM provider: 'unknown_provider'"):
        llm_manager.get_llm(role="test_unknown")

@patch("core.llm_manager.config_manager")
def test_llm_manager_caches_instances(mock_config_mgr):
    """
    Tests if LLMManager caches and reuses LLM instances.
    """
    # --- Arrange (準備) ---
    mock_config = MagicMock()
    mock_config.provider = "ollama"
    mock_config.model_name = "test-model"
    mock_config.base_url = "http://test-host:11434"
    mock_config_mgr.get_llm_config_by_role.return_value = mock_config

    # This time, let the original Ollama class be called, but we don't care about the instance
    with patch("core.llm_manager.ChatOllama") as mock_chat_ollama_class:
        mock_chat_ollama_class.return_value = MagicMock()

        from core.llm_manager import LLMManager
        llm_manager = LLMManager()

        # --- Act (実行) ---
        llm1 = llm_manager.get_llm(role="test_ollama")
        llm2 = llm_manager.get_llm(role="test_ollama") # Call it again

        # --- Assert (検証) ---
        assert llm1 is llm2 # Should be the exact same object
        mock_chat_ollama_class.assert_called_once() # Ollama should only be instantiated once
