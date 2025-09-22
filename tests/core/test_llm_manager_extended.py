import pytest
from unittest.mock import patch, MagicMock

# --- Fixtures ---
@pytest.fixture(autouse=True)
def clear_llm_manager_cache():
    """A fixture to automatically clear the LLMManager cache before each test."""
    from core.llm_manager import LLMManager
    LLMManager._llms.clear()

# --- Test Cases ---

@patch("core.llm_manager.config_manager")
@patch("core.llm_manager.ChatOpenAI")
def test_llm_manager_creates_openai_instance(mock_chat_openai_class, mock_config_mgr):
    """
    Tests if LLMManager correctly initializes a LangChain OpenAI instance.
    """
    # --- Arrange (準備) ---
    mock_config = MagicMock()
    mock_config.provider = "openai"
    mock_config.model_name = "gpt-4"
    mock_config.api_key = "test-key"
    mock_config.temperature = 0.5
    mock_config_mgr.get_llm_config_by_role.return_value = mock_config

    mock_llm_instance = MagicMock()
    mock_chat_openai_class.return_value = mock_llm_instance

    from core.llm_manager import LLMManager
    llm_manager = LLMManager()

    # --- Act (実行) ---
    llm = llm_manager.get_llm(role="test_openai")

    # --- Assert (検証) ---
    mock_config_mgr.get_llm_config_by_role.assert_called_once_with("test_openai")
    mock_chat_openai_class.assert_called_once_with(
        model=mock_config.model_name,
        api_key=mock_config.api_key,
        temperature=mock_config.temperature
    )
    assert llm == mock_llm_instance

@patch("core.llm_manager.config_manager")
@patch("core.llm_manager.AzureChatOpenAI")
def test_llm_manager_creates_azure_instance(mock_azure_chat_class, mock_config_mgr):
    """
    Tests if LLMManager correctly initializes a LangChain Azure OpenAI instance.
    """
    # --- Arrange (準備) ---
    mock_config = MagicMock()
    mock_config.provider = "azure"
    mock_config.deployment_name = "test-deployment"
    mock_config.azure_endpoint = "http://test.azure.com"
    mock_config.api_version = "2023-05-15"
    mock_config.api_key = "azure-test-key"
    mock_config.temperature = 0.8
    mock_config_mgr.get_llm_config_by_role.return_value = mock_config

    mock_llm_instance = MagicMock()
    mock_azure_chat_class.return_value = mock_llm_instance

    from core.llm_manager import LLMManager
    llm_manager = LLMManager()

    # --- Act (実行) ---
    llm = llm_manager.get_llm(role="test_azure")

    # --- Assert (検証) ---
    mock_config_mgr.get_llm_config_by_role.assert_called_once_with("test_azure")
    mock_azure_chat_class.assert_called_once_with(
        azure_deployment=mock_config.deployment_name,
        azure_endpoint=mock_config.azure_endpoint,
        api_version=mock_config.api_version,
        api_key=mock_config.api_key,
        temperature=mock_config.temperature
    )
    assert llm == mock_llm_instance

@patch("core.llm_manager.config_manager")
@patch("core.llm_manager.ChatGoogleGenerativeAI")
def test_llm_manager_creates_gemini_instance(mock_gemini_chat_class, mock_config_mgr):
    """
    Tests if LLMManager correctly initializes a LangChain Google Gemini instance.
    """
    # --- Arrange (準備) ---
    mock_config = MagicMock()
    mock_config.provider = "gemini"
    mock_config.model_name = "gemini-pro"
    mock_config.api_key = "gemini-test-key"
    mock_config.temperature = 0.9
    mock_config_mgr.get_llm_config_by_role.return_value = mock_config

    mock_llm_instance = MagicMock()
    mock_gemini_chat_class.return_value = mock_llm_instance

    from core.llm_manager import LLMManager
    llm_manager = LLMManager()

    # --- Act (実行) ---
    llm = llm_manager.get_llm(role="test_gemini")

    # --- Assert (検証) ---
    mock_config_mgr.get_llm_config_by_role.assert_called_once_with("test_gemini")
    mock_gemini_chat_class.assert_called_once_with(
        model=mock_config.model_name,
        google_api_key=mock_config.api_key,
        temperature=mock_config.temperature
    )
    assert llm == mock_llm_instance
