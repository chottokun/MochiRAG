import pytest
from unittest.mock import patch, MagicMock
from langchain_core.language_models import BaseLanguageModel
from langchain_ollama import ChatOllama
# from langchain_openai import ChatOpenAI # 将来的にOpenAIをテストする場合

from core.llm_manager import LLMManager
from core.config_loader import StrategyConfigError

# テスト用の設定ファイルの内容 (例)
SAMPLE_LLM_CONFIG_VALID = {
    "llm_config": {
        "default_provider": "ollama_default",
        "providers": [
            {"name": "ollama_default", "type": "ollama", "model": "test_model_1", "params": {"temperature": 0.1}},
            {"name": "ollama_custom_url", "type": "ollama", "model": "test_model_2", "params": {"temperature": 0.5, "base_url": "http://localhost:11223"}},
            # {"name": "openai_test", "type": "openai", "model": "gpt-3.5-turbo", "api_key_env": "TEST_OPENAI_API_KEY", "params": {"temperature": 0.7}}
        ]
    }
}

SAMPLE_LLM_CONFIG_MINIMAL = {
    "llm_config": {
        "providers": [
            {"name": "ollama_minimal", "type": "ollama", "model": "minimal_model"}
        ]
    }
}

SAMPLE_LLM_CONFIG_INVALID_TYPE = {
    "llm_config": {
        "providers": [
            {"name": "invalid_llm_type", "type": "non_existent_llm_type", "model": "test_model"}
        ]
    }
}

@pytest.fixture
def mock_load_config_llm(monkeypatch):
    def _mock_load_config(config_data_to_return):
        mock_func = MagicMock(return_value=config_data_to_return)
        monkeypatch.setattr("core.llm_manager.load_strategy_config", mock_func)
        return mock_func
    return _mock_load_config

@pytest.fixture
def mock_chat_ollama(monkeypatch):
    mock_instance = MagicMock(spec=ChatOllama)
    # invokeメソッドのモック (BaseLanguageModelインターフェースに合わせる)
    mock_instance.invoke.return_value = "Mocked Ollama Response"

    original_init = ChatOllama.__init__
    def mock_ollama_init(self, model, **kwargs):
        # 元の__init__の一部を模倣するか、必要な属性だけ設定
        self.model = model
        self.kwargs = kwargs
        # BaseLanguageModelとしての基本的な属性をモックに追加
        self._llm_type = "mocked_ollama"

    monkeypatch.setattr("langchain_ollama.ChatOllama.__init__", mock_ollama_init)
    # invokeがChatOllamaのインスタンスメソッドなので、クラスに直接ではなく、
    # インスタンスが作られた後にそのインスタンスのinvokeをモックするか、
    # __init__でインスタンスにモックされたinvokeをセットする。
    # ここでは、ChatOllamaがインスタンス化されるたびに、そのinvokeがMagicMockになるようにする。
    # ただし、上記の方法では ChatOllama のインスタンスの invoke はモックされない。
    # より確実なのは、ChatOllama自体をMagicMockで置き換えること。

    # ChatOllamaクラス自体をモックファクトリに置き換える
    mock_ollama_class = MagicMock(return_value=mock_instance)
    monkeypatch.setattr("core.llm_manager.ChatOllama", mock_ollama_class)
    return mock_ollama_class


# @pytest.fixture
# def mock_chat_openai(monkeypatch): # OpenAIをテストする場合
#     mock_instance = MagicMock(spec=ChatOpenAI)
#     mock_instance.invoke.return_value = "Mocked OpenAI Response"
#     def mock_openai_init(self, model_name, openai_api_key, **kwargs):
#         self.model_name = model_name
#         self.openai_api_key = openai_api_key
#         self.kwargs = kwargs
#         self._llm_type = "mocked_openai"
#     monkeypatch.setattr("langchain_openai.ChatOpenAI.__init__", mock_openai_init)
#     mock_openai_class = MagicMock(return_value=mock_instance)
#     monkeypatch.setattr("core.llm_manager.ChatOpenAI", mock_openai_class)
#     return mock_openai_class


def test_llm_manager_load_valid_config(mock_load_config_llm, mock_chat_ollama): # mock_chat_openai
    mock_load_config_llm(SAMPLE_LLM_CONFIG_VALID)
    manager = LLMManager()

    assert manager.default_provider_name == "ollama_default"
    available_providers = manager.get_available_providers()
    assert "ollama_default" in available_providers
    assert "ollama_custom_url" in available_providers
    # assert "openai_test" in available_providers # OpenAIテスト有効化時

    # ChatOllamaの呼び出し確認
    # mock_chat_ollama (モックされたクラス) が呼ばれたことを確認
    assert mock_chat_ollama.call_count >= 2 # 設定ファイル内のOllamaプロバイダの数

    # 最初のOllamaプロバイダの呼び出し引数を確認
    first_call_args = mock_chat_ollama.call_args_list[0][1] # kwargs
    assert first_call_args['model'] == "test_model_1"
    assert first_call_args['temperature'] == 0.1

    # 2番目のOllamaプロバイダの呼び出し引数を確認
    second_call_args = mock_chat_ollama.call_args_list[1][1]
    assert second_call_args['model'] == "test_model_2"
    assert second_call_args['temperature'] == 0.5
    assert second_call_args['base_url'] == "http://localhost:11223"

    # OpenAIのテスト (有効化した場合)
    # openai_provider = manager.get_llm("openai_test")
    # assert isinstance(openai_provider, BaseLanguageModel)
    # assert mock_chat_openai.call_args[1]['model_name'] == "gpt-3.5-turbo"


def test_llm_manager_minimal_config(mock_load_config_llm, mock_chat_ollama):
    mock_load_config_llm(SAMPLE_LLM_CONFIG_MINIMAL)
    manager = LLMManager()
    assert manager.default_provider_name == "ollama_minimal"
    assert "ollama_minimal" in manager.get_available_providers()

    llm_instance = manager.get_llm() # Default
    assert isinstance(llm_instance, BaseLanguageModel) # モックなので実際の型ではないが
    mock_chat_ollama.assert_called_with(model="minimal_model")


def test_llm_manager_get_llm_instance(mock_load_config_llm, mock_chat_ollama):
    mock_load_config_llm(SAMPLE_LLM_CONFIG_VALID)
    manager = LLMManager()

    llm_default = manager.get_llm("ollama_default")
    assert isinstance(llm_default, BaseLanguageModel)

    # invokeを呼び出せるか（モックされているので実際には通信しない）
    response = llm_default.invoke("test")
    assert response == "Mocked Ollama Response"

    # デフォルトLLMの取得
    default_llm = manager.get_llm() # nameなし
    assert isinstance(default_llm, BaseLanguageModel)
    response_default = default_llm.invoke("test default")
    assert response_default == "Mocked Ollama Response"


def test_llm_manager_invalid_type_in_config(mock_load_config_llm, caplog, mock_chat_ollama):
    mock_load_config_llm(SAMPLE_LLM_CONFIG_INVALID_TYPE)
    manager = LLMManager()
    assert "Unsupported LLM provider type: non_existent_llm_type" in caplog.text
    assert "invalid_llm_type" not in manager.get_available_providers()


def test_llm_manager_get_non_existent_provider(mock_load_config_llm, mock_chat_ollama):
    mock_load_config_llm(SAMPLE_LLM_CONFIG_VALID)
    manager = LLMManager()
    with pytest.raises(ValueError, match="LLM provider 'non_existent_provider_123' not found"):
        manager.get_llm("non_existent_provider_123")


def test_llm_manager_config_file_not_found(monkeypatch, caplog):
    def mock_load_raises_error():
        raise StrategyConfigError("LLM Config file not found for test")
    monkeypatch.setattr("core.llm_manager.load_strategy_config", mock_load_raises_error)

    manager = LLMManager()
    assert "Failed to load LLM configuration: LLM Config file not found for test" in caplog.text
    assert not manager.get_available_providers()
    assert manager.default_provider_name is None

# グローバルなllmインスタンスが正しく設定されるかのテスト
def test_global_llm_instance_after_manager_init(mock_load_config_llm, mock_chat_ollama):
    mock_load_config_llm(SAMPLE_LLM_CONFIG_VALID)
    # LLMManagerのインスタンス化時にグローバルなllmも設定されるはず
    # ただし、モジュールレベルのllmはインポート時に評価されるため、
    # managerの初期化後に再度llmを参照する形でテストする
    from core import llm_manager as lm_module
    lm_module.llm_manager = LLMManager() # 再初期化

    assert lm_module.llm is not None
    assert isinstance(lm_module.llm, BaseLanguageModel)
    # デフォルトプロバイダ ("ollama_default") のモデルが設定されるはず
    # mock_chat_ollama が最後に呼ばれた際の model を確認
    # ただし、mock_chat_ollama はクラスのモックなので、インスタンスの属性確認は難しい
    # ここでは、llm_manager.get_llm() の結果と一致するかで代用
    assert lm_module.llm == lm_module.llm_manager.get_llm(lm_module.llm_manager.default_provider_name)

    # もしグローバルllmがモックされたChatOllamaインスタンスを指しているなら
    # そのinvokeもモックされているはず
    assert lm_module.llm.invoke("test global") == "Mocked Ollama Response"
