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
    assert "Unsupported LLM provider type: non_existent_llm_type" in caplog.text # ログメッセージの確認
    assert "invalid_llm_type" not in manager.get_available_providers()


def test_llm_manager_get_non_existent_provider(mock_load_config_llm, mock_chat_ollama, caplog):
    mock_load_config_llm(SAMPLE_LLM_CONFIG_VALID)
    manager = LLMManager()
    non_existent_name = "non_existent_provider_123"
    with pytest.raises(ValueError, match=f"LLMManager: LLM provider '{non_existent_name}' not found."):
        manager.get_llm(non_existent_name)

    caplog.clear()
    mock_load_config_llm({"llm_config": {"providers": []}}) # 空のプロバイダーリスト
    manager_no_providers = LLMManager()
    assert not manager_no_providers.get_available_providers()
    assert manager_no_providers.default_provider_name is None
    with pytest.raises(StrategyConfigError, match="LLMManager: No LLM provider name specified and no default provider is configured or available."):
        manager_no_providers.get_llm()
    with pytest.raises(ValueError, match="LLMManager: LLM provider 'any_name' not found."):
        manager_no_providers.get_llm("any_name")


def test_llm_manager_config_file_not_found(monkeypatch, caplog):
    def mock_load_raises_error():
        raise StrategyConfigError("LLM Config file not found for test")
    monkeypatch.setattr("core.llm_manager.load_strategy_config", mock_load_raises_error)

    manager = LLMManager()
    assert "LLMManager: Failed to load strategy configuration: LLM Config file not found for test" in caplog.text # Manager名をログに追加
    assert not manager.get_available_providers()
    assert manager.default_provider_name is None

# グローバルなllmインスタンスの代わりに、manager経由での取得をテスト
def test_get_llm_via_manager_instance(mock_load_config_llm, mock_chat_ollama):
    mock_load_config_llm(SAMPLE_LLM_CONFIG_VALID)

    # core.llm_manager を再インポートするかのように、新しいインスタンスでテストする
    # ただし、グローバルな llm_manager インスタンスはモジュールロード時に作られるので、
    # それを直接テストする（再代入はpytestの挙動を複雑にする可能性がある）
    from core.llm_manager import llm_manager as global_llm_manager_instance
    # ここで global_llm_manager_instance はモックされた設定で再初期化されている想定
    # （mock_load_config_llmがグローバルなload_strategy_configをモックしているため）
    # 必要であれば、LLMManager()を直接呼び出して新しいインスタンスを作る
    manager_to_test = LLMManager() # これでモックされたconfigが使われる

    assert manager_to_test.default_provider_name == "ollama_default"
    default_llm = manager_to_test.get_llm() # デフォルトプロバイダのLLMを取得

    assert default_llm is not None
    assert isinstance(default_llm, BaseLanguageModel)

    # ChatOllamaの呼び出しが期待通りか（model名で判断）
    # mock_chat_ollama はクラスのモックなので、最後に呼ばれた際の引数を見る
    # manager_to_test の初期化で複数回呼ばれる可能性があるため、
    # default_provider_name に紐づく呼び出しを特定するのは難しい。
    # ここでは、get_llm() で取得したインスタンスが期待する型であること、
    # そしてその invoke がモックされていることを確認する。
    assert default_llm.invoke("test global") == "Mocked Ollama Response"

    # 特定のプロバイダーも取得できるか
    custom_llm = manager_to_test.get_llm("ollama_custom_url")
    assert isinstance(custom_llm, BaseLanguageModel)
    assert custom_llm.invoke("test custom") == "Mocked Ollama Response"
