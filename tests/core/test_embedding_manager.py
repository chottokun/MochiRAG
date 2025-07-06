import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from core.embedding_manager import EmbeddingManager, SentenceTransformerEmbedding, OllamaEmbeddingStrategy # EmbeddingStrategyは不要
from core.config_loader import StrategyConfigError # StrategyConfigErrorをインポート
from langchain_core.embeddings import Embeddings # Embeddingsの型チェック用

# テスト用の設定ファイルの内容 (例)
SAMPLE_CONFIG_VALID = {
    "embedding_strategies": {
        "default": "st_default",
        "available": [
            {"name": "st_default", "type": "sentence_transformer", "model_name": "all-MiniLM-L6-v2"},
            {"name": "ollama_test_embed", "type": "ollama_embedding", "model_name": "test_ollama_embed", "base_url": "http://localhost:11444"},
            {"name": "st_custom_params", "type": "sentence_transformer", "model_name": "paraphrase-MiniLM-L6-v2", "params": {"cache_folder": "./cache_test"}},
        ]
    }
}

SAMPLE_CONFIG_MINIMAL = {
    "embedding_strategies": {
        "available": [
            {"name": "st_minimal", "type": "sentence_transformer", "model_name": "all-MiniLM-L6-v2"}
        ]
    }
}

SAMPLE_CONFIG_INVALID_TYPE = {
    "embedding_strategies": {
        "available": [
            {"name": "invalid_type_strat", "type": "non_existent_type", "model_name": "test_model"}
        ]
    }
}

SAMPLE_CONFIG_MISSING_FIELDS = {
    "embedding_strategies": {
        "available": [
            {"name": "missing_fields_strat", "type": "sentence_transformer"} # model_name がない
        ]
    }
}


@pytest.fixture
def mock_load_config(monkeypatch):
    """load_strategy_configをモックし、指定したconfigデータを返すフィクスチャ"""
    def _mock_load_config(config_data_to_return):
        mock_func = MagicMock(return_value=config_data_to_return)
        monkeypatch.setattr("core.embedding_manager.load_strategy_config", mock_func)
        return mock_func
    return _mock_load_config

@pytest.fixture
def mock_sentence_transformer_embedding(monkeypatch):
    # SentenceTransformerEmbeddingの__init__とget_embedding_modelをモック
    mock_embedding_model_instance = MagicMock(spec=Embeddings)

    def mock_st_init(self, model_name, **kwargs):
        self.model_name = model_name
        self.config_kwargs = kwargs
        self._model = mock_embedding_model_instance # モックされたモデルインスタンスをセット
        # get_name がインスタンスのプロパティに依存する場合、それも設定
        self.name_for_get_name = f"sentence_transformer_{model_name.replace('/', '_')}"


    monkeypatch.setattr("core.embedding_manager.SentenceTransformerEmbedding.__init__", mock_st_init)
    monkeypatch.setattr("core.embedding_manager.SentenceTransformerEmbedding.get_embedding_model", lambda self: self._model)
    monkeypatch.setattr("core.embedding_manager.SentenceTransformerEmbedding.get_name", lambda self: self.name_for_get_name)


@pytest.fixture
def mock_ollama_embedding_strategy(monkeypatch):
    mock_embedding_model_instance = MagicMock(spec=Embeddings)

    def mock_ollama_init(self, model_name, base_url=None, **kwargs):
        self.model_name = model_name
        self.base_url = base_url
        self.config_kwargs = kwargs
        self._model = mock_embedding_model_instance
        self.name_for_get_name = f"ollama_embedding_{model_name}"

    monkeypatch.setattr("core.embedding_manager.OllamaEmbeddingStrategy.__init__", mock_ollama_init)
    monkeypatch.setattr("core.embedding_manager.OllamaEmbeddingStrategy.get_embedding_model", lambda self: self._model)
    monkeypatch.setattr("core.embedding_manager.OllamaEmbeddingStrategy.get_name", lambda self: self.name_for_get_name)


def test_embedding_manager_load_valid_config(mock_load_config, mock_sentence_transformer_embedding, mock_ollama_embedding_strategy):
    """正当な設定ファイルでEmbeddingManagerが正しく初期化されるか"""
    mock_load_config(SAMPLE_CONFIG_VALID)
    manager = EmbeddingManager()

    assert manager.default_strategy_name == "st_default"
    available_strategies = manager.get_available_strategies()
    assert "st_default" in available_strategies
    assert "ollama_test_embed" in available_strategies
    assert "st_custom_params" in available_strategies
    assert len(available_strategies) == 3

    st_default_strat = manager.get_strategy("st_default")
    assert isinstance(st_default_strat, SentenceTransformerEmbedding)
    assert st_default_strat.model_name == "all-MiniLM-L6-v2"

    ollama_strat = manager.get_strategy("ollama_test_embed")
    assert isinstance(ollama_strat, OllamaEmbeddingStrategy)
    assert ollama_strat.model_name == "test_ollama_embed"
    assert ollama_strat.base_url == "http://localhost:11444"

    st_custom_strat = manager.get_strategy("st_custom_params")
    assert isinstance(st_custom_strat, SentenceTransformerEmbedding)
    assert st_custom_strat.model_name == "paraphrase-MiniLM-L6-v2"
    assert st_custom_strat.config_kwargs == {"cache_folder": "./cache_test"}


def test_embedding_manager_minimal_config(mock_load_config, mock_sentence_transformer_embedding):
    """最小限の設定ファイルで正しく動作するか（デフォルト戦略のフォールバックなど）"""
    mock_load_config(SAMPLE_CONFIG_MINIMAL)
    manager = EmbeddingManager()

    assert manager.default_strategy_name == "st_minimal"
    assert "st_minimal" in manager.get_available_strategies()
    assert len(manager.get_available_strategies()) == 1

    strat = manager.get_strategy("st_minimal")
    assert isinstance(strat, SentenceTransformerEmbedding)
    assert strat.model_name == "all-MiniLM-L6-v2"

def test_embedding_manager_get_model(mock_load_config, mock_sentence_transformer_embedding):
    """get_embedding_modelが正しくモデルインスタンスを返すか"""
    mock_load_config(SAMPLE_CONFIG_VALID)
    manager = EmbeddingManager()
    model = manager.get_embedding_model("st_default")
    assert isinstance(model, Embeddings)

    default_model = manager.get_embedding_model()
    assert isinstance(default_model, Embeddings)


def test_embedding_manager_invalid_type_in_config(mock_load_config, caplog, mock_sentence_transformer_embedding, mock_ollama_embedding_strategy):
    """設定ファイルに不正な戦略タイプが含まれている場合の警告ログ"""
    mock_load_config(SAMPLE_CONFIG_INVALID_TYPE)
    manager = EmbeddingManager() # ここでロードが試みられる
    # caplog.text を確認して、期待する警告が出ているかチェック
    assert "Unsupported embedding strategy type: non_existent_type" in caplog.text
    assert "invalid_type_strat" not in manager.get_available_strategies()

def test_embedding_manager_missing_fields_in_config(mock_load_config, caplog, mock_sentence_transformer_embedding, mock_ollama_embedding_strategy):
    """設定ファイルの戦略定義で必須フィールドが欠けている場合の警告ログ"""
    mock_load_config(SAMPLE_CONFIG_MISSING_FIELDS)
    manager = EmbeddingManager()
    assert "Skipping incomplete embedding strategy config" in caplog.text
    assert "missing_fields_strat" not in manager.get_available_strategies()


def test_embedding_manager_get_non_existent_strategy(mock_load_config, mock_sentence_transformer_embedding, caplog):
    """存在しない戦略名でget_strategyを呼んだ場合にValueErrorが発生するか"""
    mock_load_config(SAMPLE_CONFIG_VALID)
    manager = EmbeddingManager()

    # 存在しない戦略名を指定
    non_existent_name = "non_existent_strategy_123"
    with pytest.raises(ValueError, match=f"EmbeddingManager: Embedding strategy '{non_existent_name}' not found."):
        manager.get_strategy(non_existent_name)

    # ログにエラーメッセージが含まれていることも確認（フォールバックしない場合）
    # manager.get_strategy のエラーメッセージがログに出力されることを期待
    # ただし、caplog は manager の初期化時のログも含むので注意
    # より厳密には、get_strategy呼び出し前後でログをキャプチャするか、
    # エラーメッセージがログに出ることを期待するなら、そのログレベルを確認する。
    # ここでは、ValueError が発生することを確認できれば十分とする。

    # デフォルト戦略も存在しないケース (例: 設定ファイルが空だった場合)
    caplog.clear()
    mock_load_config({"embedding_strategies": {"available": []}}) # 空の設定
    manager_no_strat = EmbeddingManager()
    assert not manager_no_strat.get_available_strategies()
    assert manager_no_strat.default_strategy_name is None
    with pytest.raises(StrategyConfigError, match="EmbeddingManager: No embedding strategy name provided and no default strategy is configured or available."):
        manager_no_strat.get_strategy() # 名前なし（デフォルト期待）でエラー
    with pytest.raises(StrategyConfigError, match="EmbeddingManager: No embedding strategy name provided and no default strategy is configured or available."):
        manager_no_strat.get_strategy(None) # 明示的にNoneでもエラー
    with pytest.raises(ValueError, match="EmbeddingManager: Embedding strategy 'any_name' not found."):
        manager_no_strat.get_strategy("any_name")


def test_embedding_manager_config_file_not_found(monkeypatch, caplog):
    """設定ファイルが見つからない場合にエラーを記録するか"""
    def mock_load_raises_error():
        raise StrategyConfigError("File not found for test")
    monkeypatch.setattr("core.embedding_manager.load_strategy_config", mock_load_raises_error)

    manager = EmbeddingManager()
    assert "Failed to load embedding strategy configuration: File not found for test" in caplog.text
    assert not manager.get_available_strategies()
    assert manager.default_strategy_name is None
