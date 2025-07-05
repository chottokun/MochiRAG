import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings # SemanticChunkingStrategyが型ヒントで必要とする可能性

from core.chunking_manager import ChunkingManager, RecursiveTextSplitterChunking, SemanticChunkingStrategy, ChunkingStrategy
from core.config_loader import StrategyConfigError
# EmbeddingManagerのモックやインスタンスが必要になる場合がある（SemanticChunkingのため）
from core.embedding_manager import EmbeddingManager


# テスト用の設定ファイルの内容 (例)
SAMPLE_CONFIG_VALID_CHUNKING = {
    "embedding_strategies": { # SemanticChunkerが参照するためダミーでも定義
        "default": "mock_embedding",
        "available": [{"name": "mock_embedding", "type": "sentence_transformer", "model_name": "mock"}]
    },
    "chunking_strategies": {
        "default": "recursive_default",
        "available": [
            {"name": "recursive_default", "type": "recursive_text_splitter", "params": {"chunk_size": 500, "chunk_overlap": 50}},
            {"name": "recursive_small", "type": "recursive_text_splitter", "params": {"chunk_size": 100, "chunk_overlap": 10}},
            # SemanticChunkerのテストのためには、対応するembedding_strategy_refが必要
            {"name": "semantic_test", "type": "semantic_chunker", "embedding_strategy_ref": "mock_embedding", "params": {"breakpoint_threshold_type": "standard_deviation"}}
        ]
    }
}

SAMPLE_CONFIG_MINIMAL_CHUNKING = {
    "chunking_strategies": {
        "available": [
            {"name": "recursive_minimal", "type": "recursive_text_splitter", "params": {"chunk_size": 800, "chunk_overlap": 80}}
        ]
    }
}


@pytest.fixture
def mock_load_config_chunking(monkeypatch):
    def _mock_load_config(config_data_to_return):
        mock_func = MagicMock(return_value=config_data_to_return)
        # chunking_manager と embedding_manager の両方で同じ config_loader を使う想定
        monkeypatch.setattr("core.chunking_manager.load_strategy_config", mock_func)
        monkeypatch.setattr("core.embedding_manager.load_strategy_config", mock_func) # EmbeddingManagerもモック
        return mock_func
    return _mock_load_config

@pytest.fixture
def mock_embedding_manager_for_chunking(monkeypatch):
    """SemanticChunkingStrategyが使用するEmbeddingManagerをモック"""
    mock_manager = MagicMock(spec=EmbeddingManager)
    mock_embedding_model = MagicMock(spec=Embeddings)
    mock_manager.get_embedding_model.return_value = mock_embedding_model
    # グローバルな `embedding_manager` インスタンスをこのモックで置き換える
    monkeypatch.setattr("core.chunking_manager.embedding_manager", mock_manager)
    return mock_manager

# RecursiveTextSplitterChunkingはLangchainの機能に依存するため、ここでは主にManagerのロジックをテスト
# SemanticChunkingStrategyも同様に、LangchainのSemanticChunkerの動作ではなく、Managerからの呼び出しをテスト

def test_chunking_manager_load_valid_config(mock_load_config_chunking, mock_embedding_manager_for_chunking):
    mock_load_config_chunking(SAMPLE_CONFIG_VALID_CHUNKING)
    manager = ChunkingManager()

    assert manager.default_strategy_name == "recursive_default"
    available = manager.get_available_strategies()
    assert "recursive_default" in available
    assert "recursive_small" in available
    assert "semantic_test" in available # embedding_managerがモックされていればロード試行される

    strat_default = manager.get_strategy("recursive_default")
    assert isinstance(strat_default, RecursiveTextSplitterChunking)
    assert strat_default.chunk_size == 500
    assert strat_default.get_config()['type'] == "recursive_text_splitter"

    # SemanticChunkerのインスタンス化が試みられることを確認（モックにより成功するはず）
    # mock_embedding_manager_for_chunking が呼ばれることで確認
    strat_semantic = manager.get_strategy("semantic_test")
    assert isinstance(strat_semantic, SemanticChunkingStrategy)
    mock_embedding_manager_for_chunking.get_embedding_model.assert_called_with("mock_embedding")


def test_chunking_manager_minimal_config(mock_load_config_chunking, mock_embedding_manager_for_chunking):
    mock_load_config_chunking(SAMPLE_CONFIG_MINIMAL_CHUNKING)
    manager = ChunkingManager()
    assert manager.default_strategy_name == "recursive_minimal"
    assert "recursive_minimal" in manager.get_available_strategies()
    strat = manager.get_strategy() # Default
    assert isinstance(strat, RecursiveTextSplitterChunking)
    assert strat.chunk_size == 800

def test_chunking_manager_get_strategy_with_params(mock_load_config_chunking, mock_embedding_manager_for_chunking):
    mock_load_config_chunking(SAMPLE_CONFIG_VALID_CHUNKING) # 初期ロード用
    manager = ChunkingManager()

    # パラメータ指定でRecursiveTextSplitterChunkingを取得
    params_recursive = {"chunk_size": 250, "chunk_overlap": 25}
    # name は基本タイプを指定し、paramsで詳細を指定する想定
    strat_recursive_dynamic = manager.get_strategy("recursive_text_splitter", params=params_recursive)
    assert isinstance(strat_recursive_dynamic, RecursiveTextSplitterChunking)
    assert strat_recursive_dynamic.chunk_size == 250
    assert strat_recursive_dynamic.chunk_overlap == 25
    # 動的に作られたものはmanager.strategiesには追加されない想定（get_strategyの実装による）

    # パラメータ指定でSemanticChunkingStrategyを取得
    params_semantic = {
        "embedding_strategy_ref": "mock_embedding", # VSM等から渡される想定
        "breakpoint_threshold_type": "interquartile",
        "additional_params": {"some_other_param": True}
    }
    strat_semantic_dynamic = manager.get_strategy("semantic_chunker", params=params_semantic)
    assert isinstance(strat_semantic_dynamic, SemanticChunkingStrategy)
    assert strat_semantic_dynamic.breakpoint_threshold_type == "interquartile"
    assert strat_semantic_dynamic.kwargs == {"some_other_param": True}
    mock_embedding_manager_for_chunking.get_embedding_model.assert_called_with("mock_embedding")


def test_chunking_manager_get_non_existent_strategy_fallback(mock_load_config_chunking, mock_embedding_manager_for_chunking, caplog):
    mock_load_config_chunking(SAMPLE_CONFIG_VALID_CHUNKING)
    manager = ChunkingManager()

    # 存在しない戦略名だが、"recursive_text_splitter_cs..." 形式なら動的生成を試みる
    # この形式にマッチしない場合は、デフォルトにフォールバックする
    strat = manager.get_strategy("non_existent_strat_name")
    assert "Chunking strategy 'non_existent_strat_name' not found" in caplog.text
    assert "Falling back to default strategy" in caplog.text
    assert isinstance(strat, RecursiveTextSplitterChunking) # デフォルトのRecursiveにフォールバック
    assert strat.get_name() == manager.default_strategy_name


def test_chunking_strategy_split_documents():
    """ChunkingStrategyの基本的な動作確認"""
    docs = [Document(page_content="This is a test document. " * 20)]

    # Recursive
    recursive_strat = RecursiveTextSplitterChunking(chunk_size=50, chunk_overlap=5)
    chunks = recursive_strat.split_documents(docs)
    assert len(chunks) > 1
    assert all(isinstance(c, Document) for c in chunks)
    assert len(chunks[0].page_content) <= 50

    # Semantic (モックされたEmbeddingsでテスト)
    # このテストはLangchainのSemanticChunkerの単体テストに近いが、インターフェース確認のため
    mock_embeddings = MagicMock(spec=Embeddings)
    # SemanticChunkerがembed_documentsなどを呼ぶので、それもモックする必要があるかもしれない
    mock_embeddings.embed_documents = MagicMock(return_value=[[0.1]*10]*len(docs[0].page_content.split(". "))) # ダミーのベクトルリスト
    mock_embeddings.embed_query = MagicMock(return_value=[0.1]*10)


    # SemanticChunkerの依存ライブラリがテスト環境にない場合を考慮
    try:
        semantic_strat = SemanticChunkingStrategy(embedding_model_instance=mock_embeddings, breakpoint_threshold_type="percentile")
        # SemanticChunkerはテキストが短すぎるとエラーになることがあるので長めのテキストで
        long_text = "Sentence 1. Sentence 2. Sentence 3 is a bit longer. Sentence 4. Sentence 5. " * 10
        semantic_docs = [Document(page_content=long_text)]
        chunks_semantic = semantic_strat.split_documents(semantic_docs)
        assert len(chunks_semantic) >= 1 # 少なくとも1つにはなるはず
        assert all(isinstance(c, Document) for c in chunks_semantic)
    except (ImportError, ValueError) as e:
        pytest.skip(f"Skipping SemanticChunkingStrategy test due to missing dependency or init error: {e}")


def test_chunking_manager_config_file_not_found(monkeypatch, caplog):
    """設定ファイルが見つからない場合にエラーを記録するか"""
    def mock_load_raises_error():
        raise StrategyConfigError("File not found for chunking test")
    monkeypatch.setattr("core.chunking_manager.load_strategy_config", mock_load_raises_error)

    manager = ChunkingManager() # 初期化時にロードが試みられる
    assert "Failed to load chunking strategy configuration: File not found for chunking test" in caplog.text
    assert not manager.get_available_strategies()
    assert manager.default_strategy_name is None
