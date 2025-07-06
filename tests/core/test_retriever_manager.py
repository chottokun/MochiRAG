import pytest
from unittest.mock import patch, MagicMock, ANY
from langchain_core.retrievers import BaseRetriever
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from core.retriever_manager import RetrieverManager, BasicRetrieverStrategy, MultiQueryRetrieverStrategy, ContextualCompressionRetrieverStrategy, ParentDocumentRetrieverStrategy
from core.config_loader import StrategyConfigError
from core.embedding_manager import EmbeddingManager
from core.vector_store_manager import VectorStoreManager
from core.llm_manager import LLMManager
from langchain_chroma import Chroma # BasicRetrieverStrategy内で直接呼ばれるため
from langchain.text_splitter import RecursiveCharacterTextSplitter # ParentDocumentRetrieverStrategyで利用


# テスト用の設定ファイルの内容 (RAG検索戦略部分のみ抜粋)
# 各戦略に "type" フィールドを追加
SAMPLE_RAG_SEARCH_CONFIG = {
    "rag_search_strategies": {
        "default": "basic_search_config_name", # config内のnameフィールドを参照
        "available": [
            {"name": "basic_search_config_name", "type": "basic", "description": "Basic"},
            {"name": "mq_search_config_name", "type": "multi_query", "description": "Multi Query"},
            {"name": "cc_search_config_name", "type": "contextual_compression", "description": "Contextual Compression"},
            {"name": "pd_search_config_name", "type": "parent_document", "description": "Parent Document"},
            {"name": "deep_rag_config_name", "type": "deep_rag", "description": "Deep RAG"},
        ]
    },
    # 他のマネージャが依存する設定も最小限定義
    "embedding_strategies": {"default": "mock_emb", "available": [{"name": "mock_emb", "type": "sentence_transformer", "model_name": "mock"}]},
    "llm_config": {"default_provider": "mock_llm", "providers": [{"name": "mock_llm", "type": "ollama", "model": "mock"}]}
}

@pytest.fixture
def mock_load_config_retriever(monkeypatch):
    def _mock_load_config(config_data_to_return):
        mock_func = MagicMock(return_value=config_data_to_return)
        monkeypatch.setattr("core.retriever_manager.load_strategy_config", mock_func)
        # 依存するマネージャーも同じ設定ファイルを読むので、それらもモック
        monkeypatch.setattr("core.embedding_manager.load_strategy_config", mock_func)
        monkeypatch.setattr("core.llm_manager.load_strategy_config", mock_func)
        monkeypatch.setattr("core.chunking_manager.load_strategy_config", mock_func) # ParentDocumentRetrieverがtext_splitter経由で利用可能性あり
        return mock_func
    return _mock_load_config

@pytest.fixture
def mock_embedding_manager(monkeypatch):
    mock_manager = MagicMock(spec=EmbeddingManager)
    mock_manager.get_embedding_model.return_value = MagicMock(spec=Embeddings)
    monkeypatch.setattr("core.retriever_manager.embedding_manager", mock_manager)
    return mock_manager

@pytest.fixture
def mock_vector_store_manager(monkeypatch):
    mock_manager = MagicMock(spec=VectorStoreManager)
    # BasicRetrieverStrategyがChromaを直接初期化する部分があるので、VSMのpersist_directoryをモック
    mock_manager.persist_directory = "/mock/persist/dir"
    monkeypatch.setattr("core.retriever_manager.vector_store_manager", mock_manager)
    return mock_manager

@pytest.fixture
def mock_llm_manager(monkeypatch):
    mock_manager = MagicMock(spec=LLMManager)
    mock_manager.get_llm.return_value = MagicMock(spec=BaseLanguageModel)
    monkeypatch.setattr("core.retriever_manager.llm_manager", mock_manager)
    # RetrieverManager内の戦略クラスがllm_managerを直接参照している場合も考慮
    monkeypatch.setattr("core.retriever_manager.MultiQueryRetrieverStrategy.llm_manager", mock_manager, raising=False) # raising=Falseで存在しなくてもエラーにしない
    monkeypatch.setattr("core.retriever_manager.ContextualCompressionRetrieverStrategy.llm_manager", mock_manager, raising=False)
    return mock_manager

@pytest.fixture
def mock_chroma_retriever(monkeypatch):
    """Chroma.as_retriever をモック"""
    mock_retriever_instance = MagicMock(spec=BaseRetriever)
    mock_retriever_instance.invoke.return_value = [] # dummy documents

    def mock_as_retriever(self, **kwargs):
        # kwargs に search_kwargs が含まれることを確認できる
        return mock_retriever_instance

    monkeypatch.setattr(Chroma, "as_retriever", mock_as_retriever)
    # Chromaのコンストラクタもモックして、実際のDBアクセスを避ける
    monkeypatch.setattr(Chroma, "__init__", MagicMock(return_value=None))
    return mock_retriever_instance

@pytest.fixture
def mock_default_text_splitter(monkeypatch):
    mock_splitter = MagicMock(spec=RecursiveCharacterTextSplitter)
    monkeypatch.setattr("core.retriever_manager.default_text_splitter", mock_splitter)
    return mock_splitter


def test_retriever_manager_load_valid_config(
    mock_load_config_retriever, mock_embedding_manager, mock_vector_store_manager, mock_llm_manager, mock_chroma_retriever, mock_default_text_splitter
):
    manager = RetrieverManager()

    assert manager.default_strategy_name == "basic_search_config_name"
    available = manager.get_available_strategies()
    assert "basic_search_config_name" in available
    assert "mq_search_config_name" in available
    assert "cc_search_config_name" in available
    assert "pd_search_config_name" in available
    assert "deep_rag_config_name" in available # deep_ragが登録されていることを確認
    assert len(available) == 5


    # 各戦略のインスタンスタイプ確認
    assert isinstance(manager.strategies["basic_search_config_name"], BasicRetrieverStrategy)
    assert isinstance(manager.strategies["mq_search_config_name"], MultiQueryRetrieverStrategy)
    assert isinstance(manager.strategies["cc_search_config_name"], ContextualCompressionRetrieverStrategy)
    assert isinstance(manager.strategies["pd_search_config_name"], ParentDocumentRetrieverStrategy)
    assert isinstance(manager.strategies["deep_rag_config_name"], DeepRagRetrieverStrategy)


def test_get_basic_retriever(
    mock_load_config_retriever, mock_embedding_manager, mock_vector_store_manager, mock_llm_manager, mock_chroma_retriever
):
    mock_load_config_retriever(SAMPLE_RAG_SEARCH_CONFIG)
    manager = RetrieverManager()
    retriever = manager.get_retriever(
        user_id="test_user",
        embedding_strategy_name="mock_emb",
        name="basic_search_config_name"
    )
    assert retriever is not None # mock_chroma_retriever が返ることを期待 (BasicRetrieverStrategy内)
    # より具体的には、mock_chroma_retriever が BasicRetrieverStrategy の get_retriever から返されることを確認
    # これは mock_chroma_retriever がグローバルなモックであるため、直接比較は難しい。
    # Chroma.as_retriever が呼ばれたことを確認する方が適切かもしれない。
    # しかし、mock_chroma_retriever は Chroma.as_retriever の返り値のモックなので、
    # get_retrieverが成功し、このモックが返ればOKとする。

@patch("langchain.retrievers.MultiQueryRetriever.from_llm")
def test_get_multi_query_retriever(
    mock_from_llm, mock_load_config_retriever, mock_embedding_manager,
    mock_vector_store_manager, mock_llm_manager, mock_chroma_retriever
):
    mock_ret_instance = MagicMock(spec=BaseRetriever)
    mock_from_llm.return_value = mock_ret_instance
    mock_load_config_retriever(SAMPLE_RAG_SEARCH_CONFIG)

    manager = RetrieverManager()
    retriever = manager.get_retriever(
        user_id="test_user",
        embedding_strategy_name="mock_emb",
        name="mq_search_config_name"
    )
    assert retriever == mock_ret_instance
    mock_from_llm.assert_called_once()
    mock_llm_manager.get_llm.assert_called_once()


@patch("langchain.retrievers.ContextualCompressionRetriever")
@patch("langchain.retrievers.document_compressors.LLMChainExtractor.from_llm")
def test_get_contextual_compression_retriever(
    mock_extractor_from_llm, mock_cc_retriever_class, mock_load_config_retriever,
    mock_embedding_manager, mock_vector_store_manager, mock_llm_manager, mock_chroma_retriever
):
    mock_compressor = MagicMock()
    mock_extractor_from_llm.return_value = mock_compressor
    mock_ret_instance = MagicMock(spec=BaseRetriever)
    mock_cc_retriever_class.return_value = mock_ret_instance

    mock_load_config_retriever(SAMPLE_RAG_SEARCH_CONFIG)
    manager = RetrieverManager()
    retriever = manager.get_retriever(
        user_id="test_user",
        embedding_strategy_name="mock_emb",
        name="cc_search_config_name"
    )
    assert retriever == mock_ret_instance
    mock_extractor_from_llm.assert_called_once_with(mock_llm_manager.get_llm())
    mock_cc_retriever_class.assert_called_once_with(
        base_compressor=mock_compressor, base_retriever=ANY
    )

@patch("langchain.retrievers.ParentDocumentRetriever")
def test_get_parent_document_retriever(
    mock_pd_retriever_class, mock_load_config_retriever, mock_embedding_manager,
    mock_vector_store_manager, mock_llm_manager, mock_chroma_retriever, mock_default_text_splitter
):
    mock_ret_instance = MagicMock(spec=BaseRetriever)
    mock_pd_retriever_class.return_value = mock_ret_instance
    mock_load_config_retriever(SAMPLE_RAG_SEARCH_CONFIG)

    manager = RetrieverManager()
    retriever = manager.get_retriever(
        user_id="test_user",
        embedding_strategy_name="mock_emb",
        name="pd_search_config_name"
    )
    assert retriever == mock_ret_instance
    mock_pd_retriever_class.assert_called_once_with(
        vectorstore=ANY,
        docstore=ANY,
        child_splitter=mock_default_text_splitter,
        search_kwargs=ANY
    )

@patch("core.retriever_manager.DeepRagRetrieverStrategy._decompose_query", new_callable=MagicMock)
@patch.object(BasicRetrieverStrategy, "get_retriever")
def test_get_deep_rag_retriever_and_custom_retriever_logic(
    mock_basic_get_retriever, mock_decompose_query,
    mock_load_config_retriever, mock_embedding_manager,
    mock_vector_store_manager, mock_llm_manager, mock_chroma_retriever
):
    # --- Setup Mocks ---
    mock_decompose_query.return_value = ["sub_query_1", "sub_query_2"]

    mock_sub_retriever_instance1 = MagicMock(spec=BaseRetriever)
    mock_sub_retriever_instance1.invoke.return_value = [
        Document(page_content="doc from sub_query_1", metadata={"data_source_id": "ds1", "id": "doc1"})
    ]
    mock_sub_retriever_instance2 = MagicMock(spec=BaseRetriever)
    mock_sub_retriever_instance2.invoke.return_value = [
        Document(page_content="doc from sub_query_2", metadata={"data_source_id": "ds2", "id": "doc2"}),
        Document(page_content="doc from sub_query_1", metadata={"data_source_id": "ds1", "id": "doc1"})
    ]
    mock_basic_get_retriever.side_effect = [mock_sub_retriever_instance1, mock_sub_retriever_instance2]

    mock_load_config_retriever(SAMPLE_RAG_SEARCH_CONFIG)
    manager = RetrieverManager()

    deep_rag_retriever = manager.get_retriever(
        user_id="test_user_deep",
        embedding_strategy_name="mock_emb",
        name="deep_rag_config_name",
        user_id="test_user_deep",
        embedding_strategy_name="mock_emb",
        data_source_ids=["ds1", "ds2"],
        n_results=2, # これはDeepRagCustomRetrieverのn_results_per_subqueryに渡される
        max_sub_queries=2
    )
    assert isinstance(deep_rag_retriever, BaseRetriever) # DeepRagCustomRetrieverもBaseRetrieverを継承
    assert hasattr(deep_rag_retriever, "_get_relevant_documents") # カスタムリトリーバーであることの確認

    # --- Test DeepRagCustomRetriever._get_relevant_documents logic ---
    # RetrieverManagerから取得したリトリーバーのinvokeを呼び出すことで、
    # DeepRagCustomRetrieverの_get_relevant_documentsが実行される
    main_query = "complex original query"
    retrieved_docs = deep_rag_retriever.invoke(main_query)

    # 1. _decompose_query (モックされたもの) がメインクエリで呼ばれたか
    #    DeepRagCustomRetriever内部で呼ばれるので、直接DeepRagRetrieverStrategyのメソッドをモックするのではなく、
    #    DeepRagCustomRetriever._decompose_query_internal をモックする必要がある。
    #    今回は、DeepRagRetrieverStrategy._decompose_query が呼ばれることを期待するのではなく、
    #    DeepRagCustomRetriever が LLM を使って分解することをテストしたい。
    #    DeepRagRetrieverStrategyの_decompose_queryはDeepRagCustomRetriever内では直接使われていない。
    #    DeepRagCustomRetriever内の_decompose_query_internalが呼ばれる。
    #    このテストでは、DeepRagCustomRetrieverのインスタンスメソッドをモックするのは難しい。
    #    代わりに、llm_manager.get_llm().invoke が呼ばれた回数などで間接的に確認する。

    # LLMがクエリ分解のために呼ばれたことを確認 (prompt | llm | parser の形で呼ばれる)
    assert mock_llm_manager.get_llm().invoke.call_count >= 1 # 少なくとも1回は分解のために呼ばれる

    # 2. サブクエリごとにBasicRetrieverStrategy().get_retrieverが呼ばれたか
    assert mock_basic_get_retriever.call_count == 2 # "sub_query_1", "sub_query_2" の2回

    # 各サブクエリ検索の呼び出し引数を確認
    first_call_args = mock_basic_get_retriever.call_args_list[0][1]
    assert first_call_args["user_id"] == "test_user_deep"
    assert first_call_args["embedding_strategy_name"] == "mock_emb"
    assert first_call_args["data_source_ids"] == ["ds1", "ds2"]
    assert first_call_args["n_results"] == 2 # n_results_per_subquery

    # 3. 各サブクエリリトリーバーのinvokeが呼ばれたか
    mock_sub_retriever_instance1.invoke.assert_called_once_with("sub_query_1")
    mock_sub_retriever_instance2.invoke.assert_called_once_with("sub_query_2")

    # 4. 結果のドキュメントリストの検証（重複排除後）
    assert len(retrieved_docs) == 2 # doc1とdoc2 (doc1の重複は排除される)
    assert any(d.page_content == "doc from sub_query_1" for d in retrieved_docs)
    assert any(d.page_content == "doc from sub_query_2" for d in retrieved_docs)


def test_retriever_manager_get_non_existent_strategy_fallback(
    mock_load_config_retriever, mock_embedding_manager, mock_vector_store_manager,
    mock_llm_manager, mock_chroma_retriever, caplog
):
    mock_load_config_retriever(SAMPLE_RAG_SEARCH_CONFIG)
    manager = RetrieverManager()

    with pytest.raises(ValueError, match="RAG search strategy 'non_existent_strategy_123' not found"):
        manager.get_retriever(name="non_existent_strategy_123", user_id="test", embedding_strategy_name="mock_emb") # type: ignore

    # デフォルトへのフォールバックのテスト（default_strategy_nameが有効な場合）
    caplog.clear()
    # default_strategy_name が "basic_search" に設定されているので、それが使われるはず
    manager.default_strategy_name = "basic_search" # 明示的に設定
    retriever = manager.get_retriever(name="non_existent_again", user_id="test", embedding_strategy_name="mock_emb") # type: ignore
    assert "RAG search strategy 'non_existent_again' not implemented/registered." in caplog.text # 警告が出る
    assert "Falling back to default RAG search strategy: basic_search" in caplog.text
    assert retriever is not None # BasicRetrieverのモックが返る


def test_retriever_manager_config_file_not_found(monkeypatch, caplog):
    def mock_load_raises_error():
        raise StrategyConfigError("Retriever Config file not found for test")
    monkeypatch.setattr("core.retriever_manager.load_strategy_config", mock_load_raises_error)

    manager = RetrieverManager()
    assert "Failed to load RAG search strategy configuration: Retriever Config file not found for test" in caplog.text
    assert not manager.get_available_strategies()
    assert manager.default_strategy_name is None
