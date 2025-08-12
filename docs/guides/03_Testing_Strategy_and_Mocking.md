# テスト戦略とモック活用ガイド

## 1. テストの基本方針

MochiRAGでは、`pytest` をテストフレームワークとして採用しています。テストの主な目的は、各コンポーネントが単体で正しく動作することを保証する**ユニットテスト**を充実させることです。

特に、`core`ロジック層のテストでは、LLMやベクトルデータベースなどの外部サービスへの依存を完全に排除し、ロジックの正当性のみを検証することに重点を置きます。これにより、高速で安定したテストスイートを維持します。

## 2. モッキング戦略

本プロジェクトのテスト容易性は、**Managerパターン**と**モック（Mock）**の活用によって実現されています。

### 2.1. Managerパターンによる依存関係の分離

`LLMManager`, `VectorStoreManager`, `RetrieverManager` などのManagerクラスは、外部ライブラリやサービスへのアクセスを一手に引き受ける窓口です。テスト対象のモジュールは、これらのManagerを介してのみ外部と対話します。

この設計により、テスト時にはManagerの挙動を偽のオブジェクト（モック）に置き換えるだけで、外部依存を簡単に切り離すことができます。

### 2.2. `pytest`フィクスチャによるManagerのモッキング

テスト全体で繰り返し利用するモックは、`pytest`の**フィクスチャ**として定義するのが効果的です。`conftest.py` や各テストファイル内で、`monkeypatch` を利用してグローバルなManagerインスタンスをモックに差し替えます。

**例: `LLMManager`をモックするフィクスチャ**
```python
# tests/core/test_retriever_manager.py より

import pytest
from unittest.mock import MagicMock
from core.llm_manager import LLMManager
from langchain_core.language_models import BaseLanguageModel

@pytest.fixture
def mock_llm_manager(monkeypatch):
    # 1. Managerクラスのモックを作成
    mock_manager = MagicMock(spec=LLMManager)

    # 2. Managerのメソッドの返り値を定義
    #    get_llm()が呼ばれたら、BaseLanguageModelのモックを返すように設定
    mock_manager.get_llm.return_value = MagicMock(spec=BaseLanguageModel)

    # 3. monkeypatchを使い、テスト対象モジュール内のManagerをモックに差し替え
    monkeypatch.setattr("core.retriever_manager.llm_manager", mock_manager)

    return mock_manager
```

### 2.3. `@patch`デコレータによる外部ライブラリのモッキング

`LangChain`などの外部ライブラリのクラスやメソッドを直接モックしたい場合は、`unittest.mock.patch`デコレータを使用します。これにより、実際のネットワークコールや重い計算処理を防ぎます。

**例: `MultiQueryRetriever`のクラスメソッドをモック**
```python
# tests/core/test_retriever_manager.py より
from unittest.mock import patch

@patch("langchain.retrievers.MultiQueryRetriever.from_llm")
def test_get_multi_query_retriever(mock_from_llm, ...): # patchの対象が引数に追加される
    # ...
```

## 3. 具体的なテスト例: `RetrieverManager`のテスト

`RetrieverManager`が`multi_query`戦略のリトリーバーを正しく構築できるかをテストする例を見てみましょう。

```python
# tests/core/test_retriever_manager.py より（簡略化）
from unittest.mock import patch, MagicMock

# `MultiQueryRetriever.from_llm`が呼ばれることをテストするため、patchでモック化
@patch("langchain.retrievers.MultiQueryRetriever.from_llm")
def test_get_multi_query_retriever(
    mock_from_llm,          # @patch からのモックオブジェクト
    mock_load_config_retriever, # 設定ファイルをモックするフィクスチャ
    mock_llm_manager,       # LLMManagerをモックするフィクスチャ
    mock_chroma_retriever   # ChromaDBをモックするフィクスチャ
):
    # --- Arrange (準備) ---
    # モックの返り値を設定
    mock_ret_instance = MagicMock(spec=BaseRetriever)
    mock_from_llm.return_value = mock_ret_instance
    mock_load_config_retriever(SAMPLE_RAG_SEARCH_CONFIG) # ダミーの設定をロード

    # --- Act (実行) ---
    manager = RetrieverManager()
    retriever = manager.get_retriever(
        user_id="test_user",
        embedding_strategy_name="mock_emb",
        name="mq_search_config_name"
    )

    # --- Assert (検証) ---
    # 1. 期待した返り値か？
    assert retriever == mock_ret_instance

    # 2. 期待通りにモックが呼ばれたか？
    # LLMManagerのget_llm()が1回呼ばれたはず
    mock_llm_manager.get_llm.assert_called_once()
    # MultiQueryRetriever.from_llm()が1回呼ばれたはず
    mock_from_llm.assert_called_once()
```

## 4. 既知の問題と今後の改善点

### 4.1. スキップされているテスト

現在、`tests/core/test_retriever_manager.py` 内の一部のテストが、タイムアウトやアサーションの失敗により、`@pytest.mark.skip`で無効化されています。

- `test_retriever_manager_load_valid_config`
- `test_get_contextual_compression_retriever`
- `test_get_parent_document_retriever`
- `test_retriever_manager_get_non_existent_strategy_fallback`

これらのテストは、モックのセットアップが複雑であることや、インスタンスの同一性に関するアサーションが難しいことに起因している可能性があります。今後のタスクとして、これらのテストを安定してパスできるように修正することが望まれます。

### 4.2. アーキテクチャの改善提案

現在のテスト戦略から、よりテスト容易性を高めるための改善点がいくつか見られます。

- **改善案1: 依存性注入 (Dependency Injection) の導入**
  - **現状:** 各モジュールがグローバルなManagerインスタンスを直接インポートしています。
  - **提案:** クラスのコンストラクタ（`__init__`）でManagerインスタンスを受け取るように変更します（依存性注入）。これにより、テスト時に`monkeypatch`でグローバル変数を書き換える必要がなくなり、よりクリーンで明示的なテストが書けるようになります。

- **改善案2: `VectorStoreManager`の責務の明確化**
  - **現状:** `BasicRetrieverStrategy`が`langchain_chroma.Chroma`を直接インスタンス化しており、テストのためにChromaクラス自体をモックする必要があります。
  - **提案:** `VectorStoreManager`に`get_vector_store_as_retriever()`のようなメソッドを追加し、リトリーバーの取得ロジックをManager内にカプセル化します。これにより、StrategyクラスはChromaDBの実装詳細を知る必要がなくなり、モックも`VectorStoreManager`一つで済むようになります。
