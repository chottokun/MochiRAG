# API: `retriever_manager.py`

このモジュールは、様々なRAG検索戦略（リトリーバー）を管理し、要求に応じて適切なリトリーバーを動的に構築する責務を持ちます。

## `RetrieverManager` クラス

`RetrieverManager` は、**Strategyパターン** を利用して、複数の検索アルゴリズムを柔軟に切り替えるための中心的な役割を担います。

### 設計思想

- **動的ローディング**: `RetrieverManager` は、初期化時に `config/strategies.yaml` ファイルを読み込み、`retrievers` セクションに登録されているすべての戦略を自動的に検出します。
- **Strategyパターン**: `strategy_class` の設定に基づき、対応する戦略クラス（例: `BasicRetrieverStrategy`）を動的にインポートしてインスタンス化します。これにより、新しい戦略を追加する際に `RetrieverManager` のコードを修正する必要がなくなります。
- **ファクトリ**: `get_retriever` メソッドは、指定された戦略名に基づいて、対応する戦略オブジェクトの `get_retriever` メソッドを呼び出し、最終的なLangChainの `BaseRetriever` インスタンスを生成して返します。

### `get_retriever(self, strategy_name: str, user_id: int, dataset_ids: Optional[List[int]] = None) -> BaseRetriever`

指定された戦略名に対応するリトリーバーを構築して返します。

- **パラメータ:**
  - `strategy_name` (str): `config/strategies.yaml` で定義されたリトリーバーのキー名（例: `basic`, `multiquery`）。
  - `user_id` (int): 現在のユーザーのID。
  - `dataset_ids` (Optional[List[int]]): 検索対象とするデータセットIDのリスト。

- **戻り値:**
  - `langchain_core.retrievers.BaseRetriever`: LangChainのインターフェースに準拠したリトリーバーのインスタンス。このリトリーバーは、`rag_chain_service` によってRAGパイプラインに組み込まれます。

- **例外:**
  - `ValueError`: 指定された `strategy_name` が設定ファイルに存在しない場合に送出されます。

---

## `RetrieverStrategy` 抽象クラス

すべての具体的なリトリーバー戦略が継承すべき抽象ベースクラス（ABC）です。

### `get_retriever(self, user_id: int, dataset_ids: Optional[List[int]] = None) -> BaseRetriever`

この抽象メソッドは、各具象戦略クラスでオーバーライドされる必要があります。リトリーバーを構築するための具体的なロジックは、このメソッド内に実装されます。

---

## 実装されている戦略クラス

本モジュールには、`RetrieverStrategy` を継承した以下の具体的な戦略クラスが実装されています。

- `BasicRetrieverStrategy`
- `MultiQueryRetrieverStrategy`
- `ContextualCompressionRetrieverStrategy`
- `ParentDocumentRetrieverStrategy`
- `StepBackPromptingRetrieverStrategy`

**各戦略のアルゴリズムや詳細な動作については、[`docs/architecture/implementation_design.md`](../architecture/implementation_design.md) を参照してください。** このドキュメントは、各戦略がどのような問題を解決し、どのように機能するかを詳細に解説しています。
