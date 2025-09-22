# API: `vector_store_manager.py`

このモジュールは、ベクトルデータベース（ChromaDB）への接続と操作を抽象化し、一元管理する責務を持ちます。

## `VectorStoreManager` クラス

`VectorStoreManager` は、シングルトンとして実装され、ChromaDBクライアントのライフサイクルを管理し、アプリケーションの他の部分がベクトルデータベースを容易に利用するためのインターフェースを提供します。

### 設計思想

- **ファサード (Facade)**: ChromaDBのクライアント接続や設定の詳細をカプセル化し、`get_vector_store` や `add_documents` といった単純なメソッドを提供することで、他のサービス（`IngestionService`や`RetrieverManager`など）がベクトルストアの具体的な実装を意識することなく利用できるようにします。
- **設定駆動 (Configuration-Driven)**: `config/strategies.yaml` の `vector_store` セクションの設定に基づき、ChromaDBへの接続モード（`http`または`persistent`）を決定します。

### `initialize_client(self)`

ChromaDBクライアントを初期化します。このメソッドは、アプリケーションの起動時に一度だけ呼び出される必要があります（例: FastAPIの`lifespan`イベント内）。

- **処理の概要:**
  1.  `config_manager` からベクトルストアの設定を読み込みます。
  2.  `mode` が `http` の場合、指定された `host` と `port` を使用して `chromadb.HttpClient` を作成します。
  3.  `mode` が `persistent` の場合、指定された `path` を使用して `chromadb.PersistentClient` を作成し、ローカルファイルとしてデータを永続化します。
  4.  初期化されたクライアントは、`self.client` に格納され、以降の操作で再利用されます。

### `get_vector_store(self, collection_name: str) -> Chroma`

指定されたコレクション名に対応する、LangChain互換のベクトルストアオブジェクトを取得します。

- **パラメータ:**
  - `collection_name` (str): 操作対象のChromaDBコレクションの名前 (例: `user_123`, `shared_main_db`)。

- **戻り値:**
  - `langchain_chroma.Chroma`: 指定されたコレクションに紐づいたLangChainのベクトルストアオブジェクト。このオブジェクトは、リトリーバーの構築 (`as_retriever()`) やドキュメントの追加 (`add_documents()`) などに使用されます。

### `add_documents(self, collection_name: str, documents: List[Document])`

指定されたコレクションにドキュメントのリストを追加します。

- **パラメータ:**
  - `collection_name` (str): ドキュメントを追加するコレクションの名前。
  - `documents` (List[`Document`]): LangChainの`Document`オブジェクトのリスト。

- **処理の概要:**
  1.  `get_vector_store` を呼び出して、対象コレクションの `Chroma` オブジェクトを取得します。
  2.  `Chroma` オブジェクトの `add_documents` メソッドを呼び出して、ドキュメントをベクトル化し、データベースに保存します。

---

**注意:** `initialize_client` が呼び出される前に他のメソッドを呼び出すと、例外が発生します。必ずアプリケーションの起動シーケンスに初期化処理を組み込んでください。
