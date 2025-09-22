# API: `ingestion_service.py`

このモジュールは、アップロードされたドキュメントファイルの読み込み、分割（チャンキング）、そしてベクトル化してデータベースに保存する「取り込み（Ingestion）」プロセス全体を管理します。

## `IngestionService` クラス

ドキュメント取り込み処理のロジックをカプセル化したシングルトンサービスです。

### `__init__(self)`

サービスの初期化時に、設定ファイル (`config/strategies.yaml`) からチャンクサイズやオーバーラップの値を読み込み、`RecursiveCharacterTextSplitter` のインスタンスを複数準備します。これにより、各RAG戦略に応じた最適なチャンキングが可能になります。

### `ingest_file(self, file_path: str, file_type: str, data_source_id: int, dataset_id: int, user_id: int, strategy: str = "basic")`

ユーザーがアップロードした単一のファイルを処理する主要なメソッド。RAG戦略に応じて、異なる内部メソッドに処理を振り分けます。

- **パラメータ:**
  - `file_path` (str): サーバー上に一時的に保存されたファイルのパス。
  - `file_type` (str): ファイルのMIMEタイプ (例: `application/pdf`)。
  - `data_source_id` (int): このファイルを表すデータベース上のID。
  - `dataset_id` (int): このファイルが属するデータセットのID。
  - `user_id` (int): ファイルの所有者であるユーザーのID。
  - `strategy` (str): 適用する取り込み戦略。現在は `basic` または `parent_document`。

### `ingest_documents_for_shared_db(self, file_paths: List[str], collection_name: str, dataset_id: int)`

管理者用CLI (`cli.py`) から呼び出されることを想定したメソッド。特定のユーザーに紐付かない「共有データベース」を作成するために、ディレクトリ内の複数のファイルを一括で取り込みます。

- **パラメータ:**
  - `file_paths` (List[str]): 取り込むファイルのパスのリスト。
  - `collection_name` (str): 保存先のChromaDBコレクション名。
  - `dataset_id` (int): この共有データセットに割り当てられたID（通常は負の数）。

### 内部メソッドと処理フロー

#### `_ingest_basic(...)` (基本戦略)

1.  `_get_loader` を使って、ファイルタイプに応じたLangChainのドキュメントローダー（`PyPDFLoader`など）を取得します。
2.  ドキュメントをロードし、`self.text_splitter` を使ってチャンクに分割します。
3.  各チャンクに、検索時のフィルタリングに利用するためのメタデータ（`user_id`, `dataset_id`など）を付与します。
4.  `vector_store_manager.add_documents` を呼び出して、チャンクをベクトル化し、ChromaDBに保存します。
    - **耐障害性**: `add_documents` 呼び出し時にはリトライ機構（指数バックオフ付き）が働き、埋め込みモデルサービスの一時的な障害に対応します。

#### `_ingest_for_parent_document(...)` (親子ドキュメント戦略)

1.  ドキュメントをロードし、`self.parent_splitter` で大きな親チャンクに分割します。
2.  各親チャンクに対して、`crud.create_parent_document` を呼び出して、その内容をSQLデータベースに保存します。このとき、各親チャンクに一意な `parent_id` (UUID) を付与します。
3.  さらに、各親チャンクを `self.child_splitter` で小さな子チャンクに分割します。
4.  子チャンクに、対応する `parent_id` を含むメタデータを付与します。
5.  すべての子チャンクを `vector_store_manager.add_documents` を使ってChromaDBに保存します。

---

## `EmbeddingServiceError` 例外

埋め込みモデルを提供する外部サービス（例: Ollama）への接続が失敗した場合など、リトライしても復旧不可能な場合に送出されるカスタム例外です。API層は、この例外を捕捉してHTTP 503 Service Unavailableエラーをクライアントに返すことができます。
