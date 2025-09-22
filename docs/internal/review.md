# 実装に関する技術レビュー: ChromaDBクライアント・サーバーモード

## 1. はじめに

このドキュメントは、MochiRAGアプリケーションにChromaDBのクライアント・サーバーモード機能を実装した際の技術的な詳細についてレビューしたものです。この変更により、アプリケーションは従来のローカルファイルベースのベクトルストア運用（`persistent`モード）に加え、独立したChromaDBサーバーに接続してベクトルストアを利用する（`http`モード）ことが可能になりました。

この機能は、複数人での利用や、より大規模なデータセットを扱うスケーラブルな本番環境への展開を目的としています。

## 2. 主要な変更点

この機能実装のために、以下のファイルが変更されました。

| ファイル名                         | 変更内容の概要                                                                                             |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `config/strategies.yaml`           | ChromaDBの接続モード（`persistent`または`http`）と、サーバーの接続情報（ホスト、ポート）を設定するセクションを追加。 |
| `core/config_manager.py`           | 新しい設定を読み込み、Pydanticモデルで検証するロジックを追加。                                               |
| `core/vector_store_manager.py`     | 設定に応じて、`PersistentClient`と`HttpClient`のいずれかのChromaDBクライアントを動的に初期化するロジックに変更。 |
| `README.md`                        | 新機能の利用方法（Dockerでのサーバー起動、設定方法）に関するドキュメントを追加。                               |

## 3. 技術的な詳細

### 3.1. 設定の拡張 (`config/strategies.yaml` & `core/config_manager.py`)

柔軟な運用モードの切り替えを実現するため、設定ファイルによる管理が中核となっています。

#### `config/strategies.yaml`
ファイルに`vector_store`セクションが新設されました。

```yaml
vector_store:
  provider: chromadb
  mode: persistent # 'persistent' または 'http' を選択可能
  host: localhost
  port: 8000
  path: "chroma_db"
```

-   `mode`: `persistent`（デフォルト）に設定すると従来通りのローカルファイルモードで動作し、`http`に設定するとクライアント・サーバーモードで動作します。
-   `host`, `port`: `http`モード時に接続するChromaDBサーバーのアドレスを指定します。
-   `path`: `persistent`モード時にデータベースファイルを保存するローカルディレクトリを指定します。

#### `core/config_manager.py`
この設定ファイルを安全に読み込むため、以下の変更が加えられました。

1.  **`VectorStoreConfig` Pydanticモデルの定義**:
    `vector_store`セクションの構造とデータ型を定義・検証するためのPydanticモデルです。これにより、設定ファイル内のキーの有無、型の不一致、不正な値などをアプリケーション起動時に検知できます（Fail-Fast）。

    ```python
    class VectorStoreConfig(BaseModel):
        provider: str
        mode: str
        host: Optional[str] = None
        port: Optional[int] = None
        path: Optional[str] = None
    ```

2.  **`AppConfig`への追加**:
    メインの設定モデルである`AppConfig`に`vector_store: VectorStoreConfig`フィールドが追加され、アプリケーション全体の設定構造に組み込まれました。

3.  **`get_vector_store_config()`メソッド**:
    `ConfigManager`クラスに、検証済みの`VectorStoreConfig`オブジェクトを取得するための新しいメソッドが追加されました。これにより、他のモジュールから型安全に設定値へアクセスできます。

### 3.2. ChromaDBクライアントの動的初期化 (`core/vector_store_manager.py`)

アプリケーションのベクトルストア管理を担当する`VectorStoreManager`が、設定に応じて適切なChromaDBクライアントを生成するように変更されました。

#### `initialize_client()`関数の変更
この関数のロジックが、設定主導型（Configuration-Driven）のアプローチにリファクタリングされました。

```python
def initialize_client(self):
    # ...
    config = config_manager.get_vector_store_config()

    if config.mode == 'http':
        self.client = chromadb.HttpClient(host=config.host, port=config.port)
    elif config.mode == 'persistent':
        self.client = chromadb.PersistentClient(path=config.path)
    else:
        raise ValueError(f"Unsupported ChromaDB mode: {config.mode}")
    # ...
```

この分岐ロジックにより、`config.mode`の値に基づいて、`chromadb`ライブラリが提供する2つの異なるクライアントクラスがインスタンス化されます。

-   **`chromadb.PersistentClient`**: 従来から利用されていたクラス。指定されたローカルパスにSQLiteとParquetファイルを使用してデータを永続化します。アプリケーションプロセス内に常駐します。
-   **`chromadb.HttpClient`**: 新しく利用されるようになったクラス。指定された`host`と`port`で稼働しているChromaDBサーバーに対して、HTTPリクエストを介してすべての操作（コレクションの作成、ドキュメントの追加・検索など）を行います。

### 3.3. 処理フロー

ユーザーがアプリケーションを起動した際の、設定読み込みからクライアント初期化までの処理フローは以下の通りです。

1.  **FastAPIアプリケーション起動**:
    `uvicorn`で`backend/main.py`が実行されます。

2.  **`lifespan`イベントのトリガー**:
    FastAPIの`lifespan`コンテキストマネージャがアプリケーションの起動イベントを捉え、`vector_store_manager.initialize_client()`を呼び出します。

3.  **設定の読み込み**:
    `vector_store_manager`内の`initialize_client`関数が、シングルトンインスタンスである`config_manager`を呼び出し、`get_vector_store_config()`メソッドを通じて`vector_store`設定を取得します。

4.  **クライアントの条件分岐**:
    取得した設定の`mode`値（`'http'`または`'persistent'`）に基づいて、適切なChromaDBクライアント（`HttpClient`または`PersistentClient`）がインスタンス化され、`vector_store_manager.client`に格納されます。

5.  **初期化完了**:
    以降、アプリケーション内のすべてのベクトルストア操作（ドキュメントのインジェスト、RAGチェーンでの検索など）は、この初期化されたクライアントを通じて実行されます。

### 3.4. 関連するライブラリ

この実装には、以下の主要なライブラリが関連しています。

-   **`fastapi`**: アプリケーションのWebフレームワーク。特に、起動時に一度だけ実行したい処理（今回のクライアント初期化など）を定義するための`lifespan`イベント機能が重要です。
-   **`chromadb`**: ベクトルストアのコアライブラリ。`PersistentClient`と`HttpClient`という2つの主要なクライアントインターフェースを提供します。
-   **`pydantic`**: 設定ファイルの構造をPythonクラスとして定義し、強力な型チェックとバリデーション機能を提供するライブラリ。設定ミスの早期発見に貢献します。
-   **`pyyaml`**: `config/strategies.yaml`ファイルをPythonの辞書オブジェクトにパースするために内部的に利用されます。

## 4. 利用方法

新しいクライアント・サーバーモードを利用する手順は`README.md`にも記載されていますが、ここでも要約します。

1.  **ChromaDBサーバーの起動**:
    Dockerを利用するのが最も簡単です。以下のコマンドでサーバーを起動します。
    ```bash
    docker run -p 8000:8000 chromadb/chroma
    ```

2.  **MochiRAGの設定変更**:
    `config/strategies.yaml`ファイルを開き、`vector_store`セクションを以下のように変更します。
    ```yaml
    vector_store:
      mode: http
      host: localhost # Dockerホストのアドレス
      port: 8000
    ```

3.  **MochiRAGバックエンドの再起動**:
    設定変更後、FastAPIサーバーを再起動します。起動ログに「Connecting to ChromaDB server at...」というメッセージが表示されれば、正しくクライアント・サーバーモードで接続されています。

## 5. 結論

この実装により、MochiRAGは単一のローカルアプリケーションから、より堅牢でスケーラブルなシステムへと進化するための重要な基盤を整えました。設定によって透過的に動作モードを切り替えられる設計は、開発の容易さと本番環境での柔軟性を両立させています。

後方互換性も維持されているため、既存ユーザーは設定を変更しない限り、従来通りの環境でアプリケーションを引き続き利用できます。
