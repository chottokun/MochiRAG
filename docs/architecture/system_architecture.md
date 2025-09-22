# MochiRAG システムアーキテクチャ

## 1. 概要

本ドキュメントは、RAG (Retrieval-Augmented Generation) アプリケーションである **MochiRAG** の技術的アーキテクチャ、主要コンポーネント、処理フロー、および開発上の指針を定義します。

MochiRAGは、ユーザーが自身のドキュメントをアップロードし、その内容に基づいて大規模言語モデル（LLM）と対話形式で質疑応答を行う、マルチテナント対応のWebアプリケーションです。

## 2. プロジェクトアーキテクチャ

### 2.1. コンポーネント図

MochiRAGは、以下の主要コンポーネントで構成されています。

```mermaid
graph TD
    subgraph User Interface
        A[Frontend (Streamlit)]
    end

    subgraph Application Server
        B[Backend (FastAPI)]
    end

    subgraph Core Logic
        C[Core Services]
    end

    subgraph External Services
        D[User DB (SQLite)]
        E[Vector DB (ChromaDB)]
        F[LLM Service (Ollama, etc.)]
    end

    A -- HTTP API --> B
    B -- CRUD --> D
    B -- Orchestrates --> C
    C -- Retrieves --> E
    C -- Generates --> F
```

- **Frontend (Streamlit)**: ユーザーインターフェースを提供します。ユーザー認証、ファイルアップロード、チャット画面の構築を担当し、BackendのAPIを通じてすべての操作を行います。
- **Backend (FastAPI)**: HTTP APIエンドポイントを提供します。認証（OAuth2）、データベース操作（SQLAlchemy）、およびCoreコンポーネントの呼び出しと連携を担当するアプリケーションサーバーです。
- **Core Services**: RAGのコアロジックを実装するモジュール群です。LLMの管理、ベクトル検索、RAG戦略の実行といった、アプリケーションの頭脳となる処理を担います。
- **External Services**: 外部の永続化・推論サービスです。
    - **User DB (SQLite)**: ユーザー情報、データセット、ドキュメントのメタデータを格納します。
    - **Vector DB (ChromaDB)**: ドキュメントから抽出されたベクトルデータを格納し、高速なセマンティック検索を可能にします。
    - **LLM Service**: プロンプトに基づいてテキストを生成する大規模言語モデルです。

### 2.2. ディレクトリ構成

プロジェクトの主要なディレクトリとその役割は以下の通りです。

- `frontend/`: StreamlitをベースとしたUIコンポーネントのコードを格納します。
- `backend/`: FastAPIをベースとしたAPIサーバーのコードを格納します。`main.py` にエンドポイント定義、`crud.py` にDB操作、`models.py` にデータモデルが定義されています。
- `core/`: RAGのコアロジックを格納します。
    - `rag_chain_service.py`: RAGチェーンの構築と実行を担当します。
    - `retriever_manager.py`: RAG戦略に応じたリトリーバーを管理します。
    - `llm_manager.py`: LLMのインスタンスを管理します。
    - `ingestion_service.py`: ドキュメントの取り込みとベクトル化を担当します。
- `config/`: `strategies.yaml` など、RAG戦略や設定を記述するファイルを格納します。
- `docs/`: 本書を含むすべてのドキュメントを格納します。
- `tests/`: `pytest` を用いた単体テストおよび結合テストのコードを格納します。
- `cli.py`: データベースの作成など、管理者向けのコマンドラインユーティリティです。

### 2.3. データ管理モデル

MochiRAGは、主に2種類のベクトルデータベース運用モデルをサポートします。

- **パーソナルデータベース**: 各ユーザーに紐づくプライベートなコレクション（例: `user_{user_id}`）。ユーザーがアップロードしたドキュメントが格納されます。
- **共有データベース**: 全ユーザーが読み取り可能な共有コレクション（例: `shared_{db_name}`）。管理者が `cli.py create-shared-db` コマンドで作成し、`shared_dbs.json` に登録情報を保持します。

## 3. 主要な処理フロー

### 3.1. ドキュメントの取り込み (Ingestion)

1.  ユーザーがFrontendからドキュメントファイルをアップロードします。
2.  Backendがファイルを受け取り、一意なIDを付与して `ingestion_service` に渡します。
3.  `ingestion_service` はドキュメントをチャンクに分割し、`embedding_manager` を使って各チャンクをベクトル化します。
4.  ベクトル化されたデータは、対応するVector DBコレクションに保存されます。

### 3.2. 質疑応答 (RAG)

1.  ユーザーがFrontendのチャット画面で質問を入力します。
2.  Backendは `/chat/query` エンドポイントでリクエストを受け取ります。
3.  `rag_chain_service` が呼び出され、指定されたRAG戦略に基づいて `retriever_manager` から適切なリトリーバーを取得します。
4.  リトリーバーがVector DBにクエリを投げ、質問に関連性の高いドキュメントチャンク（コンテキスト）を取得します。
5.  取得したコンテキストと元の質問を組み合わせ、`llm_manager` を通じてLLMにプロンプトを送信します。
6.  LLMが生成した回答と、参照したソースドキュメントがFrontendに返却され、ユーザーに表示されます。

## 4. 主要ライブラリと選定理由

| ライブラリ | 役割 | 選定理由 |
|---|---|---|
| FastAPI | バックエンドフレームワーク | 高速な非同期処理、Pydanticとの強力な連携による型安全性、自動APIドキュメント生成機能により、堅牢で開発効率の高いAPIサーバーを構築できるため。 |
| Streamlit | フロントエンドフレームワーク | Pythonのみで迅速にインタラクティブなUIを構築できるため、プロトタイピングやデータアプリケーションに適している。 |
| LangChain | RAGコアフレームワーク | LLM、リトリーバー、プロンプトといったRAGの構成要素を抽象化し、パイプライン（チェーン）として容易に組み合わせることを可能にするため。 |
| ChromaDB | ベクトルストア | ローカル環境でのセットアップが容易で、ファイルベースでの運用も可能なため、開発時の導入ハードルが低い。 |
| SQLAlchemy | ORM | Pythonオブジェクトとリレーショナルデータベースのテーブルをマッピングし、SQLを直接記述することなく安全なDB操作を実現するため。 |
| Pydantic | データ検証 | FastAPIと連携し、リクエスト/レスポンスのデータ型を定義・検証することで、コードの信頼性を向上させるため。 |
| Poetry | 依存関係管理 | プロジェクトの依存関係を`pyproject.toml`で明確に管理し、再現性の高い開発環境を保証するため。 |

---

*(本ドキュメントは、開発の進捗やアーキテクチャの変更に応じて継続的に更新してください)*
