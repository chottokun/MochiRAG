# MochiRAG システムアーキテクチャ
## 1. 概要
## 2. プロジェクトアーキテクチャ
### 2.1. コンポーネント図
# MochiRAG システムアーキテクチャ

## 1. 概要

本ドキュメントは、MochiRAGの技術的アーキテクチャ、主要コンポーネント、処理フロー、および開発上の指針をまとめたものです。MochiRAGは、ユーザーがドキュメントをアップロードし、その内容に基づいてLLMと対話するRAGアプリケーションです（マルチテナント対応）。

## 2. プロジェクトアーキテクチャ

### 2.1. コンポーネント図

```text
+----------------+      +------------------+      +----------------+
|   Frontend     |      |     Backend      |      | External       |
|  (Streamlit)   |      |     (FastAPI)    |      | Services       |
+----------------+      +------------------+      +----------------+
       |                      |                      |
       |---(HTTP API)-------->|                      |
       |                      |---(SQL)------------->|  User DB
       |                      |                      |  (SQLite)
       |                      |                      |
       |                      |---(Core Logic)------>|  Core Services
       |                      |                      |  (RAG, LLM, etc.)
       |                      |                      |
       |                      |<--(LLM API)----------|  Ollama
       |                      |                      |
       |                      |<--(Vector DB)--------|  ChromaDB
       |                      |                      |
+----------------+      +------------------+      +----------------+
```

- Frontend (Streamlit): UIを提供。ユーザーの認証、ファイルアップロード、チャット画面を担当し、BackendのAPIを通じて操作します。
- Backend (FastAPI): APIエンドポイント、認証、DB操作、Coreコンポーネントの仲介を担当します。
- Core Services: RAGロジック（LLM呼び出し、ベクトル検索、戦略の実行など）を封装したモジュール群です。
- External Services: ユーザーデータはSQLiteに格納され、LLMはOllama/OpenAI等、ベクトルデータはChromaDB等を利用します。

### 2.2. ディレクトリ構成（概要）

- `frontend/`: StreamlitベースのUIコード
- `backend/`: FastAPIベースのAPIサーバー（`models.py`, `crud.py`, `main.py` 等）
- `core/`: RAGコアロジック（retriever, llm manager, chain service 等）
# MochiRAG システムアーキテクチャ

## 1. 概要

本ドキュメントは、MochiRAGの技術アーキテクチャ、主要コンポーネント、処理フロー、および開発上の指針をまとめたものです。MochiRAGは、ユーザーがドキュメントをアップロードし、その内容に基づいてLLMと対話するRAGアプリケーションです（マルチテナント対応）。

## 2. プロジェクトアーキテクチャ

### 2.1. コンポーネント図
# MochiRAG システムアーキテクチャ

## 1. 概要

本ドキュメントは、MochiRAGの技術アーキテクチャ、主要コンポーネント、処理フロー、および開発上の指針をまとめたものです。MochiRAGは、ユーザーがドキュメントをアップロードし、その内容に基づいてLLMと対話するRAGアプリケーションです（マルチテナント対応）。

## 2. プロジェクトアーキテクチャ

### 2.1. コンポーネント図

以下は主要コンポーネントと相互作用の概略図です。

```text
+----------------+      +------------------+      +----------------+
|   Frontend     |      |     Backend      |      | External       |
|  (Streamlit)   |      |     (FastAPI)    |      | Services       |
+----------------+      +------------------+      +----------------+
       |                      |                      |
       |---(HTTP API)-------->|                      |
       |                      |---(SQL)------------->|  User DB
       |                      |                      |  (SQLite)
       |                      |                      |
       |                      |---(Core Logic)------>|  Core Services
       |                      |                      |  (RAG, LLM, etc.)
       |                      |                      |
       |                      |<--(LLM API)----------|  Ollama
       |                      |                      |
       |                      |<--(Vector DB)--------|  ChromaDB
       |                      |                      |
+----------------+      +------------------+      +----------------+
```

- Frontend (Streamlit): UIを提供します。ユーザー認証、ファイルアップロード、チャット画面を担当し、BackendのAPIを通じて操作します。
- Backend (FastAPI): APIエンドポイント、認証、DB操作、Coreコンポーネントへの仲介を担当します。
- Core Services: RAGロジック（LLM呼び出し、ベクトル検索、戦略の実行など）を実装するモジュール群です。
- External Services: ユーザーデータはSQLiteに格納され、LLMはOllama/OpenAI等、ベクトルデータはChromaDB等を利用します。

### 2.2. ディレクトリ構成（概要）

- `frontend/`: StreamlitベースのUIコード
- `backend/`: FastAPIベースのAPIサーバー（`models.py`, `crud.py`, `main.py` 等）
- `core/`: RAGコアロジック（retriever, llm manager, chain service 等）
- `config/`: `strategies.yaml` などの設定ファイル
- `docs/`: ドキュメント
- `tests/`: `pytest` を使ったテスト
- `cli.py`: 管理者用コマンドラインユーティリティ

### 2.3. データ管理モデル

MochiRAGは主に2つのベクトルデータベース運用モデルを想定しています。

- パーソナルデータベース: 各ユーザーに紐づくプライベートなコレクション（例: `user_{user_id}`）。メタデータはSQLiteに保存します。
- 共有データベース: 全ユーザーが読み取り可能な共有コレクション（例: `shared_{db_name}`）。`cli.py create-shared-db` で作成・登録し、`shared_dbs.json` に登録情報を保持します。

## 3. 主要ライブラリと選定理由

| ライブラリ | 役割 | 選定理由 |
|---|---|---|
| FastAPI | バックエンドフレームワーク | 高速な非同期処理、Pydanticとの連携、良好な開発体験のため採用。 |
| Streamlit | フロントエンド | PythonのみでリッチなUIを構築できるため、プロトタイプやデモに適する。 |
| LangChain | RAGコアフレームワーク | LLM、チェーン、リトリーバー等の抽象化と統合を容易にするため採用。 |
| ChromaDB | ベクトルストア | ローカル運用やファイルベース運用が容易で、開発時の導入が簡単。 |
| Pydantic | バリデーション | データ検証と設定の一元管理に有用。FastAPIと親和性が高い。 |
| Poetry | 依存管理 | 再現性の高い依存管理と仮想環境の簡易管理のため採用。 |
| SQLAlchemy | ORM | DB操作の抽象化に利用。 |

---

(本ファイルは開発の進行に合わせて更新してください)
