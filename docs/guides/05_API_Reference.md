# MochiRAG APIリファレンス

このドキュメントは、MochiRAGバックエンドAPIの主要なエンドポイントについて説明します。

**ベースURL:** `http://localhost:8000`

## 1. 認証

### `POST /token`

ユーザーを認証し、アクセストークンを取得します。

- **リクエスト形式:** `application/x-www-form-urlencoded`
- **リクエストボディ:**
  - `username`: ユーザーのメールアドレス
  - `password`: パスワード
- **レスポンス (成功時):**
  ```json
  {
    "access_token": "your_jwt_token_string",
    "token_type": "bearer"
  }
  ```

## 2. ユーザー

### `POST /users/`

新規ユーザーを登録します。

- **リクエストボディ:**
  ```json
  {
    "email": "user@example.com",
    "password": "your_strong_password"
  }
  ```

### `GET /users/me`

現在認証されているユーザーの情報を取得します。

- **認証:** 要Bearerトークン

## 3. データセット

- `POST /users/me/datasets/`: 新規データセットを作成します。
  - **認証:** 要Bearerトークン
  - **リクエストボディ:** `{"name": "My Dataset", "description": "Optional description"}`
- `GET /users/me/datasets/`: 認証ユーザーのデータセット一覧を取得します。
  - **認証:** 要Bearerトークン
- `GET /users/me/datasets/{dataset_id}/`: 特定のデータセットの詳細を取得します。
  - **認証:** 要Bearerトークン
- `DELETE /users/me/datasets/{dataset_id}/`: データセットを削除します。
  - **認証:** 要Bearerトークン

## 4. ドキュメント

- `POST /users/me/datasets/{dataset_id}/documents/upload/`: 指定したデータセットにファイルをアップロードします。
  - **認証:** 要Bearerトークン
  - **リクエスト形式:** `multipart/form-data`
  - **リクエストボディ:** ファイルデータ
- `GET /users/me/datasets/{dataset_id}/documents/`: データセット内のファイル一覧を取得します。
  - **認証:** 要Bearerトークン
- `DELETE /users/me/datasets/{dataset_id}/documents/{data_source_id}/`: データセットからファイルを削除します。
  - **認証:** 要Bearerトークン

## 5. チャット

### `POST /chat/query/`

RAGチャット機能のメインエンドポイント。質問を送信し、回答を取得します。

- **認証:** 要Bearerトークン
- **リクエストボディ (JSON):**
  ```json
  {
    "query": "MochiRAGのアーキテクチャについて教えてください。",
    "strategy": "basic",
    "dataset_ids": ["f8f276a6-a3e1-45db-a438-e63cf26d9c49"],
    "data_source_ids": null
  }
  ```
  - `strategy`: (任意) 使用するRAG戦略名。例: `basic`, `multi_query`。デフォルトは設定ファイルの値。
  - `dataset_ids`: (任意) 検索対象とするデータセットIDのリスト。
  - `data_source_ids`: (任意) 検索対象を特定のドキュメントに限定する場合のIDリスト。

- **レスポンス (成功時):**
  ```json
  {
    "answer": "MochiRAGは、フロントエンド、バックエンド、コアロジックの3層からなるアーキテクチャを採用しています...",
    "sources": [
      {
        "page_content": "MochiRAGは、3つの主要なコンポーネントから構成される...",
        "metadata": {
          "source": "docs/MochiRAG_design.md",
          "page": 0,
          "data_source_id": "some_id",
          "original_filename": "MochiRAG_design.md"
        }
      }
    ]
  }
  ```

## 6. 自動生成APIドキュメント

FastAPIによって、完全でインタラクティブなAPIドキュメントが自動的に生成されます。APIサーバーの起動後、以下のURLにアクセスしてください。

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`
