# MochiRAG APIリファレンス

## データセット (Datasets)

### データセット一覧の取得
  - **説明:** ユーザーが利用可能なデータセットの一覧を取得します。これには、ユーザー自身のパーソナルデータベースと、システム全体で利用可能なすべての共有データベースが含まれます。
  - **成功レスポンス (200 OK):**

### データセットの作成
  - **説明:** 新しいパーソナルデータセットを作成します。
  - **リクエストボディ:**
  - **成功レスポンス (201 Created):** 作成されたデータセットのオブジェクト。

### データセットの削除
  - **説明:** 指定したIDのパーソナルデータセットを削除します。
  - **成功レスポンス (200 OK):** 削除されたデータセットのオブジェクト。

（...既存のセクションは省略...）

## 7. エラーレスポンス

APIは、エラー発生時に標準化されたエラーレスポンスを返します。レスポンスボディには、エラーの詳細情報が含まれます。


## MochiRAG APIリファレンス

## データセット (Datasets)

### データセット一覧の取得
- **`GET /users/me/datasets/`**
  - **説明:** ユーザーが利用可能なデータセットの一覧を取得します。これには、ユーザー自身のパーソナルデータベースと、システム全体で利用可能なすべての共有データベースが含まれます。
  - **成功レスポンス (200 OK):**
    ```json
    [
      {
        "name": "My Personal DB",
        "description": "A personal database.",
        "id": 1,
        "owner_id": 1,
        "data_sources": []
      },
      {
        "name": "Common Test DB",
        "description": "A shared database for testing purposes.",
        "id": -1,
        "owner_id": -1,
        "data_sources": []
      }
    ]
    # MochiRAG APIリファレンス

    ## データセット (Datasets)
    "detail": "ここにエラーメッセージが入ります"
  }
  ```

- **主要なHTTPステータスコード:**
  - **`400 Bad Request`**: リクエストの形式が不正な場合（例: 必須パラメータの欠如、不正なJSON形式）。
    - `{"detail": "Query field is required"}`
  - **`401 Unauthorized`**: 認証トークンが無効または提供されていない場合。
    - `{"detail": "Not authenticated"}`
  - **`403 Forbidden`**: 認証はされているが、要求されたリソースへのアクセス権がない場合。
    - `{"detail": "User does not have access to this dataset"}`
  - **`404 Not Found`**: 要求されたリソース（例: `dataset_id`）が存在しない場合。
    - `{"detail": "Dataset not found"}`
  - **`422 Unprocessable Entity`**: リクエストのセマンティクスは正しいが、何らかのバリデーションエラーが発生した場合（例: サポート外のファイル形式）。
    - `{"detail": "File type .docx is not supported"}`
  - **`500 Internal Server Error`**: サーバー内部で予期せぬエラーが発生した場合。
    - `{"detail": "An unexpected error occurred with the LLM service"}`
  - **`503 Service Unavailable`**: 外部サービス（LLMやデータベース）が一時的に利用不可能な場合。
    - `{"detail": "The LLM service is currently unavailable. Please try again later."}`