# MochiRAG APIリファレンス

（...既存のセクションは省略...）

## 7. エラーレスポンス

APIは、エラー発生時に標準化されたエラーレスポンスを返します。レスポンスボディには、エラーの詳細情報が含まれます。

- **エラーレスポンス形式:**
  ```json
  {
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