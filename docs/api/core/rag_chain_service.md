# API: `rag_chain_service.py`

このモジュールは、RAG（Retrieval-Augmented Generation）の応答生成パイプライン（チェーン）を構築し、実行する責務を持ちます。

## `RAGChainService` クラス

RAG応答を生成するための主要なサービスです。

### `get_rag_response(self, query: QueryRequest, user_id: int) -> QueryResponse`

ユーザーからのクエリを受け取り、RAGパイプラインを実行して回答とソースドキュメントを返します。

- **パラメータ:**
  - `query` (`schemas.QueryRequest`): ユーザーからのリクエスト。以下の情報を含みます。
    - `query` (str): ユーザーの質問文。
    - `strategy` (str): 使用するRAG戦略の名前 (例: "basic", "multiquery")。
    - `dataset_ids` (List[int]): 検索対象のデータセットIDのリスト。
    - `history` (Optional[List[schemas.HistoryItem]]): チャットの履歴。
  - `user_id` (int): 現在のユーザーのID。

- **戻り値:**
  - `schemas.QueryResponse`: 生成された応答。以下の情報を含みます。
    - `answer` (str): LLMによって生成された回答文。
    - `sources` (List[`schemas.Source`]): 回答の生成に使用されたソースドキュメントのリスト。各ソースには、ドキュメントの本文 (`page_content`) とメタデータ (`metadata`) が含まれます。

- **処理の概要:**
  1.  指定された `strategy` に基づいて、`retriever_manager` から適切なリトリーバーを取得します。
  2.  チャット履歴の有無に応じて、通常応答用または会話形式用のプロンプトを選択します。
  3.  LangChain Expression Language (LCEL) を用いて、以下の処理を行うチェーンを構築します。
      a. 質問文を使ってリトリーバーで関連ドキュメントを取得する。
      b. 取得したドキュメントを `format_docs_with_sources` で整形し、プロンプトに注入可能な形式にする。
      c. 整形されたコンテキスト、質問文、チャット履歴をプロンプトに渡し、`llm_manager` を通じてLLMに送信する。
      d. LLMからの回答 (`answer`) と、使用したソースドキュメント (`sources`) を並行して取得し、`QueryResponse`として返す。
  4.  `deeprag` 戦略が指定された場合は、専用の `DeepRAGStrategy` クラスに処理を委譲します。

---

## `format_docs_with_sources` 関数

`format_docs_with_sources(docs: List[Document]) -> str`

リトリーバーによって取得された `Document` オブジェクトのリストを、LLMのプロンプトにコンテキストとして埋め込むための単一の文字列に整形します。

- **パラメータ:**
  - `docs` (List[`langchain_core.documents.Document`]): リトリーバーが返したドキュメントのリスト。

- **戻り値:**
  - `str`: 整形されたコンテキスト文字列。各ドキュメントは、その出典情報（ファイル名、ページ番号など）と共にフォーマットされ、`---` で区切られます。

- **出力例:**
  ```
  Source (ID: 123, Original: project_overview.pdf), Page: 2
  Content: MochiRAG is a system designed for multi-tenant RAG applications.

  ---

  Source (ID: 124, Original: project_overview.pdf), Page: 5
  Content: It uses a combination of FastAPI, Streamlit, and LangChain.
  ```
