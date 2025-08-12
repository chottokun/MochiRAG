# MochiRAG 実装設計書

## 1. 全体アーキテクチャ

MochiRAGは、3つの主要なコンポーネントから構成される分離型アーキテクチャを採用しています。

- **Frontend (UI層):** `Streamlit` を使用して構築された、ユーザーが操作するWebインターフェースです。ユーザー認証、データセット管理、チャット画面などの機能を提供します。バックエンドとはHTTP経由で通信します。
- **Backend (API層):** `FastAPI` を使用して構築された、システムのビジネスロジックを担うAPIサーバーです。認証処理、データベース操作、そしてコアロジックの呼び出しを担当します。
- **Core (コアロジック層):** RAG機能の中核をなす純粋なPythonモジュール群です。`LangChain` を利用して、ドキュメント処理、ベクトル化、検索、回答生成などの複雑な処理を抽象化・実行します。

```mermaid
graph TD
    subgraph User Interface
        Frontend[Streamlit UI]
    end

    subgraph Application Server
        Backend[FastAPI Backend]
    end

    subgraph Core Logic
        Core[RAG Core Engine]
    end

    subgraph External Services
        LLM[LLM (Ollama, etc.)]
        VectorDB[Vector DB (ChromaDB)]
    end

    Frontend -- HTTP API Calls --> Backend
    Backend -- Python Calls --> Core
    Core -- Manages --> LLM
    Core -- Manages --> VectorDB
```

## 2. コアロジック設計: Managerパターン

コアロジック層は、システムの関心事を分離し、コンポーネントの交換を容易にするため、**Managerパターン** を中心に設計されています。主要な外部サービスや設定は、それぞれ専用のManagerクラスを介してアクセスされます。

- **`LLMManager`**:
  - **責務:** 大規模言語モデル（LLM）のインスタンス化と管理。
  - **詳細:** `config/strategies.yaml` の設定に基づき、`Ollama` や将来的に `AzureOpenAI` などのLLMプロバイダーを切り替えて、言語モデルのインスタンスを提供します。

- **`EmbeddingManager`**:
  - **責務:** テキストのベクトル化に使用する埋め込みモデルの管理。
  - **詳細:** `Sentence Transformers` などのライブラリから、設定ファイルで指定されたモデルをロードし、テキストをベクトルに変換する機能を提供します。

- **`VectorStoreManager`**:
  - **責務:** ベクトルデータベース（ChromaDB）への接続と操作の管理。
  - **詳細:** ドキュメントのチャンクとベクトルを永続化し、データベースへの追加や検索のインターフェースを統一します。

- **`RetrieverManager`**:
  - **責務:** RAGにおける検索部分を担当する「リトリーバー」オブジェクトの生成と管理。
  - **詳細:** 本システムのRAG機能の柔軟性を実現する最も重要なコンポーネントです。**Strategyパターン** を採用しており、ユーザーが選択したRAG戦略（`basic`, `multi_query`など）に応じて、適切な設定が施されたリトリーバーを動的に構築します。

## 3. RAG戦略詳解

`RetrieverManager` は、以下のRAG戦略をサポートします。これらの戦略は `core/retriever_manager.py` 内で個別のクラスとして実装されています。

- **`BasicRetrieverStrategy`**:
  - **仕組み:** 最も基本的なベクトル検索。ユーザーの質問をベクトル化し、ベクトルデータベース内でコサイン類似度が最も高いドキュメントチャンクを検索します。
  - **特徴:** 高速でシンプル。他の高度な戦略の基礎となります。

- **`MultiQueryRetrieverStrategy`**:
  - **仕組み:** ユーザーの質問を一度LLMに渡し、異なる視点からの複数の類似クエリを生成させます。これらのクエリすべてで検索を実行し、結果を統合します。
  - **特徴:** ユーザーの意図が曖昧な場合や、単一のクエリではヒットしにくい情報を見つけ出すのに有効です。

- **`ContextualCompressionRetrieverStrategy`**:
  - **仕組み:** まず`BasicRetriever`で関連する可能性のあるドキュメントを少し多めに取得します。その後、取得した各ドキュメントの内容と元の質問をLLMに渡し、「本当に質問と関連する部分だけ」を抽出（圧縮）させます。
  - **特徴:** 最終的にLLMに渡すコンテキストのノイズを削減し、より精度の高い回答を生成するのに役立ちます。

- **`ParentDocumentRetrieverStrategy`**:
  - **仕組み:** 小さなチャンク（子ドキュメント）で検索を行いますが、最終的にユーザーに返すのはそのチャンクが含まれる大きな元のドキュメント（親ドキュメント）です。
  - **特徴:** 回答の文脈が失われがちなRAGの課題を解決し、より人間が理解しやすい形で情報を提供します。
  - **実装上の注意:** ソースコードのコメントにある通り、本戦略が最大限効果を発揮するには、ドキュメントの登録（インジェスト）プロセスがこの戦略専用の形式で行われる必要があります。現状の実装では、その効果が限定的である可能性があります。

- **`DeepRagRetrieverStrategy`**:
  - **仕組み:** 複雑な質問に対して、LLMを使って複数の単純なサブクエリに分解します。各サブクエリで検索を実行し、得られた結果を統合して最終的なコンテキストを構築します。
  - **特徴:** 「AとBを比較し、Cについて要約せよ」のような複合的な質問に対して、より網羅的で正確な回答を生成する能力があります。

## 4. データフロー (RAGクエリの例)

ユーザーが質問を送信してから回答を受け取るまでの、典型的なデータフローは以下の通りです。

1.  **Frontend:** ユーザーがStreamlitのUIで質問を入力し、送信ボタンをクリックします。
2.  **API Call:** UIは、FastAPIバックエンドの `/chat/query/` エンドポイントにHTTP POSTリクエストを送信します。リクエストボディには、質問内容、選択されたRAG戦略、データセットID、そしてヘッダーには認証用JWTが含まれます。
3.  **Backend (FastAPI):**
    - エンドポイントはリクエストを受け取り、JWTを検証して `user_id` を特定します。
    - `core.rag_chain.get_rag_response` 関数を、受け取ったパラメータと共に呼び出します。
4.  **Core Logic (`rag_chain.py`):**
    - `get_rag_response` は `retriever_manager.get_retriever` を呼び出し、`user_id`、戦略、データセットIDなどを渡して、このクエリ専用のリトリーバーの構築を依頼します。
5.  **Retriever Construction (`retriever_manager.py`):**
    - `RetrieverManager` は、指定された戦略名に対応する戦略クラス（例: `MultiQueryRetrieverStrategy`）を選択します。
    - 戦略クラスの `get_retriever` メソッドが実行され、`VectorStoreManager` や `LLMManager` から必要なコンポーネントを取得し、最終的なLangChainリトリーバーオブジェクトを構築して返します。この際、後述するマルチテナント用のフィルターが設定されます。
6.  **LCEL Chain Execution (`rag_chain.py`):**
    - 構築されたリトリーバーは、LangChain Expression Language (LCEL) で定義されたチェーンに組み込まれます。
    - チェーンはまずリトリーバーを `invoke` して、質問に関連するドキュメントチャンクを取得します。
    - 取得したチャンクは、プロンプトテンプレートに埋め込むための単一のコンテキスト文字列に整形されます。
    - 整形されたコンテキストと元の質問が、プロンプトテンプレートを介してLLMに渡されます。
    - LLMが最終的な回答を生成します。
7.  **Response:** 生成された回答と、根拠となったソースドキュメントの情報が、呼び出し元を遡ってFastAPIバックエンドに返され、最終的にJSON形式でStreamlitフロントエンドに送信されます。
8.  **Frontend:** UIがレスポンスを受け取り、回答と出典情報を画面に表示します。

## 5. マルチテナント設計

本システムのデータ分離は、主にベクトルデータベースのクエリレベルで実現されています。これにより、各ユーザーは自身のデータにしかアクセスできません。

この中核を担うのが、`BasicRetrieverStrategy` 内でのフィルター構築ロジックです。他の多くの戦略も内部でこの基本戦略を利用しているため、このフィルター機能は広範囲に適用されます。

```python
# core/retriever_manager.py (BasicRetrieverStrategy.get_retrieverより抜粋)

# ...
filter_conditions = [{"user_id": user_id}]
if data_source_ids:
    # ... data_source_id の条件を追加
if dataset_ids:
    # ... dataset_id の条件を追加

final_filter = {"$and": filter_conditions}
# ...

search_kwargs = {"k": n_results, "filter": final_filter}

return vectorstore.as_retriever(search_kwargs=search_kwargs)
```

- **仕組み:** リトリーバーを構築する際、`search_kwargs` に `filter` オプションが追加されます。
- **フィルター内容:** このフィルターには、認証されたユーザーの `user_id` が必須条件として含まれます。さらに、ユーザーがUIで特定のデータセットを選択した場合は、その `dataset_id` も条件に追加されます。
- **適用:** この `search_kwargs` はChromaDBの `as_retriever` メソッドに渡されます。これにより、このリトリーバーが行うすべてのベクトル検索は、自動的に指定されたユーザーIDとデータセットIDの範囲に限定されます。
