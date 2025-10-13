# MochiRAG 実装設計書

## 1. 概要

本ドキュメントは、MochiRAGのコアコンポーネントである `LLMManager` と `RetrieverManager` の内部実装、およびそれらが提供する各種戦略について詳細に解説します。

## 2. LLMプロバイダーの管理 (`LLMManager`)

`core/llm_manager.py` に実装されている `LLMManager` は、複数の大規模言語モデル（LLM）プロバイダーを効率的に管理するためのシングルトンクラスです。

### 2.1. 役割と設計

- **シングルトンパターン**: アプリケーション全体で唯一のインスタンスを保証し、LLMクライアントの生成を一度だけに限定することで、リソースを効率的に使用します。
- **動的インスタンス化**: `config/strategies.yaml` の `llms` セクションの設定に基づき、要求されたLLMプロバイダー (`Ollama`, `OpenAI`, `Azure OpenAI`, `Gemini`) に対応するLangChainの `Chat` モデルを動的にインスタンス化します。
- **キャッシュ**: 一度生成されたLLMクライアントは内部でキャッシュされ、後続の呼び出しでは再利用されます。

### 2.2. 設定例

`config/strategies.yaml` で以下のようにLLMを定義します。

```yaml
llms:
  # Ollama
  gemma3:4b-it-qat:
    provider: ollama
    model_name: "gemma3:4b-it-qat"
    base_url: "http://localhost:11434"

  # OpenAI
  gpt-4o:
    provider: openai
    model_name: "gpt-4o"
    api_key: "YOUR_OPENAI_API_KEY"

  # Azure OpenAI
  azure-gpt-4:
    provider: azure
    deployment_name: "YOUR_DEPLOYMENT_NAME"
    azure_endpoint: "YOUR_AZURE_ENDPOINT"
    api_version: "2024-02-01"
    api_key: "YOUR_AZURE_API_KEY"
```

## 3. RAG検索戦略の管理 (`RetrieverManager`)

`core/retriever_manager.py` に実装されている `RetrieverManager` は、`config/strategies.yaml` の設定に基づき、様々なRAG検索戦略を動的に構築・提供します。以下に、現在実装されている各戦略の詳細を解説します。

---

### 3.1. `BasicRetrieverStrategy` (基本戦略)

- **概要**: 最も基本的なベクトル検索戦略。ユーザーが選択した複数のデータソース（個人用・共有）を横断して同時に検索します。
- **動作原理**:
    1.  選択されたデータセットIDに基づき、個人用 (`user_{user_id}`) と共有用 (`shared_{db_name}`) の各ChromaDBコレクションに対応するリトリーバーを生成します。
    2.  生成されたリトリーバーが複数ある場合、LangChainの `EnsembleRetriever` を用いてそれらを束ねます。
    3.  `EnsembleRetriever` は、各リトリーバーからの検索結果をReciprocal Rank Fusion (RRF) アルゴリズムで統合し、最も関連性が高いと判断されたドキュメントを最終的な検索結果として返します。
- **長所**: 高速かつシンプル。複数のナレッジベースを一度に検索できるため利便性が高いです。
- **短所**: 質問文をそのままベクトル化するため、単語の表面的な意味に頼りがちで、複雑な意図を汲み取った検索は困難です。

---

### 3.2. `MultiQueryRetrieverStrategy` (複数クエリ生成戦略)

- **概要**: ユーザーの質問をLLMが分析し、異なる角度からの複数の検索クエリを自動生成して検索を実行する戦略です。
- **動作原理**:
    1.  `BasicRetrieverStrategy` で構築されたリトリーバーを内部に持ちます。
    2.  ユーザーの質問を受け取ると、LLMがその質問を解釈し、類似の質問や異なる視点からの質問を3〜5個生成します (例: 「RAGとは？」→「Retrieval-Augmented Generationの仕組みは？」、「RAGの利点は？」)。
    3.  生成されたすべてのクエリで並行して検索を実行し、得られた結果を統合して返します。
- **長所**: 検索網羅性が向上し、ユーザーが曖昧な質問をしても関連ドキュメントを見つけやすくなります。
- **短所**: LLMの呼び出しが増えるため、応答速度がやや低下し、コストが増加します。

---

### 3.3. `ContextualCompressionRetrieverStrategy` (コンテキスト圧縮戦略)

- **概要**: 一旦取得したドキュメントの内容をLLMが要約・抽出し、質問に直接関連する部分だけを最終的なコンテキストとして利用する戦略です。
- **動作原理**:
    1.  `BasicRetrieverStrategy` でドキュメントを検索します。
    2.  取得した各ドキュメントの内容とユーザーの質問をLLMに渡し、「この質問に答えるために必要な情報だけを抽出せよ」と指示します。
    3.  LLMによって圧縮・抽出された情報のみを後段のRAGプロンプトに渡します。
- **長所**: プロンプトに含めるコンテキストのノイズが減少し、LLMがより正確な回答を生成しやすくなります。
- **短所**: 検索後に再度LLMを呼び出すため、応答速度が低下し、コストが増加します。

---

### 3.4. `ParentDocumentRetrieverStrategy` (親子ドキュメント戦略)

- **概要**: ドキュメントを大きな「親チャンク」と、検索対象となる小さな「子チャンク」に分割して管理する戦略です。検索は子チャンクで行い、回答生成には親チャンクの完全なコンテキストを利用します。
- **動作原理**:
    1.  **Ingestion時**: ドキュメントを例えば400文字の「子チャンク」と、2000文字の「親チャンク」に分割します。子チャンクのベクトルのみをVector DBに保存し、親チャンクの全文はSQLデータベース (`SQLDocStore`) に保存します。
    2.  **検索時**: ユーザーの質問で子チャンクを検索します。
    3.  **回答生成時**: 見つかった子チャンクが属する親チャンクの全文をSQLデータベースから取得し、LLMに渡します。
- **長所**: 検索の精度（小さいチャンク）と、回答生成時の文脈の豊富さ（大きいチャンク）を両立できます。
- **短所**: Ingestion時の処理が複雑になり、SQLデータベースという追加の依存コンポーネントが必要になります。

---

### 3.5. `StepBackPromptingRetrieverStrategy` (一歩下がった質問戦略)

- **概要**: ユーザーの具体的な質問から一歩引いた、より一般的・抽象的な質問をLLMに生成させ、その質問で検索を行う戦略です。
- **動作原理**:
    1.  ユーザーの質問 (例: 「MochiRAGの`BasicRetrieverStrategy`はEnsembleRetrieverを使っていますか？」) を受け取ります。
    2.  LLMがその質問の背景にある、より一般的な概念についての質問を生成します (例: 「RAGにおけるEnsembleRetrieverの役割とは？」)。
    3.  この生成された一般的な質問を使って `BasicRetrieverStrategy` で検索を実行します。
- **長所**: 元の質問が具体的すぎたり、ニッチすぎて直接的な回答が見つからない場合に、関連性の高い上位概念のドキュメントを検索できる可能性が高まります。
- **短所**: LLMによる質問生成の品質に依存し、意図しない検索結果になる可能性もあります。

---

### 3.6. `HyDE` (Hypothetical Document Embeddings) Strategy

**注意**: HyDE戦略に関連する実装は、LangChainのバージョン間の互換性問題により、現在コードベースから**削除されています**。

HyDEは、質問に対する「架空の回答」をまずLLMに生成させ、その架空の回答をベクトル化して検索に利用する高度な戦略です。将来的に再導入を検討する際は、LangChainのバージョン互換性を慎重に検証し、十分なテストを行う必要があります。
