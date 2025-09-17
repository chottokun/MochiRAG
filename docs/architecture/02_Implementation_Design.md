# MochiRAG 実装設計書

(...既存のセクションは省略...)

## 2. LLMプロバイダーの管理 (LLM Provider Management)

MochiRAGは、複数の大規模言語モデル（LLM）プロバイダーをサポートするように設計されています。
これにより、特定のユースケースや要件に応じて、最適なLLMを柔軟に選択できます。中心的な役割を担うのが
`core/llm_manager.py` に実装されている `LLMManager` シングルトンです。

### 2.1. サポートするプロバイダー

現在、以下のプロバイダーがサポートされています。

- **Ollama**: ローカル環境でオープンソースモデルを実行するためのフレームワーク。
- **OpenAI**: `gpt-4o` などの高性能なモデルを提供。
- **Azure OpenAI**: Microsoft Azure上でOpenAIのモデルをセキュアに利用するためのサービス。
- **Google Gemini**: Googleの次世代モデルファミリー。

### 2.2. 設定方法

使用するLLMは `config/strategies.yaml` ファイルの `llms` セクションで設定します。各プロバイダーごとに必要なパラメータが異なります。

**設定例:**
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

  # Google Gemini
  gemini-1.5-pro:
    provider: gemini
    model_name: "gemini-1.5-pro-latest"
    api_key: "YOUR_GOOGLE_API_KEY"
```

### 2.3. `LLMManager` の役割

`LLMManager` は、設定ファイルに基づいて適切なLangChainの `Chat` モデル（`ChatOllama`,
`ChatOpenAI`, `AzureChatOpenAI`, `ChatGoogleGenerativeAI`）を動的にインスタンス化します。一度インスタンス化されたモデルはキャッシュされ、アプリケーション全体で再利用されるため、効率的なリソース管理が実現されます。

このアーキテクチャにより、新しいLLMプロバイダーの追加も容易になっています。`LLMManager` に新しいプロバイダーのロジックを追加し、設定モデルを更新するだけで、システム全体で新しいLLMが利用可能になります。

## 3. RAG戦略詳解

`RetrieverManager`は、`config/strategies.yaml` の設定に基づき、要求されたRAG戦略を動的に構築します。以下に各戦略の実装詳細と考慮事項を記述します。

---

### 3.1. BasicRetrieverStrategy (基本戦略)

- **概要:** 最も基本的なベクトル検索リトリーバー。ユーザーが選択した複数のデータソース（個人用・共有）を横断して同時に検索する能力を持つ。
- **得意な質問タイプ:** キーワードが明確な質問や、シンプルな事実確認。
- **長所:** 高速、低コスト。複数のデータソースを一度に検索できるため、利便性が高い。
- **短所:** 質問の解釈は行わないため、単語のニュアンスや背景知識を必要とする検索には不向きな場合がある。
- **技術的詳細:** ユーザーが複数のデータソースを選択した場合、内部的にデータソースごとにリトリーバーを生成し、`EnsembleRetriever`を用いてそれらを束ねる。各リトリーバーからの検索結果はReciprocal Rank Fusion (RRF) アルゴリズムによって統合され、最終的なランキングが決定される。

**実装サンプル:**
```python
# 複数のデータソースIDを受け取り、EnsembleRetrieverを構築するロジック
def get_retriever(self, user_id: int, dataset_ids: List[int]) -> BaseRetriever:
    retrievers = []
    # ... 個人用データセットからリトリーバーを生成 ...
    if personal_ids:
        # ...
        retrievers.append(personal_retriever)

    # ... 共有データセットからリトリーバーを生成 ...
    if shared_ids:
        # ...
        retrievers.append(shared_retriever)

    # ... リトリーバーが複数あればEnsembleRetrieverを返す ...
    if len(retrievers) > 1:
        return EnsembleRetriever(retrievers=retrievers, weights=...)

    return retrievers[0]
```

---

### 3.2. HyDE (Hypothetical Document Embeddings) Strategy

(注) 重要: 本リポジトリでは一時的に HyDE に関連する実装を削除しています。HyDE を再導入する場合は、LangChain の当該バージョンで `HypotheticalDocumentEmbedder` が期待する `llm_chain` のスキーマ（`input_schema.model_json_schema()` 等）を満たすことを十分に確認し、Runnable 型互換性に関する回帰テストを用意してください。

HyDE を再検討する際の注意点:
- LangChain の内部 API はバージョン間で変わる可能性があり、HyDE の実装はチェーンの入力スキーマを直接参照する部分があるため、ランタイムのチェーン実装（LLMChain / Runnable 等）と互換性が取れていることを確認する必要があります。
- 安定した再導入を行うには、バージョン固定（pinning）や適切な adapter を追加して互換性を保証するのが安全です。

---

### 3.3. StepBackPromptingStrategy
          question_gen_chain = LLMChain(llm=llm, prompt=step_back_prompt)

          # 2. 検索と結合のロジックをLCELで記述
          # response_chain = question_gen_chain | retriever | ... (さらに続く)
          # 実際の構築にはLCELの知識が必要
          pass # 概念を示すためのサンプル
      ```

    (...既存のMultiQuery, ContextualCompression, ParentDocument戦略の解説は省略...)
