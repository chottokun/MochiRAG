# MochiRAG 実装設計書

（...既存のセクションは省略...）

## 2. LLMプロバイダーの管理 (LLM Provider Management)

MochiRAGは、複数の大規模言語モデル（LLM）プロバイダーをサポートするように設計されています。これにより、特定のユースケースや要件に応じて、最適なLLMを柔軟に選択できます。中心的な役割を担うのが `core/llm_manager.py` に実装されている `LLMManager` シングルトンです。

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
    # MochiRAG 実装設計書

    (...既存のセクションは省略...)

    ## 2. LLMプロバイダーの管理 (LLM Provider Management)

    MochiRAGは、複数の大規模言語モデル（LLM）プロバイダーをサポートするように設計されています。これにより、特定のユースケースや要件に応じて、最適なLLMを柔軟に選択できます。中心的な役割を担うのが `core/llm_manager.py` に実装されている `LLMManager` シングルトンです。

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

    `LLMManager` は、設定ファイルに基づいて適切なLangChainの `Chat` モデル（`ChatOllama`, `ChatOpenAI`, `AzureChatOpenAI`, `ChatGoogleGenerativeAI`）を動的にインスタンス化します。一度インスタンス化されたモデルはキャッシュされ、アプリケーション全体で再利用されるため、効率的なリソース管理が実現されます。

    このアーキテクチャにより、新しいLLMプロバイダーの追加も容易になっています。`LLMManager` に新しいプロバイダーのロジックを追加し、設定モデルを更新するだけで、システム全体で新しいLLMが利用可能になります。

    ## 3. RAG戦略詳解

    `RetrieverManager`は、`config/strategies.yaml` の設定に基づき、要求されたRAG戦略を動的に構築します。以下に各戦略の実装詳細と考慮事項を記述します。

    --- 

    ### 3.1. BasicRetrieverStrategy (基本戦略)

    - **概要:** 最も基本的なベクトル検索リトリーバー。ユーザーが選択した複数のデータソース（個人用・共有）を横断して同時に検索する能力を持つ。
    - **得意な質問タイプ:** キーワードが明確な質問や、シンプルな事実確認。「MochiRAGの認証方式は？」など。
    - **長所:** 高速、低コスト。複数のデータソースを一度に検索できるため、利便性が高い。
    - **短所:** 質問の解釈は行わないため、単語のニュアンスや背景知識を必要とする検索には不向きな場合がある。
    - **技術的詳細:** ユーザーが複数のデータソースを選択した場合、内部的にデータソースごとにリトリーバーを生成し、`EnsembleRetriever`を用いてそれらを束ねる。各リトリーバーからの検索結果はReciprocal Rank Fusion (RRF) アルゴリズムによってインテリジェントに統合され、最終的なランキングが決定される。
    - **実装サンプル:**
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

    - **概要:** ユーザーの質問から、まずLLMに「仮説的な回答ドキュメント」を生成させ、その仮説ドキュメントのベクトルを使って検索する手法。`BasicRetrieverStrategy`と同様に、個人用・共有データを横断した同時検索に対応している。
    - **得意な質問タイプ:** 質問が短すぎる、あるいは抽象的で、検索のヒントが少ない場合。「RAGとは？」のような質問でも、LLMが生成する詳細な仮説ドキュメントによって、関連性の高い文書を見つけやすくなります。
    - **長所:** 検索の再現率（Recall）が向上することがある。
    - **短所:** LLMコールが追加で必要。LLMの生成する仮説が的外れだと、逆にノイズを拾う可能性がある。
    - **技術的詳細:** この戦略は、データソースごとに`HypotheticalDocumentEmbedder`を持つリトリーバーを生成します。共有されているVectorStoreインスタンスの状態を変更する（副作用）のを避けるため、リトリーバーを生成するたびに、HyDE用の埋め込み関数を持つ一時的な`Chroma`インスタンスを内部で作成しています。これにより、安全な並列検索を実現しています。
    - **実装サンプル:**
      ```python
      # HyDE用のリトリーバーを安全に生成する内部ヘルパーメソッド
      def create_hyde_retriever(vector_store: Chroma, filter_dict: dict) -> BaseRetriever:
          # ...
          hyde_embeddings = HypotheticalDocumentEmbedder(...)

          # 一時的なVectorStoreインスタンスを生成して副作用を回避
          temp_vector_store = Chroma(
              client=vector_store._client,
              collection_name=vector_store._collection.name,
              embedding_function=hyde_embeddings,
          )
          return temp_vector_store.as_retriever(...)
      ```

    --- 

    ### 3.3. StepBackPromptingStrategy

    - **概要:** 複雑な質問に対し、まず一歩引いた抽象的な質問をLLMに生成させ、その質問で大まかなコンテキストを検索。その後、元の具体的な質問で詳細情報を検索する二段階の検索手法。
    - **得意な質問タイプ:** 「`SelfQueryRetriever`は`ParentDocumentRetriever`と比較してどのような利点があるか？」のような、比較や深い分析を求める複雑な質問。
    - **長所:** 複数の視点から情報を集めるため、より網羅的で深い回答を生成できる。
    - **短所:** 処理が複雑になり、LLMコールと検索が複数回行われるため、コストとレイテンシが大幅に増加する。
    - **実装サンプル:**
      ```python
      # この戦略は複数のチェーンとリトリーバーを組み合わせるため、実装はより複雑になる
      def get_retriever(self, llm: BaseLanguageModel, retriever: BaseRetriever) -> BaseRetriever:
          # 1. Step-back質問を生成するプロンプトとチェーン
          step_back_prompt = PromptTemplate(template=..., input_variables=["question"])
          question_gen_chain = LLMChain(llm=llm, prompt=step_back_prompt)

          # 2. 検索と結合のロジックをLCELで記述
          # response_chain = question_gen_chain | retriever | ... (さらに続く)
          # 実際の構築にはLCELの知識が必要
          pass # 概念を示すためのサンプル
      ```

    (...既存のMultiQuery, ContextualCompression, ParentDocument戦略の解説は省略...)
