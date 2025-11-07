# ガイド: RAG戦略におけるプロンプトのカスタマイズ

MochiRAGに実装されているRAG戦略の多くは、大規模言語モデル（LLM）への指示、すなわち「プロンプト」によってその振る舞いが制御されています。これらのプロンプトをカスタマイズすることで、システムの検索や回答生成のロジックを、特定のユースケースに合わせて細かく調整することが可能です。

**プロンプトのカスタマイズは、`config/strategies.yaml` ファイルを編集するだけで簡単に行えます。**

---

## 1. カスタマイズの基本手順

1.  プロジェクトのルートディレクトリにある `config/strategies.yaml` ファイルを開きます。
2.  ファイルの末尾にある `prompts:` セクションを探します。
3.  カスタマイズしたいプロンプト（例: `step_back`）のテンプレートを、要件に合わせて編集します。
4.  ファイルを保存し、MochiRAGアプリケーションを再起動すると、変更が反映されます。

---

## 2. カスタマイズ可能なプロンプト詳細

以下に、`prompts` セクションで定義されている各プロンプトの目的と、カスタマイズの指針を示します。

### 2.1. `step_back`

- **目的**: `StepBackPromptingRetrieverStrategy` で使用されます。ユーザーの具体的な質問から、より一般的で抽象的な検索クエリを生成し、直接的な回答が見つからない場合でも関連性の高い上位概念のドキュメントを発見しやすくします。
- **デフォルトテンプレート**:
  ```yaml
  step_back: |
    You are an expert at world knowledge. I am going to ask you a question. Your job is to formulate a single, more general question that captures the essence of the original question. Frame the question from the perspective of a historian or a researcher.
    Original question: {question}
    Step-back question:
  ```
- **カスタマイズ指針**:
  - **ペルソナ（役割）の変更**: `a historian or a researcher` の部分を、`a senior software architect` や `a product manager` など、対象ドメインに合わせた専門家の役割に変更することで、生成される質問の視点を変えることができます。
  - **具体性の指示**: 生成される質問が抽象的すぎると感じる場合は、「generate a slightly broader question that is still grounded in the same technical domain.」のような一文を追加して、具体性を保つよう指示します。

### 2.2. `ace_topic`

- **目的**: `ACERetrieverStrategy`（自己進化戦略）で使用されます。ユーザーの質問から、後で知識を検索・保存するためのキーワード（トピック）を生成します。
- **デフォルトテンプレート**:
  ```yaml
  ace_topic: |
    Based on the following user question, identify the main topic or entity in one or two words.
    Your answer should be concise and suitable for use as a database search key.
    Examples:
    # (省略)
    Original question: {question}
    Topic:
  ```
- **カスタマイズ指針**: トピックの粒度（具体的か、抽象的か）や言語（例：日本語のキーワードを生成させる）を調整することで、知識がどの範囲で共有・再利用されるかを制御できます。

### 2.3. `ace_evolution`

- **目的**: `ACERetrieverStrategy` のバックグラウンド自己進化プロセスで使用されます。対話が完了した後、その内容から将来役立つ普遍的な「知識」や「戦略」を合成します。
- **デフォルトテンプレート**:
  ```yaml
  ace_evolution: |
    You are an expert in synthesizing knowledge. Based on the user's question and the provided answer, formulate a single, concise, and reusable insight...
    # (省略)
    Concise Insight:
  ```
- **カスタマイズ指針**: 生成される知識のスタイル（例：戦略的アドバイス、コードスニペット）や具体性（例：初心者向けに平易に）を制御できます。ペルソナを明確に指定するのも有効です。

---

## 3. LangChainデフォルトプロンプトを利用する戦略のカスタマイズ

`MultiQueryRetrieverStrategy` や `ContextualCompressionRetrieverStrategy` は、デフォルトではLangChainライブラリに組み込まれたプロンプトを利用しており、`strategies.yaml` の `prompts` セクションには定義がありません。

これらのプロンプトをカスタマイズしたい場合は、以下の手順で設定を追加できます。

1.  `core/retriever_manager.py` の該当する戦略クラス（例: `MultiQueryRetrieverStrategy`）を修正し、`config_manager.get_prompt` を呼び出すようにリファクタリングします（`StepBackPromptingRetrieverStrategy` の実装が参考になります）。
2.  `config/strategies.yaml` の `prompts` セクションに、新しいプロンプト名（例: `multi_query`）でテンプレートを追加します。

この方法により、MochiRAGのすべてのプロンプト駆動型RAG戦略を、YAMLファイルを編集するだけで、安全かつ柔軟にカスタマイズすることが可能です。
