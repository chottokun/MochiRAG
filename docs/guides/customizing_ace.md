# ガイド: ACE戦略におけるプロンプトのカスタマイズ

MochiRAGに実装されている `ACERetrieverStrategy` (ACE戦略) は、2つの主要な大規模言語モデル（LLM）プロンプトによって駆動されています。これらのプロンプトをカスタマイズすることで、システムの自己進化の振る舞いを特定のユースケースに合わせて調整することが可能です。

プロンプトは、`core/retriever_manager.py` の `ACERetrieverStrategy` クラスと、`core/context_evolution_service.py` の `ContextEvolutionService` クラス内にハードコードされています。将来的なアップデートで、これらのプロンプトをYAML設定ファイルから読み込むように拡張することも考えられます。

---

## 1. トピック生成プロンプト

このプロンプトは、ACE戦略における2つの重要な役割を担います。

1.  **コンテキスト検索 (Retrieval):** ユーザーの質問から、データベース (`EvolvedContext` テーブル) を検索するためのキーワード（トピック）を生成します。
2.  **コンテキスト保存 (Storage):** 対話から新しい知識を抽出した後、その知識をどのトピックでデータベースに保存するかを決定します。

### 目的

このプロンプトの目的は、ユーザーの質問の「主題」を、一貫性のある、再利用可能なキーワードとして抽出することです。良いトピックを生成することで、類似の質問が来た際に、過去に得られた知見を効率的に再利用できます。

### デフォルトのプロンプトテンプレート

**場所:** `core/retriever_manager.py` 内、`ACERetrieverStrategy` クラス

```python
template = """Based on the following user question, identify the main topic or entity in one or two words.
Your answer should be concise and suitable for use as a database search key.
Examples:
- Question: "How does the ParentDocumentRetriever work in MochiRAG?" -> Answer: "ParentDocumentRetriever"
- Question: "Tell me about ensemble retrievers" -> Answer: "EnsembleRetriever"
- Question: "What are the key features?" -> Answer: "Features"

Original question: {question}
Topic:"""
```

### カスタマイズの指針と例

- **粒度の調整:**
    - より**広範なトピック**で知識を共有させたい場合（例：「リトリーバー全般」）、プロンプトをより抽象的な概念を抽出するように変更します。
      ```
      "Identify the general software engineering concept from this question."
      ```
    - より**具体的なトピック**で知識を分けたい場合（例：「ParentDocumentRetrieverの設定方法」）、より詳細なエンティティを抽出するように指示します。
      ```
      "Extract the specific function name or class name from this question."
      ```
- **言語の変更:**
    - LLMが日本語の質問に対して、より安定して日本語のトピックを生成するように、プロンプト全体を日本語で記述し、例も日本語にすることが有効です。
      ```
      以下のユーザーの質問から、主要なトピックやエンティティを1〜2単語で特定してください。
      あなたの回答は、データベースの検索キーとして使用するのに適した、簡潔なものであるべきです。
      ...
      ```

---

## 2. 知識抽出（自己進化）プロンプト

このプロンプトは、ACE戦略の自己進化プロセスの核心です。対話が完了した後、その内容から将来的に役立つ普遍的な「知識」や「戦略」を合成する役割を担います。

### 目的

このプロンプトの目的は、一回の具体的な質疑応答から、他の類似した質問にも応用可能な、汎用的な知見を抽出することです。ここで生成されたコンテキストの品質が、システムの自己改善能力を直接決定します。

### デフォルトのプロンプトテンプレート

**場所:** `core/context_evolution_service.py` 内、`ContextEvolutionService` クラス

```python
evolution_template = """You are an expert in synthesizing knowledge. Based on the user's question and the provided answer, formulate a single, concise, and reusable insight. This insight should be a piece of general knowledge that could help answer similar questions more effectively in the future.

Do not repeat the question or the answer. Focus on extracting the core principle or strategy.

User Question:
"{question}"

Provided Answer:
"{answer}"

Concise Insight:"""
```

### カスタマイズの指針と例

- **生成される知識のスタイルの変更:**
    - より**戦略的なアドバイス**を生成させたい場合、「As a senior architect, provide a strategic recommendation based on this interaction.」のように、ペルソナ（役割）を明確に指示します。
    - **コードスニペット**を含む知識を生成させたい場合、「If applicable, include a short, relevant code snippet in your insight.」のように、具体的なアウトプット形式を指示します。
- **知識の具体性の制御:**
    - より**初心者向け**の平易な知識を生成させたい場合、「Explain the core concept in a way a beginner could understand.」のように、対象読者を指定します。
- **フォーマットの指定:**
    - 生成される知識を常に特定のフォーマット（例：マークダウンの箇条書き）にしたい場合、その旨を明確に指示します。
      ```
      "Provide the insight as a markdown bullet point list."
      ```

これらのプロンプトを適切に調整することで、MochiRAGの自己進化機能を、あなたの組織やチームが持つ知識の性質や、ユーザーの要求に、より最適化させることが可能です。
