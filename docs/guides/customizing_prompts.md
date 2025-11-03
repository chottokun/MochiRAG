# ガイド: RAG戦略におけるプロンプトのカスタマイズ

MochiRAGに実装されているRAG戦略の多くは、大規模言語モデル（LLM）への指示、すなわち「プロンプト」によってその振る舞いが制御されています。これらのプロンプトをカスタマイズすることで、システムの検索や回答生成のロジックを、特定のユースケースに合わせて細かく調整することが可能です。

---

## 1. カスタマイズ可能なプロンプト一覧

現在、以下のRAG戦略で利用されているプロンプトがカスタマイズの対象となります。

| 戦略名 | プロンプトの目的 | 実装場所 |
|---|---|---|
| `ACERetrieverStrategy` | ①トピック生成、②知識抽出 | `core/retriever_manager.py`, `core/context_evolution_service.py` |
| `StepBackPromptingRetrieverStrategy` | より一般的・抽象的な質問を生成 | `core/retriever_manager.py` |
| `MultiQueryRetrieverStrategy` | 複数視点からの質問を生成 | (LangChainのデフォルト) |
| `ContextualCompressionRetrieverStrategy`| 取得ドキュメントを要約・圧縮 | (LangChainのデフォルト) |

---

## 2. `ACERetrieverStrategy` (自己進化戦略)

ACE戦略は、2つの主要なプロンプトによって駆動されています。

### 2.1. トピック生成プロンプト

- **目的**: ユーザーの質問から、データベース検索と保存のためのキーワード（トピック）を生成します。
- **場所**: `core/retriever_manager.py` 内、`ACERetrieverStrategy` クラス
- **デフォルトテンプレート**:
  ```python
  template = """Based on the following user question, identify the main topic or entity in one or two words..."""
  ```
- **カスタマイズ指針**: トピックの粒度（具体的か、抽象的か）や言語を調整することで、知識がどの範囲で共有・再利用されるかを制御できます。詳細は[ACE戦略の解説](#)をご参照ください（*訳注: リンク先は後で更新*）。

### 2.2. 知識抽出（自己進化）プロンプト

- **目的**: 対話完了後、その内容から将来役立つ普遍的な「知識」を合成します。
- **場所**: `core/context_evolution_service.py` 内、`ContextEvolutionService` クラス
- **デフォルトテンプレート**:
  ```python
  evolution_template = """You are an expert in synthesizing knowledge. Based on the user's question and the provided answer, formulate a single, concise, and reusable insight..."""
  ```
- **カスタマイズ指針**: 生成される知識のスタイル（例：戦略的アドバイス、コードスニペット）や具体性を制御できます。

---

## 3. `StepBackPromptingRetrieverStrategy` (一歩下がった質問戦略)

- **目的**: ユーザーの具体的な質問から、より一般的で抽象的な検索クエリを生成し、直接的な回答が見つからない場合でも関連性の高い上位概念のドキュメントを発見しやすくします。
- **場所**: `core/retriever_manager.py` 内、`StepBackPromptingRetrieverStrategy` クラス
- **デフォルトテンプレート**:
  ```python
  template = """You are an expert at world knowledge. I am going to ask you a question. Your job is to formulate a single, more general question that captures the essence of the original question. Frame the question from the perspective of a historian or a researcher.
  Original question: {question}
  Step-back question:"""
  ```
- **カスタマイズ指針**:
  - **ペルソナの変更**: `a historian or a researcher` の部分を、`a senior software architect` や `a product manager` など、対象ドメインに合わせた専門家の役割に変更することで、生成される質問の視点を変えることができます。
  - **具体性の指示**: 生成される質問が抽象的すぎると感じる場合は、「generate a slightly broader question that is still grounded in the same technical domain.」のように、具体性を保つような指示を追加します。

---

## 4. LangChainデフォルトプロンプトを利用する戦略

`MultiQueryRetrieverStrategy` と `ContextualCompressionRetrieverStrategy` は、LangChainライブラリに組み込まれた、汎用性の高いデフォルトプロンプトを利用しています。

- **`MultiQueryRetrieverStrategy`**: ユーザーの質問を複数の異なる角度から捉え直した質問リストを生成します。
- **`ContextualCompressionRetrieverStrategy`**: 取得したドキュメントの中から、質問に直接関連する箇所だけを抽出・要約します。

### カスタマイズ方法

これらの戦略のプロンプトをカスタマイズするには、`core/retriever_manager.py` 内の該当する戦略クラスの実装を修正する必要があります。

LangChainが提供する `from_llm` のようなファクトリメソッドの代わりに、`PromptTemplate` を自分で定義し、それを `LLMChain` と組み合わせてリトリーバーを直接インスタンス化することで、プロンプトを完全に制御できます。

**例: `MultiQueryRetriever` のプロンプトをカスタマイズする場合**

```python
# core/retriever_manager.py 内 (修正例)
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate

# ( ... )

class MultiQueryRetrieverStrategy(RetrieverStrategy):
    def get_retriever(self, user_id: int, dataset_ids: Optional[List[int]] = None) -> BaseRetriever:
        base_retriever = BasicRetrieverStrategy().get_retriever(user_id, dataset_ids)
        llm = llm_manager.get_llm()

        # --- ここからがカスタマイズ部分 ---

        # 1. カスタムプロンプトを定義
        prompt_template = """（ここに、あなたのユースケースに特化した、複数クエリを生成するためのプロンプトを記述）

        Original question: {question}"""
        prompt = PromptTemplate.from_template(prompt_template)

        # 2. LLMChainを作成
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        # 3. リトリーバーを直接インスタンス化
        return MultiQueryRetriever(
            retriever=base_retriever,
            llm_chain=llm_chain
        )
```

このアプローチにより、MochiRAGのすべてのプロンプト駆動型RAG戦略を、用途に応じて柔軟にカスタマイズすることが可能です。
