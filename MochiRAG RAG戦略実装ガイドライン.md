# MochiRAG RAG戦略実装ガイドライン

## 1. はじめに

本ガイドラインは、MochiRAGプロジェクトに新たなRetrieval-Augmented Generation (RAG) 戦略を実装する開発者向けに、そのプロセスとベストプラクティスをまとめたものです。MochiRAGは、モジュール化されたRAGアーキテクチャを採用しており、多様なRAG戦略を柔軟に組み込むことが可能です。このガイドラインに従うことで、既存のコードベースとの整合性を保ちつつ、効率的かつ堅牢なRAG戦略を開発・統合することができます。

## 2. RAG戦略の基本構造

全てのRAG戦略は、`core/rag_strategies/base.py` に定義されている抽象基底クラス `BaseRAGStrategy` を継承する必要があります。このクラスは、RAG戦略が持つべき共通のインターフェースを定義しており、以下の抽象メソッドの実装が必須です。

### 2.1. `BaseRAGStrategy` クラスの概要

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from langchain_core.documents import Document

class BaseRAGStrategy(ABC):
    """
    全てのRAG戦略が継承すべき抽象基底クラス。
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def get_name(self) -> str:
        """
        戦略の名前を返す。
        """
        pass

    @abstractmethod
    def retrieve(self, user_id: str, question: str, data_source_ids: List[str], embedding_strategy_for_retrieval: str) -> List[Document]:
        """
        与えられた質問に基づいて関連ドキュメントを検索する。
        """
        pass

    @abstractmethod
    def generate_response(self, question: str, retrieved_documents: List[Document]) -> Dict[str, Any]:
        """
        検索されたドキュメントと質問に基づいて応答を生成する。
        """
        pass

    def execute(self, user_id: str, question: str, data_source_ids: List[str], embedding_strategy_for_retrieval: str) -> Dict[str, Any]:
        """
        RAG戦略の実行フローを定義する。
        このメソッドは通常、`retrieve` と `generate_response` を呼び出す。
        より複雑な戦略（例: DeepRAG）では、このメソッドをオーバーライドして、
        動的な意思決定ロジックを実装することができる。
        """
        retrieved_documents = self.retrieve(user_id, question, data_source_ids, embedding_strategy_for_retrieval)
        response = self.generate_response(question, retrieved_documents)
        return response

```

### 2.2. 必須メソッドの実装

*   **`__init__(self, config: Dict[str, Any])`**: 
    *   コンストラクタは、そのRAG戦略に特有の設定を辞書形式で受け取ります。これにより、コードを変更することなく、外部から戦略の挙動を調整できます。
    *   `self.config` に設定を保存し、`retrieve` や `generate_response` メソッド内で利用します。

*   **`get_name(self) -> str`**: 
    *   この戦略の一意な名前を文字列で返します。この名前は、APIリクエストで戦略を指定する際に使用されます。
    *   例: `"simple_rag"`, `"advanced_rag"`, `"deep_rag"`。

*   **`retrieve(self, user_id: str, question: str, data_source_ids: List[str], embedding_strategy_for_retrieval: str) -> List[Document]`**: 
    *   与えられた質問に基づいて、関連するドキュメントをベクトルストアから検索するロジックを実装します。
    *   `user_id`: 現在のユーザーのID。ユーザー固有のデータにアクセスするために使用します。
    *   `question`: ユーザーからの質問文字列。
    *   `data_source_ids`: 検索対象となるデータソース（ファイル）のIDリスト。このリストに含まれるデータソースのみを検索対象とすべきです。
    *   `embedding_strategy_for_retrieval`: 検索に使用するエンベディング戦略の名前。`vector_store_manager.query_documents` に渡します。
    *   戻り値は、`langchain_core.documents.Document` オブジェクトのリストである必要があります。

*   **`generate_response(self, question: str, retrieved_documents: List[Document]) -> Dict[str, Any]`**: 
    *   検索されたドキュメントと元の質問を使用して、LLMで最終的な応答を生成するロジックを実装します。
    *   `question`: ユーザーからの質問文字列。
    *   `retrieved_documents`: `retrieve` メソッドによって取得された関連ドキュメントのリスト。
    *   戻り値は辞書形式で、少なくとも `"answer"` キーに生成された応答文字列を含める必要があります。オプションで `"sources"` キーに参照したドキュメントのリストを含めることができます。

*   **`execute(self, user_id: str, question: str, data_source_ids: List[str], embedding_strategy_for_retrieval: str) -> Dict[str, Any]`**: 
    *   RAG戦略の全体的な実行フローを定義します。デフォルトの実装では `retrieve` を呼び出し、その結果を `generate_response` に渡します。
    *   DeepRAGのような複雑な自律型RAG戦略を実装する場合、このメソッドをオーバーライドして、質問の複雑さ分析、複数回の検索・生成サイクル、動的なツール選択などのロジックを組み込むことができます。

## 3. 検索（Retrieve）の実装詳細

`retrieve` メソッドの実装では、`core.vector_store_manager.vector_store_manager` を活用します。これは、ChromaDBをバックエンドとするベクトルストアへのアクセスを抽象化したものです。

### 3.1. `vector_store_manager.query_documents` の利用

最も基本的な検索は、`vector_store_manager.query_documents` メソッドを呼び出すことで実現できます。

```python
from core.vector_store_manager import vector_store_manager

# ... (BaseRAGStrategyを継承したクラス内)

def retrieve(self, user_id: str, question: str, data_source_ids: List[str], embedding_strategy_for_retrieval: str) -> List[Document]:
    n_results = self.config.get("retrieval_k", 4) # 設定から検索数を取得
    
    retrieved_docs = vector_store_manager.query_documents(
        user_id=user_id,
        query=question,
        embedding_strategy_name=embedding_strategy_for_retrieval,
        n_results=n_results,
        data_source_ids=data_source_ids
    )
    return retrieved_docs
```

### 3.2. 高度な検索ロジック

*   **リランキング**: 最初に多めにドキュメントを検索し（例: `n_results` を大きく設定）、その後、より高度なリランキングモデル（例: Cross-encoder）を使用して、関連性の高いドキュメントを絞り込むことができます。リランキングモデルは、LangChainの `rerank` モジュールなどを利用して統合可能です。
*   **ハイブリッド検索**: ベクトル検索だけでなく、キーワード検索（BM25など）を組み合わせることで、検索の精度を向上させることができます。両方の結果をマージし、リランキングで最終的なドキュメントセットを決定します。
*   **マルチホップ検索**: 複雑な質問に対して、複数回の検索ステップを実行します。最初の検索結果から中間的な回答を生成し、その回答を基に次の質問を生成して再度検索を行う、といった連鎖的な検索フローを `execute` メソッド内で実装できます。
*   **クエリ拡張/書き換え**: ユーザーの質問を直接検索するのではなく、LLMを使用して質問を拡張したり、複数のサブクエリに分割したり、異なる視点から質問を書き換えたりすることで、より網羅的な検索結果を得ることができます。

## 4. 生成（Generate）の実装詳細

`generate_response` メソッドでは、検索されたドキュメントと質問を基に、LLMを用いて最終的な応答を生成します。LangChainのLLM、ChatModel、プロンプトテンプレートを活用します。

### 4.1. LLMとプロンプトテンプレートの利用

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ... (BaseRAGStrategyを継承したクラス内)

def generate_response(self, question: str, retrieved_documents: List[Document]) -> Dict[str, Any]:
    llm_model_name = self.config.get("llm_model_name", "gpt-3.5-turbo")
    temperature = self.config.get("temperature", 0) # 設定から温度を取得
    
    llm = ChatOpenAI(model=llm_model_name, temperature=temperature)

    template = self.config.get("prompt_template", 
        "以下のコンテキストのみを使用して、質問に答えてください。\n"
        "コンテキスト: {context}\n"
        "質問: {question}\n"
        "回答:"
    )
    
    prompt = ChatPromptTemplate.from_template(template)

    context = "\n\n".join([doc.page_content for doc in retrieved_documents])
    chain = prompt | llm
    answer = chain.invoke({"context": context, "question": question}).content

    return {"answer": answer, "sources": retrieved_documents}
```

### 4.2. 高度な生成ロジック

*   **プロンプトエンジニアリング**: 質問の種類や検索されたドキュメントの特性に応じて、異なるプロンプトテンプレートを動的に選択したり、Few-shotの例を組み込んだりすることで、LLMの応答品質を向上させることができます。
*   **応答の検証と修正**: LLMが生成した応答が、検索されたドキュメントの内容と矛盾していないか、あるいは特定の基準を満たしているかを検証するステップを追加できます。必要に応じて、LLMに自己修正を促したり、別のLLMで再生成させたりするロジックを実装します。
*   **要約と統合**: 複数のドキュメントから情報を抽出し、それらを統合して一貫性のある回答を生成する際に、より高度な要約技術や情報統合技術を適用できます。
*   **対話管理**: ユーザーとの複数ターンにわたる対話において、過去の対話履歴を考慮して質問を理解し、応答を生成するロジックを組み込むことができます。

## 5. 設定の管理

各RAG戦略は、その挙動を制御するための設定を `__init__` メソッドで `config` 辞書として受け取ります。これにより、コードの変更なしに戦略のパラメータを調整できます。

### 5.1. `config.json` の利用

RAG戦略の設定は、アプリケーションの起動時に読み込まれる `config.json` (または同様の設定ファイル) で管理することが推奨されます。

**`config.json` の例:**

```json
{
    "rag_strategies": {
        "simple_rag": {
            "retrieval_k": 5,
            "llm_model_name": "gpt-3.5-turbo",
            "temperature": 0.7,
            "prompt_template": "以下の情報に基づいて質問に答えてください。\nコンテキスト: {context}\n質問: {question}\n回答:"
        },
        "advanced_rag": {
            "initial_retrieval_k": 10,
            "final_retrieval_k": 5,
            "use_reranker": true,
            "llm_model_name": "gpt-4o",
            "temperature": 0.1,
            "prompt_templates": [
                {
                    "name": "detailed",
                    "template": "..."
                },
                {
                    "name": "concise",
                    "template": "..."
                }
            ],
            "selected_template": "detailed"
        },
        "deep_rag": {
            "max_hops": 3,
            "max_iterations": 3,
            "quality_threshold": 0.75,
            "llm_model_name": "gpt-4o",
            "temperature": 0.2,
            "strategy_rules": {
                "high_complexity": "iterative",
                "medium_complexity": "multi_hop",
                "low_complexity": "simple"
            }
        }
    },
    "default_rag_strategy": "simple_rag"
}
```

`backend/main.py` や `RAGStrategyFactory` は、この設定ファイルを読み込み、適切な `config` 辞書を各RAG戦略インスタンスに渡すように実装されます。

## 6. ファクトリへの登録

新しいRAG戦略クラスを作成したら、`core/rag_strategies/factory.py` の `RAGStrategyFactory` に登録する必要があります。これにより、アプリケーションがその戦略を認識し、APIリクエストを通じて利用できるようになります。

### 6.1. `factory.py` の修正

`RAGStrategyFactory` の `_register_default_strategies` メソッドに、新しく作成したRAG戦略クラスのインスタンスを登録するコードを追加します。

```python
# core/rag_strategies/factory.py

from .base import BaseRAGStrategy
from .simple_rag import SimpleRAGStrategy
from .advanced_rag import AdvancedRAGStrategy
from .deep_rag import DeepRAGStrategy
# 新しい戦略クラスをインポート
from .your_new_rag_strategy import YourNewRAGStrategy # 例

class RAGStrategyFactory:
    # ... (既存のコード) ...

    def _register_default_strategies(self):
        self.register_strategy(SimpleRAGStrategy({}).get_name(), SimpleRAGStrategy)
        self.register_strategy(AdvancedRAGStrategy({}).get_name(), AdvancedRAGStrategy)
        self.register_strategy(DeepRAGStrategy({}).get_name(), DeepRAGStrategy)
        # 新しい戦略を登録
        self.register_strategy(YourNewRAGStrategy({}).get_name(), YourNewRAGStrategy) # 例

```

`get_name()` メソッドはクラスメソッドとして定義することも可能ですが、現在の実装ではインスタンスメソッドとして定義されているため、一時的なインスタンスを作成して名前を取得しています。これは、将来的に `get_name` をクラスメソッドに変更することで改善できます。

## 7. テストの考慮事項

新しいRAG戦略を実装する際には、以下のテスト項目を考慮することが重要です。

*   **単体テスト**: `retrieve` メソッドと `generate_response` メソッドが、それぞれ独立して期待通りに動作するかを確認します。モックデータやモックLLMを使用して、外部依存なしでテストできるようにします。
*   **統合テスト**: `RAGStrategyFactory` を介して戦略をロードし、`execute` メソッドを呼び出して、RAGパイプライン全体が正しく機能するかを確認します。ベクトルストアやLLMとの実際の連携をテストします。
*   **エッジケーステスト**: 検索結果が空の場合、LLMがエラーを返す場合、入力が異常な場合など、様々なエッジケースでの挙動を確認します。
*   **パフォーマンステスト**: 特に複雑なRAG戦略の場合、応答時間やリソース使用量（CPU, メモリ）が許容範囲内であるかを確認します。

## 8. まとめ

このガイドラインは、MochiRAGの拡張可能なRAGアーキテクチャを活用し、新しいRAG戦略を効果的に実装するためのロードマップを提供します。モジュール性、抽象化、設定可能性の原則に従うことで、MochiRAGは多様なRAGの進化に対応し、より強力な知識検索・生成システムへと発展していくでしょう。

---

**著者**: Manus AI
**日付**: 2025年7月26日


