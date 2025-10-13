# ガイド: 新規RAGリトリーバー戦略の追加

## 1. 概要

このガイドは、MochiRAGシステムに新しい検索ロジック（リトリーバー戦略）を追加するための手順を解説します。

本システムは**Strategyパターン**を採用しており、すべてのリトリーバー戦略は `core/retriever_manager.py` の `RetrieverManager` によって管理されます。設定ファイル `config/strategies.yaml` に基づいて戦略が動的にロードされるため、開発者は以下の2つの作業を行うだけで、新しい戦略を簡単に追加できます。

1.  **新しい戦略クラスを実装する。**
2.  **YAML設定ファイルにその戦略を登録する。**

アプリケーションを再起動すると、`RetrieverManager` が新しい戦略を自動的に検出し、チャット画面のドロップダウンメニューから利用可能になります。

## 2. 実装手順

### ステップ1: 戦略クラスの実装

新しいリトリーバー戦略クラスは、`core/retriever_manager.py` 内に定義する必要があります。

#### クラスの要件

- `RetrieverStrategy` 抽象ベースクラスを継承すること。
- `get_retriever(self, user_id: int, dataset_ids: Optional[List[int]] = None) -> BaseRetriever` というシグネチャを持つメソッドを実装すること。このメソッドが、LangChainの `BaseRetriever` オブジェクトを返すロジックの中心です。

#### 実装例

以下は、既存の `BasicRetrieverStrategy` をラップして、検索結果を逆順にするという単純なカスタム戦略の例です。多くのカスタム戦略は、このように基本的なリトリーバーを拡張する形で実装できます。

```python
# core/retriever_manager.py に追加するクラスの例

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List

# ... 他のimport文

# 既存の戦略をラップするカスタムリトリーバー
class ReverseRetriever(BaseRetriever):
    """取得したドキュメントの順序を逆にするリトリーバー"""
    base_retriever: BaseRetriever

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # ベースとなるリトリーバーでドキュメントを取得
        docs = self.base_retriever.get_relevant_documents(query, callbacks=run_manager.get_child())
        # 順序を逆にして返す
        return docs[::-1]

# 新しい戦略クラス
class ReverseRetrieverStrategy(RetrieverStrategy):
    """
    基本戦略の検索結果を逆順にする戦略。
    """
    def get_retriever(self, user_id: int, dataset_ids: Optional[List[int]] = None) -> BaseRetriever:
        # 1. 基本となるリトリーバーをBasicRetrieverStrategyから取得
        base_retriever = BasicRetrieverStrategy().get_retriever(user_id, dataset_ids)

        # 2. カスタムリトリーバーでラップして返す
        return ReverseRetriever(base_retriever=base_retriever)
```

### ステップ2: YAML設定ファイルへの登録

クラスを実装したら、`config/strategies.yaml` の `retrievers` セクションに新しいエントリを追加して戦略を登録します。

#### 設定の要件

- `reverse_retriever` のような一意なキー名で新しいエントリを追加します（このキー名がAPIで使われます）。
- `strategy_class` に、ステップ1で作成したクラスの名前を文字列で指定します。
- `description` に、UIのドロップダウンメニューに表示される説明文を記述します。
- `parameters` は、この戦略に特有の設定値がない場合は空 (`{}`) のままで構いません。

#### 設定例

`config/strategies.yaml` の `retrievers` セクションに以下を追記します。

```yaml
# config/strategies.yaml

retrievers:
  # ... 既存の戦略 ...

  reverse_retriever:
    strategy_class: "ReverseRetrieverStrategy"
    description: "デモ用：基本検索の結果を逆順にします。"
    parameters: {}
```

## 3. 検証

上記2つのステップが完了したら、アプリケーションを再起動してください。

バックエンドとフロントエンドを起動し、チャット画面の「Select RAG Strategy」ドロップダウンメニューを開くと、新しい戦略（この例では `reverse_retriever`）が説明文と共に表示されているはずです。これを選択してクエリを実行し、意図通りに動作することを確認してください。
