# Guide: Adding New RAG Retriever Strategies

このガイドは、RAGシステムに新しいリトリーバー戦略を追加する手順を示します。日本語の説明に英語のサンプルコードを含みます。

## 概要

本システムではStrategyパターンを用いて複数のretriever実装を管理しています。すべてのretriever戦略は`core/retriever_manager.py`の`RetrieverManager`によって管理されます。

設定ファイル`config/strategies.yaml`に基づき戦略が動的にロードされるため、新しい戦略を追加するには次の2つだけです。
1. 新しい戦略クラスを実装する。
2. YAML設定にその戦略を登録する。

アプリケーションを再起動すると自動的に新しい戦略が検出され、利用可能になります。

## ステップ1: 戦略クラスを実装する

retriever戦略クラスはすべて`RetrieverStrategy`（抽象クラス）を継承し、`get_retriever`メソッドを実装する必要があります。通常、これらのクラスは`core/retriever_manager.py`に置きます。

### 要件
- クラスは `core/retriever_manager.py` に追加すること。
- `RetrieverStrategy` を継承すること。
- 次のシグネチャのメソッドを実装すること: `get_retriever(self, user_id: int, dataset_ids: Optional[List[int]] = None) -> BaseRetriever`

### 例
以下は簡単なTF-IDFベースのretriever戦略の例です（デモ用の簡略実装）。実際の実装ではデータベースからドキュメントを取得する処理が必要です。

```python
# 例: core/retriever_manager.py に追加
from langchain.retrievers import TFIDFRetriever
from langchain.schema import Document

class TfidfRetrieverStrategy(RetrieverStrategy):
    """
    TF-IDFを用いた簡易retriever戦略の例。
    実運用ではユーザーやデータセットに紐づくドキュメントをDBから取得する実装が必要です。
    """
    def get_retriever(self, user_id: int, dataset_ids: Optional[List[int]] = None) -> BaseRetriever:
        # デモ用のモック実装
        mock_documents = [
            Document(page_content="Mochi is a Japanese rice cake made of mochigome."),
            Document(page_content="Software engineering is a systematic approach to software development."),
            Document(page_content="The MochiRAG system is extensible."),
        ]

        return TFIDFRetriever.from_documents(
            documents=mock_documents,
            k=3
        )
```

## ステップ2: YAMLで戦略を登録する

クラスを実装したら、`config/strategies.yaml` の `retrievers` セクションにエントリを追加して登録します。

### 要件
- `tfidf_retriever` のようなキー名で新しいエントリを追加すること。
- `strategy_class` に作成したクラス名を指定すること。
- `description` や必要な `parameters` を追加すること。

### 例
`config/strategies.yaml` に次を追加します:

```yaml
# config/strategies.yaml の 'retrievers' セクション内

  tfidf_retriever:
    strategy_class: "TfidfRetrieverStrategy"
    description: "A simple TF-IDF based retriever for demonstration."
    parameters: {}
```

## ステップ3: 検証

上記の変更を行ったらアプリケーションを再起動してください。チャット画面の「Select RAG Strategy」ドロップダウンに新しい戦略（この例では `tfidf_retriever`）が表示されるはずです。

`RetrieverManager` が戦略のロードとインスタンス化を管理します。
