# EnsembleRetrieverによる複数データベース同時検索の実装案

## 1. 目的 (Objective)

ユーザーがUI上で選択した複数のデータセット（個人用・共有を問わず）を横断し、関連ドキュメントを一度に検索・取得できる機能を実装する。これにより、RAG（Retrieval-Augmented Generation）の利便性を向上させる。

## 2. 主要な課題 (Key Challenge)

- **コレクションの分離**: 個人用データは `user_{user_id}` という名前のChroma DBコレクションに、共有データは `shared_*` という名前のコレクションに、それぞれ物理的に異なる場所へ格納されている。
- **リトリーバーの制約**: LangChainの基本的なリトリーバー（`vector_store.as_retriever()`）は、単一のコレクションしか検索対象にできない。

## 3. 提案する解決策 (Proposed Solution)

LangChainが提供する **`EnsembleRetriever`** を利用する。

`EnsembleRetriever`は、複数のリトリーバー（それぞれが異なるコレクションや設定を持つ）をリストとして受け取り、それらを並行して実行します。その後、各リトリーバーから得られた検索結果を、Reciprocal Rank Fusion (RRF) などのアルゴリズムを用いてインテリジェントに統合し、最終的なランク付けされたドキュメントリストを生成します。

この機能を利用することで、個人用コレクションを検索するリトリーバーと、各共有コレクションを検索するリトリーバーを動的に作成し、それらを束ねて単一の検索インターフェースとして扱うことが可能になる。

## 4. 実装ステップ (Implementation Steps)

修正の中心は `core/retriever_manager.py` の `BasicRetrieverStrategy` クラスです。

---

### **ファイル:** `core/retriever_manager.py`
### **クラス:** `BasicRetrieverStrategy`
### **メソッド:** `get_retriever`

#### ステップ 1: `dataset_ids` の分類

引数として渡される `dataset_ids` （例: `[1, 2, -1, -3]`）を、個人用ID（正の数）と共有ID（負の数）に分類します。

```python
personal_ids = [ds_id for ds_id in dataset_ids if ds_id > 0]
shared_ids = [ds_id for ds_id in dataset_ids if ds_id < 0]
```

#### ステップ 2: 個人用リトリーバーの作成

`personal_ids` のリストが空でない場合、ユーザー個人のコレクションを対象とするリトリーバーを1つ作成します。

- **コレクション名**: `f"user_{user_id}"`
- **検索フィルタ**: `{"$and": [{"user_id": user_id}, {"dataset_id": {"$in": personal_ids}}]}`

この設定で `vector_store.as_retriever()` を呼び出してリトリーバーを作成し、最終的に`EnsembleRetriever`に渡すためのリストに追加します。

#### ステップ 3: 共有用リトリーバーの作成

`shared_ids` のリストが空でない場合、共有コレクションを対象とするリトリーバーを**コレクションごと**に作成します。

1.  `shared_dbs.json` を読み込み、IDとコレクション名の対応表（マッピング）を作成します。
2.  共有IDを、属するコレクション名でグループ化します。
    - 例: `shared_ids` が `[-2, -4]` で、両方とも `shared_main_db` コレクションに属する場合 -> `{"shared_main_db": [-2, -4]}`
3.  グループ化されたコレクションごとにループ処理を行います。
    - **コレクション名**: `shared_main_db`
    - **検索フィルタ**: `{"dataset_id": {"$in": [-2, -4]}}`
    - 上記設定でリトリーバーを作成し、リストに追加します。

#### ステップ 4: `EnsembleRetriever` の構築

ステップ2と3で作成したリトリーバーのリストを使って、`EnsembleRetriever`を構築します。

- リストが空の場合: 何も返さないリトリーバーを返す。
- リストにリトリーバーが1つしかない場合: そのリトリーバーをそのまま返す。
- リストに複数のリトリーバーがある場合: `EnsembleRetriever`を初期化して返す。

```python
from langchain.retrievers import EnsembleRetriever

# ...

ensemble_retriever = EnsembleRetriever(
    retrievers=list_of_created_retrievers,
    weights=[0.5, 0.5] # オプション: 各リトリーバーの重み付け
)
return ensemble_retriever
```
`weights`パラメータで各リトリーバーの重要度を調整できますが、まずは均等（例: すべて0.5）で実装します。

## 5. コード例

以下は、`BasicRetrieverStrategy.get_retriever` メソッドの修正後のイメージです。

```python
# core/retriever_manager.py

import json
from langchain.retrievers import EnsembleRetriever
# ... other imports

class BasicRetrieverStrategy(RetrieverStrategy):
    def get_retriever(self, user_id: int, dataset_ids: Optional[List[int]] = None) -> BaseRetriever:
        if not dataset_ids:
            # データセットが指定されていない場合は、ユーザーの全ドキュメントを対象とする
            collection_name = f"user_{user_id}"
            vector_store = vector_store_manager.get_vector_store(collection_name)
            return vector_store.as_retriever(search_kwargs={"filter": {"user_id": user_id}})

        retrievers = []
        config = config_manager.get_retriever_config("basic")
        search_k = config.parameters.get("k", 5)

        # 1. 個人用データセットのIDを分離
        personal_ids = [ds_id for ds_id in dataset_ids if ds_id > 0]
        if personal_ids:
            personal_collection_name = f"user_{user_id}"
            personal_vector_store = vector_store_manager.get_vector_store(personal_collection_name)
            personal_filter = {"$and": [{"user_id": user_id}, {"dataset_id": {"$in": personal_ids}}]}
            retrievers.append(
                personal_vector_store.as_retriever(search_kwargs={"k": search_k, "filter": personal_filter})
            )

        # 2. 共有データセットのIDを分離し、コレクションごとにグループ化
        shared_ids = [ds_id for ds_id in dataset_ids if ds_id < 0]
        if shared_ids:
            try:
                with open("shared_dbs.json", "r") as f:
                    shared_dbs_config = json.load(f)
                
                id_to_collection_map = {db["id"]: db["collection_name"] for db in shared_dbs_config}
                
                collection_to_ids_map = {}
                for ds_id in shared_ids:
                    if ds_id in id_to_collection_map:
                        collection_name = id_to_collection_map[ds_id]
                        if collection_name not in collection_to_ids_map:
                            collection_to_ids_map[collection_name] = []
                        collection_to_ids_map[collection_name].append(ds_id)
                
                for collection_name, ids in collection_to_ids_map.items():
                    shared_vector_store = vector_store_manager.get_vector_store(collection_name)
                    shared_filter = {"dataset_id": {"$in": ids}}
                    retrievers.append(
                        shared_vector_store.as_retriever(search_kwargs={"k": search_k, "filter": shared_filter})
                    )

            except (FileNotFoundError, json.JSONDecodeError):
                print("Warning: Could not load or parse shared_dbs.json")


        # 3. EnsembleRetrieverを構築
        if not retrievers:
            # 有効なデータセットが見つからなかった場合
            empty_vs = vector_store_manager.get_vector_store(f"user_{user_id}")
            return empty_vs.as_retriever(search_kwargs={"k": 0})
        elif len(retrievers) == 1:
            return retrievers[0]
        else:
            # 均等な重み付けでEnsembleRetrieverを初期化
            weights = [1.0 / len(retrievers)] * len(retrievers)
            return EnsembleRetriever(retrievers=retrievers, weights=weights)

```

## 6. その他の考慮事項

- **他戦略への展開**: `MultiQueryRetrieverStrategy` や `ContextualCompressionRetrieverStrategy` は内部で `BasicRetrieverStrategy` を利用しているため、この修正の恩恵を自動的に受けます。しかし、`HydeRetrieverStrategy` は独自のロジックでリトリーバーを構築しているため、同様の `EnsembleRetriever` を用いた修正が別途必要になります。
- **対象外の戦略**: `ParentDocumentRetrieverStrategy` は、ドキュメントの親子関係を管理するために特殊な `SQLDocStore` を利用しており、構造が大きく異なります。そのため、今回の同時検索機能の実装の対象外とします。
