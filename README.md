# MochiRAG プロジェクトへようこそ

MochiRAG は、個人のドキュメントに基づいて AI と対話できる Retrieval-Augmented Generation (RAG) システムです。ユーザーは自身のドキュメントをアップロードし、お好きな RAG 戦略を選択して、どのような回答が得られるかお試しすることができます。  

まだ、実装途中のものがたくさんあります。

---

## 1. プロジェクト概要

- **目的**: ユーザー単位でドキュメントを分離管理し、安全かつ信頼性の高い AI 応答を実現  
- **RAG 戦略**:  
  - 実装済み: `basic`、`parent_document` (現在は basic と同等)、`multi_query`、`contextual_compression`
  - 設定: [`config/strategies.yaml`](config/strategies.yaml) を参照  
- **主な機能**  
  - ユーザー認証 (FastAPI OAuth2/JWT)  
  - ドキュメント管理 (TXT/MD/PDF のアップロード、メタデータ管理)  
  - RAG チャット (LangChain + Ollama llama3)  
  - データ分離 (ユーザーごとのベクトル DB 分離)

---

## 2. 技術スタック

- バックエンド: Python + FastAPI  
- フロントエンド: Python + Streamlit  
- RAG・LLM: LangChain, Ollama (llama3 モデル)  
- ベクトルストア: ChromaDB (永続化)  
- エンベディング: Sentence Transformers (`all-MiniLM-L6-v2`)  
- テスト: pytest  

---

## 3. 機能一覧

1. **ユーザー認証**  
   - メール／パスワード登録、JWT トークン発行  
   - パスワードはハッシュ化保存  
2. **ドキュメント管理**  
   - TXT, MD, PDF のアップロード・一覧表示  
   - アップロード履歴・メタデータ管理  
3. **RAG チャット**  
   - 自然言語クエリに基づく応答生成  
   - 戦略選択: `basic`, `parent_document`, `multi_query`, `contextual_compression`  
   - 回答の根拠となった参照元ドキュメントの情報をオプションで表示（ファイル名、ページ番号、内容の断片など）
   - チャット時に特定のデータセットを検索対象として指定可能
4. **データセット管理**
   - ユーザーは複数のデータセットを作成し、ドキュメントを整理可能
   - データセットの作成、一覧表示、削除
   - 各データセットへのファイルのアップロード、ファイル一覧表示、ファイル削除
5. **データ分離・セキュリティ**
   - ユーザー毎に ChromaDB を分離  
   - 認証ユーザーのみ自身のデータアクセス可能  

---

## 4. 依存関係とセットアップ

### 4.1 前提条件

- Python 3.10 以上  
- （任意）Ollama が動作し、例えば`llama3` モデルが動作可能な状況であること。

### 4.2 インストール

#### 自動セットアップ 

```bash
git clone <リポジトリURL>
cd MochiRAG

# 開発用スクリプトで仮想環境＆依存関係インストール
chmod +x setup_dev.sh
./setup_dev.sh

# 仮想環境をアクティベート
source venv/bin/activate
```

#### 手動セットアップ

```bash
# 仮想環境の作成・有効化
python3 -m venv venv
source venv/bin/activate

# プロジェクト依存のインストール
pip install --upgrade pip
pip install ".[test]"
```

### 4.3 設定

- 環境変数: `.env` を利用可（FastAPI, Streamlit 設定）  
- RAG 戦略: `config/strategies.yaml` で詳細設定  

---

## 5. 実行方法

uv を使うとより簡便に起動できます。

### 5.1 バックエンド API 起動

```bash
# 仮想環境内で
uvicorn backend.main:app --reload --port 8000
```

### 5.2 フロントエンド UI 起動

```bash
# 仮想環境内で
streamlit run frontend/app.py
```

ブラウザで `http://localhost:8501` を開いて操作してください。

---

## 6. 使用例

### 6.1 cURL からの RAG クエリ

```bash
curl -X POST http://localhost:8000/chat/query/ \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "MochiRAG のアーキテクチャを教えて",
    "strategy": "basic"
  }'
```

### 6.2 Python スニペット

```python
import requests

token = "<YOUR_JWT_TOKEN>"
url = "http://localhost:8000/chat/query/"
data = {"query": "ドキュメント管理機能は？", "strategy": "multi_query"}

resp = requests.post(url, json=data, headers={"Authorization": f"Bearer {token}"})
print(resp.json())
```

---

## 7. テストとコントリビューション

### 7.1 テスト

```bash
# 全テスト実行
pytest -q --disable-warnings --maxfail=1
```

詳細は [`testing_guide.md`](testing_guide.md) を参照。

---

## 8. ディレクトリ構造

```
MochiRAG/
├── backend/              FastAPI バックエンド
├── core/                 RAG ロジック (ドキュメント処理／ベクトルストア)
├── data/                 ChromaDB 永続化データ、サンプルdocs
├── frontend/             Streamlit UI
├── tests/                pytest テストコード
├── config/               RAG 戦略設定 (strategies.yaml)
├── status.md             プロジェクト進捗・ステータス
├── testing_guide.md      テスト実行ガイド
├── setup_dev.sh          セットアップスクリプト
├── requirements.txt      依存関係定義
└── README.md             本ドキュメント
```

---

## 9. API エンドポイント

- `POST /users/`                                                 : 新規ユーザー登録
- `POST /token`                                                  : トークン取得 (ログイン)
- `GET  /users/me`                                               : 認証ユーザー情報取得
- `POST /users/me/datasets/`                                     : データセット作成
- `GET  /users/me/datasets/`                                     : データセット一覧取得
- `GET  /users/me/datasets/{dataset_id}/`                        : 特定データセット詳細取得
- `DELETE /users/me/datasets/{dataset_id}/`                      : データセット削除
- `POST /users/me/datasets/{dataset_id}/documents/upload/`       : データセットへのファイルアップロード
- `GET  /users/me/datasets/{dataset_id}/documents/`              : データセット内ファイル一覧取得
- `DELETE /users/me/datasets/{dataset_id}/documents/{data_source_id}/`: データセットからのファイル削除
- `POST /chat/query/`                                            : RAG チャットクエリ (データセット指定可)

- `POST /documents/upload/` (非推奨: `/users/me/datasets/{dataset_id}/documents/upload/` を使用してください)
- `GET  /documents/`       (非推奨: `/users/me/datasets/{dataset_id}/documents/` を使用してください)

詳細は FastAPI 自動生成ドキュメント (`/docs` または `/redoc`) を参照。

---

## 10. 今後の展望・ロードマップ

- Azure OpenAI など外部 LLM の対応
- チャット UI での引用元表示のさらなる強化（例: クリックで該当箇所表示など）
- ドキュメント処理の非同期／バッチ化  
- 管理者向けダッシュボード
