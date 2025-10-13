# 開発者セットアップガイド

## 1. 概要

このガイドは、MochiRAGプロジェクトの開発環境をローカルマシンにセットアップし、アプリケーションを実行、テストするための一連の手順を説明します。

## 2. 前提条件

開発を始める前に、以下のツールがインストールされていることを確認してください。

- **Python**: 3.10以上
- **Poetry**: Pythonの依存関係管理ツール。
- **Git**: バージョン管理システム。
- **Ollama** (任意): オープンソースLLMをローカルで実行する場合に必要です。公式サイトの指示に従ってインストールしてください。

## 3. 環境構築手順

### ステップ1: リポジトリのクローン

```bash
git clone <リポジトリのURL>
cd mochirag
```

### ステップ2: 依存関係のインストール

本プロジェクトでは、`Poetry`を使用してPythonのライブラリを管理します。以下のコマンドを実行して、必要な依存関係をインストールし、プロジェクト用の仮想環境を作成します。

```bash
poetry install
```

### ステップ3: 仮想環境のアクティベート

Poetryが作成した仮想環境を有効化します。これにより、以降のコマンドはプロジェクト専用の環境で実行されます。

```bash
poetry shell
```
ターミナルのプロンプトの前に仮想環境名が表示されれば成功です。

### ステップ4: 環境変数の設定

プロジェクトのルートディレクトリに `.env` という名前のファイルを作成し、アプリケーションの実行に必要な設定を記述します。

**`.env` ファイルの例:**

```dotenv
# --- 必須 ---
# JWTトークンの署名に使用する秘密鍵。任意の長い文字列を設定してください。
SECRET_KEY="your-super-secret-and-long-string-for-jwt"

# --- 任意 (使用するLLMに応じて設定) ---
# Ollama
OLLAMA_BASE_URL=http://localhost:11434

# OpenAI
OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Azure OpenAI
AZURE_OPENAI_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"

# Google Gemini
GOOGLE_API_KEY="AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxx"

# --- 任意 (LangSmithでのデバッグ用) ---
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY="ls__xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
LANGCHAIN_PROJECT="MochiRAG-Dev" # プロジェクト名（任意）
```
**注意**: `.env` ファイルは `.gitignore` に含まれており、Gitの管理対象外です。APIキーなどの機密情報をコミットしないようにしてください。

## 4. アプリケーションの実行

バックエンドAPIサーバーとフロントエンドUIは、それぞれ個別のプロセスとして起動する必要があります。

### 4.1. バックエンドの起動

以下のコマンドを実行して、FastAPIサーバーを起動します。`--reload`フラグにより、コード変更時にサーバーが自動的に再起動します。

```bash
# poetry shell で仮想環境に入っていることを確認
uvicorn backend.main:app --reload --port 8000
```
または、Poetryのスクリプト機能を利用することもできます。
```bash
poetry run start
```
サーバーが起動すると、`http://localhost:8000/docs` でAPIの対話的なドキュメントを閲覧できます。

### 4.2. フロントエンドの起動

別のターミナルを開き、同様に `poetry shell` で仮想環境に入った後、以下のコマンドでStreamlitアプリケーションを起動します。

```bash
streamlit run frontend/app.py
```
起動後、Webブラウザで `http://localhost:8501` にアクセスすると、アプリケーションのUIが表示されます。

## 5. テストの実行

プロジェクト全体の単体テスト・結合テストを実行するには、以下のコマンドを使用します。

```bash
pytest
```
または、Poetryのスクリプト機能を利用します。
```bash
poetry run test
```

## 6. 管理者用CLI (`cli.py`)

`cli.py` は、管理者向けの操作を行うためのコマンドラインインターフェースです。

### 共有データベースの作成

全ユーザーが利用できる共有のナレッジベースを作成するには、`create-shared-db`コマンドを使用します。

**コマンド書式:**
```bash
python cli.py create-shared-db --name <表示名> --source-dir <ドキュメントのパス>
```

**実行例:**
```bash
# /path/to/shared_docs ディレクトリ内のドキュメントから「全社共通ドキュメント」という名前の共有DBを作成
python cli.py create-shared-db --name "全社共通ドキュメント" --source-dir "/path/to/shared_docs"
```

このコマンドは、指定されたディレクトリ内のドキュメントをスキャンし、ベクトル化して専用のコレクションに保存します。成功すると、`shared_dbs.json` ファイルが自動的に更新され、アプリケーションからこの共有DBが利用可能になります。
