# 開発者セットアップガイド

このガイドは、MochiRAGプロジェクトの開発環境をセットアップし、アプリケーションを実行するための手順を説明します。

## 1. 開発ツール

本プロジェクトでは、以下のツールを組み合わせて利用します。

- **Poetry:** `pyproject.toml`に基づいた依存関係の解決と`poetry.lock`ファイルの生成を担当します。
- **uv (推奨):** Rust製の高速なPythonパッケージインストーラー。Poetryが生成したロックファイルから、仮想環境の作成とパッケージのインストールを高速に実行します。
- **Git:** バージョン管理システム。
- **Ollama (推奨):** ローカルでのLLM実行環境。

## 2. セットアップ手順

`uv`と`Poetry`を組み合わせた、高速でモダンなセットアップ手順を推奨します。

### 2.1. 必要なツールのインストール

```bash
# uvとpoetryを未インストールの場合は、pipxまたはpipでインストール
pip install uv
pip install poetry
```

### 2.2. プロジェクトのセットアップ

```bash
# 1. リポジトリをクローン
git clone <リポジトリURL>
cd MochiRAG

# 2. 仮想環境の作成
# .venvという名前で仮想環境を作成します
uv venv

# 3. 依存関係のインストール
# poetry.lockから本番・開発・テスト用の依存関係を高速にインストールします
uv pip sync --all

# 4. 仮想環境の有効化
# 以降、この仮想環境で作業します
source .venv/bin/activate
# Windowsの場合: .venv\Scripts\activate
```

### 2.3. 依存関係の追加・更新

プロジェクトに新しいライブラリを追加する場合は、`poetry`を使用します。

```bash
# 例: FastAPIをプロジェクトに追加
poetry add fastapi

# 変更をpoetry.lockに反映させた後、uvで仮想環境に同期
uv pip sync --all
```

## 3. 環境設定

### 3.1. 基本設定

プロジェクトルートに`.env`ファイルを作成し、必要な環境変数を設定します。

```
# .env file

# JWT署名のための秘密鍵 (必須)
SECRET_KEY=your-super-secret-key-for-jwt

# Ollamaを使用する場合
OLLAMA_BASE_URL=http://localhost:11434

# --- 各LLMプロバイダーのAPIキー ---
# 使用するプロバイダーに応じて設定してください。
# config/strategies.yamlで直接指定することも可能ですが、
# .envファイルで管理することが推奨されます。

# OpenAI
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Azure OpenAI
AZURE_OPENAI_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# Google Gemini
GOOGLE_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

`python-dotenv`が自動的にこのファイルを読み込み、環境変数として設定します。

### 3.2. LangSmith連携（強く推奨）

開発効率を向上させるため、LangSmithとの連携を推奨します。LangSmithのサイトでAPIキーを取得し、以下の環境変数を`.env`ファイルに追記してください。

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<your-langsmith-api-key>
# LANGCHAIN_PROJECT=<your-project-name> # (任意) プロジェクト名を指定
```

## 4. 実行とテスト

### 4.1. バックエンドAPIの起動

```bash
# 有効化された仮想環境内で実行
uvicorn backend.main:app --reload --port 8000
```

### 4.2. テストの実行

```bash
pytest
```
