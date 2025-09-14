# 開発者セットアップガイド

このガイドは、MochiRAGプロジェクトの開発環境をセットアップし、アプリケーションを実行するための手順を説明します。

## 1. 開発ツール

本プロジェクトでは、以下のツールを組み合わせて利用します。

- **Poetry:** `pyproject.toml`に基づいた依存関係の解決と`poetry.lock`ファイルの生成を担当します。
- **uv (推奨):** Rust製の高速なPythonパッケージインストーラー。Poetryが生成したロックファイルから、仮想環境の作成とパッケージのインストールを高速に実行します。
- **Git:** バージョン管理システム。
- **Ollama (推奨):** ローカルでのLLM実行環境。

## 2. セットアップ手順
# 開発者セットアップガイド

このガイドは、MochiRAGプロジェクトの開発環境をセットアップし、アプリケーションを実行するための手順を説明します。

## 1. 開発ツール

本プロジェクトでは、以下のツールを組み合わせて利用します。

- **Poetry:** `pyproject.toml`に基づいた依存関係の解決と仮想環境管理を担当します。
- **Git:** バージョン管理システム。
- **Ollama (任意):** ローカルでのLLM実行環境。ローカルでモデルを動かす場合に使用します。

（注）従来ドキュメントで紹介していた `uv` ベースのワークフローはオプションです。本ガイドではより標準的な `poetry` + 仮想環境手順を推奨します。

## 2. セットアップ手順 (Poetry 推奨)

### 2.1. 必要なツールのインストール

# 開発者セットアップガイド

このガイドは、MochiRAGプロジェクトの開発環境をセットアップし、アプリケーションを実行するための手順を説明します。ここでは標準的なPoetryワークフローを推奨します。

## 1. 開発ツール

本プロジェクトでは、以下のツールを組み合わせて利用します。

- **Poetry:** `pyproject.toml`に基づいた依存関係の解決と仮想環境管理を担当します。
- **Git:** バージョン管理システム。
- **Ollama (任意):** ローカルでのLLM実行環境。ローカルでモデルを動かす場合に使用します。

注: `uv`ベースのワークフローはオプションです。本ドキュメントではPoetryを中心とした手順を示します。

## 2. セットアップ手順 (Poetry 推奨)

### 2.1. 必要なツールのインストール

以下は一般的なインストール例です。OSや好みに合わせて調整してください。

```bash
# システムにPythonが無い場合は先にインストールしてください (推奨: 3.10+)
pip install --user poetry
```

### 2.2. プロジェクトのセットアップ

```bash
# 1. リポジトリをクローン
git clone <リポジトリURL>
cd MochiRAG

# 2. Poetryで依存関係をインストールし、仮想環境を作成
poetry install

# 3. 仮想環境に入る (推奨)
# これにより以降のコマンドはPoetryの仮想環境内で実行されます
poetry shell
```

### 2.3. 依存関係の追加・更新

プロジェクトに新しいライブラリを追加する場合は、`poetry`を使用します。

```bash
# 例: FastAPIをプロジェクトに追加
poetry add fastapi

# 変更をローカル環境に反映
poetry install
```

## 3. 環境設定

### 3.1. 基本設定

プロジェクトルートに`.env`ファイルを作成し、必要な環境変数を設定します。例:

```text
# JWT署名のための秘密鍵 (必須)
SECRET_KEY=your-super-secret-key-for-jwt

# Ollamaを使用する場合
OLLAMA_BASE_URL=http://localhost:11434

# OpenAI
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Azure OpenAI
AZURE_OPENAI_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# Google Gemini
GOOGLE_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

`python-dotenv`がプロジェクトで利用されていれば、`.env`の内容は自動的に読み込まれます。

### 3.2. LangSmith連携（任意）

開発効率を向上させるため、LangSmithとの連携を推奨します。LangSmithのサイトでAPIキーを取得し、以下の環境変数を`.env`ファイルに追記してください。

```text
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<your-langsmith-api-key>
# LANGCHAIN_PROJECT=<your-project-name> # (任意)
```

## 4. 実行とテスト

### 4.1. バックエンドAPIの起動

```bash
# Poetryの仮想環境内で実行
uvicorn backend.main:app --reload --port 8000
```

### 4.2. テストの実行

```bash
pytest
```

## 5. 管理者用CLI

プロジェクトには、管理者向けの操作を行うためのコマンドラインインターフェース（`cli.py`）が含まれています。

### 5.1. 共有データベースの作成

全ユーザーが利用できる共有データベースを作成するには、`create-shared-db`コマンドを使用します。

コマンド例:
```bash
python cli.py create-shared-db [OPTIONS]
```

オプション:
- `--name TEXT`: UIに表示される共有データベースの名前（必須）。
- `--source-dir PATH`: インジェストするドキュメント（PDF, TXT, MD）が含まれるディレクトリのパス（必須）。

実行例:
```bash
# Poetryの仮想環境を有効化した状態で、プロジェクトのルートディレクトリから実行
python cli.py create-shared-db --name "全社共通ドキュメント" --source-dir "/path/to/shared_docs"
```

処理内容:
このコマンドを実行すると、以下の処理が自動的に行われます。
1.  `--source-dir`内の対応する拡張子のファイルが検索されます。
2.  見つかったファイルがベクトル化され、新しいコレクションとしてベクトルデータベースに保存されます。
3.  成功すると、`shared_dbs.json`ファイルが自動的に更新され、新しい共有データベースがアプリケーションに登録されます。
