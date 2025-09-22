# デプロイガイド

## 1. 概要

このドキュメントは、MochiRAGアプリケーションをビルドし、デプロイするための2つの主要なワークフローについて説明します。

1.  **ローカル開発環境での実行**: `docker-compose` を使用して、すべてのサービス（バックエンド、フロントエンド、データベース）をローカルで一度に起動します。
2.  **本番環境へのデプロイ**: `Makefile` を使用して、バックエンドとフロントエンドのDockerイメージをビルドし、コンテナリポジトリにプッシュします。その後、デプロイ先のホストでそのイメージを使用してコンテナを起動します。

---

## 2. ローカル開発環境での実行

`docker-compose.yml` は、開発目的で完全なアプリケーション環境を起動するために最適化されています。

### 手順

1.  **前提条件**: Dockerと`docker-compose`がインストールされていることを確認してください。
2.  **環境変数の設定**: プロジェクトルートに `.env` ファイルが正しく設定されていることを確認します。（詳細は[開発者セットアップガイド](./developer_setup_guide.md)を参照）
3.  **アプリケーションの起動**: 以下のコマンドをプロジェクトのルートで実行します。

    ```bash
    docker-compose up --build
    ```
    - `--build` フラグは、初回起動時やDockerfileに変更があった場合にイメージを再ビルドします。
    - 起動後、各サービスは以下のポートでアクセス可能になります。
      - **Frontend**: `http://localhost:8501`
      - **Backend**: `http://localhost:8000`
      - **ChromaDB**: `http://localhost:8001`

4.  **アプリケーションの停止**:
    ```bash
    docker-compose down
    ```

---

## 3. 本番環境へのデプロイ

このワークフローでは、アプリケーションを再利用可能なDockerイメージとしてビルドし、コンテナリポジトリ（例: GitHub Container Registry, Docker Hub）にプッシュします。

### 3.1. Makefileによるイメージのビルドとプッシュ

`Makefile` は、イメージのビルドとプッシュのプロセスを簡略化します。

#### ステップ1: イメージのビルド

まず、デプロイしたいアプリケーションのイメージをローカルでビルドします。このコマンドは `:local` というタグが付いたイメージを生成します。

```bash
# バックエンドのイメージをビルド
make build-backend

# フロントエンドのイメージをビルド
make build-frontend
```

#### ステップ2: イメージのプッシュ

次に、ビルドしたイメージにバージョンタグを付け、コンテナリポジトリにプッシュします。`TAG` 環境変数を設定する必要があります。

```bash
# 例: v1.2.3 タグでバックエンドイメージをプッシュ
export TAG=v1.2.3
make push-backend

# 例: Gitのコミットハッシュをタグとしてフロントエンドイメージをプッシュ
export TAG=sha-$(git rev-parse --short HEAD)
make push-frontend
```
- `Makefile` 内の `REGISTRY` 変数を変更することで、プッシュ先のリポジトリをカスタマイズできます。

### 3.2. デプロイ先ホストでの実行

デプロイ先のサーバーで、プッシュしたイメージを使ってアプリケーションを起動します。

1.  **イメージのプル**:
    ```bash
    docker pull ghcr.io/your-org/mochirag/mochirag-backend:v1.2.3
    docker pull ghcr.io/your-org/mochirag/mochirag-frontend:v1.2.3
    ```

2.  **docker-composeの準備**:
    本番環境用の `docker-compose.prod.yml` のようなファイルを用意し、`build`ディレクティブの代わりに `image`ディレクティブでプルしたイメージを指定します。

    **`docker-compose.prod.yml` の例:**
    ```yaml
    services:
      backend:
        image: ghcr.io/your-org/mochirag/mochirag-backend:v1.2.3
        restart: always
        # ... その他の設定 ...
      frontend:
        image: ghcr.io/your-org/mochirag/mochirag-frontend:v1.2.3
        restart: always
        # ... その他の設定 ...
      # ... chromaサービスなども同様に定義 ...
    ```

3.  **サービスの起動**:
    ```bash
    docker-compose -f docker-compose.prod.yml up -d
    ```

### 3.3. CI/CD (GitHub Actions)

リポジトリには `.github/workflows/docker-build-push.yml` にサンプルワークフローが含まれている可能性があります。これを参考に、Gitのタグがプッシュされたタイミングで自動的にイメージをビルド＆プッシュするCI/CDパイプラインを構築することを推奨します。その際、コンテナリポジトリへの認証情報をGitHub ActionsのSecretsに設定する必要があります。
