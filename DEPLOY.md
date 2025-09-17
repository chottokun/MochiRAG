MochiRAG Deployment Guide

This document explains how to build, tag, push Docker images and deploy the MochiRAG backend and frontend. It covers both local and CI-based workflows.

Prerequisites

- Docker and docker-compose installed on the deployment host.
- A container registry (GHCR or Docker Hub) and a token with push rights.
- On CI: repository secrets set. For GHCR use `GHCR_PAT` (a personal access token with `write:packages`) or use `GITHUB_TOKEN` for simple cases.

Recommended tags

- Commit SHA short: `$(git rev-parse --short HEAD)`
- Semantic version tags for releases: `v1.2.3`

Local build & push

This workflow uses the `Makefile` to simplify building and pushing images.

1.  **Build Local Images:**
    First, build the images you want to deploy. This will create images tagged with `:local`.

    ```bash
    # Build the backend image
    make build-backend

    # Build the frontend image
    make build-frontend
    ```

2.  **Push Images to a Registry:**
    Next, use the `push-*` targets to tag the local images and push them to your container registry.

    You must set the `TAG` environment variable. You can also override `REGISTRY`, `OWNER`, and `REPO` if needed.

    ```bash
    # Example: Push the backend image with a specific tag
    export TAG=v1.2.3
    make push-backend

    # Example: Push the frontend image with a tag based on the git commit
    export TAG=sha-$(git rev-parse --short HEAD)
    make push-frontend
    ```

CI (GitHub Actions)

A sample workflow is included at `.github/workflows/docker-build-push.yml`. It builds backend and frontend images and pushes them to GHCR. Ensure `GHCR_PAT` is configured in repository secrets.

Deploy to host with docker-compose

1. On the deployment host pull images:

   docker pull ghcr.io/<owner>/MochiRAG/mochirag-backend:<tag>
   docker pull ghcr.io/<owner>/MochiRAG/mochirag-frontend:<tag>

2. Update `docker-compose.yml` to reference the pulled images or set environment variables accordingly.

3. Restart the services:

   docker-compose down
   docker-compose up -d

Rollback

- To rollback to a previous tag, change the tag in `docker-compose.yml` and `docker-compose pull` then `docker-compose up -d`.

Secrets and CI

- GHCR: create a personal access token (scope: write:packages, delete:packages optionally) and add it to the repository secrets as `GHCR_PAT`.
- Docker Hub: create a repository access token and store it in `DOCKERHUB_TOKEN` and `DOCKERHUB_USER`.

Notes

- If your registry requires additional steps, adapt the CI workflow's login step accordingly.
- For zero-downtime deploys consider using a rolling update strategy or orchestrator (k8s, Nomad, etc.).
