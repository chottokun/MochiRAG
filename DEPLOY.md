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

1. Build locally (development):

   make build-backend
   make build-frontend

2. Push an image to GHCR (example):

   TAG=sha-$(git rev-parse --short HEAD) \
     OWNER=your-username \
     REPO=your-repo \
     IMAGE=mochirag-backend \
     REGISTRY=ghcr.io \
     ./scripts/push-image.sh

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
