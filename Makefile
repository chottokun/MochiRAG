# Makefile for building and pushing MochiRAG images

REGISTRY ?= ghcr.io/${GITHUB_OWNER:-your-org}/${GITHUB_REPO:-MochiRAG}
BACKEND_IMAGE = $(REGISTRY)/mochirag-backend
FRONTEND_IMAGE = $(REGISTRY)/mochirag-frontend

.PHONY: build-backend build-frontend push-backend push-frontend build-all push-all

build-backend:
	docker build --progress=plain -f Dockerfile.backend -t $(BACKEND_IMAGE):local .

build-frontend:
	docker build --progress=plain -f Dockerfile.frontend -t $(FRONTEND_IMAGE):local .

push-backend:
	docker build -f Dockerfile.backend -t $(BACKEND_IMAGE):$(TAG) .
	docker push $(BACKEND_IMAGE):$(TAG)

push-frontend:
	docker build -f Dockerfile.frontend -t $(FRONTEND_IMAGE):$(TAG) .
	docker push $(FRONTEND_IMAGE):$(TAG)

build-all: build-backend build-frontend

push-all: push-backend push-frontend
