#!/usr/bin/env bash
set -eu

if [ -z "${TAG:-}" ]; then
  echo "Usage: TAG=<tag> $0" 1>&2
  exit 2
fi

REGISTRY=${REGISTRY:-ghcr.io}
OWNER=${OWNER:-your-org}
REPO=${REPO:-MochiRAG}
IMAGE=${IMAGE:-mochirag-backend}

# Determine Dockerfile based on image name
DOCKERFILE="Dockerfile.backend"
if [ "$IMAGE" = "mochirag-frontend" ]; then
  DOCKERFILE="Dockerfile.frontend"
fi

FULL_NAME="$REGISTRY/$OWNER/$REPO/$IMAGE:$TAG"

echo "Building $FULL_NAME using $DOCKERFILE"
docker build -f "$DOCKERFILE" -t "$FULL_NAME" .

echo "Pushing $FULL_NAME"
docker push "$FULL_NAME"
