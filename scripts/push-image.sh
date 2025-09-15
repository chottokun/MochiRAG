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

FULL_NAME="$REGISTRY/$OWNER/$REPO/$IMAGE:$TAG"

echo "Building $FULL_NAME"
docker build -f Dockerfile.backend -t "$FULL_NAME" .

echo "Pushing $FULL_NAME"
docker push "$FULL_NAME"
