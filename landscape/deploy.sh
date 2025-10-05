#!/bin/bash
set -euo pipefail

image="registry.erhardt.net/erhardt-patent-landscape-worker"

# Optional: tag with git SHA/date
tag_sha="$(git rev-parse --short HEAD 2>/dev/null || echo manual)"
tag_date="$(date +%Y%m%d-%H%M%S)"
tags="-t ${image}:latest -t ${image}:${tag_date}-${tag_sha}"

echo "==> Ensuring buildx is ready"
docker buildx inspect multi >/dev/null 2>&1 || docker buildx create --name multi --use
docker buildx use multi

echo "==> Logging in to registry"
cat .pw-docker-registry | docker login --username docker --password-stdin registry.erhardt.net

echo "==> Building & pushing multi-arch image (amd64, arm64)"
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  ${tags} \
  --push \
  .


echo "==> Image pushed: ${image}"

echo "==> Rolling out on server"
ssh server.02 'cd /home/docker-compose/patent-landscaping-worker && sh deploy.sh'

echo "==> Done."
