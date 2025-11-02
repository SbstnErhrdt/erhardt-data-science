#!/bin/bash
set -euo pipefail

image="registry.erhardt.net/erhardt-encoder-paecter"
torch_version="${TORCH_VERSION:-2.2.2}"
torch_index_url="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

tag_sha="$(git rev-parse --short HEAD 2>/dev/null || echo manual)"
tag_date="$(date +%Y%m%d-%H%M%S)"
tags="-t ${image}:latest -t ${image}:${tag_date}-${tag_sha}"

build_args=(
  "--build-arg" "TORCH_VERSION=${torch_version}"
  "--build-arg" "TORCH_INDEX_URL=${torch_index_url}"
)

echo "==> Ensuring buildx is ready"
docker buildx inspect multi >/dev/null 2>&1 || docker buildx create --name multi --use
docker buildx use multi

echo "==> Logging in to registry"
cat .pw-docker-registry | docker login --username docker --password-stdin registry.erhardt.net

echo "==> Building & pushing CUDA-enabled image (linux/amd64 only)"
docker buildx build \
  --platform linux/amd64 \
  "${build_args[@]}" \
  ${tags} \
  --push \
  -f encoder-paecter/Dockerfile \
  .

echo "==> Image pushed: ${image}"

echo "==> Rolling out on server"
ssh server.02 'cd /home/docker-compose/encoder-paecter && sh deploy.sh'

echo "==> Done."
