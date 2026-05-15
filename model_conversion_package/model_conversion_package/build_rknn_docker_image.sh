#!/bin/bash

set -e

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)

"/Applications/Docker.app/Contents/Resources/bin/docker" build \
  -t local/rknn-toolkit2:2.3.2-runtime \
  "${ROOT_DIR}/docker/rknn"

echo "Docker image ready: local/rknn-toolkit2:2.3.2-runtime"
