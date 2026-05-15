#!/bin/bash

set -e

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)

"/Applications/Docker.app/Contents/Resources/bin/docker" run --rm \
  -v "${ROOT_DIR}:/workspace" \
  local/rknn-toolkit2:2.3.2-runtime \
  /bin/sh -lc '
    apt-get update >/dev/null &&
    DEBIAN_FRONTEND=noninteractive apt-get install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 >/dev/null &&
    cd /workspace &&
    python3 scripts/export_to_onnx.py &&
    python3 scripts/convert_to_rknn.py
  '

echo "Conversion finished."
