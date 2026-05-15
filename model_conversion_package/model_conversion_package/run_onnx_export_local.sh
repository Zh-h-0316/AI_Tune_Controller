#!/bin/bash

set -e

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)

if [ ! -x "${ROOT_DIR}/../.venv311/bin/python" ]; then
  echo "找不到 /Users/wuhongfei/workspace/model_install/.venv311/bin/python"
  echo "请先准备 Python 3.11 环境。"
  exit 1
fi

"${ROOT_DIR}/../.venv311/bin/python" "${ROOT_DIR}/scripts/export_to_onnx.py"
