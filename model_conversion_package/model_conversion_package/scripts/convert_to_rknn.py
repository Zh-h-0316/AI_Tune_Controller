#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
用 RKNN Toolkit2 把当前 ONNX 模型转换成 Rockchip NPU 可用的 .rknn 文件。

推荐在 Docker 镜像 local/rknn-toolkit2:2.3.2-runtime 中运行。
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PACKAGE_ROOT / "output_models"
INPUT_MODEL_DIR = PACKAGE_ROOT / "input_models"
ONNX_PATH = MODEL_DIR / "adaptive_net_android.fixed.onnx"
RKNN_PATH = MODEL_DIR / "adaptive_net_android.rknn"
TARGET_PLATFORM = "rk3568"


def _find_deploy_checkpoint() -> Path | None:
    candidates = sorted(INPUT_MODEL_DIR.glob("*deploy_checkpoint_*.pt"), reverse=True)
    return candidates[0] if candidates else None


def _shorten_model_stem(model_stem: str) -> str:
    return model_stem.replace("deploy_checkpoint_", "")


def _resolve_deploy_package_dir() -> Path:
    deploy_checkpoint = _find_deploy_checkpoint()
    if deploy_checkpoint is not None:
        source_model_stem = _shorten_model_stem(deploy_checkpoint.stem)
    else:
        source_model_stem = "best_adaptive_net_20260316_200459"
    deploy_package_dir = MODEL_DIR / f"{source_model_stem}_terminal_deploy"
    deploy_package_dir.mkdir(parents=True, exist_ok=True)
    return deploy_package_dir


def _copy_rknn_to_deploy_package(src_path: Path, dst_path: Path) -> None:
    try:
        shutil.copy2(src_path, dst_path)
    except PermissionError as exc:
        # WSL 写入 /mnt/* 挂载盘时，文件内容可能已复制成功，但元数据同步会被拒绝。
        shutil.copyfile(src_path, dst_path)
        print(
            f"警告: 复制 RKNN 元数据失败，已回退为普通文件复制: {exc}",
            file=sys.stderr,
        )


def main() -> int:
    try:
        from rknn.api import RKNN
    except Exception as exc:  # pragma: no cover
        print(f"无法导入 RKNN Toolkit2: {exc}", file=sys.stderr)
        return 1

    if not ONNX_PATH.exists():
        print(f"没有找到 ONNX 文件: {ONNX_PATH}", file=sys.stderr)
        return 1

    deploy_package_dir = _resolve_deploy_package_dir()
    deploy_rknn_path = deploy_package_dir / "adaptive_net_android.rknn"

    for stale_path in (RKNN_PATH, deploy_rknn_path):
        if stale_path.exists():
            stale_path.unlink()
            print(f"已清理旧RKNN文件: {stale_path}")

    rknn = RKNN(verbose=True)

    try:
        print(f"正在配置 RKNN，目标平台: {TARGET_PLATFORM}")
        ret = rknn.config(
            target_platform=TARGET_PLATFORM,
            quantized_dtype="asymmetric_quantized-8",
            optimization_level=3,
            disable_rules=["merge_parallel_op_after_split"],
        )
        if ret != 0:
            print(f"rknn.config 失败，返回值: {ret}", file=sys.stderr)
            return ret

        print(f"正在加载 ONNX: {ONNX_PATH}")
        ret = rknn.load_onnx(
            model=ONNX_PATH.as_posix(),
        )
        if ret != 0:
            print(f"rknn.load_onnx 失败，返回值: {ret}", file=sys.stderr)
            return ret

        print("正在构建 RKNN 模型，不做量化...")
        ret = rknn.build(do_quantization=False)
        if ret != 0:
            print(f"rknn.build 失败，返回值: {ret}", file=sys.stderr)
            return ret

        print(f"正在导出 RKNN 文件: {RKNN_PATH}")
        ret = rknn.export_rknn(RKNN_PATH.as_posix())
        if ret != 0:
            print(f"rknn.export_rknn 失败，返回值: {ret}", file=sys.stderr)
            return ret

        _copy_rknn_to_deploy_package(RKNN_PATH, deploy_rknn_path)
        print(f"终端部署 RKNN 文件已写入: {deploy_rknn_path}")

        print("转换成功")
        print(RKNN_PATH)
        print(f"终端部署包目录: {deploy_package_dir}")
        return 0
    finally:
        rknn.release()


if __name__ == "__main__":
    raise SystemExit(main())
