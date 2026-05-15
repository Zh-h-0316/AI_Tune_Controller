#!/usr/bin/env python3
"""
把当前目录里的 PyTorch 模型权重导出为 Android 更容易使用的 ONNX 文件。

支持两种输入格式：
  1. 新格式 - 单个部署合并文件 deploy_checkpoint_*.pt（包含 state_dict + config + stats）
  2. 旧格式 - 分离的 .pth (权重) + .pt (配置) 文件对

使用前提:
1. 建议使用 Python 3.10 或 3.11
2. 安装依赖:
   pip install torch onnx onnxruntime

运行:
   python export_to_onnx.py

输出:
   output_models/adaptive_net_android.onnx
"""

from __future__ import annotations

from pathlib import Path
import shutil

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
INPUT_MODEL_DIR = PACKAGE_ROOT / "input_models"
OUTPUT_MODEL_DIR = PACKAGE_ROOT / "output_models"
OUTPUT_PATH = OUTPUT_MODEL_DIR / "adaptive_net_android.onnx"
FIXED_OUTPUT_PATH = OUTPUT_MODEL_DIR / "adaptive_net_android.fixed.onnx"
ANDROID_GUIDE_PATH = PACKAGE_ROOT / "docs" / "Android_Studio_端侧部署指导_中文.md"

# 旧格式路径（向后兼容）
LEGACY_WEIGHT_PATH = INPUT_MODEL_DIR / "best_adaptive_net_20260316_200459.pth"
LEGACY_CONFIG_PATH = INPUT_MODEL_DIR / "best_config_20260316_200459.pt"


def _find_deploy_checkpoint() -> Path | None:
    """查找 input_models/ 下最新的部署合并文件"""
    candidates = sorted(INPUT_MODEL_DIR.glob("*deploy_checkpoint_*.pt"), reverse=True)
    return candidates[0] if candidates else None


def _shorten_model_stem(model_stem: str) -> str:
    return model_stem.replace("deploy_checkpoint_", "")


def _resolve_source_model_stem(deploy_checkpoint: Path | None) -> str:
    if deploy_checkpoint is not None:
        return _shorten_model_stem(deploy_checkpoint.stem)
    if LEGACY_WEIGHT_PATH.exists():
        return LEGACY_WEIGHT_PATH.stem
    return "unknown_input_model"


def _prepare_deploy_package_dir(source_model_stem: str) -> Path:
    deploy_package_dir = OUTPUT_MODEL_DIR / f"{source_model_stem}_terminal_deploy"
    deploy_package_dir.mkdir(parents=True, exist_ok=True)
    return deploy_package_dir


def _copy_to_deploy_package(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)
        print(f"已写入终端部署包: {dst}")


def main() -> None:
    try:
        import torch
        import torch.nn as nn
        import onnx
    except ImportError as exc:
        raise SystemExit(
            "没有找到 torch。请先用 Python 3.10/3.11 安装: pip install torch onnx onnxruntime"
        ) from exc

    # ====== 加载模型配置和权重 ======
    deploy_checkpoint = _find_deploy_checkpoint()
    source_model_stem = _resolve_source_model_stem(deploy_checkpoint)
    deploy_package_dir = _prepare_deploy_package_dir(source_model_stem)
    deploy_fixed_onnx_path = deploy_package_dir / "adaptive_net_android.fixed.onnx"
    deploy_dynamic_onnx_path = deploy_package_dir / "adaptive_net_android.onnx"
    deploy_meta_path = deploy_package_dir / "model_deploy_meta.json"
    deploy_guide_path = deploy_package_dir / "Android_Studio_端侧部署指导_中文.md"

    OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if deploy_checkpoint is not None:
        # —— 新格式：单个部署合并文件 ——
        print(f"检测到部署合并文件: {deploy_checkpoint.name}")
        bundle = torch.load(deploy_checkpoint, map_location="cpu", weights_only=False)
        config = bundle["config"]
        state_dict = bundle["state_dict"]
        stats = bundle.get("stats", {})
        time_dim = bundle.get("time_dim", stats["time_mean"].shape[0] if "time_mean" in stats else 5)
        print(f"  格式版本: {bundle.get('format_version', '?')}")
        print(f"  控制模式: {config.get('MODE', '?')}")
        print(f"  time_dim: {time_dim}")
        print(f"  参数层数: {len(state_dict)}")
    else:
        # —— 旧格式：.pth + .pt 分离文件 ——
        print("未找到部署合并文件，使用旧格式 (.pth + .pt)...")
        from torch_zip_checkpoint_loader import load_checkpoint

        if not LEGACY_WEIGHT_PATH.exists():
            raise SystemExit(f"没有找到权重文件: {LEGACY_WEIGHT_PATH}")
        if not LEGACY_CONFIG_PATH.exists():
            raise SystemExit(f"没有找到配置文件: {LEGACY_CONFIG_PATH}")

        cfg_bundle = load_checkpoint(LEGACY_CONFIG_PATH)
        config = cfg_bundle["config"]
        state_dict = load_checkpoint(LEGACY_WEIGHT_PATH)
        time_dim = 5  # 旧格式默认

    def _upgrade_legacy_scalar_net_state_dict(raw_state_dict, model_state_dict):
        upgraded = dict(raw_state_dict)
        legacy_w1 = upgraded.get("scalar_net.0.weight")
        legacy_b1 = upgraded.get("scalar_net.0.bias")
        legacy_w2 = upgraded.get("scalar_net.2.weight")
        legacy_b2 = upgraded.get("scalar_net.2.bias")

        target_w1 = model_state_dict.get("scalar_net.0.weight")
        target_b1 = model_state_dict.get("scalar_net.0.bias")
        target_w2 = model_state_dict.get("scalar_net.2.weight")
        target_b2 = model_state_dict.get("scalar_net.2.bias")

        if any(item is None for item in (legacy_w1, legacy_b1, legacy_w2, legacy_b2, target_w1, target_b1, target_w2, target_b2)):
            return upgraded, False

        is_legacy_layout = (
            tuple(legacy_w1.shape) == (32, 2)
            and tuple(legacy_b1.shape) == (32,)
            and tuple(legacy_w2.shape) == (32, 32)
            and tuple(legacy_b2.shape) == (32,)
            and tuple(target_w1.shape) == (64, 7)
            and tuple(target_b1.shape) == (64,)
            and tuple(target_w2.shape) == (32, 64)
            and tuple(target_b2.shape) == (32,)
        )
        if not is_legacy_layout:
            return upgraded, False

        new_w1 = target_w1.clone()
        new_b1 = target_b1.clone()
        new_w2 = target_w2.clone()
        new_b2 = target_b2.clone()

        new_w1.zero_()
        new_b1.zero_()
        new_w2.zero_()
        new_b2.copy_(legacy_b2)

        new_w1[:32, :2].copy_(legacy_w1)
        new_b1[:32].copy_(legacy_b1)
        new_w2[:, :32].copy_(legacy_w2)

        upgraded["scalar_net.0.weight"] = new_w1
        upgraded["scalar_net.0.bias"] = new_b1
        upgraded["scalar_net.2.weight"] = new_w2
        upgraded["scalar_net.2.bias"] = new_b2
        return upgraded, True

    class AdaptiveNet(nn.Module):
        def __init__(self, cfg: dict, input_time_dim: int = 5) -> None:
            super().__init__()
            hidden_size = int(cfg["HIDDEN_SIZE"])
            lstm_layers = int(cfg["LSTM_LAYERS"])
            lstm_dropout = float(cfg["LSTM_DROPOUT"]) if lstm_layers > 1 else 0.0
            mlp_hidden = list(cfg["MLP_HIDDEN"])
            mlp_dropout = float(cfg["MLP_DROPOUT"])
            use_attention = bool(cfg["USE_ATTENTION"])
            speed_feature_gain = float(cfg.get("SPEED_FEATURE_GAIN", 1.8))

            self.use_attention = use_attention
            self.speed_feature_gain = speed_feature_gain

            self.lstm = nn.LSTM(
                input_size=input_time_dim,
                hidden_size=hidden_size,
                num_layers=lstm_layers,
                dropout=lstm_dropout,
                batch_first=True,
            )

            self.attention = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.Tanh(),
                nn.Linear(hidden_size // 2, 1),
            )

            scalar_feature_dim = 7
            self.scalar_net = nn.Sequential(
                nn.Linear(scalar_feature_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
            )

            fused_size = hidden_size + 32
            self.mlp_backbone = nn.Sequential(
                nn.Linear(fused_size, mlp_hidden[0]),
                nn.BatchNorm1d(mlp_hidden[0]),
                nn.ReLU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(mlp_hidden[0], mlp_hidden[1]),
                nn.BatchNorm1d(mlp_hidden[1]),
                nn.ReLU(),
                nn.Dropout(mlp_dropout),
            )
            self.residual_proj = nn.Linear(fused_size, mlp_hidden[1])
            self.output_head = nn.Linear(mlp_hidden[1], 4)

        def _build_speed_aware_scalar_features(self, scalar_features):
            speed = scalar_features[:, 0:1]
            wheelbase = scalar_features[:, 1:2]
            speed_sq = speed * speed
            speed_wheelbase = speed * wheelbase
            speed_tanh = torch.tanh(speed * self.speed_feature_gain)
            low_speed_focus = torch.exp(-torch.abs(speed) * self.speed_feature_gain)
            speed_ratio = speed / (1.0 + torch.abs(speed))
            return torch.cat(
                [speed, wheelbase, speed_sq, speed_wheelbase, speed_tanh, low_speed_focus, speed_ratio],
                dim=1,
            )

        def forward(self, time_features, scalar_features):
            lstm_out, _ = self.lstm(time_features)
            if self.use_attention:
                attn_scores = self.attention(lstm_out)
                attn_weights = torch.softmax(attn_scores, dim=1)
                time_embed = (attn_weights * lstm_out).sum(dim=1)
            else:
                time_embed = lstm_out[:, -1, :]

            scalar_embed = self.scalar_net(self._build_speed_aware_scalar_features(scalar_features))
            fused = torch.cat([time_embed, scalar_embed], dim=1)
            hidden = self.mlp_backbone(fused) + self.residual_proj(fused)
            return self.output_head(hidden)

    model = AdaptiveNet(config, input_time_dim=time_dim)
    upgraded_state_dict, upgraded_legacy = _upgrade_legacy_scalar_net_state_dict(state_dict, model.state_dict())
    model.load_state_dict(upgraded_state_dict, strict=True)
    if upgraded_legacy:
        print("检测到旧版 scalar_net 权重布局，已自动升级后再导出 ONNX。")
    model.eval()

    seq_len = int(config["SEQ_LEN"])
    dummy_time = torch.randn(1, seq_len, time_dim, dtype=torch.float32)
    dummy_scalar = torch.randn(1, 2, dtype=torch.float32)

    # 如果这里两组不同输入给出完全相同输出，说明权重或模型结构仍然有问题。
    with torch.no_grad():
        probe_a = model(torch.zeros(1, seq_len, time_dim), torch.tensor([[1.0, 2.0]], dtype=torch.float32))
        probe_b = model(torch.ones(1, seq_len, time_dim), torch.tensor([[5.0, -3.0]], dtype=torch.float32))
        if torch.allclose(probe_a, probe_b):
            raise SystemExit(
                "导出终止: 当前加载出来的模型对不同输入输出完全相同，说明权重解析或模型结构仍有问题。"
            )

    torch.onnx.export(
        model,
        (dummy_time, dummy_scalar),
        OUTPUT_PATH.as_posix(),
        input_names=["time_features", "scalar_features"],
        output_names=["output"],
        dynamic_axes={
            "time_features": {0: "batch_size"},
            "scalar_features": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )

    # 某些导出器会额外生成 .onnx.data 文件。这里强制合并成单文件，
    # 这样 Android 端只需要拷贝一个 .onnx 到 assets 即可。
    merged_model = onnx.load(OUTPUT_PATH.as_posix(), load_external_data=True)
    onnx.save_model(merged_model, OUTPUT_PATH.as_posix(), save_as_external_data=False)
    external_data_path = OUTPUT_PATH.with_suffix(OUTPUT_PATH.suffix + ".data")
    if external_data_path.exists():
        external_data_path.unlink()

    fixed_model = onnx.load(OUTPUT_PATH.as_posix())
    for value_info in list(fixed_model.graph.input) + list(fixed_model.graph.output):
        for dim in value_info.type.tensor_type.shape.dim:
            if dim.HasField("dim_param") and dim.dim_param == "batch_size":
                dim.ClearField("dim_param")
                dim.dim_value = 1
    onnx.save_model(fixed_model, FIXED_OUTPUT_PATH.as_posix(), save_as_external_data=False)

    _copy_to_deploy_package(OUTPUT_PATH, deploy_dynamic_onnx_path)
    _copy_to_deploy_package(FIXED_OUTPUT_PATH, deploy_fixed_onnx_path)

    print(f"导出完成: {OUTPUT_PATH}")
    print(f"固定输入形状版本: {FIXED_OUTPUT_PATH}")
    print(f"输入1: time_features, 形状类似 [1, {seq_len}, {time_dim}]")
    print("输入2: scalar_features, 形状类似 [1, 2]")
    print("输出 : output, 形状类似 [1, 4]")

    # 导出 stats 和 config 为 JSON（供 Android 端数据预处理使用）
    if stats:
        import json
        deploy_meta = {
            "model_config": {
                "MODE": str(config.get("MODE", "A")),
                "SEQ_LEN": int(config.get("SEQ_LEN", 10)),
                "HIDDEN_SIZE": int(config.get("HIDDEN_SIZE", 64)),
                "time_dim": int(time_dim),
                "scalar_dim": 2,
                "USE_ATTENTION": bool(config.get("USE_ATTENTION", True)),
                "MODE_A_ALPHA_RANGE": list(config.get("MODE_A_ALPHA_RANGE", [0.3, 1.8])),
                "MODE_A_BETA_SCALE": float(config.get("MODE_A_BETA_SCALE", 0.25)),
                "MODE_D_DELTA_SCALE": float(config.get("MODE_D_DELTA_SCALE", 0.30)),
            },
            "normalization_stats": {
                "time_mean": stats["time_mean"].tolist() if hasattr(stats["time_mean"], "tolist") else list(stats["time_mean"]),
                "time_std": stats["time_std"].tolist() if hasattr(stats["time_std"], "tolist") else list(stats["time_std"]),
                "scalar_mean": stats["scalar_mean"].tolist() if hasattr(stats["scalar_mean"], "tolist") else list(stats["scalar_mean"]),
                "scalar_std": stats["scalar_std"].tolist() if hasattr(stats["scalar_std"], "tolist") else list(stats["scalar_std"]),
            },
        }
        meta_path = OUTPUT_MODEL_DIR / "model_deploy_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(deploy_meta, f, indent=2, ensure_ascii=False)
        print(f"模型部署元数据已导出: {meta_path}")
        _copy_to_deploy_package(meta_path, deploy_meta_path)
    else:
        print("⚠️  旧格式不包含 stats，请手动提供 model_deploy_meta.json 给 Android 端。")

    _copy_to_deploy_package(ANDROID_GUIDE_PATH, deploy_guide_path)
    print(f"终端部署包目录已更新: {deploy_package_dir}")


if __name__ == "__main__":
    main()
