#!/usr/bin/env python3
"""
读取新版 PyTorch zip checkpoint，但绕开当前环境里 torch.load 对部分浮点权重的错误解析。

支持当前项目里的:
- state_dict .pth
- 包含 config/stats 的 .pt
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import io
import pickle
import zipfile

import numpy as np
import torch

PACKED_INT16_SCALE = 262144.0


@dataclass(frozen=True)
class StorageRef:
    storage_type: str
    key: str
    location: str
    size: int


@dataclass(frozen=True)
class TensorRef:
    storage: StorageRef
    storage_offset: int
    size: tuple[int, ...]
    stride: tuple[int, ...]


class _RestrictedUnpickler(pickle.Unpickler):
    def persistent_load(self, pid):
        if isinstance(pid, tuple) and pid and pid[0] == "storage":
            _, storage_type, key, location, size = pid
            return StorageRef(str(storage_type), str(key), str(location), int(size))
        return pid

    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return OrderedDict

        if module == "torch._utils" and name in ("_rebuild_tensor", "_rebuild_tensor_v2"):
            def rebuild_tensor(storage, storage_offset, size, stride, requires_grad=False, backward_hooks=None, metadata=None):
                return TensorRef(
                    storage=storage,
                    storage_offset=int(storage_offset),
                    size=tuple(int(x) for x in size),
                    stride=tuple(int(x) for x in stride),
                )

            return rebuild_tensor

        if module == "torch._utils" and name == "_rebuild_parameter":
            def rebuild_parameter(data, requires_grad, backward_hooks):
                return data

            return rebuild_parameter

        if module == "torch" and name.endswith("Storage"):
            return f"{module}.{name}"

        if module in ("builtins", "__builtin__") and name == "set":
            return set

        raise pickle.UnpicklingError(f"Unsupported global during checkpoint load: {module}.{name}")


def _dtype_for_storage(storage_type: str):
    mapping = {
        "torch.FloatStorage": np.dtype("<f4"),
        "torch.DoubleStorage": np.dtype("<f8"),
        "torch.HalfStorage": np.dtype("<f2"),
        "torch.BFloat16Storage": np.dtype("<u2"),
        "torch.LongStorage": np.dtype("<i8"),
        "torch.IntStorage": np.dtype("<i4"),
        "torch.ShortStorage": np.dtype("<i2"),
        "torch.CharStorage": np.dtype("i1"),
        "torch.ByteStorage": np.dtype("u1"),
        "torch.BoolStorage": np.dtype("?"),
    }
    if storage_type not in mapping:
        raise ValueError(f"Unsupported storage type: {storage_type}")
    return mapping[storage_type]


def _torch_dtype_for_storage(storage_type: str):
    mapping = {
        "torch.FloatStorage": torch.float32,
        "torch.DoubleStorage": torch.float64,
        "torch.HalfStorage": torch.float16,
        "torch.BFloat16Storage": torch.bfloat16,
        "torch.LongStorage": torch.int64,
        "torch.IntStorage": torch.int32,
        "torch.ShortStorage": torch.int16,
        "torch.CharStorage": torch.int8,
        "torch.ByteStorage": torch.uint8,
        "torch.BoolStorage": torch.bool,
    }
    if storage_type not in mapping:
        raise ValueError(f"Unsupported storage type: {storage_type}")
    return mapping[storage_type]


def _maybe_decode_packed_int16(raw: bytes, storage_type: str):
    if storage_type != "torch.FloatStorage":
        return None

    fp32_values = np.frombuffer(raw, dtype=np.dtype("<f4"))
    fp32_max = float(np.max(np.abs(fp32_values))) if fp32_values.size else 0.0
    if fp32_max > 1e-20:
        return None

    u16 = np.frombuffer(raw, dtype=np.dtype("<u2"))
    high_words = u16[1::2]
    signpad_ratio = float(np.mean((high_words == 0) | (high_words == 32768))) if high_words.size else 0.0
    if signpad_ratio < 0.75:
        return None

    low_words = np.frombuffer(raw, dtype=np.dtype("<i2"))[0::2].astype(np.float32)
    decoded = low_words / PACKED_INT16_SCALE
    return torch.from_numpy(decoded.copy()).to(torch.float32)


def _read_pickle_root(path: Path):
    with zipfile.ZipFile(path) as zf:
        prefix = next(name.split("/")[0] for name in zf.namelist() if name.endswith("data.pkl"))
        raw_pickle = zf.read(f"{prefix}/data.pkl")
    return prefix, _RestrictedUnpickler(io.BytesIO(raw_pickle)).load()


def _load_storage_arrays(path: Path, prefix: str, root) -> dict[str, torch.Tensor]:
    cache: dict[str, torch.Tensor] = {}

    def visit(obj):
        if isinstance(obj, TensorRef):
            key = obj.storage.key
            if key not in cache:
                with zipfile.ZipFile(path) as zf:
                    raw = zf.read(f"{prefix}/data/{key}")
                np_dtype = _dtype_for_storage(obj.storage.storage_type)
                torch_dtype = _torch_dtype_for_storage(obj.storage.storage_type)

                packed_int16 = _maybe_decode_packed_int16(raw, obj.storage.storage_type)
                if packed_int16 is not None:
                    tensor = packed_int16
                elif obj.storage.storage_type == "torch.BFloat16Storage":
                    words = np.frombuffer(raw, dtype=np_dtype).copy()
                    ints = torch.from_numpy(words.astype(np.int32)) << 16
                    tensor = ints.view(torch.float32).to(torch_dtype)
                else:
                    arr = np.frombuffer(raw, dtype=np_dtype).copy()
                    tensor = torch.from_numpy(arr).to(torch_dtype)

                cache[key] = tensor
        elif isinstance(obj, dict):
            for value in obj.values():
                visit(value)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                visit(value)

    visit(root)
    return cache


def _materialize(obj, storage_cache: dict[str, torch.Tensor]):
    if isinstance(obj, TensorRef):
        base = storage_cache[obj.storage.key]
        tensor = torch.as_strided(
            base,
            size=obj.size,
            stride=obj.stride,
            storage_offset=obj.storage_offset,
        ).clone()
        if obj.size == ():
            return tensor.reshape(())
        return tensor

    if isinstance(obj, OrderedDict):
        return OrderedDict((key, _materialize(value, storage_cache)) for key, value in obj.items())

    if isinstance(obj, dict):
        return {key: _materialize(value, storage_cache) for key, value in obj.items()}

    if isinstance(obj, list):
        return [_materialize(value, storage_cache) for value in obj]

    if isinstance(obj, tuple):
        return tuple(_materialize(value, storage_cache) for value in obj)

    return obj


def load_checkpoint(path: str | Path):
    checkpoint_path = Path(path)
    prefix, root = _read_pickle_root(checkpoint_path)
    storages = _load_storage_arrays(checkpoint_path, prefix, root)
    return _materialize(root, storages)
