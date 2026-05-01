# -*- coding: UTF-8 -*-
import hashlib
import json

import numpy as np


def _normalize_value(value):
    if isinstance(value, dict):
        return {str(k): _normalize_value(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.tolist(),
        }
    if hasattr(value, "detach") and callable(value.detach):
        return _normalize_value(value.detach().cpu().numpy())
    if hasattr(value, "cpu") and callable(value.cpu) and hasattr(value, "numpy"):
        return _normalize_value(value.cpu().numpy())
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, bytes):
        return value.hex()
    return value


def canonical_json(obj):
    return json.dumps(
        _normalize_value(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_json(obj):
    return sha256_text(canonical_json(obj))


def sha256_array(arr):
    array = np.asarray(arr)
    payload = {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "values": array.tolist(),
    }
    return sha256_json(payload)


def sha256_model_state_dict(state_dict):
    normalized = {str(k): _normalize_value(v) for k, v in sorted(state_dict.items())}
    return sha256_json(normalized)


def sha256_torch_checkpoint(path):
    return sha256_file(path)
