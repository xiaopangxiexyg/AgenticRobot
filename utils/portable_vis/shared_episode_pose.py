"""Shared offline episode selection and pose math for portable visualizers."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        payload = json.loads(text)
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _episode_index_from_path(path: Path) -> int | None:
    stem = path.stem
    prefix = "episode_"
    if not stem.startswith(prefix):
        return None
    suffix = stem[len(prefix) :]
    return int(suffix) if suffix.isdigit() else None


def _episode_id_from_index(index: int | None, fallback: str) -> str:
    if index is None:
        return fallback
    return f"episode_{int(index):06d}"


def _episode_matches(record: Mapping[str, Any], token: str, path: Path) -> bool:
    normalized = str(token).strip()
    if not normalized:
        return False
    episode_id = str(record.get("episode_id") or path.stem)
    if normalized == episode_id or normalized == path.stem:
        return True
    try:
        requested_index = int(normalized)
    except ValueError:
        return False
    record_index = record.get("episode_index")
    if record_index is None:
        record_index = _episode_index_from_path(path)
    try:
        return int(record_index) == requested_index
    except (TypeError, ValueError):
        return False


def _vector_or_none(value: Any, *, length: int) -> np.ndarray | None:
    if value is None:
        return None
    try:
        vector = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if vector.shape != (int(length),) or not np.all(np.isfinite(vector)):
        return None
    return vector


def _quat_xyzw_to_matrix(quat: Sequence[float]) -> np.ndarray | None:
    q = np.asarray(quat, dtype=float).reshape(-1)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        return None
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        return None
    x, y, z, w = (q / norm).tolist()
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def _rotation_matrix_to_quat_xyzw(matrix: np.ndarray) -> np.ndarray | None:
    m = np.asarray(matrix, dtype=float).reshape(3, 3)
    if not np.all(np.isfinite(m)):
        return None
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        axis = int(np.argmax(np.diag(m)))
        if axis == 0:
            s = math.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) * 2.0
            if s <= 1e-12:
                return None
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif axis == 1:
            s = math.sqrt(max(0.0, 1.0 + m[1, 1] - m[0, 0] - m[2, 2])) * 2.0
            if s <= 1e-12:
                return None
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(max(0.0, 1.0 + m[2, 2] - m[0, 0] - m[1, 1])) * 2.0
            if s <= 1e-12:
                return None
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    quat = np.asarray([x, y, z, w], dtype=float)
    norm = float(np.linalg.norm(quat))
    return quat / norm if norm > 1e-12 else None


def _downsample_indices(count: int, max_frames: int) -> list[int]:
    if max_frames <= 0 or count <= max_frames:
        return list(range(count))
    return sorted({int(i) for i in np.linspace(0, count - 1, num=max_frames, dtype=int).tolist()})


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None
