from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from paths import ensure_import_paths


ensure_import_paths()


@lru_cache(maxsize=1)
def _converter_and_union_idx():
    from uv_conversion.universal_UV_converter import UniTacConverter
    from uv_conversion.utils.glove_hand import HandType

    converter = UniTacConverter(return_union=False)
    try:
        from uv_conversion.universal_UV_converter import UNION_IDX

        union_idx = np.asarray(UNION_IDX, dtype=np.int64)
    except Exception:
        valid = []
        for hand_type in (HandType.RIGHT, HandType.LEFT):
            uv891 = converter.glove137_to_uv891(
                np.ones(137, dtype=np.float32),
                hand_type=hand_type,
                mask=True,
            )
            valid.append(np.where(np.asarray(uv891).reshape(-1) != -1)[0])
        union_idx = np.union1d(valid[0], valid[1]).astype(np.int64)
    if union_idx.shape != (351,):
        raise RuntimeError(f"Expected 351 UV-union indices, got {union_idx.shape}.")
    return converter, union_idx, HandType


def tactile_to_351(value: Any, side: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((351,), dtype=np.float32)
    if arr.ndim == 1:
        arr2 = arr[None, :]
        squeeze = True
    else:
        arr2 = arr.reshape((-1, arr.shape[-1]))
        squeeze = False

    if arr2.shape[-1] == 351:
        valid = np.isfinite(arr2) & (arr2 != -1.0)
        out = np.where(valid, np.clip(arr2, 0.0, 1.0), 0.0).astype(np.float32)
    elif arr2.shape[-1] == 1062:
        converter, union_idx, hand_type_cls = _converter_and_union_idx()
        hand_type = hand_type_cls.LEFT if str(side).lower() == "left" else hand_type_cls.RIGHT
        safe = np.nan_to_num(arr2, nan=0.0, posinf=0.0, neginf=0.0)
        uv891 = converter.inspire1062_to_uv891(
            safe,
            hand_type=hand_type,
            correct=True,
            mask=True,
        )
        raw = np.asarray(uv891, dtype=np.float32)[..., union_idx]
        valid = np.isfinite(raw) & (raw != -1.0)
        out = np.where(valid, np.clip(raw, 0.0, 1.0), 0.0).astype(np.float32)
    else:
        raise ValueError(f"{side} tactile expected 351D or 1062D, got {arr2.shape[-1]}D.")

    return out[0] if squeeze else out.reshape(arr.shape[:-1] + (351,))


class TactileHistory:
    def __init__(self, history_size: int, history_stride: int):
        self.history_size = max(1, int(history_size))
        self.history_stride = max(1, int(history_stride))
        self._buffers: dict[str, list[np.ndarray]] = {"left": [], "right": []}

    def append(self, side: str, tactile: Any) -> None:
        side = str(side).lower()
        vec = tactile_to_351(tactile, side)
        buf = self._buffers.setdefault(side, [])
        buf.append(vec.astype(np.float32, copy=True))
        keep = self.history_size * self.history_stride + 1
        if len(buf) > keep:
            del buf[: len(buf) - keep]

    def stack(self, side: str) -> np.ndarray:
        side = str(side).lower()
        buf = self._buffers.setdefault(side, [])
        if not buf:
            return np.zeros((self.history_size, 351), dtype=np.float32)
        samples = []
        for offset in range(self.history_size - 1, -1, -1):
            idx = len(buf) - 1 - offset * self.history_stride
            idx = max(0, min(idx, len(buf) - 1))
            samples.append(buf[idx])
        return np.stack(samples, axis=0).astype(np.float32)

