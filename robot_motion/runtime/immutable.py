"""Deep detach/freeze helpers for admitted runtime facts."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


def _freeze_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("Object arrays cannot cross an immutable runtime boundary.")
    contiguous = np.ascontiguousarray(array)
    immutable_bytes = contiguous.tobytes(order="C")
    frozen = np.frombuffer(immutable_bytes, dtype=contiguous.dtype).reshape(contiguous.shape)
    return frozen


def deep_freeze(value: Any) -> Any:
    """Detach a supported value and recursively make it immutable."""

    if isinstance(value, np.ndarray):
        return _freeze_array(value)
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(deep_freeze(item) for item in value)
    if isinstance(value, (str, bytes, int, float, bool, type(None), np.generic)):
        return value.item() if isinstance(value, np.generic) else value
    raise TypeError(f"Unsupported value at immutable runtime boundary: {type(value).__name__}.")


def deep_thaw(value: Any) -> Any:
    """Return a mutable, detached copy suitable for an external consumer."""

    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {str(key): deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(deep_thaw(item) for item in value)
    return value


__all__ = ["deep_freeze", "deep_thaw"]
