from __future__ import annotations

import math
from typing import Sequence


def finite_tuple(
    values: Sequence[float],
    length: int,
    name: str,
) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != length or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain {length} finite values.")
    return result


__all__ = ["finite_tuple"]
