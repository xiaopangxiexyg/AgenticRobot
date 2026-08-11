"""Validation primitives shared by trigger contracts."""

from __future__ import annotations

import math


def required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def finite(value: object, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


def positive(value: object, *, name: str) -> float:
    normalized = finite(value, name=name)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return normalized


def nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def lineage(
    ids: tuple[str, ...],
    hashes: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    normalized_ids = tuple(required(value, name="source_fact_id") for value in ids)
    normalized_hashes = tuple(
        required(value, name="source_fact_hash") for value in hashes
    )
    if not normalized_ids or len(normalized_ids) != len(normalized_hashes):
        raise ValueError(
            "source fact ids/hashes must be non-empty and have equal length."
        )
    if len(normalized_ids) != len(set(normalized_ids)):
        raise ValueError("source_fact_ids must be unique.")
    return normalized_ids, normalized_hashes


__all__ = ["finite", "lineage", "nonnegative_int", "positive", "required"]
