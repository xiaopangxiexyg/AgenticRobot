"""Shared private validation helpers for trajectory contracts."""

from __future__ import annotations

import math
from typing import Sequence

from robot_motion.runtime.control_admission import ControlCut


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _optional(value: object | None, *, name: str) -> str | None:
    if value is None:
        return None
    return _required(value, name=name)


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _identity_pair(
    identity: object | None,
    identity_hash: object | None,
    *,
    name: str,
) -> tuple[str | None, str | None]:
    if (identity is None) != (identity_hash is None):
        raise ValueError(f"{name} id/hash must be both present or both absent.")
    if identity is None:
        return None, None
    return (
        _required(identity, name=f"{name}_id"),
        _required(identity_hash, name=f"{name}_hash"),
    )


def _components(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(_required(value, name="component_id") for value in values)
    if not result or len(result) != len(set(result)):
        raise ValueError("component_ids must be non-empty and unique.")
    return result


def _reasons(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(_required(value, name="reason_code") for value in values)
    if len(result) != len(set(result)):
        raise ValueError("reason_codes must be unique.")
    return result


def _cut_payload(cut: ControlCut) -> dict[str, object]:
    if not isinstance(cut, ControlCut):
        raise TypeError("control_cut must be a ControlCut.")
    return {
        "runtime_session_id": cut.runtime_session_id,
        "control_domain_id": cut.control_domain_id,
        "clock_domain_id": cut.clock_domain_id,
        "clock_reset_epoch": cut.clock_reset_epoch,
        "cutoff_at": cut.cutoff_at,
        "cutoff_admission_sequence": cut.cutoff_admission_sequence,
    }
