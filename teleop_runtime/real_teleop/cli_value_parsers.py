"""Pure ``argparse`` value parsers for the real-teleop CLI.

These validators do not read devices or mutate an ``argparse.Namespace``.
Compatibility aliases in the legacy entrypoint preserve its public helpers.
"""

from __future__ import annotations

import argparse
import re

import numpy as np

from ..backends.grippers import normalize_dmtac_tactile_fields

def _non_negative_float(text: str) -> float:
    value = float(text)
    if value < 0.0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return value

def _canfd_trajectory_mode(text: str) -> int:
    value = int(text)
    if value not in (0, 1, 2):
        raise argparse.ArgumentTypeError("trajectory mode must be 0, 1, or 2")
    return value

def _canfd_radio(text: str) -> int:
    value = int(text)
    if not 0 <= value <= 100:
        raise argparse.ArgumentTypeError("radio must be in [0, 100]")
    return value

def _movement_multiple_arg(text: str) -> float:
    value = float(text)
    if not np.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("movement_multiple must be a positive finite value")
    return value

def _local_step_clip_scale_arg(text: str) -> float:
    value = float(text)
    if not np.isfinite(value) or value < 1.0:
        raise argparse.ArgumentTypeError("local-step clip scale must be finite and >= 1")
    return value

def _positive_unit_scale_arg(text: str) -> float:
    value = float(text)
    if not np.isfinite(value) or not 0.0 < value <= 1.0:
        raise argparse.ArgumentTypeError("scale must be finite and in (0, 1]")
    return value

def _positive_finite_float_arg(text: str) -> float:
    value = float(text)
    if not np.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive and finite")
    return value

def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value

def _dmtac_worker_cpu_affinity(text: str) -> tuple[int, ...]:
    parts = [part.strip() for part in re.split(r"[,\s]+", str(text)) if part.strip()]
    if not parts:
        return ()
    cpus: list[int] = []
    for part in parts:
        try:
            cpu = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("CPU affinity must be a comma-separated integer list") from exc
        if cpu < 0:
            raise argparse.ArgumentTypeError("CPU affinity indices must be >= 0")
        cpus.append(cpu)
    return tuple(dict.fromkeys(cpus))

def _dmtac_worker_nice(text: str) -> int:
    value = int(text)
    if not 0 <= value <= 19:
        raise argparse.ArgumentTypeError("worker nice must be in [0, 19]")
    return value

def _parse_dmtac_tactile_fields(text: str) -> tuple[str, ...]:
    try:
        return normalize_dmtac_tactile_fields(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc

__all__ = [
    "_non_negative_float",
    "_canfd_trajectory_mode",
    "_canfd_radio",
    "_movement_multiple_arg",
    "_local_step_clip_scale_arg",
    "_positive_unit_scale_arg",
    "_positive_finite_float_arg",
    "_positive_int",
    "_dmtac_worker_cpu_affinity",
    "_dmtac_worker_nice",
    "_parse_dmtac_tactile_fields",
]
