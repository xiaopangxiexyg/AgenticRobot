"""Pure calibration-profile discovery and qpos normalization helpers.

This module keeps file selection and six-joint value normalization separate
from collection lifecycle composition.  It does not construct a calibration,
connect to a device, or dispatch a command.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import numpy as np

from .collect_cli_parser import DEFAULT_CALIBRATION_PROFILE_DIR

CALIBRATION_PROFILE_TIMESTAMP_RE = re.compile(r"^(\d{8})_(\d{6})")


def calibration_profile_timestamp(path: Path) -> float | None:
    """Return a filename timestamp, or ``None`` when the name is unrecognized."""

    match = CALIBRATION_PROFILE_TIMESTAMP_RE.match(path.name)
    if match is None:
        return None
    try:
        return datetime.strptime(  # noqa: DTZ007 - preserve local-time ordering.
            "_".join(match.groups()), "%Y%m%d_%H%M%S"
        ).timestamp()
    except ValueError:
        return None


def latest_calibration_profile_path(
    profile_dir: Path = DEFAULT_CALIBRATION_PROFILE_DIR,
) -> Path | None:
    """Select the newest JSON calibration profile using legacy precedence."""

    if not profile_dir.exists():
        return None
    candidates = [path for path in profile_dir.glob("*.json") if path.is_file()]
    if not candidates:
        return None

    def sort_key(path: Path) -> tuple[int, float, float, str]:
        name_timestamp = calibration_profile_timestamp(path)
        try:
            modified_at = path.stat().st_mtime
        except OSError:
            modified_at = 0.0
        if name_timestamp is not None:
            return (1, float(name_timestamp), float(modified_at), path.name)
        return (0, float(modified_at), float(modified_at), path.name)

    return max(candidates, key=sort_key)


def qpos6_deg_to_rad(values: Sequence[float]) -> tuple[float, ...]:
    """Convert exactly six joint positions from degrees to radians."""

    qpos_deg = np.asarray(values, dtype=float).reshape(6)
    return tuple(float(value) for value in np.deg2rad(qpos_deg))


def qpos6_radians_look_suspicious(values: Sequence[float]) -> bool:
    """Detect likely degree-valued data passed to a radians field."""

    qpos = np.asarray(values, dtype=float).reshape(6)
    return bool(np.max(np.abs(qpos)) > (2.0 * np.pi + 0.25))


__all__ = [
    "CALIBRATION_PROFILE_TIMESTAMP_RE",
    "calibration_profile_timestamp",
    "latest_calibration_profile_path",
    "qpos6_deg_to_rad",
    "qpos6_radians_look_suspicious",
]
