"""Row-level validation shared by v2.3 writers and offline acceptance scans."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .causal_snapshot import LOCAL_WALL_CLOCK, RECEIVER_WALL_CLOCK


class V23RowValidationError(ValueError):
    pass


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _validate_cut(
    failures: list[str],
    *,
    label: str,
    source_timestamp: Any,
    available_timestamp: Any,
    source_clock: Any,
    available_clock: Any,
    t0: float,
    expected_source_clocks: set[str],
) -> None:
    source = _finite(source_timestamp)
    available = _finite(available_timestamp)
    if source is None:
        failures.append(f"{label}:missing_source_timestamp")
    elif source > t0 + 1e-6:
        failures.append(f"{label}:source_timestamp_after_T0")
    if available is None:
        failures.append(f"{label}:missing_available_timestamp")
    elif available > t0 + 1e-6:
        failures.append(f"{label}:available_timestamp_after_T0")
    if str(source_clock) not in expected_source_clocks:
        failures.append(f"{label}:unsupported_source_clock:{source_clock}")
    if str(available_clock) != LOCAL_WALL_CLOCK:
        failures.append(f"{label}:available_clock_not_local_wall_time")


def validate_v23_row_causal_snapshots(
    row: Mapping[str, Any],
    *,
    active_sides: Sequence[str] = ("left", "right"),
    arm_source_clocks: Sequence[str] = (LOCAL_WALL_CLOCK,),
) -> None:
    """Validate every present causal state selection against the row T0."""

    t0 = _finite(row.get("timestamp"))
    if t0 is None:
        raise V23RowValidationError("v2.3 row requires a finite timestamp T0.")
    failures: list[str] = []
    for side in active_sides:
        arm_prefix = f"observation.arm.{side}."
        _validate_cut(
            failures,
            label=f"arm.{side}",
            source_timestamp=row.get(f"{arm_prefix}source_timestamp"),
            available_timestamp=row.get(f"{arm_prefix}available_timestamp"),
            source_clock=row.get(f"{arm_prefix}source_clock"),
            available_clock=row.get(f"{arm_prefix}available_clock"),
            t0=t0,
            expected_source_clocks=set(str(value) for value in arm_source_clocks),
        )
        for stream in ("telemetry", "tactile"):
            prefix = f"observation.hand.{side}.{stream}_snapshot_"
            sequence = row.get(f"{prefix}sequence")
            if sequence is None:
                continue
            _validate_cut(
                failures,
                label=f"hand.{side}.{stream}",
                source_timestamp=row.get(f"{prefix}source_timestamp"),
                available_timestamp=row.get(f"{prefix}available_timestamp"),
                source_clock=row.get(f"{prefix}source_clock"),
                available_clock=row.get(f"{prefix}available_clock"),
                t0=t0,
                expected_source_clocks={LOCAL_WALL_CLOCK},
            )

    suffix = ".source_timestamp"
    camera_prefix = "observation.images."
    for key in row:
        if not str(key).startswith(camera_prefix) or not str(key).endswith(suffix):
            continue
        prefix = str(key)[: -len(suffix)]
        source_timestamp = row.get(str(key))
        if source_timestamp is None:
            continue
        _validate_cut(
            failures,
            label=prefix,
            source_timestamp=source_timestamp,
            available_timestamp=row.get(f"{prefix}.available_timestamp"),
            source_clock=row.get(f"{prefix}.source_clock"),
            available_clock=row.get(f"{prefix}.available_clock"),
            t0=t0,
            expected_source_clocks={RECEIVER_WALL_CLOCK},
        )
    if failures:
        raise V23RowValidationError("; ".join(failures))


__all__ = ["V23RowValidationError", "validate_v23_row_causal_snapshots"]
