from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FaultNanTail:
    episode_index: int
    start_frame: int
    end_frame_exclusive: int
    start_row: int
    end_row_exclusive: int
    fault_state: str
    nan_action_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_fault_state(row: Mapping[str, Any], *, state_column: str, fault_values: set[str]) -> bool:
    value = row.get(state_column)
    return str(value).upper() in fault_values


def _action_is_nan(row: Mapping[str, Any], *, action_column: str) -> bool:
    action = row.get(action_column)
    if action is None:
        return False
    try:
        array = np.asarray(action, dtype=float).reshape(-1)
    except Exception:
        return False
    return bool(array.size > 0 and np.all(np.isnan(array)))


def detect_fault_nan_tail(
    rows: Sequence[Mapping[str, Any]],
    *,
    episode_index: int,
    state_column: str = "session_state",
    action_column: str = "action",
    frame_index_column: str = "frame_index",
    fault_values: Sequence[str] = ("FAULT", "ESTOP"),
    min_nan_fraction: float = 0.8,
) -> FaultNanTail | None:
    """Return the suffix where FAULT/ESTOP rows continue with NaN actions."""
    if not rows:
        return None
    faults = {str(value).upper() for value in fault_values}
    threshold = float(min_nan_fraction)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("min_nan_fraction must be in [0, 1].")
    for start_row, row in enumerate(rows):
        if not _is_fault_state(row, state_column=state_column, fault_values=faults):
            continue
        suffix = rows[start_row:]
        nan_flags = [_action_is_nan(item, action_column=action_column) for item in suffix]
        if not nan_flags:
            continue
        nan_fraction = float(sum(nan_flags) / len(nan_flags))
        if nan_fraction < threshold:
            continue
        start_frame = int(row.get(frame_index_column, start_row))
        last_row = rows[-1]
        end_frame = int(last_row.get(frame_index_column, len(rows) - 1)) + 1
        return FaultNanTail(
            episode_index=int(episode_index),
            start_frame=start_frame,
            end_frame_exclusive=end_frame,
            start_row=int(start_row),
            end_row_exclusive=len(rows),
            fault_state=str(row.get(state_column)),
            nan_action_fraction=nan_fraction,
        )
    return None
