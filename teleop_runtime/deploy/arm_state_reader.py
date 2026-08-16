"""Causal arm-state sampling for the real policy deployment frontend."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from robot_motion.runtime.causal_history import CausalSnapshotHistory

from ..interfaces import EffectorBackend


class ArmStateReader:
    """Refresh backend state and select the latest snapshot admitted at a cut.

    The reader owns only refresh cadence and bounded causal history.  Backends
    continue to own device state and FK, while deployment retains control,
    safety, authority, and dispatch decisions.
    """

    def __init__(
        self,
        backends: Mapping[str, EffectorBackend],
        *,
        refresh_hz: float,
        compute_tcp_pose: bool = True,
    ):
        self.backends = dict(backends)
        self.refresh_hz = float(refresh_hz)
        self.compute_tcp_pose = bool(compute_tcp_pose)
        self._last_refresh_at = {side: 0.0 for side in self.backends}
        self._history_by_side = {
            side: CausalSnapshotHistory(maxlen=256, name=f"deploy:arm:{side}")
            for side in self.backends
        }

    def snapshot(self, now: float) -> dict[str, dict[str, Any]]:
        states: dict[str, dict[str, Any]] = {}
        refresh_period = 0.0 if self.refresh_hz <= 0.0 else 1.0 / self.refresh_hz
        for side, backend in self.backends.items():
            refresh_error: str | None = None
            if (
                refresh_period > 0.0
                and backend.is_ready()
                and now - self._last_refresh_at.get(side, 0.0) >= refresh_period
            ):
                try:
                    refresh = getattr(backend, "refresh_state_from_hardware", None)
                    if callable(refresh):
                        refresh()
                except Exception as exc:  # noqa: BLE001
                    refresh_error = f"refresh_failed:{type(exc).__name__}:{exc}"
                self._last_refresh_at[side] = now
            state = dict(backend.get_state())
            if refresh_error is not None:
                state["state_error"] = refresh_error
            state["actual_tcp_pose"] = None
            state["actual_tcp_pose_source"] = None
            state["actual_tcp_pose_timestamp"] = None
            if self.compute_tcp_pose:
                try:
                    hardware_qpos = state.get("hardware_qpos")
                    compute_pose = getattr(backend, "compute_tcp_pose", None)
                    if not callable(compute_pose):
                        raise TypeError("backend has no compute_tcp_pose")
                    if hardware_qpos is not None:
                        measured_pose = compute_pose(qpos=hardware_qpos)
                        state["current_tcp_pose"] = measured_pose
                        state["tcp_pose_source"] = "hardware_qpos_fk"
                        try:
                            hardware_qpos_vector = np.asarray(
                                hardware_qpos, dtype=float
                            ).reshape(-1)
                            measurement_timestamp = float(
                                state.get("hardware_measurement_timestamp")
                            )
                            measurement_sequence = state.get(
                                "hardware_measurement_sequence"
                            )
                            measurement_valid = state.get("hardware_measurement_valid")
                            read_success = state.get("hardware_read_success")
                            sequence_valid = (
                                measurement_sequence is None
                                or int(measurement_sequence) > 0
                            )
                            actual_valid = bool(
                                hardware_qpos_vector.shape == (6,)
                                and np.all(np.isfinite(hardware_qpos_vector))
                                and np.isfinite(measurement_timestamp)
                                and isinstance(measurement_valid, (bool, np.bool_))
                                and bool(measurement_valid)
                                and not (
                                    isinstance(read_success, (bool, np.bool_))
                                    and not bool(read_success)
                                )
                                and sequence_valid
                            )
                        except (TypeError, ValueError, OverflowError):
                            actual_valid = False
                            measurement_timestamp = None
                        if actual_valid:
                            state["actual_tcp_pose"] = np.asarray(
                                measured_pose, dtype=float
                            ).copy()
                            state["actual_tcp_pose_source"] = "hardware_qpos_fk"
                            state["actual_tcp_pose_timestamp"] = float(
                                measurement_timestamp
                            )
                    else:
                        state["current_tcp_pose"] = compute_pose()
                        state["tcp_pose_source"] = "current_qpos_fk"
                except Exception as exc:  # noqa: BLE001
                    state.setdefault(
                        "pose_error",
                        f"compute_tcp_pose_failed:{type(exc).__name__}:{exc}",
                    )
            source_timestamp = state.get(
                "hardware_measurement_timestamp",
                state.get("hardware_refresh_time"),
            )
            self._history_by_side[side].publish(
                state,
                source_timestamp=source_timestamp,
                available_timestamp=None,
            )
            selected = self._history_by_side[side].latest_at_or_before(float(now))
            if selected is None:
                states[side] = {
                    "causal_snapshot_missing": True,
                    "causal_snapshot_cutoff_timestamp": float(now),
                    "causal_snapshot_contract": (
                        "source_timestamp<=T0_and_available_timestamp<=T0"
                    ),
                }
            else:
                selected_state, selection = selected
                selected_state.update(
                    {
                        "causal_snapshot_missing": False,
                        "causal_snapshot_sequence": selection.sequence,
                        "causal_snapshot_source_timestamp": selection.source_timestamp,
                        "causal_snapshot_available_timestamp": (
                            selection.available_timestamp
                        ),
                        "causal_snapshot_age_sec": selection.age_sec,
                        "causal_snapshot_cutoff_timestamp": float(now),
                        "causal_snapshot_contract": (
                            "source_timestamp<=T0_and_available_timestamp<=T0"
                        ),
                    }
                )
                states[side] = selected_state
        return states


__all__ = ["ArmStateReader"]
