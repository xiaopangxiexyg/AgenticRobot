"""Bounded, non-authoritative trace for exclusive RM65 Skill motion.

All high-rate callbacks copy fixed-size numeric fields into preallocated NumPy
buffers.  No JSON encoding, FK, queue wait, or disk write occurs while control
and CANFD workers are active.  The owner detaches the callbacks and calls
``close`` only after the lower application has stopped.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from .control_runtime import RealSkillControlTraceSample
from .motion_port import REAL_AGENT_MOTION_COMMAND_METADATA_KEY

SKILL_MOTION_TRACE_SCHEMA = "dexteleop.exclusive_rm65_skill_motion_trace.v2"
SKILL_MOTION_TRACE_DATA_NAME = "motion_trace.npz"
SKILL_MOTION_TRACE_MANIFEST_NAME = "manifest.json"
SKILL_MOTION_TRACE_SUMMARY_NAME = "summary.json"
DEFAULT_SKILL_MOTION_TRACE_MAX_DURATION_SEC = 120.0

_SIDES = ("left", "right")
_I64_MISSING = np.iinfo(np.int64).min
_I32_MISSING = np.iinfo(np.int32).min
_HARDWARE_READ_TIMING_FIELDS = (
    ("hardware_read_total_wall_ms", "total_wall_ms"),
    ("hardware_read_total_thread_cpu_ms", "total_thread_cpu_ms"),
    ("hardware_read_lock_wait_wall_ms", "hardware_read_lock_wait_wall_ms"),
    (
        "hardware_read_lock_wait_thread_cpu_ms",
        "hardware_read_lock_wait_thread_cpu_ms",
    ),
    ("hardware_read_sdk_call_wall_ms", "sdk_call_wall_ms"),
    ("hardware_read_sdk_call_thread_cpu_ms", "sdk_call_thread_cpu_ms"),
    (
        "hardware_read_parse_validation_wall_ms",
        "parse_validation_wall_ms",
    ),
    (
        "hardware_read_parse_validation_thread_cpu_ms",
        "parse_validation_thread_cpu_ms",
    ),
    ("hardware_read_retry_sleep_wall_ms", "retry_sleep_wall_ms"),
    ("hardware_read_attempt_count", "attempt_count"),
)

_REFERENCE_DTYPE = np.dtype(
    [
        ("cycle_sequence", "<i8"),
        ("control_wall_time_ns", "<i8"),
        ("cycle_started_monotonic_ns", "<i8"),
        ("cycle_finished_monotonic_ns", "<i8"),
        ("measurement_wall_time_ns", "<i8"),
        ("measurement_available_wall_time_ns", "<i8"),
        ("measurement_span_ms", "<f8"),
        ("measurement_available_delay_ms", "<f8"),
        ("source_sequence", "<i8", (2,)),
        *[
            (trace_field, "<f8", (2,))
            for trace_field, _ in _HARDWARE_READ_TIMING_FIELDS
        ],
        ("measured_pose_revision", "<i8"),
        ("measured_qpos_rad", "<f8", (2, 6)),
        ("measured_tool_tcp_pose_world", "<f8", (2, 4, 4)),
        ("target_tool_tcp_pose_world", "<f8", (4, 4)),
        ("qref_rad", "<f8", (2, 6)),
        ("raw_ik_qpos_rad", "<f8", (2, 6)),
        ("solver_output_tool_tcp_pose_base", "<f8", (2, 4, 4)),
        ("active_side_index", "i1"),
        ("operation_id", "S96"),
        ("request_id", "S96"),
        ("port_phase", "S32"),
        ("outcome", "S40"),
        ("error", "S256"),
        ("ik_status", "S48"),
        ("ik_branch", "S64"),
        ("ik_fallback_reason", "S128"),
        ("ik_jump_gate_status", "S48"),
        ("speed_scale", "<f8"),
        ("collision_distance_m", "<f8"),
        ("joint_limit_margin_deg", "<f8", (2,)),
        ("ik_iterations", "<i4"),
        ("ik_frame_total_ms", "<f8"),
        ("ik_time_budget_ms", "<f8"),
        ("ik_time_budget_exceeded", "u1"),
        ("ik_profile_solve_calls", "<f8"),
        ("ik_profile_solve_total_ms", "<f8"),
        ("ik_profile_solve_qp_ms", "<f8"),
        ("ik_profile_solve_integrate_ms", "<f8"),
        ("ik_profile_solve_set_configuration_ms", "<f8"),
        ("ik_profile_min_distance_calls", "<f8"),
        ("ik_profile_min_distance_total_ms", "<f8"),
        ("command_present", "u1"),
        ("admitted", "u1"),
        ("expired_reference", "u1"),
        ("dispatch_attempted", "u1"),
        ("dispatch_succeeded", "u1"),
        ("observation_ms", "<f8"),
        ("synchronization_ms", "<f8"),
        ("owner_requests_ms", "<f8"),
        ("ik_ms", "<f8"),
        ("admission_ms", "<f8"),
        ("dispatch_ms", "<f8"),
        ("control_cycle_ms", "<f8"),
        ("measurement_age_at_dispatch_ms", "<f8"),
        ("backend_target_sequence", "<i8", (2,)),
        ("backend_actual_hz", "<f8", (2,)),
        ("backend_send_gap_ms", "<f8", (2,)),
        ("backend_sdk_call_ms", "<f8", (2,)),
        ("backend_deadline_lag_ms", "<f8", (2,)),
        ("backend_deadline_miss_count", "<i8", (2,)),
        ("backend_skipped_tick_count", "<i8", (2,)),
        ("backend_low_rate_count", "<i8", (2,)),
    ],
    align=False,
)

_ACTUATOR_DTYPE = np.dtype(
    [
        ("tick_sequence", "<i8"),
        ("target_sequence", "<i8"),
        ("reference_epoch", "<i8"),
        ("reference_sequence", "<i8"),
        ("sdk_attempt_sequence", "<i8"),
        ("sdk_accepted_sequence", "<i8"),
        ("scheduled_monotonic_ns", "<i8"),
        ("wakeup_monotonic_ns", "<i8"),
        ("sdk_attempt_wall_time_ns", "<i8"),
        ("sdk_attempt_monotonic_ns", "<i8"),
        ("sdk_result_monotonic_ns", "<i8"),
        ("sdk_result_code", "<i4"),
        ("sdk_accepted", "u1"),
        ("q_cmd_rad", "<f8", (6,)),
    ],
    align=False,
)

_IK_DTYPE = np.dtype(
    [
        ("sequence", "<i8"),
        ("monotonic_time_ns", "<i8"),
        ("pre_q_rad", "<f8", (12,)),
        ("raw_solver_q_rad", "<f8", (12,)),
        ("canonical_solver_q_rad", "<f8", (12,)),
        ("admitted_q_rad", "<f8", (12,)),
        ("target_world_pose", "<f8", (2, 4, 4)),
        ("jacobian_singular_values", "<f8", (2, 6)),
        ("wrist_null_scale", "<f8", (2,)),
        ("status", "S48"),
    ],
    align=False,
)


def _seconds_to_ns(value: object) -> int:
    if value is None:
        return int(_I64_MISSING)
    try:
        seconds = float(value)
    except (TypeError, ValueError, OverflowError):
        return int(_I64_MISSING)
    return (
        round(seconds * 1_000_000_000.0)
        if math.isfinite(seconds)
        else int(_I64_MISSING)
    )


def _number(value: object, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def _integer(value: object) -> int:
    if value is None or isinstance(value, bool):
        return int(_I64_MISSING)
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return int(_I64_MISSING)


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _matrix(value: object, shape: tuple[int, ...]) -> np.ndarray | None:
    try:
        result = np.asarray(value, dtype=float).reshape(shape)
    except (TypeError, ValueError):
        return None
    return result if np.all(np.isfinite(result)) else None


def _duration_ms(end: object, start: object) -> float:
    end_value = _number(end)
    start_value = _number(start)
    if not math.isfinite(end_value) or not math.isfinite(start_value):
        return float("nan")
    return max(0.0, end_value - start_value) * 1000.0


def _text(value: object, limit: int) -> bytes:
    return str(value or "").encode("utf-8", errors="replace")[:limit]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


@dataclass(frozen=True)
class SkillMotionTraceConfig:
    output_dir: Path
    reference_hz: float
    command_hz: float
    measurement_hz: float | None = None
    max_duration_sec: float = DEFAULT_SKILL_MOTION_TRACE_MAX_DURATION_SEC
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("reference_hz", "command_hz", "max_duration_sec"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)
        if self.measurement_hz is not None:
            measurement_hz = float(self.measurement_hz)
            if not math.isfinite(measurement_hz) or measurement_hz <= 0.0:
                raise ValueError("measurement_hz must be finite and positive.")
            object.__setattr__(self, "measurement_hz", measurement_hz)
        object.__setattr__(self, "output_dir", Path(self.output_dir))

    @property
    def reference_capacity(self) -> int:
        return math.ceil((self.max_duration_sec + 2.0) * self.reference_hz)

    @property
    def command_capacity_per_side(self) -> int:
        return math.ceil((self.max_duration_sec + 2.0) * self.command_hz)


class SkillMotionTraceRecorder:
    """Single-owner reference trace plus single-producer-per-side CANFD trace."""

    def __init__(self, config: SkillMotionTraceConfig) -> None:
        self.config = config
        self.output_dir = config.output_dir
        self.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
        self._reference = np.empty(config.reference_capacity, dtype=_REFERENCE_DTYPE)
        self._reference.view(np.uint8).fill(0)
        self._actuator = {
            side: np.empty(config.command_capacity_per_side, dtype=_ACTUATOR_DTYPE)
            for side in _SIDES
        }
        self._ik = np.empty(config.reference_capacity, dtype=_IK_DTYPE)
        self._ik.view(np.uint8).fill(0)
        for buffer in self._actuator.values():
            buffer.view(np.uint8).fill(0)
        self._reference_next = 0
        self._actuator_next = {side: 0 for side in _SIDES}
        self._ik_next = 0
        self._drop_count = {"reference": 0, "ik": 0, "left": 0, "right": 0}
        self._error_count = {"reference": 0, "ik": 0, "left": 0, "right": 0}
        self._last_error = {key: None for key in self._error_count}
        self._accepting = True
        self._closed = False
        self._started_wall_time_ns = time.time_ns()
        self._started_monotonic_time_ns = time.monotonic_ns()

    def _record_error(self, stream: str, exc: BaseException) -> bool:
        self._error_count[stream] += 1
        self._last_error[stream] = f"{type(exc).__name__}: {exc}"[:512]
        return False

    def record_control_cycle(self, sample: RealSkillControlTraceSample) -> bool:
        if not self._accepting:
            return False
        try:
            if not isinstance(sample, RealSkillControlTraceSample):
                raise TypeError("sample must be RealSkillControlTraceSample")
            index = self._reference_next
            if index >= self._reference.shape[0]:
                self._drop_count["reference"] += 1
                return False
            row = self._reference[index]
            for name in (
                "measured_qpos_rad",
                "measured_tool_tcp_pose_world",
                "target_tool_tcp_pose_world",
                "qref_rad",
                "raw_ik_qpos_rad",
                "solver_output_tool_tcp_pose_base",
                "joint_limit_margin_deg",
                "backend_actual_hz",
                "backend_send_gap_ms",
                "backend_sdk_call_ms",
                "backend_deadline_lag_ms",
                *(trace_field for trace_field, _ in _HARDWARE_READ_TIMING_FIELDS),
            ):
                row[name].fill(float("nan"))
            row["source_sequence"].fill(_I64_MISSING)
            row["backend_target_sequence"].fill(_I64_MISSING)
            row["backend_deadline_miss_count"].fill(_I64_MISSING)
            row["backend_skipped_tick_count"].fill(_I64_MISSING)
            row["backend_low_rate_count"].fill(_I64_MISSING)
            for name in (
                "speed_scale",
                "collision_distance_m",
                "ik_frame_total_ms",
                "ik_time_budget_ms",
                "ik_profile_solve_total_ms",
                "ik_profile_solve_qp_ms",
                "ik_profile_solve_integrate_ms",
                "ik_profile_solve_set_configuration_ms",
                "ik_profile_solve_calls",
                "ik_profile_min_distance_calls",
                "ik_profile_min_distance_total_ms",
                "observation_ms",
                "synchronization_ms",
                "owner_requests_ms",
                "ik_ms",
                "admission_ms",
                "dispatch_ms",
                "control_cycle_ms",
                "measurement_age_at_dispatch_ms",
                "measurement_span_ms",
                "measurement_available_delay_ms",
            ):
                row[name] = float("nan")
            row["ik_iterations"] = _I32_MISSING
            row["active_side_index"] = -1
            row["cycle_sequence"] = int(sample.cycle_sequence)
            row["control_wall_time_ns"] = _seconds_to_ns(sample.control_time)
            row["cycle_started_monotonic_ns"] = _seconds_to_ns(
                sample.cycle_started_monotonic
            )
            row["cycle_finished_monotonic_ns"] = _seconds_to_ns(
                sample.cycle_finished_monotonic
            )
            row["outcome"] = _text(sample.outcome, 40)
            row["error"] = _text(sample.error, 256)
            row["command_present"] = int(sample.command_present)
            row["admitted"] = int(sample.admitted)
            row["expired_reference"] = int(sample.expired_reference)
            row["dispatch_attempted"] = int(sample.dispatch_attempted)
            row["dispatch_succeeded"] = int(sample.dispatch_succeeded)

            cut = sample.cut
            if cut is not None:
                row["measurement_wall_time_ns"] = _seconds_to_ns(cut.measured_at)
                row["measurement_available_wall_time_ns"] = _seconds_to_ns(
                    cut.available_at
                )
                row["measurement_available_delay_ms"] = max(
                    0.0,
                    float(cut.available_at) - float(cut.measured_at),
                ) * 1000.0
                cut_metadata = _mapping(cut.observation.metadata)
                row["measurement_span_ms"] = (
                    _number(cut_metadata.get("hardware_measurement_span_sec"))
                    * 1000.0
                )
                hardware_timing_by_side = _mapping(
                    cut_metadata.get("hardware_read_phase_timing_by_side")
                )
                for side_index, side in enumerate(_SIDES):
                    row["source_sequence"][side_index] = _integer(
                        cut.source_sequence_by_side.get(side)
                    )
                    row["measured_qpos_rad"][side_index] = cut.qpos_by_side[side]
                    side_timing = _mapping(hardware_timing_by_side.get(side))
                    for trace_field, source_field in _HARDWARE_READ_TIMING_FIELDS:
                        row[trace_field][side_index] = _number(
                            side_timing.get(source_field)
                        )
            else:
                row["measurement_wall_time_ns"] = _I64_MISSING
                row["measurement_available_wall_time_ns"] = _I64_MISSING

            pose_cut = sample.measured_pose_cut
            if pose_cut is None:
                row["measured_pose_revision"] = _I64_MISSING
            else:
                row["measured_pose_revision"] = _integer(
                    getattr(pose_cut, "revision", None)
                )
                poses = _mapping(getattr(pose_cut, "pose_world_by_component", {}))
                for side_index, side in enumerate(_SIDES):
                    pose = _matrix(poses.get(f"{side}_arm"), (4, 4))
                    if pose is not None:
                        row["measured_tool_tcp_pose_world"][side_index] = pose

            command = sample.command
            operation = {}
            metadata = {}
            if command is not None:
                metadata = _mapping(command.metadata)
                operation = _mapping(
                    metadata.get(REAL_AGENT_MOTION_COMMAND_METADATA_KEY)
                )
                row["operation_id"] = _text(operation.get("operation_id"), 96)
                row["request_id"] = _text(operation.get("request_id"), 96)
                active_side = str(operation.get("side") or "")
                if active_side in _SIDES:
                    row["active_side_index"] = _SIDES.index(active_side)
                row["speed_scale"] = _number(operation.get("speed_scale"))
                target = _matrix(
                    operation.get("target_tool_tcp_pose_world"),
                    (4, 4),
                )
                if target is not None:
                    row["target_tool_tcp_pose_world"] = target
                for side_index, side in enumerate(_SIDES):
                    effector = command.commands.get(f"{side}_arm")
                    if effector is None:
                        continue
                    qref = _matrix(effector.value, (6,))
                    if qref is not None:
                        row["qref_rad"][side_index] = qref
                    effector_metadata = _mapping(effector.metadata)
                    raw_q = _matrix(effector_metadata.get("q_ik_raw"), (6,))
                    if raw_q is not None:
                        row["raw_ik_qpos_rad"][side_index] = raw_q
                    output_pose = _matrix(
                        effector_metadata.get("target_pose_tcp_base"),
                        (4, 4),
                    )
                    if output_pose is not None:
                        row["solver_output_tool_tcp_pose_base"][side_index] = (
                            output_pose
                        )
                    margin = _mapping(effector_metadata.get("joint_limit_margin"))
                    row["joint_limit_margin_deg"][side_index] = _number(
                        margin.get(
                            "minimum_margin_deg",
                            margin.get("min_margin_deg"),
                        )
                    )
            else:
                row["operation_id"] = b""
                row["request_id"] = b""

            port_snapshot = sample.port_snapshot
            row["port_phase"] = _text(
                (
                    getattr(getattr(port_snapshot, "phase", ""), "value", "")
                    if port_snapshot is not None
                    else ""
                ),
                32,
            )
            row["ik_status"] = _text(
                metadata.get("collision_avoidance_status"), 48
            )
            row["ik_branch"] = _text(metadata.get("ik_branch_candidate"), 64)
            row["ik_fallback_reason"] = _text(
                metadata.get("collision_avoidance_fallback_reason"), 128
            )
            row["ik_jump_gate_status"] = _text(
                metadata.get("ik_jump_gate_status"), 48
            )
            row["collision_distance_m"] = _number(
                metadata.get("collision_distance_m")
            )
            row["ik_iterations"] = int(
                _integer(metadata.get("ik_solve_iterations"))
            )
            row["ik_frame_total_ms"] = _number(metadata.get("ik_frame_total_ms"))
            row["ik_time_budget_ms"] = _number(metadata.get("ik_time_budget_ms"))
            row["ik_time_budget_exceeded"] = int(
                bool(metadata.get("ik_time_budget_exceeded", False))
            )
            ik_profile = _mapping(metadata.get("ik_debug_profile"))
            row["ik_profile_solve_calls"] = _number(
                ik_profile.get("solve_calls")
            )
            row["ik_profile_solve_total_ms"] = _number(
                ik_profile.get("solve_total_ms")
            )
            row["ik_profile_solve_qp_ms"] = _number(
                ik_profile.get("solve_qp_ms")
            )
            row["ik_profile_solve_integrate_ms"] = _number(
                ik_profile.get("solve_integrate_ms")
            )
            row["ik_profile_solve_set_configuration_ms"] = _number(
                ik_profile.get("solve_set_configuration_ms")
            )
            row["ik_profile_min_distance_calls"] = _number(
                ik_profile.get("min_distance_calls")
            )
            row["ik_profile_min_distance_total_ms"] = _number(
                ik_profile.get("min_distance_total_ms")
            )

            start = sample.cycle_started_monotonic
            observation_end = sample.observation_completed_monotonic
            sync_end = sample.synchronization_completed_monotonic
            owner_end = sample.owner_requests_completed_monotonic
            ik_end = sample.ik_completed_monotonic
            admission_end = sample.admission_completed_monotonic
            dispatch_end = sample.dispatch_completed_monotonic
            row["observation_ms"] = _duration_ms(observation_end, start)
            row["synchronization_ms"] = _duration_ms(sync_end, observation_end)
            row["owner_requests_ms"] = _duration_ms(owner_end, sync_end)
            row["ik_ms"] = _duration_ms(ik_end, owner_end)
            row["admission_ms"] = _duration_ms(admission_end, ik_end)
            row["dispatch_ms"] = _duration_ms(dispatch_end, admission_end)
            row["control_cycle_ms"] = _duration_ms(
                sample.cycle_finished_monotonic,
                start,
            )
            if cut is not None and sample.dispatch_ready_at is not None:
                row["measurement_age_at_dispatch_ms"] = max(
                    0.0,
                    float(sample.dispatch_ready_at) - float(cut.measured_at),
                ) * 1000.0

            receipt_diagnostics = (
                _mapping(sample.dispatch_receipt.diagnostics)
                if sample.dispatch_receipt is not None
                else {}
            )
            streams = _mapping(receipt_diagnostics.get("backend_streaming_by_side"))
            for side_index, side in enumerate(_SIDES):
                stream = _mapping(streams.get(side))
                row["backend_target_sequence"][side_index] = _integer(
                    stream.get("target_sequence")
                )
                row["backend_actual_hz"][side_index] = _number(
                    stream.get("actual_hz")
                )
                row["backend_send_gap_ms"][side_index] = (
                    _number(stream.get("last_send_dt_sec")) * 1000.0
                )
                row["backend_sdk_call_ms"][side_index] = _number(
                    stream.get("last_call_ms")
                )
                row["backend_deadline_lag_ms"][side_index] = _number(
                    stream.get("last_deadline_lag_ms")
                )
                row["backend_deadline_miss_count"][side_index] = _integer(
                    stream.get("deadline_miss_count")
                )
                row["backend_skipped_tick_count"][side_index] = _integer(
                    stream.get("skipped_tick_count")
                )
                row["backend_low_rate_count"][side_index] = _integer(
                    stream.get("low_rate_count")
                )
            self._reference_next = index + 1
            return True
        except Exception as exc:  # noqa: BLE001 - trace cannot stop control.
            return self._record_error("reference", exc)

    def record_actuator_command(
        self,
        side: str,
        tick_sequence: int,
        target_sequence: int | None,
        reference_epoch: int | None,
        reference_sequence: int | None,
        scheduled_monotonic_time: float,
        wakeup_monotonic_time: float,
        sdk_attempt_sequence: int | None,
        sdk_attempt_wall_time: float | None,
        sdk_attempt_monotonic_time: float | None,
        sdk_result_monotonic_time: float | None,
        q_cmd: Sequence[float] | np.ndarray,
        sdk_result_code: int | None,
        sdk_accepted: bool | None,
        sdk_accepted_sequence: int | None,
    ) -> bool:
        if not self._accepting:
            return False
        normalized_side = str(side)
        try:
            buffer = self._actuator[normalized_side]
            index = self._actuator_next[normalized_side]
            if index >= buffer.shape[0]:
                self._drop_count[normalized_side] += 1
                return False
            qcmd = np.asarray(q_cmd, dtype=float)
            if qcmd.shape != (6,):
                raise ValueError("q_cmd must be a 6-vector")
            row = buffer[index]
            row["tick_sequence"] = int(tick_sequence)
            row["target_sequence"] = _integer(target_sequence)
            row["reference_epoch"] = _integer(reference_epoch)
            row["reference_sequence"] = _integer(reference_sequence)
            row["sdk_attempt_sequence"] = _integer(sdk_attempt_sequence)
            row["sdk_accepted_sequence"] = _integer(sdk_accepted_sequence)
            row["scheduled_monotonic_ns"] = _seconds_to_ns(scheduled_monotonic_time)
            row["wakeup_monotonic_ns"] = _seconds_to_ns(wakeup_monotonic_time)
            row["sdk_attempt_wall_time_ns"] = _seconds_to_ns(sdk_attempt_wall_time)
            row["sdk_attempt_monotonic_ns"] = _seconds_to_ns(
                sdk_attempt_monotonic_time
            )
            row["sdk_result_monotonic_ns"] = _seconds_to_ns(
                sdk_result_monotonic_time
            )
            row["sdk_result_code"] = (
                _I32_MISSING if sdk_result_code is None else int(sdk_result_code)
            )
            row["sdk_accepted"] = int(bool(sdk_accepted))
            row["q_cmd_rad"] = qcmd
            self._actuator_next[normalized_side] = index + 1
            return True
        except Exception as exc:  # noqa: BLE001 - trace cannot stop CANFD.
            if normalized_side not in self._error_count:
                return False
            return self._record_error(normalized_side, exc)

    def record_ik_shadow(
        self,
        monotonic_time: float,
        pre_q: Sequence[float] | np.ndarray,
        raw_solver_q: Sequence[float] | np.ndarray | None,
        canonical_solver_q: Sequence[float] | np.ndarray | None,
        admitted_q: Sequence[float] | np.ndarray,
        left_target_world_pose: np.ndarray,
        right_target_world_pose: np.ndarray,
        left_jacobian_singular_values: Sequence[float] | np.ndarray | None,
        right_jacobian_singular_values: Sequence[float] | np.ndarray | None,
        left_wrist_null_scale: float | None,
        right_wrist_null_scale: float | None,
        status: str,
    ) -> bool:
        if not self._accepting:
            return False
        try:
            index = self._ik_next
            if index >= self._ik.shape[0]:
                self._drop_count["ik"] += 1
                return False
            row = self._ik[index]
            for name in (
                "pre_q_rad",
                "raw_solver_q_rad",
                "canonical_solver_q_rad",
                "admitted_q_rad",
                "target_world_pose",
                "jacobian_singular_values",
                "wrist_null_scale",
            ):
                row[name].fill(float("nan"))
            row["sequence"] = index + 1
            row["monotonic_time_ns"] = _seconds_to_ns(monotonic_time)
            for name, value in (
                ("pre_q_rad", pre_q),
                ("raw_solver_q_rad", raw_solver_q),
                ("canonical_solver_q_rad", canonical_solver_q),
                ("admitted_q_rad", admitted_q),
            ):
                vector = _matrix(value, (12,))
                if vector is not None:
                    row[name] = vector
            for side_index, value in enumerate(
                (left_target_world_pose, right_target_world_pose)
            ):
                pose = _matrix(value, (4, 4))
                if pose is not None:
                    row["target_world_pose"][side_index] = pose
            for side_index, value in enumerate(
                (
                    left_jacobian_singular_values,
                    right_jacobian_singular_values,
                )
            ):
                singular = _matrix(value, (6,))
                if singular is not None:
                    row["jacobian_singular_values"][side_index] = singular
            row["wrist_null_scale"] = (
                _number(left_wrist_null_scale),
                _number(right_wrist_null_scale),
            )
            row["status"] = _text(status, 48)
            self._ik_next = index + 1
            return True
        except Exception as exc:  # noqa: BLE001 - trace cannot stop IK.
            return self._record_error("ik", exc)

    def stop_accepting(self) -> None:
        self._accepting = False

    def snapshot(self) -> dict[str, object]:
        return {
            "schema": SKILL_MOTION_TRACE_SCHEMA,
            "output_dir": str(self.output_dir),
            "accepting": self._accepting,
            "closed": self._closed,
            "counts": {
                "reference": self._reference_next,
                "ik": self._ik_next,
                "left_actuator": self._actuator_next["left"],
                "right_actuator": self._actuator_next["right"],
            },
            "drop_count": dict(self._drop_count),
            "error_count": dict(self._error_count),
            "last_error": dict(self._last_error),
        }

    @staticmethod
    def _percentiles(values: np.ndarray) -> dict[str, float | None]:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return {"max": None, "p99": None, "p95": None, "mean": None}
        return {
            "max": float(np.max(finite)),
            "p99": float(np.percentile(finite, 99)),
            "p95": float(np.percentile(finite, 95)),
            "mean": float(np.mean(finite)),
        }

    @staticmethod
    def _dynamics(q: np.ndarray, time_ns: np.ndarray) -> dict[str, float | None]:
        values = np.asarray(q, dtype=float)
        times = np.asarray(time_ns, dtype=np.int64)
        valid = np.all(np.isfinite(values), axis=1) & (times != _I64_MISSING)
        values = values[valid]
        times = times[valid]
        if values.shape[0] < 2:
            return {
                "max_joint_step_deg": None,
                "max_joint_velocity_deg_s": None,
                "max_joint_acceleration_deg_s2": None,
                "max_joint_jerk_deg_s3": None,
            }

        # A restarted or duplicated diagnostic clock must not create an
        # artificial high derivative.  Keep the first row, then only rows
        # strictly newer than the last retained row so every derivative uses
        # one coherent position/time sequence.
        retained = [0]
        last_time = int(times[0])
        for index in range(1, times.shape[0]):
            candidate = int(times[index])
            if candidate > last_time:
                retained.append(index)
                last_time = candidate
        if len(retained) < 2:
            return SkillMotionTraceRecorder._dynamics(values[:1], times[:1])
        values = values[retained]
        seconds = times[retained].astype(float) / 1e9
        dt = np.diff(seconds)
        step = np.diff(values, axis=0)
        velocity = step / dt[:, None]
        velocity_time = (seconds[1:] + seconds[:-1]) * 0.5
        acceleration = (
            np.diff(velocity, axis=0) / np.diff(velocity_time)[:, None]
            if velocity.shape[0] >= 2
            else np.empty((0, values.shape[1]))
        )
        acceleration_time = (
            (velocity_time[1:] + velocity_time[:-1]) * 0.5
            if velocity.shape[0] >= 2
            else np.empty(0)
        )
        jerk = (
            np.diff(acceleration, axis=0) / np.diff(acceleration_time)[:, None]
            if acceleration.shape[0] >= 2
            else np.empty((0, values.shape[1]))
        )

        def maximum(array: np.ndarray) -> float | None:
            return None if array.size == 0 else float(np.max(np.abs(array)) * 180.0 / np.pi)

        return {
            "max_joint_step_deg": maximum(step),
            "max_joint_velocity_deg_s": maximum(velocity),
            "max_joint_acceleration_deg_s2": maximum(acceleration),
            "max_joint_jerk_deg_s3": maximum(jerk),
        }

    @staticmethod
    def _paired_duration_ms(
        end_ns: np.ndarray,
        start_ns: np.ndarray,
    ) -> np.ndarray:
        """Return only valid non-negative paired monotonic durations."""

        end = np.asarray(end_ns, dtype=np.int64)
        start = np.asarray(start_ns, dtype=np.int64)
        if end.shape != start.shape:
            raise ValueError("paired trace timestamps must have matching shapes")
        valid = (
            (end != _I64_MISSING)
            & (start != _I64_MISSING)
            & (end >= start)
        )
        # Mask before subtraction: subtracting the int64 missing sentinel can
        # overflow and look like a plausible latency.
        return (end[valid].astype(float) - start[valid].astype(float)) / 1e6

    @staticmethod
    def _successive_gap_ms(time_ns: np.ndarray) -> np.ndarray:
        times = np.asarray(time_ns, dtype=np.int64)
        times = times[times != _I64_MISSING]
        if times.shape[0] < 2:
            return np.empty(0)
        gaps = np.diff(times.astype(float)) / 1e6
        return gaps[gaps >= 0.0]

    @staticmethod
    def _pose_errors(reference: np.ndarray) -> dict[str, float | None]:
        translations: list[float] = []
        rotations: list[float] = []
        for row in reference:
            side = int(row["active_side_index"])
            if side not in (0, 1):
                continue
            target = row["target_tool_tcp_pose_world"]
            measured = row["measured_tool_tcp_pose_world"][side]
            if not np.all(np.isfinite(target)) or not np.all(np.isfinite(measured)):
                continue
            translations.append(float(np.linalg.norm(target[:3, 3] - measured[:3, 3])))
            relative = target[:3, :3].T @ measured[:3, :3]
            cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
            rotations.append(float(np.degrees(np.arccos(cosine))))
        return {
            "final_translation_error_m": None if not translations else translations[-1],
            "max_translation_error_m": None if not translations else max(translations),
            "final_rotation_error_deg": None if not rotations else rotations[-1],
            "max_rotation_error_deg": None if not rotations else max(rotations),
        }

    @staticmethod
    def _unique_measurement_mask(reference: np.ndarray) -> np.ndarray:
        unique = np.zeros(reference.shape[0], dtype=bool)
        previous_sequence: tuple[int, int] | None = None
        for index, row in enumerate(reference):
            sequence = tuple(int(value) for value in row["source_sequence"])
            timestamp_ns = int(row["measurement_wall_time_ns"])
            if timestamp_ns <= 0 or any(value == _I64_MISSING for value in sequence):
                continue
            if sequence != previous_sequence:
                unique[index] = True
                previous_sequence = sequence
        return unique

    def _summary(self) -> dict[str, object]:
        reference = self._reference[: self._reference_next]
        ik = self._ik[: self._ik_next]
        unique_measurements = reference[self._unique_measurement_mask(reference)]
        measurement_intervals_ms = self._successive_gap_ms(
            unique_measurements["measurement_wall_time_ns"]
        )
        measurement_duration_ns = (
            0
            if unique_measurements.shape[0] < 2
            else int(unique_measurements["measurement_wall_time_ns"][-1])
            - int(unique_measurements["measurement_wall_time_ns"][0])
        )
        achieved_measurement_hz = (
            None
            if measurement_duration_ns <= 0
            else (unique_measurements.shape[0] - 1)
            / (measurement_duration_ns / 1_000_000_000.0)
        )
        measurement_stall_threshold_ms = (
            None
            if self.config.measurement_hz is None
            else 2000.0 / self.config.measurement_hz
        )
        outcomes = Counter(
            bytes(value).rstrip(b"\x00").decode("utf-8", errors="replace")
            for value in reference["outcome"]
        )
        branches = [
            bytes(value).rstrip(b"\x00").decode("utf-8", errors="replace")
            for value in reference["ik_branch"]
        ]
        branch_switches = sum(
            1
            for previous, current in pairwise(branches)
            if previous and current and previous != current
        )
        operations = sorted(
            {
                bytes(value).rstrip(b"\x00").decode("utf-8", errors="replace")
                for value in reference["operation_id"]
                if bytes(value).rstrip(b"\x00")
            }
        )
        result: dict[str, object] = {
            "schema": SKILL_MOTION_TRACE_SCHEMA,
            "operation_ids": operations,
            "reference_count": int(reference.shape[0]),
            "ik_shadow_count": int(ik.shape[0]),
            "outcomes": dict(sorted(outcomes.items())),
            "dispatch_succeeded_count": int(np.sum(reference["dispatch_succeeded"])),
            "expired_reference_count": int(np.sum(reference["expired_reference"])),
            "ik_time_budget_exceeded_count": int(
                np.sum(reference["ik_time_budget_exceeded"])
            ),
            "ik_branch_switch_count": int(branch_switches),
            "ik_profile": {
                "profiled_reference_count": int(
                    np.sum(np.isfinite(reference["ik_profile_solve_calls"]))
                ),
                "solve_calls": self._percentiles(
                    reference["ik_profile_solve_calls"].astype(float)
                ),
                "min_distance_calls": self._percentiles(
                    reference["ik_profile_min_distance_calls"].astype(float)
                ),
                **{
                    name.removeprefix("ik_profile_"): self._percentiles(
                        reference[name]
                    )
                    for name in (
                        "ik_profile_solve_total_ms",
                        "ik_profile_solve_qp_ms",
                        "ik_profile_solve_integrate_ms",
                        "ik_profile_solve_set_configuration_ms",
                        "ik_profile_min_distance_total_ms",
                    )
                },
            },
            "timing_ms": {
                name: self._percentiles(reference[name])
                for name in (
                    "control_cycle_ms",
                    "observation_ms",
                    "synchronization_ms",
                    "owner_requests_ms",
                    "ik_ms",
                    "admission_ms",
                    "dispatch_ms",
                    "measurement_age_at_dispatch_ms",
                    "ik_frame_total_ms",
                )
            },
            "collision_distance_m": {
                "min": (
                    None
                    if not np.any(np.isfinite(reference["collision_distance_m"]))
                    else float(np.nanmin(reference["collision_distance_m"]))
                )
            },
            "target_vs_measured_tcp": self._pose_errors(reference),
            "reference_dynamics": {},
            "measured_dynamics": {},
            "actuator_dynamics": {},
            "actuator_timing": {},
            "measurement_timing": {
                "target_hz": self.config.measurement_hz,
                "unique_measurement_count": int(unique_measurements.shape[0]),
                "achieved_hz": achieved_measurement_hz,
                "interval_ms": self._percentiles(measurement_intervals_ms),
                "stall_threshold_ms": measurement_stall_threshold_ms,
                "stall_count": (
                    None
                    if measurement_stall_threshold_ms is None
                    else int(
                        np.sum(
                            measurement_intervals_ms
                            > measurement_stall_threshold_ms
                        )
                    )
                ),
                "available_delay_ms": self._percentiles(
                    unique_measurements["measurement_available_delay_ms"]
                ),
                "measurement_span_ms": self._percentiles(
                    unique_measurements["measurement_span_ms"]
                ),
                "hardware_read_timing_by_side": {},
            },
            "trace_health": self.snapshot(),
        }
        for side_index, side in enumerate(_SIDES):
            result["reference_dynamics"][side] = self._dynamics(
                reference["qref_rad"][:, side_index],
                reference["cycle_finished_monotonic_ns"],
            )
            result["measured_dynamics"][side] = self._dynamics(
                unique_measurements["measured_qpos_rad"][:, side_index],
                unique_measurements["measurement_wall_time_ns"],
            )
            result["measurement_timing"]["hardware_read_timing_by_side"][side] = {
                source_field: self._percentiles(
                    unique_measurements[trace_field][:, side_index]
                )
                for trace_field, source_field in _HARDWARE_READ_TIMING_FIELDS
            }
            actuator = self._actuator[side][: self._actuator_next[side]]
            accepted = actuator["sdk_accepted"].astype(bool)
            accepted_rows = actuator[accepted]
            result["actuator_dynamics"][side] = self._dynamics(
                accepted_rows["q_cmd_rad"],
                accepted_rows["sdk_result_monotonic_ns"],
            )
            gaps_ms = self._successive_gap_ms(
                accepted_rows["sdk_result_monotonic_ns"]
            )
            moving_steps = (
                np.empty(0, dtype=bool)
                if accepted_rows.shape[0] < 2
                else np.nanmax(
                    np.abs(np.diff(accepted_rows["q_cmd_rad"], axis=0)),
                    axis=1,
                )
                > 1e-9
            )
            command_period_ms = 1000.0 / self.config.command_hz
            long_gap = gaps_ms >= 1.5 * command_period_ms
            post_stall_gap_ms = (
                np.empty(0, dtype=float)
                if gaps_ms.shape[0] < 2
                else gaps_ms[1:][long_gap[:-1]]
            )
            post_stall_moving = (
                np.empty(0, dtype=bool)
                if moving_steps.shape[0] < 2
                else moving_steps[1:][long_gap[:-1]]
            )
            result["actuator_timing"][side] = {
                "attempt_count": int(actuator.shape[0]),
                "accepted_count": int(accepted_rows.shape[0]),
                "send_gap_ms": self._percentiles(gaps_ms),
                "sdk_call_ms": self._percentiles(
                    self._paired_duration_ms(
                        actuator["sdk_result_monotonic_ns"],
                        actuator["sdk_attempt_monotonic_ns"],
                    )
                ),
                "wakeup_lag_ms": self._percentiles(
                    self._paired_duration_ms(
                        actuator["wakeup_monotonic_ns"],
                        actuator["scheduled_monotonic_ns"],
                    )
                ),
                "long_gap_threshold_ms": 1.5 * command_period_ms,
                "post_stall_minimum_gap_ms": (
                    None
                    if post_stall_gap_ms.shape[0] == 0
                    else float(np.min(post_stall_gap_ms))
                ),
                "post_stall_burst_count": int(
                    np.sum(
                        (post_stall_gap_ms < 0.5 * command_period_ms)
                        & post_stall_moving
                    )
                ),
                "moving_send_under_4ms_count": int(
                    np.sum((gaps_ms < 4.0) & moving_steps)
                ),
            }
        if ik.shape[0]:
            delta = np.abs(ik["admitted_q_rad"] - ik["raw_solver_q_rad"])
            result["raw_to_admitted_max_joint_delta_deg"] = (
                None
                if not np.any(np.isfinite(delta))
                else float(np.nanmax(delta) * 180.0 / np.pi)
            )
            result["ik_status_counts"] = dict(
                Counter(
                    bytes(value).rstrip(b"\x00").decode("utf-8", errors="replace")
                    for value in ik["status"]
                )
            )
        else:
            result["raw_to_admitted_max_joint_delta_deg"] = None
            result["ik_status_counts"] = {}
        return result

    def close(self) -> dict[str, object]:
        if self._closed:
            return self.snapshot()
        self.stop_accepting()
        arrays = {
            "reference": self._reference[: self._reference_next],
            "ik_shadow": self._ik[: self._ik_next],
            **{
                f"{side}_actuator": self._actuator[side][
                    : self._actuator_next[side]
                ]
                for side in _SIDES
            },
        }
        data_path = self.output_dir / SKILL_MOTION_TRACE_DATA_NAME
        temporary_data_path = self.output_dir / ("." + SKILL_MOTION_TRACE_DATA_NAME + ".tmp")
        with temporary_data_path.open("wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_data_path, data_path)
        summary = _json_safe(self._summary())
        summary_path = self.output_dir / SKILL_MOTION_TRACE_SUMMARY_NAME
        temporary_summary_path = self.output_dir / ("." + SKILL_MOTION_TRACE_SUMMARY_NAME + ".tmp")
        with temporary_summary_path.open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_summary_path, summary_path)
        stopped_wall_time_ns = time.time_ns()
        stopped_monotonic_time_ns = time.monotonic_ns()
        final_snapshot = self.snapshot()
        final_snapshot["closed"] = True
        manifest = {
            "schema": SKILL_MOTION_TRACE_SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "data_file": SKILL_MOTION_TRACE_DATA_NAME,
            "summary_file": SKILL_MOTION_TRACE_SUMMARY_NAME,
            "data_format": "numpy_npz_uncompressed_structured_arrays",
            "hot_path_policy": (
                "fixed_numeric_preallocated_memory; no queue/json/fk/disk_write; "
                "overflow drops diagnostics without blocking control"
            ),
            "clock_contract": {
                "wall_time": "host CLOCK_REALTIME converted from seconds to ns",
                "monotonic_time": "host CLOCK_MONOTONIC converted from seconds to ns",
            },
            "side_order": list(_SIDES),
            "reference_dtype": _REFERENCE_DTYPE.descr,
            "actuator_dtype": _ACTUATOR_DTYPE.descr,
            "ik_shadow_dtype": _IK_DTYPE.descr,
            "started_wall_time_ns": self._started_wall_time_ns,
            "started_monotonic_time_ns": self._started_monotonic_time_ns,
            "stopped_wall_time_ns": stopped_wall_time_ns,
            "stopped_monotonic_time_ns": stopped_monotonic_time_ns,
            "config": {
                "reference_hz": self.config.reference_hz,
                "command_hz": self.config.command_hz,
                "measurement_hz": self.config.measurement_hz,
                "max_duration_sec": self.config.max_duration_sec,
                "reference_capacity": self.config.reference_capacity,
                "command_capacity_per_side": self.config.command_capacity_per_side,
            },
            "runtime_metadata": _json_safe(dict(self.config.metadata)),
            "result": final_snapshot,
        }
        manifest_path = self.output_dir / SKILL_MOTION_TRACE_MANIFEST_NAME
        temporary_manifest_path = self.output_dir / ("." + SKILL_MOTION_TRACE_MANIFEST_NAME + ".tmp")
        with temporary_manifest_path.open("w", encoding="utf-8") as stream:
            json.dump(_json_safe(manifest), stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_manifest_path, manifest_path)
        self._closed = True
        return self.snapshot()


def inspect_trace(path: str | Path) -> dict[str, object]:
    root = Path(path).expanduser().resolve()
    summary_path = root / SKILL_MOTION_TRACE_SUMMARY_NAME
    manifest_path = root / SKILL_MOTION_TRACE_MANIFEST_NAME
    if not summary_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"motion trace is incomplete: {root}")
    return {
        "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect an exclusive RM65 Skill motion trace")
    parser.add_argument("command", choices=("inspect",))
    parser.add_argument("trace_dir", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(inspect_trace(args.trace_dir), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_SKILL_MOTION_TRACE_MAX_DURATION_SEC",
    "SKILL_MOTION_TRACE_DATA_NAME",
    "SKILL_MOTION_TRACE_MANIFEST_NAME",
    "SKILL_MOTION_TRACE_SCHEMA",
    "SKILL_MOTION_TRACE_SUMMARY_NAME",
    "SkillMotionTraceConfig",
    "SkillMotionTraceRecorder",
    "inspect_trace",
]
