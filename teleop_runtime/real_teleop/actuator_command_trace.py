"""Low-overhead real-arm command trace for actuator identification.

The hot path only copies fixed-size numeric fields into preallocated NumPy
buffers.  Files are created at setup time but command rows are persisted only
after the command workers have been detached and stopped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np


ACTUATOR_COMMAND_TRACE_SCHEMA = "dexteleop.rm65_actuator_command_trace.v4"
ACTUATOR_COMMAND_TRACE_DATA_NAME = "command_trace.npz"
ACTUATOR_COMMAND_TRACE_MANIFEST_NAME = "manifest.json"
DEFAULT_ACTUATOR_TRACE_ROOT = Path("teleop_records/actuator_identification")
DEFAULT_ACTUATOR_TRACE_MAX_DURATION_SEC = 600.0

_INT64_MISSING = np.iinfo(np.int64).min
_INT32_MISSING = np.iinfo(np.int32).min
_COMMAND_DTYPE = np.dtype(
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
_SINGULARITY_DTYPE = np.dtype(
    [
        ("event_sequence", "<i8"),
        ("target_sequence", "<i8"),
        ("monotonic_time_ns", "<i8"),
        ("candidate_certificate_sequence", "<i8"),
        ("target_singularity_code", "<i4"),
        ("current_singularity_code", "<i4"),
        ("fully_non_singular_count", "<i4"),
        ("fallback_requested_mask", "u1"),
        ("fallback_allowed_mask", "u1"),
        ("selected_singular_mask", "u1"),
        ("candidate_certificate_valid", "u1"),
        ("gate_status", "S48"),
        ("target_label", "S32"),
        ("current_label", "S32"),
        ("candidate_certificate_status", "S40"),
        ("qref_fingerprint", "S32"),
        ("q_ref_rad", "<f8", (6,)),
        ("reference_qpos_rad", "<f8", (6,)),
        ("target_wrist_center_distance_m", "<f8"),
        ("current_wrist_center_distance_m", "<f8"),
    ],
    align=False,
)
_IK_SHADOW_DTYPE = np.dtype(
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
        ("status", "S32"),
    ],
    align=False,
)
_VIVE_SOURCE_DTYPE = np.dtype(
    [
        ("sequence", "<i8"),
        ("control_wall_time_ns", "<i8"),
        ("control_monotonic_time_ns", "<i8"),
        ("present_mask", "u1"),
        ("valid_mask", "u1"),
        ("velocity_valid_mask", "u1"),
        ("channel_seq", "<i8", (2,)),
        ("channel_timestamp_ns", "<i8", (2,)),
        ("receiver_timestamp_ns", "<i8", (2,)),
        ("receiver_monotonic_time_ns", "<i8", (2,)),
        ("availability_monotonic_time_ns", "<i8", (2,)),
        ("pose_sample_monotonic_time_ns", "<i8", (2,)),
        ("capture_receiver_monotonic_estimate_ns", "<i8", (2,)),
        ("sender_timestamp_sec", "<f8", (2,)),
        ("pose_sample_timestamp_sec", "<f8", (2,)),
        ("capture_estimate_uncertainty_sec", "<f8", (2,)),
        ("clock_sync_offset_sec", "<f8", (2,)),
        ("pose_vive_world", "<f8", (2, 4, 4)),
        ("linear_velocity_m_s", "<f8", (2, 3)),
        ("angular_velocity_rad_s", "<f8", (2, 3)),
        ("tracker_name", "S64", (2,)),
        ("tracker_role", "S32", (2,)),
        ("velocity_frame_name", "S32", (2,)),
        ("velocity_source", "S64", (2,)),
        ("velocity_error", "S128", (2,)),
        ("teleop_state", "S24"),
        ("motion_hold_mode", "S32"),
    ],
    align=False,
)


def _normalize_side(side: object) -> str:
    value = str(side).strip().lower()
    if value not in {"left", "right"}:
        raise ValueError(f"Unsupported actuator trace side: {side!r}")
    return value


def _optional_i64(value: object) -> int:
    return int(_INT64_MISSING) if value is None else int(value)


def _optional_i32(value: object) -> int:
    return int(_INT32_MISSING) if value is None else int(value)


def _seconds_to_ns(value: object) -> int:
    if value is None:
        return int(_INT64_MISSING)
    seconds = float(value)
    if not math.isfinite(seconds):
        return int(_INT64_MISSING)
    return int(round(seconds * 1_000_000_000.0))


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
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
class ActuatorCommandTraceConfig:
    output_dir: Path
    active_sides: tuple[str, ...]
    command_hz: float
    max_duration_sec: float = DEFAULT_ACTUATOR_TRACE_MAX_DURATION_SEC
    reference_hz: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sides = tuple(_normalize_side(side) for side in self.active_sides)
        if not sides or len(set(sides)) != len(sides):
            raise ValueError("active_sides must contain unique left/right entries")
        if not math.isfinite(float(self.command_hz)) or float(self.command_hz) <= 0.0:
            raise ValueError("command_hz must be finite and positive")
        if (
            not math.isfinite(float(self.max_duration_sec))
            or float(self.max_duration_sec) <= 0.0
        ):
            raise ValueError("max_duration_sec must be finite and positive")
        reference_hz = (
            float(self.command_hz)
            if self.reference_hz is None
            else float(self.reference_hz)
        )
        if not math.isfinite(reference_hz) or reference_hz <= 0.0:
            raise ValueError("reference_hz must be finite and positive")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "active_sides", sides)
        object.__setattr__(self, "reference_hz", reference_hz)

    @property
    def capacity_per_side(self) -> int:
        # Two extra seconds absorb startup/shutdown boundary jitter without
        # allocating or growing a buffer while commands are active.
        return int(
            math.ceil(
                (float(self.max_duration_sec) + 2.0) * float(self.command_hz)
            )
        )

    @property
    def singularity_capacity_per_side(self) -> int:
        return int(
            math.ceil(
                (float(self.max_duration_sec) + 2.0)
                * float(self.reference_hz or self.command_hz)
            )
        )


class ActuatorCommandTraceRecorder:
    """Single-producer-per-side, fixed-buffer command trace.

    ``record_command`` is intentionally a positional API: the RM65 command
    worker does not construct dictionaries, dataclasses, JSON, or queue items.
    Exceptions are contained and exposed in the final manifest.
    """

    def __init__(self, config: ActuatorCommandTraceConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        capacity = int(config.capacity_per_side)
        self._buffers: dict[str, np.ndarray] = {}
        self._singularity_buffers: dict[str, np.ndarray] = {}
        for side in config.active_sides:
            buffer = np.empty(capacity, dtype=_COMMAND_DTYPE)
            # Touch every page during setup.  Otherwise Linux may defer page
            # faults until individual command samples arrive.
            buffer.view(np.uint8).fill(0)
            self._buffers[side] = buffer
            singularity_buffer = np.empty(
                int(config.singularity_capacity_per_side),
                dtype=_SINGULARITY_DTYPE,
            )
            singularity_buffer.view(np.uint8).fill(0)
            self._singularity_buffers[side] = singularity_buffer
        self._ik_shadow_buffer = np.empty(
            int(config.singularity_capacity_per_side),
            dtype=_IK_SHADOW_DTYPE,
        )
        self._ik_shadow_buffer.view(np.uint8).fill(0)
        self._vive_source_buffer = np.empty(
            int(config.singularity_capacity_per_side),
            dtype=_VIVE_SOURCE_DTYPE,
        )
        self._vive_source_buffer.view(np.uint8).fill(0)
        self._next_index = {side: 0 for side in config.active_sides}
        self._singularity_next_index = {
            side: 0 for side in config.active_sides
        }
        self._ik_shadow_next_index = 0
        self._vive_source_next_index = 0
        self._drop_count = {side: 0 for side in config.active_sides}
        self._singularity_drop_count = {
            side: 0 for side in config.active_sides
        }
        self._ik_shadow_drop_count = 0
        self._vive_source_drop_count = 0
        self._error_count = {side: 0 for side in config.active_sides}
        self._singularity_error_count = {
            side: 0 for side in config.active_sides
        }
        self._ik_shadow_error_count = 0
        self._vive_source_error_count = 0
        self._last_error = {side: None for side in config.active_sides}
        self._last_singularity_error = {
            side: None for side in config.active_sides
        }
        self._last_ik_shadow_error: str | None = None
        self._last_vive_source_error: str | None = None
        self._accepting = True
        self._closed = False
        self._started_wall_time_ns = time.time_ns()
        self._started_monotonic_time_ns = time.monotonic_ns()

    def record_command(
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
            buffer = self._buffers[normalized_side]
            index = self._next_index[normalized_side]
            if index >= buffer.shape[0]:
                self._drop_count[normalized_side] += 1
                return False
            command = np.asarray(q_cmd, dtype=float)
            # RM65 validates the final joint vector immediately before the SDK
            # call.  Rechecking finiteness here would allocate a temporary bool
            # vector on every traced command.
            if command.shape != (6,):
                raise ValueError("q_cmd must be a 6-vector")

            buffer["tick_sequence"][index] = int(tick_sequence)
            buffer["target_sequence"][index] = _optional_i64(target_sequence)
            buffer["reference_epoch"][index] = _optional_i64(reference_epoch)
            buffer["reference_sequence"][index] = _optional_i64(reference_sequence)
            buffer["sdk_attempt_sequence"][index] = _optional_i64(
                sdk_attempt_sequence
            )
            buffer["sdk_accepted_sequence"][index] = _optional_i64(
                sdk_accepted_sequence
            )
            buffer["scheduled_monotonic_ns"][index] = _seconds_to_ns(
                scheduled_monotonic_time
            )
            buffer["wakeup_monotonic_ns"][index] = _seconds_to_ns(
                wakeup_monotonic_time
            )
            buffer["sdk_attempt_wall_time_ns"][index] = _seconds_to_ns(
                sdk_attempt_wall_time
            )
            buffer["sdk_attempt_monotonic_ns"][index] = _seconds_to_ns(
                sdk_attempt_monotonic_time
            )
            buffer["sdk_result_monotonic_ns"][index] = _seconds_to_ns(
                sdk_result_monotonic_time
            )
            buffer["sdk_result_code"][index] = _optional_i32(sdk_result_code)
            buffer["sdk_accepted"][index] = int(bool(sdk_accepted))
            buffer["q_cmd_rad"][index] = command
            self._next_index[normalized_side] = index + 1
            return True
        except Exception as exc:  # noqa: BLE001 - tracing must never stop control.
            if normalized_side in self._error_count:
                self._error_count[normalized_side] += 1
                self._last_error[normalized_side] = (
                    f"{type(exc).__name__}: {exc}"
                )
            return False

    def record_singularity_transition(
        self,
        side: str,
        target_sequence: int,
        monotonic_time: float,
        singularity_metadata: Mapping[str, object],
        q_ref: Sequence[float] | np.ndarray,
        reference_qpos: Sequence[float] | np.ndarray,
    ) -> bool:
        """Record one sparse transition/non-normal singularity sample in memory."""

        if not self._accepting:
            return False
        normalized_side = str(side)
        try:
            buffer = self._singularity_buffers[normalized_side]
            index = self._singularity_next_index[normalized_side]
            if index >= buffer.shape[0]:
                self._singularity_drop_count[normalized_side] += 1
                return False
            qref = np.asarray(q_ref, dtype=float)
            reference = np.asarray(reference_qpos, dtype=float)
            if qref.shape != (6,) or reference.shape != (6,):
                raise ValueError("singularity trace qref/reference must be 6-vectors")

            metadata = singularity_metadata
            validation = metadata.get("position_candidate_certificate_validation")
            validation = validation if isinstance(validation, Mapping) else {}
            certificate = validation.get("certificate")
            certificate = certificate if isinstance(certificate, Mapping) else {}
            preference = metadata.get("position_singularity_fallback")
            preference = preference if isinstance(preference, Mapping) else {}
            final_position = certificate.get("final_qref_position_singularity")
            final_position = final_position if isinstance(final_position, Mapping) else {}
            sides = final_position.get("sides")
            sides = sides if isinstance(sides, Mapping) else {}
            selected_side = sides.get(normalized_side)
            selected_side = selected_side if isinstance(selected_side, Mapping) else {}

            requested_mask = (
                int(bool(metadata.get("position_wrist_singularity_fallback_requested")))
                | (
                    int(bool(metadata.get("position_elbow_singularity_fallback_requested")))
                    << 1
                )
                | (
                    int(bool(metadata.get("position_shoulder_singularity_fallback_requested")))
                    << 2
                )
            )
            allowed_mask = (
                int(bool(metadata.get("position_wrist_singularity_fallback_allowed")))
                | (
                    int(bool(metadata.get("position_elbow_singularity_fallback_allowed")))
                    << 1
                )
                | (
                    int(bool(metadata.get("position_shoulder_singularity_fallback_allowed")))
                    << 2
                )
            )
            selected_mask = (
                int(bool(selected_side.get("is_wrist_singular")))
                | (int(bool(selected_side.get("is_elbow_singular"))) << 1)
                | (int(bool(selected_side.get("is_shoulder_singular"))) << 2)
            )
            target_distance = metadata.get("target_wrist_center_distance_m")
            current_distance = metadata.get("current_wrist_center_distance_m")

            buffer["event_sequence"][index] = int(index)
            buffer["target_sequence"][index] = int(target_sequence)
            buffer["monotonic_time_ns"][index] = _seconds_to_ns(monotonic_time)
            buffer["candidate_certificate_sequence"][index] = _optional_i64(
                validation.get("sequence")
            )
            buffer["target_singularity_code"][index] = _optional_i32(
                metadata.get("target_singularity_code")
            )
            buffer["current_singularity_code"][index] = _optional_i32(
                metadata.get("current_singularity_code")
            )
            buffer["fully_non_singular_count"][index] = _optional_i32(
                preference.get("fully_non_singular_count")
            )
            buffer["fallback_requested_mask"][index] = requested_mask
            buffer["fallback_allowed_mask"][index] = allowed_mask
            buffer["selected_singular_mask"][index] = selected_mask
            buffer["candidate_certificate_valid"][index] = int(
                bool(validation.get("valid", False))
            )
            buffer["gate_status"][index] = str(
                metadata.get("singularity_gate_status") or ""
            ).encode("utf-8")
            buffer["target_label"][index] = str(
                metadata.get("target_singularity_label") or ""
            ).encode("utf-8")
            buffer["current_label"][index] = str(
                metadata.get("current_singularity_label") or ""
            ).encode("utf-8")
            buffer["candidate_certificate_status"][index] = str(
                validation.get("status") or ""
            ).encode("utf-8")
            buffer["qref_fingerprint"][index] = str(
                validation.get("actual_qref_fingerprint") or ""
            ).encode("ascii", errors="ignore")
            buffer["q_ref_rad"][index] = qref
            buffer["reference_qpos_rad"][index] = reference
            buffer["target_wrist_center_distance_m"][index] = (
                float("nan") if target_distance is None else float(target_distance)
            )
            buffer["current_wrist_center_distance_m"][index] = (
                float("nan") if current_distance is None else float(current_distance)
            )
            self._singularity_next_index[normalized_side] = index + 1
            return True
        except Exception as exc:  # noqa: BLE001 - tracing must never stop control.
            if normalized_side in self._singularity_error_count:
                self._singularity_error_count[normalized_side] += 1
                self._last_singularity_error[normalized_side] = (
                    f"{type(exc).__name__}: {exc}"
                )
            return False

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
        """Record one 60 Hz IK shadow row into preallocated memory."""

        if not self._accepting:
            return False
        try:
            index = self._ik_shadow_next_index
            buffer = self._ik_shadow_buffer
            if index >= buffer.shape[0]:
                self._ik_shadow_drop_count += 1
                return False

            def q12(value: object, *, allow_missing: bool = False) -> np.ndarray:
                if value is None and allow_missing:
                    return np.full(12, np.nan, dtype=float)
                array = np.asarray(value, dtype=float)
                if array.shape != (12,):
                    raise ValueError("IK shadow q fields must be 12-vectors")
                return array

            def sigma6(value: object) -> np.ndarray:
                if value is None:
                    return np.full(6, np.nan, dtype=float)
                array = np.asarray(value, dtype=float)
                if array.shape != (6,):
                    raise ValueError(
                        "IK shadow singular-value fields must be 6-vectors"
                    )
                return array

            left_pose = np.asarray(left_target_world_pose, dtype=float)
            right_pose = np.asarray(right_target_world_pose, dtype=float)
            if left_pose.shape != (4, 4) or right_pose.shape != (4, 4):
                raise ValueError("IK shadow target poses must be 4x4 matrices")

            buffer["sequence"][index] = int(index)
            buffer["monotonic_time_ns"][index] = _seconds_to_ns(monotonic_time)
            buffer["pre_q_rad"][index] = q12(pre_q)
            buffer["raw_solver_q_rad"][index] = q12(
                raw_solver_q,
                allow_missing=True,
            )
            buffer["canonical_solver_q_rad"][index] = q12(
                canonical_solver_q,
                allow_missing=True,
            )
            buffer["admitted_q_rad"][index] = q12(admitted_q)
            buffer["target_world_pose"][index, 0] = left_pose
            buffer["target_world_pose"][index, 1] = right_pose
            buffer["jacobian_singular_values"][index, 0] = sigma6(
                left_jacobian_singular_values
            )
            buffer["jacobian_singular_values"][index, 1] = sigma6(
                right_jacobian_singular_values
            )
            buffer["wrist_null_scale"][index] = (
                np.nan
                if left_wrist_null_scale is None
                else float(left_wrist_null_scale),
                np.nan
                if right_wrist_null_scale is None
                else float(right_wrist_null_scale),
            )
            buffer["status"][index] = str(status).encode("utf-8")
            self._ik_shadow_next_index = index + 1
            return True
        except Exception as exc:  # noqa: BLE001 - tracing must never stop control.
            self._ik_shadow_error_count += 1
            self._last_ik_shadow_error = f"{type(exc).__name__}: {exc}"
            return False

    def record_vive_source(
        self,
        *,
        control_wall_time: float,
        control_monotonic_time_ns: int,
        raw_channels: Mapping[str, object],
        teleop_state: str,
        motion_hold_mode: str,
    ) -> bool:
        """Copy one pre-filter logical wrist-pose cut into fixed memory."""

        if not self._accepting:
            return False
        try:
            index = self._vive_source_next_index
            buffer = self._vive_source_buffer
            if index >= buffer.shape[0]:
                self._vive_source_drop_count += 1
                return False
            row = buffer[index]
            row["sequence"] = index
            row["control_wall_time_ns"] = _seconds_to_ns(control_wall_time)
            row["control_monotonic_time_ns"] = int(control_monotonic_time_ns)
            row["present_mask"] = 0
            row["valid_mask"] = 0
            row["velocity_valid_mask"] = 0
            for field_name in (
                "channel_seq",
                "channel_timestamp_ns",
                "receiver_timestamp_ns",
                "receiver_monotonic_time_ns",
                "availability_monotonic_time_ns",
                "pose_sample_monotonic_time_ns",
                "capture_receiver_monotonic_estimate_ns",
            ):
                row[field_name].fill(_INT64_MISSING)
            for field_name in (
                "sender_timestamp_sec",
                "pose_sample_timestamp_sec",
                "capture_estimate_uncertainty_sec",
                "clock_sync_offset_sec",
                "pose_vive_world",
                "linear_velocity_m_s",
                "angular_velocity_rad_s",
            ):
                row[field_name].fill(np.nan)
            row["tracker_name"].fill(b"")
            row["tracker_role"].fill(b"")
            row["velocity_frame_name"].fill(b"")
            row["velocity_source"].fill(b"")
            row["velocity_error"].fill(b"")
            for side_index, channel_name in enumerate(
                ("left_wrist_pose", "right_wrist_pose")
            ):
                channel = raw_channels.get(channel_name)
                if channel is None:
                    continue
                pose = np.asarray(getattr(channel, "value"), dtype=float)
                if pose.shape != (4, 4):
                    raise ValueError(
                        f"{channel_name} source pose must be a 4x4 matrix"
                    )
                metadata = getattr(channel, "metadata", {}) or {}
                row["present_mask"] |= 1 << side_index
                if bool(getattr(channel, "valid", False)):
                    row["valid_mask"] |= 1 << side_index
                row["channel_seq"][side_index] = _optional_i64(metadata.get("seq"))
                row["channel_timestamp_ns"][side_index] = _seconds_to_ns(
                    getattr(channel, "timestamp", None)
                )
                for field_name, metadata_name in (
                    ("receiver_timestamp_ns", "receiver_timestamp"),
                    ("receiver_monotonic_time_ns", "receiver_monotonic_time"),
                    (
                        "availability_monotonic_time_ns",
                        "availability_monotonic_time",
                    ),
                    (
                        "pose_sample_monotonic_time_ns",
                        "pose_sample_monotonic_time",
                    ),
                    (
                        "capture_receiver_monotonic_estimate_ns",
                        "capture_receiver_monotonic_estimate",
                    ),
                ):
                    row[field_name][side_index] = _seconds_to_ns(
                        metadata.get(metadata_name)
                    )
                for field_name, metadata_name in (
                    ("sender_timestamp_sec", "sender_timestamp"),
                    ("pose_sample_timestamp_sec", "pose_sample_timestamp"),
                    (
                        "capture_estimate_uncertainty_sec",
                        "capture_estimate_uncertainty_sec",
                    ),
                    ("clock_sync_offset_sec", "clock_sync_offset_sec"),
                ):
                    value = metadata.get(metadata_name)
                    row[field_name][side_index] = (
                        np.nan if value is None else float(value)
                    )
                row["pose_vive_world"][side_index] = pose
                linear_velocity = metadata.get("linear_velocity_m_s")
                angular_velocity = metadata.get("angular_velocity_rad_s")
                velocity_vectors_valid = False
                if linear_velocity is not None and angular_velocity is not None:
                    linear_array = np.asarray(linear_velocity, dtype=float)
                    angular_array = np.asarray(angular_velocity, dtype=float)
                    velocity_vectors_valid = bool(
                        linear_array.shape == (3,)
                        and angular_array.shape == (3,)
                        and np.all(np.isfinite(linear_array))
                        and np.all(np.isfinite(angular_array))
                    )
                    if velocity_vectors_valid:
                        row["linear_velocity_m_s"][side_index] = linear_array
                        row["angular_velocity_rad_s"][side_index] = angular_array
                if bool(metadata.get("velocity_valid", False)) and velocity_vectors_valid:
                    row["velocity_valid_mask"] |= 1 << side_index
                row["tracker_name"][side_index] = str(
                    metadata.get("tracker_name") or ""
                ).encode("utf-8")
                row["tracker_role"][side_index] = str(
                    metadata.get("tracker_role") or ""
                ).encode("utf-8")
                row["velocity_frame_name"][side_index] = str(
                    metadata.get("velocity_frame_name") or ""
                ).encode("utf-8")
                row["velocity_source"][side_index] = str(
                    metadata.get("velocity_source") or ""
                ).encode("utf-8")
                row["velocity_error"][side_index] = str(
                    metadata.get("velocity_error") or ""
                ).encode("utf-8")
            row["teleop_state"] = str(teleop_state).encode("utf-8")
            row["motion_hold_mode"] = str(motion_hold_mode).encode("utf-8")
            self._vive_source_next_index = index + 1
            return True
        except Exception as exc:  # noqa: BLE001 - tracing cannot stop control.
            self._vive_source_error_count += 1
            self._last_vive_source_error = f"{type(exc).__name__}: {exc}"
            return False

    def stop_accepting(self) -> None:
        self._accepting = False

    def snapshot(self) -> dict[str, object]:
        return {
            "output_dir": str(self.output_dir),
            "capacity_per_side": int(self.config.capacity_per_side),
            "preallocated_bytes_total": int(
                sum(buffer.nbytes for buffer in self._buffers.values())
                + sum(
                    buffer.nbytes
                    for buffer in self._singularity_buffers.values()
                )
                + self._ik_shadow_buffer.nbytes
                + self._vive_source_buffer.nbytes
            ),
            "rows_by_side": {
                side: int(self._next_index[side]) for side in self.config.active_sides
            },
            "drop_count_by_side": {
                side: int(self._drop_count[side]) for side in self.config.active_sides
            },
            "singularity_rows_by_side": {
                side: int(self._singularity_next_index[side])
                for side in self.config.active_sides
            },
            "singularity_drop_count_by_side": {
                side: int(self._singularity_drop_count[side])
                for side in self.config.active_sides
            },
            "singularity_error_count_by_side": {
                side: int(self._singularity_error_count[side])
                for side in self.config.active_sides
            },
            "ik_shadow_rows": int(self._ik_shadow_next_index),
            "ik_shadow_drop_count": int(self._ik_shadow_drop_count),
            "ik_shadow_error_count": int(self._ik_shadow_error_count),
            "last_ik_shadow_error": self._last_ik_shadow_error,
            "vive_source_rows": int(self._vive_source_next_index),
            "vive_source_drop_count": int(self._vive_source_drop_count),
            "vive_source_error_count": int(self._vive_source_error_count),
            "last_vive_source_error": self._last_vive_source_error,
            "last_singularity_error_by_side": dict(
                self._last_singularity_error
            ),
            "error_count_by_side": {
                side: int(self._error_count[side]) for side in self.config.active_sides
            },
            "last_error_by_side": dict(self._last_error),
            "accepting": bool(self._accepting),
            "closed": bool(self._closed),
        }

    def close(self) -> None:
        if self._closed:
            return
        self.stop_accepting()
        stopped_wall_time_ns = time.time_ns()
        stopped_monotonic_time_ns = time.monotonic_ns()

        data_path = self.output_dir / ACTUATOR_COMMAND_TRACE_DATA_NAME
        temporary_data_path = self.output_dir / f".{ACTUATOR_COMMAND_TRACE_DATA_NAME}.tmp"
        arrays = {
            side: self._buffers[side][: self._next_index[side]]
            for side in self.config.active_sides
        }
        arrays.update(
            {
                f"{side}_singularity": self._singularity_buffers[side][
                    : self._singularity_next_index[side]
                ]
                for side in self.config.active_sides
            }
        )
        arrays["ik_shadow"] = self._ik_shadow_buffer[
            : self._ik_shadow_next_index
        ]
        arrays["vive_source"] = self._vive_source_buffer[
            : self._vive_source_next_index
        ]
        with temporary_data_path.open("wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_data_path, data_path)

        result = self.snapshot()
        result["closed"] = True
        manifest = {
            "schema": ACTUATOR_COMMAND_TRACE_SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "data_file": ACTUATOR_COMMAND_TRACE_DATA_NAME,
            "data_format": (
                "numpy_npz_uncompressed_structured_arrays_by_side; "
                "command=<side>, singularity=<side>_singularity"
            ),
            "hot_path_policy": (
                "fixed_numeric_preallocated_memory; no queue/json/fk/disk_write"
            ),
            "clock_contract": {
                "command_monotonic": "host CLOCK_MONOTONIC converted from seconds to ns",
                "sdk_attempt_wall": "host CLOCK_REALTIME converted from seconds to ns",
                "udp_alignment": (
                    "same-host UDP capture receive_monotonic_time_ns is comparable"
                ),
            },
            "joint_contract": {
                "q_cmd_rad_order": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
                "sdk_udp_feedback_position_unit": "degree",
            },
            "missing_integer_sentinels": {
                "int64": int(_INT64_MISSING),
                "int32": int(_INT32_MISSING),
            },
            "dtype": _COMMAND_DTYPE.descr,
            "singularity_dtype": _SINGULARITY_DTYPE.descr,
            "ik_shadow_dtype": _IK_SHADOW_DTYPE.descr,
            "vive_source_dtype": _VIVE_SOURCE_DTYPE.descr,
            "started_wall_time_ns": int(self._started_wall_time_ns),
            "started_monotonic_time_ns": int(self._started_monotonic_time_ns),
            "stopped_wall_time_ns": int(stopped_wall_time_ns),
            "stopped_monotonic_time_ns": int(stopped_monotonic_time_ns),
            "config": {
                "active_sides": list(self.config.active_sides),
                "command_hz": float(self.config.command_hz),
                "reference_hz": float(self.config.reference_hz or self.config.command_hz),
                "max_duration_sec": float(self.config.max_duration_sec),
                "capacity_per_side": int(self.config.capacity_per_side),
                "singularity_capacity_per_side": int(
                    self.config.singularity_capacity_per_side
                ),
            },
            "result": result,
            "runtime_metadata": _json_safe(dict(self.config.metadata)),
        }
        manifest_path = self.output_dir / ACTUATOR_COMMAND_TRACE_MANIFEST_NAME
        temporary_manifest_path = self.output_dir / (
            f".{ACTUATOR_COMMAND_TRACE_MANIFEST_NAME}.tmp"
        )
        with temporary_manifest_path.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_manifest_path, manifest_path)
        self._closed = True


__all__ = [
    "ACTUATOR_COMMAND_TRACE_DATA_NAME",
    "ACTUATOR_COMMAND_TRACE_MANIFEST_NAME",
    "ACTUATOR_COMMAND_TRACE_SCHEMA",
    "DEFAULT_ACTUATOR_TRACE_MAX_DURATION_SEC",
    "DEFAULT_ACTUATOR_TRACE_ROOT",
    "ActuatorCommandTraceConfig",
    "ActuatorCommandTraceRecorder",
]
