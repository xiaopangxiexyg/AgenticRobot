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


ACTUATOR_COMMAND_TRACE_SCHEMA = "dexteleop.rm65_actuator_command_trace.v2"
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
        self._next_index = {side: 0 for side in config.active_sides}
        self._singularity_next_index = {
            side: 0 for side in config.active_sides
        }
        self._drop_count = {side: 0 for side in config.active_sides}
        self._singularity_drop_count = {
            side: 0 for side in config.active_sides
        }
        self._error_count = {side: 0 for side in config.active_sides}
        self._singularity_error_count = {
            side: 0 for side in config.active_sides
        }
        self._last_error = {side: None for side in config.active_sides}
        self._last_singularity_error = {
            side: None for side in config.active_sides
        }
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
