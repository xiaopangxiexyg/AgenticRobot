from __future__ import annotations

import argparse
import base64
import json
import math
import platform
import re
import socket
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence, TextIO

from project_paths import teleop_records_root
from runtime_artifact_paths import (
    LEGACY_ARTIFACT_LAYOUT_CLI_NAME,
    RUNTIME_ARTIFACT_LAYOUT_CLI_NAME,
    RuntimeArtifactLayout,
    configured_runtime_artifact_layout,
    resolve_artifact_write_path,
)


EXPECTED_PROTOCOL = "teleop_runtime.vive/v1"
DEFAULT_EXPECTED_HZ = 300.0
DEFAULT_RECEIVE_BUFFER_BYTES = 4 * 1024 * 1024
DEFAULT_FLUSH_EVERY_PACKETS = 300
DEFAULT_SOCKET_TIMEOUT_SEC = 0.25
MAX_UDP_DATAGRAM_BYTES = 65535


Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]


@dataclass(frozen=True)
class PoseHistory:
    position_m: Vector3
    rotation: Matrix3
    pose_matrix_4x4: tuple[tuple[float, float, float, float], ...]
    sample_clock: str
    sample_time: float
    receiver_monotonic_time: float
    seq: int | None


@dataclass
class TrackerStatistics:
    rows: int = 0
    valid_pose_rows: int = 0
    invalid_pose_rows: int = 0
    valid_velocity_rows: int = 0
    invalid_velocity_rows: int = 0
    repeated_pose_rows: int = 0
    contiguous_pose_pairs: int = 0
    metrics: dict[str, list[float]] = field(default_factory=dict)

    def add_metric(self, name: str, value: float | None) -> None:
        if value is None or not math.isfinite(value):
            return
        self.metrics.setdefault(name, []).append(float(value))


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, not bool")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def _optional_finite_float(
    value: object,
    *,
    label: str,
    errors: list[str],
) -> float | None:
    if value is None:
        return None
    try:
        return _finite_float(value, label=label)
    except ValueError as exc:
        errors.append(str(exc))
        return None


def _vector3(value: object, *, label: str) -> Vector3:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must contain exactly 3 values")
    return (
        _finite_float(value[0], label=f"{label}[0]"),
        _finite_float(value[1], label=f"{label}[1]"),
        _finite_float(value[2], label=f"{label}[2]"),
    )


def _pose_matrix4x4(
    value: object,
) -> tuple[tuple[float, float, float, float], ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("pose_matrix_4x4 must contain exactly 4 rows")
    rows: list[tuple[float, float, float, float]] = []
    for row_index, raw_row in enumerate(value):
        if not isinstance(raw_row, (list, tuple)) or len(raw_row) != 4:
            raise ValueError(f"pose_matrix_4x4[{row_index}] must contain exactly 4 values")
        rows.append(
            tuple(
                _finite_float(
                    raw_row[column_index],
                    label=f"pose_matrix_4x4[{row_index}][{column_index}]",
                )
                for column_index in range(4)
            )  # type: ignore[arg-type]
        )
    return tuple(rows)


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in vector))


def _sub(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2],
    )


def _scale(vector: Vector3, scale: float) -> Vector3:
    return (
        vector[0] * scale,
        vector[1] * scale,
        vector[2] * scale,
    )


def _transpose(matrix: Matrix3) -> Matrix3:
    return (
        (matrix[0][0], matrix[1][0], matrix[2][0]),
        (matrix[0][1], matrix[1][1], matrix[2][1]),
        (matrix[0][2], matrix[1][2], matrix[2][2]),
    )


def _matmul(left: Matrix3, right: Matrix3) -> Matrix3:
    return tuple(
        tuple(
            sum(left[row][index] * right[index][column] for index in range(3))
            for column in range(3)
        )
        for row in range(3)
    )  # type: ignore[return-value]


def _determinant(matrix: Matrix3) -> float:
    return (
        matrix[0][0]
        * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1]
        * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2]
        * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def _orthogonality_error(matrix: Matrix3) -> float:
    product = _matmul(_transpose(matrix), matrix)
    return math.sqrt(
        sum(
            (
                product[row][column]
                - (1.0 if row == column else 0.0)
            )
            ** 2
            for row in range(3)
            for column in range(3)
        )
    )


def _rotation_quaternion_xyzw(matrix: Matrix3) -> tuple[float, float, float, float]:
    def scale_for(value: float) -> float:
        scale = math.sqrt(max(value, 0.0)) * 2.0
        if scale <= 1e-12:
            raise ValueError("rotation matrix cannot be converted to a finite quaternion")
        return scale

    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        scale = scale_for(trace + 1.0)
        w = 0.25 * scale
        x = (matrix[2][1] - matrix[1][2]) / scale
        y = (matrix[0][2] - matrix[2][0]) / scale
        z = (matrix[1][0] - matrix[0][1]) / scale
    elif matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        scale = scale_for(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2])
        w = (matrix[2][1] - matrix[1][2]) / scale
        x = 0.25 * scale
        y = (matrix[0][1] + matrix[1][0]) / scale
        z = (matrix[0][2] + matrix[2][0]) / scale
    elif matrix[1][1] > matrix[2][2]:
        scale = scale_for(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2])
        w = (matrix[0][2] - matrix[2][0]) / scale
        x = (matrix[0][1] + matrix[1][0]) / scale
        y = 0.25 * scale
        z = (matrix[1][2] + matrix[2][1]) / scale
    else:
        scale = scale_for(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1])
        w = (matrix[1][0] - matrix[0][1]) / scale
        x = (matrix[0][2] + matrix[2][0]) / scale
        y = (matrix[1][2] + matrix[2][1]) / scale
        z = 0.25 * scale

    quaternion_norm = math.sqrt(x * x + y * y + z * z + w * w)
    if quaternion_norm <= 1e-12:
        raise ValueError("rotation matrix cannot be converted to a finite quaternion")
    x /= quaternion_norm
    y /= quaternion_norm
    z /= quaternion_norm
    w /= quaternion_norm
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    return x, y, z, w


def _rotation_vector_world(previous: Matrix3, current: Matrix3) -> Vector3:
    # R_current * R_previous^T expresses the shortest incremental rotation in
    # the VIVE world frame, matching the frame used by the sender velocity.
    relative = _matmul(current, _transpose(previous))
    x, y, z, w = _rotation_quaternion_xyzw(relative)
    vector_norm = math.sqrt(x * x + y * y + z * z)
    if vector_norm <= 1e-12:
        return 2.0 * x, 2.0 * y, 2.0 * z
    angle = 2.0 * math.atan2(vector_norm, max(w, 0.0))
    scale = angle / vector_norm
    return x * scale, y * scale, z * scale


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * percentile
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = rank - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def metric_summary(values: Sequence[float]) -> dict[str, float | int]:
    finite_values = [float(value) for value in values if math.isfinite(value)]
    if not finite_values:
        return {"count": 0}
    ordered = sorted(finite_values)
    count = len(ordered)
    mean = sum(ordered) / count
    variance = sum((value - mean) ** 2 for value in ordered) / count
    rms = math.sqrt(sum(value * value for value in ordered) / count)
    return {
        "count": count,
        "min": ordered[0],
        "max": ordered[-1],
        "mean": mean,
        "std": math.sqrt(variance),
        "rms": rms,
        "p50": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
    }


def _safe_label(label: str) -> str:
    normalized = re.sub(r"[^\w.-]+", "_", label.strip(), flags=re.UNICODE).strip("._-")
    if not normalized:
        raise ValueError("capture label must contain at least one letter or number")
    return normalized


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _write_jsonl(handle: TextIO, payload: object) -> None:
    handle.write(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    )


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


class ViveDiagnosticRecorder:
    """Loss-observing recorder for every VIVE UDP datagram.

    This class intentionally does not use ``VivePoseSource``: the control-path
    source drains stale datagrams to minimize latency, while a diagnostic
    capture must retain every received datagram so 300 Hz timing and loss can
    be measured.
    """

    def __init__(
        self,
        *,
        output_root: Path,
        label: str,
        expected_hz: float = DEFAULT_EXPECTED_HZ,
        flush_every_packets: int = DEFAULT_FLUSH_EVERY_PACKETS,
        bind_host: str = "0.0.0.0",
        bind_port: int = 9999,
        requested_receive_buffer_bytes: int = DEFAULT_RECEIVE_BUFFER_BYTES,
        actual_receive_buffer_bytes: int | None = None,
    ) -> None:
        self.expected_hz = float(expected_hz)
        self.flush_every_packets = max(1, int(flush_every_packets))
        self.started_wall_time = time.time()
        self.started_monotonic_time = time.monotonic()
        self.closed = False

        stamp = datetime.fromtimestamp(self.started_wall_time).strftime("%Y%m%d_%H%M%S_%f")
        self.label = _safe_label(label)
        self.output_dir = Path(output_root) / f"{stamp}_{self.label}"
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.packet_path = self.output_dir / "vive_packets.jsonl"
        self.tracker_path = self.output_dir / "vive_trackers.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        self.metadata_path = self.output_dir / "metadata.json"
        self._packet_handle = self.packet_path.open("w", encoding="utf-8", buffering=256 * 1024)
        self._tracker_handle = self.tracker_path.open("w", encoding="utf-8", buffering=256 * 1024)

        self.datagram_count = 0
        self.decoded_packet_count = 0
        self.decode_error_count = 0
        self.protocol_mismatch_count = 0
        self.packet_validation_error_count = 0
        self.tracker_row_count = 0
        self.sequence_packet_count = 0
        self.sequence_missing_count = 0
        self.sequence_duplicate_count = 0
        self.sequence_out_of_order_count = 0
        self.sequence_reset_count = 0
        self._last_seq_by_source: dict[str, int] = {}
        self._last_receive_monotonic_by_source: dict[str, float] = {}
        self._pose_history: dict[str, PoseHistory] = {}
        self._tracker_statistics: dict[str, TrackerStatistics] = {}
        self._receiver_inter_arrival_sec: list[float] = []
        self._first_datagram_monotonic_time: float | None = None
        self._last_datagram_monotonic_time: float | None = None

        self._metadata: dict[str, Any] = {
            "schema": "dexteleop.vive_diagnostic_capture/v1",
            "status": "recording",
            "purpose": "passive VIVE UDP timing, pose, and velocity diagnosis; no robot control",
            "label": self.label,
            "started_wall_time": self.started_wall_time,
            "started_local_time": datetime.fromtimestamp(self.started_wall_time).astimezone().isoformat(),
            "expected_protocol": EXPECTED_PROTOCOL,
            "expected_hz": self.expected_hz,
            "bind": {"host": bind_host, "port": int(bind_port)},
            "socket": {
                "requested_receive_buffer_bytes": int(requested_receive_buffer_bytes),
                "actual_receive_buffer_bytes": actual_receive_buffer_bytes,
            },
            "host": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": sys.version,
            },
            "files": {
                "packets": self.packet_path.name,
                "trackers": self.tracker_path.name,
                "summary": self.summary_path.name,
            },
        }
        _write_json_atomic(self.metadata_path, self._metadata)

    @staticmethod
    def _source_key(source: tuple[Any, ...] | None) -> str:
        if not source:
            return "unknown"
        host = str(source[0])
        port = str(source[1]) if len(source) > 1 else "unknown"
        return f"{host}:{port}"

    def _sequence_fields(
        self,
        seq_value: object,
        *,
        source_key: str,
        errors: list[str],
    ) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "seq": None,
            "seq_delta": None,
            "sequence_status": "unavailable",
            "missing_packets_before": 0,
        }
        if isinstance(seq_value, bool) or not isinstance(seq_value, int):
            errors.append("seq must be an integer")
            return fields

        seq = int(seq_value)
        fields["seq"] = seq
        self.sequence_packet_count += 1
        previous = self._last_seq_by_source.get(source_key)
        if previous is None:
            fields["sequence_status"] = "first"
            self._last_seq_by_source[source_key] = seq
            return fields

        delta = seq - previous
        fields["seq_delta"] = delta
        if delta == 1:
            fields["sequence_status"] = "consecutive"
            self._last_seq_by_source[source_key] = seq
        elif delta > 1:
            missing = delta - 1
            fields["sequence_status"] = "gap"
            fields["missing_packets_before"] = missing
            self.sequence_missing_count += missing
            self._last_seq_by_source[source_key] = seq
        elif delta == 0:
            fields["sequence_status"] = "duplicate"
            self.sequence_duplicate_count += 1
        elif seq == 0:
            fields["sequence_status"] = "reset"
            self.sequence_reset_count += 1
            self._last_seq_by_source[source_key] = seq
        else:
            fields["sequence_status"] = "out_of_order"
            self.sequence_out_of_order_count += 1
        return fields

    @staticmethod
    def _tracker_id(tracker: dict[str, Any], tracker_index: int) -> str:
        role = str(tracker.get("tracker_role") or "unknown_role")
        name = str(tracker.get("tracker_name") or f"index_{tracker_index}")
        return f"{role}:{name}"

    @staticmethod
    def _sample_clock(
        packet: dict[str, Any],
        *,
        receiver_monotonic_time: float,
        errors: list[str],
    ) -> tuple[str, float]:
        for field_name in (
            "pose_sample_monotonic_time",
            "pose_sample_timestamp",
            "sender_timestamp",
        ):
            raw_value = packet.get(field_name)
            if raw_value is None:
                continue
            parsed = _optional_finite_float(
                raw_value,
                label=field_name,
                errors=errors,
            )
            if parsed is not None:
                return field_name, parsed
        return "receiver_monotonic_time", receiver_monotonic_time

    def _record_tracker(
        self,
        tracker_value: object,
        *,
        tracker_index: int,
        packet: dict[str, Any],
        packet_fields: dict[str, Any],
        receiver_wall_time: float,
        receiver_monotonic_time: float,
        source_key: str,
    ) -> bool:
        self.tracker_row_count += 1
        base_row: dict[str, Any] = {
            "schema": "dexteleop.vive_tracker_sample/v1",
            "receive_index": self.datagram_count - 1,
            "receiver_wall_time": receiver_wall_time,
            "receiver_monotonic_time": receiver_monotonic_time,
            "source": source_key,
            "protocol": packet.get("protocol"),
            "seq": packet_fields.get("seq"),
            "seq_delta": packet_fields.get("seq_delta"),
            "sequence_status": packet_fields.get("sequence_status"),
            "missing_packets_before": packet_fields.get("missing_packets_before", 0),
            "sender_timestamp": packet.get("sender_timestamp"),
            "pose_sample_timestamp": packet.get("pose_sample_timestamp"),
            "pose_sample_monotonic_time": packet.get("pose_sample_monotonic_time"),
            "tracker_index": tracker_index,
            "tracker_payload": tracker_value,
            "validation_errors": [],
        }
        if not isinstance(tracker_value, dict):
            base_row.update(
                {
                    "tracker_id": f"invalid_tracker_index_{tracker_index}",
                    "tracker_name": None,
                    "tracker_role": None,
                    "valid": False,
                    "pose_valid": False,
                    "velocity_valid": False,
                }
            )
            base_row["validation_errors"].append("tracker entry must be a JSON object")
            statistics = self._tracker_statistics.setdefault(
                base_row["tracker_id"], TrackerStatistics()
            )
            statistics.rows += 1
            statistics.invalid_pose_rows += 1
            statistics.invalid_velocity_rows += 1
            _write_jsonl(self._tracker_handle, base_row)
            return True

        errors: list[str] = base_row["validation_errors"]
        tracker_id = self._tracker_id(tracker_value, tracker_index)
        statistics = self._tracker_statistics.setdefault(tracker_id, TrackerStatistics())
        statistics.rows += 1

        valid_flag = tracker_value.get("valid")
        if not isinstance(valid_flag, bool):
            errors.append("valid must be a bool")
        tracker_valid = valid_flag is True
        velocity_valid_flag = tracker_value.get("velocity_valid")
        if not isinstance(velocity_valid_flag, bool):
            errors.append("velocity_valid must be a bool")

        base_row.update(
            {
                "tracker_id": tracker_id,
                "tracker_name": tracker_value.get("tracker_name"),
                "tracker_role": tracker_value.get("tracker_role"),
                "valid": tracker_valid,
                "tracker_error": tracker_value.get("error"),
                "read_elapsed_ms": tracker_value.get("read_elapsed_ms"),
                "velocity_frame_name": tracker_value.get("velocity_frame_name"),
                "velocity_source": tracker_value.get("velocity_source"),
                "velocity_error": tracker_value.get("velocity_error"),
            }
        )

        pose: tuple[tuple[float, float, float, float], ...] | None = None
        try:
            pose = _pose_matrix4x4(tracker_value.get("pose_matrix_4x4"))
        except ValueError as exc:
            errors.append(str(exc))
        pose_valid = tracker_valid and pose is not None
        base_row["pose_valid"] = pose_valid
        if pose_valid:
            statistics.valid_pose_rows += 1
        else:
            statistics.invalid_pose_rows += 1

        linear_velocity: Vector3 | None = None
        angular_velocity: Vector3 | None = None
        if velocity_valid_flag is True:
            try:
                linear_velocity = _vector3(
                    tracker_value.get("linear_velocity_m_s"),
                    label="linear_velocity_m_s",
                )
                angular_velocity = _vector3(
                    tracker_value.get("angular_velocity_rad_s"),
                    label="angular_velocity_rad_s",
                )
            except ValueError as exc:
                errors.append(str(exc))
        velocity_valid = (
            velocity_valid_flag is True
            and linear_velocity is not None
            and angular_velocity is not None
        )
        base_row["velocity_valid"] = velocity_valid
        base_row["linear_velocity_m_s"] = linear_velocity
        base_row["angular_velocity_rad_s"] = angular_velocity
        if velocity_valid:
            statistics.valid_velocity_rows += 1
            linear_speed = _norm(linear_velocity)
            angular_speed = _norm(angular_velocity)
            base_row["linear_speed_m_s"] = linear_speed
            base_row["angular_speed_rad_s"] = angular_speed
            statistics.add_metric("openvr_linear_speed_m_s", linear_speed)
            statistics.add_metric("openvr_angular_speed_rad_s", angular_speed)
            for axis, value in zip("xyz", linear_velocity):
                statistics.add_metric(f"openvr_linear_velocity_{axis}_m_s", value)
            for axis, value in zip("xyz", angular_velocity):
                statistics.add_metric(f"openvr_angular_velocity_{axis}_rad_s", value)
        else:
            statistics.invalid_velocity_rows += 1
            base_row["linear_speed_m_s"] = None
            base_row["angular_speed_rad_s"] = None

        if pose_valid and pose is not None:
            position: Vector3 = (pose[0][3], pose[1][3], pose[2][3])
            rotation: Matrix3 = (
                (pose[0][0], pose[0][1], pose[0][2]),
                (pose[1][0], pose[1][1], pose[1][2]),
                (pose[2][0], pose[2][1], pose[2][2]),
            )
            determinant = _determinant(rotation)
            orthogonality_error = _orthogonality_error(rotation)
            base_row.update(
                {
                    "position_m": position,
                    "rotation_matrix_3x3": rotation,
                    "rotation_quaternion_xyzw": _rotation_quaternion_xyzw(rotation),
                    "rotation_determinant": determinant,
                    "rotation_orthogonality_error": orthogonality_error,
                }
            )
            statistics.add_metric("rotation_determinant", determinant)
            statistics.add_metric("rotation_orthogonality_error", orthogonality_error)

            sample_clock, sample_time = self._sample_clock(
                packet,
                receiver_monotonic_time=receiver_monotonic_time,
                errors=errors,
            )
            base_row["derivative_sample_clock"] = sample_clock
            previous = self._pose_history.get(tracker_id)
            if previous is not None:
                if sample_clock == previous.sample_clock and sample_time > previous.sample_time:
                    delta_time = sample_time - previous.sample_time
                    delta_time_source = sample_clock
                else:
                    delta_time = receiver_monotonic_time - previous.receiver_monotonic_time
                    delta_time_source = "receiver_monotonic_time_fallback"

                if delta_time > 0.0 and math.isfinite(delta_time):
                    position_delta = _sub(position, previous.position_m)
                    rotation_delta_vector = _rotation_vector_world(previous.rotation, rotation)
                    position_step = _norm(position_delta)
                    rotation_step = _norm(rotation_delta_vector)
                    derived_linear_velocity = _scale(position_delta, 1.0 / delta_time)
                    derived_angular_velocity = _scale(rotation_delta_vector, 1.0 / delta_time)
                    pose_repeated = pose == previous.pose_matrix_4x4
                    contiguous = (
                        previous.seq is not None
                        and packet_fields.get("seq") is not None
                        and int(packet_fields["seq"]) - previous.seq == 1
                    )
                    base_row.update(
                        {
                            "pose_pair_contiguous": contiguous,
                            "pose_repeated": pose_repeated,
                            "pose_delta_time_sec": delta_time,
                            "pose_delta_time_source": delta_time_source,
                            "position_delta_m": position_delta,
                            "position_step_m": position_step,
                            "rotation_delta_vector_world_rad": rotation_delta_vector,
                            "rotation_step_rad": rotation_step,
                            "rotation_step_deg": math.degrees(rotation_step),
                            "pose_derived_linear_velocity_m_s": derived_linear_velocity,
                            "pose_derived_linear_speed_m_s": _norm(derived_linear_velocity),
                            "pose_derived_angular_velocity_world_rad_s": derived_angular_velocity,
                            "pose_derived_angular_speed_rad_s": _norm(derived_angular_velocity),
                        }
                    )
                    statistics.add_metric("pose_delta_time_sec", delta_time)
                    statistics.add_metric("position_step_m", position_step)
                    statistics.add_metric("rotation_step_rad", rotation_step)
                    statistics.add_metric(
                        "pose_derived_linear_speed_m_s",
                        base_row["pose_derived_linear_speed_m_s"],
                    )
                    statistics.add_metric(
                        "pose_derived_angular_speed_rad_s",
                        base_row["pose_derived_angular_speed_rad_s"],
                    )
                    if pose_repeated:
                        statistics.repeated_pose_rows += 1
                    if contiguous:
                        statistics.contiguous_pose_pairs += 1
                        statistics.add_metric("contiguous_pose_delta_time_sec", delta_time)
                        statistics.add_metric("contiguous_position_step_m", position_step)
                        statistics.add_metric("contiguous_rotation_step_rad", rotation_step)
                        statistics.add_metric(
                            "contiguous_pose_derived_linear_speed_m_s",
                            base_row["pose_derived_linear_speed_m_s"],
                        )
                        statistics.add_metric(
                            "contiguous_pose_derived_angular_speed_rad_s",
                            base_row["pose_derived_angular_speed_rad_s"],
                        )
                else:
                    errors.append("pose derivative time did not advance")

            self._pose_history[tracker_id] = PoseHistory(
                position_m=position,
                rotation=rotation,
                pose_matrix_4x4=pose,
                sample_clock=sample_clock,
                sample_time=sample_time,
                receiver_monotonic_time=receiver_monotonic_time,
                seq=packet_fields.get("seq"),
            )

        _write_jsonl(self._tracker_handle, base_row)
        return bool(errors)

    def record_datagram(
        self,
        payload: bytes,
        *,
        source: tuple[Any, ...] | None = None,
        receiver_wall_time: float | None = None,
        receiver_monotonic_time: float | None = None,
    ) -> None:
        if self.closed:
            raise RuntimeError("cannot record after the capture has been closed")
        wall_time = time.time() if receiver_wall_time is None else float(receiver_wall_time)
        monotonic_time = (
            time.monotonic()
            if receiver_monotonic_time is None
            else float(receiver_monotonic_time)
        )
        source_key = self._source_key(source)
        receive_index = self.datagram_count
        self.datagram_count += 1
        if self._first_datagram_monotonic_time is None:
            self._first_datagram_monotonic_time = monotonic_time
        self._last_datagram_monotonic_time = monotonic_time

        previous_receive = self._last_receive_monotonic_by_source.get(source_key)
        receiver_inter_arrival_sec: float | None = None
        if previous_receive is not None and monotonic_time > previous_receive:
            receiver_inter_arrival_sec = monotonic_time - previous_receive
            self._receiver_inter_arrival_sec.append(receiver_inter_arrival_sec)
        self._last_receive_monotonic_by_source[source_key] = monotonic_time

        packet_row: dict[str, Any] = {
            "schema": "dexteleop.vive_udp_packet/v1",
            "receive_index": receive_index,
            "receiver_wall_time": wall_time,
            "receiver_monotonic_time": monotonic_time,
            "receiver_inter_arrival_sec": receiver_inter_arrival_sec,
            "source": source_key,
            "datagram_bytes": len(payload),
            "decode_ok": False,
            "protocol_ok": False,
            "validation_errors": [],
        }
        try:
            packet = json.loads(
                payload.decode("utf-8"),
                parse_constant=_reject_nonfinite_json_constant,
            )
            if not isinstance(packet, dict):
                raise ValueError("UDP JSON root must be an object")
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            self.decode_error_count += 1
            packet_row["decode_error"] = f"{type(exc).__name__}: {exc}"
            packet_row["raw_payload_base64"] = base64.b64encode(payload).decode("ascii")
            _write_jsonl(self._packet_handle, packet_row)
            self._flush_if_due()
            return

        self.decoded_packet_count += 1
        packet_row["decode_ok"] = True
        packet_row["packet"] = packet
        errors: list[str] = packet_row["validation_errors"]
        protocol_ok = packet.get("protocol") == EXPECTED_PROTOCOL
        packet_row["protocol_ok"] = protocol_ok
        if not protocol_ok:
            self.protocol_mismatch_count += 1
            errors.append(
                f"protocol must be {EXPECTED_PROTOCOL!r}, got {packet.get('protocol')!r}"
            )

        packet_fields = self._sequence_fields(
            packet.get("seq"),
            source_key=source_key,
            errors=errors,
        )
        packet_row.update(packet_fields)
        for field_name in (
            "sender_timestamp",
            "pose_sample_timestamp",
            "pose_sample_monotonic_time",
        ):
            _optional_finite_float(
                packet.get(field_name),
                label=field_name,
                errors=errors,
            )

        trackers = packet.get("trackers")
        if not isinstance(trackers, list):
            errors.append("trackers must be a list")
        else:
            packet_row["tracker_count"] = len(trackers)
            for tracker_index, tracker in enumerate(trackers):
                try:
                    tracker_has_errors = self._record_tracker(
                        tracker,
                        tracker_index=tracker_index,
                        packet=packet,
                        packet_fields=packet_fields,
                        receiver_wall_time=wall_time,
                        receiver_monotonic_time=monotonic_time,
                        source_key=source_key,
                    )
                except Exception as exc:
                    # Preserve the full packet and continue the capture even if
                    # a malformed rotation reaches a derived-metric helper.
                    tracker_has_errors = True
                    _write_jsonl(
                        self._tracker_handle,
                        {
                            "schema": "dexteleop.vive_tracker_sample/v1",
                            "receive_index": receive_index,
                            "receiver_wall_time": wall_time,
                            "receiver_monotonic_time": monotonic_time,
                            "source": source_key,
                            "seq": packet_fields.get("seq"),
                            "tracker_index": tracker_index,
                            "tracker_payload": tracker,
                            "processing_error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                if tracker_has_errors:
                    errors.append(f"tracker[{tracker_index}] failed validation")

        if errors:
            self.packet_validation_error_count += 1
        _write_jsonl(self._packet_handle, packet_row)
        self._flush_if_due()

    def _flush_if_due(self) -> None:
        if self.datagram_count % self.flush_every_packets == 0:
            self._packet_handle.flush()
            self._tracker_handle.flush()

    def live_status(self) -> dict[str, float | int]:
        traffic_elapsed = 0.0
        if (
            self._first_datagram_monotonic_time is not None
            and self._last_datagram_monotonic_time is not None
        ):
            traffic_elapsed = max(
                0.0,
                self._last_datagram_monotonic_time - self._first_datagram_monotonic_time,
            )
        observed_hz = (
            (self.datagram_count - 1) / traffic_elapsed
            if self.datagram_count > 1 and traffic_elapsed > 0.0
            else 0.0
        )
        return {
            "datagrams": self.datagram_count,
            "decoded": self.decoded_packet_count,
            "decode_errors": self.decode_error_count,
            "observed_hz": observed_hz,
            "missing_seq": self.sequence_missing_count,
            "duplicates": self.sequence_duplicate_count,
            "out_of_order": self.sequence_out_of_order_count,
        }

    def _summary(self, *, reason: str, ended_wall_time: float, ended_monotonic_time: float) -> dict[str, Any]:
        traffic_elapsed = 0.0
        if (
            self._first_datagram_monotonic_time is not None
            and self._last_datagram_monotonic_time is not None
        ):
            traffic_elapsed = max(
                0.0,
                self._last_datagram_monotonic_time - self._first_datagram_monotonic_time,
            )
        observed_hz = (
            (self.datagram_count - 1) / traffic_elapsed
            if self.datagram_count > 1 and traffic_elapsed > 0.0
            else None
        )
        expected_period = 1.0 / self.expected_hz if self.expected_hz > 0.0 else None
        estimated_sent_packets = self.sequence_packet_count + self.sequence_missing_count
        return {
            "schema": "dexteleop.vive_diagnostic_summary/v1",
            "reason": reason,
            "label": self.label,
            "started_wall_time": self.started_wall_time,
            "ended_wall_time": ended_wall_time,
            "capture_process_elapsed_sec": max(
                0.0, ended_monotonic_time - self.started_monotonic_time
            ),
            "traffic_elapsed_sec": traffic_elapsed,
            "expected_hz": self.expected_hz,
            "expected_period_sec": expected_period,
            "observed_receive_hz": observed_hz,
            "counts": {
                "datagrams": self.datagram_count,
                "decoded_packets": self.decoded_packet_count,
                "decode_errors": self.decode_error_count,
                "protocol_mismatches": self.protocol_mismatch_count,
                "packets_with_validation_errors": self.packet_validation_error_count,
                "tracker_rows": self.tracker_row_count,
                "sequence_packets": self.sequence_packet_count,
                "sequence_missing_estimate": self.sequence_missing_count,
                "sequence_duplicates": self.sequence_duplicate_count,
                "sequence_out_of_order": self.sequence_out_of_order_count,
                "sequence_resets": self.sequence_reset_count,
            },
            "estimated_sequence_loss_ratio": (
                self.sequence_missing_count / estimated_sent_packets
                if estimated_sent_packets > 0
                else None
            ),
            "receiver_inter_arrival_sec": metric_summary(self._receiver_inter_arrival_sec),
            "trackers": {
                tracker_id: {
                    "counts": {
                        "rows": statistics.rows,
                        "valid_pose_rows": statistics.valid_pose_rows,
                        "invalid_pose_rows": statistics.invalid_pose_rows,
                        "valid_velocity_rows": statistics.valid_velocity_rows,
                        "invalid_velocity_rows": statistics.invalid_velocity_rows,
                        "repeated_pose_rows": statistics.repeated_pose_rows,
                        "contiguous_pose_pairs": statistics.contiguous_pose_pairs,
                    },
                    "repeated_pose_ratio_of_valid_pairs": (
                        statistics.repeated_pose_rows / max(statistics.valid_pose_rows - 1, 1)
                        if statistics.valid_pose_rows > 1
                        else None
                    ),
                    "metrics": {
                        metric_name: metric_summary(values)
                        for metric_name, values in sorted(statistics.metrics.items())
                    },
                }
                for tracker_id, statistics in sorted(self._tracker_statistics.items())
            },
        }

    def close(self, *, reason: str = "completed") -> Path:
        if self.closed:
            return self.summary_path
        ended_wall_time = time.time()
        ended_monotonic_time = time.monotonic()
        self._packet_handle.flush()
        self._tracker_handle.flush()
        self._packet_handle.close()
        self._tracker_handle.close()
        summary = self._summary(
            reason=reason,
            ended_wall_time=ended_wall_time,
            ended_monotonic_time=ended_monotonic_time,
        )
        _write_json_atomic(self.summary_path, summary)
        self._metadata.update(
            {
                "status": "complete",
                "completion_reason": reason,
                "ended_wall_time": ended_wall_time,
                "ended_local_time": datetime.fromtimestamp(ended_wall_time).astimezone().isoformat(),
                "summary_counts": summary["counts"],
            }
        )
        _write_json_atomic(self.metadata_path, self._metadata)
        self.closed = True
        return self.summary_path

    def __enter__(self) -> "ViveDiagnosticRecorder":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close(reason="error" if exc is not None else "completed")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be finite and >= 0")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Passively record every VIVE v1 UDP packet for static/motion diagnostics. "
            "This test receiver never connects to or commands a robot."
        )
    )
    parser.add_argument(
        "--label",
        required=True,
        help="Short capture label, normally 'static' or 'motion'.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Local UDP bind address.")
    parser.add_argument("--port", type=int, default=9999, help="Local UDP bind port.")
    parser.add_argument(
        "--duration-sec",
        type=_nonnegative_float,
        default=0.0,
        help=(
            "Seconds to record, starting at the first received datagram. "
            "Use 0 to record until Ctrl-C."
        ),
    )
    parser.add_argument(
        "--expected-hz",
        type=_nonnegative_float,
        default=DEFAULT_EXPECTED_HZ,
        help="Expected sender frequency used only for diagnostics (default: 300).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Exact capture parent directory. When provided, this takes precedence "
            "over --artifact-layout."
        ),
    )
    parser.add_argument(
        "--artifact-layout",
        choices=(
            LEGACY_ARTIFACT_LAYOUT_CLI_NAME,
            RUNTIME_ARTIFACT_LAYOUT_CLI_NAME,
        ),
        default=configured_runtime_artifact_layout(),
        help=(
            "Default output layout used only when --output-root is omitted: "
            "'legacy' writes teleop_records/vive_diagnostics; "
            "'runtime-artifacts-v1' writes the canonical "
            "evidence/preflight/vive subtree."
        ),
    )
    parser.add_argument(
        "--receive-buffer-bytes",
        type=_positive_int,
        default=DEFAULT_RECEIVE_BUFFER_BYTES,
        help="Requested UDP SO_RCVBUF size.",
    )
    parser.add_argument(
        "--flush-every-packets",
        type=_positive_int,
        default=DEFAULT_FLUSH_EVERY_PACKETS,
        help="Flush JSONL streams after this many UDP datagrams.",
    )
    parser.add_argument(
        "--socket-timeout-sec",
        type=_nonnegative_float,
        default=DEFAULT_SOCKET_TIMEOUT_SEC,
        help="Receive polling timeout used for clean Ctrl-C/duration handling.",
    )
    parser.add_argument(
        "--print-hz",
        type=_nonnegative_float,
        default=1.0,
        help="Console status rate. Use 0 to disable periodic status.",
    )
    return parser.parse_args(argv)


def resolve_diagnostic_output_root(args: argparse.Namespace) -> Path:
    """Resolve one write root with explicit-path precedence and no side effects."""

    canonical_root = (
        RuntimeArtifactLayout.configured().evidence_dir(kind="preflight") / "vive"
    )
    return resolve_artifact_write_path(
        explicit_path=args.output_root,
        artifact_layout=getattr(args, "artifact_layout", None),
        legacy_path=teleop_records_root() / "vive_diagnostics",
        canonical_path=canonical_root,
    ).path


def run_receiver(args: argparse.Namespace) -> Path:
    output_root = resolve_diagnostic_output_root(args)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_RCVBUF,
            int(args.receive_buffer_bytes),
        )
        sock.settimeout(max(float(args.socket_timeout_sec), 1e-3))
        try:
            sock.bind((str(args.host), int(args.port)))
        except OSError as exc:
            raise RuntimeError(
                f"cannot bind UDP {args.host}:{args.port}: {exc}. "
                "Stop the teleop VIVE receiver first; diagnostic and control receivers "
                "must not share the same UDP stream."
            ) from exc
        actual_receive_buffer = int(
            sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        )
        recorder = ViveDiagnosticRecorder(
            output_root=output_root,
            label=args.label,
            expected_hz=float(args.expected_hz),
            flush_every_packets=int(args.flush_every_packets),
            bind_host=str(args.host),
            bind_port=int(args.port),
            requested_receive_buffer_bytes=int(args.receive_buffer_bytes),
            actual_receive_buffer_bytes=actual_receive_buffer,
        )
        print(
            f"[VIVE capture] listening on {args.host}:{args.port}; "
            f"output={recorder.output_dir} expected={args.expected_hz:g}Hz"
        )
        if float(args.duration_sec) > 0.0:
            print(
                f"[VIVE capture] waiting for first packet; then recording "
                f"{args.duration_sec:g}s"
            )
        else:
            print("[VIVE capture] recording until Ctrl-C")

        first_packet_time: float | None = None
        print_interval = 1.0 / float(args.print_hz) if float(args.print_hz) > 0.0 else 0.0
        next_print_time = time.monotonic() + print_interval if print_interval > 0.0 else math.inf
        completion_reason = "duration_complete"
        try:
            while True:
                now = time.monotonic()
                if (
                    first_packet_time is not None
                    and float(args.duration_sec) > 0.0
                    and now - first_packet_time >= float(args.duration_sec)
                ):
                    break
                try:
                    payload, source = sock.recvfrom(MAX_UDP_DATAGRAM_BYTES)
                except socket.timeout:
                    continue
                received_monotonic_time = time.monotonic()
                received_wall_time = time.time()
                if first_packet_time is None:
                    first_packet_time = received_monotonic_time
                recorder.record_datagram(
                    payload,
                    source=source,
                    receiver_wall_time=received_wall_time,
                    receiver_monotonic_time=received_monotonic_time,
                )
                if received_monotonic_time >= next_print_time:
                    status = recorder.live_status()
                    print(
                        f"[VIVE capture] packets={status['datagrams']} "
                        f"rate={status['observed_hz']:.1f}Hz "
                        f"decoded={status['decoded']} bad={status['decode_errors']} "
                        f"missing_seq={status['missing_seq']} "
                        f"duplicate={status['duplicates']} "
                        f"out_of_order={status['out_of_order']}"
                    )
                    next_print_time = received_monotonic_time + print_interval
        except KeyboardInterrupt:
            completion_reason = "keyboard_interrupt"
            print("\n[VIVE capture] Ctrl-C received; finalizing capture")
        except BaseException:
            recorder.close(reason="error")
            raise

        summary_path = recorder.close(reason=completion_reason)
        status = recorder.live_status()
        print(
            f"[VIVE capture] complete packets={status['datagrams']} "
            f"rate={status['observed_hz']:.1f}Hz missing_seq={status['missing_seq']}"
        )
        print(f"[VIVE capture] summary={summary_path}")
        return summary_path
    finally:
        sock.close()


def main() -> None:
    args = parse_args()
    try:
        run_receiver(args)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"[VIVE capture error] {exc}") from exc


if __name__ == "__main__":
    main()
