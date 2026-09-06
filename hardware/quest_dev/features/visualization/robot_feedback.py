"""Optional, read-only Redis adapter for robot TCP feedback.

The legacy Redis shape remains diagnostic-only. Quest-space projection requires
an explicit AcceptedFrameBinding and a nested pose_binding metadata projection
from the passive journal. Missing or mismatched identity never produces a
Quest-space matrix.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any, Protocol

from .frame_binding import (
    AcceptedFrameBinding,
    project_quest_from_tcp,
)
from .protocol import robot_tcp_message
from .transforms import (
    matrix_to_column_major,
    matrix_to_pose_payload,
    pose_payload_to_matrix,
)


class RedisReader(Protocol):
    def get(self, key: str) -> Any: ...


def connect_redis(
    host: str,
    port: int,
    db: int = 0,
    socket_timeout_s: float = 0.2,
) -> RedisReader:
    try:
        import redis
    except ImportError as exc:
        raise RuntimeError(
            "robot_feedback.enabled requires the optional 'redis' package"
        ) from exc
    return redis.Redis(
        host=host,
        port=port,
        db=db,
        decode_responses=True,
        socket_connect_timeout=float(socket_timeout_s),
        socket_timeout=float(socket_timeout_s),
    )


def read_json(reader: RedisReader, key: str) -> dict[str, Any] | None:
    raw = reader.get(key)
    if raw is None or raw == b"" or raw == "":
        return None
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None
    try:
        payload = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _base_record(*, expected_robot: str, key: str) -> dict[str, Any]:
    return {
        "robot": expected_robot,
        "key": key,
        "fresh": False,
        "control_ready": False,
        "alignment_ready": False,
    }


def _project_bound_pose(
    record: dict[str, Any],
    *,
    payload: Mapping[str, Any],
    robot_from_tcp: Any,
    frame_binding: AcceptedFrameBinding,
) -> dict[str, Any]:
    raw_pose_binding = payload.get("pose_binding")
    if not isinstance(raw_pose_binding, Mapping):
        return {
            **record,
            "alignment_error": "missing_pose_binding",
        }
    metadata_error = frame_binding.robot_metadata_error(
        robot=str(record["robot"]),
        metadata=raw_pose_binding,
    )
    if metadata_error is not None:
        return {
            **record,
            "alignment_error": metadata_error,
        }
    try:
        quest_from_tcp = project_quest_from_tcp(robot_from_tcp, frame_binding)
    except (TypeError, ValueError) as exc:
        return {
            **record,
            "alignment_error": f"invalid_quest_projection: {exc}",
        }
    return {
        **record,
        "alignment_ready": True,
        "matrix_quest_from_tcp": matrix_to_column_major(quest_from_tcp),
        "frame_binding": frame_binding.telemetry_identity(),
    }


def feedback_record(
    payload: dict[str, Any] | None,
    *,
    expected_robot: str,
    key: str,
    max_age_s: float,
    now: float | None = None,
    future_tolerance_s: float = 0.1,
    frame_binding: AcceptedFrameBinding | None = None,
) -> dict[str, Any]:
    current_time = time.time() if now is None else float(now)
    base = _base_record(expected_robot=expected_robot, key=key)
    if not payload:
        return {**base, "error": "missing"}
    if payload.get("robot") != expected_robot:
        return {**base, "error": "robot_mismatch"}

    timestamp = payload.get("timestamp")
    if not isinstance(timestamp, (int, float)):
        return {**base, "error": "missing_timestamp"}
    age_s = current_time - float(timestamp)
    base["age_s"] = age_s
    if age_s < -abs(float(future_tolerance_s)):
        return {**base, "error": "timestamp_in_future"}
    if age_s > float(max_age_s):
        return {**base, "error": "stale"}

    pose = payload.get("tcp_pose")
    if not isinstance(pose, dict):
        return {**base, "error": "missing_tcp_pose"}
    try:
        robot_from_tcp = pose_payload_to_matrix(pose)
    except ValueError as exc:
        return {**base, "error": f"invalid_tcp_pose: {exc}"}

    normalized_pose = matrix_to_pose_payload(robot_from_tcp)
    record = {
        **base,
        "fresh": True,
        # This feature is an observation-only boundary.  A source payload's
        # control readiness is not an authorization or capability grant and
        # must never be exposed as one to Quest consumers.
        "control_ready": False,
        "mode": payload.get("mode"),
        "frame": pose.get("frame", "robot_world"),
        "position": robot_from_tcp[:3, 3].tolist(),
        "orientation_quat_xyzw": normalized_pose["orientation_quat_xyzw"],
        "matrix_robot_from_tcp": matrix_to_column_major(robot_from_tcp),
    }
    if frame_binding is None:
        return record
    return _project_bound_pose(
        record,
        payload=payload,
        robot_from_tcp=robot_from_tcp,
        frame_binding=frame_binding,
    )


def project_pose_publication(
    publication: Any,
    *,
    expected_robot: str,
    key: str,
    max_age_s: float,
    now_monotonic_ns: int | None = None,
    frame_binding: AcceptedFrameBinding,
    future_tolerance_s: float = 0.1,
) -> dict[str, Any]:
    """Project one canonical passive-journal publication into Quest space.

    The function consumes an already-created typed publication. It never reads
    the arm, advances a journal, or treats a legacy Redis value as canonical.
    """

    from robot_motion.runtime.pose_observation import PoseMeasurementPublication

    base = _base_record(expected_robot=expected_robot, key=key)
    if not isinstance(publication, PoseMeasurementPublication):
        return {**base, "error": "invalid_pose_publication"}
    if not isinstance(frame_binding, AcceptedFrameBinding):
        return {**base, "error": "invalid_frame_binding"}

    coordinate = publication.coordinate_binding
    binding_error = frame_binding.coordinate_binding_error(
        robot=expected_robot,
        coordinate_binding=coordinate,
    )
    if binding_error is not None:
        return {**base, "error": binding_error}

    current_ns = (
        time.monotonic_ns()
        if now_monotonic_ns is None
        else int(now_monotonic_ns)
    )
    received_ns = int(publication.received_monotonic_ns)
    age_s = (current_ns - received_ns) / 1_000_000_000.0
    result_base = {
        **base,
        "source": "passive_pose_journal",
        "source_sequence": int(publication.source_sequence),
        "received_monotonic_ns": received_ns,
        "age_s": age_s,
    }
    if age_s < -abs(float(future_tolerance_s)):
        return {**result_base, "error": "timestamp_in_future"}
    if age_s > float(max_age_s):
        return {**result_base, "error": "stale"}

    values = publication.measurement.value_by_component.get(
        coordinate.component_id
    )
    if values is None or len(values) != 7:
        return {**result_base, "error": "invalid_cartesian_pose"}
    try:
        robot_from_tcp = pose_payload_to_matrix(
            {
                "position": list(values[:3]),
                "orientation_quat_xyzw": list(values[3:]),
            }
        )
        quest_from_tcp = project_quest_from_tcp(robot_from_tcp, frame_binding)
    except (TypeError, ValueError) as exc:
        return {**result_base, "error": f"invalid_pose_projection: {exc}"}

    normalized_pose = matrix_to_pose_payload(robot_from_tcp)
    return {
        **result_base,
        "fresh": True,
        "control_ready": False,
        "alignment_ready": True,
        "frame": coordinate.reference_frame_id,
        "position": robot_from_tcp[:3, 3].tolist(),
        "orientation_quat_xyzw": normalized_pose["orientation_quat_xyzw"],
        "matrix_robot_from_tcp": matrix_to_column_major(robot_from_tcp),
        "matrix_quest_from_tcp": matrix_to_column_major(quest_from_tcp),
        "frame_binding": frame_binding.telemetry_identity(),
    }


def poll_robot_feedback(
    reader: RedisReader,
    sources: Mapping[str, str],
    *,
    max_age_s: float,
    now: float | None = None,
    frame_bindings: Mapping[str, AcceptedFrameBinding] | None = None,
) -> dict[str, Any]:
    robots: dict[str, dict[str, Any]] = {}
    for robot, key in sources.items():
        key_text = str(key)
        frame_binding = (
            None
            if frame_bindings is None
            else frame_bindings.get(str(robot))
        )
        robots[str(robot)] = feedback_record(
            read_json(reader, key_text),
            expected_robot=str(robot),
            key=key_text,
            max_age_s=max_age_s,
            now=now,
            frame_binding=frame_binding,
        )
    return robot_tcp_message(robots)
