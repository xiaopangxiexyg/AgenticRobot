"""Optional, read-only Redis adapter for AR5 TCP feedback.

The module has no top-level Redis dependency. ``redis`` is imported only when
``connect_redis`` is called, so the HTTPS/WebXR scaffold remains usable when
robot feedback is disabled.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any, Protocol

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


def feedback_record(
    payload: dict[str, Any] | None,
    *,
    expected_robot: str,
    key: str,
    max_age_s: float,
    now: float | None = None,
    future_tolerance_s: float = 0.1,
) -> dict[str, Any]:
    current_time = time.time() if now is None else float(now)
    base: dict[str, Any] = {
        "robot": expected_robot,
        "key": key,
        "fresh": False,
        "control_ready": False,
    }
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
    return {
        **base,
        "fresh": True,
        "control_ready": payload.get("control_ready") is True,
        "mode": payload.get("mode"),
        "frame": pose.get("frame", "robot_world"),
        "position": robot_from_tcp[:3, 3].tolist(),
        "orientation_quat_xyzw": normalized_pose["orientation_quat_xyzw"],
        "matrix_robot_from_tcp": matrix_to_column_major(robot_from_tcp),
    }


def poll_robot_feedback(
    reader: RedisReader,
    sources: Mapping[str, str],
    *,
    max_age_s: float,
    now: float | None = None,
) -> dict[str, Any]:
    robots: dict[str, dict[str, Any]] = {}
    for robot, key in sources.items():
        key_text = str(key)
        robots[str(robot)] = feedback_record(
            read_json(reader, key_text),
            expected_robot=str(robot),
            key=key_text,
            max_age_s=max_age_s,
            now=now,
        )
    return robot_tcp_message(robots)
