"""Canonical passive-pose to Quest-space projection.

This module is transport-neutral.  It consumes a typed passive publication
and an externally reviewed ``AcceptedFrameBinding``; it never reads a robot,
advances a journal, or writes a command channel.
"""

from __future__ import annotations

import time
from typing import Any

from .frame_binding import AcceptedFrameBinding, project_quest_from_tcp
from .transforms import (
    matrix_to_column_major,
    matrix_to_pose_payload,
    pose_payload_to_matrix,
)


def _base_record(*, expected_robot: str, key: str) -> dict[str, Any]:
    return {
        "robot": expected_robot,
        "key": key,
        "fresh": False,
        "control_ready": False,
        "alignment_ready": False,
    }


def _coordinate_metadata(
    *,
    expected_robot: str,
    coordinate_binding: Any,
) -> dict[str, Any]:
    fields = (
        "source_instance_id",
        "source_epoch_id",
        "source_binding_hash",
        "component_id",
        "stream_id",
        "reference_frame_id",
        "reference_frame_spec_hash",
        "moving_frame_id",
        "tool_profile_hash",
        "pose_schema_hash",
        "fk_contract_id",
        "fk_contract_hash",
        "source_kind",
    )
    return {
        "robot_id": expected_robot,
        "coordinate_binding_hash": coordinate_binding.semantic_content_hash,
        **{field: getattr(coordinate_binding, field) for field in fields},
    }


def project_unaligned_pose_publication(
    publication: Any,
    *,
    expected_robot: str,
    key: str,
    expected_coordinate_binding: Any,
    max_age_s: float,
    now_monotonic_ns: int | None = None,
    future_tolerance_s: float = 0.1,
) -> dict[str, Any]:
    """Project one journal fact in its robot frame without accepted alignment.

    This is the raw-TCP half of a diagnostic calibration preview.  The caller
    must combine it only with a preview carrying the same coordinate-binding
    hash.  It deliberately does not claim Quest alignment or control readiness.
    """

    from robot_motion.runtime.pose_observation import (
        PoseCoordinateBinding,
        PoseMeasurementPublication,
    )

    base = _base_record(expected_robot=expected_robot, key=key)
    if not isinstance(publication, PoseMeasurementPublication):
        return {**base, "error": "invalid_pose_publication"}
    if not isinstance(expected_coordinate_binding, PoseCoordinateBinding):
        return {**base, "error": "invalid_coordinate_binding"}
    coordinate = publication.coordinate_binding
    if coordinate != expected_coordinate_binding:
        return {**base, "error": "coordinate_binding_mismatch"}

    current_ns = (
        time.monotonic_ns() if now_monotonic_ns is None else int(now_monotonic_ns)
    )
    received_ns = int(publication.received_monotonic_ns)
    age_s = (current_ns - received_ns) / 1_000_000_000.0
    result_base = {
        **base,
        "source": "passive_pose_journal",
        "source_sequence": int(publication.source_sequence),
        "received_monotonic_ns": received_ns,
        "age_s": age_s,
        "pose_binding": _coordinate_metadata(
            expected_robot=expected_robot,
            coordinate_binding=coordinate,
        ),
    }
    if age_s < -abs(float(future_tolerance_s)):
        return {**result_base, "error": "timestamp_in_future"}
    if age_s > float(max_age_s):
        return {**result_base, "error": "stale"}

    values = publication.measurement.value_by_component.get(coordinate.component_id)
    if values is None or len(values) != 7:
        return {**result_base, "error": "invalid_cartesian_pose"}
    try:
        robot_from_tcp = pose_payload_to_matrix(
            {
                "position": list(values[:3]),
                "orientation_quat_xyzw": list(values[3:]),
            }
        )
    except (TypeError, ValueError) as exc:
        return {**result_base, "error": f"invalid_pose_projection: {exc}"}
    normalized_pose = matrix_to_pose_payload(robot_from_tcp)
    return {
        **result_base,
        "fresh": True,
        "control_ready": False,
        "alignment_ready": False,
        "mode": "observation_only",
        "frame": coordinate.reference_frame_id,
        "position": robot_from_tcp[:3, 3].tolist(),
        "orientation_quat_xyzw": normalized_pose["orientation_quat_xyzw"],
        "matrix_robot_from_tcp": matrix_to_column_major(robot_from_tcp),
    }


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
    """Project one canonical passive-journal publication into Quest space."""

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
        time.monotonic_ns() if now_monotonic_ns is None else int(now_monotonic_ns)
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

    values = publication.measurement.value_by_component.get(coordinate.component_id)
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


__all__ = ["project_pose_publication", "project_unaligned_pose_publication"]
