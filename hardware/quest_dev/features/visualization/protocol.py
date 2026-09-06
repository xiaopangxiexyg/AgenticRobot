"""Host-to-Quest messages owned by the visualization feature."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

try:
    from ...protocol import PROTOCOL_VERSION
except ImportError:  # pragma: no cover - direct module compatibility.
    from protocol import PROTOCOL_VERSION


FEATURE_NAME = "visualization"
FEATURE_PROTOCOL_VERSION = 2


def robot_tcp_message(robots: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "robot_tcp",
        "feature": FEATURE_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "feature_protocol_version": FEATURE_PROTOCOL_VERSION,
        "server_timestamp": time.time(),
        "robots": dict(robots),
    }


def feedback_error_message(error: str) -> dict[str, Any]:
    return {
        "type": "robot_tcp",
        "feature": FEATURE_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "feature_protocol_version": FEATURE_PROTOCOL_VERSION,
        "server_timestamp": time.time(),
        "robots": {},
        "source_error": str(error),
    }


def capture_status_message(
    capture: Mapping[str, Any],
    *,
    terminal_message: str | None = None,
) -> dict[str, Any]:
    """Publish pose-redacted terminal-capture state to the Quest page.

    The capture reducer owns the source evidence; this message is only an
    operator-facing projection.  In particular, it carries no pose samples or
    calibration matrix.
    """

    payload: dict[str, Any] = {
        "type": "capture_status",
        "feature": FEATURE_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "feature_protocol_version": FEATURE_PROTOCOL_VERSION,
        "capture": dict(capture),
        "operator_controls": {
            "start_or_finish_window": "r",
            "finalize_and_solve": "c",
        },
    }
    if terminal_message is not None:
        payload["terminal_message"] = str(terminal_message)
    return payload


def calibration_preview_message(
    *,
    robot: str,
    capture_id: str,
    evidence_semantic_content_hash: str,
    diagnostic_fit_semantic_content_hash: str,
    quest_session_id: str,
    reference_space_type: str,
    reference_space_generation: int,
    controller_source_session_id: str,
    matrix_quest_from_robot_base: Sequence[float],
    robot_frame: str,
    tcp_frame: str,
) -> dict[str, Any]:
    """Publish a candidate-only Quest-space robot-frame preview.

    ``accepted`` deliberately remains false.  The preview is bound to the
    exact Quest session/reference-space epoch used by the diagnostic fit and
    is never an accepted frame binding or a control input.
    """

    matrix = [float(value) for value in matrix_quest_from_robot_base]
    if len(matrix) != 16:
        raise ValueError("matrix_quest_from_robot_base must contain 16 values")
    return {
        "type": "calibration_preview",
        "feature": FEATURE_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "feature_protocol_version": FEATURE_PROTOCOL_VERSION,
        "preview": True,
        "accepted": False,
        "alignment_status": "diagnostic_preview",
        "server_timestamp": time.time(),
        "robot": str(robot),
        "capture_id": str(capture_id),
        "evidence_semantic_content_hash": str(evidence_semantic_content_hash),
        "diagnostic_fit_semantic_content_hash": str(
            diagnostic_fit_semantic_content_hash
        ),
        "quest_session_id": str(quest_session_id),
        "reference_space_type": str(reference_space_type),
        "reference_space_generation": int(reference_space_generation),
        "controller_source_session_id": str(controller_source_session_id),
        "matrix_quest_from_robot_base": matrix,
        "robot_frame": str(robot_frame),
        "tcp_frame": str(tcp_frame),
    }
