"""Strict wire parser for session-scoped Quest controller grip poses."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ...protocol import PROTOCOL_VERSION

FEATURE_NAME = "controller_pose"
FEATURE_PROTOCOL_VERSION = 1
_HANDEDNESS = {"left", "right", "none"}
_TARGET_RAY_MODES = {"gaze", "tracked-pointer", "screen", "transient-pointer"}
_TRACKING_STATES = {"tracked", "emulated", "untracked"}
REFERENCE_SPACE_TYPES = {"local", "local-floor", "bounded-floor", "unbounded"}


def _identifier(value: Any, name: str, *, max_length: int = 160) -> str:
    if not isinstance(value, str) or not value or len(value) > max_length:
        raise ValueError(f"{name} must be a non-empty string up to {max_length} chars")
    if not all(character.isalnum() or character in "._:-" for character in value):
        raise ValueError(f"{name} contains unsupported characters")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def validate_column_major_rigid_matrix(value: Any) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("matrix_reference_from_grip must be a sequence")
    if len(value) != 16:
        raise ValueError("matrix_reference_from_grip must contain 16 values")
    matrix = tuple(_finite_float(item, "matrix_reference_from_grip") for item in value)
    if any(abs(matrix[index]) > 1e-6 for index in (3, 7, 11)) or not math.isclose(
        matrix[15], 1.0, abs_tol=1e-6
    ):
        raise ValueError("matrix_reference_from_grip has an invalid homogeneous row")
    columns = (
        (matrix[0], matrix[1], matrix[2]),
        (matrix[4], matrix[5], matrix[6]),
        (matrix[8], matrix[9], matrix[10]),
    )
    for index, column in enumerate(columns):
        norm = sum(component * component for component in column)
        if not math.isclose(norm, 1.0, abs_tol=1e-5):
            raise ValueError(
                f"matrix_reference_from_grip rotation column {index} is not unit"
            )
    for first, second in ((0, 1), (0, 2), (1, 2)):
        dot = sum(columns[first][i] * columns[second][i] for i in range(3))
        if not math.isclose(dot, 0.0, abs_tol=1e-5):
            raise ValueError("matrix_reference_from_grip rotation is not orthogonal")
    determinant = (
        columns[0][0] * (columns[1][1] * columns[2][2] - columns[1][2] * columns[2][1])
        - columns[1][0]
        * (columns[0][1] * columns[2][2] - columns[0][2] * columns[2][1])
        + columns[2][0]
        * (columns[0][1] * columns[1][2] - columns[0][2] * columns[1][1])
    )
    if not math.isclose(determinant, 1.0, abs_tol=1e-5):
        raise ValueError("matrix_reference_from_grip rotation must have determinant +1")
    return matrix


@dataclass(frozen=True)
class ControllerGripPose:
    source_session_id: str
    handedness: str
    target_ray_mode: str
    profiles: tuple[str, ...]
    emulated_position: bool
    matrix_reference_from_grip: tuple[float, ...]


@dataclass(frozen=True)
class ControllerTrackingDiagnostic:
    """Pose-redacted reason reported for one current XR input source."""

    source_session_id: str
    handedness: str
    target_ray_mode: str
    tracking_state: str
    reason: str
    error_name: str | None = None


@dataclass(frozen=True)
class ControllerPoseSample:
    client_instance_id: str
    xr_session_id: str
    reference_space_type: str
    reference_space_generation: int
    sample_sequence: int
    xr_frame_time_ms: float
    controllers: tuple[ControllerGripPose, ...]
    received_wall_time_s: float
    received_monotonic_ns: int
    # Operational diagnostics explain current selection availability but are
    # deliberately excluded from calibration-evidence equality/serialization.
    controller_diagnostics: tuple[ControllerTrackingDiagnostic, ...] = field(
        default=(),
        compare=False,
    )


def parse_controller_pose_sample(
    payload: Mapping[str, Any],
    *,
    received_wall_time_s: float,
    received_monotonic_ns: int,
    max_controllers: int = 2,
) -> ControllerPoseSample:
    if payload.get("type") != "controller_pose_sample":
        raise ValueError("unexpected controller pose message type")
    if payload.get("feature") != FEATURE_NAME:
        raise ValueError("controller pose feature identity mismatch")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("core protocol version mismatch")
    if payload.get("feature_protocol_version") != FEATURE_PROTOCOL_VERSION:
        raise ValueError("controller pose feature protocol version mismatch")
    forbidden = {"action", "command", "control", "target"}.intersection(payload)
    if forbidden:
        raise ValueError(
            f"controller pose message contains forbidden fields: {sorted(forbidden)}"
        )

    reference_space = payload.get("reference_space")
    if not isinstance(reference_space, Mapping):
        raise TypeError("reference_space must be a mapping")
    reference_space_type = _identifier(
        reference_space.get("type"), "reference_space.type"
    )
    if reference_space_type not in REFERENCE_SPACE_TYPES:
        raise ValueError("unsupported reference-space type")
    generation = _non_negative_int(
        reference_space.get("generation"), "reference_space.generation"
    )

    raw_controllers = payload.get("controllers")
    if not isinstance(raw_controllers, list):
        raise TypeError("controllers must be a list")
    if len(raw_controllers) > max(0, int(max_controllers)):
        raise ValueError("controller count exceeds admission limit")
    controllers: list[ControllerGripPose] = []
    source_ids: set[str] = set()
    for raw in raw_controllers:
        if not isinstance(raw, Mapping):
            raise TypeError("each controller must be a mapping")
        source_id = _identifier(raw.get("source_session_id"), "source_session_id")
        if source_id in source_ids:
            raise ValueError("source_session_id values must be unique within a sample")
        source_ids.add(source_id)
        handedness = _identifier(raw.get("handedness"), "handedness")
        if handedness not in _HANDEDNESS:
            raise ValueError("unsupported handedness")
        target_ray_mode = _identifier(raw.get("target_ray_mode"), "target_ray_mode")
        if target_ray_mode not in _TARGET_RAY_MODES:
            raise ValueError("unsupported target_ray_mode")
        raw_profiles = raw.get("profiles", [])
        if not isinstance(raw_profiles, list) or len(raw_profiles) > 8:
            raise ValueError("profiles must be a list with at most 8 entries")
        profiles = tuple(
            _identifier(profile, "profile", max_length=80) for profile in raw_profiles
        )
        emulated = raw.get("emulated_position")
        if not isinstance(emulated, bool):
            raise TypeError("emulated_position must be boolean")
        controllers.append(
            ControllerGripPose(
                source_session_id=source_id,
                handedness=handedness,
                target_ray_mode=target_ray_mode,
                profiles=profiles,
                emulated_position=emulated,
                matrix_reference_from_grip=validate_column_major_rigid_matrix(
                    raw.get("matrix_reference_from_grip")
                ),
            )
        )

    raw_diagnostics = payload.get("controller_diagnostics")
    diagnostics: list[ControllerTrackingDiagnostic] = []
    if raw_diagnostics is None:
        # Keep protocol-v1 browser compatibility. Older clients only sent
        # pose-bearing controllers, so their state can be inferred exactly but
        # they cannot explain an absent pose.
        diagnostics = [
            ControllerTrackingDiagnostic(
                source_session_id=controller.source_session_id,
                handedness=controller.handedness,
                target_ray_mode=controller.target_ray_mode,
                tracking_state=(
                    "emulated" if controller.emulated_position else "tracked"
                ),
                reason=(
                    "emulated_position"
                    if controller.emulated_position
                    else "tracked"
                ),
            )
            for controller in controllers
        ]
    else:
        if not isinstance(raw_diagnostics, list):
            raise TypeError("controller_diagnostics must be a list")
        if len(raw_diagnostics) > max(0, int(max_controllers)):
            raise ValueError("controller diagnostic count exceeds admission limit")
        diagnostic_source_ids: set[str] = set()
        for raw in raw_diagnostics:
            if not isinstance(raw, Mapping):
                raise TypeError("each controller diagnostic must be a mapping")
            source_id = _identifier(
                raw.get("source_session_id"),
                "controller_diagnostics.source_session_id",
            )
            if source_id in diagnostic_source_ids:
                raise ValueError(
                    "controller diagnostic source IDs must be unique within a sample"
                )
            diagnostic_source_ids.add(source_id)
            handedness = _identifier(
                raw.get("handedness"),
                "controller_diagnostics.handedness",
            )
            if handedness not in _HANDEDNESS:
                raise ValueError("unsupported controller diagnostic handedness")
            target_ray_mode = _identifier(
                raw.get("target_ray_mode"),
                "controller_diagnostics.target_ray_mode",
            )
            if target_ray_mode not in _TARGET_RAY_MODES:
                raise ValueError("unsupported controller diagnostic target_ray_mode")
            tracking_state = _identifier(
                raw.get("tracking_state"),
                "controller_diagnostics.tracking_state",
            )
            if tracking_state not in _TRACKING_STATES:
                raise ValueError("unsupported controller diagnostic tracking_state")
            reason = _identifier(
                raw.get("reason"),
                "controller_diagnostics.reason",
                max_length=80,
            )
            raw_error_name = raw.get("error_name")
            error_name = (
                None
                if raw_error_name is None
                else _identifier(
                    raw_error_name,
                    "controller_diagnostics.error_name",
                    max_length=80,
                )
            )
            diagnostics.append(
                ControllerTrackingDiagnostic(
                    source_session_id=source_id,
                    handedness=handedness,
                    target_ray_mode=target_ray_mode,
                    tracking_state=tracking_state,
                    reason=reason,
                    error_name=error_name,
                )
            )

        controllers_by_source = {
            controller.source_session_id: controller for controller in controllers
        }
        diagnostics_by_source = {
            diagnostic.source_session_id: diagnostic for diagnostic in diagnostics
        }
        if set(controllers_by_source).difference(diagnostics_by_source):
            raise ValueError("every controller pose must have a tracking diagnostic")
        for source_id, diagnostic_item in diagnostics_by_source.items():
            controller = controllers_by_source.get(source_id)
            if diagnostic_item.tracking_state == "untracked":
                if controller is not None:
                    raise ValueError("untracked diagnostic cannot carry a pose")
                continue
            if controller is None:
                raise ValueError("tracked/emulated diagnostic requires a pose")
            if (
                diagnostic_item.handedness != controller.handedness
                or diagnostic_item.target_ray_mode != controller.target_ray_mode
            ):
                raise ValueError("controller pose/diagnostic metadata mismatch")
            expected_state = (
                "emulated" if controller.emulated_position else "tracked"
            )
            if diagnostic_item.tracking_state != expected_state:
                raise ValueError("controller pose/diagnostic tracking state mismatch")
    return ControllerPoseSample(
        client_instance_id=_identifier(
            payload.get("client_instance_id"), "client_instance_id"
        ),
        xr_session_id=_identifier(payload.get("xr_session_id"), "xr_session_id"),
        reference_space_type=reference_space_type,
        reference_space_generation=generation,
        sample_sequence=_non_negative_int(
            payload.get("sample_sequence"), "sample_sequence"
        ),
        xr_frame_time_ms=_finite_float(
            payload.get("xr_frame_time_ms"), "xr_frame_time_ms"
        ),
        controllers=tuple(controllers),
        received_wall_time_s=_finite_float(
            received_wall_time_s, "received_wall_time_s"
        ),
        received_monotonic_ns=_non_negative_int(
            received_monotonic_ns, "received_monotonic_ns"
        ),
        controller_diagnostics=tuple(diagnostics),
    )
