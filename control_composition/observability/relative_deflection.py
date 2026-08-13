"""Backend-neutral observation of a relative pose action.

This module interprets an already-canonical ``relative_delta`` action for
diagnostics.  It owns no input selection, anchor lifecycle, composition state,
authority, safety, or execution.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


def _pose_matrix(value: Any, *, name: str) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError(f"{name} must be a finite 4x4 pose matrix")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise ValueError(f"{name} must have a homogeneous final row")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation must be right-handed")
    return pose


def validated_pose_matrix(value: Any, *, name: str = "pose") -> Any:
    """Return an isolated validated pose snapshot for passive consumers."""

    return _pose_matrix(value, name=name).copy()


def projectable_pose_matrix(value: Any, *, name: str = "pose") -> Any:
    """Return a display-only pose with rotation projected to nearest SO(3).

    Translation is preserved exactly.  This helper is intentionally passive:
    it must never be used to admit or rewrite a control action.
    """

    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError(f"{name} must be a finite 4x4 pose matrix")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
        raise ValueError(f"{name} must have a homogeneous final row")
    determinant = float(np.linalg.det(pose[:3, :3]))
    if not math.isfinite(determinant) or determinant <= 1e-8:
        raise ValueError(f"{name} rotation must have positive determinant")
    projected = pose.copy()
    projected[:3, :3] = Rotation.from_matrix(pose[:3, :3]).as_matrix()
    projected[3] = [0.0, 0.0, 0.0, 1.0]
    return projected


def project_pose_matrix_deflection(
    home_pose: Any,
    current_pose: Any,
    *,
    component_id: str,
    translation_reference_frame: str,
    rotation_controlled_frame: str,
) -> dict[str, Any]:
    """Project current-minus-Home using world translation and local rotation.

    This is a passive numerical observation.  It neither selects a source nor
    changes an anchor, composition state, authority, safety, or execution.
    """

    home = _pose_matrix(home_pose, name="home_pose")
    current = _pose_matrix(current_pose, name="current_pose")
    translation = current[:3, 3] - home[:3, 3]
    local_rotation = home[:3, :3].T @ current[:3, :3]
    rotvec = Rotation.from_matrix(local_rotation).as_rotvec()
    distance = float(np.linalg.norm(translation))
    angle = float(np.linalg.norm(rotvec))
    return {
        "component_id": str(component_id),
        "translation_xyz_m": translation.tolist(),
        "translation_distance_m": distance,
        "translation_direction_xyz": (
            [0.0, 0.0, 0.0]
            if distance <= 1e-12
            else (translation / distance).tolist()
        ),
        "translation_reference_frame": str(translation_reference_frame),
        "rotation_rotvec_rad": rotvec.tolist(),
        "rotation_angle_rad": angle,
        "rotation_axis_xyz": (
            [0.0, 0.0, 0.0]
            if angle <= 1e-12
            else (rotvec / angle).tolist()
        ),
        "rotation_controlled_frame": str(rotation_controlled_frame),
        "rotation_application": "home_local_right_multiply",
    }


def project_relative_pose_deflections(
    action: Any,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Return pose deflections plus invalid and non-pose component IDs."""

    action_schema = getattr(action, "action_schema", None)
    values_by_component = dict(
        getattr(action, "values_by_component", {}) or {}
    )
    payload_by_component = dict(
        getattr(action_schema, "payload_schema_by_component", {}) or {}
    )
    if action is None or not values_by_component or not payload_by_component:
        raise ValueError("relative action is missing values or its action schema")

    components: list[dict[str, Any]] = []
    invalid_components: list[str] = []
    omitted_components: list[str] = []
    for component_id, component_value in values_by_component.items():
        payload = payload_by_component.get(component_id)
        if (
            getattr(payload, "representation_family", None)
            != "cartesian_pose"
            or getattr(payload, "value_semantics", None) != "relative_delta"
        ):
            omitted_components.append(str(component_id))
            continue
        try:
            values = np.asarray(
                getattr(component_value, "values", component_value),
                dtype=np.float64,
            ).reshape(7)
            if not np.all(np.isfinite(values)):
                raise ValueError("non-finite pose residual")
            quaternion = values[3:7]
            quaternion_norm = float(np.linalg.norm(quaternion))
            if quaternion_norm <= 1e-12:
                raise ValueError("zero-norm residual quaternion")
            rotvec = Rotation.from_quat(
                quaternion / quaternion_norm
            ).as_rotvec()
            translation = values[:3]
            distance = float(np.linalg.norm(translation))
            angle = float(np.linalg.norm(rotvec))
            components.append(
                {
                    "component_id": str(component_id),
                    "translation_xyz_m": translation.tolist(),
                    "translation_distance_m": distance,
                    "translation_direction_xyz": (
                        [0.0, 0.0, 0.0]
                        if distance <= 1e-12
                        else (translation / distance).tolist()
                    ),
                    "translation_reference_frame": getattr(
                        payload,
                        "reference_frame",
                        None,
                    ),
                    "rotation_rotvec_rad": rotvec.tolist(),
                    "rotation_angle_rad": angle,
                    "rotation_axis_xyz": (
                        [0.0, 0.0, 0.0]
                        if angle <= 1e-12
                        else (rotvec / angle).tolist()
                    ),
                    "rotation_controlled_frame": getattr(
                        payload,
                        "controlled_frame",
                        None,
                    ),
                    "rotation_application": dict(
                        getattr(payload, "semantic_parameters", {}) or {}
                    ).get("rotation_application"),
                }
            )
        except (TypeError, ValueError) as error:
            invalid_components.append(f"{component_id}:{error}")
    return components, invalid_components, omitted_components


__all__ = [
    "project_pose_matrix_deflection",
    "projectable_pose_matrix",
    "project_relative_pose_deflections",
    "validated_pose_matrix",
]
