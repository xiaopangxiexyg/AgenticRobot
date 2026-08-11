"""Backend-neutral observation of a relative pose action.

This module interprets an already-canonical ``relative_delta`` action for
diagnostics.  It owns no input selection, anchor lifecycle, composition state,
authority, safety, or execution.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


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


__all__ = ["project_relative_pose_deflections"]
