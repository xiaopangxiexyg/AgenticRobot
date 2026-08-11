"""Deployable geometric guidance for the Cube-Lift reach objective.

The helper deliberately depends only on an external object position, measured
TCP pose, and measured arm-base pose.  It does not read MuJoCo contacts,
collision distances, reward state, or any other simulator-only information.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReachGuidance:
    """One smooth overhead target expressed in world and arm-base frames."""

    target_position_world: np.ndarray
    delta_in_base: np.ndarray
    overhead_blend: float
    target_height_offset_m: float


def build_reach_guidance(
    object_position_world: np.ndarray,
    tcp_position_world: np.ndarray,
    base_rotation_world: np.ndarray,
    *,
    target_height_offset_m: float,
    guidance_height_offset_m: float = 0.0,
    guidance_xy_blend_radius_m: float = 0.0,
) -> ReachGuidance:
    """Build a real-computable target that approaches the object from above.

    With guidance disabled, this is exactly the historical target at
    ``object_z + target_height_offset_m``.  When enabled, the target uses the
    higher guidance offset outside the XY blend radius and smoothly descends
    to the final target as horizontal error approaches zero.  The cubic
    smoothstep has zero slope at both ends, avoiding a discrete task phase or
    a hidden recurrent state.
    """

    object_position = np.asarray(object_position_world, dtype=np.float64).reshape(-1)
    tcp_position = np.asarray(tcp_position_world, dtype=np.float64).reshape(-1)
    base_rotation = np.asarray(base_rotation_world, dtype=np.float64)
    if object_position.shape != (3,) or not np.all(np.isfinite(object_position)):
        raise ValueError("object_position_world must be finite 3-D.")
    if tcp_position.shape != (3,) or not np.all(np.isfinite(tcp_position)):
        raise ValueError("tcp_position_world must be finite 3-D.")
    if base_rotation.shape != (3, 3) or not np.all(np.isfinite(base_rotation)):
        raise ValueError("base_rotation_world must be a finite 3x3 matrix.")

    final_height = float(target_height_offset_m)
    guidance_height = float(guidance_height_offset_m)
    blend_radius = float(guidance_xy_blend_radius_m)
    if not np.isfinite(final_height) or final_height <= 0.0:
        raise ValueError("target_height_offset_m must be positive and finite.")
    guidance_enabled = guidance_height > 0.0 or blend_radius > 0.0
    if guidance_enabled:
        if (
            not np.isfinite(guidance_height)
            or not np.isfinite(blend_radius)
            or guidance_height <= final_height
            or blend_radius <= 0.0
        ):
            raise ValueError(
                "Overhead reach guidance requires a height above the final "
                "target and a positive finite XY blend radius."
            )
        horizontal_distance = float(
            np.linalg.norm(object_position[:2] - tcp_position[:2])
        )
        phase = float(np.clip(horizontal_distance / blend_radius, 0.0, 1.0))
        overhead_blend = phase * phase * (3.0 - 2.0 * phase)
        target_height = final_height + overhead_blend * (
            guidance_height - final_height
        )
    else:
        if guidance_height != 0.0 or blend_radius != 0.0:
            raise ValueError(
                "reach guidance height and XY blend radius must be enabled together."
            )
        overhead_blend = 0.0
        target_height = final_height

    target_position = object_position.copy()
    target_position[2] += target_height
    delta_in_base = base_rotation.T @ (target_position - tcp_position)
    return ReachGuidance(
        target_position_world=target_position,
        delta_in_base=np.ascontiguousarray(delta_in_base, dtype=np.float64),
        overhead_blend=float(overhead_blend),
        target_height_offset_m=float(target_height),
    )


def coarse_to_fine_reach_action_scale(
    reach_target_delta: np.ndarray,
    *,
    minimum_scale: float = 1.0,
    slow_radius_m: float = 0.0,
) -> np.ndarray:
    """Return a smooth deployable action scale from a target-delta batch.

    The scale is one at and beyond ``slow_radius_m``. Inside the radius it
    follows the same zero-slope cubic smoothstep down to ``minimum_scale`` at
    zero distance. With the default values the function is exactly identity.
    """

    delta = np.asarray(reach_target_delta, dtype=np.float32)
    if delta.ndim == 0 or delta.shape[-1] != 3 or not np.all(np.isfinite(delta)):
        raise ValueError("reach_target_delta must end in three finite values.")
    minimum = float(minimum_scale)
    radius = float(slow_radius_m)
    if not np.isfinite(minimum) or not 0.0 < minimum <= 1.0:
        raise ValueError("minimum_scale must lie in (0, 1].")
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("slow_radius_m must be finite and non-negative.")
    if radius == 0.0:
        if minimum != 1.0:
            raise ValueError(
                "A non-identity minimum_scale requires a positive slow radius."
            )
        return np.ones(delta.shape[:-1], dtype=np.float32)
    if minimum >= 1.0:
        raise ValueError(
            "A positive slow radius requires minimum_scale below one."
        )
    distance = np.linalg.norm(delta, axis=-1)
    phase = np.clip(distance / radius, 0.0, 1.0)
    blend = phase * phase * (3.0 - 2.0 * phase)
    return np.asarray(minimum + (1.0 - minimum) * blend, dtype=np.float32)


def contact_close_translation_multiplier(
    reach_target_delta: np.ndarray,
    measured_gripper_close: np.ndarray | float,
    *,
    translation_scale: float = 1.0,
    close_distance_m: float = 0.0,
    keep_measured_close: float = 1.0,
) -> np.ndarray:
    """Return a stateless, actor-observable contact precision multiplier."""

    delta = np.asarray(reach_target_delta, dtype=np.float32)
    if delta.ndim == 0 or delta.shape[-1] != 3 or not np.all(np.isfinite(delta)):
        raise ValueError("reach_target_delta must end in three finite values.")
    measured = np.asarray(measured_gripper_close, dtype=np.float32)
    try:
        measured = np.broadcast_to(measured, delta.shape[:-1])
    except ValueError as exc:
        raise ValueError(
            "measured_gripper_close must broadcast to the target-delta batch."
        ) from exc
    if not np.all(np.isfinite(measured)):
        raise ValueError("measured_gripper_close must be finite.")
    scale = float(translation_scale)
    distance = float(close_distance_m)
    keep_close = float(keep_measured_close)
    values = (scale, distance, keep_close)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Contact-close scaling values must be finite.")
    enabled = scale < 1.0 or distance > 0.0 or keep_close < 1.0
    if not enabled:
        return np.ones(delta.shape[:-1], dtype=np.float32)
    if (
        not 0.0 < scale < 1.0
        or distance <= 0.0
        or not 0.0 < keep_close < 1.0
    ):
        raise ValueError(
            "Enabled contact-close scaling requires scale in (0, 1), a "
            "positive distance, and keep_measured_close in (0, 1)."
        )
    closing = np.logical_or(
        np.linalg.norm(delta, axis=-1) <= distance,
        measured >= keep_close,
    )
    return np.where(closing, scale, 1.0).astype(np.float32)


__all__ = [
    "ReachGuidance",
    "build_reach_guidance",
    "coarse_to_fine_reach_action_scale",
    "contact_close_translation_multiplier",
]
