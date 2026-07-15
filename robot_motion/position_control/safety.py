from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R

from robot_motion.safety_constraints import (
    constrain_joint_vector,
    constrain_translation_l2,
)


@dataclass(frozen=True)
class PositionKeyframeSafetyConfig:
    """Minimal, sim/real-neutral safety limits for 15 Hz keyframes.

    This guard intentionally does not contain a low-pass filter.  A value is
    changed only when it is invalid or exceeds an explicit workspace, absolute
    joint, or one-keyframe step bound.  IK, singularity, collision endpoint,
    and per-reference joint checks remain the responsibility of the high-rate
    execution stage.
    """

    workspace_min: tuple[float, float, float] = (-0.60, -0.80, 0.00)
    workspace_max: tuple[float, float, float] = (0.60, 0.80, 0.90)
    workspace_soft_margin_m: float = 0.05
    max_translation_step_m: float = 0.10
    max_rotation_step_deg: float = 15.0
    max_joint_step_rad: float = float(np.deg2rad(6.0))
    joint_lower_rad: tuple[float, ...] | None = None
    joint_upper_rad: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        lower_xyz = np.asarray(self.workspace_min, dtype=float).reshape(3)
        upper_xyz = np.asarray(self.workspace_max, dtype=float).reshape(3)
        if not np.all(np.isfinite(lower_xyz)) or not np.all(np.isfinite(upper_xyz)):
            raise ValueError("workspace bounds must be finite")
        if np.any(lower_xyz > upper_xyz):
            raise ValueError("workspace_min must not exceed workspace_max")
        for name in (
            "workspace_soft_margin_m",
            "max_translation_step_m",
            "max_rotation_step_deg",
            "max_joint_step_rad",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
        if (self.joint_lower_rad is None) != (self.joint_upper_rad is None):
            raise ValueError("joint_lower_rad and joint_upper_rad must be provided together")
        if self.joint_lower_rad is not None:
            lower = np.asarray(self.joint_lower_rad, dtype=float).reshape(-1)
            upper = np.asarray(self.joint_upper_rad, dtype=float).reshape(-1)
            if lower.size == 0 or lower.shape != upper.shape:
                raise ValueError("joint bounds must be non-empty and have matching shapes")
            if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
                raise ValueError("joint bounds must be finite")
            if np.any(lower > upper):
                raise ValueError("joint lower bounds must not exceed upper bounds")


@dataclass(frozen=True)
class PositionKeyframeSafetyResult:
    values_by_side: Mapping[str, np.ndarray]
    modified_by_side: Mapping[str, bool]
    reasons_by_side: Mapping[str, tuple[str, ...]]
    metadata_by_side: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    faults_by_side: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def modified(self) -> bool:
        return any(bool(value) for value in self.modified_by_side.values())

    @property
    def faults(self) -> tuple[str, ...]:
        return tuple(
            fault
            for side in self.values_by_side
            for fault in self.faults_by_side.get(str(side), ())
        )

    @property
    def valid(self) -> bool:
        return not self.faults


def limit_pose_keyframes(
    target_by_side: Mapping[str, Sequence[float] | np.ndarray],
    *,
    reference_by_side: Mapping[str, Sequence[float] | np.ndarray] | None = None,
    config: PositionKeyframeSafetyConfig = PositionKeyframeSafetyConfig(),
) -> PositionKeyframeSafetyResult:
    references = {} if reference_by_side is None else dict(reference_by_side)
    output: dict[str, np.ndarray] = {}
    modified: dict[str, bool] = {}
    reasons: dict[str, tuple[str, ...]] = {}
    metadata: dict[str, Mapping[str, object]] = {}
    faults: dict[str, tuple[str, ...]] = {}
    workspace_min = np.asarray(config.workspace_min, dtype=float)
    workspace_max = np.asarray(config.workspace_max, dtype=float)

    for raw_side, raw_target in target_by_side.items():
        side = str(raw_side)
        target = _normalize_xyzquat(raw_target, name=f"{side}.target_pose")
        original = target.copy()
        side_reasons: list[str] = []

        reference_value = references.get(side)
        reference = (
            None
            if reference_value is None
            else _normalize_xyzquat(reference_value, name=f"{side}.reference_pose")
        )
        raw_translation_step = None
        applied_translation_step = None
        raw_rotation_step_deg = None
        applied_rotation_step_deg = None
        if reference is not None:
            raw_translation_step = float(np.linalg.norm(target[:3] - reference[:3]))

        translation = constrain_translation_l2(
            target[:3],
            reference=(reference[:3] if reference is not None else None),
            workspace_lower=workspace_min,
            workspace_upper=workspace_max,
            workspace_soft_margin_m=float(config.workspace_soft_margin_m),
            max_step_m=float(config.max_translation_step_m),
        )
        if translation.workspace_limited:
            side_reasons.append("workspace_soft_limit")
        if translation.step_limited:
            side_reasons.append("translation_step_limit")
        side_faults: list[str] = []
        if translation.infeasible or translation.value is None:
            side_faults.append("translation_constraint_infeasible")
        else:
            target[:3] = np.asarray(translation.value, dtype=float)
        if reference is not None:
            applied_translation_step = float(np.linalg.norm(target[:3] - reference[:3]))

        if reference is not None:
            reference_rotation = R.from_quat(reference[3:7])
            target_rotation = R.from_quat(target[3:7])
            delta_rotation = target_rotation * reference_rotation.inv()
            raw_rotation_step_deg = float(np.rad2deg(delta_rotation.magnitude()))
            max_rotation_rad = float(np.deg2rad(config.max_rotation_step_deg))
            if max_rotation_rad > 0.0 and delta_rotation.magnitude() > max_rotation_rad:
                rotvec = delta_rotation.as_rotvec()
                rotvec *= max_rotation_rad / max(float(np.linalg.norm(rotvec)), 1e-12)
                target[3:7] = (R.from_rotvec(rotvec) * reference_rotation).as_quat()
                side_reasons.append("rotation_step_limit")
            applied_rotation_step_deg = float(
                np.rad2deg((R.from_quat(target[3:7]) * reference_rotation.inv()).magnitude())
            )

        output[side] = target
        modified[side] = not np.allclose(target, original, atol=1e-12)
        reasons[side] = tuple(side_reasons)
        faults[side] = tuple(side_faults)
        metadata[side] = {
            "raw_translation_step_m": raw_translation_step,
            "applied_translation_step_m": applied_translation_step,
            "target_workspace_distance_m": float(translation.target_workspace_distance_m),
            "reference_workspace_distance_m": (
                None
                if translation.reference_workspace_distance_m is None
                else float(translation.reference_workspace_distance_m)
            ),
            "workspace_soft_margin_m": float(config.workspace_soft_margin_m),
            "raw_rotation_step_deg": raw_rotation_step_deg,
            "applied_rotation_step_deg": applied_rotation_step_deg,
        }

    if not output:
        raise ValueError("target_by_side must contain at least one pose")
    return PositionKeyframeSafetyResult(output, modified, reasons, metadata, faults)


def limit_joint_keyframes(
    target_by_side: Mapping[str, Sequence[float] | np.ndarray],
    *,
    reference_by_side: Mapping[str, Sequence[float] | np.ndarray] | None = None,
    config: PositionKeyframeSafetyConfig = PositionKeyframeSafetyConfig(),
) -> PositionKeyframeSafetyResult:
    references = {} if reference_by_side is None else dict(reference_by_side)
    output: dict[str, np.ndarray] = {}
    modified: dict[str, bool] = {}
    reasons: dict[str, tuple[str, ...]] = {}
    metadata: dict[str, Mapping[str, object]] = {}
    faults: dict[str, tuple[str, ...]] = {}
    absolute_lower = (
        None
        if config.joint_lower_rad is None
        else np.asarray(config.joint_lower_rad, dtype=float).reshape(-1)
    )
    absolute_upper = (
        None
        if config.joint_upper_rad is None
        else np.asarray(config.joint_upper_rad, dtype=float).reshape(-1)
    )

    for raw_side, raw_target in target_by_side.items():
        side = str(raw_side)
        target = _finite_vector(raw_target, name=f"{side}.target_qpos")
        original = target.copy()
        side_reasons: list[str] = []
        reference_value = references.get(side)
        reference = (
            None
            if reference_value is None
            else _finite_vector(reference_value, name=f"{side}.reference_qpos")
        )
        raw_max_step = None
        applied_max_step = None
        if reference is not None:
            if reference.shape != target.shape:
                raise ValueError(f"{side} target/reference qpos shapes do not match")
            raw_delta = target - reference
            raw_max_step = float(np.max(np.abs(raw_delta))) if raw_delta.size else 0.0
        constrained = constrain_joint_vector(
            target,
            reference=reference,
            lower=absolute_lower,
            upper=absolute_upper,
            max_step_rad=float(config.max_joint_step_rad),
        )
        if constrained.absolute_limited:
            side_reasons.append("absolute_joint_limit")
        if constrained.step_limited:
            side_reasons.append("joint_step_limit")
        side_faults: list[str] = []
        if constrained.infeasible or constrained.value is None:
            side_faults.append("joint_constraint_infeasible")
        else:
            target = np.asarray(constrained.value, dtype=float)
        if reference is not None:
            applied_max_step = float(np.max(np.abs(target - reference))) if target.size else 0.0

        output[side] = target
        modified[side] = not np.allclose(target, original, atol=1e-12)
        reasons[side] = tuple(side_reasons)
        faults[side] = tuple(side_faults)
        metadata[side] = {
            "raw_max_joint_step_rad": raw_max_step,
            "applied_max_joint_step_rad": applied_max_step,
            "joint_constraint_infeasible_indices": list(constrained.infeasible_indices),
        }

    if not output:
        raise ValueError("target_by_side must contain at least one qpos target")
    return PositionKeyframeSafetyResult(output, modified, reasons, metadata, faults)


def _normalize_xyzquat(value: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    pose = _finite_vector(value, name=name)
    if pose.shape != (7,):
        raise ValueError(f"{name} must have shape (7,), got {pose.shape}")
    quat_norm = float(np.linalg.norm(pose[3:7]))
    if quat_norm <= 1e-12:
        raise ValueError(f"{name} quaternion must be non-zero")
    pose[3:7] /= quat_norm
    return pose


def _finite_vector(value: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain finite values")
    return vector.copy()


__all__ = [
    "PositionKeyframeSafetyConfig",
    "PositionKeyframeSafetyResult",
    "limit_joint_keyframes",
    "limit_pose_keyframes",
]
