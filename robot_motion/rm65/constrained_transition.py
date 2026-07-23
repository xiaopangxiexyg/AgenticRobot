from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robot_motion.arm_init.constrained_transition import (
    ArmConstrainedTransitionConfig,
    ArmConstrainedTransitionPlan,
    ArmRuntimeSignals,
    ArmTransitionModel,
    ArmTransitionPlanStep,
    ArmTransitionRiskAssessment,
    ArmTransitionRiskEvent,
    ArmTransitionRiskLevel,
    evaluate_transition_state,
    plan_constrained_qpos_transition,
)
from robot_motion.arm_init.sampling_transition import (
    ArmSamplingTransitionConfig,
    plan_sampling_constrained_qpos_transition,
)
from robot_motion.pose import as_strict_pose_matrix

from .constants import ARM_DOF, LEFT, RIGHT, SIDES
from .ik import RM65DualArmCollisionIKBundle, RM65DualArmCollisionIKConfig, build_rm65_dual_arm_collision_ik_solver
from .qpos_layout import as_rm65_qpos, validate_qpos_by_side


RM65TransitionRiskLevel = ArmTransitionRiskLevel
RM65TransitionRiskEvent = ArmTransitionRiskEvent
RM65TransitionRiskAssessment = ArmTransitionRiskAssessment
RM65RuntimeSignals = ArmRuntimeSignals
RM65ConstrainedTransitionConfig = ArmConstrainedTransitionConfig
RM65TransitionPlanStep = ArmTransitionPlanStep
RM65ConstrainedTransitionPlan = ArmConstrainedTransitionPlan
RM65DualArmTransitionModel = ArmTransitionModel
RM65SamplingTransitionConfig = ArmSamplingTransitionConfig


@dataclass
class RM65DualArmCollisionIKTransitionModel:
    bundle: RM65DualArmCollisionIKBundle
    reference_full_q: np.ndarray | None = None
    last_collision_details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls,
        config: RM65DualArmCollisionIKConfig | None = None,
        *,
        reference_qpos_by_side: Mapping[str, Any] | None = None,
    ) -> "RM65DualArmCollisionIKTransitionModel":
        bundle = build_rm65_dual_arm_collision_ik_solver(config)
        reference_full_q = None
        if reference_qpos_by_side is not None:
            current = _current_solver_q(bundle.solver)
            qpos = _as_dual_qpos(reference_qpos_by_side)
            reference_full_q = _embed_dual_arm_qpos(bundle.scene_profile, current, qpos[LEFT], qpos[RIGHT])
            bundle.solver.set_configuration_q(reference_full_q)
        return cls(bundle=bundle, reference_full_q=reference_full_q)

    def fk_by_side(self, qpos_by_side: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        qpos = _as_dual_qpos(qpos_by_side)
        full_q = self._full_q(qpos)
        return {
            LEFT: _compute_frame_pose(self.bundle.solver, self.bundle.left_frame_name, full_q),
            RIGHT: _compute_frame_pose(self.bundle.solver, self.bundle.right_frame_name, full_q),
        }

    def min_collision_distance(self, qpos_by_side: Mapping[str, np.ndarray]) -> float | None:
        method = getattr(self.bundle.solver, "compute_min_collision_distance", None)
        if method is None:
            return None
        value = method(self._full_q(_as_dual_qpos(qpos_by_side)))
        if value is None:
            self.last_collision_details = {"min_distance_m": None}
            return None
        self.last_collision_details = self._closest_collision_details(float(value))
        return float(value)

    def collision_distance_details(self) -> Mapping[str, Any]:
        return dict(self.last_collision_details)

    def singularity_margin_by_side(self, qpos_by_side: Mapping[str, np.ndarray]) -> Mapping[str, Any]:
        del qpos_by_side
        return {}

    def _full_q(self, qpos_by_side: Mapping[str, np.ndarray]) -> np.ndarray:
        reference = self.reference_full_q
        if reference is None:
            reference = _current_solver_q(self.bundle.solver)
        full_q = _embed_dual_arm_qpos(
            self.bundle.scene_profile,
            reference,
            qpos_by_side[LEFT],
            qpos_by_side[RIGHT],
        )
        self.reference_full_q = full_q.copy()
        return full_q

    def _closest_collision_details(self, min_distance: float) -> dict[str, Any]:
        try:
            robot = self.bundle.solver.robot
            pairs = robot.collision_model.collisionPairs
            results = robot.collision_data.distanceResults
            geometries = robot.collision_model.geometryObjects

            def distance_at(index: int) -> float:
                raw = getattr(results[index], "min_distance", None)
                if raw is None:
                    raw = getattr(results[index], "minimum_distance", float("inf"))
                return float(raw)

            best_index = min(
                range(min(len(pairs), len(results))),
                key=distance_at,
            )
            pair = pairs[best_index]
            return {
                "min_distance_m": float(min_distance),
                "pair_index": int(best_index),
                "geometry_pair": (
                    str(geometries[pair.first].name),
                    str(geometries[pair.second].name),
                ),
            }
        except Exception:
            return {"min_distance_m": float(min_distance)}


def evaluate_dual_arm_transition_state(
    *,
    qpos_by_side: Mapping[str, Any],
    start_qpos_by_side: Mapping[str, Any],
    target_qpos_by_side: Mapping[str, Any],
    model: RM65DualArmTransitionModel | None,
    config: RM65ConstrainedTransitionConfig | None = None,
    start_pose_by_side: Mapping[str, Any] | None = None,
    target_pose_by_side: Mapping[str, Any] | None = None,
    runtime_signals: RM65RuntimeSignals | None = None,
) -> RM65TransitionRiskAssessment:
    qpos = _as_dual_qpos(qpos_by_side)
    return evaluate_transition_state(
        qpos_by_side=qpos,
        start_qpos_by_side=_as_dual_qpos(start_qpos_by_side),
        target_qpos_by_side=_as_dual_qpos(target_qpos_by_side),
        sides=SIDES,
        model=model,
        config=config,
        start_pose_by_side=start_pose_by_side,
        target_pose_by_side=target_pose_by_side,
        runtime_signals=runtime_signals,
    )


def plan_dual_arm_constrained_qpos_transition(
    *,
    start_qpos_by_side: Mapping[str, Any],
    target_qpos_by_side: Mapping[str, Any],
    model: RM65DualArmTransitionModel | None,
    config: RM65ConstrainedTransitionConfig | None = None,
) -> RM65ConstrainedTransitionPlan:
    return plan_constrained_qpos_transition(
        start_qpos_by_side=_as_dual_qpos(start_qpos_by_side),
        target_qpos_by_side=_as_dual_qpos(target_qpos_by_side),
        sides=SIDES,
        model=model,
        config=config,
    )


def plan_dual_arm_sampling_qpos_transition(
    *,
    start_qpos_by_side: Mapping[str, Any],
    target_qpos_by_side: Mapping[str, Any],
    model: RM65DualArmTransitionModel,
    config: RM65ConstrainedTransitionConfig | None = None,
    search_config: RM65SamplingTransitionConfig | None = None,
) -> RM65ConstrainedTransitionPlan:
    return plan_sampling_constrained_qpos_transition(
        start_qpos_by_side=_as_dual_qpos(start_qpos_by_side),
        target_qpos_by_side=_as_dual_qpos(target_qpos_by_side),
        sides=SIDES,
        model=model,
        transition_config=config,
        search_config=search_config,
    )


def _as_dual_qpos(qpos_by_side: Mapping[str, Any]) -> dict[str, np.ndarray]:
    qpos = validate_qpos_by_side(qpos_by_side)
    missing = [side for side in SIDES if side not in qpos]
    if missing:
        raise ValueError(f"dual-arm transition requires qpos for both sides; missing={missing}.")
    return {side: as_rm65_qpos(qpos[side], label=f"{side}.qpos") for side in SIDES}


def _current_solver_q(solver: Any) -> np.ndarray:
    current = getattr(solver, "current_q", None)
    if current is None:
        current = np.zeros(12, dtype=float)
    return np.asarray(current, dtype=float).reshape(-1).copy()


def _embed_dual_arm_qpos(profile: Any, current_full_q: np.ndarray, left_rad: np.ndarray, right_rad: np.ndarray) -> np.ndarray:
    full_q = np.asarray(current_full_q, dtype=float).reshape(-1).copy()
    layout = str(getattr(profile, "solution_layout", ""))
    if layout == "dual_arm_hand" and full_q.shape[0] >= 24:
        full_q[0:6] = np.asarray(left_rad, dtype=float).reshape(ARM_DOF)
        full_q[18:24] = np.asarray(right_rad, dtype=float).reshape(ARM_DOF)
        return full_q
    if full_q.shape[0] == 14:
        full_q[1:7] = np.asarray(left_rad, dtype=float).reshape(ARM_DOF)
        full_q[8:14] = np.asarray(right_rad, dtype=float).reshape(ARM_DOF)
        return full_q
    if full_q.shape[0] == 12:
        full_q[0:6] = np.asarray(left_rad, dtype=float).reshape(ARM_DOF)
        full_q[6:12] = np.asarray(right_rad, dtype=float).reshape(ARM_DOF)
        return full_q
    raise RuntimeError(f"unsupported dual-arm solver q shape {full_q.shape}")


def _compute_frame_pose(solver: Any, frame_name: str, full_q: np.ndarray) -> np.ndarray:
    method = getattr(solver, "compute_frame_pose")
    try:
        return as_strict_pose_matrix(np.asarray(method(frame_name, full_q), dtype=float))
    except TypeError:
        set_q = getattr(solver, "set_configuration_q", None)
        if set_q is not None:
            set_q(full_q)
        return as_strict_pose_matrix(np.asarray(method(frame_name), dtype=float))


__all__ = [
    "RM65ConstrainedTransitionConfig",
    "RM65ConstrainedTransitionPlan",
    "RM65DualArmCollisionIKTransitionModel",
    "RM65DualArmTransitionModel",
    "RM65RuntimeSignals",
    "RM65SamplingTransitionConfig",
    "RM65TransitionPlanStep",
    "RM65TransitionRiskAssessment",
    "RM65TransitionRiskEvent",
    "RM65TransitionRiskLevel",
    "evaluate_dual_arm_transition_state",
    "plan_dual_arm_constrained_qpos_transition",
    "plan_dual_arm_sampling_qpos_transition",
]
