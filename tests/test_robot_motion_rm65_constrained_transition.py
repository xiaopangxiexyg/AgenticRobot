from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from robot_motion.rm65 import (
    RM65ConstrainedTransitionConfig,
    RM65RuntimeSignals,
    RM65TransitionRiskLevel,
    evaluate_dual_arm_transition_state,
    plan_dual_arm_constrained_qpos_transition,
)


def _qpos(left0: float = 0.0, right0: float = 0.0) -> dict[str, np.ndarray]:
    left = np.zeros(6, dtype=float)
    right = np.zeros(6, dtype=float)
    left[0] = float(left0)
    right[0] = float(right0)
    return {"left": left, "right": right}


def test_transition_evaluator_stops_on_joint_hard_limit() -> None:
    model = _FakeDualArmModel()
    assessment = evaluate_dual_arm_transition_state(
        qpos_by_side=_qpos(left0=np.deg2rad(121.0)),
        start_qpos_by_side=_qpos(),
        target_qpos_by_side=_qpos(),
        model=model,
    )

    assert assessment.level == RM65TransitionRiskLevel.STOP
    assert any(event.code == "joint_limit_hard" for event in assessment.events)
    assert assessment.speed_scale == 0.0


def test_transition_evaluator_stops_on_endpoint_excursion() -> None:
    model = _FakeDualArmModel()
    assessment = evaluate_dual_arm_transition_state(
        qpos_by_side=_qpos(left0=0.51),
        start_qpos_by_side=_qpos(),
        target_qpos_by_side=_qpos(),
        model=model,
    )

    assert assessment.level == RM65TransitionRiskLevel.STOP
    assert any(event.code == "endpoint_excursion_hard" for event in assessment.events)
    assert assessment.metadata["endpoint_excursion_m"]["left"]["from_start_m"] > 0.5


def test_transition_evaluator_slows_near_collision_margin() -> None:
    model = _FakeDualArmModel(collision_distance=0.03)
    assessment = evaluate_dual_arm_transition_state(
        qpos_by_side=_qpos(),
        start_qpos_by_side=_qpos(),
        target_qpos_by_side=_qpos(),
        model=model,
    )

    assert assessment.level == RM65TransitionRiskLevel.SLOWDOWN
    assert assessment.speed_scale == 0.35
    assert any(event.code == "collision_near" for event in assessment.events)


def test_transition_evaluator_maps_runtime_signals_to_levels() -> None:
    model = _FakeDualArmModel()
    soft = evaluate_dual_arm_transition_state(
        qpos_by_side=_qpos(),
        start_qpos_by_side=_qpos(),
        target_qpos_by_side=_qpos(),
        model=model,
        runtime_signals=RM65RuntimeSignals(force_norm_n=16.0),
    )
    hard = evaluate_dual_arm_transition_state(
        qpos_by_side=_qpos(),
        start_qpos_by_side=_qpos(),
        target_qpos_by_side=_qpos(),
        model=model,
        runtime_signals=RM65RuntimeSignals(tracking_error_m=0.2),
    )

    assert soft.level == RM65TransitionRiskLevel.HOLD
    assert any(event.code == "force_soft" for event in soft.events)
    assert hard.level == RM65TransitionRiskLevel.STOP
    assert any(event.code == "tracking_error_hard" for event in hard.events)


def test_transition_evaluator_slows_near_singularity_without_hard_rejecting() -> None:
    model = _FakeDualArmModel(singularity_margin={"left": 0.02, "right": 0.1})
    assessment = evaluate_dual_arm_transition_state(
        qpos_by_side=_qpos(),
        start_qpos_by_side=_qpos(),
        target_qpos_by_side=_qpos(),
        model=model,
    )

    assert assessment.level == RM65TransitionRiskLevel.SLOWDOWN
    assert any(event.code == "singularity_near" for event in assessment.events)


def test_constrained_transition_planner_completes_smooth_dual_arm_plan() -> None:
    model = _FakeDualArmModel()
    plan = plan_dual_arm_constrained_qpos_transition(
        start_qpos_by_side=_qpos(),
        target_qpos_by_side=_qpos(left0=0.05, right0=-0.04),
        model=model,
        config=RM65ConstrainedTransitionConfig(max_joint_step_deg=1.0),
    )

    assert plan.completed
    assert len(plan.steps) >= 3
    np.testing.assert_allclose(plan.final_qpos_by_side["left"], _qpos(left0=0.05)["left"])
    np.testing.assert_allclose(plan.final_qpos_by_side["right"], _qpos(right0=-0.04)["right"])
    assert plan.summary()["max_risk_level"] == 0


def test_constrained_transition_planner_keeps_slowdown_metadata() -> None:
    model = _FakeDualArmModel(collision_distance=lambda qpos: 0.03 if qpos["left"][0] >= 0.03 else 0.1)
    plan = plan_dual_arm_constrained_qpos_transition(
        start_qpos_by_side=_qpos(),
        target_qpos_by_side=_qpos(left0=0.05),
        model=model,
        config=RM65ConstrainedTransitionConfig(max_joint_step_deg=5.0),
    )

    assert plan.completed
    assert any(step.assessment.level == RM65TransitionRiskLevel.SLOWDOWN for step in plan.steps)
    assert plan.summary()["max_risk_label"] == "slowdown"


def test_constrained_transition_planner_blocks_unsafe_target() -> None:
    model = _FakeDualArmModel(collision_distance=lambda qpos: 0.005 if qpos["left"][0] > 0.02 else 0.1)
    plan = plan_dual_arm_constrained_qpos_transition(
        start_qpos_by_side=_qpos(),
        target_qpos_by_side=_qpos(left0=0.05),
        model=model,
    )

    assert not plan.completed
    assert plan.status == "blocked_target"
    assert plan.blocked_reason == "collision_hard"


class _FakeDualArmModel:
    def __init__(
        self,
        *,
        collision_distance: float | None | Any = 0.1,
        singularity_margin: Mapping[str, float] | None = None,
    ) -> None:
        self.collision_distance = collision_distance
        self.singularity_margin = dict(singularity_margin or {})

    def fk_by_side(self, qpos_by_side: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for side, qpos in qpos_by_side.items():
            pose = np.eye(4, dtype=float)
            pose[0, 3] = float(np.asarray(qpos, dtype=float).reshape(6)[0])
            pose[1, 3] = -0.2 if side == "left" else 0.2
            out[str(side)] = pose
        return out

    def min_collision_distance(self, qpos_by_side: Mapping[str, np.ndarray]) -> float | None:
        if callable(self.collision_distance):
            return self.collision_distance(qpos_by_side)
        return self.collision_distance

    def singularity_margin_by_side(self, qpos_by_side: Mapping[str, np.ndarray]) -> Mapping[str, float]:
        del qpos_by_side
        return dict(self.singularity_margin)
