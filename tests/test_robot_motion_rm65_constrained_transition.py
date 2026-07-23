from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from robot_motion.rm65 import (
    RM65ConstrainedTransitionConfig,
    RM65RuntimeSignals,
    RM65SamplingTransitionConfig,
    RM65TransitionRiskLevel,
    evaluate_dual_arm_transition_state,
    plan_dual_arm_constrained_qpos_transition,
    plan_dual_arm_sampling_qpos_transition,
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


def test_endpoint_corridor_allows_safe_motion_farther_than_half_meter() -> None:
    model = _FakeDualArmModel()
    plan = plan_dual_arm_constrained_qpos_transition(
        start_qpos_by_side=_qpos(),
        target_qpos_by_side=_qpos(left0=0.6),
        model=model,
        config=RM65ConstrainedTransitionConfig(max_joint_step_deg=5.0),
    )

    assert plan.completed
    assert plan.initial_assessment.metadata["endpoint_excursion_m"]["left"][
        "from_start_target_segment_m"
    ] == 0.0


def test_required_collision_check_fails_closed_when_distance_is_unavailable() -> None:
    assessment = evaluate_dual_arm_transition_state(
        qpos_by_side=_qpos(),
        start_qpos_by_side=_qpos(),
        target_qpos_by_side=_qpos(left0=0.01),
        model=_FakeDualArmModel(collision_distance=None),
        config=RM65ConstrainedTransitionConfig(require_collision_check=True),
    )

    assert assessment.stop
    assert any(event.code == "collision_check_unavailable" for event in assessment.events)


def test_per_joint_limits_reject_joint_specific_out_of_range_qpos() -> None:
    lower = tuple(np.deg2rad([-178.0, -90.0, -135.0, -140.0, -128.0, -360.0]))
    upper = tuple(-value for value in lower)
    assessment = evaluate_dual_arm_transition_state(
        qpos_by_side={"left": np.deg2rad([0, 100, 0, 0, 0, 0]), "right": np.zeros(6)},
        start_qpos_by_side=_qpos(),
        target_qpos_by_side=_qpos(),
        model=_FakeDualArmModel(),
        config=RM65ConstrainedTransitionConfig(joint_lower_rad=lower, joint_upper_rad=upper),
    )

    assert assessment.stop
    assert any(
        event.code == "joint_limit_hard" and event.metadata.get("joint_index") == 2
        for event in assessment.events
    )


def test_joint_hold_margin_escape_is_opt_in_and_only_moves_away_from_limit() -> None:
    lower = tuple([-1.0] * 6)
    upper = tuple([1.0] * 6)
    start = _qpos(left0=0.99)
    target = _qpos(left0=0.0)
    base_config = dict(
        joint_lower_rad=lower,
        joint_upper_rad=upper,
        joint_slow_margin_deg=10.0,
        joint_hold_margin_deg=5.0,
        max_joint_step_deg=1.0,
        max_steps=1000,
    )

    blocked = plan_dual_arm_constrained_qpos_transition(
        start_qpos_by_side=start,
        target_qpos_by_side=target,
        model=_FakeDualArmModel(),
        config=RM65ConstrainedTransitionConfig(**base_config),
    )
    escaped = plan_dual_arm_constrained_qpos_transition(
        start_qpos_by_side=start,
        target_qpos_by_side=target,
        model=_FakeDualArmModel(),
        config=RM65ConstrainedTransitionConfig(
            **base_config,
            allow_joint_limit_escape=True,
        ),
    )

    assert blocked.status == "hold_current"
    assert blocked.blocked_reason == "joint_limit_hold_margin"
    assert not blocked.metadata["allow_joint_limit_escape"]
    assert escaped.completed, escaped.summary()
    assert escaped.metadata["allow_joint_limit_escape"]
    assert escaped.metadata["joint_limit_escape_used"]
    assert escaped.steps[0].metadata["joint_limit_escape"]
    assert escaped.steps[0].qpos_by_side["left"][0] < start["left"][0]
    np.testing.assert_allclose(escaped.final_qpos_by_side["left"], target["left"])


def test_joint_hold_margin_escape_rejects_target_that_remains_in_margin() -> None:
    lower = tuple([-1.0] * 6)
    upper = tuple([1.0] * 6)
    start = _qpos(left0=0.99)
    target = _qpos(left0=0.98)

    plan = plan_dual_arm_constrained_qpos_transition(
        start_qpos_by_side=start,
        target_qpos_by_side=target,
        model=_FakeDualArmModel(),
        config=RM65ConstrainedTransitionConfig(
            joint_lower_rad=lower,
            joint_upper_rad=upper,
            joint_slow_margin_deg=10.0,
            joint_hold_margin_deg=5.0,
            max_joint_step_deg=1.0,
            allow_joint_limit_escape=True,
        ),
    )

    assert plan.status == "hold_current"
    assert plan.blocked_reason == "joint_limit_hold_margin"
    current_risk = plan.metadata["blocked_current_assessment"]
    assert current_risk["events"][0]["metadata"]["joint_index"] == 1


def test_joint_hold_margin_escape_does_not_bypass_collision_hold() -> None:
    lower = tuple([-1.0] * 6)
    upper = tuple([1.0] * 6)

    plan = plan_dual_arm_constrained_qpos_transition(
        start_qpos_by_side=_qpos(left0=0.99),
        target_qpos_by_side=_qpos(left0=0.0),
        model=_FakeDualArmModel(collision_distance=0.015),
        config=RM65ConstrainedTransitionConfig(
            joint_lower_rad=lower,
            joint_upper_rad=upper,
            joint_slow_margin_deg=10.0,
            joint_hold_margin_deg=5.0,
            max_joint_step_deg=1.0,
            allow_joint_limit_escape=True,
        ),
    )

    assert plan.status == "hold_current"
    events = plan.metadata["blocked_current_assessment"]["events"]
    assert any(event["code"] == "collision_soft" for event in events)
    assert not plan.metadata["joint_limit_escape_used"]


def test_joint_hold_margin_escape_allows_static_support_joint() -> None:
    lower = tuple([-1.0] * 6)
    upper = tuple([1.0] * 6)
    start = _qpos(right0=0.99)
    target = _qpos(left0=0.2, right0=0.99)

    plan = plan_dual_arm_constrained_qpos_transition(
        start_qpos_by_side=start,
        target_qpos_by_side=target,
        model=_FakeDualArmModel(),
        config=RM65ConstrainedTransitionConfig(
            joint_lower_rad=lower,
            joint_upper_rad=upper,
            joint_slow_margin_deg=10.0,
            joint_hold_margin_deg=5.0,
            max_joint_step_deg=1.0,
            allow_joint_limit_escape=True,
        ),
    )

    assert plan.completed, plan.summary()
    assert plan.metadata["joint_limit_escape_used"]
    assert all(
        step.qpos_by_side["right"][0] == start["right"][0]
        for step in plan.steps
    )
    np.testing.assert_allclose(
        plan.final_qpos_by_side["left"],
        target["left"],
        atol=1e-4,
    )


def test_sampling_fallback_finds_non_monotonic_detour() -> None:
    start = {"left": np.zeros(6), "right": np.zeros(6)}
    target_left = np.zeros(6)
    target_left[:2] = [1.0, 1.0]
    target = {"left": target_left, "right": np.zeros(6)}

    plan = plan_dual_arm_sampling_qpos_transition(
        start_qpos_by_side=start,
        target_qpos_by_side=target,
        model=_DetourDualArmModel(),
        config=RM65ConstrainedTransitionConfig(
            max_joint_step_deg=3.0,
            max_steps=4000,
            max_joint_abs_deg=178.0,
        ),
        search_config=RM65SamplingTransitionConfig(
            seed=7,
            max_iterations=1000,
            max_nodes=1800,
            extension_step_deg=15.0,
            edge_resolution_deg=1.0,
            sample_padding_deg=45.0,
        ),
    )

    assert plan.completed, plan.summary()
    np.testing.assert_allclose(plan.final_qpos_by_side["left"], target_left, atol=1e-4)
    assert plan.metadata["planner"] == "bounded_birrt_connect"


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


class _DetourDualArmModel(_FakeDualArmModel):
    def fk_by_side(self, qpos_by_side: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        del qpos_by_side
        return {"left": np.eye(4), "right": np.eye(4)}

    def min_collision_distance(self, qpos_by_side: Mapping[str, np.ndarray]) -> float:
        x, y = np.asarray(qpos_by_side["left"], dtype=float)[:2]
        blocked = (
            (0.2 < x < 0.8 and abs(x - y) < 0.04)
            or (0.45 < x < 0.55 and abs(y) < 0.04)
            or (abs(x - 1.0) < 0.04 and 0.45 < y < 0.55)
            or (abs(x) < 0.04 and 0.45 < y < 0.55)
            or (0.45 < x < 0.55 and abs(y - 1.0) < 0.04)
        )
        return 0.0 if blocked else 1.0
