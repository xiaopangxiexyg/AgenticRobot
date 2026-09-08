from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robot_motion.action import RobotAction
from robot_motion.action.executor import RobotActionResolver
from teleop_runtime.real_teleop.rule_based_initial_alignment import (
    RuleBasedInitialAlignmentConfig,
    RuleBasedInitialAlignmentPlanner,
    RuleBasedInitialAlignmentPlanningError,
    rule_based_coupled_pose_command_fault,
    rule_based_initial_alignment_command_fault,
)


class _Envelope:
    apply_to_target = True
    profile = SimpleNamespace(enabled_sides=("left", "right"))

    def __init__(self, *, reject_above: float | None = None) -> None:
        self.reject_above = reject_above

    def __call__(self, targets):
        result = {
            side: np.asarray(value, dtype=float).copy()
            for side, value in targets.items()
        }
        if self.reject_above is not None and result["right"][0, 3] > self.reject_above:
            result["right"][0, 3] = self.reject_above
        return result


class _Retargeter:
    def __init__(self, *, collision_above: float | None = None) -> None:
        self.collision_above = collision_above

    def _distance(self, left_qpos, right_qpos) -> float:
        maximum = max(
            float(np.max(np.abs(np.asarray(left_qpos, dtype=float)))),
            float(np.max(np.abs(np.asarray(right_qpos, dtype=float)))),
        )
        if self.collision_above is not None and maximum >= self.collision_above:
            return 0.0
        return 0.1

    def check_qpos_pair_collision(self, *, left_qpos, right_qpos):
        distance = self._distance(left_qpos, right_qpos)
        return SimpleNamespace(
            safe=distance >= 0.01,
            reason="" if distance >= 0.01 else "collision_distance_below_minimum",
            min_distance_m=distance,
            required_min_distance_m=0.01,
            collision_pair_count=3,
        )

    @staticmethod
    def _targets(left_qpos, right_qpos):
        result = {}
        for side, qpos in (("left", left_qpos), ("right", right_qpos)):
            pose = np.eye(4)
            pose[0, 3] = float(np.asarray(qpos, dtype=float)[0])
            result[side] = pose
        return result

    def solver_world_targets_for_qpos_pair(self, *, left_qpos, right_qpos):
        return self._targets(left_qpos, right_qpos)

    def tool_tcp_world_targets_for_qpos_pair(self, *, left_qpos, right_qpos):
        return self._targets(left_qpos, right_qpos)


def _pair(value: float) -> dict[str, np.ndarray]:
    return {
        "left": np.full(6, value, dtype=float),
        "right": np.full(6, value, dtype=float),
    }


def _planner(*, retargeter=None, envelope=None):
    return RuleBasedInitialAlignmentPlanner(
        dual_arm_retargeter=retargeter or _Retargeter(),
        target_filter=envelope or _Envelope(),
        config=RuleBasedInitialAlignmentConfig(
            max_joint_step_deg=0.5,
            edge_check_resolution_deg=0.25,
            collision_slow_margin_m=0.0,
            collision_hold_margin_m=0.0,
        ),
    )


def test_planner_bounds_every_waypoint_and_densely_certifies_edges() -> None:
    planner = _planner()
    start = _pair(0.0)
    target = _pair(np.deg2rad(1.2))

    plan = planner.plan(
        start_qpos_by_side=start,
        target_qpos_by_side=target,
    )

    assert plan.semantic_content_hash.startswith("sha256:")
    assert plan.metadata["certified_waypoint_count"] == 4
    assert len(plan.waypoints) > plan.metadata["certified_waypoint_count"]
    assert plan.dense_check_count > plan.metadata["certified_waypoint_count"]
    np.testing.assert_allclose(
        np.diff(plan.reference_timestamps_sec),
        1.0 / planner.config.reference_hz,
        rtol=0.0,
        atol=1e-12,
    )
    for side in ("left", "right"):
        np.testing.assert_array_equal(plan.waypoints[0][side], start[side])
    previous = start
    for waypoint in plan.waypoints:
        maximum = max(
            float(np.max(np.abs(waypoint[side] - previous[side])))
            for side in ("left", "right")
        )
        assert float(np.rad2deg(maximum)) <= 0.5000001
        previous = waypoint
    np.testing.assert_allclose(plan.waypoints[-1]["left"], target["left"])
    assert plan.minimum_collision_distance_m == pytest.approx(0.1)


def test_default_full_scale_is_twice_the_legacy_rate_and_remains_scalable() -> None:
    start = _pair(0.0)
    target = _pair(np.deg2rad(2.0))
    full = RuleBasedInitialAlignmentPlanner(
        dual_arm_retargeter=_Retargeter(),
        target_filter=_Envelope(),
        config=RuleBasedInitialAlignmentConfig(
            collision_slow_margin_m=0.0,
            collision_hold_margin_m=0.0,
        ),
    ).plan(start_qpos_by_side=start, target_qpos_by_side=target)
    half = RuleBasedInitialAlignmentPlanner(
        dual_arm_retargeter=_Retargeter(),
        target_filter=_Envelope(),
        config=RuleBasedInitialAlignmentConfig(
            speed_scale=0.5,
            collision_slow_margin_m=0.0,
            collision_hold_margin_m=0.0,
        ),
    ).plan(start_qpos_by_side=start, target_qpos_by_side=target)

    assert RuleBasedInitialAlignmentConfig().max_joint_step_deg == pytest.approx(1.0)
    assert full.metadata["certified_waypoint_count"] == 3
    assert half.metadata["certified_waypoint_count"] == 3
    assert half.reference_timestamps_sec[-1] == pytest.approx(
        2.0 * full.reference_timestamps_sec[-1],
        abs=1.0 / full.metadata["reference_hz"],
    )
    assert full.metadata["startup_home_continuity_waypoint"] is True
    assert full.metadata["transition_waypoint_count"] == 2
    assert full.metadata["max_joint_step_deg"] == pytest.approx(1.0)
    assert half.metadata["speed_scale"] == pytest.approx(0.5)
    assert full.metadata["streaming_max_lag_deg"] == pytest.approx(4.0)
    assert half.metadata["streaming_max_lag_deg"] == pytest.approx(4.0)
    for plan in (full, half):
        certificate = plan.metadata["reference_trajectory"][
            "continuous_hard_limit_certificate"
        ]
        assert certificate["passed"] is True


def test_planner_rejects_collision_and_tcp_envelope_projection() -> None:
    with pytest.raises(RuleBasedInitialAlignmentPlanningError, match="target state"):
        _planner(retargeter=_Retargeter(collision_above=0.01)).plan(
            start_qpos_by_side=_pair(0.0),
            target_qpos_by_side=_pair(0.02),
        )

    with pytest.raises(RuleBasedInitialAlignmentPlanningError, match="target state"):
        _planner(envelope=_Envelope(reject_above=0.01)).plan(
            start_qpos_by_side=_pair(0.0),
            target_qpos_by_side=_pair(0.02),
        )


def test_predispatch_guard_requires_active_plan_hash_and_rechecks_pair() -> None:
    planner = _planner()
    plan = planner.plan(
        start_qpos_by_side=_pair(0.0),
        target_qpos_by_side=_pair(0.01),
    )
    action = RobotAction.bimanual_qpos(
        left_arm=plan.waypoints[-1]["left"],
        right_arm=plan.waypoints[-1]["right"],
    )
    command = RobotActionResolver().resolve(action)
    command.metadata.update(
        {
            "rule_based_directive_kind": "initialize",
            "rule_based_initial_alignment_plan_hash": plan.semantic_content_hash,
        }
    )

    assert (
        rule_based_initial_alignment_command_fault(
            command,
            planner=planner,
            expected_plan_hash=plan.semantic_content_hash,
            now=1.0,
        )
        is None
    )

    command.metadata["rule_based_initial_alignment_plan_hash"] = "sha256:stale"
    fault = rule_based_initial_alignment_command_fault(
        command,
        planner=planner,
        expected_plan_hash=plan.semantic_content_hash,
        now=1.0,
    )
    assert fault is not None
    assert fault.code == "rule_based_initial_alignment_safety_rejected"


def test_pose_predispatch_guard_rejects_coupled_solver_bypass() -> None:
    planner = _planner()
    command = RobotActionResolver().resolve(
        RobotAction.bimanual_qpos(
            left_arm=np.zeros(6),
            right_arm=np.zeros(6),
        )
    )
    command.metadata["rule_based_action_has_pose"] = True

    fault = rule_based_coupled_pose_command_fault(
        command,
        planner=planner,
        now=2.0,
    )

    assert fault is not None
    assert fault.metadata["reason"] == "coupled_pose_solver_bypassed"
