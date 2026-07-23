from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pytest

from robot_motion.rm65 import RM65SamplingTransitionConfig
from robot_motion.rm65.constants import RM65_ABSOLUTE_Q_MAX_RAD

from robot_motion.arm_init.constrained_transition import (
    ArmConstrainedTransitionPlan,
    ArmTransitionRiskAssessment,
    ArmTransitionRiskEvent,
    ArmTransitionRiskLevel,
)
from robot_motion.rm65.arm_init.schema import QposTargetSpec, load_qpos_target, save_qpos_target
from robot_motion.rm65.arm_init.qpos_sequence import (
    QposSequenceEntry,
    QposTestSequence,
    save_qpos_test_sequence,
)
from teleop_runtime.replay.arm_init.qpos_execute import (
    CONFIRM_TEXT,
    RM65LiveQposInitConfig,
    _normalize_cli_argv,
    _require_plan_completed,
    build_rm65_qpos_init_plan,
    execute_rm65_live_qpos_init,
    QPOS_INIT_PLANNER_CAPABILITY,
    read_current_qpos_by_side,
    resolve_rm65_arm_ips,
)
from teleop_runtime.replay.arm_init.qpos_sequence_execute import (
    CONFIRM_TEXT as SEQUENCE_CONFIRM_TEXT,
    RM65QposSequenceExecuteConfig,
    _normalize_cli_argv as _normalize_sequence_cli_argv,
    execute_rm65_qpos_test_sequence,
)


def test_read_current_qpos_by_side_converts_degrees_to_radians() -> None:
    robot = _FakeRM65Robot([90.0, 0.0, -90.0, 0.0, 0.0, 180.0])

    qpos = read_current_qpos_by_side({"right": robot}, active_sides=("right",))

    np.testing.assert_allclose(qpos["right"], np.deg2rad([90.0, 0.0, -90.0, 0.0, 0.0, 180.0]))


def test_live_qpos_init_requires_confirmation_for_real_execution(tmp_path: Path) -> None:
    target_path = tmp_path / "target.json"
    save_qpos_target(QposTargetSpec.from_qpos_by_side({"right": np.zeros(6)}), target_path)

    with pytest.raises(ValueError, match=CONFIRM_TEXT):
        RM65LiveQposInitConfig(
            qpos_target_path=target_path,
            active_sides=("right",),
            execute=True,
            confirm="",
        )


def test_qpos_target_loader_accepts_measured_initial_alignment_schema(tmp_path: Path) -> None:
    path = tmp_path / "measured.json"
    path.write_text(
        '{"format":"teleop_initial_alignment","format_version":1,"sides":'
        '{"left":{"arm_qpos":[1,2,3,4,5,6]},"right":{"arm_qpos":[-1,-2,-3,-4,-5,-6]}}}',
        encoding="utf-8",
    )

    target = load_qpos_target(path)

    assert target.name == "measured_init_pose"
    np.testing.assert_allclose(target.qpos_by_side()["left"], [1, 2, 3, 4, 5, 6])
    assert target.metadata["source_format"] == "teleop_initial_alignment"


def test_live_qpos_init_executes_plan_with_fake_robot(tmp_path: Path) -> None:
    target_path = tmp_path / "target.json"
    save_qpos_target(
        QposTargetSpec.from_qpos_by_side({"right": np.deg2rad([3.0, 0.0, 0.0, 0.0, 0.0, 0.0])}),
        target_path,
    )
    robot = _FakeRM65Robot([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    report = execute_rm65_live_qpos_init(
        RM65LiveQposInitConfig(
            qpos_target_path=target_path,
            active_sides=("right",),
            execute=True,
            confirm=CONFIRM_TEXT,
            qpos_dt_sec=0.25,
            tracking_check_every=1,
            max_tracking_error_deg=0.001,
        ),
        robots_by_side={"right": robot},
    )

    assert report["execution"]["executed"]
    assert report["execution"]["sent_step_count"] == report["plan"]["summary"]["step_count"]
    assert robot.commands
    np.testing.assert_allclose(robot.commands[-1], [3.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_single_arm_qpos_init_uses_slow_plan_for_moderate_delta() -> None:
    plan = build_rm65_qpos_init_plan(
        start_qpos_by_side={"right": np.zeros(6)},
        target_qpos_by_side={"right": np.deg2rad([119.0, 0.0, 0.0, 0.0, 0.0, 0.0])},
        active_sides=("right",),
        qpos_dt_sec=0.25,
        joint_velocity_profile="rm65_init_slow",
        max_steps_per_side=10000,
    )

    assert plan.metadata["planner"] == "plan_slow_qpos_transition"
    assert plan.completed


def test_single_arm_qpos_init_accepts_custom_max_angvel() -> None:
    plan = build_rm65_qpos_init_plan(
        start_qpos_by_side={"right": np.zeros(6)},
        target_qpos_by_side={"right": np.deg2rad([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])},
        active_sides=("right",),
        qpos_dt_sec=0.5,
        joint_velocity_profile="rm65_init_slow",
        qpos_init_max_angvel_rad_s=0.1,
        max_steps_per_side=10000,
    )

    assert plan.metadata["joint_velocity_profile"] == "custom_qpos_init_max_angvel"
    assert plan.metadata["qpos_init_max_angvel_rad_s"] == [0.1] * 6
    assert len(plan.steps) > 1
    np.testing.assert_allclose(
        plan.steps[0].qpos_by_side["right"],
        np.asarray([0.05, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )


def test_single_arm_qpos_init_rejects_delta_over_120_deg() -> None:
    with pytest.raises(ValueError, match="delta 121.000deg exceeds 120.000deg"):
        build_rm65_qpos_init_plan(
            start_qpos_by_side={"right": np.zeros(6)},
            target_qpos_by_side={"right": np.deg2rad([121.0, 0.0, 0.0, 0.0, 0.0, 0.0])},
            active_sides=("right",),
            qpos_dt_sec=0.25,
            joint_velocity_profile="rm65_init_slow",
            max_steps_per_side=10000,
        )


def test_single_arm_qpos_init_rejects_j6_outside_absolute_limit() -> None:
    target = np.zeros(6)
    target[5] = np.deg2rad(120.1)

    with pytest.raises(ValueError, match=r"target_qpos_by_side\.right\.joint6=.*outside"):
        build_rm65_qpos_init_plan(
            start_qpos_by_side={"right": np.zeros(6)},
            target_qpos_by_side={"right": target},
            active_sides=("right",),
            qpos_dt_sec=0.25,
            joint_velocity_profile="rm65_init_slow",
            max_steps_per_side=10000,
        )


def test_qpos_init_blocked_start_error_mentions_safe_pose_recovery() -> None:
    assessment = ArmTransitionRiskAssessment(
        events=(
            ArmTransitionRiskEvent(
                level=ArmTransitionRiskLevel.STOP,
                code="collision_hard",
                message="collision distance is near or below margin",
                value=0.0,
                threshold=0.012,
            ),
        ),
        speed_scale=0.0,
    )
    plan = ArmConstrainedTransitionPlan(
        status="blocked_start",
        start_qpos_by_side={"left": np.zeros(6), "right": np.zeros(6)},
        target_qpos_by_side={"left": np.zeros(6), "right": np.zeros(6)},
        steps=(),
        initial_assessment=assessment,
        target_assessment=ArmTransitionRiskAssessment(),
        blocked_reason="collision_hard",
    )

    with pytest.raises(RuntimeError, match="start qpos is already unsafe.*known safe pose"):
        _require_plan_completed(plan)


def test_dual_arm_qpos_init_branch_search_recovers_from_direct_collision_band() -> None:
    target_left = np.zeros(6)
    target_left[:2] = [0.2, 0.2]

    plan = build_rm65_qpos_init_plan(
        start_qpos_by_side={"left": np.zeros(6), "right": np.zeros(6)},
        target_qpos_by_side={"left": target_left, "right": np.zeros(6)},
        active_sides=("left", "right"),
        qpos_dt_sec=0.25,
        joint_velocity_profile="rm65_init_slow",
        max_steps_per_side=10000,
        collision_model=_DirectPathCollisionBandModel(),
        branch_search=True,
        branch_search_max_branches=2,
    )

    assert plan.completed
    assert plan.metadata["planner"] == "plan_dual_arm_constrained_qpos_transition_branch_search"
    assert plan.metadata["branch_name"] == "both_arms_j1_to_j6"
    assert plan.metadata["direct_plan_summary"]["status"] == "blocked_candidate"
    np.testing.assert_allclose(plan.final_qpos_by_side["left"], target_left)


def test_dual_arm_qpos_init_safely_escapes_joint_hold_margin() -> None:
    start_left = np.zeros(6)
    start_left[1] = float(RM65_ABSOLUTE_Q_MAX_RAD[1] - np.deg2rad(0.5))
    start = {"left": start_left, "right": np.zeros(6)}
    target = {"left": np.zeros(6), "right": np.zeros(6)}

    plan = build_rm65_qpos_init_plan(
        start_qpos_by_side=start,
        target_qpos_by_side=target,
        active_sides=("left", "right"),
        qpos_dt_sec=0.05,
        joint_velocity_profile="rm65_init_slow",
        max_steps_per_side=10000,
        collision_model=_DirectPathCollisionBandModel(),
        branch_search=False,
    )

    assert plan.completed, plan.summary()
    assert plan.metadata["allow_joint_limit_escape"]
    assert plan.metadata["planner_capability"] == QPOS_INIT_PLANNER_CAPABILITY
    assert plan.metadata["joint_limit_escape_used"]
    assert plan.steps[0].qpos_by_side["left"][1] < start_left[1]
    assert all(
        step.qpos_by_side["left"][1] <= RM65_ABSOLUTE_Q_MAX_RAD[1]
        for step in plan.steps
    )
    np.testing.assert_allclose(plan.final_qpos_by_side["left"], target["left"], atol=1e-4)


def test_dual_arm_qpos_init_sampling_recovers_when_fixed_branches_are_exhausted() -> None:
    target_left = np.zeros(6)
    target_left[:2] = [1.0, 1.0]

    plan = build_rm65_qpos_init_plan(
        start_qpos_by_side={"left": np.zeros(6), "right": np.zeros(6)},
        target_qpos_by_side={"left": target_left, "right": np.zeros(6)},
        active_sides=("left", "right"),
        qpos_dt_sec=0.25,
        joint_velocity_profile="rm65_init_slow",
        max_steps_per_side=10000,
        collision_model=_DetourCollisionModel(),
        branch_search=True,
        branch_search_max_branches=8,
        sampling_search_config=RM65SamplingTransitionConfig(
            seed=7,
            max_iterations=1000,
            max_nodes=1800,
            extension_step_deg=15.0,
            edge_resolution_deg=1.0,
            sample_padding_deg=45.0,
        ),
    )

    assert plan.completed, plan.summary()
    assert plan.metadata["planner"] == "bounded_birrt_connect"
    assert plan.metadata["branch_search_status"] == "failed"
    assert len(plan.metadata["branch_search_attempts"]) == 8
    np.testing.assert_allclose(plan.final_qpos_by_side["left"], target_left, atol=1e-4)


def test_resolve_rm65_arm_ips_rejects_documentation_ip_in_latest_without_fallback(tmp_path: Path) -> None:
    config_path = tmp_path / "latest.json"
    config_path.write_text(
        """
{
  "schema_version": "rm65_arm_ip_assignment.v1",
  "assignments": {
    "left": {"arm_ip": "192.0.2.10"},
    "right": {"arm_ip": "192.168.1.20"}
  }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="No adapter default IP fallback"):
        resolve_rm65_arm_ips({}, active_sides=("left", "right"), assignment_path=config_path)


def test_resolve_rm65_arm_ips_allows_explicit_override_when_latest_is_bad(tmp_path: Path) -> None:
    config_path = tmp_path / "latest.json"
    config_path.write_text(
        """
{
  "schema_version": "rm65_arm_ip_assignment.v1",
  "assignments": {
    "left": {"arm_ip": "192.0.2.10"},
    "right": {"arm_ip": "192.168.1.20"}
  }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    ips = resolve_rm65_arm_ips(
        {"left": "192.168.101.21"},
        active_sides=("left", "right"),
        assignment_path=config_path,
    )

    assert ips == {"left": "192.168.101.21", "right": "192.168.1.20"}


def test_resolve_rm65_arm_ips_rejects_explicit_documentation_ip() -> None:
    with pytest.raises(ValueError, match="documentation/test network"):
        resolve_rm65_arm_ips({"left": "192.0.2.10"}, active_sides=("left",))


def test_resolve_rm65_arm_ips_accepts_markdown_link_style_explicit_ip() -> None:
    ips = resolve_rm65_arm_ips(
        {"left": "[192.168.101.21](http://192.168.101.21)"},
        active_sides=("left",),
    )

    assert ips == {"left": "192.168.101.21"}


def test_cli_argv_normalizer_strips_accidental_escaped_space_prefix() -> None:
    argv = [
        "--output",
        "/tmp/report.json",
        " --left-ip",
        "[192.168.101.21](http://192.168.101.21)",
    ]

    assert _normalize_cli_argv(argv) == [
        "--output",
        "/tmp/report.json",
        "--left-ip",
        "[192.168.101.21](http://192.168.101.21)",
    ]
    assert _normalize_sequence_cli_argv(argv) == _normalize_cli_argv(argv)


def test_qpos_sequence_execution_requires_sequence_confirmation(tmp_path: Path) -> None:
    sequence_path = _write_sequence(tmp_path)

    with pytest.raises(ValueError, match=SEQUENCE_CONFIRM_TEXT):
        RM65QposSequenceExecuteConfig(
            sequence_path=sequence_path,
            active_sides=("right",),
            execute=True,
            confirm="",
        )


def test_qpos_sequence_execution_runs_targets_in_order_with_fake_robot(tmp_path: Path) -> None:
    sequence_path = _write_sequence(tmp_path)
    robot = _FakeRM65Robot([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    report = execute_rm65_qpos_test_sequence(
        RM65QposSequenceExecuteConfig(
            sequence_path=sequence_path,
            active_sides=("right",),
            execute=True,
            confirm=SEQUENCE_CONFIRM_TEXT,
            qpos_dt_sec=0.25,
            hold_sec_between_targets=0.0,
            tracking_check_every=1,
            max_tracking_error_deg=0.001,
        ),
        robots_by_side={"right": robot},
    )

    assert report["status"] == "completed"
    assert report["target_count"] == 2
    assert report["completed_count"] == 2
    assert report["execute_requested"]
    assert report["total_sent_step_count"] == report["total_plan_step_count"]
    np.testing.assert_allclose(robot.commands[-1], [4.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert report["results"][0]["target_name"] == "first"
    assert report["results"][1]["target_name"] == "second"


class _FakeRM65Robot:
    def __init__(self, joint_deg: list[float]) -> None:
        self.joint_deg = list(joint_deg)
        self.commands: list[list[float]] = []
        self.stopped = False

    def Get_Current_Arm_State(self, retry: int = 0):
        del retry
        return 0, list(self.joint_deg), [0.0] * 6, 0, 0

    def Get_Force_Data(self):
        zeros = [0.0] * 6
        return 0, zeros, zeros, zeros, zeros

    def Movej_CANFD(self, joint, follow, expand=0, trajectory_mode=0, radio=0):
        del follow, expand, trajectory_mode, radio
        self.joint_deg = [float(value) for value in list(joint)[:6]]
        self.commands.append(list(self.joint_deg))
        return 0

    def Arm_Stop(self):
        self.stopped = True
        return 0

    def Arm_Socket_Close(self):
        return 0


class _DirectPathCollisionBandModel:
    def fk_by_side(self, qpos_by_side: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        del qpos_by_side
        return {"left": np.eye(4), "right": np.eye(4)}

    def min_collision_distance(self, qpos_by_side: Mapping[str, np.ndarray]) -> float:
        left = np.asarray(qpos_by_side["left"], dtype=float).reshape(6)
        if 0.04 < float(left[0]) < 0.16 and 0.04 < float(left[1]) < 0.16:
            return 0.005
        return 0.1

    def singularity_margin_by_side(self, qpos_by_side: Mapping[str, np.ndarray]) -> Mapping[str, float]:
        del qpos_by_side
        return {}


class _DetourCollisionModel(_DirectPathCollisionBandModel):
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


def _write_sequence(tmp_path: Path) -> Path:
    path = tmp_path / "sequence.json"
    save_qpos_test_sequence(
        QposTestSequence(
            name="unit_sequence",
            entries=(
                QposSequenceEntry(
                    index=0,
                    name="first",
                    qpos_by_side={"right": np.deg2rad([3.0, 0.0, 0.0, 0.0, 0.0, 0.0])},
                ),
                QposSequenceEntry(
                    index=1,
                    name="second",
                    qpos_by_side={"right": np.deg2rad([4.0, 0.0, 0.0, 0.0, 0.0, 0.0])},
                ),
            ),
        ),
        path,
    )
    return path
