from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robot_motion.joint_limits import JointVelocityLimitProfile
from robot_motion.profiles import IKSafetyProfile
from robot_motion.rm65 import (
    BimanualIKResult,
    BimanualPoseTarget,
    BimanualPoseTargetSource,
)
from robot_motion.rm65.dual_arm_ik import MultiTaskIKSolver
from robot_motion.types import PoseTolerance
import teleop_runtime.retargeters.dual_arm_collision_aware as dual_module
from teleop_runtime.retargeters.dual_arm_collision_aware import (
    DualArmCollisionAwareRetargeter,
    DualArmCollisionAwareRetargeterConfig,
    DualArmTargetContext,
)
from teleop_runtime.schemas import EffectorCommand


class _FakeModel:
    def __init__(self, *, lower: np.ndarray, upper: np.ndarray):
        self.lowerPositionLimit = lower
        self.upperPositionLimit = upper


class _FakeRobot:
    def __init__(self, *, lower: np.ndarray, upper: np.ndarray):
        self.model = _FakeModel(lower=lower, upper=upper)


class _FakeDualArmSolver:
    def __init__(self, *, lower: np.ndarray | None = None, upper: np.ndarray | None = None):
        self.q = np.zeros(12, dtype=float)
        lower_array = np.full(12, -np.pi, dtype=float) if lower is None else lower
        upper_array = np.full(12, np.pi, dtype=float) if upper is None else upper
        self.robot = _FakeRobot(lower=lower_array, upper=upper_array)
        self.last_solve_profile = {}
        self.last_collision_distance_profile = {}
        self.min_collision_distance = 0.25

    @property
    def current_q(self) -> np.ndarray:
        return self.q.copy()

    def add_frame_task(self, *args, **kwargs) -> None:
        return None

    def add_posture_task(self, *args, **kwargs) -> None:
        return None

    def set_configuration_q(self, qpos: np.ndarray) -> None:
        self.q = np.asarray(qpos, dtype=float).reshape(12).copy()

    def compute_frame_pose(self, frame_name: str, q: np.ndarray | None = None) -> np.ndarray:
        qpos = self.q if q is None else np.asarray(q, dtype=float).reshape(12)
        pose = np.eye(4, dtype=float)
        if frame_name == "arm_link6":
            pose[0, 3] = -float(qpos[0])
        elif frame_name == "right_arm_link6":
            pose[0, 3] = float(qpos[6])
        else:
            raise ValueError(frame_name)
        return pose

    def compute_min_collision_distance(self, q: np.ndarray | None = None) -> float:
        del q
        return float(self.min_collision_distance)


def test_solver_generated_configuration_is_strictly_clamped_before_pink_update() -> None:
    solver = object.__new__(MultiTaskIKSolver)
    solver.robot = _FakeRobot(
        lower=np.array([-2.355, -np.inf, -1.0]),
        upper=np.array([2.355, np.inf, 1.0]),
    )

    bounded = solver._clamp_integrated_configuration_to_limits(
        np.array([2.3585, 42.0, -1.2])
    )

    np.testing.assert_allclose(bounded, [2.355, 42.0, -1.0])


def _rm65_posture_metadata(*, local_escape_enabled: bool = True) -> dict[str, object]:
    return {
        "joint_posture_preference": {
            "enabled": True,
            "name": "rm65_elbow_good_branch",
            "j2_index": 1,
            "j3_index": 2,
            "j2_high_deg": 70.0,
            "j2_preferred_min_deg": -20.0,
            "j2_preferred_max_deg": 60.0,
            "j3_zero_margin_deg": 10.0,
            "j3_low_deg": 30.0,
            "j3_bad_max_deg": 60.0,
            "j3_preferred_min_deg": 80.0,
            "j3_preferred_max_deg": 130.0,
            "bad_branch_penalty_deg": 180.0,
            "j3_negative_penalty_deg": 160.0,
            "high_j2_low_j3_penalty_deg": 120.0,
            "hard_branch_escape_enabled": True,
            "hard_branch_escape_allowed_joint_indices": [1, 2],
            "hard_branch_escape_min_improvement_deg": 40.0,
            "local_escape_enabled": local_escape_enabled,
            "local_escape_max_candidates": 6,
            "local_escape_max_position_worsen_m": 0.04,
            "local_escape_max_orientation_worsen_deg": 15.0,
            "local_escape_min_improvement_deg": 1.0,
            "wrist_unload_enabled": True,
            "wrist_unload_j1_index": 0,
            "wrist_unload_j4_index": 3,
            "wrist_unload_j5_index": 4,
            "wrist_unload_j6_index": 5,
            "wrist_unload_j4_abs_activation_deg": 160.0,
            "wrist_unload_j5_abs_activation_deg": 75.0,
            "wrist_unload_j6_wrap_activation_deg": 165.0,
            "wrist_unload_step_fraction": 1.0,
            "wrist_unload_max_candidates": 8,
            "wrist_unload_max_position_worsen_m": 0.05,
            "wrist_unload_max_orientation_worsen_deg": 20.0,
            "wrist_unload_min_cost_improvement_deg": 0.5,
        }
    }


def _profile(
    *,
    rm65_posture: bool = False,
    rm65_local_escape: bool = True,
    component_progress: bool = False,
    loose_dynamic_worsening: bool = False,
) -> IKSafetyProfile:
    return IKSafetyProfile(
        name="dual_test",
        target_tolerance=PoseTolerance(position_m=0.03, orientation_deg=10.0),
        hard_branch_joint_indices=(0, 1, 2),
        hard_branch_max_delta_deg=(90.0, 90.0, 90.0),
        local_step_enabled=True,
        local_step_scale_factors=(1.0, 0.5, 0.25),
        local_step_limit_escape_enabled=True,
        local_step_limit_escape_activation_margin_deg=2.0,
        local_step_limit_escape_max_candidates=8,
        local_step_accept_worse_within_tolerance=False,
        local_step_component_progress_enabled=component_progress,
        local_step_component_progress_min_position_m=5e-4,
        local_step_component_progress_min_orientation_deg=0.25,
        local_step_component_progress_max_score_regression=1.0,
        local_step_component_progress_max_position_worsen_m=0.015,
        local_step_component_progress_max_orientation_worsen_deg=6.0,
        local_step_component_progress_max_after_position_error_m=0.25,
        local_step_dynamic_worsen_enabled=loose_dynamic_worsening,
        local_step_dynamic_worsen_error_start_m=0.02,
        local_step_dynamic_worsen_position_fraction=1.0,
        local_step_dynamic_worsen_max_position_m=0.10,
        metadata=_rm65_posture_metadata(local_escape_enabled=rm65_local_escape) if rm65_posture else {},
    )


def _velocity() -> JointVelocityLimitProfile:
    return JointVelocityLimitProfile(
        name="test",
        max_velocity_deg_s=(10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
        max_dt_sec=1.0,
    )


def _scene_profile() -> SimpleNamespace:
    return SimpleNamespace(
        name="dual_arm_hand_proxy_open",
        urdf_path="dual.urdf",
        package_dirs=("assets",),
        collision_pairs=(("left", "right"),),
        left_frame_name="arm_link6",
        right_frame_name="right_arm_link6",
        solution_layout="dual_arm",
    )


def _target_pose(*, x_m: float = 0.0, rotation_deg: float = 0.0) -> np.ndarray:
    angle_rad = np.deg2rad(rotation_deg)
    pose = np.eye(4, dtype=float)
    pose[:2, :2] = (
        (np.cos(angle_rad), -np.sin(angle_rad)),
        (np.sin(angle_rad), np.cos(angle_rad)),
    )
    pose[0, 3] = float(x_m)
    return pose


def _retargeter(
    solver: _FakeDualArmSolver,
    *,
    rm65_posture: bool = False,
    rm65_local_escape: bool = True,
    component_progress: bool = False,
    loose_dynamic_worsening: bool = False,
    joint_operating_margin_deg: float = 0.0,
) -> DualArmCollisionAwareRetargeter:
    return DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=15.0,
            enforce_realistic_joint_step=True,
            ik_safety_profile=_profile(
                rm65_posture=rm65_posture,
                rm65_local_escape=rm65_local_escape,
                component_progress=component_progress,
                loose_dynamic_worsening=loose_dynamic_worsening,
            ),
            joint_velocity_profile=_velocity(),
            joint_operating_margin_deg=joint_operating_margin_deg,
            inactive_side_hold_execution_release_enabled=True,
        ),
        ik_solver=solver,
    )


def test_dual_arm_reset_target_pose_smoothing_clears_target_history() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    targets = retargeter.current_world_targets()
    retargeter._last_target_poses = {
        "left": targets["left"].copy(),
        "right": targets["right"].copy(),
    }

    retargeter.reset_target_pose_smoothing()

    assert retargeter._last_target_poses == {"left": None, "right": None}


def test_dual_arm_solve_min_iterations_precedes_tolerance_and_budget_exit() -> None:
    solver = _FakeDualArmSolver()
    solve_calls = 0

    def solve(*, targets):
        nonlocal solve_calls
        del targets
        solve_calls += 1
        return solver.current_q

    solver.solve = solve  # type: ignore[attr-defined]
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=60.0,
            solve_max_iterations=3,
            solve_min_iterations=2,
            solve_adaptive_iterations_enabled=True,
            solve_time_budget_ms=0.01,
            solve_iteration_reserve_ms=0.0,
            solve_translation_tolerance_m=100.0,
            solve_orientation_tolerance_deg=360.0,
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
        ),
        ik_solver=solver,
    )

    _, iterations, _ = retargeter._solve_toward_targets(
        left_target=np.eye(4, dtype=float),
        right_target=np.eye(4, dtype=float),
    )

    assert solve_calls == 2
    assert iterations == 2
    assert retargeter._last_solve_iteration_stop_reason == "pose_tolerance"


def test_dual_arm_target_activity_prefers_tracker_motion_state_machine() -> None:
    retargeter = _retargeter(_FakeDualArmSolver())
    targets = retargeter.current_world_targets()
    left_command = EffectorCommand(
        effector_name="left_arm",
        side="left",
        command_type="cartesian_pose",
        value=np.eye(4, dtype=float),
        frame_name="arm_base",
        metadata={
            "tracker_motion_intent": {
                "state": "INACTIVE",
                "inactive": True,
                "hold_weight": 1.0,
                "transition_reason": "enter_dwell_complete",
                "motion_window": {"quality_valid": True},
            }
        },
    )

    activity = retargeter._target_activity_metadata(
        targets,
        cartesian_commands={"left": left_command, "right": None},
    )

    assert activity["left"]["source"] == "tracker_motion_state_machine"
    assert activity["left"]["state"] == "INACTIVE"
    assert activity["left"]["hold_weight"] == 1.0
    assert activity["right"]["source"] == "target_delta_legacy"


def test_dual_arm_target_activity_accepts_pose_action_interval_intent() -> None:
    retargeter = _retargeter(_FakeDualArmSolver())
    targets = retargeter.current_world_targets()
    activity = retargeter._target_activity_metadata(
        targets,
        source_metadata_by_side={
            "left": {
                "pose_action_motion_intent": {
                    "source": "pose_action_interval_velocity_state_machine",
                    "state": "ACTIVE",
                    "inactive": False,
                    "hold_weight": 0.0,
                    "evidence": {
                        "quality_valid": True,
                        "linear_speed_p90_m_s": 0.12,
                        "angular_speed_p90_rad_s": 0.3,
                    },
                }
            },
            "right": None,
        },
    )

    assert activity["left"]["source"] == (
        "pose_action_interval_velocity_state_machine"
    )
    assert activity["left"]["state"] == "ACTIVE"
    assert activity["left"]["inactive"] is False
    assert activity["left"]["evidence"]["quality_valid"] is True


def test_dual_arm_public_kinematics_interface_is_read_only() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    solver.set_configuration_q(np.linspace(-0.3, 0.3, 12))
    initial_full_q = solver.current_q.copy()

    current = retargeter.current_qpos_by_side()
    np.testing.assert_allclose(current["left"], initial_full_q[:6])
    np.testing.assert_allclose(current["right"], initial_full_q[6:12])
    current["left"][:] = 99.0
    np.testing.assert_allclose(solver.current_q, initial_full_q)

    left = np.linspace(0.1, 0.6, 6)
    right = np.linspace(-0.1, -0.6, 6)
    embedded = retargeter.solver_q_from_qpos_by_side(
        {"left": left, "right": right}
    )
    np.testing.assert_allclose(embedded[:6], left)
    np.testing.assert_allclose(embedded[6:12], right)

    solver_targets = retargeter.solver_world_targets_for_qpos_pair(
        left_qpos=left,
        right_qpos=right,
    )
    tcp_targets = retargeter.tool_tcp_world_targets_for_qpos_pair(
        left_qpos=left,
        right_qpos=right,
    )
    np.testing.assert_allclose(solver_targets["left"][0, 3], -left[0])
    np.testing.assert_allclose(solver_targets["right"][0, 3], right[0])
    np.testing.assert_allclose(tcp_targets["left"], solver_targets["left"])
    np.testing.assert_allclose(tcp_targets["right"], solver_targets["right"])
    np.testing.assert_allclose(solver.current_q, initial_full_q)


def test_bimanual_pose_target_owns_detached_pure_data() -> None:
    left_target = _target_pose(x_m=0.10)
    source_value = _target_pose(x_m=0.05)
    tracker_intent = {"state": "ACTIVE", "inactive": False}
    target = BimanualPoseTarget(
        timestamp=1.25,
        target_pose_by_side={"left": left_target, "right": _target_pose()},
        active_sides=("left",),
        source_by_side={
            "left": BimanualPoseTargetSource(
                command_type="cartesian_pose",
                value=source_value,
                frame_name="arm_base",
                metadata={"tracker_motion_intent": tracker_intent},
            ),
            "right": None,
        },
    )

    left_target[0, 3] = 99.0
    source_value[0, 3] = 99.0
    tracker_intent["state"] = "INACTIVE"

    assert target.target_pose_by_side["left"][0, 3] == pytest.approx(0.10)
    assert target.source_by_side["left"] is not None
    assert target.source_by_side["left"].value[0, 3] == pytest.approx(0.05)
    assert (
        target.source_by_side["left"].metadata["tracker_motion_intent"]["state"]
        == "ACTIVE"
    )


def test_pure_bimanual_solve_returns_atomic_result_and_facade_round_trips() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    candidate = np.zeros(12, dtype=float)
    candidate[0] = 0.04
    candidate[6] = -0.06
    solver.solve = lambda *, targets: candidate.copy()  # type: ignore[attr-defined]
    target = BimanualPoseTarget(
        timestamp=2.0,
        target_pose_by_side={
            "left": _target_pose(x_m=0.10),
            "right": _target_pose(x_m=-0.10),
        },
        active_sides=("left", "right"),
        source_by_side={
            side: BimanualPoseTargetSource(
                command_type="cartesian_pose",
                value=_target_pose(),
                frame_name="arm_base",
                metadata={
                    "tracker_motion_intent": {
                        "state": "ACTIVE",
                        "inactive": False,
                        "hold_weight": 0.0,
                    }
                },
            )
            for side in ("left", "right")
        },
    )

    result = retargeter.solve_target_pair(target)

    assert isinstance(result, BimanualIKResult)
    assert tuple(result.side_results) == ("left", "right")
    assert result.status in {"tracking", "boundary_active"}
    assert result.metadata["target_activity"]["left"]["source"] == (
        "tracker_motion_state_machine"
    )
    np.testing.assert_allclose(
        result.qpos_by_side()["left"],
        result.side_results["left"].metadata["solver_output_qpos"],
    )
    np.testing.assert_allclose(
        result.qpos_by_side()["right"],
        result.side_results["right"].metadata["solver_output_qpos"],
    )

    command = retargeter._command_from_bimanual_result(result)
    assert command.timestamp == 2.0
    assert command.metadata == result.metadata
    for side in ("left", "right"):
        effector = command.commands[f"{side}_arm"]
        side_result = result.side_results[side]
        assert effector.command_type == side_result.command_type
        assert effector.frame_name == side_result.command_frame_name
        np.testing.assert_allclose(effector.value, side_result.command_value)


def test_bimanual_contract_facade_has_explicit_compatibility_rollback() -> None:
    solver = _FakeDualArmSolver()
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            enforce_realistic_joint_step=False,
            bimanual_result_contract_enabled=False,
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
        ),
        ik_solver=solver,
    )
    context = DualArmTargetContext(
        timestamp=3.0,
        left_target=_target_pose(),
        right_target=_target_pose(),
        left_world_to_base=None,
        right_world_to_base=None,
        left_cartesian=None,
        right_cartesian=None,
        left_input_skipped=None,
        right_input_skipped=None,
    )

    command = retargeter.build_command_for_context(context)

    assert command.timestamp == 3.0
    assert tuple(command.commands) == ("left_arm", "right_arm")
    assert command.metadata["collision_avoidance_status"] in {
        "tracking",
        "boundary_active",
        "hold_last_safe",
    }


def test_bimanual_contract_facade_can_be_enabled_explicitly() -> None:
    retargeter = _retargeter(_FakeDualArmSolver())
    retargeter.config.bimanual_result_contract_enabled = True
    results: list[BimanualIKResult] = []
    original_convert = retargeter._command_from_bimanual_result

    def recording_convert(result: BimanualIKResult):
        results.append(result)
        return original_convert(result)

    retargeter._command_from_bimanual_result = recording_convert  # type: ignore[method-assign]
    context = DualArmTargetContext(
        timestamp=4.0,
        left_target=_target_pose(),
        right_target=_target_pose(),
        left_world_to_base=None,
        right_world_to_base=None,
        left_cartesian=None,
        right_cartesian=None,
        left_input_skipped=None,
        right_input_skipped=None,
    )

    command = retargeter.build_command_for_context(context)

    assert len(results) == 1
    assert isinstance(results[0], BimanualIKResult)
    assert command.timestamp == 4.0
    assert tuple(command.commands) == ("left_arm", "right_arm")


@pytest.mark.parametrize("output_command_type", ["joint_vector", "cartesian_pose"])
def test_native_and_compat_bimanual_outputs_are_field_equivalent(
    output_command_type: str,
) -> None:
    compat = _retargeter(_FakeDualArmSolver())
    native = _retargeter(_FakeDualArmSolver())
    compat.config.output_command_type = output_command_type
    native.config.output_command_type = output_command_type
    compat.config.bimanual_result_contract_enabled = False
    native.config.bimanual_result_contract_enabled = True
    source = EffectorCommand(
        effector_name="left_arm",
        side="left",
        command_type="cartesian_pose",
        value=_target_pose(),
        frame_name="arm_base",
        metadata={
            "tracker_motion_intent": {
                "state": "ACTIVE",
                "inactive": False,
                "hold_weight": 0.0,
            }
        },
    )
    context = DualArmTargetContext(
        timestamp=5.0,
        left_target=_target_pose(),
        right_target=_target_pose(),
        left_world_to_base=None,
        right_world_to_base=None,
        left_cartesian=source,
        right_cartesian=None,
        left_input_skipped=None,
        right_input_skipped="inactive",
    )

    compat_command = compat.build_command_for_context(context)
    native_command = native.build_command_for_context(context)

    def assert_equivalent(left, right, *, key: str | None = None) -> None:
        if key == "ik_frame_total_ms":
            assert isinstance(left, float)
            assert isinstance(right, float)
            return
        if isinstance(left, dict):
            assert isinstance(right, dict)
            assert set(left) == set(right)
            for child_key in left:
                assert_equivalent(left[child_key], right[child_key], key=child_key)
            return
        if isinstance(left, np.ndarray):
            np.testing.assert_allclose(left, right)
            return
        if isinstance(left, float):
            assert float(right) == pytest.approx(left)
            return
        assert right == left

    assert tuple(compat_command.commands) == tuple(native_command.commands)
    assert_equivalent(compat_command.metadata, native_command.metadata)
    for effector_name in compat_command.commands:
        left = compat_command.commands[effector_name]
        right = native_command.commands[effector_name]
        assert left.effector_name == right.effector_name
        assert left.side == right.side
        assert left.command_type == right.command_type
        assert left.frame_name == right.frame_name
        np.testing.assert_allclose(left.value, right.value)
        assert_equivalent(left.metadata, right.metadata)


def test_dual_arm_collision_retargeter_defaults_to_robot_motion_builder(monkeypatch) -> None:
    scene = _scene_profile()
    solver = _FakeDualArmSolver()
    calls: list[str] = []

    def fake_robot_motion_builder(config):
        calls.append(config.scene_profile)
        return SimpleNamespace(solver=solver, scene_profile=scene)

    monkeypatch.setattr(dual_module, "build_rm65_dual_arm_scene_profile", lambda _name: scene)
    monkeypatch.setattr(dual_module, "build_rm65_dual_arm_collision_ik_solver", fake_robot_motion_builder)

    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            scene_profile="dual_arm_hand_proxy_open",
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
        )
    )

    assert retargeter.config.ik_framework == "robot_motion"
    assert retargeter.ik_solver is solver
    assert retargeter.profile is scene
    assert calls == ["dual_arm_hand_proxy_open"]


def test_dual_arm_collision_retargeter_rejects_legacy_framework() -> None:
    try:
        DualArmCollisionAwareRetargeterConfig(
            ik_framework="legacy",
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
        )
    except ValueError as exc:
        assert "Unsupported dual-arm IK framework" in str(exc)
    else:
        raise AssertionError("legacy IK framework should not be accepted")


def test_dual_arm_candidate_layer_generates_joint_limit_escape_for_left_arm() -> None:
    lower = np.full(12, -1.0, dtype=float)
    upper = np.full(12, 1.0, dtype=float)
    solver = _FakeDualArmSolver(lower=lower, upper=upper)
    retargeter = _retargeter(solver)
    reference = np.zeros(12, dtype=float)
    reference[0] = 0.99
    solver.set_configuration_q(reference)
    retargeter.set_execution_dt_sec(1.0 / 15.0)
    targets = retargeter.current_world_targets()
    targets["left"] = targets["left"].copy()
    targets["left"][0, 3] = -0.50

    qpos, name, _score, _min_distance, metadata = retargeter._select_safe_tracking_candidate(
        raw_full_q=reference.copy(),
        reference_full_q=reference.copy(),
        current_targets=targets,
        profile=None,
    )

    assert name == "joint_limit_escape_left_j1_neg"
    assert qpos[0] < reference[0]
    assert metadata["limit_escape"]["status"] == "active"
    assert metadata["selected_source"] == "joint_limit_escape"


def test_dual_arm_operating_margin_rejects_outward_tracking_and_selects_escape() -> None:
    lower = np.full(12, -1.0, dtype=float)
    upper = np.full(12, 1.0, dtype=float)
    solver = _FakeDualArmSolver(lower=lower, upper=upper)
    retargeter = _retargeter(solver, joint_operating_margin_deg=3.0)
    reference = np.zeros(12, dtype=float)
    reference[0] = 0.99
    raw = reference.copy()
    raw[0] = 1.0
    solver.set_configuration_q(reference)
    retargeter.set_execution_dt_sec(1.0 / 15.0)
    targets = retargeter.current_world_targets()
    targets["left"] = targets["left"].copy()
    # The raw tracking solution improves Cartesian error by moving J1 outward,
    # but would consume the last operating margin and must be rejected.
    targets["left"][0, 3] = -1.5

    qpos, name, _score, _min_distance, metadata = retargeter._select_safe_tracking_candidate(
        raw_full_q=raw,
        reference_full_q=reference,
        current_targets=targets,
        profile=None,
    )

    assert name == "joint_limit_escape_left_j1_neg"
    assert qpos[0] < reference[0]
    assert metadata["selected_source"] == "joint_limit_escape"
    assert any(
        item["reject_reason"] == "joint_operating_margin"
        for item in metadata["last_rejections"]
    )
    operating_margin = metadata["selected_metadata"]["joint_operating_margin"]
    assert operating_margin["status"] == "recovery_only"
    assert operating_margin["violations"][0]["moving_inward"] is True


def test_dual_arm_candidate_layer_returns_base_candidate_without_auxiliary_expansion() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver, rm65_posture=True)
    reference = np.zeros(12, dtype=float)
    raw = reference.copy()
    raw[6] = np.deg2rad(2.0)
    solver.set_configuration_q(reference)
    targets = retargeter.current_world_targets()
    targets["right"] = targets["right"].copy()
    targets["right"][0, 3] = np.deg2rad(2.0)

    qpos, name, _score, _min_distance, metadata = retargeter._select_safe_tracking_candidate(
        raw_full_q=raw,
        reference_full_q=reference,
        current_targets=targets,
        profile=None,
    )

    assert name == "original"
    assert qpos[6] > reference[6]
    assert metadata["selected_source"] == "multitask"
    assert metadata["candidate_eval_mode"] == "base_only"
    assert metadata["limit_escape"]["status"] == "deferred"
    assert [stage["stage"] for stage in metadata["candidate_eval_stages"]] == ["base_primary"]


def test_position_candidate_layer_prefers_feasible_non_wrist_singular_branch(
    monkeypatch,
) -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    retargeter.config.prefer_non_wrist_singular_branch = True
    retargeter.config.wrist_singularity_threshold_deg = 3.0
    reference = np.zeros(12, dtype=float)
    singular = reference.copy()
    singular[4] = np.deg2rad(1.0)
    singular[10] = np.deg2rad(5.0)
    non_singular = reference.copy()
    non_singular[4] = np.deg2rad(5.0)
    non_singular[10] = np.deg2rad(5.0)
    candidates = [
        dual_module._DualArmCandidate(
            name="singular_low_score",
            qpos=singular,
            source="multitask",
            method="singular_low_score",
        ),
        dual_module._DualArmCandidate(
            name="non_singular_high_score",
            qpos=non_singular,
            source="multitask",
            method="non_singular_high_score",
        ),
    ]

    monkeypatch.setattr(
        retargeter,
        "_dual_arm_tracking_candidates",
        lambda **_kwargs: (candidates, {"enabled": False, "status": "deferred"}),
    )

    def evaluate(*, candidate, **_kwargs):
        return (
            True,
            candidate.qpos.copy(),
            {"score": 1.0 if candidate.name.startswith("singular") else 100.0},
        )

    monkeypatch.setattr(retargeter, "_evaluate_dual_arm_step_candidate", evaluate)
    monkeypatch.setattr(retargeter, "_profiled_min_collision_distance", lambda *_a, **_k: 1.0)

    qpos, name, _score, _distance, metadata = retargeter._select_safe_tracking_candidate(
        raw_full_q=singular,
        reference_full_q=reference,
        current_targets=retargeter.current_world_targets(),
        profile=None,
    )

    assert name == "non_singular_high_score"
    assert np.allclose(qpos, non_singular)
    preference = metadata["wrist_singularity_preference"]
    assert preference["fully_non_singular_count"] == 1
    assert preference["fallback_to_singular"] is False
    assert preference["selected"]["singular_side_count"] == 0

    retargeter.config.prefer_non_wrist_singular_branch = False
    rollback_qpos, rollback_name, *_ = retargeter._select_safe_tracking_candidate(
        raw_full_q=singular,
        reference_full_q=reference,
        current_targets=retargeter.current_world_targets(),
        profile=None,
    )
    assert rollback_name == "singular_low_score"
    assert np.allclose(rollback_qpos, singular)


def test_position_candidate_layer_accepts_singular_when_no_clean_branch(
    monkeypatch,
) -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    retargeter.config.prefer_non_wrist_singular_branch = True
    reference = np.zeros(12, dtype=float)
    lower_score = reference.copy()
    lower_score[4] = np.deg2rad(1.0)
    lower_score[10] = np.deg2rad(5.0)
    higher_score = lower_score.copy()
    higher_score[4] = np.deg2rad(2.0)
    candidates = [
        dual_module._DualArmCandidate("singular_best", lower_score, "multitask", "best"),
        dual_module._DualArmCandidate("singular_other", higher_score, "multitask", "other"),
    ]
    monkeypatch.setattr(
        retargeter,
        "_dual_arm_tracking_candidates",
        lambda **_kwargs: (candidates, {"enabled": False, "status": "deferred"}),
    )
    monkeypatch.setattr(
        retargeter,
        "_evaluate_dual_arm_step_candidate",
        lambda *, candidate, **_kwargs: (
            True,
            candidate.qpos.copy(),
            {"score": 1.0 if candidate.name == "singular_best" else 2.0},
        ),
    )
    monkeypatch.setattr(retargeter, "_profiled_min_collision_distance", lambda *_a, **_k: 1.0)

    qpos, name, _score, _distance, metadata = retargeter._select_safe_tracking_candidate(
        raw_full_q=lower_score,
        reference_full_q=reference,
        current_targets=retargeter.current_world_targets(),
        profile=None,
    )

    assert name == "singular_best"
    assert np.allclose(qpos, lower_score)
    preference = metadata["wrist_singularity_preference"]
    assert preference["fully_non_singular_count"] == 0
    assert preference["fallback_to_singular"] is True
    assert preference["selected"]["sides"]["left"]["is_wrist_singular"] is True


def test_position_candidate_layer_expands_auxiliary_for_elbow_singularity(
    monkeypatch,
) -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver, rm65_posture=True)
    retargeter.config.prefer_non_elbow_singular_branch = True
    retargeter.config.prefer_non_wrist_singular_branch = False
    reference = np.zeros(12, dtype=float)
    reference[2] = np.deg2rad(6.0)
    reference[8] = np.deg2rad(6.0)
    elbow_singular = reference.copy()
    elbow_singular[2] = np.deg2rad(1.0)
    elbow_clean = reference.copy()
    elbow_clean[2] = np.deg2rad(7.0)
    base = dual_module._DualArmCandidate(
        "original", elbow_singular, "multitask", "original"
    )
    auxiliary = dual_module._DualArmCandidate(
        "posture_escape_left_j3",
        elbow_clean,
        "posture_escape",
        "posture_escape_left_j3",
    )

    def candidates(**kwargs):
        if kwargs.get("include_posture_escape", False):
            return [auxiliary], {
                "posture_escape": {"enabled": True, "status": "active"}
            }
        return [base], {"enabled": False, "status": "deferred"}

    monkeypatch.setattr(retargeter, "_dual_arm_tracking_candidates", candidates)
    monkeypatch.setattr(
        retargeter,
        "_evaluate_dual_arm_step_candidate",
        lambda *, candidate, **_kwargs: (
            True,
            candidate.qpos.copy(),
            {"score": 1.0 if candidate.name == "original" else 100.0},
        ),
    )
    monkeypatch.setattr(retargeter, "_profiled_min_collision_distance", lambda *_a, **_k: 1.0)

    qpos, name, _score, _distance, metadata = retargeter._select_safe_tracking_candidate(
        raw_full_q=elbow_singular,
        reference_full_q=reference,
        current_targets=retargeter.current_world_targets(),
        profile=None,
    )

    assert name == "posture_escape_left_j3"
    assert np.allclose(qpos, elbow_clean)
    assert metadata["candidate_eval_mode"] == "base_then_auxiliary_singularity_search"
    assert metadata["expanded_for_elbow_singularity"] is True
    preference = metadata["position_singularity_preference"]
    assert preference["fully_non_singular_count"] == 1
    assert preference["fallback_to_singular"] is False
    assert preference["selected"]["any_elbow_singular"] is False


def test_position_candidate_layer_marks_elbow_fallback_when_no_clean_candidate(
    monkeypatch,
) -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver, rm65_posture=True)
    retargeter.config.prefer_non_elbow_singular_branch = True
    retargeter.config.prefer_non_wrist_singular_branch = False
    reference = np.zeros(12, dtype=float)
    reference[2] = np.deg2rad(6.0)
    reference[8] = np.deg2rad(6.0)
    singular = reference.copy()
    singular[2] = np.deg2rad(1.0)
    candidate = dual_module._DualArmCandidate(
        "original", singular, "multitask", "original"
    )

    monkeypatch.setattr(
        retargeter,
        "_dual_arm_tracking_candidates",
        lambda **_kwargs: ([candidate], {"enabled": False, "status": "deferred"}),
    )
    monkeypatch.setattr(
        retargeter,
        "_evaluate_dual_arm_step_candidate",
        lambda *, candidate, **_kwargs: (
            True,
            candidate.qpos.copy(),
            {"score": 1.0},
        ),
    )
    monkeypatch.setattr(retargeter, "_profiled_min_collision_distance", lambda *_a, **_k: 1.0)

    qpos, name, _score, _distance, metadata = retargeter._select_safe_tracking_candidate(
        raw_full_q=singular,
        reference_full_q=reference,
        current_targets=retargeter.current_world_targets(),
        profile=None,
    )

    assert name == "original"
    assert np.allclose(qpos, singular)
    preference = metadata["position_singularity_preference"]
    assert preference["fully_non_singular_count"] == 0
    assert preference["fallback_to_singular"] is True
    assert preference["selected"]["sides"]["left"]["is_elbow_singular"] is True

    retargeter.config.prefer_non_elbow_singular_branch = False
    rollback = retargeter._position_singularity_metadata(singular)
    assert rollback["any_elbow_singular"] is False


def test_position_candidate_layer_prefers_clean_shoulder_branch_and_is_rollbackable(
    monkeypatch,
) -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver, rm65_posture=True)
    retargeter.config.prefer_non_elbow_singular_branch = False
    retargeter.config.prefer_non_wrist_singular_branch = False
    retargeter.config.prefer_non_shoulder_singular_branch = True
    reference = np.zeros(12, dtype=float)
    reference[0] = 0.2
    reference[6] = 0.2
    shoulder_singular = reference.copy()
    shoulder_singular[0] = 0.0
    shoulder_clean = reference.copy()
    shoulder_clean[0] = 0.3
    base = dual_module._DualArmCandidate(
        "original", shoulder_singular, "multitask", "original"
    )
    auxiliary = dual_module._DualArmCandidate(
        "shoulder_escape", shoulder_clean, "posture_escape", "shoulder_escape"
    )

    def analyze(qpos: np.ndarray) -> dict[str, object]:
        singular = abs(float(np.asarray(qpos)[0])) < 0.05
        return {
            "status_code": -1 if singular else 0,
            "label": "shoulder_singularity" if singular else "normal",
            "wrist_center_distance_m": 0.01 if singular else 0.20,
        }

    retargeter.set_position_singularity_analyzers(
        {"left": analyze, "right": analyze}
    )

    def candidates(**kwargs):
        if kwargs.get("include_posture_escape", False):
            return [auxiliary], {
                "posture_escape": {"enabled": True, "status": "active"}
            }
        return [base], {"enabled": False, "status": "deferred"}

    monkeypatch.setattr(retargeter, "_dual_arm_tracking_candidates", candidates)
    monkeypatch.setattr(
        retargeter,
        "_evaluate_dual_arm_step_candidate",
        lambda *, candidate, **_kwargs: (
            True,
            candidate.qpos.copy(),
            {"score": 1.0 if candidate.name == "original" else 100.0},
        ),
    )
    monkeypatch.setattr(
        retargeter, "_profiled_min_collision_distance", lambda *_a, **_k: 1.0
    )

    qpos, name, _score, _distance, metadata = (
        retargeter._select_safe_tracking_candidate(
            raw_full_q=shoulder_singular,
            reference_full_q=reference,
            current_targets=retargeter.current_world_targets(),
            profile=None,
        )
    )

    assert name == "shoulder_escape"
    assert np.allclose(qpos, shoulder_clean)
    assert metadata["expanded_for_shoulder_singularity"] is True
    preference = metadata["position_singularity_preference"]
    assert preference["fully_non_singular_count"] == 1
    assert preference["fallback_to_singular"] is False
    assert preference["selected"]["any_shoulder_singular"] is False

    retargeter.config.prefer_non_shoulder_singular_branch = False
    rollback = retargeter._position_singularity_metadata(shoulder_singular)
    assert rollback["any_shoulder_singular"] is False


def test_position_candidate_layer_certifies_shoulder_fallback_when_no_clean_branch(
    monkeypatch,
) -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    retargeter.config.prefer_non_elbow_singular_branch = False
    retargeter.config.prefer_non_wrist_singular_branch = False
    retargeter.config.prefer_non_shoulder_singular_branch = True
    reference = np.full(12, 0.2, dtype=float)
    singular = reference.copy()
    singular[0] = 0.0
    candidate = dual_module._DualArmCandidate(
        "original", singular, "multitask", "original"
    )

    def analyze(qpos: np.ndarray) -> dict[str, object]:
        singular_side = abs(float(np.asarray(qpos)[0])) < 0.05
        return {
            "status_code": -1 if singular_side else 0,
            "label": "shoulder_singularity" if singular_side else "normal",
            "wrist_center_distance_m": 0.01 if singular_side else 0.20,
        }

    retargeter.set_position_singularity_analyzers(
        {"left": analyze, "right": analyze}
    )
    monkeypatch.setattr(
        retargeter,
        "_dual_arm_tracking_candidates",
        lambda **_kwargs: ([candidate], {"enabled": False, "status": "deferred"}),
    )
    monkeypatch.setattr(
        retargeter,
        "_evaluate_dual_arm_step_candidate",
        lambda *, candidate, **_kwargs: (
            True,
            candidate.qpos.copy(),
            {"score": 1.0},
        ),
    )
    monkeypatch.setattr(
        retargeter, "_profiled_min_collision_distance", lambda *_a, **_k: 1.0
    )

    qpos, name, _score, _distance, metadata = (
        retargeter._select_safe_tracking_candidate(
            raw_full_q=singular,
            reference_full_q=reference,
            current_targets=retargeter.current_world_targets(),
            profile=None,
        )
    )

    assert name == "original"
    assert np.allclose(qpos, singular)
    preference = metadata["position_singularity_preference"]
    assert preference["fully_non_singular_count"] == 0
    assert preference["fallback_to_singular"] is True
    selected_left = preference["selected"]["sides"]["left"]
    assert selected_left["analyzer_available"] is True
    assert selected_left["is_shoulder_singular"] is True


def test_dual_arm_candidate_deadline_holds_without_fallback_or_auxiliary_expansion(monkeypatch) -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    retargeter.config.solve_time_budget_ms = 10.0
    reference = np.zeros(12, dtype=float)
    raw = np.deg2rad(np.arange(1.0, 13.0, dtype=float))
    solver.set_configuration_q(reference)
    targets = retargeter.current_world_targets()
    evaluation_calls: list[str] = []
    candidate_generation_calls: list[dict[str, object]] = []
    clock = {"value": 0.0}
    original_candidates = retargeter._dual_arm_tracking_candidates

    def fake_candidates(**kwargs):
        candidate_generation_calls.append(dict(kwargs))
        return original_candidates(**kwargs)

    def reject_candidate(*, candidate, **_kwargs):
        evaluation_calls.append(candidate.name)
        clock["value"] += 0.002
        return False, candidate.qpos.copy(), {"score": 1e18, "reject_reason": "test_reject"}

    monkeypatch.setattr(dual_module.time, "perf_counter", lambda: clock["value"])
    monkeypatch.setattr(retargeter, "_dual_arm_tracking_candidates", fake_candidates)
    monkeypatch.setattr(retargeter, "_evaluate_dual_arm_step_candidate", reject_candidate)

    qpos, name, _score, _distance, metadata = retargeter._select_safe_tracking_candidate(
        raw_full_q=raw,
        reference_full_q=reference,
        current_targets=targets,
        profile={"solve_total_ms": 0.0},
    )

    assert name == "local_step_hold"
    assert np.allclose(qpos, reference)
    assert len(evaluation_calls) == 3
    assert len(candidate_generation_calls) == 1
    assert metadata["reason"] == "branch_budget_exhausted_no_safe_candidate"
    assert metadata["candidate_eval_mode"] == "deadline_hold"
    assert metadata["candidate_eval_stop_reason"] == "branch_budget_exhausted_before_candidate"
    assert [stage["stage"] for stage in metadata["candidate_eval_stages"]] == ["base_primary"]


def test_dual_arm_candidate_deadline_accounts_for_pre_selection_frame_time(monkeypatch) -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    retargeter.config.solve_time_budget_ms = 10.0
    reference = np.zeros(12, dtype=float)
    solver.set_configuration_q(reference)
    targets = retargeter.current_world_targets()
    evaluation_calls: list[str] = []

    def fail_if_evaluated(*, candidate, **_kwargs):
        evaluation_calls.append(candidate.name)
        raise AssertionError("candidate evaluation must not start after the frame budget is exhausted")

    monkeypatch.setattr(retargeter, "_evaluate_dual_arm_step_candidate", fail_if_evaluated)

    qpos, name, _score, _distance, metadata = retargeter._select_safe_tracking_candidate(
        raw_full_q=np.deg2rad(np.arange(1.0, 13.0, dtype=float)),
        reference_full_q=reference,
        current_targets=targets,
        profile=None,
        elapsed_before_selection_ms=10.0,
    )

    assert name == "local_step_hold"
    assert np.allclose(qpos, reference)
    assert evaluation_calls == []
    assert metadata["candidate_eval_stop_reason"] == "branch_budget_exhausted_before_candidates"
    assert metadata["candidate_eval_pre_selection_elapsed_ms"] == 10.0


def test_candidate_certification_reserve_reassigns_existing_total_reserve() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    retargeter.config.solve_time_budget_ms = 10.0
    retargeter.config.solve_iteration_reserve_ms = 5.0
    reference = np.zeros(12, dtype=float)
    raw = np.deg2rad(np.arange(1.0, 13.0, dtype=float))
    solver.set_configuration_q(reference)
    targets = retargeter.current_world_targets()

    retargeter.config.candidate_certification_reserve_ms = 3.5
    _qpos, _name, _score, _distance, reserved = (
        retargeter._select_safe_tracking_candidate(
            raw_full_q=raw,
            reference_full_q=reference,
            current_targets=targets,
            profile=None,
            elapsed_before_selection_ms=4.0,
        )
    )
    assert reserved["candidate_certification_reserve_ms"] == pytest.approx(3.5)
    assert reserved["candidate_post_selection_reserve_ms"] == pytest.approx(1.5)
    assert reserved["candidate_eval_branch_budget_ms"] == pytest.approx(4.5)

    retargeter.config.candidate_certification_reserve_ms = 0.0
    _qpos, _name, _score, _distance, rollback = (
        retargeter._select_safe_tracking_candidate(
            raw_full_q=raw,
            reference_full_q=reference,
            current_targets=targets,
            profile=None,
            elapsed_before_selection_ms=4.0,
        )
    )
    assert rollback["candidate_post_selection_reserve_ms"] == pytest.approx(5.0)
    assert rollback["candidate_eval_branch_budget_ms"] == pytest.approx(1.0)


def test_dual_arm_candidate_deadline_selects_best_already_safe_candidate(monkeypatch) -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    retargeter.config.solve_time_budget_ms = 10.0
    reference = np.zeros(12, dtype=float)
    raw = np.deg2rad(np.arange(1.0, 13.0, dtype=float))
    solver.set_configuration_q(reference)
    targets = retargeter.current_world_targets()
    evaluation_calls: list[str] = []
    clock = {"value": 0.0}

    def accept_candidate(*, candidate, **_kwargs):
        evaluation_calls.append(candidate.name)
        clock["value"] += 0.002
        return True, candidate.qpos.copy(), {"score": float(10 - len(evaluation_calls))}

    monkeypatch.setattr(dual_module.time, "perf_counter", lambda: clock["value"])
    monkeypatch.setattr(retargeter, "_evaluate_dual_arm_step_candidate", accept_candidate)

    _qpos, name, _score, _distance, metadata = retargeter._select_safe_tracking_candidate(
        raw_full_q=raw,
        reference_full_q=reference,
        current_targets=targets,
        profile={"solve_total_ms": 0.0},
    )

    assert len(evaluation_calls) == 3
    assert name == evaluation_calls[1]
    assert metadata["candidate_eval_mode"] == "deadline_partial"
    assert metadata["candidate_eval_stopped_early"] is True
    assert metadata["collision_safe_count"] == 2


def test_dual_arm_candidate_layer_holds_on_hard_branch_jump() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    reference = np.zeros(12, dtype=float)
    raw = reference.copy()
    raw[0] = np.deg2rad(100.0)
    targets = retargeter.current_world_targets()

    qpos, name, _score, _min_distance, metadata = retargeter._select_safe_tracking_candidate(
        raw_full_q=raw,
        reference_full_q=reference,
        current_targets=targets,
        profile=None,
    )

    assert name == "local_step_hold"
    assert np.allclose(qpos, reference)
    assert metadata["reason"] == "no_progress_candidate"
    assert metadata["last_rejections"][-1]["reject_reason"] == "hard_branch_jump:left"


def test_dual_arm_position_passthrough_skips_local_step_hold_and_keeps_final_step_limit() -> None:
    solver = _FakeDualArmSolver()
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=15.0,
            enforce_realistic_joint_step=True,
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
            local_step_candidate_selection_enabled=False,
            inactive_side_hold_enabled=False,
            inactive_side_hold_execution_cap_enabled=False,
        ),
        ik_solver=solver,
    )
    reference = np.zeros(12, dtype=float)
    raw = reference.copy()
    raw[0] = np.deg2rad(100.0)
    targets = retargeter.current_world_targets()

    qpos, name, _score, _min_distance, metadata = retargeter._select_safe_tracking_candidate(
        raw_full_q=raw,
        reference_full_q=reference,
        current_targets=targets,
        profile=None,
    )
    limited_qpos, step_metadata = retargeter._limit_to_execution_step(
        reference_full_q=reference,
        candidate_full_q=qpos,
    )

    assert name == "original"
    assert metadata["enabled"] is False
    assert metadata["reason"] == "local_step_candidate_selection_disabled"
    assert np.isclose(limited_qpos[0], np.deg2rad(10.0 / 15.0))
    assert step_metadata["step_limited"] is True


def test_dual_arm_component_progress_accepts_bounded_fine_tracking_step() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver, component_progress=True)
    reference = np.zeros(12, dtype=float)
    raw = reference.copy()
    raw[0] = 0.002
    raw[6] = -0.002
    solver.set_configuration_q(reference)
    targets = retargeter.current_world_targets()
    targets["left"] = targets["left"].copy()
    targets["right"] = targets["right"].copy()
    targets["left"][0, 3] = -0.010
    targets["right"][0, 3] = 0.180

    qpos, name, _score, _min_distance, metadata = retargeter._select_safe_tracking_candidate(
        raw_full_q=raw,
        reference_full_q=reference,
        current_targets=targets,
        profile=None,
    )

    assert name == "original"
    assert qpos[0] > reference[0]
    assert qpos[6] < reference[6]
    selected = metadata["selected_metadata"]
    assert selected["dual_progress_accept_reason"] == "dual_component_progress_bounded_score_regression"
    assert selected["dual_component_progress_ok"] is True
    assert selected["dual_max_position_progress_m"] >= 5e-4


def test_dual_arm_component_progress_still_holds_on_large_score_regression() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver, component_progress=True)
    reference = np.zeros(12, dtype=float)
    raw = reference.copy()
    raw[0] = 0.002
    raw[6] = -0.020
    solver.set_configuration_q(reference)
    targets = retargeter.current_world_targets()
    targets["left"] = targets["left"].copy()
    targets["right"] = targets["right"].copy()
    targets["left"][0, 3] = -0.010
    targets["right"][0, 3] = 0.180

    qpos, name, _score, _min_distance, metadata = retargeter._select_safe_tracking_candidate(
        raw_full_q=raw,
        reference_full_q=reference,
        current_targets=targets,
        profile=None,
    )

    assert name == "local_step_hold"
    assert np.allclose(qpos, reference)
    assert metadata["reason"] == "no_progress_candidate"
    assert metadata["last_rejections"][-1]["dual_component_progress_ok"] is False


def test_dual_arm_activity_hold_reduces_unmoved_side_motion() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver, loose_dynamic_worsening=True)
    reference = np.zeros(12, dtype=float)
    raw = reference.copy()
    raw[0] = 0.050
    raw[6] = 0.050
    solver.set_configuration_q(reference)
    retargeter.set_execution_dt_sec(1.0)
    targets = retargeter.current_world_targets()
    retargeter._last_target_poses = {
        "left": targets["left"].copy(),
        "right": targets["right"].copy(),
    }
    targets["left"] = targets["left"].copy()
    targets["right"] = targets["right"].copy()
    targets["right"][0, 3] = 0.080

    qpos, name, _score, _min_distance, metadata = retargeter._select_safe_tracking_candidate(
        raw_full_q=raw,
        reference_full_q=reference,
        current_targets=targets,
        profile=None,
    )

    assert name == "original"
    assert np.isclose(qpos[0], np.deg2rad(0.5))
    assert np.isclose(qpos[6], 0.050)
    assert metadata["inactive_side_hold_active"] is True
    selected = metadata["selected_metadata"]
    assert selected["step_scale"] == 1.0
    assert selected["inactive_side_hold_execution_cap"]["applied"] is True
    assert selected["inactive_side_hold_execution_cap"]["sides"]["left"]["applied"] is True
    assert selected["inactive_side_hold"]["sides"]["left"]["hold_weight"] == 1.0
    assert selected["inactive_side_hold"]["sides"]["left"]["unexplained_position_m"] <= 0.025


def test_dual_arm_inactive_execution_step_cap_limits_stationary_side() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    reference = np.zeros(12, dtype=float)
    candidate = reference.copy()
    candidate[6] = np.deg2rad(3.0)
    target_activity = {
        "enabled": True,
        "right": {"inactive": True, "hold_weight": 1.0},
        "left": {"inactive": False, "hold_weight": 0.0},
    }
    solver.set_configuration_q(reference)
    retargeter.set_execution_dt_sec(1.0)

    qpos, metadata = retargeter._limit_to_execution_step(
        reference_full_q=reference,
        candidate_full_q=candidate,
        target_activity=target_activity,
    )

    assert np.isclose(qpos[6], np.deg2rad(0.5))
    assert metadata["step_limited"] is True
    cap = metadata["inactive_side_hold_execution_cap"]
    assert cap["applied"] is True
    assert np.isclose(cap["sides"]["right"]["requested_max_abs_delta_deg"], 3.0)
    assert np.isclose(cap["sides"]["right"]["capped_max_abs_delta_deg"], 0.5)


def test_tracker_confirmed_inactive_side_freezes_below_static_error_thresholds() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    reference = np.zeros(12, dtype=float)
    candidate = reference.copy()
    candidate[:6] = np.deg2rad([0.4, -0.3, 0.2, -0.1, 0.4, -0.2])
    candidate[6:] = np.deg2rad([-0.4, 0.3, -0.2, 0.1, -0.4, 0.2])
    target_activity = {
        "enabled": True,
        "left": {
            "source": "tracker_motion_state_machine",
            "state": "INACTIVE",
            "inactive": True,
            "hold_weight": 1.0,
        },
        "right": {
            "source": "tracker_motion_state_machine",
            "state": "INACTIVE",
            "inactive": True,
            "hold_weight": 1.0,
        },
    }

    qpos, metadata = retargeter._limit_to_execution_step(
        reference_full_q=reference,
        candidate_full_q=candidate,
        target_activity=target_activity,
        current_targets={
            "left": _target_pose(x_m=0.029, rotation_deg=4.9),
            "right": _target_pose(x_m=0.029, rotation_deg=4.9),
        },
    )

    np.testing.assert_allclose(qpos, reference, atol=1e-12)
    cap = metadata["inactive_side_hold_execution_cap"]
    assert cap["applied"] is True
    for side in ("left", "right"):
        assert cap["sides"][side]["tracker_inactive_freeze"] is True
        assert cap["sides"][side]["effective_cap_deg"] == 0.0
        assert cap["sides"][side]["capped_max_abs_delta_deg"] == 0.0
        release = cap["sides"][side]["tracker_inactive_execution_release"]
        assert release["available"] is True
        assert release["velocity_release_scale"] == 0.0


def test_tracker_inactive_execution_release_linearly_adds_pose_error_weights() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    reference = np.zeros(12, dtype=float)
    candidate = reference.copy()
    candidate[6] = np.deg2rad(0.4)
    target_activity = {
        "enabled": True,
        "left": {"inactive": False, "hold_weight": 0.0},
        "right": {
            "source": "tracker_motion_state_machine",
            "state": "INACTIVE",
            "inactive": True,
            "hold_weight": 1.0,
        },
    }
    solver.set_configuration_q(reference)
    retargeter.set_execution_dt_sec(1.0)

    qpos, metadata = retargeter._limit_to_execution_step(
        reference_full_q=reference,
        candidate_full_q=candidate,
        target_activity=target_activity,
        current_targets={
            "left": _target_pose(),
            "right": _target_pose(x_m=0.0475, rotation_deg=11.25),
        },
    )

    # (4.75 - 3.0) / (10.0 - 3.0) == 0.25 and
    # (11.25 - 5.0) / (30.0 - 5.0) == 0.25, hence 50% release.
    assert np.isclose(qpos[6], np.deg2rad(0.2))
    release = metadata["inactive_side_hold_execution_cap"]["sides"]["right"][
        "tracker_inactive_execution_release"
    ]
    assert release["reference_qpos_source"] == "execution_reference_qcmd_or_qref"
    assert np.isclose(release["translation_weight"], 0.25)
    assert np.isclose(release["rotation_weight"], 0.25)
    assert np.isclose(release["velocity_release_scale"], 0.5)

    qpos, metadata = retargeter._limit_to_execution_step(
        reference_full_q=reference,
        candidate_full_q=candidate,
        target_activity=target_activity,
        current_targets={
            "left": _target_pose(),
            "right": _target_pose(x_m=0.065, rotation_deg=17.5),
        },
    )

    # Both normalized weights are 0.5, so their equal-weight sum releases the
    # inactive cap completely and preserves the normal execution-step limit.
    assert np.isclose(qpos[6], np.deg2rad(0.4))
    release = metadata["inactive_side_hold_execution_cap"]["sides"]["right"][
        "tracker_inactive_execution_release"
    ]
    assert np.isclose(release["translation_weight"], 0.5)
    assert np.isclose(release["rotation_weight"], 0.5)
    assert np.isclose(release["velocity_release_scale"], 1.0)
    assert metadata["inactive_side_hold_execution_cap"]["sides"]["right"]["applied"] is False


def test_candidate_local_step_uses_tracker_inactive_execution_release() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    reference = np.zeros(12, dtype=float)
    candidate = reference.copy()
    candidate[6] = np.deg2rad(0.4)
    target_activity = {
        "enabled": True,
        "left": {"inactive": False, "hold_weight": 0.0},
        "right": {
            "source": "tracker_motion_state_machine",
            "state": "INACTIVE",
            "inactive": True,
            "hold_weight": 1.0,
        },
    }
    solver.set_configuration_q(reference)
    retargeter.set_execution_dt_sec(1.0)

    qpos, metadata = retargeter._limit_full_joint_step(
        reference_full_q=reference,
        candidate_full_q=candidate,
        step_scale=1.0,
        target_activity=target_activity,
        current_targets={
            "left": _target_pose(),
            "right": _target_pose(x_m=0.0475, rotation_deg=11.25),
        },
    )

    assert np.isclose(qpos[6], np.deg2rad(0.2))
    release = metadata["inactive_side_hold_execution_cap"]["sides"]["right"][
        "tracker_inactive_execution_release"
    ]
    assert release["available"] is True
    assert np.isclose(release["velocity_release_scale"], 0.5)


def test_tracker_inactive_execution_release_is_opt_in_for_position_only() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    retargeter.config.inactive_side_hold_execution_release_enabled = False
    reference = np.zeros(12, dtype=float)
    candidate = reference.copy()
    candidate[6] = np.deg2rad(0.4)
    target_activity = {
        "enabled": True,
        "left": {"inactive": False, "hold_weight": 0.0},
        "right": {
            "source": "tracker_motion_state_machine",
            "state": "INACTIVE",
            "inactive": True,
            "hold_weight": 1.0,
        },
    }
    retargeter.set_execution_dt_sec(1.0)

    qpos, metadata = retargeter._limit_to_execution_step(
        reference_full_q=reference,
        candidate_full_q=candidate,
        target_activity=target_activity,
        current_targets={"left": _target_pose(), "right": _target_pose(x_m=0.10)},
    )

    # Generic callers (including force-control callers) retain the historical
    # confirmed-inactive freeze unless a position entrypoint opts in.
    assert np.isclose(qpos[6], 0.0)
    side = metadata["inactive_side_hold_execution_cap"]["sides"]["right"]
    assert side["tracker_inactive_execution_release_enabled"] is False
    assert side["tracker_inactive_execution_release"] is None
    assert side["tracker_inactive_freeze"] is True


def test_build_command_passes_targets_to_inactive_execution_release() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    candidate = np.zeros(12, dtype=float)
    candidate[6] = np.deg2rad(5.0)
    solver.solve = lambda *, targets: candidate.copy()  # type: ignore[attr-defined]
    inactive_right = EffectorCommand(
        effector_name="right_arm",
        side="right",
        command_type="cartesian_pose",
        value=_target_pose(x_m=0.10),
        frame_name="arm_base",
        metadata={
            "tracker_motion_intent": {
                "state": "INACTIVE",
                "inactive": True,
                "hold_weight": 1.0,
            }
        },
    )
    context = DualArmTargetContext(
        timestamp=0.0,
        left_target=_target_pose(),
        right_target=_target_pose(x_m=0.10),
        left_world_to_base=None,
        right_world_to_base=None,
        left_cartesian=None,
        right_cartesian=inactive_right,
        left_input_skipped=None,
        right_input_skipped=None,
    )

    command = retargeter.build_command_for_context(context)

    right = command.commands["right_arm"].metadata
    assert np.max(np.abs(command.commands["right_arm"].value)) > 0.0
    release = right["execution_step_metadata"]["inactive_side_hold_execution_cap"]["sides"][
        "right"
    ]["tracker_inactive_execution_release"]
    assert release["available"] is True
    assert "unavailable_reason" not in release
    assert release["velocity_release_scale"] == 1.0


def test_dual_arm_velocity_profile_and_explicit_execution_step_use_strictest_limit() -> None:
    solver = _FakeDualArmSolver()
    explicit_step = np.deg2rad(np.full(6, 0.25, dtype=float))
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=15.0,
            enforce_realistic_joint_step=True,
            ik_safety_profile=_profile(),
            joint_velocity_profile=JointVelocityLimitProfile(
                name="fast_profile",
                max_velocity_deg_s=(1000.0,) * 6,
                max_dt_sec=1.0,
            ),
            left_execution_max_joint_step_rad=explicit_step,
            right_execution_max_joint_step_rad=explicit_step,
        ),
        ik_solver=solver,
    )
    reference = np.zeros(12, dtype=float)
    candidate = reference.copy()
    candidate[0] = np.deg2rad(3.0)
    candidate[6] = -np.deg2rad(4.0)

    qpos, metadata = retargeter._limit_to_execution_step(
        reference_full_q=reference,
        candidate_full_q=candidate,
    )

    assert np.isclose(qpos[0], np.deg2rad(0.25))
    assert np.isclose(qpos[6], -np.deg2rad(0.25))
    assert metadata["step_limited"] is True
    assert metadata["execution_step_source"] == "joint_velocity_profile+explicit_execution_max_joint_step"
    assert np.allclose(metadata["left_max_step_deg"], np.full(6, 0.25))
    assert np.allclose(metadata["right_max_step_deg"], np.full(6, 0.25))


def test_candidate_local_clip_scale_does_not_raise_final_execution_step() -> None:
    solver = _FakeDualArmSolver()
    explicit_step = np.deg2rad(np.full(6, 0.25, dtype=float))
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=15.0,
            enforce_realistic_joint_step=True,
            ik_safety_profile=_profile(),
            joint_velocity_profile=JointVelocityLimitProfile(
                name="fast_profile",
                max_velocity_deg_s=(1000.0,) * 6,
                max_dt_sec=1.0,
            ),
            left_execution_max_joint_step_rad=explicit_step,
            right_execution_max_joint_step_rad=explicit_step,
            local_step_candidate_clip_scale=2.0,
        ),
        ik_solver=solver,
    )
    reference = np.zeros(12, dtype=float)
    candidate = reference.copy()
    candidate[0] = np.deg2rad(3.0)

    local_qpos, local_metadata = retargeter._limit_full_joint_step(
        reference_full_q=reference,
        candidate_full_q=candidate,
        step_scale=1.0,
    )
    final_qpos, final_metadata = retargeter._limit_to_execution_step(
        reference_full_q=reference,
        candidate_full_q=local_qpos,
    )

    assert np.isclose(local_qpos[0], np.deg2rad(0.5))
    assert local_metadata["candidate_local_step_clip_scale"] == pytest.approx(2.0)
    assert np.isclose(final_qpos[0], np.deg2rad(0.25))
    assert final_metadata["step_limited"] is True


def test_dual_arm_runtime_execution_step_update_limits_velocity_profile() -> None:
    solver = _FakeDualArmSolver()
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=15.0,
            enforce_realistic_joint_step=True,
            ik_safety_profile=_profile(),
            joint_velocity_profile=JointVelocityLimitProfile(
                name="fast_profile",
                max_velocity_deg_s=(1000.0,) * 6,
                max_dt_sec=1.0,
            ),
        ),
        ik_solver=solver,
    )
    runtime_step = np.deg2rad(np.full(6, 0.2, dtype=float))
    retargeter.set_execution_max_joint_step(
        left_step_rad=runtime_step,
        right_step_rad=runtime_step,
    )
    reference = np.zeros(12, dtype=float)
    candidate = reference.copy()
    candidate[0] = np.deg2rad(3.0)
    candidate[6] = -np.deg2rad(4.0)

    qpos, metadata = retargeter._limit_to_execution_step(
        reference_full_q=reference,
        candidate_full_q=candidate,
    )

    assert np.isclose(qpos[0], np.deg2rad(0.2))
    assert np.isclose(qpos[6], -np.deg2rad(0.2))
    assert metadata["step_limited"] is True
    assert metadata["execution_step_source"] == "joint_velocity_profile+explicit_execution_max_joint_step"
    assert np.allclose(metadata["left_max_step_deg"], np.full(6, 0.2))
    assert np.allclose(metadata["right_max_step_deg"], np.full(6, 0.2))


def test_dual_arm_output_qpos_smoothing_runs_before_action_fk() -> None:
    solver = _FakeDualArmSolver()
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=15.0,
            enforce_realistic_joint_step=True,
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
            output_qpos_smoothing_alpha=0.5,
        ),
        ik_solver=solver,
    )
    raw_left = np.deg2rad(np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    raw_right = np.deg2rad(np.array([-8.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    raw_full = retargeter._embed_arm_qpos(
        current_full_q=solver.current_q,
        left_qpos=raw_left,
        right_qpos=raw_right,
    )

    first_full, first_left, first_right, first_metadata = retargeter._smooth_output_qpos_before_action(
        active_q=raw_full,
        left_qpos=raw_left,
        right_qpos=raw_right,
    )
    second_full, second_left, second_right, second_metadata = retargeter._smooth_output_qpos_before_action(
        active_q=np.zeros_like(raw_full),
        left_qpos=np.zeros(6, dtype=float),
        right_qpos=np.zeros(6, dtype=float),
    )

    np.testing.assert_allclose(first_left, raw_left)
    np.testing.assert_allclose(first_right, raw_right)
    assert first_metadata["initialized"] is True
    assert first_metadata["applied"] is False
    np.testing.assert_allclose(first_full, raw_full)
    np.testing.assert_allclose(second_left[0], np.deg2rad(5.0))
    np.testing.assert_allclose(second_right[0], -np.deg2rad(4.0))
    assert second_metadata["applied"] is True
    assert second_metadata["stage"] == "post_ik_pre_action_fk"
    assert second_metadata["downstream_command_stage"] == "interpolation_only"
    assert np.isclose(second_metadata["left"]["output_target_joint_deg"][0], 5.0)
    assert np.isclose(second_metadata["right"]["output_target_joint_deg"][0], -4.0)


def test_dual_arm_output_qpos_smoothing_adapts_to_valid_tracker_speed() -> None:
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=60.0,
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
            output_qpos_smoothing_alpha=0.18,
            output_qpos_smoothing_adaptive_enabled=True,
            output_qpos_smoothing_fast_alpha=0.7,
        ),
        ik_solver=_FakeDualArmSolver(),
    )
    activity = {
        "left": {
            "evidence": {
                "quality_valid": True,
                "linear_speed_p90_m_s": 0.30,
                "angular_speed_p90_rad_s": 0.2,
            }
        }
    }

    alpha, metadata = retargeter._effective_output_qpos_smoothing_alpha(activity)

    assert 0.18 < alpha["left"] <= 0.7
    assert alpha["right"] == pytest.approx(0.18)
    assert metadata["selection"] == "per_side_tracker_speed"
    assert metadata["left"]["reason"] == "tracker_speed"
    assert metadata["left"]["requested_alpha"] == 0.7
    assert metadata["left"]["transition"] == "attack"
    assert metadata["right"]["reason"] == "invalid_or_missing_speed"


def test_dual_arm_output_qpos_smoothing_uses_motion_window_validity() -> None:
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=60.0,
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
            output_qpos_smoothing_alpha=0.18,
            output_qpos_smoothing_adaptive_enabled=True,
            output_qpos_smoothing_fast_alpha=0.7,
        ),
        ik_solver=_FakeDualArmSolver(),
    )

    alpha, metadata = retargeter._effective_output_qpos_smoothing_alpha(
        {
            "right": {
                "evidence": {
                    "linear_speed_p90_m_s": 0.30,
                    "angular_speed_p90_rad_s": 0.2,
                },
                "motion_window": {"quality_valid": True},
            }
        }
    )

    assert alpha["left"] == pytest.approx(0.18)
    assert 0.18 < alpha["right"] <= 0.7
    assert metadata["right"]["reason"] == "tracker_speed"
    assert metadata["right"]["requested_alpha"] == 0.7


def test_dual_arm_output_qpos_smoothing_ignores_invalid_tracker_speed() -> None:
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=60.0,
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
            output_qpos_smoothing_alpha=0.18,
            output_qpos_smoothing_adaptive_enabled=True,
            output_qpos_smoothing_fast_alpha=0.7,
        ),
        ik_solver=_FakeDualArmSolver(),
    )

    alpha, metadata = retargeter._effective_output_qpos_smoothing_alpha(
        {"left": {"evidence": {"quality_valid": False, "linear_speed_p90_m_s": 1.0}}}
    )

    assert alpha == {"left": pytest.approx(0.18), "right": pytest.approx(0.18)}
    assert metadata["left"]["reason"] == "invalid_or_missing_speed"


def test_dual_arm_output_qpos_smoothing_applies_distinct_side_alphas() -> None:
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=60.0,
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
            output_qpos_smoothing_alpha=0.14,
            output_qpos_smoothing_adaptive_enabled=True,
            output_qpos_smoothing_fast_alpha=0.45,
        ),
        ik_solver=_FakeDualArmSolver(),
    )
    retargeter._last_output_qpos_smoothing_full_q = np.zeros(12, dtype=float)
    activity = {
        "left": {
            "evidence": {
                "quality_valid": True,
                "linear_speed_p90_m_s": 0.30,
                "angular_speed_p90_rad_s": 0.2,
            }
        },
        "right": {"evidence": {"quality_valid": False}},
    }
    _, left, right, metadata = retargeter._smooth_output_qpos_before_action(
        active_q=np.ones(12, dtype=float),
        left_qpos=np.ones(6, dtype=float),
        right_qpos=np.ones(6, dtype=float),
        target_activity=activity,
    )

    assert metadata["left"]["alpha"] > metadata["right"]["alpha"]
    assert np.all(left > right)


def test_dual_arm_current_qpos_sync_can_preserve_output_qpos_smoothing_state() -> None:
    solver = _FakeDualArmSolver()
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=15.0,
            enforce_realistic_joint_step=True,
            ik_safety_profile=_profile(),
            joint_velocity_profile=_velocity(),
            output_qpos_smoothing_alpha=0.5,
        ),
        ik_solver=solver,
    )
    raw_left = np.deg2rad(np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    raw_right = np.deg2rad(np.array([-8.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    raw_full = retargeter._embed_arm_qpos(
        current_full_q=solver.current_q,
        left_qpos=raw_left,
        right_qpos=raw_right,
    )

    retargeter._smooth_output_qpos_before_action(
        active_q=raw_full,
        left_qpos=raw_left,
        right_qpos=raw_right,
    )
    retargeter.set_current_qpos_pair(
        left_qpos=raw_left,
        right_qpos=raw_right,
        reset_tracking=False,
    )
    _, preserved_left, preserved_right, preserved_metadata = (
        retargeter._smooth_output_qpos_before_action(
            active_q=np.zeros_like(raw_full),
            left_qpos=np.zeros(6, dtype=float),
            right_qpos=np.zeros(6, dtype=float),
        )
    )

    assert preserved_metadata["initialized"] is False
    assert preserved_metadata["applied"] is True
    np.testing.assert_allclose(preserved_left[0], np.deg2rad(5.0))
    np.testing.assert_allclose(preserved_right[0], -np.deg2rad(4.0))

    retargeter.set_current_qpos_pair(
        left_qpos=raw_left,
        right_qpos=raw_right,
        reset_tracking=True,
    )
    _, reset_left, reset_right, reset_metadata = retargeter._smooth_output_qpos_before_action(
        active_q=np.zeros_like(raw_full),
        left_qpos=np.zeros(6, dtype=float),
        right_qpos=np.zeros(6, dtype=float),
    )

    assert reset_metadata["initialized"] is True
    assert reset_metadata["applied"] is False
    np.testing.assert_allclose(reset_left, np.zeros(6, dtype=float))
    np.testing.assert_allclose(reset_right, np.zeros(6, dtype=float))


def test_dual_arm_explicit_execution_step_is_fallback_without_velocity_profile() -> None:
    solver = _FakeDualArmSolver()
    explicit_step = np.deg2rad(np.full(6, 0.25, dtype=float))
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            solver_frequency=15.0,
            enforce_realistic_joint_step=True,
            ik_safety_profile=_profile(),
            left_execution_max_joint_step_rad=explicit_step,
            right_execution_max_joint_step_rad=explicit_step,
        ),
        ik_solver=solver,
    )
    retargeter.joint_velocity_profile = None
    retargeter.config.joint_velocity_profile = None
    retargeter._full_execution_joint_step = retargeter._make_full_execution_joint_step(
        solver.current_q
    )
    reference = np.zeros(12, dtype=float)
    candidate = reference.copy()
    candidate[0] = np.deg2rad(3.0)
    candidate[6] = -np.deg2rad(4.0)

    qpos, metadata = retargeter._limit_to_execution_step(
        reference_full_q=reference,
        candidate_full_q=candidate,
    )

    assert np.isclose(qpos[0], np.deg2rad(0.25))
    assert np.isclose(qpos[6], -np.deg2rad(0.25))
    assert metadata["step_limited"] is True
    assert metadata["execution_step_source"] == "explicit_execution_max_joint_step"


def test_dual_arm_activity_hold_allows_unmoved_side_to_reduce_existing_error() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver, loose_dynamic_worsening=True)
    reference = np.zeros(12, dtype=float)
    reference[0] = -0.050
    raw = reference.copy()
    raw[0] = -0.020
    raw[6] = 0.030
    solver.set_configuration_q(reference)
    retargeter.set_execution_dt_sec(1.0)
    targets = retargeter.current_world_targets()
    targets["left"] = np.eye(4, dtype=float)
    retargeter._last_target_poses = {
        "left": targets["left"].copy(),
        "right": targets["right"].copy(),
    }
    targets["right"] = targets["right"].copy()
    targets["right"][0, 3] = 0.060

    qpos, name, _score, _min_distance, metadata = retargeter._select_safe_tracking_candidate(
        raw_full_q=raw,
        reference_full_q=reference,
        current_targets=targets,
        profile=None,
    )

    assert name == "original"
    assert qpos[0] > reference[0]
    selected = metadata["selected_metadata"]
    left_hold = selected["inactive_side_hold"]["sides"]["left"]
    assert left_hold["hold_weight"] == 1.0
    assert left_hold["unexplained_position_m"] == 0.0
    assert selected["reject_reason"] is None


def test_dual_arm_rm65_posture_escape_allows_right_j23_bad_to_good_branch() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver, rm65_posture=True, rm65_local_escape=False)
    reference = np.zeros(12, dtype=float)
    reference[1] = np.deg2rad(20.0)
    reference[2] = np.deg2rad(120.0)
    reference[7] = np.deg2rad(100.0)
    reference[8] = np.deg2rad(-90.0)
    raw = reference.copy()
    raw[7] = np.deg2rad(20.0)
    raw[8] = np.deg2rad(130.0)
    solver.set_configuration_q(reference)
    targets = retargeter.current_world_targets()

    qpos, name, _score, _min_distance, metadata = retargeter._select_safe_tracking_candidate(
        raw_full_q=raw,
        reference_full_q=reference,
        current_targets=targets,
        profile=None,
    )

    assert name == "original"
    assert qpos[8] > reference[8]
    selected = metadata["selected_metadata"]
    assert selected["right_hard_branch_escape_allowed"] is True
    assert selected["posture_preference"]["right"]["candidate"]["score_deg"] < selected["posture_preference"]["right"]["reference"]["score_deg"]


def test_dual_arm_rm65_wrist_unload_candidate_is_collision_checked() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver, rm65_posture=True)
    reference = np.zeros(12, dtype=float)
    reference[3] = np.deg2rad(177.0)
    reference[4] = np.deg2rad(88.0)
    reference[5] = np.deg2rad(-170.0)
    solver.set_configuration_q(reference)
    targets = retargeter.current_world_targets()
    targets["left"] = targets["left"].copy()
    targets["left"][0, 3] = -0.5

    qpos, name, _score, _min_distance, metadata = retargeter._select_safe_tracking_candidate(
        raw_full_q=reference.copy(),
        reference_full_q=reference.copy(),
        current_targets=targets,
        profile=None,
    )

    assert name.startswith("wrist_unload_left_")
    assert not np.allclose(qpos, reference)
    assert metadata["wrist_unload"]["status"] == "active"
    assert metadata["selected_source"] == "wrist_unload"


def test_dual_arm_prefers_tracking_candidate_over_wrist_unload_auxiliary_recovery() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver, rm65_posture=True)
    reference = np.zeros(12, dtype=float)
    reference[3] = np.deg2rad(177.0)
    reference[4] = np.deg2rad(88.0)
    reference[5] = np.deg2rad(-170.0)
    solver.set_configuration_q(reference)
    targets = retargeter.current_world_targets()

    qpos, name, _score, _min_distance, metadata = retargeter._select_safe_tracking_candidate(
        raw_full_q=reference.copy(),
        reference_full_q=reference.copy(),
        current_targets=targets,
        profile=None,
    )

    assert name == "original"
    assert np.allclose(qpos, reference)
    assert metadata["selected_source"] == "multitask"
    assert metadata["candidate_eval_mode"] == "base_only"
    assert metadata["wrist_unload"]["status"] == "deferred"


def test_dual_arm_jump_gate_warns_but_does_not_hold_for_wrist_jump() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    reference = np.zeros(12, dtype=float)
    candidate = reference.copy()
    candidate[3] = np.deg2rad(55.0)
    targets = retargeter.current_world_targets()
    retargeter._last_target_poses = {"left": targets["left"].copy(), "right": targets["right"].copy()}

    gate = retargeter._evaluate_jump_gate(
        reference_full_q=reference,
        candidate_full_q=candidate,
        current_targets=targets,
    )

    assert gate["hold"] is False
    assert gate["status"] == "warning"
    assert gate["reason"] == "small_pose_large_q_jump_warning_only:left"


def test_final_qpos_pair_collision_check_is_read_only_and_fail_closed() -> None:
    solver = _FakeDualArmSolver()
    retargeter = _retargeter(solver)
    initial_q = solver.current_q.copy()

    safe = retargeter.check_qpos_pair_collision(
        left_qpos=np.zeros(6),
        right_qpos=np.zeros(6),
    )
    assert safe.safe
    assert safe.collision_pair_count == len(retargeter.profile.collision_pairs)
    assert safe.min_distance_m == 0.25
    np.testing.assert_allclose(solver.current_q, initial_q)

    solver.min_collision_distance = -0.01
    unsafe = retargeter.check_qpos_pair_collision(
        left_qpos=np.zeros(6),
        right_qpos=np.zeros(6),
    )
    assert not unsafe.safe
    assert unsafe.reason == "collision_margin_low"
    np.testing.assert_allclose(solver.current_q, initial_q)
