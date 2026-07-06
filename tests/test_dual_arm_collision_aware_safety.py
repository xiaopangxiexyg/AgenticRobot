from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from robot_motion.joint_limits import JointVelocityLimitProfile
from robot_motion.profiles import IKSafetyProfile
from robot_motion.types import PoseTolerance
import teleop_runtime.retargeters.dual_arm_collision_aware as dual_module
from teleop_runtime.retargeters.dual_arm_collision_aware import (
    DualArmCollisionAwareRetargeter,
    DualArmCollisionAwareRetargeterConfig,
)


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
        return 0.25


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


def _retargeter(
    solver: _FakeDualArmSolver,
    *,
    rm65_posture: bool = False,
    rm65_local_escape: bool = True,
    component_progress: bool = False,
    loose_dynamic_worsening: bool = False,
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
