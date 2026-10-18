from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from robot_motion.rm65 import (
    ARM_LINK6_FRAME,
    DEFAULT_COLLISION_SCENE,
    HAND_BASE_FRAME,
    RM65DualArmCollisionIKConfig,
    RM65SingleArmIKConfig,
    build_rm65_dual_arm_collision_ik_solver,
    build_rm65_single_arm_ik_solver,
    canonical_rm65_controlled_frame,
    rm65_single_arm_candidate_frames,
    solve_rm65_single_arm_ik,
)
from robot_motion.rm65.constants import ARM_ENDPOINT_FRAME, BARE_ENDPOINT_FRAME, TOOL_TCP_FRAME


class _Objective:
    dual_arm_position_cost = 11.0
    dual_arm_orientation_cost = 7.0


class _IkTuning:
    def objective_weights(self):
        return _Objective()


class _FakeSingleArmSolver:
    calls: list[dict[str, object]] = []

    def __init__(self, *, urdf_path, eef_name, ik_tuning):
        self.calls.append(
            {
                "urdf_path": urdf_path,
                "eef_name": eef_name,
                "ik_tuning": ik_tuning,
            }
        )
        if eef_name == ARM_LINK6_FRAME:
            raise ValueError("missing arm_link6")
        self.urdf_path = urdf_path
        self.eef_name = eef_name
        self.ik_tuning = ik_tuning


class _FakeDualArmSolver:
    calls: list[dict[str, object]] = []

    def __init__(self, **kwargs):
        self.calls.append(dict(kwargs))
        self.tasks: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def add_frame_task(self, *args, **kwargs) -> None:
        self.tasks.append(("frame", args, dict(kwargs)))

    def add_posture_task(self, *args, **kwargs) -> None:
        self.tasks.append(("posture", args, dict(kwargs)))


class _FakeSingleArmSolveMethods:
    def __init__(self):
        self.calls: list[tuple[str, dict[str, object]]] = []

    def solve_iterative(self, **kwargs):
        self.calls.append(("iterative", dict(kwargs)))
        return np.arange(6, dtype=float)

    def solve_step(self, **kwargs):
        self.calls.append(("step", dict(kwargs)))
        return np.arange(6, dtype=float) + 10.0


def test_rm65_single_arm_frame_candidates_are_shared() -> None:
    assert canonical_rm65_controlled_frame(BARE_ENDPOINT_FRAME) == ARM_ENDPOINT_FRAME
    assert rm65_single_arm_candidate_frames(BARE_ENDPOINT_FRAME) == (ARM_LINK6_FRAME, HAND_BASE_FRAME)
    assert rm65_single_arm_candidate_frames(TOOL_TCP_FRAME) == (HAND_BASE_FRAME, ARM_LINK6_FRAME)
    assert rm65_single_arm_candidate_frames("custom_tcp") == ("custom_tcp", HAND_BASE_FRAME, ARM_LINK6_FRAME)


def test_build_rm65_single_arm_ik_solver_tries_shared_fallback_frames(monkeypatch) -> None:
    import robot_motion.rm65.single_arm_ik as single_ik_module

    _FakeSingleArmSolver.calls.clear()
    tuning = object()
    monkeypatch.setattr(single_ik_module, "SingleArmIKSolver", _FakeSingleArmSolver)

    solver = build_rm65_single_arm_ik_solver(
        RM65SingleArmIKConfig(
            arm_urdf_path="rm65_arm.urdf",
            controlled_frame=BARE_ENDPOINT_FRAME,
            ik_tuning=tuning,
        )
    )

    assert solver.eef_name == HAND_BASE_FRAME
    assert [call["eef_name"] for call in _FakeSingleArmSolver.calls] == [ARM_LINK6_FRAME, HAND_BASE_FRAME]
    assert _FakeSingleArmSolver.calls[-1]["ik_tuning"] is tuning


def test_solve_rm65_single_arm_ik_uses_shared_virtual_preview_defaults() -> None:
    solver = _FakeSingleArmSolveMethods()
    target_pose = np.eye(4, dtype=float)
    qpos = np.zeros(6, dtype=float)

    solved = solve_rm65_single_arm_ik(
        solver,
        target_pose=target_pose,
        current_qpos=qpos,
        fast_iterative=True,
        dry_run_max_iterations=12,
        dry_run_max_joint_step_rad=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        dry_run_damping=0.07,
    )

    assert solved.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    name, kwargs = solver.calls[0]
    assert name == "iterative"
    np.testing.assert_allclose(kwargs["target_pose"], target_pose)
    np.testing.assert_allclose(kwargs["q_init"], qpos)
    assert kwargs["max_iterations"] == 12
    np.testing.assert_allclose(kwargs["max_joint_step"], np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    assert kwargs["damping"] == 0.07
    assert kwargs["tolerance"] == 1e-5


def test_solve_rm65_single_arm_ik_uses_bounded_step_for_normal_commands() -> None:
    solver = _FakeSingleArmSolveMethods()
    target_pose = np.eye(4, dtype=float)
    qpos = np.zeros(6, dtype=float)

    solved = solve_rm65_single_arm_ik(
        solver,
        target_pose=target_pose,
        current_qpos=qpos,
        fast_iterative=False,
        max_joint_step_rad=(0.01, 0.02, 0.03, 0.04, 0.05, 0.06),
    )

    assert solved.tolist() == [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    name, kwargs = solver.calls[0]
    assert name == "step"
    np.testing.assert_allclose(kwargs["target_pose"], target_pose)
    np.testing.assert_allclose(kwargs["current_qpos"], qpos)
    np.testing.assert_allclose(kwargs["max_joint_step"], np.asarray([0.01, 0.02, 0.03, 0.04, 0.05, 0.06]))


def test_rm65_dual_arm_collision_ik_config_matches_real_defaults() -> None:
    config = RM65DualArmCollisionIKConfig()

    assert config.scene_profile == DEFAULT_COLLISION_SCENE
    assert config.solver_frequency == 30.0
    assert config.damping == 0.01
    assert config.posture_cost == 1e-1
    assert config.collision_min_distance_m == 0.03
    assert config.collision_distance_epsilon_m == 1e-4
    assert config.collision_barrier_gain == 10.0
    assert config.collision_safe_displacement_gain == 0.001


def test_build_rm65_dual_arm_collision_ik_solver_adds_shared_tasks(monkeypatch) -> None:
    import robot_motion.rm65.dual_arm_ik as multitask_module
    import robot_motion.rm65.ik as rm65_ik_module

    scene = SimpleNamespace(
        name="dual_arm_hand_proxy_open",
        urdf_path="dual.urdf",
        package_dirs=("assets",),
        collision_pairs=(("left_hand_proxy_0", "right_hand_proxy_0"),),
        fixed_scene_box_overrides=(),
        left_frame_name="arm_link6",
        right_frame_name="right_arm_link6",
    )
    _FakeDualArmSolver.calls.clear()
    monkeypatch.setattr(multitask_module, "MultiTaskIKSolver", _FakeDualArmSolver)
    monkeypatch.setattr(rm65_ik_module, "build_rm65_dual_arm_scene_profile", lambda name: scene)

    bundle = build_rm65_dual_arm_collision_ik_solver(
        RM65DualArmCollisionIKConfig(
            ik_tuning=_IkTuning(),
            solver_frequency=42.0,
            damping=0.03,
            debug_profile=True,
        )
    )

    assert bundle.scene_profile is scene
    assert bundle.collision_pair_count == 1
    call = _FakeDualArmSolver.calls[0]
    assert call["robot_urdf_path"] == "dual.urdf"
    assert call["robot_package_dirs"] == ["assets"]
    assert call["robot_collision_pair_names"] == [("left_hand_proxy_0", "right_hand_proxy_0")]
    assert call["frequency"] == 42.0
    assert call["damping"] == 0.03
    assert "sleep" not in call
    assert call["debug_profile"] is True
    assert [task[0] for task in bundle.solver.tasks] == ["frame", "frame", "posture"]
    assert bundle.solver.tasks[0][2]["position_cost"] == 11.0
    assert bundle.solver.tasks[0][2]["orientation_cost"] == 7.0
    assert np.isclose(bundle.solver.tasks[-1][2]["cost"], 1e-1)
