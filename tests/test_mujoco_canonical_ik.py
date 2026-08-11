from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robot_motion.rm65 import RM65SingleArmIKConfig, build_rm65_single_arm_ik_solver
from robot_motion.rm65.constants import ARM_ENDPOINT_FRAME, TOOL_TCP_FRAME
from robot_motion.runtime import EffectorCommand
from sim.contracts import SimulationContract
from sim.mujoco.backend import MuJoCoArmBackend, MuJoCoArmBackendConfig
from sim.mujoco.config import ArmSpec
from sim.mujoco.controllers import PoseIKController
from sim.mujoco.ik import PoseTarget, RM65CanonicalIK


class _FakeScene:
    def __init__(self) -> None:
        self.arm_specs = {
            side: ArmSpec(
                side=side,
                joint_names=tuple(f"{side}_joint{index}" for index in range(1, 7)),
                controlled_body=f"{side}_arm_link6",
                effector_name=f"{side}_arm",
            )
            for side in ("left", "right")
        }
        self.qpos = {
            "left": np.zeros(6, dtype=float),
            "right": np.ones(6, dtype=float),
        }
        self.group_writes: list[dict[str, np.ndarray]] = []

    def get_arm_qpos(self, side: str) -> np.ndarray:
        return self.qpos[side].copy()

    def arm_qpos_by_side(self) -> dict[str, list[float]]:
        return {side: values.tolist() for side, values in self.qpos.items()}

    def set_arm_qpos(self, side: str, qpos) -> None:
        self.qpos[side] = np.asarray(qpos, dtype=float).reshape(6).copy()

    def set_qpos_by_side(self, qpos_by_side) -> None:
        normalized = {
            side: np.asarray(values, dtype=float).reshape(6).copy()
            for side, values in qpos_by_side.items()
        }
        self.group_writes.append(normalized)
        self.qpos.update(normalized)

    def has_body(self, body_name: str) -> bool:
        return body_name in {
            "left_arm_base_link",
            "right_arm_base_link",
            "left_arm_link6",
            "right_arm_link6",
        }

    def body_pose(self, body_name: str) -> np.ndarray:
        pose = np.eye(4, dtype=float)
        if body_name.endswith("arm_base_link"):
            pose[0, 3] = 1.0
        elif body_name.endswith("arm_link6"):
            pose[0, 3] = 1.3
        else:
            raise KeyError(body_name)
        return pose


class _FakePlanner:
    def __init__(self, *, hold: bool = False) -> None:
        self.hold = bool(hold)
        self.reset_calls: list[dict[str, np.ndarray]] = []
        self.plan_calls: list[tuple[dict[str, np.ndarray], dict[str, object]]] = []

    def reset(self, qpos_by_side):
        self.reset_calls.append(
            {
                side: np.asarray(values, dtype=float).copy()
                for side, values in qpos_by_side.items()
            }
        )
        return {"initialized": True}

    def plan_pose_reference(self, pose_by_side, **kwargs):
        self.plan_calls.append(
            (
                {side: np.asarray(pose, dtype=float).copy() for side, pose in pose_by_side.items()},
                dict(kwargs),
            )
        )
        identity = np.eye(4, dtype=float)
        return SimpleNamespace(
            valid=True,
            hold=self.hold,
            status="hold_last_safe" if self.hold else "tracking",
            q_ref_by_side={
                "left": np.full(6, 0.25, dtype=float),
                "right": np.full(6, -0.25, dtype=float),
            },
            target_endpoint_pose_world_by_side={
                side: identity.copy() for side in pose_by_side
            },
            realized_endpoint_pose_world_by_side={
                side: identity.copy() for side in pose_by_side
            },
            modification_reasons=("hold_last_safe",) if self.hold else (),
            metadata={"planner": "rm65_coupled_position_collision"},
        )


def test_canonical_ik_resets_from_both_arms_and_is_pure_wrt_scene() -> None:
    scene = _FakeScene()
    planner = _FakePlanner()
    ik = RM65CanonicalIK(scene, planner)
    before = {side: values.copy() for side, values in scene.qpos.items()}

    result = ik.solve(
        {
            "right": PoseTarget(
                side="right",
                pose=np.eye(4, dtype=float),
                controlled_frame=TOOL_TCP_FRAME,
            )
        }
    )

    assert set(planner.reset_calls[0]) == {"left", "right"}
    np.testing.assert_allclose(planner.reset_calls[0]["right"], np.ones(6))
    pose_by_side, kwargs = planner.plan_calls[0]
    assert set(pose_by_side) == {"right"}
    assert kwargs["target_reference_frame"] == "arm_base"
    assert kwargs["target_controlled_frame"] == TOOL_TCP_FRAME
    assert result.success is True
    assert result.solver == "robot_motion.rm65.position_collision"
    assert scene.group_writes == []
    for side in ("left", "right"):
        np.testing.assert_allclose(scene.qpos[side], before[side])


def test_pose_controller_applies_accepted_pair_with_one_group_write() -> None:
    scene = _FakeScene()
    controller = PoseIKController(scene, RM65CanonicalIK(scene, _FakePlanner()))

    result = controller.apply({"right": np.eye(4, dtype=float)})

    assert result.success is True
    assert len(scene.group_writes) == 1
    assert set(scene.group_writes[0]) == {"left", "right"}
    np.testing.assert_allclose(scene.qpos["left"], np.full(6, 0.25))
    np.testing.assert_allclose(scene.qpos["right"], np.full(6, -0.25))


def test_pose_controller_hold_does_not_change_mujoco_qpos() -> None:
    scene = _FakeScene()
    before = {side: values.copy() for side, values in scene.qpos.items()}
    controller = PoseIKController(scene, RM65CanonicalIK(scene, _FakePlanner(hold=True)))

    result = controller.apply({"left": np.eye(4, dtype=float)})

    assert result.success is False
    assert result.hold is True
    assert scene.group_writes == []
    for side in ("left", "right"):
        np.testing.assert_allclose(scene.qpos[side], before[side])


def test_canonical_group_call_rejects_mixed_controlled_frames() -> None:
    scene = _FakeScene()
    ik = RM65CanonicalIK(scene, _FakePlanner())

    with pytest.raises(ValueError, match="same controlled frame"):
        ik.solve(
            {
                "left": PoseTarget("left", np.eye(4), TOOL_TCP_FRAME),
                "right": PoseTarget("right", np.eye(4), ARM_ENDPOINT_FRAME),
            }
        )


def test_canonical_ik_refuses_to_guess_missing_tool_profile() -> None:
    with pytest.raises(ValueError, match="explicit left/right tool profiles"):
        RM65CanonicalIK.from_tool_profiles(
            _FakeScene(),
            {"right": "position_endpoint"},
        )


def test_mujoco_arm_backend_is_joint_sink_and_rejects_per_arm_cartesian_ik() -> None:
    scene = _FakeScene()
    backend = MuJoCoArmBackend(scene, MuJoCoArmBackendConfig(side="right"))

    backend.send_command(
        EffectorCommand(
            effector_name="right_arm",
            side="right",
            command_type="joint_vector",
            value=np.full(6, 0.1),
        )
    )
    np.testing.assert_allclose(scene.qpos["right"], np.full(6, 0.1))

    with pytest.raises(ValueError, match="coupled robot_motion.rm65 canonical planner"):
        backend.send_command(
            EffectorCommand(
                effector_name="right_arm",
                side="right",
                command_type="cartesian_pose",
                value=np.eye(4),
            )
        )

    with pytest.raises(ValueError, match="finite"):
        backend.send_command(
            EffectorCommand(
                effector_name="right_arm",
                side="right",
                command_type="joint_vector",
                value=[0.0, 0.0, 0.0, 0.0, 0.0, np.nan],
            )
        )
    with pytest.raises(ValueError, match="canonical RM65 joint limits"):
        backend.send_command(
            EffectorCommand(
                effector_name="right_arm",
                side="right",
                command_type="joint_vector",
                value=[4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            )
        )

    source = inspect.getsource(__import__("sim.mujoco.backend", fromlist=["*"]))
    assert "DampedLeastSquaresIK" not in source
    assert "mj_jac" not in source


def test_mujoco_backend_does_not_label_endpoint_as_tcp_without_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sim.mujoco.backend as backend_module

    scene = _FakeScene()
    without_profile = MuJoCoArmBackend(scene, MuJoCoArmBackendConfig(side="right"))
    state = without_profile.get_state()
    assert state["current_tcp_pose"] is None
    assert state["tcp_feedback_available"] is False
    np.testing.assert_allclose(state["current_endpoint_pose"][:3, 3], [0.3, 0.0, 0.0])

    endpoint_to_tool = np.eye(4, dtype=float)
    endpoint_to_tool[2, 3] = 0.14
    monkeypatch.setattr(backend_module, "load_rm65_tool_profile", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        backend_module,
        "endpoint_to_tool_transform",
        lambda _profile: endpoint_to_tool.copy(),
    )
    monkeypatch.setattr(
        backend_module,
        "tool_tcp_pose_flange_m_rad",
        lambda _profile: np.asarray([0.0, 0.0, 0.14, 0.0, 0.0, 0.0]),
    )
    with_profile = MuJoCoArmBackend(
        scene,
        MuJoCoArmBackendConfig(side="right", tool_profile="explicit_right_tool"),
    )
    profiled_state = with_profile.get_state()
    assert profiled_state["tcp_feedback_available"] is True
    np.testing.assert_allclose(
        profiled_state["current_endpoint_pose"][:3, 3],
        [0.3, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        profiled_state["current_tcp_pose"][:3, 3],
        [0.3, 0.0, 0.14],
    )


@pytest.mark.optional
def test_rm65_arm_endpoint_fk_matches_mujoco_for_both_sides() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("pinocchio")
    contract = SimulationContract.load(
        Path("sim/configs/rm65_bimanual_physics_v1.json")
    )
    from sim.mujoco.scene import MujocoScene

    scene = MujocoScene.from_contract(contract)
    from robot_motion.rm65 import DEFAULT_LEFT_ARM_QPOS, DEFAULT_RIGHT_ARM_QPOS

    qpos_by_side = {
        "left": DEFAULT_LEFT_ARM_QPOS.copy(),
        "right": DEFAULT_RIGHT_ARM_QPOS.copy(),
    }
    scene.set_qpos_by_side(qpos_by_side)
    canonical_fk = build_rm65_single_arm_ik_solver(
        RM65SingleArmIKConfig(controlled_frame=ARM_ENDPOINT_FRAME)
    )

    for side, qpos in qpos_by_side.items():
        world_from_base = scene.body_pose(f"{side}_arm_base_link")
        world_from_endpoint = scene.body_pose(f"{side}_arm_link6")
        mujoco_base_from_endpoint = np.linalg.inv(world_from_base) @ world_from_endpoint
        robot_motion_base_from_endpoint = canonical_fk.compute_pose(qpos)
        np.testing.assert_allclose(
            mujoco_base_from_endpoint,
            robot_motion_base_from_endpoint,
            atol=2e-9,
            rtol=0.0,
        )
