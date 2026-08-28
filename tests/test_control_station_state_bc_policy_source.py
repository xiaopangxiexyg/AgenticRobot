from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from control_station.sim_hil_profile import state_bc_policy_source as source_module
from control_station.sim_hil_profile.state_bc_policy_source import (
    MujocoStateBCPolicySourceProfile,
    build_mujoco_state_bc_policy_source,
    preflight_mujoco_state_bc_policy_source,
)
from experiments.sim_learning.state_bc.contracts import StateBCArtifactMetadata
from experiments.sim_learning.state_bc.model import fit_linear_state_bc


def _artifact(tmp_path):
    states = np.zeros((4, 16), dtype=np.float32)
    actions = np.zeros((4, 16), dtype=np.float32)
    actions[:, 0:3] = (0.20, 0.0, 0.30)
    actions[:, 3:7] = (0.0, 0.0, 0.0, 1.0)
    actions[:, 7:10] = (0.40, 0.0, 0.30)
    actions[:, 10:14] = (0.0, 0.0, 0.0, 1.0)
    actions[:, 14:16] = 0.5
    model = fit_linear_state_bc(
        states,
        actions,
        artifact_metadata=StateBCArtifactMetadata(
            selector_id="sim-hil-fixture",
            train_episode_count=1,
            test_episode_count=0,
            train_frame_count=len(states),
            test_frame_count=0,
        ),
    )
    return model.save(tmp_path / "state_bc_fixture.npz")


class _Scene:
    arm_specs = {
        "left": SimpleNamespace(controlled_body="left_tcp"),
        "right": SimpleNamespace(controlled_body="right_tcp"),
    }
    gripper_specs = {
        "left_gripper": SimpleNamespace(side="left"),
        "right_gripper": SimpleNamespace(side="right"),
    }

    def get_arm_qpos(self, side: str) -> np.ndarray:
        del side
        return np.zeros(6, dtype=np.float32)

    def get_gripper_qpos(self, name: str) -> np.ndarray:
        del name
        return np.zeros(2, dtype=np.float32)

    def get_gripper_close(self, name: str) -> float:
        del name
        return 0.5

    def body_pose(self, name: str) -> np.ndarray:
        del name
        return np.eye(4, dtype=np.float64)


def _context(scene: _Scene):
    task_config = SimpleNamespace(
        left_gripper_name="left_gripper",
        right_gripper_name="right_gripper",
    )
    target_runtime = SimpleNamespace(
        clock=time.monotonic,
        config=SimpleNamespace(
            policy_node_id="sim-hil-state-bc-fixture",
            clock_domain_id="mujoco.sim_hil.monotonic",
        ),
    )
    return SimpleNamespace(
        scene=scene,
        task_config=task_config,
        task_config_path="unused-task-config.json",
        target_runtime=target_runtime,
        downstream_target_lead_sec=0.0,
        scene_capture=lambda capture: capture(scene),
    )


def test_state_bc_preflight_reports_artifact_and_right_scope(tmp_path):
    artifact = _artifact(tmp_path)
    profile = MujocoStateBCPolicySourceProfile(artifact)

    report = preflight_mujoco_state_bc_policy_source(profile)

    assert report["artifact_schema"] == "dexteleop.state_bc_model.v1"
    assert report["artifact_hash"].startswith("sha256:")
    assert report["policy_dimensions"] == [16, 16]
    assert report["policy_topology"] == {
        "model_observation_sides": ["left", "right"],
        "output_active_sides": ["right"],
    }
    assert report["full_bimanual_execution"] is False


def test_state_bc_source_emits_through_threaded_runtime_and_resets(tmp_path, monkeypatch):
    artifact = _artifact(tmp_path)
    monkeypatch.setattr(
        source_module,
        "capture_mujoco_sim_hil_tool_tcp_pose_by_side",
        lambda _scene, _task: {"left": np.eye(4), "right": np.eye(4)},
    )
    source = build_mujoco_state_bc_policy_source(
        MujocoStateBCPolicySourceProfile(
            artifact,
            action_hz=40.0,
            policy_hz=40.0,
        ),
        _context(_Scene()),
    )
    source.start()
    try:
        deadline = time.monotonic() + 2.0
        emission = None
        while emission is None and time.monotonic() < deadline:
            emission = source.step(now=time.monotonic())
            if emission is None:
                time.sleep(0.002)
        assert emission is not None
        assert emission.decoded_absolute_action is not None
        assert emission.decoded_absolute_action.vector.shape == (26,)
        assert emission.pre_safety_robot_action is not None
        assert emission.pre_safety_robot_action.active_effectors == (
            "right_arm",
            "right_hand",
        )
        status = source.status()
        assert status["backend"] == "state_bc"
        assert status["model_state_dim"] == 16
        assert status["model_action_dim"] == 16
        assert status["model_observation_sides"] == ("left", "right")
        assert status["output_active_sides"] == ("right",)

        source.reset_for_activation()
        assert source.last_emission is None
        assert source.status()["activation_generation"] >= 2
    finally:
        source.close()
