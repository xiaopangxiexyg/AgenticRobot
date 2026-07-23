from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from dataset_contracts.policy_artifact import PolicyArtifactContract
from dataset_contracts.pose_frames import PoseFrameContract
from dataset_contracts.v22 import PolicyIOLayout
from deployments.openpi.client import (
    OpenPISimAdapter,
    OpenPISimAdapterConfig,
)
from robot_motion.deploy import PolicyAdapterConfig
from robot_motion.deploy.observation import build_deployment_observation
from robot_motion.deploy.schema import ACTION_ARM_POSE_SLICES
from sim.mujoco.deploy import (
    MujocoDeploymentObservationAdapter,
    MujocoDeploymentObservationConfig,
)


def _runtime_observation(*, x: float = 0.1):
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [x, 0.0, 0.3]
    return build_deployment_observation(
        active_sides=("right",),
        arms_enabled=True,
        hands_enabled=True,
        end_effector_sides=("right",),
        step_index=3,
        arm_states={
            "right": {
                "current_qpos": np.zeros(6),
                "current_tcp_pose": pose,
                "actual_tcp_pose": pose,
            }
        },
        hand_states={"right": {"joint_pos": np.zeros(6), "joint_timestamp": 10.0}},
        timestamp=10.0,
    )


def test_policy_adapter_config_roundtrip_binds_four_representation_rings() -> None:
    config = PolicyAdapterConfig(
        adapter_id="unit.openpi.relative",
        active_sides=("right",),
        model_action_dim=26,
        action_pose_mode="relative",
        action_chunk_execution_steps=2,
        prediction_horizon_steps=4,
        minimum_execution_horizon_steps=1,
        artifact_hash="artifact:v1",
        normalization_contract={"schema": "normalized_v1", "range": "unit"},
    )

    restored = PolicyAdapterConfig.from_dict(config.to_dict())

    assert restored == config
    assert restored.semantic_content_hash == config.semantic_content_hash
    assert restored.execution_policy.select((1, 2, 3)) == (1, 2)
    assert restored.to_runtime_config().policy_action_pose_mode == "relative"


def test_policy_adapter_config_requires_frame_contract_for_canonical_actions() -> None:
    with pytest.raises(ValueError, match="runtime_pose_frame_contract"):
        PolicyAdapterConfig(
            adapter_id="canonical",
            policy_absolute_frame="canonical_desk",
        )

    contract = PoseFrameContract.legacy_v22_raw(active_sides=("right",))
    config = PolicyAdapterConfig(
        adapter_id="canonical",
        active_sides=("right",),
        policy_absolute_frame="canonical_desk",
        runtime_pose_frame_contract=contract,
    )
    assert config.frame_contract_hash == contract.contract_hash


class _FakeOpenPIPolicy:
    metadata = {"server": {"model_id": "fake"}}

    def __init__(self, actions: np.ndarray):
        self.actions = actions
        self.calls = 0

    def predict(self, observation):
        assert observation.active_sides == ("right",)
        self.calls += 1
        return self.actions

    def reset(self):
        pass


def test_openpi_sim_adapter_uses_shared_relative_pose_and_chunk_decoder() -> None:
    artifact = PolicyArtifactContract(
        policy_io_layout=PolicyIOLayout.legacy_v21_q6(),
        model_id="fake",
        checkpoint_step=1,
    )
    action = np.zeros((2, 26), dtype=np.float32)
    action[:, ACTION_ARM_POSE_SLICES["right"]] = [
        0.2,
        0.0,
        0.3,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    adapter_config = PolicyAdapterConfig(
        adapter_id="openpi.fake",
        active_sides=("right",),
        hands_enabled=True,
        action_pose_mode="relative",
        model_action_dim=artifact.policy_io_layout.action_dim,
        action_chunk_execution_steps=1,
        prediction_horizon_steps=2,
        artifact_hash=None,
    )
    policy = _FakeOpenPIPolicy(action)
    adapter = OpenPISimAdapter(
        OpenPISimAdapterConfig(adapter=adapter_config, policy_artifact=artifact),
        policy=policy,
    )

    prediction = adapter.predict(_runtime_observation(x=0.1))

    assert policy.calls == 1
    assert prediction.source_chunk_size == 2
    assert prediction.executed_chunk_size == 1
    np.testing.assert_allclose(
        prediction.actions[0].vector[ACTION_ARM_POSE_SLICES["right"]][:3],
        [0.3, 0.0, 0.6],
        atol=1e-6,
    )
    assert prediction.metadata["normalization_hash"] == adapter_config.normalization_hash


class _FakeScene:
    gripper_specs = {"right_gripper": SimpleNamespace(side="right")}

    def get_arm_qpos(self, side):
        assert side == "right"
        return np.arange(6, dtype=np.float32)

    def body_pose(self, body_name):
        assert body_name == "right_tcp"
        pose = np.eye(4, dtype=float)
        pose[:3, 3] = [0.1, 0.2, 0.3]
        return pose

    def get_gripper_qpos(self, name):
        assert name == "right_gripper"
        return np.asarray([0.2, 0.2], dtype=np.float32)

    def get_gripper_close(self, name):
        assert name == "right_gripper"
        return 0.25


def test_mujoco_observation_bridge_is_read_only_and_renders_required_images() -> None:
    adapter = MujocoDeploymentObservationAdapter(
        _FakeScene(),
        MujocoDeploymentObservationConfig(
            active_sides=("right",),
            tcp_body_by_side={"right": "right_tcp"},
            camera_names=("right_wrist",),
            require_images=True,
        ),
        renderer=lambda camera: np.zeros((8, 10, 3), dtype=np.uint8),
        clock=lambda: 12.0,
    )

    observation = adapter.capture()

    assert observation.step_index == 0
    assert observation.metadata["source"] == "mujoco"
    assert observation.metadata["tcp_body_by_side"] == {"right": "right_tcp"}
    assert observation.images["right_wrist"].shape == (8, 10, 3)
    assert observation.faults == []
    np.testing.assert_allclose(observation.arm_states["right"]["actual_qpos"], np.arange(6))
    assert observation.hand_states["right"]["gripper_state_scalar"] == 0.25

