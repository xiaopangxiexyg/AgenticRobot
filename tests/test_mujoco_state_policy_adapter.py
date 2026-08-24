from __future__ import annotations

import numpy as np
import pytest

from experiments.sim_learning.state_bc.contracts import StateBCArtifactMetadata
from experiments.sim_learning.state_bc.model import fit_linear_state_bc
from robot_motion.deploy import DeploymentObservation
from sim.mujoco.deploy.state_policy_adapter import (
    STATE_BC_LAYOUT_HASH_V1,
    StatePolicyAdapter,
    StatePolicyAdapterConfig,
)


def _observation() -> DeploymentObservation:
    left_pose = np.asarray([0.10, 0.20, 0.30, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    right_pose = np.asarray([0.40, 0.50, 0.60, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    state = np.full(38, np.nan, dtype=np.float32)
    state[6:13] = left_pose
    state[25:32] = right_pose
    state[13:19] = np.asarray([0.2, 0, 0, 0, 0, 0], dtype=np.float32)
    state[32:38] = np.asarray([0.4, 0, 0, 0, 0, 0], dtype=np.float32)
    return DeploymentObservation(
        timestamp=1.5,
        step_index=3,
        active_sides=("left", "right"),
        arms_enabled=True,
        hands_enabled=True,
        state=state,
        state_valid=np.isfinite(state),
        arm_states={"left": {}, "right": {}},
        hand_states={
            "left": {"gripper_measured_scalar": 0.2},
            "right": {"gripper_measured_scalar": 0.4},
        },
        last_action=np.full(26, np.nan, dtype=np.float32),
    )


class _DummyStatePolicy:
    metadata = {"layout_hash": STATE_BC_LAYOUT_HASH_V1}
    artifact_hash = "sha256:fixture"

    def predict(self, state):
        state = np.asarray(state, dtype=np.float32)
        assert state.shape == (16,)
        action = state.copy()
        action[14] = 0.3
        action[15] = 0.7
        return action


def test_state_policy_adapter_projects_16d_model_output_to_existing_26d_action():
    adapter = StatePolicyAdapter(
        model=_DummyStatePolicy(),
        config=StatePolicyAdapterConfig(policy_absolute_frame="runtime_raw"),
    )
    prediction = adapter.predict(_observation())

    assert prediction.model_state.shape == (16,)
    assert prediction.model_action.shape == (16,)
    assert len(prediction.actions) == 1
    action = prediction.actions[0]
    assert action.vector.shape == (26,)
    np.testing.assert_allclose(action.vector[0:7], prediction.model_action[0:7])
    np.testing.assert_allclose(action.vector[7:14], prediction.model_action[7:14])
    assert action.vector[14] == pytest.approx(0.3)
    assert action.vector[20] == pytest.approx(0.7)
    assert action.metadata["state_bc_model_artifact_hash"] == "sha256:fixture"


def test_canonical_desk_mode_requires_an_explicit_pose_frame_contract():
    adapter = StatePolicyAdapter(model=_DummyStatePolicy())
    with pytest.raises(ValueError, match="pose frame contract"):
        adapter.predict(_observation())


def test_state_policy_adapter_validates_the_serialized_artifact_hash(tmp_path):
    states = np.tile(np.arange(16, dtype=np.float32), (3, 1))
    actions = states.copy()
    model = fit_linear_state_bc(
        states,
        actions,
        artifact_metadata=StateBCArtifactMetadata(
            selector_id="fixture-artifact",
            train_episode_count=2,
            test_episode_count=0,
            train_frame_count=3,
            test_frame_count=0,
        ),
    )
    artifact = model.save(tmp_path / "state_policy.npz")

    adapter = StatePolicyAdapter(
        artifact_path=artifact,
        config=StatePolicyAdapterConfig(policy_absolute_frame="runtime_raw"),
    )
    assert adapter.artifact_hash == model.artifact_hash
