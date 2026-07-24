from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robot_motion.deploy import DeployRobotActionAdapter, build_deployment_observation
from robot_motion.deploy.schema import ACTION_ARM_POSE_SLICES
from robot_motion.action import HandActionSpec
from sim.mujoco.deploy import (
    MujocoPolicyRollout,
    MujocoPolicyRolloutError,
)


def _observation():
    pose_left = np.eye(4, dtype=float)
    pose_left[:3, 3] = [-0.3, 0.0, 0.4]
    pose_right = np.eye(4, dtype=float)
    pose_right[:3, 3] = [0.3, 0.0, 0.4]
    return build_deployment_observation(
        active_sides=("left", "right"),
        arms_enabled=True,
        hands_enabled=True,
        step_index=4,
        arm_states={
            "left": {"current_qpos": np.zeros(6), "current_tcp_pose": pose_left},
            "right": {"current_qpos": np.zeros(6), "current_tcp_pose": pose_right},
        },
        hand_states={
            "left": {"joint_pos": np.zeros(6), "joint_timestamp": 10.0},
            "right": {"joint_pos": np.zeros(6), "joint_timestamp": 10.0},
        },
        timestamp=10.0,
    )


def _action(x: float) -> np.ndarray:
    value = np.zeros(26, dtype=np.float32)
    value[ACTION_ARM_POSE_SLICES["left"]] = [x, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0]
    value[ACTION_ARM_POSE_SLICES["right"]] = [-x, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0]
    return value


class _Source:
    def __init__(self):
        self.observation = _observation()
        self.last_actions = []

    def capture(self, **kwargs):
        del kwargs
        return self.observation

    def set_last_action(self, action):
        self.last_actions.append(np.asarray(action).copy())


class _Policy:
    def __init__(self, actions):
        self.actions = actions
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1

    def predict(self, observation):
        assert observation is not None
        return SimpleNamespace(actions=self.actions, metadata={"chunk_id": "chunk-7"})


class _Runtime:
    def __init__(self):
        self.started = False
        self.closed = False
        self.submissions = []
        self.steps = []
        self.transitions = []

    def start(self):
        self.started = True

    def close(self):
        self.closed = True

    def submit_policy_action_chunk(self, actions, **kwargs):
        self.submissions.append((tuple(actions), kwargs))
        return {"accepted": True}

    def step(self, **kwargs):
        self.steps.append(kwargs)
        return {"status": "dispatched"}

    def transition_execution_gate(self, status, **kwargs):
        self.transitions.append((status, kwargs))


def _rollout(policy, source=None, runtime=None):
    return MujocoPolicyRollout(
        source or _Source(),
        policy,
        DeployRobotActionAdapter(
            active_sides=("left", "right"),
            arms_enabled=True,
            hands_enabled=True,
        ),
        runtime or _Runtime(),
    )


def test_rollout_admits_one_atomic_decoded_chunk_and_updates_last_action():
    source = _Source()
    runtime = _Runtime()
    rollout = _rollout(_Policy((_action(0.1), _action(0.11))), source, runtime)

    rollout.start()
    result = rollout.run_cycle()

    assert rollout.started
    assert result.cycle_index == 0
    assert result.chunk_id == "chunk-7"
    assert len(result.actions) == 2
    assert len(runtime.submissions) == 1
    assert runtime.submissions[0][1]["policy_output_id"] == "mujoco-openpi:0"
    assert runtime.submissions[0][1]["observation_cut_id"] == "mujoco-observation:4"
    assert len(runtime.steps) == 1
    assert len(source.last_actions) == 1
    np.testing.assert_allclose(source.last_actions[0], _action(0.11))

    rollout.close()
    assert runtime.closed


def test_rollout_rejects_the_whole_chunk_when_one_action_is_invalid():
    source = _Source()
    runtime = _Runtime()
    rollout = _rollout(_Policy((_action(0.1), np.full(26, np.nan))), source, runtime)
    rollout.start()

    with pytest.raises(MujocoPolicyRolloutError, match="rejected before admission"):
        rollout.run_cycle()

    assert runtime.submissions == []
    assert runtime.steps == []
    assert len(runtime.transitions) == 1
    assert runtime.transitions[0][0] == "fault"
    assert source.last_actions == []
    rollout.close()


def test_rollout_carries_observation_hand_specs_for_legacy_dense_actions():
    source = _Source()
    hand_specs = {
        side: HandActionSpec.dmtac_gripper(side)
        for side in ("left", "right")
    }
    source.observation = build_deployment_observation(
        active_sides=("left", "right"),
        arms_enabled=True,
        hands_enabled=True,
        step_index=4,
        arm_states={
            "left": {"current_qpos": np.zeros(6), "current_tcp_pose": np.eye(4)},
            "right": {"current_qpos": np.zeros(6), "current_tcp_pose": np.eye(4)},
        },
        hand_states={
            "left": {"joint_pos": np.zeros(6), "joint_timestamp": 10.0},
            "right": {"joint_pos": np.zeros(6), "joint_timestamp": 10.0},
        },
        timestamp=10.0,
        metadata={
            "hand_action_specs_by_side": {
                side: spec.to_metadata() for side, spec in hand_specs.items()
            }
        },
    )
    runtime = _Runtime()
    rollout = _rollout(_Policy((_action(0.1),)), source, runtime)

    rollout.start()
    result = rollout.run_cycle()

    for action in result.actions:
        specs = action.metadata["hand_action_specs_by_side"]
        for side in ("left", "right"):
            assert specs[side]["hand_model"] == "dmtac_gripper"
            assert specs[side]["active_dof"] == 1
    rollout.close()
