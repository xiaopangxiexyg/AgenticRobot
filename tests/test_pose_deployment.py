from __future__ import annotations

import numpy as np
import pytest

from robot_motion.deploy.actions import NormalizedAction
from robot_motion.deploy.observation import DeploymentObservation
from robot_motion.deploy.pose_adapter import (
    PoseDeploymentError,
    convert_last_action_to_policy_mode,
    convert_normalized_action_chunk_to_absolute,
    convert_normalized_action_chunk_to_relative,
    prepare_observation_for_policy_pose_mode,
)
from robot_motion.deploy.schema import ACTION_ARM_POSE_SLICES, ACTION_DIM, OBSERVATION_STATE_DIM
from robot_motion.relative_action import compare_pose_chunks


def _observation(*, actual_x: float = 0.1) -> DeploymentObservation:
    return DeploymentObservation(
        timestamp=10.0,
        step_index=1,
        active_sides=("right",),
        arms_enabled=True,
        hands_enabled=False,
        state=np.full((OBSERVATION_STATE_DIM,), np.nan, dtype=np.float32),
        state_valid=np.zeros((OBSERVATION_STATE_DIM,), dtype=bool),
        arm_states={
            "right": {
                "actual_tcp_pose": [actual_x, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0],
            }
        },
    )


def _action(x: float) -> NormalizedAction:
    vector = np.full((ACTION_DIM,), np.nan, dtype=np.float32)
    vector[ACTION_ARM_POSE_SLICES["right"]] = [x, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]
    return NormalizedAction(vector=vector, metadata={"source": "test"})


def test_relative_policy_chunk_is_converted_before_canonical_safety() -> None:
    observation = _observation(actual_x=0.1)
    relative = _action(0.2)
    converted = convert_normalized_action_chunk_to_absolute(
        [relative], observation, source_mode="relative", active_sides=("right",)
    )
    np.testing.assert_allclose(
        converted[0].vector[ACTION_ARM_POSE_SLICES["right"]][:3],
        [0.3, 0.0, 0.6],
        atol=1e-6,
    )
    assert converted[0].metadata["pose_action_mode"] == "absolute"
    assert converted[0].metadata["pose_action_source_mode"] == "relative"


def test_absolute_replay_chunk_can_be_emitted_relative_and_decoded_back() -> None:
    observation = _observation(actual_x=0.1)
    absolute = _action(0.4)
    relative = convert_normalized_action_chunk_to_relative(
        [absolute], observation, active_sides=("right",)
    )
    reconstructed = convert_normalized_action_chunk_to_absolute(
        relative, observation, source_mode="relative", active_sides=("right",)
    )
    metrics = compare_pose_chunks(
        np.stack([absolute.vector]),
        np.stack([reconstructed[0].vector]),
        active_sides=("right",),
    )["right"]
    assert metrics.max_translation_error_m < 1e-6
    assert metrics.max_rotation_error_rad < 1e-6


def test_last_action_relative_uses_current_measured_tcp() -> None:
    last = _action(0.2).vector
    converted = convert_last_action_to_policy_mode(
        last, _observation(actual_x=0.1), target_mode="relative", active_sides=("right",)
    )
    np.testing.assert_allclose(
        converted[ACTION_ARM_POSE_SLICES["right"]][:3],
        [0.1, 0.0, 0.0],
        atol=1e-6,
    )


def test_policy_observation_converts_finite_last_action_and_preserves_startup_nan() -> None:
    observation = _observation(actual_x=0.1)
    observation = DeploymentObservation(
        **{
            **observation.__dict__,
            "last_action": _action(0.2).vector,
        }
    )
    converted = prepare_observation_for_policy_pose_mode(
        observation, target_mode="relative", active_sides=("right",)
    )
    np.testing.assert_allclose(
        converted.last_action[ACTION_ARM_POSE_SLICES["right"]][:3],
        [0.1, 0.0, 0.0],
        atol=1e-6,
    )
    startup = prepare_observation_for_policy_pose_mode(
        _observation(actual_x=0.1), target_mode="relative", active_sides=("right",)
    )
    assert np.all(np.isnan(startup.last_action))


def test_relative_deployment_fails_closed_without_actual_tcp() -> None:
    observation = DeploymentObservation(
        timestamp=10.0,
        step_index=1,
        active_sides=("right",),
        arms_enabled=True,
        hands_enabled=False,
        state=np.full((OBSERVATION_STATE_DIM,), np.nan, dtype=np.float32),
        state_valid=np.zeros((OBSERVATION_STATE_DIM,), dtype=bool),
        arm_states={"right": {"current_tcp_pose": [0.1, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]}},
    )
    with pytest.raises(PoseDeploymentError, match="actual_tcp_pose"):
        convert_normalized_action_chunk_to_absolute(
            [_action(0.1)], observation, source_mode="relative", active_sides=("right",)
        )
