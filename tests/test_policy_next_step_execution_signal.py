from __future__ import annotations

import numpy as np
import pytest

from robot_motion.deploy import (
    ACTION_ARM_POSE_SLICES,
    DEPLOY_ACTION_DIM,
    PolicyDeployRuntime,
    PolicyDeployRuntimeConfig,
    build_deployment_observation,
    nan_vector,
    normalized_policy_action_hash,
    sync_queue_preserves_same_chunk_successor,
)


class _ChunkPolicy:
    name = "next-step-contract-policy"

    def __init__(self, chunk: np.ndarray) -> None:
        self._chunk = np.asarray(chunk, dtype=np.float32)

    def predict(self, _observation):
        return self._chunk.copy()


class _Preparer:
    def prepare(self, action, observation, *, policy_name=""):
        del observation, policy_name
        return action


def _observation():
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [0.1, 0.2, 0.3]
    return build_deployment_observation(
        active_sides=("right",),
        arms_enabled=True,
        hands_enabled=False,
        step_index=0,
        arm_states={
            "right": {
                "current_qpos": np.zeros(6, dtype=np.float32),
                "current_tcp_pose": pose,
            }
        },
        hand_states={},
        timestamp=10.0,
    )


def _action(x: float) -> np.ndarray:
    action = nan_vector(DEPLOY_ACTION_DIM)
    action[ACTION_ARM_POSE_SLICES["right"]] = np.asarray(
        [x, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
        dtype=np.float32,
    )
    return action


def _runtime_with_two_step_chunk() -> tuple[PolicyDeployRuntime, object]:
    observation = _observation()
    runtime = PolicyDeployRuntime(
        PolicyDeployRuntimeConfig(
            active_sides=("right",),
            arms_enabled=True,
            hands_enabled=False,
            action_keyframe_period_sec=1.0 / 15.0,
        )
    )
    result = runtime.predict_and_queue(
        _ChunkPolicy(np.stack((_action(0.11), _action(0.12)))),
        observation,
        now=10.0,
        queue_policy="replace",
    )
    assert result.accepted is True
    return runtime, observation


def test_default_preparation_exposes_no_execution_certainty_signal() -> None:
    runtime, observation = _runtime_with_two_step_chunk()

    step = runtime.prepare_next_action(
        observation,
        preparer=_Preparer(),
        policy_name="default-off",
    )

    assert step.raw_lookahead_action is not None
    assert step.next_step_execution_signal is None


def test_existing_non_replacing_interval_exposes_bound_same_chunk_successor() -> None:
    runtime, observation = _runtime_with_two_step_chunk()

    first = runtime.prepare_next_action(
        observation,
        preparer=_Preparer(),
        policy_name="stable-successor",
        same_chunk_next_step_will_execute=True,
    )

    assert first.emission is not None
    assert first.raw_lookahead_action is not None
    signal = first.next_step_execution_signal
    assert signal is not None
    assert signal.current_emission_id == first.emission.emission_id
    assert signal.current_emission_hash == first.emission.semantic_content_hash
    assert signal.chunk_id == first.emission.chunk_id
    assert signal.current_chunk_keyframe_index == 0
    assert signal.next_chunk_keyframe_index == 1
    assert signal.next_action_hash == normalized_policy_action_hash(
        first.raw_lookahead_action
    )
    assert signal.next_target_effective_at == pytest.approx(
        first.emission.effective_until
    )
    assert signal.guarantee_id.endswith(signal.semantic_content_hash)

    second = runtime.prepare_next_action(
        observation,
        preparer=_Preparer(),
        policy_name="stable-successor",
        same_chunk_next_step_will_execute=True,
    )
    assert second.raw_lookahead_action is None
    assert second.next_step_execution_signal is None


@pytest.mark.parametrize(
    ("queue_policy", "min_action_queue", "expected"),
    (
        ("append", 0, True),
        ("append", 4, True),
        ("replace", 0, True),
        ("replace", 1, False),
    ),
)
def test_sync_queue_certainty_uses_only_existing_replacement_semantics(
    queue_policy: str,
    min_action_queue: int,
    expected: bool,
) -> None:
    assert (
        sync_queue_preserves_same_chunk_successor(
            queue_policy=queue_policy,
            min_action_queue=min_action_queue,
        )
        is expected
    )


def test_sync_queue_certainty_rejects_ambiguous_inputs() -> None:
    with pytest.raises(ValueError, match="queue_policy"):
        sync_queue_preserves_same_chunk_successor(
            queue_policy="auto",
            min_action_queue=0,
        )
    with pytest.raises(ValueError, match="non-negative integer"):
        sync_queue_preserves_same_chunk_successor(
            queue_policy="replace",
            min_action_queue=-1,
        )
    runtime, observation = _runtime_with_two_step_chunk()
    with pytest.raises(TypeError, match="must be a bool"):
        runtime.prepare_next_action(
            observation,
            preparer=_Preparer(),
            same_chunk_next_step_will_execute=1,
        )
