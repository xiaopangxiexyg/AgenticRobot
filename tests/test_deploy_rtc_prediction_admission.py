from __future__ import annotations

import numpy as np
import pytest

from robot_motion.deploy import (
    ACTION_ARM_POSE_SLICES,
    DEPLOY_ACTION_DIM,
    PolicyPredictionTransactionRequiresSynchronousQueueError,
    PolicyDeployRuntime,
    PolicyDeployRuntimeConfig,
    build_deployment_observation,
    nan_vector,
)


class _Clock:
    def __init__(self, *values: float) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return float(next(self._values))


class _Policy:
    name = "rtc-sync-parity-policy"

    def __init__(self, output: np.ndarray) -> None:
        self.output = np.asarray(output, dtype=np.float32)

    def predict(self, _observation):
        return self.output.copy()


class _TransactionalPolicy(_Policy):
    def __init__(self, output: np.ndarray) -> None:
        super().__init__(output)
        self.events: list[str] = []

    def begin_prediction_transaction(self):
        self.events.append("begin")
        return "checkpoint"

    def predict(self, observation):
        self.events.append("predict")
        return super().predict(observation)

    def commit_prediction_transaction(self, checkpoint):
        assert checkpoint == "checkpoint"
        self.events.append("commit")

    def rollback_prediction_transaction(self, checkpoint):
        assert checkpoint == "checkpoint"
        self.events.append("rollback")


class _Preparer:
    def prepare(self, action, _observation, *, policy_name: str = ""):
        return {
            "action": action,
            "policy_name": policy_name,
        }


def _observation():
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [0.10, 0.20, 0.30]
    return build_deployment_observation(
        active_sides=("right",),
        arms_enabled=True,
        hands_enabled=False,
        step_index=7,
        arm_states={
            "right": {
                "current_qpos": np.zeros(6, dtype=np.float32),
                "current_tcp_pose": pose,
                "actual_tcp_pose": pose,
            }
        },
        hand_states={},
        timestamp=10.0,
    )


def _action(x: float) -> np.ndarray:
    action = nan_vector(DEPLOY_ACTION_DIM)
    action[ACTION_ARM_POSE_SLICES["right"]] = np.asarray(
        [x, 0.20, 0.30, 0.0, 0.0, 0.0, 1.0],
        dtype=np.float32,
    )
    return action


def _config(*, execution_steps: int | None = None) -> PolicyDeployRuntimeConfig:
    return PolicyDeployRuntimeConfig(
        active_sides=("right",),
        arms_enabled=True,
        hands_enabled=False,
        action_chunk_execution_steps=execution_steps,
    )


def _action_x(action) -> float:
    return float(action.vector[ACTION_ARM_POSE_SLICES["right"]][0])


def test_split_prediction_replace_matches_legacy_sync_prefix_and_pop_order() -> None:
    observation = _observation()
    output = np.stack((_action(0.11), _action(0.12), _action(0.13)))
    legacy = PolicyDeployRuntime(
        _config(execution_steps=2),
        monotonic_clock=_Clock(10.001, 10.002, 10.003, 10.004),
    )
    split = PolicyDeployRuntime(
        _config(execution_steps=2),
        monotonic_clock=_Clock(10.001, 10.002, 10.003, 10.004),
    )

    legacy_result = legacy.predict_and_queue(
        _Policy(output),
        observation,
        now=10.0,
        queue_policy="replace",
        action_keyframe_period_sec=0.025,
    )
    inference_id = split.reserve_prediction_inference_id()
    candidate = split.predict_chunk(
        _Policy(output),
        observation,
        inference_id=inference_id,
    )

    assert split.action_queue_length() == 0
    assert split.action_queue_pop_count() == 0
    assert candidate.source_chunk_size == 3
    assert len(candidate.actions) == 2
    with pytest.raises(ValueError):
        candidate.actions[0].vector[0] = 0.0
    with pytest.raises(TypeError):
        candidate.actions[0].metadata["unexpected"] = True

    split_result = split.admit_prediction_result(
        candidate,
        now=10.0,
        queue_policy="replace",
        action_keyframe_period_sec=0.025,
    )

    assert legacy_result.accepted is split_result.accepted is True
    assert legacy_result.source_chunk_size == split_result.source_chunk_size == 3
    assert legacy_result.executed_chunk_size == split_result.executed_chunk_size == 2
    assert legacy_result.chunk_lineage is not None
    assert split_result.chunk_lineage is not None
    assert (
        legacy_result.chunk_lineage.semantic_content_hash
        == split_result.chunk_lineage.semantic_content_hash
    )
    for legacy_action, split_action in zip(
        legacy_result.actions,
        split_result.actions,
        strict=True,
    ):
        np.testing.assert_array_equal(legacy_action.vector, split_action.vector)
        assert dict(legacy_action.metadata) == dict(split_action.metadata)

    for expected_index in (0, 1):
        legacy_step = legacy.prepare_next_action(
            observation,
            preparer=_Preparer(),
            policy_name="legacy",
        )
        split_step = split.prepare_next_action(
            observation,
            preparer=_Preparer(),
            policy_name="split",
        )
        assert legacy_step.raw_action is not None
        assert split_step.raw_action is not None
        np.testing.assert_array_equal(
            legacy_step.raw_action.vector,
            split_step.raw_action.vector,
        )
        assert (
            legacy_step.raw_action.metadata["deploy_action_chunk_index"]
            == split_step.raw_action.metadata["deploy_action_chunk_index"]
            == expected_index
        )
        assert legacy_step.emission is not None
        assert split_step.emission is not None
        assert (
            legacy_step.emission.semantic_content_hash
            == split_step.emission.semantic_content_hash
        )

    assert legacy.action_queue_length() == split.action_queue_length() == 0
    assert legacy.action_queue_pop_count() == split.action_queue_pop_count() == 2


def test_split_prediction_replace_discards_the_existing_chunk_as_one_unit() -> None:
    observation = _observation()
    runtime = PolicyDeployRuntime(
        _config(),
        monotonic_clock=_Clock(10.001, 10.002, 10.003, 10.004, 10.005, 10.006),
    )
    original = np.stack((_action(0.11), _action(0.12)))
    replacement = np.stack((_action(0.21), _action(0.22)))
    assert runtime.predict_and_queue(
        _Policy(original),
        observation,
        now=10.0,
    ).accepted

    candidate = runtime.predict_chunk(
        _Policy(replacement),
        observation,
        inference_id=runtime.reserve_prediction_inference_id(),
    )
    assert runtime.action_queue_length() == 2
    result = runtime.admit_prediction_result(
        candidate,
        now=10.1,
        queue_policy="replace",
    )

    assert result.accepted is True
    assert result.queue_len_before == 2
    assert result.queue_len_after == 2
    popped = [
        runtime.prepare_next_action(
            observation,
            preparer=_Preparer(),
            policy_name="split",
        ).raw_action
        for _ in range(2)
    ]
    assert all(action is not None for action in popped)
    assert [_action_x(action) for action in popped] == pytest.approx([0.21, 0.22])
    assert [action.metadata["deploy_action_chunk_index"] for action in popped] == [0, 1]


def test_split_prediction_append_preserves_remaining_then_new_chunk_order() -> None:
    observation = _observation()
    runtime = PolicyDeployRuntime(
        _config(),
        monotonic_clock=_Clock(
            10.001,
            10.002,
            10.003,
            10.004,
            10.005,
            10.006,
            10.007,
            10.008,
        ),
    )
    original = np.stack((_action(0.11), _action(0.12)))
    appended = np.stack((_action(0.21), _action(0.22)))
    initial = runtime.predict_and_queue(
        _Policy(original),
        observation,
        now=10.0,
        action_keyframe_period_sec=0.05,
    )
    assert initial.accepted is True

    first = runtime.prepare_next_action(
        observation,
        preparer=_Preparer(),
        policy_name="legacy",
    )
    assert first.raw_action is not None
    old_chunk_id = first.raw_action.metadata["deploy_action_chunk_id"]
    assert first.raw_action.metadata["deploy_action_chunk_index"] == 0

    candidate = runtime.predict_chunk(
        _Policy(appended),
        observation,
        inference_id=runtime.reserve_prediction_inference_id(),
    )
    assert runtime.action_queue_length() == 1
    appended_result = runtime.admit_prediction_result(
        candidate,
        now=10.1,
        queue_policy="append",
        action_keyframe_period_sec=0.05,
    )
    assert appended_result.accepted is True
    assert appended_result.queue_len_before == 1
    assert appended_result.queue_len_after == 3
    assert appended_result.chunk_lineage is not None
    assert initial.chunk_lineage is not None
    assert (
        appended_result.chunk_lineage.schedule.start_at
        >= initial.chunk_lineage.schedule.start_at + 0.10
    )

    old_tail = runtime.prepare_next_action(
        observation,
        preparer=_Preparer(),
        policy_name="legacy",
    )
    new_head = runtime.prepare_next_action(
        observation,
        preparer=_Preparer(),
        policy_name="split",
    )
    new_tail = runtime.prepare_next_action(
        observation,
        preparer=_Preparer(),
        policy_name="split",
    )
    assert old_tail.raw_action is not None
    assert new_head.raw_action is not None
    assert new_tail.raw_action is not None
    new_chunk_id = new_head.raw_action.metadata["deploy_action_chunk_id"]

    assert [_action_x(step.raw_action) for step in (old_tail, new_head, new_tail)] == (
        pytest.approx([0.12, 0.21, 0.22])
    )
    assert old_tail.raw_action.metadata["deploy_action_chunk_id"] == old_chunk_id
    assert old_tail.raw_action.metadata["deploy_action_chunk_index"] == 1
    assert old_tail.raw_lookahead_action is None
    assert new_chunk_id != old_chunk_id
    assert new_head.raw_action.metadata["deploy_action_chunk_index"] == 0
    assert new_head.raw_lookahead_action is not None
    assert new_tail.raw_action.metadata["deploy_action_chunk_id"] == new_chunk_id
    assert new_tail.raw_action.metadata["deploy_action_chunk_index"] == 1
    assert runtime.action_queue_length() == 0
    assert runtime.action_queue_pop_count() == 4


def test_transactional_policy_remains_on_existing_sync_path() -> None:
    observation = _observation()
    runtime = PolicyDeployRuntime(
        _config(),
        monotonic_clock=_Clock(10.001, 10.002),
    )
    policy = _TransactionalPolicy(_action(0.11))

    inference_id = runtime.reserve_prediction_inference_id()
    with pytest.raises(PolicyPredictionTransactionRequiresSynchronousQueueError):
        runtime.predict_chunk(
            policy,
            observation,
            inference_id=inference_id,
        )
    assert policy.events == []
    assert runtime.action_queue_length() == 0

    result = runtime.predict_and_queue(
        policy,
        observation,
        now=10.0,
    )
    assert result.accepted is True
    assert policy.events == ["begin", "predict", "commit"]
    assert runtime.action_queue_length() == 1
