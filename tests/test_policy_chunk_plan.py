from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from robot_motion.action import EndEffectorTarget, RobotAction
from robot_motion.deploy.actions import NormalizedAction
from robot_motion.deploy.canonical_policy_action import build_canonical_policy_action
from robot_motion.deploy.chunk_plan import (
    POLICY_PLAN_KEYFRAME_PRODUCER_COMMITTED,
    commit_policy_action_plan_through,
    empty_policy_action_plan,
    mark_policy_action_plan_in_flight,
    merge_policy_chunk_result,
    pop_due_policy_action,
)
from robot_motion.deploy.inference import (
    ASYNC_ALIGNED_CHUNK_MODE_V1,
    NEAREST_RANK_QUANTILE_ESTIMATOR_V1,
    POLICY_ACTION_PAYLOAD_LEGACY_NORMALIZED_V1,
    POLICY_ACTION_PAYLOAD_TYPED_CANONICAL_V1,
    InferenceDelayEstimatorSpec,
    InferenceDelayEstimatorState,
    PolicyChunkResult,
    PolicyInferencePrefix,
    PolicyInferenceRequest,
)
from robot_motion.deploy.observation import DeploymentObservation
from robot_motion.deploy.pose_adapter import (
    convert_normalized_action_chunk_to_absolute,
    convert_normalized_action_chunk_to_relative,
)
from robot_motion.deploy.schema import (
    ACTION_ARM_POSE_SLICES,
    ACTION_DIM,
    OBSERVATION_STATE_DIM,
)


def _action(value: float, *, qpos: bool = False) -> NormalizedAction:
    vector = np.full((ACTION_DIM,), np.nan, dtype=np.float32)
    vector[ACTION_ARM_POSE_SLICES["right"]] = [
        value,
        0.0,
        0.3,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    joints = {"right": np.full((6,), value, dtype=np.float32)} if qpos else {}
    return NormalizedAction(
        vector=vector,
        arm_joint_targets=joints,
        metadata={"source": "unit"},
    )


def _typed_action(
    value: float,
    *,
    observation_cut_id: str,
    source_pose_mode: str = "absolute",
):
    robot_action = RobotAction(
        action_space="bimanual_ee_pose_v1",
        timestamp=10.0,
        arm_pose_by_side={
            "right": np.asarray(
                [value, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0],
                dtype=np.float32,
            )
        },
        end_effector_targets_by_side={
            "right": EndEffectorTarget(
                value_type="continuous",
                values=np.asarray([value], dtype=np.float32),
                adapter="parallel_gripper_position_v1",
                names=("position",),
            )
        },
        source="typed-policy-unit",
    )
    return build_canonical_policy_action(
        robot_action,
        source_pose_mode=source_pose_mode,
        observation_cut_id=observation_cut_id,
        source_action_hash=f"source:{source_pose_mode}:{value}",
        action_codec_hash="codec-hash",
        frame_contract_hash="frame-hash",
        layout_hash="layout-hash",
        normalization_hash="normalization-hash",
        conversion_contract_hash="conversion:unit",
        anchor_identity_hash=(
            "anchor:unit" if source_pose_mode == "relative" else None
        ),
    )


def _empty_plan(*, cursor_at: float = 0.0, epoch: int = 3):
    return empty_policy_action_plan(
        plan_id="plan:unit",
        session_id="session:unit",
        control_domain_id="control:unit",
        producer_id="policy:unit",
        producer_epoch=epoch,
        policy_node_id="node:unit",
        clock_domain_id="monotonic:unit",
        cursor_at=cursor_at,
        created_at=cursor_at,
    )


def _request(
    plan,
    *,
    request_id: str,
    admitted_at: float,
    first_target_at: float,
    horizon: int,
    period: float = 0.1,
    valid_until: float | None = None,
    minimum_horizon: int = 1,
) -> PolicyInferenceRequest:
    return PolicyInferenceRequest(
        request_id=request_id,
        inference_id=f"inference:{request_id}",
        rtc_mode=ASYNC_ALIGNED_CHUNK_MODE_V1,
        session_id=plan.session_id,
        control_domain_id=plan.control_domain_id,
        producer_id=plan.producer_id,
        producer_epoch=plan.producer_epoch,
        policy_node_id=plan.policy_node_id,
        artifact_hash="artifact-hash",
        observation_cut_id=f"observation:{request_id}",
        observation_timestamp=admitted_at,
        observation_step_index=10,
        triggered_at=admitted_at,
        admitted_at=admitted_at,
        first_target_effective_at=first_target_at,
        clock_domain_id=plan.clock_domain_id,
        action_period_sec=period,
        prediction_horizon_steps=horizon,
        minimum_execution_horizon_steps=minimum_horizon,
        source_plan_id=plan.plan_id,
        source_plan_generation=plan.generation,
        source_plan_hash=plan.semantic_content_hash,
        source_plan_cursor_at=plan.cursor_at,
        producer_plan_committed_through=plan.producer_committed_through,
        forecast_delay_steps=1,
        delay_estimator_spec_hash="delay-spec-hash",
        action_codec_hash="codec-hash",
        frame_contract_hash="frame-hash",
        layout_hash="layout-hash",
        normalization_hash="normalization-hash",
        valid_until=(
            first_target_at + period * horizon
            if valid_until is None
            else valid_until
        ),
    )


def _result(
    request: PolicyInferenceRequest,
    values: tuple[float, ...],
    *,
    received_at: float,
    qpos: bool = False,
) -> PolicyChunkResult:
    return PolicyChunkResult(
        request_id=request.request_id,
        request_hash=request.semantic_content_hash,
        inference_id=request.inference_id,
        rtc_mode=request.rtc_mode,
        session_id=request.session_id,
        control_domain_id=request.control_domain_id,
        producer_id=request.producer_id,
        producer_epoch=request.producer_epoch,
        policy_node_id=request.policy_node_id,
        artifact_hash=request.artifact_hash,
        action_codec_hash=request.action_codec_hash,
        frame_contract_hash=request.frame_contract_hash,
        layout_hash=request.layout_hash,
        normalization_hash=request.normalization_hash,
        observation_cut_id=request.observation_cut_id,
        observation_timestamp=request.observation_timestamp,
        observation_step_index=request.observation_step_index,
        policy_output_id=f"output:{request.request_id}",
        chunk_id=int(request.request_id.rsplit(":", 1)[-1]),
        inference_started_at=request.admitted_at,
        inference_completed_at=received_at,
        result_received_at=received_at,
        clock_domain_id=request.clock_domain_id,
        status="ready",
        source_chunk_size=len(values),
        decoded_absolute_actions=tuple(_action(value, qpos=qpos) for value in values),
        model_output_hash=f"model-output:{request.request_id}",
        execution_horizon_steps=len(values),
    )


def _fault_result(
    request: PolicyInferenceRequest,
    *,
    received_at: float,
) -> PolicyChunkResult:
    return PolicyChunkResult(
        request_id=request.request_id,
        request_hash=request.semantic_content_hash,
        inference_id=request.inference_id,
        rtc_mode=request.rtc_mode,
        session_id=request.session_id,
        control_domain_id=request.control_domain_id,
        producer_id=request.producer_id,
        producer_epoch=request.producer_epoch,
        policy_node_id=request.policy_node_id,
        artifact_hash=request.artifact_hash,
        action_codec_hash=request.action_codec_hash,
        frame_contract_hash=request.frame_contract_hash,
        layout_hash=request.layout_hash,
        normalization_hash=request.normalization_hash,
        observation_cut_id=request.observation_cut_id,
        observation_timestamp=request.observation_timestamp,
        observation_step_index=request.observation_step_index,
        policy_output_id=None,
        chunk_id=None,
        inference_started_at=request.admitted_at,
        inference_completed_at=received_at,
        result_received_at=received_at,
        clock_domain_id=request.clock_domain_id,
        status="fault",
        source_chunk_size=0,
        decoded_absolute_actions=(),
        model_output_hash=None,
        fault_code="transport_error",
        fault_detail="unit fixture",
    )


def _merge_new_chunk(
    plan,
    request: PolicyInferenceRequest,
    result: PolicyChunkResult,
    *,
    admitted_at: float,
    cursor_at: float | None = None,
    committed_through: float | None = None,
):
    in_flight = mark_policy_action_plan_in_flight(plan, request)
    return merge_policy_chunk_result(
        in_flight,
        request,
        result,
        admitted_at=admitted_at,
        actual_cursor_at=cursor_at,
        producer_committed_through=committed_through,
    )


def _right_observation(actual_x: float) -> DeploymentObservation:
    return DeploymentObservation(
        timestamp=20.0,
        step_index=20,
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


def test_delay_estimator_has_versioned_deterministic_forecast() -> None:
    spec = InferenceDelayEstimatorSpec(
        spec_id="delay:p95",
        action_period_sec=0.05,
        window_size=3,
        estimator_kind=NEAREST_RANK_QUANTILE_ESTIMATOR_V1,
        quantile=0.8,
        warmup_latency_sec=0.06,
        network_guard_sec=0.01,
        control_jitter_guard_steps=1,
        maximum_delay_steps=5,
    )
    state = InferenceDelayEstimatorState.empty(spec)
    assert state.forecast_steps(spec) == 3
    for sample in (0.02, 0.07, 0.11, 0.04):
        state = state.record(spec, sample)
    assert state.latency_samples_sec == (0.07, 0.11, 0.04)
    assert state.accepted_sample_count == 4
    assert state.forecast_steps(spec) == 4
    assert spec.semantic_content_hash == replace(spec).semantic_content_hash
    assert state.semantic_content_hash == replace(state).semantic_content_hash


def test_request_prefix_and_result_detach_mutable_arrays() -> None:
    model_prefix = np.arange(12, dtype=np.float32).reshape(3, 4)
    prefix = PolicyInferencePrefix(
        model_space_actions=model_prefix,
        valid_steps=2,
        source_plan_id="plan:unit",
        source_plan_generation=0,
        source_plan_hash="plan-hash",
    )
    prefix_hash = prefix.semantic_content_hash
    model_prefix[:] = -1
    assert prefix.semantic_content_hash == prefix_hash
    assert not prefix.model_space_actions.flags.writeable

    enabled_sides = ["right"]
    source_action = _action(0.2, qpos=True)
    source_action.metadata["end_effector_enabled_sides"] = enabled_sides
    plan = _empty_plan()
    request = _request(
        plan,
        request_id="request:1",
        admitted_at=0.0,
        first_target_at=0.1,
        horizon=1,
    )
    result = PolicyChunkResult(
        **{
            **_result(request, (0.2,), received_at=0.05).__dict__,
            "decoded_absolute_actions": (source_action,),
        }
    )
    result_hash = result.semantic_content_hash
    source_action.vector[:] = 9.0
    source_action.arm_joint_targets["right"][:] = 9.0
    enabled_sides.append("left")
    assert result.semantic_content_hash == result_hash
    assert not result.decoded_absolute_actions[0].vector.flags.writeable
    assert not result.decoded_absolute_actions[0].arm_joint_targets["right"].flags.writeable


def test_result_rejects_nonfinite_decoded_joint_action_before_plan_admission() -> None:
    plan = _empty_plan()
    request = _request(
        plan,
        request_id="request:18",
        admitted_at=1.0,
        first_target_at=1.1,
        horizon=1,
    )
    result = _result(request, (0.1,), received_at=1.05, qpos=True)
    invalid = NormalizedAction(
        vector=result.decoded_absolute_actions[0].vector,
        arm_joint_targets={"right": np.full((6,), np.inf, dtype=np.float32)},
        metadata={"source": "invalid-unit"},
    )

    with pytest.raises(ValueError, match="finite one-dimensional"):
        replace(result, decoded_absolute_actions=(invalid,))


def test_legacy_result_keyframe_and_plan_hashes_remain_v1_compatible() -> None:
    plan = _empty_plan()
    request = _request(
        plan,
        request_id="request:91",
        admitted_at=1.0,
        first_target_at=1.1,
        horizon=2,
    )
    result = _result(request, (0.2, 0.3), received_at=1.05)
    merged = _merge_new_chunk(
        plan,
        request,
        result,
        admitted_at=1.05,
    )

    assert result.action_payload_kind == POLICY_ACTION_PAYLOAD_LEGACY_NORMALIZED_V1
    assert (
        result.semantic_content_hash
        == "e62e9c2d21dbd6d2daa748cdf063081455cfc88120621c23cab5a2cc697fca2e"
    )
    assert (
        merged.plan.keyframes[0].semantic_content_hash
        == "cfa0a92d6ccd11ddda8157b213ce34744484c79c0f7b4c57a5847a8a8e7eedcf"
    )
    assert (
        merged.plan.semantic_content_hash
        == "e24872015bee21a183b3338b2c8b9db3f81d21870446658e65f849f68f424d89"
    )


def test_typed_result_is_exclusive_and_survives_merge_and_pop() -> None:
    plan = _empty_plan()
    request = _request(
        plan,
        request_id="request:92",
        admitted_at=1.0,
        first_target_at=1.1,
        horizon=2,
    )
    typed_actions = tuple(
        _typed_action(
            value,
            observation_cut_id=request.observation_cut_id,
        )
        for value in (0.25, 0.35)
    )
    typed_result = replace(
        _result(request, (0.25, 0.35), received_at=1.05),
        decoded_absolute_actions=(),
        canonical_actions=typed_actions,
    )

    assert typed_result.action_payload_kind == POLICY_ACTION_PAYLOAD_TYPED_CANONICAL_V1
    with pytest.raises(ValueError, match="exactly one action payload"):
        replace(
            typed_result,
            decoded_absolute_actions=(_action(0.25), _action(0.35)),
        )
    with pytest.raises(ValueError, match="exactly one action payload"):
        replace(typed_result, canonical_actions=())
    with pytest.raises(ValueError, match="cannot carry an action payload"):
        replace(
            _fault_result(request, received_at=1.05),
            canonical_actions=(typed_actions[0],),
        )

    merged = _merge_new_chunk(
        plan,
        request,
        typed_result,
        admitted_at=1.05,
    )
    assert merged.receipt.accepted
    assert len(merged.plan.keyframes) == 2
    assert all(
        keyframe.action_payload_kind == POLICY_ACTION_PAYLOAD_TYPED_CANONICAL_V1
        for keyframe in merged.plan.keyframes
    )
    assert all(
        keyframe.decoded_absolute_action is None
        for keyframe in merged.plan.keyframes
    )
    assert all(
        keyframe.canonical_action is not None
        for keyframe in merged.plan.keyframes
    )

    popped = pop_due_policy_action(merged.plan, decision_at=1.1)
    assert popped.keyframe is not None
    assert popped.keyframe.canonical_action is not None
    np.testing.assert_allclose(
        popped.keyframe.canonical_action.action.arm_pose_by_side["right"],
        typed_actions[0].action.arm_pose_by_side["right"],
    )
    np.testing.assert_allclose(
        popped.keyframe.canonical_action.action.end_effector_targets_by_side[
            "right"
        ].values,
        np.asarray([0.25], dtype=np.float32),
    )
    assert not (
        popped.keyframe.canonical_action.action.end_effector_targets_by_side[
            "right"
        ].values.flags.writeable
    )


def test_typed_result_rejects_receipt_drift_within_one_chunk() -> None:
    plan = _empty_plan()
    request = _request(
        plan,
        request_id="request:92",
        admitted_at=1.0,
        first_target_at=1.1,
        horizon=2,
    )
    first = _typed_action(
        0.25,
        observation_cut_id=request.observation_cut_id,
    )
    wrong_cut = _typed_action(
        0.35,
        observation_cut_id="observation:other",
    )
    relative = _typed_action(
        0.35,
        observation_cut_id=request.observation_cut_id,
        source_pose_mode="relative",
    )
    legacy_result = _result(request, (0.25, 0.35), received_at=1.05)

    with pytest.raises(ValueError, match="observation_cut_id"):
        replace(
            legacy_result,
            decoded_absolute_actions=(),
            canonical_actions=(first, wrong_cut),
        )
    with pytest.raises(ValueError, match="source_pose_mode"):
        replace(
            legacy_result,
            decoded_absolute_actions=(),
            canonical_actions=(first, relative),
        )
    with pytest.raises(ValueError, match="action_codec_hash"):
        replace(
            legacy_result,
            decoded_absolute_actions=(),
            canonical_actions=(
                first,
                build_canonical_policy_action(
                    relative.action,
                    source_pose_mode="absolute",
                    observation_cut_id=request.observation_cut_id,
                    source_action_hash="source:absolute:codec-drift",
                    action_codec_hash="codec:other",
                    frame_contract_hash=request.frame_contract_hash,
                    layout_hash=request.layout_hash,
                    normalization_hash=request.normalization_hash,
                    conversion_contract_hash="conversion:unit",
                ),
            ),
        )


def test_plan_rejects_switching_action_payload_kind_within_one_session() -> None:
    plan = _empty_plan()
    legacy_request = _request(
        plan,
        request_id="request:93",
        admitted_at=1.0,
        first_target_at=1.1,
        horizon=2,
    )
    legacy = _merge_new_chunk(
        plan,
        legacy_request,
        _result(legacy_request, (0.2, 0.3), received_at=1.05),
        admitted_at=1.05,
    )
    assert legacy.receipt.accepted

    typed_request = _request(
        legacy.plan,
        request_id="request:94",
        admitted_at=1.06,
        first_target_at=1.2,
        horizon=2,
    )
    typed_result = replace(
        _result(typed_request, (0.4, 0.5), received_at=1.07),
        decoded_absolute_actions=(),
        canonical_actions=tuple(
            _typed_action(
                value,
                observation_cut_id=typed_request.observation_cut_id,
            )
            for value in (0.4, 0.5)
        ),
    )
    rejected = _merge_new_chunk(
        legacy.plan,
        typed_request,
        typed_result,
        admitted_at=1.07,
    )
    assert rejected.receipt.accepted is False
    assert "action_payload_kind_mismatch" in rejected.receipt.reasons
    assert all(
        keyframe.action_payload_kind == POLICY_ACTION_PAYLOAD_LEGACY_NORMALIZED_V1
        for keyframe in rejected.plan.keyframes
    )


def test_merge_crops_past_preserves_committed_and_replaces_mutable_future() -> None:
    plan = _empty_plan()
    first_request = _request(
        plan,
        request_id="request:1",
        admitted_at=0.0,
        first_target_at=1.0,
        horizon=4,
    )
    first = _merge_new_chunk(
        plan,
        first_request,
        _result(first_request, (1.0, 1.1, 1.2, 1.3), received_at=0.2),
        admitted_at=0.2,
    )
    assert first.receipt.accepted
    committed = commit_policy_action_plan_through(
        first.plan,
        committed_through=1.1,
        committed_at=0.3,
    )
    second_request = _request(
        committed,
        request_id="request:2",
        admitted_at=0.4,
        first_target_at=1.0,
        horizon=4,
    )
    second = _merge_new_chunk(
        committed,
        second_request,
        _result(second_request, (2.0, 2.1, 2.2, 2.3), received_at=1.04),
        admitted_at=1.05,
        cursor_at=1.05,
        committed_through=1.1,
    )
    assert second.receipt.accepted
    assert second.receipt.preserved_committed_count == 1
    assert second.receipt.replaced_mutable_count == 2
    assert second.receipt.admitted_keyframe_count == 2
    assert [item.target_effective_at for item in second.plan.keyframes] == [1.1, 1.2, 1.3]
    assert second.plan.keyframes[0].state == POLICY_PLAN_KEYFRAME_PRODUCER_COMMITTED
    np.testing.assert_allclose(
        [item.decoded_absolute_action.vector[ACTION_ARM_POSE_SLICES["right"]][0]
         for item in second.plan.keyframes],
        [1.1, 2.2, 2.3],
    )


def test_zero_latency_chunk_projects_to_the_requested_target_schedule() -> None:
    plan = _empty_plan(cursor_at=2.0)
    request = _request(
        plan,
        request_id="request:1",
        admitted_at=2.0,
        first_target_at=2.0,
        horizon=3,
    )
    outcome = _merge_new_chunk(
        plan,
        request,
        _result(request, (0.1, 0.2, 0.3), received_at=2.0),
        admitted_at=2.0,
        cursor_at=2.0,
    )
    assert outcome.receipt.accepted
    assert [item.target_effective_at for item in outcome.plan.keyframes] == [
        2.0,
        2.1,
        2.2,
    ]
    assert [item.chunk_keyframe_index for item in outcome.plan.keyframes] == [0, 1, 2]


def test_result_execution_horizon_limits_the_admitted_future() -> None:
    plan = _empty_plan()
    request = _request(
        plan,
        request_id="request:1",
        admitted_at=0.0,
        first_target_at=0.2,
        horizon=4,
    )
    result = replace(
        _result(request, (0.1, 0.2, 0.3, 0.4), received_at=0.1),
        execution_horizon_steps=2,
    )
    outcome = _merge_new_chunk(
        plan,
        request,
        result,
        admitted_at=0.1,
    )
    assert outcome.receipt.accepted
    assert outcome.receipt.admitted_keyframe_count == 2
    assert outcome.receipt.discarded_beyond_horizon_count == 2
    assert [item.chunk_keyframe_index for item in outcome.plan.keyframes] == [0, 1]


def test_matching_late_result_is_rejected_and_releases_in_flight_slot() -> None:
    plan = _empty_plan()
    request = _request(
        plan,
        request_id="request:1",
        admitted_at=0.0,
        first_target_at=0.2,
        horizon=3,
        valid_until=0.5,
    )
    result = _result(request, (1.0, 1.1, 1.2), received_at=0.4)
    outcome = _merge_new_chunk(
        plan,
        request,
        result,
        admitted_at=0.6,
        cursor_at=0.6,
    )
    assert not outcome.receipt.accepted
    assert outcome.receipt.reasons == ("request_deadline_expired",)
    assert outcome.plan.in_flight_request_id is None
    assert not outcome.plan.keyframes


def test_worker_fault_result_is_a_terminal_rejection_without_placeholder_actions() -> None:
    plan = _empty_plan()
    request = _request(
        plan,
        request_id="request:1",
        admitted_at=0.0,
        first_target_at=0.2,
        horizon=2,
    )
    outcome = _merge_new_chunk(
        plan,
        request,
        _fault_result(request, received_at=0.1),
        admitted_at=0.1,
    )
    assert not outcome.receipt.accepted
    assert outcome.receipt.reasons == ("worker_result_fault",)
    assert outcome.plan.in_flight_request_id is None
    assert not outcome.plan.keyframes


def test_fully_obsolete_result_is_rejected_as_empty_future() -> None:
    plan = _empty_plan()
    request = _request(
        plan,
        request_id="request:1",
        admitted_at=0.0,
        first_target_at=0.2,
        horizon=2,
    )
    outcome = _merge_new_chunk(
        plan,
        request,
        _result(request, (1.0, 1.1), received_at=0.3),
        admitted_at=0.35,
        cursor_at=0.35,
    )
    assert not outcome.receipt.accepted
    assert outcome.receipt.reasons == ("no_eligible_future",)
    assert outcome.receipt.discarded_obsolete_count == 2


def test_rejected_result_still_records_new_cursor_and_committed_boundary() -> None:
    plan = _empty_plan()
    first_request = _request(
        plan,
        request_id="request:1",
        admitted_at=0.0,
        first_target_at=1.0,
        horizon=4,
    )
    first = _merge_new_chunk(
        plan,
        first_request,
        _result(first_request, (1.0, 1.1, 1.2, 1.3), received_at=0.2),
        admitted_at=0.2,
    )
    second_request = _request(
        first.plan,
        request_id="request:2",
        admitted_at=0.4,
        first_target_at=1.0,
        horizon=4,
    )
    rejected = _merge_new_chunk(
        first.plan,
        second_request,
        replace(
            _result(second_request, (2.0, 2.1, 2.2, 2.3), received_at=1.04),
            producer_epoch=second_request.producer_epoch + 1,
        ),
        admitted_at=1.05,
        cursor_at=1.05,
        committed_through=1.1,
    )
    assert not rejected.receipt.accepted
    assert rejected.plan.cursor_at == 1.05
    assert rejected.plan.producer_committed_through == 1.1
    assert [item.target_effective_at for item in rejected.plan.keyframes] == [
        1.1,
        1.2,
        1.3,
    ]
    assert rejected.plan.keyframes[0].state == POLICY_PLAN_KEYFRAME_PRODUCER_COMMITTED
    assert rejected.receipt.discarded_obsolete_count == 1


def test_malformed_epoch_result_is_rejected_without_action_admission() -> None:
    plan = _empty_plan()
    request = _request(
        plan,
        request_id="request:1",
        admitted_at=0.0,
        first_target_at=0.2,
        horizon=2,
    )
    result = replace(
        _result(request, (1.0, 1.1), received_at=0.1),
        producer_epoch=request.producer_epoch + 1,
    )
    outcome = _merge_new_chunk(
        plan,
        request,
        result,
        admitted_at=0.1,
    )
    assert not outcome.receipt.accepted
    assert "producer_epoch_mismatch" in outcome.receipt.reasons
    assert not outcome.plan.keyframes


def test_representation_hash_mismatch_is_rejected_before_plan_admission() -> None:
    plan = _empty_plan()
    request = _request(
        plan,
        request_id="request:1",
        admitted_at=0.0,
        first_target_at=0.2,
        horizon=2,
    )
    result = replace(
        _result(request, (1.0, 1.1), received_at=0.1),
        frame_contract_hash="wrong-frame-hash",
    )
    outcome = _merge_new_chunk(
        plan,
        request,
        result,
        admitted_at=0.1,
    )
    assert not outcome.receipt.accepted
    assert "frame_contract_mismatch" in outcome.receipt.reasons
    assert not outcome.plan.keyframes


def test_duplicate_result_cannot_change_an_already_merged_plan() -> None:
    plan = _empty_plan()
    request = _request(
        plan,
        request_id="request:1",
        admitted_at=0.0,
        first_target_at=0.2,
        horizon=2,
    )
    result = _result(request, (1.0, 1.1), received_at=0.1)
    first = _merge_new_chunk(plan, request, result, admitted_at=0.1)
    first_hash = first.plan.semantic_content_hash
    duplicate = merge_policy_chunk_result(
        first.plan,
        request,
        result,
        admitted_at=0.15,
    )
    assert not duplicate.receipt.accepted
    assert duplicate.receipt.reasons == ("request_not_in_flight",)
    assert duplicate.plan.semantic_content_hash == first_hash


def test_out_of_order_old_result_cannot_clear_the_current_in_flight_request() -> None:
    plan = _empty_plan()
    first_request = _request(
        plan,
        request_id="request:1",
        admitted_at=0.0,
        first_target_at=0.2,
        horizon=2,
    )
    first_result = _result(first_request, (1.0, 1.1), received_at=0.1)
    first = _merge_new_chunk(plan, first_request, first_result, admitted_at=0.1)
    second_request = _request(
        first.plan,
        request_id="request:2",
        admitted_at=0.15,
        first_target_at=0.3,
        horizon=2,
    )
    with_second_in_flight = mark_policy_action_plan_in_flight(
        first.plan,
        second_request,
    )
    before_hash = with_second_in_flight.semantic_content_hash
    stale = merge_policy_chunk_result(
        with_second_in_flight,
        first_request,
        first_result,
        admitted_at=0.2,
    )
    assert not stale.receipt.accepted
    assert stale.receipt.reasons == ("request_not_in_flight",)
    assert stale.plan.semantic_content_hash == before_hash
    assert stale.plan.in_flight_request_id == second_request.request_id


def test_noninteger_cadence_produces_replayable_target_times_and_hashes() -> None:
    period = 1.0 / 15.0
    plan = _empty_plan()
    request = _request(
        plan,
        request_id="request:1",
        admitted_at=10.0,
        first_target_at=10.2,
        horizon=3,
        period=period,
    )
    result = _result(request, (1.0, 1.1, 1.2), received_at=10.1)
    first = _merge_new_chunk(plan, request, result, admitted_at=10.1)

    replay_plan = _empty_plan()
    replay_request = _request(
        replay_plan,
        request_id="request:1",
        admitted_at=10.0,
        first_target_at=10.2,
        horizon=3,
        period=period,
    )
    replay_result = _result(replay_request, (1.0, 1.1, 1.2), received_at=10.1)
    replay = _merge_new_chunk(
        replay_plan,
        replay_request,
        replay_result,
        admitted_at=10.1,
    )
    assert [item.target_effective_at for item in first.plan.keyframes] == [
        10.2 + index * period for index in range(3)
    ]
    assert replay.plan.semantic_content_hash == first.plan.semantic_content_hash
    assert replay.receipt.semantic_content_hash == first.receipt.semantic_content_hash


def test_pop_returns_only_latest_current_keyframe_and_discards_expired_due_items() -> None:
    plan = _empty_plan()
    request = _request(
        plan,
        request_id="request:1",
        admitted_at=0.0,
        first_target_at=0.2,
        horizon=4,
    )
    merged = _merge_new_chunk(
        plan,
        request,
        _result(request, (1.0, 1.1, 1.2, 1.3), received_at=0.1),
        admitted_at=0.1,
    )
    popped = pop_due_policy_action(merged.plan, decision_at=0.45)
    assert popped.keyframe is not None
    assert popped.keyframe.chunk_keyframe_index == 2
    assert popped.discarded_expired_count == 2
    assert popped.decision_at == 0.45
    assert popped.selected_target_effective_at == pytest.approx(0.4)
    assert popped.oldest_skipped_target_effective_at == pytest.approx(0.2)
    assert popped.selected_lateness_sec == pytest.approx(0.05)
    assert [item.chunk_keyframe_index for item in popped.plan.keyframes] == [3]


def test_qpos_targets_survive_merge_as_immutable_absolute_actions() -> None:
    plan = _empty_plan()
    request = _request(
        plan,
        request_id="request:1",
        admitted_at=0.0,
        first_target_at=0.2,
        horizon=2,
    )
    outcome = _merge_new_chunk(
        plan,
        request,
        _result(request, (0.2, 0.3), received_at=0.1, qpos=True),
        admitted_at=0.1,
    )
    assert outcome.receipt.accepted
    np.testing.assert_allclose(
        outcome.plan.keyframes[1].decoded_absolute_action.arm_joint_targets["right"],
        np.full((6,), 0.3, dtype=np.float32),
    )
    assert not outcome.plan.keyframes[1].decoded_absolute_action.arm_joint_targets[
        "right"
    ].flags.writeable


def test_relative_reencoding_uses_one_new_fixed_anchor_not_previous_plan_targets() -> None:
    plan = _empty_plan()
    request = _request(
        plan,
        request_id="request:1",
        admitted_at=0.0,
        first_target_at=0.2,
        horizon=2,
    )
    outcome = _merge_new_chunk(
        plan,
        request,
        _result(request, (0.4, 0.5), received_at=0.1),
        admitted_at=0.1,
    )
    absolute = [item.decoded_absolute_action for item in outcome.plan.keyframes]
    observation = _right_observation(actual_x=0.2)
    relative = convert_normalized_action_chunk_to_relative(
        absolute,
        observation,
        active_sides=("right",),
    )
    relative_x = [
        action.vector[ACTION_ARM_POSE_SLICES["right"]][0] for action in relative
    ]
    np.testing.assert_allclose(relative_x, [0.2, 0.3], atol=1e-6)
    reconstructed = convert_normalized_action_chunk_to_absolute(
        relative,
        observation,
        source_mode="relative",
        active_sides=("right",),
    )
    reconstructed_x = [
        action.vector[ACTION_ARM_POSE_SLICES["right"]][0]
        for action in reconstructed
    ]
    np.testing.assert_allclose(reconstructed_x, [0.4, 0.5], atol=1e-6)


def test_no_rtc_absolute_gold_precedes_absolute_and_relative_raw_rtc() -> None:
    """Change scheduler and representation one axis at a time."""

    values = (0.4, 0.5, 0.6)
    no_rtc_absolute_gold = tuple(_action(value) for value in values)

    def admit_and_pop(
        decoded_actions: tuple[NormalizedAction, ...],
        *,
        request_id: str,
    ) -> tuple[NormalizedAction, ...]:
        plan = _empty_plan(cursor_at=0.0)
        request = _request(
            plan,
            request_id=request_id,
            admitted_at=0.0,
            first_target_at=0.1,
            horizon=len(decoded_actions),
        )
        result = replace(
            _result(request, values, received_at=0.0),
            decoded_absolute_actions=decoded_actions,
        )
        merged = _merge_new_chunk(
            plan,
            request,
            result,
            admitted_at=0.0,
            cursor_at=0.0,
        )
        assert merged.receipt.accepted
        current = merged.plan
        popped: list[NormalizedAction] = []
        for decision_at in (0.1, 0.2, 0.300001):
            outcome = pop_due_policy_action(current, decision_at=decision_at)
            assert outcome.keyframe is not None
            popped.append(outcome.keyframe.decoded_absolute_action)
            current = outcome.plan
        return tuple(popped)

    # G0 -> G1: same captured absolute chunk; only the timed-plan scheduler changes.
    absolute_raw_rtc = admit_and_pop(
        no_rtc_absolute_gold,
        request_id="request:31",
    )
    for gold, candidate in zip(no_rtc_absolute_gold, absolute_raw_rtc, strict=True):
        np.testing.assert_allclose(candidate.vector, gold.vector, equal_nan=True)

    # G0 -> G2: same targets under a fixed observation anchor; scheduler stays out.
    original_observation = _right_observation(actual_x=0.2)
    relative_chunk = convert_normalized_action_chunk_to_relative(
        no_rtc_absolute_gold,
        original_observation,
        active_sides=("right",),
    )
    relative_no_rtc = tuple(
        convert_normalized_action_chunk_to_absolute(
            relative_chunk,
            original_observation,
            source_mode="relative",
            active_sides=("right",),
        )
    )
    for gold, candidate in zip(no_rtc_absolute_gold, relative_no_rtc, strict=True):
        np.testing.assert_allclose(candidate.vector, gold.vector, equal_nan=True)

    # G2 -> G3: same decoded relative chunk; only the timed-plan scheduler changes.
    relative_raw_rtc = admit_and_pop(
        relative_no_rtc,
        request_id="request:32",
    )
    for gold, candidate in zip(no_rtc_absolute_gold, relative_raw_rtc, strict=True):
        np.testing.assert_allclose(candidate.vector, gold.vector, equal_nan=True)

    # Negative control: completion-time re-anchoring must not match the gold.
    moved_observation = _right_observation(actual_x=0.35)
    wrongly_reanchored = convert_normalized_action_chunk_to_absolute(
        relative_chunk,
        moved_observation,
        source_mode="relative",
        active_sides=("right",),
    )
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            wrongly_reanchored[0].vector,
            no_rtc_absolute_gold[0].vector,
            equal_nan=True,
        )
