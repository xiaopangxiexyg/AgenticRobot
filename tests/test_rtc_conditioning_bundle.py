from __future__ import annotations

from dataclasses import replace
import threading
import time

import numpy as np
import pytest

from deployments.client.rtc_conditioning import (
    RtcConditioningCodecIdentity,
    RtcEncodedConditioning,
    encode_external_continuation_artifact_actions,
    encode_rtc_conditioning_bundle,
    rtc_binary_mask_hash,
    rtc_float32_array_hash,
)
from deployments.client.rtc_conditioning_protocol import (
    RTC_CONDITIONING_TRANSFORM_CLIENT_V1,
    RTC_CONDITIONING_TRANSFORM_PROVIDER_V1,
    RtcConditionedModelBackendV3,
    RtcConditionedModelResponseV3,
    build_rtc_conditioned_inference_envelope_v3,
    build_rtc_conditioning_consumption_receipt_v3,
    rtc_model_continuation_capability_from_loaded_codec,
    validate_rtc_conditioned_response_v3,
    validate_rtc_model_continuation_capability_v3_binding,
)
from deployments.client.rtc_conditioned_client import (
    RtcConditionedModelClientV3,
)
from deployments.client.rtc import RtcClientProtocolError
from deployments.client.rtc_executor import (
    RtcConditionedModelClientInferenceExecutorV3,
    RtcPreparedModelInput,
    rtc_client_inference_bundle_debug_payload,
    rtc_conditioned_client_inference_bundle_debug_payload,
)
from deployments.client.rtc_frontend import (
    build_rtc_conditioned_model_shadow_frontend_v3,
    rtc_conditioned_config_from_model_adapter_v3,
)
from dataset_contracts.model_continuation import (
    MODEL_CONTINUATION_FLOW_INPAINT_V1,
    MODEL_CONTINUATION_NATIVE_V1,
    MODEL_CONTINUATION_ORIGIN_INFERENCE_ONLY_V1,
    MODEL_CONTINUATION_SAMPLER_CLAMPED_V1,
    MODEL_CONTINUATION_SAMPLER_WEIGHTED_V1,
    RUNTIME_PREFIX_PRODUCER_LOCAL_OVERLAP_V1,
    TRAINING_PREFIX_TRAJECTORY_OVERLAP_TEACHER_V1,
    ModelContinuationContract,
    ModelContinuationDataContract,
    ModelContinuationDataContractV2,
    ModelContinuationRuntimeContractV2,
    normalize_model_continuation_contract,
)
from robot_motion.action import RobotAction
from robot_motion.deploy.action_representation import (
    ACTION_ANCHOR_POLICY_NONE_V1,
    ACTION_ANCHOR_POLICY_REQUEST_ACTUAL_TCP_FIXED_V1,
    PolicyActionRepresentationContext,
)
from robot_motion.deploy.async_producer import PolicyAsyncExecutionProfile
from robot_motion.deploy.canonical_policy_action import (
    build_canonical_policy_action,
)
from robot_motion.deploy.chunk_plan import (
    POLICY_PLAN_KEYFRAME_MUTABLE,
    TimedPolicyActionKeyframe,
    TimedPolicyActionPlanSnapshot,
    mark_policy_action_plan_in_flight,
    merge_policy_chunk_result,
)
from robot_motion.deploy.inference import (
    InferenceDelayEstimatorSpec,
    RTC_FLOW_INPAINT_MODE_V1,
    PolicyInferenceRequest,
)
from robot_motion.deploy.observation_view import PolicyObservationView
from robot_motion.deploy.policy_worker import (
    BoundedPolicyInferenceWorker,
    PolicyInferenceWorkItem,
)
from robot_motion.deploy.rtc_runtime import (
    RtcPolicyShadowConfig,
    RtcPolicyShadowFrontend,
)
from robot_motion.deploy.rtc import RtcModelContinuationRuntimeSpec
from robot_motion.deploy.runtime import (
    PolicyDeployRuntime,
    PolicyDeployRuntimeConfig,
)
from robot_motion.deploy.observation import build_deployment_observation
from robot_motion.deploy.rtc import (
    RTC_CONDITIONING_MODE_PREFIX_V1,
    RTC_CONDITIONING_MODE_UNCONDITIONED_BOOTSTRAP_V1,
    RTC_PREFIX_SCHEDULE_HARD_DELAY_PREFIX_V1,
    CanonicalRtcConditioningBundle,
    RtcContinuationSelectionOutcome,
    RtcContinuationSelectionSpec,
    RtcContinuationTimeline,
    build_canonical_rtc_conditioning_bundle,
    select_rtc_plan_continuation,
)


def _continuation_contract(
    *,
    bootstrap: bool = False,
    action_codec_hash: str = "codec:test",
) -> ModelContinuationDataContract:
    return ModelContinuationDataContract(
        continuation_mode=MODEL_CONTINUATION_FLOW_INPAINT_V1,
        continuation_implementation_id="fake.flow-inpaint.v1",
        continuation_implementation_parameters_hash="sha256:implementation",
        prediction_horizon_steps=6,
        minimum_forecast_delay_steps=1,
        maximum_forecast_delay_steps=4,
        minimum_prefix_steps=0 if bootstrap else 1,
        maximum_prefix_steps=3,
        model_action_dim=7,
        model_action_discrete_dim=0,
        component_scope=("arm_pose:right",),
        supported_action_pose_modes=("absolute",),
        sampler_constraints=(MODEL_CONTINUATION_SAMPLER_CLAMPED_V1,),
        training_prefix_source_kinds=(
            TRAINING_PREFIX_TRAJECTORY_OVERLAP_TEACHER_V1,
        ),
        runtime_prefix_source_roles=(
            RUNTIME_PREFIX_PRODUCER_LOCAL_OVERLAP_V1,
        ),
        action_codec_hash=action_codec_hash,
        layout_hash="layout:test",
        frame_contract_hash="frame:test",
        normalization_hash="normalization:test",
        training_projection_implementation_id="teacher.v1",
        training_projection_parameters_hash="sha256:projection",
        supports_unconditioned_bootstrap=bootstrap,
    )


def _representation_context(
    *,
    relative: bool = False,
) -> PolicyActionRepresentationContext:
    return PolicyActionRepresentationContext(
        source_pose_mode="relative" if relative else "absolute",
        observation_cut_id="observation:new",
        observation_contract_hash="observation-contract:test",
        observation_timestamp=10.0,
        observation_step_index=100,
        active_sides=("right",),
        anchor_policy=(
            ACTION_ANCHOR_POLICY_REQUEST_ACTUAL_TCP_FIXED_V1
            if relative
            else ACTION_ANCHOR_POLICY_NONE_V1
        ),
        arm_reference_frame_by_side={"right": "base"},
        arm_controlled_frame_by_side={"right": "tcp"},
        arm_anchor_pose_by_side=(
            {
                "right": np.asarray(
                    [0.4, -0.1, 0.3, 0.0, 0.0, 0.0, 1.0],
                    dtype=np.float32,
                )
            }
            if relative
            else {}
        ),
    )


def _canonical_action(index: int):
    action = RobotAction(
        action_space="single_arm_ee_pose_v1",
        timestamp=10.0 + index * 0.1,
        arm_pose_by_side={
            "right": np.asarray(
                [
                    0.4 + index * 0.01,
                    -0.1,
                    0.3,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                dtype=np.float32,
            )
        },
        source="rtc-conditioning-test",
    )
    return build_canonical_policy_action(
        action,
        source_pose_mode="absolute",
        observation_cut_id="observation:old",
        source_action_hash=f"source-action:{index}",
        action_codec_hash="codec:test",
        frame_contract_hash="frame:test",
        layout_hash="layout:test",
        normalization_hash="normalization:test",
        conversion_contract_hash="conversion:absolute:test",
    )


def _keyframe(index: int) -> TimedPolicyActionKeyframe:
    target = 10.0 + index * 0.1
    return TimedPolicyActionKeyframe(
        target_effective_at=target,
        effective_until=target + 0.1,
        effective_clock_domain_id="clock:test",
        action_period_sec=0.1,
        decoded_absolute_action=None,
        canonical_action=_canonical_action(index),
        request_id="request:old",
        inference_id="inference:old",
        policy_output_id="output:old",
        chunk_id=1,
        chunk_keyframe_index=index,
        chunk_size=6,
        observation_cut_id="observation:old",
        producer_epoch=2,
        policy_node_id="policy:test",
        artifact_hash="artifact:test",
        produced_at=9.8,
        schedule_spec_id="schedule:test",
        schedule_spec_hash="schedule-hash:test",
        admitted_plan_generation=3,
        state=POLICY_PLAN_KEYFRAME_MUTABLE,
        valid_until=target + 1.0,
    )


def _plan(indices: tuple[int, ...]) -> TimedPolicyActionPlanSnapshot:
    return TimedPolicyActionPlanSnapshot(
        plan_id="plan:test",
        session_id="session:test",
        control_domain_id="domain:test",
        producer_id="producer:test",
        producer_epoch=2,
        policy_node_id="policy:test",
        clock_domain_id="clock:test",
        generation=3,
        cursor_at=10.0,
        created_at=9.9,
        keyframes=tuple(_keyframe(index) for index in indices),
    )


def _timeline() -> RtcContinuationTimeline:
    return RtcContinuationTimeline(
        clock_domain_id="clock:test",
        generation_start_at=10.0,
        first_target_effective_at=10.3,
        action_period_sec=0.1,
        prediction_horizon_steps=6,
        forecast_delay_steps=3,
    )


def _selection(indices: tuple[int, ...]):
    return select_rtc_plan_continuation(
        _plan(indices),
        timeline=_timeline(),
        spec=RtcContinuationSelectionSpec(
            prefix_horizon_steps=3,
            minimum_selected_steps=1,
        ),
    )


def _bundle(
    indices: tuple[int, ...] = (1, 2),
    *,
    bootstrap: bool = False,
    relative: bool = False,
    schedule: str | None = None,
    continuation_contract: ModelContinuationContract | None = None,
) -> CanonicalRtcConditioningBundle:
    kwargs = {}
    if schedule is not None:
        kwargs["prefix_schedule_semantics"] = schedule
    return build_canonical_rtc_conditioning_bundle(
        runtime_session_id="session:test",
        control_domain_id="domain:test",
        artifact_hash="artifact:test",
        timeline=_timeline(),
        selection=_selection(indices),
        action_representation_context=_representation_context(
            relative=relative,
        ),
        continuation_contract=(
            _continuation_contract(bootstrap=bootstrap)
            if continuation_contract is None
            else continuation_contract
        ),
        **kwargs,
    )


def _bundle_for_delay_and_prefix(
    *,
    forecast_delay_steps: int,
    prefix_steps: int,
) -> CanonicalRtcConditioningBundle:
    """Build one exact K<=D<H matrix case without changing production code."""

    timeline = replace(
        _timeline(),
        first_target_effective_at=(
            _timeline().generation_start_at
            + forecast_delay_steps * _timeline().action_period_sec
        ),
        forecast_delay_steps=forecast_delay_steps,
    )
    indices = tuple(range(prefix_steps))
    selection = select_rtc_plan_continuation(
        _plan(indices),
        timeline=timeline,
        spec=RtcContinuationSelectionSpec(
            prefix_horizon_steps=forecast_delay_steps,
            minimum_selected_steps=1,
        ),
    )
    return build_canonical_rtc_conditioning_bundle(
        runtime_session_id="session:test",
        control_domain_id="domain:test",
        artifact_hash="artifact:test",
        timeline=timeline,
        selection=selection,
        action_representation_context=_representation_context(),
        continuation_contract=replace(
            _continuation_contract(bootstrap=prefix_steps == 0),
            maximum_prefix_steps=4,
        ),
    )


def _request(bundle: CanonicalRtcConditioningBundle) -> PolicyInferenceRequest:
    runtime_contract = normalize_model_continuation_contract(
        bundle.continuation_contract
    ).runtime
    continuation_slice = bundle.selection.continuation_slice
    if continuation_slice is None:
        source_plan = _plan(())
        producer_id = source_plan.producer_id
        producer_epoch = source_plan.producer_epoch
        policy_node_id = source_plan.policy_node_id
        source_plan_id = source_plan.plan_id
        source_plan_generation = source_plan.generation
        source_plan_hash = source_plan.semantic_content_hash
    else:
        producer_id = continuation_slice.producer_id
        producer_epoch = continuation_slice.producer_epoch
        policy_node_id = continuation_slice.policy_node_id
        source_plan_id = continuation_slice.source_plan_id
        source_plan_generation = continuation_slice.source_plan_generation
        source_plan_hash = continuation_slice.source_plan_hash
    return PolicyInferenceRequest(
        request_id="request:new",
        inference_id="inference:new",
        rtc_mode=RTC_FLOW_INPAINT_MODE_V1,
        session_id=bundle.runtime_session_id,
        control_domain_id=bundle.control_domain_id,
        producer_id=producer_id,
        producer_epoch=producer_epoch,
        policy_node_id=policy_node_id,
        artifact_hash=bundle.artifact_hash,
        observation_cut_id=(
            bundle.action_representation_context.observation_cut_id
        ),
        observation_timestamp=(
            bundle.action_representation_context.observation_timestamp
        ),
        observation_step_index=(
            bundle.action_representation_context.observation_step_index
        ),
        triggered_at=10.0,
        admitted_at=10.0,
        first_target_effective_at=bundle.timeline.first_target_effective_at,
        clock_domain_id=bundle.timeline.clock_domain_id,
        action_period_sec=bundle.timeline.action_period_sec,
        prediction_horizon_steps=bundle.timeline.prediction_horizon_steps,
        minimum_execution_horizon_steps=1,
        source_plan_id=source_plan_id,
        source_plan_generation=source_plan_generation,
        source_plan_hash=source_plan_hash,
        source_plan_cursor_at=10.0,
        producer_plan_committed_through=None,
        forecast_delay_steps=bundle.timeline.forecast_delay_steps,
        delay_estimator_spec_hash="delay:test",
        action_codec_hash=runtime_contract.action_codec_hash,
        frame_contract_hash=runtime_contract.frame_contract_hash,
        layout_hash=runtime_contract.layout_hash,
        normalization_hash=runtime_contract.normalization_hash,
        valid_until=11.0,
        action_representation_context=(
            bundle.action_representation_context
        ),
        conditioning_binding_hash=bundle.conditioning_binding_hash,
    )


def _work_item(
    bundle: CanonicalRtcConditioningBundle,
) -> PolicyInferenceWorkItem:
    request = _request(bundle)
    observation = PolicyObservationView(
        contract_id="observation-view:test",
        contract_hash=(
            bundle.action_representation_context.observation_contract_hash
        ),
        payload={
            "timestamp": request.observation_timestamp,
            "step_index": request.observation_step_index,
        },
    )
    return PolicyInferenceWorkItem(
        worker_id="worker:test",
        worker_session_id="worker-session:test",
        submit_sequence=0,
        request=request,
        actor_observation=observation,
        submitted_at=request.admitted_at,
        conditioning_bundle=bundle,
    )


def test_bundle_binds_exact_actions_and_full_generation_mask() -> None:
    bundle = _bundle()

    assert bundle.selected_generation_indices == (1, 2)
    assert bundle.prefix_mask == (0.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    assert len(bundle.canonical_actions) == 2
    assert bundle.conditioning_binding_hash.startswith("sha256:")
    assert bundle.conditioning_binding_hash == _bundle().conditioning_binding_hash

    changed = _bundle(indices=(0, 1))
    assert (
        changed.canonical_action_content_hash
        != bundle.canonical_action_content_hash
    )
    assert changed.conditioning_binding_hash != bundle.conditioning_binding_hash


def test_inference_only_v2_uses_existing_generic_conditioning_path() -> None:
    legacy = _continuation_contract()
    contract = ModelContinuationDataContractV2(
        conditioning_origin=MODEL_CONTINUATION_ORIGIN_INFERENCE_ONLY_V1,
        runtime=ModelContinuationRuntimeContractV2.from_v1(legacy),
    )
    bundle = _bundle(continuation_contract=contract)
    encoded = encode_rtc_conditioning_bundle(
        _FakeConditioningCodec(),
        bundle,
    )

    assert bundle.continuation_contract is contract
    assert encoded.external_prefix.shape == (6, 7)
    assert encoded.prefix_mask.tolist() == list(bundle.prefix_mask)
    assert _request(bundle).conditioning_binding_hash == (
        bundle.conditioning_binding_hash
    )


def test_bundle_rejects_mask_tampering_and_identity_mismatches() -> None:
    bundle = _bundle()
    with pytest.raises(ValueError, match="exactly the selected"):
        replace(
            bundle,
            prefix_mask=(1.0, 1.0, 1.0, 0.0, 0.0, 0.0),
        )
    with pytest.raises(ValueError, match="artifact identity mismatch"):
        replace(bundle, artifact_hash="artifact:other")
    with pytest.raises(ValueError, match="action_codec_hash mismatch"):
        replace(
            bundle,
            continuation_contract=_continuation_contract(
                action_codec_hash="codec:other",
            ),
        )


def test_c1_bundle_and_runtime_reject_non_flow_continuation_modes() -> None:
    native_contract = replace(
        _continuation_contract(),
        continuation_mode=MODEL_CONTINUATION_NATIVE_V1,
    )

    with pytest.raises(ValueError, match="requires flow_inpaint"):
        _bundle(continuation_contract=native_contract)
    with pytest.raises(ValueError, match="requires flow_inpaint"):
        RtcModelContinuationRuntimeSpec(
            continuation_contract=native_contract,
        )


def test_bootstrap_is_explicit_all_zero_and_artifact_gated() -> None:
    bundle = _bundle(indices=(), bootstrap=True)

    assert (
        bundle.conditioning_mode
        == RTC_CONDITIONING_MODE_UNCONDITIONED_BOOTSTRAP_V1
    )
    assert bundle.selected_generation_indices == ()
    assert bundle.prefix_mask == (0.0,) * 6

    with pytest.raises(ValueError, match="does not support"):
        replace(
            bundle,
            continuation_contract=_continuation_contract(bootstrap=False),
        )
    with pytest.raises(ValueError, match="all-zero"):
        replace(
            bundle,
            prefix_mask=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )


def test_non_bootstrap_selector_rejection_cannot_be_encoded_as_empty_prefix() -> None:
    selected = _selection(())
    rejected = RtcContinuationSelectionOutcome(
        accepted=False,
        source_plan_hash=selected.source_plan_hash,
        timeline_hash=selected.timeline_hash,
        selection_spec_hash=selected.selection_spec_hash,
        continuation_slice=None,
        reasons=("source_plan_already_in_flight",),
    )
    with pytest.raises(ValueError, match="explicit selector bootstrap"):
        build_canonical_rtc_conditioning_bundle(
            runtime_session_id="session:test",
            control_domain_id="domain:test",
            artifact_hash="artifact:test",
            timeline=_timeline(),
            selection=rejected,
            action_representation_context=_representation_context(),
            continuation_contract=_continuation_contract(bootstrap=True),
        )


def test_hard_delay_schedule_requires_complete_delayed_prefix() -> None:
    complete = _bundle(
        indices=(0, 1, 2),
        schedule=RTC_PREFIX_SCHEDULE_HARD_DELAY_PREFIX_V1,
    )
    assert complete.prefix_mask == (1.0, 1.0, 1.0, 0.0, 0.0, 0.0)

    with pytest.raises(ValueError, match="every delayed generation step"):
        _bundle(
            indices=(1, 2),
            schedule=RTC_PREFIX_SCHEDULE_HARD_DELAY_PREFIX_V1,
        )


def test_relative_continuation_remains_fail_closed_in_c1_v1() -> None:
    with pytest.raises(ValueError, match="absolute actions only"):
        _bundle(relative=True)


def test_work_item_carries_bundle_outside_the_causal_request() -> None:
    bundle = _bundle()
    item = _work_item(bundle)

    assert item.request.conditioning_prefix is None
    assert item.conditioning_bundle is bundle
    assert (
        item.request.conditioning_binding_hash
        == bundle.conditioning_binding_hash
    )

    with pytest.raises(ValueError, match="bundle hash mismatch"):
        replace(
            item,
            request=replace(
                item.request,
                conditioning_binding_hash="sha256:wrong",
            ),
        )
    with pytest.raises(ValueError, match="requires its immutable bundle"):
        replace(item, conditioning_bundle=None)


def test_worker_uses_work_item_executor_only_for_bound_conditioning() -> None:
    class _TypedExecutor:
        def __init__(self) -> None:
            self.legacy_calls = 0
            self.work_item_calls = 0

        def __call__(self, request, actor_observation):
            self.legacy_calls += 1
            raise AssertionError("legacy executor path must not be used")

        def execute_work_item(self, work_item):
            self.work_item_calls += 1
            assert work_item.conditioning_bundle is not None
            raise RuntimeError("typed-work-item-path")

    executor = _TypedExecutor()
    worker = BoundedPolicyInferenceWorker(
        worker_id="worker:test",
        worker_session_id="worker-session:test",
        executor=executor,
        idle_poll_sec=0.001,
    )
    worker.start()
    try:
        submission = worker.submit(_work_item(_bundle()))
        assert submission.accepted
        deadline = time.monotonic() + 1.0
        completion = None
        while completion is None and time.monotonic() < deadline:
            completion = worker.poll_completion()
            if completion is None:
                time.sleep(0.001)
        assert completion is not None
        assert completion.result.status == "fault"
        assert completion.result.fault_detail == "typed-work-item-path"
        assert executor.work_item_calls == 1
        assert executor.legacy_calls == 0
    finally:
        worker.stop(timeout_sec=1.0)


class _FakeConditioningCodec:
    def __init__(self) -> None:
        self.identity = RtcConditioningCodecIdentity(
            implementation_id="fake.conditioning-codec.v1",
            implementation_parameters_hash="sha256:fake-codec",
            action_codec_hash="codec:test",
            layout_hash="layout:test",
            frame_contract_hash="frame:test",
            normalization_hash="normalization:test",
            artifact_action_dim=7,
            external_action_dim=7,
        )

    def canonical_to_artifact_actions(self, actions, *, bundle):
        return np.stack(
            [
                np.asarray(
                    action.action.arm_pose_by_side["right"],
                    dtype=np.float32,
                )
                for action in actions
            ],
            axis=0,
        ) if actions else np.zeros((0, 7), dtype=np.float32)

    def artifact_to_canonical_actions(self, artifact_actions, *, bundle):
        return bundle.canonical_actions

    def artifact_to_external_actions(self, artifact_actions, *, bundle):
        return np.asarray(artifact_actions, dtype=np.float32) * 2.0 + 1.0

    def external_to_artifact_actions(self, external_actions, *, bundle):
        return (
            np.asarray(external_actions, dtype=np.float32) - 1.0
        ) / 2.0


def _client_transform_capability(bundle, codec):
    runtime_contract = normalize_model_continuation_contract(
        bundle.continuation_contract
    ).runtime
    return rtc_model_continuation_capability_from_loaded_codec(
        bundle=bundle,
        codec_identity=codec.identity,
        sampler_constraint=(
            runtime_contract.sampler_constraints[0]
        ),
        conditioning_transform_owner=(
            RTC_CONDITIONING_TRANSFORM_CLIENT_V1
        ),
        internal_sampler_action_dim=codec.identity.external_action_dim,
    )


def _consumption_receipt(
    envelope,
    *,
    external_prefix=None,
    internal_prefix=None,
    mask=None,
):
    external = (
        envelope.encoded_conditioning.external_prefix
        if external_prefix is None
        else external_prefix
    )
    internal = external if internal_prefix is None else internal_prefix
    consumed_mask = (
        envelope.encoded_conditioning.prefix_mask if mask is None else mask
    )
    return build_rtc_conditioning_consumption_receipt_v3(
        envelope,
        consumed_external_prefix=external,
        consumed_internal_prefix=internal,
        consumed_mask=consumed_mask,
    )


def _clamp_selected_output(envelope, actions):
    """Make a fake output satisfy the declared exact-clamp capability."""

    output = np.asarray(actions, dtype=np.float32).copy()
    selected = (
        envelope.encoded_conditioning.receipt.selected_generation_indices
    )
    if selected:
        indices = np.asarray(selected, dtype=np.intp)
        output[indices] = (
            envelope.encoded_conditioning.external_prefix[indices]
        )
    return output


def _valid_v3_exchange():
    bundle = _bundle()
    request = _request(bundle)
    codec = _FakeConditioningCodec()
    encoded = encode_rtc_conditioning_bundle(codec, bundle)
    capability = _client_transform_capability(bundle, codec)
    envelope = build_rtc_conditioned_inference_envelope_v3(
        request=request,
        bundle=bundle,
        capability=capability,
        encoded_conditioning=encoded,
        model_input_hash="model-input:valid-exchange",
    )
    response = RtcConditionedModelResponseV3(
        envelope_hash=envelope.semantic_content_hash,
        request_hash=request.semantic_content_hash,
        capability_hash=capability.semantic_content_hash,
        model_output_id="model-output:valid-exchange",
        model_space_actions=_clamp_selected_output(
            envelope,
            np.zeros((6, 7), dtype=np.float32),
        ),
        inference_started_at=10.0,
        inference_completed_at=10.1,
        result_received_at=10.1,
        consumption_receipt=_consumption_receipt(envelope),
    )
    return bundle, request, codec, capability, envelope, response


def test_v3_capability_binding_uses_one_selected_schedule_identity() -> None:
    bundle = _bundle()
    codec = _FakeConditioningCodec()
    capability = _client_transform_capability(bundle, codec)

    validate_rtc_model_continuation_capability_v3_binding(
        capability,
        continuation_contract=bundle.continuation_contract,
        artifact_hash=bundle.artifact_hash,
        codec_identity_hash=codec.identity.semantic_content_hash,
        selected_prefix_schedule=bundle.prefix_schedule_semantics,
    )
    assert (
        capability.selected_prefix_schedule
        == bundle.prefix_schedule_semantics
    )
    assert "prefix_schedule_semantics" not in capability.__dataclass_fields__
    with pytest.raises(
        ValueError,
        match="selected_prefix_schedule mismatch",
    ):
        validate_rtc_model_continuation_capability_v3_binding(
            replace(
                capability,
                selected_prefix_schedule="other_schedule_v1",
            ),
            continuation_contract=bundle.continuation_contract,
            artifact_hash=bundle.artifact_hash,
            codec_identity_hash=codec.identity.semantic_content_hash,
            selected_prefix_schedule=bundle.prefix_schedule_semantics,
        )


def test_injected_codec_encodes_only_selected_rows_with_exact_receipt() -> None:
    bundle = _bundle()
    codec = _FakeConditioningCodec()
    encoded = encode_rtc_conditioning_bundle(codec, bundle)
    artifact_rows = codec.canonical_to_artifact_actions(
        bundle.canonical_actions,
        bundle=bundle,
    )
    offline_external_rows = encode_external_continuation_artifact_actions(
        codec,
        artifact_rows,
        bundle=bundle,
    )

    assert encoded.external_prefix.shape == (6, 7)
    assert encoded.prefix_mask.tolist() == list(bundle.prefix_mask)
    assert np.array_equal(
        encoded.external_prefix[[1, 2]],
        offline_external_rows,
    )
    assert np.all(encoded.external_prefix[[0, 3, 4, 5]] == 0.0)
    assert (
        encoded.receipt.conditioning_binding_hash
        == bundle.conditioning_binding_hash
    )
    assert (
        encoded.receipt.canonical_action_content_hash
        == bundle.canonical_action_content_hash
    )
    assert not encoded.external_prefix.flags.writeable
    assert not encoded.prefix_mask.flags.writeable


def test_codec_identity_and_round_trip_fail_closed() -> None:
    bundle = _bundle()
    wrong_identity = _FakeConditioningCodec()
    wrong_identity.identity = replace(
        wrong_identity.identity,
        normalization_hash="normalization:other",
    )
    with pytest.raises(ValueError, match="normalization_hash mismatch"):
        encode_rtc_conditioning_bundle(wrong_identity, bundle)

    class _BrokenRoundTrip(_FakeConditioningCodec):
        def external_to_artifact_actions(self, external_actions, *, bundle):
            return super().external_to_artifact_actions(
                external_actions,
                bundle=bundle,
            ) + 0.1

    with pytest.raises(ValueError, match="round-trip exceeds tolerance"):
        encode_rtc_conditioning_bundle(_BrokenRoundTrip(), bundle)


def test_cross_repository_conditioning_array_hash_golden_values() -> None:
    assert rtc_float32_array_hash(
        np.asarray([[1.0, -2.5], [0.0, 3.25]], dtype=np.float64)
    ) == (
        "sha256:"
        "1903cfa78cbe0da239a48268c0f4858ecc9ac34709999a00607440f208fba4e4"
    )
    assert rtc_binary_mask_hash(
        np.asarray([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
    ) == (
        "sha256:"
        "f92310d796cb04d1bd1c32bda0712fa526e25a23a6e42924a19a599fad0a1af5"
    )


def test_conditioning_array_hashes_canonicalize_numeric_representation() -> None:
    base = np.asarray([[1.0, -2.5], [0.0, 3.25]], dtype=np.float32)
    padded = np.zeros((2, 4), dtype=np.float32)
    padded[:, ::2] = base
    noncontiguous = padded[:, ::2]
    assert not noncontiguous.flags.c_contiguous

    expected = rtc_float32_array_hash(base)
    assert rtc_float32_array_hash(base.astype(np.float64)) == expected
    assert rtc_float32_array_hash(base.astype(">f4")) == expected
    assert rtc_float32_array_hash(noncontiguous) == expected
    assert rtc_float32_array_hash(base.reshape(-1)) != expected

    mask_expected = rtc_binary_mask_hash(
        np.asarray([0, 1, 1, 0], dtype=np.uint8)
    )
    assert rtc_binary_mask_hash([False, True, True, False]) == mask_expected
    assert rtc_binary_mask_hash(
        np.asarray([0.0, 1.0, 1.0, 0.0], dtype=">f4")
    ) == mask_expected


@pytest.mark.parametrize(
    "bad_values",
    (
        np.asarray([0.0, np.nan], dtype=np.float32),
        np.asarray([0.0, np.inf], dtype=np.float32),
    ),
)
def test_conditioning_array_hashes_reject_nonfinite_values(
    bad_values: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="finite"):
        rtc_float32_array_hash(bad_values)
    with pytest.raises(ValueError, match="finite"):
        rtc_binary_mask_hash(bad_values)


@pytest.mark.parametrize(
    "bad_mask",
    (
        np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        np.asarray([[0.0, 1.0]], dtype=np.float32),
    ),
)
def test_binary_mask_hash_rejects_fractional_or_nonvector_masks(
    bad_mask: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="one-dimensional|only 0/1"):
        rtc_binary_mask_hash(bad_mask)


def test_encoded_and_response_arrays_detach_inputs_and_remain_immutable() -> None:
    _bundle_value, _request_value, _codec, _capability, envelope, response = (
        _valid_v3_exchange()
    )
    prefix_input = np.asarray(
        envelope.encoded_conditioning.external_prefix,
        dtype=np.float32,
    ).copy()
    mask_input = np.asarray(
        envelope.encoded_conditioning.prefix_mask,
        dtype=np.float32,
    ).copy()
    detached_encoded = RtcEncodedConditioning(
        external_prefix=prefix_input,
        prefix_mask=mask_input,
        receipt=envelope.encoded_conditioning.receipt,
    )
    response_input = np.asarray(
        response.model_space_actions,
        dtype=np.float32,
    ).copy()
    detached_response = replace(
        response,
        model_space_actions=response_input,
    )
    encoded_hash = detached_encoded.receipt.semantic_content_hash
    response_hash = detached_response.semantic_content_hash

    prefix_input[:] = 123.0
    mask_input[:] = 0.0
    response_input[:] = 456.0

    assert (
        detached_encoded.receipt.semantic_content_hash
        == encoded_hash
    )
    assert detached_response.semantic_content_hash == response_hash
    assert not np.any(detached_encoded.external_prefix == 123.0)
    assert not np.any(detached_response.model_space_actions == 456.0)
    with pytest.raises(ValueError):
        detached_encoded.external_prefix[0, 0] = 1.0
    with pytest.raises(ValueError):
        detached_encoded.prefix_mask[0] = 1.0
    with pytest.raises(ValueError):
        detached_response.model_space_actions[0, 0] = 1.0


@pytest.mark.parametrize(
    "bad_indices",
    (
        (-1,),
        (1, 1),
        (True,),
        (1.5,),
    ),
)
def test_encoding_receipt_rejects_nonexact_or_nonmonotonic_indices(
    bad_indices: tuple[object, ...],
) -> None:
    encoded = encode_rtc_conditioning_bundle(
        _FakeConditioningCodec(),
        _bundle(),
    )
    with pytest.raises(ValueError, match="generation_indices"):
        replace(
            encoded.receipt,
            selected_generation_indices=bad_indices,
        )


def test_v3_fake_backend_receipts_actual_conditioning_consumption() -> None:
    bundle = _bundle(indices=(0, 1, 2))
    request = _request(bundle)
    codec = _FakeConditioningCodec()
    encoded = encode_rtc_conditioning_bundle(codec, bundle)
    capability = _client_transform_capability(bundle, codec)
    envelope = build_rtc_conditioned_inference_envelope_v3(
        request=request,
        bundle=bundle,
        capability=capability,
        encoded_conditioning=encoded,
        model_input_hash="model-input:test",
    )

    class _FakeBackend:
        rtc_continuation_capability = capability

        def infer_conditioned_rtc_v3(self, envelope, model_input):
            assert model_input == {"observation": "fake"}
            consumed_prefix = np.asarray(
                envelope.encoded_conditioning.external_prefix,
                dtype=np.float32,
            ).copy()
            consumed_mask = np.asarray(
                envelope.encoded_conditioning.prefix_mask,
                dtype=np.float32,
            ).copy()
            assert np.count_nonzero(consumed_mask) == 3
            receipt = _consumption_receipt(
                envelope,
                external_prefix=consumed_prefix,
                internal_prefix=consumed_prefix,
                mask=consumed_mask,
            )
            return RtcConditionedModelResponseV3(
                envelope_hash=envelope.semantic_content_hash,
                request_hash=envelope.request.semantic_content_hash,
                capability_hash=(
                    envelope.capability.semantic_content_hash
                ),
                model_output_id="model-output:test",
                model_space_actions=consumed_prefix,
                inference_started_at=10.0,
                inference_completed_at=10.1,
                result_received_at=10.1,
                consumption_receipt=receipt,
            )

    backend = _FakeBackend()
    assert isinstance(backend, RtcConditionedModelBackendV3)
    response = backend.infer_conditioned_rtc_v3(
        envelope,
        {"observation": "fake"},
    )
    validation = validate_rtc_conditioned_response_v3(
        envelope=envelope,
        response=response,
    )

    assert validation.accepted
    assert validation.reasons == ()
    assert (
        response.consumption_receipt.consumed_generation_indices
        == bundle.selected_generation_indices
    )
    assert response.model_space_actions.shape == (6, 7)


def test_v3_exact_clamp_rejects_output_prefix_mismatch() -> None:
    bundle = _bundle(indices=(0, 1, 2))
    request = _request(bundle)
    codec = _FakeConditioningCodec()
    encoded = encode_rtc_conditioning_bundle(codec, bundle)
    capability = _client_transform_capability(bundle, codec)
    envelope = build_rtc_conditioned_inference_envelope_v3(
        request=request,
        bundle=bundle,
        capability=capability,
        encoded_conditioning=encoded,
        model_input_hash="model-input:clamp-mismatch",
    )
    actions = np.asarray(encoded.external_prefix, dtype=np.float32).copy()
    actions[1, 0] += 0.25
    response = RtcConditionedModelResponseV3(
        envelope_hash=envelope.semantic_content_hash,
        request_hash=request.semantic_content_hash,
        capability_hash=capability.semantic_content_hash,
        model_output_id="model-output:clamp-mismatch",
        model_space_actions=actions,
        inference_started_at=10.0,
        inference_completed_at=10.1,
        result_received_at=10.1,
        consumption_receipt=_consumption_receipt(envelope),
    )

    validation = validate_rtc_conditioned_response_v3(
        envelope=envelope,
        response=response,
    )
    assert not validation.accepted
    assert validation.reasons == ("clamped_output_prefix_mismatch",)


def test_v3_weighted_sampler_does_not_claim_exact_output_clamp() -> None:
    contract = replace(
        _continuation_contract(),
        sampler_constraints=(MODEL_CONTINUATION_SAMPLER_WEIGHTED_V1,),
    )
    bundle = _bundle(
        indices=(0, 1, 2),
        continuation_contract=contract,
    )
    request = _request(bundle)
    codec = _FakeConditioningCodec()
    encoded = encode_rtc_conditioning_bundle(codec, bundle)
    capability = _client_transform_capability(bundle, codec)
    envelope = build_rtc_conditioned_inference_envelope_v3(
        request=request,
        bundle=bundle,
        capability=capability,
        encoded_conditioning=encoded,
        model_input_hash="model-input:weighted",
    )
    response = RtcConditionedModelResponseV3(
        envelope_hash=envelope.semantic_content_hash,
        request_hash=request.semantic_content_hash,
        capability_hash=capability.semantic_content_hash,
        model_output_id="model-output:weighted",
        model_space_actions=np.zeros((6, 7), dtype=np.float32),
        inference_started_at=10.0,
        inference_completed_at=10.1,
        result_received_at=10.1,
        consumption_receipt=_consumption_receipt(envelope),
    )

    validation = validate_rtc_conditioned_response_v3(
        envelope=envelope,
        response=response,
    )
    assert validation.accepted
    assert validation.reasons == ()


def test_v3_provider_transform_binds_external_and_internal_prefixes() -> None:
    bundle = _bundle(indices=(0, 1, 2))
    request = _request(bundle)
    codec = _FakeConditioningCodec()
    encoded = encode_rtc_conditioning_bundle(codec, bundle)
    capability = rtc_model_continuation_capability_from_loaded_codec(
        bundle=bundle,
        codec_identity=codec.identity,
        sampler_constraint=(
            bundle.continuation_contract.sampler_constraints[0]
        ),
        conditioning_transform_owner=(
            RTC_CONDITIONING_TRANSFORM_PROVIDER_V1
        ),
        internal_sampler_action_dim=9,
    )
    envelope = build_rtc_conditioned_inference_envelope_v3(
        request=request,
        bundle=bundle,
        capability=capability,
        encoded_conditioning=encoded,
        model_input_hash="model-input:provider-transform",
    )
    external = np.asarray(encoded.external_prefix, dtype=np.float32).copy()
    internal = np.zeros((6, 9), dtype=np.float32)
    internal[:, :7] = external * 0.5
    receipt = _consumption_receipt(
        envelope,
        external_prefix=external,
        internal_prefix=internal,
    )
    response = RtcConditionedModelResponseV3(
        envelope_hash=envelope.semantic_content_hash,
        request_hash=request.semantic_content_hash,
        capability_hash=capability.semantic_content_hash,
        model_output_id="model-output:provider-transform",
        model_space_actions=external,
        inference_started_at=10.0,
        inference_completed_at=10.1,
        result_received_at=10.1,
        consumption_receipt=receipt,
    )

    validation = validate_rtc_conditioned_response_v3(
        envelope=envelope,
        response=response,
    )
    assert validation.accepted
    assert (
        receipt.consumed_external_prefix_hash
        == encoded.receipt.external_prefix_hash
    )
    assert (
        receipt.consumed_internal_prefix_hash
        != receipt.consumed_external_prefix_hash
    )


def test_v3_rejects_provider_receipt_for_different_external_prefix() -> None:
    bundle = _bundle(indices=(0, 1, 2))
    request = _request(bundle)
    codec = _FakeConditioningCodec()
    encoded = encode_rtc_conditioning_bundle(codec, bundle)
    capability = _client_transform_capability(bundle, codec)
    envelope = build_rtc_conditioned_inference_envelope_v3(
        request=request,
        bundle=bundle,
        capability=capability,
        encoded_conditioning=encoded,
        model_input_hash="model-input:wrong-external",
    )
    wrong_external = np.asarray(
        encoded.external_prefix,
        dtype=np.float32,
    ).copy()
    wrong_external[0, 0] += 1.0
    receipt = _consumption_receipt(
        envelope,
        external_prefix=wrong_external,
        internal_prefix=wrong_external,
    )
    response = RtcConditionedModelResponseV3(
        envelope_hash=envelope.semantic_content_hash,
        request_hash=request.semantic_content_hash,
        capability_hash=capability.semantic_content_hash,
        model_output_id="model-output:wrong-external",
        model_space_actions=_clamp_selected_output(
            envelope,
            np.zeros((6, 7), dtype=np.float32),
        ),
        inference_started_at=10.0,
        inference_completed_at=10.1,
        result_received_at=10.1,
        consumption_receipt=receipt,
    )

    validation = validate_rtc_conditioned_response_v3(
        envelope=envelope,
        response=response,
    )
    assert not validation.accepted
    assert validation.reasons == (
        "consumed_external_prefix_hash_mismatch",
    )


def test_v3_provider_bootstrap_consumes_explicit_zero_mask() -> None:
    bundle = _bundle(indices=(), bootstrap=True)
    request = _request(bundle)
    codec = _FakeConditioningCodec()
    encoded = encode_rtc_conditioning_bundle(codec, bundle)
    capability = rtc_model_continuation_capability_from_loaded_codec(
        bundle=bundle,
        codec_identity=codec.identity,
        sampler_constraint=(
            bundle.continuation_contract.sampler_constraints[0]
        ),
        conditioning_transform_owner=(
            RTC_CONDITIONING_TRANSFORM_PROVIDER_V1
        ),
        internal_sampler_action_dim=9,
    )
    envelope = build_rtc_conditioned_inference_envelope_v3(
        request=request,
        bundle=bundle,
        capability=capability,
        encoded_conditioning=encoded,
        model_input_hash="model-input:bootstrap",
    )
    internal = np.zeros((6, 9), dtype=np.float32)
    receipt = _consumption_receipt(
        envelope,
        internal_prefix=internal,
    )
    response = RtcConditionedModelResponseV3(
        envelope_hash=envelope.semantic_content_hash,
        request_hash=request.semantic_content_hash,
        capability_hash=capability.semantic_content_hash,
        model_output_id="model-output:bootstrap",
        model_space_actions=_clamp_selected_output(
            envelope,
            np.zeros((6, 7), dtype=np.float32),
        ),
        inference_started_at=10.0,
        inference_completed_at=10.1,
        result_received_at=10.1,
        consumption_receipt=receipt,
    )

    assert receipt.consumed_generation_indices == ()
    assert validate_rtc_conditioned_response_v3(
        envelope=envelope,
        response=response,
    ).accepted


def test_v3_response_rejects_false_consumption_identity_before_decode() -> None:
    bundle = _bundle()
    request = _request(bundle)
    codec = _FakeConditioningCodec()
    encoded = encode_rtc_conditioning_bundle(codec, bundle)
    capability = _client_transform_capability(bundle, codec)
    envelope = build_rtc_conditioned_inference_envelope_v3(
        request=request,
        bundle=bundle,
        capability=capability,
        encoded_conditioning=encoded,
        model_input_hash="model-input:test",
    )
    receipt = replace(
        _consumption_receipt(envelope),
        consumed_generation_indices=(0,),
    )
    response = RtcConditionedModelResponseV3(
        envelope_hash=envelope.semantic_content_hash,
        request_hash=request.semantic_content_hash,
        capability_hash=capability.semantic_content_hash,
        model_output_id="model-output:test",
        model_space_actions=_clamp_selected_output(
            envelope,
            np.zeros((6, 7), dtype=np.float32),
        ),
        inference_started_at=10.0,
        inference_completed_at=10.1,
        result_received_at=10.1,
        consumption_receipt=receipt,
    )
    validation = validate_rtc_conditioned_response_v3(
        envelope=envelope,
        response=response,
    )

    assert not validation.accepted
    assert validation.reasons == (
        "consumed_generation_indices_mismatch",
    )


@pytest.mark.parametrize(
    ("field_name", "replacement_value", "expected_reasons"),
    (
        (
            "envelope_hash",
            "sha256:wrong-envelope",
            ("receipt_envelope_hash_mismatch",),
        ),
        (
            "request_hash",
            "sha256:wrong-request",
            ("receipt_request_hash_mismatch",),
        ),
        (
            "capability_hash",
            "sha256:wrong-capability",
            ("receipt_capability_hash_mismatch",),
        ),
        (
            "conditioning_binding_hash",
            "sha256:wrong-binding",
            ("conditioning_binding_hash_mismatch",),
        ),
        (
            "encoding_receipt_hash",
            "sha256:wrong-encoding",
            ("encoding_receipt_hash_mismatch",),
        ),
        (
            "model_input_hash",
            "sha256:wrong-input",
            ("model_input_hash_mismatch",),
        ),
        (
            "consumed_external_prefix_hash",
            "sha256:wrong-external",
            (
                "client_transform_internal_prefix_hash_mismatch",
                "consumed_external_prefix_hash_mismatch",
            ),
        ),
        (
            "consumed_internal_prefix_hash",
            "sha256:wrong-internal",
            ("client_transform_internal_prefix_hash_mismatch",),
        ),
        (
            "consumed_mask_hash",
            "sha256:wrong-mask",
            ("consumed_mask_hash_mismatch",),
        ),
        (
            "consumed_generation_indices",
            (0,),
            ("consumed_generation_indices_mismatch",),
        ),
        (
            "sampler_constraint",
            MODEL_CONTINUATION_SAMPLER_WEIGHTED_V1,
            ("sampler_constraint_mismatch",),
        ),
        (
            "continuation_implementation_id",
            "fake.other-implementation.v1",
            ("continuation_implementation_id_mismatch",),
        ),
        (
            "continuation_implementation_parameters_hash",
            "sha256:wrong-implementation",
            ("continuation_implementation_parameters_hash_mismatch",),
        ),
        (
            "effective_delay_steps",
            2,
            ("effective_delay_steps_mismatch",),
        ),
        (
            "prediction_horizon_steps",
            5,
            ("prediction_horizon_steps_mismatch",),
        ),
    ),
)
def test_v3_receipt_identity_mutations_fail_closed_with_exact_reasons(
    field_name: str,
    replacement_value: object,
    expected_reasons: tuple[str, ...],
) -> None:
    bundle = _bundle()
    request = _request(bundle)
    codec = _FakeConditioningCodec()
    encoded = encode_rtc_conditioning_bundle(codec, bundle)
    capability = _client_transform_capability(bundle, codec)
    envelope = build_rtc_conditioned_inference_envelope_v3(
        request=request,
        bundle=bundle,
        capability=capability,
        encoded_conditioning=encoded,
        model_input_hash="model-input:receipt-matrix",
    )
    receipt = replace(
        _consumption_receipt(envelope),
        **{field_name: replacement_value},
    )
    response = RtcConditionedModelResponseV3(
        envelope_hash=envelope.semantic_content_hash,
        request_hash=request.semantic_content_hash,
        capability_hash=capability.semantic_content_hash,
        model_output_id=f"model-output:receipt-matrix:{field_name}",
        model_space_actions=_clamp_selected_output(
            envelope,
            np.zeros((6, 7), dtype=np.float32),
        ),
        inference_started_at=10.0,
        inference_completed_at=10.1,
        result_received_at=10.1,
        consumption_receipt=receipt,
    )

    validation = validate_rtc_conditioned_response_v3(
        envelope=envelope,
        response=response,
    )

    assert not validation.accepted
    assert validation.reasons == expected_reasons


@pytest.mark.parametrize(
    ("external_shape", "internal_shape", "mask", "message"),
    (
        ((5, 7), (6, 7), np.zeros((6,), dtype=np.float32), "external"),
        ((6, 7), (5, 7), np.zeros((6,), dtype=np.float32), "internal"),
        ((6, 7), (6, 7), np.zeros((5,), dtype=np.float32), "prediction horizon"),
        (
            (6, 7),
            (6, 7),
            np.asarray([0.0, 1.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float32),
            "only 0/1",
        ),
    ),
)
def test_v3_consumption_receipt_builder_rejects_shape_and_mask_errors(
    external_shape: tuple[int, int],
    internal_shape: tuple[int, int],
    mask: np.ndarray,
    message: str,
) -> None:
    _bundle_value, _request_value, _codec, _capability, envelope, _response = (
        _valid_v3_exchange()
    )
    with pytest.raises(ValueError, match=message):
        build_rtc_conditioning_consumption_receipt_v3(
            envelope,
            consumed_external_prefix=np.zeros(
                external_shape,
                dtype=np.float32,
            ),
            consumed_internal_prefix=np.zeros(
                internal_shape,
                dtype=np.float32,
            ),
            consumed_mask=mask,
        )


@pytest.mark.parametrize(
    ("field_name", "replacement_value", "expected_reason"),
    (
        ("envelope_hash", "sha256:wrong-envelope", "envelope_hash_mismatch"),
        ("request_hash", "sha256:wrong-request", "request_hash_mismatch"),
        (
            "capability_hash",
            "sha256:wrong-capability",
            "capability_hash_mismatch",
        ),
    ),
)
def test_v3_response_top_level_identity_mutations_fail_closed(
    field_name: str,
    replacement_value: str,
    expected_reason: str,
) -> None:
    _bundle_value, _request_value, _codec, _capability, envelope, response = (
        _valid_v3_exchange()
    )
    validation = validate_rtc_conditioned_response_v3(
        envelope=envelope,
        response=replace(
            response,
            **{field_name: replacement_value},
        ),
    )
    assert not validation.accepted
    assert validation.reasons == (expected_reason,)


@pytest.mark.parametrize(
    ("shape", "expected_reason"),
    (
        ((5, 7), "model_output_horizon_mismatch"),
        ((6, 8), "model_output_dimension_mismatch"),
    ),
)
def test_v3_response_shape_mismatch_is_rejected_before_decode(
    shape: tuple[int, int],
    expected_reason: str,
) -> None:
    _bundle_value, _request_value, _codec, _capability, envelope, response = (
        _valid_v3_exchange()
    )
    validation = validate_rtc_conditioned_response_v3(
        envelope=envelope,
        response=replace(
            response,
            model_space_actions=np.zeros(shape, dtype=np.float32),
        ),
    )
    assert not validation.accepted
    assert validation.reasons == (expected_reason,)


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"model_space_actions": np.full((6, 7), np.nan)}, "finite"),
        (
            {
                "inference_started_at": 10.2,
                "inference_completed_at": 10.1,
            },
            "not ordered",
        ),
        (
            {
                "inference_completed_at": 10.2,
                "result_received_at": 10.1,
            },
            "not ordered",
        ),
    ),
)
def test_v3_response_constructor_rejects_nonfinite_or_regressing_facts(
    changes: dict[str, object],
    message: str,
) -> None:
    _bundle_value, _request_value, _codec, _capability, _envelope, response = (
        _valid_v3_exchange()
    )
    with pytest.raises(ValueError, match=message):
        replace(response, **changes)


def test_v3_response_validation_rejects_unknown_schema() -> None:
    _bundle_value, _request_value, _codec, _capability, envelope, response = (
        _valid_v3_exchange()
    )
    validation = validate_rtc_conditioned_response_v3(
        envelope=envelope,
        response=response,
    )
    with pytest.raises(ValueError, match="Unsupported.*validation"):
        replace(validation, schema="rtc_conditioned_response_validation_v999")


def test_v3_work_item_client_crops_once_and_binds_result_receipts() -> None:
    bundle = _bundle(indices=(0, 1, 2))
    request = _request(bundle)
    codec = _FakeConditioningCodec()
    capability = _client_transform_capability(bundle, codec)

    class _Backend:
        rtc_continuation_capability = capability
        last_envelope = None

        def infer_conditioned_rtc_v3(self, envelope, model_input):
            self.last_envelope = envelope
            actions = np.zeros((6, 7), dtype=np.float32)
            for index in range(6):
                actions[index] = np.asarray(
                    [
                        0.5 + index * 0.01,
                        -0.1,
                        0.3,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    dtype=np.float32,
                )
            actions = _clamp_selected_output(envelope, actions)
            return RtcConditionedModelResponseV3(
                envelope_hash=envelope.semantic_content_hash,
                request_hash=envelope.request.semantic_content_hash,
                capability_hash=(
                    envelope.capability.semantic_content_hash
                ),
                model_output_id="model-output:client",
                model_space_actions=actions,
                inference_started_at=10.0,
                inference_completed_at=10.1,
                result_received_at=10.1,
                consumption_receipt=(
                    _consumption_receipt(envelope)
                ),
            )

    decoded_rows = []

    def _decoder(model_actions, discrete_actions, decode_request):
        assert discrete_actions is None
        decoded_rows.append(np.asarray(model_actions).copy())
        return tuple(
            build_canonical_policy_action(
                RobotAction(
                    action_space="single_arm_ee_pose_v1",
                    timestamp=(
                        decode_request.first_target_effective_at
                        + index * decode_request.action_period_sec
                    ),
                    arm_pose_by_side={
                        "right": np.asarray(row, dtype=np.float32)
                    },
                    source="rtc-conditioned-client-test",
                ),
                source_pose_mode="absolute",
                observation_cut_id=decode_request.observation_cut_id,
                source_action_hash=f"model-row:{index}",
                action_codec_hash=decode_request.action_codec_hash,
                frame_contract_hash=decode_request.frame_contract_hash,
                layout_hash=decode_request.layout_hash,
                normalization_hash=decode_request.normalization_hash,
                conversion_contract_hash="conversion:absolute:test",
            )
            for index, row in enumerate(model_actions)
        )

    backend = _Backend()
    observed = []
    executor = RtcConditionedModelClientInferenceExecutorV3(
        client=RtcConditionedModelClientV3(
            backend=backend,
            codec=codec,
            decoder=_decoder,
        ),
        prepare_input=lambda request, observation: RtcPreparedModelInput(
            model_input={"observation": "fake"},
            model_input_hash="model-input:client",
        ),
        observe_bundle=observed.append,
    )
    item = _work_item(bundle)
    result = executor.execute_work_item(item)

    assert len(decoded_rows) == 1
    assert decoded_rows[0].shape == (3, 7)
    assert np.allclose(decoded_rows[0][:, 0], [0.53, 0.54, 0.55])
    assert result.source_chunk_size == 3
    assert result.effective_delay_steps == 3
    assert result.execution_horizon_steps == 3
    assert result.conditioning_binding_hash == bundle.conditioning_binding_hash
    assert result.conditioning_encoding_receipt_hash is not None
    assert result.conditioning_consumption_receipt_hash is not None
    assert len(observed) == 1
    assert backend.last_envelope is not None
    observed_bundle = observed[0]
    assert observed_bundle.generation_horizon_steps == 6
    assert observed_bundle.conditioning_window_steps == 3
    assert observed_bundle.selected_prefix_steps == 3
    assert observed_bundle.generation_prefix_crop_steps == 3
    assert observed_bundle.executable_candidate_horizon_steps == 3
    before_response_hash = observed_bundle.response.semantic_content_hash
    before_result_hash = (
        observed_bundle.policy_chunk_result.semantic_content_hash
    )
    debug_payload = rtc_conditioned_client_inference_bundle_debug_payload(
        observed_bundle
    )
    assert (
        rtc_client_inference_bundle_debug_payload(observed_bundle)
        == debug_payload
    )
    assert debug_payload["debug_only"] is True
    assert debug_payload["non_authoritative"] is True
    assert debug_payload["action_send_authority"] is False
    assert debug_payload["conditioning_mode"] == (
        RTC_CONDITIONING_MODE_PREFIX_V1
    )
    assert debug_payload["selected_prefix_steps"] == 3
    assert debug_payload["encoded_prefix_steps"] == 3
    assert debug_payload["consumed_prefix_steps"] == 3
    assert debug_payload["prefix_step_count_parity"] is True
    assert debug_payload["masked_output_max_abs_error"] == 0.0
    assert (
        observed_bundle.response.semantic_content_hash
        == before_response_hash
    )
    assert (
        observed_bundle.policy_chunk_result.semantic_content_hash
        == before_result_hash
    )
    non_debug_executor = RtcConditionedModelClientInferenceExecutorV3(
        client=RtcConditionedModelClientV3(
            backend=_Backend(),
            codec=codec,
            decoder=_decoder,
        ),
        prepare_input=lambda request, observation: RtcPreparedModelInput(
            model_input={"observation": "fake"},
            model_input_hash="model-input:client",
        ),
        observe_bundle=None,
    )
    non_debug_result = non_debug_executor.execute_work_item(item)
    assert non_debug_executor.observe_bundle is None
    assert (
        non_debug_result.semantic_content_hash
        == result.semantic_content_hash
    )
    assert (
        observed_bundle.capability
        is observed_bundle.envelope.capability
    )
    assert (
        observed_bundle.encoded_conditioning
        is observed_bundle.envelope.encoded_conditioning
    )
    assert "capability" not in observed_bundle.__dataclass_fields__
    assert "encoded_conditioning" not in observed_bundle.__dataclass_fields__
    with pytest.raises(ValueError, match="validation evidence mismatch"):
        replace(
            observed_bundle,
            response_validation=replace(
                observed_bundle.response_validation,
                envelope_hash="sha256:other",
            ),
        )
    with pytest.raises(ValueError, match="result/request mismatch"):
        replace(
            observed_bundle,
            policy_chunk_result=replace(
                observed_bundle.policy_chunk_result,
                request_hash="sha256:other",
            ),
        )
    different_canonical = _bundle(indices=(1, 2))
    with pytest.raises(ValueError, match="canonical/envelope mismatch"):
        replace(
            observed_bundle,
            canonical_conditioning=different_canonical,
            policy_chunk_result=replace(
                observed_bundle.policy_chunk_result,
                conditioning_binding_hash=(
                    different_canonical.conditioning_binding_hash
                ),
            ),
        )


@pytest.mark.parametrize(
    ("forecast_delay_steps", "prefix_steps"),
    (
        (1, 0),
        (1, 1),
        (2, 1),
        (2, 2),
        (3, 1),
        (3, 2),
        (3, 3),
        (4, 0),
        (4, 1),
        (4, 3),
        (4, 4),
    ),
)
def test_v3_client_k_d_h_matrix_crops_generation_prefix_exactly_once(
    forecast_delay_steps: int,
    prefix_steps: int,
) -> None:
    bundle = _bundle_for_delay_and_prefix(
        forecast_delay_steps=forecast_delay_steps,
        prefix_steps=prefix_steps,
    )
    request = _request(bundle)
    codec = _FakeConditioningCodec()
    capability = _client_transform_capability(bundle, codec)
    generated = np.zeros((6, 7), dtype=np.float32)
    for index in range(6):
        generated[index] = np.asarray(
            [
                0.5 + index * 0.01,
                -0.1,
                0.3,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            dtype=np.float32,
        )

    class _Backend:
        rtc_continuation_capability = capability

        def infer_conditioned_rtc_v3(self, envelope, model_input):
            assert model_input == {"observation": "matrix"}
            actions = _clamp_selected_output(envelope, generated)
            return RtcConditionedModelResponseV3(
                envelope_hash=envelope.semantic_content_hash,
                request_hash=envelope.request.semantic_content_hash,
                capability_hash=envelope.capability.semantic_content_hash,
                model_output_id=(
                    f"model-output:matrix:{forecast_delay_steps}:{prefix_steps}"
                ),
                model_space_actions=actions,
                inference_started_at=10.0,
                inference_completed_at=10.1,
                result_received_at=10.1,
                consumption_receipt=_consumption_receipt(envelope),
            )

    decoded_rows: list[np.ndarray] = []

    def _decoder(model_actions, discrete_actions, decode_request):
        assert discrete_actions is None
        rows = np.asarray(model_actions, dtype=np.float32).copy()
        decoded_rows.append(rows)
        return tuple(
            build_canonical_policy_action(
                RobotAction(
                    action_space="single_arm_ee_pose_v1",
                    timestamp=(
                        decode_request.first_target_effective_at
                        + index * decode_request.action_period_sec
                    ),
                    arm_pose_by_side={
                        "right": np.asarray(row, dtype=np.float32)
                    },
                    source="rtc-conditioned-matrix-test",
                ),
                source_pose_mode="absolute",
                observation_cut_id=decode_request.observation_cut_id,
                source_action_hash=(
                    f"matrix-row:{forecast_delay_steps}:{prefix_steps}:{index}"
                ),
                action_codec_hash=decode_request.action_codec_hash,
                frame_contract_hash=decode_request.frame_contract_hash,
                layout_hash=decode_request.layout_hash,
                normalization_hash=decode_request.normalization_hash,
                conversion_contract_hash="conversion:absolute:test",
            )
            for index, row in enumerate(rows)
        )

    result_bundle = RtcConditionedModelClientV3(
        backend=_Backend(),
        codec=codec,
        decoder=_decoder,
    ).infer(
        request,
        canonical_conditioning=bundle,
        model_input={"observation": "matrix"},
        model_input_hash="model-input:matrix",
    )

    assert bundle.selected_generation_indices == tuple(range(prefix_steps))
    assert len(decoded_rows) == 1
    assert decoded_rows[0].shape == (
        6 - forecast_delay_steps,
        7,
    )
    np.testing.assert_allclose(
        decoded_rows[0],
        generated[forecast_delay_steps:],
    )
    assert result_bundle.generation_horizon_steps == 6
    assert (
        result_bundle.conditioning_window_steps
        == forecast_delay_steps
    )
    assert result_bundle.selected_prefix_steps == prefix_steps
    assert (
        result_bundle.generation_prefix_crop_steps
        == forecast_delay_steps
    )
    assert (
        result_bundle.executable_candidate_horizon_steps
        == 6 - forecast_delay_steps
    )
    assert (
        result_bundle.policy_chunk_result.execution_horizon_steps
        == 6 - forecast_delay_steps
    )


def test_v3_client_rejects_loaded_capability_drift_before_provider_call() -> None:
    bundle = _bundle(indices=(0, 1, 2))
    request = _request(bundle)
    codec = _FakeConditioningCodec()
    capability = _client_transform_capability(bundle, codec)

    class _Backend:
        rtc_continuation_capability = capability
        call_count = 0

        def infer_conditioned_rtc_v3(self, envelope, model_input):
            self.call_count += 1
            raise AssertionError("capability drift must fail before provider call")

    backend = _Backend()
    client = RtcConditionedModelClientV3(
        backend=backend,
        codec=codec,
        decoder=lambda *_args: (),
    )
    backend.rtc_continuation_capability = replace(
        capability,
        artifact_hash="artifact:drifted",
    )

    with pytest.raises(RtcClientProtocolError) as exc_info:
        client.infer(
            request,
            canonical_conditioning=bundle,
            model_input={"observation": "drift"},
            model_input_hash="model-input:drift",
        )

    assert exc_info.value.reasons == ("capability_hash_changed",)
    assert backend.call_count == 0


@pytest.mark.parametrize(
    ("provider_mode", "expected_reasons"),
    (
        ("wrong_type", ("response_type_mismatch",)),
        ("wrong_identity", ("envelope_hash_mismatch",)),
    ),
)
def test_v3_client_rejects_provider_faults_before_decoder(
    provider_mode: str,
    expected_reasons: tuple[str, ...],
) -> None:
    bundle = _bundle(indices=(0, 1, 2))
    request = _request(bundle)
    codec = _FakeConditioningCodec()
    capability = _client_transform_capability(bundle, codec)

    class _Backend:
        rtc_continuation_capability = capability

        def infer_conditioned_rtc_v3(self, envelope, model_input):
            if provider_mode == "wrong_type":
                return {"actions": np.zeros((6, 7), dtype=np.float32)}
            return RtcConditionedModelResponseV3(
                envelope_hash="sha256:wrong-envelope",
                request_hash=envelope.request.semantic_content_hash,
                capability_hash=envelope.capability.semantic_content_hash,
                model_output_id="model-output:provider-fault",
                model_space_actions=_clamp_selected_output(
                    envelope,
                    np.zeros((6, 7), dtype=np.float32),
                ),
                inference_started_at=10.0,
                inference_completed_at=10.1,
                result_received_at=10.1,
                consumption_receipt=_consumption_receipt(envelope),
            )

    decoder_calls = 0

    def _decoder(*_args):
        nonlocal decoder_calls
        decoder_calls += 1
        return ()

    client = RtcConditionedModelClientV3(
        backend=_Backend(),
        codec=codec,
        decoder=_decoder,
    )
    with pytest.raises(RtcClientProtocolError) as exc_info:
        client.infer(
            request,
            canonical_conditioning=bundle,
            model_input={"observation": "provider-fault"},
            model_input_hash="model-input:provider-fault",
        )

    assert exc_info.value.reasons == expected_reasons
    assert decoder_calls == 0


@pytest.mark.parametrize(
    ("decoder_mode", "expected_reasons"),
    (
        ("wrong_count", ("decoded_action_count_mismatch",)),
        ("wrong_type", ("typed_canonical_action_required",)),
    ),
)
def test_v3_client_decoder_faults_never_create_partial_results(
    decoder_mode: str,
    expected_reasons: tuple[str, ...],
) -> None:
    bundle = _bundle(indices=(0, 1, 2))
    request = _request(bundle)
    codec = _FakeConditioningCodec()
    capability = _client_transform_capability(bundle, codec)

    class _Backend:
        rtc_continuation_capability = capability

        def infer_conditioned_rtc_v3(self, envelope, model_input):
            return RtcConditionedModelResponseV3(
                envelope_hash=envelope.semantic_content_hash,
                request_hash=envelope.request.semantic_content_hash,
                capability_hash=envelope.capability.semantic_content_hash,
                model_output_id="model-output:decoder-fault",
                model_space_actions=_clamp_selected_output(
                    envelope,
                    np.zeros((6, 7), dtype=np.float32),
                ),
                inference_started_at=10.0,
                inference_completed_at=10.1,
                result_received_at=10.1,
                consumption_receipt=_consumption_receipt(envelope),
            )

    def _decoder(model_actions, discrete_actions, decode_request):
        del discrete_actions, decode_request
        if decoder_mode == "wrong_count":
            return ()
        return tuple(object() for _row in model_actions)

    client = RtcConditionedModelClientV3(
        backend=_Backend(),
        codec=codec,
        decoder=_decoder,
    )
    with pytest.raises(RtcClientProtocolError) as exc_info:
        client.infer(
            request,
            canonical_conditioning=bundle,
            model_input={"observation": "decoder-fault"},
            model_input_hash="model-input:decoder-fault",
        )

    assert exc_info.value.reasons == expected_reasons


@pytest.mark.parametrize(
    ("cancel_result", "expected"),
    (
        (True, True),
        (False, False),
    ),
)
def test_v3_client_cancel_preserves_provider_boolean_contract(
    cancel_result: bool,
    expected: bool,
) -> None:
    bundle = _bundle()
    codec = _FakeConditioningCodec()
    capability = _client_transform_capability(bundle, codec)

    class _Backend:
        rtc_continuation_capability = capability

        def infer_conditioned_rtc_v3(self, envelope, model_input):
            raise AssertionError("cancel test must not call inference")

        def request_cancel(self):
            return cancel_result

    client = RtcConditionedModelClientV3(
        backend=_Backend(),
        codec=codec,
        decoder=lambda *_args: (),
    )
    assert client.request_cancel() is expected


def test_v3_client_cancel_rejects_nonboolean_provider_result() -> None:
    bundle = _bundle()
    codec = _FakeConditioningCodec()
    capability = _client_transform_capability(bundle, codec)

    class _Backend:
        rtc_continuation_capability = capability

        def infer_conditioned_rtc_v3(self, envelope, model_input):
            raise AssertionError("cancel test must not call inference")

        @staticmethod
        def request_cancel():
            return 1

    client = RtcConditionedModelClientV3(
        backend=_Backend(),
        codec=codec,
        decoder=lambda *_args: (),
    )
    with pytest.raises(TypeError, match="must return bool"):
        client.request_cancel()


def test_v3_adapter_runs_bootstrap_then_prefix_on_shared_timed_plan() -> None:
    contract = ModelContinuationDataContractV2(
        conditioning_origin=MODEL_CONTINUATION_ORIGIN_INFERENCE_ONLY_V1,
        runtime=ModelContinuationRuntimeContractV2.from_v1(
            _continuation_contract(bootstrap=True)
        ),
    )
    runtime_contract = normalize_model_continuation_contract(contract).runtime
    codec = _FakeConditioningCodec()
    capability = _client_transform_capability(
        _bundle(bootstrap=True, continuation_contract=contract),
        codec,
    )
    observed_bundles = []
    provider_thread_ids = []

    class _Backend:
        rtc_continuation_capability = capability

        def infer_conditioned_rtc_v3(self, envelope, model_input):
            del model_input
            provider_thread_ids.append(threading.get_ident())
            now = time.monotonic()
            actions = np.tile(
                np.asarray(
                    [0.5, -0.1, 0.3, 0.0, 0.0, 0.0, 1.0],
                    dtype=np.float32,
                ),
                (runtime_contract.prediction_horizon_steps, 1),
            )
            actions = _clamp_selected_output(envelope, actions)
            return RtcConditionedModelResponseV3(
                envelope_hash=envelope.semantic_content_hash,
                request_hash=envelope.request.semantic_content_hash,
                capability_hash=envelope.capability.semantic_content_hash,
                model_output_id=(
                    f"model-output:frontend:{len(provider_thread_ids)}"
                ),
                model_space_actions=actions,
                inference_started_at=now,
                inference_completed_at=now,
                result_received_at=now,
                consumption_receipt=_consumption_receipt(envelope),
            )

    def _decoder(model_actions, discrete_actions, request):
        assert discrete_actions is None
        return tuple(
            build_canonical_policy_action(
                RobotAction(
                    action_space="single_arm_ee_pose_v1",
                    timestamp=(
                        request.first_target_effective_at
                        + index * request.action_period_sec
                    ),
                    arm_pose_by_side={
                        "right": np.asarray(row, dtype=np.float32)
                    },
                    source="rtc-conditioned-frontend-test",
                ),
                source_pose_mode="absolute",
                observation_cut_id=request.observation_cut_id,
                source_action_hash=f"frontend-row:{index}",
                action_codec_hash=request.action_codec_hash,
                frame_contract_hash=request.frame_contract_hash,
                layout_hash=request.layout_hash,
                normalization_hash=request.normalization_hash,
                conversion_contract_hash="conversion:absolute:test",
            )
            for index, row in enumerate(model_actions)
        )

    class _Adapter:
        rtc_continuation_capability = capability
        continuation_contract = contract

        @staticmethod
        def representation_context_builder(observation, cut_id):
            return replace(
                _representation_context(),
                observation_cut_id=cut_id,
                observation_contract_hash=observation.contract_hash,
                observation_timestamp=float(
                    observation.payload["timestamp"]
                ),
                observation_step_index=int(
                    observation.payload["step_index"]
                ),
            )

        @staticmethod
        def create_worker_executor(*, observe_bundle=None):
            return RtcConditionedModelClientInferenceExecutorV3(
                client=RtcConditionedModelClientV3(
                    backend=_Backend(),
                    codec=codec,
                    decoder=_decoder,
                ),
                prepare_input=lambda request, observation: (
                    RtcPreparedModelInput(
                        model_input={"step": observation.payload["step_index"]},
                        model_input_hash=(
                            f"model-input:{request.observation_cut_id}"
                        ),
                    )
                ),
                observe_bundle=observe_bundle,
            )

    adapter = _Adapter()
    config = rtc_conditioned_config_from_model_adapter_v3(
        adapter,
        runtime_session_id="session:frontend-v3",
        control_domain_id="domain:frontend-v3",
        producer_id="producer:frontend-v3",
        policy_node_id="single_policy",
        request_trigger_coverage_steps=2,
    )
    assert config.model_continuation is not None
    assert not config.profile.send_actions
    assert not config.profile.position_lookahead_enabled

    runtime = PolicyDeployRuntime(
        PolicyDeployRuntimeConfig(
            active_sides=("right",),
            arms_enabled=True,
            hands_enabled=False,
            action_keyframe_period_sec=1.0 / 30.0,
            action_maximum_hold_sec=1.0 / 30.0,
        )
    )
    base = time.monotonic()
    frontend = build_rtc_conditioned_model_shadow_frontend_v3(
        adapter,
        runtime_session_id="session:frontend-v3",
        control_domain_id="domain:frontend-v3",
        producer_id="producer:frontend-v3",
        policy_node_id="single_policy",
        deploy_runtime=runtime,
        preparer=object(),
        created_at=base,
        request_trigger_coverage_steps=2,
        observe_bundle=observed_bundles.append,
    )

    frontend.start()
    emissions = 0
    try:
        for index in range(240):
            now = (
                base
                + index / 30.0
                + (0.001 if index % 3 == 1 else 0.0)
            )
            step = frontend.step(
                build_deployment_observation(
                    active_sides=("right",),
                    arms_enabled=True,
                    hands_enabled=False,
                    step_index=index,
                    arm_states={
                        "right": {
                            "current_qpos": np.zeros(
                                6,
                                dtype=np.float32,
                            ),
                            "current_tcp_pose": np.eye(4, dtype=float),
                            "actual_tcp_pose": np.eye(4, dtype=float),
                        }
                    },
                    hand_states={},
                    timestamp=now,
                ),
                now=now,
            )
            if step.action_preparation.emission is not None:
                emissions += 1
            if (
                emissions
                and any(
                    bundle.canonical_conditioning.selected_generation_indices
                    for bundle in observed_bundles
                )
            ):
                break
            time.sleep(0.001)
    finally:
        frontend.close(now=base + 10.0)

    assert emissions > 0
    assert observed_bundles
    assert (
        observed_bundles[0].canonical_conditioning.conditioning_mode
        == RTC_CONDITIONING_MODE_UNCONDITIONED_BOOTSTRAP_V1
    )
    assert any(
        bundle.canonical_conditioning.selected_generation_indices
        for bundle in observed_bundles[1:]
    )
    bootstrap_debug = (
        rtc_conditioned_client_inference_bundle_debug_payload(
            observed_bundles[0]
        )
    )
    conditioned_debug = next(
        rtc_conditioned_client_inference_bundle_debug_payload(bundle)
        for bundle in observed_bundles[1:]
        if bundle.canonical_conditioning.selected_generation_indices
    )
    assert bootstrap_debug["conditioning_mode"] == (
        RTC_CONDITIONING_MODE_UNCONDITIONED_BOOTSTRAP_V1
    )
    assert bootstrap_debug["selected_prefix_steps"] == 0
    assert bootstrap_debug["encoded_prefix_steps"] == 0
    assert bootstrap_debug["consumed_prefix_steps"] == 0
    assert bootstrap_debug["prefix_step_count_parity"] is True
    assert conditioned_debug["conditioning_mode"] == (
        RTC_CONDITIONING_MODE_PREFIX_V1
    )
    assert int(conditioned_debug["selected_prefix_steps"]) > 0
    assert conditioned_debug["prefix_step_count_parity"] is True
    assert provider_thread_ids
    assert provider_thread_ids[0] != threading.get_ident()
    assert runtime.action_queue_pop_count() == 0


def test_v3_frontend_rejects_missing_work_item_capability_at_build() -> None:
    contract = ModelContinuationDataContractV2(
        conditioning_origin=MODEL_CONTINUATION_ORIGIN_INFERENCE_ONLY_V1,
        runtime=ModelContinuationRuntimeContractV2.from_v1(
            _continuation_contract(bootstrap=True)
        ),
    )
    codec = _FakeConditioningCodec()
    capability = _client_transform_capability(
        _bundle(bootstrap=True, continuation_contract=contract),
        codec,
    )

    class _BadAdapter:
        rtc_continuation_capability = capability
        continuation_contract = contract

        @staticmethod
        def representation_context_builder(observation, cut_id):
            raise AssertionError("construction must not consume observations")

        @staticmethod
        def create_worker_executor(*, observe_bundle=None):
            del observe_bundle
            return lambda request, observation: None

    runtime = PolicyDeployRuntime(
        PolicyDeployRuntimeConfig(
            active_sides=("right",),
            arms_enabled=True,
            hands_enabled=False,
            action_keyframe_period_sec=1.0 / 30.0,
            action_maximum_hold_sec=1.0 / 30.0,
        )
    )
    with pytest.raises(TypeError, match="execute_work_item"):
        build_rtc_conditioned_model_shadow_frontend_v3(
            _BadAdapter(),
            runtime_session_id="session:missing-work-item",
            control_domain_id="domain:missing-work-item",
            producer_id="producer:missing-work-item",
            policy_node_id="single_policy",
            deploy_runtime=runtime,
            preparer=object(),
            created_at=10.0,
        )


def test_plan_merge_rejects_conditioned_result_binding_tampering() -> None:
    bundle = _bundle(indices=(0, 1, 2))
    codec = _FakeConditioningCodec()
    capability = _client_transform_capability(bundle, codec)

    class _Backend:
        rtc_continuation_capability = capability

        def infer_conditioned_rtc_v3(self, envelope, model_input):
            actions = np.tile(
                np.asarray(
                    [0.5, -0.1, 0.3, 0.0, 0.0, 0.0, 1.0],
                    dtype=np.float32,
                ),
                (6, 1),
            )
            actions = _clamp_selected_output(envelope, actions)
            return RtcConditionedModelResponseV3(
                envelope_hash=envelope.semantic_content_hash,
                request_hash=envelope.request.semantic_content_hash,
                capability_hash=(
                    envelope.capability.semantic_content_hash
                ),
                model_output_id="model-output:merge",
                model_space_actions=actions,
                inference_started_at=10.0,
                inference_completed_at=10.1,
                result_received_at=10.1,
                consumption_receipt=(
                    _consumption_receipt(envelope)
                ),
            )

    def _decoder(model_actions, discrete_actions, decode_request):
        return tuple(
            build_canonical_policy_action(
                RobotAction(
                    action_space="single_arm_ee_pose_v1",
                    timestamp=10.3 + index * 0.1,
                    arm_pose_by_side={
                        "right": np.asarray(row, dtype=np.float32)
                    },
                    source="rtc-conditioned-merge-test",
                ),
                source_pose_mode="absolute",
                observation_cut_id=decode_request.observation_cut_id,
                source_action_hash=f"merge-row:{index}",
                action_codec_hash=decode_request.action_codec_hash,
                frame_contract_hash=decode_request.frame_contract_hash,
                layout_hash=decode_request.layout_hash,
                normalization_hash=decode_request.normalization_hash,
                conversion_contract_hash="conversion:absolute:test",
            )
            for index, row in enumerate(model_actions)
        )

    client = RtcConditionedModelClientV3(
        backend=_Backend(),
        codec=codec,
        decoder=_decoder,
    )
    request = _request(bundle)
    result = client.infer(
        request,
        canonical_conditioning=bundle,
        model_input={},
        model_input_hash="model-input:merge",
    ).policy_chunk_result
    plan = mark_policy_action_plan_in_flight(
        _plan((0, 1, 2)),
        request,
    )

    accepted = merge_policy_chunk_result(
        plan,
        request,
        result,
        admitted_at=10.1,
        actual_cursor_at=10.0,
    )
    assert accepted.receipt.accepted
    assert accepted.receipt.discarded_obsolete_count == 0
    assert accepted.receipt.admitted_keyframe_count == 3
    assert accepted.plan.producer_committed_through is None
    assert all(
        keyframe.state == POLICY_PLAN_KEYFRAME_MUTABLE
        for keyframe in accepted.plan.keyframes
    )

    cursor_advanced = merge_policy_chunk_result(
        plan,
        request,
        result,
        admitted_at=10.4,
        actual_cursor_at=10.35,
    )
    assert cursor_advanced.receipt.accepted
    assert cursor_advanced.receipt.discarded_obsolete_count > 0
    assert (
        cursor_advanced.receipt.admitted_keyframe_count
        < result.source_chunk_size
    )
    assert (
        cursor_advanced.receipt.discarded_obsolete_count
        >= result.source_chunk_size
        - cursor_advanced.receipt.admitted_keyframe_count
    )

    rotated_epoch = merge_policy_chunk_result(
        replace(
            plan,
            producer_epoch=plan.producer_epoch + 1,
            keyframes=(),
            in_flight_request_id=None,
            in_flight_inference_id=None,
        ),
        request,
        result,
        admitted_at=10.1,
        actual_cursor_at=10.0,
    )
    assert not rotated_epoch.receipt.accepted
    assert "producer_epoch_mismatch" in rotated_epoch.receipt.reasons

    tampered = merge_policy_chunk_result(
        plan,
        request,
        replace(result, conditioning_binding_hash="sha256:wrong"),
        admitted_at=10.1,
        actual_cursor_at=10.0,
    )
    assert not tampered.receipt.accepted
    assert "conditioning_binding_mismatch" in tampered.receipt.reasons


def test_frontend_builds_default_off_model_rtc_bootstrap_binding() -> None:
    contract = _continuation_contract(bootstrap=True)
    profile = PolicyAsyncExecutionProfile(
        profile_id="model-rtc-c1-shadow-test",
        rtc_mode=RTC_FLOW_INPAINT_MODE_V1,
    )
    delay_spec = InferenceDelayEstimatorSpec(
        spec_id="model-rtc-delay:test",
        action_period_sec=1.0 / 30.0,
        window_size=4,
        minimum_delay_steps=3,
        maximum_delay_steps=3,
    )
    config = RtcPolicyShadowConfig(
        runtime_session_id="session:test",
        control_domain_id="domain:test",
        producer_id="producer:test",
        policy_node_id="single_policy",
        artifact_hash="artifact:test",
        action_codec_hash=contract.action_codec_hash,
        frame_contract_hash=contract.frame_contract_hash,
        layout_hash=contract.layout_hash,
        normalization_hash=contract.normalization_hash,
        profile=profile,
        prediction_horizon_steps=contract.prediction_horizon_steps,
        minimum_execution_horizon_steps=1,
        delay_estimator_spec=delay_spec,
        action_pose_mode="absolute",
        model_continuation=RtcModelContinuationRuntimeSpec(
            continuation_contract=contract,
        ),
    )
    actor = PolicyObservationView(
        contract_id="observation-view:test",
        contract_hash="observation-contract:test",
        payload={"timestamp": 10.0, "step_index": 100},
    )

    def _context_builder(observation, observation_cut_id):
        return replace(
            _representation_context(),
            observation_cut_id=observation_cut_id,
            observation_contract_hash=observation.contract_hash,
            observation_timestamp=float(observation.payload["timestamp"]),
            observation_step_index=int(observation.payload["step_index"]),
        )

    frontend = RtcPolicyShadowFrontend(
        config=config,
        deploy_runtime=PolicyDeployRuntime(
            PolicyDeployRuntimeConfig(
                active_sides=("right",),
                arms_enabled=True,
                hands_enabled=False,
                action_keyframe_period_sec=1.0 / 30.0,
                action_maximum_hold_sec=1.0 / 30.0,
            )
        ),
        policy=object(),
        created_at=10.0,
        representation_context_builder=_context_builder,
    )
    request, bundle = frontend._build_request_and_conditioning(
        actor,
        now=10.0,
    )

    assert bundle is not None
    assert bundle.selection.bootstrap_required
    assert bundle.prefix_mask == (0.0,) * 6
    assert request.rtc_mode == RTC_FLOW_INPAINT_MODE_V1
    assert request.conditioning_prefix is None
    assert (
        request.conditioning_binding_hash
        == bundle.conditioning_binding_hash
    )
