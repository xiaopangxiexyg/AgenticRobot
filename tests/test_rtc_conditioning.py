from __future__ import annotations

import numpy as np
import pytest

from deployments.client.rtc_conditioning import (
    RtcConditioningEncodingReceipt,
    RtcEncodedConditioning,
    rtc_binary_mask_hash,
    rtc_float32_array_hash,
)
from deployments.client.rtc_conditioning_protocol import (
    RTC_CONDITIONING_TRANSFORM_CLIENT_V1,
    RtcConditionedInferenceEnvelopeV3,
    RtcModelContinuationCapabilityV3,
)
from deployments.openpi.client.rtc_conditioned import (
    OPENPI_RTC_REQUEST_DEADLINE_POLICY_V1,
    _require_request_deadline,
)
from dataset_contracts.model_continuation import (
    MODEL_CONTINUATION_SAMPLER_CLAMPED_V1,
)
from robot_motion.deploy.inference import (
    RTC_FLOW_INPAINT_MODE_V1,
    PolicyInferencePrefix,
    PolicyInferenceRequest,
)


def _request(*, valid_until: float = 10.0) -> PolicyInferenceRequest:
    return PolicyInferenceRequest(
        request_id="rtc-conditioning-request",
        inference_id="rtc-conditioning-inference",
        rtc_mode=RTC_FLOW_INPAINT_MODE_V1,
        session_id="rtc-conditioning-session",
        control_domain_id="rtc-conditioning-control",
        producer_id="rtc-conditioning-producer",
        producer_epoch=0,
        policy_node_id="rtc-conditioning-policy",
        artifact_hash="artifact:test",
        observation_cut_id="observation:test",
        observation_timestamp=0.0,
        observation_step_index=0,
        triggered_at=0.0,
        admitted_at=0.0,
        first_target_effective_at=1.0,
        clock_domain_id="local_monotonic",
        action_period_sec=1.0,
        prediction_horizon_steps=1,
        minimum_execution_horizon_steps=1,
        source_plan_id="plan:test",
        source_plan_generation=0,
        source_plan_hash="plan-hash:test",
        source_plan_cursor_at=0.0,
        producer_plan_committed_through=None,
        forecast_delay_steps=1,
        delay_estimator_spec_hash="delay-spec:test",
        action_codec_hash="codec:test",
        frame_contract_hash="frame:test",
        layout_hash="layout:test",
        normalization_hash="normalization:test",
        valid_until=valid_until,
        conditioning_prefix=PolicyInferencePrefix(
            model_space_actions=np.zeros((1, 1), dtype=np.float32),
            valid_steps=0,
            source_plan_id="plan:test",
            source_plan_generation=0,
            source_plan_hash="plan-hash:test",
        ),
    )


def test_provider_deadline_policy_rejects_before_and_after_the_boundary() -> None:
    request = _request()

    assert OPENPI_RTC_REQUEST_DEADLINE_POLICY_V1.startswith("reject_")
    assert _require_request_deadline(request, now=9.999, boundary="before") == 9.999
    with pytest.raises(ValueError, match="before"):
        _require_request_deadline(request, now=10.0, boundary="before")
    with pytest.raises(ValueError, match="after"):
        _require_request_deadline(request, now=10.0, boundary="after")
    with pytest.raises(ValueError, match="finite"):
        _require_request_deadline(request, now=float("nan"), boundary="before")


def test_rtc_v3_capability_identity_is_independent_of_provenance() -> None:
    """Compatible runtime fields, not artifact/contract labels, define identity."""

    capability = RtcModelContinuationCapabilityV3(
        artifact_hash="artifact:server-step-1000",
        continuation_contract_hash="contract:test",
        codec_identity_hash="codec:test",
        continuation_implementation_id="impl:test",
        continuation_implementation_parameters_hash="params:test",
        sampler_constraint=MODEL_CONTINUATION_SAMPLER_CLAMPED_V1,
        selected_prefix_schedule="schedule:test",
        conditioning_transform_owner=RTC_CONDITIONING_TRANSFORM_CLIENT_V1,
        external_action_dim=1,
        internal_sampler_action_dim=1,
        supports_unconditioned_bootstrap=False,
    )
    replacement = RtcModelContinuationCapabilityV3(
        artifact_hash="artifact:server-step-2000",
        continuation_contract_hash="contract:server-v2",
        codec_identity_hash=capability.codec_identity_hash,
        continuation_implementation_id=capability.continuation_implementation_id,
        continuation_implementation_parameters_hash=(
            capability.continuation_implementation_parameters_hash
        ),
        sampler_constraint=capability.sampler_constraint,
        selected_prefix_schedule=capability.selected_prefix_schedule,
        conditioning_transform_owner=capability.conditioning_transform_owner,
        external_action_dim=capability.external_action_dim,
        internal_sampler_action_dim=capability.internal_sampler_action_dim,
        supports_unconditioned_bootstrap=(
            capability.supports_unconditioned_bootstrap
        ),
    )

    assert capability.semantic_content_hash == replacement.semantic_content_hash

    prefix = np.asarray([[0.5], [0.0]], dtype=np.float32)
    mask = np.asarray([1.0, 0.0], dtype=np.float32)
    receipt = RtcConditioningEncodingReceipt(
        conditioning_binding_hash="binding:test",
        codec_identity_hash=capability.codec_identity_hash,
        selected_generation_indices=(0,),
        canonical_action_content_hash="canonical:test",
        artifact_action_rows_hash="artifact-rows:test",
        external_prefix_hash=rtc_float32_array_hash(prefix),
        prefix_mask_hash=rtc_binary_mask_hash(mask),
        decoded_canonical_action_content_hash="canonical:test",
        round_trip_max_abs_error=0.0,
    )
    encoded = RtcEncodedConditioning(
        external_prefix=prefix,
        prefix_mask=mask,
        receipt=receipt,
    )
    request = PolicyInferenceRequest(
        request_id="request:test",
        inference_id="inference:test",
        rtc_mode=RTC_FLOW_INPAINT_MODE_V1,
        session_id="session:test",
        control_domain_id="domain:test",
        producer_id="producer:test",
        producer_epoch=0,
        policy_node_id="node:test",
        artifact_hash="artifact:client-policy",
        observation_cut_id="observation:test",
        observation_timestamp=0.0,
        observation_step_index=0,
        triggered_at=0.0,
        admitted_at=0.0,
        first_target_effective_at=1.0,
        clock_domain_id="local_monotonic",
        action_period_sec=1.0,
        prediction_horizon_steps=2,
        minimum_execution_horizon_steps=1,
        source_plan_id="plan:test",
        source_plan_generation=0,
        source_plan_hash="plan-hash:test",
        source_plan_cursor_at=0.0,
        producer_plan_committed_through=None,
        forecast_delay_steps=1,
        delay_estimator_spec_hash="delay-spec:test",
        action_codec_hash="action-codec:test",
        frame_contract_hash="frame:test",
        layout_hash="layout:test",
        normalization_hash="normalization:test",
        valid_until=3.0,
        conditioning_binding_hash=receipt.conditioning_binding_hash,
    )
    envelope = RtcConditionedInferenceEnvelopeV3(
        request=request,
        capability=capability,
        encoded_conditioning=encoded,
        model_input_hash="model-input:test",
    )

    assert envelope.capability.artifact_hash != envelope.request.artifact_hash
