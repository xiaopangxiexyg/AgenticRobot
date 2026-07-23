from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from deployments.client.rtc import ModelAgnosticRtcClient, RtcClientProtocolError
from robot_motion.deploy.actions import NormalizedAction
from robot_motion.deploy.inference import (
    ASYNC_ALIGNED_CHUNK_MODE_V1,
    RTC_FLOW_INPAINT_MODE_V1,
    SYNC_QUEUE_MODE_V1,
    PolicyInferencePrefix,
    PolicyInferenceRequest,
)
from robot_motion.deploy.rtc_protocol import (
    RTC_CONTINUATION_FLOW_INPAINT,
    RTC_CONTINUATION_NATIVE,
    RTC_CONTINUATION_NONE,
    RTC_PREFIX_FORBIDDEN,
    RTC_PREFIX_REQUIRED,
    RTC_PROTOCOL_V2,
    RTC_SCHEDULER_ASYNC_TIMED_PLAN,
    RTC_SCHEDULER_SYNC_QUEUE,
    RtcConditioningReceiptV2,
    RtcContinuationCapability,
    RtcInferenceEnvelopeV2,
    RtcModelCapability,
    RtcModelResponseEnvelopeV2,
    RtcProtocolMode,
    RtcProtocolRequestSpec,
    build_rtc_inference_envelope,
    negotiate_rtc_capability,
    rtc_v1_mode_from_v2,
    rtc_v2_mode_from_v1,
    validate_rtc_model_response,
)
from robot_motion.deploy.schema import ACTION_DIM


MODEL_DIM = 4
HORIZON = 4


def _continuation(
    mode: str,
    *,
    implementation_id: str,
) -> RtcContinuationCapability:
    uses_prefix = mode != RTC_CONTINUATION_NONE
    return RtcContinuationCapability(
        continuation_mode=mode,
        implementation_id=implementation_id,
        implementation_parameters_hash="parameters:deterministic-v1",
        model_action_space="unit-model-space-v1",
        model_action_dim=MODEL_DIM,
        model_action_discrete_dim=0,
        supported_action_pose_modes=("joint_position",),
        component_scope=("right_arm",),
        action_codec_hash="codec:unit-v1",
        frame_contract_hash="frame:unit-v1",
        layout_hash="layout:unit-v1",
        normalization_hash="normalization:unit-v1",
        minimum_prediction_horizon_steps=2,
        maximum_prediction_horizon_steps=8,
        minimum_execution_horizon_steps=1,
        maximum_execution_horizon_steps=8,
        prefix_requirement=(RTC_PREFIX_REQUIRED if uses_prefix else RTC_PREFIX_FORBIDDEN),
        supports_prefix_mask=uses_prefix,
        supports_conditioning_receipt=uses_prefix,
    )


def _capability(provider: str = "local-a") -> RtcModelCapability:
    return RtcModelCapability(
        capability_id=f"capability:{provider}",
        provider_id=provider,
        artifact_hash="artifact:unit-v1",
        scheduler_modes=(RTC_SCHEDULER_SYNC_QUEUE, RTC_SCHEDULER_ASYNC_TIMED_PLAN),
        continuations=(
            _continuation(RTC_CONTINUATION_NONE, implementation_id="none-v1"),
            _continuation(
                RTC_CONTINUATION_FLOW_INPAINT,
                implementation_id=f"flow:{provider}:v1",
            ),
        ),
    )


def _spec(*, flow: bool = True) -> RtcProtocolRequestSpec:
    return RtcProtocolRequestSpec(
        mode=RtcProtocolMode(
            RTC_SCHEDULER_ASYNC_TIMED_PLAN,
            RTC_CONTINUATION_FLOW_INPAINT if flow else RTC_CONTINUATION_NONE,
        ),
        artifact_hash="artifact:unit-v1",
        model_action_space="unit-model-space-v1",
        model_action_dim=MODEL_DIM,
        model_action_discrete_dim=0,
        action_pose_mode="joint_position",
        component_scope=("right_arm",),
        action_codec_hash="codec:unit-v1",
        frame_contract_hash="frame:unit-v1",
        layout_hash="layout:unit-v1",
        normalization_hash="normalization:unit-v1",
        prediction_horizon_steps=HORIZON,
        minimum_execution_horizon_steps=2,
        require_prefix_mask=flow,
        require_conditioning_receipt=flow,
    )


def _request(*, flow: bool = True) -> PolicyInferenceRequest:
    prefix = None
    if flow:
        prefix = PolicyInferencePrefix(
            model_space_actions=np.arange(HORIZON * MODEL_DIM, dtype=np.float32).reshape(
                HORIZON,
                MODEL_DIM,
            ),
            valid_steps=2,
            source_plan_id="plan:unit",
            source_plan_generation=3,
            source_plan_hash="plan-hash:unit",
        )
    return PolicyInferenceRequest(
        request_id="request:unit",
        inference_id="inference:unit",
        rtc_mode=(RTC_FLOW_INPAINT_MODE_V1 if flow else ASYNC_ALIGNED_CHUNK_MODE_V1),
        session_id="session:unit",
        control_domain_id="domain:unit",
        producer_id="producer:unit",
        producer_epoch=2,
        policy_node_id="node:unit",
        artifact_hash="artifact:unit-v1",
        observation_cut_id="observation-cut:unit",
        observation_timestamp=9.9,
        observation_step_index=99,
        triggered_at=10.0,
        admitted_at=10.0,
        first_target_effective_at=10.1,
        clock_domain_id="monotonic:unit",
        action_period_sec=0.1,
        prediction_horizon_steps=HORIZON,
        minimum_execution_horizon_steps=2,
        source_plan_id="plan:unit",
        source_plan_generation=3,
        source_plan_hash="plan-hash:unit",
        source_plan_cursor_at=10.0,
        producer_plan_committed_through=9.9,
        forecast_delay_steps=1,
        delay_estimator_spec_hash="delay-estimator:unit",
        action_codec_hash="codec:unit-v1",
        frame_contract_hash="frame:unit-v1",
        layout_hash="layout:unit-v1",
        normalization_hash="normalization:unit-v1",
        valid_until=11.0,
        conditioning_prefix=prefix,
    )


def _mask() -> np.ndarray:
    return np.asarray([1.0, 0.5, 0.0, 0.0], dtype=np.float32)


def _envelope(
    *,
    provider: str = "local-a",
    flow: bool = True,
) -> tuple[RtcInferenceEnvelopeV2, RtcModelCapability]:
    capability = _capability(provider)
    spec = _spec(flow=flow)
    negotiation = negotiate_rtc_capability(spec, capability)
    assert negotiation.accepted
    envelope = build_rtc_inference_envelope(
        request=_request(flow=flow),
        request_spec=spec,
        capability=capability,
        negotiation=negotiation,
        model_input_hash="model-input-hash:unit",
        prefix_mask=_mask() if flow else None,
    )
    return envelope, capability


def _discrete_none_binding() -> tuple[RtcInferenceEnvelopeV2, RtcModelCapability]:
    capability = _capability()
    none_capability = capability.continuation(RTC_CONTINUATION_NONE)
    assert none_capability is not None
    capability = replace(
        capability,
        continuations=(
            replace(none_capability, model_action_discrete_dim=1),
            *tuple(
                value
                for value in capability.continuations
                if value.continuation_mode != RTC_CONTINUATION_NONE
            ),
        ),
    )
    spec = replace(_spec(flow=False), model_action_discrete_dim=1)
    negotiation = negotiate_rtc_capability(spec, capability)
    assert negotiation.accepted
    return (
        build_rtc_inference_envelope(
            request=_request(flow=False),
            request_spec=spec,
            capability=capability,
            negotiation=negotiation,
            model_input_hash="model-input-hash:unit",
        ),
        capability,
    )


def _response(
    envelope: RtcInferenceEnvelopeV2,
    capability: RtcModelCapability,
    *,
    values: np.ndarray | None = None,
) -> RtcModelResponseEnvelopeV2:
    selected = capability.continuation(envelope.request_spec.mode.continuation_mode)
    assert selected is not None
    actions = (
        np.arange(HORIZON * MODEL_DIM, dtype=np.float32).reshape(HORIZON, MODEL_DIM)
        if values is None
        else values
    )
    receipt = None
    if envelope.request_spec.mode.continuation_mode != RTC_CONTINUATION_NONE:
        prefix = envelope.request.conditioning_prefix
        assert prefix is not None
        receipt = RtcConditioningReceiptV2(
            envelope_hash=envelope.semantic_content_hash,
            request_hash=envelope.request.semantic_content_hash,
            capability_hash=capability.semantic_content_hash,
            selected_continuation_hash=envelope.selected_continuation_hash,
            implementation_id=selected.implementation_id,
            artifact_hash=capability.artifact_hash,
            conditioning_prefix_hash=prefix.semantic_content_hash,
            prefix_mask_hash=envelope.prefix_mask_hash,
            effective_prefix_steps=prefix.valid_steps,
            effective_delay_steps=1,
            execution_horizon_steps=3,
            implementation_parameters_hash="parameters:deterministic-v1",
        )
    return RtcModelResponseEnvelopeV2(
        envelope_hash=envelope.semantic_content_hash,
        request_id=envelope.request.request_id,
        request_hash=envelope.request.semantic_content_hash,
        model_input_hash=envelope.model_input_hash,
        capability_hash=capability.semantic_content_hash,
        selected_continuation_hash=envelope.selected_continuation_hash,
        protocol_version=RTC_PROTOCOL_V2,
        scheduler_mode=envelope.request_spec.mode.scheduler_mode,
        continuation_mode=envelope.request_spec.mode.continuation_mode,
        artifact_hash=capability.artifact_hash,
        action_codec_hash=envelope.request.action_codec_hash,
        frame_contract_hash=envelope.request.frame_contract_hash,
        layout_hash=envelope.request.layout_hash,
        normalization_hash=envelope.request.normalization_hash,
        model_output_id=f"output:{capability.provider_id}",
        model_space_actions=actions,
        effective_delay_steps=1,
        execution_horizon_steps=3,
        inference_started_at=10.01,
        inference_completed_at=10.02,
        result_received_at=10.03,
        conditioning_receipt=receipt,
    )


class _LocalBackendBase:
    def __init__(self, provider: str) -> None:
        self._capability = _capability(provider)

    @property
    def rtc_capability(self) -> RtcModelCapability:
        return self._capability

    def infer_rtc(
        self,
        envelope: RtcInferenceEnvelopeV2,
        model_input: Any,
    ) -> RtcModelResponseEnvelopeV2:
        base = float(model_input["base"])
        values = np.arange(HORIZON * MODEL_DIM, dtype=np.float32).reshape(
            HORIZON,
            MODEL_DIM,
        )
        return _response(envelope, self._capability, values=values + base)


class _LocalArrayBackend(_LocalBackendBase):
    pass


class _LocalCallableBackend(_LocalBackendBase):
    pass


def _decoder(
    model_space_actions: np.ndarray,
    model_space_discrete_actions: np.ndarray | None,
    request: PolicyInferenceRequest,
) -> tuple[NormalizedAction, ...]:
    del request
    assert model_space_discrete_actions is None
    decoded = []
    for row in model_space_actions:
        vector = np.zeros((ACTION_DIM,), dtype=np.float32)
        vector[:MODEL_DIM] = row
        decoded.append(NormalizedAction(vector=vector, metadata={"decoder": "unit-v1"}))
    return tuple(decoded)


@pytest.mark.parametrize(
    ("v1_mode", "scheduler", "continuation"),
    (
        (SYNC_QUEUE_MODE_V1, RTC_SCHEDULER_SYNC_QUEUE, RTC_CONTINUATION_NONE),
        (
            ASYNC_ALIGNED_CHUNK_MODE_V1,
            RTC_SCHEDULER_ASYNC_TIMED_PLAN,
            RTC_CONTINUATION_NONE,
        ),
        (
            RTC_FLOW_INPAINT_MODE_V1,
            RTC_SCHEDULER_ASYNC_TIMED_PLAN,
            RTC_CONTINUATION_FLOW_INPAINT,
        ),
    ),
)
def test_v1_mapping_is_explicit_and_lossless(
    v1_mode: str,
    scheduler: str,
    continuation: str,
) -> None:
    mode = rtc_v2_mode_from_v1(v1_mode)
    assert (mode.scheduler_mode, mode.continuation_mode) == (scheduler, continuation)
    assert rtc_v1_mode_from_v2(mode) == v1_mode


def test_v2_only_continuation_cannot_be_misrepresented_as_v1() -> None:
    mode = RtcProtocolMode(RTC_SCHEDULER_ASYNC_TIMED_PLAN, RTC_CONTINUATION_NATIVE)
    with pytest.raises(ValueError, match="no lossless"):
        rtc_v1_mode_from_v2(mode)


def test_capability_negotiation_accepts_model_neutral_binding() -> None:
    receipt = negotiate_rtc_capability(_spec(), _capability())
    assert receipt.accepted
    assert receipt.mapped_v1_mode == RTC_FLOW_INPAINT_MODE_V1
    assert receipt.selected_continuation_hash is not None


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    (
        ("artifact_hash", "wrong", "artifact_hash_mismatch"),
        ("model_action_space", "wrong", "model_action_space_mismatch"),
        ("model_action_dim", MODEL_DIM + 1, "model_action_dim_mismatch"),
        ("action_pose_mode", "wrong", "action_pose_mode_unsupported"),
        ("action_codec_hash", "wrong", "action_codec_hash_mismatch"),
        ("frame_contract_hash", "wrong", "frame_contract_hash_mismatch"),
        ("layout_hash", "wrong", "layout_hash_mismatch"),
        ("normalization_hash", "wrong", "normalization_hash_mismatch"),
        ("prediction_horizon_steps", 99, "prediction_horizon_unsupported"),
    ),
)
def test_capability_negotiation_fails_closed(
    field: str,
    value: object,
    reason: str,
) -> None:
    receipt = negotiate_rtc_capability(replace(_spec(), **{field: value}), _capability())
    assert not receipt.accepted
    assert reason in receipt.rejection_reasons


@pytest.mark.parametrize(
    "mask",
    (
        None,
        np.asarray([1.0, 0.0], dtype=np.float32),
        np.asarray([1.0, -0.1, 0.0, 0.0], dtype=np.float32),
        np.asarray([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    ),
)
def test_prefix_mask_is_required_bounded_and_causal(mask: np.ndarray | None) -> None:
    capability = _capability()
    spec = _spec()
    negotiation = negotiate_rtc_capability(spec, capability)
    with pytest.raises(ValueError):
        build_rtc_inference_envelope(
            request=_request(),
            request_spec=spec,
            capability=capability,
            negotiation=negotiation,
            model_input_hash="input:unit",
            prefix_mask=mask,
        )


def test_envelope_detaches_prefix_mask_from_mutable_caller_array() -> None:
    mask = _mask()
    capability = _capability()
    spec = _spec()
    envelope = build_rtc_inference_envelope(
        request=_request(),
        request_spec=spec,
        capability=capability,
        negotiation=negotiate_rtc_capability(spec, capability),
        model_input_hash="input:unit",
        prefix_mask=mask,
    )
    before = envelope.semantic_content_hash
    mask[:] = 0.0
    assert envelope.semantic_content_hash == before
    assert envelope.prefix_mask is not None
    assert float(envelope.prefix_mask[0]) == 1.0


def test_conditioning_prefix_model_dimension_must_match_capability() -> None:
    capability = _capability()
    spec = _spec()
    request = _request()
    assert request.conditioning_prefix is not None
    wrong_prefix = replace(
        request.conditioning_prefix,
        model_space_actions=np.zeros((HORIZON, MODEL_DIM + 1), dtype=np.float32),
    )
    request = replace(request, conditioning_prefix=wrong_prefix)
    with pytest.raises(ValueError, match="dimension mismatch"):
        build_rtc_inference_envelope(
            request=request,
            request_spec=spec,
            capability=capability,
            negotiation=negotiate_rtc_capability(spec, capability),
            model_input_hash="input:unit",
            prefix_mask=_mask(),
        )


def test_valid_response_binds_all_receipts_and_hashes() -> None:
    envelope, capability = _envelope()
    response = _response(envelope, capability)
    receipt = validate_rtc_model_response(
        envelope=envelope,
        response=response,
        capability=capability,
    )
    assert receipt.accepted
    assert response.conditioning_receipt is not None
    assert response.conditioning_receipt.prefix_mask_hash == envelope.prefix_mask_hash


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    (
        ("model_input_hash", "wrong", "model_input_hash_mismatch"),
        ("capability_hash", "wrong", "capability_hash_mismatch"),
        ("artifact_hash", "wrong", "artifact_hash_mismatch"),
        ("action_codec_hash", "wrong", "action_codec_hash_mismatch"),
        ("layout_hash", "wrong", "layout_hash_mismatch"),
        ("normalization_hash", "wrong", "normalization_hash_mismatch"),
        ("effective_delay_steps", 2, "effective_delay_mismatch"),
    ),
)
def test_response_echo_mismatch_fails_closed(
    field: str,
    value: object,
    reason: str,
) -> None:
    envelope, capability = _envelope()
    response = replace(_response(envelope, capability), **{field: value})
    receipt = validate_rtc_model_response(
        envelope=envelope,
        response=response,
        capability=capability,
    )
    assert not receipt.accepted
    assert reason in receipt.rejection_reasons


def test_missing_conditioning_receipt_fails_closed() -> None:
    envelope, capability = _envelope()
    response = replace(_response(envelope, capability), conditioning_receipt=None)
    receipt = validate_rtc_model_response(
        envelope=envelope,
        response=response,
        capability=capability,
    )
    assert not receipt.accepted
    assert receipt.rejection_reasons == ("missing_conditioning_receipt",)


def test_conditioning_implementation_parameters_mismatch_fails_closed() -> None:
    envelope, capability = _envelope()
    response = _response(envelope, capability)
    assert response.conditioning_receipt is not None
    response = replace(
        response,
        conditioning_receipt=replace(
            response.conditioning_receipt,
            implementation_parameters_hash="parameters:wrong",
        ),
    )
    receipt = validate_rtc_model_response(
        envelope=envelope,
        response=response,
        capability=capability,
    )
    assert not receipt.accepted
    assert "conditioning_implementation_parameters_hash_mismatch" in (
        receipt.rejection_reasons
    )


def test_none_continuation_neither_consumes_prefix_nor_returns_receipt() -> None:
    envelope, capability = _envelope(flow=False)
    response = _response(envelope, capability)
    receipt = validate_rtc_model_response(
        envelope=envelope,
        response=response,
        capability=capability,
    )
    assert receipt.accepted
    assert envelope.request.conditioning_prefix is None
    assert response.conditioning_receipt is None


def test_none_continuation_preserves_independent_discrete_action_tokens() -> None:
    envelope, capability = _discrete_none_binding()
    tokens = np.asarray([[0], [1], [1], [0]], dtype=np.int64)
    response = replace(
        _response(envelope, capability),
        model_space_discrete_actions=tokens,
    )
    receipt = validate_rtc_model_response(
        envelope=envelope,
        response=response,
        capability=capability,
    )
    assert receipt.accepted
    assert response.model_space_discrete_actions is not None
    np.testing.assert_array_equal(response.model_space_discrete_actions, tokens)
    assert response.semantic_content_hash != replace(
        response,
        model_space_discrete_actions=np.asarray([[1], [1], [1], [0]], dtype=np.int64),
    ).semantic_content_hash


@pytest.mark.parametrize(
    "tokens",
    (
        np.asarray([[0.5], [1.0], [1.0], [0.0]]),
        np.asarray([[True], [False], [True], [False]]),
        np.asarray([[0], [1]], dtype=np.int64),
    ),
)
def test_discrete_action_tokens_reject_fraction_bool_and_horizon_mismatch(
    tokens: np.ndarray,
) -> None:
    envelope, capability = _discrete_none_binding()
    with pytest.raises(ValueError):
        replace(
            _response(envelope, capability),
            model_space_discrete_actions=tokens,
        )


def test_expected_discrete_output_cannot_be_silently_omitted() -> None:
    envelope, capability = _discrete_none_binding()
    response = _response(envelope, capability)
    receipt = validate_rtc_model_response(
        envelope=envelope,
        response=response,
        capability=capability,
    )
    assert not receipt.accepted
    assert "model_action_discrete_dim_mismatch" in receipt.rejection_reasons


def test_discrete_continuation_is_blocked_until_typed_prefix_exists() -> None:
    capability = _capability()
    flow_capability = capability.continuation(RTC_CONTINUATION_FLOW_INPAINT)
    assert flow_capability is not None
    capability = replace(
        capability,
        continuations=tuple(
            replace(value, model_action_discrete_dim=1)
            if value.continuation_mode == RTC_CONTINUATION_FLOW_INPAINT
            else value
            for value in capability.continuations
        ),
    )
    receipt = negotiate_rtc_capability(
        replace(_spec(), model_action_discrete_dim=1),
        capability,
    )
    assert not receipt.accepted
    assert "discrete_conditioning_prefix_unsupported_v2" in receipt.rejection_reasons


def test_two_local_backend_implementations_migrate_without_request_or_action_change() -> None:
    request = _request()
    bundles = []
    for backend in (_LocalArrayBackend("local-a"), _LocalCallableBackend("local-b")):
        client = ModelAgnosticRtcClient(
            backend=backend,
            decoder=_decoder,
            request_spec=_spec(),
        )
        bundles.append(
            client.infer(
                request,
                model_input={"base": 0.25},
                model_input_hash="model-input-hash:unit",
                prefix_mask=_mask(),
            )
        )
    first, second = bundles
    assert first.envelope.request.semantic_content_hash == second.envelope.request.semantic_content_hash
    assert first.envelope.request_spec.semantic_content_hash == second.envelope.request_spec.semantic_content_hash
    assert first.envelope.capability_hash != second.envelope.capability_hash
    assert first.response_validation.accepted and second.response_validation.accepted
    assert len(first.policy_chunk_result.decoded_absolute_actions) == HORIZON
    for left, right in zip(
        first.policy_chunk_result.decoded_absolute_actions,
        second.policy_chunk_result.decoded_absolute_actions,
        strict=True,
    ):
        np.testing.assert_array_equal(left.vector, right.vector)
    assert first.policy_chunk_result.execution_horizon_steps == 3
    assert second.policy_chunk_result.execution_horizon_steps == 3


def test_model_neutral_client_cancel_hook_is_optional_and_forwarded_once() -> None:
    without_hook = ModelAgnosticRtcClient(
        backend=_LocalArrayBackend("local-a"),
        decoder=_decoder,
        request_spec=_spec(),
    )
    assert not without_hook.request_cancel()

    class CancellableBackend(_LocalArrayBackend):
        def __init__(self) -> None:
            super().__init__("local-a")
            self.cancel_count = 0

        def request_cancel(self) -> bool:
            self.cancel_count += 1
            return True

    backend = CancellableBackend()
    client = ModelAgnosticRtcClient(
        backend=backend,
        decoder=_decoder,
        request_spec=_spec(),
    )
    assert client.request_cancel()
    assert backend.cancel_count == 1


def test_model_neutral_client_rejects_untyped_cancel_outcome() -> None:
    class InvalidCancellationBackend(_LocalArrayBackend):
        def request_cancel(self):
            return None

    client = ModelAgnosticRtcClient(
        backend=InvalidCancellationBackend("local-a"),
        decoder=_decoder,
        request_spec=_spec(),
    )
    with pytest.raises(TypeError, match="must return bool"):
        client.request_cancel()


def test_client_result_is_deterministic_for_same_backend_exchange() -> None:
    client = ModelAgnosticRtcClient(
        backend=_LocalArrayBackend("local-a"),
        decoder=_decoder,
        request_spec=_spec(),
    )
    kwargs = {
        "model_input": {"base": 0.25},
        "model_input_hash": "model-input-hash:unit",
        "prefix_mask": _mask(),
    }
    first = client.infer(_request(), **kwargs)
    second = client.infer(_request(), **kwargs)
    assert first.envelope.semantic_content_hash == second.envelope.semantic_content_hash
    assert first.response.semantic_content_hash == second.response.semantic_content_hash
    assert (
        first.policy_chunk_result.semantic_content_hash
        == second.policy_chunk_result.semantic_content_hash
    )


def test_client_detects_capability_drift_before_backend_invocation() -> None:
    backend = _LocalArrayBackend("local-a")
    client = ModelAgnosticRtcClient(
        backend=backend,
        decoder=_decoder,
        request_spec=_spec(),
    )
    backend._capability = _capability("local-b")
    with pytest.raises(RtcClientProtocolError) as exc_info:
        client.infer(
            _request(),
            model_input={"base": 0.0},
            model_input_hash="model-input-hash:unit",
            prefix_mask=_mask(),
        )
    assert exc_info.value.reasons == ("capability_hash_changed",)


def test_rtc4a_modules_have_no_model_transport_sim_or_hardware_imports() -> None:
    project_root = Path(__file__).resolve().parents[1]
    paths = (
        project_root / "robot_motion/deploy/rtc_protocol.py",
        project_root / "deployments/client/rtc.py",
    )
    forbidden = (
        "torch",
        "jax",
        "websocket",
        "deployments.openpi",
        "teleop_runtime",
        "sim",
        "hardware",
    )
    imports: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
    assert not [
        name
        for name in imports
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in forbidden)
    ]
