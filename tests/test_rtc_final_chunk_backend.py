from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from dataset_contracts.canonical_encoding import canonical_content_hash
from deployments.client import (
    FinalChunkModelOutput,
    FinalChunkRtcBackend,
    FinalChunkRtcBinding,
    MappingFinalChunkOutputAdapter,
    ModelAgnosticRtcClient,
    RtcModelRuntimeAdapter,
    RtcPreparedModelInput,
    create_final_chunk_rtc_runtime_adapter,
)
from robot_motion.deploy import DEPLOY_ACTION_DIM, NormalizedAction, nan_vector
from robot_motion.deploy.inference import (
    ASYNC_ALIGNED_CHUNK_MODE_V1,
    PolicyInferenceRequest,
)
from robot_motion.deploy.observation_view import PolicyObservationView


HORIZON = 3


def _binding(*, discrete_dim: int = 1) -> FinalChunkRtcBinding:
    return FinalChunkRtcBinding(
        implementation_id="final-chunk-unit-v1",
        implementation_parameters_hash="parameters:final-chunk-unit-v1",
        provider_id="provider:final-chunk-unit-v1",
        artifact_hash="artifact:final-chunk-unit-v1",
        model_action_space="model-space:final-chunk-unit-v1",
        model_action_dim=2,
        model_action_discrete_dim=discrete_dim,
        action_pose_mode="absolute",
        component_scope=("right_arm", "right_hand"),
        action_codec_hash="codec:final-chunk-unit-v1",
        frame_contract_hash="frame:final-chunk-unit-v1",
        layout_hash="layout:final-chunk-unit-v1",
        normalization_hash="normalization:final-chunk-unit-v1",
        prediction_horizon_steps=HORIZON,
        minimum_execution_horizon_steps=2,
    )


def _request(backend: FinalChunkRtcBackend) -> PolicyInferenceRequest:
    spec = backend.request_spec
    return PolicyInferenceRequest(
        request_id="request:final-chunk-unit",
        inference_id="inference:final-chunk-unit",
        rtc_mode=ASYNC_ALIGNED_CHUNK_MODE_V1,
        session_id="session:final-chunk-unit",
        control_domain_id="domain:final-chunk-unit",
        producer_id="producer:final-chunk-unit",
        producer_epoch=1,
        policy_node_id="node:final-chunk-unit",
        artifact_hash=spec.artifact_hash,
        observation_cut_id="observation:final-chunk-unit",
        observation_timestamp=9.9,
        observation_step_index=3,
        triggered_at=10.0,
        admitted_at=10.0,
        first_target_effective_at=10.1,
        clock_domain_id="monotonic:unit",
        action_period_sec=0.1,
        prediction_horizon_steps=spec.prediction_horizon_steps,
        minimum_execution_horizon_steps=spec.minimum_execution_horizon_steps,
        source_plan_id="plan:final-chunk-unit",
        source_plan_generation=0,
        source_plan_hash="plan-hash:final-chunk-unit",
        source_plan_cursor_at=10.0,
        producer_plan_committed_through=None,
        forecast_delay_steps=1,
        delay_estimator_spec_hash="delay:final-chunk-unit",
        action_codec_hash=spec.action_codec_hash,
        frame_contract_hash=spec.frame_contract_hash,
        layout_hash=spec.layout_hash,
        normalization_hash=spec.normalization_hash,
        valid_until=11.0,
    )


def _input_hash(value: Any) -> str:
    return canonical_content_hash(
        value,
        domain="dexteleop.test.final_chunk_input.v1",
    )


def _clock():
    values = iter((10.01, 10.02, 10.03))
    return lambda: next(values)


def test_generic_final_chunk_backend_preserves_continuous_and_discrete_rows() -> None:
    actions = np.arange(HORIZON * 2, dtype=np.float32).reshape(HORIZON, 2)
    discrete = np.arange(HORIZON, dtype=np.int64).reshape(HORIZON, 1)
    decoded_discrete = []

    def decode(model_actions, discrete_actions, request):
        del request
        np.testing.assert_array_equal(model_actions, actions)
        assert discrete_actions is not None
        decoded_discrete.append(discrete_actions.copy())
        result = []
        for row in model_actions:
            vector = nan_vector(DEPLOY_ACTION_DIM)
            vector[0:2] = row
            result.append(NormalizedAction(vector=vector))
        return tuple(result)

    backend = FinalChunkRtcBackend(
        binding=_binding(),
        infer_model=lambda _value: FinalChunkModelOutput(
            actions=actions,
            discrete_actions=discrete,
            source_output_id="source-output:unit",
        ),
        hash_model_input=_input_hash,
        clock=_clock(),
    )
    client = ModelAgnosticRtcClient(
        backend=backend,
        decoder=decode,
        request_spec=backend.request_spec,
    )
    model_input = {"state": np.zeros(2, dtype=np.float32)}
    bundle = client.infer(
        _request(backend),
        model_input=model_input,
        model_input_hash=_input_hash(model_input),
    )

    assert bundle.response_validation.accepted
    assert len(bundle.policy_chunk_result.decoded_absolute_actions) == HORIZON
    assert len(decoded_discrete) == 1
    np.testing.assert_array_equal(decoded_discrete[0], discrete)


def test_generic_final_chunk_backend_rejects_input_hash_before_model_call() -> None:
    calls = []
    backend = FinalChunkRtcBackend(
        binding=_binding(discrete_dim=0),
        infer_model=lambda value: calls.append(value),
        hash_model_input=_input_hash,
        clock=_clock(),
    )
    client = ModelAgnosticRtcClient(
        backend=backend,
        decoder=lambda *_args: (),
        request_spec=backend.request_spec,
    )

    with pytest.raises(ValueError, match="input hash does not match"):
        client.infer(
            _request(backend),
            model_input={"state": np.zeros(2, dtype=np.float32)},
            model_input_hash="model-input:wrong",
        )
    assert calls == []


def test_generic_final_chunk_backend_cancel_is_optional_and_typed() -> None:
    no_cancel = FinalChunkRtcBackend(
        binding=_binding(discrete_dim=0),
        infer_model=lambda _value: FinalChunkModelOutput(
            actions=np.zeros((HORIZON, 2), dtype=np.float32)
        ),
        hash_model_input=_input_hash,
    )
    assert not no_cancel.request_cancel()

    invalid_cancel = FinalChunkRtcBackend(
        binding=_binding(discrete_dim=0),
        infer_model=lambda _value: FinalChunkModelOutput(
            actions=np.zeros((HORIZON, 2), dtype=np.float32)
        ),
        hash_model_input=_input_hash,
        request_cancel=lambda: "yes",
    )
    with pytest.raises(TypeError, match="must return bool"):
        invalid_cancel.request_cancel()


def test_sessionless_mapping_model_composes_without_request_spec_copy() -> None:
    actions = np.arange(HORIZON * 2, dtype=np.float32).reshape(HORIZON, 2)
    discrete = np.arange(HORIZON, dtype=np.int64).reshape(HORIZON, 1)
    model_calls: list[dict[str, Any]] = []
    cancel_calls: list[bool] = []

    def prepare_input(_request, observation):
        model_input = observation.model_input()
        return RtcPreparedModelInput(
            model_input=model_input,
            model_input_hash=_input_hash(model_input),
        )

    def call_model(model_input):
        model_calls.append(model_input)
        return {
            "chunk": actions,
            "tokens": discrete,
            "output_id": "server-output:unit",
            "server_timing": {"infer_sec": 0.01},
        }

    decoded_discrete = []

    def decode(model_actions, discrete_actions, _request):
        assert discrete_actions is not None
        decoded_discrete.append(discrete_actions.copy())
        decoded = []
        for row in model_actions:
            vector = nan_vector(DEPLOY_ACTION_DIM)
            vector[:2] = row
            decoded.append(NormalizedAction(vector=vector))
        return tuple(decoded)

    adapter = create_final_chunk_rtc_runtime_adapter(
        binding=_binding(),
        prepare_input=prepare_input,
        call_model=call_model,
        hash_model_input=_input_hash,
        decoder=decode,
        adapt_output=MappingFinalChunkOutputAdapter(
            actions_key="chunk",
            discrete_actions_key="tokens",
            source_output_id_key="output_id",
        ),
        request_cancel=lambda: cancel_calls.append(True) is None,
        clock=_clock(),
    )

    assert isinstance(adapter, RtcModelRuntimeAdapter)
    assert adapter.request_spec is adapter.backend.request_spec
    observation = PolicyObservationView(
        contract_id="contract:final-chunk-factory-unit",
        contract_hash="contract-hash:final-chunk-factory-unit",
        payload={"state": np.asarray([1.0, 2.0], dtype=np.float32)},
    )
    result = adapter.create_worker_executor()(
        _request(adapter.backend),
        observation,
    )

    assert len(model_calls) == 1
    assert result.source_chunk_size == HORIZON
    assert len(result.decoded_absolute_actions) == HORIZON
    assert len(decoded_discrete) == 1
    np.testing.assert_array_equal(decoded_discrete[0], discrete)
    assert adapter.create_worker_executor().request_cancel()
    assert cancel_calls == [True]


@pytest.mark.parametrize(
    ("binding", "output_adapter"),
    (
        (_binding(discrete_dim=1), MappingFinalChunkOutputAdapter()),
        (
            _binding(discrete_dim=0),
            MappingFinalChunkOutputAdapter(discrete_actions_key="tokens"),
        ),
    ),
)
def test_mapping_output_contract_matches_discrete_binding_at_construction(
    binding: FinalChunkRtcBinding,
    output_adapter: MappingFinalChunkOutputAdapter,
) -> None:
    with pytest.raises(ValueError, match="discrete output key"):
        create_final_chunk_rtc_runtime_adapter(
            binding=binding,
            prepare_input=lambda *_args: None,
            call_model=lambda _value: {},
            hash_model_input=_input_hash,
            decoder=lambda *_args: (),
            adapt_output=output_adapter,
        )


def test_mapping_output_adapter_rejects_ambiguous_or_missing_keys() -> None:
    with pytest.raises(ValueError, match="must be distinct"):
        MappingFinalChunkOutputAdapter(
            actions_key="output",
            source_output_id_key="output",
        )

    output_adapter = MappingFinalChunkOutputAdapter(actions_key="chunk")
    with pytest.raises(ValueError, match="missing exact output key"):
        output_adapter({"actions": np.zeros((HORIZON, 2), dtype=np.float32)})
    with pytest.raises(TypeError, match="must be a mapping"):
        output_adapter(np.zeros((HORIZON, 2), dtype=np.float32))
    with pytest.raises(TypeError, match="source output id"):
        MappingFinalChunkOutputAdapter(
            actions_key="chunk",
            source_output_id_key="output_id",
        )(
            {
                "chunk": np.zeros((HORIZON, 2), dtype=np.float32),
                "output_id": 7,
            }
        )


def test_typed_final_chunk_output_needs_no_mapping_adapter() -> None:
    calls = []

    def decode(model_actions, _discrete_actions, _request):
        decoded = []
        for row in model_actions:
            vector = nan_vector(DEPLOY_ACTION_DIM)
            vector[:2] = row
            decoded.append(NormalizedAction(vector=vector))
        return tuple(decoded)

    adapter = create_final_chunk_rtc_runtime_adapter(
        binding=_binding(discrete_dim=0),
        prepare_input=lambda *_args: None,
        call_model=lambda value: (
            calls.append(value)
            or FinalChunkModelOutput(
                actions=np.zeros((HORIZON, 2), dtype=np.float32),
            )
        ),
        hash_model_input=_input_hash,
        decoder=decode,
    )
    model_input = {"state": np.zeros(2, dtype=np.float32)}
    bundle = ModelAgnosticRtcClient(
        backend=adapter.backend,
        decoder=adapter.decoder,
        request_spec=adapter.request_spec,
    ).infer(
        _request(adapter.backend),
        model_input=model_input,
        model_input_hash=_input_hash(model_input),
    )

    assert bundle.response.model_space_actions.shape == (HORIZON, 2)
    assert calls == [model_input]
