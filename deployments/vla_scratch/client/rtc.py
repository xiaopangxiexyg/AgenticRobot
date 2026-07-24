"""Additive final-chunk RTC adapter for the external vla-scratch server.

The adapter consumes only the existing ``infer(dict) -> dict`` client shape.
It does not import vla-scratch, ZMQ or Torch and does not require changes in
the baseline repository.  The caller supplies exact checkpoint, frame,
normalization and representation bindings because the current baseline server
does not advertise metadata.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.vla_scratch import (
    VLA_SCRATCH_ACTIONS_KEY,
    VlaScratchProcessedObservation,
    VlaScratchProcessedSampleSpec,
    build_vla_scratch_processed_observation,
)
from deployments.client.rtc import (
    RtcModelSpaceDecoder,
)
from deployments.client.rtc_executor import (
    RtcModelInputPreparer,
    RtcPreparedModelInput,
)
from deployments.client.rtc_final_chunk import (
    FinalChunkRtcBackend,
    FinalChunkRtcBinding,
)
from deployments.client.rtc_final_chunk_adapter import (
    MappingFinalChunkOutputAdapter,
    create_final_chunk_rtc_runtime_adapter,
)
from deployments.client.rtc_frontend import SimpleRtcModelRuntimeAdapter
from robot_motion.deploy.inference import PolicyInferenceRequest
from robot_motion.deploy.observation_view import PolicyObservationView


VLA_SCRATCH_FINAL_CHUNK_RTC_ADAPTER_V1 = (
    "vla_scratch_final_chunk_rtc_adapter_v1"
)
VLA_SCRATCH_IMPLEMENTATION_HASH_DOMAIN_V1 = (
    "dexteleop.vla_scratch_rtc_implementation.v1"
)


def _nonempty(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


@runtime_checkable
class VlaScratchPolicyTransport(Protocol):
    """Existing vla-scratch client surface used through dependency injection."""

    def infer(self, observation: dict[str, Any]) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class VlaScratchFinalChunkRtcConfig:
    """Explicit binding facts absent from the current vla-scratch handshake."""

    sample_spec: VlaScratchProcessedSampleSpec
    provider_id: str
    artifact_hash: str
    action_codec_hash: str
    frame_contract_hash: str
    layout_hash: str
    normalization_hash: str
    component_scope: tuple[str, ...]
    action_pose_mode: str = "absolute"
    minimum_execution_horizon_steps: int = 1
    output_key: str = VLA_SCRATCH_ACTIONS_KEY

    def __post_init__(self) -> None:
        if not isinstance(self.sample_spec, VlaScratchProcessedSampleSpec):
            raise TypeError("sample_spec must be VlaScratchProcessedSampleSpec.")
        for name in (
            "provider_id",
            "artifact_hash",
            "action_codec_hash",
            "frame_contract_hash",
            "layout_hash",
            "normalization_hash",
            "action_pose_mode",
            "output_key",
        ):
            object.__setattr__(
                self,
                name,
                _nonempty(getattr(self, name), name=name),
            )
        scope = tuple(
            _nonempty(value, name="component_scope")
            for value in self.component_scope
        )
        if not scope or len(set(scope)) != len(scope):
            raise ValueError("component_scope must be non-empty and unique.")
        object.__setattr__(self, "component_scope", scope)
        minimum = self.minimum_execution_horizon_steps
        if isinstance(minimum, bool) or int(minimum) != minimum or minimum <= 0:
            raise ValueError(
                "minimum_execution_horizon_steps must be a positive integer."
            )
        minimum = int(minimum)
        if minimum > self.sample_spec.action_horizon:
            raise ValueError(
                "minimum_execution_horizon_steps exceeds action_horizon."
            )
        object.__setattr__(self, "minimum_execution_horizon_steps", minimum)

    @property
    def implementation_parameters_hash(self) -> str:
        return canonical_content_hash(
            {
                "adapter": VLA_SCRATCH_FINAL_CHUNK_RTC_ADAPTER_V1,
                "sample_spec_hash": self.sample_spec.semantic_content_hash,
                "output_key": self.output_key,
                "artifact_hash": self.artifact_hash,
                "action_codec_hash": self.action_codec_hash,
                "frame_contract_hash": self.frame_contract_hash,
                "layout_hash": self.layout_hash,
                "normalization_hash": self.normalization_hash,
            },
            domain=VLA_SCRATCH_IMPLEMENTATION_HASH_DOMAIN_V1,
        )

    @property
    def final_chunk_binding(self) -> FinalChunkRtcBinding:
        spec = self.sample_spec
        return FinalChunkRtcBinding(
            implementation_id=VLA_SCRATCH_FINAL_CHUNK_RTC_ADAPTER_V1,
            implementation_parameters_hash=self.implementation_parameters_hash,
            provider_id=self.provider_id,
            artifact_hash=self.artifact_hash,
            model_action_space=spec.action_representation_id,
            model_action_dim=spec.action_dim,
            model_action_discrete_dim=0,
            action_pose_mode=self.action_pose_mode,
            component_scope=self.component_scope,
            action_codec_hash=self.action_codec_hash,
            frame_contract_hash=self.frame_contract_hash,
            layout_hash=self.layout_hash,
            normalization_hash=self.normalization_hash,
            prediction_horizon_steps=spec.action_horizon,
            minimum_execution_horizon_steps=(
                self.minimum_execution_horizon_steps
            ),
        )


def _vla_scratch_model_seam(
    *,
    transport: VlaScratchPolicyTransport,
    config: VlaScratchFinalChunkRtcConfig,
) -> tuple[
    Callable[[Any], Mapping[str, Any]],
    Callable[[Any], str],
    Callable[[], bool] | None,
]:
    """Return only the framework-specific call/hash/cancel functions."""

    if not isinstance(transport, VlaScratchPolicyTransport):
        raise TypeError("transport must implement VlaScratchPolicyTransport.")
    if not isinstance(config, VlaScratchFinalChunkRtcConfig):
        raise TypeError("config must be VlaScratchFinalChunkRtcConfig.")
    expected_spec_hash = config.sample_spec.semantic_content_hash

    def validate_input(value: Any) -> VlaScratchProcessedObservation:
        if not isinstance(value, VlaScratchProcessedObservation):
            raise TypeError(
                "vla-scratch RTC backend requires VlaScratchProcessedObservation."
            )
        if value.spec.semantic_content_hash != expected_spec_hash:
            raise ValueError("vla-scratch processed sample spec changed.")
        return value

    def hash_model_input(value: Any) -> str:
        return validate_input(value).semantic_content_hash

    def call_model(value: Any) -> Mapping[str, Any]:
        prepared = validate_input(value)
        raw_response = transport.infer(prepared.to_wire_payload())
        if not isinstance(raw_response, Mapping):
            raise TypeError("vla-scratch infer response must be a mapping.")
        return raw_response

    cancel_hook = getattr(transport, "request_cancel", None)
    return (
        call_model,
        hash_model_input,
        cancel_hook if callable(cancel_hook) else None,
    )


class VlaScratchFinalChunkRtcBackend(FinalChunkRtcBackend):
    """Thin extraction seam over the shared final-chunk RTC backend."""

    def __init__(
        self,
        *,
        transport: VlaScratchPolicyTransport,
        config: VlaScratchFinalChunkRtcConfig,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable.")
        self._transport = transport
        self.config = config
        call_model, hash_model_input, cancel_hook = _vla_scratch_model_seam(
            transport=transport,
            config=config,
        )
        adapt_output = MappingFinalChunkOutputAdapter(
            actions_key=config.output_key,
        )
        super().__init__(
            binding=config.final_chunk_binding,
            infer_model=lambda value: adapt_output(call_model(value)),
            hash_model_input=hash_model_input,
            request_cancel=cancel_hook,
            clock=clock,
        )


@dataclass(frozen=True)
class VlaScratchRtcInputPreparer:
    """Build one exact processed sample from the actor-safe observation view."""

    sample_spec: VlaScratchProcessedSampleSpec
    task: str

    def __post_init__(self) -> None:
        if not isinstance(self.sample_spec, VlaScratchProcessedSampleSpec):
            raise TypeError("sample_spec must be VlaScratchProcessedSampleSpec.")
        object.__setattr__(self, "task", _nonempty(self.task, name="task"))

    def __call__(
        self,
        request: PolicyInferenceRequest,
        actor_observation: PolicyObservationView,
    ) -> RtcPreparedModelInput:
        if not isinstance(request, PolicyInferenceRequest):
            raise TypeError("request must be PolicyInferenceRequest.")
        if not isinstance(actor_observation, PolicyObservationView):
            raise TypeError("actor_observation must be PolicyObservationView.")
        prepared = build_vla_scratch_processed_observation(
            spec=self.sample_spec,
            images_by_source=actor_observation.images,
            state=actor_observation.state,
            task=self.task,
        )
        return RtcPreparedModelInput(
            model_input=prepared,
            model_input_hash=prepared.semantic_content_hash,
        )


def create_vla_scratch_runtime_adapter(
    *,
    transport: VlaScratchPolicyTransport,
    config: VlaScratchFinalChunkRtcConfig,
    decoder: RtcModelSpaceDecoder,
    task: str,
    clock: Callable[[], float] = time.monotonic,
) -> SimpleRtcModelRuntimeAdapter:
    """Compose through the generic final-chunk runtime adapter factory."""

    if not callable(clock):
        raise TypeError("clock must be callable.")
    call_model, hash_model_input, cancel_hook = _vla_scratch_model_seam(
        transport=transport,
        config=config,
    )
    preparer: RtcModelInputPreparer = VlaScratchRtcInputPreparer(
        sample_spec=config.sample_spec,
        task=task,
    )
    return create_final_chunk_rtc_runtime_adapter(
        binding=config.final_chunk_binding,
        prepare_input=preparer,
        call_model=call_model,
        hash_model_input=hash_model_input,
        decoder=decoder,
        adapt_output=MappingFinalChunkOutputAdapter(
            actions_key=config.output_key,
        ),
        request_cancel=cancel_hook,
        clock=clock,
    )


__all__ = [
    "VLA_SCRATCH_FINAL_CHUNK_RTC_ADAPTER_V1",
    "VlaScratchFinalChunkRtcBackend",
    "VlaScratchFinalChunkRtcConfig",
    "VlaScratchPolicyTransport",
    "VlaScratchRtcInputPreparer",
    "create_vla_scratch_runtime_adapter",
]
