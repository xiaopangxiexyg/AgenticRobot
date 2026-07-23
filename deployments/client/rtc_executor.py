"""Model-neutral adapter from an RTC client to a bounded inference worker.

This module owns no thread, timed plan, control loop, target backend or motion
authority.  It only prepares one immutable observation cut for a negotiated
``ModelAgnosticRtcClient`` exchange and returns the resulting
``PolicyChunkResult`` to the caller-owned worker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

from robot_motion.deploy.inference import PolicyChunkResult, PolicyInferenceRequest
from robot_motion.deploy.observation_view import PolicyObservationView

from .rtc import ModelAgnosticRtcClient, RtcClientInferenceBundle


def _nonempty(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


@dataclass(frozen=True)
class RtcPreparedModelInput:
    """One content-addressed model input prepared from an observation cut.

    ``model_input`` remains model-family-owned.  Its preparer must detach it
    from caller-mutated state before returning.  The protocol envelope binds
    the supplied semantic hash and the backend must echo that exact identity.
    """

    model_input: Any
    model_input_hash: str
    prefix_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_input_hash",
            _nonempty(self.model_input_hash, name="model_input_hash"),
        )
        if self.prefix_mask is None:
            return
        mask = np.asarray(self.prefix_mask, dtype=np.float32).copy()
        mask.setflags(write=False)
        object.__setattr__(self, "prefix_mask", mask)


@runtime_checkable
class RtcModelInputPreparer(Protocol):
    """Map a frozen actor observation to one model-family-specific input."""

    def __call__(
        self,
        request: PolicyInferenceRequest,
        actor_observation: PolicyObservationView,
    ) -> RtcPreparedModelInput:
        ...


class RtcModelClientInferenceExecutor:
    """Blocking callable suitable for the shared bounded RTC worker.

    The blocking backend call occurs only when the embedding worker invokes
    this object.  The executor never admits the result to a timed plan and
    never interacts with safety, reference generation, dispatch or hardware.
    """

    def __init__(
        self,
        *,
        client: ModelAgnosticRtcClient,
        prepare_input: RtcModelInputPreparer,
        observe_bundle: Callable[[RtcClientInferenceBundle], None] | None = None,
    ) -> None:
        if not isinstance(client, ModelAgnosticRtcClient):
            raise TypeError("client must be a ModelAgnosticRtcClient.")
        if not callable(prepare_input):
            raise TypeError("prepare_input must be callable.")
        if observe_bundle is not None and not callable(observe_bundle):
            raise TypeError("observe_bundle must be callable or None.")
        self.client = client
        self.prepare_input = prepare_input
        self.observe_bundle = observe_bundle

    def __call__(
        self,
        request: PolicyInferenceRequest,
        actor_observation: PolicyObservationView,
    ) -> PolicyChunkResult:
        if not isinstance(request, PolicyInferenceRequest):
            raise TypeError("request must be a PolicyInferenceRequest.")
        if not isinstance(actor_observation, PolicyObservationView):
            raise TypeError("actor_observation must be a PolicyObservationView.")
        prepared = self.prepare_input(request, actor_observation)
        if not isinstance(prepared, RtcPreparedModelInput):
            raise TypeError("prepare_input must return RtcPreparedModelInput.")
        bundle = self.client.infer(
            request,
            model_input=prepared.model_input,
            model_input_hash=prepared.model_input_hash,
            prefix_mask=prepared.prefix_mask,
        )
        if self.observe_bundle is not None:
            self.observe_bundle(bundle)
        return bundle.policy_chunk_result

    def request_cancel(self) -> bool:
        """Signal cancellation without claiming worker or transport drain."""

        return self.client.request_cancel()


def rtc_client_inference_bundle_debug_payload(
    bundle: RtcClientInferenceBundle,
) -> dict[str, object]:
    """Return non-authoritative, serializable exchange and latency facts.

    The projection contains only identities and timings already attested by
    the RTC envelope/response. It is for observability, not plan admission,
    safety, dispatch, or worker-drain evidence. Keeping it in the
    model-neutral client seam avoids framework-specific scheduler copies.
    """

    if not isinstance(bundle, RtcClientInferenceBundle):
        raise TypeError("bundle must be an RtcClientInferenceBundle.")
    request = bundle.envelope.request
    response = bundle.response
    result = bundle.policy_chunk_result
    inference_duration = float(
        response.inference_completed_at - response.inference_started_at
    )
    result_duration = float(
        response.result_received_at - response.inference_started_at
    )
    postprocess_duration = float(
        response.result_received_at - response.inference_completed_at
    )
    return {
        "schema": "dexteleop_rtc_client_exchange_v1",
        "non_authoritative": True,
        "request_id": request.request_id,
        "inference_id": request.inference_id,
        "observation_cut_id": request.observation_cut_id,
        "observation_step_index": int(request.observation_step_index),
        "request_hash": request.semantic_content_hash,
        "envelope_hash": bundle.envelope.semantic_content_hash,
        "model_input_hash": bundle.envelope.model_input_hash,
        "capability_hash": bundle.negotiation.capability_hash,
        "negotiation_receipt_hash": bundle.negotiation.semantic_content_hash,
        "response_hash": response.semantic_content_hash,
        "response_validation_hash": bundle.response_validation.semantic_content_hash,
        "conditioning_receipt_hash": (
            None
            if bundle.conditioning_receipt is None
            else bundle.conditioning_receipt.semantic_content_hash
        ),
        "model_output_id": response.model_output_id,
        "model_output_hash": result.model_output_hash,
        "chunk_id": result.chunk_id,
        "source_chunk_size": int(result.source_chunk_size),
        "execution_horizon_steps": int(result.execution_horizon_steps or 0),
        "protocol_version": response.protocol_version,
        "scheduler_mode": response.scheduler_mode,
        "continuation_mode": response.continuation_mode,
        "clock_domain_id": request.clock_domain_id,
        "inference_started_at": float(response.inference_started_at),
        "inference_completed_at": float(response.inference_completed_at),
        "result_received_at": float(response.result_received_at),
        "inference_duration_sec": inference_duration,
        "result_duration_sec": result_duration,
        "postprocess_duration_sec": postprocess_duration,
        "effective_delay_steps": int(result.effective_delay_steps or 0),
        "action_send_authority": False,
    }


__all__ = [
    "RtcModelClientInferenceExecutor",
    "RtcModelInputPreparer",
    "RtcPreparedModelInput",
    "rtc_client_inference_bundle_debug_payload",
]
