"""Model-family-neutral RTC client seam.

This module connects a negotiated protocol envelope to an arbitrary local or
remote backend and a model-space decoder.  It does not schedule workers, own a
transport, or know about OpenPI, simulation, teleoperation, or hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

from robot_motion.deploy.actions import NormalizedAction
from robot_motion.deploy.action_semantics import ArmHoldKeyframePrediction
from robot_motion.deploy.canonical_policy_action import CanonicalPolicyAction
from robot_motion.deploy.inference import PolicyChunkResult, PolicyInferenceRequest
from robot_motion.deploy.rtc_protocol import (
    RtcCapabilityNegotiationReceipt,
    RtcConditioningReceiptV2,
    RtcInferenceEnvelopeV2,
    RtcModelCapability,
    RtcModelResponseEnvelopeV2,
    RtcProtocolRequestSpec,
    RtcResponseValidationReceipt,
    build_rtc_inference_envelope,
    negotiate_rtc_capability,
    validate_rtc_model_response,
)


@runtime_checkable
class RtcModelBackend(Protocol):
    """Smallest backend contract shared by local and remote implementations."""

    @property
    def rtc_capability(self) -> RtcModelCapability:
        ...

    def infer_rtc(
        self,
        envelope: RtcInferenceEnvelopeV2,
        model_input: Any,
    ) -> RtcModelResponseEnvelopeV2:
        ...


@runtime_checkable
class RtcModelSpaceDecoder(Protocol):
    """Decode model rows to exactly one legacy or typed representation."""

    def __call__(
        self,
        model_space_actions: np.ndarray,
        model_space_discrete_actions: np.ndarray | None,
        request: PolicyInferenceRequest,
    ) -> Sequence[NormalizedAction | CanonicalPolicyAction]:
        ...


class RtcClientProtocolError(RuntimeError):
    """Fail-closed capability or response validation failure."""

    def __init__(self, phase: str, reasons: Sequence[str]) -> None:
        normalized = tuple(str(reason) for reason in reasons)
        self.phase = str(phase)
        self.reasons = normalized
        super().__init__(f"RTC {self.phase} rejected: {', '.join(normalized)}")


@dataclass(frozen=True)
class RtcClientInferenceBundle:
    """Auditable raw/decoded result for later control-thread plan admission."""

    negotiation: RtcCapabilityNegotiationReceipt
    envelope: RtcInferenceEnvelopeV2
    response: RtcModelResponseEnvelopeV2
    response_validation: RtcResponseValidationReceipt
    conditioning_receipt: RtcConditioningReceiptV2 | None
    policy_chunk_result: PolicyChunkResult

    def __post_init__(self) -> None:
        if not self.negotiation.accepted:
            raise ValueError("An inference bundle requires accepted capability negotiation.")
        if not self.response_validation.accepted:
            raise ValueError("An inference bundle requires an accepted model response.")
        if self.conditioning_receipt is not self.response.conditioning_receipt:
            raise ValueError("Bundle conditioning receipt must be the response receipt.")
        if self.policy_chunk_result.request_hash != self.envelope.request.semantic_content_hash:
            raise ValueError("Bundle result/request hash mismatch.")


class ModelAgnosticRtcClient:
    """Validate, invoke, decode, and return facts with no plan-admission power."""

    def __init__(
        self,
        *,
        backend: RtcModelBackend,
        decoder: RtcModelSpaceDecoder,
        request_spec: RtcProtocolRequestSpec,
    ) -> None:
        self._backend = backend
        self._decoder = decoder
        self._request_spec = request_spec
        self._capability = backend.rtc_capability
        if not isinstance(self._capability, RtcModelCapability):
            raise TypeError("backend.rtc_capability must be an RtcModelCapability.")
        self._negotiation = negotiate_rtc_capability(
            request_spec,
            self._capability,
        )
        if not self._negotiation.accepted:
            raise RtcClientProtocolError(
                "capability negotiation",
                self._negotiation.rejection_reasons,
            )

    @property
    def request_spec(self) -> RtcProtocolRequestSpec:
        return self._request_spec

    @property
    def capability(self) -> RtcModelCapability:
        return self._capability

    @property
    def negotiation(self) -> RtcCapabilityNegotiationReceipt:
        return self._negotiation

    def infer(
        self,
        request: PolicyInferenceRequest,
        *,
        model_input: Any,
        model_input_hash: str,
        prefix_mask: np.ndarray | None = None,
    ) -> RtcClientInferenceBundle:
        """Run one exchange; caller still owns worker and plan admission."""

        context = request.action_representation_context
        if context is not None and (
            context.source_pose_mode != self._request_spec.action_pose_mode
        ):
            raise RtcClientProtocolError(
                "request representation",
                ("action_pose_mode_mismatch",),
            )
        if self._request_spec.action_pose_mode == "relative" and context is None:
            raise RtcClientProtocolError(
                "request representation",
                ("relative_representation_context_missing",),
            )
        current_capability = self._backend.rtc_capability
        if current_capability.semantic_content_hash != self._capability.semantic_content_hash:
            raise RtcClientProtocolError("capability drift", ("capability_hash_changed",))
        envelope = build_rtc_inference_envelope(
            request=request,
            request_spec=self._request_spec,
            capability=self._capability,
            negotiation=self._negotiation,
            model_input_hash=model_input_hash,
            prefix_mask=prefix_mask,
        )
        response = self._backend.infer_rtc(envelope, model_input)
        if not isinstance(response, RtcModelResponseEnvelopeV2):
            raise RtcClientProtocolError(
                "model response",
                ("response_type_mismatch",),
            )
        validation = validate_rtc_model_response(
            envelope=envelope,
            response=response,
            capability=self._capability,
        )
        if not validation.accepted:
            raise RtcClientProtocolError("model response", validation.rejection_reasons)
        decoded = tuple(
            self._decoder(
                np.asarray(response.model_space_actions, dtype=np.float32),
                (
                    None
                    if response.model_space_discrete_actions is None
                    else np.asarray(
                        response.model_space_discrete_actions,
                        dtype=np.int64,
                    )
                ),
                request,
            )
        )
        if len(decoded) != response.model_space_actions.shape[0]:
            raise RtcClientProtocolError(
                "model-space decoding",
                ("decoded_action_count_mismatch",),
            )
        legacy_decoded = tuple(
            action for action in decoded if isinstance(action, NormalizedAction)
        )
        canonical_decoded = tuple(
            action for action in decoded if isinstance(action, CanonicalPolicyAction)
        )
        if len(legacy_decoded) + len(canonical_decoded) != len(decoded):
            raise RtcClientProtocolError(
                "model-space decoding",
                ("decoded_action_type_mismatch",),
            )
        if legacy_decoded and canonical_decoded:
            raise RtcClientProtocolError(
                "model-space decoding",
                ("mixed_decoded_action_representations",),
            )
        if (
            self._request_spec.action_pose_mode == "relative"
            and legacy_decoded
        ):
            raise RtcClientProtocolError(
                "model-space decoding",
                ("relative_requires_typed_canonical_actions",),
            )
        if canonical_decoded:
            source_modes = {
                action.canonicalization_receipt.source_pose_mode
                for action in canonical_decoded
            }
            if source_modes != {self._request_spec.action_pose_mode}:
                raise RtcClientProtocolError(
                    "model-space decoding",
                    ("decoded_action_pose_mode_mismatch",),
                )
            if self._request_spec.action_pose_mode == "relative":
                assert context is not None
                if any(
                    action.canonicalization_receipt.anchor_identity_hash
                    != context.semantic_content_hash
                    for action in canonical_decoded
                ):
                    raise RtcClientProtocolError(
                        "model-space decoding",
                        ("decoded_anchor_identity_mismatch",),
                    )
        if response.arm_hold_probabilities is not None:
            if canonical_decoded:
                raise RtcClientProtocolError(
                    "model-space decoding",
                    ("arm_hold_typed_canonical_unsupported",),
                )
            contract_hash = response.arm_hold_output_contract_hash
            assert contract_hash is not None
            probabilities = np.asarray(
                response.arm_hold_probabilities,
                dtype=np.float32,
            )
            legacy_decoded = tuple(
                replace(
                    action,
                    arm_hold_prediction=ArmHoldKeyframePrediction(
                        probabilities=(
                            float(probabilities[index, 0]),
                            float(probabilities[index, 1]),
                        ),
                        contract_hash=contract_hash,
                    ),
                )
                for index, action in enumerate(legacy_decoded)
            )
        response_hash = response.semantic_content_hash
        result = PolicyChunkResult(
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
            policy_output_id=response.model_output_id,
            chunk_id=int(response_hash[:16], 16),
            inference_started_at=response.inference_started_at,
            inference_completed_at=response.inference_completed_at,
            result_received_at=response.result_received_at,
            clock_domain_id=request.clock_domain_id,
            status="ready",
            source_chunk_size=len(decoded),
            decoded_absolute_actions=legacy_decoded,
            model_output_hash=response_hash,
            protocol_version=response.protocol_version,
            conditioning_prefix_hash=request.conditioning_prefix_hash,
            effective_delay_steps=response.effective_delay_steps,
            execution_horizon_steps=response.execution_horizon_steps,
            canonical_actions=canonical_decoded,
        )
        return RtcClientInferenceBundle(
            negotiation=self._negotiation,
            envelope=envelope,
            response=response,
            response_validation=validation,
            conditioning_receipt=response.conditioning_receipt,
            policy_chunk_result=result,
        )

    def request_cancel(self) -> bool:
        """Forward an optional non-blocking backend session-cancel signal.

        ``True`` only means the backend accepted the signal.  It never proves
        that a model call, transport, worker thread, or execution target has
        drained.  Backends without this optional hook retain their existing
        behavior and return ``False``.
        """

        hook = getattr(self._backend, "request_cancel", None)
        if not callable(hook):
            return False
        accepted = hook()
        if not isinstance(accepted, bool):
            raise TypeError("RTC backend request_cancel() must return bool.")
        return accepted


__all__ = [
    "ModelAgnosticRtcClient",
    "RtcClientInferenceBundle",
    "RtcClientProtocolError",
    "RtcModelBackend",
    "RtcModelSpaceDecoder",
]
