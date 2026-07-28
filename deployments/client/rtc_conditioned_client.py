"""Validated model-RTC v3 client with exactly-once output cropping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_motion.deploy.canonical_policy_action import CanonicalPolicyAction
from robot_motion.deploy.inference import (
    PolicyChunkResult,
    PolicyInferenceRequest,
)
from robot_motion.deploy.rtc.conditioning import (
    CanonicalRtcConditioningBundle,
)

from .rtc import RtcClientProtocolError, RtcModelSpaceDecoder
from .rtc_conditioning import (
    RtcConditioningCodec,
    RtcEncodedConditioning,
    encode_rtc_conditioning_bundle,
)
from .rtc_conditioning_protocol import (
    RtcConditionedInferenceEnvelopeV3,
    RtcConditionedModelBackendV3,
    RtcConditionedModelResponseV3,
    RtcConditionedResponseValidationV3,
    RtcModelContinuationCapabilityV3,
    build_rtc_conditioned_inference_envelope_v3,
    validate_rtc_conditioned_response_v3,
)


@dataclass(frozen=True)
class RtcConditionedClientInferenceBundleV3:
    canonical_conditioning: CanonicalRtcConditioningBundle
    envelope: RtcConditionedInferenceEnvelopeV3
    response: RtcConditionedModelResponseV3
    response_validation: RtcConditionedResponseValidationV3
    policy_chunk_result: PolicyChunkResult

    def __post_init__(self) -> None:
        envelope_hash = self.envelope.semantic_content_hash
        response_hash = self.response.semantic_content_hash
        if not self.response_validation.accepted:
            raise ValueError(
                "A conditioned client bundle requires an accepted response."
            )
        if (
            self.canonical_conditioning.conditioning_binding_hash
            != self.envelope.request.conditioning_binding_hash
        ):
            raise ValueError(
                "Conditioned client canonical/envelope mismatch."
            )
        if (
            self.response_validation.envelope_hash != envelope_hash
            or self.response_validation.response_hash != response_hash
        ):
            raise ValueError(
                "Conditioned client validation evidence mismatch."
            )
        if self.policy_chunk_result.request_hash != (
            self.envelope.request.semantic_content_hash
        ):
            raise ValueError(
                "Conditioned client result/request mismatch."
            )
        if (
            self.policy_chunk_result.conditioning_binding_hash
            != self.canonical_conditioning.conditioning_binding_hash
        ):
            raise ValueError(
                "Conditioned client result/canonical binding mismatch."
            )
        if (
            self.policy_chunk_result.conditioning_encoding_receipt_hash
            != self.encoded_conditioning.receipt.semantic_content_hash
        ):
            raise ValueError(
                "Conditioned client result/encoding receipt mismatch."
            )
        if (
            self.policy_chunk_result.conditioning_consumption_receipt_hash
            != self.response.consumption_receipt.semantic_content_hash
        ):
            raise ValueError(
                "Conditioned client result/consumption receipt mismatch."
            )
        if (
            self.policy_chunk_result.model_output_hash
            != self.response.model_output_hash
        ):
            raise ValueError(
                "Conditioned client result/model output mismatch."
            )
        if (
            self.policy_chunk_result.protocol_version
            != self.response.protocol_version
        ):
            raise ValueError(
                "Conditioned client result/protocol version mismatch."
            )

    @property
    def generation_horizon_steps(self) -> int:
        """Full model output horizon before the generation-prefix crop."""

        return self.envelope.request.prediction_horizon_steps

    @property
    def encoded_conditioning(self) -> RtcEncodedConditioning:
        """Exact encoded prefix owned by the inference envelope."""

        return self.envelope.encoded_conditioning

    @property
    def capability(self) -> RtcModelContinuationCapabilityV3:
        """Loaded capability owned by the inference envelope."""

        return self.envelope.capability

    @property
    def conditioning_window_steps(self) -> int:
        """Generation prefix reserved for conditioning, selected or empty."""

        return self.envelope.request.forecast_delay_steps

    @property
    def selected_prefix_steps(self) -> int:
        """Actually selected producer-overlap rows within the window."""

        return len(
            self.encoded_conditioning.receipt.selected_generation_indices
        )

    @property
    def generation_prefix_crop_steps(self) -> int:
        """Rows removed exactly once after validating the full output."""

        return self.envelope.request.forecast_delay_steps

    @property
    def executable_candidate_horizon_steps(self) -> int:
        """Candidate rows offered to timed-plan admission, before obsolescence."""

        return (
            self.generation_horizon_steps
            - self.generation_prefix_crop_steps
        )


class RtcConditionedModelClientV3:
    """Invoke a loaded v3 backend without owning a worker or timed plan."""

    def __init__(
        self,
        *,
        backend: RtcConditionedModelBackendV3,
        codec: RtcConditioningCodec,
        decoder: RtcModelSpaceDecoder,
    ) -> None:
        if not isinstance(backend, RtcConditionedModelBackendV3):
            raise TypeError(
                "backend must implement RtcConditionedModelBackendV3."
            )
        if not isinstance(codec, RtcConditioningCodec):
            raise TypeError("codec must implement RtcConditioningCodec.")
        if not callable(decoder):
            raise TypeError("decoder must be callable.")
        capability = backend.rtc_continuation_capability
        if not isinstance(capability, RtcModelContinuationCapabilityV3):
            raise TypeError(
                "backend capability must be "
                "RtcModelContinuationCapabilityV3."
            )
        self._backend = backend
        self._codec = codec
        self._decoder = decoder
        self._capability = capability

    @property
    def capability(self) -> RtcModelContinuationCapabilityV3:
        return self._capability

    def request_cancel(self) -> bool:
        hook = getattr(self._backend, "request_cancel", None)
        if not callable(hook):
            return False
        accepted = hook()
        if not isinstance(accepted, bool):
            raise TypeError("RTC backend request_cancel() must return bool.")
        return accepted

    def infer(
        self,
        request: PolicyInferenceRequest,
        *,
        canonical_conditioning: CanonicalRtcConditioningBundle,
        model_input: Any,
        model_input_hash: str,
    ) -> RtcConditionedClientInferenceBundleV3:
        current_capability = self._backend.rtc_continuation_capability
        if (
            current_capability.semantic_content_hash
            != self._capability.semantic_content_hash
        ):
            raise RtcClientProtocolError(
                "conditioned capability drift",
                ("capability_hash_changed",),
            )
        encoded = encode_rtc_conditioning_bundle(
            self._codec,
            canonical_conditioning,
        )
        envelope = build_rtc_conditioned_inference_envelope_v3(
            request=request,
            bundle=canonical_conditioning,
            capability=self._capability,
            encoded_conditioning=encoded,
            model_input_hash=model_input_hash,
        )
        response = self._backend.infer_conditioned_rtc_v3(
            envelope,
            model_input,
        )
        if not isinstance(response, RtcConditionedModelResponseV3):
            raise RtcClientProtocolError(
                "conditioned model response",
                ("response_type_mismatch",),
            )
        validation = validate_rtc_conditioned_response_v3(
            envelope=envelope,
            response=response,
        )
        if not validation.accepted:
            raise RtcClientProtocolError(
                "conditioned model response",
                validation.reasons,
            )

        # This is the sole generation-prefix crop.  Timed-plan admission may
        # later discard rows that became obsolete at the actual cursor, but it
        # must not repeat this model-output operation.
        delay = request.forecast_delay_steps
        executable_model_actions = np.asarray(
            response.model_space_actions[delay:],
            dtype=np.float32,
        )
        decoded = tuple(
            self._decoder(
                executable_model_actions,
                None,
                request,
            )
        )
        expected_count = (
            request.prediction_horizon_steps - request.forecast_delay_steps
        )
        if len(decoded) != expected_count:
            raise RtcClientProtocolError(
                "conditioned model-space decoding",
                ("decoded_action_count_mismatch",),
            )
        if any(
            not isinstance(action, CanonicalPolicyAction)
            for action in decoded
        ):
            raise RtcClientProtocolError(
                "conditioned model-space decoding",
                ("typed_canonical_action_required",),
            )
        canonical_actions = tuple(
            action for action in decoded
            if isinstance(action, CanonicalPolicyAction)
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
            source_chunk_size=len(canonical_actions),
            decoded_absolute_actions=(),
            canonical_actions=canonical_actions,
            model_output_hash=response.model_output_hash,
            protocol_version=response.protocol_version,
            effective_delay_steps=delay,
            execution_horizon_steps=len(canonical_actions),
            conditioning_binding_hash=(
                canonical_conditioning.conditioning_binding_hash
            ),
            conditioning_encoding_receipt_hash=(
                encoded.receipt.semantic_content_hash
            ),
            conditioning_consumption_receipt_hash=(
                response.consumption_receipt.semantic_content_hash
            ),
        )
        return RtcConditionedClientInferenceBundleV3(
            canonical_conditioning=canonical_conditioning,
            envelope=envelope,
            response=response,
            response_validation=validation,
            policy_chunk_result=result,
        )
