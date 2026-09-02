"""Deployments-owned model-RTC v3 envelope and consumption receipt.

V3 is additive and does not reinterpret the frozen RTC v2 protocol.  The
backend receipt is valid only when it is produced at the sampler/model call
site that actually consumed the encoded prefix and mask.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.model_continuation import (
    MODEL_CONTINUATION_SAMPLER_CLAMPED_V1,
    ModelContinuationContract,
    normalize_model_continuation_contract,
)
from robot_motion.deploy.inference import PolicyInferenceRequest
from robot_motion.deploy.rtc.conditioning import (
    RTC_CONDITIONING_MODE_UNCONDITIONED_BOOTSTRAP_V1,
    CanonicalRtcConditioningBundle,
)

from .rtc_conditioning import (
    RtcConditioningCodecIdentity,
    RtcEncodedConditioning,
    rtc_binary_mask_hash,
    rtc_float32_array_hash,
)


RTC_MODEL_CONTINUATION_CAPABILITY_SCHEMA_V3 = (
    "rtc_model_continuation_capability_v3"
)
RTC_MODEL_CONTINUATION_CAPABILITY_HASH_DOMAIN_V3 = (
    "dexteleop.deployments.rtc_model_continuation_capability.v3"
)
RTC_CONDITIONED_INFERENCE_ENVELOPE_SCHEMA_V3 = (
    "rtc_conditioned_inference_envelope_v3"
)
RTC_CONDITIONED_INFERENCE_ENVELOPE_HASH_DOMAIN_V3 = (
    "dexteleop.deployments.rtc_conditioned_inference_envelope.v3"
)
RTC_CONDITIONING_CONSUMPTION_RECEIPT_SCHEMA_V3 = (
    "rtc_conditioning_consumption_receipt_v3"
)
RTC_CONDITIONING_CONSUMPTION_RECEIPT_HASH_DOMAIN_V3 = (
    "dexteleop.deployments.rtc_conditioning_consumption_receipt.v3"
)
RTC_CONDITIONED_MODEL_RESPONSE_SCHEMA_V3 = (
    "rtc_conditioned_model_response_v3"
)
RTC_CONDITIONED_MODEL_RESPONSE_HASH_DOMAIN_V3 = (
    "dexteleop.deployments.rtc_conditioned_model_response.v3"
)
RTC_CONDITIONED_RESPONSE_VALIDATION_SCHEMA_V3 = (
    "rtc_conditioned_response_validation_v3"
)
RTC_CONDITIONED_RESPONSE_VALIDATION_HASH_DOMAIN_V3 = (
    "dexteleop.deployments.rtc_conditioned_response_validation.v3"
)
RTC_MODEL_OUTPUT_ROWS_HASH_DOMAIN_V3 = (
    "dexteleop.deployments.rtc_model_output_rows.v3"
)
RTC_CONDITIONING_TRANSFORM_PROVIDER_V1 = "provider_v1"
RTC_CONDITIONING_TRANSFORM_CLIENT_V1 = "client_v1"
RTC_CONDITIONING_TRANSFORM_OWNERS = frozenset(
    {
        RTC_CONDITIONING_TRANSFORM_PROVIDER_V1,
        RTC_CONDITIONING_TRANSFORM_CLIENT_V1,
    }
)


def _nonempty(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer.")
    normalized = int(value)
    if normalized != value or normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return normalized


def _positive_int(value: object, *, name: str) -> int:
    normalized = _nonnegative_int(value, name=name)
    if normalized == 0:
        raise ValueError(f"{name} must be a positive integer.")
    return normalized


def _finite(value: object, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


@dataclass(frozen=True)
class RtcModelContinuationCapabilityV3:
    """Continuation capability exported by one loaded model instance.

    ``artifact_hash`` and ``continuation_contract_hash`` are retained as model
    and implementation provenance for diagnostics and receipts.  They are
    intentionally not part of the runtime capability identity: a client may
    exercise the same continuation protocol against a different server
    checkpoint or implementation revision when the declared runtime contracts
    remain compatible.  Runtime admission is represented by the codec,
    implementation parameters, schedule, dimensions, sampler, and bootstrap
    fields below.
    """

    artifact_hash: str
    continuation_contract_hash: str
    codec_identity_hash: str
    continuation_implementation_id: str
    continuation_implementation_parameters_hash: str
    sampler_constraint: str
    selected_prefix_schedule: str
    conditioning_transform_owner: str
    external_action_dim: int
    internal_sampler_action_dim: int
    supports_unconditioned_bootstrap: bool
    schema: str = RTC_MODEL_CONTINUATION_CAPABILITY_SCHEMA_V3

    def __post_init__(self) -> None:
        if self.schema != RTC_MODEL_CONTINUATION_CAPABILITY_SCHEMA_V3:
            raise ValueError(
                f"Unsupported model continuation capability {self.schema!r}."
            )
        for name in (
            "artifact_hash",
            "continuation_contract_hash",
            "codec_identity_hash",
            "continuation_implementation_id",
            "continuation_implementation_parameters_hash",
            "sampler_constraint",
            "selected_prefix_schedule",
            "conditioning_transform_owner",
        ):
            object.__setattr__(
                self,
                name,
                _nonempty(getattr(self, name), name=name),
            )
        if self.conditioning_transform_owner not in (
            RTC_CONDITIONING_TRANSFORM_OWNERS
        ):
            raise ValueError(
                "Unsupported conditioning_transform_owner "
                f"{self.conditioning_transform_owner!r}."
            )
        object.__setattr__(
            self,
            "external_action_dim",
            _positive_int(
                self.external_action_dim,
                name="external_action_dim",
            ),
        )
        object.__setattr__(
            self,
            "internal_sampler_action_dim",
            _positive_int(
                self.internal_sampler_action_dim,
                name="internal_sampler_action_dim",
            ),
        )
        if (
            self.conditioning_transform_owner
            == RTC_CONDITIONING_TRANSFORM_CLIENT_V1
            and self.external_action_dim != self.internal_sampler_action_dim
        ):
            raise ValueError(
                "client_v1 conditioning requires equal external and internal "
                "action dimensions."
            )
        if not isinstance(self.supports_unconditioned_bootstrap, bool):
            raise ValueError(
                "supports_unconditioned_bootstrap must be a boolean."
            )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "schema": self.schema,
                "codec_identity_hash": self.codec_identity_hash,
                "continuation_implementation_id": (
                    self.continuation_implementation_id
                ),
                "continuation_implementation_parameters_hash": (
                    self.continuation_implementation_parameters_hash
                ),
                "sampler_constraint": self.sampler_constraint,
                "selected_prefix_schedule": (
                    self.selected_prefix_schedule
                ),
                "conditioning_transform_owner": (
                    self.conditioning_transform_owner
                ),
                "external_action_dim": self.external_action_dim,
                "internal_sampler_action_dim": (
                    self.internal_sampler_action_dim
                ),
                "supports_unconditioned_bootstrap": (
                    self.supports_unconditioned_bootstrap
                ),
            },
            domain=RTC_MODEL_CONTINUATION_CAPABILITY_HASH_DOMAIN_V3,
        )


def validate_rtc_model_continuation_capability_v3_binding(
    capability: RtcModelContinuationCapabilityV3,
    *,
    continuation_contract: ModelContinuationContract,
    artifact_hash: str | None = None,
    codec_identity_hash: str | None = None,
    selected_prefix_schedule: str | None = None,
) -> None:
    """Validate one loaded capability against authoritative runtime facts.

    ``artifact_hash`` and ``continuation_contract_hash`` remain accepted for
    source compatibility and provenance, but are deliberately not compared.
    Runtime admission is governed by the continuation's actual implementation,
    codec, schedule, dimensions, sampler, and bootstrap contracts instead of a
    particular checkpoint or contract label.
    """

    if not isinstance(capability, RtcModelContinuationCapabilityV3):
        raise TypeError(
            "capability must be an RtcModelContinuationCapabilityV3."
        )
    contract = normalize_model_continuation_contract(
        continuation_contract
    ).runtime
    expected = {
        "continuation_implementation_id": (
            contract.continuation_implementation_id
        ),
        "continuation_implementation_parameters_hash": (
            contract.continuation_implementation_parameters_hash
        ),
        "external_action_dim": contract.model_action_dim,
    }
    if artifact_hash is not None:
        _nonempty(
            artifact_hash,
            name="artifact_hash",
        )
    if codec_identity_hash is not None:
        expected["codec_identity_hash"] = _nonempty(
            codec_identity_hash,
            name="codec_identity_hash",
        )
    if selected_prefix_schedule is not None:
        expected["selected_prefix_schedule"] = _nonempty(
            selected_prefix_schedule,
            name="selected_prefix_schedule",
        )
    for name, expected_value in expected.items():
        if getattr(capability, name) != expected_value:
            raise ValueError(f"Continuation capability {name} mismatch.")
    if capability.sampler_constraint not in contract.sampler_constraints:
        raise ValueError(
            "Continuation capability sampler constraint is unsupported."
        )
    if (
        contract.supports_unconditioned_bootstrap
        and not capability.supports_unconditioned_bootstrap
    ):
        raise ValueError(
            "Continuation capability does not support artifact bootstrap."
        )


def rtc_model_continuation_capability_from_loaded_codec(
    *,
    bundle: CanonicalRtcConditioningBundle,
    codec_identity: RtcConditioningCodecIdentity,
    sampler_constraint: str,
    conditioning_transform_owner: str,
    internal_sampler_action_dim: int,
) -> RtcModelContinuationCapabilityV3:
    """Construct a capability from loaded instances, never model names."""

    source_contract = bundle.continuation_contract
    contract = normalize_model_continuation_contract(
        source_contract
    ).runtime
    selected_constraint = _nonempty(
        sampler_constraint,
        name="sampler_constraint",
    )
    if selected_constraint not in contract.sampler_constraints:
        raise ValueError(
            "sampler_constraint is not supported by the continuation contract."
        )
    return RtcModelContinuationCapabilityV3(
        artifact_hash=bundle.artifact_hash,
        continuation_contract_hash=source_contract.contract_hash,
        codec_identity_hash=codec_identity.semantic_content_hash,
        continuation_implementation_id=(
            contract.continuation_implementation_id
        ),
        continuation_implementation_parameters_hash=(
            contract.continuation_implementation_parameters_hash
        ),
        sampler_constraint=selected_constraint,
        selected_prefix_schedule=bundle.prefix_schedule_semantics,
        conditioning_transform_owner=conditioning_transform_owner,
        external_action_dim=codec_identity.external_action_dim,
        internal_sampler_action_dim=internal_sampler_action_dim,
        supports_unconditioned_bootstrap=(
            contract.supports_unconditioned_bootstrap
        ),
    )


@dataclass(frozen=True)
class RtcConditionedInferenceEnvelopeV3:
    request: PolicyInferenceRequest
    capability: RtcModelContinuationCapabilityV3
    encoded_conditioning: RtcEncodedConditioning
    model_input_hash: str
    schema: str = RTC_CONDITIONED_INFERENCE_ENVELOPE_SCHEMA_V3

    def __post_init__(self) -> None:
        if self.schema != RTC_CONDITIONED_INFERENCE_ENVELOPE_SCHEMA_V3:
            raise ValueError(
                f"Unsupported conditioned inference envelope {self.schema!r}."
            )
        if not isinstance(self.request, PolicyInferenceRequest):
            raise TypeError("request must be a PolicyInferenceRequest.")
        if not isinstance(
            self.capability,
            RtcModelContinuationCapabilityV3,
        ):
            raise TypeError(
                "capability must be an RtcModelContinuationCapabilityV3."
            )
        if not isinstance(self.encoded_conditioning, RtcEncodedConditioning):
            raise TypeError(
                "encoded_conditioning must be an RtcEncodedConditioning."
            )
        object.__setattr__(
            self,
            "model_input_hash",
            _nonempty(self.model_input_hash, name="model_input_hash"),
        )
        binding = self.request.conditioning_binding_hash
        receipt = self.encoded_conditioning.receipt
        if binding is None or binding != receipt.conditioning_binding_hash:
            raise ValueError(
                "Request/encoded conditioning binding hash mismatch."
            )
        if (
            receipt.codec_identity_hash
            != self.capability.codec_identity_hash
        ):
            raise ValueError(
                "Encoding/continuation capability codec identity mismatch."
            )
        if (
            self.encoded_conditioning.external_prefix.shape[0]
            != self.request.prediction_horizon_steps
        ):
            raise ValueError(
                "Encoded conditioning/request prediction horizon mismatch."
            )
        if (
            self.encoded_conditioning.external_prefix.shape[1]
            != self.capability.external_action_dim
        ):
            raise ValueError(
                "Encoded conditioning/capability external action dimension "
                "mismatch."
            )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "schema": self.schema,
                "request_hash": self.request.semantic_content_hash,
                "capability_hash": self.capability.semantic_content_hash,
                "encoding_receipt_hash": (
                    self.encoded_conditioning.receipt.semantic_content_hash
                ),
                "model_input_hash": self.model_input_hash,
            },
            domain=RTC_CONDITIONED_INFERENCE_ENVELOPE_HASH_DOMAIN_V3,
        )


def build_rtc_conditioned_inference_envelope_v3(
    *,
    request: PolicyInferenceRequest,
    bundle: CanonicalRtcConditioningBundle,
    capability: RtcModelContinuationCapabilityV3,
    encoded_conditioning: RtcEncodedConditioning,
    model_input_hash: str,
) -> RtcConditionedInferenceEnvelopeV3:
    """Bind exact request, canonical bundle, encoding, and loaded capability."""

    source_contract = bundle.continuation_contract
    contract = normalize_model_continuation_contract(
        source_contract
    ).runtime
    validate_rtc_model_continuation_capability_v3_binding(
        capability,
        continuation_contract=source_contract,
        codec_identity_hash=(
            encoded_conditioning.receipt.codec_identity_hash
        ),
        selected_prefix_schedule=bundle.prefix_schedule_semantics,
    )
    if (
        bundle.conditioning_mode
        == RTC_CONDITIONING_MODE_UNCONDITIONED_BOOTSTRAP_V1
        and not capability.supports_unconditioned_bootstrap
    ):
        raise ValueError(
            "Continuation capability does not support bootstrap."
        )
    if (
        request.conditioning_binding_hash
        != bundle.conditioning_binding_hash
    ):
        raise ValueError("Request/canonical conditioning binding mismatch.")
    if (
        encoded_conditioning.receipt.conditioning_binding_hash
        != bundle.conditioning_binding_hash
    ):
        raise ValueError("Encoding/canonical conditioning binding mismatch.")
    return RtcConditionedInferenceEnvelopeV3(
        request=request,
        capability=capability,
        encoded_conditioning=encoded_conditioning,
        model_input_hash=model_input_hash,
    )


@dataclass(frozen=True)
class RtcConditioningConsumptionReceiptV3:
    """Provider assertion emitted where the sampler consumed conditioning."""

    envelope_hash: str
    request_hash: str
    capability_hash: str
    conditioning_binding_hash: str
    encoding_receipt_hash: str
    model_input_hash: str
    consumed_external_prefix_hash: str
    consumed_internal_prefix_hash: str
    consumed_mask_hash: str
    consumed_generation_indices: tuple[int, ...]
    sampler_constraint: str
    continuation_implementation_id: str
    continuation_implementation_parameters_hash: str
    effective_delay_steps: int
    prediction_horizon_steps: int
    schema: str = RTC_CONDITIONING_CONSUMPTION_RECEIPT_SCHEMA_V3

    def __post_init__(self) -> None:
        if self.schema != RTC_CONDITIONING_CONSUMPTION_RECEIPT_SCHEMA_V3:
            raise ValueError(
                f"Unsupported conditioning consumption receipt {self.schema!r}."
            )
        for name in (
            "envelope_hash",
            "request_hash",
            "capability_hash",
            "conditioning_binding_hash",
            "encoding_receipt_hash",
            "model_input_hash",
            "consumed_external_prefix_hash",
            "consumed_internal_prefix_hash",
            "consumed_mask_hash",
            "sampler_constraint",
            "continuation_implementation_id",
            "continuation_implementation_parameters_hash",
        ):
            object.__setattr__(
                self,
                name,
                _nonempty(getattr(self, name), name=name),
            )
        indices = tuple(
            _nonnegative_int(index, name="consumed generation index")
            for index in self.consumed_generation_indices
        )
        if any(
            right <= left for left, right in zip(indices, indices[1:])
        ):
            raise ValueError(
                "consumed_generation_indices must be strictly increasing."
            )
        object.__setattr__(self, "consumed_generation_indices", indices)
        delay = _nonnegative_int(
            self.effective_delay_steps,
            name="effective_delay_steps",
        )
        horizon = _nonnegative_int(
            self.prediction_horizon_steps,
            name="prediction_horizon_steps",
        )
        if horizon <= 1 or delay >= horizon:
            raise ValueError(
                "Consumption delay must leave a non-empty output suffix."
            )
        object.__setattr__(self, "effective_delay_steps", delay)
        object.__setattr__(self, "prediction_horizon_steps", horizon)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "schema": self.schema,
                "envelope_hash": self.envelope_hash,
                "request_hash": self.request_hash,
                "capability_hash": self.capability_hash,
                "conditioning_binding_hash": (
                    self.conditioning_binding_hash
                ),
                "encoding_receipt_hash": self.encoding_receipt_hash,
                "model_input_hash": self.model_input_hash,
                "consumed_external_prefix_hash": (
                    self.consumed_external_prefix_hash
                ),
                "consumed_internal_prefix_hash": (
                    self.consumed_internal_prefix_hash
                ),
                "consumed_mask_hash": self.consumed_mask_hash,
                "consumed_generation_indices": (
                    self.consumed_generation_indices
                ),
                "sampler_constraint": self.sampler_constraint,
                "continuation_implementation_id": (
                    self.continuation_implementation_id
                ),
                "continuation_implementation_parameters_hash": (
                    self.continuation_implementation_parameters_hash
                ),
                "effective_delay_steps": self.effective_delay_steps,
                "prediction_horizon_steps": self.prediction_horizon_steps,
            },
            domain=RTC_CONDITIONING_CONSUMPTION_RECEIPT_HASH_DOMAIN_V3,
        )


def build_rtc_conditioning_consumption_receipt_v3(
    envelope: RtcConditionedInferenceEnvelopeV3,
    *,
    consumed_external_prefix: np.ndarray,
    consumed_internal_prefix: np.ndarray,
    consumed_mask: np.ndarray,
) -> RtcConditioningConsumptionReceiptV3:
    """Bind arrays consumed by the exact sampler invocation."""

    if not isinstance(envelope, RtcConditionedInferenceEnvelopeV3):
        raise TypeError(
            "envelope must be an RtcConditionedInferenceEnvelopeV3."
        )
    request = envelope.request
    capability = envelope.capability
    encoding = envelope.encoded_conditioning.receipt
    external = np.asarray(consumed_external_prefix, dtype=np.float32)
    internal = np.asarray(consumed_internal_prefix, dtype=np.float32)
    mask = np.asarray(consumed_mask)
    expected_external_shape = (
        request.prediction_horizon_steps,
        capability.external_action_dim,
    )
    expected_internal_shape = (
        request.prediction_horizon_steps,
        capability.internal_sampler_action_dim,
    )
    if external.shape != expected_external_shape:
        raise ValueError(
            "consumed_external_prefix does not match the declared horizon "
            "and external action dimension."
        )
    if internal.shape != expected_internal_shape:
        raise ValueError(
            "consumed_internal_prefix does not match the declared horizon "
            "and internal sampler action dimension."
        )
    if mask.shape != (request.prediction_horizon_steps,):
        raise ValueError("consumed_mask must cover the prediction horizon.")
    external_hash = rtc_float32_array_hash(external)
    internal_hash = rtc_float32_array_hash(internal)
    mask_hash = rtc_binary_mask_hash(mask)
    consumed_indices = tuple(
        int(index) for index in np.flatnonzero(np.asarray(mask) == 1)
    )
    return RtcConditioningConsumptionReceiptV3(
        envelope_hash=envelope.semantic_content_hash,
        request_hash=request.semantic_content_hash,
        capability_hash=capability.semantic_content_hash,
        conditioning_binding_hash=encoding.conditioning_binding_hash,
        encoding_receipt_hash=encoding.semantic_content_hash,
        model_input_hash=envelope.model_input_hash,
        consumed_external_prefix_hash=external_hash,
        consumed_internal_prefix_hash=internal_hash,
        consumed_mask_hash=mask_hash,
        consumed_generation_indices=consumed_indices,
        sampler_constraint=capability.sampler_constraint,
        continuation_implementation_id=(
            capability.continuation_implementation_id
        ),
        continuation_implementation_parameters_hash=(
            capability.continuation_implementation_parameters_hash
        ),
        effective_delay_steps=request.forecast_delay_steps,
        prediction_horizon_steps=request.prediction_horizon_steps,
    )


@dataclass(frozen=True)
class RtcConditionedModelResponseV3:
    envelope_hash: str
    request_hash: str
    capability_hash: str
    model_output_id: str
    model_space_actions: np.ndarray
    inference_started_at: float
    inference_completed_at: float
    result_received_at: float
    consumption_receipt: RtcConditioningConsumptionReceiptV3
    protocol_version: str = RTC_CONDITIONED_MODEL_RESPONSE_SCHEMA_V3

    def __post_init__(self) -> None:
        for name in (
            "envelope_hash",
            "request_hash",
            "capability_hash",
            "model_output_id",
        ):
            object.__setattr__(
                self,
                name,
                _nonempty(getattr(self, name), name=name),
            )
        if self.protocol_version != RTC_CONDITIONED_MODEL_RESPONSE_SCHEMA_V3:
            raise ValueError(
                f"Unsupported conditioned response {self.protocol_version!r}."
            )
        actions = np.asarray(self.model_space_actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[0] < 2 or actions.shape[1] < 1:
            raise ValueError(
                "model_space_actions must be [prediction_horizon, action_dim]."
            )
        if not np.all(np.isfinite(actions)):
            raise ValueError("model_space_actions must contain finite values.")
        actions = np.ascontiguousarray(actions).copy()
        actions.setflags(write=False)
        object.__setattr__(self, "model_space_actions", actions)
        started = _finite(
            self.inference_started_at,
            name="inference_started_at",
        )
        completed = _finite(
            self.inference_completed_at,
            name="inference_completed_at",
        )
        received = _finite(
            self.result_received_at,
            name="result_received_at",
        )
        if started > completed or completed > received:
            raise ValueError("Conditioned response timestamps are not ordered.")
        object.__setattr__(self, "inference_started_at", started)
        object.__setattr__(self, "inference_completed_at", completed)
        object.__setattr__(self, "result_received_at", received)
        if not isinstance(
            self.consumption_receipt,
            RtcConditioningConsumptionReceiptV3,
        ):
            raise TypeError(
                "consumption_receipt must be an "
                "RtcConditioningConsumptionReceiptV3."
            )

    @property
    def model_output_hash(self) -> str:
        return canonical_content_hash(
            self.model_space_actions,
            domain=RTC_MODEL_OUTPUT_ROWS_HASH_DOMAIN_V3,
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "protocol_version": self.protocol_version,
                "envelope_hash": self.envelope_hash,
                "request_hash": self.request_hash,
                "capability_hash": self.capability_hash,
                "model_output_id": self.model_output_id,
                "model_output_hash": self.model_output_hash,
                "inference_started_at": self.inference_started_at,
                "inference_completed_at": self.inference_completed_at,
                "result_received_at": self.result_received_at,
                "consumption_receipt_hash": (
                    self.consumption_receipt.semantic_content_hash
                ),
            },
            domain=RTC_CONDITIONED_MODEL_RESPONSE_HASH_DOMAIN_V3,
        )


@dataclass(frozen=True)
class RtcConditionedResponseValidationV3:
    envelope_hash: str
    response_hash: str
    accepted: bool
    reasons: tuple[str, ...]
    schema: str = RTC_CONDITIONED_RESPONSE_VALIDATION_SCHEMA_V3

    def __post_init__(self) -> None:
        if self.schema != RTC_CONDITIONED_RESPONSE_VALIDATION_SCHEMA_V3:
            raise ValueError(
                f"Unsupported conditioned response validation {self.schema!r}."
            )
        for name in ("envelope_hash", "response_hash"):
            object.__setattr__(
                self,
                name,
                _nonempty(getattr(self, name), name=name),
            )
        reasons = tuple(sorted({_nonempty(reason, name="reason") for reason in self.reasons}))
        if bool(self.accepted) == bool(reasons):
            raise ValueError(
                "Accepted validation has no reasons; rejected validation has reasons."
            )
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(self, "reasons", reasons)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "schema": self.schema,
                "envelope_hash": self.envelope_hash,
                "response_hash": self.response_hash,
                "accepted": self.accepted,
                "reasons": self.reasons,
            },
            domain=RTC_CONDITIONED_RESPONSE_VALIDATION_HASH_DOMAIN_V3,
        )


def validate_rtc_conditioned_response_v3(
    *,
    envelope: RtcConditionedInferenceEnvelopeV3,
    response: RtcConditionedModelResponseV3,
) -> RtcConditionedResponseValidationV3:
    """Validate actual-consumption identity before any output decoding."""

    receipt = response.consumption_receipt
    request = envelope.request
    capability = envelope.capability
    encoding = envelope.encoded_conditioning.receipt
    model_output_shape_matches = (
        response.model_space_actions.shape
        == (
            request.prediction_horizon_steps,
            capability.external_action_dim,
        )
    )
    selected_indices = encoding.selected_generation_indices
    clamped_output_prefix_matches = True
    if (
        capability.sampler_constraint
        == MODEL_CONTINUATION_SAMPLER_CLAMPED_V1
        and selected_indices
        and model_output_shape_matches
    ):
        selected = np.asarray(selected_indices, dtype=np.intp)
        clamped_output_prefix_matches = np.array_equal(
            response.model_space_actions[selected],
            envelope.encoded_conditioning.external_prefix[selected],
        )
    checks = (
        (response.envelope_hash == envelope.semantic_content_hash, "envelope_hash_mismatch"),
        (response.request_hash == request.semantic_content_hash, "request_hash_mismatch"),
        (response.capability_hash == capability.semantic_content_hash, "capability_hash_mismatch"),
        (receipt.envelope_hash == envelope.semantic_content_hash, "receipt_envelope_hash_mismatch"),
        (receipt.request_hash == request.semantic_content_hash, "receipt_request_hash_mismatch"),
        (receipt.capability_hash == capability.semantic_content_hash, "receipt_capability_hash_mismatch"),
        (receipt.conditioning_binding_hash == encoding.conditioning_binding_hash, "conditioning_binding_hash_mismatch"),
        (receipt.encoding_receipt_hash == encoding.semantic_content_hash, "encoding_receipt_hash_mismatch"),
        (receipt.model_input_hash == envelope.model_input_hash, "model_input_hash_mismatch"),
        (
            receipt.consumed_external_prefix_hash
            == encoding.external_prefix_hash,
            "consumed_external_prefix_hash_mismatch",
        ),
        (
            receipt.consumed_mask_hash == encoding.prefix_mask_hash,
            "consumed_mask_hash_mismatch",
        ),
        (
            capability.conditioning_transform_owner
            != RTC_CONDITIONING_TRANSFORM_CLIENT_V1
            or receipt.consumed_internal_prefix_hash
            == receipt.consumed_external_prefix_hash,
            "client_transform_internal_prefix_hash_mismatch",
        ),
        (receipt.consumed_generation_indices == encoding.selected_generation_indices, "consumed_generation_indices_mismatch"),
        (receipt.sampler_constraint == capability.sampler_constraint, "sampler_constraint_mismatch"),
        (receipt.continuation_implementation_id == capability.continuation_implementation_id, "continuation_implementation_id_mismatch"),
        (
            receipt.continuation_implementation_parameters_hash
            == capability.continuation_implementation_parameters_hash,
            "continuation_implementation_parameters_hash_mismatch",
        ),
        (receipt.effective_delay_steps == request.forecast_delay_steps, "effective_delay_steps_mismatch"),
        (receipt.prediction_horizon_steps == request.prediction_horizon_steps, "prediction_horizon_steps_mismatch"),
        (response.model_space_actions.shape[0] == request.prediction_horizon_steps, "model_output_horizon_mismatch"),
        (
            response.model_space_actions.shape[1]
            == capability.external_action_dim,
            "model_output_dimension_mismatch",
        ),
        (
            clamped_output_prefix_matches,
            "clamped_output_prefix_mismatch",
        ),
    )
    reasons = tuple(reason for accepted, reason in checks if not accepted)
    return RtcConditionedResponseValidationV3(
        envelope_hash=envelope.semantic_content_hash,
        response_hash=response.semantic_content_hash,
        accepted=not reasons,
        reasons=reasons,
    )


@runtime_checkable
class RtcConditionedModelBackendV3(Protocol):
    @property
    def rtc_continuation_capability(
        self,
    ) -> RtcModelContinuationCapabilityV3:
        ...

    def infer_conditioned_rtc_v3(
        self,
        envelope: RtcConditionedInferenceEnvelopeV3,
        model_input: Any,
    ) -> RtcConditionedModelResponseV3:
        ...
