"""Model-, transport-, and backend-neutral RTC protocol contracts.

The objects in this module describe what an RTC-capable producer promises and
what one inference exchange actually consumed.  They deliberately do not own
model invocation, threads, transport, simulation, or hardware.  Existing v1
``PolicyInferenceRequest`` objects remain the causal request facts; v2 wraps
them with independently negotiable scheduling and continuation capabilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.runtime.immutable import deep_freeze

from .inference import (
    ASYNC_ALIGNED_CHUNK_MODE_V1,
    RTC_FLOW_INPAINT_MODE_V1,
    SYNC_QUEUE_MODE_V1,
    PolicyInferenceRequest,
)


RTC_PROTOCOL_V2 = "dexteleop_rtc_protocol_v2"

RTC_SCHEDULER_SYNC_QUEUE = "sync_queue"
RTC_SCHEDULER_ASYNC_TIMED_PLAN = "async_timed_plan"
RTC_SCHEDULER_MODES_V2 = (
    RTC_SCHEDULER_SYNC_QUEUE,
    RTC_SCHEDULER_ASYNC_TIMED_PLAN,
)

RTC_CONTINUATION_NONE = "none"
RTC_CONTINUATION_FLOW_INPAINT = "flow_inpaint"
RTC_CONTINUATION_TRAINED_PREFIX = "trained_prefix_conditioning"
RTC_CONTINUATION_NATIVE = "native_continuation"
RTC_CONTINUATION_MODES_V2 = (
    RTC_CONTINUATION_NONE,
    RTC_CONTINUATION_FLOW_INPAINT,
    RTC_CONTINUATION_TRAINED_PREFIX,
    RTC_CONTINUATION_NATIVE,
)

RTC_PREFIX_FORBIDDEN = "forbidden"
RTC_PREFIX_OPTIONAL = "optional"
RTC_PREFIX_REQUIRED = "required"
RTC_PREFIX_REQUIREMENTS_V2 = (
    RTC_PREFIX_FORBIDDEN,
    RTC_PREFIX_OPTIONAL,
    RTC_PREFIX_REQUIRED,
)

RTC_PROTOCOL_MODE_HASH_DOMAIN_V2 = "dexteleop.rtc_protocol_mode.v2"
RTC_CONTINUATION_CAPABILITY_HASH_DOMAIN_V2 = (
    "dexteleop.rtc_continuation_capability.v2"
)
RTC_MODEL_CAPABILITY_HASH_DOMAIN_V2 = "dexteleop.rtc_model_capability.v2"
RTC_PROTOCOL_REQUEST_SPEC_HASH_DOMAIN_V2 = "dexteleop.rtc_protocol_request_spec.v2"
RTC_NEGOTIATION_RECEIPT_HASH_DOMAIN_V2 = "dexteleop.rtc_negotiation_receipt.v2"
RTC_INFERENCE_ENVELOPE_HASH_DOMAIN_V2 = "dexteleop.rtc_inference_envelope.v2"
RTC_CONDITIONING_RECEIPT_HASH_DOMAIN_V2 = "dexteleop.rtc_conditioning_receipt.v2"
RTC_MODEL_RESPONSE_HASH_DOMAIN_V2 = "dexteleop.rtc_model_response.v2"
RTC_RESPONSE_VALIDATION_HASH_DOMAIN_V2 = "dexteleop.rtc_response_validation.v2"
RTC_PREFIX_MASK_HASH_DOMAIN_V2 = "dexteleop.rtc_prefix_mask.v2"


def _nonempty(value: str, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _finite(value: float, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


def _nonnegative_int(value: int, *, name: str) -> int:
    normalized = int(value)
    if normalized < 0 or normalized != value:
        raise ValueError(f"{name} must be a non-negative integer.")
    return normalized


def _positive_int(value: int, *, name: str) -> int:
    normalized = int(value)
    if normalized <= 0 or normalized != value:
        raise ValueError(f"{name} must be a positive integer.")
    return normalized


def _normalized_unique_strings(values: tuple[str, ...], *, name: str) -> tuple[str, ...]:
    normalized = tuple(_nonempty(value, name=name) for value in values)
    if not normalized:
        raise ValueError(f"{name} must not be empty.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates.")
    return normalized


@dataclass(frozen=True)
class RtcProtocolMode:
    """Independent scheduling and continuation axes for RTC v2."""

    scheduler_mode: str
    continuation_mode: str

    def __post_init__(self) -> None:
        scheduler = _nonempty(self.scheduler_mode, name="scheduler_mode")
        continuation = _nonempty(self.continuation_mode, name="continuation_mode")
        if scheduler not in RTC_SCHEDULER_MODES_V2:
            raise ValueError(f"Unsupported RTC scheduler mode: {scheduler!r}.")
        if continuation not in RTC_CONTINUATION_MODES_V2:
            raise ValueError(f"Unsupported RTC continuation mode: {continuation!r}.")
        if scheduler == RTC_SCHEDULER_SYNC_QUEUE and continuation != RTC_CONTINUATION_NONE:
            raise ValueError("Synchronous queue scheduling cannot request continuation.")
        object.__setattr__(self, "scheduler_mode", scheduler)
        object.__setattr__(self, "continuation_mode", continuation)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "scheduler_mode": self.scheduler_mode,
                "continuation_mode": self.continuation_mode,
            },
            domain=RTC_PROTOCOL_MODE_HASH_DOMAIN_V2,
        )


_V1_TO_V2_MODE = {
    SYNC_QUEUE_MODE_V1: RtcProtocolMode(
        RTC_SCHEDULER_SYNC_QUEUE,
        RTC_CONTINUATION_NONE,
    ),
    ASYNC_ALIGNED_CHUNK_MODE_V1: RtcProtocolMode(
        RTC_SCHEDULER_ASYNC_TIMED_PLAN,
        RTC_CONTINUATION_NONE,
    ),
    RTC_FLOW_INPAINT_MODE_V1: RtcProtocolMode(
        RTC_SCHEDULER_ASYNC_TIMED_PLAN,
        RTC_CONTINUATION_FLOW_INPAINT,
    ),
}
_V2_TO_V1_MODE = {
    (mode.scheduler_mode, mode.continuation_mode): v1_mode
    for v1_mode, mode in _V1_TO_V2_MODE.items()
}


def rtc_v2_mode_from_v1(v1_mode: str) -> RtcProtocolMode:
    """Map the combined v1 mode to its explicit v2 axes."""

    try:
        return _V1_TO_V2_MODE[str(v1_mode)]
    except KeyError as exc:
        raise ValueError(f"Unsupported RTC v1 mode: {v1_mode!r}.") from exc


def rtc_v1_mode_from_v2(mode: RtcProtocolMode) -> str:
    """Return the v1 representation, rejecting v2-only continuation modes."""

    key = (mode.scheduler_mode, mode.continuation_mode)
    try:
        return _V2_TO_V1_MODE[key]
    except KeyError as exc:
        raise ValueError(
            "RTC v2 mode has no lossless PolicyInferenceRequest v1 mapping: "
            f"{key!r}."
        ) from exc


@dataclass(frozen=True)
class RtcContinuationCapability:
    """One continuation implementation with exact representation bindings."""

    continuation_mode: str
    implementation_id: str
    implementation_parameters_hash: str
    model_action_space: str
    model_action_dim: int
    model_action_discrete_dim: int
    supported_action_pose_modes: tuple[str, ...]
    component_scope: tuple[str, ...]
    action_codec_hash: str
    frame_contract_hash: str
    layout_hash: str
    normalization_hash: str
    minimum_prediction_horizon_steps: int
    maximum_prediction_horizon_steps: int
    minimum_execution_horizon_steps: int
    maximum_execution_horizon_steps: int
    prefix_requirement: str = RTC_PREFIX_FORBIDDEN
    supports_prefix_mask: bool = False
    supports_conditioning_receipt: bool = False

    def __post_init__(self) -> None:
        continuation = _nonempty(self.continuation_mode, name="continuation_mode")
        if continuation not in RTC_CONTINUATION_MODES_V2:
            raise ValueError(f"Unsupported continuation mode: {continuation!r}.")
        object.__setattr__(self, "continuation_mode", continuation)
        for name in (
            "implementation_id",
            "implementation_parameters_hash",
            "model_action_space",
            "action_codec_hash",
            "frame_contract_hash",
            "layout_hash",
            "normalization_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "model_action_dim",
            _positive_int(self.model_action_dim, name="model_action_dim"),
        )
        object.__setattr__(
            self,
            "model_action_discrete_dim",
            _nonnegative_int(
                self.model_action_discrete_dim,
                name="model_action_discrete_dim",
            ),
        )
        object.__setattr__(
            self,
            "supported_action_pose_modes",
            _normalized_unique_strings(
                self.supported_action_pose_modes,
                name="supported_action_pose_modes",
            ),
        )
        object.__setattr__(
            self,
            "component_scope",
            _normalized_unique_strings(self.component_scope, name="component_scope"),
        )
        minimum_prediction = _positive_int(
            self.minimum_prediction_horizon_steps,
            name="minimum_prediction_horizon_steps",
        )
        maximum_prediction = _positive_int(
            self.maximum_prediction_horizon_steps,
            name="maximum_prediction_horizon_steps",
        )
        minimum_execution = _positive_int(
            self.minimum_execution_horizon_steps,
            name="minimum_execution_horizon_steps",
        )
        maximum_execution = _positive_int(
            self.maximum_execution_horizon_steps,
            name="maximum_execution_horizon_steps",
        )
        if maximum_prediction < minimum_prediction:
            raise ValueError("Maximum prediction horizon is below its minimum.")
        if maximum_execution < minimum_execution:
            raise ValueError("Maximum execution horizon is below its minimum.")
        if maximum_execution > maximum_prediction:
            raise ValueError("Execution horizon cannot exceed prediction horizon.")
        object.__setattr__(self, "minimum_prediction_horizon_steps", minimum_prediction)
        object.__setattr__(self, "maximum_prediction_horizon_steps", maximum_prediction)
        object.__setattr__(self, "minimum_execution_horizon_steps", minimum_execution)
        object.__setattr__(self, "maximum_execution_horizon_steps", maximum_execution)
        requirement = _nonempty(self.prefix_requirement, name="prefix_requirement")
        if requirement not in RTC_PREFIX_REQUIREMENTS_V2:
            raise ValueError(f"Unsupported prefix requirement: {requirement!r}.")
        if continuation == RTC_CONTINUATION_NONE and requirement != RTC_PREFIX_FORBIDDEN:
            raise ValueError("Continuation 'none' must forbid a conditioning prefix.")
        if requirement == RTC_PREFIX_FORBIDDEN and self.supports_prefix_mask:
            raise ValueError("A prefix-forbidden capability cannot support a prefix mask.")
        if continuation != RTC_CONTINUATION_NONE and not self.supports_conditioning_receipt:
            raise ValueError("Every continuation implementation must produce a receipt.")
        object.__setattr__(self, "prefix_requirement", requirement)
        object.__setattr__(self, "supports_prefix_mask", bool(self.supports_prefix_mask))
        object.__setattr__(
            self,
            "supports_conditioning_receipt",
            bool(self.supports_conditioning_receipt),
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "continuation_mode": self.continuation_mode,
                "implementation_id": self.implementation_id,
                "implementation_parameters_hash": self.implementation_parameters_hash,
                "model_action_space": self.model_action_space,
                "model_action_dim": self.model_action_dim,
                "model_action_discrete_dim": self.model_action_discrete_dim,
                "supported_action_pose_modes": self.supported_action_pose_modes,
                "component_scope": self.component_scope,
                "action_codec_hash": self.action_codec_hash,
                "frame_contract_hash": self.frame_contract_hash,
                "layout_hash": self.layout_hash,
                "normalization_hash": self.normalization_hash,
                "minimum_prediction_horizon_steps": self.minimum_prediction_horizon_steps,
                "maximum_prediction_horizon_steps": self.maximum_prediction_horizon_steps,
                "minimum_execution_horizon_steps": self.minimum_execution_horizon_steps,
                "maximum_execution_horizon_steps": self.maximum_execution_horizon_steps,
                "prefix_requirement": self.prefix_requirement,
                "supports_prefix_mask": self.supports_prefix_mask,
                "supports_conditioning_receipt": self.supports_conditioning_receipt,
            },
            domain=RTC_CONTINUATION_CAPABILITY_HASH_DOMAIN_V2,
        )


@dataclass(frozen=True)
class RtcModelCapability:
    """Versioned capability advertised by any local or remote backend."""

    capability_id: str
    provider_id: str
    artifact_hash: str
    scheduler_modes: tuple[str, ...]
    continuations: tuple[RtcContinuationCapability, ...]
    protocol_version: str = RTC_PROTOCOL_V2

    def __post_init__(self) -> None:
        for name in ("capability_id", "provider_id", "artifact_hash", "protocol_version"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name=name))
        schedulers = _normalized_unique_strings(
            self.scheduler_modes,
            name="scheduler_modes",
        )
        unsupported = set(schedulers) - set(RTC_SCHEDULER_MODES_V2)
        if unsupported:
            raise ValueError(f"Unsupported scheduler modes: {sorted(unsupported)!r}.")
        continuations = tuple(self.continuations)
        if not continuations:
            raise ValueError("At least one continuation capability is required.")
        modes = tuple(value.continuation_mode for value in continuations)
        if len(set(modes)) != len(modes):
            raise ValueError("Continuation modes must be unique within a capability.")
        if any(mode != RTC_CONTINUATION_NONE for mode in modes):
            if RTC_SCHEDULER_ASYNC_TIMED_PLAN not in schedulers:
                raise ValueError("Continuation support requires async timed-plan scheduling.")
        object.__setattr__(self, "scheduler_modes", schedulers)
        object.__setattr__(self, "continuations", continuations)

    def continuation(self, continuation_mode: str) -> RtcContinuationCapability | None:
        return next(
            (
                value
                for value in self.continuations
                if value.continuation_mode == continuation_mode
            ),
            None,
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "capability_id": self.capability_id,
                "provider_id": self.provider_id,
                "artifact_hash": self.artifact_hash,
                "scheduler_modes": self.scheduler_modes,
                "continuation_hashes": tuple(
                    value.semantic_content_hash for value in self.continuations
                ),
                "protocol_version": self.protocol_version,
            },
            domain=RTC_MODEL_CAPABILITY_HASH_DOMAIN_V2,
        )


@dataclass(frozen=True)
class RtcProtocolRequestSpec:
    """Backend-neutral requirements for one RTC client binding."""

    mode: RtcProtocolMode
    artifact_hash: str
    model_action_space: str
    model_action_dim: int
    model_action_discrete_dim: int
    action_pose_mode: str
    component_scope: tuple[str, ...]
    action_codec_hash: str
    frame_contract_hash: str
    layout_hash: str
    normalization_hash: str
    prediction_horizon_steps: int
    minimum_execution_horizon_steps: int
    require_prefix_mask: bool = False
    require_conditioning_receipt: bool = False
    protocol_version: str = RTC_PROTOCOL_V2

    def __post_init__(self) -> None:
        if not isinstance(self.mode, RtcProtocolMode):
            raise TypeError("mode must be an RtcProtocolMode.")
        for name in (
            "artifact_hash",
            "model_action_space",
            "action_pose_mode",
            "action_codec_hash",
            "frame_contract_hash",
            "layout_hash",
            "normalization_hash",
            "protocol_version",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "model_action_dim",
            _positive_int(self.model_action_dim, name="model_action_dim"),
        )
        object.__setattr__(
            self,
            "model_action_discrete_dim",
            _nonnegative_int(
                self.model_action_discrete_dim,
                name="model_action_discrete_dim",
            ),
        )
        object.__setattr__(
            self,
            "component_scope",
            _normalized_unique_strings(self.component_scope, name="component_scope"),
        )
        prediction = _positive_int(
            self.prediction_horizon_steps,
            name="prediction_horizon_steps",
        )
        minimum_execution = _positive_int(
            self.minimum_execution_horizon_steps,
            name="minimum_execution_horizon_steps",
        )
        if minimum_execution > prediction:
            raise ValueError("Minimum execution horizon exceeds prediction horizon.")
        object.__setattr__(self, "prediction_horizon_steps", prediction)
        object.__setattr__(self, "minimum_execution_horizon_steps", minimum_execution)
        object.__setattr__(self, "require_prefix_mask", bool(self.require_prefix_mask))
        object.__setattr__(
            self,
            "require_conditioning_receipt",
            bool(self.require_conditioning_receipt),
        )
        if self.mode.continuation_mode == RTC_CONTINUATION_NONE:
            if self.require_prefix_mask or self.require_conditioning_receipt:
                raise ValueError("Continuation 'none' cannot require prefix semantics.")
        elif not self.require_conditioning_receipt:
            raise ValueError("Every continuation request must require a conditioning receipt.")

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "mode_hash": self.mode.semantic_content_hash,
                "artifact_hash": self.artifact_hash,
                "model_action_space": self.model_action_space,
                "model_action_dim": self.model_action_dim,
                "model_action_discrete_dim": self.model_action_discrete_dim,
                "action_pose_mode": self.action_pose_mode,
                "component_scope": self.component_scope,
                "action_codec_hash": self.action_codec_hash,
                "frame_contract_hash": self.frame_contract_hash,
                "layout_hash": self.layout_hash,
                "normalization_hash": self.normalization_hash,
                "prediction_horizon_steps": self.prediction_horizon_steps,
                "minimum_execution_horizon_steps": self.minimum_execution_horizon_steps,
                "require_prefix_mask": self.require_prefix_mask,
                "require_conditioning_receipt": self.require_conditioning_receipt,
                "protocol_version": self.protocol_version,
            },
            domain=RTC_PROTOCOL_REQUEST_SPEC_HASH_DOMAIN_V2,
        )


@dataclass(frozen=True)
class RtcCapabilityNegotiationReceipt:
    accepted: bool
    request_spec_hash: str
    capability_hash: str
    selected_continuation_hash: str | None
    mapped_v1_mode: str | None
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "accepted", bool(self.accepted))
        for name in ("request_spec_hash", "capability_hash"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name=name))
        for name in ("selected_continuation_hash", "mapped_v1_mode"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _nonempty(value, name=name))
        reasons = tuple(_nonempty(value, name="rejection_reason") for value in self.rejection_reasons)
        if self.accepted and reasons:
            raise ValueError("An accepted negotiation cannot have rejection reasons.")
        if not self.accepted and not reasons:
            raise ValueError("A rejected negotiation requires rejection reasons.")
        if self.accepted and self.selected_continuation_hash is None:
            raise ValueError("An accepted negotiation requires a selected continuation.")
        object.__setattr__(self, "rejection_reasons", reasons)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "accepted": self.accepted,
                "request_spec_hash": self.request_spec_hash,
                "capability_hash": self.capability_hash,
                "selected_continuation_hash": self.selected_continuation_hash,
                "mapped_v1_mode": self.mapped_v1_mode,
                "rejection_reasons": self.rejection_reasons,
            },
            domain=RTC_NEGOTIATION_RECEIPT_HASH_DOMAIN_V2,
        )


def negotiate_rtc_capability(
    request: RtcProtocolRequestSpec,
    capability: RtcModelCapability,
) -> RtcCapabilityNegotiationReceipt:
    """Fail-closed comparison with stable, machine-testable rejection codes."""

    reasons: list[str] = []
    selected = capability.continuation(request.mode.continuation_mode)
    if request.protocol_version != RTC_PROTOCOL_V2:
        reasons.append("request_protocol_unsupported")
    if capability.protocol_version != request.protocol_version:
        reasons.append("protocol_mismatch")
    if capability.artifact_hash != request.artifact_hash:
        reasons.append("artifact_hash_mismatch")
    if request.mode.scheduler_mode not in capability.scheduler_modes:
        reasons.append("scheduler_mode_unsupported")
    if selected is None:
        reasons.append("continuation_mode_unsupported")
    else:
        if selected.model_action_space != request.model_action_space:
            reasons.append("model_action_space_mismatch")
        if selected.model_action_dim != request.model_action_dim:
            reasons.append("model_action_dim_mismatch")
        if selected.model_action_discrete_dim != request.model_action_discrete_dim:
            reasons.append("model_action_discrete_dim_mismatch")
        if request.action_pose_mode not in selected.supported_action_pose_modes:
            reasons.append("action_pose_mode_unsupported")
        if not set(request.component_scope).issubset(selected.component_scope):
            reasons.append("component_scope_unsupported")
        for name in (
            "action_codec_hash",
            "frame_contract_hash",
            "layout_hash",
            "normalization_hash",
        ):
            if getattr(selected, name) != getattr(request, name):
                reasons.append(f"{name}_mismatch")
        if not (
            selected.minimum_prediction_horizon_steps
            <= request.prediction_horizon_steps
            <= selected.maximum_prediction_horizon_steps
        ):
            reasons.append("prediction_horizon_unsupported")
        if not (
            selected.minimum_execution_horizon_steps
            <= request.minimum_execution_horizon_steps
            <= selected.maximum_execution_horizon_steps
        ):
            reasons.append("execution_horizon_unsupported")
        if request.require_prefix_mask and not selected.supports_prefix_mask:
            reasons.append("prefix_mask_unsupported")
        if (
            request.require_conditioning_receipt
            and not selected.supports_conditioning_receipt
        ):
            reasons.append("conditioning_receipt_unsupported")
        if (
            request.mode.continuation_mode != RTC_CONTINUATION_NONE
            and selected.prefix_requirement == RTC_PREFIX_FORBIDDEN
        ):
            reasons.append("conditioning_prefix_unsupported")
        if (
            request.mode.continuation_mode != RTC_CONTINUATION_NONE
            and request.model_action_discrete_dim > 0
        ):
            reasons.append("discrete_conditioning_prefix_unsupported_v2")
    try:
        mapped_v1_mode = rtc_v1_mode_from_v2(request.mode)
    except ValueError:
        mapped_v1_mode = None
    return RtcCapabilityNegotiationReceipt(
        accepted=not reasons,
        request_spec_hash=request.semantic_content_hash,
        capability_hash=capability.semantic_content_hash,
        selected_continuation_hash=(
            selected.semantic_content_hash if selected is not None and not reasons else None
        ),
        mapped_v1_mode=mapped_v1_mode,
        rejection_reasons=tuple(reasons),
    )


def _freeze_prefix_mask(
    value: np.ndarray | None,
    *,
    request: PolicyInferenceRequest,
    required: bool,
) -> np.ndarray | None:
    prefix = request.conditioning_prefix
    if value is None:
        if required:
            raise ValueError("This RTC binding requires a prefix mask.")
        return None
    if prefix is None:
        raise ValueError("A prefix mask cannot exist without a conditioning prefix.")
    mask = np.asarray(value, dtype=np.float32).copy()
    if mask.ndim != 1 or mask.shape[0] != prefix.model_space_actions.shape[0]:
        raise ValueError("Prefix mask shape must match the prefix step dimension.")
    if np.any(~np.isfinite(mask)) or np.any(mask < 0.0) or np.any(mask > 1.0):
        raise ValueError("Prefix mask values must be finite and in [0, 1].")
    if np.any(mask[prefix.valid_steps :] != 0.0):
        raise ValueError("Prefix mask cannot select steps beyond prefix.valid_steps.")
    if required and not np.any(mask[: prefix.valid_steps] > 0.0):
        raise ValueError("A required prefix mask must select at least one valid step.")
    return deep_freeze(mask)


def rtc_prefix_mask_hash(prefix_mask: np.ndarray | None) -> str | None:
    if prefix_mask is None:
        return None
    return canonical_content_hash(
        {"prefix_mask": np.asarray(prefix_mask, dtype=np.float32)},
        domain=RTC_PREFIX_MASK_HASH_DOMAIN_V2,
    )


def _freeze_discrete_action_matrix(
    value: np.ndarray | None,
    *,
    horizon: int,
) -> np.ndarray | None:
    if value is None:
        return None
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[0] != int(horizon):
        raise ValueError(
            "model_space_discrete_actions must be rank-2 and share the float horizon."
        )
    if raw.shape[1] == 0:
        return None
    tokens = np.empty(raw.shape, dtype=np.int64)
    for index, item in np.ndenumerate(raw):
        if isinstance(item, (bool, np.bool_)):
            raise ValueError("Discrete model actions cannot contain booleans.")
        try:
            token = int(item)
            if float(item) != float(token):
                raise ValueError
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"model_space_discrete_actions{index} must be an exact integer token."
            ) from exc
        tokens[index] = token
    return deep_freeze(tokens)


@dataclass(frozen=True)
class RtcInferenceEnvelopeV2:
    """A v1 causal request plus negotiated, model-neutral v2 semantics."""

    request: PolicyInferenceRequest
    request_spec: RtcProtocolRequestSpec
    capability_hash: str
    selected_continuation_hash: str
    negotiation_receipt_hash: str
    model_input_hash: str
    prefix_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request, PolicyInferenceRequest):
            raise TypeError("request must be a PolicyInferenceRequest.")
        if not isinstance(self.request_spec, RtcProtocolRequestSpec):
            raise TypeError("request_spec must be an RtcProtocolRequestSpec.")
        for name in (
            "capability_hash",
            "selected_continuation_hash",
            "negotiation_receipt_hash",
            "model_input_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name=name))
        mapped = rtc_v1_mode_from_v2(self.request_spec.mode)
        if self.request.rtc_mode != mapped:
            raise ValueError("PolicyInferenceRequest v1 mode does not match RTC v2 axes.")
        for name in (
            "artifact_hash",
            "action_codec_hash",
            "frame_contract_hash",
            "layout_hash",
            "normalization_hash",
            "prediction_horizon_steps",
            "minimum_execution_horizon_steps",
        ):
            if getattr(self.request, name) != getattr(self.request_spec, name):
                raise ValueError(f"Inference request/spec mismatch: {name}.")
        prefix = self.request.conditioning_prefix
        continuation = self.request_spec.mode.continuation_mode
        if continuation == RTC_CONTINUATION_NONE and prefix is not None:
            raise ValueError("Continuation 'none' forbids a conditioning prefix.")
        if continuation != RTC_CONTINUATION_NONE and prefix is None:
            raise ValueError("Continuation mode requires a conditioning prefix.")
        if prefix is not None:
            if prefix.valid_steps <= 0:
                raise ValueError("A conditioning prefix requires at least one valid step.")
            if prefix.model_space_actions.shape[1] != self.request_spec.model_action_dim:
                raise ValueError("Conditioning prefix model action dimension mismatch.")
            if (
                prefix.model_space_actions.shape[0]
                > self.request_spec.prediction_horizon_steps
            ):
                raise ValueError("Conditioning prefix exceeds the prediction horizon.")
        mask = _freeze_prefix_mask(
            self.prefix_mask,
            request=self.request,
            required=self.request_spec.require_prefix_mask,
        )
        object.__setattr__(self, "prefix_mask", mask)

    @property
    def prefix_mask_hash(self) -> str | None:
        return rtc_prefix_mask_hash(self.prefix_mask)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "request_hash": self.request.semantic_content_hash,
                "request_spec_hash": self.request_spec.semantic_content_hash,
                "capability_hash": self.capability_hash,
                "selected_continuation_hash": self.selected_continuation_hash,
                "negotiation_receipt_hash": self.negotiation_receipt_hash,
                "model_input_hash": self.model_input_hash,
                "prefix_mask_hash": self.prefix_mask_hash,
            },
            domain=RTC_INFERENCE_ENVELOPE_HASH_DOMAIN_V2,
        )


def build_rtc_inference_envelope(
    *,
    request: PolicyInferenceRequest,
    request_spec: RtcProtocolRequestSpec,
    capability: RtcModelCapability,
    negotiation: RtcCapabilityNegotiationReceipt,
    model_input_hash: str,
    prefix_mask: np.ndarray | None = None,
) -> RtcInferenceEnvelopeV2:
    """Build only from the exact capability negotiation that was accepted."""

    if not negotiation.accepted:
        raise ValueError("Cannot build an RTC envelope from a rejected negotiation.")
    if negotiation.request_spec_hash != request_spec.semantic_content_hash:
        raise ValueError("RTC negotiation/request spec hash mismatch.")
    if negotiation.capability_hash != capability.semantic_content_hash:
        raise ValueError("RTC negotiation/capability hash mismatch.")
    if negotiation.selected_continuation_hash is None:
        raise ValueError("RTC negotiation has no selected continuation.")
    return RtcInferenceEnvelopeV2(
        request=request,
        request_spec=request_spec,
        capability_hash=capability.semantic_content_hash,
        selected_continuation_hash=negotiation.selected_continuation_hash,
        negotiation_receipt_hash=negotiation.semantic_content_hash,
        model_input_hash=model_input_hash,
        prefix_mask=prefix_mask,
    )


@dataclass(frozen=True)
class RtcConditioningReceiptV2:
    """Proof of the exact prefix/mask and implementation consumed by a model."""

    envelope_hash: str
    request_hash: str
    capability_hash: str
    selected_continuation_hash: str
    implementation_id: str
    artifact_hash: str
    conditioning_prefix_hash: str
    prefix_mask_hash: str | None
    effective_prefix_steps: int
    effective_delay_steps: int
    execution_horizon_steps: int
    implementation_parameters_hash: str

    def __post_init__(self) -> None:
        for name in (
            "envelope_hash",
            "request_hash",
            "capability_hash",
            "selected_continuation_hash",
            "implementation_id",
            "artifact_hash",
            "conditioning_prefix_hash",
            "implementation_parameters_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name=name))
        if self.prefix_mask_hash is not None:
            object.__setattr__(
                self,
                "prefix_mask_hash",
                _nonempty(self.prefix_mask_hash, name="prefix_mask_hash"),
            )
        object.__setattr__(
            self,
            "effective_prefix_steps",
            _positive_int(self.effective_prefix_steps, name="effective_prefix_steps"),
        )
        object.__setattr__(
            self,
            "effective_delay_steps",
            _nonnegative_int(self.effective_delay_steps, name="effective_delay_steps"),
        )
        object.__setattr__(
            self,
            "execution_horizon_steps",
            _positive_int(self.execution_horizon_steps, name="execution_horizon_steps"),
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "envelope_hash": self.envelope_hash,
                "request_hash": self.request_hash,
                "capability_hash": self.capability_hash,
                "selected_continuation_hash": self.selected_continuation_hash,
                "implementation_id": self.implementation_id,
                "artifact_hash": self.artifact_hash,
                "conditioning_prefix_hash": self.conditioning_prefix_hash,
                "prefix_mask_hash": self.prefix_mask_hash,
                "effective_prefix_steps": self.effective_prefix_steps,
                "effective_delay_steps": self.effective_delay_steps,
                "execution_horizon_steps": self.execution_horizon_steps,
                "implementation_parameters_hash": self.implementation_parameters_hash,
            },
            domain=RTC_CONDITIONING_RECEIPT_HASH_DOMAIN_V2,
        )


@dataclass(frozen=True)
class RtcModelResponseEnvelopeV2:
    """Untrusted model-space output with echoed protocol identities."""

    envelope_hash: str
    request_id: str
    request_hash: str
    model_input_hash: str
    capability_hash: str
    selected_continuation_hash: str
    protocol_version: str
    scheduler_mode: str
    continuation_mode: str
    artifact_hash: str
    action_codec_hash: str
    frame_contract_hash: str
    layout_hash: str
    normalization_hash: str
    model_output_id: str
    model_space_actions: np.ndarray
    effective_delay_steps: int
    execution_horizon_steps: int
    inference_started_at: float
    inference_completed_at: float
    result_received_at: float
    model_space_discrete_actions: np.ndarray | None = None
    conditioning_receipt: RtcConditioningReceiptV2 | None = None

    def __post_init__(self) -> None:
        for name in (
            "envelope_hash",
            "request_id",
            "request_hash",
            "model_input_hash",
            "capability_hash",
            "selected_continuation_hash",
            "protocol_version",
            "scheduler_mode",
            "continuation_mode",
            "artifact_hash",
            "action_codec_hash",
            "frame_contract_hash",
            "layout_hash",
            "normalization_hash",
            "model_output_id",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name=name))
        actions = np.asarray(self.model_space_actions, dtype=np.float32).copy()
        if actions.ndim != 2 or actions.shape[0] <= 0 or actions.shape[1] <= 0:
            raise ValueError("model_space_actions must be a non-empty rank-2 array.")
        if np.any(~np.isfinite(actions)):
            raise ValueError("model_space_actions must be finite.")
        object.__setattr__(self, "model_space_actions", deep_freeze(actions))
        object.__setattr__(
            self,
            "model_space_discrete_actions",
            _freeze_discrete_action_matrix(
                self.model_space_discrete_actions,
                horizon=actions.shape[0],
            ),
        )
        object.__setattr__(
            self,
            "effective_delay_steps",
            _nonnegative_int(self.effective_delay_steps, name="effective_delay_steps"),
        )
        execution = _positive_int(
            self.execution_horizon_steps,
            name="execution_horizon_steps",
        )
        if execution > actions.shape[0]:
            raise ValueError("Execution horizon exceeds returned model-space actions.")
        object.__setattr__(self, "execution_horizon_steps", execution)
        for name in (
            "inference_started_at",
            "inference_completed_at",
            "result_received_at",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name=name))
        if self.inference_completed_at < self.inference_started_at:
            raise ValueError("Inference completion predates its start.")
        if self.result_received_at < self.inference_completed_at:
            raise ValueError("Result receipt predates inference completion.")

    @property
    def conditioning_receipt_hash(self) -> str | None:
        if self.conditioning_receipt is None:
            return None
        return self.conditioning_receipt.semantic_content_hash

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "envelope_hash": self.envelope_hash,
                "request_id": self.request_id,
                "request_hash": self.request_hash,
                "model_input_hash": self.model_input_hash,
                "capability_hash": self.capability_hash,
                "selected_continuation_hash": self.selected_continuation_hash,
                "protocol_version": self.protocol_version,
                "scheduler_mode": self.scheduler_mode,
                "continuation_mode": self.continuation_mode,
                "artifact_hash": self.artifact_hash,
                "action_codec_hash": self.action_codec_hash,
                "frame_contract_hash": self.frame_contract_hash,
                "layout_hash": self.layout_hash,
                "normalization_hash": self.normalization_hash,
                "model_output_id": self.model_output_id,
                "model_space_actions": self.model_space_actions,
                "model_space_discrete_actions": self.model_space_discrete_actions,
                "effective_delay_steps": self.effective_delay_steps,
                "execution_horizon_steps": self.execution_horizon_steps,
                "inference_started_at": self.inference_started_at,
                "inference_completed_at": self.inference_completed_at,
                "result_received_at": self.result_received_at,
                "conditioning_receipt_hash": self.conditioning_receipt_hash,
            },
            domain=RTC_MODEL_RESPONSE_HASH_DOMAIN_V2,
        )


@dataclass(frozen=True)
class RtcResponseValidationReceipt:
    accepted: bool
    envelope_hash: str
    response_hash: str
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "accepted", bool(self.accepted))
        for name in ("envelope_hash", "response_hash"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name=name))
        reasons = tuple(_nonempty(value, name="rejection_reason") for value in self.rejection_reasons)
        if self.accepted and reasons:
            raise ValueError("An accepted response cannot have rejection reasons.")
        if not self.accepted and not reasons:
            raise ValueError("A rejected response requires rejection reasons.")
        object.__setattr__(self, "rejection_reasons", reasons)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "accepted": self.accepted,
                "envelope_hash": self.envelope_hash,
                "response_hash": self.response_hash,
                "rejection_reasons": self.rejection_reasons,
            },
            domain=RTC_RESPONSE_VALIDATION_HASH_DOMAIN_V2,
        )


def validate_rtc_model_response(
    *,
    envelope: RtcInferenceEnvelopeV2,
    response: RtcModelResponseEnvelopeV2,
    capability: RtcModelCapability,
) -> RtcResponseValidationReceipt:
    """Validate all echoed identity, representation, horizon, and prefix facts."""

    request = envelope.request
    spec = envelope.request_spec
    selected = capability.continuation(spec.mode.continuation_mode)
    reasons: list[str] = []
    expected = {
        "envelope_hash": envelope.semantic_content_hash,
        "request_id": request.request_id,
        "request_hash": request.semantic_content_hash,
        "model_input_hash": envelope.model_input_hash,
        "capability_hash": capability.semantic_content_hash,
        "selected_continuation_hash": envelope.selected_continuation_hash,
        "protocol_version": spec.protocol_version,
        "scheduler_mode": spec.mode.scheduler_mode,
        "continuation_mode": spec.mode.continuation_mode,
        "artifact_hash": spec.artifact_hash,
        "action_codec_hash": spec.action_codec_hash,
        "frame_contract_hash": spec.frame_contract_hash,
        "layout_hash": spec.layout_hash,
        "normalization_hash": spec.normalization_hash,
    }
    for name, value in expected.items():
        if getattr(response, name) != value:
            reasons.append(f"{name}_mismatch")
    if response.model_space_actions.shape[1] != spec.model_action_dim:
        reasons.append("model_action_dim_mismatch")
    discrete = response.model_space_discrete_actions
    actual_discrete_dim = 0 if discrete is None else int(discrete.shape[1])
    if actual_discrete_dim != spec.model_action_discrete_dim:
        reasons.append("model_action_discrete_dim_mismatch")
    if response.model_space_actions.shape[0] > spec.prediction_horizon_steps:
        reasons.append("prediction_horizon_exceeded")
    if response.execution_horizon_steps < spec.minimum_execution_horizon_steps:
        reasons.append("execution_horizon_too_short")
    if response.execution_horizon_steps > spec.prediction_horizon_steps:
        reasons.append("execution_horizon_exceeded")
    if response.effective_delay_steps != request.forecast_delay_steps:
        reasons.append("effective_delay_mismatch")
    receipt = response.conditioning_receipt
    if spec.mode.continuation_mode == RTC_CONTINUATION_NONE:
        if receipt is not None:
            reasons.append("unexpected_conditioning_receipt")
    elif receipt is None:
        reasons.append("missing_conditioning_receipt")
    else:
        prefix = request.conditioning_prefix
        receipt_expected = {
            "envelope_hash": envelope.semantic_content_hash,
            "request_hash": request.semantic_content_hash,
            "capability_hash": capability.semantic_content_hash,
            "selected_continuation_hash": envelope.selected_continuation_hash,
            "artifact_hash": spec.artifact_hash,
            "conditioning_prefix_hash": request.conditioning_prefix_hash,
            "prefix_mask_hash": envelope.prefix_mask_hash,
            "effective_delay_steps": response.effective_delay_steps,
            "execution_horizon_steps": response.execution_horizon_steps,
        }
        if selected is not None:
            receipt_expected["implementation_id"] = selected.implementation_id
            receipt_expected["implementation_parameters_hash"] = (
                selected.implementation_parameters_hash
            )
        for name, value in receipt_expected.items():
            if getattr(receipt, name) != value:
                reasons.append(f"conditioning_{name}_mismatch")
        if prefix is not None and receipt.effective_prefix_steps > prefix.valid_steps:
            reasons.append("conditioning_prefix_steps_exceeded")
    if spec.require_conditioning_receipt and receipt is None:
        if "missing_conditioning_receipt" not in reasons:
            reasons.append("missing_conditioning_receipt")
    return RtcResponseValidationReceipt(
        accepted=not reasons,
        envelope_hash=envelope.semantic_content_hash,
        response_hash=response.semantic_content_hash,
        rejection_reasons=tuple(reasons),
    )


__all__ = [
    "RTC_CONTINUATION_FLOW_INPAINT",
    "RTC_CONTINUATION_MODES_V2",
    "RTC_CONTINUATION_NATIVE",
    "RTC_CONTINUATION_NONE",
    "RTC_CONTINUATION_TRAINED_PREFIX",
    "RTC_PREFIX_FORBIDDEN",
    "RTC_PREFIX_OPTIONAL",
    "RTC_PREFIX_REQUIRED",
    "RTC_PROTOCOL_V2",
    "RTC_SCHEDULER_ASYNC_TIMED_PLAN",
    "RTC_SCHEDULER_MODES_V2",
    "RTC_SCHEDULER_SYNC_QUEUE",
    "RtcCapabilityNegotiationReceipt",
    "RtcConditioningReceiptV2",
    "RtcContinuationCapability",
    "RtcInferenceEnvelopeV2",
    "RtcModelCapability",
    "RtcModelResponseEnvelopeV2",
    "RtcProtocolMode",
    "RtcProtocolRequestSpec",
    "RtcResponseValidationReceipt",
    "build_rtc_inference_envelope",
    "negotiate_rtc_capability",
    "rtc_prefix_mask_hash",
    "rtc_v1_mode_from_v2",
    "rtc_v2_mode_from_v1",
    "validate_rtc_model_response",
]
