"""Pure policy-inference contracts for target-time action chunk planning.

This module intentionally contains no worker, transport, model, queue, or
hardware ownership.  It defines immutable facts that a synchronous caller or
an asynchronous worker boundary can exchange before a control-thread-owned
plan admission step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.runtime.immutable import deep_freeze

from .action_representation import PolicyActionRepresentationContext
from .actions import NormalizedAction
from .canonical_policy_action import (
    CanonicalPolicyAction,
    canonical_policy_robot_action_chunk_hash,
)


SYNC_QUEUE_MODE_V1 = "sync_queue_v1"
ASYNC_ALIGNED_CHUNK_MODE_V1 = "async_aligned_chunk_v1"
RTC_FLOW_INPAINT_MODE_V1 = "rtc_flow_inpaint_v1"
POLICY_INFERENCE_MODES_V1 = (
    SYNC_QUEUE_MODE_V1,
    ASYNC_ALIGNED_CHUNK_MODE_V1,
    RTC_FLOW_INPAINT_MODE_V1,
)

ROLLING_MAX_ESTIMATOR_V1 = "rolling_max_v1"
NEAREST_RANK_QUANTILE_ESTIMATOR_V1 = "nearest_rank_quantile_v1"
INFERENCE_DELAY_ESTIMATORS_V1 = (
    ROLLING_MAX_ESTIMATOR_V1,
    NEAREST_RANK_QUANTILE_ESTIMATOR_V1,
)

INFERENCE_DELAY_ESTIMATOR_SPEC_HASH_DOMAIN_V1 = (
    "dexteleop.inference_delay_estimator_spec.v1"
)
INFERENCE_DELAY_ESTIMATOR_STATE_HASH_DOMAIN_V1 = (
    "dexteleop.inference_delay_estimator_state.v1"
)
POLICY_INFERENCE_PREFIX_HASH_DOMAIN_V1 = "dexteleop.policy_inference_prefix.v1"
POLICY_INFERENCE_REQUEST_HASH_DOMAIN_V1 = "dexteleop.policy_inference_request.v1"
POLICY_INFERENCE_REQUEST_HASH_DOMAIN_V2 = "dexteleop.policy_inference_request.v2"
POLICY_CHUNK_RESULT_HASH_DOMAIN_V1 = "dexteleop.policy_chunk_result.v1"
POLICY_CHUNK_RESULT_HASH_DOMAIN_V2 = "dexteleop.policy_chunk_result.v2"

POLICY_ACTION_PAYLOAD_LEGACY_NORMALIZED_V1 = "legacy_normalized_absolute_v1"
POLICY_ACTION_PAYLOAD_TYPED_CANONICAL_V1 = "typed_canonical_policy_action_v1"

_DECODED_ACTION_SEMANTIC_METADATA_KEYS_V1 = (
    "arm_action_specs_by_side",
    "hand_action_specs_by_side",
    "hand_active_mask_by_side",
    "end_effector_enabled_sides",
    "pose_action_source_mode",
    "pose_action_mode",
    "pose_action_source_absolute_frame",
    "pose_action_output_absolute_frame",
    "pose_anchor_policy",
    "pose_anchor_sides",
)


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


def _normalized_action_semantic_payload(action: NormalizedAction) -> dict[str, Any]:
    """Project a decoded action without relying on arbitrary metadata values.

    Representation/frame/layout contracts are explicit request/result fields.
    The action payload therefore hashes only the canonical vector and optional
    joint targets; diagnostic metadata is detached but deliberately excluded.
    """

    return {
        "vector": np.asarray(action.vector, dtype=np.float32),
        "arm_joint_targets": {
            str(side): np.asarray(value, dtype=np.float32)
            for side, value in sorted(action.arm_joint_targets.items())
        },
        "semantic_metadata": {
            key: action.metadata[key]
            for key in _DECODED_ACTION_SEMANTIC_METADATA_KEYS_V1
            if key in action.metadata
        },
    }


def _freeze_normalized_action(action: NormalizedAction) -> NormalizedAction:
    vector = np.asarray(action.vector, dtype=np.float32).copy()
    if vector.ndim != 1:
        raise ValueError("NormalizedAction.vector must be one-dimensional.")
    if np.any(np.isinf(vector)):
        raise ValueError("NormalizedAction.vector must not contain infinity.")
    joints: dict[str, np.ndarray] = {}
    for side, value in action.arm_joint_targets.items():
        key = _nonempty(str(side), name="arm joint target side")
        array = np.asarray(value, dtype=np.float32).copy()
        if array.ndim != 1 or np.any(~np.isfinite(array)):
            raise ValueError("Arm joint targets must be finite one-dimensional arrays.")
        joints[key] = array
    frozen = NormalizedAction(
        vector=deep_freeze(vector),
        arm_joint_targets=joints,
        metadata=deep_freeze(dict(action.metadata)),
    )
    object.__setattr__(frozen, "arm_joint_targets", deep_freeze(joints))
    return frozen


@dataclass(frozen=True)
class InferenceDelayEstimatorSpec:
    """Versioned conversion from measured latency to forecast action steps."""

    spec_id: str
    action_period_sec: float
    window_size: int
    estimator_kind: str = ROLLING_MAX_ESTIMATOR_V1
    quantile: float = 0.99
    warmup_latency_sec: float = 0.0
    network_guard_sec: float = 0.0
    control_jitter_guard_steps: int = 0
    minimum_delay_steps: int = 0
    maximum_delay_steps: int = 64

    def __post_init__(self) -> None:
        object.__setattr__(self, "spec_id", _nonempty(self.spec_id, name="spec_id"))
        period = _finite(self.action_period_sec, name="action_period_sec")
        if period <= 0:
            raise ValueError("action_period_sec must be positive.")
        object.__setattr__(self, "action_period_sec", period)
        object.__setattr__(self, "window_size", _positive_int(self.window_size, name="window_size"))
        kind = _nonempty(self.estimator_kind, name="estimator_kind")
        if kind not in INFERENCE_DELAY_ESTIMATORS_V1:
            raise ValueError(f"Unsupported inference delay estimator: {kind!r}.")
        object.__setattr__(self, "estimator_kind", kind)
        quantile = _finite(self.quantile, name="quantile")
        if not 0 < quantile <= 1:
            raise ValueError("quantile must be in (0, 1].")
        object.__setattr__(self, "quantile", quantile)
        for name in ("warmup_latency_sec", "network_guard_sec"):
            value = _finite(getattr(self, name), name=name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")
            object.__setattr__(self, name, value)
        jitter = _nonnegative_int(
            self.control_jitter_guard_steps,
            name="control_jitter_guard_steps",
        )
        minimum = _nonnegative_int(self.minimum_delay_steps, name="minimum_delay_steps")
        maximum = _nonnegative_int(self.maximum_delay_steps, name="maximum_delay_steps")
        if maximum < minimum:
            raise ValueError("maximum_delay_steps must be >= minimum_delay_steps.")
        object.__setattr__(self, "control_jitter_guard_steps", jitter)
        object.__setattr__(self, "minimum_delay_steps", minimum)
        object.__setattr__(self, "maximum_delay_steps", maximum)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "spec_id": self.spec_id,
                "action_period_sec": self.action_period_sec,
                "window_size": self.window_size,
                "estimator_kind": self.estimator_kind,
                "quantile": self.quantile,
                "warmup_latency_sec": self.warmup_latency_sec,
                "network_guard_sec": self.network_guard_sec,
                "control_jitter_guard_steps": self.control_jitter_guard_steps,
                "minimum_delay_steps": self.minimum_delay_steps,
                "maximum_delay_steps": self.maximum_delay_steps,
            },
            domain=INFERENCE_DELAY_ESTIMATOR_SPEC_HASH_DOMAIN_V1,
        )

    def forecast_steps(self, latency_samples_sec: Sequence[float]) -> int:
        samples = tuple(float(value) for value in latency_samples_sec)[-self.window_size :]
        if any(not math.isfinite(value) or value < 0 for value in samples):
            raise ValueError("Inference latency samples must be finite and non-negative.")
        if not samples:
            estimated_latency = self.warmup_latency_sec
        elif self.estimator_kind == ROLLING_MAX_ESTIMATOR_V1:
            estimated_latency = max(samples)
        else:
            ordered = sorted(samples)
            rank = max(1, math.ceil(self.quantile * len(ordered)))
            estimated_latency = ordered[rank - 1]
        raw_steps = math.ceil(
            (estimated_latency + self.network_guard_sec) / self.action_period_sec
        ) + self.control_jitter_guard_steps
        return min(max(raw_steps, self.minimum_delay_steps), self.maximum_delay_steps)


@dataclass(frozen=True)
class InferenceDelayEstimatorState:
    """Immutable rolling samples; state is distinct from the logical spec."""

    spec_hash: str
    latency_samples_sec: tuple[float, ...] = ()
    accepted_sample_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "spec_hash", _nonempty(self.spec_hash, name="spec_hash"))
        samples = tuple(float(value) for value in self.latency_samples_sec)
        if any(not math.isfinite(value) or value < 0 for value in samples):
            raise ValueError("Inference latency samples must be finite and non-negative.")
        object.__setattr__(self, "latency_samples_sec", samples)
        object.__setattr__(
            self,
            "accepted_sample_count",
            _nonnegative_int(self.accepted_sample_count, name="accepted_sample_count"),
        )

    @classmethod
    def empty(cls, spec: InferenceDelayEstimatorSpec) -> "InferenceDelayEstimatorState":
        return cls(spec_hash=spec.semantic_content_hash)

    def record(
        self,
        spec: InferenceDelayEstimatorSpec,
        latency_sec: float,
    ) -> "InferenceDelayEstimatorState":
        if self.spec_hash != spec.semantic_content_hash:
            raise ValueError("Delay estimator state/spec hash mismatch.")
        latency = _finite(latency_sec, name="latency_sec")
        if latency < 0:
            raise ValueError("latency_sec must be non-negative.")
        samples = (self.latency_samples_sec + (latency,))[-spec.window_size :]
        return InferenceDelayEstimatorState(
            spec_hash=self.spec_hash,
            latency_samples_sec=samples,
            accepted_sample_count=self.accepted_sample_count + 1,
        )

    def forecast_steps(self, spec: InferenceDelayEstimatorSpec) -> int:
        if self.spec_hash != spec.semantic_content_hash:
            raise ValueError("Delay estimator state/spec hash mismatch.")
        return spec.forecast_steps(self.latency_samples_sec)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "spec_hash": self.spec_hash,
                "latency_samples_sec": self.latency_samples_sec,
                "accepted_sample_count": self.accepted_sample_count,
            },
            domain=INFERENCE_DELAY_ESTIMATOR_STATE_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class PolicyInferencePrefix:
    """Model-space suffix supplied as explicit RTC conditioning context."""

    model_space_actions: np.ndarray
    valid_steps: int
    source_plan_id: str
    source_plan_generation: int
    source_plan_hash: str

    def __post_init__(self) -> None:
        values = np.asarray(self.model_space_actions, dtype=np.float32).copy()
        if values.ndim != 2 or values.shape[0] <= 0 or values.shape[1] <= 0:
            raise ValueError("model_space_actions must be a non-empty rank-2 array.")
        if np.any(~np.isfinite(values)):
            raise ValueError("model_space_actions must be finite.")
        valid = _nonnegative_int(self.valid_steps, name="valid_steps")
        if valid > values.shape[0]:
            raise ValueError("valid_steps exceeds the supplied model-space prefix.")
        object.__setattr__(self, "model_space_actions", deep_freeze(values))
        object.__setattr__(self, "valid_steps", valid)
        object.__setattr__(self, "source_plan_id", _nonempty(self.source_plan_id, name="source_plan_id"))
        object.__setattr__(
            self,
            "source_plan_generation",
            _nonnegative_int(self.source_plan_generation, name="source_plan_generation"),
        )
        object.__setattr__(self, "source_plan_hash", _nonempty(self.source_plan_hash, name="source_plan_hash"))

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "model_space_actions": self.model_space_actions,
                "valid_steps": self.valid_steps,
                "source_plan_id": self.source_plan_id,
                "source_plan_generation": self.source_plan_generation,
                "source_plan_hash": self.source_plan_hash,
            },
            domain=POLICY_INFERENCE_PREFIX_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class PolicyInferenceRequest:
    """One admitted inference request bound to an observation and plan cut."""

    request_id: str
    inference_id: str
    rtc_mode: str
    session_id: str
    control_domain_id: str
    producer_id: str
    producer_epoch: int
    policy_node_id: str
    artifact_hash: str
    observation_cut_id: str
    observation_timestamp: float
    observation_step_index: int
    triggered_at: float
    admitted_at: float
    first_target_effective_at: float
    clock_domain_id: str
    action_period_sec: float
    prediction_horizon_steps: int
    minimum_execution_horizon_steps: int
    source_plan_id: str
    source_plan_generation: int
    source_plan_hash: str
    source_plan_cursor_at: float
    producer_plan_committed_through: float | None
    forecast_delay_steps: int
    delay_estimator_spec_hash: str
    action_codec_hash: str
    frame_contract_hash: str
    layout_hash: str
    normalization_hash: str
    valid_until: float
    conditioning_prefix: PolicyInferencePrefix | None = None
    action_representation_context: PolicyActionRepresentationContext | None = None

    def __post_init__(self) -> None:
        for name in (
            "request_id",
            "inference_id",
            "session_id",
            "control_domain_id",
            "producer_id",
            "policy_node_id",
            "artifact_hash",
            "observation_cut_id",
            "clock_domain_id",
            "source_plan_id",
            "source_plan_hash",
            "delay_estimator_spec_hash",
            "action_codec_hash",
            "frame_contract_hash",
            "layout_hash",
            "normalization_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name=name))
        mode = _nonempty(self.rtc_mode, name="rtc_mode")
        if mode not in POLICY_INFERENCE_MODES_V1:
            raise ValueError(f"Unsupported policy inference mode: {mode!r}.")
        object.__setattr__(self, "rtc_mode", mode)
        for name in (
            "observation_timestamp",
            "triggered_at",
            "admitted_at",
            "first_target_effective_at",
            "source_plan_cursor_at",
            "valid_until",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name=name))
        period = _finite(self.action_period_sec, name="action_period_sec")
        if period <= 0:
            raise ValueError("action_period_sec must be positive.")
        object.__setattr__(self, "action_period_sec", period)
        if self.triggered_at > self.admitted_at:
            raise ValueError("triggered_at must not be after admitted_at.")
        if self.first_target_effective_at < self.admitted_at:
            raise ValueError("first_target_effective_at must not predate request admission.")
        if self.valid_until <= self.first_target_effective_at:
            raise ValueError("valid_until must extend beyond the first target time.")
        if self.producer_plan_committed_through is not None:
            object.__setattr__(
                self,
                "producer_plan_committed_through",
                _finite(
                    self.producer_plan_committed_through,
                    name="producer_plan_committed_through",
                ),
            )
        for name in (
            "producer_epoch",
            "observation_step_index",
            "source_plan_generation",
            "forecast_delay_steps",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name),
            )
        prediction = _positive_int(
            self.prediction_horizon_steps,
            name="prediction_horizon_steps",
        )
        minimum = _positive_int(
            self.minimum_execution_horizon_steps,
            name="minimum_execution_horizon_steps",
        )
        if minimum > prediction:
            raise ValueError("minimum execution horizon exceeds prediction horizon.")
        object.__setattr__(self, "prediction_horizon_steps", prediction)
        object.__setattr__(self, "minimum_execution_horizon_steps", minimum)
        prefix = self.conditioning_prefix
        if prefix is not None:
            if prefix.source_plan_id != self.source_plan_id:
                raise ValueError("Conditioning prefix source plan id mismatch.")
            if prefix.source_plan_generation != self.source_plan_generation:
                raise ValueError("Conditioning prefix source plan generation mismatch.")
            if prefix.source_plan_hash != self.source_plan_hash:
                raise ValueError("Conditioning prefix source plan hash mismatch.")
        if mode == RTC_FLOW_INPAINT_MODE_V1 and prefix is None:
            raise ValueError("rtc_flow_inpaint_v1 requires a conditioning prefix.")
        context = self.action_representation_context
        if context is not None:
            if not isinstance(context, PolicyActionRepresentationContext):
                raise TypeError(
                    "action_representation_context must be "
                    "PolicyActionRepresentationContext or None."
                )
            if context.observation_cut_id != self.observation_cut_id:
                raise ValueError(
                    "Action representation context observation cut differs from request."
                )
            if context.observation_timestamp != self.observation_timestamp:
                raise ValueError(
                    "Action representation context timestamp differs from request."
                )
            if context.observation_step_index != self.observation_step_index:
                raise ValueError(
                    "Action representation context step index differs from request."
                )

    @property
    def conditioning_prefix_hash(self) -> str | None:
        if self.conditioning_prefix is None:
            return None
        return self.conditioning_prefix.semantic_content_hash

    @property
    def semantic_content_hash(self) -> str:
        payload = {
            "request_id": self.request_id,
            "inference_id": self.inference_id,
            "rtc_mode": self.rtc_mode,
            "session_id": self.session_id,
            "control_domain_id": self.control_domain_id,
            "producer_id": self.producer_id,
            "producer_epoch": self.producer_epoch,
            "policy_node_id": self.policy_node_id,
            "artifact_hash": self.artifact_hash,
            "observation_cut_id": self.observation_cut_id,
            "observation_timestamp": self.observation_timestamp,
            "observation_step_index": self.observation_step_index,
            "triggered_at": self.triggered_at,
            "admitted_at": self.admitted_at,
            "first_target_effective_at": self.first_target_effective_at,
            "clock_domain_id": self.clock_domain_id,
            "action_period_sec": self.action_period_sec,
            "prediction_horizon_steps": self.prediction_horizon_steps,
            "minimum_execution_horizon_steps": self.minimum_execution_horizon_steps,
            "source_plan_id": self.source_plan_id,
            "source_plan_generation": self.source_plan_generation,
            "source_plan_hash": self.source_plan_hash,
            "source_plan_cursor_at": self.source_plan_cursor_at,
            "producer_plan_committed_through": self.producer_plan_committed_through,
            "forecast_delay_steps": self.forecast_delay_steps,
            "delay_estimator_spec_hash": self.delay_estimator_spec_hash,
            "action_codec_hash": self.action_codec_hash,
            "frame_contract_hash": self.frame_contract_hash,
            "layout_hash": self.layout_hash,
            "normalization_hash": self.normalization_hash,
            "valid_until": self.valid_until,
            "conditioning_prefix_hash": self.conditioning_prefix_hash,
        }
        domain = POLICY_INFERENCE_REQUEST_HASH_DOMAIN_V1
        if self.action_representation_context is not None:
            payload["action_representation_context_hash"] = (
                self.action_representation_context.semantic_content_hash
            )
            domain = POLICY_INFERENCE_REQUEST_HASH_DOMAIN_V2
        return canonical_content_hash(payload, domain=domain)


@dataclass(frozen=True)
class PolicyChunkResult:
    """Immutable decoded worker output; it has no plan-admission authority."""

    request_id: str
    request_hash: str
    inference_id: str
    rtc_mode: str
    session_id: str
    control_domain_id: str
    producer_id: str
    producer_epoch: int
    policy_node_id: str
    artifact_hash: str
    action_codec_hash: str
    frame_contract_hash: str
    layout_hash: str
    normalization_hash: str
    observation_cut_id: str
    observation_timestamp: float
    observation_step_index: int
    policy_output_id: str | None
    chunk_id: int | None
    inference_started_at: float
    inference_completed_at: float
    result_received_at: float
    clock_domain_id: str
    status: str
    source_chunk_size: int
    decoded_absolute_actions: tuple[NormalizedAction, ...]
    model_output_hash: str | None
    protocol_version: str | None = None
    conditioning_prefix_hash: str | None = None
    effective_delay_steps: int | None = None
    execution_horizon_steps: int | None = None
    fault_code: str | None = None
    fault_detail: str | None = None
    canonical_actions: tuple[CanonicalPolicyAction, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "request_id",
            "request_hash",
            "inference_id",
            "session_id",
            "control_domain_id",
            "producer_id",
            "policy_node_id",
            "artifact_hash",
            "action_codec_hash",
            "frame_contract_hash",
            "layout_hash",
            "normalization_hash",
            "observation_cut_id",
            "clock_domain_id",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name=name))
        mode = _nonempty(self.rtc_mode, name="rtc_mode")
        if mode not in POLICY_INFERENCE_MODES_V1:
            raise ValueError(f"Unsupported policy inference mode: {mode!r}.")
        object.__setattr__(self, "rtc_mode", mode)
        status = _nonempty(self.status, name="status")
        if status not in {"ready", "fault"}:
            raise ValueError("PolicyChunkResult.status must be 'ready' or 'fault'.")
        object.__setattr__(self, "status", status)
        for name in (
            "observation_timestamp",
            "inference_started_at",
            "inference_completed_at",
            "result_received_at",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name=name))
        if self.inference_completed_at < self.inference_started_at:
            raise ValueError("Inference completion predates its start.")
        if self.result_received_at < self.inference_completed_at:
            raise ValueError("Result receipt predates inference completion.")
        object.__setattr__(
            self,
            "producer_epoch",
            _nonnegative_int(self.producer_epoch, name="producer_epoch"),
        )
        object.__setattr__(
            self,
            "observation_step_index",
            _nonnegative_int(
                self.observation_step_index,
                name="observation_step_index",
            ),
        )
        size = _nonnegative_int(self.source_chunk_size, name="source_chunk_size")
        actions = tuple(
            _freeze_normalized_action(action)
            for action in self.decoded_absolute_actions
        )
        canonical_actions = tuple(self.canonical_actions)
        if any(
            not isinstance(action, CanonicalPolicyAction)
            for action in canonical_actions
        ):
            raise TypeError("canonical_actions must contain CanonicalPolicyAction values.")
        if canonical_actions:
            expected_receipt_fields = {
                "observation_cut_id": self.observation_cut_id,
                "action_codec_hash": self.action_codec_hash,
                "frame_contract_hash": self.frame_contract_hash,
                "layout_hash": self.layout_hash,
                "normalization_hash": self.normalization_hash,
            }
            first_receipt = canonical_actions[0].canonicalization_receipt
            chunk_receipt_fields = {
                "source_pose_mode": first_receipt.source_pose_mode,
                "conversion_contract_hash": first_receipt.conversion_contract_hash,
                "anchor_identity_hash": first_receipt.anchor_identity_hash,
                "source_chunk_hash": first_receipt.source_chunk_hash,
                "canonical_chunk_hash": first_receipt.canonical_chunk_hash,
            }
            for action in canonical_actions:
                receipt = action.canonicalization_receipt
                for name, expected in expected_receipt_fields.items():
                    if getattr(receipt, name) != expected:
                        raise ValueError(
                            f"Canonical action receipt {name} does not match result."
                        )
                for name, expected in chunk_receipt_fields.items():
                    if getattr(receipt, name) != expected:
                        raise ValueError(
                            "Canonical actions in one result must share "
                            f"{name}."
                        )
            chunk_bound = first_receipt.source_chunk_hash is not None
            if chunk_bound:
                source_indices = tuple(
                    action.canonicalization_receipt.source_index
                    for action in canonical_actions
                )
                if source_indices != tuple(range(len(canonical_actions))):
                    raise ValueError(
                        "Chunk-bound canonical actions must preserve consecutive source indices."
                    )
                actual_chunk_hash = canonical_policy_robot_action_chunk_hash(
                    tuple(action.action for action in canonical_actions)
                )
                if actual_chunk_hash != first_receipt.canonical_chunk_hash:
                    raise ValueError(
                        "Canonical action chunk hash does not match its receipts."
                    )
        object.__setattr__(self, "source_chunk_size", size)
        object.__setattr__(self, "decoded_absolute_actions", actions)
        object.__setattr__(self, "canonical_actions", canonical_actions)
        if status == "ready":
            if bool(actions) == bool(canonical_actions):
                raise ValueError(
                    "A ready result must carry exactly one action payload representation."
                )
            selected_actions = canonical_actions if canonical_actions else actions
            if size <= 0 or len(selected_actions) != size:
                raise ValueError(
                    "A ready result must contain exactly source_chunk_size actions."
                )
            object.__setattr__(
                self,
                "policy_output_id",
                _nonempty(self.policy_output_id or "", name="policy_output_id"),
            )
            if self.chunk_id is None:
                raise ValueError("A ready result requires chunk_id.")
            object.__setattr__(
                self,
                "chunk_id",
                _nonnegative_int(self.chunk_id, name="chunk_id"),
            )
            object.__setattr__(
                self,
                "model_output_hash",
                _nonempty(self.model_output_hash or "", name="model_output_hash"),
            )
            if self.fault_code is not None or self.fault_detail is not None:
                raise ValueError("A ready result cannot carry a fault descriptor.")
        else:
            if size != 0 or actions or canonical_actions:
                raise ValueError("A fault result cannot carry an action payload.")
            object.__setattr__(self, "fault_code", _nonempty(self.fault_code or "", name="fault_code"))
            if self.policy_output_id is not None:
                object.__setattr__(
                    self,
                    "policy_output_id",
                    _nonempty(self.policy_output_id, name="policy_output_id"),
                )
            if self.chunk_id is not None:
                object.__setattr__(
                    self,
                    "chunk_id",
                    _nonnegative_int(self.chunk_id, name="chunk_id"),
                )
            if self.model_output_hash is not None:
                object.__setattr__(
                    self,
                    "model_output_hash",
                    _nonempty(self.model_output_hash, name="model_output_hash"),
                )
        for name in ("protocol_version", "conditioning_prefix_hash", "fault_detail"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, str(value))
        for name in ("effective_delay_steps", "execution_horizon_steps"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _nonnegative_int(value, name=name),
                )

    @property
    def action_payload_kind(self) -> str | None:
        if self.canonical_actions:
            return POLICY_ACTION_PAYLOAD_TYPED_CANONICAL_V1
        if self.decoded_absolute_actions:
            return POLICY_ACTION_PAYLOAD_LEGACY_NORMALIZED_V1
        return None

    @property
    def semantic_content_hash(self) -> str:
        payload = {
            "request_id": self.request_id,
            "request_hash": self.request_hash,
            "inference_id": self.inference_id,
            "rtc_mode": self.rtc_mode,
            "session_id": self.session_id,
            "control_domain_id": self.control_domain_id,
            "producer_id": self.producer_id,
            "producer_epoch": self.producer_epoch,
            "policy_node_id": self.policy_node_id,
            "artifact_hash": self.artifact_hash,
            "action_codec_hash": self.action_codec_hash,
            "frame_contract_hash": self.frame_contract_hash,
            "layout_hash": self.layout_hash,
            "normalization_hash": self.normalization_hash,
            "observation_cut_id": self.observation_cut_id,
            "observation_timestamp": self.observation_timestamp,
            "observation_step_index": self.observation_step_index,
            "policy_output_id": self.policy_output_id,
            "chunk_id": self.chunk_id,
            "inference_started_at": self.inference_started_at,
            "inference_completed_at": self.inference_completed_at,
            "result_received_at": self.result_received_at,
            "clock_domain_id": self.clock_domain_id,
            "status": self.status,
            "source_chunk_size": self.source_chunk_size,
            "decoded_absolute_actions": tuple(
                _normalized_action_semantic_payload(action)
                for action in self.decoded_absolute_actions
            ),
            "model_output_hash": self.model_output_hash,
            "protocol_version": self.protocol_version,
            "conditioning_prefix_hash": self.conditioning_prefix_hash,
            "effective_delay_steps": self.effective_delay_steps,
            "execution_horizon_steps": self.execution_horizon_steps,
            "fault_code": self.fault_code,
            "fault_detail": self.fault_detail,
        }
        if self.canonical_actions:
            payload.pop("decoded_absolute_actions")
            payload["action_payload_kind"] = self.action_payload_kind
            payload["canonical_actions"] = tuple(
                action.semantic_content_hash for action in self.canonical_actions
            )
            domain = POLICY_CHUNK_RESULT_HASH_DOMAIN_V2
        else:
            domain = POLICY_CHUNK_RESULT_HASH_DOMAIN_V1
        return canonical_content_hash(
            payload,
            domain=domain,
        )


__all__ = [
    "ASYNC_ALIGNED_CHUNK_MODE_V1",
    "INFERENCE_DELAY_ESTIMATORS_V1",
    "INFERENCE_DELAY_ESTIMATOR_SPEC_HASH_DOMAIN_V1",
    "INFERENCE_DELAY_ESTIMATOR_STATE_HASH_DOMAIN_V1",
    "InferenceDelayEstimatorSpec",
    "InferenceDelayEstimatorState",
    "NEAREST_RANK_QUANTILE_ESTIMATOR_V1",
    "POLICY_CHUNK_RESULT_HASH_DOMAIN_V1",
    "POLICY_INFERENCE_MODES_V1",
    "POLICY_INFERENCE_PREFIX_HASH_DOMAIN_V1",
    "POLICY_INFERENCE_REQUEST_HASH_DOMAIN_V1",
    "PolicyChunkResult",
    "PolicyInferencePrefix",
    "PolicyInferenceRequest",
    "ROLLING_MAX_ESTIMATOR_V1",
    "RTC_FLOW_INPAINT_MODE_V1",
    "SYNC_QUEUE_MODE_V1",
]
