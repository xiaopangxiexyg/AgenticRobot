"""Default-off, target-neutral RTC chunk deployment coordinator.

The coordinator owns one bounded inference worker and one immutable timed-plan
producer snapshot.  It is independent of teleoperation, replay, Control
Station, simulation, model family, embodiment, target backend and hardware.
Device ownership, canonical safety, position reference and dispatch remain in
the embedding deployment application.  This module never writes RTC actions
into the legacy synchronous deque.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.model_continuation import (
    normalize_model_continuation_contract,
)
from dataset_contracts.policy_action_semantics import (
    PolicyArmHoldOutputContract,
)
from robot_motion.relative_action import normalize_action_pose_mode

from .action_adapter import DeployRobotActionAdapter
from .action_representation import PolicyActionRepresentationContext
from .async_producer import (
    POLICY_ASYNC_PRODUCER_ELIGIBLE,
    POLICY_ASYNC_PRODUCER_FAULT,
    POLICY_ASYNC_PRODUCER_HOLD,
    POLICY_ASYNC_PRODUCER_TRANSITIONING,
    POLICY_WORKER_SESSION_ACTIVE,
    POLICY_WORKER_SESSION_DRAINED,
    POLICY_WORKER_SESSION_RETIRED_WAITING_FOR_COMPLETION,
    RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1,
    RTC_3C_VIRTUAL_SHADOW_PROFILE_V1,
    PolicyAsyncExecutionProfile,
    PolicyAsyncProducerSnapshot,
    PolicyAsyncProducerTransitionOutcome,
    admit_policy_chunk_result,
    admit_policy_inference_request,
    commit_emitted_policy_async_producer_action,
    create_policy_async_producer,
    invalidate_policy_async_producer,
    pop_due_policy_async_producer_action,
    rearm_policy_async_producer,
    update_policy_worker_session_state,
)
from .chunk_plan import empty_policy_action_plan
from .inference import (
    INFERENCE_DELAY_VALIDITY_POLICY_V1,
    RTC_FLOW_INPAINT_MODE_V1,
    InferenceDelayEstimatorSpec,
    InferenceDelayEstimatorState,
    PolicyChunkResult,
    PolicyInferenceRequest,
)
from .observation import DeploymentObservation
from .observation_view import PolicyObservationView
from .policy_worker import (
    BoundedPolicyInferenceWorker,
    PolicyInferenceExecutor,
    PolicyInferenceWorkItem,
    PolicyWorkerCompletion,
    PolicyWorkerStopReceipt,
    PolicyWorkerSubmissionReceipt,
)
from .rtc.admission import (
    RTC_REQUEST_BRIDGE_GUARD_STEPS,
    RtcExecutionCoverageAssessment,
    RtcLatencyEnvelopeAssessment,
    assess_rtc_execution_coverage,
    assess_rtc_latency_envelope,
    rtc_latency_budget_steps,
)
from .rtc.conditioning import (
    CanonicalRtcConditioningBundle,
    RtcModelContinuationRuntimeSpec,
    build_canonical_rtc_conditioning_bundle,
)
from .rtc.continuation import (
    RtcContinuationTimeline,
    select_rtc_plan_continuation,
)
from .rtc.profile import RtcProducerProfile
from .runtime import (
    PolicyActionEmission,
    PolicyActionEmissionResult,
    PolicyActionPreparationResult,
    PolicyDeployRuntime,
)

RTC_3C_FRONTEND_CONFIG_HASH_DOMAIN_V1 = (
    "dexteleop.deploy.rtc_3c_frontend_config.v1"
)
RTC_3C_FRONTEND_CONFIG_HASH_DOMAIN_V2 = (
    "dexteleop.deploy.rtc_3c_frontend_config.v2"
)
RTC_3C_FRONTEND_CONFIG_HASH_DOMAIN_V3 = (
    "dexteleop.deploy.rtc_3c_frontend_config.v3"
)
RTC_3C_FRONTEND_CONFIG_HASH_DOMAIN_V4 = (
    "dexteleop.deploy.rtc_3c_frontend_config.v4"
)
RTC_TIMING_CONTRACT_HASH_DOMAIN_V1 = (
    "dexteleop.deploy.rtc_timing_contract.v1"
)
RTC_REQUEST_VALIDITY_POLICY_V1 = (
    INFERENCE_DELAY_VALIDITY_POLICY_V1
)
RTC_MODEL_WORKER_STOP_POLICY_V1 = (
    "model_continuation_default_drain_timeout_v1"
)
RTC_MODEL_WORKER_STOP_DISABLED_POLICY_V1 = (
    "disabled_for_non_model_rtc_v1"
)
RTC_MODEL_WORKER_STOP_DEFAULT_TIMEOUT_SEC = 2.0
RTC_3C_LOCAL_POLICY_PROTOCOL_V1 = "rtc_3c_local_policy_worker_v1"
RTC_OBSERVATION_CUT_IDENTITY_HASH_DOMAIN_V2 = (
    "dexteleop.rtc_observation_cut_identity.v2"
)
RTC_POLICY_EMISSION_SOURCE_BINDING_HASH_DOMAIN_V1 = (
    "dexteleop.deploy.rtc_frontend_step_source_binding.v1"
)
RTC_POLICY_RUNTIME_SOURCE_CONFIG_HASH_DOMAIN_V1 = (
    "dexteleop.deploy.rtc_policy_runtime_source_config.v1"
)
RTC_TYPED_CANONICAL_SOURCE_ADAPTER_CONTRACT_HASH_V1 = (
    "not-applicable:rtc-typed-canonical-source-adapter-v1"
)
RTC_POLICY_EMISSION_SOURCE_DESCRIPTOR_HASH_DOMAIN_V1 = (
    "dexteleop.deploy.rtc_policy_emission_source_descriptor.v1"
)
POLICY_ACTION_EMISSION_COMMIT_RECEIPT_HASH_DOMAIN_V1 = (
    "dexteleop.deploy.policy_action_emission_commit_receipt.v1"
)
RTC_PRODUCER_ACTIVATION_READINESS_HASH_DOMAIN_V1 = (
    "dexteleop.deploy.rtc_producer_activation_readiness.v1"
)


def _required(value: object, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


def _finite(value: Any, *, field_name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


def _positive_integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer.")
    normalized = int(value)
    if normalized != value or normalized <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return normalized


def _nonnegative_integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer.")
    normalized = int(value)
    if normalized != value or normalized < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return normalized


def _default_delay_spec() -> InferenceDelayEstimatorSpec:
    period = 1.0 / RTC_3C_VIRTUAL_SHADOW_PROFILE_V1.action_hz
    return InferenceDelayEstimatorSpec(
        spec_id="rtc_3c_virtual_shadow_delay_rolling_max_v1",
        action_period_sec=period,
        window_size=32,
        warmup_latency_sec=period,
        network_guard_sec=0.0,
        control_jitter_guard_steps=1,
        minimum_delay_steps=1,
        maximum_delay_steps=16,
    )


@dataclass(frozen=True)
class RtcPolicyShadowConfig:
    runtime_session_id: str
    control_domain_id: str
    producer_id: str
    policy_node_id: str
    artifact_hash: str
    action_codec_hash: str
    frame_contract_hash: str
    layout_hash: str
    normalization_hash: str
    profile: PolicyAsyncExecutionProfile = RTC_3C_VIRTUAL_SHADOW_PROFILE_V1
    clock_domain_id: str = "local_monotonic"
    prediction_horizon_steps: int = 8
    minimum_execution_horizon_steps: int = 1
    request_validity_steps: int = 10
    request_trigger_coverage_steps: int = 2
    worker_result_capacity: int = 1
    worker_idle_poll_sec: float = 0.001
    worker_stop_timeout_sec: float = 0.0
    delay_estimator_spec: InferenceDelayEstimatorSpec = field(
        default_factory=_default_delay_spec
    )
    action_pose_mode: str | None = None
    model_continuation: RtcModelContinuationRuntimeSpec | None = None

    def __post_init__(self) -> None:
        for name in (
            "runtime_session_id",
            "control_domain_id",
            "producer_id",
            "policy_node_id",
            "artifact_hash",
            "action_codec_hash",
            "frame_contract_hash",
            "layout_hash",
            "normalization_hash",
            "clock_domain_id",
        ):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), field_name=name),
            )
        if not isinstance(self.profile, PolicyAsyncExecutionProfile):
            raise TypeError("profile must be PolicyAsyncExecutionProfile.")
        if self.profile.position_lookahead_enabled:
            raise ValueError("RTC shadow profiles require lookahead disabled.")
        if (
            self.profile.profile_id == RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1
            and self.profile.send_actions
        ):
            raise ValueError(
                "RTC-3c virtual profile has no model action source and does not "
                "accept --send-actions."
            )
        for name in (
            "prediction_horizon_steps",
            "minimum_execution_horizon_steps",
            "request_validity_steps",
            "worker_result_capacity",
        ):
            object.__setattr__(
                self,
                name,
                _positive_integer(getattr(self, name), field_name=name),
            )
        coverage = _nonnegative_integer(
            self.request_trigger_coverage_steps,
            field_name="request_trigger_coverage_steps",
        )
        object.__setattr__(self, "request_trigger_coverage_steps", coverage)
        if self.minimum_execution_horizon_steps > self.prediction_horizon_steps:
            raise ValueError("minimum execution horizon exceeds prediction horizon.")
        if self.request_validity_steps < self.minimum_execution_horizon_steps:
            raise ValueError("request validity cannot be below the minimum horizon.")
        idle_poll = _finite(
            self.worker_idle_poll_sec,
            field_name="worker_idle_poll_sec",
        )
        stop_timeout = _finite(
            self.worker_stop_timeout_sec,
            field_name="worker_stop_timeout_sec",
        )
        if idle_poll <= 0.0 or stop_timeout < 0.0:
            raise ValueError("worker poll must be positive and stop timeout non-negative.")
        object.__setattr__(self, "worker_idle_poll_sec", idle_poll)
        object.__setattr__(self, "worker_stop_timeout_sec", stop_timeout)
        if not isinstance(self.delay_estimator_spec, InferenceDelayEstimatorSpec):
            raise TypeError("delay_estimator_spec must be InferenceDelayEstimatorSpec.")
        if self.action_pose_mode is not None:
            object.__setattr__(
                self,
                "action_pose_mode",
                normalize_action_pose_mode(self.action_pose_mode),
            )
        continuation = self.model_continuation
        if continuation is not None:
            if not isinstance(
                continuation,
                RtcModelContinuationRuntimeSpec,
            ):
                raise TypeError(
                    "model_continuation must be an "
                    "RtcModelContinuationRuntimeSpec or None."
                )
            contract = normalize_model_continuation_contract(
                continuation.continuation_contract
            ).runtime
            if self.profile.rtc_mode != RTC_FLOW_INPAINT_MODE_V1:
                raise ValueError(
                    "model_continuation requires rtc_flow_inpaint_v1."
                )
            if self.action_pose_mode != "absolute":
                raise ValueError(
                    "Model-RTC C1 v1 requires explicit absolute pose mode."
                )
            expected = {
                "prediction_horizon_steps": self.prediction_horizon_steps,
                "action_codec_hash": self.action_codec_hash,
                "frame_contract_hash": self.frame_contract_hash,
                "layout_hash": self.layout_hash,
                "normalization_hash": self.normalization_hash,
            }
            for name, expected_value in expected.items():
                if getattr(contract, name) != expected_value:
                    raise ValueError(
                        f"model_continuation {name} mismatch."
                    )
            if (
                self.delay_estimator_spec.minimum_delay_steps
                < contract.minimum_forecast_delay_steps
                or self.delay_estimator_spec.maximum_delay_steps
                > contract.maximum_forecast_delay_steps
            ):
                raise ValueError(
                    "Delay estimator range exceeds the continuation contract."
                )
        expected_period = 1.0 / self.profile.action_hz
        if not math.isclose(
            self.delay_estimator_spec.action_period_sec,
            expected_period,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("delay estimator/action profile period mismatch.")

    @property
    def minimum_safe_request_validity_steps(self) -> int:
        """Conservative TTL budget for the configured delay range."""

        return self.delay_estimator_spec.minimum_safe_request_validity_steps(
            self.minimum_execution_horizon_steps,
        )

    def effective_request_validity_steps(self, forecast_delay_steps: int) -> int:
        """Return the per-request TTL, including the forecast and execution tail.

        ``valid_until`` is measured from the first target rather than from
        request admission.  A request whose current delay forecast is large
        therefore needs a correspondingly larger validity window to leave the
        provider enough time to return a result and the plan enough time to
        expose the minimum executable suffix.
        """

        forecast = _nonnegative_integer(
            forecast_delay_steps,
            field_name="forecast_delay_steps",
        )
        spec = self.delay_estimator_spec
        if not spec.minimum_delay_steps <= forecast <= spec.maximum_delay_steps:
            raise ValueError(
                "forecast_delay_steps must lie within the delay estimator range."
            )
        return max(
            self.request_validity_steps,
            forecast + self.minimum_execution_horizon_steps,
        )

    @property
    def effective_worker_stop_timeout_sec(self) -> float:
        """Bound model-RTC shutdown long enough to drain its single worker.

        Ordinary synchronous/shadow profiles retain their historical zero
        timeout.  Model-RTC has a retired-session/rearm invariant, so an
        omitted timeout must not silently abandon an in-flight provider call.
        An explicit positive timeout remains authoritative.
        """

        if (
            self.model_continuation is not None
            and self.worker_stop_timeout_sec == 0.0
        ):
            return RTC_MODEL_WORKER_STOP_DEFAULT_TIMEOUT_SEC
        return self.worker_stop_timeout_sec

    @property
    def timing_contract(self) -> dict[str, Any]:
        """Portable timing facts exposed in hashes and debug receipts."""

        return {
            "schema": "dexteleop.rtc_timing_contract.v1",
            "action_period_sec": self.delay_estimator_spec.action_period_sec,
            "minimum_execution_horizon_steps": (
                self.minimum_execution_horizon_steps
            ),
            "maximum_forecast_delay_steps": (
                self.delay_estimator_spec.maximum_delay_steps
            ),
            "minimum_safe_request_validity_steps": (
                self.minimum_safe_request_validity_steps
            ),
            "configured_request_validity_steps": self.request_validity_steps,
            "request_validity_policy": RTC_REQUEST_VALIDITY_POLICY_V1,
            "configured_worker_stop_timeout_sec": self.worker_stop_timeout_sec,
            "effective_worker_stop_timeout_sec": (
                self.effective_worker_stop_timeout_sec
            ),
            "worker_stop_policy": (
                RTC_MODEL_WORKER_STOP_POLICY_V1
                if self.model_continuation is not None
                else RTC_MODEL_WORKER_STOP_DISABLED_POLICY_V1
            ),
        }

    @property
    def timing_contract_hash(self) -> str:
        return canonical_content_hash(
            self.timing_contract,
            domain=RTC_TIMING_CONTRACT_HASH_DOMAIN_V1,
        )

    @property
    def semantic_content_hash(self) -> str:
        payload = {
            "runtime_session_id": self.runtime_session_id,
            "control_domain_id": self.control_domain_id,
            "producer_id": self.producer_id,
            "policy_node_id": self.policy_node_id,
            "artifact_hash": self.artifact_hash,
            "action_codec_hash": self.action_codec_hash,
            "frame_contract_hash": self.frame_contract_hash,
            "layout_hash": self.layout_hash,
            "normalization_hash": self.normalization_hash,
            "profile_hash": self.profile.semantic_content_hash,
            "clock_domain_id": self.clock_domain_id,
            "prediction_horizon_steps": self.prediction_horizon_steps,
            "minimum_execution_horizon_steps": (
                self.minimum_execution_horizon_steps
            ),
            "request_validity_steps": self.request_validity_steps,
            "request_trigger_coverage_steps": (
                self.request_trigger_coverage_steps
            ),
            "worker_result_capacity": self.worker_result_capacity,
            "worker_idle_poll_sec": self.worker_idle_poll_sec,
            "worker_stop_timeout_sec": self.worker_stop_timeout_sec,
            "timing_contract_hash": self.timing_contract_hash,
            "delay_estimator_spec_hash": (
                self.delay_estimator_spec.semantic_content_hash
            ),
        }
        # V4 is the single current hash domain. V1--V3 remain exported as
        # historical identifiers for receipt readers, but a new config must
        # never silently select an older semantic domain when optional pose
        # or model-continuation fields are present.
        domain = RTC_3C_FRONTEND_CONFIG_HASH_DOMAIN_V4
        if self.action_pose_mode is not None:
            payload["action_pose_mode"] = self.action_pose_mode
        if self.model_continuation is not None:
            payload["model_continuation_hash"] = (
                self.model_continuation.semantic_content_hash
            )
        return canonical_content_hash(payload, domain=domain)

    @property
    def producer_profile(self) -> RtcProducerProfile:
        """Target-neutral schedule used by the lower RTC producer."""

        return self.profile.producer_profile


@dataclass(frozen=True)
class RtcFrontendLifecycleTransition:
    producer_transition: PolicyAsyncProducerTransitionOutcome
    worker_stop_receipt: PolicyWorkerStopReceipt


@dataclass(frozen=True)
class PolicyActionEmissionCommitReceipt:
    """Producer-local proof that one popped keyframe became one emission."""

    accepted: bool
    emission_id: str
    emission_semantic_content_hash: str
    source_binding_hash: str
    producer_id: str
    producer_epoch: int
    committed_at: float
    producer_transition_receipt_hash: str
    semantic_content_hash: str

    def __post_init__(self) -> None:
        if not self.accepted:
            raise ValueError("Emission commit receipts represent accepted commits only.")
        for name in (
            "emission_id",
            "emission_semantic_content_hash",
            "source_binding_hash",
            "producer_id",
            "producer_transition_receipt_hash",
            "semantic_content_hash",
        ):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), field_name=name),
            )
        object.__setattr__(
            self,
            "producer_epoch",
            _nonnegative_integer(self.producer_epoch, field_name="producer_epoch"),
        )
        object.__setattr__(
            self,
            "committed_at",
            _finite(self.committed_at, field_name="committed_at"),
        )
        expected_hash = canonical_content_hash(
            {
                "accepted": True,
                "emission_id": self.emission_id,
                "emission_semantic_content_hash": (
                    self.emission_semantic_content_hash
                ),
                "source_binding_hash": self.source_binding_hash,
                "producer_id": self.producer_id,
                "producer_epoch": self.producer_epoch,
                "committed_at": self.committed_at,
                "producer_transition_receipt_hash": (
                    self.producer_transition_receipt_hash
                ),
            },
            domain=POLICY_ACTION_EMISSION_COMMIT_RECEIPT_HASH_DOMAIN_V1,
        )
        if self.semantic_content_hash != expected_hash:
            raise ValueError("Emission commit receipt semantic hash mismatch.")


@dataclass(frozen=True)
class RtcProducerActivationReadinessReceipt:
    """Backend-neutral evidence that a new producer epoch may be activated."""

    ready: bool
    readiness_kind: str
    producer_id: str
    old_producer_epoch: int
    new_producer_epoch: int
    occurred_at: float
    evidence_hashes: tuple[str, ...]
    reasons: tuple[str, ...]
    semantic_content_hash: str

    def __post_init__(self) -> None:
        for name in ("readiness_kind", "producer_id", "semantic_content_hash"):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), field_name=name),
            )
        old_epoch = _nonnegative_integer(
            self.old_producer_epoch,
            field_name="old_producer_epoch",
        )
        new_epoch = _nonnegative_integer(
            self.new_producer_epoch,
            field_name="new_producer_epoch",
        )
        if new_epoch <= old_epoch:
            raise ValueError("new_producer_epoch must advance.")
        object.__setattr__(self, "old_producer_epoch", old_epoch)
        object.__setattr__(self, "new_producer_epoch", new_epoch)
        object.__setattr__(
            self,
            "occurred_at",
            _finite(self.occurred_at, field_name="occurred_at"),
        )
        evidence = tuple(
            _required(value, field_name="evidence_hash")
            for value in self.evidence_hashes
        )
        reasons = tuple(
            _required(value, field_name="reason") for value in self.reasons
        )
        if self.ready and reasons:
            raise ValueError("A ready receipt cannot carry rejection reasons.")
        if not self.ready and not reasons:
            raise ValueError("A non-ready receipt requires rejection reasons.")
        object.__setattr__(self, "evidence_hashes", evidence)
        object.__setattr__(self, "reasons", reasons)
        expected_hash = canonical_content_hash(
            {
                "ready": self.ready,
                "readiness_kind": self.readiness_kind,
                "producer_id": self.producer_id,
                "old_producer_epoch": self.old_producer_epoch,
                "new_producer_epoch": self.new_producer_epoch,
                "occurred_at": self.occurred_at,
                "evidence_hashes": self.evidence_hashes,
                "reasons": self.reasons,
            },
            domain=RTC_PRODUCER_ACTIVATION_READINESS_HASH_DOMAIN_V1,
        )
        if self.semantic_content_hash != expected_hash:
            raise ValueError("Producer activation readiness semantic hash mismatch.")


@dataclass(frozen=True)
class RtcFrontendStep:
    state: PolicyAsyncProducerSnapshot
    completion: PolicyWorkerCompletion | None
    result_transition: PolicyAsyncProducerTransitionOutcome | None
    pop_transition: PolicyAsyncProducerTransitionOutcome | None
    emission_commit_transition: PolicyAsyncProducerTransitionOutcome | None
    emission_commit_receipt: PolicyActionEmissionCommitReceipt | None
    request_transition: PolicyAsyncProducerTransitionOutcome | None
    worker_submission: PolicyWorkerSubmissionReceipt | None
    emission_result: PolicyActionEmissionResult
    lifecycle_transition: RtcFrontendLifecycleTransition | None
    underrun: bool
    control_duration_sec: float
    control_budget_sec: float
    selected_emission_count: int
    late_completion_rejection_count: int
    forecast_delay_steps: int | None = None
    request_validity_steps: int | None = None
    effective_request_trigger_coverage_steps: int = 0
    plan_coverage_steps: int = 0
    completion_coverage_assessment: RtcExecutionCoverageAssessment | None = None
    timing_contract_hash: str = ""
    preflight_completion: bool = False

    @property
    def budget_exceeded(self) -> bool:
        return self.control_duration_sec > self.control_budget_sec

    @property
    def action_preparation(self) -> PolicyActionPreparationResult:
        """Compatibility view with no downstream prepared action.

        RTC now terminates at the formal emission.  Embedding applications
        explicitly prepare that emission after ``step()`` returns.
        """

        return PolicyActionPreparationResult(
            raw_action=self.emission_result.raw_action,
            raw_lookahead_action=None,
            prepared_action=None,
            emission=self.emission_result.emission,
            emission_error=self.emission_result.emission_error,
            queue_len_before=self.emission_result.queue_len_before,
            queue_len_after=self.emission_result.queue_len_after,
            canonical_action=self.emission_result.canonical_action,
            model_raw_action=(
                None
                if self.emission_result.emission is None
                else self.emission_result.emission.model_decoded_absolute_action
            ),
            arm_hold_resolution=(
                None
                if self.emission_result.emission is None
                else self.emission_result.emission.arm_hold_resolution
            ),
        )

    def to_debug_payload(self) -> dict[str, Any]:
        emission = self.emission_result.emission
        pop = None if self.pop_transition is None else self.pop_transition.pop_outcome
        result_receipt = (
            None if self.result_transition is None else self.result_transition.receipt
        )
        merge_receipt = (
            None
            if self.result_transition is None
            else self.result_transition.merge_receipt
        )
        return {
            "schema": "dexteleop_rtc_3c_frontend_step_v1",
            "state_hash": self.state.semantic_content_hash,
            "lifecycle_state": self.state.lifecycle_state,
            "worker_session_id": self.state.worker_session_id,
            "worker_session_state": self.state.worker_session_state,
            "active_request_id": (
                None
                if self.state.active_request is None
                else self.state.active_request.request_id
            ),
            "plan_generation": self.state.plan.generation,
            "plan_keyframe_count": len(self.state.plan.keyframes),
            "producer_plan_committed_through": (
                self.state.plan.producer_committed_through
            ),
            "completion_hash": (
                None if self.completion is None else self.completion.semantic_content_hash
            ),
            "completion_status": (
                None if self.completion is None else self.completion.status
            ),
            "completion_fault_code": (
                None if self.completion is None else self.completion.result.fault_code
            ),
            "completion_fault_detail": (
                None if self.completion is None else self.completion.result.fault_detail
            ),
            "preflight_completion": self.preflight_completion,
            "completion_worker_started_at": (
                None
                if self.completion is None
                else self.completion.worker_started_at
            ),
            "completion_worker_completed_at": (
                None
                if self.completion is None
                else self.completion.worker_completed_at
            ),
            "completion_released_at": (
                None if self.completion is None else self.completion.released_at
            ),
            "result_transition_hash": (
                None
                if result_receipt is None
                else result_receipt.semantic_content_hash
            ),
            "result_transition_accepted": (
                None if result_receipt is None else result_receipt.accepted
            ),
            "result_transition_reasons": (
                [] if result_receipt is None else list(result_receipt.reasons)
            ),
            "plan_merge_accepted": (
                None if merge_receipt is None else merge_receipt.accepted
            ),
            "plan_merge_reasons": (
                [] if merge_receipt is None else list(merge_receipt.reasons)
            ),
            "plan_merge_admitted_keyframe_count": (
                0 if merge_receipt is None else merge_receipt.admitted_keyframe_count
            ),
            "plan_merge_discarded_obsolete_count": (
                0 if merge_receipt is None else merge_receipt.discarded_obsolete_count
            ),
            "emission_commit_transition_hash": (
                None
                if self.emission_commit_transition is None
                else self.emission_commit_transition.receipt.semantic_content_hash
            ),
            "emission_commit_receipt_hash": (
                None
                if self.emission_commit_receipt is None
                else self.emission_commit_receipt.semantic_content_hash
            ),
            "request_transition_hash": (
                None
                if self.request_transition is None
                else self.request_transition.receipt.semantic_content_hash
            ),
            "request_transition_accepted": (
                None
                if self.request_transition is None
                else self.request_transition.receipt.accepted
            ),
            "request_transition_reasons": (
                []
                if self.request_transition is None
                else list(self.request_transition.receipt.reasons)
            ),
            "worker_submission_hash": (
                None
                if self.worker_submission is None
                else self.worker_submission.semantic_content_hash
            ),
            "worker_submission_accepted": (
                None
                if self.worker_submission is None
                else self.worker_submission.accepted
            ),
            "worker_submission_status": (
                None
                if self.worker_submission is None
                else self.worker_submission.status
            ),
            "worker_submission_request_id": (
                None
                if self.worker_submission is None
                else self.worker_submission.request_id
            ),
            "lifecycle_transition_hash": (
                None
                if self.lifecycle_transition is None
                else self.lifecycle_transition.producer_transition.receipt.semantic_content_hash
            ),
            "lifecycle_transition_reason": (
                None
                if self.lifecycle_transition is None
                else self.lifecycle_transition.producer_transition.state.last_transition_reason
            ),
            "selected_target_effective_at": (
                None if pop is None else pop.selected_target_effective_at
            ),
            "oldest_skipped_target_effective_at": (
                None if pop is None else pop.oldest_skipped_target_effective_at
            ),
            "selected_lateness_sec": (
                None if pop is None else pop.selected_lateness_sec
            ),
            "emission_id": None if emission is None else emission.emission_id,
            "emission_source_binding_hash": (
                None if emission is None else emission.source_binding_hash
            ),
            "underrun": self.underrun,
            "control_duration_sec": self.control_duration_sec,
            "control_budget_sec": self.control_budget_sec,
            "budget_exceeded": self.budget_exceeded,
            "selected_emission_count": self.selected_emission_count,
            "late_completion_rejection_count": (
                self.late_completion_rejection_count
            ),
            "forecast_delay_steps": self.forecast_delay_steps,
            "request_validity_steps": self.request_validity_steps,
            "effective_request_trigger_coverage_steps": (
                self.effective_request_trigger_coverage_steps
            ),
            "plan_coverage_steps": self.plan_coverage_steps,
            "completion_coverage_assessment": (
                None
                if self.completion_coverage_assessment is None
                else self.completion_coverage_assessment.to_payload()
            ),
            "timing_contract_hash": self.timing_contract_hash,
            "position_lookahead_enabled": False,
            "send_actions": False,
        }


@dataclass(frozen=True)
class RtcPolicyEmissionSourceDescriptor:
    """Stable policy/action description shared by multiple producer epochs."""

    source_id: str
    policy_node_id: str
    policy_runtime_config_hash: str
    action_adapter_contract_hash: str
    rtc_profile_hash: str
    artifact_hash: str
    action_codec_hash: str
    frame_contract_hash: str
    layout_hash: str
    normalization_hash: str

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "source_id": self.source_id,
                "policy_node_id": self.policy_node_id,
                "policy_runtime_config_hash": self.policy_runtime_config_hash,
                "action_adapter_contract_hash": self.action_adapter_contract_hash,
                "rtc_profile_hash": self.rtc_profile_hash,
                "artifact_hash": self.artifact_hash,
                "action_codec_hash": self.action_codec_hash,
                "frame_contract_hash": self.frame_contract_hash,
                "layout_hash": self.layout_hash,
                "normalization_hash": self.normalization_hash,
            },
            domain=RTC_POLICY_EMISSION_SOURCE_DESCRIPTOR_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class RtcPolicyEmissionSourceBinding:
    """Immutable identity for one RTC producer epoch and emission stream."""

    source_slot_id: str
    runtime_session_id: str
    control_domain_id: str
    producer_id: str
    producer_epoch: int
    policy_node_id: str
    policy_clock_domain_id: str
    policy_runtime_config_hash: str
    action_adapter_contract_hash: str
    rtc_frontend_config_hash: str
    rtc_profile_hash: str
    artifact_hash: str
    action_codec_hash: str
    frame_contract_hash: str
    layout_hash: str
    normalization_hash: str

    @property
    def source_descriptor(self) -> RtcPolicyEmissionSourceDescriptor:
        return RtcPolicyEmissionSourceDescriptor(
            source_id=self.source_slot_id,
            policy_node_id=self.policy_node_id,
            policy_runtime_config_hash=self.policy_runtime_config_hash,
            action_adapter_contract_hash=self.action_adapter_contract_hash,
            rtc_profile_hash=self.rtc_profile_hash,
            artifact_hash=self.artifact_hash,
            action_codec_hash=self.action_codec_hash,
            frame_contract_hash=self.frame_contract_hash,
            layout_hash=self.layout_hash,
            normalization_hash=self.normalization_hash,
        )

    @property
    def descriptor_hash(self) -> str:
        return self.source_descriptor.semantic_content_hash

    @property
    def binding_generation(self) -> int:
        """V1 compatibility generation; explicit generation starts at epoch."""

        return self.producer_epoch

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "source_slot_id": self.source_slot_id,
                "runtime_session_id": self.runtime_session_id,
                "control_domain_id": self.control_domain_id,
                "producer_id": self.producer_id,
                "producer_epoch": self.producer_epoch,
                "policy_node_id": self.policy_node_id,
                "policy_clock_domain_id": self.policy_clock_domain_id,
                "policy_runtime_config_hash": self.policy_runtime_config_hash,
                "action_adapter_contract_hash": self.action_adapter_contract_hash,
                "rtc_frontend_config_hash": self.rtc_frontend_config_hash,
                "rtc_profile_hash": self.rtc_profile_hash,
                "artifact_hash": self.artifact_hash,
                "action_codec_hash": self.action_codec_hash,
                "frame_contract_hash": self.frame_contract_hash,
                "layout_hash": self.layout_hash,
                "normalization_hash": self.normalization_hash,
            },
            domain=RTC_POLICY_EMISSION_SOURCE_BINDING_HASH_DOMAIN_V1,
        )

    def validate_emission(
        self,
        emission: PolicyActionEmission,
    ) -> PolicyActionEmission:
        if not isinstance(emission, PolicyActionEmission):
            raise TypeError("emission must be PolicyActionEmission.")
        if emission.source_binding_hash != self.semantic_content_hash:
            raise ValueError("RTC emission source binding identity mismatch.")
        if emission.policy_node_id != self.policy_node_id:
            raise ValueError("RTC emission belongs to another policy node.")
        if (
            emission.produced_clock_domain_id != self.policy_clock_domain_id
            or emission.effective_clock_domain_id != self.policy_clock_domain_id
        ):
            raise ValueError("RTC emission clock domain does not match source binding.")
        if emission.queue_pop_count <= 0:
            raise ValueError("RTC emission has no lower-owned plan-pop evidence.")
        receipt = emission.representation_receipt
        if receipt is not None and (
            receipt.adapter_contract_hash != self.action_adapter_contract_hash
        ):
            raise ValueError("RTC emission used another action adapter contract.")
        return emission

    def validate_step(self, step: RtcFrontendStep) -> PolicyActionEmission | None:
        """Compatibility/debug validation of producer pop and commit evidence."""

        if not isinstance(step, RtcFrontendStep):
            raise TypeError("step must be an RtcFrontendStep.")
        plan = step.state.plan
        if (
            plan.session_id != self.runtime_session_id
            or plan.control_domain_id != self.control_domain_id
            or plan.producer_id != self.producer_id
            or plan.producer_epoch != self.producer_epoch
            or plan.policy_node_id != self.policy_node_id
            or plan.clock_domain_id != self.policy_clock_domain_id
        ):
            raise ValueError("RTC step producer state does not match source binding.")
        request = step.state.active_request
        if request is not None and (
            request.session_id != self.runtime_session_id
            or request.control_domain_id != self.control_domain_id
            or request.producer_id != self.producer_id
            or request.producer_epoch != self.producer_epoch
            or request.policy_node_id != self.policy_node_id
            or request.clock_domain_id != self.policy_clock_domain_id
            or request.artifact_hash != self.artifact_hash
            or request.action_codec_hash != self.action_codec_hash
            or request.frame_contract_hash != self.frame_contract_hash
            or request.layout_hash != self.layout_hash
            or request.normalization_hash != self.normalization_hash
        ):
            raise ValueError("RTC active request does not match source binding.")
        completion = step.completion
        if completion is not None:
            result = completion.result
            if (
                result.session_id != self.runtime_session_id
                or result.control_domain_id != self.control_domain_id
                or result.producer_id != self.producer_id
                or result.producer_epoch != self.producer_epoch
                or result.policy_node_id != self.policy_node_id
                or result.clock_domain_id != self.policy_clock_domain_id
                or result.artifact_hash != self.artifact_hash
                or result.action_codec_hash != self.action_codec_hash
                or result.frame_contract_hash != self.frame_contract_hash
                or result.layout_hash != self.layout_hash
                or result.normalization_hash != self.normalization_hash
            ):
                raise ValueError("RTC worker completion does not match source binding.")
        emission = step.emission_result.emission
        if emission is None:
            if step.emission_commit_transition is not None:
                raise ValueError("RTC step committed an emission that is not present.")
            return None
        pop_transition = step.pop_transition
        commit_transition = step.emission_commit_transition
        if pop_transition is None or not pop_transition.receipt.accepted:
            raise ValueError("RTC emission lacks an accepted lower-owned due-pop.")
        pop = pop_transition.pop_outcome
        if pop is None or pop.keyframe is None:
            raise ValueError("RTC emission due-pop selected no keyframe.")
        if commit_transition is None or not commit_transition.receipt.accepted:
            raise ValueError("RTC emission lacks accepted commit-through evidence.")
        if commit_transition.receipt.emission_id != emission.emission_id:
            raise ValueError("RTC emission/commit identity mismatch.")
        keyframe = pop.keyframe
        if (
            keyframe.inference_id != emission.inference_id
            or keyframe.chunk_id != emission.chunk_id
            or keyframe.chunk_keyframe_index != emission.chunk_keyframe_index
            or keyframe.target_effective_at != emission.target_effective_at
        ):
            raise ValueError("RTC due-pop keyframe/emission lineage mismatch.")
        return self.validate_emission(emission)


class RtcDecodedPolicyExecutor:
    """Run one local policy/decode call without any plan or queue authority."""

    def __init__(
        self,
        *,
        policy: Any,
        deploy_runtime: PolicyDeployRuntime,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.policy = policy
        self.deploy_runtime = deploy_runtime
        self.clock = clock

    def _time(self) -> float:
        return _finite(self.clock(), field_name="worker clock")

    def __call__(
        self,
        request: PolicyInferenceRequest,
        actor_observation: PolicyObservationView,
    ) -> PolicyChunkResult:
        started_at = max(self._time(), request.admitted_at)
        prediction_input: Any = actor_observation
        if self.deploy_runtime.config.observation_firewall_mode == "dual_validate":
            prediction_input = actor_observation.to_deployment_observation()
        policy_output = self.policy.predict(prediction_input)
        decoded = self.deploy_runtime.normalize_and_decode(
            policy_output,
            actor_observation.to_deployment_observation(),
        )
        execution_steps = self.deploy_runtime.config.action_chunk_execution_steps
        if execution_steps is not None:
            decoded = decoded[:execution_steps]
        decoded = decoded[: request.prediction_horizon_steps]
        if not decoded:
            raise ValueError("RTC policy returned an empty decoded chunk.")
        completed_at = max(self._time(), started_at)
        model_output_hash = canonical_content_hash(
            tuple(
                {
                    "vector": np.asarray(action.vector, dtype=np.float32),
                    "arm_joint_targets": {
                        str(side): np.asarray(value, dtype=np.float32)
                        for side, value in sorted(action.arm_joint_targets.items())
                    },
                }
                for action in decoded
            ),
            domain="dexteleop.rtc_3c.local_decoded_output.v1",
        )
        return PolicyChunkResult(
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
            policy_output_id=f"rtc-output:{request.request_id}",
            chunk_id=int(model_output_hash[:16], 16),
            inference_started_at=started_at,
            inference_completed_at=completed_at,
            result_received_at=completed_at,
            clock_domain_id=request.clock_domain_id,
            status="ready",
            source_chunk_size=len(decoded),
            decoded_absolute_actions=tuple(decoded),
            model_output_hash=model_output_hash,
            protocol_version=RTC_3C_LOCAL_POLICY_PROTOCOL_V1,
            effective_delay_steps=math.ceil(
                max(0.0, completed_at - request.admitted_at)
                / request.action_period_sec
                - 1e-12
            ),
            execution_horizon_steps=len(decoded),
        )


class RtcPolicyShadowFrontend:
    """Control-thread coordinator for a default-off RTC no-action shadow profile."""

    def __init__(
        self,
        *,
        config: RtcPolicyShadowConfig,
        deploy_runtime: PolicyDeployRuntime,
        policy: Any,
        preparer: Any | None = None,
        created_at: float,
        emission_adapter: DeployRobotActionAdapter | None = None,
        clock: Callable[[], float] = time.monotonic,
        executor_factory: Callable[[str], PolicyInferenceExecutor] | None = None,
        observation_observer: Callable[[DeploymentObservation], None] | None = None,
        request_observation_ready: Callable[[DeploymentObservation], bool] | None = None,
        representation_context_builder: Callable[
            [PolicyObservationView, str], PolicyActionRepresentationContext
        ]
        | None = None,
    ) -> None:
        if not isinstance(config, RtcPolicyShadowConfig):
            raise TypeError("config must be RtcPolicyShadowConfig.")
        if not isinstance(deploy_runtime, PolicyDeployRuntime):
            raise TypeError("deploy_runtime must be PolicyDeployRuntime.")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        legacy_adapter = (
            None if preparer is None else getattr(preparer, "action_adapter", None)
        )
        if emission_adapter is None:
            emission_adapter = legacy_adapter
        elif (
            isinstance(legacy_adapter, DeployRobotActionAdapter)
            and isinstance(emission_adapter, DeployRobotActionAdapter)
            and legacy_adapter.contract_hash != emission_adapter.contract_hash
        ):
            raise ValueError("preparer and emission_adapter contracts differ.")
        if emission_adapter is not None and not isinstance(
            emission_adapter,
            DeployRobotActionAdapter,
        ):
            raise TypeError(
                "emission_adapter must be DeployRobotActionAdapter or None."
            )
        self.config = config
        self.deploy_runtime = deploy_runtime
        self.policy = policy
        self.emission_adapter = emission_adapter
        self.clock = clock
        self._executor_factory = executor_factory
        if observation_observer is not None and not callable(observation_observer):
            raise TypeError("observation_observer must be callable or None.")
        if request_observation_ready is not None and not callable(request_observation_ready):
            raise TypeError("request_observation_ready must be callable or None.")
        self._observation_observer = observation_observer
        self._request_observation_ready = request_observation_ready
        if representation_context_builder is not None and not callable(
            representation_context_builder
        ):
            raise TypeError("representation_context_builder must be callable or None.")
        self._representation_context_builder = representation_context_builder
        if (
            config.model_continuation is not None
            and representation_context_builder is None
        ):
            raise ValueError(
                "Model-RTC requires a representation context builder."
            )
        self._worker_session_sequence = 0
        worker_session_id = self._worker_session_id(0)
        plan = empty_policy_action_plan(
            plan_id=f"rtc-plan:{config.runtime_session_id}",
            session_id=config.runtime_session_id,
            control_domain_id=config.control_domain_id,
            producer_id=config.producer_id,
            producer_epoch=1,
            policy_node_id=config.policy_node_id,
            clock_domain_id=config.clock_domain_id,
            cursor_at=created_at,
            created_at=created_at,
        )
        self._state = create_policy_async_producer(
            state_id=f"rtc-producer:{config.runtime_session_id}",
            plan=plan,
            worker_session_id=worker_session_id,
            created_at=created_at,
        )
        self._source_binding_candidate = (
            build_rtc_policy_emission_source_binding_from_contracts(
                config=config,
                deploy_runtime=deploy_runtime,
                emission_adapter=emission_adapter,
                producer_epoch=plan.producer_epoch,
            )
        )
        self._last_activation_readiness_receipt: (
            RtcProducerActivationReadinessReceipt | None
        ) = None
        self._worker = self._build_worker(worker_session_id)
        self._delay_state = InferenceDelayEstimatorState.empty(
            config.delay_estimator_spec
        )
        self._next_request_sequence = 0
        self._last_request_triggered_at: float | None = None
        self._selected_emission_count = 0
        self._late_completion_rejection_count = 0
        self._last_forecast_delay_steps: int | None = None
        self._last_request_validity_steps: int | None = None
        self._active_request_plan_coverage_steps: int | None = None
        self._preflight_request: PolicyInferenceRequest | None = None
        self._preflight_record_latency = False
        self._emitted_keyframe_ids: set[tuple[str, int, int]] = set()
        self._started = False
        self._closed = False

    @property
    def state(self) -> PolicyAsyncProducerSnapshot:
        return self._state

    @property
    def source_binding_candidate(self) -> RtcPolicyEmissionSourceBinding:
        """Current producer-owned candidate; activation belongs to the publisher."""

        return self._source_binding_candidate

    @property
    def last_activation_readiness_receipt(
        self,
    ) -> RtcProducerActivationReadinessReceipt | None:
        return self._last_activation_readiness_receipt

    @property
    def selected_emission_count(self) -> int:
        return self._selected_emission_count

    @property
    def late_completion_rejection_count(self) -> int:
        return self._late_completion_rejection_count

    @property
    def timing_contract(self) -> dict[str, Any]:
        return self.config.timing_contract

    @property
    def timing_contract_hash(self) -> str:
        return self.config.timing_contract_hash

    @property
    def last_forecast_delay_steps(self) -> int | None:
        return self._last_forecast_delay_steps

    @property
    def last_request_validity_steps(self) -> int | None:
        return self._last_request_validity_steps

    @property
    def pending_completion_count(self) -> int:
        """Expose bounded-mailbox occupancy without leaking worker ownership."""

        return self._worker.pending_completion_count

    def _latency_budget_steps(
        self,
        delay_state: InferenceDelayEstimatorState | None = None,
    ) -> int:
        """Return the unclamped measured latency budget for request timing."""

        spec = self.config.delay_estimator_spec
        state = self._delay_state if delay_state is None else delay_state
        samples = state.latency_samples_sec
        latency_sec = max(samples) if samples else spec.warmup_latency_sec
        return rtc_latency_budget_steps(
            latency_sec,
            action_period_sec=spec.action_period_sec,
            network_guard_sec=spec.network_guard_sec,
            control_jitter_guard_steps=spec.control_jitter_guard_steps,
        )

    @property
    def effective_request_trigger_coverage_steps(self) -> int:
        """Start the next conditioned request early enough to retain a bridge."""

        configured = self.config.request_trigger_coverage_steps
        if self.config.model_continuation is None:
            return configured
        return max(
            configured,
            self._latency_budget_steps()
            + RTC_REQUEST_BRIDGE_GUARD_STEPS,
        )

    def _assess_execution_coverage(
        self,
        latency_sec: float,
        *,
        model_forecast_delay_steps: int,
        request_plan_coverage_steps: int | None,
    ) -> RtcExecutionCoverageAssessment:
        spec = self.config.delay_estimator_spec
        return assess_rtc_execution_coverage(
            latency_sec,
            action_period_sec=spec.action_period_sec,
            prediction_horizon_steps=self.config.prediction_horizon_steps,
            model_forecast_delay_steps=model_forecast_delay_steps,
            configured_trigger_coverage_steps=(
                self.config.request_trigger_coverage_steps
            ),
            minimum_execution_horizon_steps=(
                self.config.minimum_execution_horizon_steps
            ),
            network_guard_sec=spec.network_guard_sec,
            control_jitter_guard_steps=spec.control_jitter_guard_steps,
            bridge_guard_steps=RTC_REQUEST_BRIDGE_GUARD_STEPS,
            request_plan_coverage_steps=request_plan_coverage_steps,
        )

    def assess_latency_envelope(
        self,
        latency_sec: float,
        *,
        performance_slo_sec: float,
    ) -> RtcLatencyEnvelopeAssessment:
        """Assess one measured latency against plan coverage and the SLO."""

        spec = self.config.delay_estimator_spec
        forecast_steps = self._last_forecast_delay_steps
        if forecast_steps is None:
            raise RuntimeError(
                "RTC latency envelope has no request-bound forecast delay."
            )
        return assess_rtc_latency_envelope(
            latency_sec,
            performance_slo_sec=performance_slo_sec,
            action_period_sec=spec.action_period_sec,
            prediction_horizon_steps=self.config.prediction_horizon_steps,
            model_forecast_delay_steps=forecast_steps,
            configured_trigger_coverage_steps=(
                self.config.request_trigger_coverage_steps
            ),
            minimum_execution_horizon_steps=(
                self.config.minimum_execution_horizon_steps
            ),
            network_guard_sec=spec.network_guard_sec,
            control_jitter_guard_steps=spec.control_jitter_guard_steps,
            bridge_guard_steps=RTC_REQUEST_BRIDGE_GUARD_STEPS,
        )

    def _worker_session_id(self, sequence: int) -> str:
        return (
            f"rtc-worker-session:{self.config.runtime_session_id}:"
            f"{sequence}"
        )

    def _build_worker(self, worker_session_id: str) -> BoundedPolicyInferenceWorker:
        if self._executor_factory is None:
            executor: PolicyInferenceExecutor = RtcDecodedPolicyExecutor(
                policy=self.policy,
                deploy_runtime=self.deploy_runtime,
                clock=self.clock,
            )
        else:
            executor = self._executor_factory(worker_session_id)
        return BoundedPolicyInferenceWorker(
            worker_id=f"rtc-worker:{self.config.runtime_session_id}",
            worker_session_id=worker_session_id,
            executor=executor,
            result_capacity=self.config.worker_result_capacity,
            clock=self.clock,
            idle_poll_sec=self.config.worker_idle_poll_sec,
        )

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("RTC frontend is closed.")
        if self._started:
            raise RuntimeError("RTC frontend can only be started once per session.")
        self._worker.start()
        self._started = True

    def _require_running(self) -> None:
        if not self._started or self._closed:
            raise RuntimeError("RTC frontend is not running.")

    def _observation_cut_id(
        self,
        actor: PolicyObservationView,
        *,
        request_sequence: int,
    ) -> str:
        """Return the small causal identity of one RTC request observation.

        Exact model-input content is attested independently by the worker-side
        ``model_input_hash``.  Camera/tactile payload bytes must therefore not
        be scanned on the control thread to create this identity.
        """

        digest = canonical_content_hash(
            {
                "runtime_session_id": self.config.runtime_session_id,
                "control_domain_id": self._state.plan.control_domain_id,
                "producer_id": self._state.plan.producer_id,
                "producer_epoch": self._state.plan.producer_epoch,
                "policy_node_id": self._state.plan.policy_node_id,
                "request_sequence": _nonnegative_integer(
                    request_sequence,
                    field_name="request_sequence",
                ),
                "observation_contract_hash": actor.contract_hash,
                "observation_timestamp": actor.payload["timestamp"],
                "observation_step_index": actor.payload["step_index"],
            },
            domain=RTC_OBSERVATION_CUT_IDENTITY_HASH_DOMAIN_V2,
        )
        return f"rtc-observation-cut:v2:{digest}"

    def _coverage_steps(self, now: float) -> int:
        valid = tuple(
            keyframe
            for keyframe in self._state.plan.keyframes
            if keyframe.effective_until > now and keyframe.valid_until > now
        )
        if not valid:
            return 0
        period = self.config.producer_profile.action_period_sec
        return max(0, math.ceil((valid[-1].effective_until - now) / period))

    def _should_request(self, now: float) -> bool:
        observation_due = (
            self._last_request_triggered_at is None
            or now - self._last_request_triggered_at
            >= 1.0 / self.config.producer_profile.observation_hz - 1e-12
        )
        return (
            self._state.lifecycle_state == POLICY_ASYNC_PRODUCER_ELIGIBLE
            and self._state.worker_session_state == POLICY_WORKER_SESSION_ACTIVE
            and self._state.active_request is None
            and observation_due
            and self._coverage_steps(now)
            <= self.effective_request_trigger_coverage_steps
        )

    def _build_request_and_conditioning(
        self,
        actor: PolicyObservationView,
        *,
        now: float,
    ) -> tuple[
        PolicyInferenceRequest,
        CanonicalRtcConditioningBundle | None,
    ]:
        sequence = self._next_request_sequence
        self._next_request_sequence += 1
        forecast_steps = self._delay_state.forecast_steps(
            self.config.delay_estimator_spec
        )
        producer_profile = self.config.producer_profile
        period = producer_profile.action_period_sec
        generation_start = max(now, self._state.plan.cursor_at)
        continuation = self.config.model_continuation
        if continuation is not None and self._state.plan.keyframes:
            tolerance = continuation.lattice_tolerance_sec
            eligible_targets = tuple(
                keyframe.target_effective_at
                for keyframe in self._state.plan.keyframes
                if keyframe.target_effective_at
                >= generation_start - tolerance
            )
            if eligible_targets:
                generation_start = eligible_targets[0]
        first_target = generation_start + forecast_steps * period
        request_validity_steps = self.config.effective_request_validity_steps(
            forecast_steps,
        )
        self._last_forecast_delay_steps = forecast_steps
        self._last_request_validity_steps = request_validity_steps
        request_id = f"rtc-request:{self.config.runtime_session_id}:{sequence}"
        observation_cut_id = self._observation_cut_id(
            actor,
            request_sequence=sequence,
        )
        representation_context = (
            None
            if self._representation_context_builder is None
            else self._representation_context_builder(actor, observation_cut_id)
        )
        if representation_context is not None:
            if not isinstance(
                representation_context,
                PolicyActionRepresentationContext,
            ):
                raise TypeError(
                    "representation_context_builder must return "
                    "PolicyActionRepresentationContext."
                )
            if representation_context.observation_cut_id != observation_cut_id:
                raise ValueError(
                    "Representation context observation cut differs from "
                    "the request observation."
                )
            if representation_context.observation_contract_hash != actor.contract_hash:
                raise ValueError(
                    "Representation context observation contract differs from "
                    "the request observation."
                )
            if representation_context.observation_timestamp != float(
                actor.payload["timestamp"]
            ):
                raise ValueError(
                    "Representation context timestamp differs from "
                    "the request observation."
                )
            if representation_context.observation_step_index != int(
                actor.payload["step_index"]
            ):
                raise ValueError(
                    "Representation context step index differs from "
                    "the request observation."
                )
            if self.config.action_pose_mode is not None and (
                representation_context.source_pose_mode
                != self.config.action_pose_mode
            ):
                raise ValueError(
                    "Representation context pose mode differs from RTC config."
                )
        if (
            self.config.action_pose_mode == "relative"
            and representation_context is None
        ):
            raise ValueError(
                "Relative RTC request requires a representation context builder."
            )
        conditioning_bundle = None
        if continuation is not None:
            if representation_context is None:
                raise ValueError(
                    "Model-RTC request requires representation context."
                )
            timeline = RtcContinuationTimeline(
                clock_domain_id=self._state.plan.clock_domain_id,
                generation_start_at=generation_start,
                first_target_effective_at=first_target,
                action_period_sec=period,
                prediction_horizon_steps=self.config.prediction_horizon_steps,
                forecast_delay_steps=forecast_steps,
            )
            selection_spec = continuation.selection_spec_for_timeline(
                timeline
            )
            selection = select_rtc_plan_continuation(
                self._state.plan,
                timeline=timeline,
                spec=selection_spec,
            )
            conditioning_bundle = build_canonical_rtc_conditioning_bundle(
                runtime_session_id=self._state.plan.session_id,
                control_domain_id=self._state.plan.control_domain_id,
                artifact_hash=self.config.artifact_hash,
                timeline=timeline,
                selection=selection,
                action_representation_context=representation_context,
                continuation_contract=(
                    continuation.continuation_contract
                ),
                prefix_schedule_semantics=(
                    continuation.prefix_schedule_semantics
                ),
            )
        request = PolicyInferenceRequest(
            request_id=request_id,
            inference_id=f"{self.config.policy_node_id}:rtc-inference:{sequence}",
            rtc_mode=producer_profile.rtc_mode,
            session_id=self._state.plan.session_id,
            control_domain_id=self._state.plan.control_domain_id,
            producer_id=self._state.plan.producer_id,
            producer_epoch=self._state.plan.producer_epoch,
            policy_node_id=self._state.plan.policy_node_id,
            artifact_hash=self.config.artifact_hash,
            observation_cut_id=observation_cut_id,
            observation_timestamp=float(actor.payload["timestamp"]),
            observation_step_index=int(actor.payload["step_index"]),
            triggered_at=now,
            admitted_at=now,
            first_target_effective_at=first_target,
            clock_domain_id=self._state.plan.clock_domain_id,
            action_period_sec=period,
            prediction_horizon_steps=self.config.prediction_horizon_steps,
            minimum_execution_horizon_steps=(
                self.config.minimum_execution_horizon_steps
            ),
            source_plan_id=self._state.plan.plan_id,
            source_plan_generation=self._state.plan.generation,
            source_plan_hash=self._state.plan.semantic_content_hash,
            source_plan_cursor_at=self._state.plan.cursor_at,
            producer_plan_committed_through=(
                self._state.plan.producer_committed_through
            ),
            forecast_delay_steps=forecast_steps,
            delay_estimator_spec_hash=(
                self.config.delay_estimator_spec.semantic_content_hash
            ),
            action_codec_hash=self.config.action_codec_hash,
            frame_contract_hash=self.config.frame_contract_hash,
            layout_hash=self.config.layout_hash,
            normalization_hash=self.config.normalization_hash,
            valid_until=first_target + request_validity_steps * period,
            action_representation_context=representation_context,
            conditioning_binding_hash=(
                None
                if conditioning_bundle is None
                else conditioning_bundle.conditioning_binding_hash
            ),
        )
        return request, conditioning_bundle

    def _build_request(
        self,
        actor: PolicyObservationView,
        *,
        now: float,
    ) -> PolicyInferenceRequest:
        request, _conditioning = self._build_request_and_conditioning(
            actor,
            now=now,
        )
        return request

    def _refresh_worker_session_state(self, *, now: float) -> None:
        worker_snapshot = self._worker.session_snapshot()
        if worker_snapshot.session_state == self._state.worker_session_state:
            return
        if self._state.active_request is not None:
            return
        transition = update_policy_worker_session_state(
            self._state,
            worker_session_state=worker_snapshot.session_state,
            occurred_at=now,
            reason="worker_session_observed",
        )
        self._state = transition.state

    def transition_to(
        self,
        lifecycle_state: str,
        *,
        now: float,
        reason: str,
    ) -> RtcFrontendLifecycleTransition:
        self._require_running()
        occurred = _finite(now, field_name="now")
        transition = invalidate_policy_async_producer(
            self._state,
            lifecycle_state=lifecycle_state,
            invalidated_at=occurred,
            reason=reason,
            worker_session_state=(
                POLICY_WORKER_SESSION_RETIRED_WAITING_FOR_COMPLETION
            ),
        )
        self._state = transition.state
        self._preflight_request = None
        self._active_request_plan_coverage_steps = None
        self.deploy_runtime.reset_arm_hold_semantics()
        # A lifecycle transition is a control-path invalidation, not a worker
        # shutdown barrier.  Signal cancellation and return immediately so the
        # application can establish its backend-owned Hold without waiting for
        # a blocked provider call.  Rearm already requires the retired session
        # to reach DRAINED, while close() retains the configured bounded wait.
        stop = self._worker.stop(timeout_sec=0.0)
        session = update_policy_worker_session_state(
            self._state,
            worker_session_state=stop.session_state,
            occurred_at=max(occurred, stop.returned_at),
            reason=f"worker_stop:{stop.status}",
        )
        self._state = session.state
        return RtcFrontendLifecycleTransition(
            producer_transition=transition,
            worker_stop_receipt=stop,
        )

    def hold(self, *, now: float, reason: str) -> RtcFrontendLifecycleTransition:
        return self.transition_to(
            POLICY_ASYNC_PRODUCER_HOLD,
            now=now,
            reason=reason,
        )

    def fault(self, *, now: float, reason: str) -> RtcFrontendLifecycleTransition:
        return self.transition_to(
            POLICY_ASYNC_PRODUCER_FAULT,
            now=now,
            reason=reason,
        )

    def rearm(self, *, now: float, reason: str) -> PolicyAsyncProducerTransitionOutcome:
        self._require_running()
        occurred = _finite(now, field_name="now")
        old_epoch = self._source_binding_candidate.producer_epoch
        retired_completion = self._worker.poll_completion()
        if retired_completion is not None:
            rejected = admit_policy_chunk_result(
                self._state,
                retired_completion.result,
                worker_session_id=retired_completion.worker_session_id,
                admitted_at=occurred,
                actual_cursor_at=occurred,
            )
            self._state = rejected.state
            if rejected.receipt.accepted:
                raise RuntimeError("Retired RTC completion was unexpectedly accepted.")
            self._late_completion_rejection_count += 1
        self._refresh_worker_session_state(now=occurred)
        if self._state.worker_session_state != POLICY_WORKER_SESSION_DRAINED:
            raise RuntimeError("RTC worker session has not drained; rearm is forbidden.")
        self._worker_session_sequence += 1
        worker_session_id = self._worker_session_id(
            self._worker_session_sequence
        )
        transition = rearm_policy_async_producer(
            self._state,
            worker_session_id=worker_session_id,
            rearmed_at=occurred,
            reason=reason,
        )
        self._worker = self._build_worker(worker_session_id)
        self._worker.start()
        self._state = transition.state
        self._source_binding_candidate = (
            build_rtc_policy_emission_source_binding_from_contracts(
                config=self.config,
                deploy_runtime=self.deploy_runtime,
                emission_adapter=self.emission_adapter,
                producer_epoch=self._state.plan.producer_epoch,
            )
        )
        readiness_payload = {
            "ready": True,
            "readiness_kind": "same_thread_worker_drained_v1",
            "producer_id": self._state.plan.producer_id,
            "old_producer_epoch": old_epoch,
            "new_producer_epoch": self._state.plan.producer_epoch,
            "occurred_at": occurred,
            "evidence_hashes": (transition.receipt.semantic_content_hash,),
            "reasons": (),
        }
        self._last_activation_readiness_receipt = (
            RtcProducerActivationReadinessReceipt(
                **readiness_payload,
                semantic_content_hash=canonical_content_hash(
                    readiness_payload,
                    domain=RTC_PRODUCER_ACTIVATION_READINESS_HASH_DOMAIN_V1,
                ),
            )
        )
        self._last_request_triggered_at = None
        self._last_forecast_delay_steps = None
        self._last_request_validity_steps = None
        self._active_request_plan_coverage_steps = None
        return transition

    def step(
        self,
        observation: DeploymentObservation,
        *,
        now: float,
        allow_output: bool = True,
        allow_request: bool | None = None,
        preflight_timeout_sec: float | None = None,
        preflight_record_latency: bool = True,
    ) -> RtcFrontendStep:
        self._require_running()
        if not isinstance(observation, DeploymentObservation):
            raise TypeError("observation must be DeploymentObservation.")
        if self._observation_observer is not None:
            self._observation_observer(observation)
        observation_ready = (
            True
            if self._request_observation_ready is None
            else self._request_observation_ready(observation)
        )
        if not isinstance(observation_ready, bool):
            raise TypeError("request_observation_ready must return bool.")
        tick_started = time.perf_counter()
        decision_at = _finite(now, field_name="now")
        preflight = preflight_timeout_sec is not None
        if preflight:
            timeout = _finite(preflight_timeout_sec, field_name="preflight_timeout_sec")
            if timeout <= 0.0:
                raise ValueError("preflight_timeout_sec must be positive.")
            if allow_output or self._state.active_request is not None or self._state.plan.keyframes:
                raise ValueError("RTC preflight cannot share an executable request or plan.")
        elif allow_output and self._preflight_request is not None:
            raise ValueError("RTC preflight must finish before executable output is enabled.")
        request_allowed = (
            bool(allow_output)
            if allow_request is None
            else bool(allow_request)
        )
        completion = self._worker.poll_completion()
        result_transition = None
        lifecycle_transition = None
        preflight_completion = False
        completion_coverage_assessment = None
        pending_preflight = self._preflight_request
        if pending_preflight is not None and completion is not None:
            # Preflight uses the model protocol, never producer admission/merge.
            # Its longer RPC deadline cannot extend any executable action TTL.
            self._preflight_request = None
            result = completion.result
            if (
                completion.worker_session_id != self._state.worker_session_id
                or result.request_hash != pending_preflight.semantic_content_hash
                or result.request_id != pending_preflight.request_id
                or result.inference_id != pending_preflight.inference_id
                or result.status == "fault"
                or decision_at >= pending_preflight.valid_until
                or completion.released_at >= pending_preflight.valid_until
            ):
                lifecycle_transition = self.fault(
                    now=decision_at,
                    reason=result.fault_code or "rtc_preflight_result_invalid_or_expired",
                )
            else:
                preflight_completion = True
                if self._preflight_record_latency:
                    self._delay_state = self._delay_state.record(
                        self.config.delay_estimator_spec,
                        max(0.0, completion.released_at - completion.worker_started_at),
                    )
        elif pending_preflight is not None and decision_at >= pending_preflight.valid_until:
            lifecycle_transition = self.fault(now=decision_at, reason="rtc_preflight_timeout")
        elif completion is not None:
            lifecycle_before_result = self._state.lifecycle_state
            completion_request = self._state.active_request
            completion_request_plan_coverage_steps = (
                self._active_request_plan_coverage_steps
            )
            result_transition = admit_policy_chunk_result(
                self._state,
                completion.result,
                worker_session_id=completion.worker_session_id,
                admitted_at=decision_at,
                actual_cursor_at=decision_at,
            )
            self._state = result_transition.state
            if (
                lifecycle_before_result != POLICY_ASYNC_PRODUCER_ELIGIBLE
                and not result_transition.receipt.accepted
            ):
                self._late_completion_rejection_count += 1
            latency = max(
                0.0,
                completion.released_at - completion.worker_started_at,
            )
            next_delay_state = self._delay_state.record(
                self.config.delay_estimator_spec,
                latency,
            )
            self._delay_state = next_delay_state
            if (
                self.config.model_continuation is not None
                and result_transition.receipt.accepted
            ):
                if completion_request is None:
                    raise RuntimeError(
                        "Accepted conditioned RTC completion has no active request."
                    )
                completion_coverage_assessment = (
                    self._assess_execution_coverage(
                        latency,
                        model_forecast_delay_steps=(
                            completion_request.forecast_delay_steps
                        ),
                        request_plan_coverage_steps=(
                            completion_request_plan_coverage_steps
                        ),
                    )
                )
            merge_receipt = result_transition.merge_receipt
            merge_reasons = (
                () if merge_receipt is None else merge_receipt.reasons
            )
            # A result that arrives after its request deadline must invalidate
            # the producer before the pop phase below.  Without this state
            # transition, already-committed keyframes could still be emitted
            # on the same control tick even though the corresponding model
            # result was late.  The committed prefix is not reused: the
            # normal fault/rearm path retires this producer session and the
            # operator can press ``g`` to start a fresh epoch.
            if (
                "request_deadline_expired" in merge_reasons
                and self._state.lifecycle_state == POLICY_ASYNC_PRODUCER_ELIGIBLE
            ):
                lifecycle_transition = self.fault(
                    now=decision_at,
                    reason="rtc_result_timeout",
                )
            elif completion.result.status == "fault" and merge_receipt is not None:
                lifecycle_transition = self.fault(
                    now=decision_at,
                    reason=completion.result.fault_code or "policy_worker_fault",
                )
            elif (
                completion_coverage_assessment is not None
                and not completion_coverage_assessment.execution_usable
                and self._state.lifecycle_state == POLICY_ASYNC_PRODUCER_ELIGIBLE
            ):
                # The result may have passed identity/TTL admission, but its
                # post-prefix, post-cursor suffix cannot bridge another request.
                # Invalidate the just-merged plan before the pop phase so no
                # action from this completion can be emitted on the same tick.
                lifecycle_transition = self.hold(
                    now=decision_at,
                    reason="rtc_latency_starvation_risk",
                )
            self._active_request_plan_coverage_steps = None
        self._refresh_worker_session_state(now=decision_at)

        pop_transition = None
        emission_commit_transition = None
        emission_commit_receipt = None
        emission_result = PolicyActionEmissionResult()
        underrun = False
        if (
            allow_output
            and self._state.lifecycle_state == POLICY_ASYNC_PRODUCER_ELIGIBLE
        ):
            before = len(self._state.plan.keyframes)
            pop_transition = pop_due_policy_async_producer_action(
                self._state,
                decision_at=decision_at,
            )
            self._state = pop_transition.state
            pop = pop_transition.pop_outcome
            assert pop is not None
            if pop.keyframe is None:
                underrun = not self._state.plan.keyframes
            else:
                keyframe_identity = (
                    pop.keyframe.inference_id,
                    pop.keyframe.chunk_id,
                    pop.keyframe.chunk_keyframe_index,
                )
                if keyframe_identity in self._emitted_keyframe_ids:
                    raise RuntimeError("RTC keyframe was selected more than once.")
                self._emitted_keyframe_ids.add(keyframe_identity)
                self._selected_emission_count += 1
                emission_result = (
                    self.deploy_runtime.emit_timed_action_keyframe(
                        pop.keyframe,
                        observation,
                        emission_adapter=self.emission_adapter,
                        policy_name=str(getattr(self.policy, "name", "rtc_policy")),
                        plan_keyframe_count_before=before,
                        plan_keyframe_count_after=len(self._state.plan.keyframes),
                        plan_pop_count=self._selected_emission_count,
                        source_binding_hash=(
                            self._source_binding_candidate.semantic_content_hash
                        ),
                    )
                )
                if (
                    emission_result.emission is None
                    or emission_result.emission_error is not None
                ):
                    lifecycle_transition = self.fault(
                        now=decision_at,
                        reason="formal_emission_failed",
                    )
                    emission_result = PolicyActionEmissionResult()
                else:
                    emission_commit_transition = (
                        commit_emitted_policy_async_producer_action(
                            self._state,
                            pop_outcome=pop,
                            emission_id=emission_result.emission.emission_id,
                            committed_at=decision_at,
                        )
                    )
                    self._state = emission_commit_transition.state
                    if not emission_commit_transition.receipt.accepted:
                        lifecycle_transition = self.fault(
                            now=decision_at,
                            reason="emission_commit_rejected",
                        )
                        emission_result = PolicyActionEmissionResult()
                    else:
                        emission = emission_result.emission
                        assert emission is not None
                        staged_arm_hold = (
                            emission_result.arm_hold_staged_resolution
                        )
                        if staged_arm_hold is not None:
                            try:
                                self.deploy_runtime.commit_staged_arm_hold(
                                    staged_arm_hold
                                )
                            except Exception as exc:  # noqa: BLE001 - invariant failure faults.
                                lifecycle_transition = self.fault(
                                    now=decision_at,
                                    reason=(
                                        "arm_hold_semantic_commit_failed:"
                                        f"{type(exc).__name__}:{exc}"
                                    ),
                                )
                                emission_result = PolicyActionEmissionResult()
                                continue_commit = False
                            else:
                                continue_commit = True
                        else:
                            continue_commit = True
                        if not continue_commit:
                            emission = None
                        else:
                            emission = emission_result.emission
                            assert emission is not None
                    if emission_result.emission is not None:
                        emission = emission_result.emission
                        assert emission is not None
                        commit_payload = {
                            "accepted": True,
                            "emission_id": emission.emission_id,
                            "emission_semantic_content_hash": (
                                emission.semantic_content_hash
                            ),
                            "source_binding_hash": (
                                self._source_binding_candidate.semantic_content_hash
                            ),
                            "producer_id": self._state.plan.producer_id,
                            "producer_epoch": self._state.plan.producer_epoch,
                            "committed_at": decision_at,
                            "producer_transition_receipt_hash": (
                                emission_commit_transition.receipt.semantic_content_hash
                            ),
                        }
                        emission_commit_receipt = (
                            PolicyActionEmissionCommitReceipt(
                                **commit_payload,
                                semantic_content_hash=canonical_content_hash(
                                    commit_payload,
                                    domain=(
                                        POLICY_ACTION_EMISSION_COMMIT_RECEIPT_HASH_DOMAIN_V1
                                    ),
                                ),
                            )
                        )

        request_transition = None
        worker_submission = None
        if (
            request_allowed
            and observation_ready
            and self._preflight_request is None
            and not (preflight and completion is not None)
            and self._should_request(decision_at)
        ):
            request_plan_coverage_steps = self._coverage_steps(decision_at)
            request_uses_bootstrap_gap = (
                self._selected_emission_count == 0
                and not self._state.plan.keyframes
            )
            policy_observation = self.deploy_runtime.prepare_policy_observation(
                observation
            )
            actor, _parity = self.deploy_runtime.prepare_actor_observation(
                policy_observation
            )
            try:
                request, conditioning_bundle = (
                    self._build_request_and_conditioning(
                        actor,
                        now=decision_at,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - invalid context must fail closed.
                lifecycle_transition = self.fault(
                    now=decision_at,
                    reason=(
                        "inference_request_invalid:"
                        f"{type(exc).__name__}:{exc}"
                    ),
                )
            else:
                if preflight:
                    request = replace(request, valid_until=decision_at + timeout)
                    self._preflight_request = request
                    self._preflight_record_latency = bool(preflight_record_latency)
                    self._active_request_plan_coverage_steps = None
                else:
                    request_transition = admit_policy_inference_request(self._state, request)
                    self._state = request_transition.state
                    if request_transition.receipt.accepted:
                        self._active_request_plan_coverage_steps = (
                            None
                            if request_uses_bootstrap_gap
                            else request_plan_coverage_steps
                        )
                if preflight or request_transition.receipt.accepted:
                    self._last_request_triggered_at = decision_at
                    work_item = PolicyInferenceWorkItem(
                        worker_id=self._worker.worker_id,
                        worker_session_id=self._worker.worker_session_id,
                        submit_sequence=self._next_request_sequence - 1,
                        request=request,
                        actor_observation=actor,
                        submitted_at=decision_at,
                        conditioning_bundle=conditioning_bundle,
                    )
                    worker_submission = self._worker.submit(work_item)
                    if not worker_submission.accepted:
                        lifecycle_transition = self.fault(
                            now=decision_at,
                            reason=f"worker_submit:{worker_submission.status}",
                        )

        duration = max(0.0, time.perf_counter() - tick_started)
        return RtcFrontendStep(
            state=self._state,
            completion=completion,
            result_transition=result_transition,
            pop_transition=pop_transition,
            emission_commit_transition=emission_commit_transition,
            emission_commit_receipt=emission_commit_receipt,
            request_transition=request_transition,
            worker_submission=worker_submission,
            emission_result=emission_result,
            lifecycle_transition=lifecycle_transition,
            underrun=underrun,
            control_duration_sec=duration,
            control_budget_sec=1.0 / self.config.producer_profile.decision_hz,
            selected_emission_count=self._selected_emission_count,
            late_completion_rejection_count=(
                self._late_completion_rejection_count
            ),
            forecast_delay_steps=self._last_forecast_delay_steps,
            request_validity_steps=self._last_request_validity_steps,
            effective_request_trigger_coverage_steps=(
                self.effective_request_trigger_coverage_steps
            ),
            plan_coverage_steps=self._coverage_steps(decision_at),
            completion_coverage_assessment=(
                completion_coverage_assessment
            ),
            timing_contract_hash=self.config.timing_contract_hash,
            preflight_completion=preflight_completion,
        )

    def close(self, *, now: float | None = None) -> PolicyWorkerStopReceipt:
        if self._closed:
            return self._worker.stop(
                timeout_sec=self.config.effective_worker_stop_timeout_sec
            )
        occurred = self.clock() if now is None else now
        occurred = _finite(occurred, field_name="now")
        if self._started and (
            self._state.lifecycle_state == POLICY_ASYNC_PRODUCER_ELIGIBLE
        ):
            self.transition_to(
                POLICY_ASYNC_PRODUCER_TRANSITIONING,
                now=occurred,
                reason="frontend_close",
            )
        stop = self._worker.stop(
            timeout_sec=self.config.effective_worker_stop_timeout_sec
        )
        self._closed = True
        return stop


def build_rtc_policy_emission_source_binding(
    frontend: RtcPolicyShadowFrontend,
) -> RtcPolicyEmissionSourceBinding:
    """Capture the current RTC producer epoch without starting or stepping it."""

    if not isinstance(frontend, RtcPolicyShadowFrontend):
        raise TypeError("frontend must be an RtcPolicyShadowFrontend.")
    if frontend.emission_adapter is not None and not isinstance(
        frontend.emission_adapter,
        DeployRobotActionAdapter,
    ):
        raise TypeError(
            "RTC frontend emission_adapter must be DeployRobotActionAdapter or None."
        )
    config = frontend.config
    plan = frontend.state.plan
    if (
        plan.session_id != config.runtime_session_id
        or plan.control_domain_id != config.control_domain_id
        or plan.producer_id != config.producer_id
        or plan.policy_node_id != config.policy_node_id
        or plan.clock_domain_id != config.clock_domain_id
    ):
        raise ValueError("RTC frontend plan/config identity differs.")
    binding = frontend.source_binding_candidate
    if (
        binding.runtime_session_id != plan.session_id
        or binding.control_domain_id != plan.control_domain_id
        or binding.producer_id != plan.producer_id
        or binding.producer_epoch != plan.producer_epoch
    ):
        raise RuntimeError("RTC cached source binding is stale.")
    return binding


def build_rtc_policy_emission_source_binding_from_contracts(
    *,
    config: RtcPolicyShadowConfig,
    deploy_runtime: PolicyDeployRuntime,
    emission_adapter: DeployRobotActionAdapter | None,
    producer_epoch: int,
) -> RtcPolicyEmissionSourceBinding:
    """Build the same source identity before a concrete frontend is supplied."""

    if not isinstance(config, RtcPolicyShadowConfig):
        raise TypeError("config must be an RtcPolicyShadowConfig.")
    if not isinstance(deploy_runtime, PolicyDeployRuntime):
        raise TypeError("deploy_runtime must be a PolicyDeployRuntime.")
    if emission_adapter is not None and not isinstance(
        emission_adapter,
        DeployRobotActionAdapter,
    ):
        raise TypeError(
            "emission_adapter must be a DeployRobotActionAdapter or None."
        )
    epoch = _nonnegative_integer(producer_epoch, field_name="producer_epoch")
    runtime_config = deploy_runtime.config
    if (
        runtime_config.policy_node_id != config.policy_node_id
        or runtime_config.policy_clock_domain_id != config.clock_domain_id
    ):
        raise ValueError("RTC frontend/runtime policy identity differs.")
    if emission_adapter is not None and (
        tuple(emission_adapter.active_sides) != tuple(runtime_config.active_sides)
        or emission_adapter.arms_enabled != runtime_config.arms_enabled
        or emission_adapter.hands_enabled != runtime_config.hands_enabled
    ):
        raise ValueError("RTC frontend action adapter/runtime coverage differs.")
    observation_contract = runtime_config.policy_observation_contract
    if observation_contract is None:
        raise RuntimeError(
            "PolicyDeployRuntimeConfig did not normalize "
            "policy_observation_contract."
        )
    arm_hold_output = runtime_config.arm_hold_output
    if arm_hold_output is not None and not isinstance(
        arm_hold_output,
        PolicyArmHoldOutputContract,
    ):
        raise RuntimeError(
            "PolicyDeployRuntimeConfig did not normalize arm_hold_output."
        )
    runtime_payload = {
        "active_sides": runtime_config.active_sides,
        "arms_enabled": runtime_config.arms_enabled,
        "hands_enabled": runtime_config.hands_enabled,
        "policy_action_pose_mode": runtime_config.policy_action_pose_mode,
        "policy_absolute_frame": runtime_config.policy_absolute_frame,
        "policy_node_id": runtime_config.policy_node_id,
        "policy_clock_domain_id": runtime_config.policy_clock_domain_id,
        "action_keyframe_period_sec": runtime_config.action_keyframe_period_sec,
        "action_maximum_hold_sec": runtime_config.action_maximum_hold_sec,
        "action_emission_admission_mode": (
            runtime_config.action_emission_admission_mode
        ),
        "action_chunk_execution_steps": runtime_config.action_chunk_execution_steps,
        "policy_observation_contract_hash": (
            observation_contract.semantic_content_hash
        ),
    }
    if arm_hold_output is not None:
        runtime_payload["arm_hold_output_contract_hash"] = (
            arm_hold_output.contract_hash
        )
    return RtcPolicyEmissionSourceBinding(
        source_slot_id=f"rtc-policy-source:{config.producer_id}",
        runtime_session_id=config.runtime_session_id,
        control_domain_id=config.control_domain_id,
        producer_id=config.producer_id,
        producer_epoch=epoch,
        policy_node_id=config.policy_node_id,
        policy_clock_domain_id=config.clock_domain_id,
        policy_runtime_config_hash=canonical_content_hash(
            runtime_payload,
            domain=RTC_POLICY_RUNTIME_SOURCE_CONFIG_HASH_DOMAIN_V1,
        ),
        action_adapter_contract_hash=(
            RTC_TYPED_CANONICAL_SOURCE_ADAPTER_CONTRACT_HASH_V1
            if emission_adapter is None
            else emission_adapter.contract_hash
        ),
        rtc_frontend_config_hash=config.semantic_content_hash,
        rtc_profile_hash=config.profile.semantic_content_hash,
        artifact_hash=config.artifact_hash,
        action_codec_hash=config.action_codec_hash,
        frame_contract_hash=config.frame_contract_hash,
        layout_hash=config.layout_hash,
        normalization_hash=config.normalization_hash,
    )


# Canonical functional names.  Existing Shadow/Frontend names remain import
# aliases and retain the same class object and serialized contracts.
RtcPolicyProducerConfig = RtcPolicyShadowConfig
RtcProducerTickResult = RtcFrontendStep
RtcPolicyProducerRuntime = RtcPolicyShadowFrontend


__all__ = [
    "POLICY_ACTION_EMISSION_COMMIT_RECEIPT_HASH_DOMAIN_V1",
    "RTC_3C_FRONTEND_CONFIG_HASH_DOMAIN_V1",
    "RTC_3C_FRONTEND_CONFIG_HASH_DOMAIN_V2",
    "RTC_3C_FRONTEND_CONFIG_HASH_DOMAIN_V3",
    "RTC_3C_FRONTEND_CONFIG_HASH_DOMAIN_V4",
    "RTC_3C_LOCAL_POLICY_PROTOCOL_V1",
    "RTC_MODEL_WORKER_STOP_DEFAULT_TIMEOUT_SEC",
    "RTC_MODEL_WORKER_STOP_DISABLED_POLICY_V1",
    "RTC_MODEL_WORKER_STOP_POLICY_V1",
    "RTC_POLICY_EMISSION_SOURCE_BINDING_HASH_DOMAIN_V1",
    "RTC_POLICY_EMISSION_SOURCE_DESCRIPTOR_HASH_DOMAIN_V1",
    "RTC_POLICY_RUNTIME_SOURCE_CONFIG_HASH_DOMAIN_V1",
    "RTC_PRODUCER_ACTIVATION_READINESS_HASH_DOMAIN_V1",
    "RTC_REQUEST_VALIDITY_POLICY_V1",
    "RTC_TIMING_CONTRACT_HASH_DOMAIN_V1",
    "RTC_TYPED_CANONICAL_SOURCE_ADAPTER_CONTRACT_HASH_V1",
    "PolicyActionEmissionCommitReceipt",
    "RtcDecodedPolicyExecutor",
    "RtcFrontendLifecycleTransition",
    "RtcFrontendStep",
    "RtcPolicyEmissionSourceDescriptor",
    "RtcPolicyEmissionSourceBinding",
    "RtcPolicyProducerConfig",
    "RtcPolicyProducerRuntime",
    "RtcPolicyShadowConfig",
    "RtcPolicyShadowFrontend",
    "RtcProducerActivationReadinessReceipt",
    "RtcProducerTickResult",
    "build_rtc_policy_emission_source_binding",
    "build_rtc_policy_emission_source_binding_from_contracts",
]
