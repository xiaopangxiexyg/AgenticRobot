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
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.relative_action import normalize_action_pose_mode

from .action_representation import PolicyActionRepresentationContext
from .async_producer import (
    POLICY_ASYNC_PRODUCER_ELIGIBLE,
    POLICY_ASYNC_PRODUCER_FAULT,
    POLICY_ASYNC_PRODUCER_HOLD,
    POLICY_ASYNC_PRODUCER_TRANSITIONING,
    POLICY_WORKER_SESSION_ACTIVE,
    POLICY_WORKER_SESSION_DRAINED,
    POLICY_WORKER_SESSION_RETIRED_WAITING_FOR_COMPLETION,
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
    InferenceDelayEstimatorSpec,
    InferenceDelayEstimatorState,
    PolicyChunkResult,
    PolicyInferenceRequest,
)
from .observation import DeploymentObservation
from .observation_view import PolicyObservationView
from .runtime import PolicyActionPreparationResult, PolicyDeployRuntime

from .policy_worker import (
    BoundedPolicyInferenceWorker,
    PolicyInferenceExecutor,
    PolicyInferenceWorkItem,
    PolicyWorkerCompletion,
    PolicyWorkerStopReceipt,
    PolicyWorkerSubmissionReceipt,
)


RTC_3C_FRONTEND_CONFIG_HASH_DOMAIN_V1 = (
    "dexteleop.deploy.rtc_3c_frontend_config.v1"
)
RTC_3C_FRONTEND_CONFIG_HASH_DOMAIN_V2 = (
    "dexteleop.deploy.rtc_3c_frontend_config.v2"
)
RTC_3C_LOCAL_POLICY_PROTOCOL_V1 = "rtc_3c_local_policy_worker_v1"
RTC_OBSERVATION_CUT_IDENTITY_HASH_DOMAIN_V2 = (
    "dexteleop.rtc_observation_cut_identity.v2"
)


def _required(value: object, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


def _finite(value: object, *, field_name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


def _positive_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer.")
    normalized = int(value)
    if normalized != value or normalized <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return normalized


def _nonnegative_integer(value: object, *, field_name: str) -> int:
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
        if self.profile.position_lookahead_enabled or self.profile.send_actions:
            raise ValueError("RTC shadow profiles require lookahead/send_actions disabled.")
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
        expected_period = 1.0 / self.profile.action_hz
        if not math.isclose(
            self.delay_estimator_spec.action_period_sec,
            expected_period,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("delay estimator/action profile period mismatch.")

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
            "delay_estimator_spec_hash": (
                self.delay_estimator_spec.semantic_content_hash
            ),
        }
        domain = RTC_3C_FRONTEND_CONFIG_HASH_DOMAIN_V1
        if self.action_pose_mode is not None:
            payload["action_pose_mode"] = self.action_pose_mode
            domain = RTC_3C_FRONTEND_CONFIG_HASH_DOMAIN_V2
        return canonical_content_hash(payload, domain=domain)


@dataclass(frozen=True)
class RtcFrontendLifecycleTransition:
    producer_transition: PolicyAsyncProducerTransitionOutcome
    worker_stop_receipt: PolicyWorkerStopReceipt


@dataclass(frozen=True)
class RtcFrontendStep:
    state: PolicyAsyncProducerSnapshot
    completion: PolicyWorkerCompletion | None
    result_transition: PolicyAsyncProducerTransitionOutcome | None
    pop_transition: PolicyAsyncProducerTransitionOutcome | None
    emission_commit_transition: PolicyAsyncProducerTransitionOutcome | None
    request_transition: PolicyAsyncProducerTransitionOutcome | None
    worker_submission: PolicyWorkerSubmissionReceipt | None
    action_preparation: PolicyActionPreparationResult
    lifecycle_transition: RtcFrontendLifecycleTransition | None
    underrun: bool
    control_duration_sec: float
    control_budget_sec: float
    selected_emission_count: int
    late_completion_rejection_count: int

    @property
    def budget_exceeded(self) -> bool:
        return self.control_duration_sec > self.control_budget_sec

    def to_debug_payload(self) -> dict[str, Any]:
        emission = self.action_preparation.emission
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
            "request_transition_hash": (
                None
                if self.request_transition is None
                else self.request_transition.receipt.semantic_content_hash
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
            "underrun": self.underrun,
            "control_duration_sec": self.control_duration_sec,
            "control_budget_sec": self.control_budget_sec,
            "budget_exceeded": self.budget_exceeded,
            "selected_emission_count": self.selected_emission_count,
            "late_completion_rejection_count": (
                self.late_completion_rejection_count
            ),
            "position_lookahead_enabled": False,
            "send_actions": False,
        }


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
        preparer: Any,
        created_at: float,
        clock: Callable[[], float] = time.monotonic,
        executor_factory: Callable[[str], PolicyInferenceExecutor] | None = None,
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
        self.config = config
        self.deploy_runtime = deploy_runtime
        self.policy = policy
        self.preparer = preparer
        self.clock = clock
        self._executor_factory = executor_factory
        if representation_context_builder is not None and not callable(
            representation_context_builder
        ):
            raise TypeError("representation_context_builder must be callable or None.")
        self._representation_context_builder = representation_context_builder
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
        self._worker = self._build_worker(worker_session_id)
        self._delay_state = InferenceDelayEstimatorState.empty(
            config.delay_estimator_spec
        )
        self._next_request_sequence = 0
        self._last_request_triggered_at: float | None = None
        self._selected_emission_count = 0
        self._late_completion_rejection_count = 0
        self._emitted_keyframe_ids: set[tuple[str, int, int]] = set()
        self._started = False
        self._closed = False

    @property
    def state(self) -> PolicyAsyncProducerSnapshot:
        return self._state

    @property
    def selected_emission_count(self) -> int:
        return self._selected_emission_count

    @property
    def late_completion_rejection_count(self) -> int:
        return self._late_completion_rejection_count

    @property
    def pending_completion_count(self) -> int:
        """Expose bounded-mailbox occupancy without leaking worker ownership."""

        return self._worker.pending_completion_count

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
        period = 1.0 / self.config.profile.action_hz
        return max(0, math.ceil((valid[-1].effective_until - now) / period))

    def _should_request(self, now: float) -> bool:
        observation_due = (
            self._last_request_triggered_at is None
            or now - self._last_request_triggered_at
            >= 1.0 / self.config.profile.observation_hz - 1e-12
        )
        return (
            self._state.lifecycle_state == POLICY_ASYNC_PRODUCER_ELIGIBLE
            and self._state.worker_session_state == POLICY_WORKER_SESSION_ACTIVE
            and self._state.active_request is None
            and observation_due
            and self._coverage_steps(now)
            <= self.config.request_trigger_coverage_steps
        )

    def _build_request(
        self,
        actor: PolicyObservationView,
        *,
        now: float,
    ) -> PolicyInferenceRequest:
        sequence = self._next_request_sequence
        self._next_request_sequence += 1
        forecast_steps = self._delay_state.forecast_steps(
            self.config.delay_estimator_spec
        )
        period = 1.0 / self.config.profile.action_hz
        first_target = max(now, self._state.plan.cursor_at) + forecast_steps * period
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
        return PolicyInferenceRequest(
            request_id=request_id,
            inference_id=f"{self.config.policy_node_id}:rtc-inference:{sequence}",
            rtc_mode=self.config.profile.rtc_mode,
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
            valid_until=first_target + self.config.request_validity_steps * period,
            action_representation_context=representation_context,
        )

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
        stop = self._worker.stop(
            timeout_sec=self.config.worker_stop_timeout_sec
        )
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
        self._last_request_triggered_at = None
        return transition

    def step(
        self,
        observation: DeploymentObservation,
        *,
        now: float,
        allow_output: bool = True,
    ) -> RtcFrontendStep:
        self._require_running()
        tick_started = time.perf_counter()
        decision_at = _finite(now, field_name="now")
        completion = self._worker.poll_completion()
        result_transition = None
        lifecycle_transition = None
        if completion is not None:
            lifecycle_before_result = self._state.lifecycle_state
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
                completion.released_at - completion.result.inference_started_at,
            )
            self._delay_state = self._delay_state.record(
                self.config.delay_estimator_spec,
                latency,
            )
            if completion.result.status == "fault" and (
                result_transition.merge_receipt is not None
            ):
                lifecycle_transition = self.fault(
                    now=decision_at,
                    reason=completion.result.fault_code or "policy_worker_fault",
                )
        self._refresh_worker_session_state(now=decision_at)

        pop_transition = None
        emission_commit_transition = None
        action_preparation = PolicyActionPreparationResult()
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
                action_preparation = (
                    self.deploy_runtime.prepare_timed_action_keyframe(
                        pop.keyframe,
                        observation,
                        preparer=self.preparer,
                        policy_name=str(getattr(self.policy, "name", "rtc_policy")),
                        plan_keyframe_count_before=before,
                        plan_keyframe_count_after=len(self._state.plan.keyframes),
                        plan_pop_count=self._selected_emission_count,
                    )
                )
                if action_preparation.emission is None:
                    lifecycle_transition = self.fault(
                        now=decision_at,
                        reason="formal_emission_failed",
                    )
                    action_preparation = PolicyActionPreparationResult()
                else:
                    emission_commit_transition = (
                        commit_emitted_policy_async_producer_action(
                            self._state,
                            pop_outcome=pop,
                            emission_id=action_preparation.emission.emission_id,
                            committed_at=decision_at,
                        )
                    )
                    self._state = emission_commit_transition.state
                    if not emission_commit_transition.receipt.accepted:
                        lifecycle_transition = self.fault(
                            now=decision_at,
                            reason="emission_commit_rejected",
                        )
                        action_preparation = PolicyActionPreparationResult()

        request_transition = None
        worker_submission = None
        if allow_output and self._should_request(decision_at):
            policy_observation = self.deploy_runtime.prepare_policy_observation(
                observation
            )
            actor, _parity = self.deploy_runtime.prepare_actor_observation(
                policy_observation
            )
            try:
                request = self._build_request(actor, now=decision_at)
            except Exception as exc:  # noqa: BLE001 - invalid context must fail closed.
                lifecycle_transition = self.fault(
                    now=decision_at,
                    reason=(
                        "inference_request_invalid:"
                        f"{type(exc).__name__}:{exc}"
                    ),
                )
            else:
                request_transition = admit_policy_inference_request(
                    self._state,
                    request,
                )
                self._state = request_transition.state
                if request_transition.receipt.accepted:
                    self._last_request_triggered_at = decision_at
                    work_item = PolicyInferenceWorkItem(
                        worker_id=self._worker.worker_id,
                        worker_session_id=self._worker.worker_session_id,
                        submit_sequence=self._next_request_sequence - 1,
                        request=request,
                        actor_observation=actor,
                        submitted_at=decision_at,
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
            request_transition=request_transition,
            worker_submission=worker_submission,
            action_preparation=action_preparation,
            lifecycle_transition=lifecycle_transition,
            underrun=underrun,
            control_duration_sec=duration,
            control_budget_sec=1.0 / self.config.profile.decision_hz,
            selected_emission_count=self._selected_emission_count,
            late_completion_rejection_count=(
                self._late_completion_rejection_count
            ),
        )

    def close(self, *, now: float | None = None) -> PolicyWorkerStopReceipt:
        if self._closed:
            return self._worker.stop(timeout_sec=0.0)
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
            timeout_sec=self.config.worker_stop_timeout_sec
        )
        self._closed = True
        return stop


__all__ = [
    "RTC_3C_FRONTEND_CONFIG_HASH_DOMAIN_V1",
    "RTC_3C_LOCAL_POLICY_PROTOCOL_V1",
    "RtcDecodedPolicyExecutor",
    "RtcFrontendLifecycleTransition",
    "RtcFrontendStep",
    "RtcPolicyShadowConfig",
    "RtcPolicyShadowFrontend",
]
