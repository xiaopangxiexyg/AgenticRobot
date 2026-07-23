"""Deterministic virtual-policy runner for deployment RTC contracts.

The runner has no thread, transport, model framework, replay source, teleop
frontend, or hardware capability.  It delays immutable virtual policy results
in a bounded heap, admits them only on an explicit decision tick, pops at most
one due keyframe, and forwards that keyframe through the existing authoritative
MuJoCo deployment runtime.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import AuthoritativeControlStep, RobotAction
from robot_motion.action.proposal import ProposalAdmissionReceipt
from robot_motion.deploy import (
    ASYNC_ALIGNED_CHUNK_MODE_V1,
    InferenceDelayEstimatorSpec,
    InferenceDelayEstimatorState,
    NormalizedAction,
    PolicyActionPlanPopOutcome,
    PolicyChunkMergeReceipt,
    PolicyChunkResult,
    PolicyInferenceRequest,
    TimedPolicyActionKeyframe,
    TimedPolicyActionPlanSnapshot,
    empty_policy_action_plan,
    mark_policy_action_plan_in_flight,
    merge_policy_chunk_result,
    pop_due_policy_action,
)
from robot_motion.deploy.schema import ACTION_DIM
from robot_motion.runtime.observation_cut import ObservationBundle

from .observation import (
    MUJOCO_QPOS_POLICY_IO_LAYOUT_HASH_V1,
    MUJOCO_QPOS_STREAM_ID_V1,
)
from .runtime import MujocoDeployRuntime


VIRTUAL_QPOS_ACTION_CODEC_HASH_V1 = canonical_content_hash(
    {
        "codec_id": "dexteleop.mujoco.virtual_qpos_chunk_codec.v1",
        "action_space": "bimanual_qpos_v1",
        "left_dof": 6,
        "right_dof": 6,
        "decoded_stage": "absolute_qpos_pre_proposal",
    },
    domain="dexteleop.policy_action_codec.v1",
)
VIRTUAL_QPOS_FRAME_CONTRACT_HASH_V1 = canonical_content_hash(
    {
        "contract_id": "dexteleop.mujoco.qpos_frame_not_applicable.v1",
        "action_space": "bimanual_qpos_v1",
        "pose_frame": "not_applicable",
    },
    domain="dexteleop.policy_frame_contract.v1",
)
VIRTUAL_QPOS_NORMALIZATION_HASH_V1 = canonical_content_hash(
    {
        "normalization_id": "dexteleop.mujoco.virtual_qpos_identity.v1",
        "input": "measured_joint_radians",
        "output": "absolute_joint_radians",
    },
    domain="dexteleop.policy_normalization.v1",
)
VIRTUAL_QPOS_POLICY_SPEC_HASH_DOMAIN_V1 = "dexteleop.mujoco.virtual_qpos_policy_spec.v1"
VIRTUAL_RTC_DECISION_HASH_DOMAIN_V1 = "dexteleop.mujoco.virtual_rtc_decision.v1"
VIRTUAL_RTC_INVALIDATION_HASH_DOMAIN_V1 = (
    "dexteleop.mujoco.virtual_rtc_plan_invalidation.v1"
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


def _joint_delta(value: tuple[float, ...], *, field_name: str) -> tuple[float, ...]:
    normalized = tuple(float(item) for item in value)
    if len(normalized) != 6 or any(not math.isfinite(item) for item in normalized):
        raise ValueError(f"{field_name} must contain six finite values.")
    return normalized


def _integer(value: object, *, field_name: str, minimum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer.")
    normalized = int(value)
    if normalized != value or normalized < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{field_name} must be a {qualifier} integer.")
    return normalized


@dataclass(frozen=True)
class VirtualBimanualQposPolicySpec:
    """A deterministic observation-relative qpos policy used only in sim."""

    spec_id: str = "mujoco-virtual-rtc-qpos-v1"
    left_delta_per_step: tuple[float, ...] = (0.01,) * 6
    right_delta_per_step: tuple[float, ...] = (-0.01,) * 6

    def __post_init__(self) -> None:
        object.__setattr__(self, "spec_id", _required(self.spec_id, field_name="spec_id"))
        object.__setattr__(
            self,
            "left_delta_per_step",
            _joint_delta(self.left_delta_per_step, field_name="left_delta_per_step"),
        )
        object.__setattr__(
            self,
            "right_delta_per_step",
            _joint_delta(self.right_delta_per_step, field_name="right_delta_per_step"),
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "spec_id": self.spec_id,
                "left_delta_per_step": self.left_delta_per_step,
                "right_delta_per_step": self.right_delta_per_step,
                "action_codec_hash": VIRTUAL_QPOS_ACTION_CODEC_HASH_V1,
                "frame_contract_hash": VIRTUAL_QPOS_FRAME_CONTRACT_HASH_V1,
                "normalization_hash": VIRTUAL_QPOS_NORMALIZATION_HASH_V1,
            },
            domain=VIRTUAL_QPOS_POLICY_SPEC_HASH_DOMAIN_V1,
        )

    @property
    def artifact_hash(self) -> str:
        return canonical_content_hash(
            {
                "artifact_kind": "deterministic_virtual_policy",
                "policy_spec_hash": self.semantic_content_hash,
            },
            domain="dexteleop.policy_artifact.v1",
        )


class VirtualBimanualQposPolicy:
    """Produce an immutable absolute-qpos chunk from one observation cut."""

    def __init__(
        self,
        spec: VirtualBimanualQposPolicySpec = VirtualBimanualQposPolicySpec(),
    ) -> None:
        if not isinstance(spec, VirtualBimanualQposPolicySpec):
            raise TypeError("spec must be a VirtualBimanualQposPolicySpec.")
        self.spec = spec

    def predict(
        self,
        request: PolicyInferenceRequest,
        observation: ObservationBundle,
        *,
        inference_started_at: float,
        inference_completed_at: float,
        result_received_at: float,
        execution_horizon_steps: int | None = None,
        fault_code: str | None = None,
    ) -> PolicyChunkResult:
        if request.artifact_hash != self.spec.artifact_hash:
            raise ValueError("Request artifact does not match the virtual policy.")
        if request.observation_cut_id != observation.record.observation_cut_id:
            raise ValueError("Request and virtual policy observation cut mismatch.")
        if request.action_codec_hash != VIRTUAL_QPOS_ACTION_CODEC_HASH_V1:
            raise ValueError("Request action codec does not match the virtual policy.")
        if request.frame_contract_hash != VIRTUAL_QPOS_FRAME_CONTRACT_HASH_V1:
            raise ValueError("Request qpos frame contract mismatch.")
        if request.layout_hash != MUJOCO_QPOS_POLICY_IO_LAYOUT_HASH_V1:
            raise ValueError("Request policy layout does not match MuJoCo qpos.")
        if request.normalization_hash != VIRTUAL_QPOS_NORMALIZATION_HASH_V1:
            raise ValueError("Request normalization does not match the virtual policy.")
        started = _finite(inference_started_at, field_name="inference_started_at")
        completed = _finite(inference_completed_at, field_name="inference_completed_at")
        received = _finite(result_received_at, field_name="result_received_at")
        if fault_code is not None:
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
                policy_output_id=None,
                chunk_id=None,
                inference_started_at=started,
                inference_completed_at=completed,
                result_received_at=received,
                clock_domain_id=request.clock_domain_id,
                status="fault",
                source_chunk_size=0,
                decoded_absolute_actions=(),
                model_output_hash=None,
                fault_code=_required(fault_code, field_name="fault_code"),
                fault_detail="deterministic virtual policy fault",
            )

        payload = observation.payload_handles_by_stream.get(MUJOCO_QPOS_STREAM_ID_V1)
        if not isinstance(payload, Mapping):
            raise ValueError("MuJoCo qpos observation payload is missing.")
        qpos_by_side = payload.get("arm_qpos_by_side")
        if not isinstance(qpos_by_side, Mapping):
            raise ValueError("MuJoCo qpos observation has no arm_qpos_by_side mapping.")
        left_anchor = np.asarray(qpos_by_side.get("left"), dtype=np.float32).reshape(-1)
        right_anchor = np.asarray(qpos_by_side.get("right"), dtype=np.float32).reshape(-1)
        if left_anchor.shape != (6,) or right_anchor.shape != (6,):
            raise ValueError("Virtual qpos policy requires two six-DoF arm observations.")
        left_delta = np.asarray(self.spec.left_delta_per_step, dtype=np.float32)
        right_delta = np.asarray(self.spec.right_delta_per_step, dtype=np.float32)
        actions: list[NormalizedAction] = []
        model_values: list[np.ndarray] = []
        for index in range(request.prediction_horizon_steps):
            scale = float(index + 1)
            left = left_anchor + scale * left_delta
            right = right_anchor + scale * right_delta
            model_values.append(np.concatenate((left, right)).astype(np.float32))
            actions.append(
                NormalizedAction(
                    vector=np.full((ACTION_DIM,), np.nan, dtype=np.float32),
                    arm_joint_targets={"left": left, "right": right},
                    metadata={
                        "source": "mujoco_virtual_rtc_policy",
                        "virtual_policy_spec_hash": self.spec.semantic_content_hash,
                    },
                )
            )
        model_output = np.stack(model_values, axis=0)
        horizon = (
            len(actions)
            if execution_horizon_steps is None
            else _integer(
                execution_horizon_steps,
                field_name="execution_horizon_steps",
                minimum=0,
            )
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
            policy_output_id=f"virtual-output:{request.request_id}",
            chunk_id=int(request.semantic_content_hash[:16], 16),
            inference_started_at=started,
            inference_completed_at=completed,
            result_received_at=received,
            clock_domain_id=request.clock_domain_id,
            status="ready",
            source_chunk_size=len(actions),
            decoded_absolute_actions=tuple(actions),
            model_output_hash=canonical_content_hash(
                model_output,
                domain="dexteleop.mujoco.virtual_qpos_model_output.v1",
            ),
            protocol_version="deterministic_virtual_policy_v1",
            effective_delay_steps=math.ceil(
                max(0.0, received - request.admitted_at)
                / request.action_period_sec
                - 1e-12
            ),
            execution_horizon_steps=horizon,
        )


@dataclass(frozen=True)
class VirtualPolicyRequestSubmission:
    accepted: bool
    request_id: str | None
    request: PolicyInferenceRequest | None = field(compare=False, repr=False)
    result: PolicyChunkResult | None = field(compare=False, repr=False)
    scheduled_result_received_at: float | None = None
    rejection_code: str | None = None
    message: str = ""


@dataclass(frozen=True)
class VirtualPolicyCompletionRelease:
    delivery_id: str
    delivered_at: float
    status: str
    result_hash: str
    merge_receipt: PolicyChunkMergeReceipt

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "delivery_id": self.delivery_id,
                "delivered_at": self.delivered_at,
                "status": self.status,
                "result_hash": self.result_hash,
                "merge_receipt_hash": self.merge_receipt.semantic_content_hash,
            },
            domain="dexteleop.mujoco.virtual_policy_completion_release.v1",
        )


@dataclass(frozen=True)
class VirtualRtcPlanInvalidationReceipt:
    """Audit one terminal invalidation of the producer-local RTC plan."""

    occurred_at: float
    status: str
    reason: str
    source_event_ids: tuple[str, ...]
    gate_event_id: str
    gate_event_hash: str
    old_producer_epoch: int
    new_producer_epoch: int
    old_plan_hash: str
    new_plan_hash: str
    discarded_keyframe_count: int
    discarded_in_flight_request: bool
    retained_pending_result_count: int

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "occurred_at": self.occurred_at,
                "status": self.status,
                "reason": self.reason,
                "source_event_ids": self.source_event_ids,
                "gate_event_id": self.gate_event_id,
                "gate_event_hash": self.gate_event_hash,
                "old_producer_epoch": self.old_producer_epoch,
                "new_producer_epoch": self.new_producer_epoch,
                "old_plan_hash": self.old_plan_hash,
                "new_plan_hash": self.new_plan_hash,
                "discarded_keyframe_count": self.discarded_keyframe_count,
                "discarded_in_flight_request": self.discarded_in_flight_request,
                "retained_pending_result_count": self.retained_pending_result_count,
            },
            domain=VIRTUAL_RTC_INVALIDATION_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class VirtualRtcDecisionResult:
    decision_at: float
    completion_releases: tuple[VirtualPolicyCompletionRelease, ...]
    plan_pop: PolicyActionPlanPopOutcome
    proposal_receipt: ProposalAdmissionReceipt | None = field(compare=False, repr=False)
    authoritative_step: AuthoritativeControlStep = field(compare=False, repr=False)
    plan_hash: str
    emission_count: int
    underrun: bool

    @property
    def emitted_keyframe(self) -> TimedPolicyActionKeyframe | None:
        return self.plan_pop.keyframe

    @property
    def semantic_content_hash(self) -> str:
        decision_hash = (
            None
            if self.authoritative_step.decision is None
            else self.authoritative_step.decision.semantic_content_hash
        )
        return canonical_content_hash(
            {
                "decision_at": self.decision_at,
                "completion_release_hashes": tuple(
                    release.semantic_content_hash for release in self.completion_releases
                ),
                "emitted_keyframe_hash": (
                    None
                    if self.emitted_keyframe is None
                    else self.emitted_keyframe.semantic_content_hash
                ),
                "proposal_receipt_hash": (
                    None
                    if self.proposal_receipt is None
                    else self.proposal_receipt.semantic_content_hash
                ),
                "authoritative_status": self.authoritative_step.status,
                "authoritative_decision_hash": decision_hash,
                "plan_hash": self.plan_hash,
                "emission_count": self.emission_count,
                "underrun": self.underrun,
            },
            domain=VIRTUAL_RTC_DECISION_HASH_DOMAIN_V1,
        )


@dataclass(order=True, frozen=True)
class _ScheduledPolicyResult:
    deliver_at: float
    delivery_sequence: int
    delivery_id: str = field(compare=False)
    request: PolicyInferenceRequest = field(compare=False, repr=False)
    result: PolicyChunkResult = field(compare=False, repr=False)


class DeterministicVirtualPolicyRtcRunner:
    """Drive RTC plan admission and one authoritative MuJoCo deployment sink."""

    def __init__(
        self,
        runtime: MujocoDeployRuntime,
        *,
        policy: VirtualBimanualQposPolicy,
        set_control_time: Callable[[float], None],
        start_at: float,
        result_mailbox_capacity: int = 8,
        delay_estimator_spec: InferenceDelayEstimatorSpec | None = None,
    ) -> None:
        if not isinstance(runtime, MujocoDeployRuntime):
            raise TypeError("runtime must be a MujocoDeployRuntime.")
        if not isinstance(policy, VirtualBimanualQposPolicy):
            raise TypeError("policy must be a VirtualBimanualQposPolicy.")
        if not callable(set_control_time):
            raise TypeError("set_control_time must be callable.")
        capacity = _integer(
            result_mailbox_capacity,
            field_name="result_mailbox_capacity",
            minimum=1,
        )
        self.runtime = runtime
        self.policy = policy
        self.set_control_time = set_control_time
        self.current_time = _finite(start_at, field_name="start_at")
        self.action_period_sec = 1.0 / float(runtime.config.decision_hz)
        self.schedule_origin = self.current_time
        self.result_mailbox_capacity = capacity
        self.delay_estimator_spec = delay_estimator_spec or InferenceDelayEstimatorSpec(
            spec_id="mujoco-virtual-rtc-delay-v1",
            action_period_sec=self.action_period_sec,
            window_size=16,
            warmup_latency_sec=0.0,
            maximum_delay_steps=64,
        )
        if not math.isclose(
            self.delay_estimator_spec.action_period_sec,
            self.action_period_sec,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Delay estimator cadence must match the sim decision cadence.")
        self.delay_estimator_state = InferenceDelayEstimatorState.empty(
            self.delay_estimator_spec
        )
        self.plan = empty_policy_action_plan(
            plan_id=(
                f"rtc-plan:{runtime.config.runtime_session_id}:"
                f"{runtime.config.control_domain_id}:{runtime.config.policy_node_id}"
            ),
            session_id=runtime.config.runtime_session_id,
            control_domain_id=runtime.config.control_domain_id,
            producer_id=runtime.config.producer_id,
            producer_epoch=runtime.config.producer_epoch,
            policy_node_id=runtime.config.policy_node_id,
            clock_domain_id=runtime.config.clock_domain_id,
            cursor_at=self.current_time,
            created_at=self.current_time,
        )
        self._next_request_sequence = 0
        self._next_delivery_sequence = 0
        self._scheduled_results: list[_ScheduledPolicyResult] = []
        self._known_delivery_ids: set[str] = set()
        self._completion_releases: list[VirtualPolicyCompletionRelease] = []
        self._decisions: list[VirtualRtcDecisionResult] = []
        self._invalidations: list[VirtualRtcPlanInvalidationReceipt] = []
        self._emission_count = 0
        self._lifecycle_invalidated = False
        self.set_control_time(self.current_time)

    @property
    def pending_result_count(self) -> int:
        return len(self._scheduled_results)

    @property
    def completion_releases(self) -> tuple[VirtualPolicyCompletionRelease, ...]:
        return tuple(self._completion_releases)

    @property
    def decisions(self) -> tuple[VirtualRtcDecisionResult, ...]:
        return tuple(self._decisions)

    @property
    def invalidations(self) -> tuple[VirtualRtcPlanInvalidationReceipt, ...]:
        return tuple(self._invalidations)

    @property
    def lifecycle_invalidated(self) -> bool:
        return self._lifecycle_invalidated

    @property
    def emission_count(self) -> int:
        return self._emission_count

    def _set_time(self, value: float) -> None:
        timestamp = _finite(value, field_name="control_time")
        if timestamp < self.current_time:
            raise ValueError("RTC runner time cannot move backwards.")
        self.current_time = timestamp
        self.set_control_time(timestamp)

    def _aligned_target_at_or_after(self, not_before: float) -> float:
        relative = (not_before - self.schedule_origin) / self.action_period_sec
        tick = max(0, math.ceil(relative - 1e-12))
        return self.schedule_origin + tick * self.action_period_sec

    def _schedule_delivery(
        self,
        *,
        request: PolicyInferenceRequest,
        result: PolicyChunkResult,
        deliver_at: float,
        delivery_id: str,
    ) -> bool:
        normalized_delivery_id = _required(delivery_id, field_name="delivery_id")
        if (
            len(self._scheduled_results) >= self.result_mailbox_capacity
            or normalized_delivery_id in self._known_delivery_ids
        ):
            return False
        delivered = _finite(deliver_at, field_name="deliver_at")
        if delivered < self.current_time:
            raise ValueError("deliver_at cannot precede current runner time.")
        heapq.heappush(
            self._scheduled_results,
            _ScheduledPolicyResult(
                deliver_at=delivered,
                delivery_sequence=self._next_delivery_sequence,
                delivery_id=normalized_delivery_id,
                request=request,
                result=result,
            ),
        )
        self._known_delivery_ids.add(normalized_delivery_id)
        self._next_delivery_sequence += 1
        return True

    def schedule_prediction(
        self,
        *,
        requested_at: float,
        inference_completed_at: float,
        result_received_at: float | None = None,
        prediction_horizon_steps: int = 4,
        minimum_execution_horizon_steps: int = 1,
        forecast_delay_steps: int | None = None,
        validity_horizon_steps: int | None = None,
        execution_horizon_steps: int | None = None,
        fault_code: str | None = None,
    ) -> VirtualPolicyRequestSubmission:
        requested = _finite(requested_at, field_name="requested_at")
        completed = _finite(
            inference_completed_at,
            field_name="inference_completed_at",
        )
        received = completed if result_received_at is None else _finite(
            result_received_at,
            field_name="result_received_at",
        )
        if requested < self.current_time:
            raise ValueError("requested_at cannot precede current runner time.")
        if self._lifecycle_invalidated:
            return VirtualPolicyRequestSubmission(
                accepted=False,
                request_id=None,
                request=None,
                result=None,
                rejection_code="lifecycle_invalidated",
                message=(
                    "The virtual RTC runner was terminally invalidated by an "
                    "execution-gate transition."
                ),
            )
        if self.plan.in_flight_request_id is not None:
            return VirtualPolicyRequestSubmission(
                accepted=False,
                request_id=None,
                request=None,
                result=None,
                rejection_code="in_flight_request_exists",
                message="The RTC plan already has one in-flight request.",
            )
        if len(self._scheduled_results) >= self.result_mailbox_capacity:
            return VirtualPolicyRequestSubmission(
                accepted=False,
                request_id=None,
                request=None,
                result=None,
                rejection_code="result_mailbox_capacity_exhausted",
                message="The deterministic result mailbox is full.",
            )
        horizon = _integer(
            prediction_horizon_steps,
            field_name="prediction_horizon_steps",
            minimum=1,
        )
        minimum = _integer(
            minimum_execution_horizon_steps,
            field_name="minimum_execution_horizon_steps",
            minimum=1,
        )
        if minimum > horizon:
            raise ValueError("Prediction/minimum execution horizons are invalid.")
        forecast = (
            self.delay_estimator_state.forecast_steps(self.delay_estimator_spec)
            if forecast_delay_steps is None
            else _integer(
                forecast_delay_steps,
                field_name="forecast_delay_steps",
                minimum=0,
            )
        )
        validity_steps = (
            horizon
            if validity_horizon_steps is None
            else _integer(
                validity_horizon_steps,
                field_name="validity_horizon_steps",
                minimum=1,
            )
        )
        self._set_time(requested)
        observation = self.runtime.capture_policy_observation(
            trigger_reason="deterministic_virtual_rtc",
        )
        first_target = self._aligned_target_at_or_after(
            requested + forecast * self.action_period_sec
        )
        request_sequence = self._next_request_sequence
        request_id = f"mujoco-rtc-request:{request_sequence}"
        inference_id = f"mujoco-rtc-inference:{request_sequence}"
        request = PolicyInferenceRequest(
            request_id=request_id,
            inference_id=inference_id,
            rtc_mode=ASYNC_ALIGNED_CHUNK_MODE_V1,
            session_id=self.runtime.config.runtime_session_id,
            control_domain_id=self.runtime.config.control_domain_id,
            producer_id=self.runtime.config.producer_id,
            producer_epoch=self.plan.producer_epoch,
            policy_node_id=self.runtime.config.policy_node_id,
            artifact_hash=self.policy.spec.artifact_hash,
            observation_cut_id=observation.record.observation_cut_id,
            observation_timestamp=observation.record.built_at,
            observation_step_index=observation.record.sequence,
            triggered_at=requested,
            admitted_at=requested,
            first_target_effective_at=first_target,
            clock_domain_id=self.runtime.config.clock_domain_id,
            action_period_sec=self.action_period_sec,
            prediction_horizon_steps=horizon,
            minimum_execution_horizon_steps=minimum,
            source_plan_id=self.plan.plan_id,
            source_plan_generation=self.plan.generation,
            source_plan_hash=self.plan.semantic_content_hash,
            source_plan_cursor_at=self.plan.cursor_at,
            producer_plan_committed_through=self.plan.producer_committed_through,
            forecast_delay_steps=forecast,
            delay_estimator_spec_hash=self.delay_estimator_spec.semantic_content_hash,
            action_codec_hash=VIRTUAL_QPOS_ACTION_CODEC_HASH_V1,
            frame_contract_hash=VIRTUAL_QPOS_FRAME_CONTRACT_HASH_V1,
            layout_hash=MUJOCO_QPOS_POLICY_IO_LAYOUT_HASH_V1,
            normalization_hash=VIRTUAL_QPOS_NORMALIZATION_HASH_V1,
            valid_until=first_target + validity_steps * self.action_period_sec,
        )
        in_flight_plan = mark_policy_action_plan_in_flight(self.plan, request)
        result = self.policy.predict(
            request,
            observation,
            inference_started_at=requested,
            inference_completed_at=completed,
            result_received_at=received,
            execution_horizon_steps=execution_horizon_steps,
            fault_code=fault_code,
        )
        scheduled = self._schedule_delivery(
            request=request,
            result=result,
            deliver_at=received,
            delivery_id=f"delivery:{request_id}:0",
        )
        if not scheduled:  # guarded by the capacity check above
            raise AssertionError("Result mailbox capacity changed during scheduling.")
        self.plan = in_flight_plan
        self._next_request_sequence += 1
        return VirtualPolicyRequestSubmission(
            accepted=True,
            request_id=request_id,
            request=request,
            result=result,
            scheduled_result_received_at=received,
        )

    def schedule_result_delivery(
        self,
        *,
        request: PolicyInferenceRequest,
        result: PolicyChunkResult,
        deliver_at: float,
        delivery_id: str,
    ) -> bool:
        """Inject a deterministic duplicate/out-of-order delivery fixture."""

        return self._schedule_delivery(
            request=request,
            result=result,
            deliver_at=deliver_at,
            delivery_id=delivery_id,
        )

    def invalidate_for_execution_gate(
        self,
        status: str,
        *,
        occurred_at: float,
        reason: str,
        source_event_ids: tuple[str, ...] = (),
    ) -> VirtualRtcPlanInvalidationReceipt:
        """Enter HOLD/FAULT/TRANSITIONING and invalidate this RTC producer epoch.

        Scheduled result envelopes are deliberately retained.  Their old-epoch
        payloads therefore still reach the merge boundary and are rejected
        against the new empty plan, exercising the same fail-closed property a
        real worker completion must satisfy.  Recovery requires a new runner;
        this fixture never silently resumes inference after a lifecycle stop.
        """

        if self._lifecycle_invalidated:
            raise RuntimeError("The virtual RTC runner is already invalidated.")
        normalized_status = _required(status, field_name="status").lower()
        if normalized_status not in {"hold", "fault", "transitioning"}:
            raise ValueError(
                "RTC plan invalidation requires HOLD, FAULT, or TRANSITIONING."
            )
        normalized_reason = _required(reason, field_name="reason")
        normalized_sources = tuple(
            _required(value, field_name="source_event_id")
            for value in source_event_ids
        )
        timestamp = _finite(occurred_at, field_name="occurred_at")
        self._set_time(timestamp)

        old_plan = self.plan
        gate_event = self.runtime.set_execution_gate(
            normalized_status,
            reason=normalized_reason,
            source_event_ids=normalized_sources,
        )
        new_producer_epoch = old_plan.producer_epoch + 1
        self.runtime.activate_policy_node_producer(
            self.runtime.config.producer_id,
            producer_epoch=new_producer_epoch,
            reason=f"virtual RTC lifecycle invalidation: {normalized_reason}",
        )
        self.plan = empty_policy_action_plan(
            plan_id=old_plan.plan_id,
            session_id=old_plan.session_id,
            control_domain_id=old_plan.control_domain_id,
            producer_id=old_plan.producer_id,
            producer_epoch=new_producer_epoch,
            policy_node_id=old_plan.policy_node_id,
            clock_domain_id=old_plan.clock_domain_id,
            cursor_at=max(old_plan.cursor_at, timestamp),
            created_at=timestamp,
        )
        self._lifecycle_invalidated = True
        receipt = VirtualRtcPlanInvalidationReceipt(
            occurred_at=timestamp,
            status=normalized_status,
            reason=normalized_reason,
            source_event_ids=normalized_sources,
            gate_event_id=gate_event.event_id,
            gate_event_hash=gate_event.semantic_content_hash,
            old_producer_epoch=old_plan.producer_epoch,
            new_producer_epoch=new_producer_epoch,
            old_plan_hash=old_plan.semantic_content_hash,
            new_plan_hash=self.plan.semantic_content_hash,
            discarded_keyframe_count=len(old_plan.keyframes),
            discarded_in_flight_request=old_plan.in_flight_request_id is not None,
            retained_pending_result_count=len(self._scheduled_results),
        )
        self._invalidations.append(receipt)
        return receipt

    @staticmethod
    def _robot_action_from_keyframe(
        keyframe: TimedPolicyActionKeyframe,
        *,
        emitted_at: float,
    ) -> RobotAction:
        joints = keyframe.decoded_absolute_action.arm_joint_targets
        if set(joints) != {"left", "right"}:
            raise ValueError("Virtual RTC qpos keyframe must contain both arm targets.")
        return RobotAction.bimanual_qpos(
            left_arm=np.asarray(joints["left"], dtype=np.float32),
            right_arm=np.asarray(joints["right"], dtype=np.float32),
            timestamp=emitted_at,
            source="mujoco_virtual_rtc_policy",
        )

    def step(
        self,
        *,
        decision_at: float,
        physics_ticks_after_dispatch: int = 0,
    ) -> VirtualRtcDecisionResult:
        decision = _finite(decision_at, field_name="decision_at")
        self._set_time(decision)
        releases: list[VirtualPolicyCompletionRelease] = []
        while self._scheduled_results and self._scheduled_results[0].deliver_at <= decision:
            scheduled = heapq.heappop(self._scheduled_results)
            outcome = merge_policy_chunk_result(
                self.plan,
                scheduled.request,
                scheduled.result,
                admitted_at=decision,
                actual_cursor_at=decision,
                producer_committed_through=self.plan.producer_committed_through,
            )
            self.plan = outcome.plan
            status = "admitted" if outcome.receipt.accepted else "rejected"
            release = VirtualPolicyCompletionRelease(
                delivery_id=scheduled.delivery_id,
                delivered_at=decision,
                status=status,
                result_hash=scheduled.result.semantic_content_hash,
                merge_receipt=outcome.receipt,
            )
            releases.append(release)
            self._completion_releases.append(release)
            if outcome.receipt.accepted:
                latency = max(
                    0.0,
                    scheduled.result.result_received_at - scheduled.request.admitted_at,
                )
                self.delay_estimator_state = self.delay_estimator_state.record(
                    self.delay_estimator_spec,
                    latency,
                )

        plan_pop = pop_due_policy_action(self.plan, decision_at=decision)
        self.plan = plan_pop.plan
        proposal_receipt: ProposalAdmissionReceipt | None = None
        if plan_pop.keyframe is not None:
            keyframe = plan_pop.keyframe
            proposal_receipt = self.runtime.submit_policy_action(
                self._robot_action_from_keyframe(keyframe, emitted_at=decision),
                policy_output_id=(
                    f"{keyframe.policy_output_id}:step:{keyframe.chunk_keyframe_index}"
                ),
                produced_at=decision,
                target_effective_at=keyframe.target_effective_at,
                observation_cut_id=keyframe.observation_cut_id,
                chunk_id=f"rtc-chunk:{keyframe.chunk_id}",
                chunk_keyframe_index=keyframe.chunk_keyframe_index,
                inference_id=keyframe.inference_id,
            )
            self._emission_count += 1
        authoritative_step = self.runtime.step(
            now=decision,
            physics_ticks_after_dispatch=physics_ticks_after_dispatch,
        )
        underrun = plan_pop.keyframe is None and not self.plan.keyframes
        result = VirtualRtcDecisionResult(
            decision_at=decision,
            completion_releases=tuple(releases),
            plan_pop=plan_pop,
            proposal_receipt=proposal_receipt,
            authoritative_step=authoritative_step,
            plan_hash=self.plan.semantic_content_hash,
            emission_count=self._emission_count,
            underrun=underrun,
        )
        self._decisions.append(result)
        return result


__all__ = [
    "DeterministicVirtualPolicyRtcRunner",
    "VIRTUAL_QPOS_ACTION_CODEC_HASH_V1",
    "VIRTUAL_QPOS_FRAME_CONTRACT_HASH_V1",
    "VIRTUAL_QPOS_NORMALIZATION_HASH_V1",
    "VIRTUAL_QPOS_POLICY_SPEC_HASH_DOMAIN_V1",
    "VIRTUAL_RTC_INVALIDATION_HASH_DOMAIN_V1",
    "VirtualBimanualQposPolicy",
    "VirtualBimanualQposPolicySpec",
    "VirtualPolicyCompletionRelease",
    "VirtualPolicyRequestSubmission",
    "VirtualRtcDecisionResult",
    "VirtualRtcPlanInvalidationReceipt",
]
