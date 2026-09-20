"""Pure control-thread state machine for asynchronous policy producers.

This module owns no thread, transport, frontend, safety, reference, dispatcher,
or backend.  It closes the request/result/timed-plan admission state at one
immutable control-thread transition boundary.  The legacy action deque and
``PolicyChunkPredictionCandidate`` are intentionally not part of this state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.position_control.config import PositionControlRates

from .chunk_plan import (
    PolicyActionPlanPopOutcome,
    PolicyChunkMergeReceipt,
    TimedPolicyActionPlanSnapshot,
    commit_policy_action_plan_through,
    empty_policy_action_plan,
    mark_policy_action_plan_in_flight,
    merge_policy_chunk_result,
    pop_due_policy_action,
)
from .inference import (
    ASYNC_ALIGNED_CHUNK_MODE_V1,
    POLICY_INFERENCE_MODES_V1,
    RTC_FLOW_INPAINT_MODE_V1,
    PolicyChunkResult,
    PolicyInferenceRequest,
)

if TYPE_CHECKING:
    from .rtc.profile import RtcProducerProfile


POLICY_ASYNC_PRODUCER_STATE_HASH_DOMAIN_V1 = (
    "dexteleop.policy_async_producer_state.v1"
)
POLICY_ASYNC_PRODUCER_TRANSITION_HASH_DOMAIN_V1 = (
    "dexteleop.policy_async_producer_transition.v1"
)
POLICY_ASYNC_EXECUTION_PROFILE_HASH_DOMAIN_V1 = (
    "dexteleop.policy_async_execution_profile.v1"
)

POLICY_ASYNC_PRODUCER_ELIGIBLE = "ELIGIBLE"
POLICY_ASYNC_PRODUCER_HOLD = "HOLD"
POLICY_ASYNC_PRODUCER_FAULT = "FAULT"
POLICY_ASYNC_PRODUCER_TRANSITIONING = "TRANSITIONING"
POLICY_ASYNC_PRODUCER_LIFECYCLE_STATES_V1 = (
    POLICY_ASYNC_PRODUCER_ELIGIBLE,
    POLICY_ASYNC_PRODUCER_HOLD,
    POLICY_ASYNC_PRODUCER_FAULT,
    POLICY_ASYNC_PRODUCER_TRANSITIONING,
)

POLICY_WORKER_SESSION_ACTIVE = "ACTIVE"
POLICY_WORKER_SESSION_RETIRED_WAITING_FOR_COMPLETION = (
    "RETIRED_WAITING_FOR_COMPLETION"
)
POLICY_WORKER_SESSION_ABANDONED_UNDRAINED = "ABANDONED_UNDRAINED"
POLICY_WORKER_SESSION_DRAINED = "DRAINED"
POLICY_WORKER_SESSION_STATES_V1 = (
    POLICY_WORKER_SESSION_ACTIVE,
    POLICY_WORKER_SESSION_RETIRED_WAITING_FOR_COMPLETION,
    POLICY_WORKER_SESSION_ABANDONED_UNDRAINED,
    POLICY_WORKER_SESSION_DRAINED,
)

POLICY_REQUEST_TRIGGER_COVERAGE_SINGLE_IN_FLIGHT_V1 = (
    "coverage_triggered_single_in_flight_v1"
)
POLICY_DUE_POP_LATEST_VALID_NO_CATCHUP_V1 = (
    "latest_valid_due_no_catchup_v1"
)
RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1 = "rtc_3c_virtual_shadow_15_30_60_120_v1"
RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1 = "raw_rtc_final_chunk_15_30_60_120_v1"
MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1 = (
    "model_rtc_flow_inpaint_shadow_15_30_60_120_v1"
)
MODEL_RTC_FLOW_INPAINT_SHADOW_RUNTIME_15HZ_PROFILE_ID_V1 = (
    "model_rtc_flow_inpaint_shadow_15_15_60_120_runtime_v1"
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


def _positive_rate(value: object, *, field_name: str) -> float:
    normalized = _finite(value, field_name=field_name)
    if normalized <= 0.0:
        raise ValueError(f"{field_name} must be positive.")
    return normalized


def _nonnegative_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer.")
    normalized = int(value)
    if normalized != value or normalized < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return normalized


@dataclass(frozen=True)
class PolicyAsyncExecutionProfile:
    """Legacy composed RTC/application profile retained for compatibility.

    The serialized contract predates the target-neutral
    :class:`robot_motion.deploy.rtc.RtcProducerProfile`, so its position rates
    and application send/lookahead gates cannot be removed or reinterpreted.
    New RTC producer code must consume :attr:`producer_profile`; embedding
    applications remain responsible for the additional execution settings.
    """

    profile_id: str
    rtc_mode: str = ASYNC_ALIGNED_CHUNK_MODE_V1
    observation_hz: float = 15.0
    decision_hz: float = 30.0
    action_hz: float = 30.0
    position_rates: PositionControlRates = field(
        default_factory=lambda: PositionControlRates(
            sample_hz=15.0,
            reference_hz=60.0,
            ik_hz=60.0,
            command_hz=120.0,
        )
    )
    request_trigger_mode: str = POLICY_REQUEST_TRIGGER_COVERAGE_SINGLE_IN_FLIGHT_V1
    due_pop_mode: str = POLICY_DUE_POP_LATEST_VALID_NO_CATCHUP_V1
    position_lookahead_enabled: bool = False
    send_actions: bool = False

    def __post_init__(self) -> None:
        profile_id = _required(self.profile_id, field_name="profile_id")
        rtc_mode = _required(self.rtc_mode, field_name="rtc_mode")
        if rtc_mode not in POLICY_INFERENCE_MODES_V1:
            raise ValueError(f"Unsupported RTC mode: {rtc_mode!r}.")
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "rtc_mode", rtc_mode)
        for name in ("observation_hz", "decision_hz", "action_hz"):
            object.__setattr__(
                self,
                name,
                _positive_rate(getattr(self, name), field_name=name),
            )
        if not isinstance(self.position_rates, PositionControlRates):
            raise TypeError("position_rates must be PositionControlRates.")
        if self.position_rates.ordering_warnings:
            raise ValueError("position_rates must preserve sample/reference/command order.")
        if self.position_rates.reference_hz < self.action_hz:
            raise ValueError("position reference rate cannot be below policy action rate.")
        trigger_mode = _required(
            self.request_trigger_mode,
            field_name="request_trigger_mode",
        )
        due_pop_mode = _required(self.due_pop_mode, field_name="due_pop_mode")
        if trigger_mode != POLICY_REQUEST_TRIGGER_COVERAGE_SINGLE_IN_FLIGHT_V1:
            raise ValueError(f"Unsupported request trigger mode: {trigger_mode!r}.")
        if due_pop_mode != POLICY_DUE_POP_LATEST_VALID_NO_CATCHUP_V1:
            raise ValueError(f"Unsupported due-pop mode: {due_pop_mode!r}.")
        object.__setattr__(self, "request_trigger_mode", trigger_mode)
        object.__setattr__(self, "due_pop_mode", due_pop_mode)

        if profile_id in {
            RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1,
            RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1,
            MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1,
        }:
            expected_rates = (15.0, 30.0, 30.0, 15.0, 60.0, 60.0, 120.0)
            actual_rates = (
                self.observation_hz,
                self.decision_hz,
                self.action_hz,
                self.position_rates.sample_hz,
                self.position_rates.reference_hz,
                self.position_rates.ik_hz,
                self.position_rates.command_hz,
            )
            if actual_rates != expected_rates:
                raise ValueError(
                    "RTC final-chunk shadow cadence is fixed at 15/30/60/120 Hz."
                )
            expected_rtc_mode = (
                RTC_FLOW_INPAINT_MODE_V1
                if profile_id
                == MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1
                else ASYNC_ALIGNED_CHUNK_MODE_V1
            )
            if rtc_mode != expected_rtc_mode:
                raise ValueError(
                    "RTC shadow mode does not match its execution profile."
                )
            if self.position_lookahead_enabled:
                raise ValueError(
                    "RTC final-chunk shadow requires position lookahead disabled."
                )
            if (
                profile_id == RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1
                and self.send_actions
            ):
                raise ValueError(
                    "RTC-3c virtual profile has no model action source and does not "
                    "accept --send-actions."
                )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "profile_id": self.profile_id,
                "rtc_mode": self.rtc_mode,
                "observation_hz": self.observation_hz,
                "decision_hz": self.decision_hz,
                "action_hz": self.action_hz,
                "position_rates": self.position_rates.to_metadata(),
                "request_trigger_mode": self.request_trigger_mode,
                "due_pop_mode": self.due_pop_mode,
                "position_lookahead_enabled": self.position_lookahead_enabled,
                "send_actions": self.send_actions,
            },
            domain=POLICY_ASYNC_EXECUTION_PROFILE_HASH_DOMAIN_V1,
        )

    @property
    def producer_profile(self) -> "RtcProducerProfile":
        """Return the target-neutral producer projection.

        The local import avoids an initialization cycle while keeping the
        legacy execution-profile hash and public fields unchanged.
        """

        from .rtc.profile import RtcProducerProfile

        return RtcProducerProfile.from_legacy_execution_profile(self)


RTC_3C_VIRTUAL_SHADOW_PROFILE_V1 = PolicyAsyncExecutionProfile(
    profile_id=RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1,
)

RAW_RTC_FINAL_CHUNK_PROFILE_V1 = PolicyAsyncExecutionProfile(
    profile_id=RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1,
)

MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_V1 = PolicyAsyncExecutionProfile(
    profile_id=MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1,
    rtc_mode=RTC_FLOW_INPAINT_MODE_V1,
)


def model_rtc_flow_inpaint_shadow_profile_for_action_hz(
    action_hz: float | None = None,
) -> PolicyAsyncExecutionProfile:
    """Return the model-RTC producer profile for an explicit runtime cadence.

    The prepared model/artifact contract is unchanged by this helper.  The
    15 Hz variant only changes the client-side keyframe/request period so a
    long but still valid inference has enough H32 suffix coverage.  The
    decision clock and the 60/120 Hz position-control clocks remain unchanged.
    ``None`` and 30 Hz intentionally return the canonical profile used by the
    existing deployment contract.
    """

    if action_hz is None:
        return MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_V1
    normalized = _positive_rate(action_hz, field_name="action_hz")
    if math.isclose(
        normalized,
        MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_V1.action_hz,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_V1
    if not math.isclose(normalized, 15.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "model-RTC runtime action cadence must be 15 or 30 Hz; "
            f"got {normalized!r}."
        )
    return replace(
        MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_V1,
        profile_id=MODEL_RTC_FLOW_INPAINT_SHADOW_RUNTIME_15HZ_PROFILE_ID_V1,
        action_hz=15.0,
    )


@dataclass(frozen=True)
class PolicyAsyncProducerSnapshot:
    """One immutable request/session/plan admission truth."""

    state_id: str
    plan: TimedPolicyActionPlanSnapshot
    lifecycle_state: str
    worker_session_id: str
    worker_session_state: str
    active_request: PolicyInferenceRequest | None
    transition_sequence: int
    updated_at: float
    last_transition_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_id", _required(self.state_id, field_name="state_id"))
        if not isinstance(self.plan, TimedPolicyActionPlanSnapshot):
            raise TypeError("plan must be TimedPolicyActionPlanSnapshot.")
        lifecycle = _required(self.lifecycle_state, field_name="lifecycle_state")
        if lifecycle not in POLICY_ASYNC_PRODUCER_LIFECYCLE_STATES_V1:
            raise ValueError(f"Unsupported producer lifecycle state: {lifecycle!r}.")
        worker_session_id = _required(
            self.worker_session_id,
            field_name="worker_session_id",
        )
        worker_state = _required(
            self.worker_session_state,
            field_name="worker_session_state",
        )
        if worker_state not in POLICY_WORKER_SESSION_STATES_V1:
            raise ValueError(f"Unsupported worker session state: {worker_state!r}.")
        object.__setattr__(self, "lifecycle_state", lifecycle)
        object.__setattr__(self, "worker_session_id", worker_session_id)
        object.__setattr__(self, "worker_session_state", worker_state)
        object.__setattr__(
            self,
            "transition_sequence",
            _nonnegative_integer(
                self.transition_sequence,
                field_name="transition_sequence",
            ),
        )
        object.__setattr__(self, "updated_at", _finite(self.updated_at, field_name="updated_at"))
        if self.last_transition_reason is not None:
            object.__setattr__(
                self,
                "last_transition_reason",
                _required(
                    self.last_transition_reason,
                    field_name="last_transition_reason",
                ),
            )

        request = self.active_request
        if request is None:
            if self.plan.in_flight_request_id is not None:
                raise ValueError("Plan in-flight identity requires an active request payload.")
            return
        if not isinstance(request, PolicyInferenceRequest):
            raise TypeError("active_request must be PolicyInferenceRequest or None.")
        if lifecycle != POLICY_ASYNC_PRODUCER_ELIGIBLE:
            raise ValueError("Only an eligible producer may retain an active request.")
        if worker_state != POLICY_WORKER_SESSION_ACTIVE:
            raise ValueError("An active request requires an active worker session.")
        if self.plan.in_flight_request_id != request.request_id:
            raise ValueError("Active request/plan request identity mismatch.")
        if self.plan.in_flight_inference_id != request.inference_id:
            raise ValueError("Active request/plan inference identity mismatch.")
        identity_matches = (
            self.plan.plan_id == request.source_plan_id
            and self.plan.session_id == request.session_id
            and self.plan.control_domain_id == request.control_domain_id
            and self.plan.producer_id == request.producer_id
            and self.plan.producer_epoch == request.producer_epoch
            and self.plan.policy_node_id == request.policy_node_id
            and self.plan.clock_domain_id == request.clock_domain_id
        )
        if not identity_matches:
            raise ValueError("Active request identity does not match its current plan.")

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "state_id": self.state_id,
                "plan_hash": self.plan.semantic_content_hash,
                "lifecycle_state": self.lifecycle_state,
                "worker_session_id": self.worker_session_id,
                "worker_session_state": self.worker_session_state,
                "active_request_hash": (
                    None
                    if self.active_request is None
                    else self.active_request.semantic_content_hash
                ),
                "transition_sequence": self.transition_sequence,
                "updated_at": self.updated_at,
                "last_transition_reason": self.last_transition_reason,
            },
            domain=POLICY_ASYNC_PRODUCER_STATE_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class PolicyAsyncProducerTransitionReceipt:
    event: str
    accepted: bool
    reasons: tuple[str, ...]
    occurred_at: float
    worker_session_id: str
    request_id: str | None
    result_hash: str | None
    old_state_hash: str
    new_state_hash: str
    plan_merge_receipt_hash: str | None = None
    pop_outcome_hash: str | None = None
    emission_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "event", _required(self.event, field_name="event"))
        reasons = tuple(_required(value, field_name="reason") for value in self.reasons)
        if self.accepted and reasons:
            raise ValueError("Accepted transition cannot carry rejection reasons.")
        if not self.accepted and not reasons:
            raise ValueError("Rejected transition requires at least one reason.")
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(
            self,
            "occurred_at",
            _finite(self.occurred_at, field_name="occurred_at"),
        )
        for name in ("worker_session_id", "old_state_hash", "new_state_hash"):
            object.__setattr__(self, name, _required(getattr(self, name), field_name=name))
        for name in (
            "request_id",
            "result_hash",
            "plan_merge_receipt_hash",
            "pop_outcome_hash",
            "emission_id",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _required(value, field_name=name))

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "event": self.event,
                "accepted": self.accepted,
                "reasons": self.reasons,
                "occurred_at": self.occurred_at,
                "worker_session_id": self.worker_session_id,
                "request_id": self.request_id,
                "result_hash": self.result_hash,
                "old_state_hash": self.old_state_hash,
                "new_state_hash": self.new_state_hash,
                "plan_merge_receipt_hash": self.plan_merge_receipt_hash,
                "pop_outcome_hash": self.pop_outcome_hash,
                "emission_id": self.emission_id,
            },
            domain=POLICY_ASYNC_PRODUCER_TRANSITION_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class PolicyAsyncProducerTransitionOutcome:
    state: PolicyAsyncProducerSnapshot
    receipt: PolicyAsyncProducerTransitionReceipt
    merge_receipt: PolicyChunkMergeReceipt | None = None
    pop_outcome: PolicyActionPlanPopOutcome | None = None


def create_policy_async_producer(
    *,
    state_id: str,
    plan: TimedPolicyActionPlanSnapshot,
    worker_session_id: str,
    created_at: float,
) -> PolicyAsyncProducerSnapshot:
    if plan.in_flight_request_id is not None:
        raise ValueError("Initial async producer plan cannot already be in flight.")
    return PolicyAsyncProducerSnapshot(
        state_id=state_id,
        plan=plan,
        lifecycle_state=POLICY_ASYNC_PRODUCER_ELIGIBLE,
        worker_session_id=worker_session_id,
        worker_session_state=POLICY_WORKER_SESSION_ACTIVE,
        active_request=None,
        transition_sequence=0,
        updated_at=created_at,
        last_transition_reason="created",
    )


def _receipt(
    *,
    event: str,
    accepted: bool,
    reasons: tuple[str, ...],
    occurred_at: float,
    old_state: PolicyAsyncProducerSnapshot,
    new_state: PolicyAsyncProducerSnapshot,
    request_id: str | None = None,
    result_hash: str | None = None,
    merge_receipt: PolicyChunkMergeReceipt | None = None,
    pop_outcome: PolicyActionPlanPopOutcome | None = None,
    emission_id: str | None = None,
) -> PolicyAsyncProducerTransitionReceipt:
    return PolicyAsyncProducerTransitionReceipt(
        event=event,
        accepted=accepted,
        reasons=reasons,
        occurred_at=occurred_at,
        worker_session_id=new_state.worker_session_id,
        request_id=request_id,
        result_hash=result_hash,
        old_state_hash=old_state.semantic_content_hash,
        new_state_hash=new_state.semantic_content_hash,
        plan_merge_receipt_hash=(
            None if merge_receipt is None else merge_receipt.semantic_content_hash
        ),
        pop_outcome_hash=(
            None if pop_outcome is None else pop_outcome.semantic_content_hash
        ),
        emission_id=emission_id,
    )


def admit_policy_inference_request(
    state: PolicyAsyncProducerSnapshot,
    request: PolicyInferenceRequest,
) -> PolicyAsyncProducerTransitionOutcome:
    if not isinstance(state, PolicyAsyncProducerSnapshot):
        raise TypeError("state must be PolicyAsyncProducerSnapshot.")
    if not isinstance(request, PolicyInferenceRequest):
        raise TypeError("request must be PolicyInferenceRequest.")
    reasons: list[str] = []
    if state.lifecycle_state != POLICY_ASYNC_PRODUCER_ELIGIBLE:
        reasons.append("producer_not_eligible")
    if state.worker_session_state != POLICY_WORKER_SESSION_ACTIVE:
        reasons.append("worker_session_not_active")
    if state.active_request is not None:
        reasons.append("active_request_exists")
    if reasons:
        receipt = _receipt(
            event="request_admission",
            accepted=False,
            reasons=tuple(reasons),
            occurred_at=request.admitted_at,
            old_state=state,
            new_state=state,
            request_id=request.request_id,
        )
        return PolicyAsyncProducerTransitionOutcome(state=state, receipt=receipt)
    try:
        plan = mark_policy_action_plan_in_flight(state.plan, request)
    except ValueError:
        receipt = _receipt(
            event="request_admission",
            accepted=False,
            reasons=("request_plan_binding_rejected",),
            occurred_at=request.admitted_at,
            old_state=state,
            new_state=state,
            request_id=request.request_id,
        )
        return PolicyAsyncProducerTransitionOutcome(state=state, receipt=receipt)
    updated = replace(
        state,
        plan=plan,
        active_request=request,
        transition_sequence=state.transition_sequence + 1,
        updated_at=request.admitted_at,
        last_transition_reason="request_admitted",
    )
    receipt = _receipt(
        event="request_admission",
        accepted=True,
        reasons=(),
        occurred_at=request.admitted_at,
        old_state=state,
        new_state=updated,
        request_id=request.request_id,
    )
    return PolicyAsyncProducerTransitionOutcome(state=updated, receipt=receipt)


def admit_policy_chunk_result(
    state: PolicyAsyncProducerSnapshot,
    result: PolicyChunkResult,
    *,
    worker_session_id: str,
    admitted_at: float,
    actual_cursor_at: float | None = None,
    producer_committed_through: float | None = None,
) -> PolicyAsyncProducerTransitionOutcome:
    if not isinstance(state, PolicyAsyncProducerSnapshot):
        raise TypeError("state must be PolicyAsyncProducerSnapshot.")
    if not isinstance(result, PolicyChunkResult):
        raise TypeError("result must be PolicyChunkResult.")
    session_id = _required(worker_session_id, field_name="worker_session_id")
    admitted = _finite(admitted_at, field_name="admitted_at")
    request = state.active_request
    reasons: list[str] = []
    if state.lifecycle_state != POLICY_ASYNC_PRODUCER_ELIGIBLE:
        reasons.append("producer_not_eligible")
    if session_id != state.worker_session_id:
        reasons.append("worker_session_mismatch")
    if state.worker_session_state != POLICY_WORKER_SESSION_ACTIVE:
        reasons.append("worker_session_not_active")
    if request is None:
        reasons.append("no_active_request")
    elif (
        result.request_id != request.request_id
        or result.inference_id != request.inference_id
        or result.request_hash != request.semantic_content_hash
    ):
        reasons.append("result_not_active_request")
    if reasons:
        receipt = _receipt(
            event="result_admission",
            accepted=False,
            reasons=tuple(reasons),
            occurred_at=admitted,
            old_state=state,
            new_state=state,
            request_id=result.request_id,
            result_hash=result.semantic_content_hash,
        )
        return PolicyAsyncProducerTransitionOutcome(state=state, receipt=receipt)

    assert request is not None
    merge = merge_policy_chunk_result(
        state.plan,
        request,
        result,
        admitted_at=admitted,
        actual_cursor_at=actual_cursor_at,
        producer_committed_through=producer_committed_through,
    )
    active_request = (
        request if merge.plan.in_flight_request_id is not None else None
    )
    updated = replace(
        state,
        plan=merge.plan,
        active_request=active_request,
        transition_sequence=state.transition_sequence + 1,
        updated_at=admitted,
        last_transition_reason=(
            "result_admitted" if merge.receipt.accepted else "result_rejected"
        ),
    )
    receipt = _receipt(
        event="result_admission",
        accepted=merge.receipt.accepted,
        reasons=merge.receipt.reasons,
        occurred_at=admitted,
        old_state=state,
        new_state=updated,
        request_id=request.request_id,
        result_hash=result.semantic_content_hash,
        merge_receipt=merge.receipt,
    )
    return PolicyAsyncProducerTransitionOutcome(
        state=updated,
        receipt=receipt,
        merge_receipt=merge.receipt,
    )


def invalidate_policy_async_producer(
    state: PolicyAsyncProducerSnapshot,
    *,
    lifecycle_state: str,
    invalidated_at: float,
    reason: str,
    worker_session_state: str | None = None,
) -> PolicyAsyncProducerTransitionOutcome:
    if not isinstance(state, PolicyAsyncProducerSnapshot):
        raise TypeError("state must be PolicyAsyncProducerSnapshot.")
    lifecycle = _required(lifecycle_state, field_name="lifecycle_state")
    if lifecycle == POLICY_ASYNC_PRODUCER_ELIGIBLE:
        raise ValueError("Invalidation must leave the producer in a non-eligible state.")
    if lifecycle not in POLICY_ASYNC_PRODUCER_LIFECYCLE_STATES_V1:
        raise ValueError(f"Unsupported producer lifecycle state: {lifecycle!r}.")
    occurred = _finite(invalidated_at, field_name="invalidated_at")
    transition_reason = _required(reason, field_name="reason")
    session_state = (
        state.worker_session_state
        if worker_session_state is None
        else _required(worker_session_state, field_name="worker_session_state")
    )
    if session_state not in POLICY_WORKER_SESSION_STATES_V1:
        raise ValueError(f"Unsupported worker session state: {session_state!r}.")
    next_plan = empty_policy_action_plan(
        plan_id=state.plan.plan_id,
        session_id=state.plan.session_id,
        control_domain_id=state.plan.control_domain_id,
        producer_id=state.plan.producer_id,
        producer_epoch=state.plan.producer_epoch + 1,
        policy_node_id=state.plan.policy_node_id,
        clock_domain_id=state.plan.clock_domain_id,
        cursor_at=max(state.plan.cursor_at, occurred),
        created_at=occurred,
    )
    previous_request_id = (
        None if state.active_request is None else state.active_request.request_id
    )
    updated = replace(
        state,
        plan=next_plan,
        lifecycle_state=lifecycle,
        worker_session_state=session_state,
        active_request=None,
        transition_sequence=state.transition_sequence + 1,
        updated_at=occurred,
        last_transition_reason=transition_reason,
    )
    receipt = _receipt(
        event="producer_invalidation",
        accepted=True,
        reasons=(),
        occurred_at=occurred,
        old_state=state,
        new_state=updated,
        request_id=previous_request_id,
    )
    return PolicyAsyncProducerTransitionOutcome(state=updated, receipt=receipt)


def update_policy_worker_session_state(
    state: PolicyAsyncProducerSnapshot,
    *,
    worker_session_state: str,
    occurred_at: float,
    reason: str,
) -> PolicyAsyncProducerTransitionOutcome:
    if state.active_request is not None:
        raise ValueError("Invalidate the producer before retiring an active worker session.")
    session_state = _required(worker_session_state, field_name="worker_session_state")
    if session_state not in POLICY_WORKER_SESSION_STATES_V1:
        raise ValueError(f"Unsupported worker session state: {session_state!r}.")
    occurred = _finite(occurred_at, field_name="occurred_at")
    transition_reason = _required(reason, field_name="reason")
    updated = replace(
        state,
        worker_session_state=session_state,
        transition_sequence=state.transition_sequence + 1,
        updated_at=occurred,
        last_transition_reason=transition_reason,
    )
    receipt = _receipt(
        event="worker_session_transition",
        accepted=True,
        reasons=(),
        occurred_at=occurred,
        old_state=state,
        new_state=updated,
    )
    return PolicyAsyncProducerTransitionOutcome(state=updated, receipt=receipt)


def rearm_policy_async_producer(
    state: PolicyAsyncProducerSnapshot,
    *,
    worker_session_id: str,
    rearmed_at: float,
    reason: str,
) -> PolicyAsyncProducerTransitionOutcome:
    if state.lifecycle_state == POLICY_ASYNC_PRODUCER_ELIGIBLE:
        raise ValueError("Producer is already eligible.")
    if state.active_request is not None or state.plan.in_flight_request_id is not None:
        raise ValueError("Cannot rearm a producer with an active request.")
    if state.plan.keyframes:
        raise ValueError("Rearm requires an invalidated empty plan.")
    session_id = _required(worker_session_id, field_name="worker_session_id")
    occurred = _finite(rearmed_at, field_name="rearmed_at")
    transition_reason = _required(reason, field_name="reason")
    updated = replace(
        state,
        lifecycle_state=POLICY_ASYNC_PRODUCER_ELIGIBLE,
        worker_session_id=session_id,
        worker_session_state=POLICY_WORKER_SESSION_ACTIVE,
        transition_sequence=state.transition_sequence + 1,
        updated_at=occurred,
        last_transition_reason=transition_reason,
    )
    receipt = _receipt(
        event="producer_rearm",
        accepted=True,
        reasons=(),
        occurred_at=occurred,
        old_state=state,
        new_state=updated,
    )
    return PolicyAsyncProducerTransitionOutcome(state=updated, receipt=receipt)


def pop_due_policy_async_producer_action(
    state: PolicyAsyncProducerSnapshot,
    *,
    decision_at: float,
) -> PolicyAsyncProducerTransitionOutcome:
    if not isinstance(state, PolicyAsyncProducerSnapshot):
        raise TypeError("state must be PolicyAsyncProducerSnapshot.")
    decision = _finite(decision_at, field_name="decision_at")
    if state.lifecycle_state != POLICY_ASYNC_PRODUCER_ELIGIBLE:
        receipt = _receipt(
            event="due_pop",
            accepted=False,
            reasons=("producer_not_eligible",),
            occurred_at=decision,
            old_state=state,
            new_state=state,
        )
        return PolicyAsyncProducerTransitionOutcome(state=state, receipt=receipt)
    pop = pop_due_policy_action(state.plan, decision_at=decision)
    updated = replace(
        state,
        plan=pop.plan,
        transition_sequence=state.transition_sequence + 1,
        updated_at=decision,
        last_transition_reason=(
            "due_keyframe_selected" if pop.keyframe is not None else "no_due_keyframe"
        ),
    )
    receipt = _receipt(
        event="due_pop",
        accepted=True,
        reasons=(),
        occurred_at=decision,
        old_state=state,
        new_state=updated,
        request_id=None if pop.keyframe is None else pop.keyframe.request_id,
        pop_outcome=pop,
    )
    return PolicyAsyncProducerTransitionOutcome(
        state=updated,
        receipt=receipt,
        pop_outcome=pop,
    )


def commit_emitted_policy_async_producer_action(
    state: PolicyAsyncProducerSnapshot,
    *,
    pop_outcome: PolicyActionPlanPopOutcome,
    emission_id: str,
    committed_at: float,
) -> PolicyAsyncProducerTransitionOutcome:
    """Advance commit-through only after a popped keyframe has an emission.

    RTC-3c has no preview/lookahead admission. This transition may protect
    only a target that the plan cursor has already popped; it never stages or
    commits an un-popped successor.
    """

    if not isinstance(state, PolicyAsyncProducerSnapshot):
        raise TypeError("state must be PolicyAsyncProducerSnapshot.")
    if not isinstance(pop_outcome, PolicyActionPlanPopOutcome):
        raise TypeError("pop_outcome must be PolicyActionPlanPopOutcome.")
    formal_emission_id = _required(emission_id, field_name="emission_id")
    occurred = _finite(committed_at, field_name="committed_at")
    reasons: list[str] = []
    if state.lifecycle_state != POLICY_ASYNC_PRODUCER_ELIGIBLE:
        reasons.append("producer_not_eligible")
    if pop_outcome.plan.semantic_content_hash != state.plan.semantic_content_hash:
        reasons.append("pop_state_mismatch")
    selected = pop_outcome.keyframe
    if selected is None:
        reasons.append("pop_has_no_selected_keyframe")
        source_request_id = None
        target = None
    else:
        source_request_id = selected.request_id
        target = selected.target_effective_at
        if target > state.plan.cursor_at:
            reasons.append("target_not_popped")
    boundary = state.plan.producer_committed_through
    if target is not None and boundary is not None and target <= boundary:
        reasons.append("commit_through_not_advancing")
    if reasons:
        receipt = _receipt(
            event="emission_commit",
            accepted=False,
            reasons=tuple(reasons),
            occurred_at=occurred,
            old_state=state,
            new_state=state,
            request_id=source_request_id,
            pop_outcome=pop_outcome,
            emission_id=formal_emission_id,
        )
        return PolicyAsyncProducerTransitionOutcome(state=state, receipt=receipt)

    assert target is not None
    plan = commit_policy_action_plan_through(
        state.plan,
        committed_through=target,
        committed_at=occurred,
    )
    updated = replace(
        state,
        plan=plan,
        transition_sequence=state.transition_sequence + 1,
        updated_at=occurred,
        last_transition_reason="emission_committed",
    )
    receipt = _receipt(
        event="emission_commit",
        accepted=True,
        reasons=(),
        occurred_at=occurred,
        old_state=state,
        new_state=updated,
        request_id=source_request_id,
        pop_outcome=pop_outcome,
        emission_id=formal_emission_id,
    )
    return PolicyAsyncProducerTransitionOutcome(state=updated, receipt=receipt)


__all__ = [
    "POLICY_ASYNC_EXECUTION_PROFILE_HASH_DOMAIN_V1",
    "POLICY_ASYNC_PRODUCER_ELIGIBLE",
    "POLICY_ASYNC_PRODUCER_FAULT",
    "POLICY_ASYNC_PRODUCER_HOLD",
    "POLICY_ASYNC_PRODUCER_LIFECYCLE_STATES_V1",
    "POLICY_ASYNC_PRODUCER_STATE_HASH_DOMAIN_V1",
    "POLICY_ASYNC_PRODUCER_TRANSITION_HASH_DOMAIN_V1",
    "POLICY_ASYNC_PRODUCER_TRANSITIONING",
    "POLICY_DUE_POP_LATEST_VALID_NO_CATCHUP_V1",
    "POLICY_REQUEST_TRIGGER_COVERAGE_SINGLE_IN_FLIGHT_V1",
    "POLICY_WORKER_SESSION_ABANDONED_UNDRAINED",
    "POLICY_WORKER_SESSION_ACTIVE",
    "POLICY_WORKER_SESSION_DRAINED",
    "POLICY_WORKER_SESSION_RETIRED_WAITING_FOR_COMPLETION",
    "POLICY_WORKER_SESSION_STATES_V1",
    "RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1",
    "RTC_3C_VIRTUAL_SHADOW_PROFILE_V1",
    "RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1",
    "RAW_RTC_FINAL_CHUNK_PROFILE_V1",
    "MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1",
    "MODEL_RTC_FLOW_INPAINT_SHADOW_RUNTIME_15HZ_PROFILE_ID_V1",
    "MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_V1",
    "model_rtc_flow_inpaint_shadow_profile_for_action_hz",
    "PolicyAsyncExecutionProfile",
    "PolicyAsyncProducerSnapshot",
    "PolicyAsyncProducerTransitionOutcome",
    "PolicyAsyncProducerTransitionReceipt",
    "admit_policy_chunk_result",
    "admit_policy_inference_request",
    "commit_emitted_policy_async_producer_action",
    "create_policy_async_producer",
    "invalidate_policy_async_producer",
    "pop_due_policy_async_producer_action",
    "rearm_policy_async_producer",
    "update_policy_worker_session_state",
]
