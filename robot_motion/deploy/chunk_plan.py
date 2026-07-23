"""Immutable target-time policy action plan and deterministic chunk merge.

The plan is producer-local preparation state.  It is not an action proposal,
shared reference segment, safety decision, execution work item, or backend
receipt.  Only a caller that already owns the control-thread admission point
may swap the returned snapshot into runtime state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action.snapshot import ActionEmissionScheduleSpec

from .actions import NormalizedAction
from .canonical_policy_action import CanonicalPolicyAction
from .inference import (
    POLICY_ACTION_PAYLOAD_LEGACY_NORMALIZED_V1,
    POLICY_ACTION_PAYLOAD_TYPED_CANONICAL_V1,
    RTC_FLOW_INPAINT_MODE_V1,
    PolicyChunkResult,
    PolicyInferenceRequest,
    _freeze_normalized_action,
    _normalized_action_semantic_payload,
)


POLICY_PLAN_KEYFRAME_MUTABLE = "mutable"
POLICY_PLAN_KEYFRAME_PRODUCER_COMMITTED = "producer_committed"
POLICY_PLAN_KEYFRAME_STATES_V1 = (
    POLICY_PLAN_KEYFRAME_MUTABLE,
    POLICY_PLAN_KEYFRAME_PRODUCER_COMMITTED,
)

REPLACE_MUTABLE_FUTURE_MERGE_V1 = "replace_mutable_future_v1"
POLICY_CHUNK_MERGE_SPEC_HASH_DOMAIN_V1 = "dexteleop.policy_chunk_merge_spec.v1"
TIMED_POLICY_ACTION_KEYFRAME_HASH_DOMAIN_V1 = (
    "dexteleop.timed_policy_action_keyframe.v1"
)
TIMED_POLICY_ACTION_PLAN_HASH_DOMAIN_V1 = "dexteleop.timed_policy_action_plan.v1"
TIMED_POLICY_ACTION_KEYFRAME_HASH_DOMAIN_V2 = (
    "dexteleop.timed_policy_action_keyframe.v2"
)
TIMED_POLICY_ACTION_KEYFRAME_HASH_DOMAIN_V3 = (
    "dexteleop.timed_policy_action_keyframe.v3"
)
TIMED_POLICY_ACTION_PLAN_HASH_DOMAIN_V2 = "dexteleop.timed_policy_action_plan.v2"
POLICY_CHUNK_MERGE_RECEIPT_HASH_DOMAIN_V1 = (
    "dexteleop.policy_chunk_merge_receipt.v1"
)
POLICY_ACTION_PLAN_POP_OUTCOME_HASH_DOMAIN_V1 = (
    "dexteleop.policy_action_plan_pop_outcome.v1"
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


def _maximum_optional(
    first: float | None,
    second: float | None,
) -> float | None:
    values = tuple(value for value in (first, second) if value is not None)
    return max(values) if values else None


@dataclass(frozen=True)
class PolicyChunkMergeSpec:
    """Versioned producer-plan merge semantics."""

    spec_id: str = REPLACE_MUTABLE_FUTURE_MERGE_V1
    maximum_new_keyframes: int | None = None
    require_minimum_execution_horizon: bool = True

    def __post_init__(self) -> None:
        spec_id = _nonempty(self.spec_id, name="spec_id")
        if spec_id != REPLACE_MUTABLE_FUTURE_MERGE_V1:
            raise ValueError(f"Unsupported policy chunk merge spec: {spec_id!r}.")
        object.__setattr__(self, "spec_id", spec_id)
        if self.maximum_new_keyframes is not None:
            maximum = _nonnegative_int(
                self.maximum_new_keyframes,
                name="maximum_new_keyframes",
            )
            if maximum <= 0:
                raise ValueError("maximum_new_keyframes must be positive when set.")
            object.__setattr__(self, "maximum_new_keyframes", maximum)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "spec_id": self.spec_id,
                "maximum_new_keyframes": self.maximum_new_keyframes,
                "require_minimum_execution_horizon": self.require_minimum_execution_horizon,
            },
            domain=POLICY_CHUNK_MERGE_SPEC_HASH_DOMAIN_V1,
        )


DEFAULT_POLICY_CHUNK_MERGE_SPEC_V1 = PolicyChunkMergeSpec()


@dataclass(frozen=True)
class TimedPolicyActionKeyframe:
    target_effective_at: float
    effective_until: float
    effective_clock_domain_id: str
    action_period_sec: float
    decoded_absolute_action: NormalizedAction | None
    request_id: str
    inference_id: str
    policy_output_id: str
    chunk_id: int
    chunk_keyframe_index: int
    chunk_size: int
    observation_cut_id: str
    producer_epoch: int
    policy_node_id: str
    artifact_hash: str
    produced_at: float
    schedule_spec_id: str
    schedule_spec_hash: str
    admitted_plan_generation: int
    state: str
    valid_until: float
    canonical_action: CanonicalPolicyAction | None = None

    def __post_init__(self) -> None:
        for name in (
            "target_effective_at",
            "effective_until",
            "action_period_sec",
            "produced_at",
            "valid_until",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name=name))
        if self.action_period_sec <= 0:
            raise ValueError("action_period_sec must be positive.")
        if self.effective_until <= self.target_effective_at:
            raise ValueError("A keyframe effective interval must be non-empty.")
        if self.valid_until < self.effective_until:
            raise ValueError("valid_until must cover the keyframe effective interval.")
        for name in (
            "effective_clock_domain_id",
            "request_id",
            "inference_id",
            "policy_output_id",
            "observation_cut_id",
            "policy_node_id",
            "artifact_hash",
            "schedule_spec_id",
            "schedule_spec_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name=name))
        for name in (
            "chunk_id",
            "chunk_keyframe_index",
            "producer_epoch",
            "admitted_plan_generation",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name),
            )
        chunk_size = _nonnegative_int(self.chunk_size, name="chunk_size")
        if chunk_size <= 0 or self.chunk_keyframe_index >= chunk_size:
            raise ValueError("chunk keyframe index/size is invalid.")
        object.__setattr__(self, "chunk_size", chunk_size)
        state = _nonempty(self.state, name="state")
        if state not in POLICY_PLAN_KEYFRAME_STATES_V1:
            raise ValueError(f"Unsupported policy plan keyframe state: {state!r}.")
        object.__setattr__(self, "state", state)
        if (self.decoded_absolute_action is None) == (self.canonical_action is None):
            raise ValueError(
                "A timed keyframe must carry exactly one action payload representation."
            )
        if self.decoded_absolute_action is not None:
            object.__setattr__(
                self,
                "decoded_absolute_action",
                _freeze_normalized_action(self.decoded_absolute_action),
            )
        elif not isinstance(self.canonical_action, CanonicalPolicyAction):
            raise TypeError("canonical_action must be a CanonicalPolicyAction.")

    @property
    def action_payload_kind(self) -> str:
        return (
            POLICY_ACTION_PAYLOAD_TYPED_CANONICAL_V1
            if self.canonical_action is not None
            else POLICY_ACTION_PAYLOAD_LEGACY_NORMALIZED_V1
        )

    @property
    def semantic_content_hash(self) -> str:
        payload = {
            "target_effective_at": self.target_effective_at,
            "effective_until": self.effective_until,
            "effective_clock_domain_id": self.effective_clock_domain_id,
            "action_period_sec": self.action_period_sec,
            "request_id": self.request_id,
            "inference_id": self.inference_id,
            "policy_output_id": self.policy_output_id,
            "chunk_id": self.chunk_id,
            "chunk_keyframe_index": self.chunk_keyframe_index,
            "chunk_size": self.chunk_size,
            "observation_cut_id": self.observation_cut_id,
            "producer_epoch": self.producer_epoch,
            "policy_node_id": self.policy_node_id,
            "artifact_hash": self.artifact_hash,
            "produced_at": self.produced_at,
            "schedule_spec_id": self.schedule_spec_id,
            "schedule_spec_hash": self.schedule_spec_hash,
            "admitted_plan_generation": self.admitted_plan_generation,
            "state": self.state,
            "valid_until": self.valid_until,
        }
        if self.canonical_action is None:
            payload["decoded_absolute_action"] = _normalized_action_semantic_payload(
                self.decoded_absolute_action
            )
            domain = TIMED_POLICY_ACTION_KEYFRAME_HASH_DOMAIN_V2
        else:
            payload["action_payload_kind"] = self.action_payload_kind
            payload["canonical_action_hash"] = self.canonical_action.semantic_content_hash
            domain = TIMED_POLICY_ACTION_KEYFRAME_HASH_DOMAIN_V3
        return canonical_content_hash(
            payload,
            domain=domain,
        )


@dataclass(frozen=True)
class TimedPolicyActionPlanSnapshot:
    plan_id: str
    session_id: str
    control_domain_id: str
    producer_id: str
    producer_epoch: int
    policy_node_id: str
    clock_domain_id: str
    generation: int
    cursor_at: float
    created_at: float
    keyframes: tuple[TimedPolicyActionKeyframe, ...]
    producer_committed_through: float | None = None
    in_flight_request_id: str | None = None
    in_flight_inference_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "plan_id",
            "session_id",
            "control_domain_id",
            "producer_id",
            "policy_node_id",
            "clock_domain_id",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name=name))
        for name in ("producer_epoch", "generation"):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name),
            )
        object.__setattr__(self, "cursor_at", _finite(self.cursor_at, name="cursor_at"))
        object.__setattr__(self, "created_at", _finite(self.created_at, name="created_at"))
        boundary = self.producer_committed_through
        if boundary is not None:
            boundary = _finite(boundary, name="producer_committed_through")
            object.__setattr__(self, "producer_committed_through", boundary)
        if (self.in_flight_request_id is None) != (self.in_flight_inference_id is None):
            raise ValueError("In-flight request and inference ids must be set together.")
        if self.in_flight_request_id is not None:
            object.__setattr__(
                self,
                "in_flight_request_id",
                _nonempty(self.in_flight_request_id, name="in_flight_request_id"),
            )
            object.__setattr__(
                self,
                "in_flight_inference_id",
                _nonempty(self.in_flight_inference_id or "", name="in_flight_inference_id"),
            )
        keyframes = tuple(self.keyframes)
        object.__setattr__(self, "keyframes", keyframes)
        previous_target: float | None = None
        saw_mutable = False
        action_payload_kind: str | None = None
        for keyframe in keyframes:
            if keyframe.effective_clock_domain_id != self.clock_domain_id:
                raise ValueError("Plan/keyframe clock domain mismatch.")
            if keyframe.producer_epoch != self.producer_epoch:
                raise ValueError("Plan/keyframe producer epoch mismatch.")
            if keyframe.policy_node_id != self.policy_node_id:
                raise ValueError("Plan/keyframe policy node mismatch.")
            if keyframe.admitted_plan_generation > self.generation:
                raise ValueError("Keyframe admission generation is newer than its plan.")
            if keyframe.target_effective_at < self.cursor_at:
                raise ValueError("A future plan cannot contain a keyframe before its cursor.")
            if action_payload_kind is None:
                action_payload_kind = keyframe.action_payload_kind
            elif keyframe.action_payload_kind != action_payload_kind:
                raise ValueError("A policy plan cannot mix action payload representations.")
            if previous_target is not None and keyframe.target_effective_at <= previous_target:
                raise ValueError("Policy plan target times must be strictly increasing.")
            previous_target = keyframe.target_effective_at
            if keyframe.state == POLICY_PLAN_KEYFRAME_MUTABLE:
                saw_mutable = True
                if boundary is not None and keyframe.target_effective_at <= boundary:
                    raise ValueError("A keyframe at/before committed-through cannot be mutable.")
            else:
                if saw_mutable:
                    raise ValueError("Producer-committed keyframes must form a plan prefix.")
                if boundary is None or keyframe.target_effective_at > boundary:
                    raise ValueError("Committed keyframe lies beyond committed-through.")

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "plan_id": self.plan_id,
                "session_id": self.session_id,
                "control_domain_id": self.control_domain_id,
                "producer_id": self.producer_id,
                "producer_epoch": self.producer_epoch,
                "policy_node_id": self.policy_node_id,
                "clock_domain_id": self.clock_domain_id,
                "generation": self.generation,
                "cursor_at": self.cursor_at,
                "created_at": self.created_at,
                "keyframe_hashes": tuple(
                    keyframe.semantic_content_hash for keyframe in self.keyframes
                ),
                "producer_committed_through": self.producer_committed_through,
                "in_flight_request_id": self.in_flight_request_id,
                "in_flight_inference_id": self.in_flight_inference_id,
            },
            domain=TIMED_POLICY_ACTION_PLAN_HASH_DOMAIN_V2,
        )


@dataclass(frozen=True)
class PolicyChunkMergeReceipt:
    accepted: bool
    reasons: tuple[str, ...]
    request_id: str
    inference_id: str
    result_hash: str
    admitted_at: float
    clock_domain_id: str
    old_plan_generation: int
    old_plan_hash: str
    new_plan_generation: int
    new_plan_hash: str
    actual_cursor_at: float
    actual_committed_through: float | None
    preserved_committed_count: int
    preserved_mutable_count: int
    discarded_obsolete_count: int
    discarded_beyond_horizon_count: int
    replaced_mutable_count: int
    admitted_keyframe_count: int
    replaced_mutable_range: tuple[float, float] | None
    new_future_coverage: tuple[float, float] | None
    merge_spec_id: str
    merge_spec_hash: str

    def __post_init__(self) -> None:
        reasons = tuple(_nonempty(str(value), name="merge reason") for value in self.reasons)
        if self.accepted and reasons:
            raise ValueError("An accepted merge receipt cannot carry rejection reasons.")
        if not self.accepted and not reasons:
            raise ValueError("A rejected merge receipt requires at least one reason.")
        object.__setattr__(self, "reasons", reasons)
        for name in (
            "request_id",
            "inference_id",
            "result_hash",
            "clock_domain_id",
            "old_plan_hash",
            "new_plan_hash",
            "merge_spec_id",
            "merge_spec_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name=name))
        for name in ("admitted_at", "actual_cursor_at"):
            object.__setattr__(self, name, _finite(getattr(self, name), name=name))
        if self.actual_committed_through is not None:
            object.__setattr__(
                self,
                "actual_committed_through",
                _finite(
                    self.actual_committed_through,
                    name="actual_committed_through",
                ),
            )
        for name in (
            "old_plan_generation",
            "new_plan_generation",
            "preserved_committed_count",
            "preserved_mutable_count",
            "discarded_obsolete_count",
            "discarded_beyond_horizon_count",
            "replaced_mutable_count",
            "admitted_keyframe_count",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name),
            )
        for name in ("replaced_mutable_range", "new_future_coverage"):
            interval = getattr(self, name)
            if interval is None:
                continue
            normalized = tuple(_finite(value, name=name) for value in interval)
            if len(normalized) != 2 or normalized[1] <= normalized[0]:
                raise ValueError(f"{name} must be a non-empty time interval.")
            object.__setattr__(self, name, normalized)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "accepted": self.accepted,
                "reasons": self.reasons,
                "request_id": self.request_id,
                "inference_id": self.inference_id,
                "result_hash": self.result_hash,
                "admitted_at": self.admitted_at,
                "clock_domain_id": self.clock_domain_id,
                "old_plan_generation": self.old_plan_generation,
                "old_plan_hash": self.old_plan_hash,
                "new_plan_generation": self.new_plan_generation,
                "new_plan_hash": self.new_plan_hash,
                "actual_cursor_at": self.actual_cursor_at,
                "actual_committed_through": self.actual_committed_through,
                "preserved_committed_count": self.preserved_committed_count,
                "preserved_mutable_count": self.preserved_mutable_count,
                "discarded_obsolete_count": self.discarded_obsolete_count,
                "discarded_beyond_horizon_count": self.discarded_beyond_horizon_count,
                "replaced_mutable_count": self.replaced_mutable_count,
                "admitted_keyframe_count": self.admitted_keyframe_count,
                "replaced_mutable_range": self.replaced_mutable_range,
                "new_future_coverage": self.new_future_coverage,
                "merge_spec_id": self.merge_spec_id,
                "merge_spec_hash": self.merge_spec_hash,
            },
            domain=POLICY_CHUNK_MERGE_RECEIPT_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class PolicyChunkMergeOutcome:
    plan: TimedPolicyActionPlanSnapshot
    receipt: PolicyChunkMergeReceipt


@dataclass(frozen=True)
class PolicyActionPlanPopOutcome:
    plan: TimedPolicyActionPlanSnapshot
    keyframe: TimedPolicyActionKeyframe | None
    discarded_expired_count: int
    decision_at: float
    selected_target_effective_at: float | None
    oldest_skipped_target_effective_at: float | None
    selected_lateness_sec: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.plan, TimedPolicyActionPlanSnapshot):
            raise TypeError("plan must be TimedPolicyActionPlanSnapshot.")
        if self.keyframe is not None and not isinstance(
            self.keyframe,
            TimedPolicyActionKeyframe,
        ):
            raise TypeError("keyframe must be TimedPolicyActionKeyframe or None.")
        object.__setattr__(
            self,
            "discarded_expired_count",
            _nonnegative_int(
                self.discarded_expired_count,
                name="discarded_expired_count",
            ),
        )
        object.__setattr__(self, "decision_at", _finite(self.decision_at, name="decision_at"))
        for name in (
            "selected_target_effective_at",
            "oldest_skipped_target_effective_at",
            "selected_lateness_sec",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite(value, name=name))
        if self.keyframe is None:
            if self.selected_target_effective_at is not None:
                raise ValueError("No selected keyframe can expose a selected target.")
            if self.selected_lateness_sec is not None:
                raise ValueError("No selected keyframe can expose selected lateness.")
        else:
            if self.selected_target_effective_at != self.keyframe.target_effective_at:
                raise ValueError("Selected target must match the selected keyframe.")
            expected_lateness = self.decision_at - self.keyframe.target_effective_at
            if self.selected_lateness_sec != expected_lateness:
                raise ValueError("Selected lateness must match decision-target time.")
            if expected_lateness < 0.0:
                raise ValueError("A selected due keyframe cannot be early.")
        if self.discarded_expired_count == 0:
            if self.oldest_skipped_target_effective_at is not None:
                raise ValueError("No skipped target may be reported when none were discarded.")
        elif self.oldest_skipped_target_effective_at is None:
            raise ValueError("Discarded due keyframes require the oldest skipped target.")

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "plan_hash": self.plan.semantic_content_hash,
                "selected_keyframe_hash": (
                    None if self.keyframe is None else self.keyframe.semantic_content_hash
                ),
                "discarded_expired_count": self.discarded_expired_count,
                "decision_at": self.decision_at,
                "selected_target_effective_at": self.selected_target_effective_at,
                "oldest_skipped_target_effective_at": (
                    self.oldest_skipped_target_effective_at
                ),
                "selected_lateness_sec": self.selected_lateness_sec,
            },
            domain=POLICY_ACTION_PLAN_POP_OUTCOME_HASH_DOMAIN_V1,
        )


def empty_policy_action_plan(
    *,
    plan_id: str,
    session_id: str,
    control_domain_id: str,
    producer_id: str,
    producer_epoch: int,
    policy_node_id: str,
    clock_domain_id: str,
    cursor_at: float,
    created_at: float,
) -> TimedPolicyActionPlanSnapshot:
    return TimedPolicyActionPlanSnapshot(
        plan_id=plan_id,
        session_id=session_id,
        control_domain_id=control_domain_id,
        producer_id=producer_id,
        producer_epoch=producer_epoch,
        policy_node_id=policy_node_id,
        clock_domain_id=clock_domain_id,
        generation=0,
        cursor_at=cursor_at,
        created_at=created_at,
        keyframes=(),
    )


def _request_plan_identity_reasons(
    plan: TimedPolicyActionPlanSnapshot,
    request: PolicyInferenceRequest,
) -> tuple[str, ...]:
    checks = (
        (plan.plan_id == request.source_plan_id, "source_plan_id_mismatch"),
        (plan.session_id == request.session_id, "session_mismatch"),
        (plan.control_domain_id == request.control_domain_id, "control_domain_mismatch"),
        (plan.producer_id == request.producer_id, "producer_mismatch"),
        (plan.producer_epoch == request.producer_epoch, "producer_epoch_mismatch"),
        (plan.policy_node_id == request.policy_node_id, "policy_node_mismatch"),
        (plan.clock_domain_id == request.clock_domain_id, "clock_domain_mismatch"),
    )
    return tuple(reason for matches, reason in checks if not matches)


def mark_policy_action_plan_in_flight(
    plan: TimedPolicyActionPlanSnapshot,
    request: PolicyInferenceRequest,
) -> TimedPolicyActionPlanSnapshot:
    """Bind the single admitted request to the current immutable plan cut."""

    reasons = _request_plan_identity_reasons(plan, request)
    if reasons:
        raise ValueError(f"Request/plan identity mismatch: {', '.join(reasons)}.")
    if plan.in_flight_request_id is not None:
        raise ValueError("The policy plan already has an in-flight request.")
    if request.source_plan_generation != plan.generation:
        raise ValueError("Request source plan generation is stale.")
    if request.source_plan_hash != plan.semantic_content_hash:
        raise ValueError("Request source plan hash mismatch.")
    if request.source_plan_cursor_at != plan.cursor_at:
        raise ValueError("Request source plan cursor mismatch.")
    if request.producer_plan_committed_through != plan.producer_committed_through:
        raise ValueError("Request source committed-through mismatch.")
    return replace(
        plan,
        generation=plan.generation + 1,
        created_at=request.admitted_at,
        in_flight_request_id=request.request_id,
        in_flight_inference_id=request.inference_id,
    )


def commit_policy_action_plan_through(
    plan: TimedPolicyActionPlanSnapshot,
    *,
    committed_through: float,
    committed_at: float,
) -> TimedPolicyActionPlanSnapshot:
    """Protect a prefix after a downstream compatibility admission receipt."""

    boundary = _finite(committed_through, name="committed_through")
    timestamp = _finite(committed_at, name="committed_at")
    if (
        plan.producer_committed_through is not None
        and boundary < plan.producer_committed_through
    ):
        raise ValueError("Producer committed-through must be monotonic.")
    keyframes = tuple(
        replace(keyframe, state=POLICY_PLAN_KEYFRAME_PRODUCER_COMMITTED)
        if keyframe.target_effective_at <= boundary
        else keyframe
        for keyframe in plan.keyframes
    )
    if boundary == plan.producer_committed_through and keyframes == plan.keyframes:
        return plan
    return replace(
        plan,
        generation=plan.generation + 1,
        created_at=timestamp,
        keyframes=keyframes,
        producer_committed_through=boundary,
    )


def pop_due_policy_action(
    plan: TimedPolicyActionPlanSnapshot,
    *,
    decision_at: float,
) -> PolicyActionPlanPopOutcome:
    """Remove elapsed keyframes and return at most one action for this decision time."""

    decision = _finite(decision_at, name="decision_at")
    if decision < plan.cursor_at:
        raise ValueError("decision_at cannot move the plan cursor backwards.")
    elapsed = tuple(
        keyframe for keyframe in plan.keyframes if keyframe.target_effective_at <= decision
    )
    future = tuple(
        keyframe for keyframe in plan.keyframes if keyframe.target_effective_at > decision
    )
    selected: TimedPolicyActionKeyframe | None = None
    for keyframe in reversed(elapsed):
        if decision < keyframe.effective_until and decision < keyframe.valid_until:
            selected = keyframe
            break
    discarded = len(elapsed) - (1 if selected is not None else 0)
    skipped = tuple(keyframe for keyframe in elapsed if keyframe is not selected)
    selected_target = (
        None if selected is None else selected.target_effective_at
    )
    selected_lateness = (
        None if selected is None else decision - selected.target_effective_at
    )
    oldest_skipped_target = (
        None if not skipped else skipped[0].target_effective_at
    )
    if decision == plan.cursor_at and not elapsed:
        return PolicyActionPlanPopOutcome(
            plan=plan,
            keyframe=None,
            discarded_expired_count=0,
            decision_at=decision,
            selected_target_effective_at=None,
            oldest_skipped_target_effective_at=None,
            selected_lateness_sec=None,
        )
    updated = replace(
        plan,
        generation=plan.generation + 1,
        cursor_at=decision,
        created_at=decision,
        keyframes=future,
    )
    return PolicyActionPlanPopOutcome(
        plan=updated,
        keyframe=selected,
        discarded_expired_count=discarded,
        decision_at=decision,
        selected_target_effective_at=selected_target,
        oldest_skipped_target_effective_at=oldest_skipped_target,
        selected_lateness_sec=selected_lateness,
    )


def _result_request_reasons(
    request: PolicyInferenceRequest,
    result: PolicyChunkResult,
) -> tuple[str, ...]:
    checks = (
        (result.request_id == request.request_id, "request_id_mismatch"),
        (result.request_hash == request.semantic_content_hash, "request_hash_mismatch"),
        (result.inference_id == request.inference_id, "inference_id_mismatch"),
        (result.rtc_mode == request.rtc_mode, "rtc_mode_mismatch"),
        (result.session_id == request.session_id, "session_mismatch"),
        (result.control_domain_id == request.control_domain_id, "control_domain_mismatch"),
        (result.producer_id == request.producer_id, "producer_mismatch"),
        (result.producer_epoch == request.producer_epoch, "producer_epoch_mismatch"),
        (result.policy_node_id == request.policy_node_id, "policy_node_mismatch"),
        (result.artifact_hash == request.artifact_hash, "artifact_mismatch"),
        (result.action_codec_hash == request.action_codec_hash, "action_codec_mismatch"),
        (result.frame_contract_hash == request.frame_contract_hash, "frame_contract_mismatch"),
        (result.layout_hash == request.layout_hash, "layout_mismatch"),
        (
            result.normalization_hash == request.normalization_hash,
            "normalization_mismatch",
        ),
        (result.observation_cut_id == request.observation_cut_id, "observation_cut_mismatch"),
        (
            result.observation_timestamp == request.observation_timestamp,
            "observation_timestamp_mismatch",
        ),
        (
            result.observation_step_index == request.observation_step_index,
            "observation_step_mismatch",
        ),
        (result.clock_domain_id == request.clock_domain_id, "clock_domain_mismatch"),
    )
    reasons = [reason for matches, reason in checks if not matches]
    context = request.action_representation_context
    if result.canonical_actions:
        receipt = result.canonical_actions[0].canonicalization_receipt
        if context is None:
            if receipt.source_pose_mode == "relative":
                reasons.append("representation_context_missing")
        else:
            if receipt.source_pose_mode != context.source_pose_mode:
                reasons.append("representation_pose_mode_mismatch")
            if receipt.anchor_identity_hash != context.anchor_identity_hash:
                reasons.append("representation_anchor_identity_mismatch")
    elif context is not None and context.source_pose_mode == "relative":
        reasons.append("relative_result_not_typed_canonical")
    return tuple(reasons)


def _receipt(
    *,
    accepted: bool,
    reasons: tuple[str, ...],
    request: PolicyInferenceRequest,
    result: PolicyChunkResult,
    admitted_at: float,
    old_plan: TimedPolicyActionPlanSnapshot,
    new_plan: TimedPolicyActionPlanSnapshot,
    actual_cursor_at: float,
    actual_committed_through: float | None,
    spec: PolicyChunkMergeSpec,
    preserved_committed_count: int = 0,
    preserved_mutable_count: int = 0,
    discarded_obsolete_count: int = 0,
    discarded_beyond_horizon_count: int = 0,
    replaced_mutable_count: int = 0,
    admitted_keyframe_count: int = 0,
    replaced_mutable_range: tuple[float, float] | None = None,
    new_future_coverage: tuple[float, float] | None = None,
) -> PolicyChunkMergeReceipt:
    return PolicyChunkMergeReceipt(
        accepted=accepted,
        reasons=reasons,
        request_id=request.request_id,
        inference_id=request.inference_id,
        result_hash=result.semantic_content_hash,
        admitted_at=admitted_at,
        clock_domain_id=request.clock_domain_id,
        old_plan_generation=old_plan.generation,
        old_plan_hash=old_plan.semantic_content_hash,
        new_plan_generation=new_plan.generation,
        new_plan_hash=new_plan.semantic_content_hash,
        actual_cursor_at=actual_cursor_at,
        actual_committed_through=actual_committed_through,
        preserved_committed_count=preserved_committed_count,
        preserved_mutable_count=preserved_mutable_count,
        discarded_obsolete_count=discarded_obsolete_count,
        discarded_beyond_horizon_count=discarded_beyond_horizon_count,
        replaced_mutable_count=replaced_mutable_count,
        admitted_keyframe_count=admitted_keyframe_count,
        replaced_mutable_range=replaced_mutable_range,
        new_future_coverage=new_future_coverage,
        merge_spec_id=spec.spec_id,
        merge_spec_hash=spec.semantic_content_hash,
    )


def _future_coverage(
    keyframes: tuple[TimedPolicyActionKeyframe, ...],
) -> tuple[float, float] | None:
    if not keyframes:
        return None
    return (keyframes[0].target_effective_at, keyframes[-1].effective_until)


def _settle_matching_in_flight_cut(
    plan: TimedPolicyActionPlanSnapshot,
    *,
    cursor_at: float,
    committed_through: float | None,
    settled_at: float,
) -> tuple[TimedPolicyActionPlanSnapshot, int, int, int]:
    """Apply the receiver-observed cut even when the new result is rejected."""

    keyframes: list[TimedPolicyActionKeyframe] = []
    discarded_obsolete = 0
    preserved_committed = 0
    preserved_mutable = 0
    for keyframe in plan.keyframes:
        if keyframe.target_effective_at < cursor_at:
            discarded_obsolete += 1
            continue
        current = keyframe
        if (
            committed_through is not None
            and current.target_effective_at <= committed_through
            and current.state != POLICY_PLAN_KEYFRAME_PRODUCER_COMMITTED
        ):
            current = replace(
                current,
                state=POLICY_PLAN_KEYFRAME_PRODUCER_COMMITTED,
            )
        keyframes.append(current)
        if current.state == POLICY_PLAN_KEYFRAME_PRODUCER_COMMITTED:
            preserved_committed += 1
        else:
            preserved_mutable += 1
    settled = replace(
        plan,
        generation=plan.generation + 1,
        cursor_at=cursor_at,
        created_at=settled_at,
        keyframes=tuple(keyframes),
        producer_committed_through=committed_through,
        in_flight_request_id=None,
        in_flight_inference_id=None,
    )
    return settled, discarded_obsolete, preserved_committed, preserved_mutable


def merge_policy_chunk_result(
    plan: TimedPolicyActionPlanSnapshot,
    request: PolicyInferenceRequest,
    result: PolicyChunkResult,
    *,
    admitted_at: float,
    actual_cursor_at: float | None = None,
    producer_committed_through: float | None = None,
    spec: PolicyChunkMergeSpec = DEFAULT_POLICY_CHUNK_MERGE_SPEC_V1,
) -> PolicyChunkMergeOutcome:
    """Validate, crop, and atomically replace only the mutable future suffix."""

    admitted = _finite(admitted_at, name="admitted_at")
    cursor = max(
        plan.cursor_at,
        plan.cursor_at
        if actual_cursor_at is None
        else _finite(actual_cursor_at, name="actual_cursor_at"),
    )
    supplied_boundary = (
        None
        if producer_committed_through is None
        else _finite(producer_committed_through, name="producer_committed_through")
    )
    committed_through = _maximum_optional(
        plan.producer_committed_through,
        supplied_boundary,
    )

    plan_identity_reasons = _request_plan_identity_reasons(plan, request)
    in_flight_matches = (
        plan.in_flight_request_id == request.request_id
        and plan.in_flight_inference_id == request.inference_id
    )
    preflight_reasons = list(plan_identity_reasons)
    if not in_flight_matches:
        preflight_reasons.append("request_not_in_flight")
    if preflight_reasons:
        receipt = _receipt(
            accepted=False,
            reasons=tuple(preflight_reasons),
            request=request,
            result=result,
            admitted_at=admitted,
            old_plan=plan,
            new_plan=plan,
            actual_cursor_at=cursor,
            actual_committed_through=committed_through,
            spec=spec,
        )
        return PolicyChunkMergeOutcome(plan=plan, receipt=receipt)

    reasons = list(_result_request_reasons(request, result))
    if result.result_received_at > admitted:
        reasons.append("result_not_yet_received")
    if admitted > request.valid_until:
        reasons.append("request_deadline_expired")
    if result.status != "ready":
        reasons.append("worker_result_fault")
    if result.source_chunk_size > request.prediction_horizon_steps:
        reasons.append("prediction_horizon_exceeded")
    if (
        request.rtc_mode == RTC_FLOW_INPAINT_MODE_V1
        and result.conditioning_prefix_hash != request.conditioning_prefix_hash
    ):
        reasons.append("conditioning_prefix_mismatch")
    if result.execution_horizon_steps is not None and (
        result.execution_horizon_steps > result.source_chunk_size
    ):
        reasons.append("execution_horizon_exceeds_chunk")
    if result.action_payload_kind is not None and any(
        keyframe.action_payload_kind != result.action_payload_kind
        for keyframe in plan.keyframes
    ):
        reasons.append("action_payload_kind_mismatch")

    if reasons:
        cleared, old_obsolete, preserved_committed, preserved_mutable = (
            _settle_matching_in_flight_cut(
                plan,
                cursor_at=cursor,
                committed_through=committed_through,
                settled_at=admitted,
            )
        )
        receipt = _receipt(
            accepted=False,
            reasons=tuple(reasons),
            request=request,
            result=result,
            admitted_at=admitted,
            old_plan=plan,
            new_plan=cleared,
            actual_cursor_at=cursor,
            actual_committed_through=committed_through,
            spec=spec,
            preserved_committed_count=preserved_committed,
            preserved_mutable_count=preserved_mutable,
            discarded_obsolete_count=old_obsolete,
            new_future_coverage=_future_coverage(cleared.keyframes),
        )
        return PolicyChunkMergeOutcome(plan=cleared, receipt=receipt)

    new_generation = plan.generation + 1
    result_schedule = ActionEmissionScheduleSpec(
        spec_id=(
            f"{request.policy_node_id}:request:{request.request_id}:"
            f"period:{request.action_period_sec:.17g}"
        ),
        start_at=request.first_target_effective_at,
        keyframe_period_sec=request.action_period_sec,
        keyframe_count=result.source_chunk_size,
        horizon_sec=request.action_period_sec * result.source_chunk_size,
        maximum_hold_sec=request.action_period_sec,
    )
    candidates: list[TimedPolicyActionKeyframe] = []
    discarded_obsolete = 0
    execution_horizon = (
        result.source_chunk_size
        if result.execution_horizon_steps is None
        else result.execution_horizon_steps
    )
    discarded_beyond_horizon = result.source_chunk_size - execution_horizon
    maximum = spec.maximum_new_keyframes
    typed_payload = bool(result.canonical_actions)
    result_actions = (
        result.canonical_actions
        if typed_payload
        else result.decoded_absolute_actions
    )
    for index, action in enumerate(result_actions[:execution_horizon]):
        if maximum is not None and len(candidates) >= maximum:
            discarded_beyond_horizon += 1
            continue
        target = request.first_target_effective_at + index * request.action_period_sec
        effective_until = min(target + request.action_period_sec, request.valid_until)
        if target < cursor or (
            committed_through is not None and target <= committed_through
        ):
            discarded_obsolete += 1
            continue
        if effective_until <= target:
            discarded_beyond_horizon += 1
            continue
        candidates.append(
            TimedPolicyActionKeyframe(
                target_effective_at=target,
                effective_until=effective_until,
                effective_clock_domain_id=request.clock_domain_id,
                action_period_sec=request.action_period_sec,
                decoded_absolute_action=None if typed_payload else action,
                request_id=request.request_id,
                inference_id=request.inference_id,
                policy_output_id=result.policy_output_id or "",
                chunk_id=0 if result.chunk_id is None else result.chunk_id,
                chunk_keyframe_index=index,
                chunk_size=result.source_chunk_size,
                observation_cut_id=request.observation_cut_id,
                producer_epoch=request.producer_epoch,
                policy_node_id=request.policy_node_id,
                artifact_hash=request.artifact_hash,
                produced_at=result.inference_completed_at,
                schedule_spec_id=result_schedule.spec_id,
                schedule_spec_hash=result_schedule.semantic_content_hash,
                admitted_plan_generation=new_generation,
                state=POLICY_PLAN_KEYFRAME_MUTABLE,
                valid_until=request.valid_until,
                canonical_action=action if typed_payload else None,
            )
        )

    if (
        not candidates
        or (
            spec.require_minimum_execution_horizon
            and len(candidates) < request.minimum_execution_horizon_steps
        )
    ):
        reason = (
            "no_eligible_future"
            if not candidates
            else "insufficient_execution_horizon"
        )
        cleared, old_obsolete, preserved_committed, preserved_mutable = (
            _settle_matching_in_flight_cut(
                plan,
                cursor_at=cursor,
                committed_through=committed_through,
                settled_at=admitted,
            )
        )
        receipt = _receipt(
            accepted=False,
            reasons=(reason,),
            request=request,
            result=result,
            admitted_at=admitted,
            old_plan=plan,
            new_plan=cleared,
            actual_cursor_at=cursor,
            actual_committed_through=committed_through,
            spec=spec,
            preserved_committed_count=preserved_committed,
            preserved_mutable_count=preserved_mutable,
            discarded_obsolete_count=discarded_obsolete + old_obsolete,
            discarded_beyond_horizon_count=discarded_beyond_horizon,
            new_future_coverage=_future_coverage(cleared.keyframes),
        )
        return PolicyChunkMergeOutcome(plan=cleared, receipt=receipt)

    first_new_target = candidates[0].target_effective_at
    preserved: list[TimedPolicyActionKeyframe] = []
    replaced_mutable: list[TimedPolicyActionKeyframe] = []
    preserved_committed_count = 0
    preserved_mutable_count = 0
    for old in plan.keyframes:
        if old.target_effective_at < cursor:
            discarded_obsolete += 1
            continue
        current = old
        if (
            committed_through is not None
            and current.target_effective_at <= committed_through
            and current.state != POLICY_PLAN_KEYFRAME_PRODUCER_COMMITTED
        ):
            current = replace(
                current,
                state=POLICY_PLAN_KEYFRAME_PRODUCER_COMMITTED,
            )
        if current.state == POLICY_PLAN_KEYFRAME_PRODUCER_COMMITTED:
            preserved.append(current)
            preserved_committed_count += 1
        elif current.target_effective_at < first_new_target:
            preserved.append(current)
            preserved_mutable_count += 1
        else:
            replaced_mutable.append(current)

    keyframes = tuple(
        sorted(
            (*preserved, *candidates),
            key=lambda item: item.target_effective_at,
        )
    )
    merged = replace(
        plan,
        generation=new_generation,
        cursor_at=cursor,
        created_at=admitted,
        keyframes=keyframes,
        producer_committed_through=committed_through,
        in_flight_request_id=None,
        in_flight_inference_id=None,
    )
    replaced_range = (
        None
        if not replaced_mutable
        else (
            replaced_mutable[0].target_effective_at,
            replaced_mutable[-1].effective_until,
        )
    )
    coverage = _future_coverage(keyframes)
    receipt = _receipt(
        accepted=True,
        reasons=(),
        request=request,
        result=result,
        admitted_at=admitted,
        old_plan=plan,
        new_plan=merged,
        actual_cursor_at=cursor,
        actual_committed_through=committed_through,
        spec=spec,
        preserved_committed_count=preserved_committed_count,
        preserved_mutable_count=preserved_mutable_count,
        discarded_obsolete_count=discarded_obsolete,
        discarded_beyond_horizon_count=discarded_beyond_horizon,
        replaced_mutable_count=len(replaced_mutable),
        admitted_keyframe_count=len(candidates),
        replaced_mutable_range=replaced_range,
        new_future_coverage=coverage,
    )
    return PolicyChunkMergeOutcome(plan=merged, receipt=receipt)


__all__ = [
    "DEFAULT_POLICY_CHUNK_MERGE_SPEC_V1",
    "POLICY_CHUNK_MERGE_RECEIPT_HASH_DOMAIN_V1",
    "POLICY_ACTION_PLAN_POP_OUTCOME_HASH_DOMAIN_V1",
    "POLICY_CHUNK_MERGE_SPEC_HASH_DOMAIN_V1",
    "POLICY_PLAN_KEYFRAME_MUTABLE",
    "POLICY_PLAN_KEYFRAME_PRODUCER_COMMITTED",
    "POLICY_PLAN_KEYFRAME_STATES_V1",
    "PolicyActionPlanPopOutcome",
    "PolicyChunkMergeOutcome",
    "PolicyChunkMergeReceipt",
    "PolicyChunkMergeSpec",
    "REPLACE_MUTABLE_FUTURE_MERGE_V1",
    "TIMED_POLICY_ACTION_KEYFRAME_HASH_DOMAIN_V1",
    "TIMED_POLICY_ACTION_KEYFRAME_HASH_DOMAIN_V2",
    "TIMED_POLICY_ACTION_PLAN_HASH_DOMAIN_V1",
    "TIMED_POLICY_ACTION_PLAN_HASH_DOMAIN_V2",
    "TimedPolicyActionKeyframe",
    "TimedPolicyActionPlanSnapshot",
    "commit_policy_action_plan_through",
    "empty_policy_action_plan",
    "mark_policy_action_plan_in_flight",
    "merge_policy_chunk_result",
    "pop_due_policy_action",
]
