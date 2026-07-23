"""Pure fail-closed source-handoff contracts for Control Station.

This module does not own a source, model worker, action queue, episode,
dispatcher, or backend.  It only orders already-produced lower-runtime
evidence for an explicit transition from an autonomous source to a human
source.  A connected human source remains non-publishable until the final
post-takeover readiness and READY receipts are accepted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from dataset_contracts.canonical_encoding import canonical_content_hash


POLICY_SUSPENSION_MODES_V1 = (
    "suspend_policy_keep_reference",
    "hold_and_release_authority",
)
EPISODE_BOUNDARY_MOTION_POLICIES_V1 = (
    "capture_only",
    "hold_before_unbind",
)
HUMAN_TAKEOVER_PHASES_V1 = (
    "awaiting_hold",
    "awaiting_source_invalidation",
    "awaiting_authority_release",
    "awaiting_human_takeover",
    "awaiting_human_readiness",
    "awaiting_ready",
    "ready",
    "failed",
)

MIXED_CONTROL_BOUNDARY_POLICY_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.mixed_control_boundary_policy.v1"
)
HUMAN_TAKEOVER_REQUEST_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.human_takeover_request.v1"
)
HUMAN_TAKEOVER_RECEIPT_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.human_takeover_receipt.v1"
)
HUMAN_TAKEOVER_STATE_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.human_takeover_state.v1"
)
HUMAN_TAKEOVER_OPERATION_SPEC_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.human_takeover_operation_spec.v1"
)
HUMAN_TAKEOVER_OPERATION_EVIDENCE_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.human_takeover_operation_evidence.v1"
)


def _required(value: object, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


def _optional(value: object | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _finite(value: object, *, field_name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


def _epoch(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer.")
    normalized = int(value)
    if normalized != value or normalized < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return normalized


def _evidence(values: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    normalized = tuple(_required(value, field_name=field_name) for value in values)
    if not normalized:
        raise ValueError(f"{field_name} must contain evidence.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} cannot contain duplicates.")
    return tuple(sorted(normalized))


def _hash_receipt(kind: str, payload: dict[str, object]) -> str:
    return canonical_content_hash(
        {"receipt_kind": kind, **payload},
        domain=HUMAN_TAKEOVER_RECEIPT_HASH_DOMAIN_V1,
    )


@dataclass(frozen=True)
class MixedControlBoundaryPolicy:
    """Versioned pause/capture/takeover semantics; never a motion permit."""

    policy_id: str
    policy_suspension_mode: str
    episode_boundary_motion_policy: str
    human_emission_max_age_sec: float = 0.1
    measured_state_max_age_sec: float = 0.1
    require_explicit_human_takeover: bool = True
    require_post_takeover_human_emission: bool = True
    require_deadman_rearm: bool = True
    require_retargeter_rebase: bool = True
    require_canonical_safety: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _required(self.policy_id, field_name="policy_id"))
        if self.policy_suspension_mode not in POLICY_SUSPENSION_MODES_V1:
            raise ValueError("Unsupported policy_suspension_mode.")
        if self.episode_boundary_motion_policy not in EPISODE_BOUNDARY_MOTION_POLICIES_V1:
            raise ValueError("Unsupported episode_boundary_motion_policy.")
        for field_name in ("human_emission_max_age_sec", "measured_state_max_age_sec"):
            value = _finite(getattr(self, field_name), field_name=field_name)
            if value <= 0.0:
                raise ValueError(f"{field_name} must be positive.")
            object.__setattr__(self, field_name, value)
        required_flags = (
            "require_explicit_human_takeover",
            "require_post_takeover_human_emission",
            "require_deadman_rearm",
            "require_retargeter_rebase",
            "require_canonical_safety",
        )
        if any(not bool(getattr(self, name)) for name in required_flags):
            raise ValueError("Human takeover v1 fixes every readiness requirement to true.")

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "policy_id": self.policy_id,
                "policy_suspension_mode": self.policy_suspension_mode,
                "episode_boundary_motion_policy": self.episode_boundary_motion_policy,
                "human_emission_max_age_sec": self.human_emission_max_age_sec,
                "measured_state_max_age_sec": self.measured_state_max_age_sec,
                "require_explicit_human_takeover": self.require_explicit_human_takeover,
                "require_post_takeover_human_emission": (
                    self.require_post_takeover_human_emission
                ),
                "require_deadman_rearm": self.require_deadman_rearm,
                "require_retargeter_rebase": self.require_retargeter_rebase,
                "require_canonical_safety": self.require_canonical_safety,
            },
            domain=MIXED_CONTROL_BOUNDARY_POLICY_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class HumanTakeoverOperationSpec:
    """Immutable pre-request binding for one backend-neutral takeover.

    The spec contains identities and compatibility hashes only.  Fresh human
    emissions, measured state, deadman/rebase events, live epochs, and lower
    receipts are produced after the operation starts and therefore belong to
    the eventual aggregate evidence rather than this pre-request object.
    """

    previous_source_slot_id: str
    target_human_source_slot_id: str
    previous_source_action_contract_hash: str
    target_human_action_contract_hash: str
    execution_binding_hash: str
    readiness_observer_binding_id: str
    boundary_policy: MixedControlBoundaryPolicy
    readiness_timeout_sec: float
    capture_boundary_id: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "previous_source_slot_id",
            "target_human_source_slot_id",
            "previous_source_action_contract_hash",
            "target_human_action_contract_hash",
            "execution_binding_hash",
            "readiness_observer_binding_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        if self.previous_source_slot_id == self.target_human_source_slot_id:
            raise ValueError("Human takeover operation requires two source slots.")
        if not isinstance(self.boundary_policy, MixedControlBoundaryPolicy):
            raise TypeError("boundary_policy must be a MixedControlBoundaryPolicy.")
        if (
            self.boundary_policy.policy_suspension_mode
            != "hold_and_release_authority"
            or self.boundary_policy.episode_boundary_motion_policy
            != "hold_before_unbind"
        ):
            raise ValueError(
                "Human takeover operation requires hold-and-release authority "
                "and hold-before-unbind episode semantics."
            )
        timeout = _finite(
            self.readiness_timeout_sec,
            field_name="readiness_timeout_sec",
        )
        if timeout <= 0.0:
            raise ValueError("readiness_timeout_sec must be positive.")
        object.__setattr__(self, "readiness_timeout_sec", timeout)
        object.__setattr__(
            self,
            "capture_boundary_id",
            _optional(self.capture_boundary_id),
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "previous_source_slot_id": self.previous_source_slot_id,
                "target_human_source_slot_id": self.target_human_source_slot_id,
                "previous_source_action_contract_hash": (
                    self.previous_source_action_contract_hash
                ),
                "target_human_action_contract_hash": (
                    self.target_human_action_contract_hash
                ),
                "execution_binding_hash": self.execution_binding_hash,
                "readiness_observer_binding_id": self.readiness_observer_binding_id,
                "boundary_policy_hash": self.boundary_policy.semantic_content_hash,
                "readiness_timeout_sec": self.readiness_timeout_sec,
                "capture_boundary_id": self.capture_boundary_id,
            },
            domain=HUMAN_TAKEOVER_OPERATION_SPEC_HASH_DOMAIN_V1,
        )

    @property
    def operation_spec_id(self) -> str:
        return f"human-takeover-spec:{self.semantic_content_hash}"


@dataclass(frozen=True)
class HumanTakeoverOperationEvidence:
    """Aggregate completion binding; it is evidence, never a motion permit."""

    operation_spec_id: str
    source_intent_id: str
    request_id: str
    request_semantic_hash: str
    final_state_semantic_hash: str
    previous_source_slot_id: str
    target_human_source_slot_id: str
    final_phase: str
    execution_gate_status: str
    succeeded: bool
    completed_at: float
    clock_domain_id: str
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "operation_spec_id",
            "source_intent_id",
            "request_id",
            "request_semantic_hash",
            "final_state_semantic_hash",
            "previous_source_slot_id",
            "target_human_source_slot_id",
            "clock_domain_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        if self.previous_source_slot_id == self.target_human_source_slot_id:
            raise ValueError("Human takeover evidence requires two source slots.")
        if self.final_phase not in HUMAN_TAKEOVER_PHASES_V1:
            raise ValueError("Unsupported final_phase.")
        if self.execution_gate_status not in {"hold", "ready", "fault"}:
            raise ValueError("Unsupported execution_gate_status.")
        object.__setattr__(self, "succeeded", bool(self.succeeded))
        if self.succeeded:
            if self.final_phase != "ready" or self.execution_gate_status != "ready":
                raise ValueError("Successful takeover evidence must bind READY.")
        elif self.final_phase != "failed" or self.execution_gate_status not in {
            "hold",
            "fault",
        }:
            raise ValueError("Failed takeover evidence must remain in HOLD or FAULT.")
        object.__setattr__(
            self,
            "completed_at",
            _finite(self.completed_at, field_name="completed_at"),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _evidence(self.evidence_refs, field_name="operation_evidence_refs"),
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "operation_spec_id": self.operation_spec_id,
                "source_intent_id": self.source_intent_id,
                "request_id": self.request_id,
                "request_semantic_hash": self.request_semantic_hash,
                "final_state_semantic_hash": self.final_state_semantic_hash,
                "previous_source_slot_id": self.previous_source_slot_id,
                "target_human_source_slot_id": self.target_human_source_slot_id,
                "final_phase": self.final_phase,
                "execution_gate_status": self.execution_gate_status,
                "succeeded": self.succeeded,
                "completed_at": self.completed_at,
                "clock_domain_id": self.clock_domain_id,
                "evidence_refs": self.evidence_refs,
            },
            domain=HUMAN_TAKEOVER_OPERATION_EVIDENCE_HASH_DOMAIN_V1,
        )

    @property
    def evidence_id(self) -> str:
        return f"human-takeover-evidence:{self.semantic_content_hash}"


@dataclass(frozen=True)
class HumanTakeoverRequest:
    request_id: str
    runtime_session_id: str
    control_domain_id: str
    selection_group_id: str
    previous_source_slot_id: str
    target_human_source_slot_id: str
    previous_producer_id: str
    target_human_producer_id: str
    previous_producer_epoch: int
    authority_epoch: int
    execution_epoch: int
    requested_at: float
    clock_domain_id: str
    reason: str
    capture_boundary_id: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "request_id",
            "runtime_session_id",
            "control_domain_id",
            "selection_group_id",
            "previous_source_slot_id",
            "target_human_source_slot_id",
            "previous_producer_id",
            "target_human_producer_id",
            "clock_domain_id",
            "reason",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        if self.previous_source_slot_id == self.target_human_source_slot_id:
            raise ValueError("Human takeover requires two distinct source slots.")
        if self.previous_producer_id == self.target_human_producer_id:
            raise ValueError("Human takeover requires two distinct producers.")
        for field_name in ("previous_producer_epoch", "authority_epoch", "execution_epoch"):
            object.__setattr__(
                self,
                field_name,
                _epoch(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "requested_at",
            _finite(self.requested_at, field_name="requested_at"),
        )
        object.__setattr__(self, "capture_boundary_id", _optional(self.capture_boundary_id))

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "request_id": self.request_id,
                "runtime_session_id": self.runtime_session_id,
                "control_domain_id": self.control_domain_id,
                "selection_group_id": self.selection_group_id,
                "previous_source_slot_id": self.previous_source_slot_id,
                "target_human_source_slot_id": self.target_human_source_slot_id,
                "previous_producer_id": self.previous_producer_id,
                "target_human_producer_id": self.target_human_producer_id,
                "previous_producer_epoch": self.previous_producer_epoch,
                "authority_epoch": self.authority_epoch,
                "execution_epoch": self.execution_epoch,
                "requested_at": self.requested_at,
                "clock_domain_id": self.clock_domain_id,
                "reason": self.reason,
                "capture_boundary_id": self.capture_boundary_id,
            },
            domain=HUMAN_TAKEOVER_REQUEST_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class ExecutionHoldReceipt:
    request_id: str
    prior_execution_epoch: int
    execution_epoch: int
    hold_acknowledged: bool
    completed_at: float
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _required(self.request_id, field_name="request_id"))
        for field_name in ("prior_execution_epoch", "execution_epoch"):
            object.__setattr__(
                self, field_name, _epoch(getattr(self, field_name), field_name=field_name)
            )
        object.__setattr__(self, "hold_acknowledged", bool(self.hold_acknowledged))
        object.__setattr__(
            self, "completed_at", _finite(self.completed_at, field_name="completed_at")
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _evidence(self.evidence_refs, field_name="hold_evidence_refs"),
        )

    @property
    def semantic_content_hash(self) -> str:
        return _hash_receipt("execution_hold", self.__dict__)


@dataclass(frozen=True)
class SourceInvalidationReceipt:
    request_id: str
    producer_id: str
    prior_producer_epoch: int
    producer_epoch: int
    proposal_buffer_invalidated: bool
    worker_drain_status: str
    completed_at: float
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _required(self.request_id, field_name="request_id"))
        object.__setattr__(self, "producer_id", _required(self.producer_id, field_name="producer_id"))
        for field_name in ("prior_producer_epoch", "producer_epoch"):
            object.__setattr__(
                self, field_name, _epoch(getattr(self, field_name), field_name=field_name)
            )
        object.__setattr__(
            self, "proposal_buffer_invalidated", bool(self.proposal_buffer_invalidated)
        )
        if self.worker_drain_status not in {"drained", "not_applicable", "failed"}:
            raise ValueError("Unsupported worker_drain_status.")
        object.__setattr__(
            self, "completed_at", _finite(self.completed_at, field_name="completed_at")
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _evidence(self.evidence_refs, field_name="invalidation_evidence_refs"),
        )

    @property
    def semantic_content_hash(self) -> str:
        return _hash_receipt("source_invalidation", self.__dict__)


@dataclass(frozen=True)
class AuthorityReleaseReceipt:
    request_id: str
    released_producer_id: str
    prior_authority_epoch: int
    authority_epoch: int
    accepted: bool
    completed_at: float
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _required(self.request_id, field_name="request_id"))
        object.__setattr__(
            self,
            "released_producer_id",
            _required(self.released_producer_id, field_name="released_producer_id"),
        )
        for field_name in ("prior_authority_epoch", "authority_epoch"):
            object.__setattr__(
                self, field_name, _epoch(getattr(self, field_name), field_name=field_name)
            )
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(
            self, "completed_at", _finite(self.completed_at, field_name="completed_at")
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _evidence(self.evidence_refs, field_name="release_evidence_refs"),
        )

    @property
    def semantic_content_hash(self) -> str:
        return _hash_receipt("authority_release", self.__dict__)


@dataclass(frozen=True)
class HumanAuthorityTakeoverReceipt:
    request_id: str
    human_producer_id: str
    prior_authority_epoch: int
    authority_epoch: int
    accepted: bool
    completed_at: float
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _required(self.request_id, field_name="request_id"))
        object.__setattr__(
            self,
            "human_producer_id",
            _required(self.human_producer_id, field_name="human_producer_id"),
        )
        for field_name in ("prior_authority_epoch", "authority_epoch"):
            object.__setattr__(
                self, field_name, _epoch(getattr(self, field_name), field_name=field_name)
            )
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(
            self, "completed_at", _finite(self.completed_at, field_name="completed_at")
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _evidence(self.evidence_refs, field_name="takeover_evidence_refs"),
        )

    @property
    def semantic_content_hash(self) -> str:
        return _hash_receipt("human_authority_takeover", self.__dict__)


@dataclass(frozen=True)
class HumanReadinessReceipt:
    request_id: str
    human_emission_id: str
    human_emission_available_at: float
    measured_state_id: str
    measured_state_available_at: float
    deadman_rearm_event_id: str
    deadman_rearmed_at: float
    retargeter_rebase_receipt_id: str
    retargeter_rebased_at: float
    canonical_safety_receipt_id: str
    canonical_safety_status: str
    completed_at: float

    def __post_init__(self) -> None:
        for field_name in (
            "request_id",
            "human_emission_id",
            "measured_state_id",
            "deadman_rearm_event_id",
            "retargeter_rebase_receipt_id",
            "canonical_safety_receipt_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        for field_name in (
            "human_emission_available_at",
            "measured_state_available_at",
            "deadman_rearmed_at",
            "retargeter_rebased_at",
            "completed_at",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite(getattr(self, field_name), field_name=field_name),
            )
        if self.canonical_safety_status not in {
            "safe",
            "modified",
            "hold",
            "fault",
        }:
            raise ValueError("Unsupported canonical_safety_status.")

    @property
    def semantic_content_hash(self) -> str:
        return _hash_receipt("human_readiness", self.__dict__)


@dataclass(frozen=True)
class ExecutionReadyReceipt:
    request_id: str
    execution_epoch: int
    authority_epoch: int
    accepted: bool
    completed_at: float
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _required(self.request_id, field_name="request_id"))
        for field_name in ("execution_epoch", "authority_epoch"):
            object.__setattr__(
                self, field_name, _epoch(getattr(self, field_name), field_name=field_name)
            )
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(
            self, "completed_at", _finite(self.completed_at, field_name="completed_at")
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _evidence(self.evidence_refs, field_name="ready_evidence_refs"),
        )

    @property
    def semantic_content_hash(self) -> str:
        return _hash_receipt("execution_ready", self.__dict__)


@dataclass(frozen=True)
class HumanTakeoverState:
    policy: MixedControlBoundaryPolicy
    request: HumanTakeoverRequest
    phase: str = "awaiting_hold"
    transition_sequence: int = 0
    execution_gate_status: str = "transitioning"
    execution_epoch: int = 0
    authority_epoch: int = 0
    producer_epoch: int = 0
    old_source_publication_enabled: bool = False
    human_publication_enabled: bool = False
    hold_receipt: ExecutionHoldReceipt | None = None
    invalidation_receipt: SourceInvalidationReceipt | None = None
    release_receipt: AuthorityReleaseReceipt | None = None
    takeover_receipt: HumanAuthorityTakeoverReceipt | None = None
    readiness_receipt: HumanReadinessReceipt | None = None
    ready_receipt: ExecutionReadyReceipt | None = None
    failure_reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.policy, MixedControlBoundaryPolicy):
            raise TypeError("policy must be a MixedControlBoundaryPolicy.")
        if not isinstance(self.request, HumanTakeoverRequest):
            raise TypeError("request must be a HumanTakeoverRequest.")
        if self.phase not in HUMAN_TAKEOVER_PHASES_V1:
            raise ValueError("Unsupported human takeover phase.")
        object.__setattr__(
            self,
            "transition_sequence",
            _epoch(self.transition_sequence, field_name="transition_sequence"),
        )
        if self.execution_gate_status not in {"transitioning", "hold", "ready", "fault"}:
            raise ValueError("Unsupported execution_gate_status.")
        for field_name in ("execution_epoch", "authority_epoch", "producer_epoch"):
            object.__setattr__(
                self, field_name, _epoch(getattr(self, field_name), field_name=field_name)
            )
        object.__setattr__(
            self, "old_source_publication_enabled", bool(self.old_source_publication_enabled)
        )
        object.__setattr__(
            self, "human_publication_enabled", bool(self.human_publication_enabled)
        )
        object.__setattr__(self, "failure_reason", str(self.failure_reason).strip())
        if self.old_source_publication_enabled:
            raise ValueError("A human takeover transaction cannot reopen old publication.")
        if self.human_publication_enabled != (self.phase == "ready"):
            raise ValueError("Human publication is allowed only in the ready phase.")
        if self.phase == "ready" and self.execution_gate_status != "ready":
            raise ValueError("A ready takeover requires the READY execution gate.")
        if self.phase == "failed" and not self.failure_reason:
            raise ValueError("A failed takeover requires failure_reason.")

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "policy_hash": self.policy.semantic_content_hash,
                "request_hash": self.request.semantic_content_hash,
                "phase": self.phase,
                "transition_sequence": self.transition_sequence,
                "execution_gate_status": self.execution_gate_status,
                "execution_epoch": self.execution_epoch,
                "authority_epoch": self.authority_epoch,
                "producer_epoch": self.producer_epoch,
                "old_source_publication_enabled": self.old_source_publication_enabled,
                "human_publication_enabled": self.human_publication_enabled,
                "hold_receipt_hash": _receipt_hash(self.hold_receipt),
                "invalidation_receipt_hash": _receipt_hash(self.invalidation_receipt),
                "release_receipt_hash": _receipt_hash(self.release_receipt),
                "takeover_receipt_hash": _receipt_hash(self.takeover_receipt),
                "readiness_receipt_hash": _receipt_hash(self.readiness_receipt),
                "ready_receipt_hash": _receipt_hash(self.ready_receipt),
                "failure_reason": self.failure_reason,
            },
            domain=HUMAN_TAKEOVER_STATE_HASH_DOMAIN_V1,
        )


def _receipt_hash(receipt: object | None) -> str | None:
    return None if receipt is None else str(getattr(receipt, "semantic_content_hash"))


def begin_human_takeover(
    policy: MixedControlBoundaryPolicy,
    request: HumanTakeoverRequest,
) -> HumanTakeoverState:
    """Close both publication paths before any lower side effect occurs."""

    if policy.policy_suspension_mode != "hold_and_release_authority":
        raise ValueError(
            "Human takeover requires hold_and_release_authority; "
            "suspend_policy_keep_reference cannot hand off."
        )
    return HumanTakeoverState(
        policy=policy,
        request=request,
        execution_epoch=request.execution_epoch,
        authority_epoch=request.authority_epoch,
        producer_epoch=request.previous_producer_epoch,
    )


def _require_receipt_request(state: HumanTakeoverState, request_id: str) -> None:
    if request_id != state.request.request_id:
        raise ValueError("Handoff receipt belongs to another request.")


def _same_receipt(existing: object | None, received: object) -> bool:
    return existing is not None and _receipt_hash(existing) == _receipt_hash(received)


def _require_phase(state: HumanTakeoverState, expected: str) -> None:
    if state.phase != expected:
        raise RuntimeError(
            f"Human takeover phase is {state.phase!r}; expected {expected!r}."
        )


def _failed(
    state: HumanTakeoverState,
    *,
    reason: str,
    gate_status: str = "hold",
    **updates: object,
) -> HumanTakeoverState:
    normalized_reason = _required(reason, field_name="failure_reason")
    return replace(
        state,
        phase="failed",
        transition_sequence=state.transition_sequence + 1,
        execution_gate_status=gate_status,
        human_publication_enabled=False,
        failure_reason=normalized_reason,
        **updates,
    )


def fail_human_takeover(
    state: HumanTakeoverState,
    *,
    reason: str,
    gate_status: str = "hold",
) -> HumanTakeoverState:
    """Terminate an in-flight transaction when an external readiness step fails."""

    if not isinstance(state, HumanTakeoverState):
        raise TypeError("state must be a HumanTakeoverState.")
    if gate_status not in {"hold", "fault"}:
        raise ValueError("gate_status must be hold or fault.")
    if state.phase == "ready":
        raise RuntimeError("A READY human takeover cannot be failed retroactively.")
    if state.phase == "failed":
        if state.failure_reason == str(reason).strip() and (
            state.execution_gate_status == gate_status
        ):
            return state
        raise RuntimeError("A failed human takeover cannot be rewritten.")
    return _failed(state, reason=reason, gate_status=gate_status)


def apply_execution_hold(
    state: HumanTakeoverState,
    receipt: ExecutionHoldReceipt,
) -> HumanTakeoverState:
    _require_receipt_request(state, receipt.request_id)
    if _same_receipt(state.hold_receipt, receipt):
        return state
    _require_phase(state, "awaiting_hold")
    if receipt.prior_execution_epoch != state.execution_epoch:
        raise ValueError("HOLD receipt prior execution epoch mismatches the request.")
    if not receipt.hold_acknowledged:
        return _failed(
            state,
            reason="execution_hold_not_acknowledged",
            gate_status="fault",
            hold_receipt=receipt,
        )
    if receipt.execution_epoch <= receipt.prior_execution_epoch:
        raise ValueError("HOLD must advance execution_epoch.")
    if receipt.completed_at < state.request.requested_at:
        raise ValueError("HOLD completion predates the takeover request.")
    return replace(
        state,
        phase="awaiting_source_invalidation",
        transition_sequence=state.transition_sequence + 1,
        execution_gate_status="hold",
        execution_epoch=receipt.execution_epoch,
        hold_receipt=receipt,
    )


def apply_source_invalidation(
    state: HumanTakeoverState,
    receipt: SourceInvalidationReceipt,
) -> HumanTakeoverState:
    _require_receipt_request(state, receipt.request_id)
    if _same_receipt(state.invalidation_receipt, receipt):
        return state
    _require_phase(state, "awaiting_source_invalidation")
    if receipt.producer_id != state.request.previous_producer_id:
        raise ValueError("Source invalidation names another producer.")
    if receipt.prior_producer_epoch != state.producer_epoch:
        raise ValueError("Source invalidation prior producer epoch mismatches.")
    if receipt.producer_epoch <= receipt.prior_producer_epoch:
        raise ValueError("Source invalidation must advance producer_epoch.")
    if receipt.completed_at < state.hold_receipt.completed_at:  # type: ignore[union-attr]
        raise ValueError("Source invalidation must follow HOLD acknowledgement.")
    if (
        not receipt.proposal_buffer_invalidated
        or receipt.worker_drain_status not in {"drained", "not_applicable"}
    ):
        return _failed(
            state,
            reason="old_source_invalidation_or_drain_failed",
            invalidation_receipt=receipt,
        )
    return replace(
        state,
        phase="awaiting_authority_release",
        transition_sequence=state.transition_sequence + 1,
        producer_epoch=receipt.producer_epoch,
        invalidation_receipt=receipt,
    )


def apply_authority_release(
    state: HumanTakeoverState,
    receipt: AuthorityReleaseReceipt,
) -> HumanTakeoverState:
    _require_receipt_request(state, receipt.request_id)
    if _same_receipt(state.release_receipt, receipt):
        return state
    _require_phase(state, "awaiting_authority_release")
    if receipt.released_producer_id != state.request.previous_producer_id:
        raise ValueError("Authority release names another producer.")
    if receipt.prior_authority_epoch != state.authority_epoch:
        raise ValueError("Authority release prior epoch mismatches.")
    if not receipt.accepted:
        return _failed(
            state,
            reason="old_authority_release_rejected",
            release_receipt=receipt,
        )
    if receipt.authority_epoch <= receipt.prior_authority_epoch:
        raise ValueError("Authority release must advance authority_epoch.")
    if receipt.completed_at < state.invalidation_receipt.completed_at:  # type: ignore[union-attr]
        raise ValueError("Authority release must follow source invalidation.")
    return replace(
        state,
        phase="awaiting_human_takeover",
        transition_sequence=state.transition_sequence + 1,
        authority_epoch=receipt.authority_epoch,
        release_receipt=receipt,
    )


def apply_human_authority_takeover(
    state: HumanTakeoverState,
    receipt: HumanAuthorityTakeoverReceipt,
) -> HumanTakeoverState:
    _require_receipt_request(state, receipt.request_id)
    if _same_receipt(state.takeover_receipt, receipt):
        return state
    _require_phase(state, "awaiting_human_takeover")
    if receipt.human_producer_id != state.request.target_human_producer_id:
        raise ValueError("Human takeover names another producer.")
    if receipt.prior_authority_epoch != state.authority_epoch:
        raise ValueError("Human takeover prior authority epoch mismatches.")
    if not receipt.accepted:
        return _failed(
            state,
            reason="human_authority_takeover_rejected",
            takeover_receipt=receipt,
        )
    if receipt.authority_epoch <= receipt.prior_authority_epoch:
        raise ValueError("Human takeover must advance authority_epoch.")
    if receipt.completed_at < state.release_receipt.completed_at:  # type: ignore[union-attr]
        raise ValueError("Human takeover must follow old authority release.")
    return replace(
        state,
        phase="awaiting_human_readiness",
        transition_sequence=state.transition_sequence + 1,
        authority_epoch=receipt.authority_epoch,
        takeover_receipt=receipt,
    )


def apply_human_readiness(
    state: HumanTakeoverState,
    receipt: HumanReadinessReceipt,
) -> HumanTakeoverState:
    _require_receipt_request(state, receipt.request_id)
    if _same_receipt(state.readiness_receipt, receipt):
        return state
    _require_phase(state, "awaiting_human_readiness")
    assert state.hold_receipt is not None
    assert state.takeover_receipt is not None
    takeover_at = state.takeover_receipt.completed_at
    if receipt.human_emission_available_at < takeover_at:
        raise ValueError("Human emission must be available after authority takeover.")
    if receipt.deadman_rearmed_at < takeover_at:
        raise ValueError("Deadman must be re-armed after authority takeover.")
    if receipt.retargeter_rebased_at < takeover_at:
        raise ValueError("Retargeter must be re-based after authority takeover.")
    if receipt.measured_state_available_at < state.hold_receipt.completed_at:
        raise ValueError("Measured state must be fresh relative to the HOLD boundary.")
    newest_input_at = max(
        receipt.human_emission_available_at,
        receipt.measured_state_available_at,
        receipt.deadman_rearmed_at,
    )
    if receipt.completed_at < newest_input_at:
        raise ValueError("Human readiness completion predates one of its inputs.")
    if (
        receipt.completed_at - receipt.human_emission_available_at
        > state.policy.human_emission_max_age_sec
    ):
        raise ValueError("Human takeover emission is stale.")
    if (
        receipt.completed_at - receipt.measured_state_available_at
        > state.policy.measured_state_max_age_sec
    ):
        raise ValueError("Human takeover measured state is stale.")
    if receipt.canonical_safety_status not in {"safe", "modified"}:
        return _failed(
            state,
            reason=f"human_readiness_{receipt.canonical_safety_status}",
            readiness_receipt=receipt,
        )
    return replace(
        state,
        phase="awaiting_ready",
        transition_sequence=state.transition_sequence + 1,
        readiness_receipt=receipt,
    )


def apply_execution_ready(
    state: HumanTakeoverState,
    receipt: ExecutionReadyReceipt,
) -> HumanTakeoverState:
    _require_receipt_request(state, receipt.request_id)
    if _same_receipt(state.ready_receipt, receipt):
        return state
    _require_phase(state, "awaiting_ready")
    assert state.readiness_receipt is not None
    if receipt.execution_epoch != state.execution_epoch:
        raise ValueError("READY cannot change the held execution epoch.")
    if receipt.authority_epoch != state.authority_epoch:
        raise ValueError("READY authority epoch mismatches human takeover.")
    if receipt.completed_at < state.readiness_receipt.completed_at:
        raise ValueError("READY must follow human readiness.")
    if not receipt.accepted:
        return _failed(
            state,
            reason="execution_ready_rejected",
            ready_receipt=receipt,
        )
    return replace(
        state,
        phase="ready",
        transition_sequence=state.transition_sequence + 1,
        execution_gate_status="ready",
        human_publication_enabled=True,
        ready_receipt=receipt,
    )


__all__ = [
    "EPISODE_BOUNDARY_MOTION_POLICIES_V1",
    "HUMAN_TAKEOVER_PHASES_V1",
    "POLICY_SUSPENSION_MODES_V1",
    "AuthorityReleaseReceipt",
    "ExecutionHoldReceipt",
    "ExecutionReadyReceipt",
    "HumanAuthorityTakeoverReceipt",
    "HumanReadinessReceipt",
    "HumanTakeoverOperationEvidence",
    "HumanTakeoverOperationSpec",
    "HumanTakeoverRequest",
    "HumanTakeoverState",
    "MixedControlBoundaryPolicy",
    "SourceInvalidationReceipt",
    "apply_authority_release",
    "apply_execution_hold",
    "apply_execution_ready",
    "apply_human_authority_takeover",
    "apply_human_readiness",
    "apply_source_invalidation",
    "begin_human_takeover",
    "fail_human_takeover",
]
