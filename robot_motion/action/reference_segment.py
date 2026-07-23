"""Pure contracts for target-time-aware position reference segments.

This module deliberately does not observe or mutate a position worker.  It
separates receiver admission, endpoint effective time, high-rate reference
derivation, and segment termination so future lookahead cannot advance the
current safety anchor early.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from dataset_contracts.canonical_encoding import canonical_content_hash

from .proposal import freeze_robot_action, robot_action_semantic_payload
from .safety_anchor import SafetyAnchorCommitEvidence, safety_anchor_action_hash
from .schema import RobotAction


REFERENCE_ENDPOINT_HASH_DOMAIN_V1 = "dexteleop.reference_endpoint_parent.v1"
REFERENCE_SEGMENT_HASH_DOMAIN_V1 = "dexteleop.reference_segment.v1"
REFERENCE_SEGMENT_ADMISSION_HASH_DOMAIN_V1 = (
    "dexteleop.reference_segment_admission.v1"
)
REFERENCE_SEGMENT_TERMINAL_HASH_DOMAIN_V1 = (
    "dexteleop.reference_segment_terminal.v1"
)
GROUP_GENERATION_RESERVATION_HASH_DOMAIN_V1 = (
    "dexteleop.pending_group_generation_reservation.v1"
)
GROUP_GENERATION_ACTIVATION_HASH_DOMAIN_V1 = (
    "dexteleop.group_generation_activation.v1"
)
SAFETY_ANCHOR_COMMIT_CANDIDATE_HASH_DOMAIN_V1 = (
    "dexteleop.safety_anchor_commit_candidate.v1"
)
SAFETY_ANCHOR_CANDIDATE_ASSESSMENT_HASH_DOMAIN_V1 = (
    "dexteleop.safety_anchor_candidate_assessment.v1"
)
SAFETY_ANCHOR_TIMELINE_RESOLUTION_HASH_DOMAIN_V1 = (
    "dexteleop.safety_anchor_timeline_resolution.v1"
)

REFERENCE_ENDPOINT_ROLES_V1 = frozenset({"start", "end"})
REFERENCE_ENDPOINT_SOURCE_KINDS_V1 = frozenset(
    {
        "measured_rebase",
        "committed_safe_endpoint",
        "current_safe_target",
        "resolved_execution_target",
    }
)
REFERENCE_SEGMENT_AUTHORITY_KINDS_V1 = frozenset(
    {
        "legacy_execution_observation",
        "counterfactual_shadow",
        "authoritative_shared",
    }
)
REFERENCE_ADMISSION_GUARANTEES_V1 = (
    "queue_only",
    "interpolation_segment_accepted",
    "endpoint_resolved",
    "trajectory_validated",
    "backend_accepted",
)
REFERENCE_SEGMENT_SAFETY_EVIDENCE_LEVELS_V1 = frozenset(
    {
        "unverified",
        "endpoints_only",
        "discrete_reference_checked",
        "continuous_certified",
    }
)
REFERENCE_SEGMENT_TERMINAL_STATUSES_V1 = frozenset(
    {
        "completed",
        "retired_by_successor_activation",
        "superseded",
        "canceled",
        "expired",
        "faulted",
    }
)
_INVALIDATING_TERMINAL_STATUSES = frozenset(
    {"superseded", "canceled", "expired", "faulted"}
)
REFERENCE_SEGMENT_RELATION_KINDS_V1 = frozenset(
    {"initial", "successor", "supersedes_pending"}
)


def _required_text(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _finite_time(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer.")
    result = int(value)
    if result != value or result < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return result


def _guarantee_rank(value: str) -> int:
    guarantee = _required_text(value, name="admission_guarantee")
    try:
        return REFERENCE_ADMISSION_GUARANTEES_V1.index(guarantee)
    except ValueError as exc:
        raise ValueError("Unsupported reference admission guarantee.") from exc


def admission_guarantee_satisfies(achieved: str, required: str) -> bool:
    """Return whether an achieved guarantee satisfies a required level."""

    return _guarantee_rank(achieved) >= _guarantee_rank(required)


@dataclass(frozen=True)
class ReferenceEndpointParent:
    endpoint_parent_id: str
    role: str
    source_kind: str
    source_fact_id: str
    projected_group_action_hash: str
    action_schema_hash: str
    frame_schema_hash: str
    target_effective_at: float
    semantic_content_hash: str

    def __post_init__(self) -> None:
        for name in (
            "endpoint_parent_id",
            "source_fact_id",
            "projected_group_action_hash",
            "action_schema_hash",
            "frame_schema_hash",
            "semantic_content_hash",
        ):
            _required_text(getattr(self, name), name=name)
        if self.role not in REFERENCE_ENDPOINT_ROLES_V1:
            raise ValueError("Unsupported reference endpoint role.")
        if self.source_kind not in REFERENCE_ENDPOINT_SOURCE_KINDS_V1:
            raise ValueError("Unsupported reference endpoint source kind.")
        if self.role == "start" and self.source_kind not in {
            "measured_rebase",
            "committed_safe_endpoint",
        }:
            raise ValueError("A START endpoint must be measured or previously committed.")
        if self.role == "end" and self.source_kind not in {
            "current_safe_target",
            "resolved_execution_target",
        }:
            raise ValueError(
                "An END endpoint must be the current shared-safe target or an "
                "explicit resolved execution target."
            )
        object.__setattr__(
            self,
            "target_effective_at",
            _finite_time(self.target_effective_at, name="target_effective_at"),
        )


def build_reference_endpoint_parent(
    *,
    role: str,
    source_kind: str,
    source_fact_id: str,
    projected_group_action_hash: str,
    action_schema_hash: str,
    frame_schema_hash: str,
    target_effective_at: float,
) -> ReferenceEndpointParent:
    payload = {
        "role": _required_text(role, name="role"),
        "source_kind": _required_text(source_kind, name="source_kind"),
        "source_fact_id": _required_text(source_fact_id, name="source_fact_id"),
        "projected_group_action_hash": _required_text(
            projected_group_action_hash,
            name="projected_group_action_hash",
        ),
        "action_schema_hash": _required_text(
            action_schema_hash, name="action_schema_hash"
        ),
        "frame_schema_hash": _required_text(
            frame_schema_hash, name="frame_schema_hash"
        ),
        "target_effective_at": _finite_time(
            target_effective_at, name="target_effective_at"
        ),
    }
    semantic_hash = canonical_content_hash(
        payload,
        domain=REFERENCE_ENDPOINT_HASH_DOMAIN_V1,
    )
    return ReferenceEndpointParent(
        endpoint_parent_id=f"reference-endpoint-parent:{semantic_hash}",
        **payload,
        semantic_content_hash=semantic_hash,
    )


@dataclass(frozen=True)
class ReferenceSegment:
    reference_segment_id: str
    control_domain_id: str
    selection_group_id: str
    execution_epoch: int
    group_segment_generation: int
    authority_kind: str
    endpoint_parents: tuple[ReferenceEndpointParent, ...]
    component_dependencies: tuple[str, ...]
    created_at: float
    scheduled_from: float
    scheduled_until: float
    clock_domain_id: str
    interpolation_contract_hash: str
    resolver_contract_hash: str
    planner_contract_hash: str | None
    input_action_schema_hash: str
    output_reference_schema_hash: str
    required_admission_guarantee: str
    segment_safety_evidence_level: str
    semantic_content_hash: str

    def __post_init__(self) -> None:
        for name in (
            "reference_segment_id",
            "control_domain_id",
            "selection_group_id",
            "clock_domain_id",
            "interpolation_contract_hash",
            "resolver_contract_hash",
            "input_action_schema_hash",
            "output_reference_schema_hash",
            "semantic_content_hash",
        ):
            _required_text(getattr(self, name), name=name)
        _non_negative_int(self.execution_epoch, name="execution_epoch")
        _non_negative_int(
            self.group_segment_generation,
            name="group_segment_generation",
        )
        if self.authority_kind not in REFERENCE_SEGMENT_AUTHORITY_KINDS_V1:
            raise ValueError("Unsupported reference segment authority kind.")
        parents = tuple(self.endpoint_parents)
        if len(parents) != 2 or tuple(parent.role for parent in parents) != (
            "start",
            "end",
        ):
            raise ValueError("Reference segment v1 requires ordered START then END parents.")
        if len({parent.endpoint_parent_id for parent in parents}) != 2:
            raise ValueError("Reference segment endpoint parents must be distinct.")
        dependencies = tuple(
            _required_text(value, name="component_dependency")
            for value in self.component_dependencies
        )
        if not dependencies or len(set(dependencies)) != len(dependencies):
            raise ValueError("component_dependencies must be non-empty and unique.")
        created = _finite_time(self.created_at, name="created_at")
        scheduled_from = _finite_time(self.scheduled_from, name="scheduled_from")
        scheduled_until = _finite_time(self.scheduled_until, name="scheduled_until")
        if scheduled_until <= scheduled_from:
            raise ValueError("Reference segment interval must be [from, until).")
        if not math.isclose(
            parents[0].target_effective_at,
            scheduled_from,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("START endpoint effective time must equal scheduled_from.")
        if not math.isclose(
            parents[1].target_effective_at,
            scheduled_until,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("END endpoint effective time must equal scheduled_until.")
        if any(
            parent.action_schema_hash != self.input_action_schema_hash
            for parent in parents
        ):
            raise ValueError("Reference endpoint action schema must match segment input.")
        if parents[0].frame_schema_hash != parents[1].frame_schema_hash:
            raise ValueError("Reference segment endpoints must share one frame schema.")
        _guarantee_rank(self.required_admission_guarantee)
        if (
            self.segment_safety_evidence_level
            not in REFERENCE_SEGMENT_SAFETY_EVIDENCE_LEVELS_V1
        ):
            raise ValueError("Unsupported segment safety evidence level.")
        if self.planner_contract_hash is not None:
            _required_text(self.planner_contract_hash, name="planner_contract_hash")
        object.__setattr__(self, "endpoint_parents", parents)
        object.__setattr__(self, "component_dependencies", dependencies)
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "scheduled_from", scheduled_from)
        object.__setattr__(self, "scheduled_until", scheduled_until)

    @property
    def start_parent(self) -> ReferenceEndpointParent:
        return self.endpoint_parents[0]

    @property
    def end_parent(self) -> ReferenceEndpointParent:
        return self.endpoint_parents[1]


def build_reference_segment(
    *,
    control_domain_id: str,
    selection_group_id: str,
    execution_epoch: int,
    group_segment_generation: int,
    authority_kind: str,
    start_parent: ReferenceEndpointParent,
    end_parent: ReferenceEndpointParent,
    component_dependencies: tuple[str, ...],
    created_at: float,
    clock_domain_id: str,
    interpolation_contract_hash: str,
    resolver_contract_hash: str,
    input_action_schema_hash: str,
    output_reference_schema_hash: str,
    required_admission_guarantee: str = "interpolation_segment_accepted",
    segment_safety_evidence_level: str = "unverified",
    planner_contract_hash: str | None = None,
) -> ReferenceSegment:
    if not isinstance(start_parent, ReferenceEndpointParent) or not isinstance(
        end_parent, ReferenceEndpointParent
    ):
        raise TypeError("start_parent/end_parent must be ReferenceEndpointParent values.")
    parents = (start_parent, end_parent)
    dependencies = tuple(
        _required_text(value, name="component_dependency")
        for value in component_dependencies
    )
    payload = {
        "control_domain_id": _required_text(
            control_domain_id, name="control_domain_id"
        ),
        "selection_group_id": _required_text(
            selection_group_id, name="selection_group_id"
        ),
        "execution_epoch": _non_negative_int(
            execution_epoch, name="execution_epoch"
        ),
        "group_segment_generation": _non_negative_int(
            group_segment_generation,
            name="group_segment_generation",
        ),
        "authority_kind": _required_text(
            authority_kind,
            name="authority_kind",
        ),
        "endpoint_parent_ids": tuple(parent.endpoint_parent_id for parent in parents),
        "endpoint_parent_hashes": tuple(
            parent.semantic_content_hash for parent in parents
        ),
        "component_dependencies": dependencies,
        "created_at": _finite_time(created_at, name="created_at"),
        "scheduled_from": float(start_parent.target_effective_at),
        "scheduled_until": float(end_parent.target_effective_at),
        "clock_domain_id": _required_text(
            clock_domain_id, name="clock_domain_id"
        ),
        "interpolation_contract_hash": _required_text(
            interpolation_contract_hash,
            name="interpolation_contract_hash",
        ),
        "resolver_contract_hash": _required_text(
            resolver_contract_hash, name="resolver_contract_hash"
        ),
        "planner_contract_hash": (
            None
            if planner_contract_hash is None
            else _required_text(
                planner_contract_hash, name="planner_contract_hash"
            )
        ),
        "input_action_schema_hash": _required_text(
            input_action_schema_hash, name="input_action_schema_hash"
        ),
        "output_reference_schema_hash": _required_text(
            output_reference_schema_hash,
            name="output_reference_schema_hash",
        ),
        "required_admission_guarantee": _required_text(
            required_admission_guarantee,
            name="required_admission_guarantee",
        ),
        "segment_safety_evidence_level": _required_text(
            segment_safety_evidence_level,
            name="segment_safety_evidence_level",
        ),
    }
    if payload["scheduled_until"] <= payload["scheduled_from"]:
        raise ValueError("Reference segment END must be later than START.")
    _guarantee_rank(payload["required_admission_guarantee"])
    if payload["authority_kind"] not in REFERENCE_SEGMENT_AUTHORITY_KINDS_V1:
        raise ValueError("Unsupported reference segment authority kind.")
    if (
        payload["segment_safety_evidence_level"]
        not in REFERENCE_SEGMENT_SAFETY_EVIDENCE_LEVELS_V1
    ):
        raise ValueError("Unsupported segment safety evidence level.")
    semantic_hash = canonical_content_hash(
        payload,
        domain=REFERENCE_SEGMENT_HASH_DOMAIN_V1,
    )
    return ReferenceSegment(
        reference_segment_id=f"reference-segment:{semantic_hash}",
        control_domain_id=payload["control_domain_id"],
        selection_group_id=payload["selection_group_id"],
        execution_epoch=payload["execution_epoch"],
        group_segment_generation=payload["group_segment_generation"],
        authority_kind=payload["authority_kind"],
        endpoint_parents=parents,
        component_dependencies=dependencies,
        created_at=payload["created_at"],
        scheduled_from=payload["scheduled_from"],
        scheduled_until=payload["scheduled_until"],
        clock_domain_id=payload["clock_domain_id"],
        interpolation_contract_hash=payload["interpolation_contract_hash"],
        resolver_contract_hash=payload["resolver_contract_hash"],
        planner_contract_hash=payload["planner_contract_hash"],
        input_action_schema_hash=payload["input_action_schema_hash"],
        output_reference_schema_hash=payload["output_reference_schema_hash"],
        required_admission_guarantee=payload["required_admission_guarantee"],
        segment_safety_evidence_level=payload[
            "segment_safety_evidence_level"
        ],
        semantic_content_hash=semantic_hash,
    )


@dataclass(frozen=True)
class ReferenceSegmentAdmissionReceipt:
    admission_id: str
    reference_segment_id: str
    reference_segment_hash: str
    control_domain_id: str
    selection_group_id: str
    execution_epoch: int
    group_segment_generation: int
    receiver_id: str
    admitted_at: float
    admission_sequence: int
    clock_domain_id: str
    achieved_admission_guarantee: str
    supersedes_segment_ids: tuple[str, ...]
    accepted: bool
    rejection_reason: str | None
    semantic_content_hash: str

    def __post_init__(self) -> None:
        for name in (
            "admission_id",
            "reference_segment_id",
            "reference_segment_hash",
            "control_domain_id",
            "selection_group_id",
            "receiver_id",
            "clock_domain_id",
            "semantic_content_hash",
        ):
            _required_text(getattr(self, name), name=name)
        _non_negative_int(self.execution_epoch, name="execution_epoch")
        _non_negative_int(
            self.group_segment_generation,
            name="group_segment_generation",
        )
        _non_negative_int(self.admission_sequence, name="admission_sequence")
        object.__setattr__(
            self,
            "admitted_at",
            _finite_time(self.admitted_at, name="admitted_at"),
        )
        _guarantee_rank(self.achieved_admission_guarantee)
        supersedes = tuple(
            _required_text(value, name="supersedes_segment_id")
            for value in self.supersedes_segment_ids
        )
        if len(set(supersedes)) != len(supersedes):
            raise ValueError("supersedes_segment_ids must be unique.")
        if self.reference_segment_id in supersedes:
            raise ValueError("A reference segment cannot supersede itself.")
        if self.accepted and self.rejection_reason is not None:
            raise ValueError("Accepted segment admission cannot have a rejection reason.")
        if not self.accepted and not str(self.rejection_reason or "").strip():
            raise ValueError("Rejected segment admission requires a reason.")
        object.__setattr__(self, "supersedes_segment_ids", supersedes)


def admit_reference_segment(
    segment: ReferenceSegment,
    *,
    receiver_id: str,
    admitted_at: float,
    admission_sequence: int,
    current_execution_epoch: int,
    expected_group_segment_generation: int,
    achieved_admission_guarantee: str,
    supersedes_segment_ids: tuple[str, ...] = (),
    receiver_accepts: bool = True,
    rejection_reason: str | None = None,
) -> ReferenceSegmentAdmissionReceipt:
    if not isinstance(segment, ReferenceSegment):
        raise TypeError("segment must be a ReferenceSegment.")
    timestamp = _finite_time(admitted_at, name="admitted_at")
    achieved = _required_text(
        achieved_admission_guarantee,
        name="achieved_admission_guarantee",
    )
    _guarantee_rank(achieved)
    reason: str | None = None
    current_epoch = _non_negative_int(
        current_execution_epoch,
        name="current_execution_epoch",
    )
    expected_generation = _non_negative_int(
        expected_group_segment_generation,
        name="expected_group_segment_generation",
    )
    if current_epoch != segment.execution_epoch:
        reason = "execution_epoch_mismatch"
    elif expected_generation != segment.group_segment_generation:
        reason = "group_segment_generation_mismatch"
    elif timestamp < segment.created_at:
        reason = "segment_not_yet_created"
    elif timestamp >= segment.scheduled_until:
        reason = "segment_interval_expired"
    elif not admission_guarantee_satisfies(
        achieved,
        segment.required_admission_guarantee,
    ):
        reason = "insufficient_admission_guarantee"
    elif not receiver_accepts:
        reason = _required_text(rejection_reason, name="rejection_reason")
    elif rejection_reason is not None:
        raise ValueError("Accepted segment admission cannot have a rejection reason.")
    accepted = reason is None
    supersedes = tuple(
        _required_text(value, name="supersedes_segment_id")
        for value in supersedes_segment_ids
    )
    payload = {
        "reference_segment_id": segment.reference_segment_id,
        "reference_segment_hash": segment.semantic_content_hash,
        "control_domain_id": segment.control_domain_id,
        "selection_group_id": segment.selection_group_id,
        "execution_epoch": segment.execution_epoch,
        "group_segment_generation": segment.group_segment_generation,
        "receiver_id": _required_text(receiver_id, name="receiver_id"),
        "admitted_at": timestamp,
        "admission_sequence": _non_negative_int(
            admission_sequence, name="admission_sequence"
        ),
        "clock_domain_id": segment.clock_domain_id,
        "achieved_admission_guarantee": achieved,
        "supersedes_segment_ids": supersedes,
        "accepted": accepted,
        "rejection_reason": reason,
    }
    semantic_hash = canonical_content_hash(
        payload,
        domain=REFERENCE_SEGMENT_ADMISSION_HASH_DOMAIN_V1,
    )
    return ReferenceSegmentAdmissionReceipt(
        admission_id=f"reference-segment-admission:{semantic_hash}",
        **payload,
        semantic_content_hash=semantic_hash,
    )


@dataclass(frozen=True)
class PendingGroupGenerationReservation:
    reservation_id: str
    control_domain_id: str
    selection_group_id: str
    execution_epoch: int
    active_generation_at_reservation: int
    reserved_generation: int
    source_segment_id: str
    source_segment_hash: str
    admission_id: str
    admission_hash: str
    relation_kind: str
    predecessor_segment_id: str | None
    supersedes_reservation_id: str | None
    effective_from: float
    admitted_at: float
    admission_sequence: int
    clock_domain_id: str
    semantic_content_hash: str

    def __post_init__(self) -> None:
        for name in (
            "reservation_id",
            "control_domain_id",
            "selection_group_id",
            "source_segment_id",
            "source_segment_hash",
            "admission_id",
            "admission_hash",
            "clock_domain_id",
            "semantic_content_hash",
        ):
            _required_text(getattr(self, name), name=name)
        _non_negative_int(self.execution_epoch, name="execution_epoch")
        active = _non_negative_int(
            self.active_generation_at_reservation,
            name="active_generation_at_reservation",
        )
        reserved = _non_negative_int(
            self.reserved_generation,
            name="reserved_generation",
        )
        if reserved != active + 1:
            raise ValueError("A pending reservation must reserve active generation + 1.")
        relation = _required_text(self.relation_kind, name="relation_kind")
        if relation not in REFERENCE_SEGMENT_RELATION_KINDS_V1:
            raise ValueError("Unsupported reference segment relation kind.")
        predecessor = self.predecessor_segment_id
        if predecessor is not None:
            predecessor = _required_text(predecessor, name="predecessor_segment_id")
        supersedes = self.supersedes_reservation_id
        if supersedes is not None:
            supersedes = _required_text(
                supersedes,
                name="supersedes_reservation_id",
            )
        if relation == "initial" and (predecessor is not None or supersedes is not None):
            raise ValueError("An initial reservation cannot reference a predecessor.")
        if relation == "successor" and (predecessor is None or supersedes is not None):
            raise ValueError("A successor reservation requires only a predecessor segment.")
        if relation == "supersedes_pending" and supersedes is None:
            raise ValueError("A superseding reservation requires the old reservation ID.")
        object.__setattr__(self, "relation_kind", relation)
        object.__setattr__(self, "predecessor_segment_id", predecessor)
        object.__setattr__(self, "supersedes_reservation_id", supersedes)
        object.__setattr__(
            self,
            "effective_from",
            _finite_time(self.effective_from, name="effective_from"),
        )
        object.__setattr__(
            self,
            "admitted_at",
            _finite_time(self.admitted_at, name="admitted_at"),
        )
        _non_negative_int(self.admission_sequence, name="admission_sequence")


def build_pending_group_generation_reservation(
    segment: ReferenceSegment,
    admission: ReferenceSegmentAdmissionReceipt,
    *,
    active_generation: int,
    relation_kind: str,
    predecessor_segment_id: str | None = None,
    supersedes_reservation_id: str | None = None,
) -> PendingGroupGenerationReservation:
    if not admission.accepted:
        raise ValueError("A rejected segment cannot reserve a group generation.")
    if admission.reference_segment_id != segment.reference_segment_id:
        raise ValueError("Reservation segment/admission IDs disagree.")
    active = _non_negative_int(active_generation, name="active_generation")
    if segment.group_segment_generation != active + 1:
        raise ValueError("Segment generation must equal active generation + 1.")
    payload = {
        "control_domain_id": segment.control_domain_id,
        "selection_group_id": segment.selection_group_id,
        "execution_epoch": segment.execution_epoch,
        "active_generation_at_reservation": active,
        "reserved_generation": segment.group_segment_generation,
        "source_segment_id": segment.reference_segment_id,
        "source_segment_hash": segment.semantic_content_hash,
        "admission_id": admission.admission_id,
        "admission_hash": admission.semantic_content_hash,
        "relation_kind": _required_text(relation_kind, name="relation_kind"),
        "predecessor_segment_id": predecessor_segment_id,
        "supersedes_reservation_id": supersedes_reservation_id,
        "effective_from": segment.scheduled_from,
        "admitted_at": admission.admitted_at,
        "admission_sequence": admission.admission_sequence,
        "clock_domain_id": segment.clock_domain_id,
    }
    semantic_hash = canonical_content_hash(
        payload,
        domain=GROUP_GENERATION_RESERVATION_HASH_DOMAIN_V1,
    )
    return PendingGroupGenerationReservation(
        reservation_id=f"group-generation-reservation:{semantic_hash}",
        **payload,
        semantic_content_hash=semantic_hash,
    )


@dataclass(frozen=True)
class GroupGenerationActivation:
    activation_id: str
    reservation_id: str
    source_segment_id: str
    control_domain_id: str
    selection_group_id: str
    execution_epoch: int
    previous_active_generation: int
    active_generation: int
    effective_at: float
    activated_at: float
    activation_sequence: int
    clock_domain_id: str
    semantic_content_hash: str

    def __post_init__(self) -> None:
        for name in (
            "activation_id",
            "reservation_id",
            "source_segment_id",
            "control_domain_id",
            "selection_group_id",
            "clock_domain_id",
            "semantic_content_hash",
        ):
            _required_text(getattr(self, name), name=name)
        _non_negative_int(self.execution_epoch, name="execution_epoch")
        previous = _non_negative_int(
            self.previous_active_generation,
            name="previous_active_generation",
        )
        active = _non_negative_int(self.active_generation, name="active_generation")
        if active != previous + 1:
            raise ValueError("Generation activation must advance exactly once.")
        effective = _finite_time(self.effective_at, name="effective_at")
        activated = _finite_time(self.activated_at, name="activated_at")
        if activated + 1e-12 < effective:
            raise ValueError("Generation activation cannot precede effective_at.")
        _non_negative_int(self.activation_sequence, name="activation_sequence")


def activate_group_generation_reservation(
    reservation: PendingGroupGenerationReservation,
    *,
    current_execution_epoch: int,
    current_active_generation: int,
    activated_at: float,
    activation_sequence: int,
) -> GroupGenerationActivation:
    if current_execution_epoch != reservation.execution_epoch:
        raise ValueError("Reservation execution epoch is no longer active.")
    if current_active_generation != reservation.active_generation_at_reservation:
        raise ValueError("Reservation predecessor generation is no longer active.")
    timestamp = _finite_time(activated_at, name="activated_at")
    if timestamp + 1e-12 < reservation.effective_from:
        raise ValueError("A generation reservation cannot activate before effective_from.")
    payload = {
        "reservation_id": reservation.reservation_id,
        "source_segment_id": reservation.source_segment_id,
        "control_domain_id": reservation.control_domain_id,
        "selection_group_id": reservation.selection_group_id,
        "execution_epoch": reservation.execution_epoch,
        "previous_active_generation": reservation.active_generation_at_reservation,
        "active_generation": reservation.reserved_generation,
        "effective_at": reservation.effective_from,
        "activated_at": timestamp,
        "activation_sequence": _non_negative_int(
            activation_sequence,
            name="activation_sequence",
        ),
        "clock_domain_id": reservation.clock_domain_id,
    }
    semantic_hash = canonical_content_hash(
        payload,
        domain=GROUP_GENERATION_ACTIVATION_HASH_DOMAIN_V1,
    )
    return GroupGenerationActivation(
        activation_id=f"group-generation-activation:{semantic_hash}",
        **payload,
        semantic_content_hash=semantic_hash,
    )


def validate_reference_segment_successor(
    predecessor: ReferenceSegment,
    successor: ReferenceSegment,
) -> None:
    if predecessor.control_domain_id != successor.control_domain_id:
        raise ValueError("Successor control domain mismatch.")
    if predecessor.selection_group_id != successor.selection_group_id:
        raise ValueError("Successor selection group mismatch.")
    if predecessor.execution_epoch != successor.execution_epoch:
        raise ValueError("Successor execution epoch mismatch.")
    if successor.group_segment_generation != predecessor.group_segment_generation + 1:
        raise ValueError("Successor generation must increment by one.")
    if not math.isclose(
        successor.scheduled_from,
        predecessor.scheduled_until,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("Successor interval must begin at predecessor END.")
    if (
        successor.start_parent.source_fact_id != predecessor.reference_segment_id
        or successor.start_parent.projected_group_action_hash
        != predecessor.end_parent.projected_group_action_hash
        or not math.isclose(
            successor.start_parent.target_effective_at,
            predecessor.end_parent.target_effective_at,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise ValueError("Successor START must exactly reference predecessor END.")


@dataclass(frozen=True)
class ReferenceSegmentTerminal:
    terminal_id: str
    reference_segment_id: str
    reference_segment_hash: str
    control_domain_id: str
    selection_group_id: str
    execution_epoch: int
    group_segment_generation: int
    status: str
    available_at: float
    terminal_admission_sequence: int
    effective_at: float
    clock_domain_id: str
    last_reference_frame_id: str | None
    replacement_segment_id: str | None
    evidence_ids: tuple[str, ...]
    semantic_content_hash: str

    def __post_init__(self) -> None:
        for name in (
            "terminal_id",
            "reference_segment_id",
            "reference_segment_hash",
            "control_domain_id",
            "selection_group_id",
            "clock_domain_id",
            "semantic_content_hash",
        ):
            _required_text(getattr(self, name), name=name)
        if self.status not in REFERENCE_SEGMENT_TERMINAL_STATUSES_V1:
            raise ValueError("Unsupported reference segment terminal status.")
        _non_negative_int(self.execution_epoch, name="execution_epoch")
        _non_negative_int(
            self.group_segment_generation,
            name="group_segment_generation",
        )
        object.__setattr__(
            self, "available_at", _finite_time(self.available_at, name="available_at")
        )
        _non_negative_int(
            self.terminal_admission_sequence,
            name="terminal_admission_sequence",
        )
        object.__setattr__(
            self, "effective_at", _finite_time(self.effective_at, name="effective_at")
        )
        for name in ("last_reference_frame_id", "replacement_segment_id"):
            value = getattr(self, name)
            if value is not None:
                _required_text(value, name=name)
        if self.status == "superseded" and self.replacement_segment_id is None:
            raise ValueError("A superseded segment requires replacement_segment_id.")
        if self.replacement_segment_id == self.reference_segment_id:
            raise ValueError("A segment cannot be replaced by itself.")
        evidence = tuple(
            _required_text(value, name="terminal_evidence_id")
            for value in self.evidence_ids
        )
        if len(set(evidence)) != len(evidence):
            raise ValueError("Reference terminal evidence_ids must be unique.")
        object.__setattr__(self, "evidence_ids", evidence)

    @property
    def recorded_at(self) -> float:
        """Compatibility alias for the receiver-side availability time."""

        return self.available_at


def build_reference_segment_terminal(
    segment: ReferenceSegment,
    *,
    status: str,
    effective_at: float,
    terminal_admission_sequence: int,
    available_at: float | None = None,
    recorded_at: float | None = None,
    last_reference_frame_id: str | None = None,
    replacement_segment_id: str | None = None,
    evidence_ids: tuple[str, ...] = (),
) -> ReferenceSegmentTerminal:
    if not isinstance(segment, ReferenceSegment):
        raise TypeError("segment must be a ReferenceSegment.")
    if available_at is None and recorded_at is None:
        raise ValueError("Reference terminal requires available_at.")
    if available_at is not None and recorded_at is not None and not math.isclose(
        float(available_at),
        float(recorded_at),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("available_at and compatibility recorded_at disagree.")
    receiver_available_at = (
        float(recorded_at) if available_at is None else float(available_at)
    )
    payload = {
        "reference_segment_id": segment.reference_segment_id,
        "reference_segment_hash": segment.semantic_content_hash,
        "control_domain_id": segment.control_domain_id,
        "selection_group_id": segment.selection_group_id,
        "execution_epoch": segment.execution_epoch,
        "group_segment_generation": segment.group_segment_generation,
        "status": _required_text(status, name="status"),
        "available_at": _finite_time(
            receiver_available_at,
            name="available_at",
        ),
        "terminal_admission_sequence": _non_negative_int(
            terminal_admission_sequence,
            name="terminal_admission_sequence",
        ),
        "effective_at": _finite_time(effective_at, name="effective_at"),
        "clock_domain_id": segment.clock_domain_id,
        "last_reference_frame_id": (
            None
            if last_reference_frame_id is None
            else _required_text(
                last_reference_frame_id,
                name="last_reference_frame_id",
            )
        ),
        "replacement_segment_id": (
            None
            if replacement_segment_id is None
            else _required_text(
                replacement_segment_id,
                name="replacement_segment_id",
            )
        ),
        "evidence_ids": tuple(
            _required_text(value, name="terminal_evidence_id")
            for value in evidence_ids
        ),
    }
    if payload["status"] not in REFERENCE_SEGMENT_TERMINAL_STATUSES_V1:
        raise ValueError("Unsupported reference segment terminal status.")
    semantic_hash = canonical_content_hash(
        payload,
        domain=REFERENCE_SEGMENT_TERMINAL_HASH_DOMAIN_V1,
    )
    return ReferenceSegmentTerminal(
        terminal_id=f"reference-segment-terminal:{semantic_hash}",
        **payload,
        semantic_content_hash=semantic_hash,
    )


@dataclass(frozen=True)
class SafetyAnchorCommitCandidate:
    candidate_id: str
    reference_segment_id: str
    reference_segment_hash: str
    admission_id: str
    admission_hash: str
    admission_sequence: int
    control_domain_id: str
    selection_group_id: str
    execution_epoch: int
    group_segment_generation: int
    authority_kind: str
    safety_result_id: str
    safe_action: RobotAction
    safe_action_hash: str
    evidence_created_at: float
    effective_from: float
    required_admission_guarantee: str
    achieved_admission_guarantee: str
    semantic_content_hash: str

    def __post_init__(self) -> None:
        for name in (
            "candidate_id",
            "reference_segment_id",
            "reference_segment_hash",
            "admission_id",
            "admission_hash",
            "control_domain_id",
            "selection_group_id",
            "safety_result_id",
            "safe_action_hash",
            "semantic_content_hash",
        ):
            _required_text(getattr(self, name), name=name)
        _non_negative_int(self.admission_sequence, name="admission_sequence")
        _non_negative_int(self.execution_epoch, name="execution_epoch")
        _non_negative_int(
            self.group_segment_generation,
            name="group_segment_generation",
        )
        if self.authority_kind not in REFERENCE_SEGMENT_AUTHORITY_KINDS_V1:
            raise ValueError("Unsupported anchor candidate authority kind.")
        object.__setattr__(
            self,
            "evidence_created_at",
            _finite_time(self.evidence_created_at, name="evidence_created_at"),
        )
        object.__setattr__(
            self,
            "effective_from",
            _finite_time(self.effective_from, name="effective_from"),
        )
        _guarantee_rank(self.required_admission_guarantee)
        _guarantee_rank(self.achieved_admission_guarantee)
        if not admission_guarantee_satisfies(
            self.achieved_admission_guarantee,
            self.required_admission_guarantee,
        ):
            raise ValueError("Anchor candidate admission guarantee is insufficient.")
        action = freeze_robot_action(self.safe_action)
        if safety_anchor_action_hash(action) != self.safe_action_hash:
            raise ValueError("Anchor candidate safe-action hash mismatch.")
        object.__setattr__(self, "safe_action", action)


def build_safety_anchor_commit_candidate(
    segment: ReferenceSegment,
    admission: ReferenceSegmentAdmissionReceipt,
    *,
    safety_result_id: str,
    safe_action: RobotAction,
    required_admission_guarantee: str | None = None,
) -> SafetyAnchorCommitCandidate:
    if not isinstance(segment, ReferenceSegment):
        raise TypeError("segment must be a ReferenceSegment.")
    if not isinstance(admission, ReferenceSegmentAdmissionReceipt):
        raise TypeError("admission must be a ReferenceSegmentAdmissionReceipt.")
    if not admission.accepted:
        raise ValueError("Rejected segment admission cannot create an anchor candidate.")
    if (
        admission.reference_segment_id != segment.reference_segment_id
        or admission.reference_segment_hash != segment.semantic_content_hash
        or admission.control_domain_id != segment.control_domain_id
        or admission.selection_group_id != segment.selection_group_id
        or admission.execution_epoch != segment.execution_epoch
        or admission.group_segment_generation != segment.group_segment_generation
        or admission.clock_domain_id != segment.clock_domain_id
    ):
        raise ValueError("Reference segment admission lineage mismatch.")
    end = segment.end_parent
    expected_safety_result = _required_text(
        safety_result_id, name="safety_result_id"
    )
    if end.source_fact_id != expected_safety_result:
        raise ValueError("Reference END parent references another safety result.")
    action = freeze_robot_action(safe_action)
    action_hash = safety_anchor_action_hash(action)
    if action_hash != end.projected_group_action_hash:
        raise ValueError("Reference END parent safe-action hash mismatch.")
    required = (
        segment.required_admission_guarantee
        if required_admission_guarantee is None
        else _required_text(
            required_admission_guarantee,
            name="required_admission_guarantee",
        )
    )
    if not admission_guarantee_satisfies(
        admission.achieved_admission_guarantee,
        required,
    ):
        raise ValueError("Segment admission does not satisfy anchor policy guarantee.")
    payload = {
        "reference_segment_id": segment.reference_segment_id,
        "reference_segment_hash": segment.semantic_content_hash,
        "admission_id": admission.admission_id,
        "admission_hash": admission.semantic_content_hash,
        "admission_sequence": admission.admission_sequence,
        "control_domain_id": segment.control_domain_id,
        "selection_group_id": segment.selection_group_id,
        "execution_epoch": segment.execution_epoch,
        "group_segment_generation": segment.group_segment_generation,
        "authority_kind": segment.authority_kind,
        "safety_result_id": expected_safety_result,
        "safe_action": robot_action_semantic_payload(action),
        "safe_action_hash": action_hash,
        "evidence_created_at": admission.admitted_at,
        "effective_from": end.target_effective_at,
        "required_admission_guarantee": required,
        "achieved_admission_guarantee": admission.achieved_admission_guarantee,
    }
    semantic_hash = canonical_content_hash(
        payload,
        domain=SAFETY_ANCHOR_COMMIT_CANDIDATE_HASH_DOMAIN_V1,
    )
    return SafetyAnchorCommitCandidate(
        candidate_id=f"safety-anchor-commit-candidate:{semantic_hash}",
        **{key: value for key, value in payload.items() if key != "safe_action"},
        safe_action=action,
        semantic_content_hash=semantic_hash,
    )


@dataclass(frozen=True)
class SafetyAnchorCommitCandidateAssessment:
    candidate_id: str
    status: str
    eligible: bool
    safety_cutoff_at: float
    target_effective_at: float
    terminal_id: str | None
    reason: str | None
    semantic_content_hash: str

    def __post_init__(self) -> None:
        _required_text(self.candidate_id, name="candidate_id")
        if self.status not in {
            "eligible",
            "scope_mismatch",
            "not_yet_available",
            "insufficient_guarantee",
            "pending_effective",
            "superseded",
            "invalidated",
        }:
            raise ValueError("Unsupported anchor candidate assessment status.")
        if bool(self.eligible) != (self.status == "eligible"):
            raise ValueError("Anchor candidate eligibility disagrees with status.")
        object.__setattr__(
            self,
            "safety_cutoff_at",
            _finite_time(self.safety_cutoff_at, name="safety_cutoff_at"),
        )
        object.__setattr__(
            self,
            "target_effective_at",
            _finite_time(self.target_effective_at, name="target_effective_at"),
        )
        for name in ("terminal_id", "reason"):
            value = getattr(self, name)
            if value is not None:
                _required_text(value, name=name)
        _required_text(self.semantic_content_hash, name="semantic_content_hash")


def assess_safety_anchor_commit_candidate(
    candidate: SafetyAnchorCommitCandidate,
    *,
    terminals: Iterable[ReferenceSegmentTerminal] = (),
    safety_cutoff_at: float,
    target_effective_at: float,
    current_execution_epoch: int,
    expected_group_segment_generation: int,
    minimum_admission_guarantee: str,
    safety_cutoff_admission_sequence: int | None = None,
) -> SafetyAnchorCommitCandidateAssessment:
    if not isinstance(candidate, SafetyAnchorCommitCandidate):
        raise TypeError("candidate must be a SafetyAnchorCommitCandidate.")
    cutoff = _finite_time(safety_cutoff_at, name="safety_cutoff_at")
    target_time = _finite_time(target_effective_at, name="target_effective_at")
    minimum = _required_text(
        minimum_admission_guarantee,
        name="minimum_admission_guarantee",
    )
    _guarantee_rank(minimum)
    current_epoch = _non_negative_int(
        current_execution_epoch,
        name="current_execution_epoch",
    )
    expected_generation = _non_negative_int(
        expected_group_segment_generation,
        name="expected_group_segment_generation",
    )
    cutoff_sequence = (
        None
        if safety_cutoff_admission_sequence is None
        else _non_negative_int(
            safety_cutoff_admission_sequence,
            name="safety_cutoff_admission_sequence",
        )
    )
    status = "eligible"
    reason: str | None = None
    terminal_id: str | None = None
    if (
        candidate.execution_epoch != current_epoch
        or candidate.group_segment_generation
        != expected_generation
    ):
        status = "scope_mismatch"
        reason = "execution_epoch_or_generation_mismatch"
    elif candidate.evidence_created_at > cutoff:
        status = "not_yet_available"
        reason = "candidate_evidence_after_safety_cutoff"
    elif not admission_guarantee_satisfies(
        candidate.achieved_admission_guarantee,
        minimum,
    ):
        status = "insufficient_guarantee"
        reason = "anchor_policy_requires_stronger_admission_guarantee"
    else:
        matching: list[ReferenceSegmentTerminal] = []
        for terminal in terminals:
            if not isinstance(terminal, ReferenceSegmentTerminal):
                raise TypeError("terminals must contain ReferenceSegmentTerminal values.")
            if terminal.reference_segment_id != candidate.reference_segment_id:
                continue
            if terminal.reference_segment_hash != candidate.reference_segment_hash:
                raise ValueError("Reference segment terminal hash mismatch.")
            if (
                terminal.control_domain_id != candidate.control_domain_id
                or terminal.selection_group_id != candidate.selection_group_id
                or terminal.execution_epoch != candidate.execution_epoch
                or terminal.group_segment_generation
                != candidate.group_segment_generation
            ):
                raise ValueError("Reference segment terminal scope mismatch.")
            terminal_is_available = terminal.available_at < cutoff or (
                math.isclose(
                    terminal.available_at,
                    cutoff,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and cutoff_sequence is not None
                and terminal.terminal_admission_sequence <= cutoff_sequence
            )
            if (
                terminal_is_available
                and terminal.effective_at <= target_time
                and terminal.effective_at <= candidate.effective_from
                and terminal.status in _INVALIDATING_TERMINAL_STATUSES
            ):
                matching.append(terminal)
        if matching:
            terminal = max(
                matching,
                key=lambda value: (
                    value.effective_at,
                    value.available_at,
                    value.terminal_admission_sequence,
                    value.terminal_id,
                ),
            )
            terminal_id = terminal.terminal_id
            status = (
                "superseded"
                if terminal.status == "superseded"
                else "invalidated"
            )
            reason = f"reference_segment_{terminal.status}"
        elif candidate.effective_from > target_time:
            status = "pending_effective"
            reason = "endpoint_effective_time_not_reached"
    eligible = status == "eligible"
    payload = {
        "candidate_id": candidate.candidate_id,
        "status": status,
        "eligible": eligible,
        "safety_cutoff_at": cutoff,
        "target_effective_at": target_time,
        "terminal_id": terminal_id,
        "reason": reason,
    }
    semantic_hash = canonical_content_hash(
        payload,
        domain=SAFETY_ANCHOR_CANDIDATE_ASSESSMENT_HASH_DOMAIN_V1,
    )
    return SafetyAnchorCommitCandidateAssessment(
        **payload,
        semantic_content_hash=semantic_hash,
    )


@dataclass(frozen=True)
class SafetyAnchorTimelineResolution:
    status: str
    selected_candidate_id: str | None
    selected_candidate: SafetyAnchorCommitCandidate | None
    assessments: tuple[SafetyAnchorCommitCandidateAssessment, ...]
    semantic_content_hash: str

    def __post_init__(self) -> None:
        if self.status not in {
            "eligible",
            "ambiguous_candidates",
            "pending_effective",
            "not_yet_available",
            "no_eligible_candidate",
            "no_candidate",
        }:
            raise ValueError("Unsupported safety-anchor timeline status.")
        assessments = tuple(self.assessments)
        if any(
            not isinstance(value, SafetyAnchorCommitCandidateAssessment)
            for value in assessments
        ):
            raise TypeError("assessments must contain candidate assessment values.")
        if self.status == "eligible":
            if self.selected_candidate is None or self.selected_candidate_id is None:
                raise ValueError("Eligible timeline resolution needs one candidate.")
            if self.selected_candidate.candidate_id != self.selected_candidate_id:
                raise ValueError("Timeline selected candidate ID mismatch.")
        elif self.selected_candidate is not None or self.selected_candidate_id is not None:
            raise ValueError("Non-eligible timeline resolution cannot select a candidate.")
        _required_text(self.semantic_content_hash, name="semantic_content_hash")
        object.__setattr__(self, "assessments", assessments)


def resolve_safety_anchor_commit_timeline(
    candidates: Iterable[SafetyAnchorCommitCandidate],
    *,
    terminals: Iterable[ReferenceSegmentTerminal] = (),
    control_domain_id: str,
    selection_group_id: str,
    safety_cutoff_at: float,
    target_effective_at: float,
    current_execution_epoch: int,
    expected_group_segment_generation: int,
    minimum_admission_guarantee: str,
    safety_cutoff_admission_sequence: int | None = None,
) -> SafetyAnchorTimelineResolution:
    domain = _required_text(control_domain_id, name="control_domain_id")
    group = _required_text(selection_group_id, name="selection_group_id")
    current_epoch = _non_negative_int(
        current_execution_epoch,
        name="current_execution_epoch",
    )
    expected_generation = _non_negative_int(
        expected_group_segment_generation,
        name="expected_group_segment_generation",
    )
    normalized = tuple(candidates)
    terminal_values = tuple(terminals)
    scoped: list[SafetyAnchorCommitCandidate] = []
    for candidate in normalized:
        if not isinstance(candidate, SafetyAnchorCommitCandidate):
            raise TypeError("candidates must contain SafetyAnchorCommitCandidate values.")
        if (
            candidate.control_domain_id == domain
            and candidate.selection_group_id == group
            and candidate.execution_epoch == current_epoch
            and candidate.group_segment_generation
            == expected_generation
        ):
            scoped.append(candidate)
    assessments = tuple(
        assess_safety_anchor_commit_candidate(
            candidate,
            terminals=terminal_values,
            safety_cutoff_at=safety_cutoff_at,
            target_effective_at=target_effective_at,
            current_execution_epoch=current_epoch,
            expected_group_segment_generation=expected_generation,
            minimum_admission_guarantee=minimum_admission_guarantee,
            safety_cutoff_admission_sequence=safety_cutoff_admission_sequence,
        )
        for candidate in sorted(
            scoped,
            key=lambda value: (value.admission_sequence, value.candidate_id),
        )
    )
    eligible_ids = {
        assessment.candidate_id
        for assessment in assessments
        if assessment.eligible
    }
    eligible = tuple(
        candidate for candidate in scoped if candidate.candidate_id in eligible_ids
    )
    selected: SafetyAnchorCommitCandidate | None = None
    if len(eligible) == 1:
        status = "eligible"
        selected = eligible[0]
    elif len(eligible) > 1:
        status = "ambiguous_candidates"
    elif any(value.status == "pending_effective" for value in assessments):
        status = "pending_effective"
    elif any(value.status == "not_yet_available" for value in assessments):
        status = "not_yet_available"
    elif assessments:
        status = "no_eligible_candidate"
    else:
        status = "no_candidate"
    payload = {
        "status": status,
        "selected_candidate_id": (
            None if selected is None else selected.candidate_id
        ),
        "assessment_hashes": tuple(
            value.semantic_content_hash for value in assessments
        ),
        "control_domain_id": domain,
        "selection_group_id": group,
        "safety_cutoff_at": _finite_time(
            safety_cutoff_at, name="safety_cutoff_at"
        ),
        "target_effective_at": _finite_time(
            target_effective_at, name="target_effective_at"
        ),
        "current_execution_epoch": current_epoch,
        "expected_group_segment_generation": expected_generation,
        "minimum_admission_guarantee": _required_text(
            minimum_admission_guarantee,
            name="minimum_admission_guarantee",
        ),
        "safety_cutoff_admission_sequence": (
            None
            if safety_cutoff_admission_sequence is None
            else _non_negative_int(
                safety_cutoff_admission_sequence,
                name="safety_cutoff_admission_sequence",
            )
        ),
    }
    semantic_hash = canonical_content_hash(
        payload,
        domain=SAFETY_ANCHOR_TIMELINE_RESOLUTION_HASH_DOMAIN_V1,
    )
    return SafetyAnchorTimelineResolution(
        status=status,
        selected_candidate_id=payload["selected_candidate_id"],
        selected_candidate=selected,
        assessments=assessments,
        semantic_content_hash=semantic_hash,
    )


def reference_segment_anchor_commit_evidence(
    resolution: SafetyAnchorTimelineResolution,
) -> SafetyAnchorCommitEvidence:
    """Convert one eligible target-time resolution into commit evidence."""

    if not isinstance(resolution, SafetyAnchorTimelineResolution):
        raise TypeError("resolution must be a SafetyAnchorTimelineResolution.")
    candidate = resolution.selected_candidate
    if resolution.status != "eligible" or candidate is None:
        raise ValueError("Only one eligible anchor candidate can create commit evidence.")
    if candidate.authority_kind != "authoritative_shared":
        raise ValueError(
            "Only authoritative_shared segment facts can commit a real safety anchor."
        )
    return SafetyAnchorCommitEvidence(
        evidence_id=(
            "reference-segment-anchor-resolution:"
            f"{resolution.semantic_content_hash}"
        ),
        evidence_kind="reference_segment_endpoint_effective",
        target_id=candidate.reference_segment_id,
        safety_result_id=candidate.safety_result_id,
        safe_action_hash=candidate.safe_action_hash,
        control_domain_id=candidate.control_domain_id,
        selection_group_id=candidate.selection_group_id,
        execution_epoch=candidate.execution_epoch,
        group_segment_generation=candidate.group_segment_generation,
        occurred_at=candidate.evidence_created_at,
        accepted=True,
        effective_from=candidate.effective_from,
    )


__all__ = [
    "GROUP_GENERATION_ACTIVATION_HASH_DOMAIN_V1",
    "GROUP_GENERATION_RESERVATION_HASH_DOMAIN_V1",
    "REFERENCE_ADMISSION_GUARANTEES_V1",
    "REFERENCE_ENDPOINT_ROLES_V1",
    "REFERENCE_ENDPOINT_SOURCE_KINDS_V1",
    "REFERENCE_SEGMENT_AUTHORITY_KINDS_V1",
    "REFERENCE_SEGMENT_SAFETY_EVIDENCE_LEVELS_V1",
    "REFERENCE_SEGMENT_TERMINAL_STATUSES_V1",
    "REFERENCE_SEGMENT_RELATION_KINDS_V1",
    "GroupGenerationActivation",
    "PendingGroupGenerationReservation",
    "ReferenceEndpointParent",
    "ReferenceSegment",
    "ReferenceSegmentAdmissionReceipt",
    "ReferenceSegmentTerminal",
    "SafetyAnchorCommitCandidate",
    "SafetyAnchorCommitCandidateAssessment",
    "SafetyAnchorTimelineResolution",
    "admission_guarantee_satisfies",
    "admit_reference_segment",
    "activate_group_generation_reservation",
    "assess_safety_anchor_commit_candidate",
    "build_reference_endpoint_parent",
    "build_pending_group_generation_reservation",
    "build_reference_segment",
    "build_reference_segment_terminal",
    "build_safety_anchor_commit_candidate",
    "reference_segment_anchor_commit_evidence",
    "resolve_safety_anchor_commit_timeline",
    "validate_reference_segment_successor",
]
