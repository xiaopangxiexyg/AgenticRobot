"""Thin, backend-neutral handoff into a target execution owner.

The contracts in this module begin after authoritative action adaptation.  They
do not prescribe safety, IK, smoothing, reference generation, command
construction, or dispatch.  A concrete target owner may perform any subset of
those stages and reports them only as opaque processing facts.

Submission, target resolution, target admission, activation, and dispatch are
separate facts.  In particular, accepting a submission is not target admission
and target admission is not motion permission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Mapping, Protocol

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.runtime import CommandDispatchReceipt
from robot_motion.runtime.immutable import deep_freeze

from .adaptation import ActionAdaptationResult
from .target_capability import ComponentTargetCapabilitySpec


TARGET_EXECUTION_REQUEST_HASH_DOMAIN = (
    "dexteleop.target_execution_request.development"
)
TARGET_EXECUTION_SUBMISSION_RECEIPT_HASH_DOMAIN = (
    "dexteleop.target_execution_submission_receipt.development"
)
TARGET_PROCESSING_FACT_REFERENCE_HASH_DOMAIN = (
    "dexteleop.target_processing_fact_reference.development"
)
COMPONENT_EXECUTION_TARGET_PAYLOAD_HASH_DOMAIN = (
    "dexteleop.component_execution_target_payload.development"
)
COMPONENT_EXECUTION_TARGET_HASH_DOMAIN = (
    "dexteleop.component_execution_target.development"
)
COMPONENT_EXECUTION_TARGET_ADMISSION_HASH_DOMAIN = (
    "dexteleop.component_execution_target_admission.development"
)
TARGET_EXECUTION_COMPLETION_HASH_DOMAIN = (
    "dexteleop.target_execution_completion.development"
)
COMPONENT_EXECUTION_TARGET_ACTIVATION_HASH_DOMAIN = (
    "dexteleop.component_execution_target_activation.development"
)
TARGET_EXECUTION_STAGE_STATUS_HASH_DOMAIN = (
    "dexteleop.target_execution_stage_status.development"
)
TARGET_EXECUTION_STATUS_SNAPSHOT_HASH_DOMAIN = (
    "dexteleop.target_execution_status_snapshot.development"
)

TARGET_TEMPORAL_SEMANTICS = frozenset({"point_target", "scheduled_interval"})
TARGET_EXECUTION_COMPLETION_STATUSES = frozenset(
    {"resolved", "rejected", "failed", "cancelled"}
)
TARGET_ACTIVATION_STATUSES = frozenset({"activated", "rejected"})
TARGET_ACTIVATION_MODES = frozenset({"required", "omitted"})
TARGET_EXECUTION_STATUS_STAGES = (
    "request",
    "submission",
    "completion",
    "admission",
    "activation",
)
TARGET_EXECUTION_STAGE_STATUSES = MappingProxyType(
    {
        "request": frozenset({"submitted"}),
        "submission": frozenset(
            {"pending", "accepted_for_processing", "rejected"}
        ),
        "completion": frozenset(
            {
                "pending",
                "not_applicable",
                *TARGET_EXECUTION_COMPLETION_STATUSES,
            }
        ),
        "admission": frozenset(
            {"pending", "not_applicable", "accepted", "rejected"}
        ),
        "activation": frozenset(
            {
                "pending",
                "not_applicable",
                *TARGET_ACTIVATION_STATUSES,
            }
        ),
    }
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _optional(value: object | None, *, name: str) -> str | None:
    if value is None:
        return None
    return _required(value, name=name)


def _finite(value: object, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


def _sequence(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _unique_identifiers(
    values: tuple[str, ...],
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    normalized = tuple(_required(value, name=name) for value in values)
    if not allow_empty and not normalized:
        raise ValueError(f"{name} must be non-empty.")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must be unique.")
    return normalized


@dataclass(frozen=True)
class TargetProcessingFactReference:
    """Opaque evidence emitted by one target-private processing stage."""

    stage: str
    fact_id: str
    fact_hash: str
    status: str
    processing_started_at: float
    processing_completed_at: float
    clock_domain_id: str
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        started = _finite(
            self.processing_started_at,
            name="processing_started_at",
        )
        completed = _finite(
            self.processing_completed_at,
            name="processing_completed_at",
        )
        if completed < started:
            raise ValueError("Target processing cannot complete before it starts.")
        payload = {
            "stage": _required(self.stage, name="stage"),
            "fact_id": _required(self.fact_id, name="fact_id"),
            "fact_hash": _required(self.fact_hash, name="fact_hash"),
            "status": _required(self.status, name="status"),
            "processing_started_at": started,
            "processing_completed_at": completed,
            "clock_domain_id": _required(
                self.clock_domain_id,
                name="clock_domain_id",
            ),
        }
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=TARGET_PROCESSING_FACT_REFERENCE_HASH_DOMAIN,
            ),
        )


@dataclass(frozen=True)
class TargetExecutionRequest:
    """One accepted adapted candidate handed to exactly one target owner."""

    control_domain_id: str
    target_handler_id: str
    adaptation_result: ActionAdaptationResult = field(repr=False, compare=False)
    submitted_at: float = 0.0
    clock_domain_id: str = ""
    request_id: str = field(init=False)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.adaptation_result, ActionAdaptationResult):
            raise TypeError("adaptation_result must be an ActionAdaptationResult.")
        adaptation = self.adaptation_result
        if not adaptation.receipt.accepted or adaptation.adapted_candidate is None:
            raise ValueError(
                "Target execution requires an accepted adaptation result."
            )
        clock = _required(self.clock_domain_id, name="clock_domain_id")
        if (
            adaptation.request.clock_domain_id != clock
            or adaptation.receipt.clock_domain_id != clock
            or adaptation.adapted_candidate.clock_domain_id != clock
        ):
            raise ValueError(
                "Target execution request and adaptation must use one clock domain."
            )
        submitted = _finite(self.submitted_at, name="submitted_at")
        if submitted < adaptation.receipt.processing_completed_at:
            raise ValueError(
                "Target execution cannot be submitted before adaptation completes."
            )
        if submitted >= adaptation.request.supported_until:
            raise ValueError(
                "Target execution cannot be submitted after support expires."
            )
        payload = {
            "control_domain_id": _required(
                self.control_domain_id,
                name="control_domain_id",
            ),
            "target_handler_id": _required(
                self.target_handler_id,
                name="target_handler_id",
            ),
            "authoritative_decision_id": (
                adaptation.request.authoritative_decision_id
            ),
            "authoritative_decision_hash": (
                adaptation.request.authoritative_decision_hash
            ),
            "adaptation_request_id": adaptation.request.request_id,
            "adaptation_request_hash": adaptation.request.semantic_content_hash,
            "capability_assessment_id": (
                adaptation.receipt.capability_assessment_id
            ),
            "capability_assessment_hash": (
                adaptation.receipt.capability_assessment_hash
            ),
            "adaptation_receipt_id": adaptation.receipt.receipt_id,
            "adaptation_receipt_hash": adaptation.receipt.semantic_content_hash,
            "adapted_candidate_id": (
                adaptation.adapted_candidate.adapted_candidate_id
            ),
            "adapted_candidate_hash": (
                adaptation.adapted_candidate.semantic_content_hash
            ),
            "candidate_action_hash": adaptation.request.candidate_action_hash,
            "action_schema_hash": adaptation.request.action_schema_hash,
            "component_target_capability_id": (
                adaptation.request.component_target_capability_id
            ),
            "component_target_capability_hash": (
                adaptation.request.component_target_capability_hash
            ),
            "component_coverage": adaptation.request.component_coverage,
            "target_effective_at": adaptation.request.target_effective_at,
            "supported_until": adaptation.request.supported_until,
            "authority_sequence": adaptation.request.authority_sequence,
            "execution_gate_sequence": (
                adaptation.request.execution_gate_sequence
            ),
            "execution_fence_sequence": (
                adaptation.request.execution_fence_sequence
            ),
            "submitted_at": submitted,
            "clock_domain_id": clock,
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=TARGET_EXECUTION_REQUEST_HASH_DOMAIN,
        )
        for name, value in (
            ("control_domain_id", payload["control_domain_id"]),
            ("target_handler_id", payload["target_handler_id"]),
            ("submitted_at", submitted),
            ("clock_domain_id", clock),
            ("request_id", f"target-execution-request:{semantic_hash}"),
            ("semantic_content_hash", semantic_hash),
        ):
            object.__setattr__(self, name, value)

    @property
    def adapted_candidate(self):
        candidate = self.adaptation_result.adapted_candidate
        assert candidate is not None
        return candidate

    @property
    def supported_until(self) -> float:
        return self.adaptation_result.request.supported_until


@dataclass(frozen=True)
class TargetExecutionSubmissionReceipt:
    """Receiver acknowledgement; never target admission or motion permission."""

    target_execution_request_id: str
    target_execution_request_hash: str
    control_domain_id: str
    target_handler_id: str
    received_at: float
    clock_domain_id: str
    accepted_for_processing: bool
    rejection_code: str | None
    processing_token: str | None
    receipt_id: str = field(init=False)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        accepted = bool(self.accepted_for_processing)
        rejection = _optional(self.rejection_code, name="rejection_code")
        token = _optional(self.processing_token, name="processing_token")
        if accepted:
            if rejection is not None or token is None:
                raise ValueError(
                    "Accepted target submission requires only a processing token."
                )
        elif rejection is None or token is not None:
            raise ValueError(
                "Rejected target submission requires only a rejection code."
            )
        payload = {
            "target_execution_request_id": _required(
                self.target_execution_request_id,
                name="target_execution_request_id",
            ),
            "target_execution_request_hash": _required(
                self.target_execution_request_hash,
                name="target_execution_request_hash",
            ),
            "control_domain_id": _required(
                self.control_domain_id,
                name="control_domain_id",
            ),
            "target_handler_id": _required(
                self.target_handler_id,
                name="target_handler_id",
            ),
            "received_at": _finite(self.received_at, name="received_at"),
            "clock_domain_id": _required(
                self.clock_domain_id,
                name="clock_domain_id",
            ),
            "accepted_for_processing": accepted,
            "rejection_code": rejection,
            "processing_token": token,
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=TARGET_EXECUTION_SUBMISSION_RECEIPT_HASH_DOMAIN,
        )
        for name, value in (
            *payload.items(),
            ("receipt_id", f"target-execution-submission:{semantic_hash}"),
            ("semantic_content_hash", semantic_hash),
        ):
            object.__setattr__(self, name, value)


def build_target_execution_submission_receipt(
    request: TargetExecutionRequest,
    *,
    received_at: float,
    accepted_for_processing: bool,
    rejection_code: str | None = None,
    processing_token: str | None = None,
) -> TargetExecutionSubmissionReceipt:
    if not isinstance(request, TargetExecutionRequest):
        raise TypeError("request must be a TargetExecutionRequest.")
    received = _finite(received_at, name="received_at")
    if received < request.submitted_at:
        raise ValueError("Target request cannot be received before submission.")
    if accepted_for_processing and received >= request.supported_until:
        raise ValueError(
            "An expired target request cannot be accepted for processing."
        )
    return TargetExecutionSubmissionReceipt(
        target_execution_request_id=request.request_id,
        target_execution_request_hash=request.semantic_content_hash,
        control_domain_id=request.control_domain_id,
        target_handler_id=request.target_handler_id,
        received_at=received,
        clock_domain_id=request.clock_domain_id,
        accepted_for_processing=accepted_for_processing,
        rejection_code=rejection_code,
        processing_token=processing_token,
    )


@dataclass(frozen=True)
class TargetExecutionCompletionReceipt:
    """Terminal outcome for one accepted target-processing submission."""

    target_execution_request_id: str
    target_execution_request_hash: str
    submission_receipt_id: str
    submission_receipt_hash: str
    processing_token: str
    control_domain_id: str
    target_handler_id: str
    status: str
    completed_at: float
    available_at: float
    clock_domain_id: str
    target_id: str | None = None
    target_hash: str | None = None
    outcome_code: str | None = None
    receipt_id: str = field(init=False)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        status = _required(self.status, name="status")
        if status not in TARGET_EXECUTION_COMPLETION_STATUSES:
            raise ValueError("Unsupported target execution completion status.")
        target_id = _optional(self.target_id, name="target_id")
        target_hash = _optional(self.target_hash, name="target_hash")
        outcome_code = _optional(self.outcome_code, name="outcome_code")
        if status == "resolved":
            if target_id is None or target_hash is None or outcome_code is not None:
                raise ValueError(
                    "Resolved completion requires a target and no outcome code."
                )
        elif target_id is not None or target_hash is not None or outcome_code is None:
            raise ValueError(
                "Non-resolved completion requires an outcome code and no target."
            )
        completed = _finite(self.completed_at, name="completed_at")
        available = _finite(self.available_at, name="available_at")
        if available < completed:
            raise ValueError(
                "Completion cannot be available before it completes."
            )
        payload = {
            "target_execution_request_id": _required(
                self.target_execution_request_id,
                name="target_execution_request_id",
            ),
            "target_execution_request_hash": _required(
                self.target_execution_request_hash,
                name="target_execution_request_hash",
            ),
            "submission_receipt_id": _required(
                self.submission_receipt_id,
                name="submission_receipt_id",
            ),
            "submission_receipt_hash": _required(
                self.submission_receipt_hash,
                name="submission_receipt_hash",
            ),
            "processing_token": _required(
                self.processing_token,
                name="processing_token",
            ),
            "control_domain_id": _required(
                self.control_domain_id,
                name="control_domain_id",
            ),
            "target_handler_id": _required(
                self.target_handler_id,
                name="target_handler_id",
            ),
            "status": status,
            "completed_at": completed,
            "available_at": available,
            "clock_domain_id": _required(
                self.clock_domain_id,
                name="clock_domain_id",
            ),
            "target_id": target_id,
            "target_hash": target_hash,
            "outcome_code": outcome_code,
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=TARGET_EXECUTION_COMPLETION_HASH_DOMAIN,
        )
        for name, value in (
            *payload.items(),
            ("receipt_id", f"target-execution-completion:{semantic_hash}"),
            ("semantic_content_hash", semantic_hash),
        ):
            object.__setattr__(self, name, value)


def build_target_execution_completion_receipt(
    request: TargetExecutionRequest,
    submission_receipt: TargetExecutionSubmissionReceipt,
    *,
    status: str,
    completed_at: float,
    available_at: float,
    target: ComponentExecutionTarget | None = None,
    outcome_code: str | None = None,
) -> TargetExecutionCompletionReceipt:
    if not isinstance(request, TargetExecutionRequest):
        raise TypeError("request must be a TargetExecutionRequest.")
    if not isinstance(
        submission_receipt,
        TargetExecutionSubmissionReceipt,
    ):
        raise TypeError(
            "submission_receipt must be a TargetExecutionSubmissionReceipt."
        )
    if (
        not submission_receipt.accepted_for_processing
        or submission_receipt.target_execution_request_id != request.request_id
        or submission_receipt.target_execution_request_hash
        != request.semantic_content_hash
    ):
        raise ValueError(
            "Completion requires the accepted receipt for this exact request."
        )
    completed = _finite(completed_at, name="completed_at")
    if completed < submission_receipt.received_at:
        raise ValueError("Completion cannot precede submission receipt time.")
    if status == "resolved":
        if not isinstance(target, ComponentExecutionTarget):
            raise TypeError("Resolved completion requires a ComponentExecutionTarget.")
        if (
            target.target_execution_request_id != request.request_id
            or target.target_execution_request_hash != request.semantic_content_hash
            or target.submission_receipt_id != submission_receipt.receipt_id
            or target.submission_receipt_hash
            != submission_receipt.semantic_content_hash
        ):
            raise ValueError("Resolved target does not match the submission.")
        if completed < target.resolved_at:
            raise ValueError("Completion cannot precede target resolution.")
    elif target is not None:
        raise ValueError("Non-resolved completion cannot carry a target.")
    return TargetExecutionCompletionReceipt(
        target_execution_request_id=request.request_id,
        target_execution_request_hash=request.semantic_content_hash,
        submission_receipt_id=submission_receipt.receipt_id,
        submission_receipt_hash=submission_receipt.semantic_content_hash,
        processing_token=submission_receipt.processing_token or "",
        control_domain_id=request.control_domain_id,
        target_handler_id=request.target_handler_id,
        status=status,
        completed_at=completed,
        available_at=available_at,
        clock_domain_id=request.clock_domain_id,
        target_id=None if target is None else target.target_id,
        target_hash=None if target is None else target.semantic_content_hash,
        outcome_code=outcome_code,
    )


@dataclass(frozen=True)
class ComponentExecutionTarget:
    """Component-keyed result of target-private processing.

    The target carries no ``safe`` claim.  Optional safety, IK, planner, limiter,
    or backend preparation evidence appears only in
    ``processing_fact_references``.
    """

    target_execution_request_id: str
    target_execution_request_hash: str
    submission_receipt_id: str
    submission_receipt_hash: str
    authoritative_decision_id: str
    authoritative_decision_hash: str
    adaptation_request_id: str
    adaptation_request_hash: str
    adaptation_receipt_id: str
    adaptation_receipt_hash: str
    adapted_candidate_id: str
    adapted_candidate_hash: str
    capability_assessment_id: str
    capability_assessment_hash: str
    component_target_capability_id: str
    component_target_capability_hash: str
    control_domain_id: str
    target_handler_id: str
    component_dependencies: tuple[str, ...]
    input_action_schema_hash: str
    output_target_schema_hash: str
    target_kind: str
    target_effective_at: float
    supported_until: float
    authority_sequence: int
    execution_gate_sequence: int
    execution_fence_sequence: int
    resolved_at: float
    temporal_semantics: str
    scheduled_for: float
    scheduled_until: float | None
    clock_domain_id: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    processing_fact_references: tuple[TargetProcessingFactReference, ...] = ()
    target_id: str = field(init=False)
    payload_hash: str = field(init=False)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        text_fields = (
            "target_execution_request_id",
            "target_execution_request_hash",
            "submission_receipt_id",
            "submission_receipt_hash",
            "authoritative_decision_id",
            "authoritative_decision_hash",
            "adaptation_request_id",
            "adaptation_request_hash",
            "adaptation_receipt_id",
            "adaptation_receipt_hash",
            "adapted_candidate_id",
            "adapted_candidate_hash",
            "capability_assessment_id",
            "capability_assessment_hash",
            "component_target_capability_id",
            "component_target_capability_hash",
            "control_domain_id",
            "target_handler_id",
            "input_action_schema_hash",
            "output_target_schema_hash",
            "target_kind",
            "clock_domain_id",
        )
        normalized_text = {
            name: _required(getattr(self, name), name=name) for name in text_fields
        }
        dependencies = _unique_identifiers(
            tuple(self.component_dependencies),
            name="component_dependency",
        )
        facts = tuple(self.processing_fact_references)
        if any(not isinstance(item, TargetProcessingFactReference) for item in facts):
            raise TypeError(
                "processing_fact_references must contain "
                "TargetProcessingFactReference values."
            )
        fact_hashes = tuple(item.semantic_content_hash for item in facts)
        if len(fact_hashes) != len(set(fact_hashes)):
            raise ValueError("processing_fact_references must be unique.")
        target_effective_at = _finite(
            self.target_effective_at,
            name="target_effective_at",
        )
        supported_until = _finite(
            self.supported_until,
            name="supported_until",
        )
        if supported_until <= target_effective_at:
            raise ValueError(
                "Execution-target support must extend beyond its target."
            )
        resolved_at = _finite(self.resolved_at, name="resolved_at")
        scheduled_for = _finite(self.scheduled_for, name="scheduled_for")
        if scheduled_for >= supported_until:
            raise ValueError(
                "Execution target cannot be scheduled after support expires."
            )
        temporal_semantics = _required(
            self.temporal_semantics,
            name="temporal_semantics",
        )
        if temporal_semantics not in TARGET_TEMPORAL_SEMANTICS:
            raise ValueError("Unsupported target temporal semantics.")
        scheduled_until = self.scheduled_until
        if temporal_semantics == "point_target":
            if scheduled_until is not None:
                raise ValueError("A point target cannot claim a scheduled interval.")
        else:
            if scheduled_until is None:
                raise ValueError(
                    "A scheduled interval requires scheduled_until."
                )
            scheduled_until = _finite(
                scheduled_until,
                name="scheduled_until",
            )
            if scheduled_until <= scheduled_for:
                raise ValueError(
                    "Target interval must be [scheduled_for, scheduled_until)."
                )
            if scheduled_until > supported_until:
                raise ValueError(
                    "Target interval cannot exceed upstream support."
                )
        payload = deep_freeze(dict(self.payload))
        payload_hash = canonical_content_hash(
            payload,
            domain=COMPONENT_EXECUTION_TARGET_PAYLOAD_HASH_DOMAIN,
        )
        semantic_payload = {
            **normalized_text,
            "component_dependencies": dependencies,
            "target_effective_at": target_effective_at,
            "supported_until": supported_until,
            "authority_sequence": _sequence(
                self.authority_sequence,
                name="authority_sequence",
            ),
            "execution_gate_sequence": _sequence(
                self.execution_gate_sequence,
                name="execution_gate_sequence",
            ),
            "execution_fence_sequence": _sequence(
                self.execution_fence_sequence,
                name="execution_fence_sequence",
            ),
            "resolved_at": resolved_at,
            "temporal_semantics": temporal_semantics,
            "scheduled_for": scheduled_for,
            "scheduled_until": scheduled_until,
            "payload_hash": payload_hash,
            "processing_fact_hashes": fact_hashes,
        }
        semantic_hash = canonical_content_hash(
            semantic_payload,
            domain=COMPONENT_EXECUTION_TARGET_HASH_DOMAIN,
        )
        for name, value in (
            *normalized_text.items(),
            ("component_dependencies", dependencies),
            ("target_effective_at", target_effective_at),
            ("supported_until", supported_until),
            ("authority_sequence", semantic_payload["authority_sequence"]),
            (
                "execution_gate_sequence",
                semantic_payload["execution_gate_sequence"],
            ),
            (
                "execution_fence_sequence",
                semantic_payload["execution_fence_sequence"],
            ),
            ("resolved_at", resolved_at),
            ("temporal_semantics", temporal_semantics),
            ("scheduled_for", scheduled_for),
            ("scheduled_until", scheduled_until),
            ("payload", MappingProxyType(dict(payload))),
            ("processing_fact_references", facts),
            ("payload_hash", payload_hash),
            ("target_id", f"component-execution-target:{semantic_hash}"),
            ("semantic_content_hash", semantic_hash),
        ):
            object.__setattr__(self, name, value)


def build_component_execution_target(
    request: TargetExecutionRequest,
    submission_receipt: TargetExecutionSubmissionReceipt,
    *,
    component_dependencies: tuple[str, ...],
    output_target_schema_hash: str,
    target_kind: str,
    resolved_at: float,
    temporal_semantics: str,
    scheduled_for: float,
    scheduled_until: float | None,
    payload: Mapping[str, Any],
    processing_fact_references: tuple[TargetProcessingFactReference, ...] = (),
) -> ComponentExecutionTarget:
    if not isinstance(request, TargetExecutionRequest):
        raise TypeError("request must be a TargetExecutionRequest.")
    if not isinstance(
        submission_receipt,
        TargetExecutionSubmissionReceipt,
    ):
        raise TypeError(
            "submission_receipt must be a TargetExecutionSubmissionReceipt."
        )
    if (
        not submission_receipt.accepted_for_processing
        or submission_receipt.target_execution_request_id != request.request_id
        or submission_receipt.target_execution_request_hash
        != request.semantic_content_hash
        or submission_receipt.control_domain_id != request.control_domain_id
        or submission_receipt.target_handler_id != request.target_handler_id
        or submission_receipt.clock_domain_id != request.clock_domain_id
    ):
        raise ValueError(
            "Execution target requires the accepted receipt for this exact request."
        )
    dependencies = _unique_identifiers(
        tuple(component_dependencies),
        name="component_dependency",
    )
    adaptation = request.adaptation_result
    if not set(dependencies).issubset(adaptation.request.component_coverage):
        raise ValueError(
            "Execution target dependencies must be covered by the candidate."
        )
    resolved = _finite(resolved_at, name="resolved_at")
    if resolved < request.submitted_at:
        raise ValueError("Execution target cannot resolve before submission.")
    return ComponentExecutionTarget(
        target_execution_request_id=request.request_id,
        target_execution_request_hash=request.semantic_content_hash,
        submission_receipt_id=submission_receipt.receipt_id,
        submission_receipt_hash=submission_receipt.semantic_content_hash,
        authoritative_decision_id=(
            adaptation.request.authoritative_decision_id
        ),
        authoritative_decision_hash=(
            adaptation.request.authoritative_decision_hash
        ),
        adaptation_request_id=adaptation.request.request_id,
        adaptation_request_hash=adaptation.request.semantic_content_hash,
        adaptation_receipt_id=adaptation.receipt.receipt_id,
        adaptation_receipt_hash=adaptation.receipt.semantic_content_hash,
        adapted_candidate_id=request.adapted_candidate.adapted_candidate_id,
        adapted_candidate_hash=request.adapted_candidate.semantic_content_hash,
        capability_assessment_id=adaptation.receipt.capability_assessment_id,
        capability_assessment_hash=adaptation.receipt.capability_assessment_hash,
        component_target_capability_id=(
            adaptation.request.component_target_capability_id
        ),
        component_target_capability_hash=(
            adaptation.request.component_target_capability_hash
        ),
        control_domain_id=request.control_domain_id,
        target_handler_id=request.target_handler_id,
        component_dependencies=dependencies,
        input_action_schema_hash=adaptation.request.action_schema_hash,
        output_target_schema_hash=output_target_schema_hash,
        target_kind=target_kind,
        target_effective_at=adaptation.request.target_effective_at,
        supported_until=adaptation.request.supported_until,
        authority_sequence=adaptation.request.authority_sequence,
        execution_gate_sequence=adaptation.request.execution_gate_sequence,
        execution_fence_sequence=adaptation.request.execution_fence_sequence,
        resolved_at=resolved,
        temporal_semantics=temporal_semantics,
        scheduled_for=scheduled_for,
        scheduled_until=scheduled_until,
        clock_domain_id=request.clock_domain_id,
        payload=payload,
        processing_fact_references=processing_fact_references,
    )


@dataclass(frozen=True)
class ComponentExecutionTargetAdmissionReceipt:
    """Exact target receiver decision; activation and dispatch remain separate."""

    target_id: str
    target_hash: str
    target_execution_request_id: str
    target_execution_request_hash: str
    control_domain_id: str
    target_handler_id: str
    receiver_id: str
    execution_fence_sequence: int
    admission_sequence: int
    admitted_at: float
    available_at: float
    clock_domain_id: str
    accepted: bool
    rejection_code: str | None
    admission_token: str | None
    receipt_id: str = field(init=False)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        accepted = bool(self.accepted)
        rejection = _optional(self.rejection_code, name="rejection_code")
        token = _optional(self.admission_token, name="admission_token")
        if accepted:
            if rejection is not None or token is None:
                raise ValueError(
                    "Accepted target admission requires only an admission token."
                )
        elif rejection is None or token is not None:
            raise ValueError(
                "Rejected target admission requires only a rejection code."
            )
        admitted = _finite(self.admitted_at, name="admitted_at")
        available = _finite(self.available_at, name="available_at")
        if available < admitted:
            raise ValueError("Target admission cannot be available before decision.")
        payload = {
            "target_id": _required(self.target_id, name="target_id"),
            "target_hash": _required(self.target_hash, name="target_hash"),
            "target_execution_request_id": _required(
                self.target_execution_request_id,
                name="target_execution_request_id",
            ),
            "target_execution_request_hash": _required(
                self.target_execution_request_hash,
                name="target_execution_request_hash",
            ),
            "control_domain_id": _required(
                self.control_domain_id,
                name="control_domain_id",
            ),
            "target_handler_id": _required(
                self.target_handler_id,
                name="target_handler_id",
            ),
            "receiver_id": _required(self.receiver_id, name="receiver_id"),
            "execution_fence_sequence": _sequence(
                self.execution_fence_sequence,
                name="execution_fence_sequence",
            ),
            "admission_sequence": _sequence(
                self.admission_sequence,
                name="admission_sequence",
            ),
            "admitted_at": admitted,
            "available_at": available,
            "clock_domain_id": _required(
                self.clock_domain_id,
                name="clock_domain_id",
            ),
            "accepted": accepted,
            "rejection_code": rejection,
            "admission_token": token,
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=COMPONENT_EXECUTION_TARGET_ADMISSION_HASH_DOMAIN,
        )
        for name, value in (
            *payload.items(),
            ("receipt_id", f"component-target-admission:{semantic_hash}"),
            ("semantic_content_hash", semantic_hash),
        ):
            object.__setattr__(self, name, value)


def admit_component_execution_target(
    target: ComponentExecutionTarget,
    *,
    receiver_id: str,
    admission_sequence: int,
    admitted_at: float,
    available_at: float,
    accepted: bool,
    rejection_code: str | None = None,
    admission_token: str | None = None,
) -> ComponentExecutionTargetAdmissionReceipt:
    if not isinstance(target, ComponentExecutionTarget):
        raise TypeError("target must be a ComponentExecutionTarget.")
    admitted = _finite(admitted_at, name="admitted_at")
    if admitted < target.resolved_at:
        raise ValueError("Execution target cannot be admitted before resolution.")
    if accepted and admitted >= target.supported_until:
        raise ValueError(
            "An expired execution target cannot be admitted."
        )
    return ComponentExecutionTargetAdmissionReceipt(
        target_id=target.target_id,
        target_hash=target.semantic_content_hash,
        target_execution_request_id=target.target_execution_request_id,
        target_execution_request_hash=target.target_execution_request_hash,
        control_domain_id=target.control_domain_id,
        target_handler_id=target.target_handler_id,
        receiver_id=receiver_id,
        execution_fence_sequence=target.execution_fence_sequence,
        admission_sequence=admission_sequence,
        admitted_at=admitted,
        available_at=available_at,
        clock_domain_id=target.clock_domain_id,
        accepted=accepted,
        rejection_code=rejection_code,
        admission_token=admission_token,
    )


@dataclass(frozen=True)
class ComponentExecutionTargetActivationReceipt:
    """Target activation fact; it is not a command or motion permit."""

    target_id: str
    target_hash: str
    target_execution_request_id: str
    target_execution_request_hash: str
    target_admission_receipt_id: str
    target_admission_receipt_hash: str
    control_domain_id: str
    target_handler_id: str
    receiver_id: str
    activation_sequence: int
    activation_status: str
    activation_at: float
    available_at: float
    clock_domain_id: str
    effective_from: float | None = None
    effective_until: float | None = None
    rejection_code: str | None = None
    activation_id: str = field(init=False)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        status = _required(self.activation_status, name="activation_status")
        if status not in TARGET_ACTIVATION_STATUSES:
            raise ValueError("Unsupported target activation status.")
        rejection = _optional(self.rejection_code, name="rejection_code")
        effective_from = self.effective_from
        effective_until = self.effective_until
        if status == "activated":
            if rejection is not None or effective_from is None:
                raise ValueError(
                    "Activated target requires effective_from and no rejection."
                )
            effective_from = _finite(effective_from, name="effective_from")
            if effective_until is not None:
                effective_until = _finite(
                    effective_until,
                    name="effective_until",
                )
                if effective_until <= effective_from:
                    raise ValueError(
                        "Activation interval must be [effective_from, effective_until)."
                    )
        elif rejection is None or effective_from is not None or effective_until is not None:
            raise ValueError(
                "Rejected activation requires a rejection code and no interval."
            )
        activated = _finite(self.activation_at, name="activation_at")
        available = _finite(self.available_at, name="available_at")
        if available < activated:
            raise ValueError(
                "Activation cannot be available before its activation decision."
            )
        payload = {
            "target_id": _required(self.target_id, name="target_id"),
            "target_hash": _required(self.target_hash, name="target_hash"),
            "target_execution_request_id": _required(
                self.target_execution_request_id,
                name="target_execution_request_id",
            ),
            "target_execution_request_hash": _required(
                self.target_execution_request_hash,
                name="target_execution_request_hash",
            ),
            "target_admission_receipt_id": _required(
                self.target_admission_receipt_id,
                name="target_admission_receipt_id",
            ),
            "target_admission_receipt_hash": _required(
                self.target_admission_receipt_hash,
                name="target_admission_receipt_hash",
            ),
            "control_domain_id": _required(
                self.control_domain_id,
                name="control_domain_id",
            ),
            "target_handler_id": _required(
                self.target_handler_id,
                name="target_handler_id",
            ),
            "receiver_id": _required(self.receiver_id, name="receiver_id"),
            "activation_sequence": _sequence(
                self.activation_sequence,
                name="activation_sequence",
            ),
            "activation_status": status,
            "activation_at": activated,
            "available_at": available,
            "clock_domain_id": _required(
                self.clock_domain_id,
                name="clock_domain_id",
            ),
            "effective_from": effective_from,
            "effective_until": effective_until,
            "rejection_code": rejection,
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=COMPONENT_EXECUTION_TARGET_ACTIVATION_HASH_DOMAIN,
        )
        for name, value in (
            *payload.items(),
            ("activation_id", f"component-target-activation:{semantic_hash}"),
            ("semantic_content_hash", semantic_hash),
        ):
            object.__setattr__(self, name, value)


def build_component_execution_target_activation_receipt(
    target: ComponentExecutionTarget,
    admission: ComponentExecutionTargetAdmissionReceipt,
    *,
    receiver_id: str,
    activation_sequence: int,
    activation_status: str,
    activation_at: float,
    available_at: float,
    effective_from: float | None = None,
    effective_until: float | None = None,
    rejection_code: str | None = None,
) -> ComponentExecutionTargetActivationReceipt:
    if not isinstance(target, ComponentExecutionTarget):
        raise TypeError("target must be a ComponentExecutionTarget.")
    if not isinstance(
        admission,
        ComponentExecutionTargetAdmissionReceipt,
    ):
        raise TypeError(
            "admission must be a ComponentExecutionTargetAdmissionReceipt."
        )
    if (
        not admission.accepted
        or admission.target_id != target.target_id
        or admission.target_hash != target.semantic_content_hash
        or admission.target_execution_request_id
        != target.target_execution_request_id
        or admission.target_execution_request_hash
        != target.target_execution_request_hash
    ):
        raise ValueError("Activation requires the accepted receipt for this target.")
    activated = _finite(activation_at, name="activation_at")
    if activated < admission.admitted_at:
        raise ValueError("Activation cannot precede target admission.")
    if activation_status == "activated":
        if activated >= target.supported_until:
            raise ValueError(
                "An expired execution target cannot be activated."
            )
        if effective_from is None or float(effective_from) >= target.supported_until:
            raise ValueError(
                "Target activation must start inside upstream support."
            )
        if (
            effective_until is not None
            and float(effective_until) > target.supported_until
        ):
            raise ValueError(
                "Target activation cannot exceed upstream support."
            )
    return ComponentExecutionTargetActivationReceipt(
        target_id=target.target_id,
        target_hash=target.semantic_content_hash,
        target_execution_request_id=target.target_execution_request_id,
        target_execution_request_hash=target.target_execution_request_hash,
        target_admission_receipt_id=admission.receipt_id,
        target_admission_receipt_hash=admission.semantic_content_hash,
        control_domain_id=target.control_domain_id,
        target_handler_id=target.target_handler_id,
        receiver_id=receiver_id,
        activation_sequence=activation_sequence,
        activation_status=activation_status,
        activation_at=activated,
        available_at=available_at,
        clock_domain_id=target.clock_domain_id,
        effective_from=effective_from,
        effective_until=effective_until,
        rejection_code=rejection_code,
    )


@dataclass(frozen=True)
class TargetExecutionStageStatus:
    """Read-only projection of one exact target-execution stage fact.

    ``pending`` and ``not_applicable`` are snapshot states and therefore have
    no record identity or availability timestamp.  Every observed durable fact
    retains its exact ID, hash, and availability time.
    """

    stage: str
    status: str
    record_id: str | None = None
    record_hash: str | None = None
    available_at: float | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        stage = _required(self.stage, name="stage")
        if stage not in TARGET_EXECUTION_STATUS_STAGES:
            raise ValueError("Unsupported target execution status stage.")
        status = _required(self.status, name="status")
        if status not in TARGET_EXECUTION_STAGE_STATUSES[stage]:
            raise ValueError(
                f"Unsupported {stage!r} target execution stage status."
            )
        record_id = _optional(self.record_id, name="record_id")
        record_hash = _optional(self.record_hash, name="record_hash")
        available_at = self.available_at
        if status in {"pending", "not_applicable"}:
            if (
                record_id is not None
                or record_hash is not None
                or available_at is not None
            ):
                raise ValueError(
                    "Unobserved target execution stages cannot carry a record."
                )
        else:
            if record_id is None or record_hash is None or available_at is None:
                raise ValueError(
                    "Observed target execution stages require an exact record."
                )
            available_at = _finite(available_at, name="available_at")
        payload = {
            "stage": stage,
            "status": status,
            "record_id": record_id,
            "record_hash": record_hash,
            "available_at": available_at,
        }
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=TARGET_EXECUTION_STAGE_STATUS_HASH_DOMAIN,
            ),
        )


@dataclass(frozen=True)
class TargetExecutionStatusSnapshot:
    """One causally closed, read-only view of a single execution request.

    The snapshot is observability only.  It is not a receipt, target admission,
    activation permission, command, authority decision, or safety verdict.
    """

    control_domain_id: str
    target_handler_id: str
    clock_domain_id: str
    activation_mode: str
    captured_at: float
    request: TargetExecutionStageStatus
    submission: TargetExecutionStageStatus
    completion: TargetExecutionStageStatus
    admission: TargetExecutionStageStatus
    activation: TargetExecutionStageStatus
    target_id: str | None = None
    target_hash: str | None = None
    snapshot_id: str = field(init=False)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        stages = (
            self.request,
            self.submission,
            self.completion,
            self.admission,
            self.activation,
        )
        if any(
            not isinstance(stage, TargetExecutionStageStatus)
            for stage in stages
        ):
            raise TypeError(
                "Target execution snapshot stages must be stage status values."
            )
        if tuple(stage.stage for stage in stages) != (
            TARGET_EXECUTION_STATUS_STAGES
        ):
            raise ValueError(
                "Target execution snapshot stages must be in lifecycle order."
            )
        mode = _required(self.activation_mode, name="activation_mode")
        if mode not in TARGET_ACTIVATION_MODES:
            raise ValueError("Unsupported target activation mode.")
        captured_at = _finite(self.captured_at, name="captured_at")
        observed_times = tuple(
            stage.available_at
            for stage in stages
            if stage.available_at is not None
        )
        if observed_times and captured_at < max(observed_times):
            raise ValueError(
                "A status snapshot cannot precede an observed stage fact."
            )
        target_id = _optional(self.target_id, name="target_id")
        target_hash = _optional(self.target_hash, name="target_hash")
        if (target_id is None) != (target_hash is None):
            raise ValueError("target_id and target_hash must appear together.")

        submission_status = self.submission.status
        completion_status = self.completion.status
        admission_status = self.admission.status
        activation_status = self.activation.status
        if submission_status == "pending":
            expected = (
                "pending",
                "pending",
                "pending" if mode == "required" else "not_applicable",
            )
            if (
                completion_status,
                admission_status,
                activation_status,
            ) != expected:
                raise ValueError(
                    "Pending submission requires all later stages to be pending."
                )
        elif submission_status == "rejected":
            if (
                completion_status,
                admission_status,
                activation_status,
            ) != ("not_applicable", "not_applicable", "not_applicable"):
                raise ValueError(
                    "Rejected submission makes all later stages not applicable."
                )
        elif completion_status == "pending":
            expected_activation = (
                "pending" if mode == "required" else "not_applicable"
            )
            if admission_status != "pending" or (
                activation_status != expected_activation
            ):
                raise ValueError(
                    "Pending completion requires later stages to be pending."
                )
        elif completion_status != "resolved":
            if admission_status != "not_applicable" or (
                activation_status != "not_applicable"
            ):
                raise ValueError(
                    "Non-resolved completion makes target stages not applicable."
                )
        elif target_id is None:
            raise ValueError("Resolved completion requires a target identity.")
        elif admission_status == "pending":
            expected_activation = (
                "pending" if mode == "required" else "not_applicable"
            )
            if activation_status != expected_activation:
                raise ValueError(
                    "Pending admission has an inconsistent activation state."
                )
        elif admission_status == "rejected":
            if activation_status != "not_applicable":
                raise ValueError(
                    "Rejected target admission makes activation not applicable."
                )
        elif mode == "omitted":
            if activation_status != "not_applicable":
                raise ValueError(
                    "Targets with omitted activation cannot report activation."
                )
        elif activation_status not in {"pending", *TARGET_ACTIVATION_STATUSES}:
            raise ValueError("Accepted target admission has invalid activation.")

        payload = {
            "control_domain_id": _required(
                self.control_domain_id,
                name="control_domain_id",
            ),
            "target_handler_id": _required(
                self.target_handler_id,
                name="target_handler_id",
            ),
            "clock_domain_id": _required(
                self.clock_domain_id,
                name="clock_domain_id",
            ),
            "activation_mode": mode,
            "captured_at": captured_at,
            "request_status_hash": self.request.semantic_content_hash,
            "submission_status_hash": self.submission.semantic_content_hash,
            "completion_status_hash": self.completion.semantic_content_hash,
            "admission_status_hash": self.admission.semantic_content_hash,
            "activation_status_hash": self.activation.semantic_content_hash,
            "target_id": target_id,
            "target_hash": target_hash,
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=TARGET_EXECUTION_STATUS_SNAPSHOT_HASH_DOMAIN,
        )
        for name, value in (
            ("control_domain_id", payload["control_domain_id"]),
            ("target_handler_id", payload["target_handler_id"]),
            ("clock_domain_id", payload["clock_domain_id"]),
            ("activation_mode", mode),
            ("captured_at", captured_at),
            ("target_id", target_id),
            ("target_hash", target_hash),
            (
                "snapshot_id",
                f"target-execution-status-snapshot:{semantic_hash}",
            ),
            ("semantic_content_hash", semantic_hash),
        ):
            object.__setattr__(self, name, value)

    @property
    def request_id(self) -> str:
        assert self.request.record_id is not None
        return self.request.record_id


def build_target_execution_status_snapshot(
    request: TargetExecutionRequest,
    *,
    activation_mode: str,
    captured_at: float,
    submission_receipt: TargetExecutionSubmissionReceipt | None = None,
    completion_receipt: TargetExecutionCompletionReceipt | None = None,
    target_admission: ComponentExecutionTargetAdmissionReceipt | None = None,
    target_activation: ComponentExecutionTargetActivationReceipt | None = None,
) -> TargetExecutionStatusSnapshot:
    """Project exact lifecycle facts for one request without changing them."""

    if not isinstance(request, TargetExecutionRequest):
        raise TypeError("request must be a TargetExecutionRequest.")
    mode = _required(activation_mode, name="activation_mode")
    if mode not in TARGET_ACTIVATION_MODES:
        raise ValueError("Unsupported target activation mode.")

    def pending(stage: str) -> TargetExecutionStageStatus:
        return TargetExecutionStageStatus(stage=stage, status="pending")

    def not_applicable(stage: str) -> TargetExecutionStageStatus:
        return TargetExecutionStageStatus(
            stage=stage,
            status="not_applicable",
        )

    request_status = TargetExecutionStageStatus(
        stage="request",
        status="submitted",
        record_id=request.request_id,
        record_hash=request.semantic_content_hash,
        available_at=request.submitted_at,
    )
    submission_status = pending("submission")
    completion_status = pending("completion")
    admission_status = pending("admission")
    activation_status = (
        pending("activation")
        if mode == "required"
        else not_applicable("activation")
    )
    target_id: str | None = None
    target_hash: str | None = None

    if submission_receipt is not None:
        if not isinstance(
            submission_receipt,
            TargetExecutionSubmissionReceipt,
        ):
            raise TypeError(
                "submission_receipt must be a "
                "TargetExecutionSubmissionReceipt or None."
            )
        if (
            submission_receipt.target_execution_request_id
            != request.request_id
            or submission_receipt.target_execution_request_hash
            != request.semantic_content_hash
            or submission_receipt.control_domain_id
            != request.control_domain_id
            or submission_receipt.target_handler_id
            != request.target_handler_id
            or submission_receipt.clock_domain_id
            != request.clock_domain_id
        ):
            raise ValueError("Submission receipt belongs to another request.")
        submission_status = TargetExecutionStageStatus(
            stage="submission",
            status=(
                "accepted_for_processing"
                if submission_receipt.accepted_for_processing
                else "rejected"
            ),
            record_id=submission_receipt.receipt_id,
            record_hash=submission_receipt.semantic_content_hash,
            available_at=submission_receipt.received_at,
        )
        if not submission_receipt.accepted_for_processing:
            completion_status = not_applicable("completion")
            admission_status = not_applicable("admission")
            activation_status = not_applicable("activation")

    if completion_receipt is not None:
        if submission_receipt is None or not (
            submission_receipt.accepted_for_processing
        ):
            raise ValueError(
                "Completion requires an observed accepted submission."
            )
        if not isinstance(
            completion_receipt,
            TargetExecutionCompletionReceipt,
        ):
            raise TypeError(
                "completion_receipt must be a "
                "TargetExecutionCompletionReceipt or None."
            )
        if (
            completion_receipt.target_execution_request_id
            != request.request_id
            or completion_receipt.target_execution_request_hash
            != request.semantic_content_hash
            or completion_receipt.submission_receipt_id
            != submission_receipt.receipt_id
            or completion_receipt.submission_receipt_hash
            != submission_receipt.semantic_content_hash
        ):
            raise ValueError("Completion receipt belongs to another request.")
        completion_status = TargetExecutionStageStatus(
            stage="completion",
            status=completion_receipt.status,
            record_id=completion_receipt.receipt_id,
            record_hash=completion_receipt.semantic_content_hash,
            available_at=completion_receipt.available_at,
        )
        if completion_receipt.status == "resolved":
            target_id = completion_receipt.target_id
            target_hash = completion_receipt.target_hash
            admission_status = pending("admission")
            activation_status = (
                pending("activation")
                if mode == "required"
                else not_applicable("activation")
            )
        else:
            admission_status = not_applicable("admission")
            activation_status = not_applicable("activation")

    if target_admission is not None:
        if (
            completion_receipt is None
            or completion_receipt.status != "resolved"
        ):
            raise ValueError(
                "Target admission requires an observed resolved completion."
            )
        if not isinstance(
            target_admission,
            ComponentExecutionTargetAdmissionReceipt,
        ):
            raise TypeError(
                "target_admission must be a "
                "ComponentExecutionTargetAdmissionReceipt or None."
            )
        if (
            target_admission.target_execution_request_id
            != request.request_id
            or target_admission.target_execution_request_hash
            != request.semantic_content_hash
            or target_admission.target_id != target_id
            or target_admission.target_hash != target_hash
        ):
            raise ValueError("Target admission belongs to another request.")
        admission_status = TargetExecutionStageStatus(
            stage="admission",
            status="accepted" if target_admission.accepted else "rejected",
            record_id=target_admission.receipt_id,
            record_hash=target_admission.semantic_content_hash,
            available_at=target_admission.available_at,
        )
        if not target_admission.accepted:
            activation_status = not_applicable("activation")

    if target_activation is not None:
        if mode != "required":
            raise ValueError(
                "An activation receipt conflicts with omitted activation mode."
            )
        if target_admission is None or not target_admission.accepted:
            raise ValueError(
                "Target activation requires an observed accepted admission."
            )
        if not isinstance(
            target_activation,
            ComponentExecutionTargetActivationReceipt,
        ):
            raise TypeError(
                "target_activation must be a "
                "ComponentExecutionTargetActivationReceipt or None."
            )
        if (
            target_activation.target_execution_request_id
            != request.request_id
            or target_activation.target_execution_request_hash
            != request.semantic_content_hash
            or target_activation.target_id != target_id
            or target_activation.target_hash != target_hash
            or target_activation.target_admission_receipt_id
            != target_admission.receipt_id
            or target_activation.target_admission_receipt_hash
            != target_admission.semantic_content_hash
        ):
            raise ValueError("Target activation belongs to another request.")
        activation_status = TargetExecutionStageStatus(
            stage="activation",
            status=target_activation.activation_status,
            record_id=target_activation.activation_id,
            record_hash=target_activation.semantic_content_hash,
            available_at=target_activation.available_at,
        )

    return TargetExecutionStatusSnapshot(
        control_domain_id=request.control_domain_id,
        target_handler_id=request.target_handler_id,
        clock_domain_id=request.clock_domain_id,
        activation_mode=mode,
        captured_at=captured_at,
        request=request_status,
        submission=submission_status,
        completion=completion_status,
        admission=admission_status,
        activation=activation_status,
        target_id=target_id,
        target_hash=target_hash,
    )


class TargetExecutionStatusProvider(Protocol):
    """Read-only observation port, separate from submission and execution."""

    def capture_target_execution_status_snapshot(
        self,
        *,
        now: float | None = None,
    ) -> TargetExecutionStatusSnapshot | None:
        ...


class TargetExecutionPort(Protocol):
    """Minimal asynchronous-friendly port owned by one concrete target."""

    @property
    def target_handler_id(self) -> str:
        ...

    @property
    def activation_mode(self) -> str:
        """Either ``required`` or ``omitted`` for this target capability."""
        ...

    @property
    def component_target_capability(self) -> ComponentTargetCapabilitySpec:
        ...

    def submit(
        self,
        request: TargetExecutionRequest,
    ) -> TargetExecutionSubmissionReceipt:
        ...


@dataclass(frozen=True)
class TargetExecutionDispatchResult:
    """Runtime envelope returned by a target-aware contextual dispatcher.

    This envelope is deliberately not another authoritative fact.  It groups
    the exact terminal processing receipt, target admission, optional
    activation receipt, and backend receipts produced during one already
    permitted dispatch call.  Their individual typed contracts remain the
    durable evidence.
    """

    completion_receipt: TargetExecutionCompletionReceipt
    target: ComponentExecutionTarget
    target_admission: ComponentExecutionTargetAdmissionReceipt
    dispatch_receipts: Mapping[str, CommandDispatchReceipt]
    target_activation: ComponentExecutionTargetActivationReceipt | None = None

    def __post_init__(self) -> None:
        completion = self.completion_receipt
        target = self.target
        admission = self.target_admission
        activation = self.target_activation
        if not isinstance(completion, TargetExecutionCompletionReceipt):
            raise TypeError(
                "completion_receipt must be a TargetExecutionCompletionReceipt."
            )
        if not isinstance(target, ComponentExecutionTarget):
            raise TypeError("target must be a ComponentExecutionTarget.")
        if not isinstance(
            admission,
            ComponentExecutionTargetAdmissionReceipt,
        ):
            raise TypeError(
                "target_admission must be a "
                "ComponentExecutionTargetAdmissionReceipt."
            )
        if (
            completion.status != "resolved"
            or completion.target_id != target.target_id
            or completion.target_hash != target.semantic_content_hash
            or admission.target_id != target.target_id
            or admission.target_hash != target.semantic_content_hash
        ):
            raise ValueError(
                "Dispatch result must bind one resolved target and its exact "
                "admission receipt."
            )
        if activation is not None:
            if not isinstance(
                activation,
                ComponentExecutionTargetActivationReceipt,
            ):
                raise TypeError(
                    "target_activation must be a "
                    "ComponentExecutionTargetActivationReceipt or None."
                )
            if (
                activation.target_id != target.target_id
                or activation.target_hash != target.semantic_content_hash
                or activation.target_admission_receipt_id
                != admission.receipt_id
                or activation.target_admission_receipt_hash
                != admission.semantic_content_hash
            ):
                raise ValueError(
                    "Target activation does not bind this dispatch target."
                )
        receipts = dict(self.dispatch_receipts)
        if any(
            not isinstance(item, CommandDispatchReceipt)
            for item in receipts.values()
        ):
            raise TypeError(
                "dispatch_receipts must contain CommandDispatchReceipt values."
            )
        object.__setattr__(
            self,
            "dispatch_receipts",
            MappingProxyType(receipts),
        )


__all__ = [
    "COMPONENT_EXECUTION_TARGET_ADMISSION_HASH_DOMAIN",
    "COMPONENT_EXECUTION_TARGET_HASH_DOMAIN",
    "COMPONENT_EXECUTION_TARGET_PAYLOAD_HASH_DOMAIN",
    "TARGET_EXECUTION_REQUEST_HASH_DOMAIN",
    "TARGET_EXECUTION_SUBMISSION_RECEIPT_HASH_DOMAIN",
    "TARGET_EXECUTION_COMPLETION_HASH_DOMAIN",
    "TARGET_EXECUTION_COMPLETION_STATUSES",
    "TARGET_ACTIVATION_MODES",
    "TARGET_ACTIVATION_STATUSES",
    "COMPONENT_EXECUTION_TARGET_ACTIVATION_HASH_DOMAIN",
    "TARGET_PROCESSING_FACT_REFERENCE_HASH_DOMAIN",
    "TARGET_TEMPORAL_SEMANTICS",
    "ComponentExecutionTarget",
    "ComponentExecutionTargetActivationReceipt",
    "ComponentExecutionTargetAdmissionReceipt",
    "TargetExecutionCompletionReceipt",
    "TargetExecutionDispatchResult",
    "TargetExecutionPort",
    "TargetExecutionRequest",
    "TargetExecutionSubmissionReceipt",
    "TargetProcessingFactReference",
    "admit_component_execution_target",
    "build_component_execution_target",
    "build_component_execution_target_activation_receipt",
    "build_target_execution_completion_receipt",
    "build_target_execution_submission_receipt",
]
