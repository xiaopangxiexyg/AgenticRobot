"""Bounded evidence projection for the exact CubeLift Agent Skill Route.

This observer consumes the immutable envelope returned by the existing
authoritative Sim-HIL application and the already-configured passive execution
difference observer.  It owns no queue, policy, authority, safety, target,
dispatch, physics, or trace lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_gateway.skills.cube_lift.definition import (
    CUBE_LIFT_EXECUTION_RESULT_SCHEMA_V1,
    validate_cube_lift_execution_result_v1,
)
from dataset_contracts.canonical_encoding import canonical_content_hash

from .terminal_evaluator import CubeLiftLowerHoldProof

CUBE_LIFT_EXECUTION_FACT_CHAIN_HASH_DOMAIN_V1 = (
    "dexteleop.sim_hil.cube_lift_execution_fact_chain.v1"
)
CUBE_LIFT_EXECUTION_EVIDENCE_HASH_DOMAIN_V1 = (
    "dexteleop.sim_hil.cube_lift_execution_evidence.v1"
)
CUBE_LIFT_TERMINAL_CONTROL_STATE_HASH_DOMAIN_V1 = (
    "dexteleop.sim_hil.cube_lift_terminal_control_state.v1"
)

_AUTHORITY_EVENT_TYPES = frozenset(
    {
        "proposal_snapshot",
        "decision_schedule_tick",
        "authority_state_snapshot",
        "execution_gate_snapshot",
        "arbitration_input_snapshot",
        "arbitration_computation",
        "arbitration_publication_attempt",
        "arbitration_decision",
        "execution_directive",
    }
)
_TARGET_REQUEST_EVENT_TYPES = frozenset(
    {
        "action_adaptation_request",
        "target_capability_assessment",
        "adapted_action_candidate",
        "action_adaptation_receipt",
        "target_execution_request",
    }
)
_TARGET_SUBMISSION_EVENT_TYPES = frozenset({"target_execution_submission"})
_TARGET_COMPLETION_EVENT_TYPES = frozenset({"target_execution_completion"})
_RESOLVED_TARGET_EVENT_TYPES = frozenset({"component_execution_target"})
_TARGET_ADMISSION_EVENT_TYPES = frozenset(
    {"component_execution_target_admission"}
)
_SAFETY_EVENT_TYPES = frozenset({"shared_safety_result"})
_DISPATCH_EVENT_TYPES = frozenset(
    {"command", "dispatch_owner", "dispatch_receipt"}
)


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot contain surrounding whitespace.")
    return value


def _nonnegative_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return value


@dataclass(frozen=True)
class CubeLiftExecutionFactRef:
    kind: str
    fact_id: str
    fact_hash: str

    def __post_init__(self) -> None:
        for field_name in ("kind", "fact_id", "fact_hash"):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )

    def to_wire(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "fact_id": self.fact_id,
            "fact_hash": self.fact_hash,
        }


def _ref(*, kind: str, value: Any, id_field: str) -> CubeLiftExecutionFactRef:
    return CubeLiftExecutionFactRef(
        kind=kind,
        fact_id=_required(getattr(value, id_field, None), field_name=id_field),
        fact_hash=_required(
            getattr(value, "semantic_content_hash", None),
            field_name=f"{kind}.semantic_content_hash",
        ),
    )


def _trace_ref(fact: Any) -> CubeLiftExecutionFactRef:
    return CubeLiftExecutionFactRef(
        kind=_required(getattr(fact, "event_type", None), field_name="event_type"),
        fact_id=_required(getattr(fact, "fact_id", None), field_name="fact_id"),
        fact_hash=_required(
            getattr(fact, "semantic_content_hash", None),
            field_name="trace semantic_content_hash",
        ),
    )


def _required_trace_refs(
    facts: tuple[Any, ...],
    *,
    event_types: frozenset[str],
    category: str,
) -> tuple[CubeLiftExecutionFactRef, ...]:
    selected = tuple(
        _trace_ref(fact)
        for fact in facts
        if getattr(fact, "event_type", None) in event_types
    )
    observed = {item.kind for item in selected}
    missing = sorted(event_types - observed)
    if missing:
        raise ValueError(
            f"CubeLift {category} trace evidence is incomplete: {missing}."
        )
    return selected


def _chain_hash(
    category: str,
    references: tuple[CubeLiftExecutionFactRef, ...],
) -> str:
    if not references:
        raise ValueError(f"CubeLift {category} evidence chain must not be empty.")
    return canonical_content_hash(
        {
            "category": category,
            "references": tuple(reference.to_wire() for reference in references),
        },
        domain=CUBE_LIFT_EXECUTION_FACT_CHAIN_HASH_DOMAIN_V1,
    )


@dataclass(frozen=True)
class CubeLiftExecutionCycleEvidence:
    cycle_index: int
    cycle_receipt_ref: str
    dispatch_receipt_ref: str
    source_refs: tuple[CubeLiftExecutionFactRef, ...]
    authority_refs: tuple[CubeLiftExecutionFactRef, ...]
    safety_refs: tuple[CubeLiftExecutionFactRef, ...]
    target_request_refs: tuple[CubeLiftExecutionFactRef, ...]
    target_submission_refs: tuple[CubeLiftExecutionFactRef, ...]
    target_completion_refs: tuple[CubeLiftExecutionFactRef, ...]
    resolved_target_refs: tuple[CubeLiftExecutionFactRef, ...]
    target_admission_refs: tuple[CubeLiftExecutionFactRef, ...]
    dispatch_refs: tuple[CubeLiftExecutionFactRef, ...]
    measurement_ref: CubeLiftExecutionFactRef

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cycle_index",
            _nonnegative_integer(self.cycle_index, field_name="cycle_index"),
        )
        for field_name in ("cycle_receipt_ref", "dispatch_receipt_ref"):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        for field_name in (
            "source_refs",
            "authority_refs",
            "safety_refs",
            "target_request_refs",
            "target_submission_refs",
            "target_completion_refs",
            "resolved_target_refs",
            "target_admission_refs",
            "dispatch_refs",
        ):
            values = tuple(getattr(self, field_name))
            if not values or any(
                not isinstance(item, CubeLiftExecutionFactRef) for item in values
            ):
                raise TypeError(f"{field_name} must contain evidence references.")
            object.__setattr__(self, field_name, values)
        if not isinstance(self.measurement_ref, CubeLiftExecutionFactRef):
            raise TypeError("measurement_ref must be an evidence reference.")


@dataclass(frozen=True)
class CubeLiftExecutionEvidenceCut:
    runtime_session_id: str
    control_domain_id: str
    execution_evidence_chain_ref: str
    source_proposal_chain_hash: str
    authority_decision_chain_hash: str
    safety_decision_chain_hash: str
    target_request_chain_hash: str
    target_submission_receipt_chain_hash: str
    target_completion_receipt_chain_hash: str
    resolved_component_target_chain_hash: str
    component_admission_receipt_chain_hash: str
    dispatch_receipt_chain_hash: str
    measurement_lineage_chain_hash: str
    first_cycle_receipt_ref: str
    last_cycle_receipt_ref: str
    terminal_state_hash: str
    cycle_count: int
    dispatched_cycle_count: int
    measurement_count: int

    def __post_init__(self) -> None:
        for field_name in (
            "runtime_session_id",
            "control_domain_id",
            "execution_evidence_chain_ref",
            "source_proposal_chain_hash",
            "authority_decision_chain_hash",
            "safety_decision_chain_hash",
            "target_request_chain_hash",
            "target_submission_receipt_chain_hash",
            "target_completion_receipt_chain_hash",
            "resolved_component_target_chain_hash",
            "component_admission_receipt_chain_hash",
            "dispatch_receipt_chain_hash",
            "measurement_lineage_chain_hash",
            "first_cycle_receipt_ref",
            "last_cycle_receipt_ref",
            "terminal_state_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        for field_name in (
            "cycle_count",
            "dispatched_cycle_count",
            "measurement_count",
        ):
            value = _nonnegative_integer(
                getattr(self, field_name),
                field_name=field_name,
            )
            if value <= 0:
                raise ValueError(f"{field_name} must be positive.")
            object.__setattr__(self, field_name, value)
        if self.dispatched_cycle_count > self.cycle_count:
            raise ValueError("Dispatched cycles cannot exceed observed cycles.")
        if self.measurement_count < self.dispatched_cycle_count:
            raise ValueError("Every dispatched cycle requires measurement coverage.")

    @property
    def execution_receipt_refs(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    self.execution_evidence_chain_ref,
                    self.first_cycle_receipt_ref,
                    self.last_cycle_receipt_ref,
                )
            )
        )

    def to_execution_result(
        self,
        *,
        lower_hold_proof: CubeLiftLowerHoldProof,
        trace_finalization: Any,
    ) -> dict[str, object]:
        if not isinstance(lower_hold_proof, CubeLiftLowerHoldProof):
            raise TypeError("lower_hold_proof must be a CubeLiftLowerHoldProof.")
        if lower_hold_proof.runtime_session_id != self.runtime_session_id:
            raise ValueError("HOLD proof runtime session mismatch.")
        if lower_hold_proof.control_domain_id != self.control_domain_id:
            raise ValueError("HOLD proof control domain mismatch.")
        for field_name, expected in (
            ("runtime_session_id", self.runtime_session_id),
            ("control_domain_id", self.control_domain_id),
        ):
            if getattr(trace_finalization, field_name, None) != expected:
                raise ValueError(f"Trace finalization {field_name} mismatch.")
        result = {
            "schema_version": CUBE_LIFT_EXECUTION_RESULT_SCHEMA_V1,
            "runtime_session_id": self.runtime_session_id,
            "control_domain_id": self.control_domain_id,
            "execution_evidence_chain_ref": self.execution_evidence_chain_ref,
            "source_proposal_chain_hash": self.source_proposal_chain_hash,
            "authority_decision_chain_hash": self.authority_decision_chain_hash,
            "safety_decision_chain_hash": self.safety_decision_chain_hash,
            "target_request_chain_hash": self.target_request_chain_hash,
            "target_submission_receipt_chain_hash": (
                self.target_submission_receipt_chain_hash
            ),
            "target_completion_receipt_chain_hash": (
                self.target_completion_receipt_chain_hash
            ),
            "resolved_component_target_chain_hash": (
                self.resolved_component_target_chain_hash
            ),
            "component_admission_receipt_chain_hash": (
                self.component_admission_receipt_chain_hash
            ),
            "dispatch_receipt_chain_hash": self.dispatch_receipt_chain_hash,
            "measurement_lineage_chain_hash": (
                self.measurement_lineage_chain_hash
            ),
            "first_cycle_receipt_ref": self.first_cycle_receipt_ref,
            "last_cycle_receipt_ref": self.last_cycle_receipt_ref,
            "terminal_hold_receipt_ref": lower_hold_proof.hold_receipt_ref,
            "terminal_state_hash": self.terminal_state_hash,
            "trace_terminal_receipt_ref": _required(
                getattr(trace_finalization, "trace_terminal_receipt_ref", None),
                field_name="trace_terminal_receipt_ref",
            ),
            "trace_commit_receipt_ref": _required(
                getattr(trace_finalization, "trace_commit_receipt_ref", None),
                field_name="trace_commit_receipt_ref",
            ),
            "cycle_count": self.cycle_count,
            "dispatched_cycle_count": self.dispatched_cycle_count,
            "measurement_count": self.measurement_count,
        }
        validate_cube_lift_execution_result_v1(result)
        return result


class CubeLiftExecutionEvidenceCollector:
    """Single-use, bounded observer over authoritative cycle return values."""

    def __init__(self, *, runtime_session_id: str, control_domain_id: str) -> None:
        self.runtime_session_id = _required(
            runtime_session_id,
            field_name="runtime_session_id",
        )
        self.control_domain_id = _required(
            control_domain_id,
            field_name="control_domain_id",
        )
        self._cycles: list[CubeLiftExecutionCycleEvidence] = []
        self._measurement_hashes: set[str] = set()
        self._last_observer_measurement_count = 0
        self._sealed = False

    @staticmethod
    def _extend(
        cycles: tuple[CubeLiftExecutionCycleEvidence, ...],
        field_name: str,
    ) -> tuple[CubeLiftExecutionFactRef, ...]:
        return tuple(
            reference
            for cycle in cycles
            for reference in getattr(cycle, field_name)
        )

    def observe_cycle(self, cycle: Any, *, measurement_snapshot: Any) -> None:
        if self._sealed:
            raise RuntimeError("CubeLift execution evidence is already sealed.")
        cycle_index = _nonnegative_integer(
            getattr(cycle, "cycle_index", None),
            field_name="cycle_index",
        )
        if cycle_index != len(self._cycles):
            raise ValueError("CubeLift cycle indexes must be consecutive from zero.")
        cycle_hash = _required(
            getattr(cycle, "semantic_content_hash", None),
            field_name="cycle.semantic_content_hash",
        )
        composition = getattr(cycle, "composition_step", None)
        control_cut = getattr(composition, "control_cut", None)
        if (
            control_cut is None
            or getattr(control_cut, "runtime_session_id", None)
            != self.runtime_session_id
            or getattr(control_cut, "control_domain_id", None)
            != self.control_domain_id
        ):
            raise ValueError("CubeLift control cut session/domain mismatch.")
        policy = getattr(cycle, "policy_submission", None)
        if (
            policy is None
            or getattr(policy, "role", None) != "policy"
            or not bool(getattr(policy, "accepted", False))
        ):
            raise ValueError("CubeLift requires an accepted policy submission.")
        input_receipt = getattr(policy, "input_receipt", None)
        disposition = getattr(composition, "disposition", None)
        proposal_admission = getattr(composition, "proposal_admission", None)
        if (
            getattr(composition, "blocked_reason", None) is not None
            or getattr(composition, "error", None) is not None
            or getattr(composition, "hold_transition", None) is not None
            or getattr(disposition, "action", None) != "publish_proposal"
            or proposal_admission is None
            or not bool(getattr(proposal_admission, "accepted", False))
        ):
            raise ValueError("CubeLift composition did not publish one accepted proposal.")
        proposal = getattr(proposal_admission, "proposal", None)
        if proposal is None:
            raise ValueError("Accepted CubeLift proposal admission has no proposal.")

        execution = getattr(cycle, "execution", None)
        if execution is None or getattr(execution, "status", None) != "dispatched":
            raise ValueError("CubeLift evidence accepts dispatched cycles only.")
        if (
            getattr(execution, "cancellation", None) is not None
            or getattr(execution, "cancellation_receipt", None) is not None
        ):
            raise ValueError("A dispatched CubeLift cycle cannot include cancellation.")
        safety = getattr(execution, "safety_result", None)
        if (
            safety is None
            or getattr(safety, "status", None) not in {"safe", "modified"}
            or getattr(safety, "safe_action", None) is None
        ):
            raise ValueError("CubeLift dispatch lacks an accepted shared safety result.")
        request = getattr(execution, "target_execution_request", None)
        submission = getattr(execution, "target_submission_receipt", None)
        completion = getattr(execution, "target_completion_receipt", None)
        dispatch_result = getattr(execution, "target_dispatch_result", None)
        validation = getattr(execution, "dispatch_validation", None)
        receipts = dict(getattr(execution, "dispatch_receipts", {}) or {})
        if any(
            value is None
            for value in (request, submission, completion, dispatch_result, validation)
        ):
            raise ValueError("CubeLift target/dispatch receipt closure is incomplete.")
        if getattr(request, "control_domain_id", None) != self.control_domain_id:
            raise ValueError("Target request control domain mismatch.")
        if (
            not bool(getattr(submission, "accepted_for_processing", False))
            or getattr(submission, "target_execution_request_id", None)
            != getattr(request, "request_id", None)
            or getattr(submission, "target_execution_request_hash", None)
            != getattr(request, "semantic_content_hash", None)
        ):
            raise ValueError("Target submission does not accept the exact request.")
        if (
            getattr(completion, "status", None) != "resolved"
            or getattr(dispatch_result, "completion_receipt", None) != completion
        ):
            raise ValueError("Target completion is not the resolved dispatch receipt.")
        target = getattr(dispatch_result, "target", None)
        admission = getattr(dispatch_result, "target_admission", None)
        if (
            target is None
            or admission is None
            or not bool(getattr(admission, "accepted", False))
            or getattr(completion, "target_id", None)
            != getattr(target, "target_id", None)
            or getattr(completion, "target_hash", None)
            != getattr(target, "semantic_content_hash", None)
            or getattr(admission, "target_id", None)
            != getattr(target, "target_id", None)
            or getattr(admission, "target_hash", None)
            != getattr(target, "semantic_content_hash", None)
        ):
            raise ValueError("Resolved target and target admission do not match.")
        if (
            not bool(getattr(validation, "accepted", False))
            or not receipts
            or not all(bool(getattr(receipt, "accepted", False)) for receipt in receipts.values())
            or dict(getattr(dispatch_result, "dispatch_receipts", {}) or {}) != receipts
        ):
            raise ValueError("CubeLift backend dispatch receipts are not all accepted.")

        trace_facts = tuple(getattr(execution, "trace_facts", ()))
        authority_refs = _required_trace_refs(
            trace_facts,
            event_types=_AUTHORITY_EVENT_TYPES,
            category="authority",
        )
        safety_refs = _required_trace_refs(
            trace_facts,
            event_types=_SAFETY_EVENT_TYPES,
            category="safety",
        )
        target_request_refs = _required_trace_refs(
            trace_facts,
            event_types=_TARGET_REQUEST_EVENT_TYPES,
            category="target request",
        )
        target_submission_refs = _required_trace_refs(
            trace_facts,
            event_types=_TARGET_SUBMISSION_EVENT_TYPES,
            category="target submission",
        )
        target_completion_refs = _required_trace_refs(
            trace_facts,
            event_types=_TARGET_COMPLETION_EVENT_TYPES,
            category="target completion",
        )
        resolved_target_refs = _required_trace_refs(
            trace_facts,
            event_types=_RESOLVED_TARGET_EVENT_TYPES,
            category="resolved target",
        )
        target_admission_refs = _required_trace_refs(
            trace_facts,
            event_types=_TARGET_ADMISSION_EVENT_TYPES,
            category="target admission",
        )
        dispatch_refs = _required_trace_refs(
            trace_facts,
            event_types=_DISPATCH_EVENT_TYPES,
            category="dispatch",
        )
        dispatch_receipt_ref = _required(
            getattr(execution, "dispatch_receipt_id", None),
            field_name="dispatch_receipt_id",
        )
        if dispatch_receipt_ref not in {
            item.fact_id for item in dispatch_refs if item.kind == "dispatch_receipt"
        }:
            raise ValueError("Dispatch receipt ID is absent from exact trace facts.")

        association = getattr(measurement_snapshot, "latest_association", None)
        observer_count = _nonnegative_integer(
            getattr(measurement_snapshot, "measurement_count", None),
            field_name="measurement_snapshot.measurement_count",
        )
        unmatched_count = _nonnegative_integer(
            getattr(measurement_snapshot, "unmatched_measurement_count", None),
            field_name="measurement_snapshot.unmatched_measurement_count",
        )
        if (
            association is None
            or getattr(association, "status", None) != "available"
            or getattr(association, "control_domain_id", None)
            != self.control_domain_id
            or observer_count <= self._last_observer_measurement_count
            or unmatched_count != 0
        ):
            raise ValueError("CubeLift dispatch lacks a fresh matched measurement.")
        if getattr(association, "execution_fence_sequence", None) != getattr(
            target,
            "execution_fence_sequence",
            None,
        ):
            raise ValueError("Measurement and dispatched target fence sequences differ.")
        admission_id = _required(
            getattr(admission, "receipt_id", None),
            field_name="target admission receipt_id",
        )
        admission_hash = _required(
            getattr(admission, "semantic_content_hash", None),
            field_name="target admission semantic_content_hash",
        )
        association_receipts = tuple(
            getattr(association, "receipt_references", ())
        )
        if not any(
            getattr(reference, "receipt_id", None) == admission_id
            and getattr(reference, "receipt_hash", None) == admission_hash
            for reference in association_receipts
        ):
            raise ValueError("Measurement lineage omits the target admission receipt.")
        association_hash = _required(
            getattr(association, "semantic_content_hash", None),
            field_name="measurement association semantic_content_hash",
        )
        if association_hash in self._measurement_hashes:
            raise ValueError("CubeLift measurement association was reused.")
        measurement_ref = CubeLiftExecutionFactRef(
            kind="measurement_reference_association",
            fact_id=_required(
                getattr(association, "association_id", None),
                field_name="measurement association_id",
            ),
            fact_hash=association_hash,
        )
        source_refs = (
            _ref(kind="policy_submission", value=policy, id_field="source_emission_id"),
            _ref(
                kind="policy_input_admission",
                value=input_receipt,
                id_field="composition_input_id",
            ),
            _ref(
                kind="composition_step",
                value=composition,
                id_field="decision_key",
            ),
            _ref(
                kind="composition_disposition",
                value=disposition,
                id_field="disposition_id",
            ),
            _ref(
                kind="action_proposal",
                value=proposal,
                id_field="proposal_id",
            ),
            _ref(
                kind="proposal_admission",
                value=proposal_admission,
                id_field="proposal_id",
            ),
        )
        self._cycles.append(
            CubeLiftExecutionCycleEvidence(
                cycle_index=cycle_index,
                cycle_receipt_ref=f"mujoco-sim-hil-cycle:{cycle_hash}",
                dispatch_receipt_ref=dispatch_receipt_ref,
                source_refs=source_refs,
                authority_refs=authority_refs,
                safety_refs=safety_refs,
                target_request_refs=target_request_refs,
                target_submission_refs=target_submission_refs,
                target_completion_refs=target_completion_refs,
                resolved_target_refs=resolved_target_refs,
                target_admission_refs=target_admission_refs,
                dispatch_refs=dispatch_refs,
                measurement_ref=measurement_ref,
            )
        )
        self._measurement_hashes.add(association_hash)
        self._last_observer_measurement_count = observer_count

    def seal_for_hold(
        self,
        lower_hold_proof: CubeLiftLowerHoldProof,
    ) -> CubeLiftExecutionEvidenceCut:
        if self._sealed:
            raise RuntimeError("CubeLift execution evidence is already sealed.")
        self._sealed = True
        if not isinstance(lower_hold_proof, CubeLiftLowerHoldProof):
            raise TypeError("lower_hold_proof must be a CubeLiftLowerHoldProof.")
        if (
            lower_hold_proof.runtime_session_id != self.runtime_session_id
            or lower_hold_proof.control_domain_id != self.control_domain_id
        ):
            raise ValueError("CubeLift terminal HOLD identity mismatch.")
        cycles = tuple(self._cycles)
        if not cycles:
            raise ValueError("CubeLift execution requires at least one dispatched cycle.")
        source_refs = self._extend(cycles, "source_refs")
        authority_refs = self._extend(cycles, "authority_refs")
        safety_refs = self._extend(cycles, "safety_refs")
        target_request_refs = self._extend(cycles, "target_request_refs")
        target_submission_refs = self._extend(cycles, "target_submission_refs")
        target_completion_refs = self._extend(cycles, "target_completion_refs")
        resolved_target_refs = self._extend(cycles, "resolved_target_refs")
        target_admission_refs = self._extend(cycles, "target_admission_refs")
        dispatch_refs = self._extend(cycles, "dispatch_refs")
        measurement_refs = tuple(cycle.measurement_ref for cycle in cycles)
        category_hashes = {
            "source_proposal_chain_hash": _chain_hash("source", source_refs),
            "authority_decision_chain_hash": _chain_hash(
                "authority",
                authority_refs,
            ),
            "safety_decision_chain_hash": _chain_hash("safety", safety_refs),
            "target_request_chain_hash": _chain_hash(
                "target_request",
                target_request_refs,
            ),
            "target_submission_receipt_chain_hash": _chain_hash(
                "target_submission",
                target_submission_refs,
            ),
            "target_completion_receipt_chain_hash": _chain_hash(
                "target_completion",
                target_completion_refs,
            ),
            "resolved_component_target_chain_hash": _chain_hash(
                "resolved_target",
                resolved_target_refs,
            ),
            "component_admission_receipt_chain_hash": _chain_hash(
                "target_admission",
                target_admission_refs,
            ),
            "dispatch_receipt_chain_hash": _chain_hash(
                "dispatch",
                dispatch_refs,
            ),
            "measurement_lineage_chain_hash": _chain_hash(
                "measurement",
                measurement_refs,
            ),
        }
        terminal_state_hash = canonical_content_hash(
            {
                "runtime_session_id": self.runtime_session_id,
                "control_domain_id": self.control_domain_id,
                "last_cycle_receipt_ref": cycles[-1].cycle_receipt_ref,
                "last_measurement_ref": measurement_refs[-1].to_wire(),
                "lower_hold_proof_hash": lower_hold_proof.semantic_content_hash,
                "lower_gate_snapshot_hash": lower_hold_proof.gate_snapshot_hash,
            },
            domain=CUBE_LIFT_TERMINAL_CONTROL_STATE_HASH_DOMAIN_V1,
        )
        evidence_hash = canonical_content_hash(
            {
                "runtime_session_id": self.runtime_session_id,
                "control_domain_id": self.control_domain_id,
                "cycle_receipt_refs": tuple(
                    cycle.cycle_receipt_ref for cycle in cycles
                ),
                "dispatch_receipt_refs": tuple(
                    cycle.dispatch_receipt_ref for cycle in cycles
                ),
                **category_hashes,
                "terminal_hold_proof_hash": lower_hold_proof.semantic_content_hash,
                "terminal_state_hash": terminal_state_hash,
            },
            domain=CUBE_LIFT_EXECUTION_EVIDENCE_HASH_DOMAIN_V1,
        )
        return CubeLiftExecutionEvidenceCut(
            runtime_session_id=self.runtime_session_id,
            control_domain_id=self.control_domain_id,
            execution_evidence_chain_ref=(
                f"cube-lift-execution-evidence:{evidence_hash}"
            ),
            first_cycle_receipt_ref=cycles[0].dispatch_receipt_ref,
            last_cycle_receipt_ref=cycles[-1].dispatch_receipt_ref,
            terminal_state_hash=(
                f"cube-lift-terminal-control-state:{terminal_state_hash}"
            ),
            cycle_count=len(cycles),
            dispatched_cycle_count=len(cycles),
            measurement_count=len(measurement_refs),
            **category_hashes,
        )


__all__ = [
    "CUBE_LIFT_EXECUTION_EVIDENCE_HASH_DOMAIN_V1",
    "CUBE_LIFT_EXECUTION_FACT_CHAIN_HASH_DOMAIN_V1",
    "CUBE_LIFT_TERMINAL_CONTROL_STATE_HASH_DOMAIN_V1",
    "CubeLiftExecutionCycleEvidence",
    "CubeLiftExecutionEvidenceCollector",
    "CubeLiftExecutionEvidenceCut",
    "CubeLiftExecutionFactRef",
]
