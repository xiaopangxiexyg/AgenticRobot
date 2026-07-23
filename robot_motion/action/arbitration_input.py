"""Complete immutable input consumed by the pure ActionArbiter."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.runtime.control_admission import ControlAdmissionCoordinator
from robot_motion.runtime.execution_gate import ExecutionGateSnapshot, ExecutionGateStore
from robot_motion.runtime.producer_epoch import ProducerEpochRegistry, ProducerEpochSnapshot

from .arbitration_runtime import DecisionScheduleTick, DecisionScheduler
from .authority import AuthorityPolicySpec, AuthorityStateSnapshot, AuthorityStore
from .candidate_authorization import (
    CandidateAuthorizationSnapshot,
    CandidateAuthorizationStore,
    build_candidate_authorization_snapshot,
    build_empty_candidate_authorization_snapshot,
    project_candidate_authorizations,
)
from .components import ActionComponentCatalog
from .proposal import ProposalBuffer
from .snapshot import ProposalSnapshot, build_proposal_snapshot, project_proposals


ARBITRATION_INPUT_HASH_DOMAIN_V1 = "dexteleop.arbitration_input_snapshot.v1"


@dataclass(frozen=True)
class ArbitrationInputSnapshot:
    arbitration_input_snapshot_id: str
    runtime_session_id: str
    control_domain_id: str
    embodiment_instance_id: str
    proposal_snapshot: ProposalSnapshot
    authority_policy: AuthorityPolicySpec
    authority_state_snapshot: AuthorityStateSnapshot
    execution_gate_snapshot: ExecutionGateSnapshot
    producer_epoch_snapshot: ProducerEpochSnapshot
    candidate_authorization_snapshot: CandidateAuthorizationSnapshot
    previous_authoritative_decision_id: str | None
    previous_selection_by_group: Mapping[str, str]
    active_selection_groups: tuple[str, ...]
    component_catalog: ActionComponentCatalog
    decision_schedule_tick: DecisionScheduleTick
    semantic_content_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "previous_selection_by_group",
            MappingProxyType(dict(self.previous_selection_by_group)),
        )
        object.__setattr__(self, "active_selection_groups", tuple(self.active_selection_groups))


def build_arbitration_input_snapshot(
    *,
    runtime_session_id: str,
    embodiment_instance_id: str,
    proposal_snapshot: ProposalSnapshot,
    authority_state_snapshot: AuthorityStateSnapshot,
    authority_policy: AuthorityPolicySpec,
    execution_gate_snapshot: ExecutionGateSnapshot,
    producer_epoch_snapshot: ProducerEpochSnapshot,
    previous_authoritative_decision_id: str | None,
    previous_selection_by_group: Mapping[str, str],
    active_selection_groups: tuple[str, ...],
    component_catalog: ActionComponentCatalog,
    decision_schedule_tick: DecisionScheduleTick,
    candidate_authorization_snapshot: CandidateAuthorizationSnapshot | None = None,
) -> ArbitrationInputSnapshot:
    domain = proposal_snapshot.control_domain_id
    if any(
        value != domain
        for value in (
            authority_state_snapshot.control_domain_id,
            execution_gate_snapshot.control_domain_id,
            producer_epoch_snapshot.control_domain_id,
        )
    ):
        raise ValueError("Arbitration child snapshots belong to different domains.")
    cutoff_sequence = proposal_snapshot.cutoff_admission_sequence
    if any(
        value != cutoff_sequence
        for value in (
            authority_state_snapshot.cutoff_admission_sequence,
            execution_gate_snapshot.cutoff_admission_sequence,
            producer_epoch_snapshot.cutoff_admission_sequence,
            decision_schedule_tick.cutoff_admission_sequence,
        )
    ):
        raise ValueError("Arbitration child snapshots do not share one committed cut.")
    if proposal_snapshot.component_schema_hash != component_catalog.semantic_content_hash:
        raise ValueError("Proposal snapshot and component catalog hashes disagree.")
    if authority_state_snapshot.authority_policy_hash != authority_policy.semantic_content_hash:
        raise ValueError("Authority snapshot and policy hashes disagree.")
    authorization_snapshot = (
        build_empty_candidate_authorization_snapshot(proposal_snapshot)
        if candidate_authorization_snapshot is None
        else candidate_authorization_snapshot
    )
    if (
        authorization_snapshot.control_domain_id != domain
        or authorization_snapshot.cutoff_admission_sequence != cutoff_sequence
    ):
        raise ValueError("Candidate authorization snapshot does not share the arbitration cut.")
    active_groups = tuple(dict.fromkeys(str(value) for value in active_selection_groups))
    if not set(active_groups).issubset(component_catalog.selection_group_ids):
        raise ValueError("Unknown active selection group.")
    payload = {
        "runtime_session_id": runtime_session_id,
        "control_domain_id": domain,
        "embodiment_instance_id": embodiment_instance_id,
        "proposal_snapshot_id": proposal_snapshot.proposal_snapshot_id,
        "proposal_snapshot_hash": proposal_snapshot.semantic_content_hash,
        "authority_state_snapshot_hash": authority_state_snapshot.semantic_content_hash,
        "execution_gate_snapshot_hash": execution_gate_snapshot.semantic_content_hash,
        "producer_epoch_snapshot": {
            "cutoff_admission_sequence": producer_epoch_snapshot.cutoff_admission_sequence,
            "active_epoch_by_producer": producer_epoch_snapshot.active_epoch_by_producer,
            "activation_id_by_producer": producer_epoch_snapshot.activation_id_by_producer,
        },
        "candidate_authorization_snapshot_id": authorization_snapshot.snapshot_id,
        "candidate_authorization_snapshot_hash": authorization_snapshot.semantic_content_hash,
        "previous_authoritative_decision_id": previous_authoritative_decision_id,
        "previous_selection_by_group": dict(previous_selection_by_group),
        "active_selection_groups": active_groups,
        "component_catalog_hash": component_catalog.semantic_content_hash,
        "decision_schedule_tick_id": decision_schedule_tick.schedule_tick_id,
        "decision_schedule_tick_hash": decision_schedule_tick.semantic_content_hash,
    }
    semantic_hash = canonical_content_hash(payload, domain=ARBITRATION_INPUT_HASH_DOMAIN_V1)
    return ArbitrationInputSnapshot(
        arbitration_input_snapshot_id=f"arbitration-input:{semantic_hash}",
        runtime_session_id=runtime_session_id,
        control_domain_id=domain,
        embodiment_instance_id=embodiment_instance_id,
        proposal_snapshot=proposal_snapshot,
        authority_policy=authority_policy,
        authority_state_snapshot=authority_state_snapshot,
        execution_gate_snapshot=execution_gate_snapshot,
        producer_epoch_snapshot=producer_epoch_snapshot,
        candidate_authorization_snapshot=authorization_snapshot,
        previous_authoritative_decision_id=previous_authoritative_decision_id,
        previous_selection_by_group=dict(previous_selection_by_group),
        active_selection_groups=active_groups,
        component_catalog=component_catalog,
        decision_schedule_tick=decision_schedule_tick,
        semantic_content_hash=semantic_hash,
    )


def capture_arbitration_input(
    *,
    coordinator: ControlAdmissionCoordinator,
    proposal_buffer: ProposalBuffer,
    producer_epochs: ProducerEpochRegistry,
    authority_store: AuthorityStore,
    execution_gate_store: ExecutionGateStore,
    scheduler: DecisionScheduler,
    embodiment_instance_id: str,
    component_catalog: ActionComponentCatalog,
    active_selection_groups: tuple[str, ...],
    previous_authoritative_decision_id: str | None = None,
    previous_selection_by_group: Mapping[str, str] | None = None,
    candidate_authorization_store: CandidateAuthorizationStore | None = None,
    target_runtime_contract_hash: str | None = None,
    required_authorization_producer_kinds: tuple[str, ...] = (),
    now: float | None = None,
) -> ArbitrationInputSnapshot | None:
    """Capture every mutable store under one coordinator read barrier."""

    timestamp = float(scheduler.now() if now is None else now)
    if not scheduler.due(timestamp):
        return None
    cutoff_at = timestamp - scheduler.spec.cutoff_lag_sec
    projectors = {
        "proposals": lambda cut: project_proposals(proposal_buffer, cut),
        "producer_epochs": producer_epochs.snapshot,
        "authority_records": authority_store.records_at_or_before,
        "gate_records": execution_gate_store.records_at_or_before,
    }
    if candidate_authorization_store is not None:
        if candidate_authorization_store.coordinator is not coordinator:
            raise ValueError("Candidate authorization store must share the coordinator.")
        projectors["candidate_authorizations"] = lambda cut: project_candidate_authorizations(
            candidate_authorization_store,
            cut,
        )
    elif required_authorization_producer_kinds:
        raise ValueError("Required candidate authorization needs an authorization store.")
    capture = coordinator.capture_cut(
        cutoff_at=cutoff_at,
        projectors=projectors,
    )
    tick = scheduler.make_tick(
        cutoff_admission_sequence=capture.cut.cutoff_admission_sequence,
        now=timestamp,
    )
    assert tick is not None
    proposal_snapshot = build_proposal_snapshot(
        capture.projections["proposals"],
        producer_epochs=capture.projections["producer_epochs"],
        target_effective_at=tick.target_effective_at,
        component_schema_hash=component_catalog.semantic_content_hash,
        active_selection_groups=active_selection_groups,
    )
    authority_snapshot = authority_store.build_snapshot(
        cut=capture.cut,
        records=capture.projections["authority_records"],
    )
    gate_snapshot = execution_gate_store.build_snapshot(
        cut=capture.cut,
        records=capture.projections["gate_records"],
    )
    candidate_authorization_snapshot = (
        build_empty_candidate_authorization_snapshot(proposal_snapshot)
        if candidate_authorization_store is None
        else build_candidate_authorization_snapshot(
            capture.projections["candidate_authorizations"],
            proposal_snapshot=proposal_snapshot,
            target_runtime_contract_hash=target_runtime_contract_hash,
            required_producer_kinds=required_authorization_producer_kinds,
        )
    )
    return build_arbitration_input_snapshot(
        runtime_session_id=coordinator.clock_domain.runtime_session_id,
        embodiment_instance_id=embodiment_instance_id,
        proposal_snapshot=proposal_snapshot,
        authority_state_snapshot=authority_snapshot,
        authority_policy=authority_store.policy,
        execution_gate_snapshot=gate_snapshot,
        producer_epoch_snapshot=capture.projections["producer_epochs"],
        previous_authoritative_decision_id=previous_authoritative_decision_id,
        previous_selection_by_group=dict(previous_selection_by_group or {}),
        active_selection_groups=active_selection_groups,
        component_catalog=component_catalog,
        decision_schedule_tick=tick,
        candidate_authorization_snapshot=candidate_authorization_snapshot,
    )


__all__ = [
    "ArbitrationInputSnapshot",
    "build_arbitration_input_snapshot",
    "capture_arbitration_input",
]
