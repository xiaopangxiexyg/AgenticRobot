"""Receiver-admitted, target-specific publication authorization evidence.

An action proposal is a source fact.  Whether that fact may be published by a
particular target runtime is a separate relation.  This module linearizes that
relation on the receiver control clock and freezes it into each arbitration
input without changing pure Arbiter selection or passive-shadow semantics.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.runtime.control_admission import (
    ControlAdmissionCoordinator,
    ControlCut,
    admitted_at_or_before,
)

from .proposal import ActionProposal, ProposalBuffer
from .snapshot import ProposalSnapshot


CANDIDATE_AUTHORIZATION_HASH_DOMAIN_V1 = (
    "dexteleop.candidate_authorization.semantic.v1"
)
CANDIDATE_AUTHORIZATION_SNAPSHOT_HASH_DOMAIN_V1 = (
    "dexteleop.candidate_authorization_snapshot.v1"
)


class CandidateAuthorizationError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = str(code)


@dataclass(frozen=True)
class CandidateAuthorizationDraft:
    proposal_id: str
    proposal_content_hash: str
    producer_kind: str
    evidence_id: str
    evidence_hash: str
    binding_eligibility_id: str
    binding_eligibility_hash: str
    target_runtime_contract_hash: str
    authoritative_allowed: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateAuthorizationRecord:
    authorization_id: str
    runtime_session_id: str
    control_domain_id: str
    proposal_id: str
    proposal_content_hash: str
    producer_kind: str
    evidence_id: str
    evidence_hash: str
    binding_eligibility_id: str
    binding_eligibility_hash: str
    target_runtime_contract_hash: str
    authoritative_allowed: bool
    reasons: tuple[str, ...]
    available_at: float
    available_clock_domain_id: str
    control_admission_sequence: int
    semantic_content_hash: str


@dataclass(frozen=True)
class CandidateAuthorizationAdmissionReceipt:
    accepted: bool
    rejection_code: str | None = None
    message: str = ""
    authorization_id: str | None = None
    available_at: float | None = None
    available_clock_domain_id: str | None = None
    control_admission_sequence: int | None = None
    semantic_content_hash: str | None = None
    record: CandidateAuthorizationRecord | None = None


@dataclass(frozen=True)
class CandidateAuthorizationProjection:
    control_domain_id: str
    cutoff_at: float
    cutoff_admission_sequence: int
    records: tuple[CandidateAuthorizationRecord, ...]


@dataclass(frozen=True)
class CandidateAuthorizationSnapshot:
    snapshot_id: str
    control_domain_id: str
    cutoff_at: float
    cutoff_admission_sequence: int
    target_runtime_contract_hash: str | None
    required_producer_kinds: tuple[str, ...]
    authorization_by_proposal_id: Mapping[str, CandidateAuthorizationRecord]
    dispositions_by_proposal_id: Mapping[str, str]
    semantic_content_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "authorization_by_proposal_id",
            MappingProxyType(dict(self.authorization_by_proposal_id)),
        )
        object.__setattr__(
            self,
            "dispositions_by_proposal_id",
            MappingProxyType(dict(self.dispositions_by_proposal_id)),
        )
        object.__setattr__(
            self,
            "required_producer_kinds",
            tuple(self.required_producer_kinds),
        )


def _required(value: str, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise CandidateAuthorizationError("invalid_field", f"{name} must be non-empty.")
    return normalized


class CandidateAuthorizationStore:
    """Bounded target relation store sharing the control admission coordinator."""

    def __init__(
        self,
        *,
        runtime_session_id: str,
        coordinator: ControlAdmissionCoordinator,
        proposals: ProposalBuffer,
        capacity: int = 4096,
    ) -> None:
        if proposals.coordinator is not coordinator:
            raise ValueError("Authorization and proposal stores must share one coordinator.")
        if int(capacity) <= 0:
            raise ValueError("Candidate authorization capacity must be positive.")
        self.runtime_session_id = _required(runtime_session_id, name="runtime_session_id")
        if self.runtime_session_id != coordinator.clock_domain.runtime_session_id:
            raise ValueError("Candidate authorization session/clock mismatch.")
        self.coordinator = coordinator
        self.control_domain_id = coordinator.control_domain_id
        self.proposals = proposals
        self.capacity = int(capacity)
        self._records: list[CandidateAuthorizationRecord] = []
        self._by_proposal_id: dict[str, CandidateAuthorizationRecord] = {}
        self._lock = threading.RLock()

    def admit(
        self,
        draft: CandidateAuthorizationDraft,
    ) -> CandidateAuthorizationAdmissionReceipt:
        try:
            proposal_id = _required(draft.proposal_id, name="proposal_id")
            proposal_hash = _required(
                draft.proposal_content_hash,
                name="proposal_content_hash",
            )
            producer_kind = _required(draft.producer_kind, name="producer_kind")
            evidence_id = _required(draft.evidence_id, name="evidence_id")
            evidence_hash = _required(draft.evidence_hash, name="evidence_hash")
            eligibility_id = _required(
                draft.binding_eligibility_id,
                name="binding_eligibility_id",
            )
            eligibility_hash = _required(
                draft.binding_eligibility_hash,
                name="binding_eligibility_hash",
            )
            target_hash = _required(
                draft.target_runtime_contract_hash,
                name="target_runtime_contract_hash",
            )
            reasons = tuple(str(value) for value in draft.reasons)
            proposal = self.proposals.get(proposal_id)
            if proposal is None:
                raise CandidateAuthorizationError(
                    "proposal_not_admitted",
                    "Candidate authorization must reference an admitted proposal.",
                )
            if proposal.semantic_content_hash != proposal_hash:
                raise CandidateAuthorizationError(
                    "proposal_hash_mismatch",
                    "Candidate authorization proposal hash does not match the admitted fact.",
                )
            if proposal.producer_kind != producer_kind:
                raise CandidateAuthorizationError(
                    "producer_kind_mismatch",
                    "Candidate authorization producer kind does not match the proposal.",
                )

            def build(available_at: float, admission_sequence: int) -> CandidateAuthorizationRecord:
                payload = {
                    "runtime_session_id": self.runtime_session_id,
                    "control_domain_id": self.control_domain_id,
                    "proposal_id": proposal_id,
                    "proposal_content_hash": proposal_hash,
                    "producer_kind": producer_kind,
                    "evidence_id": evidence_id,
                    "evidence_hash": evidence_hash,
                    "binding_eligibility_id": eligibility_id,
                    "binding_eligibility_hash": eligibility_hash,
                    "target_runtime_contract_hash": target_hash,
                    "authoritative_allowed": bool(draft.authoritative_allowed),
                    "reasons": reasons,
                    "available_at": available_at,
                    "available_clock_domain_id": self.coordinator.clock_domain.clock_domain_id,
                    "control_admission_sequence": admission_sequence,
                }
                semantic_hash = canonical_content_hash(
                    payload,
                    domain=CANDIDATE_AUTHORIZATION_HASH_DOMAIN_V1,
                )
                return CandidateAuthorizationRecord(
                    authorization_id=f"candidate-authorization:{semantic_hash}",
                    **payload,
                    semantic_content_hash=semantic_hash,
                )

            record = self.coordinator.commit(store=self, build=build)
            return CandidateAuthorizationAdmissionReceipt(
                accepted=True,
                authorization_id=record.authorization_id,
                available_at=record.available_at,
                available_clock_domain_id=record.available_clock_domain_id,
                control_admission_sequence=record.control_admission_sequence,
                semantic_content_hash=record.semantic_content_hash,
                record=record,
            )
        except (CandidateAuthorizationError, TypeError, ValueError) as exc:
            return CandidateAuthorizationAdmissionReceipt(
                accepted=False,
                rejection_code=(
                    exc.code if isinstance(exc, CandidateAuthorizationError) else "validation_failed"
                ),
                message=str(exc),
            )

    def _insert_admitted(self, record: CandidateAuthorizationRecord) -> None:
        with self._lock:
            if record.proposal_id in self._by_proposal_id:
                raise CandidateAuthorizationError(
                    "duplicate",
                    "Candidate authorization already exists for this proposal.",
                )
            if len(self._records) >= self.capacity:
                raise CandidateAuthorizationError(
                    "capacity_exhausted",
                    "Candidate authorization store is full.",
                )
            self._records.append(record)
            self._by_proposal_id[record.proposal_id] = record

    def records_at_or_before(
        self,
        cut: ControlCut,
    ) -> tuple[CandidateAuthorizationRecord, ...]:
        if cut.control_domain_id != self.control_domain_id:
            raise ValueError("Control cut belongs to another domain.")
        with self._lock:
            return tuple(
                record for record in self._records if admitted_at_or_before(record, cut)
            )


def project_candidate_authorizations(
    store: CandidateAuthorizationStore,
    cut: ControlCut,
) -> CandidateAuthorizationProjection:
    return CandidateAuthorizationProjection(
        control_domain_id=cut.control_domain_id,
        cutoff_at=cut.cutoff_at,
        cutoff_admission_sequence=cut.cutoff_admission_sequence,
        records=store.records_at_or_before(cut),
    )


def build_candidate_authorization_snapshot(
    projection: CandidateAuthorizationProjection,
    *,
    proposal_snapshot: ProposalSnapshot,
    target_runtime_contract_hash: str | None,
    required_producer_kinds: tuple[str, ...],
) -> CandidateAuthorizationSnapshot:
    if projection.control_domain_id != proposal_snapshot.control_domain_id:
        raise ValueError("Authorization and proposal snapshots belong to different domains.")
    if projection.cutoff_admission_sequence != proposal_snapshot.cutoff_admission_sequence:
        raise ValueError("Authorization and proposal snapshots do not share one cut.")
    required = tuple(dict.fromkeys(str(value).strip() for value in required_producer_kinds))
    if any(not value for value in required):
        raise ValueError("Required producer kinds must be non-empty strings.")
    target_hash = (
        None
        if target_runtime_contract_hash is None
        else _required(target_runtime_contract_hash, name="target_runtime_contract_hash")
    )
    candidate_by_id = {
        item.proposal.proposal_id: item.proposal
        for item in proposal_snapshot.candidates
    }
    records_by_id = {record.proposal_id: record for record in projection.records}
    selected: dict[str, CandidateAuthorizationRecord] = {}
    dispositions: dict[str, str] = {}
    for proposal_id, proposal in sorted(candidate_by_id.items()):
        if proposal.producer_kind not in required:
            dispositions[proposal_id] = "authorization_not_required"
            continue
        record = records_by_id.get(proposal_id)
        if record is None:
            dispositions[proposal_id] = "authorization_missing"
            continue
        if record.proposal_content_hash != proposal.semantic_content_hash:
            dispositions[proposal_id] = "proposal_hash_mismatch"
            continue
        if target_hash is None or record.target_runtime_contract_hash != target_hash:
            dispositions[proposal_id] = "target_runtime_contract_mismatch"
            continue
        selected[proposal_id] = record
        dispositions[proposal_id] = (
            "authoritative_allowed"
            if record.authoritative_allowed
            else "authoritative_denied"
        )
    payload = {
        "control_domain_id": projection.control_domain_id,
        "cutoff_at": projection.cutoff_at,
        "cutoff_admission_sequence": projection.cutoff_admission_sequence,
        "target_runtime_contract_hash": target_hash,
        "required_producer_kinds": required,
        "authorization_by_proposal_id": {
            proposal_id: {
                "authorization_id": record.authorization_id,
                "authorization_hash": record.semantic_content_hash,
                "proposal_content_hash": record.proposal_content_hash,
                "evidence_id": record.evidence_id,
                "evidence_hash": record.evidence_hash,
                "binding_eligibility_id": record.binding_eligibility_id,
                "binding_eligibility_hash": record.binding_eligibility_hash,
                "authoritative_allowed": record.authoritative_allowed,
            }
            for proposal_id, record in sorted(selected.items())
        },
        "dispositions_by_proposal_id": dict(sorted(dispositions.items())),
    }
    semantic_hash = canonical_content_hash(
        payload,
        domain=CANDIDATE_AUTHORIZATION_SNAPSHOT_HASH_DOMAIN_V1,
    )
    return CandidateAuthorizationSnapshot(
        snapshot_id=f"candidate-authorization-snapshot:{semantic_hash}",
        **{
            key: value
            for key, value in payload.items()
            if key != "authorization_by_proposal_id"
        },
        authorization_by_proposal_id=selected,
        semantic_content_hash=semantic_hash,
    )


def build_empty_candidate_authorization_snapshot(
    proposal_snapshot: ProposalSnapshot,
) -> CandidateAuthorizationSnapshot:
    projection = CandidateAuthorizationProjection(
        control_domain_id=proposal_snapshot.control_domain_id,
        cutoff_at=proposal_snapshot.cutoff_at,
        cutoff_admission_sequence=proposal_snapshot.cutoff_admission_sequence,
        records=(),
    )
    return build_candidate_authorization_snapshot(
        projection,
        proposal_snapshot=proposal_snapshot,
        target_runtime_contract_hash=None,
        required_producer_kinds=(),
    )


__all__ = [
    "CandidateAuthorizationAdmissionReceipt",
    "CandidateAuthorizationDraft",
    "CandidateAuthorizationError",
    "CandidateAuthorizationProjection",
    "CandidateAuthorizationRecord",
    "CandidateAuthorizationSnapshot",
    "CandidateAuthorizationStore",
    "build_candidate_authorization_snapshot",
    "build_empty_candidate_authorization_snapshot",
    "project_candidate_authorizations",
]
