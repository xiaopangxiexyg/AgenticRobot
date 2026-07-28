"""Target-time-aware immutable proposal snapshots."""

from __future__ import annotations

import math
from dataclasses import dataclass

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.runtime.control_admission import ControlCut
from robot_motion.runtime.producer_epoch import ProducerEpochSnapshot

from .proposal import ActionProposal, ProposalBuffer


PROPOSAL_SNAPSHOT_HASH_DOMAIN_V1 = "dexteleop.proposal_snapshot.v1"
ACTION_EMISSION_SCHEDULE_HASH_DOMAIN_V1 = "dexteleop.action_emission_schedule.v1"


@dataclass(frozen=True)
class ActionEmissionScheduleSpec:
    spec_id: str
    start_at: float
    keyframe_period_sec: float
    keyframe_count: int
    horizon_sec: float
    maximum_hold_sec: float

    def __post_init__(self) -> None:
        if not str(self.spec_id).strip():
            raise ValueError("spec_id must be non-empty.")
        numeric = (
            float(self.start_at),
            float(self.keyframe_period_sec),
            float(self.horizon_sec),
            float(self.maximum_hold_sec),
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("Action emission schedule values must be finite.")
        if self.keyframe_period_sec <= 0 or int(self.keyframe_count) <= 0:
            raise ValueError("Action emission cadence/count must be positive.")
        if self.horizon_sec <= 0 or self.maximum_hold_sec <= 0:
            raise ValueError("Action emission horizon/maximum hold must be positive.")
        if self.horizon_sec <= self.keyframe_period_sec * (self.keyframe_count - 1):
            raise ValueError("horizon_sec must extend beyond the final keyframe start.")

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "spec_id": self.spec_id,
                "start_at": self.start_at,
                "keyframe_period_sec": self.keyframe_period_sec,
                "keyframe_count": self.keyframe_count,
                "horizon_sec": self.horizon_sec,
                "maximum_hold_sec": self.maximum_hold_sec,
            },
            domain=ACTION_EMISSION_SCHEDULE_HASH_DOMAIN_V1,
        )

    def interval_for(self, keyframe_index: int) -> tuple[float, float]:
        index = int(keyframe_index)
        if index < 0 or index >= self.keyframe_count:
            raise IndexError("keyframe_index is outside the schedule.")
        effective_from = self.start_at + index * self.keyframe_period_sec
        horizon_end = self.start_at + self.horizon_sec
        if index + 1 < self.keyframe_count:
            effective_until = min(
                self.start_at + (index + 1) * self.keyframe_period_sec,
                horizon_end,
            )
        else:
            effective_until = min(
                horizon_end,
                effective_from + self.maximum_hold_sec,
            )
        if effective_until <= effective_from:
            raise ValueError("Schedule produces an empty final effective interval.")
        return effective_from, effective_until


@dataclass(frozen=True)
class ProposalProjection:
    control_domain_id: str
    cutoff_at: float
    cutoff_admission_sequence: int
    included_admission_sequence_min: int | None
    included_admission_sequence_max: int | None
    proposals: tuple[ActionProposal, ...]


@dataclass(frozen=True)
class ProposalCandidate:
    selection_group_id: str
    proposal: ActionProposal


@dataclass(frozen=True)
class ProposalDisposition:
    proposal_id: str
    selection_group_id: str | None
    disposition: str
    detail: str = ""


@dataclass(frozen=True)
class ProposalSnapshot:
    proposal_snapshot_id: str
    control_domain_id: str
    cutoff_at: float
    cutoff_admission_sequence: int
    included_admission_sequence_min: int | None
    included_admission_sequence_max: int | None
    target_effective_at: float
    action_schema_hash: str
    producer_epoch_snapshot_hash: str
    reduction_policy_id: str
    candidates: tuple[ProposalCandidate, ...]
    dispositions: tuple[ProposalDisposition, ...]
    semantic_content_hash: str


def project_proposals(buffer: ProposalBuffer, cut: ControlCut) -> ProposalProjection:
    """Copy a bounded in-memory projection; performs no hashing."""

    records = buffer.records_at_or_before(cut)
    sequences = tuple(item.control_admission_sequence for item in records)
    return ProposalProjection(
        control_domain_id=cut.control_domain_id,
        cutoff_at=cut.cutoff_at,
        cutoff_admission_sequence=cut.cutoff_admission_sequence,
        included_admission_sequence_min=min(sequences) if sequences else None,
        included_admission_sequence_max=max(sequences) if sequences else None,
        proposals=records,
    )


def _producer_epoch_snapshot_hash(snapshot: ProducerEpochSnapshot) -> str:
    return canonical_content_hash(
        {
            "control_domain_id": snapshot.control_domain_id,
            "cutoff_at": snapshot.cutoff_at,
            "cutoff_admission_sequence": snapshot.cutoff_admission_sequence,
            "included_admission_sequence_min": snapshot.included_admission_sequence_min,
            "included_admission_sequence_max": snapshot.included_admission_sequence_max,
            "active_epoch_by_producer": snapshot.active_epoch_by_producer,
            "activation_id_by_producer": snapshot.activation_id_by_producer,
        },
        domain="dexteleop.producer_epoch_snapshot.v1",
    )


def build_proposal_snapshot(
    projection: ProposalProjection,
    *,
    producer_epochs: ProducerEpochSnapshot,
    target_effective_at: float,
    action_schema_hash: str,
    active_selection_groups: tuple[str, ...],
    reduction_policy_id: str = "target_interval_latest_per_producer_group_v1",
) -> ProposalSnapshot:
    target = float(target_effective_at)
    if not math.isfinite(target):
        raise ValueError("target_effective_at must be finite.")
    active_groups = tuple(dict.fromkeys(str(value) for value in active_selection_groups))
    if producer_epochs.control_domain_id != projection.control_domain_id:
        raise ValueError("Proposal and producer-epoch snapshots belong to different domains.")
    if producer_epochs.cutoff_admission_sequence != projection.cutoff_admission_sequence:
        raise ValueError("Proposal and producer-epoch snapshots do not share one cut.")
    active_set = set(active_groups)
    buckets: dict[tuple[str, int, str], list[ActionProposal]] = {}
    dispositions: list[ProposalDisposition] = []

    for proposal in projection.proposals:
        if producer_epochs.active_epoch_by_producer.get(proposal.producer_id) != proposal.producer_epoch:
            dispositions.append(
                ProposalDisposition(proposal.proposal_id, None, "inactive_producer_epoch")
            )
            continue
        if proposal.action_schema_hash != action_schema_hash:
            dispositions.append(
                ProposalDisposition(proposal.proposal_id, None, "action_schema_mismatch")
            )
            continue
        if proposal.validity != "valid" or proposal.status != "ready":
            dispositions.append(
                ProposalDisposition(proposal.proposal_id, None, "invalid_status")
            )
            continue
        if not proposal.effective_from <= target < proposal.effective_until:
            dispositions.append(
                ProposalDisposition(proposal.proposal_id, None, "effective_interval_missed")
            )
            continue
        groups = tuple(group for group in proposal.selection_groups if group in active_set)
        if not groups:
            dispositions.append(
                ProposalDisposition(proposal.proposal_id, None, "inactive_selection_group")
            )
            continue
        for group in groups:
            buckets.setdefault((proposal.producer_id, proposal.producer_epoch, group), []).append(proposal)

    candidates: list[ProposalCandidate] = []
    for key in sorted(buckets):
        group = key[2]
        ordered = sorted(
            buckets[key],
            key=lambda item: (-item.effective_from, -item.sequence, item.proposal_id),
        )
        winner = ordered[0]
        candidates.append(ProposalCandidate(selection_group_id=group, proposal=winner))
        dispositions.extend(
            ProposalDisposition(item.proposal_id, group, "superseded_before_snapshot")
            for item in ordered[1:]
        )

    candidates.sort(key=lambda item: (item.selection_group_id, item.proposal.proposal_id))
    dispositions.sort(
        key=lambda item: (item.proposal_id, item.selection_group_id or "", item.disposition)
    )
    epoch_hash = _producer_epoch_snapshot_hash(producer_epochs)
    payload = {
        "control_domain_id": projection.control_domain_id,
        "cutoff_at": projection.cutoff_at,
        "cutoff_admission_sequence": projection.cutoff_admission_sequence,
        "included_admission_sequence_min": projection.included_admission_sequence_min,
        "included_admission_sequence_max": projection.included_admission_sequence_max,
        "target_effective_at": target,
        "action_schema_hash": action_schema_hash,
        "producer_epoch_snapshot_hash": epoch_hash,
        "reduction_policy_id": reduction_policy_id,
        "candidates": tuple(
            {
                "selection_group_id": item.selection_group_id,
                "proposal_id": item.proposal.proposal_id,
                "proposal_content_hash": item.proposal.semantic_content_hash,
            }
            for item in candidates
        ),
        "dispositions": tuple(
            {
                "proposal_id": item.proposal_id,
                "selection_group_id": item.selection_group_id,
                "disposition": item.disposition,
                "detail": item.detail,
            }
            for item in dispositions
        ),
    }
    semantic_hash = canonical_content_hash(payload, domain=PROPOSAL_SNAPSHOT_HASH_DOMAIN_V1)
    return ProposalSnapshot(
        proposal_snapshot_id=f"proposal-snapshot:{semantic_hash}",
        control_domain_id=projection.control_domain_id,
        cutoff_at=projection.cutoff_at,
        cutoff_admission_sequence=projection.cutoff_admission_sequence,
        included_admission_sequence_min=projection.included_admission_sequence_min,
        included_admission_sequence_max=projection.included_admission_sequence_max,
        target_effective_at=target,
        action_schema_hash=action_schema_hash,
        producer_epoch_snapshot_hash=epoch_hash,
        reduction_policy_id=reduction_policy_id,
        candidates=tuple(candidates),
        dispositions=tuple(dispositions),
        semantic_content_hash=semantic_hash,
    )


__all__ = [
    "ActionEmissionScheduleSpec",
    "ProposalCandidate",
    "ProposalDisposition",
    "ProposalProjection",
    "ProposalSnapshot",
    "build_proposal_snapshot",
    "project_proposals",
]
