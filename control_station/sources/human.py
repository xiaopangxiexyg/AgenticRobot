"""Station takeover-readiness contract and Human data-plane compatibility.

Generic observation, fanout, and receiver admission live in
``teleop_runtime.action``.  This application module retains only the explicit
deadman/rebase readiness candidate used by Station takeover operations, while
re-exporting the established data-plane names during migration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from control_composition.lifecycle import (
    HumanTakeoverOperationSpec,
    HumanTakeoverState,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from teleop_runtime.action.human_admission import (
    HumanObservationCutAdmissionCore,
    HumanProposalAdmissionCore,
    admit_human_command_emission,
    admit_human_observation_cut,
)
from teleop_runtime.action.human_observer import (
    HumanPoseSourceContract,
    PassiveHumanCommandEmissionIngress,
    PassiveHumanCommandEmissionIngressStatus,
    PassiveHumanCommandObserverFanout,
)
from teleop_runtime.action.human_proposal import HumanCommandEmission


HUMAN_TAKEOVER_READINESS_CANDIDATE_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.human_takeover_readiness_candidate.v1"
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


@dataclass(frozen=True)
class HumanTakeoverReadinessCandidate:
    observer_binding_id: str
    emission: HumanCommandEmission
    effective_from: float
    effective_until: float
    target_effective_at: float
    deadman_rearm_event_id: str
    deadman_rearmed_at: float
    retargeter_rebase_receipt_id: str
    retargeter_rebased_at: float

    def __post_init__(self) -> None:
        for field_name in (
            "observer_binding_id",
            "deadman_rearm_event_id",
            "retargeter_rebase_receipt_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        if not isinstance(self.emission, HumanCommandEmission):
            raise TypeError("emission must be a HumanCommandEmission.")
        for field_name in (
            "effective_from",
            "effective_until",
            "target_effective_at",
            "deadman_rearmed_at",
            "retargeter_rebased_at",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite(getattr(self, field_name), field_name=field_name),
            )
        if self.effective_until <= self.effective_from:
            raise ValueError("effective_until must be later than effective_from.")
        if not self.effective_from <= self.target_effective_at < self.effective_until:
            raise ValueError("target_effective_at must be inside the effective interval.")

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "observer_binding_id": self.observer_binding_id,
                "emission_hash": self.emission.semantic_content_hash,
                "effective_from": self.effective_from,
                "effective_until": self.effective_until,
                "target_effective_at": self.target_effective_at,
                "deadman_rearm_event_id": self.deadman_rearm_event_id,
                "deadman_rearmed_at": self.deadman_rearmed_at,
                "retargeter_rebase_receipt_id": (
                    self.retargeter_rebase_receipt_id
                ),
                "retargeter_rebased_at": self.retargeter_rebased_at,
            },
            domain=HUMAN_TAKEOVER_READINESS_CANDIDATE_HASH_DOMAIN_V1,
        )


class HumanTakeoverReadinessProvider(Protocol):
    """Bounded Station adapter over lower-owned Human readiness receipts."""

    @property
    def observer_binding_id(self) -> str: ...

    def acquire(
        self,
        *,
        operation_spec: HumanTakeoverOperationSpec,
        takeover_state: HumanTakeoverState,
        timeout_sec: float,
    ) -> HumanTakeoverReadinessCandidate: ...


__all__ = [
    "HUMAN_TAKEOVER_READINESS_CANDIDATE_HASH_DOMAIN_V1",
    "HumanObservationCutAdmissionCore",
    "HumanPoseSourceContract",
    "HumanProposalAdmissionCore",
    "HumanTakeoverReadinessCandidate",
    "HumanTakeoverReadinessProvider",
    "PassiveHumanCommandEmissionIngress",
    "PassiveHumanCommandEmissionIngressStatus",
    "PassiveHumanCommandObserverFanout",
    "admit_human_command_emission",
    "admit_human_observation_cut",
]
