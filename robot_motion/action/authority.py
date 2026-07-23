"""Minimal explicit authority policy, event, lease, and snapshot contracts."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.runtime.control_admission import (
    ControlAdmissionCoordinator,
    ControlCut,
    admitted_at_or_before,
)
from robot_motion.runtime.identity import RuntimeFactIdentity
from robot_motion.runtime.immutable import deep_freeze

from .proposal import ActionProposal


AUTHORITY_POLICY_HASH_DOMAIN_V1 = "dexteleop.authority_policy.v1"
AUTHORITY_EVENT_HASH_DOMAIN_V1 = "dexteleop.authority_event.v1"
AUTHORITY_STATE_HASH_DOMAIN_V1 = "dexteleop.authority_state_snapshot.v1"


@dataclass(frozen=True)
class AuthorityPolicySpec:
    policy_id: str
    priority_by_group: Mapping[str, Mapping[str, int]]
    fallback_by_group: Mapping[str, str] = field(default_factory=dict)
    shadow_owner_by_group: Mapping[str, str] = field(default_factory=dict)
    takeover_allowed_producers: tuple[str, ...] = ()
    recovery_producers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.policy_id).strip():
            raise ValueError("policy_id must be non-empty.")
        priorities: dict[str, Mapping[str, int]] = {}
        for group, rules in self.priority_by_group.items():
            group_id = str(group).strip()
            if not group_id or not rules:
                raise ValueError("Authority priority groups/rules must be non-empty.")
            priorities[group_id] = MappingProxyType(
                {str(selector): int(priority) for selector, priority in rules.items()}
            )
        fallbacks = {str(key): str(value) for key, value in self.fallback_by_group.items()}
        if any(value not in {"no_action", "hold"} for value in fallbacks.values()):
            raise ValueError("Authority fallback must be no_action or hold.")
        object.__setattr__(self, "priority_by_group", MappingProxyType(priorities))
        object.__setattr__(self, "fallback_by_group", MappingProxyType(fallbacks))
        object.__setattr__(
            self,
            "shadow_owner_by_group",
            MappingProxyType({str(key): str(value) for key, value in self.shadow_owner_by_group.items()}),
        )
        object.__setattr__(self, "takeover_allowed_producers", tuple(self.takeover_allowed_producers))
        object.__setattr__(self, "recovery_producers", tuple(self.recovery_producers))

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "policy_id": self.policy_id,
                "priority_by_group": self.priority_by_group,
                "fallback_by_group": self.fallback_by_group,
                "shadow_owner_by_group": self.shadow_owner_by_group,
                "takeover_allowed_producers": self.takeover_allowed_producers,
                "recovery_producers": self.recovery_producers,
            },
            domain=AUTHORITY_POLICY_HASH_DOMAIN_V1,
        )

    def priority_for(self, proposal: ActionProposal, group_id: str) -> int | None:
        rules = self.priority_by_group.get(str(group_id), {})
        selectors = (
            proposal.producer_id,
            f"kind:{proposal.producer_kind}",
            "*",
        )
        for selector in selectors:
            if selector in rules:
                return int(rules[selector])
        return None


@dataclass(frozen=True)
class AuthorityLease:
    selection_group_id: str
    owner_producer_id: str
    acquired_event_id: str
    acquired_at: float
    expires_at: float | None

    def active_at(self, timestamp: float) -> bool:
        return self.expires_at is None or float(timestamp) < self.expires_at


@dataclass(frozen=True)
class AuthorityEventDraft:
    runtime_session_id: str
    control_domain_id: str
    sequence: int
    event_type: str
    produced_at: float
    produced_clock_domain_id: str
    selection_group_id: str | None = None
    owner_producer_id: str | None = None
    expires_at: float | None = None
    expires_clock_domain_id: str | None = None
    control_mode: str | None = None
    reason: str = "unspecified"
    audit_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AuthorityEvent:
    event_id: str
    runtime_session_id: str
    control_domain_id: str
    sequence: int
    authority_epoch: int
    event_type: str
    selection_group_id: str | None
    owner_producer_id: str | None
    control_mode: str | None
    expires_at: float | None
    expires_clock_domain_id: str | None
    produced_at: float
    produced_clock_domain_id: str
    available_at: float
    available_clock_domain_id: str
    control_admission_sequence: int
    reason: str
    semantic_content_hash: str
    audit_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class AuthorityStateSnapshot:
    control_domain_id: str
    cutoff_at: float
    cutoff_admission_sequence: int
    included_admission_sequence_min: int | None
    included_admission_sequence_max: int | None
    authority_policy_hash: str
    control_mode: str
    authority_epoch: int
    active_leases: Mapping[str, AuthorityLease]
    causal_event_ids: tuple[str, ...]
    semantic_content_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "active_leases", MappingProxyType(dict(self.active_leases)))
        object.__setattr__(self, "causal_event_ids", tuple(self.causal_event_ids))


class AuthorityStore:
    def __init__(
        self,
        *,
        runtime_session_id: str,
        coordinator: ControlAdmissionCoordinator,
        policy: AuthorityPolicySpec,
        capacity: int = 1024,
        initial_mode: str = "policy",
    ) -> None:
        if runtime_session_id != coordinator.clock_domain.runtime_session_id:
            raise ValueError("AuthorityStore session must match coordinator.")
        if int(capacity) <= 0:
            raise ValueError("AuthorityStore capacity must be positive.")
        self.runtime_session_id = runtime_session_id
        self.control_domain_id = coordinator.control_domain_id
        self.coordinator = coordinator
        self.policy = policy
        self.capacity = int(capacity)
        self.initial_mode = str(initial_mode)
        self._records: list[AuthorityEvent] = []
        self._last_sequence = -1
        self._lock = threading.RLock()

    def admit(self, draft: AuthorityEventDraft) -> AuthorityEvent:
        event_type = str(draft.event_type).strip()
        if event_type not in {"takeover", "release", "mode"}:
            raise ValueError("Authority event_type must be takeover, release, or mode.")
        if draft.runtime_session_id != self.runtime_session_id or draft.control_domain_id != self.control_domain_id:
            raise ValueError("Authority event belongs to another session/domain.")
        if int(draft.sequence) < 0 or not math.isfinite(float(draft.produced_at)):
            raise ValueError("Authority sequence/time is invalid.")
        group = None if draft.selection_group_id is None else str(draft.selection_group_id).strip()
        owner = None if draft.owner_producer_id is None else str(draft.owner_producer_id).strip()
        mode = None if draft.control_mode is None else str(draft.control_mode).strip()
        if event_type in {"takeover", "release"} and not group:
            raise ValueError("takeover/release requires selection_group_id.")
        if event_type == "takeover" and not owner:
            raise ValueError("takeover requires owner_producer_id.")
        if (
            event_type == "takeover"
            and owner not in self.policy.takeover_allowed_producers
        ):
            raise ValueError("takeover owner is not allowed by AuthorityPolicySpec.")
        if event_type == "mode" and not mode:
            raise ValueError("mode event requires control_mode.")
        if mode is not None and mode not in {
            "human",
            "policy",
            "replay",
            "hybrid",
            "shadow",
            "hold",
            "fault",
        }:
            raise ValueError("Unsupported control_mode.")
        expires_at = None if draft.expires_at is None else float(draft.expires_at)
        if expires_at is not None and not math.isfinite(expires_at):
            raise ValueError("expires_at must be finite.")
        expires_clock = (
            None
            if draft.expires_clock_domain_id is None
            else str(draft.expires_clock_domain_id).strip()
        )
        if expires_at is not None and expires_clock != self.coordinator.clock_domain.clock_domain_id:
            raise ValueError("Lease expiry must use the receiver control clock domain.")
        if expires_at is None and expires_clock is not None:
            raise ValueError("expires_clock_domain_id requires expires_at.")
        if not str(draft.produced_clock_domain_id).strip() or not str(draft.reason).strip():
            raise ValueError("Authority produced clock and reason must be non-empty.")

        def build(available_at: float, admission_sequence: int) -> AuthorityEvent:
            with self._lock:
                authority_epoch = len(self._records) + 1
                identity = RuntimeFactIdentity(
                    runtime_session_id=self.runtime_session_id,
                    control_domain_id=self.control_domain_id,
                    producer_id="authority-controller",
                    producer_epoch=0,
                    sequence=int(draft.sequence),
                    namespace="authority_event",
                )
                payload = {
                    "event_id": identity.fact_id,
                    "runtime_session_id": self.runtime_session_id,
                    "control_domain_id": self.control_domain_id,
                    "sequence": int(draft.sequence),
                    "authority_epoch": authority_epoch,
                    "event_type": event_type,
                    "selection_group_id": group,
                    "owner_producer_id": owner,
                    "control_mode": mode,
                    "expires_at": expires_at,
                    "expires_clock_domain_id": expires_clock,
                    "produced_at": float(draft.produced_at),
                    "produced_clock_domain_id": str(draft.produced_clock_domain_id),
                    "available_at": available_at,
                    "available_clock_domain_id": self.coordinator.clock_domain.clock_domain_id,
                    "control_admission_sequence": admission_sequence,
                    "reason": str(draft.reason),
                }
                return AuthorityEvent(
                    **payload,
                    semantic_content_hash=canonical_content_hash(payload, domain=AUTHORITY_EVENT_HASH_DOMAIN_V1),
                    audit_metadata=deep_freeze(dict(draft.audit_metadata)),
                )

        return self.coordinator.commit(store=self, build=build)

    def _insert_admitted(self, record: AuthorityEvent) -> None:
        with self._lock:
            if len(self._records) >= self.capacity:
                raise ValueError("AuthorityStore is full.")
            if record.sequence <= self._last_sequence:
                raise ValueError("Authority event sequence must strictly increase.")
            self._records.append(record)
            self._last_sequence = record.sequence

    def records_at_or_before(self, cut: ControlCut) -> tuple[AuthorityEvent, ...]:
        with self._lock:
            return tuple(item for item in self._records if admitted_at_or_before(item, cut))

    def build_snapshot(
        self,
        *,
        cut: ControlCut,
        records: tuple[AuthorityEvent, ...] | None = None,
    ) -> AuthorityStateSnapshot:
        causal = self.records_at_or_before(cut) if records is None else tuple(records)
        mode = self.initial_mode
        leases: dict[str, AuthorityLease] = {}
        for event in causal:
            if event.event_type == "mode":
                assert event.control_mode is not None
                mode = event.control_mode
            elif event.event_type == "takeover":
                assert event.selection_group_id is not None and event.owner_producer_id is not None
                leases[event.selection_group_id] = AuthorityLease(
                    selection_group_id=event.selection_group_id,
                    owner_producer_id=event.owner_producer_id,
                    acquired_event_id=event.event_id,
                    acquired_at=event.available_at,
                    expires_at=event.expires_at,
                )
            elif event.event_type == "release":
                assert event.selection_group_id is not None
                leases.pop(event.selection_group_id, None)
        leases = {group: lease for group, lease in leases.items() if lease.active_at(cut.cutoff_at)}
        sequences = tuple(item.control_admission_sequence for item in causal)
        payload = {
            "control_domain_id": self.control_domain_id,
            "cutoff_at": cut.cutoff_at,
            "cutoff_admission_sequence": cut.cutoff_admission_sequence,
            "included_admission_sequence_min": min(sequences) if sequences else None,
            "included_admission_sequence_max": max(sequences) if sequences else None,
            "authority_policy_hash": self.policy.semantic_content_hash,
            "control_mode": mode,
            "authority_epoch": causal[-1].authority_epoch if causal else 0,
            "active_leases": {
                group: {
                    "selection_group_id": lease.selection_group_id,
                    "owner_producer_id": lease.owner_producer_id,
                    "acquired_event_id": lease.acquired_event_id,
                    "acquired_at": lease.acquired_at,
                    "expires_at": lease.expires_at,
                }
                for group, lease in leases.items()
            },
            "causal_event_ids": tuple(item.event_id for item in causal),
        }
        return AuthorityStateSnapshot(
            control_domain_id=self.control_domain_id,
            cutoff_at=cut.cutoff_at,
            cutoff_admission_sequence=cut.cutoff_admission_sequence,
            included_admission_sequence_min=payload["included_admission_sequence_min"],
            included_admission_sequence_max=payload["included_admission_sequence_max"],
            authority_policy_hash=self.policy.semantic_content_hash,
            control_mode=mode,
            authority_epoch=payload["authority_epoch"],
            active_leases=leases,
            causal_event_ids=payload["causal_event_ids"],
            semantic_content_hash=canonical_content_hash(payload, domain=AUTHORITY_STATE_HASH_DOMAIN_V1),
        )


__all__ = [
    "AUTHORITY_POLICY_HASH_DOMAIN_V1",
    "AuthorityEvent",
    "AuthorityEventDraft",
    "AuthorityLease",
    "AuthorityPolicySpec",
    "AuthorityStateSnapshot",
    "AuthorityStore",
]
