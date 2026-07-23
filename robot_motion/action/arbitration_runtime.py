"""Decision scheduling and the production <=60 Hz publication gate."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable

from dataset_contracts.canonical_encoding import canonical_content_hash


MAX_PRODUCTION_ARBITER_HZ = 60.0
PRODUCTION_DECISION_RATE_POLICY_V1 = "dexteleop.production_decision_rate.v1"
DECISION_SCHEDULE_HASH_DOMAIN_V1 = "dexteleop.decision_schedule.v1"
DECISION_TICK_HASH_DOMAIN_V1 = "dexteleop.decision_schedule_tick.v1"
ARBITRATION_PUBLICATION_HASH_DOMAIN_V1 = "dexteleop.arbitration_publication.v1"


@dataclass(frozen=True)
class DecisionScheduleSpec:
    spec_id: str
    schedule_mode: str = "fixed"
    decision_hz: float = 30.0
    cutoff_lag_sec: float = 0.0
    target_lead_time_sec: float = 0.0
    publication_deadline_offset_sec: float = 0.02
    max_target_lateness_sec: float = 0.02
    late_decision_policy: str = "no_action"
    missed_tick_policy: str = "skip"
    trigger_coalescing_policy: str = "latest"
    decision_runtime_budget_sec: float = 0.005
    max_pending_trigger_evidence: int = 4096

    def __post_init__(self) -> None:
        if not str(self.spec_id).strip():
            raise ValueError("Decision schedule spec_id must be non-empty.")
        if self.schedule_mode not in {"fixed", "event_limited"}:
            raise ValueError("schedule_mode must be fixed or event_limited.")
        rate = float(self.decision_hz)
        if not math.isfinite(rate) or rate <= 0 or rate > MAX_PRODUCTION_ARBITER_HZ:
            raise ValueError(
                f"production decision_hz must be in (0, {MAX_PRODUCTION_ARBITER_HZ:g}]."
            )
        for name in (
            "cutoff_lag_sec",
            "target_lead_time_sec",
            "publication_deadline_offset_sec",
            "max_target_lateness_sec",
            "decision_runtime_budget_sec",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.late_decision_policy not in {"no_action", "hold"}:
            raise ValueError("late_decision_policy must be no_action or hold.")
        if self.missed_tick_policy != "skip":
            raise ValueError("Decision schedule v1 only supports missed_tick_policy=skip.")
        if self.trigger_coalescing_policy != "latest":
            raise ValueError("Decision schedule v1 only supports latest trigger coalescing.")
        if int(self.max_pending_trigger_evidence) <= 0:
            raise ValueError("max_pending_trigger_evidence must be positive.")

    @property
    def period_sec(self) -> float:
        return 1.0 / float(self.decision_hz)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "policy_id": PRODUCTION_DECISION_RATE_POLICY_V1,
                "spec_id": self.spec_id,
                "schedule_mode": self.schedule_mode,
                "decision_hz": self.decision_hz,
                "cutoff_lag_sec": self.cutoff_lag_sec,
                "target_lead_time_sec": self.target_lead_time_sec,
                "publication_deadline_offset_sec": self.publication_deadline_offset_sec,
                "max_target_lateness_sec": self.max_target_lateness_sec,
                "late_decision_policy": self.late_decision_policy,
                "missed_tick_policy": self.missed_tick_policy,
                "trigger_coalescing_policy": self.trigger_coalescing_policy,
                "decision_runtime_budget_sec": self.decision_runtime_budget_sec,
                "max_pending_trigger_evidence": self.max_pending_trigger_evidence,
            },
            domain=DECISION_SCHEDULE_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class DecisionScheduleTick:
    schedule_tick_id: str
    tick_index: int
    scheduled_decision_at: float
    decision_cutoff_at: float
    cutoff_admission_sequence: int
    target_effective_at: float
    publication_deadline_at: float
    trigger_ids: tuple[str, ...]
    trigger_reasons: tuple[str, ...]
    coalesced_trigger_ids: tuple[str, ...]
    coalesced_trigger_count: int
    trigger_evidence_truncated: bool
    schedule_spec_hash: str
    semantic_content_hash: str


class DecisionScheduler:
    """No-catch-up scheduler; event bursts are represented in one next tick."""

    def __init__(
        self,
        spec: DecisionScheduleSpec,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.spec = spec
        self._clock = clock
        self._next_due_at: float | None = None
        self._tick_index = 0
        self._pending_triggers: list[tuple[str, str]] = []
        self._pending_trigger_count = 0

    def trigger(self, trigger_id: str, reason: str) -> None:
        self._pending_trigger_count += 1
        self._pending_triggers.append((str(trigger_id), str(reason)))
        overflow = len(self._pending_triggers) - int(self.spec.max_pending_trigger_evidence)
        if overflow > 0:
            del self._pending_triggers[:overflow]

    def now(self) -> float:
        return float(self._clock())

    def due(self, now: float | None = None) -> bool:
        timestamp = float(self.now() if now is None else now)
        if self._next_due_at is None:
            return self.spec.schedule_mode == "fixed" or bool(self._pending_triggers)
        if self.spec.schedule_mode == "event_limited" and not self._pending_triggers:
            return False
        return timestamp + 1e-12 >= self._next_due_at

    def make_tick(
        self,
        *,
        cutoff_admission_sequence: int,
        now: float | None = None,
    ) -> DecisionScheduleTick | None:
        timestamp = float(self.now() if now is None else now)
        if not self.due(timestamp):
            return None
        triggers = tuple(self._pending_triggers)
        self._pending_triggers.clear()
        trigger_count = self._pending_trigger_count
        self._pending_trigger_count = 0
        cutoff_at = timestamp - float(self.spec.cutoff_lag_sec)
        target_at = timestamp + float(self.spec.target_lead_time_sec)
        payload = {
            "tick_index": self._tick_index,
            "scheduled_decision_at": timestamp,
            "decision_cutoff_at": cutoff_at,
            "cutoff_admission_sequence": int(cutoff_admission_sequence),
            "target_effective_at": target_at,
            "publication_deadline_at": target_at + float(self.spec.publication_deadline_offset_sec),
            "trigger_ids": tuple(item[0] for item in triggers[-1:]),
            "trigger_reasons": tuple(item[1] for item in triggers[-1:]),
            "coalesced_trigger_ids": tuple(item[0] for item in triggers[:-1]),
            "coalesced_trigger_count": max(0, trigger_count - 1),
            "trigger_evidence_truncated": trigger_count > len(triggers),
            "schedule_spec_hash": self.spec.semantic_content_hash,
        }
        semantic_hash = canonical_content_hash(payload, domain=DECISION_TICK_HASH_DOMAIN_V1)
        tick = DecisionScheduleTick(
            schedule_tick_id=f"decision-tick:{semantic_hash}",
            **payload,
            semantic_content_hash=semantic_hash,
        )
        self._tick_index += 1
        # Skip missed periods; never create a catch-up burst.
        self._next_due_at = timestamp + self.spec.period_sec
        return tick


@dataclass(frozen=True)
class ArbitrationPublicationAttempt:
    publication_attempt_id: str
    result_id: str
    attempted_at: float
    accepted: bool
    rejection_reason: str | None
    required_fallback: str | None
    coalesced_before_publication: bool
    semantic_content_hash: str


@dataclass(frozen=True)
class AuthoritativeArbitrationDecision:
    decision_id: str
    result_id: str
    arbitration_input_snapshot_id: str
    selected_action: Any
    selected_by_group: Any
    published_at: float
    authority_epoch: int
    gate_epoch: int
    semantic_content_hash: str


@dataclass(frozen=True)
class ArbitrationComputationMetrics:
    result_id: str
    started_at: float
    completed_at: float
    runtime_sec: float
    runtime_budget_sec: float
    runtime_budget_missed: bool
    publication_deadline_missed_at_completion: bool


class ArbitrationRuntime:
    """The only normal authoritative publication gate for decision policy v1."""

    def __init__(self, spec: DecisionScheduleSpec) -> None:
        self.spec = spec
        self._last_published_at: float | None = None

    def compute(
        self,
        arbiter: Any,
        snapshot: Any,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> tuple[Any, ArbitrationComputationMetrics]:
        started_at = float(clock())
        result = arbiter.arbitrate(snapshot)
        completed_at = float(clock())
        runtime_sec = max(0.0, completed_at - started_at)
        return result, ArbitrationComputationMetrics(
            result_id=result.result_id,
            started_at=started_at,
            completed_at=completed_at,
            runtime_sec=runtime_sec,
            runtime_budget_sec=self.spec.decision_runtime_budget_sec,
            runtime_budget_missed=runtime_sec > self.spec.decision_runtime_budget_sec,
            publication_deadline_missed_at_completion=(
                completed_at > result.publication_deadline_at
            ),
        )

    def publish(
        self,
        computation: Any,
        *,
        attempted_at: float,
        live_gate_snapshot: Any,
        input_snapshot: Any | None = None,
    ) -> tuple[ArbitrationPublicationAttempt, AuthoritativeArbitrationDecision | None]:
        now = float(attempted_at)
        reason: str | None = None
        fallback: str | None = None
        if getattr(computation, "output_status", None) != "selected":
            reason = "result_not_selected"
        elif getattr(computation, "decision_schedule_spec_hash", None) != self.spec.semantic_content_hash:
            reason = "decision_schedule_mismatch"
            fallback = "hold"
        elif getattr(live_gate_snapshot, "status", None) != "ready":
            reason = "live_execution_gate_not_ready"
            fallback = "hold"
        elif int(getattr(live_gate_snapshot, "gate_epoch")) != int(computation.gate_epoch):
            reason = "execution_gate_epoch_changed"
            fallback = "hold"
        elif input_snapshot is not None:
            authorization = input_snapshot.candidate_authorization_snapshot
            proposal_by_id = {
                item.proposal.proposal_id: item.proposal
                for item in input_snapshot.proposal_snapshot.candidates
            }
            for proposal_id in dict(computation.selected_by_group).values():
                proposal = proposal_by_id.get(proposal_id)
                if proposal is None:
                    reason = "selected_proposal_missing_from_input_snapshot"
                    fallback = "no_action"
                    break
                if proposal.producer_kind not in authorization.required_producer_kinds:
                    continue
                record = authorization.authorization_by_proposal_id.get(proposal_id)
                disposition = authorization.dispositions_by_proposal_id.get(
                    proposal_id,
                    "authorization_missing",
                )
                if record is None:
                    reason = f"candidate_authorization_rejected:{disposition}"
                    fallback = "no_action"
                    break
                if not record.authoritative_allowed:
                    reason = "candidate_authorization_rejected:authoritative_denied"
                    fallback = "no_action"
                    break
        if reason is None and now > float(computation.publication_deadline_at):
            reason = f"publication_deadline_missed:{self.spec.late_decision_policy}"
            fallback = self.spec.late_decision_policy
        elif reason is None and now - float(computation.target_effective_at) > float(self.spec.max_target_lateness_sec):
            reason = f"target_late:{self.spec.late_decision_policy}"
            fallback = self.spec.late_decision_policy
        elif reason is None and (
            self._last_published_at is not None
            and now - self._last_published_at + 1e-12 < self.spec.period_sec
        ):
            reason = "coalesced_before_publication"
            fallback = "no_action"

        accepted = reason is None
        payload = {
            "result_id": computation.result_id,
            "attempted_at": now,
            "accepted": accepted,
            "rejection_reason": reason,
            "required_fallback": fallback,
            "coalesced_before_publication": reason == "coalesced_before_publication",
        }
        publication_hash = canonical_content_hash(
            payload, domain=ARBITRATION_PUBLICATION_HASH_DOMAIN_V1
        )
        attempt = ArbitrationPublicationAttempt(
            publication_attempt_id=f"arbitration-publication:{publication_hash}",
            **payload,
            semantic_content_hash=publication_hash,
        )
        if not accepted:
            return attempt, None

        decision_payload = {
            "result_id": computation.result_id,
            "arbitration_input_snapshot_id": computation.arbitration_input_snapshot_id,
            "selected_action_hash": computation.selected_action_hash,
            "selected_by_group": computation.selected_by_group,
            "published_at": now,
            "authority_epoch": computation.authority_epoch,
            "gate_epoch": computation.gate_epoch,
        }
        decision_hash = canonical_content_hash(
            decision_payload, domain="dexteleop.authoritative_arbitration_decision.v1"
        )
        self._last_published_at = now
        return attempt, AuthoritativeArbitrationDecision(
            decision_id=f"arbitration-decision:{decision_hash}",
            result_id=computation.result_id,
            arbitration_input_snapshot_id=computation.arbitration_input_snapshot_id,
            selected_action=computation.composed_action,
            selected_by_group=computation.selected_by_group,
            published_at=now,
            authority_epoch=computation.authority_epoch,
            gate_epoch=computation.gate_epoch,
            semantic_content_hash=decision_hash,
        )


__all__ = [
    "MAX_PRODUCTION_ARBITER_HZ",
    "PRODUCTION_DECISION_RATE_POLICY_V1",
    "ArbitrationPublicationAttempt",
    "ArbitrationComputationMetrics",
    "ArbitrationRuntime",
    "AuthoritativeArbitrationDecision",
    "DecisionScheduleSpec",
    "DecisionScheduleTick",
    "DecisionScheduler",
]
