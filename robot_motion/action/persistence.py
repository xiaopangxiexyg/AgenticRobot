"""Pure conversion from arbitration outcome to explicit execution directives."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash

from .arbiter import ArbitrationResult


ACTION_PERSISTENCE_HASH_DOMAIN_V1 = "dexteleop.action_persistence_policy.v1"
EXECUTION_DIRECTIVE_HASH_DOMAIN_V1 = "dexteleop.execution_directive.v1"


@dataclass(frozen=True)
class ActiveExecutionSegment:
    selection_group_id: str
    segment_id: str
    selected_action_ref: str
    expires_at: float
    group_segment_generation: int


@dataclass(frozen=True)
class ActiveExecutionSnapshot:
    control_domain_id: str
    execution_epoch: int
    active_segment_by_group: Mapping[str, ActiveExecutionSegment]
    backend_hold_active: bool = False
    fault_latched: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "active_segment_by_group",
            MappingProxyType(dict(self.active_segment_by_group)),
        )


@dataclass(frozen=True)
class ActionPersistencePolicy:
    policy_id: str
    selected_segment_max_lifetime_sec: float
    no_action_behavior: str = "keep_existing_until_expiry"
    proposal_starvation_behavior: str = "keep_existing_until_expiry"
    segment_expiry_behavior: str = "cancel_group_and_hold"
    hold_behavior: str = "cancel_and_hold"
    fault_behavior: str = "cancel_and_latch_fault"
    backend_hold_behavior: str = "request_backend_hold"

    def __post_init__(self) -> None:
        if not str(self.policy_id).strip() or float(self.selected_segment_max_lifetime_sec) <= 0:
            raise ValueError("Persistence policy ID/lifetime is invalid.")
        if self.no_action_behavior != "keep_existing_until_expiry":
            raise ValueError("Persistence v1 fixes no_action to keep_existing_until_expiry.")
        if self.proposal_starvation_behavior != "keep_existing_until_expiry":
            raise ValueError("Persistence v1 fixes starvation to keep_existing_until_expiry.")
        if self.segment_expiry_behavior != "cancel_group_and_hold":
            raise ValueError("Persistence v1 fixes expiry to cancel_group_and_hold.")

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "policy_id": self.policy_id,
                "selected_segment_max_lifetime_sec": self.selected_segment_max_lifetime_sec,
                "no_action_behavior": self.no_action_behavior,
                "proposal_starvation_behavior": self.proposal_starvation_behavior,
                "segment_expiry_behavior": self.segment_expiry_behavior,
                "hold_behavior": self.hold_behavior,
                "fault_behavior": self.fault_behavior,
                "backend_hold_behavior": self.backend_hold_behavior,
            },
            domain=ACTION_PERSISTENCE_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class GroupExecutionDirective:
    selection_group_id: str
    directive: str
    selected_action_ref: str | None
    existing_segment_id: str | None
    segment_expiry: float | None
    group_segment_generation: int


@dataclass(frozen=True)
class ExecutionDirective:
    directive_id: str
    arbitration_result_id: str
    global_status: str
    directive_by_selection_group: Mapping[str, GroupExecutionDirective]
    persistence_policy_hash: str
    publication_fallback: str | None
    action_adaptation_receipt_id: str | None
    action_adaptation_receipt_hash: str | None
    action_adaptation_fallback: str | None
    requires_execution_epoch_increment: bool
    requires_backend_hold: bool
    fault_latched: bool
    semantic_content_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "directive_by_selection_group",
            MappingProxyType(dict(self.directive_by_selection_group)),
        )


def build_execution_directive(
    result: ArbitrationResult,
    *,
    active_execution: ActiveExecutionSnapshot,
    policy: ActionPersistencePolicy,
    now: float,
    supports_local_group_stop: bool,
    publication_fallback: str | None = None,
    action_adaptation_receipt_id: str | None = None,
    action_adaptation_receipt_hash: str | None = None,
    action_adaptation_fallback: str | None = None,
) -> ExecutionDirective:
    if publication_fallback not in {None, "no_action", "hold"}:
        raise ValueError("publication_fallback must be no_action, hold, or None.")
    if action_adaptation_fallback not in {None, "no_action", "hold"}:
        raise ValueError(
            "action_adaptation_fallback must be no_action, hold, or None."
        )
    if publication_fallback is not None and (
        action_adaptation_receipt_id is not None
        or action_adaptation_receipt_hash is not None
        or action_adaptation_fallback is not None
    ):
        raise ValueError(
            "A publication-rejected result cannot have adaptation evidence."
        )
    if (action_adaptation_receipt_id is None) != (
        action_adaptation_receipt_hash is None
    ):
        raise ValueError(
            "Action adaptation receipt ID/hash must be supplied together."
        )
    if action_adaptation_fallback is not None and (
        action_adaptation_receipt_id is None
    ):
        raise ValueError(
            "Action adaptation fallback requires a typed receipt."
        )
    for name, value in (
        ("action_adaptation_receipt_id", action_adaptation_receipt_id),
        ("action_adaptation_receipt_hash", action_adaptation_receipt_hash),
    ):
        if value is not None and not str(value).strip():
            raise ValueError(f"{name} must be non-empty when set.")
    current_time = float(now)
    directives: dict[str, GroupExecutionDirective] = {}
    global_status = "normal"
    requires_epoch = False
    requires_hold = False
    fault_latched = False
    effective_output_status = result.output_status
    effective_status_by_group = dict(result.status_by_group)
    effective_fallback = (
        publication_fallback
        if publication_fallback is not None
        else action_adaptation_fallback
    )
    if effective_fallback is not None:
        effective_output_status = effective_fallback
        effective_status_by_group = {
            group: effective_fallback for group in result.status_by_group
        }

    if effective_output_status in {"hold", "fault"}:
        global_status = "fault_all" if effective_output_status == "fault" else "hold_all"
        fault_latched = effective_output_status == "fault"
        requires_transition = bool(active_execution.active_segment_by_group) or not (
            active_execution.backend_hold_active
        )
        if fault_latched and not active_execution.fault_latched:
            requires_transition = True
        requires_epoch = requires_transition
        requires_hold = requires_transition

    for group, status in effective_status_by_group.items():
        active = active_execution.active_segment_by_group.get(group)
        generation = 0 if active is None else active.group_segment_generation
        if global_status != "normal" or status == "hold":
            already_holding = (
                active is None
                and active_execution.backend_hold_active
                and not requires_epoch
            )
            directives[group] = GroupExecutionDirective(
                group,
                "keep_backend_hold" if already_holding else "cancel_group_and_hold",
                None,
                None if active is None else active.segment_id,
                None if active is None else active.expires_at,
                generation if already_holding else generation + 1,
            )
            if not already_holding and not supports_local_group_stop:
                global_status = "fault_all" if fault_latched else "hold_all"
                requires_epoch = True
                requires_hold = True
        elif status == "selected":
            proposal_id = result.selected_by_group[group]
            directives[group] = GroupExecutionDirective(
                group,
                "publish_selected",
                proposal_id,
                None,
                min(
                    result.target_effective_at
                    + policy.selected_segment_max_lifetime_sec,
                    result.supported_until,
                ),
                generation + 1,
            )
        elif active is not None and current_time < active.expires_at:
            directives[group] = GroupExecutionDirective(
                group,
                "keep_existing_until_expiry",
                None,
                active.segment_id,
                active.expires_at,
                generation,
            )
        else:
            already_holding = active is None and active_execution.backend_hold_active
            directives[group] = GroupExecutionDirective(
                group,
                "keep_backend_hold" if already_holding else "cancel_group_and_hold",
                None,
                None if active is None else active.segment_id,
                None if active is None else active.expires_at,
                generation if already_holding else generation + 1,
            )
            if not already_holding and not supports_local_group_stop:
                global_status = "hold_all"
                requires_epoch = True
                requires_hold = True

    payload = {
        "arbitration_result_id": result.result_id,
        "global_status": global_status,
        "directive_by_selection_group": {
            group: {
                "directive": item.directive,
                "selected_action_ref": item.selected_action_ref,
                "existing_segment_id": item.existing_segment_id,
                "segment_expiry": item.segment_expiry,
                "group_segment_generation": item.group_segment_generation,
            }
            for group, item in directives.items()
        },
        "persistence_policy_hash": policy.semantic_content_hash,
        "publication_fallback": publication_fallback,
        "action_adaptation_receipt_id": action_adaptation_receipt_id,
        "action_adaptation_receipt_hash": action_adaptation_receipt_hash,
        "action_adaptation_fallback": action_adaptation_fallback,
        "requires_execution_epoch_increment": requires_epoch,
        "requires_backend_hold": requires_hold,
        "fault_latched": fault_latched,
    }
    semantic_hash = canonical_content_hash(payload, domain=EXECUTION_DIRECTIVE_HASH_DOMAIN_V1)
    return ExecutionDirective(
        directive_id=f"execution-directive:{semantic_hash}",
        arbitration_result_id=result.result_id,
        global_status=global_status,
        directive_by_selection_group=directives,
        persistence_policy_hash=policy.semantic_content_hash,
        publication_fallback=publication_fallback,
        action_adaptation_receipt_id=action_adaptation_receipt_id,
        action_adaptation_receipt_hash=action_adaptation_receipt_hash,
        action_adaptation_fallback=action_adaptation_fallback,
        requires_execution_epoch_increment=requires_epoch,
        requires_backend_hold=requires_hold,
        fault_latched=fault_latched,
        semantic_content_hash=semantic_hash,
    )


__all__ = [
    "ActionPersistencePolicy",
    "ActiveExecutionSegment",
    "ActiveExecutionSnapshot",
    "ExecutionDirective",
    "GroupExecutionDirective",
    "build_execution_directive",
]
