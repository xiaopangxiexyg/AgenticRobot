"""Pure deterministic authority and component selection."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from dataset_contracts.canonical_encoding import (
    CANONICAL_ENCODING_SCHEMA_V1,
    canonical_content_hash,
)

from .arbitration_input import ArbitrationInputSnapshot
from .proposal import ActionProposal, freeze_robot_action, robot_action_semantic_payload
from .schema import RobotAction, split_effector_name


ARBITER_CONTRACT_ID_V1 = "dexteleop.action_arbiter.v1"
ARBITER_CONTRACT_HASH_V1 = canonical_content_hash(
    {
        "contract_id": ARBITER_CONTRACT_ID_V1,
        "selection": "authority_priority_then_effective_from_sequence_lexical_id",
        "composition": "component_union_no_pose_blend",
        "gate": "input_snapshot_only",
    },
    domain="dexteleop.action_arbiter.contract.v1",
)
ARBITER_IMPLEMENTATION_HASH_V1 = canonical_content_hash(
    {"implementation": "pure_authority_component_selection_v1"},
    domain="dexteleop.action_arbiter.implementation.v1",
)
ARBITRATION_RESULT_HASH_DOMAIN_V1 = "dexteleop.arbitration_result.v1"


@dataclass(frozen=True)
class ArbitrationRejection:
    proposal_id: str
    selection_group_id: str
    reason: str


@dataclass(frozen=True)
class ArbitrationResult:
    result_id: str
    arbitration_input_snapshot_id: str
    arbitration_input_hash: str
    proposal_snapshot_id: str
    arbiter_contract_id: str
    arbiter_contract_hash: str
    arbiter_implementation_hash: str
    authority_policy_hash: str
    component_selection_policy_hash: str
    decision_schedule_spec_hash: str
    canonical_encoding_version: str
    schedule_tick_id: str
    decision_cutoff_at: float
    decided_at: float
    target_effective_at: float
    publication_deadline_at: float
    authority_epoch: int
    control_mode: str
    execution_gate_status: str
    gate_epoch: int
    selected_by_group: Mapping[str, str]
    selected_by_component: Mapping[str, str]
    owner_by_component: Mapping[str, str]
    status_by_group: Mapping[str, str]
    composed_action: RobotAction | None
    selected_action_hash: str | None
    output_status: str
    rejections: tuple[ArbitrationRejection, ...]
    attribution: Mapping[str, Mapping[str, Any]]
    semantic_content_hash: str

    def __post_init__(self) -> None:
        for name in (
            "selected_by_group",
            "selected_by_component",
            "owner_by_component",
            "status_by_group",
            "attribution",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        object.__setattr__(self, "rejections", tuple(self.rejections))


def _copy_selected_component(
    proposal: ActionProposal,
    *,
    effector_name: str,
    arm_pose: dict[str, Any],
    arm_qpos: dict[str, Any],
    hand_qpos: dict[str, Any],
    native_targets: dict[str, Any],
    frame_by_effector: dict[str, str],
    controlled_frame_by_effector: dict[str, str],
    execution_metadata: dict[str, dict[str, Any]],
) -> None:
    side, kind = split_effector_name(effector_name)
    action = proposal.action
    if kind == "arm":
        if side in action.arm_pose_by_side:
            arm_pose[side] = action.arm_pose_by_side[side]
        elif side in action.arm_qpos_by_side:
            arm_qpos[side] = action.arm_qpos_by_side[side]
        else:
            raise ValueError(f"Proposal {proposal.proposal_id} lacks {effector_name} payload.")
        metadata_key = "arm_action_specs_by_side"
    else:
        if side in action.hand_qpos_by_side:
            hand_qpos[side] = action.hand_qpos_by_side[side]
        elif side in action.end_effector_targets_by_side:
            native_targets[side] = action.end_effector_targets_by_side[side]
        else:
            raise ValueError(f"Proposal {proposal.proposal_id} lacks {effector_name} payload.")
        metadata_key = "hand_action_specs_by_side"
    if effector_name in action.frame_by_effector:
        frame_by_effector[effector_name] = action.frame_by_effector[effector_name]
    if effector_name in action.controlled_frame_by_effector:
        controlled_frame_by_effector[effector_name] = action.controlled_frame_by_effector[effector_name]
    specs = action.metadata.get(metadata_key, {})
    if side in specs:
        execution_metadata.setdefault(metadata_key, {})[side] = specs[side]


def _compose_selected_action(
    snapshot: ArbitrationInputSnapshot,
    selected_candidate_by_group: Mapping[str, Any],
) -> tuple[RobotAction | None, dict[str, str], dict[str, str], dict[str, Mapping[str, Any]]]:
    arm_pose: dict[str, Any] = {}
    arm_qpos: dict[str, Any] = {}
    hand_qpos: dict[str, Any] = {}
    native_targets: dict[str, Any] = {}
    frames: dict[str, str] = {}
    controlled_frames: dict[str, str] = {}
    metadata: dict[str, dict[str, Any]] = {}
    selected_by_component: dict[str, str] = {}
    owner_by_component: dict[str, str] = {}
    attribution: dict[str, Mapping[str, Any]] = {}
    active_effectors: list[str] = []

    for group in snapshot.active_selection_groups:
        candidate = selected_candidate_by_group.get(group)
        if candidate is None:
            continue
        proposal = candidate.proposal
        for component in snapshot.component_catalog.components_for_group(group):
            if component.component_id not in proposal.component_coverage:
                if component.coupled:
                    raise ValueError(
                        f"Coupled proposal {proposal.proposal_id} lacks {component.component_id}."
                    )
                continue
            if component.component_id in selected_by_component:
                raise ValueError("A component was selected by multiple groups.")
            _copy_selected_component(
                proposal,
                effector_name=component.effector_name,
                arm_pose=arm_pose,
                arm_qpos=arm_qpos,
                hand_qpos=hand_qpos,
                native_targets=native_targets,
                frame_by_effector=frames,
                controlled_frame_by_effector=controlled_frames,
                execution_metadata=metadata,
            )
            active_effectors.append(component.effector_name)
            selected_by_component[component.component_id] = proposal.proposal_id
            owner_by_component[component.component_id] = proposal.producer_id
            attribution[component.component_id] = MappingProxyType(
                {
                    "proposal_id": proposal.proposal_id,
                    "producer_id": proposal.producer_id,
                    "producer_kind": proposal.producer_kind,
                    "policy_node_id": proposal.policy_node_id,
                    "context_ids": proposal.context_ids,
                }
            )

    if not active_effectors:
        return None, selected_by_component, owner_by_component, attribution
    composed = RobotAction(
        action_space=snapshot.component_catalog.action_space,
        timestamp=snapshot.decision_schedule_tick.target_effective_at,
        arm_pose_by_side=arm_pose,
        arm_qpos_by_side=arm_qpos,
        hand_qpos_by_side=hand_qpos,
        end_effector_targets_by_side=native_targets,
        active_effectors=tuple(active_effectors),
        frame_by_effector=frames,
        controlled_frame_by_effector=controlled_frames,
        source="action_arbiter",
        metadata=metadata,
    )
    return freeze_robot_action(composed), selected_by_component, owner_by_component, attribution


class ActionArbiter:
    """Pure function: no stores, clocks, environment variables, or frontend reads."""

    def arbitrate(self, snapshot: ArbitrationInputSnapshot) -> ArbitrationResult:
        gate = snapshot.execution_gate_snapshot
        selected_by_group: dict[str, str] = {}
        status_by_group: dict[str, str] = {}
        selected_candidates: dict[str, Any] = {}
        rejections: list[ArbitrationRejection] = []

        if gate.status in {"hold", "fault", "transitioning"}:
            output_status = "fault" if gate.status == "fault" else "hold"
            status_by_group = {group: output_status for group in snapshot.active_selection_groups}
        else:
            candidates_by_group: dict[str, list[Any]] = {
                group: [] for group in snapshot.active_selection_groups
            }
            for candidate in snapshot.proposal_snapshot.candidates:
                if candidate.selection_group_id in candidates_by_group:
                    candidates_by_group[candidate.selection_group_id].append(candidate)

            for group in snapshot.active_selection_groups:
                lease = snapshot.authority_state_snapshot.active_leases.get(group)
                shadow_owner = snapshot.authority_policy.shadow_owner_by_group.get(group)
                ranked: list[tuple[int, float, int, str, Any]] = []
                required_components = {
                    item.component_id
                    for item in snapshot.component_catalog.components_for_group(group)
                    if item.coupled
                }
                for candidate in candidates_by_group[group]:
                    proposal = candidate.proposal
                    reason = None
                    group_components = snapshot.component_catalog.components_for_group(group)
                    covered_group_components = {
                        item.component_id
                        for item in group_components
                        if item.component_id in proposal.component_coverage
                    }
                    known_component_ids = set(snapshot.component_catalog.component_by_id)
                    if proposal.action.action_space != snapshot.component_catalog.action_space:
                        reason = "action_space_mismatch"
                    elif not set(proposal.component_coverage).issubset(known_component_ids):
                        reason = "unknown_component_coverage"
                    elif not covered_group_components:
                        reason = "selection_group_has_no_covered_component"
                    elif any(
                        item.effector_name not in proposal.action.active_effectors
                        for item in group_components
                        if item.component_id in proposal.component_coverage
                    ):
                        reason = "component_payload_missing"
                    elif lease is not None and proposal.producer_id != lease.owner_producer_id:
                        reason = "authority_lease_owner_mismatch"
                    elif (
                        snapshot.authority_state_snapshot.control_mode == "shadow"
                        and shadow_owner
                        and proposal.producer_id != shadow_owner
                    ):
                        reason = "shadow_fixed_owner_mismatch"
                    elif proposal.producer_kind == "recovery" and proposal.producer_id not in snapshot.authority_policy.recovery_producers:
                        reason = "recovery_producer_not_allowed"
                    elif proposal.authority_request != "none" and proposal.producer_id not in snapshot.authority_policy.takeover_allowed_producers:
                        reason = "takeover_request_not_allowed"
                    elif required_components and not required_components.issubset(proposal.component_coverage):
                        reason = "coupled_group_incomplete"
                    priority = snapshot.authority_policy.priority_for(proposal, group)
                    if reason is None and priority is None:
                        reason = "source_not_authorized"
                    if reason is not None:
                        rejections.append(ArbitrationRejection(proposal.proposal_id, group, reason))
                        continue
                    assert priority is not None
                    ranked.append(
                        (-priority, -proposal.effective_from, -proposal.sequence, proposal.proposal_id, candidate)
                    )
                if ranked:
                    ranked.sort()
                    winner = ranked[0][-1]
                    selected_candidates[group] = winner
                    selected_by_group[group] = winner.proposal.proposal_id
                    status_by_group[group] = "selected"
                    rejections.extend(
                        ArbitrationRejection(item[-1].proposal.proposal_id, group, "authority_superseded")
                        for item in ranked[1:]
                    )
                else:
                    status_by_group[group] = snapshot.authority_policy.fallback_by_group.get(
                        group, "no_action"
                    )

            composed, selected_by_component, owner_by_component, attribution = _compose_selected_action(
                snapshot, selected_candidates
            )
            output_status = (
                "selected"
                if composed is not None
                else "hold"
                if any(value == "hold" for value in status_by_group.values())
                else "no_action"
            )

        if gate.status in {"hold", "fault", "transitioning"}:
            composed = None
            selected_by_component = {}
            owner_by_component = {}
            attribution = {}

        selected_action_hash = (
            None
            if composed is None
            else canonical_content_hash(
                robot_action_semantic_payload(composed),
                domain="dexteleop.arbitration_selected_action.v1",
            )
        )
        tick = snapshot.decision_schedule_tick
        payload = {
            "arbitration_input_snapshot_id": snapshot.arbitration_input_snapshot_id,
            "arbitration_input_hash": snapshot.semantic_content_hash,
            "proposal_snapshot_id": snapshot.proposal_snapshot.proposal_snapshot_id,
            "arbiter_contract_id": ARBITER_CONTRACT_ID_V1,
            "arbiter_contract_hash": ARBITER_CONTRACT_HASH_V1,
            "arbiter_implementation_hash": ARBITER_IMPLEMENTATION_HASH_V1,
            "authority_policy_hash": snapshot.authority_policy.semantic_content_hash,
            "component_selection_policy_hash": snapshot.component_catalog.semantic_content_hash,
            "decision_schedule_spec_hash": tick.schedule_spec_hash,
            "canonical_encoding_version": CANONICAL_ENCODING_SCHEMA_V1,
            "schedule_tick_id": tick.schedule_tick_id,
            "decision_cutoff_at": tick.decision_cutoff_at,
            "decided_at": tick.scheduled_decision_at,
            "target_effective_at": tick.target_effective_at,
            "publication_deadline_at": tick.publication_deadline_at,
            "authority_epoch": snapshot.authority_state_snapshot.authority_epoch,
            "control_mode": snapshot.authority_state_snapshot.control_mode,
            "execution_gate_status": gate.status,
            "gate_epoch": gate.gate_epoch,
            "selected_by_group": selected_by_group,
            "selected_by_component": selected_by_component,
            "owner_by_component": owner_by_component,
            "status_by_group": status_by_group,
            "selected_action_hash": selected_action_hash,
            "output_status": output_status,
            "rejections": tuple(
                {
                    "proposal_id": item.proposal_id,
                    "selection_group_id": item.selection_group_id,
                    "reason": item.reason,
                }
                for item in rejections
            ),
            "attribution": attribution,
        }
        semantic_hash = canonical_content_hash(payload, domain=ARBITRATION_RESULT_HASH_DOMAIN_V1)
        return ArbitrationResult(
            result_id=f"arbitration-result:{semantic_hash}",
            **{key: value for key, value in payload.items() if key not in {"rejections", "attribution"}},
            composed_action=composed,
            rejections=tuple(rejections),
            attribution=attribution,
            semantic_content_hash=semantic_hash,
        )


__all__ = [
    "ARBITER_CONTRACT_ID_V1",
    "ARBITER_CONTRACT_HASH_V1",
    "ARBITER_IMPLEMENTATION_HASH_V1",
    "ActionArbiter",
    "ArbitrationRejection",
    "ArbitrationResult",
]
