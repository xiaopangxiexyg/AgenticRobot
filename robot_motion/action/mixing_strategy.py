"""Pure, replaceable arm-pose mixed-action strategy contracts.

The strategy layer is intentionally narrower than authority and execution.  It
may inspect an immutable causal snapshot and propose one pose-level action, but
it cannot mutate authority, consume producer queues, run safety/IK, or dispatch
commands.  Those responsibilities remain in the authoritative control path.

The v1 seam is arm-only.  It must not silently select or blend gripper/hand
payloads; those components retain their qpos/native representation and require
a separate mask-aware component strategy before a later compound action is
assembled.  The first arm implementation is deliberately conservative: it
passes through the single candidate owned by the current authority lease.
Richer arm strategies may later use the same snapshot's history,
future-action, modality, and label contexts without changing the surrounding
sim/real/model integrations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol

import numpy as np
from scipy.spatial.transform import Rotation

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.runtime.immutable import deep_freeze

from .proposal import (
    ActionProposal,
    freeze_robot_action,
    robot_action_semantic_payload,
)
from .execution_target import project_robot_action_to_components
from .hand_spec import HandActionSpec
from .schema import (
    BIMANUAL_EE_POSE_HAND16_V1,
    BIMANUAL_EE_POSE_V1,
    RobotAction,
)


POSE_MIXING_CONTEXT_KINDS_V1 = frozenset(
    {"history", "future_action", "modality", "label", "policy_context"}
)
POSE_MIXING_CONTEXT_HASH_DOMAIN_V1 = "dexteleop.pose_mixing_context.v1"
POSE_MIXING_SNAPSHOT_HASH_DOMAIN_V1 = "dexteleop.pose_mixing_snapshot.v1"
POSE_MIXING_STRATEGY_SPEC_HASH_DOMAIN_V1 = (
    "dexteleop.pose_mixing_strategy_spec.v1"
)
POSE_MIXING_DECISION_HASH_DOMAIN_V1 = "dexteleop.pose_mixing_decision.v1"
DISCRETE_OWNER_POSE_STRATEGY_ID_V1 = "discrete_authority_owner_pose_v1"
WEIGHTED_SE3_POSE_FUSION_STRATEGY_ID_V1 = "weighted_se3_pose_fusion_v1"
POSE_MIXING_ARM_COMPONENTS_V1 = ("left_arm", "right_arm")
POSE_MIXING_ACTION_SPACES_V1 = frozenset(
    {BIMANUAL_EE_POSE_V1, BIMANUAL_EE_POSE_HAND16_V1}
)
HAND_QPOS_MIXING_SNAPSHOT_HASH_DOMAIN_V1 = (
    "dexteleop.hand_qpos_mixing_snapshot.v1"
)
HAND_QPOS_MIXING_SPEC_HASH_DOMAIN_V1 = "dexteleop.hand_qpos_mixing_spec.v1"
HAND_QPOS_MIXING_DECISION_HASH_DOMAIN_V1 = (
    "dexteleop.hand_qpos_mixing_decision.v1"
)
COMPONENT_MIXING_SHADOW_HASH_DOMAIN_V1 = (
    "dexteleop.component_mixing_shadow.v1"
)
LINEAR_HAND_QPOS_INTERPOLATION_STRATEGY_ID_V1 = (
    "linear_active_mask_hand_qpos_interpolation_v1"
)


def _required(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _nonnegative(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer.")
    result = int(value)
    if result != value or result < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return result


def _pose_action_hash(action: RobotAction) -> str:
    return canonical_content_hash(
        robot_action_semantic_payload(action),
        domain="dexteleop.pose_mixing_action.v1",
    )


def _hand_action_hash(action: RobotAction) -> str:
    return canonical_content_hash(
        robot_action_semantic_payload(action),
        domain="dexteleop.hand_qpos_mixing_action.v1",
    )


def _require_pose_action(action: RobotAction) -> RobotAction:
    frozen = freeze_robot_action(action)
    if frozen.action_space not in POSE_MIXING_ACTION_SPACES_V1:
        raise ValueError(
            "Pose mixing v1 requires bimanual_ee_pose_v1 or "
            "bimanual_ee_pose_hand16_v1 actions; got "
            f"{frozen.action_space!r}."
        )
    if frozen.arm_qpos_by_side:
        raise ValueError("Pose mixing inputs cannot contain arm qpos.")
    if frozen.hand_qpos_by_side or frozen.end_effector_targets_by_side:
        raise ValueError(
            "Arm pose mixing v1 excludes hand/end-effector payloads; "
            "route them through a qpos/native component strategy."
        )
    if set(frozen.arm_pose_by_side) != {"left", "right"}:
        raise ValueError("Pose mixing v1 requires exact left/right arm poses.")
    if not {"left_arm", "right_arm"}.issubset(frozen.active_effectors):
        raise ValueError("Pose mixing v1 requires both arm effectors to be active.")
    return frozen


@dataclass(frozen=True)
class PoseMixingContext:
    """One already-admitted causal fact available to a mixing strategy.

    ``semantic_payload`` is limited to immutable semantic data or stable blob
    references.  A strategy must not read a mutable device/model backend after
    the decision cut through this object.
    """

    context_id: str
    context_kind: str
    source_fact_id: str
    source_fact_hash: str
    payload_contract_hash: str
    semantic_payload: Mapping[str, Any]
    available_at: float
    available_clock_domain_id: str
    control_admission_sequence: int
    effective_from: float | None = None
    effective_until: float | None = None
    blob_refs: tuple[str, ...] = ()
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        context_id = _required(self.context_id, name="context_id")
        kind = _required(self.context_kind, name="context_kind")
        if kind not in POSE_MIXING_CONTEXT_KINDS_V1:
            raise ValueError(f"Unsupported pose mixing context kind {kind!r}.")
        source_fact_id = _required(self.source_fact_id, name="source_fact_id")
        source_fact_hash = _required(self.source_fact_hash, name="source_fact_hash")
        contract_hash = _required(
            self.payload_contract_hash,
            name="payload_contract_hash",
        )
        available_at = _finite(self.available_at, name="available_at")
        clock_domain = _required(
            self.available_clock_domain_id,
            name="available_clock_domain_id",
        )
        admission_sequence = _nonnegative(
            self.control_admission_sequence,
            name="control_admission_sequence",
        )
        effective_from = (
            None
            if self.effective_from is None
            else _finite(self.effective_from, name="effective_from")
        )
        effective_until = (
            None
            if self.effective_until is None
            else _finite(self.effective_until, name="effective_until")
        )
        if (effective_from is None) != (effective_until is None):
            raise ValueError("Mixing context effective bounds must be both set or absent.")
        if (
            effective_from is not None
            and effective_until is not None
            and effective_until <= effective_from
        ):
            raise ValueError("Mixing context effective interval must be non-empty.")
        blob_refs = tuple(_required(value, name="blob_ref") for value in self.blob_refs)
        if len(blob_refs) != len(set(blob_refs)):
            raise ValueError("Mixing context blob_refs must be unique.")
        payload = deep_freeze(dict(self.semantic_payload))
        semantic = {
            "context_id": context_id,
            "context_kind": kind,
            "source_fact_id": source_fact_id,
            "source_fact_hash": source_fact_hash,
            "payload_contract_hash": contract_hash,
            "semantic_payload": payload,
            "available_at": available_at,
            "available_clock_domain_id": clock_domain,
            "control_admission_sequence": admission_sequence,
            "effective_from": effective_from,
            "effective_until": effective_until,
            "blob_refs": blob_refs,
        }
        object.__setattr__(self, "context_id", context_id)
        object.__setattr__(self, "context_kind", kind)
        object.__setattr__(self, "source_fact_id", source_fact_id)
        object.__setattr__(self, "source_fact_hash", source_fact_hash)
        object.__setattr__(self, "payload_contract_hash", contract_hash)
        object.__setattr__(self, "semantic_payload", payload)
        object.__setattr__(self, "available_at", available_at)
        object.__setattr__(self, "available_clock_domain_id", clock_domain)
        object.__setattr__(self, "control_admission_sequence", admission_sequence)
        object.__setattr__(self, "effective_from", effective_from)
        object.__setattr__(self, "effective_until", effective_until)
        object.__setattr__(self, "blob_refs", blob_refs)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                semantic,
                domain=POSE_MIXING_CONTEXT_HASH_DOMAIN_V1,
            ),
        )


@dataclass(frozen=True)
class PoseMixingCandidate:
    source_slot_id: str
    proposal: ActionProposal
    projected_action: RobotAction | None = field(default=None, repr=False, compare=False)
    action: RobotAction = field(init=False)
    action_hash: str = field(init=False)

    def __post_init__(self) -> None:
        source_slot_id = _required(self.source_slot_id, name="source_slot_id")
        if not isinstance(self.proposal, ActionProposal):
            raise TypeError("proposal must be an admitted ActionProposal.")
        action = _require_pose_action(
            self.proposal.action
            if self.projected_action is None
            else self.projected_action
        )
        object.__setattr__(self, "source_slot_id", source_slot_id)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "action_hash", _pose_action_hash(action))


@dataclass(frozen=True)
class PoseMixingStrategySpec:
    strategy_id: str
    algorithm: str
    accepted_context_kinds: tuple[str, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        strategy_id = _required(self.strategy_id, name="strategy_id")
        algorithm = _required(self.algorithm, name="algorithm")
        kinds = tuple(
            _required(value, name="accepted_context_kind")
            for value in self.accepted_context_kinds
        )
        if len(kinds) != len(set(kinds)) or not set(kinds).issubset(
            POSE_MIXING_CONTEXT_KINDS_V1
        ):
            raise ValueError("accepted_context_kinds must be unique supported kinds.")
        object.__setattr__(self, "strategy_id", strategy_id)
        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "accepted_context_kinds", kinds)
        object.__setattr__(self, "parameters", deep_freeze(dict(self.parameters)))

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "strategy_id": self.strategy_id,
                "algorithm": self.algorithm,
                "accepted_context_kinds": self.accepted_context_kinds,
                "parameters": self.parameters,
            },
            domain=POSE_MIXING_STRATEGY_SPEC_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class PoseMixingInputSnapshot:
    snapshot_id: str
    runtime_session_id: str
    control_domain_id: str
    embodiment_instance_id: str
    cutoff_at: float
    cutoff_admission_sequence: int
    target_effective_at: float
    clock_domain_id: str
    arbitration_input_snapshot_id: str
    arbitration_input_snapshot_hash: str
    authority_state_snapshot_hash: str
    execution_gate_status: str
    current_owner_by_group: Mapping[str, str]
    candidate_by_source_slot: Mapping[str, PoseMixingCandidate]
    contexts: tuple[PoseMixingContext, ...]
    semantic_content_hash: str
    component_dependencies: tuple[str, ...] = POSE_MIXING_ARM_COMPONENTS_V1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "current_owner_by_group",
            MappingProxyType(dict(self.current_owner_by_group)),
        )
        object.__setattr__(
            self,
            "candidate_by_source_slot",
            MappingProxyType(dict(self.candidate_by_source_slot)),
        )
        object.__setattr__(self, "contexts", tuple(self.contexts))
        dependencies = tuple(_required(value, name="component_dependency") for value in self.component_dependencies)
        if len(dependencies) != len(set(dependencies)) or not dependencies:
            raise ValueError("Pose mixing component_dependencies must be unique and non-empty.")
        object.__setattr__(self, "component_dependencies", dependencies)


@dataclass(frozen=True)
class PoseMixingDecision:
    decision_id: str
    strategy_id: str
    strategy_spec_hash: str
    input_snapshot_id: str
    input_snapshot_hash: str
    status: str
    reason: str
    selected_source_slot_ids: tuple[str, ...]
    selected_proposal_ids: tuple[str, ...]
    used_context_ids: tuple[str, ...]
    output_action: RobotAction | None
    output_action_hash: str | None
    semantic_content_hash: str


class PoseMixingStrategy(Protocol):
    @property
    def spec(self) -> PoseMixingStrategySpec: ...

    def decide(self, snapshot: PoseMixingInputSnapshot) -> PoseMixingDecision: ...


def build_pose_mixing_input_snapshot(
    arbitration_input: Any,
    *,
    source_slot_by_producer: Mapping[str, str],
    clock_domain_id: str,
    contexts: tuple[PoseMixingContext, ...] = (),
    component_dependencies: tuple[str, ...] | None = None,
) -> PoseMixingInputSnapshot:
    """Project one arbitration cut into the model/device-independent seam."""

    proposal_snapshot = arbitration_input.proposal_snapshot
    authority_snapshot = arbitration_input.authority_state_snapshot
    tick = arbitration_input.decision_schedule_tick
    cutoff_at = _finite(proposal_snapshot.cutoff_at, name="cutoff_at")
    cutoff_sequence = _nonnegative(
        proposal_snapshot.cutoff_admission_sequence,
        name="cutoff_admission_sequence",
    )
    clock_domain_id = _required(clock_domain_id, name="clock_domain_id")
    dependencies = (
        POSE_MIXING_ARM_COMPONENTS_V1
        if component_dependencies is None
        else tuple(
            _required(value, name="component_dependency")
            for value in component_dependencies
        )
    )
    if dependencies != POSE_MIXING_ARM_COMPONENTS_V1:
        raise ValueError(
            "Pose mixing v1 currently requires the coupled left/right arm components."
        )
    candidates: dict[str, PoseMixingCandidate] = {}
    producer_slots = {
        _required(producer, name="producer_id"): _required(slot, name="source_slot_id")
        for producer, slot in source_slot_by_producer.items()
    }
    for item in proposal_snapshot.candidates:
        proposal = item.proposal
        slot = producer_slots.get(proposal.producer_id)
        if slot is None:
            continue
        if slot in candidates:
            raise ValueError("One source slot produced multiple pose mixing candidates.")
        projected_action = None
        if component_dependencies is not None:
            projected_action = project_robot_action_to_components(
                proposal.action,
                dependencies,
            )
        candidates[slot] = PoseMixingCandidate(
            source_slot_id=slot,
            proposal=proposal,
            projected_action=projected_action,
        )
    context_tuple = tuple(contexts)
    if len({item.context_id for item in context_tuple}) != len(context_tuple):
        raise ValueError("Pose mixing context IDs must be unique.")
    for context in context_tuple:
        if context.available_clock_domain_id != clock_domain_id:
            raise ValueError("Mixing context uses a different control clock domain.")
        if (
            context.available_at > cutoff_at
            or context.control_admission_sequence > cutoff_sequence
        ):
            raise ValueError("Mixing context was not available at the decision cut.")
    owners = {
        group: lease.owner_producer_id
        for group, lease in authority_snapshot.active_leases.items()
    }
    payload = {
        "runtime_session_id": arbitration_input.runtime_session_id,
        "control_domain_id": arbitration_input.control_domain_id,
        "embodiment_instance_id": arbitration_input.embodiment_instance_id,
        "cutoff_at": cutoff_at,
        "cutoff_admission_sequence": cutoff_sequence,
        "target_effective_at": tick.target_effective_at,
        "clock_domain_id": clock_domain_id,
        "arbitration_input_snapshot_id": (
            arbitration_input.arbitration_input_snapshot_id
        ),
        "arbitration_input_snapshot_hash": arbitration_input.semantic_content_hash,
        "authority_state_snapshot_hash": authority_snapshot.semantic_content_hash,
        "execution_gate_status": arbitration_input.execution_gate_snapshot.status,
        "current_owner_by_group": owners,
        "candidate_by_source_slot": {
            slot: {
                "proposal_id": candidate.proposal.proposal_id,
                "proposal_hash": candidate.proposal.semantic_content_hash,
                "producer_id": candidate.proposal.producer_id,
                "action_hash": candidate.action_hash,
            }
            for slot, candidate in sorted(candidates.items())
        },
        "component_dependencies": dependencies,
        "contexts": tuple(
            {
                "context_id": context.context_id,
                "context_kind": context.context_kind,
                "semantic_content_hash": context.semantic_content_hash,
            }
            for context in context_tuple
        ),
    }
    digest = canonical_content_hash(
        payload,
        domain=POSE_MIXING_SNAPSHOT_HASH_DOMAIN_V1,
    )
    return PoseMixingInputSnapshot(
        snapshot_id=f"pose-mixing-snapshot:{digest}",
        runtime_session_id=arbitration_input.runtime_session_id,
        control_domain_id=arbitration_input.control_domain_id,
        embodiment_instance_id=arbitration_input.embodiment_instance_id,
        cutoff_at=cutoff_at,
        cutoff_admission_sequence=cutoff_sequence,
        target_effective_at=tick.target_effective_at,
        clock_domain_id=clock_domain_id,
        arbitration_input_snapshot_id=(
            arbitration_input.arbitration_input_snapshot_id
        ),
        arbitration_input_snapshot_hash=arbitration_input.semantic_content_hash,
        authority_state_snapshot_hash=authority_snapshot.semantic_content_hash,
        execution_gate_status=arbitration_input.execution_gate_snapshot.status,
        current_owner_by_group=owners,
        candidate_by_source_slot=candidates,
        contexts=context_tuple,
        semantic_content_hash=digest,
        component_dependencies=dependencies,
    )


def build_pose_mixing_decision(
    snapshot: PoseMixingInputSnapshot,
    *,
    spec: PoseMixingStrategySpec,
    status: str,
    reason: str,
    selected: tuple[PoseMixingCandidate, ...] = (),
    used_context_ids: tuple[str, ...] = (),
    output_action: RobotAction | None = None,
) -> PoseMixingDecision:
    normalized_status = _required(status, name="status")
    if normalized_status not in {"selected", "no_action", "invalid"}:
        raise ValueError("Unsupported pose mixing decision status.")
    normalized_reason = _required(reason, name="reason")
    context_ids = tuple(_required(value, name="used_context_id") for value in used_context_ids)
    available_context_ids = {item.context_id for item in snapshot.contexts}
    if len(context_ids) != len(set(context_ids)) or not set(context_ids).issubset(
        available_context_ids
    ):
        raise ValueError("used_context_ids must uniquely reference the input snapshot.")
    frozen_action = None if output_action is None else _require_pose_action(output_action)
    if (normalized_status == "selected") != (frozen_action is not None):
        raise ValueError("Only a selected pose mixing decision may contain an action.")
    selected_slots = tuple(item.source_slot_id for item in selected)
    selected_proposals = tuple(item.proposal.proposal_id for item in selected)
    if len(selected_slots) != len(set(selected_slots)):
        raise ValueError("Selected pose mixing source slots must be unique.")
    for candidate in selected:
        snapshot_candidate = snapshot.candidate_by_source_slot.get(
            candidate.source_slot_id
        )
        if (
            snapshot_candidate is None
            or snapshot_candidate.proposal.proposal_id
            != candidate.proposal.proposal_id
        ):
            raise ValueError("Selected candidate does not belong to the input snapshot.")
    context_by_id = {item.context_id: item for item in snapshot.contexts}
    if any(
        context_by_id[context_id].context_kind
        not in spec.accepted_context_kinds
        for context_id in context_ids
    ):
        raise ValueError("Strategy used a context kind not declared by its spec.")
    action_hash = None if frozen_action is None else _pose_action_hash(frozen_action)
    payload = {
        "strategy_id": spec.strategy_id,
        "strategy_spec_hash": spec.semantic_content_hash,
        "input_snapshot_id": snapshot.snapshot_id,
        "input_snapshot_hash": snapshot.semantic_content_hash,
        "status": normalized_status,
        "reason": normalized_reason,
        "selected_source_slot_ids": selected_slots,
        "selected_proposal_ids": selected_proposals,
        "used_context_ids": context_ids,
        "output_action_hash": action_hash,
    }
    digest = canonical_content_hash(
        payload,
        domain=POSE_MIXING_DECISION_HASH_DOMAIN_V1,
    )
    return PoseMixingDecision(
        decision_id=f"pose-mixing-decision:{digest}",
        output_action=frozen_action,
        semantic_content_hash=digest,
        **payload,
    )


class DiscreteOwnerPoseMixingStrategy:
    """Behavior-equivalent v1 branch: pass through the active lease owner."""

    def __init__(self, *, selection_group_id: str = "arms") -> None:
        group = _required(selection_group_id, name="selection_group_id")
        self.selection_group_id = group
        self._spec = PoseMixingStrategySpec(
            strategy_id=DISCRETE_OWNER_POSE_STRATEGY_ID_V1,
            algorithm="pass_through_current_authority_owner",
            accepted_context_kinds=(),
            parameters={"selection_group_id": group},
        )

    @property
    def spec(self) -> PoseMixingStrategySpec:
        return self._spec

    def decide(self, snapshot: PoseMixingInputSnapshot) -> PoseMixingDecision:
        if snapshot.execution_gate_status != "ready":
            return build_pose_mixing_decision(
                snapshot,
                spec=self.spec,
                status="no_action",
                reason=f"execution_gate_{snapshot.execution_gate_status}",
            )
        owner = snapshot.current_owner_by_group.get(self.selection_group_id)
        if owner is None:
            return build_pose_mixing_decision(
                snapshot,
                spec=self.spec,
                status="no_action",
                reason="authority_owner_missing",
            )
        selected = tuple(
            candidate
            for candidate in snapshot.candidate_by_source_slot.values()
            if candidate.proposal.producer_id == owner
            and self.selection_group_id in candidate.proposal.selection_groups
        )
        if len(selected) != 1:
            return build_pose_mixing_decision(
                snapshot,
                spec=self.spec,
                status="no_action" if not selected else "invalid",
                reason=(
                    "owner_candidate_missing"
                    if not selected
                    else "owner_candidate_ambiguous"
                ),
            )
        candidate = selected[0]
        return build_pose_mixing_decision(
            snapshot,
            spec=self.spec,
            status="selected",
            reason="current_authority_owner_passthrough",
            selected=selected,
            output_action=candidate.action,
        )


def _normalized_pose_quaternion(
    pose: np.ndarray,
    *,
    source_slot_id: str,
    side: str,
) -> np.ndarray:
    quaternion = np.asarray(pose[3:7], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError(
            f"{source_slot_id}:{side} has a zero or invalid quaternion."
        )
    return quaternion / norm


def _pose_action_frame_signature(
    action: RobotAction,
) -> tuple[str, tuple[tuple[str, str, str], ...]]:
    return (
        action.action_space,
        tuple(
            (
                side,
                action.arm_frame(side),
                action.arm_controlled_frame(side),
            )
            for side in ("left", "right")
        ),
    )


def fuse_weighted_se3_pose_actions(
    actions_by_source_slot: Mapping[str, RobotAction],
    *,
    weights_by_source_slot: Mapping[str, float],
    target_effective_at: float,
    max_translation_disagreement_m: float,
    max_rotation_disagreement_rad: float,
    source: str = "pose_fusion:weighted_se3_pose_fusion_v1",
) -> RobotAction:
    """Fuse compatible bilateral arm poses without authority or side effects."""

    actions = {
        _required(slot, name="source_slot_id"): _require_pose_action(action)
        for slot, action in dict(actions_by_source_slot).items()
    }
    weights = {
        _required(slot, name="source_slot_id"): _finite(
            value,
            name=f"weight[{slot}]",
        )
        for slot, value in dict(weights_by_source_slot).items()
    }
    if set(actions) != set(weights) or not actions:
        raise ValueError("Pose fusion actions and weights must have identical slots.")
    if any(value <= 0.0 for value in weights.values()):
        raise ValueError("Pose fusion selected weights must be positive.")
    translation_limit = _finite(
        max_translation_disagreement_m,
        name="max_translation_disagreement_m",
    )
    rotation_limit = _finite(
        max_rotation_disagreement_rad,
        name="max_rotation_disagreement_rad",
    )
    if translation_limit < 0.0:
        raise ValueError("max_translation_disagreement_m must be non-negative.")
    if rotation_limit < 0.0 or rotation_limit >= math.pi:
        raise ValueError("max_rotation_disagreement_rad must be in [0, pi).")
    target = _finite(target_effective_at, name="target_effective_at")
    total_weight = float(sum(weights.values()))
    if total_weight <= 0.0:
        raise ValueError("Pose fusion selected weights must have positive sum.")
    normalized_weights = {
        slot: weight / total_weight for slot, weight in sorted(weights.items())
    }
    signatures = {
        _pose_action_frame_signature(action) for action in actions.values()
    }
    if len(signatures) != 1:
        raise ValueError("frame_or_action_space_mismatch")
    reference_slot = min(
        normalized_weights,
        key=lambda slot: (-normalized_weights[slot], slot),
    )
    reference_action = actions[reference_slot]
    fused_pose_by_side: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        translations = {
            slot: np.asarray(action.arm_pose_by_side[side][:3], dtype=np.float64)
            for slot, action in actions.items()
        }
        quaternions = {
            slot: _normalized_pose_quaternion(
                action.arm_pose_by_side[side],
                source_slot_id=slot,
                side=side,
            )
            for slot, action in actions.items()
        }
        slots = tuple(normalized_weights)
        for first_index, first_slot in enumerate(slots):
            for second_slot in slots[first_index + 1 :]:
                translation_disagreement = float(
                    np.linalg.norm(
                        translations[first_slot] - translations[second_slot]
                    )
                )
                if translation_disagreement > translation_limit:
                    raise ValueError(
                        "translation_disagreement:"
                        f"{side}:{translation_disagreement:.9g}>"
                        f"{translation_limit:.9g}"
                    )
                relative_rotation = (
                    Rotation.from_quat(quaternions[first_slot]).inv()
                    * Rotation.from_quat(quaternions[second_slot])
                )
                rotation_disagreement = float(relative_rotation.magnitude())
                if rotation_disagreement > rotation_limit:
                    raise ValueError(
                        "rotation_disagreement:"
                        f"{side}:{rotation_disagreement:.9g}>"
                        f"{rotation_limit:.9g}"
                    )
        fused_translation = sum(
            normalized_weights[slot] * translations[slot] for slot in slots
        )
        reference_rotation = Rotation.from_quat(quaternions[reference_slot])
        fused_rotvec = sum(
            normalized_weights[slot]
            * (
                reference_rotation.inv()
                * Rotation.from_quat(quaternions[slot])
            ).as_rotvec()
            for slot in slots
        )
        fused_quaternion = (
            reference_rotation * Rotation.from_rotvec(fused_rotvec)
        ).as_quat()
        if float(np.dot(fused_quaternion, quaternions[reference_slot])) < 0.0:
            fused_quaternion = -fused_quaternion
        fused_pose_by_side[side] = np.concatenate(
            (fused_translation, fused_quaternion)
        ).astype(np.float32)
    return RobotAction(
        action_space=reference_action.action_space,
        timestamp=target,
        arm_pose_by_side=fused_pose_by_side,
        active_effectors=POSE_MIXING_ARM_COMPONENTS_V1,
        frame_by_effector={
            effector: reference_action.frame_by_effector[effector]
            for effector in POSE_MIXING_ARM_COMPONENTS_V1
        },
        controlled_frame_by_effector={
            effector: reference_action.controlled_frame_by_effector[effector]
            for effector in POSE_MIXING_ARM_COMPONENTS_V1
        },
        source=_required(source, name="source"),
    )


class WeightedSE3PoseFusionStrategy:
    """Fixed-weight, frame-explicit arm pose fusion for pure/shadow use.

    Translation uses a normalized Euclidean weighted mean. Rotation uses a
    shortest-path SO(3) tangent mean around a deterministic reference
    candidate. Canonical safety, IK, authority, and dispatch remain downstream
    and are intentionally not called here.
    """

    def __init__(
        self,
        *,
        weights_by_source_slot: Mapping[str, float],
        max_translation_disagreement_m: float = 0.20,
        max_rotation_disagreement_rad: float = math.radians(60.0),
        missing_candidate_behavior: str = "no_action",
        selection_group_id: str = "arms",
    ) -> None:
        weights: dict[str, float] = {}
        for raw_slot, raw_weight in dict(weights_by_source_slot).items():
            slot = _required(raw_slot, name="source_slot_id")
            weight = _finite(raw_weight, name=f"weight[{slot}]")
            if weight < 0.0:
                raise ValueError("Pose fusion weights must be non-negative.")
            weights[slot] = weight
        if not weights or not any(value > 0.0 for value in weights.values()):
            raise ValueError("Pose fusion requires at least one positive weight.")
        translation_limit = _finite(
            max_translation_disagreement_m,
            name="max_translation_disagreement_m",
        )
        if translation_limit < 0.0:
            raise ValueError(
                "max_translation_disagreement_m must be non-negative."
            )
        rotation_limit = _finite(
            max_rotation_disagreement_rad,
            name="max_rotation_disagreement_rad",
        )
        if rotation_limit < 0.0 or rotation_limit >= math.pi:
            raise ValueError(
                "max_rotation_disagreement_rad must be in [0, pi)."
            )
        missing_behavior = _required(
            missing_candidate_behavior,
            name="missing_candidate_behavior",
        )
        if missing_behavior not in {"no_action", "renormalize_available"}:
            raise ValueError("Unsupported pose fusion missing_candidate_behavior.")
        selection_group = _required(
            selection_group_id,
            name="selection_group_id",
        )
        self.weights_by_source_slot = MappingProxyType(weights)
        self.max_translation_disagreement_m = translation_limit
        self.max_rotation_disagreement_rad = rotation_limit
        self.missing_candidate_behavior = missing_behavior
        self.selection_group_id = selection_group
        self._spec = PoseMixingStrategySpec(
            strategy_id=WEIGHTED_SE3_POSE_FUSION_STRATEGY_ID_V1,
            algorithm=(
                "translation_weighted_mean+"
                "so3_reference_tangent_weighted_mean_v1"
            ),
            accepted_context_kinds=(),
            parameters={
                "weights_by_source_slot": weights,
                "max_translation_disagreement_m": translation_limit,
                "max_rotation_disagreement_rad": rotation_limit,
                "missing_candidate_behavior": missing_behavior,
                "selection_group_id": selection_group,
                "quaternion_layout": "xyzw",
                "reference_selection": (
                    "highest_normalized_weight_then_source_slot"
                ),
            },
        )

    @property
    def spec(self) -> PoseMixingStrategySpec:
        return self._spec

    def decide(self, snapshot: PoseMixingInputSnapshot) -> PoseMixingDecision:
        if snapshot.execution_gate_status != "ready":
            return build_pose_mixing_decision(
                snapshot,
                spec=self.spec,
                status="no_action",
                reason=f"execution_gate_{snapshot.execution_gate_status}",
            )
        positive_weights = {
            slot: weight
            for slot, weight in self.weights_by_source_slot.items()
            if weight > 0.0
        }
        missing = tuple(
            sorted(
                slot
                for slot in positive_weights
                if slot not in snapshot.candidate_by_source_slot
            )
        )
        if missing and self.missing_candidate_behavior == "no_action":
            return build_pose_mixing_decision(
                snapshot,
                spec=self.spec,
                status="no_action",
                reason=f"weighted_candidate_missing:{','.join(missing)}",
            )
        available_weights = {
            slot: weight
            for slot, weight in positive_weights.items()
            if slot in snapshot.candidate_by_source_slot
        }
        total_weight = float(sum(available_weights.values()))
        if total_weight <= 0.0:
            return build_pose_mixing_decision(
                snapshot,
                spec=self.spec,
                status="no_action",
                reason="weighted_candidate_unavailable",
            )
        normalized_weights = {
            slot: weight / total_weight
            for slot, weight in sorted(available_weights.items())
        }
        selected = tuple(
            snapshot.candidate_by_source_slot[slot]
            for slot in normalized_weights
        )
        if any(
            self.selection_group_id not in candidate.proposal.selection_groups
            for candidate in selected
        ):
            return build_pose_mixing_decision(
                snapshot,
                spec=self.spec,
                status="invalid",
                reason="pose_fusion_selection_group_mismatch",
            )
        try:
            output_action = fuse_weighted_se3_pose_actions(
                {
                    candidate.source_slot_id: candidate.action
                    for candidate in selected
                },
                weights_by_source_slot=normalized_weights,
                target_effective_at=snapshot.target_effective_at,
                max_translation_disagreement_m=(
                    self.max_translation_disagreement_m
                ),
                max_rotation_disagreement_rad=(
                    self.max_rotation_disagreement_rad
                ),
                source=f"pose_fusion:{self.spec.strategy_id}",
            )
        except ValueError as exc:
            reason = (
                "pose_fusion_frame_or_action_space_mismatch"
                if str(exc) == "frame_or_action_space_mismatch"
                else f"pose_fusion_invalid:{exc}"
            )
            return build_pose_mixing_decision(
                snapshot,
                spec=self.spec,
                status="invalid",
                reason=reason,
            )

        return build_pose_mixing_decision(
            snapshot,
            spec=self.spec,
            status="selected",
            reason=(
                "weighted_se3_pose_fusion_renormalized"
                if missing
                else "weighted_se3_pose_fusion"
            ),
            selected=selected,
            output_action=output_action,
        )


def _hand_action_spec_hash(spec: HandActionSpec) -> str:
    return canonical_content_hash(
        spec.to_metadata(),
        domain="dexteleop.hand_action_spec.semantic.v1",
    )


def _hand_padding_value(spec: HandActionSpec) -> float:
    return float("nan") if spec.invalid_slot_value.strip().lower() == "nan" else 0.0


@dataclass(frozen=True)
class HandQposMixingCandidate:
    """One admitted proposal projected to one hand/gripper component."""

    source_slot_id: str
    proposal: ActionProposal
    hand_spec: HandActionSpec
    action: RobotAction = field(init=False)
    full_values: np.ndarray = field(init=False)
    active_values: np.ndarray = field(init=False)
    action_hash: str = field(init=False)

    def __post_init__(self) -> None:
        source_slot_id = _required(self.source_slot_id, name="source_slot_id")
        if not isinstance(self.proposal, ActionProposal):
            raise TypeError("proposal must be an admitted ActionProposal.")
        if not isinstance(self.hand_spec, HandActionSpec):
            raise TypeError("hand_spec must be a HandActionSpec.")
        action = freeze_robot_action(self.proposal.action)
        side = self.hand_spec.side
        if side not in action.hand_qpos_by_side:
            raise ValueError(
                f"Proposal {self.proposal.proposal_id} lacks {side}_hand qpos."
            )
        effector = f"{side}_hand"
        if effector not in action.active_effectors:
            raise ValueError(f"Proposal {self.proposal.proposal_id} does not activate {effector}.")
        values = np.asarray(action.hand_qpos_by_side[side], dtype=np.float32).reshape(-1)
        if values.shape != (self.hand_spec.slot_count,):
            raise ValueError(
                f"{effector} qpos shape {values.shape} does not match hand spec slot_count="
                f"{self.hand_spec.slot_count}."
            )
        active = self.hand_spec.active_values(values)
        object.__setattr__(self, "source_slot_id", source_slot_id)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "full_values", deep_freeze(values))
        object.__setattr__(self, "active_values", deep_freeze(active))
        object.__setattr__(self, "action_hash", _hand_action_hash(action))


@dataclass(frozen=True)
class HandQposMixingInputSnapshot:
    component_id: str
    side: str
    action_space: str
    hand_action_spec_hash: str
    cutoff_at: float
    cutoff_admission_sequence: int
    target_effective_at: float
    clock_domain_id: str
    arbitration_input_snapshot_id: str
    arbitration_input_snapshot_hash: str
    authority_state_snapshot_hash: str
    execution_gate_status: str
    candidate_by_source_slot: Mapping[str, HandQposMixingCandidate]
    semantic_content_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_by_source_slot",
            MappingProxyType(dict(self.candidate_by_source_slot)),
        )

    @property
    def snapshot_id(self) -> str:
        return f"hand-qpos-mixing-snapshot:{self.semantic_content_hash}"


def build_hand_qpos_mixing_input_snapshot(
    arbitration_input: Any,
    *,
    source_slot_by_producer: Mapping[str, str],
    side: str,
    hand_spec: HandActionSpec,
    clock_domain_id: str,
    component_id: str | None = None,
) -> HandQposMixingInputSnapshot:
    """Project one arbitration cut into a hand/gripper qpos component cut."""

    if not isinstance(hand_spec, HandActionSpec):
        raise TypeError("hand_spec must be a HandActionSpec.")
    if str(side) != hand_spec.side:
        raise ValueError("side must match hand_spec.side.")
    proposal_snapshot = arbitration_input.proposal_snapshot
    authority_snapshot = arbitration_input.authority_state_snapshot
    tick = arbitration_input.decision_schedule_tick
    cutoff_at = _finite(proposal_snapshot.cutoff_at, name="cutoff_at")
    cutoff_sequence = _nonnegative(
        proposal_snapshot.cutoff_admission_sequence,
        name="cutoff_admission_sequence",
    )
    clock_domain_id = _required(clock_domain_id, name="clock_domain_id")
    producer_slots = {
        _required(producer, name="producer_id"): _required(slot, name="source_slot_id")
        for producer, slot in source_slot_by_producer.items()
    }
    candidates: dict[str, HandQposMixingCandidate] = {}
    for item in proposal_snapshot.candidates:
        proposal = item.proposal
        slot = producer_slots.get(proposal.producer_id)
        if slot is None:
            continue
        if slot in candidates:
            raise ValueError("One source slot produced multiple hand qpos candidates.")
        candidate = HandQposMixingCandidate(
            source_slot_id=slot,
            proposal=proposal,
            hand_spec=hand_spec,
        )
        if candidates and candidate.action.action_space != next(iter(candidates.values())).action.action_space:
            raise ValueError("Hand qpos candidates must use one action space.")
        candidates[slot] = candidate
    action_space = next(iter(candidates.values())).action.action_space if candidates else ""
    component = _required(component_id or f"{hand_spec.side}_hand", name="component_id")
    payload = {
        "component_id": component,
        "side": hand_spec.side,
        "action_space": action_space,
        "hand_action_spec_hash": _hand_action_spec_hash(hand_spec),
        "cutoff_at": cutoff_at,
        "cutoff_admission_sequence": cutoff_sequence,
        "target_effective_at": tick.target_effective_at,
        "clock_domain_id": clock_domain_id,
        "arbitration_input_snapshot_id": arbitration_input.arbitration_input_snapshot_id,
        "arbitration_input_snapshot_hash": arbitration_input.semantic_content_hash,
        "authority_state_snapshot_hash": authority_snapshot.semantic_content_hash,
        "execution_gate_status": arbitration_input.execution_gate_snapshot.status,
        "candidate_by_source_slot": {
            slot: {
                "proposal_id": candidate.proposal.proposal_id,
                "proposal_hash": candidate.proposal.semantic_content_hash,
                "producer_id": candidate.proposal.producer_id,
                "action_hash": candidate.action_hash,
                "active_values": tuple(float(value) for value in candidate.active_values),
            }
            for slot, candidate in sorted(candidates.items())
        },
    }
    digest = canonical_content_hash(
        payload,
        domain=HAND_QPOS_MIXING_SNAPSHOT_HASH_DOMAIN_V1,
    )
    return HandQposMixingInputSnapshot(
        component_id=component,
        side=hand_spec.side,
        action_space=action_space,
        hand_action_spec_hash=_hand_action_spec_hash(hand_spec),
        cutoff_at=cutoff_at,
        cutoff_admission_sequence=cutoff_sequence,
        target_effective_at=tick.target_effective_at,
        clock_domain_id=clock_domain_id,
        arbitration_input_snapshot_id=arbitration_input.arbitration_input_snapshot_id,
        arbitration_input_snapshot_hash=arbitration_input.semantic_content_hash,
        authority_state_snapshot_hash=authority_snapshot.semantic_content_hash,
        execution_gate_status=arbitration_input.execution_gate_snapshot.status,
        candidate_by_source_slot=candidates,
        semantic_content_hash=digest,
    )


@dataclass(frozen=True)
class HandQposMixingStrategySpec:
    strategy_id: str
    algorithm: str
    component_id: str
    action_space: str
    hand_action_spec_hash: str
    weights_by_source_slot: Mapping[str, float]
    missing_candidate_behavior: str = "no_action"

    def __post_init__(self) -> None:
        strategy_id = _required(self.strategy_id, name="strategy_id")
        algorithm = _required(self.algorithm, name="algorithm")
        component_id = _required(self.component_id, name="component_id")
        action_space = _required(self.action_space, name="action_space")
        spec_hash = _required(self.hand_action_spec_hash, name="hand_action_spec_hash")
        weights: dict[str, float] = {}
        for source_slot, raw_weight in dict(self.weights_by_source_slot).items():
            slot = _required(source_slot, name="source_slot_id")
            weight = _finite(raw_weight, name=f"weight[{slot}]")
            if weight < 0.0:
                raise ValueError("Hand qpos interpolation weights must be non-negative.")
            weights[slot] = weight
        if not weights or not any(value > 0.0 for value in weights.values()):
            raise ValueError("Hand qpos interpolation requires a positive weight.")
        if self.missing_candidate_behavior not in {"no_action", "renormalize_available"}:
            raise ValueError("Unsupported missing_candidate_behavior.")
        object.__setattr__(self, "strategy_id", strategy_id)
        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "component_id", component_id)
        object.__setattr__(self, "action_space", action_space)
        object.__setattr__(self, "hand_action_spec_hash", spec_hash)
        object.__setattr__(self, "weights_by_source_slot", MappingProxyType(weights))

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "strategy_id": self.strategy_id,
                "algorithm": self.algorithm,
                "component_id": self.component_id,
                "action_space": self.action_space,
                "hand_action_spec_hash": self.hand_action_spec_hash,
                "weights_by_source_slot": self.weights_by_source_slot,
                "missing_candidate_behavior": self.missing_candidate_behavior,
            },
            domain=HAND_QPOS_MIXING_SPEC_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class HandQposMixingDecision:
    decision_id: str
    strategy_id: str
    strategy_spec_hash: str
    input_snapshot_id: str
    input_snapshot_hash: str
    component_id: str
    side: str
    status: str
    reason: str
    selected_source_slot_ids: tuple[str, ...]
    selected_proposal_ids: tuple[str, ...]
    normalized_weights_by_source_slot: Mapping[str, float]
    output_hand_qpos: np.ndarray | None
    output_hand_qpos_hash: str | None
    semantic_content_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "normalized_weights_by_source_slot",
            MappingProxyType(dict(self.normalized_weights_by_source_slot)),
        )
        if self.output_hand_qpos is not None:
            object.__setattr__(self, "output_hand_qpos", deep_freeze(self.output_hand_qpos))


class HandQposMixingStrategy(Protocol):
    @property
    def spec(self) -> HandQposMixingStrategySpec: ...

    def decide(self, snapshot: HandQposMixingInputSnapshot) -> HandQposMixingDecision: ...


def _hand_qpos_hash(values: np.ndarray | None) -> str | None:
    if values is None:
        return None
    return canonical_content_hash(
        tuple(None if not np.isfinite(value) else float(value) for value in values),
        domain="dexteleop.hand_qpos_mixing_output.v1",
    )


def _build_hand_qpos_mixing_decision(
    snapshot: HandQposMixingInputSnapshot,
    *,
    spec: HandQposMixingStrategySpec,
    status: str,
    reason: str,
    selected: tuple[HandQposMixingCandidate, ...] = (),
    normalized_weights: Mapping[str, float] = MappingProxyType({}),
    output_hand_qpos: np.ndarray | None = None,
) -> HandQposMixingDecision:
    if status not in {"selected", "no_action", "invalid"}:
        raise ValueError("Unsupported hand qpos mixing decision status.")
    if snapshot.component_id != spec.component_id:
        raise ValueError("Hand qpos mixing component/spec mismatch.")
    if snapshot.hand_action_spec_hash != spec.hand_action_spec_hash:
        raise ValueError("Hand qpos mixing hand spec mismatch.")
    if status == "selected" and snapshot.action_space != spec.action_space:
        raise ValueError("Selected hand qpos mixing action space/spec mismatch.")
    if status == "selected":
        if output_hand_qpos is None or not selected:
            raise ValueError("Selected hand qpos decision requires output and candidates.")
    elif output_hand_qpos is not None:
        raise ValueError("Only selected hand qpos decisions may contain output.")
    slots = tuple(candidate.source_slot_id for candidate in selected)
    proposals = tuple(candidate.proposal.proposal_id for candidate in selected)
    if len(slots) != len(set(slots)):
        raise ValueError("Selected hand qpos source slots must be unique.")
    if any(
        snapshot.candidate_by_source_slot.get(slot) is not candidate
        for slot, candidate in (
            (item.source_slot_id, item) for item in selected
        )
    ):
        raise ValueError("Selected hand qpos candidate is not in the input snapshot.")
    output_hash = _hand_qpos_hash(output_hand_qpos)
    payload = {
        "strategy_id": spec.strategy_id,
        "strategy_spec_hash": spec.semantic_content_hash,
        "input_snapshot_id": snapshot.snapshot_id,
        "component_id": snapshot.component_id,
        "side": snapshot.side,
        "status": status,
        "reason": str(reason),
        "selected_source_slot_ids": slots,
        "selected_proposal_ids": proposals,
        "normalized_weights_by_source_slot": dict(normalized_weights),
        "output_hand_qpos_hash": output_hash,
    }
    digest = canonical_content_hash(payload, domain=HAND_QPOS_MIXING_DECISION_HASH_DOMAIN_V1)
    return HandQposMixingDecision(
        decision_id=f"hand-qpos-mixing-decision:{digest}",
        strategy_id=spec.strategy_id,
        strategy_spec_hash=spec.semantic_content_hash,
        input_snapshot_id=snapshot.snapshot_id,
        input_snapshot_hash=snapshot.semantic_content_hash,
        component_id=snapshot.component_id,
        side=snapshot.side,
        status=status,
        reason=str(reason),
        selected_source_slot_ids=slots,
        selected_proposal_ids=proposals,
        normalized_weights_by_source_slot=dict(normalized_weights),
        output_hand_qpos=output_hand_qpos,
        output_hand_qpos_hash=output_hash,
        semantic_content_hash=digest,
    )


class LinearHandQposInterpolationStrategy:
    """Pure convex interpolation for one continuous hand/gripper component."""

    def __init__(
        self,
        hand_spec: HandActionSpec,
        *,
        action_space: str,
        weights_by_source_slot: Mapping[str, float],
        missing_candidate_behavior: str = "no_action",
        component_id: str | None = None,
    ) -> None:
        if not isinstance(hand_spec, HandActionSpec):
            raise TypeError("hand_spec must be a HandActionSpec.")
        self.hand_spec = hand_spec
        self.action_space = _required(action_space, name="action_space")
        self.component_id = _required(
            component_id or f"{hand_spec.side}_hand",
            name="component_id",
        )
        self._spec = HandQposMixingStrategySpec(
            strategy_id=LINEAR_HAND_QPOS_INTERPOLATION_STRATEGY_ID_V1,
            algorithm="weighted_active_mask_linear_interpolation",
            component_id=self.component_id,
            action_space=self.action_space,
            hand_action_spec_hash=_hand_action_spec_hash(hand_spec),
            weights_by_source_slot=weights_by_source_slot,
            missing_candidate_behavior=missing_candidate_behavior,
        )

    @property
    def spec(self) -> HandQposMixingStrategySpec:
        return self._spec

    def decide(self, snapshot: HandQposMixingInputSnapshot) -> HandQposMixingDecision:
        if not snapshot.candidate_by_source_slot:
            return _build_hand_qpos_mixing_decision(
                snapshot,
                spec=self.spec,
                status="no_action",
                reason="no_candidate",
            )
        if snapshot.action_space != self.action_space:
            return _build_hand_qpos_mixing_decision(
                snapshot,
                spec=self.spec,
                status="invalid",
                reason="action_space_mismatch",
            )
        if snapshot.execution_gate_status != "ready":
            return _build_hand_qpos_mixing_decision(
                snapshot,
                spec=self.spec,
                status="no_action",
                reason=f"execution_gate_{snapshot.execution_gate_status}",
            )
        positive_weights = {
            slot: weight
            for slot, weight in self.spec.weights_by_source_slot.items()
            if weight > 0.0
        }
        missing = tuple(slot for slot in positive_weights if slot not in snapshot.candidate_by_source_slot)
        if missing and self.spec.missing_candidate_behavior == "no_action":
            return _build_hand_qpos_mixing_decision(
                snapshot,
                spec=self.spec,
                status="no_action",
                reason="missing_candidate:" + ",".join(sorted(missing)),
            )
        selected = tuple(
            snapshot.candidate_by_source_slot[slot]
            for slot in positive_weights
            if slot in snapshot.candidate_by_source_slot
        )
        if not selected:
            return _build_hand_qpos_mixing_decision(
                snapshot,
                spec=self.spec,
                status="no_action",
                reason="no_weighted_candidate",
            )
        total = sum(positive_weights[slot] for slot in (item.source_slot_id for item in selected))
        normalized = {
            item.source_slot_id: positive_weights[item.source_slot_id] / total
            for item in selected
        }
        active = sum(
            normalized[item.source_slot_id] * np.asarray(item.active_values, dtype=np.float32)
            for item in selected
        )
        output = np.full(
            (self.hand_spec.slot_count,),
            _hand_padding_value(self.hand_spec),
            dtype=np.float32,
        )
        output[np.asarray(self.hand_spec.active_mask, dtype=bool)] = active
        return _build_hand_qpos_mixing_decision(
            snapshot,
            spec=self.spec,
            status="selected",
            reason="active_mask_weighted_qpos_interpolation",
            selected=selected,
            normalized_weights=normalized,
            output_hand_qpos=output,
        )


@dataclass(frozen=True)
class ComponentMixingShadowDecision:
    """Pure arm-plus-hand component decision; never an authority decision."""

    status: str
    reason: str
    arm_decision: PoseMixingDecision
    hand_decisions_by_side: Mapping[str, HandQposMixingDecision]
    output_action: RobotAction | None
    output_action_hash: str | None
    component_decision_hash_by_id: Mapping[str, str]
    semantic_content_hash: str

    def __post_init__(self) -> None:
        if self.status not in {"selected", "no_action", "invalid"}:
            raise ValueError("Unsupported component mixing shadow status.")
        if not isinstance(self.arm_decision, PoseMixingDecision):
            raise TypeError("arm_decision must be a PoseMixingDecision.")
        decisions = {
            str(side): decision
            for side, decision in dict(self.hand_decisions_by_side).items()
        }
        if any(not isinstance(item, HandQposMixingDecision) for item in decisions.values()):
            raise TypeError("hand_decisions_by_side must contain hand decisions.")
        object.__setattr__(self, "hand_decisions_by_side", MappingProxyType(decisions))
        if self.output_action is not None:
            object.__setattr__(self, "output_action", freeze_robot_action(self.output_action))
        object.__setattr__(
            self,
            "component_decision_hash_by_id",
            MappingProxyType(dict(self.component_decision_hash_by_id)),
        )


def evaluate_component_mixing_shadow(
    arbitration_input: Any,
    *,
    source_slot_by_producer: Mapping[str, str],
    clock_domain_id: str,
    arm_strategy: PoseMixingStrategy,
    hand_strategy_by_side: Mapping[str, HandQposMixingStrategy] | None = None,
    hand_spec_by_side: Mapping[str, HandActionSpec] | None = None,
    contexts: tuple[PoseMixingContext, ...] = (),
) -> ComponentMixingShadowDecision:
    """Evaluate replaceable arm and hand strategies at one causal cut.

    This function only builds immutable snapshots, executes pure strategies and
    performs a strict structural join.  It intentionally does not admit a new
    proposal, mutate an authority/epoch, run safety/IK, enqueue a reference or
    dispatch a command.
    """

    if not callable(getattr(arm_strategy, "decide", None)):
        raise TypeError("arm_strategy must implement decide().")
    if getattr(arm_strategy, "spec", None) is None:
        raise TypeError("arm_strategy must expose a spec.")
    hand_strategies = {
        str(side): strategy
        for side, strategy in dict(hand_strategy_by_side or {}).items()
    }
    hand_specs = {
        str(side): spec for side, spec in dict(hand_spec_by_side or {}).items()
    }
    if set(hand_strategies) != set(hand_specs):
        raise ValueError("Hand strategy and hand spec sides must match exactly.")
    for side, strategy in hand_strategies.items():
        if side not in {"left", "right"}:
            raise ValueError(f"Unsupported hand side {side!r}.")
        if not callable(getattr(strategy, "decide", None)) or getattr(strategy, "spec", None) is None:
            raise TypeError(f"Hand strategy for {side!r} does not implement the protocol.")
        if not isinstance(hand_specs[side], HandActionSpec):
            raise TypeError(f"Hand spec for {side!r} must be a HandActionSpec.")

    arm_snapshot = build_pose_mixing_input_snapshot(
        arbitration_input,
        source_slot_by_producer=source_slot_by_producer,
        clock_domain_id=clock_domain_id,
        contexts=tuple(contexts),
        component_dependencies=POSE_MIXING_ARM_COMPONENTS_V1,
    )
    arm_decision = arm_strategy.decide(arm_snapshot)
    hand_decisions: dict[str, HandQposMixingDecision] = {}
    hand_snapshot_ids: dict[str, str] = {}
    for side in sorted(hand_strategies):
        hand_snapshot = build_hand_qpos_mixing_input_snapshot(
            arbitration_input,
            source_slot_by_producer=source_slot_by_producer,
            side=side,
            hand_spec=hand_specs[side],
            clock_domain_id=clock_domain_id,
        )
        hand_snapshot_ids[side] = hand_snapshot.snapshot_id
        hand_decisions[side] = hand_strategies[side].decide(hand_snapshot)

    status = arm_decision.status
    reason = "arm_strategy_" + arm_decision.reason
    output_action: RobotAction | None = None
    if status == "selected" and arm_decision.output_action is not None:
        if any(decision.status == "invalid" for decision in hand_decisions.values()):
            status = "invalid"
            reason = "hand_strategy_invalid"
        elif any(decision.status != "selected" for decision in hand_decisions.values()):
            status = "no_action"
            reason = "hand_strategy_no_action"
        else:
            output_action = assemble_compound_component_action(
                arm_decision.output_action,
                hand_qpos_by_side={
                    side: decision.output_hand_qpos
                    for side, decision in hand_decisions.items()
                    if decision.output_hand_qpos is not None
                },
            )
            reason = "arm_pose_plus_hand_qpos_component_join"
    elif hand_decisions and any(decision.status == "invalid" for decision in hand_decisions.values()):
        status = "invalid" if status == "selected" else status

    output_hash = (
        None
        if output_action is None
        else canonical_content_hash(
            robot_action_semantic_payload(output_action),
            domain="dexteleop.component_mixing_shadow_action.v1",
        )
    )
    component_hashes = {"left_arm": arm_decision.semantic_content_hash, "right_arm": arm_decision.semantic_content_hash}
    component_hashes.update(
        {
            f"{side}_hand": decision.semantic_content_hash
            for side, decision in hand_decisions.items()
        }
    )
    payload = {
        "status": status,
        "reason": reason,
        "arm_decision_id": arm_decision.decision_id,
        "arm_decision_hash": arm_decision.semantic_content_hash,
        "hand_decision_id_by_side": {
            side: decision.decision_id for side, decision in sorted(hand_decisions.items())
        },
        "hand_decision_hash_by_side": {
            side: decision.semantic_content_hash
            for side, decision in sorted(hand_decisions.items())
        },
        "arm_snapshot_id": arm_snapshot.snapshot_id,
        "hand_snapshot_id_by_side": hand_snapshot_ids,
        "output_action_hash": output_hash,
        "component_decision_hash_by_id": component_hashes,
    }
    digest = canonical_content_hash(payload, domain=COMPONENT_MIXING_SHADOW_HASH_DOMAIN_V1)
    return ComponentMixingShadowDecision(
        status=status,
        reason=reason,
        arm_decision=arm_decision,
        hand_decisions_by_side=hand_decisions,
        output_action=output_action,
        output_action_hash=output_hash,
        component_decision_hash_by_id=component_hashes,
        semantic_content_hash=digest,
    )


def assemble_compound_component_action(
    base_action: RobotAction,
    *,
    hand_qpos_by_side: Mapping[str, Any],
    expected_action_space: str | None = None,
    timestamp: float | None = None,
) -> RobotAction:
    """Attach hand/gripper qpos decisions to an arm action without authority.

    This is a strict structural join: it rejects duplicate hand ownership and
    native end-effector targets.  Canonical safety, proposal admission and
    dispatch remain outside this helper.
    """

    base = freeze_robot_action(base_action)
    if expected_action_space is not None and base.action_space != str(expected_action_space):
        raise ValueError("Compound action space does not match the expected component contract.")
    if base.end_effector_targets_by_side:
        raise ValueError("Compound assembly cannot mix hand qpos with native end-effector targets.")
    hand_values = dict(base.hand_qpos_by_side)
    overlap = set(hand_values) & set(hand_qpos_by_side)
    if overlap:
        raise ValueError(f"Compound assembly has duplicate hand ownership: {sorted(overlap)!r}.")
    for side, values in hand_qpos_by_side.items():
        normalized_side = str(side)
        if normalized_side not in {"left", "right"}:
            raise ValueError(f"Unsupported hand side {normalized_side!r}.")
        hand_values[normalized_side] = np.asarray(values, dtype=np.float32).reshape(-1)
    active = tuple(
        dict.fromkeys(
            (*base.active_effectors, *(f"{side}_hand" for side in hand_qpos_by_side))
        )
    )
    return freeze_robot_action(
        RobotAction(
            action_space=base.action_space,
            timestamp=base.timestamp if timestamp is None else float(timestamp),
            arm_pose_by_side=base.arm_pose_by_side,
            arm_qpos_by_side=base.arm_qpos_by_side,
            hand_qpos_by_side=hand_values,
            active_effectors=active,
            frame_by_effector=base.frame_by_effector,
            controlled_frame_by_effector=base.controlled_frame_by_effector,
            source="component_assembler",
            metadata=base.metadata,
        )
    )


__all__ = [
    "COMPONENT_MIXING_SHADOW_HASH_DOMAIN_V1",
    "DISCRETE_OWNER_POSE_STRATEGY_ID_V1",
    "HAND_QPOS_MIXING_DECISION_HASH_DOMAIN_V1",
    "HAND_QPOS_MIXING_SNAPSHOT_HASH_DOMAIN_V1",
    "HAND_QPOS_MIXING_SPEC_HASH_DOMAIN_V1",
    "LINEAR_HAND_QPOS_INTERPOLATION_STRATEGY_ID_V1",
    "WEIGHTED_SE3_POSE_FUSION_STRATEGY_ID_V1",
    "POSE_MIXING_CONTEXT_KINDS_V1",
    "POSE_MIXING_ACTION_SPACES_V1",
    "POSE_MIXING_ARM_COMPONENTS_V1",
    "ComponentMixingShadowDecision",
    "DiscreteOwnerPoseMixingStrategy",
    "HandQposMixingCandidate",
    "HandQposMixingDecision",
    "HandQposMixingInputSnapshot",
    "HandQposMixingStrategy",
    "HandQposMixingStrategySpec",
    "LinearHandQposInterpolationStrategy",
    "PoseMixingCandidate",
    "PoseMixingContext",
    "PoseMixingDecision",
    "PoseMixingInputSnapshot",
    "PoseMixingStrategy",
    "PoseMixingStrategySpec",
    "WeightedSE3PoseFusionStrategy",
    "build_pose_mixing_decision",
    "build_pose_mixing_input_snapshot",
    "assemble_compound_component_action",
    "build_hand_qpos_mixing_input_snapshot",
    "evaluate_component_mixing_shadow",
    "fuse_weighted_se3_pose_actions",
]
