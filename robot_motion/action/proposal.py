"""Immutable receiver-admitted action proposal contracts."""

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
from robot_motion.runtime.producer_epoch import ProducerEpochError, ProducerEpochRegistry

from .arm_spec import ArmActionSpec
from .component_action import ComponentAction
from .hand_spec import HandActionSpec
from .schema import EndEffectorTarget, RobotAction


ACTION_PROPOSAL_SEMANTIC_HASH_DOMAIN_V1 = "dexteleop.action_proposal.semantic.v1"
ACTION_PROPOSAL_SEMANTIC_HASH_DOMAIN_V2 = "dexteleop.action_proposal.semantic.v2"
ACTION_PROPOSAL_SEMANTIC_HASH_DOMAIN_V3 = "dexteleop.action_proposal.semantic.v3"
ACTION_PROPOSAL_SEMANTIC_HASH_DOMAIN_V4 = "dexteleop.action_proposal.semantic.v4"
ROBOT_ACTION_SEMANTIC_METADATA_KEYS_V1 = frozenset(
    {"arm_action_specs_by_side", "hand_action_specs_by_side"}
)


class ProposalAdmissionError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = str(code)


@dataclass(frozen=True)
class ActionProposalDraft:
    """Producer submission; deliberately has no authoritative available_at."""

    runtime_session_id: str
    control_domain_id: str
    embodiment_instance_id: str
    producer_kind: str
    producer_id: str
    producer_epoch: int
    sequence: int
    action: ComponentAction
    produced_at: float
    produced_clock_domain_id: str
    effective_from: float
    effective_until: float
    target_effective_at: float
    effective_clock_domain_id: str
    authority_priority: int = 0
    authority_request: str = "none"
    validity: str = "valid"
    status: str = "ready"
    fallback_policy: str = "no_action"
    confidence: float | None = None
    policy_node_id: str | None = None
    observation_cut_id: str | None = None
    context_ids: tuple[str, ...] = ()
    inference_id: str | None = None
    policy_output_id: str | None = None
    chunk_id: str | None = None
    chunk_keyframe_index: int | None = None
    representation_trace_id: str | None = None
    composition_input_ids: tuple[str, ...] = ()
    composition_input_hashes: tuple[str, ...] = ()
    composition_spec_hash: str | None = None
    composition_evaluation_id: str | None = None
    composition_evaluation_hash: str | None = None
    audit_metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def action_schema_hash(self) -> str:
        return self.action.action_schema.semantic_content_hash

    @property
    def component_coverage(self) -> tuple[str, ...]:
        return self.action.component_coverage

    @property
    def selection_groups(self) -> tuple[str, ...]:
        catalog = self.action.action_schema.component_catalog
        return tuple(
            dict.fromkeys(
                catalog.component_by_id[
                    component_id
                ].selection_group_id
                for component_id in self.action.component_coverage
            )
        )


@dataclass(frozen=True)
class ActionProposal:
    proposal_id: str
    runtime_session_id: str
    control_domain_id: str
    embodiment_instance_id: str
    producer_kind: str
    producer_id: str
    producer_epoch: int
    sequence: int
    action: ComponentAction
    produced_at: float
    produced_clock_domain_id: str
    available_at: float
    available_clock_domain_id: str
    control_admission_sequence: int
    effective_from: float
    effective_until: float
    target_effective_at: float
    effective_clock_domain_id: str
    action_schema_hash: str
    component_coverage: tuple[str, ...]
    selection_groups: tuple[str, ...]
    authority_priority: int
    authority_request: str
    validity: str
    status: str
    fallback_policy: str
    confidence: float | None
    policy_node_id: str | None
    observation_cut_id: str | None
    context_ids: tuple[str, ...]
    inference_id: str | None
    policy_output_id: str | None
    chunk_id: str | None
    chunk_keyframe_index: int | None
    representation_trace_id: str | None
    composition_input_ids: tuple[str, ...]
    composition_input_hashes: tuple[str, ...]
    composition_spec_hash: str | None
    composition_evaluation_id: str | None
    composition_evaluation_hash: str | None
    semantic_hash_domain: str
    semantic_content_hash: str
    audit_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ProposalAdmissionReceipt:
    accepted: bool
    rejection_code: str | None = None
    message: str = ""
    proposal_id: str | None = None
    available_at: float | None = None
    available_clock_domain_id: str | None = None
    control_admission_sequence: int | None = None
    semantic_content_hash: str | None = None
    semantic_hash_domain: str | None = None
    proposal: ActionProposal | None = None


def _required_text(value: Any, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ProposalAdmissionError("invalid_field", f"{field_name} must be non-empty.")
    return normalized


def _finite(value: Any, *, field_name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ProposalAdmissionError("invalid_time", f"{field_name} must be finite.")
    return normalized


def _freeze_end_effector_target(target: EndEffectorTarget) -> EndEffectorTarget:
    frozen = EndEffectorTarget(
        value_type=target.value_type,
        values=target.values,
        adapter=target.adapter,
        names=target.names,
        metadata=target.metadata,
    )
    object.__setattr__(frozen, "values", deep_freeze(frozen.values))
    object.__setattr__(frozen, "metadata", deep_freeze(frozen.metadata))
    return frozen


def _canonical_execution_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    canonical: dict[str, Any] = {}
    arm_payload = metadata.get("arm_action_specs_by_side")
    if arm_payload is not None:
        if not isinstance(arm_payload, Mapping):
            raise ProposalAdmissionError(
                "invalid_action_metadata", "arm_action_specs_by_side must be a mapping."
            )
        arm_specs: dict[str, Any] = {}
        for side, payload in arm_payload.items():
            try:
                spec = (
                    payload
                    if isinstance(payload, ArmActionSpec)
                    else ArmActionSpec.from_metadata(payload)
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ProposalAdmissionError(
                    "invalid_action_metadata", f"Invalid arm action spec for {side!r}: {exc}"
                ) from exc
            if str(side) != spec.side:
                raise ProposalAdmissionError(
                    "invalid_action_metadata", "Arm action spec key must match spec.side."
                )
            arm_specs[spec.side] = spec.to_metadata()
        canonical["arm_action_specs_by_side"] = arm_specs
    hand_payload = metadata.get("hand_action_specs_by_side")
    if hand_payload is not None:
        if not isinstance(hand_payload, Mapping):
            raise ProposalAdmissionError(
                "invalid_action_metadata", "hand_action_specs_by_side must be a mapping."
            )
        hand_specs: dict[str, Any] = {}
        for side, payload in hand_payload.items():
            try:
                spec = (
                    payload
                    if isinstance(payload, HandActionSpec)
                    else HandActionSpec.from_metadata(payload)
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ProposalAdmissionError(
                    "invalid_action_metadata", f"Invalid hand action spec for {side!r}: {exc}"
                ) from exc
            if str(side) != spec.side:
                raise ProposalAdmissionError(
                    "invalid_action_metadata", "Hand action spec key must match spec.side."
                )
            hand_specs[spec.side] = spec.to_metadata()
        canonical["hand_action_specs_by_side"] = hand_specs
    return canonical


def freeze_robot_action(action: RobotAction) -> RobotAction:
    """Create a detached immutable RobotAction at the proposal boundary."""

    if not isinstance(action, RobotAction):
        raise ProposalAdmissionError("invalid_action", "action must be a RobotAction.")
    unknown_metadata = set(action.metadata) - ROBOT_ACTION_SEMANTIC_METADATA_KEYS_V1
    if unknown_metadata:
        raise ProposalAdmissionError(
            "unsupported_action_metadata",
            "RobotAction metadata must be split into typed semantic fields or audit metadata; "
            f"unsupported keys: {sorted(unknown_metadata)}.",
        )
    execution_metadata = _canonical_execution_metadata(action.metadata)
    detached_targets = {
        side: EndEffectorTarget(
            value_type=target.value_type,
            values=target.values,
            adapter=target.adapter,
            names=target.names,
            metadata=target.metadata,
        )
        for side, target in action.end_effector_targets_by_side.items()
    }
    frozen = RobotAction(
        action_space=action.action_space,
        timestamp=action.timestamp,
        arm_pose_by_side=action.arm_pose_by_side,
        arm_qpos_by_side=action.arm_qpos_by_side,
        hand_qpos_by_side=action.hand_qpos_by_side,
        end_effector_targets_by_side=detached_targets,
        active_effectors=action.active_effectors,
        frame_by_effector=action.frame_by_effector,
        controlled_frame_by_effector=action.controlled_frame_by_effector,
        source=action.source,
        metadata=execution_metadata,
    )
    object.__setattr__(frozen, "arm_pose_by_side", deep_freeze(frozen.arm_pose_by_side))
    object.__setattr__(frozen, "arm_qpos_by_side", deep_freeze(frozen.arm_qpos_by_side))
    object.__setattr__(frozen, "hand_qpos_by_side", deep_freeze(frozen.hand_qpos_by_side))
    object.__setattr__(
        frozen,
        "end_effector_targets_by_side",
        MappingProxyType(
            {
                side: _freeze_end_effector_target(target)
                for side, target in frozen.end_effector_targets_by_side.items()
            }
        ),
    )
    object.__setattr__(frozen, "frame_by_effector", deep_freeze(frozen.frame_by_effector))
    object.__setattr__(
        frozen,
        "controlled_frame_by_effector",
        deep_freeze(frozen.controlled_frame_by_effector),
    )
    object.__setattr__(frozen, "metadata", deep_freeze(frozen.metadata))
    return frozen


def _target_payload(target: EndEffectorTarget) -> dict[str, Any]:
    return {
        "value_type": target.value_type,
        "values": target.values,
        "adapter": target.adapter,
        "names": target.names,
        "metadata": target.metadata,
    }


def robot_action_semantic_payload(action: RobotAction) -> dict[str, Any]:
    return {
        "action_space": action.action_space,
        "timestamp": action.timestamp,
        "arm_pose_by_side": action.arm_pose_by_side,
        "arm_qpos_by_side": action.arm_qpos_by_side,
        "hand_qpos_by_side": action.hand_qpos_by_side,
        "end_effector_targets_by_side": {
            side: _target_payload(target)
            for side, target in action.end_effector_targets_by_side.items()
        },
        "active_effectors": action.active_effectors,
        "frame_by_effector": action.frame_by_effector,
        "controlled_frame_by_effector": action.controlled_frame_by_effector,
        "source": action.source,
        "execution_metadata": action.metadata,
    }


@dataclass(frozen=True)
class _PreparedProposal:
    identity: RuntimeFactIdentity
    draft: ActionProposalDraft
    action: ComponentAction
    audit_metadata: Mapping[str, Any]
    component_coverage: tuple[str, ...]
    selection_groups: tuple[str, ...]
    produced_at: float
    produced_clock_domain_id: str
    effective_from: float
    effective_until: float
    target_effective_at: float
    effective_clock_domain_id: str


def _prepare(draft: ActionProposalDraft) -> _PreparedProposal:
    session = _required_text(draft.runtime_session_id, field_name="runtime_session_id")
    domain = _required_text(draft.control_domain_id, field_name="control_domain_id")
    producer = _required_text(draft.producer_id, field_name="producer_id")
    _required_text(draft.embodiment_instance_id, field_name="embodiment_instance_id")
    _required_text(draft.producer_kind, field_name="producer_kind")
    _required_text(draft.authority_request, field_name="authority_request")
    _required_text(draft.validity, field_name="validity")
    _required_text(draft.status, field_name="status")
    _required_text(draft.fallback_policy, field_name="fallback_policy")
    produced_clock_domain_id = _required_text(
        draft.produced_clock_domain_id,
        field_name="produced_clock_domain_id",
    )
    effective_clock_domain_id = _required_text(
        draft.effective_clock_domain_id,
        field_name="effective_clock_domain_id",
    )
    if int(draft.producer_epoch) < 0 or int(draft.sequence) < 0:
        raise ProposalAdmissionError("invalid_identity", "producer_epoch and sequence must be non-negative.")
    produced_at = _finite(draft.produced_at, field_name="produced_at")
    effective_from = _finite(draft.effective_from, field_name="effective_from")
    effective_until = _finite(draft.effective_until, field_name="effective_until")
    target_effective_at = _finite(draft.target_effective_at, field_name="target_effective_at")
    if effective_until <= effective_from:
        raise ProposalAdmissionError(
            "invalid_effective_interval", "Proposal interval must be [effective_from, effective_until)."
        )
    if not effective_from <= target_effective_at < effective_until:
        raise ProposalAdmissionError(
            "invalid_target_time", "target_effective_at must lie in [effective_from, effective_until)."
        )
    if draft.chunk_keyframe_index is not None and int(draft.chunk_keyframe_index) < 0:
        raise ProposalAdmissionError("invalid_field", "chunk_keyframe_index must be non-negative.")
    if draft.confidence is not None:
        confidence = float(draft.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ProposalAdmissionError("invalid_field", "confidence must be finite in [0, 1].")
    for field_name in (
        "policy_node_id",
        "observation_cut_id",
        "inference_id",
        "policy_output_id",
        "chunk_id",
        "representation_trace_id",
        "composition_evaluation_id",
        "composition_evaluation_hash",
    ):
        value = getattr(draft, field_name)
        if value is not None:
            _required_text(value, field_name=field_name)
    if any(not str(value).strip() for value in draft.context_ids):
        raise ProposalAdmissionError("invalid_field", "context_ids must be non-empty strings.")
    composition_input_ids = tuple(str(value).strip() for value in draft.composition_input_ids)
    composition_input_hashes = tuple(
        str(value).strip() for value in draft.composition_input_hashes
    )
    if (
        any(not value for value in composition_input_ids)
        or any(not value for value in composition_input_hashes)
        or len(composition_input_ids) != len(composition_input_hashes)
        or len(composition_input_ids) != len(set(composition_input_ids))
    ):
        raise ProposalAdmissionError(
            "invalid_composition_lineage",
            "Composition input IDs/hashes must be non-empty, unique, and paired.",
        )
    if bool(composition_input_ids) != bool(draft.composition_spec_hash):
        raise ProposalAdmissionError(
            "invalid_composition_lineage",
            "Composition inputs and composition_spec_hash must appear together.",
        )
    if draft.composition_spec_hash is not None:
        _required_text(draft.composition_spec_hash, field_name="composition_spec_hash")
    if (draft.composition_evaluation_id is None) != (
        draft.composition_evaluation_hash is None
    ):
        raise ProposalAdmissionError(
            "invalid_composition_lineage",
            "Composition evaluation ID/hash must appear together.",
        )
    if draft.producer_kind == "composer" and (
        draft.composition_evaluation_id is None
        or not composition_input_ids
    ):
        raise ProposalAdmissionError(
            "invalid_composition_lineage",
            "Composer proposals require exact evaluation and input lineage.",
        )

    if not isinstance(draft.action, ComponentAction):
        raise ProposalAdmissionError(
            "invalid_action",
            "ActionProposalDraft.action must be a ComponentAction.",
        )
    action = draft.action
    coverage = action.component_coverage
    catalog = action.action_schema.component_catalog
    groups = tuple(
        dict.fromkeys(
            catalog.component_by_id[component_id].selection_group_id
            for component_id in coverage
        )
    )
    identity = RuntimeFactIdentity(
        runtime_session_id=session,
        control_domain_id=domain,
        producer_id=producer,
        producer_epoch=int(draft.producer_epoch),
        sequence=int(draft.sequence),
        namespace="action_proposal",
    )
    return _PreparedProposal(
        identity=identity,
        draft=draft,
        action=action,
        audit_metadata=deep_freeze(dict(draft.audit_metadata)),
        component_coverage=coverage,
        selection_groups=groups,
        produced_at=produced_at,
        produced_clock_domain_id=produced_clock_domain_id,
        effective_from=effective_from,
        effective_until=effective_until,
        target_effective_at=target_effective_at,
        effective_clock_domain_id=effective_clock_domain_id,
    )


def _proposal_semantic_payload(
    prepared: _PreparedProposal,
    *,
    available_at: float,
    control_admission_sequence: int,
) -> dict[str, Any]:
    draft = prepared.draft
    payload = {
        "proposal_id": prepared.identity.fact_id,
        "runtime_session_id": prepared.identity.runtime_session_id,
        "control_domain_id": prepared.identity.control_domain_id,
        "embodiment_instance_id": draft.embodiment_instance_id,
        "producer_kind": draft.producer_kind,
        "producer_id": prepared.identity.producer_id,
        "producer_epoch": prepared.identity.producer_epoch,
        "sequence": prepared.identity.sequence,
        "action": prepared.action.semantic_payload(),
        "produced_at": prepared.produced_at,
        "produced_clock_domain_id": prepared.produced_clock_domain_id,
        "available_at": available_at,
        "available_clock_domain_id": prepared.effective_clock_domain_id,
        "control_admission_sequence": control_admission_sequence,
        "effective_from": prepared.effective_from,
        "effective_until": prepared.effective_until,
        "target_effective_at": prepared.target_effective_at,
        "effective_clock_domain_id": prepared.effective_clock_domain_id,
        "action_schema_hash": (
            prepared.action.action_schema.semantic_content_hash
        ),
        "component_coverage": prepared.component_coverage,
        "selection_groups": prepared.selection_groups,
        "authority_priority": int(draft.authority_priority),
        "authority_request": draft.authority_request,
        "validity": draft.validity,
        "status": draft.status,
        "fallback_policy": draft.fallback_policy,
        "confidence": draft.confidence,
        "policy_node_id": draft.policy_node_id,
        "observation_cut_id": draft.observation_cut_id,
        "context_ids": tuple(draft.context_ids),
        "inference_id": draft.inference_id,
        "policy_output_id": draft.policy_output_id,
        "chunk_id": draft.chunk_id,
        "chunk_keyframe_index": draft.chunk_keyframe_index,
        "representation_trace_id": draft.representation_trace_id,
    }
    if draft.composition_input_ids:
        payload.update(
            {
                "composition_input_ids": tuple(draft.composition_input_ids),
                "composition_input_hashes": tuple(draft.composition_input_hashes),
                "composition_spec_hash": draft.composition_spec_hash,
                "composition_evaluation_id": draft.composition_evaluation_id,
                "composition_evaluation_hash": draft.composition_evaluation_hash,
            }
        )
    return payload


def action_proposal_semantic_payload(proposal: ActionProposal) -> dict[str, Any]:
    """Public primitive projection used by runtime/dataset conformance checks."""

    payload = {
        "proposal_id": proposal.proposal_id,
        "runtime_session_id": proposal.runtime_session_id,
        "control_domain_id": proposal.control_domain_id,
        "embodiment_instance_id": proposal.embodiment_instance_id,
        "producer_kind": proposal.producer_kind,
        "producer_id": proposal.producer_id,
        "producer_epoch": proposal.producer_epoch,
        "sequence": proposal.sequence,
        "action": proposal.action.semantic_payload(),
        "produced_at": proposal.produced_at,
        "produced_clock_domain_id": proposal.produced_clock_domain_id,
        "available_at": proposal.available_at,
        "available_clock_domain_id": proposal.available_clock_domain_id,
        "control_admission_sequence": proposal.control_admission_sequence,
        "effective_from": proposal.effective_from,
        "effective_until": proposal.effective_until,
        "target_effective_at": proposal.target_effective_at,
        "effective_clock_domain_id": proposal.effective_clock_domain_id,
        "action_schema_hash": proposal.action_schema_hash,
        "component_coverage": proposal.component_coverage,
        "selection_groups": proposal.selection_groups,
        "authority_priority": proposal.authority_priority,
        "authority_request": proposal.authority_request,
        "validity": proposal.validity,
        "status": proposal.status,
        "fallback_policy": proposal.fallback_policy,
        "confidence": proposal.confidence,
        "policy_node_id": proposal.policy_node_id,
        "observation_cut_id": proposal.observation_cut_id,
        "context_ids": proposal.context_ids,
        "inference_id": proposal.inference_id,
        "policy_output_id": proposal.policy_output_id,
        "chunk_id": proposal.chunk_id,
        "chunk_keyframe_index": proposal.chunk_keyframe_index,
        "representation_trace_id": proposal.representation_trace_id,
    }
    if proposal.composition_input_ids:
        payload.update(
            {
                "composition_input_ids": proposal.composition_input_ids,
                "composition_input_hashes": proposal.composition_input_hashes,
                "composition_spec_hash": proposal.composition_spec_hash,
                "composition_evaluation_id": proposal.composition_evaluation_id,
                "composition_evaluation_hash": proposal.composition_evaluation_hash,
            }
        )
    return payload


class ProposalBuffer:
    """Bounded fail-closed store of admitted proposals."""

    def __init__(
        self,
        *,
        runtime_session_id: str,
        coordinator: ControlAdmissionCoordinator,
        producer_epochs: ProducerEpochRegistry,
        capacity: int,
    ) -> None:
        if int(capacity) <= 0:
            raise ValueError("ProposalBuffer capacity must be positive.")
        if producer_epochs.coordinator is not coordinator:
            raise ValueError("ProposalBuffer and ProducerEpochRegistry must share one coordinator.")
        self.runtime_session_id = str(runtime_session_id).strip()
        if self.runtime_session_id != coordinator.clock_domain.runtime_session_id:
            raise ValueError("ProposalBuffer session must match the control clock session.")
        self.coordinator = coordinator
        self.control_domain_id = coordinator.control_domain_id
        self.producer_epochs = producer_epochs
        self.capacity = int(capacity)
        self._records: list[ActionProposal] = []
        self._by_id: dict[str, ActionProposal] = {}
        self._last_sequence: dict[tuple[str, int], int] = {}
        self._lock = threading.RLock()

    def admit(self, draft: ActionProposalDraft) -> ProposalAdmissionReceipt:
        proposal_id: str | None = None
        try:
            prepared = _prepare(draft)
            proposal_id = prepared.identity.fact_id
            if prepared.identity.runtime_session_id != self.runtime_session_id:
                raise ProposalAdmissionError("session_mismatch", "Draft belongs to another runtime session.")
            if prepared.identity.control_domain_id != self.control_domain_id:
                raise ProposalAdmissionError("domain_mismatch", "Draft belongs to another control domain.")
            if prepared.effective_clock_domain_id != self.coordinator.clock_domain.clock_domain_id:
                raise ProposalAdmissionError(
                    "clock_domain_mismatch",
                    "Proposal effective interval must use the receiver control clock domain.",
                )

            def build(available_at: float, admission_sequence: int) -> ActionProposal:
                semantic_hash_domain = (
                    ACTION_PROPOSAL_SEMANTIC_HASH_DOMAIN_V4
                    if prepared.draft.composition_evaluation_id is not None
                    else ACTION_PROPOSAL_SEMANTIC_HASH_DOMAIN_V3
                )
                semantic_hash = canonical_content_hash(
                    _proposal_semantic_payload(
                        prepared,
                        available_at=available_at,
                        control_admission_sequence=admission_sequence,
                    ),
                    domain=semantic_hash_domain,
                )
                source = prepared.draft
                return ActionProposal(
                        proposal_id=prepared.identity.fact_id,
                        runtime_session_id=prepared.identity.runtime_session_id,
                        control_domain_id=prepared.identity.control_domain_id,
                        embodiment_instance_id=str(source.embodiment_instance_id),
                        producer_kind=str(source.producer_kind),
                        producer_id=prepared.identity.producer_id,
                        producer_epoch=prepared.identity.producer_epoch,
                        sequence=prepared.identity.sequence,
                        action=prepared.action,
                        produced_at=prepared.produced_at,
                        produced_clock_domain_id=prepared.produced_clock_domain_id,
                        available_at=available_at,
                        available_clock_domain_id=self.coordinator.clock_domain.clock_domain_id,
                        control_admission_sequence=admission_sequence,
                        effective_from=prepared.effective_from,
                        effective_until=prepared.effective_until,
                        target_effective_at=prepared.target_effective_at,
                        effective_clock_domain_id=prepared.effective_clock_domain_id,
                        action_schema_hash=(
                            prepared.action.action_schema.semantic_content_hash
                        ),
                        component_coverage=prepared.component_coverage,
                        selection_groups=prepared.selection_groups,
                        authority_priority=int(source.authority_priority),
                        authority_request=str(source.authority_request),
                        validity=str(source.validity),
                        status=str(source.status),
                        fallback_policy=str(source.fallback_policy),
                        confidence=None if source.confidence is None else float(source.confidence),
                        policy_node_id=None if source.policy_node_id is None else str(source.policy_node_id),
                        observation_cut_id=(
                            None if source.observation_cut_id is None else str(source.observation_cut_id)
                        ),
                        context_ids=tuple(str(value) for value in source.context_ids),
                        inference_id=None if source.inference_id is None else str(source.inference_id),
                        policy_output_id=(
                            None if source.policy_output_id is None else str(source.policy_output_id)
                        ),
                        chunk_id=None if source.chunk_id is None else str(source.chunk_id),
                        chunk_keyframe_index=(
                            None
                            if source.chunk_keyframe_index is None
                            else int(source.chunk_keyframe_index)
                        ),
                        representation_trace_id=(
                            None
                            if source.representation_trace_id is None
                            else str(source.representation_trace_id)
                        ),
                        composition_input_ids=tuple(
                            str(value) for value in source.composition_input_ids
                        ),
                        composition_input_hashes=tuple(
                            str(value) for value in source.composition_input_hashes
                        ),
                        composition_spec_hash=(
                            None
                            if source.composition_spec_hash is None
                            else str(source.composition_spec_hash)
                        ),
                        composition_evaluation_id=(
                            None
                            if source.composition_evaluation_id is None
                            else str(source.composition_evaluation_id)
                        ),
                        composition_evaluation_hash=(
                            None
                            if source.composition_evaluation_hash is None
                            else str(source.composition_evaluation_hash)
                        ),
                        semantic_hash_domain=semantic_hash_domain,
                        semantic_content_hash=semantic_hash,
                        audit_metadata=prepared.audit_metadata,
                    )

            proposal = self.coordinator.commit(store=self, build=build)
            return ProposalAdmissionReceipt(
                accepted=True,
                proposal_id=proposal.proposal_id,
                available_at=proposal.available_at,
                available_clock_domain_id=proposal.available_clock_domain_id,
                control_admission_sequence=proposal.control_admission_sequence,
                semantic_content_hash=proposal.semantic_content_hash,
                semantic_hash_domain=proposal.semantic_hash_domain,
                proposal=proposal,
            )
        except (ProposalAdmissionError, ProducerEpochError, TypeError, ValueError) as exc:
            code = exc.code if isinstance(exc, ProposalAdmissionError) else "validation_failed"
            return ProposalAdmissionReceipt(
                accepted=False,
                rejection_code=code,
                message=str(exc),
                proposal_id=proposal_id,
            )

    def _insert_admitted(self, proposal: ActionProposal) -> None:
        with self._lock:
            try:
                self.producer_epochs.assert_active(
                    producer_id=proposal.producer_id,
                    producer_epoch=proposal.producer_epoch,
                )
            except ProducerEpochError as exc:
                raise ProposalAdmissionError("inactive_producer_epoch", str(exc)) from exc
            if proposal.proposal_id in self._by_id:
                raise ProposalAdmissionError("duplicate", "Proposal ID is already admitted.")
            sequence_key = (proposal.producer_id, proposal.producer_epoch)
            previous_sequence = self._last_sequence.get(sequence_key)
            if previous_sequence is not None and proposal.sequence <= previous_sequence:
                raise ProposalAdmissionError(
                    "sequence_rollback",
                    f"Producer sequence must increase beyond {previous_sequence}.",
                )
            if len(self._records) >= self.capacity:
                raise ProposalAdmissionError("capacity_exhausted", "ProposalBuffer is full.")
            self._records.append(proposal)
            self._by_id[proposal.proposal_id] = proposal
            self._last_sequence[sequence_key] = proposal.sequence

    def records_at_or_before(self, cut: ControlCut) -> tuple[ActionProposal, ...]:
        if cut.control_domain_id != self.control_domain_id:
            raise ValueError("Control cut belongs to another domain.")
        with self._lock:
            return tuple(record for record in self._records if admitted_at_or_before(record, cut))

    def get(self, proposal_id: str) -> ActionProposal | None:
        """Return one immutable admitted fact without changing buffer state."""

        with self._lock:
            return self._by_id.get(str(proposal_id))

    def records_for_active_epochs(self, cut: ControlCut) -> tuple[ActionProposal, ...]:
        active = self.producer_epochs.snapshot(cut).active_epoch_by_producer
        return tuple(
            proposal
            for proposal in self.records_at_or_before(cut)
            if active.get(proposal.producer_id) == proposal.producer_epoch
        )

    def prune_expired(self, *, at_or_before: float) -> tuple[str, ...]:
        """Drop proposals that can never be eligible at a later target time.

        Sequence high-water marks are intentionally retained, so pruning does
        not permit a producer to reuse an old identity. Already-created
        immutable snapshots keep their proposal objects independently.
        """

        cutoff = float(at_or_before)
        if not math.isfinite(cutoff):
            raise ValueError("Proposal prune cutoff must be finite.")
        with self._lock:
            expired = tuple(
                record.proposal_id
                for record in self._records
                if record.effective_until <= cutoff
            )
            if not expired:
                return ()
            expired_set = set(expired)
            self._records = [
                record
                for record in self._records
                if record.proposal_id not in expired_set
            ]
            for proposal_id in expired:
                self._by_id.pop(proposal_id, None)
            return expired

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


__all__ = [
    "ACTION_PROPOSAL_SEMANTIC_HASH_DOMAIN_V1",
    "ACTION_PROPOSAL_SEMANTIC_HASH_DOMAIN_V2",
    "ACTION_PROPOSAL_SEMANTIC_HASH_DOMAIN_V3",
    "ACTION_PROPOSAL_SEMANTIC_HASH_DOMAIN_V4",
    "ROBOT_ACTION_SEMANTIC_METADATA_KEYS_V1",
    "ActionProposal",
    "ActionProposalDraft",
    "ProposalAdmissionError",
    "ProposalAdmissionReceipt",
    "ProposalBuffer",
    "action_proposal_semantic_payload",
    "freeze_robot_action",
    "robot_action_semantic_payload",
]
