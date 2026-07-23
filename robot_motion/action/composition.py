"""Deterministic pre-arbitration action composition contracts.

Composition inputs are internal admitted facts, not authority candidates.  A
composer may publish exactly one new ``ActionProposalDraft`` carrying complete
typed input lineage; raw inputs never enter the global ProposalBuffer unless a
producer explicitly publishes a separate proposal fact.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.runtime.control_admission import (
    ControlAdmissionCoordinator,
    ControlCut,
    admitted_at_or_before,
)
from robot_motion.runtime.identity import RuntimeFactIdentity
from robot_motion.runtime.immutable import deep_freeze
from robot_motion.runtime.producer_epoch import (
    ProducerEpochError,
    ProducerEpochRegistry,
    ProducerEpochSnapshot,
)

from .mixing_strategy import fuse_weighted_se3_pose_actions
from .proposal import ActionProposalDraft, freeze_robot_action, robot_action_semantic_payload
from .schema import RobotAction, split_effector_name


COMPOSITION_INPUT_HASH_DOMAIN_V1 = "dexteleop.composition_input.semantic.v1"
COMPOSITION_INPUT_SNAPSHOT_HASH_DOMAIN_V1 = (
    "dexteleop.composition_input_snapshot.v1"
)
COMPOSITION_SPEC_HASH_DOMAIN_V1 = "dexteleop.composition_spec.v1"
COMPOSITION_RESULT_HASH_DOMAIN_V1 = "dexteleop.composition_result.v1"
COMPOSITION_ALGEBRAS_V1 = frozenset(
    {
        "component_union",
        "base_plus_bounded_residual",
        "weighted_se3_pose_fusion",
    }
)


class CompositionAdmissionError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = str(code)


def _required(value: object, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise CompositionAdmissionError(
            "invalid_field", f"{field_name} must be non-empty."
        )
    return normalized


def _finite(value: object, *, field_name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise CompositionAdmissionError(
            "invalid_time", f"{field_name} must be finite."
        )
    return normalized


@dataclass(frozen=True)
class CompositionInputDraft:
    runtime_session_id: str
    control_domain_id: str
    producer_id: str
    producer_epoch: int
    sequence: int
    input_role: str
    action: RobotAction
    produced_at: float
    produced_clock_domain_id: str
    effective_from: float
    effective_until: float
    effective_clock_domain_id: str
    context_ids: tuple[str, ...] = ()
    observation_cut_id: str | None = None
    policy_output_id: str | None = None
    authority_candidate: bool = False
    audit_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompositionInputRecord:
    composition_input_id: str
    runtime_session_id: str
    control_domain_id: str
    producer_id: str
    producer_epoch: int
    sequence: int
    input_role: str
    action: RobotAction
    produced_at: float
    produced_clock_domain_id: str
    available_at: float
    available_clock_domain_id: str
    control_admission_sequence: int
    effective_from: float
    effective_until: float
    effective_clock_domain_id: str
    context_ids: tuple[str, ...]
    observation_cut_id: str | None
    policy_output_id: str | None
    authority_candidate: bool
    semantic_content_hash: str
    audit_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class CompositionInputAdmissionReceipt:
    accepted: bool
    rejection_code: str | None = None
    message: str = ""
    composition_input_id: str | None = None
    available_at: float | None = None
    available_clock_domain_id: str | None = None
    control_admission_sequence: int | None = None
    semantic_content_hash: str | None = None
    composition_input: CompositionInputRecord | None = None


@dataclass(frozen=True)
class CompositionSpec:
    spec_id: str
    algebra: str
    input_roles: tuple[str, ...]
    required_roles: tuple[str, ...]
    missing_required_behavior: str = "no_action"
    missing_residual_behavior: str = "zero_residual"
    residual_limit_by_component: Mapping[str, float] = field(default_factory=dict)
    weight_by_role: Mapping[str, float] = field(default_factory=dict)
    max_translation_disagreement_m: float | None = None
    max_rotation_disagreement_rad: float | None = None

    def __post_init__(self) -> None:
        _required(self.spec_id, field_name="spec_id")
        algebra = _required(self.algebra, field_name="algebra")
        if algebra not in COMPOSITION_ALGEBRAS_V1:
            raise ValueError(f"Unsupported composition algebra {algebra!r}.")
        roles = tuple(_required(value, field_name="input_role") for value in self.input_roles)
        required = tuple(
            _required(value, field_name="required_role") for value in self.required_roles
        )
        if not roles or len(roles) != len(set(roles)):
            raise ValueError("Composition input_roles must be non-empty and unique.")
        if not set(required).issubset(roles):
            raise ValueError("required_roles must be a subset of input_roles.")
        if self.missing_required_behavior not in {"no_action", "invalid"}:
            raise ValueError("Unsupported missing_required_behavior.")
        if self.missing_residual_behavior not in {
            "zero_residual",
            "no_action",
            "invalid",
        }:
            raise ValueError("Unsupported missing_residual_behavior.")
        limits: dict[str, float] = {}
        for component, raw in dict(self.residual_limit_by_component).items():
            key = _required(component, field_name="residual component")
            limit = float(raw)
            if not math.isfinite(limit) or limit <= 0.0:
                raise ValueError("Residual limits must be finite and positive.")
            limits[key] = limit
        if algebra == "base_plus_bounded_residual":
            if "base" not in roles or "residual" not in roles:
                raise ValueError("base_plus_bounded_residual requires base and residual roles.")
            if not limits:
                raise ValueError("base_plus_bounded_residual requires component limits.")
        weights: dict[str, float] = {}
        for role, raw in dict(self.weight_by_role).items():
            key = _required(role, field_name="weighted role")
            weight = float(raw)
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(
                    "Composition role weights must be finite and non-negative."
                )
            weights[key] = weight
        translation_limit = self.max_translation_disagreement_m
        rotation_limit = self.max_rotation_disagreement_rad
        if algebra == "weighted_se3_pose_fusion":
            if set(weights) != set(roles) or not any(
                weight > 0.0 for weight in weights.values()
            ):
                raise ValueError(
                    "weighted_se3_pose_fusion requires one role weight per input "
                    "and at least one positive weight."
                )
            if translation_limit is None or rotation_limit is None:
                raise ValueError(
                    "weighted_se3_pose_fusion requires explicit disagreement limits."
                )
            translation_limit = float(translation_limit)
            rotation_limit = float(rotation_limit)
            if not math.isfinite(translation_limit) or translation_limit < 0.0:
                raise ValueError(
                    "max_translation_disagreement_m must be finite and non-negative."
                )
            if (
                not math.isfinite(rotation_limit)
                or rotation_limit < 0.0
                or rotation_limit >= math.pi
            ):
                raise ValueError(
                    "max_rotation_disagreement_rad must be in [0, pi)."
                )
            if limits:
                raise ValueError(
                    "weighted_se3_pose_fusion does not use residual limits."
                )
        elif weights or translation_limit is not None or rotation_limit is not None:
            raise ValueError(
                "Pose fusion parameters require weighted_se3_pose_fusion."
            )
        object.__setattr__(self, "algebra", algebra)
        object.__setattr__(self, "input_roles", roles)
        object.__setattr__(self, "required_roles", required)
        object.__setattr__(self, "residual_limit_by_component", MappingProxyType(limits))
        object.__setattr__(self, "weight_by_role", MappingProxyType(weights))
        object.__setattr__(
            self,
            "max_translation_disagreement_m",
            translation_limit,
        )
        object.__setattr__(
            self,
            "max_rotation_disagreement_rad",
            rotation_limit,
        )

    @property
    def semantic_content_hash(self) -> str:
        payload = {
            "spec_id": self.spec_id,
            "algebra": self.algebra,
            "input_roles": self.input_roles,
            "required_roles": self.required_roles,
            "missing_required_behavior": self.missing_required_behavior,
            "missing_residual_behavior": self.missing_residual_behavior,
            "residual_limit_by_component": self.residual_limit_by_component,
        }
        if self.algebra == "weighted_se3_pose_fusion":
            payload.update(
                {
                    "weight_by_role": self.weight_by_role,
                    "max_translation_disagreement_m": (
                        self.max_translation_disagreement_m
                    ),
                    "max_rotation_disagreement_rad": (
                        self.max_rotation_disagreement_rad
                    ),
                    "rotation_fusion": (
                        "deterministic_reference_shortest_path_so3_tangent_mean_v1"
                    ),
                }
            )
        return canonical_content_hash(
            payload,
            domain=COMPOSITION_SPEC_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class CompositionInputProjection:
    control_domain_id: str
    cutoff_at: float
    cutoff_admission_sequence: int
    records: tuple[CompositionInputRecord, ...]


@dataclass(frozen=True)
class CompositionInputSnapshot:
    composition_input_snapshot_id: str
    control_domain_id: str
    cutoff_at: float
    cutoff_admission_sequence: int
    target_effective_at: float
    composition_spec_hash: str
    selected_input_by_role: Mapping[str, CompositionInputRecord]
    missing_roles: tuple[str, ...]
    producer_epoch_snapshot_hash: str
    semantic_content_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selected_input_by_role",
            MappingProxyType(dict(self.selected_input_by_role)),
        )


@dataclass(frozen=True)
class CompositionResult:
    composition_result_id: str
    composition_input_snapshot_id: str
    composition_input_snapshot_hash: str
    composition_spec_hash: str
    output_status: str
    reason: str
    composed_action: RobotAction | None
    composed_action_hash: str | None
    input_ids: tuple[str, ...]
    input_hashes: tuple[str, ...]
    context_ids: tuple[str, ...]
    semantic_content_hash: str


@dataclass(frozen=True)
class _PreparedInput:
    identity: RuntimeFactIdentity
    draft: CompositionInputDraft
    action: RobotAction
    produced_at: float
    effective_from: float
    effective_until: float
    context_ids: tuple[str, ...]
    audit_metadata: Mapping[str, Any]


def _prepare(draft: CompositionInputDraft) -> _PreparedInput:
    session = _required(draft.runtime_session_id, field_name="runtime_session_id")
    domain = _required(draft.control_domain_id, field_name="control_domain_id")
    producer = _required(draft.producer_id, field_name="producer_id")
    _required(draft.input_role, field_name="input_role")
    _required(draft.produced_clock_domain_id, field_name="produced_clock_domain_id")
    _required(draft.effective_clock_domain_id, field_name="effective_clock_domain_id")
    if draft.authority_candidate:
        raise CompositionAdmissionError(
            "authority_candidate_forbidden",
            "Internal composition inputs cannot be authority candidates.",
        )
    if int(draft.producer_epoch) < 0 or int(draft.sequence) < 0:
        raise CompositionAdmissionError(
            "invalid_identity", "producer_epoch and sequence must be non-negative."
        )
    produced_at = _finite(draft.produced_at, field_name="produced_at")
    effective_from = _finite(draft.effective_from, field_name="effective_from")
    effective_until = _finite(draft.effective_until, field_name="effective_until")
    if effective_until <= effective_from:
        raise CompositionAdmissionError(
            "invalid_effective_interval",
            "Composition input interval must be [effective_from, effective_until).",
        )
    for field_name in ("observation_cut_id", "policy_output_id"):
        value = getattr(draft, field_name)
        if value is not None:
            _required(value, field_name=field_name)
    context_ids = tuple(_required(value, field_name="context_id") for value in draft.context_ids)
    if len(context_ids) != len(set(context_ids)):
        raise CompositionAdmissionError("invalid_field", "context_ids must be unique.")
    return _PreparedInput(
        identity=RuntimeFactIdentity(
            runtime_session_id=session,
            control_domain_id=domain,
            producer_id=producer,
            producer_epoch=int(draft.producer_epoch),
            sequence=int(draft.sequence),
            namespace="composition_input",
        ),
        draft=draft,
        action=freeze_robot_action(draft.action),
        produced_at=produced_at,
        effective_from=effective_from,
        effective_until=effective_until,
        context_ids=context_ids,
        audit_metadata=deep_freeze(dict(draft.audit_metadata)),
    )


def composition_input_semantic_payload(record: CompositionInputRecord) -> dict[str, Any]:
    return {
        "composition_input_id": record.composition_input_id,
        "runtime_session_id": record.runtime_session_id,
        "control_domain_id": record.control_domain_id,
        "producer_id": record.producer_id,
        "producer_epoch": record.producer_epoch,
        "sequence": record.sequence,
        "input_role": record.input_role,
        "action": robot_action_semantic_payload(record.action),
        "produced_at": record.produced_at,
        "produced_clock_domain_id": record.produced_clock_domain_id,
        "available_at": record.available_at,
        "available_clock_domain_id": record.available_clock_domain_id,
        "control_admission_sequence": record.control_admission_sequence,
        "effective_from": record.effective_from,
        "effective_until": record.effective_until,
        "effective_clock_domain_id": record.effective_clock_domain_id,
        "context_ids": record.context_ids,
        "observation_cut_id": record.observation_cut_id,
        "policy_output_id": record.policy_output_id,
        "authority_candidate": record.authority_candidate,
    }


class CompositionInputStore:
    def __init__(
        self,
        *,
        runtime_session_id: str,
        coordinator: ControlAdmissionCoordinator,
        producer_epochs: ProducerEpochRegistry,
        capacity: int,
    ) -> None:
        if int(capacity) <= 0:
            raise ValueError("CompositionInputStore capacity must be positive.")
        if producer_epochs.coordinator is not coordinator:
            raise ValueError("Composition store and epochs must share one coordinator.")
        self.runtime_session_id = _required(runtime_session_id, field_name="runtime_session_id")
        if self.runtime_session_id != coordinator.clock_domain.runtime_session_id:
            raise ValueError("Composition store session must match the control clock session.")
        self.coordinator = coordinator
        self.control_domain_id = coordinator.control_domain_id
        self.producer_epochs = producer_epochs
        self.capacity = int(capacity)
        self._records: list[CompositionInputRecord] = []
        self._by_id: dict[str, CompositionInputRecord] = {}
        self._last_sequence: dict[tuple[str, int], int] = {}
        self._lock = threading.RLock()

    def admit(self, draft: CompositionInputDraft) -> CompositionInputAdmissionReceipt:
        input_id: str | None = None
        try:
            prepared = _prepare(draft)
            input_id = prepared.identity.fact_id
            if prepared.identity.runtime_session_id != self.runtime_session_id:
                raise CompositionAdmissionError("session_mismatch", "Draft belongs to another session.")
            if prepared.identity.control_domain_id != self.control_domain_id:
                raise CompositionAdmissionError("domain_mismatch", "Draft belongs to another domain.")
            if draft.effective_clock_domain_id != self.coordinator.clock_domain.clock_domain_id:
                raise CompositionAdmissionError(
                    "clock_domain_mismatch",
                    "Composition input validity must use the receiver control clock.",
                )

            def build(available_at: float, admission_sequence: int) -> CompositionInputRecord:
                payload = {
                    "composition_input_id": prepared.identity.fact_id,
                    "runtime_session_id": prepared.identity.runtime_session_id,
                    "control_domain_id": prepared.identity.control_domain_id,
                    "producer_id": prepared.identity.producer_id,
                    "producer_epoch": prepared.identity.producer_epoch,
                    "sequence": prepared.identity.sequence,
                    "input_role": draft.input_role,
                    "action": prepared.action,
                    "produced_at": prepared.produced_at,
                    "produced_clock_domain_id": draft.produced_clock_domain_id,
                    "available_at": available_at,
                    "available_clock_domain_id": self.coordinator.clock_domain.clock_domain_id,
                    "control_admission_sequence": admission_sequence,
                    "effective_from": prepared.effective_from,
                    "effective_until": prepared.effective_until,
                    "effective_clock_domain_id": draft.effective_clock_domain_id,
                    "context_ids": prepared.context_ids,
                    "observation_cut_id": draft.observation_cut_id,
                    "policy_output_id": draft.policy_output_id,
                    "authority_candidate": False,
                }
                semantic_payload = dict(payload)
                semantic_payload["action"] = robot_action_semantic_payload(prepared.action)
                return CompositionInputRecord(
                    **payload,
                    semantic_content_hash=canonical_content_hash(
                        semantic_payload, domain=COMPOSITION_INPUT_HASH_DOMAIN_V1
                    ),
                    audit_metadata=prepared.audit_metadata,
                )

            record = self.coordinator.commit(store=self, build=build)
            return CompositionInputAdmissionReceipt(
                accepted=True,
                composition_input_id=record.composition_input_id,
                available_at=record.available_at,
                available_clock_domain_id=record.available_clock_domain_id,
                control_admission_sequence=record.control_admission_sequence,
                semantic_content_hash=record.semantic_content_hash,
                composition_input=record,
            )
        except (CompositionAdmissionError, ProducerEpochError, TypeError, ValueError) as exc:
            code = exc.code if isinstance(exc, CompositionAdmissionError) else "validation_failed"
            return CompositionInputAdmissionReceipt(
                accepted=False,
                rejection_code=code,
                message=str(exc),
                composition_input_id=input_id,
            )

    def _insert_admitted(self, record: CompositionInputRecord) -> None:
        with self._lock:
            try:
                self.producer_epochs.assert_active(
                    producer_id=record.producer_id,
                    producer_epoch=record.producer_epoch,
                )
            except ProducerEpochError as exc:
                raise CompositionAdmissionError("inactive_producer_epoch", str(exc)) from exc
            if record.composition_input_id in self._by_id:
                raise CompositionAdmissionError("duplicate", "Composition input is already admitted.")
            key = (record.producer_id, record.producer_epoch)
            previous = self._last_sequence.get(key)
            if previous is not None and record.sequence <= previous:
                raise CompositionAdmissionError(
                    "sequence_rollback", f"Producer sequence must increase beyond {previous}."
                )
            if len(self._records) >= self.capacity:
                raise CompositionAdmissionError("capacity_exhausted", "CompositionInputStore is full.")
            self._records.append(record)
            self._by_id[record.composition_input_id] = record
            self._last_sequence[key] = record.sequence

    def records_at_or_before(self, cut: ControlCut) -> tuple[CompositionInputRecord, ...]:
        if cut.control_domain_id != self.control_domain_id:
            raise ValueError("Control cut belongs to another domain.")
        with self._lock:
            return tuple(record for record in self._records if admitted_at_or_before(record, cut))

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


def project_composition_inputs(
    store: CompositionInputStore, cut: ControlCut
) -> CompositionInputProjection:
    return CompositionInputProjection(
        control_domain_id=cut.control_domain_id,
        cutoff_at=cut.cutoff_at,
        cutoff_admission_sequence=cut.cutoff_admission_sequence,
        records=store.records_at_or_before(cut),
    )


def _epoch_hash(snapshot: ProducerEpochSnapshot) -> str:
    return canonical_content_hash(
        {
            "control_domain_id": snapshot.control_domain_id,
            "cutoff_at": snapshot.cutoff_at,
            "cutoff_admission_sequence": snapshot.cutoff_admission_sequence,
            "active_epoch_by_producer": snapshot.active_epoch_by_producer,
            "activation_id_by_producer": snapshot.activation_id_by_producer,
        },
        domain="dexteleop.composition_epoch_snapshot.v1",
    )


def build_composition_input_snapshot(
    projection: CompositionInputProjection,
    *,
    producer_epochs: ProducerEpochSnapshot,
    spec: CompositionSpec,
    target_effective_at: float,
) -> CompositionInputSnapshot:
    target = _finite(target_effective_at, field_name="target_effective_at")
    if producer_epochs.control_domain_id != projection.control_domain_id:
        raise ValueError("Composition projection and epochs belong to different domains.")
    if producer_epochs.cutoff_admission_sequence != projection.cutoff_admission_sequence:
        raise ValueError("Composition projection and epochs must share one committed cut.")
    selected: dict[str, CompositionInputRecord] = {}
    for record in projection.records:
        if record.input_role not in spec.input_roles:
            continue
        if producer_epochs.active_epoch_by_producer.get(record.producer_id) != record.producer_epoch:
            continue
        if not record.effective_from <= target < record.effective_until:
            continue
        current = selected.get(record.input_role)
        rank = (record.effective_from, record.sequence, record.composition_input_id)
        if current is None or rank > (
            current.effective_from,
            current.sequence,
            current.composition_input_id,
        ):
            selected[record.input_role] = record
    missing = tuple(role for role in spec.input_roles if role not in selected)
    epoch_hash = _epoch_hash(producer_epochs)
    payload = {
        "control_domain_id": projection.control_domain_id,
        "cutoff_at": projection.cutoff_at,
        "cutoff_admission_sequence": projection.cutoff_admission_sequence,
        "target_effective_at": target,
        "composition_spec_hash": spec.semantic_content_hash,
        "selected_input_ids_by_role": {
            role: selected[role].composition_input_id for role in spec.input_roles if role in selected
        },
        "selected_input_hashes_by_role": {
            role: selected[role].semantic_content_hash for role in spec.input_roles if role in selected
        },
        "missing_roles": missing,
        "producer_epoch_snapshot_hash": epoch_hash,
    }
    digest = canonical_content_hash(payload, domain=COMPOSITION_INPUT_SNAPSHOT_HASH_DOMAIN_V1)
    return CompositionInputSnapshot(
        composition_input_snapshot_id=f"composition-input-snapshot:{digest}",
        control_domain_id=projection.control_domain_id,
        cutoff_at=projection.cutoff_at,
        cutoff_admission_sequence=projection.cutoff_admission_sequence,
        target_effective_at=target,
        composition_spec_hash=spec.semantic_content_hash,
        selected_input_by_role=selected,
        missing_roles=missing,
        producer_epoch_snapshot_hash=epoch_hash,
        semantic_content_hash=digest,
    )


def _component_union(inputs: tuple[CompositionInputRecord, ...], target: float) -> RobotAction:
    if not inputs:
        raise ValueError("component_union requires at least one input.")
    action_space = inputs[0].action.action_space
    arm_pose: dict[str, Any] = {}
    arm_qpos: dict[str, Any] = {}
    hand_qpos: dict[str, Any] = {}
    native: dict[str, Any] = {}
    frames: dict[str, str] = {}
    controlled: dict[str, str] = {}
    metadata: dict[str, dict[str, Any]] = {}
    effectors: list[str] = []
    for record in inputs:
        action = record.action
        if action.action_space != action_space:
            raise ValueError("component_union action spaces do not match.")
        overlap = set(effectors) & set(action.active_effectors)
        if overlap:
            raise ValueError(f"component_union inputs overlap: {sorted(overlap)!r}.")
        arm_pose.update(action.arm_pose_by_side)
        arm_qpos.update(action.arm_qpos_by_side)
        hand_qpos.update(action.hand_qpos_by_side)
        native.update(action.end_effector_targets_by_side)
        frames.update(action.frame_by_effector)
        controlled.update(action.controlled_frame_by_effector)
        for key, values in action.metadata.items():
            destination = metadata.setdefault(key, {})
            for side, value in values.items():
                if side in destination:
                    raise ValueError("component_union execution metadata overlaps.")
                destination[side] = value
        effectors.extend(action.active_effectors)
    return freeze_robot_action(
        RobotAction(
            action_space=action_space,
            timestamp=target,
            arm_pose_by_side=arm_pose,
            arm_qpos_by_side=arm_qpos,
            hand_qpos_by_side=hand_qpos,
            end_effector_targets_by_side=native,
            active_effectors=tuple(effectors),
            frame_by_effector=frames,
            controlled_frame_by_effector=controlled,
            source="deterministic_composer",
            metadata=metadata,
        )
    )


def _base_plus_residual(
    base: CompositionInputRecord,
    residual: CompositionInputRecord | None,
    *,
    spec: CompositionSpec,
    target: float,
) -> RobotAction:
    base_action = base.action
    if base_action.arm_pose_by_side or base_action.end_effector_targets_by_side:
        raise ValueError("v1 bounded residual supports qpos components only.")
    arm = {side: value.copy() for side, value in base_action.arm_qpos_by_side.items()}
    hand = {side: value.copy() for side, value in base_action.hand_qpos_by_side.items()}
    if residual is not None:
        residual_action = residual.action
        if residual_action.action_space != base_action.action_space:
            raise ValueError("Base and residual action spaces do not match.")
        if residual_action.arm_pose_by_side or residual_action.end_effector_targets_by_side:
            raise ValueError("v1 bounded residual supports qpos components only.")
        for side, value in residual_action.arm_qpos_by_side.items():
            component = f"{side}_arm"
            if side not in arm or component not in spec.residual_limit_by_component:
                raise ValueError(f"Residual component {component!r} has no compatible base/limit.")
            limit = spec.residual_limit_by_component[component]
            arm[side] = arm[side] + np.clip(value, -limit, limit)
        for side, value in residual_action.hand_qpos_by_side.items():
            component = f"{side}_hand"
            if side not in hand or component not in spec.residual_limit_by_component:
                raise ValueError(f"Residual component {component!r} has no compatible base/limit.")
            limit = spec.residual_limit_by_component[component]
            hand[side] = hand[side] + np.clip(value, -limit, limit)
    return freeze_robot_action(
        RobotAction(
            action_space=base_action.action_space,
            timestamp=target,
            arm_qpos_by_side=arm,
            hand_qpos_by_side=hand,
            active_effectors=base_action.active_effectors,
            frame_by_effector=base_action.frame_by_effector,
            controlled_frame_by_effector=base_action.controlled_frame_by_effector,
            source="deterministic_composer",
            metadata=base_action.metadata,
        )
    )


def _weighted_se3_pose_fusion(
    inputs_by_role: Mapping[str, CompositionInputRecord],
    *,
    spec: CompositionSpec,
    target: float,
) -> RobotAction:
    selected_weights = {
        role: spec.weight_by_role[role]
        for role in inputs_by_role
        if spec.weight_by_role[role] > 0.0
    }
    if not selected_weights:
        raise ValueError("No positive weighted pose input is available.")
    return freeze_robot_action(
        fuse_weighted_se3_pose_actions(
            {
                role: inputs_by_role[role].action
                for role in selected_weights
            },
            weights_by_source_slot=selected_weights,
            target_effective_at=target,
            max_translation_disagreement_m=float(
                spec.max_translation_disagreement_m
            ),
            max_rotation_disagreement_rad=float(
                spec.max_rotation_disagreement_rad
            ),
            source=f"deterministic_composer:{spec.spec_id}",
        )
    )


class DeterministicComposer:
    def compose(
        self,
        snapshot: CompositionInputSnapshot,
        *,
        spec: CompositionSpec,
    ) -> CompositionResult:
        if snapshot.composition_spec_hash != spec.semantic_content_hash:
            raise ValueError("Composition snapshot/spec hash mismatch.")
        selected = snapshot.selected_input_by_role
        missing_required = tuple(role for role in spec.required_roles if role not in selected)
        status = "ready"
        reason = "composed"
        action: RobotAction | None = None
        if missing_required:
            status = spec.missing_required_behavior
            reason = "missing_required:" + ",".join(missing_required)
        elif spec.algebra == "base_plus_bounded_residual" and "residual" not in selected:
            if spec.missing_residual_behavior == "zero_residual":
                action = _base_plus_residual(
                    selected["base"], None, spec=spec, target=snapshot.target_effective_at
                )
                reason = "zero_residual"
            else:
                status = spec.missing_residual_behavior
                reason = "missing_residual"
        else:
            try:
                ordered = tuple(
                    selected[role] for role in spec.input_roles if role in selected
                )
                if spec.algebra == "component_union":
                    action = _component_union(ordered, snapshot.target_effective_at)
                elif spec.algebra == "base_plus_bounded_residual":
                    action = _base_plus_residual(
                        selected["base"],
                        selected["residual"],
                        spec=spec,
                        target=snapshot.target_effective_at,
                    )
                else:
                    action = _weighted_se3_pose_fusion(
                        selected,
                        spec=spec,
                        target=snapshot.target_effective_at,
                    )
                    reason = (
                        "weighted_se3_pose_fusion"
                        if not snapshot.missing_roles
                        else "weighted_se3_pose_fusion_renormalized"
                    )
            except (TypeError, ValueError) as exc:
                status = "invalid"
                reason = str(exc)
        ordered_inputs = tuple(
            selected[role] for role in spec.input_roles if role in selected
        )
        input_ids = tuple(item.composition_input_id for item in ordered_inputs)
        input_hashes = tuple(item.semantic_content_hash for item in ordered_inputs)
        context_ids = tuple(
            dict.fromkeys(
                context_id for item in ordered_inputs for context_id in item.context_ids
            )
        )
        action_hash = (
            None
            if action is None
            else canonical_content_hash(
                robot_action_semantic_payload(action),
                domain="dexteleop.composed_robot_action.v1",
            )
        )
        payload = {
            "composition_input_snapshot_id": snapshot.composition_input_snapshot_id,
            "composition_input_snapshot_hash": snapshot.semantic_content_hash,
            "composition_spec_hash": spec.semantic_content_hash,
            "output_status": status,
            "reason": reason,
            "composed_action_hash": action_hash,
            "input_ids": input_ids,
            "input_hashes": input_hashes,
            "context_ids": context_ids,
        }
        digest = canonical_content_hash(payload, domain=COMPOSITION_RESULT_HASH_DOMAIN_V1)
        return CompositionResult(
            composition_result_id=f"composition-result:{digest}",
            composed_action=action,
            semantic_content_hash=digest,
            **payload,
        )


def build_composite_action_proposal_draft(
    result: CompositionResult,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str,
    producer_id: str,
    producer_epoch: int,
    sequence: int,
    produced_at: float,
    produced_clock_domain_id: str,
    effective_from: float,
    effective_until: float,
    effective_clock_domain_id: str,
    component_schema_hash: str,
    component_coverage: tuple[str, ...],
    selection_groups: tuple[str, ...],
    policy_node_id: str | None = None,
) -> ActionProposalDraft:
    if result.output_status != "ready" or result.composed_action is None:
        raise ValueError("Only a ready composition result can become a proposal draft.")
    return ActionProposalDraft(
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
        producer_kind="composer",
        producer_id=producer_id,
        producer_epoch=producer_epoch,
        sequence=sequence,
        action=result.composed_action,
        produced_at=produced_at,
        produced_clock_domain_id=produced_clock_domain_id,
        effective_from=effective_from,
        effective_until=effective_until,
        target_effective_at=result.composed_action.timestamp,
        effective_clock_domain_id=effective_clock_domain_id,
        component_schema_hash=component_schema_hash,
        component_coverage=component_coverage,
        selection_groups=selection_groups,
        policy_node_id=policy_node_id,
        context_ids=result.context_ids,
        policy_output_id=result.composition_result_id,
        composition_input_ids=result.input_ids,
        composition_input_hashes=result.input_hashes,
        composition_spec_hash=result.composition_spec_hash,
    )


def build_blocked_composition_result(
    snapshot: CompositionInputSnapshot,
    *,
    spec: CompositionSpec,
    status: str,
    reason: str,
) -> CompositionResult:
    """Build a deterministic non-action result for an external dependency gate.

    Context freshness is owned by the policy-topology runtime rather than the
    numerical composer.  This helper lets that runtime preserve the exact
    composition-input snapshot while failing closed before proposal admission.
    """

    normalized_status = _required(status, field_name="composition status")
    if normalized_status not in {"no_action", "invalid"}:
        raise ValueError("Blocked composition status must be no_action or invalid.")
    normalized_reason = _required(reason, field_name="composition reason")
    if snapshot.composition_spec_hash != spec.semantic_content_hash:
        raise ValueError("Composition snapshot/spec hash mismatch.")
    ordered_inputs = tuple(
        snapshot.selected_input_by_role[role]
        for role in spec.input_roles
        if role in snapshot.selected_input_by_role
    )
    input_ids = tuple(item.composition_input_id for item in ordered_inputs)
    input_hashes = tuple(item.semantic_content_hash for item in ordered_inputs)
    context_ids = tuple(
        dict.fromkeys(
            context_id for item in ordered_inputs for context_id in item.context_ids
        )
    )
    payload = {
        "composition_input_snapshot_id": snapshot.composition_input_snapshot_id,
        "composition_input_snapshot_hash": snapshot.semantic_content_hash,
        "composition_spec_hash": spec.semantic_content_hash,
        "output_status": normalized_status,
        "reason": normalized_reason,
        "composed_action_hash": None,
        "input_ids": input_ids,
        "input_hashes": input_hashes,
        "context_ids": context_ids,
    }
    digest = canonical_content_hash(payload, domain=COMPOSITION_RESULT_HASH_DOMAIN_V1)
    return CompositionResult(
        composition_result_id=f"composition-result:{digest}",
        composed_action=None,
        semantic_content_hash=digest,
        **payload,
    )


__all__ = [
    "COMPOSITION_ALGEBRAS_V1",
    "COMPOSITION_INPUT_HASH_DOMAIN_V1",
    "COMPOSITION_INPUT_SNAPSHOT_HASH_DOMAIN_V1",
    "COMPOSITION_RESULT_HASH_DOMAIN_V1",
    "COMPOSITION_SPEC_HASH_DOMAIN_V1",
    "CompositionAdmissionError",
    "CompositionInputAdmissionReceipt",
    "CompositionInputDraft",
    "CompositionInputProjection",
    "CompositionInputRecord",
    "CompositionInputSnapshot",
    "CompositionInputStore",
    "CompositionResult",
    "CompositionSpec",
    "DeterministicComposer",
    "build_composite_action_proposal_draft",
    "build_blocked_composition_result",
    "build_composition_input_snapshot",
    "composition_input_semantic_payload",
    "project_composition_inputs",
]
