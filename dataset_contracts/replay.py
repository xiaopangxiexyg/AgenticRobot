"""Transport- and runtime-neutral replay provenance contracts.

These records describe what historical fact is being replayed.  They do not
own a reader, scheduler, action queue, safety pass, or dispatcher.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .canonical_encoding import canonical_content_hash


REPLAY_MODE_DECISION = "decision_replay"
REPLAY_MODE_POLICY_OBSERVATION = "policy_observation_replay"
REPLAY_MODE_CANONICAL_ACTION = "canonical_action_replay"
REPLAY_MODE_COMMAND_AUDIT = "command_audit"
REPLAY_MODES_V1 = (
    REPLAY_MODE_DECISION,
    REPLAY_MODE_POLICY_OBSERVATION,
    REPLAY_MODE_CANONICAL_ACTION,
    REPLAY_MODE_COMMAND_AUDIT,
)

REPLAY_ACTION_STAGE_PROPOSAL = "proposal_pre_safety"
REPLAY_ACTION_STAGE_SELECTED = "selected_pre_safety"
REPLAY_ACTION_STAGE_SAFE = "safe_canonical"
REPLAY_ACTION_STAGE_COMMANDED = "commanded_backend"
REPLAY_ACTION_STAGES_V1 = (
    REPLAY_ACTION_STAGE_PROPOSAL,
    REPLAY_ACTION_STAGE_SELECTED,
    REPLAY_ACTION_STAGE_SAFE,
    REPLAY_ACTION_STAGE_COMMANDED,
)
REPLAY_CANONICAL_ACTION_STAGES_V1 = (
    REPLAY_ACTION_STAGE_PROPOSAL,
    REPLAY_ACTION_STAGE_SELECTED,
    REPLAY_ACTION_STAGE_SAFE,
)

REPLAY_SCHEDULE_NATIVE = "native_timestamps"
REPLAY_SCHEDULE_FIXED_RATE = "fixed_rate"
REPLAY_SCHEDULE_RESAMPLED = "resampled"
REPLAY_SCHEDULE_MODES_V1 = (
    REPLAY_SCHEDULE_NATIVE,
    REPLAY_SCHEDULE_FIXED_RATE,
    REPLAY_SCHEDULE_RESAMPLED,
)

REPLAY_SOURCE_CONTRACT_HASH_DOMAIN_V1 = "dexteleop.replay_source_contract.v1"
REPLAY_SCHEDULE_SPEC_HASH_DOMAIN_V1 = "dexteleop.replay_schedule_spec.v1"
REPLAY_SOURCE_ACTION_CONTRACT_HASH_DOMAIN_V1 = (
    "dexteleop.replay_source_action_contract.v1"
)
LEGACY_REPLAY_DERIVATION_HASH_DOMAIN_V1 = (
    "dexteleop.legacy_replay_contract_derivation.v1"
)
REPLAY_TARGET_RUNTIME_CONTRACT_HASH_DOMAIN_V1 = (
    "dexteleop.target_runtime_action_contract.v1"
)
REPLAY_TARGET_BINDING_SPEC_HASH_DOMAIN_V1 = "dexteleop.replay_target_binding_spec.v1"
REPLAY_BINDING_ASSESSMENT_HASH_DOMAIN_V1 = "dexteleop.replay_binding_assessment.v1"
REPLAY_BINDING_ELIGIBILITY_HASH_DOMAIN_V1 = "dexteleop.replay_binding_eligibility.v1"
REPLAY_EXECUTION_AUTHORIZATION_HASH_DOMAIN_V1 = (
    "dexteleop.replay_execution_authorization.v1"
)
REPLAY_ACTION_RECORD_HASH_DOMAIN_V1 = "dexteleop.replay_action_record.v1"

REPLAY_SOURCE_INTERPRETATION_VERIFIED = "verified"
REPLAY_SOURCE_INTERPRETATION_UNSUPPORTED = "unsupported"
REPLAY_SOURCE_INTERPRETATION_INVALID = "invalid"
REPLAY_SOURCE_INTERPRETATION_STATUSES_V1 = (
    REPLAY_SOURCE_INTERPRETATION_VERIFIED,
    REPLAY_SOURCE_INTERPRETATION_UNSUPPORTED,
    REPLAY_SOURCE_INTERPRETATION_INVALID,
)

REPLAY_BINDING_VERIFIED_COMPATIBLE = "verified_compatible"
REPLAY_BINDING_VERIFIED_INCOMPATIBLE = "verified_incompatible"
REPLAY_BINDING_INDETERMINATE = "indeterminate"
REPLAY_BINDING_VERDICTS_V1 = (
    REPLAY_BINDING_VERIFIED_COMPATIBLE,
    REPLAY_BINDING_VERIFIED_INCOMPATIBLE,
    REPLAY_BINDING_INDETERMINATE,
)

REPLAY_COMPAT_EXACT_MATCH = "exact_match"
REPLAY_COMPAT_DERIVED_MATCH = "derived_match"
REPLAY_COMPAT_VERIFIED_TRANSFORM = "verified_transform"
REPLAY_COMPAT_COMPATIBLE_SUBSET = "compatible_subset"
REPLAY_COMPAT_UNKNOWN = "unknown"
REPLAY_COMPAT_MISMATCH = "mismatch"
REPLAY_COMPAT_NOT_APPLICABLE = "not_applicable"
REPLAY_COMPATIBILITY_CHECK_STATUSES_V1 = (
    REPLAY_COMPAT_EXACT_MATCH,
    REPLAY_COMPAT_DERIVED_MATCH,
    REPLAY_COMPAT_VERIFIED_TRANSFORM,
    REPLAY_COMPAT_COMPATIBLE_SUBSET,
    REPLAY_COMPAT_UNKNOWN,
    REPLAY_COMPAT_MISMATCH,
    REPLAY_COMPAT_NOT_APPLICABLE,
)

REPLAY_BINDING_USE_RECORD_ONLY = "record_only"
REPLAY_BINDING_USE_PASSIVE_SHADOW = "passive_shadow"
REPLAY_BINDING_USE_EXECUTION_CANDIDATE = "execution_candidate"
REPLAY_BINDING_USES_V1 = (
    REPLAY_BINDING_USE_RECORD_ONLY,
    REPLAY_BINDING_USE_PASSIVE_SHADOW,
    REPLAY_BINDING_USE_EXECUTION_CANDIDATE,
)
REPLAY_EXECUTION_MODE_SIM_AUTHORITY = "sim_authority"
REPLAY_EXECUTION_MODE_REAL_AUTHORITY = "real_authority"
REPLAY_EXECUTION_MODES_V1 = (
    REPLAY_EXECUTION_MODE_SIM_AUTHORITY,
    REPLAY_EXECUTION_MODE_REAL_AUTHORITY,
)


def _required(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _optional(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _finite(value: object, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _freeze(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(np.asarray(value))
        if array.dtype.hasobject:
            raise TypeError("Replay facts reject object arrays.")
        return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(
            array.shape
        )
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class ReplaySourceContract:
    contract_id: str
    replay_mode: str
    dataset_schema: str
    source_dataset_id: str
    episode_id: str
    action_stage: str | None
    action_field: str | None
    source_action_space: str | None
    capture_id: str | None = None
    source_frame_contract_hash: str | None = None
    source_component_schema_hash: str | None = None
    old_safety_profile_hash: str | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("contract_id", "dataset_schema", "source_dataset_id", "episode_id"):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        mode = _required(self.replay_mode, name="replay_mode")
        if mode not in REPLAY_MODES_V1:
            raise ValueError(f"Unsupported replay_mode: {mode!r}.")
        object.__setattr__(self, "replay_mode", mode)
        for name in (
            "action_stage",
            "action_field",
            "source_action_space",
            "capture_id",
            "source_frame_contract_hash",
            "source_component_schema_hash",
            "old_safety_profile_hash",
        ):
            object.__setattr__(self, name, _optional(getattr(self, name)))
        if mode == REPLAY_MODE_CANONICAL_ACTION:
            if self.action_stage not in REPLAY_CANONICAL_ACTION_STAGES_V1:
                raise ValueError(
                    "canonical_action_replay requires proposal, selected, or safe canonical stage."
                )
            if self.action_field is None or self.source_action_space is None:
                raise ValueError(
                    "canonical_action_replay requires action_field and source_action_space."
                )
        if self.action_stage == REPLAY_ACTION_STAGE_COMMANDED and mode != REPLAY_MODE_COMMAND_AUDIT:
            raise ValueError("Backend command facts are audit-only replay inputs.")
        if mode == REPLAY_MODE_COMMAND_AUDIT and self.action_stage != REPLAY_ACTION_STAGE_COMMANDED:
            raise ValueError("command_audit must bind commanded_backend stage.")
        payload = replay_source_contract_payload(self)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(payload, domain=REPLAY_SOURCE_CONTRACT_HASH_DOMAIN_V1),
        )


def replay_source_contract_payload(contract: ReplaySourceContract) -> dict[str, Any]:
    return {
        "contract_id": contract.contract_id,
        "replay_mode": contract.replay_mode,
        "dataset_schema": contract.dataset_schema,
        "source_dataset_id": contract.source_dataset_id,
        "capture_id": contract.capture_id,
        "episode_id": contract.episode_id,
        "action_stage": contract.action_stage,
        "action_field": contract.action_field,
        "source_action_space": contract.source_action_space,
        "source_frame_contract_hash": contract.source_frame_contract_hash,
        "source_component_schema_hash": contract.source_component_schema_hash,
        "old_safety_profile_hash": contract.old_safety_profile_hash,
    }


@dataclass(frozen=True)
class ReplayScheduleSpec:
    schedule_id: str
    schedule_mode: str
    source_hz: float | None
    target_hz: float | None
    resampling_policy: str
    source_start_index: int = 0
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "schedule_id", _required(self.schedule_id, name="schedule_id"))
        mode = _required(self.schedule_mode, name="schedule_mode")
        if mode not in REPLAY_SCHEDULE_MODES_V1:
            raise ValueError(f"Unsupported replay schedule mode: {mode!r}.")
        object.__setattr__(self, "schedule_mode", mode)
        for name in ("source_hz", "target_hz"):
            value = getattr(self, name)
            if value is not None:
                rate = _finite(value, name=name)
                if rate <= 0.0:
                    raise ValueError(f"{name} must be positive when present.")
                object.__setattr__(self, name, rate)
        if mode == REPLAY_SCHEDULE_FIXED_RATE and self.source_hz is None:
            raise ValueError("fixed_rate replay requires source_hz.")
        if mode == REPLAY_SCHEDULE_RESAMPLED and self.target_hz is None:
            raise ValueError("resampled replay requires target_hz.")
        policy = _required(self.resampling_policy, name="resampling_policy")
        if mode != REPLAY_SCHEDULE_RESAMPLED and policy != "none":
            raise ValueError("Only resampled replay may declare a non-none resampling policy.")
        object.__setattr__(self, "resampling_policy", policy)
        if int(self.source_start_index) < 0:
            raise ValueError("source_start_index must be non-negative.")
        object.__setattr__(self, "source_start_index", int(self.source_start_index))
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                replay_schedule_spec_payload(self),
                domain=REPLAY_SCHEDULE_SPEC_HASH_DOMAIN_V1,
            ),
        )

    def relative_time_for_index(self, source_action_index: int) -> float:
        index = int(source_action_index)
        if index < self.source_start_index:
            raise ValueError("source_action_index precedes source_start_index.")
        if self.source_hz is None:
            raise ValueError("This schedule requires an explicit source timestamp.")
        return float(index - self.source_start_index) / self.source_hz


def replay_schedule_spec_payload(spec: ReplayScheduleSpec) -> dict[str, Any]:
    return {
        "schedule_id": spec.schedule_id,
        "schedule_mode": spec.schedule_mode,
        "source_hz": spec.source_hz,
        "target_hz": spec.target_hz,
        "resampling_policy": spec.resampling_policy,
        "source_start_index": spec.source_start_index,
    }


@dataclass(frozen=True)
class LegacyReplayContractDerivationReceipt:
    derivation_receipt_id: str
    derivation_contract_id: str
    derivation_contract_hash: str
    source_dataset_schema: str
    consumed_metadata_fields: tuple[str, ...]
    derived_action_stage: str
    derived_pose_frame_contract_hash: str | None
    derived_component_schema_hash: str | None
    unresolved_dimensions: tuple[str, ...]
    derivation_status: str
    semantic_content_hash: str

    def __post_init__(self) -> None:
        for name in (
            "derivation_receipt_id",
            "derivation_contract_id",
            "derivation_contract_hash",
            "source_dataset_schema",
            "derived_action_stage",
        ):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        status = _required(self.derivation_status, name="derivation_status")
        if status not in REPLAY_SOURCE_INTERPRETATION_STATUSES_V1:
            raise ValueError(f"Unsupported replay derivation status {status!r}.")
        object.__setattr__(self, "derivation_status", status)
        object.__setattr__(
            self,
            "consumed_metadata_fields",
            tuple(
                dict.fromkeys(
                    _required(value, name="consumed_metadata_field")
                    for value in self.consumed_metadata_fields
                )
            ),
        )
        object.__setattr__(
            self,
            "unresolved_dimensions",
            tuple(
                dict.fromkeys(
                    _required(value, name="unresolved_dimension")
                    for value in self.unresolved_dimensions
                )
            ),
        )
        for name in (
            "derived_pose_frame_contract_hash",
            "derived_component_schema_hash",
        ):
            object.__setattr__(self, name, _optional(getattr(self, name)))
        expected_hash = canonical_content_hash(
            legacy_replay_contract_derivation_payload(self),
            domain=LEGACY_REPLAY_DERIVATION_HASH_DOMAIN_V1,
        )
        if self.semantic_content_hash != expected_hash:
            raise ValueError("Legacy replay derivation receipt hash mismatch.")
        if self.derivation_receipt_id != f"legacy-replay-derivation:{expected_hash}":
            raise ValueError("Legacy replay derivation receipt ID/hash mismatch.")


def legacy_replay_contract_derivation_payload(
    receipt: LegacyReplayContractDerivationReceipt,
) -> dict[str, Any]:
    return {
        "derivation_contract_id": receipt.derivation_contract_id,
        "derivation_contract_hash": receipt.derivation_contract_hash,
        "source_dataset_schema": receipt.source_dataset_schema,
        "consumed_metadata_fields": receipt.consumed_metadata_fields,
        "derived_action_stage": receipt.derived_action_stage,
        "derived_pose_frame_contract_hash": (
            receipt.derived_pose_frame_contract_hash
        ),
        "derived_component_schema_hash": receipt.derived_component_schema_hash,
        "unresolved_dimensions": receipt.unresolved_dimensions,
        "derivation_status": receipt.derivation_status,
    }


@dataclass(frozen=True)
class ReplaySourceActionContract:
    contract_id: str
    interpretation_status: str
    action_stage: str
    action_space: str
    component_ids: tuple[str, ...]
    selection_group_by_component: Mapping[str, str]
    coupled_selection_groups: tuple[str, ...] = ()
    pose_frame_contract_hash: str | None = None
    component_schema_hash: str | None = None
    action_spec_hash_by_component: Mapping[str, str] = field(default_factory=dict)
    units_convention_id: str | None = None
    derivation_receipt_id: str | None = None
    derivation_receipt_hash: str | None = None
    unresolved_dimensions: tuple[str, ...] = ()
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("contract_id", "action_stage", "action_space"):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        if self.action_stage not in REPLAY_CANONICAL_ACTION_STAGES_V1:
            raise ValueError("Replay source action contract requires a canonical action stage.")
        status = _required(self.interpretation_status, name="interpretation_status")
        if status not in REPLAY_SOURCE_INTERPRETATION_STATUSES_V1:
            raise ValueError(f"Unsupported replay source interpretation status {status!r}.")
        object.__setattr__(self, "interpretation_status", status)
        components = tuple(dict.fromkeys(_required(v, name="component_id") for v in self.component_ids))
        if not components:
            raise ValueError("Replay source action contract requires components.")
        groups = {
            str(key): _required(value, name="selection_group_id")
            for key, value in dict(self.selection_group_by_component).items()
        }
        if set(groups) != set(components):
            raise ValueError("Every replay source component requires one selection group.")
        object.__setattr__(self, "component_ids", components)
        object.__setattr__(self, "selection_group_by_component", MappingProxyType(groups))
        object.__setattr__(
            self,
            "coupled_selection_groups",
            tuple(dict.fromkeys(_required(v, name="coupled_group") for v in self.coupled_selection_groups)),
        )
        if any(
            group not in set(groups.values())
            for group in self.coupled_selection_groups
        ):
            raise ValueError("Replay source coupled group is not in its selection groups.")
        specs = {str(k): _required(v, name="action_spec_hash") for k, v in dict(self.action_spec_hash_by_component).items()}
        if any(key not in components for key in specs):
            raise ValueError("Replay source action spec references an unknown component.")
        object.__setattr__(self, "action_spec_hash_by_component", MappingProxyType(specs))
        for name in (
            "pose_frame_contract_hash",
            "component_schema_hash",
            "units_convention_id",
            "derivation_receipt_id",
            "derivation_receipt_hash",
        ):
            object.__setattr__(self, name, _optional(getattr(self, name)))
        if (self.derivation_receipt_id is None) != (
            self.derivation_receipt_hash is None
        ):
            raise ValueError("Replay derivation receipt ID/hash must be both present or absent.")
        unresolved = tuple(dict.fromkeys(_required(v, name="unresolved_dimension") for v in self.unresolved_dimensions))
        object.__setattr__(self, "unresolved_dimensions", unresolved)
        if status == REPLAY_SOURCE_INTERPRETATION_VERIFIED and unresolved:
            raise ValueError("A verified replay source contract cannot have unresolved dimensions.")
        payload = replay_source_action_contract_payload(self)
        object.__setattr__(self, "semantic_content_hash", canonical_content_hash(payload, domain=REPLAY_SOURCE_ACTION_CONTRACT_HASH_DOMAIN_V1))


def replay_source_action_contract_payload(contract: ReplaySourceActionContract) -> dict[str, Any]:
    return {
        "contract_id": contract.contract_id,
        "interpretation_status": contract.interpretation_status,
        "action_stage": contract.action_stage,
        "action_space": contract.action_space,
        "component_ids": contract.component_ids,
        "selection_group_by_component": dict(contract.selection_group_by_component),
        "coupled_selection_groups": contract.coupled_selection_groups,
        "pose_frame_contract_hash": contract.pose_frame_contract_hash,
        "component_schema_hash": contract.component_schema_hash,
        "action_spec_hash_by_component": dict(contract.action_spec_hash_by_component),
        "units_convention_id": contract.units_convention_id,
        "derivation_receipt_id": contract.derivation_receipt_id,
        "derivation_receipt_hash": contract.derivation_receipt_hash,
        "unresolved_dimensions": contract.unresolved_dimensions,
    }


@dataclass(frozen=True)
class TargetRuntimeActionContract:
    target_contract_id: str
    control_domain_id: str
    embodiment_instance_id: str
    component_schema_hash: str
    component_ids: tuple[str, ...]
    selection_group_by_component: Mapping[str, str]
    accepted_action_spaces_by_component: Mapping[str, tuple[str, ...]]
    coupled_selection_groups: tuple[str, ...] = ()
    pose_frame_contract_hash: str | None = None
    action_spec_hash_by_component: Mapping[str, str] = field(default_factory=dict)
    units_convention_id: str | None = None
    supported_transform_hash_by_source_frame: Mapping[str, str] = field(default_factory=dict)
    execution_capability_profile_hash: str | None = None
    unresolved_dimensions: tuple[str, ...] = ()
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("target_contract_id", "control_domain_id", "embodiment_instance_id", "component_schema_hash"):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        components = tuple(dict.fromkeys(_required(v, name="component_id") for v in self.component_ids))
        if not components:
            raise ValueError("Target runtime action contract requires components.")
        groups = {str(k): _required(v, name="selection_group_id") for k, v in dict(self.selection_group_by_component).items()}
        spaces = {
            str(k): tuple(dict.fromkeys(_required(v, name="accepted_action_space") for v in values))
            for k, values in dict(self.accepted_action_spaces_by_component).items()
        }
        if set(groups) != set(components) or set(spaces) != set(components):
            raise ValueError("Target components require selection-group and action-space entries.")
        if any(not values for values in spaces.values()):
            raise ValueError("Target component action-space sets must be non-empty.")
        object.__setattr__(self, "component_ids", components)
        object.__setattr__(self, "selection_group_by_component", MappingProxyType(groups))
        object.__setattr__(self, "accepted_action_spaces_by_component", MappingProxyType(spaces))
        coupled = tuple(
            dict.fromkeys(
                _required(value, name="coupled_group")
                for value in self.coupled_selection_groups
            )
        )
        if any(group not in set(groups.values()) for group in coupled):
            raise ValueError("Target coupled group is not in its selection groups.")
        object.__setattr__(self, "coupled_selection_groups", coupled)
        specs = {
            _required(key, name="action_spec_component"): _required(
                value,
                name="action_spec_hash",
            )
            for key, value in dict(self.action_spec_hash_by_component).items()
        }
        if any(component not in components for component in specs):
            raise ValueError("Target action spec references an unknown component.")
        object.__setattr__(
            self,
            "action_spec_hash_by_component",
            MappingProxyType(specs),
        )
        transforms = {
            _required(key, name="source_frame_contract_hash"): _required(
                value,
                name="transform_receipt_hash",
            )
            for key, value in dict(
                self.supported_transform_hash_by_source_frame
            ).items()
        }
        object.__setattr__(
            self,
            "supported_transform_hash_by_source_frame",
            MappingProxyType(transforms),
        )
        for name in ("pose_frame_contract_hash", "units_convention_id", "execution_capability_profile_hash"):
            object.__setattr__(self, name, _optional(getattr(self, name)))
        object.__setattr__(
            self,
            "unresolved_dimensions",
            tuple(
                dict.fromkeys(
                    _required(value, name="unresolved_dimension")
                    for value in self.unresolved_dimensions
                )
            ),
        )
        payload = target_runtime_action_contract_payload(self)
        object.__setattr__(self, "semantic_content_hash", canonical_content_hash(payload, domain=REPLAY_TARGET_RUNTIME_CONTRACT_HASH_DOMAIN_V1))


def target_runtime_action_contract_payload(contract: TargetRuntimeActionContract) -> dict[str, Any]:
    return {
        "target_contract_id": contract.target_contract_id,
        "control_domain_id": contract.control_domain_id,
        "embodiment_instance_id": contract.embodiment_instance_id,
        "component_schema_hash": contract.component_schema_hash,
        "component_ids": contract.component_ids,
        "selection_group_by_component": dict(contract.selection_group_by_component),
        "accepted_action_spaces_by_component": dict(contract.accepted_action_spaces_by_component),
        "coupled_selection_groups": contract.coupled_selection_groups,
        "pose_frame_contract_hash": contract.pose_frame_contract_hash,
        "action_spec_hash_by_component": dict(contract.action_spec_hash_by_component),
        "units_convention_id": contract.units_convention_id,
        "supported_transform_hash_by_source_frame": dict(contract.supported_transform_hash_by_source_frame),
        "execution_capability_profile_hash": contract.execution_capability_profile_hash,
        "unresolved_dimensions": contract.unresolved_dimensions,
    }


@dataclass(frozen=True)
class ReplayTargetBindingSpec:
    binding_id: str
    source_action_contract_id: str
    source_action_contract_hash: str
    target_runtime_contract_id: str
    target_runtime_contract_hash: str
    transform_receipt_id: str | None = None
    transform_receipt_hash: str | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("binding_id", "source_action_contract_id", "source_action_contract_hash", "target_runtime_contract_id", "target_runtime_contract_hash"):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        object.__setattr__(self, "transform_receipt_id", _optional(self.transform_receipt_id))
        object.__setattr__(self, "transform_receipt_hash", _optional(self.transform_receipt_hash))
        if (self.transform_receipt_id is None) != (self.transform_receipt_hash is None):
            raise ValueError("Transform receipt id/hash must be both present or both absent.")
        payload = replay_target_binding_spec_payload(self)
        object.__setattr__(self, "semantic_content_hash", canonical_content_hash(payload, domain=REPLAY_TARGET_BINDING_SPEC_HASH_DOMAIN_V1))


def replay_target_binding_spec_payload(binding: ReplayTargetBindingSpec) -> dict[str, Any]:
    return {
        "binding_id": binding.binding_id,
        "source_action_contract_id": binding.source_action_contract_id,
        "source_action_contract_hash": binding.source_action_contract_hash,
        "target_runtime_contract_id": binding.target_runtime_contract_id,
        "target_runtime_contract_hash": binding.target_runtime_contract_hash,
        "transform_receipt_id": binding.transform_receipt_id,
        "transform_receipt_hash": binding.transform_receipt_hash,
    }


@dataclass(frozen=True)
class ReplayCompatibilityCheck:
    dimension: str
    status: str
    reason: str
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "dimension", _required(self.dimension, name="dimension"))
        status = _required(self.status, name="compatibility_status")
        if status not in REPLAY_COMPATIBILITY_CHECK_STATUSES_V1:
            raise ValueError(f"Unsupported replay compatibility status {status!r}.")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reason", _required(self.reason, name="reason"))
        object.__setattr__(self, "evidence_refs", tuple(str(v) for v in self.evidence_refs))


@dataclass(frozen=True)
class ReplayBindingAssessment:
    assessment_id: str
    binding_spec_id: str
    binding_spec_hash: str
    source_action_contract_id: str
    source_action_contract_hash: str
    target_runtime_contract_id: str
    target_runtime_contract_hash: str
    verdict: str
    checks: tuple[ReplayCompatibilityCheck, ...]
    permitted_uses: tuple[str, ...]
    blocking_reasons: tuple[str, ...]
    verifier_contract_id: str
    verifier_contract_hash: str
    semantic_content_hash: str

    def __post_init__(self) -> None:
        for name in (
            "assessment_id",
            "binding_spec_id",
            "binding_spec_hash",
            "source_action_contract_id",
            "source_action_contract_hash",
            "target_runtime_contract_id",
            "target_runtime_contract_hash",
            "verifier_contract_id",
            "verifier_contract_hash",
        ):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        verdict = _required(self.verdict, name="verdict")
        if verdict not in REPLAY_BINDING_VERDICTS_V1:
            raise ValueError(f"Unsupported replay binding verdict {verdict!r}.")
        object.__setattr__(self, "verdict", verdict)
        checks = tuple(self.checks)
        if not checks or any(not isinstance(check, ReplayCompatibilityCheck) for check in checks):
            raise ValueError("Replay binding assessment requires typed checks.")
        dimensions = tuple(check.dimension for check in checks)
        if len(set(dimensions)) != len(dimensions):
            raise ValueError("Replay binding assessment dimensions must be unique.")
        uses = tuple(dict.fromkeys(_required(value, name="permitted_use") for value in self.permitted_uses))
        if any(use not in REPLAY_BINDING_USES_V1 for use in uses):
            raise ValueError("Replay binding assessment contains an unsupported use.")
        reasons = tuple(str(value) for value in self.blocking_reasons)
        has_mismatch = any(check.status == REPLAY_COMPAT_MISMATCH for check in checks)
        has_unknown = any(check.status == REPLAY_COMPAT_UNKNOWN for check in checks)
        if verdict == REPLAY_BINDING_VERIFIED_COMPATIBLE and (has_mismatch or has_unknown):
            raise ValueError("Compatible replay assessment cannot contain mismatch/unknown checks.")
        if verdict == REPLAY_BINDING_VERIFIED_INCOMPATIBLE and not has_mismatch:
            raise ValueError("Incompatible replay assessment requires a mismatch check.")
        if verdict == REPLAY_BINDING_INDETERMINATE and (has_mismatch or not has_unknown):
            raise ValueError("Indeterminate replay assessment requires unknown and no mismatch.")
        object.__setattr__(self, "checks", checks)
        object.__setattr__(self, "permitted_uses", uses)
        object.__setattr__(self, "blocking_reasons", reasons)
        expected_hash = canonical_content_hash(
            replay_binding_assessment_payload(self),
            domain=REPLAY_BINDING_ASSESSMENT_HASH_DOMAIN_V1,
        )
        if self.semantic_content_hash != expected_hash:
            raise ValueError("Replay binding assessment hash mismatch.")
        if self.assessment_id != f"replay-binding-assessment:{expected_hash}":
            raise ValueError("Replay binding assessment ID/hash mismatch.")

    def allows(self, use: str) -> bool:
        return str(use) in self.permitted_uses


def replay_binding_assessment_payload(
    assessment: ReplayBindingAssessment,
) -> dict[str, Any]:
    return {
        "binding_spec_id": assessment.binding_spec_id,
        "binding_spec_hash": assessment.binding_spec_hash,
        "source_action_contract_id": assessment.source_action_contract_id,
        "source_action_contract_hash": assessment.source_action_contract_hash,
        "target_runtime_contract_id": assessment.target_runtime_contract_id,
        "target_runtime_contract_hash": assessment.target_runtime_contract_hash,
        "verdict": assessment.verdict,
        "checks": tuple(
            {
                "dimension": check.dimension,
                "status": check.status,
                "reason": check.reason,
                "evidence_refs": check.evidence_refs,
            }
            for check in assessment.checks
        ),
        "permitted_uses": assessment.permitted_uses,
        "blocking_reasons": assessment.blocking_reasons,
        "verifier_contract_id": assessment.verifier_contract_id,
        "verifier_contract_hash": assessment.verifier_contract_hash,
    }


@dataclass(frozen=True)
class CandidateBindingEligibility:
    eligibility_id: str
    assessment_id: str
    assessment_hash: str
    source_emission_id: str
    requested_use: str
    evaluation_allowed: bool
    authoritative_allowed: bool
    conditional_evaluation: bool
    same_legacy_emission: bool
    reasons: tuple[str, ...]
    target_runtime_contract_hash: str
    semantic_content_hash: str

    def __post_init__(self) -> None:
        for name in (
            "eligibility_id",
            "assessment_id",
            "assessment_hash",
            "source_emission_id",
            "target_runtime_contract_hash",
        ):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        use = _required(self.requested_use, name="requested_use")
        if use not in REPLAY_BINDING_USES_V1:
            raise ValueError(f"Unsupported replay binding use {use!r}.")
        object.__setattr__(self, "requested_use", use)
        object.__setattr__(self, "reasons", tuple(str(value) for value in self.reasons))
        if self.authoritative_allowed and use != REPLAY_BINDING_USE_EXECUTION_CANDIDATE:
            raise ValueError("Only an execution candidate can be authoritative-allowed.")
        if self.evaluation_allowed and use != REPLAY_BINDING_USE_PASSIVE_SHADOW:
            raise ValueError("Only passive-shadow use can be evaluation-allowed.")
        if self.conditional_evaluation and not (
            self.evaluation_allowed and self.same_legacy_emission
        ):
            raise ValueError("Conditional replay evaluation requires the same legacy emission.")
        expected_hash = canonical_content_hash(
            candidate_binding_eligibility_payload(self),
            domain=REPLAY_BINDING_ELIGIBILITY_HASH_DOMAIN_V1,
        )
        if self.semantic_content_hash != expected_hash:
            raise ValueError("Replay candidate eligibility hash mismatch.")
        if self.eligibility_id != f"replay-binding-eligibility:{expected_hash}":
            raise ValueError("Replay candidate eligibility ID/hash mismatch.")


def candidate_binding_eligibility_payload(
    eligibility: CandidateBindingEligibility,
) -> dict[str, Any]:
    return {
        "assessment_id": eligibility.assessment_id,
        "assessment_hash": eligibility.assessment_hash,
        "source_emission_id": eligibility.source_emission_id,
        "requested_use": eligibility.requested_use,
        "evaluation_allowed": eligibility.evaluation_allowed,
        "authoritative_allowed": eligibility.authoritative_allowed,
        "conditional_evaluation": eligibility.conditional_evaluation,
        "same_legacy_emission": eligibility.same_legacy_emission,
        "reasons": eligibility.reasons,
        "target_runtime_contract_hash": eligibility.target_runtime_contract_hash,
    }


@dataclass(frozen=True)
class ReplayExecutionAuthorization:
    authorization_id: str
    requested_mode: str
    assessment_id: str
    assessment_hash: str
    safety_preflight_id: str
    execution_capability_evidence_id: str
    dispatch_owner_state_id: str
    operator_authorization_id: str
    authorized: bool
    reasons: tuple[str, ...]
    semantic_content_hash: str

    def __post_init__(self) -> None:
        for name in (
            "authorization_id",
            "assessment_id",
            "assessment_hash",
            "safety_preflight_id",
            "execution_capability_evidence_id",
            "dispatch_owner_state_id",
            "operator_authorization_id",
        ):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        mode = _required(self.requested_mode, name="requested_mode")
        if mode not in REPLAY_EXECUTION_MODES_V1:
            raise ValueError(f"Unsupported replay execution mode {mode!r}.")
        object.__setattr__(self, "requested_mode", mode)
        object.__setattr__(self, "reasons", tuple(str(value) for value in self.reasons))
        expected_hash = canonical_content_hash(
            replay_execution_authorization_payload(self),
            domain=REPLAY_EXECUTION_AUTHORIZATION_HASH_DOMAIN_V1,
        )
        if self.semantic_content_hash != expected_hash:
            raise ValueError("Replay execution authorization hash mismatch.")
        if self.authorization_id != f"replay-execution-authorization:{expected_hash}":
            raise ValueError("Replay execution authorization ID/hash mismatch.")


def replay_execution_authorization_payload(
    authorization: ReplayExecutionAuthorization,
) -> dict[str, Any]:
    return {
        "requested_mode": authorization.requested_mode,
        "assessment_id": authorization.assessment_id,
        "assessment_hash": authorization.assessment_hash,
        "safety_preflight_id": authorization.safety_preflight_id,
        "execution_capability_evidence_id": (
            authorization.execution_capability_evidence_id
        ),
        "dispatch_owner_state_id": authorization.dispatch_owner_state_id,
        "operator_authorization_id": authorization.operator_authorization_id,
        "authorized": authorization.authorized,
        "reasons": authorization.reasons,
    }


@dataclass(frozen=True)
class ReplayActionRecord:
    replay_action_record_id: str
    source_contract_id: str
    source_contract_hash: str
    episode_id: str
    source_event_id: str
    source_action_index: int
    source_action_stage: str
    source_timestamp: float
    source_timestamp_clock_domain_id: str
    canonical_action_payload: Mapping[str, Any]
    canonical_action_hash: str
    schedule_id: str
    schedule_hash: str
    old_safety_profile_hash: str | None
    semantic_content_hash: str

    @classmethod
    def create(
        cls,
        *,
        source_contract: ReplaySourceContract,
        source_event_id: str,
        source_action_index: int,
        source_timestamp: float,
        source_timestamp_clock_domain_id: str,
        canonical_action_payload: Mapping[str, Any],
        canonical_action_hash: str,
        schedule: ReplayScheduleSpec,
    ) -> "ReplayActionRecord":
        if source_contract.replay_mode != REPLAY_MODE_CANONICAL_ACTION:
            raise ValueError("ReplayActionRecord requires canonical_action_replay mode.")
        if source_contract.action_stage not in REPLAY_CANONICAL_ACTION_STAGES_V1:
            raise ValueError("ReplayActionRecord source stage is not canonical-action eligible.")
        index = int(source_action_index)
        if index < 0:
            raise ValueError("source_action_index must be non-negative.")
        timestamp = _finite(source_timestamp, name="source_timestamp")
        frozen_payload = _freeze(dict(canonical_action_payload))
        payload = {
            "source_contract_id": source_contract.contract_id,
            "source_contract_hash": source_contract.semantic_content_hash,
            "episode_id": source_contract.episode_id,
            "source_event_id": _required(source_event_id, name="source_event_id"),
            "source_action_index": index,
            "source_action_stage": source_contract.action_stage,
            "source_timestamp": timestamp,
            "source_timestamp_clock_domain_id": _required(
                source_timestamp_clock_domain_id,
                name="source_timestamp_clock_domain_id",
            ),
            "canonical_action_payload": frozen_payload,
            "canonical_action_hash": _required(
                canonical_action_hash,
                name="canonical_action_hash",
            ),
            "schedule_id": schedule.schedule_id,
            "schedule_hash": schedule.semantic_content_hash,
            "old_safety_profile_hash": source_contract.old_safety_profile_hash,
        }
        semantic_hash = canonical_content_hash(payload, domain=REPLAY_ACTION_RECORD_HASH_DOMAIN_V1)
        return cls(
            replay_action_record_id=f"replay-action-record:{semantic_hash}",
            **payload,
            semantic_content_hash=semantic_hash,
        )


__all__ = [
    "REPLAY_ACTION_STAGE_COMMANDED",
    "REPLAY_ACTION_STAGE_PROPOSAL",
    "REPLAY_ACTION_STAGE_SAFE",
    "REPLAY_ACTION_STAGE_SELECTED",
    "REPLAY_ACTION_STAGES_V1",
    "REPLAY_BINDING_INDETERMINATE",
    "REPLAY_BINDING_USE_EXECUTION_CANDIDATE",
    "REPLAY_BINDING_USE_PASSIVE_SHADOW",
    "REPLAY_BINDING_USE_RECORD_ONLY",
    "REPLAY_BINDING_USES_V1",
    "REPLAY_BINDING_VERDICTS_V1",
    "REPLAY_BINDING_VERIFIED_COMPATIBLE",
    "REPLAY_BINDING_VERIFIED_INCOMPATIBLE",
    "REPLAY_EXECUTION_MODE_REAL_AUTHORITY",
    "REPLAY_EXECUTION_MODE_SIM_AUTHORITY",
    "REPLAY_EXECUTION_MODES_V1",
    "REPLAY_CANONICAL_ACTION_STAGES_V1",
    "REPLAY_COMPATIBILITY_CHECK_STATUSES_V1",
    "REPLAY_COMPAT_COMPATIBLE_SUBSET",
    "REPLAY_COMPAT_DERIVED_MATCH",
    "REPLAY_COMPAT_EXACT_MATCH",
    "REPLAY_COMPAT_MISMATCH",
    "REPLAY_COMPAT_NOT_APPLICABLE",
    "REPLAY_COMPAT_UNKNOWN",
    "REPLAY_COMPAT_VERIFIED_TRANSFORM",
    "REPLAY_MODE_CANONICAL_ACTION",
    "REPLAY_MODE_COMMAND_AUDIT",
    "REPLAY_MODE_DECISION",
    "REPLAY_MODE_POLICY_OBSERVATION",
    "REPLAY_MODES_V1",
    "REPLAY_SCHEDULE_FIXED_RATE",
    "REPLAY_SCHEDULE_MODES_V1",
    "REPLAY_SCHEDULE_NATIVE",
    "REPLAY_SCHEDULE_RESAMPLED",
    "REPLAY_SOURCE_INTERPRETATION_INVALID",
    "REPLAY_SOURCE_INTERPRETATION_STATUSES_V1",
    "REPLAY_SOURCE_INTERPRETATION_UNSUPPORTED",
    "REPLAY_SOURCE_INTERPRETATION_VERIFIED",
    "CandidateBindingEligibility",
    "LegacyReplayContractDerivationReceipt",
    "ReplayActionRecord",
    "ReplayBindingAssessment",
    "ReplayCompatibilityCheck",
    "ReplayExecutionAuthorization",
    "ReplayScheduleSpec",
    "ReplaySourceActionContract",
    "ReplaySourceContract",
    "ReplayTargetBindingSpec",
    "TargetRuntimeActionContract",
    "candidate_binding_eligibility_payload",
    "replay_execution_authorization_payload",
    "legacy_replay_contract_derivation_payload",
    "replay_binding_assessment_payload",
    "replay_source_action_contract_payload",
    "replay_schedule_spec_payload",
    "replay_source_contract_payload",
    "replay_target_binding_spec_payload",
    "target_runtime_action_contract_payload",
]
