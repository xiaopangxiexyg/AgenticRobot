"""Passive human-command emission and proposal adaptation contracts.

This module observes the existing retargeter output.  It does not call a
retargeter, apply safety, arbitrate, or dispatch a command.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import (
    ActionComponentCatalog,
    ActionProposalDraft,
    ArmActionSpec,
    HandActionSpec,
    RobotAction,
    ComponentActionSchema,
    SourceValidationResult,
    split_effector_name,
    project_robot_action_v1_to_component_action,
)
from robot_motion.action.proposal import freeze_robot_action, robot_action_semantic_payload
from robot_motion.runtime import EffectorCommand, TeleopCommand
from robot_motion.runtime.dispatch import CommandDispatchReceipt
from robot_motion.runtime.identity import RuntimeFactIdentity
from robot_motion.runtime.immutable import deep_freeze, deep_thaw
from robot_motion.runtime.observation_cut import ObservationSampleRef

from .adapters import TeleopCommandActionAdapterConfig, robot_action_from_teleop_command


HUMAN_SOURCE_ADAPTER_CONTRACT_ID_V1 = "teleop_command_to_action_proposal_v1"
HUMAN_SOURCE_ADAPTER_HASH_DOMAIN_V1 = "dexteleop.human_source_adapter.v1"
HUMAN_SOURCE_RECEIPT_HASH_DOMAIN_V1 = "dexteleop.human_source_adapter_receipt.v1"

_LOSSINESS_ORDER = {
    "lossless": 0,
    "lossy_but_explicit": 1,
    "unsupported": 2,
}
_NONCANONICAL_CONTROL_METADATA_KEYS = frozenset(
    {
        "backend_mode",
        "safety_bypass",
        "servo_mode",
    }
)


def _freeze_metadata(
    metadata: Mapping[str, Any],
) -> tuple[Mapping[str, Any], tuple[str, ...]]:
    frozen: dict[str, Any] = {}
    unsupported: list[str] = []
    for key, value in metadata.items():
        normalized_key = str(key)
        try:
            frozen[normalized_key] = deep_freeze(value)
        except TypeError:
            unsupported.append(normalized_key)
    return MappingProxyType(frozen), tuple(sorted(unsupported))


@dataclass(frozen=True)
class HumanEffectorCommandSnapshot:
    command_name: str
    effector_name: str
    side: str | None
    command_type: str
    value: Any
    frame_name: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    unfreezable_metadata_keys: tuple[str, ...] = ()

    @classmethod
    def capture(
        cls,
        command_name: str,
        command: EffectorCommand,
    ) -> "HumanEffectorCommandSnapshot":
        metadata, unsupported = _freeze_metadata(command.metadata)
        return cls(
            command_name=str(command_name),
            effector_name=str(command.effector_name),
            side=None if command.side is None else str(command.side),
            command_type=str(command.command_type),
            value=deep_freeze(command.value),
            frame_name=None if command.frame_name is None else str(command.frame_name),
            metadata=metadata,
            unfreezable_metadata_keys=unsupported,
        )

    def to_command(self) -> EffectorCommand:
        return EffectorCommand(
            effector_name=self.effector_name,
            side=self.side,
            command_type=self.command_type,
            value=deep_thaw(self.value),
            frame_name=self.frame_name,
            metadata=deep_thaw(self.metadata),
        )


@dataclass(frozen=True)
class HumanCommandContribution:
    contribution_index: int
    retargeter_name: str
    command_timestamp: float | None
    commands: tuple[HumanEffectorCommandSnapshot, ...]
    command_metadata: Mapping[str, Any] = field(default_factory=dict)
    unfreezable_metadata_keys: tuple[str, ...] = ()

    @classmethod
    def capture(
        cls,
        *,
        contribution_index: int,
        retargeter_name: str,
        command: TeleopCommand,
    ) -> "HumanCommandContribution":
        metadata, unsupported = _freeze_metadata(command.metadata)
        return cls(
            contribution_index=int(contribution_index),
            retargeter_name=str(retargeter_name),
            command_timestamp=(
                None if command.timestamp is None else float(command.timestamp)
            ),
            commands=tuple(
                HumanEffectorCommandSnapshot.capture(name, value)
                for name, value in command.commands.items()
            ),
            command_metadata=metadata,
            unfreezable_metadata_keys=unsupported,
        )


@dataclass(frozen=True)
class HumanCommandEmission:
    emission_sequence: int
    emitted_at: float
    command_timestamp: float | None
    contributions: tuple[HumanCommandContribution, ...]
    merged_commands: tuple[HumanEffectorCommandSnapshot, ...]
    winning_contribution_by_effector: Mapping[str, int]
    overwritten_contributions_by_effector: Mapping[str, tuple[int, ...]]
    merged_command_metadata: Mapping[str, Any] = field(default_factory=dict)
    unfreezable_merged_metadata_keys: tuple[str, ...] = ()
    observation_sample_refs: Mapping[str, ObservationSampleRef] = field(
        default_factory=dict
    )

    @classmethod
    def capture(
        cls,
        *,
        emission_sequence: int,
        emitted_at: float,
        contributions: Sequence[tuple[str, TeleopCommand]],
        merged_command: TeleopCommand,
        observation_sample_refs: Mapping[str, ObservationSampleRef] | None = None,
    ) -> "HumanCommandEmission":
        emitted = float(emitted_at)
        if int(emission_sequence) < 0 or not math.isfinite(emitted):
            raise ValueError("emission_sequence and emitted_at must be valid.")
        captured = tuple(
            HumanCommandContribution.capture(
                contribution_index=index,
                retargeter_name=name,
                command=command,
            )
            for index, (name, command) in enumerate(contributions)
        )
        seen: dict[str, list[int]] = {}
        for contribution in captured:
            for command in contribution.commands:
                seen.setdefault(command.effector_name, []).append(
                    contribution.contribution_index
                )
        winning = {name: indices[-1] for name, indices in seen.items()}
        overwritten = {
            name: tuple(indices[:-1])
            for name, indices in seen.items()
            if len(indices) > 1
        }
        metadata, unsupported = _freeze_metadata(merged_command.metadata)
        sample_refs = dict(observation_sample_refs or {})
        for stream_id, sample in sample_refs.items():
            if not isinstance(sample, ObservationSampleRef):
                raise TypeError(
                    f"observation_sample_refs[{stream_id!r}] must be "
                    "ObservationSampleRef."
                )
            if str(stream_id) != sample.stream_id:
                raise ValueError(
                    "observation sample mapping key must match sample.stream_id."
                )
        return cls(
            emission_sequence=int(emission_sequence),
            emitted_at=emitted,
            command_timestamp=(
                None
                if merged_command.timestamp is None
                else float(merged_command.timestamp)
            ),
            contributions=captured,
            merged_commands=tuple(
                HumanEffectorCommandSnapshot.capture(name, value)
                for name, value in merged_command.commands.items()
            ),
            winning_contribution_by_effector=MappingProxyType(winning),
            overwritten_contributions_by_effector=MappingProxyType(overwritten),
            merged_command_metadata=metadata,
            unfreezable_merged_metadata_keys=unsupported,
            observation_sample_refs=MappingProxyType(sample_refs),
        )

    def to_merged_command(self) -> TeleopCommand:
        command = TeleopCommand(
            timestamp=self.command_timestamp,
            metadata=deep_thaw(self.merged_command_metadata),
        )
        for snapshot in self.merged_commands:
            command.commands[snapshot.command_name] = snapshot.to_command()
        return command

    @property
    def semantic_content_hash(self) -> str:
        payload = {
                "emission_sequence": self.emission_sequence,
                "emitted_at": self.emitted_at,
                "command_timestamp": self.command_timestamp,
                "contributions": tuple(
                    {
                        "contribution_index": contribution.contribution_index,
                        "retargeter_name": contribution.retargeter_name,
                        "command_timestamp": contribution.command_timestamp,
                        "commands": tuple(
                            _effector_snapshot_payload(command)
                            for command in contribution.commands
                        ),
                        "command_metadata": contribution.command_metadata,
                        "unfreezable_metadata_keys": (
                            contribution.unfreezable_metadata_keys
                        ),
                    }
                    for contribution in self.contributions
                ),
                "merged_commands": tuple(
                    _effector_snapshot_payload(command)
                    for command in self.merged_commands
                ),
                "winning_contribution_by_effector": (
                    self.winning_contribution_by_effector
                ),
                "overwritten_contributions_by_effector": (
                    self.overwritten_contributions_by_effector
                ),
                "merged_command_metadata": self.merged_command_metadata,
                "unfreezable_merged_metadata_keys": (
                    self.unfreezable_merged_metadata_keys
                ),
            }
        if self.observation_sample_refs:
            payload["observation_sample_refs"] = {
                stream_id: _observation_sample_ref_payload(sample)
                for stream_id, sample in sorted(
                    self.observation_sample_refs.items()
                )
            }
        return canonical_content_hash(
            payload,
            domain="dexteleop.human_command_emission.v1",
        )

    @property
    def observation_sample_set_hash(self) -> str | None:
        if not self.observation_sample_refs:
            return None
        return canonical_content_hash(
            {
                stream_id: _observation_sample_ref_payload(sample)
                for stream_id, sample in sorted(
                    self.observation_sample_refs.items()
                )
            },
            domain="dexteleop.human_observation_sample_set.v1",
        )


@dataclass(frozen=True)
class HumanDispatchReceiptSnapshot:
    command_name: str
    receipt_kind: str
    effector_name: str | None
    accepted: bool | None
    operation: str | None
    status: str | None
    timestamp: float | None
    result_code: int | None
    result_text: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    unfreezable_metadata_keys: tuple[str, ...] = ()
    backend_receipt_type: str | None = None

    @classmethod
    def capture(
        cls,
        command_name: str,
        receipt: object | None,
    ) -> "HumanDispatchReceiptSnapshot":
        normalized_name = str(command_name).strip()
        if not normalized_name:
            raise ValueError("command_name must be non-empty.")
        if isinstance(receipt, CommandDispatchReceipt):
            metadata, unsupported = _freeze_metadata(receipt.metadata)
            return cls(
                command_name=normalized_name,
                receipt_kind="typed_command_dispatch_receipt",
                effector_name=receipt.effector_name,
                accepted=bool(receipt.accepted),
                operation=receipt.operation,
                status=receipt.status,
                timestamp=float(receipt.timestamp),
                result_code=(
                    None if receipt.result_code is None else int(receipt.result_code)
                ),
                result_text=receipt.result_text,
                metadata=metadata,
                unfreezable_metadata_keys=unsupported,
                backend_receipt_type=type(receipt).__name__,
            )
        return cls(
            command_name=normalized_name,
            receipt_kind=("legacy_none" if receipt is None else "backend_specific"),
            effector_name=None,
            accepted=None,
            operation=None,
            status=None,
            timestamp=None,
            result_code=None,
            result_text=None,
            metadata=MappingProxyType({}),
            backend_receipt_type=(
                None if receipt is None else type(receipt).__name__
            ),
        )


@dataclass(frozen=True)
class HumanLegacyCommandObservation:
    emission_sequence: int
    observed_at: float
    outcome: str
    command_dispatched: bool
    run_state: str
    guard_fault_code: str | None
    final_commands: tuple[HumanEffectorCommandSnapshot, ...]
    final_command_timestamp: float | None
    final_command_metadata: Mapping[str, Any]
    unfreezable_final_metadata_keys: tuple[str, ...]
    dispatch_receipts: tuple[HumanDispatchReceiptSnapshot, ...] = ()

    @classmethod
    def capture(
        cls,
        *,
        emission: HumanCommandEmission,
        observed_at: float,
        outcome: str,
        command_dispatched: bool,
        run_state: str,
        command: TeleopCommand,
        guard_fault_code: str | None = None,
        dispatch_receipts: Mapping[str, object | None] | None = None,
    ) -> "HumanLegacyCommandObservation":
        normalized_outcome = str(outcome).strip()
        if normalized_outcome not in {
            "guard_rejected",
            "dispatched",
            "not_dispatched",
        }:
            raise ValueError("Unsupported human legacy-command outcome.")
        timestamp = float(observed_at)
        if not math.isfinite(timestamp):
            raise ValueError("observed_at must be finite.")
        metadata, unsupported = _freeze_metadata(command.metadata)
        return cls(
            emission_sequence=emission.emission_sequence,
            observed_at=timestamp,
            outcome=normalized_outcome,
            command_dispatched=bool(command_dispatched),
            run_state=str(run_state),
            guard_fault_code=(
                None if guard_fault_code is None else str(guard_fault_code)
            ),
            final_commands=tuple(
                HumanEffectorCommandSnapshot.capture(name, value)
                for name, value in command.commands.items()
            ),
            final_command_timestamp=(
                None if command.timestamp is None else float(command.timestamp)
            ),
            final_command_metadata=metadata,
            unfreezable_final_metadata_keys=unsupported,
            dispatch_receipts=tuple(
                HumanDispatchReceiptSnapshot.capture(name, receipt)
                for name, receipt in sorted(
                    dict(dispatch_receipts or {}).items()
                )
            ),
        )

    def to_command(self) -> TeleopCommand:
        command = TeleopCommand(
            timestamp=self.final_command_timestamp,
            metadata=deep_thaw(self.final_command_metadata),
        )
        for snapshot in self.final_commands:
            command.commands[snapshot.command_name] = snapshot.to_command()
        return command


@dataclass(frozen=True)
class HumanCommandFilterObservation:
    """Immutable post-filter/pre-guard command paired to one human emission."""

    emission_sequence: int
    observed_at: float
    filtered_commands: tuple[HumanEffectorCommandSnapshot, ...]
    filtered_command_timestamp: float | None
    filtered_command_metadata: Mapping[str, Any]
    unfreezable_filtered_metadata_keys: tuple[str, ...]

    @classmethod
    def capture(
        cls,
        *,
        emission: HumanCommandEmission,
        observed_at: float,
        command: TeleopCommand,
    ) -> "HumanCommandFilterObservation":
        timestamp = float(observed_at)
        if not math.isfinite(timestamp):
            raise ValueError("observed_at must be finite.")
        metadata, unsupported = _freeze_metadata(command.metadata)
        return cls(
            emission_sequence=emission.emission_sequence,
            observed_at=timestamp,
            filtered_commands=tuple(
                HumanEffectorCommandSnapshot.capture(name, value)
                for name, value in command.commands.items()
            ),
            filtered_command_timestamp=(
                None if command.timestamp is None else float(command.timestamp)
            ),
            filtered_command_metadata=metadata,
            unfreezable_filtered_metadata_keys=unsupported,
        )

    def to_command(self) -> TeleopCommand:
        command = TeleopCommand(
            timestamp=self.filtered_command_timestamp,
            metadata=deep_thaw(self.filtered_command_metadata),
        )
        for snapshot in self.filtered_commands:
            command.commands[snapshot.command_name] = snapshot.to_command()
        return command


def _effector_snapshot_payload(
    command: HumanEffectorCommandSnapshot,
) -> dict[str, Any]:
    return {
        "command_name": command.command_name,
        "effector_name": command.effector_name,
        "side": command.side,
        "command_type": command.command_type,
        "value": command.value,
        "frame_name": command.frame_name,
        "metadata": command.metadata,
        "unfreezable_metadata_keys": command.unfreezable_metadata_keys,
    }


def _observation_sample_ref_payload(
    sample: ObservationSampleRef,
) -> dict[str, Any]:
    return {
        "stream_id": sample.stream_id,
        "sample_id": sample.sample_id,
        "sequence": int(sample.sequence),
        "measured_at": float(sample.measured_at),
        "measured_clock_domain_id": sample.measured_clock_domain_id,
        "available_at": float(sample.available_at),
        "available_clock_domain_id": sample.available_clock_domain_id,
        "payload_object_id": sample.payload_object_id,
        "valid": bool(sample.valid),
        "stale": bool(sample.stale),
        "control_admission_sequence": sample.control_admission_sequence,
        "content_metadata": sample.content_metadata,
    }


@dataclass(frozen=True)
class SourceComponentMappingReceipt:
    component_id: str | None
    selection_group_id: str | None
    effector_name: str
    contribution_index: int | None
    retargeter_name: str | None
    input_command_type: str
    input_frame: str | None
    output_representation: str | None
    output_reference_frame: str | None
    output_controlled_frame: str | None
    lossiness: str
    reasons: tuple[str, ...] = ()
    overwritten_contribution_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.lossiness not in _LOSSINESS_ORDER:
            raise ValueError("Unsupported source-adapter lossiness.")
        object.__setattr__(self, "reasons", tuple(str(value) for value in self.reasons))
        object.__setattr__(
            self,
            "overwritten_contribution_indices",
            tuple(int(value) for value in self.overwritten_contribution_indices),
        )


@dataclass(frozen=True)
class SourceAdapterReceipt:
    representation_trace_id: str
    adapter_contract_id: str
    adapter_contract_hash: str
    source_emission_id: str
    source_emission_sequence: int
    source_emission_semantic_hash: str
    component_mappings: tuple[SourceComponentMappingReceipt, ...]
    lossiness: str
    source_validation_accepted: bool
    authoritative_eligible: bool
    source_validation_reasons: tuple[str, ...]
    output_action_space: str | None
    output_action_semantic_hash: str | None
    semantic_content_hash: str


@dataclass(frozen=True)
class HumanProposalAdapterResult:
    proposal_draft: ActionProposalDraft | None
    robot_action: RobotAction | None
    source_validation: SourceValidationResult
    source_adapter_receipt: SourceAdapterReceipt


@dataclass(frozen=True)
class HumanLegacyActionAdapterResult:
    robot_action: RobotAction | None
    source_validation: SourceValidationResult
    component_mappings: tuple[SourceComponentMappingReceipt, ...]
    lossiness: str


@dataclass(frozen=True)
class HumanProposalAdapterConfig:
    runtime_session_id: str
    control_domain_id: str
    embodiment_instance_id: str
    producer_id: str
    producer_epoch: int
    produced_clock_domain_id: str
    effective_clock_domain_id: str
    component_catalog: ActionComponentCatalog
    action_schema: ComponentActionSchema
    action_adapter: TeleopCommandActionAdapterConfig = field(
        default_factory=lambda: TeleopCommandActionAdapterConfig(
            include_command_metadata=False,
            source="human",
        )
    )
    authority_priority: int = 0
    authority_request: str = "none"
    fallback_policy: str = "no_action"

    def __post_init__(self) -> None:
        for name in (
            "runtime_session_id",
            "control_domain_id",
            "embodiment_instance_id",
            "producer_id",
            "produced_clock_domain_id",
            "effective_clock_domain_id",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty.")
        if int(self.producer_epoch) < 0:
            raise ValueError("producer_epoch must be non-negative.")
        if not isinstance(self.action_schema, ComponentActionSchema):
            raise TypeError("action_schema must be a ComponentActionSchema.")


class HumanProposalAdapter:
    """Convert one passive pre-filter human emission into a proposal draft."""

    def __init__(self, config: HumanProposalAdapterConfig) -> None:
        self.config = config
        adapter = replace(
            config.action_adapter,
            include_command_metadata=False,
            metadata={},
            output_action_space=config.component_catalog.action_space,
        )
        self._action_adapter = adapter
        self._component_by_effector = {
            item.effector_name: item for item in config.component_catalog.components
        }
        self.contract_hash = canonical_content_hash(
            {
                "adapter_contract_id": HUMAN_SOURCE_ADAPTER_CONTRACT_ID_V1,
                "component_catalog_hash": config.component_catalog.semantic_content_hash,
                "arm_joint_policy": adapter.arm_joint_policy,
                "pose_metadata_keys": adapter.pose_metadata_keys,
                "hand_slot_dof": adapter.hand_slot_dof,
                "arm_specs": {
                    side: spec.to_metadata()
                    for side, spec in dict(adapter.arm_specs or {}).items()
                },
                "hand_specs": {
                    side: spec.to_metadata()
                    for side, spec in dict(adapter.hand_specs or {}).items()
                },
                "native_end_effector_action_space": (
                    adapter.native_end_effector_action_space
                ),
            },
            domain=HUMAN_SOURCE_ADAPTER_HASH_DOMAIN_V1,
        )

    def adapt(
        self,
        emission: HumanCommandEmission,
        *,
        effective_from: float,
        effective_until: float,
        target_effective_at: float,
        observation_cut_id: str | None = None,
    ) -> HumanProposalAdapterResult:
        normalized_observation_cut_id = (
            None
            if observation_cut_id is None
            else str(observation_cut_id).strip()
        )
        if observation_cut_id is not None and not normalized_observation_cut_id:
            raise ValueError("observation_cut_id must be non-empty when provided.")
        mappings = tuple(self._mapping_for(value, emission) for value in emission.merged_commands)
        reasons = tuple(
            reason
            for mapping in mappings
            for reason in mapping.reasons
            if mapping.lossiness == "unsupported"
        )
        action: RobotAction | None = None
        if not reasons and mappings:
            try:
                converted = robot_action_from_teleop_command(
                    emission.to_merged_command(),
                    config=self._action_adapter,
                )
                if converted is None:
                    reasons = ("empty_human_command",)
                else:
                    action = self._semantic_action(converted)
                    if action.action_space != self.config.component_catalog.action_space:
                        reasons = (
                            "action_space_catalog_mismatch:"
                            f"{action.action_space}!={self.config.component_catalog.action_space}",
                        )
                        action = None
            except (KeyError, TypeError, ValueError) as exc:
                reasons = (f"human_adapter_invalid:{type(exc).__name__}:{exc}",)
        elif not mappings:
            reasons = ("empty_human_command",)

        accepted = action is not None and not reasons
        validation = SourceValidationResult(accepted=accepted, reasons=reasons)
        lossiness = self._overall_lossiness(mappings, accepted=accepted)
        authoritative_eligible = bool(accepted and lossiness == "lossless")
        source_emission_identity = RuntimeFactIdentity(
            runtime_session_id=self.config.runtime_session_id,
            control_domain_id=self.config.control_domain_id,
            producer_id=self.config.producer_id,
            producer_epoch=self.config.producer_epoch,
            sequence=emission.emission_sequence,
            namespace="human_command_emission",
        )
        receipt = self._receipt(
            emission=emission,
            source_emission_id=source_emission_identity.fact_id,
            mappings=mappings,
            action=action,
            validation=validation,
            lossiness=lossiness,
            authoritative_eligible=authoritative_eligible,
        )
        proposal = None
        if accepted and action is not None:
            component_coverage = tuple(
                mapping.component_id
                for mapping in mappings
                if mapping.component_id is not None
            )
            selection_groups = tuple(
                dict.fromkeys(
                    mapping.selection_group_id
                    for mapping in mappings
                    if mapping.selection_group_id is not None
                )
            )
            proposal = ActionProposalDraft(
                runtime_session_id=self.config.runtime_session_id,
                control_domain_id=self.config.control_domain_id,
                embodiment_instance_id=self.config.embodiment_instance_id,
                producer_kind="human",
                producer_id=self.config.producer_id,
                producer_epoch=self.config.producer_epoch,
                sequence=emission.emission_sequence,
                action=project_robot_action_v1_to_component_action(
                    action,
                    action_schema=self.config.action_schema,
                    component_id_by_effector={
                        item.effector_name: item.component_id
                        for item in self.config.component_catalog.components
                    },
                ),
                produced_at=emission.emitted_at,
                produced_clock_domain_id=self.config.produced_clock_domain_id,
                effective_from=float(effective_from),
                effective_until=float(effective_until),
                target_effective_at=float(target_effective_at),
                effective_clock_domain_id=self.config.effective_clock_domain_id,
                authority_priority=int(self.config.authority_priority),
                authority_request=self.config.authority_request,
                validity=("valid" if authoritative_eligible else "shadow_only"),
                status="ready",
                fallback_policy=self.config.fallback_policy,
                observation_cut_id=normalized_observation_cut_id,
                representation_trace_id=receipt.representation_trace_id,
                audit_metadata={
                    "source_emission_id": source_emission_identity.fact_id,
                    "source_emission_semantic_hash": emission.semantic_content_hash,
                    **(
                        {
                            "source_observation_cut_id": (
                                normalized_observation_cut_id
                            )
                        }
                        if normalized_observation_cut_id is not None
                        else {}
                    ),
                    "source_observation_sample_set_hash": (
                        emission.observation_sample_set_hash
                    ),
                    "source_observation_sample_ids": tuple(
                        sample.sample_id
                        for _, sample in sorted(
                            emission.observation_sample_refs.items()
                        )
                    ),
                    "source_adapter_lossiness": lossiness,
                },
            )
        return HumanProposalAdapterResult(
            proposal_draft=proposal,
            robot_action=action,
            source_validation=validation,
            source_adapter_receipt=receipt,
        )

    def adapt_legacy_observation(
        self,
        observation: HumanLegacyCommandObservation,
    ) -> HumanLegacyActionAdapterResult:
        """Lift the final legacy command for parity only; never emit a proposal."""

        return self._adapt_observed_command(
            observation.to_command(),
            emission_sequence=observation.emission_sequence,
            observed_at=observation.observed_at,
            source_name="legacy_final_command",
        )

    def adapt_filter_observation(
        self,
        observation: HumanCommandFilterObservation,
    ) -> HumanLegacyActionAdapterResult:
        """Lift post-filter/pre-guard evidence without creating a proposal."""

        return self._adapt_observed_command(
            observation.to_command(),
            emission_sequence=observation.emission_sequence,
            observed_at=observation.observed_at,
            source_name="legacy_filtered_command",
        )

    def _adapt_observed_command(
        self,
        command: TeleopCommand,
        *,
        emission_sequence: int,
        observed_at: float,
        source_name: str,
    ) -> HumanLegacyActionAdapterResult:
        synthetic = HumanCommandEmission.capture(
            emission_sequence=emission_sequence,
            emitted_at=observed_at,
            contributions=((source_name, command),),
            merged_command=command,
        )
        mappings = tuple(
            self._mapping_for(value, synthetic) for value in synthetic.merged_commands
        )
        reasons = tuple(
            reason
            for mapping in mappings
            if mapping.lossiness == "unsupported"
            for reason in mapping.reasons
        )
        action: RobotAction | None = None
        if not reasons and mappings:
            try:
                converted = robot_action_from_teleop_command(
                    command,
                    config=self._action_adapter,
                )
                if converted is None:
                    reasons = ("empty_legacy_command",)
                else:
                    action = self._semantic_action(converted)
                    if action.action_space != self.config.component_catalog.action_space:
                        reasons = (
                            "action_space_catalog_mismatch:"
                            f"{action.action_space}!={self.config.component_catalog.action_space}",
                        )
                        action = None
            except (KeyError, TypeError, ValueError) as exc:
                reasons = (f"legacy_action_adapter_invalid:{type(exc).__name__}:{exc}",)
        elif not mappings:
            reasons = ("empty_legacy_command",)
        validation = SourceValidationResult(
            accepted=action is not None and not reasons,
            reasons=reasons,
        )
        return HumanLegacyActionAdapterResult(
            robot_action=action,
            source_validation=validation,
            component_mappings=mappings,
            lossiness=self._overall_lossiness(
                mappings,
                accepted=validation.accepted,
            ),
        )

    @staticmethod
    def _semantic_action(action: RobotAction) -> RobotAction:
        metadata = {
            key: action.metadata[key]
            for key in ("arm_action_specs_by_side", "hand_action_specs_by_side")
            if key in action.metadata
        }
        return freeze_robot_action(
            RobotAction(
                action_space=action.action_space,
                timestamp=action.timestamp,
                arm_pose_by_side=action.arm_pose_by_side,
                arm_qpos_by_side=action.arm_qpos_by_side,
                hand_qpos_by_side=action.hand_qpos_by_side,
                end_effector_targets_by_side=action.end_effector_targets_by_side,
                active_effectors=action.active_effectors,
                frame_by_effector=action.frame_by_effector,
                controlled_frame_by_effector=action.controlled_frame_by_effector,
                source=action.source,
                metadata=metadata,
            )
        )

    def _mapping_for(
        self,
        command: HumanEffectorCommandSnapshot,
        emission: HumanCommandEmission,
    ) -> SourceComponentMappingReceipt:
        reasons: list[str] = []
        lossiness = "lossless"
        component = self._component_by_effector.get(command.effector_name)
        try:
            side, kind = split_effector_name(command.effector_name)
        except ValueError as exc:
            side, kind = "", ""
            reasons.append(str(exc))
        if command.command_name != command.effector_name:
            reasons.append("command_key_effector_mismatch")
        if side and command.side != side:
            reasons.append("command_side_effector_mismatch")
        if component is None:
            reasons.append("effector_not_in_component_catalog")
        if command.unfreezable_metadata_keys:
            reasons.append(
                "unfreezable_metadata:" + ",".join(command.unfreezable_metadata_keys)
            )

        output_representation: str | None = None
        output_reference_frame: str | None = command.frame_name
        output_controlled_frame: str | None = None
        metadata = command.metadata
        if kind == "arm":
            if command.command_type == "cartesian_pose":
                output_representation = "absolute_xyz_quat"
                spec = self._arm_spec(side, metadata)
                if spec is None:
                    reasons.append("arm_pose_requires_explicit_arm_action_spec")
                else:
                    output_reference_frame = spec.reference_frame
                    output_controlled_frame = spec.controlled_frame
                    if command.frame_name != spec.reference_frame:
                        reasons.append("arm_reference_frame_mismatch")
                if not command.frame_name:
                    reasons.append("arm_pose_requires_explicit_reference_frame")
            elif command.command_type == "joint_vector":
                output_representation = "arm_qpos"
                if self._action_adapter.arm_joint_policy == "source_pose_when_available":
                    pose_keys = tuple(
                        key
                        for key in self._action_adapter.pose_metadata_keys
                        if key in metadata
                    )
                    if pose_keys:
                        output_representation = "absolute_xyz_quat_from_command_metadata"
                        lossiness = "lossy_but_explicit"
                        reasons.append("joint_target_replaced_by_explicit_pose_metadata")
            else:
                reasons.append(f"unsupported_arm_command_type:{command.command_type}")
        elif kind == "hand":
            if command.command_type == "joint_vector":
                output_representation = "hand_qpos"
                if self._hand_spec(side, metadata) is None:
                    reasons.append("hand_joint_vector_requires_explicit_hand_action_spec")
            elif command.command_type in {
                "end_effector_continuous",
                "end_effector_discrete",
            }:
                output_representation = command.command_type
                if not str(metadata.get("end_effector_adapter") or "").strip():
                    reasons.append("native_end_effector_requires_explicit_adapter")
            else:
                reasons.append(f"unsupported_hand_command_type:{command.command_type}")

        control_keys = sorted(_NONCANONICAL_CONTROL_METADATA_KEYS.intersection(metadata))
        if control_keys and not reasons:
            lossiness = "lossy_but_explicit"
            reasons.append("noncanonical_control_metadata:" + ",".join(control_keys))
        if any(
            not reason.startswith("joint_target_replaced_by_explicit_pose_metadata")
            and not reason.startswith("noncanonical_control_metadata:")
            for reason in reasons
        ):
            lossiness = "unsupported"

        winner = emission.winning_contribution_by_effector.get(command.effector_name)
        retargeter_name = None
        if winner is not None and 0 <= winner < len(emission.contributions):
            retargeter_name = emission.contributions[winner].retargeter_name
        return SourceComponentMappingReceipt(
            component_id=None if component is None else component.component_id,
            selection_group_id=(
                None if component is None else component.selection_group_id
            ),
            effector_name=command.effector_name,
            contribution_index=winner,
            retargeter_name=retargeter_name,
            input_command_type=command.command_type,
            input_frame=command.frame_name,
            output_representation=output_representation,
            output_reference_frame=output_reference_frame,
            output_controlled_frame=output_controlled_frame,
            lossiness=lossiness,
            reasons=tuple(reasons),
            overwritten_contribution_indices=emission.overwritten_contributions_by_effector.get(
                command.effector_name, ()
            ),
        )

    def _arm_spec(
        self,
        side: str,
        metadata: Mapping[str, Any],
    ) -> ArmActionSpec | None:
        configured = dict(self._action_adapter.arm_specs or {}).get(side)
        if configured is not None:
            return configured
        payload = metadata.get("arm_action_spec")
        if isinstance(payload, ArmActionSpec):
            return payload
        if isinstance(payload, Mapping):
            try:
                return ArmActionSpec.from_metadata(payload)
            except (KeyError, TypeError, ValueError):
                return None
        return None

    def _hand_spec(
        self,
        side: str,
        metadata: Mapping[str, Any],
    ) -> HandActionSpec | None:
        configured = dict(self._action_adapter.hand_specs or {}).get(side)
        if configured is not None:
            return configured
        payload = metadata.get("hand_action_spec")
        if isinstance(payload, HandActionSpec):
            return payload
        if isinstance(payload, Mapping):
            try:
                return HandActionSpec.from_metadata(payload)
            except (KeyError, TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _overall_lossiness(
        mappings: tuple[SourceComponentMappingReceipt, ...],
        *,
        accepted: bool,
    ) -> str:
        if not accepted or not mappings:
            return "unsupported"
        return max(
            (mapping.lossiness for mapping in mappings),
            key=_LOSSINESS_ORDER.__getitem__,
        )

    def _receipt(
        self,
        *,
        emission: HumanCommandEmission,
        source_emission_id: str,
        mappings: tuple[SourceComponentMappingReceipt, ...],
        action: RobotAction | None,
        validation: SourceValidationResult,
        lossiness: str,
        authoritative_eligible: bool,
    ) -> SourceAdapterReceipt:
        action_hash = (
            None
            if action is None
            else canonical_content_hash(
                robot_action_semantic_payload(action),
                domain="dexteleop.robot_action.semantic.v1",
            )
        )
        payload = {
            "adapter_contract_id": HUMAN_SOURCE_ADAPTER_CONTRACT_ID_V1,
            "adapter_contract_hash": self.contract_hash,
            "source_emission_id": source_emission_id,
            "source_emission_sequence": emission.emission_sequence,
            "source_emission_semantic_hash": emission.semantic_content_hash,
            "component_mappings": tuple(
                _component_mapping_payload(mapping) for mapping in mappings
            ),
            "lossiness": lossiness,
            "source_validation_accepted": validation.accepted,
            "authoritative_eligible": authoritative_eligible,
            "source_validation_reasons": validation.reasons,
            "output_action_space": None if action is None else action.action_space,
            "output_action_semantic_hash": action_hash,
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=HUMAN_SOURCE_RECEIPT_HASH_DOMAIN_V1,
        )
        return SourceAdapterReceipt(
            representation_trace_id=f"human-source-representation:{semantic_hash}",
            adapter_contract_id=HUMAN_SOURCE_ADAPTER_CONTRACT_ID_V1,
            adapter_contract_hash=self.contract_hash,
            source_emission_id=source_emission_id,
            source_emission_sequence=emission.emission_sequence,
            source_emission_semantic_hash=emission.semantic_content_hash,
            component_mappings=mappings,
            lossiness=lossiness,
            source_validation_accepted=validation.accepted,
            authoritative_eligible=authoritative_eligible,
            source_validation_reasons=validation.reasons,
            output_action_space=None if action is None else action.action_space,
            output_action_semantic_hash=action_hash,
            semantic_content_hash=semantic_hash,
        )


def _component_mapping_payload(
    mapping: SourceComponentMappingReceipt,
) -> dict[str, Any]:
    return {
        "component_id": mapping.component_id,
        "selection_group_id": mapping.selection_group_id,
        "effector_name": mapping.effector_name,
        "contribution_index": mapping.contribution_index,
        "retargeter_name": mapping.retargeter_name,
        "input_command_type": mapping.input_command_type,
        "input_frame": mapping.input_frame,
        "output_representation": mapping.output_representation,
        "output_reference_frame": mapping.output_reference_frame,
        "output_controlled_frame": mapping.output_controlled_frame,
        "lossiness": mapping.lossiness,
        "reasons": mapping.reasons,
        "overwritten_contribution_indices": (
            mapping.overwritten_contribution_indices
        ),
    }


__all__ = [
    "HUMAN_SOURCE_ADAPTER_CONTRACT_ID_V1",
    "HumanCommandContribution",
    "HumanCommandEmission",
    "HumanCommandFilterObservation",
    "HumanDispatchReceiptSnapshot",
    "HumanLegacyCommandObservation",
    "HumanLegacyActionAdapterResult",
    "HumanEffectorCommandSnapshot",
    "HumanProposalAdapter",
    "HumanProposalAdapterConfig",
    "HumanProposalAdapterResult",
    "SourceAdapterReceipt",
    "SourceComponentMappingReceipt",
]
