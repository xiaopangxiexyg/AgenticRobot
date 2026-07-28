"""Hardware-free canonical replay action -> proposal adapter.

The adapter is passive: it owns no reader cursor, action queue, scheduler,
Arbiter, safety pass, or dispatcher.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.replay import (
    REPLAY_ACTION_STAGE_SAFE,
    REPLAY_BINDING_USE_PASSIVE_SHADOW,
    REPLAY_MODE_CANONICAL_ACTION,
    REPLAY_SOURCE_INTERPRETATION_VERIFIED,
    CandidateBindingEligibility,
    LegacyReplayContractDerivationReceipt,
    ReplayActionRecord,
    ReplayBindingAssessment,
    ReplayScheduleSpec,
    ReplaySourceActionContract,
    ReplaySourceContract,
    ReplayTargetBindingSpec,
    TargetRuntimeActionContract,
)
from robot_motion.action.proposal import (
    ROBOT_ACTION_SEMANTIC_METADATA_KEYS_V1,
    ActionProposalDraft,
    freeze_robot_action,
    robot_action_semantic_payload,
)
from robot_motion.action.component_action import ComponentActionSchema
from robot_motion.action.component_action_v1 import (
    project_robot_action_v1_to_component_action,
)
from robot_motion.action.schema import RobotAction
from robot_motion.deploy.runtime import PolicyActionEmission
from robot_motion.replay.action_frame import replay_robot_action_at_index
from robot_motion.replay.trajectory import ReplayTrajectoryGroup
from robot_motion.replay.compatibility import build_candidate_binding_eligibility


REPLAY_SOURCE_ADAPTER_CONTRACT_ID_V1 = "replay-canonical-action-adapter-v1"
REPLAY_SOURCE_ADAPTER_RECEIPT_HASH_DOMAIN_V1 = (
    "dexteleop.replay_source_adapter_receipt.v1"
)
REPLAY_SOURCE_FRAME_EMISSION_HASH_DOMAIN_V1 = (
    "dexteleop.replay_source_frame_emission.v1"
)


def _clean_replay_action(action: RobotAction) -> RobotAction:
    """Move replay/debug provenance out of execution metadata."""

    semantic_metadata = {
        key: value
        for key, value in action.metadata.items()
        if key in ROBOT_ACTION_SEMANTIC_METADATA_KEYS_V1
    }
    cleaned = RobotAction(
        action_space=action.action_space,
        timestamp=action.timestamp,
        arm_pose_by_side=action.arm_pose_by_side,
        arm_qpos_by_side=action.arm_qpos_by_side,
        hand_qpos_by_side=action.hand_qpos_by_side,
        end_effector_targets_by_side=action.end_effector_targets_by_side,
        active_effectors=action.active_effectors,
        frame_by_effector=action.frame_by_effector,
        controlled_frame_by_effector=action.controlled_frame_by_effector,
        source="replay",
        metadata=semantic_metadata,
    )
    return freeze_robot_action(cleaned)


@dataclass(frozen=True)
class ReplaySourceAdapterReceipt:
    representation_trace_id: str
    adapter_contract_id: str
    source_contract_id: str
    source_contract_hash: str
    replay_action_record_id: str | None
    replay_action_record_hash: str | None
    source_event_id: str
    source_emission_id: str
    source_action_index: int
    source_action_stage: str
    target_binding_spec_id: str
    target_binding_spec_hash: str
    binding_assessment_id: str
    binding_assessment_hash: str
    binding_verdict: str
    binding_eligibility_id: str
    binding_eligibility_hash: str
    accepted: bool
    evaluation_allowed: bool
    authoritative_eligible: bool
    conditional_evaluation: bool
    old_safe_does_not_bypass_current_safety: bool
    rejection_reasons: tuple[str, ...]
    output_action_semantic_hash: str | None
    semantic_content_hash: str


@dataclass(frozen=True)
class ReplayProposalAdapterResult:
    replay_action_record: ReplayActionRecord | None
    source_action_contract: ReplaySourceActionContract
    target_runtime_contract: TargetRuntimeActionContract
    target_binding_spec: ReplayTargetBindingSpec
    derivation_receipt: LegacyReplayContractDerivationReceipt | None
    binding_assessment: ReplayBindingAssessment
    binding_eligibility: CandidateBindingEligibility
    source_adapter_receipt: ReplaySourceAdapterReceipt
    proposal_draft: ActionProposalDraft | None


@dataclass(frozen=True)
class ReplaySourceFrameEmission:
    emission_id: str
    source_path: str
    source_episode_index: int
    group_key: str
    local_frame_index: int
    source_action_index: int
    source_relative_time: float
    action: RobotAction
    observed_at: float
    target_effective_at: float
    effective_from: float
    effective_until: float
    clock_domain_id: str
    semantic_content_hash: str


def create_replay_source_frame_emission(
    *,
    source_path: str,
    source_episode_index: int,
    group_key: str,
    local_frame_index: int,
    source_action_index: int,
    source_relative_time: float,
    action: RobotAction,
    observed_at: float,
    target_effective_at: float,
    effective_until: float,
    clock_domain_id: str,
) -> ReplaySourceFrameEmission:
    cleaned_action = _clean_replay_action(action)
    target_at = float(target_effective_at)
    interval_end = float(effective_until)
    if interval_end <= target_at:
        raise ValueError("Replay source-frame effective interval must be non-empty.")
    payload = {
        "source_path": str(source_path),
        "source_episode_index": int(source_episode_index),
        "group_key": str(group_key),
        "local_frame_index": int(local_frame_index),
        "source_action_index": int(source_action_index),
        "source_relative_time": float(source_relative_time),
        "action": robot_action_semantic_payload(cleaned_action),
        "observed_at": float(observed_at),
        "target_effective_at": target_at,
        "effective_from": target_at,
        "effective_until": interval_end,
        "clock_domain_id": str(clock_domain_id),
    }
    semantic_hash = canonical_content_hash(
        payload,
        domain=REPLAY_SOURCE_FRAME_EMISSION_HASH_DOMAIN_V1,
    )
    return ReplaySourceFrameEmission(
        emission_id=f"replay-source-frame-emission:{semantic_hash}",
        **{key: value for key, value in payload.items() if key != "action"},
        action=cleaned_action,
        semantic_content_hash=semantic_hash,
    )


def build_replay_source_frame_emission(
    group: ReplayTrajectoryGroup,
    *,
    active_sides: tuple[str, ...],
    index: int,
    include_arms: bool,
    include_end_effectors: bool,
    observed_at: float,
    schedule_start_at: float,
    target_effective_at: float | None = None,
    effective_until_at: float | None = None,
    clock_domain_id: str = "local_monotonic",
) -> ReplaySourceFrameEmission:
    local_index = int(index)
    relative_time = group.relative_time_sec(local_index)
    target_at = (
        float(schedule_start_at) + relative_time
        if target_effective_at is None
        else float(target_effective_at)
    )
    if local_index + 1 < int(group.frame_count):
        next_relative = group.relative_time_sec(local_index + 1)
    else:
        rate = float(group.action_hz)
        if rate <= 0.0:
            raise ValueError("Replay final-frame validity requires positive action_hz.")
        next_relative = relative_time + 1.0 / rate
    action = _clean_replay_action(
        replay_robot_action_at_index(
            group,
            active_sides,
            index=local_index,
            include_arms=include_arms,
            include_end_effectors=include_end_effectors,
            source="replay_source_frame",
            timestamp=target_at,
        )
    )
    effective_until = (
        float(schedule_start_at) + next_relative
        if effective_until_at is None
        else float(effective_until_at)
    )
    if effective_until <= target_at:
        raise ValueError("Replay source-frame effective interval must be non-empty.")
    return create_replay_source_frame_emission(
        source_path=str(group.source_path),
        source_episode_index=int(group.source_episode_index),
        group_key=str(group.key),
        local_frame_index=local_index,
        source_action_index=int(group.start_index) + local_index,
        source_relative_time=relative_time,
        action=action,
        observed_at=float(observed_at),
        target_effective_at=target_at,
        effective_until=effective_until,
        clock_domain_id=str(clock_domain_id),
    )


@dataclass(frozen=True)
class ReplayProposalSourceConfig:
    runtime_session_id: str
    control_domain_id: str
    embodiment_instance_id: str
    producer_id: str
    producer_epoch: int
    action_schema: ComponentActionSchema
    source_contract: ReplaySourceContract
    source_action_contract: ReplaySourceActionContract
    target_runtime_contract: TargetRuntimeActionContract
    derivation_receipt: LegacyReplayContractDerivationReceipt | None
    schedule: ReplayScheduleSpec
    target_binding_spec: ReplayTargetBindingSpec
    binding_assessment: ReplayBindingAssessment
    requested_use: str = REPLAY_BINDING_USE_PASSIVE_SHADOW
    authority_priority: int = 100
    fallback_policy: str = "no_action"

    def __post_init__(self) -> None:
        for name in (
            "runtime_session_id",
            "control_domain_id",
            "embodiment_instance_id",
            "producer_id",
            "fallback_policy",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty.")
        if self.source_contract.replay_mode != REPLAY_MODE_CANONICAL_ACTION:
            raise ValueError("ReplayProposalSource only accepts canonical_action_replay.")
        if self.source_action_contract.action_stage != self.source_contract.action_stage:
            raise ValueError("Replay source dataset/action contracts disagree on stage.")
        if self.source_action_contract.action_space != self.source_contract.source_action_space:
            raise ValueError("Replay source dataset/action contracts disagree on action space.")
        if self.source_action_contract.pose_frame_contract_hash != (
            self.source_contract.source_frame_contract_hash
        ):
            raise ValueError("Replay source dataset/action contracts disagree on pose frame.")
        if self.source_action_contract.component_schema_hash != (
            self.source_contract.source_component_schema_hash
        ):
            raise ValueError("Replay source dataset/action contracts disagree on component schema.")
        if self.target_runtime_contract.control_domain_id != self.control_domain_id:
            raise ValueError("Replay target/control domains disagree.")
        if self.target_runtime_contract.embodiment_instance_id != (
            self.embodiment_instance_id
        ):
            raise ValueError("Replay target/embodiment instances disagree.")
        if not isinstance(self.action_schema, ComponentActionSchema):
            raise TypeError("action_schema must be a ComponentActionSchema.")
        if self.target_binding_spec.source_action_contract_hash != (
            self.source_action_contract.semantic_content_hash
        ):
            raise ValueError("Replay target binding does not reference the source action contract.")
        if self.target_binding_spec.target_runtime_contract_hash != (
            self.target_runtime_contract.semantic_content_hash
        ):
            raise ValueError("Replay target binding does not reference the target runtime contract.")
        if self.binding_assessment.binding_spec_hash != (
            self.target_binding_spec.semantic_content_hash
        ):
            raise ValueError("Replay binding assessment does not reference the binding spec.")
        if int(self.producer_epoch) < 0:
            raise ValueError("producer_epoch must be non-negative.")
        if not set(self.source_action_contract.component_ids).issubset(
            self.action_schema.component_ids
        ):
            raise ValueError("Replay source action contract/component coverage disagree.")


class ReplayProposalSource:
    def __init__(self, config: ReplayProposalSourceConfig) -> None:
        if not isinstance(config, ReplayProposalSourceConfig):
            raise TypeError("config must be ReplayProposalSourceConfig.")
        self.config = config

    def _eligibility(
        self,
        *,
        source_emission_id: str,
        same_legacy_emission: bool,
    ) -> CandidateBindingEligibility:
        return build_candidate_binding_eligibility(
            self.config.binding_assessment,
            source_emission_id=source_emission_id,
            requested_use=self.config.requested_use,
            same_legacy_emission=bool(same_legacy_emission),
            source_interpretation_verified=(
                self.config.source_action_contract.interpretation_status
                == REPLAY_SOURCE_INTERPRETATION_VERIFIED
            ),
        )

    def _receipt(
        self,
        *,
        source_event_id: str,
        source_emission_id: str,
        source_action_index: int,
        record: ReplayActionRecord | None,
        action_hash: str | None,
        eligibility: CandidateBindingEligibility,
        rejection_reasons: tuple[str, ...],
    ) -> ReplaySourceAdapterReceipt:
        proposal_allowed = bool(
            eligibility.evaluation_allowed or eligibility.authoritative_allowed
        )
        accepted = bool(
            not rejection_reasons
            and record is not None
            and action_hash is not None
            and proposal_allowed
        )
        payload = {
            "adapter_contract_id": REPLAY_SOURCE_ADAPTER_CONTRACT_ID_V1,
            "source_contract_id": self.config.source_contract.contract_id,
            "source_contract_hash": self.config.source_contract.semantic_content_hash,
            "replay_action_record_id": (
                None if record is None else record.replay_action_record_id
            ),
            "replay_action_record_hash": (
                None if record is None else record.semantic_content_hash
            ),
            "source_event_id": str(source_event_id),
            "source_emission_id": str(source_emission_id),
            "source_action_index": int(source_action_index),
            "source_action_stage": self.config.source_contract.action_stage,
            "target_binding_spec_id": self.config.target_binding_spec.binding_id,
            "target_binding_spec_hash": self.config.target_binding_spec.semantic_content_hash,
            "binding_assessment_id": self.config.binding_assessment.assessment_id,
            "binding_assessment_hash": self.config.binding_assessment.semantic_content_hash,
            "binding_verdict": self.config.binding_assessment.verdict,
            "binding_eligibility_id": eligibility.eligibility_id,
            "binding_eligibility_hash": eligibility.semantic_content_hash,
            "accepted": accepted,
            "evaluation_allowed": eligibility.evaluation_allowed,
            "authoritative_eligible": eligibility.authoritative_allowed,
            "conditional_evaluation": eligibility.conditional_evaluation,
            "old_safe_does_not_bypass_current_safety": (
                self.config.source_contract.action_stage == REPLAY_ACTION_STAGE_SAFE
            ),
            "rejection_reasons": rejection_reasons,
            "output_action_semantic_hash": action_hash,
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=REPLAY_SOURCE_ADAPTER_RECEIPT_HASH_DOMAIN_V1,
        )
        return ReplaySourceAdapterReceipt(
            representation_trace_id=f"replay-source-adapter:{semantic_hash}",
            **payload,
            semantic_content_hash=semantic_hash,
        )

    def adapt_action(
        self,
        action: RobotAction,
        *,
        source_event_id: str,
        source_emission_id: str,
        source_action_index: int,
        source_timestamp: float,
        source_timestamp_clock_domain_id: str,
        sequence: int,
        produced_at: float,
        produced_clock_domain_id: str,
        effective_from: float,
        effective_until: float,
        target_effective_at: float,
        effective_clock_domain_id: str,
        same_legacy_emission: bool = False,
        chunk_id: str | None = None,
        chunk_keyframe_index: int | None = None,
    ) -> ReplayProposalAdapterResult:
        reasons: list[str] = []
        if action.action_space != self.config.source_action_contract.action_space:
            reasons.append("source_action_space_mismatch")
        if set(action.active_effectors) != set(
            self.config.source_action_contract.component_ids
        ):
            reasons.append("component_coverage_mismatch")

        cleaned_action: RobotAction | None = None
        action_payload: Mapping[str, Any] | None = None
        action_hash: str | None = None
        if not reasons:
            try:
                cleaned_action = _clean_replay_action(action)
                action_payload = robot_action_semantic_payload(cleaned_action)
                action_hash = canonical_content_hash(
                    action_payload,
                    domain="dexteleop.robot_action.semantic.v1",
                )
            except (TypeError, ValueError) as exc:
                reasons.append(f"action_validation_failed:{type(exc).__name__}")

        record: ReplayActionRecord | None = None
        if action_payload is not None and action_hash is not None:
            record = ReplayActionRecord.create(
                source_contract=self.config.source_contract,
                source_event_id=source_event_id,
                source_action_index=int(source_action_index),
                source_timestamp=float(source_timestamp),
                source_timestamp_clock_domain_id=source_timestamp_clock_domain_id,
                canonical_action_payload=action_payload,
                canonical_action_hash=action_hash,
                schedule=self.config.schedule,
            )
        eligibility = self._eligibility(
            source_emission_id=source_emission_id,
            same_legacy_emission=same_legacy_emission,
        )
        if not eligibility.evaluation_allowed and not eligibility.authoritative_allowed:
            reasons.extend(eligibility.reasons or ("binding_use_not_allowed",))
        receipt = self._receipt(
            source_event_id=source_event_id,
            source_emission_id=source_emission_id,
            source_action_index=source_action_index,
            record=record,
            action_hash=action_hash,
            eligibility=eligibility,
            rejection_reasons=tuple(dict.fromkeys(reasons)),
        )
        if not receipt.accepted or record is None or cleaned_action is None:
            return ReplayProposalAdapterResult(
                record,
                self.config.source_action_contract,
                self.config.target_runtime_contract,
                self.config.target_binding_spec,
                self.config.derivation_receipt,
                self.config.binding_assessment,
                eligibility,
                receipt,
                None,
            )

        draft = ActionProposalDraft(
            runtime_session_id=self.config.runtime_session_id,
            control_domain_id=self.config.control_domain_id,
            embodiment_instance_id=self.config.embodiment_instance_id,
            producer_kind="replay",
            producer_id=self.config.producer_id,
            producer_epoch=self.config.producer_epoch,
            sequence=int(sequence),
            action=project_robot_action_v1_to_component_action(
                cleaned_action,
                action_schema=self.config.action_schema,
            ),
            produced_at=float(produced_at),
            produced_clock_domain_id=str(produced_clock_domain_id),
            effective_from=float(effective_from),
            effective_until=float(effective_until),
            target_effective_at=float(target_effective_at),
            effective_clock_domain_id=str(effective_clock_domain_id),
            authority_priority=int(self.config.authority_priority),
            fallback_policy=self.config.fallback_policy,
            chunk_id=chunk_id,
            chunk_keyframe_index=chunk_keyframe_index,
            representation_trace_id=receipt.representation_trace_id,
            audit_metadata={
                "replay_action_record_id": record.replay_action_record_id,
                "replay_source_emission_id": str(source_emission_id),
            },
        )
        return ReplayProposalAdapterResult(
            record,
            self.config.source_action_contract,
            self.config.target_runtime_contract,
            self.config.target_binding_spec,
            self.config.derivation_receipt,
            self.config.binding_assessment,
            eligibility,
            receipt,
            draft,
        )

    def adapt_policy_emission(
        self,
        emission: PolicyActionEmission,
        *,
        source_action_index: int,
        sequence: int,
        source_timestamp: float | None = None,
    ) -> ReplayProposalAdapterResult:
        if not isinstance(emission, PolicyActionEmission):
            raise TypeError("adapt_policy_emission requires PolicyActionEmission.")
        action = emission.pre_safety_robot_action
        if (
            action is None
            or emission.source_validation is None
            or not emission.source_validation.accepted
            or emission.pre_safety_faults
        ):
            eligibility = self._eligibility(
                source_emission_id=emission.emission_id,
                same_legacy_emission=True,
            )
            receipt = self._receipt(
                source_event_id=(
                    f"{self.config.source_contract.episode_id}:frame:{int(source_action_index)}"
                ),
                source_emission_id=emission.emission_id,
                source_action_index=source_action_index,
                record=None,
                action_hash=None,
                eligibility=eligibility,
                rejection_reasons=("pre_safety_action_unavailable_or_invalid",),
            )
            return ReplayProposalAdapterResult(
                None,
                self.config.source_action_contract,
                self.config.target_runtime_contract,
                self.config.target_binding_spec,
                self.config.derivation_receipt,
                self.config.binding_assessment,
                eligibility,
                receipt,
                None,
            )
        timestamp = (
            self.config.schedule.relative_time_for_index(source_action_index)
            if source_timestamp is None
            else float(source_timestamp)
        )
        return self.adapt_action(
            action,
            source_event_id=(
                f"{self.config.source_contract.episode_id}:frame:{int(source_action_index)}"
            ),
            source_emission_id=emission.emission_id,
            source_action_index=int(source_action_index),
            source_timestamp=timestamp,
            source_timestamp_clock_domain_id="capture_relative",
            sequence=int(sequence),
            produced_at=emission.produced_at,
            produced_clock_domain_id=emission.produced_clock_domain_id,
            effective_from=emission.effective_from,
            effective_until=emission.effective_until,
            target_effective_at=emission.target_effective_at,
            effective_clock_domain_id=emission.effective_clock_domain_id,
            same_legacy_emission=True,
            chunk_id=f"replay:{self.config.source_contract.contract_id}:{emission.chunk_id}",
            chunk_keyframe_index=emission.chunk_keyframe_index,
        )

    def adapt_source_frame_emission(
        self,
        emission: ReplaySourceFrameEmission,
        *,
        sequence: int,
    ) -> ReplayProposalAdapterResult:
        if not isinstance(emission, ReplaySourceFrameEmission):
            raise TypeError("emission must be ReplaySourceFrameEmission.")
        return self.adapt_action(
            emission.action,
            source_event_id=(
                f"{self.config.source_contract.episode_id}:"
                f"frame:{emission.source_action_index}"
            ),
            source_emission_id=emission.emission_id,
            source_action_index=emission.source_action_index,
            source_timestamp=emission.source_relative_time,
            source_timestamp_clock_domain_id="capture_relative",
            sequence=int(sequence),
            produced_at=emission.observed_at,
            produced_clock_domain_id=emission.clock_domain_id,
            effective_from=emission.effective_from,
            effective_until=emission.effective_until,
            target_effective_at=emission.target_effective_at,
            effective_clock_domain_id=emission.clock_domain_id,
            same_legacy_emission=True,
            chunk_id=(
                f"replay-group:{emission.group_key}:"
                f"{emission.source_episode_index}"
            ),
            chunk_keyframe_index=emission.local_frame_index,
        )


__all__ = [
    "REPLAY_SOURCE_ADAPTER_CONTRACT_ID_V1",
    "ReplayProposalAdapterResult",
    "ReplayProposalSource",
    "ReplayProposalSourceConfig",
    "ReplaySourceAdapterReceipt",
    "ReplaySourceFrameEmission",
    "build_replay_source_frame_emission",
    "create_replay_source_frame_emission",
]
