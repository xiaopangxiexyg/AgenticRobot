"""Passive frontend wiring for dataset replay proposal shadowing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from dataset_contracts.replay import (
    REPLAY_SCHEDULE_FIXED_RATE,
    ReplayScheduleSpec,
    TargetRuntimeActionContract,
)
from dataset_interface.replay import (
    derive_replay_source_action_contract,
    replay_source_contract_from_metadata,
)
from robot_motion.deploy.runtime import PolicyActionEmission
from robot_motion.replay.trajectory import ReplayTrajectoryGroup
from robot_motion.replay.proposal_source import (
    ReplayProposalAdapterResult,
    ReplayProposalSource,
    ReplayProposalSourceConfig,
    ReplaySourceFrameEmission,
    build_replay_source_frame_emission,
)
from .replay_binding import build_replay_binding_assessment


@dataclass(frozen=True)
class DatasetReplayProposalShadowConfig:
    runtime_session_id: str
    control_domain_id: str
    embodiment_instance_id: str
    producer_id: str
    producer_epoch: int
    component_schema_hash: str
    component_coverage: tuple[str, ...]
    source_path: str
    episode_id: str
    action_field: str
    source_action_space: str
    source_hz: float
    source_start_index: int
    dataset_metadata: Mapping[str, Any]
    target_frame_contract_hash: str | None
    target_execution_capability_profile_hash: str | None = None


class DatasetReplayProposalShadowAdapter:
    """Converts an already-popped policy emission into replay provenance.

    It never calls ``predict``, reads/mutates the replay cursor, or pops the
    policy action queue.
    """

    def __init__(self, config: DatasetReplayProposalShadowConfig) -> None:
        if not isinstance(config, DatasetReplayProposalShadowConfig):
            raise TypeError("config must be DatasetReplayProposalShadowConfig.")
        source_contract = replay_source_contract_from_metadata(
            config.dataset_metadata,
            source_path=Path(config.source_path),
            episode_id=config.episode_id,
            action_field=config.action_field,
            source_action_space=config.source_action_space,
        )
        schedule = ReplayScheduleSpec(
            schedule_id=(
                f"debug-replay:{source_contract.source_dataset_id}:"
                f"{source_contract.episode_id}:fixed-rate"
            ),
            schedule_mode=REPLAY_SCHEDULE_FIXED_RATE,
            source_hz=float(config.source_hz),
            target_hz=None,
            resampling_policy="none",
            source_start_index=int(config.source_start_index),
        )
        source_action_contract, self.derivation_receipt = (
            derive_replay_source_action_contract(
                source_contract,
                component_ids=config.component_coverage,
                selection_groups=config.component_coverage,
            )
        )
        target_contract = TargetRuntimeActionContract(
            target_contract_id=(
                f"replay-shadow-target:{config.control_domain_id}:"
                f"{config.embodiment_instance_id}:{config.component_schema_hash}"
            ),
            control_domain_id=config.control_domain_id,
            embodiment_instance_id=config.embodiment_instance_id,
            component_schema_hash=config.component_schema_hash,
            component_ids=config.component_coverage,
            selection_group_by_component={
                component: component for component in config.component_coverage
            },
            accepted_action_spaces_by_component={
                component: (config.source_action_space,)
                for component in config.component_coverage
            },
            pose_frame_contract_hash=config.target_frame_contract_hash,
            units_convention_id="si_m_rad_xyzw_v1",
            execution_capability_profile_hash=(
                config.target_execution_capability_profile_hash
            ),
            unresolved_dimensions=(
                ("pose_frame_contract",)
                if config.target_frame_contract_hash is None
                else ()
            ),
        )
        binding, assessment = build_replay_binding_assessment(
            source_action_contract,
            target_contract,
            binding_id=(
                f"debug-replay-target:{config.embodiment_instance_id}:"
                f"{config.component_schema_hash}"
            ),
        )
        self.config = config
        self.source_action_contract = source_action_contract
        self.target_runtime_contract = target_contract
        self.target_binding_spec = binding
        self.binding_assessment = assessment
        self.source = ReplayProposalSource(
            ReplayProposalSourceConfig(
                runtime_session_id=config.runtime_session_id,
                control_domain_id=config.control_domain_id,
                embodiment_instance_id=config.embodiment_instance_id,
                producer_id=config.producer_id,
                producer_epoch=config.producer_epoch,
                component_schema_hash=config.component_schema_hash,
                component_coverage=config.component_coverage,
                selection_groups=config.component_coverage,
                source_contract=source_contract,
                source_action_contract=source_action_contract,
                target_runtime_contract=target_contract,
                derivation_receipt=self.derivation_receipt,
                schedule=schedule,
                target_binding_spec=binding,
                binding_assessment=assessment,
            )
        )
        self._next_sequence = 0

    def adapt_emission(
        self,
        emission: PolicyActionEmission,
        *,
        raw_action_metadata: Mapping[str, Any],
    ) -> ReplayProposalAdapterResult:
        try:
            source_index = int(raw_action_metadata["debug_replay_source_action_index"])
            local_index = int(raw_action_metadata["debug_replay_local_action_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Dataset replay shadow requires typed source/local action indices "
                "from the unique queue-pop emission seam."
            ) from exc
        if source_index != int(self.config.source_start_index) + local_index:
            raise ValueError("Dataset replay source/local action indices disagree.")
        sequence = self._next_sequence
        self._next_sequence += 1
        return self.source.adapt_policy_emission(
            emission,
            source_action_index=source_index,
            sequence=sequence,
        )

    def adapt_trajectory_frame(
        self,
        group: ReplayTrajectoryGroup,
        *,
        index: int,
        include_arms: bool,
        include_end_effectors: bool,
        produced_at: float,
        produced_clock_domain_id: str,
        effective_from: float,
        effective_until: float,
        target_effective_at: float,
        effective_clock_domain_id: str,
    ) -> ReplayProposalAdapterResult:
        """Adapt the existing hardware-free replay action-frame seam.

        The real replay executor may call this passively in a later owner
        migration; this method itself does not start or schedule replay.
        """

        local_index = int(index)
        if local_index < 0 or local_index >= int(group.frame_count):
            raise IndexError("Replay trajectory frame is outside the group.")
        if Path(group.source_path).expanduser().resolve() != Path(
            self.config.source_path
        ).expanduser().resolve():
            raise ValueError("Replay trajectory source_path does not match the bound source.")
        if int(group.start_index) != int(self.config.source_start_index):
            raise ValueError("Replay trajectory start_index does not match the bound schedule.")
        emission = build_replay_source_frame_emission(
            group,
            active_sides=tuple(
                dict.fromkeys(
                    str(effector).split("_", 1)[0]
                    for effector in self.config.component_coverage
                )
            ),
            index=local_index,
            include_arms=include_arms,
            include_end_effectors=include_end_effectors,
            observed_at=produced_at,
            schedule_start_at=float(target_effective_at) - group.relative_time_sec(local_index),
            clock_domain_id=effective_clock_domain_id,
        )
        if not math.isclose(
            emission.target_effective_at,
            float(target_effective_at),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Replay source-frame target time does not match the caller schedule.")
        if not math.isclose(
            emission.effective_from,
            float(effective_from),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Replay source-frame effective_from does not match the caller schedule.")
        if not math.isclose(
            emission.effective_until,
            float(effective_until),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Replay source-frame effective_until does not match the caller schedule.")
        if str(produced_clock_domain_id) != str(effective_clock_domain_id):
            raise ValueError("Replay source-frame adapter currently requires one local clock domain.")
        sequence = self._next_sequence
        self._next_sequence += 1
        return self.source.adapt_source_frame_emission(emission, sequence=sequence)

    def adapt_source_frame_emission(
        self,
        emission: ReplaySourceFrameEmission,
    ) -> ReplayProposalAdapterResult:
        """Consume an immutable, nonblocking executor observer emission."""

        if Path(emission.source_path).expanduser().resolve() != Path(
            self.config.source_path
        ).expanduser().resolve():
            raise ValueError("Replay source-frame emission is from another bound source.")
        sequence = self._next_sequence
        self._next_sequence += 1
        return self.source.adapt_source_frame_emission(emission, sequence=sequence)


def replay_adapter_debug_payload(result: ReplayProposalAdapterResult) -> dict[str, Any]:
    receipt = result.source_adapter_receipt
    return {
        "representation_trace_id": receipt.representation_trace_id,
        "source_contract_id": receipt.source_contract_id,
        "source_contract_hash": receipt.source_contract_hash,
        "replay_action_record_id": receipt.replay_action_record_id,
        "replay_action_record_hash": receipt.replay_action_record_hash,
        "source_event_id": receipt.source_event_id,
        "source_emission_id": receipt.source_emission_id,
        "source_action_index": receipt.source_action_index,
        "source_action_stage": receipt.source_action_stage,
        "target_binding_spec_id": receipt.target_binding_spec_id,
        "target_binding_spec_hash": receipt.target_binding_spec_hash,
        "binding_assessment_id": receipt.binding_assessment_id,
        "binding_assessment_hash": receipt.binding_assessment_hash,
        "binding_verdict": receipt.binding_verdict,
        "binding_eligibility_id": receipt.binding_eligibility_id,
        "binding_eligibility_hash": receipt.binding_eligibility_hash,
        "accepted": receipt.accepted,
        "evaluation_allowed": receipt.evaluation_allowed,
        "authoritative_eligible": receipt.authoritative_eligible,
        "conditional_evaluation": receipt.conditional_evaluation,
        "old_safe_does_not_bypass_current_safety": (
            receipt.old_safe_does_not_bypass_current_safety
        ),
        "rejection_reasons": receipt.rejection_reasons,
        "output_action_semantic_hash": receipt.output_action_semantic_hash,
        "semantic_content_hash": receipt.semantic_content_hash,
        "proposal_draft_present": result.proposal_draft is not None,
    }


__all__ = [
    "DatasetReplayProposalShadowAdapter",
    "DatasetReplayProposalShadowConfig",
    "replay_adapter_debug_payload",
]
