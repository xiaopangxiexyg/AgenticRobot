"""Thin MuJoCo binding for the backend-neutral authoritative control runtime."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

from dataset_contracts.control_trace import ControlTraceFact
from dataset_interface.composition_control_trace import (
    CompositionControlTraceEmitter,
)
from control_composition.activation import CompositionActivationScope
from control_composition.information import CompositionInformationPlan
from control_composition.eligibility import (
    CompositionSourceEligibilityDraft,
    CompositionSourceEligibilitySpec,
)
from control_composition.composition import (
    CompositionContextDraft,
    CompositionInputDraft,
    CompositionSpec,
    FormalCompositionStrategy,
)
from control_composition.observability import SourceDifferenceSpec
from control_composition.retention import TimeWindowRetentionSpec
from control_composition.runtime import (
    CompositeProposalSubmission,
    ControlCompositionRuntime,
)
from dataset_contracts.replay import (
    REPLAY_EXECUTION_MODE_SIM_AUTHORITY,
    TargetRuntimeActionContract,
)
from robot_motion.action import (
    BIMANUAL_EE_POSE_V1,
    BIMANUAL_QPOS_V1,
    ActionComponentCatalog,
    ActionComponentSpec,
    ActionPersistencePolicy,
    ActionProposalDraft,
    CandidateAuthorizationAdmissionReceipt,
    CompoundDispatchManifest,
    ConservativeCompoundAuthorityProfile,
    AuthorityPolicySpec,
    AuthoritativeControlConfig,
    AuthoritativeControlRuntime,
    AuthoritativeControlStep,
    AuthoritativeCommandDispatchContext,
    DecisionScheduleSpec,
    IMMEDIATE_QREF_SIM_V1,
    HandActionSpec,
    RobotAction,
    SCHEDULED_REFERENCE_SEGMENT_SIM_V1,
    ReferenceActivationSpec,
    ExecutionGateTransitionOutcome,
    build_immediate_qref_activation_spec,
    build_compound_dispatch_manifest,
    build_scheduled_reference_segment_activation_spec,
    validate_reference_activation_binding,
)
from robot_motion.action.proposal import ProposalAdmissionReceipt
from robot_motion.replay import (
    ReplayProposalAdapterResult,
    ReplayProposalSource,
    ReplaySourceFrameEmission,
    build_replay_execution_authorization,
    build_target_runtime_action_contract,
    create_replay_source_frame_emission,
)
from robot_motion.replay.control_trace import (
    project_replay_adapter_result,
    project_replay_source_frame_emission,
)
from robot_motion.runtime.identity import ClockDomainDescriptor
from robot_motion.runtime.control_admission import ControlCut
from robot_motion.runtime.observation_cut import ObservationBundle
from robot_motion.position_control import ExecutionCapabilityProfile
from robot_motion.deploy import (
    ActionChunkExecutionPolicy,
    PolicyActionEmission,
    project_policy_action_emission,
)

from ..adapters import MujocoPositionExecutionRig
from .measurement import (
    MujocoQposMeasurement,
    build_mujoco_backend_measurement,
    capture_mujoco_qpos_measurement,
)
from .observation import MujocoPolicyObservationAdapter
from .reference_receiver import MujocoScheduledQposReferenceReceiver
from .pose_resolution import (
    MujocoCoupledPoseToQposResolver,
    build_mujoco_pose_qpos_execution_target,
)
from .resolved_reference_receiver import (
    MujocoScheduledResolvedQposReferenceReceiver,
)
from .reference_trace import (
    project_execution_target_admission,
    project_group_generation_activation,
    project_group_generation_reservation,
    project_planner_command_frame,
    project_reference_receiver_validation,
    project_reference_segment,
    project_reference_segment_admission,
    project_reference_segment_terminal,
    project_reference_target_activation,
    project_resolved_execution_target,
    project_safety_anchor_commit_evidence,
    project_safety_anchor_commit_candidate,
)


@dataclass(frozen=True)
class MujocoDeployConfig:
    runtime_session_id: str = "mujoco-deploy-session"
    control_domain_id: str = "mujoco-env-0"
    embodiment_instance_id: str = "mujoco-rm65-pair-0"
    owner_runtime_id: str = "mujoco-deploy-control"
    producer_id: str = "policy-main"
    composer_producer_id: str = "policy-composer"
    enable_composer_authority: bool = False
    producer_epoch: int = 1
    policy_node_id: str = "single-policy"
    replay_producer_id: str = "replay-main"
    enable_replay_authority: bool = False
    decision_hz: float = 30.0
    reference_activation_profile_id: str = IMMEDIATE_QREF_SIM_V1
    selected_segment_max_lifetime_sec: float = 0.1
    proposal_horizon_sec: float = 0.1
    clock_domain_id: str = "mujoco.control.monotonic"
    boot_id: str = "mujoco-process"
    # ``None`` preserves single-action/full-chunk compatibility.  When a
    # policy submits a chunk, only this prefix is admitted for execution.
    # Keep this new option after the historical fields so positional config
    # construction retains its pre-chunk meaning.
    action_chunk_execution_steps: int | None = None
    authority_policy_id: str = "mujoco-single-policy-authority-v1"
    initial_control_mode: str = "policy"
    # Set only for the explicit RM65 pose→coupled-IK→qpos execution profile.
    # The default qpos profile keeps this unset and retains its existing hash.
    pose_frame_contract_hash: str | None = None
    # Explicitly empty for all existing single-source profiles.  A scoped
    # multi-source composition must name every producer that may own a typed
    # authority lease; priority alone is never treated as a handoff protocol.
    authority_takeover_allowed_producers: tuple[str, ...] = ()
    # Empty preserves the existing arms-only catalog and hashes.  The first
    # executable compound profile intentionally accepts only existing
    # DM-Tac q6 gripper specs; dexterous-hand drivers remain a later profile.
    compound_hand_action_specs_by_side: Mapping[
        str, HandActionSpec | Mapping[str, object]
    ] = field(default_factory=dict)
    # Default-on preserves every existing runtime.  A formal composition
    # profile sets this false so raw contributor/policy proposals cannot race
    # the single composer proposal for actuator authority.
    enable_primary_policy_authority: bool = True
    # Optional, additive declaration of context/difference information needed
    # by the composer. It does not select dataset streams or change authority.
    composition_information_plan: CompositionInformationPlan | None = None
    # Explicitly opt-in shadow source-liveness admission. Existing MuJoCo
    # composition profiles keep their action-only behavior and hashes.
    enable_composition_source_eligibility: bool = False
    # Default-off formal gate for composer-owned compound motion.  The typed
    # profile binds exact composition semantics and an all-required scope; it
    # is intentionally independent from any model/checkpoint identity.
    compound_composer_authority_profile: (
        ConservativeCompoundAuthorityProfile | None
    ) = None

    def __post_init__(self) -> None:
        for name in (
            "runtime_session_id",
            "control_domain_id",
            "embodiment_instance_id",
            "owner_runtime_id",
            "producer_id",
            "composer_producer_id",
            "policy_node_id",
            "replay_producer_id",
            "clock_domain_id",
            "boot_id",
            "authority_policy_id",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty.")
        object.__setattr__(
            self,
            "action_chunk_execution_steps",
            ActionChunkExecutionPolicy(
                self.action_chunk_execution_steps
            ).execution_steps,
        )
        if int(self.producer_epoch) < 0:
            raise ValueError("producer_epoch must be non-negative.")
        for name in (
            "enable_primary_policy_authority",
            "enable_composer_authority",
            "enable_replay_authority",
            "enable_composition_source_eligibility",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool.")
        if not any(
            (
                self.enable_primary_policy_authority,
                self.enable_composer_authority,
                self.enable_replay_authority,
            )
        ):
            raise ValueError("At least one MuJoCo proposal authority must be enabled.")
        if self.reference_activation_profile_id not in {
            IMMEDIATE_QREF_SIM_V1,
            SCHEDULED_REFERENCE_SEGMENT_SIM_V1,
        }:
            raise ValueError("Unsupported MuJoCo reference activation profile.")
        if self.initial_control_mode not in {
            "human",
            "policy",
            "replay",
            "hybrid",
            "shadow",
            "hold",
            "fault",
        }:
            raise ValueError("Unsupported initial_control_mode.")
        if self.pose_frame_contract_hash is not None and not str(
            self.pose_frame_contract_hash
        ).strip():
            raise ValueError("pose_frame_contract_hash must be non-empty when set.")
        allowed = tuple(
            str(value).strip()
            for value in self.authority_takeover_allowed_producers
        )
        if any(not value for value in allowed) or len(set(allowed)) != len(allowed):
            raise ValueError(
                "authority_takeover_allowed_producers must be non-empty and unique."
            )
        object.__setattr__(
            self,
            "authority_takeover_allowed_producers",
            allowed,
        )
        hand_specs: dict[str, HandActionSpec] = {}
        for raw_side, raw_spec in self.compound_hand_action_specs_by_side.items():
            side = str(raw_side).strip()
            spec = (
                raw_spec
                if isinstance(raw_spec, HandActionSpec)
                else HandActionSpec.from_metadata(raw_spec)
            )
            if spec.side != side:
                raise ValueError("Compound hand spec key/side mismatch.")
            if (
                spec.hand_model != "dmtac_gripper"
                or spec.slot_count != 6
                or spec.active_dof != 1
            ):
                raise ValueError(
                    "The current executable MuJoCo compound profile supports "
                    "only explicit DM-Tac q6 gripper specs."
                )
            hand_specs[side] = spec
        object.__setattr__(
            self,
            "compound_hand_action_specs_by_side",
            MappingProxyType(dict(sorted(hand_specs.items()))),
        )
        if self.composition_information_plan is not None and not isinstance(
            self.composition_information_plan,
            CompositionInformationPlan,
        ):
            raise TypeError(
                "composition_information_plan must be a "
                "CompositionInformationPlan or None."
            )
        if (
            self.compound_composer_authority_profile is not None
            and not isinstance(
                self.compound_composer_authority_profile,
                ConservativeCompoundAuthorityProfile,
            )
        ):
            raise TypeError(
                "compound_composer_authority_profile must be a "
                "ConservativeCompoundAuthorityProfile or None."
            )
        rate = float(self.decision_hz)
        if not math.isfinite(rate) or rate <= 0.0 or rate > 60.0:
            raise ValueError("deployable sim ActionArbiter rate must be in (0, 60].")
        for name in (
            "selected_segment_max_lifetime_sec",
            "proposal_horizon_sec",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")


@dataclass(frozen=True)
class MujocoReplaySubmission:
    source_emission: ReplaySourceFrameEmission
    adapter_result: ReplayProposalAdapterResult
    proposal_admission: ProposalAdmissionReceipt | None
    authorization_admission: CandidateAuthorizationAdmissionReceipt | None


@dataclass(frozen=True)
class MujocoPolicyActionChunkSubmission:
    """Admission receipts for the executable prefix of one policy chunk."""

    chunk_id: str
    source_chunk_size: int
    executed_chunk_size: int
    action_chunk_execution_steps: int | None
    receipts: tuple[ProposalAdmissionReceipt, ...]

    def __post_init__(self) -> None:
        chunk_id = str(self.chunk_id).strip()
        source_chunk_size = int(self.source_chunk_size)
        executed_chunk_size = int(self.executed_chunk_size)
        receipts = tuple(self.receipts)
        if not chunk_id:
            raise ValueError("chunk_id must be non-empty.")
        if source_chunk_size <= 0:
            raise ValueError("source_chunk_size must be positive.")
        if not 0 < executed_chunk_size <= source_chunk_size:
            raise ValueError(
                "executed_chunk_size must be positive and no larger than source_chunk_size."
            )
        if len(receipts) != executed_chunk_size:
            raise ValueError(
                "receipts length must match executed_chunk_size."
            )
        object.__setattr__(self, "chunk_id", chunk_id)
        object.__setattr__(self, "source_chunk_size", source_chunk_size)
        object.__setattr__(self, "executed_chunk_size", executed_chunk_size)
        object.__setattr__(
            self,
            "action_chunk_execution_steps",
            ActionChunkExecutionPolicy(self.action_chunk_execution_steps).execution_steps,
        )
        object.__setattr__(self, "receipts", receipts)

    @property
    def accepted_count(self) -> int:
        return sum(1 for receipt in self.receipts if receipt.accepted)

    @property
    def accepted(self) -> bool:
        return bool(self.receipts) and self.accepted_count == len(self.receipts)


@dataclass(frozen=True)
class MujocoPolicyEmissionSubmission:
    """Result of passively admitting one lower-owned queue-pop emission."""

    source_emission: PolicyActionEmission
    proposal_admission: ProposalAdmissionReceipt | None

    @property
    def accepted(self) -> bool:
        return bool(
            self.proposal_admission is not None
            and self.proposal_admission.accepted
        )


def _bimanual_qpos_catalog(
    hand_sides: Sequence[str] = (),
) -> ActionComponentCatalog:
    sides = tuple(dict.fromkeys(str(side) for side in hand_sides))
    return ActionComponentCatalog(
        catalog_id=(
            "mujoco-bimanual-qpos-v1"
            if not sides
            else "mujoco-bimanual-qpos-dmtac-compound-v1"
        ),
        action_space=BIMANUAL_QPOS_V1,
        components=(
            ActionComponentSpec("left_arm", "arms", "left_arm", coupled=True),
            ActionComponentSpec("right_arm", "arms", "right_arm", coupled=True),
            *(
                ActionComponentSpec(
                    f"{side}_hand",
                    f"{side}_hand",
                    f"{side}_hand",
                    coupled=False,
                )
                for side in sides
            ),
        ),
    )


def _bimanual_pose_catalog(
    hand_sides: Sequence[str] = (),
) -> ActionComponentCatalog:
    sides = tuple(dict.fromkeys(str(side) for side in hand_sides))
    return ActionComponentCatalog(
        catalog_id=(
            "mujoco-bimanual-tool-tcp-pose-v1"
            if not sides
            else "mujoco-bimanual-tool-tcp-pose-dmtac-compound-v1"
        ),
        action_space=BIMANUAL_EE_POSE_V1,
        components=(
            ActionComponentSpec("left_arm", "arms", "left_arm", coupled=True),
            ActionComponentSpec("right_arm", "arms", "right_arm", coupled=True),
            *(
                ActionComponentSpec(
                    f"{side}_hand",
                    f"{side}_hand",
                    f"{side}_hand",
                    coupled=False,
                )
                for side in sides
            ),
        ),
    )


class MujocoDeployRuntime:
    """One deterministic producer→Arbiter→MuJoCo authority chain."""

    def __init__(
        self,
        rig: MujocoPositionExecutionRig,
        config: MujocoDeployConfig = MujocoDeployConfig(),
        *,
        trace_sink: Callable[[ControlTraceFact], None] | None = None,
        pose_resolver: MujocoCoupledPoseToQposResolver | None = None,
        shared_safety=None,
        composition_strategy: FormalCompositionStrategy | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.rig = rig
        self.config = config
        self.clock = clock
        self.pose_resolver = pose_resolver
        self._pose_execution_enabled = pose_resolver is not None
        self._explicit_shared_safety = shared_safety is not None
        self._compound_hand_specs = dict(
            config.compound_hand_action_specs_by_side
        )
        self._compound_execution_enabled = bool(self._compound_hand_specs)
        self._compound_composer_authority_profile = (
            config.compound_composer_authority_profile
        )
        if self._compound_execution_enabled:
            if not self._pose_execution_enabled:
                raise ValueError(
                    "Compound hand execution requires the pose execution profile."
                )
            if shared_safety is None:
                raise ValueError(
                    "Compound hand execution requires an explicit shared_safety "
                    "implementation."
                )
            if config.enable_replay_authority:
                raise ValueError(
                    "The conservative compound profile does not enable replay "
                    "authority."
                )
            if (
                config.enable_composer_authority
                and self._compound_composer_authority_profile is None
            ):
                raise ValueError(
                    "Compound composer authority requires an explicit "
                    "ConservativeCompoundAuthorityProfile."
                )
            if (
                self._compound_composer_authority_profile is not None
                and not config.enable_composer_authority
            ):
                raise ValueError(
                    "A compound composer authority profile requires "
                    "enable_composer_authority=True."
                )
            if (
                self._compound_composer_authority_profile is not None
                and config.enable_primary_policy_authority
            ):
                raise ValueError(
                    "Conservative compound composer authority requires raw "
                    "primary policy authority to be disabled."
                )
            gripper_profile = rig.config.gripper_actuation_profile
            if gripper_profile is None:
                raise ValueError(
                    "Compound DM-Tac execution requires a configured MuJoCo "
                    "gripper actuation profile."
                )
            if set(self._compound_hand_specs) != set(
                gripper_profile.side_by_gripper.values()
            ):
                raise ValueError(
                    "Compound hand specs must exactly match configured MuJoCo "
                    "gripper sides."
                )
        elif self._compound_composer_authority_profile is not None:
            raise ValueError(
                "Compound composer authority requires explicit compound hand specs."
            )
        if self._compound_composer_authority_profile is not None:
            if composition_strategy is None:
                raise ValueError(
                    "Compound composer authority requires a formal component "
                    "composition strategy."
                )
            strategy_spec = composition_strategy.spec
            component_rule_set = getattr(
                composition_strategy,
                "component_rule_set",
                None,
            )
            if component_rule_set is None:
                raise ValueError(
                    "Compound composer authority requires a typed component "
                    "rule set."
                )
            profile = self._compound_composer_authority_profile
            if (
                strategy_spec.composition_spec_hash
                != profile.composition_spec_hash
                or strategy_spec.semantic_content_hash
                != profile.formal_strategy_spec_hash
                or component_rule_set.semantic_content_hash
                != profile.component_rule_set_hash
            ):
                raise ValueError(
                    "Compound composer authority profile does not bind the "
                    "configured formal composition strategy."
                )
        if self._pose_execution_enabled:
            if config.pose_frame_contract_hash is None:
                raise ValueError(
                    "Pose execution requires an explicit pose_frame_contract_hash."
                )
            if (
                config.reference_activation_profile_id
                != SCHEDULED_REFERENCE_SEGMENT_SIM_V1
            ):
                raise ValueError(
                    "Pose execution v1 requires scheduled qpos reference segments."
                )
            transformer = pose_resolver.pose_transformer
            if transformer is not None and (
                transformer.binding_receipt.target_pose_frame_contract_hash
                != config.pose_frame_contract_hash
            ):
                raise ValueError(
                    "Pose transform target contract does not match runtime pose contract."
                )
        self._chunk_execution_policy = ActionChunkExecutionPolicy(
            config.action_chunk_execution_steps
        )
        self._next_policy_chunk_target_at: float | None = None
        hand_sides = tuple(self._compound_hand_specs)
        self.catalog = (
            _bimanual_pose_catalog(hand_sides)
            if self._pose_execution_enabled
            else _bimanual_qpos_catalog(hand_sides)
        )
        self.reference_action_schema_hash = (
            _bimanual_qpos_catalog(hand_sides).semantic_content_hash
        )
        self._component_dependencies_by_group = {
            group: tuple(
                item.effector_name
                for item in self.catalog.components_for_group(group)
            )
            for group in self.catalog.selection_group_ids
        }
        timing = rig.config.timing
        self.execution_capability = ExecutionCapabilityProfile(
            profile_id=(
                "mujoco-bimanual-qpos-execution-v1"
                if not self._compound_execution_enabled
                else "mujoco-bimanual-pose-dmtac-compound-execution-v1"
            ),
            validated_max_selected_keyframe_hz=60.0,
            configured_selected_keyframe_hz=config.decision_hz,
            validated_max_coupled_ik_hz=(
                60.0 if self._pose_execution_enabled else None
            ),
            configured_coupled_ik_hz=(
                config.decision_hz if self._pose_execution_enabled else None
            ),
            reference_resample_hz=timing.reference_hz,
            command_hz=timing.command_hz,
            backend_servo_hz=timing.physics_hz,
            validated_action_spaces=(self.catalog.action_space,),
            validated_selection_groups=self.catalog.selection_group_ids,
            supports_local_group_stop=False,
            acceptance_evidence_id=(
                "mujoco-coupled-pose-dmtac-compound-contract-tests-v1"
                if self._compound_execution_enabled
                else "mujoco-coupled-pose-ik-contract-tests-v1"
                if self._pose_execution_enabled
                else "mujoco-deploy-contract-tests-v1"
            ),
            decision_runtime_budget_sec=0.005,
            ik_runtime_budget_sec=1.0 / 60.0,
            end_to_end_control_budget_sec=1.0 / config.decision_hz,
            required_cancellation_ack_levels=("software_queue",),
        )
        capability_hash = self.execution_capability.semantic_content_hash
        self.execution_capability_profile_hash = capability_hash
        self.target_runtime_action_contract: TargetRuntimeActionContract = (
            build_target_runtime_action_contract(
                target_contract_id=(
                    f"mujoco-target-action-contract:{config.control_domain_id}"
                ),
                control_domain_id=config.control_domain_id,
                embodiment_instance_id=config.embodiment_instance_id,
                component_catalog=self.catalog,
                pose_frame_contract_hash=(
                    config.pose_frame_contract_hash
                    if self._pose_execution_enabled
                    else None
                ),
                action_spec_hash_by_component={
                    "left_arm": (
                        "rm65-tool-tcp-pose-v1"
                        if self._pose_execution_enabled
                        else "rm65-arm-qpos6-joint-order-v1"
                    ),
                    "right_arm": (
                        "rm65-tool-tcp-pose-v1"
                        if self._pose_execution_enabled
                        else "rm65-arm-qpos6-joint-order-v1"
                    ),
                    **{
                        f"{side}_hand": (
                            f"dmtac-gripper-q6:{side}:"
                            f"{self._compound_hand_specs[side].command_space}"
                        )
                        for side in self._compound_hand_specs
                    },
                },
                units_convention_id=(
                    "si_m_rad_xyzw_plus_normalized_hand_v1"
                    if self._compound_execution_enabled
                    else "si_m_rad_xyzw_v1"
                    if self._pose_execution_enabled
                    else "si_rad_joint_order_v1"
                ),
                supported_transform_hash_by_source_frame=(
                    {}
                    if not self._pose_execution_enabled
                    or pose_resolver.pose_transformer is None
                    else {
                        pose_resolver.pose_transformer.binding_receipt.source_pose_frame_contract_hash: (
                            pose_resolver.pose_transformer.binding_receipt.semantic_content_hash
                        )
                    }
                ),
                execution_capability_profile_hash=capability_hash,
            )
        )
        target_lead_time_sec = 0.0
        schedule_spec_id = f"mujoco-{config.decision_hz:g}hz-v1"
        if (
            config.reference_activation_profile_id
            == SCHEDULED_REFERENCE_SEGMENT_SIM_V1
        ):
            target_lead_time_sec = (
                1.0 / config.decision_hz + 1.0 / timing.reference_hz
            )
            schedule_spec_id = (
                f"mujoco-{config.decision_hz:g}hz-scheduled-segment-v1"
            )
        decision_schedule = DecisionScheduleSpec(
            spec_id=schedule_spec_id,
            decision_hz=config.decision_hz,
            target_lead_time_sec=target_lead_time_sec,
            max_target_lateness_sec=max(
                0.02,
                1.0 / config.decision_hz,
            ),
        )
        if config.reference_activation_profile_id == IMMEDIATE_QREF_SIM_V1:
            self.reference_activation_spec: ReferenceActivationSpec = (
                build_immediate_qref_activation_spec(
                    decision_schedule,
                    self.execution_capability,
                )
            )
        else:
            self.reference_activation_spec = (
                build_scheduled_reference_segment_activation_spec(
                    decision_schedule,
                    self.execution_capability,
                    minimum_activation_margin_sec=1.0 / timing.reference_hz,
                    activation_margin_evidence_id=(
                        "mujoco-deterministic-one-reference-period-margin-v1"
                    ),
                )
            )
        validate_reference_activation_binding(
            self.reference_activation_spec,
            schedule=decision_schedule,
            capability=self.execution_capability,
        )
        priority: dict[str, int] = {}
        if config.enable_primary_policy_authority:
            priority[config.producer_id] = 10
        if config.enable_replay_authority:
            priority[config.replay_producer_id] = 20
        if config.enable_composer_authority:
            priority[config.composer_producer_id] = 15
        self.core = AuthoritativeControlRuntime(
            AuthoritativeControlConfig(
                runtime_session_id=config.runtime_session_id,
                control_domain_id=config.control_domain_id,
                embodiment_instance_id=config.embodiment_instance_id,
                owner_runtime_id=config.owner_runtime_id,
                clock_domain=ClockDomainDescriptor(
                    clock_domain_id=config.clock_domain_id,
                    runtime_session_id=config.runtime_session_id,
                    boot_id=config.boot_id,
                    reset_epoch=0,
                ),
                component_catalog=self.catalog,
                active_selection_groups=self.catalog.selection_group_ids,
                authority_policy=AuthorityPolicySpec(
                    policy_id=config.authority_policy_id,
                    priority_by_group={
                        group: priority
                        for group in self.catalog.selection_group_ids
                    },
                    fallback_by_group={
                        group: (
                            "hold"
                            if self._compound_composer_authority_profile
                            is not None
                            else "no_action"
                        )
                        for group in self.catalog.selection_group_ids
                    },
                    takeover_allowed_producers=(
                        config.authority_takeover_allowed_producers
                    ),
                ),
                decision_schedule=decision_schedule,
                persistence_policy=ActionPersistencePolicy(
                    policy_id="mujoco-selected-segment-v1",
                    selected_segment_max_lifetime_sec=(
                        config.selected_segment_max_lifetime_sec
                    ),
                ),
                execution_capability=self.execution_capability,
                target_runtime_contract_hash=(
                    self.target_runtime_action_contract.semantic_content_hash
                ),
                initial_control_mode=config.initial_control_mode,
                defer_selected_segment_activation=(
                    self.reference_activation_spec.activation_mode
                    == "scheduled_segment"
                ),
                compound_authority_profiles=(
                    ()
                    if self._compound_composer_authority_profile is None
                    else (self._compound_composer_authority_profile,)
                ),
            ),
            dispatcher=rig.send_commands,
            contextual_dispatcher=self._dispatch_with_context,
            backend_hold=self._backend_hold,
            backend_name=rig.name,
            resolver=pose_resolver,
            shared_safety=shared_safety,
            trace_sink=trace_sink,
            clock=clock,
        )
        self.reference_receiver: MujocoScheduledQposReferenceReceiver | None = None
        if self.reference_activation_spec.activation_mode == "scheduled_segment":
            receiver_args = {
                "activation_spec": self.reference_activation_spec,
                "action_schema_hash": self.reference_action_schema_hash,
                "clock_domain_id": config.clock_domain_id,
                "live_state_provider": self.core.capture_receiver_state,
                "activation_callback": (
                    lambda group, generation: self.core.activate_pending_execution_segment(
                        selection_group_id=group,
                        group_segment_generation=generation,
                    )
                ),
                "clock": clock,
            }
            if self._pose_execution_enabled:
                self.reference_receiver = (
                    MujocoScheduledResolvedQposReferenceReceiver(
                        rig,
                        pose_action_schema_hash=self.catalog.semantic_content_hash,
                        compound_activation_callback=(
                            self.core.activate_pending_execution_segments
                            if self._compound_execution_enabled
                            else None
                        ),
                        **receiver_args,
                    )
                )
            else:
                self.reference_receiver = MujocoScheduledQposReferenceReceiver(
                    rig,
                    **receiver_args,
                )
        self._composition_trace_emitter = CompositionControlTraceEmitter(
            trace_context=self.core.trace_context,
            trace_sink=self.core.record_external_facts,
        )
        self.policy_topology = ControlCompositionRuntime(
            runtime_session_id=config.runtime_session_id,
            control_domain_id=config.control_domain_id,
            embodiment_instance_id=config.embodiment_instance_id,
            coordinator=self.core.coordinator,
            producer_epochs=self.core.producer_epochs,
            component_catalog=self.catalog,
            event_emitter=self._composition_trace_emitter,
            proposal_admitter=self.core.admit_proposal,
            context_retention_spec=TimeWindowRetentionSpec(
                spec_id="mujoco-composition-context-latest-v1",
                history_window_sec=0.0,
                max_future_lead_sec=1.0,
                max_effective_duration_sec=1.0,
                max_late_arrival_sec=0.0,
            ),
            composition_input_retention_spec=TimeWindowRetentionSpec(
                spec_id="mujoco-composition-input-latest-v1",
                history_window_sec=0.0,
                max_future_lead_sec=1.0,
                max_effective_duration_sec=1.0,
                max_late_arrival_sec=0.0,
            ),
            source_eligibility_retention_spec=(
                TimeWindowRetentionSpec(
                    spec_id="mujoco-composition-source-eligibility-latest-v1",
                    history_window_sec=0.0,
                    max_future_lead_sec=1.0,
                    max_effective_duration_sec=1.0,
                    max_late_arrival_sec=0.0,
                )
                if config.enable_composition_source_eligibility
                else None
            ),
            information_plan=config.composition_information_plan,
            composition_strategy=composition_strategy,
            clock=clock,
        )
        self._started = False
        self._last_dispatch_context: AuthoritativeCommandDispatchContext | None = None
        self._last_compound_dispatch_manifest: (
            CompoundDispatchManifest | None
        ) = None
        self._projected_reference_fact_ids: set[str] = set()
        self._projected_policy_emission_ids: set[str] = set()
        self._next_proposal_sequence = 0
        self._next_replay_proposal_sequence = 0
        self._next_composite_proposal_sequence = 0
        self._composer_producer_epoch = config.producer_epoch
        self.observation_adapter = MujocoPolicyObservationAdapter(
            rig,
            policy_node_id=config.policy_node_id,
            policy_node_epoch=config.producer_epoch,
            clock=clock,
        )
        self._observation_adapter_by_node = {
            config.policy_node_id: self.observation_adapter
        }

    def _dispatch_with_context(
        self,
        command,
        context: AuthoritativeCommandDispatchContext,
    ):
        """Forward one already-permitted command without upgrading it to a segment.

        M5.7e2-c uses this seam so the future sim reference receiver can enforce
        epoch/generation and preserve safety lineage at its actual mailbox
        admission point.  Capturing the context alone is not admission evidence.
        """

        self._last_dispatch_context = context
        if self.reference_receiver is not None:
            if self._pose_execution_enabled:
                assert self.pose_resolver is not None
                assert isinstance(
                    self.reference_receiver,
                    MujocoScheduledResolvedQposReferenceReceiver,
                )
                resolution = self.pose_resolver.resolution_for_command(command)
                compound_manifest = None
                if self._compound_execution_enabled:
                    compound_manifest = build_compound_dispatch_manifest(
                        context,
                        component_dependencies_by_group=(
                            self._component_dependencies_by_group
                        ),
                    )
                    self._last_compound_dispatch_manifest = compound_manifest
                target = build_mujoco_pose_qpos_execution_target(
                    resolution,
                    context,
                    resolved_at=float(self.clock()),
                    target_validity_sec=self.config.proposal_horizon_sec,
                    clock_domain_id=self.config.clock_domain_id,
                    input_action_schema_hash=self.catalog.semantic_content_hash,
                    output_target_schema_hash=self.reference_action_schema_hash,
                    compound_manifest=compound_manifest,
                )
                return self.reference_receiver.dispatch_resolved_target(
                    command,
                    context,
                    target=target,
                    resolution=resolution,
                    compound_manifest=compound_manifest,
                )
            return self.reference_receiver.dispatch(command, context)
        return self.core.dispatcher(command)

    @property
    def last_dispatch_context(self) -> AuthoritativeCommandDispatchContext | None:
        return self._last_dispatch_context

    @property
    def last_compound_dispatch_manifest(
        self,
    ) -> CompoundDispatchManifest | None:
        return self._last_compound_dispatch_manifest

    def _backend_hold(self, cancellation) -> dict[str, bool]:
        if self.reference_receiver is not None:
            self.reference_receiver.cancel_all(
                reason=cancellation.cancellation_id,
                effective_at=float(self.clock()),
            )
        self.rig.hold_motion(
            f"{cancellation.reason}:{cancellation.cancellation_id}"
        )
        return {"software_queue": True}

    @property
    def trace_facts(self) -> tuple[ControlTraceFact, ...]:
        return self.core.trace_facts

    @property
    def pose_execution_enabled(self) -> bool:
        return self._pose_execution_enabled

    @property
    def explicit_shared_safety(self) -> bool:
        return self._explicit_shared_safety

    def start(self) -> None:
        if self._started:
            return
        self.rig.start()
        self.core.record_compound_authority_profiles()
        self.core.activate_producer(
            producer_id=self.config.producer_id,
            producer_epoch=self.config.producer_epoch,
            reason="mujoco deploy startup",
        )
        if self.config.enable_replay_authority:
            self.core.activate_producer(
                producer_id=self.config.replay_producer_id,
                producer_epoch=self.config.producer_epoch,
                reason="mujoco replay authority startup",
            )
        if self.config.enable_composer_authority:
            self.core.activate_producer(
                producer_id=self.config.composer_producer_id,
                producer_epoch=self.config.producer_epoch,
                reason="mujoco composer authority startup",
            )
        self._started = True

    def activate_policy_node_producer(
        self,
        producer_id: str,
        *,
        producer_epoch: int | None = None,
        reason: str = "mujoco policy topology startup",
    ):
        if not self._started:
            raise RuntimeError("MujocoDeployRuntime.start() is required.")
        activation = self.core.activate_producer(
            producer_id=producer_id,
            producer_epoch=(
                self.config.producer_epoch
                if producer_epoch is None
                else int(producer_epoch)
            ),
            reason=reason,
        )
        if producer_id == self.config.composer_producer_id:
            self._composer_producer_epoch = activation.producer_epoch
        return activation

    def submit_policy_action(
        self,
        action: RobotAction,
        *,
        policy_output_id: str,
        produced_at: float | None = None,
        target_effective_at: float | None = None,
        observation_cut_id: str | None = None,
        chunk_id: str | None = None,
        chunk_keyframe_index: int | None = None,
        inference_id: str | None = None,
    ):
        if not self._started:
            raise RuntimeError("MujocoDeployRuntime.start() is required.")
        if action.action_space != self.catalog.action_space:
            raise ValueError(
                "Deployable sim action does not match its explicit execution profile."
            )
        expected_effectors = {
            component.effector_name for component in self.catalog.components
        }
        if set(action.active_effectors) != expected_effectors:
            raise ValueError(
                "Policy action must exactly cover its execution profile components."
            )
        produced = float(self.clock() if produced_at is None else produced_at)
        target = float(produced if target_effective_at is None else target_effective_at)
        sequence = self._next_proposal_sequence
        draft = ActionProposalDraft(
            runtime_session_id=self.config.runtime_session_id,
            control_domain_id=self.config.control_domain_id,
            embodiment_instance_id=self.config.embodiment_instance_id,
            producer_kind="policy",
            producer_id=self.config.producer_id,
            producer_epoch=self.config.producer_epoch,
            sequence=sequence,
            action=action,
            produced_at=produced,
            produced_clock_domain_id=self.config.clock_domain_id,
            effective_from=target,
            effective_until=target + self.config.proposal_horizon_sec,
            target_effective_at=target,
            effective_clock_domain_id=self.config.clock_domain_id,
            component_schema_hash=self.catalog.semantic_content_hash,
            component_coverage=tuple(
                component.component_id for component in self.catalog.components
            ),
            selection_groups=self.catalog.selection_group_ids,
            policy_node_id=self.config.policy_node_id,
            observation_cut_id=observation_cut_id,
            inference_id=inference_id,
            policy_output_id=policy_output_id,
            chunk_id=chunk_id,
            chunk_keyframe_index=chunk_keyframe_index,
        )
        receipt = self.core.admit_proposal(
            draft,
            source_event_id=policy_output_id,
        )
        if receipt.accepted:
            self._next_proposal_sequence += 1
        return receipt

    def submit_policy_action_emission(
        self,
        emission: PolicyActionEmission,
    ) -> MujocoPolicyEmissionSubmission:
        """Admit the exact immutable emission produced by the deploy queue owner.

        This method never predicts, merges, peeks, or pops.  It records the
        lower-owned emission once and uses its exact target/effective interval
        and inference/chunk lineage for proposal admission.
        """

        if not self._started:
            raise RuntimeError("MujocoDeployRuntime.start() is required.")
        if not isinstance(emission, PolicyActionEmission):
            raise TypeError("emission must be a PolicyActionEmission.")
        if emission.emission_id in self._projected_policy_emission_ids:
            raise ValueError("PolicyActionEmission may be submitted only once.")
        if emission.policy_node_id != self.config.policy_node_id:
            raise ValueError("Policy emission belongs to another policy node.")
        if (
            emission.produced_clock_domain_id != self.config.clock_domain_id
            or emission.effective_clock_domain_id != self.config.clock_domain_id
        ):
            raise ValueError(
                "Policy emission and MuJoCo runtime require one clock domain."
            )
        if not (
            emission.effective_from
            <= emission.target_effective_at
            < emission.effective_until
        ):
            raise ValueError("Policy emission target is outside its effective interval.")

        action = emission.pre_safety_robot_action
        validation = emission.source_validation
        if action is not None and validation is not None and validation.accepted:
            if action.action_space != self.catalog.action_space:
                raise ValueError(
                    "Policy emission action does not match the MuJoCo execution profile."
                )
            if set(action.active_effectors) != {"left_arm", "right_arm"}:
                raise ValueError(
                    "CS-2c requires one atomic coupled bimanual policy emission."
                )

        self.core.record_external_facts(
            project_policy_action_emission(
                emission,
                runtime_session_id=self.config.runtime_session_id,
                control_domain_id=self.config.control_domain_id,
                embodiment_instance_id=self.config.embodiment_instance_id,
                clock_domain_id=self.config.clock_domain_id,
            )
        )
        self._projected_policy_emission_ids.add(emission.emission_id)

        if (
            action is None
            or validation is None
            or not validation.accepted
            or emission.pre_safety_faults
            or emission.representation_receipt is None
        ):
            return MujocoPolicyEmissionSubmission(emission, None)
        coverage = tuple(
            component.component_id
            for component in self.catalog.components
            if component.effector_name in action.active_effectors
        )
        selection_groups = tuple(
            group
            for group in self.catalog.selection_group_ids
            if any(
                component.component_id in coverage
                for component in self.catalog.components_for_group(group)
            )
        )

        sequence = self._next_proposal_sequence
        draft = ActionProposalDraft(
            runtime_session_id=self.config.runtime_session_id,
            control_domain_id=self.config.control_domain_id,
            embodiment_instance_id=self.config.embodiment_instance_id,
            producer_kind="policy",
            producer_id=self.config.producer_id,
            producer_epoch=self.config.producer_epoch,
            sequence=sequence,
            action=action,
            produced_at=emission.produced_at,
            produced_clock_domain_id=emission.produced_clock_domain_id,
            effective_from=emission.effective_from,
            effective_until=emission.effective_until,
            target_effective_at=emission.target_effective_at,
            effective_clock_domain_id=emission.effective_clock_domain_id,
            component_schema_hash=self.catalog.semantic_content_hash,
            component_coverage=coverage,
            selection_groups=selection_groups,
            policy_node_id=emission.policy_node_id,
            inference_id=emission.inference_id,
            policy_output_id=emission.policy_output_id,
            chunk_id=str(emission.chunk_id),
            chunk_keyframe_index=emission.chunk_keyframe_index,
            representation_trace_id=(
                emission.representation_receipt.representation_trace_id
            ),
            audit_metadata={"policy_action_emission_id": emission.emission_id},
        )
        receipt = self.core.admit_proposal(
            draft,
            source_event_id=emission.emission_id,
        )
        if receipt.accepted:
            self._next_proposal_sequence += 1
        return MujocoPolicyEmissionSubmission(emission, receipt)

    def submit_policy_action_chunk(
        self,
        actions: Sequence[RobotAction],
        *,
        policy_output_id: str,
        chunk_id: str | None = None,
        produced_at: float | None = None,
        target_effective_at: float | None = None,
        observation_cut_id: str | None = None,
        action_chunk_execution_steps: int | None = None,
    ) -> MujocoPolicyActionChunkSubmission:
        """Admit only the configured prefix of a policy action chunk.

        Prefix actions are assigned consecutive decision timestamps.  When no
        explicit start is supplied, a later chunk starts at the end of the
        previously submitted chunk (or immediately if that end is already in
        the past), allowing an asynchronous producer to submit the next chunk
        before the current prefix is exhausted.
        """

        if not self._started:
            raise RuntimeError("MujocoDeployRuntime.start() is required.")
        values = tuple(actions)
        if not values:
            raise ValueError("actions must contain at least one RobotAction.")
        if any(not isinstance(action, RobotAction) for action in values):
            raise TypeError("actions must contain only RobotAction values.")
        for action in values:
            if action.action_space != self.catalog.action_space:
                raise ValueError(
                    "Deployable sim action does not match its explicit execution profile."
                )
            expected_effectors = {
                component.effector_name for component in self.catalog.components
            }
            if set(action.active_effectors) != expected_effectors:
                raise ValueError(
                    "Policy action must exactly cover the explicit execution "
                    f"profile components: expected={sorted(expected_effectors)!r}, "
                    f"got={sorted(action.active_effectors)!r}."
                )
        normalized_policy_output_id = str(policy_output_id).strip()
        if not normalized_policy_output_id:
            raise ValueError("policy_output_id must be non-empty.")
        policy = (
            self._chunk_execution_policy
            if action_chunk_execution_steps is None
            else ActionChunkExecutionPolicy(action_chunk_execution_steps)
        )
        selected = policy.select(values)
        if not selected:
            raise ValueError("action chunk execution prefix is empty.")
        resolved_chunk_id = (
            f"{normalized_policy_output_id}:chunk"
            if chunk_id is None
            else str(chunk_id).strip()
        )
        if not resolved_chunk_id.strip():
            raise ValueError("chunk_id must be non-empty when supplied.")
        clock_now: float | None = None
        if produced_at is None or target_effective_at is None:
            clock_now = float(self.clock())
        produced = float(clock_now if produced_at is None else produced_at)
        if not math.isfinite(produced):
            raise ValueError("produced_at must be finite.")
        if target_effective_at is None:
            assert clock_now is not None
            target = float(clock_now)
            if self._next_policy_chunk_target_at is not None:
                target = max(target, self._next_policy_chunk_target_at)
        else:
            target = float(target_effective_at)
        if not math.isfinite(target):
            raise ValueError("target_effective_at must be finite.")
        period = 1.0 / float(self.config.decision_hz)
        receipts: list[ProposalAdmissionReceipt] = []
        for index, action in enumerate(selected):
            step_target = target + index * period
            receipt = self.submit_policy_action(
                action,
                policy_output_id=f"{normalized_policy_output_id}:step:{index}",
                produced_at=produced,
                target_effective_at=step_target,
                observation_cut_id=observation_cut_id,
                chunk_id=resolved_chunk_id,
                chunk_keyframe_index=index,
                inference_id=resolved_chunk_id,
            )
            receipts.append(receipt)
        self._next_policy_chunk_target_at = target + len(selected) * period
        return MujocoPolicyActionChunkSubmission(
            chunk_id=resolved_chunk_id,
            source_chunk_size=len(values),
            executed_chunk_size=len(selected),
            action_chunk_execution_steps=policy.execution_steps,
            receipts=tuple(receipts),
        )

    def submit_replay_action(
        self,
        source: ReplayProposalSource,
        action: RobotAction,
        *,
        source_event_id: str,
        source_path: str,
        source_episode_index: int,
        group_key: str,
        local_frame_index: int,
        source_action_index: int,
        source_timestamp: float,
        produced_at: float | None = None,
        target_effective_at: float | None = None,
    ) -> MujocoReplaySubmission:
        """Admit one verified replay candidate and its exact target relation."""

        if not self._started:
            raise RuntimeError("MujocoDeployRuntime.start() is required.")
        if not self.config.enable_replay_authority:
            raise RuntimeError("Replay authority is not enabled for this sim runtime.")
        source_config = source.config
        if source_config.producer_id != self.config.replay_producer_id:
            raise ValueError("Replay producer does not match the sim authority profile.")
        if (
            source_config.runtime_session_id != self.config.runtime_session_id
            or source_config.control_domain_id != self.config.control_domain_id
            or source_config.embodiment_instance_id != self.config.embodiment_instance_id
        ):
            raise ValueError("Replay proposal source belongs to another sim target.")
        if source_config.target_runtime_contract.semantic_content_hash != (
            self.target_runtime_action_contract.semantic_content_hash
        ):
            raise ValueError("Replay source is not assessed against this exact sim target.")
        produced = float(self.clock() if produced_at is None else produced_at)
        target = float(produced if target_effective_at is None else target_effective_at)
        source_emission = create_replay_source_frame_emission(
            source_path=source_path,
            source_episode_index=source_episode_index,
            group_key=group_key,
            local_frame_index=local_frame_index,
            source_action_index=source_action_index,
            source_relative_time=source_timestamp,
            action=action,
            observed_at=produced,
            target_effective_at=target,
            effective_until=target + self.config.proposal_horizon_sec,
            clock_domain_id=self.config.clock_domain_id,
        )
        adapter_result = source.adapt_action(
            source_emission.action,
            source_event_id=source_event_id,
            source_emission_id=source_emission.emission_id,
            source_action_index=source_action_index,
            source_timestamp=source_timestamp,
            source_timestamp_clock_domain_id="capture_relative",
            sequence=self._next_replay_proposal_sequence,
            produced_at=produced,
            produced_clock_domain_id=self.config.clock_domain_id,
            effective_from=target,
            effective_until=target + self.config.proposal_horizon_sec,
            target_effective_at=target,
            effective_clock_domain_id=self.config.clock_domain_id,
            same_legacy_emission=False,
        )
        return self._admit_replay_adapter_result(
            source_emission=source_emission,
            adapter_result=adapter_result,
            occurred_at=produced,
        )

    def submit_replay_source_frame_emission(
        self,
        source: ReplayProposalSource,
        emission: ReplaySourceFrameEmission,
    ) -> MujocoReplaySubmission:
        """Admit the existing immutable replay emission without rebuilding it.

        This is the application-composition seam for a passive observer of the
        canonical replay cursor/scheduler.  It never reads a dataset, advances
        a cursor, resamples time, or creates a second source-frame identity.
        """

        if not self._started:
            raise RuntimeError("MujocoDeployRuntime.start() is required.")
        if not self.config.enable_replay_authority:
            raise RuntimeError("Replay authority is not enabled for this sim runtime.")
        if not isinstance(emission, ReplaySourceFrameEmission):
            raise TypeError("emission must be a ReplaySourceFrameEmission.")
        source_config = source.config
        if source_config.producer_id != self.config.replay_producer_id:
            raise ValueError("Replay producer does not match the sim authority profile.")
        if (
            source_config.runtime_session_id != self.config.runtime_session_id
            or source_config.control_domain_id != self.config.control_domain_id
            or source_config.embodiment_instance_id != self.config.embodiment_instance_id
        ):
            raise ValueError("Replay proposal source belongs to another sim target.")
        if source_config.target_runtime_contract.semantic_content_hash != (
            self.target_runtime_action_contract.semantic_content_hash
        ):
            raise ValueError("Replay source is not assessed against this exact sim target.")
        if emission.clock_domain_id != self.config.clock_domain_id:
            raise ValueError(
                "Replay source-frame and sim runtime require one explicit clock domain."
            )
        adapter_result = source.adapt_source_frame_emission(
            emission,
            sequence=self._next_replay_proposal_sequence,
        )
        return self._admit_replay_adapter_result(
            source_emission=emission,
            adapter_result=adapter_result,
            occurred_at=emission.observed_at,
        )

    def _admit_replay_adapter_result(
        self,
        *,
        source_emission: ReplaySourceFrameEmission,
        adapter_result: ReplayProposalAdapterResult,
        occurred_at: float,
    ) -> MujocoReplaySubmission:
        """Record and admit one already-adapted replay source-frame fact."""

        produced = float(occurred_at)
        evidence_facts = project_replay_adapter_result(
            adapter_result,
            runtime_session_id=self.config.runtime_session_id,
            control_domain_id=self.config.control_domain_id,
            embodiment_instance_id=self.config.embodiment_instance_id,
            occurred_at=produced,
            clock_domain_id=self.config.clock_domain_id,
        )
        self.core.record_external_facts(
            (
                project_replay_source_frame_emission(
                    source_emission,
                    runtime_session_id=self.config.runtime_session_id,
                    control_domain_id=self.config.control_domain_id,
                    embodiment_instance_id=self.config.embodiment_instance_id,
                ),
                *evidence_facts,
            )
        )
        if adapter_result.proposal_draft is None:
            return MujocoReplaySubmission(source_emission, adapter_result, None, None)
        proposal_receipt = self.core.admit_proposal(
            adapter_result.proposal_draft,
            source_event_id=source_emission.emission_id,
        )
        authorization_receipt: CandidateAuthorizationAdmissionReceipt | None = None
        if proposal_receipt.accepted and proposal_receipt.proposal is not None:
            eligibility = adapter_result.binding_eligibility
            execution_authorization = build_replay_execution_authorization(
                adapter_result.binding_assessment,
                requested_mode=REPLAY_EXECUTION_MODE_SIM_AUTHORITY,
                safety_preflight_id="mujoco-shared-canonical-safety-v1",
                execution_capability_evidence_id=(
                    self.execution_capability_profile_hash
                ),
                dispatch_owner_state_id=self.core.dispatch_owner_token.token_id,
                operator_authorization_id="mujoco-config:enable_replay_authority",
                authorized=self.config.enable_replay_authority,
            )
            authorization_receipt = self.core.admit_replay_candidate_authorization(
                proposal_id=proposal_receipt.proposal.proposal_id,
                assessment=adapter_result.binding_assessment,
                eligibility=eligibility,
                execution_authorization=execution_authorization,
            )
            if authorization_receipt.accepted:
                self._next_replay_proposal_sequence += 1
        return MujocoReplaySubmission(
            source_emission,
            adapter_result,
            proposal_receipt,
            authorization_receipt,
        )

    def capture_policy_observation(
        self,
        *,
        policy_node_id: str | None = None,
        trigger_reason: str = "policy_schedule",
    ) -> ObservationBundle:
        if not self._started:
            raise RuntimeError("MujocoDeployRuntime.start() is required.")
        node_id = self.config.policy_node_id if policy_node_id is None else str(
            policy_node_id
        ).strip()
        if not node_id:
            raise ValueError("policy_node_id must be non-empty.")
        adapter = self._observation_adapter_by_node.get(node_id)
        if adapter is None:
            adapter = MujocoPolicyObservationAdapter(
                self.rig,
                policy_node_id=node_id,
                policy_node_epoch=self.config.producer_epoch,
                clock=self.clock,
            )
            self._observation_adapter_by_node[node_id] = adapter
        bundle = adapter.capture(
            self.core.coordinator,
            trigger_reason=trigger_reason,
        )
        self.core.record_observation_cut(bundle.record)
        return bundle

    def submit_composition_context(self, draft: CompositionContextDraft):
        if not self._started:
            raise RuntimeError("MujocoDeployRuntime.start() is required.")
        return self.policy_topology.admit_composition_context(draft)

    def capture_composition_context_snapshot(
        self,
        *,
        target_effective_at: float,
        required_channels: tuple[str, ...],
    ):
        if not self._started:
            raise RuntimeError("MujocoDeployRuntime.start() is required.")
        return self.policy_topology.capture_composition_context_snapshot(
            target_effective_at=target_effective_at,
            required_channels=required_channels,
        )

    def submit_composition_input(
        self,
        draft: CompositionInputDraft,
        *,
        activation_scope: CompositionActivationScope | None = None,
    ):
        if not self._started:
            raise RuntimeError("MujocoDeployRuntime.start() is required.")
        return self.policy_topology.admit_composition_input(
            draft,
            activation_scope=activation_scope,
        )

    def submit_composition_source_eligibility(
        self,
        draft: CompositionSourceEligibilityDraft,
    ):
        if not self._started:
            raise RuntimeError("MujocoDeployRuntime.start() is required.")
        return self.policy_topology.admit_source_eligibility(draft)

    def evaluate_composition_shadow(
        self,
        *,
        spec: CompositionSpec,
        target_effective_at: float,
        required_context_channels: tuple[str, ...] = (),
        cutoff_at: float | None = None,
        control_cut: ControlCut | None = None,
        source_difference_spec: SourceDifferenceSpec | None = None,
        activation_scope: CompositionActivationScope | None = None,
        source_eligibility_spec: CompositionSourceEligibilitySpec | None = None,
    ):
        """Evaluate a composition cut without proposal, authority, or dispatch."""

        if not self._started:
            raise RuntimeError("MujocoDeployRuntime.start() is required.")
        submission = self.policy_topology.evaluate_composition(
            spec=spec,
            target_effective_at=target_effective_at,
            required_context_channels=required_context_channels,
            cutoff_at=cutoff_at,
            control_cut=control_cut,
            source_difference_spec=source_difference_spec,
            activation_scope=activation_scope,
            source_eligibility_spec=source_eligibility_spec,
        )
        if submission.proposal_admission is not None:
            raise RuntimeError("Shadow composition unexpectedly admitted a proposal.")
        return submission

    def compose_and_submit(
        self,
        *,
        spec: CompositionSpec,
        target_effective_at: float,
        policy_node_id: str = "fast-composer",
        required_context_channels: tuple[str, ...] = (),
        source_difference_spec: SourceDifferenceSpec | None = None,
        component_coverage: tuple[str, ...] | None = None,
        selection_groups: tuple[str, ...] | None = None,
    ):
        if not self._started:
            raise RuntimeError("MujocoDeployRuntime.start() is required.")
        if not self.config.enable_composer_authority:
            raise RuntimeError("Composer authority is not enabled for this sim runtime.")
        evaluation = self.policy_topology.evaluate_composition(
            spec=spec,
            target_effective_at=target_effective_at,
            required_context_channels=required_context_channels,
            source_difference_spec=source_difference_spec,
        )
        if (
            evaluation.composition_result.output_status != "ready"
            or evaluation.composition_result.composed_action is None
        ):
            return evaluation
        return self.submit_evaluated_composition(
            evaluation,
            policy_node_id=policy_node_id,
            component_coverage=component_coverage,
            selection_groups=selection_groups,
        )

    def _validate_composition_promotion_scope(
        self,
        *,
        component_coverage: tuple[str, ...] | None,
        selection_groups: tuple[str, ...] | None,
        action: RobotAction,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if component_coverage is None and selection_groups is None:
            if self._compound_composer_authority_profile is None:
                coverage = ("left_arm", "right_arm")
                groups = ("arms",)
            else:
                coverage = (
                    self._compound_composer_authority_profile.component_coverage
                )
                groups = (
                    self._compound_composer_authority_profile.selection_groups
                )
        elif component_coverage is None or selection_groups is None:
            raise ValueError(
                "component_coverage and selection_groups must be provided "
                "together."
            )
        else:
            coverage = tuple(str(value).strip() for value in component_coverage)
            groups = tuple(str(value).strip() for value in selection_groups)
        if (
            not coverage
            or not groups
            or any(not value for value in (*coverage, *groups))
            or len(coverage) != len(set(coverage))
            or len(groups) != len(set(groups))
        ):
            raise ValueError(
                "Composition promotion scope must be non-empty and unique."
            )
        catalog_by_component = self.catalog.component_by_id
        unknown_components = tuple(
            component
            for component in coverage
            if component not in catalog_by_component
        )
        if unknown_components:
            raise ValueError(
                "Composition promotion scope contains components outside the "
                f"active MuJoCo catalog: {unknown_components!r}."
            )
        catalog_order = tuple(
            component.component_id
            for component in self.catalog.components
            if component.component_id in coverage
        )
        if coverage != catalog_order:
            raise ValueError(
                "Composition component_coverage must follow catalog order."
            )
        expected_groups = tuple(
            dict.fromkeys(
                catalog_by_component[component].selection_group_id
                for component in coverage
            )
        )
        if groups != expected_groups:
            raise ValueError(
                "Composition selection_groups must exactly match component "
                "catalog ownership."
            )
        for group in groups:
            members = self.catalog.components_for_group(group)
            if members and members[0].coupled:
                covered_members = {
                    component for component in coverage
                    if catalog_by_component[component].selection_group_id
                    == group
                }
                required_members = {
                    component.component_id for component in members
                }
                if covered_members != required_members:
                    raise ValueError(
                        "Composition promotion cannot partially cover coupled "
                        f"selection group {group!r}."
                    )
        active_effectors = set(action.active_effectors)
        unmapped_effectors = tuple(
            effector
            for effector in action.active_effectors
            if effector
            not in {
                component.effector_name
                for component in self.catalog.components
            }
        )
        if unmapped_effectors:
            raise ValueError(
                "Composed action contains effectors outside the active MuJoCo "
                f"catalog: {unmapped_effectors!r}."
            )
        action_coverage = tuple(
            component.component_id
            for component in self.catalog.components
            if component.effector_name in active_effectors
        )
        if coverage != action_coverage:
            raise ValueError(
                "Composition promotion scope must exactly match the composed "
                "action components."
            )
        if self._compound_composer_authority_profile is not None and (
            coverage
            != self._compound_composer_authority_profile.component_coverage
            or groups
            != self._compound_composer_authority_profile.selection_groups
        ):
            raise ValueError(
                "Conservative compound composer authority requires the exact "
                "all-required component/group scope."
            )
        return coverage, groups

    def submit_evaluated_composition(
        self,
        evaluation: CompositeProposalSubmission,
        *,
        policy_node_id: str = "formal-composer",
        component_coverage: tuple[str, ...] | None = None,
        selection_groups: tuple[str, ...] | None = None,
    ) -> CompositeProposalSubmission:
        """Promote the exact gated result without capturing a later cut."""

        if not self._started:
            raise RuntimeError("MujocoDeployRuntime.start() is required.")
        if not self.config.enable_composer_authority:
            raise RuntimeError("Composer authority is not enabled for this sim runtime.")
        action = evaluation.composition_result.composed_action
        if (
            evaluation.composition_result.output_status != "ready"
            or action is None
        ):
            raise ValueError(
                "Only a ready composition evaluation may be promoted."
            )
        coverage, groups = self._validate_composition_promotion_scope(
            component_coverage=component_coverage,
            selection_groups=selection_groups,
            action=action,
        )
        submission = self.policy_topology.admit_evaluated_composition(
            evaluation,
            runtime_session_id=self.config.runtime_session_id,
            producer_id=self.config.composer_producer_id,
            producer_epoch=self._composer_producer_epoch,
            sequence=self._next_composite_proposal_sequence,
            effective_until=(
                evaluation.input_snapshot.target_effective_at
                + (
                    1.0 / self.config.decision_hz
                    if self._compound_composer_authority_profile is not None
                    else self.config.proposal_horizon_sec
                )
            ),
            component_coverage=coverage,
            selection_groups=groups,
            policy_node_id=policy_node_id,
        )
        if (
            submission.proposal_admission is not None
            and submission.proposal_admission.accepted
            and submission.proposal_admission
            is not evaluation.proposal_admission
        ):
            proposal = submission.proposal_admission.proposal
            if (
                proposal is not None
                and proposal.sequence == self._next_composite_proposal_sequence
            ):
                self._next_composite_proposal_sequence += 1
        return submission

    def step(
        self,
        *,
        now: float | None = None,
        physics_ticks_after_dispatch: int = 0,
    ) -> AuthoritativeControlStep:
        if not self._started:
            raise RuntimeError("MujocoDeployRuntime.start() is required.")
        if int(physics_ticks_after_dispatch) < 0:
            raise ValueError("physics_ticks_after_dispatch must be non-negative.")
        result = self.core.run_decision(now=now)
        self._sync_reference_trace()
        if result.status == "dispatched":
            for _ in range(int(physics_ticks_after_dispatch)):
                self.rig.step_physics()
            self._sync_reference_trace()
            self.core.record_measurement(
                result,
                measurement=build_mujoco_backend_measurement(
                    self.rig,
                    physics_ticks_after_dispatch=int(physics_ticks_after_dispatch),
                ),
            )
        return result

    def _sync_reference_trace(self) -> None:
        receiver = self.reference_receiver
        if receiver is None:
            return
        context = self.core.trace_context
        facts = []
        facts.extend(
            project_reference_receiver_validation(context, item)
            for item in receiver.validations
        )
        facts.extend(
            project_reference_segment(context, item)
            for item in receiver.segment_attempts
        )
        facts.extend(
            project_reference_segment_admission(
                context,
                item,
                receiver_validation=receiver.validation_for_admission(
                    item.admission_id
                ),
            )
            for item in receiver.admissions
        )
        facts.extend(
            project_safety_anchor_commit_candidate(context, item)
            for item in receiver.anchor_candidates
        )
        if isinstance(receiver, MujocoScheduledResolvedQposReferenceReceiver):
            facts.extend(
                project_resolved_execution_target(context, item)
                for item in receiver.resolved_targets
            )
            facts.extend(
                project_execution_target_admission(context, item)
                for item in receiver.target_admissions
            )
            facts.extend(
                project_safety_anchor_commit_evidence(context, item)
                for item in receiver.anchor_commit_evidence
            )
        facts.extend(
            project_group_generation_reservation(context, item)
            for item in receiver.generation_reservations
        )
        facts.extend(
            project_group_generation_activation(context, item)
            for item in receiver.generation_activations
        )
        facts.extend(
            project_reference_target_activation(context, item)
            for item in self.rig.reference_activation_observations
        )
        facts.extend(
            project_planner_command_frame(context, item)
            for item in self.rig.planner_command_frame_observations
        )
        facts.extend(
            project_reference_segment_terminal(context, item)
            for item in receiver.terminals
        )
        pending = tuple(
            fact
            for fact in facts
            if fact.fact_id not in self._projected_reference_fact_ids
        )
        if not pending:
            return
        self.core.record_external_facts(pending)
        self._projected_reference_fact_ids.update(
            fact.fact_id for fact in pending
        )

    def set_execution_gate(
        self,
        status: str,
        *,
        reason: str,
        source_event_ids: tuple[str, ...] = (),
    ):
        if not self._started:
            raise RuntimeError("MujocoDeployRuntime.start() is required.")
        return self.core.admit_execution_gate(
            status=status,
            reason=reason,
            source_event_ids=source_event_ids,
        )

    def transition_execution_gate(
        self,
        status: str,
        *,
        reason: str,
        source_event_ids: tuple[str, ...] = (),
        force_cancellation: bool = False,
    ) -> ExecutionGateTransitionOutcome:
        """Return typed gate/cancellation evidence without changing legacy callers."""

        if not self._started:
            raise RuntimeError("MujocoDeployRuntime.start() is required.")
        return self.core.transition_execution_gate(
            status=status,
            reason=reason,
            source_event_ids=source_event_ids,
            force_cancellation=force_cancellation,
        )

    def capture_qpos_measurement(self) -> MujocoQposMeasurement:
        """Capture raw sim state without assigning policy or source identity."""

        return capture_mujoco_qpos_measurement(
            self.rig,
            sides=self.rig.config.active_sides,
            clock_domain_id=self.config.clock_domain_id,
            clock=self.clock,
        )

    def close(self) -> None:
        if not self._started:
            return
        self._sync_reference_trace()
        self.core.close_trace()
        self.rig.close()
        self._started = False


__all__ = [
    "MujocoDeployConfig",
    "MujocoDeployRuntime",
    "MujocoPolicyActionChunkSubmission",
    "MujocoReplaySubmission",
]
