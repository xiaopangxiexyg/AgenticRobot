"""Scoped human/replay discrete-authority handoff over one MuJoCo core.

The historical CS-6a path remains a direct-qpos debug/contract fixture, not
the mainline mixed-action representation.  The same module also contains the
pose-native CS-6b1d/CS-6b1f path: human/replay mixing stays in bilateral pose
space and derives qpos only inside the target's verified transform/IK
pipeline.  Direct-qpos mixing remains deferred to the future homogeneous-arm
profile.

Handoff is discrete: publication is closed,
the shared runtime enters HOLD and cancels the prior execution epoch, then a
typed authority lease is transferred before the new source may publish.

The legacy CS-6a path still transfers authority and reopens READY immediately.
The incremental CS-6b0 replay-to-human path instead remains in HOLD until a
fresh synthetic-human emission, measured MuJoCo state, deadman re-arm,
retargeter rebase, and canonical-safety preview have all been bound.  Neither
path owns a tracker, replay cursor, dataset reader, resampler, episode
lifecycle, policy worker, RTC producer, or real backend.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, replace
from typing import Callable

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.replay import (
    REPLAY_BINDING_USE_EXECUTION_CANDIDATE,
    REPLAY_BINDING_VERIFIED_COMPATIBLE,
)
from control_composition.composition import (
    CompositionInputAdmissionReceipt,
    CompositionInputDraft,
    CompositionSpec,
    DiscreteOwnerPoseMixingStrategy,
    PoseMixingContext,
    PoseMixingDecision,
    PoseMixingStrategy,
    build_pose_mixing_input_snapshot,
)
from control_composition.runtime import CompositeProposalSubmission
from control_composition.lifecycle import (
    AuthorityReleaseReceipt,
    ExecutionHoldReceipt,
    ExecutionReadyReceipt,
    HumanAuthorityTakeoverReceipt,
    HumanReadinessReceipt,
    HumanTakeoverOperationEvidence,
    HumanTakeoverOperationSpec,
    HumanTakeoverRequest,
    HumanTakeoverState,
    MixedControlBoundaryPolicy,
    SourceInvalidationReceipt,
    apply_authority_release,
    apply_execution_hold,
    apply_execution_ready,
    apply_human_authority_takeover,
    apply_human_readiness,
    apply_source_invalidation,
    begin_human_takeover,
    fail_human_takeover,
)
from robot_motion.action import (
    BIMANUAL_EE_POSE_V1,
    BIMANUAL_QPOS_V1,
    ComponentAction,
    ControlExecutionManifest,
    ExecutionProfileExclusion,
    build_control_execution_manifest,
    materialize_robot_action_v1,
    split_effector_name,
)
from robot_motion.action.component_action_v1 import (
    project_robot_action_v1_to_component_action,
)
from robot_motion.action.proposal import (
    ProposalAdmissionReceipt,
    robot_action_semantic_payload,
)
from robot_motion.runtime.control_admission import ControlCut
from robot_motion.diagnostics import TeleopRunState
from robot_motion.replay import ReplayProposalSource, ReplaySourceFrameEmission
from sim.mujoco.deploy import (
    MujocoDeployRuntime,
    MujocoReplaySubmission,
)
from teleop_runtime.action.human_proposal import (
    HumanCommandEmission,
    HumanProposalAdapter,
    HumanProposalAdapterResult,
)
from teleop_runtime.control_trace.projection import (
    project_human_proposal_adapter_result,
)
from ..contracts import (
    ControlStationRunManifest,
    ExecutionDomainBinding,
    SourceSlotSpec,
    StationIntent,
    StationOperationResultDraft,
)
from ..adapters import StationOperationAdapter
from ..publication_barrier import (
    StationPublicationBarrier,
    StationPublicationBarrierObserverPort,
    StationPublicationBarrierSnapshot,
)
from ..runtime import (
    ControlStationRuntime,
    StationOperationExecution,
    StationPendingOperationSuperseded,
)
from ..state_machine import (
    HUMAN_TAKEOVER_OPERATION_FAULTED_STATUS_V1,
    HUMAN_TAKEOVER_OPERATION_HELD_STATUS_V1,
    HUMAN_TAKEOVER_OPERATION_READY_STATUS_V1,
    ControlStationOperationTransition,
    ControlStationTransition,
)
from ..sources.human import (
    HumanPoseSourceContract,
    HumanTakeoverReadinessCandidate,
    HumanTakeoverReadinessProvider,
    admit_human_command_emission,
)
from ..sources.synthetic_human import (
    SyntheticHumanQposSourceSpec,
)
from ..sources.teleop_pose import (
    PassiveTeleopPoseSessionSource,
    PassiveTeleopPoseStep,
)
from .mujoco_human import (
    build_mujoco_human_proposal_adapter,
    build_mujoco_pose_human_proposal_adapter,
)
from ..sources.replay import ReplaySourceObserverBinding
from ..targets.authoritative import (
    StationAuthoritativeTargetAdapterBase,
    authoritative_authority_epoch_source_id,
    authoritative_execution_epoch_source_id,
    authoritative_owner_registry_id,
    capture_authoritative_execution_gate_snapshot,
)


CS6A_HUMAN_REPLAY_HANDOFF_MANIFEST_ID_V1 = (
    "control_station_cs6a_human_replay_qpos_handoff_mujoco_v1"
)
CS6A_HUMAN_REPLAY_EXECUTION_MANIFEST_ID_V1 = (
    "cs6a_human_replay_qpos_handoff_sim_v1"
)
CS6A_HUMAN_REPLAY_TRACE_PROFILE_ID_V1 = (
    "deployable_sim_scheduled_qpos_human_replay_handoff_v1"
)
CS6B1D_POSE_HUMAN_REPLAY_HANDOFF_MANIFEST_ID_V1 = (
    "control_station_cs6b1d_human_replay_pose_handoff_mujoco_v1"
)
CS6B1D_POSE_HUMAN_REPLAY_TRACE_PROFILE_ID_V1 = (
    "deployable_sim_scheduled_pose_to_qpos_station_v1"
)
CS6B1F_POSE_HUMAN_REPLAY_STATION_MANIFEST_ID_V1 = (
    "control_station_cs6b1f_pose_human_replay_station_mujoco_v1"
)
CS6B1F_POSE_HUMAN_REPLAY_EXECUTION_MANIFEST_ID_V1 = (
    "cs6b1f_pose_human_replay_station_sim_v1"
)
CS6B1F_POSE_HUMAN_REPLAY_TRACE_PROFILE_ID_V1 = (
    "deployable_sim_scheduled_pose_human_replay_station_v1"
)
CS6A_BINDING_RECEIPT_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.cs6a_human_replay_binding_receipt.v1"
)
CS6A_HANDOFF_RECEIPT_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.cs6a_human_replay_handoff_receipt.v1"
)
CS6A_PRODUCER_EPOCH_POLICY_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.cs6a_fixed_producer_epoch.v1"
)
CS6B0_HUMAN_READINESS_MEASUREMENT_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.cs6b0_human_readiness_measurement.v1"
)
CS6B0_CANONICAL_SAFETY_PREVIEW_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.cs6b0_canonical_safety_preview.v1"
)
CS6B1D_POSE_MIXING_SHADOW_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.cs6b1d_pose_mixing_shadow.v1"
)
CS6B1G_HUMAN_POSE_INPUT_ROLE_V1 = "human_pose"
CS6B1G_REPLAY_POSE_INPUT_ROLE_V1 = "replay_pose"


def _producer_epoch_policy_hash(runtime: MujocoDeployRuntime, producer_id: str) -> str:
    return canonical_content_hash(
        {
            "producer_id": producer_id,
            "producer_epoch": runtime.config.producer_epoch,
            "activation": "fixed_on_mujoco_runtime_start",
        },
        domain=CS6A_PRODUCER_EPOCH_POLICY_HASH_DOMAIN_V1,
    )


def build_cs6a_human_replay_handoff_manifest(
    runtime: MujocoDeployRuntime,
    human_adapter: HumanProposalAdapter,
    human_source_spec: SyntheticHumanQposSourceSpec | HumanPoseSourceContract,
    replay_source: ReplayProposalSource,
    replay_observer_binding: ReplaySourceObserverBinding,
) -> ControlStationRunManifest:
    """Freeze two exact source slots and one shared sim execution domain.

    The qpos variant is the historical CS-6a debug fixture.  Passing a pose
    source produces the distinct CS-6b1d mainline pose manifest.
    """

    if not isinstance(runtime, MujocoDeployRuntime):
        raise TypeError("runtime must be a MujocoDeployRuntime.")
    if not isinstance(human_adapter, HumanProposalAdapter):
        raise TypeError("human_adapter must be a HumanProposalAdapter.")
    if not (
        isinstance(human_source_spec, SyntheticHumanQposSourceSpec)
        or isinstance(human_source_spec, HumanPoseSourceContract)
    ):
        raise TypeError("human_source_spec must satisfy a supported source contract.")
    if not isinstance(replay_source, ReplayProposalSource):
        raise TypeError("replay_source must be a ReplayProposalSource.")
    if not isinstance(replay_observer_binding, ReplaySourceObserverBinding):
        raise TypeError(
            "replay_observer_binding must be a ReplaySourceObserverBinding."
        )
    pose_native = isinstance(human_source_spec, HumanPoseSourceContract)
    manifest_id = (
        CS6B1D_POSE_HUMAN_REPLAY_HANDOFF_MANIFEST_ID_V1
        if pose_native
        else CS6A_HUMAN_REPLAY_HANDOFF_MANIFEST_ID_V1
    )
    trace_profile_id = (
        CS6B1D_POSE_HUMAN_REPLAY_TRACE_PROFILE_ID_V1
        if pose_native
        else CS6A_HUMAN_REPLAY_TRACE_PROFILE_ID_V1
    )
    binding = ExecutionDomainBinding(
        binding_id=f"mujoco-target:{runtime.config.control_domain_id}",
        target_kind="sim",
        control_domain_id=runtime.config.control_domain_id,
        embodiment_instance_id=runtime.config.embodiment_instance_id,
        target_runtime_action_contract_hash=(
            runtime.target_runtime_action_contract.semantic_content_hash
        ),
        execution_capability_profile_hash=(
            runtime.execution_capability.semantic_content_hash
        ),
        component_catalog_hash=runtime.catalog.semantic_content_hash,
        selection_group_ids=runtime.authoritative_port.profile.selection_group_ids,
        owner_registry_id=authoritative_owner_registry_id(runtime),
        authority_epoch_source_id=authoritative_authority_epoch_source_id(runtime),
        execution_epoch_source_id=authoritative_execution_epoch_source_id(runtime),
        trace_requirement_profile_id=trace_profile_id,
        explicit_exclusions=(
            "hardware_device_initialization",
            "real_backend",
        ),
    )
    replay_config = replay_source.config
    return ControlStationRunManifest(
        manifest_id=manifest_id,
        manifest_version="1",
        execution_binding=binding,
        source_slots=(
            SourceSlotSpec(
                source_slot_id=human_source_spec.source_id,
                source_kind="human",
                producer_id=runtime.config.producer_id,
                source_action_contract_hash=human_adapter.contract_hash,
                producer_epoch_policy_hash=_producer_epoch_policy_hash(
                    runtime,
                    runtime.config.producer_id,
                ),
                supported_uses=("candidate", "authority"),
                expected_emission_seam=(
                    "teleop_runtime.action.human_proposal.HumanCommandEmission"
                ),
                required_binding_refs=(
                    human_source_spec.semantic_content_hash,
                    runtime.catalog.semantic_content_hash,
                ),
                trace_requirement_refs=(
                    "human_command_emission",
                    "source_adapter_receipt",
                    "source_validation",
                    "action_proposal",
                    "authority_event",
                ),
                explicit_exclusions=(
                    "physical_teleoperation_device",
                    "source_owned_action_queue",
                ),
            ),
            SourceSlotSpec(
                source_slot_id=replay_observer_binding.source_slot_id,
                source_kind="replay",
                producer_id=runtime.config.replay_producer_id,
                source_action_contract_hash=(
                    replay_config.source_action_contract.semantic_content_hash
                ),
                producer_epoch_policy_hash=_producer_epoch_policy_hash(
                    runtime,
                    runtime.config.replay_producer_id,
                ),
                supported_uses=("candidate", "authority"),
                expected_emission_seam=(
                    "robot_motion.replay.proposal_source.ReplaySourceFrameEmission"
                ),
                required_binding_refs=(
                    replay_config.source_contract.semantic_content_hash,
                    replay_config.binding_assessment.semantic_content_hash,
                    replay_observer_binding.semantic_content_hash,
                ),
                trace_requirement_refs=(
                    "replay_source_frame_emission",
                    "replay_source_adapter_receipt",
                    "replay_execution_authorization",
                    "action_proposal",
                    "authority_event",
                ),
                explicit_exclusions=(
                    "control_station_dataset_reader",
                    "control_station_replay_cursor",
                    "control_station_replay_resampler",
                ),
            ),
        ),
        authority_policy_hash=(
            runtime.authoritative_port.profile.authority_policy_hash
        ),
        decision_schedule_hash=(
            runtime.authoritative_port.profile.decision_schedule_hash
        ),
        trace_requirement_profile_id=trace_profile_id,
        capture_policy_id=(
            "capture_not_owned_by_cs6b1d_v1"
            if pose_native
            else "capture_not_owned_by_cs6a_v1"
        ),
        allowed_intent_kinds=(
            "acquire_control_lease",
            "connect_source",
            "connect_target",
            "disconnect_source",
            "disconnect_target",
            "estop",
            "hold",
            "release_candidate",
            "release_control_lease",
            "release_motion_authority",
            "request_candidate",
            "request_human_takeover",
            "request_motion_authority",
            "run_preflight",
            "stop_runtime",
        ),
        explicit_exclusions=(
            "browser_raw_action_submission",
            "continuous_source_blend",
            "formal_episode_lifecycle_owner",
            "human_replay_simultaneous_component_ownership",
            "physical_human_device",
            "policy_model",
            "production_entrypoint",
            "real_motion",
            "replay_cursor_ownership",
            "rtc_policy_source",
            "source_reactivation_within_session",
        ),
    )


def build_cs6b1d_pose_human_replay_handoff_manifest(
    runtime: MujocoDeployRuntime,
    human_adapter: HumanProposalAdapter,
    human_source_spec: HumanPoseSourceContract,
    replay_source: ReplayProposalSource,
    replay_observer_binding: ReplaySourceObserverBinding,
) -> ControlStationRunManifest:
    """Build the pose-native CS-6 manifest without adding another reducer."""

    if not isinstance(human_source_spec, HumanPoseSourceContract):
        raise TypeError("CS6b1d requires a HumanPoseSourceContract.")
    manifest = build_cs6a_human_replay_handoff_manifest(
        runtime,
        human_adapter,
        human_source_spec,
        replay_source,
        replay_observer_binding,
    )
    if manifest.manifest_id != CS6B1D_POSE_HUMAN_REPLAY_HANDOFF_MANIFEST_ID_V1:
        raise RuntimeError("CS6b1d pose manifest resolved to the wrong profile.")
    return manifest


def build_cs6b1f_pose_human_replay_station_manifest(
    runtime: MujocoDeployRuntime,
    human_adapter: HumanProposalAdapter,
    human_source_spec: HumanPoseSourceContract,
    replay_source: ReplayProposalSource,
    replay_observer_binding: ReplaySourceObserverBinding,
) -> ControlStationRunManifest:
    """Build the opt-in formal sim HIL manifest.

    This is the same pose source/target and discrete authority transaction as
    CS-6b1d.  Only the trace profile and capture policy are promoted so an
    external :class:`EpisodeTraceRecorderSink` may bind a real episode; the
    Station composition still owns no recorder, cursor, or episode UUID.
    """

    base = build_cs6b1d_pose_human_replay_handoff_manifest(
        runtime,
        human_adapter,
        human_source_spec,
        replay_source,
        replay_observer_binding,
    )
    source_slots = tuple(
        replace(
            slot,
            trace_requirement_refs=tuple(
                sorted(
                    {
                        *slot.trace_requirement_refs,
                        "station_run_manifest",
                        "station_intent_transition",
                        "station_operation_transition",
                        "station_publication_barrier_transition",
                    }
                )
            ),
        )
        for slot in base.source_slots
    )
    explicit_exclusions = tuple(
        value
        for value in base.explicit_exclusions
        if value != "formal_episode_lifecycle_owner"
    )
    return replace(
        base,
        manifest_id=CS6B1F_POSE_HUMAN_REPLAY_STATION_MANIFEST_ID_V1,
        execution_binding=replace(
            base.execution_binding,
            trace_requirement_profile_id=(
                CS6B1F_POSE_HUMAN_REPLAY_TRACE_PROFILE_ID_V1
            ),
        ),
        source_slots=source_slots,
        trace_requirement_profile_id=CS6B1F_POSE_HUMAN_REPLAY_TRACE_PROFILE_ID_V1,
        capture_policy_id="external_episode_owner_hold_before_unbind_v1",
        explicit_exclusions=explicit_exclusions,
    )


def build_cs6a_human_replay_execution_manifest(
    runtime: MujocoDeployRuntime,
    station_manifest: ControlStationRunManifest,
) -> ControlExecutionManifest:
    """Freeze the source/target scope required by formal sim capture."""

    if not isinstance(runtime, MujocoDeployRuntime):
        raise TypeError("runtime must be a MujocoDeployRuntime.")
    if not isinstance(station_manifest, ControlStationRunManifest):
        raise TypeError("station_manifest must be a ControlStationRunManifest.")
    if (
        station_manifest.manifest_id
        != CS6A_HUMAN_REPLAY_HANDOFF_MANIFEST_ID_V1
    ):
        raise ValueError("CS6a execution manifest requires the exact run manifest.")
    if runtime.reference_activation_spec.activation_mode != "scheduled_segment":
        raise ValueError("CS6a formal capture requires scheduled qpos activation.")
    evidence = (station_manifest.semantic_content_hash,)
    exclusions = tuple(
        ExecutionProfileExclusion.create(
            capability_id=capability_id,
            reason="human_replay_discrete_handoff_sim_fixture_only",
            evidence_ids=evidence,
        )
        for capability_id in (
            "continuous_source_blend",
            "hardware_device_initialization",
            "physical_human_device",
            "policy_model",
            "production_station_entrypoint",
            "real_backend",
            "rtc_policy_source",
        )
    )
    return build_control_execution_manifest(
        manifest_id=CS6A_HUMAN_REPLAY_EXECUTION_MANIFEST_ID_V1,
        manifest_version="v1",
        action_space=runtime.catalog.action_space,
        action_schema=runtime.action_schema,
        component_target_capability=runtime.component_target_capability,
        catalog=runtime.catalog,
        motion_authority_sources=(
            "human:synthetic_fixture",
            "replay:immutable_source_frame",
        ),
        human_motion_proposal_status="authorized",
        safety_control_sources=("runtime_safety_supervisor",),
        safety_control_requests=("hold",),
        lifecycle_control_sources=("cs6a_test_composition",),
        lifecycle_control_requests=("start", "stop"),
        capability=runtime.execution_capability,
        activation=runtime.reference_activation_spec,
        actual_reference_frame_producer="not_applicable",
        exclusions=exclusions,
    )


def build_cs6b1f_pose_human_replay_execution_manifest(
    runtime: MujocoDeployRuntime,
    station_manifest: ControlStationRunManifest,
) -> ControlExecutionManifest:
    """Freeze the opt-in pose HIL source/target facts used by sim capture."""

    if not isinstance(runtime, MujocoDeployRuntime):
        raise TypeError("runtime must be a MujocoDeployRuntime.")
    if not isinstance(station_manifest, ControlStationRunManifest):
        raise TypeError("station_manifest must be a ControlStationRunManifest.")
    if (
        station_manifest.manifest_id
        != CS6B1F_POSE_HUMAN_REPLAY_STATION_MANIFEST_ID_V1
    ):
        raise ValueError("CS6b1f execution manifest requires the exact HIL manifest.")
    if runtime.catalog.action_space != BIMANUAL_EE_POSE_V1:
        raise ValueError("CS6b1f formal capture requires pose proposals.")
    if runtime.reference_activation_spec.activation_mode != "scheduled_segment":
        raise ValueError("CS6b1f formal capture requires scheduled activation.")
    if (
        station_manifest.trace_requirement_profile_id
        != CS6B1F_POSE_HUMAN_REPLAY_TRACE_PROFILE_ID_V1
    ):
        raise ValueError("CS6b1f manifest lacks the pose HIL formal profile.")
    evidence = (station_manifest.semantic_content_hash,)
    exclusions = tuple(
        ExecutionProfileExclusion.create(
            capability_id=capability_id,
            reason="pose_human_replay_sim_hil_fixture_only",
            evidence_ids=evidence,
        )
        for capability_id in (
            "continuous_source_blend",
            "hardware_device_initialization",
            "physical_human_device",
            "policy_model",
            "production_station_entrypoint",
            "real_backend",
            "rtc_policy_source",
        )
    )
    return build_control_execution_manifest(
        manifest_id=CS6B1F_POSE_HUMAN_REPLAY_EXECUTION_MANIFEST_ID_V1,
        manifest_version="v1",
        action_space=runtime.catalog.action_space,
        action_schema=runtime.action_schema,
        component_target_capability=runtime.component_target_capability,
        catalog=runtime.catalog,
        motion_authority_sources=(
            "human:passive_pose_via_virtual_sim_readiness",
            "replay:immutable_source_frame",
        ),
        human_motion_proposal_status="authorized",
        safety_control_sources=("runtime_safety_supervisor",),
        safety_control_requests=("hold",),
        lifecycle_control_sources=("external_episode_trace_owner",),
        lifecycle_control_requests=("prepare", "bind", "release", "finalize"),
        capability=runtime.execution_capability,
        activation=runtime.reference_activation_spec,
        actual_reference_frame_producer="mujoco_pose_to_qpos_resolver",
        exclusions=exclusions,
    )


@dataclass(frozen=True)
class MujocoHumanReplayBindingReceipt:
    manifest_hash: str
    human_adapter_contract_hash: str
    replay_source_contract_hash: str
    replay_action_contract_hash: str
    replay_observer_binding_hash: str
    target_runtime_action_contract_hash: str
    authority_policy_hash: str
    owner_token_id: str
    execution_epoch_snapshot_hash: str
    discrete_handoff_only: bool
    cursor_owned_by_control_station: bool
    authoritative_data_plane_bound: bool
    source_reactivation_supported: bool
    station_authority_lifecycle_bound: bool
    semantic_content_hash: str


@dataclass(frozen=True)
class HumanReplayAuthorityHandoffReceipt:
    handoff_sequence: int
    previous_source_slot_id: str | None
    active_source_slot_id: str
    active_producer_id: str
    hold_gate_event_id: str
    released_authority_event_id: str | None
    takeover_authority_event_id: str
    ready_gate_event_id: str
    authority_epoch: int
    execution_epoch: int
    execution_epoch_snapshot_hash: str
    semantic_content_hash: str


@dataclass(frozen=True)
class MujocoHumanReplayHumanSubmission:
    emission: HumanCommandEmission
    adapter_result: HumanProposalAdapterResult
    proposal_admission: ProposalAdmissionReceipt | None

    @property
    def accepted(self) -> bool:
        return bool(
            self.adapter_result.source_adapter_receipt.authoritative_eligible
            and self.proposal_admission is not None
            and self.proposal_admission.accepted
        )


@dataclass(frozen=True)
class PoseMixingShadowEvaluation:
    """Non-authoritative strategy result compared with the existing arbiter."""

    strategy_id: str
    strategy_spec_hash: str
    mixing_decision_id: str | None
    mixing_decision_hash: str | None
    authoritative_result_id: str | None
    authoritative_selected_proposal_ids: tuple[str, ...]
    strategy_selected_proposal_ids: tuple[str, ...]
    selection_parity: bool
    output_pose_parity: bool
    status: str
    detail: str
    semantic_content_hash: str


@dataclass(frozen=True)
class _PendingHumanEmissionPromotion:
    """Exact preview identity that must become the first formal emission."""

    source_emission_id: str
    source_emission_semantic_hash: str
    effective_from: float
    effective_until: float
    target_effective_at: float
    canonical_safety_receipt_id: str | None


class VirtualSimHumanTakeoverReadinessProvider:
    """Turn a passive pose observation into an explicitly virtual sim gate.

    The provider is intentionally application-local and never used by the real
    teleoperation path.  It waits on the passive source's bounded ingress in
    the Station side-effect worker, then supplies synthetic deadman/rebase
    evidence so the existing CS-6b0 transaction can be exercised end to end.
    The resulting emission is still promoted exactly once by the existing
    ``submit_human_emission`` seam; this class owns neither a queue nor an
    authority lease.
    """

    def __init__(
        self,
        passive_source: PassiveTeleopPoseSessionSource,
        *,
        clock: Callable[[], float],
        evidence_prefix: str = "virtual-sim-hil",
        effective_horizon_sec: float = 0.1,
    ) -> None:
        if not isinstance(passive_source, PassiveTeleopPoseSessionSource):
            raise TypeError("passive_source must be a PassiveTeleopPoseSessionSource.")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        prefix = str(evidence_prefix).strip()
        if not prefix:
            raise ValueError("evidence_prefix must be non-empty.")
        horizon = float(effective_horizon_sec)
        if horizon <= 0.0:
            raise ValueError("effective_horizon_sec must be positive.")
        self.passive_source = passive_source
        self.clock = clock
        self.evidence_prefix = prefix
        self.effective_horizon_sec = horizon

    @property
    def observer_binding_id(self) -> str:
        return self.passive_source.spec.observer_binding_id

    def acquire(
        self,
        *,
        operation_spec: HumanTakeoverOperationSpec,
        takeover_state: HumanTakeoverState,
        timeout_sec: float,
    ) -> HumanTakeoverReadinessCandidate:
        if not isinstance(operation_spec, HumanTakeoverOperationSpec):
            raise TypeError("operation_spec must be a HumanTakeoverOperationSpec.")
        if not isinstance(takeover_state, HumanTakeoverState):
            raise TypeError("takeover_state must be a HumanTakeoverState.")
        if takeover_state.phase != "awaiting_human_readiness":
            raise RuntimeError(
                "Sim HIL readiness requires an awaiting_human_readiness takeover state."
            )
        takeover = takeover_state.takeover_receipt
        if takeover is None:
            raise RuntimeError("Human takeover readiness has no authority receipt.")
        if operation_spec.readiness_observer_binding_id != self.observer_binding_id:
            raise ValueError("Readiness operation is bound to another observer.")
        timeout = float(timeout_sec)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout_sec must be finite and positive.")
        boundary = float(takeover.completed_at)
        # A passive observer may have sampled before HOLD.  Do not let that
        # sample satisfy the post-takeover gate even when timestamps tie.
        emission = self.passive_source.ingress.wait_for_post_boundary_emission(
            boundary_at=boundary + 1e-12,
            timeout_sec=timeout,
        )
        if emission.emitted_at <= boundary:
            raise RuntimeError("Passive human emission did not cross the takeover boundary.")
        deadline = time.monotonic() + timeout
        while float(self.clock()) < float(emission.emitted_at):
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Sim HIL clock did not reach the passive emission timestamp."
                )
            time.sleep(0.0005)
        effective_from = max(float(self.clock()), boundary, float(emission.emitted_at))
        return HumanTakeoverReadinessCandidate(
            observer_binding_id=self.observer_binding_id,
            emission=emission,
            effective_from=effective_from,
            effective_until=effective_from + self.effective_horizon_sec,
            target_effective_at=effective_from,
            deadman_rearm_event_id=(
                f"{self.evidence_prefix}:deadman:{takeover.request_id}"
            ),
            deadman_rearmed_at=effective_from,
            retargeter_rebase_receipt_id=(
                f"{self.evidence_prefix}:rebase:{takeover.request_id}"
            ),
            retargeter_rebased_at=effective_from,
        )


class MujocoHumanReplayHandoffComposition:
    """One shared core with exactly one publishable source at a time."""

    def __init__(
        self,
        *,
        manifest: ControlStationRunManifest,
        runtime: MujocoDeployRuntime,
        human_adapter: HumanProposalAdapter,
        human_source_spec: (
            SyntheticHumanQposSourceSpec | HumanPoseSourceContract
        ),
        replay_source: ReplayProposalSource,
        replay_observer_binding: ReplaySourceObserverBinding,
        pose_mixing_strategy: PoseMixingStrategy | None = None,
        pose_fusion_shadow_spec: CompositionSpec | None = None,
    ) -> None:
        self.manifest = manifest
        self.runtime = runtime
        self.human_adapter = human_adapter
        self.human_source_spec = human_source_spec
        self.replay_source = replay_source
        self.replay_observer_binding = replay_observer_binding
        self.pose_mixing_strategy = pose_mixing_strategy
        self.pose_fusion_shadow_spec = pose_fusion_shadow_spec
        self._lock = threading.RLock()
        self._started = False
        self._active_source_slot_id: str | None = None
        self._activated_source_slot_ids: set[str] = set()
        self._handoff_sequence = 0
        self._safe_human_takeover_state: HumanTakeoverState | None = None
        self._pending_human_emission_promotion: (
            _PendingHumanEmissionPromotion | None
        ) = None
        self._last_pose_mixing_decision: PoseMixingDecision | None = None
        self._last_pose_mixing_shadow_evaluation: (
            PoseMixingShadowEvaluation | None
        ) = None
        self._last_pose_fusion_shadow_submission: (
            CompositeProposalSubmission | None
        ) = None
        self._pose_fusion_shadow_input_receipt_by_role: dict[
            str, CompositionInputAdmissionReceipt
        ] = {}
        self._pose_fusion_shadow_error_by_role: dict[str, str] = {}
        self._last_pose_fusion_shadow_error: str | None = None
        self._pose_fusion_shadow_human_fact_time_by_emission_id: dict[
            str, float
        ] = {}
        self.binding_receipt = self._validate_exact_binding()

    @property
    def started(self) -> bool:
        with self._lock:
            return self._started

    @property
    def active_source_slot_id(self) -> str | None:
        with self._lock:
            return self._active_source_slot_id

    @property
    def safe_human_takeover_state(self) -> HumanTakeoverState | None:
        with self._lock:
            return self._safe_human_takeover_state

    @property
    def last_pose_mixing_decision(self) -> PoseMixingDecision | None:
        with self._lock:
            return self._last_pose_mixing_decision

    @property
    def last_pose_mixing_shadow_evaluation(
        self,
    ) -> PoseMixingShadowEvaluation | None:
        with self._lock:
            return self._last_pose_mixing_shadow_evaluation

    @property
    def last_pose_fusion_shadow_submission(
        self,
    ) -> CompositeProposalSubmission | None:
        with self._lock:
            return self._last_pose_fusion_shadow_submission

    @property
    def pose_fusion_shadow_input_receipt_by_role(
        self,
    ) -> dict[str, CompositionInputAdmissionReceipt]:
        with self._lock:
            return dict(self._pose_fusion_shadow_input_receipt_by_role)

    @property
    def pose_fusion_shadow_error_by_role(self) -> dict[str, str]:
        with self._lock:
            return dict(self._pose_fusion_shadow_error_by_role)

    @property
    def last_pose_fusion_shadow_error(self) -> str | None:
        with self._lock:
            return self._last_pose_fusion_shadow_error

    def _validate_exact_binding(self) -> MujocoHumanReplayBindingReceipt:
        manifest_builder = (
            build_cs6b1f_pose_human_replay_station_manifest
            if self.manifest.manifest_id
            == CS6B1F_POSE_HUMAN_REPLAY_STATION_MANIFEST_ID_V1
            else build_cs6a_human_replay_handoff_manifest
        )
        expected = manifest_builder(
            self.runtime,
            self.human_adapter,
            self.human_source_spec,
            self.replay_source,
            self.replay_observer_binding,
        )
        if self.manifest.semantic_content_hash != expected.semantic_content_hash:
            raise ValueError("CS6a human/replay manifest binding drifted.")
        config = self.runtime.config
        replay_config = self.replay_source.config
        pose_native = isinstance(
            self.human_source_spec,
            HumanPoseSourceContract,
        )
        expected_action_space = (
            BIMANUAL_EE_POSE_V1 if pose_native else BIMANUAL_QPOS_V1
        )
        if not pose_native and self.pose_mixing_strategy is not None:
            raise ValueError("Pose mixing strategy cannot be attached to qpos CS6a.")
        if not pose_native and self.pose_fusion_shadow_spec is not None:
            raise ValueError("Pose fusion composition cannot be attached to qpos CS6a.")
        if self.pose_mixing_strategy is not None:
            strategy_spec = getattr(self.pose_mixing_strategy, "spec", None)
            if strategy_spec is None or not callable(
                getattr(self.pose_mixing_strategy, "decide", None)
            ):
                raise TypeError("pose_mixing_strategy does not satisfy its protocol.")
        if self.pose_fusion_shadow_spec is not None:
            spec = self.pose_fusion_shadow_spec
            if not isinstance(spec, CompositionSpec):
                raise TypeError("pose_fusion_shadow_spec must be a CompositionSpec.")
            if (
                spec.algebra != "weighted_se3_pose_fusion"
                or spec.input_roles
                != (
                    CS6B1G_HUMAN_POSE_INPUT_ROLE_V1,
                    CS6B1G_REPLAY_POSE_INPUT_ROLE_V1,
                )
                or spec.required_roles != spec.input_roles
            ):
                raise ValueError(
                    "CS6b1g shadow requires fail-closed human_pose/replay_pose "
                    "weighted SE(3) composition."
                )
        if self.runtime.catalog.action_space != expected_action_space:
            raise ValueError(
                "Mixed human/replay source representation mismatches the target."
            )
        if self.runtime.reference_activation_spec.activation_mode != "scheduled_segment":
            raise ValueError("Mixed human/replay requires scheduled execution.")
        if not config.enable_replay_authority or config.enable_composer_authority:
            raise ValueError(
                "CS6a requires replay authority and forbids composer authority."
            )
        if self.runtime.authoritative_port.profile.initial_control_mode != "hybrid":
            raise ValueError("CS6a requires explicit hybrid control mode.")
        if config.producer_id == config.replay_producer_id:
            raise ValueError("Human and replay producers must be distinct.")
        if (
            self.human_source_spec.source_id
            == self.replay_observer_binding.source_slot_id
        ):
            raise ValueError("Human and replay source slots must be distinct.")
        allowed = set(
            self.runtime.authoritative_port.profile.takeover_allowed_producers
        )
        expected_allowed = {config.producer_id, config.replay_producer_id}
        if allowed != expected_allowed:
            raise ValueError("CS6a requires exact human/replay takeover allowlist.")
        human_config = self.human_adapter.config
        if (
            human_config.runtime_session_id != config.runtime_session_id
            or human_config.control_domain_id != config.control_domain_id
            or human_config.embodiment_instance_id != config.embodiment_instance_id
            or human_config.producer_id != config.producer_id
            or human_config.producer_epoch != config.producer_epoch
            or human_config.component_catalog.semantic_content_hash
            != self.runtime.catalog.semantic_content_hash
            or human_config.action_schema.semantic_content_hash
            != self.runtime.action_schema.semantic_content_hash
        ):
            raise ValueError("Human source is not bound to the exact MuJoCo runtime.")
        if pose_native:
            reference_frames = dict(
                self.human_source_spec.reference_frame_by_effector
            )
            target_effectors = {
                component.effector_name for component in self.runtime.catalog.components
            }
            if set(reference_frames) != target_effectors:
                raise ValueError(
                    "Pose mixed human source must exactly cover the target pose effectors."
                )
            configured_specs = dict(human_config.action_adapter.arm_specs or {})
            expected_sides: set[str] = set()
            for effector_name in target_effectors:
                try:
                    side, kind = split_effector_name(effector_name)
                except ValueError as exc:
                    raise ValueError(
                        "Pose human adapter currently requires canonical arm effector IDs; "
                        f"got {effector_name!r}."
                    ) from exc
                if kind != "arm":
                    raise ValueError(
                        "Pose human adapter accepts arm pose components only; "
                        f"got {effector_name!r}."
                    )
                expected_sides.add(side)
            if set(configured_specs) != expected_sides:
                raise ValueError(
                    "Pose human adapter must configure one action spec per target arm."
                )
            if {
                f"{side}_arm": configured_specs[side].reference_frame
                for side in sorted(expected_sides)
            } != reference_frames:
                raise ValueError(
                    "Pose human adapter reference frames mismatch the source spec."
                )
        elif self.human_source_spec.joint_dof_by_effector != (
            ("left_arm", 6),
            ("right_arm", 6),
        ):
            raise ValueError("CS6a human source must be exact coupled bimanual q6.")
        if (
            replay_config.runtime_session_id != config.runtime_session_id
            or replay_config.control_domain_id != config.control_domain_id
            or replay_config.embodiment_instance_id != config.embodiment_instance_id
            or replay_config.producer_id != config.replay_producer_id
            or replay_config.producer_epoch != config.producer_epoch
            or replay_config.action_schema.semantic_content_hash
            != self.runtime.action_schema.semantic_content_hash
        ):
            raise ValueError("Replay source is not bound to the exact MuJoCo runtime.")
        if (
            replay_config.requested_use
            != REPLAY_BINDING_USE_EXECUTION_CANDIDATE
            or replay_config.binding_assessment.verdict
            != REPLAY_BINDING_VERIFIED_COMPATIBLE
            or replay_config.source_action_contract.action_space
            != expected_action_space
            or replay_config.target_runtime_contract.semantic_content_hash
            != self.runtime.target_runtime_action_contract.semantic_content_hash
        ):
            raise ValueError(
                "Replay source is not exact-target representation compatible."
            )
        if (
            self.replay_observer_binding.source_locator_id
            != replay_config.source_contract.source_dataset_id
            or self.replay_observer_binding.source_start_index
            != replay_config.schedule.source_start_index
            or self.replay_observer_binding.clock_domain_id
            != config.clock_domain_id
        ):
            raise ValueError("Replay observer binding does not match the source/runtime.")
        owner_snapshot = (
            self.runtime.authoritative_port.capture_dispatch_owner_snapshot()
        )
        token = owner_snapshot.current_token
        if token is None or token.owner_kind != "shared_arbitration_runtime":
            raise ValueError("MuJoCo target lacks the shared authoritative owner token.")
        execution_snapshot = (
            self.runtime.authoritative_port.capture_execution_epoch_snapshot()
        )
        payload = {
            "manifest_hash": self.manifest.semantic_content_hash,
            "human_adapter_contract_hash": self.human_adapter.contract_hash,
            "replay_source_contract_hash": (
                replay_config.source_contract.semantic_content_hash
            ),
            "replay_action_contract_hash": (
                replay_config.source_action_contract.semantic_content_hash
            ),
            "replay_observer_binding_hash": (
                self.replay_observer_binding.semantic_content_hash
            ),
            "target_runtime_action_contract_hash": (
                self.runtime.target_runtime_action_contract.semantic_content_hash
            ),
            "authority_policy_hash": (
                self.runtime.authoritative_port.profile.authority_policy_hash
            ),
            "owner_token_id": token.token_id,
            "execution_epoch_snapshot_hash": execution_snapshot.semantic_content_hash,
            "discrete_handoff_only": True,
            "cursor_owned_by_control_station": False,
            "authoritative_data_plane_bound": True,
            "source_reactivation_supported": False,
            "station_authority_lifecycle_bound": True,
        }
        return MujocoHumanReplayBindingReceipt(
            **payload,
            semantic_content_hash=canonical_content_hash(
                payload,
                domain=CS6A_BINDING_RECEIPT_HASH_DOMAIN_V1,
            ),
        )

    def start(self) -> MujocoHumanReplayBindingReceipt:
        with self._lock:
            if not self._started:
                self.runtime.start()
                self._started = True
            return self.binding_receipt

    def handoff_to(
        self,
        source_slot_id: str,
        *,
        reason: str,
    ) -> HumanReplayAuthorityHandoffReceipt:
        """HOLD/cancel the old epoch, then transfer the exact arms lease."""

        target_slot = str(source_slot_id).strip()
        normalized_reason = str(reason).strip()
        if not target_slot or not normalized_reason:
            raise ValueError("source_slot_id and reason must be non-empty.")
        producer_by_slot = {
            self.human_source_spec.source_id: self.runtime.config.producer_id,
            self.replay_observer_binding.source_slot_id: (
                self.runtime.config.replay_producer_id
            ),
        }
        try:
            target_producer = producer_by_slot[target_slot]
        except KeyError as exc:
            raise ValueError(f"Unknown CS6a source slot: {target_slot!r}.") from exc
        with self._lock:
            if not self._started:
                raise RuntimeError("MujocoHumanReplayHandoffComposition.start() is required.")
            if self._safe_human_takeover_state is not None:
                raise RuntimeError(
                    "Legacy CS6a handoff is unavailable after a CS6b0 safe "
                    "human takeover transaction starts."
                )
            if target_slot == self._active_source_slot_id:
                raise RuntimeError("Requested CS6a source is already active.")
            if target_slot in self._activated_source_slot_ids:
                raise RuntimeError(
                    "CS6a v1 forbids source reactivation within one session."
                )
            sequence = self._handoff_sequence
            self._handoff_sequence += 1
            previous_slot = self._active_source_slot_id
            self._active_source_slot_id = None
            hold_outcome = self.runtime.transition_execution_gate(
                "hold",
                reason=f"cs6a_handoff_hold:{normalized_reason}",
                source_event_ids=(f"cs6a-handoff:{sequence}",),
            )
            hold_event = hold_outcome.event
            if (
                hold_outcome.cancellation_receipt is not None
                and not hold_outcome.cancellation_receipt.complete
            ):
                raise RuntimeError(
                    "CS6a authority handoff requires completed lower HOLD."
                )
            release_event = None
            if previous_slot is not None:
                release_event = self.runtime.authoritative_port.admit_authority_event(
                    event_type="release",
                    selection_group_id="arms",
                    reason=f"cs6a_handoff_release:{normalized_reason}",
                )
            takeover_event = self.runtime.authoritative_port.admit_authority_event(
                event_type="takeover",
                selection_group_id="arms",
                owner_producer_id=target_producer,
                reason=f"cs6a_handoff_takeover:{normalized_reason}",
            )
            ready_event = self.runtime.set_execution_gate(
                "ready",
                reason=f"cs6a_handoff_ready:{normalized_reason}",
                source_event_ids=(takeover_event.event_id,),
            )
            execution_snapshot = (
                self.runtime.authoritative_port.capture_execution_epoch_snapshot()
            )
            self._active_source_slot_id = target_slot
            self._activated_source_slot_ids.add(target_slot)
            payload = {
                "handoff_sequence": sequence,
                "previous_source_slot_id": previous_slot,
                "active_source_slot_id": target_slot,
                "active_producer_id": target_producer,
                "hold_gate_event_id": hold_event.event_id,
                "released_authority_event_id": (
                    None if release_event is None else release_event.event_id
                ),
                "takeover_authority_event_id": takeover_event.event_id,
                "ready_gate_event_id": ready_event.event_id,
                "authority_epoch": takeover_event.authority_epoch,
                "execution_epoch": execution_snapshot.execution_epoch,
                "execution_epoch_snapshot_hash": (
                    execution_snapshot.semantic_content_hash
                ),
            }
            return HumanReplayAuthorityHandoffReceipt(
                **payload,
                semantic_content_hash=canonical_content_hash(
                    payload,
                    domain=CS6A_HANDOFF_RECEIPT_HASH_DOMAIN_V1,
                ),
            )

    def begin_safe_replay_to_human_takeover(
        self,
        policy: MixedControlBoundaryPolicy,
        *,
        reason: str,
        capture_boundary_id: str | None = None,
    ) -> HumanTakeoverState:
        """Run the lower HOLD/invalidate/release/takeover half of CS-6b0.

        This incremental path is deliberately one-way and sim-only.  It uses
        the real MuJoCo execution cancellation receipt and producer/authority
        epochs, but stops with both source publication paths closed.  A later
        :meth:`complete_human_takeover_readiness` call must provide the post-
        takeover human readiness evidence before READY can reopen.
        """

        if not isinstance(policy, MixedControlBoundaryPolicy):
            raise TypeError("policy must be a MixedControlBoundaryPolicy.")
        normalized_reason = str(reason).strip()
        if not normalized_reason:
            raise ValueError("reason must be non-empty.")
        if policy.episode_boundary_motion_policy != "hold_before_unbind":
            raise ValueError(
                "CS6b0 safe takeover requires hold_before_unbind episode semantics."
            )
        with self._lock:
            if not self._started:
                raise RuntimeError(
                    "MujocoHumanReplayHandoffComposition.start() is required."
                )
            if self._safe_human_takeover_state is not None:
                raise RuntimeError("A CS6b0 safe human takeover already exists.")
            replay_slot = self.replay_observer_binding.source_slot_id
            human_slot = self.human_source_spec.source_id
            if self._active_source_slot_id != replay_slot:
                raise RuntimeError(
                    "CS6b0 v1 only supports an active replay-to-human transition."
                )
            if human_slot in self._activated_source_slot_ids:
                raise RuntimeError(
                    "CS6b0 v1 requires a human source not previously activated."
                )

            sequence = self._handoff_sequence
            self._handoff_sequence += 1
            requested_at = float(self.runtime.clock())
            execution_before = (
                self.runtime.authoritative_port.capture_execution_epoch_snapshot()
            )
            owner_before = (
                self.runtime.authoritative_port.capture_dispatch_owner_snapshot()
            )
            replay_producer = self.runtime.config.replay_producer_id
            human_producer = self.runtime.config.producer_id
            replay_epoch = self.runtime.authoritative_port.current_producer_epoch(
                replay_producer
            )
            if replay_epoch is None:
                raise RuntimeError("The replay producer has no active epoch.")
            request = HumanTakeoverRequest(
                request_id=(
                    "cs6b0-human-takeover:"
                    f"{self.runtime.config.runtime_session_id}:{sequence}"
                ),
                runtime_session_id=self.runtime.config.runtime_session_id,
                control_domain_id=self.runtime.config.control_domain_id,
                selection_group_id="arms",
                previous_source_slot_id=replay_slot,
                target_human_source_slot_id=human_slot,
                previous_producer_id=replay_producer,
                target_human_producer_id=human_producer,
                previous_producer_epoch=replay_epoch,
                authority_epoch=owner_before.current_authority_epoch,
                execution_epoch=execution_before.execution_epoch,
                requested_at=requested_at,
                clock_domain_id=self.runtime.config.clock_domain_id,
                reason=normalized_reason,
                capture_boundary_id=capture_boundary_id,
            )
            state = begin_human_takeover(policy, request)
            self._safe_human_takeover_state = state

            # Publication closure is the application-level linearization point.
            # Every lower mutation below happens while neither source can submit.
            self._active_source_slot_id = None
            hold_outcome = self.runtime.transition_execution_gate(
                "hold",
                reason=f"cs6b0_human_takeover_hold:{normalized_reason}",
                source_event_ids=(request.request_id,),
                force_cancellation=True,
            )
            hold_event = hold_outcome.event
            execution_after = (
                self.runtime.authoritative_port.capture_execution_epoch_snapshot()
            )
            cancellation = hold_outcome.cancellation
            cancellation_receipt = hold_outcome.cancellation_receipt
            if cancellation is None or cancellation_receipt is None:
                raise RuntimeError("CS6b0 HOLD produced no execution cancellation.")
            cancellation_id = cancellation.cancellation_id
            state = apply_execution_hold(
                state,
                ExecutionHoldReceipt(
                    request_id=request.request_id,
                    prior_execution_epoch=execution_before.execution_epoch,
                    execution_epoch=execution_after.execution_epoch,
                    hold_acknowledged=cancellation_receipt.complete,
                    completed_at=cancellation_receipt.observed_at,
                    evidence_refs=(hold_event.event_id, cancellation_id),
                ),
            )
            self._safe_human_takeover_state = state
            if state.phase == "failed":
                self.runtime.set_execution_gate(
                    "fault",
                    reason="cs6b0_human_takeover_hold_failed",
                    source_event_ids=(cancellation_id,),
                )
                return state

            next_replay_epoch = replay_epoch + 1
            activation = self.runtime.authoritative_port.activate_producer(
                producer_id=replay_producer,
                producer_epoch=next_replay_epoch,
                reason=f"cs6b0_invalidate_replay:{normalized_reason}",
            )
            replay_invalidated = (
                self.runtime.authoritative_port.current_producer_epoch(replay_producer)
                == next_replay_epoch
            )
            state = apply_source_invalidation(
                state,
                SourceInvalidationReceipt(
                    request_id=request.request_id,
                    producer_id=replay_producer,
                    prior_producer_epoch=replay_epoch,
                    producer_epoch=next_replay_epoch,
                    proposal_buffer_invalidated=replay_invalidated,
                    worker_drain_status="not_applicable",
                    completed_at=float(self.runtime.clock()),
                    evidence_refs=(activation.activation_id,),
                ),
            )
            self._safe_human_takeover_state = state
            if state.phase == "failed":
                return state

            release_event = self.runtime.authoritative_port.admit_authority_event(
                event_type="release",
                selection_group_id="arms",
                reason=f"cs6b0_release_replay:{normalized_reason}",
            )
            state = apply_authority_release(
                state,
                AuthorityReleaseReceipt(
                    request_id=request.request_id,
                    released_producer_id=replay_producer,
                    prior_authority_epoch=owner_before.current_authority_epoch,
                    authority_epoch=release_event.authority_epoch,
                    accepted=True,
                    completed_at=release_event.available_at,
                    evidence_refs=(release_event.event_id,),
                ),
            )
            self._safe_human_takeover_state = state

            takeover_event = self.runtime.authoritative_port.admit_authority_event(
                event_type="takeover",
                selection_group_id="arms",
                owner_producer_id=human_producer,
                reason=f"cs6b0_takeover_human:{normalized_reason}",
            )
            state = apply_human_authority_takeover(
                state,
                HumanAuthorityTakeoverReceipt(
                    request_id=request.request_id,
                    human_producer_id=human_producer,
                    prior_authority_epoch=release_event.authority_epoch,
                    authority_epoch=takeover_event.authority_epoch,
                    accepted=True,
                    completed_at=takeover_event.available_at,
                    evidence_refs=(takeover_event.event_id,),
                ),
            )
            self._safe_human_takeover_state = state
            return state

    def complete_safe_human_takeover(
        self,
        emission: HumanCommandEmission,
        *,
        effective_from: float,
        effective_until: float,
        target_effective_at: float,
        deadman_rearm_event_id: str,
        deadman_rearmed_at: float,
        retargeter_rebase_receipt_id: str,
        retargeter_rebased_at: float,
    ) -> HumanTakeoverState:
        """Validate exact post-takeover source readiness, then reopen READY.

        Adaptation is non-publishing.  Canonical-safety preview is performed
        only for the legacy compatibility policy; the mainline policy leaves
        safety to the later authoritative execution path.  The exact emission
        identity and timing are retained and must still pass once through
        :meth:`submit_human_emission`, which remains the sole formal emission
        and proposal-admission seam.
        """

        with self._lock:
            if not self._started:
                raise RuntimeError(
                    "MujocoHumanReplayHandoffComposition.start() is required."
                )
            state = self._safe_human_takeover_state
            if state is None:
                raise RuntimeError("No CS6b0 safe human takeover is pending.")
            if state.phase != "awaiting_human_readiness":
                raise RuntimeError(
                    "CS6b0 human readiness requires awaiting_human_readiness."
                )
            assert state.takeover_receipt is not None
            if emission.emitted_at <= state.takeover_receipt.completed_at:
                raise ValueError(
                    "The first human sample must be emitted after authority takeover."
                )

            preview = self.human_adapter.adapt(
                emission,
                effective_from=effective_from,
                effective_until=effective_until,
                target_effective_at=target_effective_at,
            )
            adapter_receipt = preview.source_adapter_receipt
            if (
                not adapter_receipt.authoritative_eligible
                or preview.proposal_draft is None
                or preview.robot_action is None
            ):
                raise ValueError(
                    "The post-takeover human emission is not authoritative eligible."
                )
            safety_receipt_id: str | None = None
            safety_status: str | None = None
            if state.policy.require_canonical_safety:
                safety_input_id = (
                    f"cs6b0-safety-preview:{state.request.request_id}:"
                    f"{adapter_receipt.source_emission_id}"
                )
                safety = self.runtime.authoritative_port.evaluate_canonical_safety(
                    preview.robot_action,
                    input_action_id=safety_input_id,
                )
                if safety.input_action_id != safety_input_id:
                    raise ValueError(
                        "Canonical safety preview references another human action."
                    )
                if (
                    safety.status in {"safe", "modified"}
                    and safety.safe_action is None
                ):
                    raise ValueError(
                        "An accepted canonical safety preview requires a safe action."
                    )
                safe_action_hash = (
                    None
                    if safety.safe_action is None
                    else canonical_content_hash(
                        robot_action_semantic_payload(safety.safe_action),
                        domain=CS6B0_CANONICAL_SAFETY_PREVIEW_HASH_DOMAIN_V1,
                    )
                )
                safety_hash = canonical_content_hash(
                    {
                        "request_id": state.request.request_id,
                        "source_emission_id": adapter_receipt.source_emission_id,
                        "input_action_id": safety.input_action_id,
                        "status": safety.status,
                        "safe_action_hash": safe_action_hash,
                        "modified_components": safety.modified_components,
                        "reasons": safety.reasons,
                        "safety_contract_id": safety.safety_contract_id,
                        "safety_contract_hash": safety.safety_contract_hash,
                        "anchor_id": safety.anchor_id,
                        "anchor_kind": safety.anchor_kind,
                        "anchor_hash": safety.anchor_hash,
                    },
                    domain=CS6B0_CANONICAL_SAFETY_PREVIEW_HASH_DOMAIN_V1,
                )
                safety_receipt_id = f"cs6b0-canonical-safety:{safety_hash}"
                safety_status = safety.status

            measurement = self.runtime.capture_qpos_measurement()
            measured_at = measurement.measured_at
            measured_qpos = measurement.qpos_by_side
            measured_hash = canonical_content_hash(
                {
                    "request_id": state.request.request_id,
                    "measured_at": measured_at,
                    "qpos_by_side": measured_qpos,
                },
                domain=CS6B0_HUMAN_READINESS_MEASUREMENT_HASH_DOMAIN_V1,
            )
            readiness = HumanReadinessReceipt(
                request_id=state.request.request_id,
                human_emission_id=adapter_receipt.source_emission_id,
                human_emission_available_at=emission.emitted_at,
                measured_state_id=f"mujoco-qpos:{measured_hash}",
                measured_state_available_at=measured_at,
                deadman_rearm_event_id=deadman_rearm_event_id,
                deadman_rearmed_at=deadman_rearmed_at,
                retargeter_rebase_receipt_id=retargeter_rebase_receipt_id,
                retargeter_rebased_at=retargeter_rebased_at,
                canonical_safety_receipt_id=safety_receipt_id,
                canonical_safety_status=safety_status,
                completed_at=float(self.runtime.clock()),
            )
            state = apply_human_readiness(state, readiness)
            self._safe_human_takeover_state = state
            if state.phase == "failed":
                return state

            ready_source_event_ids = (
                (
                    safety_receipt_id,
                    state.takeover_receipt.evidence_refs[0],
                )
                if safety_receipt_id is not None
                else (
                    adapter_receipt.source_emission_id,
                    deadman_rearm_event_id,
                    retargeter_rebase_receipt_id,
                    state.takeover_receipt.evidence_refs[0],
                )
            )
            ready_evidence_refs = (
                (safety_receipt_id,)
                if safety_receipt_id is not None
                else (
                    adapter_receipt.source_emission_id,
                    deadman_rearm_event_id,
                    retargeter_rebase_receipt_id,
                )
            )
            ready_event = self.runtime.set_execution_gate(
                "ready",
                reason=f"cs6b0_human_ready:{state.request.reason}",
                source_event_ids=ready_source_event_ids,
            )
            state = apply_execution_ready(
                state,
                ExecutionReadyReceipt(
                    request_id=state.request.request_id,
                    execution_epoch=state.execution_epoch,
                    authority_epoch=state.authority_epoch,
                    accepted=True,
                    completed_at=ready_event.available_at,
                    evidence_refs=(
                        ready_event.event_id,
                        *ready_evidence_refs,
                    ),
                ),
            )
            self._safe_human_takeover_state = state
            self._pending_human_emission_promotion = (
                _PendingHumanEmissionPromotion(
                    source_emission_id=adapter_receipt.source_emission_id,
                    source_emission_semantic_hash=(
                        adapter_receipt.source_emission_semantic_hash
                    ),
                    effective_from=float(effective_from),
                    effective_until=float(effective_until),
                    target_effective_at=float(target_effective_at),
                    canonical_safety_receipt_id=safety_receipt_id,
                )
            )
            self._active_source_slot_id = self.human_source_spec.source_id
            self._activated_source_slot_ids.add(self.human_source_spec.source_id)
            return state

    def complete_human_takeover_readiness(
        self,
        emission: HumanCommandEmission,
        *,
        effective_from: float,
        effective_until: float,
        target_effective_at: float,
        deadman_rearm_event_id: str,
        deadman_rearmed_at: float,
        retargeter_rebase_receipt_id: str,
        retargeter_rebased_at: float,
    ) -> HumanTakeoverState:
        """Mainline name for source readiness without implied safety ownership."""

        return self.complete_safe_human_takeover(
            emission,
            effective_from=effective_from,
            effective_until=effective_until,
            target_effective_at=target_effective_at,
            deadman_rearm_event_id=deadman_rearm_event_id,
            deadman_rearmed_at=deadman_rearmed_at,
            retargeter_rebase_receipt_id=retargeter_rebase_receipt_id,
            retargeter_rebased_at=retargeter_rebased_at,
        )

    def fail_safe_human_takeover(
        self,
        *,
        reason: str,
        fault: bool = False,
    ) -> HumanTakeoverState:
        """Fail an in-flight readiness wait without reopening either source."""

        normalized_reason = str(reason).strip()
        if not normalized_reason:
            raise ValueError("reason must be non-empty.")
        with self._lock:
            state = self._safe_human_takeover_state
            if state is None:
                raise RuntimeError("No CS6b0 safe human takeover is pending.")
            next_state = fail_human_takeover(
                state,
                reason=normalized_reason,
                gate_status="fault" if fault else "hold",
            )
            self._safe_human_takeover_state = next_state
            self._pending_human_emission_promotion = None
            self._active_source_slot_id = None
            if fault:
                self.runtime.transition_execution_gate(
                    "fault",
                    reason=f"cs6b0_human_takeover_failed:{normalized_reason}",
                    source_event_ids=(state.request.request_id,),
                )
            return next_state

    def _submit_pose_fusion_shadow_input(
        self,
        *,
        role: str,
        producer_id: str,
        producer_epoch: int,
        sequence: int,
        action,
        produced_at: float,
        produced_clock_domain_id: str,
        effective_from: float,
        effective_until: float,
        observation_cut_id: str | None,
        source_fact_ids: tuple[str, ...],
        source_fact_hashes: tuple[str, ...],
        parent_fact_id: str,
        parent_fact_hash: str,
    ) -> CompositionInputAdmissionReceipt:
        if self.pose_fusion_shadow_spec is None:
            raise RuntimeError("Pose fusion composition shadow is not enabled.")
        receipt = self.runtime.composition_port.admit_composition_input(
            CompositionInputDraft(
                runtime_session_id=self.runtime.config.runtime_session_id,
                control_domain_id=self.runtime.config.control_domain_id,
                producer_id=producer_id,
                producer_epoch=producer_epoch,
                sequence=sequence,
                input_role=role,
                action=project_robot_action_v1_to_component_action(
                    action,
                    action_schema=self.runtime.action_schema,
                ),
                produced_at=produced_at,
                produced_clock_domain_id=produced_clock_domain_id,
                effective_from=effective_from,
                effective_until=effective_until,
                effective_clock_domain_id=self.runtime.config.clock_domain_id,
                context_ids=(),
                observation_cut_id=observation_cut_id,
                policy_output_id=None,
                source_fact_ids=source_fact_ids,
                source_fact_hashes=source_fact_hashes,
                authority_candidate=False,
                audit_metadata={
                    "non_authoritative": True,
                    "shadow_profile": "cs6b1g_weighted_se3_pose_fusion_v1",
                    "parent_fact_id": parent_fact_id,
                    "parent_fact_hash": parent_fact_hash,
                },
            )
        )
        self._pose_fusion_shadow_input_receipt_by_role[role] = receipt
        if receipt.accepted:
            self._pose_fusion_shadow_error_by_role.pop(role, None)
        else:
            self._pose_fusion_shadow_error_by_role[role] = (
                f"{receipt.rejection_code}:{receipt.message}"
            )
        return receipt

    def submit_passive_human_pose_fusion_shadow_input(
        self,
        emission: HumanCommandEmission,
        *,
        observed_at: float,
    ) -> CompositionInputAdmissionReceipt | None:
        """Admit one passive pose only as a generic composition input."""

        with self._lock:
            if self.pose_fusion_shadow_spec is None:
                return None
            if not self._started:
                raise RuntimeError(
                    "MujocoHumanReplayHandoffComposition.start() is required."
                )
            try:
                effective_from = float(observed_at)
                effective_until = (
                    effective_from + self.runtime.config.proposal_horizon_sec
                )
                adaptation = self.human_adapter.adapt(
                    emission,
                    effective_from=effective_from,
                    effective_until=effective_until,
                    target_effective_at=effective_from,
                )
                action = adaptation.robot_action
                adapter_receipt = adaptation.source_adapter_receipt
                effective_from = (
                    self._pose_fusion_shadow_human_fact_time_by_emission_id.setdefault(
                        adapter_receipt.source_emission_id,
                        effective_from,
                    )
                )
                effective_until = (
                    effective_from + self.runtime.config.proposal_horizon_sec
                )
                source_facts = project_human_proposal_adapter_result(
                    emission,
                    adaptation,
                    runtime_session_id=self.runtime.config.runtime_session_id,
                    control_domain_id=self.runtime.config.control_domain_id,
                    embodiment_instance_id=(
                        self.runtime.config.embodiment_instance_id
                    ),
                    occurred_at=effective_from,
                    clock_domain_id=self.runtime.config.clock_domain_id,
                )
                self.runtime.authoritative_port.record_external_facts(source_facts)
                source_fact_hash_by_id = {
                    fact.fact_id: fact.semantic_content_hash
                    for fact in source_facts
                }
                if action is None or not adaptation.source_validation.accepted:
                    self._pose_fusion_shadow_error_by_role[
                        CS6B1G_HUMAN_POSE_INPUT_ROLE_V1
                    ] = "human_adapter_rejected:" + ",".join(
                        adaptation.source_validation.reasons
                    )
                    return None
                previous = self._pose_fusion_shadow_input_receipt_by_role.get(
                    CS6B1G_HUMAN_POSE_INPUT_ROLE_V1
                )
                if (
                    previous is not None
                    and previous.accepted
                    and previous.composition_input is not None
                    and previous.composition_input.audit_metadata.get(
                        "parent_fact_id"
                    )
                    == adapter_receipt.source_emission_id
                    and previous.composition_input.audit_metadata.get(
                        "parent_fact_hash"
                    )
                    == adapter_receipt.source_emission_semantic_hash
                ):
                    return previous
                return self._submit_pose_fusion_shadow_input(
                    role=CS6B1G_HUMAN_POSE_INPUT_ROLE_V1,
                    producer_id=self.runtime.config.producer_id,
                    producer_epoch=self.runtime.config.producer_epoch,
                    sequence=emission.emission_sequence,
                    action=action,
                    produced_at=effective_from,
                    produced_clock_domain_id=(
                        self.human_adapter.config.produced_clock_domain_id
                    ),
                    effective_from=effective_from,
                    effective_until=effective_until,
                    observation_cut_id=None,
                    source_fact_ids=(
                        adapter_receipt.source_emission_id,
                        adapter_receipt.representation_trace_id,
                    ),
                    source_fact_hashes=(
                        source_fact_hash_by_id[
                            adapter_receipt.source_emission_id
                        ],
                        source_fact_hash_by_id[
                            adapter_receipt.representation_trace_id
                        ],
                    ),
                    parent_fact_id=adapter_receipt.source_emission_id,
                    parent_fact_hash=(
                        adapter_receipt.source_emission_semantic_hash
                    ),
                )
            except Exception as error:  # Shadow must not alter passive source use.
                self._pose_fusion_shadow_error_by_role[
                    CS6B1G_HUMAN_POSE_INPUT_ROLE_V1
                ] = f"{type(error).__name__}:{error}"
                return None

    def submit_human_emission(
        self,
        emission: HumanCommandEmission,
        *,
        effective_from: float,
        effective_until: float,
        target_effective_at: float,
    ) -> MujocoHumanReplayHumanSubmission:
        with self._lock:
            if not self._started:
                raise RuntimeError(
                    "MujocoHumanReplayHandoffComposition.start() is required."
                )
            if self._active_source_slot_id != self.human_source_spec.source_id:
                raise RuntimeError("Human publication is not the active CS6a authority.")
            result = self.human_adapter.adapt(
                emission,
                effective_from=effective_from,
                effective_until=effective_until,
                target_effective_at=target_effective_at,
            )
            pending_promotion = self._pending_human_emission_promotion
            if pending_promotion is not None:
                adapter_receipt = result.source_adapter_receipt
                if (
                    adapter_receipt.source_emission_id
                    != pending_promotion.source_emission_id
                    or adapter_receipt.source_emission_semantic_hash
                    != pending_promotion.source_emission_semantic_hash
                    or float(effective_from) != pending_promotion.effective_from
                    or float(effective_until) != pending_promotion.effective_until
                    or float(target_effective_at)
                    != pending_promotion.target_effective_at
                ):
                    raise RuntimeError(
                        "The first formal human emission must exactly promote "
                        "the CS6b0 readiness preview."
                    )
            shadow_fact_time = (
                self._pose_fusion_shadow_human_fact_time_by_emission_id.get(
                    result.source_adapter_receipt.source_emission_id
                )
            )
            result, receipt = admit_human_command_emission(
                core=self.runtime.authoritative_port,
                adapter=self.human_adapter,
                emission=emission,
                effective_from=effective_from,
                effective_until=effective_until,
                target_effective_at=target_effective_at,
                occurred_at=(
                    float(self.runtime.clock())
                    if shadow_fact_time is None
                    else shadow_fact_time
                ),
                clock_domain_id=self.runtime.config.clock_domain_id,
                adaptation=result,
            )
            if shadow_fact_time is not None:
                self._pose_fusion_shadow_human_fact_time_by_emission_id.pop(
                    result.source_adapter_receipt.source_emission_id,
                    None,
                )
            if pending_promotion is not None and (
                receipt is not None and receipt.accepted
            ):
                self._pending_human_emission_promotion = None
            return MujocoHumanReplayHumanSubmission(
                emission=emission,
                adapter_result=result,
                proposal_admission=receipt,
            )

    def submit_replay_source_frame_emission(
        self,
        emission: ReplaySourceFrameEmission,
    ) -> MujocoReplaySubmission:
        with self._lock:
            if not self._started:
                raise RuntimeError(
                    "MujocoHumanReplayHandoffComposition.start() is required."
                )
            if (
                self._active_source_slot_id
                != self.replay_observer_binding.source_slot_id
            ):
                raise RuntimeError("Replay publication is not the active CS6a authority.")
            self.replay_observer_binding.validate_emission(emission)
            submission = self.runtime.submit_replay_source_frame_emission(
                self.replay_source,
                emission,
            )
            if submission.source_emission is not emission:
                raise RuntimeError(
                    "Replay source-frame identity was rebuilt unexpectedly."
                )
            if self.pose_fusion_shadow_spec is not None:
                try:
                    draft = submission.adapter_result.proposal_draft
                    if draft is None:
                        self._pose_fusion_shadow_error_by_role[
                            CS6B1G_REPLAY_POSE_INPUT_ROLE_V1
                        ] = "replay_adapter_produced_no_action"
                    else:
                        replay_source_facts = {
                            fact.fact_id: fact.semantic_content_hash
                            for fact in self.runtime.trace_facts
                        }
                        self._submit_pose_fusion_shadow_input(
                            role=CS6B1G_REPLAY_POSE_INPUT_ROLE_V1,
                            producer_id=draft.producer_id,
                            producer_epoch=draft.producer_epoch,
                            sequence=draft.sequence,
                            # The generic composition input remains on its
                            # existing RobotAction-v1 compatibility contract.
                            # Proposal adaptation is ComponentAction-native;
                            # do not feed that production carrier back through
                            # the legacy composition shadow.
                            action=emission.action,
                            produced_at=draft.produced_at,
                            produced_clock_domain_id=(
                                draft.produced_clock_domain_id
                            ),
                            effective_from=draft.effective_from,
                            effective_until=draft.effective_until,
                            observation_cut_id=draft.observation_cut_id,
                            source_fact_ids=(
                                emission.emission_id,
                                submission.adapter_result.source_adapter_receipt
                                .representation_trace_id,
                            ),
                            source_fact_hashes=(
                                replay_source_facts[emission.emission_id],
                                replay_source_facts[
                                    submission.adapter_result
                                    .source_adapter_receipt
                                    .representation_trace_id
                                ],
                            ),
                            parent_fact_id=emission.emission_id,
                            parent_fact_hash=emission.semantic_content_hash,
                        )
                except Exception as error:
                    self._pose_fusion_shadow_error_by_role[
                        CS6B1G_REPLAY_POSE_INPUT_ROLE_V1
                    ] = f"{type(error).__name__}:{error}"
            return submission

    @staticmethod
    def _pose_execution_payload(action) -> dict:
        """Compare executable pose content while ignoring producer/source labels."""

        if isinstance(action, ComponentAction):
            action = materialize_robot_action_v1(action)
        return {
            "action_space": action.action_space,
            "arm_pose_by_side": action.arm_pose_by_side,
            "hand_qpos_by_side": action.hand_qpos_by_side,
            "end_effector_targets_by_side": {
                side: {
                    "value_type": target.value_type,
                    "values": target.values,
                    "adapter": target.adapter,
                    "names": target.names,
                    "metadata": target.metadata,
                }
                for side, target in action.end_effector_targets_by_side.items()
            },
            "active_effectors": action.active_effectors,
            "frame_by_effector": action.frame_by_effector,
            "controlled_frame_by_effector": action.controlled_frame_by_effector,
            "execution_metadata": action.metadata,
        }

    def _evaluate_pose_mixing_shadow(
        self,
        control_step,
        *,
        contexts: tuple[PoseMixingContext, ...],
    ) -> None:
        strategy = self.pose_mixing_strategy
        snapshot = control_step.snapshot
        if strategy is None or snapshot is None:
            return
        authoritative = control_step.computation
        authoritative_ids = (
            ()
            if authoritative is None
            else tuple(dict(authoritative.selected_by_group).values())
        )
        decision: PoseMixingDecision | None = None
        selection_parity = False
        output_pose_parity = False
        status = "strategy_error"
        detail = ""
        try:
            mixing_input = build_pose_mixing_input_snapshot(
                snapshot,
                source_slot_by_producer={
                    self.runtime.config.producer_id: self.human_source_spec.source_id,
                    self.runtime.config.replay_producer_id: (
                        self.replay_observer_binding.source_slot_id
                    ),
                },
                clock_domain_id=self.runtime.config.clock_domain_id,
                contexts=contexts,
            )
            decision = strategy.decide(mixing_input)
            selection_parity = (
                tuple(decision.selected_proposal_ids) == authoritative_ids
            )
            if (
                decision.output_action is None
                and (
                    authoritative is None
                    or authoritative.composed_candidate is None
                )
            ):
                output_pose_parity = True
            elif (
                decision.output_action is not None
                and authoritative is not None
                and authoritative.composed_candidate is not None
            ):
                shadow_hash = canonical_content_hash(
                    self._pose_execution_payload(decision.output_action),
                    domain="dexteleop.cs6b1d.pose_execution_payload.v1",
                )
                authoritative_hash = canonical_content_hash(
                    self._pose_execution_payload(authoritative.composed_candidate),
                    domain="dexteleop.cs6b1d.pose_execution_payload.v1",
                )
                output_pose_parity = shadow_hash == authoritative_hash
            status = (
                "parity"
                if selection_parity and output_pose_parity
                else "shadow_difference"
            )
            detail = decision.reason
        except Exception as error:  # Shadow must never alter authoritative dispatch.
            detail = f"{type(error).__name__}:{error}"
        payload = {
            "strategy_id": strategy.spec.strategy_id,
            "strategy_spec_hash": strategy.spec.semantic_content_hash,
            "mixing_decision_id": (
                None if decision is None else decision.decision_id
            ),
            "mixing_decision_hash": (
                None if decision is None else decision.semantic_content_hash
            ),
            "authoritative_result_id": (
                None if authoritative is None else authoritative.result_id
            ),
            "authoritative_selected_proposal_ids": authoritative_ids,
            "strategy_selected_proposal_ids": (
                () if decision is None else decision.selected_proposal_ids
            ),
            "selection_parity": selection_parity,
            "output_pose_parity": output_pose_parity,
            "status": status,
            "detail": detail,
        }
        self._last_pose_mixing_decision = decision
        self._last_pose_mixing_shadow_evaluation = PoseMixingShadowEvaluation(
            **payload,
            semantic_content_hash=canonical_content_hash(
                payload,
                domain=CS6B1D_POSE_MIXING_SHADOW_HASH_DOMAIN_V1,
            ),
        )

    def _evaluate_pose_fusion_composition_shadow(self, control_step) -> None:
        spec = self.pose_fusion_shadow_spec
        snapshot = control_step.snapshot
        if spec is None or snapshot is None:
            return
        proposal_snapshot = snapshot.proposal_snapshot
        descriptor = self.runtime.authoritative_port.clock_domain
        cut = ControlCut(
            runtime_session_id=snapshot.runtime_session_id,
            control_domain_id=snapshot.control_domain_id,
            clock_domain_id=descriptor.clock_domain_id,
            clock_reset_epoch=descriptor.reset_epoch,
            cutoff_at=proposal_snapshot.cutoff_at,
            cutoff_admission_sequence=(
                proposal_snapshot.cutoff_admission_sequence
            ),
        )
        try:
            submission = self.runtime.composition_port.evaluate_composition_shadow(
                spec=spec,
                target_effective_at=(
                    snapshot.decision_schedule_tick.target_effective_at
                ),
                control_cut=cut,
            )
            if submission.proposal_admission is not None:
                raise RuntimeError(
                    "Pose fusion shadow unexpectedly admitted a proposal."
                )
            self._last_pose_fusion_shadow_submission = submission
            self._last_pose_fusion_shadow_error = None
        except Exception as error:  # Shadow must never alter authoritative dispatch.
            self._last_pose_fusion_shadow_error = (
                f"{type(error).__name__}:{error}"
            )

    def step(
        self,
        *,
        now: float | None = None,
        physics_ticks_after_dispatch: int = 0,
        pose_mixing_contexts: tuple[PoseMixingContext, ...] = (),
    ):
        with self._lock:
            if not self._started:
                raise RuntimeError(
                    "MujocoHumanReplayHandoffComposition.start() is required."
                )
            result = self.runtime.step(
                now=now,
                physics_ticks_after_dispatch=physics_ticks_after_dispatch,
            )
            self._evaluate_pose_mixing_shadow(
                result,
                contexts=tuple(pose_mixing_contexts),
            )
            self._evaluate_pose_fusion_composition_shadow(result)
            return result

    def close(self) -> None:
        with self._lock:
            if not self._started:
                return
            self._active_source_slot_id = None
            self.runtime.close()
            self._started = False


class MujocoHumanReplayStationTargetAdapter(StationAuthoritativeTargetAdapterBase):
    """Station lifecycle adapter over the existing mixed MuJoCo core."""

    def __init__(
        self,
        *,
        composition: MujocoHumanReplayHandoffComposition,
        clock: Callable[[], float],
        clock_domain_id: str,
    ) -> None:
        super().__init__(
            adapter_id="cs6b1-mujoco-human-replay-target-adapter",
            subject_id=composition.manifest.execution_binding.binding_id,
            supported_operations=(
                "connect_target",
                "disconnect_target",
                "run_preflight",
            ),
            composition=composition,
            clock=clock,
            clock_domain_id=clock_domain_id,
            expected_control_mode="hybrid",
        )

    def perform(self, intent: StationIntent) -> StationOperationResultDraft:
        self._validate_intent(intent)
        cached = self._result_by_intent_id.get(intent.intent_id)
        if cached is not None:
            return cached
        if intent.intent_kind == "connect_target":
            self.composition.start()
            status = "mujoco_human_replay_target_started"
        elif intent.intent_kind == "run_preflight":
            self._validate_live_binding(require_started=True)
            status = "mujoco_human_replay_target_preflighted"
        else:
            self.composition.close()
            status = "mujoco_human_replay_target_closed"
        return self._build_result(
            intent,
            publication_enabled=False,
            evidence_status=status,
        )


class MujocoHumanReplayStationSourceAdapter(StationAuthoritativeTargetAdapterBase):
    """One source-slot lifecycle adapter.

    The default synthetic/replay path owns no producer.  An exact live-pose
    composition may inject one observe-only TeleopSession source for the human
    slot; it still has no publication authority or target dispatcher.
    """

    def __init__(
        self,
        *,
        composition: MujocoHumanReplayHandoffComposition,
        source_slot_id: str,
        source_kind: str,
        clock: Callable[[], float],
        clock_domain_id: str,
        passive_pose_source: PassiveTeleopPoseSessionSource | None = None,
    ) -> None:
        normalized_kind = str(source_kind).strip()
        if normalized_kind not in {"human", "replay"}:
            raise ValueError("Mixed Station source_kind must be human or replay.")
        if passive_pose_source is not None and normalized_kind != "human":
            raise ValueError("Passive teleop pose source may bind only the human slot.")
        if (
            passive_pose_source is not None
            and passive_pose_source.spec.source_id != source_slot_id
        ):
            raise ValueError("Passive teleop pose source slot mismatches Station.")
        operations = [
            "connect_source",
            "disconnect_source",
            "run_preflight",
            "request_candidate",
            "release_candidate",
            "release_motion_authority",
        ]
        if normalized_kind == "replay":
            operations.append("request_motion_authority")
        else:
            operations.append("request_human_takeover")
        super().__init__(
            adapter_id=f"cs6b1-{normalized_kind}-source-adapter",
            subject_id=source_slot_id,
            supported_operations=tuple(operations),
            composition=composition,
            clock=clock,
            clock_domain_id=clock_domain_id,
            expected_control_mode="hybrid",
        )
        self.source_kind = normalized_kind
        self.passive_pose_source = passive_pose_source

    def perform(self, intent: StationIntent) -> StationOperationResultDraft:
        self._validate_intent(intent)
        cached = self._result_by_intent_id.get(intent.intent_id)
        if cached is not None:
            return cached
        kind = intent.intent_kind
        lower_event_ids: tuple[str, ...] = ()
        if kind == "connect_source":
            if self.passive_pose_source is not None:
                self.passive_pose_source.start()
            status = f"{self.source_kind}_source_connected"
        elif kind == "disconnect_source":
            if self.passive_pose_source is not None:
                self.passive_pose_source.close()
            status = f"{self.source_kind}_source_disconnected"
        elif kind == "run_preflight":
            self._validate_live_binding(require_started=True)
            if (
                self.passive_pose_source is not None
                and self.passive_pose_source.session_state
                != TeleopRunState.DEBUG_ONLY
            ):
                raise RuntimeError(
                    "Passive teleop pose source requires completed calibration "
                    "and DEBUG_ONLY state before Station preflight."
                )
            status = f"{self.source_kind}_source_preflighted"
        elif kind == "request_candidate":
            self._validate_live_binding(require_started=True)
            status = f"{self.source_kind}_candidate_bound"
        elif kind == "release_candidate":
            status = f"{self.source_kind}_candidate_released"
        elif kind == "request_motion_authority":
            if self.source_kind != "replay":
                raise ValueError("Initial mixed authority is replay-only in CS6b1.")
            self._validate_live_binding(require_started=True)
            handoff = self.composition.handoff_to(
                self.subject_id,
                reason=f"station_initial_replay_authority:{intent.reason}",
            )
            lower_event_ids = tuple(
                event_id
                for event_id in (
                    handoff.hold_gate_event_id,
                    handoff.released_authority_event_id,
                    handoff.takeover_authority_event_id,
                    handoff.ready_gate_event_id,
                )
                if event_id is not None
            )
            status = "initial_replay_authority_ready"
        elif kind == "request_human_takeover":
            raise RuntimeError(
                "Human takeover must use the Station pending-finalization seam."
            )
        else:
            outcome = self.composition.runtime.transition_execution_gate(
                "hold",
                reason=f"station_release_{self.source_kind}_authority",
                source_event_ids=(intent.intent_id,),
            )
            lower_event_ids = outcome.lower_event_ids
            status = f"{self.source_kind}_publication_closed_and_execution_held"
        return self._build_result(
            intent,
            publication_enabled=False,
            lower_event_ids=lower_event_ids,
            evidence_status=status,
        )


class ScopedMujocoHumanReplayStationComposition:
    """CS6b1 Station control plane over the existing CS6b0 data plane.

    The wrapper owns two application publication barriers and one-shot
    operation-spec bindings.  It does not own an action queue, replay cursor,
    human device, Arbiter, epoch store, dispatcher, or episode lifecycle.
    """

    def __init__(
        self,
        *,
        station: ControlStationRuntime,
        data_plane: MujocoHumanReplayHandoffComposition,
        clock: Callable[[], float],
        clock_domain_id: str,
        passive_human_pose_source: PassiveTeleopPoseSessionSource | None = None,
    ) -> None:
        if not isinstance(station, ControlStationRuntime):
            raise TypeError("station must be a ControlStationRuntime.")
        if not isinstance(data_plane, MujocoHumanReplayHandoffComposition):
            raise TypeError("data_plane must be a mixed MuJoCo composition.")
        if station.manifest.semantic_content_hash != data_plane.manifest.semantic_content_hash:
            raise ValueError("Station and mixed data plane must share one manifest.")
        if (
            station.state.runtime_session_id
            != data_plane.runtime.config.runtime_session_id
            or station.state.control_domain_id
            != data_plane.runtime.config.control_domain_id
        ):
            raise ValueError("Station and mixed data-plane runtime scope differs.")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        self._station = station
        self._data_plane = data_plane
        self.passive_human_pose_source = passive_human_pose_source
        self.clock = clock
        self.clock_domain_id = str(clock_domain_id).strip()
        if not self.clock_domain_id:
            raise ValueError("clock_domain_id must be non-empty.")
        if passive_human_pose_source is not None:
            human_slot = station.manifest.source_slot(
                data_plane.human_source_spec.source_id
            )
            if (
                passive_human_pose_source.spec.source_id
                != human_slot.source_slot_id
                or passive_human_pose_source.spec.semantic_content_hash
                not in human_slot.required_binding_refs
            ):
                raise ValueError(
                    "Passive human pose source does not match the Station manifest."
                )
        self._lock = threading.RLock()
        self._takeover_binding_by_spec_id: dict[
            str,
            tuple[HumanTakeoverOperationSpec, HumanTakeoverReadinessProvider],
        ] = {}
        self._consumed_operation_spec_ids: set[str] = set()
        self._takeover_result_by_intent_id: dict[str, StationOperationResultDraft] = {}
        self._takeover_evidence_by_intent_id: dict[
            str, HumanTakeoverOperationEvidence
        ] = {}
        self._takeover_failure_reason_by_intent_id: dict[str, str] = {}
        self._ready_candidate_by_spec_id: dict[
            str, HumanTakeoverReadinessCandidate
        ] = {}

        self.target_adapter: StationOperationAdapter = (
            MujocoHumanReplayStationTargetAdapter(
                composition=data_plane,
                clock=clock,
                clock_domain_id=self.clock_domain_id,
            )
        )
        self.replay_source_adapter = MujocoHumanReplayStationSourceAdapter(
            composition=data_plane,
            source_slot_id=data_plane.replay_observer_binding.source_slot_id,
            source_kind="replay",
            clock=clock,
            clock_domain_id=self.clock_domain_id,
        )
        self.human_source_adapter = MujocoHumanReplayStationSourceAdapter(
            composition=data_plane,
            source_slot_id=data_plane.human_source_spec.source_id,
            source_kind="human",
            clock=clock,
            clock_domain_id=self.clock_domain_id,
            passive_pose_source=passive_human_pose_source,
        )
        self.publication_barrier_by_source = {
            self.replay_source_adapter.subject_id: StationPublicationBarrier(
                source_slot_id=self.replay_source_adapter.subject_id
            ),
            self.human_source_adapter.subject_id: StationPublicationBarrier(
                source_slot_id=self.human_source_adapter.subject_id
            ),
        }
        self._barrier_observer_port = StationPublicationBarrierObserverPort(
            suppress_callback_errors=True
        )

    @property
    def station(self) -> ControlStationRuntime:
        return self._station

    @property
    def data_plane(self) -> MujocoHumanReplayHandoffComposition:
        return self._data_plane

    @property
    def adapters(self) -> tuple[StationOperationAdapter, ...]:
        return (
            self.target_adapter,
            self.human_source_adapter,
            self.replay_source_adapter,
        )

    @property
    def barrier_observer_failure_count(self) -> int:
        return self._barrier_observer_port.failure_count

    def bind_publication_barrier_observer(
        self,
        observer: Callable[..., object],
    ) -> None:
        """Bind the optional formal-capture projection before Station use."""

        if self._station.state.state_revision != 0:
            raise RuntimeError("Publication barrier observer must bind before use.")
        self._barrier_observer_port.bind(observer)

    def _observe_barrier_transition(
        self,
        snapshot: StationPublicationBarrierSnapshot,
        *,
        causal_station_fact_id: str,
        occurred_at: float,
        clock_domain_id: str,
    ) -> None:
        self._barrier_observer_port.observe(
            snapshot,
            causal_station_fact_id=causal_station_fact_id,
            occurred_at=occurred_at,
            clock_domain_id=clock_domain_id,
        )

    @property
    def takeover_evidence_by_intent_id(
        self,
    ) -> dict[str, HumanTakeoverOperationEvidence]:
        with self._lock:
            return dict(self._takeover_evidence_by_intent_id)

    @property
    def takeover_failure_reason_by_intent_id(self) -> dict[str, str]:
        with self._lock:
            return dict(self._takeover_failure_reason_by_intent_id)

    def register_human_takeover_operation(
        self,
        spec: HumanTakeoverOperationSpec,
        provider: HumanTakeoverReadinessProvider,
    ) -> str:
        """Bind a one-shot spec to a lower-owned readiness provider."""

        if not isinstance(spec, HumanTakeoverOperationSpec):
            raise TypeError("spec must be a HumanTakeoverOperationSpec.")
        observer_binding_id = str(getattr(provider, "observer_binding_id", "")).strip()
        if not observer_binding_id or not callable(getattr(provider, "acquire", None)):
            raise TypeError("provider must implement HumanTakeoverReadinessProvider.")
        manifest = self._station.manifest
        previous = manifest.source_slot(spec.previous_source_slot_id)
        target = manifest.source_slot(spec.target_human_source_slot_id)
        if (
            previous.source_kind == "human"
            or target.source_kind != "human"
            or spec.previous_source_action_contract_hash
            != previous.source_action_contract_hash
            or spec.target_human_action_contract_hash
            != target.source_action_contract_hash
            or spec.execution_binding_hash
            != manifest.execution_binding.semantic_content_hash
            or spec.readiness_observer_binding_id != observer_binding_id
        ):
            raise ValueError("Human takeover operation spec mismatches the manifest.")
        spec_id = spec.operation_spec_id
        with self._lock:
            if spec_id in self._takeover_binding_by_spec_id:
                raise RuntimeError("Human takeover operation spec is already registered.")
            self._takeover_binding_by_spec_id[spec_id] = (spec, provider)
        return spec_id

    def prepare_pending_operation(
        self,
        transition: ControlStationTransition,
    ) -> None:
        if transition.receipt.status != "operation_requested":
            return
        kind = transition.intent.intent_kind
        if kind == "request_human_takeover":
            barriers = tuple(self.publication_barrier_by_source.values())
        elif kind in {
            "release_motion_authority",
            "release_candidate",
            "disconnect_source",
        }:
            barrier = self.publication_barrier_by_source.get(
                str(transition.intent.subject_id)
            )
            barriers = () if barrier is None else (barrier,)
        elif kind == "disconnect_target":
            barriers = tuple(self.publication_barrier_by_source.values())
        else:
            barriers = ()
        for barrier in barriers:
            snapshot = barrier.disable(reason=f"station_operation_admitted:{kind}")
            self._observe_barrier_transition(
                snapshot,
                causal_station_fact_id=transition.intent.intent_id,
                occurred_at=transition.intent.available_at,
                clock_domain_id=transition.intent.available_clock_domain_id,
            )

    def complete_pending_operation(
        self,
        transition: ControlStationTransition,
    ) -> StationOperationExecution:
        self.prepare_pending_operation(transition)
        subject = transition.intent.subject_id
        kind = transition.intent.intent_kind
        if kind == "request_human_takeover":
            completion = self._execute_human_takeover(transition)
            execution = StationOperationExecution(
                request_transition=transition,
                completion_transition=completion,
            )
        else:
            if subject == self._station.manifest.execution_binding.binding_id:
                adapter = self.target_adapter
            elif subject == self.replay_source_adapter.subject_id:
                adapter = self.replay_source_adapter
            elif subject == self.human_source_adapter.subject_id:
                adapter = self.human_source_adapter
            else:
                raise ValueError("Pending operation has no mixed Station adapter.")
            execution = self._station.execute_pending_operation(transition, adapter)
            completion = execution.completion_transition
        if kind == "request_motion_authority":
            barrier = self.publication_barrier_by_source[str(subject)]
            if completion.accepted and completion.result.succeeded:
                snapshot = barrier.enable_from_authority_result(completion.result)
            else:
                snapshot = barrier.disable(reason="authority_completion_failed")
            self._observe_barrier_transition(
                snapshot,
                causal_station_fact_id=transition.intent.intent_id,
                occurred_at=completion.result.completed_at,
                clock_domain_id=completion.result.completed_clock_domain_id,
            )
        elif kind == "request_human_takeover":
            human_barrier = self.publication_barrier_by_source[
                self.human_source_adapter.subject_id
            ]
            if completion.accepted and completion.result.succeeded:
                snapshot = human_barrier.enable_from_human_takeover_result(
                    completion.result
                )
                self._observe_barrier_transition(
                    snapshot,
                    causal_station_fact_id=transition.intent.intent_id,
                    occurred_at=completion.result.completed_at,
                    clock_domain_id=completion.result.completed_clock_domain_id,
                )
            else:
                for barrier in self.publication_barrier_by_source.values():
                    snapshot = barrier.disable(
                        reason="human_takeover_completion_failed"
                    )
                    self._observe_barrier_transition(
                        snapshot,
                        causal_station_fact_id=transition.intent.intent_id,
                        occurred_at=completion.result.completed_at,
                        clock_domain_id=completion.result.completed_clock_domain_id,
                    )
        return execution

    def _execute_human_takeover(
        self,
        transition: ControlStationTransition,
    ) -> ControlStationOperationTransition:
        intent = transition.intent
        with self._lock:
            cached = self._takeover_result_by_intent_id.get(intent.intent_id)
            if cached is not None:
                return self._finalize_takeover_draft(transition, cached)
            binding = self._takeover_binding_by_spec_id.get(
                str(intent.operation_spec_id)
            )
            if binding is None:
                return self._finalize_takeover_draft(
                    transition,
                    self._fail_unbound_takeover(
                        intent,
                        reason="unknown_human_takeover_operation_spec",
                    ),
                )
            spec, provider = binding
            if (
                intent.subject_id != spec.target_human_source_slot_id
                or intent.resource_binding_id != spec.previous_source_slot_id
            ):
                return self._finalize_takeover_draft(
                    transition,
                    self._fail_unbound_takeover(
                        intent,
                        reason="human_takeover_intent_spec_binding_mismatch",
                    ),
                )
            if spec.operation_spec_id in self._consumed_operation_spec_ids:
                return self._finalize_takeover_draft(
                    transition,
                    self._fail_unbound_takeover(
                        intent,
                        reason="human_takeover_operation_spec_already_consumed",
                    ),
                )
            self._consumed_operation_spec_ids.add(spec.operation_spec_id)

            state: HumanTakeoverState | None = None
            candidate: HumanTakeoverReadinessCandidate | None = None
            try:
                state = self._data_plane.begin_safe_replay_to_human_takeover(
                    spec.boundary_policy,
                    reason=intent.reason,
                    capture_boundary_id=spec.capture_boundary_id,
                )
                if state.phase == "awaiting_human_readiness":
                    wait_started_at = float(self.clock())
                    candidate = provider.acquire(
                        operation_spec=spec,
                        takeover_state=state,
                        timeout_sec=spec.readiness_timeout_sec,
                    )
                    wait_completed_at = float(self.clock())
                    if not isinstance(candidate, HumanTakeoverReadinessCandidate):
                        raise TypeError("Readiness provider returned an invalid candidate.")
                    if candidate.observer_binding_id != spec.readiness_observer_binding_id:
                        raise ValueError("Readiness candidate observer binding mismatches.")
                    if (
                        wait_completed_at < wait_started_at
                        or wait_completed_at - wait_started_at
                        > spec.readiness_timeout_sec
                    ):
                        raise TimeoutError("Human takeover readiness timed out.")
            except Exception as error:
                self._takeover_failure_reason_by_intent_id[intent.intent_id] = (
                    f"{type(error).__name__}:{error}"
                )
                current = self._data_plane.safe_human_takeover_state
                if current is not None and current.phase not in {"ready", "failed"}:
                    state = self._data_plane.fail_safe_human_takeover(
                        reason=f"readiness_{type(error).__name__}",
                        fault=not isinstance(error, (TimeoutError, ValueError)),
                    )
                elif current is not None:
                    state = current
                else:
                    return self._finalize_takeover_draft(
                        transition,
                        self._fail_unbound_takeover(
                            intent,
                            reason=(
                                "human_takeover_executor_"
                                f"{type(error).__name__}"
                            ),
                        ),
                    )

            assert state is not None
            if state.phase != "awaiting_human_readiness":
                return self._finalize_takeover_draft(
                    transition,
                    self._build_takeover_result_draft(
                        intent=intent,
                        spec=spec,
                        state=state,
                        candidate=candidate,
                    ),
                )
            assert candidate is not None

            def finalize_ready() -> StationOperationResultDraft:
                final_state = self._data_plane.complete_human_takeover_readiness(
                    candidate.emission,
                    effective_from=candidate.effective_from,
                    effective_until=candidate.effective_until,
                    target_effective_at=candidate.target_effective_at,
                    deadman_rearm_event_id=candidate.deadman_rearm_event_id,
                    deadman_rearmed_at=candidate.deadman_rearmed_at,
                    retargeter_rebase_receipt_id=(
                        candidate.retargeter_rebase_receipt_id
                    ),
                    retargeter_rebased_at=candidate.retargeter_rebased_at,
                )
                return self._build_takeover_result_draft(
                    intent=intent,
                    spec=spec,
                    state=final_state,
                    candidate=candidate,
                )

            try:
                return self._station.finalize_pending_operation(
                    transition,
                    finalize_ready,
                )
            except StationPendingOperationSuperseded:
                current = self._data_plane.safe_human_takeover_state
                if current is not None and current.phase not in {"ready", "failed"}:
                    station_faulted = self._station.state.lifecycle_state == "fault"
                    current = self._data_plane.fail_safe_human_takeover(
                        reason="station_pending_operation_superseded",
                        fault=station_faulted,
                    )
                assert current is not None
                late_draft = self._build_takeover_result_draft(
                    intent=intent,
                    spec=spec,
                    state=current,
                    candidate=candidate,
                )
                return self._station.admit_operation_result(late_draft)
            except Exception as error:
                self._takeover_failure_reason_by_intent_id[intent.intent_id] = (
                    f"{type(error).__name__}:{error}"
                )
                current = self._data_plane.safe_human_takeover_state
                if current is not None and current.phase not in {"ready", "failed"}:
                    current = self._data_plane.fail_safe_human_takeover(
                        reason=f"readiness_{type(error).__name__}",
                        fault=not isinstance(error, (TimeoutError, ValueError)),
                    )
                    return self._finalize_takeover_draft(
                        transition,
                        self._build_takeover_result_draft(
                            intent=intent,
                            spec=spec,
                            state=current,
                            candidate=candidate,
                        ),
                    )
                return self._finalize_takeover_draft(
                    transition,
                    self._fail_unbound_takeover(
                        intent,
                        reason=f"human_takeover_finalizer_{type(error).__name__}",
                    ),
                )

    def _finalize_takeover_draft(
        self,
        transition: ControlStationTransition,
        draft: StationOperationResultDraft,
    ) -> ControlStationOperationTransition:
        try:
            return self._station.finalize_pending_operation(
                transition,
                lambda: draft,
            )
        except StationPendingOperationSuperseded:
            return self._station.admit_operation_result(draft)

    def _build_takeover_result_draft(
        self,
        *,
        intent: StationIntent,
        spec: HumanTakeoverOperationSpec,
        state: HumanTakeoverState,
        candidate: HumanTakeoverReadinessCandidate | None,
    ) -> StationOperationResultDraft:
        evidence = self._build_takeover_evidence(
            intent=intent,
            spec=spec,
            state=state,
            candidate=candidate,
        )
        self._takeover_evidence_by_intent_id[intent.intent_id] = evidence
        succeeded = state.phase == "ready"
        if succeeded and candidate is not None:
            self._ready_candidate_by_spec_id[spec.operation_spec_id] = candidate
        status = (
            HUMAN_TAKEOVER_OPERATION_READY_STATUS_V1
            if succeeded
            else (
                HUMAN_TAKEOVER_OPERATION_FAULTED_STATUS_V1
                if state.execution_gate_status == "fault"
                else HUMAN_TAKEOVER_OPERATION_HELD_STATUS_V1
            )
        )
        result = StationOperationResultDraft(
            source_intent_id=intent.intent_id,
            operation_kind=intent.intent_kind,
            subject_id=intent.subject_id,
            succeeded=succeeded,
            completed_at=float(self.clock()),
            completed_clock_domain_id=self.clock_domain_id,
            status=status,
            evidence_refs=(
                evidence.evidence_id,
                evidence.semantic_content_hash,
                *evidence.evidence_refs,
            ),
        )
        self._takeover_result_by_intent_id[intent.intent_id] = result
        return result

    def _fail_unbound_takeover(
        self,
        intent: StationIntent,
        *,
        reason: str,
    ) -> StationOperationResultDraft:
        self._takeover_failure_reason_by_intent_id[intent.intent_id] = reason
        outcome = self._data_plane.runtime.transition_execution_gate(
            "fault",
            reason=reason,
            source_event_ids=(intent.intent_id,),
        )
        result = StationOperationResultDraft(
            source_intent_id=intent.intent_id,
            operation_kind=intent.intent_kind,
            subject_id=intent.subject_id,
            succeeded=False,
            completed_at=float(self.clock()),
            completed_clock_domain_id=self.clock_domain_id,
            status=HUMAN_TAKEOVER_OPERATION_FAULTED_STATUS_V1,
            evidence_refs=outcome.lower_event_ids,
        )
        self._takeover_result_by_intent_id[intent.intent_id] = result
        return result

    def _build_takeover_evidence(
        self,
        *,
        intent: StationIntent,
        spec: HumanTakeoverOperationSpec,
        state: HumanTakeoverState,
        candidate: HumanTakeoverReadinessCandidate | None,
    ) -> HumanTakeoverOperationEvidence:
        receipt_refs = tuple(
            receipt.semantic_content_hash
            for receipt in (
                state.hold_receipt,
                state.invalidation_receipt,
                state.release_receipt,
                state.takeover_receipt,
                state.readiness_receipt,
                state.ready_receipt,
            )
            if receipt is not None
        )
        refs = (
            state.request.semantic_content_hash,
            *receipt_refs,
            *(
                ()
                if candidate is None
                else (candidate.semantic_content_hash,)
            ),
        )
        return HumanTakeoverOperationEvidence(
            operation_spec_id=spec.operation_spec_id,
            source_intent_id=intent.intent_id,
            request_id=state.request.request_id,
            request_semantic_hash=state.request.semantic_content_hash,
            final_state_semantic_hash=state.semantic_content_hash,
            previous_source_slot_id=spec.previous_source_slot_id,
            target_human_source_slot_id=spec.target_human_source_slot_id,
            final_phase=state.phase,
            execution_gate_status=state.execution_gate_status,
            succeeded=state.phase == "ready",
            completed_at=float(self.clock()),
            clock_domain_id=self.clock_domain_id,
            evidence_refs=refs,
        )

    def promote_human_takeover_emission(
        self,
        operation_spec_id: str,
    ) -> MujocoHumanReplayHumanSubmission:
        spec_id = str(operation_spec_id).strip()
        with self._lock:
            candidate = self._ready_candidate_by_spec_id.get(spec_id)
            if candidate is None:
                raise RuntimeError("No READY human takeover emission is pending.")
            state = self._station.state
            human_slot = self.human_source_adapter.subject_id
            if (
                state.lifecycle_state != "running"
                or state.source_state_by_slot[human_slot] != "active"
            ):
                raise RuntimeError("Station human authority is not active.")
            barrier = self.publication_barrier_by_source[human_slot]

            def submit() -> MujocoHumanReplayHumanSubmission:
                return self._data_plane.submit_human_emission(
                    candidate.emission,
                    effective_from=candidate.effective_from,
                    effective_until=candidate.effective_until,
                    target_effective_at=candidate.target_effective_at,
                )

            submission = barrier.execute_if_enabled(submit)
            if submission.accepted:
                self._ready_candidate_by_spec_id.pop(spec_id, None)
            return submission

    def submit_replay_source_frame_emission(
        self,
        emission: ReplaySourceFrameEmission,
    ) -> MujocoReplaySubmission:
        replay_slot = self.replay_source_adapter.subject_id
        state = self._station.state
        if (
            state.lifecycle_state != "running"
            or state.source_state_by_slot[replay_slot] != "active"
        ):
            raise RuntimeError("Station replay authority is not active.")
        return self.publication_barrier_by_source[replay_slot].execute_if_enabled(
            lambda: self._data_plane.submit_replay_source_frame_emission(emission)
        )

    def set_passive_human_calibration_ready(self) -> None:
        source = self.passive_human_pose_source
        if source is None:
            raise RuntimeError("No passive human pose source is bound.")
        human_slot = self.human_source_adapter.subject_id
        if self._station.state.source_state_by_slot[human_slot] != "connected":
            raise RuntimeError(
                "Passive human calibration may be completed only after source connect."
            )
        source.set_calibration_ready()

    def step_passive_human_pose_source(
        self,
        *,
        now: float | None = None,
    ) -> PassiveTeleopPoseStep:
        """Poll a real/virtual teleop input without proposal or dispatch rights."""

        source = self.passive_human_pose_source
        if source is None:
            raise RuntimeError("No passive human pose source is bound.")
        human_slot = self.human_source_adapter.subject_id
        state = self._station.state.source_state_by_slot[human_slot]
        if state == "disconnected":
            raise RuntimeError("Passive human pose source is not connected.")
        step = source.step(now=now)
        if (
            step.emission_valid
            and step.emission is not None
            and self._data_plane.pose_fusion_shadow_spec is not None
        ):
            self._data_plane.submit_passive_human_pose_fusion_shadow_input(
                step.emission,
                observed_at=float(self.clock() if now is None else now),
            )
        return step

    def submit_passive_human_pose_step(
        self,
        step: PassiveTeleopPoseStep,
        *,
        effective_from: float,
        effective_until: float,
        target_effective_at: float,
    ) -> MujocoHumanReplayHumanSubmission:
        """Publish one already-observed pose through the existing human seam.

        Polling and publication stay separate so replay-authority observation
        remains harmless.  Callers may use this only after the typed human
        authority result has enabled the human publication barrier; no action
        is rebuilt from the passive session and no second queue is introduced.
        """

        if not isinstance(step, PassiveTeleopPoseStep):
            raise TypeError("step must be a PassiveTeleopPoseStep.")
        if not step.emission_valid or step.emission is None:
            raise ValueError(
                "A valid passive human pose emission is required for publication."
            )
        human_slot = self.human_source_adapter.subject_id
        submission = self.publication_barrier_by_source[human_slot].execute_if_enabled(
            lambda: self._data_plane.submit_human_emission(
                step.emission,
                effective_from=float(effective_from),
                effective_until=float(effective_until),
                target_effective_at=float(target_effective_at),
            )
        )
        if not isinstance(submission, MujocoHumanReplayHumanSubmission):
            raise RuntimeError("Human publication returned an invalid submission.")
        return submission

    def step(
        self,
        *,
        now: float | None = None,
        physics_ticks_after_dispatch: int = 0,
        pose_mixing_contexts: tuple[PoseMixingContext, ...] = (),
    ):
        return self._data_plane.step(
            now=now,
            physics_ticks_after_dispatch=physics_ticks_after_dispatch,
            pose_mixing_contexts=pose_mixing_contexts,
        )

    def close(self) -> None:
        for barrier in self.publication_barrier_by_source.values():
            barrier.disable(reason="composition_closed")
        if self.passive_human_pose_source is not None:
            self.passive_human_pose_source.close()
        self._data_plane.close()


def build_scoped_cs6a_mujoco_human_replay_handoff(
    runtime: MujocoDeployRuntime,
    human_source_spec: SyntheticHumanQposSourceSpec,
    replay_source: ReplayProposalSource,
    replay_observer_binding: ReplaySourceObserverBinding,
) -> MujocoHumanReplayHandoffComposition:
    """Build the scoped sim-only handoff composition; never start it."""

    human_adapter = build_mujoco_human_proposal_adapter(runtime)
    manifest = build_cs6a_human_replay_handoff_manifest(
        runtime,
        human_adapter,
        human_source_spec,
        replay_source,
        replay_observer_binding,
    )
    return MujocoHumanReplayHandoffComposition(
        manifest=manifest,
        runtime=runtime,
        human_adapter=human_adapter,
        human_source_spec=human_source_spec,
        replay_source=replay_source,
        replay_observer_binding=replay_observer_binding,
    )


def build_scoped_cs6b1_mujoco_human_replay_station(
    runtime: MujocoDeployRuntime,
    human_source_spec: SyntheticHumanQposSourceSpec,
    replay_source: ReplayProposalSource,
    replay_observer_binding: ReplaySourceObserverBinding,
    *,
    station_clock: Callable[[], float] | None = None,
    station_clock_domain_id: str = "control-station-mujoco-monotonic",
) -> ScopedMujocoHumanReplayStationComposition:
    """Add the Station control plane without replacing the CS6b0 data plane."""

    data_plane = build_scoped_cs6a_mujoco_human_replay_handoff(
        runtime,
        human_source_spec,
        replay_source,
        replay_observer_binding,
    )
    clock = runtime.clock if station_clock is None else station_clock
    station = ControlStationRuntime(
        manifest=data_plane.manifest,
        runtime_session_id=runtime.config.runtime_session_id,
        clock_domain_id=station_clock_domain_id,
        clock=clock,
    )
    return ScopedMujocoHumanReplayStationComposition(
        station=station,
        data_plane=data_plane,
        clock=clock,
        clock_domain_id=station_clock_domain_id,
    )


def build_scoped_cs6b1d_mujoco_pose_human_replay_handoff(
    runtime: MujocoDeployRuntime,
    human_source_spec: HumanPoseSourceContract,
    replay_source: ReplayProposalSource,
    replay_observer_binding: ReplaySourceObserverBinding,
    *,
    pose_mixing_strategy: PoseMixingStrategy | None = None,
    pose_fusion_shadow_spec: CompositionSpec | None = None,
) -> MujocoHumanReplayHandoffComposition:
    """Build the pose-native sim data plane; never start it implicitly."""

    human_adapter = build_mujoco_pose_human_proposal_adapter(
        runtime,
        human_source_spec,
    )
    manifest = build_cs6b1d_pose_human_replay_handoff_manifest(
        runtime,
        human_adapter,
        human_source_spec,
        replay_source,
        replay_observer_binding,
    )
    return MujocoHumanReplayHandoffComposition(
        manifest=manifest,
        runtime=runtime,
        human_adapter=human_adapter,
        human_source_spec=human_source_spec,
        replay_source=replay_source,
        replay_observer_binding=replay_observer_binding,
        pose_mixing_strategy=(
            DiscreteOwnerPoseMixingStrategy()
            if pose_mixing_strategy is None
            else pose_mixing_strategy
        ),
        pose_fusion_shadow_spec=pose_fusion_shadow_spec,
    )


def build_scoped_cs6b1d_mujoco_pose_human_replay_station(
    runtime: MujocoDeployRuntime,
    human_source_spec: HumanPoseSourceContract,
    replay_source: ReplayProposalSource,
    replay_observer_binding: ReplaySourceObserverBinding,
    *,
    pose_mixing_strategy: PoseMixingStrategy | None = None,
    pose_fusion_shadow_spec: CompositionSpec | None = None,
    passive_human_pose_source: PassiveTeleopPoseSessionSource | None = None,
    station_clock: Callable[[], float] | None = None,
    station_clock_domain_id: str = "control-station-mujoco-monotonic",
) -> ScopedMujocoHumanReplayStationComposition:
    """Add the existing typed Station transaction to the pose data plane."""

    data_plane = build_scoped_cs6b1d_mujoco_pose_human_replay_handoff(
        runtime,
        human_source_spec,
        replay_source,
        replay_observer_binding,
        pose_mixing_strategy=pose_mixing_strategy,
        pose_fusion_shadow_spec=pose_fusion_shadow_spec,
    )
    clock = runtime.clock if station_clock is None else station_clock
    station = ControlStationRuntime(
        manifest=data_plane.manifest,
        runtime_session_id=runtime.config.runtime_session_id,
        clock_domain_id=station_clock_domain_id,
        clock=clock,
    )
    return ScopedMujocoHumanReplayStationComposition(
        station=station,
        data_plane=data_plane,
        clock=clock,
        clock_domain_id=station_clock_domain_id,
        passive_human_pose_source=passive_human_pose_source,
    )


def build_scoped_cs6b1f_mujoco_pose_human_replay_station(
    runtime: MujocoDeployRuntime,
    human_source_spec: HumanPoseSourceContract,
    replay_source: ReplayProposalSource,
    replay_observer_binding: ReplaySourceObserverBinding,
    *,
    pose_mixing_strategy: PoseMixingStrategy | None = None,
    pose_fusion_shadow_spec: CompositionSpec | None = None,
    passive_human_pose_source: PassiveTeleopPoseSessionSource | None = None,
    station_clock: Callable[[], float] | None = None,
    station_clock_domain_id: str = "control-station-mujoco-monotonic",
) -> ScopedMujocoHumanReplayStationComposition:
    """Build the opt-in sim HIL station; never start or bind an episode.

    The returned Station uses the existing CS-6b1d data plane, barriers and
    operation coordinator.  ``passive_human_pose_source`` is optional and is
    the only new input seam; capture remains an explicitly injected external
    ``EpisodeTraceRecorderSink``.
    """

    if passive_human_pose_source is not None and not isinstance(
        passive_human_pose_source,
        PassiveTeleopPoseSessionSource,
    ):
        raise TypeError(
            "passive_human_pose_source must be a PassiveTeleopPoseSessionSource."
        )
    if pose_fusion_shadow_spec is not None and passive_human_pose_source is None:
        raise ValueError(
            "CS6b1g pose fusion shadow requires a passive human pose source."
        )
    human_adapter = build_mujoco_pose_human_proposal_adapter(
        runtime,
        human_source_spec,
    )
    manifest = build_cs6b1f_pose_human_replay_station_manifest(
        runtime,
        human_adapter,
        human_source_spec,
        replay_source,
        replay_observer_binding,
    )
    data_plane = MujocoHumanReplayHandoffComposition(
        manifest=manifest,
        runtime=runtime,
        human_adapter=human_adapter,
        human_source_spec=human_source_spec,
        replay_source=replay_source,
        replay_observer_binding=replay_observer_binding,
        pose_mixing_strategy=(
            DiscreteOwnerPoseMixingStrategy()
            if pose_mixing_strategy is None
            else pose_mixing_strategy
        ),
        pose_fusion_shadow_spec=pose_fusion_shadow_spec,
    )
    clock = runtime.clock if station_clock is None else station_clock
    station = ControlStationRuntime(
        manifest=data_plane.manifest,
        runtime_session_id=runtime.config.runtime_session_id,
        clock_domain_id=station_clock_domain_id,
        clock=clock,
    )
    return ScopedMujocoHumanReplayStationComposition(
        station=station,
        data_plane=data_plane,
        clock=clock,
        clock_domain_id=station_clock_domain_id,
        passive_human_pose_source=passive_human_pose_source,
    )


__all__ = [
    "CS6A_HUMAN_REPLAY_EXECUTION_MANIFEST_ID_V1",
    "CS6A_HUMAN_REPLAY_HANDOFF_MANIFEST_ID_V1",
    "CS6A_HUMAN_REPLAY_TRACE_PROFILE_ID_V1",
    "CS6B1D_POSE_HUMAN_REPLAY_HANDOFF_MANIFEST_ID_V1",
    "CS6B1D_POSE_HUMAN_REPLAY_TRACE_PROFILE_ID_V1",
    "CS6B1F_POSE_HUMAN_REPLAY_EXECUTION_MANIFEST_ID_V1",
    "CS6B1F_POSE_HUMAN_REPLAY_STATION_MANIFEST_ID_V1",
    "CS6B1F_POSE_HUMAN_REPLAY_TRACE_PROFILE_ID_V1",
    "CS6B1D_POSE_MIXING_SHADOW_HASH_DOMAIN_V1",
    "HumanReplayAuthorityHandoffReceipt",
    "MujocoHumanReplayBindingReceipt",
    "MujocoHumanReplayHandoffComposition",
    "MujocoHumanReplayHumanSubmission",
    "VirtualSimHumanTakeoverReadinessProvider",
    "PoseMixingShadowEvaluation",
    "MujocoHumanReplayStationSourceAdapter",
    "MujocoHumanReplayStationTargetAdapter",
    "ScopedMujocoHumanReplayStationComposition",
    "build_cs6a_human_replay_execution_manifest",
    "build_cs6a_human_replay_handoff_manifest",
    "build_cs6b1d_pose_human_replay_handoff_manifest",
    "build_cs6b1f_pose_human_replay_execution_manifest",
    "build_cs6b1f_pose_human_replay_station_manifest",
    "build_scoped_cs6a_mujoco_human_replay_handoff",
    "build_scoped_cs6b1_mujoco_human_replay_station",
    "build_scoped_cs6b1d_mujoco_pose_human_replay_handoff",
    "build_scoped_cs6b1d_mujoco_pose_human_replay_station",
    "build_scoped_cs6b1f_mujoco_pose_human_replay_station",
]
