from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from control_composition.composition import (
    CompositionAuthorityGateSpec,
    CompositionSpec,
    DeterministicComposer,
    DeterministicCompositionStrategyV2,
    FormalCompositionStrategySpec,
)
from control_composition.activation import (
    ACTIVATION_LOCAL_INPUT_FRESHNESS_POLICY_V2,
    ACTIVATION_LOCAL_STRICT_SOURCE_FRESHNESS_POLICY_V2,
)
from control_composition.lifecycle import (
    CompositionSessionOperationSpec,
    CompositionSessionRearmSpec,
    HumanTakeoverOperationSpec,
    MixedControlBoundaryPolicy,
)
from control_composition.observability import SourceDifferenceSpec
from control_station import (
    SideEffectStationOperationCoordinator,
    StationIntentDraft,
    StationControlTraceBridge,
    project_station_run_manifest,
)
from control_station.composition import (
    CS6B1D_POSE_HUMAN_REPLAY_HANDOFF_MANIFEST_ID_V1,
    CS6B1F_POSE_HUMAN_REPLAY_STATION_MANIFEST_ID_V1,
    VirtualSimHumanTakeoverReadinessProvider,
    ReplaySourceObserverBinding,
    build_cs6b1i_pose_composition_session_execution_manifest,
    build_scoped_cs6b1i_mujoco_pose_composition_session,
    build_scoped_cs6b1h_mujoco_pose_composer_authority,
    build_mujoco_pose_human_proposal_adapter,
    build_scoped_cs6b1f_mujoco_pose_human_replay_station,
    build_cs6b1f_pose_human_replay_execution_manifest,
    build_scoped_cs6b1d_mujoco_pose_human_replay_station,
)
from control_station.sources import (
    HumanTakeoverReadinessCandidate,
    PassiveTeleopPoseSessionSource,
    PassiveTeleopPoseSourceSpec,
    SyntheticHumanPoseSource,
    SyntheticHumanPoseSourceSpec,
)
from dataset_contracts.completeness import trace_requirement_profile
from dataset_contracts.control_trace_episode import CONTROL_TRACE_SESSION_RELATIVE_ROOT_V1
from dataset_contracts.replay import (
    REPLAY_ACTION_STAGE_SELECTED,
    REPLAY_BINDING_USE_EXECUTION_CANDIDATE,
    REPLAY_BINDING_VERIFIED_COMPATIBLE,
    REPLAY_MODE_CANONICAL_ACTION,
    REPLAY_SCHEDULE_FIXED_RATE,
    REPLAY_SOURCE_INTERPRETATION_VERIFIED,
    ReplayScheduleSpec,
    ReplaySourceActionContract,
    ReplaySourceContract,
)
from robot_motion.action import (
    RobotAction,
    SCHEDULED_REFERENCE_SEGMENT_SIM_V1,
    SharedCanonicalSafetyResult,
)
from robot_motion.action.proposal import robot_action_semantic_payload
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action.authoritative_trace import project_control_execution_manifest
from robot_motion.runtime import EffectorCommand, ObservationChannel, TeleopCommand
from robot_motion.replay import (
    ReplayProposalSource,
    ReplayProposalSourceConfig,
    build_replay_binding_assessment,
    create_replay_source_frame_emission,
)
from sim.contracts import TimingSpec
from sim.mujoco.adapters import (
    MujocoPositionExecutionConfig,
    MujocoPositionExecutionRig,
)
from sim.mujoco.deploy import (
    MujocoCoupledPoseToQposResolver,
    MujocoDeployConfig,
    MujocoDeployRuntime,
)
from sim.mujoco.ik import RM65CanonicalIK
from teleop_runtime.session import TeleopSession, TeleopSessionConfig
from teleop_runtime.control_trace import (
    ControlTraceRuntime,
    ControlTraceSpool,
    ControlTraceSpoolSink,
    ControlTraceWriterHealthLedger,
    EpisodeCaptureManifestStore,
    EpisodeTraceRecorderSink,
    RuntimeTraceSession,
)
from dataset_interface.control_trace import load_episode_control_trace, materialize_actual_decision_view


POSE_FRAME_HASH = "rm65-arm-base-tool-tcp-pose-frame-v1"


class ManualClock:
    def __init__(self, value: float = 30.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


class DeterministicScene:
    arm_specs = {"left": object(), "right": object()}
    gripper_specs = {}

    def __init__(self) -> None:
        self.qpos = {
            "left": np.zeros(6, dtype=float),
            "right": np.zeros(6, dtype=float),
        }

    def get_arm_qpos(self, side: str) -> np.ndarray:
        return self.qpos[side].copy()

    def step_kinematic_direct_qpos_debug(
        self,
        qpos_by_side,
        *,
        gripper_close_by_name,
        zero_controlled_qvel,
    ) -> None:
        del gripper_close_by_name, zero_controlled_qvel
        self.qpos = {
            side: np.asarray(value, dtype=float).reshape(6).copy()
            for side, value in qpos_by_side.items()
        }


class DeterministicCoupledPlanner:
    def reset(self, _qpos_by_side):
        return {"initialized": True}

    def plan_pose_reference(self, pose_by_side, **_kwargs):
        identity = np.eye(4, dtype=float)
        return SimpleNamespace(
            valid=True,
            hold=False,
            status="tracking",
            q_ref_by_side={
                "left": np.full(6, 0.02, dtype=float),
                "right": np.full(6, -0.02, dtype=float),
            },
            target_endpoint_pose_world_by_side={
                side: identity.copy() for side in pose_by_side
            },
            realized_endpoint_pose_world_by_side={
                side: identity.copy() for side in pose_by_side
            },
            modification_reasons=(),
            metadata={"planner": "cs6b1d-pose-fixture"},
        )


def _runtime(
    clock: ManualClock,
    *,
    trace_sink=None,
    composer_authority: bool = False,
    composition_strategy=None,
) -> MujocoDeployRuntime:
    scene = DeterministicScene()
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
            timing=TimingSpec(
                sample_hz=15.0,
                policy_hz=15.0,
                reference_hz=60.0,
                ik_hz=60.0,
                command_hz=120.0,
                physics_hz=1000.0,
            ),
            actuation_profile=None,
            gripper_actuation_profile=None,
            realtime_worker=False,
            follow_qpos_alpha=1.0,
        ),
        monotonic_clock=clock,
    )
    resolver = MujocoCoupledPoseToQposResolver(
        RM65CanonicalIK(scene, DeterministicCoupledPlanner())
    )
    return MujocoDeployRuntime(
        rig,
        MujocoDeployConfig(
            runtime_session_id="cs6b1d-pose-runtime",
            control_domain_id="cs6b1d-pose-domain",
            embodiment_instance_id="cs6b1d-pose-rm65",
            producer_id="human-main",
            composer_producer_id="pose-composer",
            replay_producer_id="replay-main",
            enable_replay_authority=not composer_authority,
            enable_composer_authority=composer_authority,
            enable_primary_policy_authority=not composer_authority,
            decision_hz=30.0,
            authority_policy_id="cs6b1d-pose-human-replay-lease-v1",
            initial_control_mode="hybrid",
            reference_activation_profile_id=SCHEDULED_REFERENCE_SEGMENT_SIM_V1,
            clock_domain_id="local_monotonic",
            pose_frame_contract_hash=POSE_FRAME_HASH,
            authority_takeover_allowed_producers=(
                ()
                if composer_authority
                else ("human-main", "replay-main")
            ),
        ),
        trace_sink=trace_sink,
        pose_resolver=resolver,
        shared_safety=(
            (
                lambda action, input_action_id: SharedCanonicalSafetyResult(
                    input_action_id=input_action_id,
                    safe_action=action,
                    status="safe",
                )
            )
            if composer_authority
            else None
        ),
        composition_strategy=composition_strategy,
        clock=clock,
    )


def _replay_source(runtime: MujocoDeployRuntime) -> ReplayProposalSource:
    action_contract = ReplaySourceActionContract(
        contract_id="cs6b1d-replay-action:arm-base-pose",
        interpretation_status=REPLAY_SOURCE_INTERPRETATION_VERIFIED,
        action_stage=REPLAY_ACTION_STAGE_SELECTED,
        action_space="bimanual_ee_pose_v1",
        component_ids=("left_arm", "right_arm"),
        selection_group_by_component={"left_arm": "arms", "right_arm": "arms"},
        coupled_selection_groups=("arms",),
        pose_frame_contract_hash=POSE_FRAME_HASH,
        component_schema_hash=runtime.catalog.semantic_content_hash,
        action_spec_hash_by_component={
            "left_arm": "rm65-tool-tcp-pose-v1",
            "right_arm": "rm65-tool-tcp-pose-v1",
        },
        units_convention_id="si_m_rad_xyzw_v1",
    )
    source_contract = ReplaySourceContract(
        contract_id="cs6b1d-replay-source:arm-base-pose",
        replay_mode=REPLAY_MODE_CANONICAL_ACTION,
        dataset_schema="dexteleop_dataset_v2.4",
        source_dataset_id="cs6b1d-pose-source",
        episode_id="episode-pose-0",
        action_stage=REPLAY_ACTION_STAGE_SELECTED,
        action_field="decision.selected_action",
        source_action_space="bimanual_ee_pose_v1",
        source_frame_contract_hash=POSE_FRAME_HASH,
        source_component_schema_hash=runtime.catalog.semantic_content_hash,
    )
    binding, assessment = build_replay_binding_assessment(
        action_contract,
        runtime.target_runtime_action_contract,
        binding_id="cs6b1d-pose-replay-to-mujoco",
    )
    assert assessment.verdict == REPLAY_BINDING_VERIFIED_COMPATIBLE
    return ReplayProposalSource(
        ReplayProposalSourceConfig(
            runtime_session_id=runtime.config.runtime_session_id,
            control_domain_id=runtime.config.control_domain_id,
            embodiment_instance_id=runtime.config.embodiment_instance_id,
            producer_id=runtime.config.replay_producer_id,
            producer_epoch=runtime.config.producer_epoch,
            component_schema_hash=runtime.catalog.semantic_content_hash,
            component_coverage=("left_arm", "right_arm"),
            selection_groups=("arms",),
            source_contract=source_contract,
            source_action_contract=action_contract,
            target_runtime_contract=runtime.target_runtime_action_contract,
            derivation_receipt=None,
            schedule=ReplayScheduleSpec(
                schedule_id="cs6b1d-pose-30hz-v1",
                schedule_mode=REPLAY_SCHEDULE_FIXED_RATE,
                source_hz=30.0,
                target_hz=None,
                resampling_policy="none",
                source_start_index=4,
            ),
            target_binding_spec=binding,
            binding_assessment=assessment,
            requested_use=REPLAY_BINDING_USE_EXECUTION_CANDIDATE,
        )
    )


def _pose_matrix(y: float) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = np.asarray([0.2, y, 0.3], dtype=float)
    return pose


class _PassiveTrackerSource:
    name = "virtual-passive-tracker"

    def __init__(self, timestamp: float = 10.0) -> None:
        self.timestamp = float(timestamp)

    def start(self) -> None:
        return None

    def close(self) -> None:
        return None

    def expected_channel_names(self) -> tuple[str, ...]:
        return ("left_wrist_pose", "right_wrist_pose")

    def get_latest_channels(self) -> dict[str, ObservationChannel]:
        return {
            f"{side}_wrist_pose": ObservationChannel(
                channel_name=f"{side}_wrist_pose",
                capability="pose6d",
                side=side,
                value=_pose_matrix(-0.1 if side == "left" else 0.1),
                frame_name="desk",
                timestamp=self.timestamp,
                source_name=self.name,
            )
            for side in ("left", "right")
        }


class _PassivePoseRetargeter:
    def __init__(self, side: str, timestamp: float = 10.0) -> None:
        self.side = side
        self.timestamp = float(timestamp)
        self.name = f"virtual-passive-{side}-pose"

    def retarget(self, observation, calibration) -> TeleopCommand:
        del observation, calibration
        command = TeleopCommand(timestamp=self.timestamp)
        command.add_command(
            EffectorCommand(
                effector_name=f"{self.side}_arm",
                side=self.side,
                command_type="cartesian_pose",
                value=_pose_matrix(-0.08 if self.side == "left" else 0.08),
                frame_name=f"{self.side}_arm_base",
            )
        )
        return command


def _passive_pose_session_source(
    spec: PassiveTeleopPoseSourceSpec,
    *,
    timestamp: float = 10.0,
):
    session = TeleopSession(
        TeleopSessionConfig(
            mode="cs6b1d-passive-pose-fixture",
            record_enabled=False,
            allow_hardware_after_debug_pass=False,
        ),
        sources=(_PassiveTrackerSource(timestamp),),
        retargeters=(
            _PassivePoseRetargeter("left", timestamp),
            _PassivePoseRetargeter("right", timestamp),
        ),
        rig=None,
    )
    return PassiveTeleopPoseSessionSource(session, spec)


def _replay_emission(
    path: Path,
    clock: ManualClock,
    *,
    source_action_index: int = 4,
    group_key: str = "pose-group-0",
    local_frame_index: int = 0,
):
    action = RobotAction(
        action_space="bimanual_ee_pose_v1",
        timestamp=clock(),
        arm_pose_by_side={
            "left": np.asarray([0.2, -0.1, 0.3, 0.0, 0.0, 0.0, 1.0]),
            "right": np.asarray([0.2, 0.1, 0.3, 0.0, 0.0, 0.0, 1.0]),
        },
        frame_by_effector={
            "left_arm": "left_arm_base",
            "right_arm": "right_arm_base",
        },
        controlled_frame_by_effector={
            "left_arm": "tool_tcp",
            "right_arm": "tool_tcp",
        },
        source="existing-replay-cursor",
    )
    return create_replay_source_frame_emission(
        source_path=str(path),
        source_episode_index=0,
        group_key=group_key,
        local_frame_index=local_frame_index,
        source_action_index=source_action_index,
        source_relative_time=0.0,
        action=action,
        observed_at=clock(),
        target_effective_at=clock(),
        effective_until=clock() + 0.1,
        clock_domain_id="local_monotonic",
    )


def _submit(station, clock, sequence, kind, subject=None, **bindings):
    state = station.station.state
    transition = station.station.submit_intent(
        StationIntentDraft(
            client_id="operator:cs6b1d",
            client_sequence=sequence,
            intent_kind=kind,
            produced_at=clock(),
            produced_clock_domain_id="local-monotonic",
            expected_state_revision=state.state_revision,
            manifest_hash=station.station.manifest.semantic_content_hash,
            subject_id=subject,
            reason="cs6b1d-pose-test",
            **bindings,
        )
    )
    execution = None
    if transition.receipt.status == "operation_requested":
        execution = station.complete_pending_operation(transition)
    return transition, execution


def _advance_to_replay(station, clock: ManualClock) -> None:
    target = station.station.manifest.execution_binding.binding_id
    operations = (
        ("acquire_control_lease", None),
        ("connect_target", target),
        ("connect_source", "replay-slot"),
        ("connect_source", "human-slot"),
        ("run_preflight", target),
        ("run_preflight", "replay-slot"),
        ("run_preflight", "human-slot"),
        ("request_candidate", "replay-slot"),
        ("request_motion_authority", "replay-slot"),
        ("request_candidate", "human-slot"),
    )
    for sequence, (kind, subject) in enumerate(operations):
        transition, execution = _submit(station, clock, sequence, kind, subject)
        assert transition.receipt.accepted
        if execution is not None:
            assert execution.completion_transition.result.succeeded


def _build_cs6b1f_station_to_replay(
    tmp_path: Path,
    clock: ManualClock,
    *,
    pose_fusion_shadow_spec: CompositionSpec | None = None,
):
    """Build the opt-in HIL fixture through replay authority only."""

    runtime = _runtime(clock)
    source_spec = PassiveTeleopPoseSourceSpec(source_id="human-slot")
    passive_source = _passive_pose_session_source(
        source_spec,
        timestamp=clock(),
    )
    replay_source = _replay_source(runtime)
    path = tmp_path / "cs6b1f-corner.parquet"
    observer = ReplaySourceObserverBinding.create(
        source_slot_id="replay-slot",
        source_locator_id="cs6b1d-pose-source",
        expected_source_path=str(path),
        source_episode_index=0,
        group_key="pose-group-0",
        source_start_index=4,
        clock_domain_id="local_monotonic",
    )
    station = build_scoped_cs6b1f_mujoco_pose_human_replay_station(
        runtime,
        source_spec,
        replay_source,
        observer,
        pose_fusion_shadow_spec=pose_fusion_shadow_spec,
        passive_human_pose_source=passive_source,
        station_clock=clock,
        station_clock_domain_id="local-monotonic",
    )
    target = station.station.manifest.execution_binding.binding_id
    try:
        setup = (
            ("acquire_control_lease", None),
            ("connect_target", target),
            ("connect_source", "replay-slot"),
            ("connect_source", "human-slot"),
        )
        for sequence, (kind, subject) in enumerate(setup):
            transition, execution = _submit(station, clock, sequence, kind, subject)
            assert transition.receipt.accepted
            if execution is not None:
                assert execution.completion_transition.result.succeeded
        station.set_passive_human_calibration_ready()
        runtime_setup = (
            ("run_preflight", target),
            ("run_preflight", "replay-slot"),
            ("run_preflight", "human-slot"),
            ("request_candidate", "replay-slot"),
            ("request_motion_authority", "replay-slot"),
            ("request_candidate", "human-slot"),
        )
        for sequence, (kind, subject) in enumerate(runtime_setup, start=len(setup)):
            transition, execution = _submit(station, clock, sequence, kind, subject)
            assert transition.receipt.accepted
            if execution is not None:
                assert execution.completion_transition.result.succeeded
    except Exception:
        station.close()
        raise
    return station, runtime, passive_source, path


def _pose_fusion_shadow_spec(
    *,
    translation_limit: float = 0.2,
) -> CompositionSpec:
    return CompositionSpec(
        spec_id="cs6b1g-human-replay-weighted-se3-v1",
        algebra="weighted_se3_pose_fusion",
        input_roles=("human_pose", "replay_pose"),
        required_roles=("human_pose", "replay_pose"),
        weight_by_role={"human_pose": 0.5, "replay_pose": 0.5},
        max_translation_disagreement_m=translation_limit,
        max_rotation_disagreement_rad=np.deg2rad(60.0),
    )


def _pose_composer_authority_gate(
    spec: CompositionSpec,
) -> CompositionAuthorityGateSpec:
    return CompositionAuthorityGateSpec(
        gate_spec_id="cs6b1h-human-replay-formal-v1",
        composition_spec_hash=spec.semantic_content_hash,
        composer_producer_id="pose-composer",
        missing_behavior_by_role={
            "human_pose": "hold",
            "replay_pose": "hold",
        },
        proposal_ttl_sec=0.1,
    )


def _build_cs6b1h_pose_composer(
    clock: ManualClock,
    *,
    translation_limit: float = 0.2,
    source_differences: bool = False,
    formal_v2: bool = False,
):
    spec = _pose_fusion_shadow_spec(translation_limit=translation_limit)
    strategy = (
        DeterministicCompositionStrategyV2(
            FormalCompositionStrategySpec(
                strategy_id="cs6b1h-deterministic-migration-v2",
                strategy_version="2.0",
                composition_spec_hash=spec.semantic_content_hash,
            )
        )
        if formal_v2
        else None
    )
    runtime = _runtime(
        clock,
        composer_authority=True,
        composition_strategy=strategy,
    )
    source_spec = PassiveTeleopPoseSourceSpec(source_id="human-slot")
    passive_source = _passive_pose_session_source(
        source_spec,
        timestamp=clock(),
    )
    difference_spec = (
        SourceDifferenceSpec(
            spec_id="cs6b1h-human-replay-difference-v1",
            composition_spec_hash=spec.semantic_content_hash,
            compared_reference_role_pairs=(
                ("human_pose", "replay_pose"),
            ),
            output_residual_roles=("human_pose", "replay_pose"),
        )
        if source_differences
        else None
    )
    composition = build_scoped_cs6b1h_mujoco_pose_composer_authority(
        runtime,
        source_spec,
        _replay_source(runtime),
        spec,
        _pose_composer_authority_gate(spec),
        source_difference_spec=difference_spec,
    )
    return composition, runtime, passive_source


def _build_cs6b1i_pose_composition_session(
    tmp_path: Path,
    clock: ManualClock,
    *,
    trace_sink=None,
    source_differences: bool = False,
):
    runtime = _runtime(
        clock,
        composer_authority=True,
        trace_sink=trace_sink,
    )
    source_spec = PassiveTeleopPoseSourceSpec(source_id="human-slot")
    passive_source = _passive_pose_session_source(
        source_spec,
        timestamp=clock(),
    )
    replay_source = _replay_source(runtime)
    composition_spec = _pose_fusion_shadow_spec()
    difference_spec = (
        SourceDifferenceSpec(
            spec_id="cs6b1i-human-replay-difference-v1",
            composition_spec_hash=composition_spec.semantic_content_hash,
            compared_reference_role_pairs=(
                ("human_pose", "replay_pose"),
            ),
            output_residual_roles=("human_pose", "replay_pose"),
        )
        if source_differences
        else None
    )
    observer = ReplaySourceObserverBinding.create(
        source_slot_id="replay-slot",
        source_locator_id=replay_source.config.source_contract.source_dataset_id,
        expected_source_path=str(tmp_path / "cs6b1i-replay.parquet"),
        source_episode_index=0,
        group_key="pose-group-0",
        source_start_index=4,
        clock_domain_id=runtime.config.clock_domain_id,
    )
    station = build_scoped_cs6b1i_mujoco_pose_composition_session(
        runtime,
        source_spec,
        replay_source,
        observer,
        composition_spec,
        _pose_composer_authority_gate(composition_spec),
        station_clock=clock,
        station_clock_domain_id="local_monotonic",
        source_difference_spec=difference_spec,
    )
    session_spec = CompositionSessionOperationSpec(
        session_key="cs6b1i-session-0",
        composer_source_slot_id=station.composer_source_slot_id,
        contributor_source_slot_ids=(
            station.human_source_slot_id,
            station.replay_source_slot_id,
        ),
        composition_spec_hash=composition_spec.semantic_content_hash,
        authority_gate_spec_hash=(
            station.data_plane.authority_gate_spec.semantic_content_hash
        ),
        execution_binding_hash=(
            station.station.manifest.execution_binding.semantic_content_hash
        ),
        capture_boundary_id="episode-boundary:cs6b1i-session-0",
    )
    return station, runtime, passive_source, session_spec


def _setup_and_start_cs6b1i_station(
    station,
    session_spec: CompositionSessionOperationSpec,
    clock: ManualClock,
) -> int:
    sequence = 0

    def submit(kind, subject_id=None, *, operation_spec_id=None):
        nonlocal sequence
        state = station.station.state
        transition = station.station.submit_intent(
            StationIntentDraft(
                client_id="operator:cs6b1i",
                client_sequence=sequence,
                intent_kind=kind,
                produced_at=clock(),
                produced_clock_domain_id="local_monotonic",
                expected_state_revision=state.state_revision,
                manifest_hash=station.station.manifest.semantic_content_hash,
                subject_id=subject_id,
                resource_binding_id=(
                    station.station.manifest.execution_binding.binding_id
                    if kind
                    in {
                        "start_composition_session",
                        "stop_composition_session",
                    }
                    else None
                ),
                operation_spec_id=operation_spec_id,
                reason="cs6b1i deterministic sim fixture",
            )
        )
        sequence += 1
        if transition.receipt.status == "operation_requested":
            execution = station.complete_pending_operation(transition)
            assert execution.completion_transition.accepted
            assert execution.completion_transition.result.succeeded
        else:
            assert transition.receipt.accepted
        return transition

    submit("acquire_control_lease")
    target_id = station.station.manifest.execution_binding.binding_id
    submit("connect_target", target_id)
    for source_id in (
        station.human_source_slot_id,
        station.replay_source_slot_id,
        station.composer_source_slot_id,
    ):
        submit("connect_source", source_id)
    submit("run_preflight", target_id)
    for source_id in (
        station.human_source_slot_id,
        station.replay_source_slot_id,
        station.composer_source_slot_id,
    ):
        submit("run_preflight", source_id)
    submit("request_candidate", station.composer_source_slot_id)
    station.register_composition_session(session_spec)
    submit(
        "start_composition_session",
        station.composer_source_slot_id,
        operation_spec_id=session_spec.operation_spec_id,
    )
    return sequence


def _action_hash(action: RobotAction) -> str:
    return canonical_content_hash(
        robot_action_semantic_payload(action),
        domain="test.cs6b1g.authoritative_action.v1",
    )


def test_cs6b1g_shared_cut_fuses_passive_human_and_replay_without_authority(
    tmp_path: Path,
) -> None:
    """Shadow facts share the receiver cut but never become proposals."""

    def run(spec: CompositionSpec | None):
        clock = ManualClock(30.0)
        station, runtime, _, path = _build_cs6b1f_station_to_replay(
            tmp_path,
            clock,
            pose_fusion_shadow_spec=spec,
        )
        try:
            replay = station.submit_replay_source_frame_emission(
                _replay_emission(path, clock)
            )
            assert replay.proposal_admission is not None
            assert replay.proposal_admission.accepted
            assert replay.proposal_admission.proposal is not None
            passive = station.step_passive_human_pose_source(now=clock())
            assert passive.emission_valid and passive.emission is not None
            step = station.step(now=clock(), physics_ticks_after_dispatch=1)
            assert step.status == "dispatched"
            assert step.decision is not None
            assert step.snapshot is not None
            return {
                "proposal_id": replay.proposal_admission.proposal.proposal_id,
                "proposal_hash": (
                    replay.proposal_admission.proposal.semantic_content_hash
                ),
                "action_hash": _action_hash(step.decision.selected_action),
                "decision_hash": step.decision.semantic_content_hash,
                "cut_sequence": (
                    step.snapshot.proposal_snapshot.cutoff_admission_sequence
                ),
                "proposal_count": len(runtime.core.proposals),
                "fusion": station.data_plane.last_pose_fusion_shadow_submission,
                "input_receipts": (
                    station.data_plane.pose_fusion_shadow_input_receipt_by_role
                ),
                "active_source": station.data_plane.active_source_slot_id,
                "trace_types": tuple(fact.event_type for fact in runtime.trace_facts),
                "trace_facts": runtime.trace_facts,
                }
        finally:
            station.close()

    baseline = run(None)
    shadow = run(_pose_fusion_shadow_spec())

    assert shadow["proposal_id"] == baseline["proposal_id"]
    assert shadow["proposal_hash"] == baseline["proposal_hash"]
    assert shadow["action_hash"] == baseline["action_hash"]
    assert shadow["proposal_count"] == baseline["proposal_count"] == 1
    assert shadow["active_source"] == baseline["active_source"] == "replay-slot"
    assert shadow["cut_sequence"] > baseline["cut_sequence"]
    assert shadow["decision_hash"] != baseline["decision_hash"]

    fusion = shadow["fusion"]
    assert fusion is not None
    assert fusion.proposal_admission is None
    assert fusion.input_snapshot.cutoff_admission_sequence == (
        shadow["cut_sequence"]
    )
    assert fusion.composition_result.output_status == "ready"
    assert fusion.composition_result.composed_action is not None
    np.testing.assert_allclose(
        fusion.composition_result.composed_action.arm_pose_by_side["left"][:3],
        np.asarray([0.2, -0.09, 0.3]),
        atol=1e-6,
    )
    receipts = shadow["input_receipts"]
    assert set(receipts) == {"human_pose", "replay_pose"}
    assert all(receipt.accepted for receipt in receipts.values())
    assert all(
        receipt.composition_input is not None
        and not receipt.composition_input.authority_candidate
        for receipt in receipts.values()
    )
    assert "composition_contributor_input" in shadow["trace_types"]
    assert "composition_input_snapshot" in shadow["trace_types"]
    assert "composition_result" in shadow["trace_types"]
    assert "human_command_emission" in shadow["trace_types"]
    human_input = receipts["human_pose"].composition_input
    assert human_input is not None
    assert human_input.policy_output_id is None
    source_representation_facts = {
        fact.fact_id: fact
        for fact in shadow["trace_facts"]
        if fact.event_type == "source_adapter_receipt"
    }
    representation_fact_ids = set(human_input.source_fact_ids).intersection(
        source_representation_facts
    )
    assert len(representation_fact_ids) == 1
    representation_fact_id = representation_fact_ids.pop()
    assert (
        source_representation_facts[representation_fact_id].payload[
            "source_emission_id"
        ]
        == human_input.audit_metadata["parent_fact_id"]
    )


def test_cs6b1g_missing_or_expired_human_input_is_no_action_only_in_shadow(
    tmp_path: Path,
) -> None:
    clock = ManualClock(30.0)
    station, _, _, path = _build_cs6b1f_station_to_replay(
        tmp_path,
        clock,
        pose_fusion_shadow_spec=_pose_fusion_shadow_spec(),
    )
    try:
        station.submit_replay_source_frame_emission(_replay_emission(path, clock))
        first = station.step(now=clock(), physics_ticks_after_dispatch=1)
        assert first.status == "dispatched"
        missing = station.data_plane.last_pose_fusion_shadow_submission
        assert missing is not None
        assert missing.composition_result.output_status == "no_action"
        assert missing.composition_result.reason == "missing_required:human_pose"
    finally:
        station.close()

    clock = ManualClock(30.0)
    station, _, _, path = _build_cs6b1f_station_to_replay(
        tmp_path,
        clock,
        pose_fusion_shadow_spec=_pose_fusion_shadow_spec(),
    )
    try:
        passive = station.step_passive_human_pose_source(now=clock())
        assert passive.emission_valid
        clock.value += 0.2
        station.submit_replay_source_frame_emission(_replay_emission(path, clock))
        dispatched = station.step(now=clock(), physics_ticks_after_dispatch=1)
        assert dispatched.status == "dispatched"
        expired = station.data_plane.last_pose_fusion_shadow_submission
        assert expired is not None
        assert expired.composition_result.output_status == "no_action"
        assert expired.composition_result.reason == "missing_required:human_pose"
    finally:
        station.close()


def test_cs6b1g_pose_conflict_is_invalid_without_changing_replay_dispatch(
    tmp_path: Path,
) -> None:
    clock = ManualClock(30.0)
    station, runtime, _, path = _build_cs6b1f_station_to_replay(
        tmp_path,
        clock,
        pose_fusion_shadow_spec=_pose_fusion_shadow_spec(
            translation_limit=0.005,
        ),
    )
    try:
        replay = station.submit_replay_source_frame_emission(
            _replay_emission(path, clock)
        )
        assert replay.proposal_admission is not None
        station.step_passive_human_pose_source(now=clock())
        step = station.step(now=clock(), physics_ticks_after_dispatch=1)
        assert step.status == "dispatched"
        assert step.decision is not None
        np.testing.assert_allclose(
            step.decision.selected_action.arm_pose_by_side["left"][:3],
            np.asarray([0.2, -0.1, 0.3]),
        )
        fusion = station.data_plane.last_pose_fusion_shadow_submission
        assert fusion is not None
        assert fusion.composition_result.output_status == "invalid"
        assert "translation_disagreement" in fusion.composition_result.reason
        assert len(runtime.core.proposals) == 1
        assert station.data_plane.active_source_slot_id == "replay-slot"
    finally:
        station.close()


def test_cs6b1g_duplicate_passive_emission_is_isolated_from_replay(
    tmp_path: Path,
) -> None:
    clock = ManualClock(30.0)
    station, runtime, _, path = _build_cs6b1f_station_to_replay(
        tmp_path,
        clock,
        pose_fusion_shadow_spec=_pose_fusion_shadow_spec(),
    )
    try:
        station.submit_replay_source_frame_emission(_replay_emission(path, clock))
        passive = station.step_passive_human_pose_source(now=clock())
        assert passive.emission_valid and passive.emission is not None
        duplicate = (
            station.data_plane.submit_passive_human_pose_fusion_shadow_input(
                passive.emission,
                observed_at=clock() + 0.01,
            )
        )
        assert duplicate is not None, (
            station.data_plane.pose_fusion_shadow_error_by_role
        )
        assert duplicate.accepted
        assert duplicate.composition_input_id == (
            station.data_plane.pose_fusion_shadow_input_receipt_by_role[
                "human_pose"
            ].composition_input_id
        )

        step = station.step(now=clock(), physics_ticks_after_dispatch=1)
        assert step.status == "dispatched"
        assert len(runtime.core.proposals) == 1
        fusion = station.data_plane.last_pose_fusion_shadow_submission
        assert fusion is not None
        assert fusion.composition_result.output_status == "ready"
        assert sum(
            fact.event_type == "human_command_emission"
            for fact in runtime.trace_facts
        ) == 1
        assert sum(
            fact.event_type == "composition_contributor_input"
            and fact.payload["input_role"] == "human_pose"
            for fact in runtime.trace_facts
        ) == 1
    finally:
        station.close()


def test_cs6b1h_composer_is_the_only_formal_pose_producer(
    tmp_path: Path,
) -> None:
    clock = ManualClock(30.0)
    composition, runtime, passive_source = _build_cs6b1h_pose_composer(
        clock
    )
    try:
        composition.start()
        passive_source.start()
        passive_source.set_calibration_ready()
        replay = composition.submit_replay_emission(
            _replay_emission(tmp_path / "cs6b1h.parquet", clock)
        )
        human_step = passive_source.step(now=clock())
        assert human_step.emission_valid and human_step.emission is not None
        human = composition.submit_human_emission(
            human_step.emission,
            observed_at=clock(),
        )
        assert replay.accepted and human.accepted
        assert all(
            receipt.composition_input is not None
            and not receipt.composition_input.authority_candidate
            for receipt in (replay, human)
        )
        formal = composition.compose_at(
            decision_key="cs6b1h-decision-0",
            target_effective_at=clock(),
        )
        assert formal.error is None
        assert formal.source_difference_observation is None
        assert formal.source_difference_error is None
        assert composition.binding.source_difference_spec_hash is None
        assert composition.binding.formal_strategy_spec_hash is None
        binding = composition.binding
        assert binding.semantic_content_hash == canonical_content_hash(
            {
                "runtime_session_id": binding.runtime_session_id,
                "control_domain_id": binding.control_domain_id,
                "embodiment_instance_id": binding.embodiment_instance_id,
                "composer_producer_id": binding.composer_producer_id,
                "human_producer_id": binding.human_producer_id,
                "replay_producer_id": binding.replay_producer_id,
                "composition_spec_hash": binding.composition_spec_hash,
                "authority_gate_spec_hash": (
                    binding.authority_gate_spec_hash
                ),
                "target_runtime_contract_hash": (
                    binding.target_runtime_contract_hash
                ),
            },
            domain="dexteleop.cs6b1h_pose_composer_binding.v1",
        )
        assert formal.disposition is not None
        assert formal.disposition.action == "publish_proposal"
        assert formal.proposal_admission is not None
        assert formal.proposal_admission.accepted
        proposal = formal.proposal_admission.proposal
        assert proposal is not None
        assert proposal.producer_kind == "composer"
        assert proposal.producer_id == "pose-composer"
        assert len(runtime.core.proposals) == 1
        facts_by_id = {fact.fact_id: fact for fact in runtime.trace_facts}
        contributor_facts = tuple(
            fact
            for fact in runtime.trace_facts
            if fact.event_type == "composition_contributor_input"
        )
        assert len(contributor_facts) == 2
        for contributor in contributor_facts:
            for source_id, source_hash in zip(
                contributor.payload["source_fact_ids"],
                contributor.payload["source_fact_hashes"],
            ):
                assert facts_by_id[source_id].semantic_content_hash == source_hash
        disposition_facts = tuple(
            fact
            for fact in runtime.trace_facts
            if fact.event_type == "composition_authority_disposition"
        )
        assert len(disposition_facts) == 1
        assert disposition_facts[0].payload["action"] == "publish_proposal"
        assert not any(
            fact.event_type
            in {"source_difference_spec", "source_difference_observation"}
            for fact in runtime.trace_facts
        )

        duplicate = composition.compose_at(
            decision_key="cs6b1h-decision-0",
            target_effective_at=clock(),
        )
        assert duplicate is formal
        assert len(runtime.core.proposals) == 1

        dispatched = runtime.step(
            now=clock(),
            physics_ticks_after_dispatch=1,
        )
        assert dispatched.status == "dispatched"
        assert dispatched.decision is not None
        assert dispatched.decision.selected_by_group == {
            "arms": proposal.proposal_id,
        }
    finally:
        passive_source.close()
        composition.close()


def test_cs6b1h_formal_v2_is_opt_in_and_preserves_pose_numerics(
    tmp_path: Path,
) -> None:
    clock = ManualClock(30.0)
    composition, runtime, passive_source = _build_cs6b1h_pose_composer(
        clock,
        formal_v2=True,
    )
    try:
        composition.start()
        passive_source.start()
        passive_source.set_calibration_ready()
        replay = composition.submit_replay_emission(
            _replay_emission(tmp_path / "cs6b1h-formal-v2.parquet", clock)
        )
        human_step = passive_source.step(now=clock())
        assert human_step.emission_valid and human_step.emission is not None
        human = composition.submit_human_emission(
            human_step.emission,
            observed_at=clock(),
        )
        assert replay.accepted and human.accepted

        formal = composition.compose_at(
            decision_key="cs6b1h-formal-v2-decision-0",
            target_effective_at=clock(),
        )
        assert formal.error is None
        assert formal.evaluation is not None
        result = formal.evaluation.composition_result
        strategy = runtime.policy_topology.composition_strategy
        assert strategy is not None
        assert result.strategy_id == strategy.spec.strategy_id
        assert result.strategy_spec_hash == strategy.spec.semantic_content_hash
        assert composition.binding.formal_strategy_spec_hash == (
            strategy.spec.semantic_content_hash
        )
        binding = composition.binding
        assert binding.semantic_content_hash == canonical_content_hash(
            {
                "runtime_session_id": binding.runtime_session_id,
                "control_domain_id": binding.control_domain_id,
                "embodiment_instance_id": binding.embodiment_instance_id,
                "composer_producer_id": binding.composer_producer_id,
                "human_producer_id": binding.human_producer_id,
                "replay_producer_id": binding.replay_producer_id,
                "composition_spec_hash": binding.composition_spec_hash,
                "authority_gate_spec_hash": (
                    binding.authority_gate_spec_hash
                ),
                "target_runtime_contract_hash": (
                    binding.target_runtime_contract_hash
                ),
                "formal_strategy_spec_hash": (
                    binding.formal_strategy_spec_hash
                ),
            },
            domain="dexteleop.cs6b1h_pose_composer_binding.v2",
        )

        legacy = DeterministicComposer().compose(
            formal.evaluation.input_snapshot,
            spec=composition.composition_spec,
        )
        assert legacy.composed_action_hash == result.composed_action_hash
        for side in ("left", "right"):
            np.testing.assert_array_equal(
                legacy.composed_action.arm_pose_by_side[side],
                result.composed_action.arm_pose_by_side[side],
            )
        assert legacy.semantic_content_hash != result.semantic_content_hash

        assert formal.proposal_admission is not None
        assert formal.proposal_admission.accepted
        proposal = formal.proposal_admission.proposal
        assert proposal is not None
        assert proposal.policy_output_id == result.composition_result_id
        assert len(runtime.core.proposals) == 1
        event_types = tuple(fact.event_type for fact in runtime.trace_facts)
        assert event_types.count("formal_composition_strategy_spec") == 1
        assert "formal_composition_strategy_input" in event_types
        assert "formal_composition_strategy_result_binding" in event_types

        dispatched = runtime.step(
            now=clock(),
            physics_ticks_after_dispatch=1,
        )
        assert dispatched.status == "dispatched"
        assert dispatched.decision is not None
        assert dispatched.decision.selected_by_group == {
            "arms": proposal.proposal_id,
        }
    finally:
        passive_source.close()
        composition.close()


def test_cs6b1h_rejects_a_formal_strategy_bound_to_another_spec() -> None:
    clock = ManualClock(30.0)
    expected_spec = _pose_fusion_shadow_spec()
    other_spec = _pose_fusion_shadow_spec(translation_limit=0.19)
    runtime = _runtime(
        clock,
        composer_authority=True,
        composition_strategy=DeterministicCompositionStrategyV2(
            FormalCompositionStrategySpec(
                strategy_id="wrong-spec-migration-v2",
                strategy_version="2.0",
                composition_spec_hash=other_spec.semantic_content_hash,
            )
        ),
    )
    source_spec = PassiveTeleopPoseSourceSpec(source_id="human-slot")
    try:
        with pytest.raises(
            ValueError,
            match="another composition spec",
        ):
            build_scoped_cs6b1h_mujoco_pose_composer_authority(
                runtime,
                source_spec,
                _replay_source(runtime),
                expected_spec,
                _pose_composer_authority_gate(expected_spec),
            )
    finally:
        runtime.close()


def test_cs6b1h_missing_or_conflicting_contributor_forces_hold(
    tmp_path: Path,
) -> None:
    clock = ManualClock(30.0)
    composition, runtime, passive_source = _build_cs6b1h_pose_composer(
        clock,
        translation_limit=0.005,
    )
    try:
        composition.start()
        passive_source.start()
        passive_source.set_calibration_ready()
        composition.submit_replay_emission(
            _replay_emission(tmp_path / "cs6b1h-missing.parquet", clock)
        )
        missing = composition.compose_at(
            decision_key="cs6b1h-missing-human",
            target_effective_at=clock(),
        )
        assert missing.proposal_admission is None
        assert missing.disposition is not None
        assert missing.disposition.reason == "required_contributor_missing"
        assert missing.hold_transition is not None
        assert missing.hold_transition.event.status == "hold"
        assert len(runtime.core.proposals) == 0
        assert any(
            fact.event_type == "composition_authority_disposition"
            and fact.payload["action"] == "hold"
            for fact in runtime.trace_facts
        )
        human_step = passive_source.step(now=clock())
        assert human_step.emission_valid and human_step.emission is not None
        composition.submit_human_emission(
            human_step.emission,
            observed_at=clock(),
        )
        still_held = composition.compose_at(
            decision_key="cs6b1h-no-auto-reopen",
            target_effective_at=clock(),
        )
        assert still_held.blocked_reason == "execution_gate_not_ready:hold"
        assert still_held.proposal_admission is None
        assert len(runtime.core.proposals) == 0
    finally:
        passive_source.close()
        composition.close()

    clock = ManualClock(30.0)
    composition, runtime, passive_source = _build_cs6b1h_pose_composer(
        clock,
        translation_limit=0.005,
    )
    try:
        composition.start()
        passive_source.start()
        passive_source.set_calibration_ready()
        composition.submit_replay_emission(
            _replay_emission(tmp_path / "cs6b1h-conflict.parquet", clock)
        )
        human_step = passive_source.step(now=clock())
        assert human_step.emission is not None
        composition.submit_human_emission(
            human_step.emission,
            observed_at=clock(),
        )
        conflict = composition.compose_at(
            decision_key="cs6b1h-conflict",
            target_effective_at=clock(),
        )
        assert conflict.proposal_admission is None
        assert conflict.disposition is not None
        assert conflict.disposition.reason == "composition_invalid"
        assert conflict.hold_transition is not None
        assert conflict.hold_transition.event.status == "hold"
        assert len(runtime.core.proposals) == 0
    finally:
        passive_source.close()
        composition.close()


def test_cs6b1h_ttl_and_producer_epoch_invalidate_old_contributors(
    tmp_path: Path,
) -> None:
    clock = ManualClock(30.0)
    composition, runtime, passive_source = _build_cs6b1h_pose_composer(
        clock
    )
    try:
        composition.start()
        passive_source.start()
        passive_source.set_calibration_ready()
        composition.submit_replay_emission(
            _replay_emission(tmp_path / "cs6b1h-ttl.parquet", clock)
        )
        human_step = passive_source.step(now=clock())
        assert human_step.emission is not None
        composition.submit_human_emission(
            human_step.emission,
            observed_at=clock(),
        )
        clock.value += 0.2
        expired = composition.compose_at(
            decision_key="cs6b1h-all-expired",
            target_effective_at=clock(),
        )
        assert expired.disposition is not None
        assert expired.disposition.action == "hold"
        assert expired.disposition.reason == "all_inputs_missing"
        assert expired.hold_transition is not None
        assert len(runtime.core.proposals) == 0
    finally:
        passive_source.close()
        composition.close()

    clock = ManualClock(30.0)
    composition, runtime, passive_source = _build_cs6b1h_pose_composer(
        clock
    )
    try:
        composition.start()
        passive_source.start()
        passive_source.set_calibration_ready()
        composition.submit_replay_emission(
            _replay_emission(tmp_path / "cs6b1h-epoch.parquet", clock)
        )
        human_step = passive_source.step(now=clock())
        assert human_step.emission is not None
        composition.submit_human_emission(
            human_step.emission,
            observed_at=clock(),
        )
        runtime.activate_policy_node_producer(
            "replay-main",
            producer_epoch=2,
            reason="cs6b1h replay contributor replacement",
        )
        invalidated = composition.compose_at(
            decision_key="cs6b1h-old-replay-epoch",
            target_effective_at=clock(),
        )
        assert invalidated.evaluation is not None
        assert invalidated.evaluation.input_snapshot.missing_roles == (
            "replay_pose",
        )
        assert invalidated.disposition is not None
        assert invalidated.disposition.reason == (
            "required_contributor_missing"
        )
        assert invalidated.hold_transition is not None
        assert len(runtime.core.proposals) == 0
    finally:
        passive_source.close()
        composition.close()


def test_cs6b1h_promotion_error_holds_without_source_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = ManualClock(30.0)
    composition, runtime, passive_source = _build_cs6b1h_pose_composer(
        clock
    )
    try:
        composition.start()
        passive_source.start()
        passive_source.set_calibration_ready()
        composition.submit_replay_emission(
            _replay_emission(tmp_path / "cs6b1h-error.parquet", clock)
        )
        human_step = passive_source.step(now=clock())
        assert human_step.emission is not None
        composition.submit_human_emission(
            human_step.emission,
            observed_at=clock(),
        )

        def fail_promotion(*_args, **_kwargs):
            raise RuntimeError("synthetic composer promotion failure")

        monkeypatch.setattr(
            runtime,
            "submit_evaluated_composition",
            fail_promotion,
        )
        failed = composition.compose_at(
            decision_key="cs6b1h-promotion-error",
            target_effective_at=clock(),
        )
        assert failed.error == (
            "RuntimeError:synthetic composer promotion failure"
        )
        assert failed.hold_transition is not None
        assert failed.hold_transition.event.status == "hold"
        assert len(runtime.core.proposals) == 0
    finally:
        passive_source.close()
        composition.close()


def test_cs6b1i_station_session_publishes_only_composer_and_stops_in_hold(
    tmp_path: Path,
) -> None:
    clock = ManualClock(30.0)
    station, runtime, passive_source, session_spec = (
        _build_cs6b1i_pose_composition_session(tmp_path, clock)
    )
    try:
        passive_source.start()
        passive_source.set_calibration_ready()
        sequence = _setup_and_start_cs6b1i_station(
            station,
            session_spec,
            clock,
        )
        session_state = station.current_session_state
        assert session_state is not None and session_state.phase == "active"
        assert station.publication_barrier.snapshot().enabled
        assert station.station.state.source_state_by_slot == {
            station.human_source_slot_id: "preflighted",
            station.replay_source_slot_id: "preflighted",
            station.composer_source_slot_id: "active",
        }
        station.submit_replay_emission(
            _replay_emission(tmp_path / "cs6b1i-replay.parquet", clock)
        )
        human_step = passive_source.step(now=clock())
        assert human_step.emission is not None
        station.submit_human_emission(
            human_step.emission,
            observed_at=clock(),
        )
        composed = station.compose_at(
            decision_key="cs6b1i-decision-0",
            target_effective_at=clock(),
        )
        assert composed.source_difference_observation is None
        assert composed.source_difference_error is None
        assert composed.proposal_admission is not None
        proposal = composed.proposal_admission.proposal
        assert proposal is not None
        assert proposal.producer_kind == "composer"
        assert proposal.producer_id == runtime.config.composer_producer_id
        assert len(runtime.core.proposals) == 1
        dispatched = station.step(
            now=clock(),
            physics_ticks_after_dispatch=60,
        )
        assert dispatched.status == "dispatched"

        state = station.station.state
        stop_request = station.station.submit_intent(
            StationIntentDraft(
                client_id="operator:cs6b1i",
                client_sequence=sequence,
                intent_kind="stop_composition_session",
                produced_at=clock(),
                produced_clock_domain_id="local_monotonic",
                expected_state_revision=state.state_revision,
                manifest_hash=station.station.manifest.semantic_content_hash,
                subject_id=station.composer_source_slot_id,
                resource_binding_id=(
                    station.station.manifest.execution_binding.binding_id
                ),
                operation_spec_id=session_spec.operation_spec_id,
                reason="cs6b1i episode end",
            )
        )
        assert stop_request.receipt.status == "operation_requested"
        stopped = station.complete_pending_operation(stop_request)
        assert stopped.completion_transition.result.succeeded
        assert station.current_session_state is not None
        assert station.current_session_state.phase == "ended"
        assert station.station.state.lifecycle_state == "hold"
        assert not station.publication_barrier.snapshot().enabled
        with pytest.raises(RuntimeError, match="not active"):
            station.submit_replay_emission(
                _replay_emission(
                    tmp_path / "cs6b1i-after-end.parquet",
                    clock,
                )
            )
    finally:
        passive_source.close()
        station.close()


def test_cs6b1i_optional_source_difference_is_same_cut_and_idempotent(
    tmp_path: Path,
) -> None:
    clock = ManualClock(30.0)
    station, runtime, passive_source, session_spec = (
        _build_cs6b1i_pose_composition_session(
            tmp_path,
            clock,
            source_differences=True,
        )
    )
    try:
        passive_source.start()
        passive_source.set_calibration_ready()
        _setup_and_start_cs6b1i_station(station, session_spec, clock)
        station.submit_replay_emission(
            _replay_emission(
                tmp_path / "cs6b1i-difference-replay.parquet",
                clock,
            )
        )
        human_step = passive_source.step(now=clock())
        assert human_step.emission is not None
        station.submit_human_emission(
            human_step.emission,
            observed_at=clock(),
        )
        composed = station.compose_at(
            decision_key="cs6b1i-difference-0",
            target_effective_at=clock(),
        )
        observation = composed.source_difference_observation
        assert composed.source_difference_error is None
        assert observation is not None
        assert composed.evaluation is not None
        assert observation.composition_input_snapshot_id == (
            composed.evaluation.input_snapshot.composition_input_snapshot_id
        )
        assert observation.composition_input_snapshot_hash == (
            composed.evaluation.input_snapshot.semantic_content_hash
        )
        assert observation.composition_result_id == (
            composed.evaluation.composition_result.composition_result_id
        )
        assert observation.composition_result_hash == (
            composed.evaluation.composition_result.semantic_content_hash
        )
        assert observation.target_effective_at == clock()
        assert len(observation.pairwise_differences) == 1
        pair = observation.pairwise_differences[0]
        assert (pair.compared_role, pair.reference_role) == (
            "human_pose",
            "replay_pose",
        )
        assert tuple(
            metric.component_id for metric in pair.component_metrics
        ) == ("left_arm", "right_arm")
        assert len(observation.output_residuals) == 2
        assert composed.evaluation.evaluation_record is not None
        human_differences = (
            composed.evaluation.evaluation_record.differences_for_role(
                "human_pose"
            )
        )
        assert len(human_differences) == 2
        assert {
            difference.reference_role for difference in human_differences
        } == {"replay_pose", "composed_output"}
        assert station.data_plane.binding.source_difference_spec_hash == (
            observation.source_difference_spec_hash
        )
        facts = tuple(
            fact
            for fact in runtime.trace_facts
            if fact.event_type == "source_difference_observation"
        )
        assert len(facts) == 1
        spec_facts = tuple(
            fact
            for fact in runtime.trace_facts
            if fact.event_type == "source_difference_spec"
        )
        assert len(spec_facts) == 1
        assert spec_facts[0].payload["semantic_content_hash"] == (
            observation.source_difference_spec_hash
        )
        assert facts[0].fact_id == observation.observation_id
        assert facts[0].payload["composition_input_snapshot_id"] == (
            observation.composition_input_snapshot_id
        )
        assert facts[0].payload["composition_result_id"] == (
            observation.composition_result_id
        )
        duplicate = station.compose_at(
            decision_key="cs6b1i-difference-0",
            target_effective_at=clock(),
        )
        assert duplicate is composed
        assert sum(
            fact.event_type == "source_difference_observation"
            for fact in runtime.trace_facts
        ) == 1
        assert composed.proposal_admission is not None
        assert composed.proposal_admission.accepted
        dispatched = station.step(
            now=clock(),
            physics_ticks_after_dispatch=1,
        )
        assert dispatched.status == "dispatched"
    finally:
        passive_source.close()
        station.close()


def test_cs6b1i_missing_contributor_holds_without_fresh_input_rearm(
    tmp_path: Path,
) -> None:
    clock = ManualClock(30.0)
    station, runtime, passive_source, session_spec = (
        _build_cs6b1i_pose_composition_session(tmp_path, clock)
    )
    try:
        passive_source.start()
        passive_source.set_calibration_ready()
        _setup_and_start_cs6b1i_station(station, session_spec, clock)
        station.submit_replay_emission(
            _replay_emission(tmp_path / "cs6b1i-replay.parquet", clock)
        )
        held = station.compose_at(
            decision_key="cs6b1i-missing-human",
            target_effective_at=clock(),
        )
        assert held.disposition is not None
        assert held.disposition.action == "hold"
        assert held.evaluation is not None
        activation = station.data_plane.composition_activation_scope
        assert activation is not None
        assert (
            held.evaluation.input_snapshot.expected_activation_id
            == activation.activation_id
        )
        assert (
            held.evaluation.composition_result.composition_activation_id
            == activation.activation_id
        )
        activation_binding_kinds = {
            fact.payload["subject_kind"]
            for fact in runtime.trace_facts
            if fact.event_type == "composition_activation_binding"
        }
        assert activation_binding_kinds == {"input", "snapshot", "result"}
        assert station.current_session_state is not None
        assert station.current_session_state.phase == "held"
        assert not station.publication_barrier.snapshot().enabled
        human_step = passive_source.step(now=clock())
        assert human_step.emission is not None
        with pytest.raises(RuntimeError, match="not active"):
            station.submit_human_emission(
                human_step.emission,
                observed_at=clock(),
            )
        assert len(station.data_plane.runtime.core.proposals) == 0
    finally:
        passive_source.close()
        station.close()


def test_cs6b1i_typed_rearm_creates_new_activation_and_rejects_old_inputs(
    tmp_path: Path,
) -> None:
    clock = ManualClock(30.0)
    station, runtime, passive_source, session_spec = (
        _build_cs6b1i_pose_composition_session(tmp_path, clock)
    )
    observed_specs = []
    observed_rearm_specs = []
    observed_rearm_receipts = []
    observed_transitions = []
    station.bind_session_trace_observers(
        spec_observer=observed_specs.append,
        transition_observer=lambda transition, **_metadata: (
            observed_transitions.append(transition)
        ),
        rearm_spec_observer=observed_rearm_specs.append,
        rearm_receipt_observer=observed_rearm_receipts.append,
    )
    try:
        passive_source.start()
        passive_source.set_calibration_ready()
        sequence = _setup_and_start_cs6b1i_station(
            station,
            session_spec,
            clock,
        )
        initial_activation = station.data_plane.composition_activation_scope
        assert initial_activation is not None
        assert initial_activation.session_epoch == 0
        assert (
            initial_activation.freshness_policy
            == ACTIVATION_LOCAL_INPUT_FRESHNESS_POLICY_V2
        )
        station.submit_replay_emission(
            _replay_emission(
                tmp_path / "cs6b1i-before-rearm.parquet",
                clock,
                source_action_index=4,
            )
        )
        human_before = passive_source.step(now=clock())
        assert human_before.emission is not None
        station.submit_human_emission(
            human_before.emission,
            observed_at=clock(),
        )
        old_composition = station.compose_at(
            decision_key="cs6b1i-before-rearm",
            target_effective_at=clock(),
        )
        assert old_composition.proposal_admission is not None
        old_proposal = old_composition.proposal_admission.proposal
        assert old_proposal is not None
        assert old_proposal.producer_epoch == runtime.config.producer_epoch

        state = station.station.state
        hold = station.station.submit_intent(
            StationIntentDraft(
                client_id="operator:cs6b1i",
                client_sequence=sequence,
                intent_kind="hold",
                produced_at=clock(),
                produced_clock_domain_id="local-monotonic",
                expected_state_revision=state.state_revision,
                manifest_hash=station.station.manifest.semantic_content_hash,
                reason="pause before typed activation rearm",
            )
        )
        sequence += 1
        assert hold.receipt.accepted
        station.prepare_pending_operation(hold)
        held_state = station.current_session_state
        assert held_state is not None and held_state.phase == "held"
        hold_outcome = station._last_hold_outcome
        assert hold_outcome is not None
        assert hold_outcome.cancellation_receipt is not None
        assert hold_outcome.cancellation_receipt.complete

        rearm_spec = CompositionSessionRearmSpec(
            rearm_key="cs6b1i-rearm-1",
            logical_operation_spec_id=session_spec.operation_spec_id,
            logical_operation_spec_hash=session_spec.semantic_content_hash,
            prior_session_epoch=held_state.session_epoch,
            prior_state_hash=held_state.semantic_content_hash,
            execution_binding_hash=session_spec.execution_binding_hash,
            readiness_observer_binding_id=(
                station.rearm_readiness_observer_binding_id
            ),
            readiness_timeout_sec=0.1,
        )
        unsupported_rearm_spec = CompositionSessionRearmSpec(
            rearm_key="cs6b1i-rearm-unsupported-real-check",
            logical_operation_spec_id=session_spec.operation_spec_id,
            logical_operation_spec_hash=session_spec.semantic_content_hash,
            prior_session_epoch=held_state.session_epoch,
            prior_state_hash=held_state.semantic_content_hash,
            execution_binding_hash=session_spec.execution_binding_hash,
            readiness_observer_binding_id=(
                station.rearm_readiness_observer_binding_id
            ),
            readiness_timeout_sec=0.1,
            required_readiness_checks=(
                *rearm_spec.required_readiness_checks,
                "physical_deadman_rearmed",
            ),
        )
        with pytest.raises(ValueError, match="unsupported readiness"):
            station.register_composition_rearm(unsupported_rearm_spec)
        station.register_composition_rearm(rearm_spec)
        state = station.station.state
        rearm = station.station.submit_intent(
            StationIntentDraft(
                client_id="operator:cs6b1i",
                client_sequence=sequence,
                intent_kind="rearm_composition_session",
                produced_at=clock(),
                produced_clock_domain_id="local-monotonic",
                expected_state_revision=state.state_revision,
                manifest_hash=station.station.manifest.semantic_content_hash,
                subject_id=station.composer_source_slot_id,
                resource_binding_id=(
                    station.station.manifest.execution_binding.binding_id
                ),
                operation_spec_id=rearm_spec.rearm_spec_id,
                reason="typed activation rearm",
            )
        )
        assert rearm.receipt.status == "operation_requested"
        sequence += 1
        rearmed = station.complete_pending_operation(rearm)
        assert rearmed.completion_transition.accepted
        assert rearmed.completion_transition.result.succeeded
        active_state = station.current_session_state
        assert active_state is not None and active_state.phase == "active"
        assert active_state.operation_spec_id == session_spec.operation_spec_id
        assert active_state.session_epoch == held_state.session_epoch + 1
        rearmed_activation = station.data_plane.composition_activation_scope
        assert rearmed_activation is not None
        assert rearmed_activation.session_epoch == active_state.session_epoch
        assert (
            rearmed_activation.freshness_policy
            == ACTIVATION_LOCAL_STRICT_SOURCE_FRESHNESS_POLICY_V2
        )
        assert rearmed_activation.activation_id != initial_activation.activation_id
        assert (
            rearmed_activation.operation_spec_id
            == initial_activation.operation_spec_id
        )
        assert station.station.state.lifecycle_state == "running"
        assert station.publication_barrier.snapshot().enabled
        receipt = station._rearm_receipt_by_spec_id[rearm_spec.rearm_spec_id]
        assert receipt.execution_epoch == (
            receipt.prior_execution_epoch + 1
        )
        assert receipt.composer_producer_epoch == (
            receipt.prior_composer_producer_epoch + 1
        )
        assert receipt.authority_epoch > receipt.prior_authority_epoch
        assert station.data_plane.input_receipt_by_role == {}
        assert observed_specs == [session_spec]
        assert observed_rearm_specs == [rearm_spec]
        assert observed_rearm_receipts == [receipt]
        assert [
            (transition.session_epoch, transition.transition)
            for transition in observed_transitions
        ] == [
            (held_state.session_epoch, "start"),
            (held_state.session_epoch, "hold"),
            (held_state.session_epoch, "supersede"),
            (active_state.session_epoch, "start"),
        ]
        stale_receipt = station.submit_replay_emission(
            _replay_emission(
                tmp_path / "cs6b1i-stale-after-rearm.parquet",
                clock,
                source_action_index=4,
            )
        )
        assert not stale_receipt.accepted
        assert (
            stale_receipt.rejection_code
            == "activation_source_barrier_violation"
        )
        assert station.data_plane.input_receipt_by_role == {}

        stale_only = station.step(now=clock())
        assert stale_only.status != "dispatched"

        # The authoritative scheduler is 30 Hz and never catch-up dispatches.
        # Cross the next decision boundary before admitting the new activation.
        clock.value += 0.04
        station.submit_replay_emission(
            _replay_emission(
                tmp_path / "cs6b1i-after-rearm.parquet",
                clock,
                source_action_index=5,
                group_key="pose-group-1",
                local_frame_index=1,
            )
        )
        human_after = passive_source.step(now=clock())
        assert human_after.emission is not None
        station.submit_human_emission(
            human_after.emission,
            observed_at=clock(),
        )
        fresh_composition = station.compose_at(
            decision_key="cs6b1i-after-rearm",
            target_effective_at=clock(),
        )
        assert fresh_composition.proposal_admission is not None
        fresh_proposal = fresh_composition.proposal_admission.proposal
        assert fresh_proposal is not None
        assert (
            fresh_proposal.producer_epoch
            == receipt.composer_producer_epoch
        )

        state = station.station.state
        second_hold = station.station.submit_intent(
            StationIntentDraft(
                client_id="operator:cs6b1i",
                client_sequence=sequence,
                intent_kind="hold",
                produced_at=clock(),
                produced_clock_domain_id="local-monotonic",
                expected_state_revision=state.state_revision,
                manifest_hash=station.station.manifest.semantic_content_hash,
                reason="second pause before typed activation rearm",
            )
        )
        sequence += 1
        assert second_hold.receipt.accepted
        station.prepare_pending_operation(second_hold)
        second_held_state = station.current_session_state
        assert second_held_state is not None
        assert second_held_state.phase == "held"
        assert second_held_state.session_epoch == active_state.session_epoch

        second_rearm_spec = CompositionSessionRearmSpec(
            rearm_key="cs6b1i-rearm-2",
            logical_operation_spec_id=session_spec.operation_spec_id,
            logical_operation_spec_hash=session_spec.semantic_content_hash,
            prior_session_epoch=second_held_state.session_epoch,
            prior_state_hash=second_held_state.semantic_content_hash,
            execution_binding_hash=session_spec.execution_binding_hash,
            readiness_observer_binding_id=(
                station.rearm_readiness_observer_binding_id
            ),
            readiness_timeout_sec=0.1,
        )
        station.register_composition_rearm(second_rearm_spec)
        state = station.station.state
        second_request = station.station.submit_intent(
            StationIntentDraft(
                client_id="operator:cs6b1i",
                client_sequence=sequence,
                intent_kind="rearm_composition_session",
                produced_at=clock(),
                produced_clock_domain_id="local-monotonic",
                expected_state_revision=state.state_revision,
                manifest_hash=station.station.manifest.semantic_content_hash,
                subject_id=station.composer_source_slot_id,
                resource_binding_id=(
                    station.station.manifest.execution_binding.binding_id
                ),
                operation_spec_id=second_rearm_spec.rearm_spec_id,
                reason="second typed activation rearm",
            )
        )
        assert second_request.receipt.status == "operation_requested"
        second_rearmed = station.complete_pending_operation(second_request)
        assert second_rearmed.completion_transition.accepted
        assert second_rearmed.completion_transition.result.succeeded
        second_active_state = station.current_session_state
        assert second_active_state is not None
        assert second_active_state.phase == "active"
        assert second_active_state.operation_spec_id == (
            session_spec.operation_spec_id
        )
        assert second_active_state.session_epoch == active_state.session_epoch + 1
        second_receipt = station._rearm_receipt_by_spec_id[
            second_rearm_spec.rearm_spec_id
        ]
        assert second_receipt.prior_composer_producer_epoch == (
            receipt.composer_producer_epoch
        )
        assert second_receipt.composer_producer_epoch == (
            receipt.composer_producer_epoch + 1
        )
        assert second_receipt.prior_execution_epoch == receipt.execution_epoch
        assert second_receipt.execution_epoch == receipt.execution_epoch + 1
        assert second_receipt.prior_authority_epoch == receipt.authority_epoch
        assert second_receipt.authority_epoch > receipt.authority_epoch
        assert station.data_plane.input_receipt_by_role == {}
        assert [transition.transition for transition in observed_transitions] == [
            "start",
            "hold",
            "supersede",
            "start",
            "hold",
            "supersede",
            "start",
        ]
    finally:
        passive_source.close()
        station.close()


def test_cs6b1i_rearm_without_prior_cancellation_receipt_stays_held(
    tmp_path: Path,
) -> None:
    clock = ManualClock(30.0)
    station, runtime, passive_source, session_spec = (
        _build_cs6b1i_pose_composition_session(tmp_path, clock)
    )
    try:
        passive_source.start()
        passive_source.set_calibration_ready()
        sequence = _setup_and_start_cs6b1i_station(
            station,
            session_spec,
            clock,
        )
        state = station.station.state
        hold = station.station.submit_intent(
            StationIntentDraft(
                client_id="operator:cs6b1i",
                client_sequence=sequence,
                intent_kind="hold",
                produced_at=clock(),
                produced_clock_domain_id="local-monotonic",
                expected_state_revision=state.state_revision,
                manifest_hash=station.station.manifest.semantic_content_hash,
                reason="negative rearm fixture",
            )
        )
        sequence += 1
        station.prepare_pending_operation(hold)
        held_state = station.current_session_state
        assert held_state is not None and held_state.phase == "held"
        producer_epoch_before = runtime.core.producer_epochs.current_epoch(
            runtime.config.composer_producer_id
        )
        station._last_hold_outcome = None
        rearm_spec = CompositionSessionRearmSpec(
            rearm_key="cs6b1i-rearm-missing-cancellation",
            logical_operation_spec_id=session_spec.operation_spec_id,
            logical_operation_spec_hash=session_spec.semantic_content_hash,
            prior_session_epoch=held_state.session_epoch,
            prior_state_hash=held_state.semantic_content_hash,
            execution_binding_hash=session_spec.execution_binding_hash,
            readiness_observer_binding_id=(
                station.rearm_readiness_observer_binding_id
            ),
            readiness_timeout_sec=0.1,
        )
        station.register_composition_rearm(rearm_spec)
        state = station.station.state
        request = station.station.submit_intent(
            StationIntentDraft(
                client_id="operator:cs6b1i",
                client_sequence=sequence,
                intent_kind="rearm_composition_session",
                produced_at=clock(),
                produced_clock_domain_id="local-monotonic",
                expected_state_revision=state.state_revision,
                manifest_hash=station.station.manifest.semantic_content_hash,
                subject_id=station.composer_source_slot_id,
                resource_binding_id=(
                    station.station.manifest.execution_binding.binding_id
                ),
                operation_spec_id=rearm_spec.rearm_spec_id,
                reason="must fail without exact cancellation receipt",
            )
        )
        assert request.receipt.status == "operation_requested"
        failed = station.complete_pending_operation(request)
        assert failed.completion_transition.accepted
        assert not failed.completion_transition.result.succeeded
        assert failed.completion_transition.result.status == (
            "composition_session_held"
        )
        assert station.station.state.lifecycle_state == "hold"
        assert station.current_session_state == held_state
        assert not station.publication_barrier.snapshot().enabled
        assert runtime.core.producer_epochs.current_epoch(
            runtime.config.composer_producer_id
        ) == producer_epoch_before
        assert station._last_hold_outcome is not None
        assert station._last_hold_outcome.cancellation_receipt is not None
        assert station._last_hold_outcome.cancellation_receipt.complete
    finally:
        passive_source.close()
        station.close()


def test_cs6b1i_station_hold_closes_publication_before_return(
    tmp_path: Path,
) -> None:
    clock = ManualClock(30.0)
    station, _, passive_source, session_spec = (
        _build_cs6b1i_pose_composition_session(tmp_path, clock)
    )
    coordinator = None
    try:
        passive_source.start()
        passive_source.set_calibration_ready()
        sequence = _setup_and_start_cs6b1i_station(
            station,
            session_spec,
            clock,
        )
        coordinator = SideEffectStationOperationCoordinator(
            coordinator_id="cs6b1i-hold-coordinator",
            runtime=station.station,
            adapters=station.adapters,
            operation_prepare=station.prepare_pending_operation,
            operation_executor=station.complete_pending_operation,
            queue_capacity=2,
            clock=clock,
            clock_domain_id="local_monotonic",
        )
        coordinator.start()
        state = station.station.state
        submission = coordinator.submit_intent(
            StationIntentDraft(
                client_id="operator:cs6b1i",
                client_sequence=sequence,
                intent_kind="hold",
                produced_at=clock(),
                produced_clock_domain_id="local_monotonic",
                expected_state_revision=state.state_revision,
                manifest_hash=station.station.manifest.semantic_content_hash,
                reason="operator pause",
            )
        )
        assert submission.transition.receipt.accepted
        assert submission.transition.state_after.lifecycle_state == "hold"
        assert not station.publication_barrier.snapshot().enabled
        assert station.current_session_state is not None
        assert station.current_session_state.phase == "held"
    finally:
        if coordinator is not None:
            assert coordinator.stop(timeout_sec=2.0)
        passive_source.close()
        station.close()


def test_cs6b1i_formal_composition_session_capture_seals(
    tmp_path: Path,
) -> None:
    clock = ManualClock(30.0)
    profile = trace_requirement_profile(
        "deployable_sim_scheduled_pose_composer_station_v1"
    )
    trace_session_id = "cs6b1i-formal-trace"
    episode_uuid = "cs6b1i-formal-episode"
    trace_runtime = ControlTraceRuntime(
        RuntimeTraceSession(
            trace_session_id=trace_session_id,
            runtime_session_id="cs6b1d-pose-runtime",
            control_domain_id="cs6b1d-pose-domain",
            profile_id=profile.profile_id,
            started_at=clock(),
            clock_domain_id="local_monotonic",
        ),
        profile=profile,
        capacity=16384,
    )
    session_root = (
        tmp_path / CONTROL_TRACE_SESSION_RELATIVE_ROOT_V1 / trace_session_id
    )
    session_root.mkdir(parents=True)
    spool = ControlTraceSpool(
        session_root / "control_trace.wal",
        trace_session_id=trace_session_id,
        max_bytes=32 << 20,
        physical_preallocation=False,
    )
    health = ControlTraceWriterHealthLedger(
        session_root / "writer_health.ledger",
        trace_session_id=trace_session_id,
    )
    durable_sink = ControlTraceSpoolSink(
        sink_id="cs6b1i-formal-spool",
        runtime=trace_runtime,
        spool=spool,
        health_ledger=health,
        clock=clock,
    )
    dependency_sequences: list[int] = []
    recorder = EpisodeTraceRecorderSink(
        trace_runtime,
        lifecycle_owner_id="cs6b1i-formal-owner",
        flush_to_durable=durable_sink.flush_pending,
        dependency_sequences=lambda _runtime: tuple(dependency_sequences),
        manifest_store=EpisodeCaptureManifestStore(tmp_path),
        uuid_factory=lambda: episode_uuid,
        control_clock=clock,
        commit_clock=clock,
    )

    def admit_trace(fact):
        receipt = trace_runtime.issue(fact)
        if not receipt.accepted:
            raise RuntimeError(
                f"CS6b1i trace rejected {fact.event_type}: "
                f"{receipt.rejection_code}"
            )
        return receipt

    station, runtime, passive_source, session_spec = (
        _build_cs6b1i_pose_composition_session(
            tmp_path,
            clock,
            trace_sink=admit_trace,
            source_differences=True,
        )
    )
    execution_manifest = (
        build_cs6b1i_pose_composition_session_execution_manifest(
            station.data_plane,
            station.station.manifest,
        )
    )
    for dependency_fact in (
        project_control_execution_manifest(
            runtime.core.trace_context,
            execution_manifest,
            occurred_at=clock(),
        ),
        project_station_run_manifest(
            station.station.manifest,
            runtime_session_id=runtime.config.runtime_session_id,
            occurred_at=clock(),
            clock_domain_id="local_monotonic",
        ),
    ):
        receipt = admit_trace(dependency_fact)
        assert receipt.sink_accepted_sequence is not None
        dependency_sequences.append(receipt.sink_accepted_sequence)
    bridge = StationControlTraceBridge(
        manifest=station.station.manifest,
        runtime_session_id=runtime.config.runtime_session_id,
        embodiment_instance_id=runtime.config.embodiment_instance_id,
        trace_sink=admit_trace,
        clock=clock,
        clock_domain_id="local_monotonic",
    )
    station.bind_publication_barrier_observer(
        bridge.record_publication_barrier_transition
    )
    station.bind_session_trace_observers(
        spec_observer=bridge.record_composition_session_spec,
        transition_observer=bridge.record_composition_session_transition,
        rearm_spec_observer=bridge.record_composition_session_rearm_spec,
        rearm_receipt_observer=(
            bridge.record_composition_session_rearm_receipt
        ),
    )
    coordinator = SideEffectStationOperationCoordinator(
        coordinator_id="cs6b1i-formal-coordinator",
        runtime=station.station,
        adapters=station.adapters,
        operation_prepare=station.prepare_pending_operation,
        operation_executor=station.complete_pending_operation,
        intent_transition_observer=bridge.record_intent_transition,
        operation_transition_observer=bridge.record_operation_transition,
        queue_capacity=2,
        clock=clock,
        clock_domain_id="local-monotonic",
    )

    sequence = 0

    def submit(kind, subject_id=None, *, spec_id=None):
        nonlocal sequence
        before = len(coordinator.completions)
        state = station.station.state
        submission = coordinator.submit_intent(
            StationIntentDraft(
                client_id="operator:cs6b1i-formal",
                client_sequence=sequence,
                intent_kind=kind,
                produced_at=clock(),
                produced_clock_domain_id="local-monotonic",
                expected_state_revision=state.state_revision,
                manifest_hash=station.station.manifest.semantic_content_hash,
                subject_id=subject_id,
                resource_binding_id=(
                    station.station.manifest.execution_binding.binding_id
                    if kind
                    in {
                        "rearm_composition_session",
                        "start_composition_session",
                        "stop_composition_session",
                    }
                    else None
                ),
                operation_spec_id=spec_id,
                reason="cs6b1i formal capture",
            )
        )
        sequence += 1
        if submission.transition.receipt.status == "operation_requested":
            assert coordinator.wait_until_idle(timeout_sec=2.0)
            completion = coordinator.completions[-1]
            assert len(coordinator.completions) == before + 1
            assert completion.completion_transition.accepted
            assert completion.completion_transition.result.succeeded
        return submission

    prepared = recorder.prepare_episode(legacy_episode_index=0, prepared_at=clock())
    recorder.bind_prepared_episode(prepared, bound_at=clock())
    coordinator.start()
    released = None
    try:
        passive_source.start()
        passive_source.set_calibration_ready()
        submit("acquire_control_lease")
        target_id = station.station.manifest.execution_binding.binding_id
        submit("connect_target", target_id)
        for source_id in (
            station.human_source_slot_id,
            station.replay_source_slot_id,
            station.composer_source_slot_id,
        ):
            submit("connect_source", source_id)
        submit("run_preflight", target_id)
        for source_id in (
            station.human_source_slot_id,
            station.replay_source_slot_id,
            station.composer_source_slot_id,
        ):
            submit("run_preflight", source_id)
        submit("request_candidate", station.composer_source_slot_id)
        station.register_composition_session(session_spec)
        submit(
            "start_composition_session",
            station.composer_source_slot_id,
            spec_id=session_spec.operation_spec_id,
        )
        station.submit_replay_emission(
            _replay_emission(tmp_path / "cs6b1i-replay.parquet", clock)
        )
        passive = passive_source.step(now=clock())
        assert passive.emission is not None
        station.submit_human_emission(passive.emission, observed_at=clock())
        formal = station.compose_at(
            decision_key="cs6b1i-formal-decision",
            target_effective_at=clock(),
        )
        assert formal.proposal_admission is not None
        dispatched = station.step(
            now=clock(),
            physics_ticks_after_dispatch=60,
        )
        assert dispatched.status == "dispatched"
        submit("hold")
        held_state = station.current_session_state
        assert held_state is not None and held_state.phase == "held"
        rearm_spec = CompositionSessionRearmSpec(
            rearm_key="cs6b1i-formal-rearm-1",
            logical_operation_spec_id=session_spec.operation_spec_id,
            logical_operation_spec_hash=session_spec.semantic_content_hash,
            prior_session_epoch=held_state.session_epoch,
            prior_state_hash=held_state.semantic_content_hash,
            execution_binding_hash=session_spec.execution_binding_hash,
            readiness_observer_binding_id=(
                station.rearm_readiness_observer_binding_id
            ),
            readiness_timeout_sec=0.1,
        )
        station.register_composition_rearm(rearm_spec)
        submit(
            "rearm_composition_session",
            station.composer_source_slot_id,
            spec_id=rearm_spec.rearm_spec_id,
        )
        active_state = station.current_session_state
        assert active_state is not None and active_state.phase == "active"
        assert active_state.session_epoch == held_state.session_epoch + 1

        # The authoritative scheduler is 30 Hz and never catch-up dispatches.
        # Cross the next decision boundary before the second formal decision.
        clock.value += 0.04
        station.submit_replay_emission(
            _replay_emission(
                tmp_path / "cs6b1i-replay-after-rearm.parquet",
                clock,
                source_action_index=5,
                group_key="pose-group-1",
                local_frame_index=1,
            )
        )
        passive_after_rearm = passive_source.step(now=clock())
        assert passive_after_rearm.emission is not None
        station.submit_human_emission(
            passive_after_rearm.emission,
            observed_at=clock(),
        )
        formal_after_rearm = station.compose_at(
            decision_key="cs6b1i-formal-decision-after-rearm",
            target_effective_at=clock(),
        )
        assert formal_after_rearm.proposal_admission is not None
        # The first dispatch already exercises the deterministic plant. Keep
        # the second check at the authority boundary: the fixture's manual
        # station clock is intentionally independent of the rig physics clock.
        dispatched_after_rearm = station.step(now=clock())
        assert dispatched_after_rearm.status == "dispatched"
        submit(
            "stop_composition_session",
            station.composer_source_slot_id,
            spec_id=session_spec.operation_spec_id,
        )
        terminal = runtime.step(now=clock())
        assert terminal.status in {"execution_hold_active", "not_due"}
        clock.value += 0.01
        released = recorder.release_episode(
            prepared,
            unbound_at=clock(),
            reason="cs6b1i_formal_finalize",
        )
        assert released.seal.trace_sealed, (
            released.seal.reason_codes,
            released.seal.missing_required_event_types,
            released.seal.lineage_issue_codes,
        )
        assert released.seal.capture_commit_eligible
        finalized = recorder.legacy_finalize_completed(
            released,
            completed_at=clock() + 0.1,
        )
        assert finalized.capture_commit_status == "committed"
        assert bridge.admission_failures == ()
        assert coordinator.transition_observer_failure_count == 0
    finally:
        assert coordinator.stop(timeout_sec=2.0)
        passive_source.close()
        station.close()
        durable_sink.flush_pending()
        health.close()
        spool.close()

    captured = load_episode_control_trace(tmp_path, episode_uuid)
    assert captured.committed
    event_types = tuple(
        record.fact.event_type for record in captured.episode_records
    )
    assert event_types.count("composition_session_spec") == 1
    assert event_types.count("composition_session_rearm_spec") == 1
    assert event_types.count("composition_session_rearm_receipt") == 1
    assert event_types.count("composition_session_transition") == 6
    assert event_types.count("composition_contributor_input") == 4
    assert event_types.count("composition_authority_disposition") == 2
    assert event_types.count("source_difference_spec") == 1
    assert event_types.count("source_difference_observation") == 2
    assert event_types.count("station_publication_barrier_transition") >= 4


def _cs6b1f_takeover_spec(station, provider, *, timeout_sec: float):
    manifest = station.station.manifest
    return HumanTakeoverOperationSpec(
        previous_source_slot_id="replay-slot",
        target_human_source_slot_id="human-slot",
        previous_source_action_contract_hash=(
            manifest.source_slot("replay-slot").source_action_contract_hash
        ),
        target_human_action_contract_hash=(
            manifest.source_slot("human-slot").source_action_contract_hash
        ),
        execution_binding_hash=manifest.execution_binding.semantic_content_hash,
        readiness_observer_binding_id=provider.observer_binding_id,
        boundary_policy=MixedControlBoundaryPolicy(
            policy_id="cs6b1f-corner-safe-v1",
            policy_suspension_mode="hold_and_release_authority",
            episode_boundary_motion_policy="hold_before_unbind",
        ),
        readiness_timeout_sec=timeout_sec,
        capture_boundary_id="episode-boundary:cs6b1f-corner",
    )


class SyntheticPoseReadinessProvider:
    observer_binding_id = "human-observer:synthetic-pose-cs6b1d"

    def __init__(self, source: SyntheticHumanPoseSource, clock: ManualClock) -> None:
        self.source = source
        self.clock = clock

    def acquire(self, *, operation_spec, takeover_state, timeout_sec):
        del operation_spec, timeout_sec
        assert takeover_state.takeover_receipt is not None
        self.clock.value = takeover_state.takeover_receipt.completed_at + 0.01
        emission = self.source.emit(
            {
                "left_arm": _pose_matrix(-0.08),
                "right_arm": _pose_matrix(0.08),
            },
            emitted_at=self.clock(),
        )
        return HumanTakeoverReadinessCandidate(
            observer_binding_id=self.observer_binding_id,
            emission=emission,
            effective_from=self.clock(),
            effective_until=self.clock() + 0.1,
            target_effective_at=self.clock(),
            deadman_rearm_event_id="virtual-deadman:cs6b1d",
            deadman_rearmed_at=self.clock(),
            retargeter_rebase_receipt_id="virtual-rebase:cs6b1d",
            retargeter_rebased_at=self.clock(),
        )


def test_cs6b1d_pose_replay_to_human_keeps_pose_until_target_resolution(
    tmp_path: Path,
) -> None:
    clock = ManualClock()
    runtime = _runtime(clock)
    human_spec = SyntheticHumanPoseSourceSpec(source_id="human-slot")
    human_source = SyntheticHumanPoseSource(human_spec)
    replay_source = _replay_source(runtime)
    path = tmp_path / "episode.parquet"
    observer = ReplaySourceObserverBinding.create(
        source_slot_id="replay-slot",
        source_locator_id="cs6b1d-pose-source",
        expected_source_path=str(path),
        source_episode_index=0,
        group_key="pose-group-0",
        source_start_index=4,
        clock_domain_id="local_monotonic",
    )
    station = build_scoped_cs6b1d_mujoco_pose_human_replay_station(
        runtime,
        human_spec,
        replay_source,
        observer,
        station_clock=clock,
        station_clock_domain_id="local-monotonic",
    )
    assert station.station.manifest.manifest_id == (
        CS6B1D_POSE_HUMAN_REPLAY_HANDOFF_MANIFEST_ID_V1
    )
    assert station.data_plane.binding_receipt.discrete_handoff_only
    _advance_to_replay(station, clock)

    replay = station.submit_replay_source_frame_emission(
        _replay_emission(path, clock)
    )
    assert replay.proposal_admission is not None
    assert replay.proposal_admission.accepted
    replay_step = station.step(now=clock(), physics_ticks_after_dispatch=2)
    assert replay_step.status == "dispatched"
    replay_mixing = station.data_plane.last_pose_mixing_shadow_evaluation
    assert replay_mixing is not None
    assert replay_mixing.status == "parity"
    assert replay_mixing.selection_parity
    assert replay_mixing.output_pose_parity
    assert replay_step.safety_result.safe_action.action_space == (
        "bimanual_ee_pose_v1"
    )
    assert all(
        command.command_type == "joint_vector"
        for command in replay_step.command.commands.values()
    )

    provider = SyntheticPoseReadinessProvider(human_source, clock)
    manifest = station.station.manifest
    spec = HumanTakeoverOperationSpec(
        previous_source_slot_id="replay-slot",
        target_human_source_slot_id="human-slot",
        previous_source_action_contract_hash=(
            manifest.source_slot("replay-slot").source_action_contract_hash
        ),
        target_human_action_contract_hash=(
            manifest.source_slot("human-slot").source_action_contract_hash
        ),
        execution_binding_hash=manifest.execution_binding.semantic_content_hash,
        readiness_observer_binding_id=provider.observer_binding_id,
        boundary_policy=MixedControlBoundaryPolicy(
            policy_id="cs6b1d-pose-safe-mixed-sim-v1",
            policy_suspension_mode="hold_and_release_authority",
            episode_boundary_motion_policy="hold_before_unbind",
        ),
        readiness_timeout_sec=0.2,
        capture_boundary_id="episode-boundary:cs6b1d",
    )
    spec_id = station.register_human_takeover_operation(spec, provider)
    clock.value += 0.2
    request, execution = _submit(
        station,
        clock,
        10,
        "request_human_takeover",
        "human-slot",
        resource_binding_id="replay-slot",
        operation_spec_id=spec_id,
    )
    assert request.receipt.status == "operation_requested"
    assert execution is not None
    assert execution.completion_transition.result.succeeded
    assert execution.completion_transition.result.status == "human_takeover_ready"

    assert not any(
        fact.event_type == "human_command_emission"
        for fact in runtime.trace_facts
    )
    promoted = station.promote_human_takeover_emission(spec_id)
    assert promoted.accepted
    assert promoted.adapter_result.robot_action.action_space == (
        "bimanual_ee_pose_v1"
    )
    human_step = station.step(now=clock(), physics_ticks_after_dispatch=2)
    assert human_step.status == "dispatched"
    human_mixing = station.data_plane.last_pose_mixing_shadow_evaluation
    assert human_mixing is not None
    assert human_mixing.status == "parity"
    assert human_mixing.selection_parity
    assert human_mixing.output_pose_parity
    assert human_step.safety_result.safe_action.action_space == (
        "bimanual_ee_pose_v1"
    )
    assert all(
        command.command_type == "joint_vector"
        for command in human_step.command.commands.values()
    )
    event_types = tuple(fact.event_type for fact in runtime.trace_facts)
    assert event_types.count("replay_source_frame_emission") == 1
    assert event_types.count("human_command_emission") == 1
    assert event_types.count("resolved_execution_target") == 2
    station.close()


@pytest.mark.parametrize(
    ("field_name", "replacement", "message"),
    (
        ("pose_frame_contract_hash", "another-pose-frame", "verified transform"),
        (
            "action_spec_hash_by_effector",
            (
                ("left_arm", "wrong-left-spec"),
                ("right_arm", "rm65-tool-tcp-pose-v1"),
            ),
            "action specs",
        ),
        ("units_convention_id", "millimeters-degrees", "units"),
    ),
)
def test_cs6b1d_pose_human_binding_rejects_contract_drift(
    field_name: str,
    replacement,
    message: str,
) -> None:
    clock = ManualClock()
    runtime = _runtime(clock)
    kwargs = {field_name: replacement}
    source_spec = SyntheticHumanPoseSourceSpec(**kwargs)

    with pytest.raises(ValueError, match=message):
        build_mujoco_pose_human_proposal_adapter(runtime, source_spec)
    runtime.close()


def test_cs6b1d_accepts_device_neutral_passive_pose_source_contract(
    tmp_path: Path,
) -> None:
    clock = ManualClock()
    runtime = _runtime(clock)
    source_spec = PassiveTeleopPoseSourceSpec(source_id="human-slot")
    replay_source = _replay_source(runtime)
    observer = ReplaySourceObserverBinding.create(
        source_slot_id="replay-slot",
        source_locator_id="cs6b1d-pose-source",
        expected_source_path=str(tmp_path / "episode.parquet"),
        source_episode_index=0,
        group_key="pose-group-0",
        source_start_index=4,
        clock_domain_id="local_monotonic",
    )

    station = build_scoped_cs6b1d_mujoco_pose_human_replay_station(
        runtime,
        source_spec,
        replay_source,
        observer,
        station_clock=clock,
        station_clock_domain_id="local-monotonic",
    )
    try:
        human_slot = station.station.manifest.source_slot("human-slot")
        assert human_slot.source_kind == "human"
        assert source_spec.semantic_content_hash in human_slot.required_binding_refs
        assert station.data_plane.pose_mixing_strategy is not None
    finally:
        station.close()


def test_single_arm_pose_contract_is_rejected_by_bimanual_target_profile() -> None:
    clock = ManualClock()
    runtime = _runtime(clock)
    source_spec = PassiveTeleopPoseSourceSpec.for_arm(
        "right_arm",
        reference_frame="right_arm_base",
        action_spec_hash="rm65-tool-tcp-pose-v1",
        source_id="single-arm-human-slot",
    )
    try:
        with pytest.raises(ValueError, match="exactly cover the target components"):
            build_mujoco_pose_human_proposal_adapter(runtime, source_spec)
    finally:
        runtime.close()


def test_passive_pose_session_can_observe_while_replay_remains_only_authority(
    tmp_path: Path,
) -> None:
    clock = ManualClock()
    runtime = _runtime(clock)
    source_spec = PassiveTeleopPoseSourceSpec(source_id="human-slot")
    passive_source = _passive_pose_session_source(source_spec)
    replay_source = _replay_source(runtime)
    path = tmp_path / "episode.parquet"
    observer = ReplaySourceObserverBinding.create(
        source_slot_id="replay-slot",
        source_locator_id="cs6b1d-pose-source",
        expected_source_path=str(path),
        source_episode_index=0,
        group_key="pose-group-0",
        source_start_index=4,
        clock_domain_id="local_monotonic",
    )
    station = build_scoped_cs6b1d_mujoco_pose_human_replay_station(
        runtime,
        source_spec,
        replay_source,
        observer,
        passive_human_pose_source=passive_source,
        station_clock=clock,
        station_clock_domain_id="local-monotonic",
    )
    target = station.station.manifest.execution_binding.binding_id
    operations_before_calibration = (
        ("acquire_control_lease", None),
        ("connect_target", target),
        ("connect_source", "replay-slot"),
        ("connect_source", "human-slot"),
    )
    try:
        for sequence, (kind, subject) in enumerate(operations_before_calibration):
            transition, execution = _submit(
                station,
                clock,
                sequence,
                kind,
                subject,
            )
            assert transition.receipt.accepted
            if execution is not None:
                assert execution.completion_transition.result.succeeded
        station.set_passive_human_calibration_ready()
        operations_after_calibration = (
            ("run_preflight", target),
            ("run_preflight", "replay-slot"),
            ("run_preflight", "human-slot"),
            ("request_candidate", "replay-slot"),
            ("request_motion_authority", "replay-slot"),
            ("request_candidate", "human-slot"),
        )
        for offset, (kind, subject) in enumerate(
            operations_after_calibration,
            start=len(operations_before_calibration),
        ):
            transition, execution = _submit(
                station,
                clock,
                offset,
                kind,
                subject,
            )
            assert transition.receipt.accepted
            if execution is not None:
                assert execution.completion_transition.result.succeeded

        passive_step = station.step_passive_human_pose_source(now=10.0)
        assert passive_step.emission_valid
        assert passive_step.emission is not None
        assert not passive_step.command_dispatched
        assert station.data_plane.active_source_slot_id == "replay-slot"
        assert not any(
            fact.event_type == "human_command_emission"
            for fact in runtime.trace_facts
        )

        replay = station.submit_replay_source_frame_emission(
            _replay_emission(path, clock)
        )
        assert replay.proposal_admission is not None
        assert replay.proposal_admission.accepted
        replay_step = station.step(now=clock(), physics_ticks_after_dispatch=1)
        assert replay_step.status == "dispatched"
        assert station.data_plane.active_source_slot_id == "replay-slot"
        assert runtime.core.dispatch_owners.snapshot().current_token is not None
    finally:
        station.close()


def test_cs6b1f_sim_hil_promotes_passive_pose_and_keeps_publish_seam_unique(
    tmp_path: Path,
) -> None:
    """Opt-in HIL path: passive polling is harmless until typed takeover READY."""

    # TeleopSession stamps HumanCommandEmission with time.monotonic(); keep
    # the virtual station clock in that explicit local-monotonic domain.
    clock = ManualClock(time.monotonic())
    runtime = _runtime(clock)
    source_spec = PassiveTeleopPoseSourceSpec(source_id="human-slot")
    passive_source = _passive_pose_session_source(
        source_spec,
        timestamp=clock(),
    )
    replay_source = _replay_source(runtime)
    path = tmp_path / "episode.parquet"
    observer = ReplaySourceObserverBinding.create(
        source_slot_id="replay-slot",
        source_locator_id="cs6b1d-pose-source",
        expected_source_path=str(path),
        source_episode_index=0,
        group_key="pose-group-0",
        source_start_index=4,
        clock_domain_id="local_monotonic",
    )
    station = build_scoped_cs6b1f_mujoco_pose_human_replay_station(
        runtime,
        source_spec,
        replay_source,
        observer,
        passive_human_pose_source=passive_source,
        pose_fusion_shadow_spec=_pose_fusion_shadow_spec(),
        station_clock=clock,
        station_clock_domain_id="local-monotonic",
    )
    assert station.station.manifest.manifest_id == (
        CS6B1F_POSE_HUMAN_REPLAY_STATION_MANIFEST_ID_V1
    )
    execution_manifest = build_cs6b1f_pose_human_replay_execution_manifest(
        runtime,
        station.station.manifest,
    )
    assert execution_manifest.action_space == "bimanual_ee_pose_v1"

    def submit(sequence: int, kind: str, subject: str | None = None, **bindings):
        return _submit(station, clock, sequence, kind, subject, **bindings)

    target = station.station.manifest.execution_binding.binding_id
    try:
        for sequence, (kind, subject) in enumerate(
            (
                ("acquire_control_lease", None),
                ("connect_target", target),
                ("connect_source", "replay-slot"),
                ("connect_source", "human-slot"),
            )
        ):
            transition, execution = submit(sequence, kind, subject)
            assert transition.receipt.accepted
            if execution is not None:
                assert execution.completion_transition.result.succeeded
        station.set_passive_human_calibration_ready()
        for sequence, (kind, subject) in enumerate(
            (
                ("run_preflight", target),
                ("run_preflight", "replay-slot"),
                ("run_preflight", "human-slot"),
                ("request_candidate", "replay-slot"),
                ("request_motion_authority", "replay-slot"),
                ("request_candidate", "human-slot"),
            ),
            start=4,
        ):
            transition, execution = submit(sequence, kind, subject)
            assert transition.receipt.accepted
            if execution is not None:
                assert execution.completion_transition.result.succeeded

        # The connected passive source remains a candidate while replay is the
        # sole authority; no human emission is admitted yet.
        replay = station.submit_replay_source_frame_emission(
            _replay_emission(path, clock)
        )
        assert replay.proposal_admission is not None
        assert replay.proposal_admission.accepted
        assert station.step(now=clock(), physics_ticks_after_dispatch=2).status == (
            "dispatched"
        )
        assert not any(
            fact.event_type == "human_command_emission"
            for fact in runtime.trace_facts
        )

        provider = VirtualSimHumanTakeoverReadinessProvider(
            passive_source,
            clock=clock,
            evidence_prefix="cs6b1f-test",
            effective_horizon_sec=10.0,
        )
        manifest = station.station.manifest
        spec = HumanTakeoverOperationSpec(
            previous_source_slot_id="replay-slot",
            target_human_source_slot_id="human-slot",
            previous_source_action_contract_hash=(
                manifest.source_slot("replay-slot").source_action_contract_hash
            ),
            target_human_action_contract_hash=(
                manifest.source_slot("human-slot").source_action_contract_hash
            ),
            execution_binding_hash=manifest.execution_binding.semantic_content_hash,
            readiness_observer_binding_id=provider.observer_binding_id,
            boundary_policy=MixedControlBoundaryPolicy(
                policy_id="cs6b1f-passive-hil-safe-v1",
                policy_suspension_mode="hold_and_release_authority",
                episode_boundary_motion_policy="hold_before_unbind",
            ),
            readiness_timeout_sec=0.2,
            capture_boundary_id="episode-boundary:cs6b1f",
        )
        spec_id = station.register_human_takeover_operation(spec, provider)
        state = station.station.state
        transition = station.station.submit_intent(
            StationIntentDraft(
                client_id="operator:cs6b1d",
                client_sequence=20,
                intent_kind="request_human_takeover",
                produced_at=clock(),
                produced_clock_domain_id="local-monotonic",
                expected_state_revision=state.state_revision,
                manifest_hash=station.station.manifest.semantic_content_hash,
                subject_id="human-slot",
                resource_binding_id="replay-slot",
                operation_spec_id=spec_id,
                reason="cs6b1f-passive-hil",
            )
        )
        assert transition.receipt.status == "operation_requested"
        result_box: dict[str, object] = {}

        def complete() -> None:
            result_box["execution"] = station.complete_pending_operation(transition)

        worker = threading.Thread(target=complete)
        worker.start()
        deadline = time.monotonic() + 2.0
        while (
            station.data_plane.safe_human_takeover_state is None
            or station.data_plane.safe_human_takeover_state.phase
            != "awaiting_human_readiness"
        ):
            if time.monotonic() >= deadline:
                raise AssertionError("HIL takeover did not reach readiness wait.")
            time.sleep(0.001)
        post_boundary = station.step_passive_human_pose_source(now=clock())
        assert post_boundary.emission_valid
        assert post_boundary.emission is not None
        clock.value = max(clock.value, post_boundary.emission.emitted_at)
        worker.join(timeout=2.0)
        assert not worker.is_alive()
        execution = result_box["execution"]
        assert execution.completion_transition.result.succeeded, (
            station.takeover_failure_reason_by_intent_id
        )
        assert execution.completion_transition.result.status == "human_takeover_ready"

        promoted = station.promote_human_takeover_emission(spec_id)
        assert promoted.accepted
        assert promoted.emission is post_boundary.emission
        assert promoted.emission.semantic_content_hash == (
            post_boundary.emission.semantic_content_hash
        )
        assert promoted.adapter_result.proposal_draft is not None
        # The deterministic rig advances its private physics timeline during
        # each tick.  Leave a small explicit margin so publication time cannot
        # fall behind that same-domain planner timeline.
        clock.value = max(
            clock.value,
            promoted.adapter_result.proposal_draft.target_effective_at,
        ) + 0.1
        human_step = station.step(now=clock(), physics_ticks_after_dispatch=60)
        assert human_step.status == "dispatched"

        clock.value += 0.2
        next_human = station.step_passive_human_pose_source(now=clock())
        assert next_human.emission_valid
        submitted = station.submit_passive_human_pose_step(
            next_human,
            effective_from=clock(),
            effective_until=clock() + 0.1,
            target_effective_at=clock(),
        )
        assert submitted.accepted
        assert sum(
            fact.event_type == "human_command_emission"
            for fact in runtime.trace_facts
        ) == 2
    finally:
        station.close()


@pytest.mark.parametrize("seed_pre_boundary_observation", (False, True))
def test_cs6b1f_virtual_readiness_timeout_stays_hold_without_promotion(
    tmp_path: Path,
    seed_pre_boundary_observation: bool,
) -> None:
    """No fresh passive sample (including a stale queued one) can open READY."""

    clock = ManualClock(time.monotonic())
    station, runtime, passive_source, _path = _build_cs6b1f_station_to_replay(
        tmp_path,
        clock,
    )
    try:
        if seed_pre_boundary_observation:
            # Put the station clock after the observer timestamp so this sample
            # is provably before the later takeover boundary.
            clock.value = max(clock.value, time.monotonic()) + 0.2
            stale_step = station.step_passive_human_pose_source(now=clock())
            assert stale_step.emission_valid
            assert passive_source.ingress.status().queue_size == 1

        provider = VirtualSimHumanTakeoverReadinessProvider(
            passive_source,
            clock=clock,
            evidence_prefix="cs6b1f-timeout",
            effective_horizon_sec=0.1,
        )
        spec_id = station.register_human_takeover_operation(
            _cs6b1f_takeover_spec(station, provider, timeout_sec=0.05),
            provider,
        )
        request, execution = _submit(
            station,
            clock,
            20,
            "request_human_takeover",
            "human-slot",
            resource_binding_id="replay-slot",
            operation_spec_id=spec_id,
        )
        assert request.receipt.status == "operation_requested"
        assert execution is not None
        result = execution.completion_transition
        assert result.accepted
        assert not result.result.succeeded
        assert result.result.status == "human_takeover_held"
        assert station.station.state.lifecycle_state == "hold"
        assert (
            station.data_plane.runtime.core.capture_execution_gate_snapshot().status
            == "hold"
        )
        assert not any(
            barrier.snapshot().enabled
            for barrier in station.publication_barrier_by_source.values()
        )
        assert not any(
            fact.event_type == "human_command_emission"
            for fact in runtime.trace_facts
        )
        assert "TimeoutError" in station.takeover_failure_reason_by_intent_id[
            request.intent.intent_id
        ]
        if seed_pre_boundary_observation:
            assert passive_source.ingress.status().stale_count >= 1
        with pytest.raises(RuntimeError, match="No READY"):
            station.promote_human_takeover_emission(spec_id)
    finally:
        station.close()


def test_cs6b1f_virtual_readiness_late_completion_cannot_reopen_after_hold(
    tmp_path: Path,
) -> None:
    """A sample arriving after operator HOLD cannot resurrect a late worker."""

    clock = ManualClock(time.monotonic())
    station, runtime, passive_source, _path = _build_cs6b1f_station_to_replay(
        tmp_path,
        clock,
    )
    provider = VirtualSimHumanTakeoverReadinessProvider(
        passive_source,
        clock=clock,
        evidence_prefix="cs6b1f-late",
        effective_horizon_sec=0.1,
    )
    spec_id = station.register_human_takeover_operation(
        _cs6b1f_takeover_spec(station, provider, timeout_sec=0.5),
        provider,
    )
    coordinator = SideEffectStationOperationCoordinator(
        coordinator_id="cs6b1f-late-worker",
        runtime=station.station,
        adapters=station.adapters,
        operation_prepare=station.prepare_pending_operation,
        operation_executor=station.complete_pending_operation,
        queue_capacity=2,
        clock=clock,
        clock_domain_id="local-monotonic",
    )
    coordinator.start()
    try:
        state = station.station.state
        takeover = coordinator.submit_intent(
            StationIntentDraft(
                client_id="operator:cs6b1d",
                client_sequence=20,
                intent_kind="request_human_takeover",
                produced_at=clock(),
                produced_clock_domain_id="local-monotonic",
                expected_state_revision=state.state_revision,
                manifest_hash=station.station.manifest.semantic_content_hash,
                subject_id="human-slot",
                resource_binding_id="replay-slot",
                operation_spec_id=spec_id,
                reason="cs6b1f-late-readiness",
            )
        )
        assert takeover.enqueue_receipt.accepted
        deadline = time.monotonic() + 2.0
        while (
            station.data_plane.safe_human_takeover_state is None
            or station.data_plane.safe_human_takeover_state.phase
            != "awaiting_human_readiness"
        ):
            if time.monotonic() >= deadline:
                raise AssertionError("Virtual HIL worker did not reach readiness wait.")
            time.sleep(0.001)

        safety_state = station.station.state
        safety = coordinator.submit_intent(
            StationIntentDraft(
                client_id="safety:cs6b1f-late",
                client_sequence=0,
                intent_kind="hold",
                produced_at=clock(),
                produced_clock_domain_id="local-monotonic",
                expected_state_revision=safety_state.state_revision,
                manifest_hash=station.station.manifest.semantic_content_hash,
                reason="operator-hold-during-virtual-readiness",
            )
        )
        assert safety.transition.receipt.accepted
        assert station.station.state.lifecycle_state == "hold"

        # This is a fresh post-takeover sample for the lower provider, but it
        # arrives after the Station safety latch has superseded the request.
        clock.value = max(clock.value, time.monotonic()) + 0.01
        late_step = station.step_passive_human_pose_source(now=clock())
        assert late_step.emission_valid
        assert coordinator.wait_until_idle(timeout_sec=2.0)
        completion = next(
            item
            for item in coordinator.completions
            if item.request_transition.intent.intent_id
            == takeover.transition.intent.intent_id
        )
        assert not completion.completion_transition.accepted
        assert (
            completion.completion_transition.status
            == "human_takeover_result_after_latched_safety_state"
        )
        assert station.data_plane.safe_human_takeover_state is not None
        assert station.data_plane.safe_human_takeover_state.phase == "failed"
        assert (
            station.data_plane.safe_human_takeover_state.failure_reason
            == "station_pending_operation_superseded"
        )
        assert not any(
            barrier.snapshot().enabled
            for barrier in station.publication_barrier_by_source.values()
        )
        assert not any(
            fact.event_type == "human_command_emission"
            for fact in runtime.trace_facts
        )
    finally:
        assert coordinator.stop(timeout_sec=2.0)
        station.close()


def test_cs6b1f_formal_hil_capture_seals_and_rebuilds_pose_lineage(
    tmp_path: Path,
) -> None:
    """Formal opt-in HIL capture uses the existing Station/WAL/episode seams."""

    clock = ManualClock(time.monotonic())
    profile = trace_requirement_profile(
        "deployable_sim_scheduled_pose_human_replay_station_v1"
    )
    trace_session_id = "cs6b1f-formal-trace"
    episode_uuid = "cs6b1f-formal-episode"
    trace_runtime = ControlTraceRuntime(
        RuntimeTraceSession(
            trace_session_id=trace_session_id,
            runtime_session_id="cs6b1d-pose-runtime",
            control_domain_id="cs6b1d-pose-domain",
            profile_id=profile.profile_id,
            started_at=clock(),
            clock_domain_id="local_monotonic",
        ),
        profile=profile,
        capacity=16384,
    )
    session_root = (
        tmp_path / CONTROL_TRACE_SESSION_RELATIVE_ROOT_V1 / trace_session_id
    )
    session_root.mkdir(parents=True)
    spool = ControlTraceSpool(
        session_root / "control_trace.wal",
        trace_session_id=trace_session_id,
        max_bytes=32 << 20,
        physical_preallocation=False,
    )
    health = ControlTraceWriterHealthLedger(
        session_root / "writer_health.ledger",
        trace_session_id=trace_session_id,
    )
    durable_sink = ControlTraceSpoolSink(
        sink_id="cs6b1f-formal-hil-spool",
        runtime=trace_runtime,
        spool=spool,
        health_ledger=health,
        clock=clock,
    )
    dependency_sequences: list[int] = []
    recorder = EpisodeTraceRecorderSink(
        trace_runtime,
        lifecycle_owner_id="cs6b1f-formal-hil-owner",
        flush_to_durable=durable_sink.flush_pending,
        dependency_sequences=lambda _runtime: tuple(dependency_sequences),
        manifest_store=EpisodeCaptureManifestStore(tmp_path),
        uuid_factory=lambda: episode_uuid,
        control_clock=clock,
        commit_clock=clock,
    )

    def admit_trace(fact):
        receipt = trace_runtime.issue(fact)
        if not receipt.accepted:
            raise RuntimeError(
                f"CS6b1f trace rejected {fact.event_type}: "
                f"{receipt.rejection_code}"
            )
        return receipt

    runtime = _runtime(clock, trace_sink=admit_trace)
    source_spec = PassiveTeleopPoseSourceSpec(source_id="human-slot")
    passive_emission_timestamp = clock.value + 0.02
    passive_source = _passive_pose_session_source(
        source_spec,
        timestamp=passive_emission_timestamp,
    )
    replay_source = _replay_source(runtime)
    path = tmp_path / "formal_episode.parquet"
    observer = ReplaySourceObserverBinding.create(
        source_slot_id="replay-slot",
        source_locator_id="cs6b1d-pose-source",
        expected_source_path=str(path),
        source_episode_index=0,
        group_key="pose-group-0",
        source_start_index=4,
        clock_domain_id="local_monotonic",
    )
    station = build_scoped_cs6b1f_mujoco_pose_human_replay_station(
        runtime,
        source_spec,
        replay_source,
        observer,
        passive_human_pose_source=passive_source,
        station_clock=clock,
        station_clock_domain_id="local_monotonic",
    )
    execution_manifest = build_cs6b1f_pose_human_replay_execution_manifest(
        runtime,
        station.station.manifest,
    )
    for dependency_fact in (
        project_control_execution_manifest(
            runtime.core.trace_context,
            execution_manifest,
            occurred_at=clock(),
        ),
        project_station_run_manifest(
            station.station.manifest,
            runtime_session_id=runtime.config.runtime_session_id,
            occurred_at=clock(),
            clock_domain_id="local_monotonic",
        ),
    ):
        receipt = admit_trace(dependency_fact)
        assert receipt.sink_accepted_sequence is not None
        dependency_sequences.append(receipt.sink_accepted_sequence)

    station_trace_bridge = StationControlTraceBridge(
        manifest=station.station.manifest,
        runtime_session_id=runtime.config.runtime_session_id,
        embodiment_instance_id=runtime.config.embodiment_instance_id,
        trace_sink=admit_trace,
        clock=clock,
        clock_domain_id="local_monotonic",
    )
    station.bind_publication_barrier_observer(
        station_trace_bridge.record_publication_barrier_transition
    )
    coordinator = SideEffectStationOperationCoordinator(
        coordinator_id="cs6b1f-formal-hil-coordinator",
        runtime=station.station,
        adapters=station.adapters,
        operation_prepare=station.prepare_pending_operation,
        operation_executor=station.complete_pending_operation,
        intent_transition_observer=station_trace_bridge.record_intent_transition,
        operation_transition_observer=(
            station_trace_bridge.record_operation_transition
        ),
        queue_capacity=2,
        clock=clock,
        clock_domain_id="local_monotonic",
    )

    def submit_station_operation(
        *,
        sequence: int,
        kind: str,
        subject_id: str | None,
        resource_binding_id: str | None = None,
        operation_spec_id: str | None = None,
    ):
        before_count = len(coordinator.completions)
        state = station.station.state
        submission = coordinator.submit_intent(
            StationIntentDraft(
                client_id="operator:cs6b1f-formal-hil",
                client_sequence=sequence,
                intent_kind=kind,
                produced_at=clock(),
                produced_clock_domain_id="local_monotonic",
                expected_state_revision=state.state_revision,
                manifest_hash=station.station.manifest.semantic_content_hash,
                subject_id=subject_id,
                resource_binding_id=resource_binding_id,
                operation_spec_id=operation_spec_id,
                reason="cs6b1f formal sim HIL capture",
            )
        )
        if submission.transition.receipt.status != "operation_requested":
            return submission, None
        assert coordinator.wait_until_idle(timeout_sec=2.0)
        completions = coordinator.completions
        assert len(completions) == before_count + 1
        completion = completions[-1]
        assert (
            completion.request_transition.intent.intent_id
            == submission.transition.intent.intent_id
        )
        return submission, completion

    prepared = recorder.prepare_episode(legacy_episode_index=0, prepared_at=clock())
    recorder.bind_prepared_episode(prepared, bound_at=clock())
    coordinator.start()
    step = None
    try:
        target_id = station.station.manifest.execution_binding.binding_id
        setup_operations = (
            ("acquire_control_lease", None),
            ("connect_target", target_id),
            ("connect_source", "replay-slot"),
            ("connect_source", "human-slot"),
        )
        for sequence, (kind, subject_id) in enumerate(setup_operations):
            _submission, completion = submit_station_operation(
                sequence=sequence,
                kind=kind,
                subject_id=subject_id,
            )
            if completion is not None:
                assert completion.completion_transition.result.succeeded
        station.set_passive_human_calibration_ready()
        runtime_operations = (
            ("run_preflight", target_id),
            ("run_preflight", "replay-slot"),
            ("run_preflight", "human-slot"),
            ("request_candidate", "replay-slot"),
            ("request_motion_authority", "replay-slot"),
            ("request_candidate", "human-slot"),
        )
        for sequence, (kind, subject_id) in enumerate(
            runtime_operations,
            start=len(setup_operations),
        ):
            _submission, completion = submit_station_operation(
                sequence=sequence,
                kind=kind,
                subject_id=subject_id,
            )
            if completion is not None:
                assert completion.completion_transition.result.succeeded

        replay_submission = station.submit_replay_source_frame_emission(
            _replay_emission(path, clock)
        )
        assert replay_submission.proposal_admission is not None
        assert replay_submission.proposal_admission.accepted
        step = station.step(now=clock(), physics_ticks_after_dispatch=60)
        assert step.status == "dispatched"

        provider = VirtualSimHumanTakeoverReadinessProvider(
            passive_source,
            clock=clock,
            evidence_prefix="cs6b1f-formal-hil",
            effective_horizon_sec=10.0,
        )
        manifest = station.station.manifest
        spec = HumanTakeoverOperationSpec(
            previous_source_slot_id="replay-slot",
            target_human_source_slot_id="human-slot",
            previous_source_action_contract_hash=(
                manifest.source_slot("replay-slot").source_action_contract_hash
            ),
            target_human_action_contract_hash=(
                manifest.source_slot("human-slot").source_action_contract_hash
            ),
            execution_binding_hash=manifest.execution_binding.semantic_content_hash,
            readiness_observer_binding_id=provider.observer_binding_id,
            boundary_policy=MixedControlBoundaryPolicy(
                policy_id="cs6b1f-formal-hil-safe-v1",
                policy_suspension_mode="hold_and_release_authority",
                episode_boundary_motion_policy="hold_before_unbind",
                human_emission_max_age_sec=10.0,
                measured_state_max_age_sec=10.0,
            ),
            readiness_timeout_sec=0.5,
            capture_boundary_id="episode-boundary:cs6b1f-formal-hil",
        )
        spec_id = station.register_human_takeover_operation(spec, provider)
        state = station.station.state
        takeover_submission = coordinator.submit_intent(
            StationIntentDraft(
                client_id="operator:cs6b1f-formal-hil",
                client_sequence=10,
                intent_kind="request_human_takeover",
                produced_at=clock(),
                produced_clock_domain_id="local_monotonic",
                expected_state_revision=state.state_revision,
                manifest_hash=station.station.manifest.semantic_content_hash,
                subject_id="human-slot",
                resource_binding_id="replay-slot",
                operation_spec_id=spec_id,
                reason="cs6b1f formal HIL takeover",
            )
        )
        assert takeover_submission.transition.receipt.status == (
            "operation_requested"
        )

        deadline = time.monotonic() + 2.0
        while (
            station.data_plane.safe_human_takeover_state is None
            or station.data_plane.safe_human_takeover_state.phase
            != "awaiting_human_readiness"
        ):
            if time.monotonic() >= deadline:
                raise AssertionError("Formal HIL takeover did not reach readiness wait.")
            time.sleep(0.001)
        passive_step = station.step_passive_human_pose_source(now=clock())
        assert passive_step.emission_valid
        assert passive_step.emission is not None
        # The source emission is stamped by TeleopSession's local-monotonic
        # observer clock.  Advance the virtual clock to that exact sample
        # before the readiness finalizer measures its age.
        clock.value = max(clock.value, passive_step.emission.emitted_at)
        assert coordinator.wait_until_idle(timeout_sec=2.0)
        completion = coordinator.completions[-1]
        assert completion.completion_transition.result.succeeded, (
            station.takeover_failure_reason_by_intent_id
        )
        assert completion.completion_transition.result.status == (
            "human_takeover_ready"
        )
        promoted = station.promote_human_takeover_emission(spec_id)
        assert promoted.accepted
        assert promoted.emission is passive_step.emission
        assert promoted.adapter_result.proposal_draft is not None
        # Keep publication after both the passive emission and the rig's
        # deterministic physics timeline (60 ticks at 1 kHz).
        clock.value = max(
            clock.value,
            promoted.adapter_result.proposal_draft.target_effective_at,
        ) + 0.1
        human_step = station.step(now=clock(), physics_ticks_after_dispatch=60)
        assert human_step.status == "dispatched"

        _submission, release_completion = submit_station_operation(
            sequence=11,
            kind="release_motion_authority",
            subject_id="human-slot",
        )
        assert release_completion is not None
        assert release_completion.completion_transition.result.succeeded
        hold_step = station.step(now=clock())
        assert hold_step.status in {"execution_hold_active", "not_due"}
        assert not station.publication_barrier_by_source["human-slot"].snapshot().enabled

        clock.value += 0.01
        released = recorder.release_episode(
            prepared,
            unbound_at=clock(),
            reason="cs6b1f_formal_hil_finalize",
        )
        assert released.seal.trace_sealed, (
            released.seal.reason_codes,
            released.seal.missing_required_event_types,
            released.seal.lineage_issue_codes,
        )
        assert released.seal.capture_commit_eligible
        finalized = recorder.legacy_finalize_completed(
            released,
            completed_at=clock() + 0.1,
        )
        assert finalized.capture_commit_status == "committed"
        assert station_trace_bridge.admission_failures == ()
        assert coordinator.transition_observer_failure_count == 0
    finally:
        coordinator.stop(timeout_sec=2.0)
        station.close()
        durable_sink.flush_pending()
        health.close()
        spool.close()

    captured = load_episode_control_trace(tmp_path, episode_uuid)
    assert captured.committed
    assert {
        record.fact.event_type for record in captured.dependency_records
    } == {"execution_manifest", "station_run_manifest"}
    event_types = tuple(
        record.fact.event_type for record in captured.episode_records
    )
    assert event_types.count("human_command_emission") == 1
    assert event_types.count("replay_source_frame_emission") == 1
    assert event_types.count("station_intent_transition") == 12
    assert event_types.count("station_operation_transition") == 11
    assert event_types.count("station_publication_barrier_transition") >= 4
    decisions = materialize_actual_decision_view(captured)
    assert len(decisions) == 2
    assert step is not None
    assert any(
        decision.decision_id == step.decision.decision_id for decision in decisions
    )
    assert any(
        decision.selected_action["action_space"] == "bimanual_ee_pose_v1"
        for decision in decisions
    )
