from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from control_station import (
    HumanTakeoverOperationSpec,
    HumanTakeoverReadinessCandidate,
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
    build_mujoco_pose_human_proposal_adapter,
    build_scoped_cs6b1f_mujoco_pose_human_replay_station,
    build_cs6b1f_pose_human_replay_execution_manifest,
    build_scoped_cs6b1d_mujoco_pose_human_replay_station,
)
from control_station.handoff import MixedControlBoundaryPolicy
from control_station.sources import (
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
    CompositionSpec,
    RobotAction,
    SCHEDULED_REFERENCE_SEGMENT_SIM_V1,
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
            replay_producer_id="replay-main",
            enable_replay_authority=True,
            decision_hz=30.0,
            authority_policy_id="cs6b1d-pose-human-replay-lease-v1",
            initial_control_mode="hybrid",
            reference_activation_profile_id=SCHEDULED_REFERENCE_SEGMENT_SIM_V1,
            clock_domain_id="local_monotonic",
            pose_frame_contract_hash=POSE_FRAME_HASH,
            authority_takeover_allowed_producers=("human-main", "replay-main"),
        ),
        trace_sink=trace_sink,
        pose_resolver=resolver,
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


def _replay_emission(path: Path, clock: ManualClock):
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
        group_key="pose-group-0",
        local_frame_index=0,
        source_action_index=4,
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
    assert "composition_input" in shadow["trace_types"]
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
    representation_fact_ids = set(human_input.context_ids).intersection(
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
            fact.event_type == "composition_input"
            and fact.payload["input_role"] == "human_pose"
            for fact in runtime.trace_facts
        ) == 1
    finally:
        station.close()


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
