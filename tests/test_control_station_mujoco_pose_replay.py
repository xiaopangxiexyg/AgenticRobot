from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from control_station import (
    SideEffectStationOperationCoordinator,
    StationControlTraceBridge,
    StationIntentDraft,
    project_station_run_manifest,
)
from control_station.composition import (
    CS2B_CANONICAL_DESK_POSE_MUJOCO_REPLAY_STATION_MANIFEST_ID_V1,
    CS2B_POSE_MUJOCO_REPLAY_STATION_MANIFEST_ID_V1,
    ReplaySourceObserverBinding,
    build_cs2bc_replay_gui_execution_manifest,
    build_scoped_cs2bc_mujoco_replay_gui_station,
    build_scoped_cs2b_mujoco_replay_station,
)
from control_station.sources import ReplayGuiStationConsumer
from dataset_contracts.completeness import trace_requirement_profile
from dataset_contracts.control_trace_episode import (
    CONTROL_TRACE_SESSION_RELATIVE_ROOT_V1,
)
from dataset_interface.control_trace import (
    load_episode_control_trace,
    materialize_actual_decision_view,
)
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
    CanonicalDeskToRuntimePoseTransformer,
    RobotAction,
    SCHEDULED_REFERENCE_SEGMENT_SIM_V1,
)
from robot_motion.action.authoritative_trace import project_control_execution_manifest
from robot_motion.pose_frame_profiles import (
    default_rm65_workspace_near_boundary_collection_pose_frame_contract,
    default_rm65_workspace_near_boundary_runtime_pose_frame_contract,
)
from robot_motion.replay import (
    ReplayTrajectoryGroup,
    ReplayProposalSource,
    ReplayProposalSourceConfig,
    build_replay_binding_assessment,
    build_replay_source_frame_emission,
    create_replay_source_frame_emission,
)
from robot_motion.deploy import ACTION_ARM_POSE_SLICES, DEPLOY_ACTION_DIM, nan_vector
from robot_motion.action.executor import RobotActionResolver
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
from teleop_runtime.replay.source_binding import build_replay_gui_source_binding
from teleop_runtime.replay.executor import ReplayExecutionSettings
from teleop_runtime.replay.hardware import ReplayHardwareConfig
from teleop_runtime.replay.session import ReplayGuiSession
from hardware.rm65.calibration_profiles import make_profile
from teleop_runtime.control_trace import (
    ControlTraceRuntime,
    ControlTraceSpool,
    ControlTraceSpoolSink,
    ControlTraceWriterHealthLedger,
    EpisodeCaptureManifestStore,
    EpisodeTraceRecorderSink,
    RuntimeTraceSession,
)


POSE_FRAME_HASH = "rm65-arm-base-tool-tcp-pose-frame-v1"


class ManualClock:
    def __init__(self, value: float = 20.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


class FakeScene:
    arm_specs = {"left": object(), "right": object()}
    gripper_specs = {}

    def __init__(self) -> None:
        self.qpos = {
            "left": np.zeros(6, dtype=float),
            "right": np.zeros(6, dtype=float),
        }
        self.physics_commands: list[dict[str, np.ndarray]] = []

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
        command = {
            side: np.asarray(qpos, dtype=float).reshape(6).copy()
            for side, qpos in qpos_by_side.items()
        }
        self.physics_commands.append(command)
        self.qpos = {side: value.copy() for side, value in command.items()}


class FakeCoupledPlanner:
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
            metadata={"planner": "cs2b-pose-fixture"},
        )


class FakeDatasetReplayRig:
    """Device-free Rig double for the real GUI replay preflight/worker."""

    connected = True

    def __init__(self) -> None:
        self.action_resolver = RobotActionResolver()
        self.sent_commands = []
        self.hold_reasons: list[str] = []
        self.begin_count = 0
        self.restore_count = 0
        self.close_count = 0
        self.command_dt_history: list[float] = []

    def current_qpos_by_side(self):
        return {
            "left": np.zeros(6, dtype=float),
            "right": np.zeros(6, dtype=float),
        }

    def current_tcp_pose_by_side(self):
        return {
            "left": np.eye(4, dtype=float),
            "right": np.eye(4, dtype=float),
        }

    def current_tcp_flange_by_side(self):
        return {
            "left": np.zeros(6, dtype=float),
            "right": np.zeros(6, dtype=float),
        }

    def end_effector_backend_sides(self):
        return ()

    def set_arm_command_dt_sec(self, control_dt_sec: float) -> None:
        self.command_dt_history.append(float(control_dt_sec))

    def begin_replay_speed(self) -> None:
        self.begin_count += 1

    def restore_replay_speed(self) -> None:
        self.restore_count += 1

    def send_commands(self, command, phase_recorder=None) -> None:
        del phase_recorder
        self.sent_commands.append(command)

    def hold_motion(self, reason: str = "test replay hold") -> None:
        self.hold_reasons.append(str(reason))

    def close(self) -> None:
        self.close_count += 1


def _write_tool_profile(path: Path, *, side: str) -> str:
    profile = make_profile(
        name=f"cs2bc_{side}_tool",
        arm=side,
        tool_name="cs2bc_test_tool",
    )
    profile["tcp"]["pose_flange_m_rad"] = [0.0] * 6
    path.write_text(json.dumps(profile), encoding="utf-8")
    return str(path)


def _write_cs2bc_dataset(root: Path, runtime: MujocoDeployRuntime) -> Path:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    left_profile = _write_tool_profile(root / "left_tool.json", side="left")
    right_profile = _write_tool_profile(root / "right_tool.json", side="right")
    info = {
        "chunk_size": 1000,
        "nominal_fps": 15.0,
        "dexteleop_dataset_schema": "dexteleop_dataset_v2.4",
        "dataset_id": "cs2bc-dataset-backed-pose-source",
        "pose_frame_contract_hash": POSE_FRAME_HASH,
        "component_schema_hash": runtime.catalog.semantic_content_hash,
        "action_spec_hash_by_component": {
            "left_arm": "rm65-tool-tcp-pose-v1",
            "right_arm": "rm65-tool-tcp-pose-v1",
        },
        "units_convention_id": "si_m_rad_xyzw_v1",
    }
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    action = nan_vector(DEPLOY_ACTION_DIM)
    action[ACTION_ARM_POSE_SLICES["left"]] = np.asarray(
        [0.2, -0.1, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
    )
    action[ACTION_ARM_POSE_SLICES["right"]] = np.asarray(
        [0.2, 0.1, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
    )
    parquet = root / "data" / "chunk-000" / "episode_000000.parquet"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "frame_index": 0,
                    "timestamp": 100.0,
                    "selected_action": action.tolist(),
                    "action.replay.left_arm_joint_target": [0.0] * 6,
                    "action.replay.right_arm_joint_target": [0.0] * 6,
                    "observation.arm.left.current_qpos": [0.0] * 6,
                    "observation.arm.right.current_qpos": [0.0] * 6,
                }
            ]
        ),
        parquet,
    )
    episode = {
        "episode_index": 0,
        "episode_id": "episode_000000",
        "frame_count": 1,
        "duration_sec": 0.0,
        "parquet_path": "data/chunk-000/episode_000000.parquet",
        "metadata": {
            "arm_control_mode": "impedance",
            "left_tool_profile": left_profile,
            "right_tool_profile": right_profile,
        },
    }
    (meta / "episodes.jsonl").write_text(
        json.dumps(episode) + "\n", encoding="utf-8"
    )
    return parquet


def _runtime(
    clock: ManualClock,
    *,
    trace_sink=None,
    runtime_session_id: str = "cs2b-pose-runtime",
    control_domain_id: str = "cs2b-pose-domain",
    embodiment_instance_id: str = "cs2b-pose-rm65",
):
    scene = FakeScene()
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
        RM65CanonicalIK(scene, FakeCoupledPlanner())
    )
    runtime = MujocoDeployRuntime(
        rig,
        MujocoDeployConfig(
            runtime_session_id=runtime_session_id,
            control_domain_id=control_domain_id,
            embodiment_instance_id=embodiment_instance_id,
            producer_id="unused-policy-source",
            replay_producer_id="replay-main",
            decision_hz=30.0,
            reference_activation_profile_id=SCHEDULED_REFERENCE_SEGMENT_SIM_V1,
            authority_policy_id="mujoco-single-pose-replay-authority-v1",
            initial_control_mode="replay",
            enable_replay_authority=True,
            clock_domain_id="local_monotonic",
            pose_frame_contract_hash=POSE_FRAME_HASH,
        ),
        pose_resolver=resolver,
        trace_sink=trace_sink,
        clock=clock,
    )
    return runtime, scene


def _source(runtime: MujocoDeployRuntime) -> ReplayProposalSource:
    action_contract = ReplaySourceActionContract(
        contract_id="cs2b-source-action:arm-base-pose",
        interpretation_status=REPLAY_SOURCE_INTERPRETATION_VERIFIED,
        action_stage=REPLAY_ACTION_STAGE_SELECTED,
        action_space="bimanual_ee_pose_v1",
        component_ids=("left_arm", "right_arm"),
        selection_group_by_component={
            "left_arm": "arms",
            "right_arm": "arms",
        },
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
        contract_id="cs2b-source:arm-base-pose",
        replay_mode=REPLAY_MODE_CANONICAL_ACTION,
        dataset_schema="dexteleop_dataset_v2.4",
        source_dataset_id="cs2b-pose-source",
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
        binding_id="cs2b-arm-base-pose-to-mujoco",
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
                schedule_id="cs2b-pose-15hz-v1",
                schedule_mode=REPLAY_SCHEDULE_FIXED_RATE,
                source_hz=15.0,
                target_hz=None,
                resampling_policy="none",
                source_start_index=3,
            ),
            target_binding_spec=binding,
            binding_assessment=assessment,
            requested_use=REPLAY_BINDING_USE_EXECUTION_CANDIDATE,
        )
    )


def _advance_to_authority(station) -> None:
    target_id = station.station.manifest.execution_binding.binding_id
    source_id = station.source_adapter.subject_id
    for sequence, (kind, subject_id) in enumerate(
        (
            ("acquire_control_lease", None),
            ("connect_target", target_id),
            ("connect_source", source_id),
            ("run_preflight", target_id),
            ("run_preflight", source_id),
            ("request_candidate", source_id),
            ("request_motion_authority", source_id),
        )
    ):
        state = station.station.state
        transition = station.station.submit_intent(
            StationIntentDraft(
                client_id="operator:pose-test",
                client_sequence=sequence,
                intent_kind=kind,
                produced_at=station.data_plane.runtime.clock(),
                produced_clock_domain_id="operator-test-monotonic",
                expected_state_revision=state.state_revision,
                manifest_hash=station.station.manifest.semantic_content_hash,
                subject_id=subject_id,
                reason="cs2b pose replay fixture",
            )
        )
        assert transition.receipt.accepted
        if transition.receipt.status == "operation_requested":
            completion = station.complete_pending_operation(transition)
            assert completion.completion_transition.accepted


def test_cs2b_arm_base_pose_replay_keeps_pose_safe_and_qpos_target_separate(
    tmp_path: Path,
) -> None:
    clock = ManualClock()
    runtime, scene = _runtime(clock)
    source = _source(runtime)
    source_path = tmp_path / "episode.parquet"
    observer = ReplaySourceObserverBinding.create(
        source_slot_id="replay-main",
        source_locator_id="cs2b-pose-source",
        expected_source_path=str(source_path),
        source_episode_index=0,
        group_key="pose-group-0",
        source_start_index=3,
        clock_domain_id="local_monotonic",
    )
    station = build_scoped_cs2b_mujoco_replay_station(
        runtime,
        source,
        observer,
        station_clock=clock,
    )
    assert (
        station.station.manifest.manifest_id
        == CS2B_POSE_MUJOCO_REPLAY_STATION_MANIFEST_ID_V1
    )
    assert "canonical_desk_pose_transform" in (
        station.station.manifest.explicit_exclusions
    )
    _advance_to_authority(station)
    identity = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    pose = RobotAction(
        action_space="bimanual_ee_pose_v1",
        timestamp=clock(),
        arm_pose_by_side={"left": identity, "right": identity},
        frame_by_effector={
            "left_arm": "left_arm_base",
            "right_arm": "right_arm_base",
        },
        controlled_frame_by_effector={
            "left_arm": "tool_tcp",
            "right_arm": "tool_tcp",
        },
        source="existing_gui_replay_cursor",
    )
    emission = create_replay_source_frame_emission(
        source_path=str(source_path),
        source_episode_index=0,
        group_key="pose-group-0",
        local_frame_index=0,
        source_action_index=3,
        source_relative_time=0.0,
        action=pose,
        observed_at=clock(),
        target_effective_at=clock(),
        effective_until=clock() + 0.1,
        clock_domain_id="local_monotonic",
    )

    submission = station.submit_source_frame_emission(emission)
    assert submission.source_emission is emission
    step = station.step(now=clock())
    assert step.status == "dispatched"
    assert step.safety_result is not None
    assert step.safety_result.safe_action is not None
    assert step.safety_result.safe_action.action_space == "bimanual_ee_pose_v1"
    assert step.command is not None
    assert all(item.command_type == "joint_vector" for item in step.command.commands.values())
    event_types = {fact.event_type for fact in runtime.trace_facts}
    assert {
        "resolved_execution_target",
        "execution_target_admission",
        "reference_segment",
        "reference_segment_admission",
        "safety_anchor_commit_evidence",
    }.issubset(event_types)
    receiver = runtime.reference_receiver
    assert receiver is not None
    assert not receiver.anchor_candidates
    assert receiver.last_result is not None
    assert receiver.last_result.segment is not None
    assert (
        receiver.last_result.segment.end_parent.source_kind
        == "resolved_execution_target"
    )
    assert scene.physics_commands == []
    station.close()


def test_cs2b_canonical_desk_replay_uses_verified_downstream_transform(
    tmp_path: Path,
) -> None:
    clock = ManualClock()
    source_frame_contract = (
        default_rm65_workspace_near_boundary_collection_pose_frame_contract(
            ("left", "right")
        )
    )
    assert source_frame_contract is not None
    target_frame_contract = (
        default_rm65_workspace_near_boundary_runtime_pose_frame_contract()
    )
    transformer = CanonicalDeskToRuntimePoseTransformer(
        source_pose_frame_contract=source_frame_contract,
        target_pose_frame_contract=target_frame_contract,
    )
    scene = FakeScene()
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
        RM65CanonicalIK(scene, FakeCoupledPlanner()),
        pose_transformer=transformer,
    )
    runtime = MujocoDeployRuntime(
        rig,
        MujocoDeployConfig(
            runtime_session_id="cs2b-desk-runtime",
            control_domain_id="cs2b-desk-domain",
            embodiment_instance_id="cs2b-desk-rm65",
            producer_id="unused-policy-source",
            replay_producer_id="replay-main",
            decision_hz=30.0,
            reference_activation_profile_id=SCHEDULED_REFERENCE_SEGMENT_SIM_V1,
            authority_policy_id="mujoco-single-desk-replay-authority-v1",
            initial_control_mode="replay",
            enable_replay_authority=True,
            clock_domain_id="local_monotonic",
            pose_frame_contract_hash=target_frame_contract.contract_hash,
        ),
        pose_resolver=resolver,
        clock=clock,
    )
    action_contract = ReplaySourceActionContract(
        contract_id="cs2b-source-action:canonical-desk-pose",
        interpretation_status=REPLAY_SOURCE_INTERPRETATION_VERIFIED,
        action_stage=REPLAY_ACTION_STAGE_SELECTED,
        action_space="bimanual_ee_pose_v1",
        component_ids=("left_arm", "right_arm"),
        selection_group_by_component={"left_arm": "arms", "right_arm": "arms"},
        coupled_selection_groups=("arms",),
        pose_frame_contract_hash=source_frame_contract.contract_hash,
        component_schema_hash=runtime.catalog.semantic_content_hash,
        action_spec_hash_by_component={
            "left_arm": "rm65-tool-tcp-pose-v1",
            "right_arm": "rm65-tool-tcp-pose-v1",
        },
        units_convention_id="si_m_rad_xyzw_v1",
    )
    source_contract = ReplaySourceContract(
        contract_id="cs2b-source:canonical-desk-pose",
        replay_mode=REPLAY_MODE_CANONICAL_ACTION,
        dataset_schema="dexteleop_dataset_v2.3",
        source_dataset_id="cs2b-desk-source",
        episode_id="episode-desk-0",
        action_stage=REPLAY_ACTION_STAGE_SELECTED,
        action_field="action",
        source_action_space="bimanual_ee_pose_v1",
        source_frame_contract_hash=source_frame_contract.contract_hash,
        source_component_schema_hash=runtime.catalog.semantic_content_hash,
    )
    binding, assessment = build_replay_binding_assessment(
        action_contract,
        runtime.target_runtime_action_contract,
        binding_id="cs2b-canonical-desk-to-mujoco",
        transform_receipt_id=transformer.binding_receipt.receipt_id,
        transform_receipt_hash=transformer.binding_receipt.semantic_content_hash,
    )
    assert assessment.verdict == REPLAY_BINDING_VERIFIED_COMPATIBLE
    source = ReplayProposalSource(
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
                schedule_id="cs2b-desk-15hz-v1",
                schedule_mode=REPLAY_SCHEDULE_FIXED_RATE,
                source_hz=15.0,
                target_hz=None,
                resampling_policy="none",
                source_start_index=0,
            ),
            target_binding_spec=binding,
            binding_assessment=assessment,
            requested_use=REPLAY_BINDING_USE_EXECUTION_CANDIDATE,
        )
    )
    source_path = tmp_path / "episode.parquet"
    observer = ReplaySourceObserverBinding.create(
        source_slot_id="replay-main",
        source_locator_id="cs2b-desk-source",
        expected_source_path=str(source_path),
        source_episode_index=0,
        group_key="desk-group-0",
        source_start_index=0,
        clock_domain_id="local_monotonic",
    )
    station = build_scoped_cs2b_mujoco_replay_station(
        runtime, source, observer, station_clock=clock
    )
    assert station.station.manifest.manifest_id == (
        CS2B_CANONICAL_DESK_POSE_MUJOCO_REPLAY_STATION_MANIFEST_ID_V1
    )
    assert "canonical_desk_pose_transform" not in (
        station.station.manifest.explicit_exclusions
    )
    _advance_to_authority(station)
    pose = np.asarray([0.45, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0])
    action = RobotAction(
        action_space="bimanual_ee_pose_v1",
        timestamp=clock(),
        arm_pose_by_side={"left": pose, "right": pose},
        frame_by_effector={
            "left_arm": "canonical_desk",
            "right_arm": "canonical_desk",
        },
        controlled_frame_by_effector={
            "left_arm": "tool_tcp",
            "right_arm": "tool_tcp",
        },
        source="existing_gui_replay_cursor",
    )
    emission = create_replay_source_frame_emission(
        source_path=str(source_path),
        source_episode_index=0,
        group_key="desk-group-0",
        local_frame_index=0,
        source_action_index=0,
        source_relative_time=0.0,
        action=action,
        observed_at=clock(),
        target_effective_at=clock(),
        effective_until=clock() + 0.1,
        clock_domain_id="local_monotonic",
    )

    station.submit_source_frame_emission(emission)
    step = station.step(now=clock())

    assert step.status == "dispatched"
    assert step.safety_result.safe_action.arm_frame("left") == "canonical_desk"
    resolution = resolver.resolution_for_command(step.command)
    assert resolution.input_safe_action.arm_frame("left") == "canonical_desk"
    assert resolution.solver_input_pose_action.arm_frame("left") == "left_arm_base"
    assert resolution.transform_contract_hash == (
        transformer.binding_receipt.semantic_content_hash
    )
    assert station.data_plane.binding_receipt.pose_transform_contract_hash == (
        transformer.binding_receipt.semantic_content_hash
    )
    station.close()


def test_cs2bc_checked_gui_binding_builds_exact_arm_base_station(
    tmp_path: Path,
) -> None:
    clock = ManualClock()
    runtime, scene = _runtime(clock)
    source_path = tmp_path / "episode.parquet"
    action = nan_vector(DEPLOY_ACTION_DIM)
    left_pose = np.asarray(
        [0.2, -0.1, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
    )
    right_pose = np.asarray(
        [0.2, 0.1, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
    )
    action[ACTION_ARM_POSE_SLICES["left"]] = left_pose
    action[ACTION_ARM_POSE_SLICES["right"]] = right_pose
    group = ReplayTrajectoryGroup(
        key="g",
        label="dataset-backed-gui-source",
        source_path=str(source_path),
        source_episode_index=0,
        start_index=0,
        end_index=1,
        frame_count=1,
        action_hz=15.0,
        actions=np.asarray([action], dtype=np.float32),
        frame_times_sec=(0.0,),
        metadata={
            "source_metadata": {
                "dataset_info": {
                    "dexteleop_dataset_schema": "dexteleop_dataset_v2.4",
                    "dataset_id": "cs2bc-pose-source",
                    "pose_frame_contract_hash": POSE_FRAME_HASH,
                    "component_schema_hash": runtime.catalog.semantic_content_hash,
                    "action_spec_hash_by_component": {
                        "left_arm": "rm65-tool-tcp-pose-v1",
                        "right_arm": "rm65-tool-tcp-pose-v1",
                    },
                    "units_convention_id": "si_m_rad_xyzw_v1",
                }
            },
            "action_column": "selected_action",
            "action_metadata": {"pose_action_absolute_frame": "arm_base"},
            "frame_timing": {"source_timestamp_mode": "dataset"},
        },
    )
    source_binding = build_replay_gui_source_binding(
        group,
        episode_id="episode_000000",
        active_sides=("left", "right"),
        include_end_effectors=False,
        trajectory_source="action_pose",
    )
    station = build_scoped_cs2bc_mujoco_replay_gui_station(
        runtime,
        source_binding,
        station_clock=clock,
    )
    _advance_to_authority(station)
    emission = build_replay_source_frame_emission(
        group,
        active_sides=("left", "right"),
        index=0,
        include_arms=True,
        include_end_effectors=False,
        observed_at=clock(),
        schedule_start_at=clock(),
        clock_domain_id="local_monotonic",
    )

    submission = station.submit_source_frame_emission(emission)
    assert submission.source_emission is emission
    step = station.step(now=clock())
    assert step.status == "dispatched"
    assert step.safety_result is not None
    assert step.safety_result.safe_action.action_space == "bimanual_ee_pose_v1"
    assert step.command is not None
    assert all(item.command_type == "joint_vector" for item in step.command.commands.values())
    assert scene.physics_commands == []
    station.close()


def test_cs2bc_dataset_backed_gui_session_reuses_one_group_and_emission(
    tmp_path: Path,
) -> None:
    """Real loader/preflight/worker; fake legacy Rig; no second cursor."""

    clock = ManualClock()
    runtime, scene = _runtime(clock)
    source_parquet = _write_cs2bc_dataset(tmp_path, runtime)
    submitted_emissions = []
    station_holder = {}

    def submit_to_station(emission):
        submitted_emissions.append(emission)
        return station_holder["station"].submit_source_frame_emission(emission)

    consumer = ReplayGuiStationConsumer(submit_to_station, queue_capacity=4)
    session = ReplayGuiSession(
        tmp_path,
        hardware_config=ReplayHardwareConfig(
            active_sides=("left", "right"),
            arm_control_mode="impedance",
            connect_hardware=False,
        ),
        execution_settings=ReplayExecutionSettings(
            send_actions_enabled=True,
            action_column="selected_action",
            position_collision_required=False,
        ),
        action_observer_queue=consumer.emission_queue,
    )
    fake_rig = FakeDatasetReplayRig()
    session.executor._rig = fake_rig  # type: ignore[assignment]
    station = None
    try:
        assert session.episode is not None
        preflight = session.executor.preflight(session.episode)
        assert preflight["preflight"]["ok"] is True
        checked_group = session.executor._group
        assert checked_group is not None
        assert Path(checked_group.source_path).resolve() == source_parquet.resolve()
        source_binding = session.checked_replay_source_binding()
        assert source_binding.expected_source_path == str(source_parquet.resolve())
        assert source_binding.frame_count == 1
        assert source_binding.source_action_contract.interpretation_status == "verified"

        station = build_scoped_cs2bc_mujoco_replay_gui_station(
            runtime,
            source_binding,
            station_clock=clock,
        )
        station_holder["station"] = station
        _advance_to_authority(station)
        consumer.start()

        # This is the exact production replay worker over the exact group cached
        # by preflight.  Only its legacy dispatch endpoint is the device-free Rig.
        with mock.patch(
            "teleop_runtime.replay.executor.time.monotonic",
            side_effect=clock,
        ):
            session.executor._replay_worker(fake_rig, checked_group)  # type: ignore[arg-type]
        consumer.emission_queue.join()

        assert session.executor._group is checked_group
        assert session.executor.status()["action_observer"]["offer_count"] == 1
        consumer_status = consumer.status()
        assert consumer_status.received_count == 1
        assert consumer_status.submitted_count == 1
        assert consumer_status.rejected_count == 0
        assert consumer_status.error_count == 0
        assert len(submitted_emissions) == 1
        emission = submitted_emissions[0]
        assert emission.source_path == str(source_parquet.resolve())
        assert emission.source_action_index == 0
        assert emission.local_frame_index == 0
        assert len(fake_rig.sent_commands) == 1

        # The test composition injects one monotonic clock into both branches;
        # the source emission still comes from the unmodified replay worker.
        clock.value = emission.target_effective_at
        step = station.step(now=clock())
        assert step.status == "dispatched"
        assert step.safety_result is not None
        assert step.safety_result.safe_action.action_space == "bimanual_ee_pose_v1"
        assert step.command is not None
        assert scene.physics_commands == []
    finally:
        consumer.close()
        if station is not None:
            station.close()
        session.close()


def test_cs2bc_dataset_replay_station_seals_commits_and_rebuilds_lineage(
    tmp_path: Path,
) -> None:
    """Formal fixture only: real loader/runtime/WAL/episode/reader, no device."""

    capture_root = tmp_path / "formal_capture"
    dataset_root = tmp_path / "source_dataset"
    capture_root.mkdir()
    dataset_root.mkdir()
    runtime_session_id = "cs2bc-formal-runtime"
    control_domain_id = "cs2bc-formal-domain"
    embodiment_instance_id = "cs2bc-formal-rm65"
    trace_session_id = "cs2bc-formal-trace"
    episode_uuid = "cs2bc-formal-episode"
    clock = ManualClock()
    profile = trace_requirement_profile(
        "deployable_sim_scheduled_pose_to_qpos_station_v1"
    )
    trace_runtime = ControlTraceRuntime(
        RuntimeTraceSession(
            trace_session_id=trace_session_id,
            runtime_session_id=runtime_session_id,
            control_domain_id=control_domain_id,
            profile_id=profile.profile_id,
            started_at=clock(),
            clock_domain_id="local_monotonic",
        ),
        profile=profile,
        capacity=8192,
    )
    session_root = (
        capture_root / CONTROL_TRACE_SESSION_RELATIVE_ROOT_V1 / trace_session_id
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
        sink_id="cs2bc-replay-sim-spool",
        runtime=trace_runtime,
        spool=spool,
        health_ledger=health,
        clock=clock,
    )
    dependency_sequences: list[int] = []
    recorder = EpisodeTraceRecorderSink(
        trace_runtime,
        lifecycle_owner_id="cs2bc-test-composition-owner",
        flush_to_durable=durable_sink.flush_pending,
        dependency_sequences=lambda _runtime: tuple(dependency_sequences),
        manifest_store=EpisodeCaptureManifestStore(capture_root),
        uuid_factory=lambda: episode_uuid,
        control_clock=clock,
        commit_clock=clock,
    )

    def admit_trace(fact):
        receipt = trace_runtime.issue(fact)
        if not receipt.accepted:
            raise RuntimeError(
                f"CS2bc trace rejected {fact.event_type}: {receipt.rejection_code}"
            )
        return receipt

    runtime, _scene = _runtime(
        clock,
        trace_sink=admit_trace,
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
    )
    source_parquet = _write_cs2bc_dataset(dataset_root, runtime)
    submitted_emissions = []
    station_holder = {}

    def submit_to_station(emission):
        submitted_emissions.append(emission)
        return station_holder["station"].submit_source_frame_emission(emission)

    consumer = ReplayGuiStationConsumer(submit_to_station, queue_capacity=4)
    session = ReplayGuiSession(
        dataset_root,
        hardware_config=ReplayHardwareConfig(
            active_sides=("left", "right"),
            arm_control_mode="impedance",
            connect_hardware=False,
        ),
        execution_settings=ReplayExecutionSettings(
            send_actions_enabled=True,
            action_column="selected_action",
            position_collision_required=False,
        ),
        action_observer_queue=consumer.emission_queue,
    )
    fake_rig = FakeDatasetReplayRig()
    session.executor._rig = fake_rig  # type: ignore[assignment]
    station = None
    coordinator = None
    prepared = None
    step = None
    try:
        assert session.episode is not None
        preflight = session.executor.preflight(session.episode)
        assert preflight["preflight"]["ok"] is True
        checked_group = session.executor._group
        assert checked_group is not None
        source_binding = session.checked_replay_source_binding()
        station = build_scoped_cs2bc_mujoco_replay_gui_station(
            runtime,
            source_binding,
            station_clock=clock,
            station_clock_domain_id="local_monotonic",
        )
        station_holder["station"] = station
        execution_manifest = build_cs2bc_replay_gui_execution_manifest(
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
                runtime_session_id=runtime_session_id,
                occurred_at=clock(),
                clock_domain_id="local_monotonic",
            ),
        ):
            receipt = admit_trace(dependency_fact)
            assert receipt.sink_accepted_sequence is not None
            dependency_sequences.append(receipt.sink_accepted_sequence)

        bridge = StationControlTraceBridge(
            manifest=station.station.manifest,
            runtime_session_id=runtime_session_id,
            embodiment_instance_id=embodiment_instance_id,
            trace_sink=admit_trace,
            clock=clock,
            clock_domain_id="local_monotonic",
        )
        station.bind_publication_barrier_observer(
            bridge.record_publication_barrier_transition
        )
        coordinator = SideEffectStationOperationCoordinator(
            coordinator_id="cs2bc-replay-capture-side-effect-worker",
            runtime=station.station,
            adapters=(station.target_adapter, station.source_adapter),
            operation_prepare=station.prepare_pending_operation,
            operation_executor=station.complete_pending_operation,
            intent_transition_observer=bridge.record_intent_transition,
            operation_transition_observer=bridge.record_operation_transition,
            queue_capacity=2,
            clock=clock,
            clock_domain_id="local_monotonic",
        )

        def station_operation(sequence: int, kind: str, subject_id: str | None):
            before = len(coordinator.completions)
            state = station.station.state
            submission = coordinator.submit_intent(
                StationIntentDraft(
                    client_id="operator:cs2bc-formal-capture",
                    client_sequence=sequence,
                    intent_kind=kind,
                    produced_at=clock(),
                    produced_clock_domain_id="local_monotonic",
                    expected_state_revision=state.state_revision,
                    manifest_hash=station.station.manifest.semantic_content_hash,
                    subject_id=subject_id,
                    reason="cs2bc dataset replay formal capture fixture",
                )
            )
            if submission.transition.receipt.status != "operation_requested":
                return None
            assert coordinator.wait_until_idle(timeout_sec=2.0)
            assert len(coordinator.completions) == before + 1
            return coordinator.completions[-1]

        prepared = recorder.prepare_episode(
            legacy_episode_index=0,
            prepared_at=clock(),
        )
        recorder.bind_prepared_episode(prepared, bound_at=clock())
        coordinator.start()
        target_id = station.station.manifest.execution_binding.binding_id
        source_id = station.source_adapter.subject_id
        for sequence, (kind, subject_id) in enumerate(
            (
                ("acquire_control_lease", None),
                ("connect_target", target_id),
                ("connect_source", source_id),
                ("run_preflight", target_id),
                ("run_preflight", source_id),
                ("request_candidate", source_id),
                ("request_motion_authority", source_id),
            )
        ):
            station_operation(sequence, kind, subject_id)
        assert station.publication_barrier.snapshot().enabled

        consumer.start()
        with mock.patch(
            "teleop_runtime.replay.executor.time.monotonic",
            side_effect=clock,
        ):
            session.executor._replay_worker(fake_rig, checked_group)  # type: ignore[arg-type]
        consumer.emission_queue.join()
        assert len(submitted_emissions) == 1
        emission = submitted_emissions[0]
        assert emission.source_path == str(source_parquet.resolve())
        clock.value = emission.target_effective_at
        step = station.step(now=clock(), physics_ticks_after_dispatch=60)
        assert step.status == "dispatched"

        clock.value = emission.target_effective_at + 0.2
        release = station_operation(7, "release_motion_authority", source_id)
        assert release is not None
        station.step(now=clock())
        assert not station.publication_barrier.snapshot().enabled
        assert bridge.admission_failures == ()

        clock.value = emission.target_effective_at + 0.3
        released = recorder.release_episode(
            prepared,
            unbound_at=clock(),
            reason="cs2bc_dataset_replay_sim_finalize",
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
    finally:
        consumer.close()
        if coordinator is not None:
            coordinator.stop(timeout_sec=2.0)
        if station is not None:
            station.close()
        session.close()
        durable_sink.flush_pending()
        health.close()
        spool.close()

    captured = load_episode_control_trace(capture_root, episode_uuid)
    assert captured.committed
    assert {
        record.fact.event_type for record in captured.dependency_records
    } == {"execution_manifest", "station_run_manifest"}
    event_types = tuple(record.fact.event_type for record in captured.episode_records)
    for required_event in (
        "replay_source_frame_emission",
        "replay_source_action_contract",
        "replay_binding_assessment",
        "replay_execution_authorization",
        "candidate_authorization_admission",
        "action_proposal",
        "arbitration_decision",
        "shared_safety_result",
        "reference_segment_admission",
        "planner_command_frame",
        "dispatch_receipt",
        "measurement_alignment",
        "reference_segment_terminal",
    ):
        assert required_event in event_types
    assert event_types.count("station_publication_barrier_transition") == 2
    decisions = materialize_actual_decision_view(captured)
    assert len(decisions) == 1
    assert step is not None
    assert decisions[0].decision_id == step.decision.decision_id
