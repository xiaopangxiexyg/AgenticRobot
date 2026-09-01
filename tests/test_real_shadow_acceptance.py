from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from control_station.real_shadow_acceptance import (
    RealMotionDisposition,
    RealShadowAcceptanceStatus,
    run_real_shadow_acceptance,
)
from dataset_contracts.pose_frames import load_pose_frame_contract
from robot_motion.rm65 import (
    DEFAULT_COLLISION_SCENE,
    load_rm65_tool_profile,
    tool_tcp_pose_flange_m_rad,
)
from robot_motion.scene_obstacles import ObstacleSceneSnapshot
from robot_motion.shadow_execution import (
    AuthoritativeRobotStateSnapshot,
    RegisteredJointTrajectoryPoint,
    RegisteredShadowJointTrajectory,
    ShadowExecutionIdentity,
    ShadowReplayAdmissionReceipt,
    ShadowReplayAdmissionStatus,
    ShadowSimulationReceipt,
    ShadowSimulationStatus,
)
from sim.contracts import SimulationContract
from sim.mujoco.actuation import (
    cad_bimanual_dmtac_gripper_profile,
    rm65_bimanual_udp_feedback_profile,
)
from sim.mujoco.control.desk_tcp_position import (
    DEFAULT_INSPIRE_V3_DUAL_ARM_URDF,
    DEFAULT_SIM_HOME_QPOS_BY_SIDE,
    DEFAULT_SIMULATION_CONTRACT_PATH,
    DEFAULT_STATION_POSE_CONTRACT_PATH,
    DEFAULT_TOOL_PROFILE_BY_SIDE,
)
from sim.sim_real_tests.mujoco.arbitrary_grasp_pose_motion import (
    OBSTACLE_SCENE_FRAME_ID,
    OBSTACLE_SCENE_FRAME_SPEC_HASH,
    build_obstacle_collision_oracle,
    build_shadow_initialized_control,
)
from sim.sim_real_tests.mujoco.exact_shadow_trajectory import (
    MujocoShadowCompatibilityExpectation,
    MujocoShadowModelBinding,
    execute_registered_shadow_trajectory,
)
from teleop_runtime.skill_deploy.control_runtime import RealSkillMotionHoldReceipt
from teleop_runtime.skill_deploy.registered_trajectory import (
    RegisteredTrajectoryAuthorityConsumptionReceipt,
    RegisteredTrajectoryAuthorityConsumptionStatus,
    RegisteredTrajectoryExecutionAuthority,
    RegisteredTrajectoryRealReplayReceipt,
    RegisteredTrajectoryRealReplayStatus,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _consumption(authority, *, consumed_at: float = 10.01):
    return RegisteredTrajectoryAuthorityConsumptionReceipt(
        status=RegisteredTrajectoryAuthorityConsumptionStatus.CONSUMED,
        authority_hash=authority.semantic_content_hash,
        authority_id=authority.authority_id,
        trajectory_hash=authority.trajectory_hash,
        simulation_receipt_hash=authority.simulation_receipt_hash,
        run_id=authority.run_id,
        owner_epoch=authority.owner_epoch,
        consumed_at=consumed_at,
        evidence_refs=(f"authority-consumed:{authority.authority_id}",),
    )


def _identity(*, robot_model_hash: str = "robot:test") -> ShadowExecutionIdentity:
    return ShadowExecutionIdentity(
        station_revision="station:test",
        reference_frame_id="desk",
        reference_frame_spec_hash="frame-spec:test",
        frame_graph_hash="frame-graph:test",
        tool_profile_hash_by_side={"left": "tool:left", "right": "tool:right"},
        robot_model_hash=robot_model_hash,
        collision_model_hash="collision:test",
        joint_layout_hash="joint-layout:test",
        gripper_model_hash="dmtac-cad:test",
    )


def _snapshot(*, identity: ShadowExecutionIdentity | None = None):
    return AuthoritativeRobotStateSnapshot(
        qpos_by_side={"left": np.zeros(6), "right": np.zeros(6)},
        gripper_close_by_side={"left": 0.2, "right": 0.3},
        arm_measured_at=9.98,
        gripper_measured_at_by_side={"left": 9.985, "right": 9.986},
        available_at=9.99,
        arm_evidence_refs=("arm-cut:test",),
        gripper_evidence_refs=("dmtac:left", "dmtac:right"),
        identity=identity or _identity(),
    )


def _trajectory(
    snapshot: AuthoritativeRobotStateSnapshot,
    *,
    scene_hash: str = "scene:test",
):
    target = {
        side: value.copy() for side, value in snapshot.qpos_by_side.items()
    }
    target["right"][1] += 0.02
    return RegisteredShadowJointTrajectory(
        trajectory_id="trajectory:acceptance-test",
        created_at=10.0,
        initial_snapshot=snapshot,
        scene_hash=scene_hash,
        target_hash="target:test",
        planner_config_hash="planner:test",
        moving_sides=("right",),
        points=(
            RegisteredJointTrajectoryPoint(
                time_from_start_sec=0.0,
                qpos_by_side=snapshot.qpos_by_side,
            ),
            RegisteredJointTrajectoryPoint(
                time_from_start_sec=0.04,
                qpos_by_side=target,
            ),
        ),
    )


def _simulation(
    trajectory: RegisteredShadowJointTrajectory,
    *,
    status: ShadowSimulationStatus = ShadowSimulationStatus.COMPLETED,
):
    completed = status is ShadowSimulationStatus.COMPLETED
    return ShadowSimulationReceipt(
        trajectory_hash=trajectory.semantic_content_hash,
        initial_snapshot_hash=trajectory.initial_snapshot.semantic_content_hash,
        identity_hash=trajectory.initial_snapshot.identity.semantic_content_hash,
        scene_hash=trajectory.scene_hash,
        status=status,
        executed_point_count=len(trajectory.points) if completed else 0,
        terminal_hold_confirmed=True,
        exact_planned_qpos_execution_confirmed=completed,
        model_identity_confirmed=True,
        evidence_refs=(f"simulation:{status.value}",),
    )


class _SimulationControl:
    def __init__(self):
        self.closed = False
        self.hold_reasons: list[str] = []
        self.rig = self

    def hold_motion(self, reason: str):
        self.hold_reasons.append(reason)

    def close(self):
        self.closed = True


def _binding(_control, trajectory):
    return MujocoShadowModelBinding(
        station_revision="station:test",
        reference_frame_id="desk",
        desk_from_arm_base_by_side={
            side: tuple(np.eye(4).reshape(-1)) for side in ("left", "right")
        },
        joint_names_by_side={
            side: tuple(f"{side}_joint{i}" for i in range(1, 7))
            for side in ("left", "right")
        },
        tool_profile_id_by_side={
            "left": "DMTAC_M:left",
            "right": "DMTAC_M:right",
        },
        tool_tcp_pose_flange_m_rad_by_side={
            side: (0.0, 0.0, 0.14, 0.0, 0.0, 0.0)
            for side in ("left", "right")
        },
        gripper_profile_name="dmtac:test",
        gripper_side_by_name={
            "left_gripper": "left",
            "right_gripper": "right",
        },
        scene_revision=trajectory.scene_hash,
        simulation_scene_profile_name="scene:test",
        collision_profile_name="collision:test",
        evidence_refs=("model-binding:test",),
    )


def _actual_binding(_snapshot, trajectory):
    station = load_pose_frame_contract(DEFAULT_STATION_POSE_CONTRACT_PATH)
    arm_profile = rm65_bimanual_udp_feedback_profile()
    gripper_profile = cad_bimanual_dmtac_gripper_profile()
    simulation_contract = SimulationContract.load(DEFAULT_SIMULATION_CONTRACT_PATH)
    tool_profile_by_side = {
        side: load_rm65_tool_profile(DEFAULT_TOOL_PROFILE_BY_SIDE[side], side=side)
        for side in ("left", "right")
    }
    return MujocoShadowCompatibilityExpectation(
        station_revision=station.source_id,
        reference_frame_id=station.canonical_absolute_frame,
        desk_from_arm_base_by_side={
            side: tuple(
                station.source.effective_raw_to_canonical_desk_transform(side).reshape(
                    -1
                )
            )
            for side in ("left", "right")
        },
        joint_names_by_side={
            side: arm_profile.joint_names_by_side[side]
            for side in ("left", "right")
        },
        tool_profile_id_by_side={
            "left": "DMTAC_M:left",
            "right": "DMTAC_M:right",
        },
        tool_tcp_pose_flange_m_rad_by_side={
            side: tool_tcp_pose_flange_m_rad(tool_profile_by_side[side])
            for side in ("left", "right")
        },
        gripper_profile_name=gripper_profile.name,
        gripper_side_by_name=gripper_profile.side_by_gripper,
        scene_revision=trajectory.scene_hash,
        simulation_scene_profile_name=simulation_contract.scene.profile_name,
        collision_profile_name=DEFAULT_COLLISION_SCENE,
        evidence_refs=("real-owner-compatibility:test",),
    )


def _run(**overrides):
    snapshot = overrides.pop("snapshot", _snapshot())
    control = overrides.pop("control", _SimulationControl())

    def simulate(*, trajectory, **_kwargs):
        return _simulation(trajectory)

    values = {
        "snapshot_provider": lambda: snapshot,
        "trajectory_builder": _trajectory,
        "simulation_control_factory": lambda _snapshot: control,
        "model_binding_provider": _binding,
        "simulation_executor": simulate,
        "wall_clock": lambda: 10.01,
        "current_scene_hash_provider": lambda: "scene:test",
    }
    values.update(overrides)
    return run_real_shadow_acceptance(**values), control


def test_default_acceptance_runs_simulation_closes_it_and_never_requests_real_ports():
    real_factory_calls: list[str] = []

    receipt, control = _run(
        real_ports_factory=lambda trajectory: real_factory_calls.append(
            trajectory.semantic_content_hash
        )
    )

    assert receipt.status is RealShadowAcceptanceStatus.SIMULATION_ACCEPTED_NO_MOTION
    assert receipt.real_motion_disposition is RealMotionDisposition.NONE
    assert receipt.simulation_receipt is not None
    assert receipt.terminal_hold_confirmed is True
    assert receipt.terminal_hold_domain == "simulation"
    assert control.closed is True
    assert real_factory_calls == []


@pytest.mark.skipif(
    not DEFAULT_INSPIRE_V3_DUAL_ARM_URDF.is_file(),
    reason="local canonical DMTac dual-arm URDF asset is unavailable",
)
def test_application_acceptance_initializes_from_snapshot_and_stops_after_mujoco():
    initial = _snapshot()
    snapshot = AuthoritativeRobotStateSnapshot(
        qpos_by_side=DEFAULT_SIM_HOME_QPOS_BY_SIDE,
        gripper_close_by_side=initial.gripper_close_by_side,
        arm_measured_at=initial.arm_measured_at,
        gripper_measured_at_by_side=initial.gripper_measured_at_by_side,
        available_at=initial.available_at,
        arm_evidence_refs=initial.arm_evidence_refs,
        gripper_evidence_refs=initial.gripper_evidence_refs,
        identity=initial.identity,
    )
    real_factory_calls: list[str] = []

    obstacle_scene = ObstacleSceneSnapshot(
        reference_frame_id=OBSTACLE_SCENE_FRAME_ID,
        reference_frame_spec_hash=OBSTACLE_SCENE_FRAME_SPEC_HASH,
        revision=0,
    )
    scene_revision = obstacle_scene.semantic_content_hash

    def build_control(captured):
        control = build_shadow_initialized_control(captured)
        control.shadow_obstacle_oracle = build_obstacle_collision_oracle(
            control,
            obstacle_scene,
        )
        return control

    receipt = run_real_shadow_acceptance(
        snapshot_provider=lambda: snapshot,
        trajectory_builder=lambda captured: _trajectory(
            captured,
            scene_hash=scene_revision,
        ),
        simulation_control_factory=build_control,
        model_binding_provider=_actual_binding,
        simulation_executor=execute_registered_shadow_trajectory,
        real_ports_factory=lambda trajectory: real_factory_calls.append(
            trajectory.semantic_content_hash
        ),
    )

    assert receipt.status is RealShadowAcceptanceStatus.SIMULATION_ACCEPTED_NO_MOTION
    assert receipt.simulation_receipt.status is ShadowSimulationStatus.COMPLETED
    assert receipt.real_motion_disposition is RealMotionDisposition.NONE
    assert receipt.terminal_hold_confirmed is True
    assert real_factory_calls == []


def test_capture_and_trajectory_binding_failures_are_typed_and_do_not_start_sim():
    sim_calls: list[str] = []
    capture, _ = _run(
        snapshot_provider=lambda: (_ for _ in ()).throw(RuntimeError("no fresh cut")),
        simulation_control_factory=lambda snapshot: sim_calls.append(
            snapshot.semantic_content_hash
        ),
    )
    assert capture.status is RealShadowAcceptanceStatus.CAPTURE_FAILED
    assert capture.real_motion_disposition is RealMotionDisposition.NONE
    assert sim_calls == []

    other = _snapshot(identity=_identity(robot_model_hash="other"))
    binding, _ = _run(trajectory_builder=lambda _snapshot: _trajectory(other))
    assert (
        binding.status
        is RealShadowAcceptanceStatus.TRAJECTORY_PREPARATION_FAILED
    )
    assert sim_calls == []


def test_simulation_initialization_error_never_requests_real_ports():
    real_factory_calls: list[str] = []
    receipt, _ = _run(
        simulation_control_factory=lambda _snapshot: (_ for _ in ()).throw(
            RuntimeError("initial state collision")
        ),
        real_replay_enabled=True,
        real_ports_factory=lambda trajectory: real_factory_calls.append(
            trajectory.semantic_content_hash
        ),
    )

    assert (
        receipt.status
        is RealShadowAcceptanceStatus.SIMULATION_INITIALIZATION_FAILED
    )
    assert receipt.real_motion_disposition is RealMotionDisposition.NONE
    assert real_factory_calls == []


def test_model_binding_error_does_not_construct_simulation() -> None:
    sim_calls: list[str] = []
    receipt, _ = _run(
        model_binding_provider=lambda *_args: (_ for _ in ()).throw(
            RuntimeError("compatibility expectation unavailable")
        ),
        simulation_control_factory=lambda snapshot: sim_calls.append(
            snapshot.semantic_content_hash
        ),
    )

    assert receipt.status is RealShadowAcceptanceStatus.MODEL_BINDING_FAILED
    assert receipt.real_motion_disposition is RealMotionDisposition.NONE
    assert sim_calls == []


def test_simulation_execution_error_holds_and_closes() -> None:
    control = _SimulationControl()
    receipt, _ = _run(
        control=control,
        simulation_executor=lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("simulation owner failed")
        ),
    )

    assert receipt.status is RealShadowAcceptanceStatus.SIMULATION_EXECUTION_FAILED
    assert receipt.real_motion_disposition is RealMotionDisposition.NONE
    assert control.hold_reasons == ["real shadow acceptance simulation error"]
    assert control.closed is True


def test_simulation_close_failure_blocks_real_replay():
    class _CloseFailureControl(_SimulationControl):
        def close(self):
            raise RuntimeError("scene close failed")

    real_factory_calls: list[str] = []
    receipt, _ = _run(
        control=_CloseFailureControl(),
        real_replay_enabled=True,
        real_ports_factory=lambda trajectory: real_factory_calls.append(
            trajectory.semantic_content_hash
        ),
    )

    assert (
        receipt.status is RealShadowAcceptanceStatus.SIMULATION_SHUTDOWN_FAILED
    )
    assert receipt.real_motion_disposition is RealMotionDisposition.NONE
    assert real_factory_calls == []


@pytest.mark.parametrize(
    "status",
    (
        ShadowSimulationStatus.PATH_COLLISION_BLOCKED,
        ShadowSimulationStatus.POINT_TRACKING_TIMEOUT,
        ShadowSimulationStatus.GRIPPER_HOLD_DRIFT,
        ShadowSimulationStatus.MODEL_IDENTITY_MISMATCH,
        ShadowSimulationStatus.TERMINAL_HOLD_FAILED,
    ),
)
def test_each_failed_simulation_status_blocks_real_replay(status):
    real_factory_calls: list[str] = []

    def simulate(*, trajectory, **_kwargs):
        return _simulation(trajectory, status=status)

    receipt, control = _run(
        simulation_executor=simulate,
        real_replay_enabled=True,
        real_ports_factory=lambda trajectory: real_factory_calls.append(
            trajectory.semantic_content_hash
        ),
    )

    assert receipt.status is RealShadowAcceptanceStatus.SIMULATION_REJECTED
    assert receipt.simulation_receipt.status is status
    assert receipt.real_motion_disposition is RealMotionDisposition.NONE
    assert control.closed is True
    assert real_factory_calls == []


def test_compatibility_failure_reason_codes_reach_application_receipt() -> None:
    def simulate(*, trajectory, **_kwargs):
        receipt = _simulation(
            trajectory,
            status=ShadowSimulationStatus.MODEL_IDENTITY_MISMATCH,
        )
        return replace(
            receipt,
            evidence_refs=(
                "mujoco-shadow-compatibility-reason:tool_tcp_mismatch",
                "mujoco-shadow-compatibility-reason:gripper_assignment_mismatch",
            ),
        )

    receipt, _ = _run(simulation_executor=simulate)

    assert receipt.status is RealShadowAcceptanceStatus.SIMULATION_REJECTED
    assert receipt.reason_codes == (
        "model_identity_mismatch",
        "gripper_assignment_mismatch",
        "tool_tcp_mismatch",
        "executed_point_count_mismatch",
    )


def test_completed_simulation_with_wrong_exact_binding_is_rejected():
    def simulate(*, trajectory, **_kwargs):
        receipt = _simulation(trajectory)
        return replace(receipt, trajectory_hash="trajectory:wrong")

    receipt, _ = _run(simulation_executor=simulate)

    assert receipt.status is RealShadowAcceptanceStatus.SIMULATION_REJECTED
    assert receipt.real_motion_disposition is RealMotionDisposition.NONE


def test_explicit_real_request_without_exact_authority_does_not_build_ports():
    real_factory_calls: list[str] = []
    receipt, _ = _run(
        real_replay_enabled=True,
        real_ports_factory=lambda trajectory: real_factory_calls.append(
            trajectory.semantic_content_hash
        ),
    )

    assert receipt.status is RealShadowAcceptanceStatus.REAL_REPLAY_REJECTED
    assert receipt.real_motion_disposition is RealMotionDisposition.NONE
    assert real_factory_calls == []


def test_current_real_scene_mismatch_rejects_before_real_ports_are_built():
    snapshot = _snapshot()
    trajectory = _trajectory(snapshot)
    simulation = _simulation(trajectory)
    authority = RegisteredTrajectoryExecutionAuthority(
        authority_id="authority:test",
        operator_confirmation_ref="operator:test",
        trajectory_hash=trajectory.semantic_content_hash,
        simulation_receipt_hash=simulation.semantic_content_hash,
        run_id="run:test",
        owner_epoch=1,
        issued_at=10.0,
        expires_at=10.1,
    )
    real_factory_calls: list[str] = []

    receipt, _ = _run(
        snapshot=snapshot,
        simulation_executor=lambda **_kwargs: simulation,
        real_replay_enabled=True,
        real_ports_factory=lambda value: real_factory_calls.append(
            value.semantic_content_hash
        ),
        current_scene_hash_provider=lambda: "scene:changed",
        authority=authority,
        consume_execution_authority=_consumption,
    )

    assert receipt.status is RealShadowAcceptanceStatus.REAL_REPLAY_REJECTED
    assert receipt.reason_codes == ("current_scene_hash_mismatch",)
    assert real_factory_calls == []


class _ReplayPorts:
    def __init__(self):
        self.hold_reasons: list[str] = []

    def current_snapshot(self):  # pragma: no cover - fake executor owns this test.
        raise AssertionError

    def begin_replay_session(self, consumption):
        return (f"session-begin:{consumption.semantic_content_hash}",)

    def end_replay_session(self, reason):
        return (f"session-end:{reason}",)

    def dispatch_point(self, *_args):  # pragma: no cover
        raise AssertionError

    def hold(self, reason):
        self.hold_reasons.append(reason)
        return RealSkillMotionHoldReceipt(
            succeeded=True,
            evidence_refs=("real-hold:test",),
        )


def test_exact_authority_delegates_to_existing_real_executor_and_reports_completion():
    observed: dict[str, object] = {}

    def real_executor(**kwargs):
        observed.update(kwargs)
        trajectory = kwargs["trajectory"]
        simulation = kwargs["simulation_receipt"]
        admission = ShadowReplayAdmissionReceipt(
            status=ShadowReplayAdmissionStatus.ADMISSIBLE_PREFLIGHT_ONLY,
            reason_codes=(),
            current_snapshot_hash=trajectory.initial_snapshot.semantic_content_hash,
            trajectory_hash=trajectory.semantic_content_hash,
            simulation_receipt_hash=simulation.semantic_content_hash,
            evaluated_at=10.01,
            maximum_arm_drift_rad=0.0,
            maximum_gripper_drift=0.0,
        )
        return RegisteredTrajectoryRealReplayReceipt(
            status=RegisteredTrajectoryRealReplayStatus.COMPLETED,
            admission=admission,
            executed_point_count=len(trajectory.points),
            terminal_hold_confirmed=True,
            real_motion_executed=True,
            evidence_refs=("real-replay:test",),
        )

    snapshot = _snapshot()
    trajectory = _trajectory(snapshot)
    simulation = _simulation(trajectory)
    authority = RegisteredTrajectoryExecutionAuthority(
        authority_id="authority:test",
        operator_confirmation_ref="operator:test",
        trajectory_hash=trajectory.semantic_content_hash,
        simulation_receipt_hash=simulation.semantic_content_hash,
        run_id="run:test",
        owner_epoch=1,
        issued_at=10.0,
        expires_at=10.1,
    )

    def simulate(**_kwargs):
        return simulation

    ports = _ReplayPorts()
    receipt, _ = _run(
        snapshot=snapshot,
        simulation_executor=simulate,
        real_replay_enabled=True,
        real_ports_factory=lambda received: ports
        if received.semantic_content_hash == trajectory.semantic_content_hash
        else None,
        authority=authority,
        consume_execution_authority=_consumption,
        real_replay_executor=real_executor,
    )

    assert receipt.status is RealShadowAcceptanceStatus.REAL_REPLAY_COMPLETED
    assert receipt.real_motion_disposition is RealMotionDisposition.EXECUTED
    assert receipt.terminal_hold_confirmed is True
    assert receipt.terminal_hold_domain == "real"
    assert observed["execution_enabled"] is True
    assert observed["current_snapshot_provider"] == ports.current_snapshot


def test_real_executor_exception_returns_unknown_motion_outcome():
    snapshot = _snapshot()
    trajectory = _trajectory(snapshot)
    simulation = _simulation(trajectory)
    authority = RegisteredTrajectoryExecutionAuthority(
        authority_id="authority:test",
        operator_confirmation_ref="operator:test",
        trajectory_hash=trajectory.semantic_content_hash,
        simulation_receipt_hash=simulation.semantic_content_hash,
        run_id="run:test",
        owner_epoch=1,
        issued_at=10.0,
        expires_at=10.1,
    )

    ports = _ReplayPorts()
    receipt, _ = _run(
        snapshot=snapshot,
        simulation_executor=lambda **_kwargs: simulation,
        real_replay_enabled=True,
        real_ports_factory=lambda _trajectory: ports,
        authority=authority,
        consume_execution_authority=_consumption,
        real_replay_executor=lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("transport lost after possible dispatch")
        ),
    )

    assert receipt.status is RealShadowAcceptanceStatus.REAL_REPLAY_FAILED
    assert receipt.real_motion_disposition is RealMotionDisposition.UNKNOWN
    assert receipt.terminal_hold_confirmed is True
    assert receipt.terminal_hold_domain == "unknown"
    assert ports.hold_reasons == ["real shadow acceptance replay executor error"]


def test_completed_real_receipt_with_wrong_point_count_is_not_accepted():
    observed_receipt: dict[str, RegisteredTrajectoryRealReplayReceipt] = {}

    def real_executor(**kwargs):
        trajectory = kwargs["trajectory"]
        simulation = kwargs["simulation_receipt"]
        admission = ShadowReplayAdmissionReceipt(
            status=ShadowReplayAdmissionStatus.ADMISSIBLE_PREFLIGHT_ONLY,
            reason_codes=(),
            current_snapshot_hash=trajectory.initial_snapshot.semantic_content_hash,
            trajectory_hash=trajectory.semantic_content_hash,
            simulation_receipt_hash=simulation.semantic_content_hash,
            evaluated_at=10.01,
            maximum_arm_drift_rad=0.0,
            maximum_gripper_drift=0.0,
        )
        result = RegisteredTrajectoryRealReplayReceipt(
            status=RegisteredTrajectoryRealReplayStatus.COMPLETED,
            admission=admission,
            executed_point_count=1,
            terminal_hold_confirmed=True,
            real_motion_executed=True,
            evidence_refs=("real-replay:incomplete",),
        )
        observed_receipt["value"] = result
        return result

    snapshot = _snapshot()
    trajectory = _trajectory(snapshot)
    simulation = _simulation(trajectory)
    authority = RegisteredTrajectoryExecutionAuthority(
        authority_id="authority:test",
        operator_confirmation_ref="operator:test",
        trajectory_hash=trajectory.semantic_content_hash,
        simulation_receipt_hash=simulation.semantic_content_hash,
        run_id="run:test",
        owner_epoch=1,
        issued_at=10.0,
        expires_at=10.1,
    )
    receipt, _ = _run(
        snapshot=snapshot,
        simulation_executor=lambda **_kwargs: simulation,
        real_replay_enabled=True,
        real_ports_factory=lambda _trajectory: _ReplayPorts(),
        authority=authority,
        consume_execution_authority=_consumption,
        real_replay_executor=real_executor,
    )

    assert observed_receipt
    assert receipt.status is RealShadowAcceptanceStatus.REAL_REPLAY_FAILED
    assert "real_executed_point_count_mismatch" in receipt.reason_codes
