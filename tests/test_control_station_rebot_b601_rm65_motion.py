from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from control_station.composition.rebot_b601_rm65_residual import B601RM65ResidualSource
from control_station.sources.rebot_b601_pose import (
    B601ReadOnlyPoseSource,
    B601ReadOnlyPoseSourceSpec,
)
from control_station.sources.rebot_b601_rm65_motion import (
    B601RM65MotionBinding,
    B601RM65MotionSession,
)
from hardware.rebot_b601_rs import (
    B601_RS_FEEDBACK_FRAME,
    B601HardwareState,
    B601JointState,
)
from robot_motion.action import (
    ExecutionMeasurementObservation,
    build_relative_delta_action_schema,
)
from robot_motion.coordinate_frames import (
    CoordinateFrameGraph,
    CoordinateFrameSpec,
    RigidFrameTransformSpec,
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
from teleop_runtime.action.human_residual import HumanResidualPoseAdapter

pytestmark = [pytest.mark.unit, pytest.mark.contract]


class _Clock:
    def __init__(self):
        self.value = 10.0

    def __call__(self):
        return self.value


class _Scene:
    def __init__(self):
        self.arm_specs = {"left": object(), "right": object()}
        self.gripper_specs = {}
        self.qpos = {side: np.zeros(6) for side in ("left", "right")}
        self.writes = []

    def get_arm_qpos(self, side):
        return self.qpos[side].copy()

    def step_kinematic_direct_qpos_debug(self, qpos_by_side, **kwargs):
        del kwargs
        self.qpos.update(
            {side: np.asarray(q).copy() for side, q in qpos_by_side.items()}
        )
        self.writes.append(qpos_by_side)


class _Planner:
    def reset(self, qpos_by_side):
        self.qpos = qpos_by_side
        return {"initialized": True}

    def plan_pose_reference(self, pose_by_side, **kwargs):
        del kwargs
        qref = {}
        for side, pose in pose_by_side.items():
            qref[side] = np.concatenate(
                (pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_rotvec())
            )
        return SimpleNamespace(
            valid=True,
            hold=False,
            status="tracking",
            q_ref_by_side=qref,
            target_endpoint_pose_world_by_side=pose_by_side,
            realized_endpoint_pose_world_by_side=pose_by_side,
            modification_reasons=(),
            metadata={"synthetic_planner": True},
        )


class _Backend:
    def __init__(self, clock):
        self.clock, self.sequence = clock, 0
        self.q = np.zeros(6)
        self.valid = True

    def poll_feedback(self, *, request_feedback, require_host_freshness):
        assert not request_feedback and require_host_freshness
        self.sequence += 1
        now = self.clock()
        joints = tuple(
            B601JointState(
                name=f"joint{i + 1}",
                motor_id=i + 1,
                feedback_id=0xFD,
                arbitration_id=i + 1,
                status_code=0,
                position_rad=float(self.q[i]),
                velocity_reported=0,
                torque_reported_nm=0,
                observed_at_sec=now,
                sample_sequence=self.sequence,
                valid=self.valid,
                stale=not self.valid,
                frame_id=B601_RS_FEEDBACK_FRAME,
                position_source="synthetic_host_fresh_state",
                status_sequence=self.sequence * 10 + i,
                receive_age_sec=0,
                host_receive_freshness_verified=True,
            )
            for i in range(6)
        )
        return B601HardwareState(
            sequence=self.sequence,
            receive_timestamp_sec=now,
            joints=joints,
            valid=self.valid,
            stale=not self.valid,
            freshness_verified=False,
            freshness_semantics="synthetic_host_receive_only",
            fault_reason=None,
            host_receive_freshness_verified=True,
        )


class _Model:
    arm_joint_names = tuple(f"joint{i}" for i in range(1, 7))
    end_effector_frame = "gripper_end"

    def forward_kinematics(self, q):
        pose = np.eye(4)
        pose[:3, 3] = q[:3]
        pose[:3, :3] = Rotation.from_rotvec(np.asarray(q)[3:]).as_matrix()
        return pose


def _source(runtime, clock):
    schema = build_relative_delta_action_schema(
        runtime.action_schema,
        component_ids=("right_arm",),
        schema_id="b601-right-residual",
    )
    frame = schema.payload_schema_by_component["right_arm"].reference_frame
    graph = CoordinateFrameGraph(
        graph_id="b601-to-rm65-test",
        graph_version="development",
        frames=tuple(
            CoordinateFrameSpec(frame_id=name)
            for name in ("base_link", "gripper_end", frame, "tool_tcp")
        ),
        transforms=(
            RigidFrameTransformSpec(
                transform_id="right-from-master",
                source_frame_id="base_link",
                target_frame_id=frame,
                matrix_target_source=np.eye(4),
            ),
            RigidFrameTransformSpec(
                transform_id="handle-from-tool",
                source_frame_id="tool_tcp",
                target_frame_id="gripper_end",
                matrix_target_source=np.eye(4),
            ),
        ),
    )
    backend = _Backend(clock)
    source = B601RM65ResidualSource(
        B601ReadOnlyPoseSource(
            backend,
            _Model(),
            spec=B601ReadOnlyPoseSourceSpec(clock_domain_id="local_monotonic"),
        ),
        HumanResidualPoseAdapter(output_action_schema=schema, frame_graph=graph),
    )
    return source, backend


def _measurement(runtime, clock, values=None):
    values = (
        values
        if values is not None
        else {
            component: (0, 0, 0, 0, 0, 0, 1)
            for component in runtime.action_schema.component_ids
        }
    )
    return ExecutionMeasurementObservation(
        measurement_id=f"target-{clock()}",
        source_measurement_hash="synthetic-target-measurement",
        control_domain_id=runtime.config.control_domain_id,
        measured_at=clock(),
        available_at=clock(),
        clock_domain_id=runtime.config.clock_domain_id,
        value_by_component=values,
        representation_by_component={
            component: "cartesian_pose" for component in values
        },
        schema_hash_by_component={
            component: runtime.action_schema.payload_schema_by_component[
                component
            ].semantic_content_hash
            for component in values
        },
    )


def _build_runtime(scene, clock, ik):
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
            timing=TimingSpec(
                sample_hz=20,
                policy_hz=20,
                reference_hz=60,
                ik_hz=60,
                command_hz=120,
                physics_hz=1000,
            ),
            actuation_profile=None,
            gripper_actuation_profile=None,
            realtime_worker=False,
            follow_qpos_alpha=1,
        ),
        monotonic_clock=clock,
    )
    runtime = MujocoDeployRuntime(
        rig,
        MujocoDeployConfig(
            producer_id="b601-human",
            initial_control_mode="human",
            clock_domain_id="local_monotonic",
            pose_frame_contract_hash="synthetic-rm65-arm-base-tool-tcp",
            decision_hz=20,
            enable_component_execution_handoff=True,
        ),
        pose_resolver=MujocoCoupledPoseToQposResolver(ik),
        clock=clock,
    )
    return runtime


@pytest.fixture
def setup_motion():
    clock, scene = _Clock(), _Scene()
    runtime = _build_runtime(scene, clock, RM65CanonicalIK(scene, _Planner()))
    source, backend = _source(runtime, clock)
    binding = B601RM65MotionBinding(
        runtime,
        source,
        maximum_source_age_sec=0.15,
        maximum_measurement_age_sec=0.1,
        maximum_translation_m=0.1,
        maximum_rotation_rad=0.3,
        enabled=True,
    )
    runtime.start()
    projection = source.poll_pose()
    master = source.capture_anchor(
        projection,
        anchor_generation=0,
        captured_at=clock(),
        captured_clock_domain_id="local_monotonic",
    )
    anchor = binding.capture_anchor(master, _measurement(runtime, clock))
    yield clock, scene, runtime, source, backend, binding, anchor
    runtime.close()


def _step(setup_motion, q=(0.01, 0, 0, 0, 0, 0.05)):
    clock, _, _, source, backend, _, anchor = setup_motion
    clock.value += 0.05
    backend.q = np.asarray(q)
    projection = source.poll_pose()
    return source.adapt(
        projection,
        anchor=anchor.master_anchor,
        adapted_at=clock(),
        adapted_clock_domain_id="local_monotonic",
    )


def _submit(setup_motion, step):
    clock, _, _, _, _, binding, anchor = setup_motion
    return binding.submit(
        step, anchor, target_effective_at=clock() + 0.01, effective_until=clock() + 0.1
    )


def test_admits_human_not_policy_and_only_target_dispatches(setup_motion):
    clock, scene, runtime, _, _, _, _ = setup_motion
    receipt = _submit(setup_motion, _step(setup_motion))
    assert receipt.accepted, receipt.message
    assert receipt.proposal.producer_kind == "human"
    assert receipt.proposal.policy_output_id is None
    assert receipt.proposal.chunk_id is None
    assert not scene.writes
    clock.value += 0.01
    outcome = runtime.step(now=clock(), physics_ticks_after_dispatch=12)
    assert outcome is not None
    assert scene.writes, (outcome.dispatch_receipts, outcome.target_completion_receipt)
    assert np.linalg.norm(scene.qpos["right"]) > 0
    np.testing.assert_allclose(scene.qpos["left"], 0)
    assert runtime.composition_port is None


@pytest.mark.parametrize(
    "failure",
    [
        "duplicate",
        "stale",
        "future",
        "invalid",
        "translation",
        "rotation",
        "expiry",
        "wrong_anchor",
        "wrong_schema",
        "wrong_emission",
    ],
)
def test_rejection_cancels_old_reference_and_cannot_auto_resume(setup_motion, failure):
    clock, scene, runtime, _source_unused, backend, binding, anchor = setup_motion
    previous = _step(setup_motion)
    assert _submit(setup_motion, previous).accepted
    if failure == "duplicate":
        rejected = previous
    else:
        if failure == "invalid":
            backend.valid = False
        rejected = _step(
            setup_motion,
            q=(0.11, 0, 0, 0, 0, 0)
            if failure == "translation"
            else (0, 0, 0, 0.31, 0, 0)
            if failure == "rotation"
            else (0.01, 0, 0, 0, 0, 0),
        )
    if failure == "stale":
        clock.value += 0.2
    if failure == "future":
        clock.value -= 0.03
    if failure == "wrong_anchor":
        replacement = replace(anchor.master_anchor, anchor_generation=1)
        setup_motion = (*setup_motion[:6], replace(anchor, master_anchor=replacement))
    if failure == "wrong_emission":
        rejected = replace(
            rejected,
            projection=replace(
                rejected.projection, source_emission_hash="unbound-emission"
            ),
        )
    if failure == "wrong_schema":
        rejected = replace(
            rejected,
            adaptation=replace(
                rejected.adaptation,
                receipt=replace(
                    rejected.adaptation.receipt, adapter_contract_hash="other-source"
                ),
            ),
        )
    if failure == "expiry":
        result = binding.submit(
            rejected, anchor, target_effective_at=clock(), effective_until=clock() + 1
        )
    else:
        result = _submit(setup_motion, rejected)
    assert not result.accepted
    assert (
        runtime.authoritative_port.capture_execution_gate_snapshot(now=clock()).status
        == "hold"
    )
    clock.value += 0.1
    runtime.step(now=clock())
    assert not scene.writes
    backend.valid = True
    assert not _submit(setup_motion, _step(setup_motion)).accepted


def test_stationary_source_has_zero_delta_without_accumulation(setup_motion):
    clock, _, _, _, _, _, _ = setup_motion
    for _ in range(4):
        receipt = _submit(setup_motion, _step(setup_motion, q=(0, 0, 0, 0, 0, 0)))
        assert receipt.accepted, receipt.message
        np.testing.assert_allclose(
            receipt.proposal.action.values_by_component["right_arm"].values,
            (0, 0, 0, 0, 0, 0, 1),
        )
        assert receipt.proposal.produced_at == clock()


def _session(setup_motion, measure_target=None):
    clock, _, runtime, _, _, binding, _ = setup_motion
    # The fixture already polled at its initial time; supply a genuinely new
    # host sample, preserving the source's monotonicity contract.
    clock.value += 0.01
    return B601RM65MotionSession(
        binding,
        measure_target=measure_target or (lambda: _measurement(runtime, clock)),
    )


def _session_step(session, clock):
    clock.value += 0.05
    return session.step(
        target_effective_at=clock() + 0.01, effective_until=clock() + 0.1
    )


def test_session_clutch_cancels_pending_and_recenter_has_zero_delta(setup_motion):
    clock, scene, runtime, _, backend, _, _ = setup_motion
    values = {
        name: (0.02, 0, 0, 0, 0, 0, 1) for name in runtime.action_schema.component_ids
    }
    session = _session(setup_motion, lambda: _measurement(runtime, clock, values))
    initial = session.activate()
    backend.q[0] = 0.01
    assert _session_step(session, clock).accepted
    session.engage_clutch()
    assert session.state == "clutched" and session.anchor is None
    seq = backend.sequence
    for _ in range(3):
        assert not _session_step(session, clock).accepted
        runtime.step(now=clock())
    assert backend.sequence == seq and not scene.writes
    # Reposition the master and independently measure the follower's current pose.
    backend.q[0] = 0.5
    values["right_arm"] = (0.03, 0, 0, 0, 0, 0, 1)
    replacement = session.release_clutch()
    assert (
        replacement.master_anchor.anchor_generation
        > initial.master_anchor.anchor_generation
    )
    assert replacement.gate_epoch != initial.gate_epoch
    receipt = _session_step(session, clock)
    assert receipt.accepted, receipt.message
    for name, value in values.items():
        np.testing.assert_allclose(
            receipt.proposal.action.values_by_component[name].values, value
        )
    assert not scene.writes  # Session does not dispatch even after a valid recenter.


@pytest.mark.parametrize("failure", ["stale", "invalid_master", "wrong_frame"])
def test_session_failed_recenter_never_reopens_gate(setup_motion, failure, monkeypatch):
    clock, _, runtime, _, backend, _, _ = setup_motion
    measurement = _measurement(runtime, clock)
    session = _session(setup_motion, lambda: measurement)
    session.activate()
    session.engage_clutch()
    clock.value += 0.05
    if failure == "stale":
        clock.value += 0.2
    elif failure == "invalid_master":
        backend.valid = False
    else:
        measurement = replace(measurement, control_domain_id="other-target")
    transitions = []
    original = runtime.transition_execution_gate

    def record(status, *, reason):
        transitions.append(status)
        return original(status, reason=reason)

    monkeypatch.setattr(runtime, "transition_execution_gate", record)
    with pytest.raises((ValueError, RuntimeError)):
        session.release_clutch()
    assert session.state == "clutched" and session.anchor is None
    assert "ready" not in transitions
    assert runtime.authoritative_port.capture_execution_gate_snapshot().status == "hold"


def test_session_rejected_tick_requires_explicit_fresh_activation(setup_motion):
    clock, scene, runtime, _, backend, _, _ = setup_motion
    session = _session(setup_motion)
    initial = session.activate()
    backend.q[0] = 0.11
    assert not _session_step(session, clock).accepted
    assert session.state == "held" and session.anchor is None
    backend.q[0] = 0
    sequence = backend.sequence
    assert not _session_step(session, clock).accepted
    assert backend.sequence == sequence and not scene.writes
    replacement = session.activate()
    assert replacement.gate_epoch != initial.gate_epoch
    assert _session_step(session, clock).accepted
    assert not scene.writes
    session.deactivate()
    runtime.step(now=clock())
    assert not scene.writes and session.state == "inactive"


@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt])
def test_session_interrupted_poll_withdraws_reference_and_reraises(
    setup_motion, monkeypatch, error_type
):
    clock, scene, runtime, source, _, _, _ = setup_motion
    session = _session(setup_motion)
    session.activate()
    assert _session_step(session, clock).accepted

    def fail():
        raise error_type("poll interrupted")

    monkeypatch.setattr(source, "poll_pose", fail)
    with pytest.raises(error_type):
        _session_step(session, clock)
    assert session.state == "held" and session.anchor is None
    runtime.step(now=clock())
    assert not scene.writes


def test_session_disabled_has_no_source_or_target_effect(setup_motion):
    clock, _, runtime, source, backend, _, _ = setup_motion
    binding = B601RM65MotionBinding(
        runtime,
        source,
        maximum_source_age_sec=0.15,
        maximum_measurement_age_sec=0.1,
        maximum_translation_m=0.1,
        maximum_rotation_rad=0.3,
    )
    session = B601RM65MotionSession(
        binding, measure_target=lambda: pytest.fail("must not measure")
    )
    sequence = backend.sequence
    session.engage_clutch()
    session.deactivate()
    with pytest.raises(RuntimeError):
        session.activate()
    assert not _session_step(session, clock).accepted
    assert backend.sequence == sequence
    assert (
        runtime.authoritative_port.capture_execution_gate_snapshot().status == "ready"
    )


@pytest.mark.parametrize("gate_status", ["fault", "transitioning"])
def test_session_cannot_downgrade_or_reopen_lower_gate(setup_motion, gate_status):
    _, _, runtime, _, _, _, _ = setup_motion
    session = _session(setup_motion)
    runtime.transition_execution_gate(gate_status, reason="lower_owner")
    with pytest.raises(RuntimeError, match="not ready"):
        session.activate()
    session.engage_clutch()
    assert session.anchor is None
    assert (
        runtime.authoritative_port.capture_execution_gate_snapshot().status
        == gate_status
    )


def test_disabled_binding_does_not_change_execution_gate(setup_motion):
    _, _, runtime, source, _, _, anchor = setup_motion
    binding = B601RM65MotionBinding(
        runtime,
        source,
        maximum_source_age_sec=0.15,
        maximum_measurement_age_sec=0.1,
        maximum_translation_m=0.1,
        maximum_rotation_rad=0.3,
    )
    assert not binding.submit(
        _step(setup_motion), anchor, target_effective_at=10.06, effective_until=10.15
    ).accepted
    assert (
        runtime.authoritative_port.capture_execution_gate_snapshot().status == "ready"
    )


def test_stale_target_or_master_cannot_establish_anchor(setup_motion):
    clock, _, _, _, _, binding, anchor = setup_motion
    clock.value += 0.2
    with pytest.raises(ValueError, match="stale"):
        binding.capture_anchor(anchor.master_anchor, anchor.target_measurement)


def test_anchor_cannot_substitute_unmeasured_target(setup_motion):
    *_, anchor = setup_motion
    values = {
        name: value.values
        for name, value in anchor.target_action.values_by_component.items()
    }
    values["right_arm"] = (0.5, 0, 0, 0, 0, 0, 1)
    with pytest.raises(ValueError, match="exactly match"):
        replace(
            anchor,
            target_action=replace(anchor.target_action, values_by_component=values),
        )


def test_hold_resume_requires_new_measured_anchors(setup_motion):
    clock, _, runtime, source, _, binding, _anchor_unused = setup_motion
    runtime.transition_execution_gate("hold", reason="operator_clutch")
    runtime.transition_execution_gate("ready", reason="operator_recenter")
    assert not _submit(setup_motion, _step(setup_motion)).accepted
    runtime.transition_execution_gate("ready", reason="explicit_rearm")
    clock.value += 0.05
    master = source.capture_anchor(
        source.poll_pose(),
        anchor_generation=1,
        captured_at=clock(),
        captured_clock_domain_id="local_monotonic",
    )
    replacement = binding.capture_anchor(master, _measurement(runtime, clock))
    recentered = (*setup_motion[:6], replacement)
    receipt = _submit(recentered, _step(recentered, q=(0.01, 0, 0, 0, 0, 0.05)))
    assert receipt.accepted, receipt.message


def test_rotation_is_baseline_local_and_translation_reference_additive(setup_motion):
    clock, _, runtime, _, _, binding, anchor = setup_motion
    quaternion = Rotation.from_euler("y", 0.7).as_quat()
    values = {
        name: value.values
        for name, value in anchor.target_action.values_by_component.items()
    }
    values["right_arm"] = np.concatenate(((0.2, 0.1, 0.3), quaternion))
    measured = _measurement(runtime, clock, values)
    anchor = binding.capture_anchor(anchor.master_anchor, measured)
    setup = (*setup_motion[:6], anchor)
    step = _step(setup)
    receipt = _submit(setup, step)
    assert receipt.accepted, receipt.message
    result = receipt.proposal.action.values_by_component["right_arm"].values
    np.testing.assert_allclose(result[:3], (0.21, 0.1, 0.3), atol=1e-7)
    expected = (
        Rotation.from_quat(quaternion).as_matrix()
        @ Rotation.from_rotvec((0, 0, 0.05)).as_matrix()
    )
    np.testing.assert_allclose(
        Rotation.from_quat(result[3:]).as_matrix(), expected, atol=1e-7
    )


@pytest.mark.optional
def test_actual_mujoco_rm65_canonical_ik_and_measured_motion():
    """Real model/IK, synthetic B601 FK; no claim about B601 hardware."""
    pytest.importorskip("mujoco")
    pytest.importorskip("pinocchio")
    if not Path("assets/rm65_inspire_gripper_v3").is_dir():
        pytest.skip("Reviewed RM65 assets are not bound.")
    from robot_motion.rm65 import (
        DEFAULT_LEFT_ARM_QPOS,
        DEFAULT_RIGHT_ARM_QPOS,
        RM65PositionCollisionPlannerConfig,
        endpoint_to_tool_transform,
        load_rm65_tool_profile,
    )
    from sim.contracts import SimulationContract
    from sim.mujoco.scene import MujocoScene

    scene = MujocoScene.from_contract(
        SimulationContract.load(Path("sim/configs/rm65_bimanual_physics_v1.json"))
    )
    initial = {
        "left": DEFAULT_LEFT_ARM_QPOS.copy(),
        "right": DEFAULT_RIGHT_ARM_QPOS.copy(),
    }
    scene.set_qpos_by_side(initial)
    profiles = {
        side: Path(
            f"experiments/sim_point_grasp/configs/rm65_{side}_DMTAC_M_4b2_cad_tool_tcp.json"
        )
        for side in ("left", "right")
    }
    ik = RM65CanonicalIK.from_tool_profiles(
        scene,
        profiles,
        planner_config=RM65PositionCollisionPlannerConfig(reference_hz=20),
    )
    values = {}
    for side in ("left", "right"):
        pose = (
            np.linalg.inv(scene.body_pose(f"{side}_arm_base_link"))
            @ scene.body_pose(f"{side}_arm_link6")
            @ endpoint_to_tool_transform(
                load_rm65_tool_profile(profiles[side], side=side)
            )
        )
        values[f"{side}_arm"] = np.concatenate(
            (pose[:3, 3], Rotation.from_matrix(pose[:3, :3].copy()).as_quat())
        )
    clock = _Clock()
    runtime = _build_runtime(scene, clock, ik)
    source, backend = _source(runtime, clock)
    binding = B601RM65MotionBinding(
        runtime,
        source,
        maximum_source_age_sec=0.15,
        maximum_measurement_age_sec=0.1,
        maximum_translation_m=0.1,
        maximum_rotation_rad=0.3,
        enabled=True,
    )
    runtime.start()
    try:
        master = source.capture_anchor(
            source.poll_pose(),
            anchor_generation=0,
            captured_at=clock(),
            captured_clock_domain_id="local_monotonic",
        )
        anchor = binding.capture_anchor(master, _measurement(runtime, clock, values))
        setup = (clock, scene, runtime, source, backend, binding, anchor)
        for _ in range(5):
            admitted = _submit(setup, _step(setup, q=(0.005, 0, 0, 0, 0, 0.01)))
            assert admitted.accepted, admitted.message
            clock.value += 0.01
            result = runtime.step(now=clock(), physics_ticks_after_dispatch=50)
            assert result.status == "dispatched", (
                result.status,
                result.resolution_failure,
                result.dispatch_receipts,
            )
            assert result.target_completion_receipt is not None
            assert all(
                receipt.accepted for receipt in result.dispatch_receipts.values()
            )
        measured = scene.get_arm_qpos("right")
        assert np.all(np.isfinite(measured))
        assert np.linalg.norm(measured - initial["right"]) > 1e-5
        assert np.linalg.norm(measured - initial["right"]) < 0.2
    finally:
        runtime.close()
