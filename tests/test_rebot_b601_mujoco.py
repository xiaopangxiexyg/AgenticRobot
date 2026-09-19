from __future__ import annotations

import ast
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import sim.mujoco.rebot_b601_rs.demo as b601_demo
from robot_motion.rebot_b601_rs import (
    B601_RS_ARM_JOINT_NAMES,
    B601_RS_END_EFFECTOR_FRAME,
    B601_RS_URDF_RELATIVE_PATH,
    B601DampedLeastSquaresIK,
    B601PinocchioModel,
)
from sim.mujoco.rebot_b601_rs import (
    B601_RS_DEFAULT_ARM_QPOS,
    B601_RS_DEFAULT_SIM_MAX_MEASURED_VELOCITY_RAD_S,
    B601_RS_DEFAULT_SIM_MEASURED_VELOCITY_MARGIN_RATIO,
    B601_RS_DEFAULT_SIM_TARGET_MAX_VELOCITY_RAD_S,
    B601_RS_GRIPPER_OPEN_QPOS,
    B601_RS_SIM_ARM_KEY,
    B601JointPositionCommand,
    B601PositionDriver,
    b601_sim_measured_velocity_limit_rad_s,
    build_b601_mjcf_xml,
    build_b601_mujoco_scene,
)
from sim.mujoco.rebot_b601_rs.demo import run_demo
from sim.mujoco.rebot_b601_rs.scene import b601_position_actuator_name

REPO_ROOT = Path(__file__).resolve().parents[1]
REBOT_SOURCE_ROOT = REPO_ROOT / "hardware" / "reBotArm_control_py"
pytestmark = [pytest.mark.optional, pytest.mark.contract]


@pytest.fixture(scope="module")
def model_and_scene():
    pytest.importorskip("mujoco")
    if not (REBOT_SOURCE_ROOT / B601_RS_URDF_RELATIVE_PATH).is_file():
        pytest.skip("initialize the pinned reBotArm_control_py gitlink")
    pinocchio_model = B601PinocchioModel.from_rebot_source_root(REBOT_SOURCE_ROOT)
    scene = build_b601_mujoco_scene(REBOT_SOURCE_ROOT)
    return pinocchio_model, scene


def _command(sequence: int, qpos, timestamp: float) -> B601JointPositionCommand:
    return B601JointPositionCommand(
        sequence=sequence,
        source_timestamp_sec=timestamp,
        receive_timestamp_sec=timestamp,
        qpos=tuple(qpos),
    )


@pytest.mark.parametrize("value", [True, 0.0, -1.0, float("nan")])
def test_measured_velocity_envelope_rejects_invalid_target_limits(value) -> None:
    with pytest.raises((TypeError, ValueError), match="finite and positive"):
        b601_sim_measured_velocity_limit_rad_s(value)


def test_command_rejects_boolean_sequence() -> None:
    with pytest.raises(TypeError, match="non-negative integer"):
        B601JointPositionCommand(
            sequence=True,
            source_timestamp_sec=0.0,
            receive_timestamp_sec=0.0,
            qpos=(0.0, 1.0, 1.0, 0.0, 0.0, 0.0),
        )


def test_compiled_scene_preserves_full_chain_and_six_arm_actuators(
    model_and_scene,
) -> None:
    _pinocchio_model, scene = model_and_scene
    mujoco = scene.mujoco

    assert scene.model.nq == 8
    assert scene.model.nv == 8
    assert scene.model.nbody == 11
    assert scene.model.nu == 6
    assert scene.model.opt.timestep == pytest.approx(0.001)
    np.testing.assert_array_equal(scene.model.opt.gravity, [0.0, 0.0, 0.0])
    assert int(scene.model.opt.integrator) == int(
        mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    )
    assert scene.arm_specs[B601_RS_SIM_ARM_KEY].joint_names == (B601_RS_ARM_JOINT_NAMES)
    assert (
        scene.arm_specs[B601_RS_SIM_ARM_KEY].controlled_body
        == B601_RS_END_EFFECTOR_FRAME
    )
    assert scene.has_body("base_link")
    assert scene.has_body(B601_RS_END_EFFECTOR_FRAME)
    assert scene.gripper_specs == {}
    assert scene.scene_spec is None


def test_mjcf_references_pinned_meshes_in_place_without_asset_copies() -> None:
    root = ET.fromstring(build_b601_mjcf_xml(REBOT_SOURCE_ROOT))
    mesh_paths = [
        Path(element.attrib["file"]).resolve(strict=True)
        for element in root.findall("./asset/mesh")
    ]
    resolved_source = REBOT_SOURCE_ROOT.resolve(strict=True)

    assert len(mesh_paths) == 23
    assert len(set(mesh_paths)) == 23
    for mesh_path in mesh_paths:
        assert mesh_path.is_file()
        mesh_path.relative_to(resolved_source)


def test_actuator_ranges_are_exact_urdf_limits_and_provisional_efforts(
    model_and_scene,
) -> None:
    pinocchio_model, scene = model_and_scene
    lower, upper = pinocchio_model.arm_position_limits()
    actuator_ids = np.asarray(
        [
            scene.mujoco.mj_name2id(
                scene.model,
                scene.mujoco.mjtObj.mjOBJ_ACTUATOR,
                b601_position_actuator_name(name),
            )
            for name in B601_RS_ARM_JOINT_NAMES
        ],
        dtype=np.int32,
    )

    assert np.all(actuator_ids >= 0)
    np.testing.assert_allclose(scene.model.actuator_ctrlrange[actuator_ids, 0], lower)
    np.testing.assert_allclose(scene.model.actuator_ctrlrange[actuator_ids, 1], upper)
    np.testing.assert_allclose(
        scene.model.actuator_forcerange[actuator_ids, 1],
        [36.0, 36.0, 36.0, 14.0, 14.0, 14.0],
    )
    np.testing.assert_allclose(
        scene.model.actuator_forcerange[actuator_ids, 0],
        [-36.0, -36.0, -36.0, -14.0, -14.0, -14.0],
    )


def test_pinocchio_and_mujoco_fk_match_over_deterministic_interior_sweep(
    model_and_scene,
) -> None:
    pinocchio_model, scene = model_and_scene
    lower, upper = pinocchio_model.arm_position_limits()
    rng = np.random.default_rng(601)

    for _ in range(16):
        qpos = lower + (0.15 + 0.70 * rng.random(6)) * (upper - lower)
        scene.set_arm_qpos(B601_RS_SIM_ARM_KEY, qpos)
        np.testing.assert_allclose(
            scene.body_pose(B601_RS_END_EFFECTOR_FRAME),
            pinocchio_model.forward_kinematics(qpos),
            atol=5e-11,
            rtol=0.0,
        )


def test_driver_schedules_500hz_commands_against_1000hz_physics(
    model_and_scene,
) -> None:
    _pinocchio_model, scene = model_and_scene
    driver = B601PositionDriver(scene)
    driver.reset()

    ticks = [driver.step_physics() for _ in range(1000)]

    assert sum(tick.command_due for tick in ticks) == 500
    assert ticks[0].command_due is True
    assert ticks[1].command_due is False
    assert ticks[2].command_due is True
    assert driver.physics_time_sec == pytest.approx(1.0)
    assert all(tick.state_valid for tick in ticks)


def test_driver_rejects_early_stale_and_out_of_limit_commands_without_mutation(
    model_and_scene,
) -> None:
    _pinocchio_model, scene = model_and_scene
    driver = B601PositionDriver(scene)
    initial = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    first_target = initial + np.array([0.05, 0.02, 0.01, 0.0, 0.0, 0.0])
    second_target = first_target + 0.01
    driver.reset(initial)

    accepted = driver.step_physics(_command(1, first_target, 1.0))
    early = driver.step_physics(_command(2, second_target, 1.001))
    stale = driver.step_physics(_command(1, second_target, 1.002))
    driver.step_physics()
    stale_source = driver.step_physics(_command(2, second_target, 0.5))
    driver.step_physics()
    stale_receive = driver.step_physics(
        B601JointPositionCommand(
            sequence=2,
            source_timestamp_sec=1.1,
            receive_timestamp_sec=0.5,
            qpos=tuple(second_target),
        )
    )
    driver.step_physics()
    outside = second_target.copy()
    outside[0] = 3.0
    out_of_limit = driver.step_physics(_command(3, outside, 2.0))

    assert accepted.command_applied is True
    assert early.command_applied is False
    assert early.rejection_reason == "command_before_deadline"
    assert stale.command_applied is False
    assert stale.rejection_reason == "stale_sequence"
    assert stale_source.rejection_reason == "stale_source_timestamp"
    assert stale_receive.rejection_reason == "stale_receive_timestamp"
    assert out_of_limit.command_applied is False
    assert out_of_limit.rejection_reason == "joint_limit_violation"
    assert out_of_limit.accepted_sequence == 1
    assert out_of_limit.accepted_source_timestamp_sec == pytest.approx(1.0)
    assert out_of_limit.accepted_receive_timestamp_sec == pytest.approx(1.0)
    assert out_of_limit.accepted_frame_id == "b601_rs/joint_position"
    np.testing.assert_allclose(out_of_limit.target_qpos, first_target)


def test_driver_slew_limits_final_target_and_continues_without_repeated_input(
    model_and_scene,
) -> None:
    _pinocchio_model, scene = model_and_scene
    driver = B601PositionDriver(scene)
    initial = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    target = initial + np.array([0.2, 0.1, 0.1, 0.1, -0.1, 0.2])
    driver.reset(initial)

    first = driver.step_physics(_command(1, target, 0.0))
    held = driver.step_physics()
    continued = driver.step_physics()
    allowed_step_rad = B601_RS_DEFAULT_SIM_TARGET_MAX_VELOCITY_RAD_S / 500.0

    assert first.command_applied is True
    np.testing.assert_allclose(first.target_qpos, target)
    np.testing.assert_allclose(held.target_qpos, target)
    np.testing.assert_allclose(continued.target_qpos, target)
    assert first.slew_limited is True
    assert first.actuator_target_changed is True
    np.testing.assert_array_less(
        np.abs(np.asarray(first.actuator_target_qpos) - initial),
        np.full(6, allowed_step_rad + 1e-12),
    )
    np.testing.assert_allclose(held.actuator_target_qpos, first.actuator_target_qpos)
    assert held.actuator_target_changed is False
    assert held.slew_limited is False
    assert held.slew_limit_ratio == pytest.approx(0.0)
    np.testing.assert_array_less(
        np.abs(
            np.asarray(continued.actuator_target_qpos)
            - np.asarray(held.actuator_target_qpos)
        ),
        np.full(6, allowed_step_rad + 1e-12),
    )
    assert continued.actuator_target_changed is True


def test_latest_target_preempts_inflight_slew_without_queue(model_and_scene) -> None:
    _pinocchio_model, scene = model_and_scene
    driver = B601PositionDriver(scene)
    initial = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    positive = initial.copy()
    positive[0] += 0.2
    negative = initial.copy()
    negative[0] -= 0.2
    driver.reset(initial)

    positive_tick = driver.step_physics(_command(1, positive, 0.0))
    driver.step_physics()
    preempted = driver.step_physics(_command(2, negative, 0.002))
    driver.step_physics()
    continued = driver.step_physics()

    assert preempted.command_applied is True
    np.testing.assert_allclose(preempted.target_qpos, negative)
    assert preempted.actuator_target_qpos[0] < positive_tick.actuator_target_qpos[0]
    assert (
        abs(preempted.actuator_target_qpos[0] - positive_tick.actuator_target_qpos[0])
        <= B601_RS_DEFAULT_SIM_TARGET_MAX_VELOCITY_RAD_S / 500.0 + 1e-12
    )
    assert continued.actuator_target_qpos[0] < preempted.actuator_target_qpos[0]


def test_large_position_step_tracks_without_target_spike_or_state_excursion(
    model_and_scene,
) -> None:
    _pinocchio_model, scene = model_and_scene
    driver = B601PositionDriver(scene)
    initial = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    target = np.array([0.8, 1.4, 1.4, 0.4, -0.4, 0.6])
    driver.reset(initial)

    previous_actuator_target = initial.copy()
    max_actuator_step = 0.0
    peak_measured_velocity = 0.0
    tick = None
    for physics_tick in range(2500):
        tick = driver.step_physics(
            _command(1, target, 0.0) if physics_tick == 0 else None
        )
        actuator_target = np.asarray(tick.actuator_target_qpos, dtype=float)
        max_actuator_step = max(
            max_actuator_step,
            float(np.max(np.abs(actuator_target - previous_actuator_target))),
        )
        peak_measured_velocity = max(
            peak_measured_velocity,
            float(np.max(np.abs(tick.measured_qvel))),
        )
        previous_actuator_target = actuator_target

    assert tick is not None
    assert (
        max_actuator_step
        <= B601_RS_DEFAULT_SIM_TARGET_MAX_VELOCITY_RAD_S / 500.0 + 1e-12
    )
    assert peak_measured_velocity < B601_RS_DEFAULT_SIM_MAX_MEASURED_VELOCITY_RAD_S
    assert tick.driver_faulted is False
    assert tick.state_within_limits is True
    np.testing.assert_allclose(tick.measured_qpos, target, atol=2e-3)


def test_measured_limit_excursion_latches_driver_fault_until_reset(
    model_and_scene,
) -> None:
    _pinocchio_model, scene = model_and_scene
    driver = B601PositionDriver(scene)
    initial = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    driver.reset(initial)
    joint_id = scene.joint_id(B601_RS_ARM_JOINT_NAMES[0])
    qpos_address = scene.joint_qpos_addr(B601_RS_ARM_JOINT_NAMES[0])
    scene.data.qpos[qpos_address] = scene.model.jnt_range[joint_id, 1] + 0.01
    scene.forward()

    faulted = driver.step_physics()
    rejected = driver.step_physics(_command(1, initial, 0.001))

    assert faulted.state_valid is True
    assert faulted.state_within_limits is False
    assert faulted.driver_faulted is True
    assert faulted.fault_reason is not None
    assert "joint1:above_upper" in faulted.fault_reason
    assert faulted.plant_advanced is True
    assert rejected.command_applied is False
    assert rejected.rejection_reason == "driver_fault_latched"
    assert rejected.driver_faulted is True

    scene.data.qpos[qpos_address] = np.nan
    driver_time_before = driver.physics_time_sec
    mujoco_time_before = float(scene.data.time)
    non_finite = driver.step_physics()
    assert non_finite.state_valid is False
    assert non_finite.fault_reason == "non_finite_simulated_state"
    assert non_finite.plant_advanced is False
    assert driver.physics_time_sec == pytest.approx(driver_time_before)
    assert float(scene.data.time) == pytest.approx(mujoco_time_before)

    driver.reset(initial)
    recovered = driver.step_physics(_command(1, initial + 0.01, 0.0))
    assert recovered.command_applied is True
    assert recovered.driver_faulted is False
    assert recovered.state_within_limits is True


def test_non_finite_state_fault_stops_native_plant_until_reset(
    model_and_scene,
) -> None:
    _pinocchio_model, scene = model_and_scene
    driver = B601PositionDriver(scene)
    initial = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    driver.reset(initial)
    qpos_address = scene.joint_qpos_addr(B601_RS_ARM_JOINT_NAMES[0])
    scene.data.qpos[qpos_address] = np.nan
    driver_time_before = driver.physics_time_sec
    mujoco_time_before = float(scene.data.time)

    faulted = driver.step_physics()

    assert faulted.state_valid is False
    assert faulted.state_within_limits is False
    assert faulted.driver_faulted is True
    assert faulted.fault_reason == "non_finite_simulated_state"
    assert faulted.plant_advanced is False
    assert driver.physics_time_sec == pytest.approx(driver_time_before)
    assert float(scene.data.time) == pytest.approx(mujoco_time_before)

    driver.reset(initial)
    recovered = driver.step_physics()
    assert recovered.state_valid is True
    assert recovered.driver_faulted is False
    assert recovered.plant_advanced is True


def test_driver_metadata_reports_provisional_slew_and_state_guard() -> None:
    scene = build_b601_mujoco_scene(
        REBOT_SOURCE_ROOT,
        gravity_mps2=(0.0, 0.0, -9.81),
    )
    metadata = B601PositionDriver(scene).execution_metadata

    np.testing.assert_allclose(metadata["gravity_mps2"], [0.0, 0.0, -9.81])
    assert metadata["gravity_compensation"] == "not_implemented_m2"
    assert metadata["target_slew_limit"] == "per_joint_velocity_limit_at_command_rate"
    assert metadata["state_limit_guard"] == "latched_simulated_joint_position_guard"
    assert metadata["state_limit_tolerance_rad"] == pytest.approx(1e-4)
    assert metadata["command_scope"] == "modest_visualization_targets_only"
    assert (
        metadata["bounded_target_stage"]
        == "simulation_only_position_target_conditioning"
    )
    profile = metadata["target_slew_profile"]
    np.testing.assert_allclose(
        profile["max_velocity_deg_s"],
        np.degrees([B601_RS_DEFAULT_SIM_TARGET_MAX_VELOCITY_RAD_S] * 6),
    )
    assert profile["metadata"]["hardware_safety_claim"] is False
    measured_envelope = metadata["measured_velocity_envelope"]
    assert measured_envelope["max_velocity_rad_s"] == pytest.approx(
        B601_RS_DEFAULT_SIM_MAX_MEASURED_VELOCITY_RAD_S
    )
    assert measured_envelope["response_margin_ratio"] == pytest.approx(
        B601_RS_DEFAULT_SIM_MEASURED_VELOCITY_MARGIN_RATIO
    )
    assert measured_envelope["hardware_safety_claim"] is False
    assert b601_sim_measured_velocity_limit_rad_s(
        B601_RS_DEFAULT_SIM_TARGET_MAX_VELOCITY_RAD_S
    ) == pytest.approx(B601_RS_DEFAULT_SIM_MAX_MEASURED_VELOCITY_RAD_S)


def test_ik_output_tracks_through_the_physical_position_servo(
    model_and_scene,
) -> None:
    pinocchio_model, scene = model_and_scene
    initial = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    target_qpos = np.array([0.15, 1.1, 1.15, 0.1, -0.1, 0.2])
    solution = B601DampedLeastSquaresIK(pinocchio_model).solve(
        pinocchio_model.forward_kinematics(target_qpos),
        seed_qpos=initial,
    )
    assert solution.success is True
    driver = B601PositionDriver(scene)
    driver.reset(initial)

    tick = driver.step_physics(_command(0, solution.qpos, 0.0))
    for _ in range(1999):
        tick = driver.step_physics()

    assert tick.state_valid is True
    assert tick.accepted_sequence == 0
    np.testing.assert_allclose(tick.measured_qpos, solution.qpos, atol=2e-6)
    for name, expected in B601_RS_GRIPPER_OPEN_QPOS.items():
        assert scene.data.qpos[scene.joint_qpos_addr(name)] == pytest.approx(expected)
        assert scene.data.qvel[scene.joint_dof_addr(name)] == pytest.approx(0.0)


def test_dynamic_demo_streams_continuous_previous_seed_ik() -> None:
    pytest.importorskip("mujoco")
    if not (REBOT_SOURCE_ROOT / B601_RS_URDF_RELATIVE_PATH).is_file():
        pytest.skip("initialize the pinned reBotArm_control_py gitlink")

    code, payload = run_demo(
        source_root=REBOT_SOURCE_ROOT,
        target_qpos=(0.15, 1.1, 1.15, 0.1, -0.1, 0.2),
        seed_qpos=B601_RS_DEFAULT_ARM_QPOS,
        duration_sec=2.0,
        viewer=False,
        continuous_ik=True,
    )

    assert code == 0
    continuous = payload["continuous_ik"]
    simulation = payload["simulation"]
    assert continuous["phase_contract"] == "hold_20pct_motion_60pct_hold_20pct"
    assert continuous["seed_policy"] == "previous_successful_solution"
    assert continuous["clock_policy"] == "independent_timestamp_deadlines"
    assert continuous["terminal_waypoint_policy"] == "exact_motion_boundary"
    assert continuous["ik_hz"] == pytest.approx(60.0)
    assert continuous["command_hz"] == pytest.approx(500.0)
    assert continuous["ik_attempts"] >= 70
    assert continuous["ik_successes"] == continuous["ik_attempts"]
    assert continuous["ik_failures"] == 0
    assert continuous["terminated_on_ik_failure"] is False
    assert continuous["max_position_error_m"] < 1e-4
    assert continuous["max_orientation_error_rad"] < 1e-4
    assert continuous["max_joint_command_step_rad"] < 1e-2
    assert continuous["final_waypoint_solved"] is True
    assert continuous["final_waypoint_accepted"] is True
    assert continuous["motion_completed"] is True
    assert (
        simulation["max_actuator_target_step_rad"]
        <= B601_RS_DEFAULT_SIM_TARGET_MAX_VELOCITY_RAD_S / 500.0 + 1e-12
    )
    assert simulation["max_joint_tracking_error_rad"] < 2e-3
    assert simulation["final_joint_tracking_error_rad"] < 2e-3
    assert (
        simulation["peak_actuator_tracking_error_rad"]
        >= simulation["final_joint_tracking_error_rad"]
    )
    assert simulation["physics_ticks"] == simulation["requested_physics_ticks"]
    assert simulation["run_completed"] is True
    assert simulation["viewer_closed_early"] is False
    assert simulation["state_valid"] is True
    assert simulation["state_within_limits"] is True
    assert simulation["driver_faulted"] is False


def test_continuous_demo_fails_closed_when_duration_cannot_reach_final_waypoint(
    model_and_scene,
) -> None:
    _pinocchio_model, _scene = model_and_scene
    code, payload = run_demo(
        source_root=REBOT_SOURCE_ROOT,
        target_qpos=(0.15, 1.1, 1.15, 0.1, -0.1, 0.2),
        seed_qpos=B601_RS_DEFAULT_ARM_QPOS,
        duration_sec=0.003,
        viewer=False,
        continuous_ik=True,
    )

    assert code == 5
    continuous = payload["continuous_ik"]
    assert continuous["final_waypoint_solved"] is False
    assert continuous["final_waypoint_accepted"] is False
    assert continuous["motion_completed"] is False


def test_continuous_demo_terminates_after_terminal_ik_failure(
    monkeypatch,
    model_and_scene,
) -> None:
    _pinocchio_model, _scene = model_and_scene
    real_solver = b601_demo.B601DampedLeastSquaresIK

    class TerminalFailingSolver:
        def __init__(self, model) -> None:
            self._delegate = real_solver(model)
            self._calls = 0
            self._terminal_pose = None

        def solve(self, target_pose, *, seed_qpos):
            result = self._delegate.solve(target_pose, seed_qpos=seed_qpos)
            self._calls += 1
            if self._calls == 1:
                self._terminal_pose = np.asarray(target_pose, dtype=float).copy()
                return result
            assert self._terminal_pose is not None
            if np.allclose(target_pose, self._terminal_pose):
                return replace(
                    result,
                    success=False,
                    status="injected_terminal_failure",
                )
            return result

    monkeypatch.setattr(
        b601_demo,
        "B601DampedLeastSquaresIK",
        TerminalFailingSolver,
    )
    code, payload = b601_demo.run_demo(
        source_root=REBOT_SOURCE_ROOT,
        target_qpos=(0.15, 1.1, 1.15, 0.1, -0.1, 0.2),
        seed_qpos=B601_RS_DEFAULT_ARM_QPOS,
        duration_sec=0.2,
        viewer=False,
        continuous_ik=True,
    )

    assert code == 5
    continuous = payload["continuous_ik"]
    assert continuous["ik_failures"] == 1
    assert continuous["terminated_on_ik_failure"] is True
    assert continuous["final_waypoint_solved"] is False
    assert continuous["final_waypoint_accepted"] is False
    assert continuous["motion_completed"] is False
    assert payload["simulation"]["run_completed"] is True


def test_continuous_demo_fails_closed_when_viewer_closes_early(
    monkeypatch,
    model_and_scene,
) -> None:
    _pinocchio_model, _scene = model_and_scene

    class ClosingViewerSession:
        def __init__(self, _scene) -> None:
            self.sync_calls = 0

        def start(self) -> None:
            pass

        def sync(self) -> bool:
            self.sync_calls += 1
            return self.sync_calls < 3

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        b601_demo,
        "MujocoPassiveViewerSession",
        ClosingViewerSession,
    )
    monkeypatch.setattr(b601_demo.time, "sleep", lambda _duration: None)
    code, payload = b601_demo.run_demo(
        source_root=REBOT_SOURCE_ROOT,
        target_qpos=(0.15, 1.1, 1.15, 0.1, -0.1, 0.2),
        seed_qpos=B601_RS_DEFAULT_ARM_QPOS,
        duration_sec=0.2,
        viewer=True,
        continuous_ik=True,
    )

    assert code == 5
    assert payload["continuous_ik"]["motion_completed"] is False
    assert payload["simulation"]["viewer_closed_early"] is True
    assert payload["simulation"]["run_completed"] is False


def test_b601_sim_package_has_no_hardware_or_rm65_runtime_imports() -> None:
    forbidden = (
        "control_composition",
        "control_station",
        "hardware",
        "sim.mujoco.actuation",
        "sim.mujoco.backend",
        "sim.mujoco.ik",
        "sim.rm65",
        "teleop_runtime",
    )
    violations: dict[str, list[str]] = {}

    for path in sorted((REPO_ROOT / "sim" / "mujoco" / "rebot_b601_rs").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imports.add(node.module)
        matched = sorted(
            name
            for name in imports
            if any(name == root or name.startswith(f"{root}.") for root in forbidden)
        )
        if matched:
            violations[path.name] = matched

    assert violations == {}
