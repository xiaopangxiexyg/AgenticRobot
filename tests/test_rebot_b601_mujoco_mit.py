from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from robot_motion.rebot_b601_rs import (
    B601_RS_ARM_JOINT_NAMES,
    B601_RS_JOINT_MIT_FRAME,
    B601_RS_URDF_RELATIVE_PATH,
    B601MitReference,
    B601PinocchioModel,
)
from sim.mujoco.rebot_b601_rs import (
    B601MujocoEffortDriver,
    B601PositionDriver,
    build_b601_mjcf_xml,
    build_b601_mujoco_scene,
    default_b601_mit_profile,
)
from sim.mujoco.rebot_b601_rs.scene import (
    B601_RS_DEFAULT_ARM_QPOS,
    B601_RS_GRIPPER_OPEN_QPOS,
    b601_effort_actuator_name,
    b601_position_actuator_name,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REBOT_SOURCE_ROOT = REPO_ROOT / "hardware" / "reBotArm_control_py"
pytestmark = [pytest.mark.optional, pytest.mark.contract]


@pytest.fixture(scope="module")
def pinocchio_model() -> B601PinocchioModel:
    pytest.importorskip("mujoco")
    pytest.importorskip("pinocchio")
    if not (REBOT_SOURCE_ROOT / B601_RS_URDF_RELATIVE_PATH).is_file():
        pytest.skip("initialize the pinned reBotArm_control_py gitlink")
    return B601PinocchioModel.from_rebot_source_root(REBOT_SOURCE_ROOT)


def _effort_scene(*, gravity=(0.0, 0.0, -9.81)):
    return build_b601_mujoco_scene(
        REBOT_SOURCE_ROOT,
        gravity_mps2=gravity,
        actuation_mode="effort",
    )


def _reference(
    sequence: int,
    qpos,
    timestamp: float,
    *,
    valid: bool = True,
    frame_id: str = B601_RS_JOINT_MIT_FRAME,
    overlay=(0.0,) * 6,
    qvel=(0.0,) * 6,
) -> B601MitReference:
    return B601MitReference(
        sequence=sequence,
        source_timestamp_sec=timestamp,
        receive_timestamp_sec=timestamp,
        qpos_rad=tuple(qpos),
        qvel_rad_s=tuple(qvel),
        joint_torque_overlay_nm=tuple(overlay),
        frame_id=frame_id,
        valid=valid,
    )


def _kinetic_energy(scene) -> float:
    mass_matrix = np.empty((scene.model.nv, scene.model.nv), dtype=float)
    scene.mujoco.mj_fullM(scene.model, scene.data, mass_matrix)
    qvel = np.asarray(scene.data.qvel, dtype=float)
    return 0.5 * float(qvel @ mass_matrix @ qvel)


def test_position_mode_remains_default_and_effort_mode_is_mutually_exclusive() -> None:
    default_root = ET.fromstring(build_b601_mjcf_xml(REBOT_SOURCE_ROOT))
    effort_root = ET.fromstring(
        build_b601_mjcf_xml(REBOT_SOURCE_ROOT, actuation_mode="effort")
    )

    assert len(default_root.findall("./actuator/position")) == 6
    assert default_root.findall("./actuator/motor") == []
    assert default_root.findall("./equality/joint") == []
    assert len(effort_root.findall("./actuator/motor")) == 6
    assert effort_root.findall("./actuator/position") == []
    fixed_gripper = effort_root.findall("./equality/joint")
    assert {element.attrib["joint1"] for element in fixed_gripper} == set(
        B601_RS_GRIPPER_OPEN_QPOS
    )
    assert all(element.attrib["solref"] == "0.002 1" for element in fixed_gripper)
    with pytest.raises(ValueError, match="position.*effort"):
        build_b601_mjcf_xml(REBOT_SOURCE_ROOT, actuation_mode="both")


def test_effort_actuators_are_unit_joint_torques_with_urdf_ceiling(
    pinocchio_model: B601PinocchioModel,
) -> None:
    scene = _effort_scene()
    actuator_ids = np.asarray(
        [
            scene.mujoco.mj_name2id(
                scene.model,
                scene.mujoco.mjtObj.mjOBJ_ACTUATOR,
                b601_effort_actuator_name(name),
            )
            for name in B601_RS_ARM_JOINT_NAMES
        ],
        dtype=np.int32,
    )
    joint_ids = np.asarray(
        [scene.joint_id(name) for name in B601_RS_ARM_JOINT_NAMES],
        dtype=np.int32,
    )
    dof_addresses = np.asarray(
        [scene.joint_dof_addr(name) for name in B601_RS_ARM_JOINT_NAMES],
        dtype=np.int32,
    )

    assert scene.model.nu == 6
    assert np.all(actuator_ids >= 0)
    np.testing.assert_array_equal(
        scene.model.actuator_trnid[actuator_ids, 0],
        joint_ids,
    )
    expected_gear = np.zeros_like(scene.model.actuator_gear[actuator_ids])
    expected_gear[:, 0] = 1.0
    np.testing.assert_allclose(scene.model.actuator_gear[actuator_ids], expected_gear)
    np.testing.assert_allclose(
        scene.model.actuator_ctrlrange[actuator_ids],
        [
            [-36.0, 36.0],
            [-36.0, 36.0],
            [-36.0, 36.0],
            [-14.0, 14.0],
            [-14.0, 14.0],
            [-14.0, 14.0],
        ],
    )
    np.testing.assert_allclose(
        scene.model.actuator_forcerange[actuator_ids],
        scene.model.actuator_ctrlrange[actuator_ids],
    )
    np.testing.assert_allclose(scene.data.ctrl, 0.0)
    for index, actuator_id in enumerate(actuator_ids):
        scene.data.ctrl[:] = 0.0
        scene.data.ctrl[actuator_id] = 1.0
        scene.forward()
        expected = np.zeros(6, dtype=float)
        expected[index] = 1.0
        np.testing.assert_allclose(
            scene.data.qfrc_actuator[dof_addresses],
            expected,
            atol=1e-12,
        )


def test_effort_scene_has_no_hidden_damping_or_position_servo(
    pinocchio_model: B601PinocchioModel,
) -> None:
    scene = _effort_scene()
    dof_addresses = np.asarray(
        [scene.joint_dof_addr(name) for name in B601_RS_ARM_JOINT_NAMES],
        dtype=np.int32,
    )

    for attribute in ("dof_damping", "dof_frictionloss", "dof_armature"):
        np.testing.assert_allclose(getattr(scene.model, attribute)[dof_addresses], 0.0)
    for name in B601_RS_ARM_JOINT_NAMES:
        assert (
            scene.mujoco.mj_name2id(
                scene.model,
                scene.mujoco.mjtObj.mjOBJ_ACTUATOR,
                b601_position_actuator_name(name),
            )
            < 0
        )
    with pytest.raises(KeyError, match="position actuator"):
        B601PositionDriver(scene)


def test_effort_driver_rejects_actuator_or_passive_force_semantic_drift(
    pinocchio_model: B601PinocchioModel,
) -> None:
    force_limited_scene = _effort_scene()
    force_limited_scene.model.actuator_forcerange[0] = (-0.01, 0.01)
    with pytest.raises(ValueError, match="force ranges must equal control ranges"):
        B601MujocoEffortDriver(force_limited_scene, pinocchio_model)

    gain_scene = _effort_scene()
    gain_scene.model.actuator_gainprm[0, 0] = 2.0
    with pytest.raises(ValueError, match="unit fixed gain"):
        B601MujocoEffortDriver(gain_scene, pinocchio_model)

    stiffness_scene = _effort_scene()
    stiffness_scene.model.jnt_stiffness[
        stiffness_scene.joint_id(B601_RS_ARM_JOINT_NAMES[0])
    ] = 1.0
    with pytest.raises(ValueError, match="zero passive joint stiffness"):
        B601MujocoEffortDriver(stiffness_scene, pinocchio_model)


def test_gravity_matches_mujoco_bias_for_open_gripper_sweep(
    pinocchio_model: B601PinocchioModel,
) -> None:
    scene = _effort_scene()
    lower, upper = pinocchio_model.arm_position_limits()
    dof_addresses = np.asarray(
        [scene.joint_dof_addr(name) for name in B601_RS_ARM_JOINT_NAMES],
        dtype=np.int32,
    )
    rng = np.random.default_rng(601)

    for _ in range(16):
        qpos = lower + (0.15 + 0.70 * rng.random(6)) * (upper - lower)
        scene.set_arm_qpos("b601_rs", qpos)
        scene.data.qvel[dof_addresses] = 0.0
        scene.data.ctrl[:] = 0.0
        for name, value in B601_RS_GRIPPER_OPEN_QPOS.items():
            scene.data.qpos[scene.joint_qpos_addr(name)] = value
            scene.data.qvel[scene.joint_dof_addr(name)] = 0.0
        scene.forward()

        expected = pinocchio_model.arm_generalized_gravity(
            qpos,
            auxiliary_joint_positions=B601_RS_GRIPPER_OPEN_QPOS,
        )
        np.testing.assert_allclose(
            scene.data.qfrc_bias[dof_addresses],
            expected,
            atol=1e-9,
            rtol=1e-9,
        )


def test_driver_fails_closed_on_wrong_scene_or_gravity_configuration(
    pinocchio_model: B601PinocchioModel,
) -> None:
    position_scene = build_b601_mujoco_scene(
        REBOT_SOURCE_ROOT,
        gravity_mps2=(0.0, 0.0, -9.81),
    )
    with pytest.raises(ValueError, match="effort actuators"):
        B601MujocoEffortDriver(position_scene, pinocchio_model)

    zero_gravity_scene = _effort_scene(gravity=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="gravity vectors do not match"):
        B601MujocoEffortDriver(
            zero_gravity_scene,
            pinocchio_model,
            enable_gravity_compensation=True,
        )
    with pytest.raises(TypeError, match="must be a bool"):
        B601MujocoEffortDriver(
            _effort_scene(),
            pinocchio_model,
            enable_gravity_compensation="false",  # type: ignore[arg-type]
        )


def test_gravity_compensation_holds_while_uncompensated_plant_falls(
    pinocchio_model: B601PinocchioModel,
) -> None:
    initial = np.asarray(B601_RS_DEFAULT_ARM_QPOS, dtype=float)
    compensated = B601MujocoEffortDriver(
        _effort_scene(),
        pinocchio_model,
        enable_gravity_compensation=True,
    )
    uncompensated = B601MujocoEffortDriver(
        _effort_scene(),
        pinocchio_model,
        enable_gravity_compensation=False,
    )
    compensated_dof_addresses = np.asarray(
        [compensated.scene.joint_dof_addr(name) for name in B601_RS_ARM_JOINT_NAMES],
        dtype=np.int32,
    )
    assert np.max(np.abs(compensated.scene.data.qacc[compensated_dof_addresses])) < 1e-7

    compensated_tick = None
    uncompensated_tick = None
    for _ in range(60):
        compensated_tick = compensated.step_physics()
        uncompensated_tick = uncompensated.step_physics()

    assert compensated_tick is not None
    assert uncompensated_tick is not None
    compensated_drift = np.max(
        np.abs(np.asarray(compensated_tick.measured_qpos) - initial)
    )
    uncompensated_drift = np.max(
        np.abs(np.asarray(uncompensated_tick.measured_qpos) - initial)
    )
    assert compensated_tick.driver_faulted is False
    assert uncompensated_tick.driver_faulted is False
    assert compensated_drift < 1e-9
    assert uncompensated_drift > 1e-2


def test_gravity_compensation_holds_random_poses_with_continuously_fixed_gripper(
    pinocchio_model: B601PinocchioModel,
) -> None:
    driver = B601MujocoEffortDriver(
        _effort_scene(),
        pinocchio_model,
        enable_gravity_compensation=True,
    )
    lower, upper = pinocchio_model.arm_position_limits()
    rng = np.random.default_rng(602)

    maximum_drift = 0.0
    for _ in range(8):
        initial = lower + (0.2 + 0.6 * rng.random(6)) * (upper - lower)
        driver.reset(initial)
        tick = None
        for _ in range(500):
            tick = driver.step_physics()
        assert tick is not None
        assert tick.driver_faulted is False
        maximum_drift = max(
            maximum_drift,
            float(np.max(np.abs(np.asarray(tick.measured_qpos) - initial))),
        )

    assert maximum_drift < 5e-5


def test_default_damping_does_not_inject_energy_at_random_poses(
    pinocchio_model: B601PinocchioModel,
) -> None:
    scene = _effort_scene()
    driver = B601MujocoEffortDriver(
        scene,
        pinocchio_model,
        enable_gravity_compensation=True,
    )
    lower, upper = pinocchio_model.arm_position_limits()
    dof_addresses = np.asarray(
        [scene.joint_dof_addr(name) for name in B601_RS_ARM_JOINT_NAMES],
        dtype=np.int32,
    )
    rng = np.random.default_rng(603)

    for _ in range(8):
        qpos = lower + (0.2 + 0.6 * rng.random(6)) * (upper - lower)
        qvel = rng.uniform(-0.08, 0.08, size=6)
        driver.reset(qpos)
        scene.data.qvel[dof_addresses] = qvel
        scene.forward()
        initial_energy = _kinetic_energy(scene)
        peak_energy = initial_energy
        tick = None
        for _ in range(500):
            tick = driver.step_physics()
            peak_energy = max(peak_energy, _kinetic_energy(scene))
        assert tick is not None
        assert tick.driver_faulted is False
        assert peak_energy <= initial_energy * (1.0 + 1e-6) + 1e-12
        assert _kinetic_energy(scene) < 0.8 * initial_energy


def test_effort_driver_schedules_500hz_references_against_1000hz_physics(
    pinocchio_model: B601PinocchioModel,
) -> None:
    driver = B601MujocoEffortDriver(
        _effort_scene(),
        pinocchio_model,
        enable_gravity_compensation=True,
    )

    ticks = [driver.step_physics() for _ in range(1000)]

    assert sum(tick.command_due for tick in ticks) == 500
    assert ticks[0].command_due is True
    assert ticks[1].command_due is False
    assert ticks[2].command_due is True
    assert driver.physics_time_sec == pytest.approx(1.0)
    assert all(not tick.driver_faulted for tick in ticks)


def test_default_profile_damping_is_dissipative_and_stable(
    pinocchio_model: B601PinocchioModel,
) -> None:
    scene = _effort_scene()
    driver = B601MujocoEffortDriver(
        scene,
        pinocchio_model,
        enable_gravity_compensation=True,
    )
    dof_addresses = np.asarray(
        [scene.joint_dof_addr(name) for name in B601_RS_ARM_JOINT_NAMES],
        dtype=np.int32,
    )
    initial_velocity = np.asarray([0.05, -0.04, 0.03, -0.02, 0.01, -0.05])
    scene.data.qvel[dof_addresses] = initial_velocity
    scene.forward()

    first = driver.step_physics()
    initial_norm = float(np.linalg.norm(initial_velocity))
    assert first.torque_breakdown is not None
    assert first.physics_time_sec == pytest.approx(0.0)
    assert first.measured_time_sec == pytest.approx(0.001)
    np.testing.assert_allclose(first.control_qvel, initial_velocity)
    assert (
        float(
            np.dot(
                first.torque_breakdown.damping_torque_nm,
                initial_velocity,
            )
        )
        <= 1e-12
    )
    tick = first
    for _ in range(99):
        tick = driver.step_physics()

    assert tick.driver_faulted is False
    assert np.linalg.norm(tick.measured_qvel) < initial_norm


def test_driver_schedules_references_and_clears_invalid_or_timed_out_overlay(
    pinocchio_model: B601PinocchioModel,
) -> None:
    scene = _effort_scene()
    base_profile = default_b601_mit_profile(scene)
    profile = replace(base_profile, max_abs_overlay_torque_nm=(0.5,) * 6)
    driver = B601MujocoEffortDriver(
        scene,
        pinocchio_model,
        profile=profile,
        enable_gravity_compensation=True,
        command_timeout_sec=0.01,
    )
    qpos = np.asarray(B601_RS_DEFAULT_ARM_QPOS, dtype=float)
    overlay = (0.25, 0.0, 0.0, 0.0, 0.0, 0.0)

    accepted = driver.step_physics(_reference(1, qpos, 0.0, overlay=overlay))
    early = driver.step_physics(_reference(2, qpos, 0.001, overlay=overlay))
    invalid = driver.step_physics(
        _reference(2, qpos, 0.002, valid=False, overlay=overlay)
    )
    driver.step_physics()
    accepted_again = driver.step_physics(_reference(2, qpos, 0.004, overlay=overlay))
    early_invalid = driver.step_physics(
        _reference(3, qpos, 0.005, valid=False, overlay=overlay)
    )
    accepted_for_timeout = driver.step_physics(
        _reference(3, qpos, 0.006, overlay=overlay)
    )

    assert accepted.command_applied is True
    assert driver.execution_metadata["profile"]["haptic_overlay_enabled"] is True
    assert accepted.torque_breakdown is not None
    assert accepted.torque_breakdown.overlay_effective_nm[0] == pytest.approx(0.25)
    assert early.rejection_reason == "command_before_deadline"
    assert invalid.rejection_reason == "invalid_sample"
    assert invalid.active_sequence is None
    assert invalid.torque_breakdown is not None
    np.testing.assert_allclose(invalid.torque_breakdown.overlay_effective_nm, 0.0)
    assert accepted_again.command_applied is True
    assert early_invalid.rejection_reason == "invalid_sample"
    assert early_invalid.active_sequence is None
    assert early_invalid.torque_breakdown is not None
    np.testing.assert_allclose(
        early_invalid.torque_breakdown.overlay_effective_nm,
        0.0,
    )
    assert accepted_for_timeout.command_applied is True

    timed_out = None
    for _ in range(20):
        tick = driver.step_physics()
        if tick.reference_timed_out:
            timed_out = tick
            break
    assert timed_out is not None
    assert timed_out.active_sequence is None
    assert timed_out.torque_breakdown is not None
    np.testing.assert_allclose(timed_out.torque_breakdown.overlay_effective_nm, 0.0)


def test_reference_timeout_clears_at_the_exact_physics_deadline(
    pinocchio_model: B601PinocchioModel,
) -> None:
    driver = B601MujocoEffortDriver(
        _effort_scene(),
        pinocchio_model,
        command_timeout_sec=0.002,
    )
    qpos = np.asarray(B601_RS_DEFAULT_ARM_QPOS, dtype=float)

    accepted = driver.step_physics(_reference(1, qpos, 0.0))
    before_deadline = driver.step_physics()
    at_deadline = driver.step_physics()

    assert accepted.command_applied is True
    assert before_deadline.active_reference_age_sec == pytest.approx(0.001)
    assert before_deadline.reference_timed_out is False
    assert at_deadline.physics_time_sec == pytest.approx(0.002)
    assert at_deadline.reference_timed_out is True
    assert at_deadline.active_sequence is None


def test_driver_rejects_local_stale_references_without_preemption(
    pinocchio_model: B601PinocchioModel,
) -> None:
    driver = B601MujocoEffortDriver(_effort_scene(), pinocchio_model)
    qpos = np.asarray(B601_RS_DEFAULT_ARM_QPOS, dtype=float)

    accepted = driver.step_physics(_reference(2, qpos, 2.0))
    driver.step_physics()
    stale_sequence = driver.step_physics(_reference(2, qpos, 3.0))
    driver.step_physics()
    stale_source = driver.step_physics(_reference(3, qpos, 1.0))
    driver.step_physics()
    stale_receive = driver.step_physics(
        B601MitReference(
            sequence=3,
            source_timestamp_sec=3.0,
            receive_timestamp_sec=1.0,
            qpos_rad=tuple(qpos),
        )
    )
    assert accepted.command_applied is True
    assert stale_sequence.rejection_reason == "stale_sequence"
    assert stale_source.rejection_reason == "stale_source_timestamp"
    assert stale_receive.rejection_reason == "stale_receive_timestamp"
    assert stale_sequence.active_sequence == 2
    assert stale_source.active_sequence == 2
    assert stale_receive.active_sequence == 2


def test_unexpected_local_joint_frame_deactivates_reference_immediately(
    pinocchio_model: B601PinocchioModel,
) -> None:
    scene = _effort_scene()
    base_profile = default_b601_mit_profile(scene)
    profile = replace(base_profile, max_abs_overlay_torque_nm=(0.5,) * 6)
    driver = B601MujocoEffortDriver(
        scene,
        pinocchio_model,
        profile=profile,
        enable_gravity_compensation=True,
    )
    qpos = np.asarray(B601_RS_DEFAULT_ARM_QPOS, dtype=float)
    overlay = (0.25, 0.0, 0.0, 0.0, 0.0, 0.0)

    accepted = driver.step_physics(_reference(2, qpos, 2.0, overlay=overlay))
    # The next 1 kHz physics tick is before the 500 Hz command deadline.  A
    # frame mismatch is nevertheless a local command-contract error and must
    # deactivate immediately rather than being hidden as an early command.
    wrong_frame = driver.step_physics(
        _reference(3, qpos, 2.001, frame_id="unexpected", overlay=overlay)
    )

    assert accepted.command_applied is True
    assert accepted.active_sequence == 2
    assert wrong_frame.rejection_reason == "unexpected_frame"
    assert wrong_frame.active_sequence is None
    assert wrong_frame.reference_timed_out is False
    assert wrong_frame.driver_faulted is False
    assert wrong_frame.plant_advanced is True
    assert wrong_frame.torque_breakdown is not None
    np.testing.assert_allclose(
        wrong_frame.torque_breakdown.overlay_effective_nm,
        0.0,
    )
    np.testing.assert_allclose(
        wrong_frame.torque_breakdown.position_torque_nm,
        0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        wrong_frame.torque_breakdown.damping_torque_nm,
        -np.asarray(wrong_frame.control_qvel),
        atol=1e-12,
    )
    expected_gravity = pinocchio_model.arm_generalized_gravity(
        wrong_frame.control_qpos,
        auxiliary_joint_positions=B601_RS_GRIPPER_OPEN_QPOS,
    )
    np.testing.assert_allclose(
        wrong_frame.torque_breakdown.gravity_torque_nm,
        expected_gravity,
        atol=1e-10,
    )


def test_reference_and_measured_velocity_limits_are_independent(
    pinocchio_model: B601PinocchioModel,
) -> None:
    scene = _effort_scene(gravity=(0.0, 0.0, 0.0))
    base_profile = default_b601_mit_profile(scene)
    profile = replace(
        base_profile,
        max_abs_reference_velocity_rad_s=(0.05,) * 6,
        max_abs_state_velocity_rad_s=(0.5,) * 6,
    )
    driver = B601MujocoEffortDriver(scene, pinocchio_model, profile=profile)
    qpos = np.asarray(B601_RS_DEFAULT_ARM_QPOS, dtype=float)

    rejected = driver.step_physics(
        _reference(1, qpos, 0.0, qvel=(0.1,) + (0.0,) * 5)
    )
    assert rejected.rejection_reason == "reference_velocity_limit_violation"
    assert rejected.driver_faulted is False

    driver.reset()
    first_dof = scene.joint_dof_addr(B601_RS_ARM_JOINT_NAMES[0])
    scene.data.qvel[first_dof] = 0.25
    scene.forward()
    within_state_limit = driver.step_physics()
    assert within_state_limit.driver_faulted is False

    driver.reset()
    scene.data.qvel[first_dof] = 0.6
    scene.forward()
    state_fault = driver.step_physics()
    assert "joint_velocity_limit_violation" in str(state_fault.fault_reason)
    assert state_fault.plant_advanced is False


def test_final_total_torque_is_clipped_then_slew_limited(
    pinocchio_model: B601PinocchioModel,
) -> None:
    scene = _effort_scene(gravity=(0.0, 0.0, 0.0))
    base_profile = default_b601_mit_profile(scene)
    profile = replace(
        base_profile,
        damping_kd_nm_s_per_rad=(0.0,) * 6,
        max_abs_torque_nm=(1.0,) * 6,
        max_torque_rate_nm_s=(100.0,) * 6,
        max_abs_overlay_torque_nm=(10.0,) * 6,
    )
    driver = B601MujocoEffortDriver(scene, pinocchio_model, profile=profile)
    qpos = np.asarray(B601_RS_DEFAULT_ARM_QPOS, dtype=float)
    reference = _reference(
        1,
        qpos,
        0.0,
        overlay=(5.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )

    first = driver.step_physics(reference)
    second = driver.step_physics()

    assert first.torque_breakdown is not None
    assert first.torque_breakdown.requested_torque_nm[0] == pytest.approx(5.0)
    assert first.clipped_torque_nm[0] == pytest.approx(1.0)
    assert first.applied_torque_nm[0] == pytest.approx(0.1)
    assert first.torque_limited is True
    assert first.slew_limited is True
    assert second.applied_torque_nm[0] == pytest.approx(0.2)
    assert abs(second.applied_torque_nm[0] - first.applied_torque_nm[0]) <= 0.1
    dof_addresses = np.asarray(
        [scene.joint_dof_addr(name) for name in B601_RS_ARM_JOINT_NAMES],
        dtype=np.int32,
    )
    np.testing.assert_allclose(
        scene.data.qfrc_actuator[dof_addresses],
        second.applied_torque_nm,
        atol=1e-12,
    )


def test_finite_gain_overflow_latches_fault_before_advancing_plant(
    pinocchio_model: B601PinocchioModel,
) -> None:
    scene = _effort_scene(gravity=(0.0, 0.0, 0.0))
    base_profile = default_b601_mit_profile(scene)
    profile = replace(
        base_profile,
        position_kp_nm_per_rad=(np.finfo(float).max,) + (0.0,) * 5,
        damping_kd_nm_s_per_rad=(0.0,) * 6,
    )
    driver = B601MujocoEffortDriver(scene, pinocchio_model, profile=profile)
    lower, upper = pinocchio_model.arm_position_limits()
    current = np.asarray(B601_RS_DEFAULT_ARM_QPOS, dtype=float)
    target = current.copy()
    target[0] = upper[0] if abs(upper[0] - current[0]) > 1.0 else lower[0]
    time_before = float(scene.data.time)

    tick = driver.step_physics(_reference(1, target, 0.0))

    assert tick.driver_faulted is True
    assert tick.fault_reason == "non_finite_computed_torque"
    assert tick.plant_advanced is False
    assert float(scene.data.time) == pytest.approx(time_before)
    np.testing.assert_allclose(scene.data.ctrl, 0.0)


def test_non_finite_position_and_velocity_faults_stop_the_plant_until_reset(
    pinocchio_model: B601PinocchioModel,
) -> None:
    scene = _effort_scene()
    driver = B601MujocoEffortDriver(scene, pinocchio_model)
    first_joint = B601_RS_ARM_JOINT_NAMES[0]
    qpos_address = scene.joint_qpos_addr(first_joint)
    qvel_address = scene.joint_dof_addr(first_joint)
    joint_id = scene.joint_id(first_joint)

    scene.data.qpos[qpos_address] = scene.model.jnt_range[joint_id, 1] + 0.01
    scene.forward()
    limit_fault = driver.step_physics()
    assert limit_fault.driver_faulted is True
    assert limit_fault.state_within_limits is False
    assert limit_fault.plant_advanced is False
    assert "joint_state_limit_violation" in str(limit_fault.fault_reason)

    driver.reset()
    scene.data.qvel[qvel_address] = 2.1
    scene.forward()
    velocity_fault = driver.step_physics()
    assert velocity_fault.driver_faulted is True
    assert velocity_fault.state_within_limits is False
    assert velocity_fault.plant_advanced is False
    assert "joint_velocity_limit_violation" in str(velocity_fault.fault_reason)

    driver.reset()
    scene.data.qpos[qpos_address] = np.nan
    time_before = float(scene.data.time)
    non_finite = driver.step_physics()
    assert non_finite.state_valid is False
    assert non_finite.fault_reason == "non_finite_simulated_state"
    assert non_finite.plant_advanced is False
    assert float(scene.data.time) == pytest.approx(time_before)

    driver.reset()
    recovered = driver.step_physics()
    assert recovered.driver_faulted is False
    assert recovered.plant_advanced is True


@pytest.mark.parametrize(
    "attribute",
    ["qacc_warmstart", "qfrc_applied", "xfrc_applied"],
)
def test_non_finite_pre_step_inputs_fail_closed_before_mujoco_step(
    pinocchio_model: B601PinocchioModel,
    attribute: str,
) -> None:
    scene = _effort_scene()
    driver = B601MujocoEffortDriver(scene, pinocchio_model)
    getattr(scene.data, attribute).flat[0] = np.nan
    time_before = float(scene.data.time)

    tick = driver.step_physics()

    assert tick.fault_reason == "non_finite_simulated_state"
    assert tick.plant_advanced is False
    assert float(scene.data.time) == pytest.approx(time_before)
    np.testing.assert_allclose(scene.data.ctrl, 0.0)


def test_runtime_gravity_or_auxiliary_constraint_mutation_fails_closed(
    pinocchio_model: B601PinocchioModel,
) -> None:
    gravity_scene = _effort_scene()
    gravity_driver = B601MujocoEffortDriver(
        gravity_scene,
        pinocchio_model,
        enable_gravity_compensation=True,
    )
    gravity_scene.model.opt.gravity[2] = 0.0
    gravity_time = float(gravity_scene.data.time)

    gravity_tick = gravity_driver.step_physics()

    assert gravity_tick.fault_reason == "gravity_configuration_changed"
    assert gravity_tick.plant_advanced is False
    assert float(gravity_scene.data.time) == pytest.approx(gravity_time)
    np.testing.assert_allclose(gravity_scene.data.ctrl, 0.0)

    constraint_scene = _effort_scene()
    constraint_driver = B601MujocoEffortDriver(
        constraint_scene,
        pinocchio_model,
        enable_gravity_compensation=True,
    )
    constraint_scene.data.eq_active[:] = False
    constraint_time = float(constraint_scene.data.time)

    constraint_tick = constraint_driver.step_physics()

    assert constraint_tick.fault_reason == "auxiliary_constraint_changed"
    assert constraint_tick.plant_advanced is False
    assert float(constraint_scene.data.time) == pytest.approx(constraint_time)
    np.testing.assert_allclose(constraint_scene.data.ctrl, 0.0)


def test_driver_fails_closed_if_an_external_owner_advances_the_plant(
    pinocchio_model: B601PinocchioModel,
) -> None:
    scene = _effort_scene(gravity=(0.0, 0.0, 0.0))
    driver = B601MujocoEffortDriver(scene, pinocchio_model)
    scene.mujoco.mj_step(scene.model, scene.data)
    externally_advanced_time = float(scene.data.time)

    tick = driver.step_physics()

    assert tick.fault_reason == "physics_clock_desynchronized"
    assert tick.plant_advanced is False
    assert float(scene.data.time) == pytest.approx(externally_advanced_time)
    np.testing.assert_allclose(scene.data.ctrl, 0.0)


def test_effort_metadata_disclaims_hardware_fidelity(
    pinocchio_model: B601PinocchioModel,
) -> None:
    metadata = B601MujocoEffortDriver(
        _effort_scene(),
        pinocchio_model,
        enable_gravity_compensation=True,
    ).execution_metadata

    assert metadata["robstride_internal_mit_fidelity"] == "not_claimed"
    assert metadata["plant_step_owner"] == "exclusive_driver_mj_step"
    assert metadata["damping_owner"] == "software_mit_profile_only"
    assert metadata["transport_scope"] == "single_host_local_wired_control"
    assert metadata["network_transport"] == "out_of_scope"
    assert metadata["timestamp_contract"] == (
        "host_local_monotonic_observability"
    )
    assert metadata["watchdog_clock"] == "driver_owned_local_acceptance_time"
    assert metadata["stale_sample_action"] == (
        "reject_without_preemption_until_local_watchdog"
    )
    assert metadata["unexpected_frame_action"] == (
        "deactivate_reference_keep_local_gravity_and_damping"
    )
    assert metadata["hardware_safety_claim"] is False
    assert metadata["profile"]["position_kp_nm_per_rad"] == (0.0,) * 6
    assert metadata["profile"]["damping_kd_nm_s_per_rad"] == (1.0,) * 6
    assert metadata["profile"]["haptic_overlay_enabled"] is False
    assert metadata["gravity_auxiliary_constraint"] == (
        "continuous_mujoco_joint_equality"
    )
