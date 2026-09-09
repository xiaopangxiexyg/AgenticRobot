from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from robot_motion.rebot_b601_rs import B601MitReference, B601PinocchioModel
from sim.contracts import TimingSpec
from sim.mujoco.rebot_b601_rs import build_b601_mujoco_scene
from sim.mujoco.rebot_b601_rs.mit_driver import B601MujocoEffortDriver
from sim.mujoco.rebot_b601_rs.real2sim import (
    B601Real2SimControllerProfile,
    B601SimStartupProfile,
)

pytestmark = [pytest.mark.contract, pytest.mark.optional]
SOURCE = Path(__file__).resolve().parents[1] / "hardware/reBotArm_control_py"


def test_profile_compensation_and_input_validation():
    profile = B601Real2SimControllerProfile(
        gravity_scale=0.5,
        gravity_joint_scales=(2.0,) * 6,
        gravity_joint_bias_nm=(0.1,) * 6,
        gravity_coulomb_friction_nm=(0.2,) * 6,
        coulomb_velocity_threshold_rad_s=0.01,
    )
    np.testing.assert_allclose(
        profile.compensate(
            np.ones(6),
            np.array(
                [0.0, 0.01, -0.01, 0.02, -0.02, 0.0],
            ),
        ),
        [1.1, 1.1, 1.1, 1.3, 0.9, 1.1],
    )
    for field, value in (
        ("gravity_scale", -1.0),
        ("driver_kd_nm_s_per_rad", (1.0,)),
        ("gravity_joint_bias_nm", (float("nan"),) * 6),
    ):
        with pytest.raises(ValueError):
            replace(profile, **{field: value})


@pytest.fixture
def driver():
    pytest.importorskip("mujoco")
    pytest.importorskip("pinocchio")
    model = B601PinocchioModel.from_rebot_source_root(SOURCE)
    scene = build_b601_mujoco_scene(
        SOURCE, actuation_mode="effort", gravity_mps2=(0.0, 0.0, -9.81)
    )
    return B601MujocoEffortDriver(
        scene,
        model,
        timing=TimingSpec(command_hz=20.0, physics_hz=1000.0),
        enable_gravity_compensation=True,
        real2sim_profile=B601Real2SimControllerProfile(),
    )


def reference(driver, sequence=0, frame_id="b601_rs/joint_mit_reference"):
    return B601MitReference(
        sequence=sequence,
        source_timestamp_sec=driver.physics_time_sec,
        receive_timestamp_sec=driver.physics_time_sec,
        qpos_rad=tuple(driver._measured_qpos()),
        frame_id=frame_id,
    )


def test_host_zoh_and_physics_rate_damping(driver):
    first = driver.step_physics(reference(driver))
    assert first.host_torque_updated
    assert not first.driver_faulted
    second = driver.step_physics()
    assert not second.host_torque_updated
    assert second.torque_breakdown == first.torque_breakdown
    np.testing.assert_allclose(
        second.driver_damping_torque_nm,
        -np.array(driver.real2sim_profile.driver_kd_nm_s_per_rad)
        * np.array(second.control_qvel),
    )
    assert np.dot(second.driver_damping_torque_nm, second.control_qvel) <= 0.0
    assert driver.profile.damping_kd_nm_s_per_rad == (0.0,) * 6
    assert np.all(driver.scene.model.dof_damping == 0.0)
    updates = 1
    for _ in range(49):
        tick = driver.step_physics(reference(driver, 1) if driver.command_due else None)
        updates += tick.host_torque_updated
        assert not tick.driver_faulted
    assert updates == 2


def test_controller_compensation_does_not_change_plant_gravity_parity(driver):
    qpos = driver._measured_qpos()
    raw = driver._gravity_torque(qpos)
    np.testing.assert_allclose(
        driver.scene.data.qfrc_bias[driver._qvel_addresses], raw, atol=1e-9
    )
    tick = driver.step_physics(reference(driver))
    np.testing.assert_allclose(
        tick.torque_breakdown.gravity_torque_nm,
        driver.real2sim_profile.compensate(raw, np.zeros(6)),
    )
    assert (
        driver.execution_metadata["real2sim_controller"]["hardware_safety_claim"]
        is False
    )


def test_revocation_invalidates_host_cache_off_cadence(driver):
    driver.step_physics(reference(driver))
    tick = driver.step_physics(reference(driver, 1, "wrong/frame"))
    assert not tick.command_due
    assert tick.host_torque_updated
    assert tick.active_sequence is None
    assert tick.rejection_reason == "unexpected_frame"


def test_final_limit_includes_local_damping(driver):
    driver.profile = replace(
        driver.profile, max_abs_torque_nm=(0.001,) * 6, max_torque_rate_nm_s=(1.0,) * 6
    )
    tick = driver.step_physics(reference(driver))
    assert tick.torque_limited
    assert max(abs(v) for v in tick.applied_torque_nm) <= 0.001


def test_timeout_clears_overlay_and_reset_starts_new_host_epoch(driver):
    driver.profile = replace(driver.profile, max_abs_overlay_torque_nm=(1.0,) * 6)
    command = replace(reference(driver), joint_torque_overlay_nm=(0.01,) * 6)
    first = driver.step_physics(command)
    assert first.active_sequence == 0
    for _ in range(100):
        tick = driver.step_physics()
    assert tick.reference_timed_out
    assert tick.active_sequence is None
    assert tick.host_torque_updated
    np.testing.assert_allclose(tick.torque_breakdown.overlay_effective_nm, np.zeros(6))
    driver.reset()
    tick = driver.step_physics(reference(driver))
    assert tick.physics_tick == 0
    assert tick.host_torque_updated


def test_offset_is_controller_only(driver):
    driver.real2sim_profile = replace(
        driver.real2sim_profile, gravity_joint_offsets_rad=(0.01,) * 6
    )
    q = driver._measured_qpos()
    expected = driver.real2sim_profile.compensate(
        driver._gravity_torque(q + 0.01), np.zeros(6)
    )
    np.testing.assert_allclose(driver._controller_gravity(q, np.zeros(6)), expected)
    driver._verify_gravity_configuration(q)


@pytest.mark.parametrize("host_hz,expected_code", [(500.0, 0), (20.0, 3)])
def test_opt_in_fixture_preserves_effect_and_guard_results(host_hz, expected_code):
    from sim.mujoco.rebot_b601_rs.haptic_demo import run_haptic_fixture_demo

    code, report = run_haptic_fixture_demo(
        source_root=SOURCE,
        fixture_name="pose-6d",
        real2sim_controller=True,
        host_hz=host_hz,
    )
    assert code == expected_code
    assert report["driver"]["host_torque_hz"] == host_hz
    assert report["driver"]["hardware_safety_claim"] is False
    if expected_code:
        assert report["simulation_acceptance"]["driver_faulted"]
        assert not report["simulation_acceptance"]["passed"]
    else:
        assert report["simulation_acceptance"]["advanced_physics_ticks"] == 1500
        assert report["effect_acceptance"]["passed"]


def test_local_feedback_bypasses_host_slew_but_retains_total_limit(driver):
    driver.profile = replace(driver.profile, max_torque_rate_nm_s=(1e-6,) * 6)
    driver.step_physics(reference(driver))
    driver.scene.data.qvel[driver._qvel_addresses[0]] = 0.01
    driver.scene.forward()
    tick = driver.step_physics()
    assert not tick.host_torque_updated
    assert tick.driver_damping_torque_nm[0] == pytest.approx(-0.002)
    assert tick.applied_torque_nm[0] - tick.host_command_torque_nm[0] == pytest.approx(
        -0.002
    )
    assert abs(tick.driver_damping_torque_nm[0]) > 100 * 1e-6 * 0.001


@pytest.fixture
def startup_driver(driver):
    # Raw model equilibrium isolates lifecycle from unidentified gravity biases.
    controller = replace(
        driver.real2sim_profile,
        gravity_joint_scales=(1.0,) * 6,
        gravity_joint_bias_nm=(0.0,) * 6,
    )
    return B601MujocoEffortDriver(
        driver.scene,
        driver.pinocchio_model,
        timing=driver.timing,
        enable_gravity_compensation=True,
        real2sim_profile=controller,
        startup_profile=B601SimStartupProfile(
            lock_kp_nm_per_rad=(2.0,) * 6,
            lock_kd_nm_s_per_rad=controller.driver_kd_nm_s_per_rad,
        ),
    )


def test_startup_reaches_active_with_full_gravity_from_first_lock(startup_driver):
    driver = startup_driver
    assert driver.execution_metadata["sim_control_phase"] == "ENABLED_UNARMED"
    phases = []
    for i in range(451):
        tick = driver.step_physics(reference(driver, i) if driver.command_due else None)
        assert not tick.driver_faulted
        if not phases or phases[-1] != tick.control_phase:
            phases.append(tick.control_phase)
        if tick.control_phase == "LOCKING":
            np.testing.assert_allclose(
                tick.host_command_torque_nm,
                driver._gravity_torque(np.array(tick.control_qpos)),
                atol=1e-9,
            )
    assert phases == ["LOCKING", "LOCKED", "GRAVITY_RAMP", "HANDOFF", "ACTIVE"]
    np.testing.assert_allclose(tick.driver_position_torque_nm, np.zeros(6))


def test_lock_uses_current_anchor_not_active_reference(startup_driver):
    driver = startup_driver
    command = reference(driver)
    shifted = list(command.qpos_rad)
    shifted[0] += 0.1
    driver.step_physics(replace(command, qpos_rad=tuple(shifted)))
    driver.scene.data.qpos[driver._qpos_addresses[0]] += 0.001
    driver.scene.forward()
    tick = driver.step_physics()
    assert tick.control_phase == "LOCKING"
    assert tick.driver_position_torque_nm[0] == pytest.approx(-0.002)
    assert tick.torque_breakdown.position_torque_nm == (0.0,) * 6


@pytest.mark.parametrize("host_hz", [20.0, 500.0])
def test_startup_deadline_does_not_slip_one_host_tick(startup_driver, host_hz):
    driver = startup_driver
    driver.timing = TimingSpec(command_hz=host_hz, physics_hz=1000.0)
    for i in range(301):
        tick = driver.step_physics(reference(driver, i) if driver.command_due else None)
        assert not tick.driver_faulted
        if i < 300:
            assert tick.control_phase != "ACTIVE"
    assert tick.physics_time_sec == pytest.approx(0.3)
    assert tick.control_phase == "ACTIVE"
    assert driver._lock_fraction == 0.0


def test_startup_deadline_still_checks_lock_displacement(startup_driver):
    driver = startup_driver
    for i in range(300):
        driver.step_physics(reference(driver, i) if driver.command_due else None)
    driver.scene.data.qpos[driver._qpos_addresses[0]] += 0.03
    driver.scene.forward()
    tick = driver.step_physics(reference(driver, 300))
    assert tick.control_phase == "PROTECTIVE_HOLD"
    assert tick.phase_reason == "startup_lock_displacement"


def test_lock_excursion_enters_latched_draggable_hold(startup_driver):
    driver = startup_driver
    driver.step_physics(reference(driver))
    driver.scene.data.qpos[driver._qpos_addresses[0]] += 0.03
    driver.scene.forward()
    # Reach the next host admission cut without fabricating a new reference.
    for _ in range(50):
        tick = driver.step_physics()
        if tick.driver_faulted:
            break
    assert not tick.driver_faulted
    assert tick.control_phase == "PROTECTIVE_HOLD"
    assert tick.phase_reason == "startup_lock_displacement"
    np.testing.assert_allclose(tick.driver_position_torque_nm, np.zeros(6))
    assert (
        driver.step_physics(reference(driver, 2)).rejection_reason
        == "lifecycle_latched"
    )


def test_protective_hold_and_explicit_disable_remain_latched(startup_driver):
    driver = startup_driver
    driver.step_physics(reference(driver))
    driver.request_protective_hold("duration_complete")
    tick = driver.step_physics(reference(driver, 1))
    assert tick.control_phase == "PROTECTIVE_HOLD"
    assert tick.rejection_reason == "lifecycle_latched"
    assert any(abs(v) > 0.1 for v in tick.applied_torque_nm)
    np.testing.assert_allclose(tick.torque_breakdown.position_torque_nm, np.zeros(6))
    np.testing.assert_allclose(tick.driver_position_torque_nm, np.zeros(6))
    driver.request_disable()
    driver.request_protective_hold()
    tick = driver.step_physics(reference(driver, 2))
    assert tick.control_phase == "DISABLED"
    assert tick.rejection_reason == "lifecycle_latched"
    np.testing.assert_allclose(tick.applied_torque_nm, np.zeros(6))
    driver.reset()
    assert driver.execution_metadata["sim_control_phase"] == "ENABLED_UNARMED"


def test_startup_fixture_reports_protective_hold_as_not_accepted():
    from sim.mujoco.rebot_b601_rs.haptic_demo import run_haptic_fixture_demo

    code, report = run_haptic_fixture_demo(
        source_root=SOURCE,
        fixture_name="pose-6d",
        real2sim_controller=True,
        host_hz=500.0,
        startup_profile=B601SimStartupProfile(
            lock_kp_nm_per_rad=(2.0,) * 6,
            lock_kd_nm_s_per_rad=B601Real2SimControllerProfile().driver_kd_nm_s_per_rad,
        ),
    )
    acceptance = report["simulation_acceptance"]
    assert code == 3
    assert not acceptance["passed"]
    assert acceptance["run_completed"]
    assert not acceptance["driver_faulted"]
    assert acceptance["final_control_phase"] == "PROTECTIVE_HOLD"
    assert acceptance["phase_transitions"][-1]["reason"] == "startup_not_settled"


def test_bad_reference_during_lock_cannot_rearm_it(startup_driver):
    driver = startup_driver
    driver.step_physics(reference(driver))
    tick = driver.step_physics(reference(driver, 1, "wrong/frame"))
    assert tick.control_phase == "PROTECTIVE_HOLD"
    assert tick.driver_position_torque_nm == (0.0,) * 6
    assert (
        driver.step_physics(reference(driver, 2)).rejection_reason
        == "lifecycle_latched"
    )


def test_vendor_local_gains_complete_startup_without_relaxing_guards():
    from sim.mujoco.rebot_b601_rs.haptic_demo import run_haptic_fixture_demo
    from sim.mujoco.rebot_b601_rs.scene import (
        B601_RS_PROVISIONAL_POSITION_KP,
        B601_RS_PROVISIONAL_POSITION_KV,
    )

    # Existing seeds match the pinned vendor YAML; they are not plant ID.
    code, report = run_haptic_fixture_demo(
        source_root=SOURCE,
        fixture_name="pose-6d",
        real2sim_controller=True,
        host_hz=500.0,
        startup_profile=B601SimStartupProfile(
            lock_kp_nm_per_rad=B601_RS_PROVISIONAL_POSITION_KP,
            lock_kd_nm_s_per_rad=B601_RS_PROVISIONAL_POSITION_KV,
        ),
    )
    acceptance = report["simulation_acceptance"]
    assert code == 0
    assert acceptance["passed"]
    assert acceptance["final_control_phase"] == "ACTIVE"
    assert not acceptance["driver_faulted"]


@pytest.mark.parametrize("stop", ["disable", "fault", "reset"])
def test_native_feedback_is_cleared_on_stop_or_reset(startup_driver, stop):
    driver = startup_driver
    driver.step_physics(reference(driver))
    ids = driver._actuator_ids
    assert np.any(driver.scene.model.actuator_biasprm[ids] != 0.0)
    if stop == "disable":
        driver.request_disable()
    elif stop == "fault":
        driver.scene.data.qvel[driver._qvel_addresses[0]] = 100.0
        assert driver.step_physics().driver_faulted
    else:
        driver.reset()
    np.testing.assert_array_equal(driver.scene.model.actuator_biasprm[ids], 0.0)
    # A nonzero velocity must not resurrect a disabled native brake or spring.
    if stop != "reset":
        driver.scene.forward()
        np.testing.assert_array_equal(driver.scene.data.actuator_force[ids], 0.0)


def test_native_total_force_limit_includes_local_spring(startup_driver):
    driver = startup_driver
    driver.profile = replace(driver.profile, max_abs_torque_nm=(0.001,) * 6)
    driver.scene.data.qpos[driver._qpos_addresses[0]] += 0.001
    driver.scene.forward()
    tick = driver.step_physics(reference(driver))
    assert tick.torque_limited
    np.testing.assert_allclose(
        tick.applied_torque_nm,
        np.clip(
            np.array(tick.host_command_torque_nm)
            + np.array(tick.driver_position_torque_nm)
            + np.array(tick.driver_damping_torque_nm),
            -0.001,
            0.001,
        ),
        atol=1e-12,
    )
