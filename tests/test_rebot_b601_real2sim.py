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
    B601Real2SimPlantProfile,
    B601SimFaultPolicy,
    B601SimStartupProfile,
)

pytestmark = [pytest.mark.contract, pytest.mark.optional]
SOURCE = Path(__file__).resolve().parents[1] / "hardware/reBotArm_control_py"


@pytest.mark.parametrize("field", ["damping_nm_s_per_rad", "frictionloss_nm", "armature_kg_m2"])
@pytest.mark.parametrize("values", [(-0.01,) * 6, (float("nan"),) * 6, (1.0,) * 7])
def test_plant_profile_rejects_invalid_coefficients(field, values):
    with pytest.raises((TypeError, ValueError)):
        B601Real2SimPlantProfile(**{field: values})


def test_explicit_plant_terms_are_compiled_once_and_do_not_change_gravity():
    pytest.importorskip("mujoco")
    profile = B601Real2SimPlantProfile(
        damping_nm_s_per_rad=(0.01, 0.02, 0.03, 0.04, 0.05, 0.06),
        frictionloss_nm=(0.001,) * 6,
        armature_kg_m2=(0.002,) * 6,
    )
    baseline = build_b601_mujoco_scene(SOURCE, actuation_mode="effort", gravity_mps2=(0, 0, -9.81))
    scene = build_b601_mujoco_scene(SOURCE, actuation_mode="effort", gravity_mps2=(0, 0, -9.81), plant_profile=profile)
    dofs = [scene.joint_dof_addr(f"joint{i}") for i in range(1, 7)]
    np.testing.assert_array_equal(scene.model.dof_damping[dofs], profile.damping_nm_s_per_rad)
    np.testing.assert_array_equal(scene.model.dof_frictionloss[dofs], profile.frictionloss_nm)
    np.testing.assert_array_equal(scene.model.dof_armature[dofs], profile.armature_kg_m2)
    np.testing.assert_array_equal(scene.model.body_mass, baseline.model.body_mass)
    np.testing.assert_array_equal(scene.model.body_inertia, baseline.model.body_inertia)
    np.testing.assert_allclose(scene.data.qfrc_bias, baseline.data.qfrc_bias, atol=1e-12)
    masses = []
    for candidate in (baseline, scene):
        matrix = np.empty((candidate.model.nv, candidate.model.nv))
        candidate.mujoco.mj_fullM(candidate.model, candidate.data, matrix)
        masses.append(matrix)
    expected = np.zeros(scene.model.nv)
    expected[dofs] = profile.armature_kg_m2
    np.testing.assert_allclose(masses[1] - masses[0], np.diag(expected), atol=1e-12)
    scene.data.qvel[dofs] = (0.1, -0.1, 0.2, -0.2, 0.3, -0.3)
    scene.forward()
    np.testing.assert_allclose(
        scene.data.qfrc_passive[dofs],
        -np.asarray(profile.damping_nm_s_per_rad) * scene.data.qvel[dofs], atol=1e-12,
    )
    assert np.dot(scene.data.qfrc_passive, scene.data.qvel) < 0


def test_driver_requires_matching_declared_plant_and_keeps_controller_separate():
    pytest.importorskip("mujoco")
    pytest.importorskip("pinocchio")
    model = B601PinocchioModel.from_rebot_source_root(SOURCE)
    plant = B601Real2SimPlantProfile(damping_nm_s_per_rad=(0.01,) * 6)
    scene = build_b601_mujoco_scene(SOURCE, actuation_mode="effort", gravity_mps2=(0, 0, -9.81), plant_profile=plant)
    with pytest.raises(ValueError, match="zero passive"):
        B601MujocoEffortDriver(scene, model)
    with pytest.raises(ValueError, match="declared"):
        B601MujocoEffortDriver(scene, model, plant_profile=B601Real2SimPlantProfile())
    controller = B601Real2SimControllerProfile.raw_model_baseline()
    driver = B601MujocoEffortDriver(
        scene, model, enable_gravity_compensation=True,
        real2sim_profile=controller, plant_profile=plant,
    )
    first = driver.step_physics(reference(driver))
    assert not first.driver_faulted
    np.testing.assert_allclose(
        first.driver_damping_torque_nm,
        -np.asarray(controller.driver_kd_nm_s_per_rad) * first.control_qvel,
    )
    assert driver.execution_metadata["plant_profile"]["identification_status"] == "provisional_not_hardware_identified"
    driver.reset()
    np.testing.assert_array_equal(scene.model.dof_damping[driver._qvel_addresses], plant.damping_nm_s_per_rad)


def test_plant_profile_is_not_implicitly_enabled_for_position_mode():
    pytest.importorskip("mujoco")
    with pytest.raises(ValueError, match="effort"):
        build_b601_mujoco_scene(SOURCE, plant_profile=B601Real2SimPlantProfile())


def test_plant_dry_friction_opposes_small_external_joint_torque():
    pytest.importorskip("mujoco")
    speeds = []
    for plant in (None, B601Real2SimPlantProfile(frictionloss_nm=(0.1,) * 6)):
        scene = build_b601_mujoco_scene(SOURCE, actuation_mode="effort", plant_profile=plant)
        joint = scene.joint_dof_addr("joint1")
        scene.data.qfrc_applied[joint] = 0.01
        for _ in range(100):
            scene.mujoco.mj_step(scene.model, scene.data)
        speeds.append(abs(scene.data.qvel[joint]))
    assert speeds[0] > 1e-5
    assert speeds[1] < speeds[0] * 0.1


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


def test_real2sim_fault_policy_is_fixed_and_rejects_alternate_semantics():
    policy = B601SimFaultPolicy()
    metadata = policy.metadata()

    assert metadata["name"] == "b601_real2sim_feedback_fault_v1"
    assert metadata["scope"] == "real2sim_feedback_admission_and_lifecycle"
    assert metadata["stale_reference_action"] == "reject_keep_until_watchdog"
    assert metadata["invalid_or_wrong_frame_action"] == (
        "revoke_reference_keep_local_fallback"
    )
    assert metadata["watchdog_timeout_action"] == (
        "revoke_then_protective_hold_when_lifecycle_enabled"
    )
    assert metadata["nonfinite_state_or_torque_action"] == (
        "latch_fault_without_physics_step"
    )
    assert metadata["cached_transport_recovery"] is False
    assert metadata["hardware_safety_claim"] is False

    with pytest.raises(ValueError, match="supported simulation contract"):
        B601SimFaultPolicy(stale_reference_action="replace_with_cached_sample")


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


def test_real2sim_driver_exposes_fault_policy_without_changing_runtime_defaults(driver):
    metadata = driver.execution_metadata

    assert metadata["fault_policy"] == B601SimFaultPolicy().metadata()
    assert metadata["stale_sample_action"] == (
        "reject_without_preemption_until_local_watchdog"
    )
    assert metadata["unexpected_frame_action"] == (
        "deactivate_reference_keep_local_gravity_and_damping"
    )
    assert metadata["protective_hold_model"] == (
        "live_feedback_only_no_cached_transport_recovery"
    )


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


@pytest.mark.parametrize("field", ["fixture_stiffness_scale", "fixture_damping_scale"])
@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf"), True])
def test_fixture_gain_scales_reject_invalid_values_before_loading_model(field, value):
    from sim.mujoco.rebot_b601_rs.haptic_demo import run_haptic_fixture_demo

    with pytest.raises(ValueError, match=field):
        run_haptic_fixture_demo(
            source_root="not-a-vendor-checkout", fixture_name="pose-6d", **{field: value}
        )


@pytest.mark.parametrize("fixture_name", ["spring", "pose-6d"])
def test_zero_fixture_gains_are_reported_without_false_effect_acceptance(fixture_name):
    from sim.mujoco.rebot_b601_rs.haptic_demo import run_haptic_fixture_demo

    code, report = run_haptic_fixture_demo(
        source_root=SOURCE,
        fixture_name=fixture_name,
        fixture_stiffness_scale=0.0,
        fixture_damping_scale=0.0,
        duration_sec=0.02,
    )
    assert code == 3
    assert not report["simulation_acceptance"]["passed"]
    assert report["fixture_contract"]["stiffness_scale"] == 0.0
    assert report["fixture_contract"]["damping_scale"] == 0.0
    assert report["simulation_acceptance"]["run_completed"]


def test_fixture_gain_cli_forwards_explicit_scales(monkeypatch, capsys):
    from sim.mujoco.rebot_b601_rs import haptic_demo

    captured = {}

    def run(**kwargs):
        captured.update(kwargs)
        return 3, {"diagnostic": True}

    monkeypatch.setattr(haptic_demo, "run_haptic_fixture_demo", run)
    assert haptic_demo.main([
        "--rebot-source-root", "vendor", "--fixture-stiffness-scale", "0.25",
        "--fixture-damping-scale", "0.1",
        "--torque-rise-time-sec", "5",
    ]) == 3
    assert captured["fixture_stiffness_scale"] == 0.25
    assert captured["fixture_damping_scale"] == 0.1
    assert captured["torque_rise_time_sec"] == 5.0
    assert '"diagnostic": true' in capsys.readouterr().out


def test_wall_rejects_both_zero_gains_before_model_load():
    from sim.mujoco.rebot_b601_rs.haptic_demo import run_haptic_fixture_demo

    with pytest.raises(ValueError, match="wall stiffness or damping"):
        run_haptic_fixture_demo(
            source_root="not-a-vendor-checkout", fixture_name="wall",
            fixture_stiffness_scale=0.0, fixture_damping_scale=0.0,
        )


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), True])
def test_fixture_rate_rejects_invalid_values_before_model_load(value):
    from sim.mujoco.rebot_b601_rs.haptic_demo import run_haptic_fixture_demo

    with pytest.raises(ValueError, match="torque_rise_time_sec"):
        run_haptic_fixture_demo(
            source_root="not-a-vendor-checkout", fixture_name="pose-6d",
            torque_rise_time_sec=value,
        )


@pytest.mark.parametrize("fixture_name, expected_code", [("pose-6d", 0), ("pose-ry", 3)])
def test_tighter_feedforward_rate_preserves_effect_acceptance(fixture_name, expected_code):
    from sim.mujoco.rebot_b601_rs.haptic_demo import run_haptic_fixture_demo

    code, report = run_haptic_fixture_demo(
        source_root=SOURCE, fixture_name=fixture_name, real2sim_controller=True,
        host_hz=125.0, duration_sec=10.0, torque_rise_time_sec=5.0,
        startup_profile=B601SimStartupProfile(
            lock_kp_nm_per_rad=(50, 150, 150, 50, 50, 50),
            lock_kd_nm_s_per_rad=(3, 10, 10, 5, 4, 4),
        ),
    )
    assert code == expected_code
    a = report["simulation_acceptance"]
    assert a["run_completed"]
    assert not a["driver_faulted"]
    assert a["final_control_phase"] == "ACTIVE"
    profile = report["driver"]["profile"]
    np.testing.assert_allclose(
        profile["max_torque_rate_nm_s"], np.asarray(profile["max_abs_torque_nm"]) / 5.0
    )
    assert bool(report["effect_acceptance"]["passed"]) == (expected_code == 0)


def test_long_run_clock_roundoff_does_not_allow_foreign_step(startup_driver):
    driver = startup_driver
    for i in range(10000):
        tick = driver.step_physics(reference(driver, i) if driver.command_due else None)
        assert not tick.driver_faulted
    assert driver.scene.data.time != driver.physics_time_sec
    driver.scene.mujoco.mj_step(driver.scene.model, driver.scene.data)
    tick = driver.step_physics()
    assert tick.fault_reason == "physics_clock_desynchronized"
    assert not tick.plant_advanced


def test_power_diagnostic_separates_delivered_and_absorbed_work():
    from sim.mujoco.rebot_b601_rs.haptic_demo import _accumulate_power

    channels = {}
    _accumulate_power(channels, "probe", -2.0, 0.1)
    _accumulate_power(channels, "probe", 3.0, 0.1)
    r = channels["probe"]
    assert r["sample_count"] == 2
    assert r["net_work_j"] == pytest.approx(0.1)
    assert r["delivered_work_j"] == pytest.approx(0.3)
    assert r["absorbed_work_j"] == pytest.approx(0.2)
    assert r["peak_positive_power_w"] == 3.0


@pytest.mark.parametrize("host_hz, expected_code", [(125.0, 3), (500.0, 0)])
def test_power_observer_exposes_held_damping_without_changing_acceptance(host_hz, expected_code):
    from sim.mujoco.rebot_b601_rs.haptic_demo import run_haptic_fixture_demo

    code, report = run_haptic_fixture_demo(
        source_root=SOURCE, fixture_name="pose-6d", host_hz=host_hz,
        real2sim_controller=True,
        startup_profile=B601SimStartupProfile(
            lock_kp_nm_per_rad=(50, 150, 150, 50, 50, 50),
            lock_kd_nm_s_per_rad=(3, 10, 10, 5, 4, 4),
        ),
    )
    assert code == expected_code
    diagnostic = report["power_diagnostics"]
    assert not diagnostic["passivity_claim"]
    assert diagnostic["nominal_sampled_peak_power_w"] <= 1e-12
    channels = diagnostic["channels"]
    assert channels["nominal_motor_local_damping"]["delivered_work_j"] == 0.0
    for record in channels.values():
        assert record["sample_count"] > 0
        assert record["net_work_j"] == pytest.approx(
            record["delivered_work_j"] - record["absorbed_work_j"]
        )
    if host_hz == 125.0:
        # Each host sample is dissipative; the held torque need not stay so.
        assert channels["held_nominal_cartesian_damping"]["delivered_work_j"] > 0.0


def test_raw_gravity_baseline_preserves_local_damping():
    raw = B601Real2SimControllerProfile.raw_model_baseline()
    operator = B601Real2SimControllerProfile()
    assert raw.driver_kd_nm_s_per_rad == operator.driver_kd_nm_s_per_rad
    np.testing.assert_array_equal(raw.compensate(np.ones(6), np.ones(6)), np.ones(6))
    assert raw.calibration_source != operator.calibration_source


def test_raw_baseline_requires_real2sim_before_loading_model():
    from sim.mujoco.rebot_b601_rs.haptic_demo import main, run_haptic_fixture_demo

    with pytest.raises(ValueError, match="requires real2sim"):
        run_haptic_fixture_demo(
            source_root="missing", fixture_name="pose-ry",
            controller_profile=B601Real2SimControllerProfile.raw_model_baseline(),
        )
    with pytest.raises(SystemExit) as error:
        main(["--rebot-source-root", "missing", "--gravity-control-baseline", "raw-model"])
    assert error.value.code == 2


@pytest.mark.parametrize("host_hz, rise", [(125.0, 5.0), (500.0, None)])
def test_raw_gravity_removes_y_axis_overshoot_in_fixed_fixture(host_hz, rise):
    from sim.mujoco.rebot_b601_rs.haptic_demo import run_haptic_fixture_demo

    code, report = run_haptic_fixture_demo(
        source_root=SOURCE, fixture_name="pose-ry", real2sim_controller=True,
        host_hz=host_hz, duration_sec=10, torque_rise_time_sec=rise,
        controller_profile=B601Real2SimControllerProfile.raw_model_baseline(),
        startup_profile=B601SimStartupProfile(
            lock_kp_nm_per_rad=(50, 150, 150, 50, 50, 50),
            lock_kd_nm_s_per_rad=(3, 10, 10, 5, 4, 4),
        ),
    )
    assert code == 0
    assert report["effect_acceptance"]["rotation_passed"]


@pytest.fixture
def probe_trace():
    return {
        "mode": "startup-replay", "status": "FAIL",
        "active_joint_names": [f"joint{i}" for i in range(1, 7)],
        "config_sha256": "a" * 64, "urdf_sha256": "b" * 64,
        "freshness_verified": False,
        "samples": [
            {
                "sequence": i, "feedback_timestamp_sec": 10.0 + 0.05 * i,
                "freshness_verified": False,
                **{key: [0.0] * 6 for key in (
                    "qpos_rad", "qvel_fd_rad_s", "conditioned_tau_nm", "dispatch_qref_rad",
                    "vendor_kp_nm_per_rad", "vendor_kd_nm_s_per_rad",
                )},
            } for i in range(3)
        ],
    }


def test_existing_probe_trace_never_fabricates_replay_qualification(probe_trace):
    from sim.mujoco.rebot_b601_rs.real2sim import audit_b601_probe_trace

    result = audit_b601_probe_trace(probe_trace)
    assert result["sample_count"] == result["unverified_sample_count"] == 3
    assert result["poll_interval_sec"]["median"] == pytest.approx(0.05)
    assert result["source_termination_status"] == "FAIL"
    assert not result["quantitative_paired_replay_ready"]
    probe_trace["freshness_verified"] = True
    for row in probe_trace["samples"]:
        row["freshness_verified"] = True
    result = audit_b601_probe_trace(probe_trace)
    assert result["unverified_sample_count"] == 0
    assert not result["quantitative_paired_replay_ready"]
    assert not result["identity_verified_against_assets"]


@pytest.mark.parametrize("corruption", ["j7", "empty", "timestamp", "sequence", "nan", "partial", "negative_kd"])
def test_probe_trace_rejects_corrupt_or_ambiguous_rows(probe_trace, corruption):
    from sim.mujoco.rebot_b601_rs.real2sim import audit_b601_probe_trace

    row = probe_trace["samples"][1]
    if corruption == "j7":
        probe_trace["active_joint_names"][-1] = "joint7"
    elif corruption == "empty":
        probe_trace["samples"] = []
    elif corruption == "timestamp":
        row["feedback_timestamp_sec"] = 10.0
    elif corruption == "sequence":
        row["sequence"] = 0
    elif corruption == "nan":
        row["qpos_rad"][0] = float("nan")
    elif corruption == "partial":
        row["conditioned_tau_nm"].pop()
    else:
        row["vendor_kd_nm_s_per_rad"][0] = -1.0
    with pytest.raises(ValueError):
        audit_b601_probe_trace(probe_trace)


@pytest.mark.parametrize("failure", ["invalid", "wrong_frame", "dropout", "duplicate"])
def test_active_reference_failure_never_rearms_or_refreshes_watchdog(startup_driver, failure):
    driver = startup_driver
    for i in range(301):
        tick = driver.step_physics(reference(driver, i) if driver.command_due else None)
    assert tick.control_phase == "ACTIVE"
    last_sequence = driver._last_accepted_sequence
    last_acceptance_time = driver._accepted_at_physics_time_sec
    changed_q = list(driver._measured_qpos())
    changed_q[0] += 0.1
    for _ in range(201):
        command = None
        if failure == "invalid":
            command = replace(reference(driver, 999), valid=False)
        elif failure == "wrong_frame":
            command = reference(driver, 999, "wrong/frame")
        elif failure == "duplicate" and driver.command_due:
            command = replace(reference(driver, last_sequence), qpos_rad=tuple(changed_q))
        tick = driver.step_physics(command)
        assert not tick.driver_faulted
        assert driver._last_accepted_sequence == last_sequence
        assert driver._accepted_at_physics_time_sec in (last_acceptance_time, None)
        if tick.control_phase == "PROTECTIVE_HOLD":
            break
    assert tick.control_phase == "PROTECTIVE_HOLD"
    assert driver._active_reference is None
    assert driver.step_physics(reference(driver, 1000)).rejection_reason == "lifecycle_latched"
