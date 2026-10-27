"""Real dual-arm physics with restored original actuator owners; no Gateway."""

import copy
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from sim.contracts import SimulationContract, TimingSpec
from sim.mujoco.actuation import (
    MujocoActuatedPositionExecutor,
    MujocoDmtacGripperExecutor,
    cad_bimanual_dmtac_gripper_profile,
    rm65_bimanual_udp_feedback_profile,
)
from sim.mujoco.actuation.checkpoint import (
    capture_arm_owner,
    capture_gripper_owner,
    restore_arm_owner,
    restore_gripper_owner,
)
from sim.mujoco.control import DirectQposCommand
from sim.mujoco.native_checkpoint import capture_native_state
from sim.mujoco.scene import MujocoScene

pytestmark = pytest.mark.integration
_BUDGET = 128 * 1024 * 1024


def _owners():
    arm_profile = rm65_bimanual_udp_feedback_profile()
    grip_profile = cad_bimanual_dmtac_gripper_profile()
    scene = MujocoScene.from_contract(
        SimulationContract.load("sim/configs/rm65_bimanual_physics_v1.json"),
        position_actuation_profile=arm_profile,
        gripper_actuation_profile=grip_profile,
    )
    timing = TimingSpec(
        sample_hz=15,
        policy_hz=15,
        reference_hz=60,
        ik_hz=60,
        command_hz=120,
        physics_hz=1000,
        parameter_source="checkpoint-test",
    )
    arm = MujocoActuatedPositionExecutor(scene, timing, arm_profile)
    grip = MujocoDmtacGripperExecutor(scene, grip_profile)
    initial = {side: scene.get_arm_qpos(side) for side in ("left", "right")}
    arm.reset(DirectQposCommand(0, initial, source="checkpoint-test"))
    grip.reset(
        {name: float(scene.get_gripper_close(name)) for name in scene.gripper_specs}
    )
    return scene, arm, grip


def _step(arm, grip, *, offset=0.001):
    command = None
    if arm.command_due:
        previous = arm._held_command
        command = DirectQposCommand(
            previous.sequence + 1,
            {
                side: np.full(6, offset if side == "left" else -offset)
                for side in ("left", "right")
            },
            source="checkpoint-test",
        )
    grip.before_physics_step(physics_time_sec=arm.physics_time_sec, timestep_sec=0.001)
    arm.step_physics(command)


def _restore_arm(state, scene, prototype):
    return restore_arm_owner(
        state,
        scene=scene,
        timing=prototype.timing,
        profile=prototype.profile,
        active_sides=prototype.active_sides,
        collision_projection_mode=prototype.collision_projection_mode,
        allow_gripper_self_contact=prototype.allow_gripper_self_contact,
    )


def test_original_arm_and_gripper_owners_continue_from_nonsettled_native_state(
    monkeypatch,
):
    scene, arm, grip = _owners()
    names = sorted(scene.gripper_specs)
    grip.queue_close_targets({names[0]: 0.6, names[1]: 0.25})
    for i in range(193):
        if i == 185:
            grip.queue_close_targets({names[1]: 0.7})
        _step(arm, grip)
    # Freeze a queued intent before its next physics/transport opportunity.
    grip.queue_close_targets({names[1]: 0.71})
    arm_state = capture_arm_owner(arm)
    grip_state = capture_gripper_owner(grip)
    assert arm_state.core["pending_delayed_targets"] > 0
    assert grip_state.core["pending"]
    assert any(grip_state.fields["_actuator_target_rate_by_gripper"].values()), (
        grip_state.core
    )
    physics = capture_native_state(scene.data, max_native_bytes=_BUDGET)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Owner restoration cannot reset or advance the scene.")

    with monkeypatch.context() as patch:
        for name in ("mj_forward", "mj_resetData", "mj_step"):
            patch.setattr(mujoco, name, forbidden)
        patch.setattr(MujocoActuatedPositionExecutor, "reset", forbidden)
        patch.setattr(MujocoDmtacGripperExecutor, "reset", forbidden)
        restored_scene = MujocoScene.from_native_checkpoint(
            physics,
            load_result=scene.load_result,
            max_native_bytes=_BUDGET,
        )
        restored_arm = _restore_arm(arm_state, restored_scene, arm)
        restored_grip = restore_gripper_owner(
            grip_state, scene=restored_scene, profile=grip.profile
        )
    assert (
        restored_scene.data.__getstate__() == physics.data == scene.data.__getstate__()
    )
    assert restored_arm.snapshot() == arm.snapshot()
    assert restored_grip.snapshot() == grip.snapshot()
    assert restored_grip._latest_command_by_gripper == grip._latest_command_by_gripper
    projection = restored_arm.last_safety_result
    assert projection is not None and arm.last_safety_result is not None
    for key in ("requested_qpos", "accepted_qpos"):
        np.testing.assert_array_equal(
            getattr(projection, key), getattr(arm.last_safety_result, key)
        )
        assert not np.shares_memory(
            getattr(projection, key), getattr(arm.last_safety_result, key)
        )
    assert not np.shares_memory(
        restored_arm._held_command.qpos_by_side["left"],
        arm._held_command.qpos_by_side["left"],
    )

    for i in range(40):
        if i == 15:
            for owner in (grip, restored_grip):
                owner.queue_close_targets({names[0]: 0.3})
        for a, g in ((arm, grip), (restored_arm, restored_grip)):
            _step(a, g, offset=-0.001)
        assert restored_arm.snapshot() == arm.snapshot()
        assert restored_grip.snapshot() == grip.snapshot()
        for field in ("qpos", "qvel", "ctrl", "act", "qacc_warmstart"):
            np.testing.assert_array_equal(
                getattr(restored_scene.data, field), getattr(scene.data, field)
            )
    assert restored_scene.collision_projector().oracle.model is restored_scene.model
    assert (
        restored_scene.collision_projector().scratch
        is not scene.collision_projector().scratch
    )


def test_invalid_components_fail_without_modifying_native_state():
    scene, arm, grip = _owners()
    state = capture_arm_owner(arm)
    grip_state = capture_gripper_owner(grip)
    before = scene.data.__getstate__()
    cases = [
        replace(state, configuration="different"),
        replace(state, core={}),
        replace(state, fields={}),
        replace(state, owner="gripper"),
    ]
    for bad in cases:
        with pytest.raises((ValueError, KeyError)):
            _restore_arm(bad, scene, arm)
        assert scene.data.__getstate__() == before
    native_time = float(scene.data.time)
    scene.data.time = native_time + 0.001
    clock_tampered = scene.data.__getstate__()
    for restore in (
        lambda: _restore_arm(state, scene, arm),
        lambda: restore_gripper_owner(grip_state, scene=scene, profile=grip.profile),
    ):
        with pytest.raises(ValueError, match="native clock"):
            restore()
        assert scene.data.__getstate__() == clock_tampered
    scene.data.time = native_time
    bad_fields = copy.deepcopy(grip_state.fields)
    bad_fields["_latest_command_by_gripper"] = {}
    with pytest.raises(ValueError):
        restore_gripper_owner(
            replace(grip_state, fields=bad_fields), scene=scene, profile=grip.profile
        )
    assert scene.data.__getstate__() == before
    name = next(iter(scene.gripper_specs))
    grip.queue_close_targets({name: 0.5})
    queued = capture_gripper_owner(grip)
    for receipt in (
        (queued.core["sequence"] + 1, 0.5),
        (queued.core["sequence"], 0.25),
        (0, 0.5),
        (1,),
    ):
        fields = copy.deepcopy(queued.fields)
        fields["_latest_command_by_gripper"][name] = receipt
        with pytest.raises(ValueError):
            restore_gripper_owner(
                replace(queued, fields=fields), scene=scene, profile=grip.profile
            )
        assert scene.data.__getstate__() == before
    scene.data.ctrl[arm._actuator_ids["left"][0]] += 0.1
    tampered = scene.data.__getstate__()
    with pytest.raises(ValueError, match="ctrl differ"):
        _restore_arm(state, scene, arm)
    assert scene.data.__getstate__() == tampered
    arm._hard_contact_latched = True
    with pytest.raises(ValueError, match="faulted"):
        capture_arm_owner(arm)
    grip.before_physics_step(physics_time_sec=0.0, timestep_sec=0.001)
    with pytest.raises(ValueError, match="native clock"):
        capture_gripper_owner(grip)


def test_existing_gripper_bootstrap_may_have_core_intent_without_local_receipt():
    scene, _arm, grip = _owners()
    name = next(iter(scene.gripper_specs))
    grip.queue_close_targets({name: 0.5})
    saved = grip.snapshot()
    bootstrapped = MujocoDmtacGripperExecutor(scene, grip.profile)
    bootstrapped.reset(
        saved["actuator_target_close_by_gripper"],
        initial_physics_time_sec=saved["physics_time_sec"],
        initial_state=saved,
    )
    assert bootstrapped.latest_command(name) == (0, None)
    assert bootstrapped._position_actuator.snapshot()["pending"]
    state = capture_gripper_owner(bootstrapped)
    before = scene.data.__getstate__()
    restored = restore_gripper_owner(state, scene=scene, profile=grip.profile)
    assert restored.latest_command(name) == (0, None)
    assert restored.snapshot() == bootstrapped.snapshot()
    assert scene.data.__getstate__() == before


def test_native_clock_keeps_accumulation_roundoff_but_rejects_mismatched_tick():
    from types import SimpleNamespace

    from sim.mujoco.actuation.checkpoint import _native_clock

    dt = 0.001
    native_time = 0.0
    for _ in range(1_000_000):
        native_time += dt
    assert abs(native_time - 1000.0) > 1e-9
    owner = SimpleNamespace(
        scene=SimpleNamespace(
            data=SimpleNamespace(time=native_time),
            model=SimpleNamespace(opt=SimpleNamespace(timestep=dt)),
        )
    )
    assert _native_clock(owner, 1000.0, saved_time=native_time) == native_time
    with pytest.raises(ValueError, match="native clock"):
        _native_clock(
            owner, 1000.0, saved_time=float(np.nextafter(native_time, float("inf")))
        )
    owner.scene.data.time += dt
    with pytest.raises(ValueError, match="native clock"):
        _native_clock(owner, 1000.0)
