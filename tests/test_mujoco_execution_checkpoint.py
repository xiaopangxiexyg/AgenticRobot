"""Real manual dual-arm rig cuts; no policy, Gateway or hardware operations."""

import copy
import hashlib
import json
import multiprocessing
import threading
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from robot_motion.position_control.trajectory_checkpoint import (
    capture_acceleration_planner,
)
from robot_motion.runtime import EffectorCommand, TeleopCommand
from sim.contracts import SimulationContract, TimingSpec
from sim.contracts.scene import SCENE_SPEC_VERSION_V2
from sim.mujoco.actuation import (
    MujocoActuatedPositionExecutor,
    MujocoDmtacGripperExecutor,
    cad_bimanual_dmtac_gripper_profile,
    rm65_bimanual_udp_feedback_profile,
)
from sim.mujoco.adapters import (
    MujocoPositionExecutionConfig,
    MujocoPositionExecutionRig,
)
from sim.mujoco.adapters.checkpoint import (
    ManualExecutionClock,
    capture_execution_rig,
    restore_execution_rig,
)
from sim.mujoco.scene import MujocoScene
from tests.test_mujoco_native_checkpoint import _continuation_witness

pytestmark = pytest.mark.integration
_BUDGET = 128 * 1024 * 1024


def _rig(contract=None):
    arm = rm65_bimanual_udp_feedback_profile()
    gripper = cad_bimanual_dmtac_gripper_profile()
    scene = MujocoScene.from_contract(
        contract
        or SimulationContract.load("sim/configs/rm65_bimanual_physics_v1.json"),
        position_actuation_profile=arm,
        gripper_actuation_profile=gripper,
    )
    config = MujocoPositionExecutionConfig(
        timing=TimingSpec(
            sample_hz=15,
            policy_hz=15,
            reference_hz=60,
            ik_hz=60,
            command_hz=120,
            physics_hz=1000,
        ),
        actuation_profile=arm,
        gripper_actuation_profile=gripper,
        realtime_worker=False,
        follow_qpos_alpha=0.7,
    )
    rig = MujocoPositionExecutionRig(
        scene, config, monotonic_clock=ManualExecutionClock(10.0)
    )
    rig.start()
    return rig


def _command(rig, offset, *, metadata=None):
    command = TeleopCommand(timestamp=rig._clock())
    for side in ("left", "right"):
        qpos = np.full(6, offset if side == "left" else -offset)
        command.add_command(
            EffectorCommand(
                effector_name=f"{side}_arm",
                side=side,
                command_type="joint_vector",
                value=qpos,
                metadata={} if metadata is None else metadata,
            )
        )
    return command


def _restore(state, prototype, **kwargs):
    return restore_execution_rig(
        state,
        config=prototype.config,
        load_result=prototype.scene.load_result,
        expected_model_sha256=state.native.model_sha256,
        max_native_bytes=_BUDGET,
        **kwargs,
    )


def _moving_cut(rig):
    names = sorted(rig.scene.gripper_specs)
    rig.send_commands(_command(rig, 0.004))
    rig.queue_gripper_close_targets({names[0]: 0.6, names[1]: 0.25})
    for i in range(193):
        if i == 85:
            rig._clock.advance_to(10.4)
            rig.send_commands(_command(rig, -0.003))
        if i == 185:
            rig.queue_gripper_close_targets({names[1]: 0.7})
        rig.step_physics()
    rig._clock.advance_to(11.0)
    rig.send_commands(
        _command(
            rig,
            0.005,
            metadata={
                "source_timestamp": 11.0,
                "nested": {"tags": ("keep", 3)},
                "witness": np.array([0.2, 0.3]),
            },
        )
    )
    rig.queue_gripper_close_targets({names[1]: 0.71})
    return capture_execution_rig(rig, max_native_bytes=_BUDGET)


def _equal(left, right, *, native=False):
    assert right.snapshot() == left.snapshot()
    assert right._clock() == left._clock()
    assert right._clock_origin == left._clock_origin
    for name in ("_next_reference_sequence", "_next_command_sequence"):
        assert getattr(right, name) == getattr(left, name)
    for name in ("qpos", "qvel", "qacc", "ctrl", "act", "qacc_warmstart"):
        np.testing.assert_array_equal(
            getattr(right.scene.data, name), getattr(left.scene.data, name)
        )
    for name in sorted(left.scene.gripper_specs):
        assert right.latest_gripper_command(name) == left.latest_gripper_command(name)
    if native:
        assert _continuation_witness(right.scene.data) == _continuation_witness(
            left.scene.data
        )


def test_nonsettled_rig_restores_pending_reference_and_continues_exactly(monkeypatch):
    source = _rig()
    state = _moving_cut(source)
    assert state.arm.core["pending_delayed_targets"] > 0
    assert state.gripper.core["pending"]
    assert any(state.gripper.fields["_actuator_target_rate_by_gripper"].values())
    assert state.fields["_pending_reference"] is not None

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Exact restore cannot start/reset/forward/step.")

    with monkeypatch.context() as patch:
        for name in ("mj_forward", "mj_resetData", "mj_step"):
            patch.setattr(mujoco, name, forbidden)
        patch.setattr(MujocoPositionExecutionRig, "start", forbidden)
        patch.setattr(MujocoActuatedPositionExecutor, "reset", forbidden)
        patch.setattr(MujocoDmtacGripperExecutor, "reset", forbidden)
        restored = _restore(state, source)
    assert restored.scene.data.__getstate__() == state.native.data
    assert restored._clock is not source._clock
    assert restored._planners["left"]._clock is restored._clock
    assert restored._execution is not source._execution
    assert restored._gripper_execution is not source._gripper_execution
    assert restored._pending_reference.sequence == source._pending_reference.sequence
    assert restored._pending_reference.metadata_by_side["left"]["nested"] == {
        "tags": ("keep", 3)
    }
    assert not np.shares_memory(
        restored._pending_reference.metadata_by_side["left"]["witness"],
        source._pending_reference.metadata_by_side["left"]["witness"],
    )
    _equal(source, restored, native=True)
    restored.start()  # Existing ready lifecycle is idempotent, not re-primed.
    _equal(source, restored)
    for i in range(90):
        for rig in (source, restored):
            if i == 15:
                rig._clock.advance_to(11.4)
                rig.send_commands(_command(rig, -0.006))
            if i == 32:
                rig.hold_motion("checkpoint-test")
            if i == 40:
                rig.hold_gripper_motion()
            if i == 60:
                rig._clock.advance_to(11.8)
                rig.send_commands(_command(rig, 0.001))
            rig.step_physics()
        _equal(source, restored, native=(i % 10 == 0 or i == 89))
    source_before = source.scene.data.__getstate__()
    restored.scene.data.qpos[0] += 0.1
    assert source.scene.data.__getstate__() == source_before
    source.close()
    restored.close()


def test_first_held_cut_restores_without_any_physics_step():
    source = _rig()
    state = capture_execution_rig(source, max_native_bytes=_BUDGET)
    restored = _restore(state, source)
    assert restored.snapshot()["physics_ticks"] == 0
    _equal(source, restored, native=True)
    source.step_physics()
    restored.step_physics()
    _equal(source, restored, native=True)
    source.close()
    restored.close()


def test_initialized_but_unprimed_planner_cannot_restore_a_started_rig():
    source = _rig()
    state = capture_execution_rig(source, max_native_bytes=_BUDGET)
    reset_only = source._new_planner()
    reset_only.reset(
        source._accepted_qcmd_by_side["left"], monotonic_time=source._clock_origin
    )
    replacement = dict(state.planners)
    replacement["left"] = capture_acceleration_planner(reset_only)
    # Reset-only is a valid standalone codec state, but start() always primes
    # both rig planners. Do not expose a ready rig whose first proposal fails.
    with pytest.raises(ValueError, match="primed"):
        _restore(replace(state, planners=replacement), source)
    assert source.is_ready()
    source.close()


def test_bootstrap_with_absolute_time_and_prior_command_watermark_can_be_captured():
    original = _rig()
    state = _moving_cut(original)
    scene = MujocoScene.from_native_checkpoint(
        state.native,
        load_result=original.scene.load_result,
        max_native_bytes=_BUDGET,
    )
    bootstrap = MujocoPositionExecutionRig(
        scene,
        original.config,
        monotonic_clock=ManualExecutionClock(30.0),
        initial_physics_time_sec=state.arm.core["physics_time_sec"],
        initial_command_sequence=state.arm.core["accepted_command_sequence"],
        initial_qcmd_by_side=state.fields["_accepted_qcmd_by_side"],
        initial_actuation_state=state.arm.core,
        initial_gripper_close_by_name=state.gripper.fields[
            "_accepted_close_by_gripper"
        ],
        initial_gripper_state=state.gripper.core,
    )
    bootstrap.start()
    cut = capture_execution_rig(bootstrap, max_native_bytes=_BUDGET)
    assert cut.fields["_worker_physics_ticks"] == 0
    assert cut.arm.core["physics_tick"] == 193
    assert cut.fields["_next_command_sequence"] > 1
    restored = _restore(cut, bootstrap)
    _equal(bootstrap, restored, native=True)
    for _ in range(15):
        bootstrap.step_physics()
        restored.step_physics()
        _equal(bootstrap, restored)
    for rig in (original, bootstrap, restored):
        rig.close()


def test_hold_cut_and_restored_close_start_keep_existing_lifecycle_semantics():
    source = _rig()
    _moving_cut(source)
    source.hold_motion("operator hold")
    state = capture_execution_rig(source, max_native_bytes=_BUDGET)
    restored = _restore(state, source)
    assert restored._pending_reference is None
    assert restored._last_reference is None
    _equal(source, restored)
    for _ in range(15):
        source.step_physics()
        restored.step_physics()
        _equal(source, restored)
    source.close()
    restored.close()
    restored.start()
    assert restored.snapshot()["physics_ticks"] == 0
    assert restored._next_reference_sequence == 1
    assert restored._next_command_sequence == 1
    assert restored._last_reference is None
    assert restored._pending_reference is None
    assert restored._clock_origin == restored._clock()
    restored.close()


def test_two_rig_branches_keep_native_and_control_state_independent():
    source = _rig()
    state = _moving_cut(source)
    first, second = _restore(state, source), _restore(state, source)
    source_before = source.scene.data.__getstate__()
    source_snapshot = source.snapshot()
    for branch, offset in ((first, 0.01), (second, -0.01)):
        branch.send_commands(_command(branch, offset))
        for _ in range(35):
            branch.step_physics()
    assert first._last_reference.sequence == second._last_reference.sequence
    assert not np.array_equal(first.scene.data.qpos, second.scene.data.qpos)
    assert source.scene.data.__getstate__() == source_before
    assert source.snapshot() == source_snapshot
    first._clock.advance_to(12.0)
    assert source._clock() == second._clock() == 11.0
    first._planners["left"]._q[0] += 0.1
    assert not np.shares_memory(first._planners["left"]._q, second._planners["left"]._q)
    assert source.scene.data.__getstate__() == source_before
    for rig in (source, first, second):
        rig.close()


def _rig_digest(rig):
    payload = {
        "native": _continuation_witness(rig.scene.data),
        "rig": rig.snapshot(),
        "clock": rig._clock(),
        "origin": rig._clock_origin,
        "next_reference": rig._next_reference_sequence,
        "next_command": rig._next_command_sequence,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _restore_in_forked_process(state, config, metadata, connection):
    # The checkpoint itself is inherited read-only, never sent to a pickle
    # importer. This tests process isolation, not a durable/fresh-exec format.
    try:
        rig = restore_execution_rig(
            state,
            config=config,
            load_result=metadata,
            expected_model_sha256=state.native.model_sha256,
            max_native_bytes=_BUDGET,
        )
        for _ in range(25):
            rig.step_physics()
        connection.send(_rig_digest(rig))
        rig.close()
    finally:
        connection.close()


def test_rig_components_restore_in_separate_forked_process():
    source = _rig()
    state = _moving_cut(source)
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_restore_in_forked_process,
        args=(
            state,
            source.config,
            source.scene.load_result,
            child,
        ),
    )
    process.start()
    child.close()
    try:
        for _ in range(25):
            source.step_physics()
        assert parent.poll(40), "Forked restore did not complete."
        assert parent.recv() == _rig_digest(source)
        process.join(5)
        assert process.exitcode == 0
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
            process.join(5)
        source.close()


def test_merged_scene_v2_metadata_and_native_primitive_survive_rig_restore():
    baseline = SimulationContract.load("sim/configs/rm65_bimanual_physics_v1.json")
    primitive = replace(
        baseline.scene.objects[0],
        shape="cylinder",
        size_m=(0.04, 0.04, 0.06),
        inertia_diagonal_kg_m2=None,
    )
    contract = replace(
        baseline,
        scene=replace(
            baseline.scene, schema_version=SCENE_SPEC_VERSION_V2, objects=(primitive,)
        ),
    )
    source = _rig(contract)
    for _ in range(31):
        source.step_physics()
    state = capture_execution_rig(source, max_native_bytes=_BUDGET)
    restored = _restore(state, source)
    assert restored.scene.scene_spec.to_dict() == contract.scene.to_dict()
    assert restored.scene.scene_spec is not source.scene.scene_spec
    geom = restored.scene.model.geom(primitive.name + "_collision")
    assert geom.type[0] == mujoco.mjtGeom.mjGEOM_CYLINDER
    for _ in range(20):
        source.step_physics()
        restored.step_physics()
    _equal(source, restored, native=True)
    for rig in (source, restored):
        rig.close()


def test_malformed_cut_fails_without_modifying_source():
    source = _rig()
    state = _moving_cut(source)
    source_before = source.scene.data.__getstate__()
    cases = [
        replace(state, configuration="wrong"),
        replace(state, scene_metadata="wrong"),
        replace(state, clock_value=float("nan")),
        replace(state, clock_value=0.0),
        replace(state, fields={}),
        replace(state, planners={}),
    ]
    for name, value in (
        ("_next_command_sequence", state.fields["_next_command_sequence"] + 1),
        ("_next_reference_sequence", 1),
        ("_worker_physics_ticks", 10**9),
        ("_worker_deadline_misses", 1),
        ("_clock_origin", -1000.0),
        ("_hold_reason", "unfinished hold"),
    ):
        broken = copy.deepcopy(state.fields)
        broken[name] = value
        cases.append(replace(state, fields=broken))
    broken = copy.deepcopy(state.fields)
    broken["_accepted_qcmd_by_side"]["left"][0] += 0.1
    cases.append(replace(state, fields=broken))
    for metadata in (None, [], np.array([1.0])):
        broken = copy.deepcopy(state.fields)
        broken["_pending_reference"].metadata_by_side["left"] = metadata
        cases.append(replace(state, fields=broken))
    broken = copy.deepcopy(state.fields)
    broken["_last_reference"] = replace(broken["_last_reference"], sequence=1)
    cases.append(replace(state, fields=broken))
    for case in cases:
        with pytest.raises(ValueError):
            _restore(case, source)
        assert source.scene.data.__getstate__() == source_before
    with pytest.raises(ValueError, match="identity"):
        restore_execution_rig(
            state,
            config=source.config,
            load_result=source.scene.load_result,
            expected_model_sha256="0" * 64,
            max_native_bytes=_BUDGET,
        )
    source.close()


def test_unsupported_bindings_and_layouts_are_rejected_without_clearing_them(
    monkeypatch,
):
    source = _rig()
    for name, value in (
        ("_worker", object()),
        ("_clock", lambda: 10.0),
        ("_last_worker_error", "failure"),
        ("_pending_proposals", {"left": object()}),
        ("_physics_step_observer", lambda _scene: None),
        ("_scheduled_activation_guard", lambda *_args: "activation"),
        ("_scheduled_active_work_guard", lambda *_args: None),
        ("_reference_activation_observations", [object()]),
        ("_planner_command_frame_observations", [object()]),
        ("_initial_qcmd_by_side", source.accepted_qcmd_by_side()),
        ("config", replace(source.config, realtime_worker=True)),
        (
            "config",
            replace(source.config, collision_projection_mode="upstream_admitted"),
        ),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(source, name, value)
            with pytest.raises(ValueError):
                capture_execution_rig(source, max_native_bytes=_BUDGET)
            assert getattr(source, name) is value
    with monkeypatch.context() as patch:
        patch.setattr(source, "unaccounted_state", 3, raising=False)
        with pytest.raises(ValueError, match="layout"):
            capture_execution_rig(source, max_native_bytes=_BUDGET)
    for name in ("_final_command_guard", "_additional_contact_oracle"):
        with monkeypatch.context() as patch:
            patch.setattr(source._execution, name, object(), raising=False)
            with pytest.raises(ValueError, match="guard bindings"):
                capture_execution_rig(source, max_native_bytes=_BUDGET)
    source.close()
    with pytest.raises(ValueError):
        capture_execution_rig(source, max_native_bytes=_BUDGET)


def test_reentrant_observer_and_pending_proposal_cannot_capture():
    source = _rig()
    source.step_physics()  # Next non-command tick has no proposal.
    failures = []

    def observer(_scene):
        with pytest.raises(ValueError, match="callback-free"):
            capture_execution_rig(source, max_native_bytes=_BUDGET)
        failures.append(True)

    source.set_physics_step_observer(observer)
    source.step_physics()
    assert failures == [True]
    source.close()
    source = _rig()
    proposal = source._planners["left"].propose(monotonic_time=source._clock())
    with pytest.raises(ValueError, match="completed command"):
        capture_execution_rig(source, max_native_bytes=_BUDGET)
    source._planners["left"].reject(proposal)
    source.close()


def test_unsupported_metadata_is_not_silently_stringified():
    source = _rig()
    source.send_commands(_command(source, 0.001, metadata={"live": object()}))
    with pytest.raises(ValueError, match="metadata"):
        capture_execution_rig(source, max_native_bytes=_BUDGET)
    assert source.snapshot()["pending_reference_sequence"] == 1
    source.close()


def test_concurrent_close_invalidates_cut_before_publication(monkeypatch):
    import sim.mujoco.adapters.checkpoint as codec

    source = _rig()
    captured = threading.Event()
    close_requested = threading.Event()
    finished = threading.Event()
    native_capture = codec.capture_native_state

    def capture_then_wait(*args, **kwargs):
        result = native_capture(*args, **kwargs)
        captured.set()
        assert close_requested.wait(5)
        assert source._stop_event.is_set()
        return result

    def closer():
        assert captured.wait(5)
        # Match the public close path's first action before it waits on lock.
        source._stop_event.set()
        close_requested.set()
        source.close()
        finished.set()

    monkeypatch.setattr(codec, "capture_native_state", capture_then_wait)
    thread = threading.Thread(target=closer)
    thread.start()
    try:
        with pytest.raises(ValueError, match="callback-free"):
            capture_execution_rig(source, max_native_bytes=_BUDGET)
    finally:
        thread.join(5)
    assert finished.is_set()
    assert not source.is_ready()


@pytest.mark.parametrize("value", [float("nan"), -1, True, "1"])
def test_manual_clock_rejects_invalid_coordinates(value):
    with pytest.raises(ValueError):
        ManualExecutionClock(value)


def test_manual_clock_is_monotonic_without_implicit_physics_or_wall_time():
    clock = ManualExecutionClock(10)
    clock.advance_to(11)
    assert clock() == 11
    with pytest.raises(ValueError, match="backwards"):
        clock.advance_to(10)
