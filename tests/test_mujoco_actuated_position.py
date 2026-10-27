from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from robot_motion.runtime import EffectorCommand, TeleopCommand
from sim.contracts import SimulationContract, TimingSpec
from sim.mujoco.actuation import (
    COLLISION_PROJECTION_MODE_UPSTREAM_ADMITTED,
    MujocoActuatedPositionExecutor,
    rm65_bimanual_udp_feedback_profile,
    rm65_bimanual_udp_feedback_profile_v1,
    rm65_bimanual_udp_feedback_profile_v2,
    rm65_bimanual_udp_feedback_profile_v3,
)
from sim.mujoco.adapters import (
    MujocoPositionExecutionConfig,
    MujocoPositionExecutionRig,
)
from sim.mujoco.control import DirectQposCommand
from sim.mujoco.scene import MujocoScene
from sim.mujoco.collision_safety import (
    CollisionCheckResult, CollisionProjectionResult, CollisionSafetyOracle,
)
from sim.rm65.actuation import PositionActuationProfile
from sim.mujoco.actuation.executor import MujocoPhysicalContactFault


CONTRACT_PATH = "sim/configs/rm65_bimanual_physics_v1.json"


def _timing() -> TimingSpec:
    return TimingSpec(
        sample_hz=15.0,
        policy_hz=15.0,
        reference_hz=60.0,
        ik_hz=60.0,
        command_hz=120.0,
        physics_hz=1000.0,
        parameter_source="unit_test_15_60_120",
    )


def test_rm65_execution_config_defaults_to_fitted_arm_and_grippers() -> None:
    config = MujocoPositionExecutionConfig(timing=_timing())

    assert config.position_command_planner == "acceleration_limited"
    assert config.sample_aligned_source_hz == pytest.approx(15.0)
    assert config.sample_aligned_horizon_source_steps == 2
    assert config.actuation_profile is not None
    assert config.actuation_profile.name == "rm65_bimanual_udp_feedback_20260721.v4"
    assert config.gripper_actuation_profile is not None
    assert (
        config.gripper_actuation_profile.name
        == "dmtac_bimanual_position_20260722.v1"
    )


def test_atomic_gripper_precondition_never_queues_after_a_rejected_scene_cut() -> None:
    rig = object.__new__(MujocoPositionExecutionRig)
    rig._lock = threading.RLock()
    rig._started = True
    rig._stop_event = threading.Event()
    rig._last_worker_error = None
    rig.scene = SimpleNamespace(data=object())
    queued = []
    latest = {"right": (4, 1.)}

    def queue(values):
        queued.append(dict(values))
        latest["right"] = (8, values["right"])
        return 8

    def rollback(sequence, previous):
        assert sequence == 8
        queued.clear()
        latest.update(previous)

    rig._gripper_execution = SimpleNamespace(queue_close_targets=lambda values:
        queued.append(dict(values)) or 7,
        latest_command=lambda name: latest[name],
        rollback_unconsumed_command=rollback)

    def reject(_scene):
        raise ValueError("measured relation unavailable")

    with pytest.raises(ValueError, match="measured relation unavailable"):
        rig.queue_gripper_close_targets_after_scene_check({"right": 0.}, scene_check=reject)
    assert queued == []
    sequence, cut = rig.queue_gripper_close_targets_after_scene_check(
        {"right": 0.}, scene_check=lambda scene: scene.data)
    assert sequence == 7 and cut is rig.scene.data
    assert queued == [{"right": 0.}]
    committed = []
    cancelled = {"value": False}

    def cancel_during_observation(scene):
        cancelled["value"] = True
        return scene.data

    sequence, _cut = rig.queue_gripper_close_targets_after_scene_check(
        {"right": 0.}, scene_check=cancel_during_observation,
        cancelled=lambda: cancelled["value"],
        on_queued=lambda _scene, _cut: committed.append(True))
    assert sequence is None and queued == [{"right": 0.}] and committed == []

    rig._gripper_execution.queue_close_targets = lambda _values: (
        _ for _ in ()).throw(RuntimeError("lower queue rejected"))
    with pytest.raises(RuntimeError, match="lower queue rejected"):
        rig.queue_gripper_close_targets_after_scene_check(
            {"right": 0.}, scene_check=lambda scene: scene.data,
            on_queued=lambda _scene, _cut: committed.append(True))
    assert committed == []
    rig._gripper_execution.queue_close_targets = queue
    restored = []

    def fail_after_queue(_scene, _cut):
        committed.append(True)
        raise RuntimeError("observer commit failed")

    with pytest.raises(RuntimeError, match="observer commit failed"):
        rig.queue_gripper_close_targets_after_scene_check(
            {"right": 0.}, scene_check=lambda scene: scene.data,
            on_queued=fail_after_queue,
            on_queue_failed=lambda _scene, _cut: restored.append(True))
    assert queued == [] and restored == [True] and latest["right"] == (4, 1.)
    rig._stop_event.set()
    with pytest.raises(RuntimeError, match="not ready"):
        rig.queue_gripper_close_targets_after_scene_check(
            {"right": 0.}, scene_check=lambda _scene: pytest.fail("must not observe"))
    assert queued == []


def _scene_and_profile():
    contract = SimulationContract.load(CONTRACT_PATH)
    profile = rm65_bimanual_udp_feedback_profile()
    scene = MujocoScene.from_contract(contract, position_actuation_profile=profile)
    return scene, profile


def test_rm65_profile_records_joint_fit_without_claiming_force_identification() -> None:
    profile = rm65_bimanual_udp_feedback_profile()

    assert profile.name == "rm65_bimanual_udp_feedback_20260721.v4"
    assert profile.transport_delay_sec_by_side == {"left": 0.018, "right": 0.017}
    assert profile.command_delay_sec("left", 5) == pytest.approx(
        0.009527118125949966
    )
    threshold = np.deg2rad(50.0)
    assert profile.target_filter_tau_sec(
        "left", 4, abs_command_velocity_rad_s=threshold
    ) == 0.0
    assert profile.target_filter_tau_sec(
        "left", 4, abs_command_velocity_rad_s=np.deg2rad(60.0)
    ) > 0.0
    assert profile.target_filter_tau_sec(
        "left", 4, abs_command_velocity_rad_s=np.deg2rad(200.0)
    ) == pytest.approx(0.01845353602325293)
    assert profile.target_filter_tau_sec(
        "left", 3, abs_command_velocity_rad_s=np.deg2rad(200.0)
    ) == 0.0
    assert profile.target_filter_tau_sec(
        "left", 5, abs_command_velocity_rad_s=np.deg2rad(200.0)
    ) == pytest.approx(0.01278000865660236)
    assert "fourteen_valid_traces_distal_phase_refitted" in profile.calibration_status
    assert "force_contact_provisional" in profile.calibration_status
    assert rm65_bimanual_udp_feedback_profile_v1().name.endswith(".v1")
    assert rm65_bimanual_udp_feedback_profile_v2().name.endswith(".v2")
    assert rm65_bimanual_udp_feedback_profile_v3().name.endswith(".v3")
    assert rm65_bimanual_udp_feedback_profile_v2().command_delay_sec(
        "left", 5
    ) == pytest.approx(0.0110777)
    assert rm65_bimanual_udp_feedback_profile_v3().command_delay_sec(
        "left", 5
    ) == pytest.approx(0.0108740)


def _command(sequence: int, qpos_by_side) -> DirectQposCommand:
    return DirectQposCommand(
        sequence=sequence,
        qpos_by_side=qpos_by_side,
        source="unit_120hz_qcmd",
    )


@pytest.mark.optional
def test_native_held_control_preview_preserves_live_causal_and_physics_state() -> None:
    pytest.importorskip("mujoco")
    scene, profile = _scene_and_profile()
    sender = MujocoActuatedPositionExecutor(scene, _timing(), profile)
    initial = {s: scene.get_arm_qpos(s) for s in ("left", "right")}
    sender.reset(_command(0, initial))
    sequence = 1
    while sender.physics_time_sec < 0.026:
        command = _command(sequence, initial) if sender.command_due else None
        if command is not None:
            sequence += 1
        sender.step_physics(command)
    before = sender.snapshot()
    physics = {n: getattr(scene.data, n).copy() for n in ("qpos", "qvel", "ctrl", "qacc_warmstart")}
    scene_time = float(scene.data.time)
    assert before["pending_delayed_targets"] > 0
    candidate = _command(sequence, initial)
    assert sender.preview_held_command_is_safe(
        candidate, qpos_check=lambda pair: all(np.all(np.isfinite(q)) for q in pair.values()),
        max_preview_time_sec=1.0,
    )
    assert not sender.preview_held_command_is_safe(
        candidate, qpos_check=lambda pair: False, max_preview_time_sec=1.0,
    )
    assert sender.snapshot() == before
    assert float(scene.data.time) == scene_time
    for name, values in physics.items():
        np.testing.assert_array_equal(getattr(scene.data, name), values)


def test_final_command_fault_rejects_before_transport_and_cannot_restore_unbound() -> None:
    scene = _PhysicalFakeScene(None)
    sender = MujocoActuatedPositionExecutor(scene, _timing(), _physical_test_profile())
    initial = {s: scene.get_arm_qpos(s) for s in ("left", "right")}
    sender.reset(_command(0, initial))
    fraction = [1.0]
    guard = SimpleNamespace(
        admission_fraction=lambda *a, **k: fraction[0],
        check_qpos_pair=lambda pair: None,
        snapshot=lambda: {"binding": {"profile_hash": "test-bound-envelope"}},
    )
    sender.install_final_command_guard(guard)
    with pytest.raises(RuntimeError, match="already installed"):
        sender.install_final_command_guard(guard)
    while not sender.command_due:
        sender.step_physics()
    before = sender.snapshot()
    steps = scene.data.physics_steps
    fraction[0] = float("nan")
    with pytest.raises(ValueError, match="fraction"):
        sender.step_physics(_command(1, initial))
    assert scene.data.physics_steps == steps
    assert sender.snapshot()["accepted_command_sequence"] == before["accepted_command_sequence"]
    assert sender.snapshot()["final_command_guard"]["fault_latched"] is True
    with pytest.raises(RuntimeError, match="faulted"):
        sender.step_physics(_command(1, initial))
    unbound = MujocoActuatedPositionExecutor(scene, _timing(), _physical_test_profile())
    with pytest.raises(ValueError, match="guard binding"):
        unbound.reset(_command(0, initial), initial_state=before)


@pytest.mark.optional
def test_actuated_executor_is_only_the_120hz_sender_and_physical_plant() -> None:
    pytest.importorskip("mujoco")
    scene, profile = _scene_and_profile()
    sender = MujocoActuatedPositionExecutor(scene, _timing(), profile)
    initial = {side: scene.get_arm_qpos(side) for side in ("left", "right")}
    target = {side: values + 0.01 for side, values in initial.items()}
    sender.reset(_command(0, initial))

    # Reset owns the already-reached initial target. The next sender call is
    # due on the independent 120 Hz deadline, not at 20 Hz or 60 Hz.
    while not sender.command_due:
        tick = sender.step_physics()
        assert not tick.command_applied
    tick = sender.step_physics(_command(1, target))
    assert tick.command_applied
    assert all(
        np.allclose(sender.accepted_arm_qpos(side), target[side])
        for side in ("left", "right")
    )

    # Keep satisfying subsequent 120 Hz deadlines with the same qcmd. The
    # accepted command is immediate, while actuator ctrl changes after the
    # measured per-side transport delay.
    sequence = 2
    while sender.physics_time_sec < 0.045:
        command = None
        if sender.command_due:
            command = _command(sequence, target)
            sequence += 1
        sender.step_physics(command)
    left_ids = [
        scene.mujoco.mj_name2id(
            scene.model,
            scene.mujoco.mjtObj.mjOBJ_ACTUATOR,
            profile.actuator_name("left", index),
        )
        for index in range(6)
    ]
    assert np.allclose(scene.data.ctrl[left_ids], target["left"])
    assert scene.state_is_finite()
    assert sender.execution_metadata["command_hz"] == 120.0


@pytest.mark.optional
def test_high_speed_distal_target_lag_is_causal_and_keeps_raw_targets() -> None:
    pytest.importorskip("mujoco")
    scene, profile = _scene_and_profile()
    sender = MujocoActuatedPositionExecutor(scene, _timing(), profile)
    initial = {side: scene.get_arm_qpos(side) for side in ("left", "right")}
    sender.reset(_command(0, initial))

    target = {side: values.copy() for side, values in initial.items()}
    sequence = 1
    saw_filtered_target = False
    while sender.physics_time_sec < 0.08:
        command = None
        if sender.command_due:
            # One radian/second is above the fitted 50 deg/s threshold.
            target["left"][4] += 1.0 / 120.0
            command = _command(sequence, target)
            sequence += 1
        sender.step_physics(command)
        snapshot = sender.snapshot()
        tau = snapshot["actuator_target_tau_sec_by_side"]["left"][4]
        raw = snapshot["raw_actuator_target_by_side"]["left"][4]
        filtered = snapshot["actuator_target_by_side"]["left"][4]
        if tau > 0.0 and not np.isclose(raw, filtered):
            saw_filtered_target = True

    assert saw_filtered_target
    assert np.isclose(sender.accepted_arm_qpos("left")[4], target["left"][4])
    assert sender.snapshot()["pending_delayed_targets"] < 60


@pytest.mark.optional
def test_real_aligned_rig_keeps_15_60_120_and_swaps_only_sender() -> None:
    pytest.importorskip("mujoco")
    scene, profile = _scene_and_profile()
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
                timing=_timing(),
                actuation_profile=profile,
                gripper_actuation_profile=None,
                realtime_worker=False,
            follow_qpos_alpha=1.0,
        ),
    )
    rig.start()
    reference = TeleopCommand(timestamp=0.0)
    for side, sign in (("left", 1.0), ("right", -1.0)):
        reference.add_command(
            EffectorCommand(
                effector_name=f"{side}_arm",
                side=side,
                command_type="joint_vector",
                value=scene.get_arm_qpos(side) + sign * 0.02,
            )
        )
    rig.send_commands(reference)
    for _ in range(45):
        rig.step_physics()

    snapshot = rig.snapshot()
    assert snapshot["timing"]["sample_hz"] == 15.0
    assert snapshot["timing"]["reference_hz"] == 60.0
    assert snapshot["timing"]["command_hz"] == 120.0
    assert snapshot["execution_mode"] == "actuated_position_servo"
    assert snapshot["robot_dynamics"] == "mujoco_position_actuator"
    assert snapshot["execution"]["accepted_command_sequence"] is not None
    assert rig._direct is None


class _PhysicalFakeMujoco:
    class mjtObj:
        mjOBJ_ACTUATOR = 1

    def mj_name2id(self, _model, _object_type, name: str) -> int:
        try:
            return int(str(name).rsplit("joint", 1)[1]) - 1
        except (IndexError, ValueError):
            return -1

    def mj_step(self, _model, data) -> None:
        data.physics_steps = int(getattr(data, "physics_steps", 0)) + 1


class _PhysicalFakeScene:
    def __init__(self, projector) -> None:
        names = {
            side: tuple(f"{side}_joint{index}" for index in range(1, 7))
            for side in ("left", "right")
        }
        self.arm_specs = {
            side: SimpleNamespace(joint_names=joint_names)
            for side, joint_names in names.items()
        }
        self.mujoco = _PhysicalFakeMujoco()
        self.model = SimpleNamespace(opt=SimpleNamespace(timestep=0.001))
        self.data = SimpleNamespace(
            ctrl=np.zeros(12, dtype=float),
            qpos=np.zeros(12, dtype=float),
            physics_steps=0,
        )
        self._projector = projector
        self.collision_safety_available = True

    def collision_projector(self):
        return self._projector

    def get_arm_qpos(self, side: str) -> np.ndarray:
        offset = 0 if side == "left" else 6
        return self.data.qpos[offset : offset + 6].copy()

    def kinematic_controlled_qpos(self) -> np.ndarray:
        return self.data.qpos.copy()

    def kinematic_target_vector(self, base_qpos, qpos_by_side, **_kwargs) -> np.ndarray:
        target = np.asarray(base_qpos, dtype=float).reshape(12).copy()
        for side, values in qpos_by_side.items():
            offset = 0 if side == "left" else 6
            target[offset : offset + 6] = np.asarray(values, dtype=float).reshape(6)
        return target

    def arm_qpos_from_kinematic_vector(self, controlled_qpos, side: str) -> np.ndarray:
        offset = 0 if side == "left" else 6
        return np.asarray(controlled_qpos, dtype=float).reshape(12)[offset : offset + 6].copy()


def _physical_test_profile() -> PositionActuationProfile:
    return PositionActuationProfile(
        name="physical_executor_test",
        joint_names_by_side={
            side: tuple(f"{side}_joint{index}" for index in range(1, 7))
            for side in ("left", "right")
        },
        transport_delay_sec_by_side={"left": 0.0, "right": 0.0},
        position_kp_nm_rad=(1.0,) * 6,
        position_kv_nm_s_rad=(1.0,) * 6,
        force_limit_nm=(1.0,) * 6,
    )


def _safe_check() -> CollisionCheckResult:
    return CollisionCheckResult(safe=True, minimum_distance_m=0.02)


def _fake_projection(start, requested) -> CollisionProjectionResult:
    start = np.asarray(start, dtype=float)
    requested = np.asarray(requested, dtype=float)
    return CollisionProjectionResult(
        status="projected",
        requested_qpos=requested.copy(),
        accepted_qpos=start + 0.5 * (requested - start),
        projection_alpha=0.5,
        modified=True,
        check=_safe_check(),
        sweep_samples=2,
    )


@pytest.mark.parametrize("post_contact_is_safe", [True, False])
@pytest.mark.parametrize("guard_source", ["canonical", "additional"])
def test_physical_executor_runs_full_scene_guard_and_latches_hard_contact(
    post_contact_is_safe: bool,
    guard_source: str,
) -> None:
    class _Oracle(CollisionSafetyOracle):
        def __init__(self) -> None:
            # This fake supplies contact results, not native geometry queries.
            self.calls = 0

        def check_contacts(self, _data):
            self.calls += 1
            # The executor checks both sides of every mj_step.  Keep the
            # pre-step/reset checks safe and inject the fault only after the
            # fake physics owner has advanced once so this regression covers
            # the post-step latch rather than only the pre-step guard.
            if (
                int(getattr(_data, "physics_steps", 0)) > 0
                and not post_contact_is_safe
            ):
                return CollisionCheckResult(
                    safe=False,
                    minimum_distance_m=-0.002,
                    body1_name="arm",
                    body2_name="obstacle",
                    pair_class="1:2",
                )
            return _safe_check()

    oracle = _Oracle()
    projector = SimpleNamespace(
        controlled_qpos_indices=np.arange(12, dtype=np.int32),
        policy=SimpleNamespace(hard_penetration_m=0.001),
        oracle=oracle,
        check_qpos=lambda _data, _qpos, **_kwargs: _safe_check(),
        project=lambda _data, *, start_qpos, requested_qpos, **_kwargs: _fake_projection(
            start_qpos, requested_qpos
        ),
        recover_from_clearance_violation=lambda _data, *, start_qpos, requested_qpos: _fake_projection(
            start_qpos, requested_qpos
        ),
    )
    scene = _PhysicalFakeScene(projector)
    oracle.model = scene.model
    if guard_source == "additional":
        projector.oracle = SimpleNamespace(check_contacts=lambda _data: _safe_check())
    profile = _physical_test_profile()
    sender = MujocoActuatedPositionExecutor(scene, _timing(), profile)
    if guard_source == "additional":
        sender.set_additional_contact_oracle(oracle)
    initial = {side: np.zeros(6) for side in ("left", "right")}
    sender.reset(_command(0, initial))
    while not sender.command_due:
        if post_contact_is_safe:
            tick = sender.step_physics()
            assert tick.safety_status == "held"
        else:
            with pytest.raises(MujocoPhysicalContactFault, match="forbidden hard contact") as fault:
                sender.step_physics()
            assert fault.value.check.body2_name == "obstacle"
            assert scene.data.physics_steps == 1
            assert sender.hard_contact_latched
            assert sender.last_active_collision_check.body2_name == "obstacle"
            with pytest.raises(RuntimeError, match="latched"):
                sender.step_physics()
            assert scene.data.physics_steps == 1
            return
    target = {side: np.full(6, 0.2) for side in ("left", "right")}
    tick = sender.step_physics(_command(1, target))
    assert tick.safety_status == "projected"
    assert tick.target_modified is True
    assert tick.projection_alpha == pytest.approx(0.5)
    assert np.allclose(sender.accepted_arm_qpos("left"), 0.1)
    assert sender.snapshot()["collision_safety_state"]["hard_contact_latched"] is False


def test_physical_executor_preserves_upstream_admitted_qcmd_but_keeps_contact_guard() -> None:
    """A staged task projection must not be applied a second time downstream."""

    class _Oracle:
        def check_contacts(self, _data):
            return _safe_check()

    project_calls = 0
    check_qpos_calls = 0

    def _unexpected_project(*_args, **_kwargs):
        nonlocal project_calls
        project_calls += 1
        raise AssertionError("upstream-admitted qcmd was projected twice")

    def _check_qpos(_data, _qpos, **_kwargs):
        nonlocal check_qpos_calls
        check_qpos_calls += 1
        return _safe_check()

    projector = SimpleNamespace(
        controlled_qpos_indices=np.arange(12, dtype=np.int32),
        policy=SimpleNamespace(hard_penetration_m=0.001),
        oracle=_Oracle(),
        check_qpos=_check_qpos,
        project=_unexpected_project,
        recover_from_clearance_violation=_unexpected_project,
    )
    scene = _PhysicalFakeScene(projector)
    sender = MujocoActuatedPositionExecutor(
        scene,
        _timing(),
        _physical_test_profile(),
        collision_projection_mode=COLLISION_PROJECTION_MODE_UPSTREAM_ADMITTED,
    )
    initial = {side: np.zeros(6) for side in ("left", "right")}
    sender.reset(_command(0, initial))
    while not sender.command_due:
        tick = sender.step_physics()
        assert tick.safety_status == "held"
    target = {side: np.full(6, 0.2) for side in ("left", "right")}
    tick = sender.step_physics(_command(1, target))

    assert project_calls == 0
    # Reset validates the initial state once; the due 120 Hz command is then
    # checked against the hard bound without being projected a second time.
    assert check_qpos_calls == 2
    assert tick.safety_status == COLLISION_PROJECTION_MODE_UPSTREAM_ADMITTED
    assert tick.target_modified is False
    assert np.allclose(sender.accepted_arm_qpos("left"), target["left"])
    assert (
        sender.execution_metadata["collision_projection_mode"]
        == COLLISION_PROJECTION_MODE_UPSTREAM_ADMITTED
    )


def test_physical_executor_rejects_unsafe_upstream_qcmd_before_physics() -> None:
    class _Oracle:
        def check_contacts(self, _data):
            return _safe_check()

    check_qpos_calls = 0

    def _check_qpos(_data, _qpos, **_kwargs):
        nonlocal check_qpos_calls
        check_qpos_calls += 1
        if check_qpos_calls == 1:
            return _safe_check()
        return CollisionCheckResult(
            safe=False,
            minimum_distance_m=-0.002,
            body1_name="arm",
            body2_name="obstacle",
            pair_class="1:2",
        )

    projector = SimpleNamespace(
        controlled_qpos_indices=np.arange(12, dtype=np.int32),
        policy=SimpleNamespace(hard_penetration_m=0.001),
        oracle=_Oracle(),
        check_qpos=_check_qpos,
    )
    scene = _PhysicalFakeScene(projector)
    sender = MujocoActuatedPositionExecutor(
        scene,
        _timing(),
        _physical_test_profile(),
        collision_projection_mode=COLLISION_PROJECTION_MODE_UPSTREAM_ADMITTED,
    )
    initial = {side: np.zeros(6) for side in ("left", "right")}
    sender.reset(_command(0, initial))
    while not sender.command_due:
        sender.step_physics()
    physics_steps_before = scene.data.physics_steps
    target = {side: np.full(6, 0.2) for side in ("left", "right")}

    with pytest.raises(RuntimeError, match="Upstream-admitted.*hard collision"):
        sender.step_physics(_command(1, target))

    assert scene.data.physics_steps == physics_steps_before
    assert sender.snapshot()["collision_safety_state"]["status"] == (
        "unsafe_upstream_admitted"
    )
def test_physical_executor_snapshot_restore_binds_actuation_profile() -> None:
    scene = _PhysicalFakeScene(projector=None)
    profile = _physical_test_profile()
    sender = MujocoActuatedPositionExecutor(scene, _timing(), profile)
    initial = {side: np.zeros(6) for side in ("left", "right")}
    sender.reset(_command(0, initial))
    snapshot = sender.snapshot()
    snapshot["profile"] = {"name": "different-profile"}

    with pytest.raises(ValueError, match="profile.*does not match"):
        sender.reset(
            _command(0, initial),
            initial_state=snapshot,
        )
