from __future__ import annotations

from concurrent.futures import Future
from copy import deepcopy
from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest

from robot_motion.position_control.horizon_governor import JointHorizonLimits
from robot_motion.position_control.offline_waypoint_trajectory import (
    build_offline_quintic_waypoint_trajectory,
)
from scripts.probes.validate_grasp_pipeline import (
    KinematicSink,
    OfflinePlanAdmission,
    ResumeRejected,
    SettledBoundary,
    prepare_resume,
    prepend_connector,
    rebuild_connector,
    simulate_case,
    split_pregrasp,
)
from teleop_runtime.skill_deploy.motion_port import (
    RealAgentJointTrajectory,
    RealAgentMotionMeasuredQposCut,
    RealAgentMotionPhase,
)


@pytest.mark.parametrize("body_failure,guard_failure,store_failure,expected_phase", (
    (None, "runtime", None, "guard"),
    (None, "runtime", "runtime", "guard"),
    ("runtime", "runtime", "runtime", "body"),
    ("keyboard", "runtime", "runtime", "body"),
    ("system_exit", "runtime", "runtime", "body"),
    ("runtime", "keyboard", "runtime", "guard"),
    ("runtime", "system_exit", "runtime", "guard"),
    ("runtime", "runtime", "keyboard", "store"),
    ("runtime", "runtime", "system_exit", "store"),
    ("keyboard", "system_exit", "runtime", "body"),
    ("system_exit", "keyboard", "runtime", "body"),
    (None, "keyboard", "system_exit", "guard"),
    (None, None, "runtime", "store"),
    (None, None, "keyboard", "store"),
    (None, None, "system_exit", "store"),
))
def test_main_attempts_independent_cleanup_and_preserves_body_failure(
    monkeypatch, tmp_path, body_failure, guard_failure, store_failure, expected_phase,
):
    import sys
    from scripts.probes import validate_grasp_pipeline as probe
    from sim.contracts import SimulationContract
    from sim.mujoco import actuation
    from sim.mujoco.scene import MujocoScene
    from teleop_runtime import position_force_guard

    events = []
    def make_error(kind, phase):
        error = {None: None, "runtime": RuntimeError(f"{phase} failed"),
            "keyboard": KeyboardInterrupt(f"{phase} interrupted"), "system_exit": SystemExit(7)}[kind]
        if error is not None:
            error.__cause__ = ValueError(f"{phase} original cause")
        return error

    errors = {"body": make_error(body_failure, "body"),
        "guard": make_error(guard_failure, "guard"),
        "store": make_error(store_failure, "store")}
    primary, guard_error, store_error = (errors[phase] for phase in ("body", "guard", "store"))
    expected = errors[expected_phase]
    original_cause = expected.__cause__
    prefix = SimpleNamespace(qpos_by_side={s: np.zeros((2, 6)) for s in ("left", "right")},
        time_from_start_sec=np.array([0., 1.]))
    planner = SimpleNamespace(target_builder=object(),
        _revalidate=lambda *a, **kw: None,
        _revalidate_visible_gripper_path=lambda *a, **kw: None)

    class Pool:
        def __enter__(self):
            events.append("pool_enter")
            return self

        def __exit__(self, kind, error, traceback):
            events.append("pool_exit")

        def submit(self, *args):
            return Future()

    class Guard:
        def __init__(self, **kwargs):
            pass

        def start(self):
            events.append("guard_start")

        def snapshot(self):
            return {}

        def close(self):
            events.append("guard_close")
            if guard_error is not None:
                raise guard_error

    def close_store():
        events.append("store_close")
        if store_error is not None:
            raise store_error

    def simulate(*args, **kwargs):
        events.append("body")
        if primary is not None:
            raise primary
        return {}

    output = tmp_path / "new-output"
    argv = ["validate_grasp_pipeline"]
    for name in ("repo", "canonical", "trajectory", "visible-scene", "geometry", "mujoco-contract", "output"):
        argv.extend(("--" + name, str(output if name == "output" else tmp_path / name)))
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(probe, "load_grasp_trajectory_record", lambda *a: (prefix, {"moving_side": "right"}))
    monkeypatch.setattr(probe, "split_pregrasp", lambda *a: (prefix, prefix))
    monkeypatch.setattr(probe, "_planner", lambda *a: planner)
    monkeypatch.setattr(probe, "_visible_scene", lambda *a: object())
    monkeypatch.setattr(SimulationContract, "load", staticmethod(lambda *a: object()))
    monkeypatch.setattr(MujocoScene, "from_contract", staticmethod(lambda *a, **kw: object()))
    monkeypatch.setattr(actuation, "cad_bimanual_dmtac_gripper_profile", lambda: object())
    monkeypatch.setattr(probe, "ProcessPoolExecutor", lambda **kw: Pool())
    monkeypatch.setattr(probe, "KinematicSink", lambda *a: SimpleNamespace(guard=None))
    monkeypatch.setattr(probe, "OfflinePlanAdmission", lambda *a: SimpleNamespace(store=SimpleNamespace(close=close_store)))
    monkeypatch.setattr(position_force_guard, "PositionForceGuardMonitor", Guard)
    monkeypatch.setattr(probe, "simulate_case", simulate)
    with pytest.raises(type(expected)) as caught:
        probe.main()
    assert caught.value is expected
    assert caught.value.__cause__ is original_cause
    if isinstance(expected, SystemExit):
        assert caught.value.code == 7
    assert events == ["pool_enter", "guard_start", "body", "guard_close", "store_close", "pool_exit"]
    notes = "\n".join(getattr(caught.value, "__notes__", ()))
    for phase, error in errors.items():
        if error is not None and error is not expected:
            assert str(error) in notes
            assert f"{phase} original cause" in notes


@pytest.mark.parametrize("output_failure", (OSError("stderr unavailable"), ValueError("stderr closed")))
@pytest.mark.parametrize("primary", (RuntimeError("body failed"), KeyboardInterrupt(), SystemExit(7)))
def test_cli_note_output_failure_preserves_original_error(monkeypatch, output_failure, primary):
    import argparse
    import runpy
    import sys
    from scripts.probes import validate_grasp_pipeline as probe

    original_cause = ValueError("original cause")
    primary.__cause__ = original_cause
    primary.__notes__ = ["secondary cleanup traceback"]

    def fail_parse(*args, **kwargs):
        raise primary

    class BrokenStderr:
        def write(self, text):
            raise output_failure

    with monkeypatch.context() as patch:
        patch.setattr(argparse.ArgumentParser, "parse_args", fail_parse)
        patch.setattr(sys, "version_info", (3, 10, 0))
        patch.setattr(sys, "stderr", BrokenStderr())
        with pytest.raises(type(primary)) as caught:
            runpy.run_path(probe.__file__, run_name="__main__")
    assert caught.value is primary
    assert primary.__cause__ is original_cause
    assert primary.__notes__[0] == "secondary cleanup traceback"
    assert any(str(output_failure) in note for note in primary.__notes__[1:])
    if isinstance(primary, SystemExit):
        assert primary.code == 7


class Builder:
    def solver_world_target_from_tool_tcp_world_target(self, *, side, tool_tcp_world_target):
        return tool_tcp_world_target.copy()

    def tool_tcp_world_targets_for_qpos_pair(self, *, left_qpos, right_qpos):
        poses = {s: np.eye(4) for s in ("left", "right")}
        poses["left"][0, 3] = left_qpos[0]
        poses["right"][0, 3] = right_qpos[0]
        return poses


def fixture():
    qpos = np.zeros((5, 6))
    qpos[:, 0] = np.linspace(0, .004, 5)
    trajectory = RealAgentJointTrajectory(np.linspace(0, .2, 5),
        {"left": np.zeros_like(qpos), "right": qpos}, ("urdf-fcl-path:offline-unit-case",))
    metadata = {"stage_boundaries": [
        {"planning_phase": "pregrasp_target", "end_sample_index": 2, "end_time_sec": .1},
        {"planning_phase": "constrained_approach", "end_sample_index": 4, "end_time_sec": .2}],
        "parameterization": {"stage_boundary_stops": True, "legs": [
            {"continuous_hard_limit_certificate": {"passed": True}},
            {"continuous_hard_limit_certificate": {"passed": True}}]}}
    return trajectory, metadata


def test_split_preserves_the_exact_joint_boundary_and_resets_only_suffix_time():
    trajectory, metadata = fixture()
    prefix, suffix = split_pregrasp(trajectory, metadata)
    for side in ("left", "right"):
        np.testing.assert_array_equal(prefix.qpos_by_side[side][-1], suffix.qpos_by_side[side][0])
        np.testing.assert_array_equal(np.concatenate((prefix.qpos_by_side[side],
            suffix.qpos_by_side[side][1:])), trajectory.qpos_by_side[side])
    assert suffix.time_from_start_sec[0] == 0.
    assert prefix.time_from_start_sec[-1] + suffix.time_from_start_sec[-1] == .2


@pytest.mark.parametrize("mutation", ("no_stop", "bad_time", "bad_dynamics", "wrong_stage", "endpoint"))
def test_split_rejects_unverified_or_nonzero_speed_handoff(mutation):
    trajectory, original = fixture()
    metadata = deepcopy(original)
    if mutation == "no_stop":
        metadata["parameterization"]["stage_boundary_stops"] = False
    elif mutation == "bad_time":
        metadata["stage_boundaries"][0]["end_time_sec"] = .15
    elif mutation == "bad_dynamics":
        metadata["parameterization"]["legs"][1]["continuous_hard_limit_certificate"]["passed"] = False
    elif mutation == "wrong_stage":
        metadata["stage_boundaries"][1]["planning_phase"] = "free_space"
    else:
        metadata["stage_boundaries"][0]["end_sample_index"] = 4
    with pytest.raises(ValueError):
        split_pregrasp(trajectory, metadata)


@pytest.mark.parametrize("scenario,outcome", (
    ("ready", "completed"), ("late", "completed"), ("timeout", "timeout"),
    ("cancel", "cancel"), ("worker_failure", "worker_failure"),
    ("state_drift", "start_state_rejected"), ("scene_changed", "scene_changed"),
    ("small_state_drift", "completed"),
    ("clock_gap", "completed"),
))
def test_staged_port_waits_and_resumes_from_a_fresh_clock(scenario, outcome):
    prefix, suffix = split_pregrasp(*fixture())
    sink = KinematicSink({s: q[0] for s, q in prefix.qpos_by_side.items()})
    def supplier():
        if scenario == "worker_failure":
            raise RuntimeError("Injected planner failure")
        return suffix
    report = simulate_case(prefix, supplier, builder=Builder(), sink=sink,
        side="right", scenario=scenario)
    assert report["outcome"] == outcome
    assert report["wait_max_qpos_change_rad"] == 0.
    waiting = report["qpos"][report["stages"] == "waiting"]
    np.testing.assert_array_equal(waiting, np.broadcast_to(waiting[0], waiting.shape))
    assert np.max(np.abs(report["qpos"][:, :6])) == 0.
    if outcome == "completed":
        assert report["suffix_activated"]
        assert report["first_suffix_elapsed_sec"] == pytest.approx(1. / 120.)
        assert report["first_suffix_elapsed_sec"] == pytest.approx(report["first_suffix_wall_advance_sec"])
        np.testing.assert_array_equal(sink.read()["right"], suffix.qpos_by_side["right"][-1])
        if scenario == "late":
            assert report["wait_duration_sec"] >= 1.
        elif scenario == "clock_gap":
            assert report["gap_resume_clock_advance_sec"] == pytest.approx(1. / 120.)
        elif scenario == "small_state_drift":
            # Admission tolerance is not a certificate of smooth splicing.
            assert report["first_suffix_qpos_step_rad"] > .004
    else:
        assert not report["suffix_activated"]
        assert report["first_suffix_elapsed_sec"] is None
        if scenario == "worker_failure":
            assert report["planning_error"]["type"] == "RuntimeError"
        if scenario == "state_drift":
            assert "start changed before activation" in report["port_hold_reason"]


def test_measured_state_c2_connector_prevents_the_unadjusted_reference_jump():
    prefix, suffix = split_pregrasp(*fixture())
    def correct(measured, frozen):
        q = np.array([np.concatenate((measured["left"], measured["right"])),
            np.concatenate((frozen.qpos_by_side["left"][0], frozen.qpos_by_side["right"][0]))])
        limits = JointHorizonLimits(min_position_rad=np.full(12, -1.),
            max_position_rad=np.ones(12), max_velocity_rad_s=np.ones(12),
            max_acceleration_rad_s2=np.full(12, 10.), max_jerk_rad_s3=np.full(12, 100.))
        part = build_offline_quintic_waypoint_trajectory([0., .5], q, command_hz=60., limits=limits)
        assert part.metadata["continuous_hard_limit_certificate"]["passed"]
        np.testing.assert_array_equal(part.qd_rad_s[[0, -1]], 0.)
        np.testing.assert_array_equal(part.qdd_rad_s2[[0, -1]], 0.)
        connector = RealAgentJointTrajectory(part.timestamps,
            {"left": part.q_rad[:, :6], "right": part.q_rad[:, 6:]}, frozen.planner_evidence_refs)
        return prepend_connector(connector, frozen)
    sink = KinematicSink({s: q[0] for s, q in prefix.qpos_by_side.items()})
    report = simulate_case(prefix, lambda: suffix, builder=Builder(), sink=sink,
        side="right", scenario="small_drift_corrected", correction=correct)
    assert report["outcome"] == "completed"
    assert report["first_suffix_qpos_step_rad"] < 1e-6
    np.testing.assert_array_equal(sink.read()["right"], suffix.qpos_by_side["right"][-1])


def test_connector_cannot_replace_the_frozen_ik_boundary():
    prefix, suffix = split_pregrasp(*fixture())
    with pytest.raises(ValueError, match="endpoint differs"):
        prepend_connector(suffix, prefix)


@pytest.mark.parametrize("side,drift,reason", (
    ("left", .001, "unauthorized arm"), ("right", .02, "unchanged start guard"),
))
def test_connector_preserves_existing_start_and_inactive_arm_guards(monkeypatch, side, drift, reason):
    from types import SimpleNamespace

    from scripts.probes import validate_grasp_pipeline as probe

    trajectory, metadata = fixture()
    metadata["moving_side"] = "right"
    monkeypatch.setattr(probe, "load_grasp_trajectory_record", lambda _: (trajectory, metadata))
    prefix, _ = split_pregrasp(trajectory, metadata)
    measured = {s: q[-1].copy() for s, q in prefix.qpos_by_side.items()}
    measured[side][0] += drift
    with pytest.raises(ValueError, match=reason):
        rebuild_connector(SimpleNamespace(trajectory="test-only"), measured)


def measured_cut(qpos, timestamp, revision=0, *, available=None):
    return RealAgentMotionMeasuredQposCut(revision=revision, qpos_by_side=qpos,
        measured_at=timestamp, available_at=timestamp if available is None else available,
        source_evidence_refs=("offline-measured-unit-case",))


def test_settling_requires_new_measured_cuts_and_dwell_not_just_a_hold_receipt():
    expected = {s: np.zeros(6) for s in ("left", "right")}
    observer = SettledBoundary(expected)
    assert not observer.observe(measured_cut(expected, 1.), 1.)
    assert not observer.observe(measured_cut(expected, 1.02), 1.02)
    assert not observer.observe(measured_cut(expected, 1.08), 1.08)
    assert observer.observe(measured_cut(expected, 1.101), 1.101)
    with pytest.raises(ResumeRejected, match="not_advancing"):
        observer.observe(measured_cut(expected, 1.101), 1.2)


@pytest.mark.parametrize("age", (-.001, .201))
def test_settling_rejects_future_and_stale_measurements(age):
    q = {s: np.zeros(6) for s in ("left", "right")}
    with pytest.raises(ResumeRejected, match="stale_or_future"):
        SettledBoundary(q).observe(measured_cut(q, 1.), 1. + age)


@pytest.mark.parametrize("side", ("left", "right"))
def test_settling_resets_dwell_for_movement_of_either_arm(side):
    q = {s: np.zeros(6) for s in ("left", "right")}
    observer = SettledBoundary(q)
    observer.observe(measured_cut(q, 1.), 1.)
    observer.observe(measured_cut(q, 1.02), 1.02)
    changed = {s: a.copy() for s, a in q.items()}
    changed[side][0] = .001  # Inside start tolerance, but 0.05 rad/s is not settled.
    assert not observer.observe(measured_cut(changed, 1.04), 1.04)
    assert not observer.observe(measured_cut(changed, 1.06), 1.06)
    assert not observer.observe(measured_cut(changed, 1.1), 1.1)
    assert observer.observe(measured_cut(changed, 1.141), 1.141)


def resume_fixture(*, change_during=False, change_after=False, never_ready=False):
    prefix, suffix = split_pregrasp(*fixture())
    q = {s: a[-1].copy() for s, a in prefix.qpos_by_side.items()}
    now, ticks, submitted = 1., 0, []
    future = Future()

    def pulse(stage):
        nonlocal now, ticks
        now += .02
        ticks += 1
        changed = {s: a.copy() for s, a in q.items()}
        if (stage == "connector_wait" and change_during) or (stage == "resume_recheck" and change_after):
            changed["right"][0] += .0001  # Still inside original admission guard.
        if stage == "connector_wait" and not never_ready:
            future.set_result({"timestamps": np.array([0., .5]),
                "q": np.tile(np.concatenate((q["left"], q["right"])), (2, 1)),
                "worker_pid": 123, "context_reused": True, "elapsed_sec": .02})
        return measured_cut(changed, now, ticks), now

    def submit(measured):
        submitted.append(measured)
        return future

    port = SimpleNamespace(snapshot=lambda: SimpleNamespace(phase=RealAgentMotionPhase.HELD,
        hold_succeeded=True, operation_id="prefix"))
    return prefix, suffix, port, pulse, lambda: now, submit, submitted, future


def test_resume_waits_for_worker_with_new_cuts_and_rechecks_after_return():
    prefix, suffix, port, pulse, clock, submit, submitted, _ = resume_fixture()
    joined, current, receipt = prepare_resume(prefix, suffix, port=port, pulse=pulse,
        clock=clock, submit_connector=submit)
    assert len(submitted) == 1
    assert current.measured_at == clock()
    assert receipt["settling_sec"] >= .08
    assert receipt["post_worker_recheck_sec"] >= .08
    assert receipt["worker_context_reused"]
    assert joined.time_from_start_sec[-1] == pytest.approx(.5 + suffix.time_from_start_sec[-1])
    for side in ("left", "right"):
        np.testing.assert_array_equal(joined.qpos_by_side[side][0], submitted[0][side])


def test_resume_wiring_uses_port_joint_cuts_not_derived_tcp_pose_cuts():
    prefix, suffix = split_pregrasp(*fixture())
    measured_revisions = []

    def resume(prefix, suffix, port, pulse, clock):
        def checked_pulse(stage):
            cut, now = pulse(stage)
            assert isinstance(cut, RealAgentMotionMeasuredQposCut)
            assert cut is port.latest_measured_qpos()
            measured_revisions.append(cut.revision)
            return cut, now

        def submit(q):
            # This scheduling-only fake is not a collision/dynamics certificate.
            future = Future()
            future.set_result({"timestamps": np.array([0., .5]), "q": np.array([
                np.concatenate((q["left"], q["right"])),
                np.concatenate((suffix.qpos_by_side["left"][0], suffix.qpos_by_side["right"][0]))]),
                "worker_pid": 123, "context_reused": True, "elapsed_sec": 0.})
            return future

        joined, _, receipt = prepare_resume(prefix, suffix, port=port, pulse=checked_pulse,
            clock=clock, submit_connector=submit)
        return joined, receipt

    sink = KinematicSink({s: q[0] for s, q in prefix.qpos_by_side.items()})
    report = simulate_case(prefix, lambda: suffix, builder=Builder(), sink=sink,
        side="right", scenario="guarded_resume", resume=resume)
    assert report["outcome"] == "completed"
    assert len(measured_revisions) >= 20
    assert all(b > a for a, b in pairwise(measured_revisions))


@pytest.mark.parametrize("kwargs,reason", (
    ({"change_during": True}, "changed_during_connector"),
    ({"change_after": True}, "changed_after_connector"),
    ({"never_ready": True}, "timeout"),
))
def test_no_resume_on_new_drift_or_connector_timeout(kwargs, reason):
    prefix, suffix, port, pulse, clock, submit, *_ = resume_fixture(**kwargs)
    with pytest.raises(ResumeRejected, match=reason):
        prepare_resume(prefix, suffix, port=port, pulse=pulse, clock=clock,
            submit_connector=submit, budget=.5)


def test_resume_cannot_treat_unconfirmed_hold_as_settled():
    prefix, suffix, _, pulse, clock, submit, submitted, _ = resume_fixture()
    port = SimpleNamespace(snapshot=lambda: SimpleNamespace(phase=RealAgentMotionPhase.HELD, hold_succeeded=False))
    with pytest.raises(ResumeRejected, match="hold_unconfirmed"):
        prepare_resume(prefix, suffix, port=port, pulse=pulse, clock=clock, submit_connector=submit)
    assert not submitted


def test_resume_dwell_cannot_cross_the_total_deadline():
    prefix, suffix, port, pulse, clock, submit, submitted, _ = resume_fixture()
    with pytest.raises(ResumeRejected, match="timeout"):
        prepare_resume(prefix, suffix, port=port, pulse=pulse, clock=clock, submit_connector=submit, budget=.08)
    assert not submitted


def test_resume_rejects_a_changed_lower_hold_during_wait():
    prefix, suffix, port, pulse, clock, submit, submitted, _ = resume_fixture()
    initial = port.snapshot()

    def invalidate(stage):
        result = pulse(stage)
        port.snapshot = lambda: SimpleNamespace(phase=RealAgentMotionPhase.HELD,
            hold_succeeded=True, operation_id="different-hold")
        return result

    assert initial.operation_id == "prefix"
    with pytest.raises(ResumeRejected, match="hold_changed"):
        prepare_resume(prefix, suffix, port=port, pulse=invalidate, clock=clock, submit_connector=submit)
    assert not submitted


def test_new_stage_needs_new_approval_and_native_store_rejects_expiry_and_replay():
    from agent_gateway.profiles.real.grasp_approach_plan import (
        RealGraspApproachPlanLookupError,
    )

    prefix, suffix = split_pregrasp(*fixture())
    cut = measured_cut({s: a[0] for s, a in suffix.qpos_by_side.items()}, 1.)
    admission = OfflinePlanAdmission("right")
    try:
        first = admission.retain(prefix, cut, np.eye(4), "prefix", 1.)
        second = admission.retain(suffix, cut, np.eye(4), "suffix", 2.)
        assert first.plan_id != second.plan_id
        with pytest.raises(ResumeRejected, match="not_authorized"):
            admission.activation(second, cut, approved_plan_id=first.plan_id, name="suffix", now=2.)
        with pytest.raises(RealGraspApproachPlanLookupError, match="station_changed"):
            admission.activation(second, cut, approved_plan_id=second.plan_id, name="suffix", now=2., station_revision=1)
        activation = admission.activation(second, cut, approved_plan_id=second.plan_id, name="suffix", now=2.)
        assert activation.expires_at == second.expires_at
        assert activation.joint_trajectory is suffix
        with pytest.raises(RealGraspApproachPlanLookupError, match="already_consumed"):
            admission.activation(second, cut, approved_plan_id=second.plan_id, name="suffix", now=2.1)
        expired = admission.retain(suffix, cut, np.eye(4), "expired", 3.)
        with pytest.raises(RealGraspApproachPlanLookupError, match="not_found|expired"):
            admission.activation(expired, cut, approved_plan_id=expired.plan_id, name="expired", now=expired.expires_at)
    finally:
        admission.store.close()


def test_force_fault_cannot_be_cleared_by_resume_or_reactivation():
    import threading
    import time

    from teleop_runtime.position_force_guard import PositionForceGuardMonitor

    trips = []
    blocked = threading.Event()
    release = threading.Event()

    def read():
        if blocked.is_set():
            release.wait(1.)
        return {"sources": {"tool": [0.] * 6}, "monotonic_time_sec": time.monotonic(),
            "wall_time_sec": time.time()}

    guard = PositionForceGuardMonitor(readers_by_side={"left": read, "right": read}, emergency_stop=trips.append)
    try:
        guard.start()
        prefix, suffix, port, pulse, clock, _, submitted, future = resume_fixture(never_ready=True)

        def paced_pulse(stage):
            time.sleep(.02)
            return pulse(stage)

        def submit(measured):
            submitted.append(measured)
            blocked.set()  # The real watchdog trips while a planner reply is pending.
            return future

        def check():
            if not guard.motion_ready:
                raise ResumeRejected("force_guard_not_ready")

        with pytest.raises(ResumeRejected, match="force_guard_not_ready"):
            prepare_resume(prefix, suffix, port=port, pulse=paced_pulse, clock=clock, submit_connector=submit,
                check_valid=check)
        assert guard.fault is not None
        assert guard.fault.reason == "force_read_stale"
        assert len(trips) == 1
        assert guard.stale_watchdog_armed
        assert not guard.snapshot()["non_motion_read_window_active"]
        assert len(submitted) == 1
        assert not future.done()
        sink = KinematicSink({s: a[0] for s, a in prefix.qpos_by_side.items()}, guard=guard)
        sink.ready = True
        assert sink.skill_motion_owner_state() == ("fault", "fault")
    finally:
        release.set()
        guard.close()
