"""Offline frozen-branch pipeline probe; never creates a hardware backend.

This tests staging of an already selected, validated local joint path, not new
grasp/IK search or a production streaming Skill. MuJoCo follows qpos commands
kinematically; actuator tracking, braking and physical grasp remain untested.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from robot_motion.grasp_visible_scene import ParallelJawSolidGeometry, VisibleGraspScene
from robot_motion.joint_limits import load_joint_velocity_limit_profiles
from robot_motion.pose import pose_delta_metrics
from robot_motion.runtime import TeleopCommand, TeleopObservation
from scripts.probes.replay_bimanual_grasp_snapshot import profile_for
from teleop_runtime.skill_deploy.grasp_trajectory_record import (
    load_grasp_trajectory_record,
)
from teleop_runtime.skill_deploy.motion_port import (
    REAL_AGENT_TRAJECTORY_START_MAX_DRIFT_RAD,
    RealAgentJointTrajectory,
    RealAgentMotionActivation,
    RealAgentMotionPhase,
    RealAgentMotionPort,
    RealAgentMotionPortConfig,
)
from teleop_runtime.skill_deploy.rm65_grasp_trajectory import (
    RM65GraspJointTrajectoryPlanner,
    RM65GraspTrajectoryPlanningConfig,
)
from teleop_runtime.skill_deploy.rm65_lower_assembly import (
    bind_exclusive_rm65_skill_target_safety,
    build_exclusive_rm65_skill_target_builder,
)

# A worker owns its native models; no planner/filter or motion port crosses
# the process boundary. Reuse this context for suffix and correction jobs.
_WORKER_CONTEXT = None


def _worker_context(args, metadata):
    global _WORKER_CONTEXT
    if multiprocessing.current_process().name == "MainProcess":
        return _planner(args, metadata), _visible_scene(args), False
    key = (args.repo, args.canonical, args.trajectory, args.visible_scene, args.geometry)
    reused = _WORKER_CONTEXT is not None
    if reused and _WORKER_CONTEXT[0] != key:
        raise ValueError("Worker cannot switch frozen scene/model inputs")
    if not reused:
        _WORKER_CONTEXT = (key, _planner(args, metadata), _visible_scene(args))
    return *_WORKER_CONTEXT[1:], reused


class ResumeRejected(RuntimeError):
    """A diagnostic handoff failed; stay HELD without admitting a suffix."""


class SettledBoundary:
    """Probe-only measured dwell, matching registered replay's defaults.

    These are not new production settings: replay already checks 0.01 rad
    tracking, 0.02 rad/s measured velocity and 0.08 s consecutive dwell. A
    HOLD receipt or zero reference derivatives alone is not measured settling.
    """
    def __init__(self, expected):
        self.expected = expected
        self.previous = None
        self.stable_since = None

    def observe(self, cut, now):
        if not 0. <= now - cut.measured_at <= .2:
            raise ResumeRejected("settling_measurement_stale_or_future")
        if not cut.measured_at <= cut.available_at <= now:
            raise ResumeRejected("settling_measurement_time_invalid")
        if self.previous is None:
            self.previous = cut
            return False
        dt = cut.measured_at - self.previous.measured_at
        if dt <= 0.:
            raise ResumeRejected("settling_measurement_not_advancing")
        velocity = max(float(np.max(np.abs(cut.qpos_by_side[s] - self.previous.qpos_by_side[s])))
            / dt for s in ("left", "right"))
        tracking = max(float(np.max(np.abs(cut.qpos_by_side[s] - self.expected[s])))
            for s in ("left", "right"))
        self.previous = cut
        if dt > .2 or velocity > .02 or tracking > REAL_AGENT_TRAJECTORY_START_MAX_DRIFT_RAD:
            self.stable_since = None
            return False
        if self.stable_since is None:
            self.stable_since = cut.measured_at
        return cut.measured_at - self.stable_since >= .08


def prepare_resume(prefix, suffix, *, port, pulse, clock, submit_connector,
                   check_valid=lambda: None, event=lambda *_: None, budget=2.):
    """Offline manager: confirmed HOLD -> settled cut -> worker -> recheck.

    The parent keeps sampling while the isolated worker runs. A changed start
    is rejected (even below admission tolerance) instead of treating that
    tolerance as a smooth-splice certificate. This exact frozen-state check
    is diagnostic, not a policy for noisy physical encoders.
    """
    if not np.isfinite(budget) or budget <= 0.:
        raise ValueError("resume budget must be finite and positive")
    state = port.snapshot()
    if state.phase != RealAgentMotionPhase.HELD or not state.hold_succeeded:
        raise ResumeRejected("boundary_hold_unconfirmed")
    started = clock()
    expected = {s: q[-1] for s, q in prefix.qpos_by_side.items()}

    def check():
        current = port.snapshot()
        if (current.phase != RealAgentMotionPhase.HELD or not current.hold_succeeded
                or current.operation_id != state.operation_id):
            raise ResumeRejected("boundary_hold_changed")
        check_valid()

    def settled(stage, target):
        observer = SettledBoundary(target)
        while clock() - started < budget:
            cut, now = pulse(stage)
            check()
            if now - started >= budget:
                raise ResumeRejected("settling_or_connector_timeout")
            if observer.observe(cut, now):
                return cut
        raise ResumeRejected("settling_or_connector_timeout")

    captured = settled("settling", expected)
    submitted_at = clock()
    future = submit_connector({s: q.copy() for s, q in captured.qpos_by_side.items()})
    event("connector_submitted", {"measured_revision": captured.revision, "now": submitted_at})
    while not future.done():
        cut, now = pulse("connector_wait")
        check()
        if now - started >= budget:
            future.cancel()
            raise ResumeRejected("settling_or_connector_timeout")
        if any(not np.array_equal(cut.qpos_by_side[s], captured.qpos_by_side[s]) for s in expected):
            future.cancel()
            raise ResumeRejected("measured_state_changed_during_connector")
        if not 0. <= now - cut.measured_at <= .2:
            future.cancel()
            raise ResumeRejected("connector_wait_measurement_stale_or_future")
    check()
    if clock() - started >= budget:
        raise ResumeRejected("settling_or_connector_timeout")
    try:
        result = future.result()
    except (RuntimeError, ValueError) as error:
        raise ResumeRejected(f"connector_failed:{type(error).__name__}:{error}") from error
    completed_at = clock()
    connector = RealAgentJointTrajectory(result["timestamps"],
        {"left": result["q"][:, :6], "right": result["q"][:, 6:]}, suffix.planner_evidence_refs)
    if any(not np.array_equal(connector.qpos_by_side[s][0], captured.qpos_by_side[s]) for s in expected):
        raise ResumeRejected("connector_does_not_start_at_submitted_measurement")
    current = settled("resume_recheck", captured.qpos_by_side)
    if any(not np.array_equal(current.qpos_by_side[s], captured.qpos_by_side[s]) for s in expected):
        raise ResumeRejected("measured_state_changed_after_connector")
    check()
    event("resume_rechecked", {"now": clock(), "measured_revision": current.revision})
    return prepend_connector(connector, suffix), current, {
        "settling_sec": submitted_at - started,
        "connector_ready_sec": completed_at - submitted_at,
        "post_worker_recheck_sec": clock() - completed_at,
        "worker_pid": result["worker_pid"], "worker_context_reused": result["context_reused"],
        "native_connector_work_sec": result["elapsed_sec"],
        "connector_duration_sec": float(connector.time_from_start_sec[-1]),
        "connector": {k: v for k, v in result.items() if k not in {"timestamps", "q"}},
    }


class OfflinePlanAdmission:
    """Reuse the real immutable store, but NEVER issue a live Skill grant.

    approved_plan_id is an explicit synthetic operator fixture. The authenticated
    Gateway Route, grant issuer and wire authorization are not exercised here.
    """
    def __init__(self, side):
        from agent_gateway.profiles.real.grasp_approach_plan import (
            BoundedRealGraspApproachPlanStore,
        )
        self.store = BoundedRealGraspApproachPlanStore(max_retained_plans=4)
        self.side = side

    def retain(self, trajectory, cut, target, name, now):
        from agent_gateway.profiles.real.grasp_approach_plan import (
            RealGraspApproachPlan,
        )
        plan = RealGraspApproachPlan(runtime_session_id="offline-session",
            profile_hash="offline-profile", planning_operation_id=f"offline-planning:{name}",
            planning_request_hash="offline-request", source_operation_id="frozen-grasp",
            source_request_hash="offline-grasp-request", source_result_hash="offline-grasp-result",
            source_grasp_hash="offline-candidate", source_station_revision=0,
            component_id=f"{self.side}-tool-tcp", side=self.side, target_pose_reference=target,
            target_pose_execution=target, speed_scale=1., measured_qpos_revision=cut.revision,
            measured_qpos_hash=cut.semantic_content_hash, measured_qpos_at=cut.measured_at,
            trajectory=trajectory, created_at=now, expires_at=now + 30.,
            evidence_refs=("offline-staged-plan:not-real-authority",))
        self.store.put(plan, now=now)
        return plan

    def activation(self, plan, cut, *, approved_plan_id, name, now, station_revision=0):
        if approved_plan_id != plan.plan_id:
            raise ResumeRejected("new_stage_not_authorized")
        self.store.reserve(plan_id=plan.plan_id, plan_hash=plan.semantic_content_hash,
            operation_id=name, runtime_session_id="offline-session", profile_hash="offline-profile",
            component_id=f"{self.side}-tool-tcp", expected_station_revision=station_revision, now=now)
        # Consumed even if lower admission subsequently rejects the measured
        # start, just as a one-use request must not become replayable on error.
        self.store.consume(plan_id=plan.plan_id, operation_id=name)
        return RealAgentMotionActivation(operation_id=name, request_id=name,
            activation_grant_id=f"offline-new-plan-grant:{plan.plan_id}", activation_grant_hash="not-real-authority",
            component_id=f"{self.side}-tool-tcp", side=self.side, primitive_id="offline.pipeline@0",
            primitive_hash="not-a-deployed-skill", target_pose_world=plan.target_pose_execution,
            execution_frame_id="offline-solver-world", execution_frame_spec_hash="offline-test-only",
            measured_state_revision=cut.revision, measured_state_hash=cut.semantic_content_hash,
            submitted_at=now, expires_at=plan.expires_at, joint_trajectory=plan.trajectory)


def split_pregrasp(trajectory, metadata):
    """Accept only an explicitly stopped, two-stage diagnostic record."""
    boundaries = metadata["stage_boundaries"]
    parameterization = metadata["parameterization"]
    if (len(boundaries) != 2 or len(parameterization["legs"]) != 2
            or not parameterization.get("stage_boundary_stops")
            or [b["planning_phase"] for b in boundaries]
            != ["pregrasp_target", "constrained_approach"]):
        raise ValueError("Pipeline probe requires stopped pregrasp + constrained approach")
    index = boundaries[0]["end_sample_index"]
    if (not isinstance(index, int) or isinstance(index, bool)
            or not 0 < index < len(trajectory.time_from_start_sec) - 1
            or trajectory.time_from_start_sec[index] != boundaries[0]["end_time_sec"]):
        raise ValueError("Invalid pregrasp boundary")
    for leg in parameterization["legs"]:
        if not leg["continuous_hard_limit_certificate"]["passed"]:
            raise ValueError("Uncertified stage dynamics")
    prefix = RealAgentJointTrajectory(
        trajectory.time_from_start_sec[:index + 1],
        {s: q[:index + 1] for s, q in trajectory.qpos_by_side.items()},
        trajectory.planner_evidence_refs,
    )
    suffix = RealAgentJointTrajectory(
        trajectory.time_from_start_sec[index:] - trajectory.time_from_start_sec[index],
        {s: q[index:] for s, q in trajectory.qpos_by_side.items()},
        trajectory.planner_evidence_refs,
    )
    return prefix, suffix


def _planner(args, metadata):
    profile = profile_for(SimpleNamespace(repo=args.repo, canonical=args.canonical,
        collision_scene_profile=metadata["collision_scene_profile"]), metadata["moving_side"])
    builder = build_exclusive_rm65_skill_target_builder(profile)
    safety = bind_exclusive_rm65_skill_target_safety(profile=profile, target_builder=builder)
    planner = RM65GraspJointTrajectoryPlanner(profile=profile, planning_target_builder=builder,
        target_envelope_filter=safety.envelope_filter,
        config=RM65GraspTrajectoryPlanningConfig(trajectory_parameterization="continuous_c2",
            trajectory_speed_profile=metadata["parameterization"]["speed_profile"]))
    if planner.scene.semantic_content_hash != metadata["scene_hash"]:
        raise ValueError("Frozen collision scene differs from current profile")
    return planner


def _visible_scene(args):
    geometry = ParallelJawSolidGeometry(**json.loads(args.geometry.read_text()))
    with np.load(args.visible_scene, allow_pickle=False) as data:
        return VisibleGraspScene(points=data["points"], target_mask=data["target_mask"], geometry=geometry)


def _validated_curve(planner, scene, waypoints, *, side, speed, terminal, deadline):
    velocity = load_joint_velocity_limit_profiles()[
        planner.profile.position_profile.ik_execution.joint_velocity_limit_profile]
    planner._remaining_path_planning_time_sec(deadline=deadline,
        monotonic=time.perf_counter, planning_phase="pipeline_stage_parameterization")
    part, _, _ = planner._continuous_leg_trajectory(waypoints=waypoints, moving_side=side,
        speed_scale=speed, joint_velocity_rad_s=np.deg2rad(velocity.max_velocity_deg_s) * speed,
        reference_hz=planner.profile.position_profile.rates.ik_hz)
    if (np.max(np.abs(part.qd_rad_s[[0, -1]])) > 1e-9
            or np.max(np.abs(part.qdd_rad_s2[[0, -1]])) > 1e-9
            or not part.metadata["continuous_hard_limit_certificate"]["passed"]):
        raise ValueError("Stage is not a certified stop-to-stop C2 trajectory")
    samples = tuple({"left": q[:6], "right": q[6:]} for q in part.q_rad)
    return {"timestamps": part.timestamps, "q": part.q_rad,
        "sparse": planner._revalidate(samples, moving_side=side, deadline=deadline),
        "visible": planner._revalidate_visible_gripper_path(samples, moving_side=side,
            visible_scene=scene, terminal_tool_tcp_world=terminal, deadline=deadline),
        "boundary_velocity_rad_s": part.qd_rad_s[[0, -1]].tolist(),
        "boundary_acceleration_rad_s2": part.qdd_rad_s2[[0, -1]].tolist()}


def rebuild_suffix(args):
    """Child process: native C2 and full safety checks, no motion port."""
    started = time.perf_counter()
    deadline = started + args.budget
    trajectory, metadata = load_grasp_trajectory_record(args.trajectory)
    prefix, _ = split_pregrasp(trajectory, metadata)
    planner, scene, reused = _worker_context(args, metadata)
    first = metadata["parameterization"]["legs"][0]["source_waypoint_count"] - 1
    with np.load(args.trajectory, allow_pickle=False) as data:
        source = {s: data[f"source_{s}_qpos"][first:].copy() for s in ("left", "right")}
    if any(not np.array_equal(q[0], prefix.qpos_by_side[s][-1]) for s, q in source.items()):
        raise ValueError("Local source path does not start at the exact prefix endpoint")
    side = metadata["moving_side"]
    waypoints = tuple({s: q[i] for s, q in source.items()} for i in range(len(source[side])))
    result = _validated_curve(planner, scene, waypoints, side=side, speed=float(metadata["speed_scale"]),
        terminal=np.asarray(metadata["target_tool_tcp_world"]), deadline=deadline)
    result["elapsed_sec"] = time.perf_counter() - started
    result.update(worker_pid=os.getpid(), context_reused=reused)
    return result


def rebuild_connector(args, measured):
    """Probe-only correction; preserve the old start guard and fixed arm."""
    started = time.perf_counter()
    trajectory, metadata = load_grasp_trajectory_record(args.trajectory)
    prefix, _ = split_pregrasp(trajectory, metadata)
    expected = {s: q[-1] for s, q in prefix.qpos_by_side.items()}
    side = metadata["moving_side"]
    inactive = "left" if side == "right" else "right"
    if not np.array_equal(measured[inactive], expected[inactive]):
        raise ValueError("Connector cannot compensate movement of the unauthorized arm")
    if np.max(np.abs(measured[side] - expected[side])) > REAL_AGENT_TRAJECTORY_START_MAX_DRIFT_RAD:
        raise ValueError("Measured drift exceeds the unchanged start guard")
    planner, scene, reused = _worker_context(args, metadata)
    result = _validated_curve(planner, scene,
        (measured, expected), side=side, speed=float(metadata["speed_scale"]),
        terminal=None, deadline=started + args.budget)
    result["elapsed_sec"] = time.perf_counter() - started
    result.update(worker_pid=os.getpid(), context_reused=reused)
    return result


def prepend_connector(connector, suffix):
    for side in ("left", "right"):
        if not np.array_equal(connector.qpos_by_side[side][-1], suffix.qpos_by_side[side][0]):
            raise ValueError("Connector endpoint differs from frozen suffix")
    return RealAgentJointTrajectory(
        np.concatenate((connector.time_from_start_sec, connector.time_from_start_sec[-1] + suffix.time_from_start_sec[1:])),
        {s: np.concatenate((connector.qpos_by_side[s], suffix.qpos_by_side[s][1:])) for s in ("left", "right")},
        tuple(set(connector.planner_evidence_refs + suffix.planner_evidence_refs)))


class KinematicSink:
    """Simulated measured state; no device method exists on this object."""
    def __init__(self, initial, scene=None, guard=None):
        self._qpos = {s: np.asarray(q).copy() for s, q in initial.items()}
        self.scene = scene
        self.guard = guard
        self.ready = True
        self.holds = []
        self.write(self._qpos)

    def read(self):
        if self.scene is not None:
            return {s: np.asarray(self.scene.get_arm_qpos(s)) for s in ("left", "right")}
        return {s: q.copy() for s, q in self._qpos.items()}

    def write(self, qpos):
        if self.scene is None:
            self._qpos = {s: np.asarray(q).copy() for s, q in qpos.items()}
        else:
            self.scene.set_qpos_by_side(qpos)
            self.scene.data.qvel[:] = 0.
            self.scene.forward()

    def skill_motion_owner_state(self):
        if self.guard is not None and not self.guard.motion_ready:
            return "fault", "fault"
        return ("running", "ready") if self.ready else ("hold", "hold")

    def hold_skill_motion(self, reason):
        self.holds.append(reason)
        self.ready = False
        return True


def simulate_case(prefix, suffix_supplier, *, builder, sink, side, scenario,
                  realtime=False, event=None, wait_budget=2., correction=None,
                  resume=None, admit=None):
    """Exercise existing port clocks/admission; the scheduler is probe-only."""
    if scenario not in {"ready", "late", "timeout", "cancel", "worker_failure", "state_drift", "small_state_drift", "small_drift_corrected", "guarded_resume", "scene_changed", "clock_gap"}:
        raise ValueError("Unknown pipeline scenario")
    if not np.isfinite(wait_budget) or wait_budget <= 0:
        raise ValueError("wait_budget must be finite and positive")
    event = event or (lambda *_: None)
    port = RealAgentMotionPort(config=RealAgentMotionPortConfig(active_sides=(side,),
        component_id_by_side={side: f"{side}-tool-tcp"}, execution_frame_id="offline-solver-world",
        execution_frame_spec_hash="offline-test-only"), target_builder=builder)
    now, dt = 1., 1. / 120.
    rows, times, stages = [], [], []
    first_suffix_elapsed = None
    first_suffix_wall_advance = None
    first_suffix_step = None
    gap_resume_advance = None
    resume_diagnostic = None
    prefix_duration = float(prefix.time_from_start_sec[-1])
    wall_origin = time.perf_counter()
    wall_tick = 0

    def publish():
        port.publish_measured_qpos_pair(qpos_by_side=sink.read(), measured_at=now,
            available_at=now, source_evidence_refs=("mujoco-kinematic-cut:offline-only",))
        # Publishing returns the derived TCP pose cut; resume must consume the
        # lower-owned joint cut, not reinterpret pose evidence as encoder data.
        return port.latest_measured_qpos()

    def activate(trajectory, name):
        nonlocal now
        sink.ready = True
        cut = publish()
        target = builder.tool_tcp_world_targets_for_qpos_pair(
            left_qpos=trajectory.qpos_by_side["left"][-1], right_qpos=trajectory.qpos_by_side["right"][-1])[side]
        activation = RealAgentMotionActivation(operation_id=name, request_id=name,
            activation_grant_id=f"offline-grant:{name}", activation_grant_hash="not-real-authority",
            component_id=f"{side}-tool-tcp", side=side, primitive_id="offline.pipeline@0",
            primitive_hash="not-a-deployed-skill", target_pose_world=target,
            execution_frame_id="offline-solver-world", execution_frame_spec_hash="offline-test-only",
            measured_state_revision=cut.revision, measured_state_hash=cut.semantic_content_hash,
            submitted_at=now, expires_at=now + 1., joint_trajectory=trajectory)
        if admit is not None:
            activation = admit(trajectory, name, cut, target, now)
        port.submit_activation(activation)
        if realtime:
            # Plan retention/hashing is non-real-time. The lower activation
            # clock starts at owner admission, not before that preparatory work.
            now = 1. + time.perf_counter() - wall_origin
            publish()
        return port.apply_control_owner_requests(owner=sink, now=now)

    def hold(name):
        port.request_hold(operation_id=name, reason="offline verified stage boundary", requested_at=now)
        state = port.apply_control_owner_requests(owner=sink, now=now)
        if state.phase != RealAgentMotionPhase.HELD or not state.hold_succeeded:
            raise RuntimeError("Simulated boundary HOLD failed")

    def advance_clock():
        nonlocal now, wall_tick
        if realtime:
            wall_tick += 1
            time.sleep(max(0., wall_origin + wall_tick * dt - time.perf_counter()))
            now = 1. + time.perf_counter() - wall_origin
        else:
            now += dt

    def record(stage):
        measured = sink.read()
        rows.append(np.concatenate((measured["left"], measured["right"])))
        times.append(now)
        stages.append(stage)

    def pulse(stage):
        advance_clock()
        cut = publish()
        record(stage)
        return cut, now

    def run_stage(trajectory, name):
        nonlocal now, first_suffix_elapsed, first_suffix_wall_advance, first_suffix_step, gap_resume_advance
        state = activate(trajectory, name)
        if state.phase != RealAgentMotionPhase.ACTIVE:
            return False
        elapsed, gap_injected = 0., False
        activated_at = now
        event("stage_started", {"stage": name, "now": now})
        # Advance through the owner's clamped terminal sample; a floating
        # tolerance here could HOLD before the exact segment endpoint.
        while elapsed < float(trajectory.time_from_start_sec[-1]):
            advance_clock()
            if sink.guard is not None and not sink.guard.motion_ready:
                hold(name)
                event("force_guard_blocked", {"stage": name, "now": now})
                return False
            if (scenario == "clock_gap" and name == "suffix" and not gap_injected
                    and elapsed > min(.1, float(trajectory.time_from_start_sec[-1]) / 4.)):
                before_gap = elapsed
                now += .5
                skipped = port.command_for_observation(TeleopObservation(timestamp=now), now)
                port.observe_session_step(command=skipped, command_dispatched=False)
                gap_injected = True
                event("skipped_delayed_frame", {"now": now})
                advance_clock()
            command = port.command_for_observation(TeleopObservation(timestamp=now), now)
            if not isinstance(command, TeleopCommand):
                raise TypeError("Active simulated trajectory emitted no command")
            before_command = sink.read()
            commanded = {s: command.commands[f"{s}_arm"].value for s in ("left", "right")}
            sink.write(commanded)
            port.observe_session_step(command=command, command_dispatched=True)
            elapsed = command.metadata["real_agent_motion_operation"]["trajectory_elapsed_sec"]
            if name == "suffix" and first_suffix_elapsed is None:
                first_suffix_elapsed = elapsed
                first_suffix_wall_advance = now - activated_at
                first_suffix_step = max(float(np.max(np.abs(commanded[s] - before_command[s])))
                    for s in commanded)
            if gap_injected and gap_resume_advance is None:
                gap_resume_advance = elapsed - before_gap
            publish()
            record(name)
        hold(name)
        event("stage_held", {"stage": name, "now": now})
        return True

    record("initial")
    if not run_stage(prefix, "prefix"):
        raise RuntimeError("Validated prefix was not admitted")
    boundary = sink.read()
    wait_started = now
    suffix, outcome, planning_error = None, "waiting", None
    while now - wait_started < wait_budget:
        advance_clock()
        publish()
        record("waiting")
        if scenario in {"cancel", "scene_changed"}:
            outcome = scenario
            break
        if now - wait_started >= wait_budget:
            break
        if scenario == "timeout" or (scenario == "late" and now - wait_started < 1.):
            continue
        try:
            suffix = suffix_supplier()
        except (RuntimeError, ValueError) as error:
            outcome = "worker_failure"
            planning_error = {"type": type(error).__name__, "message": str(error)[:512]}
            break
        if suffix is not None:
            break
    wait_error = max(float(np.max(np.abs(sink.read()[s] - boundary[s]))) for s in boundary)
    if wait_error != 0.:
        raise RuntimeError("Waiting at verified boundary changed qpos")
    suffix_activated = False
    waited = now - wait_started
    if suffix is not None:
        if scenario in {"state_drift", "small_state_drift", "small_drift_corrected", "guarded_resume"}:
            changed = sink.read()
            changed[side][0] += (2 if scenario == "state_drift" else .5) * REAL_AGENT_TRAJECTORY_START_MAX_DRIFT_RAD
            sink.write(changed)
        if scenario == "small_drift_corrected":
            if correction is None:
                raise ValueError("Corrected diagnostic case requires a certified connector")
            suffix = correction(sink.read(), suffix)
        try:
            if resume is not None:
                suffix, resume_diagnostic = resume(prefix, suffix, port, pulse, lambda: now)
            suffix_activated = run_stage(suffix, "suffix")
            outcome = "completed" if suffix_activated else "start_state_rejected"
        except ResumeRejected as error:
            outcome = str(error)
    elif outcome == "waiting":
        outcome = "timeout"
    event("case_finished", {"scenario": scenario, "outcome": outcome, "now": now})
    return {"scenario": scenario, "outcome": outcome, "suffix_activated": suffix_activated,
        "wait_duration_sec": waited,
        "wait_max_qpos_change_rad": wait_error, "prefix_duration_sec": prefix_duration,
        "first_suffix_elapsed_sec": first_suffix_elapsed,
        "first_suffix_wall_advance_sec": first_suffix_wall_advance,
        "first_suffix_qpos_step_rad": first_suffix_step,
        "gap_resume_clock_advance_sec": gap_resume_advance, "planning_error": planning_error,
        "resume": resume_diagnostic,
        "trajectory_clock_max_advance_sec": port.config.trajectory_clock_max_advance_sec,
        "hold_count": len(sink.holds), "port_hold_reason": port.snapshot().hold_reason,
        "timestamps": np.asarray(times), "qpos": np.asarray(rows), "stages": np.asarray(stages)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "canonical", "trajectory", "visible-scene", "geometry", "mujoco-contract", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--budget", type=float, default=12.,
        help="Planning/resume-check budget in seconds, not a hard process wall-time limit")
    args = p.parse_args()
    if not np.isfinite(args.budget) or args.budget <= 0:
        p.error("--budget must be finite and positive")
    args.output.mkdir(parents=True, exist_ok=False)
    trajectory, metadata = load_grasp_trajectory_record(args.trajectory)
    prefix, saved_suffix = split_pregrasp(trajectory, metadata)
    planner = _planner(args, metadata)
    scene_points = _visible_scene(args)
    prefix_rows = tuple({s: q[i] for s, q in prefix.qpos_by_side.items()}
        for i in range(len(prefix.time_from_start_sec)))
    planner._revalidate(prefix_rows, moving_side=metadata["moving_side"])
    planner._revalidate_visible_gripper_path(prefix_rows, moving_side=metadata["moving_side"],
        visible_scene=scene_points, terminal_tool_tcp_world=None)
    from sim.contracts import SimulationContract
    from sim.mujoco.actuation import cad_bimanual_dmtac_gripper_profile
    from sim.mujoco.scene import MujocoScene
    sim = MujocoScene.from_contract(SimulationContract.load(args.mujoco_contract),
        mjcf_path=args.output / "scene.mjcf", gripper_actuation_profile=cad_bimanual_dmtac_gripper_profile())
    reports = []
    connector_receipt = None
    with (args.output / "events.jsonl").open("w", buffering=1) as journal:
        journal_lock = threading.Lock()
        def event(name, fields):
            with journal_lock:
                journal.write(json.dumps({"event": name, "wall_time": time.time(), **fields}) + "\n")
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
            submitted = time.perf_counter()
            future = pool.submit(rebuild_suffix, args)
            event("worker_submitted", {})
            future.add_done_callback(lambda done: event("worker_returned", {
                "ready_after_sec": time.perf_counter() - submitted, "cancelled": done.cancelled()}))
            rebuilt = None
            def supply():
                nonlocal rebuilt
                if not future.done():
                    return None
                if rebuilt is None:
                    result = future.result()
                    rebuilt = RealAgentJointTrajectory(result["timestamps"],
                        {"left": result["q"][:, :6], "right": result["q"][:, 6:]}, trajectory.planner_evidence_refs)
                    event("worker_validated", {"native_work_sec": result["elapsed_sec"],
                        "observed_ready_sec": time.perf_counter() - submitted})
                return rebuilt
            def run(scenario, supplier, realtime=False):
                def correct(measured, suffix):
                    nonlocal connector_receipt
                    connector_receipt = pool.submit(rebuild_connector, args, measured).result(timeout=args.budget + 1.)
                    connector = RealAgentJointTrajectory(connector_receipt["timestamps"],
                        {"left": connector_receipt["q"][:, :6], "right": connector_receipt["q"][:, 6:]}, trajectory.planner_evidence_refs)
                    return prepend_connector(connector, suffix)
                sink = KinematicSink({s: q[0] for s, q in prefix.qpos_by_side.items()}, sim)
                admission = OfflinePlanAdmission(metadata["moving_side"]) if scenario == "guarded_resume" else None
                resumed_plan = None
                force_counts, last_force_times, max_force_gaps = {}, {}, {}
                force_lock = threading.Lock()
                if admission is not None:
                    from teleop_runtime.position_force_guard import (
                        PositionForceGuardMonitor,
                    )

                    def force_reader(side):
                        now = time.monotonic()
                        with force_lock:
                            force_counts[side] = force_counts.get(side, 0) + 1
                            if side in last_force_times:
                                max_force_gaps[side] = max(max_force_gaps.get(side, 0.), now - last_force_times[side])
                            last_force_times[side] = now
                        return {"sources": {"tool": [0.] * 6}, "monotonic_time_sec": now,
                            "wall_time_sec": time.time()}

                    sink.guard = PositionForceGuardMonitor(readers_by_side={
                        s: lambda s=s: force_reader(s) for s in ("left", "right")},
                        emergency_stop=lambda fault: event("synthetic_force_stop", fault.to_dict()))

                def resume(prefix, suffix, port, pulse, clock):
                    nonlocal resumed_plan

                    def check():
                        if not sink.guard.motion_ready:
                            raise ResumeRejected("force_guard_not_ready")

                    joined, cut, diagnostic = prepare_resume(prefix, suffix, port=port,
                        pulse=pulse, clock=clock, submit_connector=lambda q: pool.submit(rebuild_connector, args, q),
                        check_valid=check, event=event, budget=args.budget)
                    target = planner.target_builder.tool_tcp_world_targets_for_qpos_pair(
                        left_qpos=joined.qpos_by_side["left"][-1], right_qpos=joined.qpos_by_side["right"][-1])[metadata["moving_side"]]
                    resumed_plan = admission.retain(joined, cut, target, "suffix", clock())
                    diagnostic.update(plan_id=resumed_plan.plan_id, plan_expires_at=resumed_plan.expires_at,
                        authorization="synthetic fixture for this NEW plan; not a Gateway grant")
                    return joined, diagnostic

                def admit(trajectory, name, cut, target, now):
                    plan = admission.retain(trajectory, cut, target, name, now) if name == "prefix" else resumed_plan
                    activation = admission.activation(plan, cut, approved_plan_id=plan.plan_id, name=name, now=now)
                    event("new_plan_admitted", {"stage": name, "plan_id": plan.plan_id, "now": now})
                    return activation

                failure = None
                failures = []
                interrupt_types = (KeyboardInterrupt, SystemExit)
                try:
                    if sink.guard is not None:
                        sink.guard.start()
                    report = simulate_case(prefix, supplier, builder=planner.target_builder, sink=sink,
                        side=metadata["moving_side"], scenario=scenario, realtime=realtime, event=event, correction=correct,
                        resume=resume if admission else None, admit=admit if admission else None)
                    if sink.guard is not None:
                        report.update(force_guard=sink.guard.snapshot(), synthetic_force_sample_counts=force_counts,
                            synthetic_force_max_sample_gap_sec=max_force_gaps)
                except BaseException as error:
                    failure = error
                    failures.append(("body", error))
                finally:
                    for phase, owner in (
                        ("force monitor close", sink.guard),
                        ("plan store close", admission.store if admission is not None else None),
                    ):
                        if owner is None:
                            continue
                        try:
                            owner.close()
                        except BaseException as error:
                            failures.append((phase, error))
                            if failure is None or (
                                isinstance(error, interrupt_types)
                                and not isinstance(failure, interrupt_types)
                            ):
                                failure = error
                if failure is not None:
                    for phase, error in failures:
                        if error is failure:
                            continue
                        note = f"Secondary failure during {phase}:\n" + "".join(
                            traceback.format_exception(type(error), error, error.__traceback__)
                        )
                        add_note = getattr(failure, "add_note", None)
                        if callable(add_note):
                            add_note(note)
                        else:
                            notes = getattr(failure, "__notes__", None)
                            if notes is None:
                                notes = []
                                failure.__notes__ = notes
                            notes.append(note)
                    raise failure
                arrays = {k: report.pop(k) for k in ("timestamps", "qpos", "stages")}
                inactive = "left" if metadata["moving_side"] == "right" else "right"
                inactive_slice = slice(0, 6) if inactive == "left" else slice(6, 12)
                report["stationary_arm_max_change_rad"] = float(np.max(np.abs(
                    arrays["qpos"][:, inactive_slice] - arrays["qpos"][0, inactive_slice])))
                if report["outcome"] == "completed":
                    from sim.mujoco.transforms import pose_from_xyz_rpy
                    side = metadata["moving_side"]
                    tcp = planner.profile.tool_tcp_pose_flange_m_rad_by_side[side]
                    pose = sim.body_pose(f"{side}_arm_link6") @ pose_from_xyz_rpy(tcp[:3], tcp[3:])
                    position, orientation = pose_delta_metrics(np.asarray(metadata["target_tool_tcp_world"]), pose)
                    report.update(terminal_position_error_m=position, terminal_orientation_error_deg=orientation)
                np.savez_compressed(args.output / f"{scenario}.npz", **arrays)
                reports.append(report)
            # A spawned worker uses wall time. Do not race it against an
            # accelerated virtual control clock; faults below are injected.
            run("guarded_resume", supply, realtime=True)
            result = future.result(timeout=args.budget + 5.)
            supply()
            same_qpos = all(np.array_equal(saved_suffix.qpos_by_side[s], rebuilt.qpos_by_side[s]) for s in ("left", "right"))
            for scenario in ("ready", "late", "timeout", "cancel", "worker_failure", "state_drift", "small_state_drift", "small_drift_corrected", "scene_changed", "clock_gap"):
                def supplier(scenario=scenario):
                    if scenario == "worker_failure":
                        raise RuntimeError("Injected isolated planner failure")
                    return rebuilt
                run(scenario, supplier)
    expected = {"guarded_resume": "completed", "ready": "completed", "late": "completed", "timeout": "timeout", "cancel": "cancel",
        "worker_failure": "worker_failure", "state_drift": "start_state_rejected",
        "small_state_drift": "completed",
        "small_drift_corrected": "completed",
        "scene_changed": "scene_changed", "clock_gap": "completed"}
    passed = all(r["outcome"] == expected[r["scenario"]] and r["wait_max_qpos_change_rad"] == 0
        and r["stationary_arm_max_change_rad"] == 0
        and (r["outcome"] != "completed" or (
            r["terminal_position_error_m"] <= planner.config.endpoint_position_tolerance_m
            and r["terminal_orientation_error_deg"] <= planner.config.endpoint_orientation_tolerance_deg))
        and (r["first_suffix_elapsed_sec"] is None or math.isclose(r["first_suffix_elapsed_sec"],
            min(r["first_suffix_wall_advance_sec"], r["trajectory_clock_max_advance_sec"]), rel_tol=0., abs_tol=1e-9))
        and (r["gap_resume_clock_advance_sec"] is None or r["gap_resume_clock_advance_sec"] <= 1. / 120. + 1e-9)
        for r in reports)
    by_scenario = {r["scenario"]: r for r in reports}
    passed = passed and same_qpos and (
        by_scenario["small_drift_corrected"]["first_suffix_qpos_step_rad"]
        < by_scenario["small_state_drift"]["first_suffix_qpos_step_rad"])
    guarded = by_scenario["guarded_resume"]
    passed = passed and (guarded["force_guard"]["fault"] is None
        and guarded["force_guard"]["stale_watchdog_armed"]
        and not guarded["force_guard"]["non_motion_read_window_active"]
        and guarded["resume"]["worker_pid"] == result["worker_pid"]
        and guarded["resume"]["worker_context_reused"]
        and guarded["first_suffix_qpos_step_rad"] < 1e-6)
    worker_event = next(json.loads(line) for line in (args.output / "events.jsonl").read_text().splitlines()
        if json.loads(line)["event"] == "worker_returned")
    report = {"status": "validated_with_limitations" if passed else "failed", "hardware_connected": False,
        "real_execution_ready": False, "worker_ready_after_sec": worker_event["ready_after_sec"],
        "native_suffix_work_sec": result["elapsed_sec"], "same_suffix_qpos": same_qpos,
        "prefix_duration_sec": float(prefix.time_from_start_sec[-1]),
        "suffix_duration_sec": float(rebuilt.time_from_start_sec[-1]), "cases": reports,
        "connector": {k: v for k, v in connector_receipt.items() if k not in {"timestamps", "q"}},
        "connector_duration_sec": float(connector_receipt["timestamps"][-1]),
        "open_issues": ["Production staged Skill admission/authorization and noisy measured-state splice policy remain unimplemented"],
        "limits": "frozen local joint path; synthetic force sensors/authorization and failure injection; real monitor/store/port software, but kinematic simulation, not actuator, SDK or authenticated live Skill acceptance"}
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException as error:
        if sys.version_info < (3, 11):
            for note in getattr(error, "__notes__", ()):
                try:
                    print(note, file=sys.stderr)
                except (OSError, ValueError) as output_error:
                    error.__notes__.append(
                        f"Could not print secondary failure: {type(output_error).__name__}: {output_error}"
                    )
                    break
        raise
    raise SystemExit(exit_code)
