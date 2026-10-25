"""Preflight uses lower geometry without replacing the requested goal."""

from types import SimpleNamespace
import math

import pytest

from agent_gateway.profiles.mujoco.motion_skill import (
    MujocoAgentSafeTcpPoseSkillLifecycle, MujocoAgentTcpPoseSkillLifecycle,
)
from agent_gateway.runtime.station_operation_worker import StationLifecycleActivationEvidence
from sim.mujoco.deploy.planned_tcp_motion import MujocoTcpPlanRejected
from test_tcp_pose_envelope import _anchor, _pose, _profile, _project
from robot_motion.target_safety.tcp_pose_envelope import TCPPoseEnvelopeSideConfig


@pytest.mark.parametrize("policy,goal_x,cut_locus,expected_reason", (
    ("within_arrival_tolerance", .105, False, "test_planner_reached"),
    ("within_arrival_tolerance", .115, False, "safe_tcp_goal_outside_safety_envelope"),
    ("preserve_target", .105, False, "safe_tcp_goal_outside_safety_envelope"),
    ("preserve_target", .102, False, "test_planner_reached"),
    ("within_arrival_tolerance", .05, True, "test_planner_reached"),
))
def test_preflight_preserves_goal_policy_and_hold(monkeypatch, policy, goal_x, cut_locus, expected_reason):
    profile = _profile({"right": TCPPoseEnvelopeSideConfig(
        relative_min_xyz_m=(-.1, -.1, -.1), relative_max_xyz_m=(.1, .1, .1),
        max_rotation_rad=math.pi - .02 if cut_locus else None,
    )})
    anchor = _anchor(profile, {"right": _pose()})
    safety = SimpleNamespace(project=lambda targets: _project(targets, profile, anchor=anchor))
    calls, holds = [], []
    lifecycle = object.__new__(MujocoAgentSafeTcpPoseSkillLifecycle)
    lifecycle.application = SimpleNamespace(tcp_pose_envelope_runtime=safety)
    lifecycle.component_id = "right_arm"
    lifecycle._activation_measured_pose = (.05, 0., 0., 0., 0., 0., 1.)
    lifecycle.target_pose = (goal_x, 0., 0., 0., 0., 0., 1.)
    if cut_locus:
        lifecycle._activation_measured_pose = (.05, 0., 0., math.sin((math.pi-.04)/2), 0., 0., math.cos((math.pi-.04)/2))
        lifecycle.target_pose = (goal_x, 0., 0., 1., 0., 0., 0.)
    lifecycle.route = SimpleNamespace(skill_id="motion.move_tcp_to_pose_safe")
    lifecycle.request = SimpleNamespace(parameters={"pose_goal_policy": policy})
    lifecycle.motion_profile = SimpleNamespace(translation_tolerance_m=.01, rotation_tolerance_rad=.09,
        normalize_ik_objective=False)
    lifecycle._lower_operation_id = "test:operation"
    lifecycle._contact_binding = None
    lifecycle._observed_planning_provider = None
    lifecycle._path_search_time_budget_sec = 3.

    def plan(target, **kwargs):
        calls.append("plan")
        assert target == lifecycle.target_pose
        assert kwargs["translation_tolerance_m"] == (.003 if policy == "preserve_target" else .01)
        raise MujocoTcpPlanRejected("test_planner_reached")

    lifecycle._planner = SimpleNamespace(plan=plan)
    lifecycle._hold = lambda **kwargs: (
        holds.append(kwargs["reason"]) or SimpleNamespace(
            evidence_refs=("test:lower-hold",), lower_hold_confirmed=True))
    monkeypatch.setattr(MujocoAgentTcpPoseSkillLifecycle, "activate", lambda *args, **kwargs:
        StationLifecycleActivationEvidence(succeeded=True, status="ready", lower_operation_id="test:operation",
            evidence_refs=("test:activation",), lower_hold_confirmed=False))
    evidence = lifecycle.activate(request=SimpleNamespace(expected_station_revision=0),
        operation=None, cancel_requested=lambda: False)
    assert evidence.status == expected_reason
    assert not evidence.succeeded and evidence.lower_hold_confirmed
    assert evidence.evidence_refs == ("test:lower-hold",)
    assert holds == [expected_reason]
    assert calls == (["plan"] if expected_reason == "test_planner_reached" else [])
    assert lifecycle.target_pose[0] == goal_x


@pytest.mark.integration
@pytest.mark.parametrize("motion_side", ("left", "right"))
def test_native_segmented_moves_keep_initial_safety_reference(monkeypatch, motion_side):
    from test_agent_gateway_mujoco_safe_tcp_pose import CLIENT, _request, _session

    facts = []
    session = _session(facts, motion_side=motion_side)
    try:
        app = session.application
        app.start_observation_ready()
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        component = motion_side + "_arm"
        safety = app.tcp_pose_envelope_runtime
        reference = safety.snapshot().safety_reference_hash
        pose = list(adapter.get_observation_snapshot()["measurement_summary"][
            "pose_xyz_quat_xyzw_by_component"][component])
        pose[2] += .03
        first = adapter.submit_skill_request(_request(session, adapter, pose).to_wire())
        assert first["accepted"]
        assert session.wait(timeout_sec=35.)
        completed = session.service.get_operation(first["operation"]["operation_id"])
        assert completed.execution_outcome == "completed", completed.to_wire()
        assert safety.snapshot().safety_reference_hash == reference
        measured = list(adapter.get_observation_snapshot()["measurement_summary"][
            "pose_xyz_quat_xyzw_by_component"][component])
        measured[2] += .085
        command_count = sum(f.event_type == "command" for f in facts)

        def unexpected_plan(*args, **kwargs):
            pytest.fail("Goal outside fixed envelope must fail before IK/path planning")

        monkeypatch.setattr(app.target.planned_tcp_motion, "plan", unexpected_plan)
        second = adapter.submit_skill_request(_request(session, adapter, measured, sequence=1).to_wire())
        assert second["accepted"]
        assert session.wait(timeout_sec=35.)
        rejected = session.service.get_operation(second["operation"]["operation_id"])
        assert rejected.reason_code == "safe_tcp_goal_outside_safety_envelope", rejected.to_wire()
        assert rejected.progress_summary["lower_hold_confirmed"] is True
        assert app.held and not app.target.planned_tcp_motion.active
        assert sum(f.event_type == "command" for f in facts) == command_count
        assert safety.snapshot().safety_reference_hash == reference
    finally:
        closed = session.close()
        assert closed.accepted and closed.port_receipt.succeeded
        assert closed.port_receipt.proves_motion_or_hold and closed.port_receipt.evidence_refs
