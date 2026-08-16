from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from agent_gateway.profiles.mujoco.motion_application import (
    MujocoAgentSkillMotionApplication,
)
from agent_gateway.profiles.mujoco.motion_skill import (
    MujocoAgentTcpPoseSkillLifecycle,
)


@pytest.mark.parametrize(
    ("target_lead_time_sec", "period_sec", "expected_target"),
    (
        (0.0, 0.05, 10.05),
        (0.15, 0.05, 10.15),
    ),
)
def test_submit_and_step_places_new_proposal_after_its_admission_cut(
    target_lead_time_sec: float,
    period_sec: float,
    expected_target: float,
) -> None:
    calls: dict[str, object] = {}
    schedule = SimpleNamespace(
        target_lead_time_sec=target_lead_time_sec,
        period_sec=period_sec,
    )

    def submit_component_action(action, **kwargs):
        calls["action"] = action
        calls["submit"] = kwargs
        return SimpleNamespace(accepted=True)

    def step(**kwargs):
        calls["step"] = kwargs
        return SimpleNamespace(status="dispatched")

    runtime = SimpleNamespace(
        clock=lambda: 10.0,
        authoritative_port=SimpleNamespace(
            profile=SimpleNamespace(
                target_lead_time_sec=schedule.target_lead_time_sec,
                decision_period_sec=schedule.period_sec,
            ),
        ),
        submit_component_action=submit_component_action,
        step=step,
    )
    application = object.__new__(MujocoAgentSkillMotionApplication)
    application.runtime = runtime
    application._lock = threading.RLock()
    application._started = True
    application._closed = False
    application._active_operation_id = "operation:fixture"

    action = object()
    result = application.submit_and_step(
        operation_id="operation:fixture",
        action=action,
        source_event_id="event:fixture",
        observation_cut_id="observation:fixture",
        context_ids=("context:fixture",),
        physics_ticks_after_dispatch=3,
    )

    assert result.proposal_admission.accepted
    assert result.authoritative_step.status == "dispatched"
    assert calls["action"] is action
    assert calls["submit"] == {
        "source_event_id": "event:fixture",
        "produced_at": 10.0,
        "target_effective_at": pytest.approx(expected_target),
        "observation_cut_id": "observation:fixture",
        "context_ids": ("operation:fixture", "context:fixture"),
    }
    assert calls["step"] == {
        "now": pytest.approx(expected_target),
        "physics_ticks_after_dispatch": 3,
    }


def test_first_safety_profile_activation_establishes_lower_hold_before_ready() -> None:
    calls: list[tuple[str, object]] = []

    def transition(status: str, **kwargs):
        calls.append((f"gate:{status}", kwargs))
        return SimpleNamespace(
            snapshot=SimpleNamespace(status=status),
            lower_event_ids=(f"lower-gate:{status}",),
        )

    runtime = SimpleNamespace(
        clock=lambda: 10.0,
        start=lambda: calls.append(("start", None)),
        capture_execution_gate_snapshot=lambda **_kwargs: SimpleNamespace(
            status="ready"
        ),
        transition_execution_gate=transition,
        authoritative_port=SimpleNamespace(
            capture_receiver_state=lambda: SimpleNamespace(
                current_execution_epoch=1
            )
        ),
    )

    def activate_envelope(**kwargs):
        calls.append(("envelope:activate", kwargs))
        return SimpleNamespace(semantic_content_hash="safety-activation:1")

    application = object.__new__(MujocoAgentSkillMotionApplication)
    application.runtime = runtime
    application.tcp_pose_envelope_runtime = SimpleNamespace(
        activate=activate_envelope,
        deactivate_after_hold=lambda **_kwargs: None,
    )
    application._lock = threading.RLock()
    application._started = False
    application._held = False
    application._closed = False
    application._active_operation_id = None

    activation = application.activate_operation(
        operation_id="operation:fixture",
        route_id="sim.mujoco.motion",
        safety_selection=object(),
    )

    assert activation.snapshot.status == "ready"
    assert [name for name, _payload in calls] == [
        "start",
        "gate:hold",
        "envelope:activate",
        "gate:ready",
    ]
    envelope_call = calls[2][1]
    assert envelope_call["execution_epoch"] == 1
    assert "lower-gate:hold" in envelope_call["source_event_ids"]


def test_tcp_pose_lifecycle_holds_without_dispatch_when_goal_is_already_stable() -> None:
    lifecycle = object.__new__(MujocoAgentTcpPoseSkillLifecycle)
    lifecycle._activated = True
    lifecycle.application = SimpleNamespace(
        runtime=SimpleNamespace(config=SimpleNamespace(decision_hz=20.0)),
        submit_and_step=lambda **_kwargs: pytest.fail(
            "an already-satisfied TCP goal must not dispatch"
        ),
    )
    lifecycle.motion_profile = SimpleNamespace(
        stable_measurement_count=3,
        max_control_cycles=4,
        physics_ticks_per_control_cycle=1,
    )
    lifecycle._validate_call = lambda **_kwargs: None
    lifecycle._capture_pose = lambda: (0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0)
    lifecycle._within_terminal_tolerance = lambda _pose: True
    hold_reasons: list[str] = []
    lifecycle._hold = lambda *, reason: (
        hold_reasons.append(reason)
        or SimpleNamespace(lower_hold_confirmed=True)
    )
    lifecycle._failure = lambda **_kwargs: pytest.fail(
        "stable goal with confirmed HOLD must succeed"
    )
    success = SimpleNamespace(succeeded=True)
    lifecycle._success = lambda **_kwargs: success
    sleeps: list[float] = []
    lifecycle.sleep = sleeps.append

    completion = lifecycle.run(
        request=object(),
        operation=object(),
        cancel_requested=lambda: False,
    )

    assert completion is success
    assert hold_reasons == ["tcp_pose_target_already_satisfied"]
    assert sleeps == [pytest.approx(0.05), pytest.approx(0.05)]


def test_tcp_pose_lifecycle_keeps_bounded_first_and_latest_step_evidence() -> None:
    lifecycle = object.__new__(MujocoAgentTcpPoseSkillLifecycle)
    lifecycle._execution_refs = {"gate-ready:1", "gate-hold:1"}
    lifecycle._first_step_refs = ()
    lifecycle._latest_step_refs = ()

    for cycle in range(200):
        facts = tuple(
            SimpleNamespace(
                event_type=f"event_type_{index}",
                fact_id=f"fact:{cycle}:{index}",
            )
            for index in range(80)
        )
        lifecycle._remember_step_evidence(
            SimpleNamespace(
                proposal_admission=SimpleNamespace(
                    proposal_id=f"proposal:{cycle}",
                    semantic_content_hash=f"admission:{cycle}",
                ),
                authoritative_step=SimpleNamespace(
                    trace_facts=facts,
                    target_execution_request=SimpleNamespace(
                        request_id=f"target-request:{cycle}"
                    ),
                ),
            )
        )

    refs = lifecycle._execution_evidence_refs()

    assert len(refs) <= 128
    assert "gate-ready:1" in refs
    assert "gate-hold:1" in refs
    assert "proposal:0" in refs
    assert "proposal:199" in refs
    assert "target-request:0" in refs
    assert "target-request:199" in refs
    assert "proposal:1" not in refs
