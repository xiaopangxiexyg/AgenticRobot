from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_gateway.profiles.mujoco.policy_application import (
    MujocoPolicySkillMotionApplication,
)
from agent_gateway.profiles.mujoco.policy_factory import (
    MujocoPolicySkillApplicationProfile,
    build_mujoco_policy_skill_application,
)
from robot_motion.deploy import PolicyActionEmission
from sim.mujoco.deploy import (
    MujocoPolicySkillMotionTarget,
    MujocoSimHILTargetConfig,
)


class _Clock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


class _FakePolicySource:
    ready = True
    last_error = None
    maximum_hold_sec = 0.25

    def __init__(self, events: list[str], emission=None) -> None:
        self.events = events
        self.emission = emission

    def start(self) -> None:
        self.events.append("source.start")

    def reset_for_activation(self) -> None:
        self.events.append("source.reset")

    def step(self, *, now=None, step_index=None):
        del now, step_index
        self.events.append("source.step")
        value, self.emission = self.emission, None
        return value

    def status(self, *, now=None):
        del now
        return {"ready": self.ready}

    def close(self) -> None:
        self.events.append("source.close")


class _FailingStartPolicySource(_FakePolicySource):
    def start(self) -> None:
        self.events.append("source.start")
        raise RuntimeError("source unavailable")


class _FakeRuntime:
    def __init__(self, events: list[str], clock: _Clock) -> None:
        self.events = events
        self.clock = clock
        self.config = SimpleNamespace(
            enable_primary_policy_authority=True,
            enable_composer_authority=False,
            enable_replay_authority=False,
            initial_control_mode="policy",
            direct_component_proposal_source=None,
        )
        self.component_execution_binding = SimpleNamespace(
            policy_action_ingress=lambda _action: object(),
        )
        self.composition_runtime_config = None
        self.gate = "hold"

    def start(self) -> None:
        self.events.append("runtime.start")

    def capture_execution_gate_snapshot(self, *, now):
        del now
        return SimpleNamespace(status=self.gate)

    def transition_execution_gate(
        self,
        status,
        *,
        reason,
        source_event_ids=(),
        force_cancellation=False,
    ):
        del reason, source_event_ids, force_cancellation
        self.events.append(f"runtime.{status}")
        self.gate = status
        return SimpleNamespace(
            snapshot=SimpleNamespace(status=status),
            lower_event_ids=(f"lower:{status}",),
        )

    def submit_policy_action_emission(self, emission):
        self.events.append("runtime.submit")
        return SimpleNamespace(emission=emission, accepted=True)

    def step(self, *, now, physics_ticks_after_dispatch):
        del now, physics_ticks_after_dispatch
        self.events.append("runtime.step")
        return SimpleNamespace(status="dispatched")

    def close(self) -> None:
        self.events.append("runtime.close")


def _target(runtime: _FakeRuntime) -> MujocoPolicySkillMotionTarget:
    return MujocoPolicySkillMotionTarget(
        task_config=object(),
        scene=object(),
        runtime=runtime,
        hand_specs_by_side={},
        target_frame_profile=object(),
        tool_frame_profile=object(),
        task_initialization_info={},
    )


def test_policy_skill_application_serializes_source_runtime_and_lower_hold() -> None:
    events: list[str] = []
    clock = _Clock()
    emission = object.__new__(PolicyActionEmission)
    runtime = _FakeRuntime(events, clock)
    source = _FakePolicySource(events, emission=emission)
    application = MujocoPolicySkillMotionApplication(
        target=_target(runtime),
        policy_source=source,
    )

    ready = application.activate_operation(operation_id="operation-1")
    cycle = application.step(operation_id="operation-1")
    hold = application.request_hold(
        reason="test_cancel",
        operation_id="operation-1",
    )
    application.release_operation(operation_id="operation-1")
    application.close_after_hold()

    assert ready.snapshot.status == "ready"
    assert cycle.emission is emission
    assert cycle.proposal_submission.accepted is True
    assert cycle.authoritative_step.status == "dispatched"
    assert hold.snapshot.status == "hold"
    assert events == [
        "runtime.start",
        "source.start",
        "source.reset",
        "runtime.ready",
        "source.step",
        "runtime.submit",
        "runtime.step",
        "runtime.hold",
        "source.reset",
        "source.close",
        "runtime.close",
    ]


def test_policy_skill_application_rejects_a_composition_target() -> None:
    events: list[str] = []
    runtime = _FakeRuntime(events, _Clock())
    runtime.composition_runtime_config = object()

    with pytest.raises(ValueError, match="single-policy, no-composition"):
        MujocoPolicySkillMotionApplication(
            target=_target(runtime),
            policy_source=_FakePolicySource(events),
        )


def test_policy_source_start_failure_leaves_closeable_lower_hold() -> None:
    events: list[str] = []
    runtime = _FakeRuntime(events, _Clock())
    application = MujocoPolicySkillMotionApplication(
        target=_target(runtime),
        policy_source=_FailingStartPolicySource(events),
    )

    with pytest.raises(RuntimeError, match="source unavailable"):
        application.activate_operation(operation_id="operation-1")

    assert application.started is True
    assert application.held is True
    application.close_after_hold()
    assert events == [
        "runtime.start",
        "source.start",
        "runtime.hold",
        "source.close",
        "runtime.close",
    ]


def test_policy_skill_factory_requires_generic_non_scripted_target_config() -> None:
    profile = MujocoPolicySkillApplicationProfile(
        output_active_sides=("right",),
    )

    with pytest.raises(ValueError, match="allow_scripted_task_reset=False"):
        build_mujoco_policy_skill_application(
            target_config=MujocoSimHILTargetConfig(
                task_config_path="does-not-need-to-exist.json",
            ),
            application_profile=profile,
            policy_source_builder=lambda _context: _FakePolicySource([]),
        )

    assert profile.output_active_sides == ("right",)


def test_policy_skill_application_profile_allows_future_bimanual_scope() -> None:
    profile = MujocoPolicySkillApplicationProfile(
        output_active_sides=("left", "right"),
        component_profile_id="fixture.bimanual-policy-skill.v1",
    )

    assert profile.output_active_sides == ("left", "right")
