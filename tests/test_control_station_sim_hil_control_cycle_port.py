from __future__ import annotations

from types import SimpleNamespace

import pytest

from control_composition import build_composition_activation_scope
from control_station.sim_hil import (
    MujocoSimHILApplication,
    MujocoSimHILApplicationConfig,
)
from control_station.sim_hil_profile.allocation_control_cycle import (
    AllocationProposalExecutionControlCyclePort,
    SimHILAllocationActivationPrime,
    SimHILAllocationCyclePublication,
)
from control_station.sim_hil_profile.control_cycle import (
    SIM_HIL_POLICY_SOURCE_ALWAYS,
    SIM_HIL_POLICY_SOURCE_CONTROL_CYCLE,
    FixedAlphaCompatibilityControlCyclePort,
    SimHILControlCyclePortSpec,
)
from control_station.sim_hil_profile.status import _composition_step_status
from robot_motion.action import AuthoritativeControlStep


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _activation(epoch: int = 1):
    return build_composition_activation_scope(
        operation_spec_id="sim-hil-allocation-test",
        operation_spec_hash="sim-hil-allocation-operation-hash",
        session_epoch=epoch,
        activated_at=9.0 + epoch,
        clock_domain_id="control.monotonic",
        input_barrier_at=9.0 + epoch,
        required_roles=("policy", "human_residual", "human_absolute"),
    )


class _VirtualAllocationBinding:
    def __init__(
        self,
        activation_scope,
        events: list[str],
        *,
        publication_status: str = "no_action",
    ) -> None:
        self.activation_scope = activation_scope
        self.events = events
        self.publication_status = publication_status
        self.closed = False
        self.join_calls: list[float] = []
        self.publish_calls: list[dict[str, object]] = []
        self.prime_calls: list[dict[str, object]] = []
        self.policy_required = True

    def policy_emission_required(self) -> bool:
        return self.policy_required

    def publish_cycle(self, **kwargs):
        assert not self.closed
        self.publish_calls.append(kwargs)
        self.events.append(f"binding.publish:{self.activation_scope.session_epoch}")
        return SimHILAllocationCyclePublication(
            proposal_publication=SimpleNamespace(
                status=self.publication_status,
                semantic_content_hash=(
                    f"proposal-cycle:{self.activation_scope.session_epoch}"
                ),
                selected_context_id=None,
                selected_context_hash=None,
                selected_mode_by_component={},
                submission=None,
            ),
            observed_at=float(kwargs["now"]),
            human_submission=SimpleNamespace(role="human_residual"),
            policy_submission=SimpleNamespace(role="policy"),
        )

    def prime_sources(self, **kwargs):
        assert not self.closed
        self.prime_calls.append(kwargs)
        self.events.append(f"binding.prime:{self.activation_scope.session_epoch}")
        return SimHILAllocationActivationPrime(
            observed_at=float(kwargs["now"]),
            human_submission=SimpleNamespace(role="human_residual"),
            policy_submission=SimpleNamespace(role="policy"),
        )

    def execute_lower(self, *, now: float):
        assert not self.closed
        self.events.append(f"binding.execute:{self.activation_scope.session_epoch}")
        return AuthoritativeControlStep(status="not_due")

    def fence(self, *, reason: str) -> None:
        if not self.closed:
            self.events.append(
                "binding.fence:" f"{self.activation_scope.session_epoch}:{reason}"
            )
            self.closed = True

    def join(self, *, timeout_sec: float) -> bool:
        self.join_calls.append(timeout_sec)
        self.events.append(f"binding.join:{self.activation_scope.session_epoch}")
        return True


def test_fixed_alpha_port_preserves_exact_step_call_and_result() -> None:
    calls: list[dict[str, object]] = []
    expected = SimpleNamespace(execution=AuthoritativeControlStep(status="not_due"))
    composition = SimpleNamespace(
        step=lambda **kwargs: (calls.append(kwargs) or expected)
    )
    port = FixedAlphaCompatibilityControlCyclePort(composition)
    emission = SimpleNamespace(emission_id="policy:0")

    result = port.run_cycle(
        now=10.0,
        policy_emission=emission,
        poll_human=True,
        target_effective_at=10.05,
    )

    assert result is expected
    assert calls == [
        {
            "now": 10.0,
            "policy_emission": emission,
            "poll_human": True,
            "target_effective_at": 10.05,
        }
    ]
    port.deactivate(reason="test_hold")
    assert port.active is False
    with pytest.raises(RuntimeError, match="not active"):
        port.run_cycle(
            now=10.1,
            policy_emission=None,
            poll_human=True,
            target_effective_at=10.1,
        )
    port.activate(activation_scope=None)
    assert port.active is True


class _RecordingCyclePort:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.spec = SimHILControlCyclePortSpec(
            port_id="test-allocation-cycle-port",
            cycle_kind="allocation_proposal_execution",
            policy_source_requirement=SIM_HIL_POLICY_SOURCE_ALWAYS,
            application_blend_failure_fallback_enabled=False,
            application_human_timeout_fallback_enabled=False,
        )
        self.active = True
        self.calls: list[dict[str, object]] = []

    def activate(self, *, activation_scope) -> None:
        self.active = True
        self.events.append("port.activate")

    def deactivate(self, *, reason: str) -> None:
        self.active = False
        self.events.append(f"port.deactivate:{reason}")

    def run_cycle(self, **kwargs):
        assert self.active
        self.calls.append(kwargs)
        self.events.append("port.run_cycle")
        return SimpleNamespace(
            execution=AuthoritativeControlStep(status="not_due"),
            composition_step=None,
        )

    def close(self) -> None:
        self.active = False
        self.events.append("port.close")


def test_application_cycle_branch_is_replaceable_without_composition_step() -> None:
    events: list[str] = []
    port = _RecordingCyclePort(events)
    policy_emission = SimpleNamespace(emission_id="policy:1")
    runtime = SimpleNamespace(
        clock=lambda: 10.0,
        config=SimpleNamespace(decision_hz=30.0),
        reference_activation_spec=SimpleNamespace(
            activation_mode="immediate_target",
            target_lead_time_sec=0.0,
        ),
        capture_execution_gate_snapshot=lambda *, now: SimpleNamespace(status="ready"),
        next_decision_due_at=None,
    )
    app = MujocoSimHILApplication(
        config=MujocoSimHILApplicationConfig(),
        task_config=SimpleNamespace(),
        scene=SimpleNamespace(),
        runtime=runtime,
        composition=SimpleNamespace(
            human_alpha_by_component={"right_arm": 1.0},
            step=lambda **_kwargs: pytest.fail(
                "application bypassed the selected cycle port"
            ),
        ),
        control_cycle_port=port,
        policy_source=SimpleNamespace(step=lambda *, now: policy_emission),
        human_source=SimpleNamespace(
            step=lambda *, now: SimpleNamespace(
                emission=SimpleNamespace(emission_id=f"human:{now}")
            ),
        ),
        readiness_provider=SimpleNamespace(),
        clock=lambda: 10.0,
    )
    app._started = True
    app._next_decision_at = 10.0

    result = app.step(now=10.0)

    assert result.execution.status == "not_due"
    assert events == ["port.run_cycle"]
    assert port.calls == [
        {
            "now": 10.0,
            "policy_emission": policy_emission,
            "poll_human": True,
            "target_effective_at": 10.0,
        }
    ]
    assert app._last_human_timeout_fallback is None
    assert app._last_blend_failure_fallback is None


def test_allocation_profile_rejects_the_legacy_fixed_alpha_control() -> None:
    events: list[str] = []
    allocation_port = AllocationProposalExecutionControlCyclePort(
        port_id="test-allocation-no-alpha",
        binding_factory=lambda _scope: pytest.fail(
            "fixed-alpha rejection must not activate allocation resources"
        ),
    )
    composition = SimpleNamespace(
        set_human_blend_alpha=lambda *_args, **_kwargs: pytest.fail(
            "allocation profile must not call fixed-alpha composition control"
        )
    )
    app = MujocoSimHILApplication(
        config=MujocoSimHILApplicationConfig(),
        task_config=SimpleNamespace(),
        scene=SimpleNamespace(),
        runtime=SimpleNamespace(),
        composition=composition,
        control_cycle_port=allocation_port,
        policy_source=SimpleNamespace(),
        human_source=SimpleNamespace(),
        readiness_provider=SimpleNamespace(),
        clock=lambda: 10.0,
    )
    app._started = True

    with pytest.raises(RuntimeError, match="does not use fixed-alpha"):
        app.set_human_blend_alpha(0.5)

    assert events == []
    allocation_port.close()


def test_allocation_port_delegates_dynamic_policy_requirement() -> None:
    events: list[str] = []
    binding: _VirtualAllocationBinding | None = None

    def build(scope):
        nonlocal binding
        binding = _VirtualAllocationBinding(scope, events)
        return binding

    port = AllocationProposalExecutionControlCyclePort(
        port_id="dynamic-policy-source",
        binding_factory=build,
    )
    assert port.spec.policy_source_requirement == (
        SIM_HIL_POLICY_SOURCE_CONTROL_CYCLE
    )
    assert port.policy_emission_required() is True
    port.activate(activation_scope=_activation())
    assert binding is not None
    binding.policy_required = False
    assert port.policy_emission_required() is False
    binding.policy_required = True
    assert port.policy_emission_required() is True
    port.close()


def test_human_absolute_cycle_does_not_run_unselected_policy_source() -> None:
    events: list[str] = []
    bindings: list[_VirtualAllocationBinding] = []

    def build(scope):
        binding = _VirtualAllocationBinding(scope, events)
        bindings.append(binding)
        return binding

    port = AllocationProposalExecutionControlCyclePort(
        port_id="human-without-policy-hot-path",
        binding_factory=build,
    )
    scope = _activation()
    port.activate(activation_scope=scope)
    port.prime_activation(
        now=10.0,
        policy_emission=SimpleNamespace(emission_id="startup-policy"),
        human_emission=SimpleNamespace(emission_id="startup-human"),
        poll_human=False,
        target_effective_at=10.0,
    )
    bindings[0].policy_required = False
    runtime = SimpleNamespace(
        clock=lambda: 10.0,
        config=SimpleNamespace(decision_hz=30.0),
        reference_activation_spec=SimpleNamespace(
            activation_mode="immediate_target",
            target_lead_time_sec=0.0,
        ),
        capture_execution_gate_snapshot=lambda *, now: SimpleNamespace(
            status="ready"
        ),
        next_decision_due_at=None,
    )
    policy_calls: list[float] = []
    app = MujocoSimHILApplication(
        config=MujocoSimHILApplicationConfig(),
        task_config=SimpleNamespace(),
        scene=SimpleNamespace(),
        runtime=runtime,
        composition=SimpleNamespace(),
        control_cycle_port=port,
        policy_source=SimpleNamespace(
            step=lambda *, now: policy_calls.append(now)
        ),
        human_source=SimpleNamespace(),
        readiness_provider=SimpleNamespace(),
        clock=lambda: 10.0,
    )
    app._started = True
    app._next_decision_at = 10.0

    app.step(now=10.0)

    assert policy_calls == []
    assert bindings[0].publish_calls[-1]["policy_emission"] is None
    port.close()


def test_application_orders_port_lifecycle_around_lower_hold_and_rearm() -> None:
    events: list[str] = []
    port = _RecordingCyclePort(events)

    class Composition:
        started = False
        activation_scope = None
        human_alpha_by_component = {"right_arm": 0.0}

        @classmethod
        def start(cls):
            cls.started = True
            events.append("composition.start")
            return SimpleNamespace(passed=True)

        @staticmethod
        def request_hold(*, reason: str):
            events.append(f"composition.hold:{reason}")
            return SimpleNamespace()

        @staticmethod
        def rearm(*, now: float):
            events.append("composition.rearm")
            return SimpleNamespace(passed=True)

        @classmethod
        def close(cls):
            events.append("composition.close")
            cls.started = False

    app = MujocoSimHILApplication(
        config=MujocoSimHILApplicationConfig(enabled=True),
        task_config=SimpleNamespace(),
        scene=SimpleNamespace(),
        runtime=SimpleNamespace(
            clock=lambda: 10.0,
            reference_activation_spec=SimpleNamespace(
                activation_mode="immediate_target",
                target_lead_time_sec=0.0,
            ),
        ),
        composition=Composition(),
        control_cycle_port=port,
        policy_source=SimpleNamespace(
            reset_for_activation=lambda: events.append("policy.reset"),
            close=lambda: events.append("policy.close"),
        ),
        human_source=SimpleNamespace(),
        readiness_provider=SimpleNamespace(),
        clock=lambda: 10.0,
    )
    app._warmup_sources = lambda **_kwargs: events.append("source.warmup")
    app._wait_for_source_readiness = lambda **_kwargs: events.append("source.refresh")
    app._prime_activation_inputs = lambda: events.append("inputs.prime")
    app.start()
    app.hold(reason="operator_pause")
    app.rearm(now=10.0)
    app.close()

    first_deactivate = events.index(
        "port.deactivate:sim_hil_application:operator_pause"
    )
    first_hold = events.index("composition.hold:sim_hil_application:operator_pause")
    assert first_deactivate < first_hold
    rearm_deactivate = events.index(
        "port.deactivate:sim_hil_application:rearm_boundary"
    )
    rearm_hold = events.index("composition.hold:sim_hil_application:rearm_boundary")
    assert rearm_deactivate < rearm_hold
    assert events.count("port.activate") == 2
    assert events.index("port.close") < events.index("composition.close")


def test_application_force_cancellation_is_forwarded_only_when_requested() -> None:
    events: list[object] = []
    port = _RecordingCyclePort([])
    composition = SimpleNamespace(
        started=True,
        human_alpha_by_component={"right_arm": 0.0},
        request_hold=lambda **kwargs: (events.append(kwargs) or SimpleNamespace()),
    )
    app = MujocoSimHILApplication(
        config=MujocoSimHILApplicationConfig(enabled=True),
        task_config=SimpleNamespace(),
        scene=SimpleNamespace(),
        runtime=SimpleNamespace(clock=lambda: 10.0),
        composition=composition,
        control_cycle_port=port,
        policy_source=SimpleNamespace(),
        human_source=SimpleNamespace(),
        readiness_provider=SimpleNamespace(),
        clock=lambda: 10.0,
    )
    app._started = True
    outcome = app.end_episode(
        reason="agent_skill_terminal",
        force_cancellation=True,
    )
    assert outcome is not None
    assert events == [
        {
            "reason": "sim_hil_application:episode_end:agent_skill_terminal",
            "force_cancellation": True,
        }
    ]
    assert app.held and app.episode_ended


def test_application_still_requests_lower_hold_when_port_fence_fails() -> None:
    events: list[str] = []

    class FailingFencePort(_RecordingCyclePort):
        def deactivate(self, *, reason: str) -> None:
            self.active = False
            self.events.append(f"port.deactivate:{reason}")
            raise RuntimeError("virtual fence failure")

    port = FailingFencePort(events)
    composition = SimpleNamespace(
        started=True,
        human_alpha_by_component={"right_arm": 0.0},
        request_hold=lambda *, reason: (
            events.append(f"composition.hold:{reason}") or SimpleNamespace()
        ),
    )
    app = MujocoSimHILApplication(
        config=MujocoSimHILApplicationConfig(enabled=True),
        task_config=SimpleNamespace(),
        scene=SimpleNamespace(),
        runtime=SimpleNamespace(clock=lambda: 10.0),
        composition=composition,
        control_cycle_port=port,
        policy_source=SimpleNamespace(),
        human_source=SimpleNamespace(
            step=lambda *, now: SimpleNamespace(
                emission=SimpleNamespace(emission_id=f"human:{now}")
            ),
        ),
        readiness_provider=SimpleNamespace(),
        clock=lambda: 10.0,
    )
    app._started = True

    with pytest.raises(
        RuntimeError,
        match="control_cycle_fence_failed_after_lower_hold",
    ):
        app.hold(reason="virtual_fault")

    assert events == [
        "port.deactivate:sim_hil_application:virtual_fault",
        "composition.hold:sim_hil_application:virtual_fault",
    ]
    assert app._held is True


def test_allocation_port_is_lazy_and_no_action_still_advances_lower_owner() -> None:
    events: list[str] = []
    bindings: list[_VirtualAllocationBinding] = []

    def build_binding(scope):
        events.append(f"binding.build:{scope.session_epoch}")
        binding = _VirtualAllocationBinding(scope, events)
        bindings.append(binding)
        return binding

    port = AllocationProposalExecutionControlCyclePort(
        port_id="virtual-allocation-cycle",
        binding_factory=build_binding,
    )
    assert port.active is False
    assert bindings == []

    activation = _activation()
    port.activate(activation_scope=activation)
    port.prime_activation(
        now=10.05,
        policy_emission=None,
        human_emission=SimpleNamespace(emission_id="human:prime"),
        poll_human=True,
        target_effective_at=10.1,
    )
    result = port.run_cycle(
        now=10.1,
        policy_emission=None,
        poll_human=True,
        target_effective_at=10.12,
    )

    assert result.cycle_index == 0
    assert result.composition_step.status == "no_action"
    assert result.execution.status == "not_due"
    assert (
        _composition_step_status(result.composition_step)["publication_status"]
        == "no_action"
    )
    assert events == [
        "binding.build:1",
        "binding.prime:1",
        "binding.publish:1",
        "binding.execute:1",
    ]
    assert bindings[0].publish_calls == [
        {
            "now": 10.1,
            "policy_emission": None,
            "poll_human": True,
            "target_effective_at": 10.12,
        }
    ]
    assert getattr(bindings[0].prime_calls[0]["human_emission"], "emission_id") == (
        "human:prime"
    )


def test_application_allocation_prime_waits_for_a_fresh_human_emission() -> None:
    events: list[str] = []
    bindings: list[_VirtualAllocationBinding] = []
    clock_state = {"now": 10.0}
    human_calls = 0
    fresh_human = SimpleNamespace(emission_id="human:fresh-after-activation")

    class FreshnessCheckingBinding(_VirtualAllocationBinding):
        def prime_sources(self, **kwargs):
            policy_emission = kwargs["policy_emission"]
            assert policy_emission.effective_until > float(kwargs["now"])
            return super().prime_sources(**kwargs)

    def build_binding(scope):
        binding = FreshnessCheckingBinding(scope, events)
        bindings.append(binding)
        return binding

    def human_step(*, now):
        nonlocal human_calls
        human_calls += 1
        events.append(f"human.step:{now:.3f}")
        return SimpleNamespace(
            emission=None if human_calls == 1 else fresh_human,
        )

    def policy_step(*, now):
        events.append(f"policy.step:{now:.3f}")
        return SimpleNamespace(
            emission_id=f"policy:{now}",
            effective_until=float(now) + 0.001,
        )

    def sleep(duration):
        clock_state["now"] += float(duration)

    port = AllocationProposalExecutionControlCyclePort(
        port_id="virtual-allocation-fresh-prime",
        binding_factory=build_binding,
    )
    app = MujocoSimHILApplication(
        config=MujocoSimHILApplicationConfig(
            human_emission_timeout_sec=0.1,
        ),
        task_config=SimpleNamespace(),
        scene=SimpleNamespace(),
        runtime=SimpleNamespace(
            clock=lambda: clock_state["now"],
            reference_activation_spec=SimpleNamespace(
                activation_mode="immediate_target",
                target_lead_time_sec=0.0,
            ),
        ),
        composition=SimpleNamespace(activation_scope=_activation(1)),
        control_cycle_port=port,
        policy_source=SimpleNamespace(
            reset_for_activation=lambda: events.append("policy.reset"),
            step=policy_step,
        ),
        human_source=SimpleNamespace(step=human_step),
        readiness_provider=SimpleNamespace(),
        clock=lambda: clock_state["now"],
        sleep=sleep,
    )

    app._activate_and_prime_control_cycle()

    assert human_calls == 2
    assert len(bindings) == 1
    assert len(bindings[0].prime_calls) == 1
    assert bindings[0].prime_calls[0]["human_emission"] is fresh_human
    assert bindings[0].prime_calls[0]["poll_human"] is False
    assert events[:5] == [
        "policy.reset",
        "human.step:10.000",
        "human.step:10.005",
        "policy.step:10.005",
        "binding.prime:1",
    ]
    assert "binding.execute:1" not in events
    port.close()


def test_fixed_alpha_activation_policy_interval_starts_after_required_human_wait() -> None:
    events: list[str] = []
    clock_state = {"now": 10.0}
    human_calls = 0
    fresh_human = SimpleNamespace(emission_id="human:fixed-alpha-fresh")

    def human_step(*, now):
        nonlocal human_calls
        human_calls += 1
        events.append(f"human.step:{now:.3f}")
        return SimpleNamespace(
            emission=None if human_calls == 1 else fresh_human,
        )

    def policy_step(*, now):
        events.append(f"policy.step:{now:.3f}")
        return SimpleNamespace(
            emission_id=f"policy:{now}",
            effective_until=float(now) + 0.001,
        )

    def submit_policy(emission, *, observed_at):
        assert emission.effective_until > float(observed_at)
        events.append("policy.admitted")
        return SimpleNamespace(accepted=True, error=None)

    def submit_human(emission, *, observed_at):
        assert emission is fresh_human
        events.append("human.admitted")
        return SimpleNamespace(accepted=True, error=None)

    composition = SimpleNamespace(
        human_alpha_by_component={"right_arm": 0.5},
        submit_policy_emission=submit_policy,
        submit_human_emission=submit_human,
    )
    app = MujocoSimHILApplication(
        config=MujocoSimHILApplicationConfig(
            human_emission_timeout_sec=0.1,
        ),
        task_config=SimpleNamespace(),
        scene=SimpleNamespace(),
        runtime=SimpleNamespace(clock=lambda: clock_state["now"]),
        composition=composition,
        policy_source=SimpleNamespace(
            reset_for_activation=lambda: events.append("policy.reset"),
            step=policy_step,
        ),
        human_source=SimpleNamespace(step=human_step),
        readiness_provider=SimpleNamespace(),
        clock=lambda: clock_state["now"],
        sleep=lambda duration: clock_state.__setitem__(
            "now", clock_state["now"] + float(duration)
        ),
    )

    app._prime_activation_inputs()

    assert events == [
        "policy.reset",
        "human.step:10.000",
        "human.step:10.005",
        "policy.step:10.005",
        "policy.admitted",
        "human.admitted",
    ]


def test_allocation_port_requires_one_prime_per_activation() -> None:
    events: list[str] = []
    binding = _VirtualAllocationBinding(_activation(), events)
    port = AllocationProposalExecutionControlCyclePort(
        port_id="virtual-allocation-cycle",
        binding_factory=lambda _scope: binding,
    )
    port.activate(activation_scope=binding.activation_scope)

    with pytest.raises(RuntimeError, match="not primed"):
        port.run_cycle(
            now=10.0,
            policy_emission=None,
            poll_human=True,
            target_effective_at=10.0,
        )
    port.prime_activation(
        now=10.0,
        policy_emission=None,
        human_emission=SimpleNamespace(emission_id="human:prime"),
        poll_human=True,
        target_effective_at=10.0,
    )
    with pytest.raises(RuntimeError, match="already primed"):
        port.prime_activation(
            now=10.01,
            policy_emission=None,
            human_emission=SimpleNamespace(emission_id="human:prime"),
            poll_human=True,
            target_effective_at=10.01,
        )

    assert events == ["binding.prime:1"]
    port.close()
    assert events[-2:] == [
        "binding.fence:1:sim_hil_control_cycle_closed",
        "binding.join:1",
    ]


def test_allocation_port_retains_binding_for_join_after_fence_failure() -> None:
    events: list[str] = []

    class FailingFenceBinding(_VirtualAllocationBinding):
        def fence(self, *, reason: str) -> None:
            super().fence(reason=reason)
            raise RuntimeError("virtual fence failure")

    binding = FailingFenceBinding(_activation(), events)
    port = AllocationProposalExecutionControlCyclePort(
        port_id="virtual-allocation-cycle",
        binding_factory=lambda _scope: binding,
    )
    port.activate(activation_scope=binding.activation_scope)
    port.prime_activation(
        now=10.0,
        policy_emission=None,
        human_emission=SimpleNamespace(emission_id="human:prime"),
        poll_human=True,
        target_effective_at=10.0,
    )

    with pytest.raises(RuntimeError, match="virtual fence failure"):
        port.deactivate(reason="virtual_pause")
    port.close()

    assert binding.join_calls == [2.0]
    assert events[-2:] == [
        "binding.fence:1:virtual_pause",
        "binding.join:1",
    ]


def test_allocation_port_fences_old_activation_before_rebuild() -> None:
    events: list[str] = []
    bindings: list[_VirtualAllocationBinding] = []

    def build_binding(scope):
        binding = _VirtualAllocationBinding(scope, events)
        bindings.append(binding)
        return binding

    port = AllocationProposalExecutionControlCyclePort(
        port_id="virtual-allocation-cycle",
        binding_factory=build_binding,
    )
    first = _activation(1)
    second = _activation(2)
    port.activate(activation_scope=first)
    port.prime_activation(
        now=10.0,
        policy_emission=None,
        human_emission=SimpleNamespace(emission_id="human:prime:1"),
        poll_human=True,
        target_effective_at=10.0,
    )
    port.deactivate(reason="operator_pause")
    assert bindings[0].closed
    with pytest.raises(RuntimeError, match="not active"):
        port.run_cycle(
            now=10.2,
            policy_emission=None,
            poll_human=True,
            target_effective_at=10.2,
        )

    port.activate(activation_scope=second)
    port.prime_activation(
        now=11.0,
        policy_emission=None,
        human_emission=SimpleNamespace(emission_id="human:prime:2"),
        poll_human=True,
        target_effective_at=11.0,
    )
    result = port.run_cycle(
        now=11.1,
        policy_emission=None,
        poll_human=True,
        target_effective_at=11.1,
    )
    assert result.cycle_index == 0
    assert bindings[0] is not bindings[1]
    assert bindings[1].activation_scope.activation_id == second.activation_id
    assert "binding.fence:1:operator_pause" in events
    assert bindings[0].join_calls == []
    port.close()
    assert bindings[0].join_calls == [2.0]
    assert bindings[1].join_calls == [2.0]


def test_application_episode_end_closes_allocation_binding_before_lower_hold() -> None:
    events: list[str] = []

    def build_binding(scope):
        events.append(f"binding.build:{scope.session_epoch}")
        return _VirtualAllocationBinding(scope, events)

    port = AllocationProposalExecutionControlCyclePort(
        port_id="virtual-allocation-cycle",
        binding_factory=build_binding,
    )

    class Composition:
        started = False
        activation_scope = None
        human_alpha_by_component = {"right_arm": 0.5}

        @classmethod
        def start(cls):
            cls.started = True
            cls.activation_scope = _activation(1)
            events.append("composition.start")
            return SimpleNamespace(passed=True)

        @staticmethod
        def request_hold(*, reason: str):
            events.append(f"composition.hold:{reason}")
            return SimpleNamespace()

        @classmethod
        def close(cls):
            cls.started = False
            events.append("composition.close")

    app = MujocoSimHILApplication(
        config=MujocoSimHILApplicationConfig(enabled=True),
        task_config=SimpleNamespace(),
        scene=SimpleNamespace(),
        runtime=SimpleNamespace(
            clock=lambda: 10.0,
            reference_activation_spec=SimpleNamespace(
                activation_mode="immediate_target",
                target_lead_time_sec=0.0,
            ),
        ),
        composition=Composition(),
        control_cycle_port=port,
        policy_source=SimpleNamespace(
            reset_for_activation=lambda: events.append("policy.reset"),
            step=lambda *, now: SimpleNamespace(emission_id=f"policy:{now}"),
            close=lambda: events.append("policy.close"),
        ),
        human_source=SimpleNamespace(
            step=lambda *, now: SimpleNamespace(
                emission=SimpleNamespace(emission_id=f"human:{now}")
            ),
            close=lambda: events.append("human.close"),
        ),
        readiness_provider=SimpleNamespace(),
        clock=lambda: 10.0,
    )
    app._warmup_sources = lambda **_kwargs: events.append("source.warmup")
    app.start()
    app.end_episode(reason="virtual_episode_end")

    binding_close = events.index(
        "binding.fence:1:sim_hil_application:episode_end:virtual_episode_end"
    )
    lower_hold = events.index(
        "composition.hold:sim_hil_application:episode_end:virtual_episode_end"
    )
    assert binding_close < lower_hold
    assert app._episode_ended is True
    assert app._held is True
    app.close()
    assert events.index("binding.join:1") > lower_hold
    assert events.count("human.close") == 1
