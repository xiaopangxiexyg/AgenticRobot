from __future__ import annotations

from types import SimpleNamespace

import pytest

from control_composition.allocation import (
    ALLOCATION_PUBLICATION_OUTCOME_ACCEPTED_NEW,
    COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
    COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY,
)
from control_station.composition import (
    MujocoAllocationControlCycleBinding,
    MujocoAllocationHumanInputSubmission,
    MujocoAllocationSourcePublication,
    MujocoPolicyHumanAllocationSourcePublisher,
)
from control_station.sim_hil_profile.allocation_control_cycle import (
    AllocationProposalExecutionControlCyclePort,
)
from control_station.sim_hil_profile.status import _input_submission_status
from robot_motion.action import AuthoritativeControlStep
from tests.test_control_station_mujoco_allocation_publication import _publisher
from tests.test_control_station_mujoco_allocation_shadow import _basis, _fixture
from tests.test_control_station_mujoco_sim_hil import (
    _emission,
    _human_emission,
)


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def test_allocation_human_submission_has_multi_role_status_shape() -> None:
    submission = MujocoAllocationHumanInputSubmission(
        source_emission_id="human:rejected",
        roles=("human_absolute", "human_residual"),
        error="human_source_validation_failed:no_action",
    )

    status = _input_submission_status(submission)

    assert status["role"] is None
    assert status["roles"] == ["human_absolute", "human_residual"]
    assert status["accepted"] is False
    assert status["input_cycle_status"] is None


def test_source_publisher_uses_supplied_prime_without_polling() -> None:
    events = []
    clock_state = {"now": 10.0}
    human = _human_emission(
        sequence=0,
        emitted_at=10.0,
        arm_x=0.1,
        hand_qpos=0.2,
    )
    publisher = MujocoPolicyHumanAllocationSourcePublisher(
        human_source=SimpleNamespace(
            step=lambda **_kwargs: pytest.fail(
                "activation prime must use its explicit fresh emission"
            )
        ),
        publish_policy_input=lambda emission, observed_at: (
            events.append(("policy", emission.emission_id, observed_at))
            or SimpleNamespace(accepted=True)
        ),
        publish_human_input=lambda emission, observed_at, target_at: (
            events.append(("human", emission.emission_sequence, observed_at, target_at))
            or SimpleNamespace(
                accepted=True,
                publication=SimpleNamespace(status="admitted"),
            )
        ),
        clock=lambda: clock_state["now"],
        clock_domain_id="control.monotonic",
    )
    clock_state["now"] = 10.02

    result = publisher(
        10.0,
        _emission(),
        human,
        False,
        10.01,
    )

    assert result.observed_at == pytest.approx(10.02)
    assert result.human_step is None
    assert result.input_publication.status == "admitted"
    assert [event[0] for event in events] == ["policy", "human"]


def test_source_publisher_normal_cycle_polls_once_and_allows_no_new_human() -> None:
    poll_calls = []
    publisher = MujocoPolicyHumanAllocationSourcePublisher(
        human_source=SimpleNamespace(
            step=lambda *, now: (
                poll_calls.append(now) or SimpleNamespace(emission=None)
            )
        ),
        publish_policy_input=lambda _emission, _observed_at: SimpleNamespace(
            accepted=True
        ),
        publish_human_input=lambda *_args: pytest.fail(
            "no human emission must not synthesize an input"
        ),
        clock=lambda: 10.0,
        clock_domain_id="control.monotonic",
    )

    result = publisher(10.0, _emission(), None, True, 10.0)

    assert poll_calls == [10.0]
    assert result.human_submission is None
    assert result.human_step is not None


def test_source_publisher_closes_receiver_cut_after_human_poll() -> None:
    clock_state = {"now": 10.0}
    observed_cuts = []
    human = _human_emission(
        sequence=1,
        emitted_at=10.02,
        arm_x=0.1,
        hand_qpos=0.2,
    )

    def poll_human(*, now: float):
        assert now == pytest.approx(10.0)
        clock_state["now"] = 10.03
        return SimpleNamespace(emission=human)

    publisher = MujocoPolicyHumanAllocationSourcePublisher(
        human_source=SimpleNamespace(step=poll_human),
        publish_policy_input=lambda _emission, observed_at: (
            observed_cuts.append(("policy", observed_at))
            or SimpleNamespace(accepted=True)
        ),
        publish_human_input=lambda _emission, observed_at, _target_at: (
            observed_cuts.append(("human", observed_at))
            or SimpleNamespace(accepted=True)
        ),
        clock=lambda: clock_state["now"],
        clock_domain_id="control.monotonic",
    )

    result = publisher(10.0, _emission(), None, True, 10.04)

    assert result.observed_at == pytest.approx(10.03)
    assert observed_cuts == [
        ("policy", pytest.approx(10.03)),
        ("human", pytest.approx(10.03)),
    ]


def test_source_publisher_prime_requires_both_receiver_admissions() -> None:
    human = _human_emission(
        sequence=0,
        emitted_at=10.0,
        arm_x=0.1,
        hand_qpos=0.2,
    )
    publisher = MujocoPolicyHumanAllocationSourcePublisher(
        human_source=SimpleNamespace(step=lambda **_kwargs: None),
        publish_policy_input=lambda _emission, _observed_at: SimpleNamespace(
            accepted=False
        ),
        publish_human_input=lambda *_args: SimpleNamespace(accepted=True),
        clock=lambda: 10.0,
        clock_domain_id="control.monotonic",
    )

    with pytest.raises(RuntimeError, match="policy activation input"):
        publisher(10.0, _emission(), human, False, 10.0)


def test_source_cut_includes_completed_policy_and_human_admissions() -> None:
    clock_state = {"now": 10.0}
    observed_cuts = []
    human = _human_emission(
        sequence=1,
        emitted_at=10.0,
        arm_x=0.1,
        hand_qpos=0.2,
    )

    def publish_policy(_emission, observed_at):
        observed_cuts.append(("policy", observed_at))
        clock_state["now"] = 10.02
        return SimpleNamespace(accepted=True)

    def publish_human(_emission, observed_at, _target):
        observed_cuts.append(("human", observed_at))
        clock_state["now"] = 10.03
        return SimpleNamespace(accepted=True)

    publisher = MujocoPolicyHumanAllocationSourcePublisher(
        human_source=SimpleNamespace(
            step=lambda **_kwargs: SimpleNamespace(emission=human)
        ),
        publish_policy_input=publish_policy,
        publish_human_input=publish_human,
        clock=lambda: clock_state["now"],
        clock_domain_id="control.monotonic",
    )

    result = publisher(10.0, _emission(), None, True, 10.04)

    assert observed_cuts == [("policy", 10.0), ("human", 10.02)]
    assert result.observed_at == pytest.approx(10.03)
    assert human.emitted_at == pytest.approx(10.0)


def _cycle_port(
    fixture,
    *,
    basis_factory,
    source_calls,
    lower_calls,
    source_observed_at=None,
    promotable_allocation_modes=None,
):
    proposal_publisher = _publisher(
        fixture,
        promotable_allocation_modes=promotable_allocation_modes,
    )

    def source_publisher(
        now,
        policy_emission,
        human_emission,
        poll_human,
        target_effective_at,
    ):
        source_calls.append(
            (
                now,
                policy_emission,
                human_emission,
                poll_human,
                target_effective_at,
            )
        )
        return MujocoAllocationSourcePublication(
            observed_at=(now if source_observed_at is None else source_observed_at),
            observed_clock_domain_id="control.monotonic",
            human_submission=SimpleNamespace(role="operator_relative"),
            policy_submission=SimpleNamespace(role="base_absolute"),
        )

    def build_binding(activation_scope):
        return MujocoAllocationControlCycleBinding(
            activation_scope=activation_scope,
            runtime=fixture.runtime,
            allocation_publication_port=fixture.port,
            proposal_publication_controller=proposal_publisher,
            source_publisher=source_publisher,
            basis_factory=basis_factory,
        )

    fixture.runtime.step = lambda **kwargs: (
        lower_calls.append(kwargs) or AuthoritativeControlStep(status="not_due")
    )
    return AllocationProposalExecutionControlCyclePort(
        port_id="virtual-mujoco-allocation-cycle",
        binding_factory=build_binding,
    )


def _activate_and_prime(port, fixture, *, now: float) -> None:
    port.activate(activation_scope=fixture.activation)
    port.prime_activation(
        now=now,
        policy_emission=None,
        human_emission=SimpleNamespace(emission_id="human:prime"),
        poll_human=True,
        target_effective_at=now,
    )


def test_missing_context_is_no_action_but_lower_owner_still_runs() -> None:
    fixture = _fixture(enable_composer_authority=True)
    source_calls = []
    lower_calls = []
    port = _cycle_port(
        fixture,
        basis_factory=lambda _now, _target: None,
        source_calls=source_calls,
        lower_calls=lower_calls,
    )
    try:
        _activate_and_prime(port, fixture, now=10.01)
        source_calls.clear()
        fixture.clock.value = 10.02

        cycle = port.run_cycle(
            now=10.02,
            policy_emission=None,
            poll_human=True,
            target_effective_at=10.02,
        )

        assert cycle.composition_step.status == "no_action"
        assert cycle.execution.status == "not_due"
        assert source_calls == [(10.02, None, None, True, 10.02)]
        assert lower_calls == [{"now": 10.02, "physics_ticks_after_dispatch": 0}]
        assert len(fixture.runtime.core.proposals) == 0
        assert fixture.scene.physics_command_count == 0
    finally:
        port.close()
        fixture.runtime.close()


def test_admitted_context_publishes_exact_proposal_then_one_lower_step() -> None:
    fixture = _fixture(enable_composer_authority=True)
    source_calls = []
    lower_calls = []
    port = _cycle_port(
        fixture,
        basis_factory=lambda _now, _target: None,
        source_calls=source_calls,
        lower_calls=lower_calls,
    )
    try:
        _activate_and_prime(port, fixture, now=10.01)
        offered = fixture.port.offer_basis(lambda: _basis(fixture))
        assert offered.worker_notified
        assert fixture.port.wait_for_outcomes(2, timeout_sec=2.0)
        assert fixture.port.snapshot().latest_outcome.outcome_kind == (
            ALLOCATION_PUBLICATION_OUTCOME_ACCEPTED_NEW
        )
        fixture.clock.value = 10.04

        cycle = port.run_cycle(
            now=10.04,
            policy_emission=None,
            poll_human=True,
            target_effective_at=10.05,
        )

        proposal_cycle = cycle.composition_step
        assert proposal_cycle.status == "proposal_admitted"
        assert proposal_cycle.proposal is not None
        assert proposal_cycle.proposal.action.semantic_content_hash == (
            proposal_cycle.conditional_action.semantic_content_hash
        )
        assert lower_calls == [{"now": 10.04, "physics_ticks_after_dispatch": 0}]
        assert len(fixture.runtime.core.proposals) == 1
        assert fixture.scene.physics_command_count == 0
    finally:
        port.close()
        fixture.runtime.close()


def test_post_source_observation_closes_basis_proposal_and_execution_cut() -> None:
    fixture = _fixture(enable_composer_authority=True)
    source_calls = []
    lower_calls = []
    basis_calls = []

    def capture_basis(cutoff_at, target_effective_at):
        basis_calls.append((cutoff_at, target_effective_at))
        return None

    port = _cycle_port(
        fixture,
        basis_factory=capture_basis,
        source_calls=source_calls,
        lower_calls=lower_calls,
        source_observed_at=10.03,
    )
    try:
        _activate_and_prime(port, fixture, now=10.01)
        source_calls.clear()
        basis_calls.clear()
        fixture.clock.value = 10.03

        cycle = port.run_cycle(
            now=10.02,
            policy_emission=None,
            poll_human=True,
            target_effective_at=10.05,
        )

        assert cycle.publication.observed_at == pytest.approx(10.03)
        assert cycle.composition_step.control_cut.cutoff_at == pytest.approx(10.03)
        assert basis_calls == [(10.03, 10.05)]
        assert lower_calls == [{"now": 10.03, "physics_ticks_after_dispatch": 0}]
    finally:
        port.close()
        fixture.runtime.close()


def test_proposal_cut_includes_time_spent_capturing_allocation_basis() -> None:
    fixture = _fixture(enable_composer_authority=True)
    source_calls = []
    lower_calls = []

    def capture_basis(_cutoff, _target) -> None:
        fixture.clock.value = 10.07

    port = _cycle_port(
        fixture,
        basis_factory=capture_basis,
        source_calls=source_calls,
        lower_calls=lower_calls,
    )
    try:
        _activate_and_prime(port, fixture, now=10.01)
        fixture.clock.value = 10.03
        cycle = port.run_cycle(
            now=10.03,
            policy_emission=None,
            poll_human=True,
            target_effective_at=10.15,
        )

        assert cycle.publication.observed_at == pytest.approx(10.07)
        assert cycle.composition_step.control_cut.cutoff_at == pytest.approx(10.07)
        assert (
            cycle.composition_step.submission.composition_evaluation.target_effective_at
            == pytest.approx(10.15)
        )
        assert lower_calls == [{"now": 10.07, "physics_ticks_after_dispatch": 0}]
        assert len(fixture.runtime.core.proposals) == 0
    finally:
        port.close()
        fixture.runtime.close()


def test_absolute_mode_blocks_proposal_but_still_advances_lower_owner() -> None:
    fixture = _fixture(enable_composer_authority=True)
    source_calls = []
    lower_calls = []
    port = _cycle_port(
        fixture,
        basis_factory=lambda _now, _target: None,
        source_calls=source_calls,
        lower_calls=lower_calls,
        promotable_allocation_modes=(
            COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
            COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY,
        ),
    )
    try:
        _activate_and_prime(port, fixture, now=10.01)
        offered = fixture.port.offer_basis(lambda: _basis(fixture, level=3.5))
        assert offered.worker_notified
        assert fixture.port.wait_for_outcomes(2, timeout_sec=2.0)
        fixture.clock.value = 10.04

        cycle = port.run_cycle(
            now=10.04,
            policy_emission=None,
            poll_human=True,
            target_effective_at=10.05,
        )

        assert cycle.composition_step.status == "promotion_blocked"
        assert cycle.composition_step.proposal is None
        assert set(
            cycle.composition_step.promotion_blocked_mode_by_component.values()
        ) == {"absolute_overlay_owner"}
        assert lower_calls == [{"now": 10.04, "physics_ticks_after_dispatch": 0}]
        assert len(fixture.runtime.core.proposals) == 0
        assert fixture.scene.physics_command_count == 0
    finally:
        port.close()
        fixture.runtime.close()
