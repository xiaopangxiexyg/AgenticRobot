from __future__ import annotations

import pytest

from control_composition.allocation import (
    ALLOCATION_PUBLICATION_OUTCOME_ACCEPTED_NEW,
    COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
    COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY,
)
from control_station.composition import (
    MujocoAllocationProposalPublicationController,
    MujocoAllocationPublicationProfile,
)
from tests.test_control_station_mujoco_allocation_shadow import _basis, _fixture


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _publisher(
    fixture,
    *,
    promotable_allocation_modes=None,
) -> MujocoAllocationProposalPublicationController:
    profile_kwargs = {}
    if promotable_allocation_modes is not None:
        profile_kwargs["promotable_allocation_modes"] = tuple(
            promotable_allocation_modes
        )
    return MujocoAllocationProposalPublicationController(
        profile=MujocoAllocationPublicationProfile(
            profile_id="mujoco-allocation-proposal-publication",
            enabled=True,
            send_actions=True,
            **profile_kwargs,
        ),
        runtime=fixture.runtime,
        composition_spec=fixture.composition_spec,
        allocation_spec=fixture.allocation_spec,
        activation_scope=fixture.activation,
        allocation_publication_port=fixture.port,
        publication_barrier=fixture.barrier,
    )


def test_default_off_publication_is_lazy_and_unbound() -> None:
    publisher = MujocoAllocationProposalPublicationController(
        profile=MujocoAllocationPublicationProfile()
    )

    cycle = publisher.publish_at(target_effective_at=10.0)

    assert cycle.status == "disabled"
    assert cycle.submission is None
    assert cycle.proposal is None
    assert cycle.authoritative is False
    assert cycle.executed is False


def test_missing_allocation_context_returns_no_action_without_proposal() -> None:
    fixture = _fixture(enable_composer_authority=True)
    try:
        fixture.controller.start()
        publisher = _publisher(fixture)
        fixture.clock.value = 10.02

        cycle = publisher.publish_at(
            target_effective_at=10.02,
            cutoff_at=10.02,
        )

        assert cycle.status == "no_action"
        assert cycle.selected_context_id is None
        assert cycle.conditional_action is None
        assert cycle.submission.proposal_admission is None
        assert len(fixture.runtime.core.proposals) == 0
        assert fixture.scene.physics_command_count == 0
    finally:
        fixture.controller.close()
        fixture.runtime.close()


def test_same_cut_ready_result_admits_exact_proposal_without_step() -> None:
    fixture = _fixture(enable_composer_authority=True)
    try:
        fixture.controller.start()
        offered = fixture.controller.offer_basis(lambda: _basis(fixture))
        assert offered.worker_notified
        assert fixture.port.wait_for_outcomes(2, timeout_sec=2.0)
        assert fixture.port.snapshot().latest_outcome.outcome_kind == (
            ALLOCATION_PUBLICATION_OUTCOME_ACCEPTED_NEW
        )
        publisher = _publisher(fixture)
        fixture.clock.value = 10.04

        cycle = publisher.publish_at(
            target_effective_at=10.05,
            cutoff_at=10.04,
        )

        assert cycle.status == "proposal_admitted"
        assert cycle.selected_mode_by_component == {
            component_id: "relative_overlay"
            for component_id in fixture.allocation_spec.component_ids
        }
        assert cycle.proposal is not None
        assert cycle.submission.proposal_admission.accepted
        assert cycle.proposal.composition_evaluation_id == (
            cycle.submission.composition_evaluation.composition_evaluation_id
        )
        assert cycle.proposal.composition_evaluation_hash == (
            cycle.submission.composition_evaluation.semantic_content_hash
        )
        assert cycle.proposal.action.semantic_content_hash == (
            cycle.conditional_action.semantic_content_hash
        )
        assert cycle.proposal.target_effective_at == pytest.approx(10.05)
        assert cycle.selected_context_id in cycle.proposal.context_ids
        assert len(fixture.runtime.core.proposals) == 1
        assert fixture.scene.physics_command_count == 0
        assert cycle.authoritative is False
        assert cycle.executed is False
    finally:
        fixture.controller.close()
        fixture.runtime.close()


def test_expired_support_is_reported_without_interrupting_lower_runtime(
    monkeypatch,
) -> None:
    fixture = _fixture(enable_composer_authority=True)
    try:
        fixture.controller.start()
        offered = fixture.controller.offer_basis(lambda: _basis(fixture))
        assert offered.worker_notified
        assert fixture.port.wait_for_outcomes(2, timeout_sec=2.0)
        publisher = _publisher(fixture)
        original_submit = fixture.runtime.submit_evaluated_composition

        def submit_after_support_deadline(evaluation, **kwargs):
            fixture.clock.value = 10.3
            return original_submit(evaluation, **kwargs)

        monkeypatch.setattr(
            fixture.runtime,
            "submit_evaluated_composition",
            submit_after_support_deadline,
        )
        fixture.clock.value = 10.04

        cycle = publisher.publish_at(
            target_effective_at=10.05,
            cutoff_at=10.04,
        )

        assert cycle.status == "proposal_not_admitted"
        assert cycle.submission.proposal_admission is not None
        assert not cycle.submission.proposal_admission.accepted
        assert cycle.submission.proposal_admission.rejection_code == (
            "composition_support_expired_before_proposal_publication"
        )
        assert len(fixture.runtime.core.proposals) == 0

        # The one lower execution owner still runs its ordinary no-proposal
        # path; publication timing does not become an application fault.
        lower = fixture.runtime.step(now=fixture.clock.value)
        assert lower.status != "dispatched"
        assert fixture.scene.physics_command_count == 0
    finally:
        fixture.controller.close()
        fixture.runtime.close()


def test_absolute_mode_remains_observable_but_is_not_promoted() -> None:
    fixture = _fixture(enable_composer_authority=True)
    try:
        fixture.controller.start()
        offered = fixture.controller.offer_basis(lambda: _basis(fixture, level=3.5))
        assert offered.worker_notified
        assert fixture.port.wait_for_outcomes(2, timeout_sec=2.0)
        publisher = _publisher(
            fixture,
            promotable_allocation_modes=(
                COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
                COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY,
            ),
        )
        fixture.clock.value = 10.04

        cycle = publisher.publish_at(
            target_effective_at=10.05,
            cutoff_at=10.04,
        )

        assert cycle.status == "promotion_blocked"
        assert cycle.submission.composition_evaluation.output_status == "ready"
        assert cycle.proposal is None
        assert cycle.promotion_blocked_mode_by_component == (
            cycle.selected_mode_by_component
        )
        assert set(cycle.selected_mode_by_component.values()) == {
            "absolute_overlay_owner"
        }
        assert len(fixture.runtime.core.proposals) == 0
        assert fixture.scene.physics_command_count == 0
    finally:
        fixture.controller.close()
        fixture.runtime.close()


def test_publication_profile_rejects_invalid_promotable_modes() -> None:
    with pytest.raises(ValueError, match="promotable_allocation_modes"):
        MujocoAllocationPublicationProfile(
            promotable_allocation_modes=("base_only", "base_only"),
        )
    with pytest.raises(ValueError, match="promotable_allocation_modes"):
        MujocoAllocationPublicationProfile(
            promotable_allocation_modes=("future_mode",),
        )


def test_disabled_station_barrier_blocks_before_cut_capture(monkeypatch) -> None:
    fixture = _fixture(enable_composer_authority=True)
    calls = []
    try:
        fixture.controller.start()
        publisher = _publisher(fixture)
        fixture.barrier.disable(reason="virtual-session-release")

        def forbidden_capture(*, cutoff_at=None):
            calls.append(cutoff_at)
            raise AssertionError("disabled barrier captured a lower cut")

        monkeypatch.setattr(
            fixture.runtime,
            "capture_control_cut",
            forbidden_capture,
        )
        with pytest.raises(RuntimeError, match="barrier is disabled"):
            publisher.publish_at(
                target_effective_at=10.05,
                cutoff_at=10.04,
            )
        assert calls == []
        assert len(fixture.runtime.core.proposals) == 0
        assert fixture.scene.physics_command_count == 0
    finally:
        fixture.controller.close()
        fixture.runtime.close()
