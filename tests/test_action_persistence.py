from __future__ import annotations

from types import SimpleNamespace

from robot_motion.action.persistence import (
    ActionPersistencePolicy,
    ActiveExecutionSegment,
    ActiveExecutionSnapshot,
    build_execution_directive,
)


def policy():
    return ActionPersistencePolicy(
        policy_id="segments-v1",
        selected_segment_max_lifetime_sec=0.1,
    )


def result(
    *,
    output="no_action",
    statuses=None,
    selected=None,
    target=10.0,
    supported_until=None,
):
    return SimpleNamespace(
        result_id=f"result-{output}",
        output_status=output,
        status_by_group={} if statuses is None else statuses,
        selected_by_group={} if selected is None else selected,
        target_effective_at=target,
        supported_until=(
            target + 0.2
            if supported_until is None
            else supported_until
        ),
    )


def active(*, expires_at=10.0):
    return ActiveExecutionSnapshot(
        control_domain_id="robot-0",
        execution_epoch=3,
        active_segment_by_group={
            "arms": ActiveExecutionSegment("arms", "segment-0", "proposal-old", expires_at, 7)
        },
    )


def test_no_action_never_refreshes_existing_segment_expiry() -> None:
    directive = build_execution_directive(
        result(statuses={"arms": "no_action"}),
        active_execution=active(expires_at=10.0),
        policy=policy(),
        now=9.0,
        supports_local_group_stop=True,
    )

    group = directive.directive_by_selection_group["arms"]
    assert group.directive == "keep_existing_until_expiry"
    assert group.segment_expiry == 10.0
    assert group.group_segment_generation == 7


def test_expired_segment_cancels_instead_of_implicit_infinite_hold() -> None:
    directive = build_execution_directive(
        result(statuses={"arms": "no_action"}),
        active_execution=active(expires_at=8.0),
        policy=policy(),
        now=9.0,
        supports_local_group_stop=True,
    )

    assert directive.directive_by_selection_group["arms"].directive == "cancel_group_and_hold"
    assert directive.directive_by_selection_group["arms"].group_segment_generation == 8


def test_selected_group_publishes_new_bounded_segment() -> None:
    directive = build_execution_directive(
        result(
            output="selected",
            statuses={"arms": "selected", "hand": "no_action"},
            selected={"arms": "proposal-new"},
            target=10.0,
        ),
        active_execution=active(expires_at=12.0),
        policy=policy(),
        now=10.0,
        supports_local_group_stop=True,
    )

    arms = directive.directive_by_selection_group["arms"]
    assert arms.directive == "publish_selected"
    assert arms.selected_action_ref == "proposal-new"
    assert arms.segment_expiry == 10.1


def test_selected_segment_cannot_outlive_arbitration_support() -> None:
    directive = build_execution_directive(
        result(
            output="selected",
            statuses={"arms": "selected"},
            selected={"arms": "proposal-new"},
            target=10.0,
            supported_until=10.04,
        ),
        active_execution=active(expires_at=12.0),
        policy=policy(),
        now=10.0,
        supports_local_group_stop=True,
    )

    assert (
        directive.directive_by_selection_group["arms"].segment_expiry
        == 10.04
    )


def test_global_hold_requires_epoch_cancellation_and_backend_hold() -> None:
    directive = build_execution_directive(
        result(output="hold", statuses={"arms": "hold"}),
        active_execution=active(),
        policy=policy(),
        now=9.0,
        supports_local_group_stop=False,
    )

    assert directive.global_status == "hold_all"
    assert directive.requires_execution_epoch_increment
    assert directive.requires_backend_hold


def test_existing_backend_hold_is_not_relinearized_on_every_tick() -> None:
    directive = build_execution_directive(
        result(output="hold", statuses={"arms": "hold"}),
        active_execution=ActiveExecutionSnapshot(
            control_domain_id="robot-0",
            execution_epoch=4,
            active_segment_by_group={},
            backend_hold_active=True,
        ),
        policy=policy(),
        now=9.0,
        supports_local_group_stop=False,
    )

    assert directive.global_status == "hold_all"
    assert not directive.requires_execution_epoch_increment
    assert not directive.requires_backend_hold
    assert directive.directive_by_selection_group["arms"].directive == (
        "keep_backend_hold"
    )


def test_publication_no_action_fallback_never_publishes_selected_result() -> None:
    directive = build_execution_directive(
        result(
            output="selected",
            statuses={"arms": "selected"},
            selected={"arms": "proposal-new"},
        ),
        active_execution=active(expires_at=12.0),
        policy=policy(),
        now=10.0,
        supports_local_group_stop=False,
        publication_fallback="no_action",
    )

    assert directive.publication_fallback == "no_action"
    assert directive.directive_by_selection_group["arms"].directive == (
        "keep_existing_until_expiry"
    )


def test_adaptation_rejection_uses_typed_fallback_without_republishing() -> None:
    directive = build_execution_directive(
        result(
            output="selected",
            statuses={"arms": "selected"},
            selected={"arms": "proposal-new"},
        ),
        active_execution=active(expires_at=12.0),
        policy=policy(),
        now=10.0,
        supports_local_group_stop=False,
        action_adaptation_receipt_id="adaptation-receipt-1",
        action_adaptation_receipt_hash="adaptation-receipt-hash-1",
        action_adaptation_fallback="no_action",
    )

    assert directive.publication_fallback is None
    assert directive.action_adaptation_fallback == "no_action"
    assert directive.directive_by_selection_group["arms"].directive == (
        "keep_existing_until_expiry"
    )
    assert directive.directive_by_selection_group[
        "arms"
    ].selected_action_ref is None
