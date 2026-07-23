from __future__ import annotations

import numpy as np
import pytest

from robot_motion.action import (
    RobotAction,
    SafetyAnchorPolicy,
    admit_resolved_execution_target,
    build_resolved_execution_target,
    commit_safe_action_anchor,
    execution_target_anchor_commit_evidence,
    make_uninitialized_safety_anchor_state,
    rebase_safety_anchor_from_measured,
    CanonicalActionSafetyAnchor,
)


def safe_action(value: float = 0.2) -> RobotAction:
    return RobotAction.bimanual_qpos(
        left_arm=np.full(6, value),
        right_arm=np.full(6, -value),
        source="shared-safety-fixture",
    )


def target(*, kind: str = "qpos_reference", payload=None):
    return build_resolved_execution_target(
        control_domain_id="robot-0",
        selection_group_id="arms",
        execution_epoch=2,
        group_segment_generation=4,
        source_safety_result_id="shared-safety:4",
        safe_action=safe_action(),
        target_kind=kind,
        component_dependencies=("left_arm", "right_arm"),
        resolved_at=10.02,
        scheduled_from=10.05,
        scheduled_until=10.15,
        clock_domain_id="local_monotonic",
        input_action_schema_hash="canonical-qpos-action-v1",
        output_target_schema_hash=f"{kind}-schema-v1",
        resolver_contract_id="fixture-resolver-v1",
        resolver_contract_hash="fixture-resolver-hash",
        planner_contract_hash="fixture-planner-hash",
        payload={
            "qpos_by_side": {
                "left": np.full(6, 0.2),
                "right": np.full(6, -0.2),
            }
        }
        if payload is None
        else payload,
    )


def admission(item, **overrides):
    values = {
        "receiver_id": "position-reference-worker:0",
        "admitted_at": 10.03,
        "current_execution_epoch": 2,
        "expected_group_segment_generation": 4,
    }
    values.update(overrides)
    return admit_resolved_execution_target(item, **values)


def active_anchor_state():
    policy = SafetyAnchorPolicy(
        policy_id="canonical-arm-anchor-v1",
        max_measurement_age_sec=0.1,
    )
    initial = make_uninitialized_safety_anchor_state(
        control_domain_id="robot-0",
        selection_group_id="arms",
        execution_epoch=2,
        group_segment_generation=3,
        policy=policy,
    )
    measured = CanonicalActionSafetyAnchor(
        anchor_id="measurement:3",
        anchor_kind="measured_state",
        action_space="bimanual_qpos_v1",
        arm_qpos_by_side={"left": np.zeros(6), "right": np.zeros(6)},
        observed_at=10.0,
    )
    return rebase_safety_anchor_from_measured(
        initial,
        measured_anchor=measured,
        measured_evidence_id="measurement:3",
        occurred_at=10.01,
        policy=policy,
    )[0]


def test_resolved_target_is_deterministic_and_deeply_immutable() -> None:
    first = target()
    second = target()

    assert first.target_id == second.target_id
    assert first.semantic_content_hash == second.semantic_content_hash
    assert first.anchor_commit_policy == "on_target_admission"
    assert first.scheduled_from < first.scheduled_until
    with pytest.raises(ValueError):
        first.payload["qpos_by_side"]["left"][0] = 9.0
    with pytest.raises(TypeError):
        first.payload["new"] = "mutable"


@pytest.mark.parametrize(
    ("kind", "expected_policy"),
    (
        ("direct_command_intent", "on_backend_acceptance"),
        ("pose_reference", "on_target_admission"),
        ("qpos_reference", "on_target_admission"),
        ("trajectory_segment", "on_target_admission"),
    ),
)
def test_target_kind_freezes_anchor_commit_linearization(
    kind: str,
    expected_policy: str,
) -> None:
    assert target(kind=kind).anchor_commit_policy == expected_policy


def test_receiver_admission_fails_closed_on_scope_generation_and_expiry() -> None:
    item = target()
    assert admission(item).accepted

    stale_epoch = admission(item, current_execution_epoch=3)
    assert not stale_epoch.accepted
    assert stale_epoch.rejection_reason == "execution_epoch_mismatch"

    stale_generation = admission(item, expected_group_segment_generation=5)
    assert not stale_generation.accepted
    assert stale_generation.rejection_reason == "group_segment_generation_mismatch"

    expired = admission(item, admitted_at=item.scheduled_until)
    assert not expired.accepted
    assert expired.rejection_reason == "target_interval_expired"

    rejected = admission(
        item,
        receiver_accepts=False,
        rejection_reason="planner_not_ready",
    )
    assert not rejected.accepted
    assert rejected.rejection_reason == "planner_not_ready"


def test_accepted_reference_target_separates_evidence_availability_and_effective_time() -> None:
    item = target()
    receipt = admission(item)
    evidence = execution_target_anchor_commit_evidence(
        item,
        receipt,
        safety_result_id="shared-safety:4",
    )

    committed, transition = commit_safe_action_anchor(
        active_anchor_state(),
        safe_action=item.group_safe_action,
        evidence=evidence,
    )

    assert evidence.accepted
    assert evidence.target_id == item.target_id
    assert evidence.occurred_at == receipt.admitted_at
    assert evidence.effective_from == item.scheduled_from
    assert committed.group_segment_generation == 4
    assert committed.available_at == receipt.admitted_at
    assert committed.committed_at == item.scheduled_from
    assert committed.anchor is not None
    assert committed.anchor.anchor_kind == "previous_safe_committed"
    assert transition.causal_evidence_ids == (
        "shared-safety:4",
        receipt.admission_id,
    )


def test_rejected_target_does_not_commit_and_direct_intent_needs_backend_receipt() -> None:
    item = target()
    rejected = admission(
        item,
        receiver_accepts=False,
        rejection_reason="reference_worker_rejected",
    )
    rejected_evidence = execution_target_anchor_commit_evidence(
        item,
        rejected,
        safety_result_id="shared-safety:4",
    )
    with pytest.raises(ValueError, match="Rejected execution-target"):
        commit_safe_action_anchor(
            active_anchor_state(),
            safe_action=item.group_safe_action,
            evidence=rejected_evidence,
        )

    direct = target(kind="direct_command_intent", payload={"command": "fixture"})
    with pytest.raises(ValueError, match="backend-acceptance"):
        execution_target_anchor_commit_evidence(
            direct,
            admission(direct),
            safety_result_id="shared-safety:4",
        )


def test_target_lineage_rejects_another_safety_result() -> None:
    item = target()
    receipt = admission(item)
    with pytest.raises(ValueError, match="another safety result"):
        execution_target_anchor_commit_evidence(
            item,
            receipt,
            safety_result_id="shared-safety:other",
        )


def test_target_projects_only_the_selection_group_components() -> None:
    item = build_resolved_execution_target(
        control_domain_id="robot-0",
        selection_group_id="left-arm",
        execution_epoch=2,
        group_segment_generation=4,
        source_safety_result_id="shared-safety:4",
        safe_action=safe_action(),
        target_kind="qpos_reference",
        component_dependencies=("left_arm",),
        resolved_at=10.02,
        scheduled_from=10.05,
        scheduled_until=10.15,
        clock_domain_id="local_monotonic",
        input_action_schema_hash="canonical-qpos-action-v1",
        output_target_schema_hash="qpos-reference-schema-v1",
        resolver_contract_id="fixture-resolver-v1",
        resolver_contract_hash="fixture-resolver-hash",
        payload={"left": np.full(6, 0.2)},
    )

    assert item.group_safe_action.active_effectors == ("left_arm",)
    assert set(item.group_safe_action.arm_qpos_by_side) == {"left"}
    assert item.source_safe_action_hash != item.group_safe_action_hash


def test_source_queue_admission_is_not_part_of_target_contract() -> None:
    fields = set(target().__dataclass_fields__)
    assert "source_queue_index" not in fields
    assert "frontend_submission_id" not in fields
    assert "position_deploy_submit_index" not in fields
