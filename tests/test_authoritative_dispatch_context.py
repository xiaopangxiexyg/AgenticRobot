from __future__ import annotations

import numpy as np
import pytest

from robot_motion.action import (
    HandActionSpec,
    RobotAction,
    build_authoritative_command_dispatch_context,
    build_authoritative_receiver_state_snapshot,
    build_compound_dispatch_manifest,
    validate_authoritative_receiver_context,
    validate_compound_dispatch_manifest,
)


def _action() -> RobotAction:
    return RobotAction.bimanual_qpos(
        left_arm=np.full(6, 0.2, dtype=np.float32),
        right_arm=np.full(6, -0.2, dtype=np.float32),
        source="dispatch-context-fixture",
    )


def _context():
    return build_authoritative_command_dispatch_context(
        runtime_session_id="session-0",
        control_domain_id="domain-0",
        embodiment_instance_id="embodiment-0",
        decision_id="decision-0",
        directive_id="directive-0",
        safety_result_id="safety-0",
        command_id="command-0",
        dispatch_permit_id="permit-0",
        owner_token_id="token-0",
        owner_generation=1,
        target_effective_at=10.1,
        authority_epoch=2,
        execution_epoch=3,
        group_segment_generation_by_group={"arms": 4},
        selected_action_ref_by_group={"arms": "proposal-0"},
        safe_action=_action(),
    )


def test_authoritative_dispatch_context_is_deterministic_and_immutable() -> None:
    first = _context()
    second = _context()

    assert first.semantic_content_hash == second.semantic_content_hash
    assert first.safe_action is not second.safe_action
    with pytest.raises(TypeError):
        first.group_segment_generation_by_group["arms"] = 5
    with pytest.raises(ValueError, match="read-only"):
        first.safe_action.arm_qpos_by_side["left"][0] = 99.0
    assert second.safe_action.arm_qpos_by_side["left"][0] == pytest.approx(0.2)


def test_authoritative_dispatch_context_requires_matching_group_scope() -> None:
    with pytest.raises(ValueError, match="same groups"):
        build_authoritative_command_dispatch_context(
            runtime_session_id="session-0",
            control_domain_id="domain-0",
            embodiment_instance_id="embodiment-0",
            decision_id="decision-0",
            directive_id="directive-0",
            safety_result_id="safety-0",
            command_id="command-0",
            dispatch_permit_id="permit-0",
            owner_token_id="token-0",
            owner_generation=1,
            target_effective_at=10.1,
            authority_epoch=2,
            execution_epoch=3,
            group_segment_generation_by_group={"arms": 4},
            selected_action_ref_by_group={"hands": "proposal-0"},
            safe_action=_action(),
        )


def test_receiver_validation_distinguishes_supplied_context_from_live_state() -> None:
    context = _context()
    live = build_authoritative_receiver_state_snapshot(
        runtime_session_id="session-0",
        control_domain_id="domain-0",
        current_owner_token_id="token-0",
        current_owner_generation=1,
        current_authority_epoch=2,
        current_execution_epoch=3,
        expected_next_group_generation_by_group={"arms": 4},
        captured_at=10.01,
    )
    assert validate_authoritative_receiver_context(context, live).accepted

    stale = build_authoritative_receiver_state_snapshot(
        runtime_session_id="session-0",
        control_domain_id="domain-0",
        current_owner_token_id="token-0",
        current_owner_generation=1,
        current_authority_epoch=2,
        current_execution_epoch=4,
        expected_next_group_generation_by_group={"arms": 4},
        captured_at=10.02,
    )
    receipt = validate_authoritative_receiver_context(context, stale)
    assert not receipt.accepted
    assert receipt.status == "execution_epoch_mismatch"


def test_compound_dispatch_manifest_freezes_exact_group_join() -> None:
    hand_specs = {
        side: HandActionSpec.dmtac_gripper(side)
        for side in ("left", "right")
    }
    safe_action = RobotAction.bimanual_qpos(
        left_arm=np.full(6, 0.2, dtype=np.float32),
        right_arm=np.full(6, -0.2, dtype=np.float32),
        left_hand=np.asarray([0.25, 0, 0, 0, 0, 0], dtype=np.float32),
        right_hand=np.asarray([0.75, 0, 0, 0, 0, 0], dtype=np.float32),
        source="compound-dispatch-fixture",
        metadata={
            "hand_action_specs_by_side": {
                side: spec.to_metadata() for side, spec in hand_specs.items()
            }
        },
    )
    context = build_authoritative_command_dispatch_context(
        runtime_session_id="session-0",
        control_domain_id="domain-0",
        embodiment_instance_id="embodiment-0",
        decision_id="decision-0",
        directive_id="directive-0",
        safety_result_id="safety-0",
        command_id="command-0",
        dispatch_permit_id="permit-0",
        owner_token_id="token-0",
        owner_generation=1,
        target_effective_at=10.1,
        authority_epoch=2,
        execution_epoch=3,
        group_segment_generation_by_group={
            "arms": 4,
            "left_hand": 8,
            "right_hand": 6,
        },
        selected_action_ref_by_group={
            "arms": "arm-proposal",
            "left_hand": "left-hand-proposal",
            "right_hand": "right-hand-proposal",
        },
        safe_action=safe_action,
    )

    manifest = build_compound_dispatch_manifest(
        context,
        component_dependencies_by_group={
            "arms": ("left_arm", "right_arm"),
            "left_hand": ("left_hand",),
            "right_hand": ("right_hand",),
        },
    )

    validate_compound_dispatch_manifest(manifest, context)
    assert manifest.component_dependencies == (
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
    )
    assert dict(manifest.group_segment_generation_by_group) == {
        "arms": 4,
        "left_hand": 8,
        "right_hand": 6,
    }
    with pytest.raises(ValueError, match="exactly cover"):
        build_compound_dispatch_manifest(
            context,
            component_dependencies_by_group={
                "arms": ("left_arm", "right_arm"),
                "left_hand": ("left_hand",),
            },
        )
