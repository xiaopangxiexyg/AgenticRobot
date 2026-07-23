from __future__ import annotations

import pytest

from robot_motion.position_control.capability import ExecutionCapabilityProfile


def profile(**overrides):
    values = {
        "profile_id": "rm65-position-evidence-v1",
        "validated_max_selected_keyframe_hz": 60.0,
        "configured_selected_keyframe_hz": 30.0,
        "validated_max_coupled_ik_hz": 60.0,
        "configured_coupled_ik_hz": 60.0,
        "reference_resample_hz": 75.0,
        "command_hz": 150.0,
        "backend_servo_hz": 150.0,
        "validated_action_spaces": ("bimanual_ee_pose_v1",),
        "validated_selection_groups": ("arms",),
        "supports_local_group_stop": False,
        "acceptance_evidence_id": "synthetic-only-m1b",
        "decision_runtime_budget_sec": 0.005,
        "ik_runtime_budget_sec": 0.012,
        "end_to_end_control_budget_sec": 0.02,
    }
    values.update(overrides)
    return ExecutionCapabilityProfile(**values)


def test_capability_keeps_decision_ik_reference_and_command_rates_independent() -> None:
    value = profile()
    assert value.configured_selected_keyframe_hz == 30.0
    assert value.configured_coupled_ik_hz == 60.0
    assert value.reference_resample_hz == 75.0
    assert value.command_hz == 150.0


def test_capability_rejects_unvalidated_selected_or_ik_rate() -> None:
    with pytest.raises(ValueError, match="selected-keyframe"):
        profile(configured_selected_keyframe_hz=61.0)
    with pytest.raises(ValueError, match="coupled-IK"):
        profile(configured_coupled_ik_hz=61.0)
