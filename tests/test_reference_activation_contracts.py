from __future__ import annotations

from dataclasses import replace

import pytest

from robot_motion.action import (
    IMMEDIATE_QREF_SIM_V1,
    SCHEDULED_REFERENCE_SEGMENT_SIM_V1,
    DecisionScheduleSpec,
    build_immediate_qref_activation_spec,
    build_scheduled_reference_segment_activation_spec,
    validate_reference_activation_binding,
)
from robot_motion.position_control import ExecutionCapabilityProfile


def capability(*, reference_hz: float = 60.0) -> ExecutionCapabilityProfile:
    return ExecutionCapabilityProfile(
        profile_id="scheduled-reference-fixture-v1",
        validated_max_selected_keyframe_hz=60.0,
        configured_selected_keyframe_hz=30.0,
        validated_max_coupled_ik_hz=None,
        configured_coupled_ik_hz=None,
        reference_resample_hz=reference_hz,
        command_hz=120.0,
        backend_servo_hz=1000.0,
        validated_action_spaces=("bimanual_qpos_v1",),
        validated_selection_groups=("arms",),
        supports_local_group_stop=False,
        acceptance_evidence_id="deterministic-reference-activation-fixture-v1",
        decision_runtime_budget_sec=0.005,
        ik_runtime_budget_sec=1.0 / 60.0,
        end_to_end_control_budget_sec=1.0 / 30.0,
        required_cancellation_ack_levels=("software_queue",),
    )


def test_immediate_profile_preserves_zero_lead_without_future_segment() -> None:
    schedule = DecisionScheduleSpec(
        spec_id="immediate-30hz-v1",
        decision_hz=30.0,
        target_lead_time_sec=0.0,
    )
    spec = build_immediate_qref_activation_spec(schedule, capability())

    assert spec.spec_id == IMMEDIATE_QREF_SIM_V1
    assert spec.activation_mode == "immediate_target"
    assert spec.required_admission_deadline == "not_applicable"
    assert spec.anchor_commit_policy == "receiver_acceptance"
    validate_reference_activation_binding(
        spec,
        schedule=schedule,
        capability=capability(),
    )


def test_scheduled_profile_uses_keyframe_period_plus_reference_margin() -> None:
    lead = 1.0 / 30.0 + 1.0 / 60.0
    schedule = DecisionScheduleSpec(
        spec_id="scheduled-30-60-v1",
        decision_hz=30.0,
        target_lead_time_sec=lead,
    )
    spec = build_scheduled_reference_segment_activation_spec(
        schedule,
        capability(),
        minimum_activation_margin_sec=1.0 / 60.0,
        activation_margin_evidence_id="one-reference-period-fixture-v1",
    )

    assert spec.spec_id == SCHEDULED_REFERENCE_SEGMENT_SIM_V1
    assert spec.activation_mode == "scheduled_segment"
    assert spec.target_lead_time_sec == pytest.approx(0.05)
    assert spec.required_admission_deadline == "before_segment_start"
    assert spec.late_admission_policy == "reject_and_hold"
    assert spec.anchor_commit_policy == "endpoint_effective_time"


def test_scheduled_profile_rejects_insufficient_lead_or_margin() -> None:
    too_short = DecisionScheduleSpec(
        spec_id="scheduled-too-short-v1",
        decision_hz=30.0,
        target_lead_time_sec=0.04,
    )
    with pytest.raises(ValueError, match="below keyframe period plus margin"):
        build_scheduled_reference_segment_activation_spec(
            too_short,
            capability(),
            minimum_activation_margin_sec=1.0 / 60.0,
            activation_margin_evidence_id="insufficient-lead-fixture",
        )

    enough_lead = replace(too_short, target_lead_time_sec=0.05)
    with pytest.raises(ValueError, match="at least one reference period"):
        build_scheduled_reference_segment_activation_spec(
            enough_lead,
            capability(),
            minimum_activation_margin_sec=0.01,
            activation_margin_evidence_id="insufficient-margin-fixture",
        )


def test_activation_binding_is_exact_to_schedule_and_capability() -> None:
    schedule = DecisionScheduleSpec(
        spec_id="scheduled-binding-v1",
        decision_hz=30.0,
        target_lead_time_sec=0.05,
    )
    actual_capability = capability()
    spec = build_scheduled_reference_segment_activation_spec(
        schedule,
        actual_capability,
        minimum_activation_margin_sec=1.0 / 60.0,
        activation_margin_evidence_id="binding-fixture-v1",
    )

    with pytest.raises(ValueError, match="schedule binding mismatch"):
        validate_reference_activation_binding(
            spec,
            schedule=replace(schedule, spec_id="another-schedule"),
            capability=actual_capability,
        )
    with pytest.raises(ValueError, match="capability binding mismatch"):
        validate_reference_activation_binding(
            spec,
            schedule=schedule,
            capability=replace(
                actual_capability,
                acceptance_evidence_id="another-capability-evidence",
            ),
        )
