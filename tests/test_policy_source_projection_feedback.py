from __future__ import annotations

from types import SimpleNamespace

import pytest

from control_station.sim_hil_profile.policy_feedback import (
    project_sim_hil_policy_execution_feedback,
)
from robot_motion.deploy import (
    POLICY_EXECUTION_RELATION_COMPOSED,
    POLICY_EXECUTION_RELATION_EXACT,
    POLICY_SOURCE_ACTION_ABSOLUTE_TARGET,
    POLICY_SOURCE_ACTION_OBSERVATION_RELATIVE_CHUNK,
    POLICY_SOURCE_ACTION_SEQUENTIAL_TARGET_DELTA,
    POLICY_SOURCE_ANCHOR_NONE,
    POLICY_SOURCE_ANCHOR_OBSERVATION_TCP_FIXED_FOR_CHUNK,
    POLICY_SOURCE_ANCHOR_PREVIOUS_POLICY_TARGET,
    PolicyExecutionFeedbackApplication,
    PolicySourceActionProjection,
    project_policy_execution_feedback,
)


def _projection(**overrides):
    values = {
        "source_semantics": POLICY_SOURCE_ACTION_ABSOLUTE_TARGET,
        "source_pose_mode": "absolute",
        "source_action_schema_id": "pose7",
        "source_action_by_channel": {"right_arm": (0.0,) * 7},
        "conditioned_action_by_channel": {"right_arm": (0.0,) * 7},
        "decoded_absolute_action_hash": "absolute-action-hash",
        "anchor_policy": POLICY_SOURCE_ANCHOR_NONE,
    }
    values.update(overrides)
    return PolicySourceActionProjection(**values)


def test_policy_source_projection_keeps_three_relative_absolute_semantics_distinct():
    absolute = _projection()
    observation_relative = _projection(
        source_semantics=POLICY_SOURCE_ACTION_OBSERVATION_RELATIVE_CHUNK,
        source_pose_mode="relative",
        anchor_policy=POLICY_SOURCE_ANCHOR_OBSERVATION_TCP_FIXED_FOR_CHUNK,
        anchor_value_by_component={"right_arm": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)},
    )
    sequential = _projection(
        source_semantics=POLICY_SOURCE_ACTION_SEQUENTIAL_TARGET_DELTA,
        source_pose_mode="relative",
        anchor_policy=POLICY_SOURCE_ANCHOR_PREVIOUS_POLICY_TARGET,
        anchor_value_by_component={"right_arm": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)},
        target_generation=3,
    )

    assert absolute.anchor_value_by_component == {}
    assert observation_relative.target_generation is None
    assert sequential.target_generation == 3
    assert PolicySourceActionProjection.from_dict(
        sequential.to_dict()
    ).semantic_content_hash == sequential.semantic_content_hash


def test_policy_source_projection_rejects_implicit_or_crossed_anchors():
    with pytest.raises(ValueError, match="cannot carry an anchor"):
        _projection(
            anchor_policy=POLICY_SOURCE_ANCHOR_PREVIOUS_POLICY_TARGET,
            anchor_value_by_component={"right_arm": (0.0,) * 7},
        )
    with pytest.raises(ValueError, match="fixed chunk anchor"):
        _projection(
            source_semantics=POLICY_SOURCE_ACTION_OBSERVATION_RELATIVE_CHUNK,
            source_pose_mode="relative",
            anchor_policy=POLICY_SOURCE_ANCHOR_PREVIOUS_POLICY_TARGET,
            anchor_value_by_component={"right_arm": (0.0,) * 7},
        )


def _feedback_cycle(*, candidate_hash: str, association_target: str = "admission:1"):
    emission = SimpleNamespace(
        emission_id="emission:old",
        semantic_content_hash="emission-hash:old",
    )
    selected_action = SimpleNamespace(semantic_content_hash="policy-action-hash")
    selected = SimpleNamespace(
        action=selected_action,
        source_fact_ids=(emission.emission_id,),
        source_fact_hashes=(emission.semantic_content_hash,),
    )
    snapshot = SimpleNamespace(selected_input_by_role={"policy": selected})
    evaluation = SimpleNamespace(
        composition_evaluation_id="evaluation:1",
        semantic_content_hash="evaluation-hash:1",
        composed_candidate=SimpleNamespace(semantic_content_hash=candidate_hash),
    )
    composition_step = SimpleNamespace(
        evaluation=SimpleNamespace(
            input_snapshot=snapshot,
            composition_evaluation=evaluation,
        )
    )
    admission = SimpleNamespace(
        receipt_id="admission:1",
        semantic_content_hash="admission-hash:1",
    )
    execution = SimpleNamespace(
        status="dispatched",
        target_dispatch_result=SimpleNamespace(target_admission=admission),
        safety_result=SimpleNamespace(
            status="modified",
            modified_components=("right_arm",),
        ),
    )
    association = SimpleNamespace(
        status="available",
        association_id="association:1",
        semantic_content_hash="association-hash:1",
        receipt_references=(
            SimpleNamespace(
                receipt_id=association_target,
                receipt_hash="admission-hash:1",
            ),
        ),
        component_difference={
            "right_arm": {
                "reference": (0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0),
                "measured": (0.11, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0),
            }
        },
    )
    observer = SimpleNamespace(
        snapshot=lambda: SimpleNamespace(latest_association=association)
    )
    return (
        SimpleNamespace(composition_step=composition_step, execution=execution),
        emission,
        observer,
    )


def test_sim_hil_feedback_uses_selected_old_emission_and_exact_target_receipt():
    cycle, old_emission, observer = _feedback_cycle(
        candidate_hash="policy-action-hash"
    )
    new_emission = SimpleNamespace(
        emission_id="emission:new",
        semantic_content_hash="emission-hash:new",
    )

    result = project_sim_hil_policy_execution_feedback(
        cycle=cycle,
        cycle_emission=new_emission,
        emission_by_id={
            old_emission.emission_id: old_emission,
            new_emission.emission_id: new_emission,
        },
        execution_difference_observer=observer,
        observed_at=12.0,
        clock_domain_id="sim-clock",
    )

    assert result.status == "available"
    assert result.feedback is not None
    assert result.feedback.policy_emission_id == old_emission.emission_id
    assert result.feedback.policy_output_relation == POLICY_EXECUTION_RELATION_EXACT
    assert result.feedback.association_id == "association:1"
    assert result.feedback.downstream_action_status == "modified"
    assert result.feedback.reference_value_by_component["right_arm"][0] == 0.1


def test_sim_hil_feedback_marks_composed_output_and_rejects_unrelated_reference():
    cycle, emission, observer = _feedback_cycle(
        candidate_hash="human-policy-composed-hash",
        association_target="admission:other",
    )
    result = project_sim_hil_policy_execution_feedback(
        cycle=cycle,
        cycle_emission=emission,
        emission_by_id={emission.emission_id: emission},
        execution_difference_observer=observer,
        observed_at=12.0,
        clock_domain_id="sim-clock",
    )

    assert result.feedback is not None
    assert result.feedback.policy_output_relation == POLICY_EXECUTION_RELATION_COMPOSED
    assert result.feedback.association_id is None
    assert result.feedback.reference_value_by_component == {}

    application = PolicyExecutionFeedbackApplication(
        feedback_id=result.feedback.feedback_id,
        feedback_hash=result.feedback.semantic_content_hash,
        decision="recorded_by_test",
        reason="passive evidence",
    )
    fact = project_policy_execution_feedback(
        result.feedback,
        application,
        runtime_session_id="session:1",
        control_domain_id="domain:1",
        embodiment_instance_id="robot:1",
    )
    assert fact.event_type == "policy_execution_feedback"
    assert fact.payload["feedback"]["policy_output_relation"] == (
        POLICY_EXECUTION_RELATION_COMPOSED
    )
