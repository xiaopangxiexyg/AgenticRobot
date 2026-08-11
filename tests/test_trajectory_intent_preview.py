from __future__ import annotations

import pytest

from control_composition.trajectory import (
    TRAJECTORY_INTENT_PREVIEW_CLASS_PRODUCER_LOCKED,
    TRAJECTORY_INTENT_PREVIEW_CLASS_SPECULATIVE,
    TrajectoryIntentPreviewPoint,
    TrajectoryIntentPreviewSnapshot,
    TrajectoryPreviewFreshnessEvidence,
    TrajectoryPreviewFreshnessSpec,
    assess_trajectory_intent_preview_freshness,
    build_trajectory_intent_preview,
)
from robot_motion.action import (
    ActionComponentCatalogV2,
    ActionPayloadSchema,
    ComponentAction,
    ComponentActionSchema,
    ComponentActionTrajectory,
    ComponentActionTrajectoryKeyframe,
    ControlComponentSpecV2,
)


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _schema() -> ComponentActionSchema:
    payloads = {
        "tool": ActionPayloadSchema(
            schema_id="preview-tool-pose",
            schema_version="development",
            representation_family="cartesian_pose",
            dtype="float64",
            shape=(7,),
            coordinate_names=("x", "y", "z", "qx", "qy", "qz", "qw"),
            units=("m", "m", "m", "1", "1", "1", "1"),
            value_semantics="absolute_target",
            reference_frame="world",
            controlled_frame="tool",
        ),
        "end_effector": ActionPayloadSchema(
            schema_id="preview-end-effector-qpos",
            schema_version="development",
            representation_family="joint_position",
            dtype="float64",
            shape=(3,),
            coordinate_names=("joint_0", "joint_1", "joint_2"),
            units=("rad", "rad", "rad"),
            value_semantics="absolute_target",
        ),
    }
    catalog = ActionComponentCatalogV2(
        catalog_id="preview-arbitrary-topology",
        catalog_version="development",
        components=tuple(
            ControlComponentSpecV2(
                component_id=component_id,
                controlled_entity_id=component_id,
                command_channel_id=component_id,
                selection_group_id=component_id,
                resource_claims=(f"resource:{component_id}",),
                accepted_payload_schema_hashes=(
                    payloads[component_id].semantic_content_hash,
                ),
            )
            for component_id in payloads
        ),
    )
    return ComponentActionSchema(
        schema_id="preview-arbitrary-topology",
        schema_version="development",
        component_catalog=catalog,
        payload_schema_by_component=payloads,
    )


def _trajectory(*, immutable_through: float | None) -> ComponentActionTrajectory:
    schema = _schema()
    keyframes = tuple(
        ComponentActionTrajectoryKeyframe(
            target_effective_at=10.0 + index * 0.1,
            effective_until=10.1 + index * 0.1,
            action=ComponentAction(
                action_schema=schema,
                values_by_component={
                    "tool": (
                        0.1 * index,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ),
                    "end_effector": (index, -index, 0.5 * index),
                },
            ),
            source_index=index,
            source_fact_ids=(f"policy-keyframe:{index}",),
            source_fact_hashes=(f"policy-keyframe-hash:{index}",),
        )
        for index in range(3)
    )
    return ComponentActionTrajectory(
        trajectory_id="generic-future-plan",
        revision=7,
        source_kind="generic_policy",
        source_instance_id="source-instance:1",
        clock_domain_id="control.monotonic",
        created_at=9.9,
        keyframes=keyframes,
        source_fact_ids=("policy-plan:7",),
        source_fact_hashes=("policy-plan-hash:7",),
        source_immutable_through=immutable_through,
    )


def test_whole_horizon_preview_distinguishes_locked_and_speculative_intent() -> None:
    trajectory = _trajectory(immutable_through=10.1)

    preview = build_trajectory_intent_preview(
        trajectory,
        generated_at=9.95,
        generated_clock_domain_id="ui.monotonic",
    )

    assert tuple(point.preview_class for point in preview.points) == (
        TRAJECTORY_INTENT_PREVIEW_CLASS_PRODUCER_LOCKED,
        TRAJECTORY_INTENT_PREVIEW_CLASS_PRODUCER_LOCKED,
        TRAJECTORY_INTENT_PREVIEW_CLASS_SPECULATIVE,
    )
    assert len(preview.producer_locked_points) == 2
    assert len(preview.speculative_points) == 1
    assert preview.points[2].source_index == 2
    assert preview.points[2].action.component_coverage == (
        "tool",
        "end_effector",
    )
    assert preview.authoritative is False
    assert all(point.execution_guaranteed is False for point in preview.points)


def test_no_immutable_boundary_means_every_future_point_is_speculative() -> None:
    preview = build_trajectory_intent_preview(
        _trajectory(immutable_through=None),
        generated_at=9.95,
        generated_clock_domain_id="ui.monotonic",
    )

    assert not preview.producer_locked_points
    assert len(preview.speculative_points) == 3


def test_preview_is_deterministic_and_rejects_a_false_trust_class() -> None:
    trajectory = _trajectory(immutable_through=10.0)
    first = build_trajectory_intent_preview(
        trajectory,
        generated_at=9.95,
        generated_clock_domain_id="ui.monotonic",
    )
    replay = build_trajectory_intent_preview(
        trajectory,
        generated_at=9.95,
        generated_clock_domain_id="ui.monotonic",
    )
    assert first.semantic_content_hash == replay.semantic_content_hash

    false_points = (
        TrajectoryIntentPreviewPoint(
            keyframe=trajectory.keyframes[0],
            preview_class=TRAJECTORY_INTENT_PREVIEW_CLASS_SPECULATIVE,
        ),
        *first.points[1:],
    )
    with pytest.raises(ValueError, match="source_immutable_through"):
        TrajectoryIntentPreviewSnapshot(
            trajectory=trajectory,
            generated_at=9.95,
            generated_clock_domain_id="ui.monotonic",
            points=false_points,
        )


def _freshness_spec() -> TrajectoryPreviewFreshnessSpec:
    return TrajectoryPreviewFreshnessSpec(
        spec_id="intent-preview-display-age",
        fresh_max_age_sec=0.1,
        aging_max_age_sec=0.3,
        stale_displayable_max_age_sec=0.8,
    )


def _freshness_evidence(
    trajectory: ComponentActionTrajectory,
    *,
    observed_at: float = 9.9,
    valid_until: float | None = None,
    clock_domain_id: str = "control.monotonic",
) -> TrajectoryPreviewFreshnessEvidence:
    return TrajectoryPreviewFreshnessEvidence(
        trajectory_id=trajectory.trajectory_id,
        trajectory_hash=trajectory.semantic_content_hash,
        observation_fact_id="policy-observation:44",
        observation_fact_hash="policy-observation-hash:44",
        observed_at=observed_at,
        observed_clock_domain_id=clock_domain_id,
        available_at=9.91,
        available_clock_domain_id=clock_domain_id,
        valid_until=valid_until,
    )


@pytest.mark.parametrize(
    ("evaluated_at", "expected_class", "displayable"),
    (
        (9.95, "fresh", True),
        (10.1, "aging", True),
        (10.5, "stale_displayable", True),
        (10.8, "expired", False),
    ),
)
def test_preview_freshness_is_orthogonal_to_locked_speculative_class(
    evaluated_at: float,
    expected_class: str,
    displayable: bool,
) -> None:
    trajectory = _trajectory(immutable_through=10.1)
    preview = build_trajectory_intent_preview(
        trajectory,
        generated_at=9.95,
        generated_clock_domain_id="control.monotonic",
    )
    classes_before = tuple(point.preview_class for point in preview.points)

    assessment = assess_trajectory_intent_preview_freshness(
        preview,
        spec=_freshness_spec(),
        evaluated_at=evaluated_at,
        evaluated_clock_domain_id="control.monotonic",
        evidence=_freshness_evidence(trajectory),
    )

    assert assessment.freshness_class == expected_class
    assert assessment.displayable is displayable
    assert tuple(point.preview_class for point in preview.points) == classes_before


def test_missing_evidence_is_unknown_and_does_not_use_trajectory_created_at() -> None:
    trajectory = _trajectory(immutable_through=None)
    preview = build_trajectory_intent_preview(
        trajectory,
        generated_at=20.0,
        generated_clock_domain_id="control.monotonic",
    )
    preview_hash = preview.semantic_content_hash

    assessment = assess_trajectory_intent_preview_freshness(
        preview,
        spec=_freshness_spec(),
        evaluated_at=20.0,
        evaluated_clock_domain_id="control.monotonic",
    )

    assert assessment.freshness_class == "unknown"
    assert assessment.observation_age_sec is None
    assert assessment.displayable is True
    assert preview.semantic_content_hash == preview_hash


def test_unmapped_observation_clock_is_unknown_instead_of_guessing_age() -> None:
    trajectory = _trajectory(immutable_through=10.0)
    preview = build_trajectory_intent_preview(
        trajectory,
        generated_at=9.95,
        generated_clock_domain_id="control.monotonic",
    )

    assessment = assess_trajectory_intent_preview_freshness(
        preview,
        spec=_freshness_spec(),
        evaluated_at=9.95,
        evaluated_clock_domain_id="control.monotonic",
        evidence=_freshness_evidence(
            trajectory,
            clock_domain_id="model.monotonic",
        ),
    )

    assert assessment.freshness_class == "unknown"
    assert assessment.reason == "freshness_clock_unmapped"
    assert assessment.observation_age_sec is None


def test_explicit_valid_until_expires_even_inside_age_window() -> None:
    trajectory = _trajectory(immutable_through=10.0)
    preview = build_trajectory_intent_preview(
        trajectory,
        generated_at=9.95,
        generated_clock_domain_id="control.monotonic",
    )

    assessment = assess_trajectory_intent_preview_freshness(
        preview,
        spec=_freshness_spec(),
        evaluated_at=9.96,
        evaluated_clock_domain_id="control.monotonic",
        evidence=_freshness_evidence(
            trajectory,
            valid_until=9.95,
        ),
    )

    assert assessment.freshness_class == "expired"
    assert assessment.reason == "explicit_validity_expired"
    assert assessment.displayable is False
