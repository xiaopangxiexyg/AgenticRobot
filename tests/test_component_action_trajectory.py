from __future__ import annotations

import numpy as np
import pytest

from robot_motion.action import (
    ActionComponentCatalogV2,
    ActionPayloadSchema,
    ComponentAction,
    ComponentActionSchema,
    ComponentActionTrajectory,
    ComponentActionTrajectoryKeyframe,
    ControlComponentSpecV2,
    project_component_action_trajectory,
)
from robot_motion.deploy import (
    POLICY_PLAN_KEYFRAME_MUTABLE,
    POLICY_PLAN_KEYFRAME_PRODUCER_COMMITTED,
    NormalizedAction,
    TimedPolicyActionKeyframe,
    TimedPolicyActionPlanSnapshot,
    project_timed_policy_plan_to_component_trajectory,
)


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _schema() -> ComponentActionSchema:
    payloads = {
        "arm": ActionPayloadSchema(
            schema_id="trajectory-arm-pose",
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
        "hand": ActionPayloadSchema(
            schema_id="trajectory-hand-qpos",
            schema_version="development",
            representation_family="joint_position",
            dtype="float64",
            shape=(2,),
            coordinate_names=("finger_0", "finger_1"),
            units=("rad", "rad"),
            value_semantics="absolute_target",
        ),
    }
    catalog = ActionComponentCatalogV2(
        catalog_id="trajectory-fixture",
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
            for component_id in ("arm", "hand")
        ),
    )
    return ComponentActionSchema(
        schema_id="trajectory-fixture",
        schema_version="development",
        component_catalog=catalog,
        payload_schema_by_component=payloads,
    )


def _action(x: float) -> ComponentAction:
    return ComponentAction(
        action_schema=_schema(),
        values_by_component={
            "arm": (x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            "hand": (x, -x),
        },
    )


def _keyframe(index: int, x: float) -> ComponentActionTrajectoryKeyframe:
    return ComponentActionTrajectoryKeyframe(
        target_effective_at=10.0 + index * 0.1,
        effective_until=10.1 + index * 0.1,
        action=_action(x),
        source_index=index,
        source_fact_ids=(f"source:{index}",),
        source_fact_hashes=(f"source-hash:{index}",),
    )


def test_component_action_trajectory_has_explicit_support_and_stable_identity() -> None:
    trajectory = ComponentActionTrajectory(
        trajectory_id="virtual-policy-plan",
        revision=4,
        source_kind="virtual_policy",
        source_instance_id="virtual-policy",
        clock_domain_id="control.monotonic",
        created_at=9.9,
        keyframes=(_keyframe(0, 0.0), _keyframe(1, 0.1)),
        source_fact_ids=("plan:4",),
        source_fact_hashes=("plan-hash:4",),
        source_immutable_through=10.0,
        coordination_group_id="arm-hand",
    )
    replay = ComponentActionTrajectory(
        trajectory_id="virtual-policy-plan",
        revision=4,
        source_kind="virtual_policy",
        source_instance_id="virtual-policy",
        clock_domain_id="control.monotonic",
        created_at=9.9,
        keyframes=(_keyframe(0, 0.0), _keyframe(1, 0.1)),
        source_fact_ids=("plan:4",),
        source_fact_hashes=("plan-hash:4",),
        source_immutable_through=10.0,
        coordination_group_id="arm-hand",
    )

    assert trajectory.semantic_content_hash == replay.semantic_content_hash
    assert trajectory.fact_id == (
        f"component-action-trajectory:{trajectory.semantic_content_hash}"
    )
    assert trajectory.effective_from == 10.0
    assert trajectory.effective_until == pytest.approx(10.2)
    assert trajectory.component_coverage == ("arm", "hand")
    assert trajectory.keyframe_at(10.1).action.semantic_content_hash == (
        _action(0.1).semantic_content_hash
    )
    assert trajectory.keyframe_at(10.15) is None

    restarted_source = ComponentActionTrajectory(
        trajectory_id="virtual-policy-plan",
        revision=4,
        source_kind="virtual_policy",
        source_instance_id="virtual-policy-restart",
        clock_domain_id="control.monotonic",
        created_at=9.9,
        keyframes=(_keyframe(0, 0.0), _keyframe(1, 0.1)),
        source_fact_ids=("plan:4",),
        source_fact_hashes=("plan-hash:4",),
        source_immutable_through=10.0,
        coordination_group_id="arm-hand",
    )
    assert restarted_source.semantic_content_hash != (
        trajectory.semantic_content_hash
    )


def test_trajectory_source_trace_preserves_every_keyframe_action() -> None:
    trajectory = ComponentActionTrajectory(
        trajectory_id="virtual-policy-plan",
        revision=4,
        source_kind="virtual_policy",
        source_instance_id="virtual-policy",
        clock_domain_id="control.monotonic",
        created_at=9.9,
        keyframes=(_keyframe(0, 0.0), _keyframe(1, 0.1)),
        source_fact_ids=("plan:4",),
        source_fact_hashes=("plan-hash:4",),
        source_activation_id="producer-activation:1",
        source_activation_hash="producer-activation-hash:1",
        source_immutable_through=10.0,
        coordination_group_id="arm-hand",
    )

    fact = project_component_action_trajectory(
        trajectory,
        runtime_session_id="runtime:trajectory",
        control_domain_id="control:trajectory",
        embodiment_instance_id="virtual-embodiment",
    )

    assert fact.event_type == "component_action_trajectory"
    assert fact.fact_id == trajectory.fact_id
    assert fact.event_family == "source"
    assert fact.occurred_at == trajectory.created_at
    assert fact.clock_domain_id == trajectory.clock_domain_id
    assert fact.payload["semantic_content_hash"] == (
        trajectory.semantic_content_hash
    )
    keyframes = fact.payload["keyframes"]
    assert len(keyframes) == 2
    assert keyframes[1]["semantic_content_hash"] == (
        trajectory.keyframes[1].semantic_content_hash
    )
    np.testing.assert_allclose(
        keyframes[1]["action"]["values_by_component"]["arm"]["values"],
        (0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    )
    np.testing.assert_allclose(
        keyframes[1]["action"]["values_by_component"]["hand"]["values"],
        (0.1, -0.1),
    )


def test_trajectory_rejects_component_coverage_changes() -> None:
    partial = ComponentAction(
        action_schema=_schema(),
        values_by_component={
            "arm": (0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        },
    )
    with pytest.raises(ValueError, match="component coverage"):
        ComponentActionTrajectory(
            trajectory_id="invalid",
            revision=0,
            source_kind="virtual",
            source_instance_id="virtual-policy",
            clock_domain_id="control.monotonic",
            created_at=9.9,
            keyframes=(
                _keyframe(0, 0.0),
                ComponentActionTrajectoryKeyframe(
                    target_effective_at=10.1,
                    effective_until=10.2,
                    action=partial,
                    source_index=1,
                    source_fact_ids=("partial",),
                    source_fact_hashes=("partial-hash",),
                ),
            ),
            source_fact_ids=("invalid-plan",),
            source_fact_hashes=("invalid-plan-hash",),
        )


def _timed_keyframe(
    *,
    target: float,
    chunk_id: int,
    chunk_index: int,
    state: str,
) -> TimedPolicyActionKeyframe:
    return TimedPolicyActionKeyframe(
        target_effective_at=target,
        effective_until=target + 0.1,
        effective_clock_domain_id="control.monotonic",
        action_period_sec=0.1,
        decoded_absolute_action=NormalizedAction(
            vector=np.asarray([target], dtype=np.float32),
            metadata={"target": target},
        ),
        canonical_action=None,
        request_id=f"request:{chunk_id}",
        inference_id=f"inference:{chunk_id}",
        policy_output_id=f"output:{chunk_id}",
        chunk_id=chunk_id,
        chunk_keyframe_index=chunk_index,
        chunk_size=1,
        observation_cut_id=f"cut:{chunk_id}",
        producer_epoch=1,
        policy_node_id="virtual-policy",
        artifact_hash="virtual-artifact",
        produced_at=9.8,
        schedule_spec_id="virtual-schedule",
        schedule_spec_hash="virtual-schedule-hash",
        admitted_plan_generation=3,
        state=state,
        valid_until=target + 0.2,
    )


def test_timed_plan_projection_is_pure_and_preserves_source_boundary() -> None:
    plan = TimedPolicyActionPlanSnapshot(
        plan_id="virtual-plan",
        session_id="session",
        control_domain_id="robot",
        producer_id="policy",
        producer_epoch=1,
        policy_node_id="virtual-policy",
        clock_domain_id="control.monotonic",
        generation=3,
        cursor_at=10.0,
        created_at=9.9,
        keyframes=(
            _timed_keyframe(
                target=10.0,
                chunk_id=7,
                chunk_index=0,
                state=POLICY_PLAN_KEYFRAME_PRODUCER_COMMITTED,
            ),
            _timed_keyframe(
                target=10.1,
                chunk_id=8,
                chunk_index=0,
                state=POLICY_PLAN_KEYFRAME_MUTABLE,
            ),
        ),
        producer_committed_through=10.0,
    )

    def project(keyframe: TimedPolicyActionKeyframe) -> ComponentAction:
        return _action(float(keyframe.decoded_absolute_action.vector[0]))

    before = plan.semantic_content_hash
    trajectory = project_timed_policy_plan_to_component_trajectory(
        plan,
        action_projector=project,
        coordination_group_id="arm-hand",
    )

    assert plan.semantic_content_hash == before
    assert trajectory.trajectory_id == "component-action-trajectory:virtual-plan"
    assert trajectory.revision == 3
    assert trajectory.source_immutable_through == 10.0
    assert tuple(item.source_index for item in trajectory.keyframes) == (0, 1)
    assert "timed-policy-action-keyframe:8:0:" in (
        trajectory.keyframes[1].source_fact_ids[0]
    )
    assert trajectory.source_instance_id == "policy"
    assert trajectory.source_activation_id is None

    activated = project_timed_policy_plan_to_component_trajectory(
        plan,
        action_projector=project,
        source_activation_id="producer-activation:1",
        source_activation_hash="producer-activation-hash:1",
    )
    assert activated.source_activation_id == "producer-activation:1"
    assert activated.semantic_content_hash != trajectory.semantic_content_hash

    with pytest.raises(ValueError, match="both present"):
        project_timed_policy_plan_to_component_trajectory(
            plan,
            action_projector=project,
            source_activation_id="producer-activation:1",
        )
