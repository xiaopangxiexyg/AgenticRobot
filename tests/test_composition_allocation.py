from __future__ import annotations

import math

import numpy as np
import pytest

from control_composition.allocation import (
    COMPOSITION_ALLOCATION_MODE_ABSOLUTE_OVERLAY_OWNER,
    COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
    COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY,
    COMPOSITION_ALLOCATION_STATUS_NO_DECISION,
    COMPOSITION_ALLOCATION_STATUS_READY,
    ComponentAllocationEvidencePolicy,
    ComponentAllocationPolicy,
    ComponentResidualMagnitudeObservation,
    ComponentSourceActivityObservation,
    CompositionAllocationSpec,
    SourceActionActivitySample,
    build_component_residual_magnitude_observation,
    build_component_source_activity_observation,
    empty_composition_allocation_state,
    evaluate_composition_allocation,
)
from control_composition import (
    ResidualAnchorComponentBinding,
    ResidualAnchorReceipt,
    ResidualAnchorRequest,
    build_composition_activation_scope,
)
from robot_motion.action import (
    ActionComponentCatalogV2,
    ActionPayloadSchema,
    ComponentAction,
    ComponentActionSchema,
    ControlComponentSpecV2,
)


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _allocation_spec(
    *,
    components: tuple[str, ...] = ("tool",),
    automatic_mode_ceiling: str = (
        COMPOSITION_ALLOCATION_MODE_ABSOLUTE_OVERLAY_OWNER
    ),
) -> CompositionAllocationSpec:
    return CompositionAllocationSpec(
        allocation_spec_id="generic-source-activity-allocation",
        base_absolute_role="base_absolute",
        relative_overlay_role="operator_relative",
        absolute_overlay_role="operator_absolute",
        component_policies=tuple(
            ComponentAllocationPolicy(
                component_id=component,
                motion_source_role="operator_activity",
                residual_source_role="operator_relative",
                motion_policy=ComponentAllocationEvidencePolicy(
                    relative_enter_level=1.0,
                    relative_exit_level=0.5,
                    absolute_enter_level=3.0,
                    absolute_exit_level=2.0,
                ),
                residual_magnitude_policy=ComponentAllocationEvidencePolicy(
                    relative_enter_level=1.0,
                    relative_exit_level=0.5,
                    absolute_enter_level=3.0,
                    absolute_exit_level=2.0,
                ),
                minimum_dwell_sec=0.2,
                automatic_mode_ceiling=automatic_mode_ceiling,
            )
            for component in components
        ),
    )


def _activity(
    component_id: str,
    *,
    level: float,
    evaluated_at: float,
    valid_until: float | None = None,
) -> ComponentSourceActivityObservation:
    return ComponentSourceActivityObservation(
        source_role="operator_activity",
        metric_provider_id="test.activity.v1",
        metric_provider_hash="test-activity-provider-hash:v1",
        component_id=component_id,
        representation_family="test.normalized_activity",
        window_started_at=evaluated_at - 0.20,
        window_ended_at=evaluated_at - 0.05,
        available_at=evaluated_at - 0.01,
        valid_until=(evaluated_at + 0.25 if valid_until is None else valid_until),
        clock_domain_id="control.monotonic",
        metric_values={"normalized_motion": level},
        normalization_scale_by_metric={"normalized_motion": 1.0},
        source_fact_ids=(f"activity-source:{component_id}:{evaluated_at}",),
        source_fact_hashes=(
            f"activity-source-hash:{component_id}:{evaluated_at}:{level}",
        ),
    )


def _magnitude(
    component_id: str,
    *,
    level: float,
    evaluated_at: float,
    valid_until: float | None = None,
    anchor_receipt_id: str = "residual-anchor-receipt:0",
    anchor_receipt_hash: str = "residual-anchor-receipt-hash:0",
    anchor_request_generation: int = 0,
) -> ComponentResidualMagnitudeObservation:
    return ComponentResidualMagnitudeObservation(
        source_role="operator_relative",
        metric_provider_id="test.residual-magnitude.v1",
        metric_provider_hash="test-residual-magnitude-provider-hash:v1",
        component_id=component_id,
        representation_family="test.normalized_residual",
        sampled_at=evaluated_at - 0.02,
        available_at=evaluated_at - 0.01,
        valid_until=(evaluated_at + 0.25 if valid_until is None else valid_until),
        clock_domain_id="control.monotonic",
        anchor_receipt_id=anchor_receipt_id,
        anchor_receipt_hash=anchor_receipt_hash,
        anchor_request_generation=anchor_request_generation,
        residual_action_id=f"residual-action:{component_id}:{evaluated_at}",
        residual_action_hash=(
            f"residual-action-hash:{component_id}:{evaluated_at}:{level}"
        ),
        metric_values={"normalized_magnitude": level},
        normalization_scale_by_metric={"normalized_magnitude": 1.0},
    )


def _evaluate(
    state,
    spec,
    *,
    at: float,
    level: float,
    magnitude_level: float | None = None,
):
    return evaluate_composition_allocation(
        state,
        spec=spec,
        activity_by_component={"tool": _activity("tool", level=level, evaluated_at=at)},
        residual_magnitude_by_component={
            "tool": _magnitude(
                "tool",
                level=level if magnitude_level is None else magnitude_level,
                evaluated_at=at,
            )
        },
        evaluated_at=at,
        clock_domain_id="control.monotonic",
    )


def test_static_small_and_large_activity_select_three_soft_modes() -> None:
    spec = _allocation_spec()
    state = empty_composition_allocation_state("allocation-reducer", spec=spec)

    static = _evaluate(state, spec, at=1.0, level=0.0)
    assert static.decision.output_status == COMPOSITION_ALLOCATION_STATUS_READY
    assert static.decision.selected_mode_by_component["tool"] == (
        COMPOSITION_ALLOCATION_MODE_BASE_ONLY
    )

    small = _evaluate(static.state, spec, at=1.1, level=1.2)
    assert small.decision.selected_mode_by_component["tool"] == (
        COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY
    )

    dwell = _evaluate(small.state, spec, at=1.15, level=4.0)
    assert dwell.decision.selected_mode_by_component["tool"] == (
        COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY
    )
    assert dwell.decision.selections[0].reason == "minimum_dwell_active"

    takeover = _evaluate(dwell.state, spec, at=1.31, level=4.0)
    assert takeover.decision.selected_mode_by_component["tool"] == (
        COMPOSITION_ALLOCATION_MODE_ABSOLUTE_OVERLAY_OWNER
    )
    assert takeover.decision.authoritative is False

    back_to_relative = _evaluate(takeover.state, spec, at=1.55, level=1.5)
    assert back_to_relative.decision.selected_mode_by_component["tool"] == (
        COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY
    )

    back_to_base = _evaluate(
        back_to_relative.state,
        spec,
        at=1.80,
        level=0.2,
    )
    assert back_to_base.decision.selected_mode_by_component["tool"] == (
        COMPOSITION_ALLOCATION_MODE_BASE_ONLY
    )


def test_relative_only_automatic_ceiling_keeps_large_evidence_publishable() -> None:
    spec = _allocation_spec(
        automatic_mode_ceiling=COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY
    )
    state = empty_composition_allocation_state("allocation-reducer", spec=spec)

    large = _evaluate(state, spec, at=1.0, level=4.0)

    assert large.decision.selected_mode_by_component["tool"] == (
        COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY
    )
    assert large.decision.selections[0].reason == (
        "evidence_crossed_absolute_enter:automatic_mode_ceiling"
    )


def test_automatic_ceiling_is_component_local_for_pose_and_qpos() -> None:
    spec = _allocation_spec(
        components=("arm", "gripper"),
        automatic_mode_ceiling=COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY,
    )
    state = empty_composition_allocation_state("allocation-reducer", spec=spec)

    result = evaluate_composition_allocation(
        state,
        spec=spec,
        activity_by_component={
            "arm": _activity("arm", level=4.0, evaluated_at=1.0),
            "gripper": _activity("gripper", level=8.0, evaluated_at=1.0),
        },
        residual_magnitude_by_component={
            "arm": _magnitude("arm", level=4.0, evaluated_at=1.0),
            "gripper": _magnitude("gripper", level=8.0, evaluated_at=1.0),
        },
        evaluated_at=1.0,
        clock_domain_id="control.monotonic",
    )

    assert result.decision.selected_mode_by_component == {
        "arm": COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY,
        "gripper": COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY,
    }
    assert all(
        selection.reason.endswith(":automatic_mode_ceiling")
        for selection in result.decision.selections
    )


def test_quiet_nonzero_residual_stays_relative_until_anchor_magnitude_clears() -> None:
    spec = _allocation_spec()
    state = empty_composition_allocation_state("allocation-reducer", spec=spec)

    entered = _evaluate(
        state,
        spec,
        at=1.0,
        level=1.2,
        magnitude_level=1.2,
    )
    assert entered.decision.selected_mode_by_component["tool"] == (
        COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY
    )

    quiet_away_from_anchor = _evaluate(
        entered.state,
        spec,
        at=1.3,
        level=0.0,
        magnitude_level=1.2,
    )
    assert quiet_away_from_anchor.decision.selected_mode_by_component["tool"] == (
        COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY
    )
    assert quiet_away_from_anchor.decision.selections[0].reason == (
        "evidence_within_relative_hysteresis"
    )

    rebased_to_identity = _evaluate(
        quiet_away_from_anchor.state,
        spec,
        at=1.6,
        level=0.0,
        magnitude_level=0.0,
    )
    assert rebased_to_identity.decision.selected_mode_by_component["tool"] == (
        COMPOSITION_ALLOCATION_MODE_BASE_ONLY
    )


def test_hysteresis_prevents_threshold_chatter() -> None:
    spec = _allocation_spec()
    state = empty_composition_allocation_state("allocation-reducer", spec=spec)
    state = _evaluate(state, spec, at=1.0, level=1.1).state

    for index, level in enumerate((0.99, 0.75, 0.51, 0.8), start=1):
        result = _evaluate(state, spec, at=1.0 + 0.25 * index, level=level)
        assert result.decision.selected_mode_by_component["tool"] == (
            COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY
        )
        state = result.state


def test_missing_and_stale_activity_are_not_treated_as_static() -> None:
    spec = _allocation_spec()
    state = empty_composition_allocation_state("allocation-reducer", spec=spec)

    missing = evaluate_composition_allocation(
        state,
        spec=spec,
        activity_by_component={},
        residual_magnitude_by_component={
            "tool": _magnitude("tool", level=0.0, evaluated_at=1.0)
        },
        evaluated_at=1.0,
        clock_domain_id="control.monotonic",
    )
    assert missing.decision.output_status == COMPOSITION_ALLOCATION_STATUS_NO_DECISION
    assert missing.decision.unavailable_reason_by_component == {
        "tool": "missing_motion_observation"
    }
    assert missing.state.semantic_content_hash == state.semantic_content_hash

    stale_observation = _activity(
        "tool",
        level=0.0,
        evaluated_at=1.0,
        valid_until=1.1,
    )
    stale = evaluate_composition_allocation(
        state,
        spec=spec,
        activity_by_component={"tool": stale_observation},
        residual_magnitude_by_component={
            "tool": _magnitude("tool", level=0.0, evaluated_at=1.1)
        },
        evaluated_at=1.1,
        clock_domain_id="control.monotonic",
    )
    assert stale.decision.output_status == COMPOSITION_ALLOCATION_STATUS_NO_DECISION
    assert stale.decision.unavailable_reason_by_component == {
        "tool": "motion_observation_stale"
    }
    assert stale.state.semantic_content_hash == state.semantic_content_hash

    fresh_static = _evaluate(state, spec, at=1.0, level=0.0)
    assert fresh_static.decision.output_status == COMPOSITION_ALLOCATION_STATUS_READY
    assert fresh_static.decision.selected_mode_by_component["tool"] == (
        COMPOSITION_ALLOCATION_MODE_BASE_ONLY
    )


def test_stale_residual_magnitude_is_not_treated_as_anchor_neutral() -> None:
    spec = _allocation_spec()
    state = empty_composition_allocation_state("allocation-reducer", spec=spec)

    stale = evaluate_composition_allocation(
        state,
        spec=spec,
        activity_by_component={
            "tool": _activity("tool", level=0.0, evaluated_at=1.1)
        },
        residual_magnitude_by_component={
            "tool": _magnitude(
                "tool",
                level=0.0,
                evaluated_at=1.0,
                valid_until=1.1,
            )
        },
        evaluated_at=1.1,
        clock_domain_id="control.monotonic",
    )

    assert stale.decision.output_status == COMPOSITION_ALLOCATION_STATUS_NO_DECISION
    assert stale.decision.unavailable_reason_by_component == {
        "tool": "residual_magnitude_observation_stale"
    }
    assert stale.state.semantic_content_hash == state.semantic_content_hash


def test_multi_component_decision_is_atomic_when_one_activity_fact_is_missing() -> None:
    spec = _allocation_spec(components=("left_tool", "gripper"))
    state = empty_composition_allocation_state("allocation-reducer", spec=spec)

    result = evaluate_composition_allocation(
        state,
        spec=spec,
        activity_by_component={
            "left_tool": _activity(
                "left_tool",
                level=1.5,
                evaluated_at=2.0,
            )
        },
        residual_magnitude_by_component={
            component: _magnitude(
                component,
                level=0.0,
                evaluated_at=2.0,
            )
            for component in ("left_tool", "gripper")
        },
        evaluated_at=2.0,
        clock_domain_id="control.monotonic",
    )

    assert result.decision.output_status == COMPOSITION_ALLOCATION_STATUS_NO_DECISION
    assert result.decision.unavailable_components == ("gripper",)
    assert result.state.semantic_content_hash == state.semantic_content_hash


def test_atomic_component_set_rejects_mixed_anchor_generations() -> None:
    spec = _allocation_spec(components=("left_tool", "right_tool"))
    state = empty_composition_allocation_state("allocation-reducer", spec=spec)

    with pytest.raises(ValueError, match="one residual anchor receipt"):
        evaluate_composition_allocation(
            state,
            spec=spec,
            activity_by_component={
                component: _activity(component, level=0.0, evaluated_at=2.0)
                for component in spec.component_ids
            },
            residual_magnitude_by_component={
                "left_tool": _magnitude(
                    "left_tool",
                    level=0.0,
                    evaluated_at=2.0,
                ),
                "right_tool": _magnitude(
                    "right_tool",
                    level=0.0,
                    evaluated_at=2.0,
                    anchor_receipt_id="residual-anchor-receipt:1",
                    anchor_receipt_hash="residual-anchor-receipt-hash:1",
                    anchor_request_generation=1,
                ),
            },
            evaluated_at=2.0,
            clock_domain_id="control.monotonic",
        )


def _action_schema() -> ComponentActionSchema:
    payloads = {
        "tool": ActionPayloadSchema(
            schema_id="activity-tool-pose",
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
            schema_id="activity-hand-qpos",
            schema_version="development",
            representation_family="joint_position",
            dtype="float64",
            shape=(5,),
            coordinate_names=tuple(f"joint_{index}" for index in range(5)),
            units=("rad",) * 5,
            value_semantics="absolute_target",
        ),
    }
    catalog = ActionComponentCatalogV2(
        catalog_id="activity-arbitrary-topology",
        catalog_version="development",
        components=tuple(
            ControlComponentSpecV2(
                component_id=component_id,
                controlled_entity_id=component_id,
                command_channel_id=component_id,
                selection_group_id=component_id,
                resource_claims=(f"resource:{component_id}",),
                accepted_payload_schema_hashes=(payload.semantic_content_hash,),
            )
            for component_id, payload in payloads.items()
        ),
    )
    return ComponentActionSchema(
        schema_id="activity-arbitrary-topology",
        schema_version="development",
        component_catalog=catalog,
        payload_schema_by_component=payloads,
    )


def _source_sample(
    index: int,
    *,
    tool_values: tuple[float, ...],
    hand_values: tuple[float, ...],
) -> SourceActionActivitySample:
    return SourceActionActivitySample(
        sampled_at=3.0 + index * 0.1,
        clock_domain_id="control.monotonic",
        action=ComponentAction(
            action_schema=_action_schema(),
            values_by_component={"tool": tool_values, "hand": hand_values},
        ),
        source_fact_id=f"operator-action:{index}",
        source_fact_hash=f"operator-action-hash:{index}",
    )


def _relative_action_schema() -> ComponentActionSchema:
    payloads = {
        "tool": ActionPayloadSchema(
            schema_id="allocation-tool-pose-residual",
            schema_version="development",
            representation_family="cartesian_pose",
            dtype="float64",
            shape=(7,),
            coordinate_names=("x", "y", "z", "qx", "qy", "qz", "qw"),
            units=("m", "m", "m", "1", "1", "1", "1"),
            value_semantics="relative_delta",
            reference_frame="world",
            controlled_frame="tool",
        ),
        "hand": ActionPayloadSchema(
            schema_id="allocation-hand-qpos-residual",
            schema_version="development",
            representation_family="joint_position",
            dtype="float64",
            shape=(5,),
            coordinate_names=tuple(f"joint_{index}" for index in range(5)),
            units=("rad",) * 5,
            value_semantics="relative_delta",
        ),
    }
    catalog = ActionComponentCatalogV2(
        catalog_id="allocation-residual-topology",
        catalog_version="development",
        components=tuple(
            ControlComponentSpecV2(
                component_id=component_id,
                controlled_entity_id=component_id,
                command_channel_id=component_id,
                selection_group_id=component_id,
                resource_claims=(f"resource:{component_id}",),
                accepted_payload_schema_hashes=(payload.semantic_content_hash,),
            )
            for component_id, payload in payloads.items()
        ),
    )
    return ComponentActionSchema(
        schema_id="allocation-residual-topology",
        schema_version="development",
        component_catalog=catalog,
        payload_schema_by_component=payloads,
    )


def _residual_anchor_receipt(
    relative_schema: ComponentActionSchema,
) -> ResidualAnchorReceipt:
    activation = build_composition_activation_scope(
        operation_spec_id="allocation-residual-evidence",
        operation_spec_hash="allocation-residual-evidence-hash",
        session_epoch=1,
        activated_at=2.5,
        clock_domain_id="control.monotonic",
        input_barrier_at=2.5,
        required_roles=("base_absolute", "operator_relative"),
    )
    request = ResidualAnchorRequest.for_activation(
        activation,
        request_generation=0,
        source_role="operator_relative",
        component_ids=("tool", "hand"),
        source_binding_id="operator-source:1",
        source_binding_hash="operator-source-hash:1",
        absolute_action_schema_hash=_action_schema().semantic_content_hash,
        relative_action_schema_hash=relative_schema.semantic_content_hash,
        requested_at=2.6,
    )
    return ResidualAnchorReceipt.capture(
        request,
        source_action_id="operator-absolute-anchor:0",
        source_action_hash="operator-absolute-anchor-hash:0",
        source_available_at=2.7,
        captured_at=2.71,
        component_bindings=tuple(
            ResidualAnchorComponentBinding(
                component_id=component_id,
                source_sample_id=f"operator-anchor:{component_id}:0",
                source_sample_hash=f"operator-anchor-hash:{component_id}:0",
                absolute_payload_schema_hash=(
                    _action_schema()
                    .payload_schema_by_component[component_id]
                    .semantic_content_hash
                ),
                anchor_id=f"anchor:{component_id}:0",
                anchor_hash=f"anchor-hash:{component_id}:0",
            )
            for component_id in ("tool", "hand")
        ),
    )


def test_standard_activity_metrics_cover_pose_and_arbitrary_qpos_dimension() -> None:
    samples = (
        _source_sample(
            0,
            tool_values=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            hand_values=(0.0, 0.0, 0.0, 0.0, 0.0),
        ),
        _source_sample(
            1,
            tool_values=(
                0.01,
                0.0,
                0.0,
                0.0,
                0.0,
                math.sin(math.pi / 4.0),
                math.cos(math.pi / 4.0),
            ),
            hand_values=(0.1, -0.2, 0.3, -0.4, 0.5),
        ),
    )
    pose = build_component_source_activity_observation(
        samples,
        source_role="operator_activity",
        component_id="tool",
        available_at=3.11,
        valid_until=3.3,
        normalization_scale_by_metric={
            "translation_displacement_m": 0.02,
            "rotation_displacement_rad": math.pi / 4.0,
        },
    )
    assert pose.metric_values["translation_displacement_m"] == pytest.approx(0.01)
    assert pose.metric_values["rotation_displacement_rad"] == pytest.approx(
        math.pi / 2.0
    )
    assert pose.activity_level == pytest.approx(2.0)
    assert pose.metric_provider_id.endswith("standard_component_activity_metrics.v1")

    qpos = build_component_source_activity_observation(
        samples,
        source_role="operator_activity",
        component_id="hand",
        available_at=3.11,
        valid_until=3.3,
        normalization_scale_by_metric={
            "coordinate_linf_displacement_native": 0.25,
        },
    )
    assert qpos.metric_values["coordinate_linf_displacement_native"] == (
        pytest.approx(0.5)
    )
    assert qpos.activity_level == pytest.approx(2.0)


def test_custom_activity_provider_requires_replayable_identity() -> None:
    samples = (
        _source_sample(
            0,
            tool_values=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            hand_values=(0.0, 0.0, 0.0, 0.0, 0.0),
        ),
        _source_sample(
            1,
            tool_values=(0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            hand_values=(0.0, 0.0, 0.0, 0.0, 0.0),
        ),
    )

    with pytest.raises(ValueError, match="requires an explicit"):
        build_component_source_activity_observation(
            samples,
            source_role="operator_activity",
            component_id="tool",
            available_at=3.11,
            valid_until=3.3,
            normalization_scale_by_metric={"custom_motion": 1.0},
            metric_provider=lambda _samples, _component: {"custom_motion": 0.2},
        )

    observation = build_component_source_activity_observation(
        samples,
        source_role="operator_activity",
        component_id="tool",
        available_at=3.11,
        valid_until=3.3,
        normalization_scale_by_metric={"custom_motion": 1.0},
        metric_provider=lambda _samples, _component: {"custom_motion": 0.2},
        metric_provider_id="test.custom_activity.v1",
        metric_provider_hash="test-custom-activity-contract-hash:v1",
    )
    assert observation.metric_provider_id == "test.custom_activity.v1"
    assert observation.metric_provider_hash == (
        "test-custom-activity-contract-hash:v1"
    )


def test_residual_magnitude_is_identity_relative_and_anchor_bound() -> None:
    schema = _relative_action_schema()
    receipt = _residual_anchor_receipt(schema)
    sample = SourceActionActivitySample(
        sampled_at=3.0,
        clock_domain_id="control.monotonic",
        action=ComponentAction(
            action_schema=schema,
            values_by_component={
                "tool": (
                    0.03,
                    0.04,
                    0.0,
                    0.0,
                    0.0,
                    math.sin(math.pi / 8.0),
                    math.cos(math.pi / 8.0),
                ),
                "hand": (0.1, -0.2, 0.3, -0.4, 0.5),
            },
        ),
        source_fact_id="operator-residual:0",
        source_fact_hash="operator-residual-hash:0",
    )

    pose = build_component_residual_magnitude_observation(
        sample,
        source_role="operator_relative",
        component_id="tool",
        anchor_receipt=receipt,
        available_at=3.01,
        valid_until=3.2,
        normalization_scale_by_metric={
            "translation_from_anchor_m": 0.05,
            "rotation_from_anchor_rad": math.pi / 4.0,
        },
    )
    assert pose.metric_values["translation_from_anchor_m"] == pytest.approx(0.05)
    assert pose.metric_values["rotation_from_anchor_rad"] == pytest.approx(
        math.pi / 4.0
    )
    assert pose.magnitude_level == pytest.approx(1.0)
    assert pose.anchor_receipt_id == receipt.receipt_id
    assert pose.metric_provider_id.endswith(
        "standard_component_residual_magnitude_metrics.v1"
    )

    qpos = build_component_residual_magnitude_observation(
        sample,
        source_role="operator_relative",
        component_id="hand",
        anchor_receipt=receipt,
        available_at=3.01,
        valid_until=3.2,
        normalization_scale_by_metric={
            "coordinate_linf_from_anchor_native": 0.25,
        },
    )
    assert qpos.metric_values["coordinate_linf_from_anchor_native"] == (
        pytest.approx(0.5)
    )
    assert qpos.magnitude_level == pytest.approx(2.0)


def test_allocation_replay_is_hash_deterministic() -> None:
    spec = _allocation_spec()
    initial_a = empty_composition_allocation_state("allocation-reducer", spec=spec)
    initial_b = empty_composition_allocation_state("allocation-reducer", spec=spec)
    observation = _activity("tool", level=1.2, evaluated_at=4.0)

    first = evaluate_composition_allocation(
        initial_a,
        spec=spec,
        activity_by_component={"tool": observation},
        residual_magnitude_by_component={
            "tool": _magnitude("tool", level=1.2, evaluated_at=4.0)
        },
        evaluated_at=4.0,
        clock_domain_id="control.monotonic",
    )
    replay = evaluate_composition_allocation(
        initial_b,
        spec=spec,
        activity_by_component={"tool": observation},
        residual_magnitude_by_component={
            "tool": _magnitude("tool", level=1.2, evaluated_at=4.0)
        },
        evaluated_at=4.0,
        clock_domain_id="control.monotonic",
    )

    assert first.decision.semantic_content_hash == replay.decision.semantic_content_hash
    assert first.state.semantic_content_hash == replay.state.semantic_content_hash
    np.testing.assert_allclose(
        first.decision.selections[0].motion_activity_level,
        replay.decision.selections[0].motion_activity_level,
    )
