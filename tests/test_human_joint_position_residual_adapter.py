from __future__ import annotations

import numpy as np
import pytest

from robot_motion.action import (
    ActionComponentCatalogV2,
    ActionPayloadSchema,
    ComponentAction,
    ComponentActionSchema,
    ControlComponentSpecV2,
)
from teleop_runtime.action.human_joint_residual import (
    HumanJointPositionResidualAdapter,
    HumanJointPositionSample,
)


def _payload(
    *,
    schema_id: str,
    semantics: str,
    coordinates: tuple[str, ...] = ("finger_0", "finger_1", "finger_2"),
) -> ActionPayloadSchema:
    return ActionPayloadSchema(
        schema_id=schema_id,
        schema_version="development",
        representation_family="joint_position",
        dtype="float64",
        shape=(len(coordinates),),
        coordinate_names=coordinates,
        units=tuple("rad" for _ in coordinates),
        value_semantics=semantics,
        semantic_parameters=(
            {"application": "elementwise_additive"}
            if semantics == "relative_delta"
            else {}
        ),
    )


def _schemas() -> tuple[
    ActionPayloadSchema,
    ActionPayloadSchema,
    ComponentActionSchema,
]:
    absolute = _payload(schema_id="human-hand-absolute", semantics="absolute_target")
    delta = _payload(schema_id="human-hand-delta", semantics="relative_delta")
    catalog = ActionComponentCatalogV2(
        catalog_id="human-hand-residual",
        catalog_version="development",
        components=(
            ControlComponentSpecV2(
                component_id="hand",
                controlled_entity_id="hand",
                command_channel_id="hand",
                selection_group_id="hand",
                resource_claims=("resource:hand",),
                accepted_payload_schema_hashes=(
                    absolute.semantic_content_hash,
                    delta.semantic_content_hash,
                ),
            ),
        ),
    )
    return (
        absolute,
        delta,
        ComponentActionSchema(
            schema_id="human-hand-residual",
            schema_version="development",
            component_catalog=catalog,
            payload_schema_by_component={"hand": delta},
        ),
    )


def _sample(
    sample_id: str,
    measured_at: float,
    values: tuple[float, ...],
    *,
    binding_id: str = "q6-retarget:calibration-1",
    binding_hash: str = "q6-retarget-hash:calibration-1",
) -> HumanJointPositionSample:
    absolute, _, _ = _schemas()
    return HumanJointPositionSample(
        component_id="hand",
        sample_id=sample_id,
        measured_at=measured_at,
        measured_clock_domain_id="q6.source",
        source_value_binding_id=binding_id,
        source_value_binding_hash=binding_hash,
        source_payload_schema=absolute,
        values=values,
    )


def _adapter(
    *,
    gain: object = 1.0,
) -> HumanJointPositionResidualAdapter:
    absolute, _, delta_schema = _schemas()
    return HumanJointPositionResidualAdapter(
        output_action_schema=delta_schema,
        source_payload_schema_by_component={"hand": absolute},
        gain_by_component={"hand": gain},
    )


def test_absolute_joint_target_is_anchor_subtracted_with_per_joint_gain() -> None:
    adapter = _adapter(gain=(2.0, 0.5, -1.0))
    anchor = adapter.capture_anchor(
        (_sample("anchor", 10.0, (0.2, 0.4, 0.6)),),
        anchor_generation=4,
        captured_at=20.0,
        captured_clock_domain_id="receiver.monotonic",
    )

    result = adapter.adapt(
        (_sample("current", 10.1, (0.3, 0.2, 0.9)),),
        anchor=anchor,
        adapted_at=20.1,
        adapted_clock_domain_id="receiver.monotonic",
    )

    np.testing.assert_allclose(
        result.relative_action.value("hand").values,
        (0.2, -0.1, -0.3),
        atol=1e-12,
    )
    assert result.receipt.anchor_generation == 4
    assert result.receipt.anchor_id == anchor.anchor_id
    assert result.receipt.output_action_hash == (
        result.relative_action.semantic_content_hash
    )


def test_same_absolute_sample_is_zero_and_never_integrates() -> None:
    adapter = _adapter()
    sample = _sample("same", 10.0, (0.2, 0.4, 0.6))
    anchor = adapter.capture_anchor(
        (sample,),
        anchor_generation=0,
        captured_at=20.0,
        captured_clock_domain_id="receiver.monotonic",
    )

    first = adapter.adapt(
        (sample,),
        anchor=anchor,
        adapted_at=20.1,
        adapted_clock_domain_id="receiver.monotonic",
    )
    second = adapter.adapt(
        (sample,),
        anchor=anchor,
        adapted_at=20.2,
        adapted_clock_domain_id="receiver.monotonic",
    )

    np.testing.assert_array_equal(
        first.relative_action.value("hand").values,
        np.zeros(3),
    )
    assert (
        first.relative_action.semantic_content_hash
        == second.relative_action.semantic_content_hash
    )
    assert first.receipt.semantic_content_hash != second.receipt.semantic_content_hash


def test_changed_retarget_or_calibration_binding_rejects_old_anchor() -> None:
    adapter = _adapter()
    anchor = adapter.capture_anchor(
        (_sample("anchor", 10.0, (0.2, 0.4, 0.6)),),
        anchor_generation=0,
        captured_at=20.0,
        captured_clock_domain_id="receiver.monotonic",
    )

    with pytest.raises(ValueError, match="source_value_binding"):
        adapter.adapt(
            (
                _sample(
                    "current",
                    10.1,
                    (0.3, 0.5, 0.7),
                    binding_id="q6-retarget:calibration-2",
                    binding_hash="q6-retarget-hash:calibration-2",
                ),
            ),
            anchor=anchor,
            adapted_at=20.1,
            adapted_clock_domain_id="receiver.monotonic",
        )


def test_sample_older_than_anchor_and_adaptation_before_capture_are_rejected() -> None:
    adapter = _adapter()
    anchor = adapter.capture_anchor(
        (_sample("anchor", 10.0, (0.2, 0.4, 0.6)),),
        anchor_generation=0,
        captured_at=20.0,
        captured_clock_domain_id="receiver.monotonic",
    )

    with pytest.raises(ValueError, match="predates its recenter anchor"):
        adapter.adapt(
            (_sample("old", 9.9, (0.1, 0.3, 0.5)),),
            anchor=anchor,
            adapted_at=20.1,
            adapted_clock_domain_id="receiver.monotonic",
        )
    with pytest.raises(ValueError, match="predates its anchor capture"):
        adapter.adapt(
            (_sample("current", 10.1, (0.3, 0.5, 0.7)),),
            anchor=anchor,
            adapted_at=19.9,
            adapted_clock_domain_id="receiver.monotonic",
        )


def test_from_component_action_selects_only_requested_absolute_component() -> None:
    absolute, _, _ = _schemas()
    catalog = ActionComponentCatalogV2(
        catalog_id="absolute-with-unrelated",
        catalog_version="development",
        components=(
            ControlComponentSpecV2(
                component_id="hand",
                controlled_entity_id="hand",
                command_channel_id="hand",
                selection_group_id="hand",
                resource_claims=("resource:hand",),
                accepted_payload_schema_hashes=(
                    absolute.semantic_content_hash,
                ),
            ),
        ),
    )
    action = ComponentAction(
        action_schema=ComponentActionSchema(
            schema_id="absolute-with-unrelated",
            schema_version="development",
            component_catalog=catalog,
            payload_schema_by_component={"hand": absolute},
        ),
        values_by_component={"hand": (0.1, 0.2, 0.3)},
    )

    sample = HumanJointPositionSample.from_component_action(
        action,
        "hand",
        sample_id="sample-1",
        measured_at=10.0,
        measured_clock_domain_id="q6.source",
        source_value_binding_id="q6-retarget:1",
        source_value_binding_hash="q6-retarget-hash:1",
    )

    np.testing.assert_array_equal(sample.values, (0.1, 0.2, 0.3))
    assert sample.source_payload_schema.semantic_content_hash == (
        absolute.semantic_content_hash
    )


def test_layout_mismatch_is_rejected_before_runtime() -> None:
    absolute, _, delta_schema = _schemas()
    wrong = _payload(
        schema_id="human-hand-wrong-order",
        semantics="absolute_target",
        coordinates=("finger_1", "finger_0", "finger_2"),
    )
    assert wrong.semantic_content_hash != absolute.semantic_content_hash

    with pytest.raises(ValueError, match="layout disagree"):
        HumanJointPositionResidualAdapter(
            output_action_schema=delta_schema,
            source_payload_schema_by_component={"hand": wrong},
        )


def test_sample_and_anchor_values_are_immutable_copies() -> None:
    absolute, _, _ = _schemas()
    raw = np.asarray((0.1, 0.2, 0.3), dtype=np.float64)
    sample = HumanJointPositionSample(
        component_id="hand",
        sample_id="sample",
        measured_at=10.0,
        measured_clock_domain_id="q6.source",
        source_value_binding_id="q6-retarget:1",
        source_value_binding_hash="q6-retarget-hash:1",
        source_payload_schema=absolute,
        values=raw,
    )
    raw[:] = 9.0

    np.testing.assert_array_equal(sample.values, (0.1, 0.2, 0.3))
    assert not sample.values.flags.writeable
