from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robot_motion.action import (
    ActionComponentCatalogV2,
    ActionPayloadSchema,
    ComponentAction,
    ComponentActionSchema,
    ControlComponentSpecV2,
)
from robot_motion.coordinate_frames import (
    CoordinateFrameGraph,
    CoordinateFrameSpec,
    RigidFrameTransformSpec,
)
from teleop_runtime.action.human_residual import (
    HumanPoseSample,
    HumanResidualPoseAdapter,
)


def _transform(
    translation: tuple[float, float, float],
    *,
    rotation_xyz_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = Rotation.from_euler(
        "xyz",
        rotation_xyz_deg,
        degrees=True,
    ).as_matrix()
    value[:3, 3] = translation
    return value


def _edge(
    transform_id: str,
    source: str,
    target: str,
    matrix: np.ndarray,
) -> RigidFrameTransformSpec:
    return RigidFrameTransformSpec(
        transform_id=transform_id,
        source_frame_id=source,
        target_frame_id=target,
        matrix_target_source=matrix,
    )


def _graph(*, desk_x: float = 1.0) -> CoordinateFrameGraph:
    return CoordinateFrameGraph(
        graph_id="virtual-human-residual",
        frames=tuple(
            CoordinateFrameSpec(frame_id)
            for frame_id in (
                "vive_world",
                "operator",
                "canonical_desk",
                "wrist_tracker",
                "tool_tcp",
            )
        ),
        transforms=(
            _edge(
                "vive-to-operator",
                "vive_world",
                "operator",
                _transform((0.0, 2.0, 0.0)),
            ),
            _edge(
                "operator-to-desk",
                "operator",
                "canonical_desk",
                _transform((desk_x, 0.0, 0.0)),
            ),
            _edge(
                "tool-to-tracker",
                "tool_tcp",
                "wrist_tracker",
                _transform((0.0, 0.0, 0.2)),
            ),
        ),
    )


def _schema() -> ComponentActionSchema:
    payload = ActionPayloadSchema(
        schema_id="human-arm-residual",
        schema_version="development",
        representation_family="cartesian_pose",
        dtype="float64",
        shape=(7,),
        coordinate_names=("x", "y", "z", "qx", "qy", "qz", "qw"),
        units=("m", "m", "m", "1", "1", "1", "1"),
        value_semantics="relative_delta",
        reference_frame="canonical_desk",
        controlled_frame="tool_tcp",
        semantic_parameters={
            "translation_application": "reference_frame_additive",
            "rotation_application": "baseline_local_right_multiply",
        },
    )
    catalog = ActionComponentCatalogV2(
        catalog_id="virtual-human-residual",
        catalog_version="development",
        components=(
            ControlComponentSpecV2(
                component_id="arm",
                controlled_entity_id="arm",
                command_channel_id="arm",
                selection_group_id="arm",
                resource_claims=("resource:arm",),
                accepted_payload_schema_hashes=(
                    payload.semantic_content_hash,
                ),
            ),
        ),
    )
    return ComponentActionSchema(
        schema_id="virtual-human-residual",
        schema_version="development",
        component_catalog=catalog,
        payload_schema_by_component={"arm": payload},
    )


def _sample(
    sample_id: str,
    *,
    measured_at: float,
    pose: np.ndarray,
    binding_id: str = "vive-calibration:1",
    binding_hash: str = "vive-calibration-hash:1",
) -> HumanPoseSample:
    return HumanPoseSample(
        component_id="arm",
        sample_id=sample_id,
        measured_at=measured_at,
        measured_clock_domain_id="vive.source",
        source_reference_frame_id="vive_world",
        source_controlled_frame_id="wrist_tracker",
        source_coordinate_binding_id=binding_id,
        source_coordinate_binding_hash=binding_hash,
        pose_source_reference_source_controlled=pose,
    )


def test_reference_translation_and_local_rotation_residual() -> None:
    graph = _graph()
    adapter = HumanResidualPoseAdapter(
        output_action_schema=_schema(),
        frame_graph=graph,
        translation_gain_by_component={"arm": (2.0, 0.5, 1.0)},
    )
    anchor_sample = _sample(
        "anchor",
        measured_at=10.0,
        pose=_transform(
            (0.2, 0.1, 0.3),
            rotation_xyz_deg=(10.0, 20.0, 30.0),
        ),
    )
    current_sample = _sample(
        "current",
        measured_at=10.1,
        pose=_transform(
            (0.3, -0.1, 0.4),
            rotation_xyz_deg=(-5.0, 25.0, 45.0),
        ),
    )
    anchor = adapter.capture_anchor(
        (anchor_sample,),
        anchor_generation=3,
        captured_at=20.0,
        captured_clock_domain_id="receiver.monotonic",
    )

    result = adapter.adapt(
        (current_sample,),
        anchor=anchor,
        adapted_at=20.1,
        adapted_clock_domain_id="receiver.monotonic",
    )

    anchor_canonical, _ = graph.transform_pose_between_frames(
        anchor_sample.pose_source_reference_source_controlled,
        source_reference_frame_id="vive_world",
        source_controlled_frame_id="wrist_tracker",
        target_reference_frame_id="canonical_desk",
        target_controlled_frame_id="tool_tcp",
    )
    current_canonical, _ = graph.transform_pose_between_frames(
        current_sample.pose_source_reference_source_controlled,
        source_reference_frame_id="vive_world",
        source_controlled_frame_id="wrist_tracker",
        target_reference_frame_id="canonical_desk",
        target_controlled_frame_id="tool_tcp",
    )
    value = np.asarray(result.relative_action.value("arm").values)
    np.testing.assert_allclose(
        value[:3],
        (current_canonical[:3, 3] - anchor_canonical[:3, 3])
        * np.array([2.0, 0.5, 1.0]),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        Rotation.from_quat(value[3:7]).as_matrix(),
        anchor_canonical[:3, :3].T @ current_canonical[:3, :3],
        atol=1e-7,
    )
    assert result.receipt.anchor_generation == 3
    assert result.receipt.anchor_id == anchor.anchor_id
    assert result.receipt.frame_graph_hash == graph.semantic_content_hash
    assert result.receipt.output_action_hash == (
        result.relative_action.semantic_content_hash
    )


def test_same_sample_and_anchor_produce_identity_without_accumulation() -> None:
    adapter = HumanResidualPoseAdapter(
        output_action_schema=_schema(),
        frame_graph=_graph(),
    )
    sample = _sample(
        "same",
        measured_at=10.0,
        pose=_transform(
            (0.2, 0.1, 0.3),
            rotation_xyz_deg=(10.0, 20.0, 30.0),
        ),
    )
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

    np.testing.assert_allclose(
        first.relative_action.value("arm").values,
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        atol=1e-7,
    )
    assert (
        first.relative_action.semantic_content_hash
        == second.relative_action.semantic_content_hash
    )
    assert first.receipt.semantic_content_hash != second.receipt.semantic_content_hash


def test_from_component_action_uses_typed_absolute_pose_and_concrete_frames() -> None:
    payload = ActionPayloadSchema(
        schema_id="human-arm-absolute",
        schema_version="development",
        representation_family="cartesian_pose",
        dtype="float64",
        shape=(7,),
        coordinate_names=("x", "y", "z", "qx", "qy", "qz", "qw"),
        units=("m", "m", "m", "1", "1", "1", "1"),
        value_semantics="absolute_target",
        reference_frame="arm_base",
        controlled_frame="tool_tcp",
    )
    catalog = ActionComponentCatalogV2(
        catalog_id="human-arm-absolute",
        catalog_version="development",
        components=(
            ControlComponentSpecV2(
                component_id="arm",
                controlled_entity_id="arm",
                command_channel_id="arm",
                selection_group_id="arm",
                resource_claims=("resource:arm",),
                accepted_payload_schema_hashes=(
                    payload.semantic_content_hash,
                ),
            ),
        ),
    )
    rotation = Rotation.from_euler("xyz", (10.0, 20.0, 30.0), degrees=True)
    action = ComponentAction(
        action_schema=ComponentActionSchema(
            schema_id="human-arm-absolute",
            schema_version="development",
            component_catalog=catalog,
            payload_schema_by_component={"arm": payload},
        ),
        values_by_component={
            "arm": np.r_[[0.1, 0.2, 0.3], rotation.as_quat()],
        },
    )

    sample = HumanPoseSample.from_component_action(
        action,
        "arm",
        sample_id="absolute-pose",
        measured_at=10.0,
        measured_clock_domain_id="receiver.monotonic",
        source_coordinate_binding_id="retarget-binding:1",
        source_coordinate_binding_hash="retarget-binding-hash:1",
        concrete_reference_frame_id="left_arm/base",
        concrete_controlled_frame_id="left_arm/tool_tcp",
    )

    np.testing.assert_allclose(
        sample.pose_source_reference_source_controlled[:3, 3],
        (0.1, 0.2, 0.3),
    )
    np.testing.assert_allclose(
        sample.pose_source_reference_source_controlled[:3, :3],
        rotation.as_matrix(),
    )
    assert sample.source_reference_frame_id == "left_arm/base"
    assert sample.source_controlled_frame_id == "left_arm/tool_tcp"


def test_old_anchor_is_rejected_after_coordinate_graph_change() -> None:
    old_adapter = HumanResidualPoseAdapter(
        output_action_schema=_schema(),
        frame_graph=_graph(desk_x=1.0),
    )
    sample = _sample("anchor", measured_at=10.0, pose=np.eye(4))
    anchor = old_adapter.capture_anchor(
        (sample,),
        anchor_generation=0,
        captured_at=20.0,
        captured_clock_domain_id="receiver.monotonic",
    )
    new_adapter = HumanResidualPoseAdapter(
        output_action_schema=_schema(),
        frame_graph=_graph(desk_x=1.1),
    )

    with pytest.raises(ValueError, match="another coordinate graph"):
        new_adapter.adapt(
            (_sample("current", measured_at=10.1, pose=np.eye(4)),),
            anchor=anchor,
            adapted_at=20.1,
            adapted_clock_domain_id="receiver.monotonic",
        )


def test_source_coordinate_binding_must_match_anchor() -> None:
    adapter = HumanResidualPoseAdapter(
        output_action_schema=_schema(),
        frame_graph=_graph(),
    )
    anchor_sample = _sample("anchor", measured_at=10.0, pose=np.eye(4))
    anchor = adapter.capture_anchor(
        (anchor_sample,),
        anchor_generation=1,
        captured_at=20.0,
        captured_clock_domain_id="receiver.monotonic",
    )
    changed = _sample(
        "current",
        measured_at=10.1,
        pose=np.eye(4),
        binding_id="vive-calibration:2",
        binding_hash="vive-calibration-hash:2",
    )

    with pytest.raises(
        ValueError,
        match="source_coordinate_binding_id",
    ):
        adapter.adapt(
            (changed,),
            anchor=anchor,
            adapted_at=20.1,
            adapted_clock_domain_id="receiver.monotonic",
        )


def test_residual_schema_tool_semantics_can_bind_scoped_physical_tool() -> None:
    graph = CoordinateFrameGraph(
        graph_id="scoped-tool",
        frames=tuple(
            CoordinateFrameSpec(frame_id)
            for frame_id in (
                "vive_world",
                "canonical_desk",
                "wrist_tracker",
                "left_arm/tool_tcp",
            )
        ),
        transforms=(
            _edge(
                "vive-to-desk",
                "vive_world",
                "canonical_desk",
                np.eye(4),
            ),
            _edge(
                "left-tool-to-tracker",
                "left_arm/tool_tcp",
                "wrist_tracker",
                np.eye(4),
            ),
        ),
    )
    adapter = HumanResidualPoseAdapter(
        output_action_schema=_schema(),
        frame_graph=graph,
        concrete_controlled_frame_by_component={
            "arm": "left_arm/tool_tcp",
        },
    )
    sample = _sample("anchor", measured_at=10.0, pose=np.eye(4))

    anchor = adapter.capture_anchor(
        (sample,),
        anchor_generation=0,
        captured_at=20.0,
        captured_clock_domain_id="receiver.monotonic",
    )

    assert anchor.entries[0].canonical_controlled_frame_id == (
        "left_arm/tool_tcp"
    )
    binding = adapter.action_coordinate_binding_by_component["arm"]
    assert binding.declared_controlled_frame_id == "tool_tcp"
    assert binding.controlled_frame is not None
    assert binding.controlled_frame.frame_id == "left_arm/tool_tcp"


def test_sample_cannot_predate_recenter_anchor() -> None:
    adapter = HumanResidualPoseAdapter(
        output_action_schema=_schema(),
        frame_graph=_graph(),
    )
    anchor = adapter.capture_anchor(
        (_sample("anchor", measured_at=10.0, pose=np.eye(4)),),
        anchor_generation=1,
        captured_at=20.0,
        captured_clock_domain_id="receiver.monotonic",
    )

    with pytest.raises(ValueError, match="predates its recenter anchor"):
        adapter.adapt(
            (_sample("old", measured_at=9.9, pose=np.eye(4)),),
            anchor=anchor,
            adapted_at=20.1,
            adapted_clock_domain_id="receiver.monotonic",
        )


def test_adaptation_cannot_predate_anchor_on_the_same_receiver_clock() -> None:
    adapter = HumanResidualPoseAdapter(
        output_action_schema=_schema(),
        frame_graph=_graph(),
    )
    sample = _sample("anchor", measured_at=10.0, pose=np.eye(4))
    anchor = adapter.capture_anchor(
        (sample,),
        anchor_generation=1,
        captured_at=20.0,
        captured_clock_domain_id="receiver.monotonic",
    )

    with pytest.raises(ValueError, match="predates its anchor capture"):
        adapter.adapt(
            (sample,),
            anchor=anchor,
            adapted_at=19.9,
            adapted_clock_domain_id="receiver.monotonic",
        )


def test_input_arrays_are_detached_from_samples_and_anchors() -> None:
    adapter = HumanResidualPoseAdapter(
        output_action_schema=_schema(),
        frame_graph=_graph(),
    )
    pose = _transform((0.1, 0.2, 0.3))
    sample = _sample("anchor", measured_at=10.0, pose=pose)
    anchor = adapter.capture_anchor(
        (sample,),
        anchor_generation=0,
        captured_at=20.0,
        captured_clock_domain_id="receiver.monotonic",
    )
    sample_hash = sample.semantic_content_hash
    anchor_hash = anchor.semantic_content_hash

    pose[0, 3] = 99.0

    assert sample.semantic_content_hash == sample_hash
    assert sample.pose_source_reference_source_controlled[0, 3] == pytest.approx(
        0.1
    )
    assert anchor.semantic_content_hash == anchor_hash
    assert anchor.entries[0].pose_canonical_reference_canonical_controlled.flags.writeable is False
