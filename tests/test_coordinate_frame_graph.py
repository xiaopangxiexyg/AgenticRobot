from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robot_motion.action import (
    ActionPayloadSchema,
    bind_action_payload_coordinates,
)
from robot_motion.coordinate_frames import (
    CoordinateFrameContractError,
    CoordinateFrameGraph,
    CoordinateFrameSpec,
    RigidFrameTransformSpec,
)


def _transform(
    translation: tuple[float, float, float],
    *,
    rotation_xyz_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_euler(
        "xyz",
        rotation_xyz_deg,
        degrees=True,
    ).as_matrix()
    result[:3, 3] = translation
    return result


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


def _pose_schema(
    *,
    reference_frame: str = "canonical_desk",
    controlled_frame: str = "tool_tcp",
) -> ActionPayloadSchema:
    return ActionPayloadSchema(
        schema_id="test-pose",
        schema_version="development",
        representation_family="cartesian_pose",
        dtype="float64",
        shape=(7,),
        coordinate_names=("x", "y", "z", "qx", "qy", "qz", "qw"),
        units=("m", "m", "m", "1", "1", "1", "1"),
        value_semantics="absolute_target",
        reference_frame=reference_frame,
        controlled_frame=controlled_frame,
    )


def test_one_new_frame_edge_connects_to_every_existing_frame() -> None:
    graph = CoordinateFrameGraph(
        graph_id="sim-hil-coordinate-profile",
        frames=(
            CoordinateFrameSpec("canonical_desk"),
            CoordinateFrameSpec("robot_base"),
        ),
        transforms=(
            _edge(
                "desk-to-base",
                "canonical_desk",
                "robot_base",
                _transform((1.0, 0.0, 0.0)),
            ),
        ),
    )
    extended = graph.with_connected_frame(
        frame=CoordinateFrameSpec(
            "vive_world",
            semantic_parameters={"owner": "vive_coordinate_profile"},
        ),
        transform=_edge(
            "vive-to-desk",
            "vive_world",
            "canonical_desk",
            _transform((0.0, 2.0, 0.0)),
        ),
    )

    resolved = extended.resolve("vive_world", "robot_base")

    np.testing.assert_allclose(
        resolved.matrix_target_source,
        _transform((1.0, 2.0, 0.0)),
    )
    assert tuple(item.transform_id for item in resolved.traversals) == (
        "vive-to-desk",
        "desk-to-base",
    )
    assert resolved.frame_graph_hash == extended.semantic_content_hash
    assert graph.semantic_content_hash != extended.semantic_content_hash


def test_resolution_supports_inverse_paths() -> None:
    transform = _transform(
        (0.4, -0.2, 0.1),
        rotation_xyz_deg=(0.0, 0.0, 30.0),
    )
    graph = CoordinateFrameGraph(
        graph_id="inverse-path",
        frames=(
            CoordinateFrameSpec("sensor"),
            CoordinateFrameSpec("task"),
        ),
        transforms=(_edge("sensor-to-task", "sensor", "task", transform),),
    )

    reverse = graph.resolve("task", "sensor")

    np.testing.assert_allclose(
        reverse.matrix_target_source,
        np.linalg.inv(transform),
    )
    assert len(reverse.traversals) == 1
    assert reverse.traversals[0].direction == "inverse"


def test_pose_transform_uses_resolved_reference_frame_path() -> None:
    world_to_desk = _transform(
        (1.0, 2.0, 0.0),
        rotation_xyz_deg=(0.0, 0.0, 90.0),
    )
    graph = CoordinateFrameGraph(
        graph_id="pose-transform",
        frames=(
            CoordinateFrameSpec("vive_world"),
            CoordinateFrameSpec("canonical_desk"),
        ),
        transforms=(
            _edge(
                "vive-to-desk",
                "vive_world",
                "canonical_desk",
                world_to_desk,
            ),
        ),
    )
    tracker_pose = _transform((0.5, 0.0, 0.0))

    transformed, receipt = graph.transform_pose_matrix(
        tracker_pose,
        source_reference_frame_id="vive_world",
        target_reference_frame_id="canonical_desk",
    )

    np.testing.assert_allclose(transformed, world_to_desk @ tracker_pose)
    assert receipt.source_frame_id == "vive_world"
    assert receipt.target_frame_id == "canonical_desk"


def test_pose_transform_can_change_reference_and_controlled_frames() -> None:
    world_to_desk = _transform(
        (1.0, 2.0, 0.0),
        rotation_xyz_deg=(0.0, 0.0, 90.0),
    )
    # T_tracker_tool: a point expressed in tool coordinates is mapped into
    # the tracker coordinates before the tracker pose is applied.
    tool_to_tracker = _transform((0.0, 0.0, 0.2))
    graph = CoordinateFrameGraph(
        graph_id="two-sided-pose-transform",
        frames=tuple(
            CoordinateFrameSpec(frame_id)
            for frame_id in (
                "vive_world",
                "canonical_desk",
                "wrist_tracker",
                "tool_tcp",
            )
        ),
        transforms=(
            _edge(
                "vive-to-desk",
                "vive_world",
                "canonical_desk",
                world_to_desk,
            ),
            _edge(
                "tool-to-tracker",
                "tool_tcp",
                "wrist_tracker",
                tool_to_tracker,
            ),
        ),
    )
    world_tracker = _transform(
        (0.5, 0.0, 0.0),
        rotation_xyz_deg=(0.0, 45.0, 0.0),
    )

    transformed, receipt = graph.transform_pose_between_frames(
        world_tracker,
        source_reference_frame_id="vive_world",
        source_controlled_frame_id="wrist_tracker",
        target_reference_frame_id="canonical_desk",
        target_controlled_frame_id="tool_tcp",
    )

    np.testing.assert_allclose(
        transformed,
        world_to_desk @ world_tracker @ tool_to_tracker,
    )
    assert receipt.reference_resolution.source_frame_id == "vive_world"
    assert receipt.reference_resolution.target_frame_id == "canonical_desk"
    assert receipt.controlled_resolution.source_frame_id == "tool_tcp"
    assert receipt.controlled_resolution.target_frame_id == "wrist_tracker"


def test_graph_rejects_an_inconsistent_cycle() -> None:
    frames = tuple(
        CoordinateFrameSpec(frame_id) for frame_id in ("a", "b", "c")
    )
    with pytest.raises(
        CoordinateFrameContractError,
        match="Inconsistent rigid-transform cycle",
    ):
        CoordinateFrameGraph(
            graph_id="bad-cycle",
            frames=frames,
            transforms=(
                _edge("a-b", "a", "b", _transform((1.0, 0.0, 0.0))),
                _edge("b-c", "b", "c", _transform((1.0, 0.0, 0.0))),
                _edge("a-c-wrong", "a", "c", _transform((3.0, 0.0, 0.0))),
            ),
        )


def test_graph_accepts_a_consistent_redundant_transform() -> None:
    graph = CoordinateFrameGraph(
        graph_id="consistent-cycle",
        frames=tuple(
            CoordinateFrameSpec(frame_id) for frame_id in ("a", "b", "c")
        ),
        transforms=(
            _edge("a-b", "a", "b", _transform((1.0, 0.0, 0.0))),
            _edge("b-c", "b", "c", _transform((1.0, 0.0, 0.0))),
            _edge("a-c", "a", "c", _transform((2.0, 0.0, 0.0))),
        ),
    )

    np.testing.assert_allclose(
        graph.resolve("a", "c").matrix_target_source,
        _transform((2.0, 0.0, 0.0)),
    )


def test_graph_hash_and_resolution_are_input_order_independent() -> None:
    frames = (
        CoordinateFrameSpec("a"),
        CoordinateFrameSpec("b"),
        CoordinateFrameSpec("c"),
    )
    edges = (
        _edge("a-b", "a", "b", _transform((1.0, 0.0, 0.0))),
        _edge("b-c", "b", "c", _transform((0.0, 1.0, 0.0))),
    )
    first = CoordinateFrameGraph(
        graph_id="ordered",
        frames=frames,
        transforms=edges,
    )
    second = CoordinateFrameGraph(
        graph_id="ordered",
        frames=tuple(reversed(frames)),
        transforms=tuple(reversed(edges)),
    )

    assert first.semantic_content_hash == second.semantic_content_hash
    assert (
        first.resolve("a", "c").semantic_content_hash
        == second.resolve("a", "c").semantic_content_hash
    )


def test_graph_payload_round_trip_preserves_exact_identity() -> None:
    graph = CoordinateFrameGraph(
        graph_id="serializable-profile",
        frames=(
            CoordinateFrameSpec(
                "source",
                semantic_parameters={"profile": "source-v1"},
            ),
            CoordinateFrameSpec("target"),
        ),
        transforms=(
            RigidFrameTransformSpec(
                transform_id="source-to-target",
                source_frame_id="source",
                target_frame_id="target",
                matrix_target_source=_transform((0.1, 0.2, 0.3)),
                semantic_parameters={"artifact_hash": "abc"},
            ),
        ),
    )

    restored = CoordinateFrameGraph.from_payload(graph.to_payload())

    assert restored.semantic_content_hash == graph.semantic_content_hash
    assert restored.to_payload() == graph.to_payload()
    np.testing.assert_allclose(
        restored.resolve("source", "target").matrix_target_source,
        graph.resolve("source", "target").matrix_target_source,
    )


def test_graph_payload_rejects_tampered_transform_content() -> None:
    graph = CoordinateFrameGraph(
        graph_id="tamper-check",
        frames=(
            CoordinateFrameSpec("source"),
            CoordinateFrameSpec("target"),
        ),
        transforms=(
            _edge(
                "source-to-target",
                "source",
                "target",
                _transform((0.1, 0.2, 0.3)),
            ),
        ),
    )
    payload = graph.to_payload()
    payload["transforms"][0]["matrix_target_source"][0][3] = 0.5

    with pytest.raises(
        CoordinateFrameContractError,
        match="rigid transform 'source-to-target' semantic_content_hash mismatch",
    ):
        CoordinateFrameGraph.from_payload(payload)


def test_disconnected_frames_are_explicitly_unresolvable() -> None:
    graph = CoordinateFrameGraph(
        graph_id="disconnected",
        frames=(
            CoordinateFrameSpec("a"),
            CoordinateFrameSpec("b"),
        ),
    )

    assert not graph.connected("a", "b")
    with pytest.raises(
        CoordinateFrameContractError,
        match="are not connected",
    ):
        graph.resolve("a", "b")


def test_frame_definition_and_transform_inputs_are_detached() -> None:
    parameters = {"source": ["calibration", 1]}
    transform = _transform((1.0, 2.0, 3.0))
    frame = CoordinateFrameSpec("frame", semantic_parameters=parameters)
    edge = _edge("frame-to-root", "frame", "root", transform)
    frame_hash = frame.semantic_content_hash
    edge_hash = edge.semantic_content_hash

    parameters["source"][0] = "mutated"
    transform[0, 3] = 100.0

    assert frame.semantic_content_hash == frame_hash
    assert frame.semantic_parameters["source"] == ("calibration", 1)
    assert edge.semantic_content_hash == edge_hash
    assert edge.matrix_target_source[0, 3] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        edge.matrix_target_source[0, 3] = 5.0


def test_connected_extension_must_link_new_and_existing_frames() -> None:
    graph = CoordinateFrameGraph(
        graph_id="extension",
        frames=(CoordinateFrameSpec("root"),),
    )
    with pytest.raises(
        CoordinateFrameContractError,
        match="already registered frame",
    ):
        graph.with_connected_frame(
            frame=CoordinateFrameSpec("new"),
            transform=_edge(
                "new-to-missing",
                "new",
                "missing",
                np.eye(4),
            ),
        )


def test_action_schema_frame_names_bind_to_exact_frame_definition_hashes() -> None:
    schema = _pose_schema()
    first_graph = CoordinateFrameGraph(
        graph_id="action-binding",
        frames=(
            CoordinateFrameSpec(
                "canonical_desk",
                semantic_parameters={"definition": "table-center-v1"},
            ),
            CoordinateFrameSpec(
                "tool_tcp",
                semantic_parameters={"definition": "tcp-v1"},
            ),
        ),
    )
    second_graph = CoordinateFrameGraph(
        graph_id="action-binding",
        frames=(
            CoordinateFrameSpec(
                "canonical_desk",
                semantic_parameters={"definition": "different-desk"},
            ),
            CoordinateFrameSpec(
                "tool_tcp",
                semantic_parameters={"definition": "tcp-v1"},
            ),
        ),
    )

    first = bind_action_payload_coordinates(schema, frame_graph=first_graph)
    second = bind_action_payload_coordinates(schema, frame_graph=second_graph)

    assert first.reference_frame is not None
    assert first.reference_frame.frame_id == schema.reference_frame
    assert first.controlled_frame is not None
    assert first.controlled_frame.frame_id == schema.controlled_frame
    assert first.action_payload_schema_hash == schema.semantic_content_hash
    assert first.semantic_content_hash != second.semantic_content_hash


def test_action_schema_semantic_frame_can_bind_component_scoped_frame() -> None:
    schema = _pose_schema(controlled_frame="tool_tcp")
    graph = CoordinateFrameGraph(
        graph_id="component-scoped-controlled-frame",
        frames=(
            CoordinateFrameSpec("canonical_desk"),
            CoordinateFrameSpec("left_arm/tool_tcp"),
        ),
    )

    binding = bind_action_payload_coordinates(
        schema,
        frame_graph=graph,
        controlled_frame_id="left_arm/tool_tcp",
    )

    assert binding.declared_controlled_frame_id == "tool_tcp"
    assert binding.controlled_frame is not None
    assert binding.controlled_frame.frame_id == "left_arm/tool_tcp"


def test_action_coordinate_binding_rejects_an_unregistered_schema_frame() -> None:
    graph = CoordinateFrameGraph(
        graph_id="missing-schema-frame",
        frames=(CoordinateFrameSpec("canonical_desk"),),
    )

    with pytest.raises(ValueError, match="'tool_tcp' is not registered"):
        bind_action_payload_coordinates(_pose_schema(), frame_graph=graph)


def test_non_spatial_joint_payload_does_not_get_a_spatial_binding() -> None:
    schema = ActionPayloadSchema(
        schema_id="hand-qpos",
        schema_version="development",
        representation_family="joint_position",
        dtype="float64",
        shape=(2,),
        coordinate_names=("finger_0", "finger_1"),
        units=("rad", "rad"),
        value_semantics="absolute_target",
    )
    graph = CoordinateFrameGraph(
        graph_id="joint-payload",
        frames=(CoordinateFrameSpec("canonical_desk"),),
    )

    with pytest.raises(ValueError, match="at least one spatial frame"):
        bind_action_payload_coordinates(schema, frame_graph=graph)
