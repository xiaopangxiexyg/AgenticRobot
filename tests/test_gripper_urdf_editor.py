from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from utils.gripper_urdf_editor.core import URDFDocument, URDFValidationError, get_origin
from utils.gripper_urdf_editor.mesh import MeshCache
from utils.gripper_urdf_editor.viewer import build_gripper_scene


ROOT = Path(__file__).resolve().parents[1]
PROXY = ROOT / "assets/inspire_gripper/urdf/eg2_4b2_parallel_proxy.urdf"
TEMPLATE = ROOT / "assets/inspire_gripper/templates/parallel_gripper_translated_end_component_template.urdf"
DUAL = ROOT / "assets/rm65_inspire_gripper_v3/rm65_dual_arm_wrist_cameras_v3_head_camera.urdf"


def _selection(path: Path):
    document = URDFDocument.load(path)
    pair = document.detect_gripper_pairs()[0]
    endpoint_a = document.default_endpoint(pair.joint_a)
    endpoint_b = document.default_endpoint(pair.joint_b)
    return document, pair, endpoint_a, endpoint_b


def test_detects_proxy_jaws_and_opening_semantics() -> None:
    document, pair, endpoint_a, endpoint_b = _selection(PROXY)

    assert pair.parent_link == "base_link"
    assert endpoint_a.link_name == "left_finger"
    assert endpoint_b.link_name == "right_finger"
    assert endpoint_a.geometry.role == "collision"
    assert endpoint_b.geometry.role == "collision"
    closed, opened = document.inferred_open_closed(pair, endpoint_a, endpoint_b)
    assert closed == pytest.approx(-0.035)
    assert opened == pytest.approx(0.0)
    assert document.opening_gap(pair, endpoint_a, endpoint_b, closed) == pytest.approx(0.0, abs=1e-12)
    assert document.opening_gap(pair, endpoint_a, endpoint_b, opened) == pytest.approx(0.070)


def test_detects_two_independent_grippers_in_dual_arm_urdf() -> None:
    document = URDFDocument.load(DUAL)
    pairs = document.detect_gripper_pairs()

    assert [pair.parent_link for pair in pairs] == ["left_gripper_base_link", "right_gripper_base_link"]
    assert all("mimic coupled" in pair.reason for pair in pairs)


def test_template_selects_terminal_component_and_fixed_mount_pose() -> None:
    document, pair, endpoint_a, endpoint_b = _selection(TEMPLATE)

    assert endpoint_a.link_name == "left_end_component"
    assert endpoint_b.link_name == "right_end_component"
    assert endpoint_a.geometry.role == "visual"
    assert document.pose_target(endpoint_a).owner_name == "left_end_component_mount_joint"
    assert document.pose_target(endpoint_b).owner_name == "right_end_component_mount_joint"
    assert get_origin(document.pose_target(endpoint_a).element)[0] == pytest.approx((0.0, 0.0, 0.071164))


def test_endpoint_resize_can_sync_visual_collision_and_contact_frames() -> None:
    document, pair, endpoint_a, endpoint_b = _selection(TEMPLATE)
    dimensions = (0.020, 0.030, 0.040)
    for endpoint in (endpoint_a, endpoint_b):
        document.set_geometry_dimensions(endpoint.geometry, dimensions)
        matching = document.matching_geometry(endpoint.geometry)
        assert matching is not None
        document.set_geometry_dimensions(matching, dimensions)

    assert document.snap_contact_frame_to_inner_face(pair, endpoint_a, endpoint_b, 0.0)
    assert document.snap_contact_frame_to_inner_face(pair, endpoint_b, endpoint_a, 0.0)
    assert document.opening_gap(pair, endpoint_a, endpoint_b, 0.0) == pytest.approx(0.060)
    assert document.opening_gap(pair, endpoint_a, endpoint_b, -0.035) == pytest.approx(-0.010)
    assert document.geometry_dimensions(endpoint_a.geometry) == pytest.approx(dimensions)
    assert document.geometry_dimensions(document.matching_geometry(endpoint_a.geometry)) == pytest.approx(dimensions)


def test_pair_limit_edit_updates_master_and_mimic_limits() -> None:
    document, pair, _endpoint_a, _endpoint_b = _selection(PROXY)
    document.set_pair_limits(pair, closed_q=-0.030, open_q=0.005)

    assert document.joint_limits("left_finger_joint") == pytest.approx((-0.030, 0.005))
    assert document.joint_limits("right_finger_joint") == pytest.approx((-0.030, 0.005))
    document.validate()


def test_mesh_dimensions_are_available_without_optional_render_dependencies() -> None:
    document = URDFDocument.load(PROXY)
    cache = MeshCache(max_triangles=20)
    mesh_ref = next(ref for ref in document.geometry_refs("left_finger") if ref.role == "visual")

    assert document.geometry_dimensions(mesh_ref, cache.dimensions) == pytest.approx((0.010, 0.0255, 0.034), rel=1e-5)


def test_proxy_box_and_matching_mesh_resize_together_around_same_center() -> None:
    document, _pair, endpoint_a, _endpoint_b = _selection(PROXY)
    cache = MeshCache(max_triangles=20)
    matching_visual = document.matching_geometry(endpoint_a.geometry, cache.dimensions)

    assert matching_visual is not None
    assert matching_visual.role == "visual"
    assert matching_visual.element.find("geometry/mesh") is not None
    transforms = document.link_transforms()
    center_before = document.geometry_transform(matching_visual, transforms)
    document.set_geometry_dimensions(
        matching_visual,
        (0.020, 0.030, 0.068),
        cache.dimensions,
        cache.center,
    )
    center_after = document.geometry_transform(matching_visual, transforms)
    mesh_center = cache.center(document.resolve_mesh_filename(matching_visual.element.find("geometry/mesh").get("filename")))
    mesh = matching_visual.element.find("geometry/mesh")
    scale = tuple(float(value) for value in mesh.get("scale").split())
    from utils.gripper_urdf_editor.core import transform_point
    point_before = transform_point(center_before, mesh_center)
    point_after = transform_point(center_after, tuple(mesh_center[index] * scale[index] for index in range(3)))

    assert document.geometry_dimensions(matching_visual, cache.dimensions) == pytest.approx((0.020, 0.030, 0.068))
    assert point_after == pytest.approx(point_before, abs=1e-9)


def test_export_is_new_atomic_urdf_and_rebases_relative_meshes(tmp_path: Path) -> None:
    document, pair, endpoint_a, endpoint_b = _selection(PROXY)
    original_bytes = PROXY.read_bytes()
    document.set_geometry_dimensions(endpoint_a.geometry, (0.012, 0.028, 0.040))
    document.set_geometry_dimensions(endpoint_b.geometry, (0.012, 0.028, 0.040))
    target = tmp_path / "edited_proxy.urdf"

    output = document.export(target)

    assert output == target.resolve()
    assert PROXY.read_bytes() == original_bytes
    parsed = ET.parse(output)
    first_mesh = parsed.find(".//mesh")
    assert first_mesh is not None
    rebased = (output.parent / first_mesh.get("filename", "")).resolve()
    source_mesh = document.resolve_mesh_filename("../meshes/eg2_4b2_proxy/eg2_4b2_body_fixed_visual.stl")
    assert rebased == source_mesh
    exported = URDFDocument.load(output)
    exported_pair = exported.detect_gripper_pairs()[0]
    assert exported.default_endpoint(exported_pair.joint_a).geometry.element.find("geometry/box").get("size") == "0.012 0.028 0.04"


def test_export_refuses_input_path_and_existing_target(tmp_path: Path) -> None:
    document = URDFDocument.load(PROXY)
    with pytest.raises(URDFValidationError, match="Refusing to overwrite"):
        document.export(PROXY, overwrite=True)

    target = tmp_path / "existing.urdf"
    target.write_text("keep", encoding="utf-8")
    with pytest.raises(URDFValidationError, match="already exists"):
        document.export(target)
    assert target.read_text(encoding="utf-8") == "keep"


def test_scene_reports_unreadable_geometry_as_warning() -> None:
    document, pair, endpoint_a, endpoint_b = _selection(PROXY)

    class UnreadableMeshCache(MeshCache):
        def geometry_triangles(self, document, ref):
            raise URDFValidationError("Preview supports STL/OBJ meshes, not .dae")

    shapes, _markers, warnings = build_gripper_scene(
        document,
        pair,
        endpoint_a,
        endpoint_b,
        0.0,
        UnreadableMeshCache(),
        show_visuals=True,
        show_collisions=True,
        show_contact_frames=False,
    )

    assert shapes == []
    assert warnings
    assert all("Preview supports STL/OBJ meshes" in warning for warning in warnings)
