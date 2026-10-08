from dataclasses import asdict, replace
from types import SimpleNamespace
import hashlib
import io
import zipfile

import numpy as np
import pytest

from dataset_contracts.grasp_multiview_snapshot import (
    GRASP_MULTIVIEW_SCHEMA_ID, decode_grasp_multiview_snapshot,
    encode_grasp_multiview_snapshot,
)
from dataset_contracts.grasp_rgbd_snapshot import (
    GRASP_RGBD_SNAPSHOT_SCHEMA_ID, encode_grasp_rgbd_snapshot,
)
from deployments.perception.grasp_multiview import GraspRgbdView, fuse_grasp_views
from robot_motion.grasp_visible_scene import ParallelJawSolidGeometry


GEOMETRY = ParallelJawSolidGeometry(-0.043, 0.05, 0.01, 0.0255, 0.07)


def view(role="main", **changes):
    return replace(GraspRgbdView(
        role=role, rgb=np.full((32, 32, 3), 80, dtype=np.uint8),
        depth_m=np.full((32, 32), 0.5, dtype=np.float32),
        intrinsics={"width": 32, "height": 32, "fx": 100., "fy": 100., "cx": 16., "cy": 16.},
        reference_from_camera=np.eye(4), acquired_at=10.,
    ), **changes)


def mask():
    result = np.zeros((32, 32), dtype=bool)
    result[10:22, 10:22] = True
    return result


def blob(role="main"):
    sample = view(role)
    return encode_grasp_rgbd_snapshot(
        rgb=sample.rgb, depth_m=sample.depth_m, target_mask=mask(),
        metadata={"schema_id": GRASP_RGBD_SNAPSHOT_SCHEMA_ID, "role": role},
        qpos_by_side={"left": np.zeros(6), "right": np.zeros(6)},
    )


def fuse(anchor=None, auxiliary=(), segment=None):
    return fuse_grasp_views(
        anchor=view() if anchor is None else anchor, anchor_mask=mask(),
        auxiliary=auxiliary, geometry=GEOMETRY,
        segment_point=(lambda *_args, **_kw: mask()) if segment is None else segment,
    )


def test_any_view_can_anchor_and_single_view_is_reported():
    result = fuse(anchor=view("wrist_right"))
    assert list(result.target_masks) == ["wrist_right"]
    assert result.view_diagnostics[0]["reason"] == "anchor"
    assert np.any(result.scene.target_mask)
    assert np.any(~result.scene.target_mask)


def test_associated_wrist_adds_metric_geometry_once():
    pose = np.eye(4)
    pose[0, 3] = .002
    calls = []

    def segment(image, *, pixel_uv):
        calls.append(pixel_uv)
        return mask()

    result = fuse(auxiliary=(view("wrist_left", reference_from_camera=pose),), segment=segment)
    assert len(calls) == 1
    assert result.view_diagnostics[1]["used"]
    assert len(result.scene.points) > len(fuse().scene.points)


@pytest.mark.parametrize("changes, reason", [
    ({"acquired_at": 10.3}, "frame_skew_exceeded"),
    ({"depth_m": np.full((32, 32), .3)}, "target_occluded"),
    ({"depth_m": np.full((32, 32), .7)}, "target_depth_unconfirmed"),
    ({"depth_m": np.zeros((32, 32))}, "target_depth_unconfirmed"),
])
def test_unusable_wrist_never_pollutes_background_or_calls_segmentation(changes, reason):
    def unexpected(*_args, **_kwargs):
        pytest.fail("An unassociated wrist must not run segmentation")

    result = fuse(auxiliary=(view("wrist_right", **changes),), segment=unexpected)
    assert result.view_diagnostics[1] == {"role": "wrist_right", "used": False, "reason": reason, **(
        {} if reason == "frame_skew_exceeded" else {"shared_target_pixels": 0})}
    np.testing.assert_array_equal(result.scene.points, fuse().scene.points)
    np.testing.assert_array_equal(result.scene.target_mask, fuse().scene.target_mask)


def test_out_of_view_wrist_is_excluded():
    pose = np.eye(4)
    pose[0, 3] = 2
    result = fuse(auxiliary=(view("wrist_left", reference_from_camera=pose),))
    assert result.view_diagnostics[1]["reason"] == "target_out_of_view"


def test_seed_depth_island_is_background_not_erased():
    depth = view().depth_m.copy()
    depth[21, 10:22] = .8
    anchor = view("wrist_left", depth_m=depth)
    before = fuse(anchor=anchor)
    after = fuse_grasp_views(anchor=anchor, anchor_mask=mask(), auxiliary=(), geometry=GEOMETRY,
        segment_point=lambda *a, **k: mask(), anchor_pixel_uv=(16, 16))
    refinement = after.view_diagnostics[0]["target_depth_refinement"]
    assert refinement["component_count"] == 2
    assert refinement["relabelled_background_points"] == 12
    assert refinement["deleted_scene_points"] == 0
    assert set(map(tuple, before.scene.points)) == set(map(tuple, after.scene.points))
    assert np.count_nonzero(before.scene.target_mask) - np.count_nonzero(after.scene.target_mask) == 12
    assert not np.any(after.scene.target_mask[after.scene.points[:, 2] > .7])


def test_seed_on_small_disconnected_blob_is_not_silently_reassigned():
    depth = view().depth_m.copy()
    depth[21, 10:22] = .8
    with pytest.raises(RuntimeError, match="ambiguous"):
        fuse_grasp_views(anchor=view(depth_m=depth), anchor_mask=mask(), auxiliary=(), geometry=GEOMETRY,
            segment_point=lambda *a, **k: mask(), anchor_pixel_uv=(16, 21))


def test_depth_refinement_protects_relabelled_geometry_during_self_filter():
    depth = view().depth_m.copy()
    depth[21, 10:22] = .8
    anchor = view(depth_m=depth)

    def clean_depth(view, protected):
        np.testing.assert_array_equal(protected, mask())
        return np.where(protected, view.depth_m, 0)

    result = fuse_grasp_views(anchor=anchor, anchor_mask=mask(), auxiliary=(), geometry=GEOMETRY,
        segment_point=lambda *a, **k: mask(), anchor_pixel_uv=(16, 16), clean_depth=clean_depth)
    assert np.count_nonzero(result.scene.points[:, 2] > .7) == 12
    assert not np.any(result.scene.target_mask[result.scene.points[:, 2] > .7])


def test_auxiliary_mask_cannot_relabel_visible_neighbour_as_target():
    contaminated = mask()
    contaminated[10:22, 22:26] = True
    result = fuse(auxiliary=(view("wrist_left"),), segment=lambda *a, **kw: contaminated)
    assert result.view_diagnostics[1]["reason"] == "target_background_conflict"
    np.testing.assert_array_equal(result.scene.points, fuse().scene.points)


def test_bad_pose_or_segmenter_fault_is_not_silently_dropped():
    pose = np.eye(4)
    pose[0, 0] = -1
    with pytest.raises(ValueError, match="rigid"):
        fuse(auxiliary=(view("wrist_left", reference_from_camera=pose),))
    with pytest.raises(RuntimeError, match="invalid mask"):
        fuse(auxiliary=(view("wrist_left"),), segment=lambda *_a, **_k: np.zeros((2, 2)))


def test_fusion_frame_transform_is_equivariant():
    reference = np.array([[0., -1, 0, .5], [1, 0, 0, .3], [0, 0, 1, -.1], [0, 0, 0, 1]])
    original = fuse()
    transformed = fuse(anchor=view(reference_from_camera=reference))
    expected = original.scene.points @ reference[:3, :3].T + reference[:3, 3]
    # All samples are separated by more than one voxel, preserving identity.
    np.testing.assert_allclose(transformed.scene.points, expected, atol=1e-8)


def test_snapshot_roundtrip_and_duplicate_roles_rejected():
    scene = fuse().scene
    data = encode_grasp_multiview_snapshot(views=(blob(), blob("wrist_right")),
        metadata={"schema_id": GRASP_MULTIVIEW_SCHEMA_ID},
        scene_points=scene.points, scene_target_mask=scene.target_mask)
    views, metadata, points, labels = decode_grasp_multiview_snapshot(data)
    assert len(views) == 2
    np.testing.assert_array_equal(points, scene.points)
    np.testing.assert_array_equal(labels, scene.target_mask)
    with pytest.raises(ValueError, match="unique"):
        encode_grasp_multiview_snapshot(views=(blob(), blob()), metadata=metadata)


def test_snapshot_rejects_unexpected_archive_members_and_unlabelled_scene():
    data = encode_grasp_multiview_snapshot(views=(blob(),), metadata={"schema_id": GRASP_MULTIVIEW_SCHEMA_ID})
    stream = io.BytesIO(data)
    with zipfile.ZipFile(stream, "a") as archive:
        archive.writestr("unexpected", "not part of the observation")
    with pytest.raises(ValueError, match="layout"):
        decode_grasp_multiview_snapshot(stream.getvalue())
    with pytest.raises(ValueError, match="scene"):
        encode_grasp_multiview_snapshot(views=(blob(),), metadata={"schema_id": GRASP_MULTIVIEW_SCHEMA_ID},
            scene_points=np.zeros((1, 3)), scene_target_mask=np.zeros(1, dtype=bool))


def test_path_reader_consumes_exact_fused_points_in_execution_frame(tmp_path):
    from agent_gateway.profiles.real.grasp_approach_plan import build_grasp_visible_scene_reader
    from agent_gateway.profiles.real.station_point_grasp_deployment import REAL_STATION_POINT_GRASP_SNAPSHOT_SLOT_ID

    capture = {"role": "main", "reference_frame_id": "reference", "T_reference_from_camera": np.eye(4).tolist()}
    points = np.array([[0, 0, .5], [.7, .8, .9]])
    labels = np.array([True, False])
    data = encode_grasp_multiview_snapshot(views=(blob(),),
        metadata={"schema_id": GRASP_MULTIVIEW_SCHEMA_ID, "capture": capture, "scene_frame": "anchor_camera",
                  "solid_geometry": asdict(GEOMETRY)}, scene_points=points, scene_target_mask=labels)
    path = tmp_path / "retained.zip"
    path.write_bytes(data)
    artifact = SimpleNamespace(schema_id=GRASP_MULTIVIEW_SCHEMA_ID, byte_length=len(data),
        sha256=hashlib.sha256(data).hexdigest(), artifact_id="scene")
    operation = SimpleNamespace(output_artifact_bindings=(SimpleNamespace(
        slot_id=REAL_STATION_POINT_GRASP_SNAPSHOT_SLOT_ID, artifact=artifact),))
    pose = np.eye(4)
    pose[0, 3] = 1.2
    reader = build_grasp_visible_scene_reader(
        artifact_store=SimpleNamespace(managed_read_path=lambda _: path),
        frame_graph=SimpleNamespace(transform_pose_matrix=lambda *a, **k: (pose, None)), execution_frame_id="solver")
    result = reader(operation, capture)
    np.testing.assert_array_equal(result.points, points + (1.2, 0, 0))
    np.testing.assert_array_equal(result.target_mask, labels)
