from dataclasses import asdict, replace
from types import SimpleNamespace
import hashlib
import io
import json
import zipfile

import numpy as np
import pytest

from dataset_contracts.grasp_multiview_snapshot import (
    GRASP_MULTIVIEW_SCHEMA_ID, decode_grasp_multiview_snapshot,
    decode_grasp_multiview_snapshot_arrays,
    encode_grasp_multiview_snapshot, encode_grasp_multiview_arrays,
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


def test_decoded_multiview_api_reuses_the_one_validated_array_decode(monkeypatch):
    import dataset_contracts.grasp_multiview_snapshot as codec
    encoded = encode_grasp_multiview_snapshot(views=[blob("main"), blob("wrist_left")],
        metadata={"schema_id": GRASP_MULTIVIEW_SCHEMA_ID})
    decode = codec.decode_grasp_rgbd_snapshot
    calls = []
    def counted(data):
        value = decode(data)
        calls.append(value)
        return value
    monkeypatch.setattr(codec, "decode_grasp_rgbd_snapshot", counted)
    views, _, points, labels = decode_grasp_multiview_snapshot_arrays(encoded)
    assert len(calls) == len(views) == 2
    assert views[0][0] is calls[0][0]
    assert views[1][1]["role"] == "wrist_left"
    assert points is labels is None
    np.testing.assert_array_equal(views[0][0]["rgb"], view().rgb)


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
    evidence = result.view_diagnostics[1]["association_evidence"]
    assert evidence["reverse_background_conflict_samples"] == 0
    assert evidence["target_depth_quantiles_m"] == [.5, .5, .5]
    json.dumps(result.view_diagnostics, allow_nan=False)
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
    evidence = result.view_diagnostics[1]["association_evidence"]
    assert evidence["reverse_confirmed_surface_samples"] == 192
    assert evidence["reverse_background_conflict_samples"] == 48
    assert evidence["reverse_background_conflict_fraction"] == .25
    assert evidence["target_xyz_span_m"] == pytest.approx([.075, .055, 0.])
    np.testing.assert_array_equal(result.scene.points, fuse().scene.points)
    np.testing.assert_array_equal(result.scene.target_mask, fuse().scene.target_mask)


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


@pytest.mark.parametrize("roles,anchor_role", (
    (("main",), "main"), (("main", "wrist_left", "wrist_right"), "wrist_left"),
))
def test_path_reader_consumes_exact_fused_points_in_execution_frame(tmp_path, monkeypatch, roles, anchor_role):
    import dataset_contracts.grasp_multiview_snapshot as codec
    import agent_gateway.profiles.real.grasp_approach_plan as plan_module
    from agent_gateway.profiles.real.grasp_approach_plan import build_grasp_visible_scene_reader
    from agent_gateway.profiles.real.station_point_grasp_deployment import REAL_STATION_POINT_GRASP_SNAPSHOT_SLOT_ID
    from agent_gateway.runtime.artifact_producer import artifact_ref_for_bytes, commit_artifact_bytes
    from agent_gateway.runtime.artifact_store import RecoverableAgentArtifactStore

    capture = {"role": anchor_role, "reference_frame_id": "reference", "T_reference_from_camera": np.eye(4).tolist()}
    points = np.array([[0, 0, .5], [.7, .8, .9]])
    labels = np.array([True, False])
    data = encode_grasp_multiview_snapshot(views=tuple(blob(role) for role in roles),
        metadata={"schema_id": GRASP_MULTIVIEW_SCHEMA_ID, "capture": capture, "scene_frame": "anchor_camera",
                  "solid_geometry": asdict(GEOMETRY)}, scene_points=points, scene_target_mask=labels)
    with RecoverableAgentArtifactStore(root=tmp_path / "store",
            quota_bytes=1024*1024, max_artifact_bytes=1024*1024) as store:
        artifact = artifact_ref_for_bytes(data, schema_id=GRASP_MULTIVIEW_SCHEMA_ID,
            media_type="application/zip", encoding="zip", compression="zip")
        commit_artifact_bytes(store=store, artifact=artifact, data=data, producer_id="test",
            runtime_session_id="test", idempotency_key="scene", now=1.)
        operation = SimpleNamespace(output_artifact_bindings=(SimpleNamespace(
            slot_id=REAL_STATION_POINT_GRASP_SNAPSHOT_SLOT_ID, artifact=artifact),))
        pose = np.eye(4)
        pose[0, 3] = 1.2
        reader = build_grasp_visible_scene_reader(artifact_store=store,
            frame_graph=SimpleNamespace(transform_pose_matrix=lambda *a, **k: (pose, None)), execution_frame_id="solver")
        decode = codec.decode_grasp_rgbd_snapshot
        decoded_views = []
        def counted(data):
            decoded_views.append(data)
            return decode(data)
        monkeypatch.setattr(codec, "decode_grasp_rgbd_snapshot", counted)
        monkeypatch.setattr(plan_module, "decode_grasp_rgbd_snapshot", counted)
        result = reader(operation, capture)
        assert len(decoded_views) == len(roles)
        np.testing.assert_array_equal(result.points, points + (1.2, 0, 0))
        np.testing.assert_array_equal(result.target_mask, labels)
        for side in ("left", "right"):
            np.testing.assert_array_equal(result.capture_qpos_by_side[side], np.zeros(6))


def test_multiview_encoder_validates_each_npz_once_and_still_rejects_corruption(monkeypatch):
    import dataset_contracts.grasp_multiview_snapshot as codec

    views = [blob("main"), blob("wrist_left"), blob("wrist_right")]
    decode = codec.decode_grasp_rgbd_snapshot
    calls = []

    def counted(data):
        calls.append(data)
        return decode(data)

    monkeypatch.setattr(codec, "decode_grasp_rgbd_snapshot", counted)
    encoded = encode_grasp_multiview_snapshot(views=views,
        metadata={"schema_id": GRASP_MULTIVIEW_SCHEMA_ID})
    assert calls == views
    decoded, _, _, _ = decode_grasp_multiview_snapshot(encoded)
    assert decoded == views
    for bad in (b"", b"not an npz", views[1][:-20], b"x" * (codec.GRASP_RGBD_SNAPSHOT_MAX_BYTES + 1)):
        with pytest.raises(ValueError):
            encode_grasp_multiview_snapshot(views=[views[0], bad],
                metadata={"schema_id": GRASP_MULTIVIEW_SCHEMA_ID})


@pytest.mark.parametrize("roles", [("main",), ("main", "wrist_left", "wrist_right")])
def test_composed_array_encoder_checks_each_final_view_once(monkeypatch, roles):
    import dataset_contracts.grasp_rgbd_snapshot as single
    import dataset_contracts.grasp_multiview_snapshot as multi

    fields = [dict(rgb=view(role).rgb, depth_m=view(role).depth_m, target_mask=mask(),
        metadata={"schema_id": GRASP_RGBD_SNAPSHOT_SCHEMA_ID, "role": role},
        qpos_by_side={"left": np.zeros(6), "right": np.ones(6)}) for role in roles]
    decode = single.decode_grasp_rgbd_snapshot
    calls = []
    def counted(data):
        calls.append(data)
        return decode(data)
    monkeypatch.setattr(single, "decode_grasp_rgbd_snapshot", counted)
    monkeypatch.setattr(multi, "decode_grasp_rgbd_snapshot", counted)
    encoded = encode_grasp_multiview_arrays(views=fields,
        metadata={"schema_id": GRASP_MULTIVIEW_SCHEMA_ID})
    assert len(calls) == len(roles)
    arrays, _, _, _ = decode_grasp_multiview_snapshot_arrays(encoded)
    assert [metadata["role"] for _, metadata in arrays] == list(roles)
    for (actual, metadata), expected in zip(arrays, fields, strict=True):
        assert metadata == expected["metadata"]
        for name in ("rgb", "depth_m", "target_mask"):
            np.testing.assert_array_equal(actual[name], expected[name])
        for side in ("left", "right"):
            np.testing.assert_array_equal(actual[side+"_qpos"], expected["qpos_by_side"][side])
    # The standalone encoder remains a fully validating public boundary.
    before = len(calls)
    encode_grasp_rgbd_snapshot(**fields[0])
    assert len(calls) == before + 1


def test_composed_array_encoder_cannot_publish_invalid_serialized_views(monkeypatch):
    import dataset_contracts.grasp_multiview_snapshot as codec
    sample = view()
    fields = dict(rgb=sample.rgb, depth_m=sample.depth_m, target_mask=mask(),
        metadata={"schema_id": GRASP_RGBD_SNAPSHOT_SCHEMA_ID, "role": "main"})
    for changes in ({"depth_m": np.full((32, 32), -1., dtype=np.float32)},
                    {"target_mask": np.zeros((1, 1), dtype=bool)},
                    {"qpos_by_side": {"left": np.zeros(6), "right": np.ones(5)}},
                    {"metadata": {"schema_id": "wrong", "role": "main"}}):
        with pytest.raises(ValueError):
            encode_grasp_multiview_arrays(views=[{**fields, **changes}],
                metadata={"schema_id": GRASP_MULTIVIEW_SCHEMA_ID})
    with pytest.raises(ValueError, match="unique"):
        encode_grasp_multiview_arrays(views=[fields, fields],
            metadata={"schema_id": GRASP_MULTIVIEW_SCHEMA_ID})
    monkeypatch.setattr(codec, "_serialize_grasp_rgbd_snapshot", lambda **_: b"truncated")
    with pytest.raises(ValueError, match="valid NPZ"):
        encode_grasp_multiview_arrays(views=[fields], metadata={"schema_id": GRASP_MULTIVIEW_SCHEMA_ID})


@pytest.mark.parametrize("order", ("C", "F"))
def test_rgbd_fast_compression_preserves_original_npy_members_and_numpy_readers(order):
    import dataset_contracts.grasp_rgbd_snapshot as codec

    rng = np.random.default_rng(29)
    fields = dict(rgb=np.array(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8), order=order),
        depth_m=np.array(rng.uniform(.1, .9, (32, 32)), dtype=np.float32, order=order),
        target_mask=np.array(rng.random((32, 32)) > .5, order=order),
        metadata={"schema_id": GRASP_RGBD_SNAPSHOT_SCHEMA_ID, "role": "wrist_left", "note": "无损米制深度"},
        qpos_by_side={"left": np.arange(6)/10., "right": -np.arange(6)/5.})
    legacy_arrays = {key: fields[key] for key in ("rgb", "depth_m", "target_mask")}
    legacy_arrays.update({side+"_qpos": fields["qpos_by_side"][side] for side in ("left", "right")})
    legacy_arrays["metadata"] = np.frombuffer(json.dumps(fields["metadata"], allow_nan=False,
        sort_keys=True).encode("utf-8"), np.uint8)
    stream = io.BytesIO()
    np.savez_compressed(stream, **legacy_arrays)
    encoded = encode_grasp_rgbd_snapshot(**fields)
    with zipfile.ZipFile(io.BytesIO(stream.getvalue())) as old, zipfile.ZipFile(io.BytesIO(encoded)) as new:
        assert set(old.namelist()) == set(new.namelist())
        for name in old.namelist():
            assert new.getinfo(name).compress_type == zipfile.ZIP_DEFLATED
            assert old.read(name) == new.read(name)
    with np.load(io.BytesIO(encoded), allow_pickle=False) as arrays:
        for key in legacy_arrays:
            np.testing.assert_array_equal(arrays[key], legacy_arrays[key])
    arrays, metadata = codec.decode_grasp_rgbd_snapshot(encoded)
    assert metadata == fields["metadata"]
    for side in ("left", "right"):
        np.testing.assert_array_equal(arrays[side+"_qpos"], fields["qpos_by_side"][side])


@pytest.mark.parametrize("fits_original_strength", (True, False))
def test_single_encoder_preserves_bound_using_original_strength_when_needed(monkeypatch, fits_original_strength):
    import dataset_contracts.grasp_rgbd_snapshot as codec

    sample = view()
    original = codec._compressed_snapshot
    levels = []
    monkeypatch.setattr(codec, "GRASP_RGBD_SNAPSHOT_MAX_BYTES", 65536)
    def sized(arrays, level):
        levels.append(level)
        data = original(arrays, level)
        # Model a near-budget compressed payload, keeping the real final decoder.
        return data if level == 6 and fits_original_strength else data + b"x"*65536
    monkeypatch.setattr(codec, "_compressed_snapshot", sized)
    fields = dict(rgb=sample.rgb, depth_m=sample.depth_m, target_mask=mask(),
        metadata={"schema_id": GRASP_RGBD_SNAPSHOT_SCHEMA_ID, "role": "main"})
    if fits_original_strength:
        data = encode_grasp_rgbd_snapshot(**fields)
        actual, metadata = codec.decode_grasp_rgbd_snapshot(data)
        np.testing.assert_array_equal(actual["depth_m"], sample.depth_m)
        assert metadata == fields["metadata"]
        assert len(data) <= 65536
    else:
        with pytest.raises(ValueError, match="byte bound"):
            encode_grasp_rgbd_snapshot(**fields)
    assert levels == [3, 6]


@pytest.mark.parametrize("fits_original_strength", (True, False))
def test_composed_encoder_retains_bound_and_does_not_reencode_supplied_bytes(monkeypatch, fits_original_strength):
    import dataset_contracts.grasp_multiview_snapshot as codec

    fields = [dict(rgb=view(role).rgb, depth_m=view(role).depth_m, target_mask=mask(),
        metadata={"schema_id": GRASP_RGBD_SNAPSHOT_SCHEMA_ID, "role": role}) for role in ("main", "wrist_left")]
    original_encode = codec._serialize_grasp_rgbd_snapshot
    original_pack = codec._serialize_grasp_multiview_snapshot
    levels, packs = [], []
    def encode(level=3, /, **view_fields):
        levels.append(level)
        return original_encode(level, **view_fields)
    def pack(**kwargs):
        packs.append(tuple(kwargs["views"]))
        data = original_pack(**kwargs)
        return data if len(packs) > 1 and fits_original_strength else data+b"x"*65536
    monkeypatch.setattr(codec, "GRASP_MULTIVIEW_MAX_BYTES", 65536)
    monkeypatch.setattr(codec, "_serialize_grasp_rgbd_snapshot", encode)
    monkeypatch.setattr(codec, "_serialize_grasp_multiview_snapshot", pack)
    metadata = {"schema_id": GRASP_MULTIVIEW_SCHEMA_ID}
    if fits_original_strength:
        data = encode_grasp_multiview_arrays(views=fields, metadata=metadata)
        views, actual_metadata, _, _ = decode_grasp_multiview_snapshot_arrays(data)
        assert actual_metadata == metadata
        for (arrays, info), expected in zip(views, fields, strict=True):
            assert info == expected["metadata"]
            for key in ("rgb", "depth_m", "target_mask"):
                np.testing.assert_array_equal(arrays[key], expected[key])
    else:
        with pytest.raises(ValueError, match="byte bound"):
            encode_grasp_multiview_arrays(views=fields, metadata=metadata)
    assert levels == [3, 3, 6, 6]
    assert len(packs) == 2
    supplied_views = packs[-1]
    levels.clear()
    packs.clear()
    with pytest.raises(ValueError, match="byte bound"):
        encode_grasp_multiview_snapshot(views=supplied_views, metadata=metadata)
    assert len(packs) == 1 and packs[0] == supplied_views
    assert not levels  # The byte-preserving API never silently recompresses.
