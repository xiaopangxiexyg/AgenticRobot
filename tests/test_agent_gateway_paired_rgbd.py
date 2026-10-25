"""Paired RGB-D primitive tests use in-memory media only, no models or renderer."""

import hashlib
import io
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from agent_gateway.profiles.back_projection import (
    CorrelatedPlacementCapture,
    LiveAlignedDepthCapture,
    project_depth_mask_geometry,
)
from agent_gateway.profiles.paired_rgbd import (
    PAIRED_RGBD_MAX_BYTES_V1,
    PAIRED_RGBD_MAX_DEPTH_BYTES_V1,
    PAIRED_RGBD_MAX_RGB_BYTES_V1,
    decode_paired_observation,
    encode_paired_observation,
)
from agent_gateway.profiles.segmentation import LiveRgbSegmentationCapture
from agent_gateway.runtime.artifact_producer import artifact_ref_for_bytes
from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.perception_artifacts import (
    DEPTH_FRAME_COMPRESSION_V1,
    DEPTH_FRAME_ENCODING_V1,
    DEPTH_FRAME_MEDIA_TYPE_V1,
    DEPTH_FRAME_SCHEMA_ID_V1,
    PAIRED_RGBD_SCHEMA_ID_V1,
    RGB_FRAME_COMPRESSION_V1,
    RGB_FRAME_ENCODING_V1,
    RGB_FRAME_MEDIA_TYPE_V1,
    RGB_FRAME_SCHEMA_ID_V1,
)

pytestmark = pytest.mark.unit


def _jpeg(*, size=(4, 3), mode="RGB", image_format="JPEG", orientation=None):
    image = Image.new(mode, size, 73)
    options = {}
    if orientation is not None:
        exif = Image.Exif()
        exif[274] = orientation
        options["exif"] = exif
    stream = io.BytesIO()
    image.save(stream, format=image_format, **options)
    return stream.getvalue()


def _npy(array):
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=array.dtype.hasobject)
    return stream.getvalue()


def _ref(data, *, rgb):
    return artifact_ref_for_bytes(
        data,
        media_type=RGB_FRAME_MEDIA_TYPE_V1 if rgb else DEPTH_FRAME_MEDIA_TYPE_V1,
        schema_id=RGB_FRAME_SCHEMA_ID_V1 if rgb else DEPTH_FRAME_SCHEMA_ID_V1,
        encoding=RGB_FRAME_ENCODING_V1 if rgb else DEPTH_FRAME_ENCODING_V1,
        compression=RGB_FRAME_COMPRESSION_V1 if rgb else DEPTH_FRAME_COMPRESSION_V1,
    )


def _depth_hash(depth):
    return canonical_content_hash(
        {
            "dtype": "float32",
            "unit": "meter",
            "invalid_value": 0.0,
            "shape": list(depth.shape),
            "byte_sha256": hashlib.sha256(depth.tobytes(order="C")).hexdigest(),
        },
        domain="dexteleop.mujoco_back_projection.depth_content.v1",
    )


def _fixture():
    rgb_jpeg = _jpeg()
    depth = np.full((3, 4), 0.5, dtype=np.float32)
    depth[0, 0] = 0.0
    depth_npy = _npy(depth)
    rgb_artifact, depth_artifact = _ref(rgb_jpeg, rgb=True), _ref(depth_npy, rgb=False)
    with Image.open(io.BytesIO(rgb_jpeg)) as image:
        rgb = LiveRgbSegmentationCapture(
            source_id="front",
            role="main",
            captured_at=12.5,
            station_revision=3,
            image=image,
            rgb_content_hash=rgb_artifact.sha256,
        )
    optical = np.diag([1.0, -1.0, -1.0, 1.0])
    optical[:3, 3] = [0.1, 0.2, 1.0]
    aligned = LiveAlignedDepthCapture(
        source_id="front",
        role="main",
        captured_at=12.5,
        station_revision=3,
        capture_group_id="mujoco-depth:test-cut",
        depth_content_hash=_depth_hash(depth),
        intrinsics={
            "width": 4,
            "height": 3,
            "fx": 2.0,
            "fy": 2.0,
            "cx": 2.0,
            "cy": 1.0,
            "fovy_deg": 90.0,
        },
        reference_from_camera=optical,
        depth_m=depth,
    )
    return CorrelatedPlacementCapture(rgb, aligned, rgb_jpeg, depth_npy), dict(
        rgb_artifact=rgb_artifact,
        depth_artifact=depth_artifact,
        reference_frame_id="mujoco_sim_hil/station",
        reference_frame_spec_hash="frame:test",
        frame_graph_hash="graph:test",
        clock_domain_id="mujoco.sim_hil.monotonic",
        source_operation_id="operation:test",
        source_request_hash="a" * 64,
        runtime_session_id="session:test",
    )


def _decode(data, cut, args, *, rgb_jpeg=None, depth_npy=None):
    return decode_paired_observation(
        data,
        rgb_jpeg=cut.rgb_jpeg if rgb_jpeg is None else rgb_jpeg,
        depth_npy=cut.depth_npy if depth_npy is None else depth_npy,
        expected_rgb_artifact=args["rgb_artifact"],
        expected_depth_artifact=args["depth_artifact"],
    )


def _dump(metadata):
    return json.dumps(
        metadata, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()


def test_roundtrip_preserves_original_bytes_full_refs_grid_and_optical_geometry_without_reencoding(
    monkeypatch,
):
    cut, args = _fixture()

    def forbidden(*unused, **kwargs):
        raise AssertionError("The codec must not encode or resize media")

    monkeypatch.setattr(Image.Image, "save", forbidden)
    monkeypatch.setattr(Image.Image, "resize", forbidden)
    data = encode_paired_observation(cut, **args)
    restored, metadata = _decode(data, cut, args)
    assert data == encode_paired_observation(cut, **args)
    assert metadata["schema_id"] == PAIRED_RGBD_SCHEMA_ID_V1
    assert metadata["rgb_artifact"] == args["rgb_artifact"].to_wire()
    assert metadata["depth_artifact"] == args["depth_artifact"].to_wire()
    assert restored.rgb_jpeg is cut.rgb_jpeg and restored.depth_npy is cut.depth_npy
    assert restored.rgb.image.tobytes() == cut.rgb.image.tobytes()
    assert (
        metadata["width"],
        metadata["height"],
        metadata["captured_at"],
        metadata["station_revision"],
    ) == (4, 3, 12.5, 3)
    assert metadata["intrinsics"] == dict(cut.depth.intrinsics)
    assert not restored.depth.depth_m.flags.writeable
    assert not restored.depth.reference_from_camera.flags.writeable
    np.testing.assert_array_equal(
        restored.depth.reference_from_camera, cut.depth.reference_from_camera
    )
    np.testing.assert_array_equal(restored.depth.depth_m, cut.depth.depth_m)
    mask = np.zeros((3, 4), dtype=bool)
    mask[1, 2] = True
    geometry = project_depth_mask_geometry(
        restored.depth, mask, minimum_valid_points=1, maximum_depth_m=10.0
    )
    assert geometry.world_center_xyz_m == pytest.approx((0.1, 0.2, 0.5))


@pytest.mark.parametrize(
    "field,bad",
    [
        ("rgb_jpeg", None),
        ("depth_npy", None),
        ("rgb_jpeg", b""),
        ("depth_npy", bytearray(b"npy")),
    ],
)
def test_legacy_cut_is_still_constructible_but_missing_original_bytes_fail_closed(
    field, bad
):
    cut, args = _fixture()
    legacy = CorrelatedPlacementCapture(cut.rgb, cut.depth)
    assert legacy.rgb_jpeg is None and legacy.depth_npy is None
    with pytest.raises(ValueError, match="original bounded"):
        encode_paired_observation(replace(cut, **{field: bad}), **args)


@pytest.mark.parametrize("side", ["rgb", "depth"])
@pytest.mark.parametrize(
    "change",
    [
        {"media_type": "application/octet-stream"},
        {"schema_id": "wrong.v1"},
        {"encoding": "other"},
        {"compression": "zip"},
        {"sha256": "b" * 64},
        {"byte_length": 1},
    ],
)
def test_exact_media_ref_format_and_bytes_are_mandatory(side, change):
    cut, args = _fixture()
    args[side + "_artifact"] = replace(args[side + "_artifact"], **change)
    with pytest.raises(ValueError, match="exact ArtifactRef"):
        encode_paired_observation(cut, **args)


@pytest.mark.parametrize("field", ["artifact_id", "artifact_ref_hash", "byte_length"])
def test_full_wire_descriptor_identity_cannot_be_edited(field):
    cut, args = _fixture()
    metadata = json.loads(encode_paired_observation(cut, **args))
    metadata["rgb_artifact"][field] = 1 if field == "byte_length" else "wrong"
    with pytest.raises(ValueError):
        _decode(_dump(metadata), cut, args)


@pytest.mark.parametrize(
    "fault", ["rgb_time", "rgb_hash", "rgb_pixels", "depth_samples", "depth_hash"]
)
def test_self_claimed_hash_or_mismatched_typed_cut_cannot_replace_original_media(fault):
    cut, args = _fixture()
    if fault == "rgb_time":
        cut = replace(cut, rgb=replace(cut.rgb, captured_at=13.0))
    elif fault == "rgb_hash":
        # A Real-style raw-array hash is not the original JPEG media hash.
        cut = replace(
            cut,
            rgb=replace(
                cut.rgb,
                rgb_content_hash=hashlib.sha256(cut.rgb.image.tobytes()).hexdigest(),
            ),
        )
    elif fault == "rgb_pixels":
        cut.rgb.image.putpixel((0, 0), (255, 0, 0))
    elif fault == "depth_samples":
        depth = cut.depth.depth_m.copy()
        depth[1, 1] += 0.1
        cut = replace(cut, depth=replace(cut.depth, depth_m=depth))
    else:
        cut = replace(cut, depth=replace(cut.depth, depth_content_hash="b" * 64))
    with pytest.raises(ValueError, match="exact typed RGB-D cut|depth_content_hash"):
        encode_paired_observation(cut, **args)


@pytest.mark.parametrize(
    "fault",
    [
        "float64",
        "fortran",
        "big_endian",
        "object",
        "one_dim",
        "trailing",
        "two_arrays",
        "negative",
        "nan",
        "inf",
        "huge_header",
    ],
)
def test_depth_requires_bounded_float32_c_order_no_pickle_or_trailing_data(
    monkeypatch, fault
):
    cut, args = _fixture()
    metadata = json.loads(encode_paired_observation(cut, **args))
    depth = cut.depth.depth_m.copy()
    if fault == "float64":
        bad = _npy(depth.astype(np.float64))
    elif fault == "fortran":
        bad = _npy(np.asfortranarray(depth))
    elif fault == "big_endian":
        bad = _npy(depth.astype(">f4"))
    elif fault == "object":
        bad = _npy(np.empty((3, 4), dtype=object))
    elif fault == "one_dim":
        bad = _npy(depth.reshape(-1))
    elif fault in ("trailing", "two_arrays"):
        bad = cut.depth_npy + (b"x" if fault == "trailing" else cut.depth_npy)
    elif fault == "huge_header":
        stream = io.BytesIO()
        np.lib.format.write_array_header_1_0(
            stream, {"shape": (10**9, 10**9), "fortran_order": False, "descr": "<f4"}
        )
        bad = stream.getvalue()

        def forbidden(*unused, **kwargs):
            raise AssertionError("A forged header must fail before creating an array")

        monkeypatch.setattr(np, "frombuffer", forbidden)
    else:
        depth[1, 1] = {"negative": -0.1, "nan": float("nan"), "inf": float("inf")}[
            fault
        ]
        bad = _npy(depth)
    args["depth_artifact"] = _ref(bad, rgb=False)
    metadata["depth_artifact"] = args["depth_artifact"].to_wire()
    with pytest.raises(ValueError, match="depth"):
        _decode(_dump(metadata), cut, args, depth_npy=bad)


@pytest.mark.parametrize(
    "fault", ["grid", "png", "grayscale", "truncated", "exif_2", "exif_6", "exif_8"]
)
def test_jpeg_grid_format_and_identity_orientation_are_checked(fault):
    cut, args = _fixture()
    metadata = json.loads(encode_paired_observation(cut, **args))
    if fault == "grid":
        bad = _jpeg(size=(8, 6))
    elif fault == "png":
        bad = _jpeg(image_format="PNG")
    elif fault == "grayscale":
        bad = _jpeg(mode="L")
    elif fault == "truncated":
        bad = cut.rgb_jpeg[: len(cut.rgb_jpeg) // 2]
    else:
        bad = _jpeg(orientation=int(fault[-1]))
    args["rgb_artifact"] = _ref(bad, rgb=True)
    metadata["rgb_artifact"] = args["rgb_artifact"].to_wire()
    with pytest.raises(ValueError, match="JPEG"):
        _decode(_dump(metadata), cut, args, rgb_jpeg=bad)


def test_explicit_identity_exif_is_preserved_without_transposing():
    cut, args = _fixture()
    original = _jpeg(orientation=1)
    with Image.open(io.BytesIO(original)) as image:
        rgb = replace(
            cut.rgb, image=image, rgb_content_hash=hashlib.sha256(original).hexdigest()
        )
    cut = replace(cut, rgb=rgb, rgb_jpeg=original)
    args["rgb_artifact"] = _ref(original, rgb=True)
    restored, _ = _decode(encode_paired_observation(cut, **args), cut, args)
    assert restored.rgb_jpeg == original


@pytest.mark.parametrize(
    "field,bad",
    [
        ("schema_id", "other.v1"),
        ("width", True),
        ("height", 0),
        ("width", 1921),
        ("height", 1081),
        ("captured_at", True),
        ("station_revision", True),
        ("station_revision", -1),
        ("role", " main"),
        ("source_id", ""),
        ("capture_group_id", None),
        ("source_request_hash", "not-a-digest"),
        ("reference_frame_id", ""),
        ("reference_frame_spec_hash", False),
        ("frame_graph_hash", " "),
        ("clock_domain_id", "x" * 257),
        ("source_operation_id", 1),
        ("runtime_session_id", " session"),
    ],
)
def test_metadata_fields_are_strict_and_bounded(field, bad):
    cut, args = _fixture()
    metadata = json.loads(encode_paired_observation(cut, **args))
    metadata[field] = bad
    with pytest.raises((TypeError, ValueError)):
        _decode(_dump(metadata), cut, args)


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "extra",
        "duplicate",
        "nested_duplicate",
        "nan",
        "overflow",
        "trailing",
    ],
)
def test_json_is_closed_native_and_has_no_duplicate_keys_or_nonfinite_numbers(fault):
    cut, args = _fixture()
    data = encode_paired_observation(cut, **args)
    metadata = json.loads(data)
    if fault == "missing":
        del metadata["role"]
        bad = _dump(metadata)
    elif fault == "extra":
        metadata["caller_claim"] = True
        bad = _dump(metadata)
    elif fault == "duplicate":
        bad = b'{"role":"main",' + data[1:]
    elif fault == "nested_duplicate":
        bad = data.replace(b'"intrinsics":{', b'"intrinsics":{"fx":2.0,')
    elif fault in ("nan", "overflow"):
        bad = data.replace(
            b'"captured_at":12.5',
            b'"captured_at":' + (b"NaN" if fault == "nan" else b"1e309"),
        )
    else:
        bad = data + b"{}"
    with pytest.raises((TypeError, ValueError)):
        _decode(bad, cut, args)


@pytest.mark.parametrize(
    "fault",
    [
        "extra_intrinsic",
        "grid",
        "bool_focal",
        "negative_focal",
        "fovy",
        "transform_nan",
        "transform_bool",
        "transform_reflection",
        "transform_shape",
    ],
)
def test_pinhole_and_finite_rigid_optical_transform_are_validated(fault):
    cut, args = _fixture()
    metadata = json.loads(encode_paired_observation(cut, **args))
    if fault == "extra_intrinsic":
        metadata["intrinsics"]["caller_calibration"] = 2.0
    elif fault == "grid":
        metadata["intrinsics"]["width"] = 8
    elif fault == "bool_focal":
        metadata["intrinsics"]["fx"] = True
    elif fault == "negative_focal":
        metadata["intrinsics"]["fx"] = -2.0
    elif fault == "fovy":
        metadata["intrinsics"]["fovy_deg"] = 180.0
    elif fault == "transform_nan":
        metadata["reference_from_camera"][0][0] = 1e308
    elif fault == "transform_bool":
        metadata["reference_from_camera"][0][0] = True
    elif fault == "transform_reflection":
        metadata["reference_from_camera"][0][0] = -1.0
    else:
        metadata["reference_from_camera"] = [0.0] * 16
    with pytest.raises((TypeError, ValueError, FloatingPointError)):
        _decode(_dump(metadata), cut, args)


@pytest.mark.parametrize(
    "media,limit",
    [
        ("manifest", PAIRED_RGBD_MAX_BYTES_V1),
        ("rgb", PAIRED_RGBD_MAX_RGB_BYTES_V1),
        ("depth", PAIRED_RGBD_MAX_DEPTH_BYTES_V1),
    ],
)
def test_byte_limits_are_checked_before_parsing_or_decoding(media, limit):
    cut, args = _fixture()
    data = encode_paired_observation(cut, **args)
    bad = b"x" * (limit + 1)
    with pytest.raises(ValueError, match="bounded"):
        _decode(
            bad if media == "manifest" else data,
            cut,
            args,
            rgb_jpeg=bad if media == "rgb" else None,
            depth_npy=bad if media == "depth" else None,
        )


def test_mujoco_adapter_passes_exact_original_media_and_optical_pose_without_rendering():
    from agent_gateway.profiles.mujoco.back_projection import (
        MujocoBackProjectionCapturePort,
    )
    from agent_gateway.profiles.mujoco.motion_application import (
        MujocoAgentSkillMotionApplication,
    )

    cut, args = _fixture()
    calls = []

    class Application(MujocoAgentSkillMotionApplication):
        def __init__(self):
            pass

        def capture_visual_depth_cut(self, *, view_id):
            calls.append(view_id)
            return SimpleNamespace(data=cut.rgb_jpeg), SimpleNamespace(
                data=cut.depth_npy,
                width=4,
                height=3,
                captured_at=12.5,
                camera_pose_world_xyz_quat_xyzw=(0.1, 0.2, 1.0, 1.0, 0.0, 0.0, 0.0),
                intrinsics=dict(cut.depth.intrinsics),
            )

        def transform_tcp_pose_xyzquat(self, pose, **unused):
            return tuple(pose)

    port = MujocoBackProjectionCapturePort(
        application=Application(),
        source_id_by_role={"main": "front"},
        source_frame_id="world",
        reference_frame_id="station",
        current_station_revision=lambda: 3,
    )
    native = port.capture_placement_rgbd(role="main", expected_station_revision=3)
    assert calls == ["front"]
    assert native.rgb_jpeg is cut.rgb_jpeg and native.depth_npy is cut.depth_npy
    restored, _ = _decode(encode_paired_observation(native, **args), native, args)
    np.testing.assert_allclose(
        restored.depth.reference_from_camera,
        cut.depth.reference_from_camera,
        atol=1e-15,
    )
