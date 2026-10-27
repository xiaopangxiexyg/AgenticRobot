"""Bounded same-cut RGB-D manifests; no capture, re-encoding, CAS or authority.

Version 1 accepts only original MuJoCo JPEG/NPY bytes and their exact ArtifactRefs.
Real captures use a raw-array RGB hash, not a JPEG media hash, and are unsupported.
Metadata is strict UTF-8 JSON; media remain separate original Artifacts. Missing
bytes fail closed, never falling back to re-encoding a PIL image or depth array.
The producer has already converted reference_from_camera from optical camera
axes into the reference frame. Preserve it without another optical/OpenGL flip.
Absent EXIF orientation means identity; an explicit orientation must be integer 1.

The codec validates representation, bytes, grids, same-cut fields and calibration
values, not producer identity, client/session authorization or camera authenticity.
Readers must first use the existing CAS/completed-operation provenance wrapper
and check deployment-bound frame/clock/producer identities. Caller-supplied
metadata alone never grants authority. This module owns no resource lifecycle.
"""

from __future__ import annotations

import hashlib
import io
import json

import numpy as np
from PIL import Image

from agent_gateway.profiles.back_projection import (
    CorrelatedPlacementCapture,
    LiveAlignedDepthCapture,
)
from agent_gateway.profiles.segmentation import LiveRgbSegmentationCapture
from agent_gateway.protocol._contract_validation import (
    finite_number,
    nonnegative_integer,
    positive_integer,
    required_text,
    sha256_digest,
)
from agent_gateway.protocol.artifact_contracts import AgentArtifactRef
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

PAIRED_RGBD_MAX_BYTES_V1 = 64 * 1024
PAIRED_RGBD_MAX_RGB_BYTES_V1 = 16 * 1024 * 1024
PAIRED_RGBD_MAX_DEPTH_BYTES_V1 = 16 * 1024 * 1024
PAIRED_RGBD_MAX_WIDTH_V1 = 1920
PAIRED_RGBD_MAX_HEIGHT_V1 = 1080

_IDENTITIES = (
    "reference_frame_id",
    "reference_frame_spec_hash",
    "frame_graph_hash",
    "clock_domain_id",
    "source_operation_id",
    "runtime_session_id",
)
_FIELDS = frozenset(
    (
        *_IDENTITIES,
        "source_request_hash",
        "schema_id",
        "width",
        "height",
        "role",
        "source_id",
        "captured_at",
        "station_revision",
        "capture_group_id",
        "rgb_artifact",
        "depth_artifact",
        "intrinsics",
        "reference_from_camera",
        "depth_content_hash",
    )
)
_INTRINSICS = frozenset(("width", "height", "fx", "fy", "cx", "cy"))


def _bytes(value, maximum, name):
    if not isinstance(value, bytes) or not 0 < len(value) <= maximum:
        raise ValueError(f"{name} requires original bounded nonempty bytes")
    return value


def _media_ref(artifact, data, *, rgb):
    if not isinstance(artifact, AgentArtifactRef):
        raise TypeError("Expected a typed AgentArtifactRef")
    expected = (
        (
            RGB_FRAME_MEDIA_TYPE_V1,
            RGB_FRAME_SCHEMA_ID_V1,
            RGB_FRAME_ENCODING_V1,
            RGB_FRAME_COMPRESSION_V1,
        )
        if rgb
        else (
            DEPTH_FRAME_MEDIA_TYPE_V1,
            DEPTH_FRAME_SCHEMA_ID_V1,
            DEPTH_FRAME_ENCODING_V1,
            DEPTH_FRAME_COMPRESSION_V1,
        )
    )
    if (
        (
            artifact.media_type,
            artifact.schema_id,
            artifact.encoding,
            artifact.compression,
        )
        != expected
        or artifact.byte_length != len(data)
        or artifact.sha256 != hashlib.sha256(data).hexdigest()
    ):
        raise ValueError("Media bytes/format differ from the exact ArtifactRef")


def _depth_array(data, width, height):
    stream = io.BytesIO(data)
    try:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError("Paired depth requires NPY v1/v2")
    except (EOFError, OSError, ValueError) as error:
        raise ValueError("Paired depth has an invalid bounded NPY header") from error
    # Check the header before any array allocation or interpretation of its body.
    if (
        shape != (height, width)
        or fortran
        or dtype != np.dtype(np.float32)
        or stream.tell() + width * height * 4 != len(data)
    ):
        raise ValueError(
            "Paired depth requires exact float32 C-order grid and no trailing data"
        )
    depth = np.frombuffer(data, dtype=np.float32, offset=stream.tell()).reshape(shape)
    if not np.isfinite(depth).all() or np.any(depth < 0):
        raise ValueError("Paired metric depth must be finite and nonnegative")
    return depth


def _depth_content_hash(depth):
    # Validate the existing MuJoCo v1 content representation, not its NPY header.
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


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate paired observation JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def decode_paired_observation(
    data, *, rgb_jpeg, depth_npy, expected_rgb_artifact, expected_depth_artifact
):
    """Decode exact external media after caller's CAS/producer provenance checks."""
    _bytes(data, PAIRED_RGBD_MAX_BYTES_V1, "Paired manifest")
    _bytes(rgb_jpeg, PAIRED_RGBD_MAX_RGB_BYTES_V1, "RGB JPEG")
    _bytes(depth_npy, PAIRED_RGBD_MAX_DEPTH_BYTES_V1, "Depth NPY")
    metadata = json.loads(
        data.decode("utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_invalid_constant,
    )
    if (
        not isinstance(metadata, dict)
        or set(metadata) != _FIELDS
        or metadata["schema_id"] != PAIRED_RGBD_SCHEMA_ID_V1
    ):
        raise ValueError("Paired observation must match the exact v1 metadata schema")
    for key in (*_IDENTITIES, "role", "source_id", "capture_group_id"):
        required_text(metadata[key], field_name=key, maximum_length=256)
    sha256_digest(metadata["source_request_hash"], field_name="source_request_hash")
    sha256_digest(metadata["depth_content_hash"], field_name="depth_content_hash")
    width = positive_integer(
        metadata["width"], field_name="width", maximum=PAIRED_RGBD_MAX_WIDTH_V1
    )
    height = positive_integer(
        metadata["height"], field_name="height", maximum=PAIRED_RGBD_MAX_HEIGHT_V1
    )
    stamp = finite_number(metadata["captured_at"], field_name="captured_at")
    revision = nonnegative_integer(
        metadata["station_revision"], field_name="station_revision"
    )
    for key, expected, media, rgb in (
        ("rgb_artifact", expected_rgb_artifact, rgb_jpeg, True),
        ("depth_artifact", expected_depth_artifact, depth_npy, False),
    ):
        actual = AgentArtifactRef.from_wire(metadata[key])
        _media_ref(expected, media, rgb=rgb)
        if actual != expected:
            raise ValueError(
                "Paired metadata differs from the expected full media ArtifactRef"
            )
    intrinsics = metadata["intrinsics"]
    if not isinstance(intrinsics, dict) or set(intrinsics) not in (
        _INTRINSICS,
        _INTRINSICS | {"fovy_deg"},
    ):
        raise ValueError(
            "Paired intrinsics must contain the exact pinhole fields and optional fovy_deg"
        )
    for key in ("width", "height"):
        positive_integer(intrinsics[key], field_name=f"intrinsics.{key}")
        if intrinsics[key] != metadata[key]:
            raise ValueError("Paired intrinsics changed the native grid")
    for key in set(intrinsics) - {"width", "height"}:
        finite_number(intrinsics[key], field_name=f"intrinsics.{key}")
    if "fovy_deg" in intrinsics and not 0 < intrinsics["fovy_deg"] < 180:
        raise ValueError("Paired camera fovy_deg must be in (0, 180)")
    transform = metadata["reference_from_camera"]
    if (
        not isinstance(transform, list)
        or len(transform) != 4
        or any(not isinstance(row, list) or len(row) != 4 for row in transform)
    ):
        raise ValueError("Paired optical reference_from_camera must be a 4x4 matrix")
    for row in transform:
        for value in row:
            finite_number(value, field_name="reference_from_camera")
    depth = _depth_array(depth_npy, width, height)
    if _depth_content_hash(depth) != metadata["depth_content_hash"]:
        raise ValueError(
            "Paired depth_content_hash differs from original depth samples"
        )
    try:
        with Image.open(io.BytesIO(rgb_jpeg)) as image:
            if (
                image.format != "JPEG"
                or image.mode != "RGB"
                or image.size != (width, height)
            ):
                raise ValueError(
                    "Paired JPEG format/grid differs; no resizing or color conversion is allowed"
                )
            orientation = image.getexif().get(274)
            if orientation is not None and (
                type(orientation) is not int or orientation != 1
            ):
                raise ValueError("Paired JPEG EXIF orientation must be identity")
            image.load()
            rgb = LiveRgbSegmentationCapture(
                source_id=metadata["source_id"],
                role=metadata["role"],
                captured_at=stamp,
                station_revision=revision,
                image=image,
                rgb_content_hash=expected_rgb_artifact.sha256,
            )
    except OSError as error:
        raise ValueError("Paired RGB is not a complete JPEG") from error
    with np.errstate(over="raise", invalid="raise"):
        aligned = LiveAlignedDepthCapture(
            source_id=metadata["source_id"],
            role=metadata["role"],
            captured_at=stamp,
            station_revision=revision,
            capture_group_id=metadata["capture_group_id"],
            depth_content_hash=metadata["depth_content_hash"],
            intrinsics=intrinsics,
            reference_from_camera=transform,
            depth_m=depth,
        )
    return CorrelatedPlacementCapture(
        rgb=rgb, depth=aligned, rgb_jpeg=rgb_jpeg, depth_npy=depth_npy
    ), metadata


def encode_paired_observation(
    cut,
    *,
    rgb_artifact,
    depth_artifact,
    reference_frame_id,
    reference_frame_spec_hash,
    frame_graph_hash,
    clock_domain_id,
    source_operation_id,
    source_request_hash,
    runtime_session_id,
):
    """Serialize one typed cut, rejecting re-encoded, resized or unpaired media."""
    if not isinstance(cut, CorrelatedPlacementCapture):
        raise TypeError("A typed CorrelatedPlacementCapture is required")
    finite_number(cut.rgb.captured_at, field_name="RGB captured_at")
    rgb_bytes = _bytes(cut.rgb_jpeg, PAIRED_RGBD_MAX_RGB_BYTES_V1, "RGB JPEG")
    depth_bytes = _bytes(cut.depth_npy, PAIRED_RGBD_MAX_DEPTH_BYTES_V1, "Depth NPY")
    _media_ref(rgb_artifact, rgb_bytes, rgb=True)
    _media_ref(depth_artifact, depth_bytes, rgb=False)
    metadata = {
        "schema_id": PAIRED_RGBD_SCHEMA_ID_V1,
        "width": cut.depth.intrinsics["width"],
        "height": cut.depth.intrinsics["height"],
        "role": cut.depth.role,
        "source_id": cut.depth.source_id,
        "captured_at": cut.depth.captured_at,
        "station_revision": cut.depth.station_revision,
        "capture_group_id": cut.depth.capture_group_id,
        "rgb_artifact": rgb_artifact.to_wire(),
        "depth_artifact": depth_artifact.to_wire(),
        "intrinsics": dict(cut.depth.intrinsics),
        "reference_from_camera": cut.depth.reference_from_camera.tolist(),
        "depth_content_hash": cut.depth.depth_content_hash,
        "reference_frame_id": reference_frame_id,
        "reference_frame_spec_hash": reference_frame_spec_hash,
        "frame_graph_hash": frame_graph_hash,
        "clock_domain_id": clock_domain_id,
        "source_operation_id": source_operation_id,
        "source_request_hash": source_request_hash,
        "runtime_session_id": runtime_session_id,
    }
    data = json.dumps(
        metadata,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    decoded, _ = decode_paired_observation(
        data,
        rgb_jpeg=rgb_bytes,
        depth_npy=depth_bytes,
        expected_rgb_artifact=rgb_artifact,
        expected_depth_artifact=depth_artifact,
    )
    if (
        (
            cut.rgb.source_id,
            cut.rgb.role,
            cut.rgb.captured_at,
            cut.rgb.station_revision,
            cut.rgb.rgb_content_hash,
        )
        != (
            decoded.rgb.source_id,
            decoded.rgb.role,
            decoded.rgb.captured_at,
            decoded.rgb.station_revision,
            decoded.rgb.rgb_content_hash,
        )
        or cut.rgb.image.mode != "RGB"
        or cut.rgb.image.size != decoded.rgb.image.size
        or cut.rgb.image.tobytes() != decoded.rgb.image.tobytes()
        or cut.depth.depth_m.dtype != np.dtype(np.float32)
        or not cut.depth.depth_m.flags.c_contiguous
        or cut.depth.depth_m.shape != decoded.depth.depth_m.shape
        or cut.depth.depth_m.tobytes(order="C")
        != decoded.depth.depth_m.tobytes(order="C")
    ):
        raise ValueError("Original media bytes differ from the exact typed RGB-D cut")
    return data


__all__ = [
    "PAIRED_RGBD_MAX_BYTES_V1",
    "PAIRED_RGBD_MAX_DEPTH_BYTES_V1",
    "PAIRED_RGBD_MAX_HEIGHT_V1",
    "PAIRED_RGBD_MAX_RGB_BYTES_V1",
    "PAIRED_RGBD_MAX_WIDTH_V1",
    "decode_paired_observation",
    "encode_paired_observation",
]
