"""Bounded, pickle-free replay representation for one point-grasp RGB-D cut."""

from __future__ import annotations

import io
import json
import math
import zipfile
from collections.abc import Mapping

import numpy as np

GRASP_RGBD_SNAPSHOT_MEDIA_TYPE = "application/x-npz"
GRASP_RGBD_SNAPSHOT_SCHEMA_ID = "dexteleop.perception.grasp_rgbd_snapshot.v1"
GRASP_RGBD_SNAPSHOT_ENCODING = "npz.rgb8.depth32.mask_bool.metadata_json.v1"
GRASP_RGBD_SNAPSHOT_MAX_BYTES = 16 * 1024 * 1024
_FIELDS = {"rgb", "depth_m", "target_mask", "left_qpos", "right_qpos", "metadata"}


def encode_grasp_rgbd_snapshot(*, rgb: np.ndarray, depth_m: np.ndarray,
                              target_mask: np.ndarray, metadata: Mapping,
                              qpos_by_side: Mapping | None = None) -> bytes:
    arrays = {
        "rgb": np.asarray(rgb), "depth_m": np.asarray(depth_m),
        "target_mask": np.asarray(target_mask),
        "metadata": np.frombuffer(json.dumps(dict(metadata), allow_nan=False,
                                  sort_keys=True).encode("utf-8"), dtype=np.uint8),
        **{side+"_qpos": np.asarray([] if qpos_by_side is None else qpos_by_side[side],
                                   dtype=np.float64) for side in ("left", "right")},
    }
    if any(array.dtype.hasobject for array in arrays.values()):
        raise ValueError("snapshot arrays cannot contain Python objects")
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    data = stream.getvalue()
    decode_grasp_rgbd_snapshot(data)
    return data


def decode_grasp_rgbd_snapshot(data: bytes) -> tuple[dict[str, np.ndarray], dict]:
    if not 0 < len(data) <= GRASP_RGBD_SNAPSHOT_MAX_BYTES:
        raise ValueError("snapshot exceeds its byte bound")
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as error:
        raise ValueError("snapshot is not a valid NPZ archive") from error
    with archive:
        infos = archive.infolist()
        if (len(infos) != len(_FIELDS) or {x.filename for x in infos} !=
                {name+".npy" for name in _FIELDS} or
                sum(x.file_size for x in infos) > GRASP_RGBD_SNAPSHOT_MAX_BYTES):
            raise ValueError("snapshot members exceed the reviewed representation")
        # Zip sizes alone cannot bound an NPY header that claims a huge shape.
        # Verify headers/body sizes before NumPy allocates any member arrays.
        for info in infos:
            with archive.open(info) as member:
                version = np.lib.format.read_magic(member)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(member)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(member)
                else:
                    raise ValueError("snapshot uses an unsupported NPY version")
                if (dtype.hasobject or dtype not in (np.dtype("uint8"), np.dtype("float32"),
                                                    np.dtype("bool"), np.dtype("float64"))
                        or any(not isinstance(x, int) or x < 0 for x in shape)
                        or math.prod(shape)*dtype.itemsize != info.file_size-member.tell()):
                    raise ValueError("snapshot NPY header exceeds its bounded body")
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in _FIELDS}
    rgb, depth, mask = (arrays[name] for name in ("rgb", "depth_m", "target_mask"))
    if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2:] != (3,)
            or not 0 < rgb.size <= 1920*1080*3
            or depth.dtype != np.float32 or depth.shape != rgb.shape[:2]
            or mask.dtype != np.bool_ or mask.shape != depth.shape
            or not np.all(np.isfinite(depth)) or np.any(depth < 0) or np.any(depth > 10)):
        raise ValueError("snapshot RGB, metric depth and mask must be aligned")
    for name in ("left_qpos", "right_qpos"):
        q = arrays[name]
        if q.dtype != np.float64 or q.shape not in {(0,), (6,)} or not np.all(np.isfinite(q)):
            raise ValueError("snapshot qpos must be absent or six finite radians")
    if arrays["left_qpos"].shape != arrays["right_qpos"].shape:
        raise ValueError("snapshot must retain both arms or explicitly neither")
    raw = arrays.pop("metadata")
    if raw.dtype != np.uint8 or raw.ndim != 1 or raw.size > 1024*1024:
        raise ValueError("snapshot metadata must be bounded UTF-8 JSON")
    metadata = json.loads(raw.tobytes().decode("utf-8"))
    if not isinstance(metadata, dict) or metadata.get("schema_id") != GRASP_RGBD_SNAPSHOT_SCHEMA_ID:
        raise ValueError("snapshot metadata schema is invalid")
    return arrays, metadata
