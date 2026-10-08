"""Bounded replay of correlated RGB-D views and the scene actually planned on.

Each view uses the existing single-view codec. No file paths, pickle payloads,
or camera recapture are involved when a consumer opens this observation.
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from collections.abc import Mapping, Sequence

import numpy as np

from .grasp_rgbd_snapshot import decode_grasp_rgbd_snapshot

GRASP_MULTIVIEW_SCHEMA_ID = "dexteleop.perception.grasp_multiview_snapshot.v1"
GRASP_MULTIVIEW_ENCODING = "zip.rgbd_views.scene_points.metadata_json.v1"
GRASP_MULTIVIEW_MAX_BYTES = 16 * 1024 * 1024
MAX_SCENE_POINTS = 300_000


def _array_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    return stream.getvalue()


def _read_array(data: bytes, *, dtype: str, columns: int | None) -> np.ndarray:
    stream = io.BytesIO(data)
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, _, actual_dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, _, actual_dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError("Unsupported multi-view NPY version")
    if (actual_dtype != np.dtype(dtype) or not shape
            or not 0 <= shape[0] <= MAX_SCENE_POINTS
            or shape[1:] != (() if columns is None else (columns,))
            or math.prod(shape) * actual_dtype.itemsize != len(data) - stream.tell()):
        raise ValueError("Multi-view scene array exceeds its bounds")
    return np.load(io.BytesIO(data), allow_pickle=False)


def encode_grasp_multiview_snapshot(
    *, views: Sequence[bytes], metadata: Mapping,
    scene_points: np.ndarray | None = None,
    scene_target_mask: np.ndarray | None = None,
) -> bytes:
    if not 1 <= len(views) <= 3:
        raise ValueError("A grasp observation requires one to three views")
    if (scene_points is None) != (scene_target_mask is None):
        raise ValueError("Scene points and labels must be provided together")
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("metadata.json", json.dumps(dict(metadata), allow_nan=False))
        for index, view in enumerate(views):
            decode_grasp_rgbd_snapshot(view)
            archive.writestr(f"view_{index}.npz", view)
        if scene_points is not None:
            archive.writestr("points.npy", _array_bytes(np.asarray(scene_points, dtype=np.float64)))
            archive.writestr("target_mask.npy", _array_bytes(np.asarray(scene_target_mask, dtype=np.bool_)))
    data = stream.getvalue()
    decode_grasp_multiview_snapshot(data)
    return data


def decode_grasp_multiview_snapshot(data: bytes) -> tuple[list[bytes], dict, np.ndarray | None, np.ndarray | None]:
    if not 0 < len(data) <= GRASP_MULTIVIEW_MAX_BYTES:
        raise ValueError("Multi-view observation exceeds its byte bound")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
        names = {item.filename for item in infos}
        if (len(infos) != len(names)
                or sum(item.file_size for item in infos) > GRASP_MULTIVIEW_MAX_BYTES
                or "metadata.json" not in names
                or archive.getinfo("metadata.json").file_size > 256 * 1024):
            raise ValueError("Invalid multi-view observation members")
        view_names = sorted(name for name in names if name.startswith("view_"))
        expected = {f"view_{i}.npz" for i in range(len(view_names))} | {"metadata.json"}
        has_scene = "points.npy" in names
        if has_scene:
            expected |= {"points.npy", "target_mask.npy"}
        if not 1 <= len(view_names) <= 3 or names != expected:
            raise ValueError("Invalid multi-view observation layout")
        metadata = json.loads(archive.read("metadata.json"))
        if not isinstance(metadata, dict) or metadata.get("schema_id") != GRASP_MULTIVIEW_SCHEMA_ID:
            raise ValueError("Invalid multi-view observation schema")
        views = [archive.read(name) for name in view_names]
        roles = []
        for view in views:
            _, view_metadata = decode_grasp_rgbd_snapshot(view)
            roles.append(view_metadata.get("role"))
        if (any(not isinstance(role, str) or not role for role in roles)
                or len(set(roles)) != len(roles)):
            raise ValueError("Multi-view observation roles must be unique")
        points = labels = None
        if has_scene:
            points = _read_array(archive.read("points.npy"), dtype="float64", columns=3)
            labels = _read_array(archive.read("target_mask.npy"), dtype="bool", columns=None)
            if (labels.shape != (len(points),) or not np.all(np.isfinite(points))
                    or not np.any(labels)):
                raise ValueError("Invalid retained multi-view scene")
        return views, metadata, points, labels
