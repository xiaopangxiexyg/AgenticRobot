"""On-demand RGB snapshot provider for the MuJoCo Agent Gateway."""

from __future__ import annotations

import base64
import hashlib
import io
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from PIL import Image

from dataset_contracts.canonical_encoding import canonical_content_hash

from .motion_application import MujocoAgentSkillMotionApplication

MUJOCO_AGENT_VISUAL_CATALOG_SCHEMA_V1 = (
    "dexteleop.agent_visual_source_catalog.v1"
)
MUJOCO_AGENT_VISUAL_SNAPSHOT_SCHEMA_V1 = "dexteleop.agent_visual_snapshot.v1"
MUJOCO_AGENT_VISUAL_FRAME_SCHEMA_V1 = "dexteleop.agent_visual_frame.v1"
_VISUAL_SNAPSHOT_HASH_DOMAIN_V1 = "dexteleop.agent_visual_snapshot.v1"
_ROLE_BY_SOURCE_ID = {
    "front": "main",
    "left_wrist": "wrist_left",
    "right_wrist": "wrist_right",
}


def _required_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty trimmed string.")
    return value


def _positive_int(value: object, *, field_name: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= maximum
    ):
        raise ValueError(f"{field_name} must be an integer in [1, {maximum}].")
    return value


class MujocoAgentVisualSnapshotProvider:
    """Expose all live RGB sources while honoring a bounded semantic selection."""

    def __init__(
        self,
        *,
        application: MujocoAgentSkillMotionApplication,
        current_station_revision: Callable[[], int],
    ) -> None:
        if not isinstance(application, MujocoAgentSkillMotionApplication):
            raise TypeError("application must be a motion application.")
        if not callable(current_station_revision):
            raise TypeError("current_station_revision must be callable.")
        self._application = application
        self._current_station_revision = current_station_revision

    def source_catalog(self) -> Mapping[str, object]:
        sources = self._catalog_sources()
        return {
            "schema": MUJOCO_AGENT_VISUAL_CATALOG_SCHEMA_V1,
            "sources": sources,
            "supported_media_types": ["image/jpeg"],
            "max_images_per_snapshot": 8,
            "max_width": 1920,
            "max_height": 1080,
        }

    def capture(self, request: Mapping[str, Any]) -> Mapping[str, object]:
        if not isinstance(request, Mapping):
            raise TypeError("visual snapshot request must be an object.")
        expected_fields = {
            "client_id",
            "request_id",
            "selected_roles",
            "max_images",
            "width",
            "height",
            "max_total_bytes",
            "expected_station_revision",
        }
        if set(request) != expected_fields:
            raise ValueError(
                "visual snapshot request fields must be exactly "
                f"{sorted(expected_fields)}."
            )
        request_id = _required_text(request["request_id"], field_name="request_id")
        roles_raw = request["selected_roles"]
        if isinstance(roles_raw, (str, bytes)) or not isinstance(roles_raw, Sequence):
            raise TypeError("selected_roles must be an array.")
        roles = tuple(
            _required_text(value, field_name="selected_roles[]") for value in roles_raw
        )
        if not roles or len(set(roles)) != len(roles):
            raise ValueError("selected_roles must be non-empty and unique.")
        max_images = _positive_int(
            request["max_images"], field_name="max_images", maximum=8
        )
        if len(roles) > max_images:
            raise ValueError("selected_roles exceeds max_images.")
        width = _positive_int(request["width"], field_name="width", maximum=1920)
        height = _positive_int(request["height"], field_name="height", maximum=1080)
        max_total_bytes = _positive_int(
            request["max_total_bytes"],
            field_name="max_total_bytes",
            maximum=16 * 1024 * 1024,
        )
        expected_revision = request["expected_station_revision"]
        if (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
            or expected_revision < 0
        ):
            raise ValueError("expected_station_revision must be non-negative.")
        if self._current_station_revision() != expected_revision:
            raise RuntimeError("visual_snapshot_station_revision_mismatch")

        catalog = self._catalog_sources()
        selected = []
        for role in roles:
            matches = [source for source in catalog if source["role"] == role]
            if len(matches) != 1:
                raise ValueError(
                    f"visual role {role!r} does not resolve to exactly one source."
                )
            selected.append(matches[0])
        source_ids = tuple(str(source["source_id"]) for source in selected)
        arrays = self._application.capture_visual_frames(
            source_ids=source_ids,
            width=width,
            height=height,
        )
        if self._current_station_revision() != expected_revision:
            raise RuntimeError("visual_snapshot_station_revision_changed")

        frames: list[dict[str, object]] = []
        total_bytes = 0
        for source in selected:
            source_id = str(source["source_id"])
            encoded = _encode_jpeg(arrays[source_id])
            total_bytes += len(encoded)
            if total_bytes > max_total_bytes:
                raise RuntimeError("visual_snapshot_payload_limit_exceeded")
            media_hash = hashlib.sha256(encoded).hexdigest()
            frames.append(
                {
                    "schema": MUJOCO_AGENT_VISUAL_FRAME_SCHEMA_V1,
                    "frame_id": f"visual-frame:{media_hash}",
                    "source_id": source_id,
                    "role": source["role"],
                    "media_type": "image/jpeg",
                    "width": width,
                    "height": height,
                    "media_content_hash": media_hash,
                    "data_base64": base64.b64encode(encoded).decode("ascii"),
                }
            )
        identity = canonical_content_hash(
            {
                "request_id": request_id,
                "station_revision": expected_revision,
                "frames": [
                    {
                        key: value
                        for key, value in frame.items()
                        if key != "data_base64"
                    }
                    for frame in frames
                ],
            },
            domain=_VISUAL_SNAPSHOT_HASH_DOMAIN_V1,
        )
        return {
            "schema": MUJOCO_AGENT_VISUAL_SNAPSHOT_SCHEMA_V1,
            "snapshot_id": f"visual-snapshot:{identity}",
            "request_id": request_id,
            "station_revision": expected_revision,
            "frames": frames,
            "total_bytes": total_bytes,
        }

    def _catalog_sources(self) -> list[dict[str, object]]:
        sources = []
        for source in self._application.visual_source_catalog():
            source_id = str(source["source_id"])
            role = _ROLE_BY_SOURCE_ID.get(source_id, f"auxiliary_{source_id}")
            sources.append({**dict(source), "role": role})
        return sources


def _encode_jpeg(image: object) -> bytes:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise RuntimeError("MuJoCo visual source did not return uint8 HWC RGB.")
    output = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(
        output,
        format="JPEG",
        quality=85,
        optimize=True,
    )
    return output.getvalue()


__all__ = [
    "MUJOCO_AGENT_VISUAL_CATALOG_SCHEMA_V1",
    "MUJOCO_AGENT_VISUAL_FRAME_SCHEMA_V1",
    "MUJOCO_AGENT_VISUAL_SNAPSHOT_SCHEMA_V1",
    "MujocoAgentVisualSnapshotProvider",
]
