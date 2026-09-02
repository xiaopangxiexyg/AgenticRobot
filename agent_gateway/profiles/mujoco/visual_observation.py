"""MuJoCo RGB role selection below the generic Skill/Artifact protocol."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from perception_contracts.artifacts import RGB_FRAME_MEDIA_TYPE_V1
from sim.mujoco.rgb_snapshot import EncodedMujocoRgbFrame

from .motion_application import MujocoAgentSkillMotionApplication

_ROLE_BY_SOURCE_ID = {
    "front": "main",
    "left_wrist": "wrist_left",
    "right_wrist": "wrist_right",
}


def mujoco_rgb_role_for_source(source_id: str) -> str:
    source = _required_text(source_id, field_name="source_id")
    return _ROLE_BY_SOURCE_ID.get(source, f"auxiliary_{source}")


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


@dataclass(frozen=True)
class MujocoAgentRgbCapture:
    captured_at: float
    station_revision: int
    frames: tuple[tuple[str, str, EncodedMujocoRgbFrame], ...]

    @property
    def capture_window_started_at(self) -> float:
        """MuJoCo renders every selected camera from one atomic scene cut."""

        return self.captured_at

    @property
    def capture_window_ended_at(self) -> float:
        return self.captured_at


class MujocoAgentRgbCaptureProvider:
    """Resolve trusted roles and capture one bounded, revision-stable scene cut."""

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

    def source_catalog(self) -> tuple[Mapping[str, object], ...]:
        sources = []
        for source in self._application.visual_source_catalog():
            source_id = str(source["source_id"])
            role = mujoco_rgb_role_for_source(source_id)
            sources.append({**dict(source), "role": role})
        return tuple(sources)

    def capture(
        self,
        parameters: Mapping[str, Any],
        *,
        expected_station_revision: int,
    ) -> MujocoAgentRgbCapture:
        parameter_fields = {"selected_roles", "width", "height", "max_total_bytes"}
        if not isinstance(parameters, Mapping) or set(parameters) != parameter_fields:
            raise ValueError("RGB capture parameters must use the exact schema fields.")
        if (
            isinstance(expected_station_revision, bool)
            or not isinstance(expected_station_revision, int)
            or expected_station_revision < 0
        ):
            raise ValueError("expected_station_revision must be non-negative.")
        if self._current_station_revision() != expected_station_revision:
            raise RuntimeError("rgb_capture_station_revision_mismatch")
        roles_raw = parameters.get("selected_roles")
        if isinstance(roles_raw, (str, bytes)) or not isinstance(
            roles_raw,
            Sequence,
        ):
            raise TypeError("selected_roles must be an array.")
        roles = tuple(
            _required_text(value, field_name="selected_roles[]") for value in roles_raw
        )
        if not roles or len(roles) > 8 or len(set(roles)) != len(roles):
            raise ValueError("selected_roles must contain one to eight unique roles.")
        width = _positive_int(
            parameters.get("width"),
            field_name="width",
            maximum=1920,
        )
        height = _positive_int(
            parameters.get("height"),
            field_name="height",
            maximum=1080,
        )
        max_total_bytes = _positive_int(
            parameters.get("max_total_bytes"),
            field_name="max_total_bytes",
            maximum=16 * 1024 * 1024,
        )
        catalog = self.source_catalog()
        selected: list[Mapping[str, object]] = []
        for role in roles:
            matches = [source for source in catalog if source["role"] == role]
            if len(matches) != 1:
                raise ValueError(
                    f"RGB role {role!r} does not resolve to exactly one source."
                )
            selected.append(matches[0])
        source_ids = tuple(str(source["source_id"]) for source in selected)
        snapshot = self._application.capture_visual_jpeg_snapshot(
            source_ids=source_ids,
            width=width,
            height=height,
        )
        if self._current_station_revision() != expected_station_revision:
            raise RuntimeError("rgb_capture_station_revision_changed")
        captured: list[tuple[str, str, EncodedMujocoRgbFrame]] = []
        total_bytes = 0
        for source in selected:
            source_id = str(source["source_id"])
            frame = snapshot.frames[source_id]
            if (
                frame.source_id != source_id
                or frame.width != width
                or frame.height != height
                or frame.media_type != RGB_FRAME_MEDIA_TYPE_V1
            ):
                raise RuntimeError("rgb_capture_frame_contract_mismatch")
            total_bytes += len(frame.data)
            if total_bytes > max_total_bytes:
                raise RuntimeError("rgb_capture_artifact_limit_exceeded")
            captured.append((source_id, str(source["role"]), frame))
        return MujocoAgentRgbCapture(
            captured_at=snapshot.captured_at,
            station_revision=expected_station_revision,
            frames=tuple(captured),
        )


__all__ = [
    "MujocoAgentRgbCapture",
    "MujocoAgentRgbCaptureProvider",
    "mujoco_rgb_role_for_source",
]
