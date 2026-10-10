"""MuJoCo renderer capture adapter for the shared segmentation Skill."""

from __future__ import annotations

import hashlib
import io

from PIL import Image, ImageOps

from agent_gateway.profiles.segmentation import LiveRgbSegmentationCapture

from .visual_observation import MujocoAgentRgbCaptureProvider


class MujocoSegmentationCapturePort:
    def __init__(
        self,
        *,
        rgb_provider: MujocoAgentRgbCaptureProvider,
        width: int = 640,
        height: int = 480,
        max_rgb_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        if not isinstance(rgb_provider, MujocoAgentRgbCaptureProvider):
            raise TypeError("rgb_provider must be typed.")
        for field_name, value, maximum in (
            ("width", width, 1920),
            ("height", height, 1080),
            ("max_rgb_bytes", max_rgb_bytes, 64 * 1024 * 1024),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= maximum
            ):
                raise ValueError(f"{field_name} is outside its reviewed bound.")
        self.rgb_provider = rgb_provider
        self.width = width
        self.height = height
        self.max_rgb_bytes = max_rgb_bytes

    def capture_segment_rgb(
        self, *, role: str, expected_station_revision: int
    ) -> LiveRgbSegmentationCapture:
        capture = self.rgb_provider.capture(
            {
                "selected_roles": [role],
                "width": self.width,
                "height": self.height,
                "max_total_bytes": self.max_rgb_bytes,
            },
            expected_station_revision=expected_station_revision,
        )
        if len(capture.frames) != 1:
            raise RuntimeError("Segmentation capture did not return exactly one frame.")
        source_id, captured_role, frame = capture.frames[0]
        try:
            source = Image.open(io.BytesIO(frame.data))
            if source.format != "JPEG" or source.size != (self.width, self.height):
                raise ValueError("MuJoCo segmentation frame identity changed.")
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()
        except OSError as exc:
            raise ValueError("MuJoCo segmentation frame is not valid JPEG.") from exc
        return LiveRgbSegmentationCapture(
            source_id=source_id,
            role=captured_role,
            captured_at=capture.captured_at,
            station_revision=capture.station_revision,
            image=image,
            rgb_content_hash=hashlib.sha256(frame.data).hexdigest(),
        )


__all__ = ["MujocoSegmentationCapturePort"]
