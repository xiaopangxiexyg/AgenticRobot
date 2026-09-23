"""RealSense RGB capture adapter for the shared segmentation Skill."""

from __future__ import annotations

import hashlib

from agent_gateway.profiles.segmentation import LiveRgbSegmentationCapture

from .localization_gateway import decode_bounded_image
from .realsense_perception import RealSenseAgentPerceptionProvider


class RealSenseSegmentationCapturePort:
    def __init__(
        self,
        *,
        perception_provider: RealSenseAgentPerceptionProvider,
        max_rgb_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        if not isinstance(perception_provider, RealSenseAgentPerceptionProvider):
            raise TypeError("perception_provider must be typed.")
        if (
            isinstance(max_rgb_bytes, bool)
            or not isinstance(max_rgb_bytes, int)
            or max_rgb_bytes <= 0
        ):
            raise ValueError("max_rgb_bytes must be positive.")
        self.perception_provider = perception_provider
        self.max_rgb_bytes = max_rgb_bytes

    def capture_segment_rgb(
        self, *, role: str, expected_station_revision: int
    ) -> LiveRgbSegmentationCapture:
        catalog = {
            str(item["role"]): item
            for item in self.perception_provider.rgb_source_catalog()
        }
        try:
            source = catalog[role]
        except KeyError as exc:
            raise ValueError("Segmentation RGB role is not registered.") from exc
        width = int(source["native_width"])
        height = int(source["native_height"])
        capture = self.perception_provider.capture_rgb(
            {
                "selected_roles": [role],
                "width": width,
                "height": height,
                "max_total_bytes": self.max_rgb_bytes,
            },
            expected_station_revision=expected_station_revision,
        )
        if len(capture.frames) != 1:
            raise RuntimeError("Segmentation capture did not return exactly one frame.")
        frame = capture.frames[0]
        image = decode_bounded_image(
            frame.data,
            expected_media_type=frame.media_type,
            maximum_pixels=width * height,
        )
        return LiveRgbSegmentationCapture(
            source_id=frame.source_id,
            role=frame.role,
            captured_at=frame.acquired_at,
            station_revision=capture.station_revision,
            image=image,
            rgb_content_hash=hashlib.sha256(frame.data).hexdigest(),
        )


__all__ = ["RealSenseSegmentationCapturePort"]
