"""RealSense aligned-depth adapter for the shared back-projection Skill."""

from __future__ import annotations

import numpy as np
from PIL import Image

from agent_gateway.profiles.back_projection import (
    CorrelatedPlacementCapture,
    LiveAlignedDepthCapture,
)
from agent_gateway.profiles.segmentation import LiveRgbSegmentationCapture
from robot_motion.relative_action import (
    pose_matrix_to_xyzquat,
    xyzquat_to_pose_matrix,
)

from .perception_frames import RealPerceptionFrameBinding
from .realsense_perception import RealSenseAgentPerceptionProvider


def live_aligned_depth_capture(
    capture,
    *,
    frame_binding: RealPerceptionFrameBinding,
) -> LiveAlignedDepthCapture:
    """Project one deployment-owned aligned RGB-D cut into the Station frame."""

    source_from_camera = np.asarray(
        capture.source_from_camera, dtype=np.float64
    ).reshape(4, 4)
    source_pose = pose_matrix_to_xyzquat(
        source_from_camera,
        name="real_back_projection.source_from_camera",
    )
    reference_pose = frame_binding.camera_pose_reference_xyz_quat_xyzw(source_pose)
    reference_from_camera = xyzquat_to_pose_matrix(
        reference_pose,
        name="real_back_projection.reference_from_camera",
    )
    return LiveAlignedDepthCapture(
        source_id=capture.source_id,
        role=capture.role,
        captured_at=capture.acquired_at,
        station_revision=capture.station_revision,
        capture_group_id=capture.capture_group_id,
        depth_content_hash=capture.depth_content_hash,
        intrinsics=capture.intrinsics,
        reference_from_camera=reference_from_camera,
        depth_m=capture.depth_m,
    )


class RealSenseBackProjectionCapturePort:
    def __init__(
        self,
        *,
        perception_provider: RealSenseAgentPerceptionProvider,
        frame_binding: RealPerceptionFrameBinding,
    ) -> None:
        if not isinstance(perception_provider, RealSenseAgentPerceptionProvider):
            raise TypeError("perception_provider must be typed.")
        if not isinstance(frame_binding, RealPerceptionFrameBinding):
            raise TypeError("frame_binding must be typed.")
        self.perception_provider = perception_provider
        self.frame_binding = frame_binding

    def capture_aligned_depth(
        self, *, role: str, expected_station_revision: int
    ) -> LiveAlignedDepthCapture:
        capture = self.perception_provider.capture_aligned_rgbd(
            role=role,
            expected_station_revision=expected_station_revision,
        )
        return live_aligned_depth_capture(
            capture,
            frame_binding=self.frame_binding,
        )

    def capture_placement_rgbd(
        self, *, role: str, expected_station_revision: int
    ) -> CorrelatedPlacementCapture:
        capture = self.perception_provider.capture_aligned_rgbd(
            role=role,
            expected_station_revision=expected_station_revision,
        )
        return CorrelatedPlacementCapture(
            rgb=LiveRgbSegmentationCapture(
                source_id=capture.source_id,
                role=capture.role,
                captured_at=capture.acquired_at,
                station_revision=capture.station_revision,
                image=Image.fromarray(capture.rgb),
                rgb_content_hash=capture.rgb_content_hash,
            ),
            depth=live_aligned_depth_capture(
                capture,
                frame_binding=self.frame_binding,
            ),
        )


__all__ = ["RealSenseBackProjectionCapturePort", "live_aligned_depth_capture"]
