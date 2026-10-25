"""MuJoCo metric-depth adapter for the shared back-projection Skill."""

from __future__ import annotations

import hashlib
import io

import numpy as np
from PIL import Image, ImageOps

from agent_gateway.profiles.back_projection import (
    CorrelatedPlacementCapture,
    LiveAlignedDepthCapture,
)
from agent_gateway.profiles.segmentation import LiveRgbSegmentationCapture
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.deploy.schema import xyzquat_to_pose_matrix

from .motion_application import MujocoAgentSkillMotionApplication


def live_aligned_depth_capture(
    frame,
    *,
    source_id: str,
    role: str,
    station_revision: int,
    application: MujocoAgentSkillMotionApplication,
    source_frame_id: str,
    reference_frame_id: str,
) -> LiveAlignedDepthCapture:
    """Convert one MuJoCo depth frame into the shared Station-frame cut."""

    try:
        depth = np.load(io.BytesIO(frame.data), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise RuntimeError("MuJoCo depth frame is not valid NPY.") from exc
    return live_aligned_depth_capture_from_array(
        depth, captured_at=frame.captured_at,
        camera_pose_world_xyz_quat_xyzw=frame.camera_pose_world_xyz_quat_xyzw,
        intrinsics=frame.intrinsics, source_id=source_id, role=role,
        station_revision=station_revision, application=application,
        source_frame_id=source_frame_id, reference_frame_id=reference_frame_id)


def live_aligned_depth_capture_from_array(
    depth, *, captured_at: float, camera_pose_world_xyz_quat_xyzw,
    intrinsics, source_id: str, role: str, station_revision: int,
    application: MujocoAgentSkillMotionApplication,
    source_frame_id: str, reference_frame_id: str,
) -> LiveAlignedDepthCapture:
    """Share exact depth/capture identity for encoded and raw native cuts."""
    pose_reference = application.transform_tcp_pose_xyzquat(
        camera_pose_world_xyz_quat_xyzw,
        source_frame_id=source_frame_id,
        target_frame_id=reference_frame_id,
    )
    depth_hash = canonical_content_hash(
        {
            "dtype": "float32",
            "unit": "meter",
            "invalid_value": 0.0,
            "shape": list(depth.shape),
            "byte_sha256": hashlib.sha256(
                np.asarray(depth, dtype=np.float32).tobytes(order="C")
            ).hexdigest(),
        },
        domain="dexteleop.mujoco_back_projection.depth_content.v1",
    )
    capture_hash = canonical_content_hash(
        {
            "source_id": source_id,
            "role": role,
            "captured_at": captured_at,
            "station_revision": station_revision,
            "depth_content_hash": depth_hash,
            "camera_pose_reference_xyz_quat_xyzw": pose_reference,
        },
        domain="dexteleop.mujoco_back_projection.capture_group.v1",
    )
    return LiveAlignedDepthCapture(
        source_id=source_id,
        role=role,
        captured_at=captured_at,
        station_revision=station_revision,
        capture_group_id="mujoco-depth:" + capture_hash,
        depth_content_hash=depth_hash,
        intrinsics=intrinsics,
        reference_from_camera=xyzquat_to_pose_matrix(pose_reference),
        depth_m=np.asarray(depth, dtype=np.float32),
    )


class MujocoBackProjectionCapturePort:
    def __init__(
        self,
        *,
        application: MujocoAgentSkillMotionApplication,
        source_id_by_role: dict[str, str],
        source_frame_id: str,
        reference_frame_id: str,
        current_station_revision,
    ) -> None:
        if not isinstance(application, MujocoAgentSkillMotionApplication):
            raise TypeError("application must be a motion application.")
        if not source_id_by_role or len(set(source_id_by_role.values())) != len(
            source_id_by_role
        ):
            raise ValueError("source_id_by_role must be a non-empty one-to-one map.")
        if not callable(current_station_revision):
            raise TypeError("current_station_revision must be callable.")
        self.application = application
        self.source_id_by_role = dict(source_id_by_role)
        self.source_frame_id = source_frame_id
        self.reference_frame_id = reference_frame_id
        self.current_station_revision = current_station_revision

    def capture_aligned_depth(
        self, *, role: str, expected_station_revision: int
    ) -> LiveAlignedDepthCapture:
        try:
            source_id = self.source_id_by_role[role]
        except KeyError as exc:
            raise ValueError("Back-projection role is not registered.") from exc
        if self.current_station_revision() != expected_station_revision:
            raise RuntimeError("back_projection_station_revision_mismatch")
        frame = self.application.capture_depth_npy_frame(view_id=source_id)
        if self.current_station_revision() != expected_station_revision:
            raise RuntimeError("back_projection_station_revision_changed")
        return live_aligned_depth_capture(
            frame,
            source_id=source_id,
            role=role,
            station_revision=expected_station_revision,
            application=self.application,
            source_frame_id=self.source_frame_id,
            reference_frame_id=self.reference_frame_id,
        )

    def capture_placement_rgbd(
        self, *, role: str, expected_station_revision: int
    ) -> CorrelatedPlacementCapture:
        try:
            source_id = self.source_id_by_role[role]
        except KeyError as exc:
            raise ValueError("Placement role is not registered.") from exc
        if self.current_station_revision() != expected_station_revision:
            raise RuntimeError("placement_station_revision_mismatch")
        rgb_frame, depth_frame = self.application.capture_visual_depth_cut(
            view_id=source_id
        )
        try:
            with Image.open(io.BytesIO(rgb_frame.data)) as source:
                if source.format != "JPEG" or source.size != (
                    depth_frame.width,
                    depth_frame.height,
                ):
                    raise ValueError("MuJoCo placement RGB identity changed.")
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.load()
        except OSError as exc:
            raise ValueError("MuJoCo placement RGB frame is invalid.") from exc
        if self.current_station_revision() != expected_station_revision:
            raise RuntimeError("placement_station_revision_changed")
        return CorrelatedPlacementCapture(
            rgb=LiveRgbSegmentationCapture(
                source_id=source_id,
                role=role,
                captured_at=depth_frame.captured_at,
                station_revision=expected_station_revision,
                image=image,
                rgb_content_hash=hashlib.sha256(rgb_frame.data).hexdigest(),
            ),
            depth=live_aligned_depth_capture(
                depth_frame,
                source_id=source_id,
                role=role,
                station_revision=expected_station_revision,
                application=self.application,
                source_frame_id=self.source_frame_id,
                reference_frame_id=self.reference_frame_id,
            ),
        )


__all__ = ["MujocoBackProjectionCapturePort", "live_aligned_depth_capture", "live_aligned_depth_capture_from_array"]
