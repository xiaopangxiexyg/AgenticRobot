"""Typed real-perception source-to-Station frame binding.

RealSense calibration provides ``T_source_camera`` poses.  This module owns
the explicit adapter that turns those poses into the canonical Station frame
used by consumers.  It is pure: constructing or applying a binding opens no
camera, robot, socket, or other runtime resource.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.coordinate_frames import (
    CoordinateFrameGraph,
    CoordinateFrameSpec,
    RigidFrameTransformSpec,
)
from robot_motion.coordinate_profiles import STATION_RETARGET_FRAME_CONTRACT_V1
from robot_motion.relative_action import (
    pose_matrix_to_xyzquat,
    xyzquat_to_pose_matrix,
)

REAL_PERCEPTION_SOURCE_FRAME_ID_V1 = "dual_arm_solver_world"
REAL_PERCEPTION_STATION_FRAME_ID_V1 = "canonical_station"


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty trimmed string.")
    return value


@dataclass(frozen=True)
class RealPerceptionFrameBinding:
    """One immutable source-reference to consumer-Station adapter."""

    frame_graph: CoordinateFrameGraph
    source_frame_id: str
    reference_frame_id: str
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.frame_graph, CoordinateFrameGraph):
            raise TypeError("frame_graph must be a CoordinateFrameGraph.")
        source = _required(self.source_frame_id, field_name="source_frame_id")
        reference = _required(
            self.reference_frame_id,
            field_name="reference_frame_id",
        )
        if source == reference:
            raise ValueError("Real depth source and reference frames must differ.")
        if source not in self.frame_graph.frame_by_id:
            raise ValueError("Real depth source frame is absent from the graph.")
        if reference not in self.frame_graph.frame_by_id:
            raise ValueError("Real depth reference frame is absent from the graph.")
        if not self.frame_graph.connected(source, reference):
            raise ValueError("Real depth source and reference frames are disconnected.")
        object.__setattr__(self, "source_frame_id", source)
        object.__setattr__(self, "reference_frame_id", reference)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                {
                    "source_frame_id": source,
                    "source_frame_spec_hash": self.source_frame_spec_hash,
                    "reference_frame_id": reference,
                    "reference_frame_spec_hash": self.reference_frame_spec_hash,
                    "frame_graph_hash": self.frame_graph.semantic_content_hash,
                    "source_pose_semantics": "T_source_camera_optical_frame",
                    "result_pose_semantics": "T_reference_camera_optical_frame",
                },
                domain="dexteleop.real_perception.frame_binding.v1",
            ),
        )

    @property
    def source_frame_spec_hash(self) -> str:
        return self.frame_graph.frame_by_id[self.source_frame_id].semantic_content_hash

    @property
    def reference_frame_spec_hash(self) -> str:
        return self.frame_graph.frame_by_id[
            self.reference_frame_id
        ].semantic_content_hash

    def camera_pose_reference_xyz_quat_xyzw(
        self,
        camera_pose_source_xyz_quat_xyzw: object,
    ) -> tuple[float, ...]:
        """Convert ``T_source_camera`` into ``T_reference_camera``."""

        pose_source_camera = xyzquat_to_pose_matrix(
            camera_pose_source_xyz_quat_xyzw,
            name="camera_pose_source_xyz_quat_xyzw",
        )
        pose_reference_camera, resolution = self.frame_graph.transform_pose_matrix(
            pose_source_camera,
            source_reference_frame_id=self.source_frame_id,
            target_reference_frame_id=self.reference_frame_id,
        )
        if resolution.frame_graph_hash != self.frame_graph.semantic_content_hash:
            raise RuntimeError(
                "Real depth frame resolution changed its graph identity."
            )
        return tuple(
            float(value)
            for value in pose_matrix_to_xyzquat(
                pose_reference_camera,
                name="camera_pose_reference_xyz_quat_xyzw",
            )
        )


def build_real_perception_station_frame_binding(
    *,
    collision_scene_profile: str,
    scene_mount_hash: str,
    scene_world_from_canonical_station: object,
) -> RealPerceptionFrameBinding:
    """Build the real RM65 source-world to canonical-Station graph."""

    profile = _required(
        collision_scene_profile,
        field_name="collision_scene_profile",
    )
    mounts_hash = _required(scene_mount_hash, field_name="scene_mount_hash")
    scene_world_from_station = np.asarray(
        scene_world_from_canonical_station,
        dtype=float,
    )
    if scene_world_from_station.shape != (4, 4):
        raise ValueError("scene_world_from_canonical_station must be 4x4.")
    canonical_station_from_source = np.linalg.inv(scene_world_from_station)
    source_frame = CoordinateFrameSpec(
        frame_id=REAL_PERCEPTION_SOURCE_FRAME_ID_V1,
        frame_version="1",
        semantic_parameters={
            "owner": "rm65_collision_scene",
            "collision_scene_profile": profile,
            "scene_mount_hash": mounts_hash,
        },
    )
    station_frame = CoordinateFrameSpec(
        frame_id=REAL_PERCEPTION_STATION_FRAME_ID_V1,
        frame_version="1",
        semantic_parameters={
            "axis_contract": STATION_RETARGET_FRAME_CONTRACT_V1,
        },
    )
    graph = CoordinateFrameGraph(
        graph_id="exclusive-rm65-skill-station-v2",
        graph_version="2",
        frames=(source_frame, station_frame),
        transforms=(
            RigidFrameTransformSpec(
                transform_id="rm65-solver-world-to-canonical-station-v2",
                transform_version="2",
                source_frame_id=source_frame.frame_id,
                target_frame_id=station_frame.frame_id,
                matrix_target_source=canonical_station_from_source,
                semantic_parameters={
                    "basis": "inverse_collision_scene_urdf_fixed_joint_chain",
                    "collision_scene_profile": profile,
                },
            ),
        ),
    )
    return RealPerceptionFrameBinding(
        frame_graph=graph,
        source_frame_id=source_frame.frame_id,
        reference_frame_id=station_frame.frame_id,
    )


__all__ = [
    "REAL_PERCEPTION_SOURCE_FRAME_ID_V1",
    "REAL_PERCEPTION_STATION_FRAME_ID_V1",
    "RealPerceptionFrameBinding",
    "build_real_perception_station_frame_binding",
]
