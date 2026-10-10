"""Real lower-owned TCP-state adapter for placement planning."""

from __future__ import annotations

import math
from collections.abc import Callable

from agent_gateway.profiles.placement import PlacementRobotStateCapture
from robot_motion.coordinate_frames import CoordinateFrameGraph
from robot_motion.relative_action import pose_matrix_to_xyzquat
from teleop_runtime.skill_deploy.motion_port import RealAgentMotionPort


class RealPlacementRobotStatePort:
    def __init__(
        self,
        *,
        port: RealAgentMotionPort,
        frame_graph: CoordinateFrameGraph,
        component_id: str,
        execution_reference_frame_id: str,
        reference_frame_id: str,
        maximum_measurement_age_ms: int,
        current_station_revision: Callable[[], int],
        clock: Callable[[], float],
    ) -> None:
        if not isinstance(port, RealAgentMotionPort):
            raise TypeError("port must be typed.")
        if not isinstance(frame_graph, CoordinateFrameGraph):
            raise TypeError("frame_graph must be typed.")
        if not callable(current_station_revision) or not callable(clock):
            raise TypeError("revision and clock providers must be callable.")
        self.port = port
        self.frame_graph = frame_graph
        self.component_id = component_id
        self.execution_reference_frame_id = execution_reference_frame_id
        self.reference_frame_id = reference_frame_id
        self.maximum_measurement_age_ms = int(maximum_measurement_age_ms)
        self.current_station_revision = current_station_revision
        self.clock = clock

    def capture_placement_robot_state(
        self, *, expected_station_revision: int
    ) -> PlacementRobotStateCapture:
        if self.current_station_revision() != expected_station_revision:
            raise RuntimeError("placement_station_revision_mismatch")
        cut = self.port.latest_measured_pose()
        if cut is None or self.component_id not in cut.pose_world_by_component:
            raise RuntimeError("No lower measured TCP pose is available.")
        now = float(self.clock())
        if not math.isfinite(now) or cut.measured_at > now or cut.available_at > now:
            raise RuntimeError("Lower measured TCP pose is not causally available.")
        if (now - cut.measured_at) * 1_000.0 > self.maximum_measurement_age_ms:
            raise RuntimeError("Lower measured TCP pose is stale.")
        public_pose, resolution = self.frame_graph.transform_pose_matrix(
            cut.pose_world_by_component[self.component_id],
            source_reference_frame_id=self.execution_reference_frame_id,
            target_reference_frame_id=self.reference_frame_id,
        )
        pose = tuple(
            float(value)
            for value in pose_matrix_to_xyzquat(
                public_pose,
                name="real_placement.tool_tcp_reference",
            )
        )
        if self.current_station_revision() != expected_station_revision:
            raise RuntimeError("placement_station_revision_changed")
        return PlacementRobotStateCapture(
            component_id=self.component_id,
            measured_at=cut.measured_at,
            station_revision=expected_station_revision,
            tool_tcp_reference_xyz_quat_xyzw=pose,
            gripper_close_fraction=None,
            evidence_refs=(
                *cut.evidence_refs,
                "real-placement-frame-resolution:" + resolution.semantic_content_hash,
            ),
        )


__all__ = ["RealPlacementRobotStatePort"]
