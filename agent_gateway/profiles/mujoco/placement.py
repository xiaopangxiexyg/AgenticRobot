"""MuJoCo robot-state adapter for correlated RGB-D placement planning."""

from __future__ import annotations

from agent_gateway.profiles.placement import PlacementRobotStateCapture
from dataset_contracts.canonical_encoding import canonical_content_hash

from .motion_application import MujocoAgentSkillMotionApplication


class MujocoPlacementRobotStatePort:
    def __init__(
        self,
        *,
        application: MujocoAgentSkillMotionApplication,
        component_id: str,
        source_frame_id: str,
        reference_frame_id: str,
        current_station_revision,
    ) -> None:
        if not isinstance(application, MujocoAgentSkillMotionApplication):
            raise TypeError("application must be a motion application.")
        if not callable(current_station_revision):
            raise TypeError("current_station_revision must be callable.")
        self.application = application
        self.component_id = component_id
        self.source_frame_id = source_frame_id
        self.reference_frame_id = reference_frame_id
        self.current_station_revision = current_station_revision

    def capture_placement_robot_state(
        self, *, expected_station_revision: int
    ) -> PlacementRobotStateCapture:
        if self.current_station_revision() != expected_station_revision:
            raise RuntimeError("placement_station_revision_mismatch")
        captured = self.application.capture_arm_gripper_state()
        pose = self.application.transform_tcp_pose_xyzquat(
            captured["right_arm"],
            source_frame_id=self.source_frame_id,
            target_frame_id=self.reference_frame_id,
        )
        gripper = captured["right_gripper"]
        measured_at = float(gripper["measured_at"])
        close = float(gripper["close_fraction"])
        evidence_hash = canonical_content_hash(
            {
                "component_id": self.component_id,
                "station_revision": expected_station_revision,
                "measured_at": measured_at,
                "tool_tcp_reference_xyz_quat_xyzw": pose,
                "gripper_close_fraction": close,
                "state_source": gripper["state_source"],
            },
            domain="dexteleop.mujoco_placement.robot_state.v1",
        )
        if self.current_station_revision() != expected_station_revision:
            raise RuntimeError("placement_station_revision_changed")
        return PlacementRobotStateCapture(
            component_id=self.component_id,
            measured_at=measured_at,
            station_revision=expected_station_revision,
            tool_tcp_reference_xyz_quat_xyzw=pose,
            gripper_close_fraction=close,
            evidence_refs=("mujoco-placement-state:" + evidence_hash,),
        )


__all__ = ["MujocoPlacementRobotStatePort"]
