from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from robot_motion.coordinate_frames import (
    CoordinateFrameGraph,
    CoordinateFrameSpec,
    RigidFrameTransformSpec,
)
from robot_motion.coordinate_profiles import (
    SessionAnchorReceipt,
    SessionPoseAnchor,
    TargetFrameProfile,
    ToolFrameBinding,
    ToolFrameProfile,
)


def _pose(x: float = 0.0) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[0, 3] = x
    return value


def _profiles() -> tuple[TargetFrameProfile, ToolFrameProfile]:
    graph = CoordinateFrameGraph(
        graph_id="target-profile-fixture",
        frames=(
            CoordinateFrameSpec("station"),
            CoordinateFrameSpec("arm_base"),
        ),
        transforms=(
            RigidFrameTransformSpec(
                transform_id="base-to-station",
                source_frame_id="arm_base",
                target_frame_id="station",
                matrix_target_source=_pose(1.0),
            ),
        ),
    )
    target = TargetFrameProfile(
        profile_id="target-profile",
        target_kind="simulation",
        artifact_id="fixture.urdf",
        artifact_hash="urdf-hash",
        frame_graph=graph,
        canonical_reference_frame_id="station",
        reference_frame_by_component={"arm": "arm_base"},
    )
    tool = ToolFrameProfile(
        profile_id="tool-profile",
        target_frame_profile_hash=target.semantic_content_hash,
        bindings=(
            ToolFrameBinding(
                component_id="arm",
                controlled_frame_id="tool_tcp",
                artifact_id="tool.json",
                artifact_hash="tool-hash",
            ),
        ),
    )
    return target, tool


def test_session_anchor_receipt_closes_target_frame_transform() -> None:
    target, tool = _profiles()
    target_home_base = _pose(0.2)
    target_home_station = _pose(1.2)
    receipt = SessionAnchorReceipt.capture(
        session_id="session",
        coordinate_profile_id="vive-coordinate",
        coordinate_profile_hash="coordinate-hash",
        target_frame_profile=target,
        tool_frame_profile=tool,
        source_anchor_origin="stored-human-home",
        target_anchor_origin="measured-reset-state",
        captured_at=10.0,
        clock_domain_id="sim-clock",
        anchors=(
            SessionPoseAnchor(
                component_id="arm",
                controlled_frame_id="tool_tcp",
                target_reference_frame_id="arm_base",
                source_home_pose_canonical=_pose(-0.1),
                target_home_pose_canonical=target_home_station,
                target_home_pose_reference=target_home_base,
            ),
        ),
    )

    assert receipt.anchor_by_component["arm"].target_reference_frame_id == "arm_base"
    assert receipt.target_frame_profile_hash == target.semantic_content_hash
    assert receipt.tool_frame_profile_hash == tool.semantic_content_hash
    assert receipt.to_payload()["receipt_id"] == receipt.receipt_id


def test_session_anchor_rejects_a_hidden_coordinate_mismatch() -> None:
    target, tool = _profiles()
    with pytest.raises(ValueError, match="target Home transform mismatch"):
        SessionAnchorReceipt.capture(
            session_id="session",
            coordinate_profile_id="vive-coordinate",
            coordinate_profile_hash="coordinate-hash",
            target_frame_profile=target,
            tool_frame_profile=tool,
            source_anchor_origin="stored-human-home",
            target_anchor_origin="measured-reset-state",
            captured_at=10.0,
            clock_domain_id="sim-clock",
            anchors=(
                SessionPoseAnchor(
                    component_id="arm",
                    controlled_frame_id="tool_tcp",
                    target_reference_frame_id="arm_base",
                    source_home_pose_canonical=_pose(),
                    target_home_pose_canonical=_pose(9.0),
                    target_home_pose_reference=_pose(0.2),
                ),
            ),
        )


def test_session_anchor_rejects_wrong_tool_or_target_identity() -> None:
    target, tool = _profiles()
    entry = SessionPoseAnchor(
        component_id="arm",
        controlled_frame_id="tool_tcp",
        target_reference_frame_id="arm_base",
        source_home_pose_canonical=_pose(),
        target_home_pose_canonical=_pose(1.2),
        target_home_pose_reference=_pose(0.2),
    )
    with pytest.raises(ValueError, match="another target-frame profile"):
        SessionAnchorReceipt.capture(
            session_id="session",
            coordinate_profile_id="vive-coordinate",
            coordinate_profile_hash="coordinate-hash",
            target_frame_profile=target,
            tool_frame_profile=replace(
                tool,
                target_frame_profile_hash="wrong-target-hash",
            ),
            source_anchor_origin="stored-human-home",
            target_anchor_origin="measured-reset-state",
            captured_at=10.0,
            clock_domain_id="sim-clock",
            anchors=(entry,),
        )

    with pytest.raises(ValueError, match="controlled frame mismatch"):
        SessionAnchorReceipt.capture(
            session_id="session",
            coordinate_profile_id="vive-coordinate",
            coordinate_profile_hash="coordinate-hash",
            target_frame_profile=target,
            tool_frame_profile=tool,
            source_anchor_origin="stored-human-home",
            target_anchor_origin="measured-reset-state",
            captured_at=10.0,
            clock_domain_id="sim-clock",
            anchors=(replace(entry, controlled_frame_id="flange"),),
        )
