from __future__ import annotations

import numpy as np

from dataset_contracts.pose_frames import PoseFrameContract, PoseSourceProfile
from robot_motion.deploy.schema import ACTION_ARM_POSE_SLICES, ACTION_DIM
from robot_motion.pose_frame_profiles import default_rm65_workspace_near_boundary_runtime_pose_frame_contract
from robot_motion.replay.action_frame import replay_teleop_command_at_index
from robot_motion.replay.proposal_source import build_replay_source_frame_emission
from robot_motion.replay.trajectory import ReplayTrajectoryGroup
from teleop_runtime.replay.pose_frames import (
    replay_action_pose_frame,
    replay_action_poses_in_hardware_base_at_index,
    replay_group_in_hardware_base,
)


def _group(actions: np.ndarray, contract: PoseFrameContract | None) -> ReplayTrajectoryGroup:
    action_metadata = {}
    if contract is not None:
        action_metadata = {
            "pose_action_absolute_frame": "canonical_desk",
            "pose_frame_contract": contract.to_dict(),
        }
    return ReplayTrajectoryGroup(
        key="g",
        label="test",
        source_path="episode.parquet",
        source_episode_index=0,
        start_index=0,
        end_index=actions.shape[0],
        frame_count=actions.shape[0],
        action_hz=15.0,
        actions=actions,
        metadata={"action_metadata": action_metadata},
    )


def _desk_contract(*, storage_frame: str, raw_to_desk: np.ndarray) -> PoseFrameContract:
    return PoseFrameContract(
        source=PoseSourceProfile(
            source_id="test_umi",
            source_family="vive_umi",
            source_description="test source",
            topology="dual_wrist",
            active_sides=("left", "right"),
            raw_absolute_frame_by_side={"left": "vive", "right": "vive"},
            raw_to_desk_transform=raw_to_desk,
            desk_frame="desk",
        ),
        pose_storage_frame=storage_frame,
        canonical_absolute_frame="desk",
    )


def test_canonical_desk_group_is_bridged_to_each_rm65_controller_base():
    runtime = default_rm65_workspace_near_boundary_runtime_pose_frame_contract()
    actions = np.zeros((1, ACTION_DIM), dtype=np.float32)
    expected_base = {
        "left": np.array([0.31, -0.12, 0.42, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "right": np.array([0.27, 0.08, 0.38, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    }
    for side, pose in expected_base.items():
        actions[0, ACTION_ARM_POSE_SLICES[side]] = (
            runtime.storage_pose_vector_to_canonical_desk(pose, side=side)
        )
    group = _group(actions, _desk_contract(storage_frame="desk_absolute", raw_to_desk=np.eye(4)))

    resolved = replay_action_poses_in_hardware_base_at_index(
        group, ("left", "right"), index=0
    )
    for side in ("left", "right"):
        np.testing.assert_allclose(resolved[side], expected_base[side], atol=1e-6)

    detached = replay_group_in_hardware_base(group, ("left", "right"))
    assert replay_action_pose_frame(group) == "canonical_desk"
    assert replay_action_pose_frame(detached) == "arm_base"
    np.testing.assert_allclose(group.actions, actions)
    episode = detached.metadata["source_metadata"]["episode_metadata"]
    assert episode["arm_reference_frame_by_side"] == {
        "left": "left_arm_base",
        "right": "right_arm_base",
    }
    assert episode["arm_controlled_frame_by_side"] == {
        "left": "tool_tcp",
        "right": "tool_tcp",
    }
    command = replay_teleop_command_at_index(
        detached, ("left", "right"), index=0
    )
    for side in ("left", "right"):
        arm = command.commands[f"{side}_arm"]
        assert arm.frame_name == f"{side}_arm_base"
        assert arm.metadata["arm_controlled_frame"] == "tool_tcp"


def test_canonical_gui_emission_preserves_desk_frame_without_source_conversion():
    actions = np.zeros((1, ACTION_DIM), dtype=np.float32)
    pose = np.asarray([0.3, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    actions[0, ACTION_ARM_POSE_SLICES["left"]] = pose
    actions[0, ACTION_ARM_POSE_SLICES["right"]] = pose
    group = _group(
        actions,
        _desk_contract(storage_frame="desk_absolute", raw_to_desk=np.eye(4)),
    )

    emission = build_replay_source_frame_emission(
        group,
        active_sides=("left", "right"),
        index=0,
        include_arms=True,
        include_end_effectors=False,
        observed_at=10.0,
        schedule_start_at=10.0,
    )

    assert emission.action.arm_frame("left") == "canonical_desk"
    assert emission.action.arm_frame("right") == "canonical_desk"
    np.testing.assert_allclose(emission.action.arm_pose_by_side["left"], pose)


def test_raw_absolute_nonidentity_is_applied_once_before_hardware_bridge():
    raw_to_desk = np.eye(4, dtype=float)
    raw_to_desk[:3, 3] = [0.4, -0.2, 0.1]
    contract = _desk_contract(storage_frame="raw_absolute", raw_to_desk=raw_to_desk)
    raw_pose = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    desk_pose = contract.storage_pose_vector_to_canonical_desk(raw_pose, side="left")
    np.testing.assert_allclose(desk_pose[:3], [0.5, 0.0, 0.4], atol=1e-6)

    actions = np.zeros((1, ACTION_DIM), dtype=np.float32)
    actions[0, ACTION_ARM_POSE_SLICES["left"]] = desk_pose
    actions[0, ACTION_ARM_POSE_SLICES["right"]] = desk_pose
    group = _group(actions, contract)
    expected = default_rm65_workspace_near_boundary_runtime_pose_frame_contract().canonical_desk_pose_vector_to_storage(
        desk_pose, side="left"
    )
    resolved = replay_action_poses_in_hardware_base_at_index(group, ("left",), index=0)
    np.testing.assert_allclose(resolved["left"], expected, atol=1e-6)


def test_desk_absolute_does_not_apply_nonidentity_raw_transform_again():
    raw_to_desk = np.eye(4, dtype=float)
    raw_to_desk[:3, 3] = [9.0, 8.0, 7.0]
    contract = _desk_contract(storage_frame="desk_absolute", raw_to_desk=raw_to_desk)
    desk_pose = np.array([0.2, -0.1, 0.4, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    canonical = contract.storage_pose_vector_to_canonical_desk(desk_pose, side="right")
    np.testing.assert_allclose(canonical, desk_pose, atol=1e-6)


def test_partial_canonical_frame_metadata_fails_closed():
    actions = np.zeros((1, ACTION_DIM), dtype=np.float32)
    group = _group(actions, None)
    group.metadata["action_metadata"] = {
        "pose_action_absolute_frame": "canonical_desk"
    }
    try:
        replay_action_pose_frame(group)
    except RuntimeError as exc:
        assert "missing its serialized pose_frame_contract" in str(exc)
    else:
        raise AssertionError("partial canonical pose metadata must fail closed")
