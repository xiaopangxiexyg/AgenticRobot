"""Checked-in pose-frame profiles for supported robot stations."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from dataset_contracts.pose_frames import PoseFrameContract, load_pose_frame_contract


RM65_DUAL_ARM_WORKSPACE_NEAR_BOUNDARY_SOURCE_ID = (
    "rm65_real_dual_arm_workspace_near_boundary_v1"
)
_POSE_FRAME_CONFIG_DIR = Path(__file__).resolve().parent / "configs" / "pose_frames"
DEFAULT_RM65_DUAL_ARM_WORKSPACE_NEAR_BOUNDARY_COLLECTION_POSE_FRAME_CONTRACT_PATH = (
    _POSE_FRAME_CONFIG_DIR
    / "rm65_dual_arm_workspace_near_boundary_v1_collection_desk_absolute.json"
)
DEFAULT_RM65_DUAL_ARM_WORKSPACE_NEAR_BOUNDARY_RUNTIME_POSE_FRAME_CONTRACT_PATH = (
    _POSE_FRAME_CONFIG_DIR
    / "rm65_dual_arm_workspace_near_boundary_v1_runtime_raw_absolute.json"
)


def default_rm65_workspace_near_boundary_collection_pose_frame_contract(
    active_sides: Sequence[str],
) -> PoseFrameContract | None:
    """Return the canonical desk-storage contract for the calibrated dual-arm station.

    The checked-in station profile is deliberately not applied to a single-arm
    topology: a separately installed RM65 may have different raw-base extrinsics.
    """

    sides = tuple(str(side).strip().lower() for side in active_sides)
    if sides != ("left", "right"):
        return None
    return load_pose_frame_contract(
        DEFAULT_RM65_DUAL_ARM_WORKSPACE_NEAR_BOUNDARY_COLLECTION_POSE_FRAME_CONTRACT_PATH
    )


def default_rm65_workspace_near_boundary_runtime_pose_frame_contract() -> PoseFrameContract:
    """Load the dual-RM65 runtime raw-base to canonical-desk binding.

    The origin is left-right centered on the tabletop surface and lies at the
    operator/robot-near boundary of the valid tabletop workspace, measured
    along the tabletop-center-to-operator/robot direction.  The station URDF
    defines desk +X from the operator toward the tabletop, +Y toward the
    operator's left, and +Z upward.  Its numeric pose remains
    ``(-0.472, 0.0, 0.01)`` with a 180-degree Z rotation in station world.
    """

    return load_pose_frame_contract(
        DEFAULT_RM65_DUAL_ARM_WORKSPACE_NEAR_BOUNDARY_RUNTIME_POSE_FRAME_CONTRACT_PATH
    )


# Compatibility aliases for callers written before the frame received its
# workspace-boundary name.  They resolve the same current contract and do not
# define a second coordinate frame.
RM65_DUAL_ARM_TABLE_CENTER_SOURCE_ID = RM65_DUAL_ARM_WORKSPACE_NEAR_BOUNDARY_SOURCE_ID
DEFAULT_RM65_DUAL_ARM_COLLECTION_POSE_FRAME_CONTRACT_PATH = (
    DEFAULT_RM65_DUAL_ARM_WORKSPACE_NEAR_BOUNDARY_COLLECTION_POSE_FRAME_CONTRACT_PATH
)
DEFAULT_RM65_DUAL_ARM_RUNTIME_POSE_FRAME_CONTRACT_PATH = (
    DEFAULT_RM65_DUAL_ARM_WORKSPACE_NEAR_BOUNDARY_RUNTIME_POSE_FRAME_CONTRACT_PATH
)
default_rm65_collection_pose_frame_contract = (
    default_rm65_workspace_near_boundary_collection_pose_frame_contract
)
default_rm65_runtime_pose_frame_contract = (
    default_rm65_workspace_near_boundary_runtime_pose_frame_contract
)


__all__ = [
    "DEFAULT_RM65_DUAL_ARM_COLLECTION_POSE_FRAME_CONTRACT_PATH",
    "DEFAULT_RM65_DUAL_ARM_RUNTIME_POSE_FRAME_CONTRACT_PATH",
    "DEFAULT_RM65_DUAL_ARM_WORKSPACE_NEAR_BOUNDARY_COLLECTION_POSE_FRAME_CONTRACT_PATH",
    "DEFAULT_RM65_DUAL_ARM_WORKSPACE_NEAR_BOUNDARY_RUNTIME_POSE_FRAME_CONTRACT_PATH",
    "RM65_DUAL_ARM_TABLE_CENTER_SOURCE_ID",
    "RM65_DUAL_ARM_WORKSPACE_NEAR_BOUNDARY_SOURCE_ID",
    "default_rm65_collection_pose_frame_contract",
    "default_rm65_runtime_pose_frame_contract",
    "default_rm65_workspace_near_boundary_collection_pose_frame_contract",
    "default_rm65_workspace_near_boundary_runtime_pose_frame_contract",
]
