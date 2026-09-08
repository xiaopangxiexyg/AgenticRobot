"""Shared collection-storage to RealTeleop runtime pose-frame projection."""

from __future__ import annotations

from dataset_contracts.pose_frames import PoseFrameContract
from robot_motion.action import RobotAction
from robot_motion.action.executor import RobotActionResolver
from robot_motion.pose_frame_profiles import (
    default_rm65_workspace_near_boundary_runtime_pose_frame_contract,
)


def resolve_runtime_pose_frame_contract(
    collection_pose_frame_contract: PoseFrameContract,
) -> PoseFrameContract:
    """Return the raw arm-base contract for measured feedback and execution.

    The formal recorder may store canonical desk poses.  RealTeleop feedback
    and arm commands remain in each RM65 base frame.  This helper only selects
    the matching contract; :class:`PoseFrameContract` remains the sole numeric
    transform owner.
    """

    if not isinstance(collection_pose_frame_contract, PoseFrameContract):
        raise TypeError("collection_pose_frame_contract must be a PoseFrameContract.")
    if collection_pose_frame_contract.is_raw_storage:
        return collection_pose_frame_contract

    default_runtime = (
        default_rm65_workspace_near_boundary_runtime_pose_frame_contract()
    )
    if (
        collection_pose_frame_contract.source.config_hash
        == default_runtime.source.config_hash
    ):
        return default_runtime
    return PoseFrameContract.v23_forced_raw(
        collection_pose_frame_contract.source
    )


class RealTeleopPoseFrameActionResolver:
    """Project explicit desk actions, then reuse the common command resolver."""

    def __init__(
        self,
        *,
        collection_contract: PoseFrameContract,
        runtime_contract: PoseFrameContract,
    ) -> None:
        self.collection_contract = collection_contract
        self.runtime_contract = runtime_contract
        self.delegate = RobotActionResolver()

    def resolve(self, action: RobotAction, **_: object):
        projected = self.project_action(action)
        command = self.delegate.resolve(projected)
        command.metadata.update(
            {
                "rule_based_source_action": action.source,
                "rule_based_collection_pose_frame_contract_hash": (
                    self.collection_contract.contract_hash
                ),
                "rule_based_runtime_pose_frame_contract_hash": (
                    self.runtime_contract.contract_hash
                ),
            }
        )
        return command

    def project_action(self, action: RobotAction) -> RobotAction:
        """Project an action into runtime arm-base frames without resolving it."""

        if not action.arm_pose_by_side:
            return action
        frames = {side: action.arm_frame(side) for side in action.arm_pose_by_side}
        if all(frame == "canonical_desk" for frame in frames.values()):
            poses = {
                side: self.runtime_contract.canonical_desk_pose_vector_to_storage(
                    pose,
                    side=side,
                )
                for side, pose in action.arm_pose_by_side.items()
            }
            frame_by_effector = dict(action.frame_by_effector)
            for side in poses:
                frame_by_effector[f"{side}_arm"] = f"{side}_arm_base"
            return RobotAction(
                action_space=action.action_space,
                timestamp=action.timestamp,
                arm_pose_by_side=poses,
                arm_qpos_by_side=action.arm_qpos_by_side,
                hand_qpos_by_side=action.hand_qpos_by_side,
                end_effector_targets_by_side=action.end_effector_targets_by_side,
                active_effectors=action.active_effectors,
                frame_by_effector=frame_by_effector,
                controlled_frame_by_effector=action.controlled_frame_by_effector,
                source="rule_based_canonical_desk_execution_transform",
                metadata={
                    **dict(action.metadata),
                    "pose_transform_owner": "PoseFrameContract",
                    "pose_transform_source_action": action.source,
                    "pose_transform_target_contract_hash": (
                        self.runtime_contract.contract_hash
                    ),
                },
            )
        expected = {side: f"{side}_arm_base" for side in frames}
        if frames != expected:
            raise ValueError(
                "rule-based pose action frames must be uniformly canonical_desk "
                f"or explicit per-side arm base frames; got {frames}."
            )
        return action

    def _project_pose(self, action: RobotAction) -> RobotAction:
        """Compatibility alias for callers predating :meth:`project_action`."""

        return self.project_action(action)


__all__ = [
    "RealTeleopPoseFrameActionResolver",
    "resolve_runtime_pose_frame_contract",
]
