"""Typed pose-frame transforms below shared canonical safety.

The first contract converts a canonical-desk arm pose into the calibrated
runtime arm-base frame.  It deliberately returns a separate immutable action
and receipt; proposal, selected, and safe facts remain in their original
canonical frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.pose_frames import (
    CANONICAL_DESK_FRAME,
    POSE_STORAGE_FRAME_DESK,
    POSE_STORAGE_FRAME_RAW,
    PoseFrameContract,
    PoseFrameContractError,
)

from .proposal import freeze_robot_action, robot_action_semantic_payload
from .safety_anchor import safety_anchor_action_hash
from .schema import (
    BIMANUAL_EE_POSE_HAND16_V1,
    BIMANUAL_EE_POSE_V1,
    RobotAction,
)


CANONICAL_DESK_POSE_TRANSFORM_RECEIPT_HASH_DOMAIN_V1 = (
    "dexteleop.canonical_desk_pose_transform_receipt.v1"
)
CANONICAL_DESK_POSE_TRANSFORM_RESULT_HASH_DOMAIN_V1 = (
    "dexteleop.canonical_desk_pose_transform_result.v1"
)
CANONICAL_DESK_POSE_TRANSFORM_ALGORITHM_V1 = (
    "pose_frame_contract.canonical_desk_pose_vector_to_storage.v1"
)


@dataclass(frozen=True)
class PoseFrameTransformBindingReceipt:
    receipt_id: str
    source_pose_frame_contract_hash: str
    target_pose_frame_contract_hash: str
    transform_algorithm_id: str
    source_reference_frame: str
    target_reference_frame_by_side: tuple[tuple[str, str], ...]
    controlled_frame: str
    semantic_content_hash: str


@dataclass(frozen=True)
class PoseFrameTransformResult:
    result_id: str
    input_safe_action: RobotAction
    input_safe_action_hash: str
    transformed_action: RobotAction
    transformed_action_hash: str
    binding_receipt_id: str
    binding_receipt_hash: str
    semantic_content_hash: str

    def __post_init__(self) -> None:
        source = freeze_robot_action(self.input_safe_action)
        transformed = freeze_robot_action(self.transformed_action)
        if safety_anchor_action_hash(source) != self.input_safe_action_hash:
            raise ValueError("Pose transform input safe-action hash mismatch.")
        if safety_anchor_action_hash(transformed) != self.transformed_action_hash:
            raise ValueError("Pose transform output action hash mismatch.")
        object.__setattr__(self, "input_safe_action", source)
        object.__setattr__(self, "transformed_action", transformed)


class CanonicalDeskToRuntimePoseTransformer:
    """Convert one shared-safe desk pose to an explicit runtime raw frame."""

    def __init__(
        self,
        *,
        source_pose_frame_contract: PoseFrameContract,
        target_pose_frame_contract: PoseFrameContract,
        controlled_frame: str = "tool_tcp",
    ) -> None:
        if not isinstance(source_pose_frame_contract, PoseFrameContract):
            raise TypeError("source_pose_frame_contract must be PoseFrameContract.")
        if not isinstance(target_pose_frame_contract, PoseFrameContract):
            raise TypeError("target_pose_frame_contract must be PoseFrameContract.")
        if source_pose_frame_contract.pose_storage_frame != POSE_STORAGE_FRAME_DESK:
            raise ValueError("Source pose transform contract must store canonical desk poses.")
        if target_pose_frame_contract.pose_storage_frame != POSE_STORAGE_FRAME_RAW:
            raise ValueError("Target pose transform contract must describe runtime raw poses.")
        if source_pose_frame_contract.canonical_absolute_frame != CANONICAL_DESK_FRAME:
            raise ValueError("Source pose contract canonical frame must be desk.")
        if target_pose_frame_contract.canonical_absolute_frame != CANONICAL_DESK_FRAME:
            raise ValueError("Target pose contract canonical frame must be desk.")
        sides = ("left", "right")
        if any(side not in source_pose_frame_contract.source.active_sides for side in sides):
            raise ValueError("Source pose contract must cover both arms.")
        if any(side not in target_pose_frame_contract.source.active_sides for side in sides):
            raise ValueError("Target pose contract must cover both arms.")
        controlled = str(controlled_frame).strip()
        if not controlled:
            raise ValueError("controlled_frame must be non-empty.")
        payload = {
            "source_pose_frame_contract_hash": source_pose_frame_contract.contract_hash,
            "target_pose_frame_contract_hash": target_pose_frame_contract.contract_hash,
            "transform_algorithm_id": CANONICAL_DESK_POSE_TRANSFORM_ALGORITHM_V1,
            "source_reference_frame": "canonical_desk",
            "target_reference_frame_by_side": (
                ("left", "left_arm_base"),
                ("right", "right_arm_base"),
            ),
            "controlled_frame": controlled,
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=CANONICAL_DESK_POSE_TRANSFORM_RECEIPT_HASH_DOMAIN_V1,
        )
        self.source_pose_frame_contract = source_pose_frame_contract
        self.target_pose_frame_contract = target_pose_frame_contract
        self.binding_receipt = PoseFrameTransformBindingReceipt(
            receipt_id=f"canonical-desk-pose-transform:{semantic_hash}",
            **payload,
            semantic_content_hash=semantic_hash,
        )

    def transform(self, action: RobotAction) -> PoseFrameTransformResult:
        source = freeze_robot_action(action)
        if source.action_space not in {
            BIMANUAL_EE_POSE_V1,
            BIMANUAL_EE_POSE_HAND16_V1,
        }:
            raise ValueError("Canonical desk transform requires a pose action space.")
        expected_effectors = {"left_arm", "right_arm"} | {
            f"{side}_hand" for side in source.hand_qpos_by_side
        }
        if set(source.active_effectors) != expected_effectors:
            raise ValueError(
                "Canonical desk transform requires coupled arms plus only the "
                "explicit hand-qpos components."
            )
        if source.end_effector_targets_by_side:
            raise ValueError(
                "Canonical desk transform does not accept native end-effector targets."
            )
        hand_specs_metadata: dict[str, object] = {}
        if source.hand_qpos_by_side:
            raw_hand_specs = source.metadata.get("hand_action_specs_by_side")
            if not isinstance(raw_hand_specs, Mapping):
                raise ValueError(
                    "Canonical desk hand-qpos passthrough requires explicit "
                    "hand_action_specs_by_side metadata."
                )
            hand_specs_metadata = dict(raw_hand_specs)
        transformed_pose: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            if source.arm_frame(side) != "canonical_desk":
                raise ValueError(
                    f"{side} pose must explicitly use canonical_desk, got "
                    f"{source.arm_frame(side)!r}."
                )
            if source.arm_controlled_frame(side) != self.binding_receipt.controlled_frame:
                raise ValueError("Canonical desk transform controlled-frame mismatch.")
            try:
                transformed_pose[side] = np.asarray(
                    self.target_pose_frame_contract.canonical_desk_pose_vector_to_storage(
                        source.arm_pose_by_side[side],
                        side=side,
                    ),
                    dtype=np.float32,
                )
            except (PoseFrameContractError, ValueError) as exc:
                raise ValueError(
                    f"{side} canonical desk pose cannot be transformed: {exc}"
                ) from exc
        transformed = freeze_robot_action(
            RobotAction(
                action_space=source.action_space,
                timestamp=source.timestamp,
                arm_pose_by_side=transformed_pose,
                hand_qpos_by_side=source.hand_qpos_by_side,
                active_effectors=source.active_effectors,
                frame_by_effector={
                    "left_arm": "left_arm_base",
                    "right_arm": "right_arm_base",
                },
                controlled_frame_by_effector={
                    "left_arm": self.binding_receipt.controlled_frame,
                    "right_arm": self.binding_receipt.controlled_frame,
                },
                source="canonical_desk_execution_transform",
                metadata=(
                    {}
                    if not source.hand_qpos_by_side
                    else {"hand_action_specs_by_side": hand_specs_metadata}
                ),
            )
        )
        source_hash = safety_anchor_action_hash(source)
        transformed_hash = safety_anchor_action_hash(transformed)
        payload = {
            "input_safe_action": robot_action_semantic_payload(source),
            "input_safe_action_hash": source_hash,
            "transformed_action": robot_action_semantic_payload(transformed),
            "transformed_action_hash": transformed_hash,
            "binding_receipt_id": self.binding_receipt.receipt_id,
            "binding_receipt_hash": self.binding_receipt.semantic_content_hash,
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=CANONICAL_DESK_POSE_TRANSFORM_RESULT_HASH_DOMAIN_V1,
        )
        return PoseFrameTransformResult(
            result_id=f"canonical-desk-pose-transform-result:{semantic_hash}",
            **{
                key: value
                for key, value in payload.items()
                if key not in {"input_safe_action", "transformed_action"}
            },
            input_safe_action=source,
            transformed_action=transformed,
            semantic_content_hash=semantic_hash,
        )


__all__ = [
    "CANONICAL_DESK_POSE_TRANSFORM_ALGORITHM_V1",
    "CANONICAL_DESK_POSE_TRANSFORM_RECEIPT_HASH_DOMAIN_V1",
    "CANONICAL_DESK_POSE_TRANSFORM_RESULT_HASH_DOMAIN_V1",
    "CanonicalDeskToRuntimePoseTransformer",
    "PoseFrameTransformBindingReceipt",
    "PoseFrameTransformResult",
]
