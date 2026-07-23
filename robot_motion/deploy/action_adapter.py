"""Pure decoded-absolute NormalizedAction to pre-safety RobotAction adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import (
    RobotAction,
    SourceValidationResult,
    VectorActionLayout,
    robot_action_from_vector_action,
)
from robot_motion.action.proposal import freeze_robot_action, robot_action_semantic_payload

from .actions import NormalizedAction
from .observation import DeploymentObservation
from .schema import (
    ACTION_ARM_POSE_SLICES,
    ACTION_DIM,
    ACTION_HAND_SLICES,
    ALL_SIDES,
    ARM_JOINT_DOF,
    ARM_POSE_DOF,
    HAND_DOF,
)


DEPLOY_ACTION_ADAPTER_CONTRACT_ID_V1 = "deploy_normalized_absolute_to_robot_action_v1"
DEPLOY_ACTION_ADAPTER_HASH_DOMAIN_V1 = "dexteleop.deploy_action_adapter.v1"
DEPLOY_REPRESENTATION_RECEIPT_HASH_DOMAIN_V1 = "dexteleop.deploy_representation_receipt.v1"

DEPLOY_VECTOR_ACTION_LAYOUT = VectorActionLayout(
    action_dim=ACTION_DIM,
    arm_pose_slices=ACTION_ARM_POSE_SLICES,
    hand_slices=ACTION_HAND_SLICES,
    arm_pose_dof=ARM_POSE_DOF,
    arm_joint_dof=ARM_JOINT_DOF,
    hand_dof=HAND_DOF,
    side_order=ALL_SIDES,
)

_ROBOT_ACTION_SEMANTIC_METADATA_KEYS = (
    "arm_action_specs_by_side",
    "hand_action_specs_by_side",
)


def _layout_payload(layout: VectorActionLayout) -> dict[str, Any]:
    def slices(values: Mapping[str, slice]) -> dict[str, tuple[int | None, int | None, int | None]]:
        return {
            str(side): (value.start, value.stop, value.step)
            for side, value in values.items()
        }

    return {
        "action_dim": layout.action_dim,
        "arm_pose_slices": slices(layout.arm_pose_slices),
        "hand_slices": slices(layout.hand_slices),
        "arm_pose_dof": layout.arm_pose_dof,
        "arm_joint_dof": layout.arm_joint_dof,
        "hand_dof": layout.hand_dof,
        "side_order": layout.side_order,
    }


@dataclass(frozen=True)
class DeployActionRepresentationReceipt:
    representation_trace_id: str
    adapter_contract_id: str
    adapter_contract_hash: str
    input_stage: str
    output_stage: str
    source_pose_mode: str
    source_absolute_frame: str | None
    output_absolute_frame: str | None
    anchor_policy: str | None
    anchor_sides: tuple[str, ...]
    active_sides: tuple[str, ...]
    active_effectors: tuple[str, ...]
    action_space: str | None
    lossiness: str
    source_validation_accepted: bool
    source_validation_reasons: tuple[str, ...]
    semantic_content_hash: str


@dataclass(frozen=True)
class DeployRobotActionAdapterResult:
    robot_action: RobotAction | None
    compatibility_robot_action: RobotAction | None
    source_validation: SourceValidationResult
    representation_receipt: DeployActionRepresentationReceipt
    faults: tuple[str, ...]
    vector_build_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "faults", tuple(str(value) for value in self.faults))
        object.__setattr__(self, "vector_build_metadata", dict(self.vector_build_metadata))


class DeployRobotActionAdapter:
    """Lift an already-decoded absolute deploy vector without applying safety."""

    def __init__(
        self,
        *,
        active_sides: Sequence[str],
        arms_enabled: bool,
        hands_enabled: bool,
        layout: VectorActionLayout = DEPLOY_VECTOR_ACTION_LAYOUT,
    ) -> None:
        self.active_sides = tuple(str(side).strip().lower() for side in active_sides)
        if not self.active_sides or any(side not in ALL_SIDES for side in self.active_sides):
            raise ValueError("active_sides must contain left and/or right.")
        if len(set(self.active_sides)) != len(self.active_sides):
            raise ValueError("active_sides must not contain duplicates.")
        self.arms_enabled = bool(arms_enabled)
        self.hands_enabled = bool(hands_enabled)
        self.layout = layout
        self.layout_hash = canonical_content_hash(
            _layout_payload(layout),
            domain="dexteleop.deploy_vector_action_layout.v1",
        )
        self.contract_hash = canonical_content_hash(
            {
                "adapter_contract_id": DEPLOY_ACTION_ADAPTER_CONTRACT_ID_V1,
                "layout_hash": self.layout_hash,
                "active_sides": self.active_sides,
                "arms_enabled": self.arms_enabled,
                "hands_enabled": self.hands_enabled,
            },
            domain=DEPLOY_ACTION_ADAPTER_HASH_DOMAIN_V1,
        )

    def adapt(
        self,
        action: NormalizedAction | Any,
        observation: DeploymentObservation,
        *,
        policy_name: str = "",
        input_stage: str = "decoded_absolute_pre_safety",
    ) -> DeployRobotActionAdapterResult:
        if not isinstance(observation, DeploymentObservation):
            raise TypeError("observation must be a DeploymentObservation.")
        normalized = (
            action
            if isinstance(action, NormalizedAction)
            else NormalizedAction(vector=np.asarray(action, dtype=np.float32).reshape(-1))
        )
        metadata = dict(normalized.metadata)
        action_source = str(metadata.get("source") or "").strip()
        if not action_source:
            action_source = (
                f"deploy_policy:{policy_name}" if policy_name else "deploy_policy"
            )

        faults: tuple[str, ...]
        build_metadata: Mapping[str, Any]
        robot_action: RobotAction | None
        compatibility_robot_action: RobotAction | None
        try:
            build = robot_action_from_vector_action(
                vector=normalized.vector,
                arm_joint_targets=normalized.arm_joint_targets,
                metadata=metadata,
                active_sides=self.active_sides,
                arms_enabled=self.arms_enabled,
                hands_enabled=self.hands_enabled,
                layout=self.layout,
                timestamp=observation.timestamp,
                source=action_source,
            )
            faults = tuple(build.faults)
            build_metadata = dict(build.metadata)
            robot_action = (
                None
                if build.robot_action is None
                else self._semantic_robot_action(build.robot_action)
            )
            compatibility_robot_action = build.robot_action
        except (TypeError, ValueError) as exc:
            faults = (f"deploy_action_adapter_invalid:{type(exc).__name__}:{exc}",)
            build_metadata = {"adapter_exception": type(exc).__name__}
            robot_action = None
            compatibility_robot_action = None

        accepted = robot_action is not None and not faults
        validation = SourceValidationResult(accepted=accepted, reasons=faults)
        receipt = self._representation_receipt(
            metadata=metadata,
            input_stage=str(input_stage),
            robot_action=robot_action,
            validation=validation,
        )
        return DeployRobotActionAdapterResult(
            robot_action=robot_action,
            compatibility_robot_action=compatibility_robot_action,
            source_validation=validation,
            representation_receipt=receipt,
            faults=faults,
            vector_build_metadata=build_metadata,
        )

    @staticmethod
    def _semantic_robot_action(action: RobotAction) -> RobotAction:
        semantic_metadata = {
            key: action.metadata[key]
            for key in _ROBOT_ACTION_SEMANTIC_METADATA_KEYS
            if key in action.metadata
        }
        normalized = RobotAction(
            action_space=action.action_space,
            timestamp=action.timestamp,
            arm_pose_by_side=action.arm_pose_by_side,
            arm_qpos_by_side=action.arm_qpos_by_side,
            hand_qpos_by_side=action.hand_qpos_by_side,
            end_effector_targets_by_side=action.end_effector_targets_by_side,
            active_effectors=action.active_effectors,
            frame_by_effector=action.frame_by_effector,
            controlled_frame_by_effector=action.controlled_frame_by_effector,
            source=action.source,
            metadata=semantic_metadata,
        )
        return freeze_robot_action(normalized)

    def _representation_receipt(
        self,
        *,
        metadata: Mapping[str, Any],
        input_stage: str,
        robot_action: RobotAction | None,
        validation: SourceValidationResult,
    ) -> DeployActionRepresentationReceipt:
        source_pose_mode = str(metadata.get("pose_action_source_mode") or "absolute")
        anchor_sides = tuple(str(value) for value in metadata.get("pose_anchor_sides", ()))
        output_stage = (
            "pre_safety_canonical_robot_action"
            if input_stage == "decoded_absolute_pre_safety"
            else "canonical_robot_action"
        )
        payload = {
            "adapter_contract_id": DEPLOY_ACTION_ADAPTER_CONTRACT_ID_V1,
            "adapter_contract_hash": self.contract_hash,
            "input_stage": input_stage,
            "output_stage": output_stage,
            "source_pose_mode": source_pose_mode,
            "source_absolute_frame": metadata.get("pose_action_source_absolute_frame"),
            "output_absolute_frame": metadata.get("pose_action_output_absolute_frame"),
            "anchor_policy": metadata.get("pose_anchor_policy"),
            "anchor_sides": anchor_sides,
            "active_sides": self.active_sides,
            "active_effectors": () if robot_action is None else robot_action.active_effectors,
            "action_space": None if robot_action is None else robot_action.action_space,
            "lossiness": "lossless" if validation.accepted else "unsupported",
            "source_validation_accepted": validation.accepted,
            "source_validation_reasons": validation.reasons,
            "robot_action_payload": (
                None
                if robot_action is None
                else robot_action_semantic_payload(robot_action)
            ),
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=DEPLOY_REPRESENTATION_RECEIPT_HASH_DOMAIN_V1,
        )
        return DeployActionRepresentationReceipt(
            representation_trace_id=f"deploy-representation:{semantic_hash}",
            adapter_contract_id=DEPLOY_ACTION_ADAPTER_CONTRACT_ID_V1,
            adapter_contract_hash=self.contract_hash,
            input_stage=input_stage,
            output_stage=output_stage,
            source_pose_mode=source_pose_mode,
            source_absolute_frame=metadata.get("pose_action_source_absolute_frame"),
            output_absolute_frame=metadata.get("pose_action_output_absolute_frame"),
            anchor_policy=metadata.get("pose_anchor_policy"),
            anchor_sides=anchor_sides,
            active_sides=self.active_sides,
            active_effectors=() if robot_action is None else robot_action.active_effectors,
            action_space=None if robot_action is None else robot_action.action_space,
            lossiness="lossless" if validation.accepted else "unsupported",
            source_validation_accepted=validation.accepted,
            source_validation_reasons=validation.reasons,
            semantic_content_hash=semantic_hash,
        )


__all__ = [
    "DEPLOY_ACTION_ADAPTER_CONTRACT_ID_V1",
    "DEPLOY_VECTOR_ACTION_LAYOUT",
    "DeployActionRepresentationReceipt",
    "DeployRobotActionAdapter",
    "DeployRobotActionAdapterResult",
]
