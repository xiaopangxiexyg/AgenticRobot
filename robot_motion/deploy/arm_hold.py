"""Per-side action-hold semantic resolution before canonical safety."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.policy_action_semantics import (
    PolicyArmHoldOutputContract,
)

from .actions import NormalizedAction
from .observation import DeploymentObservation
from .schema import (
    ACTION_ARM_POSE_SLICES,
    ACTION_HAND_SLICES,
    ALL_SIDES,
    ARM_JOINT_DOF,
    HAND_DOF,
    STATE_ARM_QPOS_SLICES,
    vector_or_none,
)


ARM_HOLD_RESOLUTION_RECEIPT_SCHEMA_V1 = (
    "dexteleop_arm_hold_resolution_receipt_v1"
)
ARM_HOLD_RESOLUTION_HASH_DOMAIN_V1 = (
    "dexteleop.arm_hold_resolution.semantic.v1"
)
ARM_HOLD_STATE_MOVING = "moving"
ARM_HOLD_STATE_HOLD = "hold"


@dataclass(frozen=True)
class ArmHoldSideResolution:
    side: str
    probability: float
    previous_state: str
    resolved_state: str
    transition: str
    consecutive_count: int
    reference_operation: str
    modified_components: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "probability": self.probability,
            "previous_state": self.previous_state,
            "resolved_state": self.resolved_state,
            "transition": self.transition,
            "consecutive_count": self.consecutive_count,
            "reference_operation": self.reference_operation,
            "modified_components": list(self.modified_components),
        }


@dataclass(frozen=True)
class ArmHoldResolutionReceipt:
    contract_hash: str
    source_combination: str
    resolved_combination: str
    side_resolutions: tuple[ArmHoldSideResolution, ...]
    reference_observation_step_index: int
    schema: str = ARM_HOLD_RESOLUTION_RECEIPT_SCHEMA_V1

    def to_dict(self, *, include_hash: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "contract_hash": self.contract_hash,
            "source_combination": self.source_combination,
            "resolved_combination": self.resolved_combination,
            "side_resolutions": [
                resolution.to_dict() for resolution in self.side_resolutions
            ],
            "reference_observation_step_index": (
                self.reference_observation_step_index
            ),
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_dict(),
            domain=ARM_HOLD_RESOLUTION_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class ArmHoldStagedResolution:
    """One validated transition that is inert until explicitly committed."""

    token: str
    state_generation: int
    resolved_action: NormalizedAction
    receipt: ArmHoldResolutionReceipt


@dataclass
class _SideState:
    state: str = ARM_HOLD_STATE_MOVING
    enter_count: int = 0
    exit_count: int = 0
    hold_vector: np.ndarray | None = None
    hold_qpos: np.ndarray | None = None


class ArmHoldSemanticResolver:
    """Resolve independent left/right hold intent at actual emission time."""

    def __init__(
        self,
        contract: PolicyArmHoldOutputContract,
        *,
        active_sides: tuple[str, ...],
    ) -> None:
        if not isinstance(contract, PolicyArmHoldOutputContract):
            raise TypeError(
                "contract must be a PolicyArmHoldOutputContract."
            )
        active = tuple(str(side).strip().lower() for side in active_sides)
        invalid = set(active) - set(ALL_SIDES)
        if not active or invalid or len(set(active)) != len(active):
            raise ValueError("active_sides must be a unique non-empty side subset.")
        self.contract = contract
        self.active_sides = active
        self._states = {side: _SideState() for side in ALL_SIDES}
        self._state_generation = 0
        self._next_stage_sequence = 0
        self._pending: tuple[
            ArmHoldStagedResolution,
            dict[str, _SideState],
        ] | None = None

    def reset(self) -> None:
        self._states = {side: _SideState() for side in ALL_SIDES}
        self._state_generation += 1
        self._pending = None

    @property
    def resolved_combination(self) -> str:
        return "".join(
            "1" if self._states[side].state == ARM_HOLD_STATE_HOLD else "0"
            for side in self.contract.side_order
        )

    def _reference_for_side(
        self,
        side: str,
        action: NormalizedAction,
        observation: DeploymentObservation,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        hold_vector = np.asarray(
            observation.hold_action(),
            dtype=np.float32,
        ).copy()
        arm_pose = vector_or_none(
            hold_vector[ACTION_ARM_POSE_SLICES[side]],
            length=7,
        )
        if side not in action.arm_joint_targets and arm_pose is None:
            raise ValueError(f"{side} arm hold reference is unavailable.")
        if side in set(observation.end_effector_sides or ()):
            hand = vector_or_none(
                hold_vector[ACTION_HAND_SLICES[side]],
                length=HAND_DOF,
            )
            if hand is None:
                raise ValueError(
                    f"{side} end-effector hold reference is unavailable."
                )
        hold_qpos = None
        if side in action.arm_joint_targets:
            hold_qpos = vector_or_none(
                observation.state[STATE_ARM_QPOS_SLICES[side]],
                length=ARM_JOINT_DOF,
            )
            if hold_qpos is None:
                raise ValueError(
                    f"{side} arm joint hold reference is unavailable."
                )
            hold_qpos = np.asarray(hold_qpos, dtype=np.float32).copy()
        return hold_vector, hold_qpos

    def stage(
        self,
        action: NormalizedAction,
        observation: DeploymentObservation,
    ) -> ArmHoldStagedResolution:
        if self._pending is not None:
            raise RuntimeError(
                "An arm-hold transition is already staged."
            )
        prediction = action.arm_hold_prediction
        if prediction is None:
            raise ValueError("Declared arm-hold deployment requires a prediction.")
        if prediction.contract_hash != self.contract.contract_hash:
            raise ValueError("Arm-hold keyframe contract hash mismatch.")

        probabilities = {
            side: float(prediction.probabilities[index])
            for index, side in enumerate(self.contract.side_order)
        }
        source_combination = "".join(
            "1"
            if probabilities[side] >= self.contract.enter_threshold
            else "0"
            for side in self.contract.side_order
        )

        candidates: dict[str, _SideState] = {}
        resolutions: list[ArmHoldSideResolution] = []
        for side in self.contract.side_order:
            previous = self._states[side]
            probability = probabilities[side]
            if side not in self.active_sides:
                candidate = _SideState()
                transition = "inactive"
                reference_operation = "none"
                decision_count = 0
            elif previous.state == ARM_HOLD_STATE_MOVING:
                enter_count = (
                    previous.enter_count + 1
                    if probability >= self.contract.enter_threshold
                    else 0
                )
                decision_count = enter_count
                if enter_count >= self.contract.enter_consecutive_steps:
                    hold_vector, hold_qpos = self._reference_for_side(
                        side,
                        action,
                        observation,
                    )
                    candidate = _SideState(
                        state=ARM_HOLD_STATE_HOLD,
                        hold_vector=hold_vector,
                        hold_qpos=hold_qpos,
                    )
                    transition = "enter_hold"
                    reference_operation = "latched"
                else:
                    candidate = _SideState(enter_count=enter_count)
                    transition = "stay_moving"
                    reference_operation = "none"
            else:
                exit_count = (
                    previous.exit_count + 1
                    if probability <= self.contract.exit_threshold
                    else 0
                )
                decision_count = exit_count
                if exit_count >= self.contract.exit_consecutive_steps:
                    candidate = _SideState()
                    transition = "exit_hold"
                    reference_operation = "released"
                else:
                    candidate = _SideState(
                        state=ARM_HOLD_STATE_HOLD,
                        exit_count=exit_count,
                        hold_vector=previous.hold_vector,
                        hold_qpos=previous.hold_qpos,
                    )
                    transition = "stay_hold"
                    reference_operation = "reused"
            candidates[side] = candidate
            modified = (
                ("arm", "end_effector")
                if (
                    candidate.state == ARM_HOLD_STATE_HOLD
                    and side in set(observation.end_effector_sides or ())
                )
                else (
                    ("arm",)
                    if candidate.state == ARM_HOLD_STATE_HOLD
                    else ()
                )
            )
            resolutions.append(
                ArmHoldSideResolution(
                    side=side,
                    probability=probability,
                    previous_state=previous.state,
                    resolved_state=candidate.state,
                    transition=transition,
                    consecutive_count=decision_count,
                    reference_operation=reference_operation,
                    modified_components=modified,
                )
            )

        vector = np.asarray(action.vector, dtype=np.float32).copy()
        joint_targets = {
            str(side): np.asarray(value, dtype=np.float32).copy()
            for side, value in action.arm_joint_targets.items()
        }
        for side in self.active_sides:
            candidate = candidates[side]
            if candidate.state != ARM_HOLD_STATE_HOLD:
                continue
            assert candidate.hold_vector is not None
            vector[ACTION_ARM_POSE_SLICES[side]] = candidate.hold_vector[
                ACTION_ARM_POSE_SLICES[side]
            ]
            if side in set(observation.end_effector_sides or ()):
                vector[ACTION_HAND_SLICES[side]] = candidate.hold_vector[
                    ACTION_HAND_SLICES[side]
                ]
            if side in joint_targets:
                assert candidate.hold_qpos is not None
                joint_targets[side] = candidate.hold_qpos.copy()

        resolved_combination = "".join(
            "1"
            if candidates[side].state == ARM_HOLD_STATE_HOLD
            else "0"
            for side in self.contract.side_order
        )
        receipt = ArmHoldResolutionReceipt(
            contract_hash=self.contract.contract_hash,
            source_combination=source_combination,
            resolved_combination=resolved_combination,
            side_resolutions=tuple(resolutions),
            reference_observation_step_index=int(observation.step_index),
        )
        resolved = NormalizedAction(
            vector=vector,
            arm_joint_targets=joint_targets,
            metadata=dict(action.metadata),
            arm_hold_prediction=prediction,
        )
        stage_sequence = self._next_stage_sequence
        self._next_stage_sequence += 1
        token = canonical_content_hash(
            {
                "state_generation": self._state_generation,
                "stage_sequence": stage_sequence,
                "receipt_hash": receipt.semantic_content_hash,
            },
            domain="dexteleop.arm_hold_staged_transition.v1",
        )
        staged = ArmHoldStagedResolution(
            token=token,
            state_generation=self._state_generation,
            resolved_action=resolved,
            receipt=receipt,
        )
        self._pending = (staged, candidates)
        return staged

    def commit(
        self,
        staged: ArmHoldStagedResolution,
    ) -> ArmHoldResolutionReceipt:
        if not isinstance(staged, ArmHoldStagedResolution):
            raise TypeError("staged must be an ArmHoldStagedResolution.")
        pending = self._pending
        if pending is None or pending[0].token != staged.token:
            raise RuntimeError(
                "Arm-hold staged transition is not pending."
            )
        if staged.state_generation != self._state_generation:
            raise RuntimeError(
                "Arm-hold staged transition is stale."
            )
        self._states = pending[1]
        self._pending = None
        self._state_generation += 1
        return staged.receipt

    def discard(self, staged: ArmHoldStagedResolution) -> None:
        if not isinstance(staged, ArmHoldStagedResolution):
            raise TypeError("staged must be an ArmHoldStagedResolution.")
        pending = self._pending
        if pending is None or pending[0].token != staged.token:
            raise RuntimeError(
                "Arm-hold staged transition is not pending."
            )
        self._pending = None

    def resolve(
        self,
        action: NormalizedAction,
        observation: DeploymentObservation,
    ) -> tuple[NormalizedAction, ArmHoldResolutionReceipt]:
        staged = self.stage(action, observation)
        self.commit(staged)
        return staged.resolved_action, staged.receipt


__all__ = [
    "ARM_HOLD_RESOLUTION_HASH_DOMAIN_V1",
    "ARM_HOLD_RESOLUTION_RECEIPT_SCHEMA_V1",
    "ARM_HOLD_STATE_HOLD",
    "ARM_HOLD_STATE_MOVING",
    "ArmHoldResolutionReceipt",
    "ArmHoldSemanticResolver",
    "ArmHoldSideResolution",
    "ArmHoldStagedResolution",
]
