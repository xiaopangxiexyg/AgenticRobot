from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from robot_motion.action import (
    RobotAction,
    RobotActionResolver,
    VectorActionLayout,
    robot_action_from_vector_action,
)
from robot_motion.runtime import TeleopCommand

from .actions import ActionSafetyFilter, FilteredNormalizedAction, NormalizedAction
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


DEPLOY_VECTOR_ACTION_LAYOUT = VectorActionLayout(
    action_dim=ACTION_DIM,
    arm_pose_slices=ACTION_ARM_POSE_SLICES,
    hand_slices=ACTION_HAND_SLICES,
    arm_pose_dof=ARM_POSE_DOF,
    arm_joint_dof=ARM_JOINT_DOF,
    hand_dof=HAND_DOF,
    side_order=ALL_SIDES,
)


@dataclass(frozen=True)
class CanonicalDeployActionResult:
    filtered: FilteredNormalizedAction
    robot_action: RobotAction | None = None
    command: TeleopCommand | None = None
    faults: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def action(self) -> NormalizedAction:
        return self.filtered.action

    def __post_init__(self) -> None:
        object.__setattr__(self, "faults", tuple(str(fault) for fault in self.faults))
        object.__setattr__(self, "metadata", dict(self.metadata))


class CanonicalDeployActionPipeline:
    """Prepare deploy actions through the canonical RobotAction boundary."""

    def __init__(
        self,
        *,
        safety_filter: ActionSafetyFilter,
        active_sides: Sequence[str],
        arms_enabled: bool,
        hands_enabled: bool,
        resolver: RobotActionResolver | None = None,
        layout: VectorActionLayout = DEPLOY_VECTOR_ACTION_LAYOUT,
    ):
        self.safety_filter = safety_filter
        self.active_sides = tuple(str(side) for side in active_sides)
        self.arms_enabled = bool(arms_enabled)
        self.hands_enabled = bool(hands_enabled)
        self.resolver = resolver if resolver is not None else RobotActionResolver()
        self.layout = layout

    def prepare(
        self,
        action: NormalizedAction | Any,
        observation: DeploymentObservation,
        *,
        policy_name: str = "",
    ) -> CanonicalDeployActionResult:
        filtered = self.safety_filter.filter_normalized(
            action,
            observation,
            policy_name=policy_name,
        )
        return self.resolve_filtered(
            filtered,
            observation,
            policy_name=policy_name,
        )

    def resolve_filtered(
        self,
        filtered: FilteredNormalizedAction,
        observation: DeploymentObservation,
        *,
        policy_name: str = "",
    ) -> CanonicalDeployActionResult:
        """Resolve an action that has already passed the keyframe safety gate.

        Position deployment deliberately separates the 15 Hz keyframe safety
        pass from the high-rate reference generator.  Calling ``prepare`` for every
        interpolated reference would incorrectly re-apply the 15 Hz step gate
        three times and would turn a safety limiter into hidden temporal
        smoothing.  This method preserves the original safety result while
        still routing every high-rate reference through the canonical
        ``RobotAction``/``TeleopCommand`` boundary.
        """

        if not isinstance(filtered, FilteredNormalizedAction):
            raise TypeError("filtered must be a FilteredNormalizedAction")
        metadata = dict(filtered.action.metadata)
        action_source = str(metadata.get("source") or "").strip()
        if not action_source:
            action_source = f"deploy_policy:{policy_name}" if policy_name else "deploy_policy"
            metadata["source"] = action_source

        build = robot_action_from_vector_action(
            vector=filtered.action.vector,
            arm_joint_targets=filtered.action.arm_joint_targets,
            metadata=metadata,
            active_sides=self.active_sides,
            arms_enabled=self.arms_enabled,
            hands_enabled=self.hands_enabled,
            layout=self.layout,
            timestamp=observation.timestamp,
            source=action_source,
        )
        faults = [*filtered.faults, *build.faults]
        command = None
        if build.robot_action is not None and not faults:
            command = self.resolver.resolve(build.robot_action)
        if command is not None:
            _annotate_canonical_command(
                command,
                policy_name=policy_name,
                safety_metadata=filtered.metadata,
            )
        return CanonicalDeployActionResult(
            filtered=filtered,
            robot_action=build.robot_action,
            command=command,
            faults=tuple(faults),
            metadata={
                "canonical_action_path": True,
                "command_blocked_by_faults": bool(faults),
                "vector_build": dict(build.metadata),
                "safety": dict(filtered.metadata),
            },
        )


def _annotate_canonical_command(
    command: TeleopCommand,
    *,
    policy_name: str,
    safety_metadata: Mapping[str, Any],
) -> None:
    command.metadata["source"] = "deploy_canonical_robot_action"
    command.metadata["policy_name"] = str(policy_name)
    command.metadata["deploy_safety"] = dict(safety_metadata)
    for effector_command in command.commands.values():
        effector_command.metadata["deploy_source"] = "deploy_canonical_robot_action"
        effector_command.metadata["policy_name"] = str(policy_name)
        effector_command.metadata["deploy_safety"] = dict(safety_metadata)


__all__ = [
    "CanonicalDeployActionPipeline",
    "CanonicalDeployActionResult",
    "DEPLOY_VECTOR_ACTION_LAYOUT",
]
