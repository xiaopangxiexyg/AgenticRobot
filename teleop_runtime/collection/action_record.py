from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from robot_motion.action import (
    ActionExecutionResult,
    ActionRecord,
    BIMANUAL_MIXED_HAND16_V1,
    BIMANUAL_MIXED_V1,
    RobotAction,
    encode_robot_action,
    split_effector_name,
)

from ..action import (
    TeleopCommandActionAdapterConfig,
    robot_action_from_teleop_command,
)
from ..recording import _to_serializable
from ..schemas import TeleopCommand
from ..session import SessionStepResult


@dataclass(frozen=True)
class CollectionActionRecordConfig:
    action_adapter: TeleopCommandActionAdapterConfig = field(
        default_factory=lambda: TeleopCommandActionAdapterConfig(
            arm_joint_policy="source_pose_when_available",
            include_command_metadata=False
        )
    )
    include_observation_ref: bool = False
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class CollectionCommandActionValidation:
    """Result of validating a command at the collection dispatch boundary.

    A mixed pose/qpos command is an execution-only intermediate form.  It has
    no stable persisted vector codec; dispatched arm qpos targets are replaced
    with each backend's accepted TCP target by ``CollectorSampleBuilder``
    before the canonical action record is encoded.
    """

    action: RobotAction
    vector_encoding_ready: bool
    backend_target_pose_sides: tuple[str, ...] = ()


_DEFERRED_BACKEND_POSE_ACTION_SPACES = frozenset(
    {BIMANUAL_MIXED_V1, BIMANUAL_MIXED_HAND16_V1}
)


def canonical_collection_action_from_command(
    command: TeleopCommand,
    *,
    config: CollectionActionRecordConfig | None = None,
) -> RobotAction | None:
    """Build the same canonical action used by formal collection metadata."""

    cfg = config if config is not None else CollectionActionRecordConfig()
    return robot_action_from_teleop_command(command, config=cfg.action_adapter)


def _validate_deferred_backend_pose_action(
    command: TeleopCommand,
    action: RobotAction,
) -> tuple[str, ...]:
    """Validate the only non-vector-ready command accepted before dispatch."""

    pose_sides = set(action.arm_pose_by_side)
    qpos_sides = set(action.arm_qpos_by_side)
    if not pose_sides or not qpos_sides or pose_sides & qpos_sides:
        raise ValueError(
            "A deferred mixed collection command requires disjoint, non-empty "
            "pose and qpos arm targets."
        )
    if action.end_effector_targets_by_side:
        raise ValueError(
            "A deferred mixed collection command cannot contain native "
            "end-effector targets."
        )

    active_arm_sides = {
        side
        for effector in action.active_effectors
        for side, kind in (split_effector_name(effector),)
        if kind == "arm"
    }
    represented_arm_sides = pose_sides | qpos_sides
    if active_arm_sides != represented_arm_sides:
        raise ValueError(
            "Deferred mixed collection arm targets do not exactly match the "
            "active arm effectors."
        )

    commands_by_effector = {}
    for command_name, effector_command in command.commands.items():
        effector = str(effector_command.effector_name or command_name)
        if effector in commands_by_effector:
            raise ValueError(
                f"Deferred mixed collection command duplicates effector {effector!r}."
            )
        commands_by_effector[effector] = effector_command
    for side in sorted(qpos_sides):
        effector = f"{side}_arm"
        effector_command = commands_by_effector.get(effector)
        if effector_command is None or effector_command.command_type != "joint_vector":
            raise ValueError(
                "Deferred backend target-pose normalization requires an exact "
                f"joint_vector command for {effector}."
            )
        if effector_command.side != side:
            raise ValueError(
                f"Deferred {effector} command declares side={effector_command.side!r}."
            )
    return tuple(sorted(qpos_sides))


def inspect_collection_command_action(
    command: TeleopCommand,
    *,
    config: CollectionActionRecordConfig | None = None,
    allow_backend_target_pose_defer: bool = False,
) -> CollectionCommandActionValidation:
    """Validate a command and describe whether persisted encoding is deferred."""

    action = canonical_collection_action_from_command(command, config=config)
    if action is None:
        raise ValueError("Collection command does not contain an active action.")
    if action.action_space in _DEFERRED_BACKEND_POSE_ACTION_SPACES:
        if not allow_backend_target_pose_defer:
            raise ValueError(
                f"Action space {action.action_space!r} requires post-dispatch "
                "backend target-pose normalization."
            )
        deferred_sides = _validate_deferred_backend_pose_action(command, action)
        return CollectionCommandActionValidation(
            action=action,
            vector_encoding_ready=False,
            backend_target_pose_sides=deferred_sides,
        )
    encode_robot_action(action)
    return CollectionCommandActionValidation(
        action=action,
        vector_encoding_ready=True,
    )


def validate_collection_command_action(
    command: TeleopCommand,
    *,
    config: CollectionActionRecordConfig | None = None,
    allow_backend_target_pose_defer: bool = False,
) -> RobotAction:
    """Require a command to satisfy the formal collection action contract."""

    return inspect_collection_command_action(
        command,
        config=config,
        allow_backend_target_pose_defer=allow_backend_target_pose_defer,
    ).action


def build_collection_action_record(
    step_result: SessionStepResult,
    *,
    config: CollectionActionRecordConfig | None = None,
) -> ActionRecord:
    existing = getattr(step_result, "action_record", None)
    if existing is not None and config is None:
        return existing
    cfg = config if config is not None else CollectionActionRecordConfig()
    action = (
        None
        if step_result.command is None
        else canonical_collection_action_from_command(
            step_result.command,
            config=cfg,
        )
    )
    execution = ActionExecutionResult(
        timestamp=float(step_result.timestamp),
        command=step_result.command,
        sent=bool(getattr(step_result, "command_dispatched", False)),
        faults=tuple(str(fault.code) for fault in step_result.faults),
        metadata={
            "session_state": step_result.state.value,
            "control_hz": float(step_result.control_hz),
        },
    )
    return ActionRecord(
        timestamp=float(step_result.timestamp),
        observation=step_result.observation if cfg.include_observation_ref else None,
        action=action,
        command=step_result.command,
        execution=execution,
        metadata={} if cfg.metadata is None else dict(cfg.metadata),
    )


def collection_action_record_metadata(
    step_result: SessionStepResult,
    *,
    config: CollectionActionRecordConfig | None = None,
    include_debug_metadata: bool = False,
) -> dict[str, Any]:
    record = build_collection_action_record(step_result, config=config)
    action_vector = None
    action_vector_error = None
    if record.action is not None:
        try:
            action_vector = encode_robot_action(record.action)
        except ValueError as exc:
            action_vector_error = str(exc)
    execution = {
        "timestamp": float(record.execution.timestamp),
        "sent": bool(record.execution.sent),
        "faults": list(record.execution.faults),
        "metadata": _to_serializable(record.execution.metadata),
    }
    if include_debug_metadata:
        execution["command"] = _to_serializable(record.execution.command)
    metadata = {
        "action": _to_serializable(record.action),
        "action_vector": _to_serializable(action_vector),
        "execution": execution,
        "record_timestamp": float(record.timestamp),
        "metadata": _to_serializable(record.metadata),
    }
    if action_vector_error is not None:
        metadata["action_vector_error"] = action_vector_error
    return metadata
