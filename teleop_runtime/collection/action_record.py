from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from robot_motion.action import ActionExecutionResult, ActionRecord, encode_robot_action

from ..action import (
    TeleopCommandActionAdapterConfig,
    robot_action_from_teleop_command,
)
from ..recording import _to_serializable
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
        else robot_action_from_teleop_command(
            step_result.command,
            config=cfg.action_adapter,
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
