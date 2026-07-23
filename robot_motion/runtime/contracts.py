from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

if TYPE_CHECKING:
    from .cancellation import ExecutionWorkItem


def copy_runtime_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: copy_runtime_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [copy_runtime_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(copy_runtime_value(item) for item in value)
    return value


@dataclass
class ObservationChannel:
    channel_name: str
    capability: str
    side: str | None
    value: Any
    frame_name: str | None
    timestamp: float
    source_name: str
    valid: bool = True
    critical: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "ObservationChannel":
        return ObservationChannel(
            channel_name=self.channel_name,
            capability=self.capability,
            side=self.side,
            value=copy_runtime_value(self.value),
            frame_name=self.frame_name,
            timestamp=self.timestamp,
            source_name=self.source_name,
            valid=self.valid,
            critical=self.critical,
            metadata=dict(self.metadata),
        )


@dataclass
class TeleopObservation:
    channels: dict[str, ObservationChannel] = field(default_factory=dict)
    timestamp: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_channel(self, channel_name: str) -> ObservationChannel | None:
        return self.channels.get(channel_name)

    def require_channel(self, channel_name: str) -> ObservationChannel:
        if channel_name not in self.channels:
            raise KeyError(f"Missing observation channel: {channel_name}")
        return self.channels[channel_name]


@dataclass
class EffectorCommand:
    effector_name: str
    side: str | None
    command_type: str
    value: Any
    frame_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Typed execution lineage is deliberately separate from free-form
    # metadata.  Legacy callers leave it unset; an epoch-mediated execution
    # profile must bind it before work can enter an asynchronous backend.
    execution_work_item: "ExecutionWorkItem | None" = None

    def copy(self) -> "EffectorCommand":
        return EffectorCommand(
            effector_name=self.effector_name,
            side=self.side,
            command_type=self.command_type,
            value=copy_runtime_value(self.value),
            frame_name=self.frame_name,
            metadata=dict(self.metadata),
            execution_work_item=self.execution_work_item,
        )


@dataclass
class TeleopCommand:
    commands: dict[str, EffectorCommand] = field(default_factory=dict)
    timestamp: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    execution_work_item: "ExecutionWorkItem | None" = None

    def add_command(self, command: EffectorCommand) -> None:
        self.commands[command.effector_name] = command


MotionObservation = TeleopObservation
MotionCommand = TeleopCommand


def merge_teleop_commands(
    commands: Mapping[str, TeleopCommand] | list[TeleopCommand] | tuple[TeleopCommand, ...]
) -> TeleopCommand:
    merged = TeleopCommand()
    if isinstance(commands, Mapping):
        iterable = commands.values()
    else:
        iterable = commands
    latest_timestamp: float | None = None
    for command in iterable:
        if command is None:
            continue
        if command.execution_work_item is not None:
            if (
                merged.execution_work_item is not None
                and merged.execution_work_item != command.execution_work_item
            ):
                raise ValueError(
                    "Cannot merge commands from different execution work items."
                )
            merged.execution_work_item = command.execution_work_item
        for effector_name, effector_command in command.commands.items():
            merged.commands[effector_name] = effector_command.copy()
        merged.metadata.update(command.metadata)
        if command.timestamp is not None and (
            latest_timestamp is None or command.timestamp > latest_timestamp
        ):
            latest_timestamp = command.timestamp
    merged.timestamp = latest_timestamp
    return merged


def latest_channel_timestamp(channels: Mapping[str, ObservationChannel]) -> float | None:
    timestamps = [channel.timestamp for channel in channels.values()]
    return max(timestamps) if timestamps else None


__all__ = [
    "EffectorCommand",
    "MotionCommand",
    "MotionObservation",
    "ObservationChannel",
    "TeleopCommand",
    "TeleopObservation",
    "copy_runtime_value",
    "latest_channel_timestamp",
    "merge_teleop_commands",
]
