from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


def _copy_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _copy_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_value(item) for item in value)
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
            value=_copy_value(self.value),
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

    def copy(self) -> "EffectorCommand":
        return EffectorCommand(
            effector_name=self.effector_name,
            side=self.side,
            command_type=self.command_type,
            value=_copy_value(self.value),
            frame_name=self.frame_name,
            metadata=dict(self.metadata),
        )


@dataclass
class TeleopCommand:
    commands: dict[str, EffectorCommand] = field(default_factory=dict)
    timestamp: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_command(self, command: EffectorCommand) -> None:
        self.commands[command.effector_name] = command


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
