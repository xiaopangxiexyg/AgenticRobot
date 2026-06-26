from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..calibration import TeleopCalibration
from ..interfaces import Retargeter
from ..schemas import TeleopCommand, TeleopObservation, merge_teleop_commands
from .adapters import TeleopCommandActionAdapterConfig, robot_action_from_teleop_command
from .schema import RobotAction


@dataclass(frozen=True)
class TeleopRetargeterActionSourceConfig:
    name: str = "teleop_retargeter_action_source"
    action_adapter: TeleopCommandActionAdapterConfig = field(
        default_factory=TeleopCommandActionAdapterConfig
    )


class TeleopRetargeterActionSource:
    """Wrap existing retargeters as a canonical RobotAction source."""

    def __init__(
        self,
        *,
        retargeters: Sequence[Retargeter],
        calibration: TeleopCalibration | None = None,
        config: TeleopRetargeterActionSourceConfig | None = None,
    ):
        self.retargeters = list(retargeters)
        self.calibration = calibration
        self.config = config if config is not None else TeleopRetargeterActionSourceConfig()
        self.last_command: TeleopCommand | None = None

    @property
    def name(self) -> str:
        return self.config.name

    def start(self) -> None:
        pass

    def produce_action(
        self,
        observation: TeleopObservation,
        *,
        context=None,
    ) -> RobotAction | None:
        commands = [
            retargeter.retarget(observation, self.calibration)
            for retargeter in self.retargeters
        ]
        command = merge_teleop_commands(commands)
        if command.timestamp is None:
            command.timestamp = observation.timestamp
        self.last_command = command
        return robot_action_from_teleop_command(
            command,
            config=self.config.action_adapter,
        )

    def close(self) -> None:
        pass
