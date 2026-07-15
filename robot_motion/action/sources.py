from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from .schema import RobotAction


class RobotActionSource(Protocol):
    """Produces canonical actions from a pre-action observation."""

    def start(self) -> None:
        ...

    def produce_action(self, observation: Any, *, context: Any = None) -> RobotAction | None:
        ...

    def close(self) -> None:
        ...


class CallableActionSource:
    def __init__(
        self,
        producer: Callable[[Any], RobotAction | None],
        *,
        name: str = "callable_action_source",
    ):
        self.producer = producer
        self.name = str(name)

    def start(self) -> None:
        pass

    def produce_action(self, observation: Any, *, context: Any = None) -> RobotAction | None:
        return self.producer(observation)

    def close(self) -> None:
        pass


class StaticActionSource:
    def __init__(self, action: RobotAction | None, *, name: str = "static_action_source"):
        self.action = action
        self.name = str(name)

    def start(self) -> None:
        pass

    def produce_action(self, observation: Any, *, context: Any = None) -> RobotAction | None:
        return self.action

    def close(self) -> None:
        pass


__all__ = ["CallableActionSource", "RobotActionSource", "StaticActionSource"]
