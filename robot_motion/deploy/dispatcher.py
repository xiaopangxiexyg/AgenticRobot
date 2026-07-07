from __future__ import annotations

from typing import Any, Protocol


class CommandDispatcher(Protocol):
    """Sim/real-neutral dispatcher for resolved runtime commands."""

    def dispatch(self, command: Any) -> None:
        ...


class DryRunCommandDispatcher:
    """Dispatcher that records commands without sending them to hardware or sim."""

    def __init__(self) -> None:
        self.commands: list[Any] = []

    def dispatch(self, command: Any) -> None:
        copy = getattr(command, "copy", None)
        self.commands.append(copy() if callable(copy) else command)
