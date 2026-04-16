from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..schemas import TeleopCommand


class Rig(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    def start(self) -> None:
        """Optional resource allocation hook."""

    def is_ready(self) -> bool:
        return True

    @abstractmethod
    def send_commands(self, command: TeleopCommand) -> None:
        raise NotImplementedError

    def close(self) -> None:
        """Optional resource release hook."""

    def __enter__(self) -> "Rig":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
