from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..schemas import EffectorCommand


class EffectorBackend(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    def start(self) -> None:
        """Optional resource allocation hook."""

    def is_ready(self) -> bool:
        return True

    @abstractmethod
    def send_command(self, command: EffectorCommand) -> None:
        raise NotImplementedError

    def hold_motion(self, reason: str = "manual hold") -> None:
        """Optional recoverable motion hold hook."""

    def reset_motion_state(self, qpos: Any | None = None) -> None:
        """Optional hook to discard buffered targets before restarting motion."""

    def close(self) -> None:
        """Optional resource release hook."""

    def __enter__(self) -> "EffectorBackend":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
