from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping

from .contracts import EffectorCommand, ObservationChannel, TeleopCommand, TeleopObservation


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
    def send_command(self, command: EffectorCommand) -> object | None:
        """Dispatch a command and optionally return a backend-specific receipt."""
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


class ObservationSource(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    def start(self) -> None:
        """Optional resource allocation hook."""

    @abstractmethod
    def get_latest_channels(self) -> dict[str, ObservationChannel]:
        raise NotImplementedError

    def expected_channel_names(self) -> tuple[str, ...]:
        return ()

    def close(self) -> None:
        """Optional resource release hook."""

    def __enter__(self) -> "ObservationSource":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


class Retargeter(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def retarget(self, observation: TeleopObservation, calibration: Any | None = None) -> TeleopCommand:
        raise NotImplementedError


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
    def send_commands(
        self,
        command: TeleopCommand,
        phase_recorder: Callable[[str, float], None] | None = None,
    ) -> Mapping[str, object | None] | None:
        """Dispatch a command and optionally return receipts by effector.

        Backward-compatible rigs may return ``None``. Explicit transport
        rejection must propagate as an exception rather than a false-success
        return value.
        """
        raise NotImplementedError

    def hold_motion(self, reason: str = "manual hold") -> None:
        """Optional recoverable motion hold hook."""

    def close(self) -> None:
        """Optional resource release hook."""

    def __enter__(self) -> "Rig":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


__all__ = ["EffectorBackend", "ObservationSource", "Retargeter", "Rig"]
