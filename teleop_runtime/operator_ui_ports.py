"""Dependency-neutral UI ports consumed by teleoperation runtimes.

The lower runtime defines only the operations it needs. Control Station owns
the concrete browser transport and satisfies these protocols structurally.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol


class OperatorUiPort(Protocol):
    """Minimal bidirectional operator UI seam shared by RealTeleop and UMI."""

    @property
    def url(self) -> str: ...

    def start(self) -> None: ...

    def close(self) -> None: ...

    def consume_events(self, *, max_events: int = 64) -> list[str]: ...

    def publish(self, **updates: Any) -> None: ...

    def append_message(self, tag: str, message: str) -> None: ...


class OperatorInputBatchView(Protocol):
    """Prioritized application input returned to the lower runtime."""

    events: tuple[str, ...]

    @property
    def terminal_quit_requested(self) -> bool: ...

    @property
    def suppress_foot_pad_events(self) -> bool: ...


class RealTeleopInputBatchPrioritizer(Protocol):
    """Application-owned precedence policy consumed by RealTeleop."""

    def __call__(
        self,
        events: Iterable[str],
        *,
        terminal_quit_requested: bool,
        teleop_started: bool,
        two_pedal_mode_active: bool,
    ) -> OperatorInputBatchView: ...


__all__ = [
    "OperatorInputBatchView",
    "OperatorUiPort",
    "RealTeleopInputBatchPrioritizer",
]
