"""Dependency-neutral UI ports consumed by teleoperation runtimes.

The lower runtime defines only the operations it needs. Control Station owns
the concrete browser transport and satisfies these protocols structurally.
"""

from __future__ import annotations

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


__all__ = ["OperatorUiPort"]
