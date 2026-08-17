"""Source-neutral command override contract for one lower control frame."""

from __future__ import annotations

from dataclasses import dataclass

from robot_motion.runtime import TeleopCommand


@dataclass(frozen=True)
class TeleopCommandOverride:
    """Claim one control frame, optionally with no command.

    ``command=None`` is an explicit no-command claim. It is distinct from an
    absent provider and prevents fallback to another command source.
    """

    command: TeleopCommand | None

    def __post_init__(self) -> None:
        if self.command is not None and not isinstance(self.command, TeleopCommand):
            raise TypeError(
                "TeleopCommandOverride.command must be TeleopCommand or None."
            )


__all__ = ["TeleopCommandOverride"]
