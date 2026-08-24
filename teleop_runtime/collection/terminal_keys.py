"""Terminal-local keyboard input for the headless collection loop.

The controller owns only its terminal file descriptor and cbreak lifecycle.
It does not interpret collection commands or mutate collection state; callers
remain responsible for mapping returned characters to operator actions.
"""

from __future__ import annotations

import os
import select
import sys
import termios
import tty


class TerminalKeyController:
    """Best-effort non-blocking terminal character reader."""

    def __init__(self) -> None:
        self.enabled = False
        self.input_source = "disabled"
        self.disabled_reason: str | None = None
        self._fd: int | None = None
        self._owns_fd = False
        self._old_attrs: list[int] | None = None

    def start(self) -> None:
        self.close()
        try:
            if sys.stdin.isatty():
                self._fd = sys.stdin.fileno()
                self._owns_fd = False
                self.input_source = "stdin"
            else:
                self._fd = os.open("/dev/tty", os.O_RDONLY)
                self._owns_fd = True
                self.input_source = "/dev/tty"
            self._old_attrs = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            self.enabled = True
            self.disabled_reason = None
        except (OSError, termios.error) as exc:
            self.enabled = False
            self.disabled_reason = f"{type(exc).__name__}: {exc}"
            self.input_source = "disabled"
            self._old_attrs = None
            if self._owns_fd and self._fd is not None:
                try:
                    os.close(self._fd)
                except OSError:
                    pass
            self._fd = None
            self._owns_fd = False

    def close(self) -> None:
        if self._fd is not None and self._old_attrs is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)
            except (OSError, termios.error):
                pass
        if self._owns_fd and self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
        self.enabled = False
        self._fd = None
        self._owns_fd = False
        self._old_attrs = None

    def poll(self) -> list[str]:
        if not self.enabled or self._fd is None:
            return []
        keys: list[str] = []
        while True:
            try:
                ready, _, _ = select.select([self._fd], [], [], 0.0)
            except OSError as exc:
                self.enabled = False
                self.disabled_reason = f"{type(exc).__name__}: {exc}"
                self._old_attrs = None
                return keys
            if not ready:
                return keys
            try:
                chunk = os.read(self._fd, 32)
            except OSError as exc:
                self.enabled = False
                self.disabled_reason = f"{type(exc).__name__}: {exc}"
                return keys
            if not chunk:
                return keys
            text = chunk.decode(errors="ignore")
            for char in text:
                if char == "\x03":
                    raise KeyboardInterrupt
                keys.append(char.lower())


__all__ = ["TerminalKeyController"]
