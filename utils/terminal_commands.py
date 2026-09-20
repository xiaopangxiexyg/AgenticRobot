"""Small foreground terminal-command reader with owned cbreak lifecycle.

The reader owns only terminal mode and buffered bytes.  Callers decide what a
key means and remain responsible for every control or simulation transition.
"""

from __future__ import annotations

import contextlib
import os
import select
import sys
import termios
import tty
from collections.abc import Collection

DEFAULT_IMMEDIATE_COMMANDS = frozenset({"a", "c", "r", "s", "q", "?"})


class ImmediateTerminalCommands:
    """Read normalized foreground commands without requiring Enter.

    A real controlling terminal is placed in cbreak mode and restored by
    :meth:`close`.  Pipes retain a newline-delimited protocol for tests and
    scripted diagnostics.  No background input thread is created.
    """

    def __init__(
        self,
        *,
        file_descriptor: int | None = None,
        immediate: bool | None = None,
        allowed_commands: Collection[str] = DEFAULT_IMMEDIATE_COMMANDS,
    ) -> None:
        normalized_commands = frozenset(
            str(command).strip().lower() for command in allowed_commands
        )
        if not normalized_commands or any(
            len(command) != 1 for command in normalized_commands
        ):
            raise ValueError("allowed_commands must contain single characters")
        self.allowed_commands = normalized_commands
        self.file_descriptor = (
            sys.stdin.fileno() if file_descriptor is None else file_descriptor
        )
        self._pending = b""
        self._eof = False
        self._immediate = False
        self._saved_attributes: list[object] | None = None
        if immediate is not None and not isinstance(immediate, bool):
            raise TypeError("immediate must be bool or None")
        try:
            is_tty = os.isatty(self.file_descriptor)
        except OSError:
            is_tty = False
        use_immediate = is_tty if immediate is None else bool(immediate)
        if use_immediate:
            try:
                self._saved_attributes = termios.tcgetattr(self.file_descriptor)
                tty.setcbreak(self.file_descriptor)
                termios.tcflush(self.file_descriptor, termios.TCIFLUSH)
                self._immediate = True
            except (OSError, termios.error):
                self._saved_attributes = None
                self._immediate = False

    def __call__(self, timeout_sec: float) -> str | None:
        if self._immediate:
            return self._poll_immediate(timeout_sec)
        return self._poll_line(timeout_sec)

    def _read(self, timeout_sec: float) -> bool:
        if not self._eof:
            readable, _, _ = select.select(
                [self.file_descriptor],
                [],
                [],
                timeout_sec,
            )
            if not readable:
                return False
            chunk = os.read(self.file_descriptor, 4096)
            self._pending += chunk
            self._eof = not chunk
            return bool(chunk)
        return False

    def _poll_line(self, timeout_sec: float) -> str | None:
        if b"\n" not in self._pending and not self._eof:
            self._read(timeout_sec)
        if b"\n" in self._pending:
            line, self._pending = self._pending.split(b"\n", 1)
            return line.decode("utf-8", errors="replace").strip().lower()
        if self._eof:
            self._pending = b""
            return "q"
        return None

    def _poll_immediate(self, timeout_sec: float) -> str | None:
        if not self._pending and not self._eof:
            self._read(timeout_sec)
        while self._pending:
            value, self._pending = self._pending[:1], self._pending[1:]
            character = value.decode("ascii", errors="ignore").lower()
            if character in self.allowed_commands:
                return character
        if self._eof:
            self._pending = b""
            return "q"
        return None

    def close(self) -> None:
        """Restore terminal mode after a foreground run."""

        if self._saved_attributes is not None:
            with contextlib.suppress(OSError, termios.error):
                termios.tcsetattr(
                    self.file_descriptor,
                    termios.TCSADRAIN,
                    self._saved_attributes,
                )
        self._saved_attributes = None
        self._immediate = False


__all__ = ["DEFAULT_IMMEDIATE_COMMANDS", "ImmediateTerminalCommands"]
