from __future__ import annotations

import os
import pty
import termios

from utils.terminal_commands import ImmediateTerminalCommands


def test_immediate_terminal_commands_filter_keys_and_restore_terminal() -> None:
    master_fd, terminal_fd = pty.openpty()
    original = termios.tcgetattr(terminal_fd)
    commands = ImmediateTerminalCommands(
        file_descriptor=terminal_fd,
        immediate=True,
        allowed_commands={"a", "c", "q"},
    )
    try:
        os.write(master_fd, b"xaCq")
        assert commands(0.0) == "a"
        assert commands(0.0) == "c"
        assert commands(0.0) == "q"
        assert commands(0.0) is None
    finally:
        commands.close()
        restored = termios.tcgetattr(terminal_fd)
        os.close(terminal_fd)
        os.close(master_fd)

    assert restored == original
