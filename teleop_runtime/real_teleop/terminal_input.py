"""Bounded terminal-key adapter for real teleoperation."""

from __future__ import annotations

import select
import sys
import termios
import tty

from ..log_utils import log_event as _log
from ..sources.misc import KeyboardButtonSource

class TerminalClutchInput:
    _MAX_CHARS_PER_POLL = 64

    def __init__(self, keyboard_source: KeyboardButtonSource):
        self.keyboard_source = keyboard_source
        self.enabled = sys.stdin.isatty()
        self._old_attrs: list[int] | None = None
        self.should_quit = False
        self.left_pressed = False
        self.right_pressed = False
        self._calibration_requested = False
        self._episode_start_requested = False
        self._episode_stop_requested = False

    def start(self) -> None:
        if not self.enabled:
            return
        try:
            self._old_attrs = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
        except (OSError, termios.error):
            self.enabled = False
            self._old_attrs = None

    def close(self) -> None:
        if self.enabled and self._old_attrs is not None:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_attrs)
            except (OSError, termios.error):
                pass
            self._old_attrs = None

    def poll(self) -> None:
        if not self.enabled:
            return
        # Bound work per control frame. A held key, key-repeat stream, or pasted
        # terminal input must not keep this method draining stdin indefinitely
        # while the robot control loop is waiting to run.
        for _ in range(self._MAX_CHARS_PER_POLL):
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            except OSError:
                self.enabled = False
                self._old_attrs = None
                _log("Keys", "terminal input became unavailable; keyboard controls disabled for this run.")
                return
            if not ready:
                return
            try:
                char = sys.stdin.read(1)
            except OSError:
                self.enabled = False
                self._old_attrs = None
                _log("Keys", "terminal input became unavailable; keyboard controls disabled for this run.")
                return
            if not char:
                return
            if char == "q":
                self.should_quit = True
                return
            if char in {"p", "\n", "\r"}:
                self._calibration_requested = True
                continue
            if char == "t":
                self._episode_start_requested = True
                continue
            if char == "s":
                self._episode_stop_requested = True
                continue
            if char == "z":
                self.left_pressed = not self.left_pressed
            elif char == "x":
                self.right_pressed = not self.right_pressed
            elif char == "b":
                new_state = not (self.left_pressed and self.right_pressed)
                self.left_pressed = new_state
                self.right_pressed = new_state
            elif char == "r":
                self.left_pressed = False
                self.right_pressed = False
            else:
                continue

            self.keyboard_source.set_button(
                "left_clutch_button",
                self.left_pressed,
                side="left",
                critical=False,
            )
            self.keyboard_source.set_button(
                "right_clutch_button",
                self.right_pressed,
                side="right",
                critical=False,
            )

    def consume_calibration_request(self) -> bool:
        requested = self._calibration_requested
        self._calibration_requested = False
        return requested

    def consume_episode_start_request(self) -> bool:
        requested = self._episode_start_requested
        self._episode_start_requested = False
        return requested

    def consume_episode_stop_request(self) -> bool:
        requested = self._episode_stop_requested
        self._episode_stop_requested = False
        return requested

__all__ = ["TerminalClutchInput"]
