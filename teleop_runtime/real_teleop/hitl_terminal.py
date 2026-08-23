"""Terminal input compatible with RealTeleop plus edge-triggered HITL keys."""

from __future__ import annotations

from collections import deque
import select
import sys
import termios
import tty


_RESERVED_REAL_TELEOP_KEYS = frozenset(
    {"q", "p", "\n", "\r", "t", "s", "z", "x", "b", "r"}
)


def validate_hitl_terminal_keys(
    takeover_key: str,
    policy_resume_key: str,
) -> tuple[str, str]:
    """Validate two additive keys without constructing a terminal resource."""

    takeover = str(takeover_key)
    resume = str(policy_resume_key)
    if len(takeover) != 1 or len(resume) != 1:
        raise ValueError("HITL terminal keys must be one character each.")
    if takeover == resume:
        raise ValueError("HITL takeover and policy-resume keys must differ.")
    conflicts = {takeover, resume} & _RESERVED_REAL_TELEOP_KEYS
    if conflicts:
        raise ValueError(
            "HITL terminal keys conflict with established RealTeleop keys: "
            + ",".join(sorted(repr(value) for value in conflicts))
        )
    return takeover, resume


class HITLTerminalClutchInput:
    """Preserve all established keys and add Human/Policy selection edges."""

    _MAX_CHARS_PER_POLL = 64

    def __init__(
        self,
        keyboard_source: object,
        *,
        takeover_key: str = "u",
        policy_resume_key: str = "i",
    ) -> None:
        takeover, resume = validate_hitl_terminal_keys(
            takeover_key,
            policy_resume_key,
        )
        self.keyboard_source = keyboard_source
        self.takeover_key = takeover
        self.policy_resume_key = resume
        self.enabled = sys.stdin.isatty()
        self._old_attrs: list[int] | None = None
        self.should_quit = False
        self.left_pressed = False
        self.right_pressed = False
        self._calibration_requested = False
        self._episode_start_requested = False
        self._episode_stop_requested = False
        self._hitl_source_requests: deque[str] = deque()

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
                termios.tcsetattr(
                    sys.stdin.fileno(),
                    termios.TCSADRAIN,
                    self._old_attrs,
                )
            except (OSError, termios.error):
                pass
        self._old_attrs = None

    def _publish_clutch_buttons(self) -> None:
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

    def _handle_character(self, char: str) -> bool:
        """Apply one sampled character; return False when polling should stop."""

        if char == "q":
            self.should_quit = True
            return False
        if char in {"p", "\n", "\r"}:
            self._calibration_requested = True
            return True
        if char == "t":
            self._episode_start_requested = True
            return True
        if char == "s":
            self._episode_stop_requested = True
            return True
        if char == self.takeover_key:
            self._hitl_source_requests.append("human")
            return True
        if char == self.policy_resume_key:
            self._hitl_source_requests.append("policy")
            return True
        changed = True
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
            changed = False
        if changed:
            self._publish_clutch_buttons()
        return True

    def poll(self) -> None:
        if not self.enabled:
            return
        for _ in range(self._MAX_CHARS_PER_POLL):
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            except OSError:
                self.enabled = False
                self._old_attrs = None
                return
            if not ready:
                return
            try:
                char = sys.stdin.read(1)
            except OSError:
                self.enabled = False
                self._old_attrs = None
                return
            if not char or not self._handle_character(char):
                return

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

    def consume_hitl_source_requests(self) -> tuple[str, ...]:
        requests = tuple(self._hitl_source_requests)
        self._hitl_source_requests.clear()
        return requests


__all__ = ["HITLTerminalClutchInput", "validate_hitl_terminal_keys"]
