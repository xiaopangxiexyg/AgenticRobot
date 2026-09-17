"""Foreground operator driving for an already assembled B601/RM65 session.

This application leaf advances the injected target owner, not a second rig or
servo worker. Device construction and master gravity compensation stay outside
this loop. No module import or operator command enables/disables real motors.
"""

from __future__ import annotations

import math
import os
import select
import sys
from collections.abc import Callable, Mapping

from control_station.sources.rebot_b601_rm65_motion import B601RM65MotionSession
from hardware.rebot_b601_rs import B601HardwareDependencyError


class B601RM65TerminalCommands:
    """Per-session nonblocking line input, including pasted commands and EOF."""

    def __init__(self, *, file_descriptor: int | None = None) -> None:
        self.file_descriptor = (
            sys.stdin.fileno() if file_descriptor is None else file_descriptor
        )
        self._pending = b""
        self._eof = False

    def __call__(self, timeout_sec: float) -> str | None:
        if b"\n" not in self._pending and not self._eof:
            readable, _, _ = select.select(
                [self.file_descriptor], [], [], timeout_sec
            )
            if not readable:
                return None
            chunk = os.read(self.file_descriptor, 4096)
            self._pending += chunk
            self._eof = not chunk
        if b"\n" in self._pending:
            line, self._pending = self._pending.split(b"\n", 1)
            return line.decode("utf-8", errors="replace").strip().lower()
        if self._eof:
            self._pending = b""
            return "q"
        return None


def _print_status(message: str) -> None:
    print(message, flush=True)


def run_b601_rm65_foreground(
    session: B601RM65MotionSession,
    *,
    advance_target: Callable[[], object],
    rate_hz: float,
    poll_command: Callable[[float], str | None] | None = None,
    emit: Callable[[str], None] = _print_status,
    record: Callable[[Mapping[str, object]], None] | None = None,
) -> None:
    """Run in the calling foreground process until q/EOF/exception/Ctrl-C.

    a activates at the two measured current poses, c clutches, r releases and
    recenters, s deactivates, and q exits. The target's existing runtime step
    belongs in advance_target. poll_command waits at most its supplied timeout;
    it must not run a blocking prompt or launch a background input thread.
    record receives events for an application-owned JSON/log sink. It must not
    issue commands. Failures propagate after withdrawing target following.
    """
    if not isinstance(session, B601RM65MotionSession):
        raise TypeError("session must be B601RM65MotionSession.")
    if not math.isfinite(rate_hz) or rate_hz <= 0:
        raise ValueError("rate_hz must be finite and positive.")
    if not all(callable(value) for value in (advance_target, emit)):
        raise TypeError("Target, terminal and status callbacks must be callable.")
    if poll_command is not None and not callable(poll_command):
        raise TypeError("poll_command must be callable or None.")
    if record is not None and not callable(record):
        raise TypeError("record must be callable or None.")
    clock = session.binding.runtime.clock
    period = 1.0 / rate_hz

    def event(name: str, **details: object) -> None:
        if record is not None:
            record({"event": name, "at": clock(), "state": session.state, **details})

    def instructions() -> None:
        emit(
            "[B601→RM65] Enter: a=activate/retry c=clutch r=recenter s=stop q=quit ?=help"
        )
        emit("[B601→RM65] Start inactive; reposition/settle first, then a. c before recentering.")
        emit(
            "[B601→RM65] A feedback-capability error stays held; repair the source and press a again."
        )
        emit("[B601→RM65] Ctrl-C/EOF: stop RM65 following; master gravity/damping is unchanged.")

    def recoverable_feedback_error(command: str, error: B601HardwareDependencyError) -> None:
        """Keep the receive-only UI alive while the host feedback port recovers.

        The backend remains fail-closed: no old observation is reused and no
        target proposal is admitted.  This handler only converts a known
        dependency/feedback-capability failure into an operator-visible retry
        state.  Other exceptions still propagate to the application boundary.
        """

        message = str(error)
        emit(f"[B601→RM65] waiting: {type(error).__name__}: {message}")
        emit(
            "[B601→RM65] no RM65 following; fix the feedback runtime/stream, then press a to retry."
        )
        event(
            "feedback_capability_unavailable",
            command=command,
            error_type=type(error).__name__,
            error=message,
            retryable=True,
        )

    try:
        if poll_command is None:
            poll_command = B601RM65TerminalCommands()
        # Never inherit a previously retained reference when launching the UI.
        session.deactivate()
        instructions()
        event("foreground_started")
        next_tick = float(clock()) + period
        while True:
            command = poll_command(max(0.0, next_tick - float(clock())))
            if command is not None:
                command = command.strip().lower()
                if command == "q":
                    event("operator_quit")
                    return
                if command in {"", "?"}:
                    instructions()
                elif command in {"a", "c", "r", "s"}:
                    # Invalid state transitions are operator mistakes, not
                    # permission to re-open a fault or publish an old anchor.
                    allowed = {
                        "a": {"inactive", "held"},
                        "c": {"active", "clutched"},
                        "r": {"clutched"},
                        "s": {"inactive", "active", "held", "clutched"},
                    }
                    if session.state not in allowed[command]:
                        emit(f"[B601→RM65] {command} rejected in {session.state}")
                        event("operator_command_rejected", command=command)
                    else:
                        try:
                            {
                                "a": session.activate,
                                "c": session.engage_clutch,
                                "r": session.release_clutch,
                                "s": session.deactivate,
                            }[command]()
                        except B601HardwareDependencyError as error:
                            # A missing host observation capability is
                            # recoverable for this receive-only application.
                            # The session has already withdrawn its target
                            # anchor before re-raising the backend error.
                            recoverable_feedback_error(command, error)
                            next_tick = float(clock()) + period
                            continue
                        if command in {"a", "r"}:
                            # Activation/recenter consumes one complete master
                            # snapshot as the anchor. Do not immediately poll
                            # the same vendor cache frame again in this tick.
                            next_tick = float(clock()) + period
                        emit(f"[B601→RM65] {session.state}")
                        event("operator_command", command=command)
                else:
                    emit("[B601→RM65] unknown command; ? for help")
                    event("operator_command_rejected", command=command)
            now = float(clock())
            if now < next_tick:
                continue
            if session.state == "active":
                try:
                    receipt = session.step()
                except B601HardwareDependencyError as error:
                    # ``step`` has moved the session to held and cleared its
                    # anchor.  Keep advancing the lower target once below so
                    # the cancellation is observed without retaining a stale
                    # RM65 proposal.
                    recoverable_feedback_error("step", error)
                else:
                    event(
                        "proposal_admission",
                        accepted=receipt.accepted,
                        rejection_code=receipt.rejection_code,
                        rejection_message=receipt.message,
                    )
                    if not receipt.accepted:
                        emit(f"[B601→RM65] held: {receipt.rejection_code}; inspect before a")
            advance_target()
            # Do not burst stale catch-up proposals after slow polling/IK.
            next_tick += period
            completed_at = float(clock())
            if next_tick <= completed_at:
                next_tick = completed_at + period
    finally:
        # This cleanup never touches the independently owned master backend.
        # Lower cancellation errors must escape, not masquerade as success.
        session.deactivate()
        event("foreground_stopped")
        emit("[B601→RM65] following stopped; master state unchanged")
