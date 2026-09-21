"""Pure operator-input batch policy for the RealTeleop application.

This module does not poll keyboards, web queues, or foot pads.  It only
applies the application-level precedence contract to one already-sampled
batch so the lower runtime can execute the resulting intents exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal


RealTeleopInputBarrier = Literal[
    "none",
    "terminal_quit",
    "ui_quit",
    "init_emergency_stop",
    "hold",
]


@dataclass(frozen=True)
class RealTeleopInputBatch:
    """Prioritized view of one already-sampled operator input batch."""

    events: tuple[str, ...]
    barrier: RealTeleopInputBarrier = "none"

    @property
    def terminal_quit_requested(self) -> bool:
        return self.barrier == "terminal_quit"

    @property
    def suppress_foot_pad_events(self) -> bool:
        return self.barrier != "none"


def prioritize_input_batch(
    events: Iterable[str],
    *,
    terminal_quit_requested: bool,
    teleop_started: bool,
    two_pedal_mode_active: bool,
) -> RealTeleopInputBatch:
    """Apply RealTeleop lifecycle/safety precedence without side effects.

    Event order and duplicates are preserved for non-barrier events.  A Hold
    barrier retains an episode-stop event from the same sampled batch because
    stopping collection is compatible with entering Hold; all other lower
    priority events are suppressed until the runtime has applied the Hold.
    """

    sampled = tuple(str(event) for event in events)
    if terminal_quit_requested:
        return RealTeleopInputBatch(events=(), barrier="terminal_quit")
    if "q" in sampled:
        return RealTeleopInputBatch(events=("q",), barrier="ui_quit")
    if "init_qpos_emergency_stop" in sampled:
        return RealTeleopInputBatch(
            events=("init_qpos_emergency_stop",),
            barrier="init_emergency_stop",
        )

    hold_requested = bool(
        teleop_started
        and (
            "h" in sampled
            or (two_pedal_mode_active and "two_pedal_c" in sampled)
        )
    )
    if hold_requested:
        return RealTeleopInputBatch(
            events=tuple(
                event
                for event in sampled
                if (
                    event in {
                        "h",
                        "two_pedal_c",
                        # Keep the explicit rule Home request observable so the
                        # lower runtime can reject it with a retryable state after
                        # Hold admission, instead of leaving the UI pending when
                        # both clicks arrived in one sampled batch.
                        "runtime_home_retry",
                    }
                    or event == "episode_stop"
                    or event.startswith("episode_stop:")
                )
            ),
            barrier="hold",
        )
    return RealTeleopInputBatch(events=sampled)


__all__ = [
    "RealTeleopInputBarrier",
    "RealTeleopInputBatch",
    "prioritize_input_batch",
]
