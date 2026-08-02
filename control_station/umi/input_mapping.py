"""Pure operator-input mapping helpers for the UMI application."""

from __future__ import annotations

from typing import Iterable, NamedTuple


class UmiOperatorEvent(NamedTuple):
    """One sampled UMI operator event with immutable provenance."""

    name: str
    source: str
    timestamp: float


def merge_operator_events(
    web_events: Iterable[str],
    foot_pad_events: Iterable[str],
    *,
    timestamp: float,
) -> tuple[UmiOperatorEvent, ...]:
    """Merge one sampled UI/foot-pad cycle while preserving legacy order."""

    sampled_at = float(timestamp)
    return tuple(
        UmiOperatorEvent(str(event), "web_gui", sampled_at)
        for event in web_events
    ) + tuple(
        UmiOperatorEvent(str(event), "foot_pad", sampled_at)
        for event in foot_pad_events
    )


def foot_pad_test_batch_active(
    events: Iterable[UmiOperatorEvent | tuple[str, str, float]],
    *,
    test_already_active: bool,
) -> bool:
    """Return whether same-cycle foot-pad edges must remain diagnostics.

    Start and stop clicks are both barriers for the sampled cycle.  This
    preserves the legacy race protection: a pedal edge sampled beside either
    click cannot leak into a real UMI annotation.
    """

    if test_already_active:
        return True
    return any(
        str(event[0]) in {"foot_pad_test_start", "foot_pad_test_stop"}
        for event in events
    )


__all__ = [
    "UmiOperatorEvent",
    "foot_pad_test_batch_active",
    "merge_operator_events",
]
