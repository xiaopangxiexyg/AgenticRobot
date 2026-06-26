from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

PhaseTiming = tuple[str, float]


@dataclass(frozen=True)
class LatencyItem:
    name: str
    age_sec: float


@dataclass(frozen=True)
class EndToEndLatencyBreakdown:
    reference_time: float
    step_age_sec: float | None = None
    observation_age_sec: float | None = None
    command_age_sec: float | None = None
    channel_age_sec: tuple[LatencyItem, ...] = ()
    receiver_age_sec: tuple[LatencyItem, ...] = ()
    sender_age_sec: tuple[LatencyItem, ...] = ()


class PhaseTimer:
    def __init__(self, *, clock: Callable[[], float] | None = None):
        self._clock = time.perf_counter if clock is None else clock
        self._started_at = float(self._clock())
        self._last_at = self._started_at
        self._phases: list[PhaseTiming] = []

    @property
    def phases(self) -> tuple[PhaseTiming, ...]:
        return tuple(self._phases)

    def mark(self, name: str) -> float:
        now = float(self._clock())
        elapsed = now - self._last_at
        self._phases.append((str(name), elapsed))
        self._last_at = now
        return elapsed

    def record(self, name: str, elapsed_sec: float) -> None:
        self._phases.append((str(name), float(elapsed_sec)))

    def elapsed_sec(self) -> float:
        return float(self._clock()) - self._started_at


def format_phase_timings(
    phases: Sequence[PhaseTiming],
    *,
    limit: int = 10,
) -> str:
    if not phases:
        return "phases=none"
    ordered = sorted(
        ((str(name), float(elapsed)) for name, elapsed in phases),
        key=lambda item: item[1],
        reverse=True,
    )
    return "phases=" + ", ".join(
        f"{name}:{elapsed * 1000.0:.1f}ms" for name, elapsed in ordered[: max(1, int(limit))]
    )


def should_report_debug_frozen(
    *,
    elapsed_sec: float,
    phases: Sequence[PhaseTiming] = (),
    threshold_sec: float,
    force: bool = False,
) -> bool:
    if force:
        return True
    threshold = float(threshold_sec)
    return float(elapsed_sec) >= threshold or any(
        float(duration) >= threshold for _, duration in phases
    )


def _timestamp_age(reference_time: float, timestamp: Any) -> float | None:
    if timestamp is None:
        return None
    try:
        value = float(timestamp)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return float(reference_time) - value


def _latency_item(reference_time: float, name: str, timestamp: Any) -> LatencyItem | None:
    age = _timestamp_age(reference_time, timestamp)
    if age is None:
        return None
    return LatencyItem(name=str(name), age_sec=age)


def build_end_to_end_latency_breakdown(
    result: Any | None,
    *,
    reference_time: float | None = None,
) -> EndToEndLatencyBreakdown:
    ref = time.time() if reference_time is None else float(reference_time)
    if result is None:
        return EndToEndLatencyBreakdown(reference_time=ref)

    observation = getattr(result, "observation", None)
    command = getattr(result, "command", None)
    channels = getattr(result, "channels", {}) or {}
    if not isinstance(channels, Mapping):
        channels = {}

    channel_items: list[LatencyItem] = []
    receiver_items: list[LatencyItem] = []
    sender_items: list[LatencyItem] = []
    for channel_name, channel in channels.items():
        item = _latency_item(ref, str(channel_name), getattr(channel, "timestamp", None))
        if item is not None:
            channel_items.append(item)
        metadata = getattr(channel, "metadata", {}) or {}
        if not isinstance(metadata, Mapping):
            continue
        receiver_item = _latency_item(
            ref,
            str(channel_name),
            metadata.get("receiver_timestamp"),
        )
        if receiver_item is not None:
            receiver_items.append(receiver_item)
        sender_item = _latency_item(
            ref,
            str(channel_name),
            metadata.get("sender_timestamp"),
        )
        if sender_item is not None:
            sender_items.append(sender_item)

    return EndToEndLatencyBreakdown(
        reference_time=ref,
        step_age_sec=_timestamp_age(ref, getattr(result, "timestamp", None)),
        observation_age_sec=_timestamp_age(ref, getattr(observation, "timestamp", None)),
        command_age_sec=_timestamp_age(ref, getattr(command, "timestamp", None)),
        channel_age_sec=tuple(channel_items),
        receiver_age_sec=tuple(receiver_items),
        sender_age_sec=tuple(sender_items),
    )


def _format_age(age_sec: float | None) -> str:
    return "none" if age_sec is None else f"{float(age_sec) * 1000.0:.1f}ms"


def _format_latency_items(
    label: str,
    items: Sequence[LatencyItem],
    *,
    limit: int,
) -> str:
    if not items:
        return f"{label}=none"
    ordered = sorted(items, key=lambda item: item.age_sec, reverse=True)
    return f"{label}=" + ",".join(
        f"{item.name}:{item.age_sec * 1000.0:.1f}ms"
        for item in ordered[: max(1, int(limit))]
    )


def format_end_to_end_latency_breakdown(
    breakdown: EndToEndLatencyBreakdown,
    *,
    limit: int = 6,
) -> str:
    return (
        "latency="
        f"step_age:{_format_age(breakdown.step_age_sec)} "
        f"obs_age:{_format_age(breakdown.observation_age_sec)} "
        f"cmd_age:{_format_age(breakdown.command_age_sec)} "
        + _format_latency_items("channel_age", breakdown.channel_age_sec, limit=limit)
        + " "
        + _format_latency_items("receiver_age", breakdown.receiver_age_sec, limit=limit)
        + " "
        + _format_latency_items("sender_age", breakdown.sender_age_sec, limit=limit)
    )
