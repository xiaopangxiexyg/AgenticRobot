"""Synchronous policy-source handoff to the position executor.

The deployment application owns when a synchronous policy chunk may cross
into the position executor.  This module expresses that handoff through a
narrow structural port: it may inspect tail/active status and request the
executor's existing clear/Hold transition, but it does not own the source
cursor, executor state, Hold implementation, or motion safety.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol


class PositionStreamHandoffPort(Protocol):
    """Minimal executor surface used at a synchronous chunk boundary."""

    def source_tail_drain_status(self) -> Mapping[str, Any]: ...

    def snapshot(self) -> Mapping[str, Any]: ...

    def clear(self) -> None: ...


def defer_position_action_after_synchronous_prediction(
    *,
    prediction_accepted: bool,
    position_executor_enabled: bool,
) -> bool:
    """Keep a newly predicted position chunk for the next observation cut.

    Legacy policy inference is synchronous.  A slow inference therefore ages
    the observation that was used to request it, even though the model output
    itself remains valid.  The first position command must not reuse that
    frozen observation as the authoritative collision-planner reset state.
    Deferring consumption by one control-loop iteration lets the frontend
    acquire a fresh dual-arm measurement without weakening collision-reset
    freshness or changing the queued action chunk.
    """

    return bool(prediction_accepted and position_executor_enabled)


def hold_active_position_stream_for_synchronous_chunk_gap(
    position_executor: PositionStreamHandoffPort | None,
) -> bool:
    """End an exhausted synchronous chunk without faulting its executor.

    The legacy OpenPI client performs inference synchronously.  With a full
    chunk execution policy, the position stream can therefore be active while
    the policy queue has just become empty.  Waiting for inference inside that
    active stream would trip the executor's missing-keyframe watchdog.  Clear
    the stream first so the arms enter the existing bounded Hold path; the next
    chunk must then establish a fresh collision reset before it can move.
    """

    if position_executor is None:
        return False
    tail_status = position_executor.source_tail_drain_status()
    if bool(tail_status.get("required")) and not bool(tail_status.get("drained")):
        return False
    if not bool(position_executor.snapshot().get("active")):
        return False
    position_executor.clear()
    return True


def position_source_tail_pending_before_synchronous_chunk_gap(
    position_executor: PositionStreamHandoffPort | None,
) -> bool:
    """Keep the upper chunk boundary open until motion emitted its tail."""

    if position_executor is None:
        return False
    status = position_executor.source_tail_drain_status()
    return bool(status.get("required")) and not bool(status.get("drained"))


__all__ = [
    "PositionStreamHandoffPort",
    "defer_position_action_after_synchronous_prediction",
    "hold_active_position_stream_for_synchronous_chunk_gap",
    "position_source_tail_pending_before_synchronous_chunk_gap",
]
