"""Lifecycle boundary between runtime traces and externally owned episodes.

This module intentionally has no begin/finalize/abort/commit methods.  A GUI,
headless collector, replay frontend, or simulator remains the sole dataset
episode lifecycle owner and may only bind its current episode membership to the
subordinate ControlTrace runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from dataset_contracts.control_trace import EpisodeMembershipRef


def _required(value: object, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


@dataclass(frozen=True)
class RuntimeTraceSession:
    trace_session_id: str
    runtime_session_id: str
    control_domain_id: str
    profile_id: str
    started_at: float
    clock_domain_id: str

    def __post_init__(self) -> None:
        for name in (
            "trace_session_id",
            "runtime_session_id",
            "control_domain_id",
            "profile_id",
            "clock_domain_id",
        ):
            object.__setattr__(self, name, _required(getattr(self, name), field_name=name))
        timestamp = float(self.started_at)
        if not math.isfinite(timestamp):
            raise ValueError("started_at must be finite.")
        object.__setattr__(self, "started_at", timestamp)


EpisodeTraceMembership = EpisodeMembershipRef


@dataclass(frozen=True)
class EpisodeTraceMembershipRelease:
    """Unbind assertion supplied by the same external lifecycle owner."""

    episode_uuid: str
    lifecycle_owner_id: str
    unbinding_event_id: str
    membership_epoch: int
    unbound_at: float
    reason: str

    def __post_init__(self) -> None:
        for name in (
            "episode_uuid",
            "lifecycle_owner_id",
            "unbinding_event_id",
            "reason",
        ):
            object.__setattr__(self, name, _required(getattr(self, name), field_name=name))
        if int(self.membership_epoch) < 0:
            raise ValueError("membership_epoch must be non-negative.")
        object.__setattr__(self, "membership_epoch", int(self.membership_epoch))
        timestamp = float(self.unbound_at)
        if not math.isfinite(timestamp):
            raise ValueError("unbound_at must be finite.")
        object.__setattr__(self, "unbound_at", timestamp)


__all__ = [
    "EpisodeTraceMembership",
    "EpisodeTraceMembershipRelease",
    "RuntimeTraceSession",
]
