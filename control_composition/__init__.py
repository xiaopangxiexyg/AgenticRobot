"""Backend-neutral multi-source control composition.

This package owns composition semantics and orchestration seams.  It does not
own canonical motion authority, safety, execution, a model worker, a replay
cursor, a device lifecycle, or an episode writer.
"""

from __future__ import annotations

from .ports import (
    CompositionClock,
    CompositionProposalAdmitter,
    CompositionTraceEmitter,
)
from .retention import TimeWindowRetentionSpec, TimeWindowStoreSnapshot
__all__ = [
    "CompositionClock",
    "CompositionProposalAdmitter",
    "CompositionTraceEmitter",
    "TimeWindowRetentionSpec",
    "TimeWindowStoreSnapshot",
]
