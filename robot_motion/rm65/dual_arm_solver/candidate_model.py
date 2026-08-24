"""Private value objects used by the RM65 dual-arm solver facade.

These records describe candidate generation and evaluation only.  They do not
own a solver, live backend, accepted reference, or dispatch authority; the
facade remains the sole solve-cycle and accepted-qref commit owner.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class _DualArmCandidate:
    """One non-authoritative candidate produced during a solve cycle."""

    name: str
    qpos: np.ndarray
    source: str
    method: str
    raw_score_deg: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class _DualArmEvaluatedCandidate:
    """A candidate after the facade's validation and scoring stages."""

    candidate: _DualArmCandidate
    output_qpos: np.ndarray
    score: float
    metadata: dict[str, object]


__all__ = ["_DualArmCandidate", "_DualArmEvaluatedCandidate"]
