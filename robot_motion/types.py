from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class IKResolveStatus(str, Enum):
    PASS = "pass"
    RELAXED = "relaxed"
    SMOOTHING_REQUIRED = "smoothing_required"
    FAILED = "failed"


@dataclass(frozen=True)
class PoseTolerance:
    position_m: float = 0.0
    orientation_deg: float = 0.0

    def scaled(self, fraction: float) -> "PoseTolerance":
        factor = max(0.0, min(1.0, float(fraction)))
        return PoseTolerance(
            position_m=float(self.position_m) * factor,
            orientation_deg=float(self.orientation_deg) * factor,
        )


@dataclass(frozen=True)
class SingularityReport:
    status_code: int | None
    label: str = "unknown"
    metric: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_singular(self) -> bool:
        return self.status_code not in (0, None)


@dataclass(frozen=True)
class IKCandidate:
    qpos: np.ndarray
    source: str
    seed_source: str = ""
    method: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IKResolveRequest:
    target_pose: np.ndarray
    reference_qpos: np.ndarray
    reference_pose: np.ndarray | None = None
    seed_qpos: tuple[np.ndarray, ...] = ()
    seed_labels: tuple[str, ...] = ()
    previous_safe_qpos: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IKResolveResult:
    status: IKResolveStatus
    qpos: np.ndarray | None
    requested_target_pose: np.ndarray
    effective_target_pose: np.ndarray
    tolerance_fraction: float
    pose_error_m: float | None = None
    pose_error_deg: float | None = None
    candidate_source: str = ""
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.status in {IKResolveStatus.PASS, IKResolveStatus.RELAXED}
