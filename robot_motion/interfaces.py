from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from .types import IKCandidate, SingularityReport


class KinematicsProvider(Protocol):
    """Robot-specific kinematics adapter.

    All public data crossing this interface uses SI units:
    - joint positions are radians
    - poses are 4x4 homogeneous matrices
    """

    def iter_ik_candidates(
        self,
        *,
        target_pose: np.ndarray,
        seeds: tuple[np.ndarray, ...],
        seed_labels: tuple[str, ...],
        options: dict[str, Any],
    ) -> list[IKCandidate]:
        ...

    def forward_kinematics(self, qpos: np.ndarray) -> np.ndarray | None:
        ...

    def analyze_singularity(self, qpos: np.ndarray) -> SingularityReport:
        ...

    def joint_position_limits(self) -> tuple[np.ndarray, np.ndarray] | None:
        ...


class CollisionChecker(Protocol):
    def check_candidate(self, qpos: np.ndarray, *, metadata: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        ...
