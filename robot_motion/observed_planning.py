"""A measured robot cut and an already-bound observed-scene query.

The caller owns acquisition and the collision implementation. This value adds
no simulator, execution owner, cache or implicit observation refresh.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from robot_motion.path_planning.contracts import JointConfigurationOracle


@dataclass(frozen=True)
class ObservedJointPlanningInput:
    oracle: JointConfigurationOracle
    qpos_by_component: Mapping[str, np.ndarray]
    acquired_at: float
    maximum_age_sec: float
    maximum_start_drift_rad: float
    clock: Callable[[], float]

    def __post_init__(self):
        if not callable(getattr(self.oracle, "check", None)):
            raise TypeError("Observed planning requires a bound collision query.")
        if not isinstance(getattr(self.oracle, "scene_hash", None), str) or not self.oracle.scene_hash:
            raise ValueError("Observed planning requires an identified scene.")
        if not callable(self.clock):
            raise TypeError("Observed planning requires the acquisition clock.")
        for name in ("acquired_at", "maximum_age_sec", "maximum_start_drift_rad"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            if name != "acquired_at" and value <= 0:
                raise ValueError(f"{name} must be positive.")
        if set(self.qpos_by_component) != {"left", "right"}:
            raise ValueError("Observed planning requires measured dual-arm qpos.")
        pair = {}
        for side, values in self.qpos_by_component.items():
            q = np.asarray(values, dtype=float)
            if q.shape != (6,) or not np.isfinite(q).all():
                raise ValueError("Observed planning requires six finite joints per arm.")
            # An immutable byte backing prevents callers from re-enabling writes.
            pair[side] = np.frombuffer(q.tobytes(), dtype=float)
        object.__setattr__(self, "qpos_by_component", MappingProxyType(pair))

    def require_current(self, measured_qpos: Mapping[str, np.ndarray]) -> None:
        now = self.clock()
        if not np.isfinite(now) or not 0 <= now - self.acquired_at <= self.maximum_age_sec:
            raise ValueError("observed_planning_scene_stale")
        if set(measured_qpos) != set(self.qpos_by_component):
            raise ValueError("observed_planning_start_changed")
        for side, start in self.qpos_by_component.items():
            q = np.asarray(measured_qpos[side], dtype=float)
            if (q.shape != (6,) or not np.isfinite(q).all()
                    or np.max(np.abs(q - start)) > self.maximum_start_drift_rad):
                raise ValueError("observed_planning_start_changed")
