"""Pure tool-axis cone geometry for sampled motion-path validation.

This is a predicate, not a planner, motion permit, or continuous-time proof.
The caller owns the common frame and calibrated TCP pose used for both inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.relative_action import xyzquat_to_pose_matrix


@dataclass(frozen=True)
class ToolAxisCone:
    tool_axis: str
    reference_axis: tuple[float, float, float]
    max_deviation_rad: float

    def __post_init__(self) -> None:
        if self.tool_axis not in ("x", "y", "z"):
            raise ValueError("tool_axis must be x, y, or z.")
        if (not isinstance(self.reference_axis, tuple) or len(self.reference_axis) != 3
                or any(isinstance(x, bool) or not isinstance(x, (int, float))
                       or not math.isfinite(x) for x in self.reference_axis)
                or not math.isclose(math.hypot(*self.reference_axis), 1., abs_tol=1e-9)):
            raise ValueError("reference_axis must be a finite unit-vector tuple.")
        if (isinstance(self.max_deviation_rad, bool)
                or not isinstance(self.max_deviation_rad, (int, float))
                or not math.isfinite(self.max_deviation_rad)
                or not 0. < self.max_deviation_rad <= math.pi):
            raise ValueError("max_deviation_rad must satisfy 0 < angle <= pi.")

    @classmethod
    def from_pose(cls, pose: Any, *, tool_axis: str, max_deviation_rad: float) -> ToolAxisCone:
        if tool_axis not in ("x", "y", "z"):
            raise ValueError("tool_axis must be x, y, or z.")
        matrix = xyzquat_to_pose_matrix(pose)
        axis = matrix[:3, ("x", "y", "z").index(tool_axis)]
        return cls(tool_axis, tuple(float(x) for x in axis), max_deviation_rad)

    def deviation_rad(self, pose: Any) -> float:
        actual = xyzquat_to_pose_matrix(pose)[:3, ("x", "y", "z").index(self.tool_axis)]
        reference = np.asarray(self.reference_axis)
        return math.atan2(float(np.linalg.norm(np.cross(actual, reference))),
                          float(np.clip(actual @ reference, -1., 1.)))

    def accepts(self, pose: Any) -> bool:
        return self.deviation_rad(pose) <= self.max_deviation_rad + 1e-9

    def to_payload(self) -> dict[str, Any]:
        return {"tool_axis": self.tool_axis, "reference_axis": list(self.reference_axis),
                "max_deviation_rad": self.max_deviation_rad}

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(self.to_payload(), domain="robot_motion.tool_axis_cone.v1")
