"""Adapter for the canonical coupled RM65 pose-to-qpos planner.

The IK, collision, joint-limit, and hold policy lives in
``robot_motion.rm65.RM65PositionCollisionPlanner``. This module only supplies
the complete MuJoCo arm state and converts the planner result into a small
controller-facing value object. It never uses MuJoCo Jacobians and never
mutates MuJoCo qpos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from robot_motion.rm65 import (
    RM65PositionCollisionPlanner,
    RM65PositionCollisionPlannerConfig,
    canonical_rm65_controlled_frame,
    endpoint_to_tool_transform,
    load_rm65_tool_profile,
)
from robot_motion.rm65.constants import TOOL_TCP_FRAME

from .scene import MujocoScene
from .transforms import rotation_error_world


def _plain_metadata(value: Any) -> Any:
    """Copy planner audit values into JSON-friendly controller metadata."""

    if isinstance(value, Mapping):
        return {str(key): _plain_metadata(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain_metadata(item) for item in value]
    return value


@dataclass(frozen=True)
class PoseTarget:
    """One canonical target expressed in that arm's base frame."""

    side: str
    pose: np.ndarray
    controlled_frame: str = TOOL_TCP_FRAME

    def __post_init__(self) -> None:
        side = str(self.side).strip().lower()
        if side not in {"left", "right"}:
            raise ValueError("PoseTarget.side must be 'left' or 'right'.")
        pose = np.asarray(self.pose, dtype=float).reshape(4, 4)
        if not np.all(np.isfinite(pose)):
            raise ValueError("PoseTarget.pose must contain finite values.")
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "pose", pose)
        object.__setattr__(
            self,
            "controlled_frame",
            canonical_rm65_controlled_frame(self.controlled_frame),
        )


@dataclass(frozen=True)
class IKResult:
    success: bool
    iterations: int
    qpos_by_side: dict[str, list[float]]
    position_error_m: dict[str, float] = field(default_factory=dict)
    orientation_error_rad: dict[str, float] = field(default_factory=dict)
    controlled_frame_by_side: dict[str, str] = field(default_factory=dict)
    message: str = ""
    status: str = ""
    hold: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    solver: str = "robot_motion.rm65.position_collision"


class RM65CanonicalIK:
    """Group-level adapter around the sim/real-neutral coupled RM65 planner.

    The planner owns both arms, so this adapter must sit above per-effector
    backends. A one-sided target is still checked in the complete dual-arm
    model while the passive arm remains at the last safe qpos.
    """

    def __init__(self, scene: MujocoScene, planner: RM65PositionCollisionPlanner):
        missing = sorted({"left", "right"} - set(scene.arm_specs))
        if missing:
            raise ValueError(
                "Canonical coupled RM65 IK requires a complete dual-arm MuJoCo scene; "
                f"missing={missing}."
            )
        self.scene = scene
        self.planner = planner
        self.reset_from_scene()

    @classmethod
    def from_tool_profiles(
        cls,
        scene: MujocoScene,
        tool_profile_by_side: Mapping[str, str | Path],
        *,
        planner_config: RM65PositionCollisionPlannerConfig | None = None,
    ) -> "RM65CanonicalIK":
        """Build the canonical planner from explicit, side-specific TCP profiles.

        No default profile is selected here. A silent identity/hand-base guess
        would change ``tool_tcp`` semantics between sim and real.
        """

        profiles = {
            str(side).strip().lower(): value
            for side, value in tool_profile_by_side.items()
        }
        missing = sorted({"left", "right"} - set(profiles))
        extra = sorted(set(profiles) - {"left", "right"})
        if missing or extra:
            raise ValueError(
                "Canonical coupled RM65 IK requires explicit left/right tool profiles; "
                f"missing={missing}, extra={extra}."
            )
        endpoint_to_tool_by_side = {
            side: endpoint_to_tool_transform(
                load_rm65_tool_profile(profiles[side], side=side)
            )
            for side in ("left", "right")
        }
        planner = RM65PositionCollisionPlanner(
            config=(
                planner_config
                if planner_config is not None
                else RM65PositionCollisionPlannerConfig()
            ),
            endpoint_to_tool_by_side=endpoint_to_tool_by_side,
        )
        return cls(scene, planner)

    def reset_from_scene(self) -> Mapping[str, Any]:
        """Reset the planner from the complete authoritative MuJoCo qpos."""

        return self.reset_from_qpos(
            {
                side: self.scene.get_arm_qpos(side).copy()
                for side in ("left", "right")
            }
        )

    def reset_from_qpos(
        self,
        qpos_by_side: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Reset the planner from an explicit complete dual-arm qpos snapshot.

        The MuJoCo pose adapter uses this seam to pass a numerically sanitized
        snapshot when the simulator's soft joint-limit solver leaves a tiny
        residual outside the canonical limit.  The MuJoCo scene itself is not
        mutated and the shared RM65 planner limits are unchanged.
        """

        return self.planner.reset(
            {
                side: np.asarray(qpos_by_side[side], dtype=float).reshape(6).copy()
                for side in ("left", "right")
            }
        )

    def solve(self, targets: Mapping[str, PoseTarget]) -> IKResult:
        """Plan one coupled reference without changing the MuJoCo scene."""

        if not targets:
            return IKResult(
                success=True,
                iterations=0,
                qpos_by_side=self.scene.arm_qpos_by_side(),
                message="no targets",
                status="no_targets",
            )

        normalized: dict[str, PoseTarget] = {}
        for side, target in targets.items():
            normalized_side = str(side).strip().lower()
            if normalized_side != target.side:
                raise ValueError(
                    f"Pose target key {side!r} does not match target.side={target.side!r}."
                )
            normalized[normalized_side] = target
        controlled_frames = {target.controlled_frame for target in normalized.values()}
        if len(controlled_frames) != 1:
            raise ValueError(
                "One coupled RM65 planning call requires the same controlled frame for both arms."
            )
        controlled_frame = next(iter(controlled_frames))

        plan = self.planner.plan_pose_reference(
            {side: target.pose for side, target in normalized.items()},
            target_reference_frame="arm_base",
            target_controlled_frame=controlled_frame,
            source="mujoco_canonical_pose_reference",
            metadata={
                "sim_backend": "mujoco",
                "qpos_sink": "scene.set_qpos_by_side",
            },
        )
        qpos_by_side = {
            side: [float(value) for value in np.asarray(plan.q_ref_by_side[side]).reshape(6)]
            for side in ("left", "right")
        }
        position_error_m: dict[str, float] = {}
        orientation_error_rad: dict[str, float] = {}
        for side in normalized:
            target_endpoint = plan.target_endpoint_pose_world_by_side.get(side)
            realized_endpoint = plan.realized_endpoint_pose_world_by_side.get(side)
            if target_endpoint is None or realized_endpoint is None:
                continue
            target_pose = np.asarray(target_endpoint, dtype=float).reshape(4, 4)
            realized_pose = np.asarray(realized_endpoint, dtype=float).reshape(4, 4)
            position_error_m[side] = float(
                np.linalg.norm(target_pose[:3, 3] - realized_pose[:3, 3])
            )
            orientation_error_rad[side] = float(
                np.linalg.norm(
                    rotation_error_world(realized_pose[:3, :3], target_pose[:3, :3])
                )
            )

        valid = bool(getattr(plan, "valid", False))
        hold = bool(getattr(plan, "hold", True))
        reasons = tuple(str(value) for value in getattr(plan, "modification_reasons", ()))
        status = str(getattr(plan, "status", "unknown"))
        return IKResult(
            success=bool(valid and not hold),
            iterations=1,
            qpos_by_side=qpos_by_side,
            position_error_m=position_error_m,
            orientation_error_rad=orientation_error_rad,
            controlled_frame_by_side={side: controlled_frame for side in normalized},
            message=status if not reasons else f"{status}:{','.join(reasons)}",
            status=status,
            hold=hold,
            metadata=_plain_metadata(getattr(plan, "metadata", {}) or {}),
        )


__all__ = ["IKResult", "PoseTarget", "RM65CanonicalIK"]
