"""Backend-neutral contracts for collision-aware joint-path planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Protocol

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash


class JointPathStatus(str, Enum):
    PATH_FOUND = "path_found"
    START_BLOCKED = "start_blocked"
    TARGET_BLOCKED = "target_blocked"
    PATH_SEARCH_EXHAUSTED = "path_search_exhausted"
    COLLISION_CHECK_UNAVAILABLE = "collision_check_unavailable"
    INVALID_REQUEST = "invalid_request"


def _freeze_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_metadata(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_metadata(item) for item in value)
    if isinstance(value, np.ndarray):
        array = np.asarray(value, dtype=float).copy()
        array.setflags(write=False)
        return array
    return value


def _freeze_qpos_mapping(value: Mapping[str, Any]) -> Mapping[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for component, raw in value.items():
        key = str(component)
        if not key or key.strip() != key:
            raise ValueError("component names must be non-empty trimmed strings.")
        array = np.asarray(raw, dtype=float)
        if array.ndim != 1 or array.size < 1 or not np.all(np.isfinite(array)):
            raise ValueError(f"qpos component {key!r} must be a non-empty finite vector.")
        frozen = array.copy()
        frozen.setflags(write=False)
        result[key] = frozen
    if not result:
        raise ValueError("qpos mapping must contain at least one component.")
    return MappingProxyType(result)


def _qpos_payload(value: Mapping[str, np.ndarray]) -> dict[str, list[float]]:
    return {key: [float(item) for item in array] for key, array in value.items()}


@dataclass(frozen=True)
class JointConfigurationCheck:
    """One typed collision-oracle answer.

    ``safe`` may only be true when the collision calculation was available.
    A backend exception or missing geometry is therefore fail-closed rather
    than being interpreted as free space.
    """

    safe: bool
    collision_check_available: bool = True
    reason: str | None = None
    minimum_distance_m: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        safe = bool(self.safe)
        available = bool(self.collision_check_available)
        if safe and not available:
            raise ValueError("safe cannot be true when collision checking is unavailable.")
        distance = self.minimum_distance_m
        if distance is not None:
            distance = float(distance)
            if not np.isfinite(distance):
                raise ValueError("minimum_distance_m must be finite when provided.")
        reason = None if self.reason is None else str(self.reason)
        object.__setattr__(self, "safe", safe)
        object.__setattr__(self, "collision_check_available", available)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "minimum_distance_m", distance)
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "safe": self.safe,
            "collision_check_available": self.collision_check_available,
            "reason": self.reason,
            "minimum_distance_m": self.minimum_distance_m,
            "metadata": dict(self.metadata),
        }


class JointConfigurationOracle(Protocol):
    def check(self, qpos_by_component: Mapping[str, np.ndarray]) -> JointConfigurationCheck:
        """Return a fail-closed collision check for one complete configuration."""


@dataclass(frozen=True)
class JointPathPlannerConfig:
    seed: int = 0
    max_iterations: int = 1200
    max_nodes: int = 2400
    extension_step_rad: float = float(np.deg2rad(12.0))
    edge_resolution_rad: float = float(np.deg2rad(1.0))
    sample_padding_rad: float = float(np.deg2rad(45.0))
    goal_bias: float = 0.2
    shortcut_attempts: int = 80
    adaptive_edge_resolution_rad: float | None = None
    adaptive_refinement_distance_m: float | None = None
    time_budget_sec: float | None = None

    def __post_init__(self) -> None:
        seed = int(self.seed)
        iterations = int(self.max_iterations)
        nodes = int(self.max_nodes)
        shortcuts = int(self.shortcut_attempts)
        if iterations < 1:
            raise ValueError("max_iterations must be >= 1.")
        if nodes < 2:
            raise ValueError("max_nodes must be >= 2.")
        if shortcuts < 0:
            raise ValueError("shortcut_attempts must be non-negative.")
        for name in ("extension_step_rad", "edge_resolution_rad", "sample_padding_rad"):
            number = float(getattr(self, name))
            if not np.isfinite(number) or number <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
            object.__setattr__(self, name, number)
        adaptive_resolution = self.adaptive_edge_resolution_rad
        adaptive_distance = self.adaptive_refinement_distance_m
        if (adaptive_resolution is None) != (adaptive_distance is None):
            raise ValueError(
                "adaptive edge resolution and refinement distance must be supplied together."
            )
        if adaptive_resolution is not None:
            adaptive_resolution = float(adaptive_resolution)
            adaptive_distance = float(adaptive_distance)
            if (
                not np.isfinite(adaptive_resolution)
                or adaptive_resolution <= 0.0
                or adaptive_resolution >= self.edge_resolution_rad
            ):
                raise ValueError(
                    "adaptive_edge_resolution_rad must be positive and finer than edge_resolution_rad."
                )
            if not np.isfinite(adaptive_distance) or adaptive_distance <= 0.0:
                raise ValueError(
                    "adaptive_refinement_distance_m must be positive and finite."
                )
            object.__setattr__(self, "adaptive_edge_resolution_rad", adaptive_resolution)
            object.__setattr__(self, "adaptive_refinement_distance_m", adaptive_distance)
        time_budget = self.time_budget_sec
        if time_budget is not None:
            time_budget = float(time_budget)
            if not np.isfinite(time_budget) or time_budget <= 0.0:
                raise ValueError("time_budget_sec must be positive and finite when provided.")
            object.__setattr__(self, "time_budget_sec", time_budget)
        bias = float(self.goal_bias)
        if not np.isfinite(bias) or not 0.0 <= bias <= 1.0:
            raise ValueError("goal_bias must be within [0, 1].")
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "max_iterations", iterations)
        object.__setattr__(self, "max_nodes", nodes)
        object.__setattr__(self, "shortcut_attempts", shortcuts)
        object.__setattr__(self, "goal_bias", bias)


@dataclass(frozen=True)
class JointPathPlanResult:
    status: JointPathStatus
    start_qpos_by_component: Mapping[str, np.ndarray]
    target_qpos_by_component: Mapping[str, np.ndarray]
    movable_components: tuple[str, ...]
    waypoints: tuple[Mapping[str, np.ndarray], ...] = ()
    scene_hash: str | None = None
    reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    execution_certified: bool = field(init=False, default=False)
    real_motion_authorized: bool = field(init=False, default=False)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        status = self.status if isinstance(self.status, JointPathStatus) else JointPathStatus(self.status)
        start = _freeze_qpos_mapping(self.start_qpos_by_component)
        target = _freeze_qpos_mapping(self.target_qpos_by_component)
        movable = tuple(str(item) for item in self.movable_components)
        if not movable or len(set(movable)) != len(movable):
            raise ValueError("movable_components must be non-empty and unique.")
        if any(not item or item.strip() != item for item in movable):
            raise ValueError("movable_components must contain trimmed names.")
        if set(start) != set(target):
            raise ValueError("start and target component names must match.")
        if any(start[item].shape != target[item].shape for item in start):
            raise ValueError("start and target component shapes must match.")
        if any(item not in start for item in movable):
            raise ValueError("movable_components must be present in the qpos mappings.")
        waypoints = tuple(_freeze_qpos_mapping(item) for item in self.waypoints)
        if status is JointPathStatus.PATH_FOUND and not waypoints:
            raise ValueError("PATH_FOUND requires at least one waypoint.")
        if status is not JointPathStatus.PATH_FOUND and waypoints:
            raise ValueError("failed path results must not carry executable waypoints.")
        if waypoints:
            for waypoint in waypoints:
                if set(waypoint) != set(start):
                    raise ValueError("waypoint component names must match the request.")
                if any(waypoint[item].shape != start[item].shape for item in start):
                    raise ValueError("waypoint component shapes must match the request.")
            if any(not np.array_equal(waypoints[0][item], start[item]) for item in start):
                raise ValueError("the first waypoint must equal the start configuration.")
            if any(not np.array_equal(waypoints[-1][item], target[item]) for item in start):
                raise ValueError("the last waypoint must equal the target configuration.")
        metadata = _freeze_metadata(self.metadata)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "start_qpos_by_component", start)
        object.__setattr__(self, "target_qpos_by_component", target)
        object.__setattr__(self, "movable_components", movable)
        object.__setattr__(self, "waypoints", waypoints)
        object.__setattr__(self, "scene_hash", None if self.scene_hash is None else str(self.scene_hash))
        object.__setattr__(self, "reason", None if self.reason is None else str(self.reason))
        object.__setattr__(self, "metadata", metadata)
        payload = {
            "status": status.value,
            "start_qpos_by_component": _qpos_payload(start),
            "target_qpos_by_component": _qpos_payload(target),
            "movable_components": list(movable),
            "waypoints": [_qpos_payload(item) for item in waypoints],
            "scene_hash": self.scene_hash,
            "reason": self.reason,
            "metadata": dict(metadata),
            "execution_certified": False,
            "real_motion_authorized": False,
        }
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(payload, domain="dexteleop.robot_motion.joint_path_plan_result.v1"),
        )

    @property
    def path_found(self) -> bool:
        return self.status is JointPathStatus.PATH_FOUND

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "start_qpos_by_component": _qpos_payload(self.start_qpos_by_component),
            "target_qpos_by_component": _qpos_payload(self.target_qpos_by_component),
            "movable_components": list(self.movable_components),
            "waypoints": [_qpos_payload(item) for item in self.waypoints],
            "scene_hash": self.scene_hash,
            "reason": self.reason,
            "metadata": dict(self.metadata),
            "execution_certified": False,
            "real_motion_authorized": False,
            "semantic_content_hash": self.semantic_content_hash,
        }


__all__ = [
    "JointConfigurationCheck",
    "JointConfigurationOracle",
    "JointPathPlanResult",
    "JointPathPlannerConfig",
    "JointPathStatus",
]
