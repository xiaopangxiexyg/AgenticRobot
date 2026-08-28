"""Deterministic bounded BiRRT-Connect over complete joint configurations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import (
    JointConfigurationCheck,
    JointConfigurationOracle,
    JointPathPlanResult,
    JointPathPlannerConfig,
    JointPathStatus,
)


@dataclass
class _Tree:
    nodes: list[np.ndarray]
    parents: list[int]

    @classmethod
    def rooted_at(cls, value: np.ndarray) -> "_Tree":
        return cls([np.asarray(value, dtype=float).copy()], [-1])

    def nearest(self, value: np.ndarray) -> int:
        distances = [float(np.dot(node - value, node - value)) for node in self.nodes]
        return int(np.argmin(distances))

    def add(self, value: np.ndarray, parent: int) -> int:
        self.nodes.append(np.asarray(value, dtype=float).copy())
        self.parents.append(int(parent))
        return len(self.nodes) - 1

    def root_to(self, index: int) -> list[np.ndarray]:
        path: list[np.ndarray] = []
        cursor = int(index)
        while cursor >= 0:
            path.append(self.nodes[cursor].copy())
            cursor = self.parents[cursor]
        path.reverse()
        return path


@dataclass(frozen=True)
class _EdgeResult:
    safe: bool
    available: bool
    rejection: Mapping[str, Any] | None = None


def _validated_mapping(value: Mapping[str, Any]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for component, raw in value.items():
        key = str(component)
        array = np.asarray(raw, dtype=float)
        if not key or key.strip() != key or array.ndim != 1 or array.size < 1:
            raise ValueError("qpos mappings require non-empty named 1-D components.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"qpos component {key!r} contains non-finite values.")
        result[key] = array.copy()
    if not result:
        raise ValueError("qpos mapping must not be empty.")
    return result


def _copy_mapping(value: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.asarray(array, dtype=float).copy() for key, array in value.items()}


def plan_collision_aware_joint_path(
    *,
    start_qpos_by_component: Mapping[str, Any],
    target_qpos_by_component: Mapping[str, Any],
    movable_components: Sequence[str],
    oracle: JointConfigurationOracle,
    config: JointPathPlannerConfig | None = None,
    joint_lower_by_component: Mapping[str, Any] | None = None,
    joint_upper_by_component: Mapping[str, Any] | None = None,
    scene_hash: str | None = None,
) -> JointPathPlanResult:
    """Plan a collision-checked path without authorizing its execution.

    Only dimensions owned by ``movable_components`` are sampled.  Every other
    component is reconstructed from the exact start configuration before every
    oracle call and in every returned waypoint.
    """

    cfg = config if config is not None else JointPathPlannerConfig()
    try:
        start = _validated_mapping(start_qpos_by_component)
        target = _validated_mapping(target_qpos_by_component)
    except (TypeError, ValueError) as exc:
        return _invalid_result(
            start_qpos_by_component,
            target_qpos_by_component,
            movable_components,
            scene_hash,
            str(exc),
        )
    movable = tuple(str(item) for item in movable_components)
    invalid_reason = _validate_request(start, target, movable)
    if invalid_reason is not None:
        fallback_target = target
        if set(start) != set(target) or any(
            start[key].shape != target[key].shape for key in set(start).intersection(target)
        ):
            fallback_target = _copy_mapping(start)
        fallback_movable = movable
        if (
            not movable
            or len(set(movable)) != len(movable)
            or any(not item or item.strip() != item or item not in start for item in movable)
        ):
            fallback_movable = tuple(start)[:1]
        return _result(
            JointPathStatus.INVALID_REQUEST,
            start,
            fallback_target,
            fallback_movable,
            scene_hash,
            invalid_reason,
        )
    target = {component: target[component] for component in start}

    order = tuple(start)
    shapes = {key: start[key].shape for key in order}
    offsets: dict[str, slice] = {}
    cursor = 0
    for component in movable:
        width = int(start[component].size)
        offsets[component] = slice(cursor, cursor + width)
        cursor += width
    start_flat = np.concatenate([start[item] for item in movable])
    target_flat = np.concatenate([target[item] for item in movable])

    def expand(flat: np.ndarray) -> dict[str, np.ndarray]:
        complete = _copy_mapping(start)
        for component in movable:
            complete[component] = np.asarray(flat[offsets[component]], dtype=float).reshape(shapes[component]).copy()
        return complete

    def check(flat: np.ndarray) -> JointConfigurationCheck:
        try:
            answer = oracle.check(expand(flat))
        except Exception as exc:  # A collision backend failure must fail closed.
            return JointConfigurationCheck(
                safe=False,
                collision_check_available=False,
                reason="collision_oracle_exception",
                metadata={"exception_type": type(exc).__name__, "exception": str(exc)},
            )
        if not isinstance(answer, JointConfigurationCheck):
            return JointConfigurationCheck(
                safe=False,
                collision_check_available=False,
                reason="collision_oracle_returned_invalid_type",
                metadata={"returned_type": type(answer).__name__},
            )
        return answer

    start_check = check(start_flat)
    base_metadata = {
        "planner": "deterministic_bounded_birrt_connect",
        "search_is_complete": False,
        "search_seed": cfg.seed,
        "search_max_iterations": cfg.max_iterations,
        "search_max_nodes": cfg.max_nodes,
        "extension_step_rad": cfg.extension_step_rad,
        "edge_resolution_rad": cfg.edge_resolution_rad,
        "shortcut_attempts": cfg.shortcut_attempts,
        "collision_scope": "complete_configuration",
        "inactive_components_frozen": True,
    }
    if not start_check.collision_check_available:
        return _result(JointPathStatus.COLLISION_CHECK_UNAVAILABLE, start, target, movable, scene_hash, "start_collision_check_unavailable", {**base_metadata, "start_check": start_check.to_dict()})
    if not start_check.safe:
        return _result(JointPathStatus.START_BLOCKED, start, target, movable, scene_hash, "start_configuration_blocked", {**base_metadata, "start_check": start_check.to_dict()})
    target_check = check(target_flat)
    if not target_check.collision_check_available:
        return _result(JointPathStatus.COLLISION_CHECK_UNAVAILABLE, start, target, movable, scene_hash, "target_collision_check_unavailable", {**base_metadata, "start_check": start_check.to_dict(), "target_check": target_check.to_dict()})
    if not target_check.safe:
        return _result(JointPathStatus.TARGET_BLOCKED, start, target, movable, scene_hash, "target_configuration_blocked", {**base_metadata, "start_check": start_check.to_dict(), "target_check": target_check.to_dict()})

    def edge_safe(a: np.ndarray, b: np.ndarray) -> _EdgeResult:
        delta = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
        count = max(1, int(np.ceil(float(np.max(np.abs(delta))) / cfg.edge_resolution_rad)))
        for sample_index in range(1, count + 1):
            alpha = float(sample_index) / float(count)
            candidate = np.asarray(a, dtype=float) + alpha * delta
            assessment = check(candidate)
            if not assessment.collision_check_available:
                return _EdgeResult(False, False, {"edge_sample_index": sample_index, "edge_sample_count": count, "edge_alpha": alpha, "check": assessment.to_dict()})
            if not assessment.safe:
                return _EdgeResult(False, True, {"edge_sample_index": sample_index, "edge_sample_count": count, "edge_alpha": alpha, "check": assessment.to_dict()})
        return _EdgeResult(True, True)

    if np.array_equal(start_flat, target_flat):
        return _result(JointPathStatus.PATH_FOUND, start, target, movable, scene_hash, "start_equals_target", {**base_metadata, "direct_path": True, "search_iterations": 0, "search_node_count": 1}, (expand(start_flat),))
    direct = edge_safe(start_flat, target_flat)
    if not direct.available:
        return _result(JointPathStatus.COLLISION_CHECK_UNAVAILABLE, start, target, movable, scene_hash, "direct_edge_collision_check_unavailable", {**base_metadata, "last_rejected_edge": direct.rejection})
    if direct.safe:
        return _result(JointPathStatus.PATH_FOUND, start, target, movable, scene_hash, "direct_path_clear", {**base_metadata, "direct_path": True, "search_iterations": 0, "search_node_count": 2}, (expand(start_flat), expand(target_flat)))

    try:
        lower, upper = _search_bounds(start, target, movable, cfg, joint_lower_by_component, joint_upper_by_component)
    except (TypeError, ValueError) as exc:
        return _result(JointPathStatus.INVALID_REQUEST, start, target, movable, scene_hash, str(exc), base_metadata)
    rng = np.random.default_rng(cfg.seed)
    start_tree = _Tree.rooted_at(start_flat)
    target_tree = _Tree.rooted_at(target_flat)
    last_rejection = direct.rejection
    unavailable_rejection: Mapping[str, Any] | None = None

    def extend(tree: _Tree, destination: np.ndarray) -> tuple[str, int | None, Mapping[str, Any] | None, bool]:
        nearest_index = tree.nearest(destination)
        nearest = tree.nodes[nearest_index]
        delta = destination - nearest
        maximum = float(np.max(np.abs(delta)))
        if maximum <= 1e-12:
            return "reached", nearest_index, None, True
        scale = min(1.0, cfg.extension_step_rad / maximum)
        candidate = nearest + scale * delta
        edge = edge_safe(nearest, candidate)
        if not edge.available:
            return "trapped", None, edge.rejection, False
        if not edge.safe:
            return "trapped", None, edge.rejection, True
        index = tree.add(candidate, nearest_index)
        return ("reached" if scale >= 1.0 - 1e-12 else "advanced"), index, None, True

    def connect(tree: _Tree, destination: np.ndarray) -> tuple[bool, int | None, Mapping[str, Any] | None, bool]:
        while len(start_tree.nodes) + len(target_tree.nodes) < cfg.max_nodes:
            status, index, rejection, available = extend(tree, destination)
            if status == "trapped":
                return False, None, rejection, available
            if status == "reached":
                return True, index, None, True
        return False, None, None, True

    found: list[np.ndarray] | None = None
    iterations = 0
    for iteration in range(1, cfg.max_iterations + 1):
        iterations = iteration
        if len(start_tree.nodes) + len(target_tree.nodes) >= cfg.max_nodes:
            break
        grow_start = iteration % 2 == 1
        active = start_tree if grow_start else target_tree
        other = target_tree if grow_start else start_tree
        opposite_root = target_flat if grow_start else start_flat
        sample = opposite_root.copy() if rng.random() < cfg.goal_bias else rng.uniform(lower, upper)
        status, active_index, rejection, available = extend(active, sample)
        if rejection is not None:
            last_rejection = rejection
        if not available:
            unavailable_rejection = rejection
            break
        if status == "trapped" or active_index is None:
            continue
        connected, other_index, rejection, available = connect(other, active.nodes[active_index])
        if rejection is not None:
            last_rejection = rejection
        if not available:
            unavailable_rejection = rejection
            break
        if not connected or other_index is None:
            continue
        if grow_start:
            from_start = start_tree.root_to(active_index)
            from_target = target_tree.root_to(other_index)
        else:
            from_start = start_tree.root_to(other_index)
            from_target = target_tree.root_to(active_index)
        found = from_start + list(reversed(from_target))[1:]
        break

    search_metadata = {
        **base_metadata,
        "direct_path": False,
        "search_iterations": iterations,
        "search_node_count": len(start_tree.nodes) + len(target_tree.nodes),
        "last_rejected_edge": last_rejection,
    }
    if unavailable_rejection is not None:
        return _result(JointPathStatus.COLLISION_CHECK_UNAVAILABLE, start, target, movable, scene_hash, "search_collision_check_unavailable", search_metadata)
    if found is None:
        return _result(JointPathStatus.PATH_SEARCH_EXHAUSTED, start, target, movable, scene_hash, "bounded_search_exhausted", search_metadata)

    shortened, shortcut_unavailable = _shortcut_path(found, edge_safe, rng, cfg.shortcut_attempts)
    if shortcut_unavailable is not None:
        return _result(JointPathStatus.COLLISION_CHECK_UNAVAILABLE, start, target, movable, scene_hash, "shortcut_collision_check_unavailable", {**search_metadata, "last_rejected_edge": shortcut_unavailable})
    final_rejection = None
    for left, right in zip(shortened, shortened[1:]):
        validation = edge_safe(left, right)
        if not validation.available:
            return _result(JointPathStatus.COLLISION_CHECK_UNAVAILABLE, start, target, movable, scene_hash, "final_path_collision_check_unavailable", {**search_metadata, "last_rejected_edge": validation.rejection})
        if not validation.safe:
            final_rejection = validation.rejection
            break
    if final_rejection is not None:
        return _result(JointPathStatus.PATH_SEARCH_EXHAUSTED, start, target, movable, scene_hash, "final_path_revalidation_failed", {**search_metadata, "last_rejected_edge": final_rejection})
    return _result(
        JointPathStatus.PATH_FOUND,
        start,
        target,
        movable,
        scene_hash,
        "bounded_search_found_path",
        {**search_metadata, "shortcut_revalidated": True, "unshortened_waypoint_count": len(found), "waypoint_count": len(shortened)},
        tuple(expand(item) for item in shortened),
    )


def _validate_request(start: Mapping[str, np.ndarray], target: Mapping[str, np.ndarray], movable: tuple[str, ...]) -> str | None:
    if set(start) != set(target):
        return "start and target component names must match exactly"
    if any(start[key].shape != target[key].shape for key in start):
        return "start and target component shapes must match"
    if not movable or len(set(movable)) != len(movable):
        return "movable_components must be non-empty and unique"
    if any(item not in start for item in movable):
        return "movable_components must be present in start and target"
    for component in start:
        if component not in movable and not np.array_equal(start[component], target[component]):
            return f"inactive component {component!r} differs between start and target"
    return None


def _search_bounds(
    start: Mapping[str, np.ndarray],
    target: Mapping[str, np.ndarray],
    movable: tuple[str, ...],
    config: JointPathPlannerConfig,
    lower_values: Mapping[str, Any] | None,
    upper_values: Mapping[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if (lower_values is None) != (upper_values is None):
        raise ValueError("joint lower and upper bounds must be supplied together")
    if lower_values is None:
        start_flat = np.concatenate([start[item] for item in movable])
        target_flat = np.concatenate([target[item] for item in movable])
        return (
            np.minimum(start_flat, target_flat) - config.sample_padding_rad,
            np.maximum(start_flat, target_flat) + config.sample_padding_rad,
        )
    lower_parts: list[np.ndarray] = []
    upper_parts: list[np.ndarray] = []
    for component in movable:
        if component not in lower_values or component not in upper_values:
            raise ValueError(f"missing joint bounds for movable component {component!r}")
        lower = np.asarray(lower_values[component], dtype=float)
        upper = np.asarray(upper_values[component], dtype=float)
        if lower.shape != start[component].shape or upper.shape != start[component].shape:
            raise ValueError(f"joint bounds for {component!r} have the wrong shape")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)) or np.any(lower >= upper):
            raise ValueError(f"joint bounds for {component!r} are invalid")
        if np.any(start[component] < lower) or np.any(start[component] > upper) or np.any(target[component] < lower) or np.any(target[component] > upper):
            raise ValueError(f"start or target for {component!r} lies outside joint bounds")
        lower_parts.append(lower)
        upper_parts.append(upper)
    return np.concatenate(lower_parts), np.concatenate(upper_parts)


def _shortcut_path(path: list[np.ndarray], edge_safe: Any, rng: np.random.Generator, attempts: int) -> tuple[list[np.ndarray], Mapping[str, Any] | None]:
    shortened = [np.asarray(item, dtype=float).copy() for item in path]
    for _ in range(attempts):
        if len(shortened) < 3:
            break
        left, right = sorted(int(item) for item in rng.choice(len(shortened), size=2, replace=False))
        if right <= left + 1:
            continue
        edge = edge_safe(shortened[left], shortened[right])
        if not edge.available:
            return shortened, edge.rejection
        if edge.safe:
            shortened = shortened[: left + 1] + shortened[right:]
    return shortened, None


def _result(
    status: JointPathStatus,
    start: Mapping[str, Any],
    target: Mapping[str, Any],
    movable: Sequence[str],
    scene_hash: str | None,
    reason: str,
    metadata: Mapping[str, Any] | None = None,
    waypoints: tuple[Mapping[str, Any], ...] = (),
) -> JointPathPlanResult:
    return JointPathPlanResult(
        status=status,
        start_qpos_by_component=start,
        target_qpos_by_component=target,
        movable_components=tuple(movable),
        waypoints=waypoints,
        scene_hash=scene_hash,
        reason=reason,
        metadata={} if metadata is None else metadata,
    )


def _invalid_result(start: Mapping[str, Any], target: Mapping[str, Any], movable: Sequence[str], scene_hash: str | None, reason: str) -> JointPathPlanResult:
    fallback_start = _safe_fallback_mapping(start)
    fallback_target = _safe_fallback_mapping(target, fallback=fallback_start)
    fallback_movable = tuple(str(item) for item in movable if str(item) in fallback_start)
    if not fallback_movable:
        fallback_movable = (next(iter(fallback_start)),)
    return _result(JointPathStatus.INVALID_REQUEST, fallback_start, fallback_target, fallback_movable, scene_hash, reason)


def _safe_fallback_mapping(value: Mapping[str, Any], *, fallback: Mapping[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    try:
        return _validated_mapping(value)
    except Exception:
        if fallback is not None:
            return _copy_mapping(fallback)
        return {"invalid_request": np.zeros(1, dtype=float)}


__all__ = ["plan_collision_aware_joint_path"]
