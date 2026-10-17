"""Deterministic bounded BiRRT-Connect over complete joint configurations."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .contracts import (
    JointConfigurationCheck,
    JointConfigurationOracle,
    JointPathPlannerConfig,
    JointPathPlanResult,
    JointPathStatus,
)


@dataclass
class _Tree:
    nodes: list[np.ndarray]
    parents: list[int]

    @classmethod
    def rooted_at(cls, value: np.ndarray) -> "_Tree":
        return cls([np.asarray(value, dtype=float).copy()], [-1])

    @classmethod
    def rooted_at_many(cls, values: Sequence[np.ndarray]) -> "_Tree":
        return cls([np.asarray(value, dtype=float).copy() for value in values], [-1] * len(values))

    def root_index(self, index: int) -> int:
        while self.parents[index] >= 0:
            index = self.parents[index]
        return index

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
    timed_out: bool = False


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
    alternative_target_qpos_by_component: Sequence[Mapping[str, Any]] = (),
    movable_components: Sequence[str],
    oracle: JointConfigurationOracle,
    config: JointPathPlannerConfig | None = None,
    joint_lower_by_component: Mapping[str, Any] | None = None,
    joint_upper_by_component: Mapping[str, Any] | None = None,
    scene_hash: str | None = None,
    edge_cost: Callable[[Mapping[str, np.ndarray], Mapping[str, np.ndarray]], float] | None = None,
    monotonic: Callable[[], float] = time.perf_counter,
) -> JointPathPlanResult:
    """Plan a collision-checked path without authorizing its execution.

    Only dimensions owned by ``movable_components`` are sampled.  Every other
    component is reconstructed from the exact start configuration before every
    oracle call and in every returned waypoint.  ``time_budget_sec`` is a
    cooperative wall-time bound checked before and after collision-oracle
    queries; it cannot preempt one oracle call already in progress.
    Alternative targets share the start tree, collision cache and total search
    budget. The result names the exact selected target; blocked targets may be
    skipped, but invalid requests and unavailable collision checks fail closed.
    """

    cfg = config if config is not None else JointPathPlannerConfig()
    if edge_cost is not None and not callable(edge_cost):
        raise TypeError("edge_cost must be callable or None.")
    deadline = (
        None
        if cfg.time_budget_sec is None
        else float(monotonic()) + cfg.time_budget_sec
    )

    def budget_expired() -> bool:
        return deadline is not None and float(monotonic()) >= deadline

    def timeout_rejection(phase: str) -> dict[str, str]:
        return {
            "reason": "time_budget_exhausted",
            "termination_phase": phase,
        }

    resolved_scene_hash, scene_binding_metadata, scene_binding_error = (
        _resolve_scene_binding(oracle=oracle, requested_scene_hash=scene_hash)
    )
    try:
        start = _validated_mapping(start_qpos_by_component)
        target = _validated_mapping(target_qpos_by_component)
    except (TypeError, ValueError) as exc:
        return _invalid_result(
            start_qpos_by_component,
            target_qpos_by_component,
            movable_components,
            resolved_scene_hash,
            str(exc),
            metadata=scene_binding_metadata,
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
            resolved_scene_hash,
            invalid_reason,
            scene_binding_metadata,
        )
    target = {component: target[component] for component in start}
    if scene_binding_error is not None:
        return _result(
            JointPathStatus.INVALID_REQUEST,
            start,
            target,
            movable,
            resolved_scene_hash,
            scene_binding_error,
            scene_binding_metadata,
        )

    targets = [target]
    try:
        for alternative in alternative_target_qpos_by_component:
            if not isinstance(alternative, Mapping):
                raise ValueError("alternative targets must be complete qpos mappings")
            candidate = _validated_mapping(alternative)
            reason = _validate_request(start, candidate, movable)
            if reason is not None:
                raise ValueError(reason)
            targets.append({component: candidate[component] for component in start})
        if len(targets) + 1 > cfg.max_nodes:
            raise ValueError("target count exceeds the shared search node budget")
        bounds = [
            _search_bounds(start, candidate, movable, cfg,
                           joint_lower_by_component, joint_upper_by_component)
            for candidate in targets
        ]
    except (TypeError, ValueError) as exc:
        return _result(JointPathStatus.INVALID_REQUEST, start, target, movable,
                       resolved_scene_hash, str(exc), scene_binding_metadata)
    lower = np.min([item[0] for item in bounds], axis=0)
    upper = np.max([item[1] for item in bounds], axis=0)

    order = tuple(start)
    shapes = {key: start[key].shape for key in order}
    offsets: dict[str, slice] = {}
    cursor = 0
    for component in movable:
        width = int(start[component].size)
        offsets[component] = slice(cursor, cursor + width)
        cursor += width
    start_flat = np.concatenate([start[item] for item in movable])
    target_flats = [np.concatenate([candidate[item] for item in movable]) for candidate in targets]

    def expand(flat: np.ndarray) -> dict[str, np.ndarray]:
        complete = _copy_mapping(start)
        for component in movable:
            complete[component] = np.asarray(flat[offsets[component]], dtype=float).reshape(shapes[component]).copy()
        return complete

    check_cache: dict[bytes, JointConfigurationCheck] = {}

    def check(flat: np.ndarray) -> JointConfigurationCheck:
        key = np.asarray(flat, dtype=float).tobytes()
        if key in check_cache:
            return check_cache[key]
        try:
            answer = oracle.check(expand(flat))
        except Exception as exc:  # A collision backend failure must fail closed.
            answer = JointConfigurationCheck(
                safe=False,
                collision_check_available=False,
                reason="collision_oracle_exception",
                metadata={"exception_type": type(exc).__name__, "exception": str(exc)},
            )
        if not isinstance(answer, JointConfigurationCheck):
            answer = JointConfigurationCheck(
                safe=False,
                collision_check_available=False,
                reason="collision_oracle_returned_invalid_type",
                metadata={"returned_type": type(answer).__name__},
            )
        check_cache[key] = answer
        return answer

    base_metadata = {
        "planner": "deterministic_bounded_birrt_connect",
        "search_is_complete": False,
        "search_seed": cfg.seed,
        "search_max_iterations": cfg.max_iterations,
        "search_max_nodes": cfg.max_nodes,
        "extension_step_rad": cfg.extension_step_rad,
        "edge_resolution_rad": cfg.edge_resolution_rad,
        "adaptive_edge_resolution_rad": cfg.adaptive_edge_resolution_rad,
        "adaptive_refinement_distance_m": cfg.adaptive_refinement_distance_m,
        "shortcut_attempts": cfg.shortcut_attempts,
        "time_budget_sec": cfg.time_budget_sec,
        "collision_scope": "complete_configuration",
        "inactive_components_frozen": True,
        **scene_binding_metadata,
    }
    if len(targets) > 1:
        base_metadata.update(target_count=len(targets), shared_goal_search=True)

    def timeout_result(
        phase: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> JointPathPlanResult:
        return _result(
            JointPathStatus.PATH_SEARCH_EXHAUSTED,
            start,
            target,
            movable,
            resolved_scene_hash,
            "time_budget_exhausted",
            {
                **base_metadata,
                **({} if metadata is None else metadata),
                "search_termination": "time_budget_exhausted",
                "termination_phase": phase,
            },
        )

    if budget_expired():
        return timeout_result("start_check")
    start_check = check(start_flat)
    if budget_expired():
        return timeout_result("start_check")
    if not start_check.collision_check_available:
        return _result(JointPathStatus.COLLISION_CHECK_UNAVAILABLE, start, target, movable, resolved_scene_hash, "start_collision_check_unavailable", {**base_metadata, "start_check": start_check.to_dict()})
    if not start_check.safe:
        return _result(JointPathStatus.START_BLOCKED, start, target, movable, resolved_scene_hash, "start_configuration_blocked", {**base_metadata, "start_check": start_check.to_dict()})
    target_checks: list[dict[str, Any]] = []
    safe_target_indices: list[int] = []
    for target_index, candidate_flat in enumerate(target_flats):
        if budget_expired():
            return timeout_result("target_check", {"start_check": start_check.to_dict()})
        target_check = check(candidate_flat)
        target_checks.append({"target_index": target_index, "check": target_check.to_dict()})
        if len(targets) > 1:
            base_metadata["target_checks"] = tuple(target_checks)
        if budget_expired():
            return timeout_result("target_check", {"start_check": start_check.to_dict(), "target_check": target_check.to_dict()})
        if not target_check.collision_check_available:
            return _result(JointPathStatus.COLLISION_CHECK_UNAVAILABLE, start, target, movable, resolved_scene_hash, "target_collision_check_unavailable", {**base_metadata, "start_check": start_check.to_dict(), "target_check": target_check.to_dict()})
        if target_check.safe:
            safe_target_indices.append(target_index)
    if not safe_target_indices:
        return _result(JointPathStatus.TARGET_BLOCKED, start, target, movable, resolved_scene_hash, "target_configuration_blocked", {**base_metadata, "start_check": start_check.to_dict(), "target_check": target_check.to_dict()})

    def sample_edge(
        a: np.ndarray,
        b: np.ndarray,
        *,
        resolution_rad: float,
    ) -> tuple[_EdgeResult, float | None]:
        delta = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
        count = max(1, int(np.ceil(float(np.max(np.abs(delta))) / resolution_rad)))
        minimum_distance_m = math.inf
        for sample_index in range(1, count + 1):
            if budget_expired():
                return _EdgeResult(
                    False,
                    True,
                    timeout_rejection("edge_check"),
                    timed_out=True,
                ), None
            alpha = float(sample_index) / float(count)
            candidate = np.asarray(a, dtype=float) + alpha * delta
            assessment = check(candidate)
            if budget_expired():
                return _EdgeResult(
                    False,
                    True,
                    timeout_rejection("edge_check"),
                    timed_out=True,
                ), None
            if not assessment.collision_check_available:
                return _EdgeResult(
                    False,
                    False,
                    {
                        "edge_sample_index": sample_index,
                        "edge_sample_count": count,
                        "edge_alpha": alpha,
                        "check": assessment.to_dict(),
                    },
                ), None
            if not assessment.safe:
                return _EdgeResult(
                    False,
                    True,
                    {
                        "edge_sample_index": sample_index,
                        "edge_sample_count": count,
                        "edge_alpha": alpha,
                        "check": assessment.to_dict(),
                    },
                ), None
            adaptive_distance = assessment.metadata.get(
                "adaptive_edge_distance_m", assessment.minimum_distance_m
            )
            if adaptive_distance is not None:
                try:
                    adaptive_distance = float(adaptive_distance)
                except (TypeError, ValueError):
                    adaptive_distance = math.nan
                if not math.isfinite(adaptive_distance) or adaptive_distance < 0.0:
                    return _EdgeResult(
                        False,
                        False,
                        {
                            "edge_sample_index": sample_index,
                            "edge_sample_count": count,
                            "edge_alpha": alpha,
                            "reason": "invalid_adaptive_edge_distance",
                        },
                    ), None
                minimum_distance_m = min(minimum_distance_m, adaptive_distance)
        return _EdgeResult(True, True), (
            minimum_distance_m if np.isfinite(minimum_distance_m) else None
        )

    def edge_safe(a: np.ndarray, b: np.ndarray) -> _EdgeResult:
        edge, minimum_distance_m = sample_edge(
            a, b, resolution_rad=cfg.edge_resolution_rad
        )
        if not edge.safe or cfg.adaptive_edge_resolution_rad is None:
            return edge
        if (
            minimum_distance_m is None
            or minimum_distance_m > cfg.adaptive_refinement_distance_m
        ):
            return edge
        refined, _ = sample_edge(
            a,
            b,
            resolution_rad=cfg.adaptive_edge_resolution_rad,
        )
        return refined

    last_rejection = None
    for target_index in safe_target_indices:
        candidate_flat = target_flats[target_index]
        selected_metadata = ({} if len(targets) == 1 else {"selected_target_index": target_index})
        if np.array_equal(start_flat, candidate_flat):
            return _result(JointPathStatus.PATH_FOUND, start, targets[target_index], movable, resolved_scene_hash, "start_equals_target", {**base_metadata, **selected_metadata, "direct_path": True, "search_iterations": 0, "search_node_count": 1}, (expand(start_flat),))
        direct = edge_safe(start_flat, candidate_flat)
        last_rejection = direct.rejection
        if direct.timed_out:
            return timeout_result("direct_edge", {"last_rejected_edge": direct.rejection})
        if not direct.available:
            return _result(JointPathStatus.COLLISION_CHECK_UNAVAILABLE, start, target, movable, resolved_scene_hash, "direct_edge_collision_check_unavailable", {**base_metadata, "last_rejected_edge": direct.rejection})
        if direct.safe:
            return _result(JointPathStatus.PATH_FOUND, start, targets[target_index], movable, resolved_scene_hash, "direct_path_clear", {**base_metadata, **selected_metadata, "direct_path": True, "search_iterations": 0, "search_node_count": 2}, (expand(start_flat), expand(candidate_flat)))

    rng = np.random.default_rng(cfg.seed)
    start_tree = _Tree.rooted_at(start_flat)
    goal_roots = [target_flats[index] for index in safe_target_indices]
    target_tree = _Tree.rooted_at_many(goal_roots)
    unavailable_rejection: Mapping[str, Any] | None = None

    def extend(tree: _Tree, destination: np.ndarray) -> tuple[str, int | None, Mapping[str, Any] | None, bool]:
        if budget_expired():
            return "timed_out", None, timeout_rejection("tree_extend"), True
        nearest_index = tree.nearest(destination)
        nearest = tree.nodes[nearest_index]
        delta = destination - nearest
        maximum = float(np.max(np.abs(delta)))
        if maximum <= 1e-12:
            return "reached", nearest_index, None, True
        scale = min(1.0, cfg.extension_step_rad / maximum)
        candidate = nearest + scale * delta
        edge = edge_safe(nearest, candidate)
        if edge.timed_out:
            return "timed_out", None, edge.rejection, True
        if not edge.available:
            return "trapped", None, edge.rejection, False
        if not edge.safe:
            return "trapped", None, edge.rejection, True
        index = tree.add(candidate, nearest_index)
        return ("reached" if scale >= 1.0 - 1e-12 else "advanced"), index, None, True

    def connect(
        tree: _Tree,
        destination: np.ndarray,
    ) -> tuple[bool, int | None, Mapping[str, Any] | None, bool, bool]:
        while len(start_tree.nodes) + len(target_tree.nodes) < cfg.max_nodes:
            if budget_expired():
                return (
                    False,
                    None,
                    timeout_rejection("tree_connect"),
                    True,
                    True,
                )
            status, index, rejection, available = extend(tree, destination)
            if status == "timed_out":
                return False, None, rejection, True, True
            if status == "trapped":
                return False, None, rejection, available, False
            if status == "reached":
                return True, index, None, True, False
        return False, None, None, True, False

    found: list[np.ndarray] | None = None
    iterations = 0
    timeout_phase: str | None = None
    for iteration in range(1, cfg.max_iterations + 1):
        iterations = iteration
        if budget_expired():
            timeout_phase = "tree_search"
            break
        if len(start_tree.nodes) + len(target_tree.nodes) >= cfg.max_nodes:
            break
        grow_start = iteration % 2 == 1
        active = start_tree if grow_start else target_tree
        other = target_tree if grow_start else start_tree
        opposite_root = start_flat
        if grow_start:
            opposite_root = goal_roots[0] if len(goal_roots) == 1 else goal_roots[(iteration // 2) % len(goal_roots)]
        sample = opposite_root.copy() if rng.random() < cfg.goal_bias else rng.uniform(lower, upper)
        status, active_index, rejection, available = extend(active, sample)
        if rejection is not None:
            last_rejection = rejection
        if status == "timed_out":
            timeout_phase = str(
                (rejection or {}).get("termination_phase", "tree_extend")
            )
            break
        if not available:
            unavailable_rejection = rejection
            break
        if status == "trapped" or active_index is None:
            continue
        connected, other_index, rejection, available, timed_out = connect(
            other,
            active.nodes[active_index],
        )
        if rejection is not None:
            last_rejection = rejection
        if timed_out:
            timeout_phase = str(
                (rejection or {}).get("termination_phase", "tree_connect")
            )
            break
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
        selected_node = other_index if grow_start else active_index
        selected_target_index = safe_target_indices[target_tree.root_index(selected_node)]
        target = targets[selected_target_index]
        if len(targets) > 1:
            base_metadata["selected_target_index"] = selected_target_index
        break

    search_metadata = {
        **base_metadata,
        "direct_path": False,
        "search_iterations": iterations,
        "search_node_count": len(start_tree.nodes) + len(target_tree.nodes),
        "last_rejected_edge": last_rejection,
    }
    if timeout_phase is not None:
        return timeout_result(timeout_phase, search_metadata)
    if unavailable_rejection is not None:
        return _result(JointPathStatus.COLLISION_CHECK_UNAVAILABLE, start, target, movable, resolved_scene_hash, "search_collision_check_unavailable", search_metadata)
    if found is None:
        return _result(JointPathStatus.PATH_SEARCH_EXHAUSTED, start, target, movable, resolved_scene_hash, "bounded_search_exhausted", search_metadata)

    cost_cache: dict[tuple[bytes, bytes], float] = {}

    def cost(a, b):
        key = a.tobytes(), b.tobytes()
        if key not in cost_cache:
            value = float(edge_cost(expand(a), expand(b)))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("edge_cost must return a finite non-negative cost.")
            cost_cache[key] = value
        return cost_cache[key]

    shortened, shortcut_unavailable, shortcut_timed_out = _shortcut_path(
        found,
        edge_safe,
        rng,
        cfg.shortcut_attempts,
        cost=cost if edge_cost is not None else None,
    )
    if shortcut_timed_out:
        return timeout_result(
            "shortcut",
            {**search_metadata, "last_rejected_edge": shortcut_unavailable},
        )
    if shortcut_unavailable is not None:
        return _result(JointPathStatus.COLLISION_CHECK_UNAVAILABLE, start, target, movable, resolved_scene_hash, "shortcut_collision_check_unavailable", {**search_metadata, "last_rejected_edge": shortcut_unavailable})
    final_rejection = None
    for left, right in zip(shortened, shortened[1:]):
        validation = edge_safe(left, right)
        if validation.timed_out:
            return timeout_result(
                "final_revalidation",
                {**search_metadata, "last_rejected_edge": validation.rejection},
            )
        if not validation.available:
            return _result(JointPathStatus.COLLISION_CHECK_UNAVAILABLE, start, target, movable, resolved_scene_hash, "final_path_collision_check_unavailable", {**search_metadata, "last_rejected_edge": validation.rejection})
        if not validation.safe:
            final_rejection = validation.rejection
            break
    if final_rejection is not None:
        return _result(JointPathStatus.PATH_SEARCH_EXHAUSTED, start, target, movable, resolved_scene_hash, "final_path_revalidation_failed", {**search_metadata, "last_rejected_edge": final_rejection})
    cost_metadata = {}
    if edge_cost is not None:
        cost_metadata = {
            "shortcut_cost_before": sum(cost(a, b) for a, b in zip(found, found[1:])),
            "shortcut_cost_after": sum(cost(a, b) for a, b in zip(shortened, shortened[1:])),
        }
    if budget_expired():
        return timeout_result("finalize", search_metadata)
    return _result(
        JointPathStatus.PATH_FOUND,
        start,
        target,
        movable,
        resolved_scene_hash,
        "bounded_search_found_path",
        {
            **search_metadata,
            "shortcut_revalidated": True,
            "unshortened_waypoint_count": len(found),
            "waypoint_count": len(shortened),
            **cost_metadata,
        },
        tuple(expand(item) for item in shortened),
    )


def _resolve_scene_binding(
    *,
    oracle: JointConfigurationOracle,
    requested_scene_hash: str | None,
) -> tuple[str | None, dict[str, Any], str | None]:
    """Bind receipts to the oracle's scene identity, never an unverified label."""

    if requested_scene_hash is not None and not isinstance(requested_scene_hash, str):
        return None, {"scene_binding": "invalid"}, "scene_hash must be a string when provided"
    requested = requested_scene_hash
    if requested is not None and (not requested or requested.strip() != requested):
        return None, {"scene_binding": "invalid"}, "scene_hash must be a non-empty trimmed string"
    missing = object()
    try:
        oracle_value = getattr(oracle, "scene_hash", missing)
    except Exception as exc:
        return None, {
            "scene_binding": "invalid",
            "scene_hash_error_type": type(exc).__name__,
        }, "oracle scene_hash could not be read"
    if oracle_value is missing:
        metadata = {
            "scene_binding": "unbound_oracle",
            "requested_scene_hash": requested,
        }
        if requested is not None:
            return None, metadata, "scene_hash cannot label an oracle without scene identity"
        return None, metadata, None
    if not isinstance(oracle_value, str):
        return None, {
            "scene_binding": "invalid",
            "oracle_scene_hash_type": type(oracle_value).__name__,
            "requested_scene_hash": requested,
        }, "oracle scene_hash must be a string"
    actual = oracle_value
    metadata = {
        "scene_binding": "oracle_bound",
        "oracle_scene_hash": actual,
        "requested_scene_hash": requested,
    }
    if not actual or actual.strip() != actual:
        return None, metadata, "oracle scene_hash must be a non-empty trimmed string"
    if requested is not None and requested != actual:
        return actual, metadata, "requested scene_hash does not match oracle scene_hash"
    return actual, metadata, None


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


def _shortcut_path(
    path: list[np.ndarray],
    edge_safe: Any,
    rng: np.random.Generator,
    attempts: int,
    *, cost: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> tuple[list[np.ndarray], Mapping[str, Any] | None, bool]:
    shortened = [np.asarray(item, dtype=float).copy() for item in path]
    examined: set[tuple[bytes, bytes]] = set()
    for _ in range(attempts):
        if len(shortened) < 3:
            break
        if cost is None:
            left, right = sorted(int(item) for item in rng.choice(len(shortened), size=2, replace=False))
        else:
            # Try the largest removable intervals first. Never repeatedly spend
            # the bounded attempts on the same rejected pair after shortening.
            pairs = [(i, j) for i in range(len(shortened) - 2)
                     for j in range(i + 2, len(shortened))
                     if (shortened[i].tobytes(), shortened[j].tobytes()) not in examined]
            if not pairs:
                break
            left, right = max(pairs, key=lambda pair: (pair[1] - pair[0], -pair[0]))
            examined.add((shortened[left].tobytes(), shortened[right].tobytes()))
        if right <= left + 1:
            continue
        edge = edge_safe(shortened[left], shortened[right])
        if edge.timed_out:
            return shortened, edge.rejection, True
        if not edge.available:
            return shortened, edge.rejection, False
        if edge.safe:
            if cost is not None:
                old_cost = sum(cost(a, b) for a, b in zip(
                    shortened[left:right], shortened[left + 1:right + 1]))
                if cost(shortened[left], shortened[right]) > old_cost + 1e-12:
                    continue
            shortened = shortened[: left + 1] + shortened[right:]
    return shortened, None, False


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


def _invalid_result(
    start: Mapping[str, Any],
    target: Mapping[str, Any],
    movable: Sequence[str],
    scene_hash: str | None,
    reason: str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> JointPathPlanResult:
    fallback_start = _safe_fallback_mapping(start)
    fallback_target = _safe_fallback_mapping(target, fallback=fallback_start)
    try:
        fallback_movable = tuple(
            str(item) for item in movable if str(item) in fallback_start
        )
    except TypeError:
        fallback_movable = ()
    if not fallback_movable or len(set(fallback_movable)) != len(fallback_movable):
        fallback_movable = (next(iter(fallback_start)),)
    if set(fallback_start) != set(fallback_target) or any(
        fallback_start[key].shape != fallback_target[key].shape
        for key in set(fallback_start).intersection(fallback_target)
    ):
        fallback_target = _copy_mapping(fallback_start)
    return _result(
        JointPathStatus.INVALID_REQUEST,
        fallback_start,
        fallback_target,
        fallback_movable,
        scene_hash,
        reason,
        metadata,
    )


def _safe_fallback_mapping(value: Mapping[str, Any], *, fallback: Mapping[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    try:
        return _validated_mapping(value)
    except Exception:
        if fallback is not None:
            return _copy_mapping(fallback)
        return {"invalid_request": np.zeros(1, dtype=float)}


__all__ = ["plan_collision_aware_joint_path"]
