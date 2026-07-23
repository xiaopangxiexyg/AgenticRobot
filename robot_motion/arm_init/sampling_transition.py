from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .constrained_transition import (
    ArmConstrainedTransitionConfig,
    ArmConstrainedTransitionPlan,
    ArmTransitionModel,
    ArmTransitionPlanStep,
    ArmTransitionRiskAssessment,
    evaluate_transition_state,
)
from .qpos_transition import validate_qpos_mapping


@dataclass(frozen=True)
class ArmSamplingTransitionConfig:
    """Bounded deterministic RRT-Connect fallback for constrained qpos init.

    The search is deliberately bounded: exhaustion means only that this search
    budget found no route, never that the configuration space has no route.
    Every returned execution sample is re-evaluated with the normal transition
    constraints before the plan is marked complete.
    """

    seed: int = 0
    max_iterations: int = 1200
    max_nodes: int = 2400
    extension_step_deg: float = 12.0
    edge_resolution_deg: float = 1.0
    sample_padding_deg: float = 45.0
    goal_bias: float = 0.2
    shortcut_attempts: int = 80

    def __post_init__(self) -> None:
        if int(self.max_iterations) < 1:
            raise ValueError("max_iterations must be >= 1.")
        if int(self.max_nodes) < 2:
            raise ValueError("max_nodes must be >= 2.")
        for name in ("extension_step_deg", "edge_resolution_deg", "sample_padding_deg"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
            object.__setattr__(self, name, value)
        bias = float(self.goal_bias)
        if not np.isfinite(bias) or not 0.0 <= bias <= 1.0:
            raise ValueError("goal_bias must be within [0, 1].")
        object.__setattr__(self, "goal_bias", bias)
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "max_iterations", int(self.max_iterations))
        object.__setattr__(self, "max_nodes", int(self.max_nodes))
        object.__setattr__(self, "shortcut_attempts", max(0, int(self.shortcut_attempts)))


@dataclass
class _Tree:
    nodes: list[np.ndarray]
    parents: list[int]

    @classmethod
    def rooted_at(cls, q: np.ndarray) -> "_Tree":
        return cls(nodes=[np.asarray(q, dtype=float).copy()], parents=[-1])

    def add(self, q: np.ndarray, parent: int) -> int:
        self.nodes.append(np.asarray(q, dtype=float).copy())
        self.parents.append(int(parent))
        return len(self.nodes) - 1

    def nearest(self, q: np.ndarray) -> int:
        distances = [float(np.dot(node - q, node - q)) for node in self.nodes]
        return int(np.argmin(distances))

    def root_to(self, index: int) -> list[np.ndarray]:
        path: list[np.ndarray] = []
        cursor = int(index)
        while cursor >= 0:
            path.append(self.nodes[cursor].copy())
            cursor = self.parents[cursor]
        path.reverse()
        return path


def plan_sampling_constrained_qpos_transition(
    *,
    start_qpos_by_side: Mapping[str, Any],
    target_qpos_by_side: Mapping[str, Any],
    model: ArmTransitionModel,
    transition_config: ArmConstrainedTransitionConfig | None = None,
    search_config: ArmSamplingTransitionConfig | None = None,
    sides: Sequence[str] | None = None,
) -> ArmConstrainedTransitionPlan:
    cfg = transition_config if transition_config is not None else ArmConstrainedTransitionConfig()
    search = search_config if search_config is not None else ArmSamplingTransitionConfig()
    start = _validated_sides(start_qpos_by_side, sides=sides)
    target = _validated_sides(target_qpos_by_side, sides=tuple(start))
    _require_same_shapes(start, target)
    order = tuple(start)
    shapes = tuple(start[side].shape for side in order)
    start_flat = _flatten(start, order)
    target_flat = _flatten(target, order)
    start_poses = model.fk_by_side(start)
    target_poses = model.fk_by_side(target)

    def assess(flat: np.ndarray) -> ArmTransitionRiskAssessment:
        return evaluate_transition_state(
            qpos_by_side=_unflatten(flat, order, shapes),
            start_qpos_by_side=start,
            target_qpos_by_side=target,
            sides=order,
            model=model,
            config=cfg,
            start_pose_by_side=start_poses,
            target_pose_by_side=target_poses,
        )

    initial = assess(start_flat)
    target_assessment = assess(target_flat)
    base_metadata: dict[str, Any] = {
        "planner": "bounded_birrt_connect",
        "search_is_complete": False,
        "search_seed": int(search.seed),
        "search_max_iterations": int(search.max_iterations),
        "search_max_nodes": int(search.max_nodes),
        "extension_step_deg": float(search.extension_step_deg),
        "edge_resolution_deg": float(search.edge_resolution_deg),
        "sample_padding_deg": float(search.sample_padding_deg),
        "dt_sec": float(cfg.dt_sec),
        "max_joint_step_deg": float(cfg.max_joint_step_deg),
        "constraint_space": "qpos",
    }
    if initial.stop or initial.hold:
        return _failed_plan(
            "blocked_start" if initial.stop else "hold_start",
            "unsafe_start_state",
            start,
            target,
            initial,
            target_assessment,
            base_metadata,
        )
    if target_assessment.stop or target_assessment.hold:
        return _failed_plan(
            "blocked_target" if target_assessment.stop else "hold_target",
            "unsafe_target_state",
            start,
            target,
            initial,
            target_assessment,
            base_metadata,
        )
    if np.max(np.abs(target_flat - start_flat)) <= float(cfg.target_tolerance_rad):
        return ArmConstrainedTransitionPlan(
            status="completed",
            start_qpos_by_side=_copy_mapping(start),
            target_qpos_by_side=_copy_mapping(target),
            steps=(),
            initial_assessment=initial,
            target_assessment=target_assessment,
            metadata={**base_metadata, "search_iterations": 0, "search_node_count": 2},
        )

    lower, upper = _search_bounds(start_flat, target_flat, cfg, search)
    edge_resolution = np.deg2rad(float(search.edge_resolution_deg))
    extension_step = np.deg2rad(float(search.extension_step_deg))
    rng = np.random.default_rng(int(search.seed))
    start_tree = _Tree.rooted_at(start_flat)
    goal_tree = _Tree.rooted_at(target_flat)
    last_rejection: dict[str, Any] | None = None

    def edge_safe(a: np.ndarray, b: np.ndarray) -> tuple[bool, dict[str, Any] | None]:
        delta = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
        count = max(1, int(np.ceil(float(np.max(np.abs(delta))) / edge_resolution)))
        for sample_index in range(1, count + 1):
            alpha = float(sample_index) / float(count)
            candidate = np.asarray(a, dtype=float) + delta * alpha
            assessment = assess(candidate)
            if assessment.stop or assessment.hold:
                return False, {
                    "edge_sample_index": int(sample_index),
                    "edge_sample_count": int(count),
                    "edge_alpha": alpha,
                    "qpos_by_side": _lists(_unflatten(candidate, order, shapes)),
                    "assessment": assessment.to_dict(),
                }
        return True, None

    def extend(tree: _Tree, target_flat_q: np.ndarray) -> tuple[str, int | None, dict[str, Any] | None]:
        nearest_index = tree.nearest(target_flat_q)
        nearest = tree.nodes[nearest_index]
        delta = target_flat_q - nearest
        max_delta = float(np.max(np.abs(delta)))
        if max_delta <= float(cfg.target_tolerance_rad):
            return "reached", nearest_index, None
        scale = min(1.0, extension_step / max(max_delta, 1e-12))
        candidate = nearest + delta * scale
        safe, rejection = edge_safe(nearest, candidate)
        if not safe:
            return "trapped", None, rejection
        new_index = tree.add(candidate, nearest_index)
        return ("reached" if scale >= 1.0 - 1e-12 else "advanced"), new_index, None

    def connect(tree: _Tree, target_flat_q: np.ndarray) -> tuple[bool, int | None, dict[str, Any] | None]:
        rejection: dict[str, Any] | None = None
        while len(start_tree.nodes) + len(goal_tree.nodes) < int(search.max_nodes):
            status, index, rejection = extend(tree, target_flat_q)
            if status == "trapped":
                return False, None, rejection
            if status == "reached":
                return True, index, None
        return False, None, rejection

    found_path: list[np.ndarray] | None = None
    iterations = 0
    for iteration in range(1, int(search.max_iterations) + 1):
        iterations = iteration
        if len(start_tree.nodes) + len(goal_tree.nodes) >= int(search.max_nodes):
            break
        grow_start = iteration % 2 == 1
        active_tree = start_tree if grow_start else goal_tree
        other_tree = goal_tree if grow_start else start_tree
        opposite_root = target_flat if grow_start else start_flat
        sample = (
            opposite_root.copy()
            if rng.random() < float(search.goal_bias)
            else rng.uniform(lower, upper)
        )
        status, active_index, rejection = extend(active_tree, sample)
        if rejection is not None:
            last_rejection = rejection
        if status == "trapped" or active_index is None:
            continue
        connected, other_index, rejection = connect(other_tree, active_tree.nodes[active_index])
        if rejection is not None:
            last_rejection = rejection
        if not connected or other_index is None:
            continue
        if grow_start:
            start_path = start_tree.root_to(active_index)
            goal_path = goal_tree.root_to(other_index)
        else:
            start_path = start_tree.root_to(other_index)
            goal_path = goal_tree.root_to(active_index)
        found_path = start_path + list(reversed(goal_path))[1:]
        break

    search_metadata = {
        **base_metadata,
        "search_iterations": int(iterations),
        "search_node_count": len(start_tree.nodes) + len(goal_tree.nodes),
        "last_rejected_edge": last_rejection,
    }
    if found_path is None:
        return _failed_plan(
            "search_exhausted",
            "bounded_sampling_search_exhausted",
            start,
            target,
            initial,
            target_assessment,
            search_metadata,
        )

    smoothed = _shortcut_path(
        found_path,
        attempts=int(search.shortcut_attempts),
        rng=rng,
        edge_safe=edge_safe,
    )
    plan = _execution_plan_from_waypoints(
        smoothed,
        start=start,
        target=target,
        order=order,
        shapes=shapes,
        assess=assess,
        config=cfg,
        initial=initial,
        target_assessment=target_assessment,
        metadata={
            **search_metadata,
            "raw_waypoint_count": len(found_path),
            "smoothed_waypoint_count": len(smoothed),
        },
    )
    return plan


def _execution_plan_from_waypoints(
    waypoints: Sequence[np.ndarray],
    *,
    start: Mapping[str, np.ndarray],
    target: Mapping[str, np.ndarray],
    order: Sequence[str],
    shapes: Sequence[tuple[int, ...]],
    assess,
    config: ArmConstrainedTransitionConfig,
    initial: ArmTransitionRiskAssessment,
    target_assessment: ArmTransitionRiskAssessment,
    metadata: Mapping[str, Any],
) -> ArmConstrainedTransitionPlan:
    steps: list[ArmTransitionPlanStep] = []
    current = np.asarray(waypoints[0], dtype=float).copy()
    max_step = np.deg2rad(float(config.max_joint_step_deg))
    for waypoint_index, raw_waypoint in enumerate(waypoints[1:], start=1):
        waypoint = np.asarray(raw_waypoint, dtype=float)
        while float(np.max(np.abs(waypoint - current))) > float(config.target_tolerance_rad):
            if len(steps) >= int(config.max_steps):
                return _failed_plan(
                    "max_steps_exceeded",
                    "sampling_path_time_parameterization_exceeded_max_steps",
                    start,
                    target,
                    initial,
                    target_assessment,
                    {**dict(metadata), "failed_waypoint_index": int(waypoint_index)},
                    steps=steps,
                )
            current_assessment = assess(current)
            if current_assessment.stop or current_assessment.hold:
                return _failed_plan(
                    "blocked_current",
                    "sampling_path_current_state_rejected",
                    start,
                    target,
                    initial,
                    target_assessment,
                    {**dict(metadata), "failed_waypoint_index": int(waypoint_index)},
                    steps=steps,
                )
            remaining = waypoint - current
            allowed = max_step * max(float(current_assessment.speed_scale), 1e-6)
            local_alpha = min(1.0, allowed / max(float(np.max(np.abs(remaining))), 1e-12))
            candidate = current + remaining * local_alpha
            assessment = assess(candidate)
            if assessment.stop or assessment.hold:
                return _failed_plan(
                    "blocked_candidate",
                    "sampling_path_execution_sample_rejected",
                    start,
                    target,
                    initial,
                    target_assessment,
                    {
                        **dict(metadata),
                        "failed_waypoint_index": int(waypoint_index),
                        "rejected_execution_sample": {
                            "qpos_by_side": _lists(_unflatten(candidate, order, shapes)),
                            "assessment": assessment.to_dict(),
                        },
                    },
                    steps=steps,
                )
            current = candidate
            steps.append(
                ArmTransitionPlanStep(
                    index=len(steps) + 1,
                    time_sec=(len(steps) + 1) * float(config.dt_sec),
                    qpos_by_side=_unflatten(current, order, shapes),
                    assessment=assessment,
                    alpha=_path_progress(waypoints, waypoint_index, local_alpha),
                    metadata={
                        "sampling_waypoint_index": int(waypoint_index),
                        "sampling_local_alpha": float(local_alpha),
                    },
                )
            )
    return ArmConstrainedTransitionPlan(
        status="completed",
        start_qpos_by_side=_copy_mapping(start),
        target_qpos_by_side=_copy_mapping(target),
        steps=tuple(steps),
        initial_assessment=initial,
        target_assessment=target_assessment,
        metadata=dict(metadata),
    )


def _shortcut_path(path, *, attempts: int, rng, edge_safe) -> list[np.ndarray]:
    out = [np.asarray(value, dtype=float).copy() for value in path]
    for _ in range(int(attempts)):
        if len(out) <= 2:
            break
        first, second = sorted(rng.choice(len(out), size=2, replace=False).tolist())
        if second <= first + 1:
            continue
        safe, _rejection = edge_safe(out[first], out[second])
        if safe:
            out = out[: first + 1] + out[second:]
    return out


def _search_bounds(start, target, cfg, search):
    dof = int(np.asarray(start).size)
    if cfg.joint_lower_rad is not None and cfg.joint_upper_rad is not None:
        per_side_lower = np.asarray(cfg.joint_lower_rad, dtype=float)
        per_side_upper = np.asarray(cfg.joint_upper_rad, dtype=float)
        repeats = dof // per_side_lower.size
        hard_lower = np.tile(per_side_lower, repeats)
        hard_upper = np.tile(per_side_upper, repeats)
    else:
        hard = np.deg2rad(float(cfg.max_joint_abs_deg))
        hard_lower = np.full(dof, -hard, dtype=float)
        hard_upper = np.full(dof, hard, dtype=float)
    padding = np.deg2rad(float(search.sample_padding_deg))
    lower = np.maximum(hard_lower, np.minimum(start, target) - padding)
    upper = np.minimum(hard_upper, np.maximum(start, target) + padding)
    if np.any(lower >= upper):
        raise ValueError("sampling search bounds are empty for one or more joints.")
    return lower, upper


def _failed_plan(status, reason, start, target, initial, target_assessment, metadata, *, steps=()):
    return ArmConstrainedTransitionPlan(
        status=str(status),
        start_qpos_by_side=_copy_mapping(start),
        target_qpos_by_side=_copy_mapping(target),
        steps=tuple(steps),
        initial_assessment=initial,
        target_assessment=target_assessment,
        blocked_reason=str(reason),
        metadata=dict(metadata),
    )


def _validated_sides(qpos_by_side, *, sides):
    qpos = validate_qpos_mapping(qpos_by_side)
    if sides is None:
        return qpos
    return {str(side): qpos[str(side)].copy() for side in sides}


def _require_same_shapes(start, target):
    if tuple(start) != tuple(target):
        raise ValueError("start and target qpos sides must match.")
    for side in start:
        if start[side].shape != target[side].shape:
            raise ValueError(f"{side}.qpos shape mismatch: {start[side].shape} != {target[side].shape}.")


def _flatten(qpos_by_side, order):
    return np.concatenate([np.asarray(qpos_by_side[side], dtype=float).reshape(-1) for side in order])


def _unflatten(flat, order, shapes):
    out: dict[str, np.ndarray] = {}
    offset = 0
    for side, shape in zip(order, shapes):
        size = int(np.prod(shape))
        out[str(side)] = np.asarray(flat[offset : offset + size], dtype=float).reshape(shape).copy()
        offset += size
    return out


def _copy_mapping(qpos_by_side):
    return {str(side): np.asarray(value, dtype=float).copy() for side, value in qpos_by_side.items()}


def _lists(qpos_by_side):
    return {str(side): np.asarray(value, dtype=float).reshape(-1).tolist() for side, value in qpos_by_side.items()}


def _path_progress(waypoints: Sequence[np.ndarray], waypoint_index: int, local_alpha: float) -> float:
    edge_count = max(1, len(waypoints) - 1)
    return float(np.clip((float(waypoint_index - 1) + float(local_alpha)) / edge_count, 0.0, 1.0))


__all__ = ["ArmSamplingTransitionConfig", "plan_sampling_constrained_qpos_transition"]
