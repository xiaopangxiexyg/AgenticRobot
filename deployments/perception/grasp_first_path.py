"""Bounded first-acceptable search; no devices, execution, or model lifecycle."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_motion.pose import pose_delta_metrics

AXIAL_GRASP_SEARCH_ORDER = "original_motion_first_then_bounded_axial_fallback_v2"
GRASP_SEARCH_PREFERENCE = "observed_closing_contact_and_weak_side_insertion_v11"


@dataclass(frozen=True)
class FirstGraspPathPolicy:
    """Fixed search policy; feasibility remains the lower planner's decision."""

    # Fast endpoint rejections must not exhaust coverage while the shared
    # wall-clock budget still has time for a later feasible grasp family.
    maximum_path_queries: int = 64
    total_budget_sec: float = 20.0
    path_query_budget_sec: float = 8.0
    endpoint_probe_budget_sec: float = 0.08
    endpoint_probe_total_budget_sec: float = 2.0
    axial_path_query_budget_sec: float = 3.0
    axial_publication_reserve_sec: float = 0.5
    maximum_contact_refinement_queries: int = 2
    maximum_original_neighbor_queries: int = 2
    original_neighbor_query_budget_sec: float = 3.0
    cheap_endpoint_rejection_sec: float = 2.0
    translation_weight: float = 1.0
    orientation_weight: float = 0.35
    arm_side_weight: float = 0.25
    model_score_weight: float = 0.10
    capture_support_weight: float = 0.35
    capture_support_saturation_score: float = 0.5
    capture_height_support_weight: float = 0.35
    opposed_capture_support_weight: float = 0.35
    surface_alignment_weight: float = 0.35
    minimum_capture_height_support_score: float = 0.5
    minimum_contact_height_evidence_weight: float = 3.0
    family_translation_m: float = 0.05
    family_orientation_deg: float = 15.0

    def __post_init__(self) -> None:
        if (
            type(self.maximum_path_queries) is not int
            or not 1 <= self.maximum_path_queries <= 64
        ):
            raise ValueError("maximum_path_queries must be an integer in [1, 64].")
        if (type(self.maximum_original_neighbor_queries) is not int
                or not 0 <= self.maximum_original_neighbor_queries <= 2):
            raise ValueError("maximum_original_neighbor_queries must be an integer in [0, 2].")
        if (type(self.maximum_contact_refinement_queries) is not int
                or not 0 <= self.maximum_contact_refinement_queries <= 2):
            raise ValueError("maximum_contact_refinement_queries must be an integer in [0, 2].")
        for name in ("total_budget_sec", "path_query_budget_sec", "axial_path_query_budget_sec",
                     "endpoint_probe_budget_sec", "endpoint_probe_total_budget_sec",
                     "axial_publication_reserve_sec", "original_neighbor_query_budget_sec",
                     "cheap_endpoint_rejection_sec"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 < value <= 120
            ):
                raise ValueError(f"{name} must be finite and in (0, 120].")
        for name in (
            "translation_weight",
            "orientation_weight",
            "arm_side_weight",
            "model_score_weight",
            "capture_support_weight",
            "capture_height_support_weight",
            "opposed_capture_support_weight",
            "surface_alignment_weight",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative.")
        if not self.translation_weight + self.orientation_weight + self.arm_side_weight:
            raise ValueError("The search policy requires a motion preference.")
        for name, maximum in (
            ("minimum_contact_height_evidence_weight", 120.0),
            ("capture_support_saturation_score", 1.0),
            ("minimum_capture_height_support_score", 1.0),
            ("family_translation_m", 0.25),
            ("family_orientation_deg", 90.0),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 < value <= maximum
            ):
                raise ValueError(f"{name} is outside its positive finite search bound.")


def _capture_score(candidate: Any, name: str) -> float:
    # Older typed candidate ports have no support observation. Keep their
    # historical ordering instead of inventing a negative geometric fact.
    score = getattr(candidate, name, 1.0)
    if (isinstance(score, bool) or not isinstance(score, (int, float))
            or not math.isfinite(score) or not 0 <= score <= 1):
        raise ValueError(f"Candidate {name} support must be finite and in [0, 1].")
    return float(score)


def grasp_capture_height_acceptable(candidate: Any, *, policy: FirstGraspPathPolicy) -> bool:
    """Require local pad support from observed density or closing-face evidence."""
    return _capture_height_score(candidate, policy=policy) >= policy.minimum_capture_height_support_score


def grasp_contact_height_observed(candidate: Any, *, policy: FirstGraspPathPolicy) -> bool:
    """Require distinct, normal-weighted evidence before replacing density.

    The producer deduplicates observations and bounds every directional vote
    by one. Three effective observations cannot be manufactured by duplicated
    pixels or one highly weighted outlier. Missing/weak views remain unknown.
    """
    weight = getattr(candidate, "contact_height_evidence_weight", 0.0)
    if (isinstance(weight, bool) or not isinstance(weight, (int, float))
            or not math.isfinite(weight) or weight < 0):
        raise ValueError("Candidate contact height evidence weight must be finite and nonnegative.")
    return weight >= policy.minimum_contact_height_evidence_weight


def _capture_height_score(candidate: Any, *, policy: FirstGraspPathPolicy) -> float:
    density = _capture_score(candidate, "capture_height_support_score")
    if (not hasattr(candidate, "contact_height_support_score")
            or not hasattr(candidate, "surface_alignment_score")):
        return density
    contact = _capture_score(candidate, "contact_height_support_score")
    surface = _capture_score(candidate, "surface_alignment_score")
    if grasp_contact_height_observed(candidate, policy=policy):
        # A dense top face is not a substitute for observed closing surfaces.
        # Reuse the existing local height floor, not object size/world Z.
        return contact
    # One reliable closing face earns at most half the surface score. Convert
    # that partial-view evidence to local band support without inventing the
    # other face. Weak/unknown normals cannot rescue a density-band miss.
    return max(density, min(contact, 2*surface))


def grasp_capture_support_cost_components(
    candidate: Any, *, policy: FirstGraspPathPolicy,
) -> dict[str, float]:
    """Return the weighted soft terms used by ranking and its diagnostics."""
    score = _capture_score(candidate, "capture_support_score")
    # The observed score saturates at 20% of finger length. Stop paying extra
    # motion for depth after half of that range (10 mm for a 100 mm finger).
    # This is a soft preference, not evidence of force closure or a hard floor.
    supported = min(score / policy.capture_support_saturation_score, 1.0)
    height = _capture_height_score(candidate, policy=policy)
    opposed = _capture_score(candidate, "opposed_capture_support_score")
    surface = _capture_score(candidate, "surface_alignment_score")
    return {
        "axial_insertion": policy.capture_support_weight * (1 - supported),
        "pad_height": policy.capture_height_support_weight * (1-height),
        "opposed_insertion": policy.opposed_capture_support_weight * (1-opposed),
        "surface_alignment": policy.surface_alignment_weight * (1-surface),
    }


def grasp_capture_support_cost(candidate: Any, *, policy: FirstGraspPathPolicy) -> float:
    """Separate axial insertion from pad-height placement in the same cost."""
    cost = sum(grasp_capture_support_cost_components(candidate, policy=policy).values())
    if not math.isfinite(cost):
        raise ValueError("Aggregate capture support cost must be finite.")
    return cost


def grasp_axial_refinement_support(candidate: Any) -> tuple[float, float]:
    """Prefer improving the weaker observed closing edge, not just the median.

    Sparse/uncomputed opposed evidence is unknown: it cannot turn an already
    deep central patch into a measured shallow edge. This is only allocation
    of existing bounded proposals, never a grasp or collision admission gate.
    The central score breaks ties when the weaker side cannot yet be seen.
    """
    central = _capture_score(candidate, "capture_support_score")
    opposed = _capture_score(candidate, "opposed_capture_support_score")
    observed = getattr(candidate, "opposed_capture_support_observed_sides", 0)
    if type(observed) is not int or not 0 <= observed <= 2:
        raise ValueError("Opposed capture observed sides must be an integer in [0, 2].")
    return (min(central, opposed) if observed else central), central


def grasp_contact_evidence(candidate: Any) -> dict[str, float | int]:
    """Bounded observed geometry, not a contact or physical grasp verdict."""
    evidence = {}
    for name in ("capture_support_score", "opposed_capture_support_score",
                 "surface_alignment_score", "surface_opposition_score",
                 "contact_height_support_score"):
        if hasattr(candidate, name):
            evidence[name] = _capture_score(candidate, name)
    for name in ("opposed_capture_support_observed_sides", "surface_alignment_observed_sides"):
        if hasattr(candidate, name):
            value = getattr(candidate, name)
            if type(value) is not int or not 0 <= value <= 2:
                raise ValueError("Contact observed sides must be an integer in [0, 2].")
            evidence[name] = value
    return evidence


def grasp_axial_refinement_improves(candidate: Any, source: Any) -> bool:
    """Do not promote loss of a measured edge to improvement of that edge."""
    before = grasp_axial_refinement_support(source)
    after = grasp_axial_refinement_support(candidate)
    if (getattr(candidate, "opposed_capture_support_observed_sides", 0)
            < getattr(source, "opposed_capture_support_observed_sides", 0)):
        return False
    return after > before


def grasp_search_preference(
    candidate: Any, motion_cost: float, *, policy: FirstGraspPathPolicy,
) -> tuple[float, float]:
    """Balance observed insertion against motion without adding a hard gate.

    Source allocation still uses the motion preference. This key only ranks
    already filtered candidates, with motion breaking equal total costs.
    """
    if (isinstance(motion_cost, bool) or not isinstance(motion_cost, (int, float))
            or not math.isfinite(motion_cost) or motion_cost < 0):
        raise ValueError("Candidate motion cost must be finite and nonnegative.")
    cost = motion_cost + grasp_capture_support_cost(candidate, policy=policy)
    if not math.isfinite(cost):
        raise ValueError("Aggregate grasp search preference must be finite.")
    return cost, motion_cost


def diverse_grasp_search_order(
    poses: Sequence[Any],
    ordered_indices: Sequence[int],
    *,
    policy: FirstGraspPathPolicy,
) -> tuple[int, ...]:
    """Round-robin nearby pose families while retaining preference within each.

    This only allocates a finite search budget. No pose is changed or removed,
    and both closing-axis branches remain distinct whole-arm IK problems.
    """
    indices = tuple(ordered_indices)
    if any(
        type(index) is not int or not 0 <= index < len(poses) for index in indices
    ) or len(set(indices)) != len(indices):
        raise ValueError("Diversity order requires distinct valid indices.")
    families: list[list[int]] = []
    matrices = {
        index: np.asarray(poses[index], dtype=float).reshape(4, 4) for index in indices
    }
    for index in indices:
        pose = matrices[index]
        if not np.isfinite(pose).all():
            raise ValueError("Diversity ordering requires finite poses.")
        for family in families:
            representative = matrices[family[0]]
            if (
                np.linalg.norm(pose[:3, 3] - representative[:3, 3])
                >= policy.family_translation_m
            ):
                continue
            angle = math.degrees(
                math.acos(
                    float(
                        np.clip(
                            (np.trace(representative[:3, :3].T @ pose[:3, :3]) - 1.0)
                            / 2.0,
                            -1.0,
                            1.0,
                        )
                    )
                )
            )
            if angle < policy.family_orientation_deg:
                family.append(index)
                break
        else:
            families.append([index])
    return tuple(
        family[depth]
        for depth in range(max(map(len, families), default=0))
        for family in families
        if depth < len(family)
    )


def original_grasp_neighbors(
    candidates: Sequence[Any], ordered_indices: Sequence[int], *, policy: FirstGraspPathPolicy,
) -> dict[int, tuple[int, ...]]:
    """Find eligible raw poses hidden by a nearby derived family representative.

    This only supplies optional short probes after a cheap endpoint rejection.
    No candidate is removed from its normal order or inherits another's IK.
    """
    indices = tuple(ordered_indices)
    if (any(type(i) is not int or not 0 <= i < len(candidates) for i in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError("Neighbor search requires distinct valid candidate indices.")
    originals = [i for i in indices if getattr(candidates[i], "pose_source", "model_scored") == "model_scored"]
    derived = [i for i in indices if i not in originals]
    if not originals or not derived or not policy.maximum_original_neighbor_queries:
        return {}
    poses = {i: np.asarray(candidates[i].base_from_tcp, dtype=float).reshape(4, 4) for i in indices}
    if any(not np.isfinite(pose).all() for pose in poses.values()):
        raise ValueError("Neighbor search requires finite poses.")
    result = {}
    for i in derived:
        neighbors = []
        for rank, j in enumerate(originals):
            distance, angle = pose_delta_metrics(poses[i], poses[j])
            if (distance < policy.family_translation_m and angle < policy.family_orientation_deg
                    and grasp_capture_support_cost(candidates[j], policy=policy)
                    <= grasp_capture_support_cost(candidates[i], policy=policy)):
                neighbors.append((distance / policy.family_translation_m + angle / policy.family_orientation_deg, rank, j))
        if neighbors:
            result[i] = tuple(j for _, _, j in sorted(neighbors))
    return result


def refined_grasp_search_order(
    candidates: Sequence[Any], ordered_indices: Sequence[int], *, policy: FirstGraspPathPolicy,
) -> tuple[int, ...]:
    """Keep the original motion-first search before optional axial refinement.

    Children do not displace other families by their inherited model score or
    improved capture score. Both the child and unchanged parent still need
    separate complete path validation. A failed parent path is not evidence
    against its child. Noneligible parents are not resurrected.
    """
    indices = tuple(ordered_indices)
    if any(type(index) is not int or not 0 <= index < len(candidates) for index in indices
           ) or len(set(indices)) != len(indices):
        raise ValueError("Refinement order requires distinct valid candidate indices.")
    children = set()
    for index in indices:
        parent = getattr(candidates[index], "axial_parent_index", None)
        if parent is None:
            continue
        if (type(parent) is not int or not 0 <= parent < len(candidates) or parent == index
                or getattr(candidates[parent], "axial_parent_index", None) is not None):
            raise ValueError("Axial refinement requires a distinct original parent.")
        children.add(index)
    roots = [index for index in indices if index not in children]
    order = diverse_grasp_search_order(
        [candidate.base_from_tcp for candidate in candidates], roots, policy=policy)
    return order + tuple(index for index in indices if index in children)


class GraspPathQueryRejected(RuntimeError):
    """An ordinary rejected goal, not an input, scene, or planner fault."""

    def __init__(self, reason_code: str) -> None:
        if not isinstance(reason_code, str) or not reason_code:
            raise ValueError("A rejected query requires a reason code.")
        self.reason_code = reason_code
        super().__init__(reason_code)


@dataclass(frozen=True)
class FirstGraspPathResult:
    selected_index: int | None
    trajectory: Any | None
    attempts: tuple[dict[str, Any], ...]
    reason_code: str
    elapsed_sec: float
    endpoint_probes: tuple[dict[str, Any], ...] = ()


def refine_first_grasp_path(
    result: FirstGraspPathResult, *, candidates: Sequence[Any],
    eligible_indices: Sequence[int], plan: Callable[[Any, float, float], Any],
    policy: FirstGraspPathPolicy, deadline: float,
    preference_costs: Mapping[int, float] | None = None,
    endpoint_probe: Callable[[Any, float], bool] | None = None,
    monotonic: Callable[[], float] = time.perf_counter,
) -> FirstGraspPathResult:
    """Refine only a complete success, retaining it on ordinary goal rejection.

    No new wall-clock budget or query allowance is created. The caller's plan
    port must preserve ownership of the original success until replacement.
    Input, cancellation, scene and planner faults are never fallback reasons.
    """
    if result.selected_index is None or result.trajectory is None:
        return result
    if not math.isfinite(deadline):
        raise ValueError("deadline must be finite.")
    began = monotonic()

    def retain_original(now: float) -> FirstGraspPathResult:
        if now >= deadline:
            return FirstGraspPathResult(None, None, result.attempts,
                "grasp_fast_search_budget_exhausted",
                result.elapsed_sec + max(0., now - began), result.endpoint_probes)
        return result

    if began >= deadline or len(result.attempts) >= policy.maximum_path_queries:
        return retain_original(began)
    # Preserve a still-valid original when no useful refinement fits.
    # Every early return must still enforce the caller-owned global deadline.
    if deadline - began < policy.axial_path_query_budget_sec + policy.axial_publication_reserve_sec:
        return retain_original(monotonic())
    tried = {attempt["index"] for attempt in result.attempts}
    selected = candidates[result.selected_index]
    alternatives = []
    for i in eligible_indices:
        if i in tried:
            continue
        candidate = candidates[i]
        if getattr(candidate, "axial_parent_index", None) == result.selected_index:
            if (not getattr(selected, "opposed_capture_support_observed_sides", 0)
                    or grasp_axial_refinement_improves(candidate, selected)):
                alternatives.append(i)
        elif (preference_costs is not None
                and preference_costs[i] < preference_costs[result.selected_index]
                and grasp_axial_refinement_improves(candidate, selected)):
            distance, rotation = pose_delta_metrics(
                np.asarray(selected.base_from_tcp).reshape(4, 4),
                np.asarray(candidate.base_from_tcp).reshape(4, 4))
            if distance < policy.family_translation_m and rotation < policy.family_orientation_deg:
                alternatives.append(i)
    if not alternatives or not policy.maximum_contact_refinement_queries:
        return retain_original(monotonic())
    alternatives.sort(key=lambda i: (
        *(-value for value in grasp_axial_refinement_support(candidates[i])), i))
    index, trajectory, reason = result.selected_index, result.trajectory, "path_found"
    refinement_deadline = min(began + policy.axial_path_query_budget_sec,
                              deadline - policy.axial_publication_reserve_sec)
    # Keep the original handle when selection has spent the optional slice.
    if refinement_deadline - monotonic() <= 0:
        return retain_original(monotonic())
    attempts = list(result.attempts)
    endpoint_probes = list(result.endpoint_probes)
    # Refinement shares the primary search's advisory probe allowance.
    # Exhaustion skips only the probe, never the full path validation.
    probe_remaining = max(0.0, policy.endpoint_probe_total_budget_sec
        - sum(probe["elapsed_sec"] for probe in endpoint_probes))
    for child in alternatives[:policy.maximum_contact_refinement_queries]:
        now = monotonic()
        if len(attempts) >= policy.maximum_path_queries or now >= refinement_deadline:
            break
        if (endpoint_probe is not None and probe_remaining > 0
                and len(endpoint_probes) < policy.maximum_path_queries):
            promising = endpoint_probe(candidates[child], min(policy.endpoint_probe_budget_sec,
                probe_remaining, refinement_deadline-now))
            if type(promising) is not bool:
                raise TypeError("Endpoint probes must return an advisory bool.")
            probe_elapsed = max(0., monotonic()-now)
            endpoint_probes.append({"index": child, "promising": promising,
                "elapsed_sec": probe_elapsed})
            probe_remaining = max(0.0, probe_remaining - probe_elapsed)
            now = monotonic()
            if not promising or now >= refinement_deadline:
                # Optional improvement only: an unknown endpoint is not a
                # veto on the original fully checked path or a new retry loop.
                continue
        try:
            improved = plan(candidates[child], min(policy.path_query_budget_sec,
                refinement_deadline - now), refinement_deadline)
        except GraspPathQueryRejected as error:
            attempt_reason = error.reason_code
        else:
            if improved is None:
                raise TypeError("A successful refinement must return a trajectory.")
            # The native owner checks the subdeadline atomically with pending
            # replacement. A later shared clock read cannot restore a handle
            # that this successful transaction has already invalidated.
            index, trajectory, attempt_reason = child, improved, "path_found"
        attempts.append({"index": child, "reason_code": attempt_reason,
            "elapsed_sec": max(0., monotonic()-now)})
        if attempt_reason == "path_found":
            break
    finished = monotonic()
    elapsed = max(0., finished - began)
    if finished >= deadline:
        index, trajectory, reason = None, None, "grasp_fast_search_budget_exhausted"
        if len(attempts) > len(result.attempts):
            attempts[-1]["reason_code"] = reason
    return FirstGraspPathResult(index, trajectory, tuple(attempts), reason, result.elapsed_sec + elapsed,
                               tuple(endpoint_probes))


def find_first_grasp_path(
    *,
    candidates: Sequence[Any],
    ordered_indices: Sequence[int],
    plan: Callable[[Any, float], Any],
    policy: FirstGraspPathPolicy,
    deadline: float,
    original_neighbors: Mapping[int, Sequence[int]] | None = None,
    endpoint_probe: Callable[[Any, float], bool] | None = None,
    monotonic: Callable[[], float] = time.perf_counter,
) -> FirstGraspPathResult:
    """Stop at the first fully validated path under one caller-owned deadline.

    Only explicit goal rejections are retryable. A broken planner, corrupt
    input, cancellation or request-wide invalid scene must escape immediately.
    The caller must include observation and generation in the same deadline.
    """

    if not isinstance(policy, FirstGraspPathPolicy):
        raise TypeError("policy must be FirstGraspPathPolicy.")
    if not math.isfinite(deadline):
        raise ValueError("deadline must be finite.")
    if endpoint_probe is not None and not callable(endpoint_probe):
        raise TypeError("endpoint_probe must be callable or None.")
    indices = tuple(ordered_indices)
    if any(
        type(index) is not int or not 0 <= index < len(candidates) for index in indices
    ) or len(set(indices)) != len(indices):
        raise ValueError("Search order must contain distinct valid candidate indices.")
    if original_neighbors is not None and not isinstance(original_neighbors, Mapping):
        raise TypeError("Original neighbors must be a candidate-index mapping.")
    neighbors = {} if original_neighbors is None else {i: tuple(v) for i, v in original_neighbors.items()}
    for index, values in neighbors.items():
        if (type(index) is not int or index not in indices
                or getattr(candidates[index], "pose_source", "model_scored") == "model_scored"
                or any(type(i) is not int or i not in indices or i == index
                    or getattr(candidates[i], "pose_source", "model_scored") != "model_scored" for i in values)
                or len(set(values)) != len(values)):
            raise ValueError("Original neighbors must reference eligible derived and raw candidates.")
    started = monotonic()
    attempts: list[dict[str, Any]] = []
    endpoint_probes: list[dict[str, Any]] = []
    reason = "grasp_fast_no_acceptable_path"
    normal = list(indices)
    deferred: list[int] = []
    probe_remaining = policy.endpoint_probe_total_budget_sec
    probes: list[int] = []
    probed: set[int] = set()
    tried: set[int] = set()
    while (normal or probes or deferred) and len(attempts) < policy.maximum_path_queries:
        if len(endpoint_probes) >= policy.maximum_path_queries:
            probe_remaining = 0.0
        if deferred and ((not normal and not probes) or probe_remaining <= 0):
            normal, deferred = deferred + normal, []
            probe_remaining = 0.0
        is_probe = bool(probes)
        index = probes.pop(0) if is_probe else normal.pop(0)
        remaining = deadline - monotonic()
        if remaining <= 0:
            reason = "grasp_fast_search_budget_exhausted"
            break
        if not is_probe and endpoint_probe is not None and probe_remaining > 0:
            began = monotonic()
            remaining = deadline - began
            if remaining <= 0:
                reason = "grasp_fast_search_budget_exhausted"
                break
            promising = endpoint_probe(candidates[index], min(
                policy.endpoint_probe_budget_sec, probe_remaining, remaining))
            elapsed = max(0.0, monotonic() - began)
            if type(promising) is not bool:
                raise TypeError("Endpoint probes must return an advisory bool.")
            endpoint_probes.append({"index": index, "promising": promising, "elapsed_sec": elapsed})
            probe_remaining = max(0.0, probe_remaining - elapsed)
            if not promising:
                # A bounded miss is unknown, never an IK-equivalence proof or
                # a veto. Keep its full-budget chance after the exploratory pass.
                deferred.append(index)
                continue
            remaining = deadline - monotonic()
            if remaining <= 0:
                reason = "grasp_fast_search_budget_exhausted"
                break
        began = monotonic()
        remaining = deadline - began
        if remaining <= 0:
            reason = "grasp_fast_search_budget_exhausted"
            break
        try:
            trajectory = plan(
                candidates[index], min(remaining, policy.path_query_budget_sec,
                    policy.original_neighbor_query_budget_sec if is_probe else policy.path_query_budget_sec,
                    policy.axial_path_query_budget_sec
                    if getattr(candidates[index], "axial_parent_index", None) is not None
                    else policy.path_query_budget_sec)
            )
        except GraspPathQueryRejected as error:
            elapsed = max(0.0, monotonic() - began)
            attempts.append(
                {
                    "index": index,
                    "reason_code": error.reason_code,
                    "elapsed_sec": elapsed,
                    **({"query_mode": "original_model_neighbor"} if is_probe else {}),
                }
            )
            tried.add(index)
            if (not is_probe and elapsed <= policy.cheap_endpoint_rejection_sec
                    and error.reason_code in {"ik_search_exhausted", "endpoint_ik_search_exhausted",
                        "safe_tcp_endpoint_search_exhausted"}
                    and len(probed) < policy.maximum_original_neighbor_queries):
                neighbor = next((i for i in neighbors.get(index, ()) if i not in tried and i not in probed), None)
                if neighbor is not None:
                    probes.append(neighbor)
                    probed.add(neighbor)
            continue
        if trajectory is None:
            raise TypeError("A successful path query must return a trajectory.")
        finished = monotonic()
        elapsed = max(0.0, finished - began)
        if finished >= deadline:
            attempts.append(
                {
                    "index": index,
                    "reason_code": "grasp_fast_search_budget_exhausted",
                    "elapsed_sec": elapsed,
                }
            )
            reason = "grasp_fast_search_budget_exhausted"
            break
        attempts.append(
            {"index": index, "reason_code": "path_found", "elapsed_sec": elapsed,
                **({"query_mode": "original_model_neighbor"} if is_probe else {})}
        )
        return FirstGraspPathResult(
            index,
            trajectory,
            tuple(attempts),
            "path_found",
            max(0.0, finished - started),
            tuple(endpoint_probes),
        )
    if monotonic() >= deadline:
        reason = "grasp_fast_search_budget_exhausted"
    return FirstGraspPathResult(
        None, None, tuple(attempts), reason, max(0.0, monotonic() - started), tuple(endpoint_probes)
    )
