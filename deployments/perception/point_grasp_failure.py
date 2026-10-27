"""Domain-owned failure classification for bounded point-grasp search."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping


def bounded_point_grasp_failure_diagnostic(diagnostic: Mapping[str, object]) -> dict:
    """Bound optional examples, never the primary cause or measured counts.

    Use the neutral executor's exact JSON byte accounting. Both provider and
    wrapper call this after adding their own fields, so Unicode retention
    errors cannot replace the actual planning failure with a size exception.
    """
    result = dict(diagnostic)

    def size():
        return len(json.dumps(result, allow_nan=False, sort_keys=True,
                              separators=(",", ":")).encode("utf-8"))

    examples = list(result.get("native_path_attempts", ()))
    while examples and size() > 4096:
        examples.pop()
        result["native_path_attempts"] = examples
        result["native_path_attempts_omitted"] = result.get("native_path_attempts_omitted", 0) + 1
    retention = result.get("debug_evidence_failure")
    if size() > 4096 and isinstance(retention, Mapping) and isinstance(retention.get("message"), str):
        retention = dict(retention)
        message = retention["message"]
        retention["message_truncated"] = True
        result["debug_evidence_failure"] = retention
        while message and size() > 4096:
            message = message[:-1]
            retention["message"] = message
    if size() > 4096:
        raise ValueError("Required point-grasp failure facts exceed the diagnostic bound")
    return result


class PointGraspPlanningError(RuntimeError):
    """No admitted proposal, distinct from a full-path or execution failure."""

    def __init__(self, summary: Mapping[str, object], *,
                 replay_arrays: Mapping | None = None) -> None:
        self.filter_summary = dict(summary)
        self.replay_arrays = {} if replay_arrays is None else dict(replay_arrays)
        self.diagnostic = point_grasp_failure_diagnostic(self.filter_summary)
        self.reason_code = self.diagnostic["reason_code"]
        super().__init__(
            f"Point-grasp bounded search rejected at {self.diagnostic['stage']}; "
            f"reason={self.reason_code}; no executable candidate was published."
        )


def point_grasp_failure_diagnostic(summary: Mapping[str, object]) -> dict:
    """Summarize the actual empty funnel; never infer a cause from one pose."""
    fields = {
        "generated": "detected",
        "width_valid": "width_valid",
        "target_capture_valid": "target_capture_valid",
        "click_distance_valid": "click_distance_valid",
        "terminal_collision_valid": "visible_collision_valid",
        "physical_pool": "arm_shortlist_pool_count",
        "side_admissible": "arm_shortlist_side_admissible_count",
        "wrist_orientation_rejected": "arm_shortlist_wrist_orientation_rejected_count",
        "geometric_checked": "arm_shortlist_geometric_checked_count",
        "geometric_retained": "arm_shortlist_geometric_retained_count",
        "endpoint_probed": "arm_shortlist_endpoint_probe_count",
        "local_chain_assessed": "arm_shortlist_local_chain_assessment_count",
        "local_chain_feasible": "arm_shortlist_local_chain_feasible_count",
        "observed_chain_checked": "arm_shortlist_visible_scene_check_count",
        "observed_chain_feasible": "arm_shortlist_visible_scene_feasible_count",
        "published": "selected",
    }
    counts = {name: summary.get(key) for name, key in fields.items()}
    # Missing means not reported, not a measured zero.
    rejections = json.loads(str(summary.get("arm_shortlist_rejections_json", "{}")))
    full_path_search = "arm_shortlist_full_path_query_count" in summary
    if full_path_search:
        # Fast selection bypasses the ordinary reachability shortlist. Its
        # complete-query outcomes are the authoritative rejection denominator.
        path_attempts = json.loads(str(summary.get("arm_shortlist_full_path_attempts_json", "[]")))
        rejections = Counter(row["reason_code"] for row in path_attempts
                             if row["reason_code"] != "path_found")
    stages = Counter()
    for reason, count in rejections.items():
        if reason == "endpoint_geometric_reach_exceeded":
            stage = "geometric_reach"
        elif reason == "grasp_path_motion_quality_exceeded":
            stage = "motion_quality"
        elif reason == "ik_search_exhausted":
            stage = "endpoint_ik"
        elif reason in {"path_planning_time_budget_exhausted", "grasp_fast_search_budget_exhausted"}:
            stage = "search_budget"
        elif reason.startswith("approach_"):
            stage = "local_grasp_chain"
        elif "collision" in reason:
            stage = "collision"
        else:
            stage = "other_local_chain_rejection"
        stages[stage] += count
    if full_path_search:
        counts["full_path_queried"] = summary["arm_shortlist_full_path_query_count"]
        stage = "full_path_search"
    elif summary.get("arm_shortlist_status") == "static_scene_blocked":
        stage = "static_scene"
    elif counts["local_chain_assessed"]:
        terminal_stages = set(stages) - {"geometric_reach"}
        stage = (next(iter(terminal_stages))
                 if terminal_stages in ({"motion_quality"}, {"endpoint_ik"})
                 else "local_grasp_chain")
    elif (counts["geometric_retained"] == 0
          and summary.get("arm_shortlist_geometric_rejected_count", 0) > 0):
        stage = "geometric_reach"
    elif "arm_shortlist_status" in summary:
        stage = "arm_candidate_selection"
    else:
        stage = "physical_filtering"
    result = {
        "schema": "dexteleop.point_grasp.failure.v1",
        "stage": stage,
        "selection_status": summary.get("arm_shortlist_status"),
        "reason_code": "point_grasp_no_" + stage + "_candidate",
        "scope": ("bounded_full_path_search_not_global_unreachability" if full_path_search
                  else "bounded_proposal_search_not_full_path_or_global_unreachability"),
        "counts": counts,
        "rejections_by_stage": dict(sorted(stages.items())),
        "search_budget_exhausted": {
            "local_chain": summary.get("arm_shortlist_budget_exhausted"),
            "endpoint_probe": summary.get("arm_shortlist_endpoint_probe_budget_exhausted"),
        },
        "same_frozen_pixel_retry_recommended": False,
    }
    if full_path_search:
        native = json.loads(str(summary.get("arm_shortlist_native_path_attempts_json", "[]")))
        result.update(full_path_rejections_by_reason=dict(sorted(rejections.items())),
                      native_path_attempts=native[:4],
                      native_path_attempts_omitted=max(0, len(native)-4))
    return result
