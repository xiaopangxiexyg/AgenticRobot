"""Advisory full-path request order, separate from proposal scores and ordinals."""

from __future__ import annotations

import json
from collections.abc import Mapping

GRASP_PLANNING_ORDER_FIELD = "arm_shortlist_planning_order_json"
GRASP_PLANNING_ORDER_SCOPE = "selector_preference_not_path_or_execution_authority"


def grasp_planning_candidate_ordinals(
    filter_summary: Mapping[str, object], *, candidate_count: int,
) -> tuple[int, ...] | None:
    """Read an exact permutation of the published candidates, never a permit.

    Older producers and unembodied perception have no recommendation. A present
    but corrupt recommendation must not silently become a score-order fallback.
    Consumers must bind the summary to its original capture/result first.
    """
    if type(candidate_count) is not int or candidate_count < 0:
        raise ValueError("candidate_count must be a nonnegative integer")
    if GRASP_PLANNING_ORDER_FIELD not in filter_summary:
        return None
    if filter_summary.get("arm_shortlist_planning_order_scope") != GRASP_PLANNING_ORDER_SCOPE:
        raise ValueError("Grasp planning priority has an unsupported scope")
    encoded = filter_summary[GRASP_PLANNING_ORDER_FIELD]
    if not isinstance(encoded, str):
        raise TypeError("Grasp planning priority must be JSON text")
    order = json.loads(encoded)
    if (not isinstance(order, list)
            or any(type(index) is not int for index in order)
            or sorted(order) != list(range(candidate_count))):
        raise ValueError("Grasp planning priority must cover each published ordinal exactly once")
    return tuple(order)
