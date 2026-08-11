"""Read-only projection of an admitted Human residual for operator display.

The projection consumes the exact residual action already admitted by the
allocation input path.  It does not read a device, reconstruct an anchor,
advance composition state, or influence control.
"""

from __future__ import annotations

import math
from typing import Any

from control_composition.observability import project_relative_pose_deflections


HUMAN_DEFLECTION_STATUS_SCHEMA_V1 = (
    "dexteleop.control_station.human_deflection_status.v1"
)


def _empty_status(
    status: str,
    message: str,
    *,
    input_publication_status: str | None = None,
) -> dict[str, Any]:
    return {
        "schema": HUMAN_DEFLECTION_STATUS_SCHEMA_V1,
        "status": status,
        "message": message,
        "input_publication_status": input_publication_status,
        "source_sample_id": None,
        "source_sample_hash": None,
        "anchor_receipt_id": None,
        "anchor_receipt_hash": None,
        "anchor_request_generation": None,
        "sampled_at": None,
        "age_sec": None,
        "components": [],
    }


def _allocation_input_publication(cycle: Any) -> Any | None:
    publication = getattr(cycle, "publication", None)
    if publication is not None:
        value = getattr(publication, "input_publication", None)
        if value is not None:
            return value
    return getattr(cycle, "input_publication", None)


def project_admitted_human_deflection(
    cycle: Any | None,
    *,
    now: float,
) -> dict[str, Any]:
    """Describe pose deflection from the active residual Home.

    Only an admitted ``relative_delta`` action is projected.  A raw tracker
    pose or a merely prepared derivation is intentionally insufficient: using
    either could display a different Home generation from the one composition
    is currently allowed to consume.
    """

    if cycle is None:
        return _empty_status("waiting", "等待已接纳的 Human relative 输入。")
    input_publication = _allocation_input_publication(cycle)
    if input_publication is None:
        return _empty_status(
            "unavailable",
            "当前控制方案未发布 relative 输入事实。",
        )
    publication_status = str(
        getattr(input_publication, "status", "") or ""
    )
    pair = getattr(input_publication, "evidence_pair", None)
    if publication_status != "batch_admitted" or pair is None:
        return _empty_status(
            "waiting",
            "等待下一份已接纳的 Human relative 输入。",
            input_publication_status=publication_status or None,
        )

    action = getattr(pair, "residual_action", None)
    try:
        (
            components,
            invalid_components,
            omitted_components,
        ) = project_relative_pose_deflections(action)
    except (TypeError, ValueError):
        return _empty_status(
            "invalid",
            "已接纳的 residual 输入缺少动作或 schema。",
            input_publication_status=publication_status,
        )

    sampled_at = getattr(pair, "sampled_at", None)
    try:
        sampled = None if sampled_at is None else float(sampled_at)
        current = float(now)
        if not math.isfinite(current) or (
            sampled is not None and not math.isfinite(sampled)
        ):
            raise ValueError
        age_sec = None if sampled is None else max(0.0, current - sampled)
    except (TypeError, ValueError):
        sampled = None
        age_sec = None

    if invalid_components:
        status = "invalid"
        message = "部分 relative pose 无法显示。"
    elif components:
        status = "available"
        message = "显示的是已接纳 residual 相对当前 anchor 的偏转。"
    else:
        status = "unavailable"
        message = "当前 relative 输入不包含 pose component。"
    return {
        "schema": HUMAN_DEFLECTION_STATUS_SCHEMA_V1,
        "status": status,
        "message": message,
        "input_publication_status": publication_status,
        "source_sample_id": getattr(pair, "source_sample_id", None),
        "source_sample_hash": getattr(pair, "source_sample_hash", None),
        "anchor_receipt_id": getattr(pair, "anchor_receipt_id", None),
        "anchor_receipt_hash": getattr(pair, "anchor_receipt_hash", None),
        "anchor_request_generation": getattr(
            pair,
            "anchor_request_generation",
            None,
        ),
        "sampled_at": sampled,
        "age_sec": age_sec,
        "components": components,
        "invalid_components": invalid_components,
        "omitted_components": omitted_components,
    }


__all__ = [
    "HUMAN_DEFLECTION_STATUS_SCHEMA_V1",
    "project_admitted_human_deflection",
]
