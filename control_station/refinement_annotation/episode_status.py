"""Stable annotation-status projection for Episode lists and other UIs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from collection_runtime.progress.stage_annotation_store import (
    LoadedStageAnnotationSet,
)

EPISODE_STATUS_SCHEMA = "dexteleop.refinement_annotation.episode_status.v1"


@dataclass(frozen=True)
class EpisodeAnnotationSummary:
    annotation_set_count: int
    review_state: str
    latest_annotation_mode: str | None
    whole_episode_abnormal: bool
    whole_episode_abnormal_reason: str | None

    @property
    def needs_second_review(self) -> bool:
        return self.review_state == "PENDING"


def summarize_annotation_sets(
    values: Sequence[LoadedStageAnnotationSet],
) -> EpisodeAnnotationSummary:
    latest = values[-1] if values else None
    latest_mode = None if latest is None else latest.binding.annotation_mode
    needs_review = bool(
        latest is not None
        and latest.bundle.episode_review_request is not None
    )
    review_state = (
        "PENDING"
        if needs_review
        else (
            "COMPLETED"
            if latest_mode == "SECOND_REVIEW"
            else "NONE"
        )
    )
    episode_abnormal = None if latest is None else latest.bundle.episode_abnormal
    return EpisodeAnnotationSummary(
        annotation_set_count=len(values),
        review_state=review_state,
        latest_annotation_mode=latest_mode,
        whole_episode_abnormal=episode_abnormal is not None,
        whole_episode_abnormal_reason=(
            None if episode_abnormal is None else episode_abnormal.reason_code
        ),
    )


def build_episode_status_payload(
    summary: EpisodeAnnotationSummary | None,
    *,
    has_draft: bool,
    detail_state: str | None,
) -> dict[str, Any]:
    """Build an additive, versioned status model while old fields remain valid."""

    badges: list[dict[str, Any]] = []

    def add_badge(code: str, label: str, tone: str, priority: int) -> None:
        badges.append(
            {
                "code": code,
                "label": label,
                "tone": tone,
                "priority": priority,
            }
        )

    if summary is None:
        add_badge("STATUS_LOADING", "状态读取中", "LOADING", 90)
        attention_level = "LOADING"
        attention_required: bool | None = None
        primary_label = "状态读取中"
    else:
        if summary.whole_episode_abnormal:
            label = "整条异常"
            if summary.whole_episode_abnormal_reason:
                label += f" · {summary.whole_episode_abnormal_reason}"
            add_badge("WHOLE_EPISODE_ABNORMAL", label, "CRITICAL", 10)
        if summary.needs_second_review:
            add_badge("SECOND_REVIEW_PENDING", "待二次复核", "WARNING", 20)
        elif summary.review_state == "COMPLETED":
            add_badge("SECOND_REVIEW_COMPLETED", "二次复核完成", "SUCCESS", 50)

        normalized_detail_state = str(detail_state or "").strip().upper()
        if normalized_detail_state == "COMPLETE":
            add_badge("DETAIL_COMPLETE", "详细标注完成", "SUCCESS", 60)
        elif normalized_detail_state == "PENDING":
            add_badge("DETAIL_PENDING", "待详细标注", "INFO", 40)
        elif normalized_detail_state == "STALE":
            add_badge("DETAIL_STALE", "详细标注需重做", "WARNING", 25)
        elif normalized_detail_state == "ERROR":
            add_badge("DETAIL_ERROR", "详细标注状态异常", "CRITICAL", 15)

        if summary.annotation_set_count:
            add_badge(
                "COARSE_SEALED",
                f"已封存 {summary.annotation_set_count} 版",
                "SEALED",
                70,
            )
        else:
            add_badge("COARSE_PENDING", "待粗标注", "INFO", 35)
        if has_draft:
            add_badge("LOCAL_DRAFT", "有未封存草稿", "DRAFT", 30)

        attention_required = bool(
            summary.whole_episode_abnormal
            or summary.needs_second_review
            or normalized_detail_state in {"STALE", "ERROR"}
        )
        if summary.whole_episode_abnormal:
            attention_level = "CRITICAL"
            primary_label = "整条异常"
        elif normalized_detail_state == "ERROR":
            attention_level = "CRITICAL"
            primary_label = "详细标注状态异常"
        elif summary.needs_second_review:
            attention_level = "WARNING"
            primary_label = "待二次复核"
        elif normalized_detail_state == "STALE":
            attention_level = "WARNING"
            primary_label = "详细标注需重做"
        else:
            attention_level = "NONE"
            primary_label = "状态正常"

    badges.sort(key=lambda value: (value["priority"], value["code"]))
    return {
        "schema": EPISODE_STATUS_SCHEMA,
        "loaded": summary is not None,
        "attention_level": attention_level,
        "attention_required": attention_required,
        "primary_label": primary_label,
        "badges": badges,
    }


__all__ = [
    "EPISODE_STATUS_SCHEMA",
    "EpisodeAnnotationSummary",
    "build_episode_status_payload",
    "summarize_annotation_sets",
]
