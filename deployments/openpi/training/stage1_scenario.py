"""Prepare coarse Stage1 scenario data without requiring fine-stage labels."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_interface.prepared_training import PreparedSourceDataset
from dataset_interface.prepared_training.skill_selection import (
    PreparedSkillEligibilityExclusion,
)
from dataset_interface.progress_annotation_jobs import (
    list_progress_annotation_work_items,
    load_episode_sample_continuity,
)
from dataset_interface.progress_stage_annotation_artifacts import (
    STAGE_ANNOTATION_RELATIVE_ROOT,
    LoadedStageAnnotationSet,
    load_stage_annotation_set,
)
from dataset_interface.sample_axis import SampleIndexInterval
from dataset_interface.training_semantics import (
    SEMANTIC_SPAN_ATTEMPT,
    build_episode_training_semantic_index,
)

from .prepared_skill import OpenPIPreparedEpisodeSelection


STAGE1_SCENARIO_SELECTION_POLICY_V1 = (
    "coarse_stage1_nominal_non_abnormal_latest_annotation_v1"
)
_REPORT_SCHEMA_V1 = "dexteleop.openpi_stage1_scenario_eligibility_report.v1"
_REPORT_HASH_DOMAIN_V1 = (
    "dexteleop.openpi_stage1_scenario_eligibility_report.identity.v1"
)
_VIEW_HASH_DOMAIN_V1 = "dexteleop.openpi_stage1_scenario_view.identity.v1"


@dataclass(frozen=True)
class Stage1ScenarioEligibilityReport:
    target_skill_id: str
    target_progress_step_id: str
    included_episode_identities: tuple[str, ...]
    selected_annotation_sets: tuple[tuple[str, str], ...]
    exclusions: tuple[PreparedSkillEligibilityExclusion, ...]
    source_statistics: tuple[Mapping[str, Any], ...]
    semantic_content_hash: str = field(init=False)

    @property
    def annotation_selection(self) -> dict[str, str]:
        return dict(self.selected_annotation_sets)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=_REPORT_HASH_DOMAIN_V1,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": _REPORT_SCHEMA_V1,
            "selection_policy_id": STAGE1_SCENARIO_SELECTION_POLICY_V1,
            "target_skill_id": self.target_skill_id,
            "target_progress_step_id": self.target_progress_step_id,
            "included_episode_identities": list(
                self.included_episode_identities
            ),
            "selected_annotation_sets": [
                {
                    "episode_capture_identity_hash": identity,
                    "annotation_set_id": annotation_set_id,
                }
                for identity, annotation_set_id in self.selected_annotation_sets
            ],
            "exclusions": [item.to_payload() for item in self.exclusions],
            "source_statistics": [dict(item) for item in self.source_statistics],
            "fine_stage_annotations_required": False,
            "unannotated_episodes_allowed": True,
            "annotation_conflict_resolution": "latest_created_at_then_id_v1",
            "whole_or_local_abnormal_annotation_excluded": True,
            "failed_and_paused_progress_regions_excluded": True,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload


def _latest_annotation(
    dataset_root: Path,
    episode_uuid: str,
) -> LoadedStageAnnotationSet | None:
    root = dataset_root / STAGE_ANNOTATION_RELATIVE_ROOT / episode_uuid
    if not root.is_dir():
        return None
    candidates = tuple(
        load_stage_annotation_set(path)
        for path in sorted(child for child in root.iterdir() if child.is_dir())
        if (path / "manifest.cjson").is_file()
    )
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            float(item.binding.created_at),
            item.binding.annotation_set_id,
        ),
    )


def _intersections(
    left: Sequence[SampleIndexInterval],
    right: Sequence[SampleIndexInterval],
    *,
    minimum_length: int,
) -> tuple[tuple[int, int], ...]:
    values: list[tuple[int, int]] = []
    for first in left:
        for second in right:
            begin = max(
                first.sample_index_begin,
                second.sample_index_begin,
            )
            end = min(
                first.sample_index_end_exclusive,
                second.sample_index_end_exclusive,
            )
            if end - begin >= minimum_length:
                values.append((begin, end))
    if not values:
        return ()
    merged: list[tuple[int, int]] = []
    for begin, end in sorted(set(values)):
        if merged and begin <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((begin, end))
    return tuple(merged)


def prepare_stage1_scenario_selections(
    sources: Sequence[PreparedSourceDataset],
    source_records: Sequence[Mapping[str, Any]],
    *,
    target_skill_id: str,
    target_progress_step_id: str,
    action_horizon: int,
) -> tuple[
    Stage1ScenarioEligibilityReport,
    tuple[OpenPIPreparedEpisodeSelection, ...],
    str,
]:
    """Select non-abnormal coarse Stage1 intervals and bind their provenance."""

    records = {
        str(item["source_dataset_id"]): item for item in source_records
    }
    included: list[str] = []
    annotations: list[tuple[str, str]] = []
    exclusions: list[PreparedSkillEligibilityExclusion] = []
    pending: list[
        tuple[PreparedSourceDataset, Any, tuple[tuple[int, int], ...]]
    ] = []
    statistics: list[Mapping[str, Any]] = []
    seen_identities: set[str] = set()

    for source in sorted(sources, key=lambda item: item.source_dataset_id):
        record = records[source.source_dataset_id]
        items = list_progress_annotation_work_items(source.dataset_root)
        raw_frames = sum(item.frame_count for item in items)
        expected_episodes = int(record["expected_raw_episode_count"])
        expected_frames = int(record["expected_raw_frame_count"])
        if len(items) != expected_episodes or raw_frames != expected_frames:
            raise ValueError(
                "Stage1 source inventory differs from the profile: "
                f"source={source.source_dataset_id!r} "
                f"actual=({len(items)}, {raw_frames}) "
                f"expected=({expected_episodes}, {expected_frames})."
            )
        source_included = 0
        source_windows = 0
        for item in items:
            identity = item.episode_identity_hash
            if identity in seen_identities:
                raise ValueError(
                    "Stage1 scenario sources contain duplicate Episode identity: "
                    f"{identity}."
                )
            seen_identities.add(identity)
            annotation = _latest_annotation(
                source.dataset_root,
                item.episode_uuid,
            )
            if annotation is not None:
                annotations.append(
                    (identity, annotation.binding.annotation_set_id)
                )
            reason: str | None = None
            if annotation is not None and annotation.bundle.episode_abnormal is not None:
                reason = "WHOLE_EPISODE_ABNORMAL"
            elif annotation is not None and annotation.bundle.abnormal_intervals:
                reason = "LOCAL_ABNORMAL_INTERVAL"

            intervals: tuple[tuple[int, int], ...] = ()
            if reason is None:
                sample_continuity = load_episode_sample_continuity(item)
                sample_axis = sample_continuity.sample_axis
                semantic = build_episode_training_semantic_index(
                    timeline=item.progress.timeline,
                    sample_axis=sample_axis,
                    continuity_break_indices=(
                        sample_continuity.break_indices
                    ),
                )
                attempts = tuple(
                    span.sample_interval
                    for span in semantic.spans_of_kind(SEMANTIC_SPAN_ATTEMPT)
                    if span.progress_step_id == target_progress_step_id
                )
                intervals = _intersections(
                    attempts,
                    semantic.nominal_intervals,
                    minimum_length=action_horizon,
                )
                if not intervals:
                    reason = "NO_COARSE_STAGE1_NOMINAL_WINDOW"

            if reason is not None:
                exclusions.append(
                    PreparedSkillEligibilityExclusion(
                        source_dataset_id=source.source_dataset_id,
                        source_episode_index=item.episode_index,
                        episode_capture_identity_hash=identity,
                        reason=reason,
                    )
                )
                continue
            included.append(identity)
            source_included += 1
            source_windows += sum(
                end - begin - action_horizon + 1 for begin, end in intervals
            )
            pending.append((source, item, intervals))
        statistics.append(
            {
                "source_dataset_id": source.source_dataset_id,
                "scenario_class": str(record["scenario_class"]),
                "task_scope": str(record["task_scope"]),
                "repeat_weight": int(record["repeat_weight"]),
                "raw_episode_count": len(items),
                "raw_frame_count": raw_frames,
                "included_episode_count": source_included,
                "prepared_interval_window_count": source_windows,
                "prepared_interval_repeated_window_count": (
                    source_windows * int(record["repeat_weight"])
                ),
            }
        )

    ordered = sorted(
        pending,
        key=lambda value: (
            value[0].source_dataset_id,
            value[1].episode_index,
            value[1].episode_identity_hash,
        ),
    )
    selections = tuple(
        OpenPIPreparedEpisodeSelection(
            source_dataset_id=source.source_dataset_id,
            source_dataset_root=source.dataset_root,
            source_episode_index=item.episode_index,
            organized_episode_ordinal=ordinal,
            episode_capture_identity_hash=item.episode_identity_hash,
            intervals=intervals,
        )
        for ordinal, (source, item, intervals) in enumerate(ordered)
    )
    report = Stage1ScenarioEligibilityReport(
        target_skill_id=str(target_skill_id),
        target_progress_step_id=str(target_progress_step_id),
        included_episode_identities=tuple(sorted(included)),
        selected_annotation_sets=tuple(sorted(annotations)),
        exclusions=tuple(
            sorted(
                exclusions,
                key=lambda item: (
                    item.source_dataset_id,
                    item.source_episode_index,
                    item.episode_capture_identity_hash,
                ),
            )
        ),
        source_statistics=tuple(statistics),
    )
    view_payload = {
        "selection_policy_id": STAGE1_SCENARIO_SELECTION_POLICY_V1,
        "eligibility_report_hash": report.semantic_content_hash,
        "target_skill_id": str(target_skill_id),
        "target_progress_step_id": str(target_progress_step_id),
        "action_horizon": int(action_horizon),
        "episodes": [
            {
                "source_dataset_id": item.source_dataset_id,
                "source_episode_index": item.source_episode_index,
                "episode_capture_identity_hash": (
                    item.episode_capture_identity_hash
                ),
                "progress_manifest_hash": next(
                    work.progress_manifest_hash
                    for source, work, _intervals in ordered
                    if source.source_dataset_id == item.source_dataset_id
                    and work.episode_index == item.source_episode_index
                ),
                "intervals": list(item.intervals),
            }
            for item in selections
        ],
        "sources": [dict(item) for item in statistics],
    }
    training_view_hash = canonical_content_hash(
        view_payload,
        domain=_VIEW_HASH_DOMAIN_V1,
    )
    return report, selections, training_view_hash


__all__ = [
    "STAGE1_SCENARIO_SELECTION_POLICY_V1",
    "Stage1ScenarioEligibilityReport",
    "prepare_stage1_scenario_selections",
]
