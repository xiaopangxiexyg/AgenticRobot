"""Provisional local models and physical layout for prepared views."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


PREPARED_TRAINING_VIEW_SCHEMA_V1 = (
    "dexteleop.prepared_training_view.provisional.v1"
)
PREPARED_TRAINING_VIEW_MANIFEST_DOMAIN_V1 = (
    "dexteleop.prepared_training_view.manifest.provisional.v1"
)
PREPARED_SOURCE_DATASETS_DOMAIN_V1 = (
    "dexteleop.prepared_training_view.sources.provisional.v1"
)
PREPARED_SELECTED_EPISODE_DOMAIN_V1 = (
    "dexteleop.prepared_training_view.selected_episode.provisional.v1"
)
PREPARED_SOURCE_INVENTORY_DOMAIN_V1 = (
    "dexteleop.prepared_training_view.source_inventory.provisional.v1"
)
PREPARED_TRAINING_VIEW_HASH_DOMAIN_V1 = (
    "dexteleop.prepared_training_view.identity.provisional.v1"
)
PREPARED_SOURCE_EPISODE_HASH_DOMAIN_V1 = (
    "dexteleop.prepared_training_view.source_episode.provisional.v1"
)

VIEW_RELATIVE_ROOT = Path("derived/training_views")
SOURCE_DATASETS_FILE = "source_datasets.cjson"
SELECTED_EPISODES_FILE = "selected_episodes.cjsonl"
EPISODES_FILE = "episodes.parquet"
INTERVALS_FILE = "semantic_intervals.parquet"
ANCHORS_FILE = "semantic_anchors.parquet"
SOURCE_INVENTORY_FILE = "source_inventory.cjsonl"
SKILL_INTERVALS_FILE = "skill_intervals.parquet"
SKILL_PROJECTION_RECEIPTS_FILE = "skill_projection_receipts.cjsonl"
PREPARED_SKILL_PROJECTION_RECEIPTS_DOMAIN_V1 = (
    "dexteleop.prepared_training_view.skill_projection_receipts.provisional.v1"
)
MANIFEST_FILE = "manifest.cjson"
DERIVED_FILES = (
    SOURCE_DATASETS_FILE,
    SELECTED_EPISODES_FILE,
    EPISODES_FILE,
    INTERVALS_FILE,
    ANCHORS_FILE,
    SOURCE_INVENTORY_FILE,
)
ALLOWED_MODALITIES = frozenset({"numeric", "video"})


class PreparedTrainingViewError(ValueError):
    """Raised when prepared-view inputs or persisted content are ambiguous."""


@dataclass(frozen=True)
class PreparedSourceDataset:
    """One explicit immutable source root in a logical organized dataset."""

    source_dataset_id: str
    dataset_root: Path

    def __post_init__(self) -> None:
        source_id = str(self.source_dataset_id).strip()
        if not source_id or any(char.isspace() for char in source_id):
            raise PreparedTrainingViewError(
                "source_dataset_id must be non-empty and contain no whitespace."
            )
        object.__setattr__(self, "source_dataset_id", source_id)
        object.__setattr__(
            self,
            "dataset_root",
            Path(self.dataset_root).expanduser().resolve(),
        )


@dataclass(frozen=True)
class PreparedTrainingViewBuildResult:
    view_root: Path
    training_view_hash: str
    episode_count: int
    interval_count: int
    anchor_count: int
    idempotent: bool
    skill_interval_count: int = 0


@dataclass(frozen=True)
class LoadedPreparedTrainingView:
    view_root: Path
    organized_dataset_root: Path
    training_view_hash: str
    manifest: Mapping[str, Any]
    source_datasets: Mapping[str, Path]
    selected_episodes: tuple[Mapping[str, Any], ...]
    source_inventory: tuple[Mapping[str, Any], ...]
    episodes: Any
    semantic_intervals: Any
    semantic_anchors: Any
    skill_intervals: Any | None = None
    skill_projection_receipts: tuple[Mapping[str, Any], ...] = ()


def arrow_schemas() -> tuple[Any, Any, Any]:
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise PreparedTrainingViewError(
            "pyarrow is required to use prepared training views."
        ) from exc
    episodes = pa.schema(
        [
            ("organized_episode_ordinal", pa.int64()),
            ("source_dataset_id", pa.string()),
            ("source_episode_index", pa.int64()),
            ("episode_uuid", pa.string()),
            ("episode_capture_identity_hash", pa.string()),
            ("source_episode_content_hash", pa.string()),
            ("frame_count", pa.int64()),
            ("clock_domain_id", pa.string()),
            ("start_timestamp", pa.float64()),
            ("end_timestamp", pa.float64()),
            ("parquet_relative_path", pa.string()),
            ("camera_keys", pa.list_(pa.string())),
            ("video_relative_paths", pa.list_(pa.string())),
            ("progress_profile_hash", pa.string()),
            ("stage_profile_hash", pa.string()),
            ("progress_manifest_hash", pa.string()),
            ("annotation_set_id", pa.string()),
            ("annotation_manifest_hash", pa.string()),
            ("annotation_status", pa.string()),
            ("failure_trigger_hash", pa.string()),
        ]
    )
    intervals = pa.schema(
        [
            ("organized_episode_ordinal", pa.int64()),
            ("episode_capture_identity_hash", pa.string()),
            ("sample_begin", pa.int64()),
            ("sample_end_exclusive", pa.int64()),
            ("semantic_kind", pa.string()),
            ("training_domain", pa.string()),
            ("progress_step_id", pa.string()),
            ("attempt_id", pa.string()),
            ("stage_id", pa.string()),
            ("outcome", pa.string()),
            ("entry_condition", pa.string()),
            ("source_id", pa.string()),
            ("source_failure_fact_id", pa.string()),
            ("resolution_kind", pa.string()),
        ]
    )
    anchors = pa.schema(
        [
            ("organized_episode_ordinal", pa.int64()),
            ("episode_capture_identity_hash", pa.string()),
            ("sample_index", pa.int64()),
            ("semantic_kind", pa.string()),
            ("progress_step_id", pa.string()),
            ("attempt_id", pa.string()),
            ("stage_id", pa.string()),
            ("source_id", pa.string()),
            ("source_failure_fact_id", pa.string()),
        ]
    )
    return episodes, intervals, anchors


def skill_interval_arrow_schema() -> Any:
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise PreparedTrainingViewError(
            "pyarrow is required to use prepared training views."
        ) from exc
    return pa.schema(
        [
            ("organized_episode_ordinal", pa.int64()),
            ("episode_capture_identity_hash", pa.string()),
            ("skill_id", pa.string()),
            ("skill_training_contract_hash", pa.string()),
            ("progress_skill_projection_hash", pa.string()),
            ("sample_begin", pa.int64()),
            ("sample_end_exclusive", pa.int64()),
            ("training_domain", pa.string()),
            ("source_attempt_id", pa.string()),
            ("source_stage_ids", pa.list_(pa.string())),
            ("source_occurrence_ids", pa.list_(pa.string())),
            ("source_failure_fact_id", pa.string()),
        ]
    )


__all__ = [
    "ALLOWED_MODALITIES",
    "ANCHORS_FILE",
    "DERIVED_FILES",
    "EPISODES_FILE",
    "INTERVALS_FILE",
    "LoadedPreparedTrainingView",
    "MANIFEST_FILE",
    "PREPARED_SELECTED_EPISODE_DOMAIN_V1",
    "PREPARED_SOURCE_DATASETS_DOMAIN_V1",
    "PREPARED_SOURCE_EPISODE_HASH_DOMAIN_V1",
    "PREPARED_SOURCE_INVENTORY_DOMAIN_V1",
    "PREPARED_TRAINING_VIEW_HASH_DOMAIN_V1",
    "PREPARED_TRAINING_VIEW_MANIFEST_DOMAIN_V1",
    "PREPARED_TRAINING_VIEW_SCHEMA_V1",
    "PREPARED_SKILL_PROJECTION_RECEIPTS_DOMAIN_V1",
    "PreparedSourceDataset",
    "PreparedTrainingViewBuildResult",
    "PreparedTrainingViewError",
    "SELECTED_EPISODES_FILE",
    "SOURCE_DATASETS_FILE",
    "SOURCE_INVENTORY_FILE",
    "SKILL_INTERVALS_FILE",
    "SKILL_PROJECTION_RECEIPTS_FILE",
    "VIEW_RELATIVE_ROOT",
    "arrow_schemas",
    "skill_interval_arrow_schema",
]
