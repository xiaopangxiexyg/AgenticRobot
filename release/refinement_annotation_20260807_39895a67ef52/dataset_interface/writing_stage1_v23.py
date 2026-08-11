"""Dataset-local admission for retrospective writing-stage1 annotation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .writing_stage1_cohort import (
    WritingStage1CohortPlan,
    WritingStage1EpisodeRef,
)


WRITING_STAGE1_V23_DEFAULT_LAYER_ID = "writing_stage1_retrospective_v1"
_LAYER_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class WritingStage1V23LayerError(ValueError):
    """Raised when a source cannot safely own the requested semantic layer."""


@dataclass(frozen=True)
class WritingStage1V23CampaignBatch:
    batch_id: str
    source_dataset_root: Path
    eligible: tuple[WritingStage1EpisodeRef, ...]
    selected: tuple[WritingStage1EpisodeRef, ...]


def _dataset_schema(info: Mapping[str, Any]) -> str:
    metadata = info.get("metadata")
    candidates = (
        info.get("dexteleop_dataset_schema"),
        info.get("codebase_version"),
        metadata.get("dexteleop_dataset_schema")
        if isinstance(metadata, Mapping)
        else None,
    )
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise WritingStage1V23LayerError(
        "Dataset info does not declare a DexTeleop schema."
    )


def require_writing_stage1_v23_dataset(
    dataset_root: str | Path,
) -> Path:
    """Require one concrete v2.3 dataset without mutating it."""

    root = Path(dataset_root).expanduser().resolve()
    info_path = root / "meta" / "info.json"
    catalog_path = root / "meta" / "episodes.jsonl"
    if not root.is_dir() or not info_path.is_file() or not catalog_path.is_file():
        raise WritingStage1V23LayerError(
            "writing-stage1 annotation requires a v2.3 dataset root with "
            "meta/info.json and meta/episodes.jsonl."
        )
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WritingStage1V23LayerError(
            f"Cannot read dataset info {info_path}: {exc}"
        ) from exc
    if not isinstance(info, Mapping):
        raise WritingStage1V23LayerError("Dataset info must contain an object.")
    schema = _dataset_schema(info)
    if schema not in {"dexteleop_dataset_v2.3", "v2.3"}:
        raise WritingStage1V23LayerError(
            "writing-stage1 retrospective annotation only accepts "
            f"dexteleop_dataset_v2.3; observed={schema!r}."
        )
    return root


def writing_stage1_v23_semantic_layer_root(
    dataset_root: str | Path,
    *,
    semantic_layer_id: str = WRITING_STAGE1_V23_DEFAULT_LAYER_ID,
    require_exists: bool = True,
) -> Path:
    """Resolve the only writable root owned by the v2.3 annotation adapter."""

    root = require_writing_stage1_v23_dataset(dataset_root)
    layer_id = str(semantic_layer_id).strip()
    if not _LAYER_ID_PATTERN.fullmatch(layer_id):
        raise WritingStage1V23LayerError(
            "semantic_layer_id must use only letters, digits, '.', '_' or '-'."
        )
    unresolved_layers_root = root / "semantic_layers"
    if unresolved_layers_root.is_symlink():
        raise WritingStage1V23LayerError(
            "semantic_layers must not be a symbolic link."
        )
    layers_root = unresolved_layers_root.resolve()
    if layers_root.parent != root:
        raise WritingStage1V23LayerError(
            "semantic_layers escaped the source dataset."
        )
    unresolved_layer_root = layers_root / layer_id
    if unresolved_layer_root.is_symlink():
        raise WritingStage1V23LayerError(
            "The selected semantic layer must not be a symbolic link."
        )
    layer_root = unresolved_layer_root.resolve()
    if layer_root.parent != layers_root:
        raise WritingStage1V23LayerError(
            "Semantic layer path escaped the source dataset."
        )
    if require_exists and not layer_root.is_dir():
        raise WritingStage1V23LayerError(
            f"Writing-stage1 semantic layer does not exist: {layer_root}"
        )
    return layer_root


def resolve_writing_stage1_v23_campaign_batches(
    plan: WritingStage1CohortPlan,
    selected: tuple[WritingStage1EpisodeRef, ...],
    *,
    expected_per_batch: int,
) -> tuple[WritingStage1V23CampaignBatch, WritingStage1V23CampaignBatch]:
    """Require exactly one distinct owning dataset for 7.23 and 7.24."""

    if not isinstance(plan, WritingStage1CohortPlan):
        raise TypeError("plan must be WritingStage1CohortPlan.")
    if (
        not isinstance(expected_per_batch, int)
        or isinstance(expected_per_batch, bool)
        or expected_per_batch <= 0
    ):
        raise WritingStage1V23LayerError(
            "expected_per_batch must be a positive integer."
        )
    selected_values = tuple(selected)
    if any(
        not isinstance(value, WritingStage1EpisodeRef)
        for value in selected_values
    ):
        raise WritingStage1V23LayerError(
            "selected must contain WritingStage1EpisodeRef values."
        )

    specs = (
        ("2026-07-23", plan.front_0723),
        ("2026-07-24", plan.three_view_0724),
    )
    results: list[WritingStage1V23CampaignBatch] = []
    roots: list[Path] = []
    selected_identities = {
        value.episode_identity_hash for value in selected_values
    }
    if len(selected_identities) != len(selected_values):
        raise WritingStage1V23LayerError(
            "Pilot selection contains duplicate Episode identities."
        )
    for batch_id, eligible in specs:
        batch_roots = {
            Path(value.source_dataset_root).expanduser().resolve()
            for value in eligible
        }
        if len(batch_roots) != 1:
            raise WritingStage1V23LayerError(
                f"{batch_id} must resolve exactly one source dataset root; "
                f"observed={sorted(str(value) for value in batch_roots)!r}."
            )
        root = next(iter(batch_roots))
        batch_selected = tuple(
            value for value in selected_values if value.batch_id == batch_id
        )
        if len(batch_selected) != expected_per_batch:
            raise WritingStage1V23LayerError(
                f"{batch_id} must select exactly {expected_per_batch} pilot "
                f"Episodes; observed={len(batch_selected)}."
            )
        eligible_identities = {
            value.episode_identity_hash for value in eligible
        }
        if any(
            value.episode_identity_hash not in eligible_identities
            or Path(value.source_dataset_root).expanduser().resolve() != root
            for value in batch_selected
        ):
            raise WritingStage1V23LayerError(
                f"{batch_id} pilot selection is not a subset of its dataset."
            )
        roots.append(root)
        results.append(
            WritingStage1V23CampaignBatch(
                batch_id=batch_id,
                source_dataset_root=root,
                eligible=eligible,
                selected=batch_selected,
            )
        )
    if len(set(roots)) != 2:
        raise WritingStage1V23LayerError(
            "7.23 and 7.24 must own two distinct dataset-local semantic layers."
        )
    if selected_identities != {
        value.episode_identity_hash
        for result in results
        for value in result.selected
    }:
        raise WritingStage1V23LayerError(
            "Pilot selection contains an unknown writing batch."
        )
    return (results[0], results[1])


__all__ = [
    "WRITING_STAGE1_V23_DEFAULT_LAYER_ID",
    "WritingStage1V23CampaignBatch",
    "WritingStage1V23LayerError",
    "require_writing_stage1_v23_dataset",
    "resolve_writing_stage1_v23_campaign_batches",
    "writing_stage1_v23_semantic_layer_root",
]
