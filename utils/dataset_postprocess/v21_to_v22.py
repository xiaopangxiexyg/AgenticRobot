"""Copy-on-write conversion from legacy fixed-q6 rows to DexTeleop v2.2.

The converter retains LeRobot-v2.1-style directory layout and all legacy
columns.  It never mutates the source dataset and refuses to write into an
existing target root, so an interrupted conversion cannot damage collection
data or a known-good dataset.
"""

from __future__ import annotations

import json
import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from dataset_contracts.parquet import write_rows_to_parquet
from dataset_contracts.v21_compat import v21_info_to_v22_info, v21_row_to_v22_row
from dataset_contracts.v22 import DEXTELEOP_DATASET_SCHEMA_V22, PolicyIOLayout, bind_policy_io_layout

from .io import (
    clone_dataset_tree,
    episode_records,
    read_json,
    read_parquet_rows,
    write_json,
    write_jsonl,
)


@dataclass(frozen=True)
class V21ToV22ConversionPlan:
    source_root: Path
    output_root: Path
    layout_hash: str
    source_schema: str
    episode_count: int
    frame_count: int | None
    parquet_count: int
    source_dataset_id: str | None
    source_info_sha256: str
    conversion_id: str
    parquet_relative_paths: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": "v21_to_v22_policy_io",
            "source_root": str(self.source_root),
            "output_root": str(self.output_root),
            "source_schema": self.source_schema,
            "target_schema": DEXTELEOP_DATASET_SCHEMA_V22,
            "policy_io_layout_hash": self.layout_hash,
            "episode_count": int(self.episode_count),
            "frame_count": self.frame_count,
            "parquet_count": int(self.parquet_count),
            "source_dataset_id": self.source_dataset_id,
            "source_info_sha256": self.source_info_sha256,
            "conversion_id": self.conversion_id,
            "parquet_relative_paths": list(self.parquet_relative_paths),
            "source_mutated": False,
            "copy_strategy": "metadata_and_parquet_copy; external_artifacts_hardlink_or_copy",
        }


def load_v22_layout_json(path: str | Path) -> PolicyIOLayout:
    """Load a layout object or metadata wrapper from JSON."""

    layout_path = Path(path).expanduser().resolve()
    payload = json.loads(layout_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Layout JSON must contain an object: {layout_path}")
    nested = payload.get("policy_io_layout")
    layout_payload = nested if nested is not None else payload
    if not isinstance(layout_payload, Mapping):
        raise ValueError(f"policy_io_layout in {layout_path} must contain an object.")
    layout = PolicyIOLayout.from_dict(layout_payload)
    declared_hash = payload.get("policy_io_layout_hash") or layout_payload.get("layout_hash")
    if declared_hash is not None and str(declared_hash) != layout.layout_hash:
        raise ValueError(
            f"Layout JSON hash mismatch: expected {layout.layout_hash}, got {declared_hash}."
        )
    return layout


def plan_v21_to_v22_conversion(
    source_root: str | Path,
    output_root: str | Path,
    layout: PolicyIOLayout,
) -> V21ToV22ConversionPlan:
    source = Path(source_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source dataset root does not exist: {source}")
    if output == source:
        raise ValueError("output_root must differ from source_root.")
    try:
        output.relative_to(source)
    except ValueError:
        pass
    else:
        raise ValueError("output_root must not be inside source_root.")
    info_path = source / "meta" / "info.json"
    if not info_path.is_file() or info_path.is_symlink():
        raise ValueError(f"Source must contain a regular meta/info.json: {info_path}")
    source_info_bytes = info_path.read_bytes()
    info = read_json(info_path)
    if not info:
        raise ValueError(f"Source meta/info.json must contain a non-empty JSON object: {info_path}")
    source_schema = str(info.get("dexteleop_dataset_schema") or "legacy_v21_or_unversioned")
    if source_schema == DEXTELEOP_DATASET_SCHEMA_V22:
        existing_hash = info.get("policy_io_layout_hash")
        if str(existing_hash or "") == layout.layout_hash:
            raise ValueError("Source is already v2.2 with the requested policy I/O layout.")
        raise ValueError(
            "Source is already v2.2 with a different layout; use a dedicated v2.2-to-v2.2 migration."
        )
    frame_count_raw = info.get("frame_count")
    try:
        frame_count = int(frame_count_raw) if frame_count_raw is not None else None
    except (TypeError, ValueError):
        frame_count = None
    records = episode_records(source)
    relative_paths: list[Path] = []
    seen_episode_indices: set[int] = set()
    seen_paths: set[Path] = set()
    for record in records:
        try:
            episode_index = int(record["episode_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Episode record has invalid episode_index: {record!r}") from exc
        if episode_index in seen_episode_indices:
            raise ValueError(f"Duplicate episode_index in source metadata: {episode_index}")
        seen_episode_indices.add(episode_index)
        relative = _relative_parquet_path(source, record)
        if relative in seen_paths:
            raise ValueError(f"Multiple episodes reference the same parquet_path: {relative}")
        seen_paths.add(relative)
        parquet_path = source / relative
        if not parquet_path.is_file() or parquet_path.is_symlink():
            raise ValueError(
                f"Episode {episode_index} parquet_path is missing, non-regular, or a symlink: "
                f"{parquet_path}"
            )
        relative_paths.append(relative)
    declared_episode_count = info.get("episode_count")
    if declared_episode_count is not None and int(declared_episode_count) != len(records):
        raise ValueError(
            "meta/info.json episode_count disagrees with episodes.jsonl: "
            f"declared={declared_episode_count}, actual={len(records)}"
        )
    source_info_sha256 = "sha256:" + hashlib.sha256(source_info_bytes).hexdigest()
    source_dataset_id_raw = info.get("dataset_id") or info.get("repo_id")
    source_dataset_id = (
        None if source_dataset_id_raw is None else str(source_dataset_id_raw).strip() or None
    )
    conversion_payload = {
        "source_info_sha256": source_info_sha256,
        "policy_io_layout_hash": layout.layout_hash,
        "episode_indices": sorted(seen_episode_indices),
        "parquet_relative_paths": [path.as_posix() for path in relative_paths],
    }
    conversion_id = "sha256:" + hashlib.sha256(
        json.dumps(conversion_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return V21ToV22ConversionPlan(
        source_root=source,
        output_root=output,
        layout_hash=layout.layout_hash,
        source_schema=source_schema,
        episode_count=len(records),
        frame_count=frame_count,
        parquet_count=len(relative_paths),
        source_dataset_id=source_dataset_id,
        source_info_sha256=source_info_sha256,
        conversion_id=conversion_id,
        parquet_relative_paths=tuple(path.as_posix() for path in relative_paths),
    )


def _relative_parquet_path(source_root: Path, record: Mapping[str, Any]) -> Path:
    value = record.get("parquet_path")
    if not isinstance(value, str) or not value:
        raise ValueError(f"Episode {record.get('episode_index')} has no parquet_path.")
    declared = Path(value)
    if declared.is_absolute():
        try:
            relative = declared.expanduser().resolve().relative_to(source_root)
        except ValueError as exc:
            raise ValueError(
                "Refusing to convert a parquet path outside source_root: "
                f"episode={record.get('episode_index')}, parquet_path={declared}"
            ) from exc
    else:
        relative = declared
    if not relative.parts or ".." in relative.parts:
        raise ValueError(
            f"Episode {record.get('episode_index')} has unsafe parquet_path={value!r}."
        )
    return relative


def _bind_episode_record(
    record: Mapping[str, Any],
    layout: PolicyIOLayout,
    *,
    parquet_relative_path: Path,
) -> dict[str, Any]:
    output = dict(record)
    metadata = output.get("metadata")
    episode_metadata = {} if not isinstance(metadata, Mapping) else dict(metadata)
    episode_metadata["dexteleop_dataset_schema"] = DEXTELEOP_DATASET_SCHEMA_V22
    episode_metadata.update(bind_policy_io_layout(None, layout))
    output["metadata"] = episode_metadata
    output["parquet_path"] = parquet_relative_path.as_posix()
    return output


def convert_v21_dataset_to_v22(
    source_root: str | Path,
    output_root: str | Path,
    layout: PolicyIOLayout,
    *,
    hardlink_external: bool = True,
) -> V21ToV22ConversionPlan:
    """Create a new v2.2 dataset root from a legacy source, copy-on-write.

    ``output_root`` must not exist.  A sibling staging directory is promoted
    only after all parquet and metadata output succeeds; the source stays
    untouched and any failed staging directory is deliberately retained for
    inspection instead of being removed automatically.
    """

    plan = plan_v21_to_v22_conversion(source_root, output_root, layout)
    if plan.output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing conversion output: {plan.output_root}. "
            "Choose a new output root or inspect/remove it explicitly."
        )
    staging = plan.output_root.parent / f".{plan.output_root.name}.v22-staging-{uuid.uuid4().hex}"
    if staging.exists():
        raise FileExistsError(f"Unexpected existing conversion staging path: {staging}")

    clone_dataset_tree(
        plan.source_root,
        staging,
        copy_prefixes=("meta", "data"),
        hardlink_external=bool(hardlink_external),
    )
    records = episode_records(plan.source_root)
    relative_parquet_paths = [Path(value) for value in plan.parquet_relative_paths]
    for record, relative_parquet_path in zip(records, relative_parquet_paths, strict=True):
        source_parquet = plan.source_root / relative_parquet_path
        target_parquet = staging / relative_parquet_path
        rows = read_parquet_rows(source_parquet)
        converted_rows = [v21_row_to_v22_row(row, layout) for row in rows]
        write_rows_to_parquet(
            converted_rows,
            target_parquet,
            fixed_size_list_columns=layout.parquet_fixed_list_columns(),
        )

    source_info = read_json(plan.source_root / "meta" / "info.json")
    provenance = {
        "conversion_id": plan.conversion_id,
        "source_dataset_id": plan.source_dataset_id,
        "source_info_sha256": plan.source_info_sha256,
        "source_schema": plan.source_schema,
        "source_mutated": False,
    }
    write_json(
        staging / "meta" / "info.json",
        v21_info_to_v22_info(
            source_info,
            layout,
            conversion_provenance=provenance,
        ),
    )
    write_json(staging / "meta" / "v21_to_v22_conversion.json", plan.to_dict())
    write_jsonl(
        staging / "meta" / "episodes.jsonl",
        [
            _bind_episode_record(
                record,
                layout,
                parquet_relative_path=relative_parquet_path,
            )
            for record, relative_parquet_path in zip(records, relative_parquet_paths)
        ],
    )
    staging.rename(plan.output_root)
    return plan


__all__ = [
    "V21ToV22ConversionPlan",
    "convert_v21_dataset_to_v22",
    "load_v22_layout_json",
    "plan_v21_to_v22_conversion",
]
