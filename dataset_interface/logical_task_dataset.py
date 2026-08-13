"""Versioned, read-only organization of immutable dataset shards by task."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from dataset_contracts.canonical_encoding import canonical_content_hash

from .episode_curation import source_dataset_base_fingerprint


LOGICAL_TASK_DATASET_SCHEMA_V1 = "dexteleop.logical_task_dataset.v1"
LOGICAL_TASK_DATASET_HASH_DOMAIN_V1 = (
    "dexteleop.logical_task_dataset.semantic.v1"
)
LOGICAL_TASK_DATASET_POLICY_V1 = (
    "canonical_task_alias_preserve_immutable_source_identity_v1"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class LogicalTaskDatasetError(ValueError):
    """Raised when a logical task-dataset manifest is invalid or stale."""


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LogicalTaskDatasetError(f"{name} must be a non-empty string.")
    return value.strip()


def _count(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise LogicalTaskDatasetError(f"{name} must be a non-negative integer.")
    return value


@dataclass(frozen=True)
class LogicalTaskDatasetSource:
    source_dataset_id: str
    dataset_root: Path
    source_task_id: str
    source_dataset_fingerprint: str
    scenario_class: str
    task_scope: str
    raw_episode_count: int
    raw_frame_count: int


@dataclass(frozen=True)
class LogicalTaskDatasetManifest:
    manifest_path: Path
    logical_dataset_id: str
    logical_dataset_version: str
    canonical_task_id: str
    canonical_task_description: str
    progress_step_id: str
    source_identity_policy: str
    scenario_classes: Mapping[str, str]
    sources: tuple[LogicalTaskDatasetSource, ...]
    semantic_content_hash: str


def logical_task_dataset_hash(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("semantic_content_hash", None)
    return canonical_content_hash(
        value,
        domain=LOGICAL_TASK_DATASET_HASH_DOMAIN_V1,
    )


def load_logical_task_dataset_manifest(
    manifest_path: str | Path,
) -> LogicalTaskDatasetManifest:
    path = Path(manifest_path).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LogicalTaskDatasetError(
            f"Cannot read logical task-dataset manifest: {path}."
        ) from exc
    if not isinstance(payload, Mapping):
        raise LogicalTaskDatasetError("Logical task-dataset manifest must be an object.")
    if payload.get("schema") != LOGICAL_TASK_DATASET_SCHEMA_V1:
        raise LogicalTaskDatasetError("Unsupported logical task-dataset schema.")
    if payload.get("source_identity_policy") != LOGICAL_TASK_DATASET_POLICY_V1:
        raise LogicalTaskDatasetError("Unsupported source identity policy.")

    classes_value = payload.get("scenario_classes")
    if not isinstance(classes_value, list) or not classes_value:
        raise LogicalTaskDatasetError("scenario_classes must be a non-empty list.")
    scenario_classes: dict[str, str] = {}
    for index, item in enumerate(classes_value):
        if not isinstance(item, Mapping):
            raise LogicalTaskDatasetError(
                f"scenario_classes[{index}] must be an object."
            )
        class_id = _text(item.get("scenario_class"), name="scenario_class")
        if class_id in scenario_classes:
            raise LogicalTaskDatasetError(
                f"Duplicate scenario_class: {class_id!r}."
            )
        scenario_classes[class_id] = _text(
            item.get("description"),
            name=f"scenario_classes[{index}].description",
        )

    source_values = payload.get("sources")
    if not isinstance(source_values, list) or not source_values:
        raise LogicalTaskDatasetError("sources must be a non-empty list.")
    sources: list[LogicalTaskDatasetSource] = []
    source_ids: set[str] = set()
    roots: set[Path] = set()
    for index, item in enumerate(source_values):
        if not isinstance(item, Mapping):
            raise LogicalTaskDatasetError(f"sources[{index}] must be an object.")
        source_id = _text(item.get("source_dataset_id"), name="source_dataset_id")
        if any(char.isspace() for char in source_id) or source_id in source_ids:
            raise LogicalTaskDatasetError(
                f"source_dataset_id must be unique and whitespace-free: {source_id!r}."
            )
        root = Path(_text(item.get("dataset_root"), name="dataset_root")).expanduser()
        if not root.is_absolute():
            raise LogicalTaskDatasetError(
                f"dataset_root must be absolute: {root}."
            )
        root = root.resolve()
        if root in roots:
            raise LogicalTaskDatasetError(f"Duplicate dataset_root: {root}.")
        scenario_class = _text(item.get("scenario_class"), name="scenario_class")
        if scenario_class not in scenario_classes:
            raise LogicalTaskDatasetError(
                f"Unknown scenario_class {scenario_class!r} for {source_id!r}."
            )
        task_scope = _text(item.get("task_scope"), name="task_scope")
        if (
            scenario_class in {"initial_multi_choice", "midcourse_interrupt"}
            and task_scope != "stage1_only"
        ):
            raise LogicalTaskDatasetError(
                f"{scenario_class} source {source_id!r} must be stage1_only."
            )
        fingerprint = _text(
            item.get("source_dataset_fingerprint"),
            name="source_dataset_fingerprint",
        )
        if not _SHA256_RE.fullmatch(fingerprint):
            raise LogicalTaskDatasetError(
                f"Invalid source_dataset_fingerprint for {source_id!r}."
            )
        sources.append(
            LogicalTaskDatasetSource(
                source_dataset_id=source_id,
                dataset_root=root,
                source_task_id=_text(item.get("source_task_id"), name="source_task_id"),
                source_dataset_fingerprint=fingerprint,
                scenario_class=scenario_class,
                task_scope=task_scope,
                raw_episode_count=_count(
                    item.get("raw_episode_count"),
                    name="raw_episode_count",
                ),
                raw_frame_count=_count(
                    item.get("raw_frame_count"),
                    name="raw_frame_count",
                ),
            )
        )
        source_ids.add(source_id)
        roots.add(root)

    totals = payload.get("totals")
    if not isinstance(totals, Mapping):
        raise LogicalTaskDatasetError("totals must be an object.")
    actual_by_class = {
        class_id: {
            "source_count": sum(item.scenario_class == class_id for item in sources),
            "raw_episode_count": sum(
                item.raw_episode_count
                for item in sources
                if item.scenario_class == class_id
            ),
            "raw_frame_count": sum(
                item.raw_frame_count
                for item in sources
                if item.scenario_class == class_id
            ),
        }
        for class_id in scenario_classes
    }
    actual_totals = {
        "source_count": len(sources),
        "raw_episode_count": sum(item.raw_episode_count for item in sources),
        "raw_frame_count": sum(item.raw_frame_count for item in sources),
        "by_scenario_class": actual_by_class,
    }
    if totals != actual_totals:
        raise LogicalTaskDatasetError(
            f"Declared totals differ from sources: declared={totals!r} "
            f"actual={actual_totals!r}."
        )

    declared_hash = _text(
        payload.get("semantic_content_hash"),
        name="semantic_content_hash",
    )
    actual_hash = logical_task_dataset_hash(payload)
    if declared_hash != actual_hash:
        raise LogicalTaskDatasetError(
            "Logical task-dataset semantic_content_hash mismatch."
        )
    return LogicalTaskDatasetManifest(
        manifest_path=path,
        logical_dataset_id=_text(
            payload.get("logical_dataset_id"),
            name="logical_dataset_id",
        ),
        logical_dataset_version=_text(
            payload.get("logical_dataset_version"),
            name="logical_dataset_version",
        ),
        canonical_task_id=_text(
            payload.get("canonical_task_id"),
            name="canonical_task_id",
        ),
        canonical_task_description=_text(
            payload.get("canonical_task_description"),
            name="canonical_task_description",
        ),
        progress_step_id=_text(
            payload.get("progress_step_id"),
            name="progress_step_id",
        ),
        source_identity_policy=LOGICAL_TASK_DATASET_POLICY_V1,
        scenario_classes=dict(scenario_classes),
        sources=tuple(sources),
        semantic_content_hash=actual_hash,
    )


def audit_logical_task_dataset_sources(
    manifest: LogicalTaskDatasetManifest,
) -> dict[str, Any]:
    """Verify pinned source metadata without modifying any source file."""

    results: list[dict[str, Any]] = []
    for source in manifest.sources:
        errors: list[str] = []
        info_path = source.dataset_root / "meta" / "info.json"
        tasks_path = source.dataset_root / "meta" / "tasks.jsonl"
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            task_rows = [
                json.loads(line)
                for line in tasks_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"SOURCE_METADATA_UNREADABLE:{type(exc).__name__}")
            info = {}
            task_rows = []
        if not isinstance(info, Mapping):
            errors.append("INFO_NOT_OBJECT")
            info = {}
        actual_task_id = str(info.get("task_id", "")).strip()
        if actual_task_id != source.source_task_id:
            errors.append("SOURCE_TASK_ID_MISMATCH")
        task_ids = tuple(
            str(row.get("task_id", "")).strip()
            for row in task_rows
            if isinstance(row, Mapping)
        )
        if not task_rows or len(task_ids) != len(task_rows):
            errors.append("TASK_MANIFEST_INVALID")
        elif any(task_id != source.source_task_id for task_id in task_ids):
            errors.append("TASK_MANIFEST_ID_MISMATCH")
        if info.get("episode_count") != source.raw_episode_count:
            errors.append("RAW_EPISODE_COUNT_MISMATCH")
        if info.get("frame_count") != source.raw_frame_count:
            errors.append("RAW_FRAME_COUNT_MISMATCH")
        try:
            fingerprint = source_dataset_base_fingerprint(source.dataset_root)
        except (OSError, ValueError) as exc:
            errors.append(f"SOURCE_FINGERPRINT_UNREADABLE:{type(exc).__name__}")
            fingerprint = ""
        if fingerprint != source.source_dataset_fingerprint:
            errors.append("SOURCE_DATASET_FINGERPRINT_MISMATCH")
        results.append(
            {
                "source_dataset_id": source.source_dataset_id,
                "scenario_class": source.scenario_class,
                "task_scope": source.task_scope,
                "source_task_id": source.source_task_id,
                "source_dataset_fingerprint": fingerprint,
                "ok": not errors,
                "errors": errors,
            }
        )
    failed = sum(not item["ok"] for item in results)
    return {
        "schema": "dexteleop.logical_task_dataset.audit.v1",
        "logical_dataset_id": manifest.logical_dataset_id,
        "logical_dataset_version": manifest.logical_dataset_version,
        "canonical_task_id": manifest.canonical_task_id,
        "semantic_content_hash": manifest.semantic_content_hash,
        "source_count": len(results),
        "failed_source_count": failed,
        "ok": failed == 0,
        "sources": results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit a versioned logical task-dataset manifest read-only."
    )
    parser.add_argument("command", choices=("audit", "hash"))
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)
    path = Path(args.manifest).expanduser().resolve()
    if args.command == "hash":
        payload = json.loads(path.read_text(encoding="utf-8"))
        print(logical_task_dataset_hash(payload))
        return 0
    report = audit_logical_task_dataset_sources(
        load_logical_task_dataset_manifest(path)
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LOGICAL_TASK_DATASET_HASH_DOMAIN_V1",
    "LOGICAL_TASK_DATASET_POLICY_V1",
    "LOGICAL_TASK_DATASET_SCHEMA_V1",
    "LogicalTaskDatasetError",
    "LogicalTaskDatasetManifest",
    "LogicalTaskDatasetSource",
    "audit_logical_task_dataset_sources",
    "load_logical_task_dataset_manifest",
    "logical_task_dataset_hash",
]
