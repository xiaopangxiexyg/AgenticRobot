from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


EPISODE_ARTIFACT_KEYS = (
    "parquet_path",
    "video_paths",
    "tactile_paths",
    "wrist_depth_mask_paths",
    "wrist_depth_raw_paths",
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            records.append(json.loads(text))
    return records


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    jsonl_path = Path(path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(dict(record), ensure_ascii=True, sort_keys=True) for record in records]
    jsonl_path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        return {}
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(dict(payload), ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def read_parquet_rows(path: str | Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Dataset postprocess parquet scans require pyarrow.") from exc
    table = pq.read_table(path)
    return table.to_pylist()


def read_parquet_table(path: str | Path) -> Any:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Dataset postprocess parquet scans require pyarrow.") from exc
    return pq.read_table(path)


def write_parquet_table(path: str | Path, table: Any, *, compression: str = "zstd") -> None:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Dataset postprocess parquet writes require pyarrow.") from exc
    parquet_path = Path(path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, parquet_path, compression=str(compression))


def _copy_or_hardlink_file(source: Path, target: Path, *, hardlink: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if hardlink:
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def clone_dataset_tree(
    source_root: str | Path,
    output_root: str | Path,
    *,
    copy_prefixes: Iterable[str] = ("meta",),
    skip_prefixes: Iterable[str] = (),
    hardlink_external: bool = True,
) -> None:
    """Clone a dataset tree without duplicating large immutable artifacts.

    Prefixes in ``copy_prefixes`` are byte-copied because callers commonly
    rewrite those files in the derived dataset. All other files are hard-linked
    by default, with a copy fallback for cross-filesystem outputs. Do not put a
    prefix in the hardlink path if the caller may rewrite files there.
    """

    source = Path(source_root).resolve()
    output = Path(output_root).resolve()
    try:
        output.relative_to(source)
    except ValueError:
        pass
    else:
        raise ValueError(f"output_root must not be inside source_root: {output}")
    copy_set = {str(value).strip("/") for value in copy_prefixes if str(value).strip("/")}
    skip_set = {str(value).strip("/") for value in skip_prefixes if str(value).strip("/")}
    output.mkdir(parents=True, exist_ok=True)
    for source_path in source.rglob("*"):
        if not source_path.is_file():
            continue
        relative_path = source_path.relative_to(source)
        top = relative_path.parts[0] if relative_path.parts else ""
        if top in skip_set:
            continue
        must_copy = top in copy_set
        _copy_or_hardlink_file(
            source_path,
            output / relative_path,
            hardlink=bool(hardlink_external and not must_copy),
        )


def episode_records(dataset_root: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(Path(dataset_root) / "meta" / "episodes.jsonl")


def episode_stats_records(dataset_root: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(Path(dataset_root) / "meta" / "episodes_stats.jsonl")


def episode_artifact_paths(record: Mapping[str, Any]) -> tuple[str, ...]:
    paths: list[str] = []
    for key in EPISODE_ARTIFACT_KEYS:
        value = record.get(key)
        if isinstance(value, str):
            paths.append(value)
        elif isinstance(value, Mapping):
            paths.extend(str(item) for item in value.values() if item)
    return tuple(paths)


def parquet_path_for_record(dataset_root: str | Path, record: Mapping[str, Any]) -> Path:
    path = record.get("parquet_path")
    if not isinstance(path, str) or not path:
        raise ValueError(f"Episode {record.get('episode_index')} has no parquet_path.")
    return Path(dataset_root) / path
