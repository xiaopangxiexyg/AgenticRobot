from __future__ import annotations

import json
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
