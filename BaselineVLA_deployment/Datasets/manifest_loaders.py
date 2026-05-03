from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .base import DatasetManifest
from .config_utils import get_positive_limit
from utils.path_config import load_local_env, optional_repo_path


load_local_env()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _parse_range_spec(spec: str, total_episodes: int) -> List[int]:
    indices: List[int] = []
    for chunk in str(spec).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            start_text, end_text = chunk.split(":", 1)
            start = int(start_text) if start_text else 0
            end = int(end_text) if end_text else total_episodes
            indices.extend(range(max(0, start), min(total_episodes, end)))
        else:
            index = int(chunk)
            if 0 <= index < total_episodes:
                indices.append(index)
    return sorted(set(indices))


def _build_split_mapping(info: Dict[str, Any], episode_records: List[Dict[str, Any]], derived_val_ratio: float) -> Dict[str, List[int]]:
    total_episodes = int(info.get("total_episodes", len(episode_records)) or len(episode_records))
    split_mapping: Dict[str, List[int]] = {}
    for split_name, split_spec in (info.get("splits") or {}).items():
        split_mapping[str(split_name)] = _parse_range_spec(str(split_spec), total_episodes)

    if "train" not in split_mapping:
        split_mapping["train"] = list(range(total_episodes))

    if "val" not in split_mapping:
        train_indices = list(split_mapping["train"])
        if len(train_indices) > 1 and derived_val_ratio > 0.0:
            val_count = max(1, int(round(len(train_indices) * float(derived_val_ratio))))
            if val_count >= len(train_indices):
                val_count = len(train_indices) - 1
            split_mapping["val"] = train_indices[-val_count:]
            split_mapping["train"] = train_indices[:-val_count]
        else:
            split_mapping["val"] = list(train_indices[:1]) if len(train_indices) > 0 else []

    return split_mapping


def _flatten_stat_vector(value: Any) -> List[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        flattened: List[float] = []
        for item in value:
            flattened.extend(_flatten_stat_vector(item))
        return flattened
    return []


def _passes_filters(manifest: DatasetManifest, filters: Dict[str, Any]) -> bool:
    state_dim = filters.get("state_dim")
    action_dim = filters.get("action_dim")
    robot_type = filters.get("robot_type")
    if state_dim is not None and int(state_dim) != manifest.state_dim:
        return False
    if action_dim is not None and int(action_dim) != manifest.action_dim:
        return False
    if robot_type is not None and str(robot_type) != manifest.robot_type:
        return False
    return True


def _resolve_episode_task_text(record: Dict[str, Any], task_lookup: Dict[int, str]) -> str:
    tasks = record.get("tasks")
    if isinstance(tasks, list):
        normalized = [str(task).strip() for task in tasks if str(task).strip()]
        if normalized:
            return normalized[0]

    task_index = record.get("task_index")
    if task_index is not None:
        try:
            resolved = task_lookup.get(int(task_index))
        except (TypeError, ValueError):
            resolved = None
        if resolved:
            return str(resolved)

    if task_lookup:
        return str(next(iter(task_lookup.values())))
    return "unknown_task"


def _product(shape: Iterable[int]) -> int:
    total = 1
    for value in shape:
        total *= int(value)
    return total


def _require_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError(
            "DeskTaskTac real-data loading requires 'pyarrow' in the ManipRep_310 environment."
        ) from error
    return pq


def _build_task_stratified_split_mapping(
    episode_rows: List[Dict[str, Any]],
    derived_val_ratio: float,
    split_seed: int,
) -> Dict[str, List[int]]:
    groups: Dict[int, List[int]] = {}
    for row in episode_rows:
        task_index = int(row.get("task_index", -1))
        groups.setdefault(task_index, []).append(int(row["episode_index"]))

    train_indices: List[int] = []
    val_indices: List[int] = []
    for task_index, episode_indices in sorted(groups.items()):
        local_indices = sorted(set(episode_indices))
        rng = random.Random(int(split_seed) + int(task_index) * 1009)
        rng.shuffle(local_indices)
        if len(local_indices) <= 1 or float(derived_val_ratio) <= 0.0:
            train_indices.extend(local_indices)
            continue
        val_count = max(1, int(round(len(local_indices) * float(derived_val_ratio))))
        if val_count >= len(local_indices):
            val_count = len(local_indices) - 1
        val_indices.extend(sorted(local_indices[:val_count]))
        train_indices.extend(sorted(local_indices[val_count:]))

    if not val_indices and float(derived_val_ratio) > 0.0 and len(train_indices) > 1:
        fallback_indices = sorted(set(train_indices))
        rng = random.Random(int(split_seed))
        rng.shuffle(fallback_indices)
        val_count = max(1, int(round(len(fallback_indices) * float(derived_val_ratio))))
        if val_count >= len(fallback_indices):
            val_count = len(fallback_indices) - 1
        fallback_val = set(fallback_indices[:val_count])
        val_indices = sorted(fallback_val)
        train_indices = sorted(index for index in fallback_indices if index not in fallback_val)

    return {
        "train": sorted(set(train_indices)),
        "val": sorted(set(val_indices)),
    }


def _passes_desktask_filters(manifest: DatasetManifest, filters: Dict[str, Any]) -> bool:
    dataset_name = filters.get("dataset_name")
    robot_type = filters.get("robot_type")
    task_allowlist = filters.get("task_allowlist")

    if dataset_name is not None:
        dataset_name = str(dataset_name)
        candidate_names = [
            manifest.dataset_id,
            manifest.metadata.get("dataset_dir_name"),
            manifest.metadata.get("dataset_name"),
            manifest.metadata.get("configured_root_name"),
        ]
        candidate_names.extend(str(name) for name in manifest.metadata.get("configured_root_aliases", []) if name is not None)
        if not any(
            candidate is not None and (dataset_name in str(candidate) or str(candidate) in dataset_name)
            for candidate in candidate_names
        ):
            return False
    if robot_type is not None and str(robot_type) != manifest.robot_type:
        return False
    if task_allowlist:
        allowed = {str(task_name) for task_name in task_allowlist}
        if not any(task_name in allowed for task_name in manifest.task_lookup.values()):
            return False
    return True


def _mapping_to_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    return {str(key): item for key, item in dict(value).items()}


def _append_candidate_paths(paths: List[str], value: Any) -> None:
    if value is None:
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            if item is not None and str(item).strip() and str(item).strip().lower() != "null":
                paths.append(str(item))
        return
    text = str(value).strip()
    if text and text.lower() != "null":
        paths.append(text)


def _resolve_root_candidate(candidate_paths: List[str]) -> tuple[Path, List[str], str]:
    normalized_candidates = [
        resolved_path
        for path in candidate_paths
        for resolved_path in [optional_repo_path(path)]
        if resolved_path is not None
    ]
    resolved_candidates = [str(path.resolve()) if path.exists() else str(path) for path in normalized_candidates]
    for index, candidate in enumerate(normalized_candidates):
        if candidate.exists():
            source = "local_preferred" if index == 0 and len(normalized_candidates) > 1 else "configured"
            return candidate.resolve(), resolved_candidates, source
    if not normalized_candidates:
        raise ValueError("At least one root_dir or dataset root candidate must be provided.")
    return normalized_candidates[0], resolved_candidates, "missing"


def _candidate_root_aliases(candidate_paths: List[str]) -> List[str]:
    aliases: List[str] = []
    for path in candidate_paths:
        resolved = optional_repo_path(path)
        if resolved is None:
            continue
        name = resolved.name
        if name and name not in aliases:
            aliases.append(name)
    return aliases


def _iter_manifest_sources(config: Any) -> List[Dict[str, Any]]:
    global_filters = _mapping_to_dict(getattr(config, "filters", None))
    sources: List[Dict[str, Any]] = []
    dataset_specs = getattr(config, "datasets", None)
    if dataset_specs is not None and len(dataset_specs) > 0:
        for dataset_index, dataset_spec in enumerate(dataset_specs):
            candidate_paths: List[str] = []
            _append_candidate_paths(candidate_paths, getattr(dataset_spec, "local_root_dir", None))
            _append_candidate_paths(candidate_paths, getattr(dataset_spec, "local_root_dirs", None))
            _append_candidate_paths(candidate_paths, getattr(dataset_spec, "root_candidates", None))
            _append_candidate_paths(candidate_paths, getattr(dataset_spec, "root_dir", None))
            _append_candidate_paths(candidate_paths, getattr(dataset_spec, "root_dirs", None))
            root_path, resolved_candidates, root_source = _resolve_root_candidate(candidate_paths)
            dataset_filters = dict(global_filters)
            dataset_filters.update(_mapping_to_dict(getattr(dataset_spec, "filters", None)))
            source_id = str(getattr(dataset_spec, "id", f"dataset_{dataset_index}"))
            sources.append(
                {
                    "id": source_id,
                    "label": str(getattr(dataset_spec, "label", source_id)),
                    "root_path": root_path,
                    "configured_root_name": Path(str(getattr(dataset_spec, "root_dir", root_path))).expanduser().name,
                    "configured_root_aliases": _candidate_root_aliases(candidate_paths),
                    "resolved_candidates": resolved_candidates,
                    "root_source": root_source,
                    "discovery_glob": str(getattr(dataset_spec, "discovery_glob", getattr(config, "discovery_glob"))),
                    "filters": dataset_filters,
                }
            )
        return sources

    configured_root_dirs = [str(root) for root in getattr(config, "root_dirs", [])]
    local_root_dirs = [str(root) for root in getattr(config, "local_root_dirs", [])] if getattr(config, "local_root_dirs", None) else []
    for root_index, configured_root in enumerate(configured_root_dirs):
        candidate_paths = []
        if root_index < len(local_root_dirs):
            candidate_paths.append(local_root_dirs[root_index])
        candidate_paths.append(configured_root)
        root_path, resolved_candidates, root_source = _resolve_root_candidate(candidate_paths)
        sources.append(
            {
                "id": f"root_{root_index}",
                "label": root_path.name,
                "root_path": root_path,
                "configured_root_name": Path(configured_root).expanduser().name,
                "configured_root_aliases": _candidate_root_aliases(candidate_paths),
                "resolved_candidates": resolved_candidates,
                "root_source": root_source,
                "discovery_glob": str(getattr(config, "discovery_glob")),
                "filters": dict(global_filters),
            }
        )
    return sources


def _lerobot_episode_length(record: Dict[str, Any]) -> int:
    for key in ("length", "frame_count", "num_frames"):
        value = record.get(key)
        if value is not None:
            return int(value)
    return 0


def _lerobot_chunk_size(info: Dict[str, Any]) -> int:
    return int(info.get("chunks_size", info.get("chunk_size", 1000)) or 1000)


def _format_lerobot_data_path(info: Dict[str, Any], episode_index: int) -> str:
    chunk_size = _lerobot_chunk_size(info)
    episode_chunk = int(episode_index) // int(chunk_size)
    data_pattern = str(info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"))
    return data_pattern.format(episode_chunk=episode_chunk, episode_index=int(episode_index))


def _lerobot_episode_parquet_lookup(info: Dict[str, Any], episodes: List[Dict[str, Any]]) -> Dict[int, str]:
    lookup: Dict[int, str] = {}
    for record in episodes:
        episode_index = int(record["episode_index"])
        parquet_path = record.get("parquet_path")
        lookup[episode_index] = str(parquet_path) if parquet_path not in (None, "null") else _format_lerobot_data_path(info, episode_index)
    return lookup


def _lerobot_video_lookup_for_key(record: Dict[str, Any], video_key: str) -> str | None:
    video_paths = record.get("video_paths")
    if not isinstance(video_paths, dict):
        return None
    if video_key in video_paths:
        return str(video_paths[video_key])
    suffix = video_key.split(".")[-1]
    if suffix in video_paths:
        return str(video_paths[suffix])
    lerobot_video_key = f"observation.images.{suffix}"
    if lerobot_video_key in video_paths:
        return str(video_paths[lerobot_video_key])
    return None


def _infer_lerobot_feature_specs(
    dataset_root: Path,
    info: Dict[str, Any],
    episodes: List[Dict[str, Any]],
    configured_video_key: str | None,
) -> Dict[str, Dict[str, Any]]:
    features: Dict[str, Dict[str, Any]] = {
        str(key): dict(value)
        for key, value in dict(info.get("features", {})).items()
        if isinstance(value, dict)
    }
    if not episodes:
        return features

    pq = _require_pyarrow()
    parquet_lookup = _lerobot_episode_parquet_lookup(info, episodes)
    first_episode = int(episodes[0]["episode_index"])
    first_parquet = dataset_root / parquet_lookup[first_episode]
    if first_parquet.exists():
        table = pq.read_table(first_parquet).slice(0, 1)
        if table.num_rows > 0:
            row = table.to_pylist()[0]
            for field in table.schema:
                field_name = str(field.name)
                features.setdefault(
                    field_name,
                    {
                        "dtype": str(field.type),
                        "shape": _infer_value_shape(row.get(field_name)),
                    },
                )

    if configured_video_key not in (None, "", "null"):
        key = str(configured_video_key)
        if f"{key}.path" in features or any(_lerobot_video_lookup_for_key(record, key) is not None for record in episodes[:5]):
            features.setdefault(key, {"dtype": "video", "shape": []})
    return features


def discover_lerobot_manifests(config: Any) -> List[DatasetManifest]:
    derived_val_ratio = float(config.derived_val_ratio)
    manifests: List[DatasetManifest] = []

    for source in _iter_manifest_sources(config):
        root_dir = source["root_path"]
        discovery_glob = str(source["discovery_glob"])
        for info_path in sorted(root_dir.glob(discovery_glob)):
            meta_dir = info_path.parent
            dataset_root = meta_dir.parent
            info = _read_json(info_path)
            stats = _read_json(meta_dir / "stats.json") if (meta_dir / "stats.json").exists() else {}
            episodes = _read_jsonl(meta_dir / "episodes.jsonl")
            tasks = _read_jsonl(meta_dir / "tasks.jsonl")

            state_key = str(config.state_key)
            action_key = str(config.action_key)
            video_config = getattr(config, "video", None)
            configured_video_key = None if video_config is None else str(getattr(video_config, "key", ""))
            features = _infer_lerobot_feature_specs(
                dataset_root=dataset_root,
                info=info,
                episodes=episodes,
                configured_video_key=configured_video_key,
            )
            if state_key not in features or action_key not in features:
                continue

            dataset_name = dataset_root.name
            parent_name = dataset_root.parent.name
            dataset_id = dataset_name if not parent_name or parent_name == dataset_name else f"{parent_name}/{dataset_name}"
            episode_lengths = {
                int(record["episode_index"]): _lerobot_episode_length(record)
                for record in episodes
            }
            episode_parquet_lookup = _lerobot_episode_parquet_lookup(info, episodes)
            episode_video_lookup = {
                int(record["episode_index"]): video_path
                for record in episodes
                for video_path in [_lerobot_video_lookup_for_key(record, str(configured_video_key))]
                if video_path is not None
            }
            split_mapping = _build_split_mapping(info, episodes, derived_val_ratio)
            video_keys = [key for key, spec in features.items() if spec.get("dtype") == "video"]
            manifest = DatasetManifest(
                dataset_id=dataset_id,
                root=str(dataset_root),
                source_format="LeRobot_MetaManifest",
                raw_data_type="RealRobot_LeRobotMeta",
                robot_type=str(info.get("robot_type", "unknown")),
                state_key=state_key,
                action_key=action_key,
                state_dim=int(features[state_key]["shape"][0]),
                action_dim=int(features[action_key]["shape"][0]),
                video_keys=video_keys,
                episode_lengths=episode_lengths,
                split_to_episode_indices=split_mapping,
                feature_specs=features,
                feature_stats=stats,
                task_lookup={
                    int(task_record["task_index"]): str(task_record["task"])
                    for task_record in tasks
                    if "task_index" in task_record and "task" in task_record
                },
                episode_task_lookup={},
                metadata={
                    "configured_source_id": source["id"],
                    "configured_source_label": source["label"],
                    "configured_root_name": source.get("configured_root_name"),
                    "configured_root_aliases": list(source.get("configured_root_aliases", [])),
                    "resolved_root": str(root_dir),
                    "resolved_root_candidates": list(source["resolved_candidates"]),
                    "root_source": source["root_source"],
                    "fps": float(info.get("fps", info.get("nominal_fps", 0.0)) or 0.0),
                    "total_episodes": int(info.get("total_episodes", info.get("episode_count", len(episode_lengths))) or len(episode_lengths)),
                    "total_frames": int(info.get("total_frames", info.get("frame_count", sum(episode_lengths.values()))) or sum(episode_lengths.values())),
                    "episode_parquet_lookup": episode_parquet_lookup,
                    "episode_video_lookup": episode_video_lookup,
                    "state_mean": _flatten_stat_vector(stats.get(state_key, {}).get("mean", [])),
                    "state_std": _flatten_stat_vector(stats.get(state_key, {}).get("std", [])),
                    "action_mean": _flatten_stat_vector(stats.get(action_key, {}).get("mean", [])),
                    "action_std": _flatten_stat_vector(stats.get(action_key, {}).get("std", [])),
                },
            )
            manifest.episode_task_lookup = {
                int(record["episode_index"]): _resolve_episode_task_text(record, manifest.task_lookup)
                for record in episodes
                if "episode_index" in record
            }
            if _passes_filters(manifest, dict(source["filters"])):
                manifests.append(manifest)

    max_datasets = get_positive_limit(config, "max_datasets")
    if max_datasets is not None:
        manifests = manifests[: int(max_datasets)]
    return manifests


def discover_desktasktac_manifests(config: Any) -> List[DatasetManifest]:
    derived_val_ratio = float(config.derived_val_ratio)
    split_seed = int(getattr(config, "split_seed", 0))
    manifests: List[DatasetManifest] = []
    pq = _require_pyarrow()

    for source in _iter_manifest_sources(config):
        root_dir = source["root_path"]
        discovery_glob = str(source["discovery_glob"])
        for info_path in sorted(root_dir.glob(discovery_glob)):
            dataset_root = info_path.parent.parent
            meta_dir = dataset_root / "meta"
            dataset_info = _read_json(info_path)
            features = _read_json(meta_dir / "features.json")
            tasks = _read_jsonl(meta_dir / "tasks.jsonl")
            episodes = pq.read_table(meta_dir / "episodes.parquet").to_pylist()

            dataset_name = dataset_root.name
            parent_name = dataset_root.parent.name
            dataset_id = dataset_name if not parent_name or parent_name == dataset_name else f"{parent_name}/{dataset_name}"
            task_lookup = {
                int(task_record["task_index"]): str(task_record["task"])
                for task_record in tasks
                if "task_index" in task_record and "task" in task_record
            }
            episode_lengths = {
                int(record["episode_index"]): int(record["length"])
                for record in episodes
                if "episode_index" in record and "length" in record
            }
            episode_task_lookup = {
                int(record["episode_index"]): str(record.get("task") or task_lookup.get(int(record.get("task_index", -1)), "unknown_task"))
                for record in episodes
                if "episode_index" in record
            }
            split_mapping = _build_task_stratified_split_mapping(
                episode_rows=episodes,
                derived_val_ratio=derived_val_ratio,
                split_seed=split_seed,
            )
            state_fields = [str(field_name) for field_name in config.sequence.state_fields]
            action_fields = [str(field_name) for field_name in config.sequence.action_fields]
            state_step_dim = sum(_product(features[field_name].get("shape", [])) or 1 for field_name in state_fields)
            action_step_dim = sum(_product(features[field_name].get("shape", [])) or 1 for field_name in action_fields)
            episode_parquet_lookup = {
                int(record["episode_index"]): str(record["parquet_path"])
                for record in episodes
                if "episode_index" in record and "parquet_path" in record
            }

            manifest = DatasetManifest(
                dataset_id=dataset_id,
                root=str(dataset_root),
                source_format="DeskTaskTac_MetaManifest",
                raw_data_type=f"Human_{dataset_root.name}",
                robot_type="human",
                state_key="sequence.state",
                action_key="sequence.action",
                state_dim=int(state_step_dim) * int(config.sequence.state_history),
                action_dim=int(action_step_dim),
                video_keys=[str(config.video.key)] if bool(config.video.enabled) else [],
                episode_lengths=episode_lengths,
                split_to_episode_indices=split_mapping,
                feature_specs=features,
                feature_stats={},
                task_lookup=task_lookup,
                episode_task_lookup=episode_task_lookup,
                metadata={
                    "configured_source_id": source["id"],
                    "configured_source_label": source["label"],
                    "configured_root_name": source.get("configured_root_name"),
                    "configured_root_aliases": list(source.get("configured_root_aliases", [])),
                    "resolved_root": str(root_dir),
                    "resolved_root_candidates": list(source["resolved_candidates"]),
                    "root_source": source["root_source"],
                    "dataset_dir_name": dataset_root.name,
                    "dataset_name": str(dataset_info.get("dataset_name", dataset_root.name)),
                    "fps": float(dataset_info.get("fps", 0.0) or 0.0),
                    "total_episodes": int(dataset_info.get("total_episodes", len(episode_lengths)) or len(episode_lengths)),
                    "total_frames": int(dataset_info.get("total_frames", sum(episode_lengths.values())) or sum(episode_lengths.values())),
                    "video_paths": dict(dataset_info.get("paths", {})),
                    "episode_parquet_lookup": episode_parquet_lookup,
                    "dataset_info": dataset_info,
                },
            )
            if _passes_desktask_filters(manifest, dict(source["filters"])):
                manifests.append(manifest)

    max_datasets = get_positive_limit(config, "max_datasets")
    if max_datasets is not None:
        manifests = manifests[: int(max_datasets)]
    return manifests


def _humanmix_child_split(child_dir: Path) -> tuple[str, str] | None:
    name = child_dir.name
    for split_name in ("train", "val"):
        suffix = f"_{split_name}"
        if name.endswith(suffix):
            return name[: -len(suffix)], split_name
    return None


def _format_humanmix_path(pattern: str, episode_index: int, chunks_size: int, video_key: str | None = None) -> str:
    episode_chunk = int(episode_index) // int(chunks_size)
    format_kwargs = {
        "episode_chunk": episode_chunk,
        "episode_index": int(episode_index),
        "video_key": "" if video_key is None else str(video_key),
    }
    return str(pattern).format(**format_kwargs)


def _infer_value_shape(value: Any) -> List[int]:
    if isinstance(value, list):
        if not value:
            return [0]
        return [len(value)] + _infer_value_shape(value[0])
    return []


def _infer_humanmix_feature_specs(parquet_path: Path, video_feature_specs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    pq = _require_pyarrow()
    table = pq.read_table(parquet_path).slice(0, 1)
    if table.num_rows <= 0:
        raise RuntimeError(f"Cannot infer HumanMix features from an empty parquet file: {parquet_path}")
    row = table.to_pylist()[0]
    feature_specs: Dict[str, Dict[str, Any]] = {}
    for field in table.schema:
        field_name = str(field.name)
        feature_specs[field_name] = {
            "dtype": str(field.type),
            "shape": _infer_value_shape(row.get(field_name)),
        }
    for video_key, video_spec in video_feature_specs.items():
        spec = dict(video_spec)
        spec.setdefault("dtype", "video")
        spec.setdefault("shape", [])
        feature_specs[str(video_key)] = spec
    return feature_specs


def _validate_humanmix_required_columns(parquet_path: Path, required_columns: List[str]) -> None:
    pq = _require_pyarrow()
    schema_names = set(pq.ParquetFile(parquet_path).schema_arrow.names)
    missing = [column for column in required_columns if column not in schema_names]
    if missing:
        raise ValueError(f"HumanMix parquet {parquet_path} is missing required columns: {missing}")


def discover_humanmix_manifests(config: Any) -> List[DatasetManifest]:
    manifests: List[DatasetManifest] = []

    for source in _iter_manifest_sources(config):
        root_dir = source["root_path"]
        if not root_dir.exists():
            continue

        dataset_id = str(getattr(config, "dataset_id", None) or source.get("configured_root_name") or "HumanMix")
        if not dataset_id or dataset_id == "None":
            dataset_id = "HumanMix"

        global_episode_index = 0
        split_mapping: Dict[str, List[int]] = {"train": [], "val": []}
        episode_lengths: Dict[int, int] = {}
        episode_task_lookup: Dict[int, str] = {}
        episode_parquet_lookup: Dict[int, str] = {}
        episode_video_lookup: Dict[int, str] = {}
        episode_original_index: Dict[int, int] = {}
        episode_source_dataset: Dict[int, str] = {}
        task_name_to_index: Dict[str, int] = {}
        source_dataset_summary: Dict[str, Dict[str, int]] = {}
        fps_values: List[float] = []
        total_frames_by_split: Dict[str, int] = {"train": 0, "val": 0}
        unique_video_keys: set[str] = set()
        video_feature_specs: Dict[str, Any] = {}
        first_parquet_path: Path | None = None
        child_required_validation_paths: List[Path] = []

        child_dirs = [child for child in sorted(root_dir.iterdir()) if child.is_dir()]
        for child_dir in child_dirs:
            child_split = _humanmix_child_split(child_dir)
            if child_split is None:
                continue
            source_dataset_name, split_name = child_split
            meta_dir = child_dir / "meta"
            info_path = meta_dir / "info.json"
            episodes_path = meta_dir / "episodes.jsonl"
            if not info_path.exists() or not episodes_path.exists() or not (child_dir / "data").exists():
                continue

            info = _read_json(info_path)
            episodes = _read_jsonl(episodes_path)
            tasks = _read_jsonl(meta_dir / "tasks.jsonl")
            if not episodes:
                continue

            features = info.get("features", {})
            video_keys = [
                str(key)
                for key, spec in features.items()
                if isinstance(spec, dict) and str(spec.get("dtype")) == "video"
            ]
            if len(video_keys) != 1:
                raise ValueError(
                    f"HumanMix child dataset {child_dir} is expected to have exactly one video key, got {video_keys}."
                )
            configured_video_key = str(config.video.key)
            if bool(config.video.enabled) and configured_video_key not in video_keys:
                raise ValueError(
                    f"HumanMix child dataset {child_dir} uses video key {video_keys[0]!r}, "
                    f"but config.video.key={configured_video_key!r}."
                )
            video_key = video_keys[0]
            unique_video_keys.add(video_key)
            video_feature_specs[video_key] = dict(features.get(video_key, {}))

            chunks_size = int(info.get("chunks_size", 1000) or 1000)
            data_pattern = str(info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"))
            video_pattern = str(info.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"))
            child_task_lookup = {
                int(task_record["task_index"]): str(task_record["task"])
                for task_record in tasks
                if "task_index" in task_record and "task" in task_record
            }
            fps_values.append(float(info.get("fps", 0.0) or 0.0))
            summary = source_dataset_summary.setdefault(
                source_dataset_name,
                {
                    "train_episodes": 0,
                    "val_episodes": 0,
                    "train_frames": 0,
                    "val_frames": 0,
                },
            )

            first_child_parquet: Path | None = None
            for record in episodes:
                original_episode_index = int(record["episode_index"])
                episode_length = int(record.get("length", 0) or 0)
                parquet_relpath = _format_humanmix_path(data_pattern, original_episode_index, chunks_size)
                video_relpath = _format_humanmix_path(video_pattern, original_episode_index, chunks_size, video_key=video_key)
                parquet_path = child_dir / parquet_relpath
                video_path = child_dir / video_relpath
                if not parquet_path.exists():
                    raise FileNotFoundError(f"HumanMix parquet file not found: {parquet_path}")
                if bool(config.video.enabled) and not video_path.exists():
                    raise FileNotFoundError(f"HumanMix video file not found: {video_path}")

                if first_parquet_path is None:
                    first_parquet_path = parquet_path
                if first_child_parquet is None:
                    first_child_parquet = parquet_path

                task_text = _resolve_episode_task_text(record, child_task_lookup)
                if task_text not in task_name_to_index:
                    task_name_to_index[task_text] = len(task_name_to_index)

                root_relative_parquet = str(parquet_path.relative_to(root_dir))
                root_relative_video = str(video_path.relative_to(root_dir))
                episode_lengths[global_episode_index] = episode_length
                episode_task_lookup[global_episode_index] = task_text
                episode_parquet_lookup[global_episode_index] = root_relative_parquet
                episode_video_lookup[global_episode_index] = root_relative_video
                episode_original_index[global_episode_index] = original_episode_index
                episode_source_dataset[global_episode_index] = source_dataset_name
                split_mapping[split_name].append(global_episode_index)
                summary[f"{split_name}_episodes"] += 1
                summary[f"{split_name}_frames"] += episode_length
                total_frames_by_split[split_name] += episode_length
                global_episode_index += 1

            if first_child_parquet is not None:
                child_required_validation_paths.append(first_child_parquet)

        if not episode_lengths:
            continue
        if len(unique_video_keys) != 1:
            raise ValueError(f"HumanMix root {root_dir} has inconsistent video keys: {sorted(unique_video_keys)}")
        if first_parquet_path is None:
            raise RuntimeError(f"HumanMix root {root_dir} did not provide a parquet file for schema inference.")

        feature_specs = _infer_humanmix_feature_specs(first_parquet_path, video_feature_specs)
        required_columns = ["episode_index", "frame_index", "task_index"]
        required_columns.extend(str(field_name) for field_name in config.sequence.state_fields)
        required_columns.extend(str(field_name) for field_name in config.sequence.action_fields)
        med_config = getattr(config.sequence, "med", config.sequence)
        med_kind = str(getattr(med_config, "kind", getattr(config.sequence, "med_representation", "vector"))).lower()
        if med_kind != "vector":
            raise ValueError("HumanMix manifest loading only supports raw.sequence.med.kind=vector.")
        med_side = str(getattr(med_config, "side", "right")).lower()
        vector_config = getattr(med_config, "vector", None)
        med_vector_field = getattr(vector_config, "field", None) if vector_config is not None else None
        med_vector_field = str(med_vector_field) if med_vector_field not in (None, "null") else f"tactile.{med_side}_tacglove"
        med_mask_field = getattr(vector_config, "mask_field", None) if vector_config is not None else None
        med_mask_field = str(med_mask_field) if med_mask_field not in (None, "null") else f"tactile.{med_side}_tacglove_mask"
        zero_invalid = bool(getattr(vector_config, "zero_invalid", False)) if vector_config is not None else False
        required_columns.append(med_vector_field)
        if zero_invalid:
            required_columns.append(med_mask_field)
        tactile_config = getattr(config.sequence, "tactile", None)
        if tactile_config is not None and bool(getattr(tactile_config, "enabled", False)):
            tactile_field = getattr(tactile_config, "field", None)
            tactile_field = str(tactile_field) if tactile_field not in (None, "null") else med_vector_field
            tactile_mask_field = getattr(tactile_config, "mask_field", None)
            tactile_mask_field = (
                str(tactile_mask_field) if tactile_mask_field not in (None, "null") else f"tactile.{med_side}_tacglove_mask"
            )
            required_columns.append(tactile_field)
            if bool(getattr(tactile_config, "zero_invalid", False)):
                required_columns.append(tactile_mask_field)

        missing_feature_specs = [column for column in required_columns if column not in feature_specs]
        if missing_feature_specs:
            raise ValueError(f"HumanMix inferred feature specs are missing required columns: {missing_feature_specs}")
        for validation_path in child_required_validation_paths:
            _validate_humanmix_required_columns(validation_path, sorted(set(required_columns)))

        state_fields = [str(field_name) for field_name in config.sequence.state_fields]
        action_fields = [str(field_name) for field_name in config.sequence.action_fields]
        state_step_dim = sum(_product(feature_specs[field_name].get("shape", [])) or 1 for field_name in state_fields)
        action_step_dim = sum(_product(feature_specs[field_name].get("shape", [])) or 1 for field_name in action_fields)
        task_lookup = {task_index: task_name for task_name, task_index in sorted(task_name_to_index.items(), key=lambda item: item[1])}
        video_key = next(iter(unique_video_keys))

        manifest = DatasetManifest(
            dataset_id=dataset_id,
            root=str(root_dir),
            source_format="HumanMix_MetaManifest",
            raw_data_type="Human_HumanMix",
            robot_type="human",
            state_key="sequence.state",
            action_key="sequence.action",
            state_dim=int(state_step_dim) * int(config.sequence.state_history),
            action_dim=int(action_step_dim),
            video_keys=[str(video_key)] if bool(config.video.enabled) else [],
            episode_lengths=episode_lengths,
            split_to_episode_indices=split_mapping,
            feature_specs=feature_specs,
            feature_stats={},
            task_lookup=task_lookup,
            episode_task_lookup=episode_task_lookup,
            metadata={
                "configured_source_id": source["id"],
                "configured_source_label": source["label"],
                "configured_root_name": source.get("configured_root_name"),
                "configured_root_aliases": list(source.get("configured_root_aliases", [])),
                "resolved_root": str(root_dir),
                "resolved_root_candidates": list(source["resolved_candidates"]),
                "root_source": source["root_source"],
                "dataset_dir_name": dataset_id,
                "dataset_name": dataset_id,
                "fps": fps_values[0] if fps_values else 0.0,
                "fps_values": sorted(set(fps_values)),
                "total_episodes": len(episode_lengths),
                "total_frames": sum(episode_lengths.values()),
                "total_frames_by_split": total_frames_by_split,
                "video_key": video_key,
                "video_keys_by_source": {source_name: video_key for source_name in source_dataset_summary},
                "episode_parquet_lookup": episode_parquet_lookup,
                "episode_video_lookup": episode_video_lookup,
                "episode_original_index": episode_original_index,
                "episode_source_dataset": episode_source_dataset,
                "source_dataset_summary": source_dataset_summary,
            },
        )
        if _passes_desktask_filters(manifest, dict(source["filters"])):
            manifests.append(manifest)

    max_datasets = get_positive_limit(config, "max_datasets")
    if max_datasets is not None:
        manifests = manifests[: int(max_datasets)]
    return manifests


MANIFEST_LOADERS = {
    "DeskTaskTac_MetaManifest": discover_desktasktac_manifests,
    "HumanMix_MetaManifest": discover_humanmix_manifests,
    "LeRobot_MetaManifest": discover_lerobot_manifests,
}
