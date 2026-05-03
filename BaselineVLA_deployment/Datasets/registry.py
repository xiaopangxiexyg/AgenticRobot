from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Dict, List, Sequence

from utils.path_config import ensure_conda_runtime_libs, load_local_env, optional_repo_path


load_local_env()
ensure_conda_runtime_libs()

from torch.utils.data import DataLoader, Dataset

from Datasets.base import DatasetManifest, collate_canonical_samples
from Datasets.config_utils import safe_config_get
from Datasets.desktasktac_dataset import (
    DeskTaskTacRawHandSequenceDataset,
    DeskTaskTacSequenceDataset,
    infer_manifest_med_capabilities,
)
from Datasets.humanmix_dataset import HumanMixSequenceDataset
from Datasets.input_adapters import INPUT_ADAPTERS
from Datasets.manifest_loaders import MANIFEST_LOADERS
from Datasets.mock_dataset import MockRawSequenceDataset
from Datasets.realrobot_lerobot_dataset import RealRobotLeRobotSequenceDataset
from GenMed.registry import attach_med_provider, build_med_provider, med_enabled_for_experiment


RAW_DATASETS = {
    "DeskTaskTac_Sequence": DeskTaskTacSequenceDataset,
    "DeskTaskTacRawHand_Sequence": DeskTaskTacRawHandSequenceDataset,
    "HumanMix_Sequence": HumanMixSequenceDataset,
    "RealRobotLeRobot_Sequence": RealRobotLeRobotSequenceDataset,
    "Mock_RawSequence": MockRawSequenceDataset,
}


@dataclass
class DataPipeline:
    train_loader: DataLoader | None
    val_loader: DataLoader | None
    input_adapter: Any
    manifests: List[DatasetManifest]
    summary: Dict[str, Any]
    med_provider_name: str | None


class RandomSubsetDataset(Dataset):
    def __init__(self, base_dataset: Dataset, selected_indices: List[int], split: str, ratio: float, seed: int) -> None:
        self.base_dataset = base_dataset
        self.selected_indices = [int(index) for index in selected_indices]
        self.split = str(split)
        self.ratio = float(ratio)
        self.seed = int(seed)
        self._summary = self._build_summary()

    def __len__(self) -> int:
        return len(self.selected_indices)

    def __getitem__(self, index: int):
        return self.base_dataset[self.selected_indices[int(index)]]

    @property
    def summary(self) -> Dict[str, Any]:
        return dict(self._summary)

    def _unwrap_base_dataset(self, dataset: Any) -> Any:
        current = dataset
        while True:
            next_dataset = getattr(current, "base_dataset", None)
            if next_dataset is None:
                next_dataset = getattr(current, "dataset", None)
            if next_dataset is None or next_dataset is current:
                return current
            current = next_dataset

    def _build_summary(self) -> Dict[str, Any]:
        if hasattr(self.base_dataset, "summary"):
            summary_source = self.base_dataset.summary
        else:
            raw_dataset = self._unwrap_base_dataset(self.base_dataset)
            summary_source = getattr(raw_dataset, "summary", {})
        summary = dict(summary_source)
        summary["sample_count"] = len(self.selected_indices)
        summary["sample_ratio_applied"] = float(self.ratio)
        summary["sample_subset_seed"] = int(self.seed)
        summary["sample_subset_base_sample_count"] = int(len(self.base_dataset))
        summary["sample_subset_split"] = self.split
        if self.split == "val":
            summary["validation_ratio_applied"] = float(self.ratio)
            summary["validation_subset_seed"] = int(self.seed)
            summary["validation_subset_base_sample_count"] = int(len(self.base_dataset))

        raw_dataset = self._unwrap_base_dataset(self.base_dataset)
        addresses = getattr(raw_dataset, "addresses", None)
        if addresses is not None:
            counts: Dict[str, int] = {}
            for selected_index in self.selected_indices:
                dataset_id = str(addresses[int(selected_index)].dataset_id)
                counts[dataset_id] = counts.get(dataset_id, 0) + 1
            summary["sample_count_by_dataset"] = counts
        return summary


def _discover_manifests(data_config: Any) -> List[DatasetManifest]:
    loader_name = str(data_config.raw.manifest_loader)
    if loader_name not in MANIFEST_LOADERS:
        raise KeyError(f"Unknown manifest loader: {loader_name}")
    manifests = MANIFEST_LOADERS[loader_name](data_config.raw)
    if not manifests:
        root_dirs = [
            str(resolved_root)
            for root in getattr(data_config.raw, "root_dirs", [])
            for resolved_root in [optional_repo_path(root)]
            if resolved_root is not None
        ]
        dataset_specs = []
        raw_datasets = getattr(data_config.raw, "datasets", None)
        if raw_datasets is not None:
            for dataset_spec in raw_datasets:
                dataset_specs.append(
                    {
                        "id": str(getattr(dataset_spec, "id", "unknown")),
                        "root_dir": str(getattr(dataset_spec, "root_dir", "")),
                        "local_root_dir": None if getattr(dataset_spec, "local_root_dir", None) is None else str(dataset_spec.local_root_dir),
                        "filters": {} if getattr(dataset_spec, "filters", None) is None else dict(dataset_spec.filters),
                    }
                )
        raise RuntimeError(
            "No manifests matched the configured filters. "
            f"loader={loader_name}, root_dirs={root_dirs}, discovery_glob={data_config.raw.discovery_glob}, "
            f"filters={dict(data_config.raw.filters)}, dataset_specs={dataset_specs}. "
            "If you want to use the built-in mock pipeline, make sure "
            "Data/RawData/Mock_RawData/*/meta/info.json exists."
        )
    return manifests


def _build_raw_dataset(manifests: List[DatasetManifest], split: str, data_config: Any):
    dataset_name = str(data_config.raw.dataset_type)
    if dataset_name not in RAW_DATASETS:
        raise KeyError(f"Unknown raw dataset type: {dataset_name}")
    return RAW_DATASETS[dataset_name](manifests=manifests, split=split, config=data_config.raw)


def _resolve_loader_value(loader_config: Any, split: str, key: str, default: Any = None) -> Any:
    split_loader_config = safe_config_get(loader_config, str(split), None)
    split_value = safe_config_get(split_loader_config, key, None)
    if split_value is not None:
        return split_value
    return safe_config_get(loader_config, key, default)


def _build_loader(dataset: Any, split: str, data_config: Any) -> DataLoader:
    loader_config = data_config.raw.loader
    batch_size = int(_resolve_loader_value(loader_config, split, "batch_size", 1))
    num_workers = int(_resolve_loader_value(loader_config, split, "num_workers", 0) or 0)
    loader_kwargs: Dict[str, Any] = {}
    if num_workers > 0:
        persistent_workers = bool(_resolve_loader_value(loader_config, split, "persistent_workers", False))
        loader_kwargs["persistent_workers"] = persistent_workers
        prefetch_factor = _resolve_loader_value(loader_config, split, "prefetch_factor", None)
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
        multiprocessing_context = _resolve_loader_value(loader_config, split, "multiprocessing_context", None)
        if multiprocessing_context not in (None, "null"):
            loader_kwargs["multiprocessing_context"] = str(multiprocessing_context)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(split == "train"),
        num_workers=num_workers,
        pin_memory=bool(_resolve_loader_value(loader_config, split, "pin_memory", False)),
        collate_fn=collate_canonical_samples,
        **loader_kwargs,
    )


def _validate_ratio(name: str, value: Any) -> float:
    normalized = float(1.0 if value is None else value)
    if not 0.0 < normalized <= 1.0:
        raise ValueError(f"{name} must be in (0, 1], got {value}")
    return normalized


def _resolve_validation_ratio(config: Any) -> float:
    train_config = getattr(config, "train", None)
    ratio = safe_config_get(train_config, "validation_ratio", 1.0)
    return _validate_ratio("train.validation_ratio", ratio)


def _resolve_split_sample_ratio(config: Any, split: str) -> float:
    raw_config = safe_config_get(getattr(config, "data", None), "raw", None)
    sampling_config = safe_config_get(raw_config, "sampling", None)
    split_ratio = safe_config_get(sampling_config, f"{split}_sample_ratio", None)
    if split_ratio is not None:
        return _validate_ratio(f"data.raw.sampling.{split}_sample_ratio", split_ratio)
    shared_ratio = safe_config_get(sampling_config, "sample_ratio", None)
    if shared_ratio is not None:
        return _validate_ratio("data.raw.sampling.sample_ratio", shared_ratio)
    if split == "val":
        return _resolve_validation_ratio(config)
    return 1.0


def _resolve_validation_seed(config: Any) -> int:
    train_config = getattr(config, "train", None)
    configured_seed = safe_config_get(train_config, "validation_seed", None)
    if configured_seed not in (None, "null"):
        return int(configured_seed)
    project_config = getattr(config, "project", None)
    return int(safe_config_get(project_config, "seed", 0) or 0)


def _resolve_split_sample_seed(config: Any, split: str) -> int:
    raw_config = safe_config_get(getattr(config, "data", None), "raw", None)
    sampling_config = safe_config_get(raw_config, "sampling", None)
    split_seed = safe_config_get(sampling_config, f"{split}_sample_seed", None)
    if split_seed not in (None, "null"):
        return int(split_seed)
    shared_seed = safe_config_get(sampling_config, "sample_seed", None)
    if shared_seed not in (None, "null"):
        return int(shared_seed)
    return _resolve_validation_seed(config)


def _maybe_subset_split_dataset(dataset: Any, config: Any, split: str) -> Any:
    ratio = _resolve_split_sample_ratio(config, split)
    if ratio >= 1.0 or len(dataset) <= 1:
        return dataset
    subset_size = max(1, int(round(len(dataset) * ratio)))
    if subset_size >= len(dataset):
        return dataset
    seed = _resolve_split_sample_seed(config, split)
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    selected_indices = sorted(indices[:subset_size])
    return RandomSubsetDataset(
        base_dataset=dataset,
        selected_indices=selected_indices,
        split=str(split),
        ratio=ratio,
        seed=seed,
    )


def _summarize_manifests(manifests: List[DatasetManifest]) -> Dict[str, Any]:
    manifest_summary = []
    for manifest in manifests:
        manifest_summary.append(
            {
                "dataset_id": manifest.dataset_id,
                "configured_source_id": manifest.metadata.get("configured_source_id"),
                "configured_source_label": manifest.metadata.get("configured_source_label"),
                "resolved_root": manifest.metadata.get("resolved_root", manifest.root),
                "root_source": manifest.metadata.get("root_source"),
                "robot_type": manifest.robot_type,
                "state_dim": manifest.state_dim,
                "action_dim": manifest.action_dim,
                "video_keys": list(manifest.video_keys),
                "med_capabilities": infer_manifest_med_capabilities(manifest),
                "split_sizes": {
                    split_name: len(indices)
                    for split_name, indices in manifest.split_to_episode_indices.items()
                },
            }
        )
    return {"manifests": manifest_summary}


def _normalize_requested_splits(requested_splits: Sequence[str] | None) -> List[str]:
    if requested_splits is None:
        return ["train", "val"]
    normalized: List[str] = []
    for split_name in requested_splits:
        normalized_name = str(split_name).lower()
        if normalized_name not in {"train", "val"}:
            raise ValueError(f"Unsupported requested split: {split_name!r}. Expected one of: train, val.")
        if normalized_name not in normalized:
            normalized.append(normalized_name)
    if not normalized:
        raise ValueError("requested_splits must include at least one split.")
    return normalized


def build_data_pipeline(config: Any, requested_splits: Sequence[str] | None = None) -> DataPipeline:
    normalized_splits = _normalize_requested_splits(requested_splits)
    manifests = _discover_manifests(config.data)
    datasets: Dict[str, Any] = {
        split_name: _build_raw_dataset(manifests, split_name, config.data)
        for split_name in normalized_splits
    }
    med_provider = None
    if med_enabled_for_experiment(config.data.med, config.experiment.name):
        med_provider = build_med_provider(config.data.med)
    for split_name in list(datasets.keys()):
        datasets[split_name] = attach_med_provider(datasets[split_name], med_provider)
    for split_name in list(datasets.keys()):
        datasets[split_name] = _maybe_subset_split_dataset(datasets[split_name], config, split_name)
    adapter_name = str(config.experiment.input_adapter)
    if adapter_name not in INPUT_ADAPTERS:
        raise KeyError(f"Unknown input adapter: {adapter_name}")
    input_adapter = INPUT_ADAPTERS[adapter_name](config=config)
    loaders: Dict[str, DataLoader] = {
        split_name: _build_loader(datasets[split_name], split_name, config.data)
        for split_name in normalized_splits
    }

    split_summaries: Dict[str, Dict[str, Any]] = {}
    for split_name, dataset in datasets.items():
        split_summaries[split_name] = dataset.dataset.summary if hasattr(dataset, "dataset") else dataset.summary
    reference_summary = split_summaries.get("train") or split_summaries.get("val")
    if reference_summary is None:
        raise ValueError("Unable to infer a reference data summary because no requested splits were built.")
    med_spec = None
    med_dim = None
    if med_provider is not None:
        med_spec = med_provider.infer_output_spec(reference_summary)
        med_dim = None if med_spec is None else int(med_spec["flat_dim"])
        if med_dim is None:
            med_dim = med_provider.infer_output_dim(reference_summary)
        if med_spec is not None or med_dim is not None:
            updated_summaries: Dict[str, Dict[str, Any]] = {}
            for split_name, split_summary in split_summaries.items():
                new_summary = dict(split_summary)
                if med_dim is not None:
                    new_summary["med_dim"] = int(med_dim)
                if med_spec is not None:
                    new_summary["med_spec"] = med_spec
                updated_summaries[split_name] = new_summary
            split_summaries = updated_summaries
    summary = {
        "med_provider": None if med_provider is None else med_provider.name,
        "med_spec": med_spec,
        "med_dim": med_dim,
        "manifests": _summarize_manifests(manifests)["manifests"],
    }
    summary.update(split_summaries)
    return DataPipeline(
        train_loader=loaders.get("train"),
        val_loader=loaders.get("val"),
        input_adapter=input_adapter,
        manifests=manifests,
        summary=summary,
        med_provider_name=None if med_provider is None else med_provider.name,
    )
