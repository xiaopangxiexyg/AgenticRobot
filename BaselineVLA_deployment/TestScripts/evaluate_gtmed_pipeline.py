from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from Datasets.base import PreparedBatch
from Datasets.input_adapters import GenMedInputAdapter, _move_mapping_to_device
from Datasets.registry import build_data_pipeline
from Models.med_support import normalize_med_spec
from Training.build_utils import resolve_med_spec
from Training.engine import resolve_device
from Training.registry import get_experiment_spec
from utils.path_config import load_local_env
from utils.runtime import save_config, save_json, set_seed

from eval_vis_utils import format_split_metrics, resolve_checkpoint_path
from evaluate_checkpoint import (
    _default_output_dir,
    _evaluate_split,
    _load_checkpoint,
    _load_state_dict_compatibly,
    _pick_loader,
    _prepare_config,
    _sanitize_name,
    _split_names,
)


load_local_env()


_EVAL_LOADER_CACHE_ATTR = "_gtmed_pipeline_eval_loaders"


class GTMedWithPredictedMedPipeline(torch.nn.Module):
    def __init__(
        self,
        *,
        genmed_model: torch.nn.Module,
        gtmed_model: torch.nn.Module,
        genmed_device: torch.device,
        gtmed_device: torch.device,
        include_med_output: bool = False,
    ) -> None:
        super().__init__()
        self.genmed_model = genmed_model
        self.gtmed_model = gtmed_model
        self.genmed_device = genmed_device
        self.gtmed_device = gtmed_device
        self.include_med_output = bool(include_med_output)

    def _move_model_inputs(self, model_inputs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        if device == self.gtmed_device:
            return model_inputs
        return _move_mapping_to_device(model_inputs, device)

    def forward(self, **model_inputs: Any) -> Dict[str, torch.Tensor]:
        genmed_inputs = self._move_model_inputs(dict(model_inputs), self.genmed_device)
        genmed_outputs = self.genmed_model(**genmed_inputs)
        if "med" not in genmed_outputs:
            raise KeyError("GenMed model outputs must contain `med` for GTMed pipeline evaluation.")
        predicted_med = genmed_outputs["med"]
        if self.gtmed_device == self.genmed_device:
            gtmed_inputs = dict(genmed_inputs)
        else:
            gtmed_inputs = self._move_model_inputs(dict(model_inputs), self.gtmed_device)
        del genmed_outputs
        del genmed_inputs
        if self.gtmed_device != self.genmed_device and torch.is_tensor(predicted_med):
            predicted_med = predicted_med.to(self.gtmed_device)
        gtmed_inputs["med"] = predicted_med
        gtmed_outputs = self.gtmed_model(**gtmed_inputs)
        combined_outputs = dict(gtmed_outputs)
        if self.include_med_output:
            combined_outputs.setdefault("med", predicted_med)
        return combined_outputs


class GTMedPipelineInputAdapter:
    name = "gtmed_pipeline"

    def __init__(self, *, genmed_config: Any, gtmed_input_adapter: Any) -> None:
        self.gtmed_input_adapter = gtmed_input_adapter
        self.genmed_input_adapter = GenMedInputAdapter(config=genmed_config)
        self.cpu_device = torch.device("cpu")

    def prepare_batch(self, batch: Dict[str, Any], device: torch.device) -> PreparedBatch:
        del device
        observations = batch["observations"]
        targets = batch["targets"]
        availability = batch["availability"]

        model_inputs = dict(observations)
        gtmed_task_inputs = self.gtmed_input_adapter._prepare_task_inputs(observations, batch["metadata"], self.cpu_device)
        genmed_task_inputs = self.genmed_input_adapter._prepare_task_inputs(observations, batch["metadata"], self.cpu_device)
        model_inputs.update(gtmed_task_inputs)
        for key, value in genmed_task_inputs.items():
            model_inputs.setdefault(key, value)
        return PreparedBatch(
            model_inputs=model_inputs,
            targets={
                "action": targets["action"],
            },
            metadata=batch["metadata"],
            availability=availability,
        )


class GenMedCacheInputAdapter:
    name = "genmed_cache_builder"

    def __init__(self, *, genmed_config: Any) -> None:
        self.genmed_input_adapter = GenMedInputAdapter(config=genmed_config)

    def prepare_batch(self, batch: Dict[str, Any], device: torch.device) -> PreparedBatch:
        observations = _move_mapping_to_device(batch["observations"], device)
        availability = _move_mapping_to_device(batch["availability"], device)
        model_inputs = dict(observations)
        model_inputs.update(self.genmed_input_adapter._prepare_task_inputs(observations, batch["metadata"], device))
        return PreparedBatch(
            model_inputs=model_inputs,
            targets={},
            metadata=batch["metadata"],
            availability=availability,
        )


class GTMedCachedMedInputAdapter:
    name = "gtmed_cached_med"

    def __init__(self, *, gtmed_input_adapter: Any, predicted_med_cache: "PredictedMedCache") -> None:
        self.gtmed_input_adapter = gtmed_input_adapter
        self.predicted_med_cache = predicted_med_cache

    def prepare_batch(self, batch: Dict[str, Any], device: torch.device) -> PreparedBatch:
        observations = _move_mapping_to_device(batch["observations"], device)
        targets = _move_mapping_to_device(batch["targets"], device)
        availability = _move_mapping_to_device(batch["availability"], device)
        cached_med = self.predicted_med_cache.load_batch(batch["metadata"]).to(device=device, dtype=torch.float32)
        model_inputs = dict(observations)
        model_inputs.update(self.gtmed_input_adapter._prepare_task_inputs(observations, batch["metadata"], device))
        model_inputs["med"] = cached_med
        return PreparedBatch(
            model_inputs=model_inputs,
            targets={
                "action": targets["action"],
            },
            metadata=batch["metadata"],
            availability=availability,
        )


class PredictedMedCache:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir.expanduser().resolve()
        self.summary_path = self.root_dir / "summary.json"

    def _sample_path(self, metadata: Dict[str, Any]) -> Path:
        split_name = _sanitize_name(str(metadata.get("split", "unknown_split")))
        dataset_id = _sanitize_name(str(metadata.get("dataset_id", "unknown_dataset")))
        episode_index = int(metadata.get("episode_index", -1))
        frame_index = int(metadata.get("frame_index", -1))
        return (
            self.root_dir
            / split_name
            / dataset_id
            / f"episode_{episode_index:06d}"
            / f"frame_{frame_index:06d}.pt"
        )

    def has_summary(self) -> bool:
        return self.summary_path.exists()

    def save_sample(self, metadata: Dict[str, Any], med_tensor: torch.Tensor) -> Path:
        sample_path = self._sample_path(metadata)
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(med_tensor.detach().to(dtype=torch.float32).cpu(), sample_path)
        return sample_path

    def load_sample(self, metadata: Dict[str, Any]) -> torch.Tensor:
        sample_path = self._sample_path(metadata)
        if not sample_path.exists():
            raise FileNotFoundError(
                "Predicted MED cache file is missing for "
                f"dataset_id={metadata.get('dataset_id')} episode_index={metadata.get('episode_index')} "
                f"frame_index={metadata.get('frame_index')}: {sample_path}"
            )
        try:
            med_tensor = torch.load(sample_path, map_location="cpu", weights_only=False)
        except TypeError:
            med_tensor = torch.load(sample_path, map_location="cpu")
        if not torch.is_tensor(med_tensor):
            raise TypeError(f"Predicted MED cache file must contain a tensor, got {type(med_tensor)!r}: {sample_path}")
        return med_tensor.to(dtype=torch.float32)

    def load_batch(self, metadata_batch: List[Dict[str, Any]]) -> torch.Tensor:
        med_tensors = [self.load_sample(sample_metadata) for sample_metadata in metadata_batch]
        if not med_tensors:
            raise ValueError("Cannot load predicted MED cache for an empty batch.")
        return torch.stack(med_tensors, dim=0)

    def save_summary(self, payload: Dict[str, Any]) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        save_json(self.summary_path, payload)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate GTMed action prediction on the same data split with an oracle-med baseline, "
            "a GenMed->GTMed pipeline, and an optional Vanilla direct-action baseline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("gtmed_checkpoint", type=str, help="GTMed checkpoint file, run dir, or run name under Outputs/.")
    parser.add_argument("--genmed-checkpoint", type=str, required=True, help="GenMed checkpoint file, run dir, or run name.")
    parser.add_argument(
        "--vanilla-checkpoint",
        type=str,
        default=None,
        help="Optional Vanilla checkpoint file, run dir, or run name for direct action comparison.",
    )
    parser.add_argument("--gtmed-checkpoint-kind", type=str, default="best", choices=["auto", "best", "last"])
    parser.add_argument("--genmed-checkpoint-kind", type=str, default="best", choices=["auto", "best", "last"])
    parser.add_argument("--vanilla-checkpoint-kind", type=str, default="best", choices=["auto", "best", "last"])
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "both"])
    parser.add_argument("--data-config", type=str, default=None, help="Optional data preset or resolved data YAML path.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--genmed-device",
        type=str,
        default=None,
        help="Optional device for the GenMed stage. Defaults to the GTMed evaluation device.",
    )
    parser.add_argument(
        "--pipeline-mode",
        type=str,
        default="inline",
        choices=["inline", "cached"],
        help="inline keeps GenMed->GTMed inside one forward pass; cached saves predicted med to disk first.",
    )
    parser.add_argument(
        "--predicted-med-cache-dir",
        type=str,
        default=None,
        help="Directory used by cached mode. Defaults to Data/MedData/GTMedPipelinePredictions/<run tag>.",
    )
    parser.add_argument(
        "--reuse-predicted-med-cache",
        action="store_true",
        help="When cached mode is enabled, reuse an existing predicted-med cache summary instead of rebuilding it.",
    )
    parser.add_argument("--loader-mode", type=str, default="safe", choices=["safe", "checkpoint"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--save-samples", type=int, default=1)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-points", type=int, default=512)
    parser.add_argument("--visualize", type=str, default="all", choices=["all", "plots", "projection", "none"])
    parser.add_argument("--projection-point-radius", type=int, default=4)
    parser.add_argument("--projection-frame-height", type=int, default=360)
    parser.add_argument("--projection-max-steps", type=int, default=0)
    parser.add_argument("--camera-serial", type=str, default="")
    parser.add_argument("--camera-intrinsics-json", type=str, default="")
    parser.add_argument(
        "--include-predicted-med-output",
        action="store_true",
        help="Keep predicted med tensors in inline pipeline outputs. Disabled by default to reduce memory pressure.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="OmegaConf dotlist override, for example data.raw.loader.batch_size=1",
    )
    return parser.parse_args()


def _build_pipeline_output_dir(gtmed_checkpoint_path: Path, gtmed_config: Any, split_name: str, genmed_checkpoint_path: Path) -> Path:
    default_base = _default_output_dir(gtmed_checkpoint_path, gtmed_config, split_name)
    genmed_tag = _sanitize_name(genmed_checkpoint_path.parent.name)
    return default_base.parent / f"{default_base.name}__from_{genmed_tag}"


def _default_predicted_med_cache_dir(gtmed_checkpoint_path: Path, gtmed_config: Any, split_name: str, genmed_checkpoint_path: Path) -> Path:
    data_name = _sanitize_name(str(getattr(gtmed_config.data, "name", "data")))
    split_tag = _sanitize_name(split_name)
    gtmed_tag = _sanitize_name(gtmed_checkpoint_path.parent.name)
    genmed_tag = _sanitize_name(genmed_checkpoint_path.parent.name)
    return PROJECT_ROOT / "Data" / "MedData" / "GTMedPipelinePredictions" / f"{data_name}_{split_tag}__{gtmed_tag}__from_{genmed_tag}"


def _prepare_secondary_config(checkpoint_config: Any, *, shared_data_config: Any, device_name: str | None) -> Any:
    config = OmegaConf.create(OmegaConf.to_container(checkpoint_config, resolve=True))
    config.data = OmegaConf.create(OmegaConf.to_container(shared_data_config, resolve=True))
    if device_name is not None:
        config.train.device = str(device_name)
    return config


def _prepare_cached_pipeline_config(base_config: Any) -> Any:
    config = OmegaConf.create(OmegaConf.to_container(base_config, resolve=True))
    raw_sequence_med = getattr(getattr(getattr(config.data, "raw", None), "sequence", None), "med", None)
    if raw_sequence_med is not None:
        raw_sequence_med.emit_intermediate = False
    enabled_for_experiments = getattr(getattr(config.data, "med", None), "enabled_for_experiments", None)
    experiment_name = str(getattr(config.experiment, "name", "")).lower()
    if enabled_for_experiments is not None and experiment_name:
        enabled_for_experiments[experiment_name] = False
    return config


def _assert_expected_experiment(config: Any, expected_name: str) -> None:
    actual_name = str(getattr(config.experiment, "name", "")).lower()
    normalized_expected = str(expected_name).lower()
    if actual_name != normalized_expected:
        raise ValueError(f"Expected a {normalized_expected} checkpoint, but got experiment={actual_name!r}.")


def _validate_med_compatibility(gtmed_config: Any, genmed_config: Any, data_summary: Dict[str, Any]) -> Dict[str, Any]:
    gtmed_med_spec = normalize_med_spec(resolve_med_spec(gtmed_config, data_summary, experiment_name="GTMed"))
    genmed_med_spec = normalize_med_spec(resolve_med_spec(genmed_config, data_summary, experiment_name="GenMed"))
    if gtmed_med_spec["kind"] != genmed_med_spec["kind"] or gtmed_med_spec["shape"] != genmed_med_spec["shape"]:
        raise ValueError(
            "GTMed and GenMed checkpoints are not compatible under the current evaluation data config: "
            f"gtmed med_spec={gtmed_med_spec}, genmed med_spec={genmed_med_spec}."
        )
    return gtmed_med_spec


def _clone_loader_without_shuffle(loader: DataLoader) -> DataLoader:
    loader_kwargs: Dict[str, Any] = {
        "dataset": loader.dataset,
        "batch_size": loader.batch_size,
        "shuffle": False,
        "num_workers": loader.num_workers,
        "collate_fn": loader.collate_fn,
        "pin_memory": loader.pin_memory,
        "drop_last": loader.drop_last,
        "timeout": loader.timeout,
        "worker_init_fn": loader.worker_init_fn,
    }
    pin_memory_device = getattr(loader, "pin_memory_device", "")
    if pin_memory_device:
        loader_kwargs["pin_memory_device"] = pin_memory_device
    if loader.num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(getattr(loader, "persistent_workers", False))
        prefetch_factor = getattr(loader, "prefetch_factor", None)
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
        multiprocessing_context = getattr(loader, "multiprocessing_context", None)
        if multiprocessing_context is not None:
            loader_kwargs["multiprocessing_context"] = multiprocessing_context
    return DataLoader(**loader_kwargs)


def _pick_evaluation_loader(data_pipeline: Any, split_name: str):
    cache = getattr(data_pipeline, _EVAL_LOADER_CACHE_ATTR, None)
    if cache is None:
        cache = {}
        setattr(data_pipeline, _EVAL_LOADER_CACHE_ATTR, cache)
    if split_name in cache:
        return cache[split_name]
    loader = _pick_loader(data_pipeline, split_name)
    if split_name == "train":
        loader = _clone_loader_without_shuffle(loader)
    cache[split_name] = loader
    return loader


def _evaluate_mode(
    *,
    mode_name: str,
    model: torch.nn.Module,
    input_adapter: Any,
    data_pipeline: Any,
    device: torch.device,
    output_dir: Path,
    max_batches: int | None,
    save_samples: int,
    sample_index: int,
    args: argparse.Namespace,
    data_summary: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    mode_output_dir = output_dir / mode_name
    split_results: Dict[str, Any] = {}
    effective_data_summary = data_pipeline.summary if data_summary is None else data_summary
    for split_name in _split_names(str(args.split)):
        split_results[split_name] = _evaluate_split(
            split_name=split_name,
            loader=_pick_evaluation_loader(data_pipeline, split_name),
            model=model,
            input_adapter=input_adapter,
            device=device,
            output_dir=mode_output_dir,
            data_summary=effective_data_summary,
            max_batches=max_batches,
            save_samples=save_samples,
            sample_index=sample_index,
            args=args,
        )
    return split_results


def _build_predicted_med_cache(
    *,
    predicted_med_cache: PredictedMedCache,
    data_pipeline: Any,
    model: torch.nn.Module,
    input_adapter: Any,
    device: torch.device,
    args: argparse.Namespace,
    med_spec: Dict[str, Any],
    gtmed_checkpoint_path: Path,
    genmed_checkpoint_path: Path,
    gtmed_checkpoint_kind: str,
    genmed_checkpoint_kind: str,
    gtmed_config: Any,
) -> Dict[str, Any]:
    split_results: Dict[str, Any] = {}
    preview_paths: List[str] = []

    model.eval()
    with torch.inference_mode():
        for split_name in _split_names(str(args.split)):
            loader = _pick_evaluation_loader(data_pipeline, split_name)
            batches = 0
            total_samples = 0
            split_preview_paths: List[str] = []
            for batch_index, batch in enumerate(loader):
                if args.max_batches is not None and batch_index >= int(args.max_batches):
                    break
                prepared = input_adapter.prepare_batch(batch, device)
                outputs = model(**prepared.model_inputs)
                predicted_med = outputs.get("med")
                if not torch.is_tensor(predicted_med):
                    raise KeyError("GenMed cache build expects model outputs['med'] to be a tensor.")
                cpu_predicted_med = predicted_med.detach().to(dtype=torch.float32).cpu()
                batch_size = len(batch["metadata"])
                batches += 1
                total_samples += batch_size
                for local_index, sample_metadata in enumerate(batch["metadata"]):
                    sample_path = predicted_med_cache.save_sample(sample_metadata, cpu_predicted_med[local_index])
                    if len(split_preview_paths) < 3:
                        split_preview_paths.append(str(sample_path))
                    if len(preview_paths) < 8:
                        preview_paths.append(str(sample_path))
                del prepared
                del outputs
                del predicted_med
                del cpu_predicted_med
            split_results[split_name] = {
                "split": split_name,
                "batches": int(batches),
                "samples": int(total_samples),
                "sample_paths_preview": split_preview_paths,
            }

    cache_summary = {
        "cache_dir": str(predicted_med_cache.root_dir),
        "summary_path": str(predicted_med_cache.summary_path),
        "reused": False,
        "requested_split": str(args.split),
        "max_batches": None if args.max_batches is None else int(args.max_batches),
        "med_spec": med_spec,
        "data_name": str(getattr(gtmed_config.data, "name", "unknown")),
        "gtmed_checkpoint": str(gtmed_checkpoint_path),
        "gtmed_checkpoint_resolution": str(gtmed_checkpoint_kind),
        "genmed_checkpoint": str(genmed_checkpoint_path),
        "genmed_checkpoint_resolution": str(genmed_checkpoint_kind),
        "splits": split_results,
        "sample_paths_preview": preview_paths,
    }
    predicted_med_cache.save_summary(cache_summary)
    return cache_summary


def _build_pairwise_comparison(
    reference_results: Dict[str, Any],
    candidate_results: Dict[str, Any],
    *,
    reference_label: str,
    candidate_label: str,
) -> Dict[str, Any]:
    comparison: Dict[str, Any] = {}
    for split_name, reference_split in reference_results.items():
        candidate_split = candidate_results.get(split_name)
        if candidate_split is None:
            continue
        reference_metrics = dict(reference_split.get("metrics", {}))
        candidate_metrics = dict(candidate_split.get("metrics", {}))
        split_comparison: Dict[str, Any] = {}
        for prediction_key, reference_metric in reference_metrics.items():
            if prediction_key not in candidate_metrics:
                continue
            candidate_metric = candidate_metrics[prediction_key]
            key_comparison: Dict[str, Any] = {}
            for metric_name in ("rmse", "mae", "mse"):
                reference_value = reference_metric.get(metric_name)
                candidate_value = candidate_metric.get(metric_name)
                if reference_value is None or candidate_value is None:
                    continue
                delta = float(candidate_value) - float(reference_value)
                relative = None if float(reference_value) == 0.0 else delta / float(reference_value)
                key_comparison[metric_name] = {
                    reference_label: float(reference_value),
                    candidate_label: float(candidate_value),
                    "delta": delta,
                    "relative_delta": relative,
                }
            if key_comparison:
                split_comparison[prediction_key] = key_comparison
        if split_comparison:
            comparison[split_name] = split_comparison
    return comparison


def main() -> None:
    args = _parse_args()

    gtmed_checkpoint_path, gtmed_checkpoint_kind = resolve_checkpoint_path(
        checkpoint_ref=str(args.gtmed_checkpoint),
        preferred_kind=str(args.gtmed_checkpoint_kind),
    )
    genmed_checkpoint_path, genmed_checkpoint_kind = resolve_checkpoint_path(
        checkpoint_ref=str(args.genmed_checkpoint),
        preferred_kind=str(args.genmed_checkpoint_kind),
    )
    vanilla_checkpoint_path: Path | None = None
    vanilla_checkpoint_kind: str | None = None
    if args.vanilla_checkpoint is not None:
        vanilla_checkpoint_path, vanilla_checkpoint_kind = resolve_checkpoint_path(
            checkpoint_ref=str(args.vanilla_checkpoint),
            preferred_kind=str(args.vanilla_checkpoint_kind),
        )

    gtmed_checkpoint, gtmed_checkpoint_config = _load_checkpoint(gtmed_checkpoint_path)
    gtmed_config = _prepare_config(args, gtmed_checkpoint_config)

    _assert_expected_experiment(gtmed_config, "gtmed")

    set_seed(int(gtmed_config.project.seed))
    requested_splits = _split_names(str(args.split))
    data_pipeline = build_data_pipeline(gtmed_config, requested_splits=requested_splits)

    gtmed_experiment_spec = get_experiment_spec("gtmed")
    gtmed_model, _ = gtmed_experiment_spec.build_model_and_optimizer(gtmed_config, data_pipeline.summary)
    _load_state_dict_compatibly(gtmed_model, gtmed_checkpoint["model"], gtmed_config)
    del gtmed_checkpoint
    gc.collect()

    device = resolve_device(str(gtmed_config.train.device))
    gtmed_model = gtmed_model.to(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    baseline_input_adapter = data_pipeline.input_adapter
    output_dir = (
        _build_pipeline_output_dir(gtmed_checkpoint_path, gtmed_config, str(args.split), genmed_checkpoint_path)
        if args.output_dir is None
        else Path(args.output_dir).expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(output_dir / "resolved_gtmed_eval_config.yaml", gtmed_config)

    baseline_results = _evaluate_mode(
        mode_name="gtmed_ground_truth_med",
        model=gtmed_model,
        input_adapter=baseline_input_adapter,
        data_pipeline=data_pipeline,
        device=device,
        output_dir=output_dir,
        max_batches=args.max_batches,
        save_samples=args.save_samples,
        sample_index=args.sample_index,
        args=args,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()

    vanilla_results: Dict[str, Any] | None = None
    if vanilla_checkpoint_path is not None:
        vanilla_checkpoint, vanilla_checkpoint_config = _load_checkpoint(vanilla_checkpoint_path)
        vanilla_config = _prepare_secondary_config(
            vanilla_checkpoint_config,
            shared_data_config=gtmed_config.data,
            device_name=str(gtmed_config.train.device),
        )
        _assert_expected_experiment(vanilla_config, "vanilla")
        save_config(output_dir / "resolved_vanilla_eval_config.yaml", vanilla_config)

        vanilla_data_pipeline = build_data_pipeline(vanilla_config, requested_splits=requested_splits)
        vanilla_experiment_spec = get_experiment_spec("vanilla")
        vanilla_model, _ = vanilla_experiment_spec.build_model_and_optimizer(vanilla_config, vanilla_data_pipeline.summary)
        _load_state_dict_compatibly(vanilla_model, vanilla_checkpoint["model"], vanilla_config)
        del vanilla_checkpoint
        gc.collect()

        vanilla_model = vanilla_model.to(device)
        vanilla_results = _evaluate_mode(
            mode_name="vanilla",
            model=vanilla_model,
            input_adapter=vanilla_data_pipeline.input_adapter,
            data_pipeline=vanilla_data_pipeline,
            device=device,
            output_dir=output_dir,
            max_batches=args.max_batches,
            save_samples=args.save_samples,
            sample_index=args.sample_index,
            args=args,
        )
        if device.type == "cuda":
            vanilla_model = vanilla_model.to(torch.device("cpu"))
            torch.cuda.empty_cache()
        del vanilla_model
        gc.collect()

    genmed_checkpoint, genmed_checkpoint_config = _load_checkpoint(genmed_checkpoint_path)
    genmed_device_name = str(args.genmed_device or gtmed_config.train.device)
    genmed_config = _prepare_secondary_config(
        genmed_checkpoint_config,
        shared_data_config=gtmed_config.data,
        device_name=genmed_device_name,
    )
    _assert_expected_experiment(genmed_config, "genmed")
    med_spec = _validate_med_compatibility(gtmed_config, genmed_config, data_pipeline.summary)
    save_config(output_dir / "resolved_genmed_eval_config.yaml", genmed_config)

    pipeline_mode = str(args.pipeline_mode).lower()
    genmed_device = resolve_device(genmed_device_name)
    predicted_med_cache_summary: Dict[str, Any] | None = None

    if pipeline_mode == "inline":
        genmed_experiment_spec = get_experiment_spec("genmed")
        genmed_model, _ = genmed_experiment_spec.build_model_and_optimizer(genmed_config, data_pipeline.summary)
        _load_state_dict_compatibly(genmed_model, genmed_checkpoint["model"], genmed_config)
        del genmed_checkpoint
        gc.collect()

        genmed_model = genmed_model.to(genmed_device)
        pipeline_model = GTMedWithPredictedMedPipeline(
            genmed_model=genmed_model,
            gtmed_model=gtmed_model,
            genmed_device=genmed_device,
            gtmed_device=device,
            include_med_output=bool(args.include_predicted_med_output),
        )
        pipeline_input_adapter = GTMedPipelineInputAdapter(
            genmed_config=genmed_config,
            gtmed_input_adapter=baseline_input_adapter,
        )
        pipeline_results = _evaluate_mode(
            mode_name="gtmed_with_genmed_predicted_med",
            model=pipeline_model,
            input_adapter=pipeline_input_adapter,
            data_pipeline=data_pipeline,
            device=device,
            output_dir=output_dir,
            max_batches=args.max_batches,
            save_samples=args.save_samples,
            sample_index=args.sample_index,
            args=args,
        )
    else:
        cached_pipeline_config = _prepare_cached_pipeline_config(gtmed_config)
        save_config(output_dir / "resolved_cached_pipeline_eval_config.yaml", cached_pipeline_config)
        cached_data_pipeline = build_data_pipeline(cached_pipeline_config, requested_splits=requested_splits)

        predicted_med_cache_dir = (
            _default_predicted_med_cache_dir(gtmed_checkpoint_path, gtmed_config, str(args.split), genmed_checkpoint_path)
            if args.predicted_med_cache_dir is None
            else Path(args.predicted_med_cache_dir)
        )
        predicted_med_cache = PredictedMedCache(predicted_med_cache_dir)
        need_build_cache = not (bool(args.reuse_predicted_med_cache) and predicted_med_cache.has_summary())

        gtmed_offloaded_for_cache = False
        if need_build_cache and device == genmed_device and device.type == "cuda":
            gtmed_model = gtmed_model.to(torch.device("cpu"))
            gtmed_offloaded_for_cache = True
            torch.cuda.empty_cache()

        if need_build_cache:
            genmed_experiment_spec = get_experiment_spec("genmed")
            genmed_model, _ = genmed_experiment_spec.build_model_and_optimizer(genmed_config, data_pipeline.summary)
            _load_state_dict_compatibly(genmed_model, genmed_checkpoint["model"], genmed_config)
            genmed_model = genmed_model.to(genmed_device)
            predicted_med_cache_summary = _build_predicted_med_cache(
                predicted_med_cache=predicted_med_cache,
                data_pipeline=cached_data_pipeline,
                model=genmed_model,
                input_adapter=GenMedCacheInputAdapter(genmed_config=genmed_config),
                device=genmed_device,
                args=args,
                med_spec=med_spec,
                gtmed_checkpoint_path=gtmed_checkpoint_path,
                genmed_checkpoint_path=genmed_checkpoint_path,
                gtmed_checkpoint_kind=gtmed_checkpoint_kind,
                genmed_checkpoint_kind=genmed_checkpoint_kind,
                gtmed_config=gtmed_config,
            )
            if genmed_device.type == "cuda":
                genmed_model = genmed_model.to(torch.device("cpu"))
                torch.cuda.empty_cache()
            del genmed_model
        else:
            predicted_med_cache_summary = {
                "cache_dir": str(predicted_med_cache.root_dir),
                "summary_path": str(predicted_med_cache.summary_path),
                "reused": True,
                "requested_split": str(args.split),
                "max_batches": None if args.max_batches is None else int(args.max_batches),
                "med_spec": med_spec,
            }

        del genmed_checkpoint
        gc.collect()
        if genmed_device.type == "cuda":
            torch.cuda.empty_cache()

        if gtmed_offloaded_for_cache:
            gtmed_model = gtmed_model.to(device)
            torch.cuda.empty_cache()

        pipeline_results = _evaluate_mode(
            mode_name="gtmed_with_genmed_predicted_med",
            model=gtmed_model,
            input_adapter=GTMedCachedMedInputAdapter(
                gtmed_input_adapter=baseline_input_adapter,
                predicted_med_cache=predicted_med_cache,
            ),
            data_pipeline=cached_data_pipeline,
            device=device,
            output_dir=output_dir,
            max_batches=args.max_batches,
            save_samples=args.save_samples,
            sample_index=args.sample_index,
            args=args,
            data_summary=data_pipeline.summary,
        )

    comparison: Dict[str, Any] = {
        "gtmed_ground_truth_med_vs_gtmed_with_genmed_predicted_med": _build_pairwise_comparison(
            baseline_results,
            pipeline_results,
            reference_label="gtmed_ground_truth_med",
            candidate_label="gtmed_with_genmed_predicted_med",
        )
    }
    if vanilla_results is not None:
        comparison["vanilla_vs_gtmed_ground_truth_med"] = _build_pairwise_comparison(
            vanilla_results,
            baseline_results,
            reference_label="vanilla",
            candidate_label="gtmed_ground_truth_med",
        )
        comparison["vanilla_vs_gtmed_with_genmed_predicted_med"] = _build_pairwise_comparison(
            vanilla_results,
            pipeline_results,
            reference_label="vanilla",
            candidate_label="gtmed_with_genmed_predicted_med",
        )

    top_level_summary = {
        "gtmed_checkpoint": str(gtmed_checkpoint_path),
        "gtmed_checkpoint_resolution": gtmed_checkpoint_kind,
        "genmed_checkpoint": str(genmed_checkpoint_path),
        "genmed_checkpoint_resolution": genmed_checkpoint_kind,
        "vanilla_checkpoint": None if vanilla_checkpoint_path is None else str(vanilla_checkpoint_path),
        "vanilla_checkpoint_resolution": vanilla_checkpoint_kind,
        "experiment": "gtmed_pipeline",
        "pipeline_mode": pipeline_mode,
        "data_name": str(getattr(gtmed_config.data, "name", "unknown")),
        "requested_split": str(args.split),
        "device": str(device),
        "genmed_device": str(genmed_device),
        "loader_mode": str(args.loader_mode),
        "visualize": str(args.visualize),
        "include_predicted_med_output": bool(args.include_predicted_med_output),
        "predicted_med_cache": predicted_med_cache_summary,
        "results": {
            "vanilla": vanilla_results,
            "gtmed_ground_truth_med": baseline_results,
            "gtmed_with_genmed_predicted_med": pipeline_results,
        },
        "comparison": comparison,
    }
    save_json(output_dir / "summary.json", top_level_summary)

    print(f"resolved GTMed checkpoint: {gtmed_checkpoint_path} ({gtmed_checkpoint_kind})")
    print(f"resolved GenMed checkpoint: {genmed_checkpoint_path} ({genmed_checkpoint_kind})")
    if vanilla_checkpoint_path is not None:
        print(f"resolved Vanilla checkpoint: {vanilla_checkpoint_path} ({vanilla_checkpoint_kind})")
    print(f"pipeline mode: {pipeline_mode}")
    if predicted_med_cache_summary is not None:
        print(f"predicted-med cache: {predicted_med_cache_summary.get('cache_dir')}")
        if predicted_med_cache_summary.get("reused"):
            print("predicted-med cache status: reused")
        else:
            print("predicted-med cache status: rebuilt")
    print(f"saved pipeline evaluation summary to {output_dir / 'summary.json'}")
    reported_modes = [
        ("gtmed_ground_truth_med", baseline_results),
        ("gtmed_with_genmed_predicted_med", pipeline_results),
    ]
    if vanilla_results is not None:
        reported_modes.insert(0, ("vanilla", vanilla_results))
    for mode_name, mode_results in reported_modes:
        print(f"[{mode_name}]")
        for split_name, result in mode_results.items():
            for line in format_split_metrics(split_name, result):
                print(line)
            if result.get("visualization_paths"):
                print(f"  visualizations: {len(result['visualization_paths'])}")
            for note in result.get("visualization_notes", []):
                print(f"  note: {note}")
    if comparison:
        print("[comparison]")
        for comparison_name, comparison_payload in comparison.items():
            if not comparison_payload:
                continue
            print(f"{comparison_name}:")
            for split_name, split_comparison in comparison_payload.items():
                print(f"  {split_name}:")
                for prediction_key, prediction_metrics in split_comparison.items():
                    for metric_name, metric_values in prediction_metrics.items():
                        delta = float(metric_values["delta"])
                        relative = metric_values["relative_delta"]
                        relative_text = "-" if relative is None else f"{relative * 100.0:.2f}%"
                        labels = [key for key in metric_values.keys() if key not in {"delta", "relative_delta"}]
                        if len(labels) != 2:
                            continue
                        reference_label, candidate_label = labels
                        print(
                            f"    {prediction_key}.{metric_name}: "
                            f"{reference_label}={metric_values[reference_label]:.6f} "
                            f"{candidate_label}={metric_values[candidate_label]:.6f} "
                            f"delta={delta:.6f} relative={relative_text}"
                        )


if __name__ == "__main__":
    main()
