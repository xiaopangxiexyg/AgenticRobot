from __future__ import annotations

import argparse
import contextlib
import gc
import io
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from Datasets.registry import build_data_pipeline
from Models.med_support import normalize_med_spec
from Training.engine import resolve_device
from Training.registry import get_experiment_spec
from utils.path_config import load_local_env
from utils.runtime import save_config, save_json, set_seed

from eval_vis_utils import (
    format_split_metrics,
    maybe_render_point_cloud_projection_artifacts,
    resolve_checkpoint_path,
)


CONFIGS_DIR = PROJECT_ROOT / "Configs"
load_local_env()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint or run directory, save metrics, and optionally render sample visualizations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default=None,
        help="Checkpoint file path, run directory, or run name under Outputs/",
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint_flag",
        type=str,
        default=None,
        help="Alias for the positional checkpoint argument.",
    )
    parser.add_argument("--checkpoint-kind", type=str, default="auto", choices=["auto", "best", "last"])
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "both"])
    parser.add_argument(
        "--data-config",
        type=str,
        default=None,
        help="Data preset under Configs/data or a standalone resolved YAML path.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--loader-mode", type=str, default="safe", choices=["safe", "checkpoint"])
    parser.add_argument("--batch-size", type=int, default=None, help="Optional loader batch size override.")
    parser.add_argument("--num-workers", type=int, default=None, help="Optional loader worker-count override.")
    parser.add_argument("--max-batches", type=int, default=None, help="Stop after this many batches per split.")
    parser.add_argument("--save-samples", type=int, default=1, help="How many sample dumps to save per split.")
    parser.add_argument("--sample-index", type=int, default=0, help="Which saved sample to visualize.")
    parser.add_argument("--max-points", type=int, default=512, help="Point cap for point-cloud plots and projections.")
    parser.add_argument(
        "--visualize",
        type=str,
        default="all",
        choices=["all", "plots", "projection", "none"],
        help="Sample visualization mode.",
    )
    parser.add_argument(
        "--projection-point-radius",
        type=int,
        default=4,
        help="Projected point radius for point-cloud video overlays.",
    )
    parser.add_argument(
        "--projection-frame-height",
        type=int,
        default=360,
        help="Output frame height for point-cloud comparison grids/videos.",
    )
    parser.add_argument(
        "--projection-max-steps",
        type=int,
        default=0,
        help="Optional cap on rendered future MED steps for point-cloud projections. 0 means all.",
    )
    parser.add_argument("--camera-serial", type=str, default="", help="Optional override for projection camera serial.")
    parser.add_argument(
        "--camera-intrinsics-json",
        type=str,
        default="",
        help="Optional camera intrinsics JSON for point-cloud video projection.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="OmegaConf dotlist override, for example data.raw.loader.batch_size=1",
    )
    return parser.parse_args()


def _resolve_requested_checkpoint(args: argparse.Namespace) -> str:
    checkpoint_ref = args.checkpoint_flag or args.checkpoint
    if checkpoint_ref is None:
        raise ValueError("Please provide a checkpoint path, run directory, or run name.")
    return str(checkpoint_ref)


def _load_checkpoint(checkpoint_path: Path) -> tuple[dict[str, Any], Any]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "config" not in checkpoint:
        raise KeyError(f"Checkpoint missing config payload: {checkpoint_path}")
    return checkpoint, OmegaConf.create(checkpoint["config"])


def _sanitize_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("_") or "eval"


def _maybe_import_matplotlib():
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            import matplotlib.pyplot as plt
    except Exception:
        return None
    return plt


def _compose_data_preset(preset_name: str) -> Any:
    preset_name = str(preset_name)
    if preset_name.endswith(".yaml"):
        preset_name = preset_name[:-5]
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS_DIR)):
        composed = compose(config_name="train_runner", overrides=[f"data={preset_name}"])
    return composed.data


def _load_data_config(data_config_ref: str) -> Any:
    candidate = Path(data_config_ref).expanduser()
    if candidate.exists():
        resolved = candidate.resolve()
        configs_data_dir = (CONFIGS_DIR / "data").resolve()
        if configs_data_dir in resolved.parents:
            return _compose_data_preset(resolved.stem)
        config = OmegaConf.load(resolved)
        if "defaults" in config:
            raise ValueError(
                "Direct YAML paths with Hydra defaults are not supported here. "
                "Pass the preset name under Configs/data, or provide a standalone resolved YAML."
            )
        return config
    return _compose_data_preset(str(data_config_ref))


def _prepare_config(args: argparse.Namespace, checkpoint_config: Any) -> Any:
    config = OmegaConf.create(OmegaConf.to_container(checkpoint_config, resolve=True))
    if args.data_config is not None:
        config.data = _load_data_config(args.data_config)

    raw_video_config = getattr(getattr(config.data, "raw", None), "video", None)
    if raw_video_config is not None and hasattr(raw_video_config, "emit_frames"):
        raw_video_config.emit_frames = False

    loader = config.data.raw.loader
    if args.loader_mode == "safe":
        loader.batch_size = 1
        loader.num_workers = 0
        loader.pin_memory = False
        if hasattr(loader, "persistent_workers"):
            loader.persistent_workers = False
        if hasattr(loader, "prefetch_factor"):
            loader.prefetch_factor = None

    if args.batch_size is not None:
        loader.batch_size = int(args.batch_size)
    if args.num_workers is not None:
        loader.num_workers = int(args.num_workers)
        if int(args.num_workers) <= 0:
            if hasattr(loader, "persistent_workers"):
                loader.persistent_workers = False
            if hasattr(loader, "prefetch_factor"):
                loader.prefetch_factor = None

    if args.device is not None:
        config.train.device = str(args.device)
    if args.overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(args.overrides)))
    return config


def _adapt_legacy_state_dict(model: torch.nn.Module, state_dict: Dict[str, Any], config: Any) -> Dict[str, Any]:
    experiment_name = str(getattr(config.experiment, "name", "")).lower()
    model_family = str(getattr(config.model, "family", "mlp")).lower()
    if experiment_name != "vanilla" or model_family not in {"vla", "diffusion"}:
        return state_dict

    model_keys = list(model.state_dict().keys())
    checkpoint_keys = list(state_dict.keys())
    model_uses_core_prefix = any(key.startswith("core.") for key in model_keys)
    checkpoint_uses_core_prefix = any(key.startswith("core.") for key in checkpoint_keys)

    if model_uses_core_prefix and not checkpoint_uses_core_prefix:
        return {f"core.{key}": value for key, value in state_dict.items()}
    if checkpoint_uses_core_prefix and not model_uses_core_prefix:
        return {key[5:] if key.startswith("core.") else key: value for key, value in state_dict.items()}
    return state_dict


def _load_state_dict_compatibly(model: torch.nn.Module, state_dict: Dict[str, Any], config: Any) -> None:
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError as original_error:
        adapted_state_dict = _adapt_legacy_state_dict(model, state_dict, config)
        if adapted_state_dict is state_dict:
            raise
        try:
            model.load_state_dict(adapted_state_dict)
            return
        except RuntimeError:
            raise original_error


def _default_output_dir(checkpoint_path: Path, config: Any, split_name: str) -> Path:
    data_name = _sanitize_name(str(getattr(config.data, "name", "data")))
    split_tag = _sanitize_name(split_name)
    return checkpoint_path.parent / "evaluation" / f"{data_name}_{split_tag}"


def _metric_keys(outputs: Dict[str, Any], targets: Dict[str, Any]) -> List[str]:
    keys = []
    for key, target in targets.items():
        if key not in outputs:
            continue
        if not torch.is_tensor(outputs[key]) or not torch.is_tensor(target):
            continue
        keys.append(str(key))
    return sorted(keys)


def _detach_to_cpu(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(dtype=torch.float32).cpu()


def _point_values(points: torch.Tensor) -> torch.Tensor:
    if int(points.shape[-1]) <= 3:
        return torch.zeros(points.shape[0], dtype=torch.float32)
    extra = points[:, 3:]
    if int(extra.shape[-1]) == 1:
        return extra[:, 0]
    return extra.mean(dim=-1)


def _save_dense_plot(plt: Any, prediction: torch.Tensor, target: torch.Tensor, output_path: Path, title: str) -> None:
    if prediction.dim() <= 1:
        figure, axis = plt.subplots(1, 1, figsize=(10, 4))
        axis.plot(target.numpy(), label="target", linewidth=2.0)
        axis.plot(prediction.numpy(), label="prediction", linewidth=1.5)
        axis.set_title(title)
        axis.set_xlabel("feature")
        axis.legend()
    else:
        figure, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].imshow(target.numpy(), aspect="auto", cmap="viridis")
        axes[0].set_title(f"Target {title}")
        axes[0].set_xlabel("feature")
        axes[0].set_ylabel("step")
        axes[1].imshow(prediction.numpy(), aspect="auto", cmap="viridis")
        axes[1].set_title(f"Predicted {title}")
        axes[1].set_xlabel("feature")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _save_point_cloud_plot(plt: Any, prediction: torch.Tensor, target: torch.Tensor, output_path: Path, max_points: int) -> None:
    if prediction.dim() == 3:
        prediction = prediction[0]
    if target.dim() == 3:
        target = target[0]
    prediction = prediction[:max_points]
    target = target[:max_points]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, points, title in (
        (axes[0], target, "Target Point Cloud"),
        (axes[1], prediction, "Predicted Point Cloud"),
    ):
        color_values = _point_values(points).numpy()
        axis.scatter(
            points[:, 0].numpy(),
            points[:, 1].numpy(),
            c=color_values,
            s=18,
            cmap="viridis",
            alpha=0.9,
        )
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _save_sample_visualizations(
    *,
    sample_payload: Dict[str, Any],
    split_dir: Path,
    sample_id: int,
    args: argparse.Namespace,
) -> tuple[List[str], List[str]]:
    image_paths: List[str] = []
    notes: List[str] = []
    predictions = sample_payload["predictions"]
    targets = sample_payload["targets"]
    med_spec = sample_payload.get("med_spec")
    normalized_med_spec = None if med_spec is None else normalize_med_spec(med_spec)

    if args.visualize in {"all", "plots"}:
        plt = _maybe_import_matplotlib()
        if plt is None:
            notes.append("matplotlib is not installed; skipped plot rendering")
        else:
            for key, prediction in predictions.items():
                if key not in targets:
                    continue
                target = targets[key]
                output_path = split_dir / f"sample_{sample_id:03d}_{key}.png"
                if key == "med" and normalized_med_spec is not None and str(normalized_med_spec["kind"]).lower() == "point_cloud":
                    _save_point_cloud_plot(plt, prediction, target, output_path, max_points=int(args.max_points))
                else:
                    _save_dense_plot(plt, prediction, target, output_path, title=key.upper())
                image_paths.append(str(output_path))

    if args.visualize in {"all", "projection"}:
        projection_result = maybe_render_point_cloud_projection_artifacts(
            sample_payload=sample_payload,
            output_dir=split_dir,
            file_stem=f"sample_{sample_id:03d}_med",
            max_points=int(args.max_points),
            point_radius=int(args.projection_point_radius),
            frame_height=int(args.projection_frame_height),
            camera_serial=str(args.camera_serial),
            camera_intrinsics_json=str(args.camera_intrinsics_json),
            max_steps=None if int(args.projection_max_steps) <= 0 else int(args.projection_max_steps),
        )
        image_paths.extend(projection_result["paths"])
        notes.extend(projection_result["notes"])

    return image_paths, notes


def _split_names(requested_split: str) -> List[str]:
    if requested_split == "both":
        return ["train", "val"]
    return [requested_split]


def _pick_loader(data_pipeline: Any, split_name: str):
    loader = data_pipeline.train_loader if split_name == "train" else data_pipeline.val_loader
    if loader is None:
        raise ValueError(
            f"Requested split={split_name!r}, but the current evaluation data pipeline did not build that loader."
        )
    return loader


def _save_sample_dump(
    split_dir: Path,
    sample_id: int,
    metadata: Dict[str, Any],
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    med_spec: Dict[str, Any] | None,
) -> Path:
    sample_path = split_dir / "samples" / f"sample_{sample_id:03d}.pt"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": metadata,
            "predictions": predictions,
            "targets": targets,
            "med_spec": med_spec,
        },
        sample_path,
    )
    return sample_path


def _evaluate_split(
    *,
    split_name: str,
    loader: Any,
    model: torch.nn.Module,
    input_adapter: Any,
    device: torch.device,
    output_dir: Path,
    data_summary: Dict[str, Any],
    max_batches: int | None,
    save_samples: int,
    sample_index: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    metric_sums: Dict[str, Dict[str, float]] = {}
    sample_paths: List[str] = []
    visualization_payload: Dict[str, Any] | None = None
    last_saved_payload: Dict[str, Any] | None = None
    batches = 0
    total_samples = 0
    discovered_keys: List[str] = []

    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            prepared = input_adapter.prepare_batch(batch, device)
            outputs = model(**prepared.model_inputs)
            metric_keys = _metric_keys(outputs, prepared.targets)
            if not metric_keys:
                raise KeyError(
                    f"No overlapping tensor outputs/targets for split={split_name}. "
                    f"output_keys={sorted(outputs.keys())}, target_keys={sorted(prepared.targets.keys())}"
                )
            discovered_keys = metric_keys
            batch_size = len(batch["metadata"])
            batches += 1
            total_samples += batch_size

            cpu_outputs = {key: _detach_to_cpu(outputs[key]) for key in metric_keys}
            cpu_targets = {key: _detach_to_cpu(prepared.targets[key]) for key in metric_keys}

            for key in metric_keys:
                prediction = cpu_outputs[key]
                target = cpu_targets[key]
                diff = prediction - target
                stats = metric_sums.setdefault(key, {"squared_error_sum": 0.0, "absolute_error_sum": 0.0, "element_count": 0.0})
                stats["squared_error_sum"] += float(diff.pow(2).sum().item())
                stats["absolute_error_sum"] += float(diff.abs().sum().item())
                stats["element_count"] += float(diff.numel())

            remaining = max(0, int(save_samples) - len(sample_paths))
            if remaining > 0:
                med_spec = data_summary.get("med_spec") or data_summary.get(split_name, {}).get("med_spec")
                for local_index in range(min(remaining, batch_size)):
                    sample_id = len(sample_paths)
                    payload = {
                        "metadata": batch["metadata"][local_index],
                        "predictions": {key: cpu_outputs[key][local_index].clone() for key in metric_keys},
                        "targets": {key: cpu_targets[key][local_index].clone() for key in metric_keys},
                        "med_spec": med_spec,
                    }
                    sample_path = _save_sample_dump(
                        split_dir=split_dir,
                        sample_id=sample_id,
                        metadata=payload["metadata"],
                        predictions=payload["predictions"],
                        targets=payload["targets"],
                        med_spec=payload["med_spec"],
                    )
                    sample_paths.append(str(sample_path))
                    last_saved_payload = payload
                    if sample_id == int(sample_index):
                        visualization_payload = payload

            del prepared
            del outputs
            del cpu_outputs
            del cpu_targets

    metric_summary: Dict[str, Dict[str, float]] = {}
    for key, stats in metric_sums.items():
        element_count = max(1.0, float(stats["element_count"]))
        mse = float(stats["squared_error_sum"]) / element_count
        mae = float(stats["absolute_error_sum"]) / element_count
        metric_summary[key] = {
            "mse": mse,
            "rmse": mse ** 0.5,
            "mae": mae,
            "element_count": int(element_count),
        }

    visualization_paths: List[str] = []
    visualization_notes: List[str] = []
    if sample_paths and args.visualize != "none":
        if visualization_payload is None:
            visualization_payload = last_saved_payload
        visual_index = max(0, min(int(sample_index), len(sample_paths) - 1))
        visualization_paths, visualization_notes = _save_sample_visualizations(
            sample_payload=visualization_payload,
            split_dir=split_dir,
            sample_id=visual_index,
            args=args,
        )

    split_summary = {
        "split": split_name,
        "batches": int(batches),
        "samples": int(total_samples),
        "prediction_keys": list(discovered_keys),
        "metrics": metric_summary,
        "sample_paths": sample_paths,
        "visualization_paths": visualization_paths,
        "visualization_notes": visualization_notes,
    }
    save_json(split_dir / "summary.json", split_summary)
    return split_summary


def main() -> None:
    args = _parse_args()
    checkpoint_ref = _resolve_requested_checkpoint(args)
    checkpoint_path, checkpoint_kind = resolve_checkpoint_path(
        checkpoint_ref=checkpoint_ref,
        preferred_kind=str(args.checkpoint_kind),
    )
    checkpoint, checkpoint_config = _load_checkpoint(checkpoint_path)
    config = _prepare_config(args, checkpoint_config)
    set_seed(int(config.project.seed))

    requested_splits = _split_names(str(args.split))
    data_pipeline = build_data_pipeline(config, requested_splits=requested_splits)
    experiment_name = str(config.experiment.name)
    experiment_spec = get_experiment_spec(experiment_name)
    model, _ = experiment_spec.build_model_and_optimizer(config, data_pipeline.summary)
    _load_state_dict_compatibly(model, checkpoint["model"], config)
    del checkpoint
    gc.collect()
    device = resolve_device(str(config.train.device))
    model = model.to(device)

    output_dir = (
        _default_output_dir(checkpoint_path, config, args.split)
        if args.output_dir is None
        else Path(args.output_dir).expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(output_dir / "resolved_eval_config.yaml", config)

    split_results: Dict[str, Any] = {}
    for split_name in _split_names(str(args.split)):
        split_results[split_name] = _evaluate_split(
            split_name=split_name,
            loader=_pick_loader(data_pipeline, split_name),
            model=model,
            input_adapter=data_pipeline.input_adapter,
            device=device,
            output_dir=output_dir,
            data_summary=data_pipeline.summary,
            max_batches=args.max_batches,
            save_samples=args.save_samples,
            sample_index=args.sample_index,
            args=args,
        )

    top_level_summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_resolution": checkpoint_kind,
        "requested_checkpoint": checkpoint_ref,
        "experiment": experiment_name,
        "model_family": str(getattr(config.model, "family", "mlp")),
        "data_name": str(getattr(config.data, "name", "unknown")),
        "loader_mode": str(args.loader_mode),
        "requested_split": str(args.split),
        "visualize": str(args.visualize),
        "device": str(device),
        "results": split_results,
    }
    save_json(output_dir / "summary.json", top_level_summary)

    print(f"resolved checkpoint: {checkpoint_path} ({checkpoint_kind})")
    print(f"saved evaluation summary to {output_dir / 'summary.json'}")
    for split_name, result in split_results.items():
        for line in format_split_metrics(split_name, result):
            print(line)
        if result.get("visualization_paths"):
            print(f"  visualizations: {len(result['visualization_paths'])}")
        for note in result.get("visualization_notes", []):
            print(f"  note: {note}")


if __name__ == "__main__":
    main()
