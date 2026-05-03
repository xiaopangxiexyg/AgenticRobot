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

from Datasets.registry import build_data_pipeline
from Models.med_support import normalize_med_spec
from Training.engine import resolve_device
from Training.registry import get_experiment_spec
from utils.runtime import set_seed

from eval_vis_utils import maybe_render_point_cloud_projection_artifacts, resolve_checkpoint_path
from evaluate_checkpoint import (
    _load_checkpoint,
    _load_state_dict_compatibly,
    _maybe_import_matplotlib,
    _pick_loader,
    _prepare_config,
    _save_dense_plot,
    _save_point_cloud_plot,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a single-sample GenMed prediction from a checkpoint, run directory, or run name.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default=None,
        help="Checkpoint file path, run directory, or run name under Outputs/.",
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint_flag",
        type=str,
        default=None,
        help="Alias for the positional checkpoint argument.",
    )
    parser.add_argument("--checkpoint-kind", type=str, default="auto", choices=["auto", "best", "last"])
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--batch-index", type=int, default=0, help="Which loader batch to visualize.")
    parser.add_argument("--sample-index", type=int, default=0, help="Which sample inside the batch to visualize.")
    parser.add_argument(
        "--data-config",
        type=str,
        default=None,
        help="Optional data preset under Configs/data or a standalone resolved YAML path.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--loader-mode", type=str, default="safe", choices=["safe", "checkpoint"])
    parser.add_argument("--batch-size", type=int, default=None, help="Optional loader batch size override.")
    parser.add_argument("--num-workers", type=int, default=None, help="Optional loader worker-count override.")
    parser.add_argument("--max-points", type=int, default=512)
    parser.add_argument("--visualize", type=str, default="all", choices=["all", "plots", "projection", "none"])
    parser.add_argument("--projection-point-radius", type=int, default=4)
    parser.add_argument("--projection-frame-height", type=int, default=360)
    parser.add_argument("--projection-max-steps", type=int, default=0)
    parser.add_argument("--camera-serial", type=str, default="")
    parser.add_argument("--camera-intrinsics-json", type=str, default="")
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


def _build_single_sample_payload(
    *,
    outputs: Dict[str, torch.Tensor],
    prepared: Any,
    batch: Dict[str, Any],
    sample_index: int,
    med_spec: Dict[str, Any] | None,
) -> Dict[str, Any]:
    target_keys = {str(key) for key, value in prepared.targets.items() if torch.is_tensor(value)}
    return {
        "metadata": batch["metadata"][sample_index],
        "predictions": {
            key: value.detach().to(dtype=torch.float32).cpu()[sample_index].clone()
            for key, value in outputs.items()
            if torch.is_tensor(value) and str(key) in target_keys
        },
        "targets": {
            key: value.detach().to(dtype=torch.float32).cpu()[sample_index].clone()
            for key, value in prepared.targets.items()
            if torch.is_tensor(value)
        },
        "med_spec": med_spec,
    }


def _save_plot_artifacts(
    *,
    sample_payload: Dict[str, Any],
    output_dir: Path,
    max_points: int,
) -> List[str]:
    if "med" not in sample_payload["predictions"] or "med" not in sample_payload["targets"]:
        return []
    plt = _maybe_import_matplotlib()
    if plt is None:
        return []

    med_spec = sample_payload.get("med_spec")
    med_kind = str((med_spec or {}).get("kind", "vector")).lower()
    prediction = sample_payload["predictions"]["med"]
    target = sample_payload["targets"]["med"]

    if med_kind == "point_cloud":
        output_path = output_dir / "med_point_cloud.png"
        _save_point_cloud_plot(plt, prediction, target, output_path, max_points=max_points)
    else:
        output_path = output_dir / "med_vector.png"
        _save_dense_plot(plt, prediction, target, output_path, title="MED")
    return [str(output_path)]


def _locate_batch(loader: Any, batch_index: int) -> Dict[str, Any]:
    requested_index = max(0, int(batch_index))
    for current_index, batch in enumerate(loader):
        if current_index == requested_index:
            return batch
    raise IndexError(f"Requested batch_index={requested_index}, but the loader ended earlier.")


def main() -> None:
    args = _parse_args()
    checkpoint_ref = _resolve_requested_checkpoint(args)
    checkpoint_path, checkpoint_kind = resolve_checkpoint_path(
        checkpoint_ref=checkpoint_ref,
        preferred_kind=str(args.checkpoint_kind),
    )
    checkpoint, config = _load_checkpoint(checkpoint_path)
    config = _prepare_config(args, config)
    set_seed(int(config.project.seed))

    data_pipeline = build_data_pipeline(config, requested_splits=[str(args.split)])
    experiment_spec = get_experiment_spec(str(config.experiment.name))
    model, _ = experiment_spec.build_model_and_optimizer(config, data_pipeline.summary)
    _load_state_dict_compatibly(model, checkpoint["model"], config)
    del checkpoint
    gc.collect()
    device = resolve_device(str(config.train.device))
    model = model.to(device)
    model.eval()

    loader = _pick_loader(data_pipeline, str(args.split))
    batch = _locate_batch(loader, int(args.batch_index))
    prepared = data_pipeline.input_adapter.prepare_batch(batch, device)
    with torch.no_grad():
        outputs = model(**prepared.model_inputs)
    if "med" not in outputs or "med" not in prepared.targets:
        raise KeyError(
            f"Expected 'med' in both outputs and targets. "
            f"output_keys={sorted(outputs.keys())}, target_keys={sorted(prepared.targets.keys())}"
        )

    batch_size = int(outputs["med"].shape[0])
    sample_index = max(0, min(int(args.sample_index), batch_size - 1))
    med_spec = data_pipeline.summary.get("med_spec") or data_pipeline.summary[str(args.split)].get("med_spec")
    normalized_med_spec = None if med_spec is None else normalize_med_spec(med_spec)
    sample_payload = _build_single_sample_payload(
        outputs=outputs,
        prepared=prepared,
        batch=batch,
        sample_index=sample_index,
        med_spec=normalized_med_spec,
    )

    output_dir = (
        checkpoint_path.parent / "visualizations"
        if args.output_dir is None
        else Path(args.output_dir).expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = output_dir / "med_prediction_sample.pt"
    torch.save(sample_payload, tensor_path)

    saved_paths: List[str] = [str(tensor_path)]
    notes: List[str] = []

    if args.visualize in {"all", "plots"}:
        plot_paths = _save_plot_artifacts(
            sample_payload=sample_payload,
            output_dir=output_dir,
            max_points=int(args.max_points),
        )
        if not plot_paths:
            notes.append("matplotlib is not installed or med tensors were unavailable; skipped plot rendering")
        saved_paths.extend(plot_paths)

    if args.visualize in {"all", "projection"}:
        projection_result = maybe_render_point_cloud_projection_artifacts(
            sample_payload=sample_payload,
            output_dir=output_dir,
            file_stem="med_prediction_sample",
            max_points=int(args.max_points),
            point_radius=int(args.projection_point_radius),
            frame_height=int(args.projection_frame_height),
            camera_serial=str(args.camera_serial),
            camera_intrinsics_json=str(args.camera_intrinsics_json),
            max_steps=None if int(args.projection_max_steps) <= 0 else int(args.projection_max_steps),
        )
        saved_paths.extend(projection_result["paths"])
        notes.extend(projection_result["notes"])

    print(f"resolved checkpoint: {checkpoint_path} ({checkpoint_kind})")
    print(f"saved sample payload to {tensor_path}")
    for path in saved_paths[1:]:
        print(f"saved visualization to {path}")
    for note in notes:
        print(f"note: {note}")


if __name__ == "__main__":
    main()
