from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from Datasets.registry import build_data_pipeline
from Datasets.config_utils import safe_config_get
from Training.engine import resolve_device
from Training.progress import resolve_progress_settings
from Training.registry import get_experiment_spec
from utils.runtime import make_run_dir, resolve_run_name, set_seed


def run_experiment(config: Any) -> Tuple[Dict[str, Any], Path]:
    set_seed(int(config.project.seed))
    experiment_name = str(config.experiment.name)
    spec = get_experiment_spec(experiment_name)

    use_validation = bool(safe_config_get(config.train, "use_validation", True))
    requested_splits = ["train", "val"] if use_validation else ["train"]
    data_pipeline = build_data_pipeline(config, requested_splits=requested_splits)
    model, optimizer = spec.build_model_and_optimizer(config, data_pipeline.summary)
    trainer = spec.trainer_cls(
        model=model,
        input_adapter=data_pipeline.input_adapter,
        optimizer=optimizer,
        device=resolve_device(str(config.train.device)),
        loss_weights=dict(config.experiment.loss_weights),
        progress_settings=resolve_progress_settings(config.train),
    )
    run_name = resolve_run_name(spec.name, config.train.run_name)
    run_dir = make_run_dir(config.paths.output_root, spec.name, run_name)
    summary = trainer.fit(
        train_loader=data_pipeline.train_loader,
        val_loader=data_pipeline.val_loader,
        epochs=int(config.train.epochs),
        run_dir=run_dir,
        config=config,
        data_summary=data_pipeline.summary,
    )
    print(f"finished {spec.name} -> {run_dir}")
    if summary.get("best_val_total_loss") is not None:
        print(f"best_val_total_loss={summary['best_val_total_loss']:.6f}")
    else:
        print(f"best_{summary.get('best_selection_split', 'train')}_total_loss={summary['best_total_loss']:.6f}")
    return summary, run_dir
