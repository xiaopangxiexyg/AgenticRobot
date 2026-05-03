from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict

from omegaconf import OmegaConf

from utils.runtime import save_json


class BaseExperimentTracker:
    name = "none"

    def log_epoch(self, epoch_metrics: Dict[str, Any]) -> None:
        del epoch_metrics

    def finish(self, summary: Dict[str, Any]) -> None:
        del summary


class NullExperimentTracker(BaseExperimentTracker):
    name = "none"


class LocalExperimentTracker(BaseExperimentTracker):
    name = "local"

    def __init__(self, run_dir: Path, epoch_metrics_filename: str = "epoch_metrics.jsonl") -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.epoch_metrics_path = self.run_dir / str(epoch_metrics_filename)
        self.latest_epoch_metrics_path = self.run_dir / "latest_epoch_metrics.json"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def log_epoch(self, epoch_metrics: Dict[str, Any]) -> None:
        self.epoch_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.epoch_metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_metrics, sort_keys=True) + "\n")
        save_json(self.latest_epoch_metrics_path, epoch_metrics)

    def finish(self, summary: Dict[str, Any]) -> None:
        del summary


class WandbExperimentTracker(BaseExperimentTracker):
    name = "wandb"

    def __init__(self, run: Any) -> None:
        self.run = run

    def log_epoch(self, epoch_metrics: Dict[str, Any]) -> None:
        payload = {
            "epoch": int(epoch_metrics["epoch"]),
        }
        for split_name in ("train", "val"):
            split_metrics = epoch_metrics.get(split_name, {})
            for key, value in split_metrics.items():
                payload[f"{split_name}/{key}"] = float(value)
        self.run.log(payload, step=int(epoch_metrics["epoch"]))

    def finish(self, summary: Dict[str, Any]) -> None:
        self.run.summary.update(
            {
                "experiment": summary.get("experiment"),
                "best_val_total_loss": summary.get("best_val_total_loss"),
                "epochs": summary.get("epochs"),
                "tracker": summary.get("tracker"),
            }
        )
        self.run.finish()


class CompositeExperimentTracker(BaseExperimentTracker):
    def __init__(self, trackers: list[BaseExperimentTracker]) -> None:
        self.trackers = trackers
        tracker_names = [tracker.name for tracker in trackers if tracker.name != "none"]
        self.name = "+".join(tracker_names) if tracker_names else "none"

    def log_epoch(self, epoch_metrics: Dict[str, Any]) -> None:
        for tracker in self.trackers:
            tracker.log_epoch(epoch_metrics)

    def finish(self, summary: Dict[str, Any]) -> None:
        for tracker in self.trackers:
            tracker.finish(summary)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(str(name))
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _wandb_disabled_by_environment() -> bool:
    mode = os.environ.get("WANDB_MODE", "").strip().lower()
    return _env_bool("WANDB_DISABLED") or _env_bool("MANIPREP_DISABLE_WANDB") or mode == "disabled"


def build_experiment_tracker(config: Any, run_dir: Path, data_summary: Dict[str, Any]) -> BaseExperimentTracker:
    trackers: list[BaseExperimentTracker] = []
    local_tracking_config = getattr(config.train, "local_tracking", None)
    local_tracking_enabled = True if local_tracking_config is None else bool(getattr(local_tracking_config, "enabled", True))
    if local_tracking_enabled:
        epoch_metrics_filename = "epoch_metrics.jsonl"
        if local_tracking_config is not None:
            epoch_metrics_filename = str(getattr(local_tracking_config, "epoch_metrics_filename", epoch_metrics_filename))
        trackers.append(LocalExperimentTracker(run_dir=run_dir, epoch_metrics_filename=epoch_metrics_filename))

    wandb_config = getattr(config.train, "wandb", None)
    if wandb_config is None or not bool(wandb_config.enabled) or _wandb_disabled_by_environment():
        return CompositeExperimentTracker(trackers) if trackers else NullExperimentTracker()

    try:
        import wandb
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "wandb is enabled in train.wandb, but the current environment does not have the 'wandb' package installed. "
            "Install wandb first, or set train.wandb.enabled=false for local dry-runs."
        ) from error

    try:
        run = wandb.init(
            project=str(wandb_config.project),
            entity=None if wandb_config.entity in (None, "null") else str(wandb_config.entity),
            name=run_dir.name,
            dir=str(run_dir),
            mode=str(wandb_config.mode),
            group=None if wandb_config.group in (None, "null") else str(wandb_config.group),
            job_type=str(wandb_config.job_type),
            tags=[str(tag) for tag in list(wandb_config.tags)],
            config={
                "config": OmegaConf.to_container(config, resolve=True),
                "data_summary": data_summary,
            },
            reinit=True,
        )
    except Exception as error:
        if _env_bool("MANIPREP_STRICT_WANDB"):
            raise
        warnings.warn(
            "wandb initialization failed; continuing with local tracking only. "
            "Set train.wandb.enabled=false or WANDB_DISABLED=true to skip wandb explicitly. "
            "Set MANIPREP_STRICT_WANDB=1 to make this error fatal. "
            f"Original error: {type(error).__name__}: {error}",
            RuntimeWarning,
        )
        return CompositeExperimentTracker(trackers) if trackers else NullExperimentTracker()
    trackers.append(WandbExperimentTracker(run))
    return CompositeExperimentTracker(trackers)
