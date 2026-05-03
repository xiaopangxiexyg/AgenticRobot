from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from Training.progress import ProgressSettings, build_epoch_progress, format_metric_postfix
from utils.tracking import build_experiment_tracker
from utils.runtime import save_config, save_json, tensor_to_float


def resolve_device(device_name: str) -> torch.device:
    normalized = str(device_name).lower()
    if normalized.startswith("cuda") and torch.cuda.is_available():
        return torch.device(normalized)
    return torch.device("cpu")


def _clone_checkpoint_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_checkpoint_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_checkpoint_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_checkpoint_value(item) for item in value)
    return copy.deepcopy(value)


def _config_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _optional_config_str(value: Any) -> str | None:
    if value in (None, "null", ""):
        return None
    return str(value)


def _torch_load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint {path} must contain a dict payload.")
    return checkpoint


class SupervisedExperimentTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        input_adapter: Any,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        loss_weights: Dict[str, float],
        progress_settings: ProgressSettings,
    ) -> None:
        self.model = model.to(device)
        self.input_adapter = input_adapter
        self.optimizer = optimizer
        self.device = device
        self.loss_weights = {str(key): float(value) for key, value in loss_weights.items()}
        self.progress_settings = progress_settings

    def fit(self, train_loader: Any, val_loader: Any, epochs: int, run_dir: Path, config: Any, data_summary: Dict[str, Any]) -> Dict[str, Any]:
        history: List[Dict[str, Any]] = []
        best_total = float("inf")
        best_selection_split = "val" if val_loader is not None else "train"
        best_state: Dict[str, Any] | None = None
        periodic_checkpoint_paths: List[str] = []
        resume_info: Dict[str, Any] | None = None
        start_epoch = 0
        total_epochs = int(epochs)
        periodic_interval = max(0, int(OmegaConf.select(config, "train.checkpoint_every_n_epochs", default=0) or 0))
        save_periodic = _config_bool(
            OmegaConf.select(config, "train.save_periodic_checkpoints", default=periodic_interval > 0),
            default=periodic_interval > 0,
        )
        periodic_dir_name = str(OmegaConf.select(config, "train.periodic_checkpoint_dir", default="checkpoints"))
        periodic_dir = Path(periodic_dir_name).expanduser()
        if not periodic_dir.is_absolute():
            periodic_dir = run_dir / periodic_dir

        resume_from = _optional_config_str(OmegaConf.select(config, "train.resume_from_checkpoint", default=None))
        if resume_from is not None:
            resume_path = Path(resume_from).expanduser()
            if resume_path.is_dir():
                resume_path = resume_path / "last.pt"
            if not resume_path.is_absolute():
                resume_path = resume_path.resolve()
            if not resume_path.exists():
                raise FileNotFoundError(f"resume_from_checkpoint does not exist: {resume_path}")

            checkpoint = _torch_load_checkpoint(resume_path, self.device)
            if "model" not in checkpoint:
                raise KeyError(f"Checkpoint {resume_path} does not contain a 'model' state_dict.")
            strict = _config_bool(OmegaConf.select(config, "train.resume_strict", default=True), default=True)
            self.model.load_state_dict(checkpoint["model"], strict=strict)
            load_optimizer = _config_bool(
                OmegaConf.select(config, "train.resume_load_optimizer", default=True),
                default=True,
            )
            optimizer_loaded = False
            if load_optimizer and "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                optimizer_loaded = True

            checkpoint_summary = checkpoint.get("summary", {})
            if isinstance(checkpoint_summary, dict):
                previous_history = checkpoint_summary.get("history", [])
                if isinstance(previous_history, list):
                    history = copy.deepcopy(previous_history)
                previous_periodic = checkpoint_summary.get("periodic_checkpoints", [])
                if isinstance(previous_periodic, list):
                    periodic_checkpoint_paths = [str(path) for path in previous_periodic]
                if checkpoint_summary.get("best_total_loss") is not None:
                    best_total = float(checkpoint_summary["best_total_loss"])
                if checkpoint_summary.get("best_selection_split") is not None:
                    best_selection_split = str(checkpoint_summary["best_selection_split"])
            if checkpoint.get("best_total_loss_so_far") is not None:
                best_total = float(checkpoint["best_total_loss_so_far"])
            elif checkpoint.get("best_total_loss") is not None:
                best_total = float(checkpoint["best_total_loss"])
            if checkpoint.get("best_selection_split") is not None:
                best_selection_split = str(checkpoint["best_selection_split"])

            start_epoch = int(checkpoint.get("epoch", len(history)) or 0)
            if _config_bool(OmegaConf.select(config, "train.resume_epochs_are_additional", default=False), default=False):
                total_epochs = start_epoch + int(epochs)
            resume_info = {
                "checkpoint": str(resume_path),
                "start_epoch": start_epoch,
                "target_total_epochs": total_epochs,
                "optimizer_loaded": optimizer_loaded,
                "strict": strict,
            }
            message = (
                f"resumed checkpoint {resume_path} at epoch {start_epoch}; "
                f"target_total_epochs={total_epochs}; optimizer_loaded={optimizer_loaded}"
            )
            if self.progress_settings.enabled:
                tqdm.write(message)
            else:
                print(message)

        save_config(run_dir / "resolved_config.yaml", config)
        save_json(run_dir / "data_summary.json", data_summary)
        tracker = build_experiment_tracker(config=config, run_dir=run_dir, data_summary=data_summary)

        for epoch_index in range(start_epoch, total_epochs):
            train_metrics = self.run_epoch(
                train_loader,
                training=True,
                epoch_index=epoch_index,
                total_epochs=total_epochs,
            )
            epoch_metrics = {
                "epoch": epoch_index + 1,
                "train": train_metrics,
            }
            if val_loader is not None:
                val_metrics = self.run_epoch(
                    val_loader,
                    training=False,
                    epoch_index=epoch_index,
                    total_epochs=total_epochs,
                )
                epoch_metrics["val"] = val_metrics
                selection_metrics = val_metrics
            else:
                selection_metrics = train_metrics
            history.append(epoch_metrics)
            tracker.log_epoch(epoch_metrics)
            if selection_metrics["total_loss"] < best_total:
                best_total = selection_metrics["total_loss"]
                best_state = {
                    "model": _clone_checkpoint_value(self.model.state_dict()),
                    "optimizer": _clone_checkpoint_value(self.optimizer.state_dict()),
                    "epoch": epoch_index + 1,
                    "best_total_loss": best_total,
                    "best_selection_split": best_selection_split,
                }
                if val_loader is not None:
                    best_state["best_val_total_loss"] = best_total

            if save_periodic and periodic_interval > 0 and (epoch_index + 1) % periodic_interval == 0:
                checkpoint_path = periodic_dir / f"epoch_{epoch_index + 1:04d}.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "config": OmegaConf.to_container(config, resolve=True),
                        "data_summary": data_summary,
                        "epoch": epoch_index + 1,
                        "epoch_metrics": epoch_metrics,
                        "best_total_loss_so_far": best_total,
                        "best_selection_split": best_selection_split,
                    },
                    checkpoint_path,
                )
                try:
                    checkpoint_label = str(checkpoint_path.relative_to(run_dir))
                except ValueError:
                    checkpoint_label = str(checkpoint_path)
                periodic_checkpoint_paths.append(checkpoint_label)

            if val_loader is not None:
                epoch_summary = (
                    f"Epoch {epoch_index + 1}/{total_epochs} | "
                    f"train total={train_metrics['total_loss']:.6f} | "
                    f"val total={epoch_metrics['val']['total_loss']:.6f} | "
                    f"best val={best_total:.6f}"
                )
            else:
                epoch_summary = (
                    f"Epoch {epoch_index + 1}/{total_epochs} | "
                    f"train total={train_metrics['total_loss']:.6f} | "
                    f"best train={best_total:.6f}"
                )
            if self.progress_settings.enabled:
                tqdm.write(epoch_summary)
            else:
                print(epoch_summary)

        summary = {
            "experiment": str(config.experiment.name),
            "device": str(self.device),
            "epochs": total_epochs,
            "start_epoch": start_epoch,
            "resume": resume_info,
            "tracker": tracker.name,
            "best_total_loss": best_total,
            "best_selection_split": best_selection_split,
            "best_val_total_loss": best_total if val_loader is not None else None,
            "periodic_checkpoint_interval": periodic_interval if save_periodic else 0,
            "periodic_checkpoints": periodic_checkpoint_paths,
            "history": history,
            "data_summary": data_summary,
        }
        save_json(run_dir / "metrics.json", summary)

        if config.train.save_last_checkpoint:
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "config": OmegaConf.to_container(config, resolve=True),
                    "summary": summary,
                },
                run_dir / "last.pt",
            )
        if config.train.save_best_checkpoint and best_state is not None:
            torch.save(
                {
                    **best_state,
                    "config": OmegaConf.to_container(config, resolve=True),
                    "summary": summary,
                },
                run_dir / "best.pt",
            )
        tracker.finish(summary)
        return summary

    def run_epoch(self, loader: Any, training: bool, epoch_index: int, total_epochs: int) -> Dict[str, float]:
        self.model.train(training)
        accumulated = {"total_loss": 0.0}
        counts = 0
        split_name = "train" if training else "val"

        with build_epoch_progress(
            loader,
            split_name=split_name,
            epoch_index=epoch_index,
            total_epochs=total_epochs,
            settings=self.progress_settings,
        ) as progress_bar:
            for step_index, batch in enumerate(loader, start=1):
                prepared = self.input_adapter.prepare_batch(batch, self.device)
                with torch.set_grad_enabled(training):
                    _, total_loss, component_losses = self._compute_batch_outputs_and_losses(prepared)
                    if training:
                        self.optimizer.zero_grad()
                        total_loss.backward()
                        self.optimizer.step()

                counts += 1
                accumulated["total_loss"] += tensor_to_float(total_loss)
                for key, value in component_losses.items():
                    metric_name = f"{key}_loss"
                    accumulated.setdefault(metric_name, 0.0)
                    accumulated[metric_name] += tensor_to_float(value)

                progress_bar.update(1)
                if (
                    step_index == 1
                    or step_index % self.progress_settings.postfix_update_interval == 0
                    or (progress_bar.total is not None and step_index == int(progress_bar.total))
                ):
                    running_metrics = {key: value / counts for key, value in accumulated.items()}
                    progress_bar.set_postfix(format_metric_postfix(running_metrics))

        if counts == 0:
            return {"total_loss": 0.0}
        return {key: value / counts for key, value in accumulated.items()}

    def _compute_batch_outputs_and_losses(self, prepared: Any) -> tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        custom_loss_fn = getattr(self.model, "compute_custom_loss", None)
        if callable(custom_loss_fn):
            outputs, total_loss, component_losses = custom_loss_fn(
                model_inputs=prepared.model_inputs,
                targets=prepared.targets,
            )
            return outputs, total_loss, component_losses

        outputs = self.model(**prepared.model_inputs)
        total_loss, component_losses = self.compute_loss(outputs, prepared.targets)
        return outputs, total_loss, component_losses

    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        total_loss = torch.tensor(0.0, device=self.device)
        component_losses: Dict[str, torch.Tensor] = {}
        for key, target in targets.items():
            weight = float(self.loss_weights.get(key, 0.0))
            if weight <= 0.0:
                continue
            if key not in outputs:
                raise KeyError(f"Model output missing required key: {key}")
            component = F.mse_loss(outputs[key], target)
            component_losses[key] = component
            total_loss = total_loss + weight * component
        return total_loss, component_losses
