from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from tqdm.auto import tqdm


@dataclass
class ProgressSettings:
    enabled: bool = True
    leave: bool = False
    mininterval: float = 0.2
    postfix_update_interval: int = 1


def resolve_progress_settings(train_config: Any) -> ProgressSettings:
    progress_config = getattr(train_config, "progress", None)
    legacy_interval = int(getattr(train_config, "log_every_n_steps", 1))
    if progress_config is None:
        return ProgressSettings(
            enabled=True,
            leave=False,
            mininterval=0.2,
            postfix_update_interval=max(1, legacy_interval),
        )
    return ProgressSettings(
        enabled=bool(getattr(progress_config, "enabled", True)),
        leave=bool(getattr(progress_config, "leave", False)),
        mininterval=float(getattr(progress_config, "mininterval", 0.2)),
        postfix_update_interval=max(
            1,
            int(getattr(progress_config, "postfix_update_interval", legacy_interval)),
        ),
    )


def build_epoch_progress(
    loader: Any,
    *,
    split_name: str,
    epoch_index: int,
    total_epochs: int,
    settings: ProgressSettings,
):
    total = len(loader) if hasattr(loader, "__len__") else None
    description = f"Epoch {epoch_index + 1}/{total_epochs} [{split_name}]"
    return tqdm(
        total=total,
        desc=description,
        leave=settings.leave,
        disable=not settings.enabled,
        dynamic_ncols=True,
        mininterval=settings.mininterval,
    )


def format_metric_postfix(metrics: Dict[str, float]) -> Dict[str, str]:
    ordered_keys = ["total_loss"] + sorted(key for key in metrics.keys() if key != "total_loss")
    return {
        key.replace("_loss", ""): f"{metrics[key]:.4f}"
        for key in ordered_keys
        if key in metrics
    }
