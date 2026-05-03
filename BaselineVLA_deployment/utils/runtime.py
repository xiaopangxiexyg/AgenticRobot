from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from omegaconf import OmegaConf

from utils.path_config import resolve_repo_path


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_run_name(experiment_name: str, configured_name: str | None) -> str:
    if configured_name:
        return configured_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{experiment_name}_mock_smoke_{timestamp}"


def make_run_dir(output_root: str | Path, experiment_name: str, run_name: str) -> Path:
    return ensure_dir(resolve_repo_path(output_root) / experiment_name / run_name)


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    file_path = Path(path).expanduser().resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def save_config(path: str | Path, config: Any) -> None:
    file_path = Path(path).expanduser().resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=config, f=str(file_path))


def tensor_to_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)
