from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
LOCAL_CONFIG_FILENAME = "local_paths.json"
LOCAL_CONFIG_ENV = "DEX_INSPIRE_LOCAL_CONFIG"


def repo_root() -> Path:
    return REPO_ROOT


def repo_path(*parts: str | os.PathLike[str]) -> Path:
    return (REPO_ROOT.joinpath(*parts)).resolve()


def local_config_path() -> Path:
    configured = os.environ.get(LOCAL_CONFIG_ENV, "").strip()
    if configured:
        return Path(os.path.expandvars(configured)).expanduser().resolve()
    return REPO_ROOT / LOCAL_CONFIG_FILENAME


@lru_cache(maxsize=1)
def local_config() -> dict[str, Any]:
    path = local_config_path()
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def reload_local_config() -> dict[str, Any]:
    local_config.cache_clear()
    return local_config()


def resolve_path(path: str | os.PathLike[str], *, base: Path | None = None) -> Path:
    text = os.path.expandvars(str(path))
    text = text.replace("${repo_root}", str(REPO_ROOT)).replace("{repo_root}", str(REPO_ROOT))
    resolved = Path(text).expanduser()
    if not resolved.is_absolute():
        resolved = (base or REPO_ROOT) / resolved
    return resolved.resolve()


def local_path(key: str, default: str | os.PathLike[str]) -> Path:
    value = local_config().get(key, default)
    if value in (None, ""):
        value = default
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"{local_config_path()} key {key!r} must be a path string.")
    return resolve_path(value)


def optional_local_path(key: str) -> Path | None:
    value = local_config().get(key)
    if value in (None, ""):
        return None
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"{local_config_path()} key {key!r} must be a path string.")
    return resolve_path(value)


def default_dataset_root() -> Path:
    return local_path("dataset_root", "datasets")


def teleop_records_root() -> Path:
    return local_path("teleop_records_root", "teleop_records")


def teleop_runtime_configs_root() -> Path:
    return repo_path("teleop_runtime", "configs")


def hardware_interface_configs_root() -> Path:
    return teleop_runtime_configs_root() / "hardware"


def calibration_profiles_dir() -> Path:
    return local_path("calibration_profiles_dir", teleop_records_root() / "calibration_profiles")


def hand_zero_pose_calibrations_dir() -> Path:
    return local_path(
        "hand_zero_pose_calibrations_dir",
        teleop_records_root() / "hand_zero_pose_calibrations",
    )


def initial_alignment_states_dir() -> Path:
    return local_path(
        "initial_alignment_states_dir",
        teleop_records_root() / "initial_alignment_states",
    )


def real_robot_dataset_root() -> Path:
    return local_path("real_robot_dataset_root", "datasets")
