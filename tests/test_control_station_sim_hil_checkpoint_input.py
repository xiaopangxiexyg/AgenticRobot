from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from control_station.sim_hil_profile.config import (
    DEFAULT_CHECKPOINT,
    DEFAULT_TASK_CONFIG,
    LOCAL_DEFAULT_CHECKPOINT,
    LOCAL_DEFAULT_TASK_CONFIG,
    SIM_HIL_CHECKPOINT_ENV,
    SIM_HIL_TASK_CONFIG_ENV,
    MujocoSimHILApplicationConfig,
    resolve_default_checkpoint_path,
    resolve_default_task_config_path,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_checkpoint_preserves_historical_local_path() -> None:
    assert resolve_default_checkpoint_path({}) == LOCAL_DEFAULT_CHECKPOINT
    assert resolve_default_task_config_path({}) == LOCAL_DEFAULT_TASK_CONFIG
    config = MujocoSimHILApplicationConfig()
    assert config.checkpoint_path == DEFAULT_CHECKPOINT
    assert config.task_config_path == DEFAULT_TASK_CONFIG


def test_default_checkpoint_accepts_one_absolute_host_input(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")

    resolved = resolve_default_checkpoint_path(
        {SIM_HIL_CHECKPOINT_ENV: str(checkpoint)}
    )

    assert resolved == checkpoint.resolve()


def test_default_task_config_accepts_one_absolute_host_input(tmp_path: Path) -> None:
    task_config = tmp_path / "task.json"
    task_config.write_text("{}\n", encoding="utf-8")

    resolved = resolve_default_task_config_path(
        {SIM_HIL_TASK_CONFIG_ENV: str(task_config)}
    )

    assert resolved == task_config.resolve()


@pytest.mark.parametrize(
    ("resolver", "environment_name", "configured"),
    (
        (resolve_default_checkpoint_path, SIM_HIL_CHECKPOINT_ENV, "checkpoint.pt"),
        (
            resolve_default_task_config_path,
            SIM_HIL_TASK_CONFIG_ENV,
            "relative/task.json",
        ),
    ),
)
def test_sim_hil_inputs_reject_relative_host_paths(
    resolver,
    environment_name: str,
    configured: str,
) -> None:
    with pytest.raises(ValueError, match="must be an absolute path"):
        resolver({environment_name: configured})


def test_fresh_process_binds_dataclass_default_to_host_input(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    task_config = tmp_path / "task.json"
    task_config.write_text("{}\n", encoding="utf-8")
    environment = dict(os.environ)
    environment[SIM_HIL_CHECKPOINT_ENV] = str(checkpoint)
    environment[SIM_HIL_TASK_CONFIG_ENV] = str(task_config)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from control_station.sim_hil_profile.config import "
                "MujocoSimHILApplicationConfig; "
                "c=MujocoSimHILApplicationConfig(); "
                "print(c.checkpoint_path); print(c.task_config_path)"
            ),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        str(checkpoint.resolve()),
        str(task_config.resolve()),
    ]
