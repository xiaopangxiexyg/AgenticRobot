from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from control_station.sim_hil_profile.config import (
    DEFAULT_CHECKPOINT,
    LOCAL_DEFAULT_CHECKPOINT,
    SIM_HIL_CHECKPOINT_ENV,
    MujocoSimHILApplicationConfig,
    resolve_default_checkpoint_path,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_checkpoint_preserves_historical_local_path() -> None:
    assert resolve_default_checkpoint_path({}) == LOCAL_DEFAULT_CHECKPOINT
    assert MujocoSimHILApplicationConfig().checkpoint_path == DEFAULT_CHECKPOINT


def test_default_checkpoint_accepts_one_absolute_host_input(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")

    resolved = resolve_default_checkpoint_path(
        {SIM_HIL_CHECKPOINT_ENV: str(checkpoint)}
    )

    assert resolved == checkpoint.resolve()


@pytest.mark.parametrize("configured", ("checkpoint.pt", "relative/checkpoint.pt"))
def test_default_checkpoint_rejects_relative_host_input(configured: str) -> None:
    with pytest.raises(ValueError, match="must be an absolute path"):
        resolve_default_checkpoint_path({SIM_HIL_CHECKPOINT_ENV: configured})


def test_fresh_process_binds_dataclass_default_to_host_input(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    environment = dict(os.environ)
    environment[SIM_HIL_CHECKPOINT_ENV] = str(checkpoint)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from control_station.sim_hil_profile.config import "
                "MujocoSimHILApplicationConfig; "
                "print(MujocoSimHILApplicationConfig().checkpoint_path)"
            ),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == str(checkpoint.resolve())
