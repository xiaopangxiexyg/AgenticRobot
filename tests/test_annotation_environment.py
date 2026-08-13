from __future__ import annotations

from pathlib import Path

import pytest


pytestmark = [pytest.mark.unit, pytest.mark.contract]

REPO_ROOT = Path(__file__).parents[1]


def _annotation_extra_source() -> str:
    source = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    start = source.index("annotation = [")
    end = source.index("\nsim = [", start)
    return source[start:end]


def test_annotation_extra_excludes_device_sim_model_and_training_stacks() -> None:
    source = _annotation_extra_source()
    for requirement in (
        '"numpy==1.26.4"',
        '"pyarrow==24.0.0"',
    ):
        assert requirement in source
    for forbidden in (
        "opencv",
        "pyrealsense",
        "pyserial",
        "mujoco",
        "gymnasium",
        "sapien",
        "torch",
        "onnxruntime",
        "grpcio",
        "scipy",
        "osqp",
        "qpsolvers",
        "pin-pink",
    ):
        assert forbidden not in source


def test_annotation_installer_syncs_only_the_frozen_annotation_extra() -> None:
    installer = (
        REPO_ROOT / "scripts" / "install_annotation_env.sh"
    ).read_text(encoding="utf-8")
    assert "uv sync" in installer
    assert "--frozen" in installer
    assert "--extra annotation" in installer
    assert "--no-dev" in installer
    assert "--all-extras" not in installer
    assert "install_conda_env.sh" not in installer
    assert "DM_Tac" not in installer
