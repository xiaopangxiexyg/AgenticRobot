from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("canonical_namespace", "canonical_dir", "legacy_namespace", "legacy_dir"),
    (
        (
            "utils.dataset_postprocess",
            REPO_ROOT / "utils" / "dataset_postprocess",
            "dataset" + "_postprocess",
            REPO_ROOT / ("dataset" + "_postprocess"),
        ),
        (
            "sim.mujoco",
            REPO_ROOT / "sim" / "mujoco",
            ".".join(("sim", "mujoco_" + "real2sim")),
            REPO_ROOT / "sim" / ("mujoco_" + "real2sim"),
        ),
        (
            "utils.wrist_cam_fixed_mask",
            REPO_ROOT / "utils" / "wrist_cam_fixed_mask",
            ".".join(("utils", "wrist_cam_" + "fix_mask")),
            REPO_ROOT / "utils" / ("wrist_cam_" + "fix_mask"),
        ),
    ),
)
def test_only_canonical_package_remains(
    canonical_namespace: str,
    canonical_dir: Path,
    legacy_namespace: str,
    legacy_dir: Path,
) -> None:
    assert canonical_dir.is_dir()
    assert importlib.util.find_spec(canonical_namespace) is not None
    assert not legacy_dir.exists()
    assert importlib.util.find_spec(legacy_namespace) is None
