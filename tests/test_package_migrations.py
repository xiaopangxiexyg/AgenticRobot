from __future__ import annotations

from importlib.machinery import ModuleSpec, PathFinder
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _repository_package_spec(namespace: str, package_dir: Path) -> ModuleSpec | None:
    # Do not let an editable install from another linked worktree satisfy this
    # repository-layout assertion.
    return PathFinder.find_spec(namespace, [str(package_dir.parent)])


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
            "sim.rm65.real2sim",
            REPO_ROOT / "sim" / "rm65" / "real2sim",
            ".".join(("sim", "real2sim")),
            REPO_ROOT / "sim" / "real2sim",
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
    assert _repository_package_spec(canonical_namespace, canonical_dir) is not None
    assert not legacy_dir.exists()
    assert _repository_package_spec(legacy_namespace, legacy_dir) is None
