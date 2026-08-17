from __future__ import annotations

from importlib.machinery import ModuleSpec, PathFinder
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _repository_package_spec(namespace: str, package_dir: Path) -> ModuleSpec | None:
    # Do not let an editable install from another linked worktree satisfy this
    # repository-layout assertion.
    package_name = namespace.rsplit(".", 1)[-1]
    return PathFinder.find_spec(package_name, [str(package_dir.parent)])


def _python_source_files(package_dir: Path) -> tuple[Path, ...]:
    if not package_dir.is_dir():
        return ()
    return tuple(sorted(path for path in package_dir.rglob("*.py") if path.is_file()))


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
    canonical_spec = _repository_package_spec(canonical_namespace, canonical_dir)
    assert canonical_spec is not None
    assert canonical_spec.loader is not None
    assert _python_source_files(legacy_dir) == ()
    legacy_spec = _repository_package_spec(legacy_namespace, legacy_dir)
    assert legacy_spec is None or legacy_spec.loader is None
