from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
STATION_ROOT = REPO_ROOT / "control_station"
pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _python_files(root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    )


def _absolute_import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def _relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def _violations(
    paths: tuple[Path, ...],
    forbidden_roots: set[str],
) -> dict[str, list[str]]:
    return {
        _relative(path): sorted(_absolute_import_roots(path) & forbidden_roots)
        for path in paths
        if _absolute_import_roots(path) & forbidden_roots
    }


def test_station_application_core_does_not_import_concrete_data_planes() -> None:
    core_paths = tuple(
        STATION_ROOT / relative_path
        for relative_path in (
            "__init__.py",
            "adapters.py",
            "composition_session.py",
            "contracts.py",
            "events.py",
            "operation_coordinator.py",
            "publication_barrier.py",
            "runtime.py",
            "state_machine.py",
            "status.py",
            "web/__init__.py",
            "web/api.py",
            "web/server.py",
        )
    )
    assert _violations(
        core_paths,
        {
            "deployments",
            "hardware",
            "numpy",
            "scipy",
            "sim",
            "teleop_runtime",
        },
    ) == {}


def test_station_dataset_projection_debt_is_explicit_and_cannot_spread() -> None:
    assert _violations(
        _python_files(STATION_ROOT),
        {"dataset_interface"},
    ) == {
        "control_station/composition/mujoco_pose_composer.py": [
            "dataset_interface"
        ],
        "control_station/composition/mujoco_sim_hil.py": ["dataset_interface"],
        "control_station/control_trace.py": ["dataset_interface"],
    }


def test_station_model_specific_preflight_debt_is_explicit_and_cannot_spread() -> None:
    assert _violations(
        _python_files(STATION_ROOT),
        {"deployments"},
    ) == {
        "control_station/preflight/openpi_rtc.py": ["deployments"],
    }
    assert not (
        STATION_ROOT / "composition/openpi_rtc_source_preflight.py"
    ).exists()


def test_station_sim_dependencies_are_confined_to_concrete_profiles() -> None:
    assert _violations(
        _python_files(STATION_ROOT),
        {"sim"},
    ) == {
        "control_station/composition/mujoco_human.py": ["sim"],
        "control_station/composition/mujoco_human_replay.py": ["sim"],
        "control_station/composition/mujoco_human_station.py": ["sim"],
        "control_station/composition/mujoco_policy.py": ["sim"],
        "control_station/composition/mujoco_pose_composer.py": ["sim"],
        "control_station/composition/mujoco_replay.py": ["sim"],
        "control_station/composition/mujoco_rtc_policy.py": ["sim"],
        "control_station/composition/mujoco_sim_hil.py": ["sim"],
        "control_station/sim_hil.py": ["sim"],
    }


def test_station_numerical_source_debt_is_explicit_and_cannot_spread() -> None:
    assert _violations(
        _python_files(STATION_ROOT),
        {"numpy", "scipy"},
    ) == {
        "control_station/sources/synthetic_human.py": ["numpy"],
        "control_station/sources/teleop_pose.py": ["numpy"],
    }


def test_generic_passive_teleop_source_does_not_construct_vive_q6() -> None:
    source_text = (STATION_ROOT / "sources" / "teleop_pose.py").read_text(
        encoding="utf-8"
    )
    assert "passive_vive_q6" not in source_text
    assert "build_passive_vive_q6_session" not in source_text


def test_lower_packages_do_not_reverse_depend_on_control_station() -> None:
    violations: dict[str, list[str]] = {}
    for package_name in (
        "control_composition",
        "dataset_contracts",
        "dataset_interface",
        "deployments",
        "hardware",
        "robot_motion",
        "sim",
        "teleop_runtime",
    ):
        package_root = REPO_ROOT / package_name
        if not package_root.exists():
            continue
        violations.update(
            _violations(_python_files(package_root), {"control_station"})
        )
    assert violations == {}
