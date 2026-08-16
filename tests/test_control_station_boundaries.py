from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.support.repository_imports import (
    absolute_import_roots as _absolute_import_roots,
    python_files as _python_files,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STATION_ROOT = REPO_ROOT / "control_station"
pytestmark = [pytest.mark.unit, pytest.mark.contract]


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
    assert (
        _violations(
            core_paths,
            {
                "deployments",
                "hardware",
                "numpy",
                "scipy",
                "sim",
                "teleop_runtime",
            },
        )
        == {}
    )


def test_hitl_core_is_backend_model_topology_and_trace_neutral() -> None:
    hitl_paths = _python_files(STATION_ROOT / "hitl")
    assert _violations(
        hitl_paths,
        {
            "deployments",
            "hardware",
            "sim",
            "teleop_runtime",
        },
    ) == {}

    source = "\n".join(
        path.read_text(encoding="utf-8") for path in hitl_paths
    ).lower()
    for concrete_name in (
        "left_arm",
        "right_arm",
        "sim_hil",
        "mujoco",
        "openpi",
        "rm65",
        "vive",
    ):
        assert concrete_name not in source
    assert "dataset_contracts.control_trace" not in source


def test_real_teleop_and_umi_ui_packages_do_not_import_entity_data_planes() -> None:
    ui_paths = (
        *_python_files(STATION_ROOT / "real_teleop"),
        *_python_files(STATION_ROOT / "umi"),
        STATION_ROOT / "web/calibration_web_ui.py",
        STATION_ROOT / "web/startup_progress.py",
    )

    assert (
        _violations(
            tuple(ui_paths),
            {
                "collection_runtime",
                "dataset_interface",
                "hardware",
                "robot_motion",
                "sim",
                "teleop_runtime",
            },
        )
        == {}
    )


def test_station_dataset_projection_debt_is_explicit_and_cannot_spread() -> None:
    assert _violations(
        _python_files(STATION_ROOT),
        {"dataset_interface"},
    ) == {
        "control_station/real_teleop_entrypoint.py": ["dataset_interface"],
        "control_station/composition/mujoco_pose_composer.py": ["dataset_interface"],
        "control_station/composition/mujoco_sim_hil.py": ["dataset_interface"],
        "control_station/control_trace.py": ["dataset_interface"],
        # Concrete post-capture application binding. Presentation stays in
        # Station; discovery/resolution/indexing remain lower-owned.
        "control_station/refinement_annotation/application.py": ["dataset_interface"],
        "control_station/refinement_annotation/cli.py": ["dataset_interface"],
    }


def test_station_model_specific_bindings_are_explicit_and_cannot_spread() -> None:
    assert _violations(
        _python_files(STATION_ROOT),
        {"deployments"},
    ) == {
        "control_station/preflight/openpi_rtc.py": ["deployments"],
        # Concrete Sim-HIL application binding. Model transport and artifact
        # semantics remain deployments-owned; Station only connects the
        # provider to the selected simulated observation/target profile.
        "control_station/sim_hil_profile/openpi_policy_source.py": ["deployments"],
    }
    assert not (STATION_ROOT / "composition/openpi_rtc_source_preflight.py").exists()


def test_station_sim_dependencies_are_confined_to_concrete_profiles() -> None:
    assert _violations(
        _python_files(STATION_ROOT),
        {"sim"},
    ) == {
        "control_station/composition/mujoco_allocation_control_cycle.py": ["sim"],
        "control_station/composition/mujoco_allocation_publication.py": ["sim"],
        "control_station/composition/mujoco_allocation_shadow.py": ["sim"],
        "control_station/composition/mujoco_policy_human_allocation_binding.py": [
            "sim"
        ],
        "control_station/composition/mujoco_human.py": ["sim"],
        "control_station/composition/mujoco_human_replay.py": ["sim"],
        "control_station/composition/mujoco_human_station.py": ["sim"],
        "control_station/composition/mujoco_policy.py": ["sim"],
        "control_station/composition/mujoco_pose_composer.py": ["sim"],
        "control_station/composition/mujoco_replay.py": ["sim"],
        "control_station/composition/mujoco_rtc_policy.py": ["sim"],
        "control_station/composition/mujoco_sim_hil.py": ["sim"],
        "control_station/sim_hil.py": ["sim"],
        "control_station/real_teleop_entrypoint.py": ["sim"],
        "control_station/rm65_calibration_debug_application.py": ["sim"],
        # Concrete Agent application bindings remain Station leaves; the
        # simulator, execution, safety, and dispatch owners stay below them.
        "control_station/sim_hil_profile/agent/gateway_cli.py": ["sim"],
        "control_station/sim_hil_profile/agent/motion_application.py": ["sim"],
        "control_station/sim_hil_profile/agent/motion_factory.py": ["sim"],
        "control_station/sim_hil_profile/agent/observation.py": ["sim"],
        "control_station/sim_hil_profile/agent/policy_application.py": ["sim"],
        "control_station/sim_hil_profile/agent/policy_factory.py": ["sim"],
        "control_station/sim_hil_profile/tasks/cube_lift/deployment.py": ["sim"],
        "control_station/sim_hil_profile/factory.py": ["sim"],
        "control_station/sim_hil_profile/openpi_policy_source.py": ["sim"],
    }


def test_station_numerical_source_debt_is_explicit_and_cannot_spread() -> None:
    assert _violations(
        _python_files(STATION_ROOT),
        {"numpy", "scipy"},
    ) == {
        "control_station/real_teleop_entrypoint.py": ["numpy", "scipy"],
        "control_station/rm65_calibration_debug_application.py": [
            "numpy",
            "scipy",
        ],
        # Thin typed-payload materialization for the concrete Agent profile;
        # no reusable numerical control algorithm is owned here.
        "control_station/sim_hil_profile/agent/motion_skill.py": ["numpy"],
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
        "collection_runtime",
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
        violations.update(_violations(_python_files(package_root), {"control_station"}))
    # Historic command shims only. Device, motion, safety, and collection
    # ownership remain below the Station leaf, and no implementation or API
    # facade may join this allowlist.
    assert violations == {
        "teleop_runtime/collection/umi/app.py": ["control_station"],
        "teleop_runtime/debug/run_vive_rm65_calibration_debug.py": ["control_station"],
        "teleop_runtime/debug/run_vive_rm65_real_teleop.py": ["control_station"],
    }


def test_historic_command_shims_are_thin_and_hold_no_runtime_owner() -> None:
    shim_paths = (
        REPO_ROOT / "teleop_runtime/collection/umi/app.py",
        REPO_ROOT / "teleop_runtime/debug/run_vive_rm65_calibration_debug.py",
        REPO_ROOT / "teleop_runtime/debug/run_vive_rm65_real_teleop.py",
    )
    allowed_import_roots = {"__future__", "control_station", "sys"}
    for path in shim_paths:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        assert len(source.splitlines()) <= 20
        assert not any(
            isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            for node in ast.walk(tree)
        )
        assert _absolute_import_roots(path) <= allowed_import_roots
