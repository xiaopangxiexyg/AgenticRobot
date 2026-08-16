from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab_control.dev_config import load_dev_config
from lab_control.project_manifest import load_project_manifest
from lab_control.workspace import LabError


def test_framework_core_contains_no_project_or_model_specific_identifiers() -> None:
    repository = Path(__file__).resolve().parents[1]
    forbidden = (
        "agenticrobot",
        "rpent",
        "openpi",
        "dual_rm65",
        "/opt/agenticrobot",
        "/run/agenticrobot",
        "model_id",
        "supported_model",
        "{model_id}",
        "robot_motion",
        "agent-gateway-contract",
        "model contents",
    )
    offenders: dict[str, list[str]] = {}
    for path in sorted((repository / "lab_control").glob("*.py")):
        lowered = path.read_text(encoding="utf-8").casefold()
        matches = [token for token in forbidden if token.casefold() in lowered]
        if matches:
            offenders[path.name] = matches

    assert offenders == {}


def test_project_dev_config_owns_imports_and_test_catalog() -> None:
    repository = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (repository / "integration" / "dev.json").read_text(encoding="utf-8")
    )

    assert payload["schema"] == "lab.dev-config.v1"
    assert payload["python_major_minor"] == "3.10"
    assert payload["import_modules"] == [
        "lab_control",
        "robot_motion",
        "deployments",
    ]
    assert "agent-gateway-contract" in payload["test_targets"]


def test_dev_config_rejects_test_paths_outside_the_repository(tmp_path: Path) -> None:
    config_path = tmp_path / "dev.json"
    config_path.write_text(
        json.dumps(
            {
                "schema": "lab.dev-config.v1",
                "python_major_minor": "3.12",
                "import_modules": ["example_package"],
                "test_targets": {"unit": ["../outside.py"]},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(LabError, match="unsafe dev test path"):
        load_dev_config(config_path)


def test_project_manifest_supplies_arbitrary_repository_catalog_and_aliases(
    tmp_path: Path,
) -> None:
    control = (tmp_path / "Controller").resolve()
    component = (tmp_path / "ComponentX").resolve()
    control.mkdir()
    component.mkdir()
    manifest_path = tmp_path / "project.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "lab.project.v1",
                "project_id": "example-stack",
                "control_repository": "Controller",
                "default_development_host": "developer-a",
                "repositories": {
                    "Controller": {
                        "aliases": ["main"],
                        "default_path": "Controller",
                        "path_environment": "LAB_REPOSITORY_CONTROLLER",
                    },
                    "ComponentX": {
                        "aliases": ["component-x"],
                        "default_path": "ComponentX",
                        "path_environment": "LAB_REPOSITORY_COMPONENT_X",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = load_project_manifest(
        manifest_path,
        control_repository_root=control,
        environ={},
    )

    assert manifest.repositories == {
        "ComponentX": component,
        "Controller": control,
    }
    assert manifest.aliases["main"] == "Controller"
    assert manifest.aliases["component-x"] == "ComponentX"


def test_project_manifest_v2_resolves_a_repository_inside_the_control_checkout(
    tmp_path: Path,
) -> None:
    control = (tmp_path / "Controller").resolve()
    embedded = control / "components" / "ComponentY"
    sibling = (tmp_path / "ComponentX").resolve()
    control.mkdir()
    embedded.mkdir(parents=True)
    sibling.mkdir()
    manifest_path = tmp_path / "project-v2.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "lab.project.v2",
                "project_id": "example-stack",
                "control_repository": "Controller",
                "default_development_host": "developer-a",
                "repositories": {
                    "Controller": {
                        "aliases": ["main"],
                        "default_path": ".",
                        "path_base": "control_repository",
                        "path_environment": "LAB_REPOSITORY_CONTROLLER",
                    },
                    "ComponentX": {
                        "aliases": [],
                        "default_path": "ComponentX",
                        "path_base": "control_repository_parent",
                        "path_environment": "LAB_REPOSITORY_COMPONENT_X",
                    },
                    "ComponentY": {
                        "aliases": [],
                        "default_path": "components/ComponentY",
                        "path_base": "control_repository",
                        "path_environment": "LAB_REPOSITORY_COMPONENT_Y",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = load_project_manifest(
        manifest_path,
        control_repository_root=control,
        environ={},
    )

    assert manifest.repositories == {
        "ComponentX": sibling,
        "ComponentY": embedded,
        "Controller": control,
    }
