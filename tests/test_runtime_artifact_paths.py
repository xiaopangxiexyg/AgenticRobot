from __future__ import annotations

from pathlib import Path

import pytest

import project_paths
from runtime_artifact_paths import (
    DEFAULT_RUNTIME_ARTIFACTS_ROOT,
    LEGACY_ARTIFACT_LAYOUT_CLI_NAME,
    RUNTIME_ARTIFACT_LAYOUT_ID,
    RUNTIME_ARTIFACT_LAYOUT_CLI_NAME,
    RuntimeArtifactLayout,
    configured_runtime_artifact_layout,
    resolve_artifact_write_path,
    runtime_artifacts_root,
)


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ({}, LEGACY_ARTIFACT_LAYOUT_CLI_NAME),
        ({"runtime_artifact_layout": ""}, LEGACY_ARTIFACT_LAYOUT_CLI_NAME),
        (
            {"runtime_artifact_layout": RUNTIME_ARTIFACT_LAYOUT_CLI_NAME},
            RUNTIME_ARTIFACT_LAYOUT_CLI_NAME,
        ),
    ],
)
def test_configured_runtime_artifact_layout_is_validated_local_switch(
    monkeypatch,
    configured: dict[str, object],
    expected: str,
) -> None:
    monkeypatch.setattr(project_paths, "local_config", lambda: configured)

    assert configured_runtime_artifact_layout() == expected


@pytest.mark.parametrize(
    "configured",
    [
        {"runtime_artifact_layout": "future-layout"},
        {"runtime_artifact_layout": 1},
    ],
)
def test_configured_runtime_artifact_layout_rejects_invalid_values(
    monkeypatch,
    configured: dict[str, object],
) -> None:
    monkeypatch.setattr(project_paths, "local_config", lambda: configured)

    with pytest.raises((TypeError, ValueError), match="runtime_artifact_layout"):
        configured_runtime_artifact_layout()


def test_runtime_artifacts_default_is_additive(monkeypatch) -> None:
    monkeypatch.setattr(project_paths, "local_config", lambda: {})

    assert DEFAULT_RUNTIME_ARTIFACTS_ROOT == "runtime_artifacts"
    assert RUNTIME_ARTIFACT_LAYOUT_ID == "dexteleop.runtime_artifacts.v1"
    assert runtime_artifacts_root() == project_paths.repo_path("runtime_artifacts")
    assert project_paths.runtime_artifacts_root() == project_paths.repo_path(
        "runtime_artifacts"
    )
    assert project_paths.teleop_records_root() == project_paths.repo_path(
        "teleop_records"
    )


def test_runtime_artifacts_override_does_not_redirect_legacy_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    legacy = tmp_path / "legacy"
    monkeypatch.setattr(
        project_paths,
        "local_config",
        lambda: {
            "runtime_artifacts_root": str(canonical),
            "teleop_records_root": str(legacy),
        },
    )

    assert runtime_artifacts_root() == canonical.resolve()
    assert project_paths.runtime_artifacts_root() == canonical.resolve()
    assert RuntimeArtifactLayout.configured().root == canonical.resolve()
    assert project_paths.teleop_records_root() == legacy.resolve()


def test_layout_allocates_semantic_roots_without_creating_them(tmp_path: Path) -> None:
    root = tmp_path / "not-created"
    layout = RuntimeArtifactLayout(root=root)

    assert layout.runs_root == root / "runs"
    assert layout.evidence_root == root / "evidence"
    assert layout.state_root == root / "state"
    assert layout.imports_root == root / "imports"
    assert layout.derived_root == root / "derived"
    assert layout.migration_root == root / "migration"

    assert layout.run_family_dir(kind="real_teleop") == (
        root / "runs" / "real_teleop"
    )
    assert layout.run_partition_dir(
        kind="real_teleop",
        date_partition="2026-08-08",
    ) == root / "runs" / "real_teleop" / "2026-08-08"
    assert layout.run_dir(
        kind="real_teleop",
        date_partition="2026-08-08",
        run_id="run-001",
    ) == root / "runs" / "real_teleop" / "2026-08-08" / "run-001"
    assert layout.evidence_dir(kind="sim_hil") == root / "evidence" / "sim_hil"
    assert layout.state_dir(owner="vive", kind="calibration") == (
        root / "state" / "vive" / "calibration"
    )
    assert layout.import_dir(source_host="dex-remote", bundle_id="bundle-001") == (
        root / "imports" / "dex-remote" / "bundle-001"
    )
    assert layout.derived_dir(kind="ik_analysis", artifact_id="artifact-001") == (
        root / "derived" / "ik_analysis" / "artifact-001"
    )
    assert layout.migration_dir(migration_id="migration-001") == (
        root / "migration" / "migration-001"
    )
    assert not root.exists()


@pytest.mark.parametrize(
    "unsafe_component",
    ["", " ", ".", "..", "nested/name", "nested\\name", "/absolute", "bad\x00id"],
)
def test_layout_rejects_unsafe_dynamic_components(
    tmp_path: Path,
    unsafe_component: str,
) -> None:
    layout = RuntimeArtifactLayout(root=tmp_path / "artifacts")

    with pytest.raises(ValueError):
        layout.run_dir(
            kind="real_teleop",
            date_partition="2026-08-08",
            run_id=unsafe_component,
        )


def test_layout_rejects_non_string_dynamic_components(tmp_path: Path) -> None:
    layout = RuntimeArtifactLayout(root=tmp_path / "artifacts")

    with pytest.raises(TypeError):
        layout.evidence_dir(kind=1)  # type: ignore[arg-type]


def test_write_path_resolution_defaults_to_one_legacy_path(tmp_path: Path) -> None:
    resolution = resolve_artifact_write_path(
        explicit_path=None,
        artifact_layout=None,
        legacy_path=tmp_path / "legacy",
        canonical_path=tmp_path / "canonical",
    )

    assert resolution.path == tmp_path / "legacy"
    assert resolution.source == "legacy"
    assert resolution.layout_name == LEGACY_ARTIFACT_LAYOUT_CLI_NAME


def test_write_path_resolution_requires_explicit_canonical_opt_in(
    tmp_path: Path,
) -> None:
    resolution = resolve_artifact_write_path(
        explicit_path=None,
        artifact_layout=RUNTIME_ARTIFACT_LAYOUT_CLI_NAME,
        legacy_path=tmp_path / "legacy",
        canonical_path=tmp_path / "canonical",
    )

    assert resolution.path == tmp_path / "canonical"
    assert resolution.source == "canonical"
    assert not resolution.path.exists()


def test_write_path_resolution_explicit_path_wins_over_layout(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit"
    resolution = resolve_artifact_write_path(
        explicit_path=explicit,
        artifact_layout=RUNTIME_ARTIFACT_LAYOUT_CLI_NAME,
        legacy_path=tmp_path / "legacy",
        canonical_path=tmp_path / "canonical",
    )

    assert resolution.path == explicit
    assert resolution.source == "explicit"
    assert resolution.layout_name == "explicit"


def test_write_path_resolution_rejects_unknown_layout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported artifact layout"):
        resolve_artifact_write_path(
            explicit_path=None,
            artifact_layout="future-layout",
            legacy_path=tmp_path / "legacy",
            canonical_path=tmp_path / "canonical",
        )
