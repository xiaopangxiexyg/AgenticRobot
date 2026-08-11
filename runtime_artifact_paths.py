"""Side-effect-free path allocation for canonical runtime artifacts.

The canonical layout is additive: importing or constructing it never creates
directories and never changes legacy readers or writers.  Callers must opt in
by receiving an explicit path from this module.

Formal datasets and package-owned configuration are intentionally outside this
layout.  This module also owns no writer, episode, safety, or process lifecycle.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import project_paths


RUNTIME_ARTIFACT_LAYOUT_ID = "dexteleop.runtime_artifacts.v1"
RUNTIME_ARTIFACT_LAYOUT_CLI_NAME = "runtime-artifacts-v1"
LEGACY_ARTIFACT_LAYOUT_CLI_NAME = "legacy"
DEFAULT_RUNTIME_ARTIFACTS_ROOT = "runtime_artifacts"

ArtifactWriteSource = Literal["explicit", "legacy", "canonical"]


@dataclass(frozen=True)
class ArtifactWriteResolution:
    path: Path
    source: ArtifactWriteSource
    layout_name: str


def configured_runtime_artifact_layout() -> str:
    """Return the validated local default used by migrated CLI entrypoints.

    The repository default remains compatibility-first.  Operators may switch
    all participating CLIs together through the ignored ``local_paths.json``
    instead of landing a scattered set of parser-default edits.
    """

    value = project_paths.local_config().get(
        "runtime_artifact_layout",
        LEGACY_ARTIFACT_LAYOUT_CLI_NAME,
    )
    if value in (None, ""):
        return LEGACY_ARTIFACT_LAYOUT_CLI_NAME
    if not isinstance(value, str):
        raise TypeError(
            "local path key 'runtime_artifact_layout' must be a string"
        )
    selected = value.strip()
    if selected not in {
        LEGACY_ARTIFACT_LAYOUT_CLI_NAME,
        RUNTIME_ARTIFACT_LAYOUT_CLI_NAME,
    }:
        raise ValueError(
            "local path key 'runtime_artifact_layout' must be one of "
            f"{LEGACY_ARTIFACT_LAYOUT_CLI_NAME!r}, "
            f"{RUNTIME_ARTIFACT_LAYOUT_CLI_NAME!r}; got {selected!r}"
        )
    return selected


def runtime_artifacts_root() -> Path:
    """Return the opt-in canonical root without creating it."""

    return project_paths.runtime_artifacts_root()


def resolve_artifact_write_path(
    *,
    explicit_path: str | os.PathLike[str] | None,
    artifact_layout: str | None,
    legacy_path: str | os.PathLike[str],
    canonical_path: str | os.PathLike[str],
) -> ArtifactWriteResolution:
    """Select exactly one write path without creating or writing to it."""

    if explicit_path is not None:
        return ArtifactWriteResolution(
            path=Path(explicit_path).expanduser(),
            source="explicit",
            layout_name="explicit",
        )

    selected_layout = str(
        artifact_layout or LEGACY_ARTIFACT_LAYOUT_CLI_NAME
    ).strip()
    if selected_layout == LEGACY_ARTIFACT_LAYOUT_CLI_NAME:
        return ArtifactWriteResolution(
            path=Path(legacy_path).expanduser(),
            source="legacy",
            layout_name=LEGACY_ARTIFACT_LAYOUT_CLI_NAME,
        )
    if selected_layout == RUNTIME_ARTIFACT_LAYOUT_CLI_NAME:
        return ArtifactWriteResolution(
            path=Path(canonical_path).expanduser(),
            source="canonical",
            layout_name=RUNTIME_ARTIFACT_LAYOUT_CLI_NAME,
        )
    raise ValueError(f"unsupported artifact layout: {selected_layout!r}")


def _single_path_component(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string path component.")
    if not value or value != value.strip():
        raise ValueError(f"{field} must be a non-empty trimmed path component.")
    if value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
        raise ValueError(f"{field} must be exactly one safe path component.")
    return value


@dataclass(frozen=True)
class RuntimeArtifactLayout:
    """Immutable allocator for the first canonical runtime-artifact layout."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", project_paths.resolve_path(self.root))

    @classmethod
    def configured(cls) -> "RuntimeArtifactLayout":
        return cls(root=runtime_artifacts_root())

    @property
    def runs_root(self) -> Path:
        return self.root / "runs"

    @property
    def evidence_root(self) -> Path:
        return self.root / "evidence"

    @property
    def state_root(self) -> Path:
        return self.root / "state"

    @property
    def imports_root(self) -> Path:
        return self.root / "imports"

    @property
    def derived_root(self) -> Path:
        return self.root / "derived"

    @property
    def migration_root(self) -> Path:
        return self.root / "migration"

    def run_family_dir(self, *, kind: str) -> Path:
        return self.runs_root / _single_path_component(kind, field="kind")

    def run_partition_dir(self, *, kind: str, date_partition: str) -> Path:
        return self.run_family_dir(kind=kind) / _single_path_component(
            date_partition,
            field="date_partition",
        )

    def run_dir(self, *, kind: str, date_partition: str, run_id: str) -> Path:
        return self.run_partition_dir(
            kind=kind,
            date_partition=date_partition,
        ) / _single_path_component(
            run_id,
            field="run_id",
        )

    def evidence_dir(self, *, kind: str) -> Path:
        return self.evidence_root / _single_path_component(kind, field="kind")

    def state_dir(self, *, owner: str, kind: str) -> Path:
        return (
            self.state_root
            / _single_path_component(owner, field="owner")
            / _single_path_component(kind, field="kind")
        )

    def import_dir(self, *, source_host: str, bundle_id: str) -> Path:
        return (
            self.imports_root
            / _single_path_component(source_host, field="source_host")
            / _single_path_component(bundle_id, field="bundle_id")
        )

    def derived_dir(self, *, kind: str, artifact_id: str) -> Path:
        return (
            self.derived_root
            / _single_path_component(kind, field="kind")
            / _single_path_component(artifact_id, field="artifact_id")
        )

    def migration_dir(self, *, migration_id: str) -> Path:
        return self.migration_root / _single_path_component(
            migration_id,
            field="migration_id",
        )
