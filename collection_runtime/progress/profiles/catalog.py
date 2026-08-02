"""Read-only catalog for deployable task progress profile bundles.

Pure profile schemas remain owned by :mod:`dataset_contracts`.  This module
only manages editable profile sources.  Source-file hashes are derived while
loading; immutable dataset artifacts retain the strict hash validation owned
by :mod:`dataset_contracts`.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
import json
from pathlib import Path
from typing import Any, Mapping

from dataset_contracts.progress_profile import TaskProgressProfileV1
from dataset_contracts.progress_refinement import (
    TaskRefinementProfileV1,
    validate_task_refinement_profile,
)
from dataset_contracts.progress_stages import (
    TaskStageProfileV1,
    validate_task_stage_profile,
)


_PROGRESS_SUFFIX = ".progress.json"
_REFINEMENT_SUFFIX = ".refinement.json"
_STAGES_SUFFIX = ".stages.json"
_TOPOLOGY_PACKAGES = {
    "LINEAR": "collection_runtime.progress.profiles.linear",
    "ROOTED_TREE": "collection_runtime.progress.profiles.tree",
}


class ProgressProfileCatalogError(ValueError):
    """Raised when a managed profile asset is ambiguous or inconsistent."""


@dataclass(frozen=True)
class ManagedTaskProfile:
    """One task profile plus optional legacy and ordered-stage semantics."""

    progress_profile: TaskProgressProfileV1
    refinement_profile: TaskRefinementProfileV1 | None
    stage_profile: TaskStageProfileV1 | None
    topology_kind: str
    progress_resource: str
    refinement_resource: str | None = None
    stage_resource: str | None = None

    @property
    def profile_id(self) -> str:
        return self.progress_profile.profile_id


def _mapping_from_text(text: str, *, source: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProgressProfileCatalogError(
            f"Invalid JSON in progress profile asset {source}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ProgressProfileCatalogError(
            f"Progress profile asset {source} must contain a JSON object."
        )
    return payload


def _editable_source_payload(
    text: str,
    *,
    source: str,
    task_progress_profile_hash: str | None = None,
) -> dict[str, Any]:
    """Normalize an editable config source before contract construction.

    Profile JSON files are operator-edited configuration, not immutable
    receipts.  Their canonical hashes are therefore outputs of loading rather
    than fields that an operator must update after every semantic edit.
    Companion stage/refinement sources are rebound to the progress profile
    loaded in the same bundle.  The constructed contract objects still carry
    exact content hashes for dataset admission and persistence.
    """

    payload = dict(_mapping_from_text(text, source=source))
    payload.pop("semantic_content_hash", None)
    if task_progress_profile_hash is not None:
        payload["task_progress_profile_hash"] = str(
            task_progress_profile_hash
        )
    return payload


def load_task_progress_profile_file(
    path: str | Path,
) -> TaskProgressProfileV1:
    """Load one explicit external task-progress profile file."""

    profile_path = Path(path).expanduser().resolve()
    try:
        text = profile_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ProgressProfileCatalogError(
            f"Cannot read task progress profile {profile_path}: {exc}"
        ) from exc
    try:
        return TaskProgressProfileV1.from_payload(
            _editable_source_payload(text, source=str(profile_path))
        )
    except (TypeError, ValueError) as exc:
        raise ProgressProfileCatalogError(
            f"Invalid task progress profile {profile_path}: {exc}"
        ) from exc


def load_task_stage_profile_file(
    path: str | Path,
    *,
    progress_profile: TaskProgressProfileV1,
) -> TaskStageProfileV1:
    """Load and bind one explicit ordered-stage profile."""

    profile_path = Path(path).expanduser().resolve()
    try:
        text = profile_path.read_text(encoding="utf-8")
        profile = TaskStageProfileV1.from_payload(
            _editable_source_payload(
                text,
                source=str(profile_path),
                task_progress_profile_hash=(
                    progress_profile.semantic_content_hash
                ),
            )
        )
        validate_task_stage_profile(
            progress_profile=progress_profile,
            stage_profile=profile,
        )
        return profile
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        if isinstance(exc, ProgressProfileCatalogError):
            raise
        raise ProgressProfileCatalogError(
            f"Invalid task stage profile {profile_path}: {exc}"
        ) from exc


def _load_resource_profile(resource: Any) -> TaskProgressProfileV1:
    source = f"{resource.name}"
    try:
        text = resource.read_text(encoding="utf-8")
        return TaskProgressProfileV1.from_payload(
            _editable_source_payload(text, source=source)
        )
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        if isinstance(exc, ProgressProfileCatalogError):
            raise
        raise ProgressProfileCatalogError(
            f"Invalid managed task progress profile {source}: {exc}"
        ) from exc


def _load_resource_refinement(
    resource: Any,
    *,
    progress_profile: TaskProgressProfileV1,
) -> TaskRefinementProfileV1:
    source = f"{resource.name}"
    try:
        text = resource.read_text(encoding="utf-8")
        profile = TaskRefinementProfileV1.from_payload(
            _editable_source_payload(
                text,
                source=source,
                task_progress_profile_hash=(
                    progress_profile.semantic_content_hash
                ),
            )
        )
        validate_task_refinement_profile(
            progress_profile=progress_profile,
            refinement_profile=profile,
        )
        return profile
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        if isinstance(exc, ProgressProfileCatalogError):
            raise
        raise ProgressProfileCatalogError(
            f"Invalid managed task refinement profile {source}: {exc}"
        ) from exc


def _load_resource_stages(
    resource: Any,
    *,
    progress_profile: TaskProgressProfileV1,
) -> TaskStageProfileV1:
    source = f"{resource.name}"
    try:
        text = resource.read_text(encoding="utf-8")
        profile = TaskStageProfileV1.from_payload(
            _editable_source_payload(
                text,
                source=source,
                task_progress_profile_hash=(
                    progress_profile.semantic_content_hash
                ),
            )
        )
        validate_task_stage_profile(
            progress_profile=progress_profile,
            stage_profile=profile,
        )
        return profile
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        if isinstance(exc, ProgressProfileCatalogError):
            raise
        raise ProgressProfileCatalogError(
            f"Invalid managed task stage profile {source}: {exc}"
        ) from exc


def _profiles_for_topology(topology_kind: str) -> tuple[ManagedTaskProfile, ...]:
    topology = str(topology_kind).strip().upper()
    package_name = _TOPOLOGY_PACKAGES.get(topology)
    if package_name is None:
        raise ProgressProfileCatalogError(
            f"Unsupported catalog topology {topology_kind!r}."
        )
    root = resources.files(package_name)
    resources_by_name = {
        item.name: item
        for item in root.iterdir()
        if item.is_file()
    }
    progress_names = tuple(
        sorted(
            name
            for name in resources_by_name
            if name.endswith(_PROGRESS_SUFFIX)
        )
    )
    refinement_names = {
        name
        for name in resources_by_name
        if name.endswith(_REFINEMENT_SUFFIX)
    }
    stage_names = {
        name
        for name in resources_by_name
        if name.endswith(_STAGES_SUFFIX)
    }
    expected_refinements: set[str] = set()
    expected_stages: set[str] = set()
    bundles: list[ManagedTaskProfile] = []
    for progress_name in progress_names:
        stem = progress_name[: -len(_PROGRESS_SUFFIX)]
        refinement_name = f"{stem}{_REFINEMENT_SUFFIX}"
        stage_name = f"{stem}{_STAGES_SUFFIX}"
        progress_profile = _load_resource_profile(
            resources_by_name[progress_name]
        )
        if progress_profile.topology_kind != topology:
            raise ProgressProfileCatalogError(
                f"Managed profile {progress_profile.profile_id!r} declares "
                f"{progress_profile.topology_kind}, but is stored under "
                f"{topology}."
            )
        refinement_resource = resources_by_name.get(refinement_name)
        refinement_profile = None
        if refinement_resource is not None:
            expected_refinements.add(refinement_name)
            refinement_profile = _load_resource_refinement(
                refinement_resource,
                progress_profile=progress_profile,
            )
        stage_resource = resources_by_name.get(stage_name)
        stage_profile = None
        if stage_resource is not None:
            expected_stages.add(stage_name)
            stage_profile = _load_resource_stages(
                stage_resource,
                progress_profile=progress_profile,
            )
        bundles.append(
            ManagedTaskProfile(
                progress_profile=progress_profile,
                refinement_profile=refinement_profile,
                stage_profile=stage_profile,
                topology_kind=topology,
                progress_resource=f"{package_name}:{progress_name}",
                refinement_resource=(
                    None
                    if refinement_resource is None
                    else f"{package_name}:{refinement_name}"
                ),
                stage_resource=(
                    None
                    if stage_resource is None
                    else f"{package_name}:{stage_name}"
                ),
            )
        )
    if orphaned := refinement_names - expected_refinements:
        raise ProgressProfileCatalogError(
            "Managed refinement profile has no same-name progress profile: "
            f"{sorted(orphaned)}."
        )
    if orphaned := stage_names - expected_stages:
        raise ProgressProfileCatalogError(
            "Managed stage profile has no same-name progress profile: "
            f"{sorted(orphaned)}."
        )
    return tuple(bundles)


def list_managed_task_profiles(
    *,
    topology_kind: str | None = None,
) -> tuple[ManagedTaskProfile, ...]:
    """List validated catalog bundles in deterministic profile-ID order."""

    topologies = (
        tuple(_TOPOLOGY_PACKAGES)
        if topology_kind is None
        else (str(topology_kind).strip().upper(),)
    )
    bundles = tuple(
        bundle
        for topology in topologies
        for bundle in _profiles_for_topology(topology)
    )
    ids = tuple(bundle.profile_id for bundle in bundles)
    if len(ids) != len(set(ids)):
        raise ProgressProfileCatalogError(
            "Managed task progress profile_id values must be globally unique."
        )
    return tuple(sorted(bundles, key=lambda item: item.profile_id))


def load_managed_task_profile(profile_id: str) -> ManagedTaskProfile:
    """Resolve one exact managed task profile by canonical profile ID."""

    normalized = str(profile_id).strip()
    if not normalized:
        raise ProgressProfileCatalogError("profile_id must be non-empty.")
    matches = tuple(
        bundle
        for bundle in list_managed_task_profiles()
        if bundle.profile_id == normalized
    )
    if len(matches) != 1:
        available = tuple(
            bundle.profile_id for bundle in list_managed_task_profiles()
        )
        raise ProgressProfileCatalogError(
            f"Unknown managed task profile {normalized!r}; "
            f"available={available}."
        )
    return matches[0]


def resolve_task_progress_profile_reference(
    reference: str | Path,
) -> TaskProgressProfileV1:
    """Resolve ``catalog:<profile_id>`` or an explicit filesystem JSON path."""

    normalized = str(reference).strip()
    if not normalized:
        raise ProgressProfileCatalogError(
            "Task progress profile reference must be non-empty."
        )
    prefix = "catalog:"
    if normalized.startswith(prefix):
        return load_managed_task_profile(
            normalized[len(prefix) :]
        ).progress_profile
    return load_task_progress_profile_file(normalized)


def resolve_task_stage_profile_reference(
    reference: str | Path,
    *,
    progress_profile: TaskProgressProfileV1,
) -> TaskStageProfileV1:
    """Resolve managed ``catalog:`` or explicit ordered-stage profile JSON."""

    normalized = str(reference).strip()
    if not normalized:
        raise ProgressProfileCatalogError(
            "Task stage profile reference must be non-empty."
        )
    prefix = "catalog:"
    if normalized.startswith(prefix):
        bundle = load_managed_task_profile(normalized[len(prefix) :])
        if bundle.stage_profile is None:
            raise ProgressProfileCatalogError(
                f"Managed profile {bundle.profile_id!r} has no ordered-stage "
                "profile."
            )
        if (
            bundle.progress_profile.semantic_content_hash
            != progress_profile.semantic_content_hash
        ):
            raise ProgressProfileCatalogError(
                "Managed stage profile is bound to a different progress "
                "profile."
            )
        return bundle.stage_profile
    return load_task_stage_profile_file(
        normalized,
        progress_profile=progress_profile,
    )


__all__ = [
    "ManagedTaskProfile",
    "ProgressProfileCatalogError",
    "list_managed_task_profiles",
    "load_managed_task_profile",
    "load_task_progress_profile_file",
    "load_task_stage_profile_file",
    "resolve_task_progress_profile_reference",
    "resolve_task_stage_profile_reference",
]
