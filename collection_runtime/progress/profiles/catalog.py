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
_DETAILED_PROGRESS_SUFFIX = ".progress_detailed.json"
_REFINEMENT_SUFFIX = ".refinement.json"
_STAGES_SUFFIX = ".stages.json"
_DETAILED_STAGES_SUFFIX = ".stages_detailed.json"
DETAILED_PROFILE_SELECTION_SUFFIX = "_detailed"
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
    variant: str = "coarse"

    @property
    def profile_id(self) -> str:
        return self.progress_profile.profile_id

    @property
    def is_detailed(self) -> bool:
        return self.variant == "detailed"

    @property
    def selection_id(self) -> str:
        """Stable GUI/CLI selection ID for this catalog variant.

        Detailed profiles intentionally keep the canonical progress profile ID
        unchanged: the progress timeline and its hash are the same as the
        coarse capture profile.  The variant suffix is therefore only a
        selection-level namespace and never becomes a persisted Episode
        profile ID.
        """

        if self.is_detailed:
            return f"{self.profile_id}{DETAILED_PROFILE_SELECTION_SUFFIX}"
        return self.profile_id


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


def _profiles_for_topology(
    topology_kind: str,
    *,
    include_detailed: bool = False,
) -> tuple[ManagedTaskProfile, ...]:
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
    detailed_progress_names = {
        name
        for name in resources_by_name
        if name.endswith(_DETAILED_PROGRESS_SUFFIX)
    }
    detailed_stage_names = {
        name
        for name in resources_by_name
        if name.endswith(_DETAILED_STAGES_SUFFIX)
    }
    expected_refinements: set[str] = set()
    expected_stages: set[str] = set()
    expected_detailed_progress: set[str] = set()
    expected_detailed_stages: set[str] = set()
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
        if include_detailed:
            detailed_progress_name = (
                f"{stem}{_DETAILED_PROGRESS_SUFFIX}"
            )
            detailed_stage_name = f"{stem}{_DETAILED_STAGES_SUFFIX}"
            detailed_progress_resource = resources_by_name.get(
                detailed_progress_name
            )
            detailed_stage_resource = resources_by_name.get(
                detailed_stage_name
            )
            if (
                detailed_progress_resource is None
                and detailed_stage_resource is None
            ):
                continue
            if (
                detailed_progress_resource is None
                or detailed_stage_resource is None
            ):
                missing = (
                    detailed_progress_name
                    if detailed_progress_resource is None
                    else detailed_stage_name
                )
                present = (
                    detailed_stage_name
                    if detailed_progress_resource is None
                    else detailed_progress_name
                )
                raise ProgressProfileCatalogError(
                    "Managed detailed profile companion is incomplete: "
                    f"{present} exists but {missing} is missing."
                )
            expected_detailed_progress.add(detailed_progress_name)
            expected_detailed_stages.add(detailed_stage_name)
            detailed_progress = _load_resource_profile(
                detailed_progress_resource
            )
            if detailed_progress.topology_kind != topology:
                raise ProgressProfileCatalogError(
                    f"Managed detailed profile {detailed_progress.profile_id!r} "
                    f"declares {detailed_progress.topology_kind}, but is "
                    f"stored under {topology}."
                )
            if (
                detailed_progress.semantic_content_hash
                != progress_profile.semantic_content_hash
            ):
                raise ProgressProfileCatalogError(
                    "Managed detailed profile must preserve the coarse "
                    "progress timeline/hash: "
                    f"{detailed_progress_name}."
                )
            detailed_stage = _load_resource_stages(
                detailed_stage_resource,
                progress_profile=detailed_progress,
            )
            bundles.append(
                ManagedTaskProfile(
                    progress_profile=detailed_progress,
                    refinement_profile=refinement_profile,
                    stage_profile=detailed_stage,
                    topology_kind=topology,
                    progress_resource=(
                        f"{package_name}:{detailed_progress_name}"
                    ),
                    refinement_resource=(
                        None
                        if refinement_resource is None
                        else f"{package_name}:{refinement_name}"
                    ),
                    stage_resource=(
                        f"{package_name}:{detailed_stage_name}"
                    ),
                    variant="detailed",
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
    if include_detailed:
        if orphaned := detailed_progress_names - expected_detailed_progress:
            raise ProgressProfileCatalogError(
                "Managed detailed progress profile has no same-name coarse "
                "progress profile: "
                f"{sorted(orphaned)}."
            )
        if orphaned := detailed_stage_names - expected_detailed_stages:
            raise ProgressProfileCatalogError(
                "Managed detailed stage profile has no same-name detailed "
                "progress profile: "
                f"{sorted(orphaned)}."
            )
    return tuple(bundles)


def list_managed_task_profiles(
    *,
    topology_kind: str | None = None,
    include_detailed: bool = False,
) -> tuple[ManagedTaskProfile, ...]:
    """List validated catalog bundles in deterministic selection-ID order.

    ``include_detailed`` is explicit so existing callers keep the original
    coarse-only catalog and its selection semantics.  Detailed variants are
    additive bundles selected with ``<profile_id>_detailed``.
    """

    topologies = (
        tuple(_TOPOLOGY_PACKAGES)
        if topology_kind is None
        else (str(topology_kind).strip().upper(),)
    )
    bundles = tuple(
        bundle
        for topology in topologies
        for bundle in _profiles_for_topology(
            topology,
            include_detailed=include_detailed,
        )
    )
    ids = tuple(bundle.selection_id for bundle in bundles)
    if len(ids) != len(set(ids)):
        raise ProgressProfileCatalogError(
            "Managed task profile selection IDs must be globally unique."
        )
    return tuple(sorted(bundles, key=lambda item: item.selection_id))


def load_managed_task_profile(profile_id: str) -> ManagedTaskProfile:
    """Resolve one exact managed profile by canonical or variant ID."""

    normalized = str(profile_id).strip()
    if not normalized:
        raise ProgressProfileCatalogError("profile_id must be non-empty.")
    include_detailed = normalized.endswith(DETAILED_PROFILE_SELECTION_SUFFIX)
    matches = tuple(
        bundle
        for bundle in list_managed_task_profiles(
            include_detailed=include_detailed,
        )
        if bundle.selection_id == normalized
    )
    if len(matches) != 1:
        available = tuple(
            bundle.selection_id
            for bundle in list_managed_task_profiles(
                include_detailed=include_detailed,
            )
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
    "DETAILED_PROFILE_SELECTION_SUFFIX",
    "ManagedTaskProfile",
    "ProgressProfileCatalogError",
    "list_managed_task_profiles",
    "load_managed_task_profile",
    "load_task_progress_profile_file",
    "load_task_stage_profile_file",
    "resolve_task_progress_profile_reference",
    "resolve_task_stage_profile_reference",
]
