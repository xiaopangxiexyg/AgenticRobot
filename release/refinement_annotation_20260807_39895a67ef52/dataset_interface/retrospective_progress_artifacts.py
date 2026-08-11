"""Read-only verification of external retrospective Progress workspaces."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from dataset_contracts.canonical_encoding import canonical_loads
from dataset_contracts.progress_facts import (
    PROGRESS_BINDING_FILE_DOMAIN_V1,
    PROGRESS_PROFILE_FILE_DOMAIN_V1,
)
from dataset_contracts.progress_profile import (
    ProgressEpisodeBindingV1,
    TaskProgressProfileV1,
)
from dataset_contracts.progress_stages import (
    TASK_STAGE_PROFILE_FILE_DOMAIN_V1,
    ProgressStageEntryBindingV1,
    TaskStageProfileV1,
    validate_task_stage_profile,
)
from dataset_contracts.retrospective_progress import (
    RETROSPECTIVE_DATASET_LOCAL_SOURCE_LOCATOR_V1,
    RETROSPECTIVE_PROGRESS_MIGRATION_MANIFEST_FILE_DOMAIN_V1,
    RETROSPECTIVE_PROGRESS_WINDOW_FILE_DOMAIN_V1,
    RETROSPECTIVE_SOURCE_BINDING_FILE_DOMAIN_V1,
    RetrospectiveProgressMigrationManifestV1,
    RetrospectiveProgressWindowV1,
    RetrospectiveSourceBindingV1,
)

from .episode_curation import (
    source_dataset_base_fingerprint,
    source_episode_manifest_hash,
)
from .retrospective_progress import (
    ResolvedRetrospectiveProgressWindow,
    resolve_retrospective_progress_window,
)
from .sample_axis import EpisodeSampleAxis


class RetrospectiveProgressArtifactReadError(ValueError):
    """Raised when a migrated workspace or its external source has drifted."""


@dataclass(frozen=True)
class LoadedRetrospectiveProgressEpisode:
    episode_dir: Path
    progress_profile: TaskProgressProfileV1
    progress_binding: ProgressEpisodeBindingV1
    stage_profile: TaskStageProfileV1
    source_binding: RetrospectiveSourceBindingV1
    window: RetrospectiveProgressWindowV1
    migration_manifest: RetrospectiveProgressMigrationManifestV1
    resolved_source_dataset_root: Path
    sample_axis: EpisodeSampleAxis
    timeline: ResolvedRetrospectiveProgressWindow

    # Annotation consumers use this capture-neutral read shape.  These
    # aliases let the existing GUI consume a visibly retrospective timeline
    # without manufacturing capture facts or a second annotation workflow.
    @property
    def profile(self) -> TaskProgressProfileV1:
        return self.progress_profile

    @property
    def binding(self) -> ProgressEpisodeBindingV1:
        return self.progress_binding

    @property
    def manifest(self) -> RetrospectiveProgressMigrationManifestV1:
        return self.migration_manifest

    @property
    def stage_entry_binding(self) -> ProgressStageEntryBindingV1:
        stages = self.stage_profile.stages_for_step(
            self.progress_binding.entry_step_id
        )
        if not stages:
            raise RetrospectiveProgressArtifactReadError(
                "Retrospective entry step has no fine-stage definition."
            )
        return ProgressStageEntryBindingV1(
            episode_capture_identity_hash=(
                self.source_binding.episode_capture_identity_hash
            ),
            task_stage_profile_hash=self.stage_profile.semantic_content_hash,
            entry_stage_id=stages[0].stage_id,
            entry_condition="NOMINAL",
        )


def _read_mapping(path: Path, *, domain: str) -> Mapping[str, Any]:
    try:
        payload = canonical_loads(path.read_bytes(), expected_domain=domain)
    except (OSError, TypeError, ValueError) as exc:
        raise RetrospectiveProgressArtifactReadError(
            f"Cannot read retrospective artifact {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise RetrospectiveProgressArtifactReadError(
            f"Retrospective artifact must contain an object: {path}"
        )
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _bound_file(root: Path, relative: str) -> Path:
    try:
        path = (root / relative).resolve(strict=True)
        path.relative_to(root)
    except (OSError, ValueError) as exc:
        raise RetrospectiveProgressArtifactReadError(
            f"External source path is missing or escapes its bound root: {relative}"
        ) from exc
    if not path.is_file():
        raise RetrospectiveProgressArtifactReadError(
            f"External source artifact is not a regular file: {path}"
        )
    return path


def _verify_hash(path: Path, expected: str, *, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise RetrospectiveProgressArtifactReadError(
            f"{label} hash drift: expected={expected} actual={actual}."
        )


def load_retrospective_progress_workspace_profiles(
    workspace_root: str | Path,
) -> tuple[TaskProgressProfileV1, TaskStageProfileV1]:
    """Load the immutable one-step and fine-stage profiles for a workspace."""

    root = Path(workspace_root).expanduser().resolve()
    try:
        progress_profile = TaskProgressProfileV1.from_payload(
            _read_mapping(
                root / "profile.cjson",
                domain=PROGRESS_PROFILE_FILE_DOMAIN_V1,
            )
        )
        stage_profile = TaskStageProfileV1.from_payload(
            _read_mapping(
                root / "stages.cjson",
                domain=TASK_STAGE_PROFILE_FILE_DOMAIN_V1,
            )
        )
        validate_task_stage_profile(
            progress_profile=progress_profile,
            stage_profile=stage_profile,
        )
    except RetrospectiveProgressArtifactReadError:
        raise
    except (TypeError, ValueError) as exc:
        raise RetrospectiveProgressArtifactReadError(
            f"Invalid retrospective workspace profiles under {root}: {exc}"
        ) from exc
    return progress_profile, stage_profile


def resolve_retrospective_source_dataset_root(
    source: RetrospectiveSourceBindingV1,
    *,
    allowed_source_roots: Sequence[str | Path],
) -> Path:
    """Resolve an external absolute locator or one dataset-local ``.``."""

    allowed = tuple(Path(value).expanduser().resolve() for value in allowed_source_roots)
    if source.source_dataset_locator == RETROSPECTIVE_DATASET_LOCAL_SOURCE_LOCATOR_V1:
        if len(allowed) != 1:
            raise RetrospectiveProgressArtifactReadError(
                "A dataset-local retrospective source requires exactly one "
                "owning dataset root."
            )
        return allowed[0]
    declared_root = Path(source.source_dataset_locator).expanduser().resolve()
    if declared_root not in allowed:
        raise RetrospectiveProgressArtifactReadError(
            "External source root is absent from the explicit allowlist."
        )
    return declared_root


def _load_source_axis(
    source: RetrospectiveSourceBindingV1,
    *,
    resolved_source_root: Path,
) -> EpisodeSampleAxis:
    declared_root = resolved_source_root
    info_path = _bound_file(declared_root, "meta/info.json")
    _verify_hash(info_path, source.source_info_hash, label="source info")
    if source_dataset_base_fingerprint(declared_root) != source.source_dataset_base_fingerprint:
        raise RetrospectiveProgressArtifactReadError(
            "External source dataset fingerprint drift."
        )
    record_hash = source_episode_manifest_hash(
        declared_root,
        source.source_episode_index,
    )
    if record_hash != source.source_episode_record_hash:
        raise RetrospectiveProgressArtifactReadError(
            "External source Episode record drift."
        )
    parquet_path = _bound_file(
        declared_root,
        source.source_parquet_relative_path,
    )
    stat = parquet_path.stat()
    if stat.st_size != source.source_parquet_byte_count:
        raise RetrospectiveProgressArtifactReadError(
            "External source Parquet size drift."
        )
    _verify_hash(parquet_path, source.source_parquet_hash, label="source Parquet")
    for video in source.admitted_videos:
        video_path = _bound_file(declared_root, video.source_relative_path)
        if video_path.stat().st_size != video.byte_count:
            raise RetrospectiveProgressArtifactReadError(
                f"{video.modality_id} video size drift."
            )
        _verify_hash(video_path, video.content_hash, label=video.modality_id)
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]

        values = tuple(
            float(value.as_py())
            for value in pq.read_table(parquet_path, columns=["timestamp"])[
                "timestamp"
            ]
        )
    except Exception as exc:
        raise RetrospectiveProgressArtifactReadError(
            f"Cannot load external source sample axis: {exc}"
        ) from exc
    if len(values) != source.admitted_videos[0].decoded_frame_count:
        raise RetrospectiveProgressArtifactReadError(
            "External source sample/video frame counts differ."
        )
    return EpisodeSampleAxis(clock_domain_id="episode_clock", cut_times=values)


def load_retrospective_progress_episode(
    episode_dir: str | Path,
    *,
    progress_profile: TaskProgressProfileV1,
    stage_profile: TaskStageProfileV1,
    allowed_source_roots: Sequence[str | Path],
) -> LoadedRetrospectiveProgressEpisode:
    """Load one migration seal and revalidate every admitted source artifact."""

    root = Path(episode_dir).expanduser().resolve()
    try:
        source = RetrospectiveSourceBindingV1.from_payload(
            _read_mapping(
                root / "source_binding.cjson",
                domain=RETROSPECTIVE_SOURCE_BINDING_FILE_DOMAIN_V1,
            )
        )
        progress_binding = ProgressEpisodeBindingV1.from_payload(
            _read_mapping(
                root / "progress_binding.cjson",
                domain=PROGRESS_BINDING_FILE_DOMAIN_V1,
            )
        )
        window = RetrospectiveProgressWindowV1.from_payload(
            _read_mapping(
                root / "retrospective_window.cjson",
                domain=RETROSPECTIVE_PROGRESS_WINDOW_FILE_DOMAIN_V1,
            )
        )
        manifest = RetrospectiveProgressMigrationManifestV1.from_payload(
            _read_mapping(
                root / "migration_manifest.cjson",
                domain=(
                    RETROSPECTIVE_PROGRESS_MIGRATION_MANIFEST_FILE_DOMAIN_V1
                ),
            )
        )
        resolved_source_root = resolve_retrospective_source_dataset_root(
            source,
            allowed_source_roots=allowed_source_roots,
        )
        sample_axis = _load_source_axis(
            source,
            resolved_source_root=resolved_source_root,
        )
        timeline = resolve_retrospective_progress_window(
            progress_profile=progress_profile,
            progress_binding=progress_binding,
            stage_profile=stage_profile,
            sample_axis=sample_axis,
            source_binding=source,
            window=window,
            manifest=manifest,
        )
    except RetrospectiveProgressArtifactReadError:
        raise
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RetrospectiveProgressArtifactReadError(
            f"Invalid retrospective Progress Episode {root}: {exc}"
        ) from exc
    return LoadedRetrospectiveProgressEpisode(
        episode_dir=root,
        progress_profile=progress_profile,
        progress_binding=progress_binding,
        stage_profile=stage_profile,
        source_binding=source,
        window=window,
        migration_manifest=manifest,
        resolved_source_dataset_root=resolved_source_root,
        sample_axis=sample_axis,
        timeline=timeline,
    )


__all__ = [
    "LoadedRetrospectiveProgressEpisode",
    "RetrospectiveProgressArtifactReadError",
    "load_retrospective_progress_episode",
    "load_retrospective_progress_workspace_profiles",
    "resolve_retrospective_source_dataset_root",
]
