"""Crash-safe persistence for post-capture ordered-stage annotations.

The mutable draft is deliberately not a dataset contract.  Only a validated
``StageAnnotationBundleV1`` is published as an immutable annotation set.
One advisory lock covers the complete dataset root so two annotation programs
cannot edit different Episodes under the same dataset concurrently.
"""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from dataset_contracts.progress_facts import ProgressEpisodeManifestV1
from dataset_contracts.progress_stage_annotations import (
    STAGE_ANNOTATION_BINDING_FILE_DOMAIN_V1,
    STAGE_ANNOTATION_BUNDLE_FILE_DOMAIN_V1,
    STAGE_ANNOTATION_MANIFEST_FILE_DOMAIN_V1,
    StageAnnotationBundleV1,
    StageAnnotationManifestV1,
    StageAnnotationSetBindingV1,
)
from dataset_contracts.progress_stages import TaskStageProfileV1
from dataset_interface.progress_stage_annotation_artifacts import (
    LoadedStageAnnotationSet,
    STAGE_ANNOTATION_RELATIVE_ROOT,
    StageAnnotationArtifactReadError,
    load_stage_annotation_set as _load_stage_annotation_set,
)

from ..immutable_canonical_store import (
    fsync_directory,
    publish_immutable_canonical,
)


STAGE_ANNOTATION_DRAFT_RELATIVE_ROOT = Path(
    "meta/progress/stage_annotation_drafts"
)
STAGE_ANNOTATION_WRITER_LOCK_RELATIVE_PATH = (
    STAGE_ANNOTATION_RELATIVE_ROOT / ".writer.lock"
)


class StageAnnotationStoreError(RuntimeError):
    """Raised when an annotation set is incomplete or contradictory."""


@dataclass(frozen=True)
class StageAnnotationCommitReport:
    annotation_dir: Path
    manifest: StageAnnotationManifestV1
    idempotent: bool


def _validate_set(
    *,
    source_progress_manifest: ProgressEpisodeManifestV1,
    stage_profile: TaskStageProfileV1,
    binding: StageAnnotationSetBindingV1,
    bundle: StageAnnotationBundleV1,
) -> None:
    if (
        binding.episode_capture_identity_hash
        != source_progress_manifest.episode_capture_identity_hash
    ):
        raise StageAnnotationStoreError(
            "Annotation binding and progress manifest Episode identity differ."
        )
    if (
        binding.source_progress_manifest_hash
        != source_progress_manifest.semantic_content_hash
    ):
        raise StageAnnotationStoreError(
            "Annotation binding does not reference the exact sealed progress "
            "manifest."
        )
    if (
        binding.task_stage_profile_hash
        != stage_profile.semantic_content_hash
    ):
        raise StageAnnotationStoreError(
            "Annotation binding and task stage profile differ."
        )
    clocks = tuple(
        item.clock_domain_id
        for item in (
            bundle.occurrences
            + bundle.abnormal_intervals
            + bundle.progress_boundary_overrides
        )
    )
    if any(clock != binding.bound_clock_domain_id for clock in clocks):
        raise StageAnnotationStoreError(
            "Every timeline annotation must use the bound Episode clock."
        )


def commit_stage_annotation_set(
    *,
    dataset_root: str | Path,
    episode_uuid: str,
    source_progress_manifest: ProgressEpisodeManifestV1,
    stage_profile: TaskStageProfileV1,
    binding: StageAnnotationSetBindingV1,
    bundle: StageAnnotationBundleV1,
) -> StageAnnotationCommitReport:
    """Publish one immutable set; the manifest is always written last."""

    normalized_episode_uuid = str(episode_uuid).strip()
    if not normalized_episode_uuid:
        raise ValueError("episode_uuid must be non-empty.")
    if binding.annotation_set_id.strip() in {".", ".."}:
        raise ValueError("annotation_set_id is not path-safe.")
    _validate_set(
        source_progress_manifest=source_progress_manifest,
        stage_profile=stage_profile,
        binding=binding,
        bundle=bundle,
    )
    root = Path(dataset_root).expanduser().resolve()
    annotation_dir = (
        root
        / STAGE_ANNOTATION_RELATIVE_ROOT
        / normalized_episode_uuid
        / binding.annotation_set_id
    ).resolve()
    expected_parent = (
        root / STAGE_ANNOTATION_RELATIVE_ROOT / normalized_episode_uuid
    ).resolve()
    if annotation_dir.parent != expected_parent:
        raise ValueError("Stage annotation path escaped its owned root.")
    manifest = StageAnnotationManifestV1(
        annotation_set_binding_hash=binding.semantic_content_hash,
        annotation_bundle_hash=bundle.semantic_content_hash,
    )
    try:
        binding_idempotent = publish_immutable_canonical(
            annotation_dir / "binding.cjson",
            binding.to_payload(),
            domain=STAGE_ANNOTATION_BINDING_FILE_DOMAIN_V1,
        )
        bundle_idempotent = publish_immutable_canonical(
            annotation_dir / "annotations.cjson",
            bundle.to_payload(),
            domain=STAGE_ANNOTATION_BUNDLE_FILE_DOMAIN_V1,
        )
        manifest_idempotent = publish_immutable_canonical(
            annotation_dir / "manifest.cjson",
            manifest.to_payload(),
            domain=STAGE_ANNOTATION_MANIFEST_FILE_DOMAIN_V1,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise StageAnnotationStoreError(
            f"Cannot publish immutable stage annotation set: {exc}"
        ) from exc
    return StageAnnotationCommitReport(
        annotation_dir=annotation_dir,
        manifest=manifest,
        idempotent=(
            binding_idempotent
            and bundle_idempotent
            and manifest_idempotent
        ),
    )


def load_stage_annotation_set(
    annotation_dir: str | Path,
) -> LoadedStageAnnotationSet:
    """Compatibility reader for the dataset-interface owned implementation."""

    try:
        return _load_stage_annotation_set(annotation_dir)
    except StageAnnotationArtifactReadError as exc:
        raise StageAnnotationStoreError(
            str(exc)
        ) from exc


class DatasetStageAnnotationWorkspace:
    """Exclusive editable workspace that may coexist with dataset capture."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        blocking: bool = False,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.blocking = bool(blocking)
        self._lock_file = None

    def acquire(self) -> None:
        if self._lock_file is not None:
            return
        lock_path = (
            self.dataset_root
            / STAGE_ANNOTATION_WRITER_LOCK_RELATIVE_PATH
        )
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = lock_path.open("a+b")
        operation = fcntl.LOCK_EX
        if not self.blocking:
            operation |= fcntl.LOCK_NB
        try:
            fcntl.flock(lock_file.fileno(), operation)
        except BlockingIOError as exc:
            lock_file.close()
            raise StageAnnotationStoreError(
                "Another stage annotation program already owns this dataset."
            ) from exc
        self._lock_file = lock_file

    def close(self) -> None:
        lock_file = self._lock_file
        self._lock_file = None
        if lock_file is None:
            return
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()

    def draft_path(self, *, episode_uuid: str) -> Path:
        normalized = str(episode_uuid).strip()
        if not normalized or normalized in {".", ".."}:
            raise ValueError("episode_uuid must be a path-safe string.")
        path = (
            self.dataset_root
            / STAGE_ANNOTATION_DRAFT_RELATIVE_ROOT
            / normalized
            / "draft.json"
        ).resolve()
        expected_parent = (
            self.dataset_root
            / STAGE_ANNOTATION_DRAFT_RELATIVE_ROOT
            / normalized
        ).resolve()
        if path.parent != expected_parent:
            raise ValueError("Stage annotation draft path escaped its root.")
        return path

    def load_draft(self, *, episode_uuid: str) -> dict[str, Any] | None:
        self._require_lock()
        path = self.draft_path(episode_uuid=episode_uuid)
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise StageAnnotationStoreError(
                f"Cannot load stage annotation draft {path}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise StageAnnotationStoreError(
                "Stage annotation draft root must be a JSON object."
            )
        return value

    def save_draft(
        self,
        *,
        episode_uuid: str,
        draft: Mapping[str, Any],
    ) -> Path:
        self._require_lock()
        if not isinstance(draft, Mapping):
            raise TypeError("draft must be a mapping.")
        path = self.draft_path(episode_uuid=episode_uuid)
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(
            dict(draft),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        fd, name = tempfile.mkstemp(
            prefix=".draft.",
            suffix=".tmp",
            dir=str(path.parent),
        )
        temporary = Path(name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def _require_lock(self) -> None:
        if self._lock_file is None:
            raise StageAnnotationStoreError(
                "Stage annotation workspace lock is not held."
            )

    def __enter__(self) -> "DatasetStageAnnotationWorkspace":
        self.acquire()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


__all__ = [
    "STAGE_ANNOTATION_DRAFT_RELATIVE_ROOT",
    "STAGE_ANNOTATION_RELATIVE_ROOT",
    "STAGE_ANNOTATION_WRITER_LOCK_RELATIVE_PATH",
    "DatasetStageAnnotationWorkspace",
    "LoadedStageAnnotationSet",
    "StageAnnotationCommitReport",
    "StageAnnotationStoreError",
    "commit_stage_annotation_set",
    "load_stage_annotation_set",
]
