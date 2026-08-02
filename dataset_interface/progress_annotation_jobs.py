"""Read-only work-item discovery for post-capture stage annotation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .progress_artifacts import (
    ProgressArtifactReadError,
    SealedProgressEpisode,
    load_sealed_progress_episode,
)
from .sample_axis import EpisodeSampleAxis


class ProgressAnnotationJobError(ValueError):
    """Raised when finalized Episode metadata cannot form a safe work item."""


@dataclass(frozen=True)
class ProgressAnnotationWorkItem:
    dataset_root: Path
    episode_index: int
    episode_uuid: str
    episode_identity_hash: str
    progress_manifest_hash: str
    progress_episode_dir: Path
    parquet_path: Path
    video_paths: Mapping[str, Path]
    frame_count: int
    start_timestamp: float
    end_timestamp: float
    progress: SealedProgressEpisode
    annotation_set_count: int
    has_draft: bool

    @property
    def default_camera_key(self) -> str:
        if "front" in self.video_paths:
            return "front"
        return next(iter(sorted(self.video_paths)))

    def summary_payload(self) -> dict[str, Any]:
        stage_count = (
            len(self.progress.stage_profile.stages)
            if self.progress.stage_profile is not None
            else 0
        )
        return {
            "episode_index": self.episode_index,
            "episode_uuid": self.episode_uuid,
            "episode_identity_hash": self.episode_identity_hash,
            "progress_manifest_hash": self.progress_manifest_hash,
            "frame_count": self.frame_count,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "duration_sec": max(
                0.0,
                self.end_timestamp - self.start_timestamp,
            ),
            "camera_keys": list(sorted(self.video_paths)),
            "default_camera_key": self.default_camera_key,
            "annotation_set_count": self.annotation_set_count,
            "has_draft": self.has_draft,
            "progress_profile_id": self.progress.profile.profile_id,
            "task_stage_profile_id": (
                None
                if self.progress.stage_profile is None
                else self.progress.stage_profile.profile_id
            ),
            # UI-only scheduling hint: a single-stage Episode can wait for an
            # explicit video request, while multi-stage Episodes benefit from
            # eager playback and next-Episode warmup.
            "stage_count": stage_count,
        }


def _safe_dataset_path(root: Path, value: object, *, name: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ProgressAnnotationJobError(f"{name} is missing.")
    path = (root / text).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ProgressAnnotationJobError(
            f"{name} escapes the dataset root."
        ) from exc
    if not path.is_file():
        raise ProgressAnnotationJobError(
            f"{name} does not exist: {text}."
        )
    return path


def _episode_records(dataset_root: Path) -> tuple[Mapping[str, Any], ...]:
    path = dataset_root / "meta" / "episodes.jsonl"
    if not path.is_file():
        return ()
    records: list[Mapping[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ProgressAnnotationJobError(
            f"Cannot read finalized Episode catalog: {exc}"
        ) from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProgressAnnotationJobError(
                f"Invalid episodes.jsonl line {line_number}: {exc}"
            ) from exc
        if not isinstance(value, Mapping):
            raise ProgressAnnotationJobError(
                f"episodes.jsonl line {line_number} is not an object."
            )
        records.append(value)
    return tuple(records)


def list_progress_annotation_work_items(
    dataset_root: str | Path,
) -> tuple[ProgressAnnotationWorkItem, ...]:
    """List only finalized Episodes with complete sealed stage semantics."""

    root = Path(dataset_root).expanduser().resolve()
    results: list[ProgressAnnotationWorkItem] = []
    for record in _episode_records(root):
        metadata = record.get("metadata")
        progress_ref = (
            metadata.get("progress_episode")
            if isinstance(metadata, Mapping)
            else None
        )
        if not isinstance(progress_ref, Mapping):
            continue
        episode_uuid = str(progress_ref.get("episode_uuid", "")).strip()
        manifest_ref = str(progress_ref.get("manifest_ref", "")).strip()
        if not episode_uuid or not manifest_ref:
            continue
        progress_dir = _safe_dataset_path(
            root,
            manifest_ref,
            name="progress manifest_ref",
        ).parent
        try:
            # The sealed Episode owns the exact profile/binding snapshot used
            # during capture.  The corresponding metadata fields are legacy
            # references and must not make an Episode disappear when an
            # editable collection configuration has since changed.
            progress = load_sealed_progress_episode(
                progress_dir,
                expected_episode_identity_hash=str(
                    progress_ref.get("episode_identity_hash", "")
                ),
            )
        except ProgressArtifactReadError:
            # An active or partially finalized Episode is intentionally
            # invisible to the annotation application.
            continue
        if (
            progress.stage_profile is None
            or progress.stage_entry_binding is None
        ):
            continue
        video_values = record.get("video_paths", {})
        if not isinstance(video_values, Mapping) or not video_values:
            continue
        videos = {
            str(camera_key): _safe_dataset_path(
                root,
                relative_path,
                name=f"video_paths[{camera_key!r}]",
            )
            for camera_key, relative_path in video_values.items()
        }
        parquet_path = _safe_dataset_path(
            root,
            record.get("parquet_path"),
            name="parquet_path",
        )
        try:
            episode_index = int(record.get("episode_index"))
            frame_count = int(record.get("frame_count"))
            start_timestamp = float(record.get("start_timestamp"))
            end_timestamp = float(record.get("end_timestamp"))
        except (TypeError, ValueError) as exc:
            raise ProgressAnnotationJobError(
                f"Episode {episode_uuid!r} has invalid summary values."
            ) from exc
        annotation_root = (
            root
            / "meta"
            / "progress"
            / "stage_annotations"
            / episode_uuid
        )
        set_count = (
            sum(
                1
                for child in annotation_root.iterdir()
                if child.is_dir() and (child / "manifest.cjson").is_file()
            )
            if annotation_root.is_dir()
            else 0
        )
        results.append(
            ProgressAnnotationWorkItem(
                dataset_root=root,
                episode_index=episode_index,
                episode_uuid=episode_uuid,
                episode_identity_hash=(
                    progress.binding.episode_capture_identity_hash
                ),
                progress_manifest_hash=(
                    progress.manifest.semantic_content_hash
                ),
                progress_episode_dir=progress_dir,
                parquet_path=parquet_path,
                video_paths=videos,
                frame_count=frame_count,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                progress=progress,
                annotation_set_count=set_count,
                has_draft=(
                    root
                    / "meta"
                    / "progress"
                    / "stage_annotation_drafts"
                    / episode_uuid
                    / "draft.json"
                ).is_file(),
            )
        )
    return tuple(sorted(results, key=lambda item: item.episode_index))


def load_episode_sample_axis(
    work_item: ProgressAnnotationWorkItem,
) -> EpisodeSampleAxis:
    """Load only the timestamp column needed by timeline indexing."""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ProgressAnnotationJobError(
            "pyarrow is required to read finalized Episode timestamps."
        ) from exc
    try:
        table = pq.read_table(work_item.parquet_path, columns=["timestamp"])
        values = tuple(float(value.as_py()) for value in table["timestamp"])
    except Exception as exc:
        raise ProgressAnnotationJobError(
            f"Cannot read Episode sample timestamps: {exc}"
        ) from exc
    if len(values) != work_item.frame_count:
        raise ProgressAnnotationJobError(
            "Parquet timestamp count differs from finalized frame_count."
        )
    return EpisodeSampleAxis(
        clock_domain_id=work_item.progress.timeline.clock_domain_id,
        cut_times=values,
    )


__all__ = [
    "ProgressAnnotationJobError",
    "ProgressAnnotationWorkItem",
    "list_progress_annotation_work_items",
    "load_episode_sample_axis",
]
