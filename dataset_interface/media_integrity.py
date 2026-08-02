"""Read-only RGB video integrity checks for training dataset admission."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class VideoIntegrityIssue:
    episode_index: int
    camera_key: str
    relative_path: str
    reason: str
    declared_frame_count: int
    decoder_frame_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "camera_key": self.camera_key,
            "relative_path": self.relative_path,
            "reason": self.reason,
            "declared_frame_count": self.declared_frame_count,
            "decoder_frame_count": self.decoder_frame_count,
        }


@dataclass
class DatasetMediaIntegrityReport:
    dataset_root: str
    camera_keys: tuple[str, ...]
    checked_video_count: int = 0
    issues: list[VideoIntegrityIssue] = field(default_factory=list)
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "dexteleop_dataset_media_integrity_v1",
            "ok": self.ok,
            "dataset_root": self.dataset_root,
            "camera_keys": list(self.camera_keys),
            "checked_video_count": self.checked_video_count,
            "issue_count": len(self.issues),
            "truncated": self.truncated,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def _read_episode_records(dataset_root: Path) -> list[dict[str, Any]]:
    path = dataset_root / "meta" / "episodes.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"dataset episode metadata is missing: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        records.append(value)
    return records


def _camera_keys(
    records: Iterable[Mapping[str, Any]],
    requested: Iterable[str] | None,
) -> tuple[str, ...]:
    if requested is not None:
        return tuple(dict.fromkeys(str(key) for key in requested))
    discovered: set[str] = set()
    for record in records:
        paths = record.get("video_paths")
        if isinstance(paths, Mapping):
            discovered.update(str(key) for key in paths)
    return tuple(sorted(discovered))


def validate_dataset_media(
    dataset_root: str | Path,
    *,
    camera_keys: Iterable[str] | None = None,
    episode_indices: Iterable[int] | None = None,
    max_issues: int = 100,
    verify_tail_decode: bool = True,
) -> DatasetMediaIntegrityReport:
    """Validate declared frame counts and seek/decode each final RGB frame."""

    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "dataset media validation requires opencv-python"
        ) from exc

    root = Path(dataset_root).expanduser().resolve()
    records = _read_episode_records(root)
    selected_episode_indices = (
        None
        if episode_indices is None
        else frozenset(int(index) for index in episode_indices)
    )
    selected_cameras = _camera_keys(records, camera_keys)
    report = DatasetMediaIntegrityReport(
        dataset_root=str(root),
        camera_keys=selected_cameras,
    )
    limit = max(1, int(max_issues))

    def add_issue(issue: VideoIntegrityIssue) -> bool:
        report.issues.append(issue)
        if len(report.issues) >= limit:
            report.truncated = True
            return True
        return False

    for record in records:
        try:
            episode_index = int(record["episode_index"])
            declared_frame_count = int(record["frame_count"])
        except (KeyError, TypeError, ValueError):
            if add_issue(
                VideoIntegrityIssue(
                    episode_index=-1,
                    camera_key="",
                    relative_path="",
                    reason="invalid_episode_or_frame_count",
                    declared_frame_count=-1,
                )
            ):
                break
            continue
        if (
            selected_episode_indices is not None
            and episode_index not in selected_episode_indices
        ):
            continue
        paths = record.get("video_paths")
        if not isinstance(paths, Mapping):
            paths = {}
        for camera_key in selected_cameras:
            relative = paths.get(camera_key)
            if not relative:
                if add_issue(
                    VideoIntegrityIssue(
                        episode_index=episode_index,
                        camera_key=camera_key,
                        relative_path="",
                        reason="missing_video_path",
                        declared_frame_count=declared_frame_count,
                    )
                ):
                    return report
                continue
            relative_path = str(relative)
            path = root / relative_path
            report.checked_video_count += 1
            if not path.is_file():
                if add_issue(
                    VideoIntegrityIssue(
                        episode_index=episode_index,
                        camera_key=camera_key,
                        relative_path=relative_path,
                        reason="missing_video_file",
                        declared_frame_count=declared_frame_count,
                    )
                ):
                    return report
                continue
            capture = cv2.VideoCapture(str(path))
            try:
                if not capture.isOpened():
                    if add_issue(
                        VideoIntegrityIssue(
                            episode_index=episode_index,
                            camera_key=camera_key,
                            relative_path=relative_path,
                            reason="decoder_open_failed",
                            declared_frame_count=declared_frame_count,
                        )
                    ):
                        return report
                    continue
                decoder_frame_count = int(
                    round(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                )
                if decoder_frame_count != declared_frame_count:
                    if add_issue(
                        VideoIntegrityIssue(
                            episode_index=episode_index,
                            camera_key=camera_key,
                            relative_path=relative_path,
                            reason="decoder_frame_count_mismatch",
                            declared_frame_count=declared_frame_count,
                            decoder_frame_count=decoder_frame_count,
                        )
                    ):
                        return report
                    continue
                if verify_tail_decode and declared_frame_count > 0:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, declared_frame_count - 1)
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        if add_issue(
                            VideoIntegrityIssue(
                                episode_index=episode_index,
                                camera_key=camera_key,
                                relative_path=relative_path,
                                reason="tail_frame_decode_failed",
                                declared_frame_count=declared_frame_count,
                                decoder_frame_count=decoder_frame_count,
                            )
                        ):
                            return report
            finally:
                capture.release()
    return report


def require_dataset_media_integrity(
    dataset_root: str | Path,
    *,
    camera_keys: Iterable[str],
    episode_indices: Iterable[int] | None = None,
) -> DatasetMediaIntegrityReport:
    report = validate_dataset_media(
        dataset_root,
        camera_keys=camera_keys,
        episode_indices=episode_indices,
        max_issues=20,
        verify_tail_decode=True,
    )
    if report.ok:
        return report
    first = report.issues[0]
    raise RuntimeError(
        "DexTeleop RGB media integrity check failed before training: "
        f"dataset={report.dataset_root}, checked={report.checked_video_count}, "
        f"issues>={len(report.issues)}, first=episode {first.episode_index} "
        f"camera {first.camera_key!r} reason={first.reason} "
        f"declared={first.declared_frame_count} decoded={first.decoder_frame_count}. "
        "Use a frozen legacy config only to reproduce an existing run; corrected "
        "configs require repaired media."
    )


__all__ = [
    "DatasetMediaIntegrityReport",
    "VideoIntegrityIssue",
    "require_dataset_media_integrity",
    "validate_dataset_media",
]
