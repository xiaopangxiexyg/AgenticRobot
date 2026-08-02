from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dataset_interface.media_integrity import (
    require_dataset_media_integrity,
    validate_dataset_media,
)


def _write_video(path: Path, frame_count: int) -> None:
    cv2 = pytest.importorskip("cv2")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        15.0,
        (32, 24),
    )
    assert writer.isOpened()
    try:
        for index in range(frame_count):
            writer.write(np.full((24, 32, 3), index, dtype=np.uint8))
    finally:
        writer.release()


def _write_episode(root: Path, *, declared: int, actual: int) -> None:
    relative = "videos/observation.images.left_wrist/chunk-000/episode_000000.mp4"
    _write_video(root / relative, actual)
    (root / "meta").mkdir(parents=True)
    (root / "meta/episodes.jsonl").write_text(
        json.dumps(
            {
                "episode_index": 0,
                "frame_count": declared,
                "video_paths": {"left_wrist": relative},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _append_episode(
    root: Path,
    *,
    episode_index: int,
    declared: int,
    actual: int,
) -> None:
    relative = (
        "videos/observation.images.left_wrist/chunk-000/"
        f"episode_{episode_index:06d}.mp4"
    )
    _write_video(root / relative, actual)
    with (root / "meta/episodes.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "episode_index": episode_index,
                    "frame_count": declared,
                    "video_paths": {"left_wrist": relative},
                }
            )
            + "\n"
        )


def test_validate_dataset_media_accepts_exact_video_contract(tmp_path: Path) -> None:
    _write_episode(tmp_path, declared=4, actual=4)

    report = validate_dataset_media(tmp_path, camera_keys=("left_wrist",))

    assert report.ok
    assert report.checked_video_count == 1


def test_require_dataset_media_integrity_rejects_truncated_video(tmp_path: Path) -> None:
    _write_episode(tmp_path, declared=4, actual=2)

    report = validate_dataset_media(tmp_path, camera_keys=("left_wrist",))

    assert not report.ok
    assert report.issues[0].reason == "decoder_frame_count_mismatch"
    with pytest.raises(RuntimeError, match="declared=4 decoded=2"):
        require_dataset_media_integrity(tmp_path, camera_keys=("left_wrist",))


def test_validate_dataset_media_limits_checks_to_selected_episodes(
    tmp_path: Path,
) -> None:
    _write_episode(tmp_path, declared=4, actual=4)
    _append_episode(tmp_path, episode_index=1, declared=4, actual=2)

    selected = validate_dataset_media(
        tmp_path,
        camera_keys=("left_wrist",),
        episode_indices=(0,),
    )
    all_episodes = validate_dataset_media(
        tmp_path,
        camera_keys=("left_wrist",),
    )

    assert selected.ok
    assert selected.checked_video_count == 1
    assert not all_episodes.ok
    assert all_episodes.issues[0].episode_index == 1
