from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from utils.dataset_postprocess.real_undistort.dataset import inspect_intrinsics, undistort_dataset
from utils.dataset_postprocess.real_undistort.intrinsics import CameraIntrinsics, build_undistort_maps, require_cv2
from utils.dataset_postprocess.real_undistort.video import process_video


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_video(path: Path, *, frame_count: int = 4, size: tuple[int, int] = (64, 48)) -> None:
    cv2 = require_cv2()
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, size)
    assert writer.isOpened()
    width, height = size
    for index in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = (index * 30) % 255
        frame[:, :, 1] = np.arange(width, dtype=np.uint8)[None, :]
        frame[:, :, 2] = np.arange(height, dtype=np.uint8)[:, None]
        writer.write(frame)
    writer.release()


def test_build_undistort_maps_returns_zero_distortion_intrinsics() -> None:
    intrinsics = CameraIntrinsics(
        width=64,
        height=48,
        fx=40.0,
        fy=39.0,
        ppx=31.5,
        ppy=23.5,
        coeffs=(-0.05, 0.03, 0.001, 0.0, 0.0),
    )

    maps = build_undistort_maps(intrinsics, alpha=0.0)

    assert maps.map1.shape[:2] == (48, 64)
    assert maps.output.width == 64
    assert maps.output.height == 48
    assert maps.output.coeffs == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_process_video_writes_undistorted_video(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    _write_video(source)
    intrinsics = CameraIntrinsics(
        width=64,
        height=48,
        fx=40.0,
        fy=39.0,
        ppx=31.5,
        ppy=23.5,
        coeffs=(-0.05, 0.03, 0.001, 0.0, 0.0),
    )

    result = process_video(source, output, intrinsics, overwrite=True)

    assert output.exists()
    assert result.frame_count == 4
    assert result.output_intrinsics.coeffs == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_process_video_does_not_write_through_hardlinked_target(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    _write_video(source, frame_count=4)
    output.hardlink_to(source)
    source_before = source.read_bytes()
    source_inode = source.stat().st_ino
    intrinsics = CameraIntrinsics(
        width=64,
        height=48,
        fx=40.0,
        fy=39.0,
        ppx=31.5,
        ppy=23.5,
    )

    with pytest.raises(RuntimeError, match="expected=5, decoded=4"):
        process_video(
            source,
            output,
            intrinsics,
            expected_frame_count=5,
            overwrite=True,
        )

    assert source.stat().st_ino == source_inode
    assert output.stat().st_ino == source_inode
    assert source.read_bytes() == source_before
    assert not tuple(tmp_path.glob(".*.partial-*"))

    result = process_video(
        source,
        output,
        intrinsics,
        expected_frame_count=4,
        overwrite=True,
    )
    assert result.frame_count == 4
    assert source.stat().st_ino == source_inode
    assert output.stat().st_ino != source_inode
    assert source.read_bytes() == source_before


def test_undistort_dataset_updates_devices_and_manifest(tmp_path: Path) -> None:
    source_root = tmp_path / "dataset"
    output_root = tmp_path / "dataset_undistorted"
    video_rel = "videos/observation.images.left_wrist/chunk-000/episode_000000.mp4"
    _write_video(source_root / video_rel)
    devices = {
        "left_wrist": {
            "intrinsics": {
                "width": 64,
                "height": 48,
                "fx": 40.0,
                "fy": 39.0,
                "ppx": 31.5,
                "ppy": 23.5,
                "coeffs": [-0.05, 0.03, 0.001, 0.0, 0.0],
            },
            "metadata": {"mount": "test"},
        }
    }
    _write_json(source_root / "meta/devices/realsense.json", devices)
    _write_json(source_root / "meta/info.json", {"dataset_name": "test"})
    _write_jsonl(
        source_root / "meta/episodes.jsonl",
        [
            {
                "episode_index": 0,
                "episode_id": "episode_000000",
                "frame_count": 4,
                "video_paths": {"left_wrist": video_rel},
            }
        ],
    )
    _write_json(source_root / "data/chunk-000/episode_000000.parquet", {"placeholder": True})

    result = undistort_dataset(
        source_root,
        output_root,
        cameras=("left_wrist",),
        copy_mode="copy",
        jobs=1,
    )

    assert result.manifest_path.exists()
    assert (output_root / video_rel).exists()
    updated = json.loads((output_root / "meta/devices/realsense.json").read_text(encoding="utf-8"))
    assert updated["left_wrist"]["intrinsics"]["coeffs"] == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert updated["left_wrist"]["metadata"]["undistorted"] is True
    assert (output_root / "meta/devices/realsense_raw_before_undistort.json").exists()


def test_inspect_intrinsics_selects_distorted_cameras(tmp_path: Path) -> None:
    devices_path = tmp_path / "realsense.json"
    _write_json(
        devices_path,
        {
            "front": {
                "intrinsics": {
                    "width": 64,
                    "height": 48,
                    "fx": 40.0,
                    "fy": 39.0,
                    "ppx": 31.5,
                    "ppy": 23.5,
                    "coeffs": [0, 0, 0, 0, 0],
                }
            },
            "left_wrist": {
                "intrinsics": {
                    "width": 64,
                    "height": 48,
                    "fx": 40.0,
                    "fy": 39.0,
                    "ppx": 31.5,
                    "ppy": 23.5,
                    "coeffs": [-0.05, 0.03, 0, 0, 0],
                }
            },
        },
    )

    payload = inspect_intrinsics(devices_path)

    assert tuple(payload) == ("left_wrist",)
