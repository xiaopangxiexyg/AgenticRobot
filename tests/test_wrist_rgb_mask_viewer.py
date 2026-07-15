from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

viewer = importlib.import_module("utils.portable_vis.wrist_mask_view.wrist_rgb_mask_viewer")

from utils.dataset_postprocess.wrist_cam_adjust.depth import (  # noqa: E402
    DepthRangeMaskConfig,
    mask_manifest_payload,
    pack_binary_mask,
)


def _write_video(path: Path, values: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (16, 12),
    )
    assert writer.isOpened()
    try:
        for value in values:
            frame = np.full((12, 16, 3), int(value), dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def _write_mask(root: Path, camera_key: str, masks: list[object]) -> str:
    relative_dir = (
        Path("wrist_depth")
        / "mask"
        / f"observation.images.{camera_key}"
        / "chunk-000"
        / "episode_000000"
    )
    stream_rel = relative_dir / "stream.bitpack.bin"
    manifest_rel = relative_dir / "manifest.json"
    stream_path = root / stream_rel
    stream_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = [np.asarray(mask, dtype=np.uint8) for mask in masks]
    stream_path.write_bytes(b"".join(pack_binary_mask(mask) for mask in arrays))
    height, width = arrays[0].shape
    records = [
        {
            "sample_frame_index": index,
            "depth_frame_index": index,
            "foreground_fraction": float(np.count_nonzero(mask)) / float(mask.size),
        }
        for index, mask in enumerate(arrays)
    ]
    payload = mask_manifest_payload(
        camera_key=camera_key,
        relative_dir=str(relative_dir),
        stream_path=str(stream_rel),
        frame_count=len(arrays),
        height=int(height),
        width=int(width),
        records=records,
        config=DepthRangeMaskConfig(),
    )
    (root / manifest_rel).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(manifest_rel)


def _write_dataset(root: Path) -> dict[str, str]:
    left_video = root / "videos/observation.images.left_wrist/chunk-000/episode_000000.avi"
    right_video = root / "videos/observation.images.right_wrist/chunk-000/episode_000000.avi"
    _write_video(left_video, [30, 90])
    _write_video(right_video, [150, 210])
    left_mask = _write_mask(
        root,
        "left_wrist",
        [
            [[1, 0], [0, 1]],
            [[1, 1], [0, 0]],
        ],
    )
    right_mask = _write_mask(
        root,
        "right_wrist",
        [
            [[0, 1], [1, 0]],
            [[0, 0], [1, 1]],
        ],
    )
    parquet_rel = "data/chunk-000/episode_000000.parquet"
    (root / parquet_rel).parent.mkdir(parents=True, exist_ok=True)
    (root / parquet_rel).touch()
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta/info.json").write_text(
        json.dumps({"format": "lerobot_v2.1_style", "chunk_size": 1000, "nominal_fps": 5.0}),
        encoding="utf-8",
    )
    record = {
        "episode_index": 0,
        "episode_id": "episode_000000",
        "frame_count": 2,
        "parquet_path": parquet_rel,
        "video_paths": {
            "left_wrist": str(left_video.relative_to(root)),
            "right_wrist": str(right_video.relative_to(root)),
        },
        "wrist_depth_mask_paths": {
            "left_wrist": left_mask,
            "right_wrist": right_mask,
        },
        "metadata": {},
        "summary": {},
    }
    (root / "meta/episodes.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    return {"parquet": parquet_rel, "left_mask": left_mask}


def test_parse_episode_index_from_common_paths() -> None:
    assert viewer._parse_episode_index_from_path("/tmp/episode_000123.parquet") == 123
    assert viewer._parse_episode_index_from_path("/tmp/no_episode.parquet") is None


def test_script_path_execution_bootstraps_repo_imports() -> None:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            "utils/portable_vis/wrist_mask_view/wrist_rgb_mask_viewer.py",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "wrist depth masks" in result.stdout


def test_load_episode_view_from_dataset_root_and_parquet(tmp_path: Path) -> None:
    paths = _write_dataset(tmp_path)

    root_view = viewer.load_episode_view(tmp_path, episode_index=0)
    parquet_view = viewer.load_episode_view(tmp_path / paths["parquet"])
    try:
        assert root_view.dataset_root == tmp_path.resolve()
        assert root_view.episode_index == 0
        assert root_view.frame_count == 2
        assert [camera.camera_key for camera in root_view.cameras] == ["left_wrist", "right_wrist"]
        assert [camera.video_frame_count for camera in root_view.cameras] == [2, 2]
        assert [camera.mask_frame_count for camera in root_view.cameras] == [2, 2]
        assert parquet_view.episode_id == "episode_000000"
    finally:
        root_view.close()
        parquet_view.close()


def test_jpeg_frame_for_episode_view(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    episode_view = viewer.load_episode_view(tmp_path, episode_index=0)
    try:
        jpeg = viewer._jpeg_frame_for_view(episode_view, frame_index=1, mode="mask", width=640)
        assert jpeg.startswith(b"\xff\xd8")
        assert jpeg.endswith(b"\xff\xd9")

        overlay = viewer.compose_episode_frame(episode_view, frame_index=1, mode="overlay", width=640)
        assert overlay.ndim == 3
        assert overlay.shape[0] > 0
        assert overlay.shape[1] >= 600
    finally:
        episode_view.close()


def test_direct_mask_directory_overrides_dataset_masks(tmp_path: Path) -> None:
    paths = _write_dataset(tmp_path)
    direct_dir = tmp_path / Path(paths["left_mask"]).parent

    overrides = viewer._direct_mask_manifest_overrides(direct_dir, ("left_wrist", "right_wrist"))

    assert overrides == {"left_wrist": (direct_dir / "manifest.json").resolve()}
