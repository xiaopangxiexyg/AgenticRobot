from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

compare = importlib.import_module("utils.portable_vis.multi_view.real_sim_video_compare")


def _write_video(path: Path, values: list[int], *, fps: float = 5.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (16, 12),
    )
    assert writer.isOpened()
    try:
        for value in values:
            frame = np.full((12, 16, 3), int(value), dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def _write_dataset(root: Path, camera_values: dict[str, list[int]]) -> None:
    video_paths: dict[str, str] = {}
    for camera_name, values in camera_values.items():
        video_rel = Path("videos") / f"observation.images.{camera_name}" / "chunk-000" / "episode_000000.avi"
        _write_video(root / video_rel, values)
        video_paths[camera_name] = str(video_rel)

    parquet_rel = "data/chunk-000/episode_000000.parquet"
    (root / parquet_rel).parent.mkdir(parents=True, exist_ok=True)
    (root / parquet_rel).touch()
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text(
        json.dumps({"format": "lerobot_v2.1_style", "nominal_fps": 5.0}),
        encoding="utf-8",
    )
    record = {
        "episode_index": 0,
        "episode_id": "episode_000000",
        "frame_count": 3,
        "parquet_path": parquet_rel,
        "video_paths": video_paths,
    }
    (root / "meta" / "episodes.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_open_camera_pairs_uses_common_preferred_camera_order(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    sim_root = tmp_path / "sim"
    _write_dataset(real_root, {"right_wrist": [10], "front": [20], "left_wrist": [30]})
    _write_dataset(sim_root, {"left_wrist": [40], "right_wrist": [50], "front": [60]})

    pairs = compare.open_camera_pairs(
        real_dataset=real_root,
        sim_dataset=sim_root,
        episode="0",
        cameras=(),
        real_label="real",
        sim_label="sim",
    )
    try:
        assert [pair.camera_name for pair in pairs] == ["front", "left_wrist", "right_wrist"]
        assert pairs[0].real.frame_count == 1
        assert pairs[0].sim.fps == pytest.approx(5.0)
    finally:
        compare.close_camera_pairs(pairs)


def test_open_camera_pairs_auto_skips_declared_but_missing_videos(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    sim_root = tmp_path / "sim"
    _write_dataset(real_root, {"front": [10], "left_wrist": [20]})
    _write_dataset(sim_root, {"front": [30], "left_wrist": [40]})
    (real_root / "videos/observation.images.front/chunk-000/episode_000000.avi").unlink()

    pairs = compare.open_camera_pairs(
        real_dataset=real_root,
        sim_dataset=sim_root,
        episode="0",
        cameras=(),
        real_label="real",
        sim_label="sim",
    )
    try:
        assert [pair.camera_name for pair in pairs] == ["left_wrist"]
    finally:
        compare.close_camera_pairs(pairs)


def test_export_real_sim_comparison_video(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    sim_root = tmp_path / "sim"
    _write_dataset(real_root, {"left_wrist": [10, 20, 30], "right_wrist": [40, 50, 60]})
    _write_dataset(sim_root, {"left_wrist": [70, 80, 90], "right_wrist": [100, 110, 120]})

    pairs = compare.open_camera_pairs(
        real_dataset=real_root,
        sim_dataset=sim_root,
        episode="0",
        cameras=("left_wrist", "right_wrist"),
        real_label="real",
        sim_label="sim",
    )
    try:
        output = tmp_path / "comparison.avi"
        summary = compare.export_real_sim_comparison(
            pairs,
            output_path=output,
            cell_size=(32, 24),
            fps=5.0,
            max_frames=3,
            codec="MJPG",
        )
    finally:
        compare.close_camera_pairs(pairs)

    assert output.exists()
    assert summary["frame_count"] == 3
    assert summary["resolution"] == [64, 96]
    assert summary["cameras"] == ["left_wrist", "right_wrist"]

    capture = cv2.VideoCapture(str(output))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
        assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 64
        assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 96
    finally:
        capture.release()
