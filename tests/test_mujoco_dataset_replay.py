from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sim.mujoco_real2sim.controllers import QposController
from sim.mujoco_real2sim.replay.dataset import (
    fallback_hz,
    parquet_column_names,
    read_parquet_rows,
    resolve_episode,
)
from sim.mujoco_real2sim.replay.runner import (
    _close_viewer,
    _source_camera_intrinsics,
    build_arg_parser,
    run,
)
from sim.mujoco_real2sim.replay.trajectory import (
    extract_gripper_sample,
    extract_qpos_sample,
    gripper_read_columns,
    qpos_read_columns,
)
from sim.mujoco_real2sim.scene import MujocoScene
from sim.mujoco_real2sim.urdf import discover_camera_mounts, parse_urdf


URDF = Path("assets/rm65_inspire_gripper_v2/rm65_dual_arm_wrist_cameras_v2_head_camera.urdf")
DATASET = Path("datasets/2026-07-03_vive_rm65_real_teleop")


def test_qpos_sample_auto_prefers_actual_qpos() -> None:
    row = {
        "frame_index": 7,
        "timestamp": 101.5,
        "observation.arm.right.actual_qpos": [1, 2, 3, 4, 5, 6],
        "observation.arm.right.current_qpos": [6, 5, 4, 3, 2, 1],
    }

    sample = extract_qpos_sample(
        row,
        active_sides=("right",),
        source="auto",
        first_timestamp=100.0,
        fallback_hz=30.0,
        row_offset=0,
    )

    assert sample.frame_index == 7
    assert sample.relative_time_sec == pytest.approx(1.5)
    assert sample.source_by_side["right"] == "observation.arm.right.actual_qpos"
    np.testing.assert_allclose(sample.qpos_by_side["right"], [1, 2, 3, 4, 5, 6])


def test_gripper_sample_auto_prefers_action_then_state() -> None:
    row = {
        "frame_index": 8,
        "timestamp": 102.0,
        "action.gripper.right.target": float("nan"),
        "observation.gripper.right.state": 0.75,
        "action.replay.right_hand_joint_target": [0, 0.2, 0, 0, 0, 0],
    }

    sample = extract_gripper_sample(
        row,
        gripper_side_by_name={"eg2_4b2": "right"},
        source="auto",
        command_dim=1,
        first_timestamp=100.0,
        fallback_hz=30.0,
        row_offset=0,
    )

    assert sample.frame_index == 8
    assert sample.relative_time_sec == pytest.approx(2.0)
    assert sample.close_by_name["eg2_4b2"] == pytest.approx(0.75)
    assert sample.source_by_name["eg2_4b2"] == "observation.gripper.right.state"


def test_close_viewer_waits_for_handle_to_stop() -> None:
    class FakeViewerHandle:
        def __init__(self) -> None:
            self.closed = False
            self.polls = 0

        def close(self) -> None:
            self.closed = True

        def is_running(self) -> bool:
            self.polls += 1
            return self.polls < 2

    handle = FakeViewerHandle()

    _close_viewer(handle, timeout_sec=0.2)

    assert handle.closed
    assert handle.polls >= 2


def test_source_dataset_intrinsics_override_front_camera_fovy() -> None:
    if not DATASET.exists():
        pytest.skip(f"Local replay dataset not found: {DATASET}")

    intrinsics = _source_camera_intrinsics(DATASET)
    mounts = discover_camera_mounts(
        parse_urdf(URDF),
        camera_intrinsics_by_name=intrinsics,
    )
    by_name = {mount.name: mount for mount in mounts}

    assert by_name["front"].fovy_deg == pytest.approx(43.2615, abs=1e-3)
    assert by_name["front"].intrinsics is not None
    assert by_name["front"].intrinsics["fy"] == pytest.approx(605.2217, abs=1e-3)


def test_dataset_qpos_can_drive_mujoco_scene_first_frame() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("pyarrow")
    if not DATASET.exists():
        pytest.skip(f"Local replay dataset not found: {DATASET}")

    episode = resolve_episode(DATASET, "0")
    available = set(parquet_column_names(episode.parquet_path))
    columns = [column for column in qpos_read_columns(("left", "right"), "auto") if column in available]
    rows = read_parquet_rows(episode.parquet_path, columns=columns)
    scene = MujocoScene.from_urdf(URDF, camera_names=("all",))

    sample = extract_qpos_sample(
        rows[0],
        active_sides=("left", "right"),
        source="auto",
        first_timestamp=rows[0].get("timestamp"),
        fallback_hz=fallback_hz(episode),
        row_offset=0,
    )
    result = QposController(scene).apply(sample.qpos_by_side)

    np.testing.assert_allclose(scene.get_arm_qpos("left"), sample.qpos_by_side["left"])
    np.testing.assert_allclose(scene.get_arm_qpos("right"), sample.qpos_by_side["right"])
    np.testing.assert_allclose(result["right"], sample.qpos_by_side["right"])


def test_qpos_replay_runner_no_render_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("pyarrow")
    if not DATASET.exists():
        pytest.skip(f"Local replay dataset not found: {DATASET}")

    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--dataset-root",
            str(DATASET),
            "--episode",
            "0",
            "--urdf",
            str(URDF),
            "--mode",
            "right",
            "--qpos-source",
            "auto",
            "--max-frames",
            "2",
            "--no-render",
            "--no-sleep",
            "--print-every",
            "0",
        ]
    )

    run(args)

    captured = capsys.readouterr()
    assert '"processed_frames": 2' in captured.out


def test_qpos_replay_exports_lerobot_style_sim_dataset(
    tmp_path: Path,
) -> None:
    pytest.importorskip("cv2")
    pytest.importorskip("mujoco")
    pytest.importorskip("pyarrow")
    if not DATASET.exists():
        pytest.skip(f"Local replay dataset not found: {DATASET}")

    export_root = tmp_path / "sim_replay_dataset"
    env = dict(os.environ)
    env["MUJOCO_GL"] = "egl"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sim.mujoco_real2sim.replay.runner",
            "--dataset-root",
            str(DATASET),
            "--episode",
            "0",
            "--urdf",
            str(URDF),
            "--mode",
            "dual",
            "--qpos-source",
            "auto",
            "--max-frames",
            "2",
            "--no-sleep",
            "--print-every",
            "0",
            "--width",
            "160",
            "--height",
            "120",
            "--export-dataset",
            str(export_root),
            "--export-fps",
            "15",
        ],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        output = result.stdout + result.stderr
        if "Failed to create MuJoCo renderer" in output or "GLFW" in output:
            pytest.skip(output)
        raise AssertionError(output)

    from teleop_runtime.replay.dataset import ReplayDatasetIndex

    index = ReplayDatasetIndex(export_root)
    episode = index.load_episode("0")

    assert episode.frame_count == 2
    assert set(episode.summary.camera_keys) == {"front", "left_wrist", "right_wrist"}
    assert episode.rows[0]["observation.images.front.video_frame_index"] == 0
    assert (export_root / "videos/observation.images.front/chunk-000/episode_000000.mp4").exists()
    info = json.loads((export_root / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["sim_replay"]["camera_metadata"]["front"]["fovy_deg"] == pytest.approx(
        43.2615,
        abs=1e-3,
    )
