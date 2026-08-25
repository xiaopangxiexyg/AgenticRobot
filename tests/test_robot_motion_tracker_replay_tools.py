from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from robot_motion.tests.tracker_replay import (
    OfflineTrackerReplayFrame,
    OfflineTrackerReplayReport,
    SideReplayError,
    _observation_from_row,
    report_exceeds_thresholds,
    summarize_offline_tracker_frames,
)
from robot_motion.tests.view_tracker_replay import (
    _parse_active_sides,
    _play_report_in_mujoco,
    _resolve_viewer_urdf,
    _should_print,
)
from robot_motion.tests.write_tracker_replay_report import write_offline_tracker_gui
from sim.contracts.scene import DEFAULT_RM65_PHYSICS_URDF


def _frame(index: int) -> OfflineTrackerReplayFrame:
    return OfflineTrackerReplayFrame(
        row_index=index,
        frame_index=index,
        timestamp=float(index),
        frame_time_sec=float(index) / 15.0,
        status="tracking",
        fallback_reason=None,
        source_pose_desk={"left": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]},
        original_action_pose_base={"left": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]},
        replay_action_pose_base={"left": [0.01 * index, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]},
        original_qpos={"left": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        replay_qpos={"left": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        errors={
            "left": SideReplayError(
                translation_error_m=0.01 * index,
                rotation_error_deg=0.5 * index,
                qpos_max_abs_error_deg=1.0 * index,
                qpos_rms_error_deg=0.5 * index,
            )
        },
        source_vs_action_errors={
            "left": SideReplayError(
                translation_error_m=0.001 * index,
                rotation_error_deg=0.1 * index,
            )
        },
    )


def test_tracker_replay_binds_historical_desk_pose_to_vive_retarget_frame() -> None:
    row = {
        "timestamp": 1.0,
        "teleop.action_source.left_wrist_pose_desk": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        "teleop.action_source.left_wrist_pose_desk_frame": "vive_retarget_frame",
    }

    observation = _observation_from_row(row, ("left",))

    assert observation.channels["left_wrist_pose"].frame_name == "vive_retarget_frame"


def test_tracker_replay_rejects_canonical_desk_label_on_historical_pose_column() -> None:
    row = {
        "timestamp": 1.0,
        "teleop.action_source.left_wrist_pose_desk": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        "teleop.action_source.left_wrist_pose_desk_frame": "canonical_desk",
    }

    with pytest.raises(ValueError, match="expected 'vive_retarget_frame'"):
        _observation_from_row(row, ("left",))


def test_offline_tracker_summary_separates_source_and_ik_errors() -> None:
    summary = summarize_offline_tracker_frames([_frame(1), _frame(2)], active_sides=("left",))

    left = summary["by_side"]["left"]
    assert left["ik_vs_action_translation_error_m"]["max"] == 0.02
    assert left["source_vs_action_translation_error_m"]["max"] == 0.002
    assert left["qpos_max_abs_error_deg"]["max"] == 2.0


def test_offline_tracker_thresholds_check_ik_error() -> None:
    report = OfflineTrackerReplayReport(
        dataset_root=Path("/tmp/dataset"),
        episode_id="episode_000000",
        episode_index=0,
        parquet_path=Path("/tmp/dataset/data/chunk-000/episode_000000.parquet"),
        active_sides=("left",),
        frame_count=2,
        source_frame_count=2,
        start_frame=0,
        max_frames=0,
        nominal_hz=15.0,
        timing_source="unit",
        summary=summarize_offline_tracker_frames([_frame(1), _frame(2)], active_sides=("left",)),
        frames=[_frame(1), _frame(2)],
    )

    exceeded, failures = report_exceeds_thresholds(report, max_translation_error_m=0.015)

    assert exceeded is True
    assert failures == ["left.translation_error_m.max=0.02 > 0.015"]


@pytest.mark.integration
def test_offline_tracker_gui_writes_portable_html(tmp_path: Path) -> None:
    report = OfflineTrackerReplayReport(
        dataset_root=tmp_path,
        episode_id="episode_000000",
        episode_index=0,
        parquet_path=tmp_path / "data/chunk-000/episode_000000.parquet",
        active_sides=("left",),
        frame_count=1,
        source_frame_count=1,
        start_frame=0,
        max_frames=1,
        nominal_hz=15.0,
        timing_source="unit",
        summary=summarize_offline_tracker_frames([_frame(1)], active_sides=("left",)),
        frames=[_frame(1)],
        video_paths={"front": "videos/front.mp4"},
    )

    output = write_offline_tracker_gui(report, tmp_path / "visualizations" / "tracker.html")

    text = output.read_text(encoding="utf-8")
    assert "Offline Tracker Replay - episode_000000" in text
    assert "source_vs_action_translation_error_m" in text
    assert "../videos/front.mp4" in text


def test_offline_tracker_viewer_helpers_use_dual_rm65_defaults() -> None:
    assert _parse_active_sides("both") == ("left", "right")
    assert _parse_active_sides("right") == ("right",)

    expected = (Path(__file__).resolve().parents[1] / DEFAULT_RM65_PHYSICS_URDF).resolve()
    resolved = _resolve_viewer_urdf(None)
    assert resolved == expected
    assert resolved.name == (
        "rm65_dual_arm_wrist_cameras_v3_head_camera_4b2_cad_with_legacy_housing.urdf"
    )


def test_offline_tracker_viewer_progress_predicate_is_keyword_only() -> None:
    assert _should_print(processed=1, total=10, print_every=30) is True
    assert _should_print(processed=2, total=10, print_every=30) is False
    assert _should_print(processed=30, total=100, print_every=30) is True


def test_offline_tracker_viewer_play_loop_runs_with_fake_mujoco(monkeypatch) -> None:
    fake_viewer = _FakeViewer()
    mujoco_module = types.ModuleType("mujoco")
    viewer_module = types.ModuleType("mujoco.viewer")
    viewer_module.launch_passive = lambda *args, **kwargs: fake_viewer
    mujoco_module.viewer = viewer_module
    monkeypatch.setitem(sys.modules, "mujoco", mujoco_module)
    monkeypatch.setitem(sys.modules, "mujoco.viewer", viewer_module)

    scene = _FakeScene()
    report = OfflineTrackerReplayReport(
        dataset_root=Path("/tmp/dataset"),
        episode_id="episode_000000",
        episode_index=0,
        parquet_path=Path("/tmp/dataset/data/chunk-000/episode_000000.parquet"),
        active_sides=("left",),
        frame_count=1,
        source_frame_count=1,
        start_frame=0,
        max_frames=1,
        nominal_hz=15.0,
        timing_source="unit",
        summary=summarize_offline_tracker_frames([_frame(1)], active_sides=("left",)),
        frames=[_frame(1)],
    )
    args = argparse.Namespace(
        hide_left_ui=True,
        hide_right_ui=True,
        viewer_duration=0.0,
        speed=1.0,
        no_sleep=True,
        qpos_source="tracker_ik",
        print_every=1,
        loop=False,
    )

    _play_report_in_mujoco(scene, report, args=args, active_sides=("left",))

    assert fake_viewer.synced is True
    assert fake_viewer.closed is True
    assert scene.qpos_by_side["left"] == [0.0] * 6


class _FakeViewer:
    def __init__(self) -> None:
        self.synced = False
        self.closed = False

    def is_running(self) -> bool:
        return True

    def sync(self) -> None:
        self.synced = True

    def close(self) -> None:
        self.closed = True


class _FakeScene:
    model = object()
    data = object()

    def __init__(self) -> None:
        self.qpos_by_side: dict[str, list[float]] = {}

    def set_qpos_by_side(self, qpos_by_side) -> None:
        self.qpos_by_side = {
            side: np.asarray(values, dtype=float).reshape(6).tolist()
            for side, values in qpos_by_side.items()
        }

    def arm_qpos_by_side(self) -> dict[str, list[float]]:
        return dict(self.qpos_by_side)
