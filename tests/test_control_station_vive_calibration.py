from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from control_station.vive_calibration import (
    ViveCoordinateCalibrationConfig,
    ViveCoordinateCalibrationController,
)
from control_station.vive_calibration_gui import ViveCoordinateCalibrationGui
from control_station.vive_calibration_profiles import (
    VIVE_COORDINATE_PROFILE_KIND,
    coordinate_profile_view,
    load_coordinate_profile,
    new_coordinate_calibration,
    save_coordinate_profile,
    validate_coordinate_profile,
)
from project_paths import calibration_profiles_dir, control_station_vive_coordinate_profiles_dir
from teleop_runtime.calibration import TeleopCalibration
from teleop_runtime.real_teleop.vive_calibration_runtime import (
    ViveCoordinateCalibrationContext,
    calibrate_vive_coordinate_profile,
)
from teleop_runtime.sources.pose import VivePoseSource, ViveTrackerConfig


def _pose(x: float, y: float, z: float) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [x, y, z]
    return pose


def _channel(name: str, pose: np.ndarray, sequence: int) -> SimpleNamespace:
    return SimpleNamespace(
        channel_name=name,
        frame_name="vive_world",
        value=pose,
        valid=True,
        critical=True,
        timestamp=1000.0 + sequence,
        metadata={"seq": sequence, "receiver_timestamp": 1000.0 + sequence},
    )


class _WorkflowUi:
    def __init__(self) -> None:
        self.state: dict[str, object] = {}
        self._started_stages: set[str] = set()

    def append_message(self, _tag: str, _message: str) -> None:
        return

    def publish(self, **updates: object) -> None:
        self.state.update(updates)

    def consume_events(self) -> list[str]:
        stage = str(self.state.get("capture_stage", ""))
        if stage and stage not in self._started_stages:
            self._started_stages.add(stage)
            return ["n"]
        return []


class _ScriptedSource:
    def __init__(self, ui: _WorkflowUi) -> None:
        self.ui = ui
        self.sequence = 0
        self._counts: dict[str, int] = {}

    def get_latest_channels(self) -> dict[str, object]:
        stage = str(self.ui.state.get("capture_stage", "VirtualDeskX"))
        index = self._counts.get(stage, 0)
        self._counts[stage] = index + 1
        trajectories = {
            "VirtualDeskX": [_pose(0.0, 0.0, 0.0), _pose(0.1, 0.0, 0.0), _pose(0.2, 0.0, 0.0)],
            "VirtualDeskY": [_pose(0.0, 0.0, 0.0), _pose(0.0, 0.1, 0.0), _pose(0.0, 0.2, 0.0)],
            "VirtualDeskZ": [_pose(0.0, 0.0, 0.0), _pose(0.0, 0.0, 0.1), _pose(0.0, 0.0, 0.2)],
            "WristOffset": [_pose(0.0, 0.0, 0.0)],
        }
        trajectory = trajectories.get(stage, trajectories["VirtualDeskX"])
        pose = trajectory[min(index, len(trajectory) - 1)]
        self.sequence += 1
        return {"right_wrist_pose": _channel("right_wrist_pose", pose, self.sequence)}

    def debug_stats(self) -> dict[str, object]:
        return {}


def test_single_side_reuses_shared_workflow_and_saves_independent_profile(tmp_path: Path) -> None:
    ui = _WorkflowUi()
    source = _ScriptedSource(ui)
    calibration = new_coordinate_calibration(active_sides=("right",))
    args = SimpleNamespace(
        hz=1000.0,
        virtual_desk_tracker_role="right_wrist",
        aux_tracker_name="",
        capture_prep_sec=0.0,
        wrist_capture_prep_sec=0.0,
        capture_extra_wait_sec=0.01,
        sweep_capture_sec=0.01,
        hold_capture_sec=0.01,
        min_sweep_samples=3,
        min_hold_samples=3,
        min_sweep_extent_m=0.05,
        min_axis_variance_ratio=0.5,
        min_start_end_distance_m=0.02,
        max_axis_non_orthogonality_deg=20.0,
        virtual_desk_post_yaw_deg=180.0,
    )
    result = calibrate_vive_coordinate_profile(
        args,
        ViveCoordinateCalibrationContext(source, calibration, ("right",)),
        ui,
        save_profile=None,
        save_on_complete=False,
    )
    assert result is None
    calibration.metadata["profile_ready"] = True
    saved_path = save_coordinate_profile(calibration, root=tmp_path, profile_name="single_right")
    loaded, loaded_path = load_coordinate_profile("single_right", root=tmp_path)
    assert loaded_path == saved_path
    assert loaded.metadata["profile_kind"] == VIVE_COORDINATE_PROFILE_KIND
    assert loaded.metadata["active_sides"] == ["right"]
    assert sorted(loaded.tracker_offsets) == ["right_wrist"]


def test_real_teleop_profile_marker_is_rejected(tmp_path: Path) -> None:
    real_like = TeleopCalibration()
    real_like.set_world_to_desk(np.eye(4))
    real_like.set_tracker_offset("right_wrist", np.eye(4))
    with pytest.raises(ValueError, match="not a Control Station"):
        validate_coordinate_profile(real_like)
    assert not list(tmp_path.glob("*.json"))


def test_coordinate_profile_view_has_no_target_or_home_state() -> None:
    calibration = new_coordinate_calibration(active_sides=("left",))
    calibration.set_world_to_desk(np.eye(4))
    calibration.set_tracker_offset("left_wrist", np.eye(4))
    calibration.metadata["profile_ready"] = True
    view = coordinate_profile_view(calibration)
    assert view["active_sides"] == ["left"]
    assert view["target_binding_policy"] == "target_profile_supplies_desk_to_base"
    assert view["home_policy"] == "session_home_is_not_part_of_coordinate_profile"
    assert not calibration.captured_home_qpos


def test_default_profile_directory_is_not_real_teleop_directory() -> None:
    assert control_station_vive_coordinate_profiles_dir() != calibration_profiles_dir()
    assert control_station_vive_coordinate_profiles_dir().name == "control_station_vive_coordinate_profiles"


def test_profile_store_rejects_real_teleop_profile_tree() -> None:
    calibration = new_coordinate_calibration(active_sides=("right",))
    calibration.set_world_to_desk(np.eye(4))
    calibration.set_tracker_offset("right_wrist", np.eye(4))
    with pytest.raises(ValueError, match="outside the real-teleop"):
        save_coordinate_profile(calibration, root=calibration_profiles_dir())


def test_gui_is_read_only_with_respect_to_motion_and_exposes_state(tmp_path: Path) -> None:
    source = VivePoseSource(
        [
            ViveTrackerConfig(
                name="tracker_1",
                side="left",
                role="left_wrist",
                channel_name="left_wrist_pose",
            )
        ],
        port=0,
        clock_sync_port=0,
    )
    controller = ViveCoordinateCalibrationController(
        ViveCoordinateCalibrationConfig(active_sides=("left",), profile_dir=tmp_path),
        source=source,
    )
    gui = ViveCoordinateCalibrationGui(controller, host="127.0.0.1", port=0)
    controller.start(start_source=False)
    gui.start()
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(gui.url, timeout=2.0) as response:
            page = response.read().decode("utf-8")
            content_security_policy = response.headers["Content-Security-Policy"]
        assert 'class="live-grid"' in page
        assert 'id="worldMatrix"' in page
        assert 'data-step="virtual_desk_x"' in page
        assert 'nonce="' in page
        assert "script-src 'self' 'nonce-" in content_security_policy
        assert "style-src 'self' 'nonce-" in content_security_policy
        assert 'JSON.stringify(state.live_channels' not in page
        with opener.open(gui.url + "api/state", timeout=2.0) as response:
            state = json.loads(response.read().decode("utf-8"))
        assert state["motion_authorized"] is False
        assert state["real_arm_motion"] is False
        assert state["no_dispatch_owner"] is True

        request = urllib.request.Request(
            gui.url + "api/event",
            data=json.dumps({"event": "begin_calibration"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with opener.open(request, timeout=2.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["accepted"] is True
        for _ in range(20):
            if controller.snapshot()["stage"] == "virtual_desk_x":
                break
            time.sleep(0.01)
        assert controller.snapshot()["stage"] == "virtual_desk_x"
    finally:
        gui.close()


def test_vive_calibration_gui_is_loopback_only(tmp_path: Path) -> None:
    controller = ViveCoordinateCalibrationController(
        ViveCoordinateCalibrationConfig(profile_dir=tmp_path)
    )
    try:
        with pytest.raises(ValueError, match="loopback"):
            ViveCoordinateCalibrationGui(controller, host="0.0.0.0")
    finally:
        controller.close()
