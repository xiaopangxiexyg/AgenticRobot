from __future__ import annotations

import json
import shutil
import socket
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request, urlopen

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from control_composition.observability import (
    project_pose_matrix_deflection,
    projectable_pose_matrix,
)
from control_station import vive_calibration_profiles
from control_station.trajectory_guidance_webgl import (
    TRAJECTORY_GUIDANCE_WEBGL_JS,
)
from control_station.vive_calibration_profiles import (
    new_coordinate_calibration,
    save_coordinate_profile,
)
from control_station.vive_deflection import (
    VIVE_DEFLECTION_STATUS_SCHEMA_V1,
    ViveDeflectionConfig,
    ViveDeflectionController,
    _build_vive_source,
    _load_coordinate_profile_for_source,
)
from control_station.vive_deflection_calibration import (
    ViveDeflectionCalibrationLauncher,
)
from control_station.vive_deflection_cli import main
from control_station.vive_deflection_gui import (
    _HTML,
    ViveDeflectionGui,
    _Handler,
    _Server,
)


def _pose(xyz=(0.0, 0.0, 0.0), *, rotation_deg=0.0) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = Rotation.from_euler(
        "z", rotation_deg, degrees=True
    ).as_matrix()
    pose[:3, 3] = xyz
    return pose


def _raw_config(**kwargs) -> ViveDeflectionConfig:
    return ViveDeflectionConfig(coordinate_profile_path=None, **kwargs)


class _FakeSource:
    def __init__(self, pose: np.ndarray, *, timestamp: float = 10.0) -> None:
        self.pose = pose
        self.timestamp = timestamp
        self.valid = True
        self.seq = 1
        self.latest_channel_calls = 0
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True

    def get_latest_channels(self):
        self.latest_channel_calls += 1
        return {
            "wrist_pose": SimpleNamespace(
                value=self.pose.copy(),
                valid=self.valid,
                timestamp=self.timestamp,
                metadata={
                    "receiver_timestamp": self.timestamp,
                    "seq": self.seq,
                },
            )
        }

    def debug_stats(self):
        return {
            "packet_count": self.seq,
            "datagram_count": self.seq,
            "source_mapping_generation": 0,
            "last_datagram_received_monotonic_time": 19.98,
            "last_packet_received_monotonic_time": 19.97,
            "last_process_finished_monotonic_time": 19.975,
            "last_process_duration_sec": 0.0015,
            "last_process_batch_size": 1,
            "last_drain_count": 0,
            "worker_alive": True,
        }


class _FakeCalibrationLauncher:
    def __init__(self) -> None:
        self.open_calls = 0

    def snapshot(self):
        return {
            "available": True,
            "active": self.open_calls > 0,
            "url": "http://127.0.0.1:45678/" if self.open_calls else None,
            "source_role": "left_wrist",
            "shared_source": True,
            "last_saved_profile_path": None,
        }

    def open(self):
        self.open_calls += 1
        return self.snapshot()

def test_standalone_controller_projects_each_source_packet_only_once() -> None:
    source = _FakeSource(_pose(), timestamp=10.0)
    controller = ViveDeflectionController(
        _raw_config(freshness_sec=0.5),
        source=source,
        clock=lambda: 10.1,
        monotonic_clock=lambda: 20.0,
    )

    controller.snapshot()
    controller.snapshot()
    controller.snapshot()

    assert source.latest_channel_calls == 1
    source.seq = 2
    source.pose = _pose((0.04, 0.0, 0.0))
    moved = controller.snapshot()["deflection"]["components"][0]
    assert source.latest_channel_calls == 2
    assert moved["translation_xyz_m"] == pytest.approx([0.04, 0.0, 0.0])
    assert moved["translation_reference_frame"] == "vive_world"
    assert "translation_axis_semantics" not in moved


def test_standalone_controller_reports_segmented_passive_latency() -> None:
    source = _FakeSource(_pose(), timestamp=10.0)
    monotonic_times = iter([20.0, 20.004])
    controller = ViveDeflectionController(
        _raw_config(freshness_sec=0.5),
        source=source,
        clock=lambda: 10.1,
        monotonic_clock=lambda: next(monotonic_times),
    )

    state = controller.snapshot()

    assert state["snapshot_duration_sec"] == pytest.approx(0.004)
    assert state["source"]["transport_datagram_age_sec"] == pytest.approx(0.024)
    assert state["source"]["processed_packet_age_sec"] == pytest.approx(0.034)
    assert state["source"]["last_process_age_sec"] == pytest.approx(0.029)
    assert state["source"]["last_process_duration_sec"] == pytest.approx(0.0015)
    assert state["source"]["last_process_batch_size"] == 1
    assert state["source"]["worker_alive"] is True


def test_calibrated_view_uses_retarget_front_left_up_axes() -> None:
    calibration = new_coordinate_calibration(active_sides=("left",))
    calibration.set_world_to_vive_retarget(_pose(rotation_deg=90.0))
    calibration.set_tracker_offset("left_wrist", np.eye(4, dtype=np.float64))
    source = _FakeSource(_pose(), timestamp=10.0)
    controller = ViveDeflectionController(
        _raw_config(freshness_sec=0.5),
        source=source,
        coordinate_calibration=calibration,
        clock=lambda: 10.1,
    )

    controller.snapshot()
    source.pose = _pose((0.0, 0.04, 0.0))
    source.seq = 2
    state = controller.snapshot()
    moved = state["deflection"]["components"][0]

    assert moved["translation_xyz_m"] == pytest.approx([0.04, 0.0, 0.0])
    assert moved["translation_reference_frame"] == "vive_retarget_frame"
    assert moved["translation_axis_semantics"] == {
        "positive_x": "forward",
        "negative_x": "backward",
        "positive_y": "left",
        "negative_y": "right",
        "positive_z": "up",
        "negative_z": "down",
    }
    assert state["source"]["coordinate_calibrated"] is True


def test_explicit_180_post_yaw_labels_physical_signed_axes() -> None:
    calibration = new_coordinate_calibration(active_sides=("left",))
    calibration.set_world_to_vive_retarget(
        np.diag([-1.0, -1.0, 1.0, 1.0])
    )
    calibration.set_tracker_offset("left_wrist", np.eye(4, dtype=np.float64))
    calibration.metadata.update(
        {
            "virtual_desk_sweep_convention_version": 2,
            "virtual_desk_post_yaw_deg": 180.0,
        }
    )
    source = _FakeSource(_pose(), timestamp=10.0)
    controller = ViveDeflectionController(
        _raw_config(freshness_sec=0.5),
        source=source,
        coordinate_calibration=calibration,
        clock=lambda: 10.1,
    )

    controller.snapshot()
    source.pose = _pose((0.04, 0.02, 0.01))
    source.seq = 2
    state = controller.snapshot()
    moved = state["deflection"]["components"][0]

    assert moved["translation_xyz_m"] == pytest.approx([-0.04, -0.02, 0.01])
    assert moved["translation_axis_semantics"] == {
        "positive_x": "backward",
        "negative_x": "forward",
        "positive_y": "right",
        "negative_y": "left",
        "positive_z": "up",
        "negative_z": "down",
    }
    assert state["source"]["axis_semantics_status"] == "explicit_post_yaw"
    assert "180" in state["source"]["axis_semantics_warning"]


def test_passive_viewer_source_disables_unused_motion_history() -> None:
    source = _build_vive_source(_raw_config())

    assert source.history_window_sec == 0.0
    assert source.motion_window_sec == 0.0


def test_latest_coordinate_profile_is_selected_for_requested_source_side(
    monkeypatch,
) -> None:
    summaries = (
        SimpleNamespace(
            valid=True,
            active_sides=("right",),
            path="newest_right.json",
        ),
        SimpleNamespace(
            valid=True,
            active_sides=("left", "right"),
            path="older_both.json",
        ),
    )
    selected = object()
    monkeypatch.setattr(
        vive_calibration_profiles,
        "list_coordinate_profiles",
        lambda: summaries,
    )
    monkeypatch.setattr(
        vive_calibration_profiles,
        "load_coordinate_profile",
        lambda path: (selected, Path(path)),
    )

    calibration, path = _load_coordinate_profile_for_source(
        "latest",
        source_role="left_wrist",
    )

    assert calibration is selected
    assert path == Path("older_both.json")


def test_saved_coordinate_profile_can_be_applied_and_resets_viewer_home(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        vive_calibration_profiles,
        "control_station_vive_coordinate_profiles_dir",
        lambda: tmp_path,
    )
    calibration = new_coordinate_calibration(active_sides=("left",))
    calibration.set_world_to_vive_retarget(_pose(rotation_deg=90.0))
    calibration.set_tracker_offset("left_wrist", np.eye(4, dtype=np.float64))
    saved = save_coordinate_profile(
        calibration,
        root=tmp_path,
        profile_name="viewer_recalibrated",
    )
    source = _FakeSource(_pose(), timestamp=10.0)
    controller = ViveDeflectionController(
        _raw_config(freshness_sec=0.5),
        source=source,
        clock=lambda: 10.1,
    )
    assert controller.snapshot()["deflection"]["anchor_request_generation"] == 1

    receipt = controller.apply_coordinate_profile(saved)
    centered = controller.snapshot()
    source.pose = _pose((0.0, 0.04, 0.0))
    source.seq = 2
    moved = controller.snapshot()["deflection"]["components"][0]

    assert receipt["viewer_home_reset"] is True
    assert centered["deflection"]["anchor_request_generation"] == 2
    assert centered["deflection"]["components"][0]["translation_distance_m"] == 0
    assert moved["translation_xyz_m"] == pytest.approx([0.04, 0.0, 0.0])
    assert centered["source"]["coordinate_profile_path"] == str(saved)


def test_trajectory_guidance_modes_execute_distinct_state_semantics() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required for the browser guidance semantic probe")
    probe = TRAJECTORY_GUIDANCE_WEBGL_JS + r"""
const flippedBasis = trajectorySemanticBasis({
  positive_x: 'backward', negative_x: 'forward',
  positive_y: 'right', negative_y: 'left',
  positive_z: 'up', negative_z: 'down',
});
const mappedNominal = trajectoryMapPhysicalVector(
  flippedBasis, [0.2, 0.1, 0.05]
);
if (Math.abs(mappedNominal[0] + 0.2) > 1e-12
    || Math.abs(mappedNominal[1] + 0.1) > 1e-12
    || Math.abs(mappedNominal[2] - 0.05) > 1e-12) {
  throw new Error(`physical-basis mapping mismatch: ${mappedNominal}`);
}
const scene = Object.create(TrajectoryGuidanceScene.prototype);
Object.assign(scene, {
  deflectionM: [0.12, 0, 0],
  fullScaleM: 0.12,
  deadzoneM: 0.005,
  maxGuidanceSpeedMps: 0.08,
  positionGain: 1.0,
  mode: 'velocity_joystick',
  integratedOffsetM: [0, 0, 0],
  lastIntegrationAt: null,
});
scene._integrateJoystick(0);
for (let index = 1; index <= 60; index += 1) {
  scene._integrateJoystick(index * 1000 / 60);
}
if (Math.abs(scene.integratedOffsetM[0] - 0.08) > 1e-9) {
  throw new Error(`joystick integration mismatch: ${scene.integratedOffsetM[0]}`);
}
scene.setDeflectionMeters([0, 0, 0]);
for (let index = 61; index <= 120; index += 1) {
  scene._integrateJoystick(index * 1000 / 60);
}
if (Math.abs(scene.integratedOffsetM[0] - 0.08) > 1e-9) {
  throw new Error('returning Home must hold the accumulated offset');
}
scene.clearIntegratedOffset();
if (scene.integratedOffsetM.some(value => value !== 0)) {
  throw new Error('explicit clear must reset the joystick accumulator');
}
scene.setMode('position_offset');
scene.setPositionGain(2.0);
scene.setDeflectionMeters([0.03, -0.01, 0]);
const fixed = scene.correctionState();
if (Math.abs(fixed.offsetM[0] - 0.06) > 1e-12
    || Math.abs(fixed.offsetM[1] + 0.02) > 1e-12
    || fixed.futureVelocityMps.some(value => value !== 0)) {
  throw new Error(`position-offset semantics mismatch: ${JSON.stringify(fixed)}`);
}
"""
    subprocess.run(
        [node, "-e", probe],
        check=True,
        capture_output=True,
        text=True,
    )


def test_pose_projection_uses_world_translation_and_home_local_rotation() -> None:
    result = project_pose_matrix_deflection(
        _pose(rotation_deg=30.0),
        _pose((0.03, -0.04, 0.12), rotation_deg=120.0),
        component_id="left_wrist",
        translation_reference_frame="vive_world",
        rotation_controlled_frame="left_wrist",
    )

    assert result["translation_xyz_m"] == pytest.approx([0.03, -0.04, 0.12])
    assert result["translation_distance_m"] == pytest.approx(0.13)
    assert result["rotation_rotvec_rad"] == pytest.approx(
        [0.0, 0.0, np.pi / 2.0]
    )
    assert result["rotation_application"] == "home_local_right_multiply"


def test_display_projection_accepts_raw_vive_rotation_drift_without_changing_translation() -> None:
    raw = _pose((0.02, -0.03, 0.04), rotation_deg=20.0)
    raw[:3, :3] *= 1.002

    projected = projectable_pose_matrix(raw, name="raw_vive")

    assert projected[:3, 3] == pytest.approx(raw[:3, 3])
    assert projected[:3, :3].T @ projected[:3, :3] == pytest.approx(
        np.eye(3), abs=1e-10
    )
    assert np.linalg.det(projected[:3, :3]) == pytest.approx(1.0)


def test_standalone_controller_auto_homes_recenter_and_never_creates_control() -> None:
    source = _FakeSource(_pose(), timestamp=10.0)
    controller = ViveDeflectionController(
        _raw_config(freshness_sec=0.5),
        source=source,
        clock=lambda: 10.1,
    )
    controller.start()

    initial = controller.snapshot()
    assert initial["schema"] == VIVE_DEFLECTION_STATUS_SCHEMA_V1
    assert initial["deflection"]["components"][0]["translation_distance_m"] == 0.0
    assert initial["motion_authorized"] is False
    assert initial["simulator_created"] is False
    assert initial["policy_created"] is False
    assert initial["composition_created"] is False
    assert initial["source"]["recenter_ready"] is True
    assert initial["source"]["receiver_timestamp"] == 10.0

    source.pose = _pose((0.05, 0.0, 0.0), rotation_deg=45.0)
    source.seq = 2
    moved = controller.snapshot()["deflection"]["components"][0]
    assert moved["translation_xyz_m"] == pytest.approx([0.05, 0.0, 0.0])
    assert moved["rotation_angle_rad"] == pytest.approx(np.pi / 4.0)

    receipt = controller.recenter()
    assert receipt == {"status": "recentered", "home_generation": 2}
    centered = controller.snapshot()["deflection"]["components"][0]
    assert centered["translation_distance_m"] == 0.0
    assert centered["rotation_angle_rad"] == 0.0

    controller.close()
    assert source.started is True
    assert source.closed is True


def test_stale_pose_is_not_drawn_and_home_is_preserved() -> None:
    source = _FakeSource(_pose(), timestamp=10.0)
    now = [10.1]
    controller = ViveDeflectionController(
        _raw_config(freshness_sec=0.5),
        source=source,
        clock=lambda: now[0],
    )
    controller.snapshot()
    now[0] = 11.0

    stale = controller.snapshot()

    assert stale["deflection"]["status"] == "stale"
    assert stale["deflection"]["components"] == []
    assert stale["deflection"]["anchor_request_generation"] == 1


def test_invalid_channel_disables_recenter_and_reports_specific_reason() -> None:
    source = _FakeSource(_pose(), timestamp=10.0)
    source.valid = False
    controller = ViveDeflectionController(
        _raw_config(freshness_sec=0.5),
        source=source,
        clock=lambda: 10.1,
    )

    status = controller.snapshot()

    assert status["source"]["recenter_ready"] is False
    assert status["source"]["channel_valid"] is False
    assert status["deflection"]["status"] == "invalid"
    with pytest.raises(RuntimeError, match="channel_invalid"):
        controller.recenter()


def test_cli_dry_run_and_page_are_explicitly_independent(capsys) -> None:
    assert main(["--dry-run", "--source-role", "right_wrist"]) == 0
    output = capsys.readouterr().out
    assert '"device_start_skipped": true' in output
    assert '"motion_authorized": false' in output
    assert '"simulator_created": false' in output
    assert '"recalibration_available": true' in output
    assert "VIVE 偏转查看器" in _HTML
    assert "renderRelativeDeflection" in _HTML
    assert "PoseDeflectionScene" in _HTML
    assert "TrajectoryGuidanceScene" in _HTML
    assert 'id="trajectory-guidance-panel"' in _HTML
    assert "offset(t+dt) = offset(t) + v × dt" in _HTML
    assert "p修正(t,τ) = p原始(t,τ) + offset(t) + τ × v" in _HTML
    assert "p修正(t,τ) = p原始(t,τ) + Δp修正" in _HTML
    assert '<option value="velocity_joystick" selected>' in _HTML
    assert '<option value="position_offset">' in _HTML
    assert 'id="trajectory-guidance-clear"' in _HTML
    assert 'id="axis-warning"' in _HTML
    assert "source.axis_semantics_warning" in _HTML
    assert "clearIntegratedOffset" in _HTML
    assert "response.ok && trajectoryGuidanceScene" in _HTML
    assert "网格间距 0.05 m" in _HTML
    assert "demo-only、无控制权" in _HTML
    assert "updateTrajectoryGuidance(value.deflection || {}, sampleToken)" in _HTML
    assert "fullScaleM = 0.12" in _HTML
    assert "horizonSec = 2.0" in _HTML
    assert "bouncePeriodSec = 2.4" in _HTML
    assert "framePeriodMs = 1000 / 30" in _HTML
    assert "this.trajectoryPointCount = 65" in _HTML
    assert "this.yaw = -0.78" in _HTML
    assert "this.pitch = 0.55" in _HTML
    assert "_animate(timestamp)" in _HTML
    assert "attribute float aTau" in _HTML
    assert "uPhaseSec+aTau" in _HTML
    assert "uBouncePeriodSec" in _HTML
    assert "uCorrectionOffset+aTau*uGuidanceVelocity" in _HTML
    assert "this.mode = 'velocity_joystick'" in _HTML
    assert "this.mode === 'position_offset'" in _HTML
    assert "this.deadzoneM = 0.005" in _HTML
    assert "item + velocity[index] * dtSec" in _HTML
    assert "gl.DYNAMIC_DRAW" not in TRAJECTORY_GUIDANCE_WEBGL_JS
    assert "轨迹刷新 30 Hz" in _HTML
    assert "+X 前 / +Y 左 / +Z 上" in _HTML
    assert "forward: '前'" in _HTML
    assert "backward: '后'" in _HTML
    assert "left: '左'" in _HTML
    assert "right: '右'" in _HTML
    assert "up: '上'" in _HTML
    assert "down: '下'" in _HTML
    assert 'data-axis-name="+X"' in _HTML
    assert 'data-axis-name="−Z"' in _HTML
    assert "canvas.getContext('webgl'" in _HTML
    assert "真正的三维视图" in _HTML
    assert "VIEWER_REFRESH_PERIOD_MS = 1000 / 60" in _HTML
    assert "refreshInFlight" in _HTML
    assert "requestAnimationFrame(applyLatestViewerState)" in _HTML
    assert "lastSampleToken" in _HTML
    assert "window.setInterval(refresh, 100)" not in _HTML
    assert "svgNode" not in _HTML
    assert "XY 投影" not in _HTML
    assert "不创建 sim、policy、HITL composition" in _HTML
    assert 'id="recenter"' in _HTML
    assert 'id="recalibrate"' in _HTML
    assert "/api/calibration/open" in _HTML
    assert "source.recenter_ready" in _HTML
    assert _Handler.protocol_version == "HTTP/1.1"


def test_loopback_server_disables_nagle_for_small_live_state_responses() -> None:
    calls: list[tuple[int, int, int]] = []
    request = SimpleNamespace(
        setsockopt=lambda level, option, value: calls.append(
            (level, option, value)
        )
    )

    _Server._configure_low_latency_request_socket(request)

    assert calls == [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)]


def test_loopback_page_serves_ball_state_and_recenter() -> None:
    source = _FakeSource(_pose(), timestamp=10.0)
    calibration_launcher = _FakeCalibrationLauncher()
    controller = ViveDeflectionController(
        _raw_config(freshness_sec=0.5),
        source=source,
        clock=lambda: 10.1,
    )
    gui = ViveDeflectionGui(
        controller,
        port=0,
        open_browser=False,
        calibration_launcher=calibration_launcher,
    )
    try:
        gui.start()
        with urlopen(gui.url, timeout=2.0) as response:
            page = response.read().decode("utf-8")
        assert response.status == 200
        assert "VIVE 偏转查看器" in page
        assert 'id="human-deflection-views"' in page
        assert "PoseDeflectionScene" in page
        assert "TrajectoryGuidanceScene" in page
        assert 'id="trajectory-guidance-scene"' in page
        assert "满量程修正速度 [m/s]" in page
        assert "deflection-scene" in page
        assert "pointerdown" in page
        assert "wheel" in page
        assert "VIEWER_REFRESH_PERIOD_MS = 1000 / 60" in page
        assert "`总呈现 ${ms(presentationAgeMs)}`" in page
        assert "`socket ${ms(transportAgeMs)}`" in page
        assert "`UI等待 ${ms(applyQueueMs)}`" in page
        assert "window.setInterval(refresh, 100)" not in page
        with urlopen(f"{gui.url}api/state", timeout=2.0) as response:
            state = response.read().decode("utf-8")
        assert '"composition_created": false' in state
        assert '"shared_source": true' in state
        request = Request(f"{gui.url}api/recenter", data=b"", method="POST")
        with urlopen(request, timeout=2.0) as response:
            receipt = response.read().decode("utf-8")
        assert '"status": "recentered"' in receipt
        request = Request(
            f"{gui.url}api/calibration/open",
            data=b"",
            method="POST",
        )
        with urlopen(request, timeout=2.0) as response:
            calibration = response.read().decode("utf-8")
        assert '"active": true' in calibration
        assert calibration_launcher.open_calls == 1
    finally:
        gui.close()


def test_recalibration_launcher_reuses_viewer_source_and_profile_store(
    tmp_path: Path,
) -> None:
    source = _FakeSource(_pose(), timestamp=time.time())
    applied: list[Path] = []
    launcher = ViveDeflectionCalibrationLauncher(
        source=source,
        source_role="left_wrist",
        tracker_name="tracker_1",
        ui_port=0,
        profile_dir=tmp_path,
        profile_applier=lambda path: (
            applied.append(Path(path))
            or {"status": "coordinate_profile_applied"}
        ),
    )
    try:
        state = launcher.open()
        assert state["active"] is True
        assert state["shared_source"] is True
        assert state["profile_dir"] == str(tmp_path.resolve())
        assert source.started is False
        with urlopen(state["url"], timeout=2.0) as response:
            page = response.read().decode("utf-8")
        assert "开始 / 重新标定" in page
        with urlopen(f"{state['url']}api/state", timeout=2.0) as response:
            calibration_state = json.loads(response.read().decode("utf-8"))
        assert calibration_state["source_started"] is False
        assert calibration_state["source_lifecycle_owned"] is False
        assert calibration_state["source_receiver_active"] is True
        assert calibration_state["active_sides"] == ["left"]
        assert calibration_state["available_active_sides"] == ["left"]
        assert calibration_state["input_preflight"]["ready"] is True
        live_channel = calibration_state["live_channels"]["left_wrist_pose"]
        assert live_channel["channel_name"] == "left_wrist_pose"
        controller = launcher._controller
        calibration_gui = launcher._gui
        assert controller is not None
        assert calibration_gui is not None
        assert controller.config.virtual_desk_post_yaw_deg == 0.0
        saved = tmp_path / "saved_profile.json"
        launcher._apply_saved_profile(controller, calibration_gui, saved)
        assert applied == [saved.resolve()]
        assert launcher.snapshot()["last_applied_profile_path"] == str(
            saved.resolve()
        )
    finally:
        launcher.close()
    assert source.closed is False


@pytest.mark.integration
def test_recalibration_reads_live_udp_from_the_viewer_owned_source(
    tmp_path: Path,
    require_udp_loopback: None,
) -> None:
    controller = ViveDeflectionController(
        _raw_config(
            vive_host="127.0.0.1",
            vive_port=0,
            vive_clock_sync_port=0,
        )
    )
    launcher = ViveDeflectionCalibrationLauncher(
        source=controller.source,
        source_role="left_wrist",
        tracker_name="tracker_1",
        ui_host="127.0.0.1",
        ui_port=0,
        profile_dir=tmp_path,
    )
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    controller.start()
    try:
        assert controller.source.sock is not None
        address = controller.source.sock.getsockname()

        def send_pose(sequence: int) -> None:
            packet = {
                "protocol": "teleop_runtime.vive/v1",
                "seq": sequence,
                "sender_timestamp": time.time(),
                "trackers": [
                    {
                        "tracker_name": "tracker_1",
                        "tracker_role": "left_wrist",
                        "valid": True,
                        "pose_matrix_4x4": _pose((0.01 * sequence, 0.0, 0.0)).tolist(),
                        "error": None,
                    }
                ],
            }
            sender.sendto(json.dumps(packet).encode("utf-8"), address)

        for sequence in range(1, 4):
            send_pose(sequence)
            time.sleep(0.01)

        deadline = time.time() + 1.0
        while time.time() < deadline:
            if controller.source.debug_stats()["packet_count"] >= 3:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("viewer-owned VIVE receiver did not read UDP packets")

        launcher.open()
        calibration_controller = launcher._controller
        assert calibration_controller is not None
        state = calibration_controller.snapshot()

        assert state["source_started"] is False
        assert state["source_lifecycle_owned"] is False
        assert state["source_receiver_active"] is True
        assert state["input_preflight"]["ready"] is True
        assert state["live_channels"]["left_wrist_pose"]["channel_name"] == (
            "left_wrist_pose"
        )
        assert state["source_stats"]["channel_update_counts"] == {
            "left_wrist_pose": 3
        }

        # Exercise the actual calibration sampler, not just its idle
        # preflight.  Fresh UDP packets must advance the aliased channel's
        # sample identity while the viewer remains the receiver owner.
        calibration_controller.config.capture_prep_sec = 0.0
        calibration_controller.config.sweep_capture_sec = 1.0
        calibration_controller.config.min_sweep_samples = 3
        calibration_controller.dispatch("begin_calibration")
        calibration_controller.dispatch("capture")
        for sequence in range(4, 16):
            send_pose(sequence)
            time.sleep(0.025)

        deadline = time.time() + 1.0
        while time.time() < deadline:
            state = calibration_controller.snapshot()
            samples = state["capture_samples"].get("left_wrist_pose", 0)
            if samples >= 3:
                break
            time.sleep(0.01)
        else:
            raise AssertionError(
                "shared-source calibration sampler did not receive fresh UDP poses"
            )
        calibration_controller.dispatch("cancel_calibration")
    finally:
        launcher.close()
        assert controller.source.debug_stats()["worker_alive"] is True
        controller.close()
        sender.close()
