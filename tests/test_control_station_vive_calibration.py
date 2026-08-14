from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from control_station.sim_hil_profile.status import _vive_receiver_status
from control_station.vive_calibration import (
    ViveCoordinateCalibrationConfig,
    ViveCoordinateCalibrationController,
)
from control_station.vive_calibration_gui import ViveCoordinateCalibrationGui
from control_station.vive_calibration_profiles import (
    LATEST_VIVE_COORDINATE_PROFILE,
    VIVE_COORDINATE_PROFILE_KIND,
    coordinate_profile_view,
    derive_coordinate_calibration,
    load_coordinate_profile,
    new_coordinate_calibration,
    save_coordinate_profile,
    validate_coordinate_profile,
)
from project_paths import (
    calibration_profiles_dir,
    control_station_vive_coordinate_profiles_dir,
)
from teleop_runtime.calibration import ArmCalibration, TeleopCalibration
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
    received_at = time.time()
    return SimpleNamespace(
        channel_name=name,
        frame_name="vive_world",
        value=pose,
        valid=True,
        critical=True,
        timestamp=received_at,
        metadata={"seq": sequence, "receiver_timestamp": received_at},
    )


def _vive_role_packet(
    *,
    sequence: int = 1,
    right_valid: bool,
) -> bytes:
    return json.dumps(
        {
            "protocol": "teleop_runtime.vive/v1",
            "seq": sequence,
            "sender_timestamp": time.time(),
            "trackers": [
                {
                    "tracker_name": "physical-single",
                    "tracker_role": "left_wrist",
                    "valid": True,
                    "pose_matrix_4x4": np.eye(4).tolist(),
                    "error": None,
                },
                {
                    "tracker_name": (
                        "physical-right"
                        if right_valid
                        else "<right_wrist_unavailable>"
                    ),
                    "tracker_role": "right_wrist",
                    "valid": right_valid,
                    "pose_matrix_4x4": np.eye(4).tolist(),
                    "error": (
                        None
                        if right_valid
                        else "single_tracker_mode:not_connected"
                    ),
                },
            ],
        }
    ).encode("utf-8")


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


class _LiveControllerSource:
    def __init__(self, available_channels: tuple[str, ...]) -> None:
        self.available_channels = available_channels
        self.sequence = 0

    def get_latest_channels(self) -> dict[str, object]:
        self.sequence += 1
        return {
            name: _channel(name, _pose(0.1, 0.2, 0.3), self.sequence)
            for name in self.available_channels
        }

    def debug_stats(self) -> dict[str, object]:
        return {"packet_count": self.sequence}

    @property
    def source_role_by_output_channel(self) -> dict[str, str]:
        return {
            name: name.removesuffix("_pose")
            for name in self.available_channels
        }


class _FixedControllerSource:
    def __init__(self, channels: dict[str, object]) -> None:
        self.channels = channels

    def get_latest_channels(self) -> dict[str, object]:
        return dict(self.channels)

    def debug_stats(self) -> dict[str, object]:
        return {}


class _RecoveringControllerSource(_LiveControllerSource):
    def __init__(self) -> None:
        super().__init__(("right_wrist_pose",))
        self.start_calls = 0
        self.closed = False

    def start(self) -> None:
        self.start_calls += 1
        if self.start_calls == 1:
            raise RuntimeError("synthetic receiver start failure")

    def close(self) -> None:
        self.closed = True


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
        virtual_desk_post_yaw_deg=0.0,
    )
    result = calibrate_vive_coordinate_profile(
        args,
        ViveCoordinateCalibrationContext(source, calibration, ("right",)),
        ui,
        save_profile=None,
        save_on_complete=False,
    )
    assert result is None
    np.testing.assert_allclose(
        calibration.world_to_desk_pose()[:3, :3],
        np.eye(3),
        atol=1e-9,
    )
    assert calibration.metadata["virtual_desk_post_yaw_deg"] == 0.0
    calibration.metadata["profile_ready"] = True
    saved_path = save_coordinate_profile(calibration, root=tmp_path, profile_name="single_right")
    loaded, loaded_path = load_coordinate_profile("single_right", root=tmp_path)
    assert loaded_path == saved_path
    assert loaded.metadata["profile_kind"] == VIVE_COORDINATE_PROFILE_KIND
    assert loaded.metadata["active_sides"] == ["right"]
    assert sorted(loaded.tracker_offsets) == ["right_wrist"]


def test_new_station_signed_sweep_defaults_to_no_xy_flip() -> None:
    config = ViveCoordinateCalibrationConfig(active_sides=("left",))

    assert config.virtual_desk_post_yaw_deg == 0.0


def test_station_snapshot_exposes_signed_sweep_post_yaw(tmp_path: Path) -> None:
    controller = ViveCoordinateCalibrationController(
        ViveCoordinateCalibrationConfig(
            active_sides=("left",),
            profile_dir=tmp_path,
        ),
        source=_LiveControllerSource(("left_wrist_pose",)),
    )
    try:
        assert controller.snapshot()["virtual_desk_post_yaw_deg"] == 0.0
    finally:
        controller.close()


def test_latest_profile_selects_newest_valid_independent_record(
    tmp_path: Path,
) -> None:
    calibration = new_coordinate_calibration(active_sides=("right",))
    calibration.set_world_to_desk(np.eye(4))
    calibration.set_tracker_offset("right_wrist", np.eye(4))
    first = save_coordinate_profile(
        calibration,
        root=tmp_path,
        profile_name="first",
    )
    time.sleep(0.002)
    second = save_coordinate_profile(
        calibration,
        root=tmp_path,
        profile_name="second",
    )
    corrupt = tmp_path / "99999999_999999_corrupt.json"
    corrupt.write_text("{invalid", encoding="utf-8")

    loaded, resolved = load_coordinate_profile(
        LATEST_VIVE_COORDINATE_PROFILE,
        root=tmp_path,
    )

    assert first != second
    assert resolved == second
    assert loaded.metadata["profile_name"] == "second"


def test_existing_relative_profile_path_is_resolved_before_store_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_root = tmp_path / "profiles"
    calibration = new_coordinate_calibration(active_sides=("right",))
    calibration.set_world_to_desk(np.eye(4))
    calibration.set_tracker_offset("right_wrist", np.eye(4))
    saved = save_coordinate_profile(
        calibration,
        root=profile_root,
        profile_name="right_relative",
    )
    monkeypatch.chdir(tmp_path)

    loaded, resolved = load_coordinate_profile(
        saved.relative_to(tmp_path),
        root=profile_root,
    )

    assert resolved == saved
    assert loaded.metadata["active_sides"] == ["right"]


def test_existing_relative_profile_outside_store_remains_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_root = tmp_path / "profiles"
    outside_root = tmp_path / "outside"
    calibration = new_coordinate_calibration(active_sides=("right",))
    calibration.set_world_to_desk(np.eye(4))
    calibration.set_tracker_offset("right_wrist", np.eye(4))
    outside = save_coordinate_profile(
        calibration,
        root=outside_root,
        profile_name="outside",
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="stored directly"):
        load_coordinate_profile(
            outside.relative_to(tmp_path),
            root=profile_root,
        )


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


def test_controller_initial_latest_profile_is_loaded_for_setup_and_runtime(
    tmp_path: Path,
) -> None:
    calibration = new_coordinate_calibration(active_sides=("right",))
    calibration.set_world_to_desk(np.eye(4))
    calibration.set_tracker_offset("right_wrist", np.eye(4))
    saved = save_coordinate_profile(
        calibration,
        root=tmp_path,
        profile_name="restart_latest",
    )

    controller = ViveCoordinateCalibrationController(
        ViveCoordinateCalibrationConfig(
            active_sides=("right",),
            profile_dir=tmp_path,
            initial_profile=LATEST_VIVE_COORDINATE_PROFILE,
        ),
        source=_LiveControllerSource(("right_wrist_pose",)),
    )
    try:
        state = controller.snapshot()
        assert controller.current_profile_path == saved
        assert state["current_profile_path"] == str(saved)
        assert state["stage"] == "view"
        assert state["phase"] == "completed"
        assert state["profile"]["profile_name"] == "restart_latest"
    finally:
        controller.close()


def test_runtime_owned_single_side_topology_is_not_expanded_by_loaded_profile(
    tmp_path: Path,
) -> None:
    calibration = new_coordinate_calibration(
        active_sides=("left", "right"),
        desk_tracker="left_wrist",
    )
    calibration.set_world_to_desk(np.eye(4))
    calibration.set_tracker_offset("left_wrist", np.eye(4))
    calibration.set_tracker_offset("right_wrist", np.eye(4))
    calibration.metadata["profile_ready"] = True
    saved = save_coordinate_profile(
        calibration,
        root=tmp_path,
        profile_name="legacy_bilateral",
    )

    controller = ViveCoordinateCalibrationController(
        ViveCoordinateCalibrationConfig(
            active_sides=("right",),
            desk_tracker="auto",
            profile_dir=tmp_path,
            initial_profile=saved,
            adopt_profile_input_topology=False,
            vive_port=0,
            vive_clock_sync_port=0,
        )
    )
    try:
        source = controller.source
        source.swap_left_right_source_roles()
        source._process_received_batch(
            [
                (
                    _vive_role_packet(right_valid=False),
                    time.time(),
                    time.monotonic(),
                    ("192.0.2.10", 4000),
                )
            ]
        )

        state = controller.snapshot()
        assert state["profile"]["active_sides"] == ["left", "right"]
        assert state["active_sides"] == ["right"]
        assert state["desk_tracker"] == "auto"
        assert state["input_preflight"]["required_channels"] == [
            "right_wrist_pose"
        ]
        assert state["input_preflight"]["ready"] is True
        assert state["source_role_by_output_channel"] == {
            "right_wrist_pose": "left_wrist"
        }

        receiver = _vive_receiver_status(
            SimpleNamespace(sources=(source,)),
            required_output_channels=("right_wrist_pose",),
        )
        assert receiver["trackers"]["left"][
            "selected_for_required_output_channels"
        ] == ["right_wrist_pose"]
        assert receiver["trackers"]["right"][
            "selected_for_required_output_channels"
        ] == []
        assert receiver["trackers"]["right"]["sender_error"] == (
            "single_tracker_mode:not_connected"
        )
    finally:
        controller.close()


def test_standalone_controller_still_adopts_loaded_profile_topology(
    tmp_path: Path,
) -> None:
    calibration = new_coordinate_calibration(
        active_sides=("left", "right"),
        desk_tracker="left_wrist",
    )
    calibration.set_world_to_desk(np.eye(4))
    calibration.set_tracker_offset("left_wrist", np.eye(4))
    calibration.set_tracker_offset("right_wrist", np.eye(4))
    calibration.metadata["profile_ready"] = True
    saved = save_coordinate_profile(
        calibration,
        root=tmp_path,
        profile_name="standalone_bilateral",
    )

    controller = ViveCoordinateCalibrationController(
        ViveCoordinateCalibrationConfig(
            active_sides=("right",),
            profile_dir=tmp_path,
            initial_profile=saved,
            vive_port=0,
            vive_clock_sync_port=0,
        )
    )
    try:
        state = controller.snapshot()
        assert state["active_sides"] == ["left", "right"]
        assert state["desk_tracker"] == "left_wrist"
        assert controller.source.source_role_by_output_channel == {
            "left_wrist_pose": "left_wrist",
            "right_wrist_pose": "right_wrist",
        }
    finally:
        controller.close()


def test_controller_missing_initial_latest_preserves_first_calibration_flow(
    tmp_path: Path,
) -> None:
    controller = ViveCoordinateCalibrationController(
        ViveCoordinateCalibrationConfig(
            active_sides=("right",),
            profile_dir=tmp_path,
            initial_profile=LATEST_VIVE_COORDINATE_PROFILE,
        ),
        source=_LiveControllerSource(("right_wrist_pose",)),
    )
    try:
        state = controller.snapshot()
        assert controller.current_profile_path is None
        assert state["stage"] == "idle"
        assert "首次标定" in state["message"]
    finally:
        controller.close()


def test_controller_missing_explicit_initial_profile_is_not_silently_ignored(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        ViveCoordinateCalibrationController(
            ViveCoordinateCalibrationConfig(
                active_sides=("right",),
                profile_dir=tmp_path,
                initial_profile="missing-profile",
            ),
            source=_LiveControllerSource(("right_wrist_pose",)),
        )


def test_legacy_profile_migration_strips_target_and_home_state(tmp_path: Path) -> None:
    source_path = tmp_path / "legacy.json"
    source = TeleopCalibration(
        arms={
            side: ArmCalibration(
                name=f"{side}_arm",
                side=side,
                T_D_B=np.eye(4),
                home_qpos=np.ones(6),
                workspace_min=np.full(3, -1.0),
                workspace_max=np.full(3, 1.0),
            )
            for side in ("left", "right")
        },
        captured_home_qpos={
            f"{side}_arm": np.ones(6) for side in ("left", "right")
        },
        metadata={
            "home_ready": True,
            "home_capture": {"must_not_survive": True},
        },
    )
    source.set_world_to_vive_retarget(np.eye(4))
    source.set_tracker_offset("left_wrist", np.eye(4))
    source.set_tracker_offset("right_wrist", np.eye(4))
    source.save_json(source_path)

    derived = derive_coordinate_calibration(
        source,
        active_sides=("left", "right"),
        source_profile_path=source_path,
        profile_name="derived",
    )

    validate_coordinate_profile(derived)
    assert derived.metadata["home_policy"] == "session_home_is_not_part_of_coordinate_profile"
    assert "home_capture" not in derived.metadata
    assert derived.captured_home_qpos == {}
    for side in ("left", "right"):
        arm = derived.get_arm_for_side(side)
        assert arm.T_D_B is None
        assert arm.home_qpos is None
        assert arm.workspace_min is None
        assert arm.workspace_max is None


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
        assert 'id="cancel"' in page
        assert 'id="exit"' in page
        assert "X 后→前" in page
        assert "Y 右→左" in page
        assert "Z 下→上" in page
        assert 'nonce="' in page
        assert "script-src 'self' 'nonce-" in content_security_policy
        assert "style-src 'self' 'nonce-" in content_security_policy
        assert 'JSON.stringify(state.live_channels' not in page
        with opener.open(gui.url + "api/state", timeout=2.0) as response:
            state = json.loads(response.read().decode("utf-8"))
        assert state["motion_authorized"] is False
        assert state["real_arm_motion"] is False
        assert state["no_dispatch_owner"] is True
        assert state["input_preflight"]["ready"] is False
        assert state["workflow_active"] is False

        request = urllib.request.Request(
            gui.url + "api/event",
            data=json.dumps({"event": "begin_calibration"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            opener.open(request, timeout=2.0)
        payload = json.loads(exc_info.value.read().decode("utf-8"))
        assert payload["accepted"] is False
        assert "开始前检查未通过" in payload["error"]
        state = controller.snapshot()
        assert state["stage"] == "idle"
        assert state["phase"] == "blocked"
        assert state["workflow_active"] is False
    finally:
        gui.close()
        controller.close()


def test_begin_requires_every_selected_channel_before_starting(tmp_path: Path) -> None:
    controller = ViveCoordinateCalibrationController(
        ViveCoordinateCalibrationConfig(
            active_sides=("right",),
            profile_dir=tmp_path,
        ),
        source=_LiveControllerSource(("left_wrist_pose",)),
    )
    controller.start(start_source=False)
    try:
        state = controller.snapshot()
        assert state["input_preflight"]["required_channels"] == ["right_wrist_pose"]
        assert state["input_preflight"]["ready"] is False
        with pytest.raises(RuntimeError, match="right_wrist_pose"):
            controller.dispatch("begin_calibration")
        assert controller.snapshot()["workflow_active"] is False
    finally:
        controller.close()


def test_station_auto_sweep_uses_active_wrist_even_when_aux_is_configured(
    tmp_path: Path,
) -> None:
    controller = ViveCoordinateCalibrationController(
        ViveCoordinateCalibrationConfig(
            active_sides=("right",),
            desk_tracker="auto",
            auxiliary_tracker_name="tracker_aux",
            profile_dir=tmp_path,
        ),
        source=_LiveControllerSource(("right_wrist_pose",)),
    )
    try:
        state = controller.snapshot()
        assert state["input_preflight"]["required_channels"] == [
            "right_wrist_pose"
        ]
        assert state["input_preflight"]["ready"] is True
    finally:
        controller.close()


@pytest.mark.parametrize(
    ("channel", "reason"),
    [
        (
            SimpleNamespace(
                channel_name="right_wrist_pose",
                frame_name="vive_world",
                value=_pose(0.0, 0.0, 0.0),
                valid=True,
                critical=True,
                timestamp=1.0,
                metadata={"seq": 1, "receiver_timestamp": 1.0},
            ),
            "stale",
        ),
        (
            SimpleNamespace(
                channel_name="right_wrist_pose",
                frame_name="vive_world",
                value=np.full((4, 4), np.nan),
                valid=True,
                critical=True,
                timestamp=time.time(),
                metadata={"seq": 1, "receiver_timestamp": time.time()},
            ),
            "invalid",
        ),
        (
            SimpleNamespace(
                channel_name="right_wrist_pose",
                frame_name="vive_world",
                value=_pose(0.0, 0.0, 0.0),
                valid=True,
                critical=True,
                timestamp="malformed",
                metadata={"seq": 1, "receiver_timestamp": "malformed"},
            ),
            "timestamp_missing",
        ),
    ],
)
def test_preflight_rejects_stale_nonfinite_or_unclocked_pose(
    tmp_path: Path,
    channel: object,
    reason: str,
) -> None:
    controller = ViveCoordinateCalibrationController(
        ViveCoordinateCalibrationConfig(
            active_sides=("right",),
            profile_dir=tmp_path,
        ),
        source=_FixedControllerSource({"right_wrist_pose": channel}),
    )
    try:
        state = controller.snapshot()
        assert state["input_preflight"]["ready"] is False
        assert state["input_preflight"]["failures"] == [
            {"channel": "right_wrist_pose", "reason": reason}
        ]
        with pytest.raises(RuntimeError, match="开始前检查未通过"):
            controller.dispatch("begin_calibration")
    finally:
        controller.close()


def test_active_calibration_can_be_cancelled_and_reconfigured(tmp_path: Path) -> None:
    controller = ViveCoordinateCalibrationController(
        ViveCoordinateCalibrationConfig(
            active_sides=("right",),
            capture_prep_sec=0.0,
            profile_dir=tmp_path,
        ),
        source=_LiveControllerSource(("right_wrist_pose",)),
    )
    controller.start(start_source=False)
    try:
        controller.dispatch("begin_calibration")
        for _ in range(100):
            state = controller.snapshot()
            if state["workflow_active"] and state["stage"] == "virtual_desk_x":
                break
            time.sleep(0.01)
        assert state["workflow_active"] is True

        controller.dispatch("capture")
        for _ in range(100):
            state = controller.snapshot()
            if state["phase"] == "capturing":
                break
            time.sleep(0.01)
        assert state["phase"] == "capturing"

        controller.dispatch("cancel_calibration")
        for _ in range(100):
            state = controller.snapshot()
            if not state["workflow_active"]:
                break
            time.sleep(0.01)
        assert state["workflow_active"] is False
        assert state["stage"] == "idle"
        assert state["phase"] == "idle"
        assert state["error"] == ""

        controller.dispatch("set_active_sides", {"active_sides": "left"})
        assert controller.snapshot()["active_sides"] == ["left"]
    finally:
        controller.close()


def test_idle_capture_and_reset_cannot_pollute_next_workflow(tmp_path: Path) -> None:
    controller = ViveCoordinateCalibrationController(
        ViveCoordinateCalibrationConfig(
            active_sides=("right",),
            profile_dir=tmp_path,
        ),
        source=_LiveControllerSource(("right_wrist_pose",)),
    )
    try:
        with pytest.raises(RuntimeError, match="No active"):
            controller.dispatch("capture")
        with pytest.raises(RuntimeError, match="No active"):
            controller.dispatch("reset_stage")
    finally:
        controller.close()


def test_receiver_start_failure_does_not_leave_controller_unretryable(
    tmp_path: Path,
) -> None:
    source = _RecoveringControllerSource()
    controller = ViveCoordinateCalibrationController(
        ViveCoordinateCalibrationConfig(
            active_sides=("right",),
            profile_dir=tmp_path,
        ),
        source=source,
    )
    with pytest.raises(RuntimeError, match="synthetic receiver"):
        controller.start(start_source=True)
    failed = controller.snapshot()
    assert failed["lifecycle"] == "stopped"
    assert failed["phase"] == "error"
    assert "synthetic receiver start failure" in failed["error"]

    controller.start(start_source=True)
    assert source.start_calls == 2
    recovered = controller.snapshot()
    assert recovered["source_started"] is True
    assert recovered["phase"] == "idle"
    assert recovered["error"] == ""
    controller.close()
    assert source.closed is True


def test_close_event_requests_outer_calibration_loop_shutdown(tmp_path: Path) -> None:
    controller = ViveCoordinateCalibrationController(
        ViveCoordinateCalibrationConfig(profile_dir=tmp_path),
        source=_LiveControllerSource(()),
    )
    controller.start(start_source=False)
    controller.dispatch("close")

    state = controller.snapshot()
    assert controller.shutdown_requested is True
    assert state["shutdown_requested"] is True
    assert state["phase"] == "shutting_down"
    controller.close()


def test_close_event_interrupts_an_active_capture(tmp_path: Path) -> None:
    controller = ViveCoordinateCalibrationController(
        ViveCoordinateCalibrationConfig(
            active_sides=("right",),
            capture_prep_sec=0.0,
            profile_dir=tmp_path,
        ),
        source=_LiveControllerSource(("right_wrist_pose",)),
    )
    controller.start(start_source=False)
    controller.dispatch("begin_calibration")
    controller.dispatch("capture")
    controller.dispatch("close")
    for _ in range(100):
        state = controller.snapshot()
        if not state["workflow_active"]:
            break
        time.sleep(0.01)

    assert controller.shutdown_requested is True
    assert state["workflow_active"] is False
    controller.close()


def test_calibration_gui_can_swap_valid_bilateral_sources_before_capture(
    tmp_path: Path,
) -> None:
    source = VivePoseSource(
        [
            ViveTrackerConfig(
                name="tracker_1",
                side="left",
                role="left_wrist",
                channel_name="left_wrist_pose",
            ),
            ViveTrackerConfig(
                name="tracker_2",
                side="right",
                role="right_wrist",
                channel_name="right_wrist_pose",
            ),
        ],
        port=0,
        clock_sync_port=0,
    )
    controller = ViveCoordinateCalibrationController(
        ViveCoordinateCalibrationConfig(
            active_sides=("right",),
            profile_dir=tmp_path,
        ),
        source=source,
    )
    source._process_received_batch(
        [
            (
                _vive_role_packet(right_valid=True),
                time.time(),
                time.monotonic(),
                ("192.0.2.10", 4000),
            )
        ]
    )

    controller.dispatch("swap_vive_sources")
    state = controller.snapshot()

    assert state["source_role_by_output_channel"] == {
        "left_wrist_pose": "right_wrist",
        "right_wrist_pose": "left_wrist",
    }
    assert state["source_swap_count"] == 1
    assert state["stage"] == "idle"


def test_calibration_gui_swaps_single_tracker_without_source_admission(
    tmp_path: Path,
) -> None:
    source = VivePoseSource(
        [
            ViveTrackerConfig(
                name="tracker_1",
                side="right",
                role="left_wrist",
                channel_name="right_wrist_pose",
            )
        ],
        port=0,
        clock_sync_port=0,
    )
    source._process_received_batch(
        [
            (
                _vive_role_packet(right_valid=False),
                time.time(),
                time.monotonic(),
                ("192.0.2.10", 4000),
            )
        ]
    )
    controller = ViveCoordinateCalibrationController(
        ViveCoordinateCalibrationConfig(
            active_sides=("right",),
            profile_dir=tmp_path,
        ),
        source=source,
    )

    controller.dispatch("swap_vive_sources")

    assert source.source_role_by_output_channel == {
        "right_wrist_pose": "right_wrist",
    }
    assert controller.snapshot()["source_swap_count"] == 1


def test_calibration_gui_can_swap_single_tracker_back_without_source_admission(
    tmp_path: Path,
) -> None:
    source = VivePoseSource(
        [
            ViveTrackerConfig(
                name="tracker_1",
                side="right",
                role="right_wrist",
                channel_name="right_wrist_pose",
            )
        ],
        port=0,
        clock_sync_port=0,
    )
    source._process_received_batch(
        [
            (
                _vive_role_packet(right_valid=False),
                time.time(),
                time.monotonic(),
                ("192.0.2.10", 4000),
            )
        ]
    )
    controller = ViveCoordinateCalibrationController(
        ViveCoordinateCalibrationConfig(
            active_sides=("right",),
            profile_dir=tmp_path,
        ),
        source=source,
    )

    controller.dispatch("swap_vive_sources")

    assert source.source_role_by_output_channel == {
        "right_wrist_pose": "left_wrist",
    }
    assert controller.snapshot()["source_swap_count"] == 1


def test_single_tracker_swap_back_restores_selected_channel_immediately(
    tmp_path: Path,
) -> None:
    source = VivePoseSource(
        [
            ViveTrackerConfig(
                name="tracker_1",
                side="left",
                role="left_wrist",
                channel_name="left_wrist_pose",
            ),
            ViveTrackerConfig(
                name="tracker_2",
                side="right",
                role="right_wrist",
                channel_name="right_wrist_pose",
            ),
        ],
        port=0,
        clock_sync_port=0,
    )
    source._process_received_batch(
        [
            (
                _vive_role_packet(right_valid=False),
                time.time(),
                time.monotonic(),
                ("192.0.2.10", 4000),
            )
        ]
    )
    source.swap_left_right_source_roles()
    controller = ViveCoordinateCalibrationController(
        ViveCoordinateCalibrationConfig(
            active_sides=("right",),
            profile_dir=tmp_path,
        ),
        source=source,
    )

    assert controller.snapshot()["input_preflight"]["ready"] is True
    controller.dispatch("swap_vive_sources")
    assert controller.snapshot()["input_preflight"]["ready"] is False
    controller.dispatch("swap_vive_sources")
    state = controller.snapshot()
    assert state["input_preflight"]["ready"] is True
    assert state["source_role_by_output_channel"]["right_wrist_pose"] == "left_wrist"


def test_vive_calibration_gui_is_loopback_only(tmp_path: Path) -> None:
    controller = ViveCoordinateCalibrationController(
        ViveCoordinateCalibrationConfig(profile_dir=tmp_path)
    )
    try:
        with pytest.raises(ValueError, match="loopback"):
            ViveCoordinateCalibrationGui(controller, host="0.0.0.0")
    finally:
        controller.close()
