import ast
import json
import threading
import time
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.probes import capture_rm65_grasp_snapshot as probe
from scripts.probes.capture_rm65_grasp_snapshot import read_arm, stationarity
from teleop_runtime.real_teleop import camera_runtime
from teleop_runtime.deploy import camera_runtime as deploy_cameras
from teleop_runtime.video_frame import VideoFrame


@pytest.fixture
def capture_args(tmp_path):
    cameras = tmp_path / "cameras.json"
    cameras.write_text(json.dumps({key: {
        "serial": key + "-test", "width": 4, "height": 3, "fps": 15,
        "color_format": "bgr8", "color_space": "rgb", "enable_depth": True,
        "depth_width": 4, "depth_height": 3, "depth_fps": 15,
        "align_depth_to_color": True, "startup_hardware_reset": True,
        "startup_hardware_reset_on_failure": True,
        "startup_hardware_reset_on_warmup_failure": True, "auto_exposure_priority": 1,
        "metadata": {"startup_priority": priority},
    } for key, priority in (("front", 0), ("left_wrist", 20), ("right_wrist", 10))}))
    arms = tmp_path / "arms.json"
    arms.write_text(json.dumps({"arm_ip_by_side": {"left": "left-test", "right": "right-test"}}))
    return SimpleNamespace(output=tmp_path / "capture", camera_config=cameras, arm_ip_assignment=arms)


def make_frame(index=1, *, received_at=None):
    received_at = time.time() if received_at is None else received_at
    return VideoFrame(
        image=np.full((3, 4, 3), [1, 2, 3], dtype=np.uint8),
        source_timestamp=received_at, available_timestamp=received_at + .0001,
        frame_index=index, depth_image=np.full((3, 4), 1000 + index, dtype=np.uint16),
        metadata={"color_space": "rgb", "source_generation": 1,
                  "camera_timestamp_sec": index * .01, "timestamp_domain": "hardware_clock",
                  "color_frame_number": index},
        depth_metadata={"aligned_to": "color", "depth_scale_m_per_unit": .001,
                        "frame_number": index, "camera_timestamp_sec": index * .01},
    )


@pytest.fixture
def camera_rig(capture_args, monkeypatch):
    """Only device calls are fake; real batch, freshness and close helpers run."""
    sources = probe.readonly_camera_sources(capture_args.camera_config)
    events, modes, releases, finished = [], {}, {}, {}
    monkeypatch.setattr(probe, "readonly_camera_sources", lambda _: sources)
    monkeypatch.setattr(probe, "assert_devices_idle", lambda _: events.append(("ownership",)))
    monkeypatch.setattr(probe, "assert_arm_connections_idle", lambda _: events.append(("arm_ownership",)))
    monkeypatch.setattr(camera_runtime, "_start_camera_sources_best_effort", partial(
        camera_runtime._start_camera_sources_best_effort,
        stagger_sec=.02, deferred_retry_cooldown_sec=.01, startup_timeout_sec=.1,
    ))
    monkeypatch.setattr(deploy_cameras, "run_camera_startup_self_check", partial(
        deploy_cameras.run_camera_startup_self_check, duration_sec=.2, interval_sec=.01,
    ))

    def publish(source):
        while not source._shutdown_event.is_set():
            frame = source._frame_buffer.publish(make_frame(source._frame_counter))
            with source._lock:
                source._frame_counter += 1
                source._latest_frame = frame
                source._last_frame_receive_timestamp = frame.source_timestamp
            source._shutdown_event.wait(.005)

    def start(source):
        key = source.camera_key
        events.append(("start", key))
        journal = [json.loads(line) for line in (capture_args.output / "capture_stages.jsonl").read_text().splitlines()]
        assert any(row.get("camera") == f"camera.{key}" for row in journal)
        assert not any(row["stage"] == "connect_arm_readonly" for row in journal)
        mode = modes.get(key)
        if mode == "blocked":
            # Model an uninterruptible native start holding the source lock.
            with source._lifecycle_lock:
                releases[key].wait(3)
            finished[key].set()
            return
        if mode == "unavailable":
            raise RuntimeError("startup_warmup_failed:policy=bounded_first_frame:no_frames")
        if mode == "native_timeout":
            raise TimeoutError("native start returned with a timeout")
        with source._lock:
            source._running = True
            source._lifecycle_state = "running"
            source._generation = 1
            source._device_info = {"intrinsics": {"fx": 10., "fy": 11., "ppx": 2., "ppy": 1.},
                                   "depth_scale_m_per_unit": .001}
        if mode != "no_frames":
            source._thread = threading.Thread(target=publish, args=(source,), daemon=True)
            source._thread.start()

    for key, source in sources.items():
        releases[key], finished[key] = threading.Event(), threading.Event()
        monkeypatch.setattr(source, "start", partial(start, source))
        original_close = source.close

        def close(*, key=key, original=original_close):
            events.append(("close", key))
            original()

        monkeypatch.setattr(source, "close", close)

    class Arm:
        def Get_Current_Arm_State(self, *, retry):
            assert retry == 0
            return 0, [0]*6, [0]*6, 0, 0

        def Arm_Socket_Close(self):
            events.append(("arm_close",))
            return 0

    def arm_factory(ip):
        events.append(("arm_open", ip))
        # Admission uses three increasing frame counts, not just start() success.
        assert all(source._frame_counter >= 3 for key, source in sources.items() if not modes.get(key))
        return Arm()

    yield SimpleNamespace(sources=sources, modes=modes, events=events, arm_factory=arm_factory)
    for event in releases.values():
        event.set()
    for key in sources:
        if modes.get(key) == "blocked":
            assert finished[key].wait(1)
    for source in sources.values():
        # Test-only release, including a fake native worker left blocked by a test.
        try:
            source.close()
        except RuntimeError:
            pass


def test_config_reuses_collection_policy_without_reset_or_stream_downgrade(capture_args):
    sources = probe.readonly_camera_sources(capture_args.camera_config)
    assert [key for key, _ in camera_runtime._camera_startup_items(sources)] == ["front", "right_wrist", "left_wrist"]
    for key, source in sources.items():
        config = source.config
        assert config.serial == key + "-test"
        assert (config.width, config.height, config.fps, config.color_format) == (4, 3, 15, "bgr8")
        assert (config.depth_width, config.depth_height, config.depth_fps) == (4, 3, 15)
        assert config.enable_depth and config.align_depth_to_color
        assert config.startup_warmup_policy == "bounded_first_frame" and config.require_complete_frameset
        assert config.restart_on_incomplete_frameset
        assert not config.startup_hardware_reset
        assert not config.startup_hardware_reset_on_failure
        assert not config.startup_hardware_reset_on_warmup_failure
        assert config.auto_exposure_priority is None
        assert source.device_info()["lifecycle_state"] == "stopped"


@pytest.mark.parametrize("field", ["enable_depth", "align_depth_to_color"])
def test_rgb_only_or_unaligned_config_is_rejected_before_devices(capture_args, field):
    configs = json.loads(capture_args.camera_config.read_text())
    configs["front"][field] = False
    capture_args.camera_config.write_text(json.dumps(configs))
    with pytest.raises(ValueError, match="aligned RGB-D"):
        probe.readonly_camera_sources(capture_args.camera_config)


def test_capture_reuses_order_live_workers_and_reverse_cleanup(capture_args, camera_rig, monkeypatch):
    original = camera_rig.sources["right_wrist"].start

    def start_second():
        first = camera_rig.sources["front"]
        before = first.device_info()["frame_count"]
        time.sleep(.025)
        assert first.device_info()["frame_count"] > before
        original()

    monkeypatch.setattr(camera_rig.sources["right_wrist"], "start", start_second)
    result = probe.capture(capture_args, arm_factory=camera_rig.arm_factory)
    assert result["status"] == "captured", result
    assert result["all_views_captured"] and result["cleanup_complete"]
    assert result["camera_lifecycle"] == "real_teleop"
    assert result["motion_commands"] == [] and not result["camera_reset"]
    assert [event[1] for event in camera_rig.events if event[0] == "start"] == ["front", "right_wrist", "left_wrist"]
    assert [event[1] for event in camera_rig.events if event[0] == "close"] == ["left_wrist", "right_wrist", "front"]
    assert camera_rig.events.index(("arm_close",)) < camera_rig.events.index(("close", "left_wrist"))
    for record in result["views"].values():
        assert len(record["frames"]) == 5
        assert len({frame["source_frame_index"] for frame in record["frames"]}) == 5
        for frame in record["frames"]:
            assert result["capture_started_at"] <= frame["received_at"] <= result["capture_finished_at"]
            with np.load(capture_args.output / frame["file"]) as data:
                assert data["rgb"][0, 0].tolist() == [1, 2, 3]
                np.testing.assert_allclose(data["depth_m"], (1000 + frame["frame_number"]) * .001,
                                           rtol=0, atol=np.finfo(np.float32).eps)


def test_unavailable_view_gets_only_canonical_deferred_retry(capture_args, camera_rig):
    camera_rig.modes.update(front="unavailable", right_wrist="unavailable")
    result = probe.capture(capture_args, arm_factory=camera_rig.arm_factory)
    assert result["status"] == "captured", result
    assert not result["all_views_captured"]
    assert set(result["missing_roles"]) == {"main", "wrist_right"}
    assert result["views"]["main"]["status"] == "unavailable"
    assert "deferred_startup_retry_failed" in result["views"]["main"]["error"]
    assert [event[1] for event in camera_rig.events if event[0] == "start"] == [
        "front", "right_wrist", "left_wrist", "front", "right_wrist"]
    assert len(result["views"]["wrist_left"]["frames"]) == 5


@pytest.mark.parametrize("blocked_key", ["front", "right_wrist"])
def test_blocked_native_start_aborts_batch_and_never_connects_arms(capture_args, camera_rig, blocked_key):
    camera_rig.modes[blocked_key] = "blocked"
    result = probe.capture(capture_args, arm_factory=camera_rig.arm_factory)
    assert result["status"] == "failed"
    assert "Camera lifecycle fault" in result["error"]["message"]
    assert "startup_aborted_after_timeout" in result["views"]["wrist_left"]["error"]
    expected = ["front"] if blocked_key == "front" else ["front", "right_wrist"]
    assert [event[1] for event in camera_rig.events if event[0] == "start"] == expected
    assert not any(event[0] == "arm_open" for event in camera_rig.events)
    assert not result["cleanup_complete"]
    assert "lifecycle lock" in result["cleanup_errors"][0]
    assert json.loads((capture_args.output / "snapshot.json").read_text())["status"] == "failed"


def test_native_timeout_also_aborts_when_an_earlier_camera_is_ready(capture_args, camera_rig):
    camera_rig.modes["right_wrist"] = "native_timeout"
    result = probe.capture(capture_args, arm_factory=camera_rig.arm_factory)
    assert result["status"] == "failed"
    assert "startup timeout" in result["error"]["message"]
    assert result["views"]["main"]["status"] == "ready"
    assert not any(event[0] == "arm_open" for event in camera_rig.events)
    assert result["cleanup_complete"]


def test_no_frames_after_start_cannot_pass_freshness_admission(capture_args, camera_rig):
    camera_rig.modes["front"] = "no_frames"
    result = probe.capture(capture_args, arm_factory=camera_rig.arm_factory)
    assert result["status"] == "failed"
    assert "startup self-check failed" in result["error"]["message"]
    assert not any(event[0] == "arm_open" for event in camera_rig.events)


def test_interrupt_during_start_is_not_swallowed_as_camera_failure(capture_args, camera_rig, monkeypatch):
    def interrupted():
        raise KeyboardInterrupt("operator cancelled")

    monkeypatch.setattr(camera_rig.sources["front"], "start", interrupted)
    result = probe.capture(capture_args, arm_factory=camera_rig.arm_factory)
    assert result["status"] == "failed"
    assert result["error"] == {"type": "KeyboardInterrupt", "message": "operator cancelled"}
    assert not any(event[0] in {"start", "arm_open"} for event in camera_rig.events)
    assert result["cleanup_complete"]


def test_valid_data_is_saved_before_native_close_failure(capture_args, camera_rig, monkeypatch):
    original = camera_rig.sources["left_wrist"].close

    def failing_close():
        retained = json.loads((capture_args.output / "snapshot.json").read_text())
        assert retained["data_captured"] and retained["qpos_samples"] and retained["stationarity"]
        assert not retained["cleanup_complete"] and retained["status"] == "failed"
        assert len(list(capture_args.output.glob("*.npz"))) == 15
        original()
        raise RuntimeError("injected native close failure")

    monkeypatch.setattr(camera_rig.sources["left_wrist"], "close", failing_close)
    result = probe.capture(capture_args, arm_factory=camera_rig.arm_factory)
    assert result["status"] == "failed" and result["data_captured"]
    assert "injected native close failure" in result["cleanup_errors"][0]
    assert sum(event[0] == "close" for event in camera_rig.events) == 3


def test_primary_failure_is_saved_before_cleanup_and_cleanup_failure_is_reported(capture_args, camera_rig, monkeypatch):
    camera_rig.modes.update({key: "unavailable" for key in probe.ROLES})
    original = camera_rig.sources["left_wrist"].close

    def failing_close():
        retained = json.loads((capture_args.output / "snapshot.json").read_text())
        # Deferred-retry close precedes the terminal acquisition error.
        if retained.get("error"):
            assert "No usable camera" in retained["error"]["message"]
            original()
            raise RuntimeError("injected close failure")
        original()

    monkeypatch.setattr(camera_rig.sources["left_wrist"], "close", failing_close)
    result = probe.capture(capture_args, arm_factory=camera_rig.arm_factory)
    assert "No usable camera" in result["error"]["message"]
    assert "injected close failure" in result["cleanup_errors"][0]
    assert not result["cleanup_complete"]


def test_storage_failure_does_not_skip_device_cleanup(capture_args, camera_rig, monkeypatch):
    def fail_save(*args, **kwargs):
        raise OSError("injected disk full")

    monkeypatch.setattr(probe.np, "savez_compressed", fail_save)
    result = probe.capture(capture_args, arm_factory=camera_rig.arm_factory)
    assert result["status"] == "failed"
    assert result["persistence_errors"] == ["OSError: injected disk full"]
    assert result["cleanup_complete"]
    assert sum(event[0] == "arm_close" for event in camera_rig.events) == 2
    assert sum(event[0] == "close" for event in camera_rig.events) == 3


@pytest.mark.parametrize("condition", ["stale", "duplicate", "before_capture", "none"])
def test_cached_frame_not_retimestamped_or_reused(condition):
    now = time.time()
    frame = make_frame(received_at=now - (1 if condition == "stale" else .01))
    source = SimpleNamespace(get_latest_frame=lambda: None if condition == "none" else frame)
    with pytest.raises(TimeoutError, match="new fresh RGB-D"):
        probe.next_frames({"front": source}, {"front": (1, 1)} if condition == "duplicate" else {},
                          after=now if condition == "before_capture" else now - 2, timeout_sec=0)


def test_new_generation_and_future_timestamp_are_not_accepted():
    frame = make_frame()
    source = SimpleNamespace(get_latest_frame=lambda: frame)
    with pytest.raises(RuntimeError, match="generation changed"):
        probe.next_frames({"front": source}, {"front": (0, 1)}, after=0, timeout_sec=0)
    frame = make_frame(received_at=time.time() + 10)
    with pytest.raises(RuntimeError, match="invalid camera receive timestamp"):
        probe.next_frames({"front": source}, {}, after=0, timeout_sec=0)


def test_rgb_depth_and_timestamps_come_from_one_source_frame():
    frame = make_frame(index=7, received_at=100.)
    frame.metadata["color_space"] = "bgr"
    source = SimpleNamespace(camera_key="front", device_info=lambda: {"intrinsics": {
        "fx": 10, "fy": 11, "ppx": 2, "ppy": 1}})
    rgb, depth, metadata = probe.frame_payload(source, frame, index=0)
    assert rgb[0, 0].tolist() == [3, 2, 1]
    np.testing.assert_allclose(depth, 1.007, rtol=0, atol=np.finfo(np.float32).eps)
    assert metadata["received_at"] == 100.
    assert metadata["available_at"] == 100.0001
    assert metadata["device_timestamp_ms"] == 70.
    assert metadata["depth_metadata"]["frame_number"] == metadata["frame_number"] == 7
    assert metadata["intrinsics"] == {"fx": 10, "fy": 11, "cx": 2, "cy": 1}


@pytest.mark.parametrize("invalid", ["depth_missing", "unaligned", "zero_scale", "nan_scale"])
def test_invalid_rgbd_cannot_be_saved_as_usable(invalid):
    frame = make_frame()
    if invalid == "depth_missing":
        frame = replace(frame, depth_image=None)
    elif invalid == "unaligned":
        frame.depth_metadata["aligned_to"] = "none"
    else:
        frame.depth_metadata["depth_scale_m_per_unit"] = 0 if invalid == "zero_scale" else float("nan")
    source = SimpleNamespace(camera_key="front", device_info=lambda: {})
    with pytest.raises(RuntimeError, match="missing aligned metric depth"):
        probe.frame_payload(source, frame, index=0)


def test_only_read_and_close_arm_methods_are_called():
    source = Path(__file__).resolve().parents[1] / "scripts/probes/capture_rm65_grasp_snapshot.py"
    tree = ast.parse(source.read_text())
    names = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
             and node.func.value.id == "arm"}
    assert names == {"Get_Current_Arm_State", "Arm_Socket_Close"}
    assert "hardware_reset" not in {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "set_option" not in {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}


def test_read_arm_does_not_retry_or_command():
    class ReadOnly:
        def Get_Current_Arm_State(self, *, retry):
            assert retry == 0
            return 0, [0]*6, [0]*6, 0, 0
    result = read_arm(ReadOnly())
    assert result["qpos_rad"] == [0]*6
    assert result["finished_at"] >= result["started_at"]


def samples():
    return [{side: {"started_at": index*.05, "finished_at": index*.05+.005,
                   "qpos_rad": [0]*6} for side in ("left", "right")} for index in range(10)]


def test_stationary_evidence_needs_coverage_cadence_and_both_arms():
    data = samples()
    assert stationarity(data, .1, .3)["left"]["sample_count"] == 10
    with pytest.raises(RuntimeError, match="bracket"):
        stationarity(data, -.1, .3)
    data[4]["right"]["qpos_rad"][0] = .01
    with pytest.raises(RuntimeError, match="right.*stationary"):
        stationarity(data, .1, .3)
    with pytest.raises(RuntimeError, match="gap"):
        stationarity(samples()[::3], .1, .3)
