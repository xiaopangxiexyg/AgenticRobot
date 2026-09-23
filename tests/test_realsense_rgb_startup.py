from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from teleop_runtime.real_teleop import camera_runtime
from teleop_runtime.cameras import capture_realsense_intrinsics
from teleop_runtime.sources.camera import realsense_rgb
from teleop_runtime.sources.camera import startup as camera_startup
from teleop_runtime.sources.camera.realsense_rgb import (
    REALSENSE_STARTUP_WARMUP_BOUNDED_FIRST_FRAME,
    REALSENSE_STARTUP_WARMUP_LEGACY,
    RealSenseRGBCameraConfig,
    RealSenseRGBSource,
    load_realsense_camera_config_map,
)
from utils.head_eye import camera as head_eye_camera
from utils.calibration_common.camera import start_camera_bounded


class _FakeStreamConfig:
    def __init__(self) -> None:
        self.device: str | None = None
        self.streams: list[tuple[object, int, int, object, int]] = []

    def enable_device(self, serial: str) -> None:
        self.device = serial

    def enable_stream(self, stream: object, width: int, height: int, fmt: object, fps: int) -> None:
        self.streams.append((stream, width, height, fmt, fps))


class _FakeDepthSensor:
    def get_depth_scale(self) -> float:
        return 0.001


class _FakeDevice:
    def first_depth_sensor(self) -> _FakeDepthSensor:
        return _FakeDepthSensor()


class _FakeProfile:
    def get_device(self) -> _FakeDevice:
        return _FakeDevice()


class _FakeFrame:
    pass


class _FakeFrameSet:
    def __init__(self, *, color: bool = True, depth: bool = False) -> None:
        self.color = _FakeFrame() if color else None
        self.depth = _FakeFrame() if depth else None

    def get_color_frame(self) -> _FakeFrame | None:
        return self.color

    def get_depth_frame(self) -> _FakeFrame | None:
        return self.depth


class _FakeAlign:
    def process(self, frames: _FakeFrameSet) -> _FakeFrameSet:
        return frames


class _FakePipeline:
    def __init__(self, owner: _FakeRealSense | None = None) -> None:
        self.owner = owner
        self.stop_calls = 0
        self.start_configs: list[_FakeStreamConfig] = []
        self.wait_timeouts: list[int] = []

    def stop(self) -> None:
        self.stop_calls += 1
        raise RuntimeError("stop before start")

    def start(self, stream_config: _FakeStreamConfig) -> _FakeProfile:
        if self.owner is not None and self.owner.start_errors:
            raise self.owner.start_errors.pop(0)
        self.start_configs.append(stream_config)
        return _FakeProfile()

    def wait_for_frames(self, *, timeout_ms: int) -> _FakeFrameSet:
        self.wait_timeouts.append(timeout_ms)
        if self.owner is not None and self.owner.wait_errors:
            raise self.owner.wait_errors.pop(0)
        if self.owner is not None and self.owner.wait_frames:
            return self.owner.wait_frames.pop(0)
        depth_enabled = bool(
            self.start_configs
            and any(stream[0] == "depth" for stream in self.start_configs[-1].streams)
        )
        return _FakeFrameSet(color=True, depth=depth_enabled)


class _FakeRealSense:
    def __init__(self) -> None:
        self.stream = SimpleNamespace(color="color", depth="depth")
        self.format = SimpleNamespace(bgr8="bgr8", yuyv="yuyv", z16="z16")
        self.camera_info = SimpleNamespace(serial_number="serial_number")
        self.last_pipeline: _FakePipeline | None = None
        self.pipelines: list[_FakePipeline] = []
        self.last_config: _FakeStreamConfig | None = None
        self.align_requests: list[object] = []
        self.devices: list[_FakeHardwareDevice] = []
        self.start_errors: list[RuntimeError] = []
        self.wait_errors: list[RuntimeError] = []
        self.wait_frames: list[_FakeFrameSet] = []

    def pipeline(self) -> _FakePipeline:
        self.last_pipeline = _FakePipeline(self)
        self.pipelines.append(self.last_pipeline)
        return self.last_pipeline

    def config(self) -> _FakeStreamConfig:
        self.last_config = _FakeStreamConfig()
        return self.last_config

    def align(self, stream: object) -> _FakeAlign:
        self.align_requests.append(stream)
        return _FakeAlign()

    def context(self) -> object:
        return SimpleNamespace(query_devices=lambda: list(self.devices))


class _FakeHardwareDevice:
    def __init__(self, serial: str) -> None:
        self.serial = serial
        self.hardware_reset_calls = 0
        self.reset_error: RuntimeError | None = None

    def get_info(self, info_key: object) -> str:
        if info_key != "serial_number":
            raise KeyError(info_key)
        return self.serial

    def hardware_reset(self) -> None:
        self.hardware_reset_calls += 1
        if self.reset_error is not None:
            raise self.reset_error


def test_realsense_legacy_startup_resets_and_warms_up_front_without_depth() -> None:
    fake_rs = _FakeRealSense()
    source = RealSenseRGBSource(
        RealSenseRGBCameraConfig(
            serial="FRONT123",
            camera_key="front",
            width=640,
            height=480,
            fps=15,
            startup_warmup_frames=3,
            startup_warmup_policy=REALSENSE_STARTUP_WARMUP_LEGACY,
        )
    )

    with patch.object(realsense_rgb.time, "sleep") as sleep_mock:
        pipeline, _ = source._start_pipeline(fake_rs)

    assert pipeline is fake_rs.last_pipeline
    assert fake_rs.last_config is not None
    assert fake_rs.last_config.device == "FRONT123"
    assert fake_rs.last_config.streams == [("color", 640, 480, "yuyv", 15)]
    assert pipeline.stop_calls == 1
    assert pipeline.wait_timeouts == [5000, 5000, 5000]
    sleep_mock.assert_called_once_with(0.2)


def test_realsense_startup_depth_stream_is_only_enabled_when_requested() -> None:
    fake_rs = _FakeRealSense()
    source = RealSenseRGBSource(
        RealSenseRGBCameraConfig(
            serial="LEFT123",
            camera_key="left_wrist",
            width=640,
            height=480,
            fps=15,
            enable_depth=True,
            depth_width=424,
            depth_height=240,
            depth_fps=15,
            startup_reset=False,
            startup_warmup_frames=0,
        )
    )

    pipeline, _ = source._start_pipeline(fake_rs)

    assert pipeline.stop_calls == 0
    assert fake_rs.last_config is not None
    assert fake_rs.last_config.streams == [
        ("color", 640, 480, "yuyv", 15),
        ("depth", 424, 240, "z16", 15),
    ]
    assert fake_rs.align_requests == ["color"]


def test_realsense_startup_can_hardware_reset_matched_serial_before_start() -> None:
    fake_rs = _FakeRealSense()
    other_device = _FakeHardwareDevice("OTHER")
    target_device = _FakeHardwareDevice("FRONT123")
    fake_rs.devices = [other_device, target_device]
    source = RealSenseRGBSource(
        RealSenseRGBCameraConfig(
            serial="FRONT123",
            camera_key="front",
            width=640,
            height=480,
            fps=15,
            startup_reset=False,
            startup_warmup_frames=0,
            startup_hardware_reset=True,
            startup_hardware_reset_sleep_sec=1.25,
        )
    )

    with patch.object(source._shutdown_event, "wait", return_value=False) as wait_mock:
        pipeline, _ = source._start_pipeline(fake_rs)

    assert pipeline.stop_calls == 0
    assert other_device.hardware_reset_calls == 0
    assert target_device.hardware_reset_calls == 1
    wait_mock.assert_called_once_with(1.25)


def test_realsense_startup_retries_after_connection_error_with_hardware_reset() -> None:
    fake_rs = _FakeRealSense()
    target_device = _FakeHardwareDevice("LEFT123")
    fake_rs.devices = [target_device]
    fake_rs.start_errors = [RuntimeError("Device or resource busy")]
    source = RealSenseRGBSource(
        RealSenseRGBCameraConfig(
            serial="LEFT123",
            camera_key="left_wrist",
            width=480,
            height=270,
            fps=15,
            startup_reset=False,
            startup_warmup_frames=0,
            startup_hardware_reset_sleep_sec=1.25,
        )
    )

    with patch.object(source._shutdown_event, "wait", return_value=False) as wait_mock:
        pipeline, _ = source._start_pipeline(fake_rs)

    assert pipeline is fake_rs.pipelines[1]
    assert len(fake_rs.pipelines) == 2
    assert fake_rs.pipelines[0].stop_calls == 1
    assert target_device.hardware_reset_calls == 1
    wait_mock.assert_called_once_with(1.25)


def test_realsense_startup_does_not_hardware_reset_unsupported_profile() -> None:
    fake_rs = _FakeRealSense()
    target_device = _FakeHardwareDevice("LEFT123")
    fake_rs.devices = [target_device]
    fake_rs.start_errors = [RuntimeError("Couldn't resolve requests")]
    source = RealSenseRGBSource(
        RealSenseRGBCameraConfig(
            serial="LEFT123",
            camera_key="left_wrist",
            width=640,
            height=480,
            fps=15,
            enable_depth=True,
            depth_width=480,
            depth_height=270,
            depth_fps=15,
            startup_reset=False,
            startup_warmup_frames=0,
        )
    )

    with pytest.raises(RuntimeError, match="configured profile was preserved") as raised:
        source._start_pipeline(fake_rs)

    assert len(fake_rs.pipelines) == 1
    assert fake_rs.pipelines[0].stop_calls == 1
    assert target_device.hardware_reset_calls == 0
    assert '"failed_stage": "pipeline_start"' in str(raised.value)
    assert '"pipeline_stop_error": "RuntimeError: stop before start"' in str(raised.value)


def test_realsense_legacy_startup_retries_when_warmup_gets_no_frames() -> None:
    fake_rs = _FakeRealSense()
    target_device = _FakeHardwareDevice("LEFT123")
    fake_rs.devices = [target_device]
    fake_rs.wait_errors = [
        RuntimeError("Frame didn't arrive within 1000"),
        RuntimeError("Frame didn't arrive within 1000"),
    ]
    source = RealSenseRGBSource(
        RealSenseRGBCameraConfig(
            serial="LEFT123",
            camera_key="left_wrist",
            width=480,
            height=270,
            fps=15,
            startup_reset=False,
            startup_warmup_frames=2,
            startup_warmup_policy=REALSENSE_STARTUP_WARMUP_LEGACY,
            startup_hardware_reset_sleep_sec=1.25,
        )
    )

    with patch.object(source._shutdown_event, "wait", return_value=False) as wait_mock:
        pipeline, _ = source._start_pipeline(fake_rs)

    assert pipeline is fake_rs.pipelines[1]
    assert len(fake_rs.pipelines) == 2
    assert fake_rs.pipelines[0].stop_calls == 1
    assert target_device.hardware_reset_calls == 1
    wait_mock.assert_called_once_with(1.25)


def test_realsense_legacy_startup_aborts_warmup_after_timeout_threshold() -> None:
    fake_rs = _FakeRealSense()
    fake_rs.wait_errors = [RuntimeError("Frame didn't arrive within 1000")] * 10
    source = RealSenseRGBSource(
        RealSenseRGBCameraConfig(
            serial="FRONT123",
            startup_reset=False,
            startup_warmup_frames=10,
            startup_warmup_policy=REALSENSE_STARTUP_WARMUP_LEGACY,
            restart_after_timeouts=3,
            startup_hardware_reset_on_failure=False,
        )
    )

    with pytest.raises(RuntimeError, match="no_frames_after=10"):
        source._start_pipeline(fake_rs)

    assert fake_rs.pipelines[0].wait_timeouts == [5000, 5000, 5000]


def test_realsense_bounded_warmup_returns_after_first_valid_frame() -> None:
    fake_rs = _FakeRealSense()
    source = RealSenseRGBSource(
        RealSenseRGBCameraConfig(
            serial="FRONT123",
            startup_reset=False,
            startup_warmup_frames=10,
            startup_warmup_policy=REALSENSE_STARTUP_WARMUP_BOUNDED_FIRST_FRAME,
        )
    )

    pipeline, _ = source._start_pipeline(fake_rs)

    assert len(pipeline.wait_timeouts) == 1
    assert 1 <= pipeline.wait_timeouts[0] <= 1000


def test_realsense_bounded_warmup_failure_does_not_immediately_reopen_pipeline() -> None:
    fake_rs = _FakeRealSense()
    target_device = _FakeHardwareDevice("FRONT123")
    fake_rs.devices = [target_device]
    fake_rs.wait_errors = [RuntimeError("Frame didn't arrive within 1000")] * 5
    source = RealSenseRGBSource(
        RealSenseRGBCameraConfig(
            serial="FRONT123",
            startup_reset=False,
            startup_warmup_frames=10,
            startup_warmup_policy=REALSENSE_STARTUP_WARMUP_BOUNDED_FIRST_FRAME,
        )
    )

    with pytest.raises(RuntimeError, match="policy=bounded_first_frame") as raised:
        source._start_pipeline(fake_rs)

    assert len(fake_rs.pipelines) == 1
    assert fake_rs.pipelines[0].wait_timeouts == [1000] * 5
    assert fake_rs.pipelines[0].stop_calls == 1
    assert target_device.hardware_reset_calls == 0
    assert '"failed_stage": "first_frame_warmup"' in str(raised.value)
    assert '"pipeline_start_sec":' in str(raised.value)
    assert '"pipeline_stop_error": "RuntimeError: stop before start"' in str(raised.value)


def test_realsense_bounded_warmup_leaves_deferred_retry_to_multi_camera_caller() -> None:
    fake_rs = _FakeRealSense()
    target_device = _FakeHardwareDevice("FRONT123")
    fake_rs.devices = [target_device]
    fake_rs.wait_errors = [RuntimeError("Frame didn't arrive within 1000")] * 5
    source = RealSenseRGBSource(
        RealSenseRGBCameraConfig(
            serial="FRONT123",
            startup_reset=False,
            startup_warmup_frames=10,
            startup_warmup_policy=REALSENSE_STARTUP_WARMUP_BOUNDED_FIRST_FRAME,
        )
    )

    with pytest.raises(RuntimeError, match="policy=bounded_first_frame"):
        source._start_pipeline(fake_rs)

    assert len(fake_rs.pipelines) == 1
    assert fake_rs.pipelines[0].wait_timeouts == [1000] * 5
    assert fake_rs.pipelines[0].stop_calls == 1
    assert target_device.hardware_reset_calls == 0


def test_realsense_bounded_warmup_can_reset_only_the_selected_camera() -> None:
    fake_rs = _FakeRealSense()
    front = _FakeHardwareDevice("FRONT123")
    other = _FakeHardwareDevice("WRIST456")
    fake_rs.devices = [front, other]
    fake_rs.wait_errors = [RuntimeError("Frame didn't arrive within 1000")] * 5
    source = RealSenseRGBSource(
        RealSenseRGBCameraConfig(
            serial="FRONT123",
            camera_key="front",
            startup_reset=False,
            startup_hardware_reset_on_failure=False,
            startup_hardware_reset_on_warmup_failure=True,
            startup_hardware_reset_sleep_sec=0.0,
        )
    )

    pipeline, _ = source._start_pipeline(fake_rs)

    assert pipeline is fake_rs.pipelines[1]
    assert len(fake_rs.pipelines) == 2
    assert front.hardware_reset_calls == 1
    assert other.hardware_reset_calls == 0
    assert source.startup_timing["hardware_reset_retries"] == 1.0
    assert source.startup_timing["total_with_recovery_sec"] >= 0.0


def test_realsense_bounded_warmup_stops_if_target_reset_does_not_reconnect() -> None:
    fake_rs = _FakeRealSense()
    target = _FakeHardwareDevice("FRONT123")
    fake_rs.devices = [target]
    fake_rs.wait_errors = [RuntimeError("Frame didn't arrive within 1000")] * 5
    source = RealSenseRGBSource(
        RealSenseRGBCameraConfig(
            serial="FRONT123",
            startup_reset=False,
            startup_hardware_reset_on_failure=False,
            startup_hardware_reset_on_warmup_failure=True,
            startup_hardware_reset_sleep_sec=0.0,
        )
    )

    with patch.object(source, "_wait_for_device_reconnect", return_value=False):
        with pytest.raises(RuntimeError, match="startup_hardware_reset_failed"):
            source._start_pipeline(fake_rs)

    assert len(fake_rs.pipelines) == 1
    assert target.hardware_reset_calls == 1


def test_realsense_reset_error_requires_observed_reenumeration_before_retry() -> None:
    fake_rs = _FakeRealSense()
    front = _FakeHardwareDevice("FRONT123")
    front.reset_error = RuntimeError("map_device_descriptor: No such device")
    fake_rs.devices = [front]
    fake_rs.wait_errors = [RuntimeError("Frame didn't arrive within 1000")] * 5
    source = RealSenseRGBSource(
        RealSenseRGBCameraConfig(
            serial="FRONT123",
            startup_reset=False,
            startup_hardware_reset_on_failure=False,
            startup_hardware_reset_on_warmup_failure=True,
        )
    )

    with patch.object(source, "_wait_for_device_reconnect", return_value=True) as reconnect:
        pipeline, _ = source._start_pipeline(fake_rs)

    assert pipeline is fake_rs.pipelines[1]
    reconnect.assert_called_once_with(fake_rs, require_disconnect=True)
    assert front.hardware_reset_calls == 1


def test_gateway_profile_limits_warmup_recovery_to_front_camera() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "teleop_runtime/configs/devices/agent_gateway_realsense.json"
    )
    configs = load_realsense_camera_config_map(config_path)
    camera_runtime.apply_real_teleop_camera_startup_defaults(configs)

    assert configs["front"].startup_hardware_reset_on_warmup_failure is True
    assert configs["front"].startup_hardware_reset is False
    assert configs["front"].startup_hardware_reset_retry_attempts == 1
    assert configs["left_wrist"].startup_hardware_reset_on_warmup_failure is False
    assert configs["right_wrist"].startup_hardware_reset_on_warmup_failure is False


@pytest.mark.parametrize(
    ("profile", "expected_keys"),
    [
        ("realsense.json", {"front", "left_wrist", "right_wrist"}),
        ("realsense_front.json", {"front"}),
    ],
)
def test_shared_station_profiles_limit_hardware_recovery_to_front_warmup(
    profile: str,
    expected_keys: set[str],
) -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "teleop_runtime/configs/devices"
        / profile
    )
    configs = load_realsense_camera_config_map(config_path)
    camera_runtime.apply_real_teleop_camera_startup_defaults(configs)

    assert set(configs) == expected_keys
    assert configs["front"].startup_hardware_reset_on_warmup_failure is True
    assert configs["front"].startup_hardware_reset_on_failure is False
    assert configs["front"].startup_hardware_reset_retry_attempts == 1
    assert configs["front"].startup_software_retry_attempts == 2
    for key in expected_keys - {"front"}:
        assert configs[key].startup_hardware_reset_on_warmup_failure is False
        assert configs[key].startup_hardware_reset_on_failure is False
        assert configs[key].startup_software_retry_attempts == 1


def test_legacy_collection_adapter_does_not_close_timed_out_native_start() -> None:
    source = RealSenseRGBSource(RealSenseRGBCameraConfig(serial="FRONT123"))
    adapter = camera_startup.BoundedRealSenseSessionSource(source)

    with (
        patch.object(
            camera_startup,
            "start_real_sense_source_with_containment",
            side_effect=TimeoutError("native start blocked"),
        ),
        patch.object(source, "close") as close,
    ):
        with pytest.raises(TimeoutError, match="native start blocked"):
            adapter.start()
        adapter.close()
        with pytest.raises(RuntimeError, match="reconstruct"):
            adapter.start()
        close.assert_not_called()


def test_legacy_collection_adapter_delegates_successful_source_lifecycle() -> None:
    source = RealSenseRGBSource(RealSenseRGBCameraConfig(serial="FRONT123"))
    adapter = camera_startup.BoundedRealSenseSessionSource(source)

    with (
        patch.object(
            camera_startup,
            "start_real_sense_source_with_containment"
        ) as start,
        patch.object(source, "close") as close,
    ):
        adapter.start()
        adapter.close()

    start.assert_called_once_with(
        source, timeout_sec=30.0, software_retry_attempts=None
    )
    close.assert_called_once_with()


def test_head_eye_camera_releases_pipeline_if_first_frame_never_arrives() -> None:
    class FakePipeline:
        stop_count = 0

        def start(self, _config):
            return object()

        def wait_for_frames(self, _timeout_ms):
            raise RuntimeError("Frame didn't arrive")

        def stop(self):
            self.stop_count += 1

    class FakeConfig:
        def enable_device(self, _serial):
            pass

        def enable_stream(self, *_args):
            pass

    pipeline = FakePipeline()
    fake_rs = SimpleNamespace(
        pipeline=lambda: pipeline,
        config=FakeConfig,
        stream=SimpleNamespace(color="color"),
        format=SimpleNamespace(bgr8="bgr8"),
    )
    config = head_eye_camera.RealSenseConfig(
        serial="FRONT123",
        enable_depth=False,
        color_format="bgr8",
        startup_frame_grace_sec=0.01,
    )

    with patch.object(head_eye_camera, "_require_pyrealsense2", return_value=fake_rs):
        camera = head_eye_camera.RealSenseCamera(config)
        with pytest.raises(RuntimeError, match="first-frame warmup failed"):
            camera.start()

    assert pipeline.stop_count == 1
    assert camera._pipeline is None


def test_calibration_camera_native_start_timeout_is_terminal() -> None:
    release = threading.Event()
    cancelled = threading.Event()

    with pytest.raises(TimeoutError, match="exit this process without retry"):
        start_camera_bounded(
            lambda: release.wait(1.0),
            timeout_sec=0.01,
            on_timeout=cancelled.set,
        )

    assert cancelled.is_set()
    release.set()


def test_head_eye_camera_fallback_uses_fresh_pipeline() -> None:
    class FakePipeline:
        def __init__(self, *, fails: bool) -> None:
            self.fails = fails
            self.stop_count = 0

        def start(self, _config):
            if self.fails:
                raise RuntimeError("unsupported profile")
            return object()

        def wait_for_frames(self, _timeout_ms):
            return SimpleNamespace(get_color_frame=lambda: object())

        def stop(self):
            self.stop_count += 1

    class FakeConfig:
        def enable_device(self, _serial):
            pass

        def enable_stream(self, *_args):
            pass

    first_pipeline = FakePipeline(fails=True)
    second_pipeline = FakePipeline(fails=False)
    pipelines = [first_pipeline, second_pipeline]
    fake_rs = SimpleNamespace(
        pipeline=lambda: pipelines.pop(0),
        config=FakeConfig,
        stream=SimpleNamespace(color="color"),
        format=SimpleNamespace(bgr8="bgr8"),
    )
    config = head_eye_camera.RealSenseConfig(
        serial="FRONT123", enable_depth=False, color_format="bgr8"
    )

    with patch.object(head_eye_camera, "_require_pyrealsense2", return_value=fake_rs):
        camera = head_eye_camera.RealSenseCamera(config)
    fallback_config = FakeConfig()
    with patch.object(
        camera,
        "_fallback_stream_config",
        return_value=(fallback_config, {"fallback_used": True}, "bgr8"),
    ):
        camera.start()

    assert camera.settings_report["fallback_used"] is True
    assert first_pipeline.stop_count == 1
    camera.stop()
    assert second_pipeline.stop_count == 1


def test_intrinsics_capture_does_not_close_a_timed_out_native_source() -> None:
    events: list[str] = []

    class FakeSource:
        def __init__(self, key: str) -> None:
            self.key = key

        def close(self) -> None:
            events.append(f"close:{self.key}")

    sources = [FakeSource("front"), FakeSource("wrist")]

    def start(source: FakeSource, **_kwargs) -> None:
        events.append(f"start:{source.key}")
        if source.key == "wrist":
            raise TimeoutError("native startup timeout")

    with (
        patch.object(
            capture_realsense_intrinsics,
            "load_realsense_camera_config_map",
            return_value={"front": "front", "wrist": "wrist"},
        ),
        patch.object(
            capture_realsense_intrinsics,
            "RealSenseRGBSource",
            side_effect=sources,
        ),
        patch.object(
            capture_realsense_intrinsics,
            "start_real_sense_source_with_containment",
            side_effect=start,
        ),
        pytest.raises(TimeoutError, match="native startup timeout"),
    ):
        capture_realsense_intrinsics.main(
            ["--camera-config", "unused.json", "--warmup-sec", "0"]
        )

    assert events == ["start:front", "start:wrist", "close:front"]


def test_no_motion_bundle_probe_disables_inherited_front_reset() -> None:
    from teleop_runtime.tools.field_validation import (  # noqa: PLC0415
        camera_tactile_bundle_capture_probe as probe,
    )

    config = RealSenseRGBCameraConfig(
        serial="FRONT123",
        startup_hardware_reset_on_warmup_failure=True,
    )
    with (
        patch.object(probe, "load_realsense_camera_config_map", return_value={"front": config}),
        patch.object(probe, "RealSenseRGBSource", side_effect=lambda item: item),
    ):
        sources = probe._camera_sources(Path("unused.json"), startup_grace_sec=15.0)

    assert sources["front"] is config
    assert config.startup_hardware_reset_on_warmup_failure is False
    assert config.startup_hardware_reset_on_failure is False
    assert config.startup_hardware_reset_retry_attempts == 0


def test_no_motion_bundle_probe_does_not_retry_native_start_timeout() -> None:
    from teleop_runtime.tools.field_validation import (  # noqa: PLC0415
        camera_tactile_bundle_capture_probe as probe,
    )

    source = SimpleNamespace(close=lambda: pytest.fail("timed-out source was closed"))
    with patch.object(
        probe,
        "start_real_sense_source_with_containment",
        side_effect=TimeoutError("native call blocked"),
    ) as start:
        with pytest.raises(TimeoutError, match="native call blocked"):
            probe._start_camera_no_reset(
                "front",
                source,
                attempts=3,
                retry_delay_sec=0.0,
                startup_timeout_sec=20.0,
            )
    start.assert_called_once()


def test_wrist_depth_smoke_clears_inherited_hardware_reset() -> None:
    from utils.dataset_postprocess.wrist_cam_adjust import (  # noqa: PLC0415
        smoke_wrist_depth_capture as smoke,
    )

    config = RealSenseRGBCameraConfig(
        serial="FRONT123",
        startup_hardware_reset_on_warmup_failure=True,
    )
    args = SimpleNamespace(
        camera_config="unused.json",
        camera=["front"],
        depth_stream_fps=15,
        wait_timeout_ms=1000,
        stale_after_sec=2.0,
    )
    with patch.object(
        smoke,
        "_runtime_camera_api",
        return_value=(RealSenseRGBSource, lambda _: {"front": config}),
    ):
        selected = smoke._selected_configs(args)

    assert selected["front"] is config
    assert config.startup_hardware_reset_on_warmup_failure is False
    assert config.startup_hardware_reset_on_failure is False
    assert config.startup_hardware_reset_retry_attempts == 0


def test_realsense_bounded_warmup_uses_full_grace_not_legacy_frame_count() -> None:
    fake_rs = _FakeRealSense()
    fake_rs.wait_errors = [RuntimeError("Frame didn't arrive within 1000")] * 10
    source = RealSenseRGBSource(
        RealSenseRGBCameraConfig(
            serial="FRONT123",
            startup_reset=False,
            startup_warmup_frames=2,
            startup_wait_timeout_ms=1000,
            startup_frame_grace_sec=5.0,
            startup_warmup_policy=REALSENSE_STARTUP_WARMUP_BOUNDED_FIRST_FRAME,
        )
    )

    with pytest.raises(RuntimeError, match="attempts=5"):
        source._start_pipeline(fake_rs)

    assert [pipeline.wait_timeouts for pipeline in fake_rs.pipelines] == [[1000] * 5]


def test_realsense_bounded_policy_keeps_explicit_connection_error_reset_retry() -> None:
    fake_rs = _FakeRealSense()
    target_device = _FakeHardwareDevice("LEFT123")
    fake_rs.devices = [target_device]
    fake_rs.start_errors = [RuntimeError("Device or resource busy")]
    source = RealSenseRGBSource(
        RealSenseRGBCameraConfig(
            serial="LEFT123",
            camera_key="left_wrist",
            startup_reset=False,
            startup_warmup_frames=1,
            startup_warmup_policy=REALSENSE_STARTUP_WARMUP_BOUNDED_FIRST_FRAME,
            startup_hardware_reset_sleep_sec=0.0,
        )
    )

    pipeline, _ = source._start_pipeline(fake_rs)

    assert pipeline is fake_rs.pipelines[1]
    assert len(fake_rs.pipelines) == 2
    assert target_device.hardware_reset_calls == 1


def test_realsense_warmup_honours_shutdown_request() -> None:
    fake_rs = _FakeRealSense()
    source = RealSenseRGBSource(
        RealSenseRGBCameraConfig(
            serial="FRONT123",
            startup_reset=False,
            startup_warmup_frames=10,
        )
    )
    pipeline = fake_rs.pipeline()
    source._shutdown_event.set()

    with pytest.raises(RuntimeError, match="startup_cancelled"):
        source._warmup_pipeline(pipeline)

    assert pipeline.wait_timeouts == []


def test_realsense_startup_reset_defaults_are_serialized() -> None:
    payload = RealSenseRGBCameraConfig(serial="ABC").to_dict()

    assert payload["color_format"] == "yuyv"
    assert payload["wait_timeout_ms"] == 1000
    assert payload["startup_wait_timeout_ms"] == 5000
    assert payload["startup_frame_grace_sec"] == 5.0
    assert payload["startup_reset"] is True
    assert payload["restart_on_runtime_error"] is True
    assert payload["startup_reset_sleep_sec"] == 0.2
    assert payload["startup_warmup_frames"] == 10
    assert payload["startup_warmup_policy"] == "bounded_first_frame"
    assert payload["startup_hardware_reset"] is False
    assert payload["startup_hardware_reset_on_failure"] is True
    assert payload["startup_hardware_reset_retry_attempts"] == 1
    assert payload["startup_hardware_reset_sleep_sec"] == 2.0
    assert payload["startup_hardware_reset_reconnect_timeout_sec"] == 6.0
    assert payload["startup_hardware_reset_reconnect_poll_sec"] == 0.25
    assert payload["restart_on_black_frames"] is True
    assert payload["restart_after_black_frames"] == 5
    assert payload["accept_near_zero_frames"] is False
    assert payload["startup_software_retry_attempts"] == 0
    assert payload["startup_software_retry_sleep_sec"] == 0.5
    assert payload["require_complete_frameset"] is False
    assert payload["restart_on_incomplete_frameset"] is False
    assert payload["restart_after_incomplete_framesets"] == 5
    assert payload["restart_max_attempts_without_complete_frame"] == 0


def test_realsense_startup_warmup_policy_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="startup_warmup_policy must be one of"):
        RealSenseRGBCameraConfig(
            serial="ABC",
            startup_warmup_policy="not_a_policy",
        )


def test_realsense_near_zero_frames_are_dropped_and_restart_pipeline() -> None:
    source = RealSenseRGBSource(
        RealSenseRGBCameraConfig(
            serial="FRONT123",
            restart_after_black_frames=2,
            black_frame_max_value=1,
        )
    )
    black = np.zeros((48, 64, 3), dtype=np.uint8)
    low_light_but_valid = np.zeros_like(black)
    low_light_but_valid[::8, ::8] = 3

    with patch.object(source, "_restart_pipeline") as restart:
        assert source._handle_black_frame(black) is True
        restart.assert_not_called()
        assert source._handle_black_frame(black) is True
        restart.assert_called_once()

    assert source._handle_black_frame(low_light_but_valid) is False
    info = source.device_info()
    assert info["black_frame_count"] == 2
    assert info["consecutive_black_frame_count"] == 0
    assert info["last_sampled_max_value"] == 3


def test_realsense_reviewed_near_zero_frames_are_published_without_error() -> None:
    source = RealSenseRGBSource(
        RealSenseRGBCameraConfig(
            serial="FRONT123",
            accept_near_zero_frames=True,
            restart_after_black_frames=1,
        )
    )
    black = np.zeros((48, 64, 3), dtype=np.uint8)

    with patch.object(source, "_restart_pipeline") as restart:
        assert source._handle_black_frame(black) is False

    restart.assert_not_called()
    info = source.device_info()
    assert info["black_frame_count"] == 1
    assert info["consecutive_black_frame_count"] == 1
    assert info["last_sampled_max_value"] == 0
    assert info["last_error"] is None


def test_realsense_software_startup_retry_rebuilds_without_hardware_reset() -> None:
    fake_rs = _FakeRealSense()
    target_device = _FakeHardwareDevice("FRONT123")
    fake_rs.devices = [target_device]
    fake_rs.wait_errors = [RuntimeError("Frame didn't arrive within 1000")]
    source = RealSenseRGBSource(
        RealSenseRGBCameraConfig(
            serial="FRONT123",
            startup_reset=False,
            startup_warmup_frames=1,
            startup_frame_grace_sec=1.0,
            startup_wait_timeout_ms=1000,
            startup_software_retry_attempts=1,
            startup_software_retry_sleep_sec=0.25,
            startup_hardware_reset_on_failure=False,
        )
    )

    with patch.object(source._shutdown_event, "wait", return_value=False) as wait_mock:
        pipeline, _ = source._start_pipeline(fake_rs)

    assert pipeline is fake_rs.pipelines[1]
    assert len(fake_rs.pipelines) == 2
    assert fake_rs.pipelines[0].stop_calls == 1
    assert target_device.hardware_reset_calls == 0
    wait_mock.assert_called_once_with(0.25)


def test_realsense_software_startup_retry_sleep_honours_shutdown() -> None:
    fake_rs = _FakeRealSense()
    fake_rs.wait_errors = [RuntimeError("Frame didn't arrive within 1000")]
    source = RealSenseRGBSource(
        RealSenseRGBCameraConfig(
            serial="FRONT123",
            startup_reset=False,
            startup_warmup_frames=1,
            startup_frame_grace_sec=1.0,
            startup_wait_timeout_ms=1000,
            startup_software_retry_attempts=1,
            startup_software_retry_sleep_sec=0.25,
            startup_hardware_reset_on_failure=False,
        )
    )

    with (
        patch.object(source._shutdown_event, "wait", return_value=True),
        pytest.raises(RuntimeError, match="startup_cancelled"),
    ):
        source._start_pipeline(fake_rs)

    assert len(fake_rs.pipelines) == 1


def test_realsense_startup_retry_override_can_disable_configured_retries() -> None:
    fake_rs = _FakeRealSense()
    fake_rs.start_errors = [RuntimeError("Device or resource busy")]
    source = RealSenseRGBSource(
        RealSenseRGBCameraConfig(
            serial="FRONT123",
            startup_reset=False,
            startup_warmup_frames=0,
            startup_software_retry_attempts=2,
            startup_hardware_reset_on_failure=False,
        )
    )

    with pytest.raises(RuntimeError, match="configured profile was preserved"):
        source._start_pipeline(fake_rs, software_retry_attempts=0)

    assert len(fake_rs.pipelines) == 1


def test_agent_gateway_front_camera_explicitly_allows_reviewed_dark_frames() -> None:
    configs = realsense_rgb.load_realsense_camera_config_map(
        Path("teleop_runtime/configs/devices/agent_gateway_realsense.json")
    )

    assert configs["front"].accept_near_zero_frames is True
    assert configs["front"].startup_software_retry_attempts == 2
    assert configs["front"].startup_software_retry_sleep_sec == 0.5
    assert configs["front"].startup_hardware_reset is False
    assert configs["front"].startup_hardware_reset_on_failure is False
    for camera_key in ("left_wrist", "right_wrist"):
        assert configs[camera_key].accept_near_zero_frames is False
        assert configs[camera_key].startup_software_retry_attempts == 0


def test_realsense_frame_timeouts_mark_stale_and_request_restart() -> None:
    source = RealSenseRGBSource(
        RealSenseRGBCameraConfig(
            serial="LEFT123",
            camera_key="left_wrist",
            enable_depth=True,
            stale_after_sec=0.01,
            startup_frame_grace_sec=0.0,
            restart_after_timeouts=2,
            restart_on_stale_timeout=True,
        )
    )
    with source._lock:
        source._started_at = time.time() - 1.0

    message = "frame didn't arrive within 1000ms"
    assert source._set_transient_or_stale_timeout_error(message) is False
    assert source._set_transient_or_stale_timeout_error(message) is True

    info = source.device_info()
    assert info["timeout_count"] == 2
    assert info["consecutive_timeout_count"] == 2
    assert str(info["last_error"]).startswith("frame_timeout_stale:")
    assert message in str(info["last_runtime_error"])


def test_realsense_frame_timeout_before_stale_is_runtime_warning_only() -> None:
    source = RealSenseRGBSource(
        RealSenseRGBCameraConfig(
            serial="LEFT123",
            camera_key="left_wrist",
            stale_after_sec=60.0,
            restart_after_timeouts=1,
            restart_on_stale_timeout=True,
        )
    )
    with source._lock:
        source._started_at = time.time()

    message = "frame didn't arrive within 1000ms"
    assert source._set_transient_or_stale_timeout_error(message) is False

    info = source.device_info()
    assert info["timeout_count"] == 1
    assert info["consecutive_timeout_count"] == 1
    assert info["last_error"] is None
    assert message in str(info["last_runtime_error"])


def test_realsense_first_frame_grace_suppresses_restart_before_startup_deadline() -> None:
    source = RealSenseRGBSource(
        RealSenseRGBCameraConfig(
            serial="FRONT123",
            stale_after_sec=0.01,
            startup_frame_grace_sec=5.0,
            restart_after_timeouts=1,
            restart_on_stale_timeout=True,
        )
    )
    with source._lock:
        source._started_at = time.time()

    message = "frame didn't arrive within 1000ms"
    assert source._set_transient_or_stale_timeout_error(message) is False
    info = source.device_info()
    assert info["timeout_count"] == 1
    assert info["consecutive_timeout_count"] == 1
    assert info["last_error"] is None

    with source._lock:
        source._started_at = time.time() - 6.0
    assert source._set_transient_or_stale_timeout_error(message) is True


def test_realsense_classifies_disconnect_errors_for_pipeline_restart() -> None:
    assert RealSenseRGBSource._is_recoverable_runtime_error("Device disconnected")
    assert RealSenseRGBSource._is_recoverable_runtime_error("UVC backend failure")
    assert RealSenseRGBSource._is_recoverable_runtime_error(
        "wait_for_frames cannot be called before start()"
    )
    assert not RealSenseRGBSource._is_recoverable_runtime_error("bad image shape")


def test_realsense_bounded_warmup_requires_color_and_depth_together() -> None:
    fake_rs = _FakeRealSense()
    fake_rs.wait_frames = [
        _FakeFrameSet(color=True, depth=False),
        _FakeFrameSet(color=False, depth=True),
        _FakeFrameSet(color=True, depth=True),
    ]
    source = RealSenseRGBSource(
        RealSenseRGBCameraConfig(
            serial="LEFT123",
            camera_key="left_wrist",
            enable_depth=True,
            require_complete_frameset=True,
            startup_reset=False,
            startup_warmup_frames=1,
            startup_frame_grace_sec=3.0,
            startup_wait_timeout_ms=1000,
            wait_timeout_ms=1000,
        )
    )

    pipeline, _ = source._start_pipeline(fake_rs)

    assert pipeline.wait_timeouts == [1000, 1000, 1000]


def test_realsense_compatibility_warmup_accepts_historical_any_frameset() -> None:
    fake_rs = _FakeRealSense()
    fake_rs.wait_frames = [_FakeFrameSet(color=True, depth=False)]
    source = RealSenseRGBSource(
        RealSenseRGBCameraConfig(
            serial="LEFT123",
            camera_key="left_wrist",
            enable_depth=True,
            require_complete_frameset=False,
            startup_reset=False,
            startup_warmup_frames=1,
        )
    )

    pipeline, _ = source._start_pipeline(fake_rs)

    assert pipeline.wait_timeouts == [1000]


def test_realsense_bounded_warmup_rejects_alternating_color_and_depth() -> None:
    fake_rs = _FakeRealSense()
    fake_rs.wait_frames = [
        _FakeFrameSet(color=True, depth=False),
        _FakeFrameSet(color=False, depth=True),
        _FakeFrameSet(color=True, depth=False),
    ]
    source = RealSenseRGBSource(
        RealSenseRGBCameraConfig(
            serial="LEFT123",
            camera_key="left_wrist",
            enable_depth=True,
            require_complete_frameset=True,
            startup_reset=False,
            startup_warmup_frames=1,
            startup_frame_grace_sec=3.0,
            startup_wait_timeout_ms=1000,
            wait_timeout_ms=1000,
        )
    )

    with pytest.raises(RuntimeError, match="incomplete_frameset"):
        source._start_pipeline(fake_rs)

    assert fake_rs.pipelines[0].wait_timeouts == [1000, 1000, 1000]


def test_realsense_incomplete_framesets_are_observable_and_request_restart() -> None:
    source = RealSenseRGBSource(
        RealSenseRGBCameraConfig(
            serial="LEFT123",
            camera_key="left_wrist",
            enable_depth=True,
            restart_on_incomplete_frameset=True,
            restart_after_incomplete_framesets=2,
        )
    )
    now = time.time()

    assert source._record_incomplete_frameset(
        _FakeFrame(),
        None,
        received_at=now,
    ) is False
    assert source._record_incomplete_frameset(
        None,
        _FakeFrame(),
        received_at=now + 0.01,
    ) is True

    info = source.device_info()
    assert info["incomplete_frameset_count"] == 2
    assert info["consecutive_incomplete_frameset_count"] == 2
    assert info["missing_color_frame_count"] == 1
    assert info["missing_depth_frame_count"] == 1
    assert str(info["last_error"]).startswith("incomplete_frameset:")

    source._record_complete_frameset(received_at=now + 0.02)
    assert source.device_info()["consecutive_incomplete_frameset_count"] == 0


def test_realsense_restart_circuit_faults_before_unbounded_reopen() -> None:
    source = RealSenseRGBSource(
        RealSenseRGBCameraConfig(
            serial="LEFT123",
            camera_key="left_wrist",
            restart_max_attempts_without_complete_frame=2,
        )
    )
    shutdown_event = threading.Event()
    with source._lock:
        source._running = True
        source._lifecycle_state = realsense_rgb._LIFECYCLE_RUNNING
        source._generation = 7
        source._restart_attempts_without_complete_frame = 2
        source._shutdown_event = shutdown_event

    source._restart_pipeline(
        "pipeline_missing",
        reason="pipeline_missing",
        generation=7,
        shutdown_event=shutdown_event,
    )

    info = source.device_info()
    assert info["lifecycle_state"] == "faulted"
    assert info["running"] is False
    assert str(info["last_error"]).startswith("restart_circuit_open:")
    assert shutdown_event.is_set()


def test_realsense_native_start_edges_are_serialized_across_sources() -> None:
    counter_lock = threading.Lock()
    active_starts = 0
    max_active_starts = 0

    class Pipeline:
        def stop(self) -> None:
            return None

    class Worker:
        def __init__(self, name: str) -> None:
            self.name = name

        def start(self) -> None:
            return None

        def join(self, timeout=None) -> None:
            del timeout

        def is_alive(self) -> bool:
            return False

    class Source(RealSenseRGBSource):
        def _start_pipeline(self, rs):
            del rs
            nonlocal active_starts, max_active_starts
            with counter_lock:
                active_starts += 1
                max_active_starts = max(max_active_starts, active_starts)
            time.sleep(0.02)
            with counter_lock:
                active_starts -= 1
            return Pipeline(), object()

        def _device_info_from_profile(self, profile):
            del profile
            return {}

    sources = [
        Source(RealSenseRGBCameraConfig(serial=f"SERIAL{index}"))
        for index in range(2)
    ]
    errors: list[BaseException] = []

    def start_source(source: Source) -> None:
        try:
            source.start()
        except BaseException as exc:  # noqa: BLE001 - retained for thread assertion.
            errors.append(exc)

    callers = [
        threading.Thread(target=start_source, args=(source,)) for source in sources
    ]
    with (
        patch.object(realsense_rgb, "_load_pyrealsense2", return_value=object()),
        patch.object(
            realsense_rgb.threading,
            "Thread",
            side_effect=lambda **kwargs: Worker(str(kwargs["name"])),
        ),
    ):
        for caller in callers:
            caller.start()
        for caller in callers:
            caller.join()

    assert errors == []
    assert max_active_starts == 1
    for source in sources:
        source.close()


def test_realsense_close_fails_fast_when_startup_owns_lifecycle_lock() -> None:
    source = RealSenseRGBSource(RealSenseRGBCameraConfig(serial="SERIAL"))
    lock_acquired = threading.Event()
    release_lock = threading.Event()

    def hold_lifecycle_lock() -> None:
        with source._lifecycle_lock:
            lock_acquired.set()
            release_lock.wait(timeout=2.0)

    holder = threading.Thread(target=hold_lifecycle_lock)
    holder.start()
    assert lock_acquired.wait(timeout=1.0)
    started = time.perf_counter()
    with pytest.raises(RuntimeError, match="lifecycle_lock"):
        source.close()
    assert time.perf_counter() - started < 0.5
    assert source.device_info()["lifecycle_state"] == "faulted"
    release_lock.set()
    holder.join(timeout=1.0)
    assert not holder.is_alive()


def test_real_teleop_camera_start_timeout_faults_without_waiting_for_worker() -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    class Source(RealSenseRGBSource):
        def start(self) -> None:
            started.set()
            try:
                release.wait(timeout=2.0)
            finally:
                finished.set()

    source = Source(RealSenseRGBCameraConfig(serial="SERIAL"))
    with pytest.raises(TimeoutError, match="native SDK worker remains blocked"):
        camera_runtime._start_camera_source_with_containment(
            source,
            timeout_sec=0.01,
        )

    assert started.is_set()
    assert source.device_info()["lifecycle_state"] == "faulted"
    release.set()
    assert finished.wait(timeout=1.0)
    source.close()


def test_real_teleop_camera_timeout_suppresses_deferred_retry() -> None:
    def source(camera_key: str) -> RealSenseRGBSource:
        return RealSenseRGBSource(
            RealSenseRGBCameraConfig(
                serial=camera_key,
                camera_key=camera_key,
                startup_warmup_policy=REALSENSE_STARTUP_WARMUP_BOUNDED_FIRST_FRAME,
            )
        )

    sources = {
        "front": source("front"),
        "left_wrist": source("left_wrist"),
        "right_wrist": source("right_wrist"),
    }
    calls: list[str] = []

    def guarded_start(camera_source: RealSenseRGBSource, **kwargs: object) -> None:
        del kwargs
        calls.append(camera_source.camera_key)
        if camera_source.camera_key == "front":
            raise RuntimeError(
                "startup_warmup_failed:policy=bounded_first_frame:no_frames_within_sec=5"
            )
        if camera_source.camera_key == "left_wrist":
            raise TimeoutError("native SDK worker remains blocked")

    with patch.object(camera_runtime, "_start_camera_source_with_containment", guarded_start):
        errors = camera_runtime._start_camera_sources_best_effort(
            sources,
            stagger_sec=0.0,
            deferred_retry_cooldown_sec=0.0,
        )

    assert calls == ["front", "left_wrist"]
    assert set(errors) == {"front", "left_wrist", "right_wrist"}
    assert "deferred_startup_retry_failed" not in errors["front"]
    assert errors["right_wrist"].startswith("startup_aborted_after_timeout:")
