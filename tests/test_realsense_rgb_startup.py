from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from teleop_runtime.sources.camera import realsense_rgb
from teleop_runtime.sources.camera.realsense_rgb import RealSenseRGBCameraConfig, RealSenseRGBSource


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

    def wait_for_frames(self, *, timeout_ms: int) -> object:
        self.wait_timeouts.append(timeout_ms)
        if self.owner is not None and self.owner.wait_errors:
            raise self.owner.wait_errors.pop(0)
        return object()


class _FakeRealSense:
    def __init__(self) -> None:
        self.stream = SimpleNamespace(color="color", depth="depth")
        self.format = SimpleNamespace(bgr8="bgr8", z16="z16")
        self.camera_info = SimpleNamespace(serial_number="serial_number")
        self.last_pipeline: _FakePipeline | None = None
        self.pipelines: list[_FakePipeline] = []
        self.last_config: _FakeStreamConfig | None = None
        self.align_requests: list[object] = []
        self.devices: list[_FakeHardwareDevice] = []
        self.start_errors: list[RuntimeError] = []
        self.wait_errors: list[RuntimeError] = []

    def pipeline(self) -> _FakePipeline:
        self.last_pipeline = _FakePipeline(self)
        self.pipelines.append(self.last_pipeline)
        return self.last_pipeline

    def config(self) -> _FakeStreamConfig:
        self.last_config = _FakeStreamConfig()
        return self.last_config

    def align(self, stream: object) -> object:
        self.align_requests.append(stream)
        return object()

    def context(self) -> object:
        return SimpleNamespace(query_devices=lambda: list(self.devices))


class _FakeHardwareDevice:
    def __init__(self, serial: str) -> None:
        self.serial = serial
        self.hardware_reset_calls = 0

    def get_info(self, info_key: object) -> str:
        if info_key != "serial_number":
            raise KeyError(info_key)
        return self.serial

    def hardware_reset(self) -> None:
        self.hardware_reset_calls += 1


def test_realsense_startup_resets_and_warms_up_front_without_depth() -> None:
    fake_rs = _FakeRealSense()
    source = RealSenseRGBSource(
        RealSenseRGBCameraConfig(
            serial="FRONT123",
            camera_key="front",
            width=640,
            height=480,
            fps=15,
            startup_warmup_frames=3,
        )
    )

    with patch.object(realsense_rgb.time, "sleep") as sleep_mock:
        pipeline, _ = source._start_pipeline(fake_rs)

    assert pipeline is fake_rs.last_pipeline
    assert fake_rs.last_config is not None
    assert fake_rs.last_config.device == "FRONT123"
    assert fake_rs.last_config.streams == [("color", 640, 480, "bgr8", 15)]
    assert pipeline.stop_calls == 1
    assert pipeline.wait_timeouts == [1000, 1000, 1000]
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
        ("color", 640, 480, "bgr8", 15),
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

    with patch.object(realsense_rgb.time, "sleep") as sleep_mock:
        pipeline, _ = source._start_pipeline(fake_rs)

    assert pipeline.stop_calls == 0
    assert other_device.hardware_reset_calls == 0
    assert target_device.hardware_reset_calls == 1
    sleep_mock.assert_called_once_with(1.25)


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

    with patch.object(realsense_rgb.time, "sleep") as sleep_mock:
        pipeline, _ = source._start_pipeline(fake_rs)

    assert pipeline is fake_rs.pipelines[1]
    assert len(fake_rs.pipelines) == 2
    assert fake_rs.pipelines[0].stop_calls == 1
    assert target_device.hardware_reset_calls == 1
    sleep_mock.assert_called_once_with(1.25)


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

    with pytest.raises(RuntimeError, match="matching resolutions"):
        source._start_pipeline(fake_rs)

    assert len(fake_rs.pipelines) == 1
    assert fake_rs.pipelines[0].stop_calls == 1
    assert target_device.hardware_reset_calls == 0


def test_realsense_startup_retries_when_warmup_gets_no_frames() -> None:
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
            startup_hardware_reset_sleep_sec=1.25,
        )
    )

    with patch.object(realsense_rgb.time, "sleep") as sleep_mock:
        pipeline, _ = source._start_pipeline(fake_rs)

    assert pipeline is fake_rs.pipelines[1]
    assert len(fake_rs.pipelines) == 2
    assert fake_rs.pipelines[0].stop_calls == 1
    assert target_device.hardware_reset_calls == 1
    sleep_mock.assert_called_once_with(1.25)


def test_realsense_startup_reset_defaults_are_serialized() -> None:
    payload = RealSenseRGBCameraConfig(serial="ABC").to_dict()

    assert payload["startup_reset"] is True
    assert payload["startup_reset_sleep_sec"] == 0.2
    assert payload["startup_warmup_frames"] == 10
    assert payload["startup_hardware_reset"] is False
    assert payload["startup_hardware_reset_on_failure"] is True
    assert payload["startup_hardware_reset_retry_attempts"] == 1
    assert payload["startup_hardware_reset_sleep_sec"] == 2.0
    assert payload["startup_hardware_reset_reconnect_timeout_sec"] == 6.0
    assert payload["startup_hardware_reset_reconnect_poll_sec"] == 0.25


def test_realsense_frame_timeouts_mark_stale_and_request_restart() -> None:
    source = RealSenseRGBSource(
        RealSenseRGBCameraConfig(
            serial="LEFT123",
            camera_key="left_wrist",
            enable_depth=True,
            stale_after_sec=0.01,
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
