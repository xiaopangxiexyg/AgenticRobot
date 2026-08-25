from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from teleop_runtime.sources.camera import realsense_rgb
from teleop_runtime.sources.camera.realsense_rgb import (
    REALSENSE_STARTUP_WARMUP_BOUNDED_FIRST_FRAME,
    REALSENSE_STARTUP_WARMUP_LEGACY,
    RealSenseRGBCameraConfig,
    RealSenseRGBSource,
)


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

    def get_info(self, info_key: object) -> str:
        if info_key != "serial_number":
            raise KeyError(info_key)
        return self.serial

    def hardware_reset(self) -> None:
        self.hardware_reset_calls += 1


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

    with pytest.raises(RuntimeError, match="configured profile was preserved"):
        source._start_pipeline(fake_rs)

    assert len(fake_rs.pipelines) == 1
    assert fake_rs.pipelines[0].stop_calls == 1
    assert target_device.hardware_reset_calls == 0


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

    with pytest.raises(RuntimeError, match="policy=bounded_first_frame"):
        source._start_pipeline(fake_rs)

    assert len(fake_rs.pipelines) == 1
    assert fake_rs.pipelines[0].wait_timeouts == [1000] * 5
    assert fake_rs.pipelines[0].stop_calls == 1
    assert target_device.hardware_reset_calls == 0


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
