from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from robot_motion.deploy.observation import build_deployment_observation
from robot_motion.runtime import EffectorCommand, TeleopCommand
from teleop_runtime.collection.build_collect_components import CollectorComponents
from teleop_runtime.collection.sample_assembler import (
    CollectorSampleBuilder,
    CollectorSampleBuilderConfig,
)
from teleop_runtime.collection.video_frame import NearestFrameBuffer, VideoFrame
from teleop_runtime.diagnostics import TeleopRunState
from teleop_runtime.schemas import TeleopObservation
from teleop_runtime.session import SessionStepResult
from teleop_runtime.real_teleop.recording_admission import recording_warmup_status


def _frame(
    timestamp: float,
    *,
    frame_index: int = 0,
    available_timestamp: float | None = None,
) -> VideoFrame:
    if available_timestamp is None:
        available_timestamp = float(timestamp)
    return VideoFrame(
        image=np.full((2, 2, 3), frame_index, dtype=np.uint8),
        source_timestamp=float(timestamp),
        receive_timestamp=float(timestamp),
        available_timestamp=available_timestamp,
        frame_index=int(frame_index),
    )


def _step(timestamp: float) -> SessionStepResult:
    return SessionStepResult(
        timestamp=float(timestamp),
        state=TeleopRunState.RUNNING,
        control_hz=15.0,
        observation=TeleopObservation(timestamp=float(timestamp)),
    )


def _dispatched_arm_step(timestamp: float) -> SessionStepResult:
    command = TeleopCommand(timestamp=float(timestamp))
    command.add_command(
        EffectorCommand(
            effector_name="left_arm",
            side="left",
            command_type="joint_vector",
            value=[0.0] * 6,
        )
    )
    return SessionStepResult(
        timestamp=float(timestamp),
        state=TeleopRunState.RUNNING,
        control_hz=15.0,
        observation=TeleopObservation(timestamp=float(timestamp)),
        command=command,
        command_dispatched=True,
    )


def test_collection_rejects_cached_frame_after_100ms_and_resumes_without_index_gap() -> None:
    buffer = NearestFrameBuffer(keep_last=2)
    buffer.push(_frame(9.95, frame_index=1))
    builder = CollectorSampleBuilder(
        CollectorSampleBuilderConfig(
            active_sides=(),
            camera_buffers={"front": buffer},
            camera_frame_max_age_sec=0.1,
        )
    )
    components = CollectorComponents(session=None, backends={})

    first = builder(_step(10.0), components)
    assert first is not None
    assert first.index == 0
    assert builder.last_camera_rejection is None

    assert builder(_step(10.051), components) is None
    assert builder.last_camera_rejection == "camera_frame_stale:front:0.101000"
    assert builder.sample_index == 1
    assert builder.frame_index == 1

    buffer.push(_frame(10.051, frame_index=2))
    resumed = builder(_step(10.052), components)
    assert resumed is not None
    assert resumed.index == 1
    assert resumed.frame_index == 1
    assert builder.last_camera_rejection is None


def test_collection_rejects_missing_required_camera_frame() -> None:
    builder = CollectorSampleBuilder(
        CollectorSampleBuilderConfig(
            active_sides=(),
            camera_buffers={"front": NearestFrameBuffer()},
            camera_frame_max_age_sec=0.1,
        )
    )

    assert builder(_step(10.0), CollectorComponents(session=None, backends={})) is None
    assert builder.last_camera_rejection == "camera_frame_unavailable:front"
    assert builder.sample_index == 0


def test_recording_resume_gate_uses_camera_availability_age() -> None:
    """Resume must not reopen a segment the collector will reject immediately."""

    buffer = NearestFrameBuffer(keep_last=2)
    # The frame was captured recently, but became available to the collector
    # 110 ms before T0.  CollectorSampleBuilder therefore rejects it at the
    # 100 ms contract boundary.
    buffer.push(_frame(9.95, frame_index=1, available_timestamp=9.89))
    source = SimpleNamespace(frame_buffer=lambda: buffer)
    args = SimpleNamespace(no_camera=False, ignore_camera=(), ignore_tactile=())
    components = SimpleNamespace(active_sides=())
    clutch = SimpleNamespace(
        hold_active=lambda: False,
        publish_camera_status=lambda **_kwargs: {},
    )
    step = _dispatched_arm_step(10.0)

    ready, message = recording_warmup_status(
        10.0,
        step_result=step,
        args=args,
        components=components,
        clutch_input=clutch,
        dataset_recorder=None,
        camera_sources={"front": source},
        preview_camera_sources={},
        camera_previewer=None,
        recording_auto_stop_reason_fn=lambda _state: None,
        gripper_warmup_probe_status=lambda **_kwargs: (True, "ready"),
        camera_status_blocking_message=lambda _status: None,
    )

    assert ready is False
    assert "front" in message
    assert "0.11" in message


def test_recording_resume_gate_rejects_future_availability_timestamp() -> None:
    buffer = NearestFrameBuffer(keep_last=2)
    buffer.push(_frame(9.99, frame_index=1, available_timestamp=10.10))
    source = SimpleNamespace(frame_buffer=lambda: buffer)
    args = SimpleNamespace(no_camera=False, ignore_camera=(), ignore_tactile=())
    step = _dispatched_arm_step(10.0)

    ready, message = recording_warmup_status(
        10.0,
        step_result=step,
        args=args,
        components=SimpleNamespace(active_sides=()),
        clutch_input=SimpleNamespace(
            hold_active=lambda: False,
            publish_camera_status=lambda **_kwargs: {},
        ),
        dataset_recorder=None,
        camera_sources={"front": source},
        preview_camera_sources={},
        camera_previewer=None,
        recording_auto_stop_reason_fn=lambda _state: None,
        gripper_warmup_probe_status=lambda **_kwargs: (True, "ready"),
        camera_status_blocking_message=lambda _status: None,
    )

    assert ready is False
    assert message == "等待相机 front 时间戳稳定。"


def test_collection_camera_lookup_requires_source_and_availability_before_t0() -> None:
    buffer = NearestFrameBuffer(keep_last=3)
    buffer.push(_frame(9.90, frame_index=1, available_timestamp=9.95))
    buffer.push(_frame(9.99, frame_index=2, available_timestamp=10.01))

    selected = buffer.get_latest_before_or_at(10.0)

    assert selected is not None
    assert selected.frame_index == 1
    assert selected.source_timestamp <= 10.0
    assert selected.available_timestamp <= 10.0


class _CameraSource:
    def __init__(self, frame: VideoFrame | None):
        self.frame = frame
        self._buffer = NearestFrameBuffer()
        if frame is not None:
            self._buffer.push(frame)

    def get_latest_frame(self) -> VideoFrame | None:
        return self.frame

    def frame_buffer(self) -> NearestFrameBuffer:
        return self._buffer


def test_deployment_observation_marks_frame_over_100ms_stale() -> None:
    observation = build_deployment_observation(
        active_sides=(),
        arms_enabled=False,
        hands_enabled=False,
        step_index=0,
        arm_states={},
        hand_states={},
        camera_sources={"front": _CameraSource(_frame(9.8))},
        timestamp=10.0,
        require_images=True,
        camera_frame_max_age_sec=0.1,
    )

    assert observation.image_ages_sec["front"] == pytest.approx(0.2)
    assert observation.metadata["camera_frame_available_timestamp_by_key"]["front"] == (
        pytest.approx(9.8)
    )
    assert observation.metadata["camera_causal_snapshot_contract"] == (
        "source_timestamp<=T0_and_available_timestamp<=T0"
    )
    assert any(fault.startswith("camera_frame_stale:front:") for fault in observation.faults)
    assert observation.metadata["camera_frame_max_age_sec"] == 0.1


def test_deployment_camera_margin_accepts_normal_15hz_jitter() -> None:
    observation = build_deployment_observation(
        active_sides=(),
        arms_enabled=False,
        hands_enabled=False,
        step_index=0,
        arm_states={},
        hand_states={},
        camera_sources={"front": _CameraSource(_frame(9.82))},
        timestamp=10.0,
        require_images=True,
        camera_frame_max_age_sec=0.2,
    )

    assert observation.faults == []
    assert observation.metadata["camera_frame_max_age_sec"] == 0.2


def test_deployment_observation_requires_a_configured_fresh_camera() -> None:
    missing = build_deployment_observation(
        active_sides=(),
        arms_enabled=False,
        hands_enabled=False,
        step_index=0,
        arm_states={},
        hand_states={},
        timestamp=10.0,
        require_images=True,
    )
    assert "camera_frame_unavailable:no_configured_camera" in missing.faults

    fresh = build_deployment_observation(
        active_sides=(),
        arms_enabled=False,
        hands_enabled=False,
        step_index=1,
        arm_states={},
        hand_states={},
        camera_sources={"front": _CameraSource(_frame(9.95))},
        timestamp=10.0,
        require_images=True,
        camera_frame_max_age_sec=0.1,
    )
    assert fresh.faults == []
