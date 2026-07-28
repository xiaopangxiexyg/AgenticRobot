from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from robot_motion.runtime import ObservationChannel
from robot_motion.runtime.observation_cut import ObservationSampleTracker
from teleop_runtime.action.human_pose_source import (
    HumanPoseChannelAdapter,
    HumanPoseChannelAdapterConfig,
    HumanPoseChannelBinding,
)


def _channel(
    name: str = "left_wrist",
    *,
    frame: str = "custom_tracker_world",
    x: float = 0.1,
) -> ObservationChannel:
    pose = np.eye(4)
    pose[0, 3] = x
    return ObservationChannel(
        channel_name=name,
        capability="pose",
        side="left",
        value=pose,
        frame_name=frame,
        timestamp=10.0,
        source_name="virtual-tracker",
        metadata={
            "measured_at": 9.9,
            "measured_clock_domain_id": "tracker.source",
        },
    )


def _adapter() -> HumanPoseChannelAdapter:
    return HumanPoseChannelAdapter(
        HumanPoseChannelAdapterConfig(
            adapter_id="virtual-human-pose",
            source_coordinate_binding_id="virtual-calibration:0",
            source_coordinate_binding_hash="virtual-calibration-hash:0",
            component_bindings=(
                HumanPoseChannelBinding(
                    component_id="left_arm",
                    channel_name="left_wrist",
                    source_controlled_frame_id="left_wrist_tracker",
                    expected_reference_frame_id="custom_tracker_world",
                ),
            ),
        )
    )


def _refs(channels: dict[str, ObservationChannel]):
    return ObservationSampleTracker().observe(
        channels,
        available_at=20.0,
        available_clock_domain_id="receiver.monotonic",
    )


def test_selective_channel_adapter_preserves_sample_and_frame_identity() -> None:
    pose_channel = _channel()
    extra_channel = ObservationChannel(
        channel_name="tactile",
        capability="tactile",
        side="left",
        value=np.array([1.0, 2.0]),
        frame_name=None,
        timestamp=10.0,
        source_name="virtual-tactile",
    )
    channels = {
        "left_wrist": pose_channel,
        "tactile": extra_channel,
    }

    result = _adapter().adapt(
        channels,
        sample_refs=_refs(channels),
        adapted_at=20.1,
        adapted_clock_domain_id="receiver.monotonic",
    )

    assert len(result.samples) == 1
    sample = result.samples[0]
    assert sample.component_id == "left_arm"
    assert sample.source_reference_frame_id == "custom_tracker_world"
    assert sample.source_controlled_frame_id == "left_wrist_tracker"
    assert sample.measured_at == pytest.approx(9.9)
    assert sample.measured_clock_domain_id == "tracker.source"
    assert set(result.receipt.source_sample_id_by_component) == {"left_arm"}
    assert "tactile" not in result.receipt.source_sample_id_by_component


def test_channel_value_must_match_receiver_sample_reference() -> None:
    channel = _channel(x=0.1)
    channels = {"left_wrist": channel}
    refs = _refs(channels)
    channel.value[0, 3] = 0.2

    with pytest.raises(
        ValueError,
        match="payload does not match its sample reference",
    ):
        _adapter().adapt(
            channels,
            sample_refs=refs,
            adapted_at=20.1,
            adapted_clock_domain_id="receiver.monotonic",
        )


def test_channel_frame_must_match_declared_binding() -> None:
    channels = {
        "left_wrist": _channel(frame="another_tracker_world"),
    }

    with pytest.raises(ValueError, match="does not match 'custom_tracker_world'"):
        _adapter().adapt(
            channels,
            sample_refs=_refs(channels),
            adapted_at=20.1,
            adapted_clock_domain_id="receiver.monotonic",
        )


def test_invalid_or_stale_sample_is_not_adapted() -> None:
    channels = {"left_wrist": _channel()}
    refs = dict(_refs(channels))
    refs["left_wrist"] = replace(refs["left_wrist"], stale=True)

    with pytest.raises(ValueError, match="invalid or stale"):
        _adapter().adapt(
            channels,
            sample_refs=refs,
            adapted_at=20.1,
            adapted_clock_domain_id="receiver.monotonic",
        )


def test_missing_declared_channel_is_explicit() -> None:
    with pytest.raises(ValueError, match="Missing declared human pose channel"):
        _adapter().adapt(
            {},
            sample_refs={},
            adapted_at=20.1,
            adapted_clock_domain_id="receiver.monotonic",
        )
