from __future__ import annotations

import numpy as np
import pytest

from robot_motion.runtime.control_admission import (
    ControlAdmissionCoordinator,
    ControlCut,
)
from robot_motion.runtime.identity import ClockDomainDescriptor
from robot_motion.runtime.observation_cut import (
    ObservationBundle,
    ObservationCutRecord,
    ObservationSampleRef,
    ObservationSampleTracker,
    ReceiverObservationCutBuilder,
)
from robot_motion.runtime import ObservationChannel


def _sample(*, measured_at: float = 8.0, available_at: float = 9.0, admission_sequence: int = 3) -> ObservationSampleRef:
    return ObservationSampleRef(
        stream_id="camera.front.rgb",
        sample_id="rgb-frame-17",
        sequence=17,
        measured_at=measured_at,
        measured_clock_domain_id="control.monotonic",
        available_at=available_at,
        available_clock_domain_id="control.monotonic",
        payload_object_id="sha256:rgb-payload",
        control_admission_sequence=admission_sequence,
        content_metadata={"shape": (2, 2, 3), "dtype": "uint8"},
    )


def _record(cut: ControlCut, *, sequence: int = 4, sample: ObservationSampleRef | None = None) -> ObservationCutRecord:
    selected = _sample() if sample is None else sample
    return ObservationCutRecord.build(
        runtime_session_id="session-0",
        control_cut=cut,
        policy_node_id="vision-policy",
        policy_node_epoch=1,
        sequence=sequence,
        built_at=cut.cutoff_at + 0.01,
        built_clock_domain_id="control.monotonic",
        selected_sample_by_stream={selected.stream_id: selected},
        context_dependencies=("context-9",),
        observation_contract_hash="observation-contract-v1",
        policy_io_layout_hash="layout-v23",
        trigger_reason="fixed_schedule",
    )


def _cut(cutoff_at: float, sequence: int) -> ControlCut:
    return ControlCut(
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        clock_domain_id="control.monotonic",
        clock_reset_epoch=0,
        cutoff_at=cutoff_at,
        cutoff_admission_sequence=sequence,
    )


def test_observation_cut_rejects_future_or_uncommitted_samples() -> None:
    cut = _cut(10.0, 4)
    with pytest.raises(ValueError, match="future"):
        _record(cut, sample=_sample(measured_at=10.1))
    with pytest.raises(ValueError, match="future"):
        _record(cut, sample=_sample(available_at=10.1))
    with pytest.raises(ValueError, match="uncommitted"):
        _record(cut, sample=_sample(admission_sequence=5))


def test_reused_rgb_frame_keeps_stable_id_and_real_capture_time() -> None:
    sample = _sample()
    first = _record(_cut(10.0, 4), sequence=4, sample=sample)
    second = _record(_cut(10.05, 4), sequence=5, sample=sample)

    first_ref = first.selected_sample_by_stream[sample.stream_id]
    second_ref = second.selected_sample_by_stream[sample.stream_id]
    assert first_ref.sample_id == second_ref.sample_id == "rgb-frame-17"
    assert first_ref.measured_at == second_ref.measured_at == 8.0
    assert first.observation_cut_id != second.observation_cut_id
    assert first.semantic_content_hash != second.semantic_content_hash


def test_observation_bundle_separates_runtime_handle_from_stable_record() -> None:
    record = _record(_cut(10.0, 4))
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    bundle = ObservationBundle(record=record, payload_handles_by_stream={"camera.front.rgb": image})
    assert bundle.payload_handles_by_stream["camera.front.rgb"] is image
    assert "payload_handles_by_stream" not in record.__dict__
    with pytest.raises(TypeError):
        bundle.payload_handles_by_stream["new"] = image
    with pytest.raises(ValueError, match="no selected sample"):
        ObservationBundle(record=record, payload_handles_by_stream={"unknown": image})


def test_latest_value_sample_tracker_reuses_identity_until_source_sample_changes() -> None:
    tracker = ObservationSampleTracker()
    channel = ObservationChannel(
        channel_name="left_hand_q6",
        capability="joint_vector",
        side="left",
        value=np.asarray([0.1] * 6, dtype=np.float32),
        frame_name=None,
        timestamp=100.0,
        source_name="virtual-q6",
        metadata={"sample_sequence": 7},
    )
    first = tracker.observe(
        {"left_hand_q6": channel},
        available_at=10.0,
        available_clock_domain_id="control.monotonic",
    )["left_hand_q6"]
    repeated = tracker.observe(
        {"left_hand_q6": channel.copy()},
        available_at=10.1,
        available_clock_domain_id="control.monotonic",
    )["left_hand_q6"]

    assert repeated is first
    assert repeated.sample_id == first.sample_id
    assert repeated.sequence == 0
    assert repeated.available_at == 10.0
    assert repeated.content_metadata["upstream_sequence"] == "7"

    changed = channel.copy()
    changed.value = np.asarray([0.2] * 6, dtype=np.float32)
    second = tracker.observe(
        {"left_hand_q6": changed},
        available_at=10.2,
        available_clock_domain_id="control.monotonic",
    )["left_hand_q6"]
    assert second.sample_id != first.sample_id
    assert second.sequence == 1
    assert second.available_at == 10.2


def test_receiver_cut_admits_upstream_clock_once_and_reuses_sample_identity() -> None:
    coordinator = ControlAdmissionCoordinator(
        "robot-0",
        clock_domain=ClockDomainDescriptor(
            clock_domain_id="receiver.monotonic",
            runtime_session_id="session-0",
            boot_id="receiver-boot",
            reset_epoch=0,
        ),
        clock=lambda: 20.0,
    )
    source = ObservationSampleRef(
        stream_id="human.left_hand_q6",
        sample_id="q6-source-4",
        sequence=4,
        measured_at=100.0,
        measured_clock_domain_id="q6.device",
        available_at=12.0,
        available_clock_domain_id="teleop_session.monotonic",
        payload_object_id="q6-payload-4",
        content_metadata={"source_name": "virtual-q6"},
    )
    payload = np.arange(6, dtype=np.float32)
    builder = ReceiverObservationCutBuilder(
        producer_id="human-device",
        producer_epoch=2,
        observation_contract_hash="human-input-contract-v1",
        io_layout_hash="human-input-layout-v1",
    )

    first = builder.capture(
        coordinator,
        selected_source_samples={source.stream_id: source},
        payload_handles_by_stream={source.stream_id: payload},
        trigger_reason="human_command_emission",
        built_at=20.0,
    )
    second = builder.capture(
        coordinator,
        selected_source_samples={source.stream_id: source},
        payload_handles_by_stream={source.stream_id: payload},
        trigger_reason="human_command_emission",
        built_at=20.1,
    )

    first_ref = first.record.selected_sample_by_stream[source.stream_id]
    second_ref = second.record.selected_sample_by_stream[source.stream_id]
    assert first_ref is second_ref
    assert first_ref.sample_id == source.sample_id
    assert first_ref.measured_at == source.measured_at
    assert first_ref.measured_clock_domain_id == source.measured_clock_domain_id
    assert first_ref.available_at == 20.0
    assert first_ref.available_clock_domain_id == "receiver.monotonic"
    assert first_ref.content_metadata["upstream_available_at"] == 12.0
    assert first.record.observation_cut_id != second.record.observation_cut_id
