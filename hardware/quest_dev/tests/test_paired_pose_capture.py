from __future__ import annotations

import copy
import inspect
import socket
import threading
from dataclasses import dataclass, replace

import pytest

from hardware.quest_dev.features.controller_pose.protocol import (
    parse_controller_pose_sample,
)
from hardware.quest_dev.features.controller_pose.runtime import (
    ControllerPoseAcceptedEvent,
    ControllerPoseActiveSessionCut,
    ControllerPoseEventCut,
    ControllerPoseInvalidationEvent,
    ControllerPoseObserverCloseCut,
)
from hardware.quest_dev.features.visualization.paired_pose_capture import (
    CaptureTerminalError,
    ManualStaticPairedPoseCapture,
    PairedPoseCaptureBinding,
    PairedPoseCaptureEvidence,
    PairedPoseCaptureFinalization,
    QuestControllerCaptureBinding,
    StaticWindowCaptureProfile,
    StaticWindowRejectedError,
    parse_paired_pose_capture_evidence,
)
from hardware.quest_dev.protocol import PROTOCOL_VERSION
from robot_motion.action.execution_observation import (
    ExecutionFactReference,
    ExecutionMeasurementObservation,
)
from robot_motion.runtime.identity import ClockDomainDescriptor
from robot_motion.runtime.observation_cut import ObservationSampleRef
from robot_motion.runtime.pose_observation import (
    POSE_COORDINATE_PARENT_STAGE_V1,
    PoseCoordinateBinding,
    PoseMeasurementAcceptedEvent,
    PoseMeasurementInvalidationEvent,
    PoseMeasurementJournalCut,
    PoseMeasurementPublication,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]

_CLOCK_ID = "host-boot-1.monotonic"
_TOOL_HASH = "sha256:" + "1" * 64


@dataclass(frozen=True)
class _RobotPoseFixture:
    coordinate_binding: PoseCoordinateBinding


def _receiver_clock(
    *, runtime_session_id: str = "quest-capture-runtime-1"
) -> ClockDomainDescriptor:
    return ClockDomainDescriptor(
        clock_domain_id=_CLOCK_ID,
        runtime_session_id=runtime_session_id,
        boot_id="host-boot-1",
        reset_epoch=0,
        resolution_sec=1e-9,
    )


def _projection(
    *,
    source_epoch_id: str = "rm65-right-epoch-7",
    receipt_clock_domain_id: str = _CLOCK_ID,
) -> _RobotPoseFixture:
    coordinate = PoseCoordinateBinding(
        binding_id=f"right-tool-pose:{source_epoch_id}",
        source_instance_id="rm65-right-runtime",
        source_epoch_id=source_epoch_id,
        source_binding_hash="rm65-source-binding-hash",
        control_domain_id="real-robot-right",
        component_id="right_arm",
        stream_id="robot-state.right-arm.tool-tcp-pose",
        reference_frame_id="right-arm-base",
        reference_frame_spec_hash="right-base-frame-spec",
        moving_frame_id="tool-tcp",
        tool_profile_hash=_TOOL_HASH,
        pose_schema_hash="pose-schema-xyzquat-v1",
        fk_contract_id="rm65-hardware-qpos-fk-v1",
        fk_contract_hash="rm65-hardware-qpos-fk-hash",
        measurement_clock_domain_id="local-wall-time",
        receipt_clock_domain_id=receipt_clock_domain_id,
        observation_binding_id="quest-calibration-observation-binding",
        observation_binding_hash="quest-calibration-observation-binding-hash",
        observation_plan_hash="quest-calibration-observation-plan-hash",
        source_kind="real_robot_hardware_qpos_fk",
    )
    return _RobotPoseFixture(coordinate_binding=coordinate)


def _robot_publication(
    *,
    projection: _RobotPoseFixture,
    source_sequence: int,
    received_monotonic_ns: int,
    x_m: float = 0.2,
) -> PoseMeasurementPublication:
    coordinate = projection.coordinate_binding
    measured_at = 90.0 + source_sequence * 0.01
    measurement_available_at = 100.0 + source_sequence * 0.01
    measurement_id = f"robot-measurement-{source_sequence}-{round(x_m * 1e6)}"
    measurement = ExecutionMeasurementObservation(
        measurement_id=measurement_id,
        source_measurement_hash=(
            f"robot-source-measurement-{source_sequence}-{round(x_m * 1e6)}"
        ),
        control_domain_id=coordinate.control_domain_id,
        measured_at=measured_at,
        available_at=measurement_available_at,
        clock_domain_id=coordinate.measurement_clock_domain_id,
        value_by_component={
            coordinate.component_id: (x_m, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        },
        representation_by_component={coordinate.component_id: "cartesian_pose"},
        schema_hash_by_component={
            coordinate.component_id: coordinate.pose_schema_hash,
        },
        observation_binding_id=coordinate.observation_binding_id,
        observation_plan_hash=coordinate.observation_plan_hash,
        parent_facts=(
            ExecutionFactReference(
                stage=POSE_COORDINATE_PARENT_STAGE_V1,
                fact_id=coordinate.binding_id,
                fact_hash=coordinate.semantic_content_hash,
                observed_at=measurement_available_at,
                clock_domain_id=coordinate.measurement_clock_domain_id,
                status="applied",
            ),
        ),
    )
    sample_ref = ObservationSampleRef(
        stream_id=coordinate.stream_id,
        sample_id=measurement.measurement_id,
        sequence=source_sequence,
        measured_at=measurement.measured_at,
        measured_clock_domain_id=measurement.clock_domain_id,
        available_at=received_monotonic_ns / 1_000_000_000.0,
        available_clock_domain_id=coordinate.receipt_clock_domain_id,
        payload_object_id=measurement.semantic_content_hash,
        content_metadata={
            "coordinate_binding_hash": coordinate.semantic_content_hash,
            "source_instance_id": coordinate.source_instance_id,
            "source_epoch_id": coordinate.source_epoch_id,
            "source_kind": coordinate.source_kind,
            "source_sequence": source_sequence,
            "reference_frame_id": coordinate.reference_frame_id,
            "reference_frame_spec_hash": coordinate.reference_frame_spec_hash,
            "moving_frame_id": coordinate.moving_frame_id,
            "tool_profile_hash": coordinate.tool_profile_hash,
            "measurement_available_at": measurement.available_at,
            "measurement_available_clock_domain_id": measurement.clock_domain_id,
            "received_monotonic_ns": received_monotonic_ns,
        },
    )
    return PoseMeasurementPublication(
        coordinate_binding=coordinate,
        source_sequence=source_sequence,
        measurement=measurement,
        sample_ref=sample_ref,
        received_monotonic_ns=received_monotonic_ns,
        receiver_clock=_receiver_clock(),
    )


def _robot_event(
    *,
    event_sequence: int,
    source_sequence: int,
    received_monotonic_ns: int,
    projection: _RobotPoseFixture,
    x_m: float = 0.2,
) -> PoseMeasurementAcceptedEvent:
    return PoseMeasurementAcceptedEvent(
        event_sequence=event_sequence,
        publication=_robot_publication(
            projection=projection,
            source_sequence=source_sequence,
            received_monotonic_ns=received_monotonic_ns,
            x_m=x_m,
        ),
    )


def _controller_matrix(*, x_m: float = 0.25) -> list[float]:
    return [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        x_m,
        0.0,
        0.0,
        1.0,
    ]


def _quest_event(
    *,
    event_sequence: int,
    sample_sequence: int,
    received_monotonic_ns: int,
    x_m: float = 0.25,
    client_id: int = 7,
    client_instance_id: str = "quest-page-1",
    xr_session_id: str = "xr-session-1",
    controller_source_session_id: str = "right-controller-source-1",
    handedness: str = "right",
    target_ray_mode: str = "tracked-pointer",
    profiles: list[str] | None = None,
    emulated_position: bool = False,
    include_controller: bool = True,
    xr_frame_time_ms: float | None = None,
) -> ControllerPoseAcceptedEvent:
    payload = {
        "type": "controller_pose_sample",
        "feature": "controller_pose",
        "protocol_version": PROTOCOL_VERSION,
        "feature_protocol_version": 1,
        "client_instance_id": client_instance_id,
        "xr_session_id": xr_session_id,
        "reference_space": {"type": "local-floor", "generation": 0},
        "sample_sequence": sample_sequence,
        "xr_frame_time_ms": (
            float(sample_sequence * 10)
            if xr_frame_time_ms is None
            else xr_frame_time_ms
        ),
        "controllers": (
            [
                {
                    "source_session_id": controller_source_session_id,
                    "handedness": handedness,
                    "target_ray_mode": target_ray_mode,
                    "profiles": (["oculus-touch-v3"] if profiles is None else profiles),
                    "emulated_position": emulated_position,
                    "matrix_reference_from_grip": _controller_matrix(x_m=x_m),
                }
            ]
            if include_controller
            else []
        ),
    }
    sample = parse_controller_pose_sample(
        payload,
        received_wall_time_s=100.0,
        received_monotonic_ns=received_monotonic_ns,
    )
    return ControllerPoseAcceptedEvent(
        event_sequence=event_sequence,
        server_instance_id="quest-host-instance",
        client_id=client_id,
        sample=sample,
        receipt_clock_domain_id=_CLOCK_ID,
    )


def _quest_invalidation(
    *, event_sequence: int, reason: str = "reference_space_reset"
) -> ControllerPoseInvalidationEvent:
    return ControllerPoseInvalidationEvent(
        event_sequence=event_sequence,
        server_instance_id="quest-host-instance",
        client_id=7,
        client_instance_id="quest-page-1",
        xr_session_id="xr-session-1",
        reference_space_type="local-floor",
        previous_reference_space_generation=0,
        next_reference_space_generation=1,
        last_accepted_sample_sequence=2,
        reason=reason,
        received_monotonic_ns=1_400_000_000,
        receipt_clock_domain_id=_CLOCK_ID,
    )


def _profile(**changes) -> StaticWindowCaptureProfile:
    values = {
        "profile_id": "offline-static-window-test-v1",
        "min_window_duration_ns": 300_000_000,
        "max_window_duration_ns": 1_000_000_000,
        "min_source_span_ns": 150_000_000,
        "max_pair_skew_ns": 20_000_000,
        "min_robot_samples": 2,
        "max_robot_samples": 4,
        "min_quest_samples": 2,
        "max_quest_samples": 4,
        "max_robot_translation_excursion_m": 0.002,
        "max_robot_rotation_excursion_deg": 0.5,
        "max_quest_translation_excursion_m": 0.002,
        "max_quest_rotation_excursion_deg": 0.5,
        "max_quest_reorder_events": 8,
        "max_event_history": 32,
        "max_completed_windows": 4,
    }
    values.update(changes)
    return StaticWindowCaptureProfile(**values)


def _binding(
    *,
    projection: _RobotPoseFixture | None = None,
    receiver_clock: ClockDomainDescriptor | None = None,
) -> PairedPoseCaptureBinding:
    projection = projection or _projection()
    receiver_clock = receiver_clock or _receiver_clock()
    quest = QuestControllerCaptureBinding(
        server_instance_id="quest-host-instance",
        subscription_token=11,
        client_id=7,
        client_instance_id="quest-page-1",
        xr_session_id="xr-session-1",
        reference_space_type="local-floor",
        reference_space_generation=0,
        controller_source_session_id="right-controller-source-1",
        handedness="right",
        target_ray_mode="tracked-pointer",
        profiles=("oculus-touch-v3",),
        reference_frame_id="quest-local-floor",
        moving_frame_id="right-controller-grip",
        xr_frame_clock_domain_id="webxr-animation-frame.xr-session-1",
        receipt_clock_domain_id=_CLOCK_ID,
        next_event_sequence=0,
        baseline_sample_sequence=0,
    )
    return PairedPoseCaptureBinding(
        capture_id="capture-session-1",
        receiver_clock=receiver_clock,
        robot=projection.coordinate_binding,
        robot_next_event_sequence=1,
        quest=quest,
    )


def _healthy_robot_cut(
    *,
    binding: PairedPoseCaptureBinding,
    latest_event: PoseMeasurementAcceptedEvent,
    event_sequence: int | None = None,
) -> PoseMeasurementJournalCut:
    stream_id = binding.robot.stream_id
    return PoseMeasurementJournalCut(
        event_sequence=(
            latest_event.event_sequence if event_sequence is None else event_sequence
        ),
        oldest_retained_event_sequence=binding.robot_next_event_sequence,
        receiver_clock=binding.receiver_clock,
        latest_by_stream={stream_id: latest_event.publication},
        binding_hash_by_stream={
            stream_id: binding.robot.semantic_content_hash,
        },
        invalidated_stream_ids=(),
        expected_next_binding_hash_by_stream={},
    )


def _quest_active_session_cut(
    *,
    binding: PairedPoseCaptureBinding,
    last_accepted_sample_sequence: int | None = 2,
) -> ControllerPoseActiveSessionCut:
    quest = binding.quest
    return ControllerPoseActiveSessionCut(
        client_id=quest.client_id,
        client_instance_id=quest.client_instance_id,
        xr_session_id=quest.xr_session_id,
        reference_space_type=quest.reference_space_type,
        reference_space_generation=quest.reference_space_generation,
        last_accepted_sample_sequence=last_accepted_sample_sequence,
    )


def _healthy_quest_cut(
    *,
    binding: PairedPoseCaptureBinding,
    next_event_sequence: int = 2,
    last_accepted_sample_sequence: int | None = 2,
    active_sessions: tuple[ControllerPoseActiveSessionCut, ...] | None = None,
    stopped: bool = False,
) -> ControllerPoseObserverCloseCut:
    if active_sessions is None:
        active_sessions = (
            _quest_active_session_cut(
                binding=binding,
                last_accepted_sample_sequence=last_accepted_sample_sequence,
            ),
        )
    return ControllerPoseObserverCloseCut(
        subscription_token=binding.quest.subscription_token,
        server_instance_id=binding.quest.server_instance_id,
        next_event_sequence=next_event_sequence,
        active_sessions=active_sessions,
        receipt_clock_domain_id=binding.quest.receipt_clock_domain_id,
        stopped=stopped,
    )


def test_profile_has_no_defaults_and_binding_requires_exact_receiver_clock() -> None:
    signature = inspect.signature(StaticWindowCaptureProfile)
    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in signature.parameters.values()
    )
    profile = _profile()
    assert profile.semantic_content_hash == profile.as_dict()["semantic_content_hash"]

    with pytest.raises(ValueError, match="robot receipt clock"):
        _binding(
            projection=_projection(receipt_clock_domain_id="other.monotonic"),
        )
    with pytest.raises(ValueError, match="Quest receipt clock"):
        valid = _binding()
        PairedPoseCaptureBinding(
            capture_id=valid.capture_id,
            receiver_clock=valid.receiver_clock,
            robot=valid.robot,
            robot_next_event_sequence=valid.robot_next_event_sequence,
            quest=replace(
                valid.quest,
                receipt_clock_domain_id="other.monotonic",
            ),
        )


def test_binding_pins_full_clock_and_all_source_identity() -> None:
    binding = _binding()
    payload = binding.as_dict()

    assert payload["receiver_clock"]["runtime_session_id"] == (
        "quest-capture-runtime-1"
    )
    assert payload["receiver_clock_descriptor_hash"]
    assert payload["robot"]["fk_contract_hash"] == "rm65-hardware-qpos-fk-hash"
    assert payload["quest"]["subscription_token"] == 11
    assert payload["quest"]["reference_epoch_id"] == ("xr-session-1:local-floor:0")
    assert payload["quest"]["profiles"] == ["oculus-touch-v3"]


def _begin(
    capture: ManualStaticPairedPoseCapture,
    *,
    window_id: str = "window-1",
) -> None:
    capture.begin_window(
        window_id=window_id,
        start_received_monotonic_ns=1_000_000_000,
        robot_start_next_event_sequence=1,
        quest_start_next_event_sequence=0,
    )


def _stationary_events(
    projection: _RobotPoseFixture,
) -> tuple[
    tuple[PoseMeasurementAcceptedEvent, PoseMeasurementAcceptedEvent],
    tuple[ControllerPoseAcceptedEvent, ControllerPoseAcceptedEvent],
]:
    robots = (
        _robot_event(
            event_sequence=1,
            source_sequence=1,
            received_monotonic_ns=1_100_000_000,
            projection=projection,
        ),
        _robot_event(
            event_sequence=2,
            source_sequence=2,
            received_monotonic_ns=1_300_000_000,
            projection=projection,
        ),
    )
    quests = (
        _quest_event(
            event_sequence=0,
            sample_sequence=1,
            received_monotonic_ns=1_105_000_000,
        ),
        _quest_event(
            event_sequence=1,
            sample_sequence=2,
            received_monotonic_ns=1_305_000_000,
        ),
    )
    return robots, quests


def _completed_capture(*, window_id: str = "window-1"):
    projection = _projection()
    binding = _binding(projection=projection)
    capture = ManualStaticPairedPoseCapture(
        profile=_profile(),
        binding=binding,
    )
    _begin(capture, window_id=window_id)
    robots, quests = _stationary_events(projection)
    for robot, quest in zip(robots, quests, strict=True):
        capture.ingest_robot_event(robot)
        capture.ingest_quest_event(quest)
    receipt = capture.complete_window(
        end_received_monotonic_ns=1_500_000_000,
        robot_end_next_event_sequence=3,
        quest_end_next_event_sequence=2,
    )
    assert receipt is not None
    return capture, binding, robots, quests, receipt


def _accepted_capture(*, window_id: str = "window-1"):
    capture, binding, robots, _, receipt = _completed_capture(window_id=window_id)
    evidence = capture.seal_capture(
        robot_cut=_healthy_robot_cut(
            binding=binding,
            latest_event=robots[-1],
        ),
        quest_cut=_healthy_quest_cut(binding=binding),
    )
    assert evidence is not None
    assert isinstance(evidence, PairedPoseCaptureEvidence)
    return capture, receipt, evidence


def test_static_window_success_is_candidate_only_and_round_trips_r0_dataset() -> None:
    _, receipt, evidence = _accepted_capture()

    assert receipt.metrics == {
        "duration_ns": 500_000_000,
        "robot_sample_count": 2,
        "quest_sample_count": 2,
        "robot_source_span_ns": 200_000_000,
        "quest_source_span_ns": 200_000_000,
        "robot_translation_excursion_m": pytest.approx(0.0),
        "robot_rotation_excursion_deg": pytest.approx(0.0),
        "quest_translation_excursion_m": pytest.approx(0.0),
        "quest_rotation_excursion_deg": pytest.approx(0.0),
        "pair_skew_ns": 5_000_000,
    }
    assert receipt.selected_robot_event_sequence == 1
    assert receipt.selected_quest_event_sequence == 0
    assert evidence.dataset.pairing_method == "operator_static_window"
    assert len(evidence.dataset.samples) == 1
    sample = evidence.dataset.samples[0]
    assert sample.first.stationary_evidence_id == receipt.stationary_evidence_id
    assert sample.second.stationary_evidence_id == receipt.stationary_evidence_id

    payload = evidence.as_dict()
    assert payload["candidate_only"] is True
    assert payload["calibration_accepted"] is None
    assert "fit" not in payload
    reparsed = parse_paired_pose_capture_evidence(payload)
    assert reparsed.semantic_content_hash == evidence.semantic_content_hash
    assert reparsed.dataset.semantic_sha256 == evidence.dataset.semantic_sha256


def test_candidate_dataset_does_not_seal_or_pin_final_source_cuts() -> None:
    capture, binding, robots, _, _ = _completed_capture()

    candidate = capture.candidate_dataset()

    status = capture.snapshot_status()
    assert status["sealed"] is False
    assert status["finalizing"] is False
    evidence = capture.seal_capture(
        robot_cut=_healthy_robot_cut(binding=binding, latest_event=robots[-1]),
        quest_cut=_healthy_quest_cut(binding=binding),
    )
    assert evidence is not None
    assert evidence.dataset.semantic_sha256 == candidate.semantic_sha256


def test_complete_window_requires_both_exclusive_watermarks_to_drain() -> None:
    projection = _projection()
    binding = _binding(projection=projection)
    capture = ManualStaticPairedPoseCapture(
        profile=_profile(),
        binding=binding,
    )
    _begin(capture)
    robots, quests = _stationary_events(projection)
    capture.ingest_robot_event(robots[0])
    capture.ingest_quest_event(quests[0])

    assert (
        capture.complete_window(
            end_received_monotonic_ns=1_500_000_000,
            robot_end_next_event_sequence=3,
            quest_end_next_event_sequence=2,
        )
        is None
    )
    assert capture.snapshot_status()["draining"] is True
    with pytest.raises(RuntimeError, match="active"):
        capture.seal_capture(
            robot_cut=_healthy_robot_cut(
                binding=binding,
                latest_event=robots[0],
            ),
            quest_cut=_healthy_quest_cut(
                binding=binding,
                next_event_sequence=1,
                last_accepted_sample_sequence=1,
            ),
        )

    capture.ingest_quest_event(quests[1])
    capture.ingest_robot_event(robots[1])
    receipt = capture.complete_window(
        end_received_monotonic_ns=1_500_000_000,
        robot_end_next_event_sequence=3,
        quest_end_next_event_sequence=2,
    )
    assert receipt is not None
    assert capture.snapshot_status()["draining"] is False


def test_final_source_cuts_must_drain_before_sealing() -> None:
    capture, binding, robots, _, _ = _completed_capture()

    evidence = capture.seal_capture(
        robot_cut=_healthy_robot_cut(
            binding=binding,
            latest_event=robots[-1],
            event_sequence=3,
        ),
        quest_cut=_healthy_quest_cut(
            binding=binding,
            next_event_sequence=3,
        ),
    )

    assert evidence is None
    assert capture.snapshot_status()["sealed"] is False


def test_final_source_cuts_seal_only_after_both_sources_drain() -> None:
    capture, binding, _, _, _ = _completed_capture()
    final_robot_event = _robot_event(
        event_sequence=3,
        source_sequence=3,
        received_monotonic_ns=1_400_000_000,
        projection=_projection(),
    )
    final_quest_event = _quest_event(
        event_sequence=2,
        sample_sequence=3,
        received_monotonic_ns=1_405_000_000,
    )
    robot_cut = _healthy_robot_cut(
        binding=binding,
        latest_event=final_robot_event,
    )
    quest_cut = _healthy_quest_cut(
        binding=binding,
        next_event_sequence=3,
        last_accepted_sample_sequence=3,
    )

    assert capture.seal_capture(robot_cut=robot_cut, quest_cut=quest_cut) is None
    with pytest.raises(RuntimeError, match="finalization"):
        _begin(capture, window_id="too-late")
    capture.ingest_quest_event(final_quest_event)
    assert capture.seal_capture(robot_cut=robot_cut, quest_cut=quest_cut) is None
    capture.ingest_robot_event(final_robot_event)

    evidence = capture.seal_capture(robot_cut=robot_cut, quest_cut=quest_cut)

    assert evidence is not None
    assert evidence.finalization.robot_end_next_event_sequence == 4
    assert evidence.finalization.quest_end_next_event_sequence == 3
    reparsed = parse_paired_pose_capture_evidence(evidence.as_dict())
    assert reparsed.semantic_content_hash == evidence.semantic_content_hash


def test_seal_capture_requires_typed_source_cuts() -> None:
    capture, binding, robots, _, _ = _completed_capture()
    robot_cut = _healthy_robot_cut(
        binding=binding,
        latest_event=robots[-1],
    )
    quest_cut = _healthy_quest_cut(binding=binding)

    with pytest.raises(TypeError, match="robot_cut"):
        capture.seal_capture(robot_cut=object(), quest_cut=quest_cut)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="quest_cut"):
        capture.seal_capture(robot_cut=robot_cut, quest_cut=object())  # type: ignore[arg-type]
    plain_event_cut = ControllerPoseEventCut(
        server_instance_id=quest_cut.server_instance_id,
        next_event_sequence=quest_cut.next_event_sequence,
        active_sessions=quest_cut.active_sessions,
        receipt_clock_domain_id=quest_cut.receipt_clock_domain_id,
        stopped=quest_cut.stopped,
    )
    with pytest.raises(TypeError, match="ControllerPoseObserverCloseCut"):
        capture.seal_capture(
            robot_cut=robot_cut,
            quest_cut=plain_event_cut,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "failure",
    ["stopped", "session_mismatch", "subscription_mismatch", "session_order"],
)
def test_final_quest_cut_must_keep_the_exact_session_active(failure: str) -> None:
    capture, binding, robots, _, _ = _completed_capture()
    if failure == "stopped":
        quest_cut = _healthy_quest_cut(
            binding=binding,
            active_sessions=(),
            stopped=True,
        )
    elif failure == "session_mismatch":
        mismatched = replace(
            _quest_active_session_cut(binding=binding),
            xr_session_id="different-xr-session",
        )
        quest_cut = _healthy_quest_cut(
            binding=binding,
            active_sessions=(mismatched,),
        )
    elif failure == "subscription_mismatch":
        quest_cut = replace(
            _healthy_quest_cut(binding=binding),
            subscription_token=binding.quest.subscription_token + 1,
        )
    else:
        bound = _quest_active_session_cut(binding=binding)
        unrelated = replace(
            bound,
            client_id=99,
            client_instance_id="other-client-instance",
            xr_session_id="other-xr-session",
        )
        quest_cut = _healthy_quest_cut(
            binding=binding,
            active_sessions=(unrelated, bound),
        )

    with pytest.raises(ValueError):
        capture.seal_capture(
            robot_cut=_healthy_robot_cut(
                binding=binding,
                latest_event=robots[-1],
            ),
            quest_cut=quest_cut,
        )


def test_pending_quest_gap_cannot_seal_at_a_later_source_cut() -> None:
    capture, binding, robots, _, _ = _completed_capture()
    capture.ingest_quest_event(
        _quest_event(
            event_sequence=3,
            sample_sequence=4,
            received_monotonic_ns=1_400_000_000,
        )
    )
    assert capture.snapshot_status()["quest_pending_event_count"] == 1

    evidence = capture.seal_capture(
        robot_cut=_healthy_robot_cut(
            binding=binding,
            latest_event=robots[-1],
        ),
        quest_cut=_healthy_quest_cut(
            binding=binding,
            next_event_sequence=4,
            last_accepted_sample_sequence=4,
        ),
    )

    assert evidence is None
    assert capture.snapshot_status()["sealed"] is False


def test_quest_global_reorder_includes_unrelated_client_without_losing_selection() -> (
    None
):
    projection = _projection()
    capture = ManualStaticPairedPoseCapture(
        profile=_profile(),
        binding=_binding(projection=projection),
    )
    _begin(capture)
    robot_first, robot_second = _stationary_events(projection)[0]
    selected_first = _quest_event(
        event_sequence=0,
        sample_sequence=1,
        received_monotonic_ns=1_105_000_000,
    )
    unrelated = _quest_event(
        event_sequence=1,
        sample_sequence=1,
        received_monotonic_ns=1_200_000_000,
        client_id=99,
        client_instance_id="other-page",
        xr_session_id="other-xr-session",
        controller_source_session_id="other-controller",
    )
    selected_second = _quest_event(
        event_sequence=2,
        sample_sequence=2,
        received_monotonic_ns=1_305_000_000,
    )

    capture.ingest_robot_event(robot_first)
    capture.ingest_robot_event(robot_second)
    capture.ingest_quest_event(selected_first)
    capture.ingest_quest_event(selected_second)
    assert capture.snapshot_status()["quest_pending_event_count"] == 1
    capture.ingest_quest_event(unrelated)
    assert capture.snapshot_status()["quest_pending_event_count"] == 0

    receipt = capture.complete_window(
        end_received_monotonic_ns=1_500_000_000,
        robot_end_next_event_sequence=3,
        quest_end_next_event_sequence=3,
    )
    assert receipt is not None
    assert [event.event_sequence for event in receipt.quest_events] == [0, 1, 2]
    assert receipt.selected_quest_event_sequence == 0


@pytest.mark.parametrize("source", ["robot", "quest"])
def test_duplicate_sequence_is_idempotent_only_for_identical_content(
    source: str,
) -> None:
    projection = _projection()
    capture = ManualStaticPairedPoseCapture(
        profile=_profile(),
        binding=_binding(projection=projection),
    )
    if source == "robot":
        original = _robot_event(
            event_sequence=1,
            source_sequence=1,
            received_monotonic_ns=1_100_000_000,
            projection=projection,
        )
        changed = _robot_event(
            event_sequence=1,
            source_sequence=1,
            received_monotonic_ns=1_100_000_000,
            projection=projection,
            x_m=0.21,
        )
        capture.ingest_robot_event(original)
        capture.ingest_robot_event(original)
        with pytest.raises(CaptureTerminalError, match="reused"):
            capture.ingest_robot_event(changed)
    else:
        original = _quest_event(
            event_sequence=0,
            sample_sequence=1,
            received_monotonic_ns=1_100_000_000,
        )
        changed = _quest_event(
            event_sequence=0,
            sample_sequence=1,
            received_monotonic_ns=1_100_000_000,
            x_m=0.26,
        )
        capture.ingest_quest_event(original)
        capture.ingest_quest_event(original)
        with pytest.raises(CaptureTerminalError, match="reused"):
            capture.ingest_quest_event(changed)

    with pytest.raises(CaptureTerminalError):
        capture.begin_window(
            window_id="after-terminal",
            start_received_monotonic_ns=2_000_000_000,
            robot_start_next_event_sequence=2,
            quest_start_next_event_sequence=1,
        )


def test_robot_global_cursor_gap_fails_closed() -> None:
    projection = _projection()
    capture = ManualStaticPairedPoseCapture(
        profile=_profile(),
        binding=_binding(projection=projection),
    )
    skipped = _robot_event(
        event_sequence=2,
        source_sequence=2,
        received_monotonic_ns=1_200_000_000,
        projection=projection,
    )
    with pytest.raises(CaptureTerminalError, match="journal event gap"):
        capture.ingest_robot_event(skipped)


@pytest.mark.parametrize("failure", ["identity", "invalidation"])
def test_selected_quest_identity_and_invalidation_are_terminal(
    failure: str,
) -> None:
    capture = ManualStaticPairedPoseCapture(profile=_profile(), binding=_binding())
    if failure == "identity":
        event = _quest_event(
            event_sequence=0,
            sample_sequence=1,
            received_monotonic_ns=1_100_000_000,
            client_instance_id="different-page",
        )
    else:
        event = _quest_invalidation(event_sequence=0)

    with pytest.raises(CaptureTerminalError):
        capture.ingest_quest_event(event)
    assert capture.snapshot_status()["terminal_reason"] is not None


def test_emulated_selected_quest_pose_is_ignored_not_used_for_calibration() -> None:
    projection = _projection()
    binding = _binding(projection=projection)
    capture = ManualStaticPairedPoseCapture(profile=_profile(), binding=binding)
    _begin(capture)

    robots = (
        _robot_event(
            event_sequence=1,
            source_sequence=1,
            received_monotonic_ns=1_100_000_000,
            projection=projection,
        ),
        _robot_event(
            event_sequence=2,
            source_sequence=2,
            received_monotonic_ns=1_300_000_000,
            projection=projection,
        ),
    )
    quests = (
        _quest_event(
            event_sequence=0,
            sample_sequence=1,
            received_monotonic_ns=1_105_000_000,
        ),
        _quest_event(
            event_sequence=1,
            sample_sequence=2,
            received_monotonic_ns=1_200_000_000,
            emulated_position=True,
        ),
        _quest_event(
            event_sequence=2,
            sample_sequence=3,
            received_monotonic_ns=1_305_000_000,
        ),
    )
    for event in robots:
        capture.ingest_robot_event(event)
    for event in quests:
        capture.ingest_quest_event(event)

    receipt = capture.complete_window(
        end_received_monotonic_ns=1_500_000_000,
        robot_end_next_event_sequence=3,
        quest_end_next_event_sequence=3,
    )

    assert receipt is not None
    assert receipt.metrics["quest_sample_count"] == 2
    assert receipt.metrics["quest_source_span_ns"] == 200_000_000
    assert len(receipt.quest_events) == 3
    assert capture.snapshot_status()["terminal_reason"] is None


def test_emulated_quest_samples_cannot_satisfy_the_window_minimum() -> None:
    projection = _projection()
    binding = _binding(projection=projection)
    capture = ManualStaticPairedPoseCapture(profile=_profile(), binding=binding)
    _begin(capture)

    for event in (
        _robot_event(
            event_sequence=1,
            source_sequence=1,
            received_monotonic_ns=1_100_000_000,
            projection=projection,
        ),
        _robot_event(
            event_sequence=2,
            source_sequence=2,
            received_monotonic_ns=1_300_000_000,
            projection=projection,
        ),
    ):
        capture.ingest_robot_event(event)
    for event in (
        _quest_event(
            event_sequence=0,
            sample_sequence=1,
            received_monotonic_ns=1_105_000_000,
            emulated_position=True,
        ),
        _quest_event(
            event_sequence=1,
            sample_sequence=2,
            received_monotonic_ns=1_305_000_000,
            emulated_position=True,
        ),
    ):
        capture.ingest_quest_event(event)

    with pytest.raises(
        StaticWindowRejectedError,
        match="Quest sample count is outside profile",
    ):
        capture.complete_window(
            end_received_monotonic_ns=1_500_000_000,
            robot_end_next_event_sequence=3,
            quest_end_next_event_sequence=2,
        )
    assert capture.snapshot_status()["terminal_reason"] is None
    assert capture.snapshot_status()["active_window_id"] is None


def test_selected_robot_invalidation_is_terminal() -> None:
    binding = _binding()
    capture = ManualStaticPairedPoseCapture(profile=_profile(), binding=binding)
    event = PoseMeasurementInvalidationEvent(
        event_sequence=1,
        coordinate_binding=binding.robot,
        reason="hardware_read_failed",
        received_monotonic_ns=1_100_000_000,
        receiver_clock=binding.receiver_clock,
    )
    with pytest.raises(CaptureTerminalError, match="invalidated"):
        capture.ingest_robot_event(event)


def test_full_receiver_clock_descriptor_drift_is_terminal_even_when_id_matches() -> (
    None
):
    binding = _binding()
    capture = ManualStaticPairedPoseCapture(profile=_profile(), binding=binding)
    publication = _robot_publication(
        projection=_projection(),
        source_sequence=1,
        received_monotonic_ns=1_100_000_000,
    )
    changed_clock = _receiver_clock(runtime_session_id="different-runtime")
    changed_publication = replace(publication, receiver_clock=changed_clock)
    event = PoseMeasurementAcceptedEvent(
        event_sequence=1,
        publication=changed_publication,
    )

    with pytest.raises(CaptureTerminalError, match="descriptor drifted"):
        capture.ingest_robot_event(event)


def test_motion_beyond_static_threshold_rejects_window_without_a_candidate() -> None:
    projection = _projection()
    binding = _binding(projection=projection)
    capture = ManualStaticPairedPoseCapture(
        profile=_profile(),
        binding=binding,
    )
    _begin(capture)
    robot_first = _robot_event(
        event_sequence=1,
        source_sequence=1,
        received_monotonic_ns=1_100_000_000,
        projection=projection,
        x_m=0.20,
    )
    robot_second = _robot_event(
        event_sequence=2,
        source_sequence=2,
        received_monotonic_ns=1_300_000_000,
        projection=projection,
        x_m=0.21,
    )
    quest_first = _quest_event(
        event_sequence=0,
        sample_sequence=1,
        received_monotonic_ns=1_105_000_000,
    )
    quest_second = _quest_event(
        event_sequence=1,
        sample_sequence=2,
        received_monotonic_ns=1_305_000_000,
    )
    capture.ingest_robot_event(robot_first)
    capture.ingest_robot_event(robot_second)
    capture.ingest_quest_event(quest_first)
    capture.ingest_quest_event(quest_second)

    with pytest.raises(StaticWindowRejectedError, match="motion"):
        capture.complete_window(
            end_received_monotonic_ns=1_500_000_000,
            robot_end_next_event_sequence=3,
            quest_end_next_event_sequence=2,
        )
    with pytest.raises(RuntimeError, match="accepted static window"):
        capture.seal_capture(
            robot_cut=_healthy_robot_cut(
                binding=binding,
                latest_event=robot_second,
            ),
            quest_cut=_healthy_quest_cut(binding=binding),
        )


def test_quest_reorder_overflow_fails_closed() -> None:
    capture = ManualStaticPairedPoseCapture(
        profile=_profile(max_quest_reorder_events=1),
        binding=_binding(),
    )
    capture.ingest_quest_event(
        _quest_event(
            event_sequence=1,
            sample_sequence=2,
            received_monotonic_ns=1_200_000_000,
        )
    )
    with pytest.raises(CaptureTerminalError, match="reorder buffer overflow"):
        capture.ingest_quest_event(
            _quest_event(
                event_sequence=2,
                sample_sequence=3,
                received_monotonic_ns=1_300_000_000,
            )
        )


def test_full_reorder_buffer_accepts_the_missing_next_event_and_drains() -> None:
    capture = ManualStaticPairedPoseCapture(
        profile=_profile(max_quest_reorder_events=1),
        binding=_binding(),
    )
    capture.ingest_quest_event(
        _quest_event(
            event_sequence=1,
            sample_sequence=2,
            received_monotonic_ns=1_200_000_000,
        )
    )
    capture.ingest_quest_event(
        _quest_event(
            event_sequence=0,
            sample_sequence=1,
            received_monotonic_ns=1_100_000_000,
        )
    )
    assert capture.snapshot_status()["quest_pending_event_count"] == 0
    assert capture.snapshot_status()["quest_next_event_sequence"] == 2


def test_active_window_event_history_overflow_fails_closed() -> None:
    projection = _projection()
    capture = ManualStaticPairedPoseCapture(
        profile=_profile(
            min_robot_samples=1,
            max_robot_samples=1,
            min_quest_samples=1,
            max_quest_samples=1,
            max_quest_reorder_events=1,
            max_event_history=1,
        ),
        binding=_binding(projection=projection),
    )
    _begin(capture)
    capture.ingest_robot_event(
        _robot_event(
            event_sequence=1,
            source_sequence=1,
            received_monotonic_ns=1_100_000_000,
            projection=projection,
        )
    )
    with pytest.raises(CaptureTerminalError, match="history overflowed"):
        capture.ingest_robot_event(
            _robot_event(
                event_sequence=2,
                source_sequence=2,
                received_monotonic_ns=1_200_000_000,
                projection=projection,
            )
        )


@pytest.mark.parametrize(
    ("failure", "second_kwargs", "message"),
    [
        ("sample", {"sample_sequence": 1}, "sample sequence"),
        ("xr_time", {"xr_frame_time_ms": 5.0}, "XR frame time"),
        ("metadata", {"handedness": "left"}, "metadata drifted"),
    ],
)
def test_quest_sequence_time_and_controller_metadata_regression_are_terminal(
    failure: str,
    second_kwargs: dict[str, object],
    message: str,
) -> None:
    del failure
    capture = ManualStaticPairedPoseCapture(profile=_profile(), binding=_binding())
    capture.ingest_quest_event(
        _quest_event(
            event_sequence=0,
            sample_sequence=2,
            xr_frame_time_ms=20.0,
            received_monotonic_ns=1_100_000_000,
        )
    )
    values = {
        "event_sequence": 1,
        "sample_sequence": 3,
        "xr_frame_time_ms": 30.0,
        "received_monotonic_ns": 1_200_000_000,
        **second_kwargs,
    }
    with pytest.raises(CaptureTerminalError, match=message):
        capture.ingest_quest_event(_quest_event(**values))


def test_missing_selected_controller_pose_is_ignored_without_terminal_capture() -> None:
    capture = ManualStaticPairedPoseCapture(profile=_profile(), binding=_binding())

    capture.ingest_quest_event(
        _quest_event(
            event_sequence=0,
            sample_sequence=1,
            include_controller=False,
            received_monotonic_ns=1_100_000_000,
        )
    )

    assert capture.snapshot_status()["terminal_reason"] is None


def test_quest_profile_metadata_can_change_without_changing_selected_source() -> None:
    capture = ManualStaticPairedPoseCapture(profile=_profile(), binding=_binding())
    capture.ingest_quest_event(
        _quest_event(
            event_sequence=0,
            sample_sequence=2,
            profiles=["generic-trigger"],
            received_monotonic_ns=1_100_000_000,
        )
    )
    capture.ingest_quest_event(
        _quest_event(
            event_sequence=1,
            sample_sequence=3,
            profiles=["oculus-touch-v3", "generic-trigger"],
            received_monotonic_ns=1_200_000_000,
        )
    )

    assert capture.snapshot_status()["terminal_reason"] is None


def test_robot_source_sequence_regression_and_quest_clock_drift_are_terminal() -> None:
    projection = _projection()
    robot_capture = ManualStaticPairedPoseCapture(
        profile=_profile(),
        binding=_binding(projection=projection),
    )
    robot_capture.ingest_robot_event(
        _robot_event(
            event_sequence=1,
            source_sequence=2,
            received_monotonic_ns=1_100_000_000,
            projection=projection,
        )
    )
    with pytest.raises(CaptureTerminalError, match="source sequence"):
        robot_capture.ingest_robot_event(
            _robot_event(
                event_sequence=2,
                source_sequence=1,
                received_monotonic_ns=1_200_000_000,
                projection=projection,
            )
        )

    quest_capture = ManualStaticPairedPoseCapture(
        profile=_profile(),
        binding=_binding(),
    )
    wrong_clock = replace(
        _quest_event(
            event_sequence=0,
            sample_sequence=1,
            received_monotonic_ns=1_100_000_000,
        ),
        receipt_clock_domain_id="different.monotonic",
    )
    with pytest.raises(CaptureTerminalError, match="receipt clock"):
        quest_capture.ingest_quest_event(wrong_clock)


def test_robot_receiver_monotonic_time_regression_is_terminal() -> None:
    projection = _projection()
    capture = ManualStaticPairedPoseCapture(
        profile=_profile(),
        binding=_binding(projection=projection),
    )
    capture.ingest_robot_event(
        _robot_event(
            event_sequence=1,
            source_sequence=1,
            received_monotonic_ns=1_200_000_000,
            projection=projection,
        )
    )
    with pytest.raises(CaptureTerminalError, match="receiver-monotonic time"):
        capture.ingest_robot_event(
            _robot_event(
                event_sequence=2,
                source_sequence=2,
                received_monotonic_ns=1_100_000_000,
                projection=projection,
            )
        )


def test_callback_arrival_order_does_not_change_window_or_dataset_hashes() -> None:
    projection = _projection()
    binding = _binding(projection=projection)
    profile = _profile()
    robots, quests = _stationary_events(projection)
    captures = [
        ManualStaticPairedPoseCapture(profile=profile, binding=binding),
        ManualStaticPairedPoseCapture(profile=profile, binding=binding),
    ]
    for capture in captures:
        _begin(capture)
        for event in robots:
            capture.ingest_robot_event(event)
    for event in quests:
        captures[0].ingest_quest_event(event)
    for event in reversed(quests):
        captures[1].ingest_quest_event(event)

    receipts = [
        capture.complete_window(
            end_received_monotonic_ns=1_500_000_000,
            robot_end_next_event_sequence=3,
            quest_end_next_event_sequence=2,
        )
        for capture in captures
    ]
    assert all(receipt is not None for receipt in receipts)
    assert receipts[0].semantic_content_hash == receipts[1].semantic_content_hash
    evidence = [
        capture.seal_capture(
            robot_cut=_healthy_robot_cut(
                binding=binding,
                latest_event=robots[-1],
            ),
            quest_cut=_healthy_quest_cut(binding=binding),
        )
        for capture in captures
    ]
    assert all(value is not None for value in evidence)
    assert evidence[0] is not None
    assert evidence[1] is not None
    assert evidence[0].semantic_content_hash == evidence[1].semantic_content_hash
    assert evidence[0].dataset.semantic_sha256 == evidence[1].dataset.semantic_sha256


def test_concurrent_quest_callback_ingestion_is_deterministic() -> None:
    _, baseline_receipt, baseline_evidence = _accepted_capture()
    projection = _projection()
    binding = _binding(projection=projection)
    robots, quests = _stationary_events(projection)
    capture = ManualStaticPairedPoseCapture(
        profile=_profile(),
        binding=binding,
    )
    _begin(capture)
    for event in robots:
        capture.ingest_robot_event(event)

    barrier = threading.Barrier(3)
    failures: list[Exception] = []

    def ingest(event: ControllerPoseAcceptedEvent) -> None:
        try:
            barrier.wait()
            capture.ingest_quest_event(event)
        except Exception as exc:  # noqa: BLE001 - surfaced on the main test thread.
            failures.append(exc)

    workers = [
        threading.Thread(target=ingest, args=(event,), daemon=True) for event in quests
    ]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join(timeout=2.0)

    assert not failures
    assert all(not worker.is_alive() for worker in workers)
    receipt = capture.complete_window(
        end_received_monotonic_ns=1_500_000_000,
        robot_end_next_event_sequence=3,
        quest_end_next_event_sequence=2,
    )
    assert receipt is not None
    evidence = capture.seal_capture(
        robot_cut=_healthy_robot_cut(
            binding=binding,
            latest_event=robots[-1],
        ),
        quest_cut=_healthy_quest_cut(binding=binding),
    )
    assert evidence is not None
    assert receipt.semantic_content_hash == baseline_receipt.semantic_content_hash
    assert evidence.semantic_content_hash == baseline_evidence.semantic_content_hash
    assert evidence.dataset.semantic_sha256 == baseline_evidence.dataset.semantic_sha256


def test_reducer_and_sealing_do_not_open_sockets_files_or_start_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection = _projection()
    robots, quests = _stationary_events(projection)
    binding = _binding(projection=projection)
    capture = ManualStaticPairedPoseCapture(
        profile=_profile(),
        binding=binding,
    )
    robot_cut = _healthy_robot_cut(
        binding=binding,
        latest_event=robots[-1],
    )
    quest_cut = _healthy_quest_cut(binding=binding)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("pure capture reducer attempted external I/O")

    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(threading.Thread, "start", forbidden)

    _begin(capture)
    for robot, quest in zip(robots, quests, strict=True):
        capture.ingest_robot_event(robot)
        capture.ingest_quest_event(quest)
    receipt = capture.complete_window(
        end_received_monotonic_ns=1_500_000_000,
        robot_end_next_event_sequence=3,
        quest_end_next_event_sequence=2,
    )
    assert receipt is not None
    evidence = capture.seal_capture(robot_cut=robot_cut, quest_cut=quest_cut)
    assert evidence is not None
    assert evidence.as_dict()["calibration_accepted"] is None


def test_evidence_constructor_rejects_overlapping_individually_valid_windows() -> None:
    _, first_receipt, first_evidence = _accepted_capture(window_id="window-1")
    _, second_receipt, second_evidence = _accepted_capture(window_id="window-2")
    combined_dataset = replace(
        first_evidence.dataset,
        samples=(
            first_evidence.dataset.samples[0],
            second_evidence.dataset.samples[0],
        ),
    )

    with pytest.raises(ValueError):
        replace(
            first_evidence,
            windows=(first_receipt, second_receipt),
            dataset=combined_dataset,
        )


def test_final_robot_cut_cannot_reuse_source_sequence_with_new_content() -> None:
    _, _, evidence = _accepted_capture()
    binding = evidence.binding
    mismatched_latest = _robot_event(
        event_sequence=2,
        source_sequence=2,
        received_monotonic_ns=1_250_000_000,
        projection=_projection(),
        x_m=0.9,
    )
    mismatched_finalization = PairedPoseCaptureFinalization(
        binding=binding,
        robot_cut=_healthy_robot_cut(
            binding=binding,
            latest_event=mismatched_latest,
        ),
        quest_cut=evidence.finalization.quest_cut,
    )

    with pytest.raises(ValueError, match="reused a source sequence"):
        replace(evidence, finalization=mismatched_finalization)
    payload = copy.deepcopy(evidence.as_dict())
    payload["finalization"] = mismatched_finalization.as_dict()
    with pytest.raises(ValueError, match="reused a source sequence"):
        parse_paired_pose_capture_evidence(payload)


def test_final_cuts_cannot_advance_without_an_event_slot() -> None:
    _, _, evidence = _accepted_capture()
    binding = evidence.binding
    advanced_robot = _robot_event(
        event_sequence=2,
        source_sequence=3,
        received_monotonic_ns=1_350_000_000,
        projection=_projection(),
    )
    robot_advance = PairedPoseCaptureFinalization(
        binding=binding,
        robot_cut=_healthy_robot_cut(
            binding=binding,
            latest_event=advanced_robot,
        ),
        quest_cut=evidence.finalization.quest_cut,
    )
    with pytest.raises(ValueError, match="without a journal event slot"):
        replace(evidence, finalization=robot_advance)

    session = evidence.finalization.quest_cut.active_sessions[0]
    quest_advance = PairedPoseCaptureFinalization(
        binding=binding,
        robot_cut=evidence.finalization.robot_cut,
        quest_cut=replace(
            evidence.finalization.quest_cut,
            active_sessions=(replace(session, last_accepted_sample_sequence=3),),
        ),
    )
    with pytest.raises(ValueError, match="without a runtime event slot"):
        replace(evidence, finalization=quest_advance)


def test_strict_evidence_parser_rejects_metric_and_dataset_tampering() -> None:
    _, _, evidence = _accepted_capture()

    metric_tamper = copy.deepcopy(evidence.as_dict())
    metric_tamper["windows"][0]["metrics"]["pair_skew_ns"] += 1
    with pytest.raises(ValueError):
        parse_paired_pose_capture_evidence(metric_tamper)

    dataset_tamper = copy.deepcopy(evidence.as_dict())
    dataset_tamper["dataset"]["samples"][0]["first_observation"]["observed_at"] += 0.01
    with pytest.raises(ValueError):
        parse_paired_pose_capture_evidence(dataset_tamper)

    event_tamper = copy.deepcopy(evidence.as_dict())
    event_tamper["windows"][0]["robot_events"][0]["publication"][
        "received_monotonic_ns"
    ] += 1
    with pytest.raises(ValueError):
        parse_paired_pose_capture_evidence(event_tamper)

    unknown_field = copy.deepcopy(evidence.as_dict())
    unknown_field["windows"][0]["unexpected"] = True
    with pytest.raises(ValueError, match="keys mismatch"):
        parse_paired_pose_capture_evidence(unknown_field)
