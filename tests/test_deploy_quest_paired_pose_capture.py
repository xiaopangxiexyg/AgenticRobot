from __future__ import annotations

import inspect
import json
import threading
from dataclasses import replace
from pathlib import Path
from unittest import mock

import pytest

from dataset_contracts.canonical_encoding import canonical_content_hash
from hardware.quest_dev.features.controller_pose.protocol import (
    FEATURE_NAME,
    FEATURE_PROTOCOL_VERSION,
    ControllerPoseSample,
    parse_controller_pose_sample,
)
from hardware.quest_dev.features.controller_pose.runtime import (
    ControllerPoseAcceptedEvent,
    ControllerPoseInvalidationEvent,
    ControllerPoseTelemetryRuntime,
)
from hardware.quest_dev.features.visualization.paired_pose_capture import (
    StaticWindowCaptureProfile,
    StaticWindowRejectedError,
)
from hardware.quest_dev.protocol import PROTOCOL_VERSION
from hardware.quest_dev.server import QuestDevHost, QuestDevLaunchInfo
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
from teleop_runtime.deploy import run_policy
from teleop_runtime.deploy.quest_paired_pose_capture import (
    QUEST_CONTROLLER_CANDIDATE_HASH_DOMAIN,
    PassiveRobotPoseJournalSource,
    QuestCaptureFrameFacts,
    QuestCaptureSelectionIntent,
    QuestPairedPoseCaptureApplication,
    QuestPairedPoseCaptureCleanupError,
    QuestPairedPoseCaptureFactory,
    QuestPairedPoseCaptureLaunchConfig,
    QuestPairedPoseCaptureStartupError,
    QuestPairedPoseCaptureTerminalError,
    build_quest_controller_runtime_loader,
)

IDENTITY_MATRIX = (
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
    0.0,
    0.0,
    0.0,
    1.0,
)


class _SequenceClock:
    def __init__(self, start: int = 1_000_000_000) -> None:
        self.value = start

    def __call__(self) -> int:
        self.value += 100_000_000
        return self.value


def _receiver_clock() -> ClockDomainDescriptor:
    return ClockDomainDescriptor(
        clock_domain_id="linux-monotonic:test-boot",
        runtime_session_id="policy-deploy:test-session",
        boot_id="test-boot",
        reset_epoch=0,
        kind="python.monotonic",
        resolution_sec=1e-9,
        monotonic=True,
    )


def _coordinate_binding(
    *,
    side: str,
    receiver_clock: ClockDomainDescriptor,
) -> PoseCoordinateBinding:
    return PoseCoordinateBinding(
        binding_id=f"binding-{side}",
        source_instance_id=f"rm65-{side}-source",
        source_epoch_id=f"rm65-{side}-epoch",
        source_binding_hash=f"source-binding-{side}",
        control_domain_id="deploy-control-0",
        component_id=f"{side}_arm.tcp_pose",
        stream_id=f"deploy:{side}:tcp-pose",
        reference_frame_id=f"rm65-{side}-arm-base",
        reference_frame_spec_hash=f"frame-spec-{side}",
        moving_frame_id=f"rm65-{side}-tool-tcp",
        tool_profile_hash="sha256:" + "a" * 64,
        pose_schema_hash="pose-schema-v1",
        fk_contract_id=f"fk-{side}",
        fk_contract_hash=f"fk-hash-{side}",
        measurement_clock_domain_id=f"rm65-{side}-measurement-clock",
        receipt_clock_domain_id=receiver_clock.clock_domain_id,
        observation_binding_id=f"observation-binding-{side}",
        observation_binding_hash=f"observation-binding-hash-{side}",
        observation_plan_hash=f"observation-plan-{side}",
        source_kind="real_robot.hardware_qpos_fk_pose",
    )


class _RobotJournal:
    def __init__(self, receiver_clock: ClockDomainDescriptor) -> None:
        self.receiver_clock = receiver_clock
        self.events: list[PoseMeasurementInvalidationEvent] = []
        self.snapshot_calls = 0
        self.get_state_calls = 0
        self.refresh_calls = 0

    def atomic_cut(self) -> PoseMeasurementJournalCut:
        event_sequence = self.events[-1].event_sequence if self.events else 0
        latest_by_stream = {}
        binding_hash_by_stream = {}
        invalidated_stream_ids: set[str] = set()
        expected_next_binding_hash_by_stream = {}
        for event in self.events:
            if isinstance(event, PoseMeasurementAcceptedEvent):
                publication = event.publication
                stream_id = publication.coordinate_binding.stream_id
                latest_by_stream[stream_id] = publication
                binding_hash_by_stream[stream_id] = (
                    publication.coordinate_binding.semantic_content_hash
                )
                invalidated_stream_ids.discard(stream_id)
                expected_next_binding_hash_by_stream.pop(stream_id, None)
                continue
            stream_id = event.coordinate_binding.stream_id
            latest_by_stream.pop(stream_id, None)
            binding_hash_by_stream.pop(stream_id, None)
            invalidated_stream_ids.add(stream_id)
            if event.next_coordinate_binding is not None:
                expected_next_binding_hash_by_stream[stream_id] = (
                    event.next_coordinate_binding.semantic_content_hash
                )
        return PoseMeasurementJournalCut(
            event_sequence=event_sequence,
            oldest_retained_event_sequence=(
                None if not self.events else self.events[0].event_sequence
            ),
            receiver_clock=self.receiver_clock,
            latest_by_stream=latest_by_stream,
            binding_hash_by_stream=binding_hash_by_stream,
            invalidated_stream_ids=tuple(invalidated_stream_ids),
            expected_next_binding_hash_by_stream=(expected_next_binding_hash_by_stream),
        )

    def events_after(self, event_sequence: int):
        return tuple(
            event for event in self.events if event.event_sequence > event_sequence
        )

    def snapshot(self, *_args, **_kwargs):
        self.snapshot_calls += 1
        raise AssertionError("paired capture must not call snapshot")

    def get_state(self, *_args, **_kwargs):
        self.get_state_calls += 1
        raise AssertionError("paired capture must not call get_state")

    def refresh(self, *_args, **_kwargs):
        self.refresh_calls += 1
        raise AssertionError("paired capture must not call refresh")


class _QuestServer:
    def __init__(self, monotonic_ns_clock) -> None:
        self.server_instance_id = "quest-server-1"
        self.monotonic_ns_clock = monotonic_ns_clock
        self.published_feature_messages: list[tuple[str, dict]] = []

    def publish_feature_message(self, feature: str, message: dict) -> None:
        self.published_feature_messages.append((feature, message))


def _profile(*, max_event_history: int = 32) -> StaticWindowCaptureProfile:
    return StaticWindowCaptureProfile(
        profile_id="test-static-profile",
        min_window_duration_ns=1,
        max_window_duration_ns=10_000_000_000,
        min_source_span_ns=0,
        max_pair_skew_ns=1_000_000_000,
        min_robot_samples=1,
        max_robot_samples=8,
        min_quest_samples=1,
        max_quest_samples=8,
        max_robot_translation_excursion_m=0.01,
        max_robot_rotation_excursion_deg=1.0,
        max_quest_translation_excursion_m=0.01,
        max_quest_rotation_excursion_deg=1.0,
        max_quest_reorder_events=8,
        max_event_history=max_event_history,
        max_completed_windows=4,
    )


def _source_bundle():
    receiver_clock = _receiver_clock()
    monotonic_ns_clock = _SequenceClock()
    journal = _RobotJournal(receiver_clock)
    bindings = {
        side: _coordinate_binding(side=side, receiver_clock=receiver_clock)
        for side in ("left", "right")
    }
    source = PassiveRobotPoseJournalSource(
        receiver_clock=receiver_clock,
        monotonic_ns_clock=monotonic_ns_clock,
        coordinate_binding_by_side=bindings,
        _atomic_cut=journal.atomic_cut,
        _events_after=journal.events_after,
        _latest_pose_measurement=lambda side: journal.atomic_cut().latest_by_stream.get(
            bindings[side].stream_id
        ),
    )
    return receiver_clock, monotonic_ns_clock, journal, bindings, source


def _robot_accepted_event(
    *,
    event_sequence: int,
    source_sequence: int,
    received_monotonic_ns: int,
    binding: PoseCoordinateBinding,
    receiver_clock: ClockDomainDescriptor,
) -> PoseMeasurementAcceptedEvent:
    measured_at = 10.0 + source_sequence * 0.01
    available_at = 20.0 + source_sequence * 0.01
    measurement = ExecutionMeasurementObservation(
        measurement_id=f"robot-pose-{source_sequence}",
        source_measurement_hash=f"robot-source-{source_sequence}",
        control_domain_id=binding.control_domain_id,
        measured_at=measured_at,
        available_at=available_at,
        clock_domain_id=binding.measurement_clock_domain_id,
        value_by_component={
            binding.component_id: (0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        },
        representation_by_component={binding.component_id: "cartesian_pose"},
        schema_hash_by_component={binding.component_id: binding.pose_schema_hash},
        observation_binding_id=binding.observation_binding_id,
        observation_plan_hash=binding.observation_plan_hash,
        parent_facts=(
            ExecutionFactReference(
                stage=POSE_COORDINATE_PARENT_STAGE_V1,
                fact_id=binding.binding_id,
                fact_hash=binding.semantic_content_hash,
                observed_at=available_at,
                clock_domain_id=binding.measurement_clock_domain_id,
                status="applied",
            ),
        ),
    )
    publication = PoseMeasurementPublication(
        coordinate_binding=binding,
        source_sequence=source_sequence,
        measurement=measurement,
        sample_ref=ObservationSampleRef(
            stream_id=binding.stream_id,
            sample_id=measurement.measurement_id,
            sequence=source_sequence,
            measured_at=measurement.measured_at,
            measured_clock_domain_id=measurement.clock_domain_id,
            available_at=received_monotonic_ns / 1_000_000_000.0,
            available_clock_domain_id=binding.receipt_clock_domain_id,
            payload_object_id=measurement.semantic_content_hash,
            content_metadata={
                "coordinate_binding_hash": binding.semantic_content_hash,
                "source_instance_id": binding.source_instance_id,
                "source_epoch_id": binding.source_epoch_id,
                "source_kind": binding.source_kind,
                "source_sequence": source_sequence,
                "reference_frame_id": binding.reference_frame_id,
                "reference_frame_spec_hash": binding.reference_frame_spec_hash,
                "moving_frame_id": binding.moving_frame_id,
                "tool_profile_hash": binding.tool_profile_hash,
                "measurement_available_at": measurement.available_at,
                "measurement_available_clock_domain_id": measurement.clock_domain_id,
                "received_monotonic_ns": received_monotonic_ns,
            },
        ),
        received_monotonic_ns=received_monotonic_ns,
        receiver_clock=receiver_clock,
    )
    return PoseMeasurementAcceptedEvent(
        event_sequence=event_sequence,
        publication=publication,
    )


def _runtime(receiver_clock, monotonic_ns_clock, *, cls=ControllerPoseTelemetryRuntime):
    server = _QuestServer(monotonic_ns_clock)
    return cls(
        server,
        {
            "receipt_clock_domain_id": receiver_clock.clock_domain_id,
            "admission": {
                "max_rate_hz": 1000,
                "max_controllers": 2,
                "max_observers": 4,
            },
        },
        threading.Event(),
        monotonic_ns_clock=monotonic_ns_clock,
    )


def _session_payload(*, generation: int = 0, state: str = "started") -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "client_instance_id": "quest-page-1",
        "xr_session_id": "xr-session-1",
        "reference_space_type": "local-floor",
        "reference_space_generation": generation,
        "state": state,
    }


def _sample_payload(
    *,
    sample_sequence: int,
    received_monotonic_ns: int,
    generation: int = 0,
    source_session_id: str = "controller-source-1",
    handedness: str = "right",
    target_ray_mode: str = "tracked-pointer",
    profiles: list[str] | None = None,
    emulated_position: bool = False,
) -> dict:
    return {
        "type": "controller_pose_sample",
        "feature": FEATURE_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "feature_protocol_version": FEATURE_PROTOCOL_VERSION,
        "client_instance_id": "quest-page-1",
        "xr_session_id": "xr-session-1",
        "reference_space": {"type": "local-floor", "generation": generation},
        "sample_sequence": sample_sequence,
        "xr_frame_time_ms": float(1000 + sample_sequence),
        "controllers": [
            {
                "source_session_id": source_session_id,
                "handedness": handedness,
                "target_ray_mode": target_ray_mode,
                "profiles": (["oculus-touch-v3"] if profiles is None else profiles),
                "emulated_position": emulated_position,
                "matrix_reference_from_grip": list(IDENTITY_MATRIX),
            }
        ],
        "server_timestamp": 1.0,
        "server_monotonic_ns": received_monotonic_ns,
    }


def _emit_candidate(
    runtime: ControllerPoseTelemetryRuntime,
    monotonic_ns_clock: _SequenceClock,
    *,
    sample_sequence: int = 0,
) -> None:
    runtime._handle_session_state(1, _session_payload())
    runtime._handle_pose_sample(
        1,
        _sample_payload(
            sample_sequence=sample_sequence,
            received_monotonic_ns=monotonic_ns_clock(),
        ),
    )


def _application(*, runtime_cls=ControllerPoseTelemetryRuntime):
    receiver_clock, clock, journal, bindings, source = _source_bundle()
    runtime = _runtime(receiver_clock, clock, cls=runtime_cls)
    app = QuestPairedPoseCaptureApplication(
        runtime=runtime,
        robot_pose_journal=source,
        profile=_profile(),
        capture_id="capture-test-1",
        callback_drain_timeout_s=0.1,
    )
    return receiver_clock, clock, journal, bindings, source, runtime, app


def _select_right(app: QuestPairedPoseCaptureApplication):
    candidate = app.candidates()[0]
    binding = app.select(
        intent=QuestCaptureSelectionIntent(
            candidate_hash=candidate.candidate_hash,
            robot_side="right",
        ),
        frame_facts=QuestCaptureFrameFacts(
            candidate_hash=candidate.candidate_hash,
            reference_frame_id="quest-local-floor-generation-0",
            moving_frame_id="quest-controller-grip-source-1",
            xr_frame_clock_domain_id="webxr-frame-clock:xr-session-1",
        ),
    )
    return candidate, binding


def test_runtime_loader_defensively_injects_the_exact_shared_clock() -> None:
    receiver_clock = _receiver_clock()
    monotonic_ns_clock = _SequenceClock()
    server = _QuestServer(monotonic_ns_clock)
    original = {"enabled": True, "admission": {"max_observers": 2}}
    loader = build_quest_controller_runtime_loader(
        receiver_clock=receiver_clock,
        monotonic_ns_clock=monotonic_ns_clock,
    )

    runtime = loader("controller_pose", server, original, threading.Event())

    assert isinstance(runtime, ControllerPoseTelemetryRuntime)
    assert runtime.receipt_clock_domain_id == receiver_clock.clock_domain_id
    assert runtime.monotonic_ns_clock is monotonic_ns_clock
    assert original == {"enabled": True, "admission": {"max_observers": 2}}


def test_runtime_loader_rejects_conflicting_clock_without_mutating_config() -> None:
    receiver_clock = _receiver_clock()
    monotonic_ns_clock = _SequenceClock()
    server = _QuestServer(monotonic_ns_clock)
    config = {"receipt_clock_domain_id": "another-clock"}
    loader = build_quest_controller_runtime_loader(
        receiver_clock=receiver_clock,
        monotonic_ns_clock=monotonic_ns_clock,
    )

    with pytest.raises(ValueError, match="conflicts"):
        loader("controller_pose", server, config, threading.Event())

    assert config == {"receipt_clock_domain_id": "another-clock"}


def test_candidate_is_pose_redacted_and_never_auto_selects() -> None:
    _, clock, journal, _, source, runtime, app = _application()
    _emit_candidate(runtime, clock)

    status = app.snapshot_status()
    encoded = json.dumps(status)

    assert status["state"] == "waiting_for_selection"
    assert status["candidate_count"] == 1
    assert status["selected_candidate_hash"] is None
    assert "matrix" not in encoded
    assert "pairing" not in encoded
    assert not hasattr(source, "snapshot")
    assert not hasattr(source, "get_state")
    assert not hasattr(source, "refresh")
    assert (
        journal.snapshot_calls == journal.get_state_calls == journal.refresh_calls == 0
    )
    app.close()


def test_selected_capture_publishes_live_tcp_from_the_same_passive_journal() -> None:
    receiver_clock, clock, journal, bindings, _, runtime, app = _application()
    _emit_candidate(runtime, clock)
    _, selected = _select_right(app)
    publication_event = _robot_accepted_event(
        event_sequence=1,
        source_sequence=1,
        received_monotonic_ns=clock(),
        binding=bindings["right"],
        receiver_clock=receiver_clock,
    )
    journal.events.append(publication_event)

    app.poll()

    feature, message = runtime.server.published_feature_messages[-1]
    record = message["robots"]["rm65_right"]
    assert feature == "visualization"
    assert message["type"] == "robot_tcp"
    assert record["source"] == "passive_pose_journal"
    assert record["fresh"] is True
    assert record["alignment_ready"] is False
    assert record["pose_binding"]["coordinate_binding_hash"] == (
        selected.robot.semantic_content_hash
    )
    assert record["matrix_robot_from_tcp"][12:15] == [0.2, 0.0, 0.0]
    assert (
        journal.snapshot_calls == journal.get_state_calls == journal.refresh_calls == 0
    )
    app.close()


def test_candidate_projection_recomputes_its_public_hash() -> None:
    _, clock, _, _, _, runtime, app = _application()
    _emit_candidate(runtime, clock)

    candidate = dict(app.snapshot_status()["candidates"][0])
    projected_hash = candidate.pop("candidate_hash")

    assert projected_hash == canonical_content_hash(
        candidate,
        domain=QUEST_CONTROLLER_CANDIDATE_HASH_DOMAIN,
    )
    app.close()


def test_explicit_selection_pins_frame_facts_and_cut_conventions() -> None:
    _, clock, journal, bindings, _, runtime, app = _application()
    _emit_candidate(runtime, clock)

    candidate, binding = _select_right(app)
    status = app.poll()

    assert binding.robot is bindings["right"]
    assert binding.robot_next_event_sequence == 1
    assert binding.quest.next_event_sequence == 1
    assert binding.quest.baseline_sample_sequence == 0
    assert binding.quest.reference_frame_id == "quest-local-floor-generation-0"
    assert binding.quest.moving_frame_id == "quest-controller-grip-source-1"
    assert binding.quest.xr_frame_clock_domain_id == "webxr-frame-clock:xr-session-1"
    assert binding.quest.controller_source_session_id == (
        candidate.controller_source_session_id
    )
    assert status["capture"]["quest_next_event_sequence"] == 1
    assert (
        journal.snapshot_calls == journal.get_state_calls == journal.refresh_calls == 0
    )
    app.close()
    assert app.evidence is None


def test_selection_rebases_robot_cut_after_long_operator_delay() -> None:
    receiver_clock, clock, journal, bindings, _, runtime, app = _application()
    for event_sequence in range(1, 301):
        journal.events.append(
            PoseMeasurementInvalidationEvent(
                event_sequence=event_sequence,
                coordinate_binding=bindings["left"],
                reason="preselection_observation",
                received_monotonic_ns=clock(),
                receiver_clock=receiver_clock,
            )
        )
    _emit_candidate(runtime, clock)

    _, binding = _select_right(app)

    assert binding.robot_next_event_sequence == 301
    assert app.poll()["capture"]["robot_next_event_sequence"] == 301
    app.close()


def test_discovery_does_not_retain_high_rate_pose_history_before_selection() -> None:
    _, clock, _, _, _, runtime, app = _application()
    _emit_candidate(runtime, clock)
    for sample_sequence in range(1, 100):
        runtime._handle_pose_sample(
            1,
            _sample_payload(
                sample_sequence=sample_sequence,
                received_monotonic_ns=clock(),
            ),
        )

    status = app.snapshot_status()
    assert status["candidate_count"] == 1
    assert status["staged_quest_event_count"] == 0

    _, binding = _select_right(app)
    assert binding.quest.next_event_sequence == 100
    assert binding.quest.baseline_sample_sequence == 99
    app.close()


def test_subscription_active_session_supplies_exact_baseline() -> None:
    receiver_clock, clock, _, _, source = _source_bundle()
    runtime = _runtime(receiver_clock, clock)
    runtime._handle_session_state(1, _session_payload())
    runtime._handle_pose_sample(
        1,
        _sample_payload(sample_sequence=4, received_monotonic_ns=clock()),
    )
    app = QuestPairedPoseCaptureApplication(
        runtime=runtime,
        robot_pose_journal=source,
        profile=_profile(),
        capture_id="capture-baseline",
        callback_drain_timeout_s=0.1,
    )
    runtime._handle_pose_sample(
        1,
        _sample_payload(sample_sequence=5, received_monotonic_ns=clock()),
    )

    _, binding = _select_right(app)

    assert binding.quest.next_event_sequence == 2
    assert binding.quest.baseline_sample_sequence == 5
    app.close()


def test_staged_invalidation_prevents_selected_epoch_resurrection() -> None:
    _, clock, _, _, _, runtime, app = _application()
    _emit_candidate(runtime, clock)
    candidate = app.candidates()[0]
    runtime._handle_reference_space_reset(
        1,
        {
            "protocol_version": PROTOCOL_VERSION,
            "client_instance_id": "quest-page-1",
            "xr_session_id": "xr-session-1",
            "reference_space_type": "local-floor",
            "previous_generation": 0,
            "new_generation": 1,
        },
    )

    assert app.candidates() == ()
    with pytest.raises(KeyError, match="not observed"):
        app.select(
            intent=QuestCaptureSelectionIntent(
                candidate_hash=candidate.candidate_hash,
                robot_side="right",
            ),
            frame_facts=QuestCaptureFrameFacts(
                candidate_hash=candidate.candidate_hash,
                reference_frame_id="quest-local-floor-generation-0",
                moving_frame_id="quest-controller-grip-source-1",
                xr_frame_clock_domain_id="webxr-frame-clock:xr-session-1",
            ),
        )

    assert app.snapshot_status()["state"] == "waiting_for_selection"
    app.close()


def test_discovery_can_select_a_new_epoch_after_reference_reset() -> None:
    _, clock, _, _, _, runtime, app = _application()
    _emit_candidate(runtime, clock)
    runtime._handle_reference_space_reset(
        1,
        {
            "protocol_version": PROTOCOL_VERSION,
            "client_instance_id": "quest-page-1",
            "xr_session_id": "xr-session-1",
            "reference_space_type": "local-floor",
            "previous_generation": 0,
            "new_generation": 1,
        },
    )
    runtime._handle_pose_sample(
        1,
        _sample_payload(
            sample_sequence=1,
            received_monotonic_ns=clock(),
            generation=1,
        ),
    )

    candidate, binding = _select_right(app)

    assert candidate.reference_space_generation == 1
    assert binding.quest.reference_space_generation == 1
    assert binding.quest.baseline_sample_sequence == 1
    app.close()


def test_terminal_robot_invalidation_blocks_selection_at_fresh_cut() -> None:
    receiver_clock, clock, journal, bindings, _, runtime, app = _application()
    journal.events.append(
        PoseMeasurementInvalidationEvent(
            event_sequence=1,
            coordinate_binding=bindings["right"],
            reason="terminal_robot_stream",
            received_monotonic_ns=clock(),
            receiver_clock=receiver_clock,
        )
    )
    _emit_candidate(runtime, clock)

    with pytest.raises(
        QuestPairedPoseCaptureTerminalError,
        match="terminal-invalidated",
    ):
        _select_right(app)

    app.close()


def test_same_binding_rearm_can_select_after_transient_invalidation() -> None:
    receiver_clock, clock, journal, bindings, _, runtime, app = _application()
    journal.events.append(
        PoseMeasurementInvalidationEvent(
            event_sequence=1,
            coordinate_binding=bindings["right"],
            reason="transient_projection_failure",
            received_monotonic_ns=clock(),
            receiver_clock=receiver_clock,
            next_coordinate_binding=bindings["right"],
        )
    )
    _emit_candidate(runtime, clock)

    _, binding = _select_right(app)

    assert binding.robot_next_event_sequence == 2
    app.close()


def test_out_of_order_old_candidate_callback_cannot_resurrect_epoch() -> None:
    receiver_clock, clock, _, _, source = _source_bundle()
    runtime = _runtime(receiver_clock, clock)
    app = QuestPairedPoseCaptureApplication(
        runtime=runtime,
        robot_pose_journal=source,
        profile=_profile(),
        capture_id="capture-out-of-order",
        callback_drain_timeout_s=0.1,
    )
    sample = parse_controller_pose_sample(
        _sample_payload(sample_sequence=0, received_monotonic_ns=clock()),
        received_wall_time_s=1.0,
        received_monotonic_ns=clock(),
    )
    accepted = ControllerPoseAcceptedEvent(
        event_sequence=0,
        server_instance_id=runtime.server_instance_id,
        client_id=1,
        sample=sample,
        receipt_clock_domain_id=runtime.receipt_clock_domain_id,
    )
    invalidation = ControllerPoseInvalidationEvent(
        event_sequence=1,
        server_instance_id=runtime.server_instance_id,
        client_id=1,
        client_instance_id=sample.client_instance_id,
        xr_session_id=sample.xr_session_id,
        reference_space_type=sample.reference_space_type,
        previous_reference_space_generation=sample.reference_space_generation,
        next_reference_space_generation=1,
        last_accepted_sample_sequence=sample.sample_sequence,
        reason="reference_space_reset",
        received_monotonic_ns=clock(),
        receipt_clock_domain_id=runtime.receipt_clock_domain_id,
    )

    app._on_quest_event(invalidation)
    assert app.candidates() == ()
    app._on_quest_event(accepted)

    assert app.candidates() == ()
    assert app.snapshot_status()["terminal_reason"] is None
    app.close()


def test_dynamic_profiles_do_not_duplicate_one_quest_candidate() -> None:
    _, clock, _, _, _, runtime, app = _application()
    _emit_candidate(runtime, clock)
    runtime._handle_pose_sample(
        1,
        _sample_payload(
            sample_sequence=1,
            received_monotonic_ns=clock(),
            profiles=["oculus-touch-v3", "generic-trigger"],
        ),
    )

    candidates = app.candidates()
    assert len(candidates) == 1
    assert candidates[0].profiles == ("oculus-touch-v3",)

    _select_right(app)
    runtime._handle_pose_sample(
        1,
        _sample_payload(
            sample_sequence=1,
            received_monotonic_ns=clock(),
            profiles=["oculus-touch-v3", "generic-trigger"],
        ),
    )
    app.poll()
    assert app.snapshot_status()["terminal_reason"] is None
    app.close()


def test_get_pose_loss_empty_sample_does_not_create_or_poison_candidate() -> None:
    _, clock, _, _, _, runtime, app = _application()
    runtime._handle_session_state(1, _session_payload())
    empty_sample = _sample_payload(
        sample_sequence=0,
        received_monotonic_ns=clock(),
    )
    empty_sample["controllers"] = []
    empty_sample["controller_diagnostics"] = [
        {
            "source_session_id": "controller-source-1",
            "handedness": "right",
            "target_ray_mode": "tracked-pointer",
            "tracking_state": "untracked",
            "reason": "pose_unavailable",
        }
    ]
    runtime._handle_pose_sample(1, empty_sample)
    app.poll()

    assert app.candidates() == ()
    assert app.snapshot_status()["terminal_reason"] is None

    runtime._handle_pose_sample(
        1,
        _sample_payload(
            sample_sequence=1,
            received_monotonic_ns=clock(),
        ),
    )
    app.poll()

    assert len(app.candidates()) == 1
    assert app.snapshot_status()["terminal_reason"] is None
    app.close()


def test_get_pose_loss_after_candidate_disallows_selection_until_pose_returns() -> None:
    _, clock, _, _, _, runtime, app = _application()
    _emit_candidate(runtime, clock)
    candidate = app.candidates()[0]

    empty_sample = _sample_payload(
        sample_sequence=1,
        received_monotonic_ns=clock(),
    )
    empty_sample["controllers"] = []
    empty_sample["controller_diagnostics"] = [
        {
            "source_session_id": candidate.controller_source_session_id,
            "handedness": candidate.handedness,
            "target_ray_mode": candidate.target_ray_mode,
            "tracking_state": "untracked",
            "reason": "pose_unavailable",
        }
    ]
    runtime._handle_pose_sample(1, empty_sample)
    app.poll()

    # The candidate remains a historical discovery record, but the latest
    # source state is empty and must not authorize capture selection.
    assert app.candidates() == (candidate,)
    with pytest.raises(RuntimeError, match="getPose.*returned null"):
        app.select_unique_candidate(
            robot_side="right",
            preferred_handedness="right",
        )
    assert app.snapshot_status()["state"] == "waiting_for_selection"
    assert app.snapshot_status()["terminal_reason"] is None

    runtime._handle_pose_sample(
        1,
        _sample_payload(
            sample_sequence=2,
            received_monotonic_ns=clock(),
        ),
    )
    app.poll()
    _, binding = _select_right(app)
    assert binding.quest.controller_source_session_id == (
        candidate.controller_source_session_id
    )
    app.close()


def test_stale_latest_pose_cannot_start_capture() -> None:
    _, clock, _, _, _, runtime, app = _application()
    _emit_candidate(runtime, clock)
    clock.value += 1_000_000_000

    with pytest.raises(RuntimeError, match=r"telemetry is stale \(1\.100s old"):
        app.select_unique_candidate(
            robot_side="right",
            preferred_handedness="right",
        )
    assert app.snapshot_status()["terminal_reason"] is None
    app.close()


def test_emulated_latest_pose_reports_why_selection_is_unusable() -> None:
    _, clock, _, _, _, runtime, app = _application()
    _emit_candidate(runtime, clock)
    runtime._handle_pose_sample(
        1,
        _sample_payload(
            sample_sequence=1,
            received_monotonic_ns=clock(),
            emulated_position=True,
        ),
    )
    app.poll()

    with pytest.raises(RuntimeError, match="emulated_position.*estimating"):
        app.select_unique_candidate(
            robot_side="right",
            preferred_handedness="right",
        )
    assert app.snapshot_status()["terminal_reason"] is None
    app.close()


def test_selected_invalidation_makes_poll_fail_closed() -> None:
    _, clock, _, _, _, runtime, app = _application()
    _emit_candidate(runtime, clock)
    _select_right(app)
    runtime._handle_reference_space_reset(
        1,
        {
            "protocol_version": PROTOCOL_VERSION,
            "client_instance_id": "quest-page-1",
            "xr_session_id": "xr-session-1",
            "reference_space_type": "local-floor",
            "previous_generation": 0,
            "new_generation": 1,
        },
    )

    with pytest.raises(QuestPairedPoseCaptureTerminalError, match="invalidated"):
        app.poll()

    assert app.snapshot_status()["state"] == "terminal"
    app.close()


def test_selected_out_of_order_invalidation_is_not_lost_by_capture_drain() -> None:
    receiver_clock, clock, _, _, source = _source_bundle()
    runtime = _runtime(receiver_clock, clock)
    app = QuestPairedPoseCaptureApplication(
        runtime=runtime,
        robot_pose_journal=source,
        profile=_profile(),
        capture_id="capture-selected-reorder",
        callback_drain_timeout_s=0.1,
    )
    _emit_candidate(runtime, clock)
    _select_right(app)
    sample = parse_controller_pose_sample(
        _sample_payload(sample_sequence=1, received_monotonic_ns=clock()),
        received_wall_time_s=1.0,
        received_monotonic_ns=clock(),
    )
    accepted = ControllerPoseAcceptedEvent(
        event_sequence=1,
        server_instance_id=runtime.server_instance_id,
        client_id=1,
        sample=sample,
        receipt_clock_domain_id=runtime.receipt_clock_domain_id,
    )
    invalidation = ControllerPoseInvalidationEvent(
        event_sequence=2,
        server_instance_id=runtime.server_instance_id,
        client_id=1,
        client_instance_id=sample.client_instance_id,
        xr_session_id=sample.xr_session_id,
        reference_space_type=sample.reference_space_type,
        previous_reference_space_generation=sample.reference_space_generation,
        next_reference_space_generation=1,
        last_accepted_sample_sequence=sample.sample_sequence,
        reason="reference_space_reset",
        received_monotonic_ns=clock(),
        receipt_clock_domain_id=runtime.receipt_clock_domain_id,
    )

    app._on_quest_event(invalidation)
    first_status = app.poll()
    assert first_status["candidate_count"] == 1
    app._on_quest_event(accepted)

    with pytest.raises(QuestPairedPoseCaptureTerminalError, match="invalidated"):
        app.poll()

    assert app.candidates() == ()
    app.close()


def test_rejected_window_does_not_poison_the_next_window_boundary() -> None:
    _, clock, _, _, _, runtime, app = _application()
    _emit_candidate(runtime, clock)
    _select_right(app)

    app.begin_window(window_id="rejected-window-1")
    with pytest.raises(StaticWindowRejectedError, match="sample count"):
        app.complete_window()

    app.begin_window(window_id="rejected-window-2")
    with pytest.raises(StaticWindowRejectedError, match="sample count"):
        app.complete_window()

    assert app.snapshot_status()["capture"]["active_window_id"] is None
    app.close()


def test_poll_advances_global_robot_cursor_through_unselected_side() -> None:
    receiver_clock, clock, journal, bindings, _, runtime, app = _application()
    _emit_candidate(runtime, clock)
    _select_right(app)
    journal.events.append(
        PoseMeasurementInvalidationEvent(
            event_sequence=1,
            coordinate_binding=bindings["left"],
            reason="left_stream_probe",
            received_monotonic_ns=clock(),
            receiver_clock=receiver_clock,
        )
    )

    status = app.poll()

    assert status["capture"]["robot_next_event_sequence"] == 2
    assert status["terminal_reason"] is None
    app.close()


def test_callback_arriving_before_subscription_return_is_staged() -> None:
    receiver_clock, clock, _, _, source = _source_bundle()
    sample: ControllerPoseSample = parse_controller_pose_sample(
        _sample_payload(sample_sequence=0, received_monotonic_ns=clock()),
        received_wall_time_s=1.0,
        received_monotonic_ns=clock(),
    )

    class _RaceRuntime(ControllerPoseTelemetryRuntime):
        def subscribe_with_cut(self, *, on_sample=None, on_invalidation=None):
            subscription = super().subscribe_with_cut(
                on_sample=on_sample,
                on_invalidation=on_invalidation,
            )
            assert on_sample is not None
            on_sample(
                ControllerPoseAcceptedEvent(
                    event_sequence=subscription.next_event_sequence,
                    server_instance_id=self.server_instance_id,
                    client_id=1,
                    sample=sample,
                    receipt_clock_domain_id=self.receipt_clock_domain_id,
                )
            )
            return subscription

    runtime = _runtime(receiver_clock, clock, cls=_RaceRuntime)

    app = QuestPairedPoseCaptureApplication(
        runtime=runtime,
        robot_pose_journal=source,
        profile=_profile(),
        capture_id="capture-race",
        callback_drain_timeout_s=0.1,
    )

    assert len(app.candidates()) == 1
    assert app.snapshot_status()["subscription_next_event_sequence"] == 0
    app.close()


def test_malformed_discovery_close_cut_does_not_reunsubscribe_removed_token() -> None:
    _, clock, _, _, _, runtime, app = _application()
    _emit_candidate(runtime, clock)
    original = runtime.unsubscribe_with_cut

    def malformed_close(token):
        return replace(original(token), server_instance_id="drifted-server")

    with mock.patch.object(
        runtime,
        "unsubscribe_with_cut",
        side_effect=malformed_close,
    ) as unsubscribe:
        with pytest.raises(
            QuestPairedPoseCaptureTerminalError,
            match="close cut drifted",
        ):
            _select_right(app)
        app.close()

    assert unsubscribe.call_count == 1
    assert runtime._observers == {}


def test_malformed_capture_subscription_is_rolled_back() -> None:
    receiver_clock, clock, _, _, source = _source_bundle()

    class _MalformedSecondSubscriptionRuntime(ControllerPoseTelemetryRuntime):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.subscribe_calls = 0

        def subscribe_with_cut(self, **kwargs):
            subscription = super().subscribe_with_cut(**kwargs)
            self.subscribe_calls += 1
            if self.subscribe_calls == 2:
                return replace(subscription, server_instance_id="drifted-server")
            return subscription

    runtime = _runtime(
        receiver_clock,
        clock,
        cls=_MalformedSecondSubscriptionRuntime,
    )
    app = QuestPairedPoseCaptureApplication(
        runtime=runtime,
        robot_pose_journal=source,
        profile=_profile(),
        capture_id="capture-malformed-rotation",
        callback_drain_timeout_s=0.1,
    )
    _emit_candidate(runtime, clock)

    with pytest.raises(
        QuestPairedPoseCaptureTerminalError,
        match="subscription cut failed",
    ):
        _select_right(app)

    assert runtime._observers == {}
    app.close()


def test_capture_rotation_rejects_event_cut_regression() -> None:
    receiver_clock, clock, _, _, source = _source_bundle()

    class _RegressedSecondSubscriptionRuntime(ControllerPoseTelemetryRuntime):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.subscribe_calls = 0

        def subscribe_with_cut(self, **kwargs):
            subscription = super().subscribe_with_cut(**kwargs)
            self.subscribe_calls += 1
            if self.subscribe_calls == 2:
                return replace(subscription, next_event_sequence=0)
            return subscription

    runtime = _runtime(
        receiver_clock,
        clock,
        cls=_RegressedSecondSubscriptionRuntime,
    )
    app = QuestPairedPoseCaptureApplication(
        runtime=runtime,
        robot_pose_journal=source,
        profile=_profile(),
        capture_id="capture-regressed-rotation",
        callback_drain_timeout_s=0.1,
    )
    _emit_candidate(runtime, clock)

    with pytest.raises(QuestPairedPoseCaptureTerminalError, match="cut regressed"):
        _select_right(app)

    app.close()


def test_close_is_idempotent_and_never_implicitly_seals() -> None:
    _, clock, _, _, _, runtime, app = _application()
    _emit_candidate(runtime, clock)
    _select_right(app)
    original = runtime.unsubscribe_with_cut
    with mock.patch.object(
        runtime, "unsubscribe_with_cut", wraps=original
    ) as unsubscribe:
        app.close()
        app.close()

    assert unsubscribe.call_count == 1
    assert app.evidence is None
    assert app.snapshot_status()["closed"] is True


def test_unsubscribe_failure_is_visible_and_close_can_retry() -> None:
    _, _, _, _, _, runtime, app = _application()

    with (
        mock.patch.object(
            runtime,
            "unsubscribe_with_cut",
            side_effect=RuntimeError("unsubscribe-probe"),
        ),
        pytest.raises(QuestPairedPoseCaptureCleanupError, match="unsubscribe"),
    ):
        app.close()

    assert app.closed is False
    app.close()
    assert app.closed is True


def test_base_exception_during_unsubscribe_never_commits_closed() -> None:
    _, _, _, _, _, runtime, app = _application()

    with (
        mock.patch.object(
            runtime,
            "unsubscribe_with_cut",
            side_effect=KeyboardInterrupt,
        ),
        pytest.raises(KeyboardInterrupt),
    ):
        app.close()

    assert app.closed is False
    app.close()
    assert app.closed is True


def test_close_cancels_inflight_seal_before_returning() -> None:
    _, clock, _, _, _, runtime, app = _application()
    _emit_candidate(runtime, clock)
    _select_right(app)
    seal_started = threading.Event()

    class _DrainingCapture:
        def snapshot_status(self):
            return {
                "completed_window_count": 1,
                "active_window_id": None,
                "quest_next_event_sequence": 0,
            }

        def ingest_quest_event(self, _event):
            return None

        def ingest_robot_event(self, _event):
            return None

        def seal_capture(self, **_kwargs):
            seal_started.set()

    app._capture = _DrainingCapture()  # type: ignore[assignment]
    seal_errors: list[Exception] = []

    def seal() -> None:
        try:
            app.request_seal()
        except Exception as exc:  # noqa: BLE001 - asserted below.
            seal_errors.append(exc)

    seal_thread = threading.Thread(target=seal)
    seal_thread.start()
    assert seal_started.wait(timeout=0.5)

    app.close()
    seal_thread.join(timeout=0.5)

    assert not seal_thread.is_alive()
    assert len(seal_errors) == 1
    assert isinstance(seal_errors[0], QuestPairedPoseCaptureTerminalError)
    assert app.evidence is None
    assert app.closed is True


def test_request_seal_is_explicit_and_uses_one_atomic_unsubscribe() -> None:
    _, clock, _, _, _, runtime, app = _application()
    _emit_candidate(runtime, clock)
    _select_right(app)
    evidence = mock.Mock(semantic_content_hash="capture-evidence-hash")

    class _Capture:
        def __init__(self) -> None:
            self.seal_calls = []

        def snapshot_status(self):
            return {
                "completed_window_count": 1,
                "active_window_id": None,
                "quest_next_event_sequence": 1,
            }

        def ingest_quest_event(self, _event):
            raise AssertionError("all staged Quest events were already drained")

        def ingest_robot_event(self, _event):
            raise AssertionError("the fake robot journal is empty")

        def seal_capture(self, *, robot_cut, quest_cut):
            self.seal_calls.append((robot_cut, quest_cut))
            return evidence

    capture = _Capture()
    app._capture = capture  # type: ignore[assignment]
    original = runtime.unsubscribe_with_cut
    with mock.patch.object(
        runtime, "unsubscribe_with_cut", wraps=original
    ) as unsubscribe:
        result = app.request_seal()

    assert result is evidence
    assert unsubscribe.call_count == 1
    assert len(capture.seal_calls) == 1
    assert capture.seal_calls[0][0].event_sequence == 0
    assert capture.seal_calls[0][1].next_event_sequence == 1
    app.close()
    assert unsubscribe.call_count == 1


def test_real_reducer_completes_and_seals_one_typed_static_window() -> None:
    receiver_clock, clock, journal, bindings, _, runtime, app = _application()
    _emit_candidate(runtime, clock)
    _select_right(app)
    app.begin_window(window_id="typed-static-window")
    journal.events.append(
        _robot_accepted_event(
            event_sequence=1,
            source_sequence=1,
            received_monotonic_ns=clock(),
            binding=bindings["right"],
            receiver_clock=receiver_clock,
        )
    )
    runtime._handle_pose_sample(
        1,
        _sample_payload(
            sample_sequence=1,
            received_monotonic_ns=clock(),
        ),
    )

    receipt = app.complete_window()
    evidence = app.request_seal()

    assert receipt is not None
    assert receipt.robot_start_next_event_sequence == 1
    assert receipt.robot_end_next_event_sequence == 2
    assert receipt.quest_start_next_event_sequence == 1
    assert receipt.quest_end_next_event_sequence == 2
    assert evidence.windows == (receipt,)
    assert evidence.finalization.robot_end_next_event_sequence == 2
    assert evidence.finalization.quest_end_next_event_sequence == 2
    assert app.evidence is evidence

    journal.events.append(
        _robot_accepted_event(
            event_sequence=2,
            source_sequence=2,
            received_monotonic_ns=clock(),
            binding=bindings["right"],
            receiver_clock=receiver_clock,
        )
    )
    app.poll()
    _, live_message = runtime.server.published_feature_messages[-1]
    live_record = live_message["robots"]["rm65_right"]
    assert live_record["fresh"] is True
    assert live_record["source_sequence"] == 2
    assert live_record["pose_binding"]["coordinate_binding_hash"] == (
        evidence.binding.robot.semantic_content_hash
    )
    app.close()


class _FakeOwnedHost:
    def __init__(
        self,
        runtime: ControllerPoseTelemetryRuntime,
        *,
        stop_on_request: bool = True,
        request_error: BaseException | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self.runtime = runtime
        self.launch_info = QuestDevLaunchInfo(
            base_url="https://192.0.2.10:8444/",
            pairing_url="https://192.0.2.10:8444/#pair=123456",
            pairing_code="123456",
            use_https=True,
        )
        self._shutdown = threading.Event()
        self.stop_on_request = stop_on_request
        self.request_error = request_error
        self.close_error = close_error
        self.close_calls = 0

    def require_running_feature_runtime(self, name, expected_type):
        assert name == "controller_pose"
        assert expected_type is ControllerPoseTelemetryRuntime
        return self.runtime

    def serve_forever(self, *, ready_callback=None):
        assert ready_callback is not None
        ready_callback()
        self._shutdown.wait(timeout=2.0)

    def request_shutdown(self):
        if self.request_error is not None:
            raise self.request_error
        if self.stop_on_request:
            self._shutdown.set()
        return True

    def close(self):
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class _FaultingCloseServer:
    server_instance_id = "faulting-server"

    def __init__(self, close_error: BaseException) -> None:
        self.close_error = close_error
        self.release = threading.Event()
        self.close_calls = 0

    def serve_forever(self) -> None:
        self.release.wait(timeout=0.5)

    def shutdown(self) -> None:
        self.release.set()

    def server_close(self) -> None:
        self.close_calls += 1
        raise self.close_error


def _real_host_with_faulting_close(
    close_error: BaseException,
) -> tuple[_FaultingCloseServer, QuestDevHost]:
    faulting_server = _FaultingCloseServer(close_error)
    launch_info = QuestDevLaunchInfo(
        base_url="https://192.0.2.10:8444/",
        pairing_url="https://192.0.2.10:8444/#pair=123456",
        pairing_code="123456",
        use_https=True,
    )
    host = QuestDevHost(
        faulting_server,  # type: ignore[arg-type]
        {},
        launch_info,
    )
    return faulting_server, host


def _owned_host_application(
    *,
    stop_on_request: bool = True,
    request_error: BaseException | None = None,
    close_error: BaseException | None = None,
    host_error_box: dict[str, BaseException] | None = None,
    join_timeout_s: float = 0.05,
):
    receiver_clock, clock, _, _, source = _source_bundle()
    runtime = _runtime(receiver_clock, clock)
    host = _FakeOwnedHost(
        runtime,
        stop_on_request=stop_on_request,
        request_error=request_error,
        close_error=close_error,
    )
    thread = threading.Thread(target=host._shutdown.wait)
    thread.start()
    app = QuestPairedPoseCaptureApplication(
        runtime=runtime,
        robot_pose_journal=source,
        profile=_profile(),
        capture_id="owned-host-capture",
        callback_drain_timeout_s=0.1,
        owned_host=host,
        owned_host_thread=thread,
        host_join_timeout_s=join_timeout_s,
        host_error_box=host_error_box,
    )
    return app, host, thread


def test_owned_host_join_timeout_is_visible_and_retryable() -> None:
    app, host, thread = _owned_host_application(stop_on_request=False)
    try:
        with pytest.raises(QuestPairedPoseCaptureCleanupError, match="did not stop"):
            app.close()
        assert app.closed is False
    finally:
        host._shutdown.set()
        thread.join(timeout=0.5)

    app.close()
    assert app.closed is True
    assert host.close_calls == 1


def test_owned_host_base_exception_never_commits_closed() -> None:
    app, host, thread = _owned_host_application(request_error=KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        app.close()

    assert app.closed is False
    host.request_error = None
    app.close()
    thread.join(timeout=0.5)
    assert app.closed is True


def test_owned_host_close_failure_is_permanently_latched() -> None:
    app, host, thread = _owned_host_application(
        close_error=RuntimeError("host-close-probe")
    )
    try:
        with pytest.raises(
            QuestPairedPoseCaptureCleanupError, match="host-close-probe"
        ):
            app.close()
        with pytest.raises(
            QuestPairedPoseCaptureCleanupError, match="host-close-probe"
        ):
            app.close()
    finally:
        host._shutdown.set()
        thread.join(timeout=0.5)

    assert app.closed is False
    assert host.close_calls == 1


def test_owned_host_close_base_exception_is_permanently_latched() -> None:
    app, host, thread = _owned_host_application(close_error=KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        app.close()
    with pytest.raises(QuestPairedPoseCaptureCleanupError, match="KeyboardInterrupt"):
        app.close()

    thread.join(timeout=0.5)
    assert app.closed is False
    assert host.close_calls == 1


def test_owned_host_serving_error_cannot_be_washed_into_success() -> None:
    app, _host, thread = _owned_host_application(
        host_error_box={"error": RuntimeError("serve-cleanup-probe")}
    )

    with pytest.raises(QuestPairedPoseCaptureCleanupError, match="serve-cleanup-probe"):
        app.close()
    with pytest.raises(QuestPairedPoseCaptureCleanupError, match="serve-cleanup-probe"):
        app.close()

    thread.join(timeout=0.5)
    assert not thread.is_alive()
    assert app.closed is False


def test_concurrent_close_waits_for_the_same_owned_host_attempt() -> None:
    close_entered = threading.Event()
    release_close = threading.Event()
    app, host, host_thread = _owned_host_application()
    original_close = host.close

    def blocking_close():
        close_entered.set()
        assert release_close.wait(timeout=0.5)
        original_close()

    host.close = blocking_close
    close_errors: list[BaseException] = []

    def close_app() -> None:
        try:
            app.close()
        except BaseException as exc:  # noqa: BLE001 - asserted below.
            close_errors.append(exc)

    first = threading.Thread(target=close_app)
    second = threading.Thread(target=close_app)
    first.start()
    assert close_entered.wait(timeout=0.5)
    second.start()
    assert second.is_alive()
    release_close.set()
    first.join(timeout=0.5)
    second.join(timeout=0.5)
    host_thread.join(timeout=0.5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert close_errors == []
    assert app.closed is True
    assert host.close_calls == 1


def _launch_config(
    *,
    startup_timeout_s: float = 0.5,
    host_join_timeout_s: float = 0.5,
) -> QuestPairedPoseCaptureLaunchConfig:
    return QuestPairedPoseCaptureLaunchConfig(
        config_path=Path("quest-test.yaml"),
        force_http=False,
        profile=_profile(),
        capture_id="factory-capture",
        startup_timeout_s=startup_timeout_s,
        callback_drain_timeout_s=0.1,
        host_join_timeout_s=host_join_timeout_s,
    )


def test_factory_owns_one_programmatic_host_without_network_or_device_io() -> None:
    receiver_clock, clock, journal, _, source = _source_bundle()
    runtime = _runtime(receiver_clock, clock)
    fake_host = _FakeOwnedHost(runtime)
    builder_calls: list[tuple[tuple, dict]] = []

    def host_builder(*args, **kwargs):
        builder_calls.append((args, kwargs))
        return fake_host

    factory = QuestPairedPoseCaptureFactory(
        _launch_config(),
        host_builder=host_builder,
    )

    app = factory(robot_pose_journal=source)

    assert factory.application is app
    assert factory.wait_for_application(timeout_s=0.1) is app
    assert factory.launch_info is fake_host.launch_info
    assert len(builder_calls) == 1
    assert builder_calls[0][1]["monotonic_ns_clock"] is clock
    assert callable(builder_calls[0][1]["runtime_loader"])
    assert "realman_left_ip" not in builder_calls[0][1]
    assert "realman_left_urdf" not in builder_calls[0][1]
    assert (
        journal.snapshot_calls == journal.get_state_calls == journal.refresh_calls == 0
    )
    with pytest.raises(RuntimeError, match="single-use"):
        factory(robot_pose_journal=source)
    app.close()
    factory.close()
    assert fake_host.close_calls == 1


def test_factory_close_before_call_prevents_late_host_start() -> None:
    receiver_clock, clock, _, _, source = _source_bundle()
    runtime = _runtime(receiver_clock, clock)
    fake_host = _FakeOwnedHost(runtime)
    factory = QuestPairedPoseCaptureFactory(
        _launch_config(),
        host_builder=lambda *_args, **_kwargs: fake_host,
    )

    factory.close()

    with pytest.raises(RuntimeError, match="closed"):
        factory(robot_pose_journal=source)
    assert fake_host.close_calls == 0


def test_factory_thread_start_failure_closes_constructed_host() -> None:
    receiver_clock, clock, _, _, source = _source_bundle()
    runtime = _runtime(receiver_clock, clock)
    fake_host = _FakeOwnedHost(runtime)
    factory = QuestPairedPoseCaptureFactory(
        _launch_config(),
        host_builder=lambda *_args, **_kwargs: fake_host,
    )

    with (
        mock.patch.object(
            threading.Thread,
            "start",
            side_effect=RuntimeError("thread-start-probe"),
        ),
        pytest.raises(QuestPairedPoseCaptureStartupError, match="startup failed"),
    ):
        factory(robot_pose_journal=source)

    assert fake_host.close_calls == 1
    factory.close()


def test_factory_interrupted_after_thread_started_still_joins_host() -> None:
    receiver_clock, clock, _, _, source = _source_bundle()
    runtime = _runtime(receiver_clock, clock)
    fake_host = _FakeOwnedHost(runtime)
    factory = QuestPairedPoseCaptureFactory(
        _launch_config(),
        host_builder=lambda *_args, **_kwargs: fake_host,
    )
    original_start = threading.Thread.start

    def start_then_interrupt(thread):
        original_start(thread)
        raise KeyboardInterrupt

    with (
        mock.patch.object(threading.Thread, "start", new=start_then_interrupt),
        pytest.raises(KeyboardInterrupt),
    ):
        factory(robot_pose_journal=source)

    factory.close()
    assert fake_host.close_calls == 1


def test_factory_builder_keyboard_interrupt_clears_starting_latch() -> None:
    _, _, _, _, source = _source_bundle()

    def interrupted_builder(*_args, **_kwargs):
        raise KeyboardInterrupt

    factory = QuestPairedPoseCaptureFactory(
        _launch_config(),
        host_builder=interrupted_builder,
    )

    with pytest.raises(KeyboardInterrupt):
        factory(robot_pose_journal=source)

    factory.close()
    with pytest.raises(RuntimeError, match="startup failed"):
        factory.wait_for_application(timeout_s=0.01)


def test_factory_serving_failure_still_closes_the_dead_host() -> None:
    receiver_clock, clock, _, _, source = _source_bundle()
    runtime = _runtime(receiver_clock, clock)
    fake_host = _FakeOwnedHost(runtime)

    def fail_after_ready(*, ready_callback=None):
        assert ready_callback is not None
        ready_callback()
        raise RuntimeError("serve-startup-probe")

    fake_host.serve_forever = fail_after_ready
    factory = QuestPairedPoseCaptureFactory(
        _launch_config(),
        host_builder=lambda *_args, **_kwargs: fake_host,
    )

    with pytest.raises(QuestPairedPoseCaptureStartupError, match="startup failed"):
        factory(robot_pose_journal=source)

    assert fake_host.close_calls == 1
    assert factory.cleanup_pending is False
    factory.close()


def test_factory_retains_real_host_after_latched_server_close_failure() -> None:
    _, _, _, _, source = _source_bundle()
    faulting_server, host = _real_host_with_faulting_close(
        RuntimeError("real-host-server-close-probe")
    )
    factory = QuestPairedPoseCaptureFactory(
        _launch_config(),
        host_builder=lambda *_args, **_kwargs: host,
    )

    with pytest.raises(
        QuestPairedPoseCaptureStartupError,
        match="cleanup remains unproven",
    ):
        factory(robot_pose_journal=source)

    assert faulting_server.close_calls == 1
    assert factory.cleanup_pending is True
    with pytest.raises(QuestPairedPoseCaptureCleanupError, match="server-close-probe"):
        factory.close()
    assert faulting_server.close_calls == 1


def test_factory_retains_real_host_after_close_base_exception() -> None:
    _, _, _, _, source = _source_bundle()
    faulting_server, host = _real_host_with_faulting_close(
        KeyboardInterrupt("real-host-server-close-interrupt")
    )
    factory = QuestPairedPoseCaptureFactory(
        _launch_config(),
        host_builder=lambda *_args, **_kwargs: host,
    )

    with pytest.raises(
        QuestPairedPoseCaptureStartupError,
        match="cleanup remains unproven",
    ):
        factory(robot_pose_journal=source)

    assert faulting_server.close_calls == 1
    assert factory.cleanup_pending is True
    with pytest.raises(
        QuestPairedPoseCaptureCleanupError,
        match="server-close-interrupt",
    ):
        factory.close()
    assert faulting_server.close_calls == 1


def test_factory_invalid_host_return_cannot_leave_close_waiting() -> None:
    _, _, _, _, source = _source_bundle()
    factory = QuestPairedPoseCaptureFactory(
        _launch_config(),
        host_builder=lambda *_args, **_kwargs: None,
    )

    with pytest.raises(QuestPairedPoseCaptureStartupError, match="startup failed"):
        factory(robot_pose_journal=source)

    factory.close()


def test_factory_timeout_retains_live_host_owner_for_close_retry() -> None:
    receiver_clock, clock, _, _, source = _source_bundle()
    runtime = _runtime(receiver_clock, clock)
    fake_host = _FakeOwnedHost(runtime, stop_on_request=False)

    def never_ready(*, ready_callback=None):
        del ready_callback
        fake_host._shutdown.wait(timeout=1.0)

    fake_host.serve_forever = never_ready
    factory = QuestPairedPoseCaptureFactory(
        _launch_config(startup_timeout_s=0.01, host_join_timeout_s=0.01),
        host_builder=lambda *_args, **_kwargs: fake_host,
    )

    with pytest.raises(
        QuestPairedPoseCaptureStartupError,
        match="cleanup remains unproven",
    ):
        factory(robot_pose_journal=source)

    with pytest.raises(RuntimeError, match="startup failed") as waiter_error:
        factory.wait_for_application(timeout_s=0.01)
    assert isinstance(
        waiter_error.value.__cause__,
        QuestPairedPoseCaptureStartupError,
    )
    assert "cleanup remains unproven" in str(waiter_error.value.__cause__)
    assert factory.cleanup_pending is True

    fake_host._shutdown.set()
    factory.close()
    assert factory.cleanup_pending is False
    assert fake_host.close_calls == 1


def test_factory_close_during_blocked_builder_cancels_startup() -> None:
    receiver_clock, clock, _, _, source = _source_bundle()
    runtime = _runtime(receiver_clock, clock)
    fake_host = _FakeOwnedHost(runtime)
    builder_entered = threading.Event()
    release_builder = threading.Event()

    def blocked_builder(*_args, **_kwargs):
        builder_entered.set()
        assert release_builder.wait(timeout=0.5)
        return fake_host

    factory = QuestPairedPoseCaptureFactory(
        _launch_config(),
        host_builder=blocked_builder,
    )
    startup_errors: list[BaseException] = []
    close_errors: list[BaseException] = []

    def start_factory() -> None:
        try:
            factory(robot_pose_journal=source)
        except BaseException as exc:  # noqa: BLE001 - asserted below.
            startup_errors.append(exc)

    def close_factory() -> None:
        try:
            factory.close()
        except BaseException as exc:  # noqa: BLE001 - asserted below.
            close_errors.append(exc)

    startup_thread = threading.Thread(target=start_factory)
    close_thread = threading.Thread(target=close_factory)
    startup_thread.start()
    assert builder_entered.wait(timeout=0.5)
    close_thread.start()
    assert close_thread.is_alive()
    release_builder.set()
    startup_thread.join(timeout=0.5)
    close_thread.join(timeout=0.5)

    assert not startup_thread.is_alive()
    assert not close_thread.is_alive()
    assert close_errors == []
    assert len(startup_errors) == 1
    assert isinstance(startup_errors[0], QuestPairedPoseCaptureStartupError)
    assert fake_host.close_calls == 1


def test_run_policy_factory_is_default_off_narrow_and_ordered() -> None:
    parameter = inspect.signature(run_policy.main).parameters[
        "quest_paired_pose_capture_factory"
    ]
    source = inspect.getsource(run_policy.main)
    control_loop_snapshot = "arm_states = arm_reader.snapshot(loop_started_at)"

    assert parameter.default is None
    assert source.count("quest_paired_pose_capture.poll()") == 1
    assert source.index("resource_lifecycle.start(") < source.index(
        "assemble_rm65_pose_observation("
    )
    assert source.index("assemble_rm65_pose_observation(") < source.index(
        "bind_passive_robot_pose_journal_source("
    )
    assert source.index("bind_passive_robot_pose_journal_source(") < source.index(
        control_loop_snapshot
    )
    assert source.index(control_loop_snapshot) < source.index(
        "quest_paired_pose_capture.poll()"
    )
    assert source.index("quest_paired_pose_capture.poll()") < source.index(
        "ik_action_gate.update("
    )
    assert "quest_paired_pose_capture.close()" not in source
    assert source.rindex("quest_paired_pose_capture_factory.close()") < source.index(
        "invalidate_pose_measurements("
    )
    assert source.index("invalidate_pose_measurements(") < source.index(
        "resource_lifecycle.close("
    )
    assert "robot_pose_journal=robot_pose_journal" in source


def test_run_policy_rejects_factory_without_pose_observation_before_backends() -> None:
    factory = QuestPairedPoseCaptureFactory(
        _launch_config(),
        host_builder=lambda *_args, **_kwargs: None,
    )
    with (
        mock.patch.object(run_policy, "build_arm_backends") as build_backends,
        pytest.raises(ValueError, match="requires --rm65-pose-observation"),
    ):
        run_policy.main([], quest_paired_pose_capture_factory=factory)

    build_backends.assert_not_called()


def test_run_policy_rejects_non_factory_before_backends() -> None:
    with (
        mock.patch.object(run_policy, "build_arm_backends") as build_backends,
        pytest.raises(TypeError, match="QuestPairedPoseCaptureFactory"),
    ):
        run_policy.main(
            ["--rm65-pose-observation"],
            quest_paired_pose_capture_factory=lambda **_: None,  # type: ignore[arg-type]
        )

    build_backends.assert_not_called()
