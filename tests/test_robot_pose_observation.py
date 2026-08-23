from __future__ import annotations

import numpy as np
import pytest

from robot_motion.action.execution_observation import (
    ExecutionObservationCapabilitySpec,
    ExecutionObservationPlan,
    bind_execution_observation_plan,
)
from robot_motion.runtime.identity import ClockDomainDescriptor
from robot_motion.runtime.pose_observation import (
    PassivePoseMeasurementJournal,
    PoseCoordinateBinding,
    PoseMeasurementAcceptedEvent,
    PoseMeasurementInvalidationEvent,
    PoseMeasurementJournalGapError,
)
from teleop_runtime.deploy.arm_state_reader import (
    ACTUAL_TCP_POSE_COORDINATE_BINDING_HASH_STATE_KEY,
    HARDWARE_POSE_COORDINATE_BINDING_STATE_KEY,
    ArmPoseObservationProjection,
    ArmPoseStateReadResult,
    ArmStateReader,
    project_arm_tcp_pose_measurement,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]

_TOOL_HASH = "sha256:" + "1" * 64


def _receiver_clock() -> ClockDomainDescriptor:
    return ClockDomainDescriptor(
        clock_domain_id="host-boot-1.monotonic",
        runtime_session_id="quest-calibration-runtime-1",
        boot_id="host-boot-1",
        reset_epoch=0,
        resolution_sec=1e-9,
    )


def _projection(
    *,
    source_epoch_id: str = "rm65-right-epoch-7",
    tool_profile_hash: str = _TOOL_HASH,
    stream_id: str = "robot_state.right_arm.tool_tcp_pose",
) -> ArmPoseObservationProjection:
    plan = ExecutionObservationPlan(
        plan_id="quest-calibration-passive-pose",
        measurement_mode="full_frames",
        component_ids=("right_arm",),
        required_representation_by_component={
            "right_arm": "cartesian_pose",
        },
    )
    capability = ExecutionObservationCapabilitySpec(
        capability_id="real-rm65-passive-pose",
        target_handler_id="real-rm65-right",
        clock_domain_id="local_wall_time",
        supported_reference_modes=("off",),
        supported_measurement_modes=("off", "full_frames"),
        representation_by_component={"right_arm": "cartesian_pose"},
        schema_hash_by_component={"right_arm": "pose-schema-xyzquat-v1"},
        measurement_stage_id="real_robot.hardware_qpos_fk_pose",
    )
    observation_binding = bind_execution_observation_plan(
        plan,
        capability,
        bound_at=90.0,
    )
    coordinate = PoseCoordinateBinding(
        binding_id=f"right-tool-pose:{source_epoch_id}",
        source_instance_id="rm65-right-runtime",
        source_epoch_id=source_epoch_id,
        source_binding_hash="rm65-source-binding-hash",
        control_domain_id="real-robot-right",
        component_id="right_arm",
        stream_id=stream_id,
        reference_frame_id="right_arm_base",
        reference_frame_spec_hash="right-base-frame-spec",
        moving_frame_id="tool_tcp",
        tool_profile_hash=tool_profile_hash,
        pose_schema_hash="pose-schema-xyzquat-v1",
        fk_contract_id="rm65-hardware-qpos-fk-v1",
        fk_contract_hash="rm65-hardware-qpos-fk-hash",
        measurement_clock_domain_id="local_wall_time",
        receipt_clock_domain_id="host-boot-1.monotonic",
        observation_binding_id=observation_binding.binding_id,
        observation_binding_hash=observation_binding.semantic_content_hash,
        observation_plan_hash=observation_binding.plan_hash,
        source_kind="real_robot_hardware_qpos_fk",
    )
    return ArmPoseObservationProjection(
        coordinate_binding=coordinate,
        capability=capability,
        observation_binding=observation_binding,
    )


def _published_state(
    *,
    sequence: int = 3,
    qpos_x: float = 0.2,
    measured_at: float = 99.0,
    available_at: float = 100.0,
    projection: ArmPoseObservationProjection | None = None,
) -> dict[str, object]:
    projection = projection or _projection()
    coordinate = projection.coordinate_binding
    pose = np.eye(4, dtype=float)
    pose[0, 3] = qpos_x
    return {
        "hardware_qpos": np.asarray(
            [qpos_x, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=float,
        ),
        "hardware_measurement_timestamp": measured_at,
        "hardware_measurement_sequence": sequence,
        "hardware_measurement_valid": True,
        "hardware_read_success": True,
        "actual_tcp_pose": pose,
        "actual_tcp_pose_source": "hardware_qpos_fk",
        "actual_tcp_pose_timestamp": measured_at,
        HARDWARE_POSE_COORDINATE_BINDING_STATE_KEY: coordinate,
        ACTUAL_TCP_POSE_COORDINATE_BINDING_HASH_STATE_KEY: (
            coordinate.semantic_content_hash
        ),
        "source_timestamp": measured_at,
        "available_timestamp": available_at,
        "source_clock": "local_wall_time",
        "available_clock": "local_wall_time",
    }


def _publication(
    *,
    projection: ArmPoseObservationProjection | None = None,
    sequence: int = 3,
    qpos_x: float = 0.2,
    measured_at: float = 99.0,
    available_at: float = 100.0,
    received_monotonic_ns: int = 5_000_000_000,
):
    projection = projection or _projection()
    return project_arm_tcp_pose_measurement(
        _published_state(
            sequence=sequence,
            qpos_x=qpos_x,
            measured_at=measured_at,
            available_at=available_at,
            projection=projection,
        ),
        projection=projection,
        received_monotonic_ns=received_monotonic_ns,
        receiver_clock=_receiver_clock(),
    )


def test_real_pose_projection_preserves_hardware_and_coordinate_lineage() -> None:
    projection = _projection()
    publication = project_arm_tcp_pose_measurement(
        _published_state(projection=projection),
        projection=projection,
        received_monotonic_ns=5_000_000_000,
        receiver_clock=_receiver_clock(),
    )

    measurement = publication.measurement
    assert measurement.control_domain_id == "real-robot-right"
    assert measurement.measured_at == 99.0
    assert measurement.available_at == 100.0
    assert measurement.value_by_component["right_arm"][0] == pytest.approx(0.2)
    assert publication.source_sequence == 3
    assert publication.sample_ref.sample_id == measurement.measurement_id
    assert publication.sample_ref.payload_object_id == measurement.semantic_content_hash
    assert publication.sample_ref.available_at == 5.0
    assert publication.sample_ref.available_clock_domain_id == ("host-boot-1.monotonic")
    assert {fact.stage for fact in measurement.parent_facts} == {
        "real_robot.source_binding",
        "real_robot.hardware_qpos",
        "real_robot.fk_contract",
        "real_robot.pose_coordinate_binding",
    }


def test_repeated_native_sample_keeps_measurement_identity_and_is_not_republished() -> (
    None
):
    projection = _projection()
    first = _publication(projection=projection)
    repeated = _publication(projection=projection)
    journal = PassivePoseMeasurementJournal(receiver_clock=_receiver_clock())

    assert first.measurement.measurement_id == repeated.measurement.measurement_id
    assert first.measurement.semantic_content_hash == (
        repeated.measurement.semantic_content_hash
    )
    first_event = journal.publish(first)
    assert isinstance(first_event, PoseMeasurementAcceptedEvent)
    assert journal.publish(repeated) is None
    assert journal.events_after(0) == (first_event,)


def test_source_sequence_reuse_with_changed_hardware_content_fails_closed() -> None:
    projection = _projection()
    journal = PassivePoseMeasurementJournal(receiver_clock=_receiver_clock())
    journal.publish(_publication(projection=projection, qpos_x=0.2))
    changed = _publication(
        projection=projection,
        qpos_x=0.25,
    )

    with pytest.raises(ValueError, match="reused with different content"):
        journal.publish(changed)


def test_journal_rejects_receiver_monotonic_time_regression() -> None:
    projection = _projection()
    journal = PassivePoseMeasurementJournal(receiver_clock=_receiver_clock())
    first = _publication(projection=projection, received_monotonic_ns=5_000_000_000)
    journal.publish(first)

    with pytest.raises(ValueError, match="timestamp regressed"):
        journal.publish(
            _publication(
                projection=projection,
                sequence=4,
                measured_at=99.1,
                available_at=100.1,
                received_monotonic_ns=4_900_000_000,
            )
        )
    with pytest.raises(ValueError, match="timestamp regressed"):
        journal.invalidate(
            coordinate_binding=projection.coordinate_binding,
            reason="late_failure",
            received_monotonic_ns=4_900_000_000,
            next_coordinate_binding=projection.coordinate_binding,
        )

    assert journal.latest(stream_id=projection.coordinate_binding.stream_id) is first
    assert journal.snapshot_status()["last_received_monotonic_ns"] == 5_000_000_000


def test_journal_exact_duplicates_do_not_regress_multi_stream_event_time() -> None:
    first_projection = _projection()
    second_projection = _projection(stream_id="robot_state.left_arm.tool_tcp_pose")
    first = _publication(
        projection=first_projection,
        received_monotonic_ns=5_000_000_000,
    )
    second = _publication(
        projection=second_projection,
        received_monotonic_ns=5_100_000_000,
    )
    journal = PassivePoseMeasurementJournal(receiver_clock=_receiver_clock())
    journal.publish(first)
    journal.publish(second)

    assert journal.publish(first) is None
    assert journal.publish(second) is None
    invalidated = journal.invalidate(
        coordinate_binding=first_projection.coordinate_binding,
        reason="temporary_read_failure",
        received_monotonic_ns=5_200_000_000,
        next_coordinate_binding=first_projection.coordinate_binding,
    )
    assert isinstance(invalidated, PoseMeasurementInvalidationEvent)
    assert (
        journal.invalidate(
            coordinate_binding=first_projection.coordinate_binding,
            reason="duplicate_failure",
            received_monotonic_ns=4_900_000_000,
            next_coordinate_binding=first_projection.coordinate_binding,
        )
        is None
    )
    assert journal.atomic_cut().event_sequence == 3
    assert journal.latest(stream_id=second_projection.coordinate_binding.stream_id) is (
        second
    )


def test_receiver_clock_requires_exact_monotonic_boolean() -> None:
    descriptor = ClockDomainDescriptor(
        clock_domain_id="host-boot-1.monotonic",
        runtime_session_id="quest-calibration-runtime-1",
        boot_id="host-boot-1",
        reset_epoch=0,
        monotonic="yes",  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="exactly True"):
        PassivePoseMeasurementJournal(receiver_clock=descriptor)


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ({"hardware_measurement_sequence": 0}, "positive integer"),
        ({"hardware_read_success": False}, "valid hardware-qpos"),
        ({"hardware_read_success": None}, "valid hardware-qpos"),
        ({"source_clock": "device-clock"}, "clocks do not match"),
        ({"available_timestamp": 98.0}, "timing is invalid"),
    ),
)
def test_pose_projection_rejects_untrustworthy_source_evidence(
    change: dict[str, object],
    message: str,
) -> None:
    state = _published_state()
    state.update(change)
    with pytest.raises(ValueError, match=message):
        project_arm_tcp_pose_measurement(
            state,
            projection=_projection(),
            received_monotonic_ns=5_000_000_000,
            receiver_clock=_receiver_clock(),
        )


def test_pose_projection_rejects_non_rigid_fk_output() -> None:
    state = _published_state()
    pose = np.eye(4, dtype=float)
    pose[0, 0] = 2.0
    state["actual_tcp_pose"] = pose

    with pytest.raises(ValueError, match="rigid 4x4"):
        project_arm_tcp_pose_measurement(
            state,
            projection=_projection(),
            received_monotonic_ns=5_000_000_000,
            receiver_clock=_receiver_clock(),
        )


def test_pose_projection_requires_source_owned_coordinate_attestation() -> None:
    projection = _projection()
    state = _published_state(projection=projection)
    state.pop(HARDWARE_POSE_COORDINATE_BINDING_STATE_KEY)

    with pytest.raises(TypeError, match="typed pose-coordinate attestation"):
        project_arm_tcp_pose_measurement(
            state,
            projection=projection,
            received_monotonic_ns=5_000_000_000,
            receiver_clock=_receiver_clock(),
        )


def test_pose_projection_rejects_unbound_receiver_clock() -> None:
    projection = _projection()
    other_clock = ClockDomainDescriptor(
        clock_domain_id="other-host.monotonic",
        runtime_session_id="other-runtime",
        boot_id="other-boot",
        reset_epoch=0,
    )

    with pytest.raises(ValueError, match="receiver clock"):
        project_arm_tcp_pose_measurement(
            _published_state(projection=projection),
            projection=projection,
            received_monotonic_ns=5_000_000_000,
            receiver_clock=other_clock,
        )


def test_journal_requires_explicit_epoch_transition_and_orders_invalidation() -> None:
    first_projection = _projection(source_epoch_id="epoch-1")
    next_projection = _projection(source_epoch_id="epoch-2")
    first = _publication(
        projection=first_projection,
        sequence=9,
    )
    next_epoch = _publication(
        projection=next_projection,
        sequence=1,
        received_monotonic_ns=5_100_000_000,
    )
    journal = PassivePoseMeasurementJournal(receiver_clock=_receiver_clock())

    accepted = journal.publish(first)
    with pytest.raises(ValueError, match="without explicit invalidation"):
        journal.publish(next_epoch)
    invalidated = journal.invalidate(
        coordinate_binding=first_projection.coordinate_binding,
        reason="source_epoch_changed",
        received_monotonic_ns=5_050_000_000,
        next_coordinate_binding=next_projection.coordinate_binding,
    )
    next_accepted = journal.publish(next_epoch)

    assert isinstance(accepted, PoseMeasurementAcceptedEvent)
    assert isinstance(invalidated, PoseMeasurementInvalidationEvent)
    assert isinstance(next_accepted, PoseMeasurementAcceptedEvent)
    assert (
        journal.latest(stream_id=next_projection.coordinate_binding.stream_id)
        == next_epoch
    )
    assert [event.event_sequence for event in journal.events_after(0)] == [1, 2, 3]


def test_journal_recoverable_invalidation_retains_sequence_fence_and_gap_fails() -> (
    None
):
    projection = _projection()
    journal = PassivePoseMeasurementJournal(
        receiver_clock=_receiver_clock(),
        max_events=2,
    )
    journal.publish(_publication(projection=projection, sequence=3))
    invalidated = journal.invalidate(
        coordinate_binding=projection.coordinate_binding,
        reason="temporary_read_failure",
        received_monotonic_ns=5_100_000_000,
        next_coordinate_binding=projection.coordinate_binding,
    )
    assert isinstance(invalidated, PoseMeasurementInvalidationEvent)
    assert journal.latest(stream_id=projection.coordinate_binding.stream_id) is None
    with pytest.raises(ValueError, match="regressed"):
        journal.publish(_publication(projection=projection, sequence=2))

    journal.publish(
        _publication(
            projection=projection,
            sequence=4,
            measured_at=99.1,
            available_at=100.1,
            received_monotonic_ns=5_200_000_000,
        )
    )
    with pytest.raises(PoseMeasurementJournalGapError):
        journal.events_after(0)


def test_journal_can_escalate_recoverable_invalidation_to_terminal() -> None:
    projection = _projection()
    journal = PassivePoseMeasurementJournal(receiver_clock=_receiver_clock())
    journal.publish(_publication(projection=projection, sequence=3))
    journal.invalidate(
        coordinate_binding=projection.coordinate_binding,
        reason="temporary_read_failure",
        received_monotonic_ns=5_100_000_000,
        next_coordinate_binding=projection.coordinate_binding,
    )

    terminal = journal.invalidate(
        coordinate_binding=projection.coordinate_binding,
        reason="reader_closed",
        received_monotonic_ns=5_200_000_000,
        next_coordinate_binding=None,
    )

    assert isinstance(terminal, PoseMeasurementInvalidationEvent)
    assert terminal.event_sequence == 3
    with pytest.raises(RuntimeError, match="explicit next binding"):
        journal.publish(
            _publication(
                projection=projection,
                sequence=4,
                measured_at=99.1,
                available_at=100.1,
                received_monotonic_ns=5_300_000_000,
            )
        )


class _Backend:
    def __init__(self, projection: ArmPoseObservationProjection) -> None:
        self.projection = projection
        self.refresh_calls = 0
        self.measurement_sequence = 4
        self.measurement_timestamp = 99.0
        self.hardware_read_success = True
        self.qpos_x = 0.3
        self.coordinate_binding = projection.coordinate_binding
        self.ready = True
        self.ready_error: Exception | None = None
        self.refresh_error: Exception | None = None
        self.get_state_error: Exception | None = None
        self.get_state_calls = 0

    def is_ready(self) -> bool:
        if self.ready_error is not None:
            raise self.ready_error
        return self.ready

    def refresh_state_from_hardware(self) -> None:
        self.refresh_calls += 1
        if self.refresh_error is not None:
            raise self.refresh_error

    def get_state(self) -> dict[str, object]:
        self.get_state_calls += 1
        if self.get_state_error is not None:
            raise self.get_state_error
        qpos = np.asarray([self.qpos_x, 0.0, 0.0, 0.0, 0.0, 0.0])
        return {
            "hardware_qpos": qpos,
            "hardware_measurement_timestamp": self.measurement_timestamp,
            "hardware_measurement_sequence": self.measurement_sequence,
            "hardware_measurement_valid": True,
            "hardware_read_success": self.hardware_read_success,
            "hardware_measurement_clock_domain_id": "local_wall_time",
            "hardware_availability_clock_domain_id": "local_wall_time",
            HARDWARE_POSE_COORDINATE_BINDING_STATE_KEY: self.coordinate_binding,
        }

    def compute_tcp_pose(self, *, qpos=None) -> np.ndarray:
        pose = np.eye(4, dtype=float)
        pose[0, 3] = float(np.asarray(qpos, dtype=float).reshape(6)[0])
        return pose


def test_arm_state_reader_uses_one_attested_state_read_and_rejects_spoofed_key() -> (
    None
):
    projection = _projection()
    backend = _Backend(projection)
    backend.coordinate_binding = _projection(
        tool_profile_hash="sha256:" + "2" * 64
    ).coordinate_binding
    state_read_calls = 0

    def read_attested_state(candidate: object) -> ArmPoseStateReadResult:
        nonlocal state_read_calls
        state_read_calls += 1
        assert candidate is backend
        return ArmPoseStateReadResult(
            state=backend.get_state(),
            coordinate_binding=projection.coordinate_binding,
        )

    reader = ArmStateReader(
        {"right": backend},
        refresh_hz=30.0,
        clock=lambda: 100.0,
        pose_projection_by_side={"right": projection},
        pose_state_read_by_side={"right": read_attested_state},
        receipt_clock=_receiver_clock(),
        receipt_monotonic_ns_clock=lambda: 5_000_000_000,
    )

    state = reader.snapshot(100.0)["right"]

    assert backend.refresh_calls == 1
    assert backend.get_state_calls == 1
    assert state_read_calls == 1
    assert state[HARDWARE_POSE_COORDINATE_BINDING_STATE_KEY] == (
        projection.coordinate_binding
    )
    assert reader.latest_pose_measurement(side="right") is not None


def test_arm_state_reader_keeps_legacy_state_when_attested_binding_is_unavailable() -> (
    None
):
    projection = _projection()
    backend = _Backend(projection)

    def read_drifted_state(candidate: object) -> ArmPoseStateReadResult:
        assert candidate is backend
        return ArmPoseStateReadResult(
            state=backend.get_state(),
            coordinate_binding=None,
            error="source_epoch_drift",
        )

    reader = ArmStateReader(
        {"right": backend},
        refresh_hz=0.0,
        clock=lambda: 100.0,
        pose_projection_by_side={"right": projection},
        pose_state_read_by_side={"right": read_drifted_state},
        receipt_clock=_receiver_clock(),
        receipt_monotonic_ns_clock=lambda: 5_000_000_000,
    )

    state = reader.snapshot(100.0)["right"]

    assert state["causal_snapshot_missing"] is False
    assert HARDWARE_POSE_COORDINATE_BINDING_STATE_KEY not in state
    assert reader.latest_pose_measurement(side="right") is None
    assert (
        "source_epoch_drift"
        in (reader.pose_observation_status()["error_by_side"]["right"])
    )


def test_arm_state_reader_rejects_mismatched_separate_binding() -> None:
    projection = _projection()
    mismatched_projection = _projection(tool_profile_hash="sha256:" + "2" * 64)
    backend = _Backend(projection)

    def read_mismatched_state(candidate: object) -> ArmPoseStateReadResult:
        assert candidate is backend
        return ArmPoseStateReadResult(
            state=backend.get_state(),
            coordinate_binding=mismatched_projection.coordinate_binding,
        )

    reader = ArmStateReader(
        {"right": backend},
        refresh_hz=0.0,
        clock=lambda: 100.0,
        pose_projection_by_side={"right": projection},
        pose_state_read_by_side={"right": read_mismatched_state},
        receipt_clock=_receiver_clock(),
        receipt_monotonic_ns_clock=lambda: 5_000_000_000,
    )

    state = reader.snapshot(100.0)["right"]

    assert state["causal_snapshot_missing"] is False
    assert HARDWARE_POSE_COORDINATE_BINDING_STATE_KEY not in state
    assert reader.latest_pose_measurement(side="right") is None
    assert (
        "differs from the configured projection"
        in reader.pose_observation_status()["error_by_side"]["right"]
    )


def test_arm_state_reader_rejects_state_read_without_matching_projection() -> None:
    projection = _projection()
    backend = _Backend(projection)

    with pytest.raises(ValueError, match="matching pose projections"):
        ArmStateReader(
            {"right": backend},
            refresh_hz=0.0,
            pose_state_read_by_side={
                "right": lambda candidate: ArmPoseStateReadResult(
                    state=backend.get_state(),
                    coordinate_binding=projection.coordinate_binding,
                )
            },
        )


def test_arm_state_reader_passive_pose_port_does_not_refresh_on_read() -> None:
    projection = _projection()
    backend = _Backend(projection)
    reader = ArmStateReader(
        {"right": backend},
        refresh_hz=30.0,
        clock=lambda: 100.0,
        pose_projection_by_side={"right": projection},
        receipt_clock=_receiver_clock(),
        receipt_monotonic_ns_clock=lambda: 5_000_000_000,
    )

    reader.snapshot(100.0)
    cut = reader.pose_measurement_journal_cut()
    assert cut is not None
    assert cut.event_sequence == 1
    assert backend.refresh_calls == 1

    latest = reader.latest_pose_measurement(side="right")
    events = reader.pose_measurement_events_after(0)
    status = reader.pose_observation_status()
    assert latest is not None
    assert len(events) == 1
    assert events[0].publication is latest
    assert status["error_by_side"]["right"] is None
    assert backend.refresh_calls == 1

    reader.snapshot(100.01)
    assert reader.pose_measurement_journal_cut().event_sequence == 1
    assert backend.refresh_calls == 1


def test_arm_state_reader_keeps_legacy_snapshot_when_optional_projection_fails() -> (
    None
):
    projection = _projection()
    backend = _Backend(projection)
    backend.measurement_sequence = 0
    reader = ArmStateReader(
        {"right": backend},
        refresh_hz=30.0,
        clock=lambda: 100.0,
        pose_projection_by_side={"right": projection},
        receipt_clock=_receiver_clock(),
        receipt_monotonic_ns_clock=lambda: 5_000_000_000,
    )

    state = reader.snapshot(100.0)["right"]

    assert state["causal_snapshot_missing"] is False
    assert state["actual_tcp_pose"] is None
    assert reader.latest_pose_measurement(side="right") is None
    assert (
        "valid hardware-qpos"
        in reader.pose_observation_status()["error_by_side"]["right"]
    )


def test_reader_reuses_first_receipt_for_repeated_native_sequence() -> None:
    projection = _projection()
    backend = _Backend(projection)
    wall_times = iter((100.0, 100.2))
    receipt_times = iter((5_000_000_000, 5_200_000_000))
    reader = ArmStateReader(
        {"right": backend},
        refresh_hz=0.0,
        clock=lambda: next(wall_times),
        pose_projection_by_side={"right": projection},
        receipt_clock=_receiver_clock(),
        receipt_monotonic_ns_clock=lambda: next(receipt_times),
    )

    reader.snapshot(101.0)
    first = reader.latest_pose_measurement(side="right")
    reader.snapshot(102.0)
    repeated = reader.latest_pose_measurement(side="right")

    assert repeated is first
    assert repeated.received_monotonic_ns == 5_000_000_000
    assert repeated.measurement.available_at == 100.0
    assert reader.pose_measurement_journal_cut().event_sequence == 1
    assert reader.pose_observation_status()["error_by_side"]["right"] is None


def test_reader_invalidates_latest_when_source_reuses_sequence_with_new_content() -> (
    None
):
    projection = _projection()
    backend = _Backend(projection)
    wall_times = iter((100.0, 100.2))
    receipt_times = iter(
        (
            5_000_000_000,
            5_200_000_000,
            5_300_000_000,
        )
    )
    reader = ArmStateReader(
        {"right": backend},
        refresh_hz=0.0,
        clock=lambda: next(wall_times),
        pose_projection_by_side={"right": projection},
        receipt_clock=_receiver_clock(),
        receipt_monotonic_ns_clock=lambda: next(receipt_times),
    )
    reader.snapshot(101.0)
    backend.qpos_x = 0.4

    reader.snapshot(102.0)

    assert reader.latest_pose_measurement(side="right") is None
    events = reader.pose_measurement_events_after(0)
    assert [type(event) for event in events] == [
        PoseMeasurementAcceptedEvent,
        PoseMeasurementInvalidationEvent,
    ]
    assert (
        "reused with different content"
        in reader.pose_observation_status()["error_by_side"]["right"]
    )


def test_reader_invalidates_latest_on_hardware_read_failure_and_can_recover() -> None:
    projection = _projection()
    backend = _Backend(projection)
    wall_times = iter((100.0, 100.1, 100.2))
    receipt_times = iter(
        (
            5_000_000_000,
            5_100_000_000,
            5_110_000_000,
            5_200_000_000,
        )
    )
    reader = ArmStateReader(
        {"right": backend},
        refresh_hz=0.0,
        clock=lambda: next(wall_times),
        pose_projection_by_side={"right": projection},
        receipt_clock=_receiver_clock(),
        receipt_monotonic_ns_clock=lambda: next(receipt_times),
    )
    reader.snapshot(101.0)
    backend.hardware_read_success = False
    backend.measurement_sequence = 5
    backend.measurement_timestamp = 99.1
    reader.snapshot(102.0)
    assert reader.latest_pose_measurement(side="right") is None

    backend.hardware_read_success = True
    reader.snapshot(103.0)

    recovered = reader.latest_pose_measurement(side="right")
    assert recovered is not None
    assert recovered.source_sequence == 5
    assert [
        event.event_sequence for event in reader.pose_measurement_events_after(0)
    ] == [
        1,
        2,
        3,
    ]


def test_reader_invalidates_cached_latest_when_refresh_fails() -> None:
    projection = _projection()
    backend = _Backend(projection)
    receipt_times = iter((5_000_000_000, 5_100_000_000))
    reader = ArmStateReader(
        {"right": backend},
        refresh_hz=30.0,
        clock=lambda: 100.0,
        pose_projection_by_side={"right": projection},
        receipt_clock=_receiver_clock(),
        receipt_monotonic_ns_clock=lambda: next(receipt_times),
    )
    reader.snapshot(100.0)
    backend.refresh_error = RuntimeError("transport timeout")

    state = reader.snapshot(101.0)["right"]

    assert state["state_error"].startswith("refresh_failed:RuntimeError")
    assert reader.latest_pose_measurement(side="right") is None
    assert isinstance(
        reader.pose_measurement_events_after(0)[-1],
        PoseMeasurementInvalidationEvent,
    )
    assert (
        "refresh_failed" in reader.pose_observation_status()["error_by_side"]["right"]
    )


def test_reader_invalidates_cached_latest_before_get_state_failure_escapes() -> None:
    projection = _projection()
    backend = _Backend(projection)
    receipt_times = iter((5_000_000_000, 5_100_000_000))
    reader = ArmStateReader(
        {"right": backend},
        refresh_hz=0.0,
        clock=lambda: 100.0,
        pose_projection_by_side={"right": projection},
        receipt_clock=_receiver_clock(),
        receipt_monotonic_ns_clock=lambda: next(receipt_times),
    )
    reader.snapshot(100.0)
    backend.get_state_error = RuntimeError("state unavailable")

    with pytest.raises(RuntimeError, match="state unavailable"):
        reader.snapshot(101.0)

    assert reader.latest_pose_measurement(side="right") is None
    assert isinstance(
        reader.pose_measurement_events_after(0)[-1],
        PoseMeasurementInvalidationEvent,
    )


@pytest.mark.parametrize("readiness_failure", ("not_ready", "check_error"))
def test_reader_invalidates_cached_latest_when_backend_is_not_ready(
    readiness_failure: str,
) -> None:
    projection = _projection()
    backend = _Backend(projection)
    receipt_times = iter((5_000_000_000, 5_100_000_000))
    reader = ArmStateReader(
        {"right": backend},
        refresh_hz=0.0,
        clock=lambda: 100.0,
        pose_projection_by_side={"right": projection},
        receipt_clock=_receiver_clock(),
        receipt_monotonic_ns_clock=lambda: next(receipt_times),
    )
    reader.snapshot(100.0)
    if readiness_failure == "not_ready":
        backend.ready = False
    else:
        backend.ready_error = RuntimeError("readiness unavailable")

    state = reader.snapshot(101.0)["right"]

    assert state["causal_snapshot_missing"] is False
    assert reader.latest_pose_measurement(side="right") is None
    assert isinstance(
        reader.pose_measurement_events_after(0)[-1],
        PoseMeasurementInvalidationEvent,
    )
    assert reader.pose_observation_status()["error_by_side"]["right"] is not None


def test_reader_invalidates_latest_when_tool_coordinate_binding_drifts() -> None:
    projection = _projection()
    backend = _Backend(projection)
    wall_times = iter((100.0, 100.1))
    receipt_times = iter((5_000_000_000, 5_100_000_000, 5_200_000_000))
    reader = ArmStateReader(
        {"right": backend},
        refresh_hz=0.0,
        clock=lambda: next(wall_times),
        pose_projection_by_side={"right": projection},
        receipt_clock=_receiver_clock(),
        receipt_monotonic_ns_clock=lambda: next(receipt_times),
    )
    reader.snapshot(101.0)
    backend.coordinate_binding = _projection(
        tool_profile_hash="sha256:" + "2" * 64
    ).coordinate_binding
    backend.measurement_sequence = 5
    backend.measurement_timestamp = 99.1

    reader.snapshot(102.0)

    assert reader.latest_pose_measurement(side="right") is None
    assert (
        "attestation differs"
        in reader.pose_observation_status()["error_by_side"]["right"]
    )
    assert isinstance(
        reader.pose_measurement_events_after(0)[-1],
        PoseMeasurementInvalidationEvent,
    )


def test_reader_terminal_invalidation_prevents_same_binding_recovery() -> None:
    projection = _projection()
    backend = _Backend(projection)
    wall_times = iter((100.0, 100.1))
    receipt_times = iter((5_000_000_000, 5_100_000_000, 5_200_000_000, 5_300_000_000))
    reader = ArmStateReader(
        {"right": backend},
        refresh_hz=0.0,
        clock=lambda: next(wall_times),
        pose_projection_by_side={"right": projection},
        receipt_clock=_receiver_clock(),
        receipt_monotonic_ns_clock=lambda: next(receipt_times),
    )
    reader.snapshot(101.0)
    terminal = reader.invalidate_pose_measurements(reason="reader_closed")
    assert len(terminal) == 1
    backend.measurement_sequence = 5
    backend.measurement_timestamp = 99.1

    reader.snapshot(102.0)

    assert reader.latest_pose_measurement(side="right") is None
    assert (
        "explicit next binding"
        in reader.pose_observation_status()["error_by_side"]["right"]
    )
