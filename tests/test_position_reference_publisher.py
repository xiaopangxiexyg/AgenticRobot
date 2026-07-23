from __future__ import annotations

import numpy as np

from robot_motion.position_control import (
    POSITION_TRANSPORT_TARGET_SEQUENCE_KEY,
    POSITION_TRANSPORT_TARGET_SEQUENCES_KEY,
    PositionReferenceAuditEvent,
    position_qref_fingerprint,
)
from teleop_runtime.diagnostics import TeleopRunState
from teleop_runtime.position_control import (
    PositionReferencePublisher,
    PositionReferencePublisherConfig,
)
from teleop_runtime.schemas import EffectorCommand, TeleopCommand, TeleopObservation
from teleop_runtime.session import SessionStepResult


class _Clock:
    def __init__(self, values):
        self._values = iter(values)

    def __call__(self):
        return next(self._values)


def _fk(qpos: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = qpos[:3]
    return pose


def _result(*, dispatched: bool = True) -> SessionStepResult:
    command = TeleopCommand(timestamp=10.0)
    for side, offset in (("left", 0.0), ("right", 1.0)):
        qpos = np.arange(6, dtype=float) + offset
        command.add_command(
            EffectorCommand(
                effector_name=f"{side}_arm",
                side=side,
                command_type="joint_vector",
                value=qpos,
                frame_name="arm_base",
                metadata={"backend_final_target_qpos": qpos.copy()},
            )
        )
    return SessionStepResult(
        timestamp=10.1,
        state=TeleopRunState.RUNNING,
        control_hz=45.0,
        observation=TeleopObservation(timestamp=10.0),
        command=command,
        command_dispatched=dispatched,
    )


def test_publisher_builds_atomic_fk_reference_and_causal_lookup() -> None:
    publisher = PositionReferencePublisher(
        config=PositionReferencePublisherConfig(
            active_sides=("left", "right"),
            reference_max_age_sec=0.2,
        ),
        fk_by_side={"left": _fk, "right": _fk},
        wall_clock=_Clock((10.11, 10.12)),
        monotonic_clock=_Clock((2.0, 2.01)),
        epoch=7,
    )

    frame = publisher.publish_step(_result())

    assert frame is not None
    assert frame.epoch == 7
    assert frame.sequence == 0
    assert frame.sides == ("left", "right")
    np.testing.assert_allclose(frame.references["right"].pose_ref[:3, 3], [1.0, 2.0, 3.0])
    assert publisher.latest_at_or_before(10.119) is None
    assert publisher.latest_at_or_before(10.12) is frame
    assert publisher.latest_at_or_before(10.321) is None


def test_publisher_requires_complete_dispatched_group() -> None:
    publisher = PositionReferencePublisher(
        config=PositionReferencePublisherConfig(active_sides=("left", "right")),
        fk_by_side={"left": _fk, "right": _fk},
        epoch=1,
    )
    not_dispatched = _result(dispatched=False)
    assert publisher.publish_step(not_dispatched) is None

    incomplete = _result()
    incomplete.command.commands.pop("right_arm")
    assert publisher.publish_step(incomplete) is None
    assert len(publisher.timeline) == 0


def test_publisher_epoch_reset_prevents_old_reference_reuse() -> None:
    publisher = PositionReferencePublisher(
        config=PositionReferencePublisherConfig(active_sides=("left", "right")),
        fk_by_side={"left": _fk, "right": _fk},
        wall_clock=_Clock((10.1, 10.2)),
        monotonic_clock=_Clock((1.0, 1.1)),
        epoch=3,
    )
    assert publisher.publish_step(_result()) is not None
    assert publisher.latest_at_or_before(10.2) is not None

    assert publisher.reset_epoch() == 4
    assert publisher.latest_at_or_before(10.2) is None


def test_publisher_records_safety_modifications_and_hold() -> None:
    result = _result()
    left = result.command.commands["left_arm"]
    left.metadata.update(
        {
            "q_ik_raw": np.asarray(left.value, dtype=float) + 0.2,
            "collision_avoidance_status": "hold_last_safe",
            "collision_avoidance_fallback_reason": "distance_below_minimum",
            "execution_step_metadata": {"step_limited": True},
        }
    )
    publisher = PositionReferencePublisher(
        config=PositionReferencePublisherConfig(active_sides=("left", "right")),
        fk_by_side={"left": _fk, "right": _fk},
        wall_clock=_Clock((10.1, 10.2)),
        monotonic_clock=_Clock((1.0, 1.1)),
        epoch=9,
    )

    frame = publisher.publish_step(result)

    assert frame is not None and frame.hold
    reasons = frame.references["left"].modification_reasons
    assert "ik_output_modified" in reasons
    assert "collision_avoidance_status:hold_last_safe" in reasons
    assert "execution_joint_step_limited" in reasons


def test_publisher_records_target_and_qref_fk_in_urdf_root_frame() -> None:
    result = _result()
    source_target_pose_base = np.eye(4, dtype=float)
    source_target_pose_base[:3, 3] = [0.5, 1.0, 2.0]
    result.command.commands["left_arm"].metadata.update(
        {
            "source_target_pose_base": source_target_pose_base,
            "target_activity": {"state": "INACTIVE", "hold_weight": 1.0},
            "collision_avoidance_status": "tracking",
        }
    )
    root_from_base = np.eye(4, dtype=float)
    root_from_base[:3, 3] = [1.0, 2.0, 3.0]
    publisher = PositionReferencePublisher(
        config=PositionReferencePublisherConfig(active_sides=("left", "right")),
        fk_by_side={"left": _fk, "right": _fk},
        urdf_root_from_base_by_side={"left": root_from_base, "right": root_from_base},
        wall_clock=_Clock((10.1, 10.2)),
        monotonic_clock=_Clock((1.0, 1.1)),
        epoch=9,
    )

    frame = publisher.publish_step(result)

    assert frame is not None
    metadata = frame.references["left"].frame_metadata
    np.testing.assert_allclose(
        metadata["source_target_pose_urdf_root"][:3, 3],
        [1.5, 3.0, 5.0],
    )
    np.testing.assert_allclose(
        metadata["q_ref_pose_urdf_root"][:3, 3],
        [1.0, 3.0, 5.0],
    )
    np.testing.assert_allclose(
        metadata["source_target_minus_q_ref_xyz_urdf_root_m"],
        [0.5, 0.0, 0.0],
    )
    assert metadata["source_target_minus_q_ref_x_urdf_root_m"] == 0.5
    assert metadata["target_activity"]["state"] == "INACTIVE"


def test_sample_selection_uses_monotonic_freshness_across_wall_clock_rewind() -> None:
    publisher = PositionReferencePublisher(
        config=PositionReferencePublisherConfig(
            active_sides=("left", "right"),
            reference_max_age_sec=0.2,
        ),
        fk_by_side={"left": _fk, "right": _fk},
        wall_clock=_Clock((99.0, 99.1)),
        monotonic_clock=_Clock((5.0, 5.01, 5.05, 5.5)),
        epoch=10,
    )
    assert publisher.publish_step(_result()) is not None

    selected = publisher.reference_for_sample(100.1)

    assert selected is not None
    assert selected["sequence"] == 0
    assert selected["selection_age_clock"] == "monotonic"
    assert np.isclose(selected["selection_age_sec"], 0.04)
    assert publisher.reference_for_sample(100.2) is None


def test_publisher_observer_links_reference_to_backend_transport_targets() -> None:
    result = _result()
    result.command.commands["left_arm"].metadata[
        POSITION_TRANSPORT_TARGET_SEQUENCE_KEY
    ] = 41
    result.command.commands["right_arm"].metadata[
        POSITION_TRANSPORT_TARGET_SEQUENCE_KEY
    ] = 92
    observed: list[PositionReferenceAuditEvent] = []
    publisher = PositionReferencePublisher(
        config=PositionReferencePublisherConfig(active_sides=("left", "right")),
        fk_by_side={"left": _fk, "right": _fk},
        wall_clock=_Clock((10.1, 10.2)),
        monotonic_clock=_Clock((1.0, 1.1)),
        epoch=12,
        audit_observer=observed.append,
    )

    frame = publisher.publish_step(result)

    assert frame is not None
    assert frame.frame_metadata[POSITION_TRANSPORT_TARGET_SEQUENCES_KEY] == {
        "left": 41,
        "right": 92,
    }
    assert result.command.metadata[POSITION_TRANSPORT_TARGET_SEQUENCES_KEY] == {
        "left": 41,
        "right": 92,
    }
    assert len(observed) == 1
    assert observed[0].epoch == 12
    assert observed[0].reference_sequence == 0
    assert observed[0].transport_target_sequences == {"left": 41, "right": 92}


def test_publisher_binds_candidate_certificate_to_qref_sequence_and_transport() -> None:
    result = _result()
    for side, target_sequence in (("left", 41), ("right", 92)):
        command = result.command.commands[f"{side}_arm"]
        qref = np.asarray(command.value, dtype=float)
        command.metadata[POSITION_TRANSPORT_TARGET_SEQUENCE_KEY] = target_sequence
        command.metadata["ik_candidate_selection"] = {
            "candidate_certificate": {
                "sequence": 17,
                "qref_fingerprint_by_side": {
                    side: position_qref_fingerprint(qref),
                },
            }
        }
    publisher = PositionReferencePublisher(
        config=PositionReferencePublisherConfig(active_sides=("left", "right")),
        fk_by_side={"left": _fk, "right": _fk},
        wall_clock=_Clock((10.1, 10.2)),
        monotonic_clock=_Clock((1.0, 1.1)),
        epoch=12,
    )

    frame = publisher.publish_step(result)

    assert frame is not None
    for side, target_sequence in (("left", 41), ("right", 92)):
        reference = frame.references[side]
        assert reference.frame_metadata["candidate_certificate_qref_match"] is True
        binding = result.command.commands[f"{side}_arm"].metadata[
            "position_reference_candidate_binding"
        ]
        assert binding == {
            "epoch": 12,
            "sequence": 0,
            "transport_target_sequence": target_sequence,
            "qref_fingerprint": reference.frame_metadata["qref_fingerprint"],
            "candidate_certificate_sequence": 17,
            "candidate_certificate_qref_match": True,
        }


def test_publisher_audit_observer_failure_is_isolated_from_reference_publish() -> None:
    def fail(_event) -> None:
        raise RuntimeError("audit unavailable")

    publisher = PositionReferencePublisher(
        config=PositionReferencePublisherConfig(active_sides=("left", "right")),
        fk_by_side={"left": _fk, "right": _fk},
        wall_clock=_Clock((10.1, 10.2)),
        monotonic_clock=_Clock((1.0, 1.1)),
        epoch=13,
        audit_observer=fail,
    )

    assert publisher.publish_step(_result()) is not None
    assert publisher.audit_observer_error_count == 1
    assert "audit unavailable" in str(publisher.last_audit_observer_error)
