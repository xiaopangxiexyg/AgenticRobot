from __future__ import annotations

from pathlib import Path

import numpy as np

from control_station.sources import (
    B601_READONLY_POSE_CLOCK_DOMAIN_V1,
    B601ReadOnlyPoseSource,
    B601ReadOnlyPoseSourceSpec,
)
from hardware.rebot_b601_rs import (
    B601_RS_FEEDBACK_FRAME,
    B601HardwareState,
    B601JointState,
)
from teleop_runtime.action.human_observer import HumanPoseSourceContract


class _Backend:
    def __init__(self, states: list[B601HardwareState]) -> None:
        self.states = list(states)
        self.calls: list[dict[str, bool]] = []

    def poll_feedback(
        self,
        *,
        request_feedback: bool,
        require_host_freshness: bool,
    ) -> B601HardwareState:
        self.calls.append(
            {
                "request_feedback": request_feedback,
                "require_host_freshness": require_host_freshness,
            }
        )
        return self.states.pop(0)


class _Model:
    arm_joint_names = (
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    )
    end_effector_frame = "gripper_end"

    def __init__(self, pose: np.ndarray | None = None) -> None:
        self.pose = np.eye(4) if pose is None else np.asarray(pose, dtype=float)
        self.calls: list[np.ndarray] = []

    def forward_kinematics(self, qpos: np.ndarray) -> np.ndarray:
        self.calls.append(np.asarray(qpos, dtype=float).copy())
        return self.pose.copy()


def _state(
    *,
    sequence: int = 4,
    positions: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
    valid: bool = True,
    stale: bool = False,
    host_fresh: bool = True,
    frame_id: str = B601_RS_FEEDBACK_FRAME,
    status_sequences: tuple[int, ...] = (11, 12, 13, 14, 15, 16),
    observed_at_sec: float = 9.9,
    receive_timestamp_sec: float = 10.0,
) -> B601HardwareState:
    joints = tuple(
        B601JointState(
            name=f"joint{index + 1}",
            motor_id=index + 1,
            feedback_id=0xFD,
            arbitration_id=index + 1,
            status_code=0,
            position_rad=positions[index],
            velocity_reported=0.0,
            torque_reported_nm=0.0,
            observed_at_sec=observed_at_sec,
            sample_sequence=sequence,
            valid=valid,
            stale=stale,
            frame_id=frame_id,
            position_source="robstride_operation_status_cache",
            status_sequence=status_sequences[index],
            receive_age_sec=receive_timestamp_sec - observed_at_sec,
            host_receive_freshness_verified=host_fresh,
        )
        for index in range(6)
    )
    return B601HardwareState(
        sequence=sequence,
        receive_timestamp_sec=receive_timestamp_sec,
        joints=joints,
        valid=valid,
        stale=stale,
        freshness_verified=False,
        freshness_semantics=(
            "host_receive_sequence_and_age_verified; "
            "device_source_timestamp_unavailable"
        ),
        fault_reason=None if valid else "synthetic invalid feedback",
        host_receive_freshness_verified=host_fresh,
    )


def _source(
    states: list[B601HardwareState],
    model: _Model | None = None,
) -> tuple[B601ReadOnlyPoseSource, _Backend, _Model]:
    backend = _Backend(states)
    resolved_model = _Model() if model is None else model
    source = B601ReadOnlyPoseSource(backend, resolved_model)
    return source, backend, resolved_model


def test_spec_is_a_typed_single_arm_pose_source_contract() -> None:
    spec = B601ReadOnlyPoseSourceSpec()

    assert isinstance(spec, HumanPoseSourceContract)
    assert spec.reference_frame_by_effector == (("right_arm", "base_link"),)
    assert spec.action_spec_hash_by_effector == (
        ("right_arm", "b601-rs-gripper-end-pose-v1"),
    )


def test_source_module_does_not_bind_the_concrete_hardware_package() -> None:
    source_path = (
        Path(__file__).parents[1] / "control_station" / "sources" / "rebot_b601_pose.py"
    )
    source = source_path.read_text(encoding="utf-8")

    assert "from hardware" not in source
    assert "import hardware" not in source


def test_step_uses_only_host_fresh_cache_and_emits_exact_fk_pose() -> None:
    expected_pose = np.eye(4)
    expected_pose[:3, 3] = (0.3, -0.2, 0.5)
    source, backend, model = _source([_state()], _Model(expected_pose))

    step = source.step()

    assert backend.calls == [
        {"request_feedback": False, "require_host_freshness": True}
    ]
    assert step.emission_valid
    assert step.emission is not None
    assert not step.command_dispatched
    assert not step.authoritative
    assert source.next_emission_sequence == 1
    np.testing.assert_allclose(
        model.calls[0],
        np.asarray((0.1, 0.2, 0.3, 0.4, 0.5, 0.6)),
    )
    command = step.emission.merged_commands[0]
    assert command.effector_name == "right_arm"
    assert command.command_type == "cartesian_pose"
    assert command.frame_name == "base_link"
    np.testing.assert_allclose(command.value, expected_pose)
    sample = step.emission.observation_sample_refs["b601_rs_arm_qpos"]
    assert sample.measured_at == 9.9
    assert sample.available_at == 10.0
    assert sample.measured_clock_domain_id == B601_READONLY_POSE_CLOCK_DOMAIN_V1
    assert sample.available_clock_domain_id == B601_READONLY_POSE_CLOCK_DOMAIN_V1
    assert sample.valid and not sample.stale
    assert sample.content_metadata["upstream_sequence"] == str((11, 12, 13, 14, 15, 16))
    assert (
        step.emission.merged_command_metadata["host_receive_freshness_verified"] is True
    )
    assert (
        step.emission.merged_command_metadata["device_source_timestamp_verified"]
        is False
    )


def test_invalid_snapshot_is_rejected_without_fk_or_sequence_advance() -> None:
    source, _, model = _source([_state(valid=False, stale=True)])

    step = source.step()

    assert not step.emission_valid
    assert "synthetic invalid feedback" in step.rejection_reason
    assert model.calls == []
    assert source.next_emission_sequence == 0


def test_unverified_host_freshness_is_rejected_without_fk() -> None:
    source, _, model = _source([_state(host_fresh=False)])

    step = source.step()

    assert not step.emission_valid
    assert "lacks verified host receive freshness" in step.rejection_reason
    assert model.calls == []


def test_wrong_feedback_frame_is_rejected_without_fk() -> None:
    source, _, model = _source([_state(frame_id="wrong-frame")])

    step = source.step()

    assert not step.emission_valid
    assert "wrong-frame" in step.rejection_reason
    assert model.calls == []


def test_incomplete_joint_order_is_rejected_without_fk() -> None:
    state = _state()
    object.__setattr__(state, "joints", state.joints[:-1])
    source, _, model = _source([state])

    step = source.step()

    assert not step.emission_valid
    assert "joint order is incomplete or incompatible" in step.rejection_reason
    assert model.calls == []


def test_invalid_fk_is_rejected_without_reusing_previous_pose() -> None:
    bad_pose = np.eye(4)
    bad_pose[0, 0] = 2.0
    source, _, _ = _source([_state()], _Model(bad_pose))

    step = source.step()

    assert not step.emission_valid
    assert "rotation is not orthonormal" in step.rejection_reason
    assert source.next_emission_sequence == 0


def test_observation_identity_is_reused_only_for_same_source_sample() -> None:
    first_state = _state()
    held_state = _state(sequence=5, receive_timestamp_sec=10.1)
    changed = _state(
        sequence=6,
        status_sequences=(21, 22, 23, 24, 25, 26),
        observed_at_sec=10.15,
        receive_timestamp_sec=10.2,
    )
    source, _, _ = _source([first_state, held_state, changed])

    first = source.step().emission
    second = source.step().emission
    third = source.step().emission

    assert first is not None and second is not None and third is not None
    first_ref = first.observation_sample_refs["b601_rs_arm_qpos"]
    second_ref = second.observation_sample_refs["b601_rs_arm_qpos"]
    third_ref = third.observation_sample_refs["b601_rs_arm_qpos"]
    assert first_ref is second_ref
    assert third_ref.sample_id != first_ref.sample_id
    assert (
        first.emission_sequence,
        second.emission_sequence,
        third.emission_sequence,
    ) == (
        0,
        1,
        2,
    )


def test_replayed_state_is_rejected_without_fk_or_sequence_advance() -> None:
    repeated = _state()
    source, _, model = _source([repeated, repeated])

    first = source.step()
    second = source.step()

    assert first.emission_valid
    assert not second.emission_valid
    assert "state sequence did not advance" in second.rejection_reason
    assert len(model.calls) == 1
    assert source.next_emission_sequence == 1
