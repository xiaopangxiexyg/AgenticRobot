from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dataset_contracts import (
    DEXTELEOP_DATASET_SCHEMA_V23,
    PolicyArtifactContract,
    PoseActionChunkContract,
    PoseFrameContract,
    PoseSourceProfile,
    PoseTrainingDataContract,
)
from dataset_contracts.v22 import PolicyIOLayout
from robot_motion.deploy.actions import NormalizedAction
from robot_motion.deploy.observation import DeploymentObservation, build_deployment_observation
from robot_motion.deploy.pose_adapter import (
    PoseDeploymentError,
    convert_last_action_to_policy_mode,
    convert_normalized_action_chunk_to_absolute,
    convert_normalized_action_chunk_to_relative,
    prepare_observation_for_policy_pose_mode,
)
from robot_motion.deploy.schema import ACTION_ARM_POSE_SLICES, ACTION_DIM, OBSERVATION_STATE_DIM
from robot_motion.deploy.schema import STATE_ARM_POSE_SLICES
from robot_motion.deploy.session_contract import (
    POLICY_ABSOLUTE_FRAME_CANONICAL_DESK,
    resolve_deploy_session_contract,
)
from robot_motion.relative_action import compare_pose_chunks
from teleop_runtime.deploy.arm_state_reader import (
    ArmStateReader as CanonicalArmStateReader,
)
from teleop_runtime.deploy.run_policy import ArmStateReader
from teleop_runtime.deploy.startup_init import resolve_startup_init_action


def _observation(*, actual_x: float = 0.1) -> DeploymentObservation:
    return DeploymentObservation(
        timestamp=10.0,
        step_index=1,
        active_sides=("right",),
        arms_enabled=True,
        hands_enabled=False,
        state=np.full((OBSERVATION_STATE_DIM,), np.nan, dtype=np.float32),
        state_valid=np.zeros((OBSERVATION_STATE_DIM,), dtype=bool),
        arm_states={
            "right": {
                "actual_tcp_pose": [actual_x, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0],
            }
        },
    )


def test_run_policy_reexports_canonical_arm_state_reader() -> None:
    assert ArmStateReader is CanonicalArmStateReader


def _action(x: float) -> NormalizedAction:
    vector = np.full((ACTION_DIM,), np.nan, dtype=np.float32)
    vector[ACTION_ARM_POSE_SLICES["right"]] = [x, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]
    return NormalizedAction(vector=vector, metadata={"source": "test"})


def _right_runtime_pose_contract() -> PoseFrameContract:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    transform[:3, 3] = [1.0, -0.5, 0.2]
    return PoseFrameContract.v23_forced_raw(
        PoseSourceProfile(
            source_id="rm65_runtime_right_test_v1",
            source_family="test",
            source_description="non-identity runtime fixture",
            topology="single_arm",
            active_sides=("right",),
            raw_absolute_frame_by_side={"right": "right_runtime_raw"},
            raw_to_desk_transform_by_side={"right": transform},
        )
    )


def test_relative_policy_chunk_is_converted_before_canonical_safety() -> None:
    observation = _observation(actual_x=0.1)
    relative = _action(0.2)
    converted = convert_normalized_action_chunk_to_absolute(
        [relative], observation, source_mode="relative", active_sides=("right",)
    )
    np.testing.assert_allclose(
        converted[0].vector[ACTION_ARM_POSE_SLICES["right"]][:3],
        [0.3, 0.0, 0.6],
        atol=1e-6,
    )
    assert converted[0].metadata["pose_action_mode"] == "absolute"
    assert converted[0].metadata["pose_action_source_mode"] == "relative"


def test_absolute_replay_chunk_can_be_emitted_relative_and_decoded_back() -> None:
    observation = _observation(actual_x=0.1)
    absolute = _action(0.4)
    relative = convert_normalized_action_chunk_to_relative(
        [absolute], observation, active_sides=("right",)
    )
    reconstructed = convert_normalized_action_chunk_to_absolute(
        relative, observation, source_mode="relative", active_sides=("right",)
    )
    metrics = compare_pose_chunks(
        np.stack([absolute.vector]),
        np.stack([reconstructed[0].vector]),
        active_sides=("right",),
    )["right"]
    assert metrics.max_translation_error_m < 1e-6
    assert metrics.max_rotation_error_rad < 1e-6


def test_last_action_relative_uses_current_measured_tcp() -> None:
    last = _action(0.2).vector
    converted = convert_last_action_to_policy_mode(
        last, _observation(actual_x=0.1), target_mode="relative", active_sides=("right",)
    )
    np.testing.assert_allclose(
        converted[ACTION_ARM_POSE_SLICES["right"]][:3],
        [0.1, 0.0, 0.0],
        atol=1e-6,
    )


def test_policy_observation_converts_finite_last_action_and_preserves_startup_nan() -> None:
    observation = _observation(actual_x=0.1)
    observation = DeploymentObservation(
        **{
            **observation.__dict__,
            "last_action": _action(0.2).vector,
        }
    )
    converted = prepare_observation_for_policy_pose_mode(
        observation, target_mode="relative", active_sides=("right",)
    )
    np.testing.assert_allclose(
        converted.last_action[ACTION_ARM_POSE_SLICES["right"]][:3],
        [0.1, 0.0, 0.0],
        atol=1e-6,
    )
    startup = prepare_observation_for_policy_pose_mode(
        _observation(actual_x=0.1), target_mode="relative", active_sides=("right",)
    )
    assert np.all(np.isnan(startup.last_action))


def test_nonidentity_v23_absolute_and_relative_decode_to_same_runtime_raw_target() -> None:
    contract = _right_runtime_pose_contract()
    raw_observation = _observation(actual_x=0.1)
    raw_state = raw_observation.state.copy()
    raw_state[STATE_ARM_POSE_SLICES["right"]] = [
        0.1,
        0.0,
        0.3,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    raw_observation = DeploymentObservation(
        **{
            **raw_observation.__dict__,
            "state": raw_state,
            "last_action": _action(0.2).vector,
        }
    )
    policy_observation = prepare_observation_for_policy_pose_mode(
        raw_observation,
        target_mode="relative",
        active_sides=("right",),
        policy_absolute_frame=POLICY_ABSOLUTE_FRAME_CANONICAL_DESK,
        runtime_pose_frame_contract=contract,
    )
    expected_state_desk = contract.storage_pose_vector_to_canonical_desk(
        raw_state[STATE_ARM_POSE_SLICES["right"]], side="right"
    )
    np.testing.assert_allclose(
        policy_observation.state[STATE_ARM_POSE_SLICES["right"]],
        expected_state_desk,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        raw_observation.state[STATE_ARM_POSE_SLICES["right"]],
        raw_state[STATE_ARM_POSE_SLICES["right"]],
        atol=0.0,
    )

    raw_target = _action(0.4)
    desk_target = raw_target.vector.copy()
    desk_target[ACTION_ARM_POSE_SLICES["right"]] = (
        contract.storage_pose_vector_to_canonical_desk(
            raw_target.vector[ACTION_ARM_POSE_SLICES["right"]], side="right"
        )
    )
    absolute_raw = convert_normalized_action_chunk_to_absolute(
        [NormalizedAction(vector=desk_target)],
        raw_observation,
        source_mode="absolute",
        active_sides=("right",),
        policy_absolute_frame=POLICY_ABSOLUTE_FRAME_CANONICAL_DESK,
        runtime_pose_frame_contract=contract,
    )[0]
    relative = convert_normalized_action_chunk_to_relative(
        [NormalizedAction(vector=desk_target)],
        policy_observation,
        active_sides=("right",),
    )
    relative_raw = convert_normalized_action_chunk_to_absolute(
        relative,
        raw_observation,
        source_mode="relative",
        active_sides=("right",),
        policy_absolute_frame=POLICY_ABSOLUTE_FRAME_CANONICAL_DESK,
        runtime_pose_frame_contract=contract,
    )[0]
    np.testing.assert_allclose(absolute_raw.vector, raw_target.vector, atol=1e-6)
    np.testing.assert_allclose(relative_raw.vector, raw_target.vector, atol=1e-6)


def test_structured_v23_session_freezes_runtime_pose_binding() -> None:
    runtime_contract = PoseFrameContract.legacy_v22_raw(active_sides=("left", "right"))
    artifact = PolicyArtifactContract(
        policy_io_layout=PolicyIOLayout(),
        dataset_schema=DEXTELEOP_DATASET_SCHEMA_V23,
        pose_action_chunk=PoseActionChunkContract(
            action_horizon=4,
            action_pose_mode="relative",
        ),
        pose_training_data=PoseTrainingDataContract.native_v23(),
        pose_frame_contract_hashes=(runtime_contract.contract_hash,),
    )
    session = resolve_deploy_session_contract(
        active_sides=("left", "right"),
        policy_artifact=artifact,
        hardware_output=False,
        runtime_pose_frame_contract=runtime_contract,
    )
    assert session.policy_absolute_frame == POLICY_ABSOLUTE_FRAME_CANONICAL_DESK
    assert session.runtime_pose_frame_contract == runtime_contract
    assert session.to_dict()["runtime_pose_frame_contract_hash"] == runtime_contract.contract_hash


def test_structured_v23_session_uses_marked_identity_compatibility_when_omitted() -> None:
    training_contract = PoseFrameContract.legacy_v22_raw(
        active_sides=("left", "right")
    )
    artifact = PolicyArtifactContract(
        policy_io_layout=PolicyIOLayout(),
        dataset_schema=DEXTELEOP_DATASET_SCHEMA_V23,
        pose_action_chunk=PoseActionChunkContract(action_horizon=2),
        pose_training_data=PoseTrainingDataContract.v22_raw_identity(
            active_sides=("left", "right")
        ),
        pose_frame_contract_hashes=(training_contract.contract_hash,),
    )
    session = resolve_deploy_session_contract(
        active_sides=("left", "right"),
        policy_artifact=artifact,
        hardware_output=False,
    )
    assert session.policy_absolute_frame == POLICY_ABSOLUTE_FRAME_CANONICAL_DESK
    assert session.runtime_pose_frame_contract is not None
    assert session.metadata["runtime_pose_identity_compatibility"] is True
    assert session.metadata["pose_training_data_origin"] == (
        "v22_raw_identity_compatibility"
    )
    assert "v22_raw_identity_compatibility_policy_state_binding" in session.warnings


def test_v22_identity_trained_policy_rejects_nonidentity_runtime_state_mapping() -> None:
    transform = np.eye(4, dtype=float)
    transform[0, 3] = 0.2
    runtime_contract = PoseFrameContract.legacy_v22_raw(
        active_sides=("left", "right"),
        raw_to_desk_transform_by_side={"left": transform},
    )
    artifact = PolicyArtifactContract(
        policy_io_layout=PolicyIOLayout(),
        dataset_schema=DEXTELEOP_DATASET_SCHEMA_V23,
        pose_action_chunk=PoseActionChunkContract(
            action_horizon=2,
            action_pose_mode="relative",
        ),
        pose_training_data=PoseTrainingDataContract.v22_raw_identity(
            active_sides=("left", "right")
        ),
        pose_frame_contract_hashes=(
            PoseFrameContract.legacy_v22_raw(
                active_sides=("left", "right")
            ).contract_hash,
        ),
    )
    with pytest.raises(ValueError, match="cannot use a non-identity runtime"):
        resolve_deploy_session_contract(
            active_sides=("left", "right"),
            policy_artifact=artifact,
            runtime_pose_frame_contract=runtime_contract,
            hardware_output=False,
        )


def test_native_v23_trained_policy_requires_explicit_runtime_pose_binding() -> None:
    training_contract = PoseFrameContract.legacy_v22_raw(
        active_sides=("left", "right")
    )
    artifact = PolicyArtifactContract(
        policy_io_layout=PolicyIOLayout(),
        dataset_schema=DEXTELEOP_DATASET_SCHEMA_V23,
        pose_action_chunk=PoseActionChunkContract(action_horizon=2),
        pose_training_data=PoseTrainingDataContract.native_v23(),
        pose_frame_contract_hashes=(training_contract.contract_hash,),
    )
    with pytest.raises(ValueError, match="requires an explicit"):
        resolve_deploy_session_contract(
            active_sides=("left", "right"),
            policy_artifact=artifact,
            hardware_output=False,
        )


def test_v22_identity_trained_policy_rejects_different_runtime_source() -> None:
    runtime_contract = PoseFrameContract.v23_forced_raw(
        PoseSourceProfile(
            source_id="different_rm65_station_v1",
            source_family="real_robot",
            source_description="identity transform but different station",
            topology="dual_arm",
        )
    )
    training_contract = PoseFrameContract.legacy_v22_raw(
        active_sides=("left", "right")
    )
    artifact = PolicyArtifactContract(
        policy_io_layout=PolicyIOLayout(),
        dataset_schema=DEXTELEOP_DATASET_SCHEMA_V23,
        pose_action_chunk=PoseActionChunkContract(action_horizon=2),
        pose_training_data=PoseTrainingDataContract.v22_raw_identity(
            active_sides=("left", "right")
        ),
        pose_frame_contract_hashes=(training_contract.contract_hash,),
    )
    with pytest.raises(ValueError, match="requires runtime pose source_id"):
        resolve_deploy_session_contract(
            active_sides=("left", "right"),
            policy_artifact=artifact,
            runtime_pose_frame_contract=runtime_contract,
            hardware_output=False,
        )


def test_dual_side_runtime_pose_transforms_are_applied_independently() -> None:
    left_transform = np.eye(4, dtype=float)
    left_transform[0, 3] = 1.0
    right_transform = np.eye(4, dtype=float)
    right_transform[1, 3] = -2.0
    contract = PoseFrameContract.v23_forced_raw(
        PoseSourceProfile(
            source_id="rm65_runtime_dual_test_v1",
            source_family="test",
            source_description="dual independent fixture",
            topology="dual_arm",
            raw_to_desk_transform_by_side={
                "left": left_transform,
                "right": right_transform,
            },
        )
    )
    observation = DeploymentObservation(
        timestamp=1.0,
        step_index=0,
        active_sides=("left", "right"),
        arms_enabled=True,
        hands_enabled=False,
    )
    desk_action = np.full((ACTION_DIM,), np.nan, dtype=np.float32)
    desk_action[ACTION_ARM_POSE_SLICES["left"]] = [
        1.2,
        0.0,
        0.3,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    desk_action[ACTION_ARM_POSE_SLICES["right"]] = [
        0.4,
        -1.5,
        0.3,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    decoded = convert_normalized_action_chunk_to_absolute(
        [NormalizedAction(vector=desk_action)],
        observation,
        source_mode="absolute",
        active_sides=("left", "right"),
        policy_absolute_frame=POLICY_ABSOLUTE_FRAME_CANONICAL_DESK,
        runtime_pose_frame_contract=contract,
    )[0].vector
    np.testing.assert_allclose(
        decoded[ACTION_ARM_POSE_SLICES["left"]][:3], [0.2, 0.0, 0.3], atol=1e-6
    )
    np.testing.assert_allclose(
        decoded[ACTION_ARM_POSE_SLICES["right"]][:3], [0.4, 0.5, 0.3], atol=1e-6
    )


def test_dataset_source_and_runtime_pose_contracts_are_not_conflated() -> None:
    dataset_transform = np.eye(4, dtype=float)
    dataset_transform[0, 3] = -0.7
    dataset_contract = PoseFrameContract.v23_forced_raw(
        PoseSourceProfile(
            source_id="umi_source_test_v1",
            source_family="handheld",
            source_description="heterogeneous source fixture",
            topology="single_arm",
            active_sides=("right",),
            raw_to_desk_transform_by_side={"right": dataset_transform},
        )
    )
    runtime_contract = _right_runtime_pose_contract()
    dataset_raw_target = _action(0.9).vector
    desk_target = dataset_raw_target.copy()
    desk_target[ACTION_ARM_POSE_SLICES["right"]] = (
        dataset_contract.storage_pose_vector_to_canonical_desk(
            dataset_raw_target[ACTION_ARM_POSE_SLICES["right"]], side="right"
        )
    )
    expected_runtime_raw = desk_target.copy()
    expected_runtime_raw[ACTION_ARM_POSE_SLICES["right"]] = (
        runtime_contract.canonical_desk_pose_vector_to_storage(
            desk_target[ACTION_ARM_POSE_SLICES["right"]], side="right"
        )
    )

    raw_observation = _observation(actual_x=0.1)
    raw_state = raw_observation.state.copy()
    raw_state[STATE_ARM_POSE_SLICES["right"]] = [
        0.1,
        0.0,
        0.3,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    raw_observation = DeploymentObservation(
        **{**raw_observation.__dict__, "state": raw_state}
    )
    policy_observation = prepare_observation_for_policy_pose_mode(
        raw_observation,
        target_mode="relative",
        active_sides=("right",),
        policy_absolute_frame=POLICY_ABSOLUTE_FRAME_CANONICAL_DESK,
        runtime_pose_frame_contract=runtime_contract,
    )
    absolute_runtime_raw = convert_normalized_action_chunk_to_absolute(
        [NormalizedAction(vector=desk_target)],
        raw_observation,
        source_mode="absolute",
        active_sides=("right",),
        policy_absolute_frame=POLICY_ABSOLUTE_FRAME_CANONICAL_DESK,
        runtime_pose_frame_contract=runtime_contract,
    )[0].vector
    relative = convert_normalized_action_chunk_to_relative(
        [NormalizedAction(vector=desk_target)],
        policy_observation,
        active_sides=("right",),
    )
    relative_runtime_raw = convert_normalized_action_chunk_to_absolute(
        relative,
        raw_observation,
        source_mode="relative",
        active_sides=("right",),
        policy_absolute_frame=POLICY_ABSOLUTE_FRAME_CANONICAL_DESK,
        runtime_pose_frame_contract=runtime_contract,
    )[0].vector
    np.testing.assert_allclose(
        absolute_runtime_raw,
        expected_runtime_raw,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        relative_runtime_raw,
        expected_runtime_raw,
        atol=1e-6,
    )


def test_v23_explicit_dataset_startup_action_is_converted_to_canonical_desk(
    monkeypatch,
) -> None:
    contract = _right_runtime_pose_contract()
    layout = PolicyIOLayout()
    dataset_info = {
        "dexteleop_dataset_schema": DEXTELEOP_DATASET_SCHEMA_V23,
        **layout.binding_metadata(),
        "pose_frame_contract": contract.to_dict(),
        "pose_storage_frame": contract.pose_storage_frame,
    }
    raw_action = _action(0.4).vector.reshape(1, -1)
    monkeypatch.setattr(
        "teleop_runtime.deploy.startup_init.resolve_replay_parquet_path",
        lambda **_: (
            Path("/tmp/init.parquet"),
            {"dataset_info": dataset_info, "episode_index": 0},
        ),
    )
    monkeypatch.setattr(
        "teleop_runtime.deploy.startup_init.load_replay_actions_from_parquet",
        lambda *_, **__: raw_action.copy(),
    )
    args = SimpleNamespace(
        init_source="dataset",
        init_time_s=1.0,
        init_dataset="/tmp/dataset",
        init_parquet=None,
        init_episode=0,
        init_frame=0,
        init_action_column="action",
    )
    action, source = resolve_startup_init_action(args, object())
    assert action is not None
    expected = contract.storage_pose_vector_to_canonical_desk(
        raw_action[0, ACTION_ARM_POSE_SLICES["right"]], side="right"
    )
    np.testing.assert_allclose(
        action[ACTION_ARM_POSE_SLICES["right"]], expected, atol=1e-6
    )
    assert "policy_absolute_frame=canonical_desk" in source


def test_relative_deployment_fails_closed_without_actual_tcp() -> None:
    observation = DeploymentObservation(
        timestamp=10.0,
        step_index=1,
        active_sides=("right",),
        arms_enabled=True,
        hands_enabled=False,
        state=np.full((OBSERVATION_STATE_DIM,), np.nan, dtype=np.float32),
        state_valid=np.zeros((OBSERVATION_STATE_DIM,), dtype=bool),
        arm_states={"right": {"current_tcp_pose": [0.1, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]}},
    )
    with pytest.raises(PoseDeploymentError, match="actual_tcp_pose"):
        convert_normalized_action_chunk_to_absolute(
            [_action(0.1)], observation, source_mode="relative", active_sides=("right",)
        )


class _MeasuredDeployArmBackend:
    def __init__(self, *, read_success: bool = True) -> None:
        self.read_success = bool(read_success)
        self.hardware_qpos = np.asarray([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.command_qpos = np.asarray([9.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.measurement_timestamp = time.time() - 0.1

    def is_ready(self) -> bool:
        return True

    def refresh_state_from_hardware(self) -> dict[str, object]:
        return self.get_state()

    def get_state(self) -> dict[str, object]:
        return {
            "current_qpos": self.command_qpos.copy(),
            "current_tcp_pose": self.compute_tcp_pose(qpos=self.command_qpos),
            "hardware_qpos": self.hardware_qpos.copy(),
            "hardware_measurement_timestamp": self.measurement_timestamp,
            "hardware_measurement_sequence": 3,
            "hardware_measurement_valid": True,
            "hardware_read_success": self.read_success,
        }

    def compute_tcp_pose(self, *, qpos=None) -> np.ndarray:
        pose = np.eye(4, dtype=float)
        pose[0, 3] = float(np.asarray(qpos, dtype=float).reshape(6)[0])
        return pose


def test_arm_state_reader_publishes_measured_actual_tcp_pose_for_relative_deploy() -> None:
    backend = _MeasuredDeployArmBackend()
    reader = ArmStateReader({"right": backend}, refresh_hz=30.0)
    states = reader.snapshot(time.time() + 1.0)

    np.testing.assert_allclose(
        states["right"]["actual_tcp_pose"],
        backend.compute_tcp_pose(qpos=backend.hardware_qpos),
    )
    assert states["right"]["actual_tcp_pose_source"] == "hardware_qpos_fk"
    assert states["right"]["actual_tcp_pose_timestamp"] == pytest.approx(
        backend.measurement_timestamp
    )

    observation = build_deployment_observation(
        timestamp=time.time() + 1.0,
        step_index=1,
        active_sides=("right",),
        arms_enabled=True,
        hands_enabled=False,
        hand_states={},
        arm_states=states,
    )
    converted = convert_normalized_action_chunk_to_absolute(
        [_action(0.1)], observation, source_mode="relative", active_sides=("right",)
    )
    assert converted[0].vector[ACTION_ARM_POSE_SLICES["right"]][0] == pytest.approx(0.3)


def test_arm_state_reader_does_not_publish_actual_tcp_pose_after_failed_read() -> None:
    backend = _MeasuredDeployArmBackend(read_success=False)
    reader = ArmStateReader({"right": backend}, refresh_hz=30.0)
    state = reader.snapshot(time.time() + 1.0)["right"]

    assert state["actual_tcp_pose"] is None
    assert state["actual_tcp_pose_source"] is None


def test_arm_state_reader_post_refresh_cut_is_live_only() -> None:
    def build_reader():
        clock_value = [100.0]
        backend = _MeasuredDeployArmBackend()

        def refresh_state_from_hardware():
            clock_value[0] = 100.1
            backend.measurement_timestamp = clock_value[0]
            return backend.get_state()

        backend.refresh_state_from_hardware = refresh_state_from_hardware
        return (
            ArmStateReader(
                {"right": backend},
                refresh_hz=30.0,
                clock=lambda: clock_value[0],
            ),
            clock_value,
        )

    historical_reader, _ = build_reader()
    historical = historical_reader.snapshot(100.0)["right"]
    assert historical["causal_snapshot_missing"] is True

    live_reader, _ = build_reader()
    live = live_reader.snapshot(100.0, cut_after_refresh=True)["right"]
    assert live["causal_snapshot_missing"] is False
    assert live["causal_snapshot_cutoff_timestamp"] == pytest.approx(100.1)
    assert live["hardware_measurement_timestamp"] == pytest.approx(100.1)
