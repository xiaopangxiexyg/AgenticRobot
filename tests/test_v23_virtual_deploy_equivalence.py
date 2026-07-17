from __future__ import annotations

import json

import numpy as np
from scipy.spatial.transform import Rotation as R

from dataset_contracts import PoseFrameContract, PoseSourceProfile
from robot_motion.deploy.actions import (
    ActionSafetyFilter,
    ActionSafetyFilterConfig,
    NormalizedAction,
    normalize_policy_output,
)
from robot_motion.deploy.canonical_action import CanonicalDeployActionPipeline
from robot_motion.deploy.observation import DeploymentObservation
from robot_motion.deploy.pose_adapter import (
    convert_normalized_action_chunk_to_absolute,
    convert_normalized_action_chunk_to_relative,
    prepare_observation_for_policy_pose_mode,
)
from robot_motion.deploy.schema import (
    ACTION_ARM_POSE_SLICES,
    ACTION_DIM,
    OBSERVATION_STATE_DIM,
    STATE_ARM_POSE_SLICES,
)
from robot_motion.deploy.session_contract import POLICY_ABSOLUTE_FRAME_CANONICAL_DESK
from robot_motion.relative_action import (
    compare_pose_chunks,
    pose_matrix_to_xyzquat,
    xyzquat_to_pose_matrix,
)
from teleop_runtime.deploy.dispatch import dispatch_teleop_command


ACTIVE_SIDES = ("left", "right")
CHUNK_SIZE = 4
CHUNK_COUNT = 3


class _CaptureArmBackend:
    """Non-hardware dispatcher target used at the final real-runtime boundary."""

    def __init__(self) -> None:
        self.commands = []

    def send_command(self, command):
        self.commands.append(command.copy())
        return True


def _pose(
    xyz: tuple[float, float, float] | np.ndarray,
    *,
    euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(xyz, dtype=np.float64),
            R.from_euler("xyz", euler_deg, degrees=True).as_quat(),
        )
    ).astype(np.float32)


def _transform(
    xyz: tuple[float, float, float],
    *,
    euler_deg: tuple[float, float, float],
) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = R.from_euler("xyz", euler_deg, degrees=True).as_matrix()
    value[:3, 3] = xyz
    return value


def _runtime_pose_contract() -> PoseFrameContract:
    return PoseFrameContract.v23_forced_raw(
        PoseSourceProfile(
            source_id="rm65_real_dual_arm_virtual_deploy_test_v1",
            source_family="test",
            source_description="Non-identity dual-arm virtual deployment fixture.",
            topology="dual_arm",
            active_sides=ACTIVE_SIDES,
            raw_absolute_frame_by_side={
                "left": "left_rm65_base_virtual",
                "right": "right_rm65_base_virtual",
            },
            raw_to_desk_transform_by_side={
                "left": _transform((0.12, -0.08, 0.04), euler_deg=(2.0, -3.0, 25.0)),
                "right": _transform((-0.06, 0.10, 0.02), euler_deg=(-2.0, 4.0, -18.0)),
            },
        )
    )


def _raw_anchor_by_side(chunk_index: int) -> dict[str, np.ndarray]:
    index = float(chunk_index)
    return {
        "left": _pose(
            (-0.24 + 0.012 * index, 0.22 - 0.004 * index, 0.36 + 0.003 * index),
            euler_deg=(3.0 + index, -5.0, 8.0 + 2.0 * index),
        ),
        "right": _pose(
            (0.24 - 0.009 * index, -0.22 + 0.006 * index, 0.35 + 0.002 * index),
            euler_deg=(-4.0, 6.0 + index, -7.0 - 1.5 * index),
        ),
    }


def _raw_observation(chunk_index: int) -> DeploymentObservation:
    anchors = _raw_anchor_by_side(chunk_index)
    state = np.full((OBSERVATION_STATE_DIM,), np.nan, dtype=np.float32)
    state_valid = np.zeros((OBSERVATION_STATE_DIM,), dtype=bool)
    last_action = np.full((ACTION_DIM,), np.nan, dtype=np.float32)
    arm_states: dict[str, dict[str, object]] = {}
    state_dict: dict[str, object] = {}
    for side, anchor in anchors.items():
        state[STATE_ARM_POSE_SLICES[side]] = anchor
        state_valid[STATE_ARM_POSE_SLICES[side]] = True
        last_action[ACTION_ARM_POSE_SLICES[side]] = anchor
        state_dict[f"{side}_arm.pose"] = anchor.copy()
        state_dict[f"{side}_arm"] = {"pose": anchor.copy()}
        arm_states[side] = {
            "actual_tcp_pose": anchor.tolist(),
            "current_tcp_pose": xyzquat_to_pose_matrix(anchor).copy(),
            "pose_measurement_source": "virtual_measured_tcp",
        }
    return DeploymentObservation(
        timestamp=100.0 + float(chunk_index),
        step_index=chunk_index * CHUNK_SIZE,
        active_sides=ACTIVE_SIDES,
        arms_enabled=True,
        hands_enabled=False,
        state=state,
        state_valid=state_valid,
        state_dict=state_dict,
        last_action=last_action,
        arm_states=arm_states,
        metadata={"virtual_deploy": True},
    )


def _canonical_desk_absolute_chunk(
    raw_observation: DeploymentObservation,
    contract: PoseFrameContract,
) -> np.ndarray:
    """Build targets from one fixed measured anchor for the whole chunk."""

    deltas = (
        ((0.004, -0.002, 0.001), (0.0, 0.0, 1.0)),
        ((0.008, 0.001, 0.003), (1.5, -0.5, 2.0)),
        ((0.012, 0.004, 0.002), (2.0, 1.0, 3.0)),
        ((0.016, 0.002, 0.005), (2.5, -1.0, 4.0)),
    )
    chunk = np.full((CHUNK_SIZE, ACTION_DIM), np.nan, dtype=np.float32)
    for side in ACTIVE_SIDES:
        raw_anchor = np.asarray(
            raw_observation.arm_states[side]["actual_tcp_pose"], dtype=np.float32
        )
        desk_anchor = contract.storage_pose_vector_to_canonical_desk(raw_anchor, side=side)
        desk_anchor_matrix = xyzquat_to_pose_matrix(desk_anchor)
        for row_index, (delta_xyz, delta_euler) in enumerate(deltas):
            signed_xyz = np.asarray(delta_xyz, dtype=np.float64)
            signed_euler = np.asarray(delta_euler, dtype=np.float64)
            if side == "right":
                signed_xyz[1] *= -1.0
                signed_euler[:2] *= -1.0
            delta = _transform(tuple(signed_xyz), euler_deg=tuple(signed_euler))
            target = desk_anchor_matrix @ delta
            chunk[row_index, ACTION_ARM_POSE_SLICES[side]] = pose_matrix_to_xyzquat(target)
    return chunk


def _pipeline() -> CanonicalDeployActionPipeline:
    # Rate limits are disabled only inside this equivalence fixture. The real
    # safety and command construction code is still exercised identically by
    # both branches, with a broad finite virtual workspace.
    return CanonicalDeployActionPipeline(
        safety_filter=ActionSafetyFilter(
            ActionSafetyFilterConfig(
                active_sides=ACTIVE_SIDES,
                arms_enabled=True,
                hands_enabled=False,
                workspace_min=(-2.0, -2.0, -2.0),
                workspace_max=(2.0, 2.0, 2.0),
                workspace_soft_margin_m=0.0,
                max_arm_xyz_step_m=0.0,
                max_arm_rot_step_deg=0.0,
                max_arm_joint_step_rad=0.0,
            )
        ),
        active_sides=ACTIVE_SIDES,
        arms_enabled=True,
        hands_enabled=False,
    )


def _dispatch_chunk(
    actions: list[NormalizedAction],
    observation: DeploymentObservation,
    backends: dict[str, _CaptureArmBackend],
    *,
    policy_name: str,
) -> None:
    pipeline = _pipeline()
    for action in actions:
        prepared = pipeline.prepare(action, observation, policy_name=policy_name)
        assert prepared.faults == ()
        assert prepared.command is not None
        receipt = dispatch_teleop_command(
            prepared.command,
            arm_backends=backends,
            hand_backends={},
        )
        assert receipt["accepted"] is True


def _max_dispatched_matrix_error(
    first: dict[str, _CaptureArmBackend],
    second: dict[str, _CaptureArmBackend],
) -> float:
    maximum = 0.0
    for side in ACTIVE_SIDES:
        first_commands = first[side].commands
        second_commands = second[side].commands
        assert len(first_commands) == CHUNK_COUNT * CHUNK_SIZE
        assert len(first_commands) == len(second_commands)
        for first_command, second_command in zip(first_commands, second_commands, strict=True):
            assert first_command.command_type == "cartesian_pose"
            assert second_command.command_type == "cartesian_pose"
            maximum = max(
                maximum,
                float(
                    np.max(
                        np.abs(
                            np.asarray(first_command.value, dtype=np.float64)
                            - np.asarray(second_command.value, dtype=np.float64)
                        )
                    )
                ),
            )
    return maximum


def test_v23_virtual_absolute_and_relative_deploy_are_equivalent() -> None:
    """A/B the two model representations through the real deploy boundaries."""

    contract = _runtime_pose_contract()
    absolute_backends = {side: _CaptureArmBackend() for side in ACTIVE_SIDES}
    relative_backends = {side: _CaptureArmBackend() for side in ACTIVE_SIDES}
    decoded_absolute_rows: list[np.ndarray] = []
    decoded_relative_rows: list[np.ndarray] = []
    wrong_anchor_translation_error_m = 0.0

    for chunk_index in range(CHUNK_COUNT):
        raw_observation = _raw_observation(chunk_index)
        desk_absolute_chunk = _canonical_desk_absolute_chunk(raw_observation, contract)

        absolute_policy_observation = prepare_observation_for_policy_pose_mode(
            raw_observation,
            target_mode="absolute",
            active_sides=ACTIVE_SIDES,
            policy_absolute_frame=POLICY_ABSOLUTE_FRAME_CANONICAL_DESK,
            runtime_pose_frame_contract=contract,
        )
        relative_policy_observation = prepare_observation_for_policy_pose_mode(
            raw_observation,
            target_mode="relative",
            active_sides=ACTIVE_SIDES,
            policy_absolute_frame=POLICY_ABSOLUTE_FRAME_CANONICAL_DESK,
            runtime_pose_frame_contract=contract,
        )
        np.testing.assert_allclose(
            absolute_policy_observation.state,
            relative_policy_observation.state,
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )

        # These are the two possible outputs of otherwise equivalent policies.
        # The normalization call matches run_policy's post-predict boundary.
        absolute_policy_actions = normalize_policy_output(
            {"action_chunk": desk_absolute_chunk},
            raw_observation,
            active_sides=ACTIVE_SIDES,
            arms_enabled=True,
            hands_enabled=False,
        )
        relative_model_actions = convert_normalized_action_chunk_to_relative(
            absolute_policy_actions,
            relative_policy_observation,
            active_sides=ACTIVE_SIDES,
        )
        relative_policy_actions = normalize_policy_output(
            {"action_chunk": np.stack([item.vector for item in relative_model_actions])},
            raw_observation,
            active_sides=ACTIVE_SIDES,
            arms_enabled=True,
            hands_enabled=False,
        )

        decoded_absolute = convert_normalized_action_chunk_to_absolute(
            absolute_policy_actions,
            raw_observation,
            source_mode="absolute",
            active_sides=ACTIVE_SIDES,
            policy_absolute_frame=POLICY_ABSOLUTE_FRAME_CANONICAL_DESK,
            runtime_pose_frame_contract=contract,
        )
        decoded_relative = convert_normalized_action_chunk_to_absolute(
            relative_policy_actions,
            raw_observation,
            source_mode="relative",
            active_sides=ACTIVE_SIDES,
            policy_absolute_frame=POLICY_ABSOLUTE_FRAME_CANONICAL_DESK,
            runtime_pose_frame_contract=contract,
        )
        absolute_array = np.stack([item.vector for item in decoded_absolute])
        relative_array = np.stack([item.vector for item in decoded_relative])
        decoded_absolute_rows.extend(absolute_array)
        decoded_relative_rows.extend(relative_array)

        _dispatch_chunk(
            decoded_absolute,
            raw_observation,
            absolute_backends,
            policy_name="virtual_absolute",
        )
        _dispatch_chunk(
            decoded_relative,
            raw_observation,
            relative_backends,
            policy_name="virtual_relative",
        )

        # Negative control: use the next measured snapshot as the relative
        # decoder anchor. This must no longer agree with the absolute branch.
        wrong_anchor_decoded = convert_normalized_action_chunk_to_absolute(
            relative_policy_actions,
            _raw_observation(chunk_index + 1),
            source_mode="relative",
            active_sides=ACTIVE_SIDES,
            policy_absolute_frame=POLICY_ABSOLUTE_FRAME_CANONICAL_DESK,
            runtime_pose_frame_contract=contract,
        )
        wrong_metrics = compare_pose_chunks(
            absolute_array,
            np.stack([item.vector for item in wrong_anchor_decoded]),
            active_sides=ACTIVE_SIDES,
        )
        wrong_anchor_translation_error_m = max(
            wrong_anchor_translation_error_m,
            *(item.max_translation_error_m for item in wrong_metrics.values()),
        )

    absolute_all = np.stack(decoded_absolute_rows)
    relative_all = np.stack(decoded_relative_rows)
    metrics_by_side = compare_pose_chunks(
        absolute_all,
        relative_all,
        active_sides=ACTIVE_SIDES,
    )
    max_translation_error_m = max(
        item.max_translation_error_m for item in metrics_by_side.values()
    )
    max_rotation_error_rad = max(
        item.max_rotation_error_rad for item in metrics_by_side.values()
    )
    max_dispatched_matrix_error = _max_dispatched_matrix_error(
        absolute_backends, relative_backends
    )

    report = {
        "active_sides": list(ACTIVE_SIDES),
        "chunk_count": CHUNK_COUNT,
        "chunk_size": CHUNK_SIZE,
        "frame_count": CHUNK_COUNT * CHUNK_SIZE,
        "runtime_pose_frame_contract_hash": contract.contract_hash,
        "max_decoded_translation_error_m": max_translation_error_m,
        "max_decoded_rotation_error_rad": max_rotation_error_rad,
        "max_dispatched_matrix_element_error": max_dispatched_matrix_error,
        "wrong_next_anchor_max_translation_error_m": wrong_anchor_translation_error_m,
    }
    print("V23_VIRTUAL_DEPLOY_REPORT=" + json.dumps(report, sort_keys=True))

    assert max_translation_error_m < 2e-6
    assert max_rotation_error_rad < 2e-6
    assert max_dispatched_matrix_error < 2e-6
    assert wrong_anchor_translation_error_m > 5e-3
