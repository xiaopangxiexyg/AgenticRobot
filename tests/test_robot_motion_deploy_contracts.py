from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import inspect

import numpy as np

import robot_motion.deploy.actions as robot_actions
import robot_motion.deploy.canonical_action as robot_canonical
import robot_motion.deploy.dispatcher as robot_dispatcher
import robot_motion.deploy.observation as robot_observation
import robot_motion.deploy.runtime as robot_runtime
import robot_motion.deploy.schema as robot_schema
import teleop_runtime.deploy.actions as legacy_actions
import teleop_runtime.deploy.canonical_action as legacy_canonical
import teleop_runtime.deploy.observation as legacy_observation
import teleop_runtime.deploy.schema as legacy_schema
from robot_motion.action import EE_POSE_VECTOR_DIM


def test_legacy_deploy_schema_reexports_robot_motion_objects() -> None:
    assert legacy_schema.ACTION_DIM == robot_schema.ACTION_DIM == 26
    assert legacy_schema.DEPLOY_ACTION_DIM == robot_schema.DEPLOY_ACTION_DIM == 26
    assert legacy_schema.OBSERVATION_STATE_DIM == robot_schema.OBSERVATION_STATE_DIM == 38
    assert legacy_schema.DEPLOY_OBSERVATION_STATE_DIM == (
        robot_schema.DEPLOY_OBSERVATION_STATE_DIM
    ) == 38
    assert legacy_schema.ACTION_ARM_POSE_SLICES == robot_schema.ACTION_ARM_POSE_SLICES
    assert legacy_schema.ACTION_HAND_SLICES == robot_schema.ACTION_HAND_SLICES
    assert legacy_schema.STATE_ARM_QPOS_SLICES == robot_schema.STATE_ARM_QPOS_SLICES
    assert legacy_schema.STATE_ARM_POSE_SLICES == robot_schema.STATE_ARM_POSE_SLICES
    assert legacy_schema.STATE_HAND_SLICES == robot_schema.STATE_HAND_SLICES
    assert legacy_schema.nan_vector is robot_schema.nan_vector
    assert legacy_schema.vector_or_none is robot_schema.vector_or_none
    assert legacy_schema.normalize_active_sides is robot_schema.normalize_active_sides
    assert legacy_schema.pose_matrix_to_xyzquat is robot_schema.pose_matrix_to_xyzquat
    assert legacy_schema.xyzquat_to_pose_matrix is robot_schema.xyzquat_to_pose_matrix
    assert legacy_schema.mapping_get is robot_schema.mapping_get
    assert legacy_schema.DEPLOY_ACTION_VECTOR_LAYOUT_VERSION == (
        robot_schema.DEPLOY_ACTION_VECTOR_LAYOUT_VERSION
    )
    assert legacy_schema.DEPLOY_ACTION_VECTOR_ACTION_SPACE == (
        robot_schema.DEPLOY_ACTION_VECTOR_ACTION_SPACE
    )
    assert legacy_schema.DEPLOY_ACTION_VECTOR_COMPONENTS == (
        robot_schema.DEPLOY_ACTION_VECTOR_COMPONENTS
    )


def test_deploy_action_vector_layout_is_explicit_and_distinct_from_codec_layer() -> None:
    assert robot_schema.ACTION_DIM == 26
    assert robot_schema.DEPLOY_ACTION_DIM == robot_schema.ACTION_DIM
    assert robot_schema.DEPLOY_OBSERVATION_STATE_DIM == robot_schema.OBSERVATION_STATE_DIM
    assert robot_schema.ACTION_DIM == EE_POSE_VECTOR_DIM
    assert robot_schema.DEPLOY_ACTION_VECTOR_LAYOUT_VERSION == "deploy_action_vector.ee_pose_q6.v1"
    assert robot_schema.DEPLOY_ACTION_VECTOR_ACTION_SPACE == "bimanual_ee_pose_v1"
    assert robot_schema.DEPLOY_ACTION_VECTOR_COMPONENTS == (
        ("left_arm_pose", "left", "arm_pose_xyzquat_xyzw", slice(0, 7), 7),
        ("right_arm_pose", "right", "arm_pose_xyzquat_xyzw", slice(7, 14), 7),
        ("left_hand_q6", "left", "hand_q6", slice(14, 20), 6),
        ("right_hand_q6", "right", "hand_q6", slice(20, 26), 6),
    )


def test_legacy_deploy_observation_reexports_robot_motion_objects() -> None:
    assert legacy_observation.DeploymentObservation is robot_observation.DeploymentObservation
    assert legacy_observation.build_deployment_observation is (
        robot_observation.build_deployment_observation
    )
    assert legacy_observation.build_state_vector is robot_observation.build_state_vector
    assert legacy_observation.build_tactile_vectors is robot_observation.build_tactile_vectors
    assert legacy_observation.observation_faults is robot_observation.observation_faults


def test_legacy_deploy_actions_reexport_robot_motion_objects() -> None:
    assert legacy_actions.NormalizedAction is robot_actions.NormalizedAction
    assert legacy_actions.FilteredNormalizedAction is robot_actions.FilteredNormalizedAction
    assert legacy_actions.DeploymentAction is robot_actions.DeploymentAction
    assert legacy_actions.ActionSafetyFilterConfig is robot_actions.ActionSafetyFilterConfig
    assert legacy_actions.ActionSafetyFilter is robot_actions.ActionSafetyFilter
    assert legacy_actions.ActionChunkBuffer is robot_actions.ActionChunkBuffer
    assert legacy_actions.normalize_policy_output is robot_actions.normalize_policy_output
    assert legacy_canonical.CanonicalDeployActionPipeline is (
        robot_canonical.CanonicalDeployActionPipeline
    )
    assert legacy_canonical.CanonicalDeployActionResult is (
        robot_canonical.CanonicalDeployActionResult
    )


def test_deploy_schema_pose_and_vector_reference_samples() -> None:
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [0.1, 0.2, 0.3]

    xyzquat = robot_schema.pose_matrix_to_xyzquat(pose)
    assert xyzquat is not None
    np.testing.assert_allclose(xyzquat, [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(robot_schema.xyzquat_to_pose_matrix(xyzquat), pose)

    vector = robot_schema.nan_vector(robot_schema.ACTION_DIM)
    assert vector.shape == (robot_schema.ACTION_DIM,)
    assert vector.dtype == np.float32
    assert np.all(np.isnan(vector))
    np.testing.assert_array_equal(
        robot_schema.vector_or_none([1.0, 2.0], length=2),
        np.asarray([1.0, 2.0], dtype=np.float32),
    )
    assert robot_schema.vector_or_none([1.0, np.nan], length=2) is None


def test_pose_matrix_to_xyzquat_accepts_immutable_runtime_snapshot() -> None:
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [0.1, 0.2, 0.3]
    pose.setflags(write=False)

    xyzquat = robot_schema.pose_matrix_to_xyzquat(pose)

    assert xyzquat is not None
    np.testing.assert_allclose(xyzquat, [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0])


def test_deployment_observation_and_action_normalization_reference_sample() -> None:
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [0.1, 0.2, 0.3]
    observation = robot_observation.build_deployment_observation(
        active_sides=("right",),
        arms_enabled=True,
        hands_enabled=True,
        step_index=7,
        arm_states={"right": {"current_qpos": np.zeros(6), "current_tcp_pose": pose}},
        hand_states={"right": {"joint_pos": np.arange(6, dtype=np.float32) / 10.0}},
        timestamp=12.5,
    )

    assert observation.timestamp == 12.5
    assert observation.active_sides == ("right",)
    assert observation.faults == []
    np.testing.assert_allclose(
        observation.state[robot_schema.STATE_ARM_POSE_SLICES["right"]],
        [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
    )

    normalized = robot_actions.normalize_policy_output(
        np.asarray([0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        observation,
        active_sides=("right",),
        arms_enabled=True,
        hands_enabled=False,
    )

    assert len(normalized) == 1
    np.testing.assert_allclose(
        normalized[0].vector[robot_schema.ACTION_ARM_POSE_SLICES["right"]],
        [0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0],
    )
    assert "arm_action_specs_by_side" in normalized[0].metadata


def test_deploy_schema_active_sides_and_serialization_reference_samples() -> None:
    @dataclass
    class Payload:
        path: Path
        value: np.ndarray

    assert robot_schema.normalize_active_sides(("right", "left", "right")) == ("left", "right")
    assert robot_schema.mapping_get({"a": 1, "b": 2}, ("x", "b")) == 2
    assert robot_schema.to_serializable(
        {
            "payload": Payload(path=Path("run"), value=np.asarray([1, 2], dtype=np.int64)),
            "scalar": np.float32(0.5),
        }
    ) == {"payload": {"path": "run", "value": [1, 2]}, "scalar": 0.5}


def test_robot_motion_deploy_schema_does_not_import_teleop_runtime() -> None:
    source = inspect.getsource(robot_schema)
    assert "from teleop_runtime" not in source
    assert "import teleop_runtime" not in source


def test_robot_motion_deploy_pipeline_modules_do_not_import_teleop_runtime() -> None:
    for module in (
        robot_observation,
        robot_actions,
        robot_canonical,
        robot_dispatcher,
        robot_runtime,
    ):
        source = inspect.getsource(module)
        assert "from teleop_runtime" not in source
        assert "import teleop_runtime" not in source


def test_teleop_deploy_defaults_use_robot_motion_common_pipeline() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_policy_source = (repo_root / "teleop_runtime/deploy/run_policy.py").read_text(
        encoding="utf-8"
    )
    assert "from robot_motion.deploy.actions import" in run_policy_source
    assert "from robot_motion.deploy.canonical_action import" in run_policy_source
    assert "from robot_motion.deploy.observation import" in run_policy_source
    assert "from robot_motion.deploy.schema import" in run_policy_source
    assert "from robot_motion.replay.trajectory import QPOS_INIT_SOURCES" in run_policy_source
    assert "from .actions import" not in run_policy_source
    assert "from .canonical_action import" not in run_policy_source
    assert "from .observation import" not in run_policy_source
    assert "from .schema import" not in run_policy_source
    assert "from ..replay.trajectory import QPOS_INIT_SOURCES" not in run_policy_source

    dispatch_source = (repo_root / "teleop_runtime/deploy/dispatch.py").read_text(
        encoding="utf-8"
    )
    assert "from robot_motion.action import split_effector_name" in dispatch_source
    assert "from robot_motion.runtime import TeleopCommand" in dispatch_source
    assert "from ..action import" not in dispatch_source
    assert "from ..schemas import" not in dispatch_source


class _FakeDeploymentObservationSource:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get_observation(self, *, last_action=None, now: float | None = None):
        self.calls.append({"last_action": last_action, "now": now})
        current_pose = np.eye(4, dtype=float)
        current_pose[:3, 3] = [0.10, 0.20, 0.30]
        return robot_observation.build_deployment_observation(
            active_sides=("right",),
            arms_enabled=True,
            hands_enabled=True,
            step_index=len(self.calls) - 1,
            arm_states={
                "right": {
                    "current_qpos": np.zeros(6, dtype=np.float32),
                    "current_tcp_pose": current_pose,
                }
            },
            hand_states={"right": {"joint_pos": np.zeros(6, dtype=np.float32)}},
            last_action=last_action,
            timestamp=now,
        )


class _FakePolicy:
    name = "fake_policy"

    def __init__(self, action: np.ndarray) -> None:
        self.action = np.asarray(action, dtype=np.float32)
        self.observations: list[object] = []

    def predict(self, observation):
        self.observations.append(observation)
        return self.action.copy()


def test_deploy_runtime_steps_source_policy_canonical_pipeline_dispatcher() -> None:
    source = _FakeDeploymentObservationSource()
    action = robot_schema.nan_vector(robot_schema.DEPLOY_ACTION_DIM)
    action[robot_schema.ACTION_ARM_POSE_SLICES["right"]] = np.asarray(
        [0.12, 0.22, 0.32, 0.0, 0.0, 0.0, 1.0],
        dtype=np.float32,
    )
    action[robot_schema.ACTION_HAND_SLICES["right"]] = np.asarray(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        dtype=np.float32,
    )
    policy = _FakePolicy(action)
    pipeline = robot_canonical.CanonicalDeployActionPipeline(
        safety_filter=robot_actions.ActionSafetyFilter(
            robot_actions.ActionSafetyFilterConfig(
                active_sides=("right",),
                arms_enabled=True,
                hands_enabled=True,
                max_arm_xyz_step_m=0.08,
                max_arm_rot_step_deg=20.0,
                max_hand_step=0.0,
            )
        ),
        active_sides=("right",),
        arms_enabled=True,
        hands_enabled=True,
    )
    dispatcher = robot_dispatcher.DryRunCommandDispatcher()
    runtime = robot_runtime.DeployRuntime(
        source=source,
        policy=policy,
        preparer=pipeline,
        dispatcher=dispatcher,
        send_actions=True,
    )
    last_action = robot_schema.nan_vector(robot_schema.DEPLOY_ACTION_DIM)

    result = runtime.step(last_action=last_action, now=123.5)

    assert len(source.calls) == 1
    np.testing.assert_array_equal(source.calls[0]["last_action"], last_action)
    assert source.calls[0]["now"] == 123.5
    assert policy.observations == [result.observation]
    np.testing.assert_array_equal(result.raw_action, action)
    assert result.metadata == {"policy_name": "fake_policy"}
    assert result.faults == ()
    assert result.sent is True
    assert len(dispatcher.commands) == 1
    assert dispatcher.commands[0] is result.command

    prepared = result.prepared_action
    assert prepared.robot_action is not None
    assert prepared.robot_action.action_space == robot_schema.DEPLOY_ACTION_VECTOR_ACTION_SPACE
    assert prepared.robot_action.active_effectors == ("right_arm", "right_hand")
    assert prepared.robot_action.source == "deploy_policy:fake_policy"
    assert result.command is not None
    assert set(result.command.commands) == {"right_arm", "right_hand"}
    assert result.command.metadata["source"] == "deploy_canonical_robot_action"
    assert result.command.metadata["policy_name"] == "fake_policy"
    np.testing.assert_allclose(
        result.command.commands["right_arm"].metadata["target_xyzquat"],
        action[robot_schema.ACTION_ARM_POSE_SLICES["right"]],
    )


def test_deploy_runtime_dry_step_prepares_without_dispatch_when_send_disabled() -> None:
    source = _FakeDeploymentObservationSource()
    action = robot_schema.nan_vector(robot_schema.DEPLOY_ACTION_DIM)
    action[robot_schema.ACTION_ARM_POSE_SLICES["right"]] = np.asarray(
        [0.10, 0.20, 0.30, 0.0, 0.0, 0.0, 1.0],
        dtype=np.float32,
    )
    policy = _FakePolicy(action)
    pipeline = robot_canonical.CanonicalDeployActionPipeline(
        safety_filter=robot_actions.ActionSafetyFilter(
            robot_actions.ActionSafetyFilterConfig(
                active_sides=("right",),
                arms_enabled=True,
                hands_enabled=False,
            )
        ),
        active_sides=("right",),
        arms_enabled=True,
        hands_enabled=False,
    )
    dispatcher = robot_dispatcher.DryRunCommandDispatcher()
    runtime = robot_runtime.DeployRuntime(
        source=source,
        policy=policy,
        preparer=pipeline,
        dispatcher=dispatcher,
        send_actions=False,
    )

    result = runtime.step(now=1.0)

    assert result.sent is False
    assert result.command is not None
    assert dispatcher.commands == []
