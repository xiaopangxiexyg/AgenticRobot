from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import inspect
import sys
from types import MappingProxyType, ModuleType

import numpy as np
import pytest

import deployments.client.policy as common_policy
import robot_motion.deploy.actions as robot_actions
import robot_motion.deploy.canonical_action as robot_canonical
import robot_motion.deploy.dispatcher as robot_dispatcher
import robot_motion.deploy.observation as robot_observation
import robot_motion.deploy.pose_adapter as robot_pose_adapter
import robot_motion.deploy.runtime as robot_runtime
import robot_motion.deploy.schema as robot_schema
import teleop_runtime.deploy.actions as legacy_actions
import teleop_runtime.deploy.canonical_action as legacy_canonical
import teleop_runtime.deploy.observation as legacy_observation
import teleop_runtime.deploy.schema as legacy_schema
import teleop_runtime.deploy.policy as legacy_policy
import teleop_runtime.deploy.run_policy as run_policy
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


def test_action_chunk_execution_steps_reject_fractional_direct_api_values() -> None:
    from robot_motion.deploy import normalize_action_chunk_execution_steps

    assert normalize_action_chunk_execution_steps(None) is None
    assert normalize_action_chunk_execution_steps(2) == 2
    assert normalize_action_chunk_execution_steps(2.0) == 2
    with pytest.raises(ValueError, match="positive integer"):
        normalize_action_chunk_execution_steps(2.5)
    with pytest.raises(ValueError, match="positive integer"):
        normalize_action_chunk_execution_steps("2.5")


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


def test_deployment_observation_rejects_stale_hand_measurement_timestamp() -> None:
    hand_state = {
        "joint_pos": np.zeros(6, dtype=np.float32),
        "joint_timestamp": 9.6,
        # A newer worker-loop timestamp must not refresh position feedback.
        "telemetry_timestamp": 9.99,
    }

    observation = robot_observation.build_deployment_observation(
        active_sides=("right",),
        arms_enabled=False,
        hands_enabled=True,
        step_index=0,
        arm_states={},
        hand_states={"right": hand_state},
        timestamp=10.0,
        hand_state_max_age_sec=0.2,
    )

    assert "right_hand_state_stale:0.400000" in observation.faults
    assert np.isclose(observation.metadata["hand_state_age_sec_by_side"]["right"], 0.4)


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
            "readonly": MappingProxyType({"nested": np.float64(1.25)}),
        }
    ) == {
        "payload": {"path": "run", "value": [1, 2]},
        "scalar": 0.5,
        "readonly": {"nested": 1.25},
    }


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


def test_policy_loader_is_shared_with_legacy_teleop_import() -> None:
    assert legacy_policy.LoadedPolicy is common_policy.LoadedPolicy
    assert legacy_policy.HoldPolicy is common_policy.HoldPolicy
    assert legacy_policy.load_policy is common_policy.load_policy
    loaded = common_policy.load_policy("hold")
    assert isinstance(loaded.policy, common_policy.HoldPolicy)
    assert loaded.name == "hold"


def test_policy_file_loader_registers_future_dataclass_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_path = tmp_path / "future_dataclass_policy.py"
    policy_path.write_text(
        """from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class Policy:
    value: int = 7
""",
        encoding="utf-8",
    )
    module_name = "_dexteleop_deployment_policy_future_dataclass_policy"
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    loaded = common_policy.load_policy(f"{policy_path}:Policy")

    assert loaded.policy.value == 7
    assert sys.modules[module_name].Policy is type(loaded.policy)


def test_policy_file_loader_restores_previous_module_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_path = tmp_path / "broken_policy.py"
    policy_path.write_text(
        "raise RuntimeError('policy import failed')\n",
        encoding="utf-8",
    )
    module_name = "_dexteleop_deployment_policy_broken_policy"
    previous = ModuleType(module_name)
    monkeypatch.setitem(sys.modules, module_name, previous)

    with pytest.raises(RuntimeError, match="policy import failed"):
        common_policy.load_policy(f"{policy_path}:Policy")

    assert sys.modules[module_name] is previous


def test_teleop_deploy_defaults_use_robot_motion_common_pipeline() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_policy_source = (repo_root / "teleop_runtime/deploy/run_policy.py").read_text(
        encoding="utf-8"
    )
    assert "from robot_motion.deploy.actions import" in run_policy_source
    assert "from robot_motion.deploy.canonical_action import" in run_policy_source
    assert "from robot_motion.deploy.observation import" in run_policy_source
    assert "from robot_motion.deploy.runtime import (" in run_policy_source
    assert "    PolicyDeployRuntime," in run_policy_source
    assert "from .policy_binding import (" in run_policy_source
    policy_binding_source = (
        repo_root / "teleop_runtime/deploy/policy_binding.py"
    ).read_text(encoding="utf-8")
    assert "resolve_policy_action_pose_mode_from_artifact" in policy_binding_source
    assert "from deployments.client.policy import load_policy" in run_policy_source
    assert "policy_deploy_runtime.predict_and_queue(" in run_policy_source
    assert "policy_deploy_runtime.prepare_next_action(" in run_policy_source
    assert "ActionChunkBuffer" not in run_policy_source
    assert "policy.begin_prediction_transaction" not in run_policy_source
    assert "policy.rollback_prediction_transaction" not in run_policy_source
    assert "def _policy_should_run" not in run_policy_source
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


def test_policy_deploy_runtime_reuses_v23_relative_decode_boundary() -> None:
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [0.10, 0.20, 0.30]
    observation = robot_observation.build_deployment_observation(
        active_sides=("right",),
        arms_enabled=True,
        hands_enabled=False,
        step_index=0,
        arm_states={
            "right": {
                "current_qpos": np.zeros(6, dtype=np.float32),
                "current_tcp_pose": pose,
                "actual_tcp_pose": pose,
            }
        },
        hand_states={},
        timestamp=12.5,
    )
    absolute = robot_schema.nan_vector(robot_schema.DEPLOY_ACTION_DIM)
    absolute[robot_schema.ACTION_ARM_POSE_SLICES["right"]] = np.asarray(
        [0.18, 0.24, 0.36, 0.0, 0.0, 0.0, 1.0],
        dtype=np.float32,
    )
    relative = robot_pose_adapter.convert_normalized_action_chunk_to_relative(
        [robot_actions.NormalizedAction(vector=absolute)],
        observation,
        active_sides=("right",),
    )[0]
    runtime = robot_runtime.PolicyDeployRuntime(
        robot_runtime.PolicyDeployRuntimeConfig(
            active_sides=("right",),
            arms_enabled=True,
            hands_enabled=False,
            policy_action_pose_mode="relative",
        )
    )

    policy_observation = runtime.prepare_policy_observation(observation)
    decoded = runtime.normalize_and_decode(relative.vector, observation)

    assert policy_observation.metadata["policy_pose_action_mode"] == "relative"
    assert len(decoded) == 1
    np.testing.assert_allclose(
        decoded[0].vector[robot_schema.ACTION_ARM_POSE_SLICES["right"]],
        absolute[robot_schema.ACTION_ARM_POSE_SLICES["right"]],
        atol=1e-6,
    )
    assert decoded[0].metadata["pose_anchor_policy"] == "current_actual_tcp_fixed_for_chunk"


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


class _TransactionalPolicy:
    name = "transactional_policy"

    def __init__(self, output=None, error: BaseException | None = None) -> None:
        self.output = output
        self.error = error
        self.events: list[str] = []

    def begin_prediction_transaction(self):
        self.events.append("begin")
        return len(self.events)

    def predict(self, observation):
        del observation
        self.events.append("predict")
        if self.error is not None:
            raise self.error
        if isinstance(self.output, np.ndarray):
            return self.output.copy()
        return self.output

    def commit_prediction_transaction(self, checkpoint):
        del checkpoint
        self.events.append("commit")

    def rollback_prediction_transaction(self, checkpoint):
        del checkpoint
        self.events.append("rollback")


class _FakeActionPreparer:
    def prepare(self, action, observation, *, policy_name=""):
        return {
            "action": action,
            "observation": observation,
            "policy_name": policy_name,
        }


def _right_arm_policy_observation():
    source = _FakeDeploymentObservationSource()
    return source.get_observation(now=10.0)


def _right_arm_policy_action() -> np.ndarray:
    action = robot_schema.nan_vector(robot_schema.DEPLOY_ACTION_DIM)
    action[robot_schema.ACTION_ARM_POSE_SLICES["right"]] = np.asarray(
        [0.12, 0.22, 0.32, 0.0, 0.0, 0.0, 1.0],
        dtype=np.float32,
    )
    return action


def test_policy_deploy_runtime_owns_schedule_transaction_and_chunk_queue() -> None:
    runtime = robot_runtime.PolicyDeployRuntime(
        robot_runtime.PolicyDeployRuntimeConfig(
            active_sides=("right",),
            arms_enabled=True,
            hands_enabled=True,
        )
    )
    policy = _TransactionalPolicy(_right_arm_policy_action())
    observation = _right_arm_policy_observation()

    assert robot_runtime.resolve_action_queue_policy(
        action_queue_policy="auto",
        debug_replay=False,
    ) == "replace"
    assert runtime.should_predict(now=10.0, policy_interval_s=0.1)

    result = runtime.predict_and_queue(
        policy,
        observation,
        now=10.0,
        queue_policy="replace",
    )

    assert result.accepted is True
    assert result.queue_len_before == 0
    assert result.queue_len_after == 1
    assert result.error is None
    assert policy.events == ["begin", "predict", "commit"]
    assert runtime.should_predict(now=10.05, policy_interval_s=0.1) is False
    action_step = runtime.prepare_next_action(
        observation,
        preparer=_FakeActionPreparer(),
        policy_name=policy.name,
    )
    assert action_step.raw_action is not None
    assert action_step.raw_lookahead_action is None
    assert action_step.prepared_action["policy_name"] == policy.name
    assert action_step.queue_len_before == 1
    assert action_step.queue_len_after == 0
    assert runtime.action_queue_length() == 0


def test_sync_position_prediction_waits_for_fresh_observation_without_popping() -> None:
    runtime = robot_runtime.PolicyDeployRuntime(
        robot_runtime.PolicyDeployRuntimeConfig(
            active_sides=("right",),
            arms_enabled=True,
            hands_enabled=True,
        )
    )
    policy = _TransactionalPolicy(_right_arm_policy_action())
    stale_request_observation = _right_arm_policy_observation()

    prediction = runtime.predict_and_queue(
        policy,
        stale_request_observation,
        now=10.0,
        queue_policy="replace",
    )
    defer = run_policy._defer_position_action_after_synchronous_prediction(
        prediction_accepted=prediction.accepted,
        position_executor_enabled=True,
    )

    deferred_step = runtime.prepare_next_action(
        stale_request_observation,
        preparer=_FakeActionPreparer(),
        policy_name=policy.name,
        allow_output=not defer,
    )
    assert deferred_step.raw_action is None
    assert deferred_step.queue_len_before == 1
    assert deferred_step.queue_len_after == 1
    assert runtime.action_queue_pop_count() == 0

    fresh_execution_observation = _right_arm_policy_observation()
    execution_step = runtime.prepare_next_action(
        fresh_execution_observation,
        preparer=_FakeActionPreparer(),
        policy_name=policy.name,
        allow_output=True,
    )
    assert execution_step.raw_action is not None
    assert execution_step.queue_len_before == 1
    assert execution_step.queue_len_after == 0
    assert runtime.action_queue_pop_count() == 1


def test_policy_deploy_runtime_exposes_only_same_chunk_successor_as_lookahead() -> None:
    observation = _right_arm_policy_observation()
    runtime = robot_runtime.PolicyDeployRuntime(
        robot_runtime.PolicyDeployRuntimeConfig(
            active_sides=("right",),
            arms_enabled=True,
            hands_enabled=True,
        )
    )
    first = _right_arm_policy_action()
    second = np.asarray(first, dtype=np.float32).copy()
    second[7] += 0.01
    policy = _TransactionalPolicy(np.stack((first, second), axis=0))

    prediction = runtime.predict_and_queue(policy, observation, now=10.0)
    assert prediction.accepted is True
    assert prediction.queue_len_after == 2

    first_step = runtime.prepare_next_action(
        observation,
        preparer=_FakeActionPreparer(),
        policy_name=policy.name,
    )
    assert first_step.raw_action is not None
    assert first_step.raw_lookahead_action is not None
    assert first_step.raw_action.metadata["deploy_action_chunk_index"] == 0
    assert first_step.raw_lookahead_action.metadata["deploy_action_chunk_index"] == 1
    assert (
        first_step.raw_action.metadata["deploy_action_chunk_id"]
        == first_step.raw_lookahead_action.metadata["deploy_action_chunk_id"]
    )

    second_step = runtime.prepare_next_action(
        observation,
        preparer=_FakeActionPreparer(),
        policy_name=policy.name,
    )
    assert second_step.raw_action is not None
    assert second_step.raw_lookahead_action is None


def test_policy_deploy_runtime_exposes_two_same_chunk_successors_without_pop() -> None:
    observation = _right_arm_policy_observation()
    runtime = robot_runtime.PolicyDeployRuntime(
        robot_runtime.PolicyDeployRuntimeConfig(
            active_sides=("right",),
            arms_enabled=True,
            hands_enabled=True,
        )
    )
    first = _right_arm_policy_action()
    second = np.asarray(first, dtype=np.float32).copy()
    third = np.asarray(first, dtype=np.float32).copy()
    second[7] += 0.01
    third[7] += 0.02
    policy = _TransactionalPolicy(np.stack((first, second, third), axis=0))
    prediction = runtime.predict_and_queue(policy, observation, now=10.0)
    assert prediction.accepted is True

    step = runtime.prepare_next_action(
        observation,
        preparer=_FakeActionPreparer(),
        policy_name=policy.name,
    )

    assert step.raw_action is not None
    assert step.raw_lookahead_action is not None
    assert step.raw_second_lookahead_action is not None
    assert step.raw_lookahead_action.metadata["deploy_action_chunk_index"] == 1
    assert (
        step.raw_second_lookahead_action.metadata["deploy_action_chunk_index"]
        == 2
    )
    assert runtime.action_queue_length() == 2


def test_policy_deploy_runtime_exposes_three_speculative_same_chunk_successors() -> None:
    observation = _right_arm_policy_observation()
    runtime = robot_runtime.PolicyDeployRuntime(
        robot_runtime.PolicyDeployRuntimeConfig(
            active_sides=("right",),
            arms_enabled=True,
            hands_enabled=True,
        )
    )
    first = _right_arm_policy_action()
    actions = []
    for index in range(4):
        action = np.asarray(first, dtype=np.float32).copy()
        action[7] += 0.01 * index
        actions.append(action)
    policy = _TransactionalPolicy(np.stack(actions, axis=0))
    prediction = runtime.predict_and_queue(policy, observation, now=10.0)
    assert prediction.accepted is True

    step = runtime.prepare_next_action(
        observation,
        preparer=_FakeActionPreparer(),
        policy_name=policy.name,
    )

    assert step.raw_action is not None
    assert step.raw_lookahead_action is not None
    assert step.raw_second_lookahead_action is not None
    assert step.raw_third_lookahead_action is not None
    assert step.raw_lookahead_action.metadata["deploy_action_chunk_index"] == 1
    assert step.raw_second_lookahead_action.metadata["deploy_action_chunk_index"] == 2
    assert step.raw_third_lookahead_action.metadata["deploy_action_chunk_index"] == 3
    assert runtime.action_queue_length() == 3


def test_policy_deploy_runtime_executes_only_configured_chunk_prefix() -> None:
    observation = _right_arm_policy_observation()
    runtime = robot_runtime.PolicyDeployRuntime(
        robot_runtime.PolicyDeployRuntimeConfig(
            active_sides=("right",),
            arms_enabled=True,
            hands_enabled=True,
            action_chunk_execution_steps=2,
        )
    )
    first = _right_arm_policy_action()
    second = np.asarray(first, dtype=np.float32).copy()
    second[7] += 0.01
    third = np.asarray(second, dtype=np.float32).copy()
    third[7] += 0.01
    policy = _TransactionalPolicy(np.stack((first, second, third), axis=0))

    prediction = runtime.predict_and_queue(policy, observation, now=10.0)

    assert prediction.accepted is True
    assert prediction.source_chunk_size == 3
    assert prediction.executed_chunk_size == 2
    assert prediction.action_chunk_execution_steps == 2
    assert prediction.queue_len_after == 2
    assert all(
        action.metadata["deploy_action_chunk_source_size"] == 3
        for action in prediction.actions
    )
    assert all(
        action.metadata["deploy_action_chunk_execution_steps"] == 2
        for action in prediction.actions
    )

    first_step = runtime.prepare_next_action(
        observation,
        preparer=_FakeActionPreparer(),
        policy_name=policy.name,
    )
    second_step = runtime.prepare_next_action(
        observation,
        preparer=_FakeActionPreparer(),
        policy_name=policy.name,
    )
    assert first_step.raw_action is not None
    assert second_step.raw_action is not None
    assert runtime.action_queue_length() == 0
    assert runtime.prepare_next_action(
        observation,
        preparer=_FakeActionPreparer(),
        policy_name=policy.name,
    ).raw_action is None


def test_policy_deploy_runtime_rolls_back_empty_and_recovery_discarded_chunks() -> None:
    observation = _right_arm_policy_observation()
    runtime = robot_runtime.PolicyDeployRuntime(
        robot_runtime.PolicyDeployRuntimeConfig(
            active_sides=("right",),
            arms_enabled=True,
            hands_enabled=True,
        )
    )

    empty_policy = _TransactionalPolicy(None)
    empty = runtime.predict_and_queue(empty_policy, observation, now=1.0)
    assert empty.empty is True
    assert empty.faults == ("policy_empty_action",)
    assert empty_policy.events == ["begin", "predict", "rollback"]

    discarded_policy = _TransactionalPolicy(_right_arm_policy_action())
    discarded = runtime.predict_and_queue(
        discarded_policy,
        observation,
        now=2.0,
        discard_policy_output=True,
    )
    assert discarded.discarded_by_gate is True
    assert discarded.accepted is False
    assert discarded_policy.events == ["begin", "predict", "rollback"]
    assert runtime.action_queue_length() == 0
    assert runtime.should_predict(now=2.05, policy_interval_s=0.1) is False


def test_policy_deploy_runtime_clears_queue_and_rolls_back_on_prediction_error() -> None:
    runtime = robot_runtime.PolicyDeployRuntime(
        robot_runtime.PolicyDeployRuntimeConfig(
            active_sides=("right",),
            arms_enabled=True,
            hands_enabled=True,
        )
    )
    policy = _TransactionalPolicy(error=RuntimeError("predict failed"))
    result = runtime.predict_and_queue(
        policy,
        _right_arm_policy_observation(),
        now=3.0,
    )

    assert isinstance(result.error, RuntimeError)
    assert result.rollback_error is None
    assert result.faults == ()
    assert policy.events == ["begin", "predict", "rollback"]
    assert runtime.action_queue_length() == 0


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


def test_common_deploy_runtime_consumes_chunk_prefix_before_next_prediction() -> None:
    source = _FakeDeploymentObservationSource()
    first = _right_arm_policy_action()
    second = np.asarray(first, dtype=np.float32).copy()
    second[7] += 0.01
    third = np.asarray(second, dtype=np.float32).copy()
    third[7] += 0.01
    policy = _FakePolicy(np.stack((first, second, third), axis=0))
    runtime = robot_runtime.DeployRuntime(
        source=source,
        policy=policy,
        preparer=_FakeActionPreparer(),
        dispatcher=robot_dispatcher.DryRunCommandDispatcher(),
        action_chunk_execution_steps=2,
    )

    first_step = runtime.step(now=1.0)
    second_step = runtime.step(now=1.0 + 1.0 / 15.0)

    assert isinstance(first_step.raw_action, robot_actions.NormalizedAction)
    assert isinstance(second_step.raw_action, robot_actions.NormalizedAction)
    assert runtime.action_queue_length() == 0
    assert len(policy.observations) == 1
    assert first_step.metadata["action_chunk_execution_steps"] == 2
