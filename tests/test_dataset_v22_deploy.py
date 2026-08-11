from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pytest

from dataset_contracts import (
    STATE_INPUT_MODE_MASKED_NORMALIZED_CONCAT,
    PolicyArtifactContract,
    PolicyCameraInput,
    PolicyCameraInputContract,
    PoseFrameContract,
    PoseSourceProfile,
)
from dataset_contracts.v22 import (
    DatasetLayoutBindingError,
    DiscreteCategory,
    DiscreteTokenSpec,
    EndEffectorSpec,
    PolicyIOLayout,
)
from dataset_contracts.v23 import DEXTELEOP_DATASET_SCHEMA_V23
from deployments.client import (
    DeployPolicyContract,
    ServerMetadataError,
    UnifiedActionChunk,
    deploy_contract_from_policy_artifact,
    validate_server_metadata,
)
from robot_motion.action import EndEffectorSafetyCapability
from robot_motion.deploy import (
    ActionSafetyFilter,
    ActionSafetyFilterConfig,
    CanonicalDeployActionPipeline,
    DeploymentObservation,
    POLICY_CONTRACT_LEGACY_V1,
    POLICY_CONTRACT_V22_NATIVE,
    POLICY_CONTRACT_V22_Q6_BRIDGE,
    RuntimeEndEffectorCapability,
    V22DeploymentAdapter,
    V22DeploymentError,
    resolve_deploy_session_contract,
)
from robot_motion.deploy.schema import (
    ACTION_ARM_POSE_SLICES,
    ACTION_DIM,
    ACTION_HAND_SLICES,
    OBSERVATION_STATE_DIM,
    STATE_ARM_POSE_SLICES,
    STATE_HAND_SLICES,
)
from robot_motion.relative_action import compare_pose_chunks
from robot_motion.runtime_gate import IKActionChunkGate
from robot_motion.replay.trajectory import (
    QPOS_INIT_OBSERVATION_ARM_QPOS,
    ReplayTrajectoryGroup,
    select_replay_qpos_init_targets,
)
from robot_motion.replay.action_frame import replay_action_metadata
from robot_motion.deploy.observation import observation_faults
from teleop_runtime.deploy.replay import (
    DatasetReplayPolicy,
    compare_v22_replay_absolute_relative_chunks,
    load_dataset_replay_policy,
    resolve_replay_parquet_path,
    v22_replay_rows_to_legacy_actions,
)
from teleop_runtime.deploy.run_policy import _debug_replay_execution_exhausted
from teleop_runtime.deploy.policy import LoadedPolicy
from teleop_runtime.deploy import run_policy
from teleop_runtime.collection import run_collect
from teleop_runtime.replay.dataset import ReplayDatasetIndex
from teleop_runtime.replay.executor import build_episode_replay_group


def _pose(x: float) -> np.ndarray:
    return np.asarray([x, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _observation(*, right_mode: str = "open") -> DeploymentObservation:
    state = np.zeros((OBSERVATION_STATE_DIM,), dtype=np.float32)
    state[STATE_ARM_POSE_SLICES["left"]] = _pose(-0.1)
    state[STATE_ARM_POSE_SLICES["right"]] = _pose(0.1)
    return DeploymentObservation(
        timestamp=1.0,
        step_index=2,
        active_sides=("left", "right"),
        arms_enabled=True,
        hands_enabled=True,
        state=state,
        state_valid=np.ones((OBSERVATION_STATE_DIM,), dtype=bool),
        hand_states={"right": {"mode": right_mode}},
    )


def _right_discrete_layout(*, legacy_slots: tuple[int, ...] | None = (0,)) -> PolicyIOLayout:
    mode = DiscreteTokenSpec(
        "mode",
        (
            DiscreteCategory(0, "open", backend_value=0.0),
            DiscreteCategory(1, "close", backend_value=0.9),
        ),
        source_metadata_key="mode",
    )
    return PolicyIOLayout(
        name="left_none_right_discrete_deploy",
        end_effectors={
            "left": EndEffectorSpec.none("left"),
            "right": EndEffectorSpec.discrete(
                "right",
                adapter="two_state_gripper_v1",
                target_tokens=(mode,),
                target_legacy_q6_slot_indices=legacy_slots,
            ),
        },
    )


def _dual_dmtac_continuous_layout() -> PolicyIOLayout:
    return PolicyIOLayout(
        name="dual_dmtac_continuous_deploy",
        end_effectors={
            side: EndEffectorSpec.continuous(
                side,
                adapter="dmtac_gripper_v1",
                target_dim=1,
                target_names=("gripper_close",),
                target_legacy_q6_slot_indices=(0,),
            )
            for side in ("left", "right")
        },
    )


def test_v22_deploy_observation_keeps_discrete_state_token_int64() -> None:
    layout = _right_discrete_layout()
    typed = V22DeploymentAdapter(layout).observation_from_deployment(_observation(right_mode="close"))

    assert typed.state.shape == (14,)
    assert typed.state_discrete.dtype == np.int64
    assert typed.state_discrete.tolist() == [1]
    assert typed.model_input()["observation.state.discrete"].dtype == np.int64


def test_v22_discrete_action_bridges_only_configured_right_end_effector() -> None:
    layout = _right_discrete_layout()
    adapter = V22DeploymentAdapter(layout)
    chunk = UnifiedActionChunk(
        actions=np.stack((_pose(-0.1), _pose(0.1)), axis=0).reshape(1, 14),
        discrete_actions=np.asarray([[1]], dtype=np.int64),
        metadata={"policy_io_layout_hash": layout.layout_hash},
    )

    actions = adapter.to_legacy_normalized_actions(chunk)

    assert len(actions) == 1
    assert actions[0].metadata["end_effector_enabled_sides"] == ["right"]
    np.testing.assert_allclose(
        actions[0].vector[ACTION_HAND_SLICES["right"]],
        [0.9, 0.0, 0.0, 0.0, 0.0, 0.0],
    )

    observation = _observation()
    pipeline = CanonicalDeployActionPipeline(
        safety_filter=ActionSafetyFilter(
            ActionSafetyFilterConfig(
                active_sides=("left", "right"),
                arms_enabled=True,
                hands_enabled=True,
            )
        ),
        active_sides=("left", "right"),
        arms_enabled=True,
        hands_enabled=True,
    )
    prepared = pipeline.prepare(actions[0], observation, policy_name="v22-test")

    assert not prepared.faults
    assert prepared.command is not None
    assert "left_hand" not in prepared.command.commands
    assert "right_hand" in prepared.command.commands


def test_v22_dual_continuous_dmtac_bridge_selects_end_effector_components() -> None:
    layout = _dual_dmtac_continuous_layout()
    float_action = np.concatenate((_pose(-0.1), _pose(0.1), [0.2, 0.8])).astype(np.float32)

    actions = V22DeploymentAdapter(layout).to_legacy_normalized_actions(
        {
            "action": float_action,
            "policy_io_layout_hash": layout.layout_hash,
        }
    )

    assert len(actions) == 1
    np.testing.assert_allclose(
        actions[0].vector[ACTION_HAND_SLICES["left"]],
        [0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        actions[0].vector[ACTION_HAND_SLICES["right"]],
        [0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
    )


def test_v22_dual_continuous_dmtac_native_selects_end_effector_components() -> None:
    layout = _dual_dmtac_continuous_layout()
    float_action = np.concatenate((_pose(-0.1), _pose(0.1), [0.2, 0.8])).astype(np.float32)

    actions = V22DeploymentAdapter(layout).to_native_robot_actions(
        {
            "action": float_action,
            "policy_io_layout_hash": layout.layout_hash,
        }
    )

    assert len(actions) == 1
    left = actions[0].end_effector_targets_by_side["left"]
    right = actions[0].end_effector_targets_by_side["right"]
    assert left.names == ("gripper_close",)
    assert right.names == ("gripper_close",)
    np.testing.assert_allclose(left.values, [0.2])
    np.testing.assert_allclose(right.values, [0.8])


@pytest.mark.optional
def test_gui_episode_replay_group_bridges_v23_umi_16d_action_to_canonical_26d(
    tmp_path: Path,
) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    layout = _dual_dmtac_continuous_layout()
    source = PoseSourceProfile(
        source_id="umi_dual_wrist_test_v1",
        source_family="real_robot",
        source_description="dual-wrist UMI replay fixture",
        topology="dual_arm",
        raw_absolute_frame_by_side={
            "left": "left_rm65_base",
            "right": "right_rm65_base",
        },
    )
    pose_contract = PoseFrameContract.v23_default_desk(source)
    root = tmp_path / "umi_v23"
    parquet_path = root / "data" / "chunk-000" / "episode_000000.parquet"
    parquet_path.parent.mkdir(parents=True)
    (root / "meta").mkdir(parents=True)
    dataset_info = {
        "dexteleop_dataset_schema": DEXTELEOP_DATASET_SCHEMA_V23,
        "nominal_fps": 15.0,
        **layout.binding_metadata(),
        "pose_frame_contract": pose_contract.to_dict(),
        "pose_storage_frame": pose_contract.pose_storage_frame,
    }
    (root / "meta" / "info.json").write_text(
        json.dumps(dataset_info), encoding="utf-8"
    )
    (root / "meta" / "episodes.jsonl").write_text(
        json.dumps(
            {
                "episode_index": 0,
                "episode_id": "episode_000000",
                "frame_count": 1,
                "parquet_path": "data/chunk-000/episode_000000.parquet",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    compact_action = np.concatenate(
        (_pose(-0.1), _pose(0.1), np.asarray([0.25, 0.75], dtype=np.float32))
    ).astype(np.float32)
    assert compact_action.shape == (16,)
    pq.write_table(
        pyarrow.Table.from_pylist(
            [
                {
                    "action": compact_action.tolist(),
                    "policy_io_layout_hash": layout.layout_hash,
                    "timestamp": 0.0,
                    "control_hz": 15.0,
                }
            ]
        ),
        parquet_path,
    )

    episode = ReplayDatasetIndex(root).load_episode(0)
    group = build_episode_replay_group(
        episode,
        active_sides=("left", "right"),
    )

    assert group.actions.shape == (1, ACTION_DIM)
    np.testing.assert_allclose(
        group.actions[0, ACTION_HAND_SLICES["left"]],
        [0.25, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        group.actions[0, ACTION_HAND_SLICES["right"]],
        [0.75, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    assert group.metadata["typed_action_contract"] is True
    action_metadata = group.metadata["action_metadata"]
    assert action_metadata["end_effector_enabled_sides"] == ["left", "right"]
    assert action_metadata["hand_active_mask_by_side"] == {
        "left": [True, False, False, False, False, False],
        "right": [True, False, False, False, False, False],
    }
    for side in ("left", "right"):
        spec = action_metadata["hand_action_specs_by_side"][side]
        assert spec["hand_model"] == "dmtac_gripper_v1"
        assert spec["slot_map"] == {"gripper_close": 0}
        assert spec["invalid_slot_value"] == "zero_padding"
    replay_metadata = replay_action_metadata(
        group, active_sides=("left", "right")
    )
    assert replay_metadata["hand_action_specs_by_side"] == action_metadata[
        "hand_action_specs_by_side"
    ]


def test_v22_q6_bridge_requires_explicit_backend_slot_mapping() -> None:
    layout = _right_discrete_layout(legacy_slots=None)
    adapter = V22DeploymentAdapter(layout)
    chunk = {
        "action": np.stack((_pose(-0.1), _pose(0.1)), axis=0).reshape(1, 14),
        "action.discrete": np.asarray([[0]], dtype=np.int64),
        "policy_io_layout_hash": layout.layout_hash,
    }

    with pytest.raises(V22DeploymentError, match="legacy_q6_slot_indices"):
        adapter.to_legacy_normalized_actions(chunk)


def test_v22_native_adapter_preserves_discrete_tokens_without_q6_projection() -> None:
    layout = _right_discrete_layout(legacy_slots=None)
    adapter = V22DeploymentAdapter(layout)
    chunk = {
        "action": np.stack((_pose(-0.1), _pose(0.1)), axis=0).reshape(1, 14),
        "action.discrete": np.asarray([[1]], dtype=np.int64),
        "policy_io_layout_hash": layout.layout_hash,
    }

    actions = adapter.to_native_robot_actions(chunk)

    assert len(actions) == 1
    target = actions[0].end_effector_targets_by_side["right"]
    assert target.value_type == "discrete"
    assert target.values.dtype == np.int64
    assert target.values.tolist() == [1]
    assert target.adapter == "two_state_gripper_v1"


def test_v22_remote_contract_binds_layout_hash_and_token_width() -> None:
    layout = _right_discrete_layout()
    metadata = {
        "local_robot": {"output_action_dim": 14, "output_action_discrete_dim": 1},
        "dataset_adapter": {"policy_io_layout_hash": layout.layout_hash},
    }
    result = validate_server_metadata(
        metadata,
        DeployPolicyContract(
            action_dim=14,
            action_discrete_dim=1,
            policy_io_layout_hash=layout.layout_hash,
            hand_layout=None,
        ),
    )

    assert result.ok
    assert result.server_summary["policy_io_layout_hash"] == layout.layout_hash
    with pytest.raises(ServerMetadataError, match="policy_io_layout_hash_mismatch"):
        validate_server_metadata(
            metadata,
            DeployPolicyContract(
                action_dim=14,
                action_discrete_dim=1,
                policy_io_layout_hash="sha256:other",
                hand_layout=None,
            ),
        )


def test_v22_replay_adapter_preserves_side_specific_end_effector_metadata(tmp_path) -> None:
    layout = _right_discrete_layout()
    rows = [
        {
            "action": np.concatenate((_pose(-0.1), _pose(0.1))).tolist(),
            "action.discrete": [1],
            "policy_io_layout_hash": layout.layout_hash,
        }
    ]

    actions, action_metadata = v22_replay_rows_to_legacy_actions(rows, layout)
    policy = DatasetReplayPolicy(
        actions=actions,
        source_path=tmp_path / "episode.parquet",
        action_metadata=action_metadata,
    )
    predicted = policy.predict(_observation())

    assert actions.shape == (1, 26)
    assert action_metadata["end_effector_enabled_sides"] == ["right"]
    assert predicted["action_chunk"][0].metadata["end_effector_enabled_sides"] == ["right"]


def test_replay_prediction_transaction_rolls_back_chunk_discarded_by_ik_recovery(
    tmp_path,
) -> None:
    actions = np.zeros((8, ACTION_DIM), dtype=np.float32)
    replay = DatasetReplayPolicy(
        actions=actions,
        source_path=tmp_path / "episode.parquet",
        chunk_size=4,
    )
    policy = LoadedPolicy(policy=replay, name="dataset_replay:test")
    gate = IKActionChunkGate(stable_frames_required=1)

    assert gate.update(
        {"right": {"ik_metadata": {"status": "smoothing_required"}}}
    ) == "ik_recovery_started"
    checkpoint = policy.begin_prediction_transaction()
    predicted = policy.predict(_observation())
    assert len(predicted["action_chunk"]) == 4
    assert [
        action.metadata["debug_replay_source_action_index"]
        for action in predicted["action_chunk"]
    ] == [0, 1, 2, 3]
    assert replay.cursor == 4
    assert gate.discards_policy_output
    policy.rollback_prediction_transaction(checkpoint)
    assert replay.cursor == 0
    assert replay.completed_passes == 0
    assert replay.finished is False

    assert gate.update(
        {"right": {"ik_metadata": {"status": "pass"}}}
    ) == "ik_recovery_stable_waiting_fresh_policy"
    checkpoint = policy.begin_prediction_transaction()
    predicted = policy.predict(_observation())
    assert len(predicted["action_chunk"]) == 4
    assert [
        action.metadata["debug_replay_source_action_index"]
        for action in predicted["action_chunk"]
    ] == [0, 1, 2, 3]
    policy.commit_prediction_transaction(checkpoint)
    gate.mark_policy_output_accepted()
    assert replay.cursor == 4
    assert not gate.blocks_action_output


def test_looping_replay_prediction_rollback_restores_pass_and_finished_state(
    tmp_path,
) -> None:
    replay = DatasetReplayPolicy(
        actions=np.zeros((3, ACTION_DIM), dtype=np.float32),
        source_path=tmp_path / "episode.parquet",
        chunk_size=2,
        loop=True,
        cursor=3,
        completed_passes=5,
        finished=True,
    )

    checkpoint = replay.begin_prediction_transaction()
    replay.predict(_observation())
    assert replay.cursor == 2
    assert replay.completed_passes == 6
    assert replay.finished is False

    replay.rollback_prediction_transaction(checkpoint)
    assert replay.cursor == 3
    assert replay.completed_passes == 5
    assert replay.finished is True


def test_finished_non_loop_replay_stops_prediction_only_after_queue_drains(tmp_path) -> None:
    replay = DatasetReplayPolicy(
        actions=np.zeros((2, ACTION_DIM), dtype=np.float32),
        source_path=tmp_path / "episode.parquet",
        chunk_size=2,
    )
    policy = LoadedPolicy(policy=replay, name="dataset_replay:test")

    replay.predict(_observation())
    assert replay.finished is True
    assert not _debug_replay_execution_exhausted(policy, queued_actions=2)
    assert _debug_replay_execution_exhausted(policy, queued_actions=0)

    replay.loop = True
    assert not _debug_replay_execution_exhausted(policy, queued_actions=0)


def test_v22_replay_binds_metadata_to_runtime_arm_only_selection(monkeypatch) -> None:
    layout = _right_discrete_layout()
    actions = np.zeros((1, ACTION_DIM), dtype=np.float32)
    actions[0, ACTION_ARM_POSE_SLICES["left"]] = _pose(-0.1)
    actions[0, ACTION_ARM_POSE_SLICES["right"]] = _pose(0.1)
    action_metadata = {
        "end_effector_enabled_sides": ["right"],
        "hand_action_specs_by_side": {"right": {"active_mask": [True, False, False, False, False, False]}},
        "hand_active_mask_by_side": {"right": [True, False, False, False, False, False]},
    }
    dataset_info = {
        "dexteleop_dataset_schema": "dexteleop_dataset_v2.2",
        **layout.binding_metadata(),
    }
    monkeypatch.setattr(
        "teleop_runtime.deploy.replay.resolve_replay_parquet_path",
        lambda **_: (Path("/tmp/v22_arm_only.parquet"), {"dataset_info": dataset_info}),
    )
    monkeypatch.setattr(
        "teleop_runtime.deploy.replay.load_v22_replay_actions_from_parquet",
        lambda *_, **__: (actions, action_metadata),
    )
    monkeypatch.setattr(
        "teleop_runtime.deploy.replay.infer_replay_action_hz",
        lambda **_: {"hz": 15.0, "source": "test"},
    )

    loaded = load_dataset_replay_policy(
        dataset_root=None,
        parquet_path="/tmp/v22_arm_only.parquet",
        episode_index=0,
        active_sides=("left", "right"),
        end_effector_sides=(),
    )
    predicted = loaded.policy.predict(_observation())
    metadata = predicted["action_chunk"][0].metadata
    assert metadata["end_effector_enabled_sides"] == []
    assert metadata["hand_action_specs_by_side"] == {}
    assert metadata["hand_active_mask_by_side"] == {}
    safety = ActionSafetyFilter(
        ActionSafetyFilterConfig(
            active_sides=("left", "right"),
            arms_enabled=True,
            hands_enabled=False,
            end_effector_sides=(),
        )
    )
    filtered = safety.filter_normalized(predicted["action_chunk"][0], _observation(), policy_name="replay")
    assert filtered.faults == []


def _v23_replay_dataset_info(*, storage_frame: str) -> tuple[dict[str, object], PolicyIOLayout]:
    layout = _right_discrete_layout()
    source = PoseSourceProfile(
        source_id="rm65_real_dual_arm_test_v1",
        source_family="real_robot",
        source_description="test fixture",
        topology="dual_arm",
        raw_absolute_frame_by_side={
            "left": "left_rm65_base",
            "right": "right_rm65_base",
        },
    )
    contract = (
        PoseFrameContract.v23_forced_raw(source)
        if storage_frame == "raw_absolute"
        else PoseFrameContract.v23_default_desk(source)
    )
    return (
        {
            "dexteleop_dataset_schema": DEXTELEOP_DATASET_SCHEMA_V23,
            **layout.binding_metadata(),
            "pose_frame_contract": contract.to_dict(),
            "pose_storage_frame": contract.pose_storage_frame,
        },
        layout,
    )


def test_v23_raw_replay_uses_typed_layout_and_preserves_pose_contract(monkeypatch) -> None:
    dataset_info, layout = _v23_replay_dataset_info(storage_frame="raw_absolute")
    actions = np.zeros((1, ACTION_DIM), dtype=np.float32)
    actions[0, ACTION_ARM_POSE_SLICES["left"]] = _pose(-0.1)
    actions[0, ACTION_ARM_POSE_SLICES["right"]] = _pose(0.1)
    monkeypatch.setattr(
        "teleop_runtime.deploy.replay.resolve_replay_parquet_path",
        lambda **_: (Path("/tmp/v23_raw.parquet"), {"dataset_info": dataset_info}),
    )
    monkeypatch.setattr(
        "teleop_runtime.deploy.replay.load_v22_replay_actions_from_parquet",
        lambda *_, **__: (actions, {}),
    )
    monkeypatch.setattr(
        "teleop_runtime.deploy.replay.infer_replay_action_hz",
        lambda **_: {"hz": 15.0, "source": "test"},
    )

    loaded = load_dataset_replay_policy(
        dataset_root=None,
        parquet_path="/tmp/v23_raw.parquet",
        episode_index=0,
        policy_io_layout=layout,
        end_effector_sides=(),
    )
    predicted = loaded.policy.predict(_observation())
    action_metadata = predicted["action_chunk"][0].metadata
    assert loaded.metadata["policy_io_layout_hash"] == layout.layout_hash
    assert action_metadata["pose_storage_frame"] == "raw_absolute"
    assert action_metadata["pose_source_id"] == "rm65_real_dual_arm_test_v1"


def test_v23_raw_replay_converts_each_side_to_canonical_desk(monkeypatch) -> None:
    layout = _right_discrete_layout()
    left_transform = np.eye(4)
    left_transform[0, 3] = 1.0
    right_transform = np.eye(4)
    right_transform[1, 3] = -0.5
    source = PoseSourceProfile(
        source_id="rm65_real_dual_arm_nonidentity_test_v1",
        source_family="real_robot",
        source_description="test fixture",
        topology="dual_arm",
        raw_to_desk_transform_by_side={
            "left": left_transform,
            "right": right_transform,
        },
    )
    contract = PoseFrameContract.v23_forced_raw(source)
    dataset_info = {
        "dexteleop_dataset_schema": DEXTELEOP_DATASET_SCHEMA_V23,
        **layout.binding_metadata(),
        "pose_frame_contract": contract.to_dict(),
        "pose_storage_frame": contract.pose_storage_frame,
    }
    actions = np.zeros((1, ACTION_DIM), dtype=np.float32)
    actions[0, ACTION_ARM_POSE_SLICES["left"]] = _pose(-0.1)
    actions[0, ACTION_ARM_POSE_SLICES["right"]] = _pose(0.1)
    monkeypatch.setattr(
        "teleop_runtime.deploy.replay.resolve_replay_parquet_path",
        lambda **_: (Path("/tmp/v23_raw_nonidentity.parquet"), {"dataset_info": dataset_info}),
    )
    monkeypatch.setattr(
        "teleop_runtime.deploy.replay.load_v22_replay_actions_from_parquet",
        lambda *_, **__: (actions, {}),
    )
    monkeypatch.setattr(
        "teleop_runtime.deploy.replay.infer_replay_action_hz",
        lambda **_: {"hz": 15.0, "source": "test"},
    )
    loaded = load_dataset_replay_policy(
        dataset_root=None,
        parquet_path="/tmp/v23_raw_nonidentity.parquet",
        episode_index=0,
        policy_io_layout=layout,
    )
    predicted = loaded.policy.predict(_observation())["action_chunk"][0].vector
    np.testing.assert_allclose(
        predicted[ACTION_ARM_POSE_SLICES["left"]][:3], [0.9, 0.0, 0.2], atol=1e-6
    )
    np.testing.assert_allclose(
        predicted[ACTION_ARM_POSE_SLICES["right"]][:3], [0.1, -0.5, 0.2], atol=1e-6
    )


def test_v23_desk_deploy_replay_emits_canonical_desk_policy_actions(monkeypatch) -> None:
    dataset_info, layout = _v23_replay_dataset_info(storage_frame="desk_absolute")
    actions = np.zeros((1, ACTION_DIM), dtype=np.float32)
    actions[0, ACTION_ARM_POSE_SLICES["left"]] = _pose(-0.1)
    actions[0, ACTION_ARM_POSE_SLICES["right"]] = _pose(0.1)
    monkeypatch.setattr(
        "teleop_runtime.deploy.replay.resolve_replay_parquet_path",
        lambda **_: (Path("/tmp/v23_desk.parquet"), {"dataset_info": dataset_info}),
    )
    monkeypatch.setattr(
        "teleop_runtime.deploy.replay.load_v22_replay_actions_from_parquet",
        lambda *_, **__: (actions, {}),
    )
    monkeypatch.setattr(
        "teleop_runtime.deploy.replay.infer_replay_action_hz",
        lambda **_: {"hz": 15.0, "source": "test"},
    )
    loaded = load_dataset_replay_policy(
        dataset_root=None,
        parquet_path="/tmp/v23_desk.parquet",
        episode_index=0,
        policy_io_layout=layout,
    )
    assert loaded.metadata["policy_absolute_frame"] == "canonical_desk"
    predicted = loaded.policy.predict(_observation())["action_chunk"][0]
    assert predicted.metadata["pose_action_absolute_frame"] == "canonical_desk"


def test_policy_artifact_accepts_v23_dataset_schema_with_pose_metadata() -> None:
    artifact = PolicyArtifactContract(
        policy_io_layout=PolicyIOLayout(),
        dataset_schema=DEXTELEOP_DATASET_SCHEMA_V23,
        metadata={"pose_representation": {"state": "absolute_desk", "action": "relative"}},
    )
    restored = PolicyArtifactContract.from_dict(artifact.to_dict())
    assert restored.dataset_schema == DEXTELEOP_DATASET_SCHEMA_V23
    assert restored.metadata["pose_representation"]["action"] == "relative"


def test_v22_replay_relative_output_round_trips_through_deployment_anchor(tmp_path) -> None:
    anchor_left = _pose(-0.1)
    anchor_right = _pose(0.1)
    actions = np.zeros((2, ACTION_DIM), dtype=np.float32)
    actions[:, ACTION_ARM_POSE_SLICES["left"]] = np.stack((_pose(-0.05), _pose(-0.02)))
    actions[:, ACTION_ARM_POSE_SLICES["right"]] = np.stack((_pose(0.05), _pose(0.08)))
    policy = DatasetReplayPolicy(
        actions=actions,
        source_path=tmp_path / "episode.parquet",
        chunk_size=2,
        output_pose_mode="relative",
        active_sides=("left", "right"),
    )
    observation = DeploymentObservation(
        timestamp=1.0,
        step_index=0,
        active_sides=("left", "right"),
        arms_enabled=True,
        hands_enabled=False,
        arm_states={
            "left": {"actual_tcp_pose": anchor_left.tolist()},
            "right": {"actual_tcp_pose": anchor_right.tolist()},
        },
    )
    relative_output = policy.predict(observation)["action_chunk"]
    from robot_motion.deploy.pose_adapter import convert_normalized_action_chunk_to_absolute

    decoded = convert_normalized_action_chunk_to_absolute(
        relative_output,
        observation,
        source_mode="relative",
        active_sides=("left", "right"),
    )
    decoded_array = np.stack([item.vector for item in decoded])
    for side in ("left", "right"):
        metrics = compare_pose_chunks(
            actions,
            decoded_array,
            active_sides=(side,),
        )[side]
        assert metrics.max_translation_error_m < 1e-6
        assert metrics.max_rotation_error_rad < 1e-6


@pytest.mark.optional
def test_v22_parquet_absolute_relative_replay_round_trip_is_zero_error(tmp_path) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    layout = PolicyIOLayout()
    rows = [
        {
            "action": np.concatenate((_pose(0.2), _pose(0.3))).tolist(),
            "policy_io_layout_hash": layout.layout_hash,
            "observation.arm.left.actual_tcp_pose": _pose(0.1).tolist(),
            "observation.arm.right.actual_tcp_pose": _pose(0.2).tolist(),
        },
        {
            "action": np.concatenate((_pose(0.25), _pose(0.35))).tolist(),
            "policy_io_layout_hash": layout.layout_hash,
            "observation.arm.left.actual_tcp_pose": _pose(0.11).tolist(),
            "observation.arm.right.actual_tcp_pose": _pose(0.21).tolist(),
        },
    ]
    path = tmp_path / "episode.parquet"
    pq.write_table(pyarrow.Table.from_pylist(rows), path)

    report = compare_v22_replay_absolute_relative_chunks(
        path,
        layout=layout,
        active_sides=("left", "right"),
        chunk_size=2,
    )
    assert len(report["reports"]) == 1
    for metrics in report["reports"][0]["metrics_by_side"].values():
        assert metrics.max_translation_error_m < 1e-6
        assert metrics.max_rotation_error_rad < 1e-6


def test_deploy_cli_allows_right_end_effector_with_two_fixed_arm_sides() -> None:
    parser = argparse.ArgumentParser()
    run_policy.add_args(parser)
    args = parser.parse_args(["--enable-hands", "--end-effector-sides", "right"])

    active_sides = run_policy.validate_args(args)

    assert active_sides == ("left", "right")
    assert args.end_effector_sides == ("right",)


def test_deploy_cli_resolves_heterogeneous_end_effector_backends_per_side() -> None:
    parser = argparse.ArgumentParser()
    run_policy.add_args(parser)
    args = parser.parse_args(
        [
            "--enable-hands",
            "--hand-backend",
            "inspire",
            "--right-hand-backend",
            "dmtac-gripper",
        ]
    )

    run_policy.validate_args(args)

    assert run_policy._hand_backend_mode_for_side(args, "left") == "inspire"
    assert run_policy._hand_backend_mode_for_side(args, "right") == "dmtac-gripper"


def test_collection_resolves_heterogeneous_end_effector_backends_per_side() -> None:
    args = argparse.Namespace(
        hand_backend="inspire",
        left_hand_backend="none",
        right_hand_backend="dmtac-gripper",
    )

    assert run_collect._hand_backend_mode_for_side(args, "left") == "none"
    assert run_collect._hand_backend_mode_for_side(args, "right") == "dmtac-gripper"


def test_side_specific_observation_faults_do_not_require_an_absent_left_end_effector() -> None:
    valid = np.zeros((OBSERVATION_STATE_DIM,), dtype=bool)
    valid[STATE_ARM_POSE_SLICES["left"]] = True
    valid[STATE_ARM_POSE_SLICES["right"]] = True
    valid[STATE_HAND_SLICES["right"]] = True

    faults = observation_faults(
        active_sides=("left", "right"),
        arms_enabled=False,
        hands_enabled=True,
        end_effector_sides=("right",),
        state_valid=valid,
        image_timestamps={},
        require_images=False,
        arm_states={},
        hand_states={"right": {"joint_pos": np.zeros((6,), dtype=np.float32)}},
    )

    assert not any(fault.startswith("left_hand") for fault in faults)


def test_policy_artifact_round_trip_derives_typed_dimensions_from_layout() -> None:
    layout = _right_discrete_layout()
    artifact = PolicyArtifactContract(
        policy_io_layout=layout,
        model_id="openpi-test",
        checkpoint_step=42,
        source_datasets=("org/task-v22",),
    )

    restored = PolicyArtifactContract.from_dict(artifact.to_dict())

    assert restored.layout_hash == layout.layout_hash
    assert [(field.dim, field.dtype) for field in restored.inputs] == [
        (14, "float32"),
        (1, "int64"),
    ]
    assert [(field.dim, field.dtype) for field in restored.outputs] == [
        (14, "float32"),
        (1, "int64"),
    ]


def test_policy_artifact_camera_contract_round_trip_binds_wrist_to_model_slot() -> None:
    layout = _right_discrete_layout()
    camera_input = PolicyCameraInputContract(
        inputs=(
            PolicyCameraInput(
                source_key="left_wrist",
                model_input_key="observation.images.base_0_rgb",
                server_camera_slot="base_0_rgb",
            ),
        ),
    )
    artifact = PolicyArtifactContract(policy_io_layout=layout, camera_input=camera_input)

    restored = PolicyArtifactContract.from_dict(artifact.to_dict())
    contract = deploy_contract_from_policy_artifact(restored)

    assert restored.camera_input == camera_input
    assert restored.camera_input.source_keys == ("left_wrist",)
    assert restored.camera_input.image_key_map == {
        "left_wrist": "observation.images.base_0_rgb"
    }
    assert contract.required_camera_slots == ("base_0_rgb",)
    with pytest.raises(ValueError, match="camera_input"):
        deploy_contract_from_policy_artifact(
            restored,
            required_camera_slots=("left_wrist_0_rgb",),
        )


def test_policy_artifact_camera_contract_rejects_duplicate_logical_view() -> None:
    with pytest.raises(DatasetLayoutBindingError, match="source_key values must be unique"):
        PolicyCameraInputContract(
            inputs=(
                PolicyCameraInput("front", "observation.images.first", "base_0_rgb"),
                PolicyCameraInput("front", "observation.images.second", "base_1_rgb"),
            )
        )


def test_masked_state_artifact_doubles_input_and_deployment_appends_all_available_mask() -> None:
    layout = _right_discrete_layout()
    artifact = PolicyArtifactContract(
        policy_io_layout=layout,
        state_input_mode=STATE_INPUT_MODE_MASKED_NORMALIZED_CONCAT,
        state_augmentation={"enabled": True, "normalized_noise_std": 0.02, "dropout_probability": 0.10},
    )

    restored = PolicyArtifactContract.from_dict(artifact.to_dict())
    assert restored.state_input_mode == STATE_INPUT_MODE_MASKED_NORMALIZED_CONCAT
    assert restored.state_augmentation["dropout_probability"] == 0.10
    assert restored.state_input_dim == layout.state_dim * 2
    assert [(field.dim, field.dtype) for field in restored.inputs] == [(28, "float32"), (1, "int64")]

    raw = V22DeploymentAdapter(layout).observation_from_deployment(_observation(right_mode="close"))
    masked = V22DeploymentAdapter(
        layout,
        state_input_mode=STATE_INPUT_MODE_MASKED_NORMALIZED_CONCAT,
    ).observation_from_deployment(_observation(right_mode="close"))
    assert masked.state.shape == (28,)
    np.testing.assert_array_equal(masked.state[:14], raw.state)
    np.testing.assert_array_equal(masked.state[14:], np.ones((14,), dtype=np.float32))
    assert masked.model_input()["state_input_mode"] == STATE_INPUT_MODE_MASKED_NORMALIZED_CONCAT

    contract = deploy_contract_from_policy_artifact(artifact)
    metadata = {
        "deploy_schema_version": "dexteleop_deploy_v2",
        "local_robot": {
            "input_state_key": "state",
            "input_state_dim": 28,
            "input_state_mode": STATE_INPUT_MODE_MASKED_NORMALIZED_CONCAT,
            "input_state_discrete_key": "state_discrete",
            "input_state_discrete_dim": 1,
            "output_action_dim": 14,
            "output_action_discrete_dim": 1,
        },
        "dataset_adapter": {"policy_io_layout_hash": layout.layout_hash},
    }
    assert validate_server_metadata(metadata, contract).ok
    bad = {**metadata, "local_robot": {**metadata["local_robot"], "input_state_mode": "raw_state_v1"}}
    with pytest.raises(ServerMetadataError, match="input_state_mode_mismatch"):
        validate_server_metadata(bad, contract)

    session = resolve_deploy_session_contract(policy_artifact=artifact)
    assert session.state_dim == 28
    assert session.state_input_mode == STATE_INPUT_MODE_MASKED_NORMALIZED_CONCAT


def test_policy_artifact_rejects_renamed_dictionary_field() -> None:
    payload = PolicyArtifactContract(policy_io_layout=_right_discrete_layout()).to_dict()
    payload["inputs"]["renamed"] = payload["inputs"].pop("state")

    with pytest.raises(DatasetLayoutBindingError, match="input key"):
        PolicyArtifactContract.from_dict(payload)


def test_deploy_session_contract_keeps_unmarked_old_model_legacy() -> None:
    contract = resolve_deploy_session_contract(model_metadata={})

    assert contract.policy_contract_mode == POLICY_CONTRACT_LEGACY_V1
    assert contract.action_dim == 26
    assert contract.state_dim == 38
    assert contract.layout_hash is None


def test_deploy_session_contract_selects_explicit_q6_bridge() -> None:
    layout = _right_discrete_layout()
    artifact = PolicyArtifactContract(policy_io_layout=layout)
    contract = resolve_deploy_session_contract(
        policy_artifact=artifact,
        runtime_end_effectors={
            "right": RuntimeEndEffectorCapability(
                side="right",
                adapter="two_state_gripper_v1",
                backend="mock",
            )
        },
        hardware_output=True,
    )

    assert contract.policy_contract_mode == POLICY_CONTRACT_V22_Q6_BRIDGE
    assert contract.action_dim == 14
    assert contract.action_discrete_dim == 1
    assert contract.end_effector_sides == ("right",)


def test_deploy_session_contract_uses_native_for_non_bridge_layout() -> None:
    layout = _right_discrete_layout(legacy_slots=None)
    contract = resolve_deploy_session_contract(
        policy_artifact=PolicyArtifactContract(policy_io_layout=layout),
        hardware_output=False,
    )

    assert contract.policy_contract_mode == POLICY_CONTRACT_V22_NATIVE
    assert contract.warnings == ()


def test_real_native_session_requires_typed_end_effector_safety() -> None:
    layout = _right_discrete_layout(legacy_slots=None)
    artifact = PolicyArtifactContract(policy_io_layout=layout)
    runtime_without_safety = RuntimeEndEffectorCapability(
        side="right",
        adapter="two_state_gripper_v1",
        backend="mock",
    )

    with pytest.raises(
        DatasetLayoutBindingError,
        match="missing end-effector safety capability",
    ):
        resolve_deploy_session_contract(
            policy_artifact=artifact,
            runtime_end_effectors={"right": runtime_without_safety},
            hardware_output=True,
        )

    safety = EndEffectorSafetyCapability(
        adapter="two_state_gripper_v1",
        value_type="discrete",
        names=("mode",),
        units=("token",),
        allowed_tokens_by_component=((0, 1),),
        allowed_transitions_by_component=(
            ((0, 0), (0, 1), (1, 0), (1, 1)),
        ),
    )
    contract = resolve_deploy_session_contract(
        policy_artifact=artifact,
        runtime_end_effectors={
            "right": RuntimeEndEffectorCapability(
                side="right",
                adapter="two_state_gripper_v1",
                backend="mock",
                safety=safety,
            )
        },
        hardware_output=True,
    )

    assert contract.policy_contract_mode == POLICY_CONTRACT_V22_NATIVE
    assert contract.runtime_end_effectors["right"].safety == safety
    assert contract.to_dict()["runtime_end_effectors"]["right"]["safety"] == (
        safety.to_dict()
    )


def test_deploy_session_contract_rejects_real_missing_or_wrong_end_effector() -> None:
    layout = _right_discrete_layout()
    artifact = PolicyArtifactContract(policy_io_layout=layout)

    with pytest.raises(DatasetLayoutBindingError, match="missing configured"):
        resolve_deploy_session_contract(
            policy_artifact=artifact,
            runtime_end_effectors={},
            hardware_output=True,
        )

    arm_only = resolve_deploy_session_contract(
        policy_artifact=artifact,
        runtime_end_effectors={},
        hardware_output=True,
        allow_missing_end_effectors=True,
    )
    assert "hardware_end_effector_omitted_arm_only:right" in arm_only.warnings
    with pytest.raises(DatasetLayoutBindingError, match="differs from layout adapter"):
        resolve_deploy_session_contract(
            policy_artifact=artifact,
            runtime_end_effectors={
                "right": {
                    "side": "right",
                    "adapter": "inspire_hand_v1",
                    "backend": "mock",
                }
            },
            hardware_output=True,
        )


def test_v22_policy_artifact_derives_strict_server_input_and_output_contract() -> None:
    layout = _right_discrete_layout()
    artifact = PolicyArtifactContract(policy_io_layout=layout, model_id="model-v22")
    contract = deploy_contract_from_policy_artifact(artifact)
    metadata = {
        "deploy_schema_version": "dexteleop_deploy_v2",
        "model_id": "model-v22",
        "local_robot": {
            "input_state_key": "state",
            "input_state_dim": 14,
            "input_state_discrete_key": "state_discrete",
            "input_state_discrete_dim": 1,
            "output_action_dim": 14,
            "output_action_discrete_dim": 1,
        },
        "dataset_adapter": {"policy_io_layout_hash": layout.layout_hash},
    }

    result = validate_server_metadata(metadata, contract)

    assert result.ok
    assert result.server_summary["state_dim"] == 14
    bad = {**metadata, "local_robot": {**metadata["local_robot"], "input_state_dim": 38}}
    with pytest.raises(ServerMetadataError, match="input_state_dim_mismatch"):
        validate_server_metadata(bad, contract)


def test_direct_parquet_replay_discovers_enclosing_v22_metadata(tmp_path) -> None:
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        '{"dexteleop_dataset_schema":"dexteleop_dataset_v2.2"}',
        encoding="utf-8",
    )
    parquet = root / "data" / "chunk-000" / "episode_000000.parquet"
    parquet.touch()

    path, metadata = resolve_replay_parquet_path(
        dataset_root=None,
        parquet_path=parquet,
        episode_index=0,
    )

    assert path == parquet.resolve()
    assert metadata["source"] == "direct_parquet_with_dataset_meta"
    assert metadata["dataset_info"]["dexteleop_dataset_schema"] == "dexteleop_dataset_v2.2"


def test_qpos_init_auto_uses_v22_independent_arm_qpos_audit_column() -> None:
    qpos = np.arange(12, dtype=np.float32)
    group = ReplayTrajectoryGroup(
        key="v",
        label="v22",
        source_path="episode.parquet",
        source_episode_index=0,
        start_index=0,
        end_index=1,
        frame_count=1,
        action_hz=1.0,
        actions=np.zeros((1, 26), dtype=np.float32),
        arm_observation_qpos_audit={
            "left": qpos[:6].reshape(1, 6),
            "right": qpos[6:].reshape(1, 6),
        },
    )

    targets, sources = select_replay_qpos_init_targets(group, ("left", "right"))

    np.testing.assert_allclose(targets["left"], qpos[:6])
    np.testing.assert_allclose(targets["right"], qpos[6:])
    assert sources == {
        "left": QPOS_INIT_OBSERVATION_ARM_QPOS,
        "right": QPOS_INIT_OBSERVATION_ARM_QPOS,
    }
