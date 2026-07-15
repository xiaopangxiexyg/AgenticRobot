from __future__ import annotations

import numpy as np
import pytest

from dataset_contracts.v22 import (
    ARM_QPOS_AUDIT_DIM,
    DatasetLayoutBindingError,
    DiscreteCategory,
    DiscreteTokenSpec,
    EndEffectorSpec,
    INVALID_DISCRETE_TOKEN,
    PolicyIOLayout,
    bind_policy_io_layout,
    policy_io_layout_from_metadata,
    validate_policy_io_layout_binding,
)
from dataset_contracts.v21_compat import v21_info_to_v22_info, v21_row_to_v22_row
from teleop_runtime.collection.sample_assembler import CollectorSample
from teleop_runtime.collection.schema_v21 import collector_sample_to_row
from teleop_runtime.collection.schema_v22 import collector_sample_to_v22_row
from teleop_runtime.collection.writer_v21 import LeRobotV21Writer, LeRobotV21WriterConfig, ROW_JOB_KIND
from dataset_interface.policy_io import policy_io_batch_from_rows, policy_io_frame_from_row
from utils.dataset_postprocess.v21_to_v22 import (
    load_v22_layout_json,
    plan_v21_to_v22_conversion,
)


def _pose(offset: float) -> np.ndarray:
    return np.asarray([offset, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def test_v22_layout_has_fixed_dual_arm_pose_core_and_side_specific_continuous_eef() -> None:
    layout = PolicyIOLayout(
        name="left_none_right_inspire",
        end_effectors={
            "left": EndEffectorSpec.none("left"),
            "right": EndEffectorSpec.continuous(
                "right",
                adapter="inspire_tac_v1",
                target_dim=6,
                target_names=("pinky", "ring", "middle", "index", "thumb_yaw", "thumb_flex"),
                target_source_indices=(0, 1, 2, 3, 4, 5),
            ),
        },
    )

    assert layout.action_dim == 20
    assert layout.state_dim == 20
    assert layout.action_discrete_dim == 0
    assert layout.state_discrete_dim == 0
    assert layout.feature_metadata()["observation.arm.qpos"]["shape"] == [ARM_QPOS_AUDIT_DIM]

    action = layout.pack_action(
        arm_pose_by_side={"left": _pose(0.1), "right": _pose(0.2)},
        end_effector_by_side={"right": np.arange(6, dtype=np.float32)},
        strict=True,
    )
    state = layout.pack_observation_state(
        arm_pose_by_side={"left": _pose(0.3), "right": _pose(0.4)},
        end_effector_by_side={"right": np.arange(6, dtype=np.float32) + 10.0},
        strict=True,
    )
    qpos = layout.pack_arm_qpos_audit(
        {"left": np.arange(6), "right": np.arange(6) + 10.0}
    )

    assert action.continuous.shape == (20,)
    assert state.continuous.shape == (20,)
    assert action.discrete.shape == (0,)
    np.testing.assert_allclose(action.continuous[:7], _pose(0.1))
    np.testing.assert_allclose(action.continuous[14:], np.arange(6, dtype=np.float32))
    np.testing.assert_allclose(
        qpos,
        np.concatenate((np.arange(6, dtype=np.float32), np.arange(6, dtype=np.float32) + 10.0)),
    )


def test_v22_discrete_tokens_are_int64_and_not_float_action_slots() -> None:
    open_close = DiscreteTokenSpec(
        name="gripper_mode",
        categories=(
            DiscreteCategory(0, "open", backend_value=0.0),
            DiscreteCategory(1, "close", backend_value=1.0),
        ),
    )
    layout = PolicyIOLayout(
        name="left_continuous_right_discrete",
        end_effectors={
            "left": EndEffectorSpec.continuous(
                "left",
                adapter="dmtac_gripper_continuous_v1",
                target_dim=1,
                target_source_indices=(0,),
            ),
            "right": EndEffectorSpec.discrete(
                "right",
                adapter="two_state_gripper_v1",
                target_tokens=(open_close,),
            ),
        },
    )

    assert layout.action_dim == 15
    assert layout.state_dim == 15
    assert layout.action_discrete_dim == 1
    assert layout.state_discrete_dim == 1
    packed = layout.pack_action(
        arm_pose_by_side={"left": _pose(0.0), "right": _pose(0.0)},
        end_effector_by_side={"left": [0.25], "right": "close"},
        strict=True,
    )

    assert packed.continuous.dtype == np.float32
    assert packed.discrete.dtype == np.int64
    assert packed.discrete.tolist() == [1]
    np.testing.assert_allclose(packed.continuous[14:], [0.25])
    missing = layout.pack_action(
        arm_pose_by_side={"left": _pose(0.0), "right": _pose(0.0)},
        end_effector_by_side={"left": [0.25]},
    )
    assert missing.discrete.tolist() == [INVALID_DISCRETE_TOKEN]


def test_v22_layout_hash_binds_dataset_checkpoint_and_runtime_metadata() -> None:
    layout = PolicyIOLayout.legacy_v21_q6()
    metadata = bind_policy_io_layout({"dataset_name": "demo"}, layout)

    restored = policy_io_layout_from_metadata(metadata, allow_legacy_v21=False)
    assert restored.layout_hash == layout.layout_hash
    validate_policy_io_layout_binding(layout, metadata, context="dataset")
    with pytest.raises(DatasetLayoutBindingError, match="layout mismatch"):
        validate_policy_io_layout_binding("sha256:not-the-layout", metadata, context="checkpoint")


def test_legacy_v21_is_explicit_compatibility_layout_not_an_untyped_fallback() -> None:
    layout = PolicyIOLayout.legacy_v21_q6()

    assert layout.name == "legacy_v21_dual_q6"
    assert layout.action_dim == 26
    assert layout.state_dim == 26
    assert layout.end_effector("left").target.dim == 6


def _collector_sample_for_v22() -> CollectorSample:
    left_pose = _pose(0.1)
    right_pose = _pose(0.2)
    action = np.full((26,), np.nan, dtype=np.float32)
    action[0:7] = left_pose
    action[7:14] = right_pose
    action[14:20] = np.arange(6, dtype=np.float32) + 20.0
    action[20:26] = np.arange(6, dtype=np.float32) + 30.0
    state = np.full((38,), np.nan, dtype=np.float32)
    state[0:6] = np.arange(6, dtype=np.float32)
    state[6:13] = left_pose + np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    state[13:19] = np.arange(6, dtype=np.float32) + 40.0
    state[19:25] = np.arange(6, dtype=np.float32) + 10.0
    state[25:32] = right_pose
    state[32:38] = np.arange(6, dtype=np.float32) + 50.0
    return CollectorSample(
        index=0,
        episode_index=0,
        frame_index=0,
        timestamp=1.0,
        session_state="RUNNING",
        control_hz=15.0,
        action=action,
        observation_state=state,
        observation_last_action=action.copy(),
        arm_state_source="unit_test",
        active_sides=("left", "right"),
        hand_states={
            "left": {"joint_pos": np.arange(6, dtype=np.float32) + 40.0},
            "right": {"joint_pos": np.arange(6, dtype=np.float32) + 50.0},
        },
    )


def test_v22_row_adapter_rewrites_dense_model_fields_but_preserves_arm_qpos_audit() -> None:
    layout = PolicyIOLayout(
        name="left_none_right_q6",
        end_effectors={
            "left": EndEffectorSpec.none("left"),
            "right": EndEffectorSpec.continuous(
                "right",
                adapter="inspire_tac_v1",
                target_dim=6,
                target_source_indices=(0, 1, 2, 3, 4, 5),
            ),
        },
    )

    row, _ = collector_sample_to_v22_row(_collector_sample_for_v22(), layout=layout)

    assert row["dexteleop.dataset_schema"] == "dexteleop_dataset_v2.2"
    assert row["policy_io_layout_hash"] == layout.layout_hash
    assert len(row["action"]) == 20
    assert len(row["observation.state"]) == 20
    assert row["action.end_effector.right.target"] == [30.0, 31.0, 32.0, 33.0, 34.0, 35.0]
    assert row["observation.end_effector.right.state"] == [50.0, 51.0, 52.0, 53.0, 54.0, 55.0]
    assert row["observation.arm.qpos"] == [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
    ]


def test_v22_writer_binds_layout_to_info_episode_and_row(tmp_path) -> None:
    layout = PolicyIOLayout(
        name="left_gripper_right_none",
        end_effectors={
            "left": EndEffectorSpec.continuous(
                "left",
                adapter="dmtac_gripper_v1",
                target_dim=1,
                target_source_indices=(0,),
                state_source_indices=(0,),
            ),
            "right": EndEffectorSpec.none("right"),
        },
    )
    writer = LeRobotV21Writer(
        LeRobotV21WriterConfig(
            root_dir=tmp_path,
            write_video=False,
            write_parquet=False,
            policy_io_layout=layout,
        )
    )

    info = __import__("json").loads((tmp_path / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["dexteleop_dataset_schema"] == "dexteleop_dataset_v2.2"
    assert info["policy_io_layout_hash"] == layout.layout_hash
    assert info["features"]["action"]["shape"] == [15]

    writer.start_episode(episode_index=0)
    row_job = next(job for job in writer.plan_sample_jobs(_collector_sample_for_v22()) if job.kind == ROW_JOB_KIND)
    row = row_job.payload["row"]
    assert len(row["action"]) == 15
    assert row["action.end_effector.left.target"] == [20.0]
    writer.handle_job(row_job)
    assert writer._episodes[0].metadata["policy_io_layout_hash"] == layout.layout_hash


def test_v22_writer_refuses_to_append_with_a_different_layout_hash(tmp_path) -> None:
    first = PolicyIOLayout(
        name="left_gripper",
        end_effectors={
            "left": EndEffectorSpec.continuous(
                "left", adapter="dmtac_gripper_v1", target_dim=1, target_source_indices=(0,)
            ),
            "right": EndEffectorSpec.none("right"),
        },
    )
    other = PolicyIOLayout(
        name="right_gripper",
        end_effectors={
            "left": EndEffectorSpec.none("left"),
            "right": EndEffectorSpec.continuous(
                "right", adapter="dmtac_gripper_v1", target_dim=1, target_source_indices=(0,)
            ),
        },
    )
    LeRobotV21Writer(
        LeRobotV21WriterConfig(
            root_dir=tmp_path,
            write_video=False,
            write_parquet=False,
            policy_io_layout=first,
        )
    )

    with pytest.raises(ValueError, match="policy_io_layout_hash differs"):
        LeRobotV21Writer(
            LeRobotV21WriterConfig(
                root_dir=tmp_path,
                write_video=False,
                write_parquet=False,
                policy_io_layout=other,
            )
        )


def test_v21_to_v22_conversion_requires_explicit_layout_and_preserves_provenance() -> None:
    legacy_row, _ = collector_sample_to_row(_collector_sample_for_v22())
    layout = PolicyIOLayout(
        name="legacy_q6_to_left_gripper",
        end_effectors={
            "left": EndEffectorSpec.continuous(
                "left",
                adapter="dmtac_gripper_v1",
                target_dim=1,
                target_source_indices=(0,),
                state_source_indices=(0,),
            ),
            "right": EndEffectorSpec.none("right"),
        },
    )

    converted = v21_row_to_v22_row(legacy_row, layout)
    info = v21_info_to_v22_info({"format": "lerobot_v2.1_style"}, layout)

    assert len(converted["action"]) == 15
    assert converted["action.end_effector.left.target"] == [20.0]
    assert converted["observation.arm.qpos"][:6] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert converted["action.left_hand.joint_target"] == [20.0, 21.0, 22.0, 23.0, 24.0, 25.0]
    assert info["dexteleop_dataset_schema"] == "dexteleop_dataset_v2.2"
    assert info["policy_io_layout_hash"] == layout.layout_hash


def test_training_reader_keeps_discrete_tokens_out_of_float_state_and_action() -> None:
    token = DiscreteTokenSpec(
        "mode",
        (DiscreteCategory(0, "open"), DiscreteCategory(1, "close")),
    )
    layout = PolicyIOLayout(
        name="typed_training_batch",
        end_effectors={
            "left": EndEffectorSpec.none("left"),
            "right": EndEffectorSpec.discrete(
                "right",
                adapter="two_state_gripper_v1",
                target_tokens=(token,),
            ),
        },
    )
    row, _ = collector_sample_to_v22_row(_collector_sample_for_v22(), layout=layout)
    # The old capture has no discrete command metadata, so make the explicit
    # recorded token visible exactly as a v2.2 writer/adapter would.
    row["action.discrete"] = [1]
    row["observation.state.discrete"] = [0]

    frame = policy_io_frame_from_row(row, layout)

    assert frame.action.dtype == np.float32
    assert frame.action_discrete.dtype == np.int64
    assert frame.action_discrete.tolist() == [1]
    assert frame.observation_state_discrete.tolist() == [0]
    batch = policy_io_batch_from_rows([row, row], layout)
    assert batch.action.shape == (2, 14)
    assert batch.action_discrete.dtype == np.int64
    assert batch.model_dict()["observation.state.discrete"].shape == (2, 1)


def test_v21_to_v22_copy_on_write_plan_and_layout_file(tmp_path) -> None:
    source = tmp_path / "legacy"
    (source / "meta").mkdir(parents=True)
    (source / "meta" / "info.json").write_text(
        '{"format":"lerobot_v2.1_style","frame_count":12}',
        encoding="utf-8",
    )
    (source / "meta" / "episodes.jsonl").write_text("", encoding="utf-8")
    layout = PolicyIOLayout.legacy_v21_q6()
    layout_file = tmp_path / "layout.json"
    layout_file.write_text(
        __import__("json").dumps(
            {
                "policy_io_layout": layout.to_dict(),
                "policy_io_layout_hash": layout.layout_hash,
            }
        ),
        encoding="utf-8",
    )

    restored = load_v22_layout_json(layout_file)
    plan = plan_v21_to_v22_conversion(source, tmp_path / "converted", restored)

    assert restored.layout_hash == layout.layout_hash
    assert plan.episode_count == 0
    assert plan.frame_count == 12
    assert plan.to_dict()["source_mutated"] is False
    assert not (tmp_path / "converted").exists()


def test_v21_to_v22_preflight_rejects_missing_parquet_before_output_creation(tmp_path) -> None:
    source = tmp_path / "legacy"
    (source / "meta").mkdir(parents=True)
    (source / "meta" / "info.json").write_text(
        '{"format":"lerobot_v2.1_style","episode_count":1}',
        encoding="utf-8",
    )
    (source / "meta" / "episodes.jsonl").write_text(
        '{"episode_index":0,"parquet_path":"data/chunk-000/episode_000000.parquet"}\n',
        encoding="utf-8",
    )
    output = tmp_path / "converted"

    with pytest.raises(ValueError, match="missing, non-regular, or a symlink"):
        plan_v21_to_v22_conversion(source, output, PolicyIOLayout.legacy_v21_q6())

    assert not output.exists()
