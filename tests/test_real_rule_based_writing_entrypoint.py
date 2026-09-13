from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from control_station import real_rule_based_writing_entrypoint, real_teleop_entrypoint
from control_station.rule_based_writing import RuleBasedWritingWebUi
from robot_motion.action import HandActionSpec
from robot_motion.deploy.schema import (
    ACTION_ARM_POSE_SLICES,
    ACTION_DIM,
    ACTION_HAND_SLICES,
)
from robot_motion.pose_frame_profiles import (
    default_rm65_workspace_near_boundary_collection_pose_frame_contract,
)
from robot_motion.relative_action import xyzquat_to_pose_matrix
from robot_motion.replay import (
    RuleBasedActionSegment,
    RuleBasedInitialize,
    RuleBasedReplay,
)
from teleop_runtime.real_teleop import rule_based_writing_plan as rule_based_plan
from teleop_runtime.real_teleop.pose_frame_runtime import (
    resolve_runtime_pose_frame_contract,
)
from teleop_runtime.real_teleop.progress_adapter import (
    load_real_teleop_progress_profile,
    resolve_real_teleop_progress_entry,
    resolve_real_teleop_task_stage_profile,
)
from teleop_runtime.replay.dataset import EpisodeSummary, LoadedReplayEpisode
from teleop_runtime.target_safety.tcp_pose_envelope_runtime import (
    load_tcp_pose_envelope_file_spec,
)

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_rule_based_writing_collection.sh"
TCP_POSE_ENVELOPE_PROFILE = (
    ROOT
    / "teleop_runtime"
    / "configs"
    / "real_teleop"
    / "rule_based_writing_bimanual_workspace_v1.json"
)


def _pose(x: float, y: float = 0.0, z: float = 0.5) -> np.ndarray:
    return np.asarray([x, y, z, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _contract():
    contract = default_rm65_workspace_near_boundary_collection_pose_frame_contract(
        ("left", "right")
    )
    assert contract is not None
    return contract


def _episode(tmp_path: Path) -> LoadedReplayEpisode:
    parquet = tmp_path / "episode_000003.parquet"
    parquet.touch()
    rows = [
        {
            "timestamp": float(index),
            "teleop.arm_mode.left.label": "fixed_hold",
        }
        for index in range(6)
    ]
    summary = EpisodeSummary(
        episode_id="episode_000003",
        episode_index=3,
        dataset_root=tmp_path,
        parquet_path=parquet,
        record={"metadata": {"progress_episode": {"episode_uuid": "source-uuid"}}},
    )
    return LoadedReplayEpisode(
        dataset_root=tmp_path,
        summary=summary,
        rows=rows,
        columns=("timestamp", "teleop.arm_mode.left.label"),
        nominal_hz=1.0,
        timing_source="fixture",
        frame_times_sec=tuple(float(index) for index in range(6)),
    )


def _actions() -> np.ndarray:
    actions = np.zeros((6, ACTION_DIM), dtype=np.float32)
    for index in range(6):
        actions[index, ACTION_ARM_POSE_SLICES["left"]] = _pose(0.1 + 0.005 * index)
        actions[index, ACTION_ARM_POSE_SLICES["right"]] = _pose(0.01 * index)
    return actions


def _build_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        rule_source_dataset=str(tmp_path),
        rule_source_episode=3,
        rule_detail_annotation_dir=str(tmp_path / "detail"),
        rule_writing_trajectory="digit_1_v1",
        rule_writing_size_m=0.08,
        rule_writing_direction="up",
        rule_writing_speed_m_s=None,
        rule_initial_alignment_speed_scale=1.0,
        left_hand_backend="none",
        right_hand_backend="none",
        dataset_root=str(tmp_path / "formal-dataset"),
        no_collection=False,
        tactile_storage="external",
        tactile_artifact_integrity=False,
        tactile_bundle_root="",
        tactile_bundle_min_free_gb=0.0,
        tactile_compression="none",
        writer_workers=4,
        writer_queue_size=512,
        collection_process_queue_size=4,
        storage_queue_size=64,
        camera_frame_max_age_sec=0.25,
        collection_warning_hz=15.0,
        hz=15.0,
        arm_control_mode="position",
        position_reference_hz=60.0,
        arm_control_hz=120.0,
        write_parquet=True,
        write_video=True,
    )


def _patch_source(
    monkeypatch,
    tmp_path: Path,
    *,
    source_contract=None,
    dual_gripper: bool = False,
    large_gripper_step: bool = False,
) -> None:
    episode = _episode(tmp_path)
    contract = source_contract or _contract()
    actions = _actions()
    hand_targets: dict[str, np.ndarray] = {}
    if dual_gripper:
        for side, start in (("left", 0.20), ("right", 0.60)):
            values = np.zeros((6, 6), dtype=np.float32)
            values[:, 0] = start + 0.01 * np.arange(6)
            if large_gripper_step and side == "right":
                values[1:, 0] = 0.10
            hand_targets[side] = values
            actions[:, ACTION_HAND_SLICES[side]] = np.nan
    actual_qpos = {
        side: np.stack([np.full(6, 0.01 * index) for index in range(6)])
        for side in ("left", "right")
    }
    actual_tcp = {
        side: np.stack([_pose(0.01 * index) for index in range(6)])
        for side in ("left", "right")
    }

    class Index:
        def __init__(self, root):
            assert Path(root) == tmp_path

        def load_episode(self, token):
            assert token == 3
            return episode

    monkeypatch.setattr(rule_based_plan, "ReplayDatasetIndex", Index)
    monkeypatch.setattr(
        rule_based_plan,
        "_load_detail_selection",
        lambda *_args, **_kwargs: rule_based_plan.DetailWritingSelection(
            episode_uuid="source-uuid",
            annotation_set_id="detail-set",
            phases={
                "right_write.pen_down": rule_based_plan.StageSpan(0, 2),
                "right_write.write": rule_based_plan.StageSpan(2, 4),
                "right_write.pen_up": rule_based_plan.StageSpan(4, 6),
            },
        ),
    )
    monkeypatch.setattr(
        rule_based_plan,
        "resolve_replay_parquet_path",
        lambda **_kwargs: (episode.summary.parquet_path, {"dataset_info": {}}),
    )
    monkeypatch.setattr(
        rule_based_plan,
        "load_contract_replay_actions_from_parquet",
        lambda *_args, **_kwargs: (
            actions.copy(),
            {
                "pose_action_absolute_frame": "canonical_desk",
                "pose_frame_contract": contract.to_dict(),
                **(
                    {
                        "hand_action_specs_by_side": {
                            side: HandActionSpec.dmtac_gripper(side).to_metadata()
                            for side in ("left", "right")
                        }
                    }
                    if dual_gripper
                    else {}
                ),
            },
            contract,
            True,
        ),
    )
    monkeypatch.setattr(
        rule_based_plan,
        "load_replay_hand_joint_targets_from_parquet",
        lambda *_args, **_kwargs: {
            side: values.copy() for side, values in hand_targets.items()
        },
    )
    monkeypatch.setattr(
        rule_based_plan,
        "load_replay_arm_observation_qpos_from_parquet",
        lambda *_args, **_kwargs: actual_qpos,
    )
    monkeypatch.setattr(
        rule_based_plan,
        "load_replay_arm_actual_tcp_pose_from_parquet",
        lambda *_args, **_kwargs: actual_tcp,
    )


def test_canonical_entrypoint_accepts_a_lightweight_extension_factory(monkeypatch) -> None:
    args = argparse.Namespace()
    extension = object()
    calls: list[object] = []
    configure = lambda parser: None

    monkeypatch.setattr(
        real_teleop_entrypoint,
        "_parse_args_with_configuration",
        lambda *, configure_parser=None: (
            calls.append(configure_parser),
            args,
        )[1],
    )
    monkeypatch.setattr(
        real_teleop_entrypoint,
        "_bind_real_teleop_runtime_support",
        lambda _namespace: None,
    )
    monkeypatch.setattr(
        real_teleop_entrypoint,
        "build_real_teleop_progress_application",
        lambda _args: "progress",
    )
    monkeypatch.setattr(
        real_teleop_entrypoint,
        "run_real_teleop",
        lambda parsed, **kwargs: calls.append((parsed, kwargs)),
    )
    monkeypatch.setattr(real_teleop_entrypoint, "cleanup_active_startup", lambda: None)

    real_teleop_entrypoint.run_real_teleop_entrypoint(
        configure_parser=configure,
        runtime_extension_factory=lambda parsed: extension if parsed is args else None,
    )

    assert calls[0] is configure
    assert args._real_teleop_runtime_extension is extension
    assert calls[1][0] is args


def test_canonical_entrypoint_can_validate_extension_before_device_preflight(
    monkeypatch,
) -> None:
    args = argparse.Namespace()
    extension = object()
    calls: list[str] = []

    def _parse(*, configure_parser=None, before_device_preflight=None):
        del configure_parser
        calls.append("parse")
        assert callable(before_device_preflight)
        before_device_preflight(args)
        calls.append("device_preflight")
        return args

    monkeypatch.setattr(real_teleop_entrypoint, "_parse_args_with_configuration", _parse)
    monkeypatch.setattr(
        real_teleop_entrypoint,
        "_bind_real_teleop_runtime_support",
        lambda _namespace: None,
    )
    monkeypatch.setattr(
        real_teleop_entrypoint,
        "build_real_teleop_progress_application",
        lambda _args: None,
    )
    monkeypatch.setattr(real_teleop_entrypoint, "run_real_teleop", lambda *_a, **_k: None)
    monkeypatch.setattr(real_teleop_entrypoint, "cleanup_active_startup", lambda: None)

    real_teleop_entrypoint.run_real_teleop_entrypoint(
        runtime_extension_factory=lambda parsed: (
            calls.append("extension_validation"),
            extension if parsed is args else None,
        )[1],
        build_runtime_extension_before_device_preflight=True,
    )

    assert calls == ["parse", "extension_validation", "device_preflight"]
    assert args._real_teleop_runtime_extension is extension


def test_specialized_entrypoint_injects_rule_only_ui(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        real_rule_based_writing_entrypoint,
        "run_real_teleop_entrypoint",
        lambda **kwargs: captured.update(kwargs),
    )

    real_rule_based_writing_entrypoint.main()

    assert captured["control_ui_type"] is RuleBasedWritingWebUi
    assert captured["runtime_extension_factory"] is (
        rule_based_plan.build_rule_based_writing_extension
    )
    assert captured["build_runtime_extension_before_device_preflight"] is True


def test_plan_composes_replay_write_and_relative_pen_up_with_left_hold(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_source(monkeypatch, tmp_path)
    contract = _contract()
    deployment = rule_based_plan.RuleBasedRealTeleopDeployment(
        active_sides=("left", "right"),
        collection_pose_frame_contract=contract,
        tcp_pose_envelope_profile="fixture-envelope",
    )

    build = rule_based_plan.build_rule_based_writing(
        _build_args(tmp_path),
        deployment=deployment,
    )

    steps = build.extension.plan.steps
    assert isinstance(steps[0], RuleBasedInitialize)
    assert steps[0].capture.enabled is False
    assert isinstance(steps[1], RuleBasedReplay)
    assert steps[1].active_sides == ("left", "right")
    assert isinstance(steps[2], RuleBasedActionSegment)
    assert steps[2].step_id == rule_based_plan.HEIGHT_DESCENT_STEP_ID
    assert steps[2].allow_empty is True
    assert steps[2].planned_steps_hint == 0
    assert isinstance(steps[3], RuleBasedActionSegment)
    assert isinstance(steps[4], RuleBasedReplay)
    assert steps[4].active_sides == ("right",)
    assert steps[4].relative_to == "right_write.write"
    assert all(step.capture.enabled for step in steps[1:])
    for group in build.extension.trajectory_groups.values():
        left = group.actions[:, ACTION_ARM_POSE_SLICES["left"]]
        assert np.allclose(left, left[0])
    assert build.writing_speed_m_s == pytest.approx(0.01)
    assert build.writing_action_hz > 0.0
    assert steps[3].stop_condition.max_steps > 3


def test_full_replay_mode_preserves_every_source_frame_without_detail_annotation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_source(
        monkeypatch,
        tmp_path,
        dual_gripper=True,
        large_gripper_step=True,
    )
    monkeypatch.setattr(
        rule_based_plan,
        "_load_detail_selection",
        lambda *_args, **_kwargs: pytest.fail(
            "complete-Episode replay must not read detail annotations"
        ),
    )
    args = _build_args(tmp_path)
    args.rule_collection_mode = rule_based_plan.RULE_COLLECTION_MODE_FULL_REPLAY
    args.left_hand_backend = "dmtac-gripper"
    args.right_hand_backend = "dmtac-gripper"

    build = rule_based_plan.build_rule_based_writing(
        args,
        deployment=rule_based_plan.RuleBasedRealTeleopDeployment(
            active_sides=("left", "right"),
            collection_pose_frame_contract=_contract(),
            tcp_pose_envelope_profile="fixture-envelope",
        ),
    )

    assert build.collection_mode == rule_based_plan.RULE_COLLECTION_MODE_FULL_REPLAY
    assert build.detail_annotation_set_id is None
    assert build.writing_speed_m_s is None
    steps = build.extension.plan.steps
    assert len(steps) == 2
    assert isinstance(steps[0], RuleBasedInitialize)
    assert steps[0].capture.enabled is False
    assert steps[0].include_end_effectors is True
    assert isinstance(steps[1], RuleBasedReplay)
    assert steps[1].start_index == 0
    assert steps[1].end_index == 6
    assert steps[1].active_sides == ("left", "right")
    assert steps[1].end_effector_sides == ("left", "right")
    assert steps[1].capture.enabled is True
    group = build.extension.trajectory_groups["full_episode"]
    expected_actions = _actions()
    for side in ("left", "right"):
        np.testing.assert_allclose(
            group.actions[:, ACTION_ARM_POSE_SLICES[side]],
            expected_actions[:, ACTION_ARM_POSE_SLICES[side]],
        )
    assert group.frame_times_sec == tuple(float(index) for index in range(6))
    assert build.extension._planned_sample_count == 6
    assert build.source_selection["source"]["source_frame_count"] == 6
    assert build.source_selection["gripper_dispatch_mode"] == (
        "causal_measured_step_limited"
    )
    assert float(
        np.max(np.abs(np.diff(group.hand_joint_targets["right"][:, 0])))
    ) > build.extension.config.gripper_max_step
    assert build.extension._configuration_options["collection_mode"] == (
        rule_based_plan.RULE_COLLECTION_MODE_FULL_REPLAY
    )
    assert build.extension._configuration_options["configuration_core_fields"] == [
        "full_replay_candidate_id",
        "initial_alignment_speed_scale",
    ]
    assert build.extension._selected_configuration["full_replay_candidate_id"] == (
        "legacy-single-source"
    )


def test_full_replay_mode_rejects_writing_speed_override(tmp_path: Path) -> None:
    args = _build_args(tmp_path)
    args.rule_collection_mode = rule_based_plan.RULE_COLLECTION_MODE_FULL_REPLAY
    args.rule_writing_speed_m_s = 0.05

    with pytest.raises(ValueError, match="preserves source timestamps"):
        rule_based_plan.build_rule_based_writing(
            args,
            deployment=rule_based_plan.RuleBasedRealTeleopDeployment(
                active_sides=("left", "right"),
                collection_pose_frame_contract=_contract(),
                tcp_pose_envelope_profile="fixture-envelope",
            ),
        )


def test_writing_mode_keeps_its_phase_hand_continuity_contract(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_source(
        monkeypatch,
        tmp_path,
        dual_gripper=True,
        large_gripper_step=True,
    )
    args = _build_args(tmp_path)
    args.left_hand_backend = "dmtac-gripper"
    args.right_hand_backend = "dmtac-gripper"

    with pytest.raises(
        ValueError,
        match=r"pen_down right gripper step .* exceeds the bounded maximum",
    ):
        rule_based_plan.build_rule_based_writing(
            args,
            deployment=rule_based_plan.RuleBasedRealTeleopDeployment(
                active_sides=("left", "right"),
                collection_pose_frame_contract=_contract(),
                tcp_pose_envelope_profile="fixture-envelope",
            ),
        )


def test_gui_writer_configuration_is_staged_then_committed_to_live_args(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_source(monkeypatch, tmp_path)
    args = _build_args(tmp_path)
    tactile_root = tmp_path / "tactile-device"
    tactile_root.mkdir()
    build = rule_based_plan.build_rule_based_writing(
        args,
        deployment=rule_based_plan.RuleBasedRealTeleopDeployment(
            active_sides=("left", "right"),
            collection_pose_frame_contract=_contract(),
            tcp_pose_envelope_profile="fixture-envelope",
        ),
    )
    options = build.extension._configuration_options
    selected = dict(build.extension._selected_configuration)
    trajectory_options = {
        value["value"]: value["label"] for value in options["trajectory_ids"]
    }
    assert trajectory_options["digit_1_v1"] == "数字 1"
    assert trajectory_options["greek_alpha_v1"] == "希腊字母 Α · alpha"
    assert trajectory_options["greek_omega_v1"] == "希腊字母 Ω · omega"
    assert selected["writing_speed_m_s"] == pytest.approx(0.01)
    assert selected["tactile_bundle_min_free_gb"] == 0.0
    assert selected["writer_workers"] == 4
    assert set(options["runtime_configuration_fields"]) >= {
        "writing_speed_m_s",
        "tactile_bundle_root",
        "tactile_bundle_min_free_gb",
        "writer_workers",
        "camera_frame_max_age_sec",
        "write_video",
    }
    assert options["runtime_info"]["sample_hz"] == pytest.approx(15.0)
    assert options["runtime_info"]["position_reference_hz"] == pytest.approx(60.0)
    assert options["runtime_info"]["outer_control_hz"] == pytest.approx(60.0)
    assert options["runtime_info"]["arm_control_hz"] == pytest.approx(120.0)
    configured = {
        **selected,
        "writing_speed_m_s": 0.05,
        "tactile_bundle_root": str(tactile_root),
        "tactile_bundle_min_free_gb": 0.0,
        "tactile_compression": "lz4",
        "writer_workers": 2,
        "writer_queue_size": 128,
        "collection_process_queue_size": 8,
        "storage_queue_size": 16,
        "camera_frame_max_age_sec": 0.30,
        "collection_warning_hz": 12.5,
        "write_video": False,
    }

    replacement = build.extension._configuration_rebuilder(configured)
    assert args.rule_writing_speed_m_s is None
    assert args.tactile_bundle_root == ""
    assert args.writer_workers == 4
    build.extension._configuration_on_commit(configured, replacement)
    assert args.rule_writing_speed_m_s == pytest.approx(0.05)
    assert args.tactile_bundle_root == str(tactile_root)
    assert args.tactile_artifact_integrity is True
    assert args.tactile_bundle_min_free_source == "rule_gui"
    assert args.writer_workers == 2
    assert args.camera_frame_max_age_sec == pytest.approx(0.30)
    assert args.write_video is False

    replacement._initial_alignment_executor.shutdown(
        wait=False,
        cancel_futures=True,
    )
    replacement._configuration_executor.shutdown(
        wait=False,
        cancel_futures=True,
    )


def test_gui_tactile_threshold_validation_rejects_currently_unavailable_reserve(
    monkeypatch,
    tmp_path: Path,
) -> None:
    target = argparse.Namespace(
        dataset_root=str(tmp_path / "formal"),
        no_collection=False,
        tactile_storage="external",
        tactile_artifact_integrity=False,
    )
    tactile_root = tmp_path / "tactile"
    tactile_root.mkdir()
    monkeypatch.setattr(
        rule_based_plan.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=10 * 1024**3),
    )
    configuration = {
        "writing_speed_m_s": 0.05,
        "tactile_bundle_root": str(tactile_root),
        "tactile_bundle_min_free_gb": 20.0,
        "tactile_compression": "none",
        "writer_workers": 4,
        "writer_queue_size": 512,
        "collection_process_queue_size": 4,
        "storage_queue_size": 64,
        "collection_warning_hz": 15.0,
        "write_parquet": True,
        "write_video": True,
    }

    with pytest.raises(ValueError, match="剩余空间低于当前 GUI 阈值"):
        rule_based_plan._apply_rule_gui_runtime_configuration(
            target,
            configuration,
        )
    assert not hasattr(target, "rule_writing_speed_m_s")


def test_dual_dmtac_plan_replays_continuous_three_phase_gripper_trajectories(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_source(monkeypatch, tmp_path, dual_gripper=True)
    args = _build_args(tmp_path)
    args.left_hand_backend = "dmtac-gripper"
    args.right_hand_backend = "dmtac-gripper"
    deployment = rule_based_plan.RuleBasedRealTeleopDeployment(
        active_sides=("left", "right"),
        collection_pose_frame_contract=_contract(),
        tcp_pose_envelope_profile="fixture-envelope",
    )

    build = rule_based_plan.build_rule_based_writing(args, deployment=deployment)

    assert build.gripper_sides == ("left", "right")
    assert build.source_selection["control_topology"] == {
        "arms": ["left", "right"],
        "grippers": ["left", "right"],
    }
    steps = build.extension.plan.steps
    assert steps[0].end_effector_sides == ("left", "right")
    assert steps[1].end_effector_sides == ("left", "right")
    assert steps[1].wait_for_terminal_arrival is True
    assert steps[4].end_effector_sides == ("left", "right")
    assert steps[4].wait_for_terminal_arrival is True

    pen_down = build.extension.trajectory_groups["pen_down"]
    pen_up = build.extension.trajectory_groups["pen_up"]
    writing = steps[3].action_source.hand_trajectory_by_side
    for side in ("left", "right"):
        down_values = pen_down.hand_joint_targets[side]
        write_values = writing[side]
        up_values = pen_up.hand_joint_targets[side]
        assert np.all(np.isfinite(down_values))
        assert np.all(np.isfinite(write_values))
        assert np.all(np.isfinite(up_values))
        assert np.allclose(down_values[:, 1:], 0.0)
        assert np.allclose(write_values[:, 1:], 0.0)
        assert np.allclose(up_values[:, 1:], 0.0)
        assert np.allclose(down_values[-1], write_values[0])
        assert np.allclose(write_values[-1], up_values[0])
        for values in (down_values, write_values, up_values):
            assert np.max(np.abs(np.diff(values, axis=0))) <= 0.050001

    initialize = build.extension.adapter.cursor.next_runtime(None)
    assert set(initialize.hand_qpos_by_side) == {"left", "right"}
    init_action = build.extension.adapter.action_for_directive(initialize)
    specs = init_action.metadata["hand_action_specs_by_side"]
    assert specs["left"]["active_mask"] == [True, False, False, False, False, False]
    assert specs["right"]["active_mask"] == [True, False, False, False, False, False]


def test_extension_factory_selects_rule_source_without_vive_dependency(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_source(monkeypatch, tmp_path, dual_gripper=True)
    args = _build_args(tmp_path)
    args.right_hand_backend = "dmtac-gripper"
    args.gripper_input_source = "none"
    args.active_sides = ("left", "right")
    args.arm_control_mode = "position"
    args.disable_dual_arm_collision_avoidance = False
    args.tcp_pose_envelope_active = True
    args.tcp_pose_envelope_profile = "fixture-envelope"
    args.position_force_guard = True
    args.position_force_guard_source = "tool"
    args.no_collection = False
    args.position_admittance_active = False
    args.position_admittance_rotation_active = False
    args.position_admittance_combined_active = False
    args.pose_frame_contract = _contract()

    extension = rule_based_plan.build_rule_based_writing_extension(args)

    assert extension.startup_home_mode == "robot_state_only"
    assert args._real_teleop_rule_command_source is True
    assert set(args._real_teleop_rule_hand_action_specs_by_side) == {"right"}
    assert (
        args._real_teleop_rule_hand_action_specs_by_side["right"].hand_model
        == "dmtac_gripper"
    )




def test_plan_fails_closed_on_pose_contract_mismatch(monkeypatch, tmp_path: Path) -> None:
    source_contract = SimpleNamespace(
        contract_hash="sha256:different",
        to_dict=lambda: _contract().to_dict(),
    )
    _patch_source(monkeypatch, tmp_path, source_contract=source_contract)
    deployment = rule_based_plan.RuleBasedRealTeleopDeployment(
        active_sides=("left", "right"),
        collection_pose_frame_contract=_contract(),
        tcp_pose_envelope_profile="fixture-envelope",
    )

    with pytest.raises(ValueError, match="contract hashes differ"):
        rule_based_plan.build_rule_based_writing(
            _build_args(tmp_path),
            deployment=deployment,
        )


def test_rule_dmtac_plan_rejects_second_q6_command_owner(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_source(monkeypatch, tmp_path, dual_gripper=True)
    args = _build_args(tmp_path)
    args.left_hand_backend = "dmtac-gripper"
    args.right_hand_backend = "dmtac-gripper"
    args.gripper_input_source = "q6"

    with pytest.raises(ValueError, match="second hand command owner"):
        rule_based_plan.build_rule_based_writing(
            args,
            deployment=rule_based_plan.RuleBasedRealTeleopDeployment(
                active_sides=("left", "right"),
                collection_pose_frame_contract=_contract(),
                tcp_pose_envelope_profile="fixture-envelope",
            ),
        )


def test_fixed_writing_anchors_in_canonical_desk_and_resamples_source_rate() -> None:
    collection_contract = _contract()
    runtime_contract = resolve_runtime_pose_frame_contract(collection_contract)
    left_hold = _pose(0.1, 0.2)
    right_desk = _pose(0.3, 0.4)
    right_raw = runtime_contract.canonical_desk_pose_vector_to_storage(
        right_desk,
        side="right",
    )
    source = rule_based_plan.FixedWritingActionSource(
        trajectory_id="digit_1_v1",
        scale_m=0.08,
        speed_m_s=0.02,
        direction="up",
        left_hold_pose=left_hold,
        reference_hz=10.0,
    )
    context = {
        "runtime_pose_frame_contract": runtime_contract,
        "actual_arm_state_by_side": {
            "right": {"actual_tcp_pose": xyzquat_to_pose_matrix(right_raw)},
        },
    }

    actions = [
        source.produce_action(None, context=context)
        for _ in range(41)
    ]

    assert np.allclose(actions[0].arm_pose_by_side["left"], left_hold)
    assert np.allclose(actions[0].arm_pose_by_side["right"], right_desk)
    assert set(actions[0].frame_by_effector.values()) == {"canonical_desk"}
    positions = np.stack(
        [action.arm_pose_by_side["right"][:3] for action in actions]
    )
    assert np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1)) <= 0.002001
    interval_velocity = actions[0].metadata["pose_action_interval_velocity"]
    assert interval_velocity["quality_valid"] is True
    assert interval_velocity["by_side"]["right"][
        "linear_speed_p90_m_s"
    ] == pytest.approx(0.02)
    assert interval_velocity["by_side"]["left"][
        "linear_speed_p90_m_s"
    ] == pytest.approx(0.0)


def test_calibrated_height_descent_is_optional_pure_z_and_c2_bounded() -> None:
    runtime_contract = resolve_runtime_pose_frame_contract(_contract())
    left_desk = _pose(0.1, 0.2, 0.45)
    right_desk = _pose(0.3, 0.4, 0.35)

    def storage_matrix(side: str, pose: np.ndarray) -> np.ndarray:
        raw = runtime_contract.canonical_desk_pose_vector_to_storage(pose, side=side)
        return xyzquat_to_pose_matrix(raw)

    source = rule_based_plan.CalibratedHeightDescentActionSource(
        reference_hz=60.0,
        linear_speed_m_s=0.02,
    )
    base_context = {
        "runtime_pose_frame_contract": runtime_contract,
        "actual_arm_state_by_side": {
            "left": {"actual_tcp_pose": storage_matrix("left", left_desk)},
            "right": {"actual_tcp_pose": storage_matrix("right", right_desk)},
        },
    }
    assert source.produce_action(None, context=base_context) is None

    source.reset()
    context = {
        **base_context,
        "writing_height_calibration": {
            "enabled": True,
            "calibrated_z_m": 0.30,
            "lower_tolerance_m": 0.001,
            "upper_tolerance_m": 0.001,
            "lower_bound_z_m": 0.299,
            "upper_bound_z_m": 0.301,
            "frame": "canonical_desk",
            "side": "right",
        },
    }
    actions = []
    while not actions or not actions[-1].metadata["rule_based_height_descent_terminal"]:
        action = source.produce_action(None, context=context)
        assert action is not None
        actions.append(action)

    right = np.stack([action.arm_pose_by_side["right"] for action in actions])
    left = np.stack([action.arm_pose_by_side["left"] for action in actions])
    assert np.allclose(left, left_desk)
    assert np.allclose(right[:, [0, 1, 3, 4, 5, 6]], right_desk[[0, 1, 3, 4, 5, 6]])
    assert np.all(np.diff(right[:, 2]) <= 1e-12)
    assert right[-1, 2] == pytest.approx(0.301)
    first_step = abs(float(right[1, 2] - right[0, 2])) if len(right) > 1 else 0.0
    last_step = abs(float(right[-1, 2] - right[-2, 2])) if len(right) > 1 else 0.0
    middle_step = float(np.max(np.abs(np.diff(right[:, 2]))))
    assert first_step < middle_step
    assert last_step < middle_step
    assert middle_step * 60.0 <= 0.0201
    assert actions[-1].metadata["rule_based_height_arrival_tolerance_m"] == pytest.approx(0.001)


def test_fixed_writing_uses_calibrated_descent_endpoint_when_supplied() -> None:
    runtime_contract = resolve_runtime_pose_frame_contract(_contract())
    right_desk = _pose(0.3, 0.4, 0.35)
    right_raw = runtime_contract.canonical_desk_pose_vector_to_storage(
        right_desk,
        side="right",
    )
    source = rule_based_plan.FixedWritingActionSource(
        trajectory_id="digit_1_v1",
        scale_m=0.02,
        speed_m_s=0.02,
        direction="up",
        left_hold_pose=_pose(0.1, 0.2),
        reference_hz=10.0,
    )
    first = source.produce_action(
        None,
        context={
            "runtime_pose_frame_contract": runtime_contract,
            "actual_arm_state_by_side": {
                "right": {"actual_tcp_pose": xyzquat_to_pose_matrix(right_raw)},
            },
            "writing_height_target_z_m": 0.301,
        },
    )
    assert first.arm_pose_by_side["right"][2] == pytest.approx(0.301)


@pytest.mark.parametrize(
    ("option", "value_hint"),
    (
        ("--mode", "NAME"),
        ("--source-dataset", "PATH"),
        ("--source-episode", "N"),
        ("--detail-annotation-dir", "PATH"),
        ("--source-pool-manifest", "PATH"),
        ("--source-pool-range", "START:END"),
        ("--selection-seed", "N"),
        ("--tactile-bundle-root", "PATH"),
        ("--dataset-root", "PATH"),
        ("--trajectory", "ID"),
        ("--size-m", "M"),
        ("--direction", "NAME"),
        ("--speed-m-s", "MPS"),
        ("--initial-alignment-speed-scale", "S"),
        ("--left-gripper-port", "PATH"),
        ("--right-gripper-port", "PATH"),
        ("--task-name", "NAME"),
        ("--tactile-min-free-gb", "N"),
        ("--camera-frame-max-age-sec", "S"),
    ),
)
def test_launcher_value_options_preserve_missing_value_errors(
    option: str,
    value_hint: str,
) -> None:
    result = subprocess.run(
        ["bash", str(LAUNCHER), option],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert result.stderr == f"[RuleBased] {option} requires {value_hint}.\n"


def test_launcher_dry_run_selects_pure_position_and_specialized_entrypoint(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    detail = tmp_path / "detail"
    tactile = tmp_path / "tactile"
    source.mkdir()
    detail.mkdir()
    tactile.mkdir()
    result = subprocess.run(
        [
            "bash",
            str(LAUNCHER),
            "--source-dataset",
            str(source),
            "--source-episode",
            "3",
            "--detail-annotation-dir",
            str(detail),
            "--tactile-bundle-root",
            str(tactile),
            "--trajectory",
            "digit_9_v1",
            "--size-m",
            "0.05",
            "--direction",
            "left",
            "--dry-run",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert "control_station.real_rule_based_writing_entrypoint" in output
    assert "--position-admittance-active" not in output
    assert "--position-control-profile rm65_bimanual_position_v2_balanced" in output
    assert "rm65_bimanual_position_v2_balanced_60hz_jerk_v2" not in output
    assert (
        "--tcp-pose-envelope-profile "
        "teleop_runtime/configs/real_teleop/"
        "rule_based_writing_bimanual_workspace_v1.json"
    ) in output
    assert "--tcp-pose-envelope-active" in output
    assert "--rule-writing-trajectory digit_9_v1" in output
    assert "--rule-writing-size-m 0.05" in output
    assert "--rule-writing-direction left" in output
    assert "--left-hand-backend none" in output
    assert "--right-hand-backend dmtac-gripper" in output
    assert "--gripper-input-source none" in output
    assert "--no-q6-default-config" in output
    assert "--special-arm-control-mode manual" in output
    assert "--q6-config" not in output
    assert "five-pedal" not in output
    assert "--dmtac-with-tactile" in output
    assert "--q6-config-required" not in output
    assert "--q6-hold-recalibration-gesture" not in output
    assert "--auto-calibrate" not in output
    assert "--collection-progress-gui" not in output
    assert (
        "--collection-progress-profile "
        "catalog:dmtac_rm65_right_gripper_rule_writing_progress"
    ) in output
    assert (
        "--collection-stage-profile "
        "catalog:dmtac_rm65_right_gripper_rule_writing_progress_detailed"
    ) in output
    assert "--collection-progress-entry-step right_write" in output
    assert "--collection-progress-entry-stage right_write.pen_down" in output
    assert "--camera-frame-max-age-sec 0.25" in output


def test_launcher_accepts_greek_trajectory_catalog_id(tmp_path: Path) -> None:
    source = tmp_path / "source"
    detail = tmp_path / "detail"
    tactile = tmp_path / "tactile"
    source.mkdir()
    detail.mkdir()
    tactile.mkdir()
    result = subprocess.run(
        [
            "bash",
            str(LAUNCHER),
            "--source-dataset",
            str(source),
            "--source-episode",
            "3",
            "--detail-annotation-dir",
            str(detail),
            "--tactile-bundle-root",
            str(tactile),
            "--trajectory",
            "greek_omega_v1",
            "--dry-run",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--rule-writing-trajectory greek_omega_v1" in result.stdout


def test_launcher_full_replay_mode_uses_complete_episode_progress_without_detail(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    tactile = tmp_path / "tactile"
    source.mkdir()
    tactile.mkdir()
    result = subprocess.run(
        [
            "bash",
            str(LAUNCHER),
            "--mode",
            "full-replay",
            "--source-dataset",
            str(source),
            "--source-episode",
            "3",
            "--tactile-bundle-root",
            str(tactile),
            "--dry-run",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert "--rule-collection-mode full_trajectory_replay" in output
    assert f"--rule-source-dataset {source}" in output
    assert "--rule-source-episode 3" in output
    assert "--rule-detail-annotation-dir" not in output
    assert "--rule-writing-trajectory" not in output
    assert "--rule-writing-size-m" not in output
    assert "--rule-writing-direction" not in output
    assert (
        "--collection-progress-profile "
        "catalog:dmtac_rm65_full_trajectory_replay_v1_progress"
    ) in output
    assert (
        "--collection-stage-profile "
        "catalog:dmtac_rm65_full_trajectory_replay_v1_progress"
    ) in output
    assert "--collection-progress-entry-step execute_task" in output
    assert "--collection-progress-entry-stage execute_task.full_replay" in output
    assert "--collection-task-id dmtac_rm65_full_trajectory_replay_v1" in output
    progress = load_real_teleop_progress_profile(
        "catalog:dmtac_rm65_full_trajectory_replay_v1_progress"
    )
    stage = resolve_real_teleop_task_stage_profile(
        progress,
        "catalog:dmtac_rm65_full_trajectory_replay_v1_progress",
    )
    assert progress.profile_id == "dmtac_rm65_full_trajectory_replay_v1_progress"
    assert stage is not None
    assert stage.profile_id == "dmtac_rm65_full_trajectory_replay_v1_stages"
    assert resolve_real_teleop_progress_entry(
        progress_profile=progress,
        stage_profile=stage,
        entry_step_id="execute_task",
        entry_stage_id="execute_task.full_replay",
        entry_condition="nominal",
    ) == ("execute_task", "execute_task.full_replay", "NOMINAL")


def test_launcher_full_replay_rejects_writing_speed_override(tmp_path: Path) -> None:
    source = tmp_path / "source"
    tactile = tmp_path / "tactile"
    source.mkdir()
    tactile.mkdir()
    result = subprocess.run(
        [
            "bash",
            str(LAUNCHER),
            "--mode",
            "full-replay",
            "--source-dataset",
            str(source),
            "--source-episode",
            "3",
            "--tactile-bundle-root",
            str(tactile),
            "--speed-m-s",
            "0.05",
            "--dry-run",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "source Episode timestamps are preserved" in result.stderr


def test_launcher_tcp_pose_envelope_covers_both_calibration_workspaces() -> None:
    workspace_by_side = {
        "left": ((-0.8, -0.7, 0.0), (0.7, 0.8, 0.9)),
        "right": ((-0.7, -0.8, 0.0), (0.8, 0.7, 0.9)),
    }

    spec = load_tcp_pose_envelope_file_spec(
        TCP_POSE_ENVELOPE_PROFILE,
        active_sides=("left", "right"),
        calibration_workspace_resolver=workspace_by_side.__getitem__,
    )

    assert spec.profile_id == "rule-based-writing-bimanual-workspace-v1"
    assert set(spec.side_config_by_side) == {"left", "right"}
    for side, (lower, upper) in workspace_by_side.items():
        config = spec.side_config_by_side[side]
        assert config.enabled is True
        assert config.absolute_min_xyz_m == lower
        assert config.absolute_max_xyz_m == upper
        assert config.requires_anchor is False


def test_launcher_can_enable_left_dmtac_later_without_enabling_q6(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    detail = tmp_path / "detail"
    tactile = tmp_path / "tactile"
    source.mkdir()
    detail.mkdir()
    tactile.mkdir()
    result = subprocess.run(
        [
            "bash",
            str(LAUNCHER),
            "--source-dataset",
            str(source),
            "--source-episode",
            "3",
            "--detail-annotation-dir",
            str(detail),
            "--tactile-bundle-root",
            str(tactile),
            "--with-left-gripper",
            "--dry-run",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert "--left-hand-backend dmtac-gripper" in output
    assert "--right-hand-backend dmtac-gripper" in output
    assert "--gripper-input-source none" in output
    assert "--q6-config-required" not in output
    assert "--q6-hold-recalibration-gesture" not in output
    assert (
        "--collection-progress-profile "
        "catalog:dmtac_rm65_bimanual_position_writing_four_stage_progress"
    ) in output
    assert (
        "--collection-stage-profile "
        "catalog:dmtac_rm65_bimanual_position_writing_four_stage_progress_detailed"
    ) in output


def test_launcher_dry_run_accepts_cross_dataset_source_pool(tmp_path: Path) -> None:
    manifest = tmp_path / "source-pool.json"
    tactile = tmp_path / "tactile"
    manifest.write_text("{}", encoding="utf-8")
    tactile.mkdir()
    result = subprocess.run(
        [
            "bash",
            str(LAUNCHER),
            "--source-pool-manifest",
            str(manifest),
            "--source-pool-range",
            "10:30",
            "--selection-seed",
            "42",
            "--tactile-bundle-root",
            str(tactile),
            "--dry-run",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert f"--rule-source-pool-manifest {manifest}" in output
    assert "--rule-source-pool-range 10:30" in output
    assert "--rule-source-selection-seed 42" in output
    assert "--rule-source-dataset" not in output
    assert "--rule-source-episode" not in output
    assert "--rule-detail-annotation-dir" not in output


def test_rule_arguments_expose_size_direction_and_source_binding() -> None:
    parser = argparse.ArgumentParser()
    rule_based_plan.add_rule_based_writing_arguments(parser)
    args = parser.parse_args(
        [
            "--rule-source-dataset",
            "/source",
            "--rule-source-episode",
            "7",
            "--rule-detail-annotation-dir",
            "/detail",
            "--rule-writing-size-m",
            "0.04",
            "--rule-writing-direction",
            "down",
        ]
    )

    assert args.rule_source_episode == 7
    assert args.rule_writing_size_m == pytest.approx(0.04)
    assert args.rule_writing_direction == "down"


def test_rule_arguments_accept_pool_without_legacy_source_arguments() -> None:
    parser = argparse.ArgumentParser()
    rule_based_plan.add_rule_based_writing_arguments(parser)
    args = parser.parse_args(
        [
            "--rule-source-pool-manifest",
            "/pool.json",
            "--rule-source-pool-range",
            "4:19",
            "--rule-source-selection-seed",
            "8",
        ]
    )

    assert args.rule_source_dataset is None
    assert args.rule_source_pool_manifest == "/pool.json"
    assert args.rule_source_pool_range == "4:19"
    assert args.rule_source_selection_seed == 8
