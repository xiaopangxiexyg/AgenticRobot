from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from control_station import real_teleop_entrypoint
from robot_motion.deploy.schema import ACTION_ARM_POSE_SLICES, ACTION_DIM
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
        actions[index, ACTION_ARM_POSE_SLICES["left"]] = _pose(0.1)
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
    )


def _patch_source(monkeypatch, tmp_path: Path, *, source_contract=None) -> None:
    episode = _episode(tmp_path)
    contract = source_contract or _contract()
    actions = _actions()
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
            },
            contract,
            True,
        ),
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
    assert isinstance(steps[3], RuleBasedReplay)
    assert steps[3].active_sides == ("right",)
    assert steps[3].relative_to == "right_write.write"
    assert all(step.capture.enabled for step in steps[1:])
    for group in build.extension.trajectory_groups.values():
        left = group.actions[:, ACTION_ARM_POSE_SLICES["left"]]
        assert np.allclose(left, left[0])
    assert build.writing_speed_m_s == pytest.approx(0.01)
    assert build.writing_action_hz > 0.0
    assert steps[2].stop_condition.max_steps > 3


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
    assert "rm65_bimanual_position_v2_balanced_60hz_jerk_v2" in output
    assert (
        "--tcp-pose-envelope-profile "
        "teleop_runtime/configs/real_teleop/"
        "rule_based_writing_bimanual_workspace_v1.json"
    ) in output
    assert "--tcp-pose-envelope-active" in output
    assert "--rule-writing-trajectory digit_9_v1" in output
    assert "--rule-writing-size-m 0.05" in output
    assert "--rule-writing-direction left" in output


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
