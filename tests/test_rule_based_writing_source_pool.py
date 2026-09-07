from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from dataset_contracts.rule_based_writing_sources import (
    RULE_BASED_WRITING_SOURCE_POOL_SCHEMA_V1,
    RuleBasedWritingSourcePoolV1,
)
from dataset_interface.rule_based_writing_sources import (
    load_rule_based_writing_source_pool,
)
from robot_motion.deploy.schema import ACTION_ARM_POSE_SLICES, ACTION_DIM
from robot_motion.pose_frame_profiles import (
    default_rm65_workspace_near_boundary_collection_pose_frame_contract,
)
from teleop_runtime.real_teleop import rule_based_writing_plan as writing_plan
from teleop_runtime.replay.dataset import EpisodeSummary, LoadedReplayEpisode


def _manifest_payload(candidates: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema": RULE_BASED_WRITING_SOURCE_POOL_SCHEMA_V1,
        "pool_id": "diverse-writing-fixture",
        "candidates": candidates,
    }


def _candidate(candidate_id: str, phase: str, episode: int) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "dataset_root": f"datasets/{candidate_id}",
        "episode_index": episode,
        "detail_annotation_dir": f"details/{candidate_id}",
        "phases": [phase],
        "tags": [f"tag-{candidate_id}"],
    }


def _write_manifest(tmp_path: Path, candidates: list[dict[str, object]]) -> Path:
    path = tmp_path / "pool.json"
    path.write_text(
        json.dumps(_manifest_payload(candidates), ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _make_dirs(tmp_path: Path, candidate_ids: tuple[str, ...]) -> None:
    for candidate_id in candidate_ids:
        (tmp_path / "datasets" / candidate_id).mkdir(parents=True)
        (tmp_path / "details" / candidate_id).mkdir(parents=True)


def test_pool_contract_rejects_duplicate_ids_and_hash_is_stable() -> None:
    payload = _manifest_payload(
        [_candidate("a", "pen_down", 1), _candidate("b", "write", 2)]
    )
    first = RuleBasedWritingSourcePoolV1.from_dict(payload)
    second = RuleBasedWritingSourcePoolV1.from_dict(json.loads(json.dumps(payload)))
    assert first.semantic_content_hash == second.semantic_content_hash

    payload["candidates"][1]["candidate_id"] = "a"  # type: ignore[index]
    with pytest.raises(ValueError, match="candidate_id values must be unique"):
        RuleBasedWritingSourcePoolV1.from_dict(payload)


def test_pool_loader_resolves_manifest_relative_paths(tmp_path: Path) -> None:
    _make_dirs(tmp_path, ("a",))
    manifest = _write_manifest(tmp_path, [_candidate("a", "pen_down", 1)])

    loaded = load_rule_based_writing_source_pool(manifest)

    assert loaded.candidates[0].dataset_root == (tmp_path / "datasets" / "a")
    assert loaded.candidates[0].detail_annotation_dir == (tmp_path / "details" / "a")


def test_phase_selection_is_deterministic_distinct_and_range_checked(
    tmp_path: Path,
) -> None:
    ids = ("a", "b", "c", "d")
    _make_dirs(tmp_path, ids)
    candidates = []
    for index, candidate_id in enumerate(ids):
        item = _candidate(candidate_id, "pen_down", index)
        item["phases"] = ["pen_down", "write", "pen_up"]
        candidates.append(item)
    manifest = _write_manifest(tmp_path, candidates)
    args = argparse.Namespace(
        rule_source_pool_manifest=str(manifest),
        rule_source_pool_range="0:3",
        rule_source_selection_seed=17,
        rule_source_dataset=None,
        rule_source_episode=None,
        rule_detail_annotation_dir=None,
    )

    first, metadata = writing_plan._resolve_source_specs(args)
    second, second_metadata = writing_plan._resolve_source_specs(args)

    assert {phase: value.candidate.candidate_id for phase, value in first.items()} == {
        phase: value.candidate.candidate_id for phase, value in second.items()
    }
    assert len({value.candidate.candidate_id for value in first.values()}) == 3
    assert metadata == second_metadata
    assert metadata["pool_range"] == {"start": 0, "end": 3}
    patterns = set()
    for seed in range(12):
        args.rule_source_selection_seed = seed
        choices, _ = writing_plan._resolve_source_specs(args)
        patterns.add(
            tuple(
                choices[phase].candidate.candidate_id
                for phase in ("pen_down", "write", "pen_up")
            )
        )
    assert len(patterns) > 1

    args.rule_source_pool_range = "2:2"
    with pytest.raises(ValueError, match="non-empty half-open interval"):
        writing_plan._resolve_source_specs(args)


def test_pool_fails_when_selected_range_has_no_phase_candidate(tmp_path: Path) -> None:
    _make_dirs(tmp_path, ("a", "b"))
    manifest = _write_manifest(
        tmp_path,
        [_candidate("a", "pen_down", 1), _candidate("b", "write", 2)],
    )
    args = argparse.Namespace(
        rule_source_pool_manifest=str(manifest),
        rule_source_pool_range=None,
        rule_source_selection_seed=0,
        rule_source_dataset=None,
        rule_source_episode=None,
        rule_detail_annotation_dir=None,
    )
    with pytest.raises(ValueError, match="no candidate for phase 'pen_up'"):
        writing_plan._resolve_source_specs(args)


def test_full_replay_source_selection_uses_whole_episode_not_phase_eligibility(
    tmp_path: Path,
) -> None:
    _make_dirs(tmp_path, ("a", "b"))
    manifest = _write_manifest(
        tmp_path,
        [_candidate("a", "pen_down", 1), _candidate("b", "pen_up", 2)],
    )
    args = argparse.Namespace(
        rule_source_pool_manifest=str(manifest),
        rule_source_pool_range=None,
        rule_source_selection_seed=0,
        rule_source_dataset=None,
        rule_source_episode=None,
        rule_detail_annotation_dir=None,
        rule_full_replay_candidate_id="b",
    )

    selected, metadata = writing_plan._resolve_full_replay_source_spec(args)

    assert selected.candidate.candidate_id == "b"
    assert selected.candidate.episode_index == 2
    assert metadata["collection_mode"] == "full_trajectory_replay"
    assert metadata["mode"] == "source_pool_gui_selection"


def test_hand_resampling_rejects_non_monotonic_times_before_division() -> None:
    values = np.zeros((2, 6), dtype=np.float32)
    with np.errstate(all="raise"):
        with pytest.raises(ValueError, match="source.*strictly increasing"):
            writing_plan._resample_hand_trajectory(
                values,
                source_times_sec=(1.0, 1.0),
                target_times_sec=(0.0, 1.0),
            )


def test_build_uses_three_datasets_and_records_exact_phase_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ids = ("down", "stroke", "up")
    phases = ("pen_down", "write", "pen_up")
    _make_dirs(tmp_path, ids)
    manifest = _write_manifest(
        tmp_path,
        [_candidate(candidate_id, phase, index + 1) for index, (candidate_id, phase) in enumerate(zip(ids, phases))],
    )
    contract = default_rm65_workspace_near_boundary_collection_pose_frame_contract(
        ("left", "right")
    )
    assert contract is not None
    episodes: dict[Path, LoadedReplayEpisode] = {}
    actions_by_path: dict[Path, np.ndarray] = {}
    for index, candidate_id in enumerate(ids, start=1):
        root = (tmp_path / "datasets" / candidate_id).resolve()
        parquet = root / f"episode_{index:06d}.parquet"
        parquet.touch()
        rows = [
            {"timestamp": float(frame), "teleop.arm_mode.left.label": "fixed_hold"}
            for frame in range(6)
        ]
        episodes[root] = LoadedReplayEpisode(
            dataset_root=root,
            summary=EpisodeSummary(
                episode_id=f"episode_{index:06d}",
                episode_index=index,
                dataset_root=root,
                parquet_path=parquet,
                record={"metadata": {"progress_episode": {"episode_uuid": candidate_id}}},
            ),
            rows=rows,
            columns=("timestamp", "teleop.arm_mode.left.label"),
            nominal_hz=float(10 + index),
            timing_source="fixture",
            frame_times_sec=tuple(float(frame) for frame in range(6)),
        )
        actions = np.zeros((6, ACTION_DIM), dtype=np.float32)
        for frame in range(6):
            actions[frame, ACTION_ARM_POSE_SLICES["left"]] = [
                0.1, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0
            ]
            actions[frame, ACTION_ARM_POSE_SLICES["right"]] = [
                0.01 * index * frame, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0
            ]
        actions_by_path[parquet.resolve()] = actions

    class Index:
        def __init__(self, root: str | Path) -> None:
            self.root = Path(root).resolve()

        def load_episode(self, token: int) -> LoadedReplayEpisode:
            episode = episodes[self.root]
            assert token == episode.summary.episode_index
            return episode

    monkeypatch.setattr(writing_plan, "ReplayDatasetIndex", Index)
    monkeypatch.setattr(
        writing_plan,
        "_load_detail_selection",
        lambda episode, _detail: writing_plan.DetailWritingSelection(
            episode_uuid=str(episode.summary.episode_index),
            annotation_set_id=f"detail-{episode.summary.episode_index}",
            phases={
                "right_write.pen_down": writing_plan.StageSpan(0, 2),
                "right_write.write": writing_plan.StageSpan(2, 4),
                "right_write.pen_up": writing_plan.StageSpan(4, 6),
            },
        ),
    )
    monkeypatch.setattr(
        writing_plan,
        "resolve_replay_parquet_path",
        lambda **kwargs: (
            episodes[Path(kwargs["dataset_root"]).resolve()].summary.parquet_path,
            {},
        ),
    )
    monkeypatch.setattr(
        writing_plan,
        "load_contract_replay_actions_from_parquet",
        lambda path, **_kwargs: (actions_by_path[Path(path).resolve()].copy(), {}, contract, True),
    )
    qpos = {side: np.zeros((6, 6), dtype=np.float32) for side in ("left", "right")}
    tcp = {
        side: np.tile(np.asarray([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]), (6, 1))
        for side in ("left", "right")
    }
    monkeypatch.setattr(
        writing_plan,
        "load_replay_arm_observation_qpos_from_parquet",
        lambda *_args, **_kwargs: qpos,
    )
    monkeypatch.setattr(
        writing_plan,
        "load_replay_hand_joint_targets_from_parquet",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        writing_plan,
        "load_replay_arm_actual_tcp_pose_from_parquet",
        lambda *_args, **_kwargs: tcp,
    )
    args = argparse.Namespace(
        rule_source_pool_manifest=str(manifest),
        rule_source_pool_range=None,
        rule_source_selection_seed=5,
        rule_source_dataset=None,
        rule_source_episode=None,
        rule_detail_annotation_dir=None,
        rule_writing_trajectory="digit_1_v1",
        rule_writing_size_m=0.08,
        rule_writing_direction="up",
        rule_writing_speed_m_s=None,
        rule_initial_alignment_speed_scale=1.0,
        left_hand_backend="none",
        right_hand_backend="none",
    )
    deployment = writing_plan.RuleBasedRealTeleopDeployment(
        active_sides=("left", "right"),
        collection_pose_frame_contract=contract,
        tcp_pose_envelope_profile="fixture",
    )

    build = writing_plan.build_rule_based_writing(args, deployment=deployment)

    assert build.extension.trajectory_groups["pen_down"].source_episode_index == 1
    assert build.extension.trajectory_groups["pen_up"].source_episode_index == 3
    assert build.writing_speed_m_s == pytest.approx(0.02)
    assert {
        phase: value["candidate_id"]
        for phase, value in build.source_selection["phases"].items()
    } == {"pen_down": "down", "write": "stroke", "pen_up": "up"}
    assert (
        build.extension.trajectory_groups["pen_up"].metadata[
            "rule_based_source_selection"
        ]["pool_manifest_hash"]
        == build.source_selection["pool_manifest_hash"]
    )
    options = build.extension._configuration_options
    assert [
        item["value"]
        for item in options["source_candidates_by_phase"]["pen_down"]
    ] == ["down"]
    assert len(options["trajectory_ids"]) == 9
    assert {item["value"] for item in options["directions"]} == {
        "up",
        "right",
        "down",
        "left",
    }
    selected_configuration = dict(build.extension._selected_configuration)
    rebuilt = build.extension._configuration_rebuilder(
        {
            **selected_configuration,
            "trajectory_id": "digit_9_v1",
            "direction": "left",
            "initial_alignment_speed_scale": 0.4,
        }
    )
    writing_step = next(
        step
        for step in rebuilt.plan.steps
        if step.step_id == "right_write.write"
    )
    assert writing_step.action_source.trajectory_id == "digit_9_v1"
    assert writing_step.action_source.direction == "left"
    assert rebuilt.config.initial_alignment_speed_scale == pytest.approx(0.4)
    rebuilt._initial_alignment_executor.shutdown(wait=False, cancel_futures=True)
    rebuilt._configuration_executor.shutdown(wait=False, cancel_futures=True)


def test_gui_phase_candidate_overrides_cover_independent_source_combinations(
    tmp_path: Path,
) -> None:
    ids = ("a", "b", "c")
    _make_dirs(tmp_path, ids)
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "candidate_id": candidate_id,
                "dataset_root": f"datasets/{candidate_id}",
                "episode_index": index,
                "detail_annotation_dir": f"details/{candidate_id}",
                "phases": ["pen_down", "write", "pen_up"],
                "tags": [],
            }
            for index, candidate_id in enumerate(ids)
        ],
    )
    args = argparse.Namespace(
        rule_source_pool_manifest=str(manifest),
        rule_source_pool_range=None,
        rule_source_dataset=None,
        rule_source_episode=None,
        rule_detail_annotation_dir=None,
        _rule_source_candidate_ids_by_phase={
            "pen_down": "c",
            "write": "a",
            "pen_up": "b",
        },
    )

    selected, metadata = writing_plan._resolve_source_specs(args)

    assert {
        phase: spec.candidate.candidate_id for phase, spec in selected.items()
    } == {"pen_down": "c", "write": "a", "pen_up": "b"}
    assert metadata["mode"] == "source_pool_gui_selection"
