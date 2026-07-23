from __future__ import annotations

import json
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from utils.dataset_postprocess.organize_episodes import (
    _output_relative_path,
    _passes_minimum_duration,
    build_datasets,
    plan_dataset_build,
)


def _hand_vector(value: float) -> list[float]:
    return [0.0, float(value), 0.0, 0.0, 0.0, 0.0]


def _make_source(root: Path) -> None:
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "videos/observation.images.front/chunk-000").mkdir(parents=True)
    action = [0.0] * 26
    action[14:20] = _hand_vector(0.7)
    action[20:26] = _hand_vector(0.4)
    state = [0.0] * 38
    state[13:19] = _hand_vector(0.6)
    state[32:38] = _hand_vector(0.3)
    table = pa.table(
        {
            "index": pa.array([40, 41]),
            "episode_index": pa.array([5, 5]),
            "frame_index": pa.array([0, 1]),
            "action": pa.array([action, action]),
            "observation.last_action": pa.array([action, action]),
            "observation.state": pa.array([state, state]),
            "action.left_hand.joint_target": pa.array([_hand_vector(0.7), _hand_vector(0.7)]),
            "action.right_hand.joint_target": pa.array([_hand_vector(0.4), _hand_vector(0.4)]),
            "observation.left_hand.qpos": pa.array([_hand_vector(0.6), _hand_vector(0.6)]),
            "observation.right_hand.qpos": pa.array([_hand_vector(0.3), _hand_vector(0.3)]),
            "observation.images.front.path": pa.array(
                [
                    "videos/observation.images.front/chunk-000/episode_000005.mp4",
                    "videos/observation.images.front/chunk-000/episode_000005.mp4",
                ]
            ),
            "hand_action_specs_json": pa.array([json.dumps({"left": {}}), json.dumps({"left": {}})]),
        }
    )
    pq.write_table(table, root / "data/chunk-000/episode_000005.parquet")
    source_video = root / "videos/observation.images.front/chunk-000/episode_000005.mp4"
    source_video.write_bytes(b"synthetic-video")
    info = {
        "format": "lerobot_v2.1_style",
        "dataset_name": "source",
        "episode_count": 1,
        "frame_count": 2,
        "nominal_fps": 15.0,
        "chunk_size": 1000,
        "features": {"action": {"shape": [26], "dtype": "float32"}},
        "hand_action_specs_by_side": {"left": {"hand_model": "inspire_tac"}},
        "hand_active_mask_by_side": {"left": [True] * 6},
    }
    record = {
        "episode_index": 5,
        "episode_id": "episode_000005",
        "frame_count": 2,
        "duration_sec": 2.0,
        "parquet_path": "data/chunk-000/episode_000005.parquet",
        "video_paths": {"front": "videos/observation.images.front/chunk-000/episode_000005.mp4"},
        "summary": {
            "video_frame_stats": {
                "front": {
                    "path": "videos/observation.images.front/chunk-000/episode_000005.mp4",
                    "written": 2,
                }
            }
        },
        "metadata": {
            "hand_action_specs_by_side": {"left": {"hand_model": "inspire_tac"}},
            "gripper_policy": {"output_dim": 1},
        },
    }
    (root / "meta").mkdir()
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta/episodes.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (root / "meta/episodes_stats.jsonl").write_text(json.dumps({"episode_index": 5, "frame_count": 2}) + "\n", encoding="utf-8")


def test_build_reindexes_and_migrates_legacy_dmtac_slot(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _make_source(source)
    manifest = {
        "min_duration_sec": 2.0,
        "datasets": [
            {
                "name": "derived",
                "output_root": "output",
                "task": {"id": "cube", "description": "cube"},
                "sources": [
                    {
                        "dataset_root": "source",
                        "episode_indices": [5],
                        "transforms": ["dmtac_slot1_to_slot0"],
                    }
                ],
            }
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    plan = plan_dataset_build(manifest_path)
    assert plan[0].episode_count == 1
    assert plan[0].minimum_duration_sec == 2.0
    assert plan[0].filtered_by_minimum_duration == ()
    assert plan[0].episodes[0].source_episode_index == 5
    build_datasets(manifest_path)

    output_record = json.loads((output / "meta/episodes.jsonl").read_text().strip())
    assert output_record["episode_index"] == 0
    assert output_record["episode_id"] == "episode_000000"
    assert output_record["parquet_path"] == "data/chunk-000/episode_000000.parquet"
    assert output_record["video_paths"]["front"].endswith("episode_000000.mp4")
    assert output_record["summary"]["video_frame_stats"]["front"]["path"].endswith(
        "episode_000000.mp4"
    )
    source_video = source / "videos/observation.images.front/chunk-000/episode_000005.mp4"
    output_video = output / "videos/observation.images.front/chunk-000/episode_000000.mp4"
    assert source_video.stat().st_ino == output_video.stat().st_ino

    output_table = pq.read_table(output / output_record["parquet_path"])
    assert output_table["episode_index"].to_pylist() == [0, 0]
    assert output_table["index"].to_pylist() == [0, 1]
    assert output_table["observation.images.front.path"].to_pylist() == [
        "videos/observation.images.front/chunk-000/episode_000000.mp4",
        "videos/observation.images.front/chunk-000/episode_000000.mp4",
    ]
    assert output_table["action"].to_pylist()[0][14:20] == pytest.approx([0.7, 0, 0, 0, 0, 0])
    assert output_table["action"].to_pylist()[0][20:26] == pytest.approx([0.4, 0, 0, 0, 0, 0])
    assert output_table["observation.state"].to_pylist()[0][13:19] == pytest.approx([0.6, 0, 0, 0, 0, 0])

    source_table = pq.read_table(source / "data/chunk-000/episode_000005.parquet")
    assert source_table["action"].to_pylist()[0][14:20] == pytest.approx([0, 0.7, 0, 0, 0, 0])
    output_info = json.loads((output / "meta/info.json").read_text())
    assert output_info["dataset_name"] == "derived"
    assert output_info["hand_action_specs_by_side"]["left"]["hand_model"] == "dmtac_gripper"
    assert output_info["hand_active_mask_by_side"]["left"] == [True, False, False, False, False, False]


def test_reindex_path_accepts_an_already_zero_episode_name() -> None:
    assert _output_relative_path(
        Path("data/chunk-000/episode_000000.parquet"),
        output_episode_index=0,
        chunk_size=1000,
    ) == Path("data/chunk-000/episode_000000.parquet")


def test_rewrite_paths_does_not_treat_episode_event_names_as_paths() -> None:
    from utils.dataset_postprocess.organize_episodes import _rewrite_paths_in_value

    payload = {
        "event": "episode_start",
        "path": "data/chunk-000/episode_000005.parquet",
    }

    assert _rewrite_paths_in_value(
        payload,
        output_episode_index=2,
        chunk_size=1000,
    ) == {
        "event": "episode_start",
        "path": "data/chunk-000/episode_000002.parquet",
    }


def test_minimum_duration_filter_is_inclusive_at_the_threshold() -> None:
    assert not _passes_minimum_duration({"episode_index": 7, "duration_sec": 1.999}, 2.0)
    assert _passes_minimum_duration({"episode_index": 8, "duration_sec": 2.0}, 2.0)
    assert _passes_minimum_duration({"episode_index": 9, "duration_sec": 2.001}, 2.0)


def test_plan_can_select_one_named_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _make_source(source)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "name": "first",
                        "output_root": "first",
                        "task": {"id": "cube", "description": "cube"},
                        "sources": [{"dataset_root": "source", "episode_indices": [5]}],
                    },
                    {
                        "name": "second",
                        "output_root": "second",
                        "task": {"id": "cube", "description": "cube"},
                        "sources": [{"dataset_root": "source", "episode_indices": [5]}],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    plan = plan_dataset_build(manifest_path, dataset_names=["second"])
    assert [item.name for item in plan] == ["second"]
