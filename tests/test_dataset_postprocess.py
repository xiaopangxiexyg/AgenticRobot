from __future__ import annotations

import json

import numpy as np
import pytest

from teleop_runtime.collection.video_frame import NearestFrameBuffer, VideoFrame
from utils.dataset_postprocess import (
    causal_align_dataset,
    clean_stutters,
    delete_episodes,
    detect_fault_nan_tail,
    detect_stutter_segments,
    label_force_collisions,
    parse_episode_selection,
    plan_causal_align,
    plan_force_collision_labels,
    plan_episode_deletion,
    plan_episode_split,
    plan_stutter_clean,
    split_episodes,
    trim_nan_tails,
)
from utils.dataset_postprocess.io import clone_dataset_tree


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_parquet(path, rows):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_camera_buffer_latest_before_or_at_does_not_look_ahead() -> None:
    buffer = NearestFrameBuffer(keep_last=4)
    buffer.push(VideoFrame(image="old", source_timestamp=1.0, available_timestamp=1.0))
    buffer.push(VideoFrame(image="future", source_timestamp=1.2, available_timestamp=1.2))

    assert buffer.get_latest_before_or_at(1.1).image == "old"
    assert buffer.get_latest_before_or_at(0.9) is None
    assert buffer.get_latest_before_or_at(1.2).image == "future"


def test_parse_episode_selection_ranges() -> None:
    assert parse_episode_selection("1,3,5-7,3") == (1, 3, 5, 6, 7)


def test_clone_dataset_tree_copies_mutable_prefixes_and_hardlinks_artifacts(tmp_path) -> None:
    root = tmp_path / "dataset"
    out = tmp_path / "dataset_clone"
    (root / "meta").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "videos/front/chunk-000").mkdir(parents=True)
    (root / "meta/info.json").write_text("{}", encoding="utf-8")
    (root / "data/chunk-000/episode_000000.parquet").write_text("rows", encoding="utf-8")
    (root / "videos/front/chunk-000/episode_000000.mp4").write_text("video", encoding="utf-8")

    clone_dataset_tree(root, out, copy_prefixes=("meta", "data"))

    source_meta = root / "meta/info.json"
    clone_meta = out / "meta/info.json"
    source_data = root / "data/chunk-000/episode_000000.parquet"
    clone_data = out / "data/chunk-000/episode_000000.parquet"
    source_video = root / "videos/front/chunk-000/episode_000000.mp4"
    clone_video = out / "videos/front/chunk-000/episode_000000.mp4"

    assert clone_meta.read_text(encoding="utf-8") == "{}"
    assert clone_data.read_text(encoding="utf-8") == "rows"
    assert clone_video.read_text(encoding="utf-8") == "video"
    assert (source_meta.stat().st_dev, source_meta.stat().st_ino) != (
        clone_meta.stat().st_dev,
        clone_meta.stat().st_ino,
    )
    assert (source_data.stat().st_dev, source_data.stat().st_ino) != (
        clone_data.stat().st_dev,
        clone_data.stat().st_ino,
    )
    assert (source_video.stat().st_dev, source_video.stat().st_ino) == (
        clone_video.stat().st_dev,
        clone_video.stat().st_ino,
    )


def test_clone_dataset_tree_rejects_output_inside_source(tmp_path) -> None:
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="must not be inside source_root"):
        clone_dataset_tree(root, root / "derived")


def test_detect_stutter_segments_splits_on_long_gap() -> None:
    rows = [
        {"frame_index": 0, "timestamp": 0.00},
        {"frame_index": 1, "timestamp": 0.06},
        {"frame_index": 2, "timestamp": 0.40},
        {"frame_index": 3, "timestamp": 0.46},
    ]

    segments = detect_stutter_segments(rows, episode_index=4, max_gap_sec=0.2)

    assert [segment.start_frame for segment in segments] == [0, 2]
    assert [segment.end_frame_exclusive for segment in segments] == [2, 4]


@pytest.mark.optional
def test_clean_stutters_splits_segments_and_reindexes(tmp_path) -> None:
    root = tmp_path / "dataset"
    out = tmp_path / "dataset_stutter_clean"
    rows = [
        {
            "index": 30,
            "episode_index": 5,
            "frame_index": 0,
            "timestamp": 0.0,
            "session_state": "RUNNING",
            "control_hz": 15.0,
            "action": [0.0, 0.0],
            "observation.last_action": [float("nan"), float("nan")],
            "fault_codes": [],
        },
        {
            "index": 31,
            "episode_index": 5,
            "frame_index": 1,
            "timestamp": 0.1,
            "session_state": "RUNNING",
            "control_hz": 15.0,
            "action": [0.1, 0.0],
            "observation.last_action": [0.0, 0.0],
            "fault_codes": [],
        },
        {
            "index": 0,
            "episode_index": 5,
            "frame_index": 2,
            "timestamp": 0.5,
            "session_state": "RUNNING",
            "control_hz": 15.0,
            "action": [0.2, 0.0],
            "observation.last_action": [0.1, 0.0],
            "fault_codes": [],
        },
        {
            "index": 1,
            "episode_index": 5,
            "frame_index": 3,
            "timestamp": 0.6,
            "session_state": "RUNNING",
            "control_hz": 15.0,
            "action": [0.3, 0.0],
            "observation.last_action": [0.2, 0.0],
            "fault_codes": [],
        },
    ]
    episodes = [
        {
            "episode_index": 5,
            "episode_id": "episode_000005",
            "frame_count": 4,
            "parquet_path": "data/chunk-000/episode_000005.parquet",
            "metadata": {"operator": "test"},
            "summary": {},
        }
    ]
    _write_jsonl(root / "meta" / "episodes.jsonl", episodes)
    _write_jsonl(root / "meta" / "episodes_stats.jsonl", [{"episode_index": 5, "frame_count": 4}])
    (root / "meta" / "info.json").write_text(
        json.dumps({"dataset_name": "dataset", "chunk_size": 1000, "episode_count": 1, "frame_count": 4}),
        encoding="utf-8",
    )
    _write_parquet(root / "data/chunk-000/episode_000005.parquet", rows)

    plan = plan_stutter_clean(root, max_gap_sec=0.2, min_segment_frames=2)
    assert plan.output_episode_count == 2
    assert plan.output_frame_count == 4

    clean_stutters(root, out, max_gap_sec=0.2, min_segment_frames=2)

    pq = pytest.importorskip("pyarrow.parquet")
    first = pq.read_table(out / "data/chunk-000/episode_000000.parquet").to_pylist()
    second = pq.read_table(out / "data/chunk-000/episode_000001.parquet").to_pylist()
    assert [row["index"] for row in first] == [0, 1]
    assert [row["frame_index"] for row in first] == [0, 1]
    assert [row["episode_index"] for row in first] == [0, 0]
    assert [row["index"] for row in second] == [2, 3]
    assert [row["frame_index"] for row in second] == [0, 1]
    assert [row["episode_index"] for row in second] == [1, 1]
    assert all(np.isnan(value) for value in second[0]["observation.last_action"])

    updated_episodes = [
        json.loads(line)
        for line in (out / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["episode_index"] for record in updated_episodes] == [0, 1]
    assert updated_episodes[1]["metadata"]["source_episode_index"] == 5
    info = json.loads((out / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["dataset_name"] == "dataset_stutter_clean"
    assert info["episode_count"] == 2
    assert info["frame_count"] == 4


def test_detect_fault_nan_tail() -> None:
    rows = [
        {"frame_index": 0, "session_state": "RUNNING", "action": [0.0, 1.0]},
        {"frame_index": 1, "session_state": "FAULT", "action": [float("nan"), float("nan")]},
        {"frame_index": 2, "session_state": "FAULT", "action": [float("nan"), float("nan")]},
    ]

    tail = detect_fault_nan_tail(rows, episode_index=8)

    assert tail is not None
    assert tail.start_frame == 1
    assert tail.end_frame_exclusive == 3


def test_delete_episodes_copies_dataset_and_removes_selected_artifacts(tmp_path) -> None:
    root = tmp_path / "dataset"
    out = tmp_path / "dataset_without_ep1"
    episodes = [
        {
            "episode_index": 0,
            "frame_count": 2,
            "parquet_path": "data/chunk-000/episode_000000.parquet",
            "video_paths": {"front": "videos/front/chunk-000/episode_000000.mp4"},
        },
        {
            "episode_index": 1,
            "frame_count": 3,
            "parquet_path": "data/chunk-000/episode_000001.parquet",
            "video_paths": {"front": "videos/front/chunk-000/episode_000001.mp4"},
            "tactile_paths": {
                "left": "tactile/left/chunk-000/episode_000001/manifest.json"
            },
        },
    ]
    stats = [
        {"episode_index": 0, "frame_count": 2},
        {"episode_index": 1, "frame_count": 3},
    ]
    _write_jsonl(root / "meta" / "episodes.jsonl", episodes)
    _write_jsonl(root / "meta" / "episodes_stats.jsonl", stats)
    (root / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 2, "total_frames": 5}),
        encoding="utf-8",
    )
    for record in episodes:
        parquet = root / record["parquet_path"]
        parquet.parent.mkdir(parents=True, exist_ok=True)
        parquet.write_text("placeholder", encoding="utf-8")
        for video in record.get("video_paths", {}).values():
            path = root / video
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("video", encoding="utf-8")
        for tactile in record.get("tactile_paths", {}).values():
            path = root / tactile
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("manifest", encoding="utf-8")

    plan = plan_episode_deletion(root, "1")
    assert plan.deleted_indices == (1,)

    delete_episodes(root, out, "1")

    assert (root / "data/chunk-000/episode_000001.parquet").exists()
    assert not (out / "data/chunk-000/episode_000001.parquet").exists()
    assert not (out / "videos/front/chunk-000/episode_000001.mp4").exists()
    assert not (out / "tactile/left/chunk-000/episode_000001").exists()
    kept = [
        json.loads(line)
        for line in (out / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["episode_index"] for record in kept] == [0]
    info = json.loads((out / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["total_episodes"] == 1
    assert info["total_frames"] == 2


def test_split_episodes_writes_clean_and_extracted_datasets(tmp_path) -> None:
    root = tmp_path / "dataset"
    clean = tmp_path / "dataset_clean"
    extracted = tmp_path / "dataset_collision"
    episodes = [
        {
            "episode_index": 0,
            "frame_count": 2,
            "parquet_path": "data/chunk-000/episode_000000.parquet",
        },
        {
            "episode_index": 1,
            "frame_count": 3,
            "parquet_path": "data/chunk-000/episode_000001.parquet",
        },
        {
            "episode_index": 2,
            "frame_count": 4,
            "parquet_path": "data/chunk-000/episode_000002.parquet",
            "video_paths": {"front": "videos/front/chunk-000/episode_000002.mp4"},
        },
    ]
    stats = [{"episode_index": item["episode_index"], "frame_count": item["frame_count"]} for item in episodes]
    _write_jsonl(root / "meta" / "episodes.jsonl", episodes)
    _write_jsonl(root / "meta" / "episodes_stats.jsonl", stats)
    (root / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 3, "total_frames": 9}),
        encoding="utf-8",
    )
    for record in episodes:
        parquet = root / record["parquet_path"]
        parquet.parent.mkdir(parents=True, exist_ok=True)
        parquet.write_text("placeholder", encoding="utf-8")
        for video in record.get("video_paths", {}).values():
            path = root / video
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("video", encoding="utf-8")

    plan = plan_episode_split(root, "1-2")
    assert plan.extracted_indices == (1, 2)
    assert plan.clean_indices == (0,)

    split_episodes(
        root,
        clean_output_root=clean,
        extract_output_root=extracted,
        episode_indices="1-2",
    )

    assert (clean / "data/chunk-000/episode_000000.parquet").exists()
    assert not (clean / "data/chunk-000/episode_000001.parquet").exists()
    assert not (clean / "videos/front/chunk-000/episode_000002.mp4").exists()
    assert not (extracted / "data/chunk-000/episode_000000.parquet").exists()
    assert (extracted / "data/chunk-000/episode_000001.parquet").exists()
    assert (extracted / "videos/front/chunk-000/episode_000002.mp4").exists()
    clean_records = [
        json.loads(line)
        for line in (clean / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    extracted_records = [
        json.loads(line)
        for line in (extracted / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["episode_index"] for record in clean_records] == [0]
    assert [record["episode_index"] for record in extracted_records] == [1, 2]
    clean_info = json.loads((clean / "meta" / "info.json").read_text(encoding="utf-8"))
    extracted_info = json.loads((extracted / "meta" / "info.json").read_text(encoding="utf-8"))
    assert clean_info["total_episodes"] == 1
    assert clean_info["total_frames"] == 2
    assert extracted_info["total_episodes"] == 2
    assert extracted_info["total_frames"] == 7


@pytest.mark.optional
def test_label_force_collisions_updates_episode_and_stats_meta(tmp_path) -> None:
    root = tmp_path / "dataset"
    episodes = [
        {
            "episode_index": 0,
            "frame_count": 1,
            "parquet_path": "data/chunk-000/episode_000000.parquet",
            "summary": {},
        },
        {
            "episode_index": 1,
            "frame_count": 1,
            "parquet_path": "data/chunk-000/episode_000001.parquet",
            "summary": {},
        },
    ]
    stats = [
        {"episode_index": 0, "frame_count": 1},
        {"episode_index": 1, "frame_count": 1},
    ]
    _write_jsonl(root / "meta" / "episodes.jsonl", episodes)
    _write_jsonl(root / "meta" / "episodes_stats.jsonl", stats)
    (root / "meta" / "info.json").write_text(
        json.dumps({"nominal_fps": 15.0}),
        encoding="utf-8",
    )
    _write_parquet(
        root / "data/chunk-000/episode_000000.parquet",
        [
            {
                "timestamp": 1.0,
                "faults_json": json.dumps([]),
                "observation.arm.left.hardware_result_text": (
                    "RuntimeError: left: force 55.0000N > "
                    "hard_stop_force_n"
                ),
                "teleop.pose_quality.status": "warning",
                "teleop.pose_quality.score": 0.7,
            }
        ],
    )
    _write_parquet(
        root / "data/chunk-000/episode_000001.parquet",
        [
            {
                "timestamp": 2.0,
                "faults_json": json.dumps([{"message": "low_control_rate"}]),
                "teleop.pose_quality.status": "good",
                "teleop.pose_quality.score": 1.0,
            }
        ],
    )

    plan = plan_force_collision_labels(root)
    assert plan.detected_indices == (0,)
    assert plan.clean_indices == (1,)

    label_force_collisions(root)

    updated_episodes = [
        json.loads(line)
        for line in (root / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    updated_stats = [
        json.loads(line)
        for line in (root / "meta" / "episodes_stats.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert updated_episodes[0]["summary"]["force_collision_detected"] is True
    assert updated_episodes[0]["summary"]["force_collision_label"] == "detected_collision"
    assert updated_episodes[0]["summary"]["trajectory_quality"]["pose_quality_warning_count"] == 1
    assert updated_episodes[0]["metadata"]["force_collision_detected"] is True
    assert updated_episodes[0]["metadata"]["force_collision_label"] == "detected_collision"
    assert updated_episodes[0]["metadata"]["trajectory_quality"]["force_collision_label"] == "detected_collision"
    assert updated_stats[0]["force_collision_detected"] is True
    assert updated_stats[1]["force_collision_label"] == "clean"
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["postprocess_history"][-1]["tool"] == "utils.dataset_postprocess.label_force_collisions"


@pytest.mark.optional
def test_label_force_collisions_override_all_detected(tmp_path) -> None:
    root = tmp_path / "dataset"
    episodes = [
        {
            "episode_index": 7,
            "frame_count": 1,
            "parquet_path": "data/chunk-000/episode_000007.parquet",
            "summary": {},
        }
    ]
    _write_jsonl(root / "meta" / "episodes.jsonl", episodes)
    _write_jsonl(root / "meta" / "episodes_stats.jsonl", [{"episode_index": 7, "frame_count": 1}])
    (root / "meta" / "info.json").write_text(json.dumps({"nominal_fps": 15.0}), encoding="utf-8")
    _write_parquet(
        root / "data/chunk-000/episode_000007.parquet",
        [{"timestamp": 1.0, "faults_json": json.dumps([])}],
    )

    plan = plan_force_collision_labels(root, override="all-detected")
    assert plan.detected_indices == (7,)
    assert plan.clean_indices == ()

    label_force_collisions(root, override="all-detected")

    updated = json.loads((root / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert updated["summary"]["force_collision_detected"] is True
    assert updated["summary"]["trajectory_quality"]["force_collision_label"] == "detected_collision"
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["postprocess_history"][-1]["override"] == "all-detected"


@pytest.mark.optional
def test_label_force_collisions_preserves_terminal_episode_evidence(tmp_path) -> None:
    root = tmp_path / "dataset"
    episode = {
        "episode_index": 9,
        "frame_count": 1,
        "parquet_path": "data/chunk-000/episode_000009.parquet",
        "summary": {
            "stop_reason": "position_force_guard",
            "force_collision_detected": True,
            "force_collision_event": {
                "detected": True,
                "source": "position_force_guard",
            },
        },
    }
    _write_jsonl(root / "meta" / "episodes.jsonl", [episode])
    _write_jsonl(
        root / "meta" / "episodes_stats.jsonl",
        [{"episode_index": 9, "frame_count": 1}],
    )
    (root / "meta" / "info.json").write_text(
        json.dumps({"nominal_fps": 15.0}),
        encoding="utf-8",
    )
    _write_parquet(
        root / "data/chunk-000/episode_000009.parquet",
        [{"timestamp": 1.0, "faults_json": json.dumps([])}],
    )

    plan = plan_force_collision_labels(root)
    assert plan.detected_indices == (9,)
    assert plan.clean_indices == ()

    label_force_collisions(root)

    updated = json.loads(
        (root / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert updated["summary"]["force_collision_detected"] is True
    assert updated["summary"]["force_collision_label"] == "detected_collision"
    assert updated["summary"]["force_collision_event"]["source"] == "position_force_guard"


@pytest.mark.optional
def test_causal_align_dataset_uses_latest_non_future_refs(tmp_path) -> None:
    root = tmp_path / "dataset"
    out = tmp_path / "dataset_causal"
    episodes = [
        {
            "episode_index": 0,
            "frame_count": 2,
            "parquet_path": "data/chunk-000/episode_000000.parquet",
        }
    ]
    rows = [
        {
            "index": 0,
            "episode_index": 0,
            "frame_index": 0,
            "timestamp": 1.0,
            "observation.images.front.source_timestamp": 1.1,
            "observation.images.front.video_frame_index": 0,
            "observation.images.front.path": "videos/front/chunk-000/episode_000000.mp4",
            "observation.images.front.age_sec": 0.0,
            "observation.hand.left.tactile_timestamp": 1.1,
            "observation.hand.left.tactile_frame_index": 0,
            "observation.hand.left.tactile_path": "tactile/left/chunk-000/episode_000000/manifest.json",
        },
        {
            "index": 1,
            "episode_index": 0,
            "frame_index": 1,
            "timestamp": 1.2,
            "observation.images.front.source_timestamp": 1.3,
            "observation.images.front.video_frame_index": 1,
            "observation.images.front.path": "videos/front/chunk-000/episode_000000.mp4",
            "observation.images.front.age_sec": 0.0,
            "observation.hand.left.tactile_timestamp": 1.3,
            "observation.hand.left.tactile_frame_index": 1,
            "observation.hand.left.tactile_path": "tactile/left/chunk-000/episode_000000/manifest.json",
        },
    ]
    _write_jsonl(root / "meta" / "episodes.jsonl", episodes)
    _write_jsonl(root / "meta" / "episodes_stats.jsonl", [{"episode_index": 0, "frame_count": 2}])
    (root / "meta" / "info.json").write_text(json.dumps({"dataset_name": "dataset"}), encoding="utf-8")
    _write_parquet(root / "data/chunk-000/episode_000000.parquet", rows)

    plan = plan_causal_align(root, no_prior_policy="null")
    assert plan.current_future_count == 4
    assert plan.aligned_future_count == 0
    assert plan.no_prior_count == 2

    causal_align_dataset(root, out, no_prior_policy="null")

    pq = pytest.importorskip("pyarrow.parquet")
    updated = pq.read_table(out / "data/chunk-000/episode_000000.parquet").to_pylist()
    assert updated[0]["observation.images.front.video_frame_index"] is None
    assert updated[0]["observation.hand.left.tactile_frame_index"] is None
    assert updated[1]["observation.images.front.video_frame_index"] == 0
    assert updated[1]["observation.hand.left.tactile_frame_index"] == 0
    assert updated[1]["observation.images.front.age_sec"] == pytest.approx(0.1)
    info = json.loads((out / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["dataset_name"] == "dataset_causal"
    assert info["postprocess_history"][-1]["tool"] == "utils.dataset_postprocess.causal_align"


@pytest.mark.optional
def test_trim_nan_tails_copies_dataset_and_updates_meta(tmp_path) -> None:
    root = tmp_path / "dataset"
    out = tmp_path / "dataset_trimmed"
    rows = [
        {"frame_index": 0, "timestamp": 10.0, "session_state": "RUNNING", "action": [0.0, 1.0]},
        {"frame_index": 1, "timestamp": 10.1, "session_state": "RUNNING", "action": [0.1, 1.0]},
        {"frame_index": 2, "timestamp": 10.2, "session_state": "FAULT", "action": [float("nan"), float("nan")]},
        {"frame_index": 3, "timestamp": 10.3, "session_state": "FAULT", "action": [float("nan"), float("nan")]},
    ]
    episodes = [
        {
            "episode_index": 0,
            "frame_count": 4,
            "start_timestamp": 10.0,
            "end_timestamp": 10.3,
            "duration_sec": 0.3,
            "parquet_path": "data/chunk-000/episode_000000.parquet",
        }
    ]
    stats = [
        {
            "episode_index": 0,
            "frame_count": 4,
            "start_timestamp": 10.0,
            "end_timestamp": 10.3,
            "duration_sec": 0.3,
        }
    ]
    _write_jsonl(root / "meta" / "episodes.jsonl", episodes)
    _write_jsonl(root / "meta" / "episodes_stats.jsonl", stats)
    (root / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 1, "total_frames": 4}),
        encoding="utf-8",
    )
    _write_parquet(root / "data/chunk-000/episode_000000.parquet", rows)

    plan = trim_nan_tails(root, out)

    assert plan.total_trimmed_frames == 2
    original_pq = pytest.importorskip("pyarrow.parquet").read_table(
        root / "data/chunk-000/episode_000000.parquet"
    )
    trimmed_pq = pytest.importorskip("pyarrow.parquet").read_table(
        out / "data/chunk-000/episode_000000.parquet"
    )
    assert original_pq.num_rows == 4
    assert trimmed_pq.num_rows == 2
    updated_episode = json.loads(
        (out / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert updated_episode["frame_count"] == 2
    assert updated_episode["end_timestamp"] == pytest.approx(10.1)
    info = json.loads((out / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["total_frames"] == 2
