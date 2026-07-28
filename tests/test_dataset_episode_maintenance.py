from __future__ import annotations

import json
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from dataset_contracts.canonical_encoding import canonical_bytes
from dataset_contracts.control_trace_episode import (
    EPISODE_COMMIT_FILE_DOMAIN_V1,
    V24EpisodeCaptureCommit,
)
from dataset_contracts.episode_identity import EpisodeCaptureIdentity
from dataset_contracts.v25p import V25P_EPISODE_IDENTITY_FILE_DOMAIN_V1
from teleop_runtime.collection.dataset_episode_maintenance import delete_episode_and_compact
from teleop_runtime.collection.writer_v21 import LeRobotV21Writer, LeRobotV21WriterConfig
from teleop_runtime.gui.calibration_web_ui import CalibrationWebUi, _encode_http_event_payload
from teleop_runtime.real_teleop.gui_controls import (
    GuiControlEvents,
    GuiControlState,
    hold_controls,
    teleop_controls,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _dataset(root: Path) -> None:
    episodes: list[dict[str, object]] = []
    stats: list[dict[str, object]] = []
    global_index = 0
    frame_counts = (2, 3, 1)
    for episode_index, frame_count in enumerate(frame_counts):
        token = f"episode_{episode_index:06d}"
        parquet_relative = f"data/chunk-000/{token}.parquet"
        video_relative = f"videos/left_wrist/chunk-000/{token}.mp4"
        tactile_relative = f"tactile/left/chunk-000/{token}/manifest.json"
        parquet_path = root / parquet_relative
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "episode_index": [episode_index] * frame_count,
                    "frame_index": list(range(frame_count)),
                    "index": list(range(global_index, global_index + frame_count)),
                    "camera_path": [video_relative] * frame_count,
                }
            ),
            parquet_path,
        )
        video_path = root / video_relative
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(bytes([episode_index]))
        tactile_path = root / tactile_relative
        tactile_path.parent.mkdir(parents=True, exist_ok=True)
        tactile_path.write_text(
            json.dumps({"episode_index": episode_index, "path": tactile_relative}),
            encoding="utf-8",
        )
        episodes.append(
            {
                "episode_index": episode_index,
                "episode_id": token,
                "frame_count": frame_count,
                "parquet_path": parquet_relative,
                "video_paths": {"left_wrist": video_relative},
                "tactile_paths": {"left": tactile_relative},
                "metadata": {},
                "summary": {},
            }
        )
        stats.append({"episode_index": episode_index, "frame_count": frame_count})
        global_index += frame_count
    _write_jsonl(root / "meta" / "episodes.jsonl", episodes)
    _write_jsonl(root / "meta" / "episodes_stats.jsonl", stats)
    (root / "meta" / "info.json").write_text(
        json.dumps({"episode_count": 3, "frame_count": 6, "chunk_size": 1000}),
        encoding="utf-8",
    )


def test_delete_middle_episode_compacts_artifacts_metadata_and_parquet(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _dataset(root)

    result = delete_episode_and_compact(root, 1)

    assert result.episode_count == 2
    assert result.frame_count == 3
    assert result.next_episode_index == 2
    assert result.reindexed == {0: 0, 2: 1}
    assert result.backup_root.is_dir()
    rows = [json.loads(line) for line in (root / "meta" / "episodes.jsonl").read_text().splitlines()]
    assert [row["episode_index"] for row in rows] == [0, 1]
    assert [row["frame_count"] for row in rows] == [2, 1]
    assert rows[1]["episode_id"] == "episode_000001"
    assert not (root / "data/chunk-000/episode_000002.parquet").exists()
    compacted = pq.read_table(root / "data/chunk-000/episode_000001.parquet").to_pydict()
    assert compacted["episode_index"] == [1]
    assert compacted["index"] == [2]
    assert compacted["camera_path"] == [
        "videos/left_wrist/chunk-000/episode_000001.mp4"
    ]
    assert (root / "videos/left_wrist/chunk-000/episode_000001.mp4").read_bytes() == b"\x02"
    manifest = json.loads(
        (root / "tactile/left/chunk-000/episode_000001/manifest.json").read_text()
    )
    assert manifest["episode_index"] == 1
    assert "episode_000001" in manifest["path"]
    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["episode_count"] == 2
    assert info["frame_count"] == 3
    assert len((root / "meta" / "episodes.jsonl").read_text().splitlines()) == 2
    assert len((result.backup_root / "meta" / "episodes.jsonl").read_text().splitlines()) == 3
    writer = LeRobotV21Writer(
        LeRobotV21WriterConfig(
            root_dir=root,
            nominal_fps=15.0,
            task_id="continuation-test",
            task_description="continuation-test",
            write_parquet=True,
            write_video=False,
        )
    )
    assert writer.next_episode_index == 2
    assert writer.next_sample_index == 3
    assert writer.start_episode() == 2
    assert writer.discard_episode(2)


def test_missing_episode_does_not_replace_original_dataset(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _dataset(root)
    original = (root / "meta" / "episodes.jsonl").read_bytes()

    with pytest.raises(IndexError, match="does not exist"):
        delete_episode_and_compact(root, 9)

    assert (root / "meta" / "episodes.jsonl").read_bytes() == original
    assert not list(tmp_path.glob("dataset.before_delete_episode_*"))


def test_gui_delete_payload_requires_non_negative_integer() -> None:
    assert _encode_http_event_payload(
        {"event": "episode_delete", "episode_index": "003"}
    ) == "episode_delete:3"
    assert _encode_http_event_payload(
        {"event": "episode_delete", "episode_index": "-1"}
    ) == ""
    state = GuiControlState(
        teleop_started=True,
        episode_recording_available=True,
        active_side_count=2,
    )
    events = GuiControlEvents(
        episode_events=("episode_start", "episode_stop", "episode_delete")
    )
    assert "episode_delete" not in teleop_controls(state, events)
    assert "episode_delete" in hold_controls(
        GuiControlState(
            teleop_started=True,
            hold_active=True,
            episode_recording_available=True,
            active_side_count=2,
        ),
        events,
    )


def test_gui_accepts_only_one_delete_until_runtime_republishes_idle_state() -> None:
    ui = CalibrationWebUi(open_browser=False)
    ui.publish(
        active_controls=["q", "episode_start", "episode_delete"],
        episode_event_status="idle",
    )

    assert ui.push_event("episode_delete:1") is True
    assert ui.push_event("episode_delete:1") is False
    snapshot = ui.snapshot()
    assert snapshot["episode_event_status"] == "delete_queued"
    assert "episode_delete" not in snapshot["active_controls"]
    assert "episode_start" not in snapshot["active_controls"]
    assert ui.consume_events() == ["episode_delete:1"]


def test_gui_start_and_delete_requests_are_mutually_exclusive_until_republish() -> None:
    ui = CalibrationWebUi(open_browser=False)
    ui.publish(
        active_controls=["q", "episode_start", "episode_delete"],
        episode_event_status="idle",
    )

    assert ui.push_event("episode_start") is True
    assert ui.push_event("episode_delete:1") is False
    snapshot = ui.snapshot()
    assert snapshot["episode_event_status"] == "preparing"
    assert "episode_start" not in snapshot["active_controls"]
    assert "episode_delete" not in snapshot["active_controls"]
    assert ui.consume_events() == ["episode_start"]


def test_gui_device_retry_is_one_shot_until_runtime_republishes_controls() -> None:
    ui = CalibrationWebUi(open_browser=False)
    ui.publish(active_controls=["q", "device_retry"])

    assert ui.push_event("device_retry") is True
    assert ui.push_event("device_retry") is False
    assert "device_retry" not in ui.snapshot()["active_controls"]
    assert ui.consume_events() == ["device_retry"]

    ui.publish(active_controls=["q", "device_retry"])
    assert ui.push_event("device_retry") is True


def test_gui_event_consumption_is_bounded_per_control_poll() -> None:
    ui = CalibrationWebUi(open_browser=False)
    ui.publish(active_controls=["movement_multiple_up"])
    for _ in range(65):
        assert ui.push_event("movement_multiple_up") is True

    assert len(ui.consume_events()) == 64
    assert ui.consume_events() == ["movement_multiple_up"]


def test_committed_v24_episode_compaction_fails_closed_without_mutation(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _dataset(root)
    episodes_path = root / "meta" / "episodes.jsonl"
    rows = [json.loads(line) for line in episodes_path.read_text().splitlines()]
    trace_root = root / "meta/v24/control_trace/episodes"
    for row in rows:
        episode_index = int(row["episode_index"])
        episode_uuid = f"trace-episode-{episode_index}"
        row["metadata"]["control_trace_episode"] = {"episode_uuid": episode_uuid}
        commit = V24EpisodeCaptureCommit(
            commit_id=f"commit-{episode_index}",
            episode_uuid=episode_uuid,
            legacy_episode_index=episode_index,
            legacy_schema="dexteleop_lerobot_v2.3",
            legacy_finalize_status="finalized",
            trace_seal_assessment_id=f"seal-{episode_index}",
            trace_seal_hash="0" * 64,
            control_trace_seal_status="sealed",
            capabilities=("control_trace",),
            committed_at=1.0 + episode_index,
            committed_clock_domain_id="local_wall_clock",
        )
        commit_path = trace_root / episode_uuid / "capture_commit.cjson"
        commit_path.parent.mkdir(parents=True, exist_ok=True)
        commit_path.write_bytes(
            canonical_bytes(commit.to_payload(), domain=EPISODE_COMMIT_FILE_DOMAIN_V1)
        )
    _write_jsonl(episodes_path, rows)

    episodes_before = episodes_path.read_bytes()
    commits_before = {
        path: path.read_bytes() for path in sorted(trace_root.glob("*/capture_commit.cjson"))
    }

    with pytest.raises(RuntimeError, match="committed v2.4 capture"):
        delete_episode_and_compact(root, 1)

    assert episodes_path.read_bytes() == episodes_before
    assert all(path.read_bytes() == payload for path, payload in commits_before.items())
    assert not list(tmp_path.glob("dataset.before_delete_episode_*"))


def test_v25p_episode_compaction_fails_closed_without_mutation(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _dataset(root)
    episodes_path = root / "meta" / "episodes.jsonl"
    identity = EpisodeCaptureIdentity(
        episode_uuid="v25p-episode-1",
        legacy_episode_index=1,
        lifecycle_owner_id="test.owner",
        prepared_at=1.0,
        prepared_clock_domain_id="test_clock",
    )
    identity_path = (
        root
        / "meta"
        / "v25p"
        / "episodes"
        / identity.episode_uuid
        / "identity.cjson"
    )
    identity_path.parent.mkdir(parents=True, exist_ok=True)
    identity_path.write_bytes(
        canonical_bytes(
            identity.to_payload(),
            domain=V25P_EPISODE_IDENTITY_FILE_DOMAIN_V1,
        )
    )
    episodes_before = episodes_path.read_bytes()
    identity_before = identity_path.read_bytes()

    with pytest.raises(RuntimeError, match="v2.5-p capture"):
        delete_episode_and_compact(root, 1)

    assert episodes_path.read_bytes() == episodes_before
    assert identity_path.read_bytes() == identity_before
    assert not list(tmp_path.glob("dataset.before_delete_episode_*"))
