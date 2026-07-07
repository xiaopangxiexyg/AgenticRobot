from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from teleop_runtime.replay.dataset import ReplayDatasetIndex
from teleop_runtime.replay.media import TactileEpisodeReader, tactile_frame_jpeg
from teleop_runtime.replay.session import ReplayGuiSession
from teleop_runtime.replay.web_gui import _should_log_http_message


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _build_dataset(root: Path) -> np.ndarray:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise unittest.SkipTest("pyarrow is not installed") from exc

    _write_json(root / "meta" / "info.json", {"chunk_size": 1000, "nominal_fps": 10.0})
    tactile_rel = "tactile/observation.hand.left.tactile/chunk-000/episode_000000/manifest.json"
    video_rel = "videos/chunk-000/episode_000000/front.mp4"
    rows = []
    for index in range(2):
        rows.append(
            {
                "frame_index": index,
                "timestamp": 100.0 + float(index) * 0.1,
                "control_hz": 10.0,
                "session_state": "running",
                "observation.images.front.path": video_rel,
                "observation.images.front.video_frame_index": index,
                "observation.hand.left.tactile_storage_format": "binary_stream_v1",
                "observation.hand.left.tactile_path": tactile_rel,
                "observation.hand.left.tactile_frame_index": index,
                "observation.hand.left.tactile_sensor_keys": ["L_IN"],
            }
        )
    parquet_path = root / "data" / "chunk-000" / "episode_000000.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)

    tactile_dir = root / "tactile" / "observation.hand.left.tactile" / "chunk-000" / "episode_000000"
    tactile_dir.mkdir(parents=True, exist_ok=True)
    tactile = np.stack(
        [
            np.arange(18, dtype=np.float32).reshape(2, 3, 3),
            (np.arange(18, dtype=np.float32) + 100.0).reshape(2, 3, 3),
        ],
        axis=0,
    )
    stream_path = tactile_dir / "L_IN.float32.bin"
    tactile.tofile(stream_path)
    _write_json(
        root / tactile_rel,
        {
            "format": "binary_stream_v1",
            "side": "left",
            "relative_dir": "tactile/observation.hand.left.tactile/chunk-000/episode_000000",
            "frame_count": 2,
            "sensor_keys": ["L_IN"],
            "streams": {
                "L_IN": {
                    "path": str(stream_path.relative_to(root)),
                    "dtype": "float32",
                    "shape": [2, 3, 3],
                    "frame_count": 2,
                }
            },
            "frames": [
                {"sample_frame_index": 0, "tactile_frame_index": 0},
                {"sample_frame_index": 1, "tactile_frame_index": 1},
            ],
        },
    )
    _append_jsonl(
        root / "meta" / "episodes.jsonl",
        {
            "episode_index": 0,
            "episode_id": "episode_000000",
            "frame_count": 2,
            "duration_sec": 0.1,
            "parquet_path": "data/chunk-000/episode_000000.parquet",
            "video_paths": {"front": video_rel},
            "tactile_paths": {"left": tactile_rel},
            "metadata": {"nominal_fps": 10.0},
        },
    )
    _append_jsonl(
        root / "meta" / "episodes_stats.jsonl",
        {
            "episode_index": 0,
            "frame_count": 2,
            "duration_sec": 0.1,
            "average_hz": 10.0,
            "camera_keys": ["front"],
        },
    )
    return tactile


class ReplayGuiDatasetTest(unittest.TestCase):
    def test_replay_gui_suppresses_successful_access_logs_by_default(self) -> None:
        self.assertFalse(
            _should_log_http_message(
                False,
                ('GET /media/camera/front.jpg?frame=0 HTTP/1.1', "200", "-"),
            )
        )
        self.assertFalse(
            _should_log_http_message(False, ('GET /api/state HTTP/1.1', "200", "-"))
        )
        self.assertTrue(
            _should_log_http_message(False, ('GET /missing HTTP/1.1', "404", "-"))
        )
        self.assertTrue(
            _should_log_http_message(
                True,
                ('GET /media/camera/front.jpg?frame=0 HTTP/1.1', "200", "-"),
            )
        )

    def test_dataset_index_and_external_tactile_reader(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tactile = _build_dataset(root)

            index = ReplayDatasetIndex(root)
            episodes = index.available_episodes()
            self.assertEqual(len(episodes), 1)
            self.assertEqual(episodes[0].episode_id, "episode_000000")
            self.assertEqual(episodes[0].camera_keys, ("front",))
            self.assertEqual(episodes[0].tactile_sides, ("left",))

            loaded = index.load_episode("0")
            self.assertEqual(loaded.frame_count, 2)
            self.assertEqual(loaded.summary.camera_keys, ("front",))
            self.assertEqual(loaded.summary.tactile_sides, ("left",))
            self.assertAlmostEqual(loaded.nominal_hz, 10.0, places=5)

            reader = TactileEpisodeReader(loaded, "left")
            frame = reader.frame_for(loaded.rows[1], replay_index=1)
            self.assertTrue(frame.available)
            np.testing.assert_allclose(frame.arrays["L_IN"], tactile[1])
            jpeg = tactile_frame_jpeg(frame)
            self.assertTrue(jpeg.startswith(b"\xff\xd8"))

    def test_missing_episode_error_lists_available_indices(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_json(root / "meta" / "info.json", {"chunk_size": 1000})
            parquet_path = root / "data" / "chunk-000" / "episode_000010.parquet"
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            parquet_path.write_bytes(b"")
            _append_jsonl(
                root / "meta" / "episodes.jsonl",
                {
                    "episode_index": 10,
                    "episode_id": "episode_000010",
                    "parquet_path": "data/chunk-000/episode_000010.parquet",
                },
            )

            index = ReplayDatasetIndex(root)
            with self.assertRaises(FileNotFoundError) as context:
                index.load_episode("0")

            message = str(context.exception)
            self.assertIn("Could not resolve replay episode '0'", message)
            self.assertIn("Available episode indices: 10", message)

    def test_replay_session_seek_step_and_media_placeholders(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _build_dataset(root)

            session = ReplayGuiSession(root)
            state = session.state()
            self.assertEqual(state["frame_count"], 2)
            self.assertEqual(state["camera_keys"], ["front"])
            self.assertEqual(state["tactile_sides"], ["left"])

            state = session.seek(1)
            self.assertEqual(state["frame_index"], 1)
            state = session.step(-1)
            self.assertEqual(state["frame_index"], 0)
            state = session.dispatch({"type": "speed", "speed": 2.0})
            self.assertEqual(state["speed"], 2.0)

            self.assertTrue(session.camera_jpeg("front").startswith(b"\xff\xd8"))
            self.assertTrue(session.tactile_jpeg("left").startswith(b"\xff\xd8"))
            session.close()

    def test_preflight_dispatch_auto_configures_and_connects(self) -> None:
        class _FakeEpisode:
            frame_count = 0
            rows = []
            frame_times_sec = []
            summary = type("_Summary", (), {"camera_keys": (), "tactile_sides": ()})()

            def to_dict(self):
                return {"episode_id": "episode_fake"}

        class _FakeExecutor:
            def __init__(self) -> None:
                self.connected = False
                self.calls: list[str] = []

            def configure_from_payload(self, payload):
                self.calls.append(f"configure:{payload.get('active_sides')}")
                self.connected = False
                return self.status()

            def connect(self):
                self.calls.append("connect")
                self.connected = True
                return {"last_message": "hardware connected"}

            def preflight(self, episode):
                self.calls.append(f"preflight:{episode.to_dict()['episode_id']}")
                return {
                    "preflight": {
                        "ok": True,
                        "checks": [{"name": "fake_check", "ok": True, "level": "info"}],
                    }
                }

            def clear_preflight(self):
                self.calls.append("clear")
                return self.status()

            def close(self):
                self.calls.append("close")

            def status(self):
                return {
                    "state": "connected" if self.connected else "configured",
                    "connected": self.connected,
                    "settings": {},
                    "hardware_config": {},
                    "frame_index": 0,
                    "group": None,
                    "preflight": None,
                    "qpos_init": {},
                    "last_error": "",
                    "last_message": "",
                    "worker_alive": False,
                }

        session = ReplayGuiSession()
        fake = _FakeExecutor()
        try:
            session.executor = fake  # type: ignore[assignment]
            session.episode = _FakeEpisode()  # type: ignore[assignment]

            state = session.dispatch(
                {
                    "type": "preflight_hardware",
                    "active_sides": "right",
                    "connect_hardware": False,
                }
            )

            self.assertEqual(fake.calls, ["configure:right", "connect", "preflight:episode_fake"])
            self.assertTrue(any("hardware preflight ok" in message for message in state["messages"]))
        finally:
            session.close()


if __name__ == "__main__":
    unittest.main()
