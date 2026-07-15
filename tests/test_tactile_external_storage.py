from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from teleop_runtime.collection.build_collect_components import CollectorComponents
from teleop_runtime.collection.sample_assembler import (
    CollectorSample,
    CollectorSampleBuilder,
    CollectorSampleBuilderConfig,
)
from teleop_runtime.collection.schema_v21 import collector_sample_to_row
from teleop_runtime.collection.writer_v21 import (
    LeRobotV21Writer,
    LeRobotV21WriterConfig,
    ROW_JOB_KIND,
    TACTILE_FRAME_JOB_KIND,
)
from teleop_runtime.diagnostics import TeleopRunState
from teleop_runtime.replay.media import ExternalTactileStore
from teleop_runtime.session import SessionStepResult


def _sample(*, frame_index: int = 0) -> CollectorSample:
    inner = np.arange(6, dtype=np.float32).reshape(2, 3, 1)
    outer = (np.arange(6, dtype=np.float32) + 10.0).reshape(2, 3, 1)
    tactile = np.concatenate((inner.reshape(-1), outer.reshape(-1))).astype(np.float32)
    return CollectorSample(
        index=frame_index,
        episode_index=0,
        frame_index=frame_index,
        timestamp=10.0 + frame_index,
        session_state="running",
        control_hz=30.0,
        action=np.zeros(26, dtype=np.float32),
        observation_state=np.zeros(38, dtype=np.float32),
        observation_last_action=np.zeros(26, dtype=np.float32),
        arm_state_source="unit",
        active_sides=("left",),
        hand_states={
            "left": {
                "tactile": tactile,
                "tactile_dict": {"L_IN": inner, "L_OUT": outer},
                "tactile_timestamp": 9.9 + frame_index,
                "telemetry_timestamp": 9.95 + frame_index,
                "tactile_regions": ("L_IN", "L_OUT"),
                "tactile_devices": ("L_IN", "L_OUT"),
                "tactile_frame_ids": {"L_IN": 101 + frame_index, "L_OUT": 201 + frame_index},
                "tactile_shapes": {"L_IN": inner.shape, "L_OUT": outer.shape},
                "tactile_version": 7 + frame_index,
                "tactile_raw_shape_per_sensor": (288, 384, 3),
            },
            "right": {},
        },
    )


def _raw_tactile_sample(*, frame_index: int = 0) -> CollectorSample:
    raw = np.arange(12, dtype=np.uint8).reshape(3, 4)
    return CollectorSample(
        index=frame_index,
        episode_index=0,
        frame_index=frame_index,
        timestamp=10.0 + frame_index,
        session_state="running",
        control_hz=30.0,
        action=np.zeros(26, dtype=np.float32),
        observation_state=np.zeros(38, dtype=np.float32),
        observation_last_action=np.zeros(26, dtype=np.float32),
        arm_state_source="unit",
        active_sides=("left",),
        hand_states={
            "left": {
                "tactile": None,
                "tactile_dict": {"M_RAW": raw},
                "tactile_timestamp": 9.9 + frame_index,
                "telemetry_timestamp": 9.95 + frame_index,
                "tactile_regions": ("M_RAW",),
                "tactile_devices": ("M_RAW",),
                "tactile_frame_ids": {"M_RAW": 11 + frame_index},
                "tactile_shapes": {"M_RAW": raw.shape},
                "tactile_field_by_key": {"M_RAW": "raw"},
                "tactile_fields": ("raw",),
                "tactile_version": 3 + frame_index,
                "tactile_raw_shape_per_sensor": raw.shape,
            },
            "right": {},
        },
    )


class _VersionedTactileBackend:
    def __init__(self) -> None:
        self.version = 1
        self.calls: list[tuple[bool, bool]] = []

    def get_state(
        self,
        *,
        include_tactile_arrays: bool = True,
        include_tactile_vector: bool = True,
    ) -> dict[str, object]:
        self.calls.append((bool(include_tactile_arrays), bool(include_tactile_vector)))
        array = np.full((2, 3, 1), float(self.version), dtype=np.float32)
        state: dict[str, object] = {
            "joint_pos": np.zeros(6, dtype=np.float32),
            "tactile_timestamp": 10.0 + float(self.version),
            "telemetry_timestamp": 10.1 + float(self.version),
            "tactile_devices": ("S0",),
            "tactile_frame_ids": {"S0": int(self.version)},
            "tactile_shapes": {"S0": array.shape},
            "tactile_version": int(self.version),
        }
        if include_tactile_arrays:
            state["tactile_dict"] = {"S0": array}
            state["tactile"] = array.reshape(-1).copy() if include_tactile_vector else None
        else:
            state["tactile_dict"] = None
            state["tactile"] = None
        return state


def _step(timestamp: float = 20.0) -> SessionStepResult:
    return SessionStepResult(
        timestamp=float(timestamp),
        state=TeleopRunState.RUNNING,
        control_hz=15.0,
    )


class TactileExternalStorageTest(unittest.TestCase):
    def test_writer_defaults_to_external_tactile_storage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LeRobotV21Writer(
                LeRobotV21WriterConfig(
                    root_dir=tmpdir,
                    write_video=False,
                    write_parquet=False,
                )
            )

            jobs = writer.plan_sample_jobs(_sample())
            row_job = next(job for job in jobs if job.kind == ROW_JOB_KIND)

            self.assertIsNone(row_job.payload["row"]["observation.hand.left.tactile"])
            self.assertEqual(
                row_job.payload["row"]["observation.hand.left.tactile_storage_format"],
                "binary_stream_v1",
            )
            self.assertTrue(any(job.kind == TACTILE_FRAME_JOB_KIND for job in jobs))

    def test_default_schema_keeps_inline_tactile_vector(self) -> None:
        sample = _sample()

        row, _camera_expectations = collector_sample_to_row(sample)

        self.assertEqual(row["observation.hand.left.tactile"], sample.hand_states["left"]["tactile"].tolist())
        self.assertEqual(row["observation.hand.left.tactile_storage_format"], "parquet_inline")
        self.assertIsNone(row["observation.hand.left.tactile_path"])
        self.assertEqual(row["observation.hand.left.tactile_sensor_keys"], ["L_IN", "L_OUT"])
        self.assertEqual(
            json.loads(row["observation.hand.left.tactile_shapes_json"]),
            {"L_IN": [2, 3, 1], "L_OUT": [2, 3, 1]},
        )

    def test_external_writer_streams_tactile_arrays_outside_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LeRobotV21Writer(
                LeRobotV21WriterConfig(
                    root_dir=tmpdir,
                    write_video=False,
                    write_parquet=False,
                    tactile_storage="external",
                    tactile_compression="none",
                )
            )
            writer.start_episode(episode_index=0)
            jobs = writer.plan_sample_jobs(_sample())
            row_job = next(job for job in jobs if job.kind == ROW_JOB_KIND)
            tactile_jobs = [job for job in jobs if job.kind == TACTILE_FRAME_JOB_KIND]

            self.assertEqual(len(tactile_jobs), 1)
            planned_row = row_job.payload["row"]
            self.assertIsNone(planned_row["observation.hand.left.tactile"])
            self.assertEqual(
                planned_row["observation.hand.left.tactile_storage_format"],
                "binary_stream_v1",
            )
            self.assertTrue(planned_row["observation.hand.left.tactile_path"].endswith("manifest.json"))
            self.assertIsNone(planned_row["observation.hand.left.tactile_frame_index"])
            self.assertEqual(writer.tactile_frame_counts(0), {})

            writer.handle_job(tactile_jobs[0])
            writer.handle_job(row_job)
            row = writer._episodes[0].rows_by_frame[0]
            self.assertEqual(row["observation.hand.left.tactile_frame_index"], 0)
            self.assertEqual(row["observation.hand.left.tactile_sensor_keys"], ["L_IN", "L_OUT"])
            self.assertEqual(writer.tactile_frame_counts(0), {"left": 1})

            writer.finalize_episode(0)
            self.assertEqual(writer.tactile_frame_counts(0), {})

            manifest_path = (
                Path(tmpdir)
                / "tactile"
                / "observation.hand.left.tactile"
                / "chunk-000"
                / "episode_000000"
                / "manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["format"], "binary_stream_v1")
            self.assertEqual(manifest["frame_count"], 1)
            self.assertEqual(manifest["sensor_keys"], ["L_IN", "L_OUT"])
            self.assertEqual(manifest["streams"]["L_IN"]["shape"], [2, 3, 1])
            inner_path = Path(tmpdir) / manifest["streams"]["L_IN"]["path"]
            np.testing.assert_allclose(
                np.fromfile(inner_path, dtype=np.float32),
                np.arange(6, dtype=np.float32),
            )

            episodes_path = Path(tmpdir) / "meta" / "episodes.jsonl"
            episode_record = json.loads(episodes_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(
                episode_record["tactile_paths"]["left"],
                "tactile/observation.hand.left.tactile/chunk-000/episode_000000/manifest.json",
            )

    def test_external_writer_preserves_raw_tactile_uint8_dtype(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LeRobotV21Writer(
                LeRobotV21WriterConfig(
                    root_dir=tmpdir,
                    write_video=False,
                    write_parquet=False,
                    tactile_storage="external",
                    tactile_compression="none",
                )
            )
            writer.start_episode(episode_index=0)
            for job in writer.plan_sample_jobs(_raw_tactile_sample()):
                writer.handle_job(job)
            writer.finalize_episode(0)

            manifest_path = (
                Path(tmpdir)
                / "tactile"
                / "observation.hand.left.tactile"
                / "chunk-000"
                / "episode_000000"
                / "manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            stream = manifest["streams"]["M_RAW"]
            self.assertEqual(stream["dtype"], "uint8")
            self.assertEqual(stream["shape"], [3, 4])
            self.assertEqual(manifest["frames"][0]["field_by_key"], {"M_RAW": "raw"})
            raw_path = Path(tmpdir) / stream["path"]
            np.testing.assert_array_equal(
                np.fromfile(raw_path, dtype=np.uint8).reshape(3, 4),
                np.arange(12, dtype=np.uint8).reshape(3, 4),
            )

    def test_external_writer_defaults_to_uncompressed_tactile_streams(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LeRobotV21Writer(
                LeRobotV21WriterConfig(
                    root_dir=tmpdir,
                    write_video=False,
                    write_parquet=False,
                    tactile_storage="external",
                )
            )
            writer.start_episode(episode_index=0)
            for frame_index in range(2):
                for job in writer.plan_sample_jobs(_raw_tactile_sample(frame_index=frame_index)):
                    writer.handle_job(job)
            writer.finalize_episode(0)

            manifest_path = (
                Path(tmpdir)
                / "tactile"
                / "observation.hand.left.tactile"
                / "chunk-000"
                / "episode_000000"
                / "manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["format"], "binary_stream_v1")
            stream = manifest["streams"]["M_RAW"]
            self.assertEqual(stream["compression"], "none")
            self.assertEqual(stream["encoding"], "raw_c_order")
            self.assertTrue(stream["path"].endswith(".uint8.bin"))
            self.assertEqual(stream["frame_count"], 2)
            self.assertEqual(stream["chunks"], [])
            raw_total = 2 * 12
            self.assertEqual((Path(tmpdir) / stream["path"]).stat().st_size, raw_total)

            store = ExternalTactileStore(dataset_root=Path(tmpdir), manifest_path=manifest_path)
            frame = store.frame(tactile_frame_index=1)
            self.assertTrue(frame.available)
            np.testing.assert_array_equal(
                frame.arrays["M_RAW"],
                np.arange(12, dtype=np.float32).reshape(3, 4),
            )

    def test_external_writer_can_use_zstd_tactile_streams(self) -> None:
        try:
            import zstandard  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("zstandard is not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LeRobotV21Writer(
                LeRobotV21WriterConfig(
                    root_dir=tmpdir,
                    write_video=False,
                    write_parquet=False,
                    tactile_storage="external",
                    tactile_compression="zstd",
                )
            )
            writer.start_episode(episode_index=0)
            for job in writer.plan_sample_jobs(_raw_tactile_sample()):
                writer.handle_job(job)
            writer.finalize_episode(0)

            manifest_path = (
                Path(tmpdir)
                / "tactile"
                / "observation.hand.left.tactile"
                / "chunk-000"
                / "episode_000000"
                / "manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            stream = manifest["streams"]["M_RAW"]
            self.assertEqual(stream["compression"], "zstd")
            self.assertEqual(stream["encoding"], "chunked_zstd")
            self.assertTrue(stream["path"].endswith(".uint8.bin.zst"))

            store = ExternalTactileStore(dataset_root=Path(tmpdir), manifest_path=manifest_path)
            frame = store.frame(tactile_frame_index=0)
            self.assertTrue(frame.available)
            np.testing.assert_array_equal(
                frame.arrays["M_RAW"],
                np.arange(12, dtype=np.float32).reshape(3, 4),
            )

    def test_external_writer_parquet_keeps_only_tactile_reference(self) -> None:
        try:
            import pyarrow.parquet as pq
        except ModuleNotFoundError:
            self.skipTest("pyarrow is not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LeRobotV21Writer(
                LeRobotV21WriterConfig(
                    root_dir=tmpdir,
                    write_video=False,
                    write_parquet=True,
                    tactile_storage="external",
                    tactile_compression="none",
                )
            )
            writer.start_episode(episode_index=0)
            for job in writer.plan_sample_jobs(_sample()):
                writer.handle_job(job)

            writer.finalize_episode(0)

            parquet_path = Path(tmpdir) / "data" / "chunk-000" / "episode_000000.parquet"
            row = pq.read_table(parquet_path).to_pylist()[0]
            self.assertIsNone(row["observation.hand.left.tactile"])
            self.assertEqual(row["observation.hand.left.tactile_storage_format"], "binary_stream_v1")
            self.assertEqual(row["observation.hand.left.tactile_frame_index"], 0)
            self.assertTrue(row["observation.hand.left.tactile_path"].endswith("manifest.json"))

    def test_sample_builder_only_copies_tactile_arrays_on_version_change(self) -> None:
        backend = _VersionedTactileBackend()
        components = CollectorComponents(
            session=None,
            backends={"left_hand": backend},
        )
        builder = CollectorSampleBuilder(
            CollectorSampleBuilderConfig(
                active_sides=("left",),
                tactile_array_snapshot_policy="on_change",
            )
        )

        first = builder(_step(20.0), components)
        second = builder(_step(20.1), components)
        backend.version = 2
        third = builder(_step(20.2), components)

        self.assertIn("S0", first.hand_states["left"]["tactile_dict"])
        self.assertIsNone(second.hand_states["left"]["tactile_dict"])
        self.assertIn("S0", third.hand_states["left"]["tactile_dict"])
        self.assertEqual(
            backend.calls,
            [
                (False, False),
                (True, False),
                (False, False),
                (False, False),
                (True, False),
            ],
        )

    def test_external_writer_reuses_latest_tactile_frame_for_sparse_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LeRobotV21Writer(
                LeRobotV21WriterConfig(
                    root_dir=tmpdir,
                    write_video=False,
                    write_parquet=False,
                    tactile_storage="external",
                    tactile_compression="none",
                )
            )
            writer.start_episode(episode_index=0)
            for job in writer.plan_sample_jobs(_sample(frame_index=0)):
                writer.handle_job(job)

            sparse_sample = _sample(frame_index=1)
            sparse_sample.hand_states["left"]["tactile"] = None
            sparse_sample.hand_states["left"]["tactile_dict"] = None
            jobs = writer.plan_sample_jobs(sparse_sample)
            self.assertFalse(any(job.kind == TACTILE_FRAME_JOB_KIND for job in jobs))
            for job in jobs:
                writer.handle_job(job)

            row = writer._episodes[0].rows_by_frame[1]
            self.assertEqual(row["observation.hand.left.tactile_storage_format"], "binary_stream_v1")
            self.assertEqual(row["observation.hand.left.tactile_frame_index"], 0)
            writer.finalize_episode(0)

            manifest_path = (
                Path(tmpdir)
                / "tactile"
                / "observation.hand.left.tactile"
                / "chunk-000"
                / "episode_000000"
                / "manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["frame_count"], 1)


if __name__ == "__main__":
    unittest.main()
