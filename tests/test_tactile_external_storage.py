from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from collection_runtime.tactile_recompression import recompress_tactile_zstd_in_place
from dataset_interface.modality_artifacts import (
    DatasetCaptureWritableError,
    PAYLOAD_HASH_REQUIRE_SEALED,
    build_modality_artifact_inventory,
)

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


def _sample(*, frame_index: int = 0, episode_index: int = 0) -> CollectorSample:
    inner = np.arange(6, dtype=np.float32).reshape(2, 3, 1)
    outer = (np.arange(6, dtype=np.float32) + 10.0).reshape(2, 3, 1)
    tactile = np.concatenate((inner.reshape(-1), outer.reshape(-1))).astype(np.float32)
    return CollectorSample(
        index=frame_index,
        episode_index=episode_index,
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


def _raw_tactile_sample(
    *,
    frame_index: int = 0,
    episode_index: int = 0,
) -> CollectorSample:
    raw = np.arange(12, dtype=np.uint8).reshape(3, 4)
    return CollectorSample(
        index=frame_index,
        episode_index=episode_index,
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
    def test_opt_in_integrity_seals_exact_bytes_and_publishes_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LeRobotV21Writer(
                LeRobotV21WriterConfig(
                    root_dir=tmpdir,
                    write_video=False,
                    write_parquet=False,
                    tactile_storage="external",
                    tactile_artifact_integrity=True,
                )
            )
            writer.start_episode(episode_index=0)
            for job in writer.plan_sample_jobs(_sample()):
                writer.handle_job(job)
            writer.finalize_episode(0)

            manifest_path = next((Path(tmpdir) / "tactile").rglob("manifest.json"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for spec in manifest["streams"].values():
                payload = Path(tmpdir) / spec["path"]
                stored = payload.read_bytes()
                self.assertEqual(spec["stored_size_bytes"], len(stored))
                self.assertEqual(spec["stored_sha256"], hashlib.sha256(stored).hexdigest())
                self.assertEqual(spec["stored_hash_profile"], "stored_bytes_sha256_v1")

            sealed = build_modality_artifact_inventory(
                tmpdir,
                payload_hash_mode=PAYLOAD_HASH_REQUIRE_SEALED,
            )
            scanned = build_modality_artifact_inventory(tmpdir)
            self.assertEqual(sealed, scanned)
            self.assertEqual(writer.last_tactile_inventory_hash, sealed.semantic_content_hash)
            inventory_path = (
                Path(tmpdir)
                / "meta/modality_artifacts/tactile/inventories"
                / f"{sealed.semantic_content_hash}.json"
            )
            self.assertTrue(inventory_path.is_file())
            self.assertIsNone(writer.last_tactile_inventory_error)

    def test_default_writer_does_not_change_manifest_or_publish_inventory(self) -> None:
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
            for job in writer.plan_sample_jobs(_sample()):
                writer.handle_job(job)
            writer.finalize_episode(0)

            manifest_path = next((Path(tmpdir) / "tactile").rglob("manifest.json"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(
                all("stored_sha256" not in spec for spec in manifest["streams"].values())
            )
            self.assertFalse((Path(tmpdir) / "meta/modality_artifacts").exists())

    def test_existing_unsealed_dataset_allows_incremental_integrity_append(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            legacy = LeRobotV21Writer(
                LeRobotV21WriterConfig(
                    root_dir=tmpdir,
                    write_video=False,
                    write_parquet=False,
                )
            )
            legacy.start_episode(episode_index=0)
            for job in legacy.plan_sample_jobs(_sample()):
                legacy.handle_job(job)
            legacy.finalize_episode(0)

            old_manifest = next((Path(tmpdir) / "tactile").rglob("manifest.json"))
            old_before = old_manifest.read_bytes()
            catalog_before = (Path(tmpdir) / "meta/episodes.jsonl").read_bytes()
            writer = LeRobotV21Writer(
                LeRobotV21WriterConfig(
                    root_dir=tmpdir,
                    write_video=False,
                    write_parquet=False,
                    tactile_artifact_integrity=True,
                )
            )
            self.assertTrue(writer.tactile_inventory_publication_deferred)
            writer.start_episode(episode_index=1)
            for job in writer.plan_sample_jobs(_sample(episode_index=1)):
                writer.handle_job(job)
            writer.finalize_episode(1)

            self.assertEqual(old_manifest.read_bytes(), old_before)
            self.assertTrue(
                (Path(tmpdir) / "meta/episodes.jsonl")
                .read_bytes()
                .startswith(catalog_before)
            )
            new_manifest = next(
                (Path(tmpdir) / "tactile").rglob("episode_000001/manifest.json")
            )
            new_payload = json.loads(new_manifest.read_text(encoding="utf-8"))
            self.assertTrue(
                all(
                    "stored_sha256" in spec
                    for spec in new_payload["streams"].values()
                )
            )
            self.assertTrue(writer.tactile_inventory_publication_deferred)
            self.assertIsNone(writer.last_tactile_inventory_hash)
            self.assertIsNone(writer.last_tactile_inventory_error)
            self.assertFalse(
                (Path(tmpdir) / "meta/modality_artifacts/tactile/inventories").exists()
            )

    def test_integrity_and_idle_recompression_share_one_revision_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            writer = LeRobotV21Writer(
                LeRobotV21WriterConfig(
                    root_dir=root,
                    write_video=False,
                    write_parquet=False,
                    tactile_storage="external",
                    tactile_compression="none",
                    tactile_artifact_integrity=True,
                )
            )
            writer.start_episode(episode_index=0)
            for job in writer.plan_sample_jobs(_raw_tactile_sample()):
                writer.handle_job(job)
            writer.finalize_episode(0)
            capture_inventory_hash = writer.last_tactile_inventory_hash
            self.assertIsNotNone(capture_inventory_hash)

            result = recompress_tactile_zstd_in_place(
                root,
                level=1,
                chunk_frames=1,
                min_free_bytes=0,
                delete_originals=True,
                backend="auto",
                enforce_global_collection_exclusion=False,
            )
            self.assertEqual(result.stream_count, 1)
            manifest_path = next((root / "tactile").rglob("manifest.json"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            spec = manifest["streams"]["M_RAW"]
            self.assertEqual(spec["compression"], "zstd")
            payload = root / spec["path"]
            self.assertEqual(
                spec["stored_sha256"],
                hashlib.sha256(payload.read_bytes()).hexdigest(),
            )
            recompressed = build_modality_artifact_inventory(root)
            self.assertNotEqual(
                recompressed.semantic_content_hash,
                capture_inventory_hash,
            )
            self.assertTrue(
                (
                    root
                    / "meta/modality_artifacts/tactile/inventories"
                    / f"{recompressed.semantic_content_hash}.json"
                ).is_file()
            )

            # Match the live RealTeleop path: the same writer object remains
            # open while the isolated idle worker mutates only already sealed
            # manifests.  No recorder reload is required before the next
            # Episode append.
            writer.start_episode(episode_index=1)
            for job in writer.plan_sample_jobs(
                _raw_tactile_sample(episode_index=1)
            ):
                writer.handle_job(job)
            writer.finalize_episode(1)
            sealed = build_modality_artifact_inventory(
                root,
                payload_hash_mode=PAYLOAD_HASH_REQUIRE_SEALED,
            )
            self.assertEqual(
                writer.last_tactile_inventory_hash,
                sealed.semantic_content_hash,
            )

    def test_shared_writer_rejects_missing_published_tactile_before_append(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LeRobotV21Writer(
                LeRobotV21WriterConfig(
                    root_dir=tmpdir,
                    write_video=False,
                    write_parquet=False,
                    tactile_artifact_integrity=True,
                )
            )
            writer.start_episode(episode_index=0)
            for job in writer.plan_sample_jobs(_sample()):
                writer.handle_job(job)
            writer.finalize_episode(0)
            root = Path(tmpdir)
            catalog_before = (root / "meta/episodes.jsonl").read_bytes()
            for manifest_path in (root / "tactile").rglob("manifest.json"):
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                for spec in manifest["streams"].values():
                    (root / spec["path"]).unlink()

            with self.assertRaisesRegex(
                DatasetCaptureWritableError,
                "no complete current hot artifact graph",
            ):
                LeRobotV21Writer(
                    LeRobotV21WriterConfig(
                        root_dir=root,
                        write_video=False,
                        write_parquet=False,
                        tactile_artifact_integrity=True,
                    )
                )
            self.assertEqual(
                (root / "meta/episodes.jsonl").read_bytes(),
                catalog_before,
            )

    def test_inventory_publish_failure_never_makes_episode_finalize_retriable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LeRobotV21Writer(
                LeRobotV21WriterConfig(
                    root_dir=tmpdir,
                    write_video=False,
                    write_parquet=False,
                    tactile_artifact_integrity=True,
                )
            )
            writer.start_episode(episode_index=0)
            for job in writer.plan_sample_jobs(_sample()):
                writer.handle_job(job)
            with mock.patch(
                "teleop_runtime.collection.writer_v21.build_and_publish_modality_inventory",
                side_effect=OSError("injected inventory publish failure"),
            ):
                with self.assertWarnsRegex(RuntimeWarning, "offline full scan"):
                    writer.finalize_episode(0)

            records = [
                json.loads(line)
                for line in (Path(tmpdir) / "meta/episodes.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(len(records), 1)
            self.assertIn("injected inventory publish failure", writer.last_tactile_inventory_error or "")
            with self.assertRaisesRegex(RuntimeError, "does not contain any rows"):
                writer.finalize_episode(0)
            records_after_retry = [
                line
                for line in (Path(tmpdir) / "meta/episodes.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(len(records_after_retry), 1)

    def test_integrity_flag_rejects_inline_tactile_storage(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires tactile_storage='external'"):
            LeRobotV21WriterConfig(
                root_dir="unused",
                tactile_storage="parquet",
                tactile_artifact_integrity=True,
            )

    def test_external_writer_reuses_unchanged_causal_tactile_snapshot(self) -> None:
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
            tactile_job_count = 0
            for frame_index in range(2):
                sample = _raw_tactile_sample(frame_index=frame_index)
                state = sample.hand_states["left"]
                state.update(
                    {
                        "tactile_snapshot_sequence": 11,
                        "tactile_snapshot_source_timestamp": 9.9,
                        "tactile_snapshot_available_timestamp": 9.95,
                        "tactile_snapshot_source_clock": "local_wall_time",
                        "tactile_snapshot_available_clock": "local_wall_time",
                        "tactile_timestamp": 9.9,
                        "tactile_version": 3,
                    }
                )
                jobs = writer.plan_sample_jobs(sample)
                tactile_job_count += sum(
                    job.kind == TACTILE_FRAME_JOB_KIND for job in jobs
                )
                for job in jobs:
                    writer.handle_job(job)

            self.assertEqual(tactile_job_count, 1)
            self.assertEqual(
                writer._episodes[0].rows_by_frame[0][
                    "observation.hand.left.tactile_frame_index"
                ],
                0,
            )
            self.assertEqual(
                writer._episodes[0].rows_by_frame[1][
                    "observation.hand.left.tactile_frame_index"
                ],
                0,
            )
            writer.finalize_episode(0)
            manifest_path = (
                Path(tmpdir)
                / "tactile/observation.hand.left.tactile/chunk-000/episode_000000/manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["frame_count"], 1)
            self.assertEqual(manifest["frames"][0]["snapshot_sequence"], 11)

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
            self.assertTrue(
                all(
                    Path(spec["path"]).parts[0] == "tactile"
                    for spec in manifest["streams"].values()
                )
            )
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
            self.assertTrue(
                all(
                    Path(relative).parts[0] == "tactile"
                    for relative in episode_record["tactile_paths"].values()
                )
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
            self.assertTrue(
                row["observation.hand.left.tactile_path"].startswith("tactile/")
            )
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
