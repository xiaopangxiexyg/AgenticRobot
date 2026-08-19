from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
import unittest
import uuid
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from collection_runtime.tactile_bundle_store import (
    NativeTactileBundleStore,
    TactileBundleStorageError,
)
from collection_runtime.tactile_bundle_materialize import (
    TactileBundleMaterializationError,
    materialize_tactile_bundle,
    plan_tactile_bundle_materialization,
)
from dataset_contracts.canonical_encoding import canonical_loads
from dataset_contracts.episode_identity import EpisodeCaptureIdentity
from dataset_contracts.tactile_bundle import (
    TACTILE_BUNDLE_EPISODE_COMMIT_FILE_DOMAIN_V1,
    TactileBundleArtifactV1,
    TactileBundleContractError,
    core_dataset_identity_relative_path,
)
from dataset_interface.tactile_bundle import (
    TactileBundleAttachmentError,
    attach_tactile_bundle,
)
from teleop_runtime.collection.sample_assembler import CollectorSample
from teleop_runtime.collection.writer_v21 import (
    LeRobotV21Writer,
    LeRobotV21WriterConfig,
)
from utils.dataset_postprocess.cli import main as dataset_postprocess_main


class NativeTactileBundleTest(unittest.TestCase):
    def _roots(self, temporary: str) -> tuple[Path, Path]:
        base = Path(temporary)
        core = base / "core"
        storage = base / "tactile-device"
        core.mkdir()
        storage.mkdir()
        return core, storage

    def _store(self, core: Path, storage: Path) -> NativeTactileBundleStore:
        return NativeTactileBundleStore(
            core_dataset_root=core,
            tactile_storage_root=storage,
            dataset_label="unit_dataset",
            require_distinct_device=False,
            uuid_factory=lambda: uuid.UUID("3ef8a7b7-22f5-44e7-812c-79d2f66f69d1"),
        )

    def _identity(
        self,
        episode_index: int = 0,
        *,
        episode_uuid: str | None = None,
    ) -> EpisodeCaptureIdentity:
        return EpisodeCaptureIdentity(
            episode_uuid=(
                episode_uuid
                if episode_uuid is not None
                else f"episode-capture-uuid-{episode_index}"
            ),
            legacy_episode_index=episode_index,
            lifecycle_owner_id="tests.native_tactile_bundle",
            prepared_at=12.5,
            prepared_clock_domain_id="test_clock",
        )

    def _collector_sample(self, episode_index: int = 0) -> CollectorSample:
        tactile = np.arange(8, dtype=np.uint8).reshape(2, 4)
        return CollectorSample(
            index=episode_index,
            episode_index=episode_index,
            frame_index=0,
            timestamp=10.0,
            session_state="running",
            control_hz=15.0,
            action=np.zeros(26, dtype=np.float32),
            observation_state=np.zeros(38, dtype=np.float32),
            observation_last_action=np.zeros(26, dtype=np.float32),
            arm_state_source="unit",
            active_sides=("left",),
            hand_states={
                "left": {
                    "tactile": None,
                    "tactile_dict": {"M_RAW": tactile},
                    "tactile_timestamp": 9.9,
                    "telemetry_timestamp": 9.95,
                    "tactile_regions": ("M_RAW",),
                    "tactile_devices": ("M_RAW",),
                    "tactile_frame_ids": {"M_RAW": 11},
                    "tactile_shapes": {"M_RAW": tactile.shape},
                    "tactile_field_by_key": {"M_RAW": "raw"},
                    "tactile_fields": ("raw",),
                    "tactile_version": 3,
                    "tactile_raw_shape_per_sensor": tactile.shape,
                },
                "right": {},
            },
        )

    def _write_tactile_episode(
        self,
        store: NativeTactileBundleStore,
    ) -> tuple[dict[str, str], list[dict[str, object]]]:
        relative_dir = "tactile/chunk-000/episode_000000/left"
        payload_relative = f"{relative_dir}/M_RAW.uint8.bin"
        manifest_relative = f"{relative_dir}/manifest.json"
        payload_path = store.tactile_artifact_path(payload_relative)
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload = bytes(range(8))
        payload_path.write_bytes(payload)
        frames = [
            {
                "sample_frame_index": 0,
                "tactile_frame_index": 0,
                "timestamp": 1.0,
                "snapshot_sequence": 7,
            },
            {
                "sample_frame_index": 1,
                "tactile_frame_index": 1,
                "timestamp": 2.0,
                "snapshot_sequence": 8,
            },
        ]
        manifest = {
            "format": "binary_stream_v1",
            "side": "left",
            "relative_dir": relative_dir,
            "frame_count": 2,
            "sensor_keys": ["M_RAW"],
            "streams": {
                "M_RAW": {
                    "path": payload_relative,
                    "dtype": "uint8",
                    "shape": [4],
                    "frame_count": 2,
                    "encoding": "raw_c_order",
                    "compression": "none",
                    "chunk_frame_count": None,
                    "chunks": [],
                    "stored_size_bytes": len(payload),
                    "stored_sha256": hashlib.sha256(payload).hexdigest(),
                    "stored_hash_profile": "stored_bytes_sha256_v1",
                }
            },
            "frames": frames,
        }
        manifest_path = store.tactile_artifact_path(manifest_relative)
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True),
            encoding="utf-8",
        )
        rows = [
            {
                "frame_index": index,
                "observation.hand.left.tactile_path": manifest_relative,
                "observation.hand.left.tactile_frame_index": index,
            }
            for index in range(2)
        ]
        return {"left": manifest_relative}, rows

    def test_contract_rejects_absolute_or_escaping_artifact_paths(self) -> None:
        digest = "0" * 64
        for value in ("/tactile/a.bin", "tactile/../a.bin", "videos/a.bin"):
            with self.subTest(value=value):
                with self.assertRaises(TactileBundleContractError):
                    TactileBundleArtifactV1(
                        logical_path=value,
                        role="payload",
                        size_bytes=1,
                        sha256=digest,
                    )

    def test_new_bundle_rejects_populated_unbound_core_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            core, storage = self._roots(temporary)
            catalog = core / "meta/episodes.jsonl"
            catalog.parent.mkdir()
            catalog.write_text('{"episode_index": 0}\n', encoding="utf-8")
            with self.assertRaisesRegex(
                TactileBundleStorageError,
                "populated unbound dataset",
            ):
                self._store(core, storage)

    def test_distinct_device_requirement_is_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            core, storage = self._roots(temporary)
            with self.assertRaisesRegex(
                TactileBundleStorageError,
                "different filesystem device",
            ):
                NativeTactileBundleStore(
                    core_dataset_root=core,
                    tactile_storage_root=storage,
                    dataset_label="unit_dataset",
                    require_distinct_device=True,
                )

    def test_same_device_native_bundle_is_allowed_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            core, storage = self._roots(temporary)
            store = NativeTactileBundleStore(
                core_dataset_root=core,
                tactile_storage_root=storage,
                dataset_label="unit_dataset",
            )
            self.assertTrue(store.bundle_dataset_root.is_dir())

    def test_seal_mirrors_commit_and_keeps_only_logical_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            core, storage = self._roots(temporary)
            store = self._store(core, storage)
            identity = self._identity()
            store.prepare_episode(identity)
            tactile_paths, rows = self._write_tactile_episode(store)

            report = store.seal_episode(
                identity=identity,
                rows=rows,
                tactile_paths=tactile_paths,
            )

            core_commit = core / report.core_commit_relative_path
            tactile_commit = (
                store.bundle_dataset_root / report.tactile_commit_relative_path
            )
            self.assertEqual(core_commit.read_bytes(), tactile_commit.read_bytes())
            payload = canonical_loads(
                core_commit.read_bytes(),
                expected_domain=TACTILE_BUNDLE_EPISODE_COMMIT_FILE_DOMAIN_V1,
            )
            self.assertEqual(payload["semantic_content_hash"], report.commit_hash)
            self.assertNotIn(str(core), json.dumps(payload))
            self.assertNotIn(str(storage), json.dumps(payload))
            self.assertTrue((core / core_dataset_identity_relative_path()).is_file())
            self.assertFalse((core / "tactile").exists())
            self.assertTrue(
                (
                    store.bundle_dataset_root
                    / tactile_paths["left"]
                ).is_file()
            )

    def test_alignment_mismatch_fails_before_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            core, storage = self._roots(temporary)
            store = self._store(core, storage)
            identity = self._identity()
            store.prepare_episode(identity)
            tactile_paths, rows = self._write_tactile_episode(store)
            rows[1]["observation.hand.left.tactile_frame_index"] = 0

            with self.assertRaisesRegex(
                TactileBundleStorageError,
                "alignment disagree",
            ):
                store.seal_episode(
                    identity=identity,
                    rows=rows,
                    tactile_paths=tactile_paths,
                )
            self.assertFalse(
                (
                    core
                    / "meta/tactile_bundle_v1/episodes/episode-capture-uuid-0/"
                    "bundle_commit.cjson"
                ).exists()
            )

    def test_alignment_allows_multiple_samples_to_reuse_one_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            core, storage = self._roots(temporary)
            store = self._store(core, storage)
            identity = self._identity()
            store.prepare_episode(identity)
            tactile_paths, rows = self._write_tactile_episode(store)
            manifest_path = store.tactile_artifact_path(tactile_paths["left"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload_path = store.tactile_artifact_path(
                manifest["streams"]["M_RAW"]["path"]
            )
            payload = bytes(range(4))
            payload_path.write_bytes(payload)
            manifest["frame_count"] = 1
            manifest["frames"] = manifest["frames"][:1]
            manifest["streams"]["M_RAW"].update(
                {
                    "frame_count": 1,
                    "stored_size_bytes": len(payload),
                    "stored_sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True),
                encoding="utf-8",
            )
            rows[1]["observation.hand.left.tactile_frame_index"] = 0

            report = store.seal_episode(
                identity=identity,
                rows=rows,
                tactile_paths=tactile_paths,
            )
            record = {
                "episode_index": 0,
                "frame_count": 2,
                "tactile_paths": tactile_paths,
                "tactile_bundle": report.episode_binding_payload(),
            }
            catalog = core / "meta/episodes.jsonl"
            catalog.parent.mkdir(parents=True, exist_ok=True)
            catalog.write_text(json.dumps(record) + "\n", encoding="utf-8")

            attached = attach_tactile_bundle(core, storage)
            self.assertEqual(attached.report.episode_count, 1)

    def test_writer_publishes_bundle_before_core_catalog_visibility(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            core, storage = self._roots(temporary)
            writer = LeRobotV21Writer(
                LeRobotV21WriterConfig(
                    root_dir=core,
                    dataset_name="unit_dataset",
                    write_video=False,
                    write_parquet=False,
                    tactile_storage="external",
                    tactile_compression="none",
                    tactile_artifact_integrity=True,
                    tactile_bundle_root=storage,
                )
            )
            identity = self._identity()
            writer.start_episode(episode_identity=identity)
            for job in writer.plan_sample_jobs(self._collector_sample()):
                writer.handle_job(job)
            writer.finalize_episode(0)

            records = [
                json.loads(line)
                for line in (core / "meta/episodes.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(len(records), 1)
            binding = records[0]["tactile_bundle"]
            commit = core / binding["core_commit_ref"]
            self.assertTrue(commit.is_file())
            self.assertFalse((core / "tactile").exists())
            bundle_root = writer.tactile_bundle_store.bundle_dataset_root
            manifest_relative = records[0]["tactile_paths"]["left"]
            self.assertTrue((bundle_root / manifest_relative).is_file())
            info = json.loads((core / "meta/info.json").read_text(encoding="utf-8"))
            self.assertEqual(
                info["tactile_bundle"]["dataset_uuid"],
                binding["dataset_uuid"],
            )
            self.assertFalse((core / "meta/modality_artifacts").exists())

            attached = attach_tactile_bundle(core, storage)
            self.assertEqual(attached.report.episode_count, 1)
            self.assertEqual(attached.report.payload_count, 1)
            self.assertEqual(attached.report.payload_bytes, 8)
            self.assertEqual(
                attached.resolve(manifest_relative),
                bundle_root / manifest_relative,
            )

    def test_attachment_fails_closed_after_payload_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            core, storage = self._roots(temporary)
            writer = LeRobotV21Writer(
                LeRobotV21WriterConfig(
                    root_dir=core,
                    dataset_name="unit_dataset",
                    write_video=False,
                    write_parquet=False,
                    tactile_artifact_integrity=True,
                    tactile_bundle_root=storage,
                    tactile_bundle_require_distinct_device=False,
                )
            )
            writer.start_episode(episode_identity=self._identity())
            for job in writer.plan_sample_jobs(self._collector_sample()):
                writer.handle_job(job)
            writer.finalize_episode(0)
            bundle_root = writer.tactile_bundle_store.bundle_dataset_root
            payload_path = next((bundle_root / "tactile").rglob("*.bin"))
            payload_path.write_bytes(b"corrupt!")

            with self.assertRaisesRegex(
                TactileBundleAttachmentError,
                "hash mismatch",
            ):
                attach_tactile_bundle(core, storage)

    def test_attachment_rejects_core_catalog_manifest_rebinding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            core, storage = self._roots(temporary)
            writer = LeRobotV21Writer(
                LeRobotV21WriterConfig(
                    root_dir=core,
                    dataset_name="unit_dataset",
                    write_video=False,
                    write_parquet=False,
                    tactile_artifact_integrity=True,
                    tactile_bundle_root=storage,
                    tactile_bundle_require_distinct_device=False,
                )
            )
            writer.start_episode(episode_identity=self._identity())
            for job in writer.plan_sample_jobs(self._collector_sample()):
                writer.handle_job(job)
            writer.finalize_episode(0)
            catalog = core / "meta/episodes.jsonl"
            record = json.loads(catalog.read_text(encoding="utf-8"))
            record["tactile_paths"]["left"] = (
                "tactile/chunk-999/episode_000000/left/manifest.json"
            )
            catalog.write_text(json.dumps(record) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(
                TactileBundleAttachmentError,
                "tactile_paths.*manifests disagree",
            ):
                attach_tactile_bundle(core, storage)

    def test_writer_rejects_empty_native_tactile_episode_before_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            core, storage = self._roots(temporary)
            writer = LeRobotV21Writer(
                LeRobotV21WriterConfig(
                    root_dir=core,
                    dataset_name="unit_dataset",
                    write_video=False,
                    write_parquet=False,
                    tactile_artifact_integrity=True,
                    tactile_bundle_root=storage,
                    tactile_bundle_require_distinct_device=False,
                )
            )
            writer.start_episode(episode_identity=self._identity())
            sample = self._collector_sample()
            sample.hand_states["left"] = {}
            for job in writer.plan_sample_jobs(sample):
                writer.handle_job(job)
            with self.assertRaisesRegex(
                TactileBundleStorageError,
                "cannot commit without tactile manifests",
            ):
                writer.finalize_episode(0)
            catalog = core / "meta/episodes.jsonl"
            self.assertFalse(
                catalog.is_file() and catalog.read_text(encoding="utf-8").strip()
            )

    def test_writer_restart_requires_and_revalidates_same_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            core, storage = self._roots(temporary)
            config = LeRobotV21WriterConfig(
                root_dir=core,
                dataset_name="unit_dataset",
                write_video=False,
                write_parquet=False,
                tactile_artifact_integrity=True,
                tactile_bundle_root=storage,
                tactile_bundle_require_distinct_device=False,
            )
            writer = LeRobotV21Writer(config)
            writer.start_episode(episode_identity=self._identity())
            for job in writer.plan_sample_jobs(self._collector_sample()):
                writer.handle_job(job)
            writer.finalize_episode(0)

            restarted = LeRobotV21Writer(config)
            self.assertEqual(restarted.next_episode_index, 1)
            self.assertEqual(
                restarted.tactile_bundle_store.dataset_identity,
                writer.tactile_bundle_store.dataset_identity,
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "bound tactile bundle root is required",
            ):
                LeRobotV21Writer(
                    LeRobotV21WriterConfig(
                        root_dir=core,
                        dataset_name="unit_dataset",
                        write_video=False,
                        write_parquet=False,
                    )
                )

    def test_writer_repairs_empty_first_start_identity_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            core, storage = self._roots(temporary)
            store = self._store(core, storage)
            storage_identity = store.storage_identity_path
            storage_identity.unlink()

            writer = LeRobotV21Writer(
                LeRobotV21WriterConfig(
                    root_dir=core,
                    dataset_name="unit_dataset",
                    write_video=False,
                    write_parquet=False,
                    tactile_artifact_integrity=True,
                    tactile_bundle_root=storage,
                    tactile_bundle_require_distinct_device=False,
                )
            )

            self.assertTrue(storage_identity.is_file())
            self.assertEqual(
                writer.tactile_bundle_store.dataset_identity,
                store.dataset_identity,
            )
            self.assertEqual(writer.next_episode_index, 0)

    def test_writer_does_not_repair_missing_identity_after_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            core, storage = self._roots(temporary)
            config = LeRobotV21WriterConfig(
                root_dir=core,
                dataset_name="unit_dataset",
                write_video=False,
                write_parquet=False,
                tactile_artifact_integrity=True,
                tactile_bundle_root=storage,
                tactile_bundle_require_distinct_device=False,
            )
            writer = LeRobotV21Writer(config)
            writer.start_episode(episode_identity=self._identity())
            for job in writer.plan_sample_jobs(self._collector_sample()):
                writer.handle_job(job)
            writer.finalize_episode(0)
            storage_identity = writer.tactile_bundle_store.storage_identity_path
            storage_identity.unlink()

            with self.assertRaisesRegex(
                TactileBundleStorageError,
                "missing for a populated dataset",
            ):
                LeRobotV21Writer(config)
            self.assertFalse(storage_identity.exists())

    def test_restart_quarantines_unique_abandoned_external_tail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            core, storage = self._roots(temporary)
            config = LeRobotV21WriterConfig(
                root_dir=core,
                dataset_name="unit_dataset",
                write_video=False,
                write_parquet=False,
                tactile_artifact_integrity=True,
                tactile_bundle_root=storage,
                tactile_bundle_require_distinct_device=False,
            )
            writer = LeRobotV21Writer(config)
            writer.start_episode(episode_identity=self._identity(0))
            for job in writer.plan_sample_jobs(self._collector_sample(0)):
                writer.handle_job(job)
            writer.finalize_episode(0)

            abandoned_identity = self._identity(
                1,
                episode_uuid="abandoned-episode-capture-uuid-1",
            )
            store = writer.tactile_bundle_store
            store.prepare_episode(abandoned_identity)
            abandoned_payload = (
                store.bundle_dataset_root
                / "tactile/chunk-000/episode_000001/left/orphan.bin"
            )
            abandoned_payload.parent.mkdir(parents=True)
            abandoned_payload.write_bytes(b"uncommitted tactile tail")
            abandoned_core = (
                core / "data/chunk-000/episode_000001.parquet"
            )
            abandoned_core.parent.mkdir(parents=True)
            abandoned_core.write_bytes(b"uncommitted core tail")

            with self.assertWarnsRegex(
                RuntimeWarning,
                "native-tactile core Episode artifacts",
            ):
                restarted = LeRobotV21Writer(config)

            self.assertEqual(restarted.next_episode_index, 1)
            self.assertFalse(abandoned_payload.exists())
            self.assertFalse(abandoned_core.exists())
            self.assertEqual(
                len(
                    tuple(
                        (
                            core
                            / "meta/capture_recovery/abandoned_episodes"
                        ).glob(
                            "episode_000001_*/artifacts/data/chunk-000/"
                            "episode_000001.parquet"
                        )
                    )
                ),
                1,
            )
            self.assertFalse(
                (
                    core
                    / "meta/tactile_bundle_v1/episodes"
                    / abandoned_identity.episode_uuid
                ).exists()
            )
            self.assertFalse(
                (
                    store.bundle_dataset_root
                    / "meta/tactile_bundle_v1/episodes"
                    / abandoned_identity.episode_uuid
                ).exists()
            )
            core_receipts = tuple(
                (
                    core / "meta/tactile_bundle_v1/recovery/receipts"
                ).glob("episode_000001_*.json")
            )
            tactile_receipts = tuple(
                (
                    store.bundle_dataset_root
                    / "meta/tactile_bundle_v1/recovery/receipts"
                ).glob("episode_000001_*.json")
            )
            self.assertEqual(len(core_receipts), 1)
            self.assertEqual(len(tactile_receipts), 1)
            self.assertEqual(
                core_receipts[0].read_bytes(),
                tactile_receipts[0].read_bytes(),
            )
            receipt = json.loads(core_receipts[0].read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "complete")
            self.assertEqual(receipt["episode_index"], 1)
            self.assertNotIn(str(core), json.dumps(receipt))
            self.assertNotIn(str(storage), json.dumps(receipt))

            restarted.start_episode(episode_identity=self._identity(1))
            for job in restarted.plan_sample_jobs(self._collector_sample(1)):
                restarted.handle_job(job)
            restarted.finalize_episode(1)
            records = [
                json.loads(line)
                for line in (core / "meta/episodes.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(
                [record["episode_index"] for record in records],
                [0, 1],
            )

    def test_restart_rejects_abandoned_non_tail_episode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            core, storage = self._roots(temporary)
            store = self._store(core, storage)
            orphan = (
                store.bundle_dataset_root
                / "tactile/chunk-000/episode_000002/left/orphan.bin"
            )
            orphan.parent.mkdir(parents=True)
            orphan.write_bytes(b"not the unique next episode")

            with self.assertRaisesRegex(
                TactileBundleStorageError,
                "unique next episode",
            ):
                self._store(core, storage)
            self.assertTrue(orphan.is_file())

    def test_restart_resumes_partially_applied_tail_quarantine(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            core, storage = self._roots(temporary)
            store = self._store(core, storage)
            identity = self._identity(
                0,
                episode_uuid="partially-abandoned-episode",
            )
            store.prepare_episode(identity)
            orphan = (
                store.bundle_dataset_root
                / "tactile/chunk-000/episode_000000/left/orphan.bin"
            )
            orphan.parent.mkdir(parents=True)
            orphan.write_bytes(b"partial recovery")
            committed, catalog_hash = store._read_catalog_bindings()
            episode_index, moves = store._discover_abandoned_tail_moves(
                committed=committed
            )
            operation_id = "a" * 32
            transaction = {
                "schema": "dexteleop.tactile_bundle.recovery.v1",
                "operation_id": operation_id,
                "status": "planned",
                "dataset_uuid": store.dataset_identity.dataset_uuid,
                "episode_index": episode_index,
                "catalog_sha256": catalog_hash,
                "moves": moves,
            }
            store._atomic_write_recovery_json(
                store._recovery_journal_path,
                transaction,
            )
            first = moves[0]
            root = (
                core
                if first["root"] == "core"
                else store.bundle_dataset_root
            )
            source = root / first["source"]
            target = (
                root
                / "meta/tactile_bundle_v1/recovery/abandoned"
                / f"episode_000000_{operation_id}"
                / "artifacts"
                / first["source"]
            )
            target.parent.mkdir(parents=True)
            os.replace(source, target)

            restarted = self._store(core, storage)

            self.assertEqual(
                restarted.dataset_identity.dataset_uuid,
                store.dataset_identity.dataset_uuid,
            )
            self.assertFalse(store._recovery_journal_path.exists())
            receipt = (
                core
                / "meta/tactile_bundle_v1/recovery/receipts"
                / f"episode_000000_{operation_id}.json"
            )
            self.assertTrue(receipt.is_file())
            self.assertFalse(orphan.exists())

    def test_materialization_uses_fresh_inodes_and_atomic_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            core, storage = self._roots(temporary)
            writer = LeRobotV21Writer(
                LeRobotV21WriterConfig(
                    root_dir=core,
                    dataset_name="unit_dataset",
                    write_video=False,
                    write_parquet=False,
                    tactile_artifact_integrity=True,
                    tactile_bundle_root=storage,
                    tactile_bundle_require_distinct_device=False,
                )
            )
            writer.start_episode(episode_identity=self._identity())
            for job in writer.plan_sample_jobs(self._collector_sample()):
                writer.handle_job(job)
            writer.finalize_episode(0)
            attached = attach_tactile_bundle(core, storage)
            source_by_path = {
                artifact.logical_path: attached.resolve(artifact.logical_path)
                for artifact in attached.iter_artifacts()
            }

            report = materialize_tactile_bundle(core, storage)

            self.assertEqual(report.target_path, core / "tactile")
            self.assertTrue(report.receipt_path.is_file())
            receipt = json.loads(report.receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(
                receipt["plan_hash"],
                plan_tactile_bundle_materialization(core, storage).plan_hash,
            )
            self.assertEqual(len(receipt["episode_commit_set_hash"]), 64)
            for logical_path, source in source_by_path.items():
                destination = core / logical_path
                self.assertTrue(destination.is_file())
                self.assertNotEqual(
                    (source.stat().st_dev, source.stat().st_ino),
                    (destination.stat().st_dev, destination.stat().st_ino),
                )
                self.assertEqual(source.read_bytes(), destination.read_bytes())
            self.assertFalse(
                tuple(core.glob(".tactile.native-bundle-staging-*"))
            )
            with self.assertRaisesRegex(
                TactileBundleMaterializationError,
                "Refusing to overwrite",
            ):
                materialize_tactile_bundle(core, storage)

    def test_materialization_rejects_stale_confirmed_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            core, storage = self._roots(temporary)
            writer = LeRobotV21Writer(
                LeRobotV21WriterConfig(
                    root_dir=core,
                    dataset_name="unit_dataset",
                    write_video=False,
                    write_parquet=False,
                    tactile_artifact_integrity=True,
                    tactile_bundle_root=storage,
                    tactile_bundle_require_distinct_device=False,
                )
            )
            writer.start_episode(episode_identity=self._identity())
            for job in writer.plan_sample_jobs(self._collector_sample()):
                writer.handle_job(job)
            writer.finalize_episode(0)

            with self.assertRaisesRegex(
                TactileBundleMaterializationError,
                "changed after the confirmed",
            ):
                materialize_tactile_bundle(
                    core,
                    storage,
                    expected_plan_hash="0" * 64,
                )
            self.assertFalse((core / "tactile").exists())

    def test_materialization_decodes_compressed_stream_head_and_tail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            core, storage = self._roots(temporary)
            writer = LeRobotV21Writer(
                LeRobotV21WriterConfig(
                    root_dir=core,
                    dataset_name="unit_dataset",
                    write_video=False,
                    write_parquet=False,
                    tactile_compression="zstd",
                    tactile_artifact_integrity=True,
                    tactile_bundle_root=storage,
                    tactile_bundle_require_distinct_device=False,
                )
            )
            writer.start_episode(episode_identity=self._identity())
            for job in writer.plan_sample_jobs(self._collector_sample()):
                writer.handle_job(job)
            writer.finalize_episode(0)

            report = materialize_tactile_bundle(core, storage)

            self.assertTrue(report.target_path.is_dir())
            manifest = next(report.target_path.rglob("manifest.json"))
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["streams"]["M_RAW"]["compression"],
                "zstd",
            )

    def test_copied_bundle_can_attach_and_materialize_after_source_is_unavailable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            core, source_storage = self._roots(temporary)
            writer = LeRobotV21Writer(
                LeRobotV21WriterConfig(
                    root_dir=core,
                    dataset_name="portable_bundle_dataset",
                    write_video=False,
                    write_parquet=False,
                    tactile_storage="external",
                    tactile_compression="zstd",
                    tactile_artifact_integrity=True,
                    tactile_bundle_root=source_storage,
                    tactile_bundle_require_distinct_device=False,
                )
            )
            writer.start_episode(episode_identity=self._identity())
            for job in writer.plan_sample_jobs(self._collector_sample()):
                writer.handle_job(job)
            writer.finalize_episode(0)

            source_bundle = writer.tactile_bundle_store.bundle_dataset_root
            dataset_uuid = (
                writer.tactile_bundle_store.dataset_identity.dataset_uuid
            )
            source_payload = next((source_bundle / "tactile").rglob("*.zst"))
            expected_payload = source_payload.read_bytes()

            migrated_storage = base / "migrated-tactile-device"
            migrated_bundle = migrated_storage / "datasets" / dataset_uuid
            migrated_bundle.parent.mkdir(parents=True)
            shutil.copytree(
                source_bundle,
                migrated_bundle,
                copy_function=shutil.copy2,
            )
            migrated_payload = (
                migrated_bundle / source_payload.relative_to(source_bundle)
            )
            self.assertNotEqual(
                (source_payload.stat().st_dev, source_payload.stat().st_ino),
                (migrated_payload.stat().st_dev, migrated_payload.stat().st_ino),
            )

            source_offline = base / "source-bundle-offline"
            os.replace(source_bundle, source_offline)
            with self.assertRaises(TactileBundleAttachmentError):
                attach_tactile_bundle(core, source_storage)

            status_stdout = io.StringIO()
            with redirect_stdout(status_stdout):
                status_code = dataset_postprocess_main(
                    [
                        "tactile-bundle-status",
                        str(core),
                        "--tactile-root",
                        str(migrated_storage),
                    ]
                )
            status = json.loads(status_stdout.getvalue())
            self.assertEqual(status_code, 0)
            self.assertEqual(status["dataset_uuid"], dataset_uuid)
            self.assertEqual(status["episode_count"], 1)
            self.assertGreater(status["payload_count"], 0)

            plan_stdout = io.StringIO()
            with redirect_stdout(plan_stdout):
                plan_code = dataset_postprocess_main(
                    [
                        "tactile-bundle-materialize",
                        str(core),
                        "--tactile-root",
                        str(migrated_storage),
                    ]
                )
            plan = json.loads(plan_stdout.getvalue())
            self.assertEqual(plan_code, 0)
            self.assertEqual(plan["dataset_uuid"], dataset_uuid)

            apply_stdout = io.StringIO()
            with redirect_stdout(apply_stdout):
                apply_code = dataset_postprocess_main(
                    [
                        "tactile-bundle-materialize",
                        str(core),
                        "--tactile-root",
                        str(migrated_storage),
                        "--plan-hash",
                        plan["plan_hash"],
                        "--confirm-dataset-uuid",
                        dataset_uuid,
                        "--confirm-no-active-writers",
                        "--confirm-no-active-readers",
                        "--apply",
                    ]
                )
            materialized = json.loads(apply_stdout.getvalue())
            self.assertEqual(apply_code, 0)
            self.assertEqual(materialized["dataset_uuid"], dataset_uuid)

            restored_payload = core / source_payload.relative_to(source_bundle)
            self.assertEqual(restored_payload.read_bytes(), expected_payload)
            self.assertNotEqual(
                (migrated_payload.stat().st_dev, migrated_payload.stat().st_ino),
                (restored_payload.stat().st_dev, restored_payload.stat().st_ino),
            )

            migrated_offline = base / "migrated-bundle-offline"
            os.replace(migrated_bundle, migrated_offline)
            self.assertEqual(restored_payload.read_bytes(), expected_payload)
            receipts = tuple(
                (core / "meta/tactile_bundle_v1/materializations").glob("*.json")
            )
            self.assertEqual(len(receipts), 1)


if __name__ == "__main__":
    unittest.main()
