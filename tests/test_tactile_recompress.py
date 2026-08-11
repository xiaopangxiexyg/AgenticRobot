from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from collection_runtime.modality_inventory import (
    build_and_publish_modality_inventory,
)
from collection_runtime.process_quiescence import is_formal_collection_argv
from collection_runtime.tactile_recompression import (
    TactileRecompressionCancelled,
    clean_uncommitted_tactile_recompression_artifacts,
    plan_tactile_zstd_recompression,
    recompress_tactile_zstd_in_place,
)
from dataset_contracts.tactile import ChunkedTactileStream
from dataset_interface.modality_artifacts import build_modality_artifact_inventory
from utils.dataset_postprocess.cli import build_parser
from utils.dataset_postprocess.tactile_recompress import (
    recompress_tactile_zstd_in_place as compatibility_recompress,
)


class TactileRecompressTest(unittest.TestCase):
    def test_collection_process_detection_matches_module_argv_tokens(self) -> None:
        self.assertTrue(
            is_formal_collection_argv(
                ("python", "-m", "teleop_runtime.collection", "umi")
            )
        )
        self.assertTrue(
            is_formal_collection_argv(
                (
                    "python",
                    "-u",
                    "-m",
                    "teleop_runtime.collection.run_collect",
                )
            )
        )
        self.assertTrue(
            is_formal_collection_argv(
                (
                    "conda",
                    "run",
                    "python",
                    "-m",
                    "teleop_runtime.debug.run_vive_rm65_real_teleop",
                    "--gui",
                )
            )
        )

    def test_collection_process_detection_ignores_embedded_command_text(self) -> None:
        command = "python -m teleop_runtime.collection umi"
        self.assertFalse(is_formal_collection_argv(("bash", "-c", command)))
        self.assertFalse(
            is_formal_collection_argv(
                (
                    "pytest",
                    "tests/test_real_teleop_progress_adapter.py",
                    "-k",
                    command,
                )
            )
        )
        self.assertFalse(
            is_formal_collection_argv(
                (
                    "python",
                    "-m",
                    "teleop_runtime.collection.replay_dataset_sapien",
                )
            )
        )

    def test_legacy_tooling_path_reexports_collection_runtime_owner(self) -> None:
        self.assertIs(
            compatibility_recompress,
            recompress_tactile_zstd_in_place,
        )
        self.assertEqual(
            recompress_tactile_zstd_in_place.__module__,
            "collection_runtime.tactile_recompression",
        )

    def test_default_postprocess_enables_canonical_tactile_compaction(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["postprocess", "/tmp/dataset"])
        self.assertTrue(args.tactile_compact)
        self.assertFalse(args.apply)
        self.assertEqual(args.min_free_gb, 20.0)

    def _dataset(
        self,
        root: Path,
        *,
        episode_index: int = 0,
        sealed: bool = False,
    ) -> tuple[Path, Path, np.ndarray]:
        manifest_relative = Path(
            "tactile/observation.hand.left.tactile/chunk-000/"
            f"episode_{int(episode_index):06d}/manifest.json"
        )
        stream_relative = manifest_relative.parent / "sensor.uint8.bin"
        stream_path = root / stream_relative
        stream_path.parent.mkdir(parents=True)
        frames = (
            np.arange(5 * 4 * 6, dtype=np.uint8) + int(episode_index)
        ).reshape(5, 4, 6)
        frames.tofile(stream_path)
        stream_spec = {
            "path": stream_relative.as_posix(),
            "encoding": "raw_c_order",
            "compression": "none",
            "dtype": "uint8",
            "shape": [4, 6],
            "frame_count": 5,
        }
        if sealed:
            stored = stream_path.read_bytes()
            stream_spec.update(
                {
                    "stored_size_bytes": len(stored),
                    "stored_sha256": hashlib.sha256(stored).hexdigest(),
                    "stored_hash_profile": "stored_bytes_sha256_v1",
                }
            )
        manifest = {
            "format": "binary_stream_v1",
            "frame_count": 5,
            "side": "left",
            "sensor_keys": ["sensor"],
            "frames": [{"tactile_frame_index": index} for index in range(5)],
            "streams": {"sensor": stream_spec},
        }
        manifest_path = root / manifest_relative
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        meta = root / "meta"
        meta.mkdir(exist_ok=True)
        with (meta / "episodes.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "episode_index": int(episode_index),
                        "tactile_paths": {"left": manifest_relative.as_posix()},
                    }
                )
                + "\n"
            )
        return manifest_path, stream_path, frames

    def test_plan_only_includes_finalized_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, stream, _ = self._dataset(root)
            extra = root / "tactile/unfinalized/manifest.json"
            extra.parent.mkdir(parents=True)
            extra.write_text("{}", encoding="utf-8")

            plan = plan_tactile_zstd_recompression(root, min_free_bytes=0)

            self.assertEqual(plan.manifest_count, 1)
            self.assertEqual(plan.stream_count, 1)
            self.assertEqual(plan.raw_bytes, stream.stat().st_size)

    def test_uncommitted_compressed_target_is_removed_without_touching_raw(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, stream_path, _ = self._dataset(root)
            manifest_before = manifest_path.read_bytes()
            raw_before = stream_path.read_bytes()
            orphan = stream_path.with_name(f"{stream_path.name}.tdelta8.zst")
            orphan.write_bytes(b"forced-stop-orphan")

            removed = clean_uncommitted_tactile_recompression_artifacts(root)

            self.assertEqual(removed, 1)
            self.assertFalse(orphan.exists())
            self.assertEqual(stream_path.read_bytes(), raw_before)
            self.assertEqual(manifest_path.read_bytes(), manifest_before)

    @unittest.skipUnless(shutil.which("zstd"), "zstd CLI is required")
    def test_in_place_zstd_commit_round_trips_and_removes_raw(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, stream_path, expected = self._dataset(root)
            (root / "meta/info.json").write_text(
                json.dumps({"tactile_artifact_integrity": True}),
                encoding="utf-8",
            )

            with mock.patch(
                "collection_runtime.tactile_recompression._active_collection_processes",
                return_value=(),
            ):
                result = recompress_tactile_zstd_in_place(
                    root,
                    level=1,
                    chunk_frames=2,
                    min_free_bytes=0,
                    max_manifests=1,
                    delete_originals=True,
                    backend="cli",
                )

            self.assertEqual(result.manifest_count, 1)
            self.assertEqual(result.stream_count, 1)
            self.assertFalse(stream_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            spec = manifest["streams"]["sensor"]
            compressed_path = root / spec["path"]
            self.assertEqual(spec["compression"], "zstd")
            self.assertEqual(spec["encoding"], "chunked_delta_zstd")
            self.assertEqual(spec["predictor"], "delta_mod_uint8")
            self.assertEqual(spec["stored_size_bytes"], compressed_path.stat().st_size)
            self.assertEqual(spec["stored_hash_profile"], "stored_bytes_sha256_v1")
            self.assertEqual(
                spec["stored_sha256"],
                hashlib.sha256(compressed_path.read_bytes()).hexdigest(),
            )
            self.assertEqual([chunk["frame_count"] for chunk in spec["chunks"]], [2, 2, 1])
            self.assertTrue(compressed_path.is_file())
            reader = ChunkedTactileStream(dataset_root=root, spec=spec)
            for index in range(len(expected)):
                np.testing.assert_array_equal(reader.frame(index), expected[index])
            receipt = root / str(result.receipt_path)
            self.assertTrue(receipt.is_file())
            receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(
                receipt_payload["schema"],
                "dexteleop.tactile_lossless_recompression_receipt.v1",
            )
            inventory_hash = receipt_payload["published_modality_inventory_hash"]
            self.assertTrue(
                (
                    root
                    / "meta/modality_artifacts/tactile/inventories"
                    / f"{inventory_hash}.json"
                ).is_file()
            )

    def test_apply_requires_explicit_original_deletion_acknowledgement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._dataset(root)
            with self.assertRaisesRegex(ValueError, "delete_originals=True"):
                recompress_tactile_zstd_in_place(root, min_free_bytes=0)

    def test_apply_refuses_while_formal_collection_is_active(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._dataset(root)
            with mock.patch(
                "collection_runtime.tactile_recompression._active_collection_processes",
                return_value=("pid=123 python -m teleop_runtime.collection umi",),
            ):
                with self.assertRaisesRegex(RuntimeError, "formal collection is active"):
                    recompress_tactile_zstd_in_place(
                        root,
                        min_free_bytes=0,
                        delete_originals=True,
                    )

    @unittest.skipUnless(shutil.which("zstd"), "zstd CLI is required")
    def test_apply_rolls_back_if_collection_starts_before_manifest_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, stream_path, _ = self._dataset(root)
            manifest_before = manifest_path.read_bytes()
            stream_before = stream_path.read_bytes()
            with mock.patch(
                "collection_runtime.tactile_recompression._active_collection_processes",
                side_effect=((), ("pid=456 formal collection",)),
            ):
                with self.assertRaisesRegex(RuntimeError, "started before tactile manifest"):
                    recompress_tactile_zstd_in_place(
                        root,
                        chunk_frames=2,
                        min_free_bytes=0,
                        delete_originals=True,
                        backend="cli",
                    )

            self.assertEqual(manifest_path.read_bytes(), manifest_before)
            self.assertEqual(stream_path.read_bytes(), stream_before)
            self.assertFalse(
                stream_path.with_name(f"{stream_path.name}.tdelta2.zst").exists()
            )
            receipts = list((root / "meta" / "tactile_recompression").glob("*.json"))
            self.assertEqual(len(receipts), 1)
            receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "failed_partial")
            self.assertEqual(receipt["manifest_count"], 0)
            self.assertTrue(receipt["error"]["resume_safe"])

    @unittest.skipUnless(shutil.which("zstd"), "zstd CLI is required")
    def test_cooperative_cancel_preserves_raw_stream_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, stream_path, _ = self._dataset(root)
            manifest_before = manifest_path.read_bytes()
            stream_before = stream_path.read_bytes()
            checks = 0

            def cancel_requested() -> bool:
                nonlocal checks
                checks += 1
                return checks >= 3

            with self.assertRaises(TactileRecompressionCancelled):
                recompress_tactile_zstd_in_place(
                    root,
                    chunk_frames=2,
                    min_free_bytes=0,
                    delete_originals=True,
                    backend="cli",
                    cancel_requested=cancel_requested,
                    enforce_global_collection_exclusion=False,
                )

            self.assertEqual(manifest_path.read_bytes(), manifest_before)
            self.assertEqual(stream_path.read_bytes(), stream_before)
            self.assertFalse(
                stream_path.with_name(f"{stream_path.name}.tdelta2.zst").exists()
            )
            receipts = list((root / "meta" / "tactile_recompression").glob("*.json"))
            self.assertEqual(len(receipts), 1)
            receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "cancelled_partial")
            self.assertTrue(receipt["error"]["resume_safe"])

    @unittest.skipUnless(shutil.which("zstd"), "zstd CLI is required")
    def test_partial_cancel_publishes_integrity_revision_for_committed_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _manifest0, stream0, _ = self._dataset(
                root,
                episode_index=0,
                sealed=True,
            )
            self._dataset(root, episode_index=1, sealed=True)
            (root / "meta/info.json").write_text(
                json.dumps({"tactile_artifact_integrity": True}),
                encoding="utf-8",
            )
            initial = build_and_publish_modality_inventory(
                root,
                payload_hash_mode="require_sealed",
            )

            def cancel_after_first_manifest() -> bool:
                return not stream0.exists()

            with self.assertRaises(TactileRecompressionCancelled):
                recompress_tactile_zstd_in_place(
                    root,
                    chunk_frames=2,
                    min_free_bytes=0,
                    delete_originals=True,
                    backend="cli",
                    cancel_requested=cancel_after_first_manifest,
                    enforce_global_collection_exclusion=False,
                )

            current = build_modality_artifact_inventory(
                root,
                payload_hash_mode="require_sealed",
            )
            self.assertNotEqual(
                current.semantic_content_hash,
                initial.semantic_content_hash,
            )
            current_path = (
                root
                / "meta/modality_artifacts/tactile/inventories"
                / f"{current.semantic_content_hash}.json"
            )
            self.assertTrue(current_path.is_file())
            receipts = list((root / "meta/tactile_recompression").glob("*.json"))
            self.assertEqual(len(receipts), 1)
            receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "cancelled_partial")
            self.assertEqual(receipt["manifest_count"], 1)
            self.assertEqual(
                receipt["published_modality_inventory_hash"],
                current.semantic_content_hash,
            )

    def test_apply_refuses_committed_v24_capture_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, stream_path, _ = self._dataset(root)
            commit_path = (
                root
                / "meta/v24/control_trace/episodes/episode-uuid-0/capture_commit.cjson"
            )
            commit_path.parent.mkdir(parents=True, exist_ok=True)
            commit_path.write_bytes(b"committed-v2.4-evidence")
            manifest_before = manifest_path.read_bytes()
            stream_before = stream_path.read_bytes()

            with self.assertRaisesRegex(RuntimeError, "committed v2.4 capture"):
                recompress_tactile_zstd_in_place(
                    root,
                    min_free_bytes=0,
                    delete_originals=True,
                )

            self.assertEqual(manifest_path.read_bytes(), manifest_before)
            self.assertEqual(stream_path.read_bytes(), stream_before)
            self.assertFalse(
                stream_path.with_name(f"{stream_path.name}.tdelta8.zst").exists()
            )

    def test_apply_refuses_hardlinked_source_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, stream_path, _ = self._dataset(root)
            linked_copy = root / "independent-backup-is-not-a-hardlink.uint8.bin"
            os.link(stream_path, linked_copy)
            manifest_before = manifest_path.read_bytes()

            with mock.patch(
                "collection_runtime.tactile_recompression._active_collection_processes",
                return_value=(),
            ):
                with self.assertRaisesRegex(RuntimeError, "link_count=2"):
                    recompress_tactile_zstd_in_place(
                        root,
                        min_free_bytes=0,
                        delete_originals=True,
                        backend="cli",
                    )

            self.assertEqual(manifest_path.read_bytes(), manifest_before)
            self.assertTrue(stream_path.is_file())
            self.assertEqual(stream_path.stat().st_ino, linked_copy.stat().st_ino)
            self.assertFalse(stream_path.with_name(f"{stream_path.name}.tdelta8.zst").exists())


if __name__ == "__main__":
    unittest.main()
