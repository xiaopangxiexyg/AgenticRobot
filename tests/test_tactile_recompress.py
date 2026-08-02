from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from dataset_contracts.tactile import ChunkedTactileStream
from utils.dataset_postprocess.cli import build_parser
from utils.dataset_postprocess.tactile_recompress import (
    plan_tactile_zstd_recompression,
    recompress_tactile_zstd_in_place,
)


class TactileRecompressTest(unittest.TestCase):
    def test_default_postprocess_enables_canonical_tactile_compaction(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["postprocess", "/tmp/dataset"])
        self.assertTrue(args.tactile_compact)
        self.assertFalse(args.apply)
        self.assertEqual(args.min_free_gb, 20.0)

    def _dataset(self, root: Path) -> tuple[Path, Path, np.ndarray]:
        manifest_relative = Path(
            "tactile/observation.hand.left.tactile/chunk-000/episode_000000/manifest.json"
        )
        stream_relative = manifest_relative.parent / "sensor.uint8.bin"
        stream_path = root / stream_relative
        stream_path.parent.mkdir(parents=True)
        frames = np.arange(5 * 4 * 6, dtype=np.uint8).reshape(5, 4, 6)
        frames.tofile(stream_path)
        manifest = {
            "format": "binary_stream_v1",
            "frame_count": 5,
            "side": "left",
            "sensor_keys": ["sensor"],
            "frames": [{"tactile_frame_index": index} for index in range(5)],
            "streams": {
                "sensor": {
                    "path": stream_relative.as_posix(),
                    "encoding": "raw_c_order",
                    "compression": "none",
                    "dtype": "uint8",
                    "shape": [4, 6],
                    "frame_count": 5,
                }
            },
        }
        manifest_path = root / manifest_relative
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        meta = root / "meta"
        meta.mkdir()
        (meta / "episodes.jsonl").write_text(
            json.dumps(
                {
                    "episode_index": 0,
                    "tactile_paths": {"left": manifest_relative.as_posix()},
                }
            )
            + "\n",
            encoding="utf-8",
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

    @unittest.skipUnless(shutil.which("zstd"), "zstd CLI is required")
    def test_in_place_zstd_commit_round_trips_and_removes_raw(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, stream_path, expected = self._dataset(root)

            with mock.patch(
                "utils.dataset_postprocess.tactile_recompress._active_collection_processes",
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
            self.assertEqual(spec["compression"], "zstd")
            self.assertEqual(spec["encoding"], "chunked_delta_zstd")
            self.assertEqual(spec["predictor"], "delta_mod_uint8")
            self.assertEqual([chunk["frame_count"] for chunk in spec["chunks"]], [2, 2, 1])
            compressed_path = root / spec["path"]
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
                "utils.dataset_postprocess.tactile_recompress._active_collection_processes",
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
                "utils.dataset_postprocess.tactile_recompress._active_collection_processes",
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
                "utils.dataset_postprocess.tactile_recompress._active_collection_processes",
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
