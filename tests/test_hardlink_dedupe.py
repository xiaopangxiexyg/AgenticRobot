from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from utils.dataset_postprocess.hardlink_dedupe import (
    apply_hardlink_dedupe,
    plan_hardlink_dedupe,
)


class HardlinkDedupeTest(unittest.TestCase):
    def test_plan_requires_full_content_match_not_only_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference"
            target = root / "target"
            reference.mkdir()
            target.mkdir()
            reference_root, target_root = reference, target
            reference = reference / "videos"
            target = target / "videos"
            reference.mkdir()
            target.mkdir()
            (reference / "same.bin").write_bytes(b"same payload")
            (target / "same-copy.bin").write_bytes(b"same payload")
            (reference / "different.bin").write_bytes(b"aaaaaaaaaaaa")
            (target / "different-copy.bin").write_bytes(b"bbbbbbbbbbbb")

            with mock.patch(
                "utils.dataset_postprocess.hardlink_dedupe._active_collection_processes",
                return_value=(),
            ):
                plan = plan_hardlink_dedupe(target_root, [reference_root], min_file_bytes=0)

            self.assertEqual(plan.replacement_count, 1)
            self.assertEqual(plan.logical_duplicate_bytes, len(b"same payload"))
            self.assertEqual(Path(plan.replacements[0].target_path).name, "same-copy.bin")

    def test_apply_atomically_replaces_only_target_with_hardlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference"
            target = root / "target"
            reference.mkdir()
            target.mkdir()
            reference_root, target_root = reference, target
            reference = reference / "videos"
            target = target / "videos"
            reference.mkdir()
            target.mkdir()
            source = reference / "artifact.bin"
            duplicate = target / "renamed.bin"
            source.write_bytes(b"verified duplicate")
            duplicate.write_bytes(source.read_bytes())
            source_inode = source.stat().st_ino

            with mock.patch(
                "utils.dataset_postprocess.hardlink_dedupe._active_collection_processes",
                return_value=(),
            ):
                plan = plan_hardlink_dedupe(target_root, [reference_root], min_file_bytes=0)
                result = apply_hardlink_dedupe(plan, confirm_replace_targets=True)

            self.assertEqual(result.replacement_count, 1)
            self.assertEqual(source.stat().st_ino, source_inode)
            self.assertEqual(duplicate.stat().st_ino, source_inode)
            self.assertEqual(duplicate.read_bytes(), b"verified duplicate")
            self.assertEqual(source.stat().st_nlink, 2)

    def test_apply_requires_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference"
            target = root / "target"
            reference.mkdir()
            target.mkdir()
            reference_root, target_root = reference, target
            reference = reference / "videos"
            target = target / "videos"
            reference.mkdir()
            target.mkdir()
            (reference / "a").write_bytes(b"x")
            (target / "b").write_bytes(b"x")
            with mock.patch(
                "utils.dataset_postprocess.hardlink_dedupe._active_collection_processes",
                return_value=(),
            ):
                plan = plan_hardlink_dedupe(target_root, [reference_root], min_file_bytes=0)
                with self.assertRaisesRegex(ValueError, "confirm_replace_targets"):
                    apply_hardlink_dedupe(plan)

    def test_scan_refuses_while_collection_is_active(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference"
            target = root / "target"
            reference.mkdir()
            target.mkdir()
            with mock.patch(
                "utils.dataset_postprocess.hardlink_dedupe._active_collection_processes",
                return_value=("pid=123 collection",),
            ):
                with self.assertRaisesRegex(RuntimeError, "I/O-heavy"):
                    plan_hardlink_dedupe(target, [reference], min_file_bytes=0)


if __name__ == "__main__":
    unittest.main()
