"""Read-only timing probe for dataset append admission.

This probe never creates a writer or touches devices.  It is intended to
separate dataset-side admission latency from process-worker and hardware
startup latency on an existing collection root.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from dataset_interface.modality_artifacts import (
    assert_dataset_capture_integrity_compatible,
    assert_dataset_capture_writable,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", nargs="?", type=Path)
    parser.add_argument(
        "--preset",
        choices=("current-writing4",),
        help="Use the current remote writing fixture without putting its path on the command line.",
    )
    parser.add_argument(
        "--integrity",
        action="store_true",
        help="Also run the read-only sealed-integrity compatibility check.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root
    if args.preset == "current-writing4":
        dataset_root = (
            Path.home()
            / "zc/datasets/RealmanDMTAC/FormalCollection"
            / "2026-08-07_dmtac_rm65_bimanual_position_v2_writing_4_stage"
        )
    if dataset_root is None:
        parser.error("dataset_root or --preset is required")

    started = time.perf_counter()
    assert_dataset_capture_writable(dataset_root)
    elapsed = time.perf_counter() - started
    print(f"dataset_append_admission=ok elapsed_sec={elapsed:.6f}")
    if args.integrity:
        started = time.perf_counter()
        assert_dataset_capture_integrity_compatible(dataset_root)
        elapsed = time.perf_counter() - started
        print(f"dataset_integrity_compatibility=ok elapsed_sec={elapsed:.6f}")


if __name__ == "__main__":
    main()
