"""Fit both fixed transforms from a replayable paired-pose JSON dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hardware.quest_dev.features.visualization.paired_pose_calibration import (
    fit_paired_pose_calibration,
    parse_paired_pose_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    dataset = parse_paired_pose_dataset(payload)
    fit = fit_paired_pose_calibration(dataset)
    print(json.dumps(fit.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
