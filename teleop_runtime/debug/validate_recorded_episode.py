from __future__ import annotations

import argparse

import numpy as np

from ..episodes import require_valid_episode
from .log_utils import log_event


def _qpos_summary(qpos: object) -> str:
    if qpos is None:
        return "None"
    return str(np.round(np.asarray(qpos, dtype=float), 3).tolist())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a recorded teleop episode and print replay-relevant summary statistics.",
    )
    parser.add_argument("episode", type=str, help="Path to episode directory or manifest.json")
    parser.add_argument(
        "--strict-warnings",
        action="store_true",
        help="Return a non-zero exit code when validation warnings are present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = require_valid_episode(args.episode)
    manifest = report.manifest
    metadata = manifest.get("metadata", {})

    log_event("Validate", f"episode={report.episode_dir}")
    log_event(
        "Validate",
        f"frames={len(report.frames)} events={len(report.events)} duration={report.duration_sec:.3f}s "
        f"avg_hz={report.average_hz:.2f}" if report.average_hz is not None else
        f"frames={len(report.frames)} events={len(report.events)} duration={report.duration_sec:.3f}s",
    )
    log_event(
        "Validate",
        f"arm_mode={metadata.get('arm_mode')} control_mode={metadata.get('control_mode')} "
        f"hz={metadata.get('hz')} scene_profile={metadata.get('collision_scene_profile')}",
    )
    log_event(
        "Validate",
        f"start_q left={_qpos_summary(manifest.get('start_qpos', {}).get('left') if isinstance(manifest.get('start_qpos'), dict) else None)} "
        f"right={_qpos_summary(manifest.get('start_qpos', {}).get('right') if isinstance(manifest.get('start_qpos'), dict) else None)}",
    )
    log_event(
        "Validate",
        f"end_q left={_qpos_summary(manifest.get('end_qpos', {}).get('left') if isinstance(manifest.get('end_qpos'), dict) else None)} "
        f"right={_qpos_summary(manifest.get('end_qpos', {}).get('right') if isinstance(manifest.get('end_qpos'), dict) else None)}",
    )
    if report.warnings:
        for warning in report.warnings:
            log_event("Validate", f"warning: {warning}")
    else:
        log_event("Validate", "validation passed without warnings")

    if args.strict_warnings and report.warnings:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
