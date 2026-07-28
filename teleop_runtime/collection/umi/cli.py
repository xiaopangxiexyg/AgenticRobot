"""Command-line frontend for formal UMI collection."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

from dataset_contracts.v25p_native_storage import V25P_NATIVE_STORAGE_LAYOUTS_V1

from ..policy_io import load_policy_io_layout
from .calibration import UmiCalibrationSet
from .config import load_umi_collection_config


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs" / "umi" / "dual_wrist_dmtac_v1.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect arm-free dual-wrist UMI demonstrations in DexTeleop v2.3."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate configuration, layout and calibration without initializing devices.",
    )
    parser.add_argument("--no-open-browser", action="store_true")
    parser.add_argument(
        "--calibration-validity-hours",
        type=float,
        default=None,
        help=(
            "Validity assigned to newly generated UMI calibrations. "
            "Overrides the config for this run; default config is 3 hours."
        ),
    )
    parser.add_argument(
        "--no-send-gripper",
        action="store_true",
        help="Connect and collect state but do not transmit gripper commands.",
    )
    parser.add_argument(
        "--foot-pad",
        dest="foot_pad",
        action="store_true",
        default=None,
        help="Enable the configured 2/4-pedal input (default: enabled with manual fallback).",
    )
    parser.add_argument(
        "--no-foot-pad",
        dest="foot_pad",
        action="store_false",
        help="Disable foot-pad input and use the legacy GUI label controls.",
    )
    parser.add_argument(
        "--ignore-camera",
        action="append",
        choices=("left_wrist", "right_wrist"),
        default=None,
        help=(
            "Explicitly omit one camera view for the complete target dataset. "
            "The dataset must be new/blank or already declare the exact same ignore set."
        ),
    )
    parser.add_argument(
        "--ignore-tactile",
        action="append",
        choices=("left", "right"),
        default=None,
        help=(
            "Explicitly omit one tactile side for the complete target dataset; "
            "gripper position/state remains required."
        ),
    )
    parser.add_argument(
        "--hardware-admission-mode",
        choices=("capability", "strict"),
        default=None,
        help="Use capability startup by default; strict restores transactional device startup.",
    )
    parser.add_argument(
        "--v25p-native-shadow",
        action="store_true",
        default=None,
        help=(
            "Enable provisional native v2.5-p dual-write verification. "
            "Default off; legacy v2.3 remains authoritative."
        ),
    )
    parser.add_argument(
        "--v25p-native-storage-layout",
        choices=V25P_NATIVE_STORAGE_LAYOUTS_V1,
        default=None,
        help=(
            "Physical layout for the explicitly enabled provisional v2.5-p "
            "shadow; omitted means the config value (record_cjson_v1 by default)."
        ),
    )
    parser.add_argument(
        "--plus-episode-curation",
        action="store_true",
        default=None,
        help="Enable the optional Plus episode curation GUI/sidecar; raw UMI rows remain unchanged.",
    )
    parser.add_argument(
        "--plus-annotation-root",
        default=None,
        help="External directory for Plus manifest/episode_curation.jsonl.",
    )
    parser.add_argument(
        "--plus-task-profile",
        default=None,
        help="Optional finite task-variant profile JSON for the Plus GUI.",
    )
    parser.add_argument(
        "--plus-source-binding",
        choices=(
            "v23_compat_content_locator_v1",
            "v24_committed_capture_membership_v1",
            "v24_compatibility_content_locator_v1",
        ),
        default=None,
        help="Explicit Plus source binding; no native v2.4 identity is guessed from paths.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_umi_collection_config(args.config)
    config = replace(
        config,
        open_browser=False if args.no_open_browser else config.open_browser,
        send_gripper_commands=False if args.no_send_gripper else config.send_gripper_commands,
        foot_pad_enabled=(
            config.foot_pad_enabled
            if args.foot_pad is None
            else bool(args.foot_pad)
        ),
        v25p_native_shadow=(
            config.v25p_native_shadow
            if args.v25p_native_shadow is None
            else bool(args.v25p_native_shadow)
        ),
        v25p_native_storage_layout=(
            config.v25p_native_storage_layout
            if args.v25p_native_storage_layout is None
            else args.v25p_native_storage_layout
        ),
        ignored_camera_keys=(
            config.ignored_camera_keys
            if args.ignore_camera is None
            else tuple(args.ignore_camera)
        ),
        ignored_tactile_keys=(
            config.ignored_tactile_keys
            if args.ignore_tactile is None
            else tuple(args.ignore_tactile)
        ),
        hardware_admission_mode=(
            config.hardware_admission_mode
            if args.hardware_admission_mode is None
            else args.hardware_admission_mode
        ),
        calibration_validity_sec=(
            float(args.calibration_validity_hours) * 3600.0
            if args.calibration_validity_hours is not None
            else config.calibration_validity_sec
        ),
        plus_episode_curation=(
            config.plus_episode_curation
            if args.plus_episode_curation is None
            else bool(args.plus_episode_curation)
        ),
        plus_annotation_root=(
            config.plus_annotation_root
            if args.plus_annotation_root is None
            else Path(args.plus_annotation_root).expanduser()
        ),
        plus_task_profile=(
            config.plus_task_profile
            if args.plus_task_profile is None
            else Path(args.plus_task_profile).expanduser()
        ),
        plus_source_binding=(
            config.plus_source_binding
            if args.plus_source_binding is None
            else args.plus_source_binding
        ),
    )
    if args.validate_only:
        calibrations = UmiCalibrationSet.load(config)
        binding = calibrations.desk
        layout = load_policy_io_layout(config.policy_io_layout)
        if layout is None:
            raise RuntimeError("UMI config did not resolve a policy I/O layout")
        print(
            json.dumps(
                {
                    "ok": True,
                    "validated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "dataset_root": str(config.dataset_root.resolve()),
                    "calibration_result": str(binding.path),
                    "calibration_sha256": binding.sha256,
                    "calibration_expires_at": binding.expires_at.isoformat(),
                    "desk_calibration_side": config.desk_calibration_side,
                    "hand_eye_results": {
                        side: {
                            "path": str(calibrations.by_side[side].path),
                            "sha256": calibrations.by_side[side].sha256,
                            "tcp_profile": calibrations.by_side[side].tcp_profile,
                        }
                        for side in ("left", "right")
                        if side in calibrations.by_side
                    },
                    "policy_io_layout": str(config.policy_io_layout.resolve()),
                    "policy_io_layout_hash": layout.layout_hash,
                    "action_dim": layout.action_dim,
                    "state_dim": layout.state_dim,
                    "send_gripper_commands": config.send_gripper_commands,
                    "foot_pad_enabled": config.foot_pad_enabled,
                    "capture_modality_contract": config.capture_modality_contract.to_dict(),
                    "plus_episode_curation": bool(config.plus_episode_curation),
                    "plus_annotation_root": (
                        None if config.plus_annotation_root is None else str(config.plus_annotation_root)
                    ),
                    "plus_task_profile": (
                        None if config.plus_task_profile is None else str(config.plus_task_profile)
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    from .app import run_umi_collection

    return run_umi_collection(config)


__all__ = ["build_parser", "main"]
