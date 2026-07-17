"""Command-line frontend for formal UMI collection."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

from ..policy_io import load_policy_io_layout
from .calibration import UmiCalibrationBinding
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
        "--no-send-gripper",
        action="store_true",
        help="Connect and collect state but do not transmit gripper commands.",
    )
    parser.add_argument("--foot-pad", action="store_true", help="Require the F13-F16 foot-pad input.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_umi_collection_config(args.config)
    config = replace(
        config,
        open_browser=False if args.no_open_browser else config.open_browser,
        send_gripper_commands=False if args.no_send_gripper else config.send_gripper_commands,
        foot_pad_enabled=True if args.foot_pad else config.foot_pad_enabled,
    )
    binding = UmiCalibrationBinding.load(config.calibration_result)
    binding.require_valid(config)
    layout = load_policy_io_layout(config.policy_io_layout)
    if layout is None:
        raise RuntimeError("UMI config did not resolve a policy I/O layout")
    if args.validate_only:
        print(
            json.dumps(
                {
                    "ok": True,
                    "validated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "dataset_root": str(config.dataset_root.resolve()),
                    "calibration_result": str(binding.path),
                    "calibration_sha256": binding.sha256,
                    "calibration_expires_at": binding.expires_at.isoformat(),
                    "policy_io_layout": str(config.policy_io_layout.resolve()),
                    "policy_io_layout_hash": layout.layout_hash,
                    "action_dim": layout.action_dim,
                    "state_dim": layout.state_dim,
                    "send_gripper_commands": config.send_gripper_commands,
                    "foot_pad_enabled": config.foot_pad_enabled,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    from .app import run_umi_collection

    return run_umi_collection(config)


__all__ = ["build_parser", "main"]
