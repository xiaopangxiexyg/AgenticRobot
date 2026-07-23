"""GUI-triggered lifecycle helpers for the existing dual-UMI calibrator."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Mapping

from utils.umi_hand_eye.config import load_config as load_hand_eye_config

from .calibration import UmiCalibrationSet
from .config import UmiCollectionConfig


def validate_hand_eye_config_binding(config: UmiCollectionConfig) -> None:
    """Fail before opening devices when collection/calibration bindings disagree."""

    hand_eye = load_hand_eye_config(config.hand_eye_config)
    scalar_fields = (
        ("vive_host", hand_eye.vive_host, config.vive_host),
        ("vive_port", hand_eye.vive_port, config.vive_port),
        ("vive_clock_sync_port", hand_eye.vive_clock_sync_port, config.vive_clock_sync_port),
    )
    mismatches = [name for name, actual, expected in scalar_fields if actual != expected]
    for side in ("left", "right"):
        source = hand_eye.side(side)
        target = config.side(side)
        fields = {
            "camera_serial": (source.camera_serial, target.camera_serial),
            "tracker_name": (source.tracker_name, target.tracker_name),
            "tracker_role": (source.tracker_role, target.tracker_role),
            "channel_name": (source.channel_name, target.tracker_channel),
            "tcp_profile": (source.tcp_profile.lower(), target.tcp_profile.lower()),
        }
        mismatches.extend(
            f"{side}.{name}" for name, (actual, expected) in fields.items() if actual != expected
        )
    if mismatches:
        raise RuntimeError(
            "UMI collection and hand-eye configs disagree: " + ", ".join(mismatches)
        )


def run_default_dual_calibration(config: UmiCollectionConfig) -> UmiCalibrationSet:
    """Run the canonical existing calibrator and require two newly written results."""

    validate_hand_eye_config_binding(config)
    before = {
        side: _file_digest(config.side(side).hand_eye_result)
        for side in ("left", "right")
    }
    # Import lazily: collection startup without calibration must still be able
    # to bring up the web UI before optional OpenCV calibration dependencies.
    from utils.umi_hand_eye.run_calibrate import main as calibrate_main

    exit_code = calibrate_main(
        [
            "--config", str(config.hand_eye_config),
            "--side", "both",
            "--output-dir", str(config.hand_eye_output_dir),
            "--validity-seconds", str(float(config.calibration_validity_sec)),
        ]
    )
    if int(exit_code) != 0:
        raise RuntimeError(f"dual UMI calibration exited with code {exit_code}")
    after = {
        side: _file_digest(config.side(side).hand_eye_result)
        for side in ("left", "right")
    }
    unchanged = [side for side in ("left", "right") if after[side] == before[side]]
    if unchanged:
        raise RuntimeError(
            "标定窗口未生成新的左右结果（可能按 q 取消）：" + ", ".join(unchanged)
        )
    return UmiCalibrationSet.load(config)


def config_for_calibration_dataset(
    config: UmiCollectionConfig,
    calibrations: UmiCalibrationSet,
) -> UmiCollectionConfig:
    """Avoid mixing immutable v2.3 calibration contracts in one dataset root."""

    root = config.dataset_root
    info = _read_info(root / "meta" / "info.json")
    if not info or _info_matches(info, calibrations):
        return config
    fingerprint = hashlib.sha256(
        "|".join(
            calibrations.by_side[side].sha256 for side in ("left", "right")
        ).encode("ascii")
    ).hexdigest()[:12]
    stamp = calibrations.created_at.strftime("%Y%m%dT%H%M%SZ")
    rotated = root.with_name(f"{root.name}__calib_{stamp}_{fingerprint}")
    return replace(config, dataset_root=rotated)


def _file_digest(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_info(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _info_matches(info: Mapping[str, object], calibrations: UmiCalibrationSet) -> bool:
    if info.get("calibration_sha256") != calibrations.sha256:
        return False
    values = info.get("tcp_extrinsic_by_side")
    if not isinstance(values, Mapping):
        return False
    for side in ("left", "right"):
        item = values.get(side)
        if not isinstance(item, Mapping):
            return False
        if item.get("hand_eye_result_sha256") != calibrations.by_side[side].sha256:
            return False
    return True


__all__ = [
    "config_for_calibration_dataset",
    "run_default_dual_calibration",
    "validate_hand_eye_config_binding",
]
