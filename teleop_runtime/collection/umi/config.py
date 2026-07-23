"""Validated configuration for the dual-wrist UMI collector."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from dataset_contracts.capture_modalities import CaptureModalityContract

from .pose_adapter import as_rigid_transform


DEFAULT_LEFT_WRIST_CAMERA_SERIAL = "261122273924"
DEFAULT_RIGHT_WRIST_CAMERA_SERIAL = "260522278055"
SIDES = ("left", "right")


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _transform(value: Any, *, name: str) -> tuple[tuple[float, ...], ...]:
    matrix = as_rigid_transform(value, name=name)
    return tuple(tuple(float(item) for item in row) for row in matrix)


@dataclass(frozen=True)
class UmiSideConfig:
    side: str
    tracker_name: str
    tracker_role: str
    tracker_channel: str
    camera_serial: str
    gripper_port: str
    hand_eye_result: Path | None = None
    tcp_profile: str = "dmgripper"
    tactile_ids: tuple[str, ...] = ()
    T_tcp_to_tracker: tuple[tuple[float, ...], ...] = field(
        default_factory=lambda: tuple(tuple(float(v) for v in row) for row in np.eye(4))
    )
    tcp_extrinsic_status: str = "hand_eye_result"

    def __post_init__(self) -> None:
        if self.side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}")
        for name in ("tracker_name", "tracker_role", "tracker_channel", "camera_serial"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{self.side}.{name} must be non-empty")
        object.__setattr__(
            self,
            "T_tcp_to_tracker",
            _transform(self.T_tcp_to_tracker, name=f"{self.side}.T_tcp_to_tracker"),
        )
        object.__setattr__(self, "tactile_ids", tuple(str(v) for v in self.tactile_ids))
        object.__setattr__(
            self,
            "hand_eye_result",
            None if self.hand_eye_result is None else Path(self.hand_eye_result).expanduser(),
        )
        if not str(self.tcp_profile).strip():
            raise ValueError(f"{self.side}.tcp_profile must be non-empty")

    @property
    def tcp_transform(self) -> np.ndarray:
        return np.asarray(self.T_tcp_to_tracker, dtype=float).copy()

    @property
    def uses_identity_tcp_placeholder(self) -> bool:
        return bool(
            np.allclose(self.tcp_transform, np.eye(4), atol=1e-9)
            and self.tcp_extrinsic_status == "identity_placeholder_unmeasured"
        )

    @classmethod
    def from_dict(
        cls,
        side: str,
        payload: Mapping[str, Any],
        *,
        base_dir: Path,
    ) -> "UmiSideConfig":
        default_camera = (
            DEFAULT_LEFT_WRIST_CAMERA_SERIAL
            if side == "left"
            else DEFAULT_RIGHT_WRIST_CAMERA_SERIAL
        )
        default_tracker_index = 1 if side == "left" else 2
        result_value = payload.get("hand_eye_result")
        result_path = None
        if result_value is not None and str(result_value).strip():
            result_path = Path(str(result_value)).expanduser()
            if not result_path.is_absolute():
                result_path = base_dir / result_path
        return cls(
            side=side,
            tracker_name=str(payload.get("tracker_name", f"tracker_{default_tracker_index}")),
            tracker_role=str(payload.get("tracker_role", f"{side}_wrist")),
            tracker_channel=str(payload.get("tracker_channel", f"{side}_wrist_pose")),
            camera_serial=str(payload.get("camera_serial", default_camera)),
            gripper_port=str(payload.get("gripper_port", "")),
            hand_eye_result=result_path,
            tcp_profile=str(payload.get("tcp_profile", "dmgripper")),
            tactile_ids=tuple(payload.get("tactile_ids", ())),
            T_tcp_to_tracker=payload.get("T_tcp_to_tracker", np.eye(4).tolist()),
            tcp_extrinsic_status=str(
                payload.get(
                    "tcp_extrinsic_status",
                    "hand_eye_result" if result_path is not None else "identity_placeholder_unmeasured",
                )
            ),
        )


@dataclass(frozen=True)
class UmiTimingConfig:
    control_hz: float = 30.0
    sample_hz: float = 15.0
    vive_max_age_sec: float = 0.10
    camera_max_age_sec: float = 0.10
    angle_trigger_max_age_sec: float = 0.20
    gripper_max_age_sec: float = 0.50
    tactile_max_age_sec: float = 0.50

    def __post_init__(self) -> None:
        for name in (
            "control_hz",
            "sample_hz",
            "vive_max_age_sec",
            "camera_max_age_sec",
            "angle_trigger_max_age_sec",
            "gripper_max_age_sec",
            "tactile_max_age_sec",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"timing.{name} must be positive")
        if self.sample_hz > self.control_hz:
            raise ValueError("timing.sample_hz cannot exceed control_hz")


@dataclass(frozen=True)
class UmiCollectionConfig:
    dataset_root: Path
    calibration_result: Path
    policy_io_layout: Path
    left: UmiSideConfig
    right: UmiSideConfig
    desk_calibration_side: str = "left"
    timing: UmiTimingConfig = field(default_factory=UmiTimingConfig)
    vive_host: str = "0.0.0.0"
    vive_port: int = 9999
    vive_clock_sync_port: int = 10001
    angle_trigger_rate_hz: float = 30.0
    angle_trigger_config_root: Path | None = None
    allow_identity_tcp_placeholder: bool = False
    calibration_validity_sec: float = 10800.0
    calibration_warning_sec: float = 600.0
    calibration_critical_sec: float = 300.0
    hand_eye_config: Path = field(
        default_factory=lambda: Path("utils/umi_hand_eye/configs/rm65_dual_wrist_umi_v1.json")
    )
    hand_eye_output_dir: Path = field(
        default_factory=lambda: Path("teleop_records/umi_hand_eye")
    )
    tactile_required: bool = True
    ignored_camera_keys: tuple[str, ...] = ()
    ignored_tactile_keys: tuple[str, ...] = ()
    hardware_admission_mode: str = "capability"
    tactile_compression: str = "none"
    dmtac_tactile_read_hz: float = 15.0
    dmtac_gripper_command_hz: float = 20.0
    dmtac_gripper_telemetry_hz: float = 20.0
    dmtac_gripper_telemetry_command_grace_s: float = 0.02
    send_gripper_commands: bool = True
    dmtac_assignment: str = "latest"
    # Prefer the configured 4-pedal service when present; component assembly
    # falls back to the legacy GUI labels if no supported device is online.
    foot_pad_enabled: bool = True
    gui_host: str = "127.0.0.1"
    gui_port: int = 0
    open_browser: bool = True
    task_id: str = "umi_collection"
    task_description: str = "Dual-wrist UMI collection"
    plus_episode_curation: bool = False
    plus_annotation_root: Path | None = None
    plus_task_profile: Path | None = None
    plus_source_binding: str = "v23_compat_content_locator_v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root).expanduser())
        object.__setattr__(
            self, "calibration_result", Path(self.calibration_result).expanduser()
        )
        object.__setattr__(self, "policy_io_layout", Path(self.policy_io_layout).expanduser())
        object.__setattr__(
            self,
            "plus_annotation_root",
            None if self.plus_annotation_root is None else Path(self.plus_annotation_root).expanduser(),
        )
        object.__setattr__(
            self,
            "plus_task_profile",
            None if self.plus_task_profile is None else Path(self.plus_task_profile).expanduser(),
        )
        object.__setattr__(self, "hand_eye_config", Path(self.hand_eye_config).expanduser())
        object.__setattr__(
            self, "hand_eye_output_dir", Path(self.hand_eye_output_dir).expanduser()
        )
        if self.left.side != "left" or self.right.side != "right":
            raise ValueError("left/right UMI side bindings are invalid")
        if self.desk_calibration_side not in SIDES:
            raise ValueError("desk_calibration_side must be left or right")
        if self.left.camera_serial == self.right.camera_serial:
            raise ValueError("left and right wrist cameras must use different serials")
        if self.left.tracker_name == self.right.tracker_name:
            raise ValueError("left and right sides must use different trackers")
        ignored_cameras = tuple(
            sorted({str(value).strip() for value in self.ignored_camera_keys})
        )
        ignored_tactile = tuple(
            sorted({str(value).strip() for value in self.ignored_tactile_keys})
        )
        if any(not value for value in (*ignored_cameras, *ignored_tactile)):
            raise ValueError("ignored modality keys must be non-empty")
        unknown_cameras = sorted(
            set(ignored_cameras) - {"left_wrist", "right_wrist"}
        )
        unknown_tactile = sorted(set(ignored_tactile) - set(SIDES))
        if unknown_cameras:
            raise ValueError(f"unknown ignored UMI camera keys: {unknown_cameras}")
        if unknown_tactile:
            raise ValueError(f"unknown ignored UMI tactile keys: {unknown_tactile}")
        if self.hardware_admission_mode not in {"capability", "strict"}:
            raise ValueError("hardware_admission_mode must be capability or strict")
        if not self.tactile_required:
            ignored_tactile = tuple(SIDES)
        object.__setattr__(self, "ignored_camera_keys", ignored_cameras)
        object.__setattr__(self, "ignored_tactile_keys", ignored_tactile)
        if int(self.vive_port) <= 0 or int(self.vive_clock_sync_port) < 0:
            raise ValueError("invalid Vive ports")
        if int(self.vive_port) == int(self.vive_clock_sync_port):
            raise ValueError("Vive data and clock-sync ports must differ")
        if float(self.calibration_validity_sec) <= 0.0:
            raise ValueError("calibration_validity_sec must be positive")
        if float(self.calibration_warning_sec) <= 0.0:
            raise ValueError("calibration_warning_sec must be positive")
        if float(self.calibration_critical_sec) <= 0.0:
            raise ValueError("calibration_critical_sec must be positive")
        if float(self.calibration_critical_sec) > float(self.calibration_warning_sec):
            raise ValueError(
                "calibration_critical_sec cannot exceed calibration_warning_sec"
            )
        if self.tactile_compression not in {"none", "lz4", "zstd"}:
            raise ValueError("tactile_compression must be one of: none, lz4, zstd")
        if float(self.dmtac_tactile_read_hz) <= 0.0:
            raise ValueError("dmtac_tactile_read_hz must be positive")
        if float(self.dmtac_gripper_command_hz) <= 0.0:
            raise ValueError("dmtac_gripper_command_hz must be positive")
        if float(self.dmtac_gripper_telemetry_hz) <= 0.0:
            raise ValueError("dmtac_gripper_telemetry_hz must be positive")
        if float(self.dmtac_gripper_telemetry_command_grace_s) < 0.0:
            raise ValueError(
                "dmtac_gripper_telemetry_command_grace_s must be non-negative"
            )
        if (
            1.0 / float(self.dmtac_gripper_telemetry_hz)
            > float(self.timing.gripper_max_age_sec)
        ):
            raise ValueError(
                "dmtac_gripper_telemetry_hz is incompatible with "
                "timing.gripper_max_age_sec"
            )
        if self.tactile_required and (
            1.0 / float(self.dmtac_tactile_read_hz)
            > float(self.timing.tactile_max_age_sec)
        ):
            raise ValueError(
                "dmtac_tactile_read_hz is incompatible with "
                "timing.tactile_max_age_sec: each read period must fit within "
                "the required freshness window"
            )
        if not self.allow_identity_tcp_placeholder and (
            self.left.uses_identity_tcp_placeholder
            or self.right.uses_identity_tcp_placeholder
        ):
            raise ValueError(
                "identity TCP extrinsics require allow_identity_tcp_placeholder=true"
            )
        if not self.allow_identity_tcp_placeholder:
            missing_results = [
                side for side in SIDES if self.side(side).hand_eye_result is None
            ]
            if missing_results:
                raise ValueError(
                    "formal UMI collection requires per-side hand_eye_result: "
                    + ", ".join(missing_results)
                )
        if self.plus_source_binding not in {
            "v23_compat_content_locator_v1",
            "v24_committed_capture_membership_v1",
            "v24_compatibility_content_locator_v1",
        }:
            raise ValueError(f"unsupported plus_source_binding: {self.plus_source_binding!r}")
        if self.plus_episode_curation and self.plus_annotation_root is None:
            raise ValueError("plus_annotation_root is required when plus_episode_curation is enabled")

    def side(self, side: str) -> UmiSideConfig:
        if side == "left":
            return self.left
        if side == "right":
            return self.right
        raise ValueError(f"unsupported side {side!r}")

    @property
    def capture_modality_contract(self) -> CaptureModalityContract:
        return CaptureModalityContract(
            ignored_camera_keys=self.ignored_camera_keys,
            ignored_tactile_keys=self.ignored_tactile_keys,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, base_dir: Path) -> "UmiCollectionConfig":
        sides = _mapping(payload.get("sides", {}), name="sides")
        timing_payload = _mapping(payload.get("timing", {}), name="timing")

        validity_seconds_value = payload.get("calibration_validity_sec")
        validity_hours_value = payload.get("calibration_validity_hours")
        if validity_seconds_value is not None and validity_hours_value is not None:
            raise ValueError(
                "configure only one of calibration_validity_hours or "
                "calibration_validity_sec"
            )
        calibration_validity_sec = (
            float(validity_hours_value) * 3600.0
            if validity_hours_value is not None
            else float(
                10800.0 if validity_seconds_value is None else validity_seconds_value
            )
        )

        def resolve(value: Any) -> Path:
            path = Path(str(value)).expanduser()
            return path if path.is_absolute() else base_dir / path

        angle_root = payload.get("angle_trigger_config_root")
        left = UmiSideConfig.from_dict(
            "left", _mapping(sides.get("left", {}), name="sides.left"), base_dir=base_dir
        )
        right = UmiSideConfig.from_dict(
            "right", _mapping(sides.get("right", {}), name="sides.right"), base_dir=base_dir
        )
        desk_side = str(payload.get("desk_calibration_side", "left")).strip().lower()
        side_by_name = {"left": left, "right": right}
        legacy_calibration = payload.get("calibration_result")
        if legacy_calibration is None and desk_side in side_by_name:
            legacy_calibration = side_by_name[desk_side].hand_eye_result
        if legacy_calibration is None:
            raise ValueError("UMI config requires calibration_result or a desk-side hand_eye_result")
        return cls(
            dataset_root=resolve(payload.get("dataset_root", "datasets/umi")),
            calibration_result=resolve(legacy_calibration),
            policy_io_layout=resolve(
                payload.get(
                    "policy_io_layout",
                    "../policy_io_layouts/dual_umi_dmtac_v23.json",
                )
            ),
            left=left,
            right=right,
            desk_calibration_side=desk_side,
            timing=UmiTimingConfig(**dict(timing_payload)),
            vive_host=str(payload.get("vive_host", "0.0.0.0")),
            vive_port=int(payload.get("vive_port", 9999)),
            vive_clock_sync_port=int(payload.get("vive_clock_sync_port", 10001)),
            angle_trigger_rate_hz=float(payload.get("angle_trigger_rate_hz", 30.0)),
            angle_trigger_config_root=None if angle_root is None else resolve(angle_root),
            allow_identity_tcp_placeholder=bool(
                payload.get("allow_identity_tcp_placeholder", False)
            ),
            calibration_validity_sec=calibration_validity_sec,
            calibration_warning_sec=float(payload.get("calibration_warning_sec", 600.0)),
            calibration_critical_sec=float(payload.get("calibration_critical_sec", 300.0)),
            hand_eye_config=resolve(
                payload.get(
                    "hand_eye_config",
                    "../../../utils/umi_hand_eye/configs/rm65_dual_wrist_umi_v1.json",
                )
            ),
            hand_eye_output_dir=resolve(
                payload.get("hand_eye_output_dir", "../../../teleop_records/umi_hand_eye")
            ),
            tactile_required=bool(payload.get("tactile_required", True)),
            ignored_camera_keys=tuple(payload.get("ignored_camera_keys", ())),
            ignored_tactile_keys=tuple(payload.get("ignored_tactile_keys", ())),
            hardware_admission_mode=str(
                payload.get("hardware_admission_mode", "capability")
            ),
            tactile_compression=str(payload.get("tactile_compression", "none")),
            dmtac_tactile_read_hz=float(payload.get("dmtac_tactile_read_hz", 15.0)),
            dmtac_gripper_command_hz=float(
                payload.get("dmtac_gripper_command_hz", 20.0)
            ),
            dmtac_gripper_telemetry_hz=float(
                payload.get("dmtac_gripper_telemetry_hz", 20.0)
            ),
            dmtac_gripper_telemetry_command_grace_s=float(
                payload.get("dmtac_gripper_telemetry_command_grace_s", 0.02)
            ),
            send_gripper_commands=bool(payload.get("send_gripper_commands", True)),
            dmtac_assignment=str(payload.get("dmtac_assignment", "latest")),
            foot_pad_enabled=bool(payload.get("foot_pad_enabled", True)),
            gui_host=str(payload.get("gui_host", "127.0.0.1")),
            gui_port=int(payload.get("gui_port", 0)),
            open_browser=bool(payload.get("open_browser", True)),
            task_id=str(payload.get("task_id", "umi_collection")),
            task_description=str(payload.get("task_description", "Dual-wrist UMI collection")),
            plus_episode_curation=bool(payload.get("plus_episode_curation", False)),
            plus_annotation_root=(
                None
                if payload.get("plus_annotation_root") in (None, "")
                else resolve(payload.get("plus_annotation_root"))
            ),
            plus_task_profile=(
                None
                if payload.get("plus_task_profile") in (None, "")
                else resolve(payload.get("plus_task_profile"))
            ),
            plus_source_binding=str(
                payload.get("plus_source_binding", "v23_compat_content_locator_v1")
            ),
        )


def load_umi_collection_config(path: str | Path) -> UmiCollectionConfig:
    config_path = Path(path).expanduser().resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("UMI collection config must contain a JSON object")
    return UmiCollectionConfig.from_dict(payload, base_dir=config_path.parent)


__all__ = [
    "DEFAULT_LEFT_WRIST_CAMERA_SERIAL",
    "DEFAULT_RIGHT_WRIST_CAMERA_SERIAL",
    "UmiCollectionConfig",
    "UmiSideConfig",
    "UmiTimingConfig",
    "load_umi_collection_config",
]
