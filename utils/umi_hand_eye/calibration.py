from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from utils.hand_eye.calibration import (
    CalibrationSample,
    solve_eye_in_hand,
)
from utils.hand_eye.transforms import as_transform, invert_transform, transform_to_list

from .tcp_extrinsic import (
    DEFAULT_TCP_PROFILE,
    DEFAULT_URDF,
    UmiTcpExtrinsic,
    derive_umi_tcp_extrinsic,
)


SAMPLE_SCHEMA = "dexteleop.umi_hand_eye.samples.v1"
RESULT_SCHEMA = "dexteleop.umi_hand_eye.result.v1"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _datetime_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@dataclass
class UmiCalibrationSample:
    """One static paired observation using explicit frame names.

    ``T_tracker_to_vive_world`` is the OpenVR device pose and
    ``T_board_to_camera`` is the ArUco/ChArUco pose returned by OpenCV.
    """

    T_tracker_to_vive_world: np.ndarray
    T_board_to_camera: np.ndarray
    timestamp_utc: str | None = None
    target_reprojection_rmse_px: float | None = None
    target_corner_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.T_tracker_to_vive_world = as_transform(
            self.T_tracker_to_vive_world, name="T_tracker_to_vive_world"
        )
        self.T_board_to_camera = as_transform(
            self.T_board_to_camera, name="T_board_to_camera"
        )

    def to_hand_eye_sample(self) -> CalibrationSample:
        return CalibrationSample(
            T_gripper_to_base=self.T_tracker_to_vive_world,
            T_target_to_camera=self.T_board_to_camera,
            target_reprojection_rmse_px=self.target_reprojection_rmse_px,
            target_corner_count=self.target_corner_count,
            timestamp_utc=self.timestamp_utc,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "T_tracker_to_vive_world": transform_to_list(self.T_tracker_to_vive_world),
            "T_board_to_camera": transform_to_list(self.T_board_to_camera),
            "timestamp_utc": self.timestamp_utc,
            "target_reprojection_rmse_px": self.target_reprojection_rmse_px,
            "target_corner_count": self.target_corner_count,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UmiCalibrationSample":
        return cls(
            T_tracker_to_vive_world=np.asarray(payload["T_tracker_to_vive_world"], dtype=float),
            T_board_to_camera=np.asarray(payload["T_board_to_camera"], dtype=float),
            timestamp_utc=payload.get("timestamp_utc"),
            target_reprojection_rmse_px=payload.get("target_reprojection_rmse_px"),
            target_corner_count=payload.get("target_corner_count"),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass
class UmiCalibrationResult:
    T_camera_to_tracker: np.ndarray
    T_board_to_vive_world: np.ndarray
    created_at_utc: str
    expires_at_utc: str
    side: str
    camera_serial: str
    tracker_name: str
    tracker_role: str
    method: str
    metrics: dict[str, Any]
    sample_count: int
    tcp_extrinsic: UmiTcpExtrinsic | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.T_camera_to_tracker = as_transform(
            self.T_camera_to_tracker, name="T_camera_to_tracker"
        )
        self.T_board_to_vive_world = as_transform(
            self.T_board_to_vive_world, name="T_board_to_vive_world"
        )
        if self.side not in {"left", "right"}:
            raise ValueError("side must be left or right")
        if _parse_utc(self.expires_at_utc) <= _parse_utc(self.created_at_utc):
            raise ValueError("expires_at_utc must be later than created_at_utc")
        if self.tcp_extrinsic is not None and self.tcp_extrinsic.side != self.side:
            raise ValueError("UMI TCP extrinsic side does not match calibration side")

    @property
    def T_tracker_to_camera(self) -> np.ndarray:
        return invert_transform(self.T_camera_to_tracker)

    @property
    def T_vive_world_to_desk(self) -> np.ndarray:
        # The v1 protocol defines board == canonical desk.
        return invert_transform(self.T_board_to_vive_world)

    @property
    def T_desk_to_vive_world(self) -> np.ndarray:
        return self.T_board_to_vive_world.copy()

    @property
    def T_tcp_to_tracker(self) -> np.ndarray | None:
        if self.tcp_extrinsic is None:
            return None
        return self.tcp_extrinsic.T_tcp_to_tracker.copy()

    def is_valid_at(self, when: datetime | None = None) -> bool:
        instant = _utc_now() if when is None else _datetime_utc(when)
        return _parse_utc(self.created_at_utc) <= instant <= _parse_utc(self.expires_at_utc)

    def require_valid(
        self,
        *,
        when: datetime | None = None,
        camera_serial: str | None = None,
        tracker_name: str | None = None,
    ) -> None:
        if not self.is_valid_at(when):
            raise RuntimeError(
                f"UMI calibration expired at {self.expires_at_utc}; recalibration is required"
            )
        if camera_serial is not None and str(camera_serial) != self.camera_serial:
            raise RuntimeError("UMI calibration camera serial does not match the active camera")
        if tracker_name is not None and str(tracker_name) != self.tracker_name:
            raise RuntimeError("UMI calibration tracker name does not match the active tracker")

    def require_tcp_profile(self, tcp_profile: str) -> None:
        if self.tcp_extrinsic is None:
            raise RuntimeError(
                "UMI calibration does not contain T_tcp_to_tracker; recalibration is required"
            )
        if self.tcp_extrinsic.tcp_profile.lower() != str(tcp_profile).strip().lower():
            raise RuntimeError(
                f"UMI TCP profile mismatch: result={self.tcp_extrinsic.tcp_profile!r}, "
                f"expected={tcp_profile!r}"
            )

    def pose_frame_contract(self, *, storage_frame: str = "desk_absolute") -> dict[str, Any]:
        if storage_frame not in {"desk_absolute", "raw_absolute"}:
            raise ValueError("storage_frame must be desk_absolute or raw_absolute")
        transform = transform_to_list(self.T_vive_world_to_desk)
        return {
            "schema": "dexteleop_pose_frame_contract_v1",
            "pose_storage_frame": storage_frame,
            "canonical_absolute_frame": "desk",
            "source": {
                "schema": "dexteleop_pose_source_contract_v1",
                "source_id": "vive_umi_dual_wrist_v1",
                "source_family": "vive_umi",
                "source_description": "Dual wrist UMI poses reported in Vive world and calibrated to a board-defined desk.",
                "topology": "dual_wrist",
                "active_sides": ["left", "right"],
                "raw_absolute_frame_by_side": {
                    "left": "vive_world",
                    "right": "vive_world",
                },
                "raw_to_canonical_desk_transform": transform,
                "raw_to_canonical_desk_transform_by_side": {},
                "canonical_desk_frame": "desk",
                "profile_version": "1",
                "metadata": {
                    "calibration_schema": RESULT_SCHEMA,
                    "calibration_side": self.side,
                    "camera_serial": self.camera_serial,
                    "tracker_name": self.tracker_name,
                    "tracker_role": self.tracker_role,
                    "created_at_utc": self.created_at_utc,
                    "expires_at_utc": self.expires_at_utc,
                    "desk_definition": "calibration_board_frame",
                    "canonical_desk_urdf_frame": "workspace_near_boundary_frame",
                    "canonical_desk_definition": (
                        "Left-right centered on the tabletop surface and located at the "
                        "operator/robot-near boundary of the valid tabletop workspace "
                        "along the tabletop-center-to-operator/robot direction."
                    ),
                },
            },
            "metadata": {
                "binding_role": "umi_collection",
                "calibration_expires_at_utc": self.expires_at_utc,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RESULT_SCHEMA,
            "created_at_utc": self.created_at_utc,
            "expires_at_utc": self.expires_at_utc,
            "validity_seconds": (
                _parse_utc(self.expires_at_utc) - _parse_utc(self.created_at_utc)
            ).total_seconds(),
            "desk_definition": "calibration_board_frame",
            "side": self.side,
            "camera_serial": self.camera_serial,
            "tracker_name": self.tracker_name,
            "tracker_role": self.tracker_role,
            "method": self.method,
            "sample_count": int(self.sample_count),
            "T_camera_to_tracker": transform_to_list(self.T_camera_to_tracker),
            "T_tracker_to_camera": transform_to_list(self.T_tracker_to_camera),
            "T_board_to_vive_world": transform_to_list(self.T_board_to_vive_world),
            "T_desk_to_vive_world": transform_to_list(self.T_desk_to_vive_world),
            "T_vive_world_to_desk": transform_to_list(self.T_vive_world_to_desk),
            "tcp_extrinsic": (
                None if self.tcp_extrinsic is None else self.tcp_extrinsic.to_dict()
            ),
            "metrics": self.metrics,
            "metadata": self.metadata,
            "matrix_convention": {
                "T_A_to_B": "p_B = T_A_to_B @ p_A",
                "calibration_equation": (
                    "T_tracker_to_vive_world @ T_camera_to_tracker @ "
                    "T_board_to_camera = T_board_to_vive_world"
                ),
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UmiCalibrationResult":
        if str(payload.get("schema")) != RESULT_SCHEMA:
            raise ValueError(f"unsupported UMI result schema {payload.get('schema')!r}")
        return cls(
            T_camera_to_tracker=np.asarray(payload["T_camera_to_tracker"], dtype=float),
            T_board_to_vive_world=np.asarray(payload["T_board_to_vive_world"], dtype=float),
            created_at_utc=str(payload["created_at_utc"]),
            expires_at_utc=str(payload["expires_at_utc"]),
            side=str(payload["side"]),
            camera_serial=str(payload["camera_serial"]),
            tracker_name=str(payload["tracker_name"]),
            tracker_role=str(payload["tracker_role"]),
            method=str(payload["method"]),
            metrics=dict(payload.get("metrics") or {}),
            sample_count=int(payload["sample_count"]),
            tcp_extrinsic=(
                None
                if payload.get("tcp_extrinsic") is None
                else UmiTcpExtrinsic.from_dict(payload["tcp_extrinsic"])
            ),
            metadata=dict(payload.get("metadata") or {}),
        )


def solve_umi_hand_eye(
    samples: Sequence[UmiCalibrationSample],
    *,
    side: str,
    camera_serial: str,
    tracker_name: str,
    tracker_role: str,
    validity_seconds: float = 10800.0,
    method: str = "AUTO",
    tcp_profile: str = DEFAULT_TCP_PROFILE,
    urdf_path: str | Path = DEFAULT_URDF,
    created_at: datetime | None = None,
    metadata: Mapping[str, Any] | None = None,
    **solver_options: Any,
) -> UmiCalibrationResult:
    if len(samples) < 3:
        raise ValueError("UMI hand-eye calibration requires at least 3 samples")
    if float(validity_seconds) <= 0.0:
        raise ValueError("validity_seconds must be > 0")
    hand_eye = solve_eye_in_hand(
        [sample.to_hand_eye_sample() for sample in samples],
        method=method,
        metadata={"calibration_kind": "umi_vive_to_board_desk"},
        **solver_options,
    )
    created = _utc_now() if created_at is None else _datetime_utc(created_at)
    expires = created + timedelta(seconds=float(validity_seconds))
    side_value = str(side).strip().lower()
    tcp_extrinsic = derive_umi_tcp_extrinsic(
        side=side_value,
        T_camera_to_tracker=hand_eye.T_camera_to_gripper,
        tcp_profile=tcp_profile,
        urdf_path=urdf_path,
    )
    return UmiCalibrationResult(
        T_camera_to_tracker=hand_eye.T_camera_to_gripper,
        T_board_to_vive_world=hand_eye.T_target_to_base,
        created_at_utc=_iso_utc(created),
        expires_at_utc=_iso_utc(expires),
        side=side_value,
        camera_serial=str(camera_serial),
        tracker_name=str(tracker_name),
        tracker_role=str(tracker_role),
        method=hand_eye.method,
        metrics=dict(hand_eye.metrics),
        sample_count=len(samples),
        tcp_extrinsic=tcp_extrinsic,
        metadata=dict(metadata or {}),
    )


def save_samples(
    path: str | Path,
    samples: Sequence[UmiCalibrationSample],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SAMPLE_SCHEMA,
        "sample_count": len(samples),
        "metadata": dict(metadata or {}),
        "samples": [sample.to_dict() for sample in samples],
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def load_samples(path: str | Path) -> tuple[list[UmiCalibrationSample], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if str(payload.get("schema")) != SAMPLE_SCHEMA:
        raise ValueError(f"unsupported UMI sample schema {payload.get('schema')!r}")
    samples = [UmiCalibrationSample.from_dict(item) for item in payload.get("samples", [])]
    return samples, dict(payload.get("metadata") or {})


def save_result(path: str | Path, result: UmiCalibrationResult) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output


def load_result(path: str | Path) -> UmiCalibrationResult:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return UmiCalibrationResult.from_dict(payload)
