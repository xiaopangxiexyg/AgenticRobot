"""Runtime-side loading and expiry checks for UMI calibration results."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from dataset_contracts.pose_frames import PoseFrameContract

from .config import UmiCollectionConfig
from .pose_adapter import as_rigid_transform
from .topology import SIDES


RESULT_SCHEMA = "dexteleop.umi_hand_eye.result.v1"


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class UmiCalibrationBinding:
    path: Path
    payload: Mapping[str, Any]
    sha256: str
    T_vive_world_to_desk: np.ndarray
    created_at: datetime
    expires_at: datetime
    side: str
    camera_serial: str
    tracker_name: str
    tcp_profile: str | None
    T_tcp_to_tracker: np.ndarray | None

    @classmethod
    def load(cls, path: str | Path) -> "UmiCalibrationBinding":
        source = Path(path).expanduser().resolve()
        raw = source.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("UMI calibration result must contain a JSON object")
        if str(payload.get("schema")) != RESULT_SCHEMA:
            raise ValueError(f"unsupported UMI calibration schema {payload.get('schema')!r}")
        if str(payload.get("desk_definition")) != "calibration_board_frame":
            raise ValueError("UMI calibration must define board frame == desk")
        transform = as_rigid_transform(
            payload.get("T_vive_world_to_desk"),
            name="T_vive_world_to_desk",
        )
        side = str(payload.get("side", "")).strip().lower()
        tcp_payload = payload.get("tcp_extrinsic")
        tcp_profile = None
        tcp_transform = None
        if tcp_payload is not None:
            if not isinstance(tcp_payload, Mapping):
                raise ValueError("UMI tcp_extrinsic must be an object")
            if str(tcp_payload.get("schema")) != "dexteleop.umi_tcp_extrinsic.v1":
                raise ValueError(
                    f"unsupported UMI TCP extrinsic schema {tcp_payload.get('schema')!r}"
                )
            if str(tcp_payload.get("side", "")).strip().lower() != side:
                raise ValueError("UMI tcp_extrinsic side does not match calibration side")
            tcp_profile = str(tcp_payload.get("tcp_profile", "")).strip()
            if not tcp_profile:
                raise ValueError("UMI tcp_extrinsic is missing tcp_profile")
            tcp_transform = as_rigid_transform(
                tcp_payload.get("T_tcp_to_tracker"),
                name="tcp_extrinsic.T_tcp_to_tracker",
            )
            inverse_value = tcp_payload.get("T_tracker_to_tcp")
            if inverse_value is not None:
                inverse_transform = as_rigid_transform(
                    inverse_value,
                    name="tcp_extrinsic.T_tracker_to_tcp",
                )
                if not np.allclose(
                    inverse_transform @ tcp_transform,
                    np.eye(4),
                    atol=1e-6,
                ):
                    raise ValueError(
                        "UMI TCP extrinsic forward/inverse transforms are inconsistent"
                    )
        return cls(
            path=source,
            payload=dict(payload),
            sha256="sha256:" + hashlib.sha256(raw).hexdigest(),
            T_vive_world_to_desk=transform.copy(),
            created_at=_utc(str(payload["created_at_utc"])),
            expires_at=_utc(str(payload["expires_at_utc"])),
            side=side,
            camera_serial=str(payload.get("camera_serial", "")),
            tracker_name=str(payload.get("tracker_name", "")),
            tcp_profile=tcp_profile,
            T_tcp_to_tracker=None if tcp_transform is None else tcp_transform.copy(),
        )

    def require_valid(
        self,
        config: UmiCollectionConfig,
        *,
        when: datetime | None = None,
    ) -> None:
        instant = datetime.now(timezone.utc) if when is None else when.astimezone(timezone.utc)
        if instant < self.created_at or instant > self.expires_at:
            raise RuntimeError(
                f"UMI calibration expired at {self.expires_at.isoformat()}; recalibration is required"
            )
        declared_lifetime = (self.expires_at - self.created_at).total_seconds()
        if declared_lifetime > float(config.calibration_validity_sec) + 1e-6:
            raise RuntimeError(
                "UMI calibration validity exceeds the configured maximum lifetime"
            )
        calibration_side = self.side
        if calibration_side not in set(SIDES):
            raise RuntimeError("UMI calibration side binding is invalid")
        side = config.side(calibration_side)
        if self.camera_serial != side.camera_serial:
            raise RuntimeError("UMI calibration camera serial does not match active config")
        if self.tracker_name != side.tracker_name:
            raise RuntimeError("UMI calibration tracker name does not match active config")

    def require_tcp_extrinsic(self, config: UmiCollectionConfig, *, side: str) -> np.ndarray:
        side_config = config.side(side)
        if self.side != side:
            raise RuntimeError(
                f"UMI TCP calibration side mismatch: result={self.side!r}, expected={side!r}"
            )
        if self.T_tcp_to_tracker is None or self.tcp_profile is None:
            raise RuntimeError(
                f"{side} UMI calibration has no recorded TCP extrinsic; recalibrate with "
                "the current umi_hand_eye tool"
            )
        if self.tcp_profile.lower() != side_config.tcp_profile.lower():
            raise RuntimeError(
                f"{side} UMI TCP profile mismatch: result={self.tcp_profile!r}, "
                f"config={side_config.tcp_profile!r}"
            )
        return self.T_tcp_to_tracker.copy()

    def pose_frame_contract(self) -> PoseFrameContract:
        transform = self.T_vive_world_to_desk.tolist()
        return PoseFrameContract.from_dict(
            {
                "schema": "dexteleop_pose_frame_contract_v1",
                "pose_storage_frame": "desk_absolute",
                "canonical_absolute_frame": "desk",
                "source": {
                    "schema": "dexteleop_pose_source_contract_v1",
                    "source_id": "vive_umi_dual_wrist_v1",
                    "source_family": "vive_umi",
                    "source_description": "Dual wrist UMI poses in Vive world calibrated to board-defined desk.",
                    "topology": "dual_wrist",
                    "active_sides": list(SIDES),
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
                        "calibration_sha256": self.sha256,
                        "created_at_utc": self.created_at.isoformat(),
                        "expires_at_utc": self.expires_at.isoformat(),
                        "desk_definition": "calibration_board_frame",
                        "canonical_desk_urdf_frame": "workspace_near_boundary_frame",
                        "canonical_desk_definition": (
                            "Left-right centered on the tabletop surface and located at "
                            "the operator/robot-near boundary of the valid tabletop "
                            "workspace along the tabletop-center-to-operator/robot "
                            "direction."
                        ),
                    },
                },
                "metadata": {
                    "binding_role": "umi_collection",
                    "calibration_result_path": str(self.path),
                    "calibration_sha256": self.sha256,
                },
            }
        )


@dataclass(frozen=True)
class UmiCalibrationSet:
    desk: UmiCalibrationBinding
    by_side: Mapping[str, UmiCalibrationBinding]
    T_tcp_to_tracker_by_side: Mapping[str, np.ndarray]

    @property
    def path(self) -> Path:
        return self.desk.path

    @property
    def sha256(self) -> str:
        return self.desk.sha256

    @property
    def created_at(self) -> datetime:
        return self.desk.created_at

    @property
    def expires_at(self) -> datetime:
        return min(
            (binding.expires_at for binding in self.by_side.values()),
            default=self.desk.expires_at,
        )

    def require_valid(
        self,
        config: UmiCollectionConfig,
        *,
        when: datetime | None = None,
    ) -> None:
        self.desk.require_valid(config, when=when)
        for binding in self.by_side.values():
            binding.require_valid(config, when=when)

    @classmethod
    def load(cls, config: UmiCollectionConfig) -> "UmiCalibrationSet":
        bindings: dict[str, UmiCalibrationBinding] = {}
        transforms: dict[str, np.ndarray] = {}
        for side in SIDES:
            result_path = config.side(side).hand_eye_result
            if result_path is None:
                if not config.allow_identity_tcp_placeholder:
                    raise RuntimeError(f"{side} UMI hand-eye result is not configured")
                transforms[side] = config.side(side).tcp_transform
                continue
            binding = UmiCalibrationBinding.load(result_path)
            binding.require_valid(config)
            bindings[side] = binding
            transforms[side] = binding.require_tcp_extrinsic(config, side=side)

        desk = bindings.get(config.desk_calibration_side)
        if desk is None:
            # Legacy configurations may still provide a separate desk-only result.
            desk = UmiCalibrationBinding.load(config.calibration_result)
            desk.require_valid(config)
        return cls(desk=desk, by_side=bindings, T_tcp_to_tracker_by_side=transforms)


__all__ = ["RESULT_SCHEMA", "UmiCalibrationBinding", "UmiCalibrationSet"]
