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
        return cls(
            path=source,
            payload=dict(payload),
            sha256="sha256:" + hashlib.sha256(raw).hexdigest(),
            T_vive_world_to_desk=transform.copy(),
            created_at=_utc(str(payload["created_at_utc"])),
            expires_at=_utc(str(payload["expires_at_utc"])),
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
        calibration_side = str(self.payload.get("side", ""))
        if calibration_side not in {"left", "right"}:
            raise RuntimeError("UMI calibration side binding is invalid")
        side = config.side(calibration_side)
        if str(self.payload.get("camera_serial")) != side.camera_serial:
            raise RuntimeError("UMI calibration camera serial does not match active config")
        if str(self.payload.get("tracker_name")) != side.tracker_name:
            raise RuntimeError("UMI calibration tracker name does not match active config")

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
                        "calibration_sha256": self.sha256,
                        "created_at_utc": self.created_at.isoformat(),
                        "expires_at_utc": self.expires_at.isoformat(),
                        "desk_definition": "calibration_board_frame",
                    },
                },
                "metadata": {
                    "binding_role": "umi_collection",
                    "calibration_result_path": str(self.path),
                    "calibration_sha256": self.sha256,
                },
            }
        )


__all__ = ["RESULT_SCHEMA", "UmiCalibrationBinding"]
