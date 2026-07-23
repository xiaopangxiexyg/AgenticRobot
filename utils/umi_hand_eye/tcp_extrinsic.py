"""Derive a named UMI TCP-to-tracker extrinsic from existing project geometry."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from robot_motion.rm65.tool_profiles import (
    endpoint_to_tool_transform,
    load_rm65_tool_profile,
)
from utils.dataset_postprocess.wrist_cam_adjust.geometry import parse_urdf
from utils.hand_eye.transforms import as_transform, invert_transform, transform_to_list


TCP_EXTRINSIC_SCHEMA = "dexteleop.umi_tcp_extrinsic.v1"
DEFAULT_TCP_PROFILE = "dmgripper"
DEFAULT_URDF = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "rm65_inspire_gripper_v3"
    / "rm65_dual_arm_wrist_cameras_v3_head_camera.urdf"
)

_TOOL_PROFILE_ALIASES = {
    "dmgripper": "DMTAC_M",
}


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class UmiTcpExtrinsic:
    side: str
    tcp_profile: str
    resolved_tool_profile: str
    endpoint_frame: str
    camera_frame: str
    urdf_path: str
    urdf_sha256: str
    tool_profile_path: str
    tool_profile_sha256: str
    T_tcp_to_endpoint: np.ndarray
    T_camera_to_endpoint: np.ndarray
    T_tcp_to_camera: np.ndarray
    T_tcp_to_tracker: np.ndarray

    def __post_init__(self) -> None:
        if self.side not in {"left", "right"}:
            raise ValueError("UMI TCP extrinsic side must be left or right")
        for name in (
            "T_tcp_to_endpoint",
            "T_camera_to_endpoint",
            "T_tcp_to_camera",
            "T_tcp_to_tracker",
        ):
            object.__setattr__(self, name, as_transform(getattr(self, name), name=name))

    @property
    def T_tracker_to_tcp(self) -> np.ndarray:
        return invert_transform(self.T_tcp_to_tracker)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TCP_EXTRINSIC_SCHEMA,
            "side": self.side,
            "tcp_profile": self.tcp_profile,
            "resolved_tool_profile": self.resolved_tool_profile,
            "tcp_frame": "tool_tcp",
            "endpoint_frame": self.endpoint_frame,
            "camera_frame": self.camera_frame,
            "camera_frame_convention": "optical_x_right_y_down_z_forward",
            "urdf_path": self.urdf_path,
            "urdf_sha256": self.urdf_sha256,
            "tool_profile_path": self.tool_profile_path,
            "tool_profile_sha256": self.tool_profile_sha256,
            "T_tcp_to_endpoint": transform_to_list(self.T_tcp_to_endpoint),
            "T_camera_to_endpoint": transform_to_list(self.T_camera_to_endpoint),
            "T_tcp_to_camera": transform_to_list(self.T_tcp_to_camera),
            "T_tcp_to_tracker": transform_to_list(self.T_tcp_to_tracker),
            "T_tracker_to_tcp": transform_to_list(self.T_tracker_to_tcp),
            "matrix_convention": "p_B = T_A_to_B @ p_A",
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UmiTcpExtrinsic":
        if str(payload.get("schema")) != TCP_EXTRINSIC_SCHEMA:
            raise ValueError(f"unsupported UMI TCP extrinsic schema {payload.get('schema')!r}")
        result = cls(
            side=str(payload["side"]),
            tcp_profile=str(payload["tcp_profile"]),
            resolved_tool_profile=str(payload["resolved_tool_profile"]),
            endpoint_frame=str(payload["endpoint_frame"]),
            camera_frame=str(payload["camera_frame"]),
            urdf_path=str(payload["urdf_path"]),
            urdf_sha256=str(payload["urdf_sha256"]),
            tool_profile_path=str(payload["tool_profile_path"]),
            tool_profile_sha256=str(payload["tool_profile_sha256"]),
            T_tcp_to_endpoint=np.asarray(payload["T_tcp_to_endpoint"], dtype=float),
            T_camera_to_endpoint=np.asarray(payload["T_camera_to_endpoint"], dtype=float),
            T_tcp_to_camera=np.asarray(payload["T_tcp_to_camera"], dtype=float),
            T_tcp_to_tracker=np.asarray(payload["T_tcp_to_tracker"], dtype=float),
        )
        if payload.get("T_tracker_to_tcp") is not None and not np.allclose(
            as_transform(payload["T_tracker_to_tcp"], name="T_tracker_to_tcp")
            @ result.T_tcp_to_tracker,
            np.eye(4),
            atol=1e-6,
        ):
            raise ValueError("UMI TCP extrinsic forward/inverse transforms are inconsistent")
        return result


def derive_umi_tcp_extrinsic(
    *,
    side: str,
    T_camera_to_tracker: np.ndarray,
    tcp_profile: str = DEFAULT_TCP_PROFILE,
    urdf_path: str | Path = DEFAULT_URDF,
) -> UmiTcpExtrinsic:
    """Resolve ``T_tcp_to_tracker`` without duplicating URDF or tool-profile data."""

    normalized_side = str(side).strip().lower()
    if normalized_side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    requested_profile = str(tcp_profile).strip()
    if not requested_profile:
        raise ValueError("tcp_profile must be non-empty")
    resolved_profile = _TOOL_PROFILE_ALIASES.get(requested_profile.lower(), requested_profile)
    profile = load_rm65_tool_profile(resolved_profile, side=normalized_side)
    profile_path = Path(str(profile.get("_profile_path") or "")).expanduser().resolve()
    if not profile_path.is_file():
        raise ValueError(f"resolved RM65 tool profile has no readable source path: {profile_path}")

    source_urdf = Path(urdf_path).expanduser().resolve()
    urdf = parse_urdf(source_urdf)
    endpoint_frame = f"{normalized_side}_arm_link6"
    camera_frame = f"{normalized_side}_camera_rgb"

    # URDF FK returns ^endpoint T_camera, which maps camera points to endpoint.
    T_camera_to_endpoint = as_transform(
        urdf.transform_between(endpoint_frame, camera_frame),
        name="T_camera_to_endpoint",
    )
    # The shared RM65 profile returns ^endpoint T_tcp, i.e. T_tcp_to_endpoint.
    T_tcp_to_endpoint = as_transform(
        endpoint_to_tool_transform(profile),
        name="T_tcp_to_endpoint",
    )
    T_tcp_to_camera = as_transform(
        invert_transform(T_camera_to_endpoint) @ T_tcp_to_endpoint,
        name="T_tcp_to_camera",
    )
    T_tcp_to_tracker = as_transform(
        as_transform(T_camera_to_tracker, name="T_camera_to_tracker") @ T_tcp_to_camera,
        name="T_tcp_to_tracker",
    )
    return UmiTcpExtrinsic(
        side=normalized_side,
        tcp_profile=requested_profile,
        resolved_tool_profile=resolved_profile,
        endpoint_frame=endpoint_frame,
        camera_frame=camera_frame,
        urdf_path=str(source_urdf),
        urdf_sha256=_sha256(source_urdf),
        tool_profile_path=str(profile_path),
        tool_profile_sha256=_sha256(profile_path),
        T_tcp_to_endpoint=T_tcp_to_endpoint,
        T_camera_to_endpoint=T_camera_to_endpoint,
        T_tcp_to_camera=T_tcp_to_camera,
        T_tcp_to_tracker=T_tcp_to_tracker,
    )


__all__ = [
    "DEFAULT_TCP_PROFILE",
    "DEFAULT_URDF",
    "TCP_EXTRINSIC_SCHEMA",
    "UmiTcpExtrinsic",
    "derive_umi_tcp_extrinsic",
]
