"""Trusted physical-input resolution for independent RM65 Skill deployment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.rm65 import (
    RM65_BIMANUAL_POSITION_V1_NAME,
    build_rm65_dual_arm_scene_profile,
    load_rm65_tool_profile,
    load_world_from_arm_base_transforms,
    load_world_from_fixed_link_transform,
    resolve_rm65_bimanual_position_profile,
    tool_tcp_pose_flange_m_rad,
)
from robot_motion.rm65.position_profile import RM65BimanualPositionProfile
from robot_motion.transforms import validate_pose_matrix
from teleop_runtime.hardware_config.rm65_arm_ips import (
    arm_ips_for_sides,
    load_assignment,
)
from teleop_runtime.target_safety.tcp_pose_envelope_runtime import (
    TCPPoseEnvelopeFileSpec,
    load_tcp_pose_envelope_file_spec,
)

_SIDES = ("left", "right")
_CANONICAL_STATION_LINK_NAME = "workspace_near_boundary_frame"
RM65_SKILL_SCENE_WORLD_FRAME_ID_V1 = "dual_arm_solver_world"
RM65_SKILL_CANONICAL_STATION_FRAME_ID_V1 = "canonical_station"


def _absolute_file(value: str | Path, *, field_name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{field_name} must be an absolute path.")
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{field_name} is not a file: {resolved}")
    return resolved


@dataclass(frozen=True)
class ExclusiveRM65SkillDeploymentProfile:
    """Operator-selected facts; none are accepted from an Agent request."""

    arm_ip_assignment_path: Path
    tool_profile_path_by_side: Mapping[str, Path]
    envelope_profile_path: Path
    position_profile_name: str = RM65_BIMANUAL_POSITION_V1_NAME
    collision_scene_profile: str = "dual_arm_hand_proxy_open"

    def __post_init__(self) -> None:
        tools = {
            str(side): Path(path)
            for side, path in dict(self.tool_profile_path_by_side).items()
        }
        if set(tools) != set(_SIDES):
            raise ValueError(
                "tool_profile_path_by_side must contain exactly left and right."
            )
        position_profile_name = str(self.position_profile_name).strip()
        collision_scene_profile = str(self.collision_scene_profile).strip()
        if not position_profile_name or not collision_scene_profile:
            raise ValueError(
                "position_profile_name and collision_scene_profile must be non-empty."
            )
        object.__setattr__(
            self,
            "arm_ip_assignment_path",
            Path(self.arm_ip_assignment_path),
        )
        object.__setattr__(
            self,
            "tool_profile_path_by_side",
            MappingProxyType(tools),
        )
        object.__setattr__(
            self,
            "envelope_profile_path",
            Path(self.envelope_profile_path),
        )
        object.__setattr__(self, "position_profile_name", position_profile_name)
        object.__setattr__(self, "collision_scene_profile", collision_scene_profile)


@dataclass(frozen=True)
class ResolvedExclusiveRM65SkillDeploymentProfile:
    """Exact local inputs used to construct one independent real process."""

    arm_ip_assignment_path: Path
    tool_profile_path_by_side: Mapping[str, Path]
    envelope_profile_path: Path
    arm_ip_by_side: Mapping[str, str]
    tool_profile_by_side: Mapping[str, Mapping[str, object]]
    tool_tcp_pose_flange_m_rad_by_side: Mapping[str, np.ndarray]
    scene_world_from_arm_base_by_side: Mapping[str, np.ndarray]
    scene_world_from_canonical_station: np.ndarray
    station_world_frame_id: str
    envelope_file_spec: TCPPoseEnvelopeFileSpec
    position_profile: RM65BimanualPositionProfile
    collision_scene_profile: str
    semantic_content_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.envelope_file_spec, TCPPoseEnvelopeFileSpec):
            raise TypeError("envelope_file_spec must be typed.")
        if not isinstance(self.position_profile, RM65BimanualPositionProfile):
            raise TypeError("position_profile must be typed.")
        ips = dict(self.arm_ip_by_side)
        tools = dict(self.tool_profile_by_side)
        tool_paths = dict(self.tool_profile_path_by_side)
        tcp = {
            side: np.asarray(value, dtype=float).reshape(6).copy()
            for side, value in dict(
                self.tool_tcp_pose_flange_m_rad_by_side
            ).items()
        }
        scene_mounts = {
            side: np.asarray(value, dtype=float).reshape(4, 4).copy()
            for side, value in dict(
                self.scene_world_from_arm_base_by_side
            ).items()
        }
        scene_world_from_station = np.asarray(
            self.scene_world_from_canonical_station,
            dtype=float,
        ).reshape(4, 4).copy()
        for name, values in (
            ("arm_ip_by_side", ips),
            ("tool_profile_by_side", tools),
            ("tool_profile_path_by_side", tool_paths),
            ("tool_tcp_pose_flange_m_rad_by_side", tcp),
            ("scene_world_from_arm_base_by_side", scene_mounts),
        ):
            if set(values) != set(_SIDES):
                raise ValueError(f"{name} must contain exactly left and right.")
        if any(not str(value).strip() for value in ips.values()):
            raise ValueError("both RM65 arm IP assignments must be non-empty.")
        if any(not np.all(np.isfinite(value)) for value in tcp.values()):
            raise ValueError("tool TCP poses must contain finite values.")
        if any(not np.all(np.isfinite(value)) for value in scene_mounts.values()):
            raise ValueError("scene arm mounts must contain finite values.")
        valid_station, station_reason = validate_pose_matrix(
            scene_world_from_station
        )
        if not valid_station:
            raise ValueError(
                "scene_world_from_canonical_station is invalid: "
                f"{station_reason}"
            )
        for value in (
            *tcp.values(),
            *scene_mounts.values(),
            scene_world_from_station,
        ):
            value.setflags(write=False)
        station_world_frame_id = str(self.station_world_frame_id).strip()
        if not station_world_frame_id:
            raise ValueError("station_world_frame_id must be non-empty.")
        object.__setattr__(self, "arm_ip_by_side", MappingProxyType(ips))
        object.__setattr__(self, "tool_profile_by_side", MappingProxyType(tools))
        object.__setattr__(
            self,
            "tool_profile_path_by_side",
            MappingProxyType(tool_paths),
        )
        object.__setattr__(
            self,
            "tool_tcp_pose_flange_m_rad_by_side",
            MappingProxyType(tcp),
        )
        object.__setattr__(
            self,
            "scene_world_from_arm_base_by_side",
            MappingProxyType(scene_mounts),
        )
        object.__setattr__(
            self,
            "scene_world_from_canonical_station",
            scene_world_from_station,
        )
        object.__setattr__(
            self,
            "station_world_frame_id",
            station_world_frame_id,
        )


def resolve_exclusive_rm65_skill_deployment_profile(
    profile: ExclusiveRM65SkillDeploymentProfile,
) -> ResolvedExclusiveRM65SkillDeploymentProfile:
    """Resolve local files without opening a device or constructing a solver."""

    if not isinstance(profile, ExclusiveRM65SkillDeploymentProfile):
        raise TypeError("profile must be ExclusiveRM65SkillDeploymentProfile.")
    arm_ip_assignment_path = _absolute_file(
        profile.arm_ip_assignment_path,
        field_name="arm_ip_assignment_path",
    )
    envelope_profile_path = _absolute_file(
        profile.envelope_profile_path,
        field_name="envelope_profile_path",
    )
    tool_paths = {
        side: _absolute_file(
            profile.tool_profile_path_by_side[side],
            field_name=f"tool_profile_path_by_side.{side}",
        )
        for side in _SIDES
    }
    scene_profile = build_rm65_dual_arm_scene_profile(
        profile.collision_scene_profile
    )
    scene_mounts = load_world_from_arm_base_transforms(scene_profile)
    scene_world_from_station = load_world_from_fixed_link_transform(
        scene_profile,
        link_name=_CANONICAL_STATION_LINK_NAME,
    )
    assignment = load_assignment(arm_ip_assignment_path)
    arm_ips = arm_ips_for_sides(assignment, sides=_SIDES)
    if set(arm_ips) != set(_SIDES):
        raise ValueError("arm IP assignment must bind both left and right RM65 arms.")
    tool_profiles = {
        side: load_rm65_tool_profile(tool_paths[side], side=side)
        for side in _SIDES
    }
    tcp_by_side = {
        side: tool_tcp_pose_flange_m_rad(tool_profiles[side])
        for side in _SIDES
    }
    envelope_spec = load_tcp_pose_envelope_file_spec(
        envelope_profile_path,
        active_sides=_SIDES,
    )
    position_profile = resolve_rm65_bimanual_position_profile(
        profile.position_profile_name
    )
    semantic_hash = canonical_content_hash(
        {
            "arm_ip_by_side": arm_ips,
            "tool_profile_by_side": {
                side: {
                    key: value
                    for key, value in tool_profiles[side].items()
                    if not str(key).startswith("_")
                }
                for side in _SIDES
            },
            "tool_tcp_pose_flange_m_rad_by_side": {
                side: tcp_by_side[side].tolist() for side in _SIDES
            },
            "envelope_profile": {
                "profile_id": envelope_spec.profile_id,
                "side_config_by_side": {
                    side: config.semantic_payload()
                    for side, config in envelope_spec.side_config_by_side.items()
                },
            },
            "position_profile": position_profile.to_metadata(),
            "collision_scene_profile": profile.collision_scene_profile,
            "station_geometry": {
                "source": "collision_scene_urdf_fixed_joint_chain",
                "world_frame_id": RM65_SKILL_SCENE_WORLD_FRAME_ID_V1,
                "world_from_arm_base_by_side": {
                    side: scene_mounts[side].tolist() for side in _SIDES
                },
                "canonical_station_link_name": _CANONICAL_STATION_LINK_NAME,
                "world_from_canonical_station": (
                    scene_world_from_station.tolist()
                ),
            },
        },
        domain="dexteleop.real_skill.rm65_deployment_profile.v2",
    )
    return ResolvedExclusiveRM65SkillDeploymentProfile(
        arm_ip_assignment_path=arm_ip_assignment_path,
        tool_profile_path_by_side=tool_paths,
        envelope_profile_path=envelope_profile_path,
        arm_ip_by_side=arm_ips,
        tool_profile_by_side=tool_profiles,
        tool_tcp_pose_flange_m_rad_by_side=tcp_by_side,
        scene_world_from_arm_base_by_side=scene_mounts,
        scene_world_from_canonical_station=scene_world_from_station,
        station_world_frame_id=RM65_SKILL_SCENE_WORLD_FRAME_ID_V1,
        envelope_file_spec=envelope_spec,
        position_profile=position_profile,
        collision_scene_profile=profile.collision_scene_profile,
        semantic_content_hash=semantic_hash,
    )


__all__ = [
    "ExclusiveRM65SkillDeploymentProfile",
    "RM65_SKILL_CANONICAL_STATION_FRAME_ID_V1",
    "RM65_SKILL_SCENE_WORLD_FRAME_ID_V1",
    "ResolvedExclusiveRM65SkillDeploymentProfile",
    "resolve_exclusive_rm65_skill_deployment_profile",
]
