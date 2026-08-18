from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..versions import SCENE_SPEC_VERSION, require_version
from .collision import CollisionSpec
from .object import FreeBodyObjectSpec
from .validation import finite_tuple
from .world import GroundSpec


DEFAULT_RM65_PHYSICS_URDF = (
    "assets/rm65_inspire_gripper_v3/"
    "rm65_dual_arm_wrist_cameras_v3_head_camera_4b2_cad_with_legacy_housing.urdf"
)
DEFAULT_CAMERA_REGISTRY = "robot_motion/configs/rm65_camera_registry.json"


@dataclass(frozen=True)
class SceneSpec:
    schema_version: str = SCENE_SPEC_VERSION
    profile_name: str = "rm65_bimanual_minimal_physics_v1"
    backend: str = "mujoco"
    backend_role: str = "canonical_task_physics"
    urdf_path: str = DEFAULT_RM65_PHYSICS_URDF
    camera_registry_path: str = DEFAULT_CAMERA_REGISTRY
    gravity_mps2: tuple[float, float, float] = (0.0, 0.0, -9.81)
    ground: GroundSpec = field(default_factory=GroundSpec)
    objects: tuple[FreeBodyObjectSpec, ...] = field(
        default_factory=lambda: (FreeBodyObjectSpec(name="debug_cube"),)
    )
    collision: CollisionSpec = field(default_factory=CollisionSpec)

    def __post_init__(self) -> None:
        require_version(self.schema_version, SCENE_SPEC_VERSION, field="scene spec version")
        if self.backend != "mujoco":
            raise ValueError("MuJoCo is the canonical task-physics backend for SceneSpec v1.")
        if not self.profile_name or not self.urdf_path or not self.camera_registry_path:
            raise ValueError("Scene profile and asset paths must be non-empty.")
        gravity = finite_tuple(self.gravity_mps2, 3, "gravity_mps2")
        names = [obj.name for obj in self.objects]
        if len(names) != len(set(names)):
            raise ValueError("Scene object names must be unique.")
        object.__setattr__(self, "gravity_mps2", gravity)
        object.__setattr__(self, "objects", tuple(self.objects))

    @classmethod
    def kinematic_replay(cls) -> "SceneSpec":
        return cls(
            profile_name="rm65_bimanual_kinematic_replay_v1",
            backend_role="kinematic_replay_diagnostic",
            gravity_mps2=(0.0, 0.0, 0.0),
            ground=GroundSpec(enabled=False),
            objects=(),
            collision=CollisionSpec(enabled=False, self_collision_enabled=False),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_name": self.profile_name,
            "backend": self.backend,
            "backend_role": self.backend_role,
            "urdf_path": self.urdf_path,
            "camera_registry_path": self.camera_registry_path,
            "gravity_mps2": list(self.gravity_mps2),
            "ground": self.ground.to_dict(),
            "objects": [obj.to_dict() for obj in self.objects],
            "collision": self.collision.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SceneSpec":
        return cls(
            schema_version=str(payload.get("schema_version", SCENE_SPEC_VERSION)),
            profile_name=str(
                payload.get("profile_name", "rm65_bimanual_minimal_physics_v1")
            ),
            backend=str(payload.get("backend", "mujoco")),
            backend_role=str(payload.get("backend_role", "canonical_task_physics")),
            urdf_path=str(payload.get("urdf_path", DEFAULT_RM65_PHYSICS_URDF)),
            camera_registry_path=str(
                payload.get("camera_registry_path", DEFAULT_CAMERA_REGISTRY)
            ),
            gravity_mps2=tuple(payload.get("gravity_mps2", (0.0, 0.0, -9.81))),
            ground=GroundSpec.from_dict(payload.get("ground")),
            objects=tuple(
                FreeBodyObjectSpec.from_dict(item)
                for item in payload.get("objects", ({"name": "debug_cube"},))
            ),
            collision=CollisionSpec.from_dict(payload.get("collision")),
        )


__all__ = [
    "DEFAULT_CAMERA_REGISTRY",
    "DEFAULT_RM65_PHYSICS_URDF",
    "SceneSpec",
]
