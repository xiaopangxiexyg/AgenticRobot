from __future__ import annotations

from .constants import DEFAULT_COLLISION_SCENE


def build_rm65_dual_arm_scene_profile(name: str = DEFAULT_COLLISION_SCENE):
    from rm65_inspire_tac.pipeline import build_dual_arm_scene_profile as _build

    return _build(str(name or DEFAULT_COLLISION_SCENE))


def build_dual_arm_scene_profile(name: str = DEFAULT_COLLISION_SCENE):
    return build_rm65_dual_arm_scene_profile(name)


__all__ = [
    "DEFAULT_COLLISION_SCENE",
    "build_dual_arm_scene_profile",
    "build_rm65_dual_arm_scene_profile",
]
