"""Schema versions for the cross-domain simulation contract."""

ACTION_SPEC_VERSION = "dexteleop.sim.action_spec.v1"
OBSERVATION_SPEC_VERSION = "dexteleop.sim.observation_spec.v1"
TIMING_SPEC_VERSION = "dexteleop.sim.timing_spec.v1"
SCENE_SPEC_VERSION = "dexteleop.sim.scene_spec.v1"
SCENE_SPEC_VERSION_V2 = "dexteleop.sim.scene_spec.v2"
SIMULATION_CONTRACT_VERSION = "dexteleop.sim.contract.v1"
MUJOCO_COLLISION_PROFILE_VERSION = "dexteleop.mujoco.collision_groups.v1"


def require_version(actual: object, expected: str, *, field: str) -> str:
    value = str(actual)
    if value != expected:
        raise ValueError(f"Unsupported {field} {value!r}; expected {expected!r}.")
    return value


__all__ = [
    "ACTION_SPEC_VERSION",
    "MUJOCO_COLLISION_PROFILE_VERSION",
    "OBSERVATION_SPEC_VERSION",
    "SCENE_SPEC_VERSION",
    "SCENE_SPEC_VERSION_V2",
    "SIMULATION_CONTRACT_VERSION",
    "TIMING_SPEC_VERSION",
    "require_version",
]
