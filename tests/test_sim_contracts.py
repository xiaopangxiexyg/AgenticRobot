from __future__ import annotations

from pathlib import Path

import pytest

from robot_motion.deploy.schema import ACTION_DIM, OBSERVATION_STATE_DIM
from robot_motion.position_control import PositionControlRates
from sim.contracts import SceneSpec, SimulationContract
from sim.contracts.scene import DEFAULT_RM65_PHYSICS_URDF


CONTRACT = Path("sim/configs/rm65_bimanual_physics_v1.json")


def test_default_scene_uses_additive_4b2_cad_gripper_urdf() -> None:
    expected = (
        "assets/rm65_inspire_gripper_v3/"
        "rm65_dual_arm_wrist_cameras_v3_head_camera_4b2_cad_with_legacy_housing.urdf"
    )

    assert DEFAULT_RM65_PHYSICS_URDF == expected
    assert SceneSpec().urdf_path == expected
    assert SimulationContract.load(CONTRACT).scene.urdf_path == expected


def test_versioned_simulation_contract_pins_existing_project_contracts() -> None:
    contract = SimulationContract.load(CONTRACT)
    rates = PositionControlRates()

    assert contract.action.vector_dim == ACTION_DIM
    assert contract.observation.state_dim == OBSERVATION_STATE_DIM
    assert contract.timing.sample_hz == rates.sample_hz
    assert contract.timing.reference_hz == rates.reference_hz
    assert contract.timing.ik_hz == rates.ik_hz
    assert contract.timing.command_hz == rates.command_hz
    assert contract.timing.physics_hz >= contract.timing.command_hz
    assert contract.scene.backend == "mujoco"
    assert contract.scene.backend_role == "canonical_task_physics"
    assert contract.metadata["sapien_role"] == "teleop_debug_preview_only"


def test_simulation_contract_round_trip_is_lossless() -> None:
    contract = SimulationContract.load(CONTRACT)

    restored = SimulationContract.from_dict(contract.to_dict())

    assert restored.to_dict() == contract.to_dict()


def test_scene_visual_colors_round_trip_and_are_not_mutable() -> None:
    colors = {"sensor": [0.5, 0.5, 0.5, 1.0]}
    scene = SceneSpec(visual_rgba_by_link=colors)
    colors["sensor"][0] = 0.0
    assert scene.visual_rgba_by_link["sensor"] == (0.5, 0.5, 0.5, 1.0)
    assert SceneSpec.from_dict(scene.to_dict()).to_dict() == scene.to_dict()
    with pytest.raises(TypeError):
        scene.visual_rgba_by_link["other"] = (1.0,) * 4
    assert "visual_rgba_by_link" not in SceneSpec().to_dict()


@pytest.mark.parametrize("colors", (
    None, [], {"": [0, 0, 0, 1]}, {" sensor": [0, 0, 0, 1]},
    {"sensor": [0, 0, 0]}, {"sensor": [0, 0, 0, float("nan")]},
    {"sensor": [-0.1, 0, 0, 1]}, {"sensor": [0, 0, 0, 1.1]},
))
def test_scene_rejects_invalid_visual_color_overrides(colors) -> None:
    with pytest.raises(ValueError):
        SceneSpec(visual_rgba_by_link=colors)


def test_contract_rejects_unknown_nested_schema_version() -> None:
    payload = SimulationContract.load(CONTRACT).to_dict()
    payload["scene"]["schema_version"] = "dexteleop.sim.scene_spec.v999"

    with pytest.raises(ValueError, match="scene spec version"):
        SimulationContract.from_dict(payload)


def test_scene_profiles_separate_physics_mainline_from_kinematic_replay() -> None:
    physical = SimulationContract.load(CONTRACT).scene
    kinematic = SceneSpec.kinematic_replay()

    assert physical.gravity_mps2 == (0.0, 0.0, -9.81)
    assert physical.collision.enabled is True
    assert physical.ground.enabled is True
    assert [obj.name for obj in physical.objects] == ["debug_cube"]
    assert physical.objects[0].inertia_diagonal_kg_m2 is not None

    assert kinematic.backend == "mujoco"
    assert kinematic.backend_role == "kinematic_replay_diagnostic"
    assert kinematic.gravity_mps2 == (0.0, 0.0, 0.0)
    assert kinematic.collision.enabled is False
    assert kinematic.objects == ()
