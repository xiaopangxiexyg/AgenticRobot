from __future__ import annotations

import ast
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from robot_motion.rebot_b601_rs import (
    B601_RS_ARM_JOINT_NAMES,
    B601_RS_END_EFFECTOR_FRAME,
    B601_RS_URDF_RELATIVE_PATH,
    B601DampedLeastSquaresIK,
    B601PinocchioModel,
)
from sim.mujoco.rebot_b601_rs import (
    B601_RS_GRIPPER_OPEN_QPOS,
    B601_RS_SIM_ARM_KEY,
    B601JointPositionCommand,
    B601PositionDriver,
    build_b601_mjcf_xml,
    build_b601_mujoco_scene,
)
from sim.mujoco.rebot_b601_rs.scene import b601_position_actuator_name

REPO_ROOT = Path(__file__).resolve().parents[1]
REBOT_SOURCE_ROOT = REPO_ROOT / "hardware" / "reBotArm_control_py"
pytestmark = [pytest.mark.optional, pytest.mark.contract]


@pytest.fixture(scope="module")
def model_and_scene():
    pytest.importorskip("mujoco")
    if not (REBOT_SOURCE_ROOT / B601_RS_URDF_RELATIVE_PATH).is_file():
        pytest.skip("initialize the pinned reBotArm_control_py gitlink")
    pinocchio_model = B601PinocchioModel.from_rebot_source_root(REBOT_SOURCE_ROOT)
    scene = build_b601_mujoco_scene(REBOT_SOURCE_ROOT)
    return pinocchio_model, scene


def _command(sequence: int, qpos, timestamp: float) -> B601JointPositionCommand:
    return B601JointPositionCommand(
        sequence=sequence,
        source_timestamp_sec=timestamp,
        receive_timestamp_sec=timestamp,
        qpos=tuple(qpos),
    )


def test_command_rejects_boolean_sequence() -> None:
    with pytest.raises(TypeError, match="non-negative integer"):
        B601JointPositionCommand(
            sequence=True,
            source_timestamp_sec=0.0,
            receive_timestamp_sec=0.0,
            qpos=(0.0, 1.0, 1.0, 0.0, 0.0, 0.0),
        )


def test_compiled_scene_preserves_full_chain_and_six_arm_actuators(
    model_and_scene,
) -> None:
    _pinocchio_model, scene = model_and_scene
    mujoco = scene.mujoco

    assert scene.model.nq == 8
    assert scene.model.nv == 8
    assert scene.model.nbody == 11
    assert scene.model.nu == 6
    assert scene.model.opt.timestep == pytest.approx(0.001)
    np.testing.assert_array_equal(scene.model.opt.gravity, [0.0, 0.0, 0.0])
    assert int(scene.model.opt.integrator) == int(
        mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    )
    assert scene.arm_specs[B601_RS_SIM_ARM_KEY].joint_names == (B601_RS_ARM_JOINT_NAMES)
    assert (
        scene.arm_specs[B601_RS_SIM_ARM_KEY].controlled_body
        == B601_RS_END_EFFECTOR_FRAME
    )
    assert scene.has_body("base_link")
    assert scene.has_body(B601_RS_END_EFFECTOR_FRAME)
    assert scene.gripper_specs == {}
    assert scene.scene_spec is None


def test_mjcf_references_pinned_meshes_in_place_without_asset_copies() -> None:
    root = ET.fromstring(build_b601_mjcf_xml(REBOT_SOURCE_ROOT))
    mesh_paths = [
        Path(element.attrib["file"]).resolve(strict=True)
        for element in root.findall("./asset/mesh")
    ]
    resolved_source = REBOT_SOURCE_ROOT.resolve(strict=True)

    assert len(mesh_paths) == 23
    assert len(set(mesh_paths)) == 23
    for mesh_path in mesh_paths:
        assert mesh_path.is_file()
        mesh_path.relative_to(resolved_source)


def test_actuator_ranges_are_exact_urdf_limits_and_provisional_efforts(
    model_and_scene,
) -> None:
    pinocchio_model, scene = model_and_scene
    lower, upper = pinocchio_model.arm_position_limits()
    actuator_ids = np.asarray(
        [
            scene.mujoco.mj_name2id(
                scene.model,
                scene.mujoco.mjtObj.mjOBJ_ACTUATOR,
                b601_position_actuator_name(name),
            )
            for name in B601_RS_ARM_JOINT_NAMES
        ],
        dtype=np.int32,
    )

    assert np.all(actuator_ids >= 0)
    np.testing.assert_allclose(scene.model.actuator_ctrlrange[actuator_ids, 0], lower)
    np.testing.assert_allclose(scene.model.actuator_ctrlrange[actuator_ids, 1], upper)
    np.testing.assert_allclose(
        scene.model.actuator_forcerange[actuator_ids, 1],
        [36.0, 36.0, 36.0, 14.0, 14.0, 14.0],
    )
    np.testing.assert_allclose(
        scene.model.actuator_forcerange[actuator_ids, 0],
        [-36.0, -36.0, -36.0, -14.0, -14.0, -14.0],
    )


def test_pinocchio_and_mujoco_fk_match_over_deterministic_interior_sweep(
    model_and_scene,
) -> None:
    pinocchio_model, scene = model_and_scene
    lower, upper = pinocchio_model.arm_position_limits()
    rng = np.random.default_rng(601)

    for _ in range(16):
        qpos = lower + (0.15 + 0.70 * rng.random(6)) * (upper - lower)
        scene.set_arm_qpos(B601_RS_SIM_ARM_KEY, qpos)
        np.testing.assert_allclose(
            scene.body_pose(B601_RS_END_EFFECTOR_FRAME),
            pinocchio_model.forward_kinematics(qpos),
            atol=5e-11,
            rtol=0.0,
        )


def test_driver_schedules_500hz_commands_against_1000hz_physics(
    model_and_scene,
) -> None:
    _pinocchio_model, scene = model_and_scene
    driver = B601PositionDriver(scene)
    driver.reset()

    ticks = [driver.step_physics() for _ in range(1000)]

    assert sum(tick.command_due for tick in ticks) == 500
    assert ticks[0].command_due is True
    assert ticks[1].command_due is False
    assert ticks[2].command_due is True
    assert driver.physics_time_sec == pytest.approx(1.0)
    assert all(tick.state_valid for tick in ticks)


def test_driver_rejects_early_stale_and_out_of_limit_commands_without_mutation(
    model_and_scene,
) -> None:
    _pinocchio_model, scene = model_and_scene
    driver = B601PositionDriver(scene)
    initial = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    first_target = initial + np.array([0.05, 0.02, 0.01, 0.0, 0.0, 0.0])
    second_target = first_target + 0.01
    driver.reset(initial)

    accepted = driver.step_physics(_command(1, first_target, 1.0))
    early = driver.step_physics(_command(2, second_target, 1.001))
    stale = driver.step_physics(_command(1, second_target, 1.002))
    driver.step_physics()
    stale_source = driver.step_physics(_command(2, second_target, 0.5))
    driver.step_physics()
    stale_receive = driver.step_physics(
        B601JointPositionCommand(
            sequence=2,
            source_timestamp_sec=1.1,
            receive_timestamp_sec=0.5,
            qpos=tuple(second_target),
        )
    )
    driver.step_physics()
    outside = second_target.copy()
    outside[0] = 3.0
    out_of_limit = driver.step_physics(_command(3, outside, 2.0))

    assert accepted.command_applied is True
    assert early.command_applied is False
    assert early.rejection_reason == "command_before_deadline"
    assert stale.command_applied is False
    assert stale.rejection_reason == "stale_sequence"
    assert stale_source.rejection_reason == "stale_source_timestamp"
    assert stale_receive.rejection_reason == "stale_receive_timestamp"
    assert out_of_limit.command_applied is False
    assert out_of_limit.rejection_reason == "joint_limit_violation"
    assert out_of_limit.accepted_sequence == 1
    assert out_of_limit.accepted_source_timestamp_sec == pytest.approx(1.0)
    assert out_of_limit.accepted_receive_timestamp_sec == pytest.approx(1.0)
    assert out_of_limit.accepted_frame_id == "b601_rs/joint_position"
    np.testing.assert_allclose(out_of_limit.target_qpos, first_target)


def test_driver_metadata_reports_actual_gravity_and_m2_safety_boundary() -> None:
    scene = build_b601_mujoco_scene(
        REBOT_SOURCE_ROOT,
        gravity_mps2=(0.0, 0.0, -9.81),
    )
    metadata = B601PositionDriver(scene).execution_metadata

    np.testing.assert_allclose(metadata["gravity_mps2"], [0.0, 0.0, -9.81])
    assert metadata["gravity_compensation"] == "not_implemented_m2"
    assert metadata["target_slew_limit"] == "not_implemented_m2"
    assert metadata["state_limit_guard"] == "not_implemented_m2"
    assert metadata["command_scope"] == "modest_visualization_targets_only"


def test_ik_output_tracks_through_the_physical_position_servo(
    model_and_scene,
) -> None:
    pinocchio_model, scene = model_and_scene
    initial = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    target_qpos = np.array([0.15, 1.1, 1.15, 0.1, -0.1, 0.2])
    solution = B601DampedLeastSquaresIK(pinocchio_model).solve(
        pinocchio_model.forward_kinematics(target_qpos),
        seed_qpos=initial,
    )
    assert solution.success is True
    driver = B601PositionDriver(scene)
    driver.reset(initial)

    tick = driver.step_physics(_command(0, solution.qpos, 0.0))
    for _ in range(1999):
        tick = driver.step_physics()

    assert tick.state_valid is True
    assert tick.accepted_sequence == 0
    np.testing.assert_allclose(tick.measured_qpos, solution.qpos, atol=2e-6)
    for name, expected in B601_RS_GRIPPER_OPEN_QPOS.items():
        assert scene.data.qpos[scene.joint_qpos_addr(name)] == pytest.approx(expected)
        assert scene.data.qvel[scene.joint_dof_addr(name)] == pytest.approx(0.0)


def test_b601_sim_package_has_no_hardware_or_rm65_runtime_imports() -> None:
    forbidden = (
        "control_composition",
        "control_station",
        "hardware",
        "sim.mujoco.actuation",
        "sim.mujoco.backend",
        "sim.mujoco.ik",
        "sim.rm65",
        "teleop_runtime",
    )
    violations: dict[str, list[str]] = {}

    for path in sorted((REPO_ROOT / "sim" / "mujoco" / "rebot_b601_rs").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imports.add(node.module)
        matched = sorted(
            name
            for name in imports
            if any(name == root or name.startswith(f"{root}.") for root in forbidden)
        )
        if matched:
            violations[path.name] = matched

    assert violations == {}
