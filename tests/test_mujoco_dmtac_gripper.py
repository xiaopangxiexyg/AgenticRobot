from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from collection_runtime.stream_sink import NativeStreamEpisodeWriter
from control_station.composition import (
    build_mujoco_component_human_proposal_adapter,
)
from control_station.sources import (
    admit_human_command_emission,
    admit_human_observation_cut,
)
from control_station.sources.teleop_pose import (
    PassiveTeleopComponentSourceSpec,
    PassiveTeleopComponentSpec,
)
from dataset_contracts.stream_catalog import NativeStreamContract
from dataset_interface.streams import load_native_stream_episode
from robot_motion.action import (
    SCHEDULED_REFERENCE_SEGMENT_SIM_V1,
    ArmActionSpec,
    CanonicalActionSafetyAnchor,
    CanonicalKeyframeSafetyProfile,
    HandActionSpec,
    HandQposSafetyLimits,
    apply_canonical_keyframe_safety,
)
from robot_motion.position_control import PositionKeyframeSafetyConfig
from robot_motion.runtime import (
    EffectorCommand,
    ObservationSampleRef,
    ReceiverObservationCutBuilder,
    TeleopCommand,
)
from sim.contracts import SimulationContract, TimingSpec
from sim.mujoco.actuation import (
    MujocoDmtacGripperExecutor,
    bounded_cad_dmtac_cube_contact_gripper_profile,
    cad_bimanual_dmtac_gripper_profile,
    cad_dmtac_cube_contact_gripper_profile,
    identified_bimanual_dmtac_gripper_profile,
    point_grasp_cad_bimanual_dmtac_gripper_profile,
    rm65_bimanual_udp_feedback_profile,
    strong_cad_bimanual_dmtac_gripper_profile,
)
from sim.mujoco.adapters import (
    MujocoPositionExecutionConfig,
    MujocoPositionExecutionRig,
)
from sim.mujoco.artifacts import ModelArtifactCache
from sim.mujoco.deploy import MujocoDeployConfig, MujocoDeployRuntime
from sim.mujoco.deploy.pose_resolution import MujocoCoupledPoseToQposResolver
from sim.mujoco.ik import RM65CanonicalIK
from sim.mujoco.modeling.gripper_actuators import add_gripper_position_actuators
from teleop_runtime.action.human_proposal import HumanCommandEmission

CONTRACT_PATH = "sim/configs/rm65_bimanual_physics_v1.json"
LEGACY_URDF_PATH = (
    "assets/rm65_inspire_gripper_v3/rm65_dual_arm_wrist_cameras_v3_head_camera.urdf"
)


def _timing() -> TimingSpec:
    return TimingSpec(
        sample_hz=15.0,
        policy_hz=15.0,
        reference_hz=60.0,
        ik_hz=60.0,
        command_hz=120.0,
        physics_hz=1000.0,
        parameter_source="unit_test_15_60_120_gripper",
    )


def _scene():
    contract = SimulationContract.load(CONTRACT_PATH)
    # These tests freeze the identified two-joint proxy actuator trace.  Keep
    # that compatibility path explicit now that the canonical contract uses
    # the additive six-joint CAD gripper.
    contract = replace(
        contract,
        scene=replace(
            contract.scene,
            urdf_path=LEGACY_URDF_PATH,
            collision=replace(contract.scene.collision, excluded_body_pairs=()),
        ),
    )
    arm_profile = rm65_bimanual_udp_feedback_profile()
    gripper_profile = identified_bimanual_dmtac_gripper_profile()
    scene, _model_artifact = ModelArtifactCache().load_scene(
        contract,
        position_actuation_profile=arm_profile,
        gripper_actuation_profile=gripper_profile,
    )
    return scene, arm_profile, gripper_profile


def _cad_scene():
    contract = SimulationContract.load(CONTRACT_PATH)
    arm_profile = rm65_bimanual_udp_feedback_profile()
    gripper_profile = cad_bimanual_dmtac_gripper_profile()
    scene, _model_artifact = ModelArtifactCache().load_scene(
        contract,
        position_actuation_profile=arm_profile,
        gripper_actuation_profile=gripper_profile,
    )
    return scene, arm_profile, gripper_profile


def test_identified_dmtac_profile_freezes_bilateral_stage_semantics() -> None:
    profile = identified_bimanual_dmtac_gripper_profile()
    assert profile.name == "dmtac_bimanual_position_20260722.v1"
    assert profile.command_hz == 20.0
    assert profile.send_threshold == 0.002
    assert profile.command_to_actuator_delay_sec_by_gripper == {
        "left_gripper": 0.0592514185,
        "right_gripper": 0.0640776765,
    }
    response = profile.normalized_position_profile()
    assert response.parameters(
        "left", direction="close"
    ).max_rate_per_sec == pytest.approx(1.4928638795218356)
    assert response.parameters("left", direction="open").delay_sec == pytest.approx(
        0.046897892
    )
    assert response.parameters(
        "right", direction="close"
    ).max_rate_per_sec == pytest.approx(1.3773242281700613)
    assert response.parameters(
        "right", direction="open"
    ).acceleration_per_sec2 == pytest.approx(12.7875)
    assert response.source_payload_sha256 == (
        "3cbb510090d008d1c58ffc5c5e721ad3a2e3b45cda1285dcacb29ef121860691"
    )
    assert "contact_unmodeled" in profile.calibration_status
    assert profile.qpos_from_close("left_gripper", 0.0) == (0.0, 0.0)
    assert profile.qpos_from_close("left_gripper", 1.0) == (-0.035, -0.035)
    assert profile.contact_tuning.friction is None
    assert profile.contact_tuning.solref is None
    assert profile.contact_tuning.solimp is None


def test_cad_dmtac_profile_preserves_official_six_joint_open_close_semantics() -> None:
    profile = cad_bimanual_dmtac_gripper_profile()

    assert profile.name == "dmtac_bimanual_position_eg2_4b2_cad_20260818.v1"
    for name, side in (
        ("left_eg2_4b2_cad", "left"),
        ("right_eg2_4b2_cad", "right"),
    ):
        assert profile.side_by_gripper[name] == side
        assert len(profile.joint_names_by_gripper[name]) == 6
        assert profile.open_qpos_by_gripper[name] == (0.0,) * 6
        assert profile.closed_qpos_by_gripper[name] == (0.864,) * 6
        assert profile.qpos_from_close(name, 0.0) == (0.0,) * 6
        assert profile.qpos_from_close(name, 1.0) == (0.864,) * 6
        assert profile.master_joint_by_gripper[name].endswith("_r_1_joint")
        assert len(profile.equality_joint_pairs(name)) == 5
        assert all(
            master == profile.master_joint_by_gripper[name]
            for master, _mimic in profile.equality_joint_pairs(name)
        )


def test_installed_cad_operating_profile_preserves_force_and_contact_tuning():
    baseline = cad_dmtac_cube_contact_gripper_profile()
    bounded = bounded_cad_dmtac_cube_contact_gripper_profile()
    changed = {"name", "closed_qpos_by_gripper", "calibration_status", "evidence"}
    assert {k:v for k,v in baseline.to_metadata().items() if k not in changed} == {
        k:v for k,v in bounded.to_metadata().items() if k not in changed}
    for name in baseline.joint_names_by_gripper:
        assert baseline.closed_qpos_by_gripper[name] == (.864,)*6
        assert bounded.qpos_from_close(name, 1.) == (.8,)*6


@pytest.mark.parametrize("hard_range", ("-0.035 0", "-0.04 0.01"))
def test_compiled_gripper_operating_range_uses_the_servo_mapping(hard_range):
    profile = identified_bimanual_dmtac_gripper_profile()
    root = ET.Element("mujoco")
    body = ET.SubElement(root, "worldbody")
    for names in profile.joint_names_by_gripper.values():
        for name in names:
            ET.SubElement(body, "joint", name=name, range=hard_range)
    add_gripper_position_actuators(root, profile)
    for joint in body.findall("joint"):
        assert joint.get("limited") == "true"
        assert tuple(map(float, joint.get("range").split())) == (-.035, 0.)


@pytest.mark.parametrize("hard_range", ("0 .035", "nan .035", ".1 0", "0"))
def test_compiled_gripper_operating_range_cannot_exceed_model_limits(hard_range):
    profile = identified_bimanual_dmtac_gripper_profile()
    root = ET.Element("mujoco")
    body = ET.SubElement(root, "worldbody")
    for names in profile.joint_names_by_gripper.values():
        for name in names:
            ET.SubElement(body, "joint", name=name, range=hard_range)
    with pytest.raises(ValueError, match="operating range"):
        add_gripper_position_actuators(root, profile)


def test_gripper_executor_rejects_a_second_normalization_mapping():
    profile = bounded_cad_dmtac_cube_contact_gripper_profile()
    scene = SimpleNamespace(gripper_specs={name: SimpleNamespace(
        joint_names=joints, open_qpos=profile.open_qpos_by_gripper[name],
        closed_qpos=(.864,)*6) for name,joints in profile.joint_names_by_gripper.items()})
    with pytest.raises(ValueError, match="operating range mismatch"):
        MujocoDmtacGripperExecutor(scene, profile)


def test_point_grasp_cad_dmtac_profile_has_separate_revolute_gain_identity() -> None:
    profile = point_grasp_cad_bimanual_dmtac_gripper_profile()

    assert profile.name == "dmtac_bimanual_position_eg2_4b2_cad_20260829.v2"
    assert profile.position_kp_n_m == pytest.approx(5.0)
    assert profile.position_kv_n_s_m == pytest.approx(0.1)
    assert profile.force_limit_n == pytest.approx(0.5)
    assert profile.joint_damping_n_s_m == pytest.approx(0.02)
    assert profile.joint_frictionloss_n == pytest.approx(0.005)
    for name, side in (
        ("left_eg2_4b2_cad", "left"),
        ("right_eg2_4b2_cad", "right"),
    ):
        assert profile.side_by_gripper[name] == side
        assert len(profile.joint_names_by_gripper[name]) == 6
        assert profile.open_qpos_by_gripper[name] == (0.0,) * 6
        assert profile.closed_qpos_by_gripper[name] == (0.864,) * 6
        assert profile.qpos_from_close(name, 0.0) == (0.0,) * 6
        assert profile.qpos_from_close(name, 1.0) == (0.864,) * 6
        assert profile.master_joint_by_gripper[name].endswith("_r_1_joint")
        assert len(profile.equality_joint_pairs(name)) == 5
        assert all(
            master == profile.master_joint_by_gripper[name]
            for master, _mimic in profile.equality_joint_pairs(name)
        )


def test_strong_cad_dmtac_profile_changes_only_reviewed_force_semantics() -> None:
    base = point_grasp_cad_bimanual_dmtac_gripper_profile()
    strong = strong_cad_bimanual_dmtac_gripper_profile()

    assert strong.name == "dmtac_bimanual_position_eg2_4b2_cad_20260831.v3_strong"
    assert strong.force_limit_n == pytest.approx(0.75)
    assert strong.position_kp_n_m == base.position_kp_n_m
    assert strong.position_kv_n_s_m == base.position_kv_n_s_m
    assert strong.joint_damping_n_s_m == base.joint_damping_n_s_m
    assert strong.joint_frictionloss_n == base.joint_frictionloss_n
    assert strong.joint_names_by_gripper == base.joint_names_by_gripper


@pytest.mark.slow_sim
def test_cad_dmtac_scene_compiles_six_actuators_and_explicit_mimic_constraints() -> (
    None
):
    pytest.importorskip("mujoco")
    scene, _arm_profile, profile = _cad_scene()

    assert set(scene.gripper_specs) == {
        "left_eg2_4b2_cad",
        "right_eg2_4b2_cad",
    }
    assert scene.model.nu == 24
    assert scene.model.neq == 10
    for name, joint_names in profile.joint_names_by_gripper.items():
        assert all(
            scene.mujoco.mj_name2id(
                scene.model,
                scene.mujoco.mjtObj.mjOBJ_ACTUATOR,
                profile.actuator_name(name, index),
            )
            >= 0
            for index in range(len(joint_names))
        )
        assert all(
            scene.mujoco.mj_name2id(
                scene.model,
                scene.mujoco.mjtObj.mjOBJ_EQUALITY,
                profile.equality_name(name, index),
            )
            >= 0
            for index in range(len(profile.equality_joint_pairs(name)))
        )


@pytest.mark.slow_sim
def test_gripper_executor_delays_then_acceleration_limits_force_limited_ctrl() -> None:
    pytest.importorskip("mujoco")
    scene, _arm_profile, profile = _scene()
    executor = MujocoDmtacGripperExecutor(scene, profile)
    executor.reset({"left_gripper": 0.0, "right_gripper": 0.0})
    executor.queue_close_targets({"left_gripper": 1.0})

    for tick_index in range(60):
        tick = executor.before_physics_step(
            physics_time_sec=tick_index / 1000.0,
            timestep_sec=0.001,
        )
        scene.mujoco.mj_step(scene.model, scene.data)
    assert tick.accepted_close_by_gripper["left_gripper"] == 1.0
    assert tick.servo_goal_close_by_gripper["left_gripper"] == 0.0
    assert tick.actuator_target_close_by_gripper["left_gripper"] == 0.0

    tick = executor.before_physics_step(
        physics_time_sec=0.060,
        timestep_sec=0.001,
    )
    scene.mujoco.mj_step(scene.model, scene.data)
    assert tick.servo_goal_close_by_gripper["left_gripper"] == 1.0
    assert tick.actuator_target_close_by_gripper["left_gripper"] == pytest.approx(
        0.00001377982793408203
    )
    assert tick.actuator_target_rate_by_gripper["left_gripper"] == pytest.approx(
        0.01377982793408203
    )
    actuator_ids = [
        scene.mujoco.mj_name2id(
            scene.model,
            scene.mujoco.mjtObj.mjOBJ_ACTUATOR,
            profile.actuator_name("left_gripper", index),
        )
        for index in range(2)
    ]
    assert np.allclose(
        scene.data.ctrl[actuator_ids],
        (-4.82293977692871e-7, -4.82293977692871e-7),
    )
    assert np.all(scene.model.actuator_forcerange[actuator_ids, 1] == 20.0)


@pytest.mark.slow_sim
def test_identified_executor_full_stroke_matches_trace_envelope_on_both_sides() -> None:
    pytest.importorskip("mujoco")
    scene, _arm_profile, profile = _scene()
    executor = MujocoDmtacGripperExecutor(scene, profile)
    executor.reset({"left_gripper": 0.0, "right_gripper": 0.0})
    executor.queue_close_targets({"left_gripper": 1.0, "right_gripper": 1.0})

    first_visible_time = {"left": None, "right": None}
    full_close_time = {"left": None, "right": None}
    first_measured_time = {"left": None, "right": None}
    full_measured_time = {"left": None, "right": None}
    for tick_index in range(1201):
        tick = executor.before_physics_step(
            physics_time_sec=tick_index / 1000.0,
            timestep_sec=0.001,
        )
        scene.mujoco.mj_step(scene.model, scene.data)
        left = tick.actuator_target_close_by_gripper["left_gripper"]
        right = tick.actuator_target_close_by_gripper["right_gripper"]
        for side, value in (("left", left), ("right", right)):
            if first_visible_time[side] is None and value >= 0.005:
                first_visible_time[side] = tick_index / 1000.0
            if full_close_time[side] is None and value >= 0.999:
                full_close_time[side] = tick_index / 1000.0
        measured = executor.measured_close_by_gripper()
        for side, name in (("left", "left_gripper"), ("right", "right_gripper")):
            if first_measured_time[side] is None and measured[name] >= 0.005:
                first_measured_time[side] = tick_index / 1000.0
            if full_measured_time[side] is None and measured[name] >= 0.999:
                full_measured_time[side] = tick_index / 1000.0

    assert first_visible_time == {"left": 0.086, "right": 0.091}
    assert full_close_time == {"left": 0.822, "right": 0.875}
    assert first_measured_time == {"left": 0.103, "right": 0.108}
    assert full_measured_time == {"left": 0.850, "right": 0.904}


@pytest.mark.slow_sim
def test_gripper_hold_cancels_commands_still_in_transport_delay() -> None:
    pytest.importorskip("mujoco")
    scene, _arm_profile, profile = _scene()
    executor = MujocoDmtacGripperExecutor(scene, profile)
    executor.reset({"left_gripper": 0.0, "right_gripper": 0.0})
    executor.queue_close_targets({"left_gripper": 1.0})
    executor.before_physics_step(physics_time_sec=0.0, timestep_sec=0.001)
    assert executor.snapshot()["pending_delayed_targets"] == 1

    executor.hold_targets()
    tick = executor.before_physics_step(
        physics_time_sec=0.500,
        timestep_sec=0.001,
    )
    assert tick.servo_goal_close_by_gripper["left_gripper"] == 0.0
    assert tick.actuator_target_close_by_gripper["left_gripper"] == 0.0
    assert executor.snapshot()["pending_delayed_targets"] == 0


@pytest.mark.slow_sim
def test_gripper_snapshot_is_bound_to_completed_physics_time() -> None:
    pytest.importorskip("mujoco")
    scene, _arm_profile, profile = _scene()
    executor = MujocoDmtacGripperExecutor(scene, profile)
    initial = {"left_gripper": 0.0, "right_gripper": 0.0}
    executor.reset(initial)
    executor.before_physics_step(
        physics_time_sec=0.0,
        timestep_sec=0.001,
    )
    scene.mujoco.mj_step(scene.model, scene.data)
    snapshot = executor.snapshot()

    assert snapshot["physics_time_sec"] == pytest.approx(float(scene.data.time))

    restored = MujocoDmtacGripperExecutor(scene, profile)
    with pytest.raises(ValueError, match="physics_time_sec.*does not match"):
        restored.reset(
            initial,
            initial_physics_time_sec=float(scene.data.time) + 0.001,
            initial_state=snapshot,
        )

    wrong_profile = dict(snapshot)
    wrong_profile["profile"] = {"name": "different-profile"}
    with pytest.raises(ValueError, match="profile.*does not match"):
        restored.reset(
            initial,
            initial_physics_time_sec=float(scene.data.time),
            initial_state=wrong_profile,
        )

    mismatched_target = dict(initial)
    mismatched_target["left_gripper"] = 0.5
    with pytest.raises(ValueError, match="actuator target.*does not match"):
        restored.reset(
            mismatched_target,
            initial_physics_time_sec=float(scene.data.time),
            initial_state=snapshot,
        )


@pytest.mark.slow_sim
def test_15_60_120_rig_dispatches_q6_gripper_on_same_physics_owner() -> None:
    pytest.importorskip("mujoco")
    scene, arm_profile, gripper_profile = _scene()
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
            timing=_timing(),
            actuation_profile=arm_profile,
            gripper_actuation_profile=gripper_profile,
            realtime_worker=False,
            follow_qpos_alpha=1.0,
        ),
    )
    rig.start()
    command = TeleopCommand(timestamp=0.0)
    for side in ("left", "right"):
        command.add_command(
            EffectorCommand(
                effector_name=f"{side}_arm",
                side=side,
                command_type="joint_vector",
                value=scene.get_arm_qpos(side),
            )
        )
    for side, close in (("left", 1.0), ("right", 0.5)):
        value = np.zeros(6, dtype=float)
        value[0] = close
        command.add_command(
            EffectorCommand(
                effector_name=f"{side}_hand",
                side=side,
                command_type="joint_vector",
                value=value,
                metadata={"gripper_slot": 0},
            )
        )

    receipts = rig.send_commands(command)
    assert receipts["left_hand"].accepted
    assert receipts["right_hand"].accepted
    for _ in range(300):
        rig.step_physics()

    snapshot = rig.snapshot()
    gripper = snapshot["gripper_execution"]
    assert snapshot["timing"]["sample_hz"] == 15.0
    assert snapshot["timing"]["reference_hz"] == 60.0
    assert snapshot["timing"]["command_hz"] == 120.0
    assert gripper["command_hz"] == 20.0
    assert gripper["accepted_close_by_gripper"] == {
        "left_gripper": 1.0,
        "right_gripper": 0.5,
    }
    assert gripper["actuator_target_close_by_gripper"]["left_gripper"] > 0.0
    assert gripper["actuator_target_close_by_gripper"]["right_gripper"] > 0.0
    assert scene.state_is_finite()
    rig.close()


@pytest.mark.slow_sim
def test_scheduled_q6_gripper_waits_for_guarded_arm_reference_activation() -> None:
    pytest.importorskip("mujoco")
    scene, arm_profile, gripper_profile = _scene()
    clock_now = 10.0
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
            timing=_timing(),
            actuation_profile=arm_profile,
            gripper_actuation_profile=gripper_profile,
            realtime_worker=False,
            follow_qpos_alpha=1.0,
        ),
        monotonic_clock=lambda: clock_now,
    )
    activations: list[tuple[str, int, int, float]] = []
    rig.bind_scheduled_generation_guards(
        activation_guard=lambda segment, epoch, generation, activated_at: (
            activations.append((segment, epoch, generation, activated_at))
            or "generation-activation-0"
        ),
        active_work_guard=lambda _segment, _epoch, _generation, _now: None,
    )
    rig.start()
    command = TeleopCommand(timestamp=0.0)
    for side in ("left", "right"):
        command.add_command(
            EffectorCommand(
                effector_name=f"{side}_arm",
                side=side,
                command_type="joint_vector",
                value=scene.get_arm_qpos(side),
            )
        )
        value = np.zeros(6, dtype=float)
        value[0] = 1.0
        command.add_command(
            EffectorCommand(
                effector_name=f"{side}_hand",
                side=side,
                command_type="joint_vector",
                value=value,
                metadata={"gripper_slot": 0},
            )
        )

    receipts = rig.send_scheduled_commands(
        command,
        activate_at=10.010,
        reference_segment_id="segment-0",
        execution_epoch=2,
        group_segment_generation=3,
    )
    assert set(receipts) == {
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
    }
    assert rig.snapshot()["gripper_execution"]["pending_delayed_targets"] == 0

    for _ in range(10):
        rig.step_physics()
    assert not activations
    assert rig.snapshot()["gripper_execution"]["pending_delayed_targets"] == 0

    rig.step_physics()
    assert activations == [("segment-0", 2, 3, pytest.approx(10.010))]
    assert rig.snapshot()["gripper_execution"]["pending_delayed_targets"] == 2
    rig.close()


@pytest.mark.parametrize("hold_before_activation", [False, True])
@pytest.mark.slow_sim
def test_opt_in_compound_runtime_activates_arm_and_hand_groups_together(
    hold_before_activation: bool,
    tmp_path,
) -> None:
    pytest.importorskip("mujoco")
    scene, arm_profile, gripper_profile = _scene()
    clock_now = 10.0
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
            timing=_timing(),
            actuation_profile=arm_profile,
            gripper_actuation_profile=gripper_profile,
            realtime_worker=False,
            follow_qpos_alpha=1.0,
        ),
        monotonic_clock=lambda: clock_now,
    )

    class _Planner:
        def reset(self, _qpos_by_side):
            return {"initialized": True}

        def plan_pose_reference(self, pose_by_side, **_kwargs):
            identity = np.eye(4, dtype=float)
            return SimpleNamespace(
                valid=True,
                hold=False,
                status="tracking",
                q_ref_by_side={side: scene.get_arm_qpos(side) for side in pose_by_side},
                target_endpoint_pose_world_by_side={
                    side: identity.copy() for side in pose_by_side
                },
                realized_endpoint_pose_world_by_side={
                    side: identity.copy() for side in pose_by_side
                },
                modification_reasons=(),
                metadata={"planner": "compound-dmtac-fixture"},
            )

    hand_specs = {
        side: HandActionSpec.dmtac_gripper(side) for side in ("left", "right")
    }
    anchor = CanonicalActionSafetyAnchor(
        anchor_id="compound-measured-anchor-0",
        anchor_kind="measured_state",
        action_space="bimanual_ee_pose_v1",
        arm_pose_by_side={
            side: np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
            for side in ("left", "right")
        },
        hand_qpos_by_side={
            side: np.zeros(6, dtype=np.float32) for side in ("left", "right")
        },
        hand_action_specs_by_side=hand_specs,
        frame_by_side={
            "left": "left_arm_base",
            "right": "right_arm_base",
        },
        observed_at=clock_now,
    )
    safety_profile = CanonicalKeyframeSafetyProfile(
        profile_id="compound-dmtac-safety-fixture-v1",
        limits=PositionKeyframeSafetyConfig(workspace_enabled=False),
        hand_limits_by_side={
            side: HandQposSafetyLimits.uniform(spec, lower=0.0, upper=1.0)
            for side, spec in hand_specs.items()
        },
    )
    runtime = MujocoDeployRuntime(
        rig,
        MujocoDeployConfig(
            runtime_session_id="compound-dmtac-session",
            control_domain_id="compound-dmtac-domain",
            embodiment_instance_id="compound-dmtac-embodiment",
            reference_activation_profile_id=SCHEDULED_REFERENCE_SEGMENT_SIM_V1,
            pose_frame_contract_hash="compound-arm-base-pose-frame-v1",
            compound_hand_action_specs_by_side=hand_specs,
        ),
        pose_resolver=MujocoCoupledPoseToQposResolver(
            RM65CanonicalIK(scene, _Planner())
        ),
        shared_safety=lambda action, input_id: apply_canonical_keyframe_safety(
            action,
            input_action_id=input_id,
            anchor=anchor,
            profile=safety_profile,
        ),
        clock=lambda: clock_now,
    )
    runtime.start()
    target_spec_hashes = dict(
        runtime.target_runtime_action_contract.action_spec_hash_by_component
    )
    source_spec = PassiveTeleopComponentSourceSpec(
        components=(
            *(
                PassiveTeleopComponentSpec.arm_pose(
                    f"{side}_arm",
                    f"{side}_arm",
                    action_spec_hash=target_spec_hashes[f"{side}_arm"],
                    arm_action_spec=ArmActionSpec.tool_tcp(
                        side,
                        reference_frame=f"{side}_arm_base",
                    ),
                )
                for side in ("left", "right")
            ),
            *(
                PassiveTeleopComponentSpec.hand_qpos(
                    f"{side}_hand",
                    f"{side}_hand",
                    action_spec_hash=target_spec_hashes[f"{side}_hand"],
                    hand_action_spec=hand_specs[side],
                )
                for side in ("left", "right")
            ),
        ),
        units_convention_id=(
            runtime.target_runtime_action_contract.units_convention_id
        ),
    )
    human_adapter = build_mujoco_component_human_proposal_adapter(
        runtime,
        source_spec,
    )
    arms_command = TeleopCommand(timestamp=clock_now)
    for side in ("left", "right"):
        arm_spec = source_spec.arm_action_specs_by_side[side]
        arms_command.add_command(
            EffectorCommand(
                effector_name=f"{side}_arm",
                side=side,
                command_type="cartesian_pose",
                value=np.eye(4, dtype=np.float32),
                frame_name=f"{side}_arm_base",
                metadata={"arm_action_spec": arm_spec.to_metadata()},
            )
        )
    commands_by_role: list[tuple[str, TeleopCommand]] = [
        ("vive-arms-fixture", arms_command)
    ]
    for side in ("left", "right"):
        hand_command = TeleopCommand(timestamp=clock_now)
        hand_value = np.zeros(6, dtype=np.float32)
        hand_value[0] = 0.25 if side == "left" else 0.75
        hand_command.add_command(
            EffectorCommand(
                effector_name=f"{side}_hand",
                side=side,
                command_type="joint_vector",
                value=hand_value,
                metadata={"hand_action_spec": hand_specs[side].to_metadata()},
            )
        )
        commands_by_role.append((f"{side}-q6-fixture", hand_command))
    emissions: list[HumanCommandEmission] = []
    payload_handles_by_stream: dict[str, np.ndarray] = {}
    for sequence, (role, command) in enumerate(commands_by_role):
        sample_refs: dict[str, ObservationSampleRef] = {}
        for command_snapshot in command.commands.values():
            stream_suffix = (
                "pose" if command_snapshot.command_type == "cartesian_pose" else "qpos"
            )
            stream_id = f"human.{command_snapshot.effector_name}_{stream_suffix}"
            payload = np.asarray(command_snapshot.value).copy()
            sample_refs[stream_id] = ObservationSampleRef(
                stream_id=stream_id,
                sample_id=f"{role}:sample:{command_snapshot.effector_name}",
                sequence=0,
                measured_at=clock_now - 0.01,
                measured_clock_domain_id=(
                    "vive.device_time" if stream_suffix == "pose" else "q6.device_time"
                ),
                available_at=clock_now - 0.001,
                available_clock_domain_id="teleop_session.monotonic",
                payload_object_id=(f"{role}:payload:{command_snapshot.effector_name}"),
            )
            payload_handles_by_stream[stream_id] = payload
        emission = HumanCommandEmission.capture(
            emission_sequence=sequence,
            emitted_at=clock_now,
            contributions=((role, command),),
            merged_command=command,
            observation_sample_refs=sample_refs,
        )
        emissions.append(emission)

    cut_builder = ReceiverObservationCutBuilder(
        producer_id=human_adapter.config.producer_id,
        producer_epoch=human_adapter.config.producer_epoch,
        observation_contract_hash=source_spec.semantic_content_hash,
        io_layout_hash=(runtime.target_runtime_action_contract.semantic_content_hash),
    )
    observation = admit_human_observation_cut(
        core=runtime.core,
        builder=cut_builder,
        emissions=tuple(emissions),
        payload_handles_by_stream=payload_handles_by_stream,
        built_at=clock_now,
    )
    proposal_ids: list[str] = []
    for emission in emissions:
        _adaptation, receipt = admit_human_command_emission(
            core=runtime.core,
            adapter=human_adapter,
            emission=emission,
            effective_from=clock_now,
            effective_until=clock_now + 0.1,
            target_effective_at=clock_now + 0.05,
            occurred_at=clock_now,
            clock_domain_id=runtime.config.clock_domain_id,
            observation_cut=observation.record,
        )
        assert receipt is not None and receipt.accepted
        assert receipt.proposal is not None
        assert receipt.proposal.observation_cut_id == (
            observation.record.observation_cut_id
        )
        proposal_ids.append(receipt.proposal.proposal_id)
    observation_facts = [
        fact for fact in runtime.trace_facts if fact.event_type == "observation_cut"
    ]
    assert [fact.fact_id for fact in observation_facts] == [
        observation.record.observation_cut_id
    ]

    input_contracts = {
        stream_id: NativeStreamContract(
            stream_id=stream_id,
            modality=(
                "human_cartesian_pose"
                if stream_id.endswith("_pose")
                else "human_joint_input"
            ),
            producer_id=source_spec.source_id,
            value_schema=(
                "float32[4,4]" if stream_id.endswith("_pose") else "float32[6]"
            ),
            shape=tuple(np.asarray(payload).shape),
            dtype=np.asarray(payload).dtype.str,
            units=(
                "homogeneous_transform"
                if stream_id.endswith("_pose")
                else "normalized_joint_position"
            ),
            frame=(
                stream_id.removeprefix("human.").removesuffix("_pose") + "_base"
                if stream_id.endswith("_pose")
                else None
            ),
            measured_clock_domain_id=sample.measured_clock_domain_id,
            available_clock_domain_id=(runtime.config.clock_domain_id),
            access_role="actor_input",
        )
        for stream_id, payload in payload_handles_by_stream.items()
        for sample in (observation.record.selected_sample_by_stream[stream_id],)
    }
    output_contracts = {
        "sim.arm_qpos": NativeStreamContract(
            stream_id="sim.arm_qpos",
            modality="robot_state",
            producer_id="mujoco-rig",
            value_schema="float64[2,6]",
            shape=(2, 6),
            dtype="<f8",
            units="rad",
            frame=None,
            measured_clock_domain_id=runtime.config.clock_domain_id,
            available_clock_domain_id=runtime.config.clock_domain_id,
            access_role="actor_input",
        ),
        "sim.hand_qpos": NativeStreamContract(
            stream_id="sim.hand_qpos",
            modality="robot_state",
            producer_id="mujoco-rig",
            value_schema="float64[2,2]",
            shape=(2, 2),
            dtype="<f8",
            units="m",
            frame=None,
            measured_clock_domain_id=runtime.config.clock_domain_id,
            available_clock_domain_id=runtime.config.clock_domain_id,
            access_role="actor_input",
        ),
        "command.hand_close": NativeStreamContract(
            stream_id="command.hand_close",
            modality="actuator_command",
            producer_id="mujoco-dmtac-executor",
            value_schema="float64[2]",
            shape=(2,),
            dtype="<f8",
            units="normalized_close",
            frame=None,
            measured_clock_domain_id=runtime.config.clock_domain_id,
            available_clock_domain_id=runtime.config.clock_domain_id,
            access_role="actor_target",
        ),
    }
    native_writer = NativeStreamEpisodeWriter(
        tmp_path,
        episode_uuid=f"compound-dmtac-{hold_before_activation}",
        legacy_episode_index=int(hold_before_activation),
        contracts={**input_contracts, **output_contracts},
    )
    for stream_id, sample in observation.record.selected_sample_by_stream.items():
        native_writer.append_observation_sample(
            sample,
            observation.payload_handles_by_stream[stream_id],
            written_at=clock_now,
            written_clock_domain_id=runtime.config.clock_domain_id,
        )

    def _seal_native_capture(
        command_close: np.ndarray | None,
    ):
        arm_qpos = np.stack(
            [scene.get_arm_qpos(side) for side in ("left", "right")]
        ).astype(np.float64, copy=False)
        hand_qpos = np.stack(
            [scene.get_gripper_qpos(f"{side}_gripper") for side in ("left", "right")]
        ).astype(np.float64, copy=False)
        native_writer.append(
            stream_id="sim.arm_qpos",
            sample_id="sim-arm-qpos-final",
            sequence=0,
            measured_at=clock_now,
            measured_clock_domain_id=runtime.config.clock_domain_id,
            available_at=clock_now,
            available_clock_domain_id=runtime.config.clock_domain_id,
            source_payload_object_id="mujoco-arm-state-final",
            payload=arm_qpos,
            written_at=clock_now,
            written_clock_domain_id=runtime.config.clock_domain_id,
        )
        native_writer.append(
            stream_id="sim.hand_qpos",
            sample_id="sim-hand-qpos-final",
            sequence=0,
            measured_at=clock_now,
            measured_clock_domain_id=runtime.config.clock_domain_id,
            available_at=clock_now,
            available_clock_domain_id=runtime.config.clock_domain_id,
            source_payload_object_id="mujoco-hand-state-final",
            payload=hand_qpos,
            written_at=clock_now,
            written_clock_domain_id=runtime.config.clock_domain_id,
        )
        if command_close is not None:
            native_writer.append(
                stream_id="command.hand_close",
                sample_id="dmtac-hand-close-command-0",
                sequence=0,
                measured_at=clock_now,
                measured_clock_domain_id=runtime.config.clock_domain_id,
                available_at=clock_now,
                available_clock_domain_id=runtime.config.clock_domain_id,
                source_payload_object_id="dmtac-command-final",
                payload=np.asarray(command_close, dtype=np.float64),
                written_at=clock_now,
                written_clock_domain_id=runtime.config.clock_domain_id,
            )
        native_manifest = native_writer.seal(
            sealed_at=clock_now,
            sealed_clock_domain_id=runtime.config.clock_domain_id,
        )
        captured = load_native_stream_episode(
            tmp_path,
            native_manifest.episode_uuid,
        )
        assert all(
            len(captured.records_by_stream[stream_id]) == 1
            for stream_id in input_contracts
        )
        assert len(captured.records_by_stream["sim.arm_qpos"]) == 1
        assert len(captured.records_by_stream["sim.hand_qpos"]) == 1
        assert len(captured.records_by_stream["command.hand_close"]) == (
            0 if command_close is None else 1
        )
        return captured

    step = runtime.step(now=clock_now)

    assert step.status == "dispatched"
    assert step.safety_result is not None
    assert step.safety_result.safety_contract_hash == (
        safety_profile.semantic_content_hash
    )
    manifest = runtime.last_compound_dispatch_manifest
    assert manifest is not None
    assert set(manifest.group_binding_by_id) == {
        "arms",
        "left_hand",
        "right_hand",
    }
    assert {
        binding.selected_action_ref for binding in manifest.group_binding_by_id.values()
    } == set(proposal_ids)
    assert set(runtime.core.pending_execution_segment_by_group) == {
        "arms",
        "left_hand",
        "right_hand",
    }
    assert runtime.rig.snapshot()["gripper_execution"]["pending_delayed_targets"] == 0
    if hold_before_activation:
        outcome = runtime.transition_execution_gate(
            "hold",
            reason="compound_fixture_hold_before_start",
        )
        assert outcome.cancellation is not None
        assert not runtime.core.pending_execution_segment_by_group
        for _ in range(70):
            runtime.rig.step_physics()
        assert not runtime.core.active_execution_segment_by_group
        gripper = runtime.rig.snapshot()["gripper_execution"]
        assert gripper["pending_delayed_targets"] == 0
        assert gripper["accepted_close_by_gripper"] == {
            "left_gripper": pytest.approx(0.0),
            "right_gripper": pytest.approx(0.0),
        }
        captured = _seal_native_capture(None)
        assert not captured.payload_by_stream_and_sample["command.hand_close"]
        runtime.close()
        return

    for _ in range(52):
        runtime.rig.step_physics()

    assert set(runtime.core.active_execution_segment_by_group) == {
        "arms",
        "left_hand",
        "right_hand",
    }
    assert not runtime.core.pending_execution_segment_by_group
    gripper = runtime.rig.snapshot()["gripper_execution"]
    assert gripper["pending_delayed_targets"] == 2
    for _ in range(80):
        runtime.rig.step_physics()
    gripper = runtime.rig.snapshot()["gripper_execution"]
    assert gripper["accepted_close_by_gripper"] == {
        "left_gripper": pytest.approx(0.25),
        "right_gripper": pytest.approx(0.75),
    }
    captured = _seal_native_capture(np.asarray([0.25, 0.75], dtype=np.float64))
    np.testing.assert_allclose(
        captured.payload_by_stream_and_sample["command.hand_close"][
            "dmtac-hand-close-command-0"
        ],
        [0.25, 0.75],
    )
    runtime.close()
