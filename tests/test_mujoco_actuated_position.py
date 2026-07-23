from __future__ import annotations

import numpy as np
import pytest

from robot_motion.runtime import EffectorCommand, TeleopCommand
from sim.contracts import SimulationContract, TimingSpec
from sim.mujoco.actuation import (
    MujocoActuatedPositionExecutor,
    rm65_bimanual_udp_feedback_profile,
    rm65_bimanual_udp_feedback_profile_v1,
    rm65_bimanual_udp_feedback_profile_v2,
    rm65_bimanual_udp_feedback_profile_v3,
)
from sim.mujoco.adapters import (
    MujocoPositionExecutionConfig,
    MujocoPositionExecutionRig,
)
from sim.mujoco.control import DirectQposCommand
from sim.mujoco.scene import MujocoScene


CONTRACT_PATH = "sim/configs/rm65_bimanual_physics_v1.json"


def _timing() -> TimingSpec:
    return TimingSpec(
        sample_hz=15.0,
        policy_hz=15.0,
        reference_hz=60.0,
        ik_hz=60.0,
        command_hz=120.0,
        physics_hz=1000.0,
        parameter_source="unit_test_15_60_120",
    )


def test_rm65_execution_config_defaults_to_fitted_arm_and_grippers() -> None:
    config = MujocoPositionExecutionConfig(timing=_timing())

    assert config.actuation_profile is not None
    assert config.actuation_profile.name == "rm65_bimanual_udp_feedback_20260721.v4"
    assert config.gripper_actuation_profile is not None
    assert (
        config.gripper_actuation_profile.name
        == "dmtac_bimanual_position_20260722.v1"
    )


def _scene_and_profile():
    contract = SimulationContract.load(CONTRACT_PATH)
    profile = rm65_bimanual_udp_feedback_profile()
    scene = MujocoScene.from_contract(contract, position_actuation_profile=profile)
    return scene, profile


def test_rm65_profile_records_joint_fit_without_claiming_force_identification() -> None:
    profile = rm65_bimanual_udp_feedback_profile()

    assert profile.name == "rm65_bimanual_udp_feedback_20260721.v4"
    assert profile.transport_delay_sec_by_side == {"left": 0.018, "right": 0.017}
    assert profile.command_delay_sec("left", 5) == pytest.approx(
        0.009527118125949966
    )
    threshold = np.deg2rad(50.0)
    assert profile.target_filter_tau_sec(
        "left", 4, abs_command_velocity_rad_s=threshold
    ) == 0.0
    assert profile.target_filter_tau_sec(
        "left", 4, abs_command_velocity_rad_s=np.deg2rad(60.0)
    ) > 0.0
    assert profile.target_filter_tau_sec(
        "left", 4, abs_command_velocity_rad_s=np.deg2rad(200.0)
    ) == pytest.approx(0.01845353602325293)
    assert profile.target_filter_tau_sec(
        "left", 3, abs_command_velocity_rad_s=np.deg2rad(200.0)
    ) == 0.0
    assert profile.target_filter_tau_sec(
        "left", 5, abs_command_velocity_rad_s=np.deg2rad(200.0)
    ) == pytest.approx(0.01278000865660236)
    assert "fourteen_valid_traces_distal_phase_refitted" in profile.calibration_status
    assert "force_contact_provisional" in profile.calibration_status
    assert rm65_bimanual_udp_feedback_profile_v1().name.endswith(".v1")
    assert rm65_bimanual_udp_feedback_profile_v2().name.endswith(".v2")
    assert rm65_bimanual_udp_feedback_profile_v3().name.endswith(".v3")
    assert rm65_bimanual_udp_feedback_profile_v2().command_delay_sec(
        "left", 5
    ) == pytest.approx(0.0110777)
    assert rm65_bimanual_udp_feedback_profile_v3().command_delay_sec(
        "left", 5
    ) == pytest.approx(0.0108740)


def _command(sequence: int, qpos_by_side) -> DirectQposCommand:
    return DirectQposCommand(
        sequence=sequence,
        qpos_by_side=qpos_by_side,
        source="unit_120hz_qcmd",
    )


def test_actuated_executor_is_only_the_120hz_sender_and_physical_plant() -> None:
    pytest.importorskip("mujoco")
    scene, profile = _scene_and_profile()
    sender = MujocoActuatedPositionExecutor(scene, _timing(), profile)
    initial = {side: scene.get_arm_qpos(side) for side in ("left", "right")}
    target = {side: values + 0.01 for side, values in initial.items()}
    sender.reset(_command(0, initial))

    # Reset owns the already-reached initial target. The next sender call is
    # due on the independent 120 Hz deadline, not at 20 Hz or 60 Hz.
    while not sender.command_due:
        tick = sender.step_physics()
        assert not tick.command_applied
    tick = sender.step_physics(_command(1, target))
    assert tick.command_applied
    assert all(
        np.allclose(sender.accepted_arm_qpos(side), target[side])
        for side in ("left", "right")
    )

    # Keep satisfying subsequent 120 Hz deadlines with the same qcmd. The
    # accepted command is immediate, while actuator ctrl changes after the
    # measured per-side transport delay.
    sequence = 2
    while sender.physics_time_sec < 0.045:
        command = None
        if sender.command_due:
            command = _command(sequence, target)
            sequence += 1
        sender.step_physics(command)
    left_ids = [
        scene.mujoco.mj_name2id(
            scene.model,
            scene.mujoco.mjtObj.mjOBJ_ACTUATOR,
            profile.actuator_name("left", index),
        )
        for index in range(6)
    ]
    assert np.allclose(scene.data.ctrl[left_ids], target["left"])
    assert scene.state_is_finite()
    assert sender.execution_metadata["command_hz"] == 120.0


def test_high_speed_distal_target_lag_is_causal_and_keeps_raw_targets() -> None:
    pytest.importorskip("mujoco")
    scene, profile = _scene_and_profile()
    sender = MujocoActuatedPositionExecutor(scene, _timing(), profile)
    initial = {side: scene.get_arm_qpos(side) for side in ("left", "right")}
    sender.reset(_command(0, initial))

    target = {side: values.copy() for side, values in initial.items()}
    sequence = 1
    saw_filtered_target = False
    while sender.physics_time_sec < 0.08:
        command = None
        if sender.command_due:
            # One radian/second is above the fitted 50 deg/s threshold.
            target["left"][4] += 1.0 / 120.0
            command = _command(sequence, target)
            sequence += 1
        sender.step_physics(command)
        snapshot = sender.snapshot()
        tau = snapshot["actuator_target_tau_sec_by_side"]["left"][4]
        raw = snapshot["raw_actuator_target_by_side"]["left"][4]
        filtered = snapshot["actuator_target_by_side"]["left"][4]
        if tau > 0.0 and not np.isclose(raw, filtered):
            saw_filtered_target = True

    assert saw_filtered_target
    assert np.isclose(sender.accepted_arm_qpos("left")[4], target["left"][4])
    assert sender.snapshot()["pending_delayed_targets"] < 60


def test_real_aligned_rig_keeps_15_60_120_and_swaps_only_sender() -> None:
    pytest.importorskip("mujoco")
    scene, profile = _scene_and_profile()
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
                timing=_timing(),
                actuation_profile=profile,
                gripper_actuation_profile=None,
                realtime_worker=False,
            follow_qpos_alpha=1.0,
        ),
    )
    rig.start()
    reference = TeleopCommand(timestamp=0.0)
    for side, sign in (("left", 1.0), ("right", -1.0)):
        reference.add_command(
            EffectorCommand(
                effector_name=f"{side}_arm",
                side=side,
                command_type="joint_vector",
                value=scene.get_arm_qpos(side) + sign * 0.02,
            )
        )
    rig.send_commands(reference)
    for _ in range(45):
        rig.step_physics()

    snapshot = rig.snapshot()
    assert snapshot["timing"]["sample_hz"] == 15.0
    assert snapshot["timing"]["reference_hz"] == 60.0
    assert snapshot["timing"]["command_hz"] == 120.0
    assert snapshot["execution_mode"] == "actuated_position_servo"
    assert snapshot["robot_dynamics"] == "mujoco_position_actuator"
    assert snapshot["execution"]["accepted_command_sequence"] is not None
    assert rig._direct is None
