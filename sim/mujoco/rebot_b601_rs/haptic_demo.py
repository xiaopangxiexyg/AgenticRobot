"""Headless-first B601 virtual-fixture and physical-contact acceptance demo."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from robot_motion.rebot_b601_rs import (
    B601_RS_ARM_JOINT_NAMES,
    B601DampedLeastSquaresIK,
    B601MitReference,
    B601PinocchioModel,
    B601PoseVirtualSpring,
    B601TranslationalVirtualSpring,
    B601VirtualWall,
    map_world_aligned_ee_wrench_to_joint_torque,
)
from sim.contracts import TimingSpec
from sim.mujoco.viewer import MujocoPassiveViewerSession

from .contact import build_b601_contact_box_fixture, read_b601_contact_observation
from .mit_driver import B601MujocoEffortDriver, default_b601_mit_profile
from .real2sim import B601Real2SimControllerProfile, B601SimStartupProfile
from .scene import B601_RS_PHYSICS_HZ, build_b601_mujoco_scene
from .shared_control_demo import (
    SHARED_CONTROL_FIXTURE,
    run_b601_shared_control_demo,
)

DEFAULT_HAPTIC_ANCHOR_QPOS = (0.15, 1.0, 1.1, 0.1, -0.15, 0.2)
DEFAULT_HAPTIC_INITIAL_QPOS = (0.23, 0.94, 1.15, 0.1, -0.15, 0.2)
DEFAULT_HAPTIC_DURATION_SEC = 1.5
DEFAULT_SPRING_STIFFNESS_N_PER_M = (80.0, 80.0, 80.0)
DEFAULT_SPRING_DAMPING_N_S_PER_M = (8.0, 8.0, 8.0)
DEFAULT_MAX_FORCE_N = 8.0
DEFAULT_MAX_OVERLAY_TORQUE_NM = 6.0
DEFAULT_WALL_PENETRATION_M = 0.025
CONTACT_FIXTURE = "contact-box"
REFLECTION_FIXTURE = "force-reflection"
POSE_FIXTURES = (
    "pose-tx",
    "pose-ty",
    "pose-tz",
    "pose-rx",
    "pose-ry",
    "pose-rz",
    "pose-6d",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a deterministic B601 virtual fixture or physical contact box through the "
            "ideal-effort MuJoCo backend. Headless is the default."
        )
    )
    parser.add_argument(
        "--rebot-source-root",
        type=Path,
        required=True,
        help="Explicit root of the pinned reBotArm_control_py source snapshot.",
    )
    parser.add_argument(
        "--fixture",
        choices=(
            "spring",
            "wall",
            CONTACT_FIXTURE,
            REFLECTION_FIXTURE,
            SHARED_CONTROL_FIXTURE,
            *POSE_FIXTURES,
        ),
        default="spring",
    )
    parser.add_argument(
        "--trace-jsonl",
        type=Path,
        help="Contact-box/reflection: create an exclusive same-state command-tick trace file.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Create a unique run directory below this root, save the full JSON as "
            "result.json, and print only a short terminal summary. Omission preserves "
            "the existing full-JSON stdout behavior."
        ),
    )
    parser.add_argument(
        "--feedback-delay-sec",
        type=float,
        default=0.0,
        help="Force-reflection only: deterministic local simulation delay.",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=DEFAULT_HAPTIC_DURATION_SEC,
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open the passive MuJoCo viewer and pace physics in real time.",
    )
    parser.add_argument(
        "--real2sim-controller",
        action="store_true",
        help="Use operator compensation and local Kd on the ideal plant.",
    )
    parser.add_argument(
        "--gravity-control-baseline", choices=("operator", "raw-model"),
        default="operator",
        help="Real2sim controller-side gravity baseline only; never changes the plant.",
    )
    parser.add_argument(
        "--host-hz",
        type=float,
        default=20.0,
        help="Host torque rate for --real2sim-controller only.",
    )
    parser.add_argument(
        "--torque-rise-time-sec", type=float,
        help=(
            "Simulation rate normalization: URDF effort ceiling divided by this time. "
            "Not the startup handoff duration; omission preserves the driver default."
        ),
    )
    parser.add_argument(
        "--fixture-stiffness-scale", type=float, default=1.0,
        help="Offline fixture stiffness multiplier; does not relax acceptance or limits.",
    )
    parser.add_argument(
        "--fixture-damping-scale", type=float, default=1.0,
        help="Offline fixture damping multiplier; local motor Kd stays unchanged.",
    )
    parser.add_argument(
        "--startup-lock-kp",
        nargs=6,
        type=float,
        help="Explicit six-joint simulation lock Kp; requires Kd and real2sim.",
    )
    parser.add_argument(
        "--startup-lock-kd",
        nargs=6,
        type=float,
        help="Explicit six-joint simulation lock Kd.",
    )
    return parser


def _arm_qvel(scene) -> np.ndarray:
    return np.asarray(
        [
            scene.data.qvel[scene.joint_dof_addr(name)]
            for name in B601_RS_ARM_JOINT_NAMES
        ],
        dtype=float,
    )


def _fixture_metrics(
    *,
    fixture_name: str,
    model: B601PinocchioModel,
    fixture,
    qpos: np.ndarray,
    qvel: np.ndarray,
) -> tuple[object, float, float]:
    pose = model.forward_kinematics(qpos)
    jacobian = model.arm_jacobian(qpos)
    spatial_velocity = jacobian @ qvel
    if fixture_name in POSE_FIXTURES:
        result = fixture.evaluate(pose, spatial_velocity)
        translation_metric = float(np.linalg.norm(result.displacement_m))
        rotation_metric = float(np.linalg.norm(result.rotation_error_rad))
    else:
        result = fixture.evaluate(pose[:3, 3], spatial_velocity[:3])
        translation_metric = (
            float(np.linalg.norm(result.displacement_m))
            if fixture_name in {"spring", CONTACT_FIXTURE}
            else float(result.penetration_m)
        )
        rotation_metric = 0.0
    return result, translation_metric, rotation_metric


def _pose_disturbance_target(
    anchor_pose: np.ndarray,
    fixture_name: str,
) -> tuple[np.ndarray, tuple[float, ...], tuple[float, ...]]:
    """Create one bounded world-translation/tool-local-rotation disturbance."""

    if fixture_name not in POSE_FIXTURES:
        raise ValueError("fixture_name is not a pose disturbance")
    translation = np.zeros(3, dtype=float)
    rotation_local = np.zeros(3, dtype=float)
    if fixture_name == "pose-6d":
        translation[:] = (0.012, -0.009, 0.007)
        rotation_local[:] = (0.012, -0.010, 0.008)
    elif fixture_name.startswith("pose-t"):
        translation["xyz".index(fixture_name[-1])] = 0.015
    else:
        rotation_local["xyz".index(fixture_name[-1])] = 0.015
    target = np.asarray(anchor_pose, dtype=float).copy()
    target[:3, 3] += translation
    target[:3, :3] = target[:3, :3] @ Rotation.from_rotvec(rotation_local).as_matrix()
    return (
        target,
        tuple(float(value) for value in translation),
        tuple(float(value) for value in rotation_local),
    )


def run_haptic_fixture_demo(
    *,
    source_root: str | Path,
    fixture_name: str,
    duration_sec: float = DEFAULT_HAPTIC_DURATION_SEC,
    viewer: bool = False,
    real2sim_controller: bool = False,
    host_hz: float = 20.0,
    startup_profile: B601SimStartupProfile | None = None,
    fixture_stiffness_scale: float = 1.0,
    fixture_damping_scale: float = 1.0,
    torque_rise_time_sec: float | None = None,
    controller_profile: B601Real2SimControllerProfile | None = None,
    trace_jsonl: str | Path | None = None,
) -> tuple[int, dict[str, object]]:
    """Run one closed-loop fixture and return an exit code plus JSON payload."""

    if fixture_name not in {"spring", "wall", CONTACT_FIXTURE, *POSE_FIXTURES}:
        raise ValueError(
            "fixture_name must name a supported spring, wall, or pose fixture"
        )
    if not isinstance(viewer, bool):
        raise TypeError("viewer must be a bool")
    if trace_jsonl is not None and fixture_name != CONTACT_FIXTURE:
        raise ValueError("trace_jsonl requires contact-box")
    if startup_profile is not None and not real2sim_controller:
        raise ValueError("startup_profile requires real2sim_controller")
    if controller_profile is not None:
        if not isinstance(controller_profile, B601Real2SimControllerProfile):
            raise TypeError("controller_profile must be a B601Real2SimControllerProfile")
        if not real2sim_controller:
            raise ValueError("controller_profile requires real2sim_controller")
    duration = float(duration_sec)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration_sec must be finite and positive")
    if torque_rise_time_sec is not None and (
        isinstance(torque_rise_time_sec, bool)
        or not math.isfinite(torque_rise_time_sec)
        or torque_rise_time_sec <= 0.0
    ):
        raise ValueError("torque_rise_time_sec must be finite and positive")
    for name, value in (
        ("fixture_stiffness_scale", fixture_stiffness_scale),
        ("fixture_damping_scale", fixture_damping_scale),
    ):
        if isinstance(value, bool) or not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    stiffness = tuple(
        v * fixture_stiffness_scale for v in DEFAULT_SPRING_STIFFNESS_N_PER_M
    )
    damping = tuple(
        v * fixture_damping_scale for v in DEFAULT_SPRING_DAMPING_N_S_PER_M
    )
    if (
        fixture_name == "wall"
        and fixture_stiffness_scale == fixture_damping_scale == 0.0
    ):
        raise ValueError("wall stiffness or damping must be positive")

    model = B601PinocchioModel.from_rebot_source_root(source_root)
    initial_qpos = np.asarray(DEFAULT_HAPTIC_INITIAL_QPOS, dtype=float)
    anchor_qpos = np.asarray(DEFAULT_HAPTIC_ANCHOR_QPOS, dtype=float)
    anchor_pose = model.forward_kinematics(anchor_qpos)
    anchor_position = anchor_pose[:3, 3]
    disturbance_contract: dict[str, object] | None = None
    if fixture_name in POSE_FIXTURES:
        target_pose, translation_world, rotation_tool_local = _pose_disturbance_target(
            anchor_pose,
            fixture_name,
        )
        ik_result = B601DampedLeastSquaresIK(model).solve(
            target_pose,
            seed_qpos=anchor_qpos,
        )
        if not ik_result.success:
            raise RuntimeError(
                f"pose disturbance IK failed: {ik_result.status}; "
                f"position_error_m={ik_result.position_error_m:.9g}; "
                f"orientation_error_rad={ik_result.orientation_error_rad:.9g}"
            )
        initial_qpos = np.asarray(ik_result.qpos, dtype=float)
        disturbance_contract = {
            "application": "kinematic_pre_run_disturbance_after_current_pose_anchor",
            "translation_world_m": list(translation_world),
            "rotation_tool_local_rad": list(rotation_tool_local),
            "ik_status": ik_result.status,
            "ik_position_error_m": ik_result.position_error_m,
            "ik_orientation_error_rad": ik_result.orientation_error_rad,
        }
    initial_position = model.forward_kinematics(initial_qpos)[:3, 3]
    if fixture_name in {"spring", CONTACT_FIXTURE}:
        if fixture_name == CONTACT_FIXTURE:
            anchor_position = initial_position.copy()
        fixture = B601TranslationalVirtualSpring(
            anchor_position_m=tuple(float(value) for value in anchor_position),
            stiffness_n_per_m=stiffness,
            damping_n_s_per_m=damping,
            max_abs_force_n=(DEFAULT_MAX_FORCE_N,) * 3,
        )
        fixture_contract: dict[str, object] = {
            "anchor_position_m": [float(value) for value in anchor_position],
            "stiffness_n_per_m": list(stiffness),
            "damping_n_s_per_m": list(damping),
            "max_abs_force_n": [DEFAULT_MAX_FORCE_N] * 3,
        }
    elif fixture_name == "wall":
        wall_normal = np.asarray((1.0, 0.0, 0.0), dtype=float)
        wall_offset = float(wall_normal @ initial_position) - DEFAULT_WALL_PENETRATION_M
        fixture = B601VirtualWall(
            normal=tuple(float(value) for value in wall_normal),
            offset_m=wall_offset,
            stiffness_n_per_m=180.0 * fixture_stiffness_scale,
            damping_n_s_per_m=12.0 * fixture_damping_scale,
            max_force_n=DEFAULT_MAX_FORCE_N,
        )
        fixture_contract = {
            "normal": list(fixture.normal),
            "offset_m": fixture.offset_m,
            "forbidden_half_space": "normal_dot_position_greater_than_offset",
            "stiffness_n_per_m": fixture.stiffness_n_per_m,
            "damping_n_s_per_m": fixture.damping_n_s_per_m,
            "max_force_n": fixture.max_force_n,
        }
    else:
        fixture = B601PoseVirtualSpring(
            anchor_pose=anchor_pose,
            translational_stiffness_n_per_m=stiffness,
            translational_damping_n_s_per_m=damping,
            rotational_stiffness_nm_per_rad=(8.0 * fixture_stiffness_scale,) * 3,
            rotational_damping_nm_s_per_rad=(0.8 * fixture_damping_scale,) * 3,
            max_abs_force_n=(DEFAULT_MAX_FORCE_N,) * 3,
            max_abs_torque_nm=(0.5,) * 3,
        )
        fixture_contract = {
            "anchor_source": "current_fk_before_disturbance",
            "anchor_pose": np.asarray(anchor_pose, dtype=float).tolist(),
            "translational_stiffness_n_per_m": list(stiffness),
            "translational_damping_n_s_per_m": list(damping),
            "rotational_stiffness_nm_per_rad": list(
                fixture.rotational_stiffness_nm_per_rad
            ),
            "rotational_damping_nm_s_per_rad": list(
                fixture.rotational_damping_nm_s_per_rad
            ),
            "max_abs_force_n": [DEFAULT_MAX_FORCE_N] * 3,
            "max_abs_torque_nm": [0.5] * 3,
            "disturbance": disturbance_contract,
        }

    fixture_contract.update(
        stiffness_scale=fixture_stiffness_scale,
        damping_scale=fixture_damping_scale,
    )
    if fixture_name == CONTACT_FIXTURE:
        scene, box = build_b601_contact_box_fixture(source_root, initial_qpos)
        fixture_contract.update(
            physical_object=box.to_dict(),
            placement="mesh_highest_world_vertex_plus_half_box_height_minus_0.0005m",
            contact_feedback_control="native_constraint_only_not_added_to_overlay",
        )
    else:
        scene = build_b601_mujoco_scene(
            source_root,
            initial_arm_qpos=initial_qpos,
            gravity_mps2=(0.0, 0.0, -9.81),
            actuation_mode="effort",
        )
    base_profile = (
        default_b601_mit_profile(scene)
        if torque_rise_time_sec is None
        else default_b601_mit_profile(scene, torque_rise_time_sec=torque_rise_time_sec)
    )
    profile = replace(
        base_profile,
        damping_kd_nm_s_per_rad=(
            (0.0,) * 6 if real2sim_controller else base_profile.damping_kd_nm_s_per_rad
        ),
        max_abs_overlay_torque_nm=(DEFAULT_MAX_OVERLAY_TORQUE_NM,) * 6,
        max_abs_state_acceleration_rad_s2=(250.0,) * 6,
        metadata={
            **base_profile.metadata,
            "fixture_demo": fixture_name,
            "fixture_parameter_source": "pre_hardware_effect_acceptance_seed",
        },
    )
    driver = B601MujocoEffortDriver(
        scene,
        model,
        profile=profile,
        enable_gravity_compensation=True,
        timing=(
            TimingSpec(
                command_hz=host_hz,
                physics_hz=B601_RS_PHYSICS_HZ,
                parameter_source="explicit_host_rate_diagnostic",
            )
            if real2sim_controller
            else None
        ),
        real2sim_profile=(
            (controller_profile or B601Real2SimControllerProfile())
            if real2sim_controller else None
        ),
        startup_profile=startup_profile,
    )
    driver.reset(initial_qpos)

    _initial_result, initial_metric, initial_rotation_metric = _fixture_metrics(
        fixture_name=fixture_name,
        model=model,
        fixture=fixture,
        qpos=initial_qpos,
        qvel=np.zeros(6, dtype=float),
    )
    requested_steps = math.ceil(duration * B601_RS_PHYSICS_HZ)
    viewer_session = MujocoPassiveViewerSession(scene) if viewer else None

    next_sequence = 0
    command_count = 0
    advanced_ticks = 0
    power_channels: dict[str, dict[str, float | int]] = {}
    held_nominal_damping = np.zeros(6)
    sampled_damping_peak_power = 0.0
    fixture_limited_ticks = 0
    overlay_limited_ticks = 0
    torque_limited_ticks = 0
    slew_limited_ticks = 0
    peak_force_n = 0.0
    peak_cartesian_torque_nm = 0.0
    peak_requested_overlay_torque_nm = 0.0
    peak_applied_torque_nm = 0.0
    peak_abs_qvel_rad_s = 0.0
    peak_abs_qacc_rad_s2 = 0.0
    maximum_metric = initial_metric
    minimum_metric = initial_metric
    maximum_rotation_metric = initial_rotation_metric
    minimum_rotation_metric = initial_rotation_metric
    final_tick = None
    viewer_closed_early = False
    contact_diagnostics = {
        "sample_count": 0, "nonzero_wrench_samples": 0,
        "peak_force_norm_n": 0.0, "max_power_identity_error_w": 0.0,
        "first_sample": None, "last_sample": None,
        "sampling_phase": "current_state_before_reference_admission",
        "projection_role": "diagnostic_only_never_dispatched",
        "trace_jsonl": None if trace_jsonl is None else str(trace_jsonl),
        "force_calibrated": False, "passivity_claim": False,
    }
    phase_transitions = [
        {
            "phase": driver.execution_metadata["sim_control_phase"],
            "time_sec": 0.0,
            "reason": None,
        }
    ]
    with ExitStack() as cleanup:
        if viewer_session is not None:
            cleanup.callback(viewer_session.close)
            viewer_session.start()
        trace_stream = (
            None if trace_jsonl is None else cleanup.enter_context(Path(trace_jsonl).open("x"))
        )
        for step in range(requested_steps):
            reference = None
            contact_record = None
            if driver.command_due:
                qpos = scene.get_arm_qpos("b601_rs")
                qvel = _arm_qvel(scene)
                fixture_result, metric, rotation_metric = _fixture_metrics(
                    fixture_name=fixture_name,
                    model=model,
                    fixture=fixture,
                    qpos=qpos,
                    qvel=qvel,
                )
                mapping = map_world_aligned_ee_wrench_to_joint_torque(
                    model,
                    qpos,
                    fixture_result.wrench,
                    qvel_rad_s=qvel,
                )
                timestamp = driver.physics_time_sec
                if fixture_name == CONTACT_FIXTURE:
                    observation = read_b601_contact_observation(scene)
                    # Native time accumulates dt; the driver clock multiplies
                    # an integer tick.  Allow only floating-point roundoff.
                    if not math.isclose(
                        observation.physics_time_sec, timestamp, rel_tol=0.0, abs_tol=1e-12,
                    ):
                        raise RuntimeError("contact observation and driver time differ")
                    projection = map_world_aligned_ee_wrench_to_joint_torque(
                        model, qpos, observation.wrench, qvel_rad_s=qvel,
                    )
                    force_norm = float(np.linalg.norm(observation.wrench.force_n))
                    contact_diagnostics["sample_count"] += 1
                    contact_diagnostics["nonzero_wrench_samples"] += int(force_norm > 1e-6)
                    contact_diagnostics["peak_force_norm_n"] = max(
                        contact_diagnostics["peak_force_norm_n"], force_norm,
                    )
                    contact_diagnostics["max_power_identity_error_w"] = max(
                        contact_diagnostics["max_power_identity_error_w"],
                        abs(projection.cartesian_power_w - projection.joint_power_w),
                    )
                    contact_record = {
                        "sequence": next_sequence,
                        "physics_time_sec": observation.physics_time_sec,
                        "driver_time_sec": timestamp,
                        "qpos_rad": qpos.tolist(), "qvel_rad_s": qvel.tolist(),
                        "end_effector_pose_world": model.forward_kinematics(qpos).tolist(),
                        "matched_contact_count": observation.matched_contact_count,
                        "wrench_frame": observation.wrench.frame_id,
                        "force_n": list(observation.wrench.force_n),
                        "torque_nm": list(observation.wrench.torque_nm),
                        "application_point_world_m": observation.application_point_world_m,
                        "projected_contact_torque_diagnostic_nm": projection.joint_torque_nm,
                        "contact_cartesian_power_w": projection.cartesian_power_w,
                        "contact_joint_power_w": projection.joint_power_w,
                        "spring_overlay_requested_nm": mapping.joint_torque_nm,
                        "contact_solver_ctrl": scene.data.ctrl.copy().tolist(),
                    }
                if fixture_name in POSE_FIXTURES:
                    # Diagnostic only: unclipped -J.T D J dq, sampled and held
                    # at the host rate. It is not a decomposition of the final
                    # limited/slewed actuator force and never enters control.
                    jacobian = model.arm_jacobian(qpos)
                    damping_diagonal = np.asarray((
                        *fixture.translational_damping_n_s_per_m,
                        *fixture.rotational_damping_nm_s_per_rad,
                    ))
                    held_nominal_damping = -jacobian.T @ (
                        damping_diagonal * (jacobian @ qvel)
                    )
                    sampled_damping_peak_power = max(
                        sampled_damping_peak_power, float(held_nominal_damping @ qvel)
                    )
                reference = B601MitReference(
                    sequence=next_sequence,
                    source_timestamp_sec=timestamp,
                    receive_timestamp_sec=timestamp,
                    qpos_rad=tuple(float(value) for value in qpos),
                    qvel_rad_s=(0.0,) * 6,
                    joint_torque_overlay_nm=mapping.joint_torque_nm,
                )
                next_sequence += 1
                command_count += 1
                maximum_metric = max(maximum_metric, metric)
                minimum_metric = min(minimum_metric, metric)
                maximum_rotation_metric = max(
                    maximum_rotation_metric,
                    rotation_metric,
                )
                minimum_rotation_metric = min(
                    minimum_rotation_metric,
                    rotation_metric,
                )
                peak_force_n = max(
                    peak_force_n,
                    float(np.linalg.norm(fixture_result.wrench.force_n)),
                )
                peak_cartesian_torque_nm = max(
                    peak_cartesian_torque_nm,
                    float(np.linalg.norm(fixture_result.wrench.torque_nm)),
                )
                peak_requested_overlay_torque_nm = max(
                    peak_requested_overlay_torque_nm,
                    float(np.max(np.abs(mapping.joint_torque_nm))),
                )
                if fixture_result.force_limited or getattr(
                    fixture_result,
                    "torque_limited",
                    False,
                ):
                    fixture_limited_ticks += 1

            final_tick = driver.step_physics(reference)
            if contact_record is not None:
                contact_record.update(
                    command_applied=final_tick.command_applied,
                    plant_advanced=final_tick.plant_advanced,
                    applied_actuator_torque_nm=final_tick.applied_torque_nm,
                    interval_end_time_sec=driver.physics_time_sec,
                )
                if contact_diagnostics["first_sample"] is None:
                    contact_diagnostics["first_sample"] = contact_record
                contact_diagnostics["last_sample"] = contact_record
                if trace_stream is not None:
                    trace_stream.write(json.dumps(contact_record, allow_nan=False) + "\n")
            if (
                not phase_transitions
                or phase_transitions[-1]["phase"] != final_tick.control_phase
            ):
                phase_transitions.append(
                    {
                        "phase": final_tick.control_phase,
                        "time_sec": final_tick.physics_time_sec,
                        "reason": final_tick.phase_reason,
                    }
                )
            if final_tick.plant_advanced:
                advanced_ticks += 1
                if (
                    final_tick.control_phase == "ACTIVE"
                    and not final_tick.driver_faulted
                ):
                    channels = {
                        "applied_total": final_tick.applied_torque_nm,
                        "nominal_motor_local_damping": final_tick.driver_damping_torque_nm,
                    }
                    if fixture_name in POSE_FIXTURES:
                        channels["held_nominal_cartesian_damping"] = held_nominal_damping
                    for name, torque in channels.items():
                        _accumulate_power(
                            power_channels, name,
                            float(np.dot(torque, final_tick.control_qvel)),
                            driver.timing.physics_timestep_sec,
                        )
            if (
                final_tick.torque_breakdown is not None
                and final_tick.torque_breakdown.overlay_limited
            ):
                overlay_limited_ticks += 1
            torque_limited_ticks += int(final_tick.torque_limited)
            slew_limited_ticks += int(final_tick.slew_limited)
            peak_applied_torque_nm = max(
                peak_applied_torque_nm,
                float(np.max(np.abs(final_tick.applied_torque_nm))),
            )
            peak_abs_qvel_rad_s = max(
                peak_abs_qvel_rad_s,
                float(np.max(np.abs(final_tick.measured_qvel))),
            )
            peak_abs_qacc_rad_s2 = max(
                peak_abs_qacc_rad_s2,
                float(np.max(np.abs(final_tick.measured_qacc))),
            )
            if final_tick.driver_faulted:
                break
            if viewer_session is not None:
                if step % 16 == 0 and not viewer_session.sync():
                    viewer_closed_early = True
                    break
                time.sleep(1.0 / B601_RS_PHYSICS_HZ)
    if final_tick is None:
        raise RuntimeError("haptic fixture demo did not execute a physics tick")
    final_qpos = np.asarray(final_tick.measured_qpos, dtype=float)
    final_qvel = np.asarray(final_tick.measured_qvel, dtype=float)
    final_result, final_metric, final_rotation_metric = _fixture_metrics(
        fixture_name=fixture_name,
        model=model,
        fixture=fixture,
        qpos=final_qpos,
        qvel=final_qvel,
    )
    maximum_metric = max(maximum_metric, final_metric)
    minimum_metric = min(minimum_metric, final_metric)
    maximum_rotation_metric = max(maximum_rotation_metric, final_rotation_metric)
    minimum_rotation_metric = min(minimum_rotation_metric, final_rotation_metric)
    run_completed = advanced_ticks == requested_steps and not viewer_closed_early
    common_pass = bool(
        run_completed
        and final_tick.control_phase == "ACTIVE"
        and not final_tick.driver_faulted
        and final_tick.state_valid
        and final_tick.state_within_limits
        and torque_limited_ticks == 0
        and peak_applied_torque_nm <= max(profile.max_abs_torque_nm) + 1e-12
    )
    if fixture_name == CONTACT_FIXTURE:
        effect_pass = bool(
            contact_diagnostics["nonzero_wrench_samples"] > 0
            and contact_diagnostics["max_power_identity_error_w"] < 1e-9
            and contact_diagnostics["sample_count"] == command_count
        )
        effect_contract = {
            "metric": "physical_contact_wrench_and_diagnostic_power_identity",
            "power_identity_tolerance_w": 1e-9,
            "contact_force_reapplied": False,
        }
    elif fixture_name == "spring":
        effect_pass = bool(
            initial_metric > 0.02
            and final_metric <= 0.45 * initial_metric
            and maximum_metric <= 1.05 * initial_metric
        )
        effect_contract = {
            "metric": "cartesian_distance_to_anchor_m",
            "initial_minimum_m": 0.02,
            "final_over_initial_max": 0.45,
            "maximum_over_initial_max": 1.05,
        }
    elif fixture_name == "wall":
        effect_pass = bool(
            initial_metric >= 0.02
            and final_metric <= 0.002
            and maximum_metric <= initial_metric + 0.002
        )
        effect_contract = {
            "metric": "unilateral_wall_penetration_m",
            "initial_minimum_m": 0.02,
            "final_maximum_m": 0.002,
            "maximum_growth_tolerance_m": 0.002,
        }
    else:
        translation_required = (
            fixture_name.startswith("pose-t") or fixture_name == "pose-6d"
        )
        rotation_required = (
            fixture_name.startswith("pose-r") or fixture_name == "pose-6d"
        )
        translation_pass = bool(
            not translation_required
            or (
                initial_metric >= 0.005
                and final_metric <= 0.65 * initial_metric
                and maximum_metric <= 1.10 * initial_metric
            )
        )
        rotation_pass = bool(
            not rotation_required
            or (
                initial_rotation_metric >= 0.005
                and final_rotation_metric <= 0.65 * initial_rotation_metric
                and maximum_rotation_metric <= 1.10 * initial_rotation_metric
            )
        )
        effect_pass = translation_pass and rotation_pass
        effect_contract = {
            "metric": "separate_translation_m_and_rotation_rad_to_pose_anchor",
            "translation_required": translation_required,
            "rotation_required": rotation_required,
            "initial_translation_minimum_m": 0.005,
            "initial_rotation_minimum_rad": 0.005,
            "final_over_initial_max": 0.65,
            "maximum_over_initial_max": 1.10,
            "translation_passed": translation_pass,
            "rotation_passed": rotation_pass,
            "initial_rotation_metric": initial_rotation_metric,
            "final_rotation_metric": final_rotation_metric,
            "minimum_rotation_metric": minimum_rotation_metric,
            "maximum_rotation_metric": maximum_rotation_metric,
        }
    accepted = common_pass and effect_pass
    payload: dict[str, object] = {
        "fixture": fixture_name,
        "fixture_contract": fixture_contract,
        "effect_acceptance": {
            **effect_contract,
            "passed": effect_pass,
            "initial_metric": initial_metric,
            "final_metric": final_metric,
            "minimum_metric": minimum_metric,
            "maximum_metric": maximum_metric,
        },
        "power_diagnostics": {
            "quadrature": "left_rectangle_tau_dot_control_qvel_times_physics_dt",
            "scope": "ACTIVE_nonfaulted_advanced_ticks_only",
            "terminal_fault_tick_excluded": final_tick.driver_faulted,
            "positive_work_means": "mechanical_energy_delivered_to_arm",
            "nominal_damping_scope": "pose_fixture_unclipped_before_startup_blending_host_hold",
            "local_damping_scope": "pre_total_actuator_saturation_component_not_actual_split",
            "nominal_sampled_peak_power_w": sampled_damping_peak_power,
            "channels": power_channels,
            "passivity_claim": False,
            "hardware_claim": False,
        },
        "simulation_acceptance": {
            "passed": accepted,
            "run_completed": run_completed,
            "requested_physics_ticks": requested_steps,
            "advanced_physics_ticks": advanced_ticks,
            "command_count": command_count,
            "expected_command_count": math.ceil(
                duration * float(driver.timing.command_hz)
            ),
            "viewer_closed_early": viewer_closed_early,
            "driver_faulted": final_tick.driver_faulted,
            "phase_transitions": phase_transitions,
            "final_control_phase": final_tick.control_phase,
            "fault_reason": final_tick.fault_reason,
            "state_valid": final_tick.state_valid,
            "state_within_limits": final_tick.state_within_limits,
            "fixture_force_limited_commands": fixture_limited_ticks,
            "overlay_limited_physics_ticks": overlay_limited_ticks,
            "torque_limited_physics_ticks": torque_limited_ticks,
            "slew_limited_physics_ticks": slew_limited_ticks,
            "peak_force_norm_n": peak_force_n,
            "peak_cartesian_torque_norm_nm": peak_cartesian_torque_nm,
            "peak_requested_overlay_torque_nm": (peak_requested_overlay_torque_nm),
            "peak_applied_torque_nm": peak_applied_torque_nm,
            "peak_abs_qvel_rad_s": peak_abs_qvel_rad_s,
            "peak_abs_qacc_rad_s2": peak_abs_qacc_rad_s2,
            "final_force_n": list(final_result.wrench.force_n),
            "final_cartesian_torque_nm": list(final_result.wrench.torque_nm),
            "final_qpos_rad": list(final_tick.measured_qpos),
            "final_qvel_rad_s": list(final_tick.measured_qvel),
            "final_qacc_rad_s2": list(final_tick.measured_qacc),
        },
        "driver": driver.execution_metadata,
        "fidelity": (
            "Opt-in host-ZOH/local-damping diagnostic on an unidentified ideal plant. "
            "Fixture acceptance is not real2sim fidelity acceptance. "
            if real2sim_controller
            else ""
        )
        + (
            "offline deterministic ideal-generalized-effort MuJoCo acceptance; "
            "validates frame math, J-transpose direction, bounded software "
            "overlay, gravity/damping composition and closed-loop fixture effect; "
            "does not validate a motor, CAN, friction, torque calibration, force "
            "fidelity, haptic feel, contact sensor, hard real-time timing or "
            "physical safety"
        ),
    }
    if fixture_name == CONTACT_FIXTURE:
        payload["contact_diagnostics"] = contact_diagnostics
        payload["fidelity"] = (
            "Offline native mesh/box contact on an unidentified ideal-effort plant; "
            "contact reaction is applied only by MuJoCo, its J-transpose projection "
            "is diagnostic and never dispatched. Does not validate a motor, CAN, "
            "absolute force, haptic feel, passivity or physical safety."
        )
    return (0 if accepted else 3), payload


def _accumulate_power(
    channels: dict[str, dict[str, float | int]], name: str, power_w: float, dt: float,
) -> None:
    """Keep signed work separate from positive delivery and absorption magnitudes."""
    record = channels.setdefault(name, {
        "sample_count": 0, "net_work_j": 0.0,
        "delivered_work_j": 0.0, "absorbed_work_j": 0.0,
        "peak_positive_power_w": 0.0,
    })
    record["sample_count"] += 1
    record["net_work_j"] += power_w * dt
    record["delivered_work_j"] += max(0.0, power_w) * dt
    record["absorbed_work_j"] += max(0.0, -power_w) * dt
    record["peak_positive_power_w"] = max(record["peak_positive_power_w"], power_w)


def _emit_payload(
    *,
    args: argparse.Namespace,
    code: int,
    payload: dict[str, object],
) -> int:
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_root is None:
        print(serialized)
        return code

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(
        tempfile.mkdtemp(prefix=f"b601-{args.fixture}-", dir=str(output_root))
    )
    result_path = run_dir / "result.json"
    result_path.write_text(serialized + "\n", encoding="utf-8")
    acceptance = payload.get("simulation_acceptance")
    passed = (
        acceptance.get("passed")
        if isinstance(acceptance, dict)
        else code == 0
    )
    status = "PASS" if code == 0 and passed is True else "FAIL"
    print(
        f"[B601-SIM] fixture={args.fixture} status={status} exit_code={code}"
    )
    print(f"[B601-SIM] full result: {result_path}")
    print("[B601-SIM] shell remains active")
    return code


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.fixture == SHARED_CONTROL_FIXTURE:
        if (
            args.real2sim_controller
            or args.gravity_control_baseline != "operator"
            or args.startup_lock_kp is not None
            or args.startup_lock_kd is not None
            or args.fixture_stiffness_scale != 1.0
            or args.fixture_damping_scale != 1.0
            or args.torque_rise_time_sec is not None
            or args.feedback_delay_sec != 0.0
            or args.trace_jsonl is not None
        ):
            parser.error(
                "policy-assist uses its fixed offline lifecycle profile; unrelated fixture tuning is unsupported"
            )
        code, payload = run_b601_shared_control_demo(
            source_root=args.rebot_source_root,
            duration_sec=args.duration_sec,
            viewer=bool(args.viewer),
            host_hz=args.host_hz,
        )
        return _emit_payload(args=args, code=code, payload=payload)
    if args.fixture == REFLECTION_FIXTURE:
        if (
            args.real2sim_controller
            or args.gravity_control_baseline != "operator"
            or args.startup_lock_kp is not None
            or args.startup_lock_kd is not None
            or args.fixture_stiffness_scale != 1.0
            or args.fixture_damping_scale != 1.0
            or args.torque_rise_time_sec is not None
        ):
            parser.error(
                "force-reflection has a fixed offline profile; fixture/real2sim retuning is not supported"
            )
        from .reflection_demo import run_force_reflection_demo

        code, payload = run_force_reflection_demo(
            source_root=args.rebot_source_root,
            duration_sec=args.duration_sec,
            initial_qpos=DEFAULT_HAPTIC_INITIAL_QPOS,
            viewer=bool(args.viewer),
            trace_jsonl=args.trace_jsonl,
            feedback_delay_sec=args.feedback_delay_sec,
        )
        return _emit_payload(args=args, code=code, payload=payload)
    if args.feedback_delay_sec != 0.0:
        parser.error("feedback-delay-sec requires force-reflection")
    controller = None
    if args.gravity_control_baseline == "raw-model":
        if not args.real2sim_controller:
            parser.error("raw-model gravity baseline requires --real2sim-controller")
        controller = B601Real2SimControllerProfile.raw_model_baseline()
    startup = None
    if args.startup_lock_kp is not None or args.startup_lock_kd is not None:
        if (
            not args.real2sim_controller
            or args.startup_lock_kp is None
            or args.startup_lock_kd is None
        ):
            parser.error(
                "startup lock requires --real2sim-controller and both --startup-lock-kp/--startup-lock-kd"
            )
        startup = B601SimStartupProfile(
            tuple(args.startup_lock_kp), tuple(args.startup_lock_kd)
        )
    code, payload = run_haptic_fixture_demo(
        source_root=args.rebot_source_root,
        fixture_name=args.fixture,
        duration_sec=args.duration_sec,
        viewer=bool(args.viewer),
        real2sim_controller=args.real2sim_controller,
        host_hz=args.host_hz,
        startup_profile=startup,
        fixture_stiffness_scale=args.fixture_stiffness_scale,
        fixture_damping_scale=args.fixture_damping_scale,
        torque_rise_time_sec=args.torque_rise_time_sec,
        controller_profile=controller,
        trace_jsonl=args.trace_jsonl,
    )
    return _emit_payload(args=args, code=code, payload=payload)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_HAPTIC_ANCHOR_QPOS",
    "DEFAULT_HAPTIC_DURATION_SEC",
    "DEFAULT_HAPTIC_INITIAL_QPOS",
    "run_haptic_fixture_demo",
]
