"""Offline dual-plant force-feedback assembly behind the existing haptic CLI."""

from __future__ import annotations

import json
import math
import time
from collections import deque
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import numpy as np

from robot_motion.rebot_b601_rs import (
    B601_RS_ARM_JOINT_NAMES,
    B601_RS_WORLD_ALIGNED_EE_WRENCH_FRAME,
    B601HapticWrenchChannel,
    B601HapticWrenchSample,
    B601MitReference,
    B601PinocchioModel,
    B601TranslationalVirtualSpring,
    map_world_aligned_ee_wrench_to_joint_torque,
    transform_spatial_twist,
    transform_spatial_wrench,
)
from sim.mujoco.viewer import MujocoPassiveViewerSession

from .contact import build_b601_contact_box_fixture, read_b601_contact_observation
from .mit_driver import B601MujocoEffortDriver, default_b601_mit_profile
from .scene import B601_RS_PHYSICS_HZ, build_b601_mujoco_scene


def run_force_reflection_demo(
    *,
    source_root: str | Path,
    initial_qpos: Sequence[float],
    duration_sec: float = 1.5,
    viewer: bool = False,
    trace_jsonl: str | Path | None = None,
    feedback_delay_sec: float = 0.0,
    feedback_gain: float = 1.0,
) -> tuple[int, dict[str, object]]:
    """Reflect native contact from a synthetic B601 follower to a separate master.

    There is no motion/teleop/policy return path. Master displacement is never
    fed back as human intent. Both exclusive drivers use the existing MIT
    gravity/damping and final torque conditioning; this assembly owns only
    the local sample delivery schedule and reference production.
    """
    duration = float(duration_sec)
    delay = float(feedback_delay_sec)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration_sec must be finite and positive")
    if not math.isfinite(delay) or delay < 0.0:
        raise ValueError("feedback_delay_sec must be finite and non-negative")
    if not isinstance(viewer, bool):
        raise TypeError("viewer must be a bool")
    # A virtual tip-origin coupling: Ry(+90 deg), with no base-offset lever arm.
    target_T_source = np.eye(4)
    target_T_source[:3, :3] = ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0))
    channel = B601HapticWrenchChannel(
        source_frame_id=B601_RS_WORLD_ALIGNED_EE_WRENCH_FRAME,
        target_T_source=target_T_source,
        gain=feedback_gain,
        max_abs_force_n=(2.0,) * 3,
        max_abs_torque_nm=(0.2,) * 3,
        max_force_rate_n_s=(5.0,) * 3,
        max_torque_rate_nm_s=(2.0,) * 3,
        filter_tau_sec=0.02,
        maximum_source_age_sec=0.1,
    )
    model = B601PinocchioModel.from_rebot_source_root(source_root)
    follower, box = build_b601_contact_box_fixture(source_root, initial_qpos)
    master = build_b601_mujoco_scene(
        source_root,
        initial_arm_qpos=initial_qpos,
        gravity_mps2=(0.0, 0.0, -9.81),
        actuation_mode="effort",
    )
    drivers = {}
    springs = {}
    for name, scene in (("follower", follower), ("master", master)):
        profile = replace(
            default_b601_mit_profile(scene), max_abs_overlay_torque_nm=(6.0,) * 6
        )
        driver = B601MujocoEffortDriver(
            scene,
            model,
            profile=profile,
            enable_gravity_compensation=True,
        )
        driver.reset(initial_qpos)
        drivers[name] = driver
        springs[name] = B601TranslationalVirtualSpring(
            anchor_position_m=tuple(model.forward_kinematics(initial_qpos)[:3, 3]),
            stiffness_n_per_m=(80.0,) * 3,
            damping_n_s_per_m=(8.0,) * 3,
            max_abs_force_n=(8.0,) * 3,
        )

    def state(scene):
        return scene.get_arm_qpos("b601_rs"), np.asarray(
            [
                scene.data.qvel[scene.joint_dof_addr(joint)]
                for joint in B601_RS_ARM_JOINT_NAMES
            ]
        )

    def spring_torque(name, qpos, qvel):
        result = springs[name].evaluate(
            model.forward_kinematics(qpos)[:3, 3],
            (model.arm_jacobian(qpos) @ qvel)[:3],
        )
        return np.asarray(
            map_world_aligned_ee_wrench_to_joint_torque(
                model, qpos, result.wrench
            ).joint_torque_nm
        )

    stats = {
        name: {
            "advanced_ticks": 0,
            "rejected_references": 0,
            "fault_reason": None,
            "torque_limited_ticks": 0,
            "overlay_limited_ticks": 0,
            "peak_abs_velocity_rad_s": 0.0,
            "peak_abs_acceleration_rad_s2": 0.0,
        }
        for name in drivers
    }
    diagnostics = {
        "sample_count": 0,
        "nonzero_source_samples": 0,
        "nonzero_master_samples": 0,
        "rejection_counts": {},
        "peak_source_force_n": 0.0,
        "peak_feedback_force_n": 0.0,
        "peak_feedback_joint_torque_nm": 0.0,
        "max_transform_power_error_w": 0.0,
        "max_projection_power_error_w": 0.0,
        "nominal_feedback_delivered_work_j": 0.0,
        "nominal_feedback_absorbed_work_j": 0.0,
        "first_sample": None,
        "last_sample": None,
    }
    pending = deque()
    sequence = 0
    held_feedback = np.zeros(6)
    requested_ticks = math.ceil(duration * B601_RS_PHYSICS_HZ)
    session = MujocoPassiveViewerSession(master) if viewer else None
    viewer_closed = False
    with ExitStack() as cleanup:
        stream = (
            None
            if trace_jsonl is None
            else cleanup.enter_context(Path(trace_jsonl).open("x"))
        )
        if session is not None:
            cleanup.callback(session.close)
            session.start()
        for _ in range(requested_ticks):
            references = {}
            record = None
            if drivers["master"].command_due:
                now = float(master.data.time)
                fq, fv = state(follower)
                mq, mv = state(master)
                observation = read_b601_contact_observation(follower)
                if observation.physics_time_sec != now:
                    raise RuntimeError(
                        "the two plant clocks must match before feedback sampling"
                    )
                source = B601HapticWrenchSample(sequence, now, now, observation.wrench)
                pending.append(source)
                delivered = None
                while pending and pending[0].source_timestamp_sec + delay <= now:
                    # Actual local delivery time changes; original acquisition time does not.
                    delivered = replace(pending.popleft(), receive_timestamp_sec=now)
                output = channel.update(delivered, now_sec=now)
                projection = map_world_aligned_ee_wrench_to_joint_torque(
                    model, mq, output.wrench, qvel_rad_s=mv
                )
                held_feedback = np.asarray(projection.joint_torque_nm)
                raw_mapped = transform_spatial_wrench(
                    observation.wrench,
                    target_frame_id=B601_RS_WORLD_ALIGNED_EE_WRENCH_FRAME,
                    target_T_source=target_T_source,
                )
                master_twist = model.arm_jacobian(mq) @ mv
                virtual_source_twist = transform_spatial_twist(
                    master_twist, target_T_source=np.linalg.inv(target_T_source)
                )
                transform_error = abs(
                    raw_mapped.as_vector() @ master_twist
                    - observation.wrench.as_vector() @ virtual_source_twist
                )
                fs = spring_torque("follower", fq, fv)
                ms = spring_torque("master", mq, mv)
                references = {
                    name: B601MitReference(
                        sequence=sequence,
                        source_timestamp_sec=now,
                        receive_timestamp_sec=now,
                        qpos_rad=tuple(q),
                        joint_torque_overlay_nm=tuple(overlay),
                    )
                    for name, q, overlay in (
                        ("follower", fq, fs),
                        ("master", mq, ms + held_feedback),
                    )
                }
                diagnostics["sample_count"] += 1
                diagnostics["nonzero_source_samples"] += int(
                    np.linalg.norm(observation.wrench.force_n) > 1e-6
                )
                diagnostics["nonzero_master_samples"] += int(
                    np.linalg.norm(output.wrench.as_vector()) > 1e-6
                )
                if output.rejection_reason is not None:
                    counts = diagnostics["rejection_counts"]
                    counts[output.rejection_reason] = (
                        counts.get(output.rejection_reason, 0) + 1
                    )
                for key, value in (
                    ("peak_source_force_n", np.linalg.norm(observation.wrench.force_n)),
                    ("peak_feedback_force_n", np.linalg.norm(output.wrench.force_n)),
                    ("peak_feedback_joint_torque_nm", np.max(np.abs(held_feedback))),
                    ("max_transform_power_error_w", transform_error),
                    (
                        "max_projection_power_error_w",
                        abs(projection.cartesian_power_w - projection.joint_power_w),
                    ),
                ):
                    diagnostics[key] = max(diagnostics[key], float(value))
                record = {
                    "sequence": sequence,
                    "physics_time_sec": now,
                    "follower_qpos_rad": fq.tolist(),
                    "follower_qvel_rad_s": fv.tolist(),
                    "master_qpos_rad": mq.tolist(),
                    "master_qvel_rad_s": mv.tolist(),
                    "source_wrench": observation.wrench.as_vector().tolist(),
                    "source_frame_id": observation.wrench.frame_id,
                    "feedback_wrench": output.wrench.as_vector().tolist(),
                    "feedback_frame_id": output.wrench.frame_id,
                    "feedback_source_sequence": output.source_sequence,
                    "feedback_source_age_sec": output.source_age_sec,
                    "feedback_source_timestamp_sec": output.source_timestamp_sec,
                    "feedback_receive_timestamp_sec": output.receive_timestamp_sec,
                    "delivered_source_timestamp_sec": None if delivered is None else delivered.source_timestamp_sec,
                    "delivered_receive_timestamp_sec": None if delivered is None else delivered.receive_timestamp_sec,
                    "feedback_rejection_reason": output.rejection_reason,
                    "follower_overlay_requested_nm": fs.tolist(),
                    "master_spring_overlay_nm": ms.tolist(),
                    "master_feedback_overlay_nm": held_feedback.tolist(),
                    "master_overlay_requested_nm": (ms + held_feedback).tolist(),
                }
                sequence += 1
            ticks = {
                name: driver.step_physics(references.get(name))
                for name, driver in drivers.items()
            }
            for name, tick in ticks.items():
                values = stats[name]
                values["advanced_ticks"] += int(tick.plant_advanced)
                values["rejected_references"] += int(
                    tick.command_received and not tick.command_applied
                )
                values["fault_reason"] = tick.fault_reason
                values["torque_limited_ticks"] += int(tick.torque_limited)
                values["overlay_limited_ticks"] += int(
                    tick.torque_breakdown is not None
                    and tick.torque_breakdown.overlay_limited
                )
                values["peak_abs_velocity_rad_s"] = max(
                    values["peak_abs_velocity_rad_s"],
                    float(np.max(np.abs(tick.measured_qvel))),
                )
                values["peak_abs_acceleration_rad_s2"] = max(
                    values["peak_abs_acceleration_rad_s2"],
                    float(np.max(np.abs(tick.measured_qacc))),
                )
            # Nominal requested-feedback work, not a decomposition after total torque slew.
            power = float(held_feedback @ np.asarray(ticks["master"].control_qvel))
            diagnostics["nominal_feedback_delivered_work_j"] += (
                max(0.0, power) / B601_RS_PHYSICS_HZ
            )
            diagnostics["nominal_feedback_absorbed_work_j"] += (
                max(0.0, -power) / B601_RS_PHYSICS_HZ
            )
            if record is not None:
                record.update(
                    interval_end_time_sec=float(master.data.time),
                    master_command_applied=ticks["master"].command_applied,
                    follower_command_applied=ticks["follower"].command_applied,
                    master_applied_torque_nm=ticks["master"].applied_torque_nm,
                )
                if diagnostics["first_sample"] is None:
                    diagnostics["first_sample"] = record
                diagnostics["last_sample"] = record
                if stream is not None:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
            if any(tick.driver_faulted for tick in ticks.values()):
                break
            if session is not None:
                if not session.sync():
                    viewer_closed = True
                    break
                time.sleep(1.0 / B601_RS_PHYSICS_HZ)
    for name, scene in (("master", master), ("follower", follower)):
        stats[name]["final_qpos_rad"] = scene.get_arm_qpos("b601_rs").tolist()
    elapsed = stats["master"]["advanced_ticks"] / B601_RS_PHYSICS_HZ
    # Finite-horizon bound only: |tau_h| <= overlay limit, |dq| <= state limit.
    work_bound = (
        float(
            np.asarray(drivers["master"].profile.max_abs_overlay_torque_nm)
            @ np.asarray(drivers["master"].profile.max_abs_state_velocity_rad_s)
        )
        * elapsed
    )
    passed = bool(
        all(
            s["advanced_ticks"] == requested_ticks
            and s["fault_reason"] is None
            and s["rejected_references"] == 0
            and s["torque_limited_ticks"] == 0
            and s["overlay_limited_ticks"] == 0
            for s in stats.values()
        )
        and diagnostics["nonzero_source_samples"] > 0
        and diagnostics["nonzero_master_samples"] > 0
        and diagnostics["max_transform_power_error_w"] < 1e-9
        and diagnostics["max_projection_power_error_w"] < 1e-9
        and diagnostics["peak_feedback_joint_torque_nm"] <= min(drivers["master"].profile.max_abs_overlay_torque_nm)
        and diagnostics["nominal_feedback_delivered_work_j"] <= work_bound
        and not viewer_closed
    )
    return (0 if passed else 3), {
        "fixture": "force-reflection",
        "passed": passed,
        "plants": stats,
        "diagnostics": diagnostics,
        "feedback_delay_sec": delay,
        "target_T_source": target_T_source.tolist(),
        "feedback_gain": channel.gain,
        "contact_object": box.to_dict(),
        "trace_jsonl": None if trace_jsonl is None else str(trace_jsonl),
        "finite_horizon_nominal_feedback_work_bound_j": work_bound,
        "viewer_target": "master_only",
        "motion_return_channel": "absent",
        "sample_phase": "both_current_states_before_reference_admission",
        "fidelity": "Synthetic B601 follower only, not RM65. Unidentified ideal-effort plants; no hardware I/O, force calibration, passivity or intent separation claim. Bounded finite-horizon nominal work is not an energy-tank guarantee.",
    }
