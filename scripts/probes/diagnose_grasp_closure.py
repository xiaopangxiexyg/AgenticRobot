"""Offline measured-closure experiment using the existing guarded MuJoCo rig.

An explicit external fixture supplies object assumptions and two saved grasp
states. Cross position/orientation about the calibrated model origin, not TCP.
Only reset fixtures assign qpos; after start the original executors own motion.
This is not full approach, Gateway, real-device or physical-pad acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from collections import deque
from dataclasses import replace
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from deployments.perception.dmtac_grasp_overlay import (
    reviewed_graspgenx_dmtac_solid_geometry,
)
from project_paths import REPO_ROOT
from robot_motion.pose import pose_delta_metrics
from robot_motion.rm65.position_collision import RM65PositionCollisionPlannerConfig
from robot_motion.runtime import EffectorCommand, TeleopCommand
from robot_motion.transforms import validate_pose_matrix
from sim.mujoco.adapters import (
    MujocoPositionExecutionConfig,
    MujocoPositionExecutionRig,
)
from sim.mujoco.grasp_contact import GripperContactSurfaceSpec
from sim.mujoco.grasp_contact_diagnostics import MujocoGraspContactSampler
from sim.mujoco.ik import PoseTarget, RM65CanonicalIK
from sim.mujoco.scene import MujocoScene
from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig
from sim.mujoco.tool_frames import load_endpoint_to_tool_transform


def _pose(value):
    matrix = np.asarray(value, dtype=float).reshape(4, 4)
    valid, reason = validate_pose_matrix(matrix, atol=1e-6)
    if not valid:
        raise ValueError(f"Invalid fixture pose: {reason}")
    return matrix


def crossed_model_target(center_pose, orientation_pose, tool_from_model):
    """Hold the declared grasp-model origin while exchanging its orientation."""
    center, orientation, offset = map(_pose, (center_pose, orientation_pose, tool_from_model))
    model_pose = orientation.copy()
    model_pose[:3, 3] = center[:3, 3]
    return model_pose, model_pose @ np.linalg.inv(offset)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _forbid_network(event, _args):
    if event in {"socket.__new__", "socket.connect", "socket.bind", "socket.getaddrinfo"}:
        raise RuntimeError("Closure diagnostic forbids network/device connections")


def _endpoint_fixture(scene, config, target_world, seeds, *, budget_sec):
    began = time.perf_counter()
    attempts = []
    side = config.controlled_side
    base_from_world = np.linalg.inv(scene.body_pose(side + "_arm_base_link"))
    for seed_index, seed in enumerate(seeds):
        scene.set_qpos_by_side(seed)
        ik = RM65CanonicalIK.from_tool_profiles(scene,
            {s: getattr(config, s + "_tool_profile_path") for s in ("left", "right")},
            planner_config=RM65PositionCollisionPlannerConfig(scene_profile="dual_arm_hand_proxy_dmtac_grasp"))
        last = None
        for cycle in range(600):
            if time.perf_counter() - began >= budget_sec:
                break
            last = ik.solve({side: PoseTarget(side=side, pose=base_from_world @ target_world)})
            if (last.success and not last.hold
                    and last.position_error_m[side] <= .0002
                    and last.orientation_error_rad[side] <= np.deg2rad(.1)):
                qpos = {s: np.asarray(last.qpos_by_side[s], dtype=float) for s in ("left", "right")}
                passive = "left" if side == "right" else "right"
                if not np.allclose(qpos[passive], seed[passive], atol=1e-10, rtol=0):
                    raise RuntimeError("Endpoint fixture changed the passive arm")
                attempts.append({"seed_index": seed_index, "cycles": cycle + 1, "status": last.status,
                    "position_error_m": last.position_error_m, "orientation_error_rad": last.orientation_error_rad})
                return qpos, {"status": "endpoint_fixture_found", "attempts": attempts,
                    "elapsed_sec": time.perf_counter() - began,
                    "scope": "Bounded local canonical IK, not a complete approach path"}
        attempts.append({"seed_index": seed_index, "status": None if last is None else last.status,
            "position_error_m": None if last is None else last.position_error_m,
            "orientation_error_rad": None if last is None else last.orientation_error_rad,
            "cycles": cycle + 1})
    return None, {"status": "endpoint_fixture_not_found_within_budget", "attempts": attempts,
        "elapsed_sec": time.perf_counter() - began}


def _build_scene(fixture, output):
    config = CubeLiftTaskConfig.load(fixture["task_config"])
    contract = config.base_contract()
    edge = float(fixture["object_edge_m"])
    if not np.isfinite(edge) or edge <= 0:
        raise ValueError("Object edge must be finite and positive")
    center = np.asarray(fixture["object_center_world_m"], dtype=float)
    quaternion = np.asarray(fixture["object_quaternion_wxyz"], dtype=float)
    if (center.shape != (3,) or quaternion.shape != (4,) or not np.isfinite(center).all()
            or not np.isfinite(quaternion).all() or not np.isclose(np.linalg.norm(quaternion), 1., atol=1e-6)):
        raise ValueError("Object fixture requires a finite center and unit quaternion")
    objects = tuple(replace(obj, size_m=(edge,)*3, initial_position_m=tuple(center),
        initial_quaternion_wxyz=tuple(quaternion), inertia_diagonal_kg_m2=None)
        if obj.name == config.object_body_name else obj for obj in contract.scene.objects)
    if sum(obj.name == config.object_body_name for obj in objects) != 1:
        raise ValueError("The fixture must contain the existing task object exactly once")
    contract = replace(contract, scene=replace(contract.scene, objects=objects,
        profile_name=contract.scene.profile_name + "_closure_diagnostic"))
    profile = config.gripper_actuation_profile()
    scene = MujocoScene.from_contract(contract, mjcf_path=output/"base.xml",
        position_actuation_profile=config.position_actuation_profile(), gripper_actuation_profile=profile)
    width, top = float(fixture["support_width_m"]), float(center[2] - edge/2)
    if not np.isfinite(width) or width <= 0 or top <= .01:
        raise ValueError("Support fixture dimensions are invalid")
    xml = ET.parse(scene.conversion.mjcf_path)
    world = xml.getroot().find("worldbody")
    if world is None:
        raise ValueError("Converted scene has no worldbody")
    body = ET.SubElement(world, "body", name="contact_fixture_support",
        pos=" ".join(map(str, (center[0], center[1], (.01+top)/2))))
    ET.SubElement(body, "geom", name="contact_fixture_support_geom", type="box",
        size=f"{width/2} {width/2} {(top-.01)/2}", contype="1", conaffinity="14",
        rgba="0.6 0.15 0.12 1", friction="1 0.005 0.0001", condim="4")
    path = output/"fixture.xml"
    xml.write(path, encoding="utf-8", xml_declaration=True)
    model = scene.mujoco.MjModel.from_xml_path(str(path))
    return config, contract, profile, MujocoScene.from_compiled_contract(contract, model=model, mjcf_path=path)


def _run_case_contents(fixture_path, output, report, register_resource, *,
                       center_key, orientation_key, lift, ik_budget_sec):
    fixture_path = Path(fixture_path).resolve(strict=True)
    fixture = json.loads(fixture_path.read_text())
    config, contract, profile, scene = _build_scene(fixture, output)
    side = config.controlled_side
    geometry = reviewed_graspgenx_dmtac_solid_geometry()
    tool_from_model = geometry.solid_pose(np.eye(4))
    poses = {key: _pose(value["model_pose_world"]) for key, value in fixture["grasps"].items()}
    target_model, target_tcp = crossed_model_target(poses[center_key], poses[orientation_key], tool_from_model)
    tcp_offset = load_endpoint_to_tool_transform(getattr(config, side+"_tool_profile_path"), side=side)
    seeds = [{s: np.asarray(value["qpos_by_side"][s], dtype=float) for s in ("left", "right")}
             for value in (fixture["grasps"][center_key], fixture["grasps"][orientation_key])]
    report.update({"scope": "Native terminal closure fixture, not full approach/Gateway/real acceptance",
        "fixture_sha256": _sha(fixture_path), "probe_sha256": _sha(__file__),
        "task_config_sha256": _sha(fixture["task_config"]),
        "native_mujoco_version": scene.mujoco.__version__,
        "tool_profile_sha256_by_side": {s: _sha(getattr(config, s+"_tool_profile_path"))
            for s in ("left", "right")},
        "sampler_sha256": _sha(REPO_ROOT/"sim/mujoco/grasp_contact_diagnostics.py"),
        "center_key": center_key, "orientation_key": orientation_key,
        "rotation_pivot": "Calibrated GraspGenX model origin, not measured pressure centroid",
        "target_model_world": target_model.tolist(), "target_tcp_world": target_tcp.tolist(),
        "object_pose_evidence": fixture["object_pose_evidence"],
        "object_edge_m": fixture["object_edge_m"], "support_width_m": fixture["support_width_m"],
        "mass_is_simulator_assumption": True, "friction_and_servo_unchanged": True,
        "physics_hz": contract.timing.physics_hz, "hardware_network_model_calls": 0,
        "pad_contract": "No finite active tactile-pad boundary is declared; collision-face labels only"})
    passive = "left" if side == "right" else "right"
    baseline = np.asarray(next(iter(fixture["grasps"].values()))["qpos_by_side"][passive], dtype=float)
    for grasp in fixture["grasps"].values():
        if not np.allclose(grasp["qpos_by_side"][passive], baseline, atol=1e-10, rtol=0):
            raise ValueError("Crossed fixtures must share the same passive arm state")
    report["passive_arm_qpos_baseline_rad"] = baseline.tolist()
    if center_key == orientation_key:
        qpos = seeds[0]
        report["endpoint"] = {"status": "retained_terminal_qpos", "elapsed_sec": 0.}
    else:
        qpos, report["endpoint"] = _endpoint_fixture(scene, config, target_tcp, seeds, budget_sec=ik_budget_sec)
    if qpos is None:
        report["status"] = "endpoint_fixture_rejected"
        return
    scene.set_qpos_by_side(qpos)
    for name in scene.gripper_specs:
        scene.set_gripper_close(name, 0.)
    scene.set_freejoint_state(config.object_freejoint_name,
        qpos=tuple(fixture["object_center_world_m"])+tuple(fixture["object_quaternion_wxyz"]), qvel=np.zeros(6))
    scene.forward()
    actual_tcp = scene.body_pose(scene.arm_specs[side].controlled_body) @ tcp_offset
    position_error, rotation_error = pose_delta_metrics(target_tcp, actual_tcp)
    report["fixture_FK_error"] = {"position_m": position_error, "orientation_deg": rotation_error}
    report["initial_state"] = {
        "qpos_by_side_rad": {s: scene.get_arm_qpos(s).tolist() for s in ("left", "right")},
        "model_qpos": scene.data.qpos.tolist(),
        "tcp_world": actual_tcp.tolist(),
        "grasp_model_world": (actual_tcp @ tool_from_model).tolist(),
        "object_world": scene.body_pose(config.object_body_name).tolist(),
    }
    if position_error > .0002 or rotation_error > .1:
        raise ValueError("Fixture qpos does not reproduce the declared crossed grasp")
    sensor_names = {f: f"{side}_gripper_{f}_dm_tac_w2m_sensor" for f in ("l", "r")}
    specs = []
    for finger, name in sensor_names.items():
        pose = scene.body_pose(name)
        opposite = scene.body_pose(sensor_names["r" if finger == "l" else "l"])
        inward = pose[:3, :3].T @ (opposite[:3, 3] - pose[:3, 3])
        specs.append(GripperContactSurfaceSpec(name, finger, tuple(inward)))
    sampler = MujocoGraspContactSampler(scene, object_body_name=config.object_body_name, surface_specs=tuple(specs))
    clock = [0.]
    rig = MujocoPositionExecutionRig(scene, MujocoPositionExecutionConfig(timing=contract.timing,
        actuation_profile=config.position_actuation_profile(), gripper_actuation_profile=profile,
        realtime_worker=False), monotonic_clock=lambda: clock[0])
    register_resource("rig", rig)
    if int(scene.model.opt.integrator) == int(scene.mujoco.mjtIntegrator.mjINT_RK4):
        raise ValueError("Pre-integration contact diagnostics do not support RK4 stage timing")
    hand_name = getattr(config, side+"_gripper_name")
    initial_object = scene.body_pose(config.object_body_name)
    initial_tcp = actual_tcp.copy()
    first_geometric, first_loaded = {}, {}
    first_pairs = {}
    tail_bilateral = deque(maxlen=25)
    report.update(first_geometric_contact_by_finger=first_geometric,
        first_load_contact_by_finger=first_loaded, first_contact_by_pair=first_pairs,
        phase_extrema={}, first_bilateral_body_force_contact=None,
        first_bilateral_inward_box_contact=None, maximum_object_penetration_m=0.,
        maximum_object_penetration_sample=None)
    renderer = None
    phase = "open"
    peak = {"cube_rotation_deg": -1., "cube_translation_m": -1.}
    trace = (output/"trace.jsonl").open("x")
    register_resource("trace", trace)
    events = (output/"events.jsonl").open("x")
    register_resource("events", events)
    evaluation_state = {"time_sec": float(scene.data.time),
        "measured_close": scene.get_gripper_close(hand_name), "evaluation_phase": "initial_forward"}
    def render(label):
        nonlocal renderer
        from PIL import Image
        if renderer is None:
            renderer = scene.mujoco.Renderer(scene.model, height=480, width=640)
            register_resource("renderer", renderer)
        camera = scene.mujoco.MjvCamera()
        camera.lookat[:] = fixture["object_center_world_m"]
        camera.distance, camera.azimuth, camera.elevation = .32, 125., -25.
        renderer.update_scene(scene.data, camera=camera)
        Image.fromarray(renderer.render()).save(output/(label+".png"))
    def sample(*, persist=False):
        object_pose = scene.body_pose(config.object_body_name)
        tcp = scene.body_pose(scene.arm_specs[side].controlled_body) @ tcp_offset
        translation, rotation = pose_delta_metrics(initial_object, object_pose)
        _, tcp_rotation = pose_delta_metrics(initial_tcp, tcp)
        contacts = sampler.sample()
        row = {**evaluation_state, "phase": phase,
            "integration_time_sec": float(scene.data.time),
            "post_integration_measured_close": scene.get_gripper_close(hand_name),
            "cube_xyz_m": object_pose[:3, 3].tolist(), "tcp_xyz_m": tcp[:3, 3].tolist(),
            "cube_rotation_deg": rotation, "tcp_rotation_deg": tcp_rotation,
            "cube_translation_m": translation, "cube_delta_world_m": (object_pose[:3, 3]-initial_object[:3, 3]).tolist(),
            "cube_rotation_world": object_pose[:3, :3].tolist(),
            "contacts": contacts}
        extrema = report["phase_extrema"].setdefault(phase, {"maximum_cube_rotation_deg": 0.,
            "maximum_cube_translation_m": 0., "maximum_tcp_rotation_deg": 0., "sample_count": 0})
        extrema["maximum_cube_rotation_deg"] = max(extrema["maximum_cube_rotation_deg"], rotation)
        extrema["maximum_cube_translation_m"] = max(extrema["maximum_cube_translation_m"], translation)
        extrema["maximum_tcp_rotation_deg"] = max(extrema["maximum_tcp_rotation_deg"], tcp_rotation)
        extrema["sample_count"] += 1
        loaded, inward = set(), set()
        for contact in contacts:
            event = {"time_sec": row["time_sec"], "phase": phase,
                "measured_close": row["measured_close"], "contact": contact}
            if -contact["distance_m"] > report["maximum_object_penetration_m"]:
                report["maximum_object_penetration_m"] = -contact["distance_m"]
                report["maximum_object_penetration_sample"] = event
            pair = "|".join(sorted((contact["geom1"], contact["geom2"])))
            if pair not in first_pairs:
                first_pairs[pair] = event
                events.write(json.dumps({"event": "first_geometric_pair", **event})+"\n")
                events.flush()
            finger = contact["finger_side"]
            if finger is None:
                continue
            if finger not in first_geometric:
                first_geometric[finger] = event
                render("first-contact-"+finger)
            if contact["load_bearing"]:
                loaded.add(finger)
                if contact["region"] == "inward_collision_box_face":
                    inward.add(finger)
                if finger not in first_loaded:
                    first_loaded[finger] = event
                    events.write(json.dumps({"event": "first_loaded_finger", **event})+"\n")
                    events.flush()
        if len(loaded) == 2 and report["first_bilateral_body_force_contact"] is None:
            report["first_bilateral_body_force_contact"] = row
            render("first-bilateral")
        if len(inward) == 2 and report["first_bilateral_inward_box_contact"] is None:
            report["first_bilateral_inward_box_contact"] = row
        row["bilateral_body_force_contact"] = len(loaded) == 2
        for key, previous in peak.items():
            if row[key] > previous:
                peak[key] = row[key]
                report["maximum_"+key+"_sample"] = {k: v for k, v in row.items() if k != "contacts"}
        if persist:
            tail_bilateral.append(row["bilateral_body_force_contact"])
            trace.write(json.dumps(row)+"\n")
            trace.flush()
        return row
    def step_and_sample(tick):
        # Native mj_step leaves contacts, forces and body transforms at the
        # pre-integration evaluation. Pair them with that time and finger qpos;
        # never call extra mj_forward on the live control/solver state.
        evaluation_state.update(time_sec=float(scene.data.time),
            measured_close=scene.get_gripper_close(hand_name), evaluation_phase="pre_integration")
        rig.step_physics()
        clock[0] = float(scene.data.time)
        sample(persist=tick % 40 == 0)

    rig.start()
    render("initial")
    sample(persist=True)
    for tick in range(round(3.*contract.timing.physics_hz)):
        if tick == round(.25*contract.timing.physics_hz):
            phase = "close"
            rig.queue_gripper_close_targets({hand_name: 1.})
        step_and_sample(tick)
    report["closure_end"] = sample(persist=True)
    render("closed")
    if lift:
        if len(tail_bilateral) != 25 or not all(tail_bilateral):
            report["lift_status"] = "not_attempted_without_sustained_bilateral_body_force_contact"
        else:
            phase = "lift"
            ik = RM65CanonicalIK.from_tool_profiles(scene,
                {s: getattr(config, s+"_tool_profile_path") for s in ("left", "right")},
                planner_config=RM65PositionCollisionPlannerConfig(scene_profile="dual_arm_hand_proxy_dmtac_grasp"))
            lift_tcp = scene.body_pose(scene.arm_specs[side].controlled_body) @ tcp_offset
            base_from_world = np.linalg.inv(scene.body_pose(side+"_arm_base_link"))
            next_reference = 0.
            for tick in range(round(4.*contract.timing.physics_hz)):
                elapsed = tick/contract.timing.physics_hz
                if elapsed + 1e-10 >= next_reference:
                    progress = min(1., elapsed/2.)
                    progress = progress**3 * (10.-15.*progress+6.*progress**2)
                    target = lift_tcp.copy()
                    target[2, 3] += .03*progress
                    solved = ik.solve({side: PoseTarget(side=side, pose=base_from_world @ target)})
                    if not solved.success or solved.hold:
                        raise RuntimeError(f"Shared trial-lift IK rejected: {solved.status}: {solved.message}")
                    command = TeleopCommand(timestamp=clock[0])
                    for s in ("left", "right"):
                        command.add_command(EffectorCommand(effector_name=s+"_arm", side=s,
                            command_type="joint_vector", value=np.asarray(solved.qpos_by_side[s])))
                    if not all(r.accepted for r in rig.send_commands(command).values()):
                        raise RuntimeError("Existing rig rejected the trial-lift reference")
                    next_reference += 1./contract.timing.reference_hz
                step_and_sample(tick)
            report["lift_end"] = sample(persist=True)
            report["lift_status"] = "completed_trial_not_grasp_success_assessment"
            render("lifted")
    report["status"] = "completed_terminal_contact_diagnostic"


def run_case(fixture_path, output, *, center_key, orientation_key, lift=False, ik_budget_sec=8.):
    """Retain primary faults and attempt every cleanup, including early failures."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    report = {"center_key": center_key, "orientation_key": orientation_key,
        "fixture_path": str(fixture_path), "cleanup_errors": []}
    resources = []
    primary_error = None

    def register_resource(name, resource):
        resources.append((name, resource))

    try:
        if not np.isfinite(ik_budget_sec) or not 0 < ik_budget_sec <= 30:
            raise ValueError("IK fixture budget must be in (0, 30] seconds")
        _run_case_contents(fixture_path, output, report, register_resource,
            center_key=center_key, orientation_key=orientation_key, lift=lift, ik_budget_sec=ik_budget_sec)
    except BaseException as error:
        primary_error = error
        rejection_prefixes = (
            "MuJoCo physical execution found forbidden hard contact",
            "MuJoCo physical execution entered forbidden hard contact after physics step",
        )
        expected = isinstance(error, RuntimeError) and str(error).startswith(rejection_prefixes)
        report["status"] = ("native_execution_rejected" if expected else
            "incomplete_or_interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "diagnostic_fault")
        report["error"] = {"type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc()}
        if not expected:
            raise
    finally:
        # LIFO cleanup, but a failing renderer/file/rig must not skip the others
        # or replace the original traceback. Cleanup faults never report success.
        for name, resource in reversed(resources):
            try:
                resource.close()
            except BaseException as error:  # noqa: BLE001 - retained below; cleanup faults cannot report success.
                report["cleanup_errors"].append({"resource": name, "type": type(error).__name__,
                    "message": str(error), "traceback": traceback.format_exc()})
        report["cleanup_complete"] = not report["cleanup_errors"]
        if report["cleanup_errors"]:
            report["primary_outcome"] = report.get("status")
            report["status"] = "diagnostic_fault"
        report.setdefault("status", "incomplete_or_interrupted")
        report["elapsed_wall_sec"] = time.perf_counter() - began
        try:
            if (output/"fixture.xml").exists():
                report["fixture_xml_sha256"] = _sha(output/"fixture.xml")
            (output/"result.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        except Exception as error:
            if primary_error is None:
                raise
            print(f"Unable to persist diagnostic report: {type(error).__name__}: {error}", file=sys.stderr)
        if report["cleanup_errors"] and primary_error is None:
            raise RuntimeError("Closure diagnostic cleanup failed; inspect result.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--center", required=True)
    parser.add_argument("--orientation", required=True)
    parser.add_argument("--lift", action="store_true")
    parser.add_argument("--ik-budget-sec", type=float, default=8.)
    args = parser.parse_args()
    if not np.isfinite(args.ik_budget_sec) or not 0 < args.ik_budget_sec <= 30:
        parser.error("IK fixture budget must be in (0, 30] seconds")
    sys.addaudithook(_forbid_network)
    report = run_case(args.fixture, args.output, center_key=args.center,
        orientation_key=args.orientation, lift=args.lift, ik_budget_sec=args.ik_budget_sec)
    print(json.dumps({k: report.get(k) for k in ("status", "center_key", "orientation_key", "error")}))
    return 0 if report["status"] == "completed_terminal_contact_diagnostic" else 2


if __name__ == "__main__":
    raise SystemExit(main())
