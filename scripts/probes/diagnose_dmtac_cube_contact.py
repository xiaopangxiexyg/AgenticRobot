"""Offline CAD-hand contact fixture, not an RM65/Agent or hardware acceptance.

Reuse the scene converter, CAD linkage and simulation-only actuator profile.
Only the free flange follows a mocap driver; the cube remains a free rigid body.
Contact friction is explicit; no object weld, latch, or qpos replay is used.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from project_paths import REPO_ROOT
from robot_motion.pose import pose_delta_metrics
from scripts.probes.compare_rm65_dmtac_sweep import load_sensor_sweeps
from sim.contracts import SimulationContract
from sim.contracts.scene import FreeBodyObjectSpec
from sim.mujoco.actuation import (
    cad_dmtac_cube_contact_gripper_profile,
    point_grasp_cad_bimanual_dmtac_gripper_profile,
)
from sim.mujoco.actuation.gripper_executor import MujocoDmtacGripperExecutor
from sim.mujoco.config import GripperSpec
from sim.mujoco.mjcf import MJCFConversion, convert_urdf_to_mjcf_xml
from sim.mujoco.scene import MujocoScene, SceneLoadResult
from sim.mujoco.transforms import format_float_list
from sim.mujoco.urdf import parse_urdf

CAD_URDF = Path("assets/realman_dual_arm_inspire_gripper_dmtacm/"
    "realman_dual_arm_inspire_gripper_dmtacm.urdf")


def contact_profile(contact_model: str, sliding_friction: float):
    if contact_model == "baseline":
        return point_grasp_cad_bimanual_dmtac_gripper_profile()
    if contact_model == "constraint_candidate":
        return cad_dmtac_cube_contact_gripper_profile(sliding_friction=sliding_friction)
    raise ValueError("Unknown contact model")


def aperture_report(urdf: Path, side: str, edge_m: float) -> dict:
    if not np.isfinite(edge_m) or edge_m <= 0:
        raise ValueError("Cube edge must be finite and positive")
    sweeps = load_sensor_sweeps(urdf, side)
    centers = [s.corners(0).mean(axis=0) for s in sweeps]
    axis = centers[1] - centers[0]
    axis /= np.linalg.norm(axis)
    samples = []
    for opening in np.linspace(*sweeps[0].joint_range_rad, 101):
        intervals = [s.corners(opening) @ axis for s in sweeps]
        samples.append({"q_rad": float(opening), "support_gap_m": float(
            intervals[1].min() - intervals[0].max())})
    return {"side": side, "cube_edge_m": edge_m,
        "method": "sensor_collision_box_support_intervals_along_center_axis",
        "closing_axis_flange": axis.tolist(),
        "sensor_midpoint_flange_m": np.mean(centers, axis=0).tolist(),
        "open_gap_m": samples[0]["support_gap_m"],
        "closed_gap_m": samples[-1]["support_gap_m"],
        "nominal_total_margin_m": samples[0]["support_gap_m"] - edge_m,
        "samples": samples}


def build_fixture(*, urdf: Path, side: str, orientation: str,
                  edge_m: float, mass_kg: float,
                  object_approach_offset_m: float = 0.,
                  contact_model: str = "baseline", sliding_friction: float = 1.2) -> tuple[str, dict]:
    if side not in ("left", "right") or orientation not in ("horizontal", "top_down"):
        raise ValueError("Unknown hand side or approach orientation")
    if not np.isfinite(object_approach_offset_m):
        raise ValueError("Object approach offset must be finite")
    if contact_model not in ("baseline", "constraint_candidate"):
        raise ValueError("Unknown contact model")
    # FreeBodyObjectSpec validates positive finite size/mass before building.
    cube = FreeBodyObjectSpec(name="debug_cube", size_m=(edge_m,) * 3,
        mass_kg=mass_kg, initial_position_m=(0, 0, edge_m / 2), rgba=(.08, .24, .9, 1))
    report = aperture_report(urdf, side, edge_m)
    profile = contact_profile(contact_model, sliding_friction)
    contract = SimulationContract.load(REPO_ROOT / "sim/configs/rm65_bimanual_physics_v1.json")
    scene = replace(contract.scene, urdf_path=str(urdf), objects=(cube,),
        ground=replace(contract.scene.ground, z_m=0))
    root = ET.fromstring(convert_urdf_to_mjcf_xml(parse_urdf(urdf), camera_mounts=(),
        timestep=contract.timing.physics_timestep_sec, gravity=scene.gravity_mps2,
        scene_spec=scene, gripper_actuation_profile=profile))
    world = root.find("worldbody")
    hand = root.find(f".//body[@name='{side}_arm_link6']")
    if hand is None or world is None:
        raise ValueError("CAD conversion has no requested flange/world body")
    # Isolate the real CAD subtree, preserving its geoms, inertias, exclusions
    # and six position actuators. This removes the arm, not hand self-contact.
    for joint in list(hand.findall("joint")):
        hand.remove(joint)
    ET.SubElement(hand, "freejoint", {"name": "fixture_flange_freejoint"})
    rotation = (Rotation.from_euler("y", np.pi / 2).as_matrix()
        if orientation == "horizontal" else np.diag((1., -1., -1.)))
    midpoint = np.asarray(report["sensor_midpoint_flange_m"])
    hand_center = np.array((0., 0., edge_m / 2))
    # Top-down fixture keeps the lowest sensor corner 2 mm above the floor;
    # report this offset instead of silently intersecting the table.
    sweeps = load_sensor_sweeps(urdf, side)
    corners = np.concatenate([s.corners(0) for s in sweeps])
    relative = (corners - midpoint) @ rotation.T
    hand_center[2] = max(hand_center[2], .002 - float(relative[:, 2].min()))
    hand_center -= rotation[:, 2] * object_approach_offset_m
    position = hand_center - rotation @ midpoint
    hand.set("pos", format_float_list(position))
    xyzw = Rotation.from_matrix(rotation).as_quat()
    hand.set("quat", format_float_list(xyzw[[3, 0, 1, 2]]))
    for child in list(world):
        if child.tag == "body" and child.get("name") != "debug_cube":
            world.remove(child)
    world.append(hand)
    # Directly teleporting a mocap-rooted hand gives it no physical base
    # velocity, suppressing tangential contact response during a lift. Reuse
    # the free-hand/driver weld arrangement from the object-transport fixture.
    ET.SubElement(world, "body", {"name": "fixture_driver", "mocap": "true",
        "pos": hand.get("pos"), "quat": hand.get("quat")})
    ET.SubElement(root.find("equality"), "weld", {"name": "fixture_flange_tracking",
        "body1": "fixture_driver", "body2": side + "_arm_link6", "solref": ".004 1"})
    bodies = {b.get("name") for b in world.iter("body")}
    joints = {j.get("name") for j in world.iter("joint")}
    geoms = {g.get("name") for g in world.iter("geom")}
    for section in ("contact", "equality", "actuator"):
        parent = root.find(section)
        if parent is None:
            continue
        for item in list(parent):
            references = [(k, v) for k, v in item.attrib.items()
                if k in ("body1", "body2", "joint", "joint1", "joint2", "geom1", "geom2")]
            if any(v not in (bodies if k.startswith("body") else geoms if k.startswith("geom") else joints)
                   for k, v in references):
                parent.remove(item)
    report.update({"orientation": orientation, "mass_kg": mass_kg,
        "mass_source": "simulation_assumption_not_measured",
        "profile": profile.to_metadata(), "cube_contact": cube.contact.to_dict(),
        "hand_center_world_m": hand_center.tolist(),
        "object_approach_offset_m": object_approach_offset_m,
        "contact_model": contact_model,
        "flange_world_m": position.tolist(), "flange_rotation": rotation.tolist(),
        "fixture": "isolated_CAD_free_hand_welded_to_driver_free_cube",
        "hardware_connected": False, "arm_path_validated": False})
    return ET.tostring(root, encoding="unicode"), report


def classify_trial(*, initial_overlap_m: float, maximum_penetration_m: float,
                   maximum_mimic_error_rad: float, maximum_limit_excess_rad: float,
                   rise_m: float, final_normal_n: float,
                   relative_slip_m: float, relative_rotation_deg: float,
                   flange_tracking_error_m: float,
                   hand_ground_penetration_m: float) -> str:
    """Diagnostic validity precedes lift success; thresholds are not safety policy."""
    values = (initial_overlap_m, maximum_penetration_m, maximum_mimic_error_rad,
        maximum_limit_excess_rad, rise_m, final_normal_n, relative_slip_m,
        relative_rotation_deg, flange_tracking_error_m, hand_ground_penetration_m)
    if not np.isfinite(values).all():
        raise ValueError("Trial metrics must be finite")
    if initial_overlap_m > .0001:
        return "invalid_initial_placement"
    if (maximum_penetration_m > .002 or maximum_mimic_error_rad > .01
            or maximum_limit_excess_rad > .01 or flange_tracking_error_m > .002
            or hand_ground_penetration_m > .002):
        return "invalid_contact_dynamics"
    if (rise_m >= .025 and final_normal_n > .01
            and relative_slip_m <= .003 and relative_rotation_deg <= 2.):
        return "lifted_in_contact_fixture"
    return "grasp_not_retained"


def run_case(output: Path, *, side: str, orientation: str,
             edge_m: float = .070, mass_kg: float = .08,
             object_approach_offset_m: float = 0., contact_model: str = "baseline",
             sliding_friction: float = 1.2) -> dict:
    import mujoco

    output.mkdir(parents=True, exist_ok=False)
    xml, report = build_fixture(urdf=(REPO_ROOT / CAD_URDF).resolve(strict=True),
        side=side, orientation=orientation, edge_m=edge_m, mass_kg=mass_kg,
        object_approach_offset_m=object_approach_offset_m, contact_model=contact_model,
        sliding_friction=sliding_friction)
    (output / "fixture.xml").write_text(xml, encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(output / "fixture.xml"))
    profile = contact_profile(contact_model, sliding_friction)
    hand_name = f"{side}_eg2_4b2_cad"
    response = profile.position_response_profile
    available_geoms = {model.geom(i).name for i in range(model.ngeom)}
    profile = replace(profile, contact_tuning=replace(profile.contact_tuning,
        geom_pairs=tuple(pair for pair in profile.contact_tuning.geom_pairs
            if all(name in available_geoms for name in pair))),
        position_response_profile=replace(response,
        parameters_by_side={side: response.parameters_by_side[side]}),
        **{field: {hand_name: getattr(profile, field)[hand_name]}
        for field in ("joint_names_by_gripper", "side_by_gripper", "open_qpos_by_gripper",
            "closed_qpos_by_gripper", "command_to_actuator_delay_sec_by_gripper",
            "master_joint_by_gripper")})
    joints = tuple(profile.joint_names_by_gripper[hand_name])
    spec = GripperSpec(name=hand_name, joint_names=joints, side=side,
        open_qpos=tuple(profile.open_qpos_by_gripper[hand_name]),
        closed_qpos=tuple(profile.closed_qpos_by_gripper[hand_name]))
    scene = MujocoScene(mujoco=mujoco, model=model, data=mujoco.MjData(model),
        load_result=SceneLoadResult(conversion=MJCFConversion(
            urdf_path=(REPO_ROOT / CAD_URDF).resolve(strict=True),
            mjcf_path=output / "fixture.xml", camera_mounts=(), joint_names=joints,
            body_names=tuple(model.body(i).name for i in range(1, model.nbody))),
            cameras=(), arms={}, grippers={hand_name: spec}))
    data = scene.data
    mujoco.mj_forward(model, data)
    sensor_ids = {finger: int(model.geom(f"{side}_gripper_{finger}_dm_tac_w2m_sensor_collision_0").id)
        for finger in ("l", "r")}
    cube_id = int(model.geom("debug_cube_collision").id)
    report["actual_sensor_friction"] = {f: model.geom_friction[i].tolist() for f, i in sensor_ids.items()}
    report["explicit_pair_friction"] = model.pair_friction.tolist()
    report["mujoco_version"] = mujoco.__version__
    cube_body = model.body("debug_cube").id
    root_body = model.body(side + "_arm_link6").id
    mocap_id = model.body_mocapid[model.body("fixture_driver").id]
    initial_flange = data.mocap_pos[mocap_id].copy()
    master = model.joint(f"{side}_eg2_4b2_cad_r_1_joint").qposadr[0]
    executor = MujocoDmtacGripperExecutor(scene, profile)
    executor.reset({hand_name: 0.})
    joint_ids = [model.joint(name).id for name in profile.joint_names_by_gripper[hand_name]]
    qpos_ids = model.jnt_qposadr[joint_ids]
    initial_contacts = [{"other_geom": model.geom(int(c.geom2 if c.geom1 == cube_id else c.geom1)).name,
        "distance_m": float(c.dist)} for c in data.contact
        if cube_id in (c.geom1, c.geom2)
        and c.geom1 != model.geom("ground_collision").id
        and c.geom2 != model.geom("ground_collision").id]
    initial_overlap = max([0.] + [-c["distance_m"] for c in initial_contacts])
    report["initial_cube_hand_contacts"] = initial_contacts
    report["execution"] = executor.execution_metadata
    (output / "configuration.json").write_text(json.dumps(report, indent=2) + "\n",
        encoding="utf-8")
    maximum_penetration = maximum_mimic_error = maximum_limit_excess = 0.
    maximum_tracking_error = maximum_ground_penetration = 0.
    sent_close = False
    bilateral_ticks = 0
    baseline_z = None
    initial_relation = None
    final = None
    # Flush each 10 ms diagnostic record to preserve partial evidence on Ctrl+C.
    with (output / "trace.jsonl").open("x", encoding="utf-8") as log:
        for tick in range(round(6. / model.opt.timestep)):
            time = float(data.time)
            if time >= 1. and not sent_close:
                executor.queue_close_targets({hand_name: 1.})
                sent_close = True
            control = executor.before_physics_step(physics_time_sec=time,
                timestep_sec=float(model.opt.timestep))
            lift_progress = float(np.clip((time - 3.) / 2., 0., 1.))
            lift = .03 * (10 * lift_progress**3 - 15 * lift_progress**4 + 6 * lift_progress**5)
            data.mocap_pos[mocap_id] = initial_flange + (0., 0., lift)
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            if not np.isfinite(data.qpos).all() or any(w.number for w in data.warning):
                raise RuntimeError("Non-finite state or MuJoCo numerical warning; inspect fixture")
            contacts = []
            normal = {"l": 0., "r": 0.}
            floor_id = int(model.geom("ground_collision").id)
            for index in range(data.ncon):
                c = data.contact[index]
                if floor_id in (c.geom1, c.geom2) and cube_id not in (c.geom1, c.geom2):
                    maximum_ground_penetration = max(maximum_ground_penetration, -float(c.dist))
                if cube_id not in (c.geom1, c.geom2):
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, index, force)
                other = c.geom2 if c.geom1 == cube_id else c.geom1
                for finger, geom_id in sensor_ids.items():
                    if other == geom_id:
                        normal[finger] += max(0., float(force[0]))
                contacts.append({"other_geom": model.geom(int(other)).name,
                    "position_m": c.pos.tolist(), "distance_m": float(c.dist),
                    "normal_force_n": float(force[0]),
                    "contact_normal_world": c.frame[:3].tolist(),
                    "tangent_force_n": float(np.linalg.norm(force[1:3]))})
                if other != model.geom("ground_collision").id:
                    maximum_penetration = max(maximum_penetration, -float(c.dist))
            joint_qpos = data.qpos[qpos_ids]
            maximum_mimic_error = max(maximum_mimic_error,
                float(np.max(np.abs(joint_qpos - data.qpos[master]))))
            maximum_limit_excess = max(maximum_limit_excess, float(np.max(np.maximum(
                model.jnt_range[joint_ids, 0] - joint_qpos,
                joint_qpos - model.jnt_range[joint_ids, 1]))))
            maximum_tracking_error = max(maximum_tracking_error,
                float(np.linalg.norm(data.xpos[root_body] - data.mocap_pos[mocap_id])))
            bilateral_ticks += int(min(normal.values()) > .01)
            if baseline_z is None and time >= 2.99:
                baseline_z = float(data.xpos[cube_body, 2])
                initial_relation = np.linalg.inv(scene.body_pose(side + "_arm_link6")) @ scene.body_pose("debug_cube")
            final = {"time_sec": float(data.time), "phase": "open" if time < 1 else
                "close" if time < 3 else "lift" if time < 5 else "hold",
                "commanded_close": float(sent_close),
                "servo_close": control.actuator_target_close_by_gripper[hand_name],
                "measured_master_rad": float(data.qpos[master]),
                "joint_qpos_rad": joint_qpos.tolist(),
                "qpos": data.qpos.tolist(),
                "flange_xyz_m": data.mocap_pos[mocap_id].tolist(),
                "actual_flange_xyz_m": data.xpos[root_body].tolist(),
                "actual_flange_qvel": data.qvel[model.joint("fixture_flange_freejoint").dofadr[0]:
                    model.joint("fixture_flange_freejoint").dofadr[0] + 6].tolist(),
                "cube_xyz_m": data.xpos[cube_body].tolist(), "flange_lift_m": lift,
                "normal_force_by_finger_n": normal, "cube_contacts": contacts}
            if tick % 10 == 0:
                log.write(json.dumps(final) + "\n")
                log.flush()
    rise = float(final["cube_xyz_m"][2] - baseline_z)
    final_relation = np.linalg.inv(scene.body_pose(side + "_arm_link6")) @ scene.body_pose("debug_cube")
    slip, relative_rotation = pose_delta_metrics(initial_relation, final_relation)
    status = classify_trial(initial_overlap_m=initial_overlap,
        maximum_penetration_m=maximum_penetration,
        maximum_mimic_error_rad=maximum_mimic_error,
        maximum_limit_excess_rad=maximum_limit_excess,
        rise_m=rise, final_normal_n=min(final["normal_force_by_finger_n"].values()),
        relative_slip_m=slip, relative_rotation_deg=relative_rotation,
        flange_tracking_error_m=maximum_tracking_error,
        hand_ground_penetration_m=maximum_ground_penetration)
    report.update({"final": final, "cube_rise_m": rise,
        "initial_cube_hand_contacts": initial_contacts,
        "maximum_cube_hand_penetration_m": maximum_penetration,
        "maximum_mimic_error_rad": maximum_mimic_error,
        "maximum_joint_limit_excess_rad": maximum_limit_excess,
        "maximum_flange_tracking_error_m": maximum_tracking_error,
        "maximum_hand_ground_penetration_m": maximum_ground_penetration,
        "relative_slip_m": slip, "relative_rotation_deg": relative_rotation,
        "diagnostic_status": status, "execution": executor.execution_metadata,
        "bilateral_contact_duration_sec": bilateral_ticks * model.opt.timestep,
        "trial_lift_passed": status == "lifted_in_contact_fixture",
        "object_weld_or_latch": False})
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cube-edge-m", type=float, default=.070)
    parser.add_argument("--cube-mass-kg", type=float, default=.08,
        help="Simulation assumption; real cube mass is not yet measured")
    parser.add_argument("--object-approach-offset-m", type=float, default=0.,
        help="Positive moves the cube toward the fingertips relative to the sensor midpoint")
    parser.add_argument("--contact-model", choices=("baseline", "constraint_candidate"),
        default="baseline", help="Explicit simulation-only contact model; actuator force limits stay unchanged")
    parser.add_argument("--sliding-friction", type=float, default=1.2,
        help="Explicit sensor/object sliding friction for the candidate; baseline stays unchanged")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for side in ("left", "right"):
        for orientation in ("horizontal", "top_down"):
            result = run_case(args.output / f"{side}-{orientation}", side=side,
                orientation=orientation, edge_m=args.cube_edge_m, mass_kg=args.cube_mass_kg,
                object_approach_offset_m=args.object_approach_offset_m,
                contact_model=args.contact_model, sliding_friction=args.sliding_friction)
            results.append(result)
            print(json.dumps({k: result[k] for k in ("side", "orientation", "open_gap_m",
                "cube_rise_m", "diagnostic_status", "trial_lift_passed")}), flush=True)
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0 if all(r["trial_lift_passed"] for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
