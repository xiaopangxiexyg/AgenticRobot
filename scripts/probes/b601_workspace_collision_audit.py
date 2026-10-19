"""Offline geometry audit of B601 acquisition paths; never construct hardware.

MuJoCo uses convex hulls for STL collision. Preserve that conservative result
and independently inspect the original triangle surfaces with installed FCL.
Disagreement is reported, never converted into hardware collision permission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from control_station.b601_torque_console.automatic_calibration import (
    AutomaticCalibrationConfig,
    load_automatic_profile,
)
from control_station.b601_torque_console.workspace_calibration import (
    WorkspaceCalibrationConfig,
)
from robot_motion.rebot_b601_rs.calibration import (
    CalibrationCondition,
    plan_local_sweeps,
    six,
)
from robot_motion.rebot_b601_rs.model import B601_RS_URDF_RELATIVE_PATH
from sim.mujoco.collision_safety import CollisionSafetyOracle
from sim.mujoco.rebot_b601_rs import b601_contact_scene_spec, build_b601_mujoco_scene
from sim.mujoco.urdf import parse_urdf

ZERO = (0.0,) * 6


class WorkspaceGeometryAudit:
    def __init__(self, source_root: Path, *, ground_z_m=None):
        import hppfcl

        self.fcl = hppfcl
        self.urdf = source_root / B601_RS_URDF_RELATIVE_PATH
        self.urdf_sha256 = hashlib.sha256(self.urdf.read_bytes()).hexdigest()
        spec = b601_contact_scene_spec()
        spec = replace(spec, ground=replace(spec.ground,
            enabled=ground_z_m is not None, z_m=-.05 if ground_z_m is None else ground_z_m))
        self.scene = build_b601_mujoco_scene(source_root, initial_arm_qpos=ZERO,
            gravity_mps2=spec.gravity_mps2, scene_spec=spec)
        self.oracle = CollisionSafetyOracle(self.scene.mujoco, self.scene.model)
        loader, parsed = hppfcl.MeshLoader(), parse_urdf(self.urdf)
        self.meshes = {}
        self.mesh_hashes = {}
        for link, body in parsed.links.items():
            for index, collision in enumerate(body.collisions):
                if collision.geometry.kind != "mesh":
                    raise ValueError("triangle cross-check requires the pinned mesh collision model")
                path = (self.urdf.parent.parent / collision.geometry.filename).resolve(strict=True)
                path.relative_to(self.urdf.parent.parent.resolve())
                self.mesh_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                self.meshes[f"{link}_collision_{index}"] = (
                    loader.load(str(path), np.ones(3)), collision.origin)
        mj, model = self.scene.mujoco, self.scene.model
        self.pairs = []
        for g1, g2 in self.oracle.forbidden_geom_pairs():
            names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g) for g in (g1, g2)]
            if all(name in self.meshes for name in names):
                bodies = [int(model.geom_bodyid[g]) for g in (g1, g2)]
                self.pairs.append((names, bodies))

    def check(self, qpos_rad):
        q = six(qpos_rad, "audit qpos")
        scene, fcl = self.scene, self.fcl
        scene.set_arm_qpos("b601_rs", q)
        scene.forward()
        clearance = self.oracle.policy.minimum_clearance_m
        convex = self.oracle.check(scene.data, distance_query_limit_m=clearance)
        transforms = {}
        for names, bodies in self.pairs:
            for name, body in zip(names, bodies):
                if name not in transforms:
                    pose = np.eye(4)
                    pose[:3, :3] = scene.data.xmat[body].reshape(3, 3)
                    pose[:3, 3] = scene.data.xpos[body]
                    pose = pose @ self.meshes[name][1]
                    transforms[name] = fcl.Transform3f(pose[:3, :3], pose[:3, 3])
        minimum, closest, collisions = math.inf, None, []
        for names, _ in self.pairs:
            a, b = names
            distance = float(fcl.distance(self.meshes[a][0], transforms[a],
                self.meshes[b][0], transforms[b], fcl.DistanceRequest(), fcl.DistanceResult()))
            if distance < minimum:
                minimum, closest = distance, list(names)
            if distance <= 0:
                collisions.append(list(names))
        ground_contacts = [asdict(c) for c in scene.contacts()
                           if "ground_collision" in (c.geom1_name, c.geom2_name)]
        return {"qpos_deg": [math.degrees(v) for v in q],
            "convex_check": convex.to_dict(),
            "minimum_triangle_surface_distance_m": minimum,
            "closest_triangle_pair": closest, "triangle_surface_collision_pairs": collisions,
            "ground_contacts": ground_contacts}


def nominal_segments(workspace, config, origin, condition):
    """Consume the same task order and local sweep generator as the owner."""
    q = list(six(origin, "origin"))
    limits = [(-2.8, 2.8)] * 6
    for j, bounds in enumerate((workspace.joint2_acquisition_range_deg,
                                workspace.joint3_acquisition_range_deg), 1):
        limits[j] = tuple(math.radians(v) for v in bounds)
    for task in workspace.tasks(origin):
        if task["kind"] in {"position", "return"}:
            targets = [task["target"]]
        elif task["kind"] == "sweep":
            plan = plan_local_sweeps(condition=condition, joint_name=task["joint"],
                anchor_qpos_rad=q, position_limits_rad=limits,
                excursion_rad=workspace.local_excursion_rad(config), speeds_rad_s=config.sweep_speeds_rad_s,
                max_reference_velocity_rad_s=max(config.sweep_speeds_rad_s), purpose="local_identification")
            targets = [step["target_qpos_rad"][int(task["joint"][-1])-1] for step in plan["steps"]]
        else:
            continue
        j = int(task["joint"][-1]) - 1
        for leg, target in enumerate(targets):
            end = q.copy()
            end[j] = target
            yield {**task, "leg": leg}, tuple(q), tuple(end)
            q = end


def audit_plan(audit, workspace, config, origin, *, max_step_deg=.5, profile_sha256):
    if not math.isfinite(max_step_deg) or not 0 < max_step_deg <= .5:
        raise ValueError("geometry audit sample step must be positive and at most 0.5 degrees")
    # Torque values are explicitly placeholders for a geometry-only plan.
    condition = CalibrationCondition("geometry_only_not_torque_identification", profile_sha256,
        audit.urdf_sha256, 1, (1.0,)*6, ZERO, ZERO, ZERO, .02, ZERO)
    records, cache = [], {}
    collision_samples = narrow_samples = convex_rejections = 0
    minimum = math.inf
    for task, start, end in nominal_segments(workspace, config, origin, condition):
        n = max(1, math.ceil(max(abs(a-b) for a, b in zip(start, end))/math.radians(max_step_deg)))
        first_collision = first_narrow = None
        for k in range(n+1):
            q = tuple(a + k/n*(b-a) for a, b in zip(start, end))
            key = tuple(round(v, 12) for v in q)
            if key not in cache:
                cache[key] = audit.check(q)
            result = cache[key]
            distance = result["minimum_triangle_surface_distance_m"]
            minimum = min(minimum, distance)
            collision = bool(result["triangle_surface_collision_pairs"] or result["ground_contacts"])
            narrow = distance < audit.oracle.policy.minimum_clearance_m
            collision_samples += collision
            narrow_samples += narrow
            convex_rejections += not result["convex_check"]["safe"]
            if collision and first_collision is None:
                first_collision = result
            if narrow and first_narrow is None:
                first_narrow = result
        records.append({"task": task, "sample_intervals": n,
            "first_surface_or_ground_collision": first_collision,
            "first_below_preventive_margin": first_narrow})
    return {"status": "COLLISIONS_DETECTED" if collision_samples else
            "CLEARANCE_REVIEW_REQUIRED" if narrow_samples or convex_rejections else "SAMPLED_GEOMETRY_CLEAR",
        "hardware_authorized": False, "workspace": asdict(workspace), "origin_qpos_rad": origin,
        "segments": records, "unique_configurations": len(cache),
        "sample_step_deg": max_step_deg, "surface_or_ground_collision_samples": collision_samples,
        "below_preventive_margin_samples": narrow_samples,
        "convex_clearance_rejection_samples": convex_rejections,
        "minimum_triangle_surface_distance_m": minimum,
        "origin_check": audit.check(origin),
        "limitations": ["kinematic sampled nominal paths, not friction/dynamic tracking acceptance",
            "triangle surfaces do not prove solid-volume containment or dimensional accuracy",
            "tracking error and support-axis drift are not covered by nominal path samples",
            "no surveyed environment, cables, support stand or changed end effector is modeled",
            "convex and triangle disagreement is retained, never treated as hardware permission"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--origin-deg", nargs=6, type=float, default=ZERO)
    parser.add_argument("--ground-z-m", type=float)
    parser.add_argument("--compare-previous-grid", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    values = load_automatic_profile(args.profile)
    workspace = WorkspaceCalibrationConfig(**values.pop("workspace"))
    values.pop("cycle")
    values.pop("motion_profile")
    config = AutomaticCalibrationConfig(**values)
    origin = tuple(math.radians(v) for v in args.origin_deg)
    audit = WorkspaceGeometryAudit(args.source_root, ground_z_m=args.ground_z_m)
    digest = hashlib.sha256(args.profile.read_bytes()).hexdigest()
    result = {"can_opened": False, "motor_command_sent": False,
        "urdf": str(audit.urdf), "urdf_sha256": audit.urdf_sha256,
        "mujoco_version": audit.scene.mujoco.__version__,
        "fcl_version": audit.fcl.__version__, "mesh_sha256": audit.mesh_hashes,
        "forbidden_pair_count": audit.oracle.forbidden_pair_count,
        "triangle_pair_count": len(audit.pairs), "calibration_settings": asdict(config),
        "gripper_assumption": "pinned_open_auxiliary_joints_not_measured_j7_or_final_tool",
        "ground_z_m": args.ground_z_m, "profile_sha256": digest,
        "candidate": audit_plan(audit, workspace, config, origin, profile_sha256=digest)}
    if args.compare_previous_grid:
        previous = replace(workspace, joint2_centres_deg=(5, 60, 115), joint3_centres_deg=(5, 45, 85),
            joint2_acquisition_range_deg=(0, 120), joint3_acquisition_range_deg=(0, 90),
            positioning_joint_order=("joint2", "joint3"))
        result["previous_grid"] = audit_plan(audit, previous, config, origin, profile_sha256=digest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
    print(json.dumps({"candidate": result["candidate"]["status"],
        "previous_grid": result.get("previous_grid", {}).get("status"), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
