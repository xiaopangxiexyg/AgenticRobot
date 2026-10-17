"""Offline DMTAC envelope comparison, not a production collision profile.

The reviewed 4B2 linkage counter-rotates its terminal joint, so each sensor
box translates along a circular arc without changing orientation. The convex
hull of arc samples is inflated by the exact arc/chord sagitta bound. This
contains every intermediate opening, not just the sampled configurations.
Only a cloned diagnostic model is modified; assets and live models are read-only.
"""
from __future__ import annotations

import argparse
import itertools
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


_SIGNS = np.asarray(tuple(itertools.product((-1., 1.), repeat=3)))


def _origin(element: ET.Element | None) -> np.ndarray:
    matrix = np.eye(4)
    if element is not None:
        matrix[:3, :3] = Rotation.from_euler("xyz", np.fromstring(
            element.get("rpy", "0 0 0"), sep=" ")).as_matrix()
        matrix[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
    return matrix


@dataclass(frozen=True)
class CircularBoxSweep:
    sensor_link: str
    primary_joint: str
    joint_range_rad: tuple[float, float]
    corners_at_zero: np.ndarray
    axis: np.ndarray
    orbit_vector: np.ndarray

    @property
    def radius_m(self) -> float:
        return float(np.linalg.norm(np.cross(self.axis, self.orbit_vector)))

    def corners(self, q: float) -> np.ndarray:
        rotation = Rotation.from_rotvec(self.axis * q).as_matrix()
        return self.corners_at_zero + rotation @ self.orbit_vector - self.orbit_vector


def load_sensor_sweeps(cad_urdf: Path, side: str) -> tuple[CircularBoxSweep, ...]:
    """Require the cancellation topology; never infer it from FK samples."""
    root = ET.parse(cad_urdf).getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    parents = {j.find("child").get("link"): j for j in joints.values()}
    sweeps = []
    for finger in ("l", "r"):
        link_name = f"{side}_gripper_{finger}_dm_tac_w2m_sensor"
        link = root.find(f"link[@name='{link_name}']")
        if link is None:
            raise ValueError(f"Missing DMTAC sensor: {link_name}")
        chain, cursor = [], link_name
        visited = set()
        while cursor != f"{side}_arm_link6":
            if cursor in visited or cursor not in parents:
                raise ValueError(f"Sensor is not connected to its arm flange: {link_name}")
            visited.add(cursor)
            joint = parents[cursor]
            chain.append(joint)
            cursor = joint.find("parent").get("link")
        chain.reverse()
        moving = [(i, j) for i, j in enumerate(chain) if j.get("type") != "fixed"]
        if len(moving) != 2 or moving[1][0] != moving[0][0] + 1:
            raise ValueError("Sensor sweep requires two consecutive counter-rotating joints")
        (first_index, first), (_, second) = moving
        primary = f"{side}_eg2_4b2_cad_r_1_joint"
        bounds = tuple(float(joints[primary].find("limit").get(k)) for k in ("lower", "upper"))
        if not np.isfinite(bounds).all() or not 0 <= bounds[0] < bounds[1] <= np.pi:
            raise ValueError("Sensor sweep requires a finite opening interval no longer than pi")
        axes = []
        for joint in (first, second):
            mimic = joint.find("mimic")
            if joint.get("type") != "revolute" or (joint.get("name") != primary and (
                mimic is None or mimic.get("joint") != primary
                or float(mimic.get("multiplier", "1")) != 1
                or float(mimic.get("offset", "0")) != 0
            )):
                raise ValueError("Counter-rotating joints must share the primary opening")
            if tuple(float(joint.find("limit").get(k)) for k in ("lower", "upper")) != bounds:
                raise ValueError("Mimic joint limits disagree with the primary opening")
            axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) <= 0:
                raise ValueError("Invalid sensor joint axis")
            axes.append(axis / np.linalg.norm(axis))
        second_origin = _origin(second.find("origin"))
        if not np.allclose(second_origin[:3, :3], np.eye(3), atol=1e-12, rtol=0) or not np.allclose(
            axes[0], -axes[1], atol=1e-12, rtol=0
        ):
            raise ValueError("Terminal orientation does not cancel continuously")
        transform, prefix = np.eye(4), None
        for index, joint in enumerate(chain):
            transform = transform @ _origin(joint.find("origin"))
            if index == first_index:
                prefix = transform.copy()
        collisions = link.findall("collision")
        if len(collisions) != 1 or collisions[0].find("geometry/box") is None:
            raise ValueError("DMTAC sweep requires one reviewed collision box per sensor")
        box = collisions[0]
        size = np.fromstring(box.find("geometry/box").get("size"), sep=" ")
        if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
            raise ValueError("Invalid sensor collision box")
        box_pose = transform @ _origin(box.find("origin"))
        corners = (_SIGNS * size / 2) @ box_pose[:3, :3].T + box_pose[:3, 3]
        sweeps.append(CircularBoxSweep(link_name, primary, bounds, corners,
            prefix[:3, :3] @ axes[0], prefix[:3, :3] @ second_origin[:3, 3]))
    return tuple(sweeps)


def conservative_sweep_hull(sweeps: tuple[CircularBoxSweep, ...], *, segments: int = 16,
                            padding_m: float = .0001) -> tuple[ConvexHull, dict]:
    if isinstance(segments, bool) or not isinstance(segments, int) or not 1 <= segments <= 256:
        raise ValueError("segments must be an integer in [1, 256]")
    if not np.isfinite(padding_m) or padding_m < 0:
        raise ValueError("padding_m must be finite and nonnegative")
    if not sweeps:
        raise ValueError("Sensor sweeps must not be empty")
    clouds, bounds = [], []
    for sweep in sweeps:
        lo, hi = sweep.joint_range_rad
        # Any arc point lies within sagitta of its chord, a subset of the
        # sample hull. An XYZ cube of this radius contains the Euclidean ball.
        sagitta = sweep.radius_m * (1 - np.cos((hi - lo) / segments / 2))
        samples = np.concatenate([sweep.corners(q) for q in np.linspace(lo, hi, segments + 1)])
        clouds.append((samples[:, None, :] + _SIGNS[None, :, :] * (sagitta + padding_m)).reshape(-1, 3))
        bounds.append({"sensor_link": sweep.sensor_link, "joint_range_rad": [lo, hi],
            "radius_m": sweep.radius_m, "arc_chord_bound_m": float(sagitta)})
    hull = ConvexHull(np.concatenate(clouds))
    return hull, {"method": "counter_rotation_translation_arc_chord_bound",
        "segments": segments, "padding_m": padding_m, "sensors": bounds,
        "volume_m3": float(hull.volume), "vertex_count": len(hull.vertices)}


def _native_hull(hull, hppfcl):
    points = hppfcl.StdVec_Vec3s()
    for point in hull.points[hull.vertices]:
        points.append(point)
    return hppfcl.Convex.convexHull(points, True, "Qt")


def validate_mesh_boxes(cad_urdf: Path, *, padding_m: float) -> dict:
    """Check reviewed sensor boxes against their visible mesh in link coordinates."""
    import trimesh

    root = ET.parse(cad_urdf).getroot()
    checks = {}
    for side, finger in itertools.product(("left", "right"), ("l", "r")):
        name = f"{side}_gripper_{finger}_dm_tac_w2m_sensor"
        link = root.find(f"link[@name='{name}']")
        visual = link.find("visual")
        mesh_spec = visual.find("geometry/mesh")
        mesh = trimesh.load_mesh(cad_urdf.resolve().parent / mesh_spec.get("filename"))
        scale = np.fromstring(mesh_spec.get("scale", "1 1 1"), sep=" ")
        transform = np.linalg.inv(_origin(link.find("collision/origin"))) @ _origin(visual.find("origin"))
        vertices = (np.asarray(mesh.vertices) * scale) @ transform[:3, :3].T + transform[:3, 3]
        half = np.fromstring(link.find("collision/geometry/box").get("size"), sep=" ") / 2
        excess = float(np.max(np.abs(vertices) - half))
        if not np.isfinite(vertices).all() or excess > padding_m:
            raise ValueError(f"Sensor mesh exceeds its padded collision box: {name}: {excess}")
        checks[name] = {"mesh_vertices": len(vertices), "maximum_box_excess_m": excess}
    return checks


def validate_sweep_fk(cad_urdf: Path, sweeps, hull, pin) -> dict:
    """Compare the analytical arc with independently parsed native CAD FK."""
    model, geometry, _ = pin.buildModelsFromUrdf(str(cad_urdf.resolve()),
        package_dirs=[str(cad_urdf.resolve().parent)])
    ids = {str(g.name): i for i, g in enumerate(geometry.geometryObjects)}
    data, geometry_data = model.createData(), pin.GeometryData(geometry)
    q = pin.neutral(model)
    maximum_delta, maximum_violation = 0., -np.inf
    for opening in np.linspace(*sweeps[0].joint_range_rad, 101):
        for j, name in enumerate(model.names):
            if "_eg2_4b2_cad_" in str(name) and model.joints[j].nq:
                q[model.joints[j].idx_q] = opening
        pin.forwardKinematics(model, data, q)
        pin.updateGeometryPlacements(model, data, geometry, geometry_data)
        for sweep in sweeps:
            side = sweep.sensor_link.split("_", 1)[0]
            flange = data.oMi[model.getJointId(side + "_arm_joint6")]
            index = ids[sweep.sensor_link + "_0"]
            box = geometry.geometryObjects[index].geometry
            relative = flange.inverse() * geometry_data.oMg[index]
            corners = (_SIGNS * np.asarray(box.halfSide)) @ relative.rotation.T + relative.translation
            maximum_delta = max(maximum_delta, float(np.max(np.abs(corners-sweep.corners(opening)))))
            maximum_violation = max(maximum_violation, float(np.max(
                corners @ hull.equations[:, :3].T + hull.equations[:, 3])))
    if maximum_delta > 1e-8 or maximum_violation > 1e-9:
        raise ValueError(f"Analytical/native sensor FK disagree: {maximum_delta}, {maximum_violation}")
    return {"samples": 101, "maximum_corner_delta_m": maximum_delta,
        "maximum_hull_violation_m": maximum_violation}


def compare_cad_in_same_scene(cad_urdf, proxy_model, baseline, cases, pin,
                              *, hand_opening_by_side=None) -> tuple[dict, list]:
    """Use identical fixed scene shapes/poses; different tables invalidate a comparison."""
    model, geometry, _ = pin.buildModelsFromUrdf(str(cad_urdf.resolve()),
        package_dirs=[str(cad_urdf.resolve().parent)])
    ids = {str(g.name): i for i, g in enumerate(geometry.geometryObjects)}
    fixed_alignment = {}
    for source in baseline.geometryObjects:
        if str(source.name) not in ("table_0", "camera_pole_0", "left_camera_pole_0", "right_camera_pole_0"):
            continue
        target = geometry.geometryObjects[ids[str(source.name)]]
        if source.parentJoint != 0 or target.parentJoint != 0:
            raise ValueError("Diagnostic scene alignment requires world-fixed geometry")
        fixed_alignment[str(source.name)] = {
            "cad_center_xyz_m": target.placement.translation.tolist(),
            "proxy_center_xyz_m": source.placement.translation.tolist(),
            "cad_size_xyz_m": (2*np.asarray(target.geometry.halfSide)).tolist(),
            "proxy_size_xyz_m": (2*np.asarray(source.geometry.halfSide)).tolist(),
            "center_difference_m": float(np.linalg.norm(target.placement.translation-source.placement.translation)),
        }
        geometry.geometryObjects[ids[str(source.name)]] = pin.GeometryObject(
            str(source.name), 0, 0, source.placement.copy(), deepcopy(source.geometry))
    geometry.removeAllCollisionPairs()
    sensors = {s: [ids[f"{s}_gripper_{f}_dm_tac_w2m_sensor_0"] for f in ("l", "r")]
        for s in ("left", "right")}
    for first in (*sensors["left"], *sensors["right"]):
        geometry.addCollisionPair(pin.CollisionPair(first, ids["table_0"]))
    data, geometry_data = model.createData(), pin.GeometryData(geometry)
    for request in geometry_data.distanceRequests:
        request.enable_signed_distance = True
    proxy_data = proxy_model.createData()
    ranges = []
    for side in ("left", "right"):
        primary = model.joints[model.getJointId(f"{side}_eg2_4b2_cad_r_1_joint")]
        ranges.append((float(model.lowerPositionLimit[primary.idx_q]),
            float(model.upperPositionLimit[primary.idx_q])))
    if ranges[0] != ranges[1]:
        raise ValueError("Diagnostic requires the reviewed symmetric opening range")
    hand_geometry = hand_data = None
    if hand_opening_by_side is not None:
        if (set(hand_opening_by_side) != {"left", "right"}
                or any(not np.isfinite(value) or not ranges[0][0] <= value <= ranges[0][1]
                    for value in hand_opening_by_side.values())):
            raise ValueError("CAD hand comparison requires explicit valid left/right openings")
        hand_geometry = deepcopy(geometry)
        hand_geometry.removeAllCollisionPairs()
        by_side = {side: [i for i, g in enumerate(hand_geometry.geometryObjects)
            if str(g.name).startswith((side+"_eg2_4b2_cad_", side+"_gripper_",
                side+"_realman_to_inspire_gripper_flange_"))] for side in ("left", "right")}
        if not all(by_side.values()):
            raise ValueError("Reviewed CAD lacks complete left/right hand collision geometry")
        for left, right in itertools.product(by_side["left"], by_side["right"]):
            hand_geometry.addCollisionPair(pin.CollisionPair(left, right))
        hand_data = pin.GeometryData(hand_geometry)
        for request in hand_data.distanceRequests:
            request.enable_signed_distance = True
    rows = []
    for label, pair in cases:
        q = pin.neutral(model)
        proxy_q = pin.neutral(proxy_model)
        for side, prefix in (("left", ""), ("right", "right_")):
            for number in range(1, 7):
                joint = model.joints[model.getJointId(f"{side}_arm_joint{number}")]
                q[joint.idx_q] = pair[side][number - 1]
                proxy_joint = proxy_model.joints[proxy_model.getJointId(f"{prefix}arm_joint{number}")]
                proxy_q[proxy_joint.idx_q] = pair[side][number - 1]
        pin.forwardKinematics(model, data, q)
        pin.forwardKinematics(proxy_model, proxy_data, proxy_q)
        for side, prefix in (("left", ""), ("right", "right_")):
            cad_frame = data.oMi[model.getJointId(f"{side}_arm_joint6")].homogeneous
            proxy_frame = proxy_data.oMi[proxy_model.getJointId(f"{prefix}arm_joint6")].homogeneous
            if not np.allclose(cad_frame, proxy_frame, atol=1e-8, rtol=0):
                raise ValueError(f"CAD/proxy arm FK differs: {label}: {side}")
        samples = []
        for opening in np.linspace(*ranges[0], 101):
            # Pinocchio's unreduced model treats mimic joints independently.
            # Set all six explicitly to the one reviewed primary coordinate.
            for j, name in enumerate(model.names):
                if "_eg2_4b2_cad_" in str(name) and model.joints[j].nq:
                    q[model.joints[j].idx_q] = opening
            pin.computeDistances(model, data, geometry, geometry_data, q)
            samples.append([float(r.min_distance) for r in geometry_data.distanceResults])
        values = np.asarray(samples)
        if not np.isfinite(values).all():
            raise RuntimeError(f"Native CAD distance query returned non-finite results: {label}")
        row = {"case": label, "opening_samples": 101,
            "opening_range_rad": list(ranges[0]),
            "sampling_is_not_continuous_distance_certification": True,
            "sensor_to_table_range_m": {side: [float(values[:, 2*i:2*i+2].min()),
                float(values[:, 2*i:2*i+2].max())] for i, side in enumerate(("left", "right"))}}
        if hand_geometry is not None:
            for j, name in enumerate(model.names):
                if "_eg2_4b2_cad_" in str(name) and model.joints[j].nq:
                    side = "left" if str(name).startswith("left_") else "right"
                    q[model.joints[j].idx_q] = hand_opening_by_side[side]
            pin.computeDistances(model, data, hand_geometry, hand_data, q)
            pin.computeCollisions(model, data, hand_geometry, hand_data, q, False)
            pairs = [{"names": [str(hand_geometry.geometryObjects[i].name)
                for i in (pair.first, pair.second)], "distance_m": float(distance.min_distance),
                "colliding": collision.isCollision()}
                for pair, distance, collision in zip(hand_geometry.collisionPairs,
                    hand_data.distanceResults, hand_data.collisionResults, strict=True)]
            if not np.isfinite([pair["distance_m"] for pair in pairs]).all():
                raise RuntimeError(f"CAD hand distance query is non-finite: {label}")
            row["cross_hand_cad"] = {"opening_rad_by_side": dict(hand_opening_by_side),
                "geometry_count_by_side": {side: len(indices) for side, indices in by_side.items()},
                "checked_pair_count": len(pairs), "closest_pair": min(pairs, key=lambda pair: pair["distance_m"]),
                "colliding_pairs": [pair for pair in pairs if pair["colliding"]],
                "clearance_below_20mm_pairs": [pair for pair in pairs if pair["distance_m"] < .02],
                "scope": "declared_openings_only_not_a_continuous_sweep_or_production_admission",
                "mesh_signed_penetration_depth_certified": False}
        rows.append(row)
    return fixed_alignment, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cad-urdf", type=Path, required=True)
    parser.add_argument("--proxy-urdf", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--endpoints", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segments", type=int, default=16)
    parser.add_argument("--hand-openings-rad", type=float, nargs=2, metavar=("LEFT", "RIGHT"),
                        help="Also compare complete CAD cross-hand geometry at these explicit openings")
    args = parser.parse_args()
    import pinocchio as pin
    import hppfcl
    from robot_motion.rm65.scene_profiles import DUAL_ARM_DMTAC_HAND_PROXY_CHECKED_COLLISION_PAIRS

    model, baseline, _ = pin.buildModelsFromUrdf(str(args.proxy_urdf.resolve()),
        package_dirs=[str(args.proxy_urdf.resolve().parent)])
    names = {str(g.name): i for i, g in enumerate(baseline.geometryObjects)}
    baseline.removeAllCollisionPairs()
    for a, b in DUAL_ARM_DMTAC_HAND_PROXY_CHECKED_COLLISION_PAIRS:
        baseline.addCollisionPair(pin.CollisionPair(names[a], names[b]))
    candidate = deepcopy(baseline)
    hulls, coverage = {}, {}
    mesh_boxes = validate_mesh_boxes(args.cad_urdf, padding_m=.0001)
    for side, frame in (("left", "arm_link6"), ("right", "right_arm_link6")):
        sweeps = load_sensor_sweeps(args.cad_urdf, side)
        hull, proof = conservative_sweep_hull(sweeps, segments=args.segments)
        # Independent dense sampling catches coding/frame defects; the sagitta
        # bound, not this sampling, supplies continuous interval coverage.
        max_violation = max(float(np.max(s.corners(q) @ hull.equations[:, :3].T
            + hull.equations[:, 3])) for s in sweeps
            for q in np.linspace(*s.joint_range_rad, 1001))
        if max_violation > 1e-9:
            raise RuntimeError(f"Hull coverage regression: {side}: {max_violation}")
        geometry = candidate.geometryObjects[names[side + "_hand_proxy_1"]]
        geometry.geometry = _native_hull(hull, hppfcl)
        geometry.placement = model.frames[model.getFrameId(frame)].placement.copy()
        coverage[side] = {**proof, "dense_check_max_violation_m": max_violation,
            "native_fk_check": validate_sweep_fk(args.cad_urdf, sweeps, hull, pin)}
        hulls[side] = {"points": hull.points[hull.vertices].tolist(),
            "faces": ConvexHull(hull.points[hull.vertices]).simplices.tolist()}

    observation = json.loads(args.observation.read_text())["qpos_by_side"]
    cases = [("measured_start", observation)]
    if args.endpoints:
        for row in json.loads(args.endpoints.read_text())["rows"]:
            pair = {s: list(q) for s, q in observation.items()}
            pair[row["side"]] = row["best"]["qpos_rad"]
            cases.append((f"{row['side']}_{row['index']}", pair))
    alignment, cad_rows = compare_cad_in_same_scene(args.cad_urdf, model, baseline, cases, pin,
        hand_opening_by_side=None if args.hand_openings_rad is None else
            dict(zip(("left", "right"), args.hand_openings_rad, strict=True)))
    data = model.createData()
    rows = []
    for (label, pair), cad_row in zip(cases, cad_rows):
        q = pin.neutral(model)
        for side, prefix in (("left", ""), ("right", "right_")):
            for number in range(1, 7):
                joint = model.joints[model.getJointId(f"{prefix}arm_joint{number}")]
                q[joint.idx_q] = pair[side][number - 1]
        comparisons = {}
        for key, geometries in (("baseline", baseline), ("candidate", candidate)):
            geometry_data = pin.GeometryData(geometries)
            for request in geometry_data.distanceRequests:
                request.enable_signed_distance = True
            pin.computeDistances(model, data, geometries, geometry_data, q)
            values = [{"names": [str(geometries.geometryObjects[i].name) for i in (p.first, p.second)],
                "distance_m": float(r.min_distance)} for p, r in zip(geometries.collisionPairs, geometry_data.distanceResults)]
            if not np.isfinite([v["distance_m"] for v in values]).all():
                raise RuntimeError(f"Native distance query returned non-finite results: {label}: {key}")
            comparisons[key] = {"minimum_distance_m": min(v["distance_m"] for v in values),
                "blocked_pairs": [v for v in values if v["distance_m"] <= .02], "pairs": values}
        rows.append({"case": label, **comparisons, "same_scene_cad": cad_row})
        print(json.dumps({"case": label, "baseline_mm": 1000*comparisons["baseline"]["minimum_distance_m"],
            "candidate_mm": 1000*comparisons["candidate"]["minimum_distance_m"]}), flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "result.json").write_text(json.dumps({"hardware_connected": False,
        "production_geometry_changed": False, "minimum_clearance_m": .02,
        "pair_count": len(candidate.collisionPairs), "coverage": coverage,
        "mesh_box_checks": mesh_boxes, "fixed_scene_alignment": alignment, "rows": rows}, indent=2) + "\n")
    (args.output / "hulls.json").write_text(json.dumps(hulls) + "\n")


if __name__ == "__main__":
    main()
