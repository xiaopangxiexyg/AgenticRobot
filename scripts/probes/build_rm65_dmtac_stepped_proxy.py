"""Build separate self-observation and rigid proxies; never modify shared assets.

The original core box remains unchanged. The terminal box is split in its
own Z direction: wide at the mount and narrow at the sensor tips. Dimensions
cover the continuous CAD mesh sweep, not only open/closed FK samples. A separate
rigid approximation extends the lower footprint upward as requested.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull

from scripts.probes.compare_rm65_dmtac_sweep import (
    _origin, conservative_sweep_hull, load_sensor_sweeps,
)


def clipped_vertices(hull: ConvexHull, z: float, *, lower: bool) -> np.ndarray:
    """Intersect a convex polyhedron with a Z halfspace, including cut edges."""
    vertices = hull.points[hull.vertices]
    selected = vertices[vertices[:, 2] <= z] if lower else vertices[vertices[:, 2] >= z]
    edges = {tuple(sorted((int(a), int(b)))) for face in hull.simplices
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))}
    crossings = []
    for a, b in edges:
        p, q = hull.points[a], hull.points[b]
        if (p[2] - z) * (q[2] - z) < 0:
            crossings.append(p + (q - p) * ((z - p[2]) / (q[2] - p[2])))
    if crossings:
        selected = np.concatenate((selected, crossings))
    if not len(selected):
        raise ValueError("Split leaves an empty half of the sensor envelope")
    return selected


def stepped_boxes(hull: ConvexHull, old_center: np.ndarray,
                  old_size: np.ndarray) -> tuple[list[dict], dict]:
    old_min, old_max = old_center - old_size / 2, old_center + old_size / 2
    vertices = hull.points[hull.vertices]
    # Retain at least the old upper width, including the analytical padding
    # when its conservative arc bound extends beyond the old sampled box.
    old_min = np.minimum(old_min, vertices.min(axis=0))
    old_max = np.maximum(old_max, vertices.max(axis=0))
    best = None
    # Require meaningful height in both halves, rather than a vanishing tip box.
    for z in np.linspace(old_min[2] + .005, old_max[2] - .005,
                         max(2, int(old_size[2] / .001))):
        lower = clipped_vertices(hull, float(z), lower=True)
        lo = np.r_[lower[:, :2].min(axis=0), old_min[2]]
        hi = np.r_[lower[:, :2].max(axis=0), z]
        upper_lo = np.r_[old_min[:2], z]
        volume = float(np.prod(hi - lo) + np.prod(old_max - upper_lo))
        if hi[0] - lo[0] >= old_max[0] - old_min[0] - .001:
            continue
        if best is None or volume < best[0]:
            best = volume, float(z), lo, hi, upper_lo
    if best is None:
        raise ValueError("CAD sweep does not permit a narrower lower terminal box")
    volume, z, lo, hi, upper_lo = best
    boxes = [dict(center_m=((upper_lo + old_max) / 2).tolist(),
                  size_m=(old_max - upper_lo).tolist()),
             dict(center_m=((lo + hi) / 2).tolist(), size_m=(hi - lo).tolist())]
    # Every half-polyhedron vertex is covered; convexity then covers its solid.
    excess = []
    for box, lower in zip(boxes, (False, True)):
        points = clipped_vertices(hull, z, lower=lower)
        excess.append(float(np.max(np.abs(points - box['center_m'])
                                   - np.asarray(box['size_m']) / 2)))
    if max(excess) > 1e-9:
        raise ValueError("Two-piece envelope lost continuous CAD coverage")
    return boxes, dict(split_z_m=z, original_volume_m3=float(np.prod(old_size)),
        candidate_volume_m3=volume, halfspace_vertex_excess_m=excess,
        upper_bound_expansion_m=max(0., float(np.max(np.maximum(
            old_center - old_size / 2 - vertices, vertices - old_center - old_size / 2)))),
        method="minimum_volume_two_stacked_boxes_over_1mm_split_search")


def _frame_pose(model, data, name: str) -> np.ndarray:
    index = model.getFrameId(name)
    if index >= model.nframes:
        raise ValueError(f"Missing native frame: {name}")
    return np.asarray(data.oMf[index].homogeneous)


def lower_box_extended_upward(boxes: list[dict]) -> dict:
    """Retain the lower footprint and bottom; extend to the upper top."""
    upper, lower = boxes
    center = np.asarray(lower['center_m'], dtype=float).copy()
    size = np.asarray(lower['size_m'], dtype=float).copy()
    bottom = center[2] - size[2] / 2
    top = upper['center_m'][2] + upper['size_m'][2] / 2
    center[2], size[2] = (bottom + top) / 2, top - bottom
    return dict(center_m=center.tolist(), size_m=size.tolist())


def build_rigid_collision_tree(stepped: ET.ElementTree) -> ET.ElementTree:
    """Create the requested rigid approximation without changing self boxes."""
    tree = deepcopy(stepped)
    for side in ('left', 'right'):
        link = tree.getroot().find(f"link[@name='{side}_hand_proxy']")
        elements = link.findall('collision')
        if len(elements) != 3:
            raise ValueError('Stepped source requires core, upper and lower boxes')
        boxes = [dict(center_m=_origin(e.find('origin'))[:3, 3].tolist(),
            size_m=np.fromstring(e.find('geometry/box').get('size'), sep=' ').tolist())
            for e in elements[1:]]
        rigid = lower_box_extended_upward(boxes)
        for tag in ('collision', 'visual'):
            parts = link.findall(tag)
            template = deepcopy(parts[2])
            for element in parts[1:]:
                link.remove(element)
            template.set('name', f'{side}_rigid_terminal')
            template.find('origin').set('xyz', ' '.join(f'{x:.12f}' for x in rigid['center_m']))
            template.find('geometry/box').set('size', ' '.join(f'{x:.12f}' for x in rigid['size_m']))
            link.append(template)
    return tree


def build_candidate(cad_urdf: Path, proxy_urdf: Path, output_dir: Path) -> dict:
    import pinocchio as pin
    import trimesh

    cad_urdf, proxy_urdf = cad_urdf.resolve(strict=True), proxy_urdf.resolve(strict=True)
    source_bytes = {"cad": cad_urdf.read_bytes(), "proxy": proxy_urdf.read_bytes()}
    if output_dir.exists():
        raise FileExistsError(f"Candidate output must be a new directory: {output_dir}")
    tree, cad_root = ET.parse(proxy_urdf), ET.parse(cad_urdf).getroot()
    model = pin.buildModelFromUrdf(str(cad_urdf))
    data, q = model.createData(), pin.neutral(model)
    pin.forwardKinematics(model, data, q); pin.updateFramePlacements(model, data)
    pm = pin.buildModelFromUrdf(str(proxy_urdf)); pd = pm.createData()
    pin.forwardKinematics(pm, pd, pin.neutral(pm)); pin.updateFramePlacements(pm, pd)
    report = dict(schema="agenticrobot.dmtac_stepped_proxy.v1", scope="offline_candidate",
        source_sha256={k: hashlib.sha256(v).hexdigest() for k, v in source_bytes.items()},
        source_paths={"cad": str(cad_urdf), "proxy": str(proxy_urdf)}, sides={})
    display = {}
    for side in ("left", "right"):
        native_flange = _frame_pose(model, data, f"{side}_arm_link6")
        proxy_flange = _frame_pose(pm, pd, "arm_link6" if side == "left" else "right_arm_link6")
        proxy_pose = _frame_pose(pm, pd, f"{side}_hand_proxy")
        proxy_from_flange = np.linalg.inv(proxy_pose) @ proxy_flange
        mesh_sweeps, meshes = [], []
        for sweep in load_sensor_sweeps(cad_urdf, side):
            link = cad_root.find(f"link[@name='{sweep.sensor_link}']")
            visual = link.find("visual"); spec = visual.find("geometry/mesh")
            mesh = trimesh.load_mesh(cad_urdf.parent / spec.get("filename"))
            vertices = np.asarray(mesh.vertices) * np.fromstring(spec.get("scale", "1 1 1"), sep=" ")
            vp = _origin(visual.find("origin"))
            link_vertices = vertices @ vp[:3, :3].T + vp[:3, 3]
            relative = np.linalg.inv(native_flange) @ _frame_pose(model, data, sweep.sensor_link)
            flange_vertices = link_vertices @ relative[:3, :3].T + relative[:3, 3]
            points = flange_vertices[ConvexHull(flange_vertices).vertices]
            mesh_sweeps.append(replace(sweep, corners_at_zero=points))
            meshes.append((sweep.sensor_link, link_vertices, flange_vertices))
        hull, proof = conservative_sweep_hull(tuple(mesh_sweeps))
        local_points = hull.points[hull.vertices] @ proxy_from_flange[:3, :3].T + proxy_from_flange[:3, 3]
        local_hull = ConvexHull(local_points)
        link = tree.getroot().find(f"link[@name='{side}_hand_proxy']")
        collisions = link.findall("collision")
        if len(collisions) != 2:
            raise ValueError("Source must have core plus one terminal collision box")
        old = collisions[1]
        if not np.allclose(_origin(old.find("origin"))[:3, :3], np.eye(3), atol=1e-12):
            raise ValueError("Source terminal box must be aligned with proxy link axes")
        old_center = _origin(old.find("origin"))[:3, 3]
        old_size = np.fromstring(old.find("geometry/box").get("size"), sep=" ")
        boxes, split = stepped_boxes(local_hull, old_center, old_size)
        max_delta, max_excess = 0., -np.inf
        for opening in np.linspace(*mesh_sweeps[0].joint_range_rad, 101):
            for j, name in enumerate(model.names):
                if "_eg2_4b2_cad_" in str(name) and model.joints[j].nq:
                    q[model.joints[j].idx_q] = opening
            pin.forwardKinematics(model, data, q); pin.updateFramePlacements(model, data)
            for sweep, (name, link_vertices, initial) in zip(mesh_sweeps, meshes):
                relative = np.linalg.inv(_frame_pose(model, data, f"{side}_arm_link6")) @ _frame_pose(model, data, name)
                actual = link_vertices @ relative[:3, :3].T + relative[:3, 3]
                # Compare independently parsed native FK with the verified arc.
                at_zero = sweep.corners_at_zero
                predicted_delta = sweep.corners(float(opening))[0] - at_zero[0]
                max_delta = max(max_delta, float(np.max(np.abs(actual - initial - predicted_delta))))
                points = actual @ proxy_from_flange[:3, :3].T + proxy_from_flange[:3, 3]
                excess = np.minimum.reduce([np.max(np.abs(points - b['center_m'])
                    - np.asarray(b['size_m']) / 2, axis=1) for b in boxes])
                max_excess = max(max_excess, float(np.max(excess)))
        if max_delta > 1e-8 or max_excess > 1e-9:
            raise ValueError(f"Native CAD coverage failed: {side}: {max_delta}, {max_excess}")
        # Mechanical replacement leaves the core, frames and all other geometry intact.
        for tag in ("collision", "visual"):
            elements = link.findall(tag)
            if len(elements) != 2:
                raise ValueError("Proxy needs matching core and terminal visual/collision boxes")
            template = deepcopy(elements[1]); link.remove(elements[1])
            for index, b in enumerate(boxes, 1):
                element = deepcopy(template)
                element.set("name", f"{side}_terminal_{index}")
                element.find("origin").set("xyz", " ".join(f"{x:.12f}" for x in b['center_m']))
                element.find("geometry/box").set("size", " ".join(f"{x:.12f}" for x in b['size_m']))
                link.append(element)
        report['sides'][side] = dict(original_center_m=old_center.tolist(),
            original_size_m=old_size.tolist(), boxes=boxes, split=split, sweep=proof,
            native_fk=dict(opening_samples=101, maximum_delta_m=max_delta,
                           maximum_mesh_union_excess_m=max_excess))
        display[side] = {"hull_vertices_m": local_points.tolist(),
            "hull_faces": local_hull.simplices.tolist()}
        # Restore neutral FK before deriving the other side's initial mesh pose.
        q = pin.neutral(model)
        pin.forwardKinematics(model, data, q); pin.updateFramePlacements(model, data)
    # Local candidate references the original immutable mesh assets explicitly.
    for spec in tree.getroot().findall(".//mesh"):
        filename = Path(spec.get("filename"))
        if not filename.is_absolute():
            spec.set("filename", str((proxy_urdf.parent / filename).resolve(strict=True)))
    if source_bytes != {"cad": cad_urdf.read_bytes(), "proxy": proxy_urdf.read_bytes()}:
        raise RuntimeError("Source assets changed during candidate construction")
    output_dir.mkdir(parents=True, exist_ok=False)
    candidate = output_dir / "rm65_dual_arm_hand_proxy_dmtac_stepped.urdf"
    tree.write(candidate, encoding="utf-8", xml_declaration=True)
    rigid = output_dir / 'rm65_dual_arm_hand_proxy_dmtac_rigid.urdf'
    build_rigid_collision_tree(tree).write(rigid, encoding='utf-8', xml_declaration=True)
    report['candidate_urdf'] = str(candidate.resolve())
    report['rigid_collision_urdf'] = str(rigid.resolve())
    report['rigid_approximation'] = {side: lower_box_extended_upward(row['boxes'])
        for side, row in report['sides'].items()}
    report['rigid_approximation_method'] = 'lower_footprint_extended_to_upper_top_not_full_cad_sweep'
    (output_dir / "geometry-proof.json").write_text(json.dumps(report, indent=2) + "\n")
    (output_dir / "sweep-display.json").write_text(json.dumps(display) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cad-urdf", type=Path, required=True)
    parser.add_argument("--proxy-urdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_candidate(args.cad_urdf, args.proxy_urdf, args.output_dir)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
