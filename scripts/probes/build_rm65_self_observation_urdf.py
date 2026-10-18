"""Build visual-only, link-bound self-observation accessories for offline review.

The supplied capsule dimensions are operator-reviewed geometry, not inferred
from arbitrary nearby obstacle points. This tool never changes collision data
or the production asset binding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np


def validate_capsules(root: ET.Element, capsules: list[dict]) -> list[dict]:
    links = {item.attrib['name'] for item in root.findall('link')}
    joints = {item.attrib['name'] for item in root.findall('joint')}
    if not isinstance(capsules, list) or not capsules:
        raise ValueError('capsules must be a nonempty list of reviewed attachments')
    result, names = [], set()
    for row in capsules:
        if not isinstance(row, dict) or set(row) != {
            'name', 'parent_link', 'start_m', 'end_m', 'radius_m'
        }:
            raise ValueError('each capsule requires name, parent_link, start_m, end_m, radius_m')
        name, parent = row['name'], row['parent_link']
        if not isinstance(name, str) or re.fullmatch(r'[a-z][a-z0-9_]*', name) is None:
            raise ValueError('capsule name must be a portable lowercase identifier')
        if not isinstance(parent, str) or parent not in links or not parent.startswith(('left_', 'right_')):
            raise ValueError('parent_link must be an existing left/right URDF link')
        side = parent.split('_', 1)[0]
        link = f'{side}_self_observation_{name}'
        joint = link + '_fixed'
        if link in links or joint in joints or link in names:
            raise ValueError('attachment name collides with an existing link/joint or attachment')
        names.add(link)
        start, end = np.asarray(row['start_m'], float), np.asarray(row['end_m'], float)
        radius = float(row['radius_m'])
        if (start.shape != (3,) or end.shape != (3,) or not np.all(np.isfinite([start, end]))
                or isinstance(row['radius_m'], bool) or not np.isfinite(radius) or radius <= 0
                or np.linalg.norm(end - start) <= 1e-9):
            raise ValueError('capsule requires distinct finite XYZ endpoints and a positive radius in meters')
        result.append(dict(name=name, parent_link=parent, start_m=start.tolist(),
                           end_m=end.tolist(), radius_m=radius, link_name=link, joint_name=joint))
    return result


def build_candidate(source_urdf: Path, capsules: list[dict], output_dir: Path) -> dict:
    import trimesh

    source = source_urdf.resolve(strict=True)
    original = source.read_bytes()
    tree = ET.ElementTree(ET.fromstring(original))
    root = tree.getroot()
    rows = validate_capsules(root, capsules)
    # A copied URDF must continue to reference the exact original meshes.
    # Package URI resolution is deliberately left to its existing owner.
    for mesh in root.iter('mesh'):
        filename = mesh.attrib['filename']
        if '://' in filename:
            raise ValueError('use a source URDF with resolved filesystem mesh paths')
        mesh.set('filename', str((source.parent / filename).resolve(strict=True)))
    output_dir.mkdir(parents=True, exist_ok=False)
    for row in rows:
        start, end = np.asarray(row['start_m']), np.asarray(row['end_m'])
        direction = end - start
        # Mesh is in the parent link's coordinates; fixed joint stays identity.
        mesh = trimesh.creation.capsule(height=float(np.linalg.norm(direction)),
                                        radius=row['radius_m'], count=[16, 16])
        transform = trimesh.geometry.align_vectors([0, 0, 1], direction)
        transform[:3, 3] = (start + end) / 2
        mesh.apply_transform(transform)
        filename = row['link_name'] + '.obj'
        mesh.export(output_dir / filename)
        link = ET.SubElement(root, 'link', name=row['link_name'])
        visual = ET.SubElement(link, 'visual')
        geometry = ET.SubElement(visual, 'geometry')
        ET.SubElement(geometry, 'mesh', filename=filename)
        material = ET.SubElement(visual, 'material', name='self_observation_only')
        ET.SubElement(material, 'color', rgba='0.1 0.8 0.8 0.55')
        joint = ET.SubElement(root, 'joint', name=row['joint_name'], type='fixed')
        ET.SubElement(joint, 'parent', link=row['parent_link'])
        ET.SubElement(joint, 'child', link=row['link_name'])
        ET.SubElement(joint, 'origin', xyz='0 0 0', rpy='0 0 0')
    candidate = output_dir / 'self_observation.urdf'
    tree.write(candidate, encoding='utf-8', xml_declaration=True)
    if source.read_bytes() != original:
        raise RuntimeError('source URDF changed during generation')
    report = dict(scope='offline_visual_only_no_devices_no_binding',
                  source_urdf=str(source), source_sha256=hashlib.sha256(original).hexdigest(),
                  observation_urdf=str(candidate.absolute()), capsules=rows,
                  collision_elements_before=len(ET.fromstring(original).findall('.//collision')),
                  collision_elements_after=len(root.findall('.//collision')),
                  geometry_status='requires_visual_and_negative_control_review')
    (output_dir / 'geometry.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-urdf', required=True, type=Path)
    parser.add_argument('--capsules-json', required=True, type=Path,
                        help='JSON list; endpoints are meters in each named parent link')
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    report = build_candidate(args.source_urdf, json.loads(args.capsules_json.read_text()), args.output_dir)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
