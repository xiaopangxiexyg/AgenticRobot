"""Compare capture-only primitive and CAD masks on closed RGB-D archives."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from robot_motion.rm65.captured_robot_observations import (
    RM65CapturedRobotObservationFilter,
    RM65CapturedRobotObservationFilterConfig,
)
from scripts.probes.inspect_safe_move_snapshot import load_captures, resolve_inputs


def compare_session(*, session, urdf, envelope_profile, output,
                    captured_robot_filter_config, captured_robot_self_envelope_urdf,
                    captured_robot_observation_urdf=None):
    session, output = Path(session), Path(output)
    session = session.resolve(strict=True)
    entries = [json.loads(line) for line in (session / 'samples.jsonl').read_text().splitlines()]
    if not entries:
        raise ValueError('Benchmark requires at least one captured state')
    states = set()
    for entry in entries:
        if entry['status'] != 'captured':
            raise ValueError('Benchmark requires captured states, not silent skipping')
        state = entry['state']
        if (not isinstance(state, str) or re.fullmatch(r'[A-Za-z0-9_-]+', state) is None
                or state in states):
            raise ValueError('Benchmark requires unique portable state names')
        states.add(state)
        snapshot = (session / entry['snapshot']).resolve(strict=True)
        if snapshot.name != 'snapshot.json' or not snapshot.is_relative_to(session):
            raise ValueError('Benchmark snapshot escapes its session')
    config = RM65CapturedRobotObservationFilterConfig.from_json_file(captured_robot_filter_config)
    if config.geometry_mode != 'cad_surface':
        raise ValueError('Benchmark requires an explicit cad_surface reference policy')
    profile, _, binding, geometry, cad = resolve_inputs(session / 'context', urdf,
        envelope_profile=envelope_profile, filter_config=config,
        observation_urdf_path=captured_robot_observation_urdf,
        self_observation_proxy_urdf_path=captured_robot_self_envelope_urdf)
    primitive = RM65CapturedRobotObservationFilter.for_binding(urdf_path=urdf,
        config=replace(config, geometry_mode='link_primitives'),
        observation_urdf_path=captured_robot_observation_urdf,
        self_observation_proxy_urdf_path=captured_robot_self_envelope_urdf,
        tool_tcp_pose_flange_m_rad_by_side=profile.tool_tcp_pose_flange_m_rad_by_side,
    )
    output.mkdir(parents=True, exist_ok=False)
    report = {'session': str(session), 'device_connections': False, 'motion_commands': [],
              'reference_is_pixel_ground_truth': False, 'views': []}
    for entry in entries:
        if entry['status'] != 'captured':
            raise ValueError('Benchmark requires captured states, not silent skipping')
        state = entry['state']
        captures = load_captures(session / entry['snapshot'].rsplit('/', 1)[0],
                                 profile, binding, geometry)
        for capture in captures:
            name = f'{state}-{capture.role}'
            kwargs = {'depth_m': capture.depth_m, 'target_mask': np.zeros(capture.depth_m.shape, bool),
                'intrinsics': capture.intrinsics,
                'source_from_camera': np.asarray(capture.source_from_camera).reshape(4, 4),
                'capture_qpos_by_side': capture.capture_qpos_by_side}
            start = time.perf_counter()
            reference = cad(**kwargs)
            cad_sec = time.perf_counter() - start
            timings = []
            previous = None
            for _ in range(3):
                start = time.perf_counter()
                actual = primitive(**kwargs)
                timings.append(time.perf_counter() - start)
                if previous is not None:
                    np.testing.assert_array_equal(actual.removed_mask, previous)
                previous = actual.removed_mask
            protected = primitive(**(kwargs | {'target_mask': np.ones(capture.depth_m.shape, bool)}))
            np.testing.assert_array_equal(protected.depth_m, capture.depth_m)
            assert not protected.removed_mask.any()
            old, new = reference.removed_mask, actual.removed_mask
            missed, added = old & ~new, new & ~old
            np.savez_compressed(output / f'{name}.npz', rgb=capture.rgb,
                depth_m=capture.depth_m, cad_removed=old, primitive_removed=new,
                source_from_camera=kwargs['source_from_camera'],
                left_qpos=capture.capture_qpos_by_side['left'],
                right_qpos=capture.capture_qpos_by_side['right'])
            rgb = capture.rgb
            overlay = rgb.astype(float).copy()
            for mask, color in ((old & new, (40, 170, 255)), (missed, (255, 50, 50)),
                                (added, (255, 210, 0))):
                overlay[mask] = .4 * overlay[mask] + .6 * np.array(color)
            image = Image.fromarray(overlay.astype('uint8'))
            image.save(output / f'{name}-overlay.png')
            Image.fromarray(rgb).save(output / f'{name}-rgb.jpg')
            sheet = Image.new('RGB', (1280, 510), '#151515')
            sheet.paste(Image.fromarray(rgb), (0, 30))
            sheet.paste(image, (640, 30))
            ImageDraw.Draw(sheet).text((8, 8),
                f'{name} | blue=both; red=CAD only; yellow=primitive only', fill='white')
            sheet.save(output / f'{name}-comparison.jpg')
            row = {'name': name, 'valid_points': int((capture.depth_m > 0).sum()),
                'cad_removed': int(old.sum()), 'primitive_removed': int(new.sum()),
                'cad_only': int(missed.sum()), 'primitive_only': int(added.sum()),
                'cad_wall_sec': cad_sec, 'primitive_wall_sec': timings,
                'primitive_evidence': dict(actual.evidence)}
            report['views'].append(row)
            (output / 'result.json').write_text(json.dumps(report, indent=2))
            print(json.dumps({k: v for k, v in row.items() if k != 'primitive_evidence'}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True, type=Path)
    parser.add_argument('--urdf', required=True, type=Path)
    parser.add_argument('--envelope-profile', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--captured-robot-filter-config', required=True, type=Path)
    parser.add_argument('--captured-robot-self-envelope-urdf', required=True, type=Path)
    parser.add_argument('--captured-robot-observation-urdf', type=Path)
    compare_session(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
