"""Offline RGB-D/self-filter/proxy inspection. No devices or motion executor."""
from __future__ import annotations

import argparse
import base64
import gzip
import io
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from agent_gateway.profiles.real.motion_observed_scene import (
    SafeMotionObservationConfig, obstacle_cloud_from_captures,
)
from agent_gateway.profiles.real.realsense_perception import (
    RealAlignedRgbdCapture, build_real_perception_source_binding,
)
from agent_gateway.profiles.real.rm65_camera_geometry import build_rm65_rgb_camera_geometry
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.grasp_visible_scene import (
    PRODUCTION_VISIBLE_GRASP_DEPTH_FILTER, filter_isolated_background_depth,
)
from robot_motion.observed_obstacles import ObservedBoxDistance
from robot_motion.rm65.captured_robot_observations import (
    RM65CapturedRobotObservationFilter, RM65CapturedRobotObservationFilterConfig,
)
from scripts.probes.capture_rm65_grasp_snapshot import stationarity
from scripts.probes.replay_bimanual_grasp_snapshot import load_snapshot_report
from teleop_runtime.skill_deploy.rm65_grasp_trajectory import RM65GraspJointTrajectoryPlanner
from teleop_runtime.skill_deploy.rm65_lower_assembly import build_exclusive_rm65_skill_target_builder
from teleop_runtime.skill_deploy.rm65_profile import (
    ExclusiveRM65SkillDeploymentProfile, resolve_exclusive_rm65_skill_deployment_profile,
)

ROLES = ('main', 'wrist_left', 'wrist_right')
SOURCES = dict(zip(ROLES, ('front', 'left_wrist', 'right_wrist')))


def json_default(value):
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def select_frames(report, index=2, maximum_skew_sec=.1):
    if report.get('status') != 'captured' or not report.get('stationarity'):
        raise ValueError('A successfully bracketed stationary RGB-D snapshot is required')
    views = report.get('views', {})
    if any(not views.get(role, {}).get('frames') for role in ROLES):
        raise ValueError('All three RGB-D views are required; no partial-view fallback')
    anchor = views['main']['frames'][index]
    timestamp = anchor['received_at']
    selected = {role: min(views[role]['frames'], key=lambda f: abs(f['received_at']-timestamp))
                for role in ROLES}
    times = np.array([frame['received_at'] for frame in selected.values()])
    if not np.isfinite(times).all() or np.ptp(times) > maximum_skew_sec:
        raise ValueError('Selected RGB-D frames exceed the inter-camera skew budget')
    return selected


def load_arrays(capture_dir, frame):
    root = Path(capture_dir).resolve(strict=True)
    path = (root / frame['file']).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError('Snapshot frame escapes its capture directory')
    with np.load(path, allow_pickle=False) as data:
        rgb, depth = data['rgb'].copy(), data['depth_m'].copy()
    if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
            or depth.shape != rgb.shape[:2] or not np.isfinite(depth).all() or np.any(depth < 0)):
        raise ValueError('Saved RGB and aligned metric depth must have matching finite dimensions')
    return rgb, depth  # Already meters. Never apply depth_scale again.


def backproject(depth, intrinsics, world_from_camera, mask):
    rows, columns = np.nonzero(mask & (depth > 0))
    z = depth[rows, columns]
    points = np.column_stack(((columns-intrinsics['cx'])*z/intrinsics['fx'],
                             (rows-intrinsics['cy'])*z/intrinsics['fy'], z))
    pose = np.asarray(world_from_camera).reshape(4, 4)
    return points @ pose[:3, :3].T + pose[:3, 3], rows, columns


def project(points, capture, depth_tolerance_m=.01):
    """Project world points into this exact RGB-D cut, retaining occlusion flags."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    pose = np.asarray(capture.source_from_camera).reshape(4, 4)
    local = (points-pose[:3, 3]) @ pose[:3, :3]
    z = local[:, 2]
    uv = np.full((len(points), 2), np.nan)
    front = np.isfinite(local).all(axis=1) & (z > 0)
    intr = capture.intrinsics
    uv[front, 0] = local[front, 0]/z[front]*intr['fx'] + intr['cx']
    uv[front, 1] = local[front, 1]/z[front]*intr['fy'] + intr['cy']
    valid = (front & (uv[:, 0] >= 0) & (uv[:, 0] <= capture.width-1)
             & (uv[:, 1] >= 0) & (uv[:, 1] <= capture.height-1))
    pixels = np.zeros((len(points), 2), dtype=int)
    pixels[valid] = np.rint(uv[valid]).astype(int)
    observed = np.zeros(len(points))
    observed[valid] = capture.depth_m[pixels[valid, 1], pixels[valid, 0]]
    consistent = valid & (observed > 0) & (np.abs(observed-z) <= depth_tolerance_m)
    return {'uv': uv, 'in_image': valid, 'depth_consistent': consistent,
            'camera_depth_m': z, 'observed_depth_m': observed}


def resolve_inputs(context, urdf, *, filter_config=None, observation_urdf_path=None,
                   self_observation_proxy_urdf_path=None, envelope_profile=None):
    context, urdf = Path(context), Path(urdf).resolve(strict=True)
    frozen_urdf = context / 'camera_mount_urdf.urdf'
    if frozen_urdf.read_bytes() != urdf.read_bytes():
        raise ValueError('Selected physical URDF differs from the frozen capture context')
    profile = resolve_exclusive_rm65_skill_deployment_profile(ExclusiveRM65SkillDeploymentProfile(
        arm_ip_assignment_path=context / 'arm_ip_assignment.json',
        tool_profile_path_by_side={side: context / f'{side}_tool_profile.json' for side in ('left', 'right')},
        envelope_profile_path=(context / 'envelope_profile.json' if envelope_profile is None
                               else Path(envelope_profile)),
        collision_scene_profile='dual_arm_hand_proxy_dmtac_strict'))
    builder = build_exclusive_rm65_skill_target_builder(profile)
    binding = build_real_perception_source_binding(camera_config_path=context / 'camera_config.json',
        camera_registry_path=context / 'camera_registry.json',
        scene_world_from_arm_base_by_side=profile.scene_world_from_arm_base_by_side)
    geometry = build_rm65_rgb_camera_geometry(source_binding=binding, target_builder=builder,
        camera_registry_path=context / 'camera_registry.json', camera_mount_urdf_path=urdf,
        scene_world_from_arm_base_by_side=profile.scene_world_from_arm_base_by_side,
        scene_world_from_canonical_station=profile.scene_world_from_canonical_station)
    cleaner = RM65CapturedRobotObservationFilter.for_binding(urdf_path=urdf,
        config=filter_config, observation_urdf_path=observation_urdf_path,
        self_observation_proxy_urdf_path=self_observation_proxy_urdf_path,
        tool_tcp_pose_flange_m_rad_by_side=profile.tool_tcp_pose_flange_m_rad_by_side,
    )
    return profile, builder, binding, geometry, cleaner


def load_captures(capture_dir, profile, binding, geometry, *, frame_index=2):
    capture_dir = Path(capture_dir)
    report = json.loads((capture_dir / 'snapshot.json').read_text())
    if report.get('schema') == 'agenticrobot.safe_move_inputs.v1':
        # Normal Skill records already contain the exact frozen camera geometry.
        # Do not reacquire frames or recompute their poses from today's qpos.
        if tuple(view['role'] for view in report['views']) != ROLES:
            raise ValueError('All three recorded Skill views are required for this inspection')
        captures = []
        for view in report['views']:
            rgb, depth = load_arrays(capture_dir, view)
            captures.append(RealAlignedRgbdCapture(source_id=view['source_id'], role=view['role'],
                acquired_at=view['acquired_at'], station_revision=view['station_revision'],
                source_generation=view['source_generation'], frame_index=view['frame_index'],
                capture_group_id=str(capture_dir), source_binding_hash=view['source_binding_hash'],
                width=rgb.shape[1], height=rgb.shape[0], intrinsics=view['intrinsics'],
                source_from_camera=tuple(np.asarray(view['T_world_from_camera']).ravel()),
                rgb=rgb, depth_m=depth, rgb_content_hash=view['rgb_content_hash'],
                depth_content_hash=view['depth_content_hash'],
                capture_qpos_by_side={side: np.asarray(q) for side, q in view['capture_qpos_by_side'].items()},
                pose_measured_at=view['pose_measured_at'], pose_available_at=view['pose_available_at'],
                pose_evidence_refs=tuple(view['pose_evidence_refs'])))
        return captures
    if report.get('schema') == 'agenticrobot.readonly_rgbd_qpos.v2':
        # Share the accepted writer/replay lifecycle, rather than introducing
        # a second persistent-session format for the offline inspector.
        report = load_snapshot_report(capture_dir)
    elif report.get('schema') != 'agenticrobot.readonly_rgbd_qpos.v1' or report.get('cleanup_complete') is not True:
        raise ValueError('This probe requires a closed one-shot read-only capture')
    selected = select_frames(report, frame_index)
    history = report['qpos_samples']
    stationarity(history, report['capture_started_at'], report['capture_finished_at'])
    captures = []
    for role in ROLES:
        frame = selected[role]
        sample = min(history, key=lambda s: abs(max(v['finished_at'] for v in s.values())-frame['received_at']))
        pair = {side: np.asarray(sample[side]['qpos_rad'], dtype=float) for side in ('left', 'right')}
        measured = min(row['started_at'] for row in sample.values())
        available = max(row['finished_at'] for row in sample.values())
        geometry.publish_measured_qpos(qpos_by_side=pair, measured_at=measured,
            available_at=available, evidence_refs=('readonly-rgbd-snapshot',))
        cut = geometry.pose_cut(source_id=SOURCES[role], frame_acquired_at=frame['received_at'],
                                now=max(frame['received_at'], available))
        rgb, depth = load_arrays(capture_dir, frame)
        captures.append(RealAlignedRgbdCapture(source_id=SOURCES[role], role=role,
            acquired_at=frame['received_at'], station_revision=0,
            source_generation=frame['source_generation'], frame_index=frame['source_frame_index'],
            capture_group_id=str(capture_dir), source_binding_hash=binding.semantic_content_hash,
            width=rgb.shape[1], height=rgb.shape[0], intrinsics=frame['intrinsics'],
            source_from_camera=cut.source_from_camera, rgb=rgb, depth_m=depth,
            rgb_content_hash=canonical_content_hash(rgb, domain='readonly-rgb'),
            depth_content_hash=canonical_content_hash(depth, domain='readonly-depth'),
            capture_qpos_by_side=pair, pose_measured_at=measured, pose_available_at=available,
            pose_evidence_refs=cut.evidence_refs))
    return captures


def corners(pose, half):
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    return (signs * half) @ pose[:3, :3].T + pose[:3, 3]


def inspect_captures(captures, *, cleaner, config, profile, builder, output, maximum_display_points=60000):
    """Persist exact computation inputs before producing a display-only sample."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    filtered = {}
    active = iter(captures)

    def record_filter(**kwargs):
        capture = next(active)
        result = cleaner(**kwargs)
        valid, _ = filter_isolated_background_depth(depth_m=result.depth_m,
            target_mask=np.zeros(result.depth_m.shape, bool), config=PRODUCTION_VISIBLE_GRASP_DEPTH_FILTER)
        filtered[capture.role] = (result, valid)
        np.savez_compressed(output / f'{capture.role}.npz', rgb=capture.rgb,
            depth_m=capture.depth_m, self_removed_mask=result.removed_mask,
            filtered_depth_m=result.depth_m, retained_obstacle_mask=valid)
        return result

    qpos = captures[0].capture_qpos_by_side
    cloud = obstacle_cloud_from_captures(captures=captures, config=config,
        reference_frame_id=profile.station_world_frame_id, self_filter=record_filter,
        planning_qpos_by_side=qpos, max_inter_camera_skew_sec=.1)
    np.savez_compressed(output / 'obstacles.npz', occupied_centers_m=cloud.points_m)
    # This constructor owns independent FK/collision data. Never call its
    # trajectory search, an executor, or any lower resource lifecycle here.
    planner = RM65GraspJointTrajectoryPlanner(profile=profile, planning_target_builder=builder,
        target_envelope_filter=None, observed_obstacles=cloud)
    check = planner.oracle.check(qpos)
    boxes = planner.oracle.world_obstacle_proxy_boxes(qpos)
    observed_distance, nearest = ObservedBoxDistance(cloud).check_boxes(boxes,
        clearance_m=planner.scene.minimum_clearance_m)
    point = None if nearest is None else np.array(nearest['occupied_center_world_m'])
    box_payload = [{'name': name, 'pose': pose.tolist(), 'half': half.tolist(),
                    'corners': corners(pose, half).tolist()} for name, pose, half in boxes]
    views, points_payload = [], []
    per_view_limit = max(1, maximum_display_points // len(captures))
    for capture in captures:
        result, valid = filtered[capture.role]
        raw_mask = (capture.depth_m > 0) & (capture.depth_m <= config.maximum_depth_m)
        points, rows, columns = backproject(capture.depth_m, capture.intrinsics,
                                          capture.source_from_camera, raw_mask)
        sampled = np.arange(0, len(points), max(1, int(np.ceil(len(points)/per_view_limit))))
        labels = np.where(result.removed_mask[rows, columns], 1,
                          np.where(valid[rows, columns], 0, 2)).astype(np.uint8)
        np.savez_compressed(output / f'{capture.role}-points.npz', points_m=points,
            uv=np.column_stack((columns, rows)), rgb=capture.rgb[rows, columns], category=labels)
        points_payload.append({'role': capture.role, 'xyz': np.round(points[sampled], 5).tolist(),
            'rgb': capture.rgb[rows[sampled], columns[sampled]].tolist(),
            'uv': np.column_stack((columns[sampled], rows[sampled])).tolist(),
            'category': labels[sampled].tolist(), 'full_count': len(points)})
        image = Image.fromarray(capture.rgb)
        jpeg = io.BytesIO()
        image.save(jpeg, format='JPEG', quality=90)
        overlay = image.copy()
        nearest_projection = None
        if point is not None:
            projection = project([point], capture)
            nearest_projection = {'uv': projection['uv'][0].tolist() if projection['in_image'][0] else None,
                'in_image': bool(projection['in_image'][0]),
                'depth_consistent': bool(projection['depth_consistent'][0]),
                'camera_depth_m': float(projection['camera_depth_m'][0]),
                'observed_depth_m': float(projection['observed_depth_m'][0])}
            if projection['in_image'][0]:
                u, v = projection['uv'][0]
                draw = ImageDraw.Draw(overlay)
                color = (0, 200, 0) if projection['depth_consistent'][0] else (255, 130, 0)
                draw.ellipse((u-8, v-8, u+8, v+8), outline=color, width=3)
                draw.line((u-12, v, u+12, v), fill=color, width=2)
                draw.line((u, v-12, u, v+12), fill=color, width=2)
        overlay.save(output / f'{capture.role}-nearest.jpg')
        views.append({'role': capture.role, 'source_id': capture.source_id,
            'acquired_at': capture.acquired_at, 'pose_measured_at': capture.pose_measured_at,
            'frame_index': capture.frame_index, 'source_generation': capture.source_generation,
            'intrinsics': dict(capture.intrinsics), 'width': capture.width, 'height': capture.height,
            'pose': np.asarray(capture.source_from_camera).reshape(4, 4).tolist(),
            'qpos': {s: v.tolist() for s, v in capture.capture_qpos_by_side.items()},
            'rgb_content_hash': capture.rgb_content_hash, 'depth_content_hash': capture.depth_content_hash,
            'nearest_projection': nearest_projection,
            'jpeg': 'data:image/jpeg;base64,' + base64.b64encode(jpeg.getvalue()).decode(),
            'depth_f32': base64.b64encode(np.asarray(capture.depth_m, dtype='<f4').tobytes()).decode(),
            'removed_pixels': int(result.removed_mask.sum()), 'retained_pixels': int(valid.sum())})
    report = {'schema': 'agenticrobot.safe_move_rgbd_inspection.v1', 'device_connections': False,
        'motion_commands': [], 'path_search_invoked': False,
        'reference_frame_id': profile.station_world_frame_id, 'configuration': asdict(config),
        'collision_check': check.to_dict(), 'observed_distance_m': observed_distance,
        'nearest_observed': nearest, 'cloud_evidence': cloud.evidence_payload(),
        'qpos_by_side': {s: v.tolist() for s, v in qpos.items()}, 'boxes': box_payload,
        'views': [{k: v for k, v in view.items() if k not in ('jpeg', 'depth_f32')} for view in views]}
    (output / 'report.json').write_text(json.dumps(report, default=json_default, indent=2, allow_nan=False))
    display = {'report': report, 'views': views, 'points': points_payload}
    packed = base64.b64encode(gzip.compress(json.dumps(display, default=json_default,
        separators=(',', ':'), allow_nan=False).encode(), mtime=0)).decode()
    template = Path(__file__).with_name('safe_move_inspection.html').read_text()
    gallery = ''.join(f'<figure><figcaption>{view["role"]}</figcaption>'
        f'<img alt="{view["role"]} 原始 RGB" src="{view["jpeg"]}"></figure>' for view in views)
    (output / 'inspection.html').write_text(template.replace('__INSPECTION_DATA__', packed)
        .replace('__RGB_PREVIEW__', gallery))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path)
    parser.add_argument('--context', required=True, type=Path)
    parser.add_argument('--urdf', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--envelope-profile', type=Path)
    parser.add_argument('--captured-robot-filter-config', type=Path)
    parser.add_argument('--captured-robot-observation-urdf', type=Path)
    parser.add_argument('--captured-robot-self-envelope-urdf', type=Path)
    args = parser.parse_args()
    profile, builder, binding, geometry, cleaner = resolve_inputs(args.context, args.urdf,
        filter_config=(None if args.captured_robot_filter_config is None else
            RM65CapturedRobotObservationFilterConfig.from_json_file(args.captured_robot_filter_config)),
        observation_urdf_path=args.captured_robot_observation_urdf,
        self_observation_proxy_urdf_path=args.captured_robot_self_envelope_urdf,
        envelope_profile=args.envelope_profile)
    if args.check_only:
        # Exercise native CAD/Embree without any camera or robot connection.
        cleaner(depth_m=np.ones((2, 2)), target_mask=np.zeros((2, 2), bool),
            intrinsics={'fx': 100., 'fy': 100., 'cx': .5, 'cy': .5}, source_from_camera=np.eye(4),
            capture_qpos_by_side={side: np.zeros(6) for side in ('left', 'right')})
        print(json.dumps({'check_only': True, 'device_connections': False, 'native_scene_ready': True}))
        return 0
    if args.capture is None or args.output is None:
        parser.error('--capture and --output are required unless --check-only is used')
    captures = load_captures(args.capture, profile, binding, geometry)
    report = inspect_captures(captures, cleaner=cleaner, config=SafeMotionObservationConfig(),
        profile=profile, builder=builder, output=args.output)
    print(json.dumps({'output': str(args.output), 'collision_reason': report['collision_check']['reason'],
        'inspection': str(args.output / 'inspection.html'), 'device_connections': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
