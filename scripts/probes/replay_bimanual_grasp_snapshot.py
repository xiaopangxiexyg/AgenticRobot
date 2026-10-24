"""Frozen RGB-D replay: shared observation/candidates, parameterized arm plans.

Only pure geometry and planning constructors are used. No resource is started;
inference arrays are supplied by the separate offline GPU probe. In particular
the camera used to observe an object never selects the arm that will grasp it.
"""
from __future__ import annotations

from deployments.dmtac_graspgenx_alignment_ids import (
    DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID,
    DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_IDS,
)

import argparse
import json
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from agent_gateway.profiles.real.perception_frames import build_real_perception_station_frame_binding
from agent_gateway.profiles.real.realsense_perception import build_real_perception_source_binding
from agent_gateway.profiles.real.rm65_camera_geometry import build_rm65_rgb_camera_geometry, load_rm65_wrist_camera_side_axis_tool
from agent_gateway.profiles.real.rm65_grasp_reachability import RM65PointGraspReachabilityAdapter, arm_base_origin_in_candidate_frame
from agent_gateway.profiles.real.station_point_grasp_deployment import StationPointGraspArmReachabilityProfile
from deployments.perception.dmtac_grasp_overlay import (
    reviewed_graspgenx_dmtac_solid_geometry,
    dmtac_graspgenx_alignment_profile,
)
from deployments.perception.grasp_multiview import GraspRgbdView, fuse_grasp_views
from deployments.perception.grasp_planning_priority import grasp_planning_candidate_ordinals
from deployments.perception.sam2_graspgenx_point_grasp import (
    GraspGenXPointGraspModelSpec, Sam2GraspGenXPointGraspPlanner,
    GRASPGENX_DMTAC_REACHABILITY_ORIENTATION_VARIANTS_LOCAL_AXIS_DEG,
)
from robot_motion.rm65.captured_robot_observations import (
    RM65CapturedRobotObservationFilter, RM65CapturedRobotObservationFilterConfig,
)
from robot_motion.grasp_visible_scene import VisibleGraspScene
from teleop_runtime.skill_deploy.rm65_grasp_trajectory import RM65GraspJointTrajectoryPlanner, RM65GraspTrajectoryPlanningConfig, RM65GraspTrajectoryPlanningError
from teleop_runtime.skill_deploy.rm65_lower_assembly import (
    bind_exclusive_rm65_skill_target_safety,
    build_exclusive_rm65_skill_target_builder,
)
from teleop_runtime.skill_deploy.rm65_profile import ExclusiveRM65SkillDeploymentProfile, resolve_exclusive_rm65_skill_deployment_profile


def json_default(value):
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not JSON evidence: {type(value).__name__}")


def profile_for(args, side):
    return resolve_exclusive_rm65_skill_deployment_profile(ExclusiveRM65SkillDeploymentProfile(
        arm_ip_assignment_path=args.canonical / "teleop_runtime/configs/hardware/rm65_arm_ips/latest.json",
        tool_profile_path_by_side={s: args.canonical / f"hardware/rm65/utils/configs/calibrations/rm65_{s}_DMTAC_M.json" for s in ("left", "right")},
        envelope_profile_path=args.repo / f"teleop_runtime/configs/skill_deploy/agent_grasp_{side}_absolute_standard_v1.json",
        collision_scene_profile=args.collision_scene_profile,
    ))


class RetainedInference:
    backend_id = "offline-retained-graspgenx"
    semantic_config = {"mode": "offline_frozen_replay"}

    def __init__(self, path):
        self.inference_calls = 0
        with np.load(path, allow_pickle=False) as data:
            self.points, self.poses, self.scores = (data[k].copy() for k in ("points", "poses", "scores"))

    def start(self):
        pass

    def close(self):
        pass

    def infer_object(self, points, **kwargs):
        if points.shape != self.points.shape or not np.allclose(points, self.points, atol=1e-7, rtol=0):
            raise ValueError("Retained inference does not describe this exact target cloud")
        self.inference_calls += 1
        return self.poses.copy(), self.scores.copy()


class RetainedAuxiliarySegmenter:
    """Replay only real SAM masks for the exact captured image and seed."""

    def __init__(self, directories):
        self.results = []
        for directory in directories:
            result = json.loads((directory / "result.json").read_text())
            if result["mode"] != "segment":
                raise ValueError("Auxiliary evidence must be a segmentation result")
            self.results.append((np.load(directory / "rgb.npy", allow_pickle=False),
                tuple(result["pixel_uv"]), np.load(directory / "mask.npy", allow_pickle=False)))

    def __call__(self, image, *, pixel_uv):
        rgb = np.asarray(image)
        for recorded_rgb, seed, mask in self.results:
            if np.array_equal(rgb, recorded_rgb) and tuple(pixel_uv) == seed:
                return mask.copy()
        raise RuntimeError(f"Auxiliary view needs real SAM2 evidence for exact image/seed {pixel_uv}; no invented mask")


def load_snapshot_report(capture: Path):
    report = json.loads((capture / "snapshot.json").read_text())
    if report.get("schema") == "agenticrobot.readonly_rgbd_qpos.v2":
        if report.get("session_lifecycle") != "../../devices/lifecycle.json":
            raise ValueError("Persistent capture has an invalid lifecycle reference")
        lifecycle = json.loads((capture.parent.parent / "devices/lifecycle.json").read_text())
        if lifecycle.get("status") != "closed" or lifecycle.get("cleanup_complete") is not True:
            raise ValueError("Persistent capture session is still active or not safely closed")
    if report["status"] != "captured" or "stationarity" not in report:
        raise ValueError("Replay requires a successfully bracketed stationary capture")
    return report


def prepare(args):
    report = load_snapshot_report(args.capture)
    profile = profile_for(args, args.sides[0])
    builder = build_exclusive_rm65_skill_target_builder(profile)
    binding = build_real_perception_source_binding(
        camera_config_path=args.repo / "teleop_runtime/configs/devices/agent_gateway_realsense.json",
        camera_registry_path=args.repo / "robot_motion/configs/rm65_camera_registry_v4.json",
        scene_world_from_arm_base_by_side=profile.scene_world_from_arm_base_by_side)
    geometry = build_rm65_rgb_camera_geometry(source_binding=binding, target_builder=builder,
        camera_registry_path=args.repo / "robot_motion/configs/rm65_camera_registry_v4.json",
        camera_mount_urdf_path=args.urdf,
        scene_world_from_arm_base_by_side=profile.scene_world_from_arm_base_by_side,
        scene_world_from_canonical_station=profile.scene_world_from_canonical_station)
    samples = report["qpos_samples"]
    anchor_record = report["views"][args.anchor]["frames"][args.frame]
    q_sample = min(samples, key=lambda s: abs(max(s[x]["finished_at"] for x in s)-anchor_record["received_at"]))
    qpos = {side: np.asarray(q_sample[side]["qpos_rad"]) for side in ("left", "right")}
    views, per_view_qpos = {}, {}
    roles = {"main": "front", "wrist_left": "left_wrist", "wrist_right": "right_wrist"}
    for role, record in report["views"].items():
        if not record.get("frames"):
            continue
        frame = min(record["frames"], key=lambda f: abs(f["received_at"]-anchor_record["received_at"]))
        sample = min(samples, key=lambda s: abs(max(s[x]["finished_at"] for x in s)-frame["received_at"]))
        pair = {s: np.asarray(sample[s]["qpos_rad"]) for s in ("left", "right")}
        per_view_qpos[role] = pair
        measured = min(sample[s]["started_at"] for s in sample)
        available = max(sample[s]["finished_at"] for s in sample)
        geometry.publish_measured_qpos(qpos_by_side=pair, measured_at=measured, available_at=available, evidence_refs=("readonly-snapshot",))
        cut = geometry.pose_cut(source_id=roles[role], frame_acquired_at=frame["received_at"], now=max(frame["received_at"], available))
        station_from_camera = np.linalg.inv(profile.scene_world_from_canonical_station) @ np.asarray(cut.source_from_camera).reshape(4, 4)
        with np.load(args.capture / frame["file"], allow_pickle=False) as data:
            rgb, depth = data["rgb"].copy(), data["depth_m"].copy()
        views[role] = GraspRgbdView(role, rgb, depth, {**frame["intrinsics"], "width": rgb.shape[1], "height": rgb.shape[0]}, station_from_camera, frame["received_at"])
    anchor = views[args.anchor]
    mask = np.load(args.mask, allow_pickle=False)
    config_path = getattr(args, "captured_robot_filter_config", None)
    cleaner = RM65CapturedRobotObservationFilter.for_binding(urdf_path=args.urdf,
        config=(None if config_path is None else
            RM65CapturedRobotObservationFilterConfig.from_json_file(config_path)),
        observation_urdf_path=getattr(args, "captured_robot_observation_urdf", None),
        self_observation_proxy_urdf_path=getattr(args, "captured_robot_self_envelope_urdf", None),
        tool_tcp_pose_flange_m_rad_by_side=profile.tool_tcp_pose_flange_m_rad_by_side)
    cleaning = {}

    def clean_depth(view, target_mask):
        result = cleaner(depth_m=view.depth_m, target_mask=target_mask, intrinsics=view.intrinsics,
            source_from_camera=profile.scene_world_from_canonical_station @ view.reference_from_camera,
            capture_qpos_by_side=per_view_qpos[view.role])
        cleaning[view.role] = result.evidence
        return result.depth_m

    segment_aux = RetainedAuxiliarySegmenter(getattr(args, "aux_segmentation", ()))

    solid = reviewed_graspgenx_dmtac_solid_geometry(alignment_profile_id=getattr(
        args, "graspgenx_alignment_profile", DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID))
    fused = fuse_grasp_views(anchor=anchor, anchor_mask=mask,
        anchor_pixel_uv=tuple(args.pixel) if args.refine_target_depth else None,
        auxiliary=tuple(v for k, v in views.items() if k != args.anchor),
        geometry=solid, clean_depth=clean_depth, segment_point=segment_aux)
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / "points.npz", points=np.asarray(fused.scene.points[fused.scene.target_mask], dtype=np.float32))
    details = {"anchor": args.anchor, "view_diagnostics": fused.view_diagnostics,
               "cleaning": cleaning, "target_points": int(fused.scene.target_mask.sum()),
               "scene_points": len(fused.scene.points), "qpos_by_side": {k: v.tolist() for k, v in qpos.items()},
               "qpos_measured_at": min(q_sample[s]["started_at"] for s in q_sample),
               "camera_transforms": {k: v.reference_from_camera.tolist() for k, v in views.items()},
               "hardware_connected": False, "missing_views": {k: v for k, v in report["views"].items() if not v.get("frames")}}
    (args.output / "observation.json").write_text(json.dumps(details, indent=2, default=json_default)+"\n")
    return anchor, mask, qpos, fused, details


def replay(args):
    selected_alignment = dmtac_graspgenx_alignment_profile(
        getattr(args, "graspgenx_alignment_profile", DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID))
    selected_alignment.require_motion_sides(args.sides)
    if len(set(args.sides)) != len(args.sides):
        raise ValueError("Each executing arm must be named only once")
    started = time.perf_counter()
    anchor, mask, qpos, fused, details = prepare(args)
    if args.inference is None:
        print(json.dumps(details, default=json_default), flush=True)
        return
    backend = RetainedInference(args.inference)
    # Frozen arrays must describe this exact target and reviewed model.
    if args.model_manifest is None:
        raise ValueError("Retained inference requires its producer --model-manifest")
    manifest = json.loads(args.model_manifest.read_text())
    spec = GraspGenXPointGraspModelSpec(
        bundle_root=args.inference.parent, model_profile_id=manifest["model_profile_id"],
        source_revision=manifest["source"]["revision"],
        checkpoint_revision=manifest["checkpoint_repository"]["revision"],
        **{name+"_sha256": manifest["files"][name]["sha256"] for name in
           ("generator_config", "generator_checkpoint", "discriminator_config", "discriminator_checkpoint")})
    contexts, selectors = {}, {}
    cut = SimpleNamespace(qpos_by_side=qpos, measured_at=details["qpos_measured_at"])
    for side in args.sides:
        side_out = args.output / side
        side_out.mkdir()
        profile = profile_for(args, side)
        builder = build_exclusive_rm65_skill_target_builder(profile)
        target_safety = bind_exclusive_rm65_skill_target_safety(
            profile=profile, target_builder=builder)
        motion = RM65GraspJointTrajectoryPlanner(profile=profile, planning_target_builder=builder,
            target_envelope_filter=target_safety.envelope_filter,
            config=RM65GraspTrajectoryPlanningConfig(path_planning_time_budget_sec=args.plan_budget,
                trajectory_parameterization="continuous_c2", trajectory_speed_profile="far_4x_approach_1x",
                endpoint_candidate_ordering="estimated_execution_time"), trajectory_record_dir=side_out)
        frames = build_real_perception_station_frame_binding(collision_scene_profile=profile.collision_scene_profile,
            scene_mount_hash=profile.semantic_content_hash, scene_world_from_canonical_station=profile.scene_world_from_canonical_station)
        config = motion.reachability_config
        reach_profile = StationPointGraspArmReachabilityProfile(arm_side=side,
            wrist_camera_side_axis_tool=load_rm65_wrist_camera_side_axis_tool(
                camera_mount_urdf_path=args.urdf, side=side,
                tool_tcp_pose_flange_m_rad=profile.tool_tcp_pose_flange_m_rad_by_side[side]),
            evaluator_profile_id="rm65.global_fk_multistart_reachability.v1", evaluator_profile_hash=config.semantic_content_hash,
            solver_reference_frame_id=frames.source_frame_id, solver_reference_frame_spec_hash=frames.source_frame_spec_hash,
            frame_graph_hash=frames.frame_graph.semantic_content_hash, time_budget_ms=config.time_budget_ms,
            max_solve_cycles=config.max_solve_cycles, standard_position_tolerance_m=config.standard_position_tolerance_m,
            standard_orientation_tolerance_deg=config.standard_orientation_tolerance_deg,
            basic_position_tolerance_m=config.basic_position_tolerance_m, basic_orientation_tolerance_deg=config.basic_orientation_tolerance_deg)
        origin = arm_base_origin_in_candidate_frame(scene_world_from_candidate=profile.scene_world_from_canonical_station,
            scene_world_from_arm_base=profile.scene_world_from_arm_base_by_side[side])
        adapter = RM65PointGraspReachabilityAdapter(profile=reach_profile, planner=motion, motion_port=None,
            frame_graph=frames.frame_graph, candidate_reference_frame_id=frames.reference_frame_id, arm_base_origin_candidate_m=origin)
        contexts[side] = SimpleNamespace(profile=profile, motion=motion, origin=origin,
            output=side_out, envelope_profile_id=target_safety.envelope_profile.profile_id)
        def select(candidates, target, maximum, *, adapter=adapter):
            return adapter.select_candidates(candidates, target, maximum,
                measured_qpos_cut=cut, measurement_reference_time=anchor.acquired_at)
        selectors[side] = select

    # Only arm kinematics/admission differ. Refuse to share geometry if the
    # selected component contexts disagree about the observation frame.
    world_from_station = contexts[args.sides[0]].profile.scene_world_from_canonical_station
    if any(not np.array_equal(context.profile.scene_world_from_canonical_station,
        world_from_station) for context in contexts.values()):
        raise ValueError("Shared candidates require the same canonical_station frame")

    def scene_factory(**kwargs):
        camera_from_station = np.linalg.inv(anchor.reference_from_camera)
        return VisibleGraspScene(points=fused.scene.points @ camera_from_station[:3, :3].T + camera_from_station[:3, 3],
            target_mask=fused.scene.target_mask, geometry=kwargs["geometry"])

    point = Sam2GraspGenXPointGraspPlanner(model_spec=spec,
        point_segmenter=SimpleNamespace(segment_point=lambda *_a, **_k: mask.copy()), client=backend,
        alignment_profile_id=getattr(args, "graspgenx_alignment_profile", DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID), enable_finger_exchange=True,
        orientation_variants_local_axis_degrees=GRASPGENX_DMTAC_REACHABILITY_ORIENTATION_VARIANTS_LOCAL_AXIS_DEG,
        preferred_approach_origins_base_m=tuple(context.origin for context in contexts.values()))
    shared, perception_error = None, None
    point.start()
    try:
        print("[shared] one physical candidate pool, parameterized per-arm local-chain checks", flush=True)
        shared = point.plan_for_components(Image.fromarray(anchor.rgb), anchor.depth_m,
            pixel_uv=tuple(args.pixel), intrinsics=anchor.intrinsics,
            base_from_camera=anchor.reference_from_camera, max_candidates=5,
            use_visible_target_points=True, visible_scene_factory=scene_factory,
            candidate_selectors=selectors, component_sides={side: side for side in args.sides})
    except RuntimeError as exc:
        perception_error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        point.close()
        shared_report = {
            "hardware_connected": False, "motion_executed": False,
            "timing_ms": point.last_plan_timing_ms, "filter_summary": point.last_filter_summary,
            "inference_calls": backend.inference_calls,
            "component_candidate_counts": ({side: len(value.candidates)
                for side, value in shared.results_by_component.items()} if shared else {}),
            "component_selection_timing_ms": dict(shared.selection_timing_ms_by_component) if shared else {},
            "candidate_pool": [asdict(candidate) for candidate in shared.candidate_pool] if shared else [],
            "error": perception_error,
        }
        (args.output / "shared_candidates.json").write_text(json.dumps(shared_report, indent=2, default=json_default)+"\n")

    scene = VisibleGraspScene(points=fused.scene.points @ world_from_station[:3, :3].T + world_from_station[:3, 3],
        target_mask=fused.scene.target_mask, geometry=fused.scene.geometry, capture_qpos_by_side=qpos)
    for side, context in contexts.items():
        motion, profile, side_out = context.motion, context.profile, context.output
        grasps = shared.results_by_component[side] if shared else None
        summary = dict(shared.filter_summary_by_component[side]) if shared else dict(point.last_filter_summary)
        selection_diagnostics = {key: value for key, value in summary.items() if key.startswith("arm_shortlist_")}
        result = {"side": side, "anchor": args.anchor, "arm_base_origin_station": context.origin,
            "motion_executed": False, "candidates": [asdict(c) for c in grasps.candidates] if grasps else [],
            "plans": [], "envelope_bound": True, "envelope_profile_id": context.envelope_profile_id,
            "shared_candidate_pool": "../shared_candidates.json", "shortlist": selection_diagnostics,
            "filter_summary": summary,
            "timing_ms": {"arm_candidate_selection": shared.selection_timing_ms_by_component[side]} if shared else {}}
        if perception_error is not None:
            result["error"] = perception_error
        start_check = motion.oracle.check(qpos)
        result["start_state_check"] = {
            "safe": start_check.safe, "collision_check_available": start_check.collision_check_available,
            "reason": start_check.reason, "minimum_distance_m": start_check.minimum_distance_m,
            "metadata": dict(start_check.metadata)}
        if grasps and grasps.candidates:
            planning_order = grasp_planning_candidate_ordinals(summary, candidate_count=len(grasps.candidates))
            if planning_order is None:
                planning_order = tuple(range(len(grasps.candidates)))
            result["planning_candidate_ordinals"] = planning_order
            for index in planning_order[:args.plan_candidates]:
                candidate = grasps.candidates[index]
                plan_started = time.perf_counter()
                print(f"[{side}] full path planning candidate={index}", flush=True)
                try:
                    trajectory = motion(qpos_by_side=qpos, side=side,
                        target_tool_tcp_world=world_from_station @ np.asarray(candidate.base_from_tcp).reshape(4, 4),
                        speed_scale=.20, visible_scene=scene)
                    fixed_side = "right" if side == "left" else "left"
                    fixed_error = float(np.max(np.abs(trajectory.qpos_by_side[fixed_side]-qpos[fixed_side])))
                    if fixed_error > 1e-12:
                        raise RuntimeError("Single-arm planning modified the other arm")
                    result["plans"].append({"candidate": index, "status": "path_found",
                        "elapsed_sec": time.perf_counter()-plan_started,
                        "trajectory_samples": len(trajectory.time_from_start_sec),
                        "duration_sec": float(trajectory.time_from_start_sec[-1]),
                        "stationary_arm_max_change_rad": fixed_error, "trajectory_hash": trajectory.semantic_content_hash})
                    break
                except RM65GraspTrajectoryPlanningError as exc:
                    result["plans"].append({"candidate": index, "status": "rejected", "reason": exc.reason_code,
                        "details": exc.details, "elapsed_sec": time.perf_counter()-plan_started})
        result["elapsed_sec"] = sum(plan["elapsed_sec"] for plan in result["plans"]) + result["timing_ms"].get("arm_candidate_selection", 0.) / 1000.
        result["status"] = ("perception_failed" if perception_error else
            "path_found" if any(p["status"] == "path_found" for p in result["plans"]) else
            "no_path_found" if result["candidates"] else "no_admitted_candidate")
        if args.diagnose_endpoints and shared and shared.candidate_pool:
            # Extra diagnostic solves never change admission or its budget.
            attempts = json.loads(selection_diagnostics.get("arm_shortlist_attempts_json", "[]"))
            indices = dict.fromkeys([0] + [int(a["index"]) for a in attempts[:1]])
            result["endpoint_diagnostics"] = []
            for index in indices:
                target = world_from_station @ np.asarray(shared.candidate_pool[index].base_from_tcp).reshape(4, 4)
                try:
                    endpoints = motion._endpoint_qpos_candidates(qpos_by_side=qpos, side=side,
                        target_tool_tcp_world=target, prefer_closest_tolerance=False,
                        deadline=time.perf_counter()+2.0)
                    diagnostic = {"index": index, "status": "endpoint_found", "branch_count": len(endpoints)}
                except RM65GraspTrajectoryPlanningError as exc:
                    diagnostic = {"index": index, "status": "rejected", "reason": exc.reason_code, "details": exc.details}
                result["endpoint_diagnostics"].append(diagnostic)
        (side_out / "result.json").write_text(json.dumps(result, indent=2, default=json_default)+"\n")
        print(json.dumps({"side": side, "candidates": len(result["candidates"]),
            "plans": [{k: p[k] for k in ("candidate", "status", "elapsed_sec")} for p in result["plans"]],
            "error_type": result.get("error", {}).get("type"), "details": str(side_out / "result.json"),
            "elapsed_sec": result["elapsed_sec"]}), flush=True)
    print(json.dumps({"shared_physical_candidates": len(shared.candidate_pool) if shared else 0,
        "inference_calls": backend.inference_calls, "total_elapsed_sec": time.perf_counter()-started}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graspgenx-alignment-profile",
        choices=DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_IDS, default=DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID,
        help="Explicit nominal alignment; v5 is restricted to the right-arm q0 calibration.")
    for name in ("repo", "canonical", "capture", "mask", "urdf", "output", "bundle"):
        p.add_argument("--"+name, required=True, type=Path)
    p.add_argument("--inference", type=Path)
    p.add_argument("--captured-robot-filter-config", type=Path)
    p.add_argument("--captured-robot-observation-urdf", type=Path)
    p.add_argument("--captured-robot-self-envelope-urdf", type=Path)
    p.add_argument("--model-manifest", type=Path)
    p.add_argument("--aux-segmentation", type=Path, action="append", default=[],
                   help="Saved real segment probe output; repeat for each auxiliary view")
    p.add_argument("--refine-target-depth", action="store_true")
    p.add_argument("--diagnose-endpoints", action="store_true")
    p.add_argument("--collision-scene-profile", choices=(
        "dual_arm_hand_proxy_dmtac_strict", "dual_arm_hand_proxy_dmtac_grasp"),
        default="dual_arm_hand_proxy_dmtac_grasp",
        help="Select the grasp CAD table or unchanged strict scene for matched replay")
    p.add_argument("--anchor", choices=("main", "wrist_left", "wrist_right"), required=True)
    p.add_argument("--pixel", nargs=2, type=int, required=True)
    p.add_argument("--frame", type=int, default=2)
    p.add_argument("--sides", nargs="+", choices=("left", "right"), default=("left", "right"))
    p.add_argument("--plan-candidates", type=int, default=2)
    p.add_argument("--plan-budget", type=float, default=12)
    replay(p.parse_args())


if __name__ == "__main__":
    main()
