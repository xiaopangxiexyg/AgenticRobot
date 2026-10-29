"""Read-only relation queries on one validated saved RGB-D/qpos cut.

This probe is an offline consumer of the production surface selector, relation
solver and safe planner. It creates no Gateway, camera, backend or dispatcher.
Every query starts at the same measured qpos; a planned endpoint never becomes
the next observation. Live execution remains the existing admitted Gateway.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import threading
import time

import numpy as np

from agent_gateway.profiles.real.motion_observed_scene import obstacle_cloud_from_captures
from agent_gateway.profiles.real.motion_primitive_profiles import RealAgentPrimitiveConvergenceProfile
from agent_gateway.profiles.real.realsense_perception import build_real_perception_source_binding
from agent_gateway.profiles.surface_anchor_selection import select_mask_interior_pixel
from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.grasp_multiview_snapshot import decode_grasp_multiview_snapshot_arrays
from robot_motion.observed_scene import SafeMotionObservationConfig
from robot_motion.pose import pose_delta_metrics
from robot_motion.relative_action import pose_matrix_to_xyzquat, xyzquat_to_pose_matrix
from robot_motion.spatial_goals import compute_spatial_goal, finite_vector
from robot_motion.spatial_goal_regions import generate_relation_region_goals, generate_proximity_region_goals
from robot_motion.tool_entity_goals import (
    HOVER_ORIENTATION_POLICIES, TOOL_ENTITY_HOVER_SEMANTICS_V1, ToolEntityBox, hover_orientation,
)
# Shared with the RealMan and MuJoCo Gateways; re-exported for existing callers.
from robot_motion.tool_entity_scene import (  # noqa: F401
    ENTITY_HOVER_CANDIDATE_BUDGET_SEC, ENTITY_HOVER_SUPPORT_WINDOW_M, OBJECT_MASK_EROSION_PX,
    SURFACE_AMBIGUITY_MINIMUM_VOXELS,
    SURFACE_AMBIGUITY_SHARE, MeasuredObjectSurface, describe_tool_object_relations as _describe_relations,
    entity_hover_proposals, hover_candidate_relation, object_surface_from_mask, pose_matrix as _pose_matrix,
    validate_entity_hover_endpoint,
)
from deployments.perception.dmtac_tool_entity import (  # noqa: F401
    default_tool_entity_by_side, dmtac_open_sensor_contact_solids,
)
from scripts.probes.inspect_safe_move_snapshot import (
    RM65CapturedRobotObservationFilterConfig, backproject, load_captures, resolve_inputs,
)
from teleop_runtime.skill_deploy.rm65_grasp_trajectory import (
    RM65_GRASP_ENDPOINT_ORDER_JOINT_DISTANCE, RM65_GRASP_ENDPOINT_ORDER_MODES,
    RM65GraspJointTrajectoryPlanner, RM65GraspTrajectoryPlanningConfig,
    RM65GraspTrajectoryPlanningError,
)
from teleop_runtime.skill_deploy.rm65_lower_assembly import bind_exclusive_rm65_skill_target_safety
from scipy.spatial.transform import Rotation

class _RegionProposal:
    """One candidate pose with its audit relation and selection fields."""

    __slots__ = ("pose", "relation", "selection")

    def __init__(self, *, pose, relation, selection):
        self.pose, self.relation, self.selection = np.asarray(pose, dtype=float), relation, selection


def _snapshot_bounds(value):
    """Detach caller-owned bounds once and keep one immutable internal view."""
    def capture(item):
        if isinstance(item, np.ndarray):
            frozen, audit = capture(item.tolist())
            return frozen, np.asarray(audit, dtype=item.dtype).copy()
        if isinstance(item, (list, tuple)):
            parts = [capture(part) for part in item]
            frozen = tuple(part[0] for part in parts)
            audit = [part[1] for part in parts]
            return frozen, tuple(audit) if isinstance(item, tuple) else audit
        return item, item
    return capture(copy.deepcopy(value))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_inputs(manifest: dict) -> None:
    """Check the declared freeze before creating native planning resources."""
    if manifest.get("schema") != "agenticrobot.saved_scene_query.v1":
        raise ValueError("Expected a saved-scene query manifest v1.")
    if type(manifest.get("source_single_start")) is not bool:
        raise ValueError("The recorded source startup policy must be explicit.")
    paths = [Path(manifest["replay_dir"]) / "snapshot.json", Path(manifest["urdf"])]
    paths.extend(Path(manifest["context_dir"]) / name for name in (
        "camera_mount_urdf.urdf", "camera_config.json", "camera_registry.json",
        "left_tool_profile.json", "right_tool_profile.json", "envelope_profile.json",
        "arm_ip_assignment.json"))
    if manifest.get("filter_config") is not None:
        paths.append(Path(manifest["filter_config"]))
    frozen = manifest["fixed_files"]
    for path in paths:
        if str(path) not in frozen:
            raise ValueError(f"Required input is not frozen: {path}")
    for name, expected in frozen.items():
        if file_sha256(Path(name)) != expected:
            raise ValueError(f"Saved-scene input changed: {name}")
    snapshot = json.loads(paths[0].read_text())
    if snapshot.get("schema") != "agenticrobot.safe_move_inputs.v1":
        raise ValueError("Export the sample with the existing validated snapshot loader first.")
    root = Path(manifest["replay_dir"]).resolve()
    for view in snapshot["views"]:
        path = (root / view["file"]).resolve()
        if not path.is_relative_to(root) or str(path) not in frozen:
            raise ValueError("Each RGB-D frame must be inside the frozen replay directory.")
    for anchor in manifest.get("anchors", []):
        if frozen.get(anchor["mask_path"]) != anchor["mask_sha256"]:
            raise ValueError("Each saved mask must be part of this input freeze.")
    if snapshot.get("operator_scene_sample_id"):
        record_path = Path(manifest["source_record"])
        if str(record_path) not in frozen:
            raise ValueError("The original operator record must be frozen.")
        record = json.loads(record_path.read_text())
        if (record.get("schema") != "rpent.operator_scene_sample.v1"
                or record.get("sample_id") != snapshot["operator_scene_sample_id"]
                or record.get("status") != "complete" or record.get("context_complete") is not True):
            raise ValueError("Export and accepted operator record differ.")
        for item in record["context"]:
            relative = Path(item["file"])
            if relative.parts != ("context", relative.name):
                raise ValueError("Invalid recorded context path.")
            path = Path(manifest["context_dir"]) / relative.name
            if frozen.get(str(path)) != item["sha256"]:
                raise ValueError("Context differs from the original operator capture.")
        artifact = record["artifacts"][record["observation_artifact_index"]]
        archive = record_path.parent / artifact["file"]
        if (not archive.resolve().is_relative_to(record_path.parent.resolve())
                or frozen.get(str(archive)) != artifact["sha256"]):
            raise ValueError("Original RGB-D observation archive is not frozen.")
        decoded, _, _, _ = decode_grasp_multiview_snapshot_arrays(archive.read_bytes())
        recorded = {metadata["role"]: (arrays, metadata) for arrays, metadata in decoded}
        states = [(a, m) for a, m in decoded if a["left_qpos"].shape == (6,)]
        if not states or snapshot.get("request"):
            raise ValueError("Operator export needs its original measured qpos, without an invented planning request.")
        for view in snapshot["views"]:
            arrays, metadata = recorded[view["role"]]
            state_arrays, state_metadata = ((arrays, metadata) if arrays["left_qpos"].shape == (6,)
                else min(states, key=lambda pair: abs(pair[1]["acquired_at"] - metadata["acquired_at"])))
            measurement = state_metadata.get("measured_qpos", state_metadata.get("camera_pose", {}))
            if (any(not np.array_equal(view["capture_qpos_by_side"][s], state_arrays[s + "_qpos"])
                    for s in ("left", "right"))
                    or view["pose_measured_at"] != measurement.get("measured_at")
                    or view["pose_available_at"] != measurement.get("available_at")
                    or tuple(view["pose_evidence_refs"]) != tuple(measurement.get("source_evidence_refs", ()))):
                raise ValueError("Replay qpos or measurement identity differs from the original observation.")
            if (view["acquired_at"] != metadata["acquired_at"]
                    or any(view[key] != metadata[key] for key in
                           ("source_binding_hash", "station_revision", "source_generation", "frame_index"))
                    or view["rgb_content_hash"] != metadata["rgb_content_hash"]
                    or view["depth_content_hash"] != metadata["depth_content_hash"]
                    or view["intrinsics"] != metadata["camera_intrinsics"]
                    or not np.array_equal(np.asarray(view["T_world_from_camera"]).reshape(4, 4),
                                          np.asarray(metadata["T_source_from_camera"]).reshape(4, 4))):
                raise ValueError("Replay frame geometry differs from the original observation.")
            with np.load(root / view["file"], allow_pickle=False) as replay:
                if any(not np.array_equal(replay[key], arrays[key]) for key in ("rgb", "depth_m")):
                    raise ValueError("Replay RGB-D arrays differ from the original observation.")


def capture_inputs(snapshot, captures, *, reference_frame_id, source_binding_hash):
    """Reconcile recorded identities, policy and original planning q0."""
    if snapshot["reference_frame_id"] != reference_frame_id:
        raise ValueError("Snapshot and planning reference frames differ.")
    config = SafeMotionObservationConfig(**snapshot["configuration"])
    for capture in captures:
        if capture.source_binding_hash != source_binding_hash:
            raise ValueError("Recorded camera binding differs from the frozen context.")
        for key, array, domain, extra in (
            ("rgb_content_hash", capture.rgb, "rgb", {}),
            ("depth_content_hash", capture.depth_m, "depth", {"unit": "meter", "invalid_value": 0.0}),
        ):
            dtype = "uint8" if domain == "rgb" else "float32"
            if array.dtype != np.dtype(dtype):
                raise ValueError("Saved RGB-D dtype differs from its source encoding.")
            actual = canonical_content_hash({"dtype": dtype, "shape": list(array.shape),
                "byte_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(), **extra},
                domain="dexteleop.real_aligned_rgbd." + domain + "_content.v1")
            if actual != getattr(capture, key):
                raise ValueError("Saved RGB-D content differs from its recorded frame identity.")
    request_qpos = snapshot.get("request", {}).get("qpos_by_side")
    if request_qpos is not None:
        qpos_source = "saved_planning_request"
    elif snapshot.get("operator_scene_sample_id") and snapshot.get("source_capture_status") == "complete":
        # The existing operator-sample loader records a bounded stationary
        # qpos association when no planning request existed at capture time.
        request_qpos = captures[0].capture_qpos_by_side
        qpos_source = "operator_sample_stationary_association"
    else:
        raise ValueError("Snapshot has neither a saved planning q0 nor an accepted operator sample.")
    if set(request_qpos) != {"left", "right"}:
        raise ValueError("Saved planning q0 must contain both arms.")
    qpos = {side: finite_vector(list(request_qpos[side]), 6, name=side).copy()
            for side in ("left", "right")}
    return config, qpos, qpos_source


class SavedSceneSkillQuery:
    """One immutable saved scene with short references and a read-only planner.

    The explicit factory validates files, calibration and observation geometry.
    The constructor is also a narrow injection seam for no-device contract tests.
    No live freshness claim or execution authority can be produced by this API.
    """

    def __init__(self, *, scene_id, reference_frame_id, qpos_by_side,
                 tcp_pose_by_side, anchors, planner, cloud, source_evidence,
                 query_budget=None, region_candidate_policy="proximity", clock=time.monotonic,
                 tool_entity_by_side=None, object_surface_by_reference=None,
                 object_surface_tcp_pose_by_reference=None):
        self._scene_id = str(scene_id)
        self._frame = reference_frame_id
        self._tool_entities = dict(tool_entity_by_side or {})
        if (not set(self._tool_entities) <= {"left", "right"}
                or not all(isinstance(e, ToolEntityBox) for e in self._tool_entities.values())):
            raise ValueError("Tool entities must be ToolEntityBox values keyed by arm.")
        self._qpos = {side: finite_vector(list(qpos_by_side[side]), 6, name=side).copy()
                      for side in ("left", "right")}
        self._tcp = {side: tuple(finite_vector(list(tcp_pose_by_side[side]), 7, name=side))
                     for side in ("left", "right")}
        for pose in self._tcp.values():
            if abs(math.hypot(*pose[3:]) - 1.) > 1e-5:
                raise ValueError("Saved TCP orientation must be a unit quaternion.")
        self._anchors = copy.deepcopy(anchors)
        if len({item["reference"] for item in self._anchors}) != len(self._anchors):
            raise ValueError("Saved observation references must be unique.")
        self._refs = {item["reference"]: item for item in self._anchors}
        self._object_surfaces = {}
        for reference, points in dict(object_surface_by_reference or {}).items():
            if reference not in self._refs or self._refs[reference]["kind"] != "saved_visible_surface_point":
                raise ValueError("Object surfaces must belong to saved surface references.")
            points = np.array(points, dtype=float).reshape(-1, 3)
            if not len(points) or not np.isfinite(points).all():
                raise ValueError("Object surfaces must be nonempty finite N x 3 arrays.")
            points.setflags(write=False)
            self._object_surfaces[reference] = points
        # Each view's gripper pose at that frame's recorded capture qpos (the
        # relation is a property of that frame, not of the planning start).
        self._surface_tcp = {}
        for reference, poses in dict(object_surface_tcp_pose_by_reference or {}).items():
            if reference not in self._object_surfaces or set(poses) != {"left", "right"}:
                raise ValueError("Capture TCP poses must name a measured surface and both arms.")
            checked = {}
            for side, pose in poses.items():
                pose = tuple(finite_vector(list(pose), 7, name="capture TCP pose"))
                if abs(math.hypot(*pose[3:]) - 1.) > 1e-5:
                    raise ValueError("Capture TCP orientation must be a unit quaternion.")
                checked[side] = pose
            self._surface_tcp[reference] = checked
        views = [(self._refs[r]["label"], self._refs[r]["role"]) for r in self._object_surfaces]
        if len(set(views)) != len(views):
            # Fusion treats one label as one physical object seen from several views.
            raise ValueError("Each measured object label may have at most one surface per camera role.")
        self._relation_reads = []
        self._planner, self._cloud = planner, cloud
        self._source_evidence = copy.deepcopy(source_evidence)
        self._lock = threading.Lock()
        self._queries = []
        self._regions = []
        self._trajectories = {}
        self._closed = False
        self._clock = clock
        if region_candidate_policy not in {"halton", "proximity"}:
            raise ValueError("Unknown region candidate policy.")
        self._region_candidate_policy = region_candidate_policy
        self._planning_calls = 0
        self._planning_seconds = 0.
        if query_budget is not None:
            if (set(query_budget) not in ({"maximum_calls", "maximum_planning_sec"},
                    {"maximum_calls", "maximum_planning_sec", "maximum_call_planning_sec"})
                    or type(query_budget["maximum_calls"]) is not int
                    or not 1 <= query_budget["maximum_calls"] <= 64
                    or type(query_budget["maximum_planning_sec"]) not in (int, float)
                    or not math.isfinite(query_budget["maximum_planning_sec"])
                    or not 0 < query_budget["maximum_planning_sec"] <= 120.):
                raise ValueError("Query budget must have 1..64 calls and 0..120 finite seconds.")
            if "maximum_call_planning_sec" in query_budget:
                value = query_budget["maximum_call_planning_sec"]
                if (type(value) not in (int, float) or not math.isfinite(value)
                        or not 0 < value <= query_budget["maximum_planning_sec"]):
                    raise ValueError("Per-call budget must be positive and within the session budget.")
        self._query_budget = None if query_budget is None else dict(query_budget)

    @classmethod
    def from_manifest(cls, manifest, *, enable_inherited_start_clearance=True,
                      endpoint_candidate_ordering=RM65_GRASP_ENDPOINT_ORDER_JOINT_DISTANCE):
        # The reviewed positive-start egress mode is this offline consumer's
        # default (user decision 2026-10-29). Strict 20 mm start admission
        # remains available with an explicit False; a manifest cannot choose.
        # Endpoint ordering selects among the planner's existing reviewed
        # modes (the RealMan Gateway CLI deploys estimated_execution_time).
        if endpoint_candidate_ordering not in RM65_GRASP_ENDPOINT_ORDER_MODES:
            raise ValueError("endpoint_candidate_ordering must be a reviewed planner mode.")
        if type(enable_inherited_start_clearance) is not bool:
            raise ValueError("enable_inherited_start_clearance must be bool.")
        verify_inputs(manifest)
        context, urdf = Path(manifest["context_dir"]), Path(manifest["urdf"])
        config = (None if manifest.get("filter_config") is None else
                  RM65CapturedRobotObservationFilterConfig.from_json_file(Path(manifest["filter_config"])))
        profile, builder, binding, geometry, cleaner = resolve_inputs(context, urdf, filter_config=config)
        bind_exclusive_rm65_skill_target_safety(profile=profile, target_builder=builder)
        if not callable(builder.pre_ik_target_safety_filter):
            raise RuntimeError("Saved deployment envelope is not bound.")
        captures = load_captures(Path(manifest["replay_dir"]), profile, binding, geometry)
        snapshot = json.loads((Path(manifest["replay_dir"]) / "snapshot.json").read_text())
        if manifest["source_single_start"]:
            # Reconstruct only the declared source identity. This factory does
            # not start or read cameras; no alternative is tried on mismatch.
            binding = build_real_perception_source_binding(
                camera_config_path=context / "camera_config.json",
                camera_registry_path=context / "camera_registry.json",
                scene_world_from_arm_base_by_side=profile.scene_world_from_arm_base_by_side,
                single_start=True)
        observation_config, qpos, qpos_source = capture_inputs(snapshot, captures,
            reference_frame_id=profile.station_world_frame_id,
            source_binding_hash=binding.semantic_content_hash)
        cloud = obstacle_cloud_from_captures(captures=captures,
            config=observation_config, reference_frame_id=profile.station_world_frame_id,
            self_filter=cleaner, planning_qpos_by_side=qpos, max_inter_camera_skew_sec=.1)
        tcp = builder.tool_tcp_world_targets_for_qpos_pair(left_qpos=qpos["left"], right_qpos=qpos["right"])
        poses = {side: pose_matrix_to_xyzquat(pose).tolist() for side, pose in tcp.items()}
        scene_id = canonical_content_hash({"fixed_files": manifest["fixed_files"],
            "anchors": manifest.get("anchors", []), "source_single_start": manifest["source_single_start"],
            "planning_qpos": qpos,
            **({"start_clearance_mode": "inherited-positive-v1"}
               if enable_inherited_start_clearance else {}),
            **({"endpoint_candidate_ordering": endpoint_candidate_ordering}
               if endpoint_candidate_ordering != RM65_GRASP_ENDPOINT_ORDER_JOINT_DISTANCE else {})},
            domain="agenticrobot.saved_scene_query.identity.v1")
        prefix = "saved:" + scene_id[:16]
        anchors = [{"reference": prefix + ":tcp_" + side, "label": side + " TCP",
                    "kind": "measured_tcp", "world_xyz_m": poses[side][:3],
                    "acquired_at": captures[0].pose_measured_at, "role": None}
                   for side in ("left", "right")]
        surfaces, surface_tcp = {}, {}
        for index, saved in enumerate(manifest.get("anchors", [])):
            capture = next((c for c in captures if c.role == saved["role"]), None)
            if (capture is None or capture.acquired_at != saved["acquired_at"]
                    or capture.rgb_content_hash != saved["rgb_content_hash"]
                    or capture.depth_content_hash != saved["depth_content_hash"]):
                raise ValueError("Saved mask and RGB-D must name exactly the same frame.")
            with np.load(saved["mask_path"], allow_pickle=False) as data:
                mask = data["mask"]
            uv, clearance = select_mask_interior_pixel(mask, capture.depth_m,
                requested_pixel_uv=None, maximum_depth_m=observation_config.maximum_depth_m)
            selected = np.zeros(mask.shape, dtype=bool)
            selected[uv[1], uv[0]] = True
            points, _, _ = backproject(capture.depth_m, capture.intrinsics,
                capture.source_from_camera, selected)
            if points.shape != (1, 3) or not np.isfinite(points).all():
                raise ValueError("Surface anchor must be one measured finite RGB-D point.")
            reference = prefix + ":surface_" + str(index)
            surface, surface_evidence = object_surface_from_mask(mask, capture,
                maximum_depth_m=observation_config.maximum_depth_m)
            if surface is not None:
                surfaces[reference] = surface
                capture_tcp = builder.tool_tcp_world_targets_for_qpos_pair(
                    left_qpos=np.asarray(capture.capture_qpos_by_side["left"], dtype=float),
                    right_qpos=np.asarray(capture.capture_qpos_by_side["right"], dtype=float))
                surface_tcp[reference] = {side: pose_matrix_to_xyzquat(pose).tolist()
                                          for side, pose in capture_tcp.items()}
            anchors.append({"reference": reference,
                "label": saved["label"], "kind": "saved_visible_surface_point",
                "world_xyz_m": points[0].tolist(), "acquired_at": capture.acquired_at,
                "role": capture.role, "pixel_uv": uv, "pixel_clearance_px": clearance,
                "mask_sha256": saved["mask_sha256"], "rgb_content_hash": capture.rgb_content_hash,
                "depth_content_hash": capture.depth_content_hash, "object_surface": surface_evidence})
        convergence = RealAgentPrimitiveConvergenceProfile()
        plan_config = RM65GraspTrajectoryPlanningConfig(
            enable_inherited_start_clearance=enable_inherited_start_clearance,
            endpoint_candidate_ordering=endpoint_candidate_ordering).for_tcp_pose_move(
            translation_tolerance_m=convergence.translation_tolerance_m,
            rotation_tolerance_rad=convergence.rotation_tolerance_rad)
        planner = RM65GraspJointTrajectoryPlanner(profile=profile,
            planning_target_builder=builder, target_envelope_filter=builder.pre_ik_target_safety_filter,
            config=plan_config)
        tool_entities = default_tool_entity_by_side(profile.tool_tcp_pose_flange_m_rad_by_side,
                                                     profile.tool_profile_by_side, urdf_path=urdf)
        return cls(scene_id=scene_id, reference_frame_id=profile.station_world_frame_id,
            qpos_by_side=qpos, tcp_pose_by_side=poses, anchors=anchors, planner=planner, cloud=cloud,
            source_evidence={"fixed_files": dict(manifest["fixed_files"]),
                "source": "saved_rgbd_and_measured_qpos", "device_connections": 0,
                "qpos_source": qpos_source, "observation_configuration": snapshot["configuration"],
                "planning_policy": {
                    "inherited_start_clearance": enable_inherited_start_clearance,
                    "ordinary_clearance_m": plan_config.minimum_clearance_m,
                    "endpoint_candidate_ordering": plan_config.endpoint_candidate_ordering},
                "observed_voxel_count": len(cloud.points_m),
                "enable_inherited_start_clearance": enable_inherited_start_clearance,
                "tool_entities": {side: entity.identity() for side, entity in tool_entities.items()}},
            query_budget=manifest.get("query_budget"),
            region_candidate_policy=manifest.get("region_candidate_policy", "proximity"),
            tool_entity_by_side=tool_entities, object_surface_by_reference=surfaces,
            object_surface_tcp_pose_by_reference=surface_tcp)

    def _require_open(self):
        if self._closed:
            raise RuntimeError("Saved-scene query has been closed.")

    def read_scene(self, *, include_geometry=False):
        self._require_open()
        public_keys = {"reference", "label", "kind", "acquired_at", "role"}
        if include_geometry:
            public_keys.add("world_xyz_m")
        result = {"scope": "one_saved_time_node_no_execution",
            "reference_frame_id": self._frame,
            "references": [{k: copy.deepcopy(v) for k, v in item.items() if k in public_keys}
                           for item in self._anchors],
            "observation_semantics": "saved_visible_surface_not_object_pose_or_tracking"}
        if include_geometry:
            result["measured_tcp_pose_by_side"] = {s: list(p) for s, p in self._tcp.items()}
        return result

    def describe_tool_object_relations(self):
        """Measured gripper-object relations at this saved time node; no planning.

        Semantics ``agenticrobot.relation.tool_object.v1``: each saved object
        surface (eroded mask, largest measured cluster) against each bound
        arm's gripper at the measured TCP: the maximum-opening screening solids
        with their closing region, plus any bound contact-only solids (the
        URDF's open DMTAC sensor boxes), reported as separate distances.
        ``objects`` fuses the camera views of one label (one physical object);
        ``views`` keeps each view with its surface-cluster counts. Unbound arms
        are listed, never silently omitted. The contact band is proximity
        evidence, never a contact, grasp or force claim.
        """
        self._require_open()
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("A query is already running for this saved scene.")
        try:
            self._require_open()
            surfaces = [MeasuredObjectSurface(reference=anchor["reference"], label=anchor["label"],
                            role=anchor["role"], points_m=self._object_surfaces[anchor["reference"]],
                            surface_cluster=anchor.get("object_surface") or {},
                            tcp_pose_by_side=self._surface_tcp.get(anchor["reference"]))
                        for anchor in self._anchors if anchor["reference"] in self._object_surfaces]
            shared = _describe_relations(surfaces=surfaces, tool_entity_by_side=self._tool_entities,
                                         fallback_tcp_pose_by_side=self._tcp)
            result = {"scope": "one_saved_time_node_no_execution",
                    "semantics": shared["semantics"],
                    "reference_frame_id": self._frame, "objects": shared["objects"], "views": shared["views"],
                    "evaluated_sides": shared["evaluated_sides"],
                    "unbound_sides": shared["unbound_sides"],
                    "gripper_geometry_by_side": shared["gripper_geometry_by_side"],
                    "unmeasured_references": [a["reference"] for a in self._anchors
                        if a["kind"] == "saved_visible_surface_point"
                        and a["reference"] not in self._object_surfaces],
                    "evidence_scope": shared["evidence_scope"]}
            self._relation_reads.append(copy.deepcopy(result))
            return result
        finally:
            self._lock.release()

    def plan_relation(self, *, side, target_reference, offset_xyz_m, offset_frame="reference"):
        self._require_open()
        if side not in self._tcp or target_reference not in self._refs:
            raise ValueError("Unknown arm or reference from another saved scene.")
        goal = compute_spatial_goal(current_tcp_pose_xyz_quat_xyzw=list(self._tcp[side]),
            anchor_xyz_m=self._refs[target_reference]["world_xyz_m"],
            offset_xyz_m=offset_xyz_m, offset_frame=offset_frame, orientation_mode="preserve")
        return self.plan_pose(side=side, pose_xyz_quat_xyzw=list(goal.pose_xyz_quat_xyzw),
            _relation={"target_reference": target_reference, "offset_xyz_m": list(offset_xyz_m),
                       "offset_frame": offset_frame})

    def plan_pose(self, *, side, pose_xyz_quat_xyzw, _relation=None):
        """Atomic baseline, with exactly the same observed scene and safety owner."""
        self._require_open()
        if side not in self._tcp:
            raise ValueError("Unknown arm.")
        pose = finite_vector(pose_xyz_quat_xyzw, 7, name="target pose")
        if abs(math.hypot(*pose[3:]) - 1.) > 1e-5:
            raise ValueError("Target orientation must be a unit quaternion.")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("A query is already running for this saved scene.")
        try:
            self._require_open()
            return self._plan_pose_locked(side=side, pose=pose, relation=_relation)
        finally:
            self._lock.release()

    def _plan_pose_locked(self, *, side, pose, relation=None, time_budget_sec=120.,
                          absolute_deadline=None):
        query_id = "plan:" + self._scene_id[:16] + ":" + str(len(self._queries))
        row = {"query_id": query_id, "side": side, "target_pose_xyz_quat_xyzw": pose.tolist(),
            "relation": copy.deepcopy(relation), "scope": "planned_not_executed",
            "physical_action_occurred": False}
        remaining = time_budget_sec
        if self._query_budget is not None:
            remaining = min(remaining, self._query_budget["maximum_planning_sec"] - self._planning_seconds)
            remaining = min(remaining, self._query_budget.get("maximum_call_planning_sec", remaining))
            if self._planning_calls >= self._query_budget["maximum_calls"]:
                remaining = 0.
        if remaining <= 0:
            row.update(status="query_incomplete", reason="query_budget_exhausted", path_planning_wall_sec=0.)
            self._queries.append(row)
            return self._public_result(row)
        started = self._clock()
        if absolute_deadline is not None:
            remaining = min(remaining, absolute_deadline - started)
            if remaining <= 0:
                row.update(status="query_incomplete", reason="query_budget_exhausted", path_planning_wall_sec=0.)
                self._queries.append(row)
                return self._public_result(row)
        self._planning_calls += 1
        try:
            trajectory = self._planner.plan_with_observed_cloud(self._cloud,
                qpos_by_side={s: q.copy() for s, q in self._qpos.items()}, side=side,
                target_tool_tcp_world=xyzquat_to_pose_matrix(pose), speed_scale=.2,
                require_grasp_approach=False, time_budget_sec=remaining,
                pose_goal_policy="within_arrival_tolerance")
            if self._clock() - started > remaining:
                raise RM65GraspTrajectoryPlanningError("path_planning_time_budget_exhausted",
                    details={"planning_phase": "query_return_budget"})
            last = {s: trajectory.qpos_by_side[s][-1] for s in ("left", "right")}
            actual = self._planner.tool_tcp_poses_for_qpos_pair(qpos_by_side=last)[side]
            translation, rotation = pose_delta_metrics(actual, xyzquat_to_pose_matrix(pose))
            row.update(status="path_found", reason="planned_not_executed",
                trajectory_hash=trajectory.semantic_content_hash,
                planner_evidence_refs=list(trajectory.planner_evidence_refs),
                trajectory_samples=len(trajectory.time_from_start_sec),
                trajectory_duration_sec=float(trajectory.time_from_start_sec[-1]),
                planned_endpoint_xyz_m=actual[:3, 3].tolist(),
                planned_endpoint_pose_xyz_quat_xyzw=[
                    *np.asarray(actual[:3, 3], dtype=float).tolist(),
                    *Rotation.from_matrix(np.asarray(actual[:3, :3], dtype=float)).as_quat().tolist()],
                final_translation_error_m=float(translation), final_rotation_error_deg=float(rotation))
            self._trajectories[query_id] = trajectory
        except RM65GraspTrajectoryPlanningError as error:
            row.update(status=("query_incomplete" if error.reason_code == "path_planning_time_budget_exhausted"
                               else "path_rejected"), reason=error.reason_code,
                       failure_details=str(error.details))
        except BaseException as error:
            row.update(status="query_failed", reason=type(error).__name__,
                       failure_details=str(error))
            raise
        finally:
            row["path_planning_wall_sec"] = self._clock() - started
            self._planning_seconds += row["path_planning_wall_sec"]
            self._queries.append(row)
        if row["status"] in {"path_found", "path_rejected"} and row["path_planning_wall_sec"] > remaining:
            row["outcome_before_budget_check"] = {"status": row["status"], "reason": row["reason"]}
            self._trajectories.pop(query_id, None)
            row.update(status="query_incomplete", reason="query_budget_exhausted")
        return self._public_result(row)

    @staticmethod
    def _public_result(row):
        return {k: copy.deepcopy(v) for k, v in row.items()
                if k not in {"target_pose_xyz_quat_xyzw", "relation", "failure_details",
                             "trajectory_hash", "planner_evidence_refs", "planned_endpoint_xyz_m",
                             "planned_endpoint_pose_xyz_quat_xyzw"}}

    def _run_region_with_lock(self, operation):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("A query is already running for this saved scene.")
        began = None
        budget_before = None
        try:
            began = self._clock()
            budget_before = self._planning_seconds
            return operation(began)
        except BaseException:
            # Initialization can fail before the search's own cleanup starts.
            # Charge its elapsed time without replacing the original exception
            # or double-counting work the search already recorded.
            if began is not None and budget_before is not None:
                try:
                    elapsed = self._clock() - began
                    if math.isfinite(elapsed) and elapsed >= 0.:
                        self._planning_seconds = max(self._planning_seconds, budget_before + elapsed)
                except BaseException:
                    pass
            raise
        finally:
            self._lock.release()

    def plan_relation_region(self, *, side, target_reference, offset_bounds_xyz_m,
                             offset_frame="reference", max_candidates=8):
        """Find the first certified path inside explicitly allowed endpoint bounds.

        Use the existing shared deterministic region sampler. The search owns
        no IK, collision exception or execution state. Each candidate starts at
        the same measured q0, and all attempts share one 20-second upper budget.
        An optional application query budget also covers atomic calls, so an
        Agent/Skill comparison cannot silently give one side extra native work.
        """
        self._require_open()
        if side not in self._tcp or target_reference not in self._refs:
            raise ValueError("Unknown arm or reference from another saved scene.")
        return self._run_region_with_lock(lambda began:
            self._plan_relation_region_locked(side=side, target_reference=target_reference,
                offset_bounds_xyz_m=offset_bounds_xyz_m, offset_frame=offset_frame,
                max_candidates=max_candidates, began=began))

    def plan_entity_hover_region(self, *, side, target_reference, tip_offset_bounds_xy_m,
                                 clearance_bounds_m, orientation_policy="preserve", max_candidates=8,
                                 tool_axis_max_tilt_rad=0.):
        """Hover the reviewed gripper box above a saved surface reference.

        Semantics ``agenticrobot.relation.tool_entity_hover.v1``: the centre of
        the gripper box's approach-side face lies inside the explicit
        horizontal offsets from the anchor, and the box's lowest point keeps a
        vertical clearance inside ``clearance_bounds_m`` above the highest
        observed point under its footprint (never below the anchor itself).
        Orientation is the explicit ``orientation_policy``; ``tool_axis_down``
        may allow a cone ``tool_axis_max_tilt_rad`` around reference -Z (least
        rotation into the cone; the achieved axis is re-checked against it,
        within the arrival rotation tolerance). Proposals reuse
        the same budgeted region loop and the same lower planner; the achieved
        FK pose is re-measured against the ORIGINAL bounds. Planning only.
        """
        self._require_open()
        if side not in self._tcp or target_reference not in self._refs:
            raise ValueError("Unknown arm or reference from another saved scene.")
        if side not in self._tool_entities:
            raise ValueError("No reviewed tool entity is bound for this arm.")
        if self._refs[target_reference].get("kind") != "saved_visible_surface_point":
            raise ValueError("Tool hover requires a saved visible surface reference.")
        if orientation_policy not in HOVER_ORIENTATION_POLICIES:
            raise ValueError("Unknown hover orientation policy.")
        # Validates the cone before any lock or native work.
        hover_orientation(self._tcp[side], orientation_policy, tool_axis_max_tilt_rad=tool_axis_max_tilt_rad)
        return self._run_region_with_lock(lambda began:
            self._plan_entity_hover_locked(side=side, target_reference=target_reference,
                tip_offset_bounds_xy_m=tip_offset_bounds_xy_m, clearance_bounds_m=clearance_bounds_m,
                orientation_policy=orientation_policy, max_candidates=max_candidates, began=began,
                tool_axis_max_tilt_rad=float(tool_axis_max_tilt_rad)))

    def _plan_entity_hover_locked(self, *, side, target_reference, tip_offset_bounds_xy_m,
                                  clearance_bounds_m, orientation_policy, max_candidates, began,
                                  tool_axis_max_tilt_rad=0.):
        anchor = np.asarray(self._refs[target_reference]["world_xyz_m"], dtype=float)
        entity = self._tool_entities[side]
        points = self._cloud.points_m
        margin = float(self._cloud.quantization_margin_m)
        window = ENTITY_HOVER_SUPPORT_WINDOW_M
        bounds = {}

        def summary_fields():
            bounds["tip"], tip_audit = _snapshot_bounds(tip_offset_bounds_xy_m)
            bounds["clearance"], clearance_audit = _snapshot_bounds(clearance_bounds_m)
            return {"semantics": TOOL_ENTITY_HOVER_SEMANTICS_V1,
                "tip_offset_bounds_xy_m": tip_audit, "clearance_bounds_m": clearance_audit,
                "orientation_policy": orientation_policy, "candidate_policy": "entity_proximity",
                "tool_axis_max_tilt_rad": tool_axis_max_tilt_rad,
                "support_window_m": window, "support_margin_m": margin,
                "candidate_budget_sec": ENTITY_HOVER_CANDIDATE_BUDGET_SEC,
                "tool_entity": entity.identity()}

        def generate():
            candidates = entity_hover_proposals(entity=entity, anchor_xyz_m=anchor.tolist(),
                current_tcp_pose_xyz_quat_xyzw=list(self._tcp[side]),
                tip_offset_bounds_xy_m=bounds["tip"], clearance_bounds_m=bounds["clearance"],
                points_m=points, support_margin_m=margin, orientation_policy=orientation_policy,
                max_candidates=max_candidates,
                boundary_inset_m=RealAgentPrimitiveConvergenceProfile().translation_tolerance_m,
                tool_axis_max_tilt_rad=tool_axis_max_tilt_rad)
            return [_RegionProposal(pose=candidate.pose_xyz_quat_xyzw,
                        relation={"entity_hover": hover_candidate_relation(candidate,
                            orientation_policy=orientation_policy,
                            tool_axis_max_tilt_rad=tool_axis_max_tilt_rad)},
                        selection={"selected_tip_offset_xy_m": list(candidate.tip_offset_xy_m),
                                   "selected_clearance_m": candidate.clearance_m})
                    for candidate in candidates]

        def validate(row):
            return validate_entity_hover_endpoint(entity=entity,
                endpoint_pose_xyz_quat_xyzw=row["planned_endpoint_pose_xyz_quat_xyzw"], anchor_xyz_m=anchor,
                points_m=points, support_margin_m=margin, tip_offset_bounds_xy_m=bounds["tip"],
                clearance_bounds_m=bounds["clearance"], orientation_policy=orientation_policy,
                tool_axis_max_tilt_rad=tool_axis_max_tilt_rad,
                rotation_tolerance_rad=RealAgentPrimitiveConvergenceProfile().rotation_tolerance_rad)

        return self._search_region_locked(side=side, target_reference=target_reference,
            summary_fields=summary_fields, began=began,
            hidden_fields=("tip_offset_bounds_xy_m", "clearance_bounds_m", "tool_entity"),
            generate=generate, validate=validate, candidate_budget_sec=ENTITY_HOVER_CANDIDATE_BUDGET_SEC)

    def _plan_relation_region_locked(self, *, side, target_reference, offset_bounds_xyz_m,
                                     offset_frame, max_candidates, began):
        anchor = self._refs[target_reference]["world_xyz_m"]
        bounds = {}

        def summary_fields():
            bounds["offset"], audit_bounds = _snapshot_bounds(offset_bounds_xyz_m)
            return {"offset_bounds_xyz_m": audit_bounds, "offset_frame": offset_frame,
                    "candidate_policy": self._region_candidate_policy}

        def generate():
            candidates = generate_relation_region_goals(
                current_tcp_pose_xyz_quat_xyzw=list(self._tcp[side]),
                anchor_xyz_m=anchor,
                offset_bounds_xyz_m=bounds["offset"], offset_frame=offset_frame,
                orientation_mode="preserve", tool_axis=None, axis_reference_xyz=None,
                tool_axis_rotation_bounds_rad=[0., 0.], max_candidates=max_candidates)
            if self._region_candidate_policy == "proximity":
                candidates = generate_proximity_region_goals(
                    current_tcp_pose_xyz_quat_xyzw=list(self._tcp[side]),
                    anchor_xyz_m=anchor,
                    offset_bounds_xyz_m=bounds["offset"], offset_frame=offset_frame,
                    max_candidates=max_candidates,
                    boundary_inset_m=RealAgentPrimitiveConvergenceProfile().translation_tolerance_m)
            return [_RegionProposal(pose=candidate.goal.pose_xyz_quat_xyzw,
                        relation={"offset_xyz_m": list(candidate.offset_xyz_m), "offset_frame": offset_frame},
                        selection={"selected_offset_xyz_m": list(candidate.offset_xyz_m)})
                    for candidate in candidates]

        def validate(row):
            actual_offset = np.asarray(row["planned_endpoint_xyz_m"]) - anchor
            if offset_frame == "tool":
                actual_offset = xyzquat_to_pose_matrix(self._tcp[side])[:3, :3].T @ actual_offset
            within = all(lo - 1e-9 <= x <= hi + 1e-9
                         for x, (lo, hi) in zip(actual_offset, bounds["offset"]))
            return {"actual_offset_xyz_m": actual_offset.tolist(), "inside_requested_region": within}, within

        return self._search_region_locked(side=side, target_reference=target_reference,
            summary_fields=summary_fields, began=began,
            hidden_fields=("offset_bounds_xyz_m",), generate=generate, validate=validate)

    def _search_region_locked(self, *, side, target_reference, summary_fields, hidden_fields, began,
                              generate, validate, candidate_budget_sec=3.):
        """Budgeted first-certified-path search shared by explicit region queries.

        ``generate`` returns ordered proposals; it runs inside the charged
        region budget. ``validate(row)`` checks a found plan's achieved FK
        against the ORIGINAL request and returns ``(record, inside)``. Budget,
        retry and revocation rules are identical for every query kind; only the
        per-candidate cap differs (3 s TCP regions, 5 s gripper-entity hover).
        """
        summary = None
        public_result = None
        query_id = None
        region_query_start = len(self._queries)
        region_calls_start = self._planning_calls
        existing_trajectory_ids = set(self._trajectories)
        region_budget_start = self._planning_seconds
        selection_keys = {"selected_candidate_index", "trajectory_duration_sec"}
        hidden = {"attempt_query_ids", "selected_plan_query_id", "candidate_validations", *hidden_fields}

        def drop_selection():
            for key in selection_keys:
                summary.pop(key, None)

        def fail_region(error):
            for trajectory_id in self._trajectories.keys() - existing_trajectory_ids:
                self._trajectories.pop(trajectory_id, None)
            if summary is not None:
                summary.update(status="query_failed", reason=type(error).__name__,
                    selected_plan_query_id=None,
                    attempted_candidate_count=self._planning_calls - region_calls_start,
                    attempt_query_ids=[row["query_id"] for row in self._queries[region_query_start:]])
                drop_selection()
        try:
            self._require_open()
            query_id = "region:" + self._scene_id[:16] + ":" + str(len(self._regions))
            summary = {"query_id": query_id, "side": side, "target_reference": target_reference,
                "candidate_count": 0, "attempted_candidate_count": 0,
                "unresolved_candidate_count": 0,
                "attempt_query_ids": [], "candidate_validations": [], "selected_plan_query_id": None,
                "status": "path_rejected", "reason": "region_no_candidate_path",
                "scope": "planned_not_executed", "physical_action_occurred": False}
            deadline = began + 20.
            def reject_expired_selection(now):
                if summary["status"] != "path_found":
                    return
                if (now <= deadline and now - candidate_started <= remaining
                        and (self._query_budget is None
                             or self._planning_seconds <= self._query_budget["maximum_planning_sec"])):
                    return
                self._trajectories.pop(summary["selected_plan_query_id"], None)
                self._trajectories.pop(query_id, None)
                self._queries[-1]["region_validation_budget_exceeded"] = True
                summary["unresolved_candidate_count"] += 1
                summary.update(status="query_incomplete", reason="query_budget_exhausted",
                               selected_plan_query_id=None)
                drop_selection()
            def finalize_budget_result(now):
                reject_expired_selection(now)
                if (summary["status"] == "path_rejected" and summary["reason"] == "region_no_candidate_path"
                        and (summary["unresolved_candidate_count"] or now > deadline
                             or (self._query_budget is not None
                                 and self._planning_seconds > self._query_budget["maximum_planning_sec"]))):
                    summary.update(status="query_incomplete", reason="region_candidate_search_incomplete")
            summary.update(summary_fields())
            try:
                now = self._clock()
                if (now >= deadline or (self._query_budget is not None
                        and region_budget_start + now - began >= self._query_budget["maximum_planning_sec"])):
                    candidates = ()
                    summary.update(status="query_incomplete", reason="query_budget_exhausted")
                else:
                    candidates = generate()
                for candidate in candidates:
                    selection_keys.update(candidate.selection)
            finally:
                generation_wall = self._clock() - began
                self._planning_seconds = region_budget_start + generation_wall
                summary["candidate_generation_wall_sec"] = generation_wall
            summary["candidate_count"] = len(candidates)
            retryable = {"endpoint_collision_blocked", "endpoint_clearance_blocked", "ik_search_exhausted",
                "path_collision_blocked", "path_target_safety_envelope_blocked",
                "grasp_path_motion_quality_exceeded", "safe_tcp_endpoint_pose_tolerance_exceeded"}
            for index, candidate in enumerate(candidates):
                candidate_started = self._clock()
                self._planning_seconds = region_budget_start + (candidate_started - began)
                remaining = min(float(candidate_budget_sec), deadline - candidate_started)
                if self._query_budget is not None:
                    remaining = min(remaining,
                        self._query_budget["maximum_planning_sec"] - self._planning_seconds,
                        self._query_budget.get("maximum_call_planning_sec", remaining))
                if remaining <= 0:
                    summary.update(status="query_incomplete", reason="query_budget_exhausted")
                    break
                prior_calls = self._planning_calls
                prior_planning_seconds = self._planning_seconds
                prior_query_count = len(self._queries)
                result = None
                within = False
                try:
                    call_remaining = max(0., remaining - (self._clock() - candidate_started))
                    result = self._plan_pose_locked(side=side,
                        pose=candidate.pose, time_budget_sec=call_remaining,
                        absolute_deadline=candidate_started + remaining,
                        relation={"region_query_id": query_id, "target_reference": target_reference,
                                  **candidate.relation})
                    if result["status"] == "path_found":
                        record, within = validate(self._queries[-1])
                        summary["candidate_validations"].append({"query_id": result["query_id"], **record})
                except BaseException:
                    for new_query in self._queries[prior_query_count:]:
                        self._trajectories.pop(new_query["query_id"], None)
                    raise
                finally:
                    candidate_finished = self._clock()
                    candidate_wall = candidate_finished - candidate_started
                    summary["attempt_query_ids"].extend(
                        row["query_id"] for row in self._queries[prior_query_count:])
                    summary["attempted_candidate_count"] += self._planning_calls - prior_calls
                    # _plan_pose_locked already charges its own work. Add only
                    # the remaining outer work, including scheduling and final
                    # region validation, including on exceptional exits.
                    already_charged = self._planning_seconds - prior_planning_seconds
                    extra = max(0., candidate_wall - already_charged)
                    self._planning_seconds = region_budget_start + (candidate_finished - began)
                    if len(self._queries) > prior_query_count:
                        self._queries[-1]["region_candidate_wall_sec"] = candidate_wall
                        self._queries[-1]["region_validation_extra_wall_sec"] = extra
                # A known fatal start or native fault must not be relabeled as
                # a retryable scheduling overrun after the lower call returned.
                prior_outcome = (self._queries[-1].get("outcome_before_budget_check")
                                 if len(self._queries) > prior_query_count else None)
                if (result["status"] == "query_incomplete" and prior_outcome is not None
                        and prior_outcome["status"] == "path_rejected" and prior_outcome["reason"] not in retryable):
                    summary["unresolved_candidate_count"] += 1
                    summary.update(status=result["status"], reason=result["reason"],
                                   terminal_fault_reason=prior_outcome["reason"])
                    break
                if result["status"] == "path_rejected" and result["reason"] not in retryable:
                    summary.update(status=result["status"], reason=result["reason"])
                    break
                if candidate_wall > remaining or candidate_finished > deadline:
                    if result["status"] == "path_found":
                        self._trajectories.pop(result["query_id"], None)
                        self._queries[-1]["region_validation_budget_exceeded"] = True
                    summary["unresolved_candidate_count"] += 1
                    if (candidate_finished < deadline and self._query_budget is not None
                            and self._planning_calls < self._query_budget["maximum_calls"]
                            and self._planning_seconds < self._query_budget["maximum_planning_sec"]):
                        continue
                    if candidate_finished < deadline and self._query_budget is None:
                        continue
                    summary.update(status="query_incomplete", reason="query_budget_exhausted")
                    break
                if result["status"] == "path_found":
                    if not within:
                        continue
                    summary.update(status="path_found", reason="planned_not_executed",
                        selected_plan_query_id=result["query_id"], selected_candidate_index=index,
                        **candidate.selection,
                        trajectory_duration_sec=result["trajectory_duration_sec"])
                    self._trajectories[query_id] = self._trajectories[result["query_id"]]
                    break
                if result["status"] == "query_incomplete":
                    summary["unresolved_candidate_count"] += 1
                    if (self._planning_calls > prior_calls and self._clock() < deadline
                            and (self._query_budget is None or (
                                self._planning_calls < self._query_budget["maximum_calls"]
                                and self._planning_seconds < self._query_budget["maximum_planning_sec"]))):
                        continue
                if result["reason"] not in retryable:
                    summary.update(status=result["status"], reason=result["reason"])
                    break
            finished = self._clock()
            summary["path_planning_wall_sec"] = finished - began
            self._planning_seconds = region_budget_start + summary["path_planning_wall_sec"]
            finalize_budget_result(finished)
            public_result = {key: copy.deepcopy(value) for key, value in summary.items()
                    if key not in hidden}
            return public_result
        except BaseException as error:
            fail_region(error)
            raise
        finally:
            try:
                if summary is not None:
                    if began is not None:
                        final_time = self._clock()
                        summary["path_planning_wall_sec"] = final_time - began
                        self._planning_seconds = region_budget_start + summary["path_planning_wall_sec"]
                        finalize_budget_result(final_time)
                        if public_result is not None:
                            public_result.clear()
                            public_result.update({key: copy.deepcopy(value) for key, value in summary.items()
                                if key not in hidden})
                    self._regions.append(summary)
            except BaseException as error:
                fail_region(error)
                if summary is not None:
                    self._regions.append(summary)
                raise

    def audit(self):
        return copy.deepcopy({"scene_id": self._scene_id, "source": self._source_evidence,
            "references": self._anchors, "queries": self._queries,
            "region_queries": self._regions,
            "planning_budget": self._query_budget, "planning_calls": self._planning_calls,
            "planning_seconds": self._planning_seconds,
            "qpos_by_side": {s: q.tolist() for s, q in self._qpos.items()},
            "relation_reads": self._relation_reads,
            "dispatches": 0, "closed": self._closed})

    def save_trajectory(self, query_id, path):
        """Explicit diagnostic export only; no plan consumption or resume handle."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Saved-scene query or export is still running.")
        try:
            self._require_open()
            trajectory = self._trajectories[query_id]
            with Path(path).open("xb") as stream:
                np.savez_compressed(stream, time_from_start_sec=trajectory.time_from_start_sec,
                    left_qpos=trajectory.qpos_by_side["left"], right_qpos=trajectory.qpos_by_side["right"])
        finally:
            self._lock.release()

    def close(self):
        # No external worker or device is created. Do not race a native query.
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Saved-scene query is still running; close is not complete.")
        try:
            self._closed = True
            self._planner = self._cloud = None
            self._trajectories.clear()
            self._object_surfaces = {}
        finally:
            self._lock.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--request", type=Path, help="JSON relation query; omit to inspect references")
    action.add_argument("--tool-object-relations", action="store_true",
        help="Print measured gripper-object relations instead of references")
    parser.add_argument("--audit-output", type=Path, help="New JSON file for full query evidence")
    clearance = parser.add_mutually_exclusive_group()
    clearance.add_argument("--enable-inherited-start-clearance", dest="inherited_start_clearance",
        action="store_const", const=True,
        help="Certified positive-start egress in offline plans (the default)")
    clearance.add_argument("--strict-start-clearance", dest="inherited_start_clearance",
        action="store_const", const=False,
        help="Reject starts below the ordinary 20 mm clearance, as before 2026-10-29")
    parser.set_defaults(inherited_start_clearance=True)
    parser.add_argument("--endpoint-candidate-ordering", choices=RM65_GRASP_ENDPOINT_ORDER_MODES,
        default=RM65_GRASP_ENDPOINT_ORDER_JOINT_DISTANCE,
        help="Planner IK endpoint ordering (the RealMan Gateway deploys estimated_execution_time)")
    args = parser.parse_args()
    if args.request is not None and args.audit_output is None:
        parser.error("--request requires --audit-output to preserve full planning evidence")
    if args.audit_output is not None and args.audit_output.exists():
        parser.error("--audit-output must not already exist")
    query = SavedSceneSkillQuery.from_manifest(json.loads(args.manifest.read_text()),
        enable_inherited_start_clearance=args.inherited_start_clearance,
        endpoint_candidate_ordering=args.endpoint_candidate_ordering)
    try:
        result = (query.plan_relation(**json.loads(args.request.read_text())) if args.request is not None
                  else query.describe_tool_object_relations() if args.tool_object_relations
                  else query.read_scene())
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    finally:
        query.close()
        if args.audit_output is not None:
            with args.audit_output.open("x") as stream:
                json.dump(query.audit(), stream, ensure_ascii=False, allow_nan=False, indent=2)


if __name__ == "__main__":
    main()
