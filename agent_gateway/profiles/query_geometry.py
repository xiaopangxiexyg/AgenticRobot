"""Profile-owned endpoint generation and exact query-evidence binding.

Neutral contracts validate public values. Only this application edge invokes
the canonical lower generators; no sampling or rotation algorithm is copied.
"""

from collections.abc import Mapping

from agent_gateway.skills.perception.object_goal_query import RELATION_FIELDS
from agent_gateway.skills.perception.region_continuation_query import (
    validate_region_continuation_parameters,
    validate_region_continuation_version,
)
from agent_gateway.skills.perception.relation_continuation_query import (
    POSES_KEY,
    explicit_continuation_parameters,
    validate_relation_continuation_parameters,
)
from agent_gateway.skills.perception.tcp_continuation_query import (
    validate_continuation_parameters,
    validate_continuation_result_for_parameters,
)
from robot_motion.spatial_goal_regions import (
    generate_continuation_region_goals,
    generate_relation_region_goals,
)
from robot_motion.spatial_goals import generate_relation_goals


def relation_candidate_poses(parameters, *, initial, anchor_xyz_m):
    return [list(item.goal.pose_xyz_quat_xyzw) for item in generate_relation_goals(
        current_tcp_pose_xyz_quat_xyzw=initial, anchor_xyz_m=anchor_xyz_m,
        **{key: parameters[key] for key in RELATION_FIELDS
           if key not in {"component_id", "reference_frame_id", "time_budget_sec"}})]


def region_candidate_poses(parameters, *, initial, anchor_xyz_m, version="4"):
    validate_region_continuation_version(version)
    generator = generate_relation_region_goals if version == "4" else generate_continuation_region_goals
    continuation = {} if version == "4" else {
        "continuation_delta_xyz_m": parameters["continuation_delta_xyz_m"],
    }
    return [list(item.goal.pose_xyz_quat_xyzw) for item in generator(
        current_tcp_pose_xyz_quat_xyzw=initial, anchor_xyz_m=anchor_xyz_m,
        **continuation,
        **{key: parameters[key] for key in (
            "offset_bounds_xyz_m", "offset_frame", "orientation_mode", "tool_axis",
            "axis_reference_xyz", "tool_axis_rotation_bounds_rad", "max_candidates")} )]


def validate_relation_geometry_parameters(parameters):
    """Retain the original admission-time numeric/overflow checks in the profile."""
    relation_candidate_poses(parameters, initial=[0., 0., 0., 0., 0., 0., 1.], anchor_xyz_m=[0., 0., 0.])


def validate_relation_continuation_geometry_parameters(parameters):
    poses = relation_candidate_poses(parameters, initial=[0., 0., 0., 0., 0., 0., 1.],
        anchor_xyz_m=[0., 0., 0.])
    validate_continuation_parameters(explicit_continuation_parameters(parameters, poses))


def validate_region_geometry_parameters(parameters, *, version="4"):
    """Use the exact lower sampler for admission, including generated-pose checks."""
    poses = region_candidate_poses(parameters, initial=[0., 0., 0., 0., 0., 0., 1.],
        anchor_xyz_m=[0., 0., 0.], version=version)
    validate_continuation_parameters(explicit_continuation_parameters(parameters, poses))


def validate_query_selection(parameters, result, candidates):
    """Check the exact relation samples after common path-evidence validation."""
    if (any(result[key] != parameters[key] for key in ("component_id", "reference_frame_id"))
            or list(result["anchor_xyz_m"]) != list(parameters["anchor_xyz_m"])):
        raise ValueError("Query result changed the requested geometry identity.")
    if len(candidates) != result["candidate_count"]:
        raise ValueError("Query result changed the candidate set.")
    for item in result["candidate_diagnostics"]:
        if "pose_target_xyz_quat_xyzw" in item:
            expected = candidates[item["candidate_index"]].goal.pose_xyz_quat_xyzw
            if any(abs(a - b) > 1e-8 for a, b in zip(
                    expected, item["pose_target_xyz_quat_xyzw"], strict=True)):
                raise ValueError("Diagnostic endpoint differs from its relation candidate.")
    if result["selected_index"] is not None:
        expected = candidates[result["selected_index"]].goal.pose_xyz_quat_xyzw
        if any(abs(a - b) > 1e-8 for a, b in zip(expected,
                result["pose_targets_by_component"][result["component_id"]], strict=True)):
            raise ValueError("Selected endpoint differs from its relation candidate.")


def validate_relation_continuation_path(parameters, result, *, anchor_xyz_m):
    validate_relation_continuation_parameters(parameters)
    if not isinstance(result, Mapping) or POSES_KEY not in result:
        raise ValueError("Relation continuation evidence requires its exact generated proposals.")
    poses = relation_candidate_poses(parameters,
        initial=result["source_tcp_pose_xyz_quat_xyzw"], anchor_xyz_m=anchor_xyz_m)
    validate_continuation_result_for_parameters(explicit_continuation_parameters(parameters, poses),
        {key: value for key, value in result.items() if key != POSES_KEY})
    actual = result[POSES_KEY]
    validate_continuation_parameters(explicit_continuation_parameters(parameters, actual))
    if len(actual) != len(poses) or any(abs(a - b) > 1e-8
            for actual_pose, pose in zip(actual, poses, strict=True)
            for a, b in zip(actual_pose, pose, strict=True)):
        raise ValueError("Generated proposals changed their frozen TCP or authenticated anchor geometry.")


def validate_region_continuation_path(parameters, result, *, anchor_xyz_m, version="4"):
    validate_region_continuation_parameters(parameters, version=version)
    if not isinstance(result, Mapping) or POSES_KEY not in result:
        raise ValueError("Region continuation must retain its exact generated proposals.")
    poses = region_candidate_poses(parameters,
        initial=result["source_tcp_pose_xyz_quat_xyzw"], anchor_xyz_m=anchor_xyz_m, version=version)
    validate_continuation_result_for_parameters(explicit_continuation_parameters(parameters, poses),
        {key: value for key, value in result.items() if key != POSES_KEY})
    actual = result[POSES_KEY]
    validate_continuation_parameters(explicit_continuation_parameters(parameters, actual))
    if len(actual) != len(poses) or any(abs(a - b) > 1e-8
            for actual_pose, pose in zip(actual, poses, strict=True)
            for a, b in zip(actual_pose, pose, strict=True)):
        raise ValueError("Region proposals changed their bounds, frozen TCP or authenticated anchor geometry.")
