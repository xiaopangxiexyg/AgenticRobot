"""Profile-owned relation endpoints and exact query-evidence binding."""
from agent_gateway.skills.perception.object_goal_query import RELATION_FIELDS
from robot_motion.spatial_goals import generate_relation_goals

def relation_candidate_poses(parameters, *, initial, anchor_xyz_m):
    return [list(item.goal.pose_xyz_quat_xyzw) for item in generate_relation_goals(
        current_tcp_pose_xyz_quat_xyzw=initial, anchor_xyz_m=anchor_xyz_m,
        **{key: parameters[key] for key in RELATION_FIELDS
           if key not in {"component_id", "reference_frame_id", "time_budget_sec"}})]



def validate_relation_geometry_parameters(parameters):
    """Retain the original admission-time numeric/overflow checks in the profile."""
    relation_candidate_poses(parameters, initial=[0., 0., 0., 0., 0., 0., 1.], anchor_xyz_m=[0., 0., 0.])



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
