from types import SimpleNamespace

from agent_gateway.profiles.real.motion_primitive_profiles import (
    REAL_AGENT_GRASP_APPROACH_ROUTE_ID_V1,
    REAL_AGENT_TCP_POSE_TARGET_ROUTE_ID_V1,
    real_agent_motion_route_ids_v1,
)
from agent_gateway.profiles.real.rm65_composition import _route_ids
from agent_gateway.profiles.real.session_authorization import (
    REAL_AGENT_ABSOLUTE_TCP_POSE_ACTION,
    REAL_AGENT_EXECUTE_GRASP_APPROACH_PLAN_ACTION,
    REAL_AGENT_GRASP_APPROACH_ACTION,
    REAL_AGENT_PRIMITIVE_ACTIONS,
    _requested_action_and_component_ids,
)


def test_standard_rm65_composition_authorizes_absolute_grasp_pose_route() -> None:
    route_ids = real_agent_motion_route_ids_v1(include_grasp_approach=True)

    assert REAL_AGENT_TCP_POSE_TARGET_ROUTE_ID_V1 in route_ids
    assert REAL_AGENT_GRASP_APPROACH_ROUTE_ID_V1 in route_ids
    assert _route_ids() == route_ids
    assert len(route_ids) == 12


def test_absolute_grasp_pose_route_has_an_exact_operator_action_class() -> None:
    request = SimpleNamespace(
        skill_id="motion.move_tcp_to_pose",
        parameters={
            "pose_targets_by_component": {
                "right_arm": [0.25, -0.1, 0.3, 0.0, 0.0, 0.0, 1.0]
            }
        },
    )

    assert REAL_AGENT_ABSOLUTE_TCP_POSE_ACTION in REAL_AGENT_PRIMITIVE_ACTIONS
    assert _requested_action_and_component_ids(request) == (
        REAL_AGENT_ABSOLUTE_TCP_POSE_ACTION,
        ("right_arm",),
    )

    approach = SimpleNamespace(
        skill_id="motion.approach_grasp_pose",
        parameters=request.parameters,
    )
    assert REAL_AGENT_GRASP_APPROACH_ACTION in REAL_AGENT_PRIMITIVE_ACTIONS
    assert _requested_action_and_component_ids(approach) == (
        REAL_AGENT_GRASP_APPROACH_ACTION,
        ("right_arm",),
    )

    execute_plan = SimpleNamespace(
        skill_id="motion.execute_grasp_approach_plan",
        parameters={
            "component_id": "right_arm",
            "grasp_approach_plan_id": "real-grasp-approach-plan:" + "a" * 64,
            "grasp_approach_plan_hash": "a" * 64,
        },
    )
    assert REAL_AGENT_EXECUTE_GRASP_APPROACH_PLAN_ACTION in REAL_AGENT_PRIMITIVE_ACTIONS
    assert _requested_action_and_component_ids(execute_plan) == (
        REAL_AGENT_EXECUTE_GRASP_APPROACH_PLAN_ACTION,
        ("right_arm",),
    )


def test_absolute_grasp_pose_authorization_fails_closed_on_component_shape() -> None:
    missing = SimpleNamespace(
        skill_id="motion.move_tcp_to_pose",
        parameters={"pose_targets_by_component": {}},
    )
    bimanual = SimpleNamespace(
        skill_id="motion.move_tcp_to_pose",
        parameters={
            "pose_targets_by_component": {
                "left_arm": [0.0] * 7,
                "right_arm": [0.0] * 7,
            }
        },
    )

    assert _requested_action_and_component_ids(missing) == (
        REAL_AGENT_ABSOLUTE_TCP_POSE_ACTION,
        (),
    )
    assert _requested_action_and_component_ids(bimanual) == (
        REAL_AGENT_ABSOLUTE_TCP_POSE_ACTION,
        ("left_arm", "right_arm"),
    )
