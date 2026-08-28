from agent_gateway.profiles.real.motion_primitive_profiles import (
    REAL_AGENT_TCP_POSE_TARGET_ROUTE_ID_V1,
    real_agent_motion_route_ids_v1,
)
from agent_gateway.profiles.real.rm65_composition import _route_ids


def test_standard_rm65_composition_authorizes_absolute_grasp_pose_route() -> None:
    route_ids = real_agent_motion_route_ids_v1()

    assert REAL_AGENT_TCP_POSE_TARGET_ROUTE_ID_V1 in route_ids
    assert _route_ids() == route_ids
    assert len(route_ids) == 11
