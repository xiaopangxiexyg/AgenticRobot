"""Bind the existing MuJoCo right-arm profile to the shared passive registration."""

from agent_gateway.profiles.tcp_continuation_query import (
    build_tcp_continuation_query_executor,
)
from agent_gateway.profiles.tcp_continuation_query import (
    build_tcp_continuation_query_registration as _build_registration,
)

__all__ = ["build_tcp_continuation_query_executor", "build_tcp_continuation_query_registration"]


def build_tcp_continuation_query_registration(
    *, base_route, base_binding, reference_frame_id, planning_profile_hash,
):
    """Preserve the published MuJoCo component, profile and Route identities."""
    return _build_registration(
        base_route=base_route, base_binding=base_binding, component_id="right_arm",
        reference_frame_id=reference_frame_id, planning_profile_hash=planning_profile_hash,
    )
