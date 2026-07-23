"""Compatibility exports for profiles now owned by :mod:`sim.rm65.actuation`."""

from sim.rm65.actuation.profile import (
    PositionActuationProfile,
    rm65_bimanual_udp_feedback_profile,
    rm65_bimanual_udp_feedback_profile_v1,
    rm65_bimanual_udp_feedback_profile_v2,
    rm65_bimanual_udp_feedback_profile_v3,
)

__all__ = [
    "PositionActuationProfile",
    "rm65_bimanual_udp_feedback_profile",
    "rm65_bimanual_udp_feedback_profile_v1",
    "rm65_bimanual_udp_feedback_profile_v2",
    "rm65_bimanual_udp_feedback_profile_v3",
]
