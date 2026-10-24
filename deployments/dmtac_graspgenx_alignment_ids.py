"""Dependency-free alignment identifiers shared by launchers and perception.

Geometry and calibration loading remain owned by the perception adapter.
Importing CLI option metadata must not initialize that optional package.
"""

DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID_V4 = "DMTAC_M_graspgenx_alignment_v4_q0"
DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID_V5 = "DMTAC_M_graspgenx_alignment_v5_q0"
DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID = DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID_V5
DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_IDS = (
    DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID_V5,
    DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID_V4,
)
