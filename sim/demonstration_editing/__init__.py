"""Backend-neutral demonstration trajectory editing for simulation tasks."""

from .contracts import (
    DEMONSTRATION_EDIT_SCHEMA_V1,
    DEMONSTRATION_TRAJECTORY_SCHEMA_V1,
    EDITED_DEMONSTRATION_TRAJECTORY_SCHEMA_V1,
    HAND_EDIT_FINAL_DELTA_RAMP_V1,
    POSE_EDIT_LEFT_MULTIPLY_RIGID_V1,
    DemonstrationEdit,
    DemonstrationTrajectory,
    EditedDemonstrationTrajectory,
)
from .editor import (
    DemonstrationReference,
    apply_demonstration_edit,
    sample_edited_trajectory,
)
from .component_action import (
    DEMONSTRATION_COMPONENT_ACTION_BINDING_HASH_DOMAIN_V1,
    DEMONSTRATION_COMPONENT_ACTION_RECEIPT_BINDING_KIND_V1,
    DEMONSTRATION_COMPONENT_ACTION_RECEIPT_HASH_DOMAIN_V1,
    NORMALIZED_GRIPPER_CLOSE_PROJECTION_HASH_DOMAIN_V1,
    DemonstrationComponentActionAdapter,
    DemonstrationComponentActionBinding,
    DemonstrationComponentActionReceipt,
    DemonstrationComponentActionResult,
    NormalizedGripperCloseToJointPositionProjection,
)
from .preflight import (
    ZERO_EDIT_REPLAY_SCOPE_V1,
    ZeroEditReplayPreflight,
    preflight_zero_edit_replay,
)

__all__ = [
    "DEMONSTRATION_EDIT_SCHEMA_V1",
    "DEMONSTRATION_COMPONENT_ACTION_BINDING_HASH_DOMAIN_V1",
    "DEMONSTRATION_COMPONENT_ACTION_RECEIPT_BINDING_KIND_V1",
    "DEMONSTRATION_COMPONENT_ACTION_RECEIPT_HASH_DOMAIN_V1",
    "NORMALIZED_GRIPPER_CLOSE_PROJECTION_HASH_DOMAIN_V1",
    "DEMONSTRATION_TRAJECTORY_SCHEMA_V1",
    "EDITED_DEMONSTRATION_TRAJECTORY_SCHEMA_V1",
    "HAND_EDIT_FINAL_DELTA_RAMP_V1",
    "POSE_EDIT_LEFT_MULTIPLY_RIGID_V1",
    "ZERO_EDIT_REPLAY_SCOPE_V1",
    "DemonstrationEdit",
    "DemonstrationComponentActionAdapter",
    "DemonstrationComponentActionBinding",
    "DemonstrationComponentActionReceipt",
    "DemonstrationComponentActionResult",
    "NormalizedGripperCloseToJointPositionProjection",
    "DemonstrationReference",
    "DemonstrationTrajectory",
    "EditedDemonstrationTrajectory",
    "ZeroEditReplayPreflight",
    "apply_demonstration_edit",
    "preflight_zero_edit_replay",
    "sample_edited_trajectory",
]
