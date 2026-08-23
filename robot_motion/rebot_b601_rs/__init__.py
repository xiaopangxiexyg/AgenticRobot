"""Model-only B601-RS motion semantics.

Importing this package has no hardware or vendor-package side effects.
"""

from .ik import B601DampedLeastSquaresIK, B601IKConfig, B601IKResult
from .mit import (
    B601_RS_JOINT_MIT_FRAME,
    B601MitControlProfile,
    B601MitReference,
    B601MitTorqueBreakdown,
    compute_b601_mit_torque,
)
from .model import (
    B601_RS_ARM_JOINT_NAMES,
    B601_RS_AUXILIARY_JOINT_NAMES,
    B601_RS_BASE_FRAME,
    B601_RS_END_EFFECTOR_FRAME,
    B601_RS_JACOBIAN_REFERENCE,
    B601_RS_TWIST_ORDER,
    B601_RS_URDF_RELATIVE_PATH,
    B601ModelDependencyError,
    B601ModelError,
    B601PinocchioModel,
    B601SingularityMetrics,
)

__all__ = [
    "B601_RS_ARM_JOINT_NAMES",
    "B601_RS_AUXILIARY_JOINT_NAMES",
    "B601_RS_BASE_FRAME",
    "B601_RS_END_EFFECTOR_FRAME",
    "B601_RS_JACOBIAN_REFERENCE",
    "B601_RS_JOINT_MIT_FRAME",
    "B601_RS_TWIST_ORDER",
    "B601_RS_URDF_RELATIVE_PATH",
    "B601DampedLeastSquaresIK",
    "B601IKConfig",
    "B601IKResult",
    "B601MitControlProfile",
    "B601MitReference",
    "B601MitTorqueBreakdown",
    "B601ModelDependencyError",
    "B601ModelError",
    "B601PinocchioModel",
    "B601SingularityMetrics",
    "compute_b601_mit_torque",
]
