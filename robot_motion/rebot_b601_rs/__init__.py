"""Backend-neutral B601-RS motion and haptic semantics.

Importing this package has no hardware or vendor-package side effects.
"""

from .haptics import (
    B601_RS_WORLD_ALIGNED_EE_WRENCH_FRAME,
    B601_RS_WRENCH_ORDER,
    B601PoseVirtualSpring,
    B601PoseVirtualSpringResult,
    B601SpatialWrench,
    B601TranslationalVirtualSpring,
    B601VirtualSpringResult,
    B601VirtualWall,
    B601VirtualWallResult,
    B601WrenchJointTorque,
    map_world_aligned_ee_wrench_to_joint_torque,
    transform_spatial_twist,
    transform_spatial_wrench,
)
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
    "B601_RS_WORLD_ALIGNED_EE_WRENCH_FRAME",
    "B601_RS_WRENCH_ORDER",
    "B601DampedLeastSquaresIK",
    "B601IKConfig",
    "B601IKResult",
    "B601MitControlProfile",
    "B601MitReference",
    "B601MitTorqueBreakdown",
    "B601ModelDependencyError",
    "B601ModelError",
    "B601PinocchioModel",
    "B601PoseVirtualSpring",
    "B601PoseVirtualSpringResult",
    "B601SingularityMetrics",
    "B601SpatialWrench",
    "B601TranslationalVirtualSpring",
    "B601VirtualSpringResult",
    "B601VirtualWall",
    "B601VirtualWallResult",
    "B601WrenchJointTorque",
    "compute_b601_mit_torque",
    "map_world_aligned_ee_wrench_to_joint_torque",
    "transform_spatial_twist",
    "transform_spatial_wrench",
]
