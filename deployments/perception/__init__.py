"""Local perception-model adapters with no Agent Gateway ownership."""

from .graspgenx_client import (
    GRASPGENX_DEFAULT_MODE,
    GraspGenXClient,
    GraspGenXInferenceBackend,
    GraspGenXInProcessBackend,
)
from .graspgenx_runtime_contract import (
    GRASPGENX_INPROCESS_RUNTIME_CONTRACT_ID,
    GraspGenXRuntimeReceipt,
    verify_graspgenx_inprocess_runtime,
)
from .sam2_graspgenx_point_grasp import (
    GRASPGENX_DMTAC_REACHABILITY_ORIENTATION_VARIANTS_LOCAL_AXIS_DEG,
    GraspGenXPointGraspModelSpec,
    Sam2GraspGenXPointGraspPlanner,
)
from .sam2_graspnet_point_grasp import (
    GraspNetPointGraspModelSpec,
    Sam2GraspNetPointGraspPlanner,
)
from .sam2_grounding_dino import (
    GroundedObjectDetection,
    GroundedObjectSegment,
    GroundedObjectSegmentArtifacts,
    Sam2GroundingDinoLocalizer,
    Sam2GroundingDinoModelSpec,
    encode_grounded_object_segment_artifacts,
)
from .sam2_vgn_point_grasp import (
    DMTAC_M_MAX_VISIBLE_COLLISION_POINTS,
    Sam2VgnPointGraspPlanner,
    VgnPointGraspModelSpec,
)

__all__ = [
    "DMTAC_M_MAX_VISIBLE_COLLISION_POINTS",
    "GRASPGENX_DEFAULT_MODE",
    "GRASPGENX_INPROCESS_RUNTIME_CONTRACT_ID",
    "GRASPGENX_DMTAC_REACHABILITY_ORIENTATION_VARIANTS_LOCAL_AXIS_DEG",
    "GraspGenXClient",
    "GraspGenXInProcessBackend",
    "GraspGenXInferenceBackend",
    "GraspGenXPointGraspModelSpec",
    "GraspGenXRuntimeReceipt",
    "GraspNetPointGraspModelSpec",
    "GroundedObjectDetection",
    "GroundedObjectSegment",
    "GroundedObjectSegmentArtifacts",
    "Sam2GraspGenXPointGraspPlanner",
    "Sam2GraspNetPointGraspPlanner",
    "Sam2GroundingDinoLocalizer",
    "Sam2GroundingDinoModelSpec",
    "Sam2VgnPointGraspPlanner",
    "VgnPointGraspModelSpec",
    "encode_grounded_object_segment_artifacts",
    "verify_graspgenx_inprocess_runtime",
]
