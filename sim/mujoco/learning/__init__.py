"""Backend-neutral learning contracts for the MuJoCo stack.

The package deliberately contains no task implementation and no hardware
initialisation.  It is the first boundary for making a simulator observation
and a real deployment observation describe the same model input semantics.
"""

from .contracts import (
    LEARNING_CONTRACT_VERSION,
    ModelInputBinding,
    ObservationContract,
    ObservationFieldSpec,
    ObservationRole,
    PolicyActionBinding,
    PolicyActionTransform,
)
from .model_io import EncodedModelInput, encode_model_input
from .manifest import (
    MODEL_MANIFEST_VERSION,
    ModelDryRunReceipt,
    PolicyModelManifest,
    dry_run_model_io,
)
from .reward import (
    EmbodimentRewardConfig,
    RewardResult,
    TaskRewardAdapter,
    compose_embodiment_reward,
    compose_reward,
)

__all__ = [
    "LEARNING_CONTRACT_VERSION",
    "ModelInputBinding",
    "ObservationContract",
    "ObservationFieldSpec",
    "ObservationRole",
    "PolicyActionBinding",
    "PolicyActionTransform",
    "EncodedModelInput",
    "encode_model_input",
    "MODEL_MANIFEST_VERSION",
    "ModelDryRunReceipt",
    "PolicyModelManifest",
    "dry_run_model_io",
    "EmbodimentRewardConfig",
    "RewardResult",
    "TaskRewardAdapter",
    "compose_embodiment_reward",
    "compose_reward",
]
