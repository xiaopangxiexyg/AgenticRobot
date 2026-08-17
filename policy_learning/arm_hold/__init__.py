"""Compatibility exports for ``experiments.policy_learning.arm_hold``."""

from __future__ import annotations

import sys

from experiments.policy_learning import arm_hold as _canonical
from experiments.policy_learning.arm_hold import (
    ARM_HOLD_DWELL_BUCKET_THRESHOLDS,
    ARM_HOLD_FUSION_MODES,
    ARM_HOLD_LABEL_CONTEXT_MODES,
    ArmHoldBaseline,
    ArmHoldBaselineConfig,
    ArmHoldLossConfig,
    ArmHoldTensorBatch,
    arm_hold_constant_logits,
    arm_hold_dwell_bucket_indices,
    arm_hold_expand_constant_logits,
    arm_hold_fit_constant_side_logits,
    arm_hold_tensor_batch,
    arm_hold_loss,
)
from experiments.policy_learning.arm_hold.evaluation import (
    ARM_HOLD_EVALUATION_SCHEMA,
    ArmHoldEvaluationError,
    arm_hold_window_evaluation,
)
from experiments.policy_learning.arm_hold.free_running import (
    ARM_HOLD_FREE_RUNNING_CONTEXT_MODES,
    ARM_HOLD_FREE_RUNNING_EVALUATION_SCHEMA,
    ArmHoldFreeRunningError,
    arm_hold_free_running_evaluation,
    arm_hold_offset_zero_evaluation,
)
from experiments.policy_learning.arm_hold.sequence_evaluation import (
    ARM_HOLD_CHUNK_SEQUENCE_SCHEMA,
    ARM_HOLD_FIXED_LEAD_SEQUENCE_SCHEMA,
    ArmHoldSequenceEvaluationError,
    arm_hold_chunk_sequence_evaluation,
    arm_hold_fixed_lead_sequence_evaluation,
)

__all__ = [
    "ARM_HOLD_DWELL_BUCKET_THRESHOLDS",
    "ARM_HOLD_FUSION_MODES",
    "ARM_HOLD_LABEL_CONTEXT_MODES",
    "ArmHoldBaseline",
    "ArmHoldBaselineConfig",
    "ArmHoldLossConfig",
    "ArmHoldTensorBatch",
    "ARM_HOLD_CHUNK_SEQUENCE_SCHEMA",
    "ARM_HOLD_EVALUATION_SCHEMA",
    "ARM_HOLD_FREE_RUNNING_CONTEXT_MODES",
    "ARM_HOLD_FREE_RUNNING_EVALUATION_SCHEMA",
    "ARM_HOLD_FIXED_LEAD_SEQUENCE_SCHEMA",
    "ArmHoldEvaluationError",
    "ArmHoldFreeRunningError",
    "ArmHoldSequenceEvaluationError",
    "arm_hold_constant_logits",
    "arm_hold_dwell_bucket_indices",
    "arm_hold_chunk_sequence_evaluation",
    "arm_hold_expand_constant_logits",
    "arm_hold_fit_constant_side_logits",
    "arm_hold_free_running_evaluation",
    "arm_hold_offset_zero_evaluation",
    "arm_hold_fixed_lead_sequence_evaluation",
    "arm_hold_tensor_batch",
    "arm_hold_window_evaluation",
    "arm_hold_loss",
]

sys.modules[f"{__name__}.baseline"] = _canonical.baseline
sys.modules[f"{__name__}.evaluation"] = _canonical.evaluation
sys.modules[f"{__name__}.free_running"] = _canonical.free_running
sys.modules[f"{__name__}.sequence_evaluation"] = _canonical.sequence_evaluation
