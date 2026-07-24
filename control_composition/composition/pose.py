"""Canonical import surface for arm-pose composition strategies."""

from __future__ import annotations

from .context import (
    COMPOSITION_CONTEXT_KINDS_V1,
    CompositionContext,
)
from .strategies import (
    DISCRETE_OWNER_POSE_STRATEGY_ID_V1,
    POSE_MIXING_ACTION_SPACES_V1,
    POSE_MIXING_ARM_COMPONENTS_V1,
    POSE_MIXING_CONTEXT_KINDS_V1,
    WEIGHTED_SE3_POSE_FUSION_STRATEGY_ID_V1,
    DiscreteOwnerPoseMixingStrategy,
    PoseMixingCandidate,
    PoseMixingContext,
    PoseMixingDecision,
    PoseMixingInputSnapshot,
    PoseMixingStrategy,
    PoseMixingStrategySpec,
    WeightedSE3PoseFusionStrategy,
    build_pose_mixing_decision,
    build_pose_mixing_input_snapshot,
    fuse_weighted_se3_pose_actions,
    normalize_pose_component_dependencies,
)

__all__ = [
    "DISCRETE_OWNER_POSE_STRATEGY_ID_V1",
    "COMPOSITION_CONTEXT_KINDS_V1",
    "CompositionContext",
    "POSE_MIXING_ACTION_SPACES_V1",
    "POSE_MIXING_ARM_COMPONENTS_V1",
    "POSE_MIXING_CONTEXT_KINDS_V1",
    "WEIGHTED_SE3_POSE_FUSION_STRATEGY_ID_V1",
    "DiscreteOwnerPoseMixingStrategy",
    "PoseMixingCandidate",
    "PoseMixingContext",
    "PoseMixingDecision",
    "PoseMixingInputSnapshot",
    "PoseMixingStrategy",
    "PoseMixingStrategySpec",
    "WeightedSE3PoseFusionStrategy",
    "build_pose_mixing_decision",
    "build_pose_mixing_input_snapshot",
    "fuse_weighted_se3_pose_actions",
    "normalize_pose_component_dependencies",
]
