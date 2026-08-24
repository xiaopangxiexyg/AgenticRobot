"""Experiment-local mode and policy-source contracts for A3 capture.

The A3 runner deliberately keeps collection admission separate from row
materialization.  This module owns the small mode/source matrix that defines
which kind of policy and human contribution is valid for each experiment
variant.  It does not own policy inference, backend execution, safety, or
durable dataset persistence.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from experiments.sim_learning.streaming_intervention.a2 import (
    A2_COMPONENT_IDS,
    A2_POLICY_SOURCE_KIND_RTC,
    A2_POLICY_SOURCE_KIND_STANDARD,
    PolicyInferenceSourceA2,
    StreamingInterventionControlFrameA2,
)
from sim.robosuite.adapter import (
    ROBOSUITE_MIXED_CONTROL_MODE_ABSOLUTE_OVERLAY_OWNER_A0,
    ROBOSUITE_MIXED_CONTROL_MODE_BASE_ONLY_A0,
    ROBOSUITE_MIXED_CONTROL_MODE_RELATIVE_OVERLAY_A0,
)

A3_COLLECTION_MODE_ORDINARY_TELEOP = "ordinary_teleop"
A3_COLLECTION_MODE_TAKEOVER_TELEOP = "takeover_teleop"
A3_COLLECTION_MODE_STREAMING_MIXED_CONTROL = "streaming_mixed_control"
A3_COLLECTION_MODES = frozenset(
    {
        A3_COLLECTION_MODE_ORDINARY_TELEOP,
        A3_COLLECTION_MODE_TAKEOVER_TELEOP,
        A3_COLLECTION_MODE_STREAMING_MIXED_CONTROL,
    }
)

A3_POLICY_VARIANT_NONE = "none"
A3_POLICY_VARIANT_STANDARD = A2_POLICY_SOURCE_KIND_STANDARD
A3_POLICY_VARIANT_RTC = A2_POLICY_SOURCE_KIND_RTC
A3_POLICY_VARIANTS = frozenset(
    {A3_POLICY_VARIANT_NONE, A3_POLICY_VARIANT_STANDARD, A3_POLICY_VARIANT_RTC}
)

A3_OBSERVATION_HASH_DOMAIN = "dexteleop.robosuite_mixed_control.observation.a3"
A3_ROW_HASH_DOMAIN = "dexteleop.robosuite_mixed_control.row.a3"
A3_DATASET_HASH_DOMAIN = "dexteleop.robosuite_mixed_control.dataset.a3"


@dataclass(frozen=True)
class A3CollectionProfile:
    """Resolved admission rules for one A3 collection mode and source.

    ``requires_human_intervention`` is resolved for the actual policy source:
    standard streaming requires a human residual, while RTC streaming is a
    policy-only intervention over a nominal base action.  The profile is an
    experiment contract, not a runtime safety or execution authority.
    """

    collection_mode: str
    policy_variant: str
    frame_mode: str
    requires_human_intervention: bool
    requires_full_human_components: bool = False
    forbids_nominal_base_action: bool = False
    requires_nominal_base_action: bool = False

    def validate_frames(
        self,
        frames: Iterable[StreamingInterventionControlFrameA2],
    ) -> tuple[StreamingInterventionControlFrameA2, ...]:
        """Validate and return an immutable frame sequence for this profile."""

        frame_values = tuple(frames)
        if not frame_values:
            raise ValueError("A3 capture requires at least one frame.")
        if any(
            not isinstance(frame, StreamingInterventionControlFrameA2)
            for frame in frame_values
        ):
            raise TypeError(
                "frames must contain StreamingInterventionControlFrameA2 values."
            )

        for frame in frame_values:
            if frame.mode != self.frame_mode:
                if self.collection_mode == A3_COLLECTION_MODE_ORDINARY_TELEOP:
                    raise ValueError(
                        "ordinary teleop requires human-only base_only frames."
                    )
                if self.collection_mode == A3_COLLECTION_MODE_TAKEOVER_TELEOP:
                    raise ValueError(
                        "takeover teleop requires full-component absolute human takeover."
                    )
                raise ValueError(
                    "streaming mixed control requires relative_overlay frames."
                )
            if self.requires_human_intervention and not frame.has_human_intervention:
                if self.collection_mode == A3_COLLECTION_MODE_ORDINARY_TELEOP:
                    raise ValueError(
                        "ordinary teleop requires human-only base_only frames."
                    )
                if self.collection_mode == A3_COLLECTION_MODE_TAKEOVER_TELEOP:
                    raise ValueError(
                        "takeover teleop requires full-component absolute human takeover."
                    )
                raise ValueError("standard streaming rows require human intervention.")
            if not self.requires_human_intervention and frame.has_human_intervention:
                raise ValueError(
                    "RTC streaming rows cannot mix human intervention."
                )
            if (
                self.requires_full_human_components
                and frame.human_component_ids != A2_COMPONENT_IDS
            ):
                raise ValueError(
                    "takeover teleop requires full-component absolute human takeover."
                )
            if self.forbids_nominal_base_action and frame.base_action is not None:
                raise ValueError(
                    "ordinary teleop requires human-only base_only frames."
                )
            if self.requires_nominal_base_action and frame.base_action is None:
                raise ValueError("RTC streaming rows require a nominal base action.")
        return frame_values


def policy_variant(policy_source: PolicyInferenceSourceA2 | None) -> str:
    """Resolve the experiment policy variant without importing capture wiring."""

    if policy_source is None:
        return A3_POLICY_VARIANT_NONE
    source_kind = str(getattr(policy_source, "source_kind", "")).strip()
    if source_kind not in {A3_POLICY_VARIANT_STANDARD, A3_POLICY_VARIANT_RTC}:
        raise ValueError(f"unsupported policy source kind {source_kind!r}.")
    return source_kind


def resolve_collection_profile(
    collection_mode: str,
    policy_source: PolicyInferenceSourceA2 | None,
) -> A3CollectionProfile:
    """Resolve the one valid source/mode contract used by an A3 run."""

    normalized_mode = str(collection_mode).strip()
    variant = policy_variant(policy_source)
    if normalized_mode not in A3_COLLECTION_MODES:
        raise ValueError(f"Unsupported A3 collection mode {normalized_mode!r}.")

    if normalized_mode == A3_COLLECTION_MODE_ORDINARY_TELEOP:
        if variant != A3_POLICY_VARIANT_NONE:
            raise ValueError("ordinary teleop cannot install a policy source.")
        return A3CollectionProfile(
            collection_mode=normalized_mode,
            policy_variant=variant,
            frame_mode=ROBOSUITE_MIXED_CONTROL_MODE_BASE_ONLY_A0,
            requires_human_intervention=True,
            forbids_nominal_base_action=True,
        )

    if normalized_mode == A3_COLLECTION_MODE_TAKEOVER_TELEOP:
        if variant != A3_POLICY_VARIANT_STANDARD:
            raise ValueError("takeover teleop requires a standard policy source.")
        return A3CollectionProfile(
            collection_mode=normalized_mode,
            policy_variant=variant,
            frame_mode=ROBOSUITE_MIXED_CONTROL_MODE_ABSOLUTE_OVERLAY_OWNER_A0,
            requires_human_intervention=True,
            requires_full_human_components=True,
        )

    if variant not in {A3_POLICY_VARIANT_STANDARD, A3_POLICY_VARIANT_RTC}:
        raise ValueError("streaming mixed control requires standard or RTC policy.")
    return A3CollectionProfile(
        collection_mode=normalized_mode,
        policy_variant=variant,
        frame_mode=ROBOSUITE_MIXED_CONTROL_MODE_RELATIVE_OVERLAY_A0,
        requires_human_intervention=variant == A3_POLICY_VARIANT_STANDARD,
        requires_nominal_base_action=variant == A3_POLICY_VARIANT_RTC,
    )


def validate_collection_frames(
    *,
    collection_mode: str,
    frames: Iterable[StreamingInterventionControlFrameA2],
    policy_source: PolicyInferenceSourceA2 | None,
) -> tuple[A3CollectionProfile, tuple[StreamingInterventionControlFrameA2, ...]]:
    """Resolve a profile and validate all frames before backend creation."""

    profile = resolve_collection_profile(collection_mode, policy_source)
    return profile, profile.validate_frames(frames)


__all__ = [
    "A3_COLLECTION_MODES",
    "A3_COLLECTION_MODE_ORDINARY_TELEOP",
    "A3_COLLECTION_MODE_STREAMING_MIXED_CONTROL",
    "A3_COLLECTION_MODE_TAKEOVER_TELEOP",
    "A3_DATASET_HASH_DOMAIN",
    "A3_OBSERVATION_HASH_DOMAIN",
    "A3_POLICY_VARIANTS",
    "A3_POLICY_VARIANT_NONE",
    "A3_POLICY_VARIANT_RTC",
    "A3_POLICY_VARIANT_STANDARD",
    "A3_ROW_HASH_DOMAIN",
    "A3CollectionProfile",
    "policy_variant",
    "resolve_collection_profile",
    "validate_collection_frames",
]
