from __future__ import annotations

import pytest

from experiments.sim_learning.robosuite_mixed_control_a3 import (
    A3_COLLECTION_MODE_ORDINARY_TELEOP,
    A3_COLLECTION_MODE_STREAMING_MIXED_CONTROL,
    A3_COLLECTION_MODE_TAKEOVER_TELEOP,
    A3_POLICY_VARIANT_NONE,
    A3_POLICY_VARIANT_RTC,
    A3_POLICY_VARIANT_STANDARD,
    resolve_collection_profile,
    validate_collection_frames,
)
from experiments.sim_learning.streaming_intervention import (
    A2_COMPONENT_IDS,
    SequencePolicySourceA2,
    StreamingInterventionControlFrameA2,
)


def _action(value: float, gripper: float = -0.5) -> tuple[float, ...]:
    return (value, 0.0, 0.0, 0.0, 0.0, 0.0, gripper)


def _standard_source() -> SequencePolicySourceA2:
    return SequencePolicySourceA2(
        source_id="a3-contract-standard",
        source_kind=A3_POLICY_VARIANT_STANDARD,
        actions=(_action(0.1),),
    )


def _rtc_source() -> SequencePolicySourceA2:
    return SequencePolicySourceA2(
        source_id="a3-contract-rtc",
        source_kind=A3_POLICY_VARIANT_RTC,
        actions=(_action(0.2),),
    )


def test_profile_resolution_is_the_single_mode_source_matrix() -> None:
    ordinary = resolve_collection_profile(A3_COLLECTION_MODE_ORDINARY_TELEOP, None)
    assert ordinary.policy_variant == A3_POLICY_VARIANT_NONE
    assert ordinary.frame_mode == "base_only"
    assert ordinary.requires_human_intervention
    assert ordinary.forbids_nominal_base_action

    takeover = resolve_collection_profile(
        A3_COLLECTION_MODE_TAKEOVER_TELEOP,
        _standard_source(),
    )
    assert takeover.policy_variant == A3_POLICY_VARIANT_STANDARD
    assert takeover.frame_mode == "absolute_overlay_owner"
    assert takeover.requires_full_human_components

    streaming_standard = resolve_collection_profile(
        A3_COLLECTION_MODE_STREAMING_MIXED_CONTROL,
        _standard_source(),
    )
    assert streaming_standard.requires_human_intervention
    assert not streaming_standard.requires_nominal_base_action

    streaming_rtc = resolve_collection_profile(
        A3_COLLECTION_MODE_STREAMING_MIXED_CONTROL,
        _rtc_source(),
    )
    assert streaming_rtc.policy_variant == A3_POLICY_VARIANT_RTC
    assert not streaming_rtc.requires_human_intervention
    assert streaming_rtc.requires_nominal_base_action


def test_profile_validation_accepts_the_three_declared_capture_shapes() -> None:
    ordinary_frame = StreamingInterventionControlFrameA2(
        frame_id="ordinary",
        mode="base_only",
        human_raw_action=(0.0,) * 7,
        human_mapped_action=_action(0.1),
        human_event_id="human-ordinary",
    )
    _, ordinary_frames = validate_collection_frames(
        collection_mode=A3_COLLECTION_MODE_ORDINARY_TELEOP,
        frames=(ordinary_frame,),
        policy_source=None,
    )
    assert ordinary_frames == (ordinary_frame,)

    takeover_frame = StreamingInterventionControlFrameA2(
        frame_id="takeover",
        mode="absolute_overlay_owner",
        human_raw_action=(0.0,) * 7,
        human_mapped_action=_action(0.2),
        human_component_ids=A2_COMPONENT_IDS,
        human_event_id="human-takeover",
    )
    takeover_profile, _ = validate_collection_frames(
        collection_mode=A3_COLLECTION_MODE_TAKEOVER_TELEOP,
        frames=(takeover_frame,),
        policy_source=_standard_source(),
    )
    assert takeover_profile.requires_full_human_components

    streaming_standard_frame = StreamingInterventionControlFrameA2(
        frame_id="stream-standard",
        mode="relative_overlay",
        human_raw_action=(0.0,) * 7,
        human_mapped_action=_action(0.1),
        human_component_ids=("arm",),
        human_event_id="human-stream",
    )
    validate_collection_frames(
        collection_mode=A3_COLLECTION_MODE_STREAMING_MIXED_CONTROL,
        frames=(streaming_standard_frame,),
        policy_source=_standard_source(),
    )

    streaming_rtc_frame = StreamingInterventionControlFrameA2(
        frame_id="stream-rtc",
        mode="relative_overlay",
        base_action=_action(0.3),
    )
    validate_collection_frames(
        collection_mode=A3_COLLECTION_MODE_STREAMING_MIXED_CONTROL,
        frames=(streaming_rtc_frame,),
        policy_source=_rtc_source(),
    )


def test_profile_validation_rejects_ambiguous_human_policy_combinations() -> None:
    frame = StreamingInterventionControlFrameA2(
        frame_id="rtc-with-human",
        mode="relative_overlay",
        human_raw_action=(0.0,) * 7,
        human_mapped_action=_action(0.1),
        human_component_ids=("arm",),
        human_event_id="human",
    )

    with pytest.raises(ValueError, match="cannot mix human intervention"):
        validate_collection_frames(
            collection_mode=A3_COLLECTION_MODE_STREAMING_MIXED_CONTROL,
            frames=(frame,),
            policy_source=_rtc_source(),
        )

    with pytest.raises(ValueError, match="require human intervention"):
        validate_collection_frames(
            collection_mode=A3_COLLECTION_MODE_STREAMING_MIXED_CONTROL,
            frames=(
                StreamingInterventionControlFrameA2(
                    frame_id="standard-without-human",
                    mode="relative_overlay",
                ),
            ),
            policy_source=_standard_source(),
        )
