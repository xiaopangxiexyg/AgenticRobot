from __future__ import annotations

from dataclasses import replace

import numpy as np

from robot_motion.action import RobotAction
from robot_motion.deploy.actions import NormalizedAction
from robot_motion.deploy.canonical_policy_action import build_canonical_policy_action
from robot_motion.deploy.chunk_plan import (
    POLICY_PLAN_KEYFRAME_MUTABLE,
    TimedPolicyActionKeyframe,
    TimedPolicyActionPlanSnapshot,
)
from robot_motion.deploy.rtc import (
    RtcContinuationSelectionOutcome,
    RtcContinuationSelectionSpec,
    RtcContinuationTimeline,
    RtcPlanContinuationEntry,
    RtcPlanContinuationSlice,
    select_rtc_plan_continuation,
)


def _canonical_action(index: int):
    x = 0.4 + 0.01 * index
    action = RobotAction(
        action_space="single_arm_ee_pose_v1",
        timestamp=10.0 + 0.1 * index,
        arm_pose_by_side={
            "right": np.asarray(
                [x, -0.1, 0.3, 0.0, 0.0, 0.0, 1.0],
                dtype=np.float32,
            )
        },
        source="rtc-continuation-selection-test",
    )
    return build_canonical_policy_action(
        action,
        source_pose_mode="absolute",
        observation_cut_id="observation:test",
        source_action_hash=f"source-action:{index}",
        action_codec_hash="codec:test",
        frame_contract_hash="frame:test",
        layout_hash="layout:test",
        normalization_hash="normalization:test",
        conversion_contract_hash="conversion:absolute:test",
    )


def _keyframe(
    index: int,
    *,
    target: float | None = None,
    period: float = 0.1,
    typed: bool = True,
) -> TimedPolicyActionKeyframe:
    target_at = 10.0 + 0.1 * index if target is None else float(target)
    return TimedPolicyActionKeyframe(
        target_effective_at=target_at,
        effective_until=target_at + period,
        effective_clock_domain_id="clock:test",
        action_period_sec=period,
        decoded_absolute_action=(
            None
            if typed
            else NormalizedAction(
                vector=np.full((26,), float(index), dtype=np.float32),
                metadata={"source": "legacy-continuation-test"},
            )
        ),
        canonical_action=_canonical_action(index) if typed else None,
        request_id="request:old",
        inference_id="inference:old",
        policy_output_id="output:old",
        chunk_id=4,
        chunk_keyframe_index=index,
        chunk_size=8,
        observation_cut_id="observation:old",
        producer_epoch=2,
        policy_node_id="policy:test",
        artifact_hash="artifact:test",
        produced_at=9.8,
        schedule_spec_id="schedule:test",
        schedule_spec_hash="schedule-hash:test",
        admitted_plan_generation=3,
        state=POLICY_PLAN_KEYFRAME_MUTABLE,
        valid_until=target_at + period + 1.0,
    )


def _plan(
    indices: tuple[int, ...],
    *,
    typed: bool = True,
    period: float = 0.1,
    targets: tuple[float, ...] | None = None,
    in_flight: bool = False,
) -> TimedPolicyActionPlanSnapshot:
    keyframes = tuple(
        _keyframe(
            index,
            typed=typed,
            period=period,
            target=None if targets is None else targets[position],
        )
        for position, index in enumerate(indices)
    )
    return TimedPolicyActionPlanSnapshot(
        plan_id="plan:test",
        session_id="session:test",
        control_domain_id="domain:test",
        producer_id="producer:test",
        producer_epoch=2,
        policy_node_id="policy:test",
        clock_domain_id="clock:test",
        generation=3,
        cursor_at=10.0,
        created_at=9.9,
        keyframes=keyframes,
        in_flight_request_id="request:active" if in_flight else None,
        in_flight_inference_id="inference:active" if in_flight else None,
    )


def _timeline() -> RtcContinuationTimeline:
    return RtcContinuationTimeline(
        clock_domain_id="clock:test",
        generation_start_at=10.0,
        first_target_effective_at=10.3,
        action_period_sec=0.1,
        prediction_horizon_steps=6,
        forecast_delay_steps=3,
    )


def _spec(
    *,
    prefix_horizon_steps: int = 3,
    minimum_selected_steps: int = 1,
) -> RtcContinuationSelectionSpec:
    return RtcContinuationSelectionSpec(
        prefix_horizon_steps=prefix_horizon_steps,
        minimum_selected_steps=minimum_selected_steps,
    )


def test_selects_exact_typed_overlap_without_mutating_source_plan() -> None:
    plan = _plan((1, 2, 3, 4))
    before = plan.semantic_content_hash
    outcome = select_rtc_plan_continuation(
        plan,
        timeline=_timeline(),
        spec=_spec(minimum_selected_steps=2),
    )

    assert outcome.accepted
    assert not outcome.bootstrap_required
    continuation = outcome.continuation_slice
    assert continuation is not None
    assert continuation.source_plan_hash == before
    assert continuation.selected_generation_indices == (1, 2)
    assert continuation.first_generation_index == 1
    assert continuation.valid_prefix_steps == 3
    assert continuation.entries[0].keyframe.canonical_action is not None
    assert plan.semantic_content_hash == before


def test_empty_or_short_overlap_requests_explicit_bootstrap() -> None:
    empty = select_rtc_plan_continuation(
        _plan(()),
        timeline=_timeline(),
        spec=_spec(),
    )
    assert not empty.accepted
    assert empty.bootstrap_required
    assert empty.reasons == ("empty_source_plan",)

    short = select_rtc_plan_continuation(
        _plan((2,)),
        timeline=_timeline(),
        spec=_spec(minimum_selected_steps=2),
    )
    assert not short.accepted
    assert short.bootstrap_required
    assert short.reasons == ("insufficient_prefix_steps",)


def test_gap_is_rejected_instead_of_hidden_or_interpolated() -> None:
    outcome = select_rtc_plan_continuation(
        _plan((0, 2)),
        timeline=_timeline(),
        spec=_spec(),
    )
    assert not outcome.accepted
    assert not outcome.bootstrap_required
    assert outcome.reasons == ("non_contiguous_overlap",)


def test_off_lattice_and_cadence_mismatch_fail_closed() -> None:
    off_lattice = select_rtc_plan_continuation(
        _plan((1,), targets=(10.15,)),
        timeline=_timeline(),
        spec=_spec(),
    )
    assert not off_lattice.accepted
    assert off_lattice.reasons == ("keyframe_off_generation_lattice",)

    wrong_period = select_rtc_plan_continuation(
        _plan((1,), period=0.2),
        timeline=_timeline(),
        spec=_spec(),
    )
    assert not wrong_period.accepted
    assert wrong_period.reasons == ("action_period_mismatch",)


def test_legacy_payload_is_not_a_model_rtc_canonical_prefix() -> None:
    outcome = select_rtc_plan_continuation(
        _plan((1, 2), typed=False),
        timeline=_timeline(),
        spec=_spec(),
    )
    assert not outcome.accepted
    assert outcome.reasons == ("typed_canonical_action_required",)


def test_prefix_horizon_cannot_reach_into_the_executable_suffix() -> None:
    outcome = select_rtc_plan_continuation(
        _plan((1, 2, 3, 4)),
        timeline=_timeline(),
        spec=_spec(prefix_horizon_steps=4),
    )
    assert not outcome.accepted
    assert outcome.reasons == ("prefix_horizon_exceeds_forecast_delay",)
    assert not outcome.bootstrap_required


def test_active_request_clock_and_cursor_mismatches_are_not_bootstrap() -> None:
    in_flight = select_rtc_plan_continuation(
        _plan((1, 2), in_flight=True),
        timeline=_timeline(),
        spec=_spec(),
    )
    assert not in_flight.accepted
    assert in_flight.reasons == ("source_plan_already_in_flight",)
    assert not in_flight.bootstrap_required

    wrong_clock = select_rtc_plan_continuation(
        _plan((1, 2)),
        timeline=replace(_timeline(), clock_domain_id="clock:other"),
        spec=_spec(),
    )
    assert not wrong_clock.accepted
    assert wrong_clock.reasons == ("clock_domain_mismatch",)

    stale_generation_start = select_rtc_plan_continuation(
        _plan((1, 2)),
        timeline=RtcContinuationTimeline(
            clock_domain_id="clock:test",
            generation_start_at=9.9,
            first_target_effective_at=10.2,
            action_period_sec=0.1,
            prediction_horizon_steps=6,
            forecast_delay_steps=3,
        ),
        spec=_spec(),
    )
    assert not stale_generation_start.accepted
    assert stale_generation_start.reasons == (
        "generation_start_before_plan_cursor",
    )


def test_selection_identity_changes_with_exact_old_plan_payload() -> None:
    first_plan = _plan((1, 2))
    changed_keyframe = replace(
        first_plan.keyframes[1],
        canonical_action=_canonical_action(7),
    )
    second_plan = replace(
        first_plan,
        keyframes=(first_plan.keyframes[0], changed_keyframe),
    )
    first = select_rtc_plan_continuation(
        first_plan,
        timeline=_timeline(),
        spec=_spec(),
    )
    second = select_rtc_plan_continuation(
        second_plan,
        timeline=_timeline(),
        spec=_spec(),
    )
    assert first.accepted and second.accepted
    assert first.semantic_content_hash != second.semantic_content_hash
    assert first.continuation_slice is not None
    assert second.continuation_slice is not None
    assert (
        first.continuation_slice.canonical_prefix_hash
        != second.continuation_slice.canonical_prefix_hash
    )


def test_public_slice_contract_cannot_bypass_selector_invariants() -> None:
    plan = _plan((0, 2))
    entries = tuple(
        RtcPlanContinuationEntry(generation_index=index, keyframe=keyframe)
        for index, keyframe in zip((0, 2), plan.keyframes, strict=True)
    )
    timeline = _timeline()
    spec = _spec()
    with np.testing.assert_raises_regex(ValueError, "must be contiguous"):
        RtcPlanContinuationSlice(
            source_plan_id=plan.plan_id,
            source_plan_generation=plan.generation,
            source_plan_hash=plan.semantic_content_hash,
            producer_id=plan.producer_id,
            producer_epoch=plan.producer_epoch,
            policy_node_id=plan.policy_node_id,
            timeline_hash=timeline.semantic_content_hash,
            selection_spec_hash=spec.semantic_content_hash,
            prefix_horizon_steps=spec.prefix_horizon_steps,
            entries=entries,
        )

    with np.testing.assert_raises_regex(
        ValueError,
        "requires a typed canonical action",
    ):
        RtcPlanContinuationEntry(
            generation_index=0,
            keyframe=_keyframe(0, typed=False),
        )


def test_accepted_outcome_binds_slice_plan_timeline_and_spec_hashes() -> None:
    plan = _plan((1, 2))
    timeline = _timeline()
    spec = _spec()
    selected = select_rtc_plan_continuation(
        plan,
        timeline=timeline,
        spec=spec,
    )
    assert selected.continuation_slice is not None
    with np.testing.assert_raises_regex(
        ValueError,
        "Selection/source plan hash mismatch",
    ):
        RtcContinuationSelectionOutcome(
            accepted=True,
            source_plan_hash="plan-hash:wrong",
            timeline_hash=timeline.semantic_content_hash,
            selection_spec_hash=spec.semantic_content_hash,
            continuation_slice=selected.continuation_slice,
            reasons=(),
        )
