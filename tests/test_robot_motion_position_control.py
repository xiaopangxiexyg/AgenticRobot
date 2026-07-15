from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from robot_motion.position_control import (
    JointCommandInterpolator,
    OutOfOrderReferenceError,
    PositionReferenceFrame,
    PositionSideReference,
    ReferenceTimeline,
    interpolate_joint_command_segment,
    reference_age_and_stale,
)


class FakeClock:
    def __init__(self, *, monotonic: float = 0.0, wall: float = 1000.0):
        self.monotonic_time = float(monotonic)
        self.wall_time = float(wall)

    def monotonic(self) -> float:
        return self.monotonic_time

    def advance(self, dt: float) -> None:
        self.monotonic_time += float(dt)
        self.wall_time += float(dt)


def test_common_joint_segment_planner_is_timestamp_driven_and_bounded():
    start = np.zeros(6, dtype=float)
    target = np.ones(6, dtype=float)

    before, before_alpha = interpolate_joint_command_segment(
        start,
        target,
        segment_start_time=10.0,
        segment_duration_sec=0.02,
        sample_time=9.0,
    )
    middle, middle_alpha = interpolate_joint_command_segment(
        start,
        target,
        segment_start_time=10.0,
        segment_duration_sec=0.02,
        sample_time=10.005,
    )
    after, after_alpha = interpolate_joint_command_segment(
        start,
        target,
        segment_start_time=10.0,
        segment_duration_sec=0.02,
        sample_time=11.0,
    )

    assert before_alpha == 0.0
    assert middle_alpha == pytest.approx(0.25)
    assert after_alpha == 1.0
    np.testing.assert_allclose(before, start)
    np.testing.assert_allclose(middle, np.full(6, 0.25))
    np.testing.assert_allclose(after, target)


def test_reference_stale_rule_is_shared_and_uses_strict_deadline_crossing():
    age, stale = reference_age_and_stale(
        published_monotonic_time=10.0,
        sample_monotonic_time=10.1,
        stale_after_sec=0.1,
    )
    assert age == pytest.approx(0.1)
    assert not stale

    age, stale = reference_age_and_stale(
        published_monotonic_time=10.0,
        sample_monotonic_time=10.100001,
        stale_after_sec=0.1,
    )
    assert age == pytest.approx(0.100001)
    assert stale


def _side(side: str, value: float, *, valid: bool = True, hold: bool = False):
    pose = np.eye(4, dtype=float)
    pose[0, 3] = value
    return PositionSideReference(
        side=side,
        q_ik_raw=np.full(6, value + 0.1, dtype=float),
        q_ref=np.full(6, value, dtype=float),
        pose_ref=pose,
        valid=valid,
        hold=hold,
        modification_reasons=("rate_limit",) if value else (),
        frame_metadata={"base": f"{side}_base"},
        profile_metadata={"tool": {"name": f"{side}_tcp"}},
    )


def _frame(
    sequence: int,
    value: float,
    *,
    epoch: int = 7,
    wall: float | None = None,
    monotonic: float | None = None,
    valid: bool = True,
    hold: bool = False,
) -> PositionReferenceFrame:
    wall_time = 1000.0 + float(sequence) / 45.0 if wall is None else float(wall)
    monotonic_time = (
        10.0 + float(sequence) / 45.0 if monotonic is None else float(monotonic)
    )
    return PositionReferenceFrame(
        epoch=epoch,
        sequence=sequence,
        source_wall_time=wall_time - 0.002,
        generated_wall_time=wall_time - 0.001,
        published_wall_time=wall_time,
        generated_monotonic_time=monotonic_time - 0.001,
        published_monotonic_time=monotonic_time,
        references={
            "left": _side("left", value, valid=valid, hold=hold),
            "right": _side("right", -value, valid=valid, hold=hold),
        },
        source="teleop",
        frame_metadata={"frames": ["left_base", "right_base"]},
        profile_metadata={"scene": {"name": "dual_rm65"}},
    )


def test_reference_frame_detaches_and_freezes_atomic_bimanual_values():
    q_ref = np.arange(6, dtype=float)
    metadata = {"nested": {"values": [1, 2]}, "array": np.array([3.0])}
    left = PositionSideReference(
        side="left",
        q_ik_raw=q_ref + 1.0,
        q_ref=q_ref,
        pose_ref=np.eye(4),
        frame_metadata=metadata,
    )
    references = {"left": left, "right": _side("right", 2.0, hold=True)}
    frame = PositionReferenceFrame(
        epoch=1,
        sequence=3,
        source_wall_time=10.0,
        generated_wall_time=10.1,
        published_wall_time=10.2,
        generated_monotonic_time=20.0,
        published_monotonic_time=20.1,
        references=references,
    )

    q_ref[:] = -999.0
    metadata["nested"]["values"].append(3)
    metadata["array"][0] = -1.0
    references.clear()

    assert frame.sides == ("left", "right")
    assert frame.valid
    assert frame.hold
    assert not frame.all_hold
    assert np.allclose(frame.q_ref_by_side["left"], np.arange(6, dtype=float))
    assert frame.references["left"].frame_metadata["nested"]["values"] == (1, 2)
    assert frame.references["left"].frame_metadata["array"][0] == 3.0
    with pytest.raises(ValueError):
        frame.q_ref_by_side["left"][0] = 4.0
    with pytest.raises(TypeError):
        frame.references["left"] = _side("left", 3.0)
    with pytest.raises(FrozenInstanceError):
        frame.sequence = 4


def test_timeline_selects_causally_for_jittered_15_and_45_hz_clocks():
    timeline = ReferenceTimeline(max_frames=8)
    reference_times = [100.0000, 100.0216, 100.0447, 100.0671, 100.0888]
    for sequence, timestamp in enumerate(reference_times):
        timeline.publish(
            _frame(sequence, float(sequence), wall=timestamp, monotonic=10.0 + timestamp - 100.0)
        )

    # Nominal 15 Hz cuts are deliberately phase shifted and not 1:3 aligned.
    sample_times = [100.0050, 100.0717]
    selected = [
        timeline.latest_at_or_before(sample_time, epoch=7, max_age=0.1)
        for sample_time in sample_times
    ]

    assert [frame.sequence if frame is not None else None for frame in selected] == [0, 3]
    assert all(
        frame is not None and frame.published_wall_time <= sample_time
        for frame, sample_time in zip(selected, sample_times, strict=True)
    )
    assert timeline.latest_at_or_before(99.9999, epoch=7, max_age=0.1) is None
    assert timeline.latest_at_or_before(100.2, epoch=7, max_age=0.05) is None
    assert timeline.latest_at_or_before(100.0717, epoch=8, max_age=0.1) is None


def test_timeline_skips_invalid_frames_and_rejects_out_of_order_publication():
    timeline = ReferenceTimeline(max_frames=3)
    timeline.publish(_frame(0, 0.0, wall=100.0, monotonic=10.0))
    timeline.publish(_frame(1, 1.0, wall=100.02, monotonic=10.02, valid=False, hold=True))

    selected = timeline.latest_at_or_before(100.03, epoch=7, max_age=0.1)
    assert selected is not None
    assert selected.sequence == 0

    with pytest.raises(OutOfOrderReferenceError):
        timeline.publish(_frame(1, 2.0, wall=100.04, monotonic=10.04))
    with pytest.raises(OutOfOrderReferenceError):
        timeline.publish(_frame(2, 2.0, wall=100.04, monotonic=10.01))

    # A wall-clock correction is not publication reordering; monotonic time and
    # sequence remain authoritative for producer order.
    timeline.publish(_frame(2, 2.0, wall=99.0, monotonic=10.04))
    assert len(timeline) == 3
    rewound = timeline.latest_at_or_before(99.5, epoch=7, max_age=1.0)
    assert rewound is not None and rewound.sequence == 2
    after_rewind = timeline.latest_at_or_before(100.1, epoch=7, max_age=2.0)
    assert after_rewind is not None and after_rewind.sequence == 2


def test_interpolator_converts_45_hz_reference_to_shared_150_hz_ticks():
    clock = FakeClock(monotonic=20.0)
    duration = 1.0 / 45.0
    interpolator = JointCommandInterpolator(
        segment_duration_sec=duration,
        stale_after_sec=1.0,
        monotonic_clock=clock.monotonic,
    )
    interpolator.reset(
        {"left": np.zeros(6), "right": np.zeros(6)},
        epoch=7,
    )
    interpolator.accept_target(_frame(0, 1.0, monotonic=20.0))

    ticks = []
    for index in range(5):
        clock.monotonic_time = 20.0 + index / 150.0
        ticks.append(interpolator.step())

    assert [tick.command_sequence for tick in ticks] == list(range(5))
    assert all(tick.reference_sequence == 0 for tick in ticks)
    assert np.allclose(
        [tick.interpolation_alpha for tick in ticks],
        [0.0, 0.3, 0.6, 0.9, 1.0],
    )
    for tick in ticks:
        assert np.allclose(tick.q_cmd_by_side["left"], tick.interpolation_alpha)
        assert np.allclose(tick.q_cmd_by_side["right"], -tick.interpolation_alpha)
        assert tick.sides == ("left", "right")


def test_target_preemption_restarts_from_last_emitted_command_without_jump():
    clock = FakeClock(monotonic=30.0)
    interpolator = JointCommandInterpolator(
        segment_duration_sec=0.03,
        stale_after_sec=1.0,
        monotonic_clock=clock.monotonic,
    )
    interpolator.reset({"left": np.zeros(6), "right": np.zeros(6)}, epoch=7)
    interpolator.accept_target(_frame(0, 1.0, monotonic=30.0))

    clock.monotonic_time = 30.015
    before_preemption = interpolator.step()
    assert np.allclose(before_preemption.q_cmd_by_side["left"], 0.5)

    clock.monotonic_time = 30.016
    interpolator.accept_target(_frame(1, -1.0, monotonic=30.016))
    at_preemption = interpolator.step()
    assert at_preemption.reference_sequence == 1
    assert at_preemption.interpolation_alpha == 0.0
    assert np.allclose(
        at_preemption.q_cmd_by_side["left"],
        before_preemption.q_cmd_by_side["left"],
    )

    clock.monotonic_time = 30.031
    after_preemption = interpolator.step()
    assert np.isclose(after_preemption.interpolation_alpha, 0.5)
    assert np.allclose(after_preemption.q_cmd_by_side["left"], -0.25)
    assert np.allclose(after_preemption.q_cmd_by_side["right"], 0.25)


def test_real_transport_can_preempt_from_last_accepted_instead_of_unaccepted_plan():
    planner = JointCommandInterpolator(
        segment_duration_sec=0.1,
        stale_after_sec=None,
    )
    planner.reset({"left": np.zeros(6)}, epoch=1, monotonic_time=0.0)
    planner.accept_qpos_target(
        {"left": np.ones(6)},
        epoch=1,
        sequence=1,
        published_monotonic_time=0.0,
        monotonic_time=0.0,
        source="real_test",
    )
    unaccepted = planner.step(monotonic_time=0.05)
    np.testing.assert_allclose(unaccepted.q_cmd_by_side["left"], 0.5)

    last_sdk_accepted = np.full(6, 0.2)
    planner.accept_qpos_target(
        {"left": np.full(6, -1.0)},
        epoch=1,
        sequence=2,
        published_monotonic_time=0.06,
        monotonic_time=0.06,
        source="real_test",
        anchor_qpos_by_side={"left": last_sdk_accepted},
    )
    preempted = planner.step(monotonic_time=0.06)

    assert preempted.reference_sequence == 2
    assert preempted.interpolation_alpha == 0.0
    np.testing.assert_allclose(preempted.q_cmd_by_side["left"], last_sdk_accepted)


def test_wall_clock_rewind_does_not_affect_monotonic_interpolation():
    clock = FakeClock(monotonic=40.0, wall=5000.0)
    interpolator = JointCommandInterpolator(
        segment_duration_sec=0.04,
        stale_after_sec=1.0,
        monotonic_clock=clock.monotonic,
    )
    interpolator.reset({"left": np.zeros(6), "right": np.zeros(6)}, epoch=7)
    interpolator.accept_target(
        _frame(0, 1.0, wall=clock.wall_time, monotonic=clock.monotonic_time)
    )

    clock.wall_time -= 4000.0
    clock.monotonic_time += 0.02
    tick = interpolator.step()

    assert np.isclose(tick.interpolation_alpha, 0.5)
    assert np.allclose(tick.q_cmd_by_side["left"], 0.5)
    assert np.allclose(tick.q_cmd_by_side["right"], -0.5)


def test_stale_reference_holds_last_emitted_qpos_until_fresh_preemption():
    clock = FakeClock(monotonic=50.0)
    interpolator = JointCommandInterpolator(
        segment_duration_sec=0.1,
        stale_after_sec=0.03,
        monotonic_clock=clock.monotonic,
    )
    interpolator.reset({"left": np.zeros(6), "right": np.zeros(6)}, epoch=7)
    interpolator.accept_target(_frame(0, 1.0, monotonic=50.0))

    clock.monotonic_time = 50.02
    fresh = interpolator.step()
    assert np.allclose(fresh.q_cmd_by_side["left"], 0.2)
    assert not fresh.stale

    clock.monotonic_time = 50.04
    stale = interpolator.step()
    assert stale.stale and stale.hold
    assert np.allclose(stale.q_cmd_by_side["left"], fresh.q_cmd_by_side["left"])

    interpolator.accept_target(_frame(1, 2.0, monotonic=50.04))
    resumed = interpolator.step()
    assert not resumed.stale
    assert np.allclose(resumed.q_cmd_by_side["left"], fresh.q_cmd_by_side["left"])


def test_reset_clears_reference_state_and_requires_matching_epoch_and_sides():
    clock = FakeClock(monotonic=60.0)
    interpolator = JointCommandInterpolator(
        segment_duration_sec=0.02,
        stale_after_sec=1.0,
        monotonic_clock=clock.monotonic,
    )
    interpolator.reset({"left": np.zeros(6), "right": np.zeros(6)}, epoch=7)
    interpolator.accept_target(_frame(0, 1.0, epoch=7, monotonic=60.0))
    clock.monotonic_time = 60.01
    interpolator.step()

    interpolator.reset(
        {"left": np.full(6, 3.0), "right": np.full(6, -3.0)},
        epoch=8,
    )
    reset_tick = interpolator.step()
    assert reset_tick.epoch == 8
    assert reset_tick.command_sequence == 0
    assert reset_tick.reference_sequence is None
    assert reset_tick.hold and not reset_tick.stale
    assert np.allclose(reset_tick.q_cmd_by_side["left"], 3.0)

    with pytest.raises(ValueError, match="epoch"):
        interpolator.accept_target(_frame(1, 1.0, epoch=7, monotonic=60.01))

    single_side = PositionReferenceFrame(
        epoch=8,
        sequence=0,
        source_wall_time=100.0,
        generated_wall_time=100.0,
        published_wall_time=100.0,
        generated_monotonic_time=60.01,
        published_monotonic_time=60.01,
        references={"left": _side("left", 1.0)},
    )
    with pytest.raises(ValueError, match="sides"):
        interpolator.accept_target(single_side)


def test_interpolator_rejects_duplicate_reference_and_backward_monotonic_step():
    clock = FakeClock(monotonic=70.0)
    interpolator = JointCommandInterpolator(
        stale_after_sec=1.0,
        monotonic_clock=clock.monotonic,
    )
    interpolator.reset({"left": np.zeros(6), "right": np.zeros(6)}, epoch=7)
    reference = _frame(0, 1.0, monotonic=70.0)
    interpolator.accept_target(reference)
    with pytest.raises(OutOfOrderReferenceError):
        interpolator.accept_target(reference)

    clock.monotonic_time = 70.01
    interpolator.step()
    clock.monotonic_time = 70.005
    with pytest.raises(ValueError, match="backwards"):
        interpolator.step()


def test_interpolator_accepts_same_atomic_sides_in_different_mapping_order():
    clock = FakeClock(monotonic=80.0)
    interpolator = JointCommandInterpolator(
        stale_after_sec=1.0,
        monotonic_clock=clock.monotonic,
    )
    interpolator.reset({"left": np.zeros(6), "right": np.zeros(6)}, epoch=7)
    frame = PositionReferenceFrame(
        epoch=7,
        sequence=0,
        source_wall_time=100.0,
        generated_wall_time=100.0,
        published_wall_time=100.0,
        generated_monotonic_time=80.0,
        published_monotonic_time=80.0,
        references={"right": _side("right", -1.0), "left": _side("left", 1.0)},
    )

    interpolator.accept_target(frame)
    tick = interpolator.step()

    assert tick.sides == ("left", "right")
    assert set(tick.q_target_by_side) == {"left", "right"}
