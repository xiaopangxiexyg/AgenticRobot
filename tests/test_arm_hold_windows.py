from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dataset_contracts import (
    ARM_HOLD_TRAINING_PROJECTION_ID_V1,
    ARM_MODE_IDS,
    ARM_MODE_SCHEMA_V2,
    arm_hold_training_projection_metadata,
)
from dataset_interface.arm_hold import project_arm_hold_training_view
from dataset_interface.arm_hold_profile import load_arm_hold_dataset_profile
from dataset_interface.arm_hold_windows import (
    ARM_HOLD_HISTORY_CONTEXT_PROVENANCE_V1,
    ARM_HOLD_HISTORY_CONTEXT_SCHEMA_V1,
    ARM_HOLD_HISTORY_SEQUENCE_CONTEXT_SCHEMA_V1,
    ArmHoldHistoryContext,
    ArmHoldHistorySequenceContext,
    ArmHoldHistoryTimeline,
    ArmHoldWindowEpisodeBinding,
    ArmHoldWindowError,
    ArmHoldWindowInventory,
    ArmHoldWindowLocation,
    ArmHoldWindowReader,
    ArmHoldWindowSample,
    derive_arm_hold_history_context,
    derive_arm_hold_history_sequence_context,
    derive_arm_hold_history_timeline,
    select_arm_hold_window_starts,
    summarize_arm_hold_window_target_counts,
    summarize_arm_hold_window_transition_coverage,
)


def _history_context(
    *,
    start: int,
    frames: list[int] | None = None,
    continuity: list[int | None] | None = None,
    running: list[bool] | None = None,
    valid: np.ndarray | None = None,
) -> ArmHoldHistoryContext:
    targets = np.asarray(
        [
            [0, 1],
            [0, 1],
            [0, 0],
            [1, 0],
            [1, 1],
            [1, 1],
        ],
        dtype=np.float32,
    )
    count = len(targets)
    effective_valid = (
        np.ones_like(targets, dtype=bool) if valid is None else valid
    )
    return derive_arm_hold_history_context(
        targets=targets,
        valid_mask=effective_valid,
        loss_mask=effective_valid.copy(),
        frame_indices=list(range(count)) if frames is None else frames,
        continuity_segments=[0] * count if continuity is None else continuity,
        running_mask=np.asarray(
            [True] * count if running is None else running, dtype=bool
        ),
        start_index=start,
    )


def _history_sequence_context(
    *,
    start: int,
    frames: list[int] | None = None,
    continuity: list[int | None] | None = None,
    running: list[bool] | None = None,
    valid: np.ndarray | None = None,
) -> ArmHoldHistorySequenceContext:
    targets = np.asarray(
        [
            [0, 1],
            [0, 1],
            [0, 0],
            [1, 0],
            [1, 1],
            [1, 1],
        ],
        dtype=np.float32,
    )
    count = len(targets)
    effective_valid = (
        np.ones_like(targets, dtype=bool) if valid is None else valid
    )
    return derive_arm_hold_history_sequence_context(
        targets=targets,
        valid_mask=effective_valid,
        loss_mask=effective_valid.copy(),
        frame_indices=list(range(count)) if frames is None else frames,
        continuity_segments=[0] * count if continuity is None else continuity,
        running_mask=np.asarray(
            [True] * count if running is None else running, dtype=bool
        ),
        start_index=start,
    )


def test_history_context_derives_previous_canonical_mode_and_dwell() -> None:
    context = _history_context(start=3)

    assert context.schema == ARM_HOLD_HISTORY_CONTEXT_SCHEMA_V1
    assert context.provenance == ARM_HOLD_HISTORY_CONTEXT_PROVENANCE_V1
    assert context.previous_mode.tolist() == [0.0, 0.0]
    assert context.available_mask.tolist() == [True, True]
    assert context.dwell_steps.tolist() == [3, 1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start": 0},
        {"start": 3, "frames": [0, 1, 2, 4, 5, 6]},
        {"start": 3, "continuity": [0, 0, 0, 1, 1, 1]},
        {"start": 3, "running": [True, True, True, False, True, True]},
    ],
)
def test_history_context_bootstrap_and_structural_boundaries_are_unavailable(
    kwargs: dict[str, object],
) -> None:
    context = _history_context(**kwargs)

    assert context.available_mask.tolist() == [False, False]
    assert context.previous_mode.tolist() == [0.0, 0.0]
    assert context.dwell_steps.tolist() == [0, 0]


def test_history_context_availability_is_per_side() -> None:
    valid = np.ones((6, 2), dtype=bool)
    valid[2, 0] = False

    context = _history_context(start=3, valid=valid)

    assert context.available_mask.tolist() == [False, True]
    assert context.dwell_steps.tolist() == [0, 1]


def test_history_context_contract_rejects_unavailable_nonzero_storage() -> None:
    with pytest.raises(ValueError, match="Unavailable"):
        ArmHoldHistoryContext(
            previous_mode=np.asarray([1.0, 0.0], dtype=np.float32),
            available_mask=np.asarray([False, False]),
            dwell_steps=np.asarray([0, 0], dtype=np.int64),
        )


@pytest.mark.parametrize("start", [True, 1.5])
def test_history_context_rejects_noninteger_start(start: object) -> None:
    with pytest.raises(ValueError, match="source arrays"):
        _history_context(start=start)  # type: ignore[arg-type]


def test_history_context_arrays_are_immutable() -> None:
    context = _history_context(start=3)

    with pytest.raises(ValueError, match="read-only"):
        context.previous_mode[0] = 1.0


def test_history_sequence_derives_two_causal_label_dwell_snapshots() -> None:
    context = _history_sequence_context(start=4)

    assert context.schema == ARM_HOLD_HISTORY_SEQUENCE_CONTEXT_SCHEMA_V1
    assert context.previous_mode.tolist() == [[0.0, 0.0], [1.0, 0.0]]
    assert context.available_mask.tolist() == [[True, True], [True, True]]
    assert context.dwell_steps.tolist() == [[3, 1], [1, 2]]


def test_history_sequence_bootstrap_is_a_contiguous_suffix() -> None:
    bootstrap = _history_sequence_context(start=0)
    one_step = _history_sequence_context(start=1)

    assert not bootstrap.available_mask.any()
    assert one_step.available_mask.tolist() == [[False, False], [True, True]]
    assert one_step.previous_mode.tolist() == [[0.0, 0.0], [0.0, 1.0]]
    assert one_step.dwell_steps.tolist() == [[0, 0], [1, 1]]


def test_history_sequence_contract_rejects_noncausal_progression() -> None:
    with pytest.raises(ValueError, match="progression"):
        ArmHoldHistorySequenceContext(
            previous_mode=np.asarray([[0, 0], [0, 1]], dtype=np.float32),
            available_mask=np.ones((2, 2), dtype=bool),
            dwell_steps=np.asarray([[3, 4], [7, 2]], dtype=np.int64),
        )


def test_history_timeline_matches_scalar_context_across_boundaries() -> None:
    targets = np.asarray(
        [
            [0, 1],
            [0, 1],
            [0, 0],
            [1, 0],
            [1, 1],
            [1, 1],
            [0, 1],
            [0, 0],
        ],
        dtype=np.float32,
    )
    valid = np.ones_like(targets, dtype=bool)
    valid[2, 0] = False
    valid[5, 1] = False
    frames = [0, 1, 2, 3, 7, 8, 9, 10]
    continuity = [0, 0, 0, 0, 1, 1, 1, 1]
    running = np.asarray([True, True, True, True, True, True, False, True])

    timeline = derive_arm_hold_history_timeline(
        targets=targets,
        valid_mask=valid,
        loss_mask=valid.copy(),
        frame_indices=frames,
        continuity_segments=continuity,
        running_mask=running,
    )

    assert isinstance(timeline, ArmHoldHistoryTimeline)
    assert len(timeline) == len(targets)
    for index in range(len(targets)):
        scalar = derive_arm_hold_history_context(
            targets=targets,
            valid_mask=valid,
            loss_mask=valid.copy(),
            frame_indices=frames,
            continuity_segments=continuity,
            running_mask=running,
            start_index=index,
        )
        derived = timeline.context_at(index)
        assert np.array_equal(derived.previous_mode, scalar.previous_mode)
        assert np.array_equal(derived.available_mask, scalar.available_mask)
        assert np.array_equal(derived.dwell_steps, scalar.dwell_steps)
    with pytest.raises(ValueError, match="out of range"):
        timeline.context_at(len(targets))
    with pytest.raises(ValueError, match="read-only"):
        timeline.dwell_steps[1, 0] = 99


PROFILE_PATH = (
    Path(__file__).parents[1]
    / "dataset_interface"
    / "arm_hold_profiles"
    / "writing_0803_0804_0805_front_h16_v1.json"
)


def _selection(
    *,
    count: int = 20,
    horizon: int = 4,
    frames: np.ndarray | None = None,
    sessions: list[str] | None = None,
    continuity: list[int | None] | None = None,
    labels: np.ndarray | None = None,
    states: np.ndarray | None = None,
    front: np.ndarray | None = None,
):
    return select_arm_hold_window_starts(
        frame_indices=np.arange(count) if frames is None else frames,
        session_states=["RUNNING"] * count if sessions is None else sessions,
        continuity_segments=[0] * count if continuity is None else continuity,
        label_valid_mask=(
            np.ones((count, 2), dtype=bool) if labels is None else labels
        ),
        state_valid_mask=(
            np.ones(count, dtype=bool) if states is None else states
        ),
        front_valid_mask=(
            np.ones(count, dtype=bool) if front is None else front
        ),
        horizon=horizon,
        running_session_state="RUNNING",
    )


def test_single_side_invalid_drops_every_overlapping_window() -> None:
    labels = np.ones((20, 2), dtype=bool)
    labels[5, 0] = False

    result = _selection(labels=labels)

    assert result.candidate_count == 17
    assert result.rejection_counts == {"LABEL_INVALID": 4}
    assert set(result.start_indices).isdisjoint({2, 3, 4, 5})


def test_both_side_invalid_has_same_whole_window_policy() -> None:
    labels = np.ones((20, 2), dtype=bool)
    labels[5] = False

    result = _selection(labels=labels)

    assert result.rejection_counts == {"LABEL_INVALID": 4}


def test_structural_rejections_are_exclusive_and_deterministic() -> None:
    frames = np.arange(20)
    frames[8:] += 1
    sessions = ["RUNNING"] * 20
    sessions[15] = "PAUSED"
    continuity = [0] * 10 + [1] * 10

    first = _selection(frames=frames, sessions=sessions, continuity=continuity)
    second = _selection(frames=frames, sessions=sessions, continuity=continuity)

    assert first == second
    assert sum(first.rejection_counts.values()) + len(first.start_indices) == 17
    assert first.rejection_counts["FRAME_GAP"] == 3
    assert first.rejection_counts["CONTINUITY_BREAK"] == 2
    assert first.rejection_counts["NON_RUNNING"] == 4


def test_state_and_front_validity_only_gate_the_observation_start() -> None:
    states = np.ones(20, dtype=bool)
    front = np.ones(20, dtype=bool)
    states[2] = False
    front[3] = False

    result = _selection(states=states, front=front)

    assert result.rejection_counts == {
        "FRONT_FRAME_INVALID": 1,
        "STATE_INVALID": 1,
    }
    assert 2 not in result.start_indices
    assert 3 not in result.start_indices
    assert 0 in result.start_indices


def _projection_metadata() -> dict[str, object]:
    return {
        "label_specs": {
            "teleop.arm_mode": {
                "schema": ARM_MODE_SCHEMA_V2,
                "sides": ["left", "right"],
                "mode_ids": dict(ARM_MODE_IDS),
                "training_projections": {
                    ARM_HOLD_TRAINING_PROJECTION_ID_V1: (
                        arm_hold_training_projection_metadata()
                    )
                },
            }
        }
    }


def _arm_columns(left: list[str], right: list[str]) -> dict[str, list[object]]:
    result: dict[str, list[object]] = {
        "teleop.arm_mode.schema": [ARM_MODE_SCHEMA_V2] * len(left),
        "teleop.arm_mode.available": [1] * len(left),
    }
    for side, labels in (("left", left), ("right", right)):
        result[f"teleop.arm_mode.{side}.label"] = labels
        result[f"teleop.arm_mode.{side}.id"] = [ARM_MODE_IDS[item] for item in labels]
        result[f"teleop.arm_mode.{side}.scope"] = [
            "arm_and_end_effector" if item == "fixed_hold" else ""
            for item in labels
        ]
    return result


def test_transition_is_computed_before_slicing_window() -> None:
    columns = _arm_columns(
        ["teleop", "fixed_hold", "fixed_hold"],
        ["fixed_hold", "fixed_hold", "fixed_hold"],
    )
    columns["collection.continuity_segment_index"] = [0, 0, 0]

    full = project_arm_hold_training_view(
        columns, metadata=_projection_metadata(), strict=True
    )

    assert full.transition_mask[1].tolist() == [True, False]
    assert full.transition_mask[1:3][0].tolist() == [True, False]


def test_transition_coverage_distinguishes_events_and_overlapping_windows() -> None:
    targets = np.zeros((6, 2), dtype=np.float32)
    targets[1:, 0] = 1.0
    transitions = np.zeros((6, 2), dtype=bool)
    transitions[1, 0] = True
    transitions[5, 1] = True

    coverage = summarize_arm_hold_window_transition_coverage(
        targets=targets,
        transition_mask=transitions,
        start_indices=(0, 1),
        horizon=3,
    )

    assert coverage.event_counts == {
        "left:teleop->fixed_hold": 1,
        "right:fixed_hold->teleop": 1,
    }
    assert coverage.covered_event_counts == {
        "left:teleop->fixed_hold": 1,
    }
    assert coverage.window_counts == {
        "left:teleop->fixed_hold": 2,
    }


def test_transition_coverage_rejects_nonbinary_transition_target() -> None:
    targets = np.zeros((3, 2), dtype=np.float32)
    targets[1, 0] = 0.5
    transitions = np.zeros((3, 2), dtype=bool)
    transitions[1, 0] = True

    with pytest.raises(ArmHoldWindowError, match="binary"):
        summarize_arm_hold_window_transition_coverage(
            targets=targets,
            transition_mask=transitions,
            start_indices=(0,),
            horizon=3,
        )


def test_window_target_counts_use_overlapping_window_prevalence() -> None:
    targets = np.asarray(
        [[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32
    )

    counts = summarize_arm_hold_window_target_counts(
        targets=targets,
        start_indices=(0, 1),
        horizon=3,
    )

    assert counts == {
        "left:teleop": 2,
        "left:fixed_hold": 4,
        "right:teleop": 3,
        "right:fixed_hold": 3,
    }


def test_window_target_counts_reject_nonbinary_admitted_target() -> None:
    targets = np.zeros((3, 2), dtype=np.float32)
    targets[1, 1] = np.nan

    with pytest.raises(ArmHoldWindowError, match="binary"):
        summarize_arm_hold_window_target_counts(
            targets=targets,
            start_indices=(0,),
            horizon=3,
        )


def test_window_location_rejects_nonboolean_transition_membership() -> None:
    with pytest.raises(ArmHoldWindowError, match="must be bool"):
        ArmHoldWindowLocation(
            source_dataset_id="source",
            source_episode_index=0,
            episode_capture_identity_hash="episode",
            start_index=0,
            front_video_frame_index=0,
            continuity_segment_index=0,
            contains_transition=1,  # type: ignore[arg-type]
        )


def test_sample_rejects_partial_side_masks() -> None:
    valid = np.ones((4, 2), dtype=bool)
    valid[1, 0] = False

    with pytest.raises(ArmHoldWindowError, match="fully valid"):
        ArmHoldWindowSample(
            front_rgb=None,
            observation_state=np.zeros(2, dtype=np.float32),
            arm_hold_target=np.zeros((4, 2), dtype=np.float32),
            arm_hold_valid_mask=valid,
            arm_hold_loss_mask=valid,
            arm_hold_transition_mask=np.zeros((4, 2), dtype=bool),
        )


def test_reader_modes_keep_video_lazy_and_convert_bgr_to_rgb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        pytest.skip("pyarrow is unavailable")
    profile = load_arm_hold_dataset_profile(PROFILE_PATH)
    count = 17
    left = ["teleop"] + ["fixed_hold"] * (count - 1)
    right = ["fixed_hold"] * 8 + ["teleop"] * (count - 8)
    columns = {
        "frame_index": list(range(count)),
        "timestamp": [index / 15.0 for index in range(count)],
        "session_state": ["RUNNING"] * count,
        "collection.continuity_segment_index": [0] * count,
        "observation.state": [[float(index), float(index + 1)] for index in range(count)],
        "observation.images.front.video_frame_index": list(range(count)),
        **_arm_columns(left, right),
    }
    parquet = tmp_path / "episode.parquet"
    pq.write_table(pa.table(columns), parquet)
    video = tmp_path / "front.mp4"
    video.write_bytes(b"not-decoded-in-state-only-mode")
    source_id = profile.sources[0].source_dataset_id
    identity = "synthetic-identity"
    binding = ArmHoldWindowEpisodeBinding(
        source_dataset_id=source_id,
        source_episode_index=0,
        episode_uuid="synthetic-episode",
        episode_capture_identity_hash=identity,
        annotation_set_id="synthetic-annotation",
        frame_count=count,
        state_dim=2,
        parquet_path=parquet,
        front_video_path=video,
        dataset_info=_projection_metadata(),
    )
    location = ArmHoldWindowLocation(
        source_dataset_id=source_id,
        source_episode_index=0,
        episode_capture_identity_hash=identity,
        start_index=0,
        front_video_frame_index=0,
        continuity_segment_index=0,
        contains_transition=True,
    )
    later_location = ArmHoldWindowLocation(
        source_dataset_id=source_id,
        source_episode_index=0,
        episode_capture_identity_hash=identity,
        start_index=1,
        front_video_frame_index=1,
        continuity_segment_index=0,
        contains_transition=True,
    )
    inventory = ArmHoldWindowInventory(
        manifest_hash="manifest-hash",
        profile=profile,
        split="train",
        episodes=(binding,),
        locations=(later_location, location),
        candidate_count=2,
        rejection_counts={},
        window_counts_by_source={source_id: 2},
    )

    assert [item.start_index for item in inventory.locations] == [0, 1]

    with ArmHoldWindowReader(inventory, input_mode="state_only") as reader:
        sample = reader[0]

    assert sample.front_rgb is None
    assert sample.observation_state.tolist() == [0.0, 1.0]
    assert sample.arm_hold_target.shape == (16, 2)
    assert sample.arm_hold_transition_mask[1].tolist() == [True, False]
    assert sample.arm_hold_transition_mask[8].tolist() == [False, True]
    assert sample.arm_hold_history_context is None

    with ArmHoldWindowReader(
        inventory,
        input_mode="state_only",
        include_history_context=True,
    ) as reader:
        bootstrap = reader[0].arm_hold_history_context
        continued = reader[1].arm_hold_history_context

    assert bootstrap is not None
    assert bootstrap.available_mask.tolist() == [False, False]
    assert continued is not None
    assert continued.previous_mode.tolist() == [0.0, 1.0]
    assert continued.available_mask.tolist() == [True, True]
    assert continued.dwell_steps.tolist() == [1, 1]

    with ArmHoldWindowReader(
        inventory,
        input_mode="state_only",
        include_history_context=True,
        history_context_steps=2,
    ) as reader:
        bootstrap_sequence = reader[0].arm_hold_history_sequence_context
        continued_sequence = reader[1].arm_hold_history_sequence_context

    assert bootstrap_sequence is not None
    assert not bootstrap_sequence.available_mask.any()
    assert continued_sequence is not None
    assert continued_sequence.available_mask.tolist() == [
        [False, False],
        [True, True],
    ]
    assert continued_sequence.previous_mode[1].tolist() == [0.0, 1.0]
    assert continued_sequence.dwell_steps[1].tolist() == [1, 1]

    stale_inventory = ArmHoldWindowInventory(
        manifest_hash="manifest-hash",
        profile=profile,
        split="train",
        episodes=(binding,),
        locations=(replace(location, contains_transition=False),),
        candidate_count=1,
        rejection_counts={},
        window_counts_by_source={source_id: 1},
    )
    with ArmHoldWindowReader(stale_inventory, input_mode="state_only") as reader:
        with pytest.raises(ArmHoldWindowError, match="membership changed"):
            reader[0]

    with ArmHoldWindowReader(inventory, input_mode="front_only") as reader:
        monkeypatch.setattr(
            reader._videos,
            "read",
            lambda *_: np.asarray([[[1, 2, 3]]], dtype=np.uint8),
        )
        image_sample = reader[0]

    assert image_sample.observation_state is None
    assert image_sample.front_rgb.tolist() == [[[3, 2, 1]]]
