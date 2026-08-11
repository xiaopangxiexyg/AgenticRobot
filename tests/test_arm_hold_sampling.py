from __future__ import annotations

from dataset_interface.arm_hold_windows import ArmHoldWindowLocation
from model.arm_hold import (
    arm_hold_natural_plus_random_once_batches,
    arm_hold_natural_plus_transition_once_batches,
    arm_hold_natural_window_batches,
)


def _locations() -> tuple[ArmHoldWindowLocation, ...]:
    return tuple(
        ArmHoldWindowLocation(
            source_dataset_id="source",
            source_episode_index=episode,
            episode_capture_identity_hash=f"episode-{episode}",
            start_index=start,
            front_video_frame_index=start,
            continuity_segment_index=0,
            contains_transition=start in {1, count - 1},
        )
        for episode, count in ((0, 5), (1, 4), (2, 3))
        for start in range(count)
    )


def test_natural_batches_are_deterministic_complete_and_episode_local() -> None:
    locations = _locations()

    first = arm_hold_natural_window_batches(
        locations, batch_size=2, seed=7, epoch=0
    )
    repeated = arm_hold_natural_window_batches(
        locations, batch_size=2, seed=7, epoch=0
    )
    next_epoch = arm_hold_natural_window_batches(
        locations, batch_size=2, seed=7, epoch=1
    )

    assert first == repeated
    assert first != next_epoch
    flattened = [index for batch in first for index in batch]
    assert sorted(flattened) == list(range(len(locations)))
    assert len(flattened) == len(set(flattened))
    for batch in first:
        identities = {
            locations[index].episode_capture_identity_hash
            for index in batch
        }
        assert len(identities) == 1


def test_unshuffled_batches_follow_canonical_location_order() -> None:
    locations = _locations()

    batches = arm_hold_natural_window_batches(
        locations,
        batch_size=3,
        seed=100,
        epoch=9,
        shuffle=False,
    )

    assert [index for batch in batches for index in batch] == list(
        range(len(locations))
    )


def test_transition_bonus_repeats_only_unique_transition_windows_once() -> None:
    locations = _locations()

    batches = arm_hold_natural_plus_transition_once_batches(
        locations,
        batch_size=3,
        seed=11,
        epoch=0,
    )

    flattened = [index for batch in batches for index in batch]
    for index, location in enumerate(locations):
        assert flattened.count(index) == (2 if location.contains_transition else 1)
    assert len(flattened) == len(locations) + sum(
        location.contains_transition for location in locations
    )
    for batch in batches:
        assert len(
            {
                locations[index].episode_capture_identity_hash
                for index in batch
            }
        ) == 1


def test_transition_bonus_is_deterministic_but_changes_by_epoch() -> None:
    locations = _locations()

    first = arm_hold_natural_plus_transition_once_batches(
        locations, batch_size=2, seed=13, epoch=0
    )
    repeated = arm_hold_natural_plus_transition_once_batches(
        locations, batch_size=2, seed=13, epoch=0
    )
    next_epoch = arm_hold_natural_plus_transition_once_batches(
        locations, batch_size=2, seed=13, epoch=1
    )

    assert first == repeated
    assert first != next_epoch


def test_random_bonus_is_compute_matched_unique_and_deterministic() -> None:
    locations = _locations()
    bonus_count = sum(location.contains_transition for location in locations)

    first = arm_hold_natural_plus_random_once_batches(
        locations,
        bonus_count=bonus_count,
        batch_size=2,
        seed=19,
        epoch=0,
    )
    repeated = arm_hold_natural_plus_random_once_batches(
        locations,
        bonus_count=bonus_count,
        batch_size=2,
        seed=19,
        epoch=0,
    )

    assert first == repeated
    flattened = [index for batch in first for index in batch]
    multiplicities = [flattened.count(index) for index in range(len(locations))]
    assert multiplicities.count(2) == bonus_count
    assert multiplicities.count(1) == len(locations) - bonus_count
    assert len(flattened) == len(locations) + bonus_count


def test_random_bonus_rejects_out_of_range_count() -> None:
    locations = _locations()

    for bonus_count in (-1, len(locations) + 1):
        try:
            arm_hold_natural_plus_random_once_batches(
                locations,
                bonus_count=bonus_count,
                batch_size=2,
                seed=0,
                epoch=0,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid random bonus count was accepted")


def test_natural_batches_reject_invalid_batch_or_epoch() -> None:
    locations = _locations()

    for kwargs in ({"batch_size": 0, "epoch": 0}, {"batch_size": 2, "epoch": -1}):
        try:
            arm_hold_natural_window_batches(locations, seed=0, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid natural batching arguments were accepted")
