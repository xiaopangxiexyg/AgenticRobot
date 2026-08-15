"""Training batch order for the frozen arm-hold window inventory.

This module owns only deterministic traversal of dataset-interface window
identities. It does not define a learner, loss, optimizer, or runtime policy.
"""

from __future__ import annotations

import random
from typing import Sequence

from dataset_interface.arm_hold_windows import ArmHoldWindowLocation


def _arm_hold_window_batches(
    locations: Sequence[ArmHoldWindowLocation],
    *,
    batch_size: int,
    seed: int,
    epoch: int,
    shuffle: bool = True,
    bonus_indices: frozenset[int],
) -> tuple[tuple[int, ...], ...]:
    resolved_batch_size = int(batch_size)
    resolved_epoch = int(epoch)
    if resolved_batch_size <= 0 or resolved_epoch < 0:
        raise ValueError("batch_size must be positive and epoch nonnegative.")
    grouped: dict[str, list[int]] = {}
    for index, location in enumerate(locations):
        grouped.setdefault(location.episode_capture_identity_hash, []).append(index)
        if index in bonus_indices:
            grouped[location.episode_capture_identity_hash].append(index)
    identities = sorted(grouped)
    if shuffle:
        generator = random.Random(int(seed) + resolved_epoch * 1_000_003)
        generator.shuffle(identities)
        for identity in identities:
            generator.shuffle(grouped[identity])
    batches: list[tuple[int, ...]] = []
    for identity in identities:
        indices = grouped[identity]
        for start in range(0, len(indices), resolved_batch_size):
            batches.append(tuple(indices[start : start + resolved_batch_size]))
    return tuple(batches)


def arm_hold_natural_window_batches(
    locations: Sequence[ArmHoldWindowLocation],
    *,
    batch_size: int,
    seed: int,
    epoch: int,
    shuffle: bool = True,
) -> tuple[tuple[int, ...], ...]:
    """Visit every legal window once while retaining Episode IO locality."""

    return _arm_hold_window_batches(
        locations,
        batch_size=batch_size,
        seed=seed,
        epoch=epoch,
        shuffle=shuffle,
        bonus_indices=frozenset(),
    )


def arm_hold_natural_plus_transition_once_batches(
    locations: Sequence[ArmHoldWindowLocation],
    *,
    batch_size: int,
    seed: int,
    epoch: int,
    shuffle: bool = True,
) -> tuple[tuple[int, ...], ...]:
    """Visit stable windows once and transition windows exactly twice.

    The second exposure is window-level and direction-neutral.  It does not
    balance directions or create a new transition-event identity.
    """

    return _arm_hold_window_batches(
        locations,
        batch_size=batch_size,
        seed=seed,
        epoch=epoch,
        shuffle=shuffle,
        bonus_indices=frozenset(
            index
            for index, location in enumerate(locations)
            if location.contains_transition
        ),
    )


def arm_hold_natural_plus_random_once_batches(
    locations: Sequence[ArmHoldWindowLocation],
    *,
    bonus_count: int,
    batch_size: int,
    seed: int,
    epoch: int,
    shuffle: bool = True,
) -> tuple[tuple[int, ...], ...]:
    """Compute-matched control that repeats random unique windows once."""

    resolved_bonus_count = int(bonus_count)
    if resolved_bonus_count < 0 or resolved_bonus_count > len(locations):
        raise ValueError("bonus_count must be between zero and location count.")
    bonus_generator = random.Random(
        int(seed) + int(epoch) * 1_000_003 + 97_409
    )
    bonus_indices = frozenset(
        bonus_generator.sample(range(len(locations)), resolved_bonus_count)
    )
    return _arm_hold_window_batches(
        locations,
        batch_size=batch_size,
        seed=seed,
        epoch=epoch,
        shuffle=shuffle,
        bonus_indices=bonus_indices,
    )


__all__ = [
    "arm_hold_natural_plus_transition_once_batches",
    "arm_hold_natural_plus_random_once_batches",
    "arm_hold_natural_window_batches",
]
