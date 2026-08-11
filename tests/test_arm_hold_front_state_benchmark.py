from __future__ import annotations

import argparse

import pytest

from todos.probes.probe_arm_hold_front_state_batch_benchmark import (
    _batch_sizes,
    _case_indices,
)


def test_batch_size_contract_is_sorted_unique_and_bounded() -> None:
    assert _batch_sizes("16,1,8,16") == (1, 8, 16)

    for value in ("", "0,1", "1,257", "one,2"):
        with pytest.raises(argparse.ArgumentTypeError):
            _batch_sizes(value)


def test_case_order_keeps_canonical_or_seeded_episode_local_indices() -> None:
    source = (10, 11, 12, 13, 14)

    assert _case_indices(
        source,
        order="canonical_within_episode",
        batch_size=3,
        seed=23,
    ) == (10, 11, 12)
    shuffled = _case_indices(
        source,
        order="shuffled_within_episode",
        batch_size=3,
        seed=23,
    )
    assert shuffled == _case_indices(
        source,
        order="shuffled_within_episode",
        batch_size=3,
        seed=23,
    )
    assert len(shuffled) == len(set(shuffled)) == 3
    assert set(shuffled).issubset(source)
    assert source == (10, 11, 12, 13, 14)


def test_case_order_rejects_an_undeclared_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported benchmark order"):
        _case_indices((0, 1), order="unknown", batch_size=1, seed=0)
