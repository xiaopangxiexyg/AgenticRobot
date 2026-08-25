from __future__ import annotations

import argparse

import pytest

from teleop_runtime.deploy.replay_cli_validation import (
    validate_debug_replay_chunk_size,
    validate_debug_replay_window_and_sources,
    validate_replay_prediction_mode,
)


def _args(**overrides: object) -> argparse.Namespace:
    values = {
        "debug_replay_chunk_size": 32,
        "debug_replay_start_index": 0,
        "debug_replay_end_index": None,
        "action_hz_start_index": None,
        "action_hz_end_index": None,
        "debug_replay_dataset": None,
        "debug_replay_parquet": None,
        "action_hz_dataset": None,
        "action_hz_parquet": None,
        "policy_spec": "hold",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_replay_chunk_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="debug-replay-chunk-size must be positive"):
        validate_debug_replay_chunk_size(_args(debug_replay_chunk_size=0))


def test_replay_window_accepts_open_end() -> None:
    validate_debug_replay_window_and_sources(_args(debug_replay_start_index=3))


def test_replay_window_end_must_follow_start() -> None:
    with pytest.raises(
        ValueError,
        match="debug-replay-end-index must be greater than --debug-replay-start-index",
    ):
        validate_debug_replay_window_and_sources(
            _args(debug_replay_start_index=4, debug_replay_end_index=4)
        )


def test_action_hz_window_defaults_to_replay_window() -> None:
    with pytest.raises(
        ValueError,
        match="action-hz-end-index must be greater than --action-hz-start-index",
    ):
        validate_debug_replay_window_and_sources(
            _args(debug_replay_start_index=4, debug_replay_end_index=8, action_hz_end_index=4)
        )


def test_action_hz_window_can_use_an_explicit_start() -> None:
    validate_debug_replay_window_and_sources(
        _args(action_hz_start_index=4, action_hz_end_index=8)
    )


def test_replay_dataset_and_parquet_are_mutually_exclusive() -> None:
    with pytest.raises(
        ValueError, match="Use only one of --debug-replay-dataset or --debug-replay-parquet"
    ):
        validate_debug_replay_window_and_sources(
            _args(debug_replay_dataset="dataset", debug_replay_parquet="episode.parquet")
        )


def test_action_hz_dataset_and_parquet_are_mutually_exclusive() -> None:
    with pytest.raises(
        ValueError, match="Use only one of --action-hz-dataset or --action-hz-parquet"
    ):
        validate_debug_replay_window_and_sources(
            _args(action_hz_dataset="dataset", action_hz_parquet="hz.parquet")
        )


@pytest.mark.parametrize("policy_spec", ["", "hold", "builtin:hold"])
def test_replay_accepts_hold_policy_aliases(policy_spec: str) -> None:
    validate_debug_replay_window_and_sources(
        _args(debug_replay_dataset="dataset", policy_spec=policy_spec)
    )


def test_replay_rejects_non_hold_policy() -> None:
    with pytest.raises(
        ValueError,
        match=r"debug-replay-\* replaces --policy; omit --policy in replay mode",
    ):
        validate_debug_replay_window_and_sources(
            _args(debug_replay_dataset="dataset", policy_spec="pi05")
        )


def test_threaded_prediction_is_rejected_for_replay() -> None:
    with pytest.raises(
        ValueError,
        match="dataset replay owns prediction transactions and must use --policy-prediction-mode synchronous",
    ):
        validate_replay_prediction_mode(
            _args(debug_replay_dataset="dataset"), threaded_prediction=True
        )


def test_non_threaded_prediction_is_allowed_for_replay() -> None:
    validate_replay_prediction_mode(
        _args(debug_replay_dataset="dataset"), threaded_prediction=False
    )
