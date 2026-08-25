"""Pure replay-related CLI admission checks for policy deployment.

The deployment facade keeps the call sites in their historical validation
order, while this module owns only replay-window, source-selection, and
prediction-mode checks.  It does not load datasets, create workers, inspect
devices, or dispatch actions.
"""

from __future__ import annotations

import argparse


def validate_replay_prediction_mode(
    args: argparse.Namespace, *, threaded_prediction: bool
) -> None:
    """Reject threaded prediction when dataset replay owns the transaction."""

    if threaded_prediction and (
        args.debug_replay_dataset is not None or args.debug_replay_parquet is not None
    ):
        raise ValueError(
            "dataset replay owns prediction transactions and must use "
            "--policy-prediction-mode synchronous."
        )


def validate_debug_replay_chunk_size(args: argparse.Namespace) -> None:
    """Require a positive replay action chunk size."""

    if int(args.debug_replay_chunk_size) <= 0:
        raise ValueError("--debug-replay-chunk-size must be positive.")


def validate_debug_replay_window_and_sources(args: argparse.Namespace) -> None:
    """Validate replay/action-rate windows and mutually exclusive sources."""

    if args.debug_replay_end_index is not None and int(
        args.debug_replay_end_index
    ) <= int(args.debug_replay_start_index):
        raise ValueError(
            "--debug-replay-end-index must be greater than --debug-replay-start-index."
        )
    action_hz_start = (
        args.debug_replay_start_index
        if args.action_hz_start_index is None
        else args.action_hz_start_index
    )
    action_hz_end = (
        args.debug_replay_end_index
        if args.action_hz_end_index is None
        else args.action_hz_end_index
    )
    if action_hz_end is not None and int(action_hz_end) <= int(action_hz_start):
        raise ValueError(
            "--action-hz-end-index must be greater than --action-hz-start-index."
        )
    if args.debug_replay_dataset and args.debug_replay_parquet:
        raise ValueError(
            "Use only one of --debug-replay-dataset or --debug-replay-parquet."
        )
    if args.action_hz_dataset and args.action_hz_parquet:
        raise ValueError("Use only one of --action-hz-dataset or --action-hz-parquet.")
    if (args.debug_replay_dataset or args.debug_replay_parquet) and str(
        args.policy_spec or "hold"
    ).strip().lower() not in {"", "hold", "builtin:hold"}:
        raise ValueError(
            "--debug-replay-* replaces --policy; omit --policy in replay mode."
        )


__all__ = [
    "validate_debug_replay_chunk_size",
    "validate_debug_replay_window_and_sources",
    "validate_replay_prediction_mode",
]
