"""Additive CLI registration for the default-off RealTeleop HITL profile."""

from __future__ import annotations

import argparse

from dataset_contracts.hitl import EXCLUSIVE_HITL_POLICY_MODES

from ..cli_value_parsers import _non_negative_int, _positive_float
from .cli_value_parsers import _positive_int


def add_exclusive_hitl_arguments(parser: argparse.ArgumentParser) -> None:
    """Extend a resolved application parser without changing the frozen core."""

    if not isinstance(parser, argparse.ArgumentParser):
        raise TypeError("parser must be an argparse.ArgumentParser.")
    if any(action.dest == "hitl_policy_control" for action in parser._actions):
        return
    parser.add_argument(
        "--hitl-policy-control",
        action="store_true",
        help=(
            "Enable the default-off exclusive Human/Policy collection profile. "
            "Both sources retain the ordinary RealTeleop safety and dispatch path."
        ),
    )
    parser.add_argument(
        "--hitl-policy-mode",
        choices=EXCLUSIVE_HITL_POLICY_MODES,
        default=EXCLUSIVE_HITL_POLICY_MODES[0],
        help=(
            "Policy producer kind for exclusive HITL. RTC accepts only committed "
            "PolicyActionEmission input and is held/no-action until the repository's "
            "RTC real-motion authorization gates are completed."
        ),
    )
    parser.add_argument(
        "--hitl-initial-source",
        choices=("policy", "human"),
        default="policy",
        help="Exclusive source selected when the HITL runtime starts.",
    )
    parser.add_argument(
        "--hitl-takeover-key",
        type=str,
        default="u",
        help="Single terminal key that requests Policy-to-Human takeover.",
    )
    parser.add_argument(
        "--hitl-policy-resume-key",
        type=str,
        default="i",
        help="Single terminal key that requests Human-to-Policy return.",
    )
    parser.add_argument(
        "--hitl-policy-artifact-json",
        type=str,
        default="",
        help=(
            "Existing policy artifact JSON for normal HITL inference. It is not "
            "accepted by the RTC committed-emission profile."
        ),
    )
    parser.add_argument(
        "--hitl-openpi-host",
        type=str,
        default="127.0.0.1",
        help="Existing OpenPI server host used by normal HITL inference.",
    )
    parser.add_argument(
        "--hitl-openpi-port",
        type=_positive_int,
        default=8000,
        help="Existing OpenPI server port used by normal HITL inference.",
    )
    parser.add_argument(
        "--hitl-policy-prompt",
        type=str,
        default="",
        help="Optional prompt passed to the existing normal policy client.",
    )
    parser.add_argument(
        "--hitl-policy-action-hz",
        type=_positive_float,
        default=15.0,
        help="Normal policy action cadence in Hz.",
    )
    parser.add_argument(
        "--hitl-policy-inference-hz",
        type=_positive_float,
        default=15.0,
        help="Normal policy inference request cadence in Hz.",
    )
    parser.add_argument(
        "--hitl-policy-chunk-execution-steps",
        type=_non_negative_int,
        default=0,
        help="Normal policy chunk steps to execute; 0 keeps artifact/client default.",
    )
    parser.add_argument(
        "--hitl-policy-prefetch-threshold",
        type=_non_negative_int,
        default=4,
        help="Existing normal policy-source prefetch queue threshold.",
    )


__all__ = ["add_exclusive_hitl_arguments"]
