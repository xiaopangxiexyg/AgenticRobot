"""Evaluate fixed rotational-admittance candidates with saved wrench in MuJoCo.

This module is an offline diagnostic only and never constructs robot hardware.
The cautious-3deg branch and explicit admitted normal-parity 5-degree branches
can exercise the production rotation Active installer.  Other larger candidates
remain replay-only.  The candidate set is intentionally declared in code before
the recorded trace is evaluated so that candidate selection remains auditable.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from robot_motion.position_control.rotational_admittance import (
    ROTATIONAL_ADMITTANCE_PROFILE_CHOICES,
    RotationalAdmittanceConfig,
    rotational_admittance_config_for_profile,
)
from robot_motion.rm65.position_profile import (
    RM65_BIMANUAL_POSITION_V2_BALANCED_NAME,
    RM65_POSITION_CONTROL_PROFILE_CHOICES,
)
from teleop_runtime.real_teleop.position_admittance_shadow import (
    POSITION_ADMITTANCE_ACTIVE_DEFAULT_ENDPOINT_POLICY,
    POSITION_ADMITTANCE_ACTIVE_ENDPOINT_POLICY_CHOICES,
)

from .replay_position_admittance_ik_trace import SIDES
from .replay_position_admittance_mujoco import (
    OFFLINE_NORMAL_PARITY_ENVELOPE,
    OFFLINE_NORMAL_PARITY_MAX_TOOL_EXCURSION_M,
    PRODUCTION_NORMAL_PARITY_ENVELOPE,
    _replay_recorded_wrench_through_mujoco_evaluation,
)


ROTATIONAL_ADMITTANCE_ENVELOPES = ROTATIONAL_ADMITTANCE_PROFILE_CHOICES


def rotational_admittance_config_for_envelope(
    envelope: str,
) -> RotationalAdmittanceConfig:
    """Return one predeclared, bounded, three-axis SO(3) candidate."""

    return rotational_admittance_config_for_profile(envelope)


def replay_recorded_rotational_admittance_through_mujoco(
    path: Path,
    *,
    envelope: str = "cautious-3deg",
    side: str = "left",
    replay_hz: float = 60.0,
    release_tail_sec: float = 10.0,
    interpolation_modes: Sequence[str] = ("zoh",),
    combined_translation_rotation_active: bool = False,
    offline_combined_normal_parity: bool = False,
    production_combined_normal_parity: bool = False,
    endpoint_policy: str = POSITION_ADMITTANCE_ACTIVE_DEFAULT_ENDPOINT_POLICY,
    position_execution_profile_name: str = RM65_BIMANUAL_POSITION_V2_BALANCED_NAME,
    translation_filter_tau_sec: float | None = None,
    rotation_filter_tau_sec: float | None = None,
    translation_output_follow_tau_sec: float | None = None,
    rotation_output_follow_tau_sec: float | None = None,
    translation_max_output_acceleration_m_s2: float | None = None,
    rotation_max_output_angular_acceleration_rad_s2: float | None = None,
) -> dict[str, Any]:
    if sum(
        bool(value)
        for value in (
            combined_translation_rotation_active,
            offline_combined_normal_parity,
            production_combined_normal_parity,
        )
    ) > 1:
        raise ValueError(
            "production cautious, production normal parity, and offline "
            "normal parity modes are mutually exclusive"
        )
    config = rotational_admittance_config_for_envelope(envelope)
    max_rotation_deg = math.degrees(max(config.max_offset_rad)) + 0.5
    report, _ = _replay_recorded_wrench_through_mujoco_evaluation(
        Path(path),
        side=side,
        admittance_envelope=(
            PRODUCTION_NORMAL_PARITY_ENVELOPE
            if production_combined_normal_parity
            else "cautious-active"
            if combined_translation_rotation_active
            else OFFLINE_NORMAL_PARITY_ENVELOPE
            if (
                offline_combined_normal_parity
                or production_combined_normal_parity
            )
            else "recorded-shadow"
        ),
        replay_hz=replay_hz,
        release_tail_sec=release_tail_sec,
        interpolation_modes=interpolation_modes,
        max_tool_excursion_m=(
            OFFLINE_NORMAL_PARITY_MAX_TOOL_EXCURSION_M
            if (
                offline_combined_normal_parity
                or production_combined_normal_parity
            )
            else None
        ),
        rotational_config=config,
        max_tool_rotation_excursion_deg=max_rotation_deg,
        combined_translation_rotation_active=(
            combined_translation_rotation_active
            or production_combined_normal_parity
        ),
        endpoint_policy=endpoint_policy,
        position_execution_profile_name=position_execution_profile_name,
        translation_filter_tau_sec=translation_filter_tau_sec,
        rotation_filter_tau_sec=rotation_filter_tau_sec,
        translation_output_follow_tau_sec=translation_output_follow_tau_sec,
        rotation_output_follow_tau_sec=rotation_output_follow_tau_sec,
        translation_max_output_acceleration_m_s2=(
            translation_max_output_acceleration_m_s2
        ),
        rotation_max_output_angular_acceleration_rad_s2=(
            rotation_max_output_angular_acceleration_rad_s2
        ),
    )
    report["rotational_envelope"] = str(envelope).strip().lower()
    report["max_tool_rotation_excursion_deg"] = max_rotation_deg
    report["offline_combined_normal_parity"] = bool(
        offline_combined_normal_parity
    )
    report["production_combined_normal_parity"] = bool(
        production_combined_normal_parity
    )
    report["active_endpoint_policy"] = endpoint_policy
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--side", choices=SIDES, default="left")
    parser.add_argument(
        "--envelope",
        choices=ROTATIONAL_ADMITTANCE_ENVELOPES,
        default="cautious-3deg",
    )
    parser.add_argument("--replay-hz", type=float, default=60.0)
    parser.add_argument("--release-tail-sec", type=float, default=10.0)
    parser.add_argument(
        "--combined-translation-rotation-active",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--offline-combined-normal-parity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Device-free counterfactual with the established 8 cm / 60 mm/s "
            "translation envelope plus the selected bounded rotation profile."
        ),
    )
    parser.add_argument(
        "--production-combined-normal-parity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Exercise the production normal_parity translation plus "
            "an admitted 5-degree rotation Active installer in device-free MuJoCo."
        ),
    )
    parser.add_argument(
        "--active-endpoint-policy",
        choices=POSITION_ADMITTANCE_ACTIVE_ENDPOINT_POLICY_CHOICES,
        default=POSITION_ADMITTANCE_ACTIVE_DEFAULT_ENDPOINT_POLICY,
        help=(
            "Select the exact production Active endpoint policy. formal_teleop "
            "is valid only with --production-combined-normal-parity."
        ),
    )
    parser.add_argument(
        "--interpolation",
        choices=("zoh", "linear", "both"),
        default="zoh",
    )
    parser.add_argument(
        "--position-execution-profile",
        choices=tuple(
            value
            for value in RM65_POSITION_CONTROL_PROFILE_CHOICES
            if value != "legacy"
        ),
        default=RM65_BIMANUAL_POSITION_V2_BALANCED_NAME,
    )
    parser.add_argument("--translation-filter-tau-sec", type=float, default=None)
    parser.add_argument("--rotation-filter-tau-sec", type=float, default=None)
    parser.add_argument("--translation-output-follow-tau-sec", type=float, default=None)
    parser.add_argument("--rotation-output-follow-tau-sec", type=float, default=None)
    parser.add_argument("--translation-max-output-acceleration-m-s2", type=float, default=None)
    parser.add_argument("--rotation-max-output-angular-acceleration-rad-s2", type=float, default=None)
    args = parser.parse_args()
    modes = (
        ("zoh", "linear")
        if args.interpolation == "both"
        else (args.interpolation,)
    )
    report = replay_recorded_rotational_admittance_through_mujoco(
        args.log,
        envelope=args.envelope,
        side=args.side,
        replay_hz=args.replay_hz,
        release_tail_sec=args.release_tail_sec,
        interpolation_modes=modes,
        combined_translation_rotation_active=(
            args.combined_translation_rotation_active
        ),
        offline_combined_normal_parity=(
            args.offline_combined_normal_parity
        ),
        production_combined_normal_parity=(
            args.production_combined_normal_parity
        ),
        endpoint_policy=args.active_endpoint_policy,
        position_execution_profile_name=args.position_execution_profile,
        translation_filter_tau_sec=args.translation_filter_tau_sec,
        rotation_filter_tau_sec=args.rotation_filter_tau_sec,
        translation_output_follow_tau_sec=args.translation_output_follow_tau_sec,
        rotation_output_follow_tau_sec=args.rotation_output_follow_tau_sec,
        translation_max_output_acceleration_m_s2=(
            args.translation_max_output_acceleration_m_s2
        ),
        rotation_max_output_angular_acceleration_rad_s2=(
            args.rotation_max_output_angular_acceleration_rad_s2
        ),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
