"""Pure RTC CLI validation for policy deployment."""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable

from robot_motion.deploy.async_producer import (
    MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1,
    RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1,
    RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1,
    RTC_3C_VIRTUAL_SHADOW_PROFILE_V1,
)
from robot_motion.deploy.schema import ALL_SIDES

from .hand_backend_runtime import HAND_BACKEND_DMTAC_GRIPPER
from .policy_binding import parse_policy_kwargs
from .rtc_debug import (
    RTC_DEBUG_LEVEL_OFF,
    RTC_DEBUG_LEVEL_SUMMARY,
    normalize_rtc_debug_level,
)


def validate_rtc_cli_args(
    args: argparse.Namespace,
    *,
    rtc_profile_id: str,
    rtc_profile_enabled: bool,
    rtc_real_shadow: bool,
    hand_backend_mode_for_side: Callable[[argparse.Namespace, str], str],
) -> None:
    """Validate RTC-related CLI values without creating runtime resources.

    The deployment entrypoint keeps ownership of orchestration and safety
    gates.  This helper owns only the existing RTC profile admission rules and
    effective CLI defaults, preserving their validation order and error text.
    """

    requested_rtc_debug_level = getattr(args, "rtc_debug_level", None)
    if requested_rtc_debug_level is None:
        rtc_debug_level = (
            RTC_DEBUG_LEVEL_SUMMARY
            if bool(getattr(args, "debug", False)) and rtc_profile_enabled
            else RTC_DEBUG_LEVEL_OFF
        )
    else:
        rtc_debug_level = normalize_rtc_debug_level(requested_rtc_debug_level)
    if rtc_debug_level != RTC_DEBUG_LEVEL_OFF and not bool(
        getattr(args, "debug", False)
    ):
        raise ValueError("--rtc-debug-level summary/trace requires --debug.")
    if rtc_debug_level != RTC_DEBUG_LEVEL_OFF and not rtc_profile_enabled:
        raise ValueError("--rtc-debug-level summary/trace requires an RTC profile.")
    args._rtc_debug_level = rtc_debug_level
    model_transport_profiles = {
        RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1,
        MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1,
    }
    requested_model_rtc_action_hz = getattr(args, "rtc_action_hz", None)
    if requested_model_rtc_action_hz is not None:
        if rtc_profile_id != MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1:
            raise ValueError(
                "--rtc-action-hz is only supported with the conditioned "
                "model-RTC profile."
            )
        if float(requested_model_rtc_action_hz) not in {15.0, 30.0}:
            raise ValueError("--rtc-action-hz must be 15 or 30 Hz.")
    if rtc_real_shadow and rtc_profile_id not in model_transport_profiles:
        raise ValueError(
            "--rtc-real-shadow requires an explicit model-backed RTC profile; "
            "it is unavailable for RTC-3c virtual policy or sync deployment."
        )
    if (
        rtc_real_shadow
        and bool(args.send_actions)
        and getattr(args, "wait_for_ready_confirm", None) is False
    ):
        raise ValueError(
            "Active model RTC requires --wait-for-ready-confirm after the "
            "dataset-frame qpos initialization; --no-wait-for-ready-confirm "
            "is observation-only."
        )
    if rtc_profile_enabled:
        if rtc_profile_id not in {
            RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1,
            RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1,
            MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1,
        }:
            raise ValueError(f"Unsupported RTC profile: {rtc_profile_id!r}.")
        if (
            rtc_profile_id == RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1
            and bool(args.send_actions)
        ):
            # RTC-3c is a synthetic builtin-hold producer.  It has no model
            # action source that can be admitted to a hardware executor.  The
            # model-backed profiles below deliberately do not share this
            # restriction: explicit --send-actions is the caller's dispatch
            # authority after the normal run_policy safety gates pass.
            raise ValueError(
                "RTC-3c virtual profile has no model action source and does not "
                "accept --send-actions."
            )
        if args.debug_replay_dataset or args.debug_replay_parquet:
            raise ValueError("RTC profiles do not modify or consume debug replay.")
        if args.action_hz_dataset or args.action_hz_parquet:
            raise ValueError(
                "RTC profiles do not consume replay/reference rate inputs."
            )
        if bool(args.append_action_chunks) or str(args.action_queue_policy) == "append":
            raise ValueError(
                "RTC profiles use timed mutable-future merge, not queue append."
            )
        if rtc_profile_id == RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1:
            if str(args.policy_spec or "hold").strip().lower() not in {
                "",
                "hold",
                "builtin:hold",
            }:
                raise ValueError(
                    "RTC-3c is restricted to the builtin hold virtual policy."
                )
            if args.policy_artifact_json is not None:
                raise ValueError(
                    "RTC-3c virtual shadow does not bind a model artifact yet."
                )
            if parse_policy_kwargs(args.policy_kwargs_json):
                raise ValueError("RTC-3c builtin hold policy accepts no policy kwargs.")
        elif args.policy_artifact_json is None:
            raise ValueError(
                "Model-backed RTC requires an explicit --policy-artifact-json "
                "sidecar; model identity must not be inferred from a model name."
            )
        if (
            rtc_profile_id in model_transport_profiles
            and not rtc_real_shadow
            and (
                bool(args.left_arm_ip)
                or bool(args.right_arm_ip)
                or bool(args.left_hand_ip)
                or bool(args.right_hand_ip)
                or (
                    bool(getattr(args, "hands_enabled", False))
                    and any(
                        hand_backend_mode_for_side(args, side)
                        == HAND_BACKEND_DMTAC_GRIPPER
                        for side in ALL_SIDES
                    )
                )
            )
        ):
            raise ValueError(
                "Model-backed RTC with live hardware endpoints requires explicit "
                "--rtc-real-shadow; otherwise use synthetic/local inputs."
            )
        requested_action_hz = float(args.action_hz)
        effective_action_hz = float(
            RTC_3C_VIRTUAL_SHADOW_PROFILE_V1.action_hz
            if requested_model_rtc_action_hz is None
            else requested_model_rtc_action_hz
        )
        if requested_model_rtc_action_hz is None:
            if requested_action_hz not in {
                0.0,
                RTC_3C_VIRTUAL_SHADOW_PROFILE_V1.action_hz,
            }:
                raise ValueError("RTC shadow action rate is fixed at 30 Hz.")
        elif requested_action_hz not in {0.0, effective_action_hz}:
            raise ValueError(
                "--action-hz conflicts with --rtc-action-hz; omit --action-hz "
                "or use the same runtime cadence."
            )
        if args.policy_hz is not None and not math.isclose(
            float(args.policy_hz),
            RTC_3C_VIRTUAL_SHADOW_PROFILE_V1.decision_hz,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("RTC shadow decision rate is fixed at 30 Hz.")
        args.position_chunk_lookahead = False
        args.action_hz = effective_action_hz
        args.policy_hz = RTC_3C_VIRTUAL_SHADOW_PROFILE_V1.decision_hz
    elif (
        int(getattr(args, "rtc_request_validity_steps", 10)) != 10
        or float(getattr(args, "rtc_worker_stop_timeout_sec", 0.0)) != 0.0
        or int(getattr(args, "rtc_active_warmup_discard_completions", 1)) != 1
        or int(getattr(args, "rtc_active_warmup_completions", 3)) != 3
        or not math.isclose(
            float(getattr(args, "rtc_active_max_inference_latency_sec", 0.4)),
            0.4,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(
            "RTC validity/worker-stop overrides and Active latency overrides "
            "require an explicit --rtc-profile."
        )
    # ``--rtc-real-shadow`` selects the live model RTC transport and hardware
    # observation path.  It is intentionally orthogonal to dispatch and
    # initialization: the caller may keep the default no-action observation
    # shadow, or explicitly add ``--send-actions`` and the normal qpos-init
    # inputs.  The latter continues through run_policy's existing execution
    # profile, strict observation firewall, collision/force guards, and ready
    # gates; this helper must not silently remove those authorities.


__all__ = ["validate_rtc_cli_args"]
