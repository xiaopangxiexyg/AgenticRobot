"""Deployment action and policy timing resolution.

This module resolves configured and observed rates.  It owns no loop clock,
queue advancement, policy worker, or backend execution lifecycle.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from robot_motion.deploy.async_producer import (
    MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1,
    model_rtc_flow_inpaint_shadow_profile_for_action_hz,
    RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1,
    RAW_RTC_FINAL_CHUNK_PROFILE_V1,
    RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1,
    RTC_3C_VIRTUAL_SHADOW_PROFILE_V1,
)

from .replay import infer_replay_action_hz, resolve_replay_parquet_path


def _runtime_action_keyframe_period_sec(args: argparse.Namespace) -> float:
    """Bind RTC runtime timing without changing the legacy sync default."""

    profile_id = str(getattr(args, "rtc_profile", "off")).strip()
    if profile_id == RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1:
        return 1.0 / RTC_3C_VIRTUAL_SHADOW_PROFILE_V1.action_hz
    if profile_id == RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1:
        return 1.0 / RAW_RTC_FINAL_CHUNK_PROFILE_V1.action_hz
    if profile_id == MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1:
        profile = model_rtc_flow_inpaint_shadow_profile_for_action_hz(
            getattr(args, "rtc_action_hz", None)
        )
        return 1.0 / profile.action_hz
    # Preserve the historical PolicyDeployRuntimeConfig default for sync_queue_v1.
    return 1.0 / 15.0


def _action_hz_reference(
    args: argparse.Namespace,
) -> tuple[Path | None, Path | None, int, int, int | None]:
    dataset = args.action_hz_dataset or args.debug_replay_dataset
    parquet = args.action_hz_parquet or args.debug_replay_parquet
    episode = (
        int(args.action_hz_episode)
        if args.action_hz_episode is not None
        else int(args.debug_replay_episode)
    )
    start_index = (
        int(args.action_hz_start_index)
        if args.action_hz_start_index is not None
        else int(args.debug_replay_start_index)
    )
    end_index = (
        int(args.action_hz_end_index)
        if args.action_hz_end_index is not None
        else args.debug_replay_end_index
    )
    return dataset, parquet, episode, start_index, end_index


def resolve_effective_action_hz(
    args: argparse.Namespace, policy: Any | None = None
) -> tuple[float, dict[str, Any]]:
    explicit = float(args.action_hz)
    if explicit > 0.0:
        return explicit, {"source": "cli.action_hz", "hz": explicit}

    if policy is not None:
        estimate = getattr(policy, "metadata", {}).get("action_hz_estimate")
        if isinstance(estimate, Mapping):
            hz = float(estimate.get("hz", 0.0) or 0.0)
            if hz > 0.0:
                return hz, dict(estimate)

    dataset, parquet, episode, start_index, end_index = _action_hz_reference(args)
    if dataset or parquet:
        path, metadata = resolve_replay_parquet_path(
            dataset_root=dataset,
            parquet_path=parquet,
            episode_index=int(episode),
        )
        estimate = infer_replay_action_hz(
            parquet_path=path,
            dataset_root=dataset,
            episode_index=int(episode),
            start_index=int(start_index),
            end_index=end_index,
            resolved_metadata=metadata,
        )
        hz = float(estimate.get("hz", 0.0) or 0.0)
        if hz > 0.0:
            estimate = dict(estimate)
            estimate["source_path"] = str(path)
            return hz, estimate

    default_hz = float(args.hz)
    return default_hz, {"source": "cli.hz", "hz": default_hz}


def resolve_effective_policy_hz(
    args: argparse.Namespace, *, action_hz: float, action_hz_source: str
) -> tuple[float, str]:
    if args.policy_hz is not None:
        return float(args.policy_hz), "cli.policy_hz"
    return float(action_hz), "action_hz"


__all__ = [
    "_action_hz_reference",
    "_runtime_action_keyframe_period_sec",
    "resolve_effective_action_hz",
    "resolve_effective_policy_hz",
]
