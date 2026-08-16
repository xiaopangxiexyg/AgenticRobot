from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from robot_motion.deploy.async_producer import (
    MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1,
    RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1,
    RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1,
)
from teleop_runtime.deploy import run_policy
from teleop_runtime.deploy import runtime_timing


_MIGRATED_NAMES = (
    "_action_hz_reference",
    "_runtime_action_keyframe_period_sec",
    "resolve_effective_action_hz",
    "resolve_effective_policy_hz",
)


def test_run_policy_preserves_migrated_runtime_timing_symbols() -> None:
    for name in _MIGRATED_NAMES:
        assert getattr(run_policy, name) is getattr(runtime_timing, name)


@pytest.mark.parametrize(
    ("profile", "expected"),
    (
        ("off", 1.0 / 15.0),
        (RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1, 1.0 / 30.0),
        (RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1, 1.0 / 30.0),
        (MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1, 1.0 / 30.0),
    ),
)
def test_runtime_action_keyframe_period_preserves_profile_binding(
    profile: str,
    expected: float,
) -> None:
    assert runtime_timing._runtime_action_keyframe_period_sec(
        SimpleNamespace(rtc_profile=profile)
    ) == pytest.approx(expected)


def _rate_args(**overrides):
    values = {
        "action_hz": 0.0,
        "hz": 15.0,
        "policy_hz": None,
        "action_hz_dataset": None,
        "action_hz_parquet": None,
        "action_hz_episode": None,
        "action_hz_start_index": None,
        "action_hz_end_index": None,
        "debug_replay_dataset": None,
        "debug_replay_parquet": None,
        "debug_replay_episode": 0,
        "debug_replay_start_index": 0,
        "debug_replay_end_index": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_effective_action_hz_preserves_cli_policy_and_default_precedence() -> None:
    explicit = _rate_args(action_hz=24.0, hz=12.0)
    assert runtime_timing.resolve_effective_action_hz(explicit) == (
        24.0,
        {"source": "cli.action_hz", "hz": 24.0},
    )

    estimate = {"source": "policy.metadata", "hz": 21.0, "samples": 7}
    policy = SimpleNamespace(metadata={"action_hz_estimate": estimate})
    resolved_hz, resolved_info = runtime_timing.resolve_effective_action_hz(
        _rate_args(hz=12.0),
        policy,
    )
    assert resolved_hz == 21.0
    assert resolved_info == estimate
    assert resolved_info is not estimate

    assert runtime_timing.resolve_effective_action_hz(_rate_args(hz=12.0)) == (
        12.0,
        {"source": "cli.hz", "hz": 12.0},
    )


def test_action_hz_reference_preserves_explicit_then_replay_fallbacks() -> None:
    args = _rate_args(
        action_hz_dataset="explicit-dataset",
        action_hz_episode=3,
        action_hz_start_index=4,
        action_hz_end_index=8,
        debug_replay_dataset="replay-dataset",
        debug_replay_parquet="replay.parquet",
        debug_replay_episode=9,
        debug_replay_start_index=10,
        debug_replay_end_index=20,
    )

    assert runtime_timing._action_hz_reference(args) == (
        "explicit-dataset",
        "replay.parquet",
        3,
        4,
        8,
    )


def test_effective_action_hz_preserves_dataset_inference_contract(
    monkeypatch,
    tmp_path,
) -> None:
    dataset = tmp_path / "dataset"
    parquet = tmp_path / "episode_000003.parquet"
    calls = {}

    def _resolve(**kwargs):
        calls["resolve"] = kwargs
        return parquet, {"dataset_schema": "test"}

    def _infer(**kwargs):
        calls["infer"] = kwargs
        return {"source": "timestamps", "hz": 18.0}

    monkeypatch.setattr(runtime_timing, "resolve_replay_parquet_path", _resolve)
    monkeypatch.setattr(runtime_timing, "infer_replay_action_hz", _infer)

    hz, info = runtime_timing.resolve_effective_action_hz(
        _rate_args(
            action_hz_dataset=dataset,
            action_hz_episode=3,
            action_hz_start_index=4,
            action_hz_end_index=8,
        )
    )

    assert hz == 18.0
    assert info == {
        "source": "timestamps",
        "hz": 18.0,
        "source_path": str(parquet),
    }
    assert calls["resolve"] == {
        "dataset_root": dataset,
        "parquet_path": None,
        "episode_index": 3,
    }
    assert calls["infer"] == {
        "parquet_path": parquet,
        "dataset_root": dataset,
        "episode_index": 3,
        "start_index": 4,
        "end_index": 8,
        "resolved_metadata": {"dataset_schema": "test"},
    }


def test_effective_policy_hz_preserves_override_and_action_fallback() -> None:
    assert runtime_timing.resolve_effective_policy_hz(
        _rate_args(policy_hz=9.0),
        action_hz=15.0,
        action_hz_source="cli.hz",
    ) == (9.0, "cli.policy_hz")
    assert runtime_timing.resolve_effective_policy_hz(
        _rate_args(),
        action_hz=15.0,
        action_hz_source="cli.hz",
    ) == (15.0, "action_hz")
