from __future__ import annotations

import argparse

import pytest

from teleop_runtime.debug.rm65_real_debug.run_rm65_stop_only_preflight import (
    _build_stop_only_backend,
    _require_explicit_scope,
)
from teleop_runtime.backends.arms.rm65_dispatch_inventory import (
    build_rm65_arms_only_dispatch_inventory,
)
from teleop_runtime.backends.arms.rm65_stop_only_binding import (
    RM65StopOnlyHardwareAuthorization,
    RM65StopOnlyLegacyBinding,
)
from teleop_runtime.backends.arms.rm65_candidate_manifest import (
    build_rm65_policy_arm_qpos_candidate_manifest,
)


def _binding() -> RM65StopOnlyLegacyBinding:
    manifest = build_rm65_policy_arm_qpos_candidate_manifest()
    inventory = build_rm65_arms_only_dispatch_inventory(manifest)
    authorization = RM65StopOnlyHardwareAuthorization.create(
        operator_authorization_id="test-operator",
        execution_manifest=manifest,
        dispatch_inventory=inventory,
        control_domain_id="rm65-arms",
        embodiment_instance_id="rm65-bimanual-0",
        allowed_sides=("left", "right"),
        issued_at=10.0,
        expires_at=20.0,
    )
    return RM65StopOnlyLegacyBinding(
        runtime_session_id="stop-only-preflight-test",
        owner_runtime_id="legacy-test",
        authorization=authorization,
        clock=lambda: 10.0,
    )


def test_stop_only_preflight_requires_all_three_explicit_confirmations() -> None:
    args = argparse.Namespace(
        execute_stop_only=True,
        confirm_arms_stationary=True,
        confirm_estop_ready=False,
        state_interval_sec=0.25,
        max_stationary_drift_rad=0.005,
        authorization_valid_sec=60.0,
    )
    with pytest.raises(RuntimeError, match="confirm_estop_ready"):
        _require_explicit_scope(args)


def test_stop_only_preflight_backend_is_strictly_scoped() -> None:
    binding = _binding()
    backend = _build_stop_only_backend(
        side="left",
        arm_ip="192.0.2.10",
        binding=binding,
    )

    assert backend.config.connect_hardware
    assert not backend.config.configure_singularity_thresholds
    assert not backend.config.force_position_control
    assert not backend.config.background_streaming_enabled
    assert backend._pre_sdk_dispatch_adapter is binding.motion_adapter
    assert backend._pre_sdk_control_adapter is binding.control_adapter
