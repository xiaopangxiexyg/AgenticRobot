"""Three application adapters share conversion, not defaults or device authority."""

from __future__ import annotations

import argparse
from dataclasses import asdict

import pytest

from teleop_runtime.backends.grippers import (
    DMTAC_GRIPPER_TELEMETRY_COMMAND_GRACE_S,
    DMTacGripperBackendConfig,
)
from teleop_runtime.collection.run_collect import (
    _apply_gripper_policy_to_dmtac_config as apply_collection,
)
from teleop_runtime.deploy.hand_backend_config import (
    _apply_gripper_policy_to_dmtac_config as apply_deploy,
)
from teleop_runtime.real_teleop.dmtac_config import (
    apply_gripper_policy_to_dmtac_config as apply_teleop,
)


@pytest.fixture(
    params=[apply_collection, apply_teleop, apply_deploy],
    ids=["collection", "teleop", "deploy"],
)
def apply_config(request):
    return request.param


@pytest.fixture
def policy():
    return {
        "speed": 0.31,
        "torque": 0.42,
        "max_action": 0.83,
        "output_dim": 0,
        "send_threshold": 0.013,
        "command_hz": 17.0,
        "continuous_grasp": False,
        "telemetry_hz": 11.0,
    }


def test_policy_changes_only_owned_fields_and_retains_assignment(apply_config, policy):
    config = DMTacGripperBackendConfig(
        side="right",
        gripper_port="/dev/example",
        connect_hardware=False,
        connect_tactile=True,
        tactile_ids=("sensor",),
        startup_recovery_mode="legacy",
        tactile_read_hz=4.0,
        enable_telemetry=False,
    )
    expected = asdict(config)
    expected.update(
        {key: value for key, value in policy.items() if key != "output_dim"}
    )
    expected.update(
        command_dim=0,
        gripper_telemetry_command_grace_s=DMTAC_GRIPPER_TELEMETRY_COMMAND_GRACE_S,
        enable_telemetry=True,
    )
    if apply_config is apply_teleop:
        expected.update(
            startup_recovery_mode="stabilized",
            connect_tactile=False,
            tactile_ids=(),
            enable_telemetry=False,
        )
    args = argparse.Namespace(no_collection=True)
    original_policy = dict(policy)

    assert apply_config(config, args=args, policy=policy) is config
    assert asdict(config) == expected
    assert vars(args) == {"no_collection": True}
    assert policy == original_policy


def test_none_options_preserve_existing_capture_values(apply_config, policy):
    config = DMTacGripperBackendConfig(
        side="left",
        tactile_read_hz=8.0,
        tactile_timeout_ms=123,
        tactile_process_cpu_affinity=(2,),
        tactile_warmup_frames=4,
        tactile_compute_profile="requested",
        tactile_fields=("force",),
    )
    expected = asdict(
        apply_config(
            DMTacGripperBackendConfig(**asdict(config)),
            args=argparse.Namespace(no_collection=True),
            policy=policy,
        )
    )
    fields = (
        "timeout_ms",
        "wait_for_new",
        "read_hz",
        "process_read",
        "worker_cpus",
        "worker_nice",
        "worker_queue_size",
        "fields",
        "compute_profile",
        "calibrate",
        "max_fps",
        "warmup_frames",
        "denoise",
        "flat_vector",
        "parallel_read",
        "global_read_lock",
    )
    args = argparse.Namespace(
        no_collection=True, **{f"dmtac_tactile_{field}": None for field in fields}
    )

    assert asdict(apply_config(config, args=args, policy=policy)) == expected


def test_only_teleop_consumes_recovery_and_calibration_overrides(
    apply_config,
    policy,
    tmp_path,
):
    config = DMTacGripperBackendConfig(side="left")
    original = asdict(config)
    args = argparse.Namespace(
        dmtac_startup_recovery_mode="legacy",
        dmtac_tactile_calibration_policy="load-required",
        dmtac_tactile_calibration_dir=str(tmp_path),
    )
    apply_config(config, args=args, policy=policy)

    overrides = {
        "startup_recovery_mode": "legacy",
        "tactile_calibration_policy": "load-required",
        "tactile_calibration_dir": tmp_path,
    }
    for field, value in overrides.items():
        assert getattr(config, field) == (
            value if apply_config is apply_teleop else original[field]
        )


@pytest.mark.parametrize(
    "argument, value, field, expected, capture_only",
    [
        ("timeout_ms", "125", "tactile_timeout_ms", 125, False),
        ("wait_for_new", False, "tactile_wait_for_new", False, False),
        ("read_hz", "7.5", "tactile_read_hz", 7.5, True),
        ("process_read", False, "tactile_process_read", False, True),
        ("worker_cpus", ["2", "3"], "tactile_process_cpu_affinity", (2, 3), True),
        ("worker_cpus", [], "tactile_process_cpu_affinity", (), True),
        ("worker_nice", "0", "tactile_process_nice", 0, True),
        ("worker_queue_size", "3", "tactile_process_queue_size", 3, True),
        ("calibrate", False, "tactile_reset_on_start", False, True),
        ("max_fps", "21", "tactile_max_fps", 21, True),
        ("warmup_frames", 0, "tactile_warmup_frames", 0, True),
        ("denoise", False, "tactile_denoise", False, True),
        ("flat_vector", False, "tactile_store_flat_vector", False, True),
        ("parallel_read", True, "tactile_parallel_read", True, True),
        ("global_read_lock", False, "tactile_global_read_lock", False, True),
    ],
)
def test_explicit_options_preserve_conversion_and_application_scope(
    apply_config,
    policy,
    argument,
    value,
    field,
    expected,
    capture_only,
):
    initial = DMTacGripperBackendConfig(
        side="left",
        tactile_wait_for_new=True,
        tactile_process_read=True,
        tactile_reset_on_start=True,
        tactile_warmup_frames=5,
        tactile_denoise=True,
        tactile_store_flat_vector=True,
    )
    before = asdict(initial)
    args = argparse.Namespace(
        no_collection=True, dmtac_tactile=True, **{f"dmtac_tactile_{argument}": value}
    )
    result = apply_config(initial, args=args, policy=policy)

    assert getattr(result, field) == (
        before[field] if capture_only and apply_config is apply_deploy else expected
    )


@pytest.mark.parametrize(
    "cli_fields, profile_fields, expected",
    [
        (None, None, ("raw",)),
        (None, ("force",), ("force",)),
        (("raw",), ("force",), ("raw",)),
    ],
)
@pytest.mark.parametrize(
    "cli_profile, profile_value, expected_profile",
    [
        (None, None, "legacy"),
        (None, "requested", "requested"),
        ("", "requested", "requested"),
        ("dist-only", "requested", "dist-only"),
    ],
)
def test_tactile_cli_profile_assignment_precedence(
    apply_config,
    policy,
    cli_fields,
    profile_fields,
    expected,
    cli_profile,
    profile_value,
    expected_profile,
):
    config = DMTacGripperBackendConfig(
        side="left", tactile_fields=("raw",), tactile_compute_profile="legacy"
    )
    policy.update(tactile_fields=profile_fields, tactile_compute_profile=profile_value)
    args = argparse.Namespace(
        no_collection=True,
        dmtac_tactile_fields=cli_fields,
        dmtac_tactile_compute_profile=cli_profile,
    )

    apply_config(config, args=args, policy=policy)

    assert config.tactile_fields == expected
    assert config.tactile_compute_profile == expected_profile


@pytest.mark.parametrize("hardware", [False, True])
@pytest.mark.parametrize("tactile", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_tactile_opt_in_never_reconnects_disabled_assignment(
    apply_config,
    policy,
    hardware,
    tactile,
    enabled,
):
    config = DMTacGripperBackendConfig(
        side="left",
        connect_hardware=hardware,
        connect_tactile=tactile,
        tactile_ids=("sensor",),
    )
    apply_config(config, args=argparse.Namespace(dmtac_tactile=enabled), policy=policy)

    assert config.connect_hardware is hardware
    assert config.connect_tactile is (tactile and enabled)
    assert config.tactile_ids == (("sensor",) if enabled else ())
    assert config.enable_telemetry is (hardware or (tactile and enabled))


@pytest.mark.parametrize(
    "parallel, sequential", [(None, True), (True, True), (False, False)]
)
def test_explicit_parallel_read_wins_only_for_capture_adapters(
    apply_config,
    policy,
    parallel,
    sequential,
):
    config = DMTacGripperBackendConfig(side="left", tactile_parallel_read=True)
    args = argparse.Namespace(
        dmtac_tactile_parallel_read=parallel, dmtac_tactile_sequential_read=sequential
    )
    apply_config(config, args=args, policy=policy)

    assert config.tactile_parallel_read is (
        not sequential if apply_config is apply_deploy or parallel is None else parallel
    )


@pytest.mark.parametrize(
    "no_collection, hz, read_hz, expected",
    [
        (False, 15.0, None, 15.0),
        (False, 90.0, None, 30.0),
        (False, 0.1, None, 1.0),
        (True, 15.0, None, 4.0),
        (False, 15.0, 7.0, 7.0),
    ],
)
def test_only_teleop_defaults_read_rate_to_collection_cadence(
    apply_config,
    policy,
    no_collection,
    hz,
    read_hz,
    expected,
):
    config = DMTacGripperBackendConfig(
        side="left", tactile_read_hz=4.0, tactile_max_fps=30
    )
    args = argparse.Namespace(
        no_collection=no_collection,
        hz=hz,
        dmtac_tactile_read_hz=read_hz,
        dmtac_tactile_max_fps=10,
    )
    apply_config(config, args=args, policy=policy)

    if apply_config is apply_teleop:
        # The established default uses the assignment's cap before the CLI override.
        assert config.tactile_read_hz == expected
    else:
        assert config.tactile_read_hz == (
            read_hz if apply_config is apply_collection and read_hz is not None else 4.0
        )
    assert config.tactile_max_fps == (30 if apply_config is apply_deploy else 10)


@pytest.mark.parametrize(
    "field, value, error",
    [
        ("timeout_ms", "invalid", ValueError),
        ("fields", ("unknown-field",), ValueError),
        ("worker_cpus", ("invalid",), ValueError),
    ],
)
def test_invalid_supported_options_are_not_silently_ignored(
    apply_config,
    policy,
    field,
    value,
    error,
):
    config = DMTacGripperBackendConfig(side="left")
    args = argparse.Namespace(**{f"dmtac_tactile_{field}": value})
    if apply_config is apply_deploy and field == "worker_cpus":
        apply_config(config, args=args, policy=policy)
        return
    with pytest.raises(error):
        apply_config(config, args=args, policy=policy)
