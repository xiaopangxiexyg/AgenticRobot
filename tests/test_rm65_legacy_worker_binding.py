from __future__ import annotations

import argparse
import numpy as np
import pytest

from teleop_runtime.backends.arms.rm65 import RM65ArmBackend, RM65ArmBackendConfig
from teleop_runtime.deploy.rm65_candidate_manifest import (
    build_rm65_policy_arm_qpos_candidate_manifest,
)
from teleop_runtime.deploy.rm65_execution_evidence import (
    build_rm65_execution_evidence_bundle,
)
from teleop_runtime.deploy.rm65_legacy_worker_binding import (
    build_fake_only_rm65_legacy_worker_binding,
)
from teleop_runtime.deploy import run_policy
from teleop_runtime.position_control.deploy_executor import (
    PositionDeployExecutor,
    PositionDeployExecutorConfig,
)
from tests.test_position_deploy_executor import _pipeline


def _binding():
    manifest = build_rm65_policy_arm_qpos_candidate_manifest()
    evidence = build_rm65_execution_evidence_bundle(manifest)
    return build_fake_only_rm65_legacy_worker_binding(
        evidence_bundle=evidence,
        runtime_session_id="rm65-binding-fixture",
    )


def test_one_context_injects_same_guard_into_all_three_required_workers() -> None:
    binding = _binding()
    position_kwargs = binding.position_executor_kwargs()
    left_kwargs = binding.arm_backend_kwargs("left", connect_hardware=False)
    right_kwargs = binding.arm_backend_kwargs("right", connect_hardware=False)

    assert position_kwargs["execution_epoch_guard"] is binding.context.execution_guard
    assert left_kwargs["execution_epoch_guard"] is binding.context.execution_guard
    assert right_kwargs["execution_epoch_guard"] is binding.context.execution_guard
    participant_ids = {
        position_kwargs["worker_drain_participant"].participant_id,
        left_kwargs["worker_drain_participant"].participant_id,
        right_kwargs["worker_drain_participant"].participant_id,
    }
    assert participant_ids == {
        "position-deploy-reference-worker",
        "rm65-left-background-stream-worker",
        "rm65-right-background-stream-worker",
    }
    assert not binding.production_binding_enabled
    assert not binding.confers_execution_authorization


def test_fake_binding_constructs_real_worker_classes_without_hardware() -> None:
    binding = _binding()
    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(
            sample_hz=15.0,
            reference_hz=60.0,
            command_hz=120.0,
        ),
        canonical_pipeline=_pipeline(("left", "right")),
        **binding.position_executor_kwargs(),
    )
    left = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="left",
            connect_hardware=False,
            initial_qpos=np.zeros(6),
            background_streaming_enabled=True,
        ),
        **binding.arm_backend_kwargs("left", connect_hardware=False),
    )
    right = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="right",
            connect_hardware=False,
            initial_qpos=np.zeros(6),
            background_streaming_enabled=True,
        ),
        **binding.arm_backend_kwargs("right", connect_hardware=False),
    )

    assert executor.snapshot()["execution_epoch_binding_enabled"] is True
    assert left.get_state()["background_streaming"]["execution_epoch_binding_enabled"]
    assert right.get_state()["background_streaming"]["execution_epoch_binding_enabled"]


def test_fake_binding_refuses_hardware_and_unknown_participant_scope() -> None:
    binding = _binding()

    with pytest.raises(RuntimeError, match="cannot be attached to hardware"):
        binding.arm_backend_kwargs("left", connect_hardware=True)
    with pytest.raises(ValueError, match="left or right"):
        binding.arm_backend_kwargs("center", connect_hardware=False)


def test_canonical_policy_backend_builder_has_default_off_fake_binding_seam() -> None:
    binding = _binding()
    parser = argparse.ArgumentParser()
    run_policy.add_args(parser)
    args = parser.parse_args(["--active-sides", "both"])
    active_sides = run_policy.validate_args(args)
    args._rm65_legacy_worker_binding = binding

    backends = run_policy.build_arm_backends(
        args,
        active_sides,
        control_hz=args.arm_control_hz,
    )

    assert tuple(backends) == ("left", "right")
    assert backends["left"]._execution_epoch_guard is binding.context.execution_guard
    assert backends["right"]._execution_epoch_guard is binding.context.execution_guard

    args.left_arm_ip = "192.0.2.1"
    args.right_arm_ip = "192.0.2.2"
    with pytest.raises(RuntimeError, match="cannot be attached to hardware"):
        run_policy.build_arm_backends(
            args,
            active_sides,
            control_hz=args.arm_control_hz,
        )
