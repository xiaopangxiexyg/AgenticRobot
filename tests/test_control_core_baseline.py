from __future__ import annotations

import ast
import argparse
import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from dataset_contracts.causal_snapshot import DEFAULT_CAUSAL_SNAPSHOT_CONTRACT
from robot_motion.deploy.actions import (
    ActionSafetyFilter,
    ActionSafetyFilterConfig,
    NormalizedAction,
)
from robot_motion.deploy.canonical_action import CanonicalDeployActionPipeline
from robot_motion.deploy.dispatcher import DryRunCommandDispatcher
from robot_motion.deploy.observation import build_deployment_observation
from robot_motion.deploy.pose_adapter import convert_normalized_action_chunk_to_relative
from robot_motion.deploy.runtime import PolicyDeployRuntime, PolicyDeployRuntimeConfig
from robot_motion.deploy.schema import (
    ACTION_ARM_POSE_SLICES,
    ACTION_DIM,
    DEPLOY_ACTION_VECTOR_ACTION_SPACE,
    DEPLOY_ACTION_VECTOR_LAYOUT_VERSION,
    STATE_ARM_POSE_SLICES,
    nan_vector,
)
from teleop_runtime.collection.sample_assembler import SampleAssembler
from teleop_runtime.deploy import run_policy
from teleop_runtime.position_control import (
    PositionDeployExecutor,
    PositionDeployExecutorConfig,
)
from teleop_runtime.session import TeleopSession


BASELINE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "control_core"
    / "v23_single_policy_baseline_v1.json"
)
BASELINE_SHA256_PATH = BASELINE_PATH.with_suffix(".sha256")


class _FixedChunkPolicy:
    name = "control_core_baseline_policy"

    def __init__(self, output: np.ndarray) -> None:
        self._output = np.asarray(output, dtype=np.float32).copy()

    def predict(self, observation: Any) -> np.ndarray:
        del observation
        return self._output.copy()


def _json_vector(value: Any) -> list[float | None]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return [float(item) if np.isfinite(item) else None for item in array]


def _baseline_observation():
    pose_matrix = np.eye(4, dtype=np.float64)
    pose_matrix[:3, 3] = [0.10, 0.20, 0.30]
    return build_deployment_observation(
        active_sides=("right",),
        arms_enabled=True,
        hands_enabled=False,
        step_index=7,
        arm_states={
            "right": {
                "current_qpos": np.zeros(6, dtype=np.float32),
                "current_tcp_pose": pose_matrix,
                "actual_tcp_pose": [0.10, 0.20, 0.30, 0.0, 0.0, 0.0, 1.0],
            }
        },
        hand_states={},
        last_action=nan_vector(ACTION_DIM),
        timestamp=12.5,
        metadata={
            "baseline_fixture": "v23_single_policy_baseline_v1",
            "selected_owner": "must_not_become_model_input_in_m1a",
            "reward": 123.0,
            "dispatch_receipt": {"accepted": True},
            "writer_health": {"status": "must_remain_control_private"},
            "backend_health": {"status": "must_remain_control_private"},
        },
    )


def _legacy_private_model_input(observation: Any) -> dict[str, Any]:
    injected = replace(
        observation,
        faults=["injected_control_fault_must_not_enter_strict_view"],
    )
    return injected.model_input()


def _deploy_schedule_profile(argv: list[str]) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    run_policy.add_args(parser)
    args = parser.parse_args(argv)
    active_sides = run_policy.validate_args(args, raw_argv=argv)
    action_hz, rate_info = run_policy.resolve_effective_action_hz(args, policy=None)
    return {
        "active_sides": list(active_sides),
        "arm_control_mode": str(args.arm_control_mode),
        "action_hz": float(action_hz),
        "action_hz_source": str(rate_info["source"]),
        "position_reference_hz": float(args.position_reference_hz),
        "arm_control_hz": float(args.arm_control_hz),
        "arm_canfd_follow": bool(args.arm_canfd_follow),
        "position_chunk_lookahead": bool(args.position_chunk_lookahead),
        "position_executor_enabled": str(args.arm_control_mode) == "position",
    }


def _ordered_source_markers(source: str, markers: tuple[tuple[str, str], ...]) -> list[str]:
    indexed: list[tuple[int, str]] = []
    for name, token in markers:
        index = source.find(token)
        if index < 0:
            raise AssertionError(f"M0 marker {name!r} disappeared: {token!r}")
        indexed.append((index, name))
    return [name for _, name in sorted(indexed)]


def _runtime_semantics_baseline() -> dict[str, Any]:
    session_source = inspect.getsource(TeleopSession.run_step)
    collection_source = inspect.getsource(SampleAssembler.build_sample)
    deploy_source = inspect.getsource(run_policy.main)
    dispatch_source = inspect.getsource(TeleopSession.dispatch_command)
    return {
        "teleop_session_order": _ordered_source_markers(
            session_source,
            (
                ("channel_fault_freeze", "if freeze_faults"),
                ("command_filter", "self.command_filter(command"),
                ("command_guard", "self.evaluate_command_guard("),
                ("record_command", "self.recorder.record_command(command)"),
                ("dispatch_command", "dispatch_receipts = self.dispatch_command("),
            ),
        ),
        "dispatch_seam": (
            "TeleopSession.dispatch_command->Rig.send_commands"
            if "self.rig.send_commands(" in dispatch_source
            else "missing"
        ),
        "collection_last_action_stage": (
            "previous_session_command_derived_action"
            if "observation_last_action=self._last_action.copy()" in collection_source
            and "self._last_action = action.copy()" in collection_source
            and "step_result.command" in collection_source
            else "missing"
        ),
        "deploy_last_action_stage": (
            "accepted_canonical_safe_action"
            if (
                "last_action = canonical_action.action.vector.astype" in deploy_source
                or "last_action = _prepared_action_last_action(" in deploy_source
            )
            and "if execution.accepted:" in deploy_source
            else "missing"
        ),
    }


def _absolute_chunk() -> list[NormalizedAction]:
    targets = (
        [0.12, 0.21, 0.31, 0.0, 0.0, 0.0, 1.0],
        [0.13, 0.22, 0.32, 0.0, 0.0, 0.0, 1.0],
    )
    actions: list[NormalizedAction] = []
    for target in targets:
        vector = nan_vector(ACTION_DIM)
        vector[ACTION_ARM_POSE_SLICES["right"]] = np.asarray(
            target,
            dtype=np.float32,
        )
        actions.append(NormalizedAction(vector=vector))
    return actions


def build_control_core_baseline() -> dict[str, Any]:
    observation = _baseline_observation()
    relative_actions = convert_normalized_action_chunk_to_relative(
        _absolute_chunk(),
        observation,
        active_sides=("right",),
    )
    policy = _FixedChunkPolicy(
        np.stack([action.vector for action in relative_actions], axis=0)
    )
    runtime = PolicyDeployRuntime(
        PolicyDeployRuntimeConfig(
            active_sides=("right",),
            arms_enabled=True,
            hands_enabled=False,
            policy_action_pose_mode="relative",
        )
    )
    prediction = runtime.predict_and_queue(
        policy,
        observation,
        now=20.0,
        queue_policy="replace",
    )
    pipeline = CanonicalDeployActionPipeline(
        safety_filter=ActionSafetyFilter(
            ActionSafetyFilterConfig(
                active_sides=("right",),
                arms_enabled=True,
                hands_enabled=False,
                workspace_min=(-2.0, -2.0, -2.0),
                workspace_max=(2.0, 2.0, 2.0),
                workspace_soft_margin_m=0.0,
                max_arm_xyz_step_m=0.0,
                max_arm_rot_step_deg=0.0,
                max_arm_joint_step_rad=0.0,
            )
        ),
        active_sides=("right",),
        arms_enabled=True,
        hands_enabled=False,
    )
    action_step = runtime.prepare_next_action(
        observation,
        preparer=pipeline,
        policy_name=policy.name,
    )
    prepared = action_step.prepared_action
    assert prepared is not None
    assert prepared.robot_action is not None
    assert prepared.command is not None
    dispatcher = DryRunCommandDispatcher()
    dispatcher.dispatch(prepared.command)

    position_executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(
            sample_hz=15.0,
            reference_hz=60.0,
            command_hz=120.0,
            debug_events_enabled=True,
        ),
        canonical_pipeline=pipeline,
    )
    try:
        position_source_index = position_executor.submit(
            prepared.filtered,
            observation,
            policy.name,
            timestamp_monotonic=20.0,
        )
        position_snapshot = position_executor.snapshot()
        position_events = position_executor.drain_debug_events()
    finally:
        position_executor.close()

    decoded = list(prediction.actions)
    raw_action = action_step.raw_action
    raw_lookahead = action_step.raw_lookahead_action
    assert raw_action is not None
    assert raw_lookahead is not None
    right_command = prepared.command.commands["right_arm"]
    model_input = observation.model_input()
    private_model_input = _legacy_private_model_input(observation)

    return {
        "schema": "dexteleop_control_core_baseline_v1",
        "fixture_id": "v23_single_policy_relative_right_arm_v1",
        "v23_contracts": {
            "causal_snapshot_schema": DEFAULT_CAUSAL_SNAPSHOT_CONTRACT.schema,
            "causal_snapshot_hash": DEFAULT_CAUSAL_SNAPSHOT_CONTRACT.contract_hash,
            "action_layout_version": DEPLOY_ACTION_VECTOR_LAYOUT_VERSION,
            "action_space": DEPLOY_ACTION_VECTOR_ACTION_SPACE,
        },
        "observation": {
            "timestamp": float(observation.timestamp),
            "step_index": int(observation.step_index),
            "active_sides": list(observation.active_sides),
            "faults": list(observation.faults),
            "right_arm_pose": _json_vector(
                observation.state[STATE_ARM_POSE_SLICES["right"]]
            ),
            "last_action": _json_vector(observation.last_action),
            "legacy_model_input_keys": sorted(model_input),
            "legacy_private_metadata_keys": sorted(
                key
                for key in observation.metadata
                if key
                in {
                    "selected_owner",
                    "reward",
                    "dispatch_receipt",
                    "writer_health",
                    "backend_health",
                }
            ),
            "legacy_private_faults": list(private_model_input["faults"]),
        },
        "policy": {
            "pose_mode": runtime.config.policy_action_pose_mode,
            "prediction_accepted": bool(prediction.accepted),
            "queue_len_before": int(prediction.queue_len_before),
            "queue_len_after": int(prediction.queue_len_after),
            "relative_actions": [
                _json_vector(action.vector) for action in relative_actions
            ],
            "decoded_absolute_actions": [
                _json_vector(action.vector) for action in decoded
            ],
            "chunk_ids": [
                int(action.metadata["deploy_action_chunk_id"]) for action in decoded
            ],
            "chunk_indices": [
                int(action.metadata["deploy_action_chunk_index"])
                for action in decoded
            ],
            "chunk_sizes": [
                int(action.metadata["deploy_action_chunk_size"]) for action in decoded
            ],
            "pose_anchor_policy": str(
                decoded[0].metadata["pose_anchor_policy"]
            ),
            "pose_anchor_sides": list(decoded[0].metadata["pose_anchor_sides"]),
        },
        "canonical": {
            "faults": list(prepared.faults),
            "filtered_action": _json_vector(prepared.action.vector),
            "robot_action_space": prepared.robot_action.action_space,
            "robot_action_source": prepared.robot_action.source,
            "active_effectors": list(prepared.robot_action.active_effectors),
            "right_arm_pose": _json_vector(
                prepared.robot_action.arm_pose_by_side["right"]
            ),
            "command_effectors": sorted(prepared.command.commands),
            "right_command_type": right_command.command_type,
            "right_target_xyzquat": _json_vector(
                right_command.metadata["target_xyzquat"]
            ),
            "command_source": str(prepared.command.metadata["source"]),
            "command_policy_name": str(prepared.command.metadata["policy_name"]),
        },
        "dispatch": {
            "mode": "dry_run",
            "intent_count": len(dispatcher.commands),
            "effectors": sorted(dispatcher.commands[0].commands),
        },
        "position_keyframe": {
            "source_index": int(position_source_index),
            "dry_run": bool(position_snapshot["dry_run"]),
            "submitted_count": int(position_snapshot["submitted_count"]),
            "queue_depth": int(position_snapshot["queue_depth"]),
            "sample_hz": float(position_snapshot["sample_hz"]),
            "reference_hz": float(position_snapshot["reference_hz"]),
            "command_hz": float(position_snapshot["command_hz"]),
            "event_names": [str(event["event"]) for event in position_events],
        },
        "deploy_profiles": {
            "position_default": _deploy_schedule_profile(
                ["--active-sides", "right"]
            ),
            "non_position_impedance": _deploy_schedule_profile(
                [
                    "--active-sides",
                    "right",
                    "--arm-execution-profile",
                    "impedance_40hz",
                ]
            ),
        },
        "runtime_semantics": _runtime_semantics_baseline(),
        "last_action_contract": {
            "initial_stage": "unavailable_nan_vector",
            "deploy_update_stage": "accepted_canonical_safe_action",
        },
    }


def test_v23_single_policy_control_core_baseline_is_unchanged() -> None:
    expected = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    assert build_control_core_baseline() == expected


def test_v23_single_policy_control_core_baseline_bytes_are_sealed() -> None:
    expected = BASELINE_SHA256_PATH.read_text(encoding="ascii").strip()
    actual = hashlib.sha256(BASELINE_PATH.read_bytes()).hexdigest()
    assert actual == expected


def test_control_core_baseline_has_no_automatic_update_path() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    write_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {"write", "write_text", "dump"}
        )
    ]
    assert write_calls == []


def test_m0_baseline_covers_required_control_boundaries() -> None:
    baseline = build_control_core_baseline()

    assert baseline["position_keyframe"]["event_names"] == ["keyframe_submitted"]
    assert baseline["position_keyframe"]["queue_depth"] == 1
    assert baseline["deploy_profiles"]["position_default"]["position_executor_enabled"]
    assert not baseline["deploy_profiles"]["non_position_impedance"][
        "position_executor_enabled"
    ]
    assert baseline["runtime_semantics"]["teleop_session_order"] == [
        "channel_fault_freeze",
        "command_filter",
        "command_guard",
        "record_command",
        "dispatch_command",
    ]
    assert baseline["runtime_semantics"]["dispatch_seam"] == (
        "TeleopSession.dispatch_command->Rig.send_commands"
    )
    assert baseline["runtime_semantics"]["collection_last_action_stage"] == (
        "previous_session_command_derived_action"
    )
    assert baseline["runtime_semantics"]["deploy_last_action_stage"] == (
        "accepted_canonical_safe_action"
    )
    assert baseline["observation"]["legacy_private_faults"] == [
        "injected_control_fault_must_not_enter_strict_view"
    ]
