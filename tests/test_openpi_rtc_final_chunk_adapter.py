from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import threading
import time
from typing import Any
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from dataset_contracts.pose_chunk import PoseActionChunkContract
from dataset_contracts.pose_training import PoseTrainingDataContract
from dataset_contracts.model_continuation import (
    MODEL_CONTINUATION_FLOW_INPAINT_V1,
    MODEL_CONTINUATION_ORIGIN_INFERENCE_ONLY_V1,
    MODEL_CONTINUATION_SAMPLER_CLAMPED_V1,
    RUNTIME_PREFIX_PRODUCER_LOCAL_OVERLAP_V1,
    TRAINING_PREFIX_TRAJECTORY_OVERLAP_TEACHER_V1,
    ModelContinuationDataContract,
    ModelContinuationDataContractV2,
    ModelContinuationRuntimeContractV2,
)
from dataset_contracts.policy_artifact import (
    PolicyArtifactContract,
    PolicyCameraInputContract,
)
from dataset_contracts import (
    DEFAULT_ARM_HOLD_TRAINING_CONTRACT_V1,
    PolicyArmHoldOutputContract,
)
from dataset_contracts.v22 import (
    DiscreteCategory,
    DiscreteTokenSpec,
    EndEffectorSpec,
    PolicyIOLayout,
)
from deployments.client import (
    RemoteModelPolicy,
    RemotePolicyConfig,
    RtcConditionedModelRuntimeAdapterV3,
    RtcModelRuntimeAdapter,
    build_rtc_model_shadow_frontend,
    rtc_client_inference_bundle_debug_payload,
    rtc_conditioned_config_from_model_adapter_v3,
    rtc_shadow_config_from_model_adapter,
    validate_server_metadata,
)
from deployments.client.rtc_conditioning import (
    rtc_binary_mask_hash,
    rtc_float32_array_hash,
)
from deployments.client.artifact import deploy_contract_from_policy_artifact
from deployments.openpi.client.policy import (
    OPENPI_LOCAL_PROTOCOL_VERSION,
    OpenPIClientPolicy,
    OpenPIClientPolicyConfig,
)
from deployments.openpi.client import real as openpi_real_launcher
from deployments.openpi.client.rtc import (
    OPENPI_FINAL_CHUNK_RTC_ADAPTER_V1,
    OPENPI_RTC_RESULT_TYPED_CANONICAL_V1,
    OpenPIFinalChunkRtcBackend,
    OpenPIFinalChunkRtcConfig,
    openpi_relative_conversion_contract_hash,
    openpi_v22_legacy_codec_hash,
    openpi_v22_native_codec_hash,
    openpi_v22_relative_native_codec_hash,
    openpi_rtc_artifact_hash,
)
from deployments.openpi.client.rtc_conditioned import (
    OPENPI_CONDITIONED_RTC_METADATA_KEY_V1,
    OPENPI_CONDITIONED_RTC_OPERATION_V1,
    OPENPI_CONDITIONED_PROVIDER_CAPABILITY_SCHEMA_V1,
    OPENPI_CONDITIONED_RTC_WIRE_V1,
    OpenPIConditionedRtcBackendV3,
    OpenPIConditionedRtcConfig,
    openpi_conditioned_provider_capability_hash,
)
from robot_motion.deploy import (
    ActionSafetyFilter,
    ActionSafetyFilterConfig,
    ArmHoldResolvedHistoryBinding,
    CanonicalDeployActionPipeline,
    MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1,
    POLICY_ASYNC_PRODUCER_FAULT,
    PolicyDeployRuntime,
    PolicyDeployRuntimeConfig,
    RAW_RTC_FINAL_CHUNK_PROFILE_V1,
)
from robot_motion.deploy.inference import (
    ASYNC_ALIGNED_CHUNK_MODE_V1,
    RTC_FLOW_INPAINT_MODE_V1,
    PolicyInferenceRequest,
)
from robot_motion.deploy.action_representation import (
    ACTION_ANCHOR_POLICY_NONE_V1,
    PolicyActionRepresentationContext,
)
from robot_motion.deploy.chunk_plan import (
    POLICY_PLAN_KEYFRAME_MUTABLE,
    TimedPolicyActionKeyframe,
    TimedPolicyActionPlanSnapshot,
)
from robot_motion.deploy.rtc import (
    RTC_PREFIX_SCHEDULE_EXPLICIT_SELECTED_STEPS_V1,
    RtcContinuationSelectionSpec,
    RtcContinuationTimeline,
    build_canonical_rtc_conditioning_bundle,
    select_rtc_plan_continuation,
)
from robot_motion.deploy.observation import DeploymentObservation
from robot_motion.deploy.policy_worker import (
    POLICY_EXECUTOR_CANCEL_REQUESTED,
    BoundedPolicyInferenceWorker,
    PolicyInferenceWorkItem,
)
from robot_motion.deploy.runtime import PolicyDeployRuntime, PolicyDeployRuntimeConfig
from robot_motion.deploy.rtc_protocol import (
    RTC_CONTINUATION_FLOW_INPAINT,
    RTC_CONTINUATION_NONE,
    RTC_SCHEDULER_ASYNC_TIMED_PLAN,
    RtcProtocolMode,
    negotiate_rtc_capability,
)
from robot_motion.deploy.schema import (
    ACTION_ARM_POSE_SLICES,
    ACTION_DIM,
    ACTION_HAND_SLICES,
    OBSERVATION_STATE_DIM,
    STATE_ARM_POSE_SLICES,
)


HORIZON = 4
RELATIVE_FRAME_HASH = "sha256:" + "1" * 64


class _CounterClock:
    def __init__(self, start: float = 10.01, step: float = 0.01) -> None:
        self.value = float(start) - float(step)
        self.step = float(step)

    def __call__(self) -> float:
        self.value += self.step
        return self.value


class _SequenceClock:
    def __init__(self, values: tuple[float, ...]) -> None:
        self.values = list(values)

    def __call__(self) -> float:
        return self.values.pop(0)


class _FakeTransport:
    def __init__(
        self,
        *,
        metadata: dict[str, Any],
        actions: np.ndarray,
        discrete_actions: np.ndarray | None = None,
        arm_hold_probabilities: np.ndarray | None = None,
        arm_hold_contract_hash: str | None = None,
        error: Exception | None = None,
    ) -> None:
        self._metadata = dict(metadata)
        self.actions = np.asarray(actions)
        self.discrete_actions = (
            None if discrete_actions is None else np.asarray(discrete_actions)
        )
        self.arm_hold_probabilities = (
            None
            if arm_hold_probabilities is None
            else np.asarray(arm_hold_probabilities)
        )
        self.arm_hold_contract_hash = arm_hold_contract_hash
        self.error = error
        self.requests: list[dict[str, Any]] = []
        self.reset_count = 0

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(dict(observation))
        if self.error is not None:
            raise self.error
        response: dict[str, Any] = {
            "robot_actions": self.actions,
            "server_timing": {"infer_ms": 2.5},
        }
        if self.discrete_actions is not None:
            response["robot_action_tokens"] = self.discrete_actions
        if self.arm_hold_probabilities is not None:
            response["arm_hold_probability"] = self.arm_hold_probabilities
            response["arm_hold_output_contract_hash"] = (
                self.arm_hold_contract_hash
            )
        return response

    def reset(self) -> None:
        self.reset_count += 1


class _CancellableBlockingTransport(_FakeTransport):
    def __init__(self, *, metadata: dict[str, Any], actions: np.ndarray) -> None:
        super().__init__(metadata=metadata, actions=actions)
        self.entered = threading.Event()
        self.release = threading.Event()
        self.cancelled = threading.Event()
        self.begin_session_count = 0
        self.cancel_count = 0

    def begin_session(self) -> bool:
        self.begin_session_count += 1
        self.cancelled.clear()
        return True

    def request_cancel(self) -> bool:
        self.cancel_count += 1
        self.cancelled.set()
        return True

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        self.entered.set()
        while not self.release.wait(timeout=0.001):
            if self.cancelled.is_set():
                raise RuntimeError("synthetic transport session cancelled")
        return super().infer(observation)


def _state_only_camera_contract() -> PolicyCameraInputContract:
    return PolicyCameraInputContract(inputs=(), required=False)


def _continuous_artifact() -> PolicyArtifactContract:
    return PolicyArtifactContract(
        policy_io_layout=PolicyIOLayout.legacy_v21_q6(),
        model_id="openpi-final-chunk-unit",
        checkpoint_step=123,
        state_key="robot_state",
        state_discrete_key="robot_state_tokens",
        action_key="robot_actions",
        action_discrete_key="robot_action_tokens",
        camera_input=_state_only_camera_contract(),
        metadata={"fixture": "continuous"},
    )


def _arm_hold_contract() -> PolicyArmHoldOutputContract:
    return PolicyArmHoldOutputContract(
        output_key="arm_hold_probability",
        training_contract_hash=(
            DEFAULT_ARM_HOLD_TRAINING_CONTRACT_V1.semantic_content_hash
        ),
        enter_threshold=0.8,
        exit_threshold=0.2,
        enter_consecutive_steps=1,
        exit_consecutive_steps=1,
    )


def _arm_hold_artifact() -> PolicyArtifactContract:
    return replace(
        _continuous_artifact(),
        model_id="openpi-final-chunk-arm-hold-unit",
        arm_hold_output=_arm_hold_contract(),
    )


def _relative_artifact(*, horizon: int = HORIZON) -> PolicyArtifactContract:
    return replace(
        _continuous_artifact(),
        model_id="openpi-final-chunk-relative-unit",
        dataset_schema="dexteleop_dataset_v2.3",
        pose_action_chunk=PoseActionChunkContract(
            action_horizon=horizon,
            action_pose_mode="relative",
            last_action_pose_mode="relative",
        ),
        pose_training_data=PoseTrainingDataContract.v22_raw_identity(
            active_sides=("left", "right"),
        ),
        pose_frame_contract_hashes=(RELATIVE_FRAME_HASH,),
        metadata={
            "fixture": "relative",
            "pose_representation": {
                "action": "relative",
                "last_action": "relative",
            },
        },
    )


def _discrete_artifact() -> PolicyArtifactContract:
    token = DiscreteTokenSpec(
        "mode",
        (
            DiscreteCategory(0, "open", backend_value=0.0),
            DiscreteCategory(1, "close", backend_value=0.9),
        ),
        source_metadata_key="mode",
    )
    layout = PolicyIOLayout(
        name="openpi_rtc_right_discrete_unit",
        end_effectors={
            "left": EndEffectorSpec.none("left"),
            "right": EndEffectorSpec.discrete(
                "right",
                adapter="two_state_gripper_v1",
                target_tokens=(token,),
                target_legacy_q6_slot_indices=(0,),
            ),
        },
    )
    return PolicyArtifactContract(
        policy_io_layout=layout,
        model_id="openpi-final-chunk-discrete-unit",
        checkpoint_step=456,
        state_key="robot_state",
        state_discrete_key="robot_state_tokens",
        action_key="robot_actions",
        action_discrete_key="robot_action_tokens",
        camera_input=_state_only_camera_contract(),
        metadata={"fixture": "discrete"},
    )


def _server_metadata(
    artifact: PolicyArtifactContract,
    *,
    horizon: int | None = HORIZON,
) -> dict[str, Any]:
    layout = artifact.policy_io_layout
    metadata: dict[str, Any] = {
        "protocol_version": OPENPI_LOCAL_PROTOCOL_VERSION,
        "deploy_schema_version": artifact.deploy_schema_version,
        "model_id": artifact.model_id,
        "deployment": {"checkpoint_step": artifact.checkpoint_step},
        "model": {
            "action_discrete_dim": layout.action_discrete_dim,
            "policy_io_layout_hash": layout.layout_hash,
        },
        "local_robot": {
            "input_state_key": artifact.state_key,
            "input_state_dim": artifact.state_input_dim,
            "input_state_mode": artifact.state_input_mode,
            "input_state_discrete_key": artifact.state_discrete_key,
            "input_state_discrete_dim": layout.state_discrete_dim,
            "output_action_dim": layout.action_dim,
            "output_action_discrete_dim": layout.action_discrete_dim,
            "active_sides": ["left", "right"],
        },
        "dataset_adapter": {
            "policy_io_layout_hash": layout.layout_hash,
            "camera_slots": [],
        },
    }
    if horizon is not None:
        metadata["model"]["action_horizon"] = int(horizon)
    if artifact.arm_hold_output is not None:
        metadata["action_semantics"] = {
            "arm_hold_output_key": artifact.arm_hold_output.output_key,
            "arm_hold_output_contract_hash": (
                artifact.arm_hold_output.contract_hash
            ),
        }
    return metadata


def _absolute_actions(dim: int, *, horizon: int = HORIZON) -> np.ndarray:
    actions = np.zeros((horizon, dim), dtype=np.float32)
    actions[:, 6] = 1.0
    actions[:, 13] = 1.0
    for index in range(horizon):
        actions[index, 0] = 0.01 * index
        actions[index, 7] = -0.01 * index
        if dim == ACTION_DIM:
            actions[index, 14:] = np.float32(index / 10.0)
    return actions


def _observation(*, discrete: bool = False) -> DeploymentObservation:
    state = np.zeros((OBSERVATION_STATE_DIM,), dtype=np.float32)
    for side in ("left", "right"):
        pose_slice = STATE_ARM_POSE_SLICES[side]
        state[pose_slice.start + 6] = 1.0
    return DeploymentObservation(
        timestamp=9.9,
        step_index=99,
        active_sides=("left", "right"),
        arms_enabled=True,
        hands_enabled=True,
        end_effector_sides=(("right",) if discrete else ("left", "right")),
        state=state,
        state_valid=np.ones((OBSERVATION_STATE_DIM,), dtype=bool),
        arm_states={
            "left": {
                "actual_tcp_pose": np.asarray(
                    [0.4, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
                    dtype=np.float32,
                )
            },
            "right": {
                "actual_tcp_pose": np.asarray(
                    [-0.4, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
                    dtype=np.float32,
                )
            },
        },
        hand_states=({"right": {"mode": "open"}} if discrete else {}),
        metadata={"prompt": "unit task"},
    )


def _config(
    artifact: PolicyArtifactContract,
    *,
    result_action_representation: str = "legacy_normalized_v1",
    action_pose_mode: str = "absolute",
    frame_contract_hash: str = "frame:openpi-unit-v1",
) -> OpenPIFinalChunkRtcConfig:
    return OpenPIFinalChunkRtcConfig(
        artifact=artifact,
        frame_contract_hash=frame_contract_hash,
        normalization_hash="normalization:openpi-unit-v1",
        action_pose_mode=action_pose_mode,
        minimum_execution_horizon_steps=2,
        result_action_representation=result_action_representation,
    )


def _backend(
    artifact: PolicyArtifactContract,
    *,
    actions: np.ndarray | None = None,
    discrete_actions: np.ndarray | None = None,
    arm_hold_probabilities: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
    error: Exception | None = None,
    clock: Any | None = None,
    result_action_representation: str = "legacy_normalized_v1",
    action_pose_mode: str = "absolute",
    frame_contract_hash: str = "frame:openpi-unit-v1",
) -> tuple[OpenPIFinalChunkRtcBackend, _FakeTransport]:
    layout = artifact.policy_io_layout
    transport = _FakeTransport(
        metadata=(_server_metadata(artifact) if metadata is None else metadata),
        actions=(
            _absolute_actions(layout.action_dim)
            if actions is None
            else actions
        ),
        discrete_actions=discrete_actions,
        arm_hold_probabilities=arm_hold_probabilities,
        arm_hold_contract_hash=(
            None
            if artifact.arm_hold_output is None
            else artifact.arm_hold_output.contract_hash
        ),
        error=error,
    )
    return (
        OpenPIFinalChunkRtcBackend(
            transport=transport,
            config=_config(
                artifact,
                result_action_representation=result_action_representation,
                action_pose_mode=action_pose_mode,
                frame_contract_hash=frame_contract_hash,
            ),
            clock=(_CounterClock() if clock is None else clock),
        ),
        transport,
    )


def test_openpi_policy_raw_rtc_adapter_reuses_existing_transport() -> None:
    artifact = _continuous_artifact()
    transport = _FakeTransport(
        metadata=_server_metadata(artifact),
        actions=_absolute_actions(artifact.policy_io_layout.action_dim),
    )
    remote = RemoteModelPolicy(
        transport=transport,
        observation_adapter=lambda value: dict(value),
        config=RemotePolicyConfig(
            action_key=artifact.action_key,
            action_dim=artifact.policy_io_layout.action_dim,
            action_discrete_key=artifact.action_discrete_key,
            action_discrete_dim=artifact.policy_io_layout.action_discrete_dim,
            contract=deploy_contract_from_policy_artifact(
                artifact,
                protocol_version=OPENPI_LOCAL_PROTOCOL_VERSION,
            ),
            source="openpi_remote_policy",
        ),
    )
    policy = OpenPIClientPolicy(
        OpenPIClientPolicyConfig(policy_artifact=artifact),
        remote_policy=remote,
    )

    adapter = policy.create_raw_rtc_adapter(
        frame_contract_hash="frame:openpi-unit-v1",
        normalization_hash="normalization:openpi-unit-v1",
    )

    assert adapter._transport is transport
    assert adapter._remote_policy is remote
    assert adapter.config.artifact is artifact
    assert adapter.request_spec.mode.continuation_mode == RTC_CONTINUATION_NONE

    typed_adapter = policy.create_raw_rtc_adapter(
        frame_contract_hash="frame:openpi-unit-v1",
        normalization_hash="normalization:openpi-unit-v1",
        result_action_representation=OPENPI_RTC_RESULT_TYPED_CANONICAL_V1,
    )
    assert typed_adapter._transport is transport
    assert (
        typed_adapter.config.result_action_representation
        == OPENPI_RTC_RESULT_TYPED_CANONICAL_V1
    )
    assert typed_adapter.config.action_codec_hash == openpi_v22_native_codec_hash(
        artifact
    )


def test_openpi_raw_rtc_ignores_inactive_discrete_key_name() -> None:
    artifact = _continuous_artifact()
    assert artifact.policy_io_layout.action_discrete_dim == 0
    assert artifact.action_discrete_key is not None
    transport = _FakeTransport(
        metadata=_server_metadata(artifact),
        actions=_absolute_actions(artifact.policy_io_layout.action_dim),
    )
    remote = RemoteModelPolicy(
        transport=transport,
        observation_adapter=lambda value: dict(value),
        config=RemotePolicyConfig(
            action_key=artifact.action_key,
            action_dim=artifact.policy_io_layout.action_dim,
            action_discrete_key=None,
            action_discrete_dim=0,
            contract=deploy_contract_from_policy_artifact(
                artifact,
                protocol_version=OPENPI_LOCAL_PROTOCOL_VERSION,
            ),
            source="openpi_remote_policy",
        ),
    )

    backend = OpenPIFinalChunkRtcBackend(
        remote_policy=remote,
        config=_config(artifact),
    )

    assert backend._remote_policy is remote
    assert backend._remote_policy.config.action_discrete_key is None


def test_openpi_policy_raw_rtc_adapter_accepts_relative_artifact_explicitly() -> None:
    artifact = _relative_artifact()
    transport = _FakeTransport(
        metadata=_server_metadata(artifact),
        actions=_absolute_actions(artifact.policy_io_layout.action_dim),
    )
    remote = RemoteModelPolicy(
        transport=transport,
        observation_adapter=lambda value: dict(value),
        config=RemotePolicyConfig(
            action_key=artifact.action_key,
            action_dim=artifact.policy_io_layout.action_dim,
            action_discrete_key=artifact.action_discrete_key,
            action_discrete_dim=artifact.policy_io_layout.action_discrete_dim,
            contract=deploy_contract_from_policy_artifact(
                artifact,
                protocol_version=OPENPI_LOCAL_PROTOCOL_VERSION,
            ),
            source="openpi_remote_policy",
        ),
    )
    policy = OpenPIClientPolicy(
        OpenPIClientPolicyConfig(policy_artifact=artifact),
        remote_policy=remote,
    )

    adapter = policy.create_raw_rtc_adapter(
        frame_contract_hash=RELATIVE_FRAME_HASH,
        normalization_hash="normalization:openpi-unit-v1",
        action_pose_mode="relative",
        result_action_representation=OPENPI_RTC_RESULT_TYPED_CANONICAL_V1,
    )

    assert adapter._transport is transport
    assert adapter._remote_policy is remote
    assert adapter.config.action_pose_mode == "relative"
    assert adapter.config.action_codec_hash == (
        openpi_v22_relative_native_codec_hash(artifact)
    )
    assert callable(adapter.representation_context_builder)


def test_openpi_raw_rtc_adapter_runs_through_shared_frontend_without_dispatch() -> None:
    artifact = _continuous_artifact()
    backend, transport = _backend(artifact)
    runtime = PolicyDeployRuntime(
        PolicyDeployRuntimeConfig(
            active_sides=("left", "right"),
            arms_enabled=True,
            hands_enabled=True,
            action_keyframe_period_sec=1.0 / 30.0,
            action_maximum_hold_sec=1.0 / 30.0,
        )
    )
    pipeline = CanonicalDeployActionPipeline(
        safety_filter=ActionSafetyFilter(
            ActionSafetyFilterConfig(
                active_sides=("left", "right"),
                arms_enabled=True,
                hands_enabled=True,
                end_effector_sides=("left", "right"),
                max_arm_xyz_step_m=0.05,
                max_arm_rot_step_deg=20.0,
                max_arm_joint_step_rad=0.2,
                max_hand_step=1.0,
            )
        ),
        active_sides=("left", "right"),
        arms_enabled=True,
        hands_enabled=True,
    )
    base = time.monotonic()
    frontend = build_rtc_model_shadow_frontend(
        backend,
        runtime_session_id="session:openpi-raw-rtc-frontend",
        control_domain_id="domain:openpi-raw-rtc",
        producer_id="producer:openpi-raw-rtc",
        policy_node_id="single_policy",
        deploy_runtime=runtime,
        preparer=pipeline,
        created_at=base,
        profile=RAW_RTC_FINAL_CHUNK_PROFILE_V1,
    )
    frontend.start()
    emitting_step = None
    emitting_history = None
    try:
        for index in range(160):
            now = base + index / 30.0
            step = frontend.step(_observation(), now=now)
            if step.action_preparation.emission is not None:
                emitting_step = step
                break
            time.sleep(0.001)
    finally:
        frontend.close(now=base + 6.0)

    assert emitting_step is not None
    assert transport.requests
    assert emitting_step.to_debug_payload()["send_actions"] is False


def test_openpi_raw_rtc_arm_hold_sidecar_resolves_and_commits_at_emission() -> None:
    artifact = _arm_hold_artifact()
    history_binding = ArmHoldResolvedHistoryBinding(
        binding_id="policy-source:openpi-raw-rtc-arm-hold",
        epoch=2,
        contract_hash=artifact.arm_hold_output.contract_hash,
    )
    actions = _absolute_actions(artifact.policy_io_layout.action_dim)
    actions[:, 0] = 0.5
    actions[:, 7] = -0.5
    probabilities = np.tile(
        np.asarray(((1.0, 0.0),), dtype=np.float32),
        (HORIZON, 1),
    )
    backend, transport = _backend(
        artifact,
        actions=actions,
        arm_hold_probabilities=probabilities,
    )
    runtime = PolicyDeployRuntime(
        PolicyDeployRuntimeConfig(
            active_sides=("left", "right"),
            arms_enabled=True,
            hands_enabled=True,
            action_keyframe_period_sec=1.0 / 30.0,
            action_maximum_hold_sec=1.0 / 30.0,
            arm_hold_output=artifact.arm_hold_output,
        ),
        arm_hold_history_binding=history_binding,
    )
    pipeline = CanonicalDeployActionPipeline(
        safety_filter=ActionSafetyFilter(
            ActionSafetyFilterConfig(
                active_sides=("left", "right"),
                arms_enabled=True,
                hands_enabled=True,
                end_effector_sides=("left", "right"),
                max_arm_xyz_step_m=1.0,
                max_arm_rot_step_deg=180.0,
                max_arm_joint_step_rad=1.0,
                max_hand_step=1.0,
            )
        ),
        active_sides=("left", "right"),
        arms_enabled=True,
        hands_enabled=True,
    )
    base = time.monotonic()
    frontend = build_rtc_model_shadow_frontend(
        backend,
        runtime_session_id="session:openpi-raw-rtc-arm-hold",
        control_domain_id="domain:openpi-raw-rtc-arm-hold",
        producer_id="producer:openpi-raw-rtc-arm-hold",
        policy_node_id="single_policy",
        deploy_runtime=runtime,
        preparer=pipeline,
        created_at=base,
        profile=RAW_RTC_FINAL_CHUNK_PROFILE_V1,
    )
    frontend.start()
    emitting_step = None
    try:
        for index in range(160):
            step = frontend.step(
                _observation(),
                now=base + index / 30.0,
            )
            if step.emission_result.emission is not None:
                emitting_step = step
                emitting_history = runtime.arm_hold_resolved_history_snapshot()
                break
            time.sleep(0.001)
    finally:
        frontend.close(now=base + 6.0)

    assert emitting_step is not None
    assert transport.requests
    emission = emitting_step.emission_result.emission
    assert emission is not None
    assert emission.arm_hold_resolution is not None
    assert emission.arm_hold_resolution.resolved_combination == "10"
    assert emission.model_decoded_absolute_action is not None
    assert emission.decoded_absolute_action is not None
    assert emission.model_decoded_absolute_action.vector[0] == pytest.approx(0.5)
    assert emission.decoded_absolute_action.vector[0] == pytest.approx(0.0)
    assert emitting_step.emission_commit_transition is not None
    assert emitting_step.emission_commit_transition.receipt.accepted
    assert emitting_step.action_preparation.arm_hold_resolution is (
        emission.arm_hold_resolution
    )
    assert emitting_history is not None
    assert emitting_history.binding.semantic_content_hash == (
        history_binding.semantic_content_hash
    )
    assert emitting_history.last_resolution_sequence == 1
    assert emitting_history.context.previous_mode.tolist() == [1.0, 0.0]
    assert emitting_history.context.dwell_steps.tolist() == [1, 1]
    closed_history = runtime.arm_hold_resolved_history_snapshot()
    assert closed_history is not None
    assert closed_history.binding.epoch == history_binding.epoch + 1
    assert closed_history.context.available_mask.tolist() == [False, False]
    runtime.clear_action_queue()
    runtime.reset_prediction_schedule()
    repeated_close_reset = runtime.arm_hold_resolved_history_snapshot()
    assert repeated_close_reset is not None
    assert repeated_close_reset.binding.semantic_content_hash == (
        closed_history.binding.semantic_content_hash
    )
    assert emitting_step.to_debug_payload()["send_actions"] is False


def test_openpi_raw_rtc_arm_hold_missing_reference_faults_without_partial_commit() -> None:
    artifact = _arm_hold_artifact()
    history_binding = ArmHoldResolvedHistoryBinding(
        binding_id="policy-source:openpi-raw-rtc-arm-hold-fault",
        epoch=0,
        contract_hash=artifact.arm_hold_output.contract_hash,
    )
    probabilities = np.tile(
        np.asarray(((1.0, 1.0),), dtype=np.float32),
        (HORIZON, 1),
    )
    backend, _transport = _backend(
        artifact,
        arm_hold_probabilities=probabilities,
    )
    runtime = PolicyDeployRuntime(
        PolicyDeployRuntimeConfig(
            active_sides=("left", "right"),
            arms_enabled=True,
            hands_enabled=True,
            action_keyframe_period_sec=1.0 / 30.0,
            action_maximum_hold_sec=1.0 / 30.0,
            arm_hold_output=artifact.arm_hold_output,
        ),
        arm_hold_history_binding=history_binding,
    )
    observation = _observation()
    state = observation.state.copy()
    state[STATE_ARM_POSE_SLICES["right"]] = np.nan
    observation = replace(observation, state=state)
    pipeline = CanonicalDeployActionPipeline(
        safety_filter=ActionSafetyFilter(
            ActionSafetyFilterConfig(
                active_sides=("left", "right"),
                arms_enabled=True,
                hands_enabled=True,
                end_effector_sides=("left", "right"),
                max_arm_xyz_step_m=1.0,
                max_arm_rot_step_deg=180.0,
                max_arm_joint_step_rad=1.0,
                max_hand_step=1.0,
            )
        ),
        active_sides=("left", "right"),
        arms_enabled=True,
        hands_enabled=True,
    )
    base = time.monotonic()
    frontend = build_rtc_model_shadow_frontend(
        backend,
        runtime_session_id="session:openpi-raw-rtc-arm-hold-fault",
        control_domain_id="domain:openpi-raw-rtc-arm-hold-fault",
        producer_id="producer:openpi-raw-rtc-arm-hold-fault",
        policy_node_id="single_policy",
        deploy_runtime=runtime,
        preparer=pipeline,
        created_at=base,
        profile=RAW_RTC_FINAL_CHUNK_PROFILE_V1,
    )
    frontend.start()
    fault_step = None
    try:
        for index in range(160):
            step = frontend.step(
                observation,
                now=base + index / 30.0,
            )
            if step.state.lifecycle_state == POLICY_ASYNC_PRODUCER_FAULT:
                fault_step = step
                break
            time.sleep(0.001)
    finally:
        frontend.close(now=base + 6.0)

    assert fault_step is not None
    assert fault_step.emission_result.emission is None
    assert fault_step.emission_commit_transition is None
    assert runtime._arm_hold_resolver.resolved_combination == "00"
    history = runtime.arm_hold_resolved_history_snapshot()
    assert history is not None
    assert history.binding.semantic_content_hash == (
        history_binding.semantic_content_hash
    )
    assert history.binding.epoch == history_binding.epoch
    assert history.last_resolution_sequence is None
    assert history.context.available_mask.tolist() == [False, False]


def test_openpi_relative_raw_rtc_runs_through_shared_frontend_without_dispatch() -> None:
    artifact = _relative_artifact()
    relative = _absolute_actions(artifact.policy_io_layout.action_dim)
    backend, transport = _backend(
        artifact,
        actions=relative,
        action_pose_mode="relative",
        frame_contract_hash=RELATIVE_FRAME_HASH,
        result_action_representation=OPENPI_RTC_RESULT_TYPED_CANONICAL_V1,
    )
    runtime = PolicyDeployRuntime(
        PolicyDeployRuntimeConfig(
            active_sides=("left", "right"),
            arms_enabled=True,
            hands_enabled=True,
            policy_action_pose_mode="relative",
            action_keyframe_period_sec=1.0 / 30.0,
            action_maximum_hold_sec=1.0 / 30.0,
        )
    )
    base = time.monotonic()
    frontend = build_rtc_model_shadow_frontend(
        backend,
        runtime_session_id="session:openpi-relative-raw-rtc-frontend",
        control_domain_id="domain:openpi-relative-raw-rtc",
        producer_id="producer:openpi-relative-raw-rtc",
        policy_node_id="single_policy",
        deploy_runtime=runtime,
        preparer=None,
        created_at=base,
        profile=RAW_RTC_FINAL_CHUNK_PROFILE_V1,
    )
    frontend.start()
    emitting_step = None
    try:
        for index in range(160):
            now = base + index / 30.0
            step = frontend.step(_observation(), now=now)
            emission = step.action_preparation.emission
            if emission is not None:
                emitting_step = step
                break
            time.sleep(0.001)
    finally:
        frontend.close(now=base + 6.0)

    assert emitting_step is not None
    emission = emitting_step.action_preparation.emission
    assert emission is not None
    assert emission.canonical_action is not None
    assert (
        emission.canonical_action.canonicalization_receipt.source_pose_mode
        == "relative"
    )
    assert emission.canonical_action.action.arm_frame("left") == "canonical_desk"
    assert transport.requests
    assert emitting_step.to_debug_payload()["send_actions"] is False


def _request(backend: OpenPIFinalChunkRtcBackend) -> PolicyInferenceRequest:
    spec = backend.request_spec
    return PolicyInferenceRequest(
        request_id="request:openpi-rtc-unit",
        inference_id="inference:openpi-rtc-unit",
        rtc_mode=ASYNC_ALIGNED_CHUNK_MODE_V1,
        session_id="session:openpi-rtc-unit",
        control_domain_id="domain:openpi-rtc-unit",
        producer_id="producer:openpi-rtc-unit",
        producer_epoch=1,
        policy_node_id="node:openpi-rtc-unit",
        artifact_hash=spec.artifact_hash,
        observation_cut_id="observation:openpi-rtc-unit",
        observation_timestamp=9.9,
        observation_step_index=99,
        triggered_at=10.0,
        admitted_at=10.0,
        first_target_effective_at=10.1,
        clock_domain_id="monotonic:unit",
        action_period_sec=0.1,
        prediction_horizon_steps=spec.prediction_horizon_steps,
        minimum_execution_horizon_steps=spec.minimum_execution_horizon_steps,
        source_plan_id="plan:openpi-rtc-unit",
        source_plan_generation=0,
        source_plan_hash="plan-hash:openpi-rtc-unit",
        source_plan_cursor_at=10.0,
        producer_plan_committed_through=None,
        forecast_delay_steps=1,
        delay_estimator_spec_hash="delay:openpi-rtc-unit",
        action_codec_hash=spec.action_codec_hash,
        frame_contract_hash=spec.frame_contract_hash,
        layout_hash=spec.layout_hash,
        normalization_hash=spec.normalization_hash,
        valid_until=11.0,
    )


def test_continuous_final_chunk_offline_round_trip_is_exact() -> None:
    artifact = _continuous_artifact()
    expected = _absolute_actions(artifact.policy_io_layout.action_dim)
    backend, transport = _backend(artifact, actions=expected)
    prepared = backend.prepare_input(_observation())
    bundle = backend.create_client().infer(
        _request(backend),
        model_input=prepared,
        model_input_hash=prepared.semantic_content_hash,
    )

    assert bundle.response_validation.accepted
    assert bundle.response.conditioning_receipt is None
    assert bundle.response.model_space_discrete_actions is None
    assert bundle.policy_chunk_result.protocol_version == "dexteleop_rtc_protocol_v2"
    assert len(bundle.policy_chunk_result.decoded_absolute_actions) == HORIZON
    for index, action in enumerate(bundle.policy_chunk_result.decoded_absolute_actions):
        np.testing.assert_array_equal(action.vector, expected[index])
        assert action.metadata["source"] == OPENPI_FINAL_CHUNK_RTC_ADAPTER_V1
        assert action.metadata["pose_action_mode"] == "absolute"
    assert len(transport.requests) == 1
    assert "robot_state" in transport.requests[0]
    assert isinstance(transport.requests[0]["dexteleop"], dict)
    assert transport.requests[0]["dexteleop"]["step_index"] == 99


def test_typed_absolute_final_chunk_is_additive_and_keeps_default_legacy() -> None:
    artifact = _continuous_artifact()
    expected = _absolute_actions(artifact.policy_io_layout.action_dim)
    legacy, _ = _backend(artifact, actions=expected)
    typed, _ = _backend(
        artifact,
        actions=expected,
        result_action_representation=OPENPI_RTC_RESULT_TYPED_CANONICAL_V1,
    )

    assert legacy.config.action_codec_hash == openpi_v22_legacy_codec_hash(artifact)
    assert typed.config.action_codec_hash == openpi_v22_native_codec_hash(artifact)
    assert legacy.config.action_codec_hash != typed.config.action_codec_hash

    prepared = typed.prepare_input(_observation())
    bundle = typed.create_client().infer(
        _request(typed),
        model_input=prepared,
        model_input_hash=prepared.semantic_content_hash,
    )
    result = bundle.policy_chunk_result

    assert result.decoded_absolute_actions == ()
    assert len(result.canonical_actions) == HORIZON
    assert tuple(
        action.canonicalization_receipt.source_index
        for action in result.canonical_actions
    ) == tuple(range(HORIZON))
    for index, canonical in enumerate(result.canonical_actions):
        np.testing.assert_array_equal(
            canonical.action.arm_pose_by_side["left"],
            expected[index, :7],
        )
        np.testing.assert_array_equal(
            canonical.action.arm_pose_by_side["right"],
            expected[index, 7:14],
        )
        assert canonical.action.arm_frame("left") == (
            artifact.policy_io_layout.arm_reference_frame_by_side["left"]
        )
        assert canonical.action.arm_controlled_frame("left") == (
            artifact.policy_io_layout.arm_controlled_frame_by_side["left"]
        )
        assert canonical.canonicalization_receipt.conversion_contract_hash == (
            typed.config.absolute_conversion_contract_hash
        )


def test_typed_relative_final_chunk_uses_original_request_anchor() -> None:
    artifact = _relative_artifact()
    relative = _absolute_actions(artifact.policy_io_layout.action_dim)
    relative[:, 14:] = np.arange(
        HORIZON * (artifact.policy_io_layout.action_dim - 14),
        dtype=np.float32,
    ).reshape(HORIZON, -1)
    backend, _ = _backend(
        artifact,
        actions=relative,
        action_pose_mode="relative",
        frame_contract_hash=RELATIVE_FRAME_HASH,
        result_action_representation=OPENPI_RTC_RESULT_TYPED_CANONICAL_V1,
    )
    runtime = PolicyDeployRuntime(
        PolicyDeployRuntimeConfig(
            active_sides=("left", "right"),
            arms_enabled=True,
            hands_enabled=True,
        )
    )
    actor_observation, parity = runtime.prepare_actor_observation(_observation())
    assert parity is None or parity.matched
    builder = backend.representation_context_builder
    assert callable(builder)
    context = builder(actor_observation, "observation:openpi-rtc-unit")
    request = replace(
        _request(backend),
        action_representation_context=context,
    )
    prepared = backend.prepare_input(_observation())
    result = backend.create_client().infer(
        request,
        model_input=prepared,
        model_input_hash=prepared.semantic_content_hash,
    ).policy_chunk_result

    assert backend.config.action_codec_hash == (
        openpi_v22_relative_native_codec_hash(artifact)
    )
    assert backend.config.conversion_contract_hash == (
        openpi_relative_conversion_contract_hash(
            artifact,
            frame_contract_hash=RELATIVE_FRAME_HASH,
            active_sides=("left", "right"),
        )
    )
    assert len(result.canonical_actions) == HORIZON
    left_anchor = np.asarray([0.4, 0.2, 0.3], dtype=np.float32)
    right_anchor = np.asarray([-0.4, -0.2, 0.3], dtype=np.float32)
    for index, canonical in enumerate(result.canonical_actions):
        action = canonical.action
        np.testing.assert_allclose(
            action.arm_pose_by_side["left"][:3],
            left_anchor + relative[index, :3],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            action.arm_pose_by_side["right"][:3],
            right_anchor + relative[index, 7:10],
            atol=1e-6,
        )
        assert action.arm_frame("left") == "canonical_desk"
        assert action.arm_frame("right") == "canonical_desk"
        np.testing.assert_array_equal(
            action.end_effector_targets_by_side["left"].values,
            relative[index, 14:20],
        )
        np.testing.assert_array_equal(
            action.end_effector_targets_by_side["right"].values,
            relative[index, 20:26],
        )
        receipt = canonical.canonicalization_receipt
        assert receipt.source_pose_mode == "relative"
        assert receipt.anchor_identity_hash == context.semantic_content_hash
        assert receipt.conversion_contract_hash == (
            backend.config.conversion_contract_hash
        )

    moved = _observation()
    moved.arm_states["left"]["actual_tcp_pose"][0] = np.float32(9.0)
    moved_actor, moved_parity = runtime.prepare_actor_observation(moved)
    assert moved_parity is None or moved_parity.matched
    moved_context = builder(moved_actor, "observation:moved")
    assert moved_context.semantic_content_hash != context.semantic_content_hash
    np.testing.assert_allclose(
        result.canonical_actions[0].action.arm_pose_by_side["left"][:3],
        left_anchor + relative[0, :3],
        atol=1e-6,
    )


def test_openpi_typed_frontend_context_is_absolute_and_legacy_stays_empty() -> None:
    artifact = _continuous_artifact()
    legacy, _ = _backend(artifact)
    typed, _ = _backend(
        artifact,
        result_action_representation=OPENPI_RTC_RESULT_TYPED_CANONICAL_V1,
    )
    runtime = PolicyDeployRuntime(
        PolicyDeployRuntimeConfig(
            active_sides=("left", "right"),
            arms_enabled=True,
            hands_enabled=True,
        )
    )
    policy_observation = runtime.prepare_policy_observation(_observation())
    actor_observation, parity = runtime.prepare_actor_observation(policy_observation)
    assert parity is None or parity.matched

    assert legacy.representation_context_builder is None
    builder = typed.representation_context_builder
    assert callable(builder)
    context = builder(actor_observation, "cut:openpi-typed-absolute")

    assert context.source_pose_mode == "absolute"
    assert context.anchor_identity_hash is None
    assert context.arm_anchor_pose_by_side == {}
    assert context.observation_cut_id == "cut:openpi-typed-absolute"
    assert context.observation_contract_hash == actor_observation.contract_hash
    assert dict(context.arm_reference_frame_by_side) == {
        side: artifact.policy_io_layout.arm_reference_frame_by_side[side]
        for side in ("left", "right")
    }


def test_typed_absolute_openpi_preserves_discrete_end_effector_tokens() -> None:
    artifact = _discrete_artifact()
    actions = _absolute_actions(artifact.policy_io_layout.action_dim)
    tokens = (np.arange(HORIZON, dtype=np.int64) % 2).reshape(HORIZON, 1)
    backend, _ = _backend(
        artifact,
        actions=actions,
        discrete_actions=tokens,
        result_action_representation=OPENPI_RTC_RESULT_TYPED_CANONICAL_V1,
    )
    prepared = backend.prepare_input(_observation(discrete=True))
    result = backend.create_client().infer(
        _request(backend),
        model_input=prepared,
        model_input_hash=prepared.semantic_content_hash,
    ).policy_chunk_result

    for index, canonical in enumerate(result.canonical_actions):
        target = canonical.action.end_effector_targets_by_side["right"]
        assert target.value_type == "discrete"
        assert target.adapter == "two_state_gripper_v1"
        np.testing.assert_array_equal(target.values, tokens[index])


def test_openpi_final_chunk_builds_model_neutral_worker_executor() -> None:
    artifact = _continuous_artifact()
    backend, transport = _backend(artifact)
    runtime = PolicyDeployRuntime(
        PolicyDeployRuntimeConfig(
            active_sides=("left", "right"),
            arms_enabled=True,
            hands_enabled=True,
        )
    )
    policy_observation = runtime.prepare_policy_observation(_observation())
    actor_observation, parity = runtime.prepare_actor_observation(policy_observation)
    assert parity is None or parity.matched
    bundles = []

    result = backend.create_worker_executor(
        observe_bundle=bundles.append,
    )(_request(backend), actor_observation)

    assert result.request_id == "request:openpi-rtc-unit"
    assert result.protocol_version == "dexteleop_rtc_protocol_v2"
    assert len(result.decoded_absolute_actions) == HORIZON
    assert len(bundles) == 1
    assert bundles[0].policy_chunk_result is result
    assert len(transport.requests) == 1


def test_rtc_client_exchange_debug_payload_is_identity_and_latency_only() -> None:
    artifact = _continuous_artifact()
    backend, _transport = _backend(artifact)
    prepared = backend.prepare_input(_observation())
    bundle = backend.create_client().infer(
        _request(backend),
        model_input=prepared,
        model_input_hash=prepared.semantic_content_hash,
    )

    payload = rtc_client_inference_bundle_debug_payload(bundle)

    assert payload["schema"] == "dexteleop_rtc_client_exchange_v1"
    assert payload["debug_only"] is True
    assert payload["non_authoritative"] is True
    assert payload["action_send_authority"] is False
    assert payload["request_hash"] == bundle.envelope.request.semantic_content_hash
    assert payload["envelope_hash"] == bundle.envelope.semantic_content_hash
    assert payload["response_hash"] == bundle.response.semantic_content_hash
    assert payload["model_input_hash"] == bundle.envelope.model_input_hash
    assert payload["conditioning_receipt_hash"] is None
    assert payload["inference_duration_sec"] >= 0.0
    assert payload["result_duration_sec"] >= payload["inference_duration_sec"]
    assert payload["postprocess_duration_sec"] >= 0.0


def test_openpi_backend_satisfies_generic_frontend_adapter_without_shim() -> None:
    backend, _ = _backend(_continuous_artifact())
    assert isinstance(backend, RtcModelRuntimeAdapter)
    config = rtc_shadow_config_from_model_adapter(
        backend,
        runtime_session_id="session:openpi-generic-binding",
        control_domain_id="domain:openpi-generic-binding",
        producer_id="producer:openpi-generic-binding",
        policy_node_id="node:openpi-generic-binding",
    )

    assert config.artifact_hash == backend.request_spec.artifact_hash
    assert config.action_codec_hash == backend.request_spec.action_codec_hash
    assert config.frame_contract_hash == backend.request_spec.frame_contract_hash
    assert config.layout_hash == backend.request_spec.layout_hash
    assert config.normalization_hash == backend.request_spec.normalization_hash
    assert (
        config.prediction_horizon_steps
        == backend.request_spec.prediction_horizon_steps
    )


class _FakeConditionedOpenPITransport(_FakeTransport):
    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        marker = observation.get("__openpi_request__")
        if marker is None:
            return super().infer(observation)
        self.requests.append(dict(observation))
        assert marker == {
            "wire_protocol": OPENPI_CONDITIONED_RTC_WIRE_V1,
            "operation": OPENPI_CONDITIONED_RTC_OPERATION_V1,
        }
        external = np.asarray(
            observation["conditioning_actions"],
            dtype=np.float32,
        )
        mask = np.asarray(observation["prefix_mask"], dtype=np.float32)
        internal = np.zeros((external.shape[0], 32), dtype=np.float32)
        internal[:, : external.shape[1]] = external * 0.5
        provider = self._metadata[OPENPI_CONDITIONED_RTC_METADATA_KEY_V1]
        return {
            "robot_actions": self.actions.copy(),
            "rtc_conditioning_receipt": {
                "wire_protocol": OPENPI_CONDITIONED_RTC_WIRE_V1,
                "conditioning_mode": observation["conditioning_mode"],
                "receipt_context": dict(observation["receipt_context"]),
                "provider_capability_hash": (
                    provider["provider_capability_hash"]
                ),
                "continuation_implementation_id": (
                    provider["continuation_implementation_id"]
                ),
                "continuation_implementation_parameters_hash": (
                    provider[
                        "continuation_implementation_parameters_hash"
                    ]
                ),
                "sampler_constraint": provider["sampler_constraint"],
                "consumed_external_prefix_hash": (
                    rtc_float32_array_hash(external)
                ),
                "consumed_internal_prefix_hash": (
                    rtc_float32_array_hash(internal)
                ),
                "consumed_mask_hash": rtc_binary_mask_hash(mask),
                "consumed_external_shape": list(external.shape),
                "consumed_internal_shape": list(internal.shape),
                "consumed_generation_indices": [
                    int(index) for index in np.flatnonzero(mask)
                ],
            },
        }


def _model_rtc_contract(
    artifact: PolicyArtifactContract,
) -> ModelContinuationDataContract:
    config = _config(
        artifact,
        result_action_representation=OPENPI_RTC_RESULT_TYPED_CANONICAL_V1,
    )
    return ModelContinuationDataContract(
        continuation_mode=MODEL_CONTINUATION_FLOW_INPAINT_V1,
        continuation_implementation_id="openpi.pi0.hard-flow-prefix.v1",
        continuation_implementation_parameters_hash=(
            "sha256:" + "a" * 64
        ),
        prediction_horizon_steps=HORIZON,
        minimum_forecast_delay_steps=1,
        maximum_forecast_delay_steps=3,
        minimum_prefix_steps=1,
        maximum_prefix_steps=1,
        model_action_dim=config.artifact.policy_io_layout.action_dim,
        model_action_discrete_dim=0,
        component_scope=config.component_scope,
        supported_action_pose_modes=("absolute",),
        sampler_constraints=(MODEL_CONTINUATION_SAMPLER_CLAMPED_V1,),
        training_prefix_source_kinds=(
            TRAINING_PREFIX_TRAJECTORY_OVERLAP_TEACHER_V1,
        ),
        runtime_prefix_source_roles=(
            RUNTIME_PREFIX_PRODUCER_LOCAL_OVERLAP_V1,
        ),
        action_codec_hash=config.action_codec_hash,
        layout_hash=config.artifact.layout_hash,
        frame_contract_hash=config.frame_contract_hash,
        normalization_hash=config.normalization_hash,
        training_projection_implementation_id="openpi-teacher.v1",
        training_projection_parameters_hash="sha256:openpi-teacher",
        supports_unconditioned_bootstrap=False,
    )


def _conditioned_bundle_and_request(
    backend: OpenPIFinalChunkRtcBackend,
):
    raw_request = _request(backend)
    canonical_actions = tuple(
        backend.decoder(
            _absolute_actions(
                backend.config.artifact.policy_io_layout.action_dim
            ),
            None,
            raw_request,
        )
    )
    keyframes = tuple(
        TimedPolicyActionKeyframe(
            target_effective_at=10.0 + index * 0.1,
            effective_until=10.1 + index * 0.1,
            effective_clock_domain_id=raw_request.clock_domain_id,
            action_period_sec=0.1,
            decoded_absolute_action=None,
            canonical_action=action,
            request_id="request:openpi-old",
            inference_id="inference:openpi-old",
            policy_output_id="output:openpi-old",
            chunk_id=1,
            chunk_keyframe_index=index,
            chunk_size=HORIZON,
            observation_cut_id="observation:openpi-old",
            producer_epoch=raw_request.producer_epoch,
            policy_node_id=raw_request.policy_node_id,
            artifact_hash=raw_request.artifact_hash,
            produced_at=9.8,
            schedule_spec_id="schedule:openpi-model-rtc",
            schedule_spec_hash="schedule-hash:openpi-model-rtc",
            admitted_plan_generation=3,
            state=POLICY_PLAN_KEYFRAME_MUTABLE,
            valid_until=11.0,
        )
        for index, action in enumerate(canonical_actions)
    )
    plan = TimedPolicyActionPlanSnapshot(
        plan_id="plan:openpi-model-rtc",
        session_id=raw_request.session_id,
        control_domain_id=raw_request.control_domain_id,
        producer_id=raw_request.producer_id,
        producer_epoch=raw_request.producer_epoch,
        policy_node_id=raw_request.policy_node_id,
        clock_domain_id=raw_request.clock_domain_id,
        generation=3,
        cursor_at=9.9,
        created_at=9.8,
        keyframes=keyframes,
    )
    timeline = RtcContinuationTimeline(
        clock_domain_id=raw_request.clock_domain_id,
        generation_start_at=10.0,
        first_target_effective_at=10.1,
        action_period_sec=0.1,
        prediction_horizon_steps=HORIZON,
        forecast_delay_steps=1,
    )
    selection = select_rtc_plan_continuation(
        plan,
        timeline=timeline,
        spec=RtcContinuationSelectionSpec(
            prefix_horizon_steps=1,
            minimum_selected_steps=1,
        ),
    )
    context = PolicyActionRepresentationContext(
        source_pose_mode="absolute",
        observation_cut_id=raw_request.observation_cut_id,
        observation_contract_hash="observation-contract:openpi-model-rtc",
        observation_timestamp=raw_request.observation_timestamp,
        observation_step_index=raw_request.observation_step_index,
        active_sides=backend.config.active_sides,
        anchor_policy=ACTION_ANCHOR_POLICY_NONE_V1,
        arm_reference_frame_by_side=(
            backend.config.arm_reference_frame_by_side
        ),
        arm_controlled_frame_by_side={
            side: (
                backend.config.artifact.policy_io_layout
                .arm_controlled_frame_by_side[side]
            )
            for side in backend.config.active_sides
        },
        arm_anchor_pose_by_side={},
    )
    contract = backend.config.artifact.model_continuation
    assert contract is not None
    bundle = build_canonical_rtc_conditioning_bundle(
        runtime_session_id=raw_request.session_id,
        control_domain_id=raw_request.control_domain_id,
        artifact_hash=raw_request.artifact_hash,
        timeline=timeline,
        selection=selection,
        action_representation_context=context,
        continuation_contract=contract,
        prefix_schedule_semantics=(
            RTC_PREFIX_SCHEDULE_EXPLICIT_SELECTED_STEPS_V1
        ),
    )
    request = replace(
        raw_request,
        rtc_mode=RTC_FLOW_INPAINT_MODE_V1,
        first_target_effective_at=timeline.first_target_effective_at,
        source_plan_id=plan.plan_id,
        source_plan_generation=plan.generation,
        source_plan_hash=plan.semantic_content_hash,
        source_plan_cursor_at=plan.cursor_at,
        forecast_delay_steps=timeline.forecast_delay_steps,
        action_representation_context=context,
        conditioning_binding_hash=bundle.conditioning_binding_hash,
    )
    return contract, bundle, request


def _conditioned_artifact_and_metadata(
    *,
    inference_only_v2: bool,
) -> tuple[PolicyArtifactContract, dict[str, Any]]:
    base_artifact = _continuous_artifact()
    contract = _model_rtc_contract(base_artifact)
    if inference_only_v2:
        contract = ModelContinuationDataContractV2(
            conditioning_origin=(
                MODEL_CONTINUATION_ORIGIN_INFERENCE_ONLY_V1
            ),
            runtime=ModelContinuationRuntimeContractV2.from_v1(contract),
        )
    artifact = replace(base_artifact, model_continuation=contract)
    metadata = _server_metadata(artifact)
    runtime = (
        contract.runtime
        if isinstance(contract, ModelContinuationDataContractV2)
        else contract
    )
    provider = {
        "provider_capability_schema": (
            OPENPI_CONDITIONED_PROVIDER_CAPABILITY_SCHEMA_V1
        ),
        "wire_protocol": OPENPI_CONDITIONED_RTC_WIRE_V1,
        "enabled": True,
        "conditioning_transform_owner": "provider_v1",
        "external_action_dim": artifact.policy_io_layout.action_dim,
        "internal_sampler_action_dim": 32,
        "prediction_horizon_steps": HORIZON,
        "sampler_constraint": MODEL_CONTINUATION_SAMPLER_CLAMPED_V1,
        "supports_unconditioned_bootstrap": True,
        "continuation_implementation_id": (
            runtime.continuation_implementation_id
        ),
        "continuation_implementation_parameters_hash": (
            runtime.continuation_implementation_parameters_hash
        ),
    }
    provider["provider_capability_hash"] = (
        openpi_conditioned_provider_capability_hash(provider)
    )
    metadata[OPENPI_CONDITIONED_RTC_METADATA_KEY_V1] = provider
    return artifact, metadata


def _conditioned_base_backend(
    artifact: PolicyArtifactContract,
    metadata: dict[str, Any],
) -> tuple[OpenPIFinalChunkRtcBackend, _FakeConditionedOpenPITransport]:
    transport = _FakeConditionedOpenPITransport(
        metadata=metadata,
        actions=_absolute_actions(artifact.policy_io_layout.action_dim),
    )
    backend = OpenPIFinalChunkRtcBackend(
        transport=transport,
        config=_config(
            artifact,
            result_action_representation=(
                OPENPI_RTC_RESULT_TYPED_CANONICAL_V1
            ),
        ),
        clock=_CounterClock(),
    )
    return backend, transport


def test_openpi_conditioned_provider_capability_hash_golden() -> None:
    _, metadata = _conditioned_artifact_and_metadata(
        inference_only_v2=True,
    )
    provider = metadata[OPENPI_CONDITIONED_RTC_METADATA_KEY_V1]

    assert provider["provider_capability_hash"] == (
        "sha256:91554d87195a5d4f7c8eb5ff1f0c99abb5f84f7fa446faf8e09ea350b346d9c7"
    )


@pytest.mark.parametrize("inference_only_v2", (False, True))
def test_openpi_model_rtc_v3_binds_provider_transform_without_raw_regression(
    inference_only_v2: bool,
) -> None:
    artifact, metadata = _conditioned_artifact_and_metadata(
        inference_only_v2=inference_only_v2,
    )
    base, transport = _conditioned_base_backend(
        artifact,
        metadata,
    )
    contract, canonical_bundle, request = _conditioned_bundle_and_request(
        base,
    )
    assert contract is artifact.model_continuation
    conditioned = OpenPIConditionedRtcBackendV3(
        base_backend=base,
        config=OpenPIConditionedRtcConfig.from_artifact(
            artifact,
            selected_prefix_schedule=(
                RTC_PREFIX_SCHEDULE_EXPLICIT_SELECTED_STEPS_V1
            ),
        ),
        clock=_CounterClock(),
    )
    assert isinstance(conditioned, RtcConditionedModelRuntimeAdapterV3)
    frontend_config = rtc_conditioned_config_from_model_adapter_v3(
        conditioned,
        runtime_session_id="session:openpi-conditioned",
        control_domain_id="domain:openpi-conditioned",
        producer_id="producer:openpi-conditioned",
        policy_node_id="single_policy",
    )
    assert frontend_config.artifact_hash == base.config.artifact_hash
    assert frontend_config.model_continuation is not None
    assert frontend_config.request_trigger_coverage_steps == 3
    assert (
        frontend_config.model_continuation.continuation_contract.contract_hash
        == artifact.model_continuation.contract_hash
    )
    prepared = base.prepare_input(_observation())
    result = conditioned.create_client().infer(
        request,
        canonical_conditioning=canonical_bundle,
        model_input=prepared,
        model_input_hash=prepared.semantic_content_hash,
    )

    assert result.response_validation.accepted
    assert result.encoded_conditioning.external_prefix.shape == (HORIZON, 26)
    assert result.encoded_conditioning.prefix_mask.tolist() == [
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    receipt = result.response.consumption_receipt
    assert (
        receipt.consumed_external_prefix_hash
        != receipt.consumed_internal_prefix_hash
    )
    assert result.policy_chunk_result.effective_delay_steps == 1
    assert len(result.policy_chunk_result.canonical_actions) == HORIZON - 1
    assert transport.requests[-1]["__openpi_request__"]["operation"] == (
        OPENPI_CONDITIONED_RTC_OPERATION_V1
    )


def test_openpi_real_launcher_wires_conditioned_adapter_only_when_selected(
    tmp_path: Path,
) -> None:
    artifact, _ = _conditioned_artifact_and_metadata(
        inference_only_v2=True,
    )
    continuation = artifact.model_continuation
    assert isinstance(continuation, ModelContinuationDataContractV2)
    artifact_path = tmp_path / "policy_artifact.json"
    artifact_path.write_text(
        json.dumps(artifact.to_dict()),
        encoding="utf-8",
    )
    with mock.patch(
        "teleop_runtime.deploy.run_policy.main",
        return_value=0,
    ) as run_main:
        result = openpi_real_launcher.main(
            [
                "--policy-artifact-json",
                str(artifact_path),
                "--rtc-profile",
                MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1,
                "--rtc-frame-contract-hash",
                continuation.runtime.frame_contract_hash,
            ]
        )

    assert result == 0
    delegated = run_main.call_args.args[0]
    assert MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1 in delegated
    factory = run_main.call_args.kwargs["rtc_model_adapter_factory"]
    creator = mock.Mock(return_value="conditioned-adapter")
    adapter = factory(
        policy=SimpleNamespace(create_model_rtc_adapter=creator),
        deploy_session_contract=SimpleNamespace(
            runtime_pose_frame_contract=None,
        ),
        policy_artifact=artifact,
        args=SimpleNamespace(policy_action_pose_mode="absolute"),
        active_sides=("left", "right"),
    )
    assert adapter == "conditioned-adapter"
    assert creator.call_args.kwargs["normalization_hash"] == (
        continuation.runtime.normalization_hash
    )
    assert "result_action_representation" not in creator.call_args.kwargs


def test_openpi_model_rtc_requires_exact_artifact_bound_contract() -> None:
    artifact, metadata = _conditioned_artifact_and_metadata(
        inference_only_v2=True,
    )
    source_contract = artifact.model_continuation
    assert isinstance(source_contract, ModelContinuationDataContractV2)

    unbound_artifact = replace(artifact, model_continuation=None)
    with pytest.raises(ValueError, match="artifact-bound"):
        OpenPIConditionedRtcConfig.from_artifact(
            unbound_artifact,
            selected_prefix_schedule=(
                RTC_PREFIX_SCHEDULE_EXPLICIT_SELECTED_STEPS_V1
            ),
        )
    unbound_base, _ = _conditioned_base_backend(
        unbound_artifact,
        metadata,
    )
    with pytest.raises(ValueError, match="artifact-bound"):
        OpenPIConditionedRtcBackendV3(
            base_backend=unbound_base,
            config=OpenPIConditionedRtcConfig(
                selected_prefix_schedule=(
                    RTC_PREFIX_SCHEDULE_EXPLICIT_SELECTED_STEPS_V1
                ),
            ),
        )

    mismatched = replace(
        source_contract,
        runtime=replace(
            source_contract.runtime,
            continuation_implementation_id="other-implementation-v1",
        ),
    )
    mismatched_artifact = replace(artifact, model_continuation=mismatched)
    mismatched_base, _ = _conditioned_base_backend(
        mismatched_artifact,
        metadata,
    )
    with pytest.raises(ValueError, match="implementation identity mismatch"):
        OpenPIConditionedRtcBackendV3(
            base_backend=mismatched_base,
            config=OpenPIConditionedRtcConfig(
                selected_prefix_schedule=(
                    RTC_PREFIX_SCHEDULE_EXPLICIT_SELECTED_STEPS_V1
                ),
            ),
        )
    assert (
        "continuation_contract"
        not in OpenPIConditionedRtcConfig.__dataclass_fields__
    )


@pytest.mark.parametrize(
    ("tamper_kind", "message"),
    (
        ("declared_hash", "capability hash mismatch"),
        ("implementation", "implementation identity mismatch"),
    ),
)
def test_openpi_model_rtc_rejects_provider_capability_drift(
    tamper_kind: str,
    message: str,
) -> None:
    artifact, metadata = _conditioned_artifact_and_metadata(
        inference_only_v2=True,
    )
    provider = dict(metadata[OPENPI_CONDITIONED_RTC_METADATA_KEY_V1])
    if tamper_kind == "declared_hash":
        provider["provider_capability_hash"] = "sha256:" + "0" * 64
    else:
        provider["continuation_implementation_id"] = "other-provider-v1"
        provider["provider_capability_hash"] = (
            openpi_conditioned_provider_capability_hash(provider)
        )
    metadata[OPENPI_CONDITIONED_RTC_METADATA_KEY_V1] = provider
    base, _ = _conditioned_base_backend(artifact, metadata)

    with pytest.raises(ValueError, match=message):
        OpenPIConditionedRtcBackendV3(
            base_backend=base,
            config=OpenPIConditionedRtcConfig.from_artifact(
                artifact,
                selected_prefix_schedule=(
                    RTC_PREFIX_SCHEDULE_EXPLICIT_SELECTED_STEPS_V1
                ),
            ),
        )


def test_openpi_model_rtc_rejects_call_site_capability_tamper() -> None:
    artifact, metadata = _conditioned_artifact_and_metadata(
        inference_only_v2=True,
    )
    base, transport = _conditioned_base_backend(artifact, metadata)
    conditioned = OpenPIConditionedRtcBackendV3(
        base_backend=base,
        config=OpenPIConditionedRtcConfig.from_artifact(
            artifact,
            selected_prefix_schedule=(
                RTC_PREFIX_SCHEDULE_EXPLICIT_SELECTED_STEPS_V1
            ),
        ),
        clock=_CounterClock(),
    )
    _, canonical_bundle, request = _conditioned_bundle_and_request(base)
    provider = transport._metadata[
        OPENPI_CONDITIONED_RTC_METADATA_KEY_V1
    ]
    provider["provider_capability_hash"] = "sha256:" + "0" * 64
    prepared = base.prepare_input(_observation())

    with pytest.raises(ValueError, match="receipt provider capability"):
        conditioned.create_client().infer(
            request,
            canonical_conditioning=canonical_bundle,
            model_input=prepared,
            model_input_hash=prepared.semantic_content_hash,
        )


def test_openpi_worker_cancel_drains_blocking_transport_and_rearms_session() -> None:
    artifact = _continuous_artifact()
    transport = _CancellableBlockingTransport(
        metadata=_server_metadata(artifact),
        actions=_absolute_actions(artifact.policy_io_layout.action_dim),
    )
    backend = OpenPIFinalChunkRtcBackend(
        transport=transport,
        config=_config(artifact),
        clock=_CounterClock(),
    )
    runtime = PolicyDeployRuntime(
        PolicyDeployRuntimeConfig(
            active_sides=("left", "right"),
            arms_enabled=True,
            hands_enabled=True,
        )
    )
    policy_observation = runtime.prepare_policy_observation(_observation())
    actor_observation, parity = runtime.prepare_actor_observation(policy_observation)
    assert parity is None or parity.matched
    request = _request(backend)
    work_item = PolicyInferenceWorkItem(
        worker_id="worker:openpi-cancel-unit",
        worker_session_id="worker-session:openpi-cancel-unit:0",
        submit_sequence=0,
        request=request,
        actor_observation=actor_observation,
        submitted_at=request.admitted_at,
    )
    worker = BoundedPolicyInferenceWorker(
        worker_id=work_item.worker_id,
        worker_session_id=work_item.worker_session_id,
        executor=backend.create_worker_executor(),
    )
    worker.start()
    assert worker.submit(work_item).accepted
    assert transport.entered.wait(timeout=1.0)

    stop = worker.stop(timeout_sec=1.0)

    assert stop.thread_exited
    assert stop.status == "stopped"
    assert stop.executor_cancellation is not None
    assert stop.executor_cancellation.status == POLICY_EXECUTOR_CANCEL_REQUESTED
    assert stop.executor_cancellation.request_accepted
    assert transport.cancel_count == 1
    assert transport.begin_session_count == 1
    completion = worker.poll_completion()
    assert completion is not None
    assert completion.result.status == "fault"
    assert "synthetic transport session cancelled" in (
        completion.result.fault_detail or ""
    )

    transport.release.set()
    result = backend.create_worker_executor()(request, actor_observation)
    assert result.status == "ready"
    assert transport.begin_session_count == 2
    assert not transport.cancelled.is_set()


def test_openpi_transport_metadata_drift_after_construction_fails_closed() -> None:
    artifact = _continuous_artifact()
    backend, transport = _backend(artifact)
    runtime = PolicyDeployRuntime(
        PolicyDeployRuntimeConfig(
            active_sides=("left", "right"),
            arms_enabled=True,
            hands_enabled=True,
        )
    )
    policy_observation = runtime.prepare_policy_observation(_observation())
    actor_observation, parity = runtime.prepare_actor_observation(policy_observation)
    assert parity is None or parity.matched
    transport._metadata["runtime_build"] = "changed-after-handshake"

    with pytest.raises(ValueError, match="metadata changed across"):
        backend.create_worker_executor()(_request(backend), actor_observation)


def test_openpi_worker_executor_fails_closed_when_transport_rearm_declines() -> None:
    artifact = _continuous_artifact()

    class DecliningSessionTransport(_FakeTransport):
        def begin_session(self) -> bool:
            return False

    transport = DecliningSessionTransport(
        metadata=_server_metadata(artifact),
        actions=_absolute_actions(artifact.policy_io_layout.action_dim),
    )
    backend = OpenPIFinalChunkRtcBackend(
        transport=transport,
        config=_config(artifact),
        clock=_CounterClock(),
    )

    with pytest.raises(RuntimeError, match="declined a new worker session"):
        backend.create_worker_executor()


def test_discrete_tokens_survive_response_and_layout_bound_decoder() -> None:
    artifact = _discrete_artifact()
    actions = _absolute_actions(artifact.policy_io_layout.action_dim)
    tokens = np.asarray([[0], [1], [1], [0]], dtype=np.int64)
    backend, _ = _backend(
        artifact,
        actions=actions,
        discrete_actions=tokens,
    )
    prepared = backend.prepare_input(_observation(discrete=True))
    bundle = backend.create_client().infer(
        _request(backend),
        model_input=prepared,
        model_input_hash=prepared.semantic_content_hash,
    )

    assert bundle.response.model_space_discrete_actions is not None
    np.testing.assert_array_equal(
        bundle.response.model_space_discrete_actions,
        tokens,
    )
    expected = (0.0, 0.9, 0.9, 0.0)
    for value, action in zip(
        expected,
        bundle.policy_chunk_result.decoded_absolute_actions,
        strict=True,
    ):
        assert action.vector[ACTION_HAND_SLICES["right"].start] == pytest.approx(value)
        np.testing.assert_array_equal(
            action.vector[ACTION_HAND_SLICES["right"]][1:],
            np.zeros((5,), dtype=np.float32),
        )


def test_openpi_adapter_advertises_final_chunk_only() -> None:
    backend, _ = _backend(_continuous_artifact())
    capability = backend.rtc_capability
    assert capability.scheduler_modes == (RTC_SCHEDULER_ASYNC_TIMED_PLAN,)
    assert capability.continuation(RTC_CONTINUATION_NONE) is not None
    assert capability.continuation(RTC_CONTINUATION_FLOW_INPAINT) is None
    flow_spec = replace(
        backend.request_spec,
        mode=RtcProtocolMode(
            RTC_SCHEDULER_ASYNC_TIMED_PLAN,
            RTC_CONTINUATION_FLOW_INPAINT,
        ),
        require_prefix_mask=True,
        require_conditioning_receipt=True,
    )
    receipt = negotiate_rtc_capability(flow_spec, capability)
    assert not receipt.accepted
    assert "continuation_mode_unsupported" in receipt.rejection_reasons


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("model_id", None, "model_id"),
        ("checkpoint_step", None, "checkpoint_step"),
        ("camera_input", None, "camera_input"),
    ),
)
def test_artifact_identity_and_input_contract_are_mandatory(
    field: str,
    value: object,
    message: str,
) -> None:
    artifact = replace(_continuous_artifact(), **{field: value})
    with pytest.raises(ValueError, match=message):
        _config(artifact)


def test_relative_pose_artifact_requires_structured_typed_adapter() -> None:
    legacy_metadata_only = replace(
        _continuous_artifact(),
        metadata={"pose_representation": {"action": "relative"}},
    )
    with pytest.raises(ValueError, match="disagrees with policy artifact"):
        _config(legacy_metadata_only)
    with pytest.raises(ValueError, match="structured relative pose"):
        OpenPIFinalChunkRtcConfig(
            artifact=legacy_metadata_only,
            frame_contract_hash="frame:unit",
            normalization_hash="normalization:unit",
            action_pose_mode="relative",
            result_action_representation=OPENPI_RTC_RESULT_TYPED_CANONICAL_V1,
        )

    artifact = _relative_artifact()
    with pytest.raises(ValueError, match="typed_canonical_v1"):
        _config(
            artifact,
            action_pose_mode="relative",
            frame_contract_hash=RELATIVE_FRAME_HASH,
        )
    with pytest.raises(ValueError, match="frame identity"):
        _config(
            artifact,
            action_pose_mode="relative",
            frame_contract_hash="sha256:" + "2" * 64,
            result_action_representation=OPENPI_RTC_RESULT_TYPED_CANONICAL_V1,
        )


def test_arm_hold_artifact_rejects_typed_canonical_raw_rtc() -> None:
    with pytest.raises(ValueError, match="legacy_normalized_v1 absolute"):
        _config(
            _arm_hold_artifact(),
            result_action_representation=OPENPI_RTC_RESULT_TYPED_CANONICAL_V1,
        )


def test_duplicate_active_side_binding_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        replace(_config(_continuous_artifact()), active_sides=("left", "left"))


def test_server_action_horizon_is_mandatory() -> None:
    artifact = _continuous_artifact()
    with pytest.raises(ValueError, match="action_horizon"):
        _backend(artifact, metadata=_server_metadata(artifact, horizon=None))


def test_relative_artifact_and_server_horizon_must_match() -> None:
    artifact = _relative_artifact(horizon=16)
    with pytest.raises(ValueError, match="policy artifact pose action chunk"):
        _backend(
            artifact,
            actions=_absolute_actions(
                artifact.policy_io_layout.action_dim,
                horizon=HORIZON,
            ),
            metadata=_server_metadata(artifact, horizon=HORIZON),
            action_pose_mode="relative",
            frame_contract_hash=RELATIVE_FRAME_HASH,
            result_action_representation=OPENPI_RTC_RESULT_TYPED_CANONICAL_V1,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("layout", "policy_io_layout_hash_mismatch"),
        ("float_dim", "output_action_dim_mismatch"),
        ("token_dim", "output_action_discrete_dim_mismatch"),
    ),
)
def test_server_metadata_drift_fails_before_inference(
    mutation: str,
    message: str,
) -> None:
    artifact = _continuous_artifact()
    metadata = _server_metadata(artifact)
    if mutation == "model":
        metadata["model_id"] = "wrong"
    elif mutation == "layout":
        metadata["dataset_adapter"]["policy_io_layout_hash"] = "wrong"
    elif mutation == "float_dim":
        metadata["local_robot"]["output_action_dim"] += 1
    else:
        metadata["local_robot"]["output_action_discrete_dim"] += 1
    with pytest.raises(ValueError, match=message):
        _backend(artifact, metadata=metadata)


def test_artifact_rtc_contract_never_binds_server_model_identity() -> None:
    artifact = _continuous_artifact()
    metadata = _server_metadata(artifact)
    metadata["model_id"] = "switched-model"
    metadata["deployment"]["checkpoint_step"] = 90000
    contract = deploy_contract_from_policy_artifact(
        artifact,
        protocol_version=OPENPI_LOCAL_PROTOCOL_VERSION,
    )

    result = validate_server_metadata(metadata, contract)
    assert result.server_summary["model_id"] == "switched-model"
    assert contract.model_id is None
    assert contract.checkpoint_step is None


def test_short_final_chunk_is_rejected_against_declared_horizon() -> None:
    artifact = _continuous_artifact()
    backend, _ = _backend(
        artifact,
        actions=_absolute_actions(artifact.policy_io_layout.action_dim, horizon=2),
    )
    prepared = backend.prepare_input(_observation())
    with pytest.raises(ValueError, match="final chunk horizon"):
        backend.create_client().infer(
            _request(backend),
            model_input=prepared,
            model_input_hash=prepared.semantic_content_hash,
        )


def test_nonfinite_model_output_is_rejected() -> None:
    artifact = _continuous_artifact()
    actions = _absolute_actions(artifact.policy_io_layout.action_dim)
    actions[2, 0] = np.nan
    backend, _ = _backend(artifact, actions=actions)
    prepared = backend.prepare_input(_observation())
    with pytest.raises(ValueError, match="non-finite"):
        backend.create_client().infer(
            _request(backend),
            model_input=prepared,
            model_input_hash=prepared.semantic_content_hash,
        )


def test_invalid_discrete_category_is_rejected_by_codec() -> None:
    artifact = _discrete_artifact()
    backend, _ = _backend(
        artifact,
        discrete_actions=np.asarray([[0], [1], [7], [0]], dtype=np.int64),
    )
    prepared = backend.prepare_input(_observation(discrete=True))
    with pytest.raises(ValueError, match="undeclared category"):
        backend.create_client().infer(
            _request(backend),
            model_input=prepared,
            model_input_hash=prepared.semantic_content_hash,
        )


def test_prepared_input_is_detached_and_wrong_hash_fails_closed() -> None:
    backend, transport = _backend(_continuous_artifact())
    observation = _observation()
    prepared = backend.prepare_input(observation)
    before = prepared.semantic_content_hash
    observation.state[:] = 123.0
    assert prepared.semantic_content_hash == before
    with pytest.raises(ValueError, match="input hash"):
        backend.create_client().infer(
            _request(backend),
            model_input=prepared,
            model_input_hash="wrong-input-hash",
        )
    assert transport.requests == []


def test_transport_fault_propagates_without_fallback_result() -> None:
    backend, transport = _backend(
        _continuous_artifact(),
        error=RuntimeError("synthetic transport fault"),
    )
    prepared = backend.prepare_input(_observation())
    with pytest.raises(RuntimeError, match="synthetic transport fault"):
        backend.create_client().infer(
            _request(backend),
            model_input=prepared,
            model_input_hash=prepared.semantic_content_hash,
        )
    assert len(transport.requests) == 1


def test_clock_regression_is_rejected() -> None:
    backend, _ = _backend(
        _continuous_artifact(),
        clock=_SequenceClock((10.0, 9.9, 10.1)),
    )
    prepared = backend.prepare_input(_observation())
    with pytest.raises(ValueError, match="clock regressed"):
        backend.create_client().infer(
            _request(backend),
            model_input=prepared,
            model_input_hash=prepared.semantic_content_hash,
        )


def test_reset_is_forwarded_but_does_not_change_capability() -> None:
    backend, transport = _backend(_continuous_artifact())
    before = backend.rtc_capability.semantic_content_hash
    backend.reset()
    assert transport.reset_count == 1
    assert backend.rtc_capability.semantic_content_hash == before


def test_artifact_and_server_metadata_change_capability_identity() -> None:
    artifact = _continuous_artifact()
    first, _ = _backend(artifact)
    changed_artifact = replace(artifact, metadata={"fixture": "changed"})
    second, _ = _backend(changed_artifact)
    metadata = _server_metadata(artifact)
    metadata["runtime_build"] = "other"
    third, _ = _backend(artifact, metadata=metadata)

    assert openpi_rtc_artifact_hash(artifact) != openpi_rtc_artifact_hash(
        changed_artifact
    )
    assert (
        first.rtc_capability.semantic_content_hash
        != second.rtc_capability.semantic_content_hash
    )
    assert (
        first.rtc_capability.semantic_content_hash
        != third.rtc_capability.semantic_content_hash
    )


def test_repeated_offline_runs_are_numerically_replayable() -> None:
    artifact = _continuous_artifact()
    bundles = []
    for _ in range(2):
        backend, _transport = _backend(artifact, clock=_CounterClock())
        prepared = backend.prepare_input(_observation())
        bundles.append(
            backend.create_client().infer(
                _request(backend),
                model_input=prepared,
                model_input_hash=prepared.semantic_content_hash,
            )
        )
    first, second = bundles
    assert first.envelope.semantic_content_hash == second.envelope.semantic_content_hash
    assert first.response.semantic_content_hash == second.response.semantic_content_hash
    assert (
        first.policy_chunk_result.semantic_content_hash
        == second.policy_chunk_result.semantic_content_hash
    )
    for left, right in zip(
        first.policy_chunk_result.decoded_absolute_actions,
        second.policy_chunk_result.decoded_absolute_actions,
        strict=True,
    ):
        np.testing.assert_array_equal(left.vector, right.vector)


def test_decoded_arm_pose_slots_remain_in_canonical_order() -> None:
    artifact = _continuous_artifact()
    actions = _absolute_actions(ACTION_DIM)
    actions[1, ACTION_ARM_POSE_SLICES["left"]] = np.asarray(
        [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
        dtype=np.float32,
    )
    backend, _ = _backend(artifact, actions=actions)
    prepared = backend.prepare_input(_observation())
    bundle = backend.create_client().infer(
        _request(backend),
        model_input=prepared,
        model_input_hash=prepared.semantic_content_hash,
    )
    np.testing.assert_array_equal(
        bundle.policy_chunk_result.decoded_absolute_actions[1].vector[
            ACTION_ARM_POSE_SLICES["left"]
        ],
        actions[1, ACTION_ARM_POSE_SLICES["left"]],
    )


@pytest.mark.parametrize(
    ("horizon", "clock_values"),
    (
        (2, (10.0, 10.0, 10.0)),
        (4, (10.0, 10.2, 10.21)),
        (8, (10.0, 10.75, 10.9)),
    ),
)
def test_fake_policy_horizon_and_latency_sweep_is_stable(
    horizon: int,
    clock_values: tuple[float, float, float],
) -> None:
    artifact = _continuous_artifact()
    actions = _absolute_actions(artifact.policy_io_layout.action_dim, horizon=horizon)
    metadata = _server_metadata(artifact, horizon=horizon)
    backend, _ = _backend(
        artifact,
        actions=actions,
        metadata=metadata,
        clock=_SequenceClock(clock_values),
    )
    prepared = backend.prepare_input(_observation())
    bundle = backend.create_client().infer(
        _request(backend),
        model_input=prepared,
        model_input_hash=prepared.semantic_content_hash,
    )
    assert bundle.response.model_space_actions.shape == (
        horizon,
        artifact.policy_io_layout.action_dim,
    )
    assert bundle.response.execution_horizon_steps == horizon
    assert bundle.response.inference_started_at == clock_values[0]
    assert bundle.response.inference_completed_at == clock_values[1]
    assert bundle.response.result_received_at == clock_values[2]
    for index, action in enumerate(bundle.policy_chunk_result.decoded_absolute_actions):
        np.testing.assert_array_equal(action.vector, actions[index])


def test_fake_transport_output_is_detached_after_result_creation() -> None:
    artifact = _continuous_artifact()
    backend, transport = _backend(artifact)
    prepared = backend.prepare_input(_observation())
    bundle = backend.create_client().infer(
        _request(backend),
        model_input=prepared,
        model_input_hash=prepared.semantic_content_hash,
    )
    response_hash = bundle.response.semantic_content_hash
    result_hash = bundle.policy_chunk_result.semantic_content_hash
    first_value = float(bundle.response.model_space_actions[0, 0])

    transport.actions[:] = 77.0

    assert float(bundle.response.model_space_actions[0, 0]) == first_value
    assert bundle.response.semantic_content_hash == response_hash
    assert bundle.policy_chunk_result.semantic_content_hash == result_hash


def test_prepared_fake_policy_payload_arrays_are_read_only() -> None:
    backend, _ = _backend(_continuous_artifact())
    prepared = backend.prepare_input(_observation())
    state = prepared.payload["robot_state"]
    assert isinstance(state, np.ndarray)
    assert not state.flags.writeable
    with pytest.raises(ValueError):
        state[0] = 1.0
