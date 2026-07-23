from __future__ import annotations

from dataclasses import replace
import threading
import time
from typing import Any

import numpy as np
import pytest

from dataset_contracts.pose_chunk import PoseActionChunkContract
from dataset_contracts.pose_training import PoseTrainingDataContract
from dataset_contracts.policy_artifact import (
    PolicyArtifactContract,
    PolicyCameraInputContract,
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
    RtcModelRuntimeAdapter,
    build_rtc_model_shadow_frontend,
    rtc_client_inference_bundle_debug_payload,
    rtc_shadow_config_from_model_adapter,
)
from deployments.client.artifact import deploy_contract_from_policy_artifact
from deployments.openpi.client.policy import (
    OPENPI_LOCAL_PROTOCOL_VERSION,
    OpenPIClientPolicy,
    OpenPIClientPolicyConfig,
)
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
from robot_motion.deploy import (
    ActionSafetyFilter,
    ActionSafetyFilterConfig,
    CanonicalDeployActionPipeline,
    PolicyDeployRuntime,
    PolicyDeployRuntimeConfig,
    RAW_RTC_FINAL_CHUNK_PROFILE_V1,
)
from robot_motion.deploy.inference import (
    ASYNC_ALIGNED_CHUNK_MODE_V1,
    PolicyInferenceRequest,
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
        error: Exception | None = None,
    ) -> None:
        self._metadata = dict(metadata)
        self.actions = np.asarray(actions)
        self.discrete_actions = (
            None if discrete_actions is None else np.asarray(discrete_actions)
        )
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
        ("model", "model_id_mismatch"),
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
