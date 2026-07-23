from __future__ import annotations

import ast
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dataset_contracts.canonical_encoding import canonical_content_hash
from deployments.client import (
    RtcModelRuntimeAdapter,
    SimpleRtcModelRuntimeAdapter,
    build_rtc_model_shadow_frontend,
    rtc_shadow_config_from_model_adapter,
)
from deployments.client.rtc_executor import RtcPreparedModelInput
from robot_motion.action import RobotAction
from robot_motion.deploy import (
    ACTION_ARM_POSE_SLICES,
    DEPLOY_ACTION_DIM,
    POLICY_ASYNC_PRODUCER_FAULT,
    RTC_CONTINUATION_NONE,
    RTC_PREFIX_FORBIDDEN,
    RTC_PROTOCOL_V2,
    RTC_SCHEDULER_ASYNC_TIMED_PLAN,
    ActionSafetyFilter,
    ActionSafetyFilterConfig,
    CanonicalDeployActionPipeline,
    NormalizedAction,
    PolicyDeployRuntime,
    PolicyDeployRuntimeConfig,
    RAW_RTC_FINAL_CHUNK_PROFILE_V1,
    RelativePolicyActionChunkCanonicalizer,
    RtcContinuationCapability,
    RtcModelCapability,
    RtcModelResponseEnvelopeV2,
    RtcPolicyShadowFrontend,
    RtcProtocolMode,
    RtcProtocolRequestSpec,
    build_policy_action_representation_context,
    build_deployment_observation,
    nan_vector,
)
from robot_motion.deploy.inference import ASYNC_ALIGNED_CHUNK_MODE_V1
from robot_motion.deploy.policy_worker import PolicyInferenceWorkItem
from robot_motion.deploy.rtc_runtime import RtcFrontendStep


MODEL_DIM = 7


def _capability(
    action_pose_mode: str = "absolute",
) -> RtcModelCapability:
    continuation = RtcContinuationCapability(
        continuation_mode=RTC_CONTINUATION_NONE,
        implementation_id="generic-final-chunk-v1",
        implementation_parameters_hash="parameters:generic-final-chunk-v1",
        model_action_space="generic-right-arm-pose-v1",
        model_action_dim=MODEL_DIM,
        model_action_discrete_dim=0,
        supported_action_pose_modes=(action_pose_mode,),
        component_scope=("right_arm",),
        action_codec_hash="codec:generic-right-arm-pose-v1",
        frame_contract_hash="frame:runtime-raw-v1",
        layout_hash="layout:generic-right-arm-pose-v1",
        normalization_hash="normalization:none-v1",
        minimum_prediction_horizon_steps=1,
        maximum_prediction_horizon_steps=16,
        minimum_execution_horizon_steps=1,
        maximum_execution_horizon_steps=16,
        prefix_requirement=RTC_PREFIX_FORBIDDEN,
        supports_prefix_mask=False,
        supports_conditioning_receipt=False,
    )
    return RtcModelCapability(
        capability_id="capability:generic-final-chunk-v1",
        provider_id="provider:generic-final-chunk-v1",
        artifact_hash="artifact:generic-final-chunk-v1",
        scheduler_modes=(RTC_SCHEDULER_ASYNC_TIMED_PLAN,),
        continuations=(continuation,),
    )


def _request_spec(
    horizon: int,
    *,
    action_pose_mode: str = "absolute",
) -> RtcProtocolRequestSpec:
    return RtcProtocolRequestSpec(
        mode=RtcProtocolMode(
            RTC_SCHEDULER_ASYNC_TIMED_PLAN,
            RTC_CONTINUATION_NONE,
        ),
        artifact_hash="artifact:generic-final-chunk-v1",
        model_action_space="generic-right-arm-pose-v1",
        model_action_dim=MODEL_DIM,
        model_action_discrete_dim=0,
        action_pose_mode=action_pose_mode,
        component_scope=("right_arm",),
        action_codec_hash="codec:generic-right-arm-pose-v1",
        frame_contract_hash="frame:runtime-raw-v1",
        layout_hash="layout:generic-right-arm-pose-v1",
        normalization_hash="normalization:none-v1",
        prediction_horizon_steps=horizon,
        minimum_execution_horizon_steps=1,
    )


class _GenericFinalChunkBackend:
    def __init__(
        self,
        *,
        horizon: int,
        block_until: threading.Event | None = None,
        entered: threading.Event | None = None,
        action_pose_mode: str = "absolute",
        actions: np.ndarray | None = None,
    ) -> None:
        self.horizon = int(horizon)
        self.action_pose_mode = str(action_pose_mode)
        self._capability = _capability(self.action_pose_mode)
        self.block_until = block_until
        self.entered = entered
        self.actions = None if actions is None else np.asarray(actions, dtype=np.float32)
        self.call_thread_ids: list[int] = []

    @property
    def rtc_capability(self) -> RtcModelCapability:
        return self._capability

    def infer_rtc(self, envelope, model_input: Any) -> RtcModelResponseEnvelopeV2:
        del model_input
        self.call_thread_ids.append(threading.get_ident())
        if self.entered is not None:
            self.entered.set()
        if self.block_until is not None:
            self.block_until.wait(timeout=2.0)
        started = max(time.monotonic(), envelope.request.admitted_at)
        if self.actions is None:
            actions = np.zeros((self.horizon, MODEL_DIM), dtype=np.float32)
            actions[:, :3] = np.asarray([0.10, 0.20, 0.30], dtype=np.float32)
            actions[:, 6] = 1.0
            for index in range(self.horizon):
                actions[index, 0] += np.float32(0.001 * index)
        else:
            actions = self.actions.copy()
        completed = max(time.monotonic(), started)
        return RtcModelResponseEnvelopeV2(
            envelope_hash=envelope.semantic_content_hash,
            request_id=envelope.request.request_id,
            request_hash=envelope.request.semantic_content_hash,
            model_input_hash=envelope.model_input_hash,
            capability_hash=self._capability.semantic_content_hash,
            selected_continuation_hash=envelope.selected_continuation_hash,
            protocol_version=RTC_PROTOCOL_V2,
            scheduler_mode=RTC_SCHEDULER_ASYNC_TIMED_PLAN,
            continuation_mode=RTC_CONTINUATION_NONE,
            artifact_hash=envelope.request.artifact_hash,
            action_codec_hash=envelope.request.action_codec_hash,
            frame_contract_hash=envelope.request.frame_contract_hash,
            layout_hash=envelope.request.layout_hash,
            normalization_hash=envelope.request.normalization_hash,
            model_output_id=f"output:{envelope.request.request_id}",
            model_space_actions=actions,
            effective_delay_steps=envelope.request.forecast_delay_steps,
            execution_horizon_steps=self.horizon,
            inference_started_at=started,
            inference_completed_at=completed,
            result_received_at=completed,
            conditioning_receipt=None,
        )


def _runtime_adapter(
    *,
    backend: _GenericFinalChunkBackend,
    horizon: int,
    request_spec: RtcProtocolRequestSpec | None = None,
) -> SimpleRtcModelRuntimeAdapter:
    return SimpleRtcModelRuntimeAdapter(
        backend=backend,
        decoder=_decoder,
        request_spec=(
            _request_spec(
                horizon,
                action_pose_mode=backend.action_pose_mode,
            )
            if request_spec is None
            else request_spec
        ),
        prepare_input=_prepare_input,
    )


def _decoder(model_actions, discrete_actions, request):
    del request
    assert discrete_actions is None
    decoded = []
    for row in model_actions:
        vector = nan_vector(DEPLOY_ACTION_DIM)
        vector[ACTION_ARM_POSE_SLICES["right"]] = row
        decoded.append(
            NormalizedAction(
                vector=vector,
                metadata={
                    "pose_action_source_mode": "absolute",
                    "pose_action_mode": "absolute",
                },
            )
        )
    return tuple(decoded)


def _prepare_input(request, actor_observation) -> RtcPreparedModelInput:
    payload = actor_observation.model_input()
    return RtcPreparedModelInput(
        model_input=payload,
        model_input_hash=canonical_content_hash(
            {
                "request_id": request.request_id,
                "observation": payload,
            },
            domain="dexteleop.test.generic_rtc_model_input.v1",
        ),
    )


def _observation(*, timestamp: float, step_index: int):
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [0.10, 0.20, 0.30]
    return build_deployment_observation(
        active_sides=("right",),
        arms_enabled=True,
        hands_enabled=False,
        step_index=step_index,
        arm_states={
            "right": {
                "current_qpos": np.zeros(6, dtype=np.float32),
                "current_tcp_pose": pose,
                "actual_tcp_pose": pose,
            }
        },
        hand_states={},
        timestamp=timestamp,
    )


def _runtime() -> PolicyDeployRuntime:
    return PolicyDeployRuntime(
        PolicyDeployRuntimeConfig(
            active_sides=("right",),
            arms_enabled=True,
            hands_enabled=False,
            action_keyframe_period_sec=1.0 / 30.0,
            action_maximum_hold_sec=1.0 / 30.0,
        )
    )


def _pipeline() -> CanonicalDeployActionPipeline:
    return CanonicalDeployActionPipeline(
        safety_filter=ActionSafetyFilter(
            ActionSafetyFilterConfig(
                active_sides=("right",),
                arms_enabled=True,
                hands_enabled=False,
                max_arm_xyz_step_m=0.05,
                max_arm_rot_step_deg=20.0,
                max_arm_joint_step_rad=0.2,
            )
        ),
        active_sides=("right",),
        arms_enabled=True,
        hands_enabled=False,
    )


def _frontend(
    *,
    horizon: int,
    backend: _GenericFinalChunkBackend,
    created_at: float,
    profile=None,
) -> tuple[RtcPolicyShadowFrontend, PolicyDeployRuntime]:
    runtime = _runtime()
    return (
        build_rtc_model_shadow_frontend(
            _runtime_adapter(backend=backend, horizon=horizon),
            runtime_session_id=f"session:generic-rtc:{horizon}",
            control_domain_id="domain:generic-rtc",
            producer_id="producer:generic-rtc",
            policy_node_id="single_policy",
            deploy_runtime=runtime,
            preparer=_pipeline(),
            created_at=created_at,
            policy_name="generic-final-chunk",
            request_validity_steps=max(10, horizon + 2),
            request_trigger_coverage_steps=1,
            **({} if profile is None else {"profile": profile}),
        ),
        runtime,
    )


def test_raw_rtc_profile_uses_the_same_model_neutral_frontend() -> None:
    base = time.monotonic()
    backend = _GenericFinalChunkBackend(horizon=4)
    frontend, runtime = _frontend(
        horizon=4,
        backend=backend,
        created_at=base,
        profile=RAW_RTC_FINAL_CHUNK_PROFILE_V1,
    )
    assert frontend.config.profile.profile_id == RAW_RTC_FINAL_CHUNK_PROFILE_V1.profile_id
    frontend.start()
    try:
        for index in range(60):
            frontend.step(
                _observation(timestamp=base + index / 30.0, step_index=index),
                now=base + index / 30.0,
            )
    finally:
        frontend.close(now=base + 3.0)
    assert runtime.config.action_keyframe_period_sec == 1.0 / 30.0


def test_model_runtime_adapter_projects_exact_request_identity() -> None:
    backend = _GenericFinalChunkBackend(horizon=8)
    adapter = _runtime_adapter(
        backend=backend,
        horizon=8,
    )
    assert isinstance(adapter, RtcModelRuntimeAdapter)
    config = rtc_shadow_config_from_model_adapter(
        adapter,
        runtime_session_id="session:projection",
        control_domain_id="domain:projection",
        producer_id="producer:projection",
        policy_node_id="node:projection",
    )
    spec = adapter.request_spec

    assert config.artifact_hash == spec.artifact_hash
    assert config.action_codec_hash == spec.action_codec_hash
    assert config.frame_contract_hash == spec.frame_contract_hash
    assert config.layout_hash == spec.layout_hash
    assert config.normalization_hash == spec.normalization_hash
    assert config.prediction_horizon_steps == spec.prediction_horizon_steps
    assert (
        config.minimum_execution_horizon_steps
        == spec.minimum_execution_horizon_steps
    )
    assert config.action_pose_mode == spec.action_pose_mode
    assert backend.call_thread_ids == []


def test_rtc_observation_cut_separates_causal_identity_from_model_input_content() -> None:
    base = time.monotonic()
    backend = _GenericFinalChunkBackend(horizon=4)
    frontend, runtime = _frontend(
        horizon=4,
        backend=backend,
        created_at=base,
    )
    observation = _observation(timestamp=base, step_index=7)
    first_observation = replace(
        observation,
        images={"front": np.zeros((8, 8, 3), dtype=np.uint8)},
        image_timestamps={"front": base},
        image_ages_sec={"front": 0.0},
    )
    second_observation = replace(
        observation,
        images={"front": np.ones((8, 8, 3), dtype=np.uint8)},
        image_timestamps={"front": base},
        image_ages_sec={"front": 0.0},
    )
    first_actor, first_parity = runtime.prepare_actor_observation(
        runtime.prepare_policy_observation(first_observation)
    )
    second_actor, second_parity = runtime.prepare_actor_observation(
        runtime.prepare_policy_observation(second_observation)
    )
    assert first_parity.matched
    assert second_parity.matched

    first_cut = frontend._observation_cut_id(
        first_actor,
        request_sequence=7,
    )
    second_cut = frontend._observation_cut_id(
        second_actor,
        request_sequence=7,
    )
    next_cut = frontend._observation_cut_id(
        first_actor,
        request_sequence=8,
    )

    assert first_cut == second_cut
    assert first_cut != next_cut
    assert first_cut.startswith("rtc-observation-cut:v2:")

    request = frontend._build_request(first_actor, now=base)
    assert (
        _prepare_input(request, first_actor).model_input_hash
        != _prepare_input(request, second_actor).model_input_hash
    )
    first_work_item = PolicyInferenceWorkItem(
        worker_id="worker:identity-test",
        worker_session_id="worker-session:identity-test",
        submit_sequence=0,
        request=request,
        actor_observation=first_actor,
        submitted_at=base,
    )
    second_work_item = replace(
        first_work_item,
        actor_observation=second_actor,
    )
    assert first_work_item.actor_observation_hash == (
        second_work_item.actor_observation_hash
    )
    assert first_work_item.semantic_content_hash == (
        second_work_item.semantic_content_hash
    )


def test_rtc_request_cut_is_unique_for_repeated_same_observation() -> None:
    base = time.monotonic()
    backend = _GenericFinalChunkBackend(horizon=4)
    frontend, runtime = _frontend(
        horizon=4,
        backend=backend,
        created_at=base,
    )
    actor, parity = runtime.prepare_actor_observation(
        runtime.prepare_policy_observation(
            _observation(timestamp=base, step_index=0)
        )
    )
    assert parity.matched

    first = frontend._build_request(actor, now=base)
    second = frontend._build_request(actor, now=base)

    assert first.request_id != second.request_id
    assert first.observation_cut_id != second.observation_cut_id
    assert first.observation_timestamp == second.observation_timestamp
    assert first.observation_step_index == second.observation_step_index


def test_relative_frontend_freezes_request_anchor_and_emits_typed_absolute() -> None:
    horizon = 4
    relative = np.zeros((horizon, MODEL_DIM), dtype=np.float32)
    relative[:, 6] = 1.0
    relative[:, 0] = np.asarray([0.01, 0.02, 0.03, 0.04], dtype=np.float32)
    backend = _GenericFinalChunkBackend(
        horizon=horizon,
        action_pose_mode="relative",
        actions=relative,
    )
    observed_requests = []

    def decode_absolute_rows(model_actions, discrete_actions, request):
        assert discrete_actions is None
        return tuple(
            RobotAction(
                action_space="generic_right_arm_pose_v1",
                timestamp=request.observation_timestamp,
                arm_pose_by_side={"right": row},
                frame_by_effector={"right_arm": "runtime_raw"},
                controlled_frame_by_effector={"right_arm": "tool_tcp"},
                source="generic-relative-final-chunk",
            )
            for row in model_actions
        )

    def prepare_input(request, actor_observation):
        observed_requests.append(request)
        return _prepare_input(request, actor_observation)

    adapter = SimpleRtcModelRuntimeAdapter(
        backend=backend,
        decoder=RelativePolicyActionChunkCanonicalizer(
            decode_absolute_robot_actions=decode_absolute_rows,
            conversion_contract_hash="conversion:generic-relative-fixed-anchor-v1",
            arm_pose_slices={"right": slice(0, 7)},
        ),
        request_spec=_request_spec(horizon, action_pose_mode="relative"),
        prepare_input=prepare_input,
        representation_context_builder=lambda observation, cut_id: (
            build_policy_action_representation_context(
                observation,
                cut_id,
                source_pose_mode="relative",
                active_sides=("right",),
                arm_reference_frame_by_side={"right": "runtime_raw"},
                arm_controlled_frame_by_side={"right": "tool_tcp"},
            )
        ),
    )
    base = time.monotonic()
    runtime = _runtime()
    frontend = build_rtc_model_shadow_frontend(
        adapter,
        runtime_session_id="session:generic-relative",
        control_domain_id="domain:generic-relative",
        producer_id="producer:generic-relative",
        policy_node_id="single_policy",
        deploy_runtime=runtime,
        preparer=object(),
        created_at=base,
        request_trigger_coverage_steps=1,
    )

    frontend.start()
    emitting_step = None
    try:
        for index in range(120):
            now = base + index / 30.0
            step = frontend.step(
                _observation(timestamp=now, step_index=index),
                now=now,
            )
            if step.action_preparation.emission is not None:
                emitting_step = step
                break
            time.sleep(0.001)
    finally:
        frontend.close(now=base + 5.0)

    assert observed_requests
    request = observed_requests[0]
    context = request.action_representation_context
    assert context is not None
    assert context.source_pose_mode == "relative"
    assert context.observation_cut_id == request.observation_cut_id
    np.testing.assert_allclose(
        context.arm_anchor_pose_by_side["right"][:3],
        np.asarray([0.10, 0.20, 0.30], dtype=np.float32),
    )
    assert emitting_step is not None
    emission = emitting_step.action_preparation.emission
    assert emission is not None
    assert emission.canonical_action is not None
    receipt = emission.canonical_action.canonicalization_receipt
    assert receipt.source_pose_mode == "relative"
    assert receipt.anchor_identity_hash == context.semantic_content_hash
    assert receipt.source_index is not None
    np.testing.assert_allclose(
        emission.canonical_action.action.arm_pose_by_side["right"][:3],
        np.asarray(
            [0.10 + relative[receipt.source_index, 0], 0.20, 0.30],
            dtype=np.float32,
        ),
        atol=1e-6,
    )
    assert runtime.action_queue_pop_count() == 0


def test_relative_frontend_rejects_missing_context_builder_before_worker() -> None:
    backend = _GenericFinalChunkBackend(
        horizon=4,
        action_pose_mode="relative",
    )
    adapter = SimpleRtcModelRuntimeAdapter(
        backend=backend,
        decoder=_decoder,
        request_spec=_request_spec(4, action_pose_mode="relative"),
        prepare_input=_prepare_input,
    )

    with pytest.raises(ValueError, match="representation_context_builder"):
        build_rtc_model_shadow_frontend(
            adapter,
            runtime_session_id="session:relative-missing-context",
            control_domain_id="domain:relative-missing-context",
            producer_id="producer:relative-missing-context",
            policy_node_id="single_policy",
            deploy_runtime=_runtime(),
            preparer=object(),
            created_at=time.monotonic(),
        )
    assert backend.call_thread_ids == []


def test_relative_frontend_faults_on_context_from_wrong_observation_contract() -> None:
    backend = _GenericFinalChunkBackend(
        horizon=4,
        action_pose_mode="relative",
    )

    def wrong_context(observation, cut_id):
        return replace(
            build_policy_action_representation_context(
                observation,
                cut_id,
                source_pose_mode="relative",
                active_sides=("right",),
                arm_reference_frame_by_side={"right": "runtime_raw"},
                arm_controlled_frame_by_side={"right": "tool_tcp"},
            ),
            observation_contract_hash="observation-contract:wrong",
        )

    adapter = SimpleRtcModelRuntimeAdapter(
        backend=backend,
        decoder=_decoder,
        request_spec=_request_spec(4, action_pose_mode="relative"),
        prepare_input=_prepare_input,
        representation_context_builder=wrong_context,
    )
    base = time.monotonic()
    frontend = build_rtc_model_shadow_frontend(
        adapter,
        runtime_session_id="session:relative-wrong-context",
        control_domain_id="domain:relative-wrong-context",
        producer_id="producer:relative-wrong-context",
        policy_node_id="single_policy",
        deploy_runtime=_runtime(),
        preparer=object(),
        created_at=base,
    )

    frontend.start()
    try:
        step = frontend.step(
            _observation(timestamp=base, step_index=0),
            now=base,
        )
    finally:
        frontend.close(now=base + 1.0)

    assert step.state.lifecycle_state == POLICY_ASYNC_PRODUCER_FAULT
    assert step.lifecycle_transition is not None
    assert backend.call_thread_ids == []


def test_model_runtime_adapter_fails_before_worker_on_capability_drift() -> None:
    backend = _GenericFinalChunkBackend(horizon=4)
    original_spec = _request_spec(4)
    adapter = _runtime_adapter(
        backend=backend,
        horizon=4,
        request_spec=replace(
            original_spec,
            artifact_hash="artifact:mismatched",
        ),
    )

    with pytest.raises(RuntimeError, match="capability negotiation"):
        rtc_shadow_config_from_model_adapter(
            adapter,
            runtime_session_id="session:drift",
            control_domain_id="domain:drift",
            producer_id="producer:drift",
            policy_node_id="node:drift",
        )
    assert backend.call_thread_ids == []


def test_model_frontend_builder_rejects_runtime_cadence_mismatch() -> None:
    backend = _GenericFinalChunkBackend(horizon=4)
    adapter = _runtime_adapter(
        backend=backend,
        horizon=4,
    )
    runtime = PolicyDeployRuntime(
        PolicyDeployRuntimeConfig(
            active_sides=("right",),
            arms_enabled=True,
            hands_enabled=False,
            action_keyframe_period_sec=1.0 / 20.0,
            action_maximum_hold_sec=1.0 / 20.0,
        )
    )

    with pytest.raises(ValueError, match="action profile period mismatch"):
        build_rtc_model_shadow_frontend(
            adapter,
            runtime_session_id="session:cadence",
            control_domain_id="domain:cadence",
            producer_id="producer:cadence",
            policy_node_id="node:cadence",
            deploy_runtime=runtime,
            preparer=_pipeline(),
            created_at=time.monotonic(),
        )
    assert backend.call_thread_ids == []


@pytest.mark.parametrize("horizon", (2, 4, 8))
def test_model_neutral_client_worker_and_timed_plan_round_trip(horizon: int) -> None:
    base = time.monotonic()
    backend = _GenericFinalChunkBackend(horizon=horizon)
    frontend, runtime = _frontend(horizon=horizon, backend=backend, created_at=base)
    frontend.start()
    emitting_step: RtcFrontendStep | None = None
    try:
        for index in range(120):
            now = base + index / 30.0
            step = frontend.step(
                _observation(timestamp=now, step_index=index),
                now=now,
            )
            if step.action_preparation.emission is not None:
                emitting_step = step
                break
            time.sleep(0.001)
    finally:
        frontend.close(now=base + 5.0)

    assert emitting_step is not None
    assert emitting_step.action_preparation.raw_lookahead_action is None
    assert emitting_step.emission_commit_transition is not None
    assert emitting_step.emission_commit_transition.receipt.accepted
    assert runtime.action_queue_pop_count() == 0
    assert backend.call_thread_ids
    assert backend.call_thread_ids[0] != threading.get_ident()


def test_blocking_model_backend_does_not_block_rtc_control_step() -> None:
    base = time.monotonic()
    release = threading.Event()
    entered = threading.Event()
    backend = _GenericFinalChunkBackend(
        horizon=4,
        block_until=release,
        entered=entered,
    )
    frontend, runtime = _frontend(horizon=4, backend=backend, created_at=base)
    frontend.start()
    try:
        frontend.step(_observation(timestamp=base, step_index=0), now=base)
        assert entered.wait(timeout=1.0)
        started = time.perf_counter()
        step = frontend.step(
            _observation(timestamp=base + 0.001, step_index=1),
            now=base + 0.001,
        )
        assert time.perf_counter() - started < 0.05
        assert step.action_preparation.emission is None

        transition = frontend.hold(now=base + 0.002, reason="test_hold")
        assert transition.worker_stop_receipt.status == "executing_call_not_drained"
        release.set()
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and frontend.pending_completion_count == 0:
            time.sleep(0.001)
        late = frontend.step(
            _observation(timestamp=base + 0.003, step_index=2),
            now=base + 0.003,
            allow_output=False,
        )
        assert late.result_transition is not None
        assert not late.result_transition.receipt.accepted
        assert runtime.action_queue_pop_count() == 0
    finally:
        release.set()
        frontend.close(now=base + 5.0)


def test_rtc_core_ownership_has_no_upper_layer_or_target_imports() -> None:
    project_root = Path(__file__).resolve().parents[1]
    paths = (
        project_root / "robot_motion/deploy/policy_worker.py",
        project_root / "robot_motion/deploy/rtc_runtime.py",
        project_root / "deployments/client/rtc.py",
        project_root / "deployments/client/rtc_executor.py",
        project_root / "deployments/client/rtc_frontend.py",
    )
    forbidden = (
        "control_station",
        "teleop_runtime",
        "sim",
        "hardware",
        "deployments.openpi",
    )
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert not any(
            value == prefix or value.startswith(prefix + ".")
            for value in imports
            for prefix in forbidden
        ), (path, imports)


def test_teleop_rtc_modules_are_import_identity_compatibility_shims() -> None:
    from robot_motion.deploy.policy_worker import BoundedPolicyInferenceWorker
    from robot_motion.deploy.rtc_runtime import RtcPolicyShadowFrontend as SharedFrontend
    from teleop_runtime.deploy.policy_worker import (
        BoundedPolicyInferenceWorker as LegacyWorker,
    )
    from teleop_runtime.deploy.rtc_frontend import (
        RtcPolicyShadowFrontend as LegacyFrontend,
    )

    assert LegacyWorker is BoundedPolicyInferenceWorker
    assert LegacyFrontend is SharedFrontend
