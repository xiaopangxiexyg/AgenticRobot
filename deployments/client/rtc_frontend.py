"""Model-family-neutral composition for the default-off RTC shadow frontend.

Concrete model packages own transport, model input preparation and model-space
decoding.  This module only projects an already negotiated request spec into
the shared worker/timed-plan coordinator without copying identity fields in
every framework adapter.  It creates no observation source, target backend,
safety authority, dispatcher or production caller.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from dataset_contracts.model_continuation import (
    ModelContinuationContract,
    normalize_model_continuation_contract,
)
from dataset_contracts.policy_action_semantics import (
    PolicyArmHoldOutputContract,
)
from robot_motion.deploy.async_producer import (
    MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_V1,
    RTC_3C_VIRTUAL_SHADOW_PROFILE_V1,
    PolicyAsyncExecutionProfile,
)
from robot_motion.deploy.action_representation import (
    PolicyActionRepresentationContext,
)
from robot_motion.deploy.inference import InferenceDelayEstimatorSpec
from robot_motion.deploy.observation_view import PolicyObservationView
from robot_motion.deploy.policy_worker import (
    PolicyInferenceExecutor,
    PolicyInferenceWorkItemExecutor,
)
from robot_motion.deploy.rtc_protocol import (
    RtcModelCapability,
    RtcProtocolRequestSpec,
    negotiate_rtc_capability,
)
from robot_motion.deploy.rtc_runtime import (
    RtcPolicyShadowConfig,
    RtcPolicyShadowFrontend,
)
from robot_motion.deploy.rtc import RtcModelContinuationRuntimeSpec
from robot_motion.deploy.runtime import PolicyDeployRuntime

from .rtc import (
    ModelAgnosticRtcClient,
    RtcClientInferenceBundle,
    RtcClientProtocolError,
    RtcModelBackend,
    RtcModelSpaceDecoder,
)
from .rtc_executor import (
    RtcModelClientInferenceExecutor,
    RtcModelInputPreparer,
)
from .rtc_conditioning_protocol import (
    RtcModelContinuationCapabilityV3,
    validate_rtc_model_continuation_capability_v3_binding,
)


def _nonempty(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


@runtime_checkable
class RtcModelRuntimeAdapter(Protocol):
    """Small composition surface implemented by any model-family package.

    The adapter may use a local runtime, HTTP, websocket, IPC or another
    transport.  It still owns no worker thread or timed-plan authority: each
    call must return a blocking executor for the caller-owned bounded worker.
    """

    @property
    def rtc_capability(self) -> RtcModelCapability:
        ...

    @property
    def request_spec(self) -> RtcProtocolRequestSpec:
        ...

    def create_worker_executor(
        self,
        *,
        observe_bundle: Callable[[RtcClientInferenceBundle], None] | None = None,
    ) -> PolicyInferenceExecutor:
        ...


@runtime_checkable
class RtcConditionedModelRuntimeAdapterV3(Protocol):
    """Composition surface for one artifact-bound model-RTC v3 provider."""

    @property
    def rtc_continuation_capability(
        self,
    ) -> RtcModelContinuationCapabilityV3:
        ...

    @property
    def continuation_contract(self) -> ModelContinuationContract:
        ...

    @property
    def representation_context_builder(
        self,
    ) -> Callable[
        [PolicyObservationView, str],
        PolicyActionRepresentationContext,
    ]:
        ...

    def create_worker_executor(
        self,
        *,
        observe_bundle: Callable[[Any], None] | None = None,
    ) -> PolicyInferenceWorkItemExecutor:
        ...


@dataclass(frozen=True)
class SimpleRtcModelRuntimeAdapter:
    """Four-piece adapter for sessionless local/HTTP/IPC model runtimes.

    Frameworks with custom reconnect or worker-session setup can implement
    ``RtcModelRuntimeAdapter`` directly, as the OpenPI adapter does.
    """

    backend: RtcModelBackend
    decoder: RtcModelSpaceDecoder
    request_spec: RtcProtocolRequestSpec
    prepare_input: RtcModelInputPreparer
    representation_context_builder: (
        Callable[
            [PolicyObservationView, str],
            PolicyActionRepresentationContext,
        ]
        | None
    ) = None

    def __post_init__(self) -> None:
        if not isinstance(self.backend, RtcModelBackend):
            raise TypeError("backend must implement RtcModelBackend.")
        if not callable(self.decoder):
            raise TypeError("decoder must be callable.")
        if not isinstance(self.request_spec, RtcProtocolRequestSpec):
            raise TypeError("request_spec must be RtcProtocolRequestSpec.")
        if not callable(self.prepare_input):
            raise TypeError("prepare_input must be callable.")
        if self.representation_context_builder is not None and not callable(
            self.representation_context_builder
        ):
            raise TypeError("representation_context_builder must be callable or None.")

    @property
    def rtc_capability(self) -> RtcModelCapability:
        return self.backend.rtc_capability

    def create_worker_executor(
        self,
        *,
        observe_bundle: Callable[[RtcClientInferenceBundle], None] | None = None,
    ) -> PolicyInferenceExecutor:
        return RtcModelClientInferenceExecutor(
            client=ModelAgnosticRtcClient(
                backend=self.backend,
                decoder=self.decoder,
                request_spec=self.request_spec,
            ),
            prepare_input=self.prepare_input,
            observe_bundle=observe_bundle,
        )


@dataclass(frozen=True)
class RtcModelPolicyIdentity:
    """Target-neutral label used only for existing emission diagnostics."""

    name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty(self.name, name="name"))


def rtc_shadow_config_from_model_adapter(
    adapter: RtcModelRuntimeAdapter,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    producer_id: str,
    policy_node_id: str,
    profile: PolicyAsyncExecutionProfile = RTC_3C_VIRTUAL_SHADOW_PROFILE_V1,
    clock_domain_id: str = "local_monotonic",
    request_validity_steps: int = 10,
    request_trigger_coverage_steps: int = 2,
    worker_result_capacity: int = 1,
    worker_idle_poll_sec: float = 0.001,
    worker_stop_timeout_sec: float = 0.0,
    delay_estimator_spec: InferenceDelayEstimatorSpec | None = None,
) -> RtcPolicyShadowConfig:
    """Project exact negotiated model identity into coordinator config."""

    if not isinstance(adapter, RtcModelRuntimeAdapter):
        raise TypeError("adapter must implement RtcModelRuntimeAdapter.")
    request_spec = adapter.request_spec
    capability = adapter.rtc_capability
    if not isinstance(request_spec, RtcProtocolRequestSpec):
        raise TypeError("adapter.request_spec must be RtcProtocolRequestSpec.")
    if not isinstance(capability, RtcModelCapability):
        raise TypeError("adapter.rtc_capability must be RtcModelCapability.")
    negotiation = negotiate_rtc_capability(request_spec, capability)
    if not negotiation.accepted:
        raise RtcClientProtocolError(
            "capability negotiation",
            negotiation.rejection_reasons,
        )
    kwargs: dict[str, Any] = {}
    if delay_estimator_spec is not None:
        kwargs["delay_estimator_spec"] = delay_estimator_spec
    return RtcPolicyShadowConfig(
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        producer_id=producer_id,
        policy_node_id=policy_node_id,
        artifact_hash=request_spec.artifact_hash,
        action_codec_hash=request_spec.action_codec_hash,
        frame_contract_hash=request_spec.frame_contract_hash,
        layout_hash=request_spec.layout_hash,
        normalization_hash=request_spec.normalization_hash,
        action_pose_mode=request_spec.action_pose_mode,
        profile=profile,
        clock_domain_id=clock_domain_id,
        prediction_horizon_steps=request_spec.prediction_horizon_steps,
        minimum_execution_horizon_steps=(
            request_spec.minimum_execution_horizon_steps
        ),
        request_validity_steps=request_validity_steps,
        request_trigger_coverage_steps=request_trigger_coverage_steps,
        worker_result_capacity=worker_result_capacity,
        worker_idle_poll_sec=worker_idle_poll_sec,
        worker_stop_timeout_sec=worker_stop_timeout_sec,
        **kwargs,
    )


def build_rtc_model_shadow_frontend(
    adapter: RtcModelRuntimeAdapter,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    producer_id: str,
    policy_node_id: str,
    deploy_runtime: PolicyDeployRuntime,
    preparer: Any,
    created_at: float,
    policy_name: str | None = None,
    observe_bundle: Callable[[RtcClientInferenceBundle], None] | None = None,
    profile: PolicyAsyncExecutionProfile = RTC_3C_VIRTUAL_SHADOW_PROFILE_V1,
    clock_domain_id: str = "local_monotonic",
    request_validity_steps: int = 10,
    request_trigger_coverage_steps: int = 2,
    worker_result_capacity: int = 1,
    worker_idle_poll_sec: float = 0.001,
    worker_stop_timeout_sec: float = 0.0,
    delay_estimator_spec: InferenceDelayEstimatorSpec | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> RtcPolicyShadowFrontend:
    """Build one default-off raw/model RTC shadow frontend from an adapter.

    This helper deliberately does not start the returned frontend.  The
    embedding application retains lifecycle ownership and must still call
    ``start()``, ``step()`` and ``close()`` explicitly.
    """

    if not isinstance(deploy_runtime, PolicyDeployRuntime):
        raise TypeError("deploy_runtime must be PolicyDeployRuntime.")
    if not isinstance(profile, PolicyAsyncExecutionProfile):
        raise TypeError("profile must be PolicyAsyncExecutionProfile.")
    if not callable(clock):
        raise TypeError("clock must be callable.")
    if observe_bundle is not None and not callable(observe_bundle):
        raise TypeError("observe_bundle must be callable or None.")
    request_spec = adapter.request_spec
    if not isinstance(request_spec, RtcProtocolRequestSpec):
        raise TypeError("adapter.request_spec must be RtcProtocolRequestSpec.")
    runtime_arm_hold = deploy_runtime.config.arm_hold_output
    if runtime_arm_hold is not None and not isinstance(
        runtime_arm_hold,
        PolicyArmHoldOutputContract,
    ):
        raise RuntimeError(
            "PolicyDeployRuntimeConfig did not normalize arm_hold_output."
        )
    runtime_arm_hold_hash = (
        None
        if runtime_arm_hold is None
        else runtime_arm_hold.contract_hash
    )
    if (
        request_spec.arm_hold_output_contract_hash
        != runtime_arm_hold_hash
    ):
        raise ValueError(
            "RTC adapter/deploy runtime arm-hold output contract mismatch."
        )
    expected_period = 1.0 / profile.action_hz
    runtime_period = deploy_runtime.config.action_keyframe_period_sec
    if not math.isclose(
        runtime_period,
        expected_period,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("deploy runtime/action profile period mismatch.")
    config = rtc_shadow_config_from_model_adapter(
        adapter,
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        producer_id=producer_id,
        policy_node_id=policy_node_id,
        profile=profile,
        clock_domain_id=clock_domain_id,
        request_validity_steps=request_validity_steps,
        request_trigger_coverage_steps=request_trigger_coverage_steps,
        worker_result_capacity=worker_result_capacity,
        worker_idle_poll_sec=worker_idle_poll_sec,
        worker_stop_timeout_sec=worker_stop_timeout_sec,
        delay_estimator_spec=delay_estimator_spec,
    )
    capability = adapter.rtc_capability
    identity = RtcModelPolicyIdentity(
        name=(capability.provider_id if policy_name is None else policy_name),
    )

    def executor_factory(
        _worker_session_id: str,
    ) -> PolicyInferenceWorkItemExecutor:
        executor = adapter.create_worker_executor(observe_bundle=observe_bundle)
        if not callable(executor):
            raise TypeError("adapter.create_worker_executor() must return callable.")
        return executor

    representation_context_builder = getattr(
        adapter,
        "representation_context_builder",
        None,
    )
    if config.action_pose_mode == "relative" and not callable(
        representation_context_builder
    ):
        raise ValueError(
            "A relative RTC model adapter must provide "
            "representation_context_builder."
        )
    emission_adapter = getattr(preparer, "action_adapter", None)

    return RtcPolicyShadowFrontend(
        config=config,
        deploy_runtime=deploy_runtime,
        policy=identity,
        created_at=created_at,
        emission_adapter=emission_adapter,
        clock=clock,
        executor_factory=executor_factory,
        representation_context_builder=representation_context_builder,
    )


def rtc_conditioned_config_from_model_adapter_v3(
    adapter: RtcConditionedModelRuntimeAdapterV3,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    producer_id: str,
    policy_node_id: str,
    profile: PolicyAsyncExecutionProfile = (
        MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_V1
    ),
    clock_domain_id: str = "local_monotonic",
    minimum_execution_horizon_steps: int = 1,
    request_validity_steps: int = 10,
    request_trigger_coverage_steps: int | None = None,
    worker_result_capacity: int = 1,
    worker_idle_poll_sec: float = 0.001,
    worker_stop_timeout_sec: float = 0.0,
    delay_estimator_spec: InferenceDelayEstimatorSpec | None = None,
) -> RtcPolicyShadowConfig:
    """Project one v3 provider into the existing bounded RTC coordinator."""

    if not isinstance(adapter, RtcConditionedModelRuntimeAdapterV3):
        raise TypeError(
            "adapter must implement RtcConditionedModelRuntimeAdapterV3."
        )
    if not isinstance(profile, PolicyAsyncExecutionProfile):
        raise TypeError("profile must be PolicyAsyncExecutionProfile.")
    capability = adapter.rtc_continuation_capability
    if not isinstance(capability, RtcModelContinuationCapabilityV3):
        raise TypeError(
            "adapter.rtc_continuation_capability must be "
            "RtcModelContinuationCapabilityV3."
        )
    source_contract = adapter.continuation_contract
    contract = normalize_model_continuation_contract(source_contract).runtime
    validate_rtc_model_continuation_capability_v3_binding(
        capability,
        continuation_contract=source_contract,
        selected_prefix_schedule=capability.selected_prefix_schedule,
    )
    if "absolute" not in contract.supported_action_pose_modes:
        raise ValueError("Model-RTC v3 currently requires absolute actions.")

    if delay_estimator_spec is None:
        period = 1.0 / profile.action_hz
        delay_estimator_spec = InferenceDelayEstimatorSpec(
            spec_id=(
                "model_rtc_v3_shadow_delay_rolling_max:"
                f"{contract.minimum_forecast_delay_steps}:"
                f"{contract.maximum_forecast_delay_steps}"
            ),
            action_period_sec=period,
            window_size=32,
            warmup_latency_sec=period,
            network_guard_sec=0.0,
            control_jitter_guard_steps=1,
            minimum_delay_steps=contract.minimum_forecast_delay_steps,
            maximum_delay_steps=contract.maximum_forecast_delay_steps,
        )
    if request_trigger_coverage_steps is None:
        request_trigger_coverage_steps = (
            contract.maximum_forecast_delay_steps
        )

    return RtcPolicyShadowConfig(
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        producer_id=producer_id,
        policy_node_id=policy_node_id,
        artifact_hash=capability.artifact_hash,
        action_codec_hash=contract.action_codec_hash,
        frame_contract_hash=contract.frame_contract_hash,
        layout_hash=contract.layout_hash,
        normalization_hash=contract.normalization_hash,
        action_pose_mode="absolute",
        profile=profile,
        clock_domain_id=clock_domain_id,
        prediction_horizon_steps=contract.prediction_horizon_steps,
        minimum_execution_horizon_steps=minimum_execution_horizon_steps,
        request_validity_steps=request_validity_steps,
        request_trigger_coverage_steps=request_trigger_coverage_steps,
        worker_result_capacity=worker_result_capacity,
        worker_idle_poll_sec=worker_idle_poll_sec,
        worker_stop_timeout_sec=worker_stop_timeout_sec,
        delay_estimator_spec=delay_estimator_spec,
        model_continuation=RtcModelContinuationRuntimeSpec(
            continuation_contract=source_contract,
            prefix_schedule_semantics=(
                capability.selected_prefix_schedule
            ),
        ),
    )


def build_rtc_conditioned_model_shadow_frontend_v3(
    adapter: RtcConditionedModelRuntimeAdapterV3,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    producer_id: str,
    policy_node_id: str,
    deploy_runtime: PolicyDeployRuntime,
    preparer: Any,
    created_at: float,
    policy_name: str | None = None,
    observe_bundle: Callable[[Any], None] | None = None,
    profile: PolicyAsyncExecutionProfile = (
        MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_V1
    ),
    clock_domain_id: str = "local_monotonic",
    minimum_execution_horizon_steps: int = 1,
    request_validity_steps: int = 10,
    request_trigger_coverage_steps: int | None = None,
    worker_result_capacity: int = 1,
    worker_idle_poll_sec: float = 0.001,
    worker_stop_timeout_sec: float = 0.0,
    delay_estimator_spec: InferenceDelayEstimatorSpec | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> RtcPolicyShadowFrontend:
    """Build, but do not start, one default-off model-RTC v3 frontend."""

    if not isinstance(deploy_runtime, PolicyDeployRuntime):
        raise TypeError("deploy_runtime must be PolicyDeployRuntime.")
    if not callable(clock):
        raise TypeError("clock must be callable.")
    if observe_bundle is not None and not callable(observe_bundle):
        raise TypeError("observe_bundle must be callable or None.")
    expected_period = 1.0 / profile.action_hz
    if not math.isclose(
        deploy_runtime.config.action_keyframe_period_sec,
        expected_period,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("deploy runtime/action profile period mismatch.")
    config = rtc_conditioned_config_from_model_adapter_v3(
        adapter,
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        producer_id=producer_id,
        policy_node_id=policy_node_id,
        profile=profile,
        clock_domain_id=clock_domain_id,
        minimum_execution_horizon_steps=(
            minimum_execution_horizon_steps
        ),
        request_validity_steps=request_validity_steps,
        request_trigger_coverage_steps=request_trigger_coverage_steps,
        worker_result_capacity=worker_result_capacity,
        worker_idle_poll_sec=worker_idle_poll_sec,
        worker_stop_timeout_sec=worker_stop_timeout_sec,
        delay_estimator_spec=delay_estimator_spec,
    )
    context_builder = adapter.representation_context_builder
    if not callable(context_builder):
        raise TypeError(
            "Model-RTC v3 adapter must provide a representation context "
            "builder."
        )
    capability = adapter.rtc_continuation_capability
    history_enabled = bool(getattr(adapter, "observation_history_enabled", False))
    observation_observer = (
        getattr(adapter, "observe_deployment_observation", None)
        if history_enabled
        else None
    )
    request_observation_ready = (
        getattr(adapter, "request_observation_ready", None)
        if history_enabled
        else None
    )
    if observation_observer is not None and not callable(observation_observer):
        raise TypeError("history-enabled adapter observer must be callable.")
    if request_observation_ready is not None and not callable(request_observation_ready):
        raise TypeError("history-enabled adapter readiness hook must be callable.")
    identity = RtcModelPolicyIdentity(
        name=(
            capability.continuation_implementation_id
            if policy_name is None
            else policy_name
        ),
    )

    def executor_factory(_worker_session_id: str) -> PolicyInferenceExecutor:
        executor = adapter.create_worker_executor(
            observe_bundle=observe_bundle
        )
        if not callable(executor):
            raise TypeError(
                "adapter.create_worker_executor() must return callable."
            )
        if not callable(getattr(executor, "execute_work_item", None)):
            raise TypeError(
                "Model-RTC v3 executor must implement callable "
                "execute_work_item()."
            )
        return executor

    return RtcPolicyShadowFrontend(
        config=config,
        deploy_runtime=deploy_runtime,
        policy=identity,
        created_at=created_at,
        emission_adapter=getattr(preparer, "action_adapter", None),
        clock=clock,
        executor_factory=executor_factory,
        observation_observer=observation_observer,
        request_observation_ready=request_observation_ready,
        representation_context_builder=context_builder,
    )


# Canonical functional names for new raw/model RTC integrations.  The existing
# Shadow/Frontend names remain the same callable objects for import and
# behavior compatibility.
rtc_producer_config_from_model_adapter = rtc_shadow_config_from_model_adapter
build_rtc_model_producer_runtime = build_rtc_model_shadow_frontend


__all__ = [
    "RtcConditionedModelRuntimeAdapterV3",
    "RtcModelPolicyIdentity",
    "RtcModelRuntimeAdapter",
    "SimpleRtcModelRuntimeAdapter",
    "build_rtc_conditioned_model_shadow_frontend_v3",
    "build_rtc_model_producer_runtime",
    "build_rtc_model_shadow_frontend",
    "rtc_conditioned_config_from_model_adapter_v3",
    "rtc_producer_config_from_model_adapter",
    "rtc_shadow_config_from_model_adapter",
]
