"""OpenPI binding for the backend-neutral threaded emission source."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping

from dataset_contracts.policy_artifact import (
    PolicyArtifactContract,
    policy_artifact_from_metadata,
)
from dataset_contracts.pose_frames import PoseFrameContract
from robot_motion.deploy import (
    DeployActionPreparer,
    DeployRobotActionAdapter,
    NormalizationContract,
    PolicyAdapterConfig,
    PolicyDeployRuntime,
    ThreadedPolicyEmissionSource,
    ThreadedPolicyEmissionSourceConfig,
)

from deployments.openpi.artifact_identity import openpi_rtc_artifact_hash
from deployments.openpi.state_only import (
    OPENPI_STATE_ONLY_CAPABILITY_KEY_V1,
    OpenPIStateOnlyInputBridge,
)

from .policy import OpenPIClientPolicy, OpenPIClientPolicyConfig
from .preflight import preflight_policy_artifact


OPENPI_POLICY_EMISSION_SOURCE_ID_V1 = "openpi.policy_emission_source.v1"


def validate_openpi_state_policy_artifact(
    artifact: PolicyArtifactContract,
) -> None:
    """Require an exact prepared external-camera-free state-only contract."""

    if not isinstance(artifact, PolicyArtifactContract):
        raise TypeError("artifact must be a PolicyArtifactContract.")
    capability = artifact.metadata.get(OPENPI_STATE_ONLY_CAPABILITY_KEY_V1)
    if not isinstance(capability, dict):
        raise ValueError(
            "OpenPI state policy requires the prepared state-only capability "
            "in policy artifact metadata."
        )
    try:
        expected_capability = OpenPIStateOnlyInputBridge(
            image_keys=tuple(capability["internal_image_keys"]),
            image_shape=tuple(capability["internal_image_shape"]),
        ).contract
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "OpenPI state policy artifact has an invalid state-only capability."
        ) from exc
    if capability != expected_capability:
        raise ValueError(
            "OpenPI state policy artifact state-only capability hash or fields "
            "are inconsistent."
        )
    camera = artifact.camera_input
    if camera is None or camera.required or camera.inputs:
        raise ValueError(
            "OpenPI state policy requires a camera-free external artifact contract."
        )


def _sides(value: tuple[str, ...], *, name: str) -> tuple[str, ...]:
    result = tuple(str(side).strip().lower() for side in value)
    if not result or any(side not in {"left", "right"} for side in result):
        raise ValueError(f"{name} must contain left and/or right.")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates.")
    return result


@dataclass(frozen=True)
class OpenPIEmissionSourceConfig:
    """Model-family binding; no simulator or hardware object is included."""

    policy_artifact: PolicyArtifactContract | Mapping[str, Any]
    runtime_pose_frame_contract: PoseFrameContract
    output_active_sides: tuple[str, ...]
    model_observation_sides: tuple[str, ...] = ("left", "right")
    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str | None = None
    prompt: str | None = None
    action_hz: float = 15.0
    policy_hz: float = 15.0
    action_chunk_execution_steps: int | None = None
    prediction_request_queue_threshold: int = 4
    downstream_target_lead_sec: float = 0.0
    policy_node_id: str = "openpi-policy"
    policy_clock_domain_id: str = "local_monotonic"
    include_debug_metadata: bool = True
    retry_interval_s: float = 2.0

    def __post_init__(self) -> None:
        artifact = self.policy_artifact
        if isinstance(artifact, Mapping):
            artifact = policy_artifact_from_metadata(artifact)
        if not isinstance(artifact, PolicyArtifactContract):
            raise TypeError("policy_artifact must be typed or a mapping.")
        if not isinstance(self.runtime_pose_frame_contract, PoseFrameContract):
            raise TypeError(
                "runtime_pose_frame_contract must be a PoseFrameContract."
            )
        output_sides = _sides(self.output_active_sides, name="output_active_sides")
        model_sides = _sides(
            self.model_observation_sides,
            name="model_observation_sides",
        )
        if not set(output_sides).issubset(set(model_sides)):
            raise ValueError(
                "model_observation_sides must cover every output_active_side."
            )
        for name in ("action_hz", "policy_hz"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)
        lead = float(self.downstream_target_lead_sec)
        if not math.isfinite(lead) or lead < 0.0:
            raise ValueError(
                "downstream_target_lead_sec must be finite and non-negative."
            )
        execution_steps = self.action_chunk_execution_steps
        if execution_steps is not None:
            execution_steps = int(execution_steps)
            if execution_steps <= 0:
                raise ValueError("action_chunk_execution_steps must be positive.")
        pose_chunk = artifact.pose_action_chunk
        if pose_chunk is None:
            raise ValueError(
                "OpenPI emission source requires an artifact-bound pose action chunk."
            )
        if execution_steps is not None and execution_steps > pose_chunk.action_horizon:
            raise ValueError(
                "action_chunk_execution_steps exceeds the artifact horizon."
            )
        threshold = int(self.prediction_request_queue_threshold)
        selected_horizon = (
            pose_chunk.action_horizon
            if execution_steps is None
            else execution_steps
        )
        if threshold < 0 or threshold >= selected_horizon:
            raise ValueError(
                "prediction_request_queue_threshold must be within the selected "
                "execution horizon."
            )
        normalization = artifact.metadata.get("normalization_contract")
        if not isinstance(normalization, Mapping):
            # ``NormalizationContract.from_optional_mapping(None)`` denotes an
            # explicitly selected identity transform at generic boundaries.
            # For an OpenPI checkpoint, however, an absent artifact field is
            # unknown ownership rather than verified identity normalization.
            raise ValueError(
                "OpenPI emission source requires a prepared deployment artifact "
                "with normalization_contract metadata."
            )
        parsed_normalization = NormalizationContract.from_optional_mapping(
            normalization,
            allow_legacy=False,
        )
        if parsed_normalization is None:
            raise ValueError(
                "OpenPI emission source requires a prepared deployment artifact "
                "with normalization_contract metadata."
            )
        # This validates the model topology, pose origin, artifact layout and
        # the explicit runtime frame binding without opening a transport.
        preflight_policy_artifact(
            artifact,
            active_sides=model_sides,
            runtime_pose_frame_contract=self.runtime_pose_frame_contract,
        )
        host = str(self.host).strip()
        port = int(self.port)
        if not host or port <= 0 or port > 65535:
            raise ValueError("OpenPI host/port are invalid.")
        for name in ("policy_node_id", "policy_clock_domain_id"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty.")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "policy_artifact", artifact)
        object.__setattr__(self, "output_active_sides", output_sides)
        object.__setattr__(self, "model_observation_sides", model_sides)
        object.__setattr__(self, "host", host)
        object.__setattr__(self, "port", port)
        object.__setattr__(self, "action_chunk_execution_steps", execution_steps)
        object.__setattr__(self, "prediction_request_queue_threshold", threshold)
        object.__setattr__(self, "downstream_target_lead_sec", lead)


def build_openpi_policy_emission_source(
    config: OpenPIEmissionSourceConfig,
    *,
    observation_capture: Callable[..., Any],
    observation_reset: Callable[[], None] | None = None,
    last_action_sink: Callable[[Any], None] | None = None,
    policy: Any | None = None,
    monotonic_clock: Callable[[], float] | None = None,
) -> ThreadedPolicyEmissionSource:
    """Bind OpenPI to the shared queue/emission source, without a target."""

    if not isinstance(config, OpenPIEmissionSourceConfig):
        raise TypeError("config must be an OpenPIEmissionSourceConfig.")
    artifact = config.policy_artifact
    assert isinstance(artifact, PolicyArtifactContract)
    pose_chunk = artifact.pose_action_chunk
    assert pose_chunk is not None
    normalization = artifact.metadata["normalization_contract"]
    adapter = PolicyAdapterConfig(
        adapter_id=OPENPI_POLICY_EMISSION_SOURCE_ID_V1,
        active_sides=config.output_active_sides,
        model_observation_sides=config.model_observation_sides,
        arms_enabled=True,
        hands_enabled=True,
        model_action_dim=artifact.policy_io_layout.action_dim,
        model_layout_hash=artifact.layout_hash,
        normalization_contract=normalization,
        action_pose_mode=pose_chunk.action_pose_mode,
        policy_absolute_frame="canonical_desk",
        runtime_pose_frame_contract=config.runtime_pose_frame_contract,
        artifact_hash=openpi_rtc_artifact_hash(artifact),
        prediction_horizon_steps=pose_chunk.action_horizon,
        minimum_execution_horizon_steps=1,
        action_period_sec=1.0 / config.action_hz,
        action_chunk_execution_steps=config.action_chunk_execution_steps,
        metadata={
            "model_observation_sides": config.model_observation_sides,
            "output_active_sides": config.output_active_sides,
        },
    )
    maximum_hold_sec = max(
        0.25,
        2.0 / config.policy_hz + config.downstream_target_lead_sec,
    )
    runtime_config = replace(
        adapter.to_runtime_config(),
        policy_node_id=config.policy_node_id,
        policy_clock_domain_id=config.policy_clock_domain_id,
        action_maximum_hold_sec=maximum_hold_sec,
        extend_final_keyframe_to_maximum_hold=True,
        policy_observation_binding_id="openpi_artifact_v1",
        arm_hold_output=artifact.arm_hold_output,
    )
    runtime = PolicyDeployRuntime(
        runtime_config,
        monotonic_clock=(time.monotonic if monotonic_clock is None else monotonic_clock),
    )
    if policy is None:
        camera_input = artifact.camera_input
        # Artifact model/checkpoint identity remains visible in status and
        # receipts, but an ordinary OpenPI client may intentionally point at a
        # different compatible server revision.  Direct callers that need an
        # identity guard can still opt into ``expected_*`` on the low-level
        # OpenPIClientPolicyConfig.
        policy = OpenPIClientPolicy(
            OpenPIClientPolicyConfig(
                host=config.host,
                port=config.port,
                api_key=config.api_key,
                prompt=config.prompt,
                active_sides=config.model_observation_sides,
                required_camera_slots=(
                    () if camera_input is None else camera_input.server_camera_slots
                ),
                required_image_keys=(
                    ()
                    if camera_input is None or not camera_input.required
                    else camera_input.source_keys
                ),
                image_key_map=(
                    None if camera_input is None else camera_input.image_key_map
                ),
                include_debug_metadata=config.include_debug_metadata,
                retry_interval_s=config.retry_interval_s,
                defer_connect=True,
                policy_artifact=artifact,
            )
        )
    clock = time.monotonic if monotonic_clock is None else monotonic_clock
    return ThreadedPolicyEmissionSource(
        ThreadedPolicyEmissionSourceConfig(
            source_id=OPENPI_POLICY_EMISSION_SOURCE_ID_V1,
            policy_name=str(getattr(policy, "name", "openpi")),
            policy_interval_sec=1.0 / config.policy_hz,
            action_period_sec=1.0 / config.action_hz,
            maximum_hold_sec=maximum_hold_sec,
            prediction_request_queue_threshold=(
                config.prediction_request_queue_threshold
            ),
            status_metadata={
                "backend": "openpi_state",
                "checkpoint": (
                    f"{artifact.model_id}@{artifact.checkpoint_step}"
                ),
                "adapter_config_hash": adapter.semantic_content_hash,
                "artifact_hash": adapter.artifact_hash,
                "model_id": artifact.model_id,
                "checkpoint_step": artifact.checkpoint_step,
                "artifact_layout_hash": artifact.layout_hash,
                "artifact_action_horizon": pose_chunk.action_horizon,
            },
        ),
        runtime=runtime,
        policy=policy,
        observation_capture=observation_capture,
        observation_reset=observation_reset,
        last_action_sink=last_action_sink,
        action_preparer=DeployActionPreparer(
            DeployRobotActionAdapter(
                active_sides=config.output_active_sides,
                arms_enabled=True,
                hands_enabled=True,
            )
        ),
        monotonic_clock=clock,
    )


__all__ = [
    "OPENPI_POLICY_EMISSION_SOURCE_ID_V1",
    "OpenPIEmissionSourceConfig",
    "build_openpi_policy_emission_source",
    "validate_openpi_state_policy_artifact",
]
