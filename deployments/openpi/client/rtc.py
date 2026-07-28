"""OpenPI final-chunk adapter for the generic RTC protocol.

This adapter proves that an existing artifact-bound final-chunk model can use
the RTC v2 envelope without being mislabeled as a continuation model.  It is
safe for an offline fake transport and can wrap the existing OpenPI transport,
but it owns neither worker scheduling nor plan admission.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.policy_artifact import PolicyArtifactContract
from deployments.client.artifact import deploy_contract_from_policy_artifact
from deployments.client.remote_policy import (
    RemoteModelPolicy,
    RemotePolicyConfig,
    RemoteTransport,
)
from deployments.client.rtc import ModelAgnosticRtcClient, RtcClientInferenceBundle
from deployments.client.rtc_executor import (
    RtcModelClientInferenceExecutor,
    RtcPreparedModelInput,
)
from robot_motion.action import RobotAction
from robot_motion.deploy.action_representation import (
    ACTION_ANCHOR_POLICY_NONE_V1,
    PolicyActionRepresentationContext,
    build_policy_action_representation_context,
)
from robot_motion.deploy.actions import NormalizedAction
from robot_motion.deploy.canonical_policy_action import (
    AbsolutePolicyActionChunkCanonicalizer,
    RelativePolicyActionChunkCanonicalizer,
)
from robot_motion.deploy.inference import PolicyInferenceRequest
from robot_motion.deploy.observation_view import PolicyObservationView
from robot_motion.deploy.rtc_protocol import (
    RTC_CONTINUATION_NONE,
    RTC_PREFIX_FORBIDDEN,
    RTC_PROTOCOL_V2,
    RTC_SCHEDULER_ASYNC_TIMED_PLAN,
    RtcContinuationCapability,
    RtcInferenceEnvelopeV2,
    RtcModelCapability,
    RtcModelResponseEnvelopeV2,
    RtcProtocolMode,
    RtcProtocolRequestSpec,
)
from robot_motion.deploy.session_contract import (
    resolve_policy_action_pose_mode_from_artifact,
)
from robot_motion.deploy.v22 import (
    V22DeploymentAdapter,
    V22PolicyActionChunk,
)
from robot_motion.runtime.immutable import deep_freeze, deep_thaw

from .observation_adapter import (
    OpenPIObservationAdapterConfig,
    deployment_observation_to_openpi,
)
from .policy import OPENPI_LOCAL_PROTOCOL_VERSION


OPENPI_FINAL_CHUNK_RTC_ADAPTER_V1 = "openpi_final_chunk_rtc_adapter_v1"
OPENPI_FINAL_CHUNK_MODEL_ACTION_SPACE_V1 = "dexteleop_policy_io_layout_v22"
OPENPI_RTC_ARTIFACT_HASH_DOMAIN_V1 = "dexteleop.openpi_rtc_artifact.v1"
OPENPI_RTC_CODEC_HASH_DOMAIN_V1 = "dexteleop.openpi_rtc_v22_codec.v1"
OPENPI_RTC_INPUT_HASH_DOMAIN_V1 = "dexteleop.openpi_rtc_input.v1"
OPENPI_RTC_SERVER_HASH_DOMAIN_V1 = "dexteleop.openpi_rtc_server_metadata.v1"
OPENPI_RTC_IMPLEMENTATION_HASH_DOMAIN_V1 = (
    "dexteleop.openpi_rtc_implementation_parameters.v1"
)
OPENPI_RTC_OUTPUT_ID_DOMAIN_V1 = "dexteleop.openpi_rtc_model_output.v1"
OPENPI_RTC_ABSOLUTE_CONVERSION_HASH_DOMAIN_V1 = (
    "dexteleop.openpi_rtc_absolute_conversion.v1"
)
OPENPI_RTC_RELATIVE_CONVERSION_HASH_DOMAIN_V1 = (
    "dexteleop.openpi_rtc_relative_conversion.v1"
)
OPENPI_RTC_RESULT_LEGACY_NORMALIZED_V1 = "legacy_normalized_v1"
OPENPI_RTC_RESULT_TYPED_CANONICAL_V1 = "typed_canonical_v1"
OPENPI_RTC_RESULT_REPRESENTATIONS_V1 = (
    OPENPI_RTC_RESULT_LEGACY_NORMALIZED_V1,
    OPENPI_RTC_RESULT_TYPED_CANONICAL_V1,
)


def _nonempty(value: str, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def openpi_rtc_artifact_hash(artifact: PolicyArtifactContract) -> str:
    """Hash the full sidecar, not only its layout."""

    if not isinstance(artifact, PolicyArtifactContract):
        raise TypeError("artifact must be a PolicyArtifactContract.")
    return canonical_content_hash(
        artifact.to_dict(),
        domain=OPENPI_RTC_ARTIFACT_HASH_DOMAIN_V1,
    )


def openpi_v22_legacy_codec_hash(artifact: PolicyArtifactContract) -> str:
    """Bind the exact typed-layout to legacy NormalizedAction bridge."""

    return canonical_content_hash(
        {
            "adapter": "V22DeploymentAdapter.to_legacy_normalized_actions",
            "adapter_version": "v1",
            "layout_hash": artifact.layout_hash,
            "action_pose_mode": "absolute",
            "output_type": "NormalizedAction",
        },
        domain=OPENPI_RTC_CODEC_HASH_DOMAIN_V1,
    )


def openpi_v22_native_codec_hash(artifact: PolicyArtifactContract) -> str:
    """Bind the exact typed layout to its native RobotAction projection."""

    return canonical_content_hash(
        {
            "adapter": "V22DeploymentAdapter.to_native_robot_actions",
            "adapter_version": "v1",
            "layout_hash": artifact.layout_hash,
            "action_pose_mode": "absolute",
            "output_type": "CanonicalPolicyAction",
        },
        domain=OPENPI_RTC_CODEC_HASH_DOMAIN_V1,
    )


def openpi_v22_relative_native_codec_hash(
    artifact: PolicyArtifactContract,
) -> str:
    """Bind relative arm decode plus the typed v2.2 component projection."""

    return canonical_content_hash(
        {
            "adapter": (
                "RelativePolicyActionChunkCanonicalizer+"
                "V22DeploymentAdapter.to_native_robot_actions"
            ),
            "adapter_version": "v1",
            "layout_hash": artifact.layout_hash,
            "action_pose_mode": "relative",
            "output_type": "CanonicalPolicyAction",
        },
        domain=OPENPI_RTC_CODEC_HASH_DOMAIN_V1,
    )


def openpi_arm_pose_slices(
    artifact: PolicyArtifactContract,
    *,
    active_sides: Sequence[str],
) -> dict[str, slice]:
    """Resolve arm pose columns from the artifact layout, never a checkpoint name."""

    components = {
        component.name: component
        for component in artifact.policy_io_layout.components("action", "continuous")
    }
    slices: dict[str, slice] = {}
    for side in active_sides:
        name = f"{side}.arm.pose"
        component = components.get(name)
        if component is None or component.dim != 7:
            raise ValueError(
                f"OpenPI RTC layout must declare one 7D continuous {name!r} component."
            )
        slices[str(side)] = slice(component.offset, component.stop)
    return slices


def openpi_arm_reference_frames(
    artifact: PolicyArtifactContract,
    *,
    active_sides: Sequence[str],
) -> dict[str, str]:
    """Return the absolute frame produced before RTC plan admission."""

    if artifact.pose_action_chunk is not None:
        return {str(side): "canonical_desk" for side in active_sides}
    layout = artifact.policy_io_layout
    return {
        str(side): str(layout.arm_reference_frame_by_side[str(side)])
        for side in active_sides
    }


def openpi_absolute_conversion_contract_hash(
    artifact: PolicyArtifactContract,
    *,
    frame_contract_hash: str,
) -> str:
    """Identity for declared absolute model rows becoming typed RobotAction."""

    layout = artifact.policy_io_layout
    return canonical_content_hash(
        {
            "adapter": "V22DeploymentAdapter.to_native_robot_actions",
            "adapter_version": "v1",
            "source_pose_mode": "absolute",
            "frame_contract_hash": str(frame_contract_hash),
            "layout_hash": artifact.layout_hash,
            "arm_pose_representation": layout.arm_pose_representation,
            "arm_reference_frame_by_side": layout.arm_reference_frame_by_side,
            "arm_controlled_frame_by_side": layout.arm_controlled_frame_by_side,
        },
        domain=OPENPI_RTC_ABSOLUTE_CONVERSION_HASH_DOMAIN_V1,
    )


def openpi_relative_conversion_contract_hash(
    artifact: PolicyArtifactContract,
    *,
    frame_contract_hash: str,
    active_sides: Sequence[str],
) -> str:
    """Identity for fixed-request-anchor relative rows becoming absolute actions."""

    pose_chunk = artifact.pose_action_chunk
    if pose_chunk is None or pose_chunk.action_pose_mode != "relative":
        raise ValueError(
            "Relative OpenPI RTC conversion requires a structured relative pose chunk."
        )
    layout = artifact.policy_io_layout
    arm_slices = openpi_arm_pose_slices(artifact, active_sides=active_sides)
    return canonical_content_hash(
        {
            "adapter": "RelativePolicyActionChunkCanonicalizer",
            "adapter_version": "v1",
            "source_pose_mode": "relative",
            "relative_action_anchor": "chunk_start_actual_tcp",
            "absolute_pose_frame": "canonical_desk",
            "pose_action_chunk_contract_hash": pose_chunk.contract_hash,
            "frame_contract_hash": str(frame_contract_hash),
            "layout_hash": artifact.layout_hash,
            "arm_pose_representation": layout.arm_pose_representation,
            "arm_pose_slices_by_side": {
                side: (value.start, value.stop)
                for side, value in arm_slices.items()
            },
            "arm_reference_frame_by_side": openpi_arm_reference_frames(
                artifact,
                active_sides=active_sides,
            ),
            "arm_controlled_frame_by_side": {
                side: layout.arm_controlled_frame_by_side[side]
                for side in active_sides
            },
        },
        domain=OPENPI_RTC_RELATIVE_CONVERSION_HASH_DOMAIN_V1,
    )


def openpi_rtc_component_scope(artifact: PolicyArtifactContract) -> tuple[str, ...]:
    layout = artifact.policy_io_layout
    names = (
        *(component.name for component in layout.components("action", "continuous")),
        *(component.name for component in layout.components("action", "discrete")),
    )
    return tuple(dict.fromkeys(str(name) for name in names))


@dataclass(frozen=True)
class OpenPIFinalChunkRtcConfig:
    """Explicit bindings that cannot be inferred from a model family name."""

    artifact: PolicyArtifactContract
    frame_contract_hash: str
    normalization_hash: str
    active_sides: tuple[str, ...] = ("left", "right")
    action_pose_mode: str = "absolute"
    minimum_execution_horizon_steps: int = 1
    include_debug_observation_metadata: bool = True
    prompt: str | None = None
    result_action_representation: str = OPENPI_RTC_RESULT_LEGACY_NORMALIZED_V1

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, PolicyArtifactContract):
            raise TypeError("artifact must be a PolicyArtifactContract.")
        if not self.artifact.model_id:
            raise ValueError("RTC OpenPI artifact must declare model_id.")
        if self.artifact.checkpoint_step is None:
            raise ValueError("RTC OpenPI artifact must declare checkpoint_step.")
        if self.artifact.camera_input is None:
            raise ValueError(
                "RTC OpenPI artifact must explicitly declare camera_input; use "
                "required=false with an empty input list for a state-only model."
            )
        object.__setattr__(
            self,
            "frame_contract_hash",
            _nonempty(self.frame_contract_hash, name="frame_contract_hash"),
        )
        object.__setattr__(
            self,
            "normalization_hash",
            _nonempty(self.normalization_hash, name="normalization_hash"),
        )
        raw_sides = tuple(str(side).strip().lower() for side in self.active_sides)
        requested_sides = set(raw_sides)
        if not requested_sides or not requested_sides.issubset({"left", "right"}):
            raise ValueError("active_sides must contain left and/or right.")
        if len(requested_sides) != len(raw_sides):
            raise ValueError("active_sides must not contain duplicates.")
        sides = tuple(side for side in ("left", "right") if side in requested_sides)
        object.__setattr__(self, "active_sides", sides)
        pose_mode = _nonempty(self.action_pose_mode, name="action_pose_mode")
        pose_mode = resolve_policy_action_pose_mode_from_artifact(
            self.artifact,
            configured_mode=pose_mode,
            explicit=True,
        )
        object.__setattr__(self, "action_pose_mode", pose_mode)
        minimum = int(self.minimum_execution_horizon_steps)
        if minimum <= 0 or minimum != self.minimum_execution_horizon_steps:
            raise ValueError("minimum_execution_horizon_steps must be positive.")
        object.__setattr__(self, "minimum_execution_horizon_steps", minimum)
        object.__setattr__(
            self,
            "include_debug_observation_metadata",
            bool(self.include_debug_observation_metadata),
        )
        if self.prompt is not None:
            object.__setattr__(self, "prompt", str(self.prompt))
        result_representation = _nonempty(
            self.result_action_representation,
            name="result_action_representation",
        )
        if result_representation not in OPENPI_RTC_RESULT_REPRESENTATIONS_V1:
            raise ValueError(
                "Unsupported OpenPI RTC result action representation: "
                f"{result_representation!r}."
            )
        if (
            pose_mode == "relative"
            and result_representation != OPENPI_RTC_RESULT_TYPED_CANONICAL_V1
        ):
            raise ValueError(
                "Relative OpenPI raw-RTC requires typed_canonical_v1 results."
            )
        if pose_mode == "relative":
            pose_chunk = self.artifact.pose_action_chunk
            if pose_chunk is None or pose_chunk.action_pose_mode != "relative":
                raise ValueError(
                    "Relative OpenPI raw-RTC requires a structured relative pose "
                    "action chunk contract."
                )
            if self.frame_contract_hash not in self.artifact.pose_frame_contract_hashes:
                raise ValueError(
                    "Relative OpenPI raw-RTC frame identity is not bound by the "
                    "policy artifact."
                )
        object.__setattr__(
            self,
            "result_action_representation",
            result_representation,
        )

    @property
    def artifact_hash(self) -> str:
        return openpi_rtc_artifact_hash(self.artifact)

    @property
    def action_codec_hash(self) -> str:
        if self.result_action_representation == OPENPI_RTC_RESULT_TYPED_CANONICAL_V1:
            if self.action_pose_mode == "relative":
                return openpi_v22_relative_native_codec_hash(self.artifact)
            return openpi_v22_native_codec_hash(self.artifact)
        return openpi_v22_legacy_codec_hash(self.artifact)

    @property
    def absolute_conversion_contract_hash(self) -> str:
        return openpi_absolute_conversion_contract_hash(
            self.artifact,
            frame_contract_hash=self.frame_contract_hash,
        )

    @property
    def conversion_contract_hash(self) -> str:
        if self.action_pose_mode == "relative":
            return openpi_relative_conversion_contract_hash(
                self.artifact,
                frame_contract_hash=self.frame_contract_hash,
                active_sides=self.active_sides,
            )
        return self.absolute_conversion_contract_hash

    @property
    def arm_pose_slices(self) -> Mapping[str, slice]:
        return openpi_arm_pose_slices(
            self.artifact,
            active_sides=self.active_sides,
        )

    @property
    def arm_reference_frame_by_side(self) -> Mapping[str, str]:
        return openpi_arm_reference_frames(
            self.artifact,
            active_sides=self.active_sides,
        )

    @property
    def expected_action_horizon(self) -> int | None:
        chunk = self.artifact.pose_action_chunk
        return None if chunk is None else int(chunk.action_horizon)

    @property
    def component_scope(self) -> tuple[str, ...]:
        return openpi_rtc_component_scope(self.artifact)


@dataclass(frozen=True)
class OpenPIRtcPreparedInput:
    """Exact OpenPI payload whose hash is bound into the RTC envelope."""

    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.payload, Mapping):
            raise TypeError("payload must be a mapping.")
        object.__setattr__(self, "payload", deep_freeze(dict(self.payload)))

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {"payload": self.payload},
            domain=OPENPI_RTC_INPUT_HASH_DOMAIN_V1,
        )


class OpenPIV22FinalChunkDecoder:
    """Decode both continuous values and independent int64 tokens."""

    def __init__(self, config: OpenPIFinalChunkRtcConfig) -> None:
        self.config = config
        self._adapter = V22DeploymentAdapter(config.artifact.policy_io_layout)

    def __call__(
        self,
        model_space_actions: np.ndarray,
        model_space_discrete_actions: np.ndarray | None,
        request: PolicyInferenceRequest,
    ) -> Sequence[NormalizedAction]:
        if self.config.action_pose_mode != "absolute":
            raise ValueError(
                "Legacy OpenPI final-chunk decoding only supports absolute actions."
            )
        horizon = int(model_space_actions.shape[0])
        discrete = (
            np.zeros((horizon, 0), dtype=np.int64)
            if model_space_discrete_actions is None
            else np.asarray(model_space_discrete_actions, dtype=np.int64)
        )
        chunk = V22PolicyActionChunk(
            actions=model_space_actions,
            discrete_actions=discrete,
            layout_hash=self.config.artifact.layout_hash,
            metadata={
                "source": OPENPI_FINAL_CHUNK_RTC_ADAPTER_V1,
                "rtc_request_id": request.request_id,
                "rtc_artifact_hash": self.config.artifact_hash,
                "rtc_frame_contract_hash": self.config.frame_contract_hash,
                "rtc_normalization_hash": self.config.normalization_hash,
                "pose_action_source_mode": "absolute",
                "pose_action_mode": "absolute",
            },
        )
        return self._adapter.to_legacy_normalized_actions(
            chunk,
            source=OPENPI_FINAL_CHUNK_RTC_ADAPTER_V1,
        )


class OpenPIV22NativeRobotActionDecoder:
    """Decode already-absolute OpenPI rows without a legacy q6 projection."""

    def __init__(self, config: OpenPIFinalChunkRtcConfig) -> None:
        self.config = config
        self._adapter = V22DeploymentAdapter(config.artifact.policy_io_layout)

    def __call__(
        self,
        model_space_actions: np.ndarray,
        model_space_discrete_actions: np.ndarray | None,
        request: PolicyInferenceRequest,
    ) -> Sequence[RobotAction]:
        horizon = int(model_space_actions.shape[0])
        discrete = (
            np.zeros((horizon, 0), dtype=np.int64)
            if model_space_discrete_actions is None
            else np.asarray(model_space_discrete_actions, dtype=np.int64)
        )
        chunk = V22PolicyActionChunk(
            actions=model_space_actions,
            discrete_actions=discrete,
            layout_hash=self.config.artifact.layout_hash,
            metadata={
                "source": OPENPI_FINAL_CHUNK_RTC_ADAPTER_V1,
                "rtc_request_id": request.request_id,
                "rtc_artifact_hash": self.config.artifact_hash,
                "rtc_frame_contract_hash": self.config.frame_contract_hash,
                "rtc_normalization_hash": self.config.normalization_hash,
                "pose_action_source_mode": self.config.action_pose_mode,
                "pose_action_mode": "absolute",
            },
        )
        decoded = self._adapter.to_native_robot_actions(
            chunk,
            source=OPENPI_FINAL_CHUNK_RTC_ADAPTER_V1,
        )
        reference_frames = self.config.arm_reference_frame_by_side
        output: list[RobotAction] = []
        for action in decoded:
            frames = dict(action.frame_by_effector)
            for side, frame in reference_frames.items():
                frames[f"{side}_arm"] = frame
            output.append(replace(action, frame_by_effector=frames))
        return tuple(output)


class OpenPIFinalChunkRtcBackend:
    """Artifact-bound OpenPI backend advertising continuation=none only."""

    def __init__(
        self,
        *,
        transport: RemoteTransport | None = None,
        remote_policy: RemoteModelPolicy | None = None,
        config: OpenPIFinalChunkRtcConfig,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._clock = clock
        if remote_policy is not None and not isinstance(remote_policy, RemoteModelPolicy):
            raise TypeError("remote_policy must be a RemoteModelPolicy or None.")
        if remote_policy is None and transport is None:
            raise ValueError("OpenPI RTC backend requires transport or remote_policy.")
        if remote_policy is not None:
            policy_transport = remote_policy.transport
            if transport is not None and transport is not policy_transport:
                raise ValueError(
                    "transport and remote_policy must refer to the same transport."
                )
            transport = policy_transport
        assert transport is not None
        self._transport = transport
        artifact = config.artifact
        camera = artifact.camera_input
        assert camera is not None
        self._observation_config = OpenPIObservationAdapterConfig(
            prompt=config.prompt,
            image_key_map=camera.image_key_map,
            required_image_keys=(camera.source_keys if camera.required else ()),
            include_debug_metadata=config.include_debug_observation_metadata,
            state_key=artifact.state_key,
            state_discrete_key=artifact.state_discrete_key,
        )
        contract = deploy_contract_from_policy_artifact(
            artifact,
            protocol_version=OPENPI_LOCAL_PROTOCOL_VERSION,
            active_sides=config.active_sides,
        )
        self._remote_policy = (
            remote_policy
            if remote_policy is not None
            else RemoteModelPolicy(
                transport=transport,
                observation_adapter=_prepared_input_payload,
                config=RemotePolicyConfig(
                    action_key=artifact.action_key,
                    action_dim=artifact.policy_io_layout.action_dim,
                    action_discrete_key=artifact.action_discrete_key,
                    action_discrete_dim=artifact.policy_io_layout.action_discrete_dim,
                    contract=contract,
                    source=OPENPI_FINAL_CHUNK_RTC_ADAPTER_V1,
                ),
            )
        )
        if (
            self._remote_policy.config.action_key != artifact.action_key
            or self._remote_policy.config.action_dim
            != artifact.policy_io_layout.action_dim
            or self._remote_policy.config.action_discrete_dim
            != artifact.policy_io_layout.action_discrete_dim
            or (
                artifact.policy_io_layout.action_discrete_dim > 0
                and self._remote_policy.config.action_discrete_key
                != artifact.action_discrete_key
            )
        ):
            raise ValueError(
                "OpenPI raw-RTC remote policy output config disagrees with the artifact."
            )
        summary = dict(self._remote_policy.validation.server_summary)
        horizon = summary.get("action_horizon")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise ValueError(
                "RTC OpenPI server metadata must declare a positive model.action_horizon."
            )
        if config.minimum_execution_horizon_steps > horizon:
            raise ValueError(
                "minimum_execution_horizon_steps exceeds server action_horizon."
            )
        expected_horizon = config.expected_action_horizon
        if expected_horizon is not None and expected_horizon != horizon:
            raise ValueError(
                "RTC OpenPI server action_horizon disagrees with the policy artifact "
                f"pose action chunk: {horizon} != {expected_horizon}."
            )
        self._prediction_horizon_steps = int(horizon)
        self._server_metadata_hash = canonical_content_hash(
            self._remote_policy.server_metadata,
            domain=OPENPI_RTC_SERVER_HASH_DOMAIN_V1,
        )
        self._implementation_parameters_hash = canonical_content_hash(
            {
                "adapter": OPENPI_FINAL_CHUNK_RTC_ADAPTER_V1,
                "openpi_protocol": OPENPI_LOCAL_PROTOCOL_VERSION,
                "artifact_hash": config.artifact_hash,
                "server_metadata_hash": self._server_metadata_hash,
                "server_summary": summary,
                "action_key": artifact.action_key,
                "action_discrete_key": artifact.action_discrete_key,
            },
            domain=OPENPI_RTC_IMPLEMENTATION_HASH_DOMAIN_V1,
        )
        continuation = RtcContinuationCapability(
            continuation_mode=RTC_CONTINUATION_NONE,
            implementation_id=OPENPI_FINAL_CHUNK_RTC_ADAPTER_V1,
            implementation_parameters_hash=self._implementation_parameters_hash,
            model_action_space=OPENPI_FINAL_CHUNK_MODEL_ACTION_SPACE_V1,
            model_action_dim=artifact.policy_io_layout.action_dim,
            model_action_discrete_dim=artifact.policy_io_layout.action_discrete_dim,
            supported_action_pose_modes=(config.action_pose_mode,),
            component_scope=config.component_scope,
            action_codec_hash=config.action_codec_hash,
            frame_contract_hash=config.frame_contract_hash,
            layout_hash=artifact.layout_hash,
            normalization_hash=config.normalization_hash,
            minimum_prediction_horizon_steps=horizon,
            maximum_prediction_horizon_steps=horizon,
            minimum_execution_horizon_steps=config.minimum_execution_horizon_steps,
            maximum_execution_horizon_steps=horizon,
            prefix_requirement=RTC_PREFIX_FORBIDDEN,
            supports_prefix_mask=False,
            supports_conditioning_receipt=False,
        )
        provider_id = f"openpi:{artifact.model_id}:{artifact.checkpoint_step}"
        self._capability = RtcModelCapability(
            capability_id=(
                f"{OPENPI_FINAL_CHUNK_RTC_ADAPTER_V1}:"
                f"{config.artifact_hash[:16]}:{self._server_metadata_hash[:16]}"
            ),
            provider_id=provider_id,
            artifact_hash=config.artifact_hash,
            scheduler_modes=(RTC_SCHEDULER_ASYNC_TIMED_PLAN,),
            continuations=(continuation,),
        )
        self._request_spec = RtcProtocolRequestSpec(
            mode=RtcProtocolMode(
                RTC_SCHEDULER_ASYNC_TIMED_PLAN,
                RTC_CONTINUATION_NONE,
            ),
            artifact_hash=config.artifact_hash,
            model_action_space=OPENPI_FINAL_CHUNK_MODEL_ACTION_SPACE_V1,
            model_action_dim=artifact.policy_io_layout.action_dim,
            model_action_discrete_dim=artifact.policy_io_layout.action_discrete_dim,
            action_pose_mode=config.action_pose_mode,
            component_scope=config.component_scope,
            action_codec_hash=config.action_codec_hash,
            frame_contract_hash=config.frame_contract_hash,
            layout_hash=artifact.layout_hash,
            normalization_hash=config.normalization_hash,
            prediction_horizon_steps=horizon,
            minimum_execution_horizon_steps=config.minimum_execution_horizon_steps,
        )
        if config.result_action_representation == OPENPI_RTC_RESULT_TYPED_CANONICAL_V1:
            native_decoder = OpenPIV22NativeRobotActionDecoder(config)
            if config.action_pose_mode == "relative":
                self._decoder = RelativePolicyActionChunkCanonicalizer(
                    decode_absolute_robot_actions=native_decoder,
                    conversion_contract_hash=config.conversion_contract_hash,
                    arm_pose_slices=config.arm_pose_slices,
                )
            else:
                self._decoder = AbsolutePolicyActionChunkCanonicalizer(
                    decode_robot_actions=native_decoder,
                    conversion_contract_hash=config.conversion_contract_hash,
                )
        else:
            self._decoder = OpenPIV22FinalChunkDecoder(config)
        self._v22_adapter = V22DeploymentAdapter(
            artifact.policy_io_layout,
            state_input_mode=artifact.state_input_mode,
        )

    @property
    def rtc_capability(self) -> RtcModelCapability:
        return self._capability

    @property
    def request_spec(self) -> RtcProtocolRequestSpec:
        return self._request_spec

    @property
    def decoder(
        self,
    ) -> (
        OpenPIV22FinalChunkDecoder
        | AbsolutePolicyActionChunkCanonicalizer
        | RelativePolicyActionChunkCanonicalizer
    ):
        return self._decoder

    @property
    def representation_context_builder(
        self,
    ) -> Callable[
        [PolicyObservationView, str], PolicyActionRepresentationContext
    ] | None:
        if (
            self.config.result_action_representation
            != OPENPI_RTC_RESULT_TYPED_CANONICAL_V1
        ):
            return None
        if self.config.action_pose_mode == "relative":
            return self._build_relative_representation_context
        return self._build_absolute_representation_context

    def _build_absolute_representation_context(
        self,
        observation: PolicyObservationView,
        observation_cut_id: str,
    ) -> PolicyActionRepresentationContext:
        sides = self.config.active_sides
        return PolicyActionRepresentationContext(
            source_pose_mode="absolute",
            observation_cut_id=observation_cut_id,
            observation_contract_hash=observation.contract_hash,
            observation_timestamp=float(observation.payload["timestamp"]),
            observation_step_index=int(observation.payload["step_index"]),
            active_sides=sides,
            anchor_policy=ACTION_ANCHOR_POLICY_NONE_V1,
            arm_reference_frame_by_side=self.config.arm_reference_frame_by_side,
            arm_controlled_frame_by_side={
                side: self.config.artifact.policy_io_layout.arm_controlled_frame_by_side[
                    side
                ]
                for side in sides
            },
            arm_anchor_pose_by_side={},
        )

    def _build_relative_representation_context(
        self,
        observation: PolicyObservationView,
        observation_cut_id: str,
    ) -> PolicyActionRepresentationContext:
        layout = self.config.artifact.policy_io_layout
        sides = self.config.active_sides
        return build_policy_action_representation_context(
            observation,
            observation_cut_id,
            source_pose_mode="relative",
            active_sides=sides,
            arm_reference_frame_by_side=self.config.arm_reference_frame_by_side,
            arm_controlled_frame_by_side={
                side: layout.arm_controlled_frame_by_side[side]
                for side in sides
            },
        )

    @property
    def server_metadata_hash(self) -> str:
        return self._server_metadata_hash

    @property
    def provider_metadata(self) -> Mapping[str, Any]:
        """Return current provider metadata for additive protocol negotiation."""

        return deep_freeze(dict(self._transport.metadata))

    @property
    def prediction_horizon_steps(self) -> int:
        return self._prediction_horizon_steps

    def exchange_provider_request(
        self,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Use the existing transport without reinterpreting the raw-RTC path."""

        response = self._transport.infer(dict(payload))
        if not isinstance(response, Mapping):
            raise TypeError("OpenPI provider response must be a mapping.")
        return response

    @property
    def last_transport_lifecycle_receipt(self) -> Any | None:
        """Expose optional transport evidence without making it plan authority."""

        return getattr(self._transport, "last_lifecycle_receipt", None)

    @property
    def last_transport_close_receipt(self) -> Any | None:
        """Expose optional close completion without treating it as worker drain."""

        return getattr(self._transport, "last_close_receipt", None)

    @property
    def last_transport_phase_receipt(self) -> Any | None:
        """Expose optional phase evidence without binding RTC to a transport."""

        return getattr(self._transport, "last_phase_receipt", None)

    def transport_phase_receipts_snapshot(self) -> tuple[Any, ...]:
        """Return optional bounded phase evidence from a capable transport."""

        snapshot = getattr(self._transport, "phase_receipts_snapshot", None)
        if not callable(snapshot):
            return ()
        receipts = snapshot()
        if not isinstance(receipts, tuple):
            raise TypeError("transport phase_receipts_snapshot() must return tuple.")
        return receipts

    def create_client(self) -> ModelAgnosticRtcClient:
        return ModelAgnosticRtcClient(
            backend=self,
            decoder=self._decoder,
            request_spec=self._request_spec,
        )

    def prepare_worker_input(
        self,
        request: PolicyInferenceRequest,
        observation: Any,
    ) -> RtcPreparedModelInput:
        """Prepare one exact input for the shared model-neutral RTC worker."""

        if not isinstance(request, PolicyInferenceRequest):
            raise TypeError("request must be a PolicyInferenceRequest.")
        model_observation = (
            observation.to_deployment_observation()
            if isinstance(observation, PolicyObservationView)
            else observation
        )
        prepared = self.prepare_input(model_observation)
        return RtcPreparedModelInput(
            model_input=prepared,
            model_input_hash=prepared.semantic_content_hash,
        )

    def create_worker_executor(
        self,
        *,
        observe_bundle: Callable[[RtcClientInferenceBundle], None] | None = None,
    ) -> RtcModelClientInferenceExecutor:
        """Build a blocking callable; the caller still owns worker/lifecycle."""

        session_aware = callable(getattr(self._transport, "begin_session", None))
        if session_aware and not self.begin_worker_session():
            raise RuntimeError(
                "OpenPI transport declined a new worker session; prior close "
                "may still be pending."
            )
        return RtcModelClientInferenceExecutor(
            client=self.create_client(),
            prepare_input=self.prepare_worker_input,
            observe_bundle=observe_bundle,
        )

    def begin_worker_session(self) -> bool:
        """Re-arm an optional session-aware transport for a new worker."""

        hook = getattr(self._transport, "begin_session", None)
        if not callable(hook):
            return False
        accepted = hook()
        if not isinstance(accepted, bool):
            raise TypeError("OpenPI transport begin_session() must return bool.")
        return accepted

    def request_cancel(self) -> bool:
        """Forward a quick, idempotent cancellation signal when supported."""

        hook = getattr(self._transport, "request_cancel", None)
        if not callable(hook):
            return False
        accepted = hook()
        if not isinstance(accepted, bool):
            raise TypeError("OpenPI transport request_cancel() must return bool.")
        return accepted

    def prepare_input(self, observation: Any) -> OpenPIRtcPreparedInput:
        typed = self._v22_adapter.observation_from_deployment(observation)
        payload = deployment_observation_to_openpi(
            typed,
            self._observation_config,
        )
        return OpenPIRtcPreparedInput(payload=payload)

    def infer_rtc(
        self,
        envelope: RtcInferenceEnvelopeV2,
        model_input: Any,
    ) -> RtcModelResponseEnvelopeV2:
        if not isinstance(model_input, OpenPIRtcPreparedInput):
            raise TypeError("OpenPI RTC backend requires OpenPIRtcPreparedInput.")
        if model_input.semantic_content_hash != envelope.model_input_hash:
            raise ValueError("Prepared OpenPI input hash does not match the RTC envelope.")
        started = float(self._clock())
        chunk = self._remote_policy.predict_prepared_chunk(
            _prepared_input_payload(model_input)
        )
        current_server_metadata_hash = canonical_content_hash(
            self._transport.metadata,
            domain=OPENPI_RTC_SERVER_HASH_DOMAIN_V1,
        )
        if current_server_metadata_hash != self._server_metadata_hash:
            raise ValueError(
                "OpenPI server metadata changed across a transport reconnect."
            )
        completed = float(self._clock())
        received = float(self._clock())
        if completed < started or received < completed:
            raise ValueError("OpenPI RTC adapter clock regressed during inference.")
        if chunk.horizon != self._prediction_horizon_steps:
            raise ValueError(
                "OpenPI final chunk horizon disagrees with declared model.action_horizon: "
                f"expected {self._prediction_horizon_steps}, got {chunk.horizon}."
            )
        model_output_id = canonical_content_hash(
            {
                "envelope_hash": envelope.semantic_content_hash,
                "actions": chunk.actions,
                "discrete_actions": chunk.discrete_actions,
                "server_metadata_hash": self._server_metadata_hash,
            },
            domain=OPENPI_RTC_OUTPUT_ID_DOMAIN_V1,
        )
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
            artifact_hash=self.config.artifact_hash,
            action_codec_hash=self.config.action_codec_hash,
            frame_contract_hash=self.config.frame_contract_hash,
            layout_hash=self.config.artifact.layout_hash,
            normalization_hash=self.config.normalization_hash,
            model_output_id=model_output_id,
            model_space_actions=chunk.actions,
            model_space_discrete_actions=chunk.discrete_actions,
            effective_delay_steps=envelope.request.forecast_delay_steps,
            execution_horizon_steps=chunk.horizon,
            inference_started_at=started,
            inference_completed_at=completed,
            result_received_at=received,
            conditioning_receipt=None,
        )

    def reset(self) -> None:
        self._remote_policy.reset()


def _prepared_input_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, OpenPIRtcPreparedInput):
        raise TypeError("Remote OpenPI RTC policy requires OpenPIRtcPreparedInput.")
    payload = deep_thaw(value.payload)
    if not isinstance(payload, dict):
        raise TypeError("Prepared OpenPI RTC payload must thaw to a dict.")
    return payload


__all__ = [
    "OPENPI_FINAL_CHUNK_MODEL_ACTION_SPACE_V1",
    "OPENPI_FINAL_CHUNK_RTC_ADAPTER_V1",
    "OPENPI_RTC_RESULT_LEGACY_NORMALIZED_V1",
    "OPENPI_RTC_RESULT_TYPED_CANONICAL_V1",
    "OpenPIFinalChunkRtcBackend",
    "OpenPIFinalChunkRtcConfig",
    "OpenPIRtcPreparedInput",
    "OpenPIV22FinalChunkDecoder",
    "OpenPIV22NativeRobotActionDecoder",
    "openpi_arm_pose_slices",
    "openpi_arm_reference_frames",
    "openpi_absolute_conversion_contract_hash",
    "openpi_relative_conversion_contract_hash",
    "openpi_rtc_artifact_hash",
    "openpi_rtc_component_scope",
    "openpi_v22_legacy_codec_hash",
    "openpi_v22_native_codec_hash",
    "openpi_v22_relative_native_codec_hash",
]
