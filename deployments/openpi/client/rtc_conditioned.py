"""OpenPI provider binding for additive model-RTC v3.

This module reuses the existing OpenPI observation transport and output
decoder.  It only adds canonical-to-external conditioning encoding and one
versioned provider operation.  It owns no worker, plan admission, safety,
target, dispatch, or hardware lifecycle.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.model_continuation import (
    MODEL_CONTINUATION_SAMPLER_CLAMPED_V1,
    ModelContinuationContract,
    normalize_model_continuation_contract,
)
from dataset_contracts.policy_artifact import PolicyArtifactContract
from deployments.client.rtc_conditioned_client import (
    RtcConditionedModelClientV3,
)
from deployments.client.rtc_conditioning import (
    RtcConditioningCodecIdentity,
    rtc_binary_mask_hash,
    rtc_float32_array_hash,
)
from deployments.client.rtc_conditioning_protocol import (
    RTC_CONDITIONING_TRANSFORM_PROVIDER_V1,
    RtcConditionedInferenceEnvelopeV3,
    RtcConditionedModelResponseV3,
    RtcConditioningConsumptionReceiptV3,
    RtcModelContinuationCapabilityV3,
    validate_rtc_model_continuation_capability_v3_binding,
)
from deployments.client.rtc_executor import (
    RtcConditionedModelClientInferenceExecutorV3,
    RtcPreparedModelInput,
)
from robot_motion.deploy.canonical_policy_action import CanonicalPolicyAction
from robot_motion.deploy.inference import PolicyInferenceRequest
from robot_motion.deploy.observation_view import PolicyObservationView
from robot_motion.deploy.rtc.conditioning import (
    RTC_CONDITIONING_MODE_PREFIX_V1,
    RTC_CONDITIONING_MODE_UNCONDITIONED_BOOTSTRAP_V1,
    CanonicalRtcConditioningBundle,
)
from robot_motion.relative_action import (
    ACTION_POSE_MODE_ABSOLUTE,
    ACTION_POSE_MODE_RELATIVE,
    convert_arm_pose_chunk,
)
from robot_motion.runtime.immutable import deep_thaw

from .rtc import (
    OPENPI_RTC_RESULT_TYPED_CANONICAL_V1,
    OpenPIFinalChunkRtcBackend,
    OpenPIRtcPreparedInput,
)


OPENPI_CONDITIONED_RTC_WIRE_V1 = "openpi.conditioned_rtc.v1"
OPENPI_CONDITIONED_RTC_OPERATION_V1 = "infer_conditioned_rtc_v1"
OPENPI_CONDITIONED_RTC_METADATA_KEY_V1 = "rtc_model_continuation_v1"
OPENPI_CONDITIONED_RTC_RECEIPT_KEY_V1 = "rtc_conditioning_receipt"
OPENPI_CONDITIONING_CODEC_IMPLEMENTATION_V1 = (
    "dexteleop.openpi.v22_conditioning_codec.v1"
)
OPENPI_CONDITIONING_CODEC_PARAMETERS_HASH_DOMAIN_V1 = (
    "dexteleop.openpi.conditioning_codec_parameters.v1"
)
OPENPI_CONDITIONED_OUTPUT_ID_HASH_DOMAIN_V1 = (
    "dexteleop.openpi.conditioned_model_output.v1"
)
OPENPI_CONDITIONED_PROVIDER_CAPABILITY_SCHEMA_V1 = (
    "openpi.conditioned_rtc.provider_capability.v1"
)


def _nonempty(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _sha256(value: object, *, name: str) -> str:
    normalized = _nonempty(value, name=name)
    if (
        not normalized.startswith("sha256:")
        or len(normalized) != 71
        or any(
            character not in "0123456789abcdef"
            for character in normalized[7:]
        )
    ):
        raise ValueError(f"{name} must be a lowercase sha256 digest.")
    return normalized


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    normalized = int(value)
    if normalized != value or normalized <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return normalized


def openpi_conditioned_provider_capability_hash(
    provider: Mapping[str, Any],
) -> str:
    """Portable identity for the loaded provider facts, excluding its hash."""

    if not isinstance(provider, Mapping):
        raise TypeError("provider capability must be a mapping.")
    schema = _nonempty(
        provider.get("provider_capability_schema"),
        name="provider_capability_schema",
    )
    if schema != OPENPI_CONDITIONED_PROVIDER_CAPABILITY_SCHEMA_V1:
        raise ValueError("OpenPI provider capability schema mismatch.")
    if provider.get("wire_protocol") != OPENPI_CONDITIONED_RTC_WIRE_V1:
        raise ValueError("OpenPI provider capability wire protocol mismatch.")
    if provider.get("enabled") is not True:
        raise ValueError("OpenPI provider capability must be enabled.")
    if (
        provider.get("conditioning_transform_owner")
        != RTC_CONDITIONING_TRANSFORM_PROVIDER_V1
    ):
        raise ValueError(
            "OpenPI provider capability must own its conditioning transform."
        )
    supports_bootstrap = provider.get("supports_unconditioned_bootstrap")
    if not isinstance(supports_bootstrap, bool):
        raise ValueError(
            "OpenPI provider bootstrap capability must be explicitly boolean."
        )
    payload = {
        "schema": schema,
        "wire_protocol": OPENPI_CONDITIONED_RTC_WIRE_V1,
        "enabled": True,
        "conditioning_transform_owner": (
            RTC_CONDITIONING_TRANSFORM_PROVIDER_V1
        ),
        "external_action_dim": _positive_int(
            provider.get("external_action_dim"),
            name="OpenPI external_action_dim",
        ),
        "internal_sampler_action_dim": _positive_int(
            provider.get("internal_sampler_action_dim"),
            name="OpenPI internal_sampler_action_dim",
        ),
        "prediction_horizon_steps": _positive_int(
            provider.get("prediction_horizon_steps"),
            name="OpenPI prediction_horizon_steps",
        ),
        "sampler_constraint": _nonempty(
            provider.get("sampler_constraint"),
            name="OpenPI sampler_constraint",
        ),
        "supports_unconditioned_bootstrap": supports_bootstrap,
        "continuation_implementation_id": _nonempty(
            provider.get("continuation_implementation_id"),
            name="OpenPI continuation_implementation_id",
        ),
        "continuation_implementation_parameters_hash": _sha256(
            provider.get("continuation_implementation_parameters_hash"),
            name="OpenPI continuation_implementation_parameters_hash",
        ),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True)
class OpenPIConditionedRtcConfig:
    """Runtime choices for one artifact-bound OpenPI provider."""

    selected_prefix_schedule: str
    sampler_constraint: str = MODEL_CONTINUATION_SAMPLER_CLAMPED_V1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selected_prefix_schedule",
            _nonempty(
                self.selected_prefix_schedule,
                name="selected_prefix_schedule",
            ),
        )
        constraint = _nonempty(
            self.sampler_constraint,
            name="sampler_constraint",
        )
        object.__setattr__(self, "sampler_constraint", constraint)

    @classmethod
    def from_artifact(
        cls,
        artifact: PolicyArtifactContract,
        *,
        selected_prefix_schedule: str,
        sampler_constraint: str = MODEL_CONTINUATION_SAMPLER_CLAMPED_V1,
    ) -> "OpenPIConditionedRtcConfig":
        if not isinstance(artifact, PolicyArtifactContract):
            raise TypeError("artifact must be a PolicyArtifactContract.")
        continuation = artifact.model_continuation
        if continuation is None:
            raise ValueError(
                "OpenPI model-RTC requires an artifact-bound continuation "
                "contract."
            )
        runtime_contract = normalize_model_continuation_contract(
            continuation
        ).runtime
        if sampler_constraint not in runtime_contract.sampler_constraints:
            raise ValueError(
                "sampler_constraint is not supported by the continuation "
                "contract."
            )
        return cls(
            selected_prefix_schedule=selected_prefix_schedule,
            sampler_constraint=sampler_constraint,
        )


class OpenPIV22ConditioningCodec:
    """Project canonical plan rows into the artifact's external action form."""

    def __init__(
        self,
        *,
        base_backend: OpenPIFinalChunkRtcBackend,
        continuation_contract: ModelContinuationContract,
    ) -> None:
        if not isinstance(base_backend, OpenPIFinalChunkRtcBackend):
            raise TypeError(
                "base_backend must be an OpenPIFinalChunkRtcBackend."
            )
        runtime_contract = normalize_model_continuation_contract(
            continuation_contract
        ).runtime
        config = base_backend.config
        artifact = config.artifact
        layout = artifact.policy_io_layout
        if layout.action_discrete_dim:
            raise ValueError(
                "OpenPI model-RTC v3 does not yet condition discrete action "
                "tokens."
            )
        expected = {
            "model_action_dim": layout.action_dim,
            "model_action_discrete_dim": layout.action_discrete_dim,
            "action_codec_hash": config.action_codec_hash,
            "layout_hash": artifact.layout_hash,
            "frame_contract_hash": config.frame_contract_hash,
            "normalization_hash": config.normalization_hash,
        }
        for name, value in expected.items():
            if getattr(runtime_contract, name) != value:
                raise ValueError(
                    f"OpenPI continuation contract {name} mismatch."
                )
        if (
            config.action_pose_mode
            not in runtime_contract.supported_action_pose_modes
        ):
            raise ValueError(
                "OpenPI action pose mode is unsupported by the continuation "
                "contract."
            )
        parameters_hash = canonical_content_hash(
            {
                "implementation_id": (
                    OPENPI_CONDITIONING_CODEC_IMPLEMENTATION_V1
                ),
                "artifact_hash": config.artifact_hash,
                "layout_hash": artifact.layout_hash,
                "action_pose_mode": config.action_pose_mode,
                "active_sides": config.active_sides,
                "conversion_contract_hash": config.conversion_contract_hash,
            },
            domain=OPENPI_CONDITIONING_CODEC_PARAMETERS_HASH_DOMAIN_V1,
        )
        self.identity = RtcConditioningCodecIdentity(
            implementation_id=OPENPI_CONDITIONING_CODEC_IMPLEMENTATION_V1,
            implementation_parameters_hash=parameters_hash,
            action_codec_hash=config.action_codec_hash,
            layout_hash=artifact.layout_hash,
            frame_contract_hash=config.frame_contract_hash,
            normalization_hash=config.normalization_hash,
            artifact_action_dim=layout.action_dim,
            external_action_dim=layout.action_dim,
            round_trip_atol=1e-5,
            round_trip_rtol=1e-5,
        )
        self._base_backend = base_backend
        self._layout = layout

    def _absolute_rows(
        self,
        actions: Sequence[CanonicalPolicyAction],
        *,
        bundle: CanonicalRtcConditioningBundle,
    ) -> np.ndarray:
        rows: list[np.ndarray] = []
        context = bundle.action_representation_context
        for canonical in actions:
            action = canonical.action
            for side in context.active_sides:
                if side not in action.arm_pose_by_side:
                    raise ValueError(
                        f"Canonical conditioning is missing {side!r} arm pose."
                    )
                if (
                    action.arm_frame(side)
                    != context.arm_reference_frame_by_side[side]
                    or action.arm_controlled_frame(side)
                    != context.arm_controlled_frame_by_side[side]
                ):
                    raise ValueError(
                        "Canonical conditioning frame differs from its "
                        f"representation context for {side!r}."
                    )
            end_effector_values = {
                side: target.values
                for side, target in (
                    action.end_effector_targets_by_side.items()
                )
            }
            packed = self._layout.pack_action(
                arm_pose_by_side=action.arm_pose_by_side,
                end_effector_by_side=end_effector_values,
                strict=True,
            )
            if packed.discrete.size:
                raise ValueError(
                    "OpenPI model-RTC cannot coerce discrete conditioning "
                    "tokens into float actions."
                )
            rows.append(np.asarray(packed.continuous, dtype=np.float32))
        if not rows:
            return np.zeros(
                (0, self.identity.artifact_action_dim),
                dtype=np.float32,
            )
        matrix = np.stack(rows, axis=0)
        if not np.all(np.isfinite(matrix)):
            raise ValueError(
                "Canonical OpenPI conditioning does not cover every artifact "
                "action component."
            )
        return matrix

    def canonical_to_artifact_actions(
        self,
        actions: Sequence[CanonicalPolicyAction],
        *,
        bundle: CanonicalRtcConditioningBundle,
    ) -> np.ndarray:
        absolute = self._absolute_rows(actions, bundle=bundle)
        if not absolute.size:
            return absolute
        context = bundle.action_representation_context
        if context.source_pose_mode == ACTION_POSE_MODE_ABSOLUTE:
            return absolute
        if context.source_pose_mode != ACTION_POSE_MODE_RELATIVE:
            raise ValueError(
                "Unsupported OpenPI conditioning action pose mode "
                f"{context.source_pose_mode!r}."
            )
        return convert_arm_pose_chunk(
            absolute,
            anchors_by_side=context.arm_anchor_pose_by_side,
            source_mode=ACTION_POSE_MODE_ABSOLUTE,
            target_mode=ACTION_POSE_MODE_RELATIVE,
            arm_pose_slices=self._base_backend.config.arm_pose_slices,
            active_sides=context.active_sides,
        )

    def artifact_to_canonical_actions(
        self,
        artifact_actions: np.ndarray,
        *,
        bundle: CanonicalRtcConditioningBundle,
    ) -> Sequence[CanonicalPolicyAction]:
        artifact = np.asarray(artifact_actions, dtype=np.float32)
        expected = self.canonical_to_artifact_actions(
            bundle.canonical_actions,
            bundle=bundle,
        )
        if artifact.shape != expected.shape or not np.allclose(
            artifact,
            expected,
            rtol=self.identity.round_trip_rtol,
            atol=self.identity.round_trip_atol,
        ):
            raise ValueError(
                "OpenPI artifact rows do not reconstruct the canonical bundle."
            )
        return bundle.canonical_actions

    def artifact_to_external_actions(
        self,
        artifact_actions: np.ndarray,
        *,
        bundle: CanonicalRtcConditioningBundle,
    ) -> np.ndarray:
        del bundle
        return np.asarray(artifact_actions, dtype=np.float32).copy()

    def external_to_artifact_actions(
        self,
        external_actions: np.ndarray,
        *,
        bundle: CanonicalRtcConditioningBundle,
    ) -> np.ndarray:
        del bundle
        return np.asarray(external_actions, dtype=np.float32).copy()


class OpenPIConditionedRtcBackendV3:
    """Loaded-instance OpenPI provider implementing only model-RTC v3."""

    def __init__(
        self,
        *,
        base_backend: OpenPIFinalChunkRtcBackend,
        config: OpenPIConditionedRtcConfig,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(base_backend, OpenPIFinalChunkRtcBackend):
            raise TypeError(
                "base_backend must be an OpenPIFinalChunkRtcBackend."
            )
        if not isinstance(config, OpenPIConditionedRtcConfig):
            raise TypeError("config must be an OpenPIConditionedRtcConfig.")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        if (
            base_backend.config.result_action_representation
            != OPENPI_RTC_RESULT_TYPED_CANONICAL_V1
        ):
            raise ValueError(
                "OpenPI model-RTC v3 requires typed canonical result decoding."
            )
        artifact_contract = base_backend.config.artifact.model_continuation
        if artifact_contract is None:
            raise ValueError(
                "OpenPI model-RTC requires an artifact-bound continuation "
                "contract."
            )
        source_contract = artifact_contract
        contract = normalize_model_continuation_contract(
            source_contract
        ).runtime
        horizon = base_backend.prediction_horizon_steps
        if contract.prediction_horizon_steps != horizon:
            raise ValueError(
                "OpenPI continuation contract prediction horizon mismatch."
            )
        metadata = _mapping(
            deep_thaw(base_backend.provider_metadata),
            name="OpenPI provider metadata",
        )
        provider = _mapping(
            metadata.get(OPENPI_CONDITIONED_RTC_METADATA_KEY_V1),
            name=OPENPI_CONDITIONED_RTC_METADATA_KEY_V1,
        )
        provider_capability_hash = _sha256(
            provider.get("provider_capability_hash"),
            name="OpenPI provider_capability_hash",
        )
        if (
            provider_capability_hash
            != openpi_conditioned_provider_capability_hash(provider)
        ):
            raise ValueError("OpenPI provider capability hash mismatch.")
        if provider.get("wire_protocol") != OPENPI_CONDITIONED_RTC_WIRE_V1:
            raise ValueError("OpenPI conditioned RTC wire protocol mismatch.")
        if provider.get("enabled") is not True:
            raise ValueError("OpenPI conditioned RTC provider is not enabled.")
        if (
            provider.get("conditioning_transform_owner")
            != RTC_CONDITIONING_TRANSFORM_PROVIDER_V1
        ):
            raise ValueError(
                "OpenPI must own its external-to-internal conditioning "
                "transform."
            )
        external_dim = _positive_int(
            provider.get("external_action_dim"),
            name="OpenPI external_action_dim",
        )
        internal_dim = _positive_int(
            provider.get("internal_sampler_action_dim"),
            name="OpenPI internal_sampler_action_dim",
        )
        provider_horizon = _positive_int(
            provider.get("prediction_horizon_steps"),
            name="OpenPI prediction_horizon_steps",
        )
        if external_dim != contract.model_action_dim:
            raise ValueError("OpenPI external action dimension mismatch.")
        if provider_horizon != horizon:
            raise ValueError("OpenPI provider action horizon mismatch.")
        if provider.get("sampler_constraint") != config.sampler_constraint:
            raise ValueError("OpenPI sampler constraint mismatch.")
        if (
            provider.get("continuation_implementation_id")
            != contract.continuation_implementation_id
        ):
            raise ValueError(
                "OpenPI continuation implementation identity mismatch."
            )
        if (
            provider.get("continuation_implementation_parameters_hash")
            != contract.continuation_implementation_parameters_hash
        ):
            raise ValueError(
                "OpenPI continuation implementation parameters mismatch."
            )
        supports_bootstrap = provider.get(
            "supports_unconditioned_bootstrap"
        )
        if not isinstance(supports_bootstrap, bool):
            raise ValueError(
                "OpenPI bootstrap capability must be explicitly boolean."
            )
        if (
            contract.supports_unconditioned_bootstrap
            and not supports_bootstrap
        ):
            raise ValueError(
                "OpenPI provider does not support contract-required bootstrap."
            )
        self._base_backend = base_backend
        self._config = config
        self._continuation_contract = source_contract
        self._clock = clock
        self._provider_capability_hash = provider_capability_hash
        self._codec = OpenPIV22ConditioningCodec(
            base_backend=base_backend,
            continuation_contract=source_contract,
        )
        self._capability = RtcModelContinuationCapabilityV3(
            artifact_hash=base_backend.config.artifact_hash,
            continuation_contract_hash=source_contract.contract_hash,
            codec_identity_hash=self._codec.identity.semantic_content_hash,
            continuation_implementation_id=(
                contract.continuation_implementation_id
            ),
            continuation_implementation_parameters_hash=(
                contract.continuation_implementation_parameters_hash
            ),
            sampler_constraint=config.sampler_constraint,
            selected_prefix_schedule=config.selected_prefix_schedule,
            conditioning_transform_owner=(
                RTC_CONDITIONING_TRANSFORM_PROVIDER_V1
            ),
            external_action_dim=external_dim,
            internal_sampler_action_dim=internal_dim,
            supports_unconditioned_bootstrap=(
                bool(supports_bootstrap)
                and contract.supports_unconditioned_bootstrap
            ),
        )
        validate_rtc_model_continuation_capability_v3_binding(
            self._capability,
            continuation_contract=source_contract,
            artifact_hash=base_backend.config.artifact_hash,
            codec_identity_hash=self._codec.identity.semantic_content_hash,
            selected_prefix_schedule=config.selected_prefix_schedule,
        )

    @property
    def rtc_continuation_capability(
        self,
    ) -> RtcModelContinuationCapabilityV3:
        return self._capability

    @property
    def conditioning_codec(self) -> OpenPIV22ConditioningCodec:
        return self._codec

    @property
    def continuation_contract(self) -> ModelContinuationContract:
        return self._continuation_contract

    @property
    def representation_context_builder(self):
        return self._base_backend.representation_context_builder

    def request_cancel(self) -> bool:
        return self._base_backend.request_cancel()

    def begin_worker_session(self) -> bool:
        return self._base_backend.begin_worker_session()

    def create_client(self) -> RtcConditionedModelClientV3:
        return RtcConditionedModelClientV3(
            backend=self,
            codec=self._codec,
            decoder=self._base_backend.decoder,
        )

    def prepare_worker_input(
        self,
        request: PolicyInferenceRequest,
        observation: Any,
    ) -> RtcPreparedModelInput:
        return self._base_backend.prepare_worker_input(request, observation)

    def create_worker_executor(
        self,
        *,
        observe_bundle: Callable[[Any], None] | None = None,
    ) -> RtcConditionedModelClientInferenceExecutorV3:
        self.begin_worker_session()
        return RtcConditionedModelClientInferenceExecutorV3(
            client=self.create_client(),
            prepare_input=self.prepare_worker_input,
            observe_bundle=observe_bundle,
        )

    def infer_conditioned_rtc_v3(
        self,
        envelope: RtcConditionedInferenceEnvelopeV3,
        model_input: Any,
    ) -> RtcConditionedModelResponseV3:
        if not isinstance(model_input, OpenPIRtcPreparedInput):
            raise TypeError(
                "OpenPI model-RTC requires OpenPIRtcPreparedInput."
            )
        if model_input.semantic_content_hash != envelope.model_input_hash:
            raise ValueError(
                "Prepared OpenPI input hash does not match the v3 envelope."
            )
        encoded = envelope.encoded_conditioning
        indices = encoded.receipt.selected_generation_indices
        conditioning_mode = (
            RTC_CONDITIONING_MODE_PREFIX_V1
            if indices
            else RTC_CONDITIONING_MODE_UNCONDITIONED_BOOTSTRAP_V1
        )
        context = {
            "envelope_hash": envelope.semantic_content_hash,
            "request_hash": envelope.request.semantic_content_hash,
            "capability_hash": envelope.capability.semantic_content_hash,
            "encoding_receipt_hash": (
                encoded.receipt.semantic_content_hash
            ),
            "model_input_hash": envelope.model_input_hash,
            "provider_capability_hash": self._provider_capability_hash,
        }
        wire_request = {
            "__openpi_request__": {
                "wire_protocol": OPENPI_CONDITIONED_RTC_WIRE_V1,
                "operation": OPENPI_CONDITIONED_RTC_OPERATION_V1,
            },
            "observation": deep_thaw(model_input.payload),
            "conditioning_actions": np.asarray(
                encoded.external_prefix,
                dtype=np.float32,
            ),
            "prefix_mask": np.asarray(
                encoded.prefix_mask,
                dtype=np.float32,
            ),
            "conditioning_mode": conditioning_mode,
            "receipt_context": context,
        }
        started = float(self._clock())
        response = _mapping(
            self._base_backend.exchange_provider_request(wire_request),
            name="OpenPI conditioned response",
        )
        completed = float(self._clock())
        received = float(self._clock())
        if started > completed or completed > received:
            raise ValueError("OpenPI conditioned RTC clock regressed.")

        provider_receipt = _mapping(
            response.get(OPENPI_CONDITIONED_RTC_RECEIPT_KEY_V1),
            name=OPENPI_CONDITIONED_RTC_RECEIPT_KEY_V1,
        )
        if (
            provider_receipt.get("wire_protocol")
            != OPENPI_CONDITIONED_RTC_WIRE_V1
        ):
            raise ValueError("OpenPI receipt wire protocol mismatch.")
        if provider_receipt.get("conditioning_mode") != conditioning_mode:
            raise ValueError("OpenPI receipt conditioning mode mismatch.")
        if provider_receipt.get("receipt_context") != context:
            raise ValueError("OpenPI receipt causal context mismatch.")
        receipt_provider_capability_hash = _sha256(
            provider_receipt.get("provider_capability_hash"),
            name="receipt provider_capability_hash",
        )
        if receipt_provider_capability_hash != self._provider_capability_hash:
            raise ValueError("OpenPI receipt provider capability mismatch.")
        receipt_implementation_id = _nonempty(
            provider_receipt.get("continuation_implementation_id"),
            name="receipt continuation_implementation_id",
        )
        if (
            receipt_implementation_id
            != envelope.capability.continuation_implementation_id
        ):
            raise ValueError(
                "OpenPI receipt continuation implementation mismatch."
            )
        receipt_implementation_parameters_hash = _sha256(
            provider_receipt.get(
                "continuation_implementation_parameters_hash"
            ),
            name="receipt continuation_implementation_parameters_hash",
        )
        if (
            receipt_implementation_parameters_hash
            != envelope.capability.continuation_implementation_parameters_hash
        ):
            raise ValueError(
                "OpenPI receipt continuation implementation parameters "
                "mismatch."
            )
        receipt_sampler_constraint = _nonempty(
            provider_receipt.get("sampler_constraint"),
            name="receipt sampler_constraint",
        )
        if receipt_sampler_constraint != envelope.capability.sampler_constraint:
            raise ValueError("OpenPI receipt sampler constraint mismatch.")
        external_hash = _sha256(
            provider_receipt.get("consumed_external_prefix_hash"),
            name="consumed_external_prefix_hash",
        )
        internal_hash = _sha256(
            provider_receipt.get("consumed_internal_prefix_hash"),
            name="consumed_internal_prefix_hash",
        )
        mask_hash = _sha256(
            provider_receipt.get("consumed_mask_hash"),
            name="consumed_mask_hash",
        )
        if tuple(provider_receipt.get("consumed_external_shape", ())) != (
            envelope.request.prediction_horizon_steps,
            envelope.capability.external_action_dim,
        ):
            raise ValueError("OpenPI consumed external prefix shape mismatch.")
        if tuple(provider_receipt.get("consumed_internal_shape", ())) != (
            envelope.request.prediction_horizon_steps,
            envelope.capability.internal_sampler_action_dim,
        ):
            raise ValueError("OpenPI consumed internal prefix shape mismatch.")
        if external_hash != rtc_float32_array_hash(encoded.external_prefix):
            raise ValueError("OpenPI consumed a different external prefix.")
        if mask_hash != rtc_binary_mask_hash(encoded.prefix_mask):
            raise ValueError("OpenPI consumed a different prefix mask.")
        consumed_indices = tuple(
            int(index)
            for index in provider_receipt.get(
                "consumed_generation_indices",
                (),
            )
        )
        if consumed_indices != indices:
            raise ValueError(
                "OpenPI consumed generation indices differ from the encoding."
            )
        consumption = RtcConditioningConsumptionReceiptV3(
            envelope_hash=envelope.semantic_content_hash,
            request_hash=envelope.request.semantic_content_hash,
            capability_hash=envelope.capability.semantic_content_hash,
            conditioning_binding_hash=(
                encoded.receipt.conditioning_binding_hash
            ),
            encoding_receipt_hash=encoded.receipt.semantic_content_hash,
            model_input_hash=envelope.model_input_hash,
            consumed_external_prefix_hash=external_hash,
            consumed_internal_prefix_hash=internal_hash,
            consumed_mask_hash=mask_hash,
            consumed_generation_indices=consumed_indices,
            sampler_constraint=receipt_sampler_constraint,
            continuation_implementation_id=receipt_implementation_id,
            continuation_implementation_parameters_hash=(
                receipt_implementation_parameters_hash
            ),
            effective_delay_steps=envelope.request.forecast_delay_steps,
            prediction_horizon_steps=(
                envelope.request.prediction_horizon_steps
            ),
        )
        action_key = self._base_backend.config.artifact.action_key
        actions = np.asarray(response.get(action_key), dtype=np.float32)
        expected_shape = (
            envelope.request.prediction_horizon_steps,
            envelope.capability.external_action_dim,
        )
        if actions.shape != expected_shape or not np.all(np.isfinite(actions)):
            raise ValueError(
                "OpenPI conditioned output shape/content mismatch: "
                f"expected {expected_shape}, got {actions.shape}."
            )
        output_id = canonical_content_hash(
            {
                "envelope_hash": envelope.semantic_content_hash,
                "actions": actions,
                "provider_receipt": dict(provider_receipt),
                "server_metadata_hash": (
                    self._base_backend.server_metadata_hash
                ),
            },
            domain=OPENPI_CONDITIONED_OUTPUT_ID_HASH_DOMAIN_V1,
        )
        return RtcConditionedModelResponseV3(
            envelope_hash=envelope.semantic_content_hash,
            request_hash=envelope.request.semantic_content_hash,
            capability_hash=envelope.capability.semantic_content_hash,
            model_output_id=output_id,
            model_space_actions=actions,
            inference_started_at=started,
            inference_completed_at=completed,
            result_received_at=received,
            consumption_receipt=consumption,
        )


__all__ = [
    "OPENPI_CONDITIONED_RTC_METADATA_KEY_V1",
    "OPENPI_CONDITIONED_RTC_OPERATION_V1",
    "OPENPI_CONDITIONED_PROVIDER_CAPABILITY_SCHEMA_V1",
    "OPENPI_CONDITIONED_RTC_RECEIPT_KEY_V1",
    "OPENPI_CONDITIONED_RTC_WIRE_V1",
    "OpenPIConditionedRtcBackendV3",
    "OpenPIConditionedRtcConfig",
    "OpenPIV22ConditioningCodec",
    "openpi_conditioned_provider_capability_hash",
]
