"""Auditable inference-only model-RTC binding for one OpenPI artifact.

The loaded OpenPI provider advertises sampler facts that cannot be inferred
from a checkpoint name.  This module intersects those facts with an explicit
DexTeleop deployment representation and produces a new artifact contract plus
an immutable receipt.  It never edits the source artifact, loads a model,
starts a worker, or owns plan/execution state.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any, Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.model_continuation import (
    MODEL_CONTINUATION_FLOW_INPAINT_V1,
    MODEL_CONTINUATION_ORIGIN_INFERENCE_ONLY_V1,
    MODEL_CONTINUATION_ORIGIN_TRAINED_PREFIX_V1,
    RUNTIME_PREFIX_PRODUCER_LOCAL_OVERLAP_V1,
    ModelContinuationDataContractV2,
    ModelContinuationRuntimeContractV2,
)
from dataset_contracts.policy_artifact import PolicyArtifactContract

from .rtc import (
    OPENPI_RTC_RESULT_TYPED_CANONICAL_V1,
    OpenPIFinalChunkRtcConfig,
    openpi_rtc_artifact_hash,
)
from .rtc_conditioned import (
    OPENPI_CONDITIONED_PROVIDER_CAPABILITY_SCHEMA_V1,
    OPENPI_CONDITIONED_RTC_WIRE_V1,
    openpi_conditioned_provider_capability_hash,
)

OPENPI_INFERENCE_ONLY_RTC_BINDING_SCHEMA_V1 = (
    "openpi_inference_only_rtc_artifact_binding_v1"
)
OPENPI_INFERENCE_ONLY_RTC_BINDING_HASH_DOMAIN_V1 = (
    "dexteleop.openpi.inference_only_rtc_artifact_binding.v1"
)
OPENPI_TRAINED_RTC_SAMPLER_BINDING_SCHEMA_V1 = (
    "openpi_trained_rtc_sampler_binding_v1"
)
OPENPI_TRAINED_RTC_SAMPLER_BINDING_HASH_DOMAIN_V1 = (
    "dexteleop.openpi.trained_rtc_sampler_binding.v1"
)
OPENPI_RTC_TRAINED_NUM_STEPS_V1 = 10
OPENPI_RTC_DEPLOYMENT_NUM_STEPS_V1 = (8, 10)


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer.")
    normalized = int(value)
    if normalized != value or normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return normalized


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _sha256_json(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class OpenPIInferenceOnlyRtcBindingSpec:
    """Explicit deployment support range selected for one bound artifact."""

    minimum_forecast_delay_steps: int
    maximum_forecast_delay_steps: int
    minimum_prefix_steps: int
    maximum_prefix_steps: int

    def __post_init__(self) -> None:
        for name in (
            "minimum_forecast_delay_steps",
            "maximum_forecast_delay_steps",
            "minimum_prefix_steps",
            "maximum_prefix_steps",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name),
            )


@dataclass(frozen=True)
class OpenPIInferenceOnlyRtcArtifactBinding:
    """A derived artifact and proof of the facts used to bind it."""

    source_artifact: PolicyArtifactContract
    bound_artifact: PolicyArtifactContract
    provider_capability_hash: str
    continuation_contract_hash: str
    source_artifact_hash: str
    bound_artifact_hash: str
    schema: str = OPENPI_INFERENCE_ONLY_RTC_BINDING_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema != OPENPI_INFERENCE_ONLY_RTC_BINDING_SCHEMA_V1:
            raise ValueError("Unsupported OpenPI model-RTC binding schema.")
        if not isinstance(self.source_artifact, PolicyArtifactContract):
            raise TypeError("source_artifact must be a PolicyArtifactContract.")
        if not isinstance(self.bound_artifact, PolicyArtifactContract):
            raise TypeError("bound_artifact must be a PolicyArtifactContract.")
        source_hash = openpi_rtc_artifact_hash(self.source_artifact)
        bound_hash = openpi_rtc_artifact_hash(self.bound_artifact)
        if self.source_artifact_hash != source_hash:
            raise ValueError("OpenPI source artifact hash mismatch.")
        if self.bound_artifact_hash != bound_hash:
            raise ValueError("OpenPI bound artifact hash mismatch.")
        continuation = self.bound_artifact.model_continuation
        if not isinstance(continuation, ModelContinuationDataContractV2):
            raise ValueError(
                "OpenPI bound artifact requires a v2 continuation contract."
            )
        if (
            continuation.conditioning_origin
            != MODEL_CONTINUATION_ORIGIN_INFERENCE_ONLY_V1
            or continuation.training is not None
        ):
            raise ValueError(
                "OpenPI binding must remain inference-only with training=null."
            )
        if self.continuation_contract_hash != continuation.contract_hash:
            raise ValueError("OpenPI continuation contract hash mismatch.")
        if self.source_artifact.model_continuation is not None:
            raise ValueError(
                "OpenPI source artifact must remain unbound and immutable."
            )
        if replace(
            self.bound_artifact,
            model_continuation=None,
        ).to_dict() != self.source_artifact.to_dict():
            raise ValueError(
                "OpenPI bound artifact differs outside model_continuation."
            )
        object.__setattr__(
            self,
            "provider_capability_hash",
            _required(
                self.provider_capability_hash,
                name="provider_capability_hash",
            ),
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "schema": self.schema,
                "source_artifact_hash": self.source_artifact_hash,
                "bound_artifact_hash": self.bound_artifact_hash,
                "provider_capability_hash": self.provider_capability_hash,
                "continuation_contract_hash": (
                    self.continuation_contract_hash
                ),
                "conditioning_origin": (
                    MODEL_CONTINUATION_ORIGIN_INFERENCE_ONLY_V1
                ),
                "training": None,
            },
            domain=OPENPI_INFERENCE_ONLY_RTC_BINDING_HASH_DOMAIN_V1,
        )

    def receipt_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "model_id": self.bound_artifact.model_id,
            "checkpoint_step": self.bound_artifact.checkpoint_step,
            "source_artifact_hash": self.source_artifact_hash,
            "bound_artifact_hash": self.bound_artifact_hash,
            "provider_capability_hash": self.provider_capability_hash,
            "continuation_contract_hash": self.continuation_contract_hash,
            "conditioning_origin": (
                MODEL_CONTINUATION_ORIGIN_INFERENCE_ONLY_V1
            ),
            "training": None,
            "semantic_content_hash": self.semantic_content_hash,
        }


@dataclass(frozen=True)
class OpenPITrainedRtcSamplerBinding:
    """A deployment-only sampler derivation of one trained RTC artifact."""

    source_artifact: PolicyArtifactContract
    bound_artifact: PolicyArtifactContract
    provider_capability_hash: str
    source_artifact_hash: str
    bound_artifact_hash: str
    source_continuation_contract_hash: str
    bound_continuation_contract_hash: str
    source_implementation_parameters_hash: str
    bound_implementation_parameters_hash: str
    trained_num_steps: int
    deployment_num_steps: int
    implementation_parameters: Mapping[str, Any]
    schema: str = OPENPI_TRAINED_RTC_SAMPLER_BINDING_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema != OPENPI_TRAINED_RTC_SAMPLER_BINDING_SCHEMA_V1:
            raise ValueError("Unsupported OpenPI trained RTC sampler binding schema.")
        source = self.source_artifact.model_continuation
        bound = self.bound_artifact.model_continuation
        if not isinstance(source, ModelContinuationDataContractV2) or not isinstance(
            bound, ModelContinuationDataContractV2
        ):
            raise ValueError("OpenPI trained sampler binding requires v2 contracts.")
        for name, continuation in (("source", source), ("bound", bound)):
            if (
                continuation.conditioning_origin
                != MODEL_CONTINUATION_ORIGIN_TRAINED_PREFIX_V1
                or continuation.training is None
            ):
                raise ValueError(
                    f"OpenPI {name} sampler binding must retain trained-prefix evidence."
                )
        if self.source_artifact_hash != openpi_rtc_artifact_hash(
            self.source_artifact
        ):
            raise ValueError("OpenPI source artifact hash mismatch.")
        if self.bound_artifact_hash != openpi_rtc_artifact_hash(
            self.bound_artifact
        ):
            raise ValueError("OpenPI bound artifact hash mismatch.")
        if self.source_continuation_contract_hash != source.contract_hash:
            raise ValueError("OpenPI source continuation contract hash mismatch.")
        if self.bound_continuation_contract_hash != bound.contract_hash:
            raise ValueError("OpenPI bound continuation contract hash mismatch.")
        if (
            self.source_implementation_parameters_hash
            != source.runtime.continuation_implementation_parameters_hash
        ):
            raise ValueError("OpenPI source implementation parameters hash mismatch.")
        if (
            self.bound_implementation_parameters_hash
            != bound.runtime.continuation_implementation_parameters_hash
        ):
            raise ValueError("OpenPI bound implementation parameters hash mismatch.")
        if self.trained_num_steps != OPENPI_RTC_TRAINED_NUM_STEPS_V1:
            raise ValueError("OpenPI trained sampler baseline must remain 10 steps.")
        if self.deployment_num_steps not in OPENPI_RTC_DEPLOYMENT_NUM_STEPS_V1:
            raise ValueError("Unsupported OpenPI RTC deployment step count.")
        parameters = dict(self.implementation_parameters)
        if parameters.get("effective_num_steps") != self.deployment_num_steps:
            raise ValueError("OpenPI deployment parameters step count mismatch.")
        if _sha256_json(parameters) != self.bound_implementation_parameters_hash:
            raise ValueError("OpenPI deployment parameters hash mismatch.")
        baseline_parameters = dict(parameters)
        baseline_parameters["effective_num_steps"] = self.trained_num_steps
        if _sha256_json(baseline_parameters) != self.source_implementation_parameters_hash:
            raise ValueError(
                "OpenPI deployment parameters differ from training beyond step count."
            )
        expected_bound = replace(
            self.source_artifact,
            model_continuation=replace(
                source,
                runtime=replace(
                    source.runtime,
                    continuation_implementation_parameters_hash=(
                        self.bound_implementation_parameters_hash
                    ),
                ),
            ),
        )
        if expected_bound.to_dict() != self.bound_artifact.to_dict():
            raise ValueError(
                "OpenPI bound artifact differs outside the RTC sampler hash."
            )
        object.__setattr__(self, "implementation_parameters", parameters)
        object.__setattr__(
            self,
            "provider_capability_hash",
            _required(
                self.provider_capability_hash,
                name="provider_capability_hash",
            ),
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "schema": self.schema,
                "source_artifact_hash": self.source_artifact_hash,
                "bound_artifact_hash": self.bound_artifact_hash,
                "provider_capability_hash": self.provider_capability_hash,
                "source_continuation_contract_hash": (
                    self.source_continuation_contract_hash
                ),
                "bound_continuation_contract_hash": (
                    self.bound_continuation_contract_hash
                ),
                "source_implementation_parameters_hash": (
                    self.source_implementation_parameters_hash
                ),
                "bound_implementation_parameters_hash": (
                    self.bound_implementation_parameters_hash
                ),
                "trained_num_steps": self.trained_num_steps,
                "deployment_num_steps": self.deployment_num_steps,
                "implementation_parameters": dict(
                    self.implementation_parameters
                ),
            },
            domain=OPENPI_TRAINED_RTC_SAMPLER_BINDING_HASH_DOMAIN_V1,
        )

    def receipt_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "model_id": self.bound_artifact.model_id,
            "checkpoint_step": self.bound_artifact.checkpoint_step,
            "source_artifact_hash": self.source_artifact_hash,
            "bound_artifact_hash": self.bound_artifact_hash,
            "provider_capability_hash": self.provider_capability_hash,
            "source_continuation_contract_hash": (
                self.source_continuation_contract_hash
            ),
            "bound_continuation_contract_hash": (
                self.bound_continuation_contract_hash
            ),
            "source_implementation_parameters_hash": (
                self.source_implementation_parameters_hash
            ),
            "bound_implementation_parameters_hash": (
                self.bound_implementation_parameters_hash
            ),
            "trained_num_steps": self.trained_num_steps,
            "deployment_num_steps": self.deployment_num_steps,
            "implementation_parameters": dict(self.implementation_parameters),
            "allowed_delta": "effective_num_steps_only_v1",
            "training_contract_unchanged": True,
            "semantic_content_hash": self.semantic_content_hash,
        }


def bind_openpi_inference_only_rtc_artifact(
    *,
    base_config: OpenPIFinalChunkRtcConfig,
    provider_capability: Mapping[str, Any],
    spec: OpenPIInferenceOnlyRtcBindingSpec,
) -> OpenPIInferenceOnlyRtcArtifactBinding:
    """Bind one immutable artifact to loaded provider facts.

    The caller must persist ``bound_artifact.to_dict()`` and ``receipt_dict()``
    together before using the artifact in a model-RTC shadow.  Constructing
    this value inside a request loop is not a valid persisted binding.
    """

    if not isinstance(base_config, OpenPIFinalChunkRtcConfig):
        raise TypeError("base_config must be an OpenPIFinalChunkRtcConfig.")
    if not isinstance(provider_capability, Mapping):
        raise TypeError("provider_capability must be a mapping.")
    if not isinstance(spec, OpenPIInferenceOnlyRtcBindingSpec):
        raise TypeError("spec must be an OpenPIInferenceOnlyRtcBindingSpec.")
    artifact = base_config.artifact
    if artifact.model_continuation is not None:
        raise ValueError(
            "Refusing to replace an existing artifact continuation binding."
        )
    if (
        base_config.result_action_representation
        != OPENPI_RTC_RESULT_TYPED_CANONICAL_V1
    ):
        raise ValueError(
            "OpenPI model-RTC binding requires typed canonical results."
        )
    if base_config.action_pose_mode != "absolute":
        raise ValueError(
            "OpenPI model-RTC C1 binding currently supports absolute actions."
        )
    layout = artifact.policy_io_layout
    if layout.action_discrete_dim != 0:
        raise ValueError(
            "OpenPI model-RTC binding does not support discrete action tokens."
        )

    expected_capability_hash = openpi_conditioned_provider_capability_hash(
        provider_capability
    )
    advertised_capability_hash = _required(
        provider_capability.get("provider_capability_hash"),
        name="provider_capability_hash",
    )
    if advertised_capability_hash != expected_capability_hash:
        raise ValueError("OpenPI provider capability hash mismatch.")
    if (
        provider_capability.get("provider_capability_schema")
        != OPENPI_CONDITIONED_PROVIDER_CAPABILITY_SCHEMA_V1
        or provider_capability.get("wire_protocol")
        != OPENPI_CONDITIONED_RTC_WIRE_V1
    ):
        raise ValueError("OpenPI provider capability protocol mismatch.")

    external_dim = _nonnegative_int(
        provider_capability.get("external_action_dim"),
        name="external_action_dim",
    )
    if external_dim != layout.action_dim:
        raise ValueError(
            "OpenPI provider external action dimension mismatches artifact."
        )
    horizon = _nonnegative_int(
        provider_capability.get("prediction_horizon_steps"),
        name="prediction_horizon_steps",
    )
    if horizon <= 1:
        raise ValueError("OpenPI prediction horizon must exceed one step.")
    supports_bootstrap = provider_capability.get(
        "supports_unconditioned_bootstrap"
    )
    if not isinstance(supports_bootstrap, bool):
        raise ValueError(
            "OpenPI provider bootstrap capability must be boolean."
        )
    binding_requests_bootstrap = spec.minimum_prefix_steps == 0
    if binding_requests_bootstrap and not supports_bootstrap:
        raise ValueError(
            "Binding requests bootstrap unsupported by the OpenPI provider."
        )

    runtime = ModelContinuationRuntimeContractV2(
        continuation_mode=MODEL_CONTINUATION_FLOW_INPAINT_V1,
        continuation_implementation_id=_required(
            provider_capability.get("continuation_implementation_id"),
            name="continuation_implementation_id",
        ),
        continuation_implementation_parameters_hash=_required(
            provider_capability.get(
                "continuation_implementation_parameters_hash"
            ),
            name="continuation_implementation_parameters_hash",
        ),
        prediction_horizon_steps=horizon,
        minimum_forecast_delay_steps=(
            spec.minimum_forecast_delay_steps
        ),
        maximum_forecast_delay_steps=(
            spec.maximum_forecast_delay_steps
        ),
        minimum_prefix_steps=spec.minimum_prefix_steps,
        maximum_prefix_steps=spec.maximum_prefix_steps,
        model_action_dim=layout.action_dim,
        model_action_discrete_dim=layout.action_discrete_dim,
        component_scope=base_config.component_scope,
        supported_action_pose_modes=(base_config.action_pose_mode,),
        sampler_constraints=(
            _required(
                provider_capability.get("sampler_constraint"),
                name="sampler_constraint",
            ),
        ),
        runtime_prefix_source_roles=(
            RUNTIME_PREFIX_PRODUCER_LOCAL_OVERLAP_V1,
        ),
        action_codec_hash=base_config.action_codec_hash,
        layout_hash=artifact.layout_hash,
        frame_contract_hash=base_config.frame_contract_hash,
        normalization_hash=base_config.normalization_hash,
        supports_unconditioned_bootstrap=binding_requests_bootstrap,
    )
    continuation = ModelContinuationDataContractV2(
        conditioning_origin=MODEL_CONTINUATION_ORIGIN_INFERENCE_ONLY_V1,
        runtime=runtime,
        training=None,
    )
    bound_artifact = replace(
        artifact,
        model_continuation=continuation,
    )
    return OpenPIInferenceOnlyRtcArtifactBinding(
        source_artifact=artifact,
        bound_artifact=bound_artifact,
        provider_capability_hash=expected_capability_hash,
        continuation_contract_hash=continuation.contract_hash,
        source_artifact_hash=openpi_rtc_artifact_hash(artifact),
        bound_artifact_hash=openpi_rtc_artifact_hash(bound_artifact),
    )


def bind_openpi_trained_rtc_sampler(
    *,
    source_artifact: PolicyArtifactContract,
    provider_capability: Mapping[str, Any],
    deployment_num_steps: int,
) -> OpenPITrainedRtcSamplerBinding:
    """Derive a trained artifact for one explicitly selected sampler budget.

    The provider must publish its canonical implementation parameters.  By
    replacing only ``effective_num_steps`` with the fixed trained value (10)
    and reproducing the source contract hash, this function proves that the
    live implementation differs from the checkpoint contract only in the
    versioned deployment step count.  It never mutates the checkpoint artifact.
    """

    if not isinstance(source_artifact, PolicyArtifactContract):
        raise TypeError("source_artifact must be a PolicyArtifactContract.")
    if not isinstance(provider_capability, Mapping):
        raise TypeError("provider_capability must be a mapping.")
    requested_steps = _positive_int(
        deployment_num_steps,
        name="deployment_num_steps",
    )
    if requested_steps not in OPENPI_RTC_DEPLOYMENT_NUM_STEPS_V1:
        raise ValueError(
            "deployment_num_steps must be one of the versioned values "
            f"{OPENPI_RTC_DEPLOYMENT_NUM_STEPS_V1}."
        )
    continuation = source_artifact.model_continuation
    if not isinstance(continuation, ModelContinuationDataContractV2):
        raise ValueError("OpenPI trained sampler binding requires a v2 contract.")
    if (
        continuation.conditioning_origin
        != MODEL_CONTINUATION_ORIGIN_TRAINED_PREFIX_V1
        or continuation.training is None
    ):
        raise ValueError(
            "OpenPI sampler rebind is permitted only for a trained-prefix artifact."
        )
    runtime = continuation.runtime

    expected_capability_hash = openpi_conditioned_provider_capability_hash(
        provider_capability
    )
    if _required(
        provider_capability.get("provider_capability_hash"),
        name="provider_capability_hash",
    ) != expected_capability_hash:
        raise ValueError("OpenPI provider capability hash mismatch.")
    if (
        provider_capability.get("provider_capability_schema")
        != OPENPI_CONDITIONED_PROVIDER_CAPABILITY_SCHEMA_V1
        or provider_capability.get("wire_protocol")
        != OPENPI_CONDITIONED_RTC_WIRE_V1
        or provider_capability.get("enabled") is not True
        or provider_capability.get("conditioning_transform_owner")
        != "provider_v1"
    ):
        raise ValueError("OpenPI provider capability protocol mismatch.")
    if _positive_int(
        provider_capability.get("external_action_dim"),
        name="external_action_dim",
    ) != runtime.model_action_dim:
        raise ValueError("OpenPI provider external action dimension mismatch.")
    if _positive_int(
        provider_capability.get("prediction_horizon_steps"),
        name="prediction_horizon_steps",
    ) != runtime.prediction_horizon_steps:
        raise ValueError("OpenPI provider prediction horizon mismatch.")
    if _required(
        provider_capability.get("continuation_implementation_id"),
        name="continuation_implementation_id",
    ) != runtime.continuation_implementation_id:
        raise ValueError("OpenPI continuation implementation identity mismatch.")
    sampler_constraint = _required(
        provider_capability.get("sampler_constraint"),
        name="sampler_constraint",
    )
    if sampler_constraint not in runtime.sampler_constraints:
        raise ValueError("OpenPI sampler constraint mismatch.")
    supports_bootstrap = provider_capability.get(
        "supports_unconditioned_bootstrap"
    )
    if not isinstance(supports_bootstrap, bool):
        raise ValueError("OpenPI provider bootstrap capability must be boolean.")
    if runtime.supports_unconditioned_bootstrap and not supports_bootstrap:
        raise ValueError("OpenPI provider does not support trained bootstrap.")

    raw_parameters = provider_capability.get(
        "continuation_implementation_parameters"
    )
    if not isinstance(raw_parameters, Mapping):
        raise ValueError(
            "OpenPI provider must publish canonical implementation parameters."
        )
    parameters = dict(raw_parameters)
    effective_steps = _positive_int(
        parameters.get("effective_num_steps"),
        name="effective_num_steps",
    )
    if effective_steps != requested_steps:
        raise ValueError(
            "OpenPI provider effective_num_steps differs from the requested deployment."
        )
    bound_parameters_hash = _sha256_json(parameters)
    if _required(
        provider_capability.get(
            "continuation_implementation_parameters_hash"
        ),
        name="continuation_implementation_parameters_hash",
    ) != bound_parameters_hash:
        raise ValueError("OpenPI provider implementation parameters hash mismatch.")
    baseline_parameters = dict(parameters)
    baseline_parameters["effective_num_steps"] = OPENPI_RTC_TRAINED_NUM_STEPS_V1
    source_parameters_hash = (
        runtime.continuation_implementation_parameters_hash
    )
    if _sha256_json(baseline_parameters) != source_parameters_hash:
        raise ValueError(
            "OpenPI provider differs from the trained implementation beyond step count."
        )

    bound_continuation = replace(
        continuation,
        runtime=replace(
            runtime,
            continuation_implementation_parameters_hash=(
                bound_parameters_hash
            ),
        ),
    )
    bound_artifact = replace(
        source_artifact,
        model_continuation=bound_continuation,
    )
    return OpenPITrainedRtcSamplerBinding(
        source_artifact=source_artifact,
        bound_artifact=bound_artifact,
        provider_capability_hash=expected_capability_hash,
        source_artifact_hash=openpi_rtc_artifact_hash(source_artifact),
        bound_artifact_hash=openpi_rtc_artifact_hash(bound_artifact),
        source_continuation_contract_hash=continuation.contract_hash,
        bound_continuation_contract_hash=bound_continuation.contract_hash,
        source_implementation_parameters_hash=source_parameters_hash,
        bound_implementation_parameters_hash=bound_parameters_hash,
        trained_num_steps=OPENPI_RTC_TRAINED_NUM_STEPS_V1,
        deployment_num_steps=requested_steps,
        implementation_parameters=parameters,
    )


__all__ = [
    "OPENPI_INFERENCE_ONLY_RTC_BINDING_HASH_DOMAIN_V1",
    "OPENPI_INFERENCE_ONLY_RTC_BINDING_SCHEMA_V1",
    "OPENPI_RTC_DEPLOYMENT_NUM_STEPS_V1",
    "OPENPI_RTC_TRAINED_NUM_STEPS_V1",
    "OPENPI_TRAINED_RTC_SAMPLER_BINDING_HASH_DOMAIN_V1",
    "OPENPI_TRAINED_RTC_SAMPLER_BINDING_SCHEMA_V1",
    "OpenPIInferenceOnlyRtcArtifactBinding",
    "OpenPIInferenceOnlyRtcBindingSpec",
    "OpenPITrainedRtcSamplerBinding",
    "bind_openpi_inference_only_rtc_artifact",
    "bind_openpi_trained_rtc_sampler",
]
