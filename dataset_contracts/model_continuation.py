"""Pure artifact/training contract for model-conditioned action continuation.

This module defines shared data semantics only.  It owns no dataset reader,
runtime request, plan, model invocation, sampler implementation, or control
authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping, Sequence

from .canonical_encoding import canonical_content_hash
from .pose_chunk import POSE_ACTION_MODES
from .v22 import DatasetLayoutBindingError


MODEL_CONTINUATION_DATA_SCHEMA_V1 = "dexteleop_model_continuation_data_v1"
MODEL_CONTINUATION_DATA_METADATA_KEY = "model_continuation_data_contract"
MODEL_CONTINUATION_DATA_HASH_KEY = "model_continuation_data_contract_hash"
MODEL_CONTINUATION_DATA_HASH_DOMAIN_V1 = (
    "dexteleop.model_continuation_data_contract.v1"
)
MODEL_CONTINUATION_DATA_SCHEMA_V2 = "dexteleop_model_continuation_data_v2"
MODEL_CONTINUATION_DATA_HASH_DOMAIN_V2 = (
    "dexteleop.model_continuation_data_contract.v2"
)
MODEL_CONTINUATION_RUNTIME_SCHEMA_V2 = (
    "dexteleop_model_continuation_runtime_v2"
)
MODEL_CONTINUATION_TRAINING_SCHEMA_V2 = (
    "dexteleop_model_continuation_training_v2"
)

MODEL_CONTINUATION_ORIGIN_INFERENCE_ONLY_V1 = (
    "inference_only_projection_v1"
)
MODEL_CONTINUATION_ORIGIN_TRAINED_PREFIX_V1 = (
    "trained_prefix_conditioning_v1"
)
MODEL_CONTINUATION_ORIGIN_LEGACY_V1_COMPAT = "legacy_v1_compat"
MODEL_CONTINUATION_PERSISTED_ORIGINS_V2 = (
    MODEL_CONTINUATION_ORIGIN_INFERENCE_ONLY_V1,
    MODEL_CONTINUATION_ORIGIN_TRAINED_PREFIX_V1,
)

MODEL_CONTINUATION_FLOW_INPAINT_V1 = "flow_inpaint"
MODEL_CONTINUATION_TRAINED_PREFIX_V1 = "trained_prefix_conditioning"
MODEL_CONTINUATION_NATIVE_V1 = "native_continuation"
MODEL_CONTINUATION_MODES_V1 = (
    MODEL_CONTINUATION_FLOW_INPAINT_V1,
    MODEL_CONTINUATION_TRAINED_PREFIX_V1,
    MODEL_CONTINUATION_NATIVE_V1,
)

MODEL_CONTINUATION_REPRESENTATION_CONTINUOUS_V1 = (
    "continuous_model_action_prefix_v1"
)

TRAINING_PREFIX_TRAJECTORY_OVERLAP_TEACHER_V1 = (
    "trajectory_overlap_teacher_v1"
)
TRAINING_PREFIX_RECORDED_POLICY_OUTPUT_V1 = "recorded_policy_output_v1"
TRAINING_PREFIX_SYNTHETIC_CORRUPTION_V1 = "synthetic_corruption_v1"
TRAINING_PREFIX_SOURCE_KINDS_V1 = (
    TRAINING_PREFIX_TRAJECTORY_OVERLAP_TEACHER_V1,
    TRAINING_PREFIX_RECORDED_POLICY_OUTPUT_V1,
    TRAINING_PREFIX_SYNTHETIC_CORRUPTION_V1,
)

RUNTIME_PREFIX_PRODUCER_LOCAL_OVERLAP_V1 = "producer_local_overlap_v1"
RUNTIME_PREFIX_SOURCE_ROLES_V1 = (
    RUNTIME_PREFIX_PRODUCER_LOCAL_OVERLAP_V1,
)

MODEL_CONTINUATION_SAMPLER_CLAMPED_V1 = "clamped_v1"
MODEL_CONTINUATION_SAMPLER_WEIGHTED_V1 = "weighted_v1"
MODEL_CONTINUATION_SAMPLER_CONSTRAINTS_V1 = (
    MODEL_CONTINUATION_SAMPLER_CLAMPED_V1,
    MODEL_CONTINUATION_SAMPLER_WEIGHTED_V1,
)

MODEL_CONTINUATION_PREFIX_MASK_SEMANTICS_V1 = (
    "float_weight_per_generation_step_v1"
)
MODEL_CONTINUATION_TARGET_PADDING_MASK_SEMANTICS_V1 = (
    "true_is_padding_per_generation_step_v1"
)
MODEL_CONTINUATION_LOSS_UNCONDITIONED_VALID_V1 = (
    "unconditioned_valid_steps_v1"
)
MODEL_CONTINUATION_LOSS_ALL_VALID_V1 = "all_valid_steps_v1"
MODEL_CONTINUATION_LOSS_MASK_SEMANTICS_V1 = (
    MODEL_CONTINUATION_LOSS_UNCONDITIONED_VALID_V1,
    MODEL_CONTINUATION_LOSS_ALL_VALID_V1,
)


def _required(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise DatasetLayoutBindingError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise DatasetLayoutBindingError(f"{name} must be non-empty.")
    return normalized


def _exact_int(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise DatasetLayoutBindingError(f"{name} must be an integer.")
    normalized = int(value)
    if normalized < minimum:
        raise DatasetLayoutBindingError(
            f"{name} must be an integer greater than or equal to {minimum}."
        )
    return normalized


def _string_sequence(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise DatasetLayoutBindingError(f"{name} must be a sequence.")
    return tuple(_required(item, name=f"{name} item") for item in value)


def _normalized_set(
    values: tuple[str, ...],
    *,
    name: str,
    allowed: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    normalized = tuple(sorted(set(values)))
    if not normalized:
        raise DatasetLayoutBindingError(f"{name} must be non-empty.")
    if allowed is not None:
        unsupported = sorted(set(normalized).difference(allowed))
        if unsupported:
            raise DatasetLayoutBindingError(
                f"{name} contains unsupported values: {unsupported}."
            )
    return normalized


@dataclass(frozen=True)
class ModelContinuationDataContract:
    """Artifact-bound semantics shared by training and model-RTC deployment."""

    continuation_mode: str
    continuation_implementation_id: str
    continuation_implementation_parameters_hash: str
    prediction_horizon_steps: int
    minimum_forecast_delay_steps: int
    maximum_forecast_delay_steps: int
    minimum_prefix_steps: int
    maximum_prefix_steps: int
    model_action_dim: int
    model_action_discrete_dim: int
    component_scope: tuple[str, ...]
    supported_action_pose_modes: tuple[str, ...]
    sampler_constraints: tuple[str, ...]
    training_prefix_source_kinds: tuple[str, ...]
    runtime_prefix_source_roles: tuple[str, ...]
    action_codec_hash: str
    layout_hash: str
    frame_contract_hash: str
    normalization_hash: str
    training_projection_implementation_id: str
    training_projection_parameters_hash: str
    supports_unconditioned_bootstrap: bool = False
    loss_mask_semantics: str = (
        MODEL_CONTINUATION_LOSS_UNCONDITIONED_VALID_V1
    )
    conditioning_representation: str = (
        MODEL_CONTINUATION_REPRESENTATION_CONTINUOUS_V1
    )
    prefix_mask_semantics: str = (
        MODEL_CONTINUATION_PREFIX_MASK_SEMANTICS_V1
    )
    target_padding_mask_semantics: str = (
        MODEL_CONTINUATION_TARGET_PADDING_MASK_SEMANTICS_V1
    )
    schema: str = MODEL_CONTINUATION_DATA_SCHEMA_V1

    def __post_init__(self) -> None:
        if str(self.schema) != MODEL_CONTINUATION_DATA_SCHEMA_V1:
            raise DatasetLayoutBindingError(
                f"Unsupported model continuation schema {self.schema!r}."
            )
        mode = _required(self.continuation_mode, name="continuation_mode")
        if mode not in MODEL_CONTINUATION_MODES_V1:
            raise DatasetLayoutBindingError(
                f"Unsupported continuation_mode {mode!r}."
            )
        for name in (
            "continuation_implementation_id",
            "continuation_implementation_parameters_hash",
            "action_codec_hash",
            "layout_hash",
            "frame_contract_hash",
            "normalization_hash",
            "training_projection_implementation_id",
            "training_projection_parameters_hash",
        ):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), name=name),
            )

        horizon = _exact_int(
            self.prediction_horizon_steps,
            name="prediction_horizon_steps",
            minimum=2,
        )
        minimum_delay = _exact_int(
            self.minimum_forecast_delay_steps,
            name="minimum_forecast_delay_steps",
        )
        maximum_delay = _exact_int(
            self.maximum_forecast_delay_steps,
            name="maximum_forecast_delay_steps",
        )
        minimum_prefix = _exact_int(
            self.minimum_prefix_steps,
            name="minimum_prefix_steps",
        )
        maximum_prefix = _exact_int(
            self.maximum_prefix_steps,
            name="maximum_prefix_steps",
        )
        if minimum_delay > maximum_delay or maximum_delay >= horizon:
            raise DatasetLayoutBindingError(
                "Forecast-delay range must be ordered and leave a non-empty "
                "executable suffix."
            )
        if maximum_delay <= 0:
            raise DatasetLayoutBindingError(
                "A continuation contract requires a positive supported delay."
            )
        if minimum_prefix > maximum_prefix or maximum_prefix <= 0:
            raise DatasetLayoutBindingError(
                "Prefix range must be ordered and include a positive prefix."
            )
        if minimum_prefix > minimum_delay or maximum_prefix > maximum_delay:
            raise DatasetLayoutBindingError(
                "Supported prefix steps cannot exceed their forecast-delay range."
            )

        if not isinstance(self.supports_unconditioned_bootstrap, bool):
            raise DatasetLayoutBindingError(
                "supports_unconditioned_bootstrap must be a boolean."
            )
        if self.supports_unconditioned_bootstrap != (minimum_prefix == 0):
            raise DatasetLayoutBindingError(
                "minimum_prefix_steps must be zero exactly when unconditioned "
                "bootstrap is supported."
            )

        model_action_dim = _exact_int(
            self.model_action_dim,
            name="model_action_dim",
            minimum=1,
        )
        discrete_dim = _exact_int(
            self.model_action_discrete_dim,
            name="model_action_discrete_dim",
        )
        if discrete_dim != 0:
            raise DatasetLayoutBindingError(
                "Model continuation schema v1 supports only continuous prefixes."
            )

        component_scope = _normalized_set(
            _string_sequence(self.component_scope, name="component_scope"),
            name="component_scope",
        )
        pose_modes = _normalized_set(
            _string_sequence(
                self.supported_action_pose_modes,
                name="supported_action_pose_modes",
            ),
            name="supported_action_pose_modes",
            allowed=POSE_ACTION_MODES,
        )
        sampler_constraints = _normalized_set(
            _string_sequence(
                self.sampler_constraints,
                name="sampler_constraints",
            ),
            name="sampler_constraints",
            allowed=MODEL_CONTINUATION_SAMPLER_CONSTRAINTS_V1,
        )
        training_sources = _normalized_set(
            _string_sequence(
                self.training_prefix_source_kinds,
                name="training_prefix_source_kinds",
            ),
            name="training_prefix_source_kinds",
            allowed=TRAINING_PREFIX_SOURCE_KINDS_V1,
        )
        runtime_roles = _normalized_set(
            _string_sequence(
                self.runtime_prefix_source_roles,
                name="runtime_prefix_source_roles",
            ),
            name="runtime_prefix_source_roles",
            allowed=RUNTIME_PREFIX_SOURCE_ROLES_V1,
        )
        if runtime_roles != RUNTIME_PREFIX_SOURCE_ROLES_V1:
            raise DatasetLayoutBindingError(
                "Schema v1 requires producer_local_overlap_v1 as its only "
                "runtime prefix source role."
            )
        if (
            self.conditioning_representation
            != MODEL_CONTINUATION_REPRESENTATION_CONTINUOUS_V1
        ):
            raise DatasetLayoutBindingError(
                "Schema v1 requires continuous_model_action_prefix_v1."
            )
        if (
            self.prefix_mask_semantics
            != MODEL_CONTINUATION_PREFIX_MASK_SEMANTICS_V1
        ):
            raise DatasetLayoutBindingError(
                "Unsupported prefix_mask_semantics."
            )
        if (
            self.target_padding_mask_semantics
            != MODEL_CONTINUATION_TARGET_PADDING_MASK_SEMANTICS_V1
        ):
            raise DatasetLayoutBindingError(
                "Unsupported target_padding_mask_semantics."
            )
        if self.loss_mask_semantics not in (
            MODEL_CONTINUATION_LOSS_MASK_SEMANTICS_V1
        ):
            raise DatasetLayoutBindingError("Unsupported loss_mask_semantics.")

        object.__setattr__(self, "schema", MODEL_CONTINUATION_DATA_SCHEMA_V1)
        object.__setattr__(self, "continuation_mode", mode)
        object.__setattr__(self, "prediction_horizon_steps", horizon)
        object.__setattr__(self, "minimum_forecast_delay_steps", minimum_delay)
        object.__setattr__(self, "maximum_forecast_delay_steps", maximum_delay)
        object.__setattr__(self, "minimum_prefix_steps", minimum_prefix)
        object.__setattr__(self, "maximum_prefix_steps", maximum_prefix)
        object.__setattr__(self, "model_action_dim", model_action_dim)
        object.__setattr__(self, "model_action_discrete_dim", discrete_dim)
        object.__setattr__(self, "component_scope", component_scope)
        object.__setattr__(self, "supported_action_pose_modes", pose_modes)
        object.__setattr__(self, "sampler_constraints", sampler_constraints)
        object.__setattr__(
            self,
            "training_prefix_source_kinds",
            training_sources,
        )
        object.__setattr__(self, "runtime_prefix_source_roles", runtime_roles)
        object.__setattr__(
            self,
            "conditioning_representation",
            MODEL_CONTINUATION_REPRESENTATION_CONTINUOUS_V1,
        )
        object.__setattr__(
            self,
            "prefix_mask_semantics",
            MODEL_CONTINUATION_PREFIX_MASK_SEMANTICS_V1,
        )
        object.__setattr__(
            self,
            "target_padding_mask_semantics",
            MODEL_CONTINUATION_TARGET_PADDING_MASK_SEMANTICS_V1,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "continuation_mode": self.continuation_mode,
            "continuation_implementation_id": (
                self.continuation_implementation_id
            ),
            "continuation_implementation_parameters_hash": (
                self.continuation_implementation_parameters_hash
            ),
            "prediction_horizon_steps": self.prediction_horizon_steps,
            "minimum_forecast_delay_steps": (
                self.minimum_forecast_delay_steps
            ),
            "maximum_forecast_delay_steps": (
                self.maximum_forecast_delay_steps
            ),
            "minimum_prefix_steps": self.minimum_prefix_steps,
            "maximum_prefix_steps": self.maximum_prefix_steps,
            "model_action_dim": self.model_action_dim,
            "model_action_discrete_dim": self.model_action_discrete_dim,
            "component_scope": list(self.component_scope),
            "supported_action_pose_modes": list(
                self.supported_action_pose_modes
            ),
            "sampler_constraints": list(self.sampler_constraints),
            "training_prefix_source_kinds": list(
                self.training_prefix_source_kinds
            ),
            "runtime_prefix_source_roles": list(
                self.runtime_prefix_source_roles
            ),
            "action_codec_hash": self.action_codec_hash,
            "layout_hash": self.layout_hash,
            "frame_contract_hash": self.frame_contract_hash,
            "normalization_hash": self.normalization_hash,
            "training_projection_implementation_id": (
                self.training_projection_implementation_id
            ),
            "training_projection_parameters_hash": (
                self.training_projection_parameters_hash
            ),
            "supports_unconditioned_bootstrap": (
                self.supports_unconditioned_bootstrap
            ),
            "conditioning_representation": self.conditioning_representation,
            "prefix_mask_semantics": self.prefix_mask_semantics,
            "target_padding_mask_semantics": (
                self.target_padding_mask_semantics
            ),
            "loss_mask_semantics": self.loss_mask_semantics,
        }

    @property
    def contract_hash(self) -> str:
        digest = canonical_content_hash(
            self.to_dict(),
            domain=MODEL_CONTINUATION_DATA_HASH_DOMAIN_V1,
        )
        return f"sha256:{digest}"

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "ModelContinuationDataContract":
        expected_keys = {
            "schema",
            "continuation_mode",
            "continuation_implementation_id",
            "continuation_implementation_parameters_hash",
            "prediction_horizon_steps",
            "minimum_forecast_delay_steps",
            "maximum_forecast_delay_steps",
            "minimum_prefix_steps",
            "maximum_prefix_steps",
            "model_action_dim",
            "model_action_discrete_dim",
            "component_scope",
            "supported_action_pose_modes",
            "sampler_constraints",
            "training_prefix_source_kinds",
            "runtime_prefix_source_roles",
            "action_codec_hash",
            "layout_hash",
            "frame_contract_hash",
            "normalization_hash",
            "training_projection_implementation_id",
            "training_projection_parameters_hash",
            "supports_unconditioned_bootstrap",
            "conditioning_representation",
            "prefix_mask_semantics",
            "target_padding_mask_semantics",
            "loss_mask_semantics",
        }
        unknown = sorted(set(payload).difference(expected_keys))
        if unknown:
            raise DatasetLayoutBindingError(
                f"Model continuation contract contains unknown entries: {unknown}."
            )
        return cls(
            continuation_mode=str(payload.get("continuation_mode", "")),
            continuation_implementation_id=payload.get(
                "continuation_implementation_id",
                "",
            ),
            continuation_implementation_parameters_hash=payload.get(
                "continuation_implementation_parameters_hash",
                "",
            ),
            prediction_horizon_steps=payload.get("prediction_horizon_steps", -1),
            minimum_forecast_delay_steps=payload.get(
                "minimum_forecast_delay_steps",
                -1,
            ),
            maximum_forecast_delay_steps=payload.get(
                "maximum_forecast_delay_steps",
                -1,
            ),
            minimum_prefix_steps=payload.get("minimum_prefix_steps", -1),
            maximum_prefix_steps=payload.get("maximum_prefix_steps", -1),
            model_action_dim=payload.get("model_action_dim", -1),
            model_action_discrete_dim=payload.get(
                "model_action_discrete_dim",
                -1,
            ),
            component_scope=_string_sequence(
                payload.get("component_scope"),
                name="component_scope",
            ),
            supported_action_pose_modes=_string_sequence(
                payload.get("supported_action_pose_modes"),
                name="supported_action_pose_modes",
            ),
            sampler_constraints=_string_sequence(
                payload.get("sampler_constraints"),
                name="sampler_constraints",
            ),
            training_prefix_source_kinds=_string_sequence(
                payload.get("training_prefix_source_kinds"),
                name="training_prefix_source_kinds",
            ),
            runtime_prefix_source_roles=_string_sequence(
                payload.get("runtime_prefix_source_roles"),
                name="runtime_prefix_source_roles",
            ),
            action_codec_hash=payload.get("action_codec_hash", ""),
            layout_hash=payload.get("layout_hash", ""),
            frame_contract_hash=payload.get("frame_contract_hash", ""),
            normalization_hash=payload.get("normalization_hash", ""),
            training_projection_implementation_id=payload.get(
                "training_projection_implementation_id",
                "",
            ),
            training_projection_parameters_hash=payload.get(
                "training_projection_parameters_hash",
                "",
            ),
            supports_unconditioned_bootstrap=payload.get(
                "supports_unconditioned_bootstrap",
                False,
            ),
            loss_mask_semantics=str(
                payload.get("loss_mask_semantics", "")
            ),
            conditioning_representation=str(
                payload.get("conditioning_representation", "")
            ),
            prefix_mask_semantics=str(
                payload.get("prefix_mask_semantics", "")
            ),
            target_padding_mask_semantics=str(
                payload.get("target_padding_mask_semantics", "")
            ),
            schema=str(payload.get("schema", "")),
        )


@dataclass(frozen=True)
class ModelContinuationRuntimeContractV2:
    """Runtime-only continuation capability, independent of training claims."""

    continuation_mode: str
    continuation_implementation_id: str
    continuation_implementation_parameters_hash: str
    prediction_horizon_steps: int
    minimum_forecast_delay_steps: int
    maximum_forecast_delay_steps: int
    minimum_prefix_steps: int
    maximum_prefix_steps: int
    model_action_dim: int
    model_action_discrete_dim: int
    component_scope: tuple[str, ...]
    supported_action_pose_modes: tuple[str, ...]
    sampler_constraints: tuple[str, ...]
    runtime_prefix_source_roles: tuple[str, ...]
    action_codec_hash: str
    layout_hash: str
    frame_contract_hash: str
    normalization_hash: str
    supports_unconditioned_bootstrap: bool = False
    conditioning_representation: str = (
        MODEL_CONTINUATION_REPRESENTATION_CONTINUOUS_V1
    )
    prefix_mask_semantics: str = (
        MODEL_CONTINUATION_PREFIX_MASK_SEMANTICS_V1
    )
    target_padding_mask_semantics: str = (
        MODEL_CONTINUATION_TARGET_PADDING_MASK_SEMANTICS_V1
    )
    schema: str = MODEL_CONTINUATION_RUNTIME_SCHEMA_V2

    def __post_init__(self) -> None:
        if str(self.schema) != MODEL_CONTINUATION_RUNTIME_SCHEMA_V2:
            raise DatasetLayoutBindingError(
                f"Unsupported model continuation runtime schema {self.schema!r}."
            )
        mode = _required(self.continuation_mode, name="continuation_mode")
        if mode not in MODEL_CONTINUATION_MODES_V1:
            raise DatasetLayoutBindingError(
                f"Unsupported continuation_mode {mode!r}."
            )
        for name in (
            "continuation_implementation_id",
            "continuation_implementation_parameters_hash",
            "action_codec_hash",
            "layout_hash",
            "frame_contract_hash",
            "normalization_hash",
        ):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), name=name),
            )
        horizon = _exact_int(
            self.prediction_horizon_steps,
            name="prediction_horizon_steps",
            minimum=2,
        )
        minimum_delay = _exact_int(
            self.minimum_forecast_delay_steps,
            name="minimum_forecast_delay_steps",
        )
        maximum_delay = _exact_int(
            self.maximum_forecast_delay_steps,
            name="maximum_forecast_delay_steps",
        )
        minimum_prefix = _exact_int(
            self.minimum_prefix_steps,
            name="minimum_prefix_steps",
        )
        maximum_prefix = _exact_int(
            self.maximum_prefix_steps,
            name="maximum_prefix_steps",
        )
        if minimum_delay > maximum_delay or maximum_delay >= horizon:
            raise DatasetLayoutBindingError(
                "Forecast-delay range must be ordered and leave a non-empty "
                "executable suffix."
            )
        if maximum_delay <= 0:
            raise DatasetLayoutBindingError(
                "A continuation contract requires a positive supported delay."
            )
        if minimum_prefix > maximum_prefix or maximum_prefix <= 0:
            raise DatasetLayoutBindingError(
                "Prefix range must be ordered and include a positive prefix."
            )
        if minimum_prefix > minimum_delay or maximum_prefix > maximum_delay:
            raise DatasetLayoutBindingError(
                "Supported prefix steps cannot exceed their forecast-delay range."
            )
        if not isinstance(self.supports_unconditioned_bootstrap, bool):
            raise DatasetLayoutBindingError(
                "supports_unconditioned_bootstrap must be a boolean."
            )
        if self.supports_unconditioned_bootstrap != (minimum_prefix == 0):
            raise DatasetLayoutBindingError(
                "minimum_prefix_steps must be zero exactly when unconditioned "
                "bootstrap is supported."
            )
        model_action_dim = _exact_int(
            self.model_action_dim,
            name="model_action_dim",
            minimum=1,
        )
        discrete_dim = _exact_int(
            self.model_action_discrete_dim,
            name="model_action_discrete_dim",
        )
        if discrete_dim != 0:
            raise DatasetLayoutBindingError(
                "Model continuation runtime v2 supports only continuous prefixes."
            )
        component_scope = _normalized_set(
            _string_sequence(self.component_scope, name="component_scope"),
            name="component_scope",
        )
        pose_modes = _normalized_set(
            _string_sequence(
                self.supported_action_pose_modes,
                name="supported_action_pose_modes",
            ),
            name="supported_action_pose_modes",
            allowed=POSE_ACTION_MODES,
        )
        sampler_constraints = _normalized_set(
            _string_sequence(
                self.sampler_constraints,
                name="sampler_constraints",
            ),
            name="sampler_constraints",
            allowed=MODEL_CONTINUATION_SAMPLER_CONSTRAINTS_V1,
        )
        runtime_roles = _normalized_set(
            _string_sequence(
                self.runtime_prefix_source_roles,
                name="runtime_prefix_source_roles",
            ),
            name="runtime_prefix_source_roles",
            allowed=RUNTIME_PREFIX_SOURCE_ROLES_V1,
        )
        if runtime_roles != RUNTIME_PREFIX_SOURCE_ROLES_V1:
            raise DatasetLayoutBindingError(
                "Runtime v2 requires producer_local_overlap_v1 as its only "
                "runtime prefix source role."
            )
        if (
            self.conditioning_representation
            != MODEL_CONTINUATION_REPRESENTATION_CONTINUOUS_V1
        ):
            raise DatasetLayoutBindingError(
                "Runtime v2 requires continuous_model_action_prefix_v1."
            )
        if (
            self.prefix_mask_semantics
            != MODEL_CONTINUATION_PREFIX_MASK_SEMANTICS_V1
        ):
            raise DatasetLayoutBindingError(
                "Unsupported prefix_mask_semantics."
            )
        if (
            self.target_padding_mask_semantics
            != MODEL_CONTINUATION_TARGET_PADDING_MASK_SEMANTICS_V1
        ):
            raise DatasetLayoutBindingError(
                "Unsupported target_padding_mask_semantics."
            )
        object.__setattr__(self, "continuation_mode", mode)
        object.__setattr__(self, "prediction_horizon_steps", horizon)
        object.__setattr__(self, "minimum_forecast_delay_steps", minimum_delay)
        object.__setattr__(self, "maximum_forecast_delay_steps", maximum_delay)
        object.__setattr__(self, "minimum_prefix_steps", minimum_prefix)
        object.__setattr__(self, "maximum_prefix_steps", maximum_prefix)
        object.__setattr__(self, "model_action_dim", model_action_dim)
        object.__setattr__(self, "model_action_discrete_dim", discrete_dim)
        object.__setattr__(self, "component_scope", component_scope)
        object.__setattr__(self, "supported_action_pose_modes", pose_modes)
        object.__setattr__(self, "sampler_constraints", sampler_constraints)
        object.__setattr__(self, "runtime_prefix_source_roles", runtime_roles)
        object.__setattr__(
            self,
            "conditioning_representation",
            MODEL_CONTINUATION_REPRESENTATION_CONTINUOUS_V1,
        )
        object.__setattr__(
            self,
            "prefix_mask_semantics",
            MODEL_CONTINUATION_PREFIX_MASK_SEMANTICS_V1,
        )
        object.__setattr__(
            self,
            "target_padding_mask_semantics",
            MODEL_CONTINUATION_TARGET_PADDING_MASK_SEMANTICS_V1,
        )
        object.__setattr__(self, "schema", MODEL_CONTINUATION_RUNTIME_SCHEMA_V2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "continuation_mode": self.continuation_mode,
            "continuation_implementation_id": (
                self.continuation_implementation_id
            ),
            "continuation_implementation_parameters_hash": (
                self.continuation_implementation_parameters_hash
            ),
            "prediction_horizon_steps": self.prediction_horizon_steps,
            "minimum_forecast_delay_steps": (
                self.minimum_forecast_delay_steps
            ),
            "maximum_forecast_delay_steps": (
                self.maximum_forecast_delay_steps
            ),
            "minimum_prefix_steps": self.minimum_prefix_steps,
            "maximum_prefix_steps": self.maximum_prefix_steps,
            "model_action_dim": self.model_action_dim,
            "model_action_discrete_dim": self.model_action_discrete_dim,
            "component_scope": list(self.component_scope),
            "supported_action_pose_modes": list(
                self.supported_action_pose_modes
            ),
            "sampler_constraints": list(self.sampler_constraints),
            "runtime_prefix_source_roles": list(
                self.runtime_prefix_source_roles
            ),
            "action_codec_hash": self.action_codec_hash,
            "layout_hash": self.layout_hash,
            "frame_contract_hash": self.frame_contract_hash,
            "normalization_hash": self.normalization_hash,
            "supports_unconditioned_bootstrap": (
                self.supports_unconditioned_bootstrap
            ),
            "conditioning_representation": self.conditioning_representation,
            "prefix_mask_semantics": self.prefix_mask_semantics,
            "target_padding_mask_semantics": (
                self.target_padding_mask_semantics
            ),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "ModelContinuationRuntimeContractV2":
        expected_keys = {
            "schema",
            "continuation_mode",
            "continuation_implementation_id",
            "continuation_implementation_parameters_hash",
            "prediction_horizon_steps",
            "minimum_forecast_delay_steps",
            "maximum_forecast_delay_steps",
            "minimum_prefix_steps",
            "maximum_prefix_steps",
            "model_action_dim",
            "model_action_discrete_dim",
            "component_scope",
            "supported_action_pose_modes",
            "sampler_constraints",
            "runtime_prefix_source_roles",
            "action_codec_hash",
            "layout_hash",
            "frame_contract_hash",
            "normalization_hash",
            "supports_unconditioned_bootstrap",
            "conditioning_representation",
            "prefix_mask_semantics",
            "target_padding_mask_semantics",
        }
        unknown = sorted(set(payload).difference(expected_keys))
        if unknown:
            raise DatasetLayoutBindingError(
                "Model continuation runtime contains unknown entries: "
                f"{unknown}."
            )
        return cls(
            continuation_mode=str(payload.get("continuation_mode", "")),
            continuation_implementation_id=payload.get(
                "continuation_implementation_id",
                "",
            ),
            continuation_implementation_parameters_hash=payload.get(
                "continuation_implementation_parameters_hash",
                "",
            ),
            prediction_horizon_steps=payload.get(
                "prediction_horizon_steps",
                -1,
            ),
            minimum_forecast_delay_steps=payload.get(
                "minimum_forecast_delay_steps",
                -1,
            ),
            maximum_forecast_delay_steps=payload.get(
                "maximum_forecast_delay_steps",
                -1,
            ),
            minimum_prefix_steps=payload.get("minimum_prefix_steps", -1),
            maximum_prefix_steps=payload.get("maximum_prefix_steps", -1),
            model_action_dim=payload.get("model_action_dim", -1),
            model_action_discrete_dim=payload.get(
                "model_action_discrete_dim",
                -1,
            ),
            component_scope=_string_sequence(
                payload.get("component_scope"),
                name="component_scope",
            ),
            supported_action_pose_modes=_string_sequence(
                payload.get("supported_action_pose_modes"),
                name="supported_action_pose_modes",
            ),
            sampler_constraints=_string_sequence(
                payload.get("sampler_constraints"),
                name="sampler_constraints",
            ),
            runtime_prefix_source_roles=_string_sequence(
                payload.get("runtime_prefix_source_roles"),
                name="runtime_prefix_source_roles",
            ),
            action_codec_hash=payload.get("action_codec_hash", ""),
            layout_hash=payload.get("layout_hash", ""),
            frame_contract_hash=payload.get("frame_contract_hash", ""),
            normalization_hash=payload.get("normalization_hash", ""),
            supports_unconditioned_bootstrap=payload.get(
                "supports_unconditioned_bootstrap",
                False,
            ),
            conditioning_representation=str(
                payload.get("conditioning_representation", "")
            ),
            prefix_mask_semantics=str(
                payload.get("prefix_mask_semantics", "")
            ),
            target_padding_mask_semantics=str(
                payload.get("target_padding_mask_semantics", "")
            ),
            schema=str(payload.get("schema", "")),
        )

    @classmethod
    def from_v1(
        cls,
        contract: ModelContinuationDataContract,
    ) -> "ModelContinuationRuntimeContractV2":
        if not isinstance(contract, ModelContinuationDataContract):
            raise TypeError("contract must be a v1 continuation contract.")
        return cls(
            continuation_mode=contract.continuation_mode,
            continuation_implementation_id=(
                contract.continuation_implementation_id
            ),
            continuation_implementation_parameters_hash=(
                contract.continuation_implementation_parameters_hash
            ),
            prediction_horizon_steps=contract.prediction_horizon_steps,
            minimum_forecast_delay_steps=(
                contract.minimum_forecast_delay_steps
            ),
            maximum_forecast_delay_steps=(
                contract.maximum_forecast_delay_steps
            ),
            minimum_prefix_steps=contract.minimum_prefix_steps,
            maximum_prefix_steps=contract.maximum_prefix_steps,
            model_action_dim=contract.model_action_dim,
            model_action_discrete_dim=contract.model_action_discrete_dim,
            component_scope=contract.component_scope,
            supported_action_pose_modes=contract.supported_action_pose_modes,
            sampler_constraints=contract.sampler_constraints,
            runtime_prefix_source_roles=(
                contract.runtime_prefix_source_roles
            ),
            action_codec_hash=contract.action_codec_hash,
            layout_hash=contract.layout_hash,
            frame_contract_hash=contract.frame_contract_hash,
            normalization_hash=contract.normalization_hash,
            supports_unconditioned_bootstrap=(
                contract.supports_unconditioned_bootstrap
            ),
            conditioning_representation=contract.conditioning_representation,
            prefix_mask_semantics=contract.prefix_mask_semantics,
            target_padding_mask_semantics=(
                contract.target_padding_mask_semantics
            ),
        )


@dataclass(frozen=True)
class ModelContinuationTrainingContractV2:
    """Training provenance required only for trained-prefix artifacts."""

    prefix_source_kinds: tuple[str, ...]
    projection_implementation_id: str
    projection_parameters_hash: str
    loss_mask_semantics: str = (
        MODEL_CONTINUATION_LOSS_UNCONDITIONED_VALID_V1
    )
    schema: str = MODEL_CONTINUATION_TRAINING_SCHEMA_V2

    def __post_init__(self) -> None:
        if str(self.schema) != MODEL_CONTINUATION_TRAINING_SCHEMA_V2:
            raise DatasetLayoutBindingError(
                f"Unsupported model continuation training schema {self.schema!r}."
            )
        sources = _normalized_set(
            _string_sequence(
                self.prefix_source_kinds,
                name="prefix_source_kinds",
            ),
            name="prefix_source_kinds",
            allowed=TRAINING_PREFIX_SOURCE_KINDS_V1,
        )
        projection_id = _required(
            self.projection_implementation_id,
            name="projection_implementation_id",
        )
        projection_hash = _required(
            self.projection_parameters_hash,
            name="projection_parameters_hash",
        )
        loss_semantics = _required(
            self.loss_mask_semantics,
            name="loss_mask_semantics",
        )
        if loss_semantics not in MODEL_CONTINUATION_LOSS_MASK_SEMANTICS_V1:
            raise DatasetLayoutBindingError("Unsupported loss_mask_semantics.")
        object.__setattr__(self, "prefix_source_kinds", sources)
        object.__setattr__(
            self,
            "projection_implementation_id",
            projection_id,
        )
        object.__setattr__(
            self,
            "projection_parameters_hash",
            projection_hash,
        )
        object.__setattr__(self, "loss_mask_semantics", loss_semantics)
        object.__setattr__(
            self,
            "schema",
            MODEL_CONTINUATION_TRAINING_SCHEMA_V2,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "prefix_source_kinds": list(self.prefix_source_kinds),
            "projection_implementation_id": (
                self.projection_implementation_id
            ),
            "projection_parameters_hash": self.projection_parameters_hash,
            "loss_mask_semantics": self.loss_mask_semantics,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "ModelContinuationTrainingContractV2":
        expected_keys = {
            "schema",
            "prefix_source_kinds",
            "projection_implementation_id",
            "projection_parameters_hash",
            "loss_mask_semantics",
        }
        unknown = sorted(set(payload).difference(expected_keys))
        if unknown:
            raise DatasetLayoutBindingError(
                "Model continuation training contains unknown entries: "
                f"{unknown}."
            )
        return cls(
            prefix_source_kinds=_string_sequence(
                payload.get("prefix_source_kinds"),
                name="prefix_source_kinds",
            ),
            projection_implementation_id=payload.get(
                "projection_implementation_id",
                "",
            ),
            projection_parameters_hash=payload.get(
                "projection_parameters_hash",
                "",
            ),
            loss_mask_semantics=str(
                payload.get("loss_mask_semantics", "")
            ),
            schema=str(payload.get("schema", "")),
        )

    @classmethod
    def from_v1(
        cls,
        contract: ModelContinuationDataContract,
    ) -> "ModelContinuationTrainingContractV2":
        if not isinstance(contract, ModelContinuationDataContract):
            raise TypeError("contract must be a v1 continuation contract.")
        return cls(
            prefix_source_kinds=contract.training_prefix_source_kinds,
            projection_implementation_id=(
                contract.training_projection_implementation_id
            ),
            projection_parameters_hash=(
                contract.training_projection_parameters_hash
            ),
            loss_mask_semantics=contract.loss_mask_semantics,
        )


@dataclass(frozen=True)
class ModelContinuationDataContractV2:
    """Artifact contract separating runtime support from training provenance."""

    conditioning_origin: str
    runtime: ModelContinuationRuntimeContractV2 | Mapping[str, Any]
    training: (
        ModelContinuationTrainingContractV2 | Mapping[str, Any] | None
    ) = None
    schema: str = MODEL_CONTINUATION_DATA_SCHEMA_V2

    def __post_init__(self) -> None:
        if str(self.schema) != MODEL_CONTINUATION_DATA_SCHEMA_V2:
            raise DatasetLayoutBindingError(
                f"Unsupported model continuation schema {self.schema!r}."
            )
        origin = _required(
            self.conditioning_origin,
            name="conditioning_origin",
        )
        if origin not in MODEL_CONTINUATION_PERSISTED_ORIGINS_V2:
            raise DatasetLayoutBindingError(
                f"Unsupported conditioning_origin {origin!r}."
            )
        runtime = self.runtime
        if isinstance(runtime, Mapping):
            runtime = ModelContinuationRuntimeContractV2.from_dict(runtime)
        elif not isinstance(runtime, ModelContinuationRuntimeContractV2):
            raise TypeError(
                "runtime must be a ModelContinuationRuntimeContractV2 "
                "or mapping."
            )
        training = self.training
        if isinstance(training, Mapping):
            training = ModelContinuationTrainingContractV2.from_dict(
                training
            )
        elif training is not None and not isinstance(
            training,
            ModelContinuationTrainingContractV2,
        ):
            raise TypeError(
                "training must be a ModelContinuationTrainingContractV2, "
                "mapping, or None."
            )
        if origin == MODEL_CONTINUATION_ORIGIN_INFERENCE_ONLY_V1:
            if training is not None:
                raise DatasetLayoutBindingError(
                    "Inference-only continuation requires training=null."
                )
        elif training is None:
            raise DatasetLayoutBindingError(
                "Trained-prefix continuation requires a training contract."
            )
        object.__setattr__(self, "conditioning_origin", origin)
        object.__setattr__(self, "runtime", runtime)
        object.__setattr__(self, "training", training)
        object.__setattr__(self, "schema", MODEL_CONTINUATION_DATA_SCHEMA_V2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "conditioning_origin": self.conditioning_origin,
            "runtime": self.runtime.to_dict(),
            "training": (
                None if self.training is None else self.training.to_dict()
            ),
        }

    @property
    def contract_hash(self) -> str:
        digest = canonical_content_hash(
            self.to_dict(),
            domain=MODEL_CONTINUATION_DATA_HASH_DOMAIN_V2,
        )
        return f"sha256:{digest}"

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "ModelContinuationDataContractV2":
        expected_keys = {
            "schema",
            "conditioning_origin",
            "runtime",
            "training",
        }
        unknown = sorted(set(payload).difference(expected_keys))
        if unknown:
            raise DatasetLayoutBindingError(
                f"Model continuation v2 contains unknown entries: {unknown}."
            )
        runtime = payload.get("runtime")
        if not isinstance(runtime, Mapping):
            raise DatasetLayoutBindingError(
                "Model continuation v2 runtime must be a mapping."
            )
        training = payload.get("training")
        if training is not None and not isinstance(training, Mapping):
            raise DatasetLayoutBindingError(
                "Model continuation v2 training must be a mapping or null."
            )
        return cls(
            conditioning_origin=str(
                payload.get("conditioning_origin", "")
            ),
            runtime=runtime,
            training=training,
            schema=str(payload.get("schema", "")),
        )


ModelContinuationContract = (
    ModelContinuationDataContract | ModelContinuationDataContractV2
)


@dataclass(frozen=True)
class NormalizedModelContinuationContract:
    """Non-persisted common view; source identity always remains authoritative."""

    source_schema: str
    source_contract_hash: str
    conditioning_origin: str
    runtime: ModelContinuationRuntimeContractV2
    training: ModelContinuationTrainingContractV2 | None

    def __post_init__(self) -> None:
        source_schema = _required(self.source_schema, name="source_schema")
        source_hash = _required(
            self.source_contract_hash,
            name="source_contract_hash",
        )
        origin = _required(
            self.conditioning_origin,
            name="conditioning_origin",
        )
        if not isinstance(self.runtime, ModelContinuationRuntimeContractV2):
            raise TypeError(
                "runtime must be a ModelContinuationRuntimeContractV2."
            )
        if self.training is not None and not isinstance(
            self.training,
            ModelContinuationTrainingContractV2,
        ):
            raise TypeError(
                "training must be a ModelContinuationTrainingContractV2 "
                "or None."
            )
        if source_schema == MODEL_CONTINUATION_DATA_SCHEMA_V1:
            if origin != MODEL_CONTINUATION_ORIGIN_LEGACY_V1_COMPAT:
                raise DatasetLayoutBindingError(
                    "A normalized v1 contract requires legacy_v1_compat."
                )
            if self.training is None:
                raise DatasetLayoutBindingError(
                    "A normalized v1 contract requires its legacy training view."
                )
        elif source_schema == MODEL_CONTINUATION_DATA_SCHEMA_V2:
            if origin not in MODEL_CONTINUATION_PERSISTED_ORIGINS_V2:
                raise DatasetLayoutBindingError(
                    "A normalized v2 contract requires an explicit v2 origin."
                )
            if (
                origin == MODEL_CONTINUATION_ORIGIN_INFERENCE_ONLY_V1
            ) != (self.training is None):
                raise DatasetLayoutBindingError(
                    "Normalized v2 origin/training semantics mismatch."
                )
        else:
            raise DatasetLayoutBindingError(
                f"Unsupported normalized source schema {source_schema!r}."
            )
        object.__setattr__(self, "source_schema", source_schema)
        object.__setattr__(self, "source_contract_hash", source_hash)
        object.__setattr__(self, "conditioning_origin", origin)


def model_continuation_contract_from_dict(
    payload: Mapping[str, Any],
) -> ModelContinuationContract:
    """Parse by schema without reserializing or promoting a v1 artifact."""

    if not isinstance(payload, Mapping):
        raise TypeError("model continuation contract must be a mapping.")
    schema = str(payload.get("schema", ""))
    if schema == MODEL_CONTINUATION_DATA_SCHEMA_V1:
        return ModelContinuationDataContract.from_dict(payload)
    if schema == MODEL_CONTINUATION_DATA_SCHEMA_V2:
        return ModelContinuationDataContractV2.from_dict(payload)
    raise DatasetLayoutBindingError(
        f"Unsupported model continuation schema {schema!r}."
    )


def normalize_model_continuation_contract(
    contract: ModelContinuationContract,
) -> NormalizedModelContinuationContract:
    """Return one runtime/training view while preserving the source hash."""

    if isinstance(contract, ModelContinuationDataContract):
        return NormalizedModelContinuationContract(
            source_schema=contract.schema,
            source_contract_hash=contract.contract_hash,
            conditioning_origin=MODEL_CONTINUATION_ORIGIN_LEGACY_V1_COMPAT,
            runtime=ModelContinuationRuntimeContractV2.from_v1(contract),
            training=ModelContinuationTrainingContractV2.from_v1(contract),
        )
    if isinstance(contract, ModelContinuationDataContractV2):
        return NormalizedModelContinuationContract(
            source_schema=contract.schema,
            source_contract_hash=contract.contract_hash,
            conditioning_origin=contract.conditioning_origin,
            runtime=contract.runtime,
            training=contract.training,
        )
    raise TypeError(
        "contract must be a ModelContinuationDataContract or "
        "ModelContinuationDataContractV2."
    )


__all__ = [
    "MODEL_CONTINUATION_DATA_HASH_DOMAIN_V2",
    "MODEL_CONTINUATION_DATA_HASH_DOMAIN_V1",
    "MODEL_CONTINUATION_DATA_HASH_KEY",
    "MODEL_CONTINUATION_DATA_METADATA_KEY",
    "MODEL_CONTINUATION_DATA_SCHEMA_V1",
    "MODEL_CONTINUATION_DATA_SCHEMA_V2",
    "MODEL_CONTINUATION_FLOW_INPAINT_V1",
    "MODEL_CONTINUATION_LOSS_ALL_VALID_V1",
    "MODEL_CONTINUATION_LOSS_UNCONDITIONED_VALID_V1",
    "MODEL_CONTINUATION_NATIVE_V1",
    "MODEL_CONTINUATION_ORIGIN_INFERENCE_ONLY_V1",
    "MODEL_CONTINUATION_ORIGIN_LEGACY_V1_COMPAT",
    "MODEL_CONTINUATION_ORIGIN_TRAINED_PREFIX_V1",
    "MODEL_CONTINUATION_REPRESENTATION_CONTINUOUS_V1",
    "MODEL_CONTINUATION_RUNTIME_SCHEMA_V2",
    "MODEL_CONTINUATION_SAMPLER_CLAMPED_V1",
    "MODEL_CONTINUATION_SAMPLER_WEIGHTED_V1",
    "MODEL_CONTINUATION_TRAINING_SCHEMA_V2",
    "MODEL_CONTINUATION_TRAINED_PREFIX_V1",
    "ModelContinuationContract",
    "RUNTIME_PREFIX_PRODUCER_LOCAL_OVERLAP_V1",
    "TRAINING_PREFIX_RECORDED_POLICY_OUTPUT_V1",
    "TRAINING_PREFIX_SYNTHETIC_CORRUPTION_V1",
    "TRAINING_PREFIX_TRAJECTORY_OVERLAP_TEACHER_V1",
    "ModelContinuationDataContract",
    "ModelContinuationDataContractV2",
    "ModelContinuationRuntimeContractV2",
    "ModelContinuationTrainingContractV2",
    "NormalizedModelContinuationContract",
    "model_continuation_contract_from_dict",
    "normalize_model_continuation_contract",
]
