"""Rollback-safe OpenPI training projection for trained model-RTC.

The module does not import OpenPI at import time.  OpenPI-specific classes are
resolved only when a caller creates a real data config or loads checkpoint
weights, so DexTeleop contract tests remain framework independent.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.model_continuation import (
    MODEL_CONTINUATION_LOSS_UNCONDITIONED_VALID_V1,
    normalize_model_continuation_contract,
    require_trained_model_rtc_v1_contract,
)
from dataset_contracts.model_continuation_training import (
    ModelContinuationTrainingRunManifest,
    ModelContinuationTrainingSourceBinding,
)
from dataset_contracts.policy_artifact import PolicyArtifactContract
from dataset_interface.camera_semantics import DatasetCameraSemanticResolver
from dataset_interface.model_continuation import (
    ModelContinuationTrainingSamplingSpec,
    model_continuation_training_masks,
    select_model_continuation_training_delay_prefix,
)

from deployments.openpi.artifact_identity import openpi_rtc_artifact_hash


OPENPI_MODEL_RTC_ACTION_LOSS_MASK_KEY_V1 = (
    "model_rtc.action_loss_mask"
)
OPENPI_MODEL_RTC_PREFIX_MASK_DEBUG_KEY_V1 = (
    "model_rtc.debug_prefix_mask"
)
OPENPI_MODEL_RTC_DELAY_DEBUG_KEY_V1 = (
    "model_rtc.debug_forecast_delay_steps"
)
OPENPI_MODEL_RTC_PREFIX_STEPS_DEBUG_KEY_V1 = (
    "model_rtc.debug_prefix_steps"
)
OPENPI_MODEL_RTC_TRAINER_IMPLEMENTATION_ID_V1 = (
    "dexteleop.openpi.masked_flow_prefix_trainer.v1"
)
OPENPI_MODEL_RTC_TRAINER_PARAMETERS_HASH_DOMAIN_V1 = (
    "dexteleop.openpi.masked_flow_prefix_trainer_parameters.v1"
)
OPENPI_MODEL_RTC_SAMPLE_IDENTITY_KEY_V1 = (
    "_dexteleop_v22_augmentation_key"
)
OPENPI_CAMERA_SEMANTIC_TRANSFORM_ID_V1 = (
    "dexteleop.openpi.camera_semantic_transform.v1"
)


def _nonempty(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


@dataclass(frozen=True)
class OpenPICameraSemanticTransform:
    """Project stored dataset views into artifact-bound model image keys.

    Dataset media remains untouched. For every sample the recorded RealSense
    serial first resolves the canonical physical role to its stored dataset
    key; the policy artifact then routes that canonical role into the model's
    checkpoint-native input key.
    """

    artifact: PolicyArtifactContract
    resolver: DatasetCameraSemanticResolver
    identity_key: str = OPENPI_MODEL_RTC_SAMPLE_IDENTITY_KEY_V1
    stored_image_key_prefix: str = "observation.images"

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, PolicyArtifactContract):
            raise TypeError("artifact must be a PolicyArtifactContract.")
        if not isinstance(self.resolver, DatasetCameraSemanticResolver):
            raise TypeError("resolver must be a DatasetCameraSemanticResolver.")
        if self.artifact.camera_input is None:
            raise ValueError(
                "OpenPI camera semantic transform requires an artifact camera_input."
            )
        object.__setattr__(
            self,
            "identity_key",
            _nonempty(self.identity_key, name="identity_key"),
        )
        object.__setattr__(
            self,
            "stored_image_key_prefix",
            _nonempty(
                self.stored_image_key_prefix,
                name="stored_image_key_prefix",
            ).rstrip("."),
        )

    def __call__(self, data: Mapping[str, Any]) -> dict[str, Any]:
        output = dict(data)
        identity = np.asarray(output.get(self.identity_key))
        if identity.shape != (2,):
            raise ValueError(
                f"{self.identity_key} must contain [episode_index, row_index]."
            )
        episode_index = int(identity[0])
        if episode_index < 0:
            raise ValueError("OpenPI camera sample episode_index must be non-negative.")
        resolution = self.resolver.resolution_for_episode(episode_index)
        source_values: dict[str, Any] = {}
        assert self.artifact.camera_input is not None
        for camera in self.artifact.camera_input.inputs:
            canonical_role = camera.source_key
            stored_key = (
                resolution.stored_key_for(canonical_role)
                if canonical_role in self.resolver.binding.canonical_roles
                else canonical_role
            )
            stored_input_key = f"{self.stored_image_key_prefix}.{stored_key}"
            if stored_input_key not in data:
                raise ValueError(
                    "OpenPI stored camera input is missing: "
                    f"episode={episode_index} canonical_role={canonical_role!r} "
                    f"stored_key={stored_key!r} input_key={stored_input_key!r}."
                )
            source_values[camera.model_input_key] = data[stored_input_key]
        output.update(source_values)
        return output


@dataclass
class OpenPIModelRtcTrainingTransform:
    """Add the optional OpenPI action-loss mask to one normalized sample."""

    artifact: PolicyArtifactContract
    sampling_spec: ModelContinuationTrainingSamplingSpec
    source_binding: ModelContinuationTrainingSourceBinding
    debug: bool = False
    identity_key: str = OPENPI_MODEL_RTC_SAMPLE_IDENTITY_KEY_V1
    _epoch: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, PolicyArtifactContract):
            raise TypeError("artifact must be a PolicyArtifactContract.")
        if not isinstance(
            self.sampling_spec,
            ModelContinuationTrainingSamplingSpec,
        ):
            raise TypeError(
                "sampling_spec must be a ModelContinuationTrainingSamplingSpec."
            )
        if not isinstance(
            self.source_binding,
            ModelContinuationTrainingSourceBinding,
        ):
            raise TypeError(
                "source_binding must be a ModelContinuationTrainingSourceBinding."
            )
        normalized = require_trained_model_rtc_v1_contract(
            self.artifact.model_continuation
        )
        training = normalized.training
        assert training is not None
        if training.loss_mask_semantics != (
            MODEL_CONTINUATION_LOSS_UNCONDITIONED_VALID_V1
        ):
            raise ValueError(
                "OpenPI trained model-RTC v1 requires loss on the "
                "unconditioned valid suffix."
            )
        self.identity_key = _nonempty(
            self.identity_key,
            name="identity_key",
        )
        self.debug = bool(self.debug)
        self._epoch = max(0, int(self._epoch))

    def set_epoch(self, epoch: int) -> None:
        self._epoch = max(0, int(epoch))

    def __call__(self, data: Mapping[str, Any]) -> dict[str, Any]:
        output = dict(data)
        actions = np.asarray(output.get("actions"), dtype=np.float32)
        normalized = normalize_model_continuation_contract(
            self.artifact.model_continuation
        )
        horizon = normalized.runtime.prediction_horizon_steps
        if actions.ndim != 2 or actions.shape[0] != horizon:
            raise ValueError(
                "OpenPI model-RTC training actions must have shape "
                f"[{horizon}, action_dim]."
            )
        identity = np.asarray(output.get(self.identity_key))
        if identity.shape != (2,):
            raise ValueError(
                f"{self.identity_key} must contain [episode_index, row_index]."
            )
        episode_index = int(identity[0])
        source_start_index = int(identity[1])
        if episode_index < 0 or source_start_index < 0:
            raise ValueError("OpenPI sample identity must be non-negative.")
        source_episode_id = (
            f"{self.source_binding.semantic_content_hash}:"
            f"episode_{episode_index:06d}"
        )
        target_is_pad = output.get("action_is_pad")
        if target_is_pad is None:
            valid_action_count = horizon
        else:
            padding = np.asarray(target_is_pad, dtype=bool)
            if padding.shape != (horizon,):
                raise ValueError(
                    "OpenPI action_is_pad must match the action horizon."
                )
            valid_action_count = int(np.count_nonzero(~padding))
            if not np.array_equal(
                padding,
                np.arange(horizon) >= valid_action_count,
            ):
                raise ValueError(
                    "OpenPI action_is_pad must be a contiguous false-prefix."
                )
        delay, prefix_steps = (
            select_model_continuation_training_delay_prefix(
                self.sampling_spec,
                self.artifact,
                source_episode_id=source_episode_id,
                source_start_index=source_start_index,
                draw_index=self._epoch,
                valid_action_count=valid_action_count,
            )
        )
        prefix_mask, expected_padding, loss_mask = (
            model_continuation_training_masks(
                self.artifact,
                forecast_delay_steps=delay,
                prefix_steps=prefix_steps,
                valid_action_count=valid_action_count,
            )
        )
        if target_is_pad is not None and not np.array_equal(
            expected_padding,
            np.asarray(target_is_pad, dtype=bool),
        ):
            raise ValueError(
                "OpenPI padding differs from the continuation contract."
            )
        output[OPENPI_MODEL_RTC_ACTION_LOSS_MASK_KEY_V1] = loss_mask
        if self.debug:
            output[OPENPI_MODEL_RTC_PREFIX_MASK_DEBUG_KEY_V1] = prefix_mask
            output[OPENPI_MODEL_RTC_DELAY_DEBUG_KEY_V1] = np.asarray(
                delay,
                dtype=np.int32,
            )
            output[OPENPI_MODEL_RTC_PREFIX_STEPS_DEBUG_KEY_V1] = np.asarray(
                prefix_steps,
                dtype=np.int32,
            )
        return output


@dataclass(frozen=True)
class OpenPIModelRtcDataConfigFactory:
    """Prepend one training-only transform to an existing OpenPI data config."""

    base_factory: Any
    transform: OpenPIModelRtcTrainingTransform
    camera_transform: OpenPICameraSemanticTransform | None = None

    def create(self, assets_dirs: Path, model_config: Any) -> Any:
        if not callable(getattr(self.base_factory, "create", None)):
            raise TypeError("base_factory must expose create().")
        data_config = self.base_factory.create(assets_dirs, model_config)
        training_transforms = getattr(
            data_config,
            "training_transforms",
            None,
        )
        if training_transforms is None:
            raise ValueError(
                "The loaded OpenPI version has no training_transforms seam."
            )
        from openpi import transforms as openpi_transforms

        prepended = (
            (self.transform,)
            if self.camera_transform is None
            else (self.camera_transform, self.transform)
        )
        adapted = openpi_transforms.Group(
            inputs=(*prepended, *tuple(training_transforms.inputs)),
            outputs=tuple(training_transforms.outputs),
        )
        return replace(data_config, training_transforms=adapted)


@dataclass(frozen=True)
class OpenPIOverlayCheckpointWeightLoader:
    """Overlay one partial OpenPI checkpoint on its original base loader."""

    base_loader: Any
    params_path: str

    def __post_init__(self) -> None:
        if not callable(getattr(self.base_loader, "load", None)):
            raise TypeError("base_loader must expose load().")
        object.__setattr__(
            self,
            "params_path",
            _nonempty(self.params_path, name="params_path"),
        )

    def load(self, params: Any) -> Any:
        import flax.traverse_util as traverse_util
        import numpy as np
        from openpi.models import model as openpi_model
        from openpi.shared import download

        base = self.base_loader.load(params)
        overlay = openpi_model.restore_params(
            download.maybe_download(self.params_path),
            restore_type=np.ndarray,
        )
        flat_base = traverse_util.flatten_dict(base)
        flat_overlay = traverse_util.flatten_dict(overlay)
        unknown = sorted(
            "/".join(key)
            for key in flat_overlay
            if key not in flat_base
        )
        if unknown:
            raise ValueError(
                "OpenPI overlay checkpoint contains unknown parameters: "
                f"{unknown[:10]}"
            )
        for key, value in flat_overlay.items():
            reference = flat_base[key]
            flat_base[key] = (
                value.astype(reference.dtype)
                if value.dtype != reference.dtype
                else value
            )
        return traverse_util.unflatten_dict(flat_base)


def build_openpi_model_rtc_training_manifest(
    *,
    output_model_id: str,
    artifact_template: PolicyArtifactContract,
    base_checkpoint_artifact: PolicyArtifactContract,
    sampling_spec: ModelContinuationTrainingSamplingSpec,
    source_binding: ModelContinuationTrainingSourceBinding,
    train_config_hash: str,
) -> ModelContinuationTrainingRunManifest:
    """Build the framework-independent evidence embedded in OpenPI metadata."""

    require_trained_model_rtc_v1_contract(
        artifact_template.model_continuation
    )
    trainer_parameters_hash = canonical_content_hash(
        {
            "action_loss_mask_key": (
                OPENPI_MODEL_RTC_ACTION_LOSS_MASK_KEY_V1
            ),
            "mask_reduction": "sum_loss_times_mask_over_sum_mask_v1",
            "prefix_projection": "target_equals_teacher_prefix_v1",
            "sampling_draw_index": "data_epoch_v1",
            "ordinary_batch_fallback": "unmasked_mean_v1",
        },
        domain=OPENPI_MODEL_RTC_TRAINER_PARAMETERS_HASH_DOMAIN_V1,
    )
    return ModelContinuationTrainingRunManifest(
        output_model_id=output_model_id,
        policy_artifact_template_hash=openpi_rtc_artifact_hash(
            artifact_template
        ),
        base_checkpoint_identity_hash=openpi_rtc_artifact_hash(
            base_checkpoint_artifact
        ),
        sampling_spec_hash=sampling_spec.semantic_content_hash,
        trainer_implementation_id=(
            OPENPI_MODEL_RTC_TRAINER_IMPLEMENTATION_ID_V1
        ),
        trainer_parameters_hash=trainer_parameters_hash,
        train_config_hash=train_config_hash,
        source_bindings=(source_binding,),
    )


__all__ = [
    "OPENPI_MODEL_RTC_ACTION_LOSS_MASK_KEY_V1",
    "OPENPI_MODEL_RTC_DELAY_DEBUG_KEY_V1",
    "OPENPI_MODEL_RTC_PREFIX_MASK_DEBUG_KEY_V1",
    "OPENPI_MODEL_RTC_PREFIX_STEPS_DEBUG_KEY_V1",
    "OPENPI_MODEL_RTC_SAMPLE_IDENTITY_KEY_V1",
    "OPENPI_CAMERA_SEMANTIC_TRANSFORM_ID_V1",
    "OPENPI_MODEL_RTC_TRAINER_IMPLEMENTATION_ID_V1",
    "OpenPIModelRtcDataConfigFactory",
    "OpenPIModelRtcTrainingTransform",
    "OpenPICameraSemanticTransform",
    "OpenPIOverlayCheckpointWeightLoader",
    "build_openpi_model_rtc_training_manifest",
]
