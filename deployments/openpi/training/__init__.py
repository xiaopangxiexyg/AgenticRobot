"""Optional OpenPI training adapters with lazy dependency loading."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "OPENPI_CAMERA_SEMANTIC_TRANSFORM_ID_V1": (".adapter",),
    "OPENPI_MODEL_RTC_ACTION_LOSS_MASK_KEY_V1": (".adapter",),
    "OPENPI_MODEL_RTC_PREFIX_MASK_KEY_V1": (".adapter",),
    "OPENPI_MODEL_RTC_DELAY_DEBUG_KEY_V1": (".adapter",),
    "OPENPI_MODEL_RTC_PREFIX_MASK_DEBUG_KEY_V1": (".adapter",),
    "OPENPI_MODEL_RTC_PREFIX_STEPS_DEBUG_KEY_V1": (".adapter",),
    "OPENPI_MODEL_RTC_TRAINER_IMPLEMENTATION_ID_V1": (".adapter",),
    "OPENPI_MODEL_RTC_TRAINER_IMPLEMENTATION_ID_V2": (".adapter",),
    "OpenPIModelRtcDataConfigFactory": (".adapter",),
    "OpenPIModelRtcTrainingTransform": (".adapter",),
    "OpenPICameraSemanticTransform": (".adapter",),
    "OpenPIOverlayCheckpointWeightLoader": (".adapter",),
    "build_openpi_model_rtc_training_manifest": (".adapter",),
    "build_openpi_trained_model_rtc_artifact_template": (".artifact",),
    "build_openpi_v22_absolute_model_rtc_artifact_template": (".artifact",),
    "MODEL_CONTINUATION_CHECKPOINT_RECEIPT_FILENAME_V1": (".evidence",),
    "MODEL_CONTINUATION_TRAINING_RUN_FILENAME_V1": (".evidence",),
    "OpenPIModelRtcCheckpointEvidence": (".evidence",),
    "verify_openpi_model_rtc_checkpoint": (".evidence",),
    "OPENPI_PREPARED_SKILL_ADAPTER_IMPLEMENTATION_V1": (".prepared_skill",),
    "OPENPI_PREPARED_SKILL_ADAPTER_IMPLEMENTATION_V2": (".prepared_skill",),
    "OPENPI_PREPARED_SKILL_BINDING_SCHEMA_V1": (".prepared_skill",),
    "OPENPI_SOURCE_MIXING_NATURAL_V1": (".prepared_skill",),
    "OpenPINaturalMergeSampler": (".prepared_skill",),
    "OpenPIPreparedEpisodeSelection": (".prepared_skill",),
    "OpenPIPreparedSkillBinding": (".prepared_skill",),
    "OpenPIPreparedSkillDataset": (".prepared_skill",),
    "OpenPIPreparedSkillError": (".prepared_skill",),
    "load_openpi_prepared_skill_binding": (".prepared_skill",),
    "materialize_openpi_prepared_skill_binding": (".prepared_skill",),
    "patch_openpi_prepared_skill_dataset": (".prepared_skill",),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module = import_module(target[0], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))


__all__ = sorted(_EXPORTS)
