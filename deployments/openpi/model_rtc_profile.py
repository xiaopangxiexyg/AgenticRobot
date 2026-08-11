"""Pure checked-in profile contract shared by OpenPI training and deployment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_interface.model_continuation import (
    ModelContinuationTrainingSamplingSpec,
)


PREPARED_MODEL_RTC_PROFILE_SCHEMA_V1 = (
    "dexteleop.openpi_prepared_model_rtc_training_profile.v1"
)
PREPARED_MODEL_RTC_PROFILE_HASH_DOMAIN_V1 = (
    "dexteleop.openpi_prepared_model_rtc_profile.v1"
)


def load_prepared_model_rtc_profile(path: str | Path) -> dict[str, Any]:
    """Load and normalize one default-off prepared model-RTC profile."""

    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != PREPARED_MODEL_RTC_PROFILE_SCHEMA_V1
    ):
        raise ValueError(f"Unsupported Prepared Skill model-RTC profile: {source}.")
    required = (
        "profile_id",
        "prepared_skill_profile_id",
        "output_config_name",
        "default_exp_name",
        "sampling_spec",
        "minimum_forecast_delay_steps",
        "maximum_forecast_delay_steps",
        "minimum_prefix_steps",
        "maximum_prefix_steps",
    )
    missing = [name for name in required if payload.get(name) is None]
    if missing:
        raise ValueError(f"Prepared model-RTC profile is missing: {missing}.")
    sampling = payload.get("sampling_spec")
    if not isinstance(sampling, Mapping):
        raise ValueError("Prepared model-RTC sampling_spec must be an object.")
    ModelContinuationTrainingSamplingSpec.from_dict(sampling)
    return payload


def prepared_model_rtc_profile_hash(profile: Mapping[str, Any]) -> str:
    """Return the canonical identity of a validated model-RTC profile."""

    return "sha256:" + canonical_content_hash(
        profile,
        domain=PREPARED_MODEL_RTC_PROFILE_HASH_DOMAIN_V1,
    )


__all__ = [
    "PREPARED_MODEL_RTC_PROFILE_HASH_DOMAIN_V1",
    "PREPARED_MODEL_RTC_PROFILE_SCHEMA_V1",
    "load_prepared_model_rtc_profile",
    "prepared_model_rtc_profile_hash",
]
