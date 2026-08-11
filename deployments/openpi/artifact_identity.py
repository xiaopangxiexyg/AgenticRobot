"""Pure identities shared by OpenPI training, evidence, and runtime adapters."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.policy_artifact import PolicyArtifactContract


OPENPI_RTC_ARTIFACT_HASH_DOMAIN_V1 = "dexteleop.openpi_rtc_artifact.v1"
OPENPI_RTC_CODEC_HASH_DOMAIN_V1 = "dexteleop.openpi_rtc_v22_codec.v1"
OPENPI_CONDITIONED_PROVIDER_CAPABILITY_SCHEMA_V1 = (
    "openpi.conditioned_rtc.provider_capability.v1"
)
OPENPI_CONDITIONED_RTC_WIRE_V1 = "openpi.conditioned_rtc.v1"
OPENPI_CONDITIONING_TRANSFORM_PROVIDER_V1 = "provider_v1"


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    normalized = int(value)
    if normalized != value or normalized <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return normalized


def _sha256(value: object, *, name: str) -> str:
    normalized = _required(value, name=name)
    digest = normalized[7:] if normalized.startswith("sha256:") else normalized
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a canonical SHA-256 identity.")
    return "sha256:" + digest


def openpi_rtc_artifact_hash(artifact: PolicyArtifactContract) -> str:
    """Hash the complete validated sidecar without importing an RTC runtime."""

    if not isinstance(artifact, PolicyArtifactContract):
        raise TypeError("artifact must be a PolicyArtifactContract.")
    return canonical_content_hash(
        artifact.to_dict(),
        domain=OPENPI_RTC_ARTIFACT_HASH_DOMAIN_V1,
    )


def openpi_conditioned_provider_capability_hash(
    provider: Mapping[str, Any],
) -> str:
    """Portable provider identity shared by training and runtime code."""

    if not isinstance(provider, Mapping):
        raise TypeError("provider capability must be a mapping.")
    schema = _required(
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
        != OPENPI_CONDITIONING_TRANSFORM_PROVIDER_V1
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
            OPENPI_CONDITIONING_TRANSFORM_PROVIDER_V1
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
        "sampler_constraint": _required(
            provider.get("sampler_constraint"),
            name="OpenPI sampler_constraint",
        ),
        "supports_unconditioned_bootstrap": supports_bootstrap,
        "continuation_implementation_id": _required(
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
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def openpi_v22_legacy_codec_hash(artifact: PolicyArtifactContract) -> str:
    """Bind the exact typed layout to the legacy normalized bridge."""

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
    """Bind the exact typed layout to its canonical action projection."""

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
    """Bind relative arm decode plus the canonical component projection."""

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


def openpi_rtc_component_scope(
    artifact: PolicyArtifactContract,
) -> tuple[str, ...]:
    """Return the ordered action-component scope without runtime imports."""

    layout = artifact.policy_io_layout
    names = (
        *(
            component.name
            for component in layout.components("action", "continuous")
        ),
        *(
            component.name
            for component in layout.components("action", "discrete")
        ),
    )
    return tuple(dict.fromkeys(str(name) for name in names))


__all__ = [
    "OPENPI_CONDITIONED_PROVIDER_CAPABILITY_SCHEMA_V1",
    "OPENPI_CONDITIONED_RTC_WIRE_V1",
    "OPENPI_CONDITIONING_TRANSFORM_PROVIDER_V1",
    "OPENPI_RTC_ARTIFACT_HASH_DOMAIN_V1",
    "OPENPI_RTC_CODEC_HASH_DOMAIN_V1",
    "openpi_rtc_artifact_hash",
    "openpi_conditioned_provider_capability_hash",
    "openpi_rtc_component_scope",
    "openpi_v22_legacy_codec_hash",
    "openpi_v22_native_codec_hash",
    "openpi_v22_relative_native_codec_hash",
]
