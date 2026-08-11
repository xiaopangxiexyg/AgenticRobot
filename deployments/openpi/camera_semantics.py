"""Explicit policy-artifact camera routes for physical wrist semantics."""

from __future__ import annotations

from dataclasses import replace

from dataset_contracts.camera_semantics import CameraRoleBinding
from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.policy_artifact import (
    PolicyArtifactContract,
    PolicyCameraInput,
    PolicyCameraInputContract,
)


LEGACY_SWAPPED_WRIST_CAMERA_SEMANTICS_V1 = (
    "legacy_swapped_wrist_capture_v1"
)
CANONICAL_WRIST_CAMERA_SEMANTICS_V1 = "canonical_physical_wrist_v1"
CAMERA_INPUT_ROUTING_HASH_DOMAIN_V1 = (
    "dexteleop.policy_camera_input_routing.v1"
)
POLICY_ARTIFACT_BASE_HASH_DOMAIN_V1 = (
    "dexteleop.policy_artifact_base_identity.v1"
)
_COMPAT_METADATA_KEYS = frozenset(
    {
        "camera_semantics",
        "camera_role_binding_hash",
        "camera_input_routing_hash",
        "base_policy_artifact_hash",
    }
)


def camera_input_routing_hash(contract: PolicyCameraInputContract) -> str:
    return canonical_content_hash(
        contract.to_dict(),
        domain=CAMERA_INPUT_ROUTING_HASH_DOMAIN_V1,
    )


def policy_artifact_base_hash(artifact: PolicyArtifactContract) -> str:
    return canonical_content_hash(
        artifact.to_dict(),
        domain=POLICY_ARTIFACT_BASE_HASH_DOMAIN_V1,
    )


def _legacy_swapped_wrist_route(
    camera: PolicyCameraInputContract,
) -> PolicyCameraInputContract:
    by_source = {item.source_key: item for item in camera.inputs}
    if set(by_source) != {"left_wrist", "right_wrist"}:
        raise ValueError(
            "Legacy wrist route requires exactly left_wrist/right_wrist sources."
        )
    return PolicyCameraInputContract(
        inputs=(
            PolicyCameraInput(
                source_key="right_wrist",
                model_input_key=by_source["left_wrist"].model_input_key,
                server_camera_slot=by_source["left_wrist"].server_camera_slot,
            ),
            PolicyCameraInput(
                source_key="left_wrist",
                model_input_key=by_source["right_wrist"].model_input_key,
                server_camera_slot=by_source["right_wrist"].server_camera_slot,
            ),
        ),
        required=camera.required,
        color_space=camera.color_space,
        layout=camera.layout,
        dtype=camera.dtype,
    )


def validate_legacy_swapped_wrist_route(
    base_camera: PolicyCameraInputContract,
    routed_camera: PolicyCameraInputContract,
) -> None:
    expected = _legacy_swapped_wrist_route(base_camera)
    if routed_camera.to_dict() != expected.to_dict():
        raise ValueError(
            "Legacy wrist compatibility route does not preserve the checkpoint's "
            "model input keys and server slots with the exact physical-role swap."
        )


def validate_declared_camera_semantics(
    artifact: PolicyArtifactContract,
) -> None:
    metadata = artifact.metadata
    semantics = str(metadata.get("camera_semantics") or "").strip()
    present = _COMPAT_METADATA_KEYS.intersection(metadata)
    if not semantics:
        if present:
            raise ValueError(
                "Policy artifact has partial camera compatibility metadata."
            )
        return
    if semantics != LEGACY_SWAPPED_WRIST_CAMERA_SEMANTICS_V1:
        raise ValueError(f"Unsupported camera semantics {semantics!r}.")
    if artifact.camera_input is None:
        raise ValueError("Legacy wrist semantics require camera_input.")
    for key in (
        "camera_role_binding_hash",
        "camera_input_routing_hash",
        "base_policy_artifact_hash",
    ):
        if not str(metadata.get(key) or "").strip():
            raise ValueError(f"Legacy wrist compatibility metadata lacks {key}.")
    if str(metadata["camera_input_routing_hash"]) != camera_input_routing_hash(
        artifact.camera_input
    ):
        raise ValueError(
            "Legacy wrist compatibility camera_input_routing_hash is invalid."
        )


def validate_legacy_swapped_wrist_artifact(
    base_artifact: PolicyArtifactContract,
    compat_artifact: PolicyArtifactContract,
) -> None:
    validate_declared_camera_semantics(compat_artifact)
    if base_artifact.camera_input is None or compat_artifact.camera_input is None:
        raise ValueError("Legacy wrist compatibility requires camera_input.")
    validate_legacy_swapped_wrist_route(
        base_artifact.camera_input,
        compat_artifact.camera_input,
    )
    if str(compat_artifact.metadata["base_policy_artifact_hash"]) != (
        policy_artifact_base_hash(base_artifact)
    ):
        raise ValueError(
            "Compatibility policy artifact is not bound to the selected "
            "checkpoint policy artifact."
        )
    expected_metadata = dict(base_artifact.metadata)
    expected_metadata.update(
        {
            "camera_semantics": LEGACY_SWAPPED_WRIST_CAMERA_SEMANTICS_V1,
            "camera_role_binding_hash": compat_artifact.metadata[
                "camera_role_binding_hash"
            ],
            "camera_input_routing_hash": camera_input_routing_hash(
                compat_artifact.camera_input
            ),
            "base_policy_artifact_hash": policy_artifact_base_hash(base_artifact),
        }
    )
    expected = replace(
        base_artifact,
        camera_input=compat_artifact.camera_input,
        metadata=expected_metadata,
    )
    if compat_artifact.to_dict() != expected.to_dict():
        raise ValueError(
            "Compatibility policy artifact changes checkpoint-native fields "
            "outside the declared wrist camera route."
        )


def legacy_swapped_wrist_artifact(
    base_artifact: PolicyArtifactContract,
    binding: CameraRoleBinding,
) -> PolicyArtifactContract:
    """Derive a non-mutating runtime/training route for a legacy checkpoint.

    The checkpoint's model input keys and server slots remain unchanged. Only
    the canonical runtime source bound to each checkpoint-native model input is
    permuted so the model continues receiving the physical view seen during
    its historical training.
    """

    if not isinstance(base_artifact, PolicyArtifactContract):
        raise TypeError("base_artifact must be a PolicyArtifactContract.")
    if not isinstance(binding, CameraRoleBinding):
        raise TypeError("binding must be a CameraRoleBinding.")
    camera = base_artifact.camera_input
    if camera is None:
        raise ValueError("Legacy wrist route requires artifact camera_input.")
    if _COMPAT_METADATA_KEYS.intersection(base_artifact.metadata):
        raise ValueError(
            "Source artifact already declares camera compatibility semantics."
        )
    if set(binding.canonical_roles) != {"left_wrist", "right_wrist"}:
        raise ValueError(
            "Legacy wrist route requires exactly left_wrist/right_wrist roles."
        )
    routed = _legacy_swapped_wrist_route(camera)
    metadata = dict(base_artifact.metadata)
    metadata.update(
        {
            "camera_semantics": LEGACY_SWAPPED_WRIST_CAMERA_SEMANTICS_V1,
            "camera_role_binding_hash": binding.contract_hash,
            "camera_input_routing_hash": camera_input_routing_hash(routed),
            "base_policy_artifact_hash": policy_artifact_base_hash(base_artifact),
        }
    )
    return replace(
        base_artifact,
        camera_input=routed,
        metadata=metadata,
    )


__all__ = [
    "CAMERA_INPUT_ROUTING_HASH_DOMAIN_V1",
    "CANONICAL_WRIST_CAMERA_SEMANTICS_V1",
    "LEGACY_SWAPPED_WRIST_CAMERA_SEMANTICS_V1",
    "POLICY_ARTIFACT_BASE_HASH_DOMAIN_V1",
    "camera_input_routing_hash",
    "policy_artifact_base_hash",
    "validate_declared_camera_semantics",
    "validate_legacy_swapped_wrist_artifact",
    "validate_legacy_swapped_wrist_route",
    "legacy_swapped_wrist_artifact",
]
