"""Client contract derivation for versioned policy artifact sidecars."""

from __future__ import annotations

from dataset_contracts.normalization import NormalizationContract
from dataset_contracts.policy_artifact import PolicyArtifactContract
from deployments.policy_artifact import load_policy_artifact

from .contracts import DeployPolicyContract


def deploy_contract_from_policy_artifact(
    artifact: PolicyArtifactContract,
    *,
    protocol_version: str | None = None,
    active_sides: tuple[str, ...] = ("left", "right"),
    required_camera_slots: tuple[str, ...] | None = None,
    strict_identity: bool = False,
) -> DeployPolicyContract:
    """Derive a server protocol contract from one semantic authority.

    Model identity is provenance, not wire compatibility.  By default the
    returned contract deliberately omits the artifact's model id and
    checkpoint step so a server can switch to another model instance that
    preserves the same protocol.  Callers performing an explicit provenance
    audit may set ``strict_identity=True``.
    """

    artifact_slots = (
        () if artifact.camera_input is None else artifact.camera_input.server_camera_slots
    )
    if required_camera_slots is not None:
        requested_slots = tuple(str(slot) for slot in required_camera_slots)
        if artifact.camera_input is not None and requested_slots != artifact_slots:
            raise ValueError(
                "required_camera_slots disagrees with policy_artifact camera_input."
            )
        camera_slots = requested_slots
    else:
        camera_slots = artifact_slots
    raw_normalization = artifact.metadata.get("normalization_contract")
    normalization = (
        None
        if raw_normalization is None
        else NormalizationContract.from_optional_mapping(
            raw_normalization,
            allow_legacy=True,
        )
    )
    return DeployPolicyContract(
        protocol_version=protocol_version,
        deploy_schema_version=artifact.deploy_schema_version,
        model_id=artifact.model_id if strict_identity else None,
        checkpoint_step=artifact.checkpoint_step if strict_identity else None,
        state_key=artifact.state_key,
        state_dim=artifact.state_input_dim,
        state_input_mode=artifact.state_input_mode,
        state_discrete_key=artifact.state_discrete_key,
        state_discrete_dim=artifact.policy_io_layout.state_discrete_dim,
        action_key=artifact.action_key,
        action_dim=artifact.policy_io_layout.action_dim,
        action_discrete_dim=artifact.policy_io_layout.action_discrete_dim,
        policy_io_layout_hash=artifact.layout_hash,
        active_sides=active_sides,
        required_camera_slots=camera_slots,
        hand_layout=None,
        normalization_hash=(
            None if normalization is None else normalization.contract_hash
        ),
        arm_hold_output_key=(
            None
            if artifact.arm_hold_output is None
            else artifact.arm_hold_output.output_key
        ),
        arm_hold_output_contract_hash=(
            None
            if artifact.arm_hold_output is None
            else artifact.arm_hold_output.contract_hash
        ),
    )


__all__ = ["deploy_contract_from_policy_artifact", "load_policy_artifact"]
