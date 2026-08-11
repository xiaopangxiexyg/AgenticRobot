"""Default-off trained model-RTC adapter for one Prepared Skill config.

Prepared Skill natural merging globalizes sample identity after each OpenPI
child dataset has produced a sample.  The RTC mask transform must therefore
run at that wrapper boundary, not inside the child training-transform group.
This module binds that ordering while preserving the ordinary prepared config,
dataset, augmentations, optimizer, schedule, and base weight loader.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.model_continuation import (
    MODEL_CONTINUATION_FLOW_INPAINT_V1,
    MODEL_CONTINUATION_ORIGIN_INFERENCE_ONLY_V1,
    MODEL_CONTINUATION_SAMPLER_CLAMPED_V1,
    RUNTIME_PREFIX_PRODUCER_LOCAL_OVERLAP_V1,
    ModelContinuationDataContractV2,
    ModelContinuationRuntimeContractV2,
)
from dataset_contracts.model_continuation_training import (
    ModelContinuationTrainingSourceBinding,
)
from dataset_contracts.normalization import (
    NormalizationContract,
    normalization_stats_hash,
)
from dataset_contracts.policy_artifact import (
    PolicyArtifactContract,
    policy_artifact_from_metadata,
)
from dataset_interface.model_continuation import (
    ModelContinuationTrainingSamplingSpec,
)
from dataset_interface.model_continuation_training import (
    model_continuation_training_source_binding,
)
from deployments.openpi.client.normalization import (
    make_openpi_quantile_normalization_contract,
)
from deployments.openpi.artifact_identity import (
    OPENPI_CONDITIONED_PROVIDER_CAPABILITY_SCHEMA_V1,
    OPENPI_CONDITIONED_RTC_WIRE_V1,
    openpi_conditioned_provider_capability_hash,
    openpi_rtc_component_scope,
    openpi_v22_native_codec_hash,
)
from deployments.openpi.model_rtc_profile import (
    PREPARED_MODEL_RTC_PROFILE_SCHEMA_V1,
    load_prepared_model_rtc_profile,
)

from .adapter import (
    OpenPIModelRtcTrainingTransform,
    build_openpi_model_rtc_training_manifest,
)
from .artifact import build_openpi_trained_model_rtc_artifact_template
from .launch import (
    MODEL_CONTINUATION_TRAINING_RUN_HASH_METADATA_KEY_V1,
    MODEL_CONTINUATION_TRAINING_RUN_METADATA_KEY_V1,
    _train_config_hash,
)
from .prepared_skill import OpenPIPreparedSkillBinding


PREPARED_MODEL_RTC_SOURCE_MEDIA_HASH_DOMAIN_V1 = (
    "dexteleop.openpi_prepared_model_rtc_source_media.v1"
)


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_json(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _openpi_provider_capability(
    config: Any,
    artifact: PolicyArtifactContract,
) -> dict[str, Any]:
    """Derive the exact live OpenPI Pi0 descriptor without loading weights."""

    from openpi.models import pi0
    from openpi.policies import policy as openpi_policy

    descriptor_property = getattr(
        pi0.Pi0,
        "rtc_continuation_implementation",
        None,
    )
    descriptor_getter = getattr(descriptor_property, "fget", None)
    if not callable(descriptor_getter):
        raise ValueError("OpenPI Pi0 has no RTC implementation descriptor.")
    descriptor = descriptor_getter(None)
    if not isinstance(descriptor, Mapping):
        raise ValueError("OpenPI Pi0 RTC implementation descriptor is invalid.")
    implementation_id = str(descriptor.get("implementation_id") or "").strip()
    model_parameters = descriptor.get("parameters")
    if not implementation_id or not isinstance(model_parameters, Mapping):
        raise ValueError("OpenPI Pi0 RTC implementation facts are incomplete.")
    implementation_parameters = {
        "schema": openpi_policy.RTC_IMPLEMENTATION_PARAMETERS_SCHEMA_V1,
        "model_descriptor": dict(model_parameters),
        "effective_num_steps": 10,
        "internal_padding_rule": (
            "zero_horizon_then_existing_action_transforms"
        ),
        "final_external_projection": (
            "exact_selected_rows_after_output_transforms"
        ),
    }
    provider = {
        "provider_capability_schema": (
            OPENPI_CONDITIONED_PROVIDER_CAPABILITY_SCHEMA_V1
        ),
        "wire_protocol": OPENPI_CONDITIONED_RTC_WIRE_V1,
        "enabled": True,
        "conditioning_transform_owner": "provider_v1",
        "external_action_dim": artifact.policy_io_layout.action_dim,
        "internal_sampler_action_dim": int(config.model.action_dim),
        "prediction_horizon_steps": int(config.model.action_horizon),
        "sampler_constraint": MODEL_CONTINUATION_SAMPLER_CLAMPED_V1,
        "supports_unconditioned_bootstrap": True,
        "continuation_implementation_id": implementation_id,
        "continuation_implementation_parameters_hash": _sha256_json(
            implementation_parameters
        ),
    }
    provider["provider_capability_hash"] = (
        openpi_conditioned_provider_capability_hash(provider)
    )
    return provider


def _normalization_bound_source_artifact(
    config: Any,
    *,
    source_assets_root: Path,
) -> tuple[PolicyArtifactContract, Path, str]:
    if not isinstance(config.policy_metadata, Mapping):
        raise ValueError("Prepared OpenPI config has no policy metadata.")
    artifact = policy_artifact_from_metadata(config.policy_metadata)
    if artifact.checkpoint_step is not None:
        raise ValueError("Prepared base config must not impersonate a checkpoint.")
    if artifact.model_continuation is not None:
        raise ValueError("Prepared base artifact already has model continuation.")
    asset_id = str(
        getattr(config.data.assets, "asset_id", None)
        or getattr(config.data, "repo_id", "")
    ).strip()
    if not asset_id:
        raise ValueError("Prepared OpenPI config has no normalization asset id.")
    assets_root = Path(source_assets_root).expanduser().resolve()
    stats_path = assets_root / asset_id / "norm_stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(
            f"Prepared OpenPI norm_stats.json is missing: {stats_path}."
        )
    stats_hash = normalization_stats_hash(stats_path.read_bytes())
    normalization = make_openpi_quantile_normalization_contract(
        stats_ref=f"checkpoint://assets/{asset_id}/norm_stats.json",
        stats_hash=stats_hash,
    )
    parsed = NormalizationContract.from_optional_mapping(normalization)
    assert parsed is not None
    metadata = dict(artifact.metadata)
    if "normalization_contract" in metadata:
        raise ValueError("Prepared base artifact already binds normalization.")
    metadata["normalization_contract"] = normalization
    return dataclasses.replace(artifact, metadata=metadata), assets_root, parsed.contract_hash


def _prepared_source_binding(
    *,
    profile_id: str,
    binding: OpenPIPreparedSkillBinding,
    binding_root: Path,
    data_sources: tuple[Any, ...],
) -> ModelContinuationTrainingSourceBinding:
    camera_keys = tuple(key for _slot, key in binding.camera_slot_to_dataset_key)
    source_bindings = tuple(
        model_continuation_training_source_binding(
            source.dataset_root,
            camera_keys=camera_keys,
            source_id=str(source.name),
        )
        for source in data_sources
    )
    schemas = {item.dataset_schema for item in source_bindings}
    if len(schemas) != 1:
        raise ValueError("Prepared model-RTC sources use different dataset schemas.")
    manifest_path = binding_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Prepared Skill binding manifest is missing: {manifest_path}."
        )
    media_hash = canonical_content_hash(
        {
            "prepared_binding_hash": binding.binding_hash,
            "source_bindings": [item.to_dict() for item in source_bindings],
        },
        domain=PREPARED_MODEL_RTC_SOURCE_MEDIA_HASH_DOMAIN_V1,
    )
    return ModelContinuationTrainingSourceBinding(
        source_id=str(profile_id),
        dataset_schema=next(iter(schemas)),
        camera_keys=camera_keys,
        dataset_identity_hash="sha256:" + binding.binding_hash,
        episode_manifest_hash=_sha256_bytes(manifest_path.read_bytes()),
        media_index_hash="sha256:" + media_hash,
    )


@dataclasses.dataclass(frozen=True)
class PreparedModelRtcTrainConfig:
    config: Any
    transform: OpenPIModelRtcTrainingTransform
    summary: Mapping[str, Any]


def adapt_prepared_skill_model_rtc_config(
    *,
    base_config: Any,
    prepared_profile: Mapping[str, Any],
    model_rtc_profile: Mapping[str, Any],
    binding: OpenPIPreparedSkillBinding,
    binding_root: Path,
    source_assets_root: Path,
    checkpoint_base_root: Path,
    exp_name: str | None,
    resume: bool,
    overwrite: bool,
    num_train_steps: int | None,
    save_interval: int | None,
    adapter_debug: bool,
) -> PreparedModelRtcTrainConfig:
    expected_profile_id = str(
        model_rtc_profile["prepared_skill_profile_id"]
    )
    actual_profile_id = str(prepared_profile["profile_id"])
    if expected_profile_id != actual_profile_id:
        raise ValueError(
            "Prepared model-RTC profile targets a different Prepared Skill profile."
        )
    if tuple(getattr(base_config.data, "dataset_sources", ()) or ()) == ():
        raise ValueError("Prepared model-RTC requires explicit dataset_sources.")
    if str(base_config.data.action_pose_mode).strip().lower() != "absolute":
        raise ValueError("Prepared model-RTC v1 requires absolute actions.")
    horizon = int(base_config.model.action_horizon)
    minimum_delay = int(model_rtc_profile["minimum_forecast_delay_steps"])
    maximum_delay = int(model_rtc_profile["maximum_forecast_delay_steps"])
    minimum_prefix = int(model_rtc_profile["minimum_prefix_steps"])
    maximum_prefix = int(model_rtc_profile["maximum_prefix_steps"])
    if not (
        0 <= minimum_prefix <= maximum_prefix <= maximum_delay < horizon
        and 0 <= minimum_delay <= maximum_delay
    ):
        raise ValueError(
            "Prepared model-RTC delay/prefix range must leave an executable suffix."
        )
    sampling_spec = ModelContinuationTrainingSamplingSpec.from_dict(
        model_rtc_profile["sampling_spec"]
    )
    source_artifact, source_assets_root, normalization_hash = (
        _normalization_bound_source_artifact(
            base_config,
            source_assets_root=source_assets_root,
        )
    )
    provider = _openpi_provider_capability(base_config, source_artifact)
    frame_hashes = tuple(source_artifact.pose_frame_contract_hashes)
    if len(frame_hashes) != 1:
        raise ValueError(
            "Prepared model-RTC v1 requires one shared pose-frame contract."
        )
    runtime = ModelContinuationRuntimeContractV2(
        continuation_mode=MODEL_CONTINUATION_FLOW_INPAINT_V1,
        continuation_implementation_id=str(
            provider["continuation_implementation_id"]
        ),
        continuation_implementation_parameters_hash=str(
            provider["continuation_implementation_parameters_hash"]
        ),
        prediction_horizon_steps=horizon,
        minimum_forecast_delay_steps=minimum_delay,
        maximum_forecast_delay_steps=maximum_delay,
        minimum_prefix_steps=minimum_prefix,
        maximum_prefix_steps=maximum_prefix,
        model_action_dim=source_artifact.policy_io_layout.action_dim,
        model_action_discrete_dim=(
            source_artifact.policy_io_layout.action_discrete_dim
        ),
        component_scope=openpi_rtc_component_scope(source_artifact),
        supported_action_pose_modes=("absolute",),
        sampler_constraints=(str(provider["sampler_constraint"]),),
        runtime_prefix_source_roles=(
            RUNTIME_PREFIX_PRODUCER_LOCAL_OVERLAP_V1,
        ),
        action_codec_hash=openpi_v22_native_codec_hash(source_artifact),
        layout_hash=source_artifact.layout_hash,
        frame_contract_hash=frame_hashes[0],
        normalization_hash=normalization_hash,
        supports_unconditioned_bootstrap=minimum_prefix == 0,
    )
    inference_source = dataclasses.replace(
        source_artifact,
        model_continuation=ModelContinuationDataContractV2(
            conditioning_origin=MODEL_CONTINUATION_ORIGIN_INFERENCE_ONLY_V1,
            runtime=runtime,
            training=None,
        ),
    )
    if source_artifact.pose_action_chunk is None:
        raise ValueError("Prepared source artifact has no pose action chunk.")
    if source_artifact.pose_training_data is None:
        raise ValueError("Prepared source artifact has no pose training contract.")
    output_config_name = str(model_rtc_profile["output_config_name"])
    artifact_template = build_openpi_trained_model_rtc_artifact_template(
        inference_source,
        output_model_id=output_config_name,
        pose_action_chunk=source_artifact.pose_action_chunk,
        pose_training_data=source_artifact.pose_training_data,
        pose_frame_contract_hash=frame_hashes[0],
    )
    source_binding = _prepared_source_binding(
        profile_id=str(model_rtc_profile["profile_id"]),
        binding=binding,
        binding_root=binding_root,
        data_sources=tuple(base_config.data.dataset_sources),
    )
    effective_exp_name = str(
        exp_name or model_rtc_profile["default_exp_name"]
    )
    output_checkpoint_dir = (
        Path(checkpoint_base_root).expanduser().resolve()
        / output_config_name
        / effective_exp_name
    )
    effective_steps = (
        int(base_config.num_train_steps)
        if num_train_steps is None
        else int(num_train_steps)
    )
    effective_save_interval = (
        int(base_config.save_interval)
        if save_interval is None
        else int(save_interval)
    )
    if effective_steps <= 0 or effective_save_interval <= 0:
        raise ValueError("Model-RTC train/save steps must be positive.")
    output_data = dataclasses.replace(
        base_config.data,
        assets=dataclasses.replace(
            base_config.data.assets,
            assets_dir=str(source_assets_root),
        ),
    )
    overrides = {
        "num_train_steps": effective_steps,
        "save_interval": effective_save_interval,
        "keep_period": base_config.keep_period,
        "batch_size": int(base_config.batch_size),
        "num_workers": int(base_config.num_workers),
        "checkpoint_base_dir": str(base_config.checkpoint_base_dir),
        "assets_source_dir": str(source_assets_root),
        "prepared_skill_binding_hash": binding.binding_hash,
        "resume": bool(resume),
        "overwrite": bool(overwrite),
    }
    train_config_hash = _train_config_hash(
        base_config=base_config,
        output_config_name=output_config_name,
        exp_name=effective_exp_name,
        base_checkpoint_artifact=inference_source,
        artifact_template=artifact_template,
        sampling_spec=sampling_spec,
        source_binding_hash=source_binding.semantic_content_hash,
        camera_semantic_audit_hash=None,
        overrides=overrides,
    )
    manifest = build_openpi_model_rtc_training_manifest(
        output_model_id=output_config_name,
        artifact_template=artifact_template,
        base_checkpoint_artifact=inference_source,
        sampling_spec=sampling_spec,
        source_binding=source_binding,
        train_config_hash=train_config_hash,
    )
    transform = OpenPIModelRtcTrainingTransform(
        artifact=artifact_template,
        sampling_spec=sampling_spec,
        source_binding=source_binding,
        debug=bool(adapter_debug),
    )
    policy_metadata = dict(base_config.policy_metadata or {})
    policy_metadata["policy_artifact"] = artifact_template.to_dict()
    policy_metadata[MODEL_CONTINUATION_TRAINING_RUN_METADATA_KEY_V1] = (
        manifest.to_dict()
    )
    policy_metadata[
        MODEL_CONTINUATION_TRAINING_RUN_HASH_METADATA_KEY_V1
    ] = manifest.semantic_content_hash
    config = dataclasses.replace(
        base_config,
        name=output_config_name,
        exp_name=effective_exp_name,
        data=output_data,
        num_train_steps=effective_steps,
        save_interval=effective_save_interval,
        resume=bool(resume),
        overwrite=bool(overwrite),
        policy_metadata=policy_metadata,
    )
    summary = {
        "schema": "dexteleop_openpi_prepared_model_rtc_preflight_v1",
        "profile_id": str(model_rtc_profile["profile_id"]),
        "prepared_skill_profile_id": actual_profile_id,
        "prepared_skill_binding_hash": binding.binding_hash,
        "base_config_name": str(base_config.name),
        "base_weight_loader_unchanged": (
            config.weight_loader == base_config.weight_loader
        ),
        "data_factory_except_asset_source_unchanged": (
            dataclasses.replace(output_data, assets=base_config.data.assets)
            == base_config.data
        ),
        "source_assets_root": str(source_assets_root),
        "output_config_name": str(config.name),
        "exp_name": str(config.exp_name),
        "checkpoint_dir": str(output_checkpoint_dir),
        "num_train_steps": int(config.num_train_steps),
        "save_interval": int(config.save_interval),
        "batch_size": int(config.batch_size),
        "num_workers": int(config.num_workers),
        "source_count": len(tuple(base_config.data.dataset_sources)),
        "source_binding_hash": source_binding.semantic_content_hash,
        "sampling_spec_hash": sampling_spec.semantic_content_hash,
        "provider_capability_hash": provider["provider_capability_hash"],
        "provider_capability_origin": "openpi_live_code_descriptor_v1",
        "artifact_template_hash": manifest.policy_artifact_template_hash,
        "training_run_manifest_hash": manifest.semantic_content_hash,
        "train_config_hash": train_config_hash,
        "ordinary_prepared_config_unchanged": True,
        "rtc_transform_order": "after_prepared_global_identity_v1",
        "adapter_debug": bool(adapter_debug),
    }
    return PreparedModelRtcTrainConfig(
        config=config,
        transform=transform,
        summary=summary,
    )


__all__ = [
    "PREPARED_MODEL_RTC_PROFILE_SCHEMA_V1",
    "PreparedModelRtcTrainConfig",
    "adapt_prepared_skill_model_rtc_config",
    "load_prepared_model_rtc_profile",
]
