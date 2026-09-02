"""Checkpoint-bound deployment binding for prepared OpenPI training configs.

The training profile, checkpoint policy artifact, derived OpenPI config name,
asset id, and exact normalization-statistics bytes are one deployment unit.
This module materializes only a disposable compatibility sidecar; it never
modifies the checkpoint, OpenPI checkout, dataset, or any teleoperation config.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.model_continuation import (
    ModelContinuationDataContractV2,
    require_trained_model_rtc_v1_contract,
)
from dataset_contracts.normalization import normalization_stats_hash
from dataset_contracts.policy_artifact import PolicyArtifactContract
from deployments.openpi.client.normalization import (
    make_openpi_quantile_normalization_contract,
)
from deployments.openpi.model_rtc_profile import (
    load_prepared_model_rtc_profile,
    prepared_model_rtc_profile_hash,
)
from deployments.openpi.image_wire import (
    OPENPI_IMAGE_WIRE_TRAINING_METADATA_KEY_V1,
    OpenPIImageWireTrainingTransform,
)
from deployments.openpi.state_only import (
    OPENPI_STATE_ONLY_CAPABILITY_KEY_V1,
    OPENPI_STATE_ONLY_PROFILE_V1,
    OpenPIStateOnlyInputBridge,
)
from deployments.openpi.training.evidence import (
    OpenPIModelRtcCheckpointEvidence,
    verify_openpi_model_rtc_checkpoint,
)
from dataset_interface.model_continuation import (
    ModelContinuationTrainingSamplingSpec,
)
from deployments.policy_artifact import (
    load_policy_artifact,
    load_policy_artifact_payload,
)


PREPARED_TRAINING_PROFILE_SCHEMA_V1 = (
    "dexteleop.openpi_prepared_skill_training_profile.v1"
)
PREPARED_PROMPT_TRAINING_PROFILE_SCHEMA_V1 = (
    "dexteleop.openpi_prepared_prompt_training_profile.v1"
)
PREPARED_PROMPT_FAMILY_TRAINING_PROFILE_SCHEMA_V2 = (
    "dexteleop.openpi_prepared_prompt_family_training_profile.v2"
)
PREPARED_STAGE1_SCENARIO_TRAINING_PROFILE_SCHEMA_V1 = (
    "dexteleop.openpi_stage1_scenario_training_profile.v1"
)
PREPARED_DEPLOYMENT_BINDING_SCHEMA_V1 = (
    "dexteleop.openpi_prepared_deployment_binding.v1"
)
PREPARED_MODEL_RTC_DEPLOYMENT_BINDING_SCHEMA_V1 = (
    "dexteleop.openpi_prepared_model_rtc_deployment_binding.v1"
)
PREPARED_DEPLOYMENT_BINDING_METADATA_KEY = (
    "openpi_prepared_deployment_binding"
)
PREPARED_DEPLOYMENT_PROFILE_HASH_DOMAIN_V1 = (
    "dexteleop.openpi_prepared_deployment_profile.v1"
)
PREPARED_DEPLOYMENT_CHECKPOINT_ARTIFACT_HASH_DOMAIN_V1 = (
    "dexteleop.openpi_prepared_deployment_checkpoint_artifact.v1"
)
PREPARED_DEPLOYMENT_BINDING_HASH_DOMAIN_V1 = (
    "dexteleop.openpi_prepared_deployment_binding.v1"
)
PREPARED_PROMPT_CONTRACT_HASH_DOMAIN_V1 = (
    "dexteleop.openpi.prepared_prompt_contract.v1"
)
STAGE1_SCENARIO_SELECTION_POLICY_V1 = (
    "coarse_stage1_nominal_non_abnormal_latest_annotation_v1"
)

_STAGE1_SCENARIO_REPEAT_WEIGHTS = {
    "fixed_middle": 1,
    "initial_multi_choice": 4,
    "midcourse_interrupt": 6,
}

_PROFILE_REQUIRED_KEYS = (
    "profile_id",
    "sources",
    "target_skill_id",
    "camera_slot_to_dataset_key",
    "action_horizon",
    "action_pose_mode",
    "tail_policy",
    "expected_policy_io_layout_hash",
    "openpi_checkout",
    "openpi_base_config",
    "openpi_config_name",
    "openpi_asset_id",
)


@dataclass(frozen=True)
class PreparedDeploymentResolution:
    checkpoint_dir: Path
    checkpoint_artifact_path: Path
    deployment_artifact_path: Path
    checkpoint_artifact: PolicyArtifactContract
    deployment_artifact: PolicyArtifactContract
    checkpoint_payload: Mapping[str, Any]
    deployment_payload: Mapping[str, Any]
    profile_path: Path
    profile: Mapping[str, Any]
    binding: Mapping[str, Any]
    norm_stats_path: Path
    prepared_binding_root: Path | None = None
    prepared_binding_payload: Mapping[str, Any] | None = None
    prepared_source_manifests: tuple[Mapping[str, Any], ...] = ()
    model_rtc_profile_path: Path | None = None
    model_rtc_profile: Mapping[str, Any] | None = None

    @property
    def openpi_config_name(self) -> str:
        if self.model_rtc_profile is not None:
            return str(self.model_rtc_profile["output_config_name"])
        return str(self.profile["openpi_config_name"])


def _json_object(path: str | Path, *, name: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must contain a JSON object: {source}")
    return {str(key): value for key, value in payload.items()}


def load_prepared_training_profile(path: str | Path) -> dict[str, Any]:
    """Load the checked-in training profile used to derive a model config."""

    source = Path(path).expanduser().resolve()
    profile = _json_object(source, name="prepared training profile")
    schema = profile.get("schema")
    if schema not in {
        PREPARED_TRAINING_PROFILE_SCHEMA_V1,
        PREPARED_PROMPT_TRAINING_PROFILE_SCHEMA_V1,
        PREPARED_PROMPT_FAMILY_TRAINING_PROFILE_SCHEMA_V2,
        PREPARED_STAGE1_SCENARIO_TRAINING_PROFILE_SCHEMA_V1,
    }:
        raise ValueError(f"Unsupported prepared training profile: {source}")
    required_keys = list(_PROFILE_REQUIRED_KEYS)
    prompt_profile = schema in {
        PREPARED_PROMPT_TRAINING_PROFILE_SCHEMA_V1,
        PREPARED_PROMPT_FAMILY_TRAINING_PROFILE_SCHEMA_V2,
    }
    if prompt_profile:
        required_keys.remove("target_skill_id")
        required_keys.extend(
            (
                "prompt_stages",
                "expected_total_window_count",
                "expected_union_episode_count",
                "prompt_stage_sampling_policy",
                "prompt_stage_sampling_seed",
                "expected_effective_epoch_window_count",
            )
        )
    elif schema == PREPARED_STAGE1_SCENARIO_TRAINING_PROFILE_SCHEMA_V1:
        required_keys.extend(
            (
                "target_progress_step_id",
                "expected_total_raw_episode_count",
                "expected_total_raw_frame_count",
                "expected_total_eligible_episode_count",
                "expected_prepared_interval_window_count",
                "expected_prepared_interval_repeated_window_count",
                "expected_openpi_quality_valid_window_count",
                "expected_openpi_repeated_epoch_sample_count",
                "source_mixing_policy",
                "default_prompt",
            )
        )
    missing = [key for key in required_keys if not profile.get(key)]
    if missing:
        raise ValueError(
            "Prepared training profile is missing required fields: "
            + ", ".join(missing)
        )
    sources = profile.get("sources")
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
        raise ValueError("Prepared training profile sources must be a sequence.")
    if not sources or any(not isinstance(item, Mapping) for item in sources):
        raise ValueError("Prepared training profile requires source mappings.")
    cameras = profile.get("camera_slot_to_dataset_key")
    if not isinstance(cameras, Mapping) or not cameras:
        raise ValueError(
            "Prepared training profile camera_slot_to_dataset_key must be a mapping."
        )
    if str(profile["action_pose_mode"]) not in {"absolute", "relative"}:
        raise ValueError("Prepared training profile action_pose_mode is unsupported.")
    if int(profile["action_horizon"]) <= 0:
        raise ValueError("Prepared training profile action_horizon must be positive.")
    if prompt_profile:
        prompt_stages = profile.get("prompt_stages")
        if (
            not isinstance(prompt_stages, Sequence)
            or isinstance(prompt_stages, (str, bytes))
            or len(prompt_stages) < 2
            or any(not isinstance(item, Mapping) for item in prompt_stages)
        ):
            raise ValueError(
                "Prepared prompt training profile requires prompt_stages."
            )
        prompt_key = (
            "prompt"
            if schema == PREPARED_PROMPT_TRAINING_PROFILE_SCHEMA_V1
            else "canonical_prompt"
        )
        for key in ("stage_id", "skill_id", prompt_key):
            values = tuple(str(item.get(key) or "").strip() for item in prompt_stages)
            if any(not value for value in values) or len(set(values)) != len(values):
                raise ValueError(
                    f"Prepared prompt training profile has invalid {key} values."
                )
        if any(
            int(item.get("expected_window_count", 0)) <= 0
            or int(item.get("expected_openpi_window_count", 0)) <= 0
            for item in prompt_stages
        ):
            raise ValueError(
                "Prepared prompt training profile requires positive prepared "
                "and OpenPI window counts for every stage."
            )
        if schema == PREPARED_PROMPT_FAMILY_TRAINING_PROFILE_SCHEMA_V2:
            required_family = (
                "prompt_family_sampling_policy",
                "prompt_family_sampling_seed",
                "canonical_prompt_percent",
                "training_variant_prompt_percent",
                "max_full_token_length",
                "prompt_state_dimension",
            )
            missing_family = [
                key for key in required_family if not profile.get(key)
            ]
            if missing_family:
                raise ValueError(
                    "Prepared prompt-family profile is missing required fields: "
                    + ", ".join(missing_family)
                )
            prompt_values: list[str] = []
            for item in prompt_stages:
                training_variants = tuple(
                    str(value).strip()
                    for value in item.get("training_prompt_variants", ())
                )
                held_out_variants = tuple(
                    str(value).strip()
                    for value in item.get("held_out_prompt_variants", ())
                )
                if (
                    len(training_variants) != 12
                    or len(held_out_variants) != 4
                    or any(not value for value in training_variants)
                    or any(not value for value in held_out_variants)
                ):
                    raise ValueError(
                        "Prepared prompt-family stages require exactly 12 "
                        "training variants and 4 held-out prompts."
                    )
                prompt_values.extend(
                    (
                        str(item["canonical_prompt"]).strip(),
                        *training_variants,
                        *held_out_variants,
                    )
                )
            if len(set(prompt_values)) != len(prompt_values):
                raise ValueError(
                    "Prepared prompt-family prompts must be globally unique."
                )
            if (
                int(profile["canonical_prompt_percent"]) != 80
                or int(profile["training_variant_prompt_percent"]) != 20
            ):
                raise ValueError(
                    "Prepared prompt-family profile requires an exact 80/20 split."
                )
    elif schema == PREPARED_STAGE1_SCENARIO_TRAINING_PROFILE_SCHEMA_V1:
        seen_source_ids: set[str] = set()
        seen_scenario_classes: set[str] = set()
        raw_episode_count = 0
        raw_frame_count = 0
        eligible_episode_count = 0
        openpi_window_count = 0
        for item in sources:
            source_id = str(item.get("source_dataset_id") or "").strip()
            scenario_class = str(item.get("scenario_class") or "").strip()
            task_scope = str(item.get("task_scope") or "").strip()
            expected_task_id = str(item.get("expected_task_id") or "").strip()
            repeat_weight = int(item.get("repeat_weight", 0))
            raw_episodes = int(item.get("expected_raw_episode_count", 0))
            raw_frames = int(item.get("expected_raw_frame_count", 0))
            eligible_episodes = int(
                item.get("expected_eligible_episode_count", 0)
            )
            openpi_windows = int(
                item.get("expected_openpi_quality_valid_window_count", 0)
            )
            if (
                not source_id
                or source_id in seen_source_ids
                or scenario_class not in _STAGE1_SCENARIO_REPEAT_WEIGHTS
                or not task_scope
                or not expected_task_id
                or repeat_weight
                != _STAGE1_SCENARIO_REPEAT_WEIGHTS.get(scenario_class)
                or raw_episodes <= 0
                or raw_frames <= 0
                or eligible_episodes <= 0
                or openpi_windows <= 0
            ):
                raise ValueError(
                    "Prepared Stage1 scenario source requires a unique id, "
                    "known scenario class, task identity, positive inventory, "
                    "and exact 1:4:6 repeat weight."
                )
            if scenario_class in {
                "initial_multi_choice",
                "midcourse_interrupt",
            } and task_scope != "stage1_only":
                raise ValueError(
                    f"Prepared {scenario_class} source must be stage1_only."
                )
            seen_source_ids.add(source_id)
            seen_scenario_classes.add(scenario_class)
            raw_episode_count += raw_episodes
            raw_frame_count += raw_frames
            eligible_episode_count += eligible_episodes
            openpi_window_count += openpi_windows
        if seen_scenario_classes != set(_STAGE1_SCENARIO_REPEAT_WEIGHTS):
            raise ValueError(
                "Prepared Stage1 scenario profile must include fixed, "
                "initial-choice, and midcourse-interrupt sources."
            )
        expected_totals = (
            int(profile["expected_total_raw_episode_count"]),
            int(profile["expected_total_raw_frame_count"]),
            int(profile["expected_total_eligible_episode_count"]),
            int(profile["expected_openpi_quality_valid_window_count"]),
        )
        actual_totals = (
            raw_episode_count,
            raw_frame_count,
            eligible_episode_count,
            openpi_window_count,
        )
        if actual_totals != expected_totals:
            raise ValueError(
                "Prepared Stage1 scenario source inventories differ from "
                "profile totals."
            )
        if str(profile["source_mixing_policy"]) != (
            "repeat_each_source_window_by_bound_weight_v1"
        ):
            raise ValueError(
                "Prepared Stage1 scenario profile requires exact per-source "
                "repetition."
            )
    if (
        profile.get("openpi_observation_mode")
        == OPENPI_STATE_ONLY_PROFILE_V1
        and not profile.get("openpi_norm_stats_assets_dir")
    ):
        raise ValueError(
            "State-only OpenPI profiles require "
            "openpi_norm_stats_assets_dir."
        )
    return profile


def prepared_training_profile_hash(profile: Mapping[str, Any]) -> str:
    return "sha256:" + canonical_content_hash(
        profile,
        domain=PREPARED_DEPLOYMENT_PROFILE_HASH_DOMAIN_V1,
    )


def _sha256_hash(value: object) -> str:
    normalized = str(value)
    return normalized if normalized.startswith("sha256:") else "sha256:" + normalized


def _validate_model_rtc_profile_binding(
    *,
    checkpoint_dir: Path,
    checkpoint_artifact: PolicyArtifactContract,
    prepared_profile: Mapping[str, Any],
    model_rtc_profile: Mapping[str, Any],
) -> OpenPIModelRtcCheckpointEvidence:
    if str(model_rtc_profile["prepared_skill_profile_id"]) != str(
        prepared_profile["profile_id"]
    ):
        raise ValueError(
            "Prepared model-RTC profile targets a different Prepared Skill profile."
        )
    if str(model_rtc_profile["output_config_name"]) != str(
        checkpoint_artifact.model_id
    ):
        raise ValueError(
            "Checkpoint model_id differs from the prepared model-RTC output config."
        )
    evidence = verify_openpi_model_rtc_checkpoint(checkpoint_dir)
    if evidence.artifact.to_dict() != checkpoint_artifact.to_dict():
        raise ValueError(
            "Verified trained model-RTC artifact differs from the checkpoint artifact."
        )
    continuation = require_trained_model_rtc_v1_contract(
        checkpoint_artifact.model_continuation
    )
    runtime = continuation.runtime
    expected_runtime_range = (
        int(model_rtc_profile["minimum_forecast_delay_steps"]),
        int(model_rtc_profile["maximum_forecast_delay_steps"]),
        int(model_rtc_profile["minimum_prefix_steps"]),
        int(model_rtc_profile["maximum_prefix_steps"]),
    )
    actual_runtime_range = (
        runtime.minimum_forecast_delay_steps,
        runtime.maximum_forecast_delay_steps,
        runtime.minimum_prefix_steps,
        runtime.maximum_prefix_steps,
    )
    if actual_runtime_range != expected_runtime_range:
        raise ValueError(
            "Prepared model-RTC profile delay/prefix range differs from the "
            "checkpoint continuation contract."
        )
    sampling = ModelContinuationTrainingSamplingSpec.from_dict(
        model_rtc_profile["sampling_spec"]
    )
    if sampling.semantic_content_hash != evidence.training_run.sampling_spec_hash:
        raise ValueError(
            "Prepared model-RTC sampling profile differs from the training run."
        )
    return evidence


def _sha256_file_bytes(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _prepared_model_rtc_binding_inventory(
    *,
    checkpoint_artifact: PolicyArtifactContract,
    prepared_profile: Mapping[str, Any],
    model_rtc_profile: Mapping[str, Any],
    evidence: OpenPIModelRtcCheckpointEvidence,
) -> tuple[Path, Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    source_bindings = evidence.training_run.source_bindings
    if len(source_bindings) != 1:
        raise ValueError(
            "Prepared model-RTC training run must bind one aggregate source."
        )
    source_binding = source_bindings[0]
    if source_binding.source_id != str(model_rtc_profile["profile_id"]):
        raise ValueError(
            "Prepared model-RTC training source differs from the model profile."
        )
    expected_camera_keys = tuple(
        str(value)
        for value in prepared_profile["camera_slot_to_dataset_key"].values()
    )
    if source_binding.camera_keys != expected_camera_keys:
        raise ValueError(
            "Prepared model-RTC training cameras differ from the prepared profile."
        )
    binding_hash = str(source_binding.dataset_identity_hash)
    if binding_hash.startswith("sha256:"):
        binding_hash = binding_hash.removeprefix("sha256:")
    if len(binding_hash) != 64 or any(
        character not in "0123456789abcdef" for character in binding_hash.lower()
    ):
        raise ValueError("Prepared model-RTC binding identity is not a SHA-256 hash.")
    workspace = Path(
        str(prepared_profile["organized_dataset_root"])
    ).expanduser().resolve()
    binding_root = (
        workspace
        / "derived"
        / "openpi_prepared_skill_bindings"
        / binding_hash
    )
    manifest_path = binding_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Prepared model-RTC binding manifest is missing: {manifest_path}"
        )
    if _sha256_file_bytes(manifest_path) != source_binding.episode_manifest_hash:
        raise ValueError(
            "Prepared binding manifest bytes differ from the training run."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("Prepared binding manifest must contain an object.")
    binding = manifest.get("binding")
    records = manifest.get("source_manifests")
    if not isinstance(binding, Mapping) or not isinstance(records, Sequence):
        raise ValueError("Prepared binding manifest structure is invalid.")
    if any(not isinstance(record, Mapping) for record in records):
        raise ValueError("Prepared source manifest inventory is invalid.")
    expected_camera_mapping = [
        {"model_slot": str(slot), "dataset_camera_key": str(key)}
        for slot, key in prepared_profile["camera_slot_to_dataset_key"].items()
    ]
    expected_binding = {
        "binding_hash": binding_hash,
        "skill_id": str(prepared_profile["target_skill_id"]),
        "camera_slot_to_dataset_key": expected_camera_mapping,
        "action_horizon": int(prepared_profile["action_horizon"]),
        "action_pose_mode": str(prepared_profile["action_pose_mode"]),
        "source_mixing_policy": str(
            prepared_profile.get("source_mixing_policy") or ""
        ),
    }
    actual_binding = {
        key: binding.get(key) for key in expected_binding
    }
    if actual_binding != expected_binding:
        raise ValueError(
            "Prepared binding manifest differs from the prepared profile."
        )
    expected_episode_count = prepared_profile.get(
        "expected_total_eligible_episode_count"
    )
    if (
        expected_episode_count is not None
        and int(binding.get("episode_count", -1)) != int(expected_episode_count)
    ):
        raise ValueError(
            "Prepared binding episode count differs from the prepared profile."
        )
    profile_sources = tuple(prepared_profile["sources"])
    records_by_id = {
        str(record.get("source_dataset_id") or ""): record
        for record in records
    }
    expected_source_ids = {
        str(source["source_dataset_id"]) for source in profile_sources
    }
    if set(records_by_id) != expected_source_ids:
        raise ValueError(
            "Prepared source manifest inventory differs from the prepared profile."
        )
    binding_episodes = binding.get("episodes")
    if not isinstance(binding_episodes, Sequence) or any(
        not isinstance(episode, Mapping) for episode in binding_episodes
    ):
        raise ValueError("Prepared binding episode inventory is invalid.")
    episodes_by_source: dict[str, list[Mapping[str, Any]]] = {}
    for episode in binding_episodes:
        episodes_by_source.setdefault(
            str(episode.get("source_dataset_id") or ""),
            [],
        ).append(episode)
    camera_slots = [
        str(value)
        for value in prepared_profile["camera_slot_to_dataset_key"].keys()
    ]
    training_view_hash = str(binding.get("training_view_hash") or "")
    normalized_records: list[Mapping[str, Any]] = []
    for source in profile_sources:
        source_id = str(source["source_dataset_id"])
        source_root = Path(str(source["dataset_root"])).expanduser().resolve()
        record = records_by_id[source_id]
        record_root = Path(
            str(record.get("source_dataset_root") or "")
        ).expanduser().resolve()
        if record_root != source_root:
            raise ValueError(
                "Prepared source root differs from the prepared profile."
            )
        episode_manifest = (
            binding_root / str(record.get("episode_manifest") or "")
        ).resolve()
        if not episode_manifest.is_relative_to(binding_root) or not episode_manifest.is_file():
            raise ValueError("Prepared source episode manifest path is invalid.")
        expected_episodes = sorted(
            episodes_by_source.get(source_id, ()),
            key=lambda value: int(value["source_episode_index"]),
        )
        actual_lines = [
            json.loads(line)
            for line in episode_manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(actual_lines) != len(expected_episodes):
            raise ValueError(
                "Prepared source episode manifest count differs from its binding."
            )
        for actual, expected in zip(actual_lines, expected_episodes, strict=True):
            expected_fields = {
                "source_episode_index": int(expected["source_episode_index"]),
                "episode_identity": str(
                    expected["episode_capture_identity_hash"]
                ),
                "camera_slots": camera_slots,
                "view_profile": "prepared_skill",
                "quality_class": "strict_success_non_abnormal",
                "prepared_training_view_hash": training_view_hash,
                "openpi_prepared_skill_binding_hash": binding_hash,
            }
            if not isinstance(actual, Mapping) or any(
                actual.get(key) != value for key, value in expected_fields.items()
            ):
                raise ValueError(
                    "Prepared source episode manifest differs from its binding."
                )
            actual_root = Path(
                str(actual.get("source_dataset_root") or "")
            ).expanduser().resolve()
            if actual_root != source_root:
                raise ValueError(
                    "Prepared source episode root differs from its binding."
                )
        normalized_records.append(
            {
                "source_dataset_id": source_id,
                "source_dataset_root": str(source_root),
                "episode_manifest": str(episode_manifest),
            }
        )
    artifact_sources = _canonical_source_roots(checkpoint_artifact.source_datasets)
    expected_artifact_sources = _canonical_source_roots(
        tuple(str(source["dataset_root"]) for source in profile_sources)
    )
    if artifact_sources != expected_artifact_sources:
        raise ValueError(
            "Prepared model-RTC artifact sources differ from the prepared profile."
        )
    return binding_root, binding, tuple(normalized_records)


def checkpoint_artifact_payload_hash(payload: Mapping[str, Any]) -> str:
    return "sha256:" + canonical_content_hash(
        payload,
        domain=PREPARED_DEPLOYMENT_CHECKPOINT_ARTIFACT_HASH_DOMAIN_V1,
    )


def _binding_hash(payload: Mapping[str, Any]) -> str:
    values = dict(payload)
    values.pop("binding_hash", None)
    return "sha256:" + canonical_content_hash(
        values,
        domain=PREPARED_DEPLOYMENT_BINDING_HASH_DOMAIN_V1,
    )


def _safe_filename(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "policy"


def _canonical_source_roots(values: Sequence[object]) -> tuple[str, ...]:
    """Normalize dataset roots without treating serialization order as identity.

    OpenPI's dataset adapter may serialize the same source collection in an
    order different from the profile. The roots still form an exact multiset:
    sorting makes order irrelevant while retaining multiplicity so a missing,
    substituted, or duplicated source cannot pass silently.
    """

    roots: list[str] = []
    for value in values:
        root = str(value).strip()
        if not root:
            raise ValueError("Source dataset roots require non-empty paths.")
        roots.append(str(Path(root).expanduser().resolve()))
    return tuple(sorted(roots))


def _canonical_source_pairs(
    sources: Sequence[object],
    *,
    source_id_key: str,
    root_key: str,
    expected_source_ids: Sequence[str] | None = None,
) -> tuple[tuple[str, str], ...]:
    """Return source bindings in their canonical, order-independent form.

    A prepared binding serializes source records in the order produced by the
    dataset adapter, while a profile may keep the order in which sources were
    selected.  The order is not part of the source identity contract; the
    source id and resolved root are.  Canonicalizing here accepts that
    serialization difference without accepting missing, duplicate, or
    substituted sources.
    """

    pairs: list[tuple[str, str]] = []
    for item in sources:
        if not isinstance(item, Mapping):
            raise ValueError("Source binding entries must be mappings.")
        source_id = str(item.get(source_id_key) or "").strip()
        dataset_root = str(item.get(root_key) or "").strip()
        if not source_id or not dataset_root:
            raise ValueError("Source bindings require an id and dataset root.")
        pairs.append(
            (
                source_id,
                str(Path(dataset_root).expanduser().resolve()),
            )
        )
    source_ids = tuple(source_id for source_id, _root in pairs)
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Source bindings must not contain duplicate source ids.")
    if expected_source_ids is not None:
        expected = tuple(str(value).strip() for value in expected_source_ids)
        if len(expected) != len(set(expected)) or set(source_ids) != set(expected):
            raise ValueError("Source bindings do not match the expected sources.")
    return tuple(sorted(pairs))


def _canonical_repeat_weights(
    weights: Sequence[object],
    *,
    source_id_key: str,
    weight_key: str,
    expected_source_ids: Sequence[str],
) -> tuple[tuple[str, int], ...]:
    """Normalize repeat-weight records while retaining exact source coverage."""

    normalized: list[tuple[str, int]] = []
    for item in weights:
        if not isinstance(item, Mapping):
            raise ValueError("Repeat-weight entries must be mappings.")
        source_id = str(item.get(source_id_key) or "").strip()
        if not source_id:
            raise ValueError("Repeat-weight entries require a source id.")
        try:
            repeat_weight = int(item.get(weight_key, 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("Repeat weights must be integers.") from exc
        if repeat_weight <= 0:
            raise ValueError("Repeat weights must be positive.")
        normalized.append((source_id, repeat_weight))
    source_ids = tuple(source_id for source_id, _weight in normalized)
    expected = tuple(str(value).strip() for value in expected_source_ids)
    if (
        len(source_ids) != len(set(source_ids))
        or len(expected) != len(set(expected))
        or set(source_ids) != set(expected)
    ):
        raise ValueError("Repeat weights must bind every source exactly once.")
    return tuple(sorted(normalized))


def _canonical_source_weight_map(
    value: object,
    *,
    expected_source_ids: Sequence[str],
) -> dict[str, int]:
    """Normalize a source-id -> repeat-weight mapping with exact coverage."""

    if not isinstance(value, Mapping):
        raise ValueError("Source repeat weights must be a mapping.")
    normalized: dict[str, int] = {}
    for source_id, weight in value.items():
        normalized_id = str(source_id).strip()
        if not normalized_id:
            raise ValueError("Source repeat weights require non-empty ids.")
        try:
            normalized_weight = int(weight)
        except (TypeError, ValueError) as exc:
            raise ValueError("Source repeat weights must be integers.") from exc
        if normalized_weight <= 0:
            raise ValueError("Source repeat weights must be positive.")
        if normalized_id in normalized:
            raise ValueError("Source repeat weights contain duplicate ids.")
        normalized[normalized_id] = normalized_weight
    expected = tuple(str(source_id).strip() for source_id in expected_source_ids)
    if set(normalized) != set(expected) or len(normalized) != len(expected):
        raise ValueError("Source repeat weights must bind every source exactly once.")
    return {source_id: normalized[source_id] for source_id in sorted(normalized)}


def _canonical_source_order(
    value: object,
    *,
    expected_source_ids: Sequence[str],
) -> tuple[str, ...]:
    """Validate and canonicalize a serialized source order."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("Source order must be a sequence.")
    source_order = tuple(str(item).strip() for item in value)
    expected = tuple(str(source_id).strip() for source_id in expected_source_ids)
    if (
        len(source_order) != len(set(source_order))
        or set(source_order) != set(expected)
        or any(not source_id for source_id in source_order)
    ):
        raise ValueError("Source order must bind every source exactly once.")
    return tuple(sorted(source_order))


def _canonical_source_profile_ids(profile: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(source_id for source_id, _root in _source_profile_pairs(profile))


def _source_profile_pairs(profile: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    return _canonical_source_pairs(
        profile["sources"],
        source_id_key="source_dataset_id",
        root_key="dataset_root",
    )


def _validate_image_wire_training_metadata(
    payload: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> None:
    transform = OpenPIImageWireTrainingTransform.from_profile_value(
        profile.get("openpi_image_wire_training"),
        image_keys=tuple(
            str(value) for value in profile["camera_slot_to_dataset_key"].keys()
        ),
    )
    expected = None if transform is None else transform.contract
    actual = payload.get(OPENPI_IMAGE_WIRE_TRAINING_METADATA_KEY_V1)
    if expected is None and actual is None:
        return
    if not isinstance(actual, Mapping) or dict(actual) != expected:
        raise ValueError(
            "Checkpoint image-wire training alignment differs from the "
            "prepared training profile."
        )


def _validate_state_only_training_metadata(
    payload: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> None:
    bridge = OpenPIStateOnlyInputBridge.from_profile_value(
        profile.get("openpi_observation_mode"),
        image_keys=tuple(
            str(value) for value in profile["camera_slot_to_dataset_key"].keys()
        ),
    )
    expected = None if bridge is None else bridge.contract
    actual = payload.get(OPENPI_STATE_ONLY_CAPABILITY_KEY_V1)
    if expected is None and actual is None:
        return
    if not isinstance(actual, Mapping) or dict(actual) != expected:
        raise ValueError(
            "Checkpoint state-only input bridge differs from the prepared profile."
        )


def _validate_checkpoint_training_metadata(
    payload: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> None:
    stage1_scenario_profile = (
        profile.get("schema")
        == PREPARED_STAGE1_SCENARIO_TRAINING_PROFILE_SCHEMA_V1
    )
    prompt_family_profile = (
        profile.get("schema")
        == PREPARED_PROMPT_FAMILY_TRAINING_PROFILE_SCHEMA_V2
    )
    prompt_profile = profile.get("schema") in {
        PREPARED_PROMPT_TRAINING_PROFILE_SCHEMA_V1,
        PREPARED_PROMPT_FAMILY_TRAINING_PROFILE_SCHEMA_V2,
    }
    metadata_key = (
        "prepared_prompt_training_binding"
        if prompt_profile
        else "prepared_skill_training_binding"
    )
    prepared = payload.get(metadata_key)
    if not isinstance(prepared, Mapping):
        raise ValueError(
            f"Checkpoint artifact lacks {metadata_key}."
        )
    expected_camera_mapping = {
        str(key): str(value)
        for key, value in profile["camera_slot_to_dataset_key"].items()
    }
    actual_camera_mapping = prepared.get("camera_slot_to_dataset_key")
    if isinstance(actual_camera_mapping, Mapping):
        normalized_actual_mapping = {
            str(key): str(value) for key, value in actual_camera_mapping.items()
        }
    elif isinstance(actual_camera_mapping, Sequence) and not isinstance(
        actual_camera_mapping,
        (str, bytes),
    ):
        if any(not isinstance(item, Mapping) for item in actual_camera_mapping):
            raise ValueError(
                "Checkpoint prepared camera mapping entries must be mappings."
            )
        normalized_actual_mapping = {
            str(item.get("model_slot") or ""): str(
                item.get("dataset_camera_key") or ""
            )
            for item in actual_camera_mapping
        }
    else:
        raise ValueError(
            "Checkpoint prepared binding lacks camera_slot_to_dataset_key."
        )
    expected_prepared: dict[str, Any] = {
        "action_horizon": int(profile["action_horizon"]),
        "action_pose_mode": str(profile["action_pose_mode"]),
        "camera_slot_to_dataset_key": expected_camera_mapping,
    }
    actual_prepared: dict[str, Any] = {
        "action_horizon": int(prepared.get("action_horizon", 0)),
        "action_pose_mode": str(prepared.get("action_pose_mode") or ""),
        "camera_slot_to_dataset_key": normalized_actual_mapping,
    }
    if prompt_profile:
        prompt_skills = prepared.get("prompt_skills")
        if not isinstance(prompt_skills, Sequence) or isinstance(
            prompt_skills, (str, bytes)
        ):
            raise ValueError(
                "Checkpoint prepared prompt binding lacks prompt_skills."
            )
        actual_prompt_stages = tuple(
            (
                str(item.get("stage_id") or ""),
                str(item.get("skill_id") or ""),
                str(item.get("prompt") or ""),
                tuple(
                    str(value)
                    for value in item.get("training_prompt_variants", ())
                ),
                tuple(
                    str(value)
                    for value in item.get("held_out_prompt_variants", ())
                ),
                int(item.get("window_count", 0)),
                int(item.get("expected_openpi_window_count", 0)),
            )
            for item in prompt_skills
            if isinstance(item, Mapping)
        )
        expected_prompt_stages = tuple(
            (
                str(item["stage_id"]),
                str(item["skill_id"]),
                str(
                    item[
                        "canonical_prompt" if prompt_family_profile else "prompt"
                    ]
                ),
                tuple(
                    str(value)
                    for value in item.get("training_prompt_variants", ())
                ),
                tuple(
                    str(value)
                    for value in item.get("held_out_prompt_variants", ())
                ),
                int(item["expected_window_count"]),
                int(item["expected_openpi_window_count"]),
            )
            for item in profile["prompt_stages"]
        )
        expected_prepared.update(
            prompt_stages=expected_prompt_stages,
            episode_count=int(profile["expected_union_episode_count"]),
            window_count=int(profile["expected_total_window_count"]),
            prompt_stage_sampling_policy=str(
                profile["prompt_stage_sampling_policy"]
            ),
            prompt_stage_sampling_seed=int(
                profile["prompt_stage_sampling_seed"]
            ),
        )
        actual_prepared.update(
            prompt_stages=actual_prompt_stages,
            episode_count=int(prepared.get("episode_count", 0)),
            window_count=int(prepared.get("window_count", 0)),
            prompt_stage_sampling_policy=str(
                prepared.get("prompt_stage_sampling_policy") or ""
            ),
            prompt_stage_sampling_seed=int(
                prepared.get("prompt_stage_sampling_seed", 0)
            ),
        )
        if prompt_family_profile:
            expected_prepared.update(
                schema="dexteleop.openpi_prepared_prompt_binding.v2",
                prompt_family_sampling_policy=str(
                    profile["prompt_family_sampling_policy"]
                ),
                prompt_family_sampling_seed=int(
                    profile["prompt_family_sampling_seed"]
                ),
                canonical_prompt_percent=80,
                training_variant_prompt_percent=20,
                held_out_prompts_in_training=False,
            )
            actual_prepared.update(
                schema=str(prepared.get("schema") or ""),
                prompt_family_sampling_policy=str(
                    prepared.get("prompt_family_sampling_policy") or ""
                ),
                prompt_family_sampling_seed=int(
                    prepared.get("prompt_family_sampling_seed", 0)
                ),
                canonical_prompt_percent=int(
                    prepared.get("canonical_prompt_percent", 0)
                ),
                training_variant_prompt_percent=int(
                    prepared.get("training_variant_prompt_percent", 0)
                ),
                held_out_prompts_in_training=bool(
                    prepared.get("held_out_prompts_in_training", True)
                ),
            )
    elif stage1_scenario_profile:
        raw_repeat_weights = prepared.get("source_repeat_weights")
        if not isinstance(raw_repeat_weights, Sequence) or isinstance(
            raw_repeat_weights,
            (str, bytes),
        ):
            raise ValueError(
                "Checkpoint prepared Stage1 scenario binding lacks "
                "source_repeat_weights."
            )
        expected_source_ids = _canonical_source_profile_ids(profile)
        actual_repeat_weights = _canonical_repeat_weights(
            raw_repeat_weights,
            source_id_key="source_dataset_id",
            weight_key="repeat_weight",
            expected_source_ids=expected_source_ids,
        )
        expected_repeat_weights = _canonical_repeat_weights(
            profile["sources"],
            source_id_key="source_dataset_id",
            weight_key="repeat_weight",
            expected_source_ids=expected_source_ids,
        )
        expected_prepared.update(
            schema="dexteleop.openpi_prepared_skill_binding.v2",
            skill_id=str(profile["target_skill_id"]),
            selection_policy_id=STAGE1_SCENARIO_SELECTION_POLICY_V1,
            source_mixing_policy=str(profile["source_mixing_policy"]),
            source_repeat_weights=expected_repeat_weights,
            episode_count=int(profile["expected_total_eligible_episode_count"]),
            window_count=int(
                profile["expected_prepared_interval_window_count"]
            ),
            prepared_interval_repeated_window_count=int(
                profile[
                    "expected_prepared_interval_repeated_window_count"
                ]
            ),
        )
        actual_prepared.update(
            schema=str(prepared.get("schema") or ""),
            skill_id=str(prepared.get("skill_id") or ""),
            selection_policy_id=str(
                prepared.get("selection_policy_id") or ""
            ),
            source_mixing_policy=str(
                prepared.get("source_mixing_policy") or ""
            ),
            source_repeat_weights=actual_repeat_weights,
            episode_count=int(prepared.get("episode_count", 0)),
            window_count=int(prepared.get("window_count", 0)),
            prepared_interval_repeated_window_count=int(
                prepared.get(
                    "prepared_interval_repeated_window_count",
                    0,
                )
            ),
        )
    else:
        expected_prepared["skill_id"] = str(profile["target_skill_id"])
        actual_prepared["skill_id"] = str(prepared.get("skill_id") or "")
    if actual_prepared != expected_prepared:
        raise ValueError(
            "Checkpoint prepared binding disagrees with the training profile."
        )

    mixing = payload.get("dataset_mixing")
    if not isinstance(mixing, Mapping):
        raise ValueError("Checkpoint artifact lacks dataset_mixing metadata.")
    sources = mixing.get("sources")
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
        raise ValueError("Checkpoint dataset_mixing.sources must be a sequence.")
    expected_source_ids = _canonical_source_profile_ids(profile)
    actual_source_pairs = _canonical_source_pairs(
        sources,
        source_id_key="name",
        root_key="dataset_root",
        expected_source_ids=expected_source_ids,
    )
    if actual_source_pairs != _source_profile_pairs(profile):
        raise ValueError(
            "Checkpoint dataset mixing sources disagree with the training profile."
        )
    if stage1_scenario_profile:
        expected_source_order = tuple(sorted(expected_source_ids))
        expected_repeat_weight_pairs = _canonical_repeat_weights(
            profile["sources"],
            source_id_key="source_dataset_id",
            weight_key="repeat_weight",
            expected_source_ids=expected_source_ids,
        )
        expected_repeat_weight_map = dict(expected_repeat_weight_pairs)
        expected_source_weights = expected_repeat_weight_pairs
        actual_repeat_weight_map = _canonical_source_weight_map(
            mixing.get("source_repeat_weights"),
            expected_source_ids=expected_source_ids,
        )
        actual_source_weights = _canonical_repeat_weights(
            sources,
            source_id_key="name",
            weight_key="sample_weight",
            expected_source_ids=expected_source_ids,
        )
        actual_scenario_mixing = {
            "schema": str(mixing.get("schema") or ""),
            "source_mixing_policy": str(
                mixing.get("source_mixing_policy") or ""
            ),
            "source_order": _canonical_source_order(
                mixing.get("source_order", ()),
                expected_source_ids=expected_source_ids,
            ),
            "each_quality_valid_window_once_per_epoch": bool(
                mixing.get("each_quality_valid_window_once_per_epoch", True)
            ),
            "repeat_each_quality_valid_window_by_source_weight": bool(
                mixing.get(
                    "repeat_each_quality_valid_window_by_source_weight",
                    False,
                )
            ),
            "source_repeat_weights": actual_repeat_weight_map,
            "source_weights": actual_source_weights,
        }
        expected_scenario_mixing = {
            "schema": "dexteleop_v23_bound_source_repeat_v1",
            "source_mixing_policy": str(profile["source_mixing_policy"]),
            "source_order": expected_source_order,
            "each_quality_valid_window_once_per_epoch": False,
            "repeat_each_quality_valid_window_by_source_weight": True,
            "source_repeat_weights": expected_repeat_weight_map,
            "source_weights": expected_source_weights,
        }
        if actual_scenario_mixing != expected_scenario_mixing:
            raise ValueError(
                "Checkpoint Stage1 scenario dataset mixing disagrees with the "
                "training profile."
            )


def _validate_artifact_contract(
    artifact: PolicyArtifactContract,
    profile: Mapping[str, Any],
    *,
    expected_model_id: str | None = None,
) -> None:
    resolved_model_id = str(expected_model_id or profile["openpi_config_name"])
    if artifact.model_id != resolved_model_id:
        raise ValueError(
            "Checkpoint model_id differs from the prepared OpenPI config name."
        )
    if artifact.layout_hash != str(profile["expected_policy_io_layout_hash"]):
        raise ValueError(
            "Checkpoint policy-I/O layout differs from the prepared profile."
        )
    pose_chunk = artifact.pose_action_chunk
    if pose_chunk is None:
        raise ValueError("Prepared checkpoint lacks a pose action chunk contract.")
    if (
        pose_chunk.action_horizon != int(profile["action_horizon"])
        or pose_chunk.action_pose_mode != str(profile["action_pose_mode"])
        or pose_chunk.last_action_pose_mode != str(profile["action_pose_mode"])
        or pose_chunk.tail_policy != str(profile["tail_policy"])
    ):
        raise ValueError(
            "Checkpoint pose action chunk contract disagrees with the prepared profile."
        )
    state_only_bridge = OpenPIStateOnlyInputBridge.from_profile_value(
        profile.get("openpi_observation_mode"),
        image_keys=tuple(
            str(value) for value in profile["camera_slot_to_dataset_key"].keys()
        ),
    )
    expected_slots = (
        ()
        if state_only_bridge is not None
        else tuple(
            str(value)
            for value in profile["camera_slot_to_dataset_key"].keys()
        )
    )
    actual_slots = (
        ()
        if artifact.camera_input is None
        else artifact.camera_input.server_camera_slots
    )
    if actual_slots != expected_slots:
        raise ValueError(
            "Checkpoint camera slots disagree with the prepared profile."
        )


def _expected_deployment(
    checkpoint_dir: Path,
    profile_path: Path,
    model_rtc_profile_path: Path | None = None,
) -> PreparedDeploymentResolution:
    checkpoint_artifact_path = checkpoint_dir / "policy_artifact.json"
    checkpoint_payload = load_policy_artifact_payload(checkpoint_artifact_path)
    checkpoint_artifact = PolicyArtifactContract.from_dict(checkpoint_payload)
    profile = load_prepared_training_profile(profile_path)
    model_rtc_profile = (
        None
        if model_rtc_profile_path is None
        else load_prepared_model_rtc_profile(model_rtc_profile_path)
    )
    expected_model_id = (
        str(profile["openpi_config_name"])
        if model_rtc_profile is None
        else str(model_rtc_profile["output_config_name"])
    )
    _validate_artifact_contract(
        checkpoint_artifact,
        profile,
        expected_model_id=expected_model_id,
    )
    model_rtc_evidence = None
    prepared_binding_root = None
    prepared_binding_payload = None
    prepared_source_manifests: tuple[Mapping[str, Any], ...] = ()
    _validate_image_wire_training_metadata(checkpoint_payload, profile)
    _validate_state_only_training_metadata(checkpoint_payload, profile)
    if model_rtc_profile is None:
        _validate_checkpoint_training_metadata(checkpoint_payload, profile)
    else:
        model_rtc_evidence = _validate_model_rtc_profile_binding(
            checkpoint_dir=checkpoint_dir,
            checkpoint_artifact=checkpoint_artifact,
            prepared_profile=profile,
            model_rtc_profile=model_rtc_profile,
        )
        (
            prepared_binding_root,
            prepared_binding_payload,
            prepared_source_manifests,
        ) = _prepared_model_rtc_binding_inventory(
            checkpoint_artifact=checkpoint_artifact,
            prepared_profile=profile,
            model_rtc_profile=model_rtc_profile,
            evidence=model_rtc_evidence,
        )

    asset_id = str(profile["openpi_asset_id"])
    norm_stats_path = checkpoint_dir / "assets" / asset_id / "norm_stats.json"
    if not norm_stats_path.is_file():
        raise FileNotFoundError(
            "Prepared checkpoint norm_stats.json is missing: "
            f"{norm_stats_path}"
        )
    stats_hash = normalization_stats_hash(norm_stats_path.read_bytes())
    stats_ref = f"checkpoint://assets/{asset_id}/norm_stats.json"
    normalization_contract = make_openpi_quantile_normalization_contract(
        stats_ref=stats_ref,
        stats_hash=stats_hash,
    )
    binding: dict[str, Any] = {
        "schema": (
            PREPARED_DEPLOYMENT_BINDING_SCHEMA_V1
            if model_rtc_profile is None
            else PREPARED_MODEL_RTC_DEPLOYMENT_BINDING_SCHEMA_V1
        ),
        "checkpoint_policy_artifact_hash": checkpoint_artifact_payload_hash(
            checkpoint_payload
        ),
        "checkpoint_step": checkpoint_artifact.checkpoint_step,
        "profile_id": str(profile["profile_id"]),
        "profile_hash": prepared_training_profile_hash(profile),
        "openpi_base_config": str(profile["openpi_base_config"]),
        "openpi_state_augmentation_from_config": str(
            profile.get("openpi_state_augmentation_from_config") or ""
        ),
        "openpi_config_name": expected_model_id,
        "openpi_asset_id": asset_id,
        "openpi_norm_stats_assets_dir": str(
            profile.get("openpi_norm_stats_assets_dir") or ""
        ),
        "policy_io_layout_hash": checkpoint_artifact.layout_hash,
        "action_horizon": int(profile["action_horizon"]),
        "action_pose_mode": str(profile["action_pose_mode"]),
        "tail_policy": str(profile["tail_policy"]),
        "camera_slots": list(profile["camera_slot_to_dataset_key"].keys()),
        "norm_stats_ref": stats_ref,
        "norm_stats_hash": stats_hash,
    }
    if profile.get("schema") in {
        PREPARED_PROMPT_TRAINING_PROFILE_SCHEMA_V1,
        PREPARED_PROMPT_FAMILY_TRAINING_PROFILE_SCHEMA_V2,
    }:
        prompt_family_profile = (
            profile.get("schema")
            == PREPARED_PROMPT_FAMILY_TRAINING_PROFILE_SCHEMA_V2
        )
        prompt_stages = [
            {
                "stage_id": str(item["stage_id"]),
                "skill_id": str(item["skill_id"]),
                "prompt": str(
                    item[
                        "canonical_prompt" if prompt_family_profile else "prompt"
                    ]
                ),
                **(
                    {
                        "allowed_prompts": [
                            str(item["canonical_prompt"]),
                            *(
                                str(value)
                                for value in item["training_prompt_variants"]
                            ),
                        ]
                    }
                    if prompt_family_profile
                    else {}
                ),
            }
            for item in profile["prompt_stages"]
        ]
        binding["prompt_stages"] = prompt_stages
        binding["prompt_contract_hash"] = "sha256:" + canonical_content_hash(
            {"prompt_stages": prompt_stages},
            domain=PREPARED_PROMPT_CONTRACT_HASH_DOMAIN_V1,
        )
    if model_rtc_profile is not None:
        assert model_rtc_evidence is not None
        assert prepared_binding_root is not None
        assert prepared_binding_payload is not None
        continuation_data = checkpoint_artifact.model_continuation
        if not isinstance(continuation_data, ModelContinuationDataContractV2):
            raise ValueError(
                "Prepared model-RTC checkpoint lacks a v2 continuation contract."
            )
        require_trained_model_rtc_v1_contract(continuation_data)
        binding.update(
            {
                "prepared_openpi_config_name": str(
                    profile["openpi_config_name"]
                ),
                "model_rtc_profile_id": str(model_rtc_profile["profile_id"]),
                "model_rtc_profile_hash": prepared_model_rtc_profile_hash(
                    model_rtc_profile
                ),
                "model_continuation_contract_hash": (
                    continuation_data.contract_hash
                ),
                "training_run_manifest_hash": _sha256_hash(
                    model_rtc_evidence.training_run.semantic_content_hash
                ),
                "checkpoint_receipt_hash": _sha256_hash(
                    model_rtc_evidence.checkpoint_receipt.semantic_content_hash
                ),
                "prepared_skill_binding_hash": str(
                    prepared_binding_payload["binding_hash"]
                ),
                "prepared_skill_manifest_hash": _sha256_file_bytes(
                    prepared_binding_root / "manifest.json"
                ),
            }
        )
    binding["binding_hash"] = _binding_hash(binding)

    deployment_payload = copy.deepcopy(checkpoint_payload)
    metadata = deployment_payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Checkpoint artifact metadata must be a mapping.")
    if PREPARED_DEPLOYMENT_BINDING_METADATA_KEY in metadata:
        raise ValueError(
            "Checkpoint artifact already declares prepared deployment metadata."
        )
    if model_rtc_profile is None and "normalization_contract" in metadata:
        raise ValueError(
            "Checkpoint artifact already declares prepared deployment metadata."
        )
    if (
        model_rtc_profile is not None
        and metadata.get("normalization_contract") != normalization_contract
    ):
        raise ValueError(
            "Trained model-RTC normalization contract differs from the "
            "checkpoint norm-stats bytes."
        )
    deployment_payload["metadata"] = {
        **dict(metadata),
        **(
            {"normalization_contract": normalization_contract}
            if model_rtc_profile is None
            else {}
        ),
        PREPARED_DEPLOYMENT_BINDING_METADATA_KEY: binding,
    }
    deployment_artifact = PolicyArtifactContract.from_dict(deployment_payload)
    filename = (
        f"dexteleop_openpi_{_safe_filename(checkpoint_artifact.model_id)}_"
        f"{_safe_filename(checkpoint_dir.name)}.prepared.policy_artifact.json"
    )
    deployment_artifact_path = Path(tempfile.gettempdir()) / filename
    return PreparedDeploymentResolution(
        checkpoint_dir=checkpoint_dir,
        checkpoint_artifact_path=checkpoint_artifact_path,
        deployment_artifact_path=deployment_artifact_path,
        checkpoint_artifact=checkpoint_artifact,
        deployment_artifact=deployment_artifact,
        checkpoint_payload=checkpoint_payload,
        deployment_payload=deployment_payload,
        profile_path=profile_path,
        profile=profile,
        binding=binding,
        norm_stats_path=norm_stats_path,
        prepared_binding_root=prepared_binding_root,
        prepared_binding_payload=prepared_binding_payload,
        prepared_source_manifests=prepared_source_manifests,
        model_rtc_profile_path=model_rtc_profile_path,
        model_rtc_profile=model_rtc_profile,
    )


def prepare_checkpoint_deployment_artifact(
    checkpoint_dir: str | Path,
    profile_path: str | Path,
    *,
    model_rtc_profile_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> PreparedDeploymentResolution:
    """Validate and atomically materialize the disposable deployment sidecar."""

    checkpoint = Path(checkpoint_dir).expanduser().resolve()
    profile = Path(profile_path).expanduser().resolve()
    model_rtc_profile = (
        None
        if model_rtc_profile_path is None
        else Path(model_rtc_profile_path).expanduser().resolve()
    )
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory is missing: {checkpoint}")
    resolution = _expected_deployment(
        checkpoint,
        profile,
        model_rtc_profile,
    )
    target = resolution.deployment_artifact_path
    if output_dir is not None:
        target = Path(output_dir).expanduser().resolve() / target.name
    target.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(resolution.deployment_payload, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    handle, staging_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    staging = Path(staging_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, target)
    finally:
        if staging.exists():
            staging.unlink()
    return replace(resolution, deployment_artifact_path=target)


def validate_prepared_deployment_artifact(
    checkpoint_dir: str | Path,
    profile_path: str | Path,
    deployment_artifact_path: str | Path,
    *,
    model_rtc_profile_path: str | Path | None = None,
) -> PreparedDeploymentResolution:
    """Recompute every binding and require an exact disposable sidecar match."""

    checkpoint = Path(checkpoint_dir).expanduser().resolve()
    profile = Path(profile_path).expanduser().resolve()
    model_rtc_profile = (
        None
        if model_rtc_profile_path is None
        else Path(model_rtc_profile_path).expanduser().resolve()
    )
    expected = _expected_deployment(
        checkpoint,
        profile,
        model_rtc_profile,
    )
    actual_path = Path(deployment_artifact_path).expanduser().resolve()
    actual_payload = load_policy_artifact_payload(actual_path)
    if actual_payload != expected.deployment_payload:
        raise ValueError(
            "Prepared deployment artifact differs from the checkpoint, profile, "
            "or current norm-stats bytes."
        )
    actual_artifact = load_policy_artifact(actual_path)
    raw_binding = actual_artifact.metadata.get(
        PREPARED_DEPLOYMENT_BINDING_METADATA_KEY
    )
    if not isinstance(raw_binding, Mapping):
        raise ValueError("Prepared deployment artifact lacks its binding.")
    if str(raw_binding.get("binding_hash") or "") != _binding_hash(raw_binding):
        raise ValueError("Prepared deployment binding hash is invalid.")
    return replace(
        expected,
        deployment_artifact_path=actual_path,
        deployment_artifact=actual_artifact,
        deployment_payload=actual_payload,
    )


__all__ = [
    "PREPARED_DEPLOYMENT_BINDING_METADATA_KEY",
    "PREPARED_DEPLOYMENT_BINDING_SCHEMA_V1",
    "PREPARED_MODEL_RTC_DEPLOYMENT_BINDING_SCHEMA_V1",
    "PreparedDeploymentResolution",
    "checkpoint_artifact_payload_hash",
    "load_prepared_training_profile",
    "prepare_checkpoint_deployment_artifact",
    "prepared_training_profile_hash",
    "validate_prepared_deployment_artifact",
]
