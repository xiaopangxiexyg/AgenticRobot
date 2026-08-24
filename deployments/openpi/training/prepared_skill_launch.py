"""Prepare, audit, normalize, and train one immutable Prepared Skill view."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import copy
import dataclasses
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
from typing import Any, Iterator, Mapping

import numpy as np

from dataset_interface.prepared_training import (
    PreparedSourceDataset,
    build_prepared_training_view,
    select_strict_successful_skill_episodes,
)
from dataset_interface.progress_skill_projection import (
    SkillProjectionCatalog,
    load_builtin_writing_four_stage_subtasks_skill_projection_catalog,
    load_builtin_writing_stage1_skill_projection_catalog,
)
from deployments.openpi.image_wire import (
    OPENPI_IMAGE_WIRE_TRAINING_METADATA_KEY_V1,
    OpenPIImageWireTrainingTransform,
)
from deployments.openpi.state_only import (
    OPENPI_STATE_ONLY_CAPABILITY_KEY_V1,
    OPENPI_STATE_ONLY_PROFILE_V1,
    OpenPIStateOnlyInputBridge,
    bind_state_only_policy_metadata,
)

from .prepared_skill import (
    OPENPI_SOURCE_MIXING_REPEAT_BY_SOURCE_V1,
    OpenPIPreparedPromptBinding,
    OpenPIPreparedSkillBinding,
    OpenPIPreparedSkillError,
    binding_source_manifests,
    compose_openpi_prepared_prompt_binding,
    effective_prompt_stage_window_count,
    load_openpi_prepared_skill_binding_manifest,
    load_openpi_prepared_skill_binding,
    materialize_openpi_prepared_skill_binding,
    patch_openpi_prepared_skill_dataset,
)
from .stage1_scenario import (
    STAGE1_SCENARIO_SELECTION_POLICY_V1,
    prepare_stage1_scenario_selections,
)
from .prepared_model_rtc import (
    adapt_prepared_skill_model_rtc_config,
    load_prepared_model_rtc_profile,
)
from .evidence import (
    OpenPIModelRtcCheckpointEvidence,
    verify_openpi_model_rtc_checkpoint,
)
from .launch import MODEL_CONTINUATION_TRAINING_RUN_METADATA_KEY_V1


PROFILE_SCHEMA_V1 = "dexteleop.openpi_prepared_skill_training_profile.v1"
PROMPT_PROFILE_SCHEMA_V1 = (
    "dexteleop.openpi_prepared_prompt_training_profile.v1"
)
PROMPT_FAMILY_PROFILE_SCHEMA_V2 = (
    "dexteleop.openpi_prepared_prompt_family_training_profile.v2"
)
STAGE1_SCENARIO_PROFILE_SCHEMA_V1 = (
    "dexteleop.openpi_stage1_scenario_training_profile.v1"
)
STANDARD_NON_RTC_TRAINING_MODE_V1 = "standard_non_rtc_v1"
CORRECTED_WRIST_BACKGROUND_PROFILE_V1 = (
    "dexteleop.corrected_wrist_background.v1"
)
DEFAULT_PROFILE = (
    Path(__file__).resolve().parents[1]
    / "configs/training/writing_stage1_0804_0805_front_h16_absolute_v1.json"
)
SKILL_PROJECTION_CATALOG_WRITING_STAGE1_V2 = "writing_stage1_v2"
SKILL_PROJECTION_CATALOG_WRITING_FOUR_STAGE_SUBTASKS_V2 = (
    "writing_four_stage_subtasks_v2"
)
_EMPTY_CHECKPOINT_ROOT_ENTRIES = frozenset({"wandb", "wandb_id.txt"})
OPENPI_SOURCE_IDENTITY_FILENAME = "OPENPI_SOURCE_IDENTITY.json"


def _corrected_wrist_background_overrides(
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    raw = profile.get("openpi_wrist_background_augmentation")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError(
            "openpi_wrist_background_augmentation must be an object."
        )
    expected = {
        "schema": CORRECTED_WRIST_BACKGROUND_PROFILE_V1,
        "provider": "mixed",
        "original_probability": 0.1,
        "real_image_probability": 0.4,
        "procedural_probability": 0.5,
        "black_probability": 0.0,
        "wrist_pair_mode": "shared_scene_family",
        "mask_source": "wrist_final_keep_mask_paths",
        "wrist_depth_mask_paths_fallback_allowed": False,
        "strict_media_integrity": True,
    }
    mismatched = {
        key: (raw.get(key), value)
        for key, value in expected.items()
        if raw.get(key) != value
    }
    if mismatched:
        raise ValueError(
            "OpenPI wrist background profile differs from the corrected "
            f"10/40/50 final-mask contract: {mismatched}."
        )
    real_image_root = Path(str(raw.get("real_image_root") or "")).expanduser()
    real_image_manifest = Path(
        str(raw.get("real_image_manifest") or "")
    ).expanduser()
    if not real_image_root.is_absolute() or not real_image_manifest.is_absolute():
        raise ValueError(
            "Corrected wrist background real-image paths must be absolute."
        )
    return {
        "background_provider": "mixed",
        "mixed_original_prob": 0.1,
        "real_image_prob": 0.4,
        "mixed_black_prob": 0.0,
        "wrist_pair_mode": "shared_scene_family",
        "real_image_root": str(real_image_root),
        "real_image_manifest": str(real_image_manifest),
        "strict_media_integrity": True,
    }


def _activate_openpi_checkout(profile: Mapping[str, Any]) -> Any:
    checkout = Path(str(profile["openpi_checkout"])).expanduser().resolve()
    source_root = checkout / "src"
    if not (source_root / "openpi").is_dir():
        raise FileNotFoundError(f"OpenPI checkout is missing: {checkout}.")

    expected_revision = str(profile.get("openpi_source_revision") or "").strip()
    if expected_revision:
        identity_path = checkout / OPENPI_SOURCE_IDENTITY_FILENAME
        if not identity_path.is_file():
            raise FileNotFoundError(
                f"Pinned OpenPI source identity is missing: {identity_path}."
            )
        identity_bytes = identity_path.read_bytes()
        expected_identity_sha256 = str(
            profile.get("openpi_source_identity_sha256") or ""
        ).strip()
        actual_identity_sha256 = hashlib.sha256(identity_bytes).hexdigest()
        if actual_identity_sha256 != expected_identity_sha256:
            raise ValueError(
                "OpenPI source identity file hash differs from the training "
                f"profile: {identity_path}."
            )
        identity = json.loads(identity_bytes)
        if (
            not isinstance(identity, Mapping)
            or identity.get("schema") != "dexteleop.openpi_source_identity.v1"
            or str(identity.get("revision") or "") != expected_revision
        ):
            raise ValueError(
                "OpenPI source identity differs from the training profile: "
                f"{identity_path}."
            )
        declared_config_sha256 = str(
            identity.get("training_config_sha256") or ""
        )
        actual_config_sha256 = hashlib.sha256(
            (source_root / "openpi/training/config.py").read_bytes()
        ).hexdigest()
        if declared_config_sha256 != actual_config_sha256:
            raise ValueError(
                "OpenPI training config content differs from its pinned "
                f"source identity: {checkout}."
            )

    for name, module in tuple(sys.modules.items()):
        if name != "openpi" and not name.startswith("openpi."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        if not Path(str(module_file)).resolve().is_relative_to(checkout):
            raise RuntimeError(
                "OpenPI was already imported from outside the profile checkout: "
                f"{name}={module_file}. Start a fresh training process."
            )

    source_text = str(source_root)
    sys.path[:] = [
        source_text,
        *(
            item
            for item in sys.path
            if str(Path(item or ".").resolve()) != source_text
        ),
    ]
    training_config = importlib.import_module("openpi.training.config")
    module_file = Path(str(training_config.__file__)).resolve()
    if not module_file.is_relative_to(checkout):
        raise RuntimeError(
            "OpenPI training config was imported from outside the profile "
            f"checkout: {module_file}."
        )
    return training_config


@dataclasses.dataclass(frozen=True)
class _TrainCheckpointPolicy:
    checkpoint_dir: Path
    mode: str
    resume: bool
    overwrite: bool
    latest_checkpoint_step: int | None


@dataclasses.dataclass(frozen=True)
class _CheckpointBoundPreparedResume:
    binding: OpenPIPreparedSkillBinding
    binding_root: Path
    checkpoint_policy: _TrainCheckpointPolicy
    evidence: OpenPIModelRtcCheckpointEvidence


@dataclasses.dataclass(frozen=True)
class _PromptStageReport:
    stage_id: str
    skill_id: str
    prompt: str
    expected_window_count: int
    report: Any


def _numbered_checkpoint_steps(checkpoint_dir: Path) -> tuple[int, ...]:
    if not checkpoint_dir.is_dir():
        return ()
    steps = []
    for child in checkpoint_dir.iterdir():
        if not child.is_dir() or not child.name.isdecimal():
            continue
        step = int(child.name)
        if child.name != str(step):
            continue
        steps.append(step)
    return tuple(sorted(steps))


def _resolve_train_checkpoint_policy(
    checkpoint_dir: str | Path,
    *,
    resume: bool,
    overwrite: bool,
) -> _TrainCheckpointPolicy:
    root = Path(str(checkpoint_dir)).expanduser().resolve()
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    if overwrite:
        return _TrainCheckpointPolicy(
            root,
            "overwrite",
            resume=False,
            overwrite=True,
            latest_checkpoint_step=None,
        )
    if not root.exists():
        if resume:
            raise FileNotFoundError(
                f"Cannot resume: checkpoint directory does not exist: {root}."
            )
        return _TrainCheckpointPolicy(
            root,
            "fresh",
            resume=False,
            overwrite=False,
            latest_checkpoint_step=None,
        )
    if not root.is_dir():
        raise FileExistsError(
            f"Checkpoint path exists but is not a directory: {root}."
        )

    steps = _numbered_checkpoint_steps(root)
    resumable_steps = tuple(step for step in steps if step > 0)
    latest_step = resumable_steps[-1] if resumable_steps else None
    if resume:
        if latest_step is None:
            raise FileNotFoundError(
                "Cannot resume: checkpoint directory contains no positive "
                f"numbered checkpoint: {root}. Run without --resume to start "
                "from the configured pi0.5 base."
            )
        return _TrainCheckpointPolicy(
            root,
            "resume",
            resume=True,
            overwrite=False,
            latest_checkpoint_step=latest_step,
        )
    if latest_step is not None:
        raise FileExistsError(
            f"Checkpoint step {latest_step} already exists under {root}. "
            "Use --resume to continue that stage."
        )

    known_entries = set(_EMPTY_CHECKPOINT_ROOT_ENTRIES)
    known_entries.update(str(step) for step in steps)
    unexpected_entries = sorted(
        child.name for child in root.iterdir() if child.name not in known_entries
    )
    if unexpected_entries:
        raise FileExistsError(
            "Checkpoint directory has no resumable step but contains unknown "
            f"entries: {unexpected_entries}. Refusing implicit recovery: {root}."
        )
    return _TrainCheckpointPolicy(
        root,
        "recover_empty_checkpoint_root",
        resume=True,
        overwrite=False,
        latest_checkpoint_step=None,
    )


def _load_profile(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") not in {
        PROFILE_SCHEMA_V1,
        PROMPT_PROFILE_SCHEMA_V1,
        PROMPT_FAMILY_PROFILE_SCHEMA_V2,
        STAGE1_SCENARIO_PROFILE_SCHEMA_V1,
    }:
        raise ValueError(f"Unsupported prepared Skill training profile: {source}.")
    required = [
        "profile_id",
        "organized_dataset_root",
        "camera_selection",
        "camera_slot_to_dataset_key",
        "action_horizon",
        "action_pose_mode",
        "openpi_checkout",
        "openpi_base_config",
        "openpi_config_name",
        "openpi_asset_id",
    ]
    schema = payload.get("schema")
    is_prompt_profile = schema in {
        PROMPT_PROFILE_SCHEMA_V1,
        PROMPT_FAMILY_PROFILE_SCHEMA_V2,
    }
    if schema == PROFILE_SCHEMA_V1:
        required.extend(("sources", "target_skill_id"))
    elif schema == STAGE1_SCENARIO_PROFILE_SCHEMA_V1:
        required.extend(
            (
                "sources",
                "target_skill_id",
                "target_progress_step_id",
                "expected_total_raw_episode_count",
                "expected_total_raw_frame_count",
                "expected_total_eligible_episode_count",
                "expected_prepared_interval_window_count",
                "expected_prepared_interval_repeated_window_count",
                "expected_openpi_quality_valid_window_count",
                "expected_openpi_repeated_epoch_sample_count",
                "expected_policy_io_layout_hash",
                "source_mixing_policy",
                "default_prompt",
                "tail_policy",
            )
        )
    else:
        required.extend(
            (
                "prompt_stages",
                "sources",
                "expected_total_window_count",
                "expected_union_episode_count",
                "expected_task_id",
                "expected_policy_io_layout_hash",
                "source_mixing_policy",
                "prompt_stage_sampling_policy",
                "prompt_stage_sampling_seed",
                "expected_effective_epoch_window_count",
                "tail_policy",
            )
        )
        if schema == PROMPT_FAMILY_PROFILE_SCHEMA_V2:
            required.extend(
                (
                    "prompt_family_sampling_policy",
                    "prompt_family_sampling_seed",
                    "canonical_prompt_percent",
                    "training_variant_prompt_percent",
                    "max_full_token_length",
                    "prompt_state_dimension",
                )
            )
    missing = [name for name in required if not payload.get(name)]
    if missing:
        raise ValueError(f"Prepared Skill profile is missing: {missing}.")
    if (
        payload.get("openpi_observation_mode")
        == OPENPI_STATE_ONLY_PROFILE_V1
        and not payload.get("openpi_norm_stats_assets_dir")
    ):
        raise ValueError(
            "State-only OpenPI training requires "
            "openpi_norm_stats_assets_dir."
        )
    training_mode = str(payload.get("openpi_training_mode", "")).strip()
    if training_mode and training_mode != STANDARD_NON_RTC_TRAINING_MODE_V1:
        raise ValueError(
            "openpi_training_mode must be standard_non_rtc_v1 when declared."
        )
    _corrected_wrist_background_overrides(payload)
    if is_prompt_profile:
        raw_stages = payload.get("prompt_stages")
        if not isinstance(raw_stages, list) or len(raw_stages) < 2:
            raise ValueError(
                "Prepared prompt profile requires at least two prompt_stages."
            )
        resolved: list[dict[str, Any]] = []
        for item in raw_stages:
            if not isinstance(item, Mapping):
                raise ValueError("Prepared prompt stage must be an object.")
            stage_id = str(item.get("stage_id", "")).strip()
            skill_id = str(item.get("skill_id", "")).strip()
            prompt_key = (
                "prompt"
                if schema == PROMPT_PROFILE_SCHEMA_V1
                else "canonical_prompt"
            )
            prompt = str(item.get(prompt_key, "")).strip()
            training_variants = tuple(
                str(value).strip()
                for value in item.get("training_prompt_variants", ())
            )
            held_out_variants = tuple(
                str(value).strip()
                for value in item.get("held_out_prompt_variants", ())
            )
            profile_value = str(item.get("profile", "")).strip()
            expected_windows = int(item.get("expected_window_count", 0))
            expected_openpi_windows = int(
                item.get("expected_openpi_window_count", 0)
            )
            child_binding_hash = str(
                item.get("expected_child_binding_hash", "")
            ).strip()
            if (
                not stage_id
                or not skill_id
                or not prompt
                or not profile_value
                or expected_windows <= 0
                or expected_openpi_windows <= 0
                or len(child_binding_hash) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in child_binding_hash
                )
            ):
                raise ValueError(
                    "Prepared prompt stage requires stage_id, skill_id, prompt, "
                    "profile, expected_child_binding_hash, and positive "
                    "expected_window_count, and positive "
                    "expected_openpi_window_count."
                )
            if schema == PROMPT_FAMILY_PROFILE_SCHEMA_V2 and (
                len(training_variants) != 12
                or len(held_out_variants) != 4
                or any(not value for value in training_variants)
                or any(not value for value in held_out_variants)
            ):
                raise ValueError(
                    "Prepared prompt-family stage requires exactly 12 "
                    "non-empty training variants and 4 non-empty held-out "
                    "prompts."
                )
            child_path = Path(profile_value).expanduser()
            if not child_path.is_absolute():
                child_path = source.parent / child_path
            child = _load_profile(child_path)
            if child.get("schema") != PROFILE_SCHEMA_V1:
                raise ValueError(
                    "Prepared prompt stages must reference single-Skill profiles."
                )
            if str(child["target_skill_id"]) != skill_id:
                raise ValueError(
                    "Prepared prompt stage skill_id differs from its child profile."
                )
            resolved.append(
                {
                    "stage_id": stage_id,
                    "skill_id": skill_id,
                    "prompt": prompt,
                    "training_prompt_variants": training_variants,
                    "held_out_prompt_variants": held_out_variants,
                    "expected_window_count": expected_windows,
                    "expected_openpi_window_count": expected_openpi_windows,
                    "expected_child_binding_hash": child_binding_hash,
                    "profile_path": str(child_path.resolve()),
                    "profile": child,
                }
            )
        for key in ("stage_id", "skill_id", "prompt"):
            values = tuple(str(item[key]) for item in resolved)
            if len(set(values)) != len(values):
                raise ValueError(
                    f"Prepared prompt profile has duplicate {key}."
                )
        if schema == PROMPT_FAMILY_PROFILE_SCHEMA_V2:
            all_prompts = tuple(
                prompt
                for item in resolved
                for prompt in (
                    item["prompt"],
                    *item["training_prompt_variants"],
                    *item["held_out_prompt_variants"],
                )
            )
            if len(set(all_prompts)) != len(all_prompts):
                raise ValueError(
                    "Prepared prompt-family profile requires every canonical, "
                    "training, and held-out prompt to be globally unique."
                )
            if (
                int(payload["canonical_prompt_percent"]) != 80
                or int(payload["training_variant_prompt_percent"]) != 20
            ):
                raise ValueError(
                    "Prepared prompt-family profile requires an exact 80/20 "
                    "canonical/training-variant split."
                )
        common_keys = (
            "expected_task_id",
            "openpi_checkout",
            "openpi_base_config",
            "expected_policy_io_layout_hash",
            "camera_selection",
            "camera_slot_to_dataset_key",
            "action_horizon",
            "action_pose_mode",
            "source_mixing_policy",
            "tail_policy",
        )
        for item in resolved:
            child = item["profile"]
            mismatched = [
                key for key in common_keys if child.get(key) != payload.get(key)
            ]
            if mismatched:
                raise ValueError(
                    "Prepared prompt child profile differs from the joint "
                    f"contract: stage={item['stage_id']!r} keys={mismatched}."
                )
            joint_sources = tuple(
                (
                    str(source["source_dataset_id"]),
                    str(Path(str(source["dataset_root"])).expanduser().resolve()),
                )
                for source in payload["sources"]
            )
            child_sources = tuple(
                (
                    str(source["source_dataset_id"]),
                    str(Path(str(source["dataset_root"])).expanduser().resolve()),
                )
                for source in child["sources"]
            )
            if child_sources != joint_sources:
                raise ValueError(
                    "Prepared prompt child sources differ from the joint profile: "
                    f"stage={item['stage_id']!r}."
                )
        payload["_resolved_prompt_stages"] = resolved
        payload["_profile_path"] = str(source)
    elif schema == STAGE1_SCENARIO_PROFILE_SCHEMA_V1:
        source_values = payload.get("sources")
        if not isinstance(source_values, list) or not source_values:
            raise ValueError("Stage1 scenario profile requires source datasets.")
        expected_weights = {
            "fixed_middle": 1,
            "initial_multi_choice": 4,
            "midcourse_interrupt": 6,
        }
        seen_classes: set[str] = set()
        seen_ids: set[str] = set()
        raw_episode_total = 0
        raw_frame_total = 0
        for item in source_values:
            if not isinstance(item, Mapping):
                raise ValueError("Stage1 scenario source must be an object.")
            source_id = str(item.get("source_dataset_id", "")).strip()
            scenario_class = str(item.get("scenario_class", "")).strip()
            task_scope = str(item.get("task_scope", "")).strip()
            expected_task_id = str(item.get("expected_task_id", "")).strip()
            raw_task_aliases = item.get("expected_task_id_aliases", [])
            if not isinstance(raw_task_aliases, list):
                raise TypeError(
                    "expected_task_id_aliases must be a list when present."
                )
            task_aliases = tuple(
                str(value).strip() for value in raw_task_aliases
            )
            repeat_weight = int(item.get("repeat_weight", 0))
            raw_episodes = int(item.get("expected_raw_episode_count", 0))
            raw_frames = int(item.get("expected_raw_frame_count", 0))
            openpi_windows = int(
                item.get("expected_openpi_quality_valid_window_count", 0)
            )
            if (
                not source_id
                or source_id in seen_ids
                or scenario_class not in expected_weights
                or not task_scope
                or not expected_task_id
                or any(not value for value in task_aliases)
                or len(set(task_aliases)) != len(task_aliases)
                or expected_task_id in task_aliases
                or repeat_weight != expected_weights.get(scenario_class)
                or raw_episodes <= 0
                or raw_frames <= 0
                or openpi_windows <= 0
            ):
                raise ValueError(
                    "Stage1 scenario source requires a unique id, known class, "
                    "expected task id, positive inventory, and exact 1:4:6 weight."
                )
            if scenario_class in {
                "initial_multi_choice",
                "midcourse_interrupt",
            } and task_scope != "stage1_only":
                raise ValueError(
                    f"{scenario_class} must be declared stage1_only."
                )
            seen_ids.add(source_id)
            seen_classes.add(scenario_class)
            raw_episode_total += raw_episodes
            raw_frame_total += raw_frames
        if seen_classes != set(expected_weights):
            raise ValueError(
                "Stage1 scenario profile must include all three scenario classes."
            )
        if (
            raw_episode_total != int(payload["expected_total_raw_episode_count"])
            or raw_frame_total != int(payload["expected_total_raw_frame_count"])
        ):
            raise ValueError(
                "Stage1 scenario source inventories differ from profile totals."
            )
        if (
            str(payload["source_mixing_policy"])
            != OPENPI_SOURCE_MIXING_REPEAT_BY_SOURCE_V1
        ):
            raise ValueError(
                "Stage1 scenario profile requires exact per-source repetition."
            )
        payload["_profile_path"] = str(source)
    return payload


def _sources(profile: Mapping[str, Any]) -> tuple[PreparedSourceDataset, ...]:
    values = profile.get("sources")
    if not isinstance(values, list) or not values:
        raise ValueError("Prepared Skill profile requires source datasets.")
    return tuple(
        PreparedSourceDataset(
            str(value["source_dataset_id"]),
            Path(str(value["dataset_root"])),
        )
        for value in values
    )


def _validate_source_task_ids(
    profile: Mapping[str, Any],
    sources: tuple[PreparedSourceDataset, ...],
) -> None:
    global_expected = str(profile.get("expected_task_id", "")).strip()
    expected_by_source = {
        str(item["source_dataset_id"]): str(
            item.get("expected_task_id", global_expected)
        ).strip()
        for item in profile.get("sources", ())
        if isinstance(item, Mapping)
    }
    aliases_by_source = {
        str(item["source_dataset_id"]): tuple(
            str(value).strip()
            for value in item.get("expected_task_id_aliases", ())
        )
        for item in profile.get("sources", ())
        if isinstance(item, Mapping)
    }
    if not global_expected and not any(expected_by_source.values()):
        return

    mismatches: list[str] = []
    for source in sources:
        expected = expected_by_source.get(
            source.source_dataset_id,
            global_expected,
        )
        if not expected:
            raise ValueError(
                f"Source {source.source_dataset_id!r} has no expected task_id."
            )
        aliases = aliases_by_source.get(source.source_dataset_id, ())
        if (
            any(not value for value in aliases)
            or len(set(aliases)) != len(aliases)
            or expected in aliases
        ):
            raise ValueError(
                "Source task aliases must be non-empty, unique, and must not "
                f"repeat the primary task id: {source.source_dataset_id!r}."
            )
        allowed_task_ids = {expected, *aliases}
        root = source.dataset_root.expanduser().resolve()
        info_path = root / "meta" / "info.json"
        tasks_path = root / "meta" / "tasks.jsonl"
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Cannot read source task metadata: {info_path}."
            ) from exc
        if not isinstance(info, Mapping):
            raise ValueError(f"Source info metadata is not an object: {info_path}.")
        info_task_id = str(info.get("task_id", "")).strip()
        if info_task_id != expected:
            mismatches.append(
                f"{source.source_dataset_id}:meta/info.json={info_task_id!r}"
            )

        try:
            task_rows = [
                json.loads(line)
                for line in tasks_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Cannot read source task metadata: {tasks_path}."
            ) from exc
        task_ids = tuple(
            str(row.get("task_id", "")).strip()
            for row in task_rows
            if isinstance(row, Mapping)
        )
        if not task_rows or len(task_ids) != len(task_rows):
            mismatches.append(
                f"{source.source_dataset_id}:meta/tasks.jsonl=invalid"
            )
        elif set(task_ids) != allowed_task_ids:
            mismatches.append(
                f"{source.source_dataset_id}:meta/tasks.jsonl={task_ids!r}"
            )

    if mismatches:
        raise ValueError(
            "Source task_id differs from its explicitly bound expected value: "
            + "; ".join(mismatches)
        )


def _skill_projection_catalog(
    profile: Mapping[str, Any],
) -> SkillProjectionCatalog:
    catalog_id = str(
        profile.get(
            "skill_projection_catalog_id",
            SKILL_PROJECTION_CATALOG_WRITING_STAGE1_V2,
        )
    ).strip()
    if catalog_id == SKILL_PROJECTION_CATALOG_WRITING_STAGE1_V2:
        return load_builtin_writing_stage1_skill_projection_catalog()
    if catalog_id == SKILL_PROJECTION_CATALOG_WRITING_FOUR_STAGE_SUBTASKS_V2:
        return load_builtin_writing_four_stage_subtasks_skill_projection_catalog()
    raise ValueError(
        f"Unsupported Skill projection catalog id: {catalog_id!r}."
    )


def _source_expected_counts(profile: Mapping[str, Any]) -> dict[str, int]:
    return {
        str(value["source_dataset_id"]): int(
            value["expected_eligible_episode_count"]
        )
        for value in profile["sources"]
    }


def _included_counts_by_source(
    report: Any,
    sources: tuple[PreparedSourceDataset, ...],
) -> dict[str, int]:
    excluded_by_source: dict[str, int] = {}
    for item in report.exclusions:
        excluded_by_source[item.source_dataset_id] = (
            excluded_by_source.get(item.source_dataset_id, 0) + 1
        )
    from dataset_interface.progress_annotation_jobs import (
        list_progress_annotation_work_items,
    )

    return {
        source.source_dataset_id: (
            len(list_progress_annotation_work_items(source.dataset_root))
            - excluded_by_source.get(source.source_dataset_id, 0)
        )
        for source in sources
    }


def _validate_profile_contracts(
    profile: Mapping[str, Any],
    view_manifest: Mapping[str, Any],
) -> None:
    expected_progress = str(profile.get("expected_progress_profile_hash", ""))
    expected_stage = str(profile.get("expected_stage_profile_hash", ""))
    if (
        view_manifest.get("progress_profile_hash") != expected_progress
        or view_manifest.get("stage_profile_hash") != expected_stage
    ):
        raise ValueError(
            "Prepared view Progress/Stage hashes differ from the profile."
        )
    projection = view_manifest.get("skill_projection")
    if not isinstance(projection, Mapping):
        raise ValueError("Prepared view has no Skill projection capability.")


def _prepare_prompt_profile(
    profile: Mapping[str, Any],
    *,
    dry_run: bool,
    verify_source_files: bool,
) -> tuple[
    tuple[Any, ...],
    OpenPIPreparedPromptBinding | None,
    Path | None,
]:
    resolved = profile.get("_resolved_prompt_stages")
    if not isinstance(resolved, list) or len(resolved) < 2:
        raise ValueError("Prepared prompt profile stages were not resolved.")
    reports: list[Any] = []
    child_values: list[tuple[str, str, OpenPIPreparedSkillBinding]] = []
    for item in resolved:
        child_profile = item["profile"]
        # Eligibility is re-audited against the pinned annotation selection,
        # while the exact windows come from an immutable child binding that
        # already passed source-file verification when it was prepared. This
        # prevents a joint run from silently absorbing mutable annotations and
        # avoids rescanning the same media four times.
        report, _audit_binding, _audit_root = prepare_profile(
            child_profile,
            dry_run=True,
            verify_source_files=False,
        )
        reports.append(
            _PromptStageReport(
                stage_id=str(item["stage_id"]),
                skill_id=str(item["skill_id"]),
                prompt=str(item["prompt"]),
                expected_window_count=int(item["expected_window_count"]),
                report=report,
            )
        )
        if dry_run:
            continue
        child_binding_hash = str(item["expected_child_binding_hash"])
        child_binding_root = (
            Path(str(child_profile["organized_dataset_root"]))
            .expanduser()
            .resolve()
            / "derived/openpi_prepared_skill_bindings"
            / child_binding_hash
        )
        binding = load_openpi_prepared_skill_binding_manifest(
            child_binding_root
        )
        if binding.binding_hash != child_binding_hash:
            raise ValueError(
                "Prepared prompt child binding hash differs from the profile."
            )
        if binding.skill_id != str(item["skill_id"]):
            raise ValueError(
                "Prepared prompt child binding Skill differs from the profile."
            )
        actual_windows = int(binding.to_payload()["window_count"])
        expected_windows = int(item["expected_window_count"])
        if actual_windows != expected_windows:
            raise ValueError(
                "Prepared prompt child window count differs: "
                f"stage={item['stage_id']!r} actual={actual_windows} "
                f"expected={expected_windows}."
            )
        child_values.append(
            (str(item["stage_id"]), str(item["prompt"]), binding)
        )
    if dry_run:
        return tuple(reports), None, None
    binding = compose_openpi_prepared_prompt_binding(
        child_values,
        organized_dataset_root=str(profile["organized_dataset_root"]),
        expected_openpi_window_counts={
            str(item["stage_id"]): int(
                item["expected_openpi_window_count"]
            )
            for item in resolved
        },
        prompt_stage_sampling_policy=str(
            profile["prompt_stage_sampling_policy"]
        ),
        prompt_stage_sampling_seed=int(profile["prompt_stage_sampling_seed"]),
        training_prompt_variants=(
            None
            if profile.get("schema") != PROMPT_FAMILY_PROFILE_SCHEMA_V2
            else {
                str(item["stage_id"]): item["training_prompt_variants"]
                for item in resolved
            }
        ),
        held_out_prompt_variants=(
            None
            if profile.get("schema") != PROMPT_FAMILY_PROFILE_SCHEMA_V2
            else {
                str(item["stage_id"]): item["held_out_prompt_variants"]
                for item in resolved
            }
        ),
        prompt_family_sampling_policy=(
            None
            if profile.get("schema") != PROMPT_FAMILY_PROFILE_SCHEMA_V2
            else str(profile["prompt_family_sampling_policy"])
        ),
        prompt_family_sampling_seed=int(
            profile.get("prompt_family_sampling_seed", 20260811)
        ),
    )
    expected_contract = (
        tuple(
            (str(key), str(value))
            for key, value in dict(
                profile["camera_slot_to_dataset_key"]
            ).items()
        ),
        int(profile["action_horizon"]),
        str(profile["action_pose_mode"]),
        str(profile["source_mixing_policy"]),
    )
    actual_contract = (
        binding.camera_slot_to_dataset_key,
        binding.action_horizon,
        binding.action_pose_mode,
        binding.source_mixing_policy,
    )
    if actual_contract != expected_contract:
        raise ValueError(
            "Prepared prompt profile differs from its child model-I/O contract."
        )
    actual_total = int(binding.to_payload()["window_count"])
    expected_total = int(profile["expected_total_window_count"])
    if actual_total != expected_total:
        raise ValueError(
            "Prepared prompt total window count differs: "
            f"actual={actual_total} expected={expected_total}."
        )
    minimum_stage_windows = effective_prompt_stage_window_count(binding)
    effective_epoch_windows = minimum_stage_windows * len(binding.prompt_skills)
    expected_effective = int(profile["expected_effective_epoch_window_count"])
    if effective_epoch_windows != expected_effective:
        raise ValueError(
            "Prepared prompt effective epoch window count differs: "
            f"actual={effective_epoch_windows} expected={expected_effective}."
        )
    expected_union = int(profile["expected_union_episode_count"])
    if len(binding.episodes) != expected_union:
        raise ValueError(
            "Prepared prompt Episode union count differs: "
            f"actual={len(binding.episodes)} expected={expected_union}."
        )
    root = materialize_openpi_prepared_skill_binding(
        binding,
        str(profile["organized_dataset_root"]),
    )
    return tuple(reports), binding, root


def _prepare_stage1_scenario_profile(
    profile: Mapping[str, Any],
    *,
    dry_run: bool,
) -> tuple[Any, OpenPIPreparedSkillBinding | None, Path | None]:
    sources = _sources(profile)
    _validate_source_task_ids(profile, sources)
    report, selections, training_view_hash = (
        prepare_stage1_scenario_selections(
            sources,
            tuple(profile["sources"]),
            target_skill_id=str(profile["target_skill_id"]),
            target_progress_step_id=str(
                profile["target_progress_step_id"]
            ),
            action_horizon=int(profile["action_horizon"]),
        )
    )
    actual_total = len(report.included_episode_identities)
    expected_total = int(profile["expected_total_eligible_episode_count"])
    actual_by_source = {
        str(item["source_dataset_id"]): int(item["included_episode_count"])
        for item in report.source_statistics
    }
    if actual_total != expected_total:
        raise ValueError(
            "Stage1 eligible Episode count differs: "
            f"actual={actual_total} expected={expected_total} "
            f"by_source={actual_by_source}."
        )
    expected_by_source = {
        str(item["source_dataset_id"]): int(
            item["expected_eligible_episode_count"]
        )
        for item in profile["sources"]
    }
    if actual_by_source != expected_by_source:
        raise ValueError(
            "Stage1 eligible source counts differ: "
            f"actual={actual_by_source} expected={expected_by_source}."
        )
    actual_raw_windows = sum(
        int(item["prepared_interval_window_count"])
        for item in report.source_statistics
    )
    actual_effective_windows = sum(
        int(item["prepared_interval_repeated_window_count"])
        for item in report.source_statistics
    )
    if (
        actual_raw_windows
        != int(profile["expected_prepared_interval_window_count"])
        or actual_effective_windows
        != int(profile["expected_prepared_interval_repeated_window_count"])
    ):
        raise ValueError(
            "Stage1 window counts differ from the profile: "
            f"actual_raw={actual_raw_windows} "
            "expected_raw="
            f"{profile['expected_prepared_interval_window_count']} "
            f"actual_effective={actual_effective_windows} "
            "expected_effective="
            f"{profile['expected_prepared_interval_repeated_window_count']}."
        )
    expected_layout_hash = str(profile["expected_policy_io_layout_hash"])
    layout_mismatches: list[str] = []
    for source in sources:
        info = json.loads(
            (source.dataset_root / "meta/info.json").read_text(
                encoding="utf-8"
            )
        )
        actual_layout_hash = str(info.get("policy_io_layout_hash", ""))
        if actual_layout_hash != expected_layout_hash:
            layout_mismatches.append(
                f"{source.source_dataset_id}={actual_layout_hash!r}"
            )
    if layout_mismatches:
        raise ValueError(
            "Stage1 source policy-I/O hashes differ from the profile: "
            + "; ".join(layout_mismatches)
        )
    if dry_run:
        return report, None, None

    catalog = load_builtin_writing_stage1_skill_projection_catalog()
    target_skill_id = str(profile["target_skill_id"])
    skill = catalog.skill_by_id.get(target_skill_id)
    if skill is None:
        raise ValueError(f"Unknown Stage1 Skill {target_skill_id!r}.")
    organized_root = Path(
        str(profile["organized_dataset_root"])
    ).expanduser().resolve()
    binding = OpenPIPreparedSkillBinding(
        view_root=(
            organized_root / "derived/training_views" / training_view_hash
        ),
        organized_dataset_root=organized_root,
        training_view_hash=training_view_hash,
        skill_id=target_skill_id,
        skill_training_contract_hash=skill.semantic_content_hash,
        training_domain="NOMINAL",
        camera_slot_to_dataset_key=tuple(
            (str(slot), str(key))
            for slot, key in dict(
                profile["camera_slot_to_dataset_key"]
            ).items()
        ),
        action_horizon=int(profile["action_horizon"]),
        action_pose_mode=str(profile["action_pose_mode"]),
        source_mixing_policy=str(profile["source_mixing_policy"]),
        episodes=selections,
        source_repeat_weights=tuple(
            (
                str(item["source_dataset_id"]),
                int(item["repeat_weight"]),
            )
            for item in profile["sources"]
        ),
        selection_policy_id=STAGE1_SCENARIO_SELECTION_POLICY_V1,
    )
    binding_root = materialize_openpi_prepared_skill_binding(
        binding,
        organized_root,
    )
    report_root = organized_root / "build_reports"
    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / f"{report.semantic_content_hash}.json"
    if not report_path.exists():
        report_path.write_text(
            json.dumps(report.to_payload(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report, binding, binding_root


def _prompt_family_token_preflight(
    profile: Mapping[str, Any],
    data_config: Any,
    model_config: Any,
) -> dict[str, Any] | None:
    """Measure untruncated PI0.5 prompt+state tokens for the full family."""

    if profile.get("schema") != PROMPT_FAMILY_PROFILE_SCHEMA_V2:
        return None
    model_inputs = tuple(getattr(data_config.model_transforms, "inputs", ()))
    candidates = tuple(
        transform
        for transform in model_inputs
        if transform.__class__.__name__ == "TokenizePrompt"
    )
    if len(candidates) != 1:
        raise OpenPIPreparedSkillError(
            "Prompt-family preflight requires exactly one TokenizePrompt transform."
        )
    transform = candidates[0]
    if not bool(getattr(transform, "discrete_state_input", False)):
        raise OpenPIPreparedSkillError(
            "Prompt-family preflight requires PI0.5 discrete state input."
        )
    tokenizer = getattr(transform, "tokenizer", None)
    sentencepiece = getattr(tokenizer, "_tokenizer", None)
    tokenizer_max_len = int(getattr(tokenizer, "_max_len", 0))
    model_max_len = int(getattr(model_config, "max_token_len", 0))
    if (
        sentencepiece is None
        or not callable(getattr(sentencepiece, "encode", None))
        or tokenizer_max_len <= 0
        or tokenizer_max_len != model_max_len
    ):
        raise OpenPIPreparedSkillError(
            "Prompt-family preflight could not bind the configured PaliGemma "
            "tokenizer length."
        )
    state_dimension = int(profile["prompt_state_dimension"])
    configured_limit = int(profile["max_full_token_length"])
    if state_dimension <= 0 or configured_limit <= 0:
        raise OpenPIPreparedSkillError(
            "Prompt-family token preflight limits must be positive."
        )
    records: list[dict[str, Any]] = []
    for stage in profile["prompt_stages"]:
        prompt_groups = {
            "canonical": (str(stage["canonical_prompt"]),),
            "training_variant": tuple(
                str(value) for value in stage["training_prompt_variants"]
            ),
            "held_out": tuple(
                str(value) for value in stage["held_out_prompt_variants"]
            ),
        }
        for prompt_kind, prompts in prompt_groups.items():
            for prompt in prompts:
                for state_value in (-1.0, 0.0, 1.0):
                    state = np.full(
                        (state_dimension,), state_value, dtype=np.float32
                    )
                    discretized_state = (
                        np.digitize(
                            state,
                            bins=np.linspace(-1, 1, 256 + 1)[:-1],
                        )
                        - 1
                    )
                    cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
                    state_text = " ".join(map(str, discretized_state))
                    full_prompt = (
                        f"Task: {cleaned}, State: {state_text};\nAction: "
                    )
                    full_length = len(
                        sentencepiece.encode(full_prompt, add_bos=True)
                    )
                    records.append(
                        {
                            "stage_id": str(stage["stage_id"]),
                            "prompt_kind": prompt_kind,
                            "state_value": state_value,
                            "full_token_length": full_length,
                        }
                    )
    longest = max(records, key=lambda item: item["full_token_length"])
    if int(longest["full_token_length"]) > configured_limit:
        raise OpenPIPreparedSkillError(
            "Prompt-family full token length exceeds the profile limit: "
            f"longest={longest} limit={configured_limit}."
        )
    if int(longest["full_token_length"]) > tokenizer_max_len:
        raise OpenPIPreparedSkillError(
            "Prompt-family tokenization would be truncated by OpenPI: "
            f"longest={longest} max_token_len={tokenizer_max_len}."
        )
    return {
        "tokenizer": "PaligemmaTokenizer",
        "discrete_state_input": True,
        "prompt_count": sum(
            1 + len(stage["training_prompt_variants"])
            + len(stage["held_out_prompt_variants"])
            for stage in profile["prompt_stages"]
        ),
        "representative_state_values": [-1.0, 0.0, 1.0],
        "state_dimension": state_dimension,
        "measured_combination_count": len(records),
        "max_full_token_length": int(longest["full_token_length"]),
        "profile_limit": configured_limit,
        "openpi_max_token_len": tokenizer_max_len,
        "longest_case": longest,
        "truncation_detected": False,
    }


def prepare_profile(
    profile: Mapping[str, Any],
    *,
    dry_run: bool = False,
    verify_source_files: bool = True,
) -> tuple[
    Any,
    OpenPIPreparedSkillBinding | OpenPIPreparedPromptBinding | None,
    Path | None,
]:
    """Compile exact eligible Episodes and optionally publish immutable views."""

    if profile.get("schema") in {
        PROMPT_PROFILE_SCHEMA_V1,
        PROMPT_FAMILY_PROFILE_SCHEMA_V2,
    }:
        return _prepare_prompt_profile(
            profile,
            dry_run=dry_run,
            verify_source_files=verify_source_files,
        )
    if profile.get("schema") == STAGE1_SCENARIO_PROFILE_SCHEMA_V1:
        return _prepare_stage1_scenario_profile(
            profile,
            dry_run=dry_run,
        )

    sources = _sources(profile)
    _validate_source_task_ids(profile, sources)
    catalog = _skill_projection_catalog(profile)
    report = select_strict_successful_skill_episodes(
        sources,
        skill_projection_catalog=catalog,
        target_skill_id=str(profile["target_skill_id"]),
        annotation_selection={
            str(key): str(value)
            for key, value in dict(
                profile.get("annotation_selection", {})
            ).items()
        },
        require_no_abnormal_intervals=bool(
            profile.get("require_no_abnormal_intervals", True)
        ),
    )
    counts = _included_counts_by_source(report, sources)
    expected_counts = _source_expected_counts(profile)
    if counts != expected_counts:
        raise ValueError(
            f"Eligible Episode counts differ: actual={counts} "
            f"expected={expected_counts}."
        )
    expected_total = int(profile["expected_total_eligible_episode_count"])
    if len(report.included_episode_identities) != expected_total:
        raise ValueError(
            "Total eligible Episode count differs: "
            f"actual={len(report.included_episode_identities)} "
            f"expected={expected_total}."
        )
    if dry_run:
        return report, None, None

    organized_root = Path(str(profile["organized_dataset_root"]))
    expected_binding_hash = str(
        profile.get("expected_prepared_binding_hash", "")
    ).strip()
    if expected_binding_hash:
        if (
            len(expected_binding_hash) != 64
            or any(char not in "0123456789abcdef" for char in expected_binding_hash)
        ):
            raise ValueError(
                "expected_prepared_binding_hash must be a lowercase SHA-256."
            )
        binding_root = (
            organized_root.expanduser().resolve()
            / "derived/openpi_prepared_skill_bindings"
            / expected_binding_hash
        )
        binding = load_openpi_prepared_skill_binding_manifest(binding_root)
        if binding.binding_hash != expected_binding_hash:
            raise ValueError(
                "Prepared binding hash differs from the replay profile."
            )
        expected_identities = tuple(report.included_episode_identities)
        binding_identities = tuple(
            episode.episode_capture_identity_hash for episode in binding.episodes
        )
        if binding_identities != expected_identities:
            raise ValueError(
                "Prepared replay binding Episode order differs from the current "
                "pinned eligibility audit."
            )
        expected_roots = {
            source.source_dataset_id: source.dataset_root.resolve()
            for source in sources
        }
        binding_roots: dict[str, Path] = {}
        for episode in binding.episodes:
            existing = binding_roots.setdefault(
                episode.source_dataset_id,
                episode.source_dataset_root.resolve(),
            )
            if existing != episode.source_dataset_root.resolve():
                raise ValueError(
                    "Prepared replay binding contains inconsistent source roots."
                )
        if binding_roots != expected_roots:
            raise ValueError(
                "Prepared replay binding source roots differ from the profile."
            )
        expected_contract = {
            "skill_id": str(profile["target_skill_id"]),
            "training_domain": str(profile.get("training_domain", "NOMINAL")),
            "camera_slot_to_dataset_key": tuple(
                (str(key), str(value))
                for key, value in dict(
                    profile["camera_slot_to_dataset_key"]
                ).items()
            ),
            "action_horizon": int(profile["action_horizon"]),
            "action_pose_mode": str(profile["action_pose_mode"]),
            "source_mixing_policy": str(profile["source_mixing_policy"]),
        }
        actual_contract = {
            "skill_id": binding.skill_id,
            "training_domain": binding.training_domain,
            "camera_slot_to_dataset_key": binding.camera_slot_to_dataset_key,
            "action_horizon": binding.action_horizon,
            "action_pose_mode": binding.action_pose_mode,
            "source_mixing_policy": binding.source_mixing_policy,
        }
        if actual_contract != expected_contract:
            raise ValueError(
                "Prepared replay binding contract differs from the profile."
            )
        if verify_source_files:
            from dataset_interface.prepared_training import (
                load_prepared_training_view,
            )

            loaded = load_prepared_training_view(
                binding.view_root,
                organized_dataset_root=organized_root,
                verify_source_files=True,
            )
            if loaded.training_view_hash != binding.training_view_hash:
                raise ValueError(
                    "Prepared replay binding points at a different training view."
                )
        return report, binding, binding_root

    result = build_prepared_training_view(
        organized_root,
        sources,
        annotation_selection=report.annotation_selection,
        include_episode_identities=report.included_episode_identities,
        modality_selection=("numeric", "video"),
        camera_selection=tuple(str(value) for value in profile["camera_selection"]),
        column_projection=("timestamp", "action", "observation.state"),
        skill_projection_catalog=catalog,
        target_skill_id=str(profile["target_skill_id"]),
    )
    from dataset_interface.prepared_training import load_prepared_training_view

    loaded = load_prepared_training_view(
        result.view_root,
        verify_source_files=verify_source_files,
    )
    _validate_profile_contracts(profile, loaded.manifest)
    binding = load_openpi_prepared_skill_binding(
        result.view_root,
        skill_id=str(profile["target_skill_id"]),
        training_domain=str(profile.get("training_domain", "NOMINAL")),
        camera_slot_to_dataset_key=dict(profile["camera_slot_to_dataset_key"]),
        action_horizon=int(profile["action_horizon"]),
        action_pose_mode=str(profile["action_pose_mode"]),
        source_mixing_policy=str(profile["source_mixing_policy"]),
        verify_source_files=verify_source_files,
    )
    binding_root = materialize_openpi_prepared_skill_binding(
        binding,
        organized_root,
    )
    report_root = organized_root / "build_reports"
    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / f"{report.semantic_content_hash}.json"
    if not report_path.exists():
        report_path.write_text(
            json.dumps(report.to_payload(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report, binding, binding_root


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import OpenPI script {path}.")
    module = importlib.util.module_from_spec(spec)
    had_previous = name in sys.modules
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if had_previous:
            sys.modules[name] = previous
        else:
            sys.modules.pop(name, None)
        raise
    return module


def _openpi_config(
    profile: Mapping[str, Any],
    binding: OpenPIPreparedSkillBinding | OpenPIPreparedPromptBinding,
    binding_root: Path,
    *,
    exp_name: str | None,
    resume: bool,
    overwrite: bool,
) -> tuple[Any, Any, Path]:
    checkout = Path(str(profile["openpi_checkout"])).expanduser().resolve()
    training_config = _activate_openpi_checkout(profile)
    from openpi.policies import dexteleop_v23_policy

    base = training_config.get_config(str(profile["openpi_base_config"]))
    state_augmentation_source_name = str(
        profile.get("openpi_state_augmentation_from_config", "")
    ).strip()
    state_augmentation = base.data.state_augmentation
    if state_augmentation_source_name:
        augmentation_source = training_config.get_config(
            state_augmentation_source_name
        )
        state_augmentation = getattr(
            augmentation_source.data,
            "state_augmentation",
            None,
        )
        if state_augmentation is None:
            raise ValueError(
                "OpenPI state-augmentation source has no compatible config."
            )
    state_augmentation_mode = str(
        profile.get("openpi_state_augmentation_mode", "configured_v1")
    ).strip()
    if state_augmentation_mode == "disabled_v1":
        state_augmentation = dataclasses.replace(
            state_augmentation,
            enabled=False,
            noise_std=0.0,
            dropout_probability=0.0,
            structured_enabled=False,
            arm_translation_std_m=0.0,
            arm_rotation_std_deg=0.0,
            gripper_noise_std=0.0,
            side_dropout_probability=0.0,
            arm_dropout_probability=0.0,
            end_effector_dropout_probability=0.0,
        )
    elif state_augmentation_mode != "configured_v1":
        raise ValueError(
            "openpi_state_augmentation_mode must be configured_v1 or disabled_v1."
        )
    if (
        int(base.model.action_horizon) != binding.action_horizon
        or tuple(base.model.image_keys) != binding.camera_slots
        or str(base.data.action_pose_mode) != binding.action_pose_mode
        or str(base.data.last_action_pose_mode) != binding.action_pose_mode
        or str(base.data.tail_policy) != str(profile.get("tail_policy", "drop"))
    ):
        raise ValueError(
            "OpenPI base config differs from the prepared h/camera/action contract."
        )
    source_values = binding_source_manifests(binding, binding_root)
    source_configs = tuple(
        dexteleop_v23_policy.DexTeleopV23DatasetSourceConfig(
            name=source_id,
            dataset_root=source_root,
            sample_weight=(
                binding.repeat_weight_for(source_id)
                if isinstance(binding, OpenPIPreparedSkillBinding)
                else 1
            ),
            front_occlusion_schedule=((0.0, 1.0),),
            explicit_mask_keys=base.data.explicit_mask_keys,
            episode_manifest=episode_manifest,
        )
        for source_id, source_root, episode_manifest in source_values
    )
    first_root = source_values[0][1]
    info_path = first_root / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    layout_hash = str(info.get("policy_io_layout_hash", ""))
    expected_layout_hash = str(profile["expected_policy_io_layout_hash"])
    if layout_hash != expected_layout_hash:
        raise ValueError(
            f"Source policy-I/O hash differs: {layout_hash} != {expected_layout_hash}."
        )
    norm_stats_assets_value = str(
        profile.get("openpi_norm_stats_assets_dir") or ""
    ).strip()
    norm_stats_assets_dir = None
    if norm_stats_assets_value:
        configured_assets_dir = Path(norm_stats_assets_value).expanduser()
        norm_stats_assets_dir = (
            configured_assets_dir
            if configured_assets_dir.is_absolute()
            else checkout / configured_assets_dir
        ).resolve()
        expected_asset_id = str(profile["openpi_asset_id"])
        norm_stats_path = (
            norm_stats_assets_dir / expected_asset_id / "norm_stats.json"
        )
        if not norm_stats_path.is_file():
            raise FileNotFoundError(
                f"Prepared OpenPI norm_stats.json is missing: {norm_stats_path}."
            )
    assets = dataclasses.replace(
        base.data.assets,
        assets_dir=(
            str(norm_stats_assets_dir)
            if norm_stats_assets_dir is not None
            else base.data.assets.assets_dir
        ),
        asset_id=str(profile["openpi_asset_id"]),
    )
    data_overrides: dict[str, Any] = {}
    configured_final_mask_root = str(
        profile.get("openpi_final_mask_root") or ""
    ).strip()
    if configured_final_mask_root:
        final_mask_root_path = Path(configured_final_mask_root).expanduser()
        if not final_mask_root_path.is_absolute():
            final_mask_root_path = checkout / final_mask_root_path
        data_overrides["final_mask_root"] = str(final_mask_root_path.resolve())
    corrected_background = _corrected_wrist_background_overrides(profile)
    data_overrides.update(corrected_background)
    data = dataclasses.replace(
        base.data,
        assets=assets,
        dataset_root=str(first_root),
        layout_path=str(info_path),
        layout_hash=expected_layout_hash,
        camera_slots=binding.camera_slots,
        action_pose_mode=binding.action_pose_mode,
        last_action_pose_mode=binding.action_pose_mode,
        tail_policy="drop",
        strict_media_integrity=True,
        state_augmentation=state_augmentation,
        dataset_sources=source_configs,
        default_prompt=str(
            profile.get("default_prompt", base.data.default_prompt)
        ),
        **data_overrides,
    )
    if corrected_background:
        explicit_mask_keys = tuple(str(value) for value in data.explicit_mask_keys)
        if (
            "wrist_final_keep_mask_paths" not in explicit_mask_keys
            or "wrist_depth_mask_paths" in explicit_mask_keys
            or not bool(data.strict_media_integrity)
        ):
            raise ValueError(
                "Corrected wrist background training requires strict media, "
                "final keep-mask input, and no raw depth-mask fallback."
            )
    train_overrides: dict[str, Any] = {}
    for profile_key, config_key in (
        ("openpi_assets_base_dir", "assets_base_dir"),
        ("openpi_checkpoint_base_dir", "checkpoint_base_dir"),
    ):
        configured_root = str(profile.get(profile_key) or "").strip()
        if configured_root:
            root_path = Path(configured_root).expanduser()
            if not root_path.is_absolute():
                root_path = checkout / root_path
            train_overrides[config_key] = str(root_path.resolve())
    if "openpi_num_train_steps" in profile:
        train_steps = int(profile["openpi_num_train_steps"])
        save_interval = int(profile.get("openpi_save_interval", base.save_interval))
        keep_period = int(profile.get("openpi_keep_period", base.keep_period))
        if train_steps <= 0 or save_interval <= 0 or keep_period <= 0:
            raise ValueError(
                "OpenPI train steps, save interval, and keep period must be positive."
            )
        if train_steps % save_interval or train_steps % keep_period:
            raise ValueError(
                "OpenPI final train step must align with save_interval and keep_period."
            )
        lr_schedule = base.lr_schedule
        if dataclasses.is_dataclass(lr_schedule) and hasattr(
            lr_schedule, "decay_steps"
        ):
            lr_schedule = dataclasses.replace(
                lr_schedule,
                decay_steps=train_steps,
            )
        train_overrides.update(
            num_train_steps=train_steps,
            save_interval=save_interval,
            keep_period=keep_period,
            lr_schedule=lr_schedule,
        )
    state_only_bridge = OpenPIStateOnlyInputBridge.from_profile_value(
        profile.get("openpi_observation_mode"),
        image_keys=tuple(str(value) for value in base.model.image_keys),
    )
    if state_only_bridge is not None:
        train_overrides["train_image_augmentation"] = False
    config = dataclasses.replace(
        base,
        name=str(profile["openpi_config_name"]),
        data=data,
        exp_name=(
            str(exp_name)
            if exp_name
            else str(profile.get("default_exp_name", ""))
        ),
        resume=bool(resume),
        overwrite=bool(overwrite),
        num_workers=int(profile.get("openpi_num_workers", base.num_workers)),
        **train_overrides,
    )
    if config.num_workers < 0:
        raise ValueError("openpi_num_workers must be non-negative.")
    return config, training_config, checkout


def _bind_metadata_value(
    metadata: dict[str, Any],
    key: str,
    value: Mapping[str, Any],
) -> None:
    metadata[key] = copy.deepcopy(dict(value))
    for container_key in ("dataset_adapter", "policy_artifact"):
        container = metadata.get(container_key)
        if isinstance(container, dict):
            container[key] = copy.deepcopy(dict(value))


def _natural_mixing_metadata(
    config: Any,
    binding: OpenPIPreparedSkillBinding | OpenPIPreparedPromptBinding,
) -> dict[str, Any]:
    sources = tuple(getattr(config.data, "dataset_sources", ()) or ())
    if not sources:
        raise ValueError("Prepared natural merge requires dataset_sources.")
    weighted = (
        isinstance(binding, OpenPIPreparedSkillBinding)
        and bool(binding.source_repeat_weights)
    )
    metadata = {
        "schema": (
            "dexteleop_v23_bound_source_repeat_v1"
            if weighted
            else "dexteleop_v23_natural_merge_v1"
        ),
        "source_mixing_policy": binding.source_mixing_policy,
        "source_order": [str(source.name) for source in sources],
        "each_quality_valid_window_once_per_epoch": not weighted,
        "repeat_each_quality_valid_window_by_source_weight": weighted,
        "shuffle": "deterministic_individual_sample_epoch_shuffle_v1",
        "sources": [source.to_dict() for source in sources],
    }
    if weighted:
        assert isinstance(binding, OpenPIPreparedSkillBinding)
        metadata["source_repeat_weights"] = {
            source_id: binding.repeat_weight_for(source_id)
            for source_id in binding.source_dataset_ids
        }
    if isinstance(binding, OpenPIPreparedPromptBinding):
        metadata.update(
            prompt_stage_sampling_policy=(
                binding.prompt_stage_sampling_policy
            ),
            prompt_stage_sampling_seed=binding.prompt_stage_sampling_seed,
            prepared_interval_window_counts_by_stage={
                skill.stage_id: skill.window_count
                for skill in binding.prompt_skills
            },
            openpi_quality_valid_window_counts_by_stage={
                skill.stage_id: skill.expected_openpi_window_count
                for skill in binding.prompt_skills
            },
            effective_window_count_per_stage=(
                effective_prompt_stage_window_count(binding)
            ),
        )
    return metadata


def _source_partitioned_final_mask_roots(
    profile: Mapping[str, Any],
    binding: OpenPIPreparedSkillBinding | OpenPIPreparedPromptBinding,
    binding_root: Path,
) -> dict[Path, tuple[str, Path]]:
    layout = str(profile.get("openpi_final_mask_root_layout") or "").strip()
    if not layout:
        return {}
    if layout != "source_dataset_id_partitioned_v1":
        raise ValueError(
            "openpi_final_mask_root_layout must be "
            "source_dataset_id_partitioned_v1 when declared."
        )
    configured_root = str(profile.get("openpi_final_mask_root") or "").strip()
    if not configured_root:
        raise ValueError(
            "source-partitioned final masks require openpi_final_mask_root."
        )
    base_root = Path(configured_root).expanduser().resolve()
    return {
        source_root.resolve(): (source_id, (base_root / source_id).resolve())
        for source_id, source_root, _episode_manifest in binding_source_manifests(
            binding,
            binding_root,
        )
    }


def _final_mask_training_contract(
    profile: Mapping[str, Any],
    binding: OpenPIPreparedSkillBinding | OpenPIPreparedPromptBinding,
    binding_root: Path,
) -> dict[str, Any] | None:
    roots = _source_partitioned_final_mask_roots(
        profile,
        binding,
        binding_root,
    )
    if not roots:
        return None
    return {
        "final_mask_root_layout": "source_dataset_id_partitioned_v1",
        "final_mask_roots_by_source": {
            source_id: str(final_root)
            for source_id, final_root in roots.values()
        },
        "final_mask_semantics": (
            "postprocessed_depth_keep_mask_or_table_geometry_keep_mask_v1"
        ),
        "wrist_depth_mask_paths_fallback_allowed": False,
    }


def _bind_source_partitioned_final_masks_to_dataset(
    prepared: Any,
    roots: Mapping[Path, tuple[str, Path]],
) -> Any:
    raw_dataset = getattr(prepared, "dataset", prepared)
    children = tuple(getattr(raw_dataset, "_datasets", (raw_dataset,)))
    seen: set[Path] = set()
    for child in children:
        source_root = Path(child.dataset_root).expanduser().resolve()
        binding_value = roots.get(source_root)
        if binding_value is None:
            raise OpenPIPreparedSkillError(
                "OpenPI mixed child is absent from the source-partitioned "
                f"final-mask binding: {source_root}."
            )
        if getattr(child, "_masked_pipeline", None) is not None:
            raise OpenPIPreparedSkillError(
                "OpenPI final-mask root was bound after the masked pipeline "
                f"was opened: {source_root}."
            )
        explicit_mask_keys = tuple(
            str(value) for value in getattr(child, "explicit_mask_keys", ())
        )
        if "wrist_depth_mask_paths" in explicit_mask_keys:
            raise OpenPIPreparedSkillError(
                "Corrected Stage1 training forbids the raw "
                "wrist_depth_mask_paths fallback."
            )
        _source_id, final_root = binding_value
        child.final_mask_root = final_root
        seen.add(source_root)
    missing = sorted(str(path) for path in set(roots) - seen)
    if missing:
        raise OpenPIPreparedSkillError(
            "OpenPI did not construct every source in the final-mask binding: "
            f"{missing}."
        )
    return prepared


@contextmanager
def _source_partitioned_final_mask_binding(
    profile: Mapping[str, Any],
    binding: OpenPIPreparedSkillBinding | OpenPIPreparedPromptBinding,
    binding_root: Path,
) -> Iterator[None]:
    """Bind every mixed child to a source-ID-partitioned final-mask root.

    OpenPI v2.3 currently accepts one global ``final_mask_root`` when it
    constructs a mixed dataset.  That path is ambiguous when two independent
    sources share the same dataset basename.  Patch only the already-installed
    Prepared dataset factory and assign each lazy child before any image or
    mask pipeline is opened.
    """

    roots = _source_partitioned_final_mask_roots(
        profile,
        binding,
        binding_root,
    )
    if not roots:
        yield
        return
    from openpi.training import data_loader

    original_create = data_loader.create_torch_dataset

    def create_with_partitioned_final_masks(*args: Any, **kwargs: Any) -> Any:
        prepared = original_create(*args, **kwargs)
        return _bind_source_partitioned_final_masks_to_dataset(prepared, roots)

    data_loader.create_torch_dataset = create_with_partitioned_final_masks
    try:
        yield
    finally:
        data_loader.create_torch_dataset = original_create


def _correct_training_input_contract(
    config: Any,
    metadata: dict[str, Any],
    binding: OpenPIPreparedSkillBinding | OpenPIPreparedPromptBinding,
    *,
    state_only_bridge: OpenPIStateOnlyInputBridge | None = None,
    final_mask_contract: Mapping[str, Any] | None = None,
) -> None:
    if "training_input_contract" not in metadata:
        return
    current = metadata["training_input_contract"]
    if not isinstance(current, Mapping):
        raise ValueError(
            "OpenPI corrected training input contract must be a mapping."
        )
    contract = copy.deepcopy(dict(current))
    provider = str(getattr(config.data, "background_provider", "unknown"))
    epoch_mode = str(getattr(config, "epoch_worker_mode", "unknown"))
    contract["profile"] = (
        f"numeric_state_only_{binding.source_mixing_policy}_{epoch_mode}_v1"
        if state_only_bridge is not None
        else (
            f"final_keep_mask_{provider}_{binding.source_mixing_policy}_"
            f"{epoch_mode}_media_strict_v1"
        )
    )
    contract["source_mixing_policy"] = binding.source_mixing_policy
    if isinstance(binding, OpenPIPreparedPromptBinding):
        contract["prompt_stage_sampling_policy"] = (
            binding.prompt_stage_sampling_policy
        )
    contract["num_workers"] = int(config.num_workers)
    if state_only_bridge is not None:
        state_only_contract = state_only_bridge.contract
        contract["observation_mode"] = "state_only"
        contract["external_camera_required"] = False
        contract["source_media_decoded"] = False
        contract["state_only_bridge_contract_hash"] = state_only_contract[
            "contract_hash"
        ]
    elif final_mask_contract is not None:
        contract.update(copy.deepcopy(dict(final_mask_contract)))
    contract.pop("contract_hash", None)
    canonical = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    contract["contract_hash"] = (
        "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
    )
    _bind_metadata_value(metadata, "training_input_contract", contract)


@contextmanager
def _prepared_metadata_binding(
    training_config: Any,
    binding: OpenPIPreparedSkillBinding | OpenPIPreparedPromptBinding,
    *,
    image_wire_training_contract: Mapping[str, Any] | None = None,
    state_only_bridge: OpenPIStateOnlyInputBridge | None = None,
    final_mask_contract: Mapping[str, Any] | None = None,
) -> Iterator[None]:
    original = training_config.resolve_policy_metadata

    def resolve(config: Any) -> Any:
        resolved = original(config)
        metadata = copy.deepcopy(resolved.policy_metadata or {})
        prepared = binding.to_payload()
        prepared_key = (
            "prepared_prompt_training_binding"
            if isinstance(binding, OpenPIPreparedPromptBinding)
            else "prepared_skill_training_binding"
        )
        metadata[prepared_key] = prepared
        mixing = _natural_mixing_metadata(resolved, binding)
        _bind_metadata_value(metadata, "dataset_mixing", mixing)
        _correct_training_input_contract(
            resolved,
            metadata,
            binding,
            state_only_bridge=state_only_bridge,
            final_mask_contract=final_mask_contract,
        )
        if image_wire_training_contract is not None:
            _bind_metadata_value(
                metadata,
                OPENPI_IMAGE_WIRE_TRAINING_METADATA_KEY_V1,
                image_wire_training_contract,
            )
        if state_only_bridge is not None:
            metadata = bind_state_only_policy_metadata(
                metadata,
                state_only_bridge,
            )
            contract = copy.deepcopy(state_only_bridge.contract)
            _bind_metadata_value(
                metadata,
                OPENPI_STATE_ONLY_CAPABILITY_KEY_V1,
                contract,
            )
        artifact = metadata.get("policy_artifact")
        if isinstance(artifact, dict):
            artifact[prepared_key] = prepared
        return dataclasses.replace(resolved, policy_metadata=metadata)

    training_config.resolve_policy_metadata = resolve
    try:
        yield
    finally:
        training_config.resolve_policy_metadata = original


@contextmanager
def _resolved_policy_metadata_binding(
    training_config: Any,
    resolved_config: Any,
) -> Iterator[None]:
    """Keep one already-resolved model-RTC config stable inside OpenPI train.

    OpenPI's train entrypoint resolves policy metadata once more.  The
    Prepared model-RTC adapter has already resolved the base checkpoint and
    then bound its artifact and training-run receipt metadata, so a second
    resolution would replace those additive bindings with the base artifact.
    """

    original = training_config.resolve_policy_metadata

    def resolve(config: Any) -> Any:
        if config is resolved_config:
            return resolved_config
        return original(config)

    training_config.resolve_policy_metadata = resolve
    try:
        yield
    finally:
        training_config.resolve_policy_metadata = original


@contextmanager
def _openpi_model_rtc_receipt_import_binding() -> Iterator[None]:
    """Keep OpenPI's receipt writer independent of robot runtime packages.

    The current OpenPI receipt hook imports ``openpi_rtc_artifact_hash`` from
    the deployment runtime module even though the function is a pure artifact
    identity.  In the isolated OpenPI training environment that runtime import
    would unnecessarily require robot-control dependencies such as OSQP.
    Supply the already-owned pure identity only for this training scope.
    """

    module_name = "deployments.openpi.client.rtc"
    existing = sys.modules.get(module_name)
    if existing is not None:
        if not callable(getattr(existing, "openpi_rtc_artifact_hash", None)):
            raise RuntimeError(
                "Loaded OpenPI RTC module has no artifact hash function."
            )
        yield
        return

    from deployments.openpi.artifact_identity import openpi_rtc_artifact_hash

    proxy = types.ModuleType(module_name)
    proxy.openpi_rtc_artifact_hash = openpi_rtc_artifact_hash
    proxy.__all__ = ("openpi_rtc_artifact_hash",)
    sys.modules[module_name] = proxy
    try:
        yield
    finally:
        if sys.modules.get(module_name) is proxy:
            sys.modules.pop(module_name, None)


@contextmanager
def _openpi_cwd(checkout: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(checkout)
    try:
        yield
    finally:
        os.chdir(previous)


def _openpi_checkpoint_dir(config: Any, checkout: Path) -> Path:
    """Resolve OpenPI's cwd-relative checkpoint directory in its checkout."""

    with _openpi_cwd(checkout):
        return Path(config.checkpoint_dir)


def _openpi_checkpoint_base_root(config: Any, checkout: Path) -> Path:
    """Resolve OpenPI's cwd-relative checkpoint base in its checkout."""

    with _openpi_cwd(checkout):
        return Path(str(config.checkpoint_base_dir)).expanduser().resolve()


def _openpi_data_assets_root(config: Any, checkout: Path) -> Path:
    """Resolve OpenPI's effective data-asset root in its checkout."""

    with _openpi_cwd(checkout):
        configured = getattr(config.data.assets, "assets_dir", None)
        return Path(str(configured or config.assets_dirs)).expanduser().resolve()


def _sha256_digest(value: object, *, name: str) -> str:
    normalized = str(value).strip().lower()
    if normalized.startswith("sha256:"):
        normalized = normalized[7:]
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a canonical SHA-256 identity.")
    return normalized


def _model_rtc_checkpoint_root(
    profile: Mapping[str, Any],
    model_rtc_profile: Mapping[str, Any],
    *,
    exp_name: str | None,
) -> Path:
    """Resolve the model-RTC run root without compiling a live data view."""

    checkout = Path(str(profile["openpi_checkout"])).expanduser().resolve()
    training_config = _activate_openpi_checkout(profile)

    base = training_config.get_config(str(profile["openpi_base_config"]))
    effective_exp_name = str(
        exp_name or model_rtc_profile["default_exp_name"]
    )
    return (
        _openpi_checkpoint_base_root(base, checkout)
        / str(model_rtc_profile["output_config_name"])
        / effective_exp_name
    ).resolve()


def _load_checkpoint_bound_prepared_resume(
    profile: Mapping[str, Any],
    model_rtc_profile: Mapping[str, Any],
    *,
    checkpoint_root: str | Path,
    verify_source_files: bool,
) -> _CheckpointBoundPreparedResume:
    """Recover the exact immutable Prepared view named by a checkpoint.

    A resume must not repeat eligibility selection against mutable annotation
    directories.  The checkpoint receipt names the aggregate Prepared binding,
    whose manifest in turn names the immutable training view.  Every identity
    is verified before that view can be used.
    """

    checkpoint_policy = _resolve_train_checkpoint_policy(
        checkpoint_root,
        resume=True,
        overwrite=False,
    )
    checkpoint_step = checkpoint_policy.latest_checkpoint_step
    assert checkpoint_step is not None
    evidence = verify_openpi_model_rtc_checkpoint(
        checkpoint_policy.checkpoint_dir / str(checkpoint_step)
    )
    expected_model_id = str(model_rtc_profile["output_config_name"])
    if evidence.training_run.output_model_id != expected_model_id:
        raise ValueError(
            "Checkpoint output model differs from the model-RTC profile: "
            f"{evidence.training_run.output_model_id!r} != "
            f"{expected_model_id!r}."
        )
    source_bindings = evidence.training_run.source_bindings
    if len(source_bindings) != 1:
        raise ValueError(
            "Prepared model-RTC resume requires exactly one aggregate "
            "checkpoint source binding."
        )
    source_binding = source_bindings[0]
    expected_source_id = str(model_rtc_profile["profile_id"])
    if source_binding.source_id != expected_source_id:
        raise ValueError(
            "Checkpoint source binding differs from the model-RTC profile: "
            f"{source_binding.source_id!r} != {expected_source_id!r}."
        )

    binding_hash = _sha256_digest(
        source_binding.dataset_identity_hash,
        name="checkpoint dataset_identity_hash",
    )
    organized_root = Path(
        str(profile["organized_dataset_root"])
    ).expanduser().resolve()
    binding_root = (
        organized_root
        / "derived/openpi_prepared_skill_bindings"
        / binding_hash
    ).resolve()
    manifest_path = binding_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Checkpoint-bound Prepared Skill manifest is missing: "
            f"{manifest_path}."
        )
    manifest_bytes = manifest_path.read_bytes()
    actual_manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    expected_manifest_hash = _sha256_digest(
        source_binding.episode_manifest_hash,
        name="checkpoint episode_manifest_hash",
    )
    if actual_manifest_hash != expected_manifest_hash:
        raise ValueError(
            "Checkpoint-bound Prepared Skill manifest bytes differ from the "
            "checkpoint receipt."
        )
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, Mapping):
        raise ValueError("Prepared Skill binding manifest must be an object.")
    binding_payload = manifest.get("binding")
    if not isinstance(binding_payload, Mapping):
        raise ValueError("Prepared Skill binding manifest has no binding object.")
    declared_binding_hash = _sha256_digest(
        binding_payload.get("binding_hash", ""),
        name="Prepared Skill binding_hash",
    )
    if declared_binding_hash != binding_hash:
        raise ValueError(
            "Prepared Skill binding manifest identity differs from the "
            "checkpoint receipt."
        )
    training_view_hash = _sha256_digest(
        binding_payload.get("training_view_hash", ""),
        name="Prepared training_view_hash",
    )
    view_root = (
        organized_root / "derived/training_views" / training_view_hash
    ).resolve()
    binding = load_openpi_prepared_skill_binding(
        view_root,
        skill_id=str(profile["target_skill_id"]),
        training_domain=str(profile.get("training_domain", "NOMINAL")),
        camera_slot_to_dataset_key=dict(profile["camera_slot_to_dataset_key"]),
        action_horizon=int(profile["action_horizon"]),
        action_pose_mode=str(profile["action_pose_mode"]),
        source_mixing_policy=str(profile["source_mixing_policy"]),
        organized_dataset_root=organized_root,
        verify_source_files=verify_source_files,
    )
    if binding.binding_hash != binding_hash:
        raise ValueError(
            "Checkpoint-bound Prepared view reconstructs a different binding."
        )
    if binding.to_payload() != dict(binding_payload):
        raise ValueError(
            "Checkpoint-bound Prepared view differs from its binding manifest."
        )

    expected_sources = {
        str(item["source_dataset_id"]): Path(
            str(item["dataset_root"])
        ).expanduser().resolve()
        for item in profile["sources"]
    }
    actual_roots: dict[str, set[Path]] = {}
    for episode in binding.episodes:
        source_id = episode.source_dataset_id
        actual_roots.setdefault(source_id, set()).add(
            episode.source_dataset_root.resolve()
        )
    if set(actual_roots) != set(expected_sources):
        raise ValueError(
            "Checkpoint-bound Prepared source ids differ from the profile: "
            f"actual={sorted(actual_roots)} "
            f"expected={sorted(expected_sources)}."
        )
    for source_id, expected_root in expected_sources.items():
        roots = actual_roots.get(source_id, set())
        if roots != {expected_root}:
            raise ValueError(
                "Checkpoint-bound Prepared source root differs from the "
                f"profile for {source_id!r}: actual={sorted(map(str, roots))} "
                f"expected={expected_root}."
            )
    return _CheckpointBoundPreparedResume(
        binding=binding,
        binding_root=binding_root,
        checkpoint_policy=checkpoint_policy,
        evidence=evidence,
    )


def _validate_checkpoint_training_run(
    config: Any,
    resume: _CheckpointBoundPreparedResume,
) -> None:
    metadata = getattr(config, "policy_metadata", None)
    if not isinstance(metadata, Mapping):
        raise ValueError("Model-RTC resume config has no policy metadata.")
    actual = metadata.get(MODEL_CONTINUATION_TRAINING_RUN_METADATA_KEY_V1)
    expected = resume.evidence.training_run.to_dict()
    if actual != expected:
        raise ValueError(
            "Reconstructed model-RTC training run differs from the latest "
            "checkpoint receipt; refusing to restore optimizer state."
        )


def _print_summary(
    report: Any,
    binding: OpenPIPreparedSkillBinding | OpenPIPreparedPromptBinding | None,
    binding_root: Path | None,
) -> None:
    if isinstance(report, tuple):
        stages = []
        prompt_skills = (
            ()
            if not isinstance(binding, OpenPIPreparedPromptBinding)
            else binding.prompt_skills
        )
        for index, stage_report in enumerate(report):
            child_report = stage_report.report
            reasons: dict[str, int] = {}
            for item in child_report.exclusions:
                reasons[item.reason] = reasons.get(item.reason, 0) + 1
            prompt_skill = (
                prompt_skills[index] if index < len(prompt_skills) else None
            )
            stages.append(
                {
                    "stage_id": (
                        stage_report.stage_id
                    ),
                    "skill_id": (
                        stage_report.skill_id
                    ),
                    "prompt": (
                        stage_report.prompt
                    ),
                    "eligible_episode_count": len(
                        child_report.included_episode_identities
                    ),
                    "excluded_episode_count": len(child_report.exclusions),
                    "exclusion_reasons": dict(sorted(reasons.items())),
                    "eligibility_report_hash": (
                        child_report.semantic_content_hash
                    ),
                    "window_count": (
                        stage_report.expected_window_count
                        if prompt_skill is None
                        else prompt_skill.window_count
                    ),
                    "expected_openpi_window_count": (
                        None
                        if prompt_skill is None
                        else prompt_skill.expected_openpi_window_count
                    ),
                }
            )
        summary = {
            "prompt_stages": stages,
            "eligible_episode_union_count": (
                None if binding is None else len(binding.episodes)
            ),
            "training_view_hash": (
                None if binding is None else binding.training_view_hash
            ),
            "openpi_binding_hash": (
                None if binding is None else binding.binding_hash
            ),
            "openpi_binding_root": (
                None if binding_root is None else str(binding_root)
            ),
            "window_count": (
                None if binding is None else binding.to_payload()["window_count"]
            ),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    reasons: dict[str, int] = {}
    for item in report.exclusions:
        reasons[item.reason] = reasons.get(item.reason, 0) + 1
    summary = {
        "eligible_episode_count": len(report.included_episode_identities),
        "excluded_episode_count": len(report.exclusions),
        "exclusion_reasons": dict(sorted(reasons.items())),
        "eligibility_report_hash": report.semantic_content_hash,
        "training_view_hash": None if binding is None else binding.training_view_hash,
        "openpi_binding_hash": None if binding is None else binding.binding_hash,
        "openpi_binding_root": None if binding_root is None else str(binding_root),
        "window_count": (
            None
            if binding is None
            else binding.to_payload()["window_count"]
        ),
    }
    if binding is not None and isinstance(binding, OpenPIPreparedSkillBinding):
        binding_payload = binding.to_payload()
        if "prepared_interval_repeated_window_count" in binding_payload:
            summary["prepared_interval_repeated_window_count"] = binding_payload[
                "prepared_interval_repeated_window_count"
            ]
            summary["source_repeat_weights"] = dict(
                binding.source_repeat_weights
            )
    if hasattr(report, "source_statistics"):
        summary["source_statistics"] = list(report.source_statistics)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train remote OpenPI from one immutable Prepared Skill view."
    )
    parser.add_argument(
        "command",
        choices=("audit", "prepare", "sample-check", "norm-stats", "train"),
    )
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--exp-name", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stats-batch-size", type=int, default=256)
    parser.add_argument("--stats-num-workers", type=int, default=0)
    parser.add_argument(
        "--sample-load-images",
        action="store_true",
        help=(
            "For sample-check, decode RGB and enforce the bound final "
            "keep-mask path instead of checking numeric samples only."
        ),
    )
    parser.add_argument("--model-rtc-profile", type=Path, default=None)
    parser.add_argument("--model-rtc-adapter-debug", action="store_true")
    parser.add_argument("--num-train-steps", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--skip-source-file-verification", action="store_true")
    parser.add_argument("--no-motion", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    if args.sample_load_images and args.command != "sample-check":
        parser.error("--sample-load-images is supported only by sample-check")
    if args.model_rtc_profile is None and (
        args.model_rtc_adapter_debug
        or args.num_train_steps is not None
        or args.save_interval is not None
    ):
        parser.error(
            "model-RTC debug/train-step overrides require --model-rtc-profile"
        )
    if args.model_rtc_profile is not None and args.command not in {
        "sample-check",
        "train",
    }:
        parser.error(
            "--model-rtc-profile is supported only by sample-check and train"
        )
    profile = _load_profile(args.profile)
    if (
        profile.get("openpi_training_mode")
        == STANDARD_NON_RTC_TRAINING_MODE_V1
        and args.model_rtc_profile is not None
    ):
        parser.error(
            "standard_non_rtc_v1 profiles reject --model-rtc-profile"
        )
    if (
        profile.get("schema") in {
            PROMPT_PROFILE_SCHEMA_V1,
            PROMPT_FAMILY_PROFILE_SCHEMA_V2,
        }
        and args.model_rtc_profile is not None
    ):
        parser.error(
            "joint prompt training does not yet support --model-rtc-profile"
        )
    model_rtc_profile = (
        None
        if args.model_rtc_profile is None
        else load_prepared_model_rtc_profile(args.model_rtc_profile)
    )
    checkpoint_bound_resume = None
    if args.command == "train" and args.resume and model_rtc_profile is not None:
        checkpoint_bound_resume = _load_checkpoint_bound_prepared_resume(
            profile,
            model_rtc_profile,
            checkpoint_root=_model_rtc_checkpoint_root(
                profile,
                model_rtc_profile,
                exp_name=args.exp_name,
            ),
            verify_source_files=not args.skip_source_file_verification,
        )
        binding = checkpoint_bound_resume.binding
        binding_root = checkpoint_bound_resume.binding_root
        print(
            json.dumps(
                {
                    "prepared_binding_source": "checkpoint_receipt",
                    "eligible_episode_count": len(binding.episodes),
                    "training_view_hash": binding.training_view_hash,
                    "openpi_binding_hash": binding.binding_hash,
                    "openpi_binding_root": str(binding_root),
                    "checkpoint_dir": str(
                        checkpoint_bound_resume.checkpoint_policy.checkpoint_dir
                    ),
                    "latest_checkpoint_step": (
                        checkpoint_bound_resume.checkpoint_policy.latest_checkpoint_step
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        report, binding, binding_root = prepare_profile(
            profile,
            dry_run=args.command == "audit",
            verify_source_files=not args.skip_source_file_verification,
        )
        _print_summary(report, binding, binding_root)
    if args.command in {"audit", "prepare"}:
        return 0
    assert binding is not None and binding_root is not None
    config, training_config, checkout = _openpi_config(
        profile,
        binding,
        binding_root,
        exp_name=args.exp_name,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    final_mask_contract = _final_mask_training_contract(
        profile,
        binding,
        binding_root,
    )
    state_only_bridge = OpenPIStateOnlyInputBridge.from_profile_value(
        profile.get("openpi_observation_mode"),
        image_keys=tuple(str(value) for value in config.model.image_keys),
    )
    image_wire_training_transform = (
        OpenPIImageWireTrainingTransform.from_profile_value(
            profile.get("openpi_image_wire_training"),
            image_keys=tuple(str(value) for value in config.model.image_keys),
        )
    )
    image_wire_training_contract = (
        None
        if image_wire_training_transform is None
        else image_wire_training_transform.contract
    )
    if state_only_bridge is not None and image_wire_training_transform is not None:
        raise ValueError(
            "State-only OpenPI training cannot also enable an image-wire profile."
        )
    if state_only_bridge is not None and model_rtc_profile is not None:
        raise ValueError(
            "The state-only profile is an ordinary OpenPI policy, not model-RTC."
        )
    model_rtc_transform = None
    if model_rtc_profile is not None:
        with _prepared_metadata_binding(
            training_config,
            binding,
            image_wire_training_contract=image_wire_training_contract,
            final_mask_contract=final_mask_contract,
        ):
            prepared_base = training_config.resolve_policy_metadata(config)
        adapted = adapt_prepared_skill_model_rtc_config(
            base_config=prepared_base,
            prepared_profile=profile,
            model_rtc_profile=model_rtc_profile,
            binding=binding,
            binding_root=binding_root,
            source_assets_root=_openpi_data_assets_root(
                prepared_base,
                checkout,
            ),
            checkpoint_base_root=_openpi_checkpoint_base_root(
                prepared_base,
                checkout,
            ),
            exp_name=args.exp_name,
            # Resume is a launch lifecycle decision, not a new training-run
            # identity. Reconstruct the original manifest first; the verified
            # checkpoint policy below enables restoration on the final config.
            resume=(False if checkpoint_bound_resume is not None else args.resume),
            overwrite=args.overwrite,
            num_train_steps=args.num_train_steps,
            save_interval=args.save_interval,
            adapter_debug=args.model_rtc_adapter_debug,
        )
        config = adapted.config
        model_rtc_transform = adapted.transform
        if checkpoint_bound_resume is not None:
            _validate_checkpoint_training_run(config, checkpoint_bound_resume)
        print(json.dumps(adapted.summary, indent=2, sort_keys=True))
    post_identity_transform = (
        state_only_bridge
        if state_only_bridge is not None
        else model_rtc_transform
    )
    if image_wire_training_transform is not None:
        post_identity_transform = image_wire_training_transform.with_upstream(
            model_rtc_transform
        )
    if args.command == "sample-check":
        from collections import Counter

        from openpi.training import data_loader

        state_only_metadata_bound = False
        if state_only_bridge is not None:
            with _openpi_cwd(checkout), _prepared_metadata_binding(
                training_config,
                binding,
                state_only_bridge=state_only_bridge,
                final_mask_contract=final_mask_contract,
            ):
                metadata_checked_config = (
                    training_config.resolve_policy_metadata(config)
                )
            metadata = metadata_checked_config.policy_metadata or {}
            artifact = metadata.get("policy_artifact")
            expected_contract = state_only_bridge.contract
            if (
                metadata.get(OPENPI_STATE_ONLY_CAPABILITY_KEY_V1)
                != expected_contract
                or not isinstance(artifact, Mapping)
                or artifact.get(OPENPI_STATE_ONLY_CAPABILITY_KEY_V1)
                != expected_contract
            ):
                raise OpenPIPreparedSkillError(
                    "State-only OpenPI policy metadata binding is incomplete."
                )
            state_only_metadata_bound = True

        with _openpi_cwd(checkout), patch_openpi_prepared_skill_dataset(
            binding,
            post_identity_transform=post_identity_transform,
            load_images=False if state_only_bridge is not None else None,
        ), _source_partitioned_final_mask_binding(
            profile,
            binding,
            binding_root,
        ):
            data_config = config.data.create(config.assets_dirs, config.model)
            prompt_token_preflight = _prompt_family_token_preflight(
                profile,
                data_config,
                config.model,
            )
            dataset = data_loader.create_torch_dataset(
                data_config,
                config.model.action_horizon,
                config.model,
                load_images=bool(args.sample_load_images),
            )
            if state_only_bridge is not None and data_config.norm_stats is None:
                raise OpenPIPreparedSkillError(
                    "State-only OpenPI sample check did not load normalization "
                    "stats."
                )
            if len(dataset) <= 0:
                raise OpenPIPreparedSkillError(
                    "OpenPI prepared dataset contains no samples."
                )
            for index in {0, len(dataset) // 2, len(dataset) - 1}:
                dataset[index]
            prompt_stage_samples = None
            if isinstance(binding, OpenPIPreparedPromptBinding):
                prompt_stage_samples = {}
                if binding.prompt_family_sampling_policy is None:
                    index_getter = getattr(
                        dataset,
                        "prepared_prompt_stage_indices",
                        None,
                    )
                    if not callable(index_getter):
                        raise OpenPIPreparedSkillError(
                            "Prompt sample-check requires stage sample indices."
                        )
                    stage_indices = index_getter()
                    prompt_indices = {
                        skill.prompt: int(stage_indices[skill.stage_id])
                        for skill in binding.prompt_skills
                    }
                else:
                    index_getter = getattr(
                        dataset,
                        "prepared_prompt_family_indices",
                        None,
                    )
                    if not callable(index_getter):
                        raise OpenPIPreparedSkillError(
                            "Prompt-family sample-check requires family indices."
                        )
                    prompt_indices = index_getter()
                    expected_training_prompts = {
                        prompt
                        for skill in binding.prompt_skills
                        for prompt in (
                            skill.prompt,
                            *skill.training_prompt_variants,
                        )
                    }
                    held_out_prompts = {
                        prompt
                        for skill in binding.prompt_skills
                        for prompt in skill.held_out_prompt_variants
                    }
                    if set(prompt_indices) != expected_training_prompts:
                        raise OpenPIPreparedSkillError(
                            "Prompt-family sample-check did not expose every and "
                            "only training-time prompt."
                        )
                    if set(prompt_indices) & held_out_prompts:
                        raise OpenPIPreparedSkillError(
                            "Held-out prompt entered the training dataset."
                        )
                for prompt_skill in binding.prompt_skills:
                    stage_prompts = (
                        prompt_skill.prompt,
                        *prompt_skill.training_prompt_variants,
                    )
                    verified_prompts: list[dict[str, Any]] = []
                    for expected_prompt in stage_prompts:
                        sample_index = int(prompt_indices[expected_prompt])
                        sample = dataset[sample_index]
                        actual_prompt = str(sample.get("prompt", ""))
                        if actual_prompt != expected_prompt:
                            raise OpenPIPreparedSkillError(
                                "OpenPI prepared sample prompt differs from the "
                                f"binding for {prompt_skill.stage_id!r}."
                            )
                        verified_prompts.append(
                            {
                                "sample_index": sample_index,
                                "prompt": actual_prompt,
                            }
                        )
                    prompt_stage_samples[prompt_skill.stage_id] = {
                        "skill_id": prompt_skill.skill_id,
                        "verified_training_prompts": verified_prompts,
                        "held_out_prompt_count": len(
                            prompt_skill.held_out_prompt_variants
                        ),
                    }
            mix_group_size = sum(
                int(source.sample_weight)
                for source in data_config.dexteleop_v23_dataset_sources
            )
            sampler = data_loader._DexTeleopV22ExactMixSampler(
                len(dataset), mix_group_size, config.seed
            )
            sampled_indices = tuple(sampler)
            if (
                len(sampled_indices) != len(dataset)
                or len(set(sampled_indices)) != len(dataset)
                or set(sampled_indices) != set(range(len(dataset)))
            ):
                raise OpenPIPreparedSkillError(
                    "OpenPI natural-merge sampler repeated or omitted samples."
                )
            image_wire_selection = None
            if image_wire_training_transform is not None:
                identity_getter = getattr(
                    dataset,
                    "prepared_sample_identity",
                    None,
                )
                if not callable(identity_getter):
                    raise OpenPIPreparedSkillError(
                        "Image-wire sample-check requires prepared sample identity."
                    )
                image_wire_selection = (
                    image_wire_training_transform.selection_summary(
                        (
                            identity_getter(index)
                            for index in sampled_indices
                        ),
                        epoch=0,
                    )
                )
            prepared_windows = Counter()
            prepared_windows_by_stage = Counter()
            root_to_source_id: dict[Path, str] = {}
            for episode in binding.episodes:
                root_to_source_id[episode.source_dataset_root] = (
                    episode.source_dataset_id
                )
                for interval in episode.intervals:
                    if hasattr(interval, "sample_begin"):
                        begin = int(interval.sample_begin)
                        end = int(interval.sample_end_exclusive)
                        stage_id = str(interval.stage_id)
                    else:
                        begin, end = interval
                        stage_id = "single_skill"
                    count = end - begin - binding.action_horizon + 1
                    prepared_windows[episode.source_dataset_id] += count
                    prepared_windows_by_stage[stage_id] += count
            concrete = dataset.dataset
            children = tuple(getattr(concrete, "_datasets", ()))
            quality_valid_windows: dict[str, int] = {}
            for child in children or (concrete,):
                source_root = Path(child.dataset_root).expanduser().resolve()
                source_id = root_to_source_id[source_root]
                quality_valid_windows[source_id] = len(child)
            mixed_epoch_samples = Counter()
            for source_position, _child_index in tuple(
                getattr(concrete, "_mix_index", ())
            ):
                child = children[int(source_position)]
                source_root = Path(child.dataset_root).expanduser().resolve()
                mixed_epoch_samples[root_to_source_id[source_root]] += 1
            if isinstance(binding, OpenPIPreparedSkillBinding):
                expected_mixed_epoch_samples = {
                    source_id: count * binding.repeat_weight_for(source_id)
                    for source_id, count in quality_valid_windows.items()
                }
                if dict(mixed_epoch_samples) != expected_mixed_epoch_samples:
                    raise OpenPIPreparedSkillError(
                        "OpenPI source repetition differs from the binding: "
                        f"actual={dict(mixed_epoch_samples)} "
                        f"expected={expected_mixed_epoch_samples}."
                    )
                if profile.get("schema") == STAGE1_SCENARIO_PROFILE_SCHEMA_V1:
                    expected_quality_by_source = {
                        str(item["source_dataset_id"]): int(
                            item[
                                "expected_openpi_quality_valid_window_count"
                            ]
                        )
                        for item in profile["sources"]
                    }
                    if quality_valid_windows != expected_quality_by_source:
                        raise OpenPIPreparedSkillError(
                            "OpenPI Stage1 quality-valid source counts differ "
                            "from the profile: "
                            f"actual={quality_valid_windows} "
                            f"expected={expected_quality_by_source}."
                        )
                    if (
                        sum(quality_valid_windows.values())
                        != int(
                            profile[
                                "expected_openpi_quality_valid_window_count"
                            ]
                        )
                        or sum(mixed_epoch_samples.values())
                        != int(
                            profile[
                                "expected_openpi_repeated_epoch_sample_count"
                            ]
                        )
                    ):
                        raise OpenPIPreparedSkillError(
                            "OpenPI Stage1 total sample counts differ from "
                            "the profile."
                        )
            openpi_source_mixing = dataset.mixing_metadata()
            if (
                isinstance(binding, OpenPIPreparedPromptBinding)
                and binding.prompt_family_sampling_policy is not None
            ):
                training_prompt_counts = dict(
                    openpi_source_mixing.get("training_prompt_counts", {})
                )
                target = effective_prompt_stage_window_count(binding)
                for skill in binding.prompt_skills:
                    canonical_count = training_prompt_counts.get(skill.prompt, 0)
                    variant_counts = tuple(
                        training_prompt_counts.get(prompt, 0)
                        for prompt in skill.training_prompt_variants
                    )
                    if (
                        canonical_count != target * 4 // 5
                        or sum(variant_counts) != target // 5
                        or max(variant_counts) - min(variant_counts) > 1
                        or any(
                            prompt in training_prompt_counts
                            for prompt in skill.held_out_prompt_variants
                        )
                    ):
                        raise OpenPIPreparedSkillError(
                            "Prompt-family sample counts violate the exact 80/20 "
                            "quota or held-out isolation."
                        )
            sample_check_summary = {
                "prepared_interval_windows_by_source": dict(
                    sorted(prepared_windows.items())
                ),
                "prepared_interval_windows_by_stage": dict(
                    sorted(prepared_windows_by_stage.items())
                ),
                "openpi_quality_valid_windows_by_source": dict(
                    sorted(quality_valid_windows.items())
                ),
                "openpi_mixed_epoch_samples_by_source": dict(
                    sorted(mixed_epoch_samples.items())
                ),
                "openpi_source_mixing": openpi_source_mixing,
                "openpi_sampler_unique_sample_count": len(sampled_indices),
                "openpi_dataset_length": len(dataset),
                "observation_mode": (
                    "state_only"
                    if state_only_bridge is not None
                    else "configured_cameras"
                ),
                "source_media_decoded": bool(args.sample_load_images),
                "norm_stats_loaded": data_config.norm_stats is not None,
                "norm_stats_asset_id": str(config.data.assets.asset_id),
                "norm_stats_assets_dir": str(config.data.assets.assets_dir),
                "state_only_policy_metadata_bound": state_only_metadata_bound,
            }
            if image_wire_selection is not None:
                sample_check_summary[
                    "openpi_image_wire_training_selection"
                ] = image_wire_selection
            if prompt_stage_samples is not None:
                sample_check_summary[
                    "openpi_prompt_stage_samples"
                ] = prompt_stage_samples
            if prompt_token_preflight is not None:
                sample_check_summary[
                    "openpi_prompt_token_preflight"
                ] = prompt_token_preflight
            print(json.dumps(sample_check_summary, indent=2))
        return 0
    if args.command == "train":
        checkpoint_policy = _resolve_train_checkpoint_policy(
            _openpi_checkpoint_dir(config, checkout),
            resume=args.resume,
            overwrite=args.overwrite,
        )
        if (
            checkpoint_bound_resume is not None
            and checkpoint_policy != checkpoint_bound_resume.checkpoint_policy
        ):
            raise RuntimeError(
                "Model-RTC checkpoint changed during resume preflight."
            )
        config = dataclasses.replace(
            config,
            resume=checkpoint_policy.resume,
            overwrite=checkpoint_policy.overwrite,
        )
        print(
            json.dumps(
                {
                    "checkpoint_launch_mode": checkpoint_policy.mode,
                    "checkpoint_dir": str(checkpoint_policy.checkpoint_dir),
                    "latest_checkpoint_step": (
                        checkpoint_policy.latest_checkpoint_step
                    ),
                    "effective_resume": checkpoint_policy.resume,
                    "effective_overwrite": checkpoint_policy.overwrite,
                },
                indent=2,
                sort_keys=True,
            )
        )
    with _openpi_cwd(checkout), patch_openpi_prepared_skill_dataset(
        binding,
        post_identity_transform=post_identity_transform,
        load_images=False if state_only_bridge is not None else None,
    ), _source_partitioned_final_mask_binding(
        profile,
        binding,
        binding_root,
    ):
        if args.command == "norm-stats":
            compute = _load_module(
                checkout / "scripts/compute_norm_stats.py",
                "_dexteleop_prepared_compute_norm_stats",
            )
            registry = training_config._CONFIGS_DICT
            previous = registry.get(config.name)
            registry[config.name] = config
            try:
                compute.main(
                    config.name,
                    stats_batch_size=args.stats_batch_size,
                    stats_num_workers=args.stats_num_workers,
                )
            finally:
                if previous is None:
                    registry.pop(config.name, None)
                else:
                    registry[config.name] = previous
            return 0
        train = _load_module(
            checkout / "scripts/train.py",
            "_dexteleop_prepared_train",
        )
        metadata_binding = (
            _resolved_policy_metadata_binding(training_config, config)
            if model_rtc_transform is not None
            else _prepared_metadata_binding(
                training_config,
                binding,
                image_wire_training_contract=image_wire_training_contract,
                state_only_bridge=state_only_bridge,
                final_mask_contract=final_mask_contract,
            )
        )
        receipt_import_binding = (
            _openpi_model_rtc_receipt_import_binding()
            if model_rtc_transform is not None
            else nullcontext()
        )
        with metadata_binding, receipt_import_binding:
            train.main(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
