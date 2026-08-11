"""Prepare, audit, normalize, and train one immutable Prepared Skill view."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
from typing import Any, Iterator, Mapping

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
    OpenPIPreparedPromptBinding,
    OpenPIPreparedSkillBinding,
    OpenPIPreparedSkillError,
    binding_source_manifests,
    compose_openpi_prepared_prompt_binding,
    load_openpi_prepared_prompt_binding,
    load_openpi_prepared_skill_binding_manifest,
    load_openpi_prepared_skill_binding,
    materialize_openpi_prepared_skill_binding,
    patch_openpi_prepared_skill_dataset,
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
DEFAULT_PROFILE = (
    Path(__file__).resolve().parents[1]
    / "configs/training/writing_stage1_0804_0805_front_h16_absolute_v1.json"
)
SKILL_PROJECTION_CATALOG_WRITING_STAGE1_V2 = "writing_stage1_v2"
SKILL_PROJECTION_CATALOG_WRITING_FOUR_STAGE_SUBTASKS_V2 = (
    "writing_four_stage_subtasks_v2"
)
_EMPTY_CHECKPOINT_ROOT_ENTRIES = frozenset({"wandb", "wandb_id.txt"})


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
    if payload.get("schema") == PROFILE_SCHEMA_V1:
        required.extend(("sources", "target_skill_id"))
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
    if payload.get("schema") == PROMPT_PROFILE_SCHEMA_V1:
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
            prompt = str(item.get("prompt", "")).strip()
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
    expected = str(profile.get("expected_task_id", "")).strip()
    if not expected:
        return

    mismatches: list[str] = []
    for source in sources:
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
        elif any(task_id != expected for task_id in task_ids):
            mismatches.append(
                f"{source.source_dataset_id}:meta/tasks.jsonl={task_ids!r}"
            )

    if mismatches:
        raise ValueError(
            f"Source task_id differs from expected {expected!r}: "
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
    minimum_stage_windows = min(
        skill.expected_openpi_window_count
        for skill in binding.prompt_skills
    )
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

    if profile.get("schema") == PROMPT_PROFILE_SCHEMA_V1:
        return _prepare_prompt_profile(
            profile,
            dry_run=dry_run,
            verify_source_files=verify_source_files,
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
    if not (checkout / "src/openpi").is_dir():
        raise FileNotFoundError(f"OpenPI checkout is missing: {checkout}.")
    import openpi.training.config as training_config
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
            sample_weight=1,
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
    )
    train_overrides: dict[str, Any] = {}
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
    metadata = {
        "schema": "dexteleop_v23_natural_merge_v1",
        "source_mixing_policy": binding.source_mixing_policy,
        "source_order": [str(source.name) for source in sources],
        "each_quality_valid_window_once_per_epoch": True,
        "shuffle": "deterministic_individual_sample_epoch_shuffle_v1",
        "sources": [source.to_dict() for source in sources],
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
            effective_window_count_per_stage=min(
                skill.expected_openpi_window_count
                for skill in binding.prompt_skills
            ),
        )
    return metadata


def _correct_training_input_contract(
    config: Any,
    metadata: dict[str, Any],
    binding: OpenPIPreparedSkillBinding | OpenPIPreparedPromptBinding,
    *,
    state_only_bridge: OpenPIStateOnlyInputBridge | None = None,
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
    if not (checkout / "src/openpi").is_dir():
        raise FileNotFoundError(f"OpenPI checkout is missing: {checkout}.")
    import openpi.training.config as training_config

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
    parser.add_argument("--model-rtc-profile", type=Path, default=None)
    parser.add_argument("--model-rtc-adapter-debug", action="store_true")
    parser.add_argument("--num-train-steps", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--skip-source-file-verification", action="store_true")
    parser.add_argument("--no-motion", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
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
        profile.get("schema") == PROMPT_PROFILE_SCHEMA_V1
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
        ):
            data_config = config.data.create(config.assets_dirs, config.model)
            dataset = data_loader.create_torch_dataset(
                data_config,
                config.model.action_horizon,
                config.model,
                load_images=False,
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
                stage_index_getter = getattr(
                    dataset,
                    "prepared_prompt_stage_indices",
                    None,
                )
                if not callable(stage_index_getter):
                    raise OpenPIPreparedSkillError(
                        "Prompt sample-check requires stage sample indices."
                    )
                stage_indices = stage_index_getter()
                prompt_stage_samples = {}
                for prompt_skill in binding.prompt_skills:
                    sample_index = int(stage_indices[prompt_skill.stage_id])
                    sample = dataset[sample_index]
                    actual_prompt = str(sample.get("prompt", ""))
                    if actual_prompt != prompt_skill.prompt:
                        raise OpenPIPreparedSkillError(
                            "OpenPI prepared sample prompt differs from the "
                            f"binding for {prompt_skill.stage_id!r}."
                        )
                    prompt_stage_samples[prompt_skill.stage_id] = {
                        "sample_index": sample_index,
                        "skill_id": prompt_skill.skill_id,
                        "prompt": actual_prompt,
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
                "openpi_source_mixing": dataset.mixing_metadata(),
                "openpi_sampler_unique_sample_count": len(sampled_indices),
                "openpi_dataset_length": len(dataset),
                "observation_mode": (
                    "state_only"
                    if state_only_bridge is not None
                    else "configured_cameras"
                ),
                # sample-check always constructs the dataset with
                # load_images=False above, regardless of observation mode.
                "source_media_decoded": False,
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
