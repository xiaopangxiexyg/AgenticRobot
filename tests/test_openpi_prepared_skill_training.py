from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from dataset_interface.prepared_training.skill_selection import (
    _successful_complete_projection,
)
from dataset_interface.pipeline import (
    CAMERA_SEMANTIC_PREPARED_VIEW_OPT_IN_ENV,
)
from deployments.openpi.training.prepared_skill import (
    OPENPI_PROMPT_FAMILY_SAMPLING_CANONICAL_80_VARIANTS_20_V1,
    OPENPI_SOURCE_MIXING_NATURAL_V1,
    OPENPI_SOURCE_MIXING_REPEAT_BY_SOURCE_V1,
    OPENPI_PROMPT_STAGE_SAMPLING_BALANCED_V1,
    OpenPINaturalMergeSampler,
    OpenPIPreparedEpisodeSelection,
    OpenPIPreparedSkillBinding,
    OpenPIPreparedSkillDataset,
    compose_openpi_prepared_prompt_binding,
    load_openpi_prepared_prompt_binding,
    load_openpi_prepared_skill_binding_manifest,
    materialize_openpi_prepared_skill_binding,
    patch_openpi_prepared_skill_dataset,
)
from deployments.openpi.training.prepared_skill_launch import (
    _CheckpointBoundPreparedResume,
    _TrainCheckpointPolicy,
    _configure_camera_semantic_prepared_view_opt_in,
    _load_module,
    _load_checkpoint_bound_prepared_resume,
    _load_profile,
    _openpi_checkpoint_base_root,
    _openpi_checkpoint_dir,
    _openpi_data_assets_root,
    _openpi_model_rtc_receipt_import_binding,
    _prepared_metadata_binding,
    _prompt_family_token_preflight,
    _resolved_policy_metadata_binding,
    _resolve_train_checkpoint_policy,
    _skill_projection_catalog,
    _sources,
    _validate_source_task_ids,
    _validate_checkpoint_training_run,
    prepare_profile,
)
from deployments.openpi.training.task_metadata import (
    normalize_profile_task_metadata,
)
from deployments.openpi.state_only import (
    OPENPI_STATE_ONLY_CAPABILITY_KEY_V1,
    OpenPIStateOnlyInputBridge,
)


@dataclasses.dataclass(frozen=True)
class _FakeOpenPITrainConfig:
    checkpoint_base_dir: str
    assets_base_dir: str = "./assets"
    name: str = "writing-stage2"
    exp_name: str = "independent-pi05"

    @property
    def checkpoint_dir(self) -> Path:
        return (
            Path(self.checkpoint_base_dir) / self.name / self.exp_name
        ).resolve()

    @property
    def assets_dirs(self) -> Path:
        return (Path(self.assets_base_dir) / self.name).resolve()


def test_prepared_launcher_binds_camera_prepared_view_opt_in_from_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        CAMERA_SEMANTIC_PREPARED_VIEW_OPT_IN_ENV,
        "ambient-value-must-not-win",
    )

    assert _configure_camera_semantic_prepared_view_opt_in(True) is True
    assert os.environ[CAMERA_SEMANTIC_PREPARED_VIEW_OPT_IN_ENV] == "1"

    assert _configure_camera_semantic_prepared_view_opt_in(False) is False
    assert CAMERA_SEMANTIC_PREPARED_VIEW_OPT_IN_ENV not in os.environ

    with pytest.raises(TypeError, match="must be a boolean"):
        _configure_camera_semantic_prepared_view_opt_in("true")  # type: ignore[arg-type]


def test_successful_skill_projection_survives_later_coarse_abort() -> None:
    attempt = types.SimpleNamespace(
        attempt_id="attempt-1",
        progress_step_id="write",
        outcome="ABORTED",
    )
    item = types.SimpleNamespace(
        progress=types.SimpleNamespace(
            timeline=types.SimpleNamespace(attempts=(attempt,))
        )
    )
    occurrences = tuple(
        types.SimpleNamespace(
            coarse_attempt_id="attempt-1",
            progress_step_id="write",
            stage_id=stage_id,
            outcome="SUCCESS",
            occurrence_index_in_attempt=index,
            start_at=float(index),
            annotation_id=f"occurrence-{index}",
        )
        for index, stage_id in enumerate(("approach", "write"))
    )
    annotation = types.SimpleNamespace(
        bundle=types.SimpleNamespace(occurrences=occurrences)
    )

    assert _successful_complete_projection(
        item=item,
        annotation=annotation,
        progress_step_id="write",
        stage_ids=("approach", "write"),
    )

    failed_annotation = types.SimpleNamespace(
        bundle=types.SimpleNamespace(
            occurrences=(
                occurrences[0],
                types.SimpleNamespace(
                    **{
                        **occurrences[1].__dict__,
                        "outcome": "ABORTED",
                    }
                ),
            )
        )
    )
    assert not _successful_complete_projection(
        item=item,
        annotation=failed_annotation,
        progress_step_id="write",
        stage_ids=("approach", "write"),
    )


def test_openpi_checkpoint_dir_resolves_relative_base_in_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caller = tmp_path / "DexTeleop"
    checkout = tmp_path / "openpi"
    caller.mkdir()
    checkout.mkdir()
    monkeypatch.chdir(caller)

    relative = _openpi_checkpoint_dir(
        _FakeOpenPITrainConfig("./checkpoints"),
        checkout,
    )
    assert relative == (
        checkout / "checkpoints/writing-stage2/independent-pi05"
    ).resolve()
    assert Path.cwd() == caller

    relative.mkdir(parents=True)
    (relative / "wandb").mkdir()
    (relative / "wandb_id.txt").write_text("run-id\n", encoding="utf-8")
    recovered = _resolve_train_checkpoint_policy(
        relative,
        resume=False,
        overwrite=False,
    )
    assert recovered.mode == "recover_empty_checkpoint_root"
    assert recovered.checkpoint_dir == relative

    absolute_base = tmp_path / "external-checkpoints"
    absolute = _openpi_checkpoint_dir(
        _FakeOpenPITrainConfig(str(absolute_base)),
        checkout,
    )
    assert absolute == (
        absolute_base / "writing-stage2/independent-pi05"
    ).resolve()
    assert Path.cwd() == caller
    assert _openpi_checkpoint_base_root(
        _FakeOpenPITrainConfig("./checkpoints"),
        checkout,
    ) == (checkout / "checkpoints").resolve()
    assert _openpi_checkpoint_base_root(
        _FakeOpenPITrainConfig(str(absolute_base)),
        checkout,
    ) == absolute_base.resolve()


def test_openpi_data_assets_root_resolves_openpi_relative_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caller = tmp_path / "DexTeleop"
    checkout = tmp_path / "openpi"
    caller.mkdir()
    checkout.mkdir()
    monkeypatch.chdir(caller)
    fake_data = types.SimpleNamespace(
        assets=types.SimpleNamespace(assets_dir=None),
    )

    class _LazyConfig:
        data = fake_data

        @property
        def assets_dirs(self) -> Path:
            return (Path("./assets") / "prepared-config").resolve()

    assert _openpi_data_assets_root(_LazyConfig(), checkout) == (
        checkout / "assets/prepared-config"
    ).resolve()
    assert Path.cwd() == caller

    absolute_assets = tmp_path / "shared-openpi-assets"
    explicit = types.SimpleNamespace(
        data=types.SimpleNamespace(
            assets=types.SimpleNamespace(assets_dir=str(absolute_assets)),
        ),
        assets_dirs=caller / "must-not-be-used",
    )
    assert _openpi_data_assets_root(explicit, checkout) == absolute_assets
    assert Path.cwd() == caller


def test_resolved_model_rtc_metadata_binding_is_scoped_and_identity_based() -> None:
    calls: list[object] = []

    def original(config: object) -> object:
        calls.append(config)
        return types.SimpleNamespace(source=config)

    training_config = types.SimpleNamespace(resolve_policy_metadata=original)
    resolved = types.SimpleNamespace(
        policy_metadata={"model_continuation_training_run": {"schema": "v1"}}
    )
    foreign = object()

    with _resolved_policy_metadata_binding(training_config, resolved):
        assert training_config.resolve_policy_metadata(resolved) is resolved
        fallback = training_config.resolve_policy_metadata(foreign)
        assert fallback.source is foreign
    assert calls == [foreign]
    assert training_config.resolve_policy_metadata is original


def test_model_rtc_receipt_import_binding_is_pure_and_scoped() -> None:
    module_name = "deployments.openpi.client.rtc"
    previous = sys.modules.pop(module_name, None)
    try:
        with _openpi_model_rtc_receipt_import_binding():
            module = sys.modules[module_name]
            assert module.__all__ == ("openpi_rtc_artifact_hash",)
            assert module.openpi_rtc_artifact_hash.__module__ == (
                "deployments.openpi.artifact_identity"
            )
        assert module_name not in sys.modules
    finally:
        if previous is not None:
            sys.modules[module_name] = previous


def test_prepared_profile_validates_canonical_source_task_id(
    tmp_path: Path,
) -> None:
    roots = (tmp_path / "source-a", tmp_path / "source-b")
    for index, root in enumerate(roots):
        meta = root / "meta"
        meta.mkdir(parents=True)
        task_id = "canonical-task" if index == 0 else "wrong-task"
        (meta / "info.json").write_text(
            json.dumps({"task_id": task_id}) + "\n",
            encoding="utf-8",
        )
        (meta / "tasks.jsonl").write_text(
            json.dumps({"task_id": task_id, "description": "writing"})
            + "\n",
            encoding="utf-8",
        )
    profile = {
        "expected_task_id": "canonical-task",
        "sources": [
            {
                "source_dataset_id": f"source-{index}",
                "dataset_root": str(root),
            }
            for index, root in enumerate(roots)
        ],
    }

    with pytest.raises(ValueError, match="source-1:meta/info.json"):
        _validate_source_task_ids(profile, _sources(profile))

    dry_run = normalize_profile_task_metadata(profile, apply=False)
    assert dry_run["changed_file_count"] == 2
    assert json.loads((roots[1] / "meta/info.json").read_text())["task_id"] == (
        "wrong-task"
    )

    applied = normalize_profile_task_metadata(profile, apply=True)
    assert applied["changed_file_count"] == 2
    assert all(
        change["old_inode"] != change["new_inode"]
        for change in applied["changes"]
    )
    _validate_source_task_ids(profile, _sources(profile))


def test_prepared_profile_accepts_only_explicit_source_task_aliases(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    meta = source / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps({"task_id": "primary-task"}) + "\n",
        encoding="utf-8",
    )
    (meta / "tasks.jsonl").write_text(
        "".join(
            json.dumps({"task_id": task_id}) + "\n"
            for task_id in ("legacy-task", "primary-task")
        ),
        encoding="utf-8",
    )
    profile = {
        "sources": [
            {
                "source_dataset_id": "source-a",
                "dataset_root": str(source),
                "expected_task_id": "primary-task",
                "expected_task_id_aliases": ["legacy-task"],
            }
        ]
    }
    _validate_source_task_ids(profile, _sources(profile))

    profile["sources"][0]["expected_task_id_aliases"] = ["wrong-alias"]
    with pytest.raises(ValueError, match="explicitly bound expected value"):
        _validate_source_task_ids(profile, _sources(profile))


def test_train_checkpoint_policy_handles_fresh_resume_and_empty_root(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    fresh = _resolve_train_checkpoint_policy(
        missing,
        resume=False,
        overwrite=False,
    )
    assert fresh.mode == "fresh"
    assert fresh.resume is False

    residual = tmp_path / "residual"
    residual.mkdir()
    (residual / "wandb").mkdir()
    (residual / "wandb_id.txt").write_text("run-id\n", encoding="utf-8")
    recovered = _resolve_train_checkpoint_policy(
        residual,
        resume=False,
        overwrite=False,
    )
    assert recovered.mode == "recover_empty_checkpoint_root"
    assert recovered.resume is True
    assert recovered.latest_checkpoint_step is None

    with pytest.raises(FileNotFoundError, match="contains no positive"):
        _resolve_train_checkpoint_policy(
            residual,
            resume=True,
            overwrite=False,
        )

    checkpoint_root = tmp_path / "checkpoint"
    (checkpoint_root / "30000").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="Use --resume"):
        _resolve_train_checkpoint_policy(
            checkpoint_root,
            resume=False,
            overwrite=False,
        )
    resumed = _resolve_train_checkpoint_policy(
        checkpoint_root,
        resume=True,
        overwrite=False,
    )
    assert resumed.mode == "resume"
    assert resumed.resume is True
    assert resumed.latest_checkpoint_step == 30000


def test_train_checkpoint_policy_rejects_unknown_empty_root_contents(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint_root.mkdir()
    (checkpoint_root / "unexpected.txt").write_text("unknown\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="unknown entries"):
        _resolve_train_checkpoint_policy(
            checkpoint_root,
            resume=False,
            overwrite=False,
        )


def test_prepared_skill_script_loader_registers_and_restores_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "_dexteleop_test_prepared_skill_script"
    good_path = tmp_path / "good_script.py"
    good_path.write_text(
        """from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class Args:
    value: int = 11
""",
        encoding="utf-8",
    )
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    module = _load_module(good_path, module_name)

    assert sys.modules[module_name] is module
    assert module.Args().value == 11

    broken_path = tmp_path / "broken_script.py"
    broken_path.write_text(
        "raise RuntimeError('prepared script import failed')\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="prepared script import failed"):
        _load_module(broken_path, module_name)

    assert sys.modules[module_name] is module


def _binding(tmp_path: Path) -> OpenPIPreparedSkillBinding:
    source_a = tmp_path / "source-a"
    source_b = tmp_path / "source-b"
    source_a.mkdir()
    source_b.mkdir()
    return OpenPIPreparedSkillBinding(
        view_root=tmp_path / "view",
        organized_dataset_root=tmp_path,
        training_view_hash="a" * 64,
        skill_id="fixture.skill",
        skill_training_contract_hash="b" * 64,
        training_domain="NOMINAL",
        camera_slot_to_dataset_key=(("base_0_rgb", "front"),),
        action_horizon=16,
        action_pose_mode="absolute",
        source_mixing_policy=OPENPI_SOURCE_MIXING_NATURAL_V1,
        episodes=(
            OpenPIPreparedEpisodeSelection(
                source_dataset_id="source-a",
                source_dataset_root=source_a,
                source_episode_index=0,
                organized_episode_ordinal=4,
                episode_capture_identity_hash="c" * 64,
                intervals=((2, 19),),
            ),
            OpenPIPreparedEpisodeSelection(
                source_dataset_id="source-b",
                source_dataset_root=source_b,
                source_episode_index=0,
                organized_episode_ordinal=9,
                episode_capture_identity_hash="d" * 64,
                intervals=((3, 19),),
            ),
        ),
    )


def _prompt_binding(tmp_path: Path):
    first = _binding(tmp_path)
    second = dataclasses.replace(
        first,
        view_root=tmp_path / "view-stage2",
        training_view_hash="e" * 64,
        skill_id="fixture.skill.stage2",
        skill_training_contract_hash="f" * 64,
        episodes=tuple(
            dataclasses.replace(
                episode,
                intervals=((20, 36),),
            )
            for episode in first.episodes
        ),
    )
    return compose_openpi_prepared_prompt_binding(
        (
            ("stage1", "pick up the brush", first),
            ("stage2", "remove the brush cap", second),
        ),
        organized_dataset_root=tmp_path / "joint",
    )


def test_prepared_profile_can_replay_one_exact_immutable_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _binding(tmp_path)
    binding_root = materialize_openpi_prepared_skill_binding(binding, tmp_path)
    report = types.SimpleNamespace(
        included_episode_identities=("c" * 64, "d" * 64),
        exclusions=(),
        annotation_selection={},
    )
    profile = {
        "organized_dataset_root": str(tmp_path),
        "sources": [
            {
                "source_dataset_id": "source-a",
                "dataset_root": str(tmp_path / "source-a"),
                "expected_eligible_episode_count": 1,
            },
            {
                "source_dataset_id": "source-b",
                "dataset_root": str(tmp_path / "source-b"),
                "expected_eligible_episode_count": 1,
            },
        ],
        "expected_total_eligible_episode_count": 2,
        "target_skill_id": binding.skill_id,
        "training_domain": binding.training_domain,
        "camera_slot_to_dataset_key": dict(
            binding.camera_slot_to_dataset_key
        ),
        "action_horizon": binding.action_horizon,
        "action_pose_mode": binding.action_pose_mode,
        "source_mixing_policy": binding.source_mixing_policy,
        "expected_prepared_binding_hash": binding.binding_hash,
    }
    monkeypatch.setattr(
        "deployments.openpi.training.prepared_skill_launch."
        "_validate_source_task_ids",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "deployments.openpi.training.prepared_skill_launch."
        "_skill_projection_catalog",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "deployments.openpi.training.prepared_skill_launch."
        "select_strict_successful_skill_episodes",
        lambda *_args, **_kwargs: report,
    )
    monkeypatch.setattr(
        "deployments.openpi.training.prepared_skill_launch."
        "_included_counts_by_source",
        lambda *_args, **_kwargs: {"source-a": 1, "source-b": 1},
    )

    loaded_report, loaded_binding, loaded_root = prepare_profile(
        profile,
        verify_source_files=False,
    )

    assert loaded_report is report
    assert loaded_binding is not None
    assert loaded_binding.to_payload() == binding.to_payload()
    assert loaded_root == binding_root


def test_prepared_profile_replay_rejects_episode_order_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _binding(tmp_path)
    materialize_openpi_prepared_skill_binding(binding, tmp_path)
    report = types.SimpleNamespace(
        included_episode_identities=("d" * 64, "c" * 64),
        exclusions=(),
        annotation_selection={},
    )
    profile = {
        "organized_dataset_root": str(tmp_path),
        "sources": [
            {
                "source_dataset_id": "source-a",
                "dataset_root": str(tmp_path / "source-a"),
                "expected_eligible_episode_count": 1,
            },
            {
                "source_dataset_id": "source-b",
                "dataset_root": str(tmp_path / "source-b"),
                "expected_eligible_episode_count": 1,
            },
        ],
        "expected_total_eligible_episode_count": 2,
        "target_skill_id": binding.skill_id,
        "training_domain": binding.training_domain,
        "camera_slot_to_dataset_key": dict(
            binding.camera_slot_to_dataset_key
        ),
        "action_horizon": binding.action_horizon,
        "action_pose_mode": binding.action_pose_mode,
        "source_mixing_policy": binding.source_mixing_policy,
        "expected_prepared_binding_hash": binding.binding_hash,
    }
    monkeypatch.setattr(
        "deployments.openpi.training.prepared_skill_launch."
        "_validate_source_task_ids",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "deployments.openpi.training.prepared_skill_launch."
        "_skill_projection_catalog",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "deployments.openpi.training.prepared_skill_launch."
        "select_strict_successful_skill_episodes",
        lambda *_args, **_kwargs: report,
    )
    monkeypatch.setattr(
        "deployments.openpi.training.prepared_skill_launch."
        "_included_counts_by_source",
        lambda *_args, **_kwargs: {"source-a": 1, "source-b": 1},
    )

    with pytest.raises(ValueError, match="Episode order differs"):
        prepare_profile(profile, verify_source_files=False)


def test_prepared_prompt_binding_injects_exact_prompt_and_round_trips(
    tmp_path: Path,
) -> None:
    binding = _prompt_binding(tmp_path)

    class Episode:
        episode_index = 0

    class NumericDataset:
        dataset_root = tmp_path / "source-a"
        _episodes = (Episode(),)
        _index = ((0, 2), (0, 20))

        @staticmethod
        def __getitem__(index: int) -> dict[str, object]:
            return {
                "row": NumericDataset._index[index][1],
                "prompt": "bimanual cube manipulation",
            }

        @staticmethod
        def __len__() -> int:
            return 2

    dataset = OpenPIPreparedSkillDataset(
        NumericDataset(),
        binding,
        "augmentation-key",
    )
    assert dataset[0]["prompt"] == "pick up the brush"
    assert dataset[1]["prompt"] == "remove the brush cap"
    assert tuple(dataset[0]["augmentation-key"]) == (0, 2)

    root = materialize_openpi_prepared_skill_binding(
        binding,
        binding.organized_dataset_root,
    )
    loaded = load_openpi_prepared_prompt_binding(root)
    assert loaded.to_payload() == binding.to_payload()


def test_prepared_prompt_binding_rejects_overlapping_legal_windows(
    tmp_path: Path,
) -> None:
    first = _binding(tmp_path)
    second = dataclasses.replace(
        first,
        training_view_hash="e" * 64,
        skill_id="fixture.skill.stage2",
        skill_training_contract_hash="f" * 64,
        episodes=tuple(
            dataclasses.replace(episode, intervals=((3, 20),))
            for episode in first.episodes
        ),
    )
    with pytest.raises(ValueError, match="same action window"):
        compose_openpi_prepared_prompt_binding(
            (
                ("stage1", "pick up the brush", first),
                ("stage2", "remove the brush cap", second),
            ),
            organized_dataset_root=tmp_path / "joint",
        )


def test_prepared_prompt_binding_balances_stage_windows_without_replacement(
    tmp_path: Path,
) -> None:
    first = _binding(tmp_path)
    first = dataclasses.replace(
        first,
        episodes=(
            dataclasses.replace(first.episodes[0], intervals=((2, 20),)),
        ),
    )
    second = dataclasses.replace(
        first,
        training_view_hash="e" * 64,
        skill_id="fixture.skill.stage2",
        skill_training_contract_hash="f" * 64,
        episodes=(
            dataclasses.replace(first.episodes[0], intervals=((20, 36),)),
        ),
    )
    binding = compose_openpi_prepared_prompt_binding(
        (
            ("stage1", "pick up the brush", first),
            ("stage2", "remove the brush cap", second),
        ),
        organized_dataset_root=tmp_path / "joint",
        prompt_stage_sampling_policy=(
            OPENPI_PROMPT_STAGE_SAMPLING_BALANCED_V1
        ),
        prompt_stage_sampling_seed=7,
    )

    class Episode:
        episode_index = 0

    class NumericDataset:
        dataset_root = tmp_path / "source-a"
        _episodes = (Episode(),)
        _index = ((0, 2), (0, 3), (0, 4), (0, 20))

        @staticmethod
        def __getitem__(index: int) -> dict[str, object]:
            return {"row": NumericDataset._index[index][1]}

        @staticmethod
        def __len__() -> int:
            return 4

    dataset = OpenPIPreparedSkillDataset(
        NumericDataset(),
        binding,
        "augmentation-key",
    )
    assert len(dataset) == 2
    assert set(dataset.prepared_prompt_stage_indices()) == {"stage1", "stage2"}
    assert sorted(dataset[index]["prompt"] for index in range(len(dataset))) == [
        "pick up the brush",
        "remove the brush cap",
    ]
    first_stage1_row = next(
        dataset[index]["row"]
        for index in range(len(dataset))
        if dataset[index]["prompt"] == "pick up the brush"
    )
    dataset.set_epoch(1)
    second_stage1_row = next(
        dataset[index]["row"]
        for index in range(len(dataset))
        if dataset[index]["prompt"] == "pick up the brush"
    )
    assert first_stage1_row in {2, 3, 4}
    assert second_stage1_row in {2, 3, 4}
    assert dataset.mixing_metadata()["effective_epoch_window_count"] == 2


def test_prepared_prompt_family_enforces_exact_80_20_and_excludes_held_out(
    tmp_path: Path,
) -> None:
    first = _binding(tmp_path)
    first = dataclasses.replace(
        first,
        episodes=(
            dataclasses.replace(first.episodes[0], intervals=((0, 82),)),
        ),
    )
    second = dataclasses.replace(
        first,
        training_view_hash="e" * 64,
        skill_id="fixture.skill.stage2",
        skill_training_contract_hash="f" * 64,
        episodes=(
            dataclasses.replace(first.episodes[0], intervals=((100, 182),)),
        ),
    )
    training_variants = {
        stage_id: tuple(
            f"{stage_id} training prompt variant {index}"
            for index in range(12)
        )
        for stage_id in ("stage1", "stage2")
    }
    held_out_variants = {
        stage_id: tuple(
            f"{stage_id} held-out prompt variant {index}"
            for index in range(4)
        )
        for stage_id in ("stage1", "stage2")
    }
    canonical = {
        "stage1": "stage1 canonical prompt",
        "stage2": "stage2 canonical prompt",
    }
    binding = compose_openpi_prepared_prompt_binding(
        (
            ("stage1", canonical["stage1"], first),
            ("stage2", canonical["stage2"], second),
        ),
        organized_dataset_root=tmp_path / "joint-family",
        prompt_stage_sampling_policy=(
            OPENPI_PROMPT_STAGE_SAMPLING_BALANCED_V1
        ),
        prompt_stage_sampling_seed=7,
        training_prompt_variants=training_variants,
        held_out_prompt_variants=held_out_variants,
        prompt_family_sampling_policy=(
            OPENPI_PROMPT_FAMILY_SAMPLING_CANONICAL_80_VARIANTS_20_V1
        ),
        prompt_family_sampling_seed=11,
    )

    class Episode:
        episode_index = 0

    class NumericDataset:
        dataset_root = tmp_path / "source-a"
        _episodes = (Episode(),)
        _index = tuple((0, row) for row in (*range(67), *range(100, 167)))

        @staticmethod
        def __getitem__(index: int) -> dict[str, object]:
            return {"row": NumericDataset._index[index][1]}

        @staticmethod
        def __len__() -> int:
            return len(NumericDataset._index)

    dataset = OpenPIPreparedSkillDataset(
        NumericDataset(),
        binding,
        "augmentation-key",
    )
    assert len(dataset) == 130
    epoch0_prompts = tuple(dataset[index]["prompt"] for index in range(len(dataset)))
    counts0 = {
        prompt: epoch0_prompts.count(prompt)
        for prompt in set(epoch0_prompts)
    }
    for stage_id in ("stage1", "stage2"):
        assert counts0[canonical[stage_id]] == 52
        assert sum(
            counts0.get(prompt, 0)
            for prompt in training_variants[stage_id]
        ) == 13
        assert counts0[training_variants[stage_id][0]] == 2
        assert all(
            prompt not in counts0 for prompt in held_out_variants[stage_id]
        )
    assert len({dataset[index]["row"] for index in range(len(dataset))}) == 130

    dataset.set_epoch(1)
    epoch1_prompts = tuple(dataset[index]["prompt"] for index in range(len(dataset)))
    counts1 = {
        prompt: epoch1_prompts.count(prompt)
        for prompt in set(epoch1_prompts)
    }
    for stage_id in ("stage1", "stage2"):
        assert counts1[canonical[stage_id]] == 52
        assert counts1[training_variants[stage_id][0]] == 1
        assert counts1[training_variants[stage_id][1]] == 2
    assert epoch1_prompts != epoch0_prompts
    dataset.set_epoch(0)
    assert tuple(
        dataset[index]["prompt"] for index in range(len(dataset))
    ) == epoch0_prompts

    metadata = dataset.mixing_metadata()
    assert metadata["openpi_quality_valid_window_counts_by_stage"] == {
        "stage1": 67,
        "stage2": 67,
    }
    assert metadata["effective_window_count_per_stage"] == 65
    assert metadata["prompt_family_quota_trimmed_window_count_per_stage"] == 2
    assert metadata["canonical_prompt_percent"] == 80
    assert metadata["training_variant_prompt_percent"] == 20
    assert metadata["held_out_prompts_in_training"] is False
    root = materialize_openpi_prepared_skill_binding(
        binding,
        binding.organized_dataset_root,
    )
    loaded = load_openpi_prepared_prompt_binding(root)
    assert loaded.to_payload() == binding.to_payload()


def test_prompt_family_token_preflight_covers_training_and_held_out(
    tmp_path: Path,
) -> None:
    profile_path = (
        Path(__file__).resolve().parents[1]
        / "deployments/openpi/configs/training"
        / "writing_multistage_prompt_family_0804_0806_front_h16_absolute_v2.json"
    )
    profile = _load_profile(profile_path)

    class SentencePiece:
        @staticmethod
        def encode(value: str, *, add_bos: bool) -> list[int]:
            assert add_bos is True
            return list(range(len(value.split()) + 1))

    class Paligemma:
        _tokenizer = SentencePiece()
        _max_len = 200

    class TokenizePrompt:
        tokenizer = Paligemma()
        discrete_state_input = True

    data_config = types.SimpleNamespace(
        model_transforms=types.SimpleNamespace(inputs=(TokenizePrompt(),))
    )
    model_config = types.SimpleNamespace(max_token_len=200)

    summary = _prompt_family_token_preflight(
        profile,
        data_config,
        model_config,
    )
    assert summary is not None
    assert summary["prompt_count"] == 68
    assert summary["measured_combination_count"] == 204
    assert summary["state_dimension"] == 16
    assert summary["truncation_detected"] is False

    too_short = dict(profile)
    too_short["max_full_token_length"] = 1
    with pytest.raises(ValueError, match="exceeds the profile limit"):
        _prompt_family_token_preflight(
            too_short,
            data_config,
            model_config,
        )


def test_post_identity_transform_runs_for_numeric_only_sample(
    tmp_path: Path,
) -> None:
    binding = _binding(tmp_path)

    class Episode:
        episode_index = 0

    class NumericDataset:
        dataset_root = tmp_path / "source-a"
        _episodes = (Episode(),)
        _index = ((0, 2),)

        @staticmethod
        def __getitem__(_index: int) -> dict[str, np.ndarray]:
            return {
                "state": np.zeros(16, dtype=np.float32),
                "actions": np.zeros((16, 16), dtype=np.float32),
            }

    observed: list[tuple[int, int]] = []

    class Transform:
        @staticmethod
        def apply_with_identity(sample, *, identity):
            observed.append(tuple(int(value) for value in identity))
            output = dict(sample)
            output["identity"] = tuple(
                int(value) for value in sample["augmentation-key"]
            )
            return output

    dataset = OpenPIPreparedSkillDataset(
        NumericDataset(),
        binding,
        "augmentation-key",
        Transform(),
    )
    output = dataset[0]
    assert output["identity"] == (4, 2)
    assert observed == [(4, 2)]


def _checkpoint_resume_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    dict[str, object],
    dict[str, object],
    OpenPIPreparedSkillBinding,
    Path,
    object,
]:
    binding = _binding(tmp_path)
    binding_root = materialize_openpi_prepared_skill_binding(binding, tmp_path)
    manifest_bytes = (binding_root / "manifest.json").read_bytes()
    checkpoint_root = tmp_path / "checkpoints/model/experiment"
    (checkpoint_root / "30000").mkdir(parents=True)
    source_binding = types.SimpleNamespace(
        source_id="fixture-model-rtc-v1",
        dataset_identity_hash="sha256:" + binding.binding_hash,
        episode_manifest_hash=(
            "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
        ),
    )
    run_payload = {
        "schema": "fixture-run-v1",
        "output_model_id": "fixture-model-rtc-output",
    }

    class _TrainingRun:
        output_model_id = "fixture-model-rtc-output"
        source_bindings = (source_binding,)

        @staticmethod
        def to_dict() -> dict[str, str]:
            return dict(run_payload)

    evidence = types.SimpleNamespace(training_run=_TrainingRun())
    monkeypatch.setattr(
        "deployments.openpi.training.prepared_skill_launch."
        "verify_openpi_model_rtc_checkpoint",
        lambda path: evidence,
    )

    def load_binding(view_root: Path, **kwargs: object) -> OpenPIPreparedSkillBinding:
        assert Path(view_root) == (
            tmp_path / "derived/training_views" / binding.training_view_hash
        )
        assert kwargs["organized_dataset_root"] == tmp_path
        return binding

    monkeypatch.setattr(
        "deployments.openpi.training.prepared_skill_launch."
        "load_openpi_prepared_skill_binding",
        load_binding,
    )
    profile: dict[str, object] = {
        "organized_dataset_root": str(tmp_path),
        "target_skill_id": binding.skill_id,
        "training_domain": binding.training_domain,
        "camera_slot_to_dataset_key": dict(
            binding.camera_slot_to_dataset_key
        ),
        "action_horizon": binding.action_horizon,
        "action_pose_mode": binding.action_pose_mode,
        "source_mixing_policy": binding.source_mixing_policy,
        "expected_total_eligible_episode_count": 2,
        "sources": [
            {
                "source_dataset_id": "source-a",
                "dataset_root": str(tmp_path / "source-a"),
                "expected_eligible_episode_count": 1,
            },
            {
                "source_dataset_id": "source-b",
                "dataset_root": str(tmp_path / "source-b"),
                "expected_eligible_episode_count": 1,
            },
        ],
    }
    model_profile: dict[str, object] = {
        "profile_id": "fixture-model-rtc-v1",
        "output_config_name": "fixture-model-rtc-output",
    }
    return (
        profile,
        model_profile,
        binding,
        checkpoint_root,
        evidence,
    )


def test_model_rtc_resume_loads_checkpoint_bound_prepared_view(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, model_profile, binding, checkpoint_root, evidence = (
        _checkpoint_resume_fixture(tmp_path, monkeypatch)
    )

    resume = _load_checkpoint_bound_prepared_resume(
        profile,
        model_profile,
        checkpoint_root=checkpoint_root,
        verify_source_files=True,
    )

    assert resume.binding is binding
    assert resume.binding_root.name == binding.binding_hash
    assert resume.checkpoint_policy.latest_checkpoint_step == 30000
    assert resume.evidence is evidence


def test_model_rtc_resume_rejects_binding_manifest_byte_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, model_profile, binding, checkpoint_root, _evidence = (
        _checkpoint_resume_fixture(tmp_path, monkeypatch)
    )
    manifest_path = (
        tmp_path
        / "derived/openpi_prepared_skill_bindings"
        / binding.binding_hash
        / "manifest.json"
    )
    manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="manifest bytes differ"):
        _load_checkpoint_bound_prepared_resume(
            profile,
            model_profile,
            checkpoint_root=checkpoint_root,
            verify_source_files=True,
        )


def test_model_rtc_resume_uses_checkpoint_counts_not_mutable_profile_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, model_profile, binding, checkpoint_root, _evidence = (
        _checkpoint_resume_fixture(tmp_path, monkeypatch)
    )
    sources = list(profile["sources"])
    sources[0] = dict(sources[0], expected_eligible_episode_count=2)
    profile["sources"] = sources
    profile["expected_total_eligible_episode_count"] = 3

    resume = _load_checkpoint_bound_prepared_resume(
        profile,
        model_profile,
        checkpoint_root=checkpoint_root,
        verify_source_files=True,
    )

    assert resume.binding is binding


def test_model_rtc_resume_rejects_profile_source_root_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, model_profile, _binding, checkpoint_root, _evidence = (
        _checkpoint_resume_fixture(tmp_path, monkeypatch)
    )
    sources = list(profile["sources"])
    sources[0] = dict(sources[0], dataset_root=str(tmp_path / "moved-source-a"))
    profile["sources"] = sources

    with pytest.raises(ValueError, match="source root differs"):
        _load_checkpoint_bound_prepared_resume(
            profile,
            model_profile,
            checkpoint_root=checkpoint_root,
            verify_source_files=True,
        )


def test_model_rtc_resume_requires_exact_reconstructed_training_run(
    tmp_path: Path,
) -> None:
    binding = _binding(tmp_path)

    class _TrainingRun:
        @staticmethod
        def to_dict() -> dict[str, str]:
            return {"schema": "exact-run-v1", "train_config_hash": "fixed"}

    resume = _CheckpointBoundPreparedResume(
        binding=binding,
        binding_root=tmp_path,
        checkpoint_policy=_TrainCheckpointPolicy(
            checkpoint_dir=tmp_path / "checkpoints",
            mode="resume",
            resume=True,
            overwrite=False,
            latest_checkpoint_step=30000,
        ),
        evidence=types.SimpleNamespace(training_run=_TrainingRun()),
    )
    exact = types.SimpleNamespace(
        policy_metadata={
            "model_continuation_training_run": _TrainingRun.to_dict()
        }
    )
    _validate_checkpoint_training_run(exact, resume)

    drifted = types.SimpleNamespace(
        policy_metadata={
            "model_continuation_training_run": {
                "schema": "exact-run-v1",
                "train_config_hash": "drifted",
            }
        }
    )
    with pytest.raises(ValueError, match="differs from the latest checkpoint"):
        _validate_checkpoint_training_run(drifted, resume)


def test_binding_materializes_source_local_episode_manifests(tmp_path: Path) -> None:
    binding = _binding(tmp_path)

    root = materialize_openpi_prepared_skill_binding(binding, tmp_path / "out")
    repeated = materialize_openpi_prepared_skill_binding(binding, tmp_path / "out")

    assert repeated == root
    assert (
        load_openpi_prepared_skill_binding_manifest(root).to_payload()
        == binding.to_payload()
    )
    audit = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert audit["binding"]["binding_hash"] == binding.binding_hash
    manifests = sorted(root.glob("source_*.jsonl"))
    assert len(manifests) == 2
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in manifests]
    assert {tuple(row["camera_slots"]) for row in rows} == {("base_0_rgb",)}
    assert {row["source_episode_index"] for row in rows} == {0}

    weighted = dataclasses.replace(
        binding,
        source_mixing_policy=OPENPI_SOURCE_MIXING_REPEAT_BY_SOURCE_V1,
        source_repeat_weights=(("source-a", 1), ("source-b", 6)),
        selection_policy_id="coarse_stage1_non_abnormal_fixture_v1",
    )
    weighted_root = materialize_openpi_prepared_skill_binding(
        weighted,
        tmp_path / "weighted-out",
    )
    assert (
        load_openpi_prepared_skill_binding_manifest(weighted_root).to_payload()
        == weighted.to_payload()
    )
    assert weighted.to_payload()["prepared_interval_repeated_window_count"] == 8
    weighted_rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(weighted_root.glob("source_*.jsonl"))
    ]
    assert {row["quality_class"] for row in weighted_rows} == {
        "coarse_stage1_non_abnormal_fixture_v1"
    }


def test_openpi_patch_intersects_horizon_and_globalizes_source_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    binding = _binding(tmp_path)

    class Episode:
        def __init__(self) -> None:
            self.episode_index = 0

    class FakeDataset:
        def __init__(self, root: Path) -> None:
            self.dataset_root = root
            self.action_horizon = 16
            self._episodes = [Episode()]
            self._index = self._build_index()

        def _build_index(self):
            return [(0, index) for index in range(0, 8)]

        def __len__(self) -> int:
            return len(self._index)

        def __getitem__(self, index: int):
            _episode_position, row = self._index[index]
            return {
                "augmentation-key": np.asarray((0, row), dtype=np.int64),
                "row": row,
            }

    class FakeMixed:
        def __init__(self) -> None:
            self.source_configs = (
                types.SimpleNamespace(
                    name="source-a", to_dict=lambda: {"name": "source-a"}
                ),
                types.SimpleNamespace(
                    name="source-b", to_dict=lambda: {"name": "source-b"}
                ),
            )
            self._datasets = [
                FakeDataset(tmp_path / "source-a"),
                FakeDataset(tmp_path / "source-b"),
            ]
            self._mix_index = [
                (source, index % len(value))
                for index in range(max(len(value) for value in self._datasets))
                for source, value in enumerate(self._datasets)
            ]

        def __len__(self) -> int:
            return len(self._mix_index)

        def __getitem__(self, index: int):
            source, child = self._mix_index[index]
            return self._datasets[source][child]

        def composition(self):
            return {
                source.name: sum(
                    source_index == position
                    for source_index, _sample_index in self._mix_index
                )
                for position, source in enumerate(self.source_configs)
            }

        def mixing_metadata(self):
            return {"schema": "equal-source-fixture"}

    class FakeExactMixSampler:
        pass

    openpi = types.ModuleType("openpi")
    policies = types.ModuleType("openpi.policies")
    policy = types.ModuleType("openpi.policies.dexteleop_v23_policy")
    policy.DexTeleopV23RelativeDataset = FakeDataset
    policy.DexTeleopV23MixedRelativeDataset = FakeMixed
    policy.STATE_AUGMENTATION_KEY = "augmentation-key"
    policies.dexteleop_v23_policy = policy
    training = types.ModuleType("openpi.training")
    data_loader = types.ModuleType("openpi.training.data_loader")
    original_create = lambda *_args, **_kwargs: FakeMixed()
    data_loader.create_torch_dataset = original_create
    data_loader._DexTeleopV22ExactMixSampler = FakeExactMixSampler
    training.data_loader = data_loader
    monkeypatch.setitem(sys.modules, "openpi", openpi)
    monkeypatch.setitem(sys.modules, "openpi.policies", policies)
    monkeypatch.setitem(
        sys.modules,
        "openpi.policies.dexteleop_v23_policy",
        policy,
    )
    monkeypatch.setitem(sys.modules, "openpi.training", training)
    monkeypatch.setitem(sys.modules, "openpi.training.data_loader", data_loader)

    with patch_openpi_prepared_skill_dataset(binding):
        dataset = data_loader.create_torch_dataset(None, 16, None)
        assert len(dataset) == 3
        first = dataset[0]
        second = dataset[1]
        third = dataset[2]
        sampler = data_loader._DexTeleopV22ExactMixSampler(3, 2, 7)
        assert isinstance(sampler, OpenPINaturalMergeSampler)
        assert sorted(sampler) == [0, 1, 2]
        assert dataset.composition() == {"source-a": 2, "source-b": 1}
        assert dataset.mixing_metadata()["source_mixing_policy"] == (
            OPENPI_SOURCE_MIXING_NATURAL_V1
        )

    weighted = dataclasses.replace(
        binding,
        source_mixing_policy=OPENPI_SOURCE_MIXING_REPEAT_BY_SOURCE_V1,
        source_repeat_weights=(("source-a", 1), ("source-b", 6)),
        selection_policy_id="coarse_stage1_non_abnormal_fixture_v1",
    )
    with patch_openpi_prepared_skill_dataset(weighted):
        weighted_dataset = data_loader.create_torch_dataset(None, 16, None)
        assert len(weighted_dataset) == 8
        assert weighted_dataset.composition() == {
            "source-a": 2,
            "source-b": 6,
        }
        metadata = weighted_dataset.mixing_metadata()
        assert metadata["source_mixing_policy"] == (
            OPENPI_SOURCE_MIXING_REPEAT_BY_SOURCE_V1
        )
        assert metadata["source_repeat_weights"] == {
            "source-a": 1,
            "source-b": 6,
        }

    assert data_loader.create_torch_dataset is original_create
    assert data_loader._DexTeleopV22ExactMixSampler is FakeExactMixSampler
    assert first["row"] == 2
    assert tuple(first["augmentation-key"]) == (4, 2)
    assert second["row"] == 3
    assert tuple(second["augmentation-key"]) == (4, 3)
    assert third["row"] == 3
    assert tuple(third["augmentation-key"]) == (9, 3)


def test_prepared_post_identity_transform_observes_global_identity_and_epoch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    binding = _binding(tmp_path)

    class Episode:
        episode_index = 0

    class FakeDataset:
        def __init__(self, root: Path) -> None:
            self.dataset_root = root
            self.action_horizon = 16
            self._episodes = [Episode()]
            self._index = [(0, 2)]

        def _build_index(self):
            return list(self._index)

        def __len__(self) -> int:
            return len(self._index)

        def __getitem__(self, _index: int):
            return {
                "augmentation-key": np.asarray((0, 2), dtype=np.int64),
            }

    class FakeMixed:
        def __init__(self) -> None:
            self.source_configs = (
                types.SimpleNamespace(
                    name="source-a", to_dict=lambda: {"name": "source-a"}
                ),
            )
            self._datasets = [FakeDataset(tmp_path / "source-a")]
            self._mix_index = [(0, 0)]

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int):
            source, child = self._mix_index[index]
            return self._datasets[source][child]

    class Transform:
        epoch = 0
        received_identity = None

        def __call__(self, sample):
            output = dict(sample)
            output["observed-identity"] = tuple(
                int(value) for value in output["augmentation-key"]
            )
            return output

        def set_epoch(self, epoch: int) -> None:
            self.epoch = int(epoch)

        def apply_with_identity(self, sample, *, identity):
            self.received_identity = tuple(int(value) for value in identity)
            return self(sample)

    class FakeExactMixSampler:
        pass

    openpi = types.ModuleType("openpi")
    policies = types.ModuleType("openpi.policies")
    policy = types.ModuleType("openpi.policies.dexteleop_v23_policy")
    policy.DexTeleopV23RelativeDataset = FakeDataset
    policy.DexTeleopV23MixedRelativeDataset = FakeMixed
    policy.STATE_AUGMENTATION_KEY = "augmentation-key"
    policies.dexteleop_v23_policy = policy
    training = types.ModuleType("openpi.training")
    data_loader = types.ModuleType("openpi.training.data_loader")
    create_kwargs: list[dict[str, object]] = []

    def create_dataset(*_args, **kwargs):
        create_kwargs.append(dict(kwargs))
        return FakeMixed()

    data_loader.create_torch_dataset = create_dataset
    data_loader._DexTeleopV22ExactMixSampler = FakeExactMixSampler
    training.data_loader = data_loader
    monkeypatch.setitem(sys.modules, "openpi", openpi)
    monkeypatch.setitem(sys.modules, "openpi.policies", policies)
    monkeypatch.setitem(
        sys.modules,
        "openpi.policies.dexteleop_v23_policy",
        policy,
    )
    monkeypatch.setitem(sys.modules, "openpi.training", training)
    monkeypatch.setitem(sys.modules, "openpi.training.data_loader", data_loader)

    transform = Transform()
    with patch_openpi_prepared_skill_dataset(
        binding,
        post_identity_transform=transform,
        load_images=False,
    ):
        dataset = data_loader.create_torch_dataset(None, 16, None)
        assert dataset.prepared_sample_identity(0) == (4, 2)
        assert dataset[0]["observed-identity"] == (4, 2)
        assert transform.received_identity == (4, 2)
        dataset.set_epoch(7)
        assert transform.epoch == 7
    assert create_kwargs == [{"load_images": False}]


def test_prepared_metadata_records_natural_merge_and_rehashes_input_contract(
    tmp_path: Path,
) -> None:
    binding = _binding(tmp_path)

    class Source:
        def __init__(self, name: str) -> None:
            self.name = name

        def to_dict(self) -> dict[str, str]:
            return {"name": self.name}

    @dataclasses.dataclass(frozen=True)
    class Config:
        data: object
        num_workers: int
        epoch_worker_mode: str
        policy_metadata: dict[str, object]

    old_contract = {
        "schema": "dexteleop_corrected_training_input_v1",
        "profile": "stale-mixed-profile",
        "num_workers": 4,
        "contract_hash": "sha256:stale",
    }
    metadata = {
        "training_input_contract": dict(old_contract),
        "dataset_mixing": {"schema": "stale-equal-source"},
        "dataset_adapter": {
            "training_input_contract": dict(old_contract),
            "dataset_mixing": {"schema": "stale-equal-source"},
        },
        "policy_artifact": {
            "training_input_contract": dict(old_contract),
            "dataset_mixing": {"schema": "stale-equal-source"},
        },
    }
    config = Config(
        data=types.SimpleNamespace(
            dataset_sources=(Source("source-a"), Source("source-b")),
            background_provider="procedural",
        ),
        num_workers=8,
        epoch_worker_mode="shared_worker_v1",
        policy_metadata=metadata,
    )
    original = lambda value: value
    training_config = types.SimpleNamespace(resolve_policy_metadata=original)

    with _prepared_metadata_binding(training_config, binding):
        resolved = training_config.resolve_policy_metadata(config)

    assert training_config.resolve_policy_metadata is original
    resolved_metadata = resolved.policy_metadata
    mixing = resolved_metadata["dataset_mixing"]
    assert mixing["schema"] == "dexteleop_v23_natural_merge_v1"
    assert mixing["source_order"] == ["source-a", "source-b"]
    assert mixing["each_quality_valid_window_once_per_epoch"] is True
    contract = resolved_metadata["training_input_contract"]
    assert contract["num_workers"] == 8
    assert contract["source_mixing_policy"] == (
        OPENPI_SOURCE_MIXING_NATURAL_V1
    )
    assert "procedural" in contract["profile"]
    unhashed = dict(contract)
    actual_hash = unhashed.pop("contract_hash")
    canonical = json.dumps(
        unhashed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    assert actual_hash == (
        "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
    )
    for container_key in ("dataset_adapter", "policy_artifact"):
        container = resolved_metadata[container_key]
        assert container["dataset_mixing"] == mixing
        assert container["training_input_contract"] == contract


def test_prepared_state_only_metadata_accepts_base_without_corrected_contract(
    tmp_path: Path,
) -> None:
    binding = _binding(tmp_path)

    @dataclasses.dataclass(frozen=True)
    class Config:
        data: object
        num_workers: int
        epoch_worker_mode: str
        policy_metadata: dict[str, object]

    metadata = {
        "policy_artifact": {
            "camera_input": {"inputs": [{"server_camera_slot": "base_0_rgb"}]},
            "metadata": {},
        },
    }
    config = Config(
        data=types.SimpleNamespace(
            dataset_sources=(
                types.SimpleNamespace(
                    name="source-a",
                    to_dict=lambda: {"name": "source-a"},
                ),
            ),
            background_provider="procedural",
        ),
        num_workers=4,
        epoch_worker_mode="shared_worker_v1",
        policy_metadata=metadata,
    )
    original = lambda value: value
    training_config = types.SimpleNamespace(resolve_policy_metadata=original)
    bridge = OpenPIStateOnlyInputBridge(image_keys=("base_0_rgb",))

    with _prepared_metadata_binding(
        training_config,
        binding,
        state_only_bridge=bridge,
    ):
        resolved = training_config.resolve_policy_metadata(config)

    assert "training_input_contract" not in resolved.policy_metadata
    artifact = resolved.policy_metadata["policy_artifact"]
    assert artifact["camera_input"] == bridge.external_camera_contract.to_dict()
    assert artifact[OPENPI_STATE_ONLY_CAPABILITY_KEY_V1] == bridge.contract


def test_writing_stage1_profile_matrix_is_complete_and_consistent() -> None:
    root = (
        Path(__file__).resolve().parents[1]
        / "deployments/openpi/configs/training"
    )
    expected = {
        ("front", "absolute"): (
            "writing_stage1_0804_0805_front_h16_absolute_v1.json",
            ("front",),
            (
                "writing-phase1-grap-pen-front-only-h16-absolute-"
                "task-ready-state-dropout-v2"
            ),
        ),
        ("front", "relative"): (
            "writing_stage1_0804_0805_front_h16_relative_v1.json",
            ("front",),
            (
                "writing-phase1-grap-pen-front-only-h16-relative-"
                "task-ready-state-dropout-v2"
            ),
        ),
        ("full", "absolute"): (
            "writing_stage1_0804_0805_full_h16_absolute_v1.json",
            ("front", "left_wrist", "right_wrist"),
            "writing-phase1-grap-pen-h16-absolute-corrected-v1",
        ),
        ("full", "relative"): (
            "writing_stage1_0804_0805_full_h16_relative_v1.json",
            ("front", "left_wrist", "right_wrist"),
            "writing-phase1-grap-pen-h16-relative-corrected-v1",
        ),
    }
    profiles = {
        key: _load_profile(root / filename)
        for key, (filename, _cameras, _base) in expected.items()
    }
    first = next(iter(profiles.values()))
    config_names: set[str] = set()
    asset_ids: set[str] = set()
    workspace_roots: set[str] = set()
    for (view, pose_mode), profile in profiles.items():
        _filename, cameras, base_config = expected[(view, pose_mode)]
        assert tuple(profile["camera_selection"]) == cameras
        assert profile["action_pose_mode"] == pose_mode
        assert profile["openpi_base_config"] == base_config
        assert profile["openpi_state_augmentation_from_config"] == (
            "writing-phase1-grap-pen-front-only-h16-"
            f"{pose_mode}-task-ready-state-dropout-v2"
        )
        assert profile["openpi_num_workers"] == 4
        assert profile["expected_total_eligible_episode_count"] == 602
        assert profile["sources"] == first["sources"]
        assert profile["annotation_selection"] == first["annotation_selection"]
        assert profile["target_skill_id"] == first["target_skill_id"]
        assert profile["source_mixing_policy"] == (
            OPENPI_SOURCE_MIXING_NATURAL_V1
        )
        config_names.add(profile["openpi_config_name"])
        asset_ids.add(profile["openpi_asset_id"])
        workspace_roots.add(profile["organized_dataset_root"])
    assert len(config_names) == 4
    assert len(asset_ids) == 4
    assert len(workspace_roots) == 4


def test_writing_later_stage_profiles_are_independent_pi05_runs() -> None:
    root = (
        Path(__file__).resolve().parents[1]
        / "deployments/openpi/configs/training"
    )
    expected = {
        2: (
            "writing_stage2_0804_0805_front_h16_absolute_v1.json",
            "writing.remove_brush_cap_and_return_cap",
            (344, 247),
        ),
        3: (
            "writing_stage3_0804_0805_front_h16_absolute_v1.json",
            "writing.write_with_brush",
            (344, 246),
        ),
        4: (
            "writing_stage4_0804_0805_front_h16_absolute_v1.json",
            "writing.return_brush",
            (344, 244),
        ),
    }
    base_config = (
        "writing-phase1-grap-pen-front-only-h16-absolute-"
        "task-ready-state-dropout-v2"
    )
    config_names: set[str] = set()
    asset_ids: set[str] = set()
    workspace_roots: set[str] = set()
    for stage, (filename, skill_id, source_counts) in expected.items():
        profile = _load_profile(root / filename)
        assert profile["skill_projection_catalog_id"] == (
            "writing_four_stage_subtasks_v2"
        )
        assert profile["target_skill_id"] == skill_id
        assert profile["openpi_base_config"] == base_config
        assert profile["openpi_state_augmentation_from_config"] == base_config
        assert profile["camera_selection"] == ["front"]
        assert profile["action_pose_mode"] == "absolute"
        assert profile["action_horizon"] == 16
        assert profile["openpi_num_workers"] == 4
        assert tuple(
            source["expected_eligible_episode_count"]
            for source in profile["sources"]
        ) == source_counts
        assert profile["expected_total_eligible_episode_count"] == sum(
            source_counts
        )
        assert not {
            "base_checkpoint_dir",
            "checkpoint_init",
            "resume",
            "weight_loader",
        }.intersection(profile)
        assert f"writing-stage{stage}" in profile["openpi_config_name"]
        assert f"writing-stage{stage}" in profile["default_exp_name"]
        catalog = _skill_projection_catalog(profile)
        skill = catalog.skill_by_id[skill_id]
        assert skill.skill_version == "2"
        assert len(catalog.projections_for_skill(skill_id)) == 1
        config_names.add(profile["openpi_config_name"])
        asset_ids.add(profile["openpi_asset_id"])
        workspace_roots.add(profile["organized_dataset_root"])
    assert len(config_names) == 3
    assert len(asset_ids) == 3
    assert len(workspace_roots) == 3


def test_fulltask_profiles_pin_the_historical_annotation_after_late_relabel() -> None:
    root = (
        Path(__file__).resolve().parents[1]
        / "deployments/openpi/configs/training"
    )
    identity = (
        "a4f276be5948bb7c1900b047c31f69c5ea7c4695e7aa3d2e7e05864d8bf90f22"
    )
    expected_set = "stage-1786011149701-e78dad71"
    affected: list[str] = []
    for path in sorted(root.glob("*.json")):
        profile = json.loads(path.read_text(encoding="utf-8"))
        sources = profile.get("sources", ())
        if not any(
            isinstance(source, dict)
            and source.get("source_dataset_id")
            == "writing4_2026_08_06_part2"
            for source in sources
        ):
            continue
        selection = profile.get("annotation_selection")
        if not isinstance(selection, dict):
            continue
        affected.append(path.name)
        assert selection.get(identity) == expected_set
    assert len(affected) == 17


def test_writing_0804_0806_profiles_are_independent_pi05_runs() -> None:
    root = (
        Path(__file__).resolve().parents[1]
        / "deployments/openpi/configs/training"
    )
    expected = {
        1: (
            "writing.bimanual_acquire_brush_and_present_for_cap_removal",
            (344, 256, 35, 204, 51),
            "writing_stage1_v2",
        ),
        2: (
            "writing.remove_brush_cap_and_return_cap",
            (344, 247, 35, 203, 51),
            "writing_four_stage_subtasks_v2",
        ),
        3: (
            "writing.write_with_brush",
            (344, 246, 35, 203, 51),
            "writing_four_stage_subtasks_v2",
        ),
        4: (
            "writing.return_brush",
            (344, 244, 35, 203, 51),
            "writing_four_stage_subtasks_v2",
        ),
    }
    base_config = (
        "writing-phase1-grap-pen-front-only-h16-absolute-"
        "task-ready-state-dropout-v2"
    )
    names: set[str] = set()
    assets: set[str] = set()
    workspaces: set[str] = set()
    for stage, (skill_id, counts, catalog_id) in expected.items():
        profile = _load_profile(
            root
            / f"writing_stage{stage}_0804_0806_front_h16_absolute_v1.json"
        )
        assert profile["target_skill_id"] == skill_id
        assert profile["skill_projection_catalog_id"] == catalog_id
        assert profile["openpi_base_config"] == base_config
        assert profile["openpi_state_augmentation_from_config"] == base_config
        assert tuple(
            source["expected_eligible_episode_count"]
            for source in profile["sources"]
        ) == counts
        assert profile["expected_total_eligible_episode_count"] == sum(counts)
        assert profile["expected_task_id"] == (
            "dmtac_rm65_bimanual_position_v2_writing_4_stage"
        )
        # The late relabel of one historical Episode made the formerly unique
        # capture-time annotation ambiguous.  Pinning the set that produced the
        # immutable historical bindings adds one explicit selector without
        # changing the eligible Episode set.
        assert len(profile["annotation_selection"]) == 19
        assert all(
            len(identity) == 64
            and all(character in "0123456789abcdef" for character in identity)
            for identity in profile["annotation_selection"]
        )
        assert not {
            "base_checkpoint_dir",
            "checkpoint_init",
            "resume",
            "weight_loader",
        }.intersection(profile)
        names.add(profile["openpi_config_name"])
        assets.add(profile["openpi_asset_id"])
        workspaces.add(profile["organized_dataset_root"])
    assert len(names) == len(assets) == len(workspaces) == 4


def test_writing_0804_0806_joint_prompt_profile_binds_all_stages() -> None:
    root = (
        Path(__file__).resolve().parents[1]
        / "deployments/openpi/configs/training"
    )
    profile = _load_profile(
        root
        / "writing_multistage_prompt_0804_0806_front_h16_absolute_v1.json"
    )
    stages = profile["_resolved_prompt_stages"]
    assert tuple(stage["stage_id"] for stage in stages) == (
        "stage1",
        "stage2",
        "stage3",
        "stage4",
    )
    assert tuple(stage["expected_window_count"] for stage in stages) == (
        71818,
        53911,
        96649,
        78048,
    )
    assert tuple(
        stage["expected_openpi_window_count"] for stage in stages
    ) == (64999, 51459, 79078, 8995)
    assert len({stage["prompt"] for stage in stages}) == 4
    assert profile["expected_total_window_count"] == 300426
    assert profile["expected_union_episode_count"] == 890
    assert profile["expected_effective_epoch_window_count"] == 35980
    assert profile["openpi_num_train_steps"] == 600000
    assert all(
        stage["profile"]["expected_task_id"]
        == profile["expected_task_id"]
        for stage in stages
    )


def test_writing_0804_0807_prompt_family_profile_binds_all_stages() -> None:
    root = (
        Path(__file__).resolve().parents[1]
        / "deployments/openpi/configs/training"
    )
    profile = _load_profile(
        root
        / "writing_multistage_prompt_family_0804_0807_front_h16_absolute_v2.json"
    )
    stages = profile["_resolved_prompt_stages"]
    assert len(profile["sources"]) == 7
    assert tuple(stage["stage_id"] for stage in stages) == (
        "stage1",
        "stage2",
        "stage3",
        "stage4",
    )
    assert tuple(stage["expected_window_count"] for stage in stages) == (
        97250,
        71511,
        123287,
        101492,
    )
    assert tuple(
        stage["expected_openpi_window_count"] for stage in stages
    ) == (84193, 68109, 99919, 9827)
    assert tuple(stage["expected_child_binding_hash"] for stage in stages) == (
        "f8b2f722c1e6bb35d70b26f5eb715c25722ad33247f88ba653ac23ba9fcd1c2e",
        "8df5e18c7853e4eb0ae8530afb9868054f58e8cc61c815a7f0196818eda21ac0",
        "634b6f1c3c416f1646754aa20cc6d8f466b6e8ad2bf6b570fbd34a29dcfde401",
        "3b5c6e48549a38a54c1a84b704aee9b73cc7e38231d2c7054cecaad2abbc71c7",
    )
    assert profile["expected_total_window_count"] == 393540
    assert profile["expected_union_episode_count"] == 1217
    assert profile["expected_effective_epoch_window_count"] == 39300
    assert profile["canonical_prompt_percent"] == 80
    assert profile["training_variant_prompt_percent"] == 20
    assert all(len(stage["profile"]["annotation_selection"]) == 35 for stage in stages)


def test_prepared_profile_rejects_unknown_skill_catalog(tmp_path: Path) -> None:
    profile = {
        "skill_projection_catalog_id": "unknown-writing-catalog",
    }

    try:
        _skill_projection_catalog(profile)
    except ValueError as exc:
        assert "Unsupported Skill projection catalog id" in str(exc)
    else:
        raise AssertionError("Unknown catalog id was accepted.")
