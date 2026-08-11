from __future__ import annotations

from pathlib import Path

import pytest

from deployments.openpi.training import prepared_writing_state_sequence as sequence


def _stage(number: int, tmp_path: Path) -> sequence.WritingStateStage:
    return sequence.WritingStateStage(
        number=number,
        profile=tmp_path / f"state-{number}.json",
    )


def _state(
    stage: sequence.WritingStateStage,
    *,
    action: str,
    latest: int | None,
) -> sequence.WritingStateStageState:
    return sequence.WritingStateStageState(
        stage=stage,
        checkpoint_root=Path(f"/checkpoints/state-{stage.number}"),
        target_step=60_000,
        latest_checkpoint_step=latest,
        action=action,
    )


def test_checked_in_state_profiles_are_independent_ordinary_60k_runs() -> None:
    model_ids: list[str] = []
    asset_ids: list[str] = []
    for expected, stage in enumerate(sequence.DEFAULT_STAGES, start=1):
        profile = sequence._load_profile(stage.profile)
        assert stage.number == expected
        assert profile["openpi_observation_mode"] == (
            "state_only_masked_dummy_v1"
        )
        assert profile["openpi_state_augmentation_mode"] == "disabled_v1"
        assert profile["openpi_num_train_steps"] == 60_000
        assert profile["openpi_save_interval"] == 10_000
        assert profile["openpi_keep_period"] == 60_000
        assert profile["openpi_norm_stats_assets_dir"] == (
            "assets/"
            f"writing-stage{expected}-0804-0806-front-h16-absolute-prepared-v1"
        )
        assert profile["action_horizon"] == 16
        assert profile["action_pose_mode"] == "absolute"
        model_ids.append(str(profile["openpi_config_name"]))
        asset_ids.append(str(profile["openpi_asset_id"]))
    assert len(set(model_ids)) == 4
    assert len(set(asset_ids)) == 4


def test_state_sequence_resumes_and_advances_only_after_60k(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stages = (_stage(1, tmp_path), _stage(2, tmp_path))
    states = iter(
        (
            _state(stages[0], action="resume", latest=30_000),
            _state(stages[0], action="complete", latest=60_000),
            _state(stages[1], action="fresh", latest=None),
            _state(stages[1], action="complete", latest=60_000),
        )
    )
    launched: list[list[str]] = []
    preflights: list[None] = []
    monkeypatch.setattr(sequence, "inspect_stage", lambda _stage: next(states))
    monkeypatch.setattr(
        sequence,
        "_require_training_gpu_available",
        lambda: preflights.append(None),
    )
    monkeypatch.setattr(
        sequence,
        "_run_stage_process",
        lambda argv: launched.append(list(argv)) or 0,
    )
    monkeypatch.setattr(sequence.sys.stdin, "isatty", lambda: True)

    assert sequence.run_sequence(stages) == 0
    assert len(launched) == 2
    assert len(preflights) == 2
    assert "--resume" in launched[0]
    assert "--resume" not in launched[1]
    assert all("--model-rtc-profile" not in argv for argv in launched)


def test_state_sequence_dry_run_starts_no_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stages = (_stage(1, tmp_path),)
    monkeypatch.setattr(
        sequence,
        "inspect_stage",
        lambda stage: _state(stage, action="resume", latest=30_000),
    )
    monkeypatch.setattr(
        sequence,
        "_run_stage_process",
        lambda _argv: pytest.fail("dry-run must not start training"),
    )
    monkeypatch.setattr(
        sequence,
        "_require_training_gpu_available",
        lambda: pytest.fail("dry-run must not claim the GPU"),
    )
    assert sequence.run_sequence(stages, dry_run=True) == 0


def test_state_sequence_blocks_material_gpu_owner_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stages = (_stage(1, tmp_path),)
    monkeypatch.setattr(
        sequence,
        "inspect_stage",
        lambda stage: _state(stage, action="fresh", latest=None),
    )
    monkeypatch.setattr(
        sequence,
        "_material_gpu_processes",
        lambda: (
            sequence.GpuComputeProcess(
                pid=119929,
                process_name="python",
                used_memory_mib=34382,
                command="python -m deployments.openpi.prepared_server_runtime",
            ),
        ),
    )
    monkeypatch.setattr(
        sequence,
        "_run_stage_process",
        lambda _argv: pytest.fail("GPU conflict must block child launch"),
    )

    with pytest.raises(RuntimeError, match=r"pid=119929.*34382"):
        sequence.run_sequence(stages)


def test_material_gpu_processes_ignores_small_desktop_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = sequence.subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=(
            "3227, /usr/libexec/gnome-remote-desktop-daemon, 467\n"
            "119929, /home/user/zc/openpi/.venv/bin/python, 34382\n"
        ),
        stderr="",
    )
    monkeypatch.setattr(sequence.subprocess, "run", lambda *_a, **_k: completed)
    monkeypatch.setattr(sequence.os, "getpid", lambda: 999)
    monkeypatch.setattr(
        sequence,
        "_process_command",
        lambda pid, fallback: f"{fallback} --pid {pid}",
    )

    assert sequence._material_gpu_processes() == (
        sequence.GpuComputeProcess(
            pid=119929,
            process_name="/home/user/zc/openpi/.venv/bin/python",
            used_memory_mib=34382,
            command="/home/user/zc/openpi/.venv/bin/python --pid 119929",
        ),
    )


def test_state_sequence_launcher_uses_supported_prepared_skill_arguments(
    tmp_path: Path,
) -> None:
    stage = _stage(1, tmp_path)
    argv = sequence._launcher_argv(
        _state(stage, action="fresh", latest=None)
    )

    assert argv[-4:] == [
        "deployments.openpi.training.prepared_skill_launch",
        "train",
        "--profile",
        str(stage.profile),
    ]
    assert "--no-motion" not in argv
