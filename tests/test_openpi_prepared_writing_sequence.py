from __future__ import annotations

from pathlib import Path
import types

import pytest

from deployments.openpi.model_rtc_profile import (
    load_prepared_model_rtc_profile,
)
from deployments.openpi.training import prepared_writing_sequence as sequence


def _stage(number: int, tmp_path: Path) -> sequence.WritingStage:
    return sequence.WritingStage(
        number=number,
        prepared_profile=tmp_path / f"prepared-{number}.json",
        model_rtc_profile=tmp_path / f"model-rtc-{number}.json",
    )


def _state(
    stage: sequence.WritingStage,
    *,
    action: str,
    latest: int | None,
    target: int = 150_000,
) -> sequence.WritingStageState:
    return sequence.WritingStageState(
        stage=stage,
        checkpoint_root=Path(f"/checkpoints/stage-{stage.number}"),
        target_step=target,
        latest_checkpoint_step=latest,
        action=action,
    )


def test_checked_in_writing_stage_profiles_form_one_model_rtc_sequence() -> None:
    prepared_ids: list[str] = []
    model_ids: list[str] = []
    salts: list[str] = []

    for expected_number, stage in enumerate(sequence.DEFAULT_STAGES, start=1):
        prepared = sequence._load_profile(stage.prepared_profile)
        model_rtc = load_prepared_model_rtc_profile(stage.model_rtc_profile)

        assert stage.number == expected_number
        assert model_rtc["prepared_skill_profile_id"] == prepared["profile_id"]
        prepared_ids.append(str(prepared["profile_id"]))
        model_ids.append(str(model_rtc["output_config_name"]))
        salts.append(str(model_rtc["sampling_spec"]["selection_salt"]))

    assert len(set(prepared_ids)) == 4
    assert len(set(model_ids)) == 4
    assert len(set(salts)) == 4


def test_inspect_stage_verifies_latest_checkpoint_before_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = _stage(1, tmp_path)
    root = tmp_path / "checkpoints"
    verified: list[tuple[Path, str, str]] = []
    resume_preflights: list[tuple[Path, bool]] = []
    monkeypatch.setattr(
        sequence,
        "_load_profile",
        lambda _path: {
            "profile_id": "prepared-v1",
            "openpi_checkout": "/openpi",
            "openpi_base_config": "base",
        },
    )
    monkeypatch.setattr(
        sequence,
        "load_prepared_model_rtc_profile",
        lambda _path: {
            "prepared_skill_profile_id": "prepared-v1",
            "profile_id": "model-rtc-source-v1",
            "output_config_name": "model-rtc-output-v1",
        },
    )
    monkeypatch.setattr(sequence, "_model_rtc_checkpoint_root", lambda *_a, **_k: root)
    monkeypatch.setattr(sequence, "_target_train_step", lambda _profile: 150_000)
    monkeypatch.setattr(sequence, "_numbered_checkpoint_steps", lambda _root: (0, 60_000, 60_069))
    monkeypatch.setattr(
        sequence,
        "_verify_stage_checkpoint",
        lambda path, *, expected_model_id, expected_source_id: verified.append(
            (path, expected_model_id, expected_source_id)
        ),
    )
    monkeypatch.setattr(
        sequence,
        "_load_checkpoint_bound_prepared_resume",
        lambda _prepared, _model_rtc, *, checkpoint_root, verify_source_files: (
            resume_preflights.append((checkpoint_root, verify_source_files))
        ),
    )

    state = sequence.inspect_stage(stage)

    assert state.action == "resume"
    assert state.latest_checkpoint_step == 60_069
    assert verified == [
        (root / "60069", "model-rtc-output-v1", "model-rtc-source-v1")
    ]
    assert resume_preflights == [(root, False)]


def test_sequence_resumes_then_starts_next_stage_only_after_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stages = (_stage(1, tmp_path), _stage(2, tmp_path))
    states = iter(
        (
            _state(stages[0], action="resume", latest=60_069),
            _state(stages[0], action="complete", latest=150_000),
            _state(stages[1], action="fresh", latest=None),
            _state(stages[1], action="complete", latest=150_000),
        )
    )
    launched: list[list[str]] = []
    monkeypatch.setattr(sequence, "inspect_stage", lambda _stage: next(states))
    monkeypatch.setattr(
        sequence,
        "_run_stage_process",
        lambda argv: launched.append(list(argv)) or 0,
    )
    monkeypatch.setattr(sequence.sys.stdin, "isatty", lambda: True)

    assert sequence.run_sequence(stages) == 0

    assert len(launched) == 2
    assert "--resume" in launched[0]
    assert "--resume" not in launched[1]


def test_sequence_stops_when_clean_child_exit_is_still_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stages = (_stage(1, tmp_path), _stage(2, tmp_path))
    inspected: list[int] = []
    states = iter(
        (
            _state(stages[0], action="resume", latest=60_069),
            _state(stages[0], action="resume", latest=90_000),
        )
    )

    def inspect(stage: sequence.WritingStage) -> sequence.WritingStageState:
        inspected.append(stage.number)
        return next(states)

    monkeypatch.setattr(sequence, "inspect_stage", inspect)
    monkeypatch.setattr(sequence, "_run_stage_process", lambda _argv: 0)
    monkeypatch.setattr(sequence.sys.stdin, "isatty", lambda: True)

    assert sequence.run_sequence(stages) == 0
    assert inspected == [1, 1]


def test_stage_process_inherits_stdio_for_openpi_terminal_save(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    popen_calls: list[tuple[list[str], dict[str, object]]] = []
    signal_calls: list[tuple[object, object]] = []

    class _Process:
        @staticmethod
        def wait() -> int:
            return 0

    def popen(argv: list[str], **kwargs: object) -> _Process:
        popen_calls.append((argv, kwargs))
        return _Process()

    def set_signal(signum: object, handler: object) -> object:
        signal_calls.append((signum, handler))
        return types.SimpleNamespace(name="previous")

    monkeypatch.setattr(sequence.subprocess, "Popen", popen)
    monkeypatch.setattr(sequence.signal, "signal", set_signal)

    assert sequence._run_stage_process(["python", "train"]) == 0
    assert popen_calls == [(["python", "train"], {})]
    assert signal_calls[0] == (sequence.signal.SIGINT, sequence.signal.SIG_IGN)
    assert signal_calls[1][0] == sequence.signal.SIGINT


def test_sequence_dry_run_never_starts_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stages = (_stage(1, tmp_path),)
    monkeypatch.setattr(
        sequence,
        "inspect_stage",
        lambda stage: _state(stage, action="resume", latest=60_069),
    )
    monkeypatch.setattr(
        sequence,
        "_run_stage_process",
        lambda _argv: pytest.fail("dry-run must not start training"),
    )

    assert sequence.run_sequence(stages, dry_run=True) == 0
    output = capsys.readouterr().out
    assert "dry-run complete" in output
    assert "all four writing model-RTC stages are complete" not in output
