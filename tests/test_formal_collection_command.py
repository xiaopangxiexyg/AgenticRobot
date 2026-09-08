from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from collection_runtime.progress.profiles import load_managed_task_profile
from teleop_runtime.real_teleop.cli import build_real_teleop_argument_parser
from teleop_runtime.real_teleop.defaults import (
    DEFAULT_COLLECTION_PROCESS_STARTUP_TIMEOUT_SEC,
)
from teleop_runtime.real_teleop.progress_adapter import (
    resolve_real_teleop_progress_entry,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPOSITORY_ROOT / "scripts" / "run_formal_60hz_writing_collection.sh"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(LAUNCHER), *args],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_real_teleop_process_writer_startup_timeout_is_bounded_and_tunable() -> None:
    parser = build_real_teleop_argument_parser()

    defaults = parser.parse_args([])
    tuned = parser.parse_args(
        ["--collection-process-startup-timeout-sec", "90"]
    )

    assert defaults.collection_process_startup_timeout_sec == (
        DEFAULT_COLLECTION_PROCESS_STARTUP_TIMEOUT_SEC
    )
    assert tuned.collection_process_startup_timeout_sec == 90.0
    with pytest.raises(SystemExit):
        parser.parse_args(["--collection-process-startup-timeout-sec", "0"])


def test_launcher_help_documents_two_terminal_commands() -> None:
    result = _run("--help")

    assert result.returncode == 0, result.stderr
    assert "foot-pad" in result.stdout
    assert "collect" in result.stdout
    assert "--tactile-bundle-root" in result.stdout
    assert "--entry-step" in result.stdout
    assert "--gripper-input-source" in result.stdout


def test_five_pedal_dry_run_is_fixed_to_count_five() -> None:
    result = _run("foot-pad", "--dry-run")

    assert result.returncode == 0, result.stderr
    assert "python -m hardware.foot_pad --count 5" in result.stdout


def test_collection_dry_run_fixes_control_progress_and_tactile_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("GRIPPER_WARMUP_PROBE", raising=False)
    monkeypatch.delenv("FORMAL_GRIPPER_INPUT_SOURCE", raising=False)
    tactile_root = tmp_path / "tactile"
    tactile_root.mkdir()
    dataset_root = tmp_path / "not-created-dataset-root"

    result = _run(
        "collect",
        "--dry-run",
        "--dataset-root",
        str(dataset_root),
        "--tactile-bundle-root",
        str(tactile_root),
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert "POSITION_FORMAL" not in output
    assert "--position-admittance-" not in output
    assert (
        "--position-control-profile "
        "rm65_bimanual_position_v2_balanced_60hz_jerk_v2"
    ) in output
    assert "--collection-task-name 4_stage_writing_detail" in output
    assert (
        "--collection-progress-profile "
        "catalog:dmtac_rm65_bimanual_position_writing_four_stage_progress"
    ) in output
    assert "--collection-progress-gui" in output
    assert "--gripper-input-source q6" in output
    assert "--gripper-input-source angle_trigger" not in output
    assert "--q6-config latest" in output
    assert "--q6-config-required" in output
    assert "--tactile-storage external" in output
    assert f"--tactile-bundle-root {tactile_root}" in output
    assert "--tactile-artifact-integrity" in output
    assert "--idle-tactile-compression" in output
    assert "--no-gripper-warmup-probe" in output


def test_collection_can_explicitly_select_angle_trigger_without_q6_probe(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("FORMAL_GRIPPER_INPUT_SOURCE", raising=False)
    tactile_root = tmp_path / "tactile"
    tactile_root.mkdir()

    result = _run(
        "collect",
        "--dry-run",
        "--tactile-bundle-root",
        str(tactile_root),
        "--gripper-input-source",
        "angle_trigger",
    )

    assert result.returncode == 0, result.stderr
    assert "--gripper-input-source angle_trigger" in result.stdout
    assert "--gripper-input-source q6" not in result.stdout
    assert "--q6-config" not in result.stdout


def test_collection_gripper_source_cli_overrides_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("FORMAL_GRIPPER_INPUT_SOURCE", "angle_trigger")
    tactile_root = tmp_path / "tactile"
    tactile_root.mkdir()

    inherited = _run(
        "collect",
        "--dry-run",
        "--tactile-bundle-root",
        str(tactile_root),
    )
    explicit = _run(
        "collect",
        "--dry-run",
        "--tactile-bundle-root",
        str(tactile_root),
        "--gripper-input-source",
        "q6",
    )

    assert inherited.returncode == 0, inherited.stderr
    assert "--gripper-input-source angle_trigger" in inherited.stdout
    assert explicit.returncode == 0, explicit.stderr
    assert "--gripper-input-source q6" in explicit.stdout
    assert "--q6-config-required" in explicit.stdout


def test_collection_rejects_unknown_gripper_input_source(tmp_path: Path) -> None:
    tactile_root = tmp_path / "tactile"
    tactile_root.mkdir()

    result = _run(
        "collect",
        "--dry-run",
        "--tactile-bundle-root",
        str(tactile_root),
        "--gripper-input-source",
        "auto",
    )

    assert result.returncode == 2
    assert "must be q6 or angle_trigger" in result.stderr


def test_collection_environment_selects_task_name_and_detailed_profile(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tactile_root = tmp_path / "tactile"
    tactile_root.mkdir()
    monkeypatch.setenv("COLLECTION_TASK_NAME", "3_stage_writing_detail")
    monkeypatch.setenv(
        "COLLECTION_PROGRESS_PROFILE",
        "catalog:dmtac_rm65_bimanual_position_writing_four_stage_progress_detailed",
    )

    result = _run(
        "collect",
        "--dry-run",
        "--tactile-bundle-root",
        str(tactile_root),
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert "--collection-task-name 3_stage_writing_detail" in output
    assert (
        "--collection-progress-profile "
        "catalog:dmtac_rm65_bimanual_position_writing_four_stage_progress"
    ) in output
    assert (
        "--collection-stage-profile "
        "catalog:dmtac_rm65_bimanual_position_writing_four_stage_progress_detailed"
    ) in output


def test_collection_selects_five_stage_detail_and_starts_at_dipping(
    tmp_path: Path,
) -> None:
    tactile_root = tmp_path / "tactile"
    tactile_root.mkdir()

    result = _run(
        "collect",
        "--dry-run",
        "--tactile-bundle-root",
        str(tactile_root),
        "--task-name",
        "5_stage_writing_detail",
        "--progress-profile",
        "catalog:dmtac_rm65_bimanual_position_writing_five_stage_progress_detailed",
        "--entry-step",
        "right_dip_ink",
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert "--collection-task-name 5_stage_writing_detail" in output
    assert (
        "--collection-progress-profile "
        "catalog:dmtac_rm65_bimanual_position_writing_five_stage_progress"
    ) in output
    assert (
        "--collection-stage-profile "
        "catalog:dmtac_rm65_bimanual_position_writing_five_stage_progress_detailed"
    ) in output
    assert "--collection-progress-entry-step right_dip_ink" in output

    detailed = load_managed_task_profile(
        "dmtac_rm65_bimanual_position_writing_five_stage_progress_detailed"
    )
    assert detailed.stage_profile is not None
    assert resolve_real_teleop_progress_entry(
        progress_profile=detailed.progress_profile,
        stage_profile=detailed.stage_profile,
        entry_step_id="right_dip_ink",
    ) == ("right_dip_ink", "right_dip_ink.approach", "NOMINAL")


def test_collection_can_explicitly_enable_gripper_motion_probe(
    tmp_path: Path,
) -> None:
    tactile_root = tmp_path / "tactile"
    tactile_root.mkdir()

    result = _run(
        "collect",
        "--dry-run",
        "--tactile-bundle-root",
        str(tactile_root),
        "--gripper-warmup-probe",
    )

    assert result.returncode == 0, result.stderr
    assert "--gripper-warmup-probe" in result.stdout
    assert "--no-gripper-warmup-probe" not in result.stdout


def test_collection_requires_explicit_tactile_root(monkeypatch) -> None:
    monkeypatch.delenv("TACTILE_BUNDLE_ROOT", raising=False)

    result = _run("collect", "--dry-run")

    assert result.returncode == 2
    assert "--tactile-bundle-root" in result.stderr


def test_collection_accepts_existing_roots_on_same_filesystem(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "datasets"
    tactile_root = tmp_path / "tactile"
    dataset_root.mkdir()
    tactile_root.mkdir()

    result = _run(
        "collect",
        "--dry-run",
        "--dataset-root",
        str(dataset_root),
        "--tactile-bundle-root",
        str(tactile_root),
    )

    assert result.returncode == 0, result.stderr
    assert "--tactile-bundle-root" in result.stdout


def test_collection_minimum_free_space_precedence_is_visible(
    monkeypatch, tmp_path: Path
) -> None:
    tactile_root = tmp_path / "tactile"
    tactile_root.mkdir()
    monkeypatch.setenv("TACTILE_BUNDLE_MIN_FREE_GB", "200")

    inherited = _run(
        "collect",
        "--dry-run",
        "--tactile-bundle-root",
        str(tactile_root),
    )
    explicit = _run(
        "collect",
        "--dry-run",
        "--tactile-bundle-root",
        str(tactile_root),
        "--tactile-min-free-gb",
        "2",
    )

    assert inherited.returncode == 0, inherited.stderr
    assert "--tactile-bundle-min-free-gb 200" in inherited.stdout
    assert "tactile_bundle_min_free_gb=200 source=environment" in inherited.stdout
    assert explicit.returncode == 0, explicit.stderr
    assert "--tactile-bundle-min-free-gb 2" in explicit.stdout
    assert "tactile_bundle_min_free_gb=2 source=command_line" in explicit.stdout
