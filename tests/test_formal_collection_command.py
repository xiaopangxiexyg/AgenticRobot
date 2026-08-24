from __future__ import annotations

from pathlib import Path
import subprocess


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


def test_launcher_help_documents_two_terminal_commands() -> None:
    result = _run("--help")

    assert result.returncode == 0, result.stderr
    assert "foot-pad" in result.stdout
    assert "collect" in result.stdout
    assert "--tactile-bundle-root" in result.stdout


def test_five_pedal_dry_run_is_fixed_to_count_five() -> None:
    result = _run("foot-pad", "--dry-run")

    assert result.returncode == 0, result.stderr
    assert "python -m hardware.foot_pad --count 5" in result.stdout


def test_collection_dry_run_fixes_control_progress_and_tactile_contract(
    tmp_path: Path,
) -> None:
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
    assert "--position-admittance-active" in output
    assert "--fix-arm" in output
    assert "--no-fix-arm" not in output
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
    assert "--tactile-storage external" in output
    assert f"--tactile-bundle-root {tactile_root}" in output
    assert "--tactile-artifact-integrity" in output
    assert "--idle-tactile-compression" in output


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
