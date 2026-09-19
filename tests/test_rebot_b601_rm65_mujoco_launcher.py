from __future__ import annotations

from pathlib import Path

import pytest

from scripts.hardware import rebot_b601_rm65_mujoco_follow as launcher

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def test_launcher_defaults_are_workspace_local_and_auto_activate() -> None:
    args = launcher._parser().parse_args([])

    assert args.auto_activate is True
    paths = launcher.resolve_paths(args)
    assert paths.repo_root == launcher.REPO_ROOT
    assert paths.mapping_profile.name == "b601_rm65_mapping_150_v2.json"
    assert paths.task_config.name == "rm65_right_cube_lift_rl_normal_v1.json"
    assert paths.config.name == "rebotarm_rs.yaml"


def test_launcher_builds_canonical_read_only_command_without_shell(tmp_path: Path) -> None:
    args = launcher._parser().parse_args(
        [
            "--mapping-controlled-rotation-rpy-deg",
            "0",
            "0",
            "90",
            "--mapping-reference-rotation-matrix",
            "0",
            "1",
            "0",
            "-1",
            "0",
            "0",
            "0",
            "0",
            "1",
        ]
    )
    paths = launcher.LauncherPaths(
        repo_root=tmp_path / "repo",
        config=tmp_path / "rebot.yaml",
        rebot_source_root=tmp_path / "vendor",
        task_config=tmp_path / "task.json",
        mapping_profile=tmp_path / "mapping.json",
        output_root=tmp_path,
    )
    command = launcher.build_control_station_command(
        args,
        paths=paths,
        run_dir=tmp_path / "run",
    )

    assert command[:5] == [
        launcher.sys.executable,
        "-m",
        "control_station",
        "b601-rm65",
        "--connect-read-only",
    ]
    assert "--auto-activate" in command
    assert "--disable-teleop-residual-envelope" in command
    assert command[command.index("--mapping-controlled-rotation-rpy-deg") + 1 :]
    assert command[command.index("--mapping-reference-rotation-matrix") + 1 :]
    assert "--shell" not in command
    assert all(";" not in value for value in command)


def test_manual_activation_and_envelope_are_explicit() -> None:
    args = launcher._parser().parse_args(
        ["--manual-activate", "--keep-residual-envelope"]
    )
    paths = launcher.LauncherPaths(
        repo_root=Path("/repo"),
        config=Path("/config.yaml"),
        rebot_source_root=Path("/vendor"),
        task_config=Path("/task.json"),
        mapping_profile=Path("/mapping.json"),
        output_root=Path("/tmp"),
    )
    command = launcher.build_control_station_command(
        args,
        paths=paths,
        run_dir=Path("/tmp/run"),
    )

    assert args.auto_activate is False
    assert "--auto-activate" not in command
    assert "--disable-teleop-residual-envelope" not in command


def test_launcher_help_is_available_without_runtime_imports() -> None:
    parser = launcher._parser()
    help_text = parser.format_help()
    assert "--pause-on-exit" in help_text
    assert "--mapping-controlled-rotation-matrix" in help_text
    assert "receive-only" in help_text
