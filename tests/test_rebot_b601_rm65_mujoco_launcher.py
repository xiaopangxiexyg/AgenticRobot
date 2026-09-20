from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.hardware import rebot_b601_rm65_mujoco_follow as launcher
from scripts.hardware import rebot_b601_rs_low_gain_probe as low_gain_probe
from scripts.hardware import rebot_b601_rs_owner as owner

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def test_launcher_defaults_are_workspace_local_and_auto_activate() -> None:
    args = launcher._parser().parse_args([])

    assert args.auto_activate is True
    paths = launcher.resolve_paths(args)
    assert paths.repo_root == launcher.REPO_ROOT
    assert paths.mapping_profile.name == "b601_rm65_mapping_150_v2.json"
    assert paths.task_config.name == "rm65_right_cube_lift_rl_normal_v1.json"
    assert paths.config.name == "rebotarm_rs.yaml"
    assert paths.rebot_source_root == launcher.REPO_ROOT / "hardware" / "reBotArm_control_py"
    assert owner._parser().parse_args([]).rebot_source_root == (
        owner.REPO_ROOT / "hardware" / "reBotArm_control_py"
    )


def test_launcher_builds_canonical_read_only_command_without_shell(tmp_path: Path) -> None:
    args = launcher._parser().parse_args(
        [
            "--controlled-rpy-deg",
            "0",
            "0",
            "90",
            "--reference-matrix",
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


def test_launcher_rotation_aliases_feed_the_canonical_application_options() -> None:
    args = launcher._parser().parse_args(
        [
            "--controlled-matrix",
            "1",
            "0",
            "0",
            "0",
            "1",
            "0",
            "0",
            "0",
            "1",
            "--reference-rpy-deg",
            "0",
            "0",
            "90",
        ]
    )
    assert args.mapping_controlled_rotation_matrix == [
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    assert args.mapping_reference_rotation_rpy_deg == [0.0, 0.0, 90.0]


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


def test_owner_launcher_requires_explicit_motion_acknowledgement() -> None:
    args = owner._parser().parse_args([])
    assert args.allow_motion is False
    assert args.mode == "hold"
    assert args.feedback_source == "active-report"
    assert args.driver_kd == owner.DEFAULT_DRIVER_KD


def test_owner_launcher_reuses_guarded_probe_and_standard_limits(tmp_path: Path) -> None:
    args = owner._parser().parse_args(["--allow-motion"])
    command = owner.build_probe_command(args, run_dir=tmp_path)

    assert command[0] == launcher.sys.executable
    assert command[1].endswith("scripts/hardware/rebot_b601_rs_low_gain_probe.py")
    assert "--allow-motion" in command
    assert "--auto-disable" in command
    assert "--allow-active-report" in command
    assert "--auto-disable-active-report" in command
    assert "--apply-driver-torque-limits" in command
    assert (
        command[command.index("--confirmation") + 1]
        == low_gain_probe.CONFIRMATION
    )
    assert command[command.index("--velocity-abort-rad-s") + 1] == "5.0"
    assert command[command.index("--acceleration-abort-rad-s2") + 1] == "80.0"

    # Re-parse the exact generated argv through the real probe contract.  This
    # is deliberately offline: _validate_cli performs no CAN or motor I/O.
    parsed_probe_args = low_gain_probe.build_parser().parse_args(command[2:])
    low_gain_probe._validate_cli(parsed_probe_args)


def test_owner_launcher_builds_one_shot_assist_inside_existing_owner(
    tmp_path: Path,
) -> None:
    args = owner._parser().parse_args(
        [
            "--allow-motion",
            "--single-joint-assist",
            "--assist-joint",
            "joint6",
            "--assist-offset-rad",
            "0.1",
            "--assist-min-response-rad",
            "0.002",
            "--assist-confirmation",
            owner.DEFAULT_ASSIST_CONFIRMATION,
        ]
    )
    command = owner.build_probe_command(args, run_dir=tmp_path)

    assert "--single-joint-assist" in command
    assert command[command.index("--assist-joint") + 1] == "joint6"
    assert command[command.index("--assist-offset-rad") + 1] == "0.1"
    assert command[command.index("--assist-min-response-rad") + 1] == "0.002"
    assert (
        command[command.index("--assist-confirmation") + 1]
        == low_gain_probe.SINGLE_JOINT_ASSIST_CONFIRMATION
    )
    parsed_probe_args = low_gain_probe.build_parser().parse_args(command[2:])
    low_gain_probe._validate_cli(parsed_probe_args)
    runtime, session = low_gain_probe._build_single_joint_assist(parsed_probe_args)
    assert session.runtime is runtime
    low_gain_probe._validate_shared_control_integration(parsed_probe_args, runtime)


def test_owner_launcher_rejects_assist_without_second_confirmation() -> None:
    with pytest.raises(SystemExit) as error:
        owner.main(["--allow-motion", "--single-joint-assist"])
    assert error.value.code == 2


def test_owner_launcher_rejects_missing_motion_ack_before_resource_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(owner, "DEFAULT_CONFIG", tmp_path / "missing.yaml")
    with pytest.raises(SystemExit) as error:
        owner.main([])
    assert error.value.code == 2


def test_foreground_launcher_does_not_duplicate_sigint_or_close_output_early(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed_handler: object | None = None
    old_handler = object()

    def fake_signal(_signum: int, handler: object) -> object:
        nonlocal installed_handler
        previous = old_handler if installed_handler is None else installed_handler
        installed_handler = handler
        return previous

    class InterruptingOutput:
        def __init__(self) -> None:
            self.calls = 0
            self.closed = False

        def readline(self) -> str:
            self.calls += 1
            if self.calls == 1:
                assert callable(installed_handler)
                installed_handler(launcher.signal.SIGINT, None)
                return "child result after Ctrl-C\n"
            return ""

        def close(self) -> None:
            self.closed = True

    output = InterruptingOutput()
    process = SimpleNamespace(stdout=output, wait=lambda: 1)
    monkeypatch.setattr(launcher.signal, "getsignal", lambda _signum: old_handler)
    monkeypatch.setattr(launcher.signal, "signal", fake_signal)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *args, **kwargs: process)

    status = launcher._run_foreground(
        ["fake-child"],
        cwd=tmp_path,
        terminal_log=tmp_path / "terminal.log",
    )

    assert status == 1
    assert output.closed is True
    assert installed_handler is old_handler
    log = (tmp_path / "terminal.log").read_text(encoding="utf-8")
    assert "child result after Ctrl-C" in log
    assert "child owned the signal lifecycle" in log
    assert "forwarding SIGINT" not in log
