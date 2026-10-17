from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _executable(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -Eeuo pipefail\n" + body)
    path.chmod(0o755)


@pytest.fixture
def wrapper_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "ssh-calls"
    _executable(
        bin_dir / "ssh",
        'printf "%s\\0" "$@" > "$SSH_CAPTURE"\nexec bash -c "${!#}"\n',
    )
    environment = os.environ.copy()
    for name in ("REMOTE_HOST", "REMOTE_USER", "REMOTE_PORT", "KEY_FILE"):
        environment.pop(name, None)
    environment.update(
        PATH=f"{bin_dir}:{environment['PATH']}",
        SSH_CAPTURE=str(calls),
        KEY_FILE=str(tmp_path / "absent-key"),
    )
    return environment, calls


def _run(
    environment: dict[str, str], wrapper: str, *args: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(ROOT / "codex_remote" / wrapper), *args],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_readonly_snapshot_preview_never_connects(wrapper_environment):
    environment, calls = wrapper_environment
    result = _run(environment, "remote_rm65_grasp_snapshot.sh", "-H", "trusted-target",
        "--repo", "/repo", "--python", "/python", "--camera-config", "/cameras.json",
        "--arm-ip-assignment", "/arms.json", "--output", "/new-evidence", "--dry-run")
    assert result.returncode == 0
    assert not calls.exists()
    assert "60s" in result.stdout


def test_readonly_snapshot_rejects_arbitrary_command(wrapper_environment):
    environment, calls = wrapper_environment
    result = _run(environment, "remote_rm65_grasp_snapshot.sh", "-H", "trusted-target", "--command", "move")
    assert result.returncode != 0
    assert not calls.exists()


def test_readonly_preflight_preview_keeps_context_and_never_connects(wrapper_environment):
    environment, calls = wrapper_environment
    result = _run(environment, "remote_rm65_grasp_snapshot.sh", "-H", "trusted-target",
        "--repo", "/repo", "--python", "/python", "--camera-config", "/cameras.json",
        "--arm-ip-assignment", "/arms.json", "--output", "/new-evidence",
        "--camera-registry", "/registry.json", "--camera-mount-urdf", "/cad.urdf",
        "--left-tool-profile", "/left.json", "--right-tool-profile", "/right.json",
        "--check-only", "--dry-run")
    assert result.returncode == 0
    assert not calls.exists()
    assert "--check-only" in result.stdout and "--interactive" in result.stdout
    assert "/cad.urdf" in result.stdout


@pytest.mark.parametrize("wrapper", ["remote_python.sh", "remote_module.sh", "remote_pytest.sh"])
def test_generic_execution_rejects_control_authority_overrides(
    wrapper_environment: tuple[dict[str, str], Path], wrapper: str
) -> None:
    environment, calls = wrapper_environment
    result = _run(
        environment,
        wrapper,
        "-H",
        "trusted-target",
        "--allow-control-risk",
        "--",
        "harmless",
    )
    assert result.returncode != 0
    assert "no longer supported by generic L3 wrappers" in result.stderr
    assert not calls.exists()


@pytest.mark.parametrize(
    ("wrapper", "argument"),
    [
        ("remote_python.sh", "robot_motion/replay.py"),
        ("remote_module.sh", "robot_motion.replay"),
        ("remote_pytest.sh", "tests/test_robot_motion.py"),
    ],
)
def test_generic_execution_rejects_control_risk_before_ssh(
    wrapper_environment: tuple[dict[str, str], Path], wrapper: str, argument: str
) -> None:
    environment, calls = wrapper_environment
    result = _run(environment, wrapper, "-H", "trusted-target", "--", argument)
    assert result.returncode == 5
    assert "Refusing a control-risk command through a generic L3 wrapper" in result.stderr
    assert not calls.exists()


def test_preflight_cannot_write_an_authority_token(
    wrapper_environment: tuple[dict[str, str], Path], tmp_path: Path
) -> None:
    environment, calls = wrapper_environment
    result = _run(
        environment,
        "remote_safety_check.sh",
        "-H",
        "trusted-target",
        "--token-path",
        str(tmp_path / "authority-token"),
    )
    assert result.returncode != 0
    assert "preflight evidence cannot authorize a later action" in result.stderr
    assert not calls.exists()


def test_arm_dry_run_only_previews_the_remote_program(
    wrapper_environment: tuple[dict[str, str], Path]
) -> None:
    environment, calls = wrapper_environment
    result = _run(
        environment,
        "remote_arm_dry_run.sh",
        "-H",
        "trusted-target",
        "--script",
        "tools/would_move_dry_run.py",
    )
    assert result.returncode == 0, result.stderr
    assert "tools/would_move_dry_run.py --dry-run" in result.stdout
    assert "[result] dry-run only" in result.stdout
    transport = calls.read_bytes().split(b"\0")
    assert b"StrictHostKeyChecking=yes" in transport


@pytest.mark.parametrize(
    ("option", "value"),
    [("-H", "trusted\x1b[31m"), ("-u", "robot\x07")],
)
def test_control_characters_are_rejected_before_ssh(
    wrapper_environment: tuple[dict[str, str], Path], option: str, value: str
) -> None:
    environment, calls = wrapper_environment
    arguments = ["-H", "trusted-target"] if option == "-u" else []
    result = _run(
        environment,
        "remote_python.sh",
        *arguments,
        option,
        value,
        "--dry-run",
        "--",
        "--version",
    )
    assert result.returncode != 0
    assert "unsafe or invalid SSH" in result.stderr
    assert not calls.exists()


def test_readonly_agent_preview_is_fixed_and_never_connects(wrapper_environment):
    environment, calls = wrapper_environment
    args = ("-H", "trusted-target", "--repo", "/repo", "--python", "/python",
        "--rpent-repo", "/upper", "--rpent-python", "/upper-python",
        "--camera-config", "/cameras.json", "--camera-registry", "/registry.json",
        "--localization-bundle", "/bundle", "--output", "/new-evidence", "--hf-home", "/cache-root", "--dry-run")
    result = _run(environment, "remote_readonly_agent_acceptance.sh", *args)
    assert result.returncode == 0
    assert not calls.exists()
    assert "300s" in result.stdout and "run_readonly_agent_acceptance.py" in result.stdout
    assert "--hf-home '/cache-root'" in result.stdout
    assert "--camera-scope 'all'" in result.stdout
    debug = _run(environment, "remote_readonly_agent_acceptance.sh", *args, "--camera-sdk-debug-log")
    assert debug.returncode == 0 and "--camera-sdk-debug-log" in debug.stdout
    assert not calls.exists()
    main = _run(environment, "remote_readonly_agent_acceptance.sh", *args, "--camera-scope", "main")
    assert main.returncode == 0 and "--camera-scope 'main'" in main.stdout
    invalid = _run(environment, "remote_readonly_agent_acceptance.sh", *args, "--camera-scope", "automatic")
    assert invalid.returncode != 0
    result = _run(environment, "remote_readonly_agent_acceptance.sh", *args, "--command", "move")
    assert result.returncode != 0
    assert not calls.exists()


def test_readonly_tcp_agent_wrapper_is_fixed_without_camera_or_control(wrapper_environment):
    environment, calls = wrapper_environment
    args = ("-H", "trusted-target", "--repo", "/repo", "--python", "/python",
        "--rpent-repo", "/upper", "--rpent-python", "/upper-python",
        "--acceptance-scope", "tcp", "--arm-ip-assignment", "/ips.json",
        "--left-tool-profile", "/left.json", "--right-tool-profile", "/right.json",
        "--envelope-profile", "/envelope.json", "--output", "/new-run", "--dry-run")
    result = _run(environment, "remote_readonly_agent_acceptance.sh", *args)
    assert result.returncode == 0
    assert "--acceptance-scope 'tcp'" in result.stdout
    assert "--arm-ip-assignment '/ips.json'" in result.stdout
    assert "--camera-config" not in result.stdout
    assert not calls.exists()
    for extra in (("--camera-config", "/cam"), ("--camera-sdk-debug-log",), ("--command", "move")):
        assert _run(environment, "remote_readonly_agent_acceptance.sh", *args, *extra).returncode != 0
    assert not calls.exists()
