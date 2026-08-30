from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "codex_remote/remote_harness.sh"


def _write_executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _environment(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "harness-args.bin"
    remote_harness = tmp_path / "home" / "HarnessConfig" / "bin" / "harness"
    _write_executable(
        fake_bin / "ssh",
        """#!/usr/bin/env bash
set -Eeuo pipefail
exec bash -c "${!#}"
""",
    )
    _write_executable(
        remote_harness,
        """#!/usr/bin/env bash
set -Eeuo pipefail
printf '%s\\0' "$@" > "$FAKE_HARNESS_CAPTURE"
""",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "FAKE_HARNESS_CAPTURE": str(capture),
        }
    )
    return environment, capture, remote_harness


def _captured(path: Path) -> list[str]:
    return [item.decode() for item in path.read_bytes().split(b"\0") if item]


def _base(remote_harness: Path) -> list[str]:
    return [
        "bash",
        str(WRAPPER),
        "-H",
        "user@192.168.70.116",
        "--harness",
        str(remote_harness),
    ]


def test_runner_preflight_is_forwarded_to_host_private_wrapper(tmp_path: Path) -> None:
    environment, capture, remote_harness = _environment(tmp_path)

    result = subprocess.run(
        [
            *_base(remote_harness),
            "runner",
            "preflight",
            "agentic-source-runner",
            "--profile",
            "agenticrobot-integration",
            "--expected-control-commit",
            "a" * 40,
            "--json",
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "command_kind=runner-preflight" in result.stdout
    assert _captured(capture) == [
        "runner",
        "preflight",
        "agentic-source-runner",
        "--profile",
        "agenticrobot-integration",
        "--expected-control-commit",
        "a" * 40,
        "--json",
    ]


def test_exchange_run_preserves_each_argument_without_shell_expansion(
    tmp_path: Path,
) -> None:
    environment, capture, remote_harness = _environment(tmp_path)
    exchange = tmp_path / "exchange dir;not-a-command"
    expansion_marker = tmp_path / "shell-expansion-must-not-run"
    literal_argument = f"$(touch {expansion_marker})"

    result = subprocess.run(
        [
            *_base(remote_harness),
            "transfer",
            "exchange-run",
            str(exchange),
            "--runner-id",
            "agentic-source-runner",
            "--slot-id",
            "source-slot-01",
            "--sender-id",
            "local-controller",
            "--recipient-id",
            "agentic-source-node",
            "--key-id",
            "agentic-transfer-v1",
            "--key-file",
            "/private/transfer.key",
            "--literal-probe",
            literal_argument,
            "--json",
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "command_kind=exchange-run" in result.stdout
    assert _captured(capture) == [
        "transfer",
        "exchange-run",
        str(exchange),
        "--runner-id",
        "agentic-source-runner",
        "--slot-id",
        "source-slot-01",
        "--sender-id",
        "local-controller",
        "--recipient-id",
        "agentic-source-node",
        "--key-id",
        "agentic-transfer-v1",
        "--key-file",
        "/private/transfer.key",
        "--literal-probe",
        literal_argument,
        "--json",
    ]
    assert not (tmp_path / "not-a-command").exists()
    assert not expansion_marker.exists()


def test_wrapper_rejects_generic_commands_and_implicit_host(tmp_path: Path) -> None:
    environment, capture, remote_harness = _environment(tmp_path)
    environment["REMOTE_HOST"] = "ambient-host-must-not-count"
    cases = (
        ([*_base(remote_harness), "status", "--json"], "unsupported Harness command"),
        (
            [*_base(remote_harness), "control", "install", "--json"],
            "unsupported Harness command",
        ),
        (
            [*_base(remote_harness), "transfer", "exchange-create", "request"],
            "unsupported Harness command",
        ),
        (
            [
                "bash",
                str(WRAPPER),
                "--harness",
                str(remote_harness),
                "runner",
                "preflight",
                "runner",
            ],
            "missing -H/--host",
        ),
        (
            [
                "bash",
                str(WRAPPER),
                "-H",
                "user@host",
                "--harness",
                "/tmp/harness",
                "runner",
                "preflight",
                "runner",
            ],
            "host-private",
        ),
    )
    for command, expected in cases:
        if capture.exists():
            capture.unlink()
        result = subprocess.run(
            command,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert expected in result.stderr
        assert not capture.exists()


def test_wrapper_rejects_unsafe_host_private_paths_and_ssh_targets(
    tmp_path: Path,
) -> None:
    environment, capture, remote_harness = _environment(tmp_path)
    unsafe_paths = (
        str(remote_harness).replace("/HarnessConfig/", "//HarnessConfig/"),
        str(remote_harness).replace("/HarnessConfig/", "/parent/../HarnessConfig/"),
        str(remote_harness).replace("/HarnessConfig/", "/./HarnessConfig/"),
        f"{remote_harness}\n",
    )
    commands = [
        [
            "bash",
            str(WRAPPER),
            "-H",
            "user@host",
            "--harness",
            path,
            "runner",
            "preflight",
            "runner",
        ]
        for path in unsafe_paths
    ]
    commands.extend(
        [
            [
                "bash",
                str(WRAPPER),
                "-H",
                "-oProxyCommand=not-allowed",
                "--harness",
                str(remote_harness),
                "runner",
                "preflight",
                "runner",
            ],
            [
                "bash",
                str(WRAPPER),
                "-H",
                "-oUser@host",
                "--harness",
                str(remote_harness),
                "runner",
                "preflight",
                "runner",
            ],
        ]
    )

    for command in commands:
        if capture.exists():
            capture.unlink()
        result = subprocess.run(
            command,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "unsafe" in result.stderr
        assert not capture.exists()


def test_remote_wrapper_must_be_regular_executable(tmp_path: Path) -> None:
    environment, capture, remote_harness = _environment(tmp_path)
    target = remote_harness.with_name("real-harness")
    remote_harness.rename(target)
    remote_harness.symlink_to(target)

    result = subprocess.run(
        [*_base(remote_harness), "runner", "preflight", "runner"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "missing or unsafe" in result.stderr
    assert not capture.exists()
