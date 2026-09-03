from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "codex_remote/remote_harness.sh"
DEPLOY_WRAPPER = ROOT / "codex_remote/remote_harness_deploy.sh"


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


def _captured_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _base(remote_harness: Path) -> list[str]:
    return [
        "bash",
        str(WRAPPER),
        "-H",
        "user@192.168.70.116",
        "--harness",
        str(remote_harness),
    ]


def _deployment_environment(
    tmp_path: Path,
    *,
    remote_commit: str = "a" * 40,
    origin_url: str = "git@github.com:CYBER-QAQ/HarnessCodingTool.git",
) -> tuple[dict[str, str], Path, Path, Path]:
    fake_bin = tmp_path / "deploy-bin"
    fake_bin.mkdir(parents=True)
    deployment_root = tmp_path / "remote" / "zc"
    source = deployment_root / "HarnessCodingTool"
    project = deployment_root / "AgenticRobot"
    harness = deployment_root / "HarnessConfig" / "bin" / "harness"
    source.mkdir(parents=True)
    project.mkdir()
    capture = tmp_path / "deploy-capture"
    capture.mkdir()

    _write_executable(
        fake_bin / "ssh",
        """#!/usr/bin/env bash
set -Eeuo pipefail
exec bash -c "${!#}"
""",
    )
    _write_executable(
        fake_bin / "git",
        """#!/usr/bin/env bash
set -Eeuo pipefail
while [[ ${1:-} == -c ]]; do shift 2; done
[[ ${1:-} == -C ]] || exit 90
repository=$2
shift 2
case "${1:-} ${2:-}" in
  "rev-parse --show-toplevel") printf '%s\n' "$repository" ;;
  "remote get-url") printf '%s\n' "$FAKE_ORIGIN_URL" ;;
  "fetch --no-tags") printf '%s\n' "$@" > "$FAKE_DEPLOY_CAPTURE/fetch.bin" ;;
  "rev-parse --verify") printf '%s\n' "$FAKE_REMOTE_COMMIT" ;;
  *) printf 'unsupported fake git invocation: %s\n' "$*" >&2; exit 91 ;;
esac
""",
    )
    _write_executable(
        harness,
        """#!/usr/bin/env bash
set -Eeuo pipefail
action=""
install_root=""
confirm=""
for ((index=1; index <= $#; index++)); do
  value=${!index}
  case "$value" in
    plan-install|install|status) action=$value ;;
    --install-root)
      next=$((index + 1))
      install_root=${!next}
      ;;
    --confirm-plan-sha256)
      next=$((index + 1))
      confirm=${!next}
      ;;
  esac
done
[[ -n "$action" ]] || exit 92
printf '%s\n' "$@" > "$FAKE_DEPLOY_CAPTURE/$action.bin"
case "$action" in
  plan-install)
    printf '{"action":"INSTALL_READY","commit":"%s","plan_sha256":"%s"}\n' \
      "$FAKE_REMOTE_COMMIT" "$FAKE_PLAN_SHA256"
    ;;
  install)
    [[ "$confirm" == "$FAKE_PLAN_SHA256" ]] || exit 93
    mkdir -p "$install_root/versions/$FAKE_REMOTE_COMMIT"
    ln -s "$install_root/versions/$FAKE_REMOTE_COMMIT" "$install_root/current"
    printf '{"status":"READY","current_commit":"%s"}\n' "$FAKE_REMOTE_COMMIT"
    ;;
  status)
    printf '{"status":"READY","current_commit":"%s"}\n' "$FAKE_REMOTE_COMMIT"
    ;;
esac
""",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "FAKE_DEPLOY_CAPTURE": str(capture),
            "FAKE_ORIGIN_URL": origin_url,
            "FAKE_PLAN_SHA256": "f" * 64,
            "FAKE_REMOTE_COMMIT": remote_commit,
        }
    )
    return environment, capture, deployment_root, harness


def _deploy_base(deployment_root: Path, expected_commit: str) -> list[str]:
    return [
        "bash",
        str(DEPLOY_WRAPPER),
        "-H",
        "user@192.168.70.116",
        "--root",
        str(deployment_root),
        "--expected-commit",
        expected_commit,
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


def test_deployment_plan_fetches_only_fixed_ref_and_emits_harness_plan(
    tmp_path: Path,
) -> None:
    expected_commit = "a" * 40
    environment, capture, deployment_root, _ = _deployment_environment(tmp_path)

    result = subprocess.run(
        [*_deploy_base(deployment_root, expected_commit), "plan"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "protected_branch=agentic-robot" in result.stdout
    assert f'"plan_sha256":"{"f" * 64}"' in result.stdout
    assert _captured_lines(capture / "fetch.bin") == [
        "fetch",
        "--no-tags",
        "origin",
        "refs/heads/agentic-robot:refs/remotes/origin/agentic-robot",
    ]
    assert _captured_lines(capture / "plan-install.bin") == [
        "--project-root",
        str(deployment_root / "AgenticRobot"),
        "control",
        "plan-install",
        "--install-root",
        str(deployment_root / "HarnessControl"),
        "--source-repository",
        str(deployment_root / "HarnessCodingTool"),
        "--expected-identity",
        "CYBER-QAQ/HarnessCodingTool",
        "--protected-branch",
        "agentic-robot",
        "--json",
    ]
    assert not (capture / "install.bin").exists()


def test_deployment_install_rechecks_plan_and_active_version(tmp_path: Path) -> None:
    expected_commit = "a" * 40
    environment, capture, deployment_root, _ = _deployment_environment(tmp_path)

    result = subprocess.run(
        [
            *_deploy_base(deployment_root, expected_commit),
            "--confirm-plan-sha256",
            "f" * 64,
            "install",
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert f"active_protected_commit={expected_commit}" in result.stdout
    assert _captured_lines(capture / "install.bin")[-5:] == [
        "--protected-branch",
        "agentic-robot",
        "--confirm-plan-sha256",
        "f" * 64,
        "--json",
    ]
    assert _captured_lines(capture / "status.bin") == [
        "--project-root",
        str(deployment_root / "AgenticRobot"),
        "control",
        "status",
        "--install-root",
        str(deployment_root / "HarnessControl"),
        "--json",
    ]
    assert (deployment_root / "HarnessControl/current").resolve() == (
        deployment_root / "HarnessControl" / "versions" / expected_commit
    )


def test_deployment_refuses_moving_commit_and_wrong_origin_before_harness(
    tmp_path: Path,
) -> None:
    expected_commit = "a" * 40
    cases = (
        ({"remote_commit": "b" * 40}, "differs from expected", True),
        (
            {"origin_url": "git@github.com:Example/Unexpected.git"},
            "source identity mismatch",
            False,
        ),
    )
    for index, (configuration, expected_error, fetched) in enumerate(cases):
        case_root = tmp_path / str(index)
        environment, capture, deployment_root, _ = _deployment_environment(
            case_root, **configuration
        )
        result = subprocess.run(
            [*_deploy_base(deployment_root, expected_commit), "plan"],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert expected_error in result.stderr
        assert (capture / "fetch.bin").exists() is fetched
        assert not (capture / "plan-install.bin").exists()


def test_deployment_requires_explicit_safe_target_root_and_hashes(
    tmp_path: Path,
) -> None:
    environment, _, deployment_root, _ = _deployment_environment(tmp_path)
    expected_commit = "a" * 40
    cases = (
        (
            [
                "bash",
                str(DEPLOY_WRAPPER),
                "--root",
                str(deployment_root),
                "--expected-commit",
                expected_commit,
                "plan",
            ],
            "missing -H/--host",
        ),
        (
            [*_deploy_base(Path("relative"), expected_commit), "plan"],
            "absolute path",
        ),
        (
            [*_deploy_base(Path(f"{deployment_root}/../escape"), expected_commit), "plan"],
            "unsafe --root",
        ),
        (
            [*_deploy_base(deployment_root, "A" * 40), "plan"],
            "full lowercase Git SHA",
        ),
        (
            [*_deploy_base(deployment_root, expected_commit), "install"],
            "requires a full lowercase --confirm-plan-sha256",
        ),
        (
            [
                *_deploy_base(deployment_root, expected_commit),
                "--confirm-plan-sha256",
                "f" * 64,
                "plan",
            ],
            "plan does not accept",
        ),
    )
    for command, expected_error in cases:
        result = subprocess.run(
            command,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert expected_error in result.stderr
