from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "codex_remote/remote_env.sh"


def _executable(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -Eeuo pipefail\n" + body)
    path.chmod(0o755)


@pytest.fixture
def remote_env(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls"
    ssh_args = tmp_path / "ssh-args"
    _executable(
        bin_dir / "ssh",
        'printf "%s\\0" "$@" > "$SSH_CAPTURE"\nexec bash -c "${!#}"\n',
    )
    _executable(
        bin_dir / "python3",
        'echo python >> "$CALL_CAPTURE"\n'
        '[[ "$1" == --version ]]\n'
        "if [[ ${FAIL_PYTHON:-0} == 1 ]]; then echo broken-python >&2; exit 17; fi\n"
        'echo "Python 3.10.99"\n',
    )
    for name in ("conda", "nvidia-smi", "nvcc"):
        _executable(
            bin_dir / name,
            f'echo {name} >> "$CALL_CAPTURE"\nprintf "{name} inventory\\n"\n',
        )
    environment = os.environ.copy()
    for name in ("REMOTE_HOST", "REMOTE_USER", "REMOTE_PORT", "KEY_FILE"):
        environment.pop(name, None)
    environment.update(
        PATH=f"{bin_dir}:{environment['PATH']}",
        CALL_CAPTURE=str(calls),
        SSH_CAPTURE=str(ssh_args),
        KEY_FILE=str(tmp_path / "absent-key"),
    )
    return environment, calls, ssh_args


def _run(environment: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(WRAPPER), *args],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )


@pytest.mark.parametrize("summary", [False, True])
def test_environment_scope_and_target(
    remote_env, tmp_path: Path, summary: bool
) -> None:
    environment, calls, ssh_args = remote_env
    project = tmp_path / "project with 'quotes'"
    project.mkdir()
    # This activation script changes only the child shell, like a venv activation.
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/activate").write_text('export SHELL="summary-test-shell"\n')
    result = _run(
        environment,
        "-H",
        "user@192.0.2.10:2222",
        "--project",
        str(project),
        "--env",
        str(venv),
        "--python",
        "python3",
        *(["--summary"] if summary else []),
    )
    assert result.returncode == 0, result.stderr
    assert "[target] host=user@192.0.2.10 port=2222" in result.stdout
    assert f"project={project} env={venv}" in result.stdout
    assert f"[remote] pwd={project}" in result.stdout
    assert "[remote] hostname=" in result.stdout
    assert "[python] executable=" in result.stdout
    assert "[python] version=Python 3.10.99" in result.stdout
    transport = ssh_args.read_bytes().split(b"\0")
    assert b"user@192.0.2.10" in transport
    assert transport[transport.index(b"-p") + 1] == b"2222"
    if summary:
        assert calls.read_text().splitlines() == ["python"]
        assert "GPU and CUDA not checked" in result.stdout
        assert "[remote] path=" not in result.stdout
        assert "[remote] shell=" not in result.stdout
    else:
        assert calls.read_text().splitlines() == [
            "python",
            "conda",
            "nvidia-smi",
            "nvcc",
        ]
        assert "[remote] path=" in result.stdout
        assert "[remote] shell=summary-test-shell" in result.stdout
        assert "[scope]" not in result.stdout


@pytest.mark.parametrize("summary", [False, True])
@pytest.mark.parametrize("failure", ["project", "activation", "python", "ssh"])
def test_failures_remain_failures(
    remote_env, tmp_path: Path, summary: bool, failure: str
) -> None:
    environment, calls, _ssh_args = remote_env
    args = ["-H", "second-target", "--python", "python3"]
    if failure == "project":
        args += ["--project", str(tmp_path / "missing")]
        expected, message = 3, "Project directory does not exist"
    elif failure == "activation":
        venv = tmp_path / "broken-venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin/activate").write_text("echo broken-activation >&2\nreturn 19\n")
        args += ["--env", str(venv)]
        expected, message = 19, "broken-activation"
    elif failure == "python":
        environment["FAIL_PYTHON"] = "1"
        expected, message = 17, "broken-python"
    else:
        _executable(tmp_path / "bin/ssh", "echo connection-failed >&2\nexit 255\n")
        expected, message = 255, "connection-failed"
    if summary:
        args += ["--summary"]
    result = _run(environment, *args)
    assert result.returncode == expected
    assert message in result.stdout + result.stderr
    assert "[target] host=second-target" in result.stdout
    assert "[scope]" not in result.stdout
    if calls.exists():
        assert calls.read_text().splitlines() == ["python"]


def test_summary_does_not_supply_missing_target(remote_env) -> None:
    environment, calls, ssh_args = remote_env
    result = _run(environment, "--summary")
    assert result.returncode != 0
    assert "missing -H/--host" in result.stderr
    assert not ssh_args.exists()
    assert not calls.exists()
