from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "codex_remote/remote_dataset_camera_semantics.sh"


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _fake_remote_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "python-args.bin"
    remote_home = tmp_path / "remote-home"
    remote_home.mkdir()
    _write_executable(
        fake_bin / "ssh",
        """#!/usr/bin/env bash
set -Eeuo pipefail
remote_command="${!#}"
export HOME="$FAKE_REMOTE_HOME"
exec bash -c "$remote_command"
""",
    )
    _write_executable(
        fake_bin / "python",
        """#!/usr/bin/env bash
set -Eeuo pipefail
printf '%s\\0' "$@" > "$FAKE_PYTHON_CAPTURE"
""",
    )
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "FAKE_PYTHON_CAPTURE": str(capture),
            "FAKE_REMOTE_HOME": str(remote_home),
        }
    )
    return env, capture


def _base_command(tmp_path: Path) -> tuple[list[str], Path, Path, Path]:
    source = tmp_path / "2026-07-23_dmtac_rm65_dataset"
    source.mkdir()
    output = tmp_path / "camera_semantic_prepared_views" / source.name
    registry = tmp_path / "camera-role-registry.json"
    registry.write_text("{}\n", encoding="utf-8")
    command = [
        "bash",
        str(WRAPPER),
        "-H",
        "user@192.168.70.116",
        "--project",
        str(REPO_ROOT),
        "--source-root",
        str(source),
        "--output-root",
        str(output),
        "--registry",
        str(registry),
        "--timeout",
        "10",
    ]
    return command, source, output, registry


def _captured_python_args(path: Path) -> list[str]:
    return [
        item.decode("utf-8")
        for item in path.read_bytes().split(b"\0")
        if item
    ]


def test_wrapper_dry_run_is_fixed_to_camera_semantic_prepared_view(
    tmp_path: Path,
) -> None:
    env, capture = _fake_remote_environment(tmp_path)
    command, source, output, registry = _base_command(tmp_path)
    result = subprocess.run(
        [*command, "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "safety=L3-dataset-non-motion" in result.stdout
    assert _captured_python_args(capture) == [
        "-u",
        "-m",
        "utils.dataset_postprocess",
        "camera-semantics-prepare-view",
        str(source),
        "--output-root",
        str(output),
        "--registry",
        str(registry),
        "--dry-run",
    ]


def test_wrapper_apply_requires_and_forwards_exact_plan_hash(tmp_path: Path) -> None:
    env, capture = _fake_remote_environment(tmp_path)
    command, source, output, registry = _base_command(tmp_path)
    plan_hash = "a" * 64
    result = subprocess.run(
        [*command, "--apply", "--expected-plan-hash", plan_hash],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert _captured_python_args(capture) == [
        "-u",
        "-m",
        "utils.dataset_postprocess",
        "camera-semantics-prepare-view",
        str(source),
        "--output-root",
        str(output),
        "--registry",
        str(registry),
        "--expected-plan-hash",
        plan_hash,
        "--apply",
    ]


def test_wrapper_fails_closed_on_ambiguous_or_extensible_arguments(
    tmp_path: Path,
) -> None:
    env, _ = _fake_remote_environment(tmp_path)
    command, _, _, _ = _base_command(tmp_path)
    cases = (
        ([*command, "--apply"], "requires a lowercase 64-hex"),
        (
            [*command, "--dry-run", "--expected-plan-hash", "a" * 64],
            "only valid with --apply",
        ),
        (
            [
                *command,
                "--dry-run",
                "--apply",
                "--expected-plan-hash",
                "a" * 64,
            ],
            "choose exactly one",
        ),
        ([*command, "--dry-run", "--", "--arbitrary"], "passthrough"),
        ([*command, "--confirm-motion"], "unknown argument"),
    )
    for argv, expected in cases:
        result = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode != 0
        assert expected in result.stderr


def test_wrapper_refuses_existing_apply_output_before_python(tmp_path: Path) -> None:
    env, capture = _fake_remote_environment(tmp_path)
    command, _, output, _ = _base_command(tmp_path)
    output.mkdir(parents=True)
    result = subprocess.run(
        [*command, "--apply", "--expected-plan-hash", "b" * 64],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode != 0
    assert "Refusing existing output path" in result.stderr
    assert not capture.exists()


def test_wrapper_resolves_named_env_from_standard_remote_prefix(
    tmp_path: Path,
) -> None:
    env, capture = _fake_remote_environment(tmp_path)
    command, source, output, registry = _base_command(tmp_path)
    activate = (
        Path(env["FAKE_REMOTE_HOME"])
        / "miniconda3/envs/dexteleop310/bin/activate"
    )
    activate.parent.mkdir(parents=True)
    activate.write_text(
        "export DEXTELEOP_TEST_NAMED_ENV_ACTIVATED=1\n",
        encoding="utf-8",
    )
    command[command.index("--project"):command.index("--project")] = [
        "--env",
        "dexteleop310",
    ]

    result = subprocess.run(
        [*command, "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert _captured_python_args(capture) == [
        "-u",
        "-m",
        "utils.dataset_postprocess",
        "camera-semantics-prepare-view",
        str(source),
        "--output-root",
        str(output),
        "--registry",
        str(registry),
        "--dry-run",
    ]
