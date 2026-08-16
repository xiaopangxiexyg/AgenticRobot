from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "codex_remote/remote_dataset_tactile_storage.sh"


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


def _base_command(tmp_path: Path) -> tuple[list[str], Path, Path]:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    cold = tmp_path / "cold"
    cold.mkdir()
    command = [
        "bash",
        str(WRAPPER),
        "-H",
        "user@192.168.70.116",
        "--project",
        str(REPO_ROOT),
        "--dataset-root",
        str(dataset),
        "--timeout",
        "10",
    ]
    return command, dataset, cold


def _captured_python_args(path: Path) -> list[str]:
    return [
        item.decode("utf-8")
        for item in path.read_bytes().split(b"\0")
        if item
    ]


def test_status_is_fixed_read_only_operation(tmp_path: Path) -> None:
    env, capture = _fake_remote_environment(tmp_path)
    command, dataset, _ = _base_command(tmp_path)
    result = subprocess.run(
        [*command, "--status"],
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
        "modality-status",
        str(dataset),
        "--modality",
        "tactile",
    ]


def test_detach_dry_run_is_fixed_to_whole_prefix(tmp_path: Path) -> None:
    env, capture = _fake_remote_environment(tmp_path)
    command, dataset, cold = _base_command(tmp_path)
    result = subprocess.run(
        [
            *command,
            "--detach-whole-prefix",
            "--cold-root",
            str(cold),
            "--min-free-gb",
            "0.5",
            "--dry-run",
        ],
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
        "modality-detach",
        str(dataset),
        "--cold-root",
        str(cold),
        "--modality",
        "tactile",
        "--detach-scope",
        "whole-prefix",
        "--min-free-gb",
        "0.5",
    ]


def test_detach_apply_requires_and_forwards_frozen_confirmations(
    tmp_path: Path,
) -> None:
    env, capture = _fake_remote_environment(tmp_path)
    command, dataset, cold = _base_command(tmp_path)
    plan_hash = "a" * 64
    fingerprint = "b" * 64
    result = subprocess.run(
        [
            *command,
            "--detach-whole-prefix",
            "--cold-root",
            str(cold),
            "--apply",
            "--plan-hash",
            plan_hash,
            "--dataset-fingerprint",
            fingerprint,
            "--confirm-no-active-readers",
        ],
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
        "modality-detach",
        str(dataset),
        "--cold-root",
        str(cold),
        "--modality",
        "tactile",
        "--detach-scope",
        "whole-prefix",
        "--min-free-gb",
        "16",
        "--apply",
        "--plan-hash",
        plan_hash,
        "--confirm-dataset-fingerprint",
        fingerprint,
        "--confirm-no-active-readers",
    ]


def test_rehydrate_dry_run_binds_inventory(tmp_path: Path) -> None:
    env, capture = _fake_remote_environment(tmp_path)
    command, dataset, cold = _base_command(tmp_path)
    inventory_hash = "c" * 64
    result = subprocess.run(
        [
            *command,
            "--rehydrate",
            "--cold-root",
            str(cold),
            "--inventory-hash",
            inventory_hash,
            "--dry-run",
        ],
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
        "modality-rehydrate",
        str(dataset),
        "--cold-root",
        str(cold),
        "--modality",
        "tactile",
        "--inventory-hash",
        inventory_hash,
        "--min-free-gb",
        "16",
    ]


def test_rehydrate_apply_forwards_all_frozen_confirmations(tmp_path: Path) -> None:
    env, capture = _fake_remote_environment(tmp_path)
    command, dataset, cold = _base_command(tmp_path)
    inventory_hash = "c" * 64
    plan_hash = "d" * 64
    fingerprint = "e" * 64
    result = subprocess.run(
        [
            *command,
            "--rehydrate",
            "--cold-root",
            str(cold),
            "--inventory-hash",
            inventory_hash,
            "--apply",
            "--plan-hash",
            plan_hash,
            "--dataset-fingerprint",
            fingerprint,
            "--confirm-no-active-readers",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert _captured_python_args(capture)[-6:] == [
        "--apply",
        "--plan-hash",
        plan_hash,
        "--confirm-dataset-fingerprint",
        fingerprint,
        "--confirm-no-active-readers",
    ]


def test_wrapper_fails_closed_on_ambiguous_or_extensible_arguments(
    tmp_path: Path,
) -> None:
    env, capture = _fake_remote_environment(tmp_path)
    command, _, cold = _base_command(tmp_path)
    cases = (
        ([*command], "choose one of"),
        ([*command, "--status", "--detach-whole-prefix"], "choose exactly one"),
        ([*command, "--status", "--cold-root", str(cold)], "not valid with --status"),
        ([*command, "--status", "--apply"], "read-only"),
        (
            [*command, "--detach-whole-prefix", "--cold-root", str(cold), "--apply"],
            "requires a lowercase 64-hex --plan-hash",
        ),
        (
            [
                *command,
                "--rehydrate",
                "--cold-root",
                str(cold),
                "--inventory-hash",
                "bad",
            ],
            "requires a lowercase 64-hex --inventory-hash",
        ),
        ([*command, "--status", "--", "--arbitrary"], "passthrough"),
        ([*command, "--status", "--allow-legacy-hardlinked-source"], "unknown argument"),
    )
    for argv, expected in cases:
        if capture.exists():
            capture.unlink()
        result = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode != 0
        assert expected in result.stderr
        assert not capture.exists()
