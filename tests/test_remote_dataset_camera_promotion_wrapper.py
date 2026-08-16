from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "codex_remote/remote_dataset_camera_promotion.sh"


def _executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _env(tmp_path: Path) -> tuple[dict[str, str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "args.bin"
    home = tmp_path / "home"
    home.mkdir()
    _executable(
        bin_dir / "ssh",
        """#!/usr/bin/env bash
set -Eeuo pipefail
export HOME="$FAKE_HOME"
exec bash -c "${!#}"
""",
    )
    _executable(
        bin_dir / "python",
        """#!/usr/bin/env bash
set -Eeuo pipefail
printf '%s\\0' "$@" > "$CAPTURE"
""",
    )
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "FAKE_HOME": str(home),
            "CAPTURE": str(capture),
        }
    )
    return env, capture


def _args(path: Path) -> list[str]:
    return [item.decode() for item in path.read_bytes().split(b"\0") if item]


def test_promote_plan_and_apply_are_fixed_and_hash_bound(tmp_path: Path) -> None:
    env, capture = _env(tmp_path)
    source = tmp_path / "source"
    corrected = tmp_path / "corrected"
    retirement = tmp_path / "retirement"
    registry = tmp_path / "registry.json"
    source.mkdir()
    corrected.mkdir()
    registry.write_text("{}", encoding="utf-8")
    base = [
        "bash",
        str(WRAPPER),
        "-H",
        "user@host",
        "--project",
        str(ROOT),
        "--promote",
        "--dataset-root",
        str(source),
        "--corrected-root",
        str(corrected),
        "--retirement-root",
        str(retirement),
        "--registry",
        str(registry),
    ]
    result = subprocess.run(base, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert _args(capture)[3:5] == ["camera-semantics-promote", str(source)]
    plan_hash = "a" * 64
    result = subprocess.run(
        [
            *base,
            "--apply",
            "--expected-plan-hash",
            plan_hash,
            "--confirm-no-active-readers",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert _args(capture)[-4:] == [
        "--apply",
        "--expected-plan-hash",
        plan_hash,
        "--confirm-no-active-readers",
    ]


def test_rollback_is_fixed_to_transaction(tmp_path: Path) -> None:
    env, capture = _env(tmp_path)
    transaction = tmp_path / "transaction.json"
    transaction.write_text("{}", encoding="utf-8")
    result = subprocess.run(
        [
            "bash",
            str(WRAPPER),
            "-H",
            "user@host",
            "--project",
            str(ROOT),
            "--rollback",
            "--transaction",
            str(transaction),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert _args(capture) == [
        "-u",
        "-m",
        "utils.dataset_postprocess",
        "camera-semantics-rollback-promotion",
        str(transaction),
    ]


def test_wrapper_rejects_passthrough_and_unconfirmed_apply(tmp_path: Path) -> None:
    env, capture = _env(tmp_path)
    transaction = tmp_path / "transaction.json"
    transaction.write_text("{}", encoding="utf-8")
    base = [
        "bash",
        str(WRAPPER),
        "-H",
        "user@host",
        "--project",
        str(ROOT),
        "--rollback",
        "--transaction",
        str(transaction),
    ]
    cases = (
        ([*base, "--apply"], "requires a lowercase 64-hex"),
        ([*base, "--", "--arbitrary"], "passthrough"),
        ([*base, "--allow-overwrite"], "unknown argument"),
    )
    for command, expected in cases:
        if capture.exists():
            capture.unlink()
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        assert result.returncode != 0
        assert expected in result.stderr
        assert not capture.exists()
