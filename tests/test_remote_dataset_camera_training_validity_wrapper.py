from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "codex_remote/remote_dataset_camera_training_validity.sh"


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


def test_wrapper_is_fixed_hash_bound_and_requires_reader_confirmation(
    tmp_path: Path,
) -> None:
    env, capture = _env(tmp_path)
    source = tmp_path / "source"
    prepared = tmp_path / "prepared"
    source.mkdir()
    prepared.mkdir()
    base = [
        "bash",
        str(WRAPPER),
        "-H",
        "user@host",
        "--project",
        str(ROOT),
        "--dataset-root",
        str(source),
        "--prepared-view-root",
        str(prepared),
    ]

    result = subprocess.run(
        base,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert _args(capture) == [
        "-u",
        "-m",
        "utils.dataset_postprocess",
        "camera-video-integrity-mark-training-invalid",
        str(source),
        "--prepared-view-root",
        str(prepared),
    ]

    plan_hash = "a" * 64
    result = subprocess.run(
        [
            *base,
            "--apply",
            "--expected-plan-hash",
            plan_hash,
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "--confirm-no-active-readers" in result.stderr

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
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert _args(capture)[-3:] == [
        "--apply",
        "--expected-plan-hash",
        plan_hash,
    ]

    result = subprocess.run(
        [*base, "--", "--arbitrary"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "passthrough" in result.stderr
