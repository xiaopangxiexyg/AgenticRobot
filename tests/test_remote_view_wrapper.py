from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "codex_remote/remote_view.sh"


def _executable(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -Eeuo pipefail\n" + body)
    path.chmod(0o755)


@pytest.fixture
def remote_view(tmp_path: Path) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _executable(bin_dir / "ssh", 'exec bash -c "${!#}"\n')
    environment = os.environ.copy()
    for name in ("REMOTE_HOST", "REMOTE_USER", "REMOTE_PORT", "KEY_FILE"):
        environment.pop(name, None)
    environment.update(
        PATH=f"{bin_dir}:{environment['PATH']}",
        KEY_FILE=str(tmp_path / "absent-key"),
    )
    return environment


def _run(environment, path: Path, compact: bool, *args: str):
    return subprocess.run(
        [
            "bash",
            str(WRAPPER),
            "-H",
            "user@192.0.2.10:2222",
            "--path",
            str(path),
            *(["--compact"] if compact else []),
            *args,
        ],
        env=environment,
        check=False,
        capture_output=True,
        timeout=10,
    )


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("selection", ["head", "tail"])
@pytest.mark.parametrize("limit,cap", [(120, 128), (2, 20000), (0, 128), (120, 0)])
def test_bounded_content_and_identity(
    remote_view, tmp_path: Path, compact: bool, selection: str, limit: int, cap: int
) -> None:
    path = tmp_path / "log with 'quotes'.txt"
    lines = [f"{i:03d} ".encode() + b"x" * 4096 + b"\n" for i in range(150)]
    path.write_bytes(b"".join(lines))
    selected = b"".join(
        lines[:limit] if selection == "head" else lines[-limit:] if limit else []
    )
    result = _run(
        remote_view,
        path,
        compact,
        f"--{selection}",
        str(limit),
        "--max-bytes",
        str(cap),
    )
    assert result.returncode == 0, result.stderr
    assert not result.stderr
    header, content = result.stdout.split(b"\n[content]\n", 1)
    assert b"[target] host=user@192.0.2.10 port=2222" in header
    assert f"[remote] resolved={path}".encode() in header
    assert b"[remote] type=file" in header
    assert (b"[view] path=" not in header) == compact
    assert (f"[file] bytes={path.stat().st_size} mode=".encode() in header) == compact
    footer = (
        f"\n[view] selection={selection} lines={limit} max_bytes={cap} "
        f"selected_bytes_omitted={max(len(selected) - cap, 0)}\n"
    ).encode()
    assert content == selected[:cap] + footer


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("payload", [b"", b"no final newline", "中文末行".encode()])
def test_small_content_byte_limit(
    remote_view, tmp_path: Path, compact: bool, payload: bytes
) -> None:
    path = tmp_path / "small.txt"
    path.write_bytes(payload)
    result = _run(remote_view, path, compact, "--max-bytes", "4")
    assert result.returncode == 0, result.stderr
    content = result.stdout.split(b"\n[content]\n", 1)[1]
    assert (
        content
        == payload[:4]
        + (
            f"\n[view] selection=head lines=120 max_bytes=4 selected_bytes_omitted={max(len(payload) - 4, 0)}\n"
        ).encode()
    )


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("failure", ["missing", "sensitive", "binary", "ssh"])
def test_refusals_and_transport_errors(
    remote_view, tmp_path: Path, compact: bool, failure: str
) -> None:
    path = tmp_path / "input.txt"
    expected = 4
    if failure == "sensitive":
        path = tmp_path / ".env"
        path.write_text("must-not-display\n")
        expected, message = 5, b"Refusing to view common secret/key path"
    elif failure == "binary":
        path.write_bytes(b"\0\xff" * 100)
        message = b"refusing to display non-text file"
    elif failure == "ssh":
        _executable(tmp_path / "bin/ssh", "echo transport-failed >&2\nexit 255\n")
        expected, message = 255, b"transport-failed"
    else:
        message = b"Remote path does not exist"
    result = _run(remote_view, path, compact)
    assert result.returncode == expected
    assert message in result.stderr
    assert b"must-not-display" not in result.stdout
    assert b"[content]" not in result.stdout
    assert b"[target] host=user@192.0.2.10 port=2222" in result.stdout


@pytest.mark.parametrize("compact", [False, True])
def test_large_directory_limit(remote_view, tmp_path: Path, compact: bool) -> None:
    directory = tmp_path / "listing"
    directory.mkdir()
    # Produce more than a pipe buffer without creating thousands of files.
    _executable(
        tmp_path / "bin/ls",
        "for ((i=0; i<5000; i++)); do printf 'entry %08d padding padding padding\\n' \"$i\"; done\n",
    )
    result = _run(remote_view, directory, compact, "--head", "2")
    assert result.returncode == 0, result.stderr
    assert not result.stderr
    assert b"entry 00000000" in result.stdout
    assert b"entry 00000001" in result.stdout
    assert b"entry 00000002" not in result.stdout
    assert result.stdout.endswith(b"[view] listing_line_limit=2\n")


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("producer", ["head", "tail", "ls"])
def test_upstream_read_error_survives_limit(
    remote_view, tmp_path: Path, compact: bool, producer: str
) -> None:
    path = tmp_path / "input.txt"
    if producer == "ls":
        path.mkdir()
    else:
        path.write_text("valid text\n")
    actual = shutil.which(producer)
    assert actual is not None
    _executable(
        tmp_path / "bin" / producer,
        # Leave the consumer's head -c and directory head -n intact.
        (
            'if [[ "$1" == -c ]]; then exec ' + shlex.quote(actual) + ' "$@"; fi\n'
            if producer == "head"
            else ""
        )
        + "printf 'partial output\\n'\necho read-failed >&2\nexit 37\n",
    )
    result = _run(
        remote_view,
        path,
        compact,
        "--tail" if producer == "tail" else "--head",
        "2",
        "--max-bytes",
        "1",
    )
    assert result.returncode == 37
    assert b"read-failed" in result.stderr


@pytest.mark.parametrize("compact", [False, True])
def test_consumer_error_is_not_success(
    remote_view, tmp_path: Path, compact: bool
) -> None:
    path = tmp_path / "input.txt"
    path.write_text("valid text\n")
    actual = shutil.which("head")
    assert actual is not None
    _executable(
        tmp_path / "bin/head",
        'if [[ "$1" == -c ]]; then echo display-failed >&2; exit 39; fi\n'
        + f'exec {shlex.quote(actual)} "$@"\n',
    )
    result = _run(remote_view, path, compact)
    assert result.returncode != 0
    assert b"display-failed" in result.stderr
    assert b"selected_bytes_omitted=" not in result.stdout
