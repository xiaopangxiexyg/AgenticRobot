from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import pytest_native_memory as native_memory


@pytest.mark.parametrize("platform,has_symbol", [("linux", True), ("linux", False), ("darwin", True)])
def test_native_trim_loader_keeps_platform_support_optional(monkeypatch, platform, has_symbol):
    calls = []
    function = SimpleNamespace()
    library = SimpleNamespace(malloc_trim=function) if has_symbol else SimpleNamespace()

    def load(handle):
        calls.append(handle)
        return library

    monkeypatch.setattr(native_memory.sys, "platform", platform)
    monkeypatch.setattr(native_memory.ctypes, "CDLL", load)

    result = native_memory._load_malloc_trim()

    if platform == "linux" and has_symbol:
        assert result is function
        assert function.argtypes == (ctypes.c_size_t,)
        assert function.restype is ctypes.c_int
    else:
        assert result is None
    assert calls == ([None] if platform == "linux" else [])


def test_native_trim_loader_preserves_unexpected_failures(monkeypatch):
    def unavailable(_handle):
        raise OSError("native allocator lookup failed")

    monkeypatch.setattr(native_memory.sys, "platform", "linux")
    monkeypatch.setattr(native_memory.ctypes, "CDLL", unavailable)

    with pytest.raises(OSError, match="native allocator lookup failed"):
        native_memory._load_malloc_trim()


def _run_probe(path: Path, *, enabled: bool) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).resolve().parents[1]
    command = ["bash", str(root / "scripts/run_pytest.sh"), "-q"]
    if enabled:
        command.extend(("-p", "scripts.pytest_native_memory"))
    command.append(str(path))
    environment = dict(os.environ)
    environment.pop("PYTEST_ADDOPTS", None)
    return subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True)


@pytest.mark.parametrize("enabled", [False, True])
def test_real_pytest_reclaims_only_after_module_teardown(tmp_path: Path, enabled: bool):
    probe = tmp_path / "probe"
    probe.mkdir()
    events = probe / "events"
    (probe / "conftest.py").write_text(
        "from pathlib import Path\n"
        "import pytest\n"
        "from scripts import pytest_native_memory as native_memory\n"
        f"EVENTS = Path({str(events)!r})\n"
        "def append(value):\n"
        "    with EVENTS.open('a') as stream:\n"
        "        stream.write(value + '\\n')\n"
        "def pytest_configure(config):\n"
        "    def trim(padding):\n"
        "        assert padding == 128 * 1024\n"
        "        append('trim')\n"
        "        return 1\n"
        "    native_memory._MALLOC_TRIM = trim\n"
        "@pytest.fixture(scope='module')\n"
        "def resource():\n"
        "    yield\n"
        "    append('fixture_closed')\n",
        encoding="utf-8",
    )
    (probe / "test_a.py").write_text(
        f"from pathlib import Path\nEVENTS = Path({str(events)!r})\n"
        "def test_first(resource):\n"
        "    assert not EVENTS.exists()\n"
        "def test_second(resource):\n"
        "    assert not EVENTS.exists()\n",
        encoding="utf-8",
    )
    expected = ["fixture_closed", "trim"] if enabled else ["fixture_closed"]
    (probe / "test_b.py").write_text(
        f"from pathlib import Path\nEVENTS = Path({str(events)!r})\n"
        "def test_next_module():\n"
        f"    assert EVENTS.read_text().splitlines() == {expected!r}\n",
        encoding="utf-8",
    )

    result = _run_probe(probe, enabled=enabled)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "3 passed" in result.stdout
    assert events.read_text().splitlines() == expected + (["trim"] if enabled else [])


def test_native_trim_failure_cannot_be_reported_as_success(tmp_path: Path):
    (tmp_path / "conftest.py").write_text(
        "from scripts import pytest_native_memory as native_memory\n"
        "def pytest_configure(config):\n"
        "    def trim(padding):\n"
        "        raise RuntimeError('native trim fault')\n"
        "    native_memory._MALLOC_TRIM = trim\n",
        encoding="utf-8",
    )
    (tmp_path / "test_probe.py").write_text("def test_passes():\n    pass\n", encoding="utf-8")

    result = _run_probe(tmp_path, enabled=True)

    assert result.returncode != 0
    assert "native trim fault" in result.stdout + result.stderr
