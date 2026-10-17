from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from agent_gateway.profiles.real import perception_cli as cli


def test_camera_sdk_logging_is_opt_in_and_does_not_open_devices(tmp_path, monkeypatch, capsys):
    calls = []
    sdk = SimpleNamespace(
        __file__="fixture-sdk.so", log_severity=SimpleNamespace(debug=7),
        log_to_file=lambda level, path: calls.append((level, path)),
    )
    monkeypatch.setitem(sys.modules, "pyrealsense2", sdk)
    monkeypatch.setattr(cli.importlib.metadata, "version", lambda _: "fixture-version")
    cli._enable_camera_sdk_debug_log(None)
    assert not calls and capsys.readouterr().out == ""
    path = tmp_path / "sdk.log"
    cli._enable_camera_sdk_debug_log(path)
    assert calls == [(7, str(path))]
    event = json.loads(capsys.readouterr().out)
    assert event["stage"] == "sdk_debug_enabled"
    assert event["sdk_version"] == "fixture-version"
    assert event["sdk_module"] == "fixture-sdk.so"
    assert event["thread_name"] == "MainThread"
    assert event["pid"] > 0 and event["thread_native_id"] > 0


@pytest.mark.parametrize("kind", ["relative", "missing_parent", "existing", "symlink"])
def test_camera_sdk_log_rejects_invalid_destination_before_loading_sdk(tmp_path, monkeypatch, kind):
    calls = []
    monkeypatch.setitem(sys.modules, "pyrealsense2", SimpleNamespace(
        log_to_file=lambda *_: calls.append(True),
    ))
    path = tmp_path / "sdk.log"
    if kind == "relative":
        path = cli.Path("relative.log")
    elif kind == "missing_parent":
        path = tmp_path / "absent" / "sdk.log"
    elif kind == "existing":
        path.write_text("retained evidence")
    else:
        path.symlink_to(tmp_path / "absent-target")
    with pytest.raises((ValueError, FileExistsError)):
        cli._enable_camera_sdk_debug_log(path)
    assert not calls


def test_camera_sdk_logging_failure_is_not_silenced(tmp_path, monkeypatch):
    def fail(*_):
        raise RuntimeError("native logging failure")
    monkeypatch.setitem(sys.modules, "pyrealsense2", SimpleNamespace(
        log_severity=SimpleNamespace(debug=7), log_to_file=fail,
    ))
    with pytest.raises(RuntimeError, match="native logging failure"):
        cli._enable_camera_sdk_debug_log(tmp_path / "sdk.log")
