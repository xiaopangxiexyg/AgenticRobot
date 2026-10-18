import subprocess
from contextlib import contextmanager

import pytest

from scripts.probes import restore_d405_driver_binding as probe


@pytest.fixture
def usb(tmp_path, monkeypatch):
    root = tmp_path / "usb"
    device = root / "2-4.1"
    device.mkdir(parents=True)
    for key, value in {"serial": "123456789012", "idVendor": "8086", "idProduct": "0b5b", "busnum": "2", "devnum": "13"}.items():
        (device / key).write_text(value)
    for index in range(4):
        interface = root / f"2-4.1:1.{index}"
        interface.mkdir()
        (interface / "bInterfaceClass").write_text("0e")
        (interface / "bInterfaceSubClass").write_text("01" if index == 0 else "02")
    videos = tmp_path / "video"
    videos.mkdir()
    monkeypatch.setattr(probe, "USB_ROOT", root)
    monkeypatch.setattr(probe, "VIDEO_ROOT", videos)
    monkeypatch.setattr(probe.os, "geteuid", lambda: 0)
    monkeypatch.setattr(probe.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 1, "", ""))
    driver = tmp_path / "uvcvideo"
    driver.mkdir()
    def bound():
        for index in range(4):
            (root / f"2-4.1:1.{index}" / "driver").symlink_to(driver)
        node = root / "2-4.1:1.0/video4linux/video99"
        node.mkdir(parents=True)
        (videos / "video99").symlink_to(node)
    return device, bound


def test_bind_once_targets_only_verified_control_interface(usb, monkeypatch):
    _, bound = usb
    calls = []
    class Binding:
        @contextmanager
        def open(self, mode):
            assert mode == "w"
            yield self
            bound()
        def write(self, interface):
            calls.append(interface)
    monkeypatch.setattr(probe, "BIND_FILE", Binding())
    result = probe.restore("2-4.1", "123456789012")
    assert calls == ["2-4.1:1.0"]
    assert result["status"] == "restored"
    assert result["after"]["video_nodes"] == ["/dev/video99"]


@pytest.mark.parametrize("reason", ["wrong_identity", "occupied", "unknown_owner", "not_root", "unknown_stderr"])
def test_preflight_failure_never_binds(usb, monkeypatch, reason):
    device, _ = usb
    class Forbidden:
        def open(self, mode):
            pytest.fail("Binding must not happen after a failed preflight")
    monkeypatch.setattr(probe, "BIND_FILE", Forbidden())
    if reason == "wrong_identity":
        (device / "serial").write_text("other")
    elif reason == "not_root":
        monkeypatch.setattr(probe.os, "geteuid", lambda: 1000)
    else:
        monkeypatch.setattr(probe.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 0 if reason == "occupied" else 1 if reason == "unknown_stderr" else 2, "123" if reason == "occupied" else "", "inspection_failed" if reason == "unknown_stderr" else ""))
    with pytest.raises((RuntimeError, PermissionError)):
        probe.restore("2-4.1", "123456789012")


def test_already_bound_is_not_rebound(usb):
    _, bound = usb
    bound()
    assert probe.restore("2-4.1", "123456789012")["status"] == "already_bound"
