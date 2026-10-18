from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("raw_pair", Path(__file__).resolve().parents[1] / "scripts/probes/capture_d405_v4l2_pair.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_capture_command_has_only_fixed_pair_parameters(tmp_path):
    command = probe.capture_command("/dev/video4", "color", tmp_path)
    assert command[:3] == ["/usr/bin/v4l2-ctl", "--device", "/dev/video4"]
    assert "--set-fmt-video=width=640,height=480,pixelformat=YUYV" in command
    assert "--set-parm=15" in command
    assert "--stream-count=30" in command
    assert "--stream-poll" in command
    assert not any("reset" in x or "ctrl" in x for x in command)


@pytest.mark.parametrize("size,rc,fmt,passed", [(30 * probe.FRAME_BYTES, 0, True, True),
    (0, 0, True, False), (30 * probe.FRAME_BYTES - 1, 0, True, False),
    (30 * probe.FRAME_BYTES, 1, True, False), (30 * probe.FRAME_BYTES, 0, False, False)])
def test_result_requires_exact_frames_format_and_exit(tmp_path, size, rc, fmt, passed):
    with (tmp_path / "depth.raw").open("wb") as stream:
        stream.truncate(size)
    (tmp_path / "depth.log").write_text("Width/Height : 640/480\nPixel Format : 'Z16 '\nFrames per second: " + ("15.000" if fmt else "30.000"))
    result = probe.stream_result("depth", tmp_path, rc)
    assert result["passed"] is passed
    assert result["complete_frames"] == size // probe.FRAME_BYTES


def test_resolution_rejects_wrong_product_and_missing_pair(tmp_path, monkeypatch):
    sysfs = tmp_path / "sysfs"
    usb = tmp_path / "usb"
    usb.mkdir()
    (usb / "serial").write_text("123456789012")
    (usb / "idVendor").write_text("8086")
    (usb / "idProduct").write_text("ffff")
    entry = sysfs / "video7"
    entry.mkdir(parents=True)
    (entry / "device").symlink_to(usb, target_is_directory=True)
    (entry / "index").write_text("0")
    with pytest.raises(RuntimeError, match="not an Intel D405"):
        probe.resolve_nodes("123456789012", sysfs, tmp_path)
    (usb / "idProduct").write_text("0b5b")
    monkeypatch.setattr(Path, "is_char_device", lambda self: True)
    with pytest.raises(RuntimeError, match="unique D405"):
        probe.resolve_nodes("123456789012", sysfs, tmp_path)
    color = sysfs / "video9"
    color.mkdir()
    (color / "device").symlink_to(usb, target_is_directory=True)
    (color / "index").write_text("4")
    result = probe.resolve_nodes("123456789012", sysfs, tmp_path)
    assert result["depth"]["device"].endswith("video7")
    assert result["color"]["device"].endswith("video9")


@pytest.mark.parametrize("returncode,stderr", [(2, "failure"), (1, "inspection_failed")])
def test_idle_rejects_unknown_fuser_result(monkeypatch, returncode, stderr):
    monkeypatch.setattr(Path, "glob", lambda self, pattern: [Path("/dev/video0")])
    monkeypatch.setattr(probe.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, returncode, "", stderr))
    with pytest.raises(RuntimeError, match="not idle"):
        probe.assert_idle()


def test_timeout_stops_owned_children_and_preserves_both_logs(tmp_path, monkeypatch):
    processes = []

    class Process:
        returncode = None
        pid = 123

        def __init__(self, command, **kwargs):
            processes.append(self)
            kwargs["stdout"].write(b"started\n")

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def wait(self, **kwargs):
            return self.returncode

    monkeypatch.setattr(probe.subprocess, "Popen", Process)
    monkeypatch.setattr(probe, "CAPTURE_LIMIT_SEC", 0)
    report = {}
    with pytest.raises(TimeoutError):
        probe.capture_pair({"depth": {"device": "/dev/video0"}, "color": {"device": "/dev/video4"}}, tmp_path, report)
    assert len(processes) == 2
    assert report["cleanup_complete"] is True
    assert all(x.returncode == -15 for x in processes)
    assert set(report["streams"]) == {"color", "depth"}
    assert (tmp_path / "depth.log").read_text() == "started\n"


def test_existing_evidence_refused_before_device_access(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "assert_idle", lambda: pytest.fail("must not touch devices"))
    with pytest.raises(FileExistsError):
        probe.main(["--usb-serial", "123456789012", "--output", str(tmp_path)])


def test_uninterruptible_owned_process_cannot_claim_cleanup():
    class Process:
        terminated = False
        killed = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

        def wait(self, **kwargs):
            raise subprocess.TimeoutExpired("owned-v4l2", 1)

    process = Process()
    assert probe.stop_children({"depth": process}) is False
    assert process.terminated and process.killed


@pytest.mark.parametrize("cancel_before_cleanup", [False, True])
def test_repeated_cancellation_finishes_all_owned_children_and_closes_logs(monkeypatch, tmp_path, cancel_before_cleanup):
    processes = []
    cancelled = False
    previous = {sig: probe.signal.getsignal(sig) for sig in (probe.signal.SIGINT, probe.signal.SIGTERM)}

    class Process:
        def __init__(self, command, **kwargs):
            self.pid = 100 + len(processes)
            self.returncode = None
            self.waits = 0
            self.confirmed = self.terminated = self.killed = False
            self.log = kwargs["stdout"]
            self.log.write(b"started\n")
            processes.append(self)

        def poll(self):
            nonlocal cancelled
            if cancel_before_cleanup and len(processes) == 2 and not cancelled:
                cancelled = True
                probe.signal.getsignal(probe.signal.SIGTERM)(probe.signal.SIGTERM, None)
            return self.returncode

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True
            self.returncode = -9

        def wait(self, timeout):
            assert 0 <= timeout <= 1
            self.waits += 1
            if self.waits == 1:
                sig = probe.signal.SIGTERM if self.pid == 100 else probe.signal.SIGINT
                probe.signal.getsignal(sig)(sig, None)
                raise subprocess.TimeoutExpired("owned-fake-v4l2", timeout)
            assert self.returncode == -9
            self.confirmed = True
            return self.returncode

    monkeypatch.setattr(probe.subprocess, "Popen", Process)
    monkeypatch.setattr(probe, "assert_idle", lambda: None)
    monkeypatch.setattr(probe, "resolve_nodes", lambda serial: {role: {"device": "/mock/"+role} for role in ("color", "depth")})
    monkeypatch.setattr(probe, "CAPTURE_LIMIT_SEC", 0)
    output = tmp_path / "new-evidence"
    try:
        assert probe.main(["--usb-serial", "123456789012", "--output", str(output)]) == 1
        assert len(processes) == 2
        assert all(p.terminated and p.killed and p.confirmed and p.waits == 2 for p in processes)
        assert all(p.log.closed for p in processes)
        import json
        report = json.loads((output / "result.json").read_text())
        assert report["status"] == "failed" and report["cleanup_complete"] is True
        assert report["cleanup_stop_signals"] == [probe.signal.SIGTERM, probe.signal.SIGINT]
        assert all(s["returncode"] == -9 for s in report["streams"].values())
        assert all(probe.signal.getsignal(sig) == handler for sig, handler in previous.items())
    finally:
        for sig, handler in previous.items():
            probe.signal.signal(sig, handler)


def test_child_api_failure_does_not_skip_sibling_confirmation():
    class Process:
        def __init__(self, broken):
            self.broken = broken
            self.returncode = None
            self.waits = 0

        def poll(self):
            return self.returncode

        def terminate(self):
            if self.broken:
                raise OSError("TERM rejected")
            self.returncode = -15

        def kill(self):
            self.returncode = -9

        def wait(self, timeout):
            self.waits += 1
            if self.returncode is None:
                raise subprocess.TimeoutExpired("owned-fake", timeout)
            return self.returncode

    children = {"color": Process(True), "depth": Process(False)}
    errors = []
    assert probe.stop_children(children, errors=errors) is False
    assert children["color"].returncode == -9
    assert children["depth"].returncode == -15
    assert all(p.waits == 2 for p in children.values())
    assert len(errors) == 1 and "color TERM" in errors[0] and "TERM rejected" in errors[0]


def test_one_log_close_failure_keeps_other_log_and_child_cleanup(monkeypatch, tmp_path):
    logs = []
    original_open = Path.open

    class Log:
        def __init__(self, stream, role):
            self.stream, self.role = stream, role
            logs.append(self)

        def write(self, value):
            return self.stream.write(value)

        def close(self):
            self.stream.close()
            if self.role == "color.log":
                raise OSError("synthetic log close failure")

    def opened(path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        return Log(stream, path.name) if mode == "xb" else stream

    class Process:
        pid = 123
        returncode = None

        def __init__(self, command, **kwargs):
            kwargs["stdout"].write(b"started\n")

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def wait(self, timeout):
            return self.returncode

    monkeypatch.setattr(Path, "open", opened)
    monkeypatch.setattr(probe.subprocess, "Popen", Process)
    monkeypatch.setattr(probe, "CAPTURE_LIMIT_SEC", 0)
    report = {}
    with pytest.raises(TimeoutError):
        probe.capture_pair({role: {"device": "/mock/"+role} for role in ("color", "depth")}, tmp_path, report)
    assert len(logs) == 2 and all(log.stream.closed for log in logs)
    assert report["cleanup_complete"] is False
    assert len(report["cleanup_errors"]) == 1
    assert "color log close" in report["cleanup_errors"][0]
    assert set(report["streams"]) == {"color", "depth"}


def test_report_write_failure_still_restores_original_signal_handlers(monkeypatch, tmp_path):
    previous = {sig: probe.signal.getsignal(sig) for sig in (probe.signal.SIGINT, probe.signal.SIGTERM)}
    original_write = Path.write_text

    def write(path, *args, **kwargs):
        if path.name == "result.json":
            raise OSError("synthetic report write failure")
        return original_write(path, *args, **kwargs)

    def rejected():
        raise RuntimeError("synthetic preflight refusal")

    monkeypatch.setattr(Path, "write_text", write)
    monkeypatch.setattr(probe, "assert_idle", rejected)
    try:
        with pytest.raises(OSError, match="synthetic report write failure"):
            probe.main(["--usb-serial", "123456789012", "--output", str(tmp_path/"new")])
        assert all(probe.signal.getsignal(sig) == handler for sig, handler in previous.items())
    finally:
        for sig, handler in previous.items():
            probe.signal.signal(sig, handler)
