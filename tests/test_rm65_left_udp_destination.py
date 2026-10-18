from __future__ import annotations

import copy
import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hardware.rm65.utils.scripts import rm65_udp_actuator_trace as trace
from scripts.probes.set_rm65_left_udp_destination import update_destination


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "occupied",
        "old_ip",
        "port",
        "disabled",
        "setter",
        "readback",
        "no_packets",
        "interrupted",
        "writer",
        "disk_full",
        "disk_full_after_commit",
    ],
)
def test_ip_only_update_commits_or_restores_without_touching_right(
    monkeypatch, tmp_path, failure
):
    ips = {"left": "192.0.2.21", "right": "198.51.100.20"}
    original = {
        "cycle": 1,
        "enable": failure != "disabled",
        "port": 8089 if failure == "port" else 8090,
        "force_coordinate": 0,
        "ip": "203.0.113.10" if failure == "old_ip" else "198.51.100.10",
        "custom_config": dict.fromkeys(trace.CUSTOM_CONFIG_FIELDS, 0),
    }
    current = copy.deepcopy(original)
    writes = []
    clients = []
    output = tmp_path / "new-output"
    tick = [0.0]
    receiver = None

    class Client:
        def __init__(self, ip):
            assert ip == ips["left"]
            self.closed = False
            clients.append(self)

        def get_config(self):
            value = copy.deepcopy(current)
            if failure == "readback" and value["ip"] == "192.0.2.10":
                value["force_coordinate"] = 1
            return value

        def set_config(self, config):
            assert (output / "original_left.json").exists()
            assert (output / "rollback.json").exists()
            writes.append(copy.deepcopy(config))
            current.clear()
            current.update(copy.deepcopy(config))
            if failure in {"setter", "disk_full"} and config["ip"] == "192.0.2.10":
                raise RuntimeError("setter failed after applying the config")

        def close(self):
            self.closed = True

    class Writer:
        error = None
        dropped = 0

        def __init__(self, path):
            self.path = path

        def start(self):
            pass

        def close(self):
            pass

    class Receiver:
        error = None
        unexpected_source_count = 0
        sequence = 0
        json_payload_count = 0
        joint_position_count = 0

        def __init__(self, spec, writer, *, trace_format):
            nonlocal receiver
            assert spec.side == "left" and spec.arm_ip == ips["left"]
            assert spec.local_ip == "192.0.2.10" and spec.push_port == 8090
            assert trace_format == "full"
            if failure == "occupied":
                raise OSError("occupied")
            self.socket = SimpleNamespace(close=lambda: None)
            receiver = self

        def start(self):
            pass

        def close(self):
            pass

    def start_guardian(path):
        journal = json.loads(path.read_text())
        assert [arm["side"] for arm in journal["arms"]] == ["left"]
        assert journal["arms"][0]["original_config"] == original
        return SimpleNamespace(wait=lambda timeout: guardian_waits.append(timeout) or 1), io.StringIO()

    def acknowledge(guardian, pipe):
        assert guardian is not None
        acknowledgements.append(copy.deepcopy(current))
        pipe.close()

    def wait(duration):
        assert duration <= 0.05
        if failure == "interrupted":
            raise KeyboardInterrupt()
        if failure == "writer":
            receiver.error = RuntimeError("recording failed")
        tick[0] += duration
        if failure != "no_packets":
            receiver.sequence += 10
            receiver.json_payload_count += 10
            receiver.joint_position_count += 10

    monkeypatch.setattr(trace, "Api2PushConfigClient", Client)
    monkeypatch.setattr(trace, "UdpTraceWriter", Writer)
    monkeypatch.setattr(trace, "UdpReceiver", Receiver)
    monkeypatch.setattr(trace, "_start_guardian", start_guardian)
    monkeypatch.setattr(trace, "_notify_guardian_restored", acknowledge)
    guardian_waits = []
    acknowledgements = []
    original_write = trace._atomic_write_json

    def write(path, value):
        if (failure == "disk_full" and writes
                or failure == "disk_full_after_commit" and path.name == "result.json"):
            raise OSError(28, "disk_full")
        original_write(path, value)

    monkeypatch.setattr(trace, "_atomic_write_json", write)
    if failure:
        with pytest.raises((RuntimeError, OSError, KeyboardInterrupt)) as caught:
            update_destination(
                ips,
                "198.51.100.10",
                "192.0.2.10",
                output,
                clock=lambda: tick[0],
                wait=wait,
            )
        assert current == original
        if failure in {"disk_full", "disk_full_after_commit"}:
            assert writes[-1] == original
            assert isinstance(caught.value.__cause__, trace.RestoreError)
            assert "disk_full" in str(caught.value.__cause__)
            if failure == "disk_full":
                assert str(caught.value) == "setter failed after applying the config"
                assert guardian_waits == [10.0] and acknowledgements == []
                assert json.loads((output / "rollback.json").read_text())["status"] == "prepared"
            else:
                assert isinstance(caught.value, OSError) and caught.value.errno == 28
                assert guardian_waits == [] and acknowledgements == [original]
                assert json.loads((output / "rollback.json").read_text())["status"] == "restored"
        else:
            result = json.loads((output / "result.json").read_text())
            assert result["status"] == "failed"
            if failure in {"occupied", "old_ip", "port", "disabled"}:
                assert not writes
            else:
                assert writes[-1] == original
                assert result["original_configuration_restored"] is True
    else:
        result = update_destination(
            ips, "198.51.100.10", "192.0.2.10", output, clock=lambda: tick[0], wait=wait
        )
        expected = copy.deepcopy(original)
        expected["ip"] = "192.0.2.10"
        assert writes == [expected]
        assert current == expected
        assert json.loads((output / "original_left.json").read_text()) == original
        assert (
            result["status"] == "passed" and result["configuration_persisted"] is True
        )
        assert result["right_controller_connected"] is False
        assert result["packet_count"] >= 2000
        assert (
            json.loads((output / "rollback.json").read_text())["status"] == "committed"
        )
    assert all(client.closed for client in clients)


@pytest.mark.parametrize(
    "failure", [None, "route", "occupied", "controller", "dirty_source"]
)
def test_default_preflight_has_no_sdk_attachment_and_rejects_identity_or_occupancy(
    monkeypatch,
    tmp_path,
    failure,
):
    from scripts.probes import capture_rm65_grasp_snapshot as snapshot
    from scripts.probes import set_rm65_left_udp_destination as probe

    assignment = tmp_path / "ips.json"
    assignment.write_text(
        json.dumps(
            {
                "arm_ip_by_side": {
                    "left": "192.0.2.21",
                    "right": "198.51.100.20",
                }
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe",
            "--repo",
            str(Path(__file__).resolve().parents[1]),
            "--arm-ip-assignment",
            str(assignment),
            "--left-controller-ip",
            "192.0.2.99" if failure == "controller" else "192.0.2.21",
            "--from-ip",
            "198.51.100.10",
            "--to-ip",
            "192.0.2.10",
            "--output",
            str(tmp_path / "new-output"),
        ],
    )

    def check_output(argv, *, text):
        if argv[0] == "ip":
            return json.dumps(
                [{"prefsrc": "203.0.113.10" if failure == "route" else "192.0.2.10"}]
            )
        if argv[0] == "ss":
            return (
                "UNCONN 0 0 0.0.0.0:8090 0.0.0.0:*\n" if failure == "occupied" else ""
            )
        assert "rev-parse" in argv
        return "0" * 40 + "\n"

    def check_source(argv, *, check):
        assert "diff" in argv and check
        if failure == "dirty_source":
            raise subprocess.CalledProcessError(1, argv)

    def idle(ips):
        assert ips == {"left": "192.0.2.21"}

    def forbidden(*args, **kwargs):
        raise AssertionError("Default preflight must not execute the SDK updater")

    monkeypatch.setattr(probe.subprocess, "check_output", check_output)
    monkeypatch.setattr(probe.subprocess, "run", check_source)
    monkeypatch.setattr(snapshot, "assert_arm_connections_idle", idle)
    monkeypatch.setattr(probe, "update_destination", forbidden)
    if failure:
        with pytest.raises((RuntimeError, subprocess.CalledProcessError)):
            probe.main()
    else:
        assert probe.main() == 0
    assert not (tmp_path / "new-output").exists()
