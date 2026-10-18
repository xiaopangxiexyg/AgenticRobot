from __future__ import annotations

import copy
import io
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hardware.rm65 import realtime_feedback
from hardware.rm65.utils.scripts import rm65_udp_actuator_trace as trace
from scripts.probes.set_rm65_udp_force_coordinate import update_force_coordinates
from tests.test_rm65_udp_feedback import decode


@pytest.mark.parametrize("failure", [None, "already_tool", "occupied", "wrong_destination", "disabled",
                                     "setter", "readback", "no_packets", "stale", "frame", "interrupted", "disk_full", "disk_full_after_commit"])
def test_dual_coordinate_only_transaction_or_verified_rollback(monkeypatch, tmp_path, failure):
    ips = {"left": "192.0.2.21", "right": "198.51.100.20"}
    locals_ = {"left": "192.0.2.10", "right": "198.51.100.10"}
    originals = {side: {"cycle": 1, "enable": failure != "disabled", "port": 8090,
                       "ip": "203.0.113.10" if failure == "wrong_destination" else locals_[side],
                       "force_coordinate": 2 if failure == "already_tool" else 0,
                       "custom_config": dict.fromkeys(trace.CUSTOM_CONFIG_FIELDS, 0)} for side in ips}
    current = copy.deepcopy(originals)
    writes, clients = [], []
    output = tmp_path / "new"
    tick = [0.]

    class Client:
        def __init__(self, ip):
            self.side = next(side for side in ips if ips[side] == ip)
            self.closed = False
            clients.append(self)

        def get_config(self):
            value = copy.deepcopy(current[self.side])
            if failure == "readback" and value["force_coordinate"] == 2:
                value["cycle"] = 2
            return value

        def set_config(self, value):
            assert (output / "original_left.json").exists()
            assert (output / "original_right.json").exists()
            assert (output / "rollback.json").exists()
            writes.append((self.side, copy.deepcopy(value)))
            current[self.side] = copy.deepcopy(value)
            if failure in {"setter", "disk_full"} and self.side == "right" and value["force_coordinate"] == 2:
                raise RuntimeError("setter failed after applying")

        def close(self):
            self.closed = True

    class Receiver:
        closed = False

        def __init__(self, *, arm_ip_by_side, port):
            assert arm_ip_by_side == ips and port == 8090

        def start(self):
            if failure == "occupied":
                raise OSError("occupied")

        def wait_ready(self, **kwargs):
            if failure == "no_packets":
                raise RuntimeError("no packets")

        def snapshot(self, side):
            from dataclasses import replace
            return replace(decode(), side=side, controller_ip=ips[side],
                           force_coordinate=0 if failure == "frame" else 2,
                           sequence=1 + int(tick[0] * 200),
                           monotonic_time_sec=time.monotonic() - (1. if failure == "stale" else 0.))

        def diagnostics(self):
            return {"invalid_packet_count_by_side": dict.fromkeys(ips, 0),
                    "unknown_source_count": 0, "fault": None}

        def close(self):
            self.closed = True

    def wait(duration):
        if failure == "interrupted":
            raise KeyboardInterrupt()
        tick[0] += duration

    def start_guardian(path):
        journal = json.loads(path.read_text())
        assert len(journal["arms"]) == 2
        return SimpleNamespace(wait=lambda timeout: guardian_waits.append(timeout) or 1), io.StringIO()

    def acknowledge(_guardian, pipe):
        acknowledgements.append(copy.deepcopy(current))
        pipe.close()

    monkeypatch.setattr(trace, "Api2PushConfigClient", Client)
    monkeypatch.setattr(realtime_feedback, "RM65UDPFeedbackReceiver", Receiver)
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
    if failure not in (None, "already_tool"):
        with pytest.raises((RuntimeError, OSError, KeyboardInterrupt)) as caught:
            update_force_coordinates(ips, locals_, output, clock=lambda: tick[0], wait=wait)
        assert current == originals
        if failure in {"disk_full", "disk_full_after_commit"}:
            assert [value for _, value in writes[-2:]] == [originals["right"], originals["left"]]
            assert isinstance(caught.value.__cause__, trace.RestoreError)
            assert "disk_full" in str(caught.value.__cause__)
            if failure == "disk_full":
                assert str(caught.value) == "setter failed after applying"
                assert guardian_waits == [10.0] and acknowledgements == []
                assert json.loads((output / "rollback.json").read_text())["status"] == "prepared"
            else:
                assert isinstance(caught.value, OSError) and caught.value.errno == 28
                assert guardian_waits == [] and acknowledgements == [originals]
                assert json.loads((output / "rollback.json").read_text())["status"] == "restored"
        else:
            result = json.loads((output / "result.json").read_text())
            if failure in {"occupied", "wrong_destination", "disabled"}:
                assert writes == []
            else:
                assert result["original_configuration_restored"]
                assert [value for _, value in writes[-2:]] == [originals["right"], originals["left"]]
    else:
        result = update_force_coordinates(ips, locals_, output, clock=lambda: tick[0], wait=wait)
        assert result["status"] == "passed" and result["cleanup_complete"]
        expected = copy.deepcopy(originals)
        for side in ips:
            expected[side]["force_coordinate"] = 2
        assert current == expected
        assert writes == ([] if failure == "already_tool" else list(expected.items()))
        assert all(count >= 400 for count in result["sampled_packet_count_by_side"].values())
    assert all(client.closed for client in clients)


def test_wrapper_preview_is_scoped_and_does_not_ssh():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(["bash", str(root / "codex_remote/remote_rm65_udp_force_coordinate.sh"),
                             "-H", "user@192.0.2.1", "--repo", "/source", "--python", "/runtime/python",
                             "--arm-ip-assignment", "/ips.json", "--left-controller-ip", "192.0.2.21",
                             "--right-controller-ip", "198.51.100.20", "--output", "/output", "--dry-run"],
                            check=True, capture_output=True, text=True)
    assert "no movement/HOLD/STOP" in result.stdout
    assert "--apply" not in result.stdout
