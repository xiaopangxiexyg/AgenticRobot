from __future__ import annotations

import json
import signal
from types import SimpleNamespace

import pytest

from agent_gateway import __main__ as dispatcher
from agent_gateway.profiles.real import readonly_tcp_cli as cli


@pytest.mark.parametrize("failure", ["sampler", "publication", "server_thread"])
def test_cli_failed_admission_closes_all_owned_resources_without_publishing_ready(
    monkeypatch, tmp_path, capsys, failure,
):
    events = []
    bundle = SimpleNamespace(attachment=SimpleNamespace(runtime_session_id="readonly:test"))

    def fail(stage):
        events.append(stage)
        if failure == stage:
            raise RuntimeError(f"{stage} admission failed")

    class Sampler:
        def start(self):
            fail("sampler")

        def close(self):
            events.append("sampler.close")

    class Server:
        def serve_forever(self, **kwargs):
            raise AssertionError("Unstarted server must not serve")

        def shutdown(self):
            raise AssertionError("Unstarted server must not be shut down")

        def server_close(self):
            events.append("server.close")

    class Thread:
        ident = None

        def __init__(self, **kwargs):
            pass

        def start(self):
            fail("server_thread")

    def publish(service):
        fail("publication")

    def build_server(**kwargs):
        events.append("server.build")
        return Server()

    session = SimpleNamespace(
        observation_producer=SimpleNamespace(capture_and_publish=publish), service=object(),
        build_unix_json_server=build_server, close=lambda: events.append("session.close"),
    )
    monkeypatch.setattr(cli, "validate_agent_gateway_unix_server_preflight", lambda **kwargs: None)
    monkeypatch.setattr(cli, "resolve_exclusive_rm65_skill_deployment_profile", lambda _: SimpleNamespace(
        arm_ip_by_side={"left": "left.fixture", "right": "right.fixture"},
    ))
    monkeypatch.setattr(cli, "build_readonly_rm65_tcp_contract", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(cli, "RM65ReadonlyTcpSampler", lambda _: Sampler())
    monkeypatch.setattr(cli, "ReadonlyRM65TcpSource", lambda *args: object())
    monkeypatch.setattr(cli, "assemble_real_tcp_pose_observe_gateway_session", lambda **kwargs: session)
    monkeypatch.setattr(cli.threading, "Thread", Thread)
    handlers = {}

    def register(sig, handler):
        handlers.setdefault(sig, []).append(handler)
        return "previous-handler"

    monkeypatch.setattr(cli.signal, "signal", register)
    args = ["--client-id", "test"]
    for name in ("socket-path", "arm-ip-assignment", "left-tool-profile", "right-tool-profile", "envelope-profile"):
        args += ["--" + name, str(tmp_path / name)]
    with pytest.raises(RuntimeError, match=f"{failure} admission failed"):
        cli.main(args)
    assert events[-2:] == ["session.close", "sampler.close"]
    if failure == "server_thread":
        assert "server.close" in events
    else:
        assert "server.build" not in events
    for sig in (signal.SIGINT, signal.SIGTERM):
        assert len(handlers[sig]) == 2 and handlers[sig][-1] == "previous-handler"
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [record["event"] for record in records] == ["gateway_stopped"]
    assert records[0]["cleanup_complete"]
    assert records[0]["command_owner_count"] == 0


def test_dispatcher_routes_readonly_tcp_without_loading_motion_cli(monkeypatch):
    monkeypatch.setattr(cli, "main", lambda args: 17 if args == ["--help"] else 18)
    assert dispatcher.main(["real-tcp-observe", "--help"]) == 17


def test_cli_rejects_relative_profile_inputs_before_preflight(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "validate_agent_gateway_unix_server_preflight", lambda **kwargs: calls.append(kwargs))
    args = ["--client-id", "test"]
    for name in ("socket-path", "arm-ip-assignment", "left-tool-profile", "right-tool-profile", "envelope-profile"):
        args += ["--" + name, "relative"]
    with pytest.raises(SystemExit):
        cli.main(args)
    assert calls == []
