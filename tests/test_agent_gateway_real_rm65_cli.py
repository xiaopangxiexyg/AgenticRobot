from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_gateway.profiles.real import gateway_cli
from agent_gateway.profiles.real.one_shot_authorization import (
    REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _argv(*, confirmation: str, socket_path: Path | None = None) -> list[str]:
    existing = str(REPO_ROOT / "pyproject.toml")
    return [
        "--socket-path",
        str(
            REPO_ROOT / "temp/test-real-rm65-gateway.sock"
            if socket_path is None
            else socket_path
        ),
        "--calibration-profile",
        existing,
        "--arm-ip-assignment",
        existing,
        "--left-tool-profile",
        existing,
        "--right-tool-profile",
        existing,
        "--envelope-profile",
        existing,
        "--authorized-side",
        "right",
        "--authorized-action",
        "forward",
        "--confirm",
        confirmation,
        "--runtime-session-id",
        "real-rm65-cli:test",
        "--embodiment-instance-id",
        "dual-rm65:test",
    ]


def test_cli_rejects_confirmation_before_profile_or_hardware_resolution(
    monkeypatch,
) -> None:
    called = False

    def should_not_resolve(profile):
        nonlocal called
        called = True
        raise AssertionError("profile resolution must remain default-off")

    monkeypatch.setattr(
        gateway_cli,
        "resolve_exclusive_rm65_skill_deployment_profile",
        should_not_resolve,
    )

    with pytest.raises(SystemExit):
        gateway_cli.main(_argv(confirmation="wrong"))

    assert called is False


def test_cli_accepts_a_list_of_reviewed_actions_for_temporary_testing() -> None:
    argv = _argv(confirmation=REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN)
    action_index = argv.index("--authorized-action")
    del argv[action_index : action_index + 2]
    argv.extend(
        [
            "--authorized-actions",
            "forward",
            "backward",
            "left",
            "right",
            "up",
            "down",
            "level_forward",
            "level_forward_left_30",
            "level_forward_right_30",
        ]
    )
    parsed = gateway_cli.build_parser().parse_args(argv)
    assert parsed.authorized_action is None
    assert tuple(parsed.authorized_actions) == (
        "forward",
        "backward",
        "left",
        "right",
        "up",
        "down",
        "level_forward",
        "level_forward_left_30",
        "level_forward_right_30",
    )


def test_cli_rejects_partial_perception_attachment_before_profile_resolution(
    monkeypatch,
) -> None:
    called = False

    def should_not_resolve(profile):
        nonlocal called
        called = True
        raise AssertionError("partial perception flags must fail first")

    monkeypatch.setattr(
        gateway_cli,
        "resolve_exclusive_rm65_skill_deployment_profile",
        should_not_resolve,
    )
    argv = _argv(confirmation=REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN)
    argv.extend(
        [
            "--camera-config",
            str(
                REPO_ROOT
                / "teleop_runtime/configs/devices/agent_gateway_realsense.json"
            ),
        ]
    )

    with pytest.raises(SystemExit):
        gateway_cli.main(argv)

    assert called is False


def test_cli_rejects_overlong_socket_before_profile_or_hardware_assembly(
    monkeypatch,
    capsys,
) -> None:
    profile_resolved = False
    lower_assembled = False

    def should_not_resolve(_profile):
        nonlocal profile_resolved
        profile_resolved = True
        raise AssertionError("socket preflight must precede profile resolution")

    def should_not_assemble(*, profile):
        del profile
        nonlocal lower_assembled
        lower_assembled = True
        raise AssertionError("socket preflight must precede hardware assembly")

    monkeypatch.setattr(
        gateway_cli,
        "resolve_exclusive_rm65_skill_deployment_profile",
        should_not_resolve,
    )
    monkeypatch.setattr(
        gateway_cli,
        "assemble_exclusive_rm65_skill_lower",
        should_not_assemble,
    )
    argv = _argv(confirmation=REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN)
    argv[1] = "/" + "a" * 108

    with pytest.raises(SystemExit) as captured:
        gateway_cli.main(argv)

    assert captured.value.code == 2
    assert profile_resolved is False
    assert lower_assembled is False
    error = capsys.readouterr().err
    assert "--socket-path preflight failed" in error
    assert "uses 109 encoded bytes" in error
    assert "will not relocate it automatically" in error
    assert "Traceback" not in error


def test_cli_assembles_then_reports_ready_and_lower_hold(
    monkeypatch,
    capsys,
    short_unix_socket_path_factory,
) -> None:
    resolved = SimpleNamespace(semantic_content_hash="a" * 64)
    lower = object()
    captured = {}

    class _Application:
        state = gateway_cli.ExclusiveRealSkillApplicationState.CREATED

        def serve_forever(self, *, ready_callback=None):
            assert ready_callback is not None
            self.state = gateway_cli.ExclusiveRealSkillApplicationState.RUNNING
            ready_callback()
            time.sleep(0.05)
            self.state = gateway_cli.ExclusiveRealSkillApplicationState.CLOSED

        def request_shutdown(self):
            return False

    route = SimpleNamespace(
        skill_id="motion.step_tcp_cardinal",
        skill_version="1",
        route_id="real.rm65.agent_skill_tcp_cardinal_step",
        route_version="1",
    )
    entry = SimpleNamespace(route=route, semantic_content_hash="route-entry:test")
    manifest = SimpleNamespace(
        target_kind="real",
        current_mode="real_active",
        runtime_session_id="real-rm65-cli:test",
        control_domain_id="exclusive-rm65-agent-skill",
        target_binding_id="real-rm65-exclusive:test",
        semantic_content_hash="manifest:test",
        station_manifest_hash="station:test",
        execution_manifest_hash="execution:test",
        skill_routes=(entry,),
    )
    hold = SimpleNamespace(succeeded=True, evidence_refs=("hold:test",))
    motion_port = SimpleNamespace(
        snapshot=lambda: SimpleNamespace(dispatch_count=0),
    )
    runtime = SimpleNamespace(motion_port=motion_port, last_hold_receipt=hold)
    socket_path = short_unix_socket_path_factory("real-cli")
    composition = SimpleNamespace(
        socket_path=socket_path,
        client_id="rpent.dexteleop",
        authorized_side="right",
        authorized_action="forward",
        lower=SimpleNamespace(profile=resolved),
        deployment=SimpleNamespace(
            gateway_session=SimpleNamespace(
                contract_bundle=SimpleNamespace(route_manifest=manifest)
            ),
            control_runtime=runtime,
        ),
        application=_Application(),
    )

    monkeypatch.setattr(
        gateway_cli,
        "resolve_exclusive_rm65_skill_deployment_profile",
        lambda profile: resolved,
    )
    monkeypatch.setattr(
        gateway_cli,
        "assemble_exclusive_rm65_skill_lower",
        lambda *, profile: lower,
    )

    def assemble(**kwargs):
        captured.update(kwargs)
        return composition

    monkeypatch.setattr(
        gateway_cli,
        "assemble_exclusive_rm65_agent_primitive_composition",
        assemble,
    )

    assert gateway_cli.main(
        _argv(
            confirmation=REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN,
            socket_path=socket_path,
        )
    ) == 0

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["event"] for event in events] == [
        "gateway_ready",
        "gateway_stopped",
    ]
    assert events[0]["authorization_scope"] == "one_matching_skill_request"
    assert events[1]["lower_hold_proved"] is True
    assert captured["lower"] is lower
    assert captured["authorized_side"] == "right"
    assert captured["authorized_action"] == "forward"
