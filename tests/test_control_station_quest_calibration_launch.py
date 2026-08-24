from __future__ import annotations

import json
import os
import stat
import threading
from pathlib import Path

import pytest

from control_station.quest_calibration.launch import (
    OPERATOR_TRIAL_PROFILE_ID_V1,
    TRUSTED_LAN_EASY_PAIRING_CODE,
    build_deploy_argv,
    build_parser,
    operator_trial_profile_v1,
    run,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


class _LaunchInfo:
    base_url = "https://192.168.70.2:8444/"
    pairing_url = "https://192.168.70.2:8444/#pair=111111"
    pairing_code = "111111"


class _FakeFactory:
    def __init__(self, config) -> None:
        self.config = config
        self.launch_info = _LaunchInfo()
        self.wait_called = threading.Event()

    def wait_for_application(self, *, timeout_s: float) -> object:
        assert timeout_s == 30.0
        self.wait_called.set()
        return object()


class _FakeCoordinator:
    url = "http://127.0.0.1:45678/"

    def __init__(self, factory, **kwargs) -> None:
        self.factory = factory
        self.kwargs = kwargs
        self.start_calls = 0
        self.close_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    def close(self) -> None:
        self.close_calls += 1


def _args(*extra: str):
    return build_parser().parse_args(
        [
            "--side",
            "right",
            "--arm-ip",
            "192.168.70.18",
            "--capture-id",
            "operator-test-capture",
            *extra,
        ]
    )


def test_operator_trial_profile_is_explicit_candidate_admission() -> None:
    first = operator_trial_profile_v1()
    second = operator_trial_profile_v1()

    assert first.profile_id == OPERATOR_TRIAL_PROFILE_ID_V1
    assert first.semantic_content_hash == second.semantic_content_hash
    assert first.min_window_duration_ns == 1_000_000_000
    assert first.min_robot_samples == 5
    assert first.min_quest_samples == 10
    assert first.max_completed_windows == 32


def test_deploy_arguments_are_closed_over_the_no_action_profile() -> None:
    argv = build_deploy_argv(_args())

    assert argv == [
        "--policy",
        "hold",
        "--rm65-pose-observation",
        "--active-sides",
        "right",
        "--right-arm-ip",
        "192.168.70.18",
        "--end-effector-sides",
        "none",
        "--camera-mode",
        "none",
        "--arm-control-mode",
        "position",
        "--arm-state-refresh-hz",
        "10.0",
    ]
    for forbidden in ("--send-actions", "--auto-start", "--allow-arm-connect-failure"):
        assert forbidden not in argv


def test_dry_run_prints_resolved_contract_without_constructing_runtime() -> None:
    output: list[str] = []

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dry-run must not construct runtime objects")

    result = run(
        _args("--trusted-lan", "--dry-run"),
        factory_builder=forbidden,
        coordinator_builder=forbidden,
        policy_runner=forbidden,
        emit=output.append,
    )
    payload = json.loads(output[0])

    assert result == 0
    assert payload["mode"] == "quest_rm65_passive_calibration_trial"
    assert payload["quest_pairing"] == {
        "fixed_code": TRUSTED_LAN_EASY_PAIRING_CODE,
        "mode": "trusted_lan_easy",
        "token_env": "QUEST_DEV_PAIRING_TOKEN",
    }
    assert payload["safety"] == {
        "artifact_publication": False,
        "calibration_acceptance": False,
        "cameras": False,
        "end_effectors": False,
        "policy": "hold",
        "send_actions": False,
    }


def test_run_composes_one_factory_and_restores_pairing_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factories: list[_FakeFactory] = []
    coordinators: list[_FakeCoordinator] = []
    policy_calls: list[tuple[list[str], object]] = []
    output: list[str] = []

    def build_factory(config):
        factory = _FakeFactory(config)
        factories.append(factory)
        return factory

    def build_coordinator(factory, **kwargs):
        coordinator = _FakeCoordinator(factory, **kwargs)
        coordinators.append(coordinator)
        return coordinator

    def policy_runner(*, argv, quest_paired_pose_capture_factory):
        factory = factories[0]
        assert factory.wait_called.wait(timeout=1.0)
        assert os.environ["QUEST_DEV_PAIRING_TOKEN"] == "111111"
        policy_calls.append((argv, quest_paired_pose_capture_factory))
        return 7

    monkeypatch.setenv("QUEST_DEV_PAIRING_TOKEN", "654321")
    result = run(
        _args("--trusted-lan", "--no-open-browser"),
        factory_builder=build_factory,
        coordinator_builder=build_coordinator,
        policy_runner=policy_runner,
        emit=output.append,
    )

    assert result == 7
    assert len(factories) == 1
    assert factories[0].config.force_http is False
    assert factories[0].config.capture_id == "operator-test-capture"
    assert factories[0].config.profile.profile_id == OPERATOR_TRIAL_PROFILE_ID_V1
    assert policy_calls == [(build_deploy_argv(_args("--trusted-lan")), factories[0])]
    assert coordinators[0].start_calls == 1
    assert coordinators[0].close_calls == 1
    assert coordinators[0].kwargs["open_browser"] is False
    assert os.environ["QUEST_DEV_PAIRING_TOKEN"] == "654321"
    assert any("Quest 打开: https://192.168.70.2:8444/" in line for line in output)
    assert any("Quest 配对六码: 111111" in line for line in output)


def test_policy_failure_still_closes_station_and_clears_new_pairing_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _FakeFactory(None)
    coordinator = _FakeCoordinator(factory)
    monkeypatch.delenv("QUEST_DEV_PAIRING_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="policy failed"):
        run(
            _args("--pairing-code", "246810"),
            factory_builder=lambda _config: factory,
            coordinator_builder=lambda _factory, **_kwargs: coordinator,
            policy_runner=lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("policy failed")
            ),
            emit=lambda _message: None,
        )

    assert coordinator.close_calls == 1
    assert "QUEST_DEV_PAIRING_TOKEN" not in os.environ


def test_random_pairing_mode_ignores_and_restores_ambient_fixed_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _FakeFactory(None)
    coordinator = _FakeCoordinator(factory)

    def policy_runner(**_kwargs):
        assert factory.wait_called.wait(timeout=1.0)
        assert "QUEST_DEV_PAIRING_TOKEN" not in os.environ
        return 0

    monkeypatch.setenv("QUEST_DEV_PAIRING_TOKEN", "654321")
    result = run(
        _args(),
        factory_builder=lambda _config: factory,
        coordinator_builder=lambda _factory, **_kwargs: coordinator,
        policy_runner=policy_runner,
        emit=lambda _message: None,
    )

    assert result == 0
    assert os.environ["QUEST_DEV_PAIRING_TOKEN"] == "654321"


def test_cli_rejects_unsafe_or_ambiguous_pairing_options() -> None:
    with pytest.raises(SystemExit):
        _args("--pairing-code", "123")
    with pytest.raises(SystemExit):
        _args("--trusted-lan", "--pairing-code", "123456")


def test_shell_entrypoint_is_executable_and_delegates_to_module() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_quest_calibration.sh"
    content = script.read_text(encoding="utf-8")

    assert script.stat().st_mode & stat.S_IXUSR
    assert "resolve_dev_python.sh" in content
    assert "control_station.quest_calibration.launch" in content
    assert "--send-actions" not in content

    html = (root / "control_station/quest_calibration/static/index.html").read_text(
        encoding="utf-8"
    )
    javascript = (root / "control_station/quest_calibration/static/app.js").read_text(
        encoding="utf-8"
    )
    assert "封存候选（当前仅内存）" in html
    assert "完成窗口（自动排空）" in html
    assert "completeWindowWithBoundedDrain" in javascript
    assert "attempt < 20" in javascript
