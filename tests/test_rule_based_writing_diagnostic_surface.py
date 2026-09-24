from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from urllib.request import urlopen

import pytest

from control_station.rule_based_writing.control_ui import RuleBasedWritingWebUi


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS_JS = (
    ROOT / "control_station" / "rule_based_writing" / "static" / "diagnostics.js"
)


def _classify(*, key: str = "", tag: str = "", message: str) -> dict[str, object]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required for the standalone browser classifier contract")
    payload = json.dumps({"key": key, "tag": tag, "message": message, "source": "blocker"})
    script = (
        "const d=require(process.argv[1]);"
        "const item=d.classifyDiagnostic(JSON.parse(process.argv[2]));"
        "process.stdout.write(JSON.stringify(item));"
    )
    completed = subprocess.run(
        [node, "-e", script, str(DIAGNOSTICS_JS), payload],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _build_active(state: dict[str, object]) -> list[dict[str, object]]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required for the standalone browser classifier contract")
    script = (
        "const d=require(process.argv[1]);"
        "const items=d.buildActiveDiagnostics(JSON.parse(process.argv[2]));"
        "process.stdout.write(JSON.stringify(items));"
    )
    completed = subprocess.run(
        [node, "-e", script, str(DIAGNOSTICS_JS), json.dumps(state)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


@pytest.mark.parametrize(
    ("key", "tag", "message", "domain", "component", "recovery"),
    [
        (
            "hand.right",
            "Device",
            "right_hand failed: connect_failed on /dev/ttyUSB0",
            "gripper",
            "right_gripper",
            "device_retry",
        ),
        (
            "hand.left",
            "DMTAC",
            "left_hand telemetry timeout",
            "gripper",
            "left_gripper",
            "device_retry",
        ),
        (
            "camera.front",
            "Camera",
            "front camera produced no fresh frame",
            "camera",
            "front_camera",
            "device_retry",
        ),
        (
            "camera.left_wrist",
            "CameraSelfCheck",
            "left_wrist camera startup failed",
            "camera",
            "left_wrist_camera",
            "device_retry",
        ),
        (
            "camera.right_wrist",
            "Camera",
            "right_wrist frame is stale",
            "camera",
            "right_wrist_camera",
            "device_retry",
        ),
        (
            "tactile.right",
            "Device",
            "right DMTAC camera tactile Bundle root is unavailable",
            "tactile",
            "tactile_capture",
            "device_retry",
        ),
        (
            "safety.force",
            "PositionForceGuard",
            "left RM65 force reading is stale: 0.8s",
            "arm_force",
            "left_arm",
            "force_telemetry_check",
        ),
        (
            "rule.writing_reachability",
            "RuleBased",
            "NoSolutionFound while checking writing reachability",
            "arm_ik",
            "right_arm_writing_path",
            "adjust_candidate",
        ),
        (
            "dataset.storage",
            "Dataset",
            "No space left on device; minimum free space is 10 GiB",
            "dataset_storage",
            "dataset_volume",
            "free_space",
        ),
        (
            "rig.session",
            "Device",
            "Session state=DEBUG_ONLY; runtime Home is required",
            "session",
            "runtime_session",
            "runtime_home",
        ),
    ],
)
def test_classifier_identifies_fault_domain_component_and_recovery(
    key: str,
    tag: str,
    message: str,
    domain: str,
    component: str,
    recovery: str,
) -> None:
    result = _classify(key=key, tag=tag, message=message)
    assert result["domain"] == domain
    assert result["component"] == component
    assert result["recovery"]["id"] == recovery


def test_force_trip_is_critical_but_ik_failure_is_not_a_device_retry() -> None:
    force = _classify(
        tag="PositionForceGuard",
        message="TRIPPED: right RM65 force reading is stale",
    )
    reachability = _classify(
        key="rule.writing_reachability",
        message="solver_guard_hold: NoSolutionFound",
    )
    assert force["severity"] == "critical"
    assert force["component"] == "right_arm"
    assert force["recovery"]["id"] == "force_telemetry_check"
    assert reachability["domain"] == "arm_ik"
    assert reachability["recovery"]["id"] == "adjust_candidate"
    assert reachability["recovery"]["id"] != "device_retry"


def test_terminal_session_does_not_advertise_in_process_runtime_home() -> None:
    diagnostic = _classify(
        key="rig.session",
        message="Session state=FAULT; runtime Home and hardware arm admission are required",
    )
    assert diagnostic["recovery"]["id"] == "restart_after_terminal_session"
    assert "重新启动程序" in diagnostic["recovery"]["instruction"]


def test_active_reachability_diagnostics_follow_runtime_lifecycle() -> None:
    blocker = {
        "rule.writing_reachability": "待检候选 4 数值 IK 未能认证；正在后台准备下一候选"
    }
    progress = _build_active(
        {
            "startup_blockers": blocker,
            "waiting_reason": "writing_reachability_skip",
            "writing_reachability_status": "skipping_candidate",
        }
    )[0]
    assert progress["planningMode"] == "progress"
    assert progress["severity"] == "warning"
    assert progress["recovery"]["id"] == "wait_candidate_check"

    exhausted = _build_active(
        {
            "startup_blockers": {
                "rule.writing_reachability": "连续 32 条候选未通过认证，已停止自动选择"
            },
            "waiting_reason": "writing_reachability_exhausted",
            "writing_reachability_status": "uncertified",
        }
    )[0]
    assert exhausted["planningMode"] == "candidate_rejected"
    assert exhausted["recovery"]["id"] == "adjust_candidate"

    failed = _build_active(
        {
            "startup_blockers": {
                "rule.writing_reachability": "书写可达性预检返回未知结果；请检查预检实现"
            },
            "waiting_reason": "writing_reachability_error",
            "writing_reachability_status": "error",
        }
    )[0]
    assert failed["planningMode"] == "error"
    assert failed["severity"] == "error"
    assert failed["recovery"]["id"] == "inspect_reachability_preflight"
    assert failed["recovery"]["id"] != "device_retry"


def test_held_session_recovery_precedes_pending_candidate_advice() -> None:
    diagnostics = _build_active(
        {
            "can_runtime_home": True,
            "startup_blockers": {
                "rig.session": "Session state=DEBUG_ONLY; runtime Home required",
                "rule.writing_reachability": "书写四边界接触点尚未完成可达性检查。",
            },
            "writing_reachability_status": "pending",
        }
    )
    planning = next(item for item in diagnostics if item["planningMode"] == "progress")
    assert planning["recovery"]["id"] == "runtime_home_then_wait_candidate_check"
    assert "重新安全初始化" in planning["recovery"]["instruction"]
    assert all(item["recovery"]["id"] != "device_retry" for item in diagnostics)


def test_right_hand_owner_requirement_is_not_misreported_as_disconnect() -> None:
    result = _classify(
        key="rule.right_hand_hold_owner",
        message="右夹爪尚未建立书写用的一次性闭合 owner。",
    )
    assert result["domain"] == "gripper"
    assert result["component"] == "right_gripper"
    assert result["recovery"]["id"] == "close_right_gripper"
    assert result["recovery"]["id"] != "device_retry"


def test_active_blockers_do_not_duplicate_generic_phase_detail() -> None:
    diagnostics = _build_active(
        {
            "phase": "blocked",
            "detail": "存在活动阻塞，请按分类处理。",
            "startup_blockers": {
                "hand.right": "right_hand failed: reconnect_failed",
                "camera.front": "front camera unavailable",
            },
        }
    )
    assert [item["domain"] for item in diagnostics] == ["gripper", "camera"]


def test_released_grip_requests_height_capture_not_device_reconnect() -> None:
    result = _classify(
        key="rule.writing_height",
        message="右夹爪已松开，原书写高度已失效；请闭紧夹爪后重新标定书写高度。",
    )
    assert result["recovery"]["id"] == "recalibrate_writing_height"


def test_ui_serves_diagnostic_asset_and_loads_it_before_application() -> None:
    ui = RuleBasedWritingWebUi(port=0, open_browser=False)
    ui.start()
    try:
        with urlopen(ui.url, timeout=2.0) as response:  # noqa: S310 - loopback fixture
            html = response.read().decode("utf-8")
        with urlopen(f"{ui.url}diagnostics.js", timeout=2.0) as response:  # noqa: S310
            script = response.read().decode("utf-8")
            content_type = response.headers["Content-Type"]
    finally:
        ui.close()
    assert html.index('<script src="/diagnostics.js"></script>') < html.index(
        '<script src="/app.js"></script>'
    )
    assert "RuleWritingDiagnostics" in script
    assert content_type.startswith("text/javascript")


def test_ui_has_separate_active_diagnostics_and_event_history() -> None:
    html = (
        ROOT / "control_station" / "rule_based_writing" / "static" / "index.html"
    ).read_text(encoding="utf-8")
    app = (
        ROOT / "control_station" / "rule_based_writing" / "static" / "app.js"
    ).read_text(encoding="utf-8")
    assert 'id="active-diagnostics"' in html
    assert "当前阻塞与故障" in html
    assert "事件历史（最近 12 条）" in html
    assert "buildActiveDiagnostics(state)" in app
    assert "renderMessageHistory(state.messages)" in app
