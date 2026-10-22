"""Offline contract checks; no motor or CAN connection is constructed."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from control_station.b601_torque_console import B601TorqueConsoleServer
from control_station.b601_torque_console.report import main as report_main
from control_station.b601_torque_console.report import summarize_trace
from control_station.b601_torque_console.server import recent_runtime_notices
from robot_motion.rebot_b601_rs.torque_experiment import (
    B601MotionOnsetCaptureConfig,
    B601OnsetProbeConfig,
    B601TorqueExperiment,
    B601TorqueExperimentLimits,
)
from scripts.hardware import rebot_b601_rs_low_gain_probe as probe
from scripts.hardware import rebot_b601_rs_owner as owner

ZERO = (0.0,) * 6


def test_gui_six_axis_degrees_use_owner_snapshot_and_mark_stale_or_missing_feedback():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for GUI contracts")
    source = Path("control_station/b601_torque_console/static/assessment.js").read_text()
    script = """
const vm = require('node:vm'), elements = {}, listeners = {};
const make = () => ({cells: [], classList: {toggle(k,v){this[k]=v;}},
  append(...children){this.cells.push(...children);},setAttribute(k,v){this[k]=v;}});
const el = id => elements[id] ||= make();
const c = {document:{getElementById:el,createElement:make,
  querySelector:()=>({content:'csrf'}),addEventListener:(name,fn)=>listeners[name]=fn}};
vm.runInNewContext(JSON.parse(process.argv[1]),c);
const base = {controller_mode:'automatic_calibration',qpos_rad:[0,.3,.4,-.1,.2,-.2],
  qvel_fd_rad_s:[0,0,0,0,0,0],feedback_age_sec:.02,dispatch_age_sec:.03,
  last_dispatch:{joint_commands:{gravity_nm:[1,2,3,4,5,6],overlay_nm:[0,0,0,0,0,0],total_nm:[1,2,3,4,5,6]}}};
const render = value => {listeners['b601-status']({detail:value});return {
  rows:el('joint-rows').cells.map(row=>row.cells.map(cell=>cell.textContent)),
  status:el('joint-feedback-status').textContent,stale:el('joint-overview').classList['stale-feedback']};};
console.log(JSON.stringify({fresh:render(base),updated:render({...base,qpos_rad:[0,.4,.5,0,0,0]}),
  stale:render({...base,feedback_age_sec:1,dispatch_age_sec:1}),
  invalidClock:render({...base,feedback_age_sec:-.01,dispatch_age_sec:-.01}),
  incomplete:render({...base,qpos_rad:[0,.3,.4]}),missing:render(null)}));
"""
    result = subprocess.run([node, "-e", script, json.dumps(source)],
        check=True, capture_output=True, text=True)
    views = json.loads(result.stdout)
    assert len(views["fresh"]["rows"]) == 6
    assert all(len(row) == 8 for row in views["fresh"]["rows"])
    assert views["fresh"]["rows"][1][1:3] == ["17.189", "0.30000"]
    assert views["fresh"]["rows"][3][1] == "-5.730"
    assert views["updated"]["rows"][1][1:3] == ["22.918", "0.40000"]
    assert views["fresh"]["stale"] is False and "反馈更新中" in views["fresh"]["status"]
    for name in ("stale", "invalidClock"):
        assert views[name]["stale"] is True
        assert "不是实时测量" in views[name]["status"]
        assert views[name]["rows"][1][5:] == ["—", "—", "—"]
    for name in ("missing", "incomplete"):
        assert views[name]["stale"] is True
        assert all(row[1:3] == ["—", "—"] for row in views[name]["rows"])
    html = Path("control_station/b601_torque_console/static/index.html").read_text()
    assert html.index('id="mode-selector"') < html.index('id="supported-return-panel"') < html.index('id="joint-overview"')
    assert html.count('id="joint-overview"') == 1


@pytest.mark.parametrize("joint,limit,independent_onset,accepted", [
    ("joint3", .05, False, True),
    ("joint3", .07, False, False),
    ("joint3", .07, True, True),
    ("joint3", .070001, True, False),
    ("joint2", .07, True, False),
])
def test_acquisition_velocity_ceiling_preserves_ordinary_debug(joint, limit, independent_onset, accepted):
    limits = B601TorqueExperimentLimits(joint, max_joint_velocity_rad_s=limit)
    capture = B601MotionOnsetCaptureConfig(raw_joint_velocity_limit_rad_s=.15) if independent_onset else None
    if accepted:
        controller = B601TorqueExperiment(limits, quantized_velocity_debug=True, onset_capture=capture)
        assert controller.limits.max_joint_velocity_rad_s == limit
    else:
        with pytest.raises(ValueError, match="hard limit at most"):
            B601TorqueExperiment(limits, quantized_velocity_debug=True, onset_capture=capture)


@pytest.mark.parametrize("limit,accepted", [(.10,True),(.13,True),(.130001,True),(.195,True),(.195001,False)])
def test_explicit_release_acquisition_accepts_only_authorized_raw_speed_ceiling(limit,accepted):
    limits = B601TorqueExperimentLimits("joint3",max_joint_velocity_rad_s=limit)
    capture = B601MotionOnsetCaptureConfig(raw_joint_velocity_limit_rad_s=.15,release_support_confirmation=True)
    if accepted:
        controller = B601TorqueExperiment(limits,quantized_velocity_debug=True,
            onset_capture=capture,onset_probe=B601OnsetProbeConfig())
        assert controller.limits.max_joint_velocity_rad_s == limit
        assert controller._hard_velocity_limit(2) == .15
        assert controller._hard_velocity_limit(3) == limit
    else:
        with pytest.raises(ValueError,match="hard limit at most 0.195"):
            B601TorqueExperiment(limits,quantized_velocity_debug=True,
                onset_capture=capture,onset_probe=B601OnsetProbeConfig())


def test_direct_gui_distinguishes_permission_target_dispatch_and_stale_receipts():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for GUI contracts")
    source = Path("control_station/b601_torque_console/static/app.js").read_text()
    functions = source[source.index("function number("):source.index("function post(")]
    functions += source[source.index("function lastOperatorEvent("):source.index("async function refresh(")]
    script = """
const vm = require('node:vm'); const c = {};
vm.runInNewContext(JSON.parse(process.argv[1]), c);
const base = {phase:'armed', owner_ready:true, feedback_age_sec:.01,
  controller_mode:'direct_torque', joint_name:'joint3', dispatch_age_sec:.01,
  desired_assist_nm:0, effective_assist_nm:0,
  last_dispatch:{joint_name:'joint3', feedback_timestamp_sec:1.1, overlay_after_profile_nm:0,
    gravity_nm:6.2,conditioned_total_nm:6.2,
    joint_commands:{gravity_nm:[0,0,6.2,0,0,0],overlay_nm:[0,0,0,0,0,0],total_nm:[0,0,6.2,0,0,0]}},
  events:[{action:'arm',time_sec:1}], onset_capture:{trial:null,probe:{state:null}}};
const target = {...base,desired_assist_nm:-.3,effective_assist_nm:-.02,
  events:[{action:'set',time_sec:1.2,requested_nm:-.3}]};
const sent = {...target,last_dispatch:{...base.last_dispatch,feedback_timestamp_sec:1.21,overlay_after_profile_nm:-.02,conditioned_total_nm:6.18}};
const values = [base, target, sent, {...sent,onset_capture:{probe:{state:{stage:'hold'}}}},
  {...sent,dispatch_age_sec:.3}, {...sent,dispatch_age_sec:-.01},
  {...sent,last_dispatch:{...sent.last_dispatch,joint_name:'joint2'}},
  {...base,last_dispatch:{...base.last_dispatch,overlay_after_profile_nm:-.02}},
  {...base,phase:'onset_release'}, {...base,phase:'inactive'},
  {...sent,feedback_age_sec:.3}, {...base,phase:'fault'}];
console.log(JSON.stringify({labels:values.map(c.directDriveLabel),
  zeroGate:c.gateMessage(base),staleCommand:c.commandState({...sent,dispatch_age_sec:.3})}));
"""
    result = subprocess.run([node, "-e", script, json.dumps(functions)], check=True, capture_output=True, text=True)
    messages = json.loads(result.stdout)
    labels = messages["labels"]
    assert "零目标，最终附加命令为零" in labels[0]
    assert "等待加力发送确认" in labels[1]
    assert "正在加力" in labels[2]
    assert "定力观察" in labels[3]
    assert all("等待新鲜发送回执" in label for label in labels[4:7])
    assert "等待最终归零回执" in labels[7]
    assert "撤力与停稳" in labels[8]
    assert labels[9] == "仅重力补偿"
    assert labels[10] == "反馈状态待确认" and labels[11] == "保护已触发"
    assert "尚未开始采集" in messages["zeroGate"] and "低速保护仍启用" in messages["zeroGate"]
    assert "等待 owner 完整发送回执" in messages["staleCommand"]


def test_direct_gui_zero_claim_requires_complete_final_command_receipt():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for GUI contracts")
    source = Path("control_station/b601_torque_console/static/app.js").read_text()
    functions = source[source.index("const STAGED_CONSUMED_PHASES"):source.index("function post(")]
    functions += source[source.index("function lastOperatorEvent("):source.index("async function refresh(")]
    script = """
const vm = require('node:vm'); const c = {};
vm.runInNewContext(JSON.parse(process.argv[1]), c);
const receipt = {joint_name:'joint3', feedback_timestamp_sec:1.1, overlay_after_profile_nm:0,
  gravity_nm:6.2, conditioned_total_nm:6.2,
  joint_commands:{gravity_nm:[0,0,6.2,0,0,0],overlay_nm:[0,0,0,0,0,0],total_nm:[0,0,6.2,0,0,0]}};
const base = {phase:'armed', owner_ready:true, feedback_age_sec:.01, dispatch_age_sec:.01,
  controller_mode:'direct_torque', joint_name:'joint3', quantized_velocity_debug:{},
  desired_assist_nm:0,effective_assist_nm:0,events:[{action:'arm',time_sec:1}],last_dispatch:receipt};
const tails = [{...receipt,conditioned_total_nm:6.23},
  {...receipt,joint_commands:{...receipt.joint_commands,total_nm:[0,0,6.2,.03,0,0]}},
  {...receipt,joint_commands:undefined}, {...receipt,gravity_nm:NaN},
  {...receipt,joint_commands:{...receipt.joint_commands,total_nm:[0,0,6.2,0,NaN,0]}},
  {...receipt,joint_commands:{...receipt.joint_commands,overlay_nm:[0,0,0,0,0]}}];
const armed = tails.map(last_dispatch => ({...base,last_dispatch}));
const withdraw = {...base,phase:'inactive',events:[{action:'withdraw',time_sec:1}]};
const zero = {...base,events:[{action:'zero',time_sec:1,requested_nm:0}],last_dispatch:tails[0]};
const pending = {...base,desired_assist_nm:.03,effective_assist_nm:.03,
  events:[{action:'set',time_sec:1,requested_nm:.03}],last_dispatch:{...receipt,overlay_after_profile_nm:.03}};
console.log(JSON.stringify({valid:c.finalZeroConfirmed(receipt),
  zeroFlags:tails.map(c.finalZeroConfirmed),labels:armed.map(c.directDriveLabel),
  armMessages:armed.map(c.commandState),
  inactive:tails.map(last_dispatch => c.directDriveLabel({...withdraw,last_dispatch})),
  withdraw:tails.map(last_dispatch => c.commandState({...withdraw,last_dispatch})),
  zeroMessage:c.commandState(zero),finalDelta:c.finalCommandDelta(tails[0]),
  pendingLabel:c.directDriveLabel(pending)}));
"""
    result = subprocess.run([node, "-e", script, json.dumps(functions)], check=True, capture_output=True, text=True)
    messages = json.loads(result.stdout)
    assert messages["valid"] is True and messages["zeroFlags"] == [False] * 6
    assert all("等待最终归零回执" in label for label in messages["labels"])
    assert all("尚未确认" in label for label in messages["armMessages"])
    assert all("等待最终零助力" in label for label in messages["inactive"])
    assert all("撤力已由 owner 确认" not in label for label in messages["withdraw"])
    assert "0.030 N·m" in messages["zeroMessage"]
    assert messages["finalDelta"] == pytest.approx(.03)
    assert "等待加力发送确认" in messages["pendingLabel"]


def test_gui_visibility_policy_backgrounds_only_the_full_cycle():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for GUI contracts")
    source = Path("control_station/b601_torque_console/static/app.js").read_text()
    helper = source[source.index("function pageVisibilityIntent("):source.index("function renderAutomatic(")]
    script = """
const vm = require('node:vm'); const c = {};
vm.runInNewContext(JSON.parse(process.argv[1]), c);
console.log(JSON.stringify([
  c.pageVisibilityIntent(true, true),
  c.pageVisibilityIntent(false, true),
  c.pageVisibilityIntent(true, false),
  c.pageVisibilityIntent(false, false),
]));
"""
    result = subprocess.run(
        [node, "-e", script, json.dumps(helper)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == [
        {"action": "visibility", "visible": False},
        {"action": "visibility", "visible": True},
        {"action": "withdraw", "source": "page_hidden"},
        None,
    ]


def test_status_response_diagnostics_preserve_payload_and_use_preceding_completed_request():
    class Controller:
        def status(self):
            return {"phase": "inactive", "history": [{"time_sec": 1., "onset_capture": {"trial": None}}],
                    "last_dispatch": {"receipt_sequence": 42, "feedback_timestamp_sec": 1.}}
        def submit(self, *args, **kwargs):
            pytest.fail("status observation must not submit intent")
    server = B601TorqueConsoleServer(Controller())
    server.start()
    try:
        with urlopen(server.url + "/api/status", timeout=2) as response:
            first = json.load(response)
        assert first["history"] == [{"time_sec": 1., "onset_capture": {"trial": None}}]
        assert first["http_status_diagnostics"]["last_completed"] is None
        # Wait on a bounded observation, not on an assumed handler completion time.
        deadline = time.monotonic()+2
        while server.status_response_diagnostics()["completed_count"] < 1 and time.monotonic() < deadline:
            time.sleep(.001)
        with urlopen(server.url + "/api/status", timeout=2) as response:
            second = json.load(response)
        previous = second["http_status_diagnostics"]["last_completed"]
        assert previous["sequence"] == 1 and previous["completed"] is True
        assert previous["owner_receipt_sequence"] == 42 and previous["owner_feedback_timestamp_sec"] == 1.
        assert previous["response_bytes"] > 0
        assert set(previous["stages"]) == {"snapshot", "encode", "write"}
        assert all(t["wall_sec"] >= 0 and t["thread_cpu_sec"] >= 0 for t in previous["stages"].values())
        assert previous["completed_perf_counter_sec"] >= previous["started_perf_counter_sec"]
        assert "records" not in second["http_status_diagnostics"]
    finally:
        server.close()


def test_status_response_write_failure_is_recorded_without_retry_or_intent(monkeypatch):
    from control_station.b601_torque_console import server as console_module
    controller = type("Controller", (), {"status": lambda self: {"phase": "inactive"}})()
    server = B601TorqueConsoleServer(controller)
    handler = object.__new__(console_module._Handler)
    handler.server = server._server
    def fail_send(*args):
        raise BrokenPipeError("offline disconnected client")
    monkeypatch.setattr(handler, "_send", fail_send)
    try:
        with pytest.raises(BrokenPipeError, match="offline disconnected"):
            handler._status_response()
        diagnostics = server.status_response_diagnostics()
        assert diagnostics["started_count"] == diagnostics["completed_count"] == 1
        assert diagnostics["records"][0]["completed"] is False
        assert "write" not in diagnostics["records"][0]["stages"]
    finally:
        server.close()


def test_status_response_diagnostics_are_bounded_copied_and_completion_ordered():
    server = B601TorqueConsoleServer(type("Controller", (), {"status": lambda self: {}})())
    try:
        for _ in range(4097):
            seq = server._server.begin_status_response()
            server._server.record_status_response({"sequence": seq, "stages": {"encode": {"wall_sec": .001}}})
        diagnostics = server.status_response_diagnostics()
        assert diagnostics["completed_count"] == 4097 and diagnostics["dropped_record_count"] == 1
        assert len(diagnostics["records"]) == 4096 and diagnostics["records"][0]["sequence"] == 2
        diagnostics["records"][-1]["stages"]["encode"]["wall_sec"] = 99
        assert server.status_response_diagnostics()["last_completed"]["stages"]["encode"]["wall_sec"] == .001
        # An overlapping later request may finish first; arrival sequence is not completion order.
        a, b = server._server.begin_status_response(), server._server.begin_status_response()
        server._server.record_status_response({"sequence": b})
        server._server.record_status_response({"sequence": a})
        assert server.status_response_diagnostics()["last_completed"]["sequence"] == a
    finally:
        server.close()


@pytest.mark.parametrize("mode", ["direct_torque", "staged_assist"])
def test_operator_status_projects_chart_history_only_and_preserves_full_endpoint(mode):
    history = [{"time_sec": i / 100, "desired_assist_nm": .2, "target_offset_rad": .02,
                "effective_assist_nm": .1, "overlay_after_profile_nm": .1, "displacement_rad": i / 10000,
                "onset_capture": {"unused_trace": "x" * 1000}, "qpos_rad": [i / 10000] * 6}
               for i in range(240)]
    current = {"controller_mode": mode, "phase": "armed", "owner_ready": True,
               "operator_armed": True, "history": history, "feedback_age_sec": .01,
               "dispatch_age_sec": .01, "last_dispatch": {"receipt_sequence": 42,
                   "joint_name": "joint3", "gravity_nm": 6.2, "conditioned_total_nm": 6.3},
               "limits": {"max_abs_assist_nm": .8, "max_joint_velocity_rad_s": .05},
               "onset_capture": {"probe": {"human_intervention": False}},
               "breakaway_eligible": True, "target_update_pending": False, "policy_authority": .5}

    class Controller:
        def status(self):
            return dict(current)

        def submit(self, *args, **kwargs):
            pytest.fail("status projection must not submit intent")

    server = B601TorqueConsoleServer(Controller(), status_provider=lambda: {"terminal_statuses": ["offline"]})
    server.start()
    try:
        def fetch(suffix):
            with urlopen(server.url + suffix, timeout=2) as response:
                body = response.read()
                return json.loads(body), len(body)

        full, full_bytes = fetch("/api/status")
        operator, operator_bytes = fetch("/api/status?view=operator")
        fields = {"time_sec", "desired_assist_nm", "target_offset_rad", "effective_assist_nm",
                  "overlay_after_profile_nm", "displacement_rad"}
        assert full["history"] == history and len(operator["history"]) == 240
        assert operator["history"] == [{k: row[k] for k in fields} for row in history]
        for key in full.keys() - {"history", "http_status_diagnostics"}:
            assert operator[key] == full[key]
        assert operator_bytes < full_bytes / 4  # Byte shape only, not a timing acceptance criterion.
        assert current["history"] == history and "onset_capture" in history[0]
        again, _ = fetch("/api/status")
        assert again["history"] == history
        assert server.status_response_diagnostics()["started_count"] == 3
        deadline = time.monotonic() + 2
        while server.status_response_diagnostics()["completed_count"] < 3 and time.monotonic() < deadline:
            time.sleep(.001)
        records = sorted(server.status_response_diagnostics()["records"], key=lambda r: r["sequence"])
        assert [r["view"] for r in records] == ["full", "operator", "full"]
    finally:
        server.close()


def test_operator_status_unknown_query_is_rejected_without_controller_observation():
    class Controller:
        def status(self):
            pytest.fail("invalid status view must not observe controller")

        def submit(self, *args, **kwargs):
            pytest.fail("status view must not submit intent")

    server = B601TorqueConsoleServer(Controller())
    server.start()
    try:
        for query in ("?view=unknown", "?view=operator&action=arm", "?view=operator%26action=arm"):
            with pytest.raises(HTTPError) as error:
                urlopen(server.url + "/api/status" + query, timeout=2)
            assert error.value.code == 404
        assert server.status_response_diagnostics()["started_count"] == 0
    finally:
        server.close()


def _onset_trial(*, direction=1, config=None, probe_config=None, ramp_ticks=50, support_limit=.05):
    controller = B601TorqueExperiment(
        B601TorqueExperimentLimits("joint3", max_abs_assist_nm=.8, max_excursion_rad=.1,
            max_joint_velocity_rad_s=support_limit),
        quantized_velocity_debug=True, onset_capture=config or B601MotionOnsetCaptureConfig(),
        onset_probe=probe_config,
    )
    if probe_config is not None:
        controller.set_human_intervention(False)
    controller.owner_ready()
    for i in range(20):
        _velocity_debug_tick(controller, 1+i*.01)
    controller.submit("arm", now_sec=1.195)
    _, frame = _velocity_debug_tick(controller, 1.2)
    assert frame["phase"] == "armed"
    controller.submit("set", now_sec=1.201, torque_nm=direction*.8)
    for i in range(1, ramp_ticks+1):
        _velocity_debug_tick(controller, 1.2+i*.01)
    return controller, 1.2+ramp_ticks*.01


def _probe_candidate(controller, now, *, direction=1):
    count = direction*.0003834962844848633
    previous = controller.status()["last_dispatch"]["overlay_after_profile_nm"]
    overlay, frame = _velocity_debug_tick(controller, now+.01, q=count, raw=count/.01)
    assert frame["phase"] == "armed" and overlay[2] == pytest.approx(previous)
    assert frame["onset_capture"]["probe"]["state"]["stage"] == "hold"
    return now+.01, count, previous


@pytest.mark.parametrize("direction", [-1, 1])
def test_probe_holds_last_accepted_overlay_then_quietly_resumes_original_slew(direction):
    controller, now = _onset_trial(direction=direction, probe_config=B601OnsetProbeConfig())
    now, q, held = _probe_candidate(controller, now, direction=direction)
    for i in range(1, 13):
        overlay, frame = _velocity_debug_tick(controller, now+i*.01, q=q)
        assert overlay[2] == pytest.approx(held)
        assert frame["phase"] == "armed" and frame["onset_capture"]["confirmation_streak"] == 0
    overlay, frame = _velocity_debug_tick(controller, now+.13, q=q)
    assert overlay[2] == pytest.approx(held+direction*.0015)
    assert frame["onset_capture"]["probe"]["reason"] == "quiet_resume_from_held_overlay"
    # The same plateau cannot repeatedly create new observation windows.
    for i in range(14, 20):
        _, frame = _velocity_debug_tick(controller, now+i*.01, q=q)
    assert frame["onset_capture"]["trial"]["probe_count"] == 1


@pytest.mark.parametrize("direction", [-1, 1])
def test_probe_staircase_progress_withdraws_without_claiming_sustained_motion(direction):
    controller, now = _onset_trial(direction=direction, probe_config=B601OnsetProbeConfig())
    count = direction*.0003834962844848633
    for level in range(1, 4):
        _, frame = _velocity_debug_tick(controller, now+.01, q=level*count, raw=count/.01)
        now += .01
        assert frame["onset_capture"]["probe"]["state"]["stage"] == "hold"
        for i in range(1, 14):
            _, frame = _velocity_debug_tick(controller, now+i*.01, q=level*count)
            assert frame["phase"] == "armed" and frame["onset_capture"]["confirmation_streak"] == 0
        now += .13
    assert frame["onset_capture"]["trial"]["outcome"] == "observing"
    assert frame["onset_capture"]["trial"]["probe_count"] == 3
    previous_dispatch = controller.status()["last_dispatch"]
    overlay, frame = _velocity_debug_tick(controller, now+.01, q=4*count, raw=count/.01)
    assert overlay == ZERO and frame["phase"] == "onset_release"
    trial = frame["onset_capture"]["trial"]
    assert trial["outcome"] == "probe_cumulative_displacement_withdrawn"
    assert trial["cumulative_displacement_at_withdrawal_rad"] == pytest.approx(4*count)
    assert trial["cumulative_withdrawal_dispatch"] == previous_dispatch
    assert "onset_confirmed_at_sec" not in trial
    assert trial["probe_count"] == 3  # No new hold or rise after the threshold.
    for i in range(2, 18):
        overlay, frame = _velocity_debug_tick(controller, now+i*.01, q=4*count)
        assert overlay == ZERO
    assert frame["phase"] == "inactive"
    # Post-trial repositioning is current feedback, not this trial's result.
    _, frame = _velocity_debug_tick(controller,now+.19,q=6*count,raw=2*count/.02)
    assert frame["qpos_rad"][2] == pytest.approx(6*count)
    assert frame["onset_capture"]["probe"]["trial_displacement_rad"] == pytest.approx(4*count)
    with pytest.raises(ValueError,match="armed"):
        controller.submit("set",now_sec=now+.20,torque_nm=direction*.8)


@pytest.mark.parametrize('failure', ['raw_speed', 'support_drift'])
def test_probe_cumulative_withdrawal_never_masks_a_simultaneous_hard_fault(failure):
    controller, now = _onset_trial(probe_config=B601OnsetProbeConfig())
    count = .0003834962844848633
    for level in range(1,4):
        for i in range(1,16):
            _velocity_debug_tick(controller,now+i*.01,q=level*count,raw=count/.01 if i==1 else 0.)
        now += .15
    overlay, frame = _velocity_debug_tick(controller,now+.01,q=4*count,
        raw=.2 if failure=='raw_speed' else count/.01,
        support=.009 if failure=='support_drift' else 0.)
    assert overlay == ZERO and frame['phase']=='fault'
    assert frame['onset_capture']['trial']['outcome']=='guard_fault'
    assert frame['fault'].startswith('joint_velocity_exceeded' if failure=='raw_speed' else 'support_joint_drift_exceeded')


def test_probe_single_count_back_and_forth_noise_does_not_accumulate_fake_net_progress():
    controller, now = _onset_trial(probe_config=B601OnsetProbeConfig())
    count = .0003834962844848633
    previous = 0.
    for i in range(1, 65):
        q = count if i%2 else 0.
        _, frame = _velocity_debug_tick(controller,now+i*.01,q=q,raw=(q-previous)/.01)
        previous = q
        assert frame["phase"] == "armed"
        assert abs(frame["onset_capture"]["probe"]["trial_displacement_rad"]) <= count
    assert frame["onset_capture"]["trial"]["outcome"] == "observing"


def test_probe_opposing_staircase_motion_also_stops_escalation():
    controller, now = _onset_trial(probe_config=B601OnsetProbeConfig())
    count = -.0003834962844848633
    for level in range(1, 5):
        for i in range(1, 16):
            overlay, frame = _velocity_debug_tick(controller,now+i*.01,q=level*count,
                raw=count/.01 if i==1 else 0.)
            if frame['phase']!='armed': break
        now += i*.01
    assert overlay == ZERO and frame['phase']=='onset_release'
    assert frame['onset_capture']['trial']['outcome']=='probe_cumulative_displacement_withdrawn'


@pytest.mark.parametrize("direction", [-1, 1])
def test_probe_requires_distinct_post_dispatch_progress_and_withdraws_on_confirmation(direction):
    controller, now = _onset_trial(direction=direction, probe_config=B601OnsetProbeConfig())
    now, q, held = _probe_candidate(controller, now, direction=direction)
    count = direction*.0003834962844848633
    for i in range(1, 4):
        overlay, frame = _velocity_debug_tick(controller, now+i*.01, q=q+i*count, raw=count/.01)
        if i < 3:
            assert frame["phase"] == "armed" and overlay[2] == pytest.approx(held)
            assert frame["onset_capture"]["confirmation_streak"] == i
    assert overlay == ZERO and frame["phase"] == "onset_release"
    assert frame["onset_capture"]["trial"]["onset_confirmation_method"] == "post_hold_distinct_progress"
    assert frame["onset_capture"]["trial"]["onset_dispatch"]["overlay_after_profile_nm"] == pytest.approx(held)


@pytest.mark.parametrize("direction", [-1, 1])
def test_probe_new_count_then_recent_plateau_resumes_without_repeating_candidate(direction):
    controller, now = _onset_trial(direction=direction, probe_config=B601OnsetProbeConfig())
    now, q, held = _probe_candidate(controller, now, direction=direction)
    for i in range(1, 21):
        overlay, frame = _velocity_debug_tick(controller, now+i*.01, q=2*q, raw=q/.01 if i==1 else 0.)
        assert frame['phase'] == 'armed'
    assert abs(overlay[2]) > abs(held)
    assert frame['onset_capture']['probe']['reason'] == 'quiet_resume_from_held_overlay'
    assert frame['onset_capture']['trial']['probe_count'] == 1
    assert frame['onset_capture']['trial']['last_probe']['distinct_progress_samples'] == 1
    assert 'onset_confirmed_at_sec' not in frame['onset_capture']['trial']


@pytest.mark.parametrize("direction", [-1, 1])
def test_probe_late_microstep_waits_a_full_new_plateau_window(direction):
    controller, now = _onset_trial(direction=direction, probe_config=B601OnsetProbeConfig(max_displacement_rad=.003))
    now, q, held = _probe_candidate(controller, now, direction=direction)
    for i in range(1, 19):
        position = q if i < 10 else 2*q
        overlay, frame = _velocity_debug_tick(controller, now+i*.01, q=position, raw=q/.01 if i==10 else 0.)
        if i <= 18:
            assert overlay[2] == pytest.approx(held)
    assert frame['onset_capture']['probe']['reason'] == 'waiting_recent_plateau'
    for i in (19, 20):
        overlay, frame = _velocity_debug_tick(controller, now+i*.01, q=2*q)
    assert frame['phase'] == 'armed' and abs(overlay[2]) > abs(held)
    assert frame['onset_capture']['trial']['probe_count'] == 1


def test_probe_recent_plateau_uses_current_window_not_entire_hold_range():
    controller, now = _onset_trial(probe_config=B601OnsetProbeConfig(max_displacement_rad=.003))
    now, q, _ = _probe_candidate(controller, now)
    for i in range(1, 20):
        position = 2*q if i < 5 else 3*q
        _, frame = _velocity_debug_tick(controller, now+i*.01, q=position, raw=q/.01 if i in (1, 5) else 0.)
    assert frame['phase'] == 'armed'
    assert frame['onset_capture']['trial']['last_probe']['maximum_progress_rad'] == pytest.approx(2*q)
    assert frame['onset_capture']['trial']['last_probe']['distinct_progress_samples'] == 2
    assert frame['onset_capture']['trial']['probe_count'] == 1
    assert frame['onset_capture']['probe']['reason'] == 'quiet_resume_from_held_overlay'


def test_probe_unresolved_late_progress_withdraws_at_bounded_hold_deadline():
    controller, now = _onset_trial(probe_config=B601OnsetProbeConfig(max_displacement_rad=.003))
    now, q, held = _probe_candidate(controller, now)
    for i in range(1, 23):
        position = q if i < 10 else 2*q if i < 17 else 3*q
        overlay, frame = _velocity_debug_tick(controller, now+i*.01, q=position, raw=q/.01 if i in (10, 17) else 0.)
        if frame['phase'] == 'onset_release':
            break
        assert overlay[2] == pytest.approx(held)
    assert frame['phase'] == 'onset_release' and overlay == ZERO
    assert frame['onset_capture']['trial']['outcome'] == 'probe_inconclusive'


def test_explicit_probe_observation_margin_resumes_late_quiet_plateau_without_counting_it_as_onset():
    config = B601OnsetProbeConfig(max_displacement_rad=.005, observation_timeout_sec=.6)
    controller, now = _onset_trial(probe_config=config)
    now, q, held = _probe_candidate(controller, now)
    trial_start = controller.status()['onset_capture']['trial']['started_at_sec']
    for i in range(1, 31):
        position = q if i < 10 else 2*q if i < 17 else 3*q
        overlay, frame = _velocity_debug_tick(controller, now+i*.01, q=position,
                                              raw=q/.01 if i in (10, 17) else 0.)
        assert frame['phase'] == 'armed'
        if frame['onset_capture']['probe']['reason'] == 'quiet_resume_from_held_overlay':
            break
        assert overlay[2] == pytest.approx(held)
    assert frame['onset_capture']['probe']['reason'] == 'quiet_resume_from_held_overlay'
    assert 'onset_confirmed_at_sec' not in frame['onset_capture']['trial']
    assert frame['onset_capture']['trial']['started_at_sec'] == trial_start
    assert frame['onset_capture']['probe']['maximum_hold_sec'] == .6


@pytest.mark.parametrize('timeout', [True, 0, -.1, float('nan'), float('inf'), .2, 10.])
def test_probe_observation_margin_validates_complete_window_and_original_trial(timeout):
    with pytest.raises((TypeError, ValueError)):
        B601OnsetProbeConfig(observation_timeout_sec=timeout)


def test_probe_default_observation_deadline_remains_exact():
    assert B601OnsetProbeConfig().observation_limit_sec == .2


def test_explicit_probe_observation_deadline_normalizes_finite_input_and_cannot_renew():
    config = B601OnsetProbeConfig(observation_timeout_sec="0.6")
    assert config.observation_limit_sec == .6
    controller, now = _onset_trial(probe_config=config)
    now, q, _ = _probe_candidate(controller, now)
    completed = controller.status()['onset_capture']['probe']['state']['first_hold_completed_at_sec']
    previous = q
    for i in range(1, 60):
        position = (2,3,2,1)[(i-1)%4]*q
        _, frame = _velocity_debug_tick(controller, now+i*.01, q=position,
                                        raw=(position-previous)/.01)
        assert frame['phase'] == 'armed'
        previous = position
    _, frame = _velocity_debug_tick(controller, completed+.6, q=previous)
    assert frame['phase'] == 'onset_release'
    assert frame['onset_capture']['trial']['outcome'] == 'probe_inconclusive'
    assert not any(e['action'] == 'probe_resume' for e in controller.status()['events'])


@pytest.mark.parametrize('exact_deadline', [False, True])
def test_probe_expired_hold_cannot_resume_on_late_quiet_feedback(exact_deadline):
    # 0.17 + 0.08 produces the exactly representable 0.25-second bound.
    controller, now = _onset_trial(probe_config=B601OnsetProbeConfig(hold_sec=.17, max_displacement_rad=.003))
    now, q, held = _probe_candidate(controller, now)
    completed = controller.status()['onset_capture']['probe']['state']['first_hold_completed_at_sec']
    for i in range(1, 25):
        position = q if i < 17 else 2*q
        overlay, frame = _velocity_debug_tick(controller, now+i*.01, q=position, raw=q/.01 if i==17 else 0.)
        assert overlay[2] == pytest.approx(held)
    end = completed + .25 if exact_deadline else now + .27
    overlay, frame = _velocity_debug_tick(controller, end, q=2*q)
    assert overlay == ZERO and frame['phase'] == 'onset_release'
    assert frame['onset_capture']['trial']['outcome'] == 'probe_inconclusive'
    assert not any(e['action'] == 'probe_resume' for e in controller.status()['events'])


@pytest.mark.parametrize("direction", [-1, 1])
def test_probe_explicit_observation_bound_preserves_trial_anchor_across_plateaus(direction):
    controller, now = _onset_trial(direction=direction, probe_config=B601OnsetProbeConfig(max_displacement_rad=.003))
    count = direction*.0003834962844848633
    for level in range(1, 9):
        for i in range(1, 17):
            overlay, frame = _velocity_debug_tick(controller, now+i*.01, q=level*count, raw=count/.01 if i==1 else 0.)
            if frame['phase'] != 'armed':
                break
        now += i*.01
        if level < 8:
            assert frame['phase'] == 'armed'
            assert frame['onset_capture']['trial']['baseline_qpos_rad'][2] == 0
    assert overlay == ZERO and frame['phase'] == 'onset_release'
    assert frame['onset_capture']['trial']['outcome'] == 'probe_cumulative_displacement_withdrawn'
    assert frame['onset_capture']['probe']['cumulative_displacement_limit_rad'] == .003
    assert frame['onset_capture']['trial']['cumulative_displacement_at_withdrawal_rad'] == pytest.approx(8*count)
    assert 'onset_confirmed_at_sec' not in frame['onset_capture']['trial']


@pytest.mark.parametrize("action", ["zero", "withdraw", "set_zero", "human"])
def test_probe_cancellation_clears_hold_and_never_resumes_old_goal(action, monkeypatch):
    controller, now = _onset_trial(probe_config=B601OnsetProbeConfig())
    now, q, _ = _probe_candidate(controller, now)
    if action == "human":
        monkeypatch.setattr("robot_motion.rebot_b601_rs.torque_experiment.time.monotonic", lambda:now+.001)
        controller.set_human_intervention(True)
        controller.set_human_intervention(False)  # Does not restore old authority.
    else:
        controller.submit("set" if action == "set_zero" else action, now_sec=now+.001,
            **({"torque_nm":0.} if action == "set_zero" else {}))
    for i in range(1, 12):
        overlay, frame = _velocity_debug_tick(controller, now+i*.01, q=q)
        assert overlay == ZERO
    assert frame["phase"] == "inactive"
    with pytest.raises(ValueError, match="armed"):
        controller.submit("set", now_sec=now+.115, torque_nm=.8)


@pytest.mark.parametrize("condition", ["hard_speed", "feedback_gap", "support_drift", "no_hold_receipt"])
def test_probe_preserves_hard_guards_and_requires_confirmed_held_command(condition):
    controller, now = _onset_trial(probe_config=B601OnsetProbeConfig())
    if condition == "no_hold_receipt":
        q = .0003834962844848633
        _velocity_debug_tick(controller, now+.01, q=q, raw=q/.01, ack=False)
        for i in range(2, 7):
            overlay, frame = _velocity_debug_tick(controller, now+i*.01, q=q, ack=False)
            if frame["phase"] == "fault": break
        assert frame["phase"] == "fault" and frame["fault"] in {"probe_dispatch_not_fresh", "probe_hold_dispatch_timeout"}
        return
    now, q, _ = _probe_candidate(controller, now)
    overlay, frame = _velocity_debug_tick(controller, now+(.041 if condition == "feedback_gap" else .01),
        q=q+.002 if condition == "hard_speed" else q, raw=.2 if condition == "hard_speed" else 0.,
        support=.009 if condition == "support_drift" else 0.)
    assert overlay == ZERO and frame["phase"] == "fault"


def test_probe_above_hard_single_count_still_withdraws_without_nonzero_observation():
    controller, now = _onset_trial(probe_config=B601OnsetProbeConfig())
    _, (overlay, frame) = _velocity_debug_spike(controller, now)
    assert overlay == ZERO and frame["phase"] == "velocity_check"
    assert frame["onset_capture"]["probe"]["state"] is None


@pytest.mark.parametrize("budget", ["time", "exposure"])
def test_probe_trial_budgets_withdraw_before_caps_and_zero_confirmation_is_still_required(budget):
    config = B601OnsetProbeConfig(max_trial_sec=.3) if budget == "time" else B601OnsetProbeConfig(max_command_exposure_nm_sec=.08)
    controller, now = _onset_trial(probe_config=config, ramp_ticks=5)
    for i in range(1, 100):
        overlay, frame = _velocity_debug_tick(controller, now+i*.01)
        if frame["phase"] == "onset_release": break
    assert frame["phase"] == "onset_release" and overlay == ZERO
    assert frame["onset_capture"]["trial"]["outcome"] == "probe_trial_budget_exhausted"
    assert controller.status()["onset_capture"]["probe"]["command_exposure_nm_sec"] <= config.max_command_exposure_nm_sec


def test_probe_requires_explicit_human_absence_and_does_not_grant_it_by_default():
    controller = B601TorqueExperiment(B601TorqueExperimentLimits("joint3"), quantized_velocity_debug=True,
        onset_capture=B601MotionOnsetCaptureConfig(), onset_probe=B601OnsetProbeConfig())
    controller.owner_ready()
    with pytest.raises(ValueError, match="no-human"):
        controller.submit("arm", now_sec=1.)
    controller.set_human_intervention(False)
    controller.submit("arm", now_sec=1.1)
    controller.set_human_intervention(True)
    controller.set_human_intervention(False)
    overlay, frame = _velocity_debug_tick(controller, 1.2)
    assert overlay == ZERO and frame["phase"] == "inactive"
    with pytest.raises(ValueError, match="armed"):
        controller.submit("set", now_sec=1.21, torque_nm=.2)


def test_probe_limits_actual_conditioned_tail_and_nonmonotonic_receipts():
    controller, now = _onset_trial(probe_config=B601OnsetProbeConfig())
    now, q, _ = _probe_candidate(controller, now)
    overlay, _ = _velocity_debug_tick(controller, now+.01, q=q, ack=False)
    _velocity_debug_ack(controller, now+.01, overlay, tail=(0.,0.,.01,0.,0.,0.))
    assert controller.status()["phase"] == "onset_release"
    assert controller.status()["onset_capture"]["trial"]["outcome"] == "probe_hold_command_changed"
    assert controller.status()["desired_assist_nm"] == 0.
    # A valid feedback cycle cannot retroactively shorten the command exposure.
    controller, now = _onset_trial(probe_config=B601OnsetProbeConfig())
    overlay, _ = _velocity_debug_tick(controller, now+.01, ack=False)
    _velocity_debug_ack(controller, now+.01, overlay, delay=.025)
    overlay, _ = _velocity_debug_tick(controller, now+.02, ack=False)
    with pytest.raises(ValueError, match="timestamps must advance"):
        _velocity_debug_ack(controller, now+.02, overlay)
    assert controller.status()["fault"] == "probe_nonmonotonic_dispatch"


def test_probe_rejects_out_of_envelope_final_composition_before_dispatch():
    controller, _ = _onset_trial(probe_config=B601OnsetProbeConfig())
    accepted = controller.status()["last_dispatch"]
    controller.validate_final_command(gravity_nm=ZERO, total_nm=(0.,0.,.8,0.,0.,0.))
    with pytest.raises(ValueError, match="before dispatch"):
        controller.validate_final_command(gravity_nm=ZERO, total_nm=(0.,0.,.801,0.,0.,0.))
    assert controller.status()["phase"] == "fault"
    assert controller.status()["desired_assist_nm"] == 0.
    assert controller.status()["last_dispatch"] == accepted


def test_probe_final_exposure_uses_previously_completed_command_until_next_receipt():
    controller, now = _onset_trial(probe_config=B601OnsetProbeConfig())
    before = controller.status()
    last = before["last_dispatch"]
    prior_exposure = before["onset_capture"]["probe"]["command_exposure_nm_sec"]
    overlay, _ = _velocity_debug_tick(controller, now+.01, ack=False)
    _velocity_debug_ack(controller, now+.01, overlay, delay=.005)
    after = controller.status()["onset_capture"]["probe"]["command_exposure_nm_sec"]
    assert after-prior_exposure == pytest.approx(abs(last["conditioned_total_nm"]-last["gravity_nm"])*(now+.015-last["completed_at_sec"]))


def test_probe_bounded_support_count_jitter_does_not_prevent_quiet_resume():
    controller, now = _onset_trial(probe_config=B601OnsetProbeConfig())
    now, q, held = _probe_candidate(controller, now)
    for i in range(1, 14):
        support = q if i % 8 >= 4 else 0.
        overlay, frame = _capture_six_tick(controller, now+i*.01, (support,0.,q,0.,0.,0.))
        assert frame["phase"] == "armed"
    assert frame["onset_capture"]["probe"]["reason"] == "quiet_resume_from_held_overlay"
    assert overlay[2] == pytest.approx(held+.0015)


def test_probe_buffered_feedback_cannot_seed_post_hold_observation_or_reset_first_receipt():
    controller, now = _onset_trial(probe_config=B601OnsetProbeConfig())
    q = .0003834962844848633
    now += .01
    overlay, _ = _velocity_debug_tick(controller, now, q=q, raw=q/.01, ack=False)
    _velocity_debug_ack(controller, now, overlay, delay=.015)
    overlay, frame = _velocity_debug_tick(controller, now+.01, q=q, ack=False)
    assert frame["phase"] == "armed"
    state = frame["onset_capture"]["probe"]["state"]
    assert state["first_hold_completed_at_sec"] == now+.015 and state["first_feedback_at_sec"] is None
    _velocity_debug_ack(controller, now+.01, overlay, delay=.02)
    _, frame = _velocity_debug_tick(controller, now+.04, q=q)
    assert frame["onset_capture"]["probe"]["state"]["first_feedback_at_sec"] == now+.04
    assert frame["onset_capture"]["probe"]["state"]["first_hold_completed_at_sec"] == now+.015


@pytest.mark.parametrize('hold_completion_delay', [.01, .015])
@pytest.mark.parametrize('displacement_rad', [.0012, .0018])
def test_probe_cached_prehold_position_cannot_be_recounted_as_new_posthold_progress(hold_completion_delay, displacement_rad):
    # Isolate the causal progress gate from the default cumulative withdrawal.
    controller, now = _onset_trial(config=B601MotionOnsetCaptureConfig(displacement_rad=displacement_rad), probe_config=B601OnsetProbeConfig())
    count = .0003834962844848633
    overlay, _ = _velocity_debug_tick(controller, now+.01, q=count, raw=count/.01, ack=False)
    _velocity_debug_ack(controller, now+.01, overlay, delay=hold_completion_delay)
    overlay, frame = _velocity_debug_tick(controller, now+.02, q=2*count, raw=count/.01, ack=False)
    assert frame['onset_capture']['probe']['state']['distinct_progress_samples'] == 0
    _velocity_debug_ack(controller, now+.02, overlay, delay=.02)
    for tick, level in ((.05,2),(.06,3),(.07,4)):
        _, frame = _velocity_debug_tick(controller, now+tick, q=level*count,
            raw=0. if level==2 else count/.01)
        if level == 4 and displacement_rad == .0012:
            assert frame['phase'] == 'onset_release'
            assert frame['onset_capture']['trial']['outcome'] == 'probe_cumulative_displacement_withdrawn'
        else:
            assert frame['phase'] == 'armed'
        assert frame['onset_capture']['probe']['state']['distinct_progress_samples'] == level-2
    if displacement_rad == .0012:
        assert 'onset_confirmed_at_sec' not in frame['onset_capture']['trial']
        return
    overlay, frame = _velocity_debug_tick(controller, now+.08, q=5*count,raw=count/.01)
    assert overlay == ZERO and frame['phase'] == 'onset_release'
    assert frame['onset_capture']['trial']['onset_confirmation_method'] == 'post_hold_distinct_progress'


def test_probe_control_context_withdraws_before_annotation_failure_and_false_cannot_restore(tmp_path, monkeypatch):
    controller, _ = _onset_trial(probe_config=B601OnsetProbeConfig())
    server = B601TorqueConsoleServer(controller, assessment_path=tmp_path/'contexts.jsonl', automatic_recording=True,
        human_intervention_handler=controller.set_human_intervention)
    server.start()
    try:
        def broken_record(*_args):
            raise OSError("annotation write unavailable")
        monkeypatch.setattr(server._server.assessment, "record", broken_record)
        def declare(human):
            return urlopen(Request(server.url+'/api/assessment', data=json.dumps({'action':'context','human_intervention':human}).encode(),
                headers={'Content-Type':'application/json','Origin':server.url,'X-B601-Console-Token':server._server.token}),timeout=2)
        with pytest.raises(HTTPError) as error:
            declare(True)
        assert error.value.code == 500
        assert controller.status()["phase"] == "onset_release" and controller.status()["desired_assist_nm"] == 0.
        with pytest.raises(HTTPError):
            declare(False)
        assert controller.status()["onset_capture"]["probe"]["human_intervention"] is True
    finally:
        server.close()


def test_probe_delayed_old_no_human_write_cannot_override_new_withdrawal(tmp_path, monkeypatch):
    controller = B601TorqueExperiment(B601TorqueExperimentLimits("joint3"), quantized_velocity_debug=True,
        onset_capture=B601MotionOnsetCaptureConfig(), onset_probe=B601OnsetProbeConfig())
    controller.owner_ready()
    controller.set_human_intervention(False)
    controller.submit("arm", now_sec=time.monotonic())
    false_write_started, true_applied, release_write = (threading.Event() for _ in range(3))
    replies, failures = [], []
    def handler(human):
        controller.set_human_intervention(human)
        if human: true_applied.set()
    server = B601TorqueConsoleServer(controller, assessment_path=tmp_path/'contexts.jsonl',
        automatic_recording=True, human_intervention_handler=handler)
    journal = server._server.assessment
    original_append = journal._append
    def delayed_append(event):
        if event.get('event') == 'declaration' and event.get('human_intervention') is False:
            false_write_started.set()
            assert release_write.wait(3)
        return original_append(event)
    monkeypatch.setattr(journal, '_append', delayed_append)
    def declare(human):
        try:
            request = Request(server.url+'/api/assessment',
                data=json.dumps({'action':'context','human_intervention':human}).encode(),
                headers={'Content-Type':'application/json','Origin':server.url,'X-B601-Console-Token':server._server.token})
            with urlopen(request,timeout=5) as response: replies.append(response.status)
        except OSError as exc:
            failures.append(exc)
    server.start()
    threads = [threading.Thread(target=declare,args=(human,)) for human in (False,True)]
    try:
        threads[0].start()
        assert false_write_started.wait(3)
        threads[1].start()
        assert true_applied.wait(3)
        assert controller.status()['phase'] == 'inactive'  # No write completion is needed to withdraw.
        release_write.set()
        for thread in threads:
            thread.join(5)
            assert not thread.is_alive()
        assert not failures and sorted(replies) == [200,200]
        assert controller.status()['onset_capture']['probe']['human_intervention'] is True
        with pytest.raises(ValueError,match='no-human'):
            controller.submit('arm',now_sec=time.monotonic())
        declarations = [json.loads(line)['human_intervention'] for line in (tmp_path/'contexts.jsonl').read_text().splitlines()
            if json.loads(line).get('event') == 'declaration']
        assert declarations == [False,True]
    finally:
        release_write.set()
        for thread in threads:
            if thread.ident is not None: thread.join(5)
        server.close()


def test_probe_cli_and_gui_consume_explicit_settings_and_owner_observation(monkeypatch, tmp_path):
    args = owner._parser().parse_args(['--allow-motion','--torque-console','--rate-hz','100',
        '--torque-console-confirmation','B601_RS_TORQUE_CONSOLE','--torque-console-quantized-velocity-debug',
        '--torque-console-onset-capture','--torque-console-onset-probe','--torque-console-onset-probe-hold-sec','.15',
        '--torque-console-onset-probe-max-displacement-rad','.003'])
    parsed = probe.build_parser().parse_args(owner.build_probe_command(args,run_dir=tmp_path)[2:])
    probe._validate_torque_console_cli(parsed)
    assert probe._direct_torque_console_from_args(parsed).onset_probe.hold_sec == .15
    assert probe._direct_torque_console_from_args(parsed).onset_probe.max_displacement_rad == .003
    parsed.torque_console_onset_capture = False
    with pytest.raises(probe.LowGainProbeError, match="capture"):
        probe._validate_torque_console_cli(parsed)
    node = shutil.which('node')
    if node is None: pytest.skip('Node is required for GUI contracts')
    controller, now = _onset_trial(probe_config=B601OnsetProbeConfig())
    now, _, _ = _probe_candidate(controller, now)
    monkeypatch.setattr('robot_motion.rebot_b601_rs.torque_experiment.time.monotonic',lambda:now+.005)
    state = controller.status()
    source = (Path(__file__).parents[1]/'control_station/b601_torque_console/static/app.js').read_text()
    functions = source[source.index('function number('):source.index('function post(')]
    script = "const vm=require('node:vm');const c={};vm.runInNewContext(JSON.parse(process.argv[1]),c); console.log(c.gateMessage(JSON.parse(require('node:fs').readFileSync(0,'utf8'))));"
    message = subprocess.run([node,'-e',script,json.dumps(functions)],input=json.dumps(state),text=True,capture_output=True,check=True).stdout
    assert '定力观察' in message and '不向目标继续爬升' in message and '不同新位置' in message
    assert '整轮净位移' in message


@pytest.mark.parametrize("options", [{"hold_sec":.04}, {"hold_sec":float("nan")}, {"max_trial_sec":.2}, {"max_command_exposure_nm_sec":0.},
                                     {"max_displacement_rad":float("nan")}, {"max_displacement_rad":.0001}])
def test_probe_config_rejects_unbounded_or_infeasible_observation(options):
    with pytest.raises(ValueError): B601OnsetProbeConfig(**options)


@pytest.mark.parametrize('bound', [.00116, .10, .11])
def test_probe_observation_bound_must_cover_onset_and_stay_inside_hard_excursion(bound):
    with pytest.raises(ValueError, match='probe displacement bound'):
        B601TorqueExperiment(B601TorqueExperimentLimits('joint3', max_excursion_rad=.10),
            quantized_velocity_debug=True, onset_capture=B601MotionOnsetCaptureConfig(),
            onset_probe=B601OnsetProbeConfig(max_displacement_rad=bound))


def test_probe_observation_bound_requires_explicit_probe_cli():
    args = probe.build_parser().parse_args(['--config', 'unused.yaml', '--rebot-source-root', '.',
                                          '--torque-console-onset-probe-max-displacement-rad', '.003'])
    with pytest.raises(probe.LowGainProbeError, match='explicit onset probe'):
        probe._validate_torque_console_cli(args)


def _confirm_onset(controller, now, *, direction=1, ack=True, tail=ZERO):
    previous = 0.
    for i in range(1, 16):
        q = direction*(i//2)*.0003834962844848633
        overlay, frame = _velocity_debug_tick(controller, now+i*.01, q=q, raw=(q-previous)/.01, ack=False)
        if frame["phase"] == "armed" or ack:
            _velocity_debug_ack(controller, now+i*.01, overlay, tail=tail if frame["phase"] == "onset_release" else ZERO)
        previous = q
        if frame["phase"] == "onset_release":
            assert overlay == ZERO and frame["desired_assist_nm"] == 0.
            return now+i*.01, q, frame
    raise AssertionError("sustained multi-count motion was not detected")


@pytest.mark.parametrize("direction", [-1, 1])
def test_onset_capture_withdraws_and_returns_to_manual_zero_assist_without_owner_fault(direction):
    controller, now = _onset_trial(direction=direction)
    now, q, frame = _confirm_onset(controller, now, direction=direction)
    trial = frame["onset_capture"]["trial"]
    assert trial["outcome"] == "motion_onset_confirmed"
    assert trial["direction"] == direction
    assert trial["onset_displacement_rad"] >= .0012
    assert trial["onset_dispatch"]["overlay_after_profile_nm"] * direction > 0
    for action in ("set", "arm", "heartbeat", "select_joint"):
        with pytest.raises(ValueError):
            controller.submit(action, now_sec=now+.001, torque_nm=.8, joint_name="joint3")
    for i in range(1, 12):
        overlay, frame = _velocity_debug_tick(controller, now+i*.01, q=q)
        assert overlay == ZERO and frame["phase"] != "fault"
    assert frame["phase"] == "inactive" and frame["owner_ready"] and frame["fault"] is None
    assert frame["onset_capture"]["completed_trials"] == 1
    assert len(controller.status()["onset_capture"]["trials"]) == 1
    with pytest.raises(ValueError, match="armed"):
        controller.submit("set", now_sec=now+.115, torque_nm=.8)
    controller.submit("arm", now_sec=now+.116)
    overlay, frame = _velocity_debug_tick(controller, now+.12, q=q)
    assert frame["phase"] == "armed" and overlay == ZERO
    controller.submit("set", now_sec=now+.121, torque_nm=.2)
    _, frame = _velocity_debug_tick(controller, now+.13, q=q)
    assert frame["onset_capture"]["trial"]["baseline_qpos_rad"][2] == q
    assert frame["onset_capture"]["completed_trials"] == 1


def test_onset_capture_ignores_count_jitter_opposite_motion_and_resets_interrupted_confirmation():
    controller, now = _onset_trial()
    previous = 0.
    for i in range(1, 41):
        q = (i%2)*.0003834962844848633
        _, frame = _velocity_debug_tick(controller, now+i*.01, q=q, raw=(q-previous)/.01)
        previous = q
        assert frame["phase"] == "armed" and frame["onset_capture"]["confirmation_streak"] == 0
    for i in range(1, 12):
        q = -(i//2)*.0003834962844848633
        _, frame = _velocity_debug_tick(controller, now+.4+i*.01, q=q, raw=(q-previous)/.01)
        previous = q
        assert frame["phase"] == "armed"
    assert frame["onset_capture"]["confirmation_streak"] == 0


@pytest.mark.parametrize("condition,reason", [
    ("selected_speed", "joint_velocity_exceeded"),
    ("support_speed", "onset_capture_raw_velocity_exceeded"),
    ("support_drift", "support_joint_drift_exceeded"),
    ("excursion", "joint_excursion_exceeded"),
    ("gap", "velocity_debug_feedback_gap_exceeded"),
])
def test_onset_capture_does_not_downgrade_actual_hard_faults(condition, reason):
    controller, now = _onset_trial()
    q = (0., 0., .002 if condition == "selected_speed" else .11 if condition == "excursion" else 0., .01 if condition == "support_drift" else 0., 0., 0.)
    raw = (0., 0., .2 if condition == "selected_speed" else 0., .06 if condition == "support_speed" else 0., 0., 0.)
    overlay, frame = controller.step(now_sec=now+(.041 if condition == "gap" else .01), qpos_rad=q, qvel_rad_s=raw)
    assert overlay == ZERO and frame["phase"] == "fault" and reason in frame["fault"]
    assert controller.status()["desired_assist_nm"] == 0.


@pytest.mark.parametrize("condition,reason", [
    ("missing_zero", "onset_capture_zero_dispatch_timeout"),
    ("tail", "onset_capture_zero_dispatch_timeout"),
    ("moving", "onset_capture_settle_timeout"),
    ("overspeed", "onset_capture_raw_velocity_exceeded"),
])
def test_onset_release_requires_actual_final_zero_rest_and_original_guards(condition, reason):
    controller, now = _onset_trial(config=B601MotionOnsetCaptureConfig(settle_timeout_sec=.2))
    now, q, _ = _confirm_onset(controller, now, ack=condition != "missing_zero", tail=(0.,0.,.02,0.,0.,0.) if condition == "tail" else ZERO)
    for i in range(1, 25):
        if condition == "moving":
            q += .00025
        overlay, frame = _velocity_debug_tick(controller, now+i*.01, q=q,
            raw=.06 if condition == "overspeed" else .025 if condition == "moving" else 0.,
            ack=condition != "missing_zero", tail=(0.,0.,.02,0.,0.,0.) if condition == "tail" else ZERO)
        assert overlay == ZERO
        if frame["phase"] == "fault":
            break
    assert frame["phase"] == "fault" and reason in frame["fault"]


def test_onset_release_zero_requests_do_not_restart_clock_or_restore_force():
    controller, now = _onset_trial()
    now, q, _ = _confirm_onset(controller, now)
    for action in ("zero", "withdraw"):
        controller.submit(action, now_sec=now+.001)
        assert controller.status()["onset_capture"]["release_started_at_sec"] == now
    for i in range(1, 10):
        overlay, frame = _velocity_debug_tick(controller, now+i*.01, q=q)
        assert overlay == ZERO
    assert frame["phase"] == "inactive"


def test_onset_release_rejects_nonzero_reappearance_after_final_zero():
    controller, now = _onset_trial()
    now, q, _ = _confirm_onset(controller, now)
    overlay, _ = _velocity_debug_tick(controller, now+.01, q=q, ack=False)
    with pytest.raises(ValueError, match="cannot reappear"):
        _velocity_debug_ack(controller, now+.01, overlay, tail=(0.,0.,.01,0.,0.,0.))
    assert controller.status()["phase"] == "fault"


def test_onset_capture_consecutive_confirmation_resets_and_baseline_does_not_follow_target_edits():
    controller, now = _onset_trial(config=B601MotionOnsetCaptureConfig(confirm_samples=8))
    previous = 0.
    for i in range(1, 14):
        q = (i//2)*.0003834962844848633
        _, frame = _velocity_debug_tick(controller, now+i*.01, q=q, raw=(q-previous)/.01)
        previous = q
    assert 0 < frame["onset_capture"]["confirmation_streak"] < 8
    baseline = frame["onset_capture"]["trial"]["baseline_qpos_rad"]
    controller.submit("set", now_sec=now+.131, torque_nm=.6)
    # Loss of a same-direction admitted dispatch interrupts the confirmation;
    # it cannot accumulate non-consecutive windows into a completed onset.
    _, frame = _velocity_debug_tick(controller, now+.14, q=q, ack=False)
    _velocity_debug_ack(controller, now+.14, ZERO)
    _, frame = _velocity_debug_tick(controller, now+.15, q=q)
    assert frame["phase"] == "armed" and frame["onset_capture"]["confirmation_streak"] == 0
    assert frame["onset_capture"]["trial"]["baseline_qpos_rad"] == baseline


def test_onset_capture_stale_gui_releases_without_dropping_original_support_anchor():
    controller, now = _onset_trial()
    for i in range(1, 39):
        overlay, frame = controller.step(now_sec=now+i*.01, qpos_rad=ZERO, qvel_rad_s=ZERO)
        _velocity_debug_ack(controller, now+i*.01, overlay)
        if frame["phase"] == "onset_release":
            break
    assert frame["phase"] == "onset_release" and overlay == ZERO
    assert frame["onset_capture"]["trial"]["outcome"] == "gui_intent_stale"
    assert frame["anchor_rad"] == 0. and frame["support_drift_rad"] is not None


def test_onset_capture_direction_change_withdraws_without_reanchoring():
    controller, now = _onset_trial()
    baseline = controller.status()["onset_capture"]["trial"]["baseline_qpos_rad"]
    controller.submit("set", now_sec=now+.001, torque_nm=-.2)
    overlay, frame = _velocity_debug_tick(controller, now+.01)
    assert overlay == ZERO and frame["phase"] == "onset_release"
    assert frame["onset_capture"]["trial"]["outcome"] == "direction_changed"
    assert frame["onset_capture"]["trial"]["baseline_qpos_rad"] == baseline


@pytest.mark.parametrize("during_check", [False, True])
def test_onset_capture_j3_quantization_exception_never_masks_support_raw_fault(during_check):
    controller, now = _onset_trial()
    count, dt = .0003834962844848633, .007518464
    if during_check:
        now, (_, frame) = _velocity_debug_spike(controller, now)
        assert frame["phase"] == "velocity_check"
        dt = .01
    q = (0.,0.,-count, .06*dt,0.,0.)
    raw = (0.,0.,0. if during_check else -count/dt,.06,0.,0.)
    overlay, frame = controller.step(now_sec=now+dt, qpos_rad=q, qvel_rad_s=raw)
    assert overlay == ZERO and frame["phase"] == "fault"
    assert "onset_capture_raw_velocity_exceeded: joint4" in frame["fault"]
    assert controller.status()["desired_assist_nm"] == 0.
    assert frame["onset_capture"]["trial"]["outcome"] == "guard_fault"


@pytest.mark.parametrize("value", [1, 0, None, "true"])
def test_release_support_confirmation_requires_boolean_explicit_capture(value):
    with pytest.raises(TypeError):
        B601MotionOnsetCaptureConfig(raw_joint_velocity_limit_rad_s=.15,release_support_confirmation=value)


def test_release_support_confirmation_cannot_be_enabled_for_an_ordinary_direct_console():
    with pytest.raises(ValueError,match="explicit acquisition"):
        B601MotionOnsetCaptureConfig(release_support_confirmation=True)
    with pytest.raises(ValueError,match="explicit onset probe"):
        B601TorqueExperiment(B601TorqueExperimentLimits("joint3"),quantized_velocity_debug=True,
            onset_capture=B601MotionOnsetCaptureConfig(raw_joint_velocity_limit_rad_s=.15,release_support_confirmation=True))


def _release_micro_trial(*, enabled=True, zero=True, support_limit=.07):
    controller, now = _onset_trial(support_limit=support_limit,
        probe_config=B601OnsetProbeConfig(max_displacement_rad=.01),
        config=B601MotionOnsetCaptureConfig(raw_joint_velocity_limit_rad_s=.15,
            release_support_confirmation=enabled))
    controller.set_human_intervention(False)
    now, q3, _ = _confirm_onset(controller, now, tail=ZERO if zero else (0.,0.,.001,0.,0.,0.))
    return controller, now, (0.,0.,q3,0.,0.,0.)


def test_acquisition_speed_headroom_does_not_expand_post_zero_support_observation():
    controller,now,q=_release_micro_trial(support_limit=.10)
    positions,raw=list(q),list(ZERO)
    positions[3]=2*.00038349442183971405
    raw[3]=.15
    overlay,frame=_capture_six_tick(controller,now+.01,tuple(positions),tuple(raw))
    assert overlay==ZERO and frame["phase"]=="fault"
    assert "onset_capture_raw_velocity_exceeded: joint4" in frame["fault"]
    assert frame["onset_capture"]["release_support_observation"] is None


@pytest.mark.parametrize("axis", [0, 1, 3, 4, 5])
@pytest.mark.parametrize("direction", [-1, 1])
def test_release_two_count_support_step_preserves_zero_and_confirmed_onset(axis, direction):
    controller, now, q = _release_micro_trial()
    positions, raw = list(q), list(ZERO)
    count=.00038349442183971405
    positions[axis]=direction*2*count
    raw[axis]=direction*2*count/.009972156  # Paired receive dt, not 10 ms snapshot.
    overlay, frame = _capture_six_tick(controller, now+.01, tuple(positions), tuple(raw))
    assert overlay==ZERO and frame["phase"]=="onset_release" and frame["fault"] is None
    observation=frame["onset_capture"]["release_support_observation"]
    assert observation["severity"]=="warning" and observation["state"]=="observing"
    started=observation["started_at_sec"]
    for n in range(2, 18):
        previous=positions[axis]
        positions[axis]=direction*count if n>=4 else direction*2*count
        raw=list(ZERO);raw[axis]=(positions[axis]-previous)/.01
        overlay, frame = _capture_six_tick(controller,now+n*.01,tuple(positions),tuple(raw))
        assert overlay==ZERO and frame["phase"]!="fault"
        if frame["phase"]=="inactive":break
    assert frame["phase"]=="inactive"
    trial=controller.status()["onset_capture"]["trials"][0]
    assert trial["outcome"]=="motion_onset_confirmed"
    assert trial["release_support_observation"]["state"]=="confirmed"
    assert trial["release_support_observation"]["started_at_sec"]==started
    with pytest.raises(ValueError,match="armed"):
        controller.submit("set",now_sec=now+.18,torque_nm=.2)


@pytest.mark.parametrize("case", ["default", "not_zero", "three_counts", "fast", "selected", "other_support", "sign", "human"])
def test_release_support_confirmation_does_not_mask_unqualified_raw_faults(case):
    controller, now, q = _release_micro_trial(enabled=case!="default",zero=case!="not_zero")
    if case=="human":controller.set_human_intervention(True)
    count=.00038349442183971405
    positions,raw=list(q),list(ZERO)
    positions[3]=2*count;raw[3]=2*count/.01
    if case=="three_counts":positions[3]=3*count;raw[3]=3*count/.01
    elif case=="fast":raw[3]=.15
    elif case=="selected":positions[2]+=.0016;raw[2]=.16
    elif case=="other_support":positions[4]=3*count;raw[4]=3*count/.01
    elif case=="sign":raw[3]=-raw[3]
    overlay, frame=_capture_six_tick(controller,now+.01,tuple(positions),tuple(raw))
    assert overlay==ZERO and frame["phase"]=="fault"
    assert frame["onset_capture"]["release_support_observation"] is None


@pytest.mark.parametrize("case,reason", [
    ("progress","onset_release_support_continued_motion"),
    ("timeout","onset_release_support_observation_timeout"),
    ("gap","velocity_debug_feedback_gap_exceeded"),
    ("missing_zero","onset_capture_zero_dispatch_timeout"),
])
def test_release_support_observation_is_bounded_without_deadline_renewal(case, reason):
    controller,now,q=_release_micro_trial()
    count=.00038349442183971405
    positions=list(q);positions[3]=2*count
    overlay,frame=_capture_six_tick(controller,now+.01,tuple(positions),(0.,0.,0.,2*count/.01,0.,0.))
    start=frame["onset_capture"]["release_support_observation"]["started_at_sec"]
    for n in range(2,25):
        previous=positions[3]
        if case=="progress":positions[3]=3*count
        elif case=="timeout":positions[3]=(n%2)*2*count
        raw=(0.,0.,0.,(positions[3]-previous)/.01,0.,0.)
        now_tick=now+n*.01 if case!="gap" else now+.061
        overlay,frame=_capture_six_tick(controller,now_tick,tuple(positions),raw,
            ack=case!="missing_zero",tail=(0.,0.,0.,.001,0.,0.) if case=="tail" else ZERO)
        assert overlay==ZERO
        if frame["phase"]=="fault":break
    assert frame["phase"]=="fault" and reason in frame["fault"]
    assert frame["onset_capture"]["release_support_observation"]["started_at_sec"]==start


def test_release_support_observation_cannot_accept_a_reappearing_final_command_tail():
    controller,now,q=_release_micro_trial()
    count=.00038349442183971405
    positions=list(q);positions[3]=2*count
    _capture_six_tick(controller,now+.01,tuple(positions),(0.,0.,0.,2*count/.01,0.,0.))
    with pytest.raises(ValueError,match="cannot reappear as nonzero"):
        _capture_six_tick(controller,now+.02,tuple(positions),tail=(0.,0.,0.,.001,0.,0.))
    assert controller.status()["phase"]=="fault"
    assert controller.status()["fault"]=="onset_capture_nonzero_after_zero"


def test_release_support_flag_never_relaxes_raw_guard_while_force_is_active():
    controller,now=_onset_trial(support_limit=.07,
        probe_config=B601OnsetProbeConfig(max_displacement_rad=.01),
        config=B601MotionOnsetCaptureConfig(raw_joint_velocity_limit_rad_s=.15,release_support_confirmation=True))
    count=.00038349442183971405
    overlay,frame=_capture_six_tick(controller,now+.01,(0.,0.,0.,2*count,0.,0.),(0.,0.,0.,2*count/.01,0.,0.))
    assert overlay==ZERO and frame["phase"]=="fault"
    assert frame["onset_capture"]["release_support_observation"] is None


def test_release_support_confirmed_episode_cannot_be_renewed_by_a_later_spike():
    controller,now,q=_release_micro_trial()
    count=.00038349442183971405
    positions=list(q);positions[3]=2*count
    _capture_six_tick(controller,now+.01,tuple(positions),(0.,0.,0.,2*count/.01,0.,0.))
    # Keep J3 moving below its original raw bound: support confirmation must
    # neither declare six-axis stillness nor complete the trial early.
    for n in range(2,12):
        positions[2]+=count
        overlay,frame=_capture_six_tick(controller,now+n*.01,tuple(positions),(0.,0.,count/.01,0.,0.,0.))
    assert frame["phase"]=="onset_release"
    assert frame["onset_capture"]["release_support_observation"]["state"]=="confirmed"
    positions[3]=0.
    overlay,frame=_capture_six_tick(controller,now+.12,tuple(positions),(0.,0.,0.,-2*count/.01,0.,0.))
    assert overlay==ZERO and frame["phase"]=="fault"
    assert frame["fault"]=="onset_release_support_repeated_raw_violation"


def _capture_six_tick(controller, now, q, raw=ZERO, *, ack=True, tail=ZERO):
    if controller.status()["phase"] == "armed":
        controller.submit("heartbeat", now_sec=now)
    overlay, frame = controller.step(now_sec=now, qpos_rad=q, qvel_rad_s=raw)
    if ack and frame["phase"] != "fault":
        _velocity_debug_ack(controller, now, overlay, tail=tail)
    return overlay, frame


@pytest.mark.parametrize("axis", [0, 1, 3, 4, 5])
@pytest.mark.parametrize("direction", [-1, 1])
def test_capture_qualified_support_count_withdraws_and_checks_all_six_axes(axis, direction):
    controller, now = _onset_trial()
    count, dt = .00038349442183971405, .006856124149635434
    q, raw = list(ZERO), list(ZERO)
    q[axis], raw[axis] = direction*count, direction*count/dt
    now += dt
    overlay, frame = _capture_six_tick(controller, now, tuple(q), tuple(raw))
    assert overlay == ZERO and frame["phase"] == "velocity_check" and frame["fault"] is None
    assert frame["quantized_velocity_debug"]["withdrawal_axes"] == (f"joint{axis+1}",)
    for i in range(1, 10):
        overlay, frame = _capture_six_tick(controller, now+i*.01, tuple(q))
        assert overlay == ZERO and frame["phase"] != "fault"
    assert frame["phase"] == "inactive" and frame["owner_ready"]
    assert frame["quantized_velocity_debug"]["confirmation_position_span_rad_six"] == ZERO
    trial = controller.status()["onset_capture"]["trials"][0]
    assert trial["outcome"] == "inconclusive_velocity_spike"
    assert "onset_dispatch" not in trial
    with pytest.raises(ValueError, match="armed"):
        controller.submit("set", now_sec=now+.095, torque_nm=.8)


def test_capture_simultaneous_j1_j5_trace_shape_repeated_spikes_never_restart_confirmation():
    controller, now = _onset_trial()
    count, dt = .00038349442183971405, .006856124149635434
    q = (-count, 0., 0., 0., count, 0.)
    raw = (-count/dt, 0., 0., 0., count/dt, 0.)
    now += dt
    overlay, frame = _capture_six_tick(controller, now, q, raw)
    assert overlay == ZERO and frame["phase"] == "velocity_check"
    assert frame["quantized_velocity_debug"]["withdrawal_axes"] == ("joint1", "joint5")
    controller.submit("zero", now_sec=now+.0015)
    controller.submit("withdraw", now_sec=now+.002)
    for i in range(1, 13):
        current = ZERO if i % 2 else q
        speed = tuple((-1 if i % 2 else 1)*v for v in raw)
        overlay, frame = _capture_six_tick(controller, now+i*dt, current, speed)
        assert overlay == ZERO and frame["phase"] == "velocity_check"
        assert frame["quantized_velocity_debug"]["started_at_sec"] == now
    for i in range(13, 18):
        overlay, frame = _capture_six_tick(controller, now+i*dt, q)
        assert overlay == ZERO and frame["phase"] != "fault"
    assert frame["phase"] == "inactive"


@pytest.mark.parametrize("shape", ["two_counts", "short_interval", "inconsistent", "sub_count", "moving_window"])
def test_capture_support_exemption_rejects_unqualified_shape(shape):
    controller, now = _onset_trial()
    count, dt = .00038349442183971405, .006856124149635434
    if shape == "moving_window":
        for i in range(1, 9):
            _capture_six_tick(controller, now+i*.01, (0., 0., i*.0001, 0., 0., 0.), (0., 0., .01, 0., 0., 0.))
        now += .08
    q = -count*(2 if shape == "two_counts" else .5 if shape == "sub_count" else 1)
    if shape == "short_interval":
        dt = .003
    elif shape == "sub_count":
        dt /= 2
    raw = -.06 if shape == "inconsistent" else q/dt
    overlay, frame = _capture_six_tick(controller, now+dt, (q, 0., .0008 if shape == "moving_window" else 0., 0., 0., 0.), (raw, 0., 0., 0., 0., 0.))
    assert overlay == ZERO and frame["phase"] == "fault"
    assert "onset_capture_raw_velocity_exceeded: joint1" in frame["fault"]


def test_capture_qualified_support_spike_never_masks_a_second_invalid_support_axis():
    controller, now = _onset_trial()
    count, dt = .00038349442183971405, .006856124149635434
    overlay, frame = _capture_six_tick(controller, now+dt, (-count, 0., 0., 0., 2*count, 0.),
        (-count/dt, 0., 0., 0., 2*count/dt, 0.))
    assert overlay == ZERO and frame["phase"] == "fault"
    assert "onset_capture_raw_velocity_exceeded: joint5" in frame["fault"]


@pytest.mark.parametrize("condition,reason", [
    ("continued_support", "velocity_check_continued_motion"),
    ("no_zero", "velocity_check_zero_dispatch_timeout"),
    ("conditioned_tail", "velocity_check_zero_dispatch_timeout"),
    ("gap", "velocity_debug_feedback_gap_exceeded"),
])
def test_capture_support_check_preserves_zero_timing_and_continued_motion_faults(condition, reason):
    controller, now = _onset_trial()
    count, dt = .00038349442183971405, .006856124149635434
    q = (-count, 0., 0., 0., 0., 0.)
    now += dt
    tail = (.01, 0., 0., 0., 0., 0.) if condition == "conditioned_tail" else ZERO
    _capture_six_tick(controller, now, q, (-count/dt, 0., 0., 0., 0., 0.), ack=condition != "no_zero", tail=tail)
    for i in range(1, 6):
        if condition == "continued_support":
            q = (-(i+1)*count, 0., 0., 0., 0., 0.)
        overlay, frame = _capture_six_tick(controller, now+i*(.041 if condition == "gap" else .01), q,
            (-count/.01, 0., 0., 0., 0., 0.) if condition == "continued_support" else ZERO,
            ack=condition != "no_zero", tail=tail)
        if frame["phase"] == "fault":
            break
    assert overlay == ZERO and frame["phase"] == "fault" and reason in frame["fault"]


@pytest.mark.parametrize("capture", [False, True])
def test_set_zero_ends_only_capture_trial_and_requires_fresh_manual_baseline(capture):
    controller, now = _onset_trial() if capture else _velocity_debug_trial()
    controller.submit("set", now_sec=now+.001, torque_nm=0.)
    overlay, frame = _velocity_debug_tick(controller, now+.01)
    assert frame["phase"] == ("onset_release" if capture else "armed")
    if not capture:
        return
    assert overlay == ZERO and frame["onset_capture"]["trial"]["outcome"] == "operator_zero"
    release = frame["onset_capture"]["release_started_at_sec"]
    with pytest.raises(ValueError):
        controller.submit("set", now_sec=now+.011, torque_nm=.8)
    controller.submit("zero", now_sec=now+.012)
    assert controller.status()["onset_capture"]["release_started_at_sec"] == release
    for i in range(2, 12):
        overlay, frame = _velocity_debug_tick(controller, now+i*.01, q=.0003834962844848633)
    assert overlay == ZERO and frame["phase"] == "inactive"
    controller.submit("arm", now_sec=now+.115)
    _velocity_debug_tick(controller, now+.12, q=.0003834962844848633)
    controller.submit("set", now_sec=now+.121, torque_nm=-.2)
    _, frame = _velocity_debug_tick(controller, now+.13, q=.0003834962844848633)
    assert frame["onset_capture"]["trial"]["baseline_qpos_rad"][2] == .0003834962844848633
    assert frame["onset_capture"]["trial"]["direction"] == -1


@pytest.mark.parametrize("zero_ack", [False, True])
def test_capture_spike_during_release_keeps_original_first_zero_deadline(zero_ack):
    controller, now = _onset_trial()
    controller.submit("set", now_sec=now+.001, torque_nm=0.)
    for i in range(1, 4):
        _capture_six_tick(controller, now+i*.01, ZERO, ack=False)
    count, dt = .00038349442183971405, .006856124149635434
    spike_at = now+.03+dt
    q = (-count, 0., 0., 0., 0., 0.)
    overlay, frame = _capture_six_tick(controller, spike_at, q, (-count/dt, 0., 0., 0., 0., 0.), ack=zero_ack)
    assert frame["phase"] == "velocity_check" and overlay == ZERO
    for i in range(1, 11):
        overlay, frame = _capture_six_tick(controller, spike_at+i*.01, q, ack=zero_ack)
        if frame["phase"] in {"fault", "inactive"}:
            break
    if zero_ack:
        assert frame["phase"] == "inactive" and frame["fault"] is None
        assert frame["onset_capture"]["first_zero_completed_at_sec"] == pytest.approx(spike_at+.001)
    else:
        assert frame["phase"] == "fault" and frame["fault"] == "onset_capture_zero_dispatch_timeout"
        assert spike_at+i*.01 < spike_at+.04  # Original release, not a renewed watchdog.


@pytest.mark.parametrize("condition", ["settle_timeout", "stale_zero"])
def test_capture_spike_during_release_never_extends_original_deadline_or_zero_freshness(condition):
    controller, now = _onset_trial(config=B601MotionOnsetCaptureConfig(settle_timeout_sec=.2))
    controller.submit("set", now_sec=now+.001, torque_nm=0.)
    count = .00038349442183971405
    for i in range(1, 16):
        q = (0., 0., min(i, 8)*count, 0., 0., 0.)
        overlay, frame = _capture_six_tick(controller, now+i*.01, q, (0., 0., count/.01 if i <= 8 else 0., 0., 0., 0.),
            ack=condition != "stale_zero" or i <= 11)
        assert frame["phase"] == "onset_release" and overlay == ZERO
    spike_at = now+.15+.006856124149635434
    q = (-count, 0., 8*count, 0., 0., 0.)
    overlay, frame = _capture_six_tick(controller, spike_at, q, (-count/.006856124149635434, 0., 0., 0., 0., 0.))
    if condition == "stale_zero":
        assert overlay == ZERO and frame["phase"] == "fault"
        assert frame["fault"] == "onset_capture_zero_dispatch_timeout"
        return
    assert frame["phase"] == "velocity_check" and overlay == ZERO
    for i in range(1, 7):
        overlay, frame = _capture_six_tick(controller, spike_at+i*.01, q)
        if frame["phase"] == "fault":
            break
    assert frame["phase"] == "fault" and frame["fault"] == "onset_capture_settle_timeout"
    assert spike_at+i*.01 < spike_at+.08


@pytest.mark.parametrize("tail", [(0., 0., .01, 0., 0., 0.), (.01, 0., 0., 0., 0., 0.)])
def test_capture_original_zero_cannot_reappear_as_tail_before_new_spike_zero(tail):
    controller, now = _onset_trial()
    controller.submit("set", now_sec=now+.001, torque_nm=0.)
    count = .00038349442183971405
    for i in range(1, 16):
        q = (0., 0., min(i, 8)*count, 0., 0., 0.)
        _capture_six_tick(controller, now+i*.01, q, (0., 0., count/.01 if i <= 8 else 0., 0., 0., 0.))
    spike_at = now+.15+.006856124149635434
    overlay, frame = _capture_six_tick(controller, spike_at, (-count, 0., 8*count, 0., 0., 0.),
        (-count/.006856124149635434, 0., 0., 0., 0., 0.), ack=False)
    assert frame["phase"] == "velocity_check" and overlay == ZERO
    assert frame["quantized_velocity_debug"]["first_zero_completed_at_sec"] is None
    assert frame["onset_capture"]["first_zero_completed_at_sec"] is not None
    with pytest.raises(ValueError, match="cannot reappear"):
        _velocity_debug_ack(controller, spike_at, overlay, tail=tail)
    assert controller.status()["phase"] == "fault"
    assert controller.status()["fault"] == "onset_capture_nonzero_after_zero"


def test_count_crossings_are_diagnostics_only_and_capture_prior_dispatch_not_future_command():
    controller, now = _onset_trial()
    count = .0003834962844848633
    _velocity_debug_tick(controller, now+.01, q=count, raw=count/.01)
    first = controller.status()["onset_capture"]["trial"]["first_1_count_observation"]
    assert first["dispatch"]["feedback_timestamp_sec"] == now
    assert first["dispatch"]["completed_at_sec"] < first["feedback_timestamp_sec"]
    assert "first_criterion_observation" not in controller.status()["onset_capture"]["trial"]
    for i in range(2, 15):
        _, frame = _velocity_debug_tick(controller, now+i*.01, q=count if i < 5 else 2*count if i < 9 else 3*count)
        assert frame["phase"] == "armed" and frame["onset_capture"]["confirmation_streak"] == 0
    for i in range(15, 18):
        _, frame = _velocity_debug_tick(controller, now+i*.01, q=4*count, raw=count/.01 if i == 15 else 0.)
    trial = frame["onset_capture"]["trial"]
    assert frame["phase"] == "onset_release"
    assert trial["first_1_count_observation"] == first
    assert trial["onset_confirmed_at_sec"]-trial["first_criterion_observation"]["feedback_timestamp_sec"] == pytest.approx(.02)
    assert all(f"first_{i}_count_observation" in trial for i in (1, 2, 3))


@pytest.mark.parametrize("capture", [False, True])
def test_only_onset_capture_requires_count_bounded_preparation(capture):
    controller = B601TorqueExperiment(B601TorqueExperimentLimits("joint3"), quantized_velocity_debug=True,
        onset_capture=B601MotionOnsetCaptureConfig() if capture else None)
    controller.owner_ready()
    for i in range(20):
        q = (i*.0001,)*6
        overlay, _ = controller.step(now_sec=1+i*.01, qpos_rad=q, qvel_rad_s=(.01,)*6)
        _velocity_debug_ack(controller, 1+i*.01, overlay)
    controller.submit("arm", now_sec=1.195)
    overlay, frame = controller.step(now_sec=1.2, qpos_rad=(.002,)*6, qvel_rad_s=(.01,)*6)
    assert overlay == ZERO
    assert frame["phase"] == ("inactive" if capture else "armed")
    if capture:
        assert frame["fault"] == "arm_rejected_onset_capture_position_not_settled"


def test_onset_capture_count_jitter_preparation_still_passes():
    controller = B601TorqueExperiment(B601TorqueExperimentLimits("joint3"), quantized_velocity_debug=True,
        onset_capture=B601MotionOnsetCaptureConfig())
    controller.owner_ready()
    count = .0003834962844848633
    for i in range(20):
        overlay, _ = controller.step(now_sec=1+i*.01, qpos_rad=((i%2)*count,)*6, qvel_rad_s=(.03835,)*6)
        _velocity_debug_ack(controller, 1+i*.01, overlay)
    controller.submit("arm", now_sec=1.195)
    overlay, frame = controller.step(now_sec=1.2, qpos_rad=ZERO, qvel_rad_s=(-.03835,)*6)
    assert overlay == ZERO and frame["phase"] == "armed"


def test_onset_release_does_not_call_slow_continued_sliding_stationary():
    controller, now = _onset_trial(config=B601MotionOnsetCaptureConfig(settle_timeout_sec=.3))
    now, q, _ = _confirm_onset(controller, now)
    for i in range(1, 33):
        q += .0001
        overlay, frame = _velocity_debug_tick(controller, now+i*.01, q=q, raw=.01)
        assert overlay == ZERO and frame["phase"] != "inactive"
        if frame["phase"] == "fault":
            break
    assert frame["fault"] == "onset_capture_settle_timeout"


def test_onset_capture_actual_feedback_not_completion_time_seeds_rest_window():
    controller, now = _onset_trial()
    now, q, _ = _confirm_onset(controller, now, ack=False)
    _velocity_debug_ack(controller, now, ZERO, delay=.015)
    _, frame = _velocity_debug_tick(controller, now+.01, q=q, ack=False)
    assert not frame["velocity_window"]["ready"] and frame["velocity_window"]["span_sec"] == 0.
    _velocity_debug_ack(controller, now+.01, ZERO, delay=.006)
    for i in range(2, 10):
        _, frame = _velocity_debug_tick(controller, now+i*.01, q=q)
    assert frame["phase"] == "onset_release" and not frame["velocity_window"]["ready"]
    _, frame = _velocity_debug_tick(controller, now+.10, q=q)
    assert frame["phase"] == "inactive" and frame["velocity_window"]["ready"]


def test_onset_capture_gui_consumes_public_pending_and_completed_state(monkeypatch):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the pure GUI message contract")
    controller, now = _onset_trial()
    now, q, _ = _confirm_onset(controller, now)
    monkeypatch.setattr("robot_motion.rebot_b601_rs.torque_experiment.time.monotonic", lambda: now+.005)
    pending = controller.status()
    for i in range(1, 12):
        _velocity_debug_tick(controller, now+i*.01, q=q)
    monkeypatch.setattr("robot_motion.rebot_b601_rs.torque_experiment.time.monotonic", lambda: now+.115)
    completed = controller.status()
    source = (Path(__file__).parents[1] / "control_station/b601_torque_console/static/app.js").read_text()
    functions = source[source.index("function number("):source.index("function post(")]
    predicate = source.split('el("arm").disabled = ', 1)[1].split(";", 1)[0]
    script = "const vm=require('node:vm'); const c={}; vm.runInNewContext(JSON.parse(process.argv[1]),c); const states=JSON.parse(require('node:fs').readFileSync(0,'utf8')); console.log(JSON.stringify(states.map(state=>[c.gateMessage(state),Boolean("+predicate+"),c.onsetCaptureDetails(state.onset_capture)])));"
    result = subprocess.run([node,"-e",script,json.dumps(functions)],input=json.dumps([pending,completed]),check=True,capture_output=True,text=True)
    messages = json.loads(result.stdout)
    assert "已确认持续起动" in messages[0][0] and "GUI 和重力补偿保留" in messages[0][0]
    assert messages[0][1] is True
    assert "起动记录已完成" in messages[1][0] and "旧驱动力不会恢复" in messages[1][0]
    assert messages[1][1] is False
    assert "1 计数首见" in messages[1][2] and "连续确认起动" in messages[1][2]
    assert "量化/预滑" in messages[1][2] and "不是实际电机或人体受力测量" in messages[1][2]


@pytest.mark.parametrize("options", [
    {"displacement_rad":.0007}, {"confirm_samples":1}, {"confirm_samples":True},
    {"window_velocity_rad_s":float("nan")}, {"settle_timeout_sec":.1},
])
def test_onset_capture_config_rejects_invalid_or_single_count_criteria(options):
    with pytest.raises((TypeError, ValueError)):
        B601MotionOnsetCaptureConfig(**options)


def test_onset_capture_launcher_forwards_and_prevalidates_opt_in_settings(tmp_path):
    args = owner._parser().parse_args([
        "--allow-motion", "--torque-console", "--rate-hz", "100",
        "--torque-console-confirmation", "B601_RS_TORQUE_CONSOLE",
        "--torque-console-quantized-velocity-debug", "--torque-console-onset-capture",
    ])
    parsed = probe.build_parser().parse_args(owner.build_probe_command(args, run_dir=tmp_path)[2:])
    probe._validate_torque_console_cli(parsed)
    controller = probe._direct_torque_console_from_args(parsed)
    assert controller.status()["onset_capture"]["settings"] == vars(B601MotionOnsetCaptureConfig())
    parsed.torque_console_quantized_velocity_debug = False
    with pytest.raises(probe.LowGainProbeError, match="explicit direct J3"):
        probe._validate_torque_console_cli(parsed)


def _velocity_debug_ack(controller, now, overlay, *, tail=ZERO, delay=.001, sequence=None):
    gravity = (0., 2., 6., 1., 0., 0.)
    total = tuple(g + u + residual for g, u, residual in zip(gravity, overlay, tail, strict=True))
    controller.record_dispatch(
        joint_name="joint3", receipt_sequence=round(now*1_000_000) if sequence is None else sequence,
        feedback_timestamp_sec=now, completed_at_sec=now+delay,
        overlay_after_profile_nm=overlay[2], gravity_nm=gravity[2],
        requested_total_nm=gravity[2]+overlay[2], conditioned_total_nm=total[2],
        joint_commands={"gravity_nm": gravity, "overlay_nm": overlay, "total_nm": total},
    )


def _velocity_debug_tick(controller, now, *, q=0., raw=0., support=0., ack=True, tail=ZERO, onset_start_permitted=None):
    if controller.status()["phase"] == "armed":
        controller.submit("heartbeat", now_sec=now)
    overlay, frame = controller.step(
        now_sec=now, qpos_rad=(0., 0., q, support, 0., 0.),
        qvel_rad_s=(0., 0., raw, 0., 0., 0.),
        onset_start_permitted=onset_start_permitted,
    )
    if ack and frame["phase"] != "fault":
        _velocity_debug_ack(controller, now, overlay, tail=tail)
    return overlay, frame


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("permission", [None, True, False])
def test_onset_start_baseline_race_waits_only_with_explicit_automatic_admission(direction, permission):
    controller, now = _onset_trial(direction=direction, ramp_ticks=0)
    count = direction * .0003834962844848633
    # A zero-only automatic admission sample creates no trial or budget.
    overlay, frame = _velocity_debug_tick(
        controller, now+.01, q=count, raw=count/.01, onset_start_permitted=False,
    )
    assert overlay == ZERO and frame["onset_capture"]["trial"] is None
    overlay, frame = _velocity_debug_tick(
        controller, now+.02, q=2*count, raw=count/.01, onset_start_permitted=permission,
    )
    assert overlay == ZERO
    assert frame["onset_capture"]["trial"] is None
    if permission is None:
        assert frame["phase"] == "fault"
        assert frame["fault"] == "onset_capture_baseline_onset_capture_position_not_settled"
    else:
        assert frame["phase"] == "armed" and frame["fault"] is None
        assert frame["onset_start_wait_reason"] == "onset_capture_position_not_settled"


def test_onset_start_permission_never_masks_raw_velocity_or_feedback_gap():
    for failure in ("raw", "gap"):
        controller, now = _onset_trial(ramp_ticks=0)
        delta = .001 if failure == "raw" else 0.
        overlay, frame = _velocity_debug_tick(
            controller, now+(.01 if failure == "raw" else .05), q=delta,
            raw=.1 if failure == "raw" else 0., onset_start_permitted=False,
        )
        assert overlay == ZERO and frame["phase"] == "fault"
        assert frame["onset_capture"]["trial"] is None


@pytest.mark.parametrize("permission", [0, 1, "yes"])
def test_onset_start_permission_rejects_non_boolean_values(permission):
    controller, now = _onset_trial(ramp_ticks=0)
    with pytest.raises(ValueError, match="onset start permission"):
        controller.step(now_sec=now+.01, qpos_rad=ZERO, qvel_rad_s=ZERO, onset_start_permitted=permission)


@pytest.mark.parametrize("axis", [2, 3])
def test_onset_start_pending_cold_window_cannot_mask_raw_hard_violation(axis):
    controller = B601TorqueExperiment(B601TorqueExperimentLimits("joint3"),
        quantized_velocity_debug=True, onset_capture=B601MotionOnsetCaptureConfig())
    controller.owner_ready()
    controller.submit("arm", now_sec=1.)
    raw = list(ZERO)
    raw[axis] = .051
    overlay, frame = controller.step(now_sec=1.01, qpos_rad=ZERO,
        qvel_rad_s=raw, onset_start_permitted=False)
    assert overlay == ZERO and frame["phase"] == "fault"
    assert frame["fault"] == "onset_capture_baseline_raw_velocity_hard_guard"
    assert frame["onset_capture"]["trial"] is None


def _velocity_debug_trial(*, enabled=True, excursion=.1):
    controller = B601TorqueExperiment(
        B601TorqueExperimentLimits("joint3", max_abs_assist_nm=.8, max_excursion_rad=excursion),
        quantized_velocity_debug=enabled,
    )
    controller.owner_ready()
    for i in range(20):
        _velocity_debug_tick(controller, 1+i*.01)
    controller.submit("arm", now_sec=1.2)
    _, frame = _velocity_debug_tick(controller, 1.2)
    assert frame["phase"] == "armed"
    controller.submit("set", now_sec=1.201, torque_nm=.2)
    for i in range(1, 151):
        _velocity_debug_tick(controller, 1.2+i*.01)
    assert controller.status()["effective_assist_nm"] == pytest.approx(.2)
    return controller, 2.7


def _velocity_debug_spike(controller, now, **kwargs):
    # Exact one-count/7.518 ms shape observed in the jp6762rh hardware trace.
    dt, count = .007518464, .0003834962844848633
    result = _velocity_debug_tick(controller, now+dt, q=-count, raw=-count/dt, **kwargs)
    return now+dt, result


def test_default_single_count_speed_spike_still_latches_original_guard():
    controller, now = _velocity_debug_trial(enabled=False)
    _, (overlay, frame) = _velocity_debug_spike(controller, now)
    assert frame["phase"] == "fault" and overlay == ZERO
    assert "joint_velocity_exceeded" in frame["fault"]
    assert "quantized_velocity_debug" not in frame


def test_debug_spike_withdraws_first_then_requires_new_manual_arm_and_target():
    controller, now = _velocity_debug_trial()
    now, (overlay, frame) = _velocity_debug_spike(controller, now)
    assert overlay == ZERO and frame["phase"] == "velocity_check"
    assert frame["fault"] is None and frame["desired_assist_nm"] == 0
    assert controller.status()["quantized_velocity_debug"]["zero_dispatch_confirmed"]
    for action in ("arm", "set", "select_joint", "heartbeat"):
        with pytest.raises(ValueError):
            controller.submit(action, now_sec=now+.001, torque_nm=.2, joint_name="joint3")
    for i in range(1, 10):
        if i == 2:
            controller.submit("withdraw", now_sec=now+i*.01)
        q = -.0003834962844848633 if i % 2 else 0.
        raw = (-1 if i % 2 else 1)*.03834962844848633
        overlay, frame = _velocity_debug_tick(controller, now+i*.01, q=q, raw=raw)
        assert overlay == ZERO and frame["phase"] != "fault"
    assert frame["phase"] == "inactive"
    assert frame["quantized_velocity_debug"]["reason"] == "zero_confirmed_manual_rearm_required"
    with pytest.raises(ValueError, match="armed"):
        controller.submit("set", now_sec=now+.095, torque_nm=.2)
    controller.submit("arm", now_sec=now+.096)
    overlay, frame = _velocity_debug_tick(controller, now+.10, q=q)
    assert frame["phase"] == "armed" and overlay == ZERO
    assert frame["desired_assist_nm"] == 0


@pytest.mark.parametrize("tail", [None, (0., 0., .02, 0., 0., 0.), (0., .001, 0., 0., 0., 0.)])
def test_debug_check_requires_all_six_actual_commands_zero_not_just_zero_request(tail):
    controller, now = _velocity_debug_trial()
    now, _ = _velocity_debug_spike(controller, now, ack=tail is not None, tail=tail or ZERO)
    for i in range(1, 5):
        overlay, frame = _velocity_debug_tick(
            controller, now+i*.01, q=-.0003834962844848633,
            ack=tail is not None, tail=tail or ZERO,
        )
    assert overlay == ZERO and frame["phase"] == "fault"
    assert frame["fault"] == "velocity_check_zero_dispatch_timeout"


@pytest.mark.parametrize("shape", ["two_counts", "short_interval", "inconsistent_speed", "fractional_count"])
def test_debug_does_not_exempt_fast_or_multi_count_or_inconsistent_motion(shape):
    controller, now = _velocity_debug_trial()
    count, dt = .0003834962844848633, .01
    q = -2*count if shape == "two_counts" else -count
    if shape == "short_interval":
        dt = .003
    elif shape == "fractional_count":
        q, dt = -count/2, .003759232
    raw = -.07 if shape == "inconsistent_speed" else q/dt
    overlay, frame = _velocity_debug_tick(controller, now+dt, q=q, raw=raw)
    assert frame["phase"] == "fault" and overlay == ZERO
    assert "joint_velocity_exceeded" in frame["fault"]


@pytest.mark.parametrize("fault", ["continued_motion", "feedback_gap", "support_drift", "confirmation_timeout"])
def test_debug_confirmation_keeps_hard_faults_and_bounded_deadline(fault):
    controller, now = _velocity_debug_trial()
    now, _ = _velocity_debug_spike(controller, now)
    count = .0003834962844848633
    if fault == "continued_motion":
        # Range is measured from the first actual post-zero feedback, not
        # from the earlier spike. Two further counts must still fault.
        for i in range(1, 4):
            overlay, frame = _velocity_debug_tick(controller, now+i*.01, q=-(i+1)*count, raw=-count/.01)
    elif fault == "feedback_gap":
        overlay, frame = _velocity_debug_tick(controller, now+.041, q=-count)
    elif fault == "support_drift":
        overlay, frame = _velocity_debug_tick(controller, now+.01, q=-count, support=.009)
    else:
        # Continued support motion prevents quiet confirmation without
        # crossing the original support drift bound.
        for i in range(1, 22):
            overlay, frame = _velocity_debug_tick(controller, now+i*.01, q=-count, support=i*.00025)
    assert overlay == ZERO and frame["phase"] == "fault"
    expected = {"continued_motion": "velocity_check_continued_motion",
                "feedback_gap": "velocity_debug_feedback_gap_exceeded",
                "support_drift": "support_joint_drift_exceeded",
                "confirmation_timeout": "velocity_check_confirmation_timeout"}[fault]
    assert expected in frame["fault"]


def test_debug_spike_cannot_bypass_original_excursion_guard():
    controller, now = _velocity_debug_trial(excursion=.0002)
    _, (overlay, frame) = _velocity_debug_spike(controller, now)
    assert overlay == ZERO and frame["phase"] == "fault"
    assert "joint_excursion_exceeded" in frame["fault"]


def test_debug_confirmation_range_starts_at_first_post_zero_feedback():
    controller, now = _velocity_debug_trial()
    count, dt = .0003834962844848633, .0065420628525316715
    now += dt
    overlay, frame = _velocity_debug_tick(controller, now, q=-count, raw=-count/dt, ack=False)
    assert frame["phase"] == "velocity_check" and overlay == ZERO
    # hgo3w0oh: the first dispatch still has a conditioning tail; the next
    # dispatch confirms zero. Do not invent a zero receipt at the spike.
    _velocity_debug_ack(controller, now, overlay, tail=(0., 0., .01013909967, 0., 0., 0.))
    overlay, _ = _velocity_debug_tick(controller, now+.008942041, q=0., raw=count/.008942041, ack=False)
    _velocity_debug_ack(controller, now+.008942041, overlay, delay=.001612176)
    for offset, q, raw in (
        (.018971026, -count, -.03823879294),
        (.029012881, -count, 0.),
        (.038948596, -count, 0.),
        (.048956082, -count, 0.),
        (.060106241, -2*count, -.03439379533),
    ):
        overlay, frame = _velocity_debug_tick(controller, now+offset, q=q, raw=raw)
        assert overlay == ZERO and frame["phase"] == "velocity_check"
    debug = frame["quantized_velocity_debug"]
    assert debug["confirmation_first_feedback_at_sec"] == pytest.approx(now+.018971026)
    assert debug["confirmation_position_span_rad"] == pytest.approx(count)
    assert not frame["velocity_window"]["ready"]
    # Synthetic quiet continuation tests completion only; it is not a claim
    # about the real run after its original guard/user repositioning.
    for i in range(7, 12):
        overlay, frame = _velocity_debug_tick(controller, now+i*.01, q=-2*count)
    assert overlay == ZERO and frame["phase"] == "inactive"
    assert frame["desired_assist_nm"] == 0.
    with pytest.raises(ValueError, match="armed"):
        controller.submit("set", now_sec=now+.115, torque_nm=.2)


def test_debug_confirmation_has_feedback_sampling_margin_after_larger_withdrawal():
    controller, now = _velocity_debug_trial()
    controller.submit("set", now_sec=now+.001, torque_nm=.6)
    for i in range(1, 301):
        _velocity_debug_tick(controller, now+i*.01)
    now += 3.0
    count, dt = .0003834962844848633, .007441427791491151
    now += dt
    overlay, _ = _velocity_debug_tick(controller, now, q=-count, raw=-count/dt, ack=False)
    _velocity_debug_ack(controller, now, overlay, tail=(0., 0., .385551, 0., 0., 0.))
    overlay, _ = _velocity_debug_tick(controller, now+.010045485, q=-count, ack=False)
    _velocity_debug_ack(controller, now+.010045485, overlay, tail=(0., 0., .096255, 0., 0., 0.))
    overlay, _ = _velocity_debug_tick(controller, now+.020062005, q=-count, ack=False)
    _velocity_debug_ack(controller, now+.020062005, overlay, delay=.001335796)
    # Exact feedback offsets of the 9882wvyz withdrawal. The penultimate
    # window is 50 microseconds short; the next frame is 8us beyond 120ms.
    for offset in (.030044664, .040100675, .0499769, .06001, .0700447,
                   .0800251, .0905447, .1000014, .10999412):
        overlay, frame = _velocity_debug_tick(controller, now+offset, q=-count)
        assert overlay == ZERO and frame["phase"] == "velocity_check"
    assert not frame["velocity_window"]["ready"]
    overlay, frame = _velocity_debug_tick(controller, now+.120008094, q=-count)
    assert overlay == ZERO and frame["phase"] == "inactive"
    assert frame["velocity_window"]["ready"] and frame["desired_assist_nm"] == 0.
    with pytest.raises(ValueError, match="armed"):
        controller.submit("set", now_sec=now+.121, torque_nm=.6)


@pytest.mark.parametrize("debug", [False, True])
def test_only_explicit_debug_preparation_accepts_quiet_window_with_single_count_raw_jitter(debug):
    controller = B601TorqueExperiment(B601TorqueExperimentLimits("joint3"), quantized_velocity_debug=debug)
    controller.owner_ready()
    count = .0003834962844848633
    for i in range(20):
        q = ((i % 2)*count,)*6
        overlay, _ = controller.step(now_sec=1+i*.01, qpos_rad=q, qvel_rad_s=(.03835,)*6)
        _velocity_debug_ack(controller, 1+i*.01, overlay)
    controller.submit("arm", now_sec=1.195)
    overlay, frame = controller.step(now_sec=1.2, qpos_rad=ZERO, qvel_rad_s=(-count/.01,)*6)
    assert overlay == ZERO
    if debug:
        assert frame["phase"] == "armed" and frame["desired_assist_nm"] == 0.
    else:
        assert frame["phase"] == "inactive" and frame["fault"] == "arm_rejected_not_stationary"


@pytest.mark.parametrize("condition,reason", [
    ("raw_speed", "raw_velocity_hard_guard"), ("window_speed", "not_stationary"),
    ("window_displacement", "velocity_window_displacement_exceeded"),
    ("zero_unconfirmed", "zero_dispatch_unconfirmed"), ("cold_window", "velocity_window_not_ready"),
])
def test_debug_preparation_keeps_hard_motion_window_and_final_zero_gates(condition, reason):
    controller = B601TorqueExperiment(B601TorqueExperimentLimits("joint3"), quantized_velocity_debug=True)
    controller.owner_ready()
    samples = 4 if condition == "cold_window" else 20
    for i in range(samples):
        position = i*.0003 if condition == "window_speed" else (i%2)*.003 if condition == "window_displacement" else 0.
        overlay, _ = controller.step(now_sec=1+i*.01, qpos_rad=(position,)*6, qvel_rad_s=ZERO)
        _velocity_debug_ack(controller, 1+i*.01, overlay, tail=(0., 0., .01, 0., 0., 0.) if condition == "zero_unconfirmed" else ZERO)
    now = 1+samples*.01
    controller.submit("arm", now_sec=now-.001)
    position = samples*.0003 if condition == "window_speed" else position
    raw = (0., 0., 0., 0., .06, 0.) if condition == "raw_speed" else ZERO
    overlay, frame = controller.step(now_sec=now, qpos_rad=(position,)*6, qvel_rad_s=raw)
    assert overlay == ZERO and frame["phase"] == "inactive"
    assert frame["fault"] == "arm_rejected_"+reason
    assert frame["quantized_velocity_debug"]["preparation_gate_reason"] == reason
    with pytest.raises(ValueError, match="armed"):
        controller.submit("set", now_sec=now+.001, torque_nm=.2)


@pytest.mark.parametrize("observation,ack,delay,reason", [
    (1.205, True, .001, "ready"),
    (1.199, True, .001, "feedback_not_fresh"),
    (1.201, True, .002, "zero_dispatch_unconfirmed"),
    (1.2005, True, .001, "zero_dispatch_unconfirmed"),
    (1.205, False, .001, "zero_dispatch_unconfirmed"),
    (1.241, True, .001, "feedback_not_fresh"),
])
def test_debug_public_status_and_gui_use_current_post_dispatch_observation(
    monkeypatch, observation, ack, delay, reason,
):
    controller = B601TorqueExperiment(B601TorqueExperimentLimits("joint3"), quantized_velocity_debug=True)
    controller.owner_ready()
    for i in range(21):
        now = 1+i*.01
        overlay, _ = controller.step(now_sec=now, qpos_rad=ZERO, qvel_rad_s=ZERO)
        if ack:
            _velocity_debug_ack(controller, now, overlay, delay=delay)
    monkeypatch.setattr("robot_motion.rebot_b601_rs.torque_experiment.time.monotonic", lambda: observation)
    state = controller.status()
    assert state["quantized_velocity_debug"]["preparation_gate_reason"] == reason
    assert state["phase"] == "inactive" and state["desired_assist_nm"] == 0.
    assert state["effective_assist_nm"] == 0.
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to execute the GUI button predicate")
    source = (Path(__file__).parents[1] / "control_station/b601_torque_console/static/app.js").read_text()
    predicate = source.split('el("arm").disabled = ', 1)[1].split(";", 1)[0]
    script = "const state=JSON.parse(process.argv[1]); console.log(Boolean("+predicate+"));"
    result = subprocess.run([node, "-e", script, json.dumps(state)], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == ("false" if reason == "ready" else "true")


def test_debug_public_status_rejects_stale_zero_receipt_with_fresh_feedback(monkeypatch):
    controller = B601TorqueExperiment(B601TorqueExperimentLimits("joint3"), quantized_velocity_debug=True)
    controller.owner_ready()
    for i in range(21):
        now = 1+i*.01
        overlay, _ = controller.step(now_sec=now, qpos_rad=ZERO, qvel_rad_s=ZERO)
        if i < 17:
            _velocity_debug_ack(controller, now, overlay)
    monkeypatch.setattr("robot_motion.rebot_b601_rs.torque_experiment.time.monotonic", lambda: 1.205)
    assert controller.status()["quantized_velocity_debug"]["preparation_gate_reason"] == "zero_dispatch_unconfirmed"


def test_debug_pre_zero_feedback_cannot_seed_or_restart_post_zero_range():
    controller, now = _velocity_debug_trial()
    now, _ = _velocity_debug_spike(controller, now, ack=False)
    count = .0003834962844848633
    overlay, frame = _velocity_debug_tick(controller, now+.01, q=-2*count, raw=-count/.01, ack=False)
    assert frame["phase"] == "velocity_check" and overlay == ZERO
    _velocity_debug_ack(controller, now+.01, overlay, tail=(0., 0., .01, 0., 0., 0.))
    overlay, _ = _velocity_debug_tick(controller, now+.02, q=-2*count)
    assert controller.status()["quantized_velocity_debug"]["confirmation_position_span_rad"] is None
    overlay, frame = _velocity_debug_tick(controller, now+.03, q=-2*count)
    assert frame["quantized_velocity_debug"]["confirmation_first_feedback_at_sec"] == now+.03
    overlay, frame = _velocity_debug_tick(controller, now+.04, q=-3*count, raw=-count/.01)
    assert frame["phase"] == "velocity_check"
    controller.submit("zero", now_sec=now+.041)
    controller.submit("withdraw", now_sec=now+.041)
    overlay, frame = _velocity_debug_tick(controller, now+.05, q=-4*count, raw=-count/.01)
    assert overlay == ZERO and frame["fault"] == "velocity_check_continued_motion"
    assert frame["quantized_velocity_debug"]["confirmation_first_feedback_at_sec"] == now+.03


def test_debug_feedback_at_or_before_zero_completion_is_not_confirmation_input():
    controller, now = _velocity_debug_trial()
    now, (overlay, _) = _velocity_debug_spike(controller, now, ack=False)
    count = .0003834962844848633
    _velocity_debug_ack(controller, now, overlay, delay=.02)
    for offset in (.01, .02):
        overlay, frame = _velocity_debug_tick(controller, now+offset, q=-count, ack=False)
        assert frame["phase"] == "velocity_check" and overlay == ZERO
        assert frame["quantized_velocity_debug"]["confirmation_first_feedback_at_sec"] is None
        assert not frame["velocity_window"]["ready"] and frame["velocity_window"]["span_sec"] == 0.
    overlay, frame = _velocity_debug_tick(controller, now+.03, q=-2*count, raw=-count/.01)
    assert frame["phase"] == "velocity_check" and overlay == ZERO
    assert frame["quantized_velocity_debug"]["confirmation_first_feedback_at_sec"] == now+.03
    assert frame["quantized_velocity_debug"]["confirmation_position_span_rad"] == 0.
    assert frame["velocity_window"]["span_sec"] == 0.


def test_debug_pre_zero_movement_still_obeys_original_total_excursion_guard():
    controller, now = _velocity_debug_trial(excursion=.0006)
    now, _ = _velocity_debug_spike(controller, now, ack=False)
    count = .0003834962844848633
    overlay, frame = _velocity_debug_tick(controller, now+.01, q=-2*count, raw=-count/.01)
    assert overlay == ZERO and frame["phase"] == "fault"
    assert "joint_excursion_exceeded" in frame["fault"]


@pytest.mark.parametrize("guard", ["raw_speed", "support_drift", "total_excursion"])
def test_debug_buffered_feedback_keeps_original_guards_before_confirmation_origin(guard):
    controller, now = _velocity_debug_trial(excursion=.0006 if guard == "total_excursion" else .1)
    now, (overlay, _) = _velocity_debug_spike(controller, now, ack=False)
    _velocity_debug_ack(controller, now, overlay, delay=.03)
    count = .0003834962844848633
    q = -3*count if guard == "raw_speed" else -2*count if guard == "total_excursion" else -count
    raw = -2*count/.01 if guard == "raw_speed" else -count/.01 if guard == "total_excursion" else 0.
    overlay, frame = _velocity_debug_tick(
        controller, now+.01, q=q, raw=raw, support=.009 if guard == "support_drift" else 0., ack=False,
    )
    assert overlay == ZERO and frame["phase"] == "fault"
    expected = {"raw_speed": "velocity_check_raw_speed_exceeded", "support_drift": "support_joint_drift_exceeded",
                "total_excursion": "joint_excursion_exceeded"}[guard]
    assert expected in frame["fault"]
    assert frame["quantized_velocity_debug"]["confirmation_first_feedback_at_sec"] is None


@pytest.mark.parametrize("delay", [-.001, .041])
def test_debug_dispatch_rejects_noncausal_or_late_completion(delay):
    controller, now = _velocity_debug_trial()
    now, (overlay, _) = _velocity_debug_spike(controller, now, ack=False)
    with pytest.raises(ValueError, match="timely completion"):
        _velocity_debug_ack(controller, now, overlay, delay=delay)
    assert not controller.status()["quantized_velocity_debug"]["zero_dispatch_confirmed"]


def test_debug_dispatch_rejects_replayed_and_missing_six_axis_receipts():
    controller, now = _velocity_debug_trial()
    now, (overlay, _) = _velocity_debug_spike(controller, now, ack=False)
    with pytest.raises(ValueError, match="sequence must advance"):
        _velocity_debug_ack(controller, now, overlay, sequence=1)
    with pytest.raises(ValueError, match="complete six-joint"):
        controller.record_dispatch(
            joint_name="joint3", receipt_sequence=round(now*1e6), feedback_timestamp_sec=now,
            completed_at_sec=now+.001, overlay_after_profile_nm=0., gravity_nm=6.,
            requested_total_nm=6., conditioned_total_nm=6.,
        )
    assert not controller.status()["quantized_velocity_debug"]["zero_dispatch_confirmed"]


def test_debug_late_zero_cannot_restart_original_dispatch_deadline():
    controller, now = _velocity_debug_trial()
    now, _ = _velocity_debug_spike(controller, now, ack=False)
    for i in range(1, 4):
        overlay, _ = _velocity_debug_tick(controller, now+i*.01, q=-.0003834962844848633, ack=False)
    # This packet is timely relative to its feedback but late relative to
    # the original withdrawal; it must not create a fresh 40 ms deadline.
    _velocity_debug_ack(controller, now+.03, overlay, delay=.011)
    assert not controller.status()["quantized_velocity_debug"]["zero_dispatch_confirmed"]
    overlay, frame = _velocity_debug_tick(controller, now+.05, q=-.0003834962844848633)
    assert overlay == ZERO and frame["fault"] == "velocity_check_zero_dispatch_timeout"


def test_debug_confirmation_uses_complete_post_zero_feedback_window_within_original_deadline():
    controller, now = _velocity_debug_trial()
    now, _ = _velocity_debug_spike(controller, now, ack=False)
    count = .0003834962844848633
    for i in range(1, 4):
        overlay, frame = _velocity_debug_tick(controller, now+i*.01, q=-count, ack=False)
    _velocity_debug_ack(controller, now+.03, overlay, delay=.001)
    assert not controller.status()["velocity_window"]["ready"]
    for i in range(4, 12):
        overlay, frame = _velocity_debug_tick(controller, now+i*.01, q=-count)
        assert frame["phase"] == "velocity_check" and overlay == ZERO
    overlay, frame = _velocity_debug_tick(controller, now+.12, q=-count)
    assert frame["phase"] == "inactive" and overlay == ZERO
    assert frame["velocity_window"]["span_sec"] >= .08-1e-9
    assert now+.12-frame["quantized_velocity_debug"]["first_zero_completed_at_sec"] >= .08


def test_debug_zero_boundary_cannot_extend_total_confirmation_deadline():
    controller, now = _velocity_debug_trial()
    now, _ = _velocity_debug_spike(controller, now, ack=False)
    count = .0003834962844848633
    for i in range(1, 4):
        overlay, _ = _velocity_debug_tick(controller, now+i*.01, q=-count, ack=False)
    _velocity_debug_ack(controller, now+.03, overlay, delay=.01)
    for i in range(4, 20):
        if i == 18:
            controller.submit("zero", now_sec=now+i*.01)
            controller.submit("withdraw", now_sec=now+i*.01)
        # Keep a support channel genuinely non-stationary inside its drift
        # envelope so that the expanded total deadline, not completion, wins.
        overlay, frame = _velocity_debug_tick(controller, now+i*.01, q=-count, support=i*.00025)
        assert frame["phase"] == "velocity_check" and overlay == ZERO
    overlay, frame = _velocity_debug_tick(controller, now+.201, q=-count, support=.005025)
    assert frame["phase"] == "fault" and overlay == ZERO
    assert frame["fault"] == "velocity_check_confirmation_timeout"


def test_debug_zero_receipt_does_not_allow_force_to_reappear():
    controller, now = _velocity_debug_trial()
    now, _ = _velocity_debug_spike(controller, now)
    overlay, _ = _velocity_debug_tick(controller, now+.01, q=-.0003834962844848633, ack=False)
    with pytest.raises(ValueError, match="cannot reappear"):
        _velocity_debug_ack(controller, now+.01, overlay, tail=(0., .001, 0., 0., 0., 0.))
    assert controller.status()["phase"] == "fault"


def test_debug_receipt_scalar_cannot_claim_zero_for_a_nonzero_six_joint_snapshot():
    controller, now = _velocity_debug_trial()
    now, _ = _velocity_debug_spike(controller, now, ack=False)
    with pytest.raises(ValueError, match="scalar receipt"):
        controller.record_dispatch(
            joint_name="joint3", receipt_sequence=round(now*1e6), feedback_timestamp_sec=now,
            completed_at_sec=now+.001, overlay_after_profile_nm=0., gravity_nm=6.,
            requested_total_nm=6., conditioned_total_nm=6.,
            joint_commands={"gravity_nm": (0., 2., 6., 1., 0., 0.), "overlay_nm": ZERO,
                            "total_nm": (0., 2., 6.01, 1., 0., 0.)},
        )
    assert not controller.status()["quantized_velocity_debug"]["zero_dispatch_confirmed"]


@pytest.mark.parametrize("extra", [
    ["--torque-console-staged-assist"], ["--torque-console-motion-profile", "unused.json"],
    ["--rate-hz", "20"], ["--torque-console-joint", "joint4"],
    ["--torque-console-max-joint-velocity-rad-s", "0.1"],
    ["--feedback-source", "parameter"],
])
def test_debug_cli_rejects_unsupported_modes_before_any_can_connection(tmp_path, extra):
    args = owner._parser().parse_args([
        "--allow-motion", "--torque-console", "--rate-hz", "100",
        "--torque-console-confirmation", "B601_RS_TORQUE_CONSOLE",
        "--torque-console-quantized-velocity-debug", *extra,
    ])
    command = owner.build_probe_command(args, run_dir=tmp_path)
    parsed = probe.build_parser().parse_args(command[2:])
    with pytest.raises((ValueError, probe.LowGainProbeError), match="debug|direct"):
        probe._validate_torque_console_cli(parsed)


def test_debug_launcher_forwards_explicit_flag_and_raw_limits(tmp_path):
    args = owner._parser().parse_args([
        "--allow-motion", "--torque-console", "--rate-hz", "100",
        "--torque-console-confirmation", "B601_RS_TORQUE_CONSOLE",
        "--torque-console-quantized-velocity-debug",
        "--debug-dynamics-trace",
    ])
    parsed = probe.build_parser().parse_args(owner.build_probe_command(args, run_dir=tmp_path)[2:])
    probe._validate_torque_console_cli(parsed)
    controller = probe._direct_torque_console_from_args(parsed)
    assert controller.status()["quantized_velocity_debug"]["confirmation_window_sec"] == .08
    assert controller.limits.max_joint_velocity_rad_s == .05
    assert parsed.debug_dynamics_trace


def _report_dispatch(
    t: float,
    *,
    joint: str = "joint3",
    desired: float = 0.7,
    sent: float = 0.2,
    q: float = 0.0,
    velocity: float = 0.0,
) -> dict[str, object]:
    index = int(joint[-1]) - 1
    qpos = [0.0] * 6
    qvel = [0.0] * 6
    assist = [0.0] * 6
    gravity = [0.0] * 6
    total = [0.0] * 6
    qpos[index], qvel[index], assist[index] = q, velocity, sent
    gravity[index], total[index] = 6.0, 6.0 + sent
    return {
        "kind": "active_dispatch",
        "recorded_monotonic_sec": t,
        "torque_console": {
            "phase": "armed",
            "joint_name": joint,
            "desired_assist_nm": desired,
            "anchor_rad": 0.0,
        },
        "qpos_rad": qpos,
        "qvel_fd_rad_s": qvel,
        "overlay_effective_tau_nm": assist,
        "gravity_tau_nm": gravity,
        "conditioned_tau_nm": total,
        "torque_limited": False,
        "slew_limited": False,
    }


def test_offline_report_uses_confirmed_dispatch_and_sustained_motion() -> None:
    rows = [
        _report_dispatch(1.0, sent=0.1),
        _report_dispatch(1.1, sent=0.2, q=0.0012, velocity=0.012),
        _report_dispatch(1.2, sent=0.3, q=0.0016, velocity=0.004),
        _report_dispatch(1.3, desired=0.0, sent=0.0, q=0.0016),
        _report_dispatch(1.9, desired=0.0, sent=0.0, q=0.0014),
    ]
    report = summarize_trace(rows)
    assert report["source_semantics"] == "owner-confirmed software dispatch; not measured motor torque"
    assert len(report["trials"]) == 1
    trial = report["trials"][0]
    assert trial["max_abs_sent_assist_nm"] == pytest.approx(0.3)
    assert trial["sustained_motion_onset"]["sent_assist_nm"] == pytest.approx(0.2)
    assert trial["target_zero_at_sec"] == pytest.approx(1.3)
    assert trial["confirmed_zero_at_sec"] == pytest.approx(1.3)
    assert trial["post_zero_qpos_rad"] == pytest.approx(0.0014)
    assert trial["post_zero_window_complete"] is True


def test_offline_report_waits_for_confirmed_zero_after_release_slew() -> None:
    rows = [
        _report_dispatch(1.0, sent=0.3, q=0.002),
        _report_dispatch(1.1, desired=0.0, sent=0.2, q=0.003),
        _report_dispatch(1.2, desired=0.0, sent=0.1, q=0.004),
        _report_dispatch(1.3, desired=0.0, sent=0.0, q=0.005),
        _report_dispatch(1.7, desired=0.0, sent=0.0, q=0.006),
        _report_dispatch(1.8, desired=0.0, sent=0.0, q=0.007),
    ]
    trial = summarize_trace(rows)["trials"][0]
    assert trial["target_zero_at_sec"] == pytest.approx(1.1)
    assert trial["confirmed_zero_at_sec"] == pytest.approx(1.3)
    assert trial["zero_qpos_rad"] == pytest.approx(0.005)
    assert trial["post_zero_qpos_rad"] == pytest.approx(0.007)
    assert trial["post_zero_window_complete"] is True


def test_offline_report_distinguishes_nonmovement_target_change_and_guard() -> None:
    rows = [
        _report_dispatch(1.0, desired=0.7, sent=0.6, q=0.0003),
        _report_dispatch(1.1, desired=0.8, sent=0.7, q=0.0004),
        _report_dispatch(1.2, desired=0.8, sent=0.8, q=0.0004),
        _report_dispatch(1.3, desired=0.0, sent=0.7, q=0.0004),
        _report_dispatch(1.4, joint="joint6", desired=0.2, sent=0.1, q=0.002),
        {
            "kind": "torque_console_guard",
            "recorded_monotonic_sec": 1.45,
            "qpos_rad": [0, 0, 0, 0, 0, 0.005],
            "qvel_fd_rad_s": [0, 0, 0, 0, 0, 0.06],
            "fault": "joint_velocity_exceeded",
        },
    ]
    report = summarize_trace(rows)
    assert [trial["end_reason"] for trial in report["trials"]] == [
        "target_changed", "zero", "guard"
    ]
    assert all(trial["sustained_motion_onset"] is None for trial in report["trials"])
    assert report["trials"][1]["post_zero_qpos_rad"] is None
    assert report["trials"][2]["guard_qpos_rad"] == pytest.approx(0.005)
    assert report["trials"][2]["guard_velocity_rad_s"] == pytest.approx(0.06)


def test_offline_report_rejects_missing_or_nonmonotonic_evidence() -> None:
    with pytest.raises(ValueError, match="no non-zero"):
        summarize_trace([_report_dispatch(1.0, desired=0.0, sent=0.0)])
    with pytest.raises(ValueError, match="monotonic"):
        summarize_trace([_report_dispatch(2.0), _report_dispatch(1.0)])


def test_offline_report_cli_preserves_trace_and_creates_separate_result(tmp_path) -> None:
    run_dir = tmp_path / "b601-owner-test"
    run_dir.mkdir()
    trace = run_dir / "b601_owner.debug.jsonl"
    source = json.dumps(_report_dispatch(1.0), separators=(",", ":")) + "\n"
    trace.write_text(source, encoding="utf-8")
    assert report_main(["--run-dir", str(run_dir)]) == 0
    assert trace.read_text(encoding="utf-8") == source
    reports = list(tmp_path.glob("b601-torque-report-*/report.json"))
    assert len(reports) == 1
    result = json.loads(reports[0].read_text(encoding="utf-8"))
    assert result["source_trace"] == str(trace)
    assert result["trials"][0]["end_reason"] == "trace_end"


def test_offline_report_cli_fails_concisely_without_trace(tmp_path, capsys) -> None:
    assert report_main(["--run-dir", str(tmp_path)]) == 2
    assert "[B601-REPORT] ERROR:" in capsys.readouterr().err
    assert list(tmp_path.glob("b601-torque-report-*")) == []


def _controller(**overrides: object) -> B601TorqueExperiment:
    return B601TorqueExperiment(
        B601TorqueExperimentLimits(joint_name="joint3", **overrides)
    )


def test_console_assist_ramps_only_selected_joint_and_withdraws() -> None:
    controller = _controller()
    assert controller.status()["owner_ready"] is False
    controller.owner_ready()
    assert controller.status()["owner_ready"] is True
    controller.submit("arm", now_sec=1.0)
    overlay, audit = controller.step(now_sec=1.0, qpos_rad=ZERO, qvel_rad_s=ZERO)
    assert audit["phase"] == "armed"
    assert overlay == ZERO
    controller.submit("set", now_sec=1.01, torque_nm=-0.3)
    overlay, audit = controller.step(now_sec=1.11, qpos_rad=ZERO, qvel_rad_s=ZERO)
    assert overlay == pytest.approx((0, 0, -0.0165, 0, 0, 0))
    assert audit["desired_assist_nm"] == -0.3
    controller.submit("withdraw", now_sec=1.12)
    overlay, audit = controller.step(now_sec=1.21, qpos_rad=ZERO, qvel_rad_s=ZERO)
    assert overlay == ZERO
    assert audit["phase"] == "inactive"


def test_console_stale_intent_withdraws_without_old_target_rebound() -> None:
    controller = _controller()
    controller.owner_ready()
    controller.submit("arm", now_sec=1.0)
    controller.step(now_sec=1.0, qpos_rad=ZERO, qvel_rad_s=ZERO)
    controller.submit("set", now_sec=1.01, torque_nm=0.3)
    controller.step(now_sec=1.11, qpos_rad=ZERO, qvel_rad_s=ZERO)
    overlay, audit = controller.step(now_sec=1.50, qpos_rad=ZERO, qvel_rad_s=ZERO)
    assert overlay == ZERO
    assert audit["phase"] == "inactive"
    assert audit["fault"] == "gui_intent_stale"
    with pytest.raises(ValueError, match="armed"):
        controller.submit("set", now_sec=1.51, torque_nm=0.3)


def test_console_reversal_releases_old_sign_before_new_sign_rises() -> None:
    controller = _controller(command_timeout_sec=2.0)
    controller.owner_ready()
    controller.submit("arm", now_sec=1.0)
    controller.step(now_sec=1.0, qpos_rad=ZERO, qvel_rad_s=ZERO)
    controller.submit("set", now_sec=1.01, torque_nm=0.3)
    positive, _ = controller.step(now_sec=2.0, qpos_rad=ZERO, qvel_rad_s=ZERO)
    assert positive[2] == pytest.approx(0.15)
    controller.submit("set", now_sec=2.01, torque_nm=-0.3)
    released, _ = controller.step(now_sec=2.10, qpos_rad=ZERO, qvel_rad_s=ZERO)
    assert released[2] == 0.0
    negative, _ = controller.step(now_sec=2.20, qpos_rad=ZERO, qvel_rad_s=ZERO)
    assert negative[2] == pytest.approx(-0.015)


def test_console_rejects_nonstationary_arm_and_unbounded_or_nan_input() -> None:
    controller = _controller()
    controller.owner_ready()
    controller.submit("arm", now_sec=1.0)
    moving = (0.0, 0.0, 0.021, 0.0, 0.0, 0.0)
    _, audit = controller.step(now_sec=1.0, qpos_rad=ZERO, qvel_rad_s=moving)
    assert audit["phase"] == "inactive"
    assert audit["fault"] == "arm_rejected_not_stationary"
    with pytest.raises(ValueError, match="armed"):
        controller.submit("set", now_sec=1.01, torque_nm=0.1)
    controller.submit("arm", now_sec=1.1)
    controller.step(now_sec=1.1, qpos_rad=ZERO, qvel_rad_s=ZERO)
    with pytest.raises(ValueError, match="limit"):
        controller.submit("set", now_sec=1.11, torque_nm=0.31)
    with pytest.raises(ValueError, match="finite"):
        controller.submit("set", now_sec=1.11, torque_nm=float("nan"))


def test_console_excursion_fault_latches_and_releases_torque() -> None:
    controller = _controller(command_timeout_sec=2.0)
    controller.owner_ready()
    controller.submit("arm", now_sec=1.0)
    controller.step(now_sec=1.0, qpos_rad=ZERO, qvel_rad_s=ZERO)
    controller.submit("set", now_sec=1.01, torque_nm=0.3)
    controller.step(now_sec=1.5, qpos_rad=ZERO, qvel_rad_s=ZERO)
    moved = (0.0, 0.0, 0.151, 0.0, 0.0, 0.0)
    overlay, audit = controller.step(now_sec=1.6, qpos_rad=moved, qvel_rad_s=ZERO)
    assert audit["phase"] == "fault"
    assert "joint_excursion_exceeded: joint3 delta=+0.151000rad" in audit["fault"]
    assert overlay == ZERO
    controller.submit("withdraw", now_sec=1.7)
    with pytest.raises(ValueError, match="inactive"):
        controller.submit("arm", now_sec=1.8)


def test_console_velocity_guard_withdraws_before_excursion_limit() -> None:
    controller = _controller(
        command_timeout_sec=2.0, max_abs_assist_nm=1.0, max_excursion_rad=0.1
    )
    controller.owner_ready()
    anchor = (0.0, 0.0, 0.293, 0.0, 0.0, 0.0)
    controller.submit("arm", now_sec=1.0)
    controller.step(now_sec=1.0, qpos_rad=anchor, qvel_rad_s=ZERO)
    controller.submit("set", now_sec=1.01, torque_nm=1.0)
    controller.step(now_sec=1.05, qpos_rad=anchor, qvel_rad_s=ZERO)
    moved = (0.0, 0.0, 0.29875, 0.0, 0.0, 0.0)
    velocity = (0.0, 0.0, 0.06135, 0.0, 0.0, 0.0)
    overlay, audit = controller.step(
        now_sec=1.10, qpos_rad=moved, qvel_rad_s=velocity
    )
    assert audit["phase"] == "fault"
    assert audit["displacement_rad"] == pytest.approx(0.00575)
    assert audit["fault"] == (
        "joint_velocity_exceeded: joint3 abs_qvel=0.061350rad/s "
        "limit=0.050000rad/s"
    )
    assert overlay == ZERO
    assert controller.status()["effective_assist_nm"] == 0.0


def test_console_velocity_guard_ignores_passive_motion_and_other_joint_velocity() -> None:
    controller = _controller(command_timeout_sec=2.0)
    controller.owner_ready()
    controller.submit("arm", now_sec=1.0)
    controller.step(now_sec=1.0, qpos_rad=ZERO, qvel_rad_s=ZERO)
    passive, audit = controller.step(
        now_sec=1.05,
        qpos_rad=(0.0, 0.0, 0.001, 0.0, 0.0, 0.0),
        qvel_rad_s=(0.0, 0.0, 0.06, 0.0, 0.0, 0.0),
    )
    assert audit["phase"] == "armed"
    assert passive == ZERO
    controller.submit("set", now_sec=1.06, torque_nm=0.2)
    _, audit = controller.step(
        now_sec=1.10,
        qpos_rad=(0.0, 0.0, 0.002, 0.001, 0.0, 0.0),
        qvel_rad_s=(0.0, 0.0, 0.023, 0.06, 0.0, 0.0),
    )
    assert audit["phase"] == "armed"
    assert audit["fault"] is None


def test_console_velocity_guard_must_exceed_arming_stationarity() -> None:
    with pytest.raises(ValueError, match="must exceed arming stationarity"):
        _controller(max_joint_velocity_rad_s=0.02)


def test_console_records_only_confirmed_software_dispatch() -> None:
    controller = _controller()
    controller.owner_ready()
    controller.submit("arm", now_sec=1.0)
    controller.step(now_sec=1.0, qpos_rad=ZERO, qvel_rad_s=ZERO)
    controller.submit("set", now_sec=1.01, torque_nm=0.20)
    controller.step(now_sec=1.11, qpos_rad=ZERO, qvel_rad_s=ZERO)
    assert controller.status()["last_dispatch"] is None
    controller.record_dispatch(
        joint_name="joint3",
        receipt_sequence=44,
        feedback_timestamp_sec=1.11,
        overlay_after_profile_nm=0.015,
        gravity_nm=-0.11,
        requested_total_nm=-0.095,
        conditioned_total_nm=-0.090,
    )
    status = controller.status()
    assert status["last_dispatch"]["overlay_after_profile_nm"] == 0.015
    assert status["last_dispatch"]["dispatch_count"] == 1
    assert status["last_dispatch"]["receipt_sequence"] == 44
    assert status["history"][-1]["conditioned_total_nm"] == -0.090


def test_console_owner_stop_removes_readiness_and_assist() -> None:
    controller = _controller()
    controller.owner_ready()
    controller.submit("arm", now_sec=1.0)
    controller.step(now_sec=1.0, qpos_rad=ZERO, qvel_rad_s=ZERO)
    controller.submit("set", now_sec=1.01, torque_nm=-0.2)
    controller.step(now_sec=1.11, qpos_rad=ZERO, qvel_rad_s=ZERO)
    controller.owner_stopped(now_sec=1.12, reason="operator_interrupt")
    status = controller.status()
    assert status["phase"] == "fault"
    assert status["owner_ready"] is False
    assert status["effective_assist_nm"] == 0.0


def test_console_joint_selection_is_zero_only_and_dispatch_receipt_is_not_misattributed() -> None:
    controller = _controller()
    controller.owner_ready()
    controller.submit("select_joint", now_sec=1.0, joint_name="joint2")
    assert controller.status()["joint_name"] == "joint2"
    controller.submit("arm", now_sec=1.01)
    controller.step(now_sec=1.01, qpos_rad=ZERO, qvel_rad_s=ZERO)
    controller.submit("set", now_sec=1.02, torque_nm=0.2)
    with pytest.raises(ValueError, match="inactive"):
        controller.submit("select_joint", now_sec=1.03, joint_name="joint4")
    overlay, audit = controller.step(now_sec=1.12, qpos_rad=ZERO, qvel_rad_s=ZERO)
    assert audit["joint_name"] == "joint2"
    assert overlay[1] > 0
    assert sum(abs(value) for value in overlay[2:]) == 0
    controller.submit("withdraw", now_sec=1.13)
    controller.step(now_sec=1.23, qpos_rad=ZERO, qvel_rad_s=ZERO)
    controller.submit("select_joint", now_sec=1.24, joint_name="joint4")
    controller.record_dispatch(
        joint_name="joint2",
        receipt_sequence=45,
        feedback_timestamp_sec=1.12,
        overlay_after_profile_nm=0.015,
        gravity_nm=0.0,
        requested_total_nm=0.015,
        conditioned_total_nm=0.015,
    )
    assert controller.status()["last_dispatch"] is None
    assert controller.status()["history"] == []


def test_console_support_drift_names_joint_and_keeps_operator_events() -> None:
    controller = _controller(command_timeout_sec=2.0)
    controller.owner_ready()
    controller.submit("arm", now_sec=1.0)
    controller.step(now_sec=1.0, qpos_rad=ZERO, qvel_rad_s=ZERO)
    controller.submit("set", now_sec=1.01, torque_nm=-0.02)
    for index in range(150):
        controller.submit("heartbeat", now_sec=1.02 + index * 0.001)
    drifted = (0.0, 0.0, 0.0, -0.016108, 0.0, 0.0)
    overlay, audit = controller.step(now_sec=1.20, qpos_rad=drifted, qvel_rad_s=ZERO)
    assert overlay == ZERO
    assert "joint4 delta=-0.016108rad" in audit["fault"]
    assert [event["action"] for event in controller.status()["events"]] == ["arm", "armed", "set", "guard_fault"]
    assert controller.status()["last_observed_support_drift_rad"]["joint4"] == -0.016108


def test_console_owner_forwards_parameters_and_enables_trace_by_default(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = owner._parser().parse_args(
        [
            "--allow-motion",
            "--torque-console",
            "--torque-console-joint",
            "joint3",
            "--torque-console-max-assist-nm",
            "0.25",
            "--torque-console-max-joint-velocity-rad-s",
            "0.075",
            "--torque-console-confirmation",
            "B601_RS_TORQUE_CONSOLE",
        ]
    )
    assert args.debug_dynamics_trace is False
    config = tmp_path / "rebotarm_rs.yaml"
    config.write_text("fixture: offline\n")
    source = tmp_path / "vendor"
    source.mkdir()
    motorbridge_root = tmp_path / "motorbridge"
    (motorbridge_root / "bindings/python/src/motorbridge").mkdir(parents=True)
    (motorbridge_root / "target/release").mkdir(parents=True)
    (motorbridge_root / "target/release/libmotor_abi.so").write_bytes(b"offline fixture")
    seen: list[str] = []
    seen_environment: dict[str, str] = {}

    def fake_foreground(command, **kwargs) -> int:
        seen.extend(command)
        seen_environment.update(kwargs["environment_overrides"])
        return 0

    monkeypatch.setattr(owner, "_run_foreground", fake_foreground)
    assert owner.main(
        [
            "--allow-motion",
            "--torque-console",
            "--torque-console-joint",
            "joint3",
            "--torque-console-max-assist-nm",
            "0.25",
            "--torque-console-max-joint-velocity-rad-s",
            "0.075",
            "--torque-console-confirmation",
            "B601_RS_TORQUE_CONSOLE",
            "--config",
            str(config),
            "--rebot-source-root",
            str(source),
            "--motorbridge-root",
            str(motorbridge_root),
            "--output-root",
            str(tmp_path),
        ]
    ) == 0
    command = seen
    parsed = probe.build_parser().parse_args(command[2:])
    assert parsed.torque_console is True
    assert parsed.torque_console_joint == "joint3"
    assert parsed.torque_console_max_assist_nm == 0.25
    assert parsed.torque_console_max_joint_velocity_rad_s == 0.075
    assert parsed.debug_dynamics_trace is True
    assert seen_environment["MOTORBRIDGE_LIB"] == str(
        motorbridge_root / "target/release/libmotor_abi.so"
    )
    probe._validate_torque_console_cli(parsed)
    parsed.debug_dynamics_trace = False
    with pytest.raises(probe.LowGainProbeError, match="dynamics trace"):
        probe.run_probe(parsed, torque_console=_controller())


@pytest.mark.parametrize("option", ["--no-gravity-debug-mode", "--without-driver-torque-limits"])
def test_console_rejects_missing_protective_envelope(option: str) -> None:
    with pytest.raises(SystemExit) as error:
        owner.main(
            [
                "--allow-motion",
                "--torque-console",
                "--torque-console-confirmation",
                "B601_RS_TORQUE_CONSOLE",
                option,
            ]
        )
    assert error.value.code == 2


def test_console_loopback_intents_require_origin_and_random_token() -> None:
    controller = _controller()
    server = B601TorqueConsoleServer(controller)
    server.start()
    try:
        with urlopen(server.url + "/", timeout=2) as response:
            page = response.read().decode()
        token = page.split('name="csrf-token" content="', 1)[1].split('"', 1)[0]
        with urlopen(server.url + "/app.js", timeout=2) as response:
            app = response.read().decode()
        assert 'confirm_cycle_full_disable' in app
        assert 'disable_confirmation_available' in app
        assert len(token) >= 32
        intent_url = server.url + "/api/intent"
        body = json.dumps({"action": "arm"}).encode()
        with pytest.raises(HTTPError) as invalid:
            urlopen(Request(intent_url, data=body, headers={"Content-Type": "application/json"}), timeout=2)
        assert invalid.value.code == 403
        controller.owner_ready()
        request = Request(
            intent_url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Origin": server.url,
                "X-B601-Console-Token": token,
            },
        )
        with urlopen(request, timeout=2) as response:
            assert json.load(response)["status"] == "accepted"
        assert controller.status()["phase"] == "arm_requested"
        controller.step(now_sec=1.0, qpos_rad=ZERO, qvel_rad_s=ZERO)
        with urlopen(server.url + "/api/status", timeout=2) as response:
            status = json.load(response)
        assert status["phase"] == "armed"
        assert status["owner_ready"] is True
    finally:
        server.close()


def test_console_withdraw_source_and_cycle_visibility_are_validated_and_forwarded():
    class Controller:
        def __init__(self):
            self.calls = []

        def status(self):
            return {"phase": "inactive"}

        def submit(self, action, **kwargs):
            self.calls.append((action, kwargs))

    controller = Controller()
    server = B601TorqueConsoleServer(controller)
    server.start()
    try:
        with urlopen(server.url + "/", timeout=2) as response:
            page = response.read().decode()
        token = page.split('name="csrf-token" content="', 1)[1].split('"', 1)[0]

        def send(payload):
            request = Request(
                server.url + "/api/intent",
                data=json.dumps(payload).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Origin": server.url,
                    "X-B601-Console-Token": token,
                },
            )
            return urlopen(request, timeout=2)

        with send({"action": "withdraw", "source": "page_unload"}) as response:
            assert json.load(response)["status"] == "accepted"
        with send({"action": "visibility", "visible": False}) as response:
            assert json.load(response)["status"] == "accepted"
        assert controller.calls[0][0] == "withdraw"
        assert controller.calls[0][1]["withdraw_source"] == "page_unload"
        assert controller.calls[1][0] == "visibility"
        assert controller.calls[1][1]["visible"] is False
        assert isinstance(controller.calls[1][1]["now_sec"], float)
        with send({"action": "arm", "operation_mode": "workspace", "return_and_disable": True}) as response:
            assert json.load(response)["status"] == "accepted"
        assert controller.calls[-1][1]["return_and_disable"] is True
        for bad in (1, "true", None):
            with pytest.raises(HTTPError) as invalid_grant:
                send({"action": "arm", "operation_mode": "workspace", "return_and_disable": bad})
            assert invalid_grant.value.code == 400
        assert len(controller.calls) == 3
        with send({"action": "return_to_origin"}) as response:
            assert json.load(response)["status"] == "accepted"
        assert controller.calls[-1][0] == "return_to_origin"
        with pytest.raises(HTTPError) as invalid_return:
            send({"action": "return_to_origin", "target_offset_rad": 1.0})
        assert invalid_return.value.code == 400
        with pytest.raises(HTTPError) as invalid:
            send({"action": "withdraw", "source": "arbitrary"})
        assert invalid.value.code == 400
    finally:
        server.close()


def test_console_shutdown_controls_use_owner_callback_only() -> None:
    controller = _controller()
    server = B601TorqueConsoleServer(controller)
    server.start()
    try:
        with urlopen(server.url + "/", timeout=2) as response:
            page = response.read().decode()
        token = page.split('name="csrf-token" content="', 1)[1].split('"', 1)[0]

        def request(action: str) -> Request:
            return Request(
                server.url + "/api/stop",
                data=json.dumps({"action": action}).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Origin": server.url,
                    "X-B601-Console-Token": token,
                },
            )

        with pytest.raises(HTTPError) as unavailable:
            urlopen(request("full_disable"), timeout=2)
        assert unavailable.value.code == 400
        called: list[str] = []
        server.set_stop_handler(called.append)
        with urlopen(server.url + "/api/status", timeout=2) as response:
            assert json.load(response)["stop_controls_available"] is True
        for action in ("protective_hold", "full_disable", "confirm_cycle_full_disable"):
            with urlopen(request(action), timeout=2) as response:
                assert json.load(response)["status"] == "accepted"
        assert called == ["protective_hold", "full_disable", "confirm_cycle_full_disable"]
        with pytest.raises(HTTPError) as invalid:
            urlopen(request("stop_motor_directly"), timeout=2)
        assert invalid.value.code == 400
    finally:
        server.close()


def test_console_runtime_notice_tail_is_bounded_and_operator_only(tmp_path) -> None:
    runtime = tmp_path / "runtime.log"
    runtime.write_text(
        "internal details\n[B601] STARTUP ACTIVE\n[B601] [TORQUE CONSOLE READY]\n"
        "[B601] support_joint_drift_exceeded: joint4\n",
        encoding="utf-8",
    )
    assert recent_runtime_notices(runtime, limit=2) == [
        "[B601] [TORQUE CONSOLE READY]",
        "[B601] support_joint_drift_exceeded: joint4",
    ]


def test_console_stop_request_reuses_owner_flags_without_double_click_escalation() -> None:
    stop = {"requested": False, "full_disable_requested": False}
    probe._console_stop_request(stop, "protective_hold")
    probe._console_stop_request(stop, "protective_hold")
    assert stop == {"requested": True, "full_disable_requested": False}
    probe._console_stop_request(stop, "full_disable")
    assert stop == {"requested": True, "full_disable_requested": True}
    with pytest.raises(ValueError, match="unsupported"):
        probe._console_stop_request(stop, "disable_joint3")
