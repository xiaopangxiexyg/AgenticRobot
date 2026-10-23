"""Record-only transparency workflow: offline fake state, never CAN or motors."""

from __future__ import annotations

import copy
import json
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from control_station.b601_torque_console.report import main as report_main
from control_station.b601_torque_console.server import B601TorqueConsoleServer
from control_station.b601_torque_console.transparency import TransparencyAssessment
from control_station.b601_torque_console.transparency_report import summarize_transparency


def _state(**overrides):
    return {"owner_ready": True, "phase": "inactive", "operator_armed": False,
            "joint_name": "joint3", "qpos_rad": [0.0] * 6,
            "qvel_fd_rad_s": [0.0] * 6, "feedback_age_sec": 0.01,
            "desired_assist_nm": 0.0, "effective_assist_nm": 0.0, "fault": None,
            **overrides}


def _begin(**overrides):
    return {"action": "begin", "kind": "background_drag", "pair_label": "pose-01",
            "condition_label": "current-gravity", "axis": "base_x",
            "direction": "positive", "entry_direction": "positive",
            "contact": "moving", "note": "", **overrides}


def _finish(**overrides):
    return {"action": "finish", "perceived_strength": "unknown",
            "perceived_direction": "unknown", "manual_force_n": None,
            "force_instrument": "", "force_observation": "none", "note": "", **overrides}


def _events(*, kind="background_drag", contact="moving", end="finish"):
    return [
        {"schema": "b601.transparency-annotation.v1", "sequence": 1, "trial_id": 1,
         "event": "begin", "recorded_monotonic_sec": 1.0,
         "metadata": {key: value for key, value in _begin(kind=kind, contact=contact).items() if key != "action"},
         "owner_snapshot": _state()},
        {"schema": "b601.transparency-annotation.v1", "sequence": 2, "trial_id": 1,
         "event": end, "recorded_monotonic_sec": 1.2,
         "observation": {key: value for key, value in _finish().items() if key != "action"}},
    ]


def _dispatch(t, *, assist=0.0, phase="inactive", q=0.0, **overrides):
    return {"kind": "active_dispatch", "recorded_monotonic_sec": t,
            "qpos_rad": [0, 0, q, 0, 0, 0], "qvel_fd_rad_s": [0, 0, 0.01, 0, 0, 0],
            "overlay_effective_tau_nm": [0, 0, assist, 0, 0, 0],
            "gravity_tau_nm": [0, 6.5, 5.0, 0, 0, 0],
            "conditioned_tau_nm": [0, 6.5, 5.0 + assist, 0, 0, 0],
            "torque_console": {"phase": phase, "joint_name": "joint3", "fault": None},
            "torque_limited": False, "slew_limited": False, **overrides}


def test_journal_is_durable_exclusive_and_never_mutates_snapshot(tmp_path):
    path = tmp_path / "trials.jsonl"
    journal = TransparencyAssessment(path)
    state = _state()
    before = copy.deepcopy(state)
    try:
        journal.record(_begin(), state)
        with pytest.raises(FileExistsError):
            TransparencyAssessment(path)
        saved = [json.loads(line) for line in path.read_text().splitlines()]
        assert saved[0]["metadata"]["kind"] == "background_drag"
        assert saved[0]["owner_snapshot"]["qpos_rad"] == [0] * 6
        journal.record({"action": "contact", "contact": "holding"}, state)
        journal.record(_finish(), state)
        assert state == before
        saved = [json.loads(line) for line in path.read_text().splitlines()]
        assert [row["event"] for row in saved] == ["begin", "contact", "finish"]
        assert saved[-1]["observation"]["manual_force_n"] is None
        assert saved[-1]["observation"]["force_source"] == "not_measured"
        assert journal.status()["control_authority"] is False
        assert journal.status()["active"] is None
    finally:
        journal.close()


def test_close_marks_unfinished_trial_interrupted(tmp_path):
    path = tmp_path / "trials.jsonl"
    journal = TransparencyAssessment(path)
    journal.record(_begin(), _state())
    journal.close()
    journal.close()
    assert json.loads(path.read_text().splitlines()[-1])["event"] == "interrupted"


@pytest.mark.parametrize("change", [
    {"phase": "armed"}, {"operator_armed": True}, {"effective_assist_nm": 0.01},
    {"desired_assist_nm": 0.02}, {"owner_ready": False}, {"fault": "guard"},
    {"feedback_age_sec": 0.2}, {"feedback_age_sec": None}, {"feedback_age_sec": float("nan")},
    {"qpos_rad": [0.0] * 5}, {"joint_name": "joint7"},
])
def test_background_requires_fresh_unarmed_zero_assist(tmp_path, change):
    journal = TransparencyAssessment(tmp_path / "trials.jsonl")
    try:
        with pytest.raises(ValueError):
            journal.record(_begin(), _state(**change))
        assert journal.status()["sequence"] == 0
    finally:
        journal.close()


@pytest.mark.parametrize("change", [
    {"kind": "automatic_friction_compensation"}, {"pair_label": ""},
    {"condition_label": "a" * 81}, {"direction": 1}, {"contact": "detected"},
    {"torque_nm": 0.8},
])
def test_metadata_is_bounded_and_cannot_carry_control_intents(tmp_path, change):
    journal = TransparencyAssessment(tmp_path / "trials.jsonl")
    try:
        with pytest.raises(ValueError):
            journal.record(_begin(**change), _state())
    finally:
        journal.close()


@pytest.mark.parametrize("change", [
    {"manual_force_n": True}, {"manual_force_n": float("inf")},
    {"manual_force_n": 0.2}, {"force_observation": "steady_drag"},
])
def test_force_observation_cannot_silently_invent_a_measurement(tmp_path, change):
    journal = TransparencyAssessment(tmp_path / "trials.jsonl")
    try:
        journal.record(_begin(), _state())
        with pytest.raises(ValueError):
            journal.record(_finish(**change), _state())
        assert journal.status()["active"] is not None
    finally:
        journal.close()


def test_manual_gauge_reading_has_frame_source_and_no_fake_timing(tmp_path):
    path = tmp_path / "trials.jsonl"
    journal = TransparencyAssessment(path)
    try:
        journal.record(_begin(axis="joint"), _state())
        reading = _finish(manual_force_n=-1.2, force_instrument="test gauge at handle",
                          force_observation="steady_drag")
        with pytest.raises(ValueError, match="base-frame"):
            journal.record(reading, _state())
        journal.record({"action": "abort", "note": "wrong frame"}, _state())
        journal.record(_begin(axis="base_x"), _state())
        journal.record(reading, _state())
        observation = json.loads(path.read_text().splitlines()[-1])["observation"]
        assert observation["manual_force_n"] == -1.2
        assert observation["force_source"] == "operator_entered_external_gauge"
        assert observation["force_time_synchronised"] is False
    finally:
        journal.close()


def test_persistence_failure_is_visible_and_disables_annotations_only(tmp_path, monkeypatch):
    journal = TransparencyAssessment(tmp_path / "trials.jsonl")

    def fail(_fd):
        raise OSError("disk full")

    monkeypatch.setattr("control_station.b601_torque_console.transparency.os.fsync", fail)
    try:
        with pytest.raises(OSError, match="disk full"):
            journal.record(_begin(), _state())
        assert journal.status()["available"] is False
        assert "disk full" in journal.status()["error"]
    finally:
        journal.close()


def test_annotation_api_does_not_submit_or_renew_control_and_stop_still_works(tmp_path):
    class NoIntentController:
        def status(self):
            return _state()

        def submit(self, *_args, **_kwargs):
            raise AssertionError("annotation attempted motor intent / heartbeat")

    server = B601TorqueConsoleServer(NoIntentController(), assessment_path=tmp_path / "trials.jsonl")
    server.start()
    stops = []
    server.set_stop_handler(stops.append)
    try:
        with urlopen(server.url, timeout=2) as response:
            page = response.read().decode()
        token = page.split('name="csrf-token" content="', 1)[1].split('"', 1)[0]

        def post(payload, *, endpoint="/api/assessment", authorized=True):
            headers = {"Content-Type": "application/json"}
            if authorized:
                headers.update({"Origin": server.url, "X-B601-Console-Token": token})
            return urlopen(Request(server.url + endpoint, data=json.dumps(payload).encode(), headers=headers), timeout=2)

        with pytest.raises(HTTPError) as error:
            post(_begin(), authorized=False)
        assert error.value.code == 403
        with post(_begin()) as response:
            assert json.load(response)["status"] == "accepted"
        with pytest.raises(HTTPError) as invalid_action:
            post({"action": "heartbeat"})
        assert invalid_action.value.code == 400
        with post({"action": "full_disable"}, endpoint="/api/stop") as response:
            assert response.status == 200
        assert stops == ["full_disable"]
        with urlopen(server.url + "/api/status", timeout=2) as response:
            status = json.load(response)
        assert status["phase"] == "inactive"
        assert status["transparency_assessment"]["active"]["trial_id"] == 1
    finally:
        server.close()


def test_automatic_observer_records_without_page_polling_or_motor_intents(tmp_path, monkeypatch):
    class NoIntentController:
        state = _state()

        def status(self):
            return copy.deepcopy(self.state)

        def submit(self, *_args, **_kwargs):
            raise AssertionError("automatic recording attempted motor intent / heartbeat")

    observed = threading.Event()
    original_append = TransparencyAssessment._append

    def append(journal, event):
        result = original_append(journal, event)
        if event.get("event") == "context" and event.get("drive_enabled"):
            assert threading.current_thread().name == "b601-diagnostic-observer"
            observed.set()
        return result

    monkeypatch.setattr(TransparencyAssessment, "_append", append)
    controller = NoIntentController()
    path = tmp_path / "automatic.jsonl"
    server = B601TorqueConsoleServer(controller, assessment_path=path, automatic_recording=True)
    server.start()
    try:
        # No browser, GET polling, arming request or lease renewal is involved.
        server._server.assessment.record({"action": "context", "human_intervention": False}, controller.status())
        controller.state = _state(phase="armed", desired_assist_nm=0.2)
        assert observed.wait(2.0)
        assert server.assessment_status()["context"]["drive_enabled"] is True
    finally:
        server.close()
    saved = [json.loads(line) for line in path.read_text().splitlines()]
    assert [event["event"] for event in saved] == ["declaration", "context", "context", "end"]
    assert saved[-2]["owner_snapshot"]["desired_assist_nm"] == 0.2


def test_background_report_uses_motion_without_claiming_force():
    report = summarize_transparency(_events(), [_dispatch(1.02), _dispatch(1.1, q=0.002)])
    trial = report["trials"][0]
    assert trial["record_usable_for_declared_trial"] is True
    assert trial["delta_qpos_rad"][2] == 0.002
    assert trial["max_abs_overlay_command_nm"] == [0] * 6
    assert trial["hand_force_n"] is None
    assert trial["force_measurement_status"] == "not_measured"
    assert "static_friction" in report["semantics"]["not_identified"]


@pytest.mark.parametrize("row,issue", [
    (_dispatch(1.1, assist=0.02), "background_contaminated_by_assist_or_arming"),
    (_dispatch(1.1, phase="armed"), "background_contaminated_by_assist_or_arming"),
    (_dispatch(1.1, torque_limited=True), "command_conditioning_active"),
    (_dispatch(1.1, torque_console={"phase": "inactive", "joint_name": "joint4"}), "selected_joint_changed"),
    ({"kind": "torque_console_guard", "recorded_monotonic_sec": 1.1}, "owner_event:torque_console_guard"),
    ({"kind": "protective_hold_dispatch", "recorded_monotonic_sec": 1.1}, "owner_event:protective_hold_dispatch"),
    ({"kind": "active_stop", "recorded_monotonic_sec": 1.1}, "owner_event:active_stop"),
])
def test_report_excludes_contaminated_or_interrupted_baseline(row, issue):
    trial = summarize_transparency(_events(), [_dispatch(1.01), row])["trials"][0]
    assert issue in trial["issues"]
    assert trial["record_usable_for_declared_trial"] is False


def test_unloaded_response_with_human_motion_is_not_a_friction_sample():
    trial = summarize_transparency(_events(kind="unloaded_response", contact="moving"),
                                   [_dispatch(1.01, assist=0.8), _dispatch(1.1, assist=0.8, q=0.05)])["trials"][0]
    assert "not_confirmed_unloaded" in trial["issues"]
    assert trial["hand_force_n"] is None


def test_contact_change_and_unfinished_recording_are_explicit():
    events = _events()[:1]
    events.append({"schema": "b601.transparency-annotation.v1", "sequence": 2, "trial_id": 1,
                   "event": "contact", "recorded_monotonic_sec": 1.05, "contact": "none"})
    trial = summarize_transparency(events, [_dispatch(1.01), _dispatch(1.1)])["trials"][0]
    assert "unfinished" in trial["issues"]
    assert "contact_changed_split_trial_required" in trial["issues"]


def test_report_does_not_fill_missing_trace_with_zeros():
    trial = summarize_transparency(_events(), [])["trials"][0]
    assert "no_dispatch_samples" in trial["issues"]
    assert trial["delta_qpos_rad"] is None
    assert trial["max_abs_overlay_command_nm"] is None
    late = summarize_transparency(_events(), [_dispatch(1.19)])["trials"][0]
    assert "incomplete_trace_coverage" in late["issues"]


def _paired_events():
    first = _events()
    second = copy.deepcopy(first)
    for row in second:
        row["trial_id"] = 2
        row["sequence"] += 2
        row["recorded_monotonic_sec"] += 1
    second[0]["metadata"]["direction"] = "negative"
    return first + second


@pytest.mark.parametrize("change,issue", [
    ({}, None), ({"pose": 0.1}, "different_full_arm_start_pose"),
    ({"entry_direction": "unknown"}, "entry_history_unknown"),
    ({"entry_direction": "negative"}, "different_entry_history"),
    ({"contact": "holding"}, "different_contact_state"),
])
def test_pairs_require_matched_full_pose_and_declared_history(change, issue):
    events = _paired_events()
    for key, value in change.items():
        if key == "pose":
            events[2]["owner_snapshot"]["qpos_rad"][1] = value
        else:
            events[2]["metadata"][key] = value
    report = summarize_transparency(events, [_dispatch(1.01), _dispatch(1.1), _dispatch(2.01), _dispatch(2.1)])
    pair = report["opposite_direction_pairs"][0]
    assert pair["comparable_recorded_conditions"] is (issue is None)
    if issue:
        assert issue in pair["issues"]
    assert pair["friction_estimate_nm"] is None


def test_assessment_report_cli_writes_unique_offline_report(tmp_path, capsys):
    run = tmp_path / "run"
    run.mkdir()
    annotation = run / "b601_owner.transparency.jsonl"
    trace = run / "b601_owner.debug.jsonl"
    annotation.write_text("".join(json.dumps(row) + "\n" for row in _events()))
    trace.write_text("".join(json.dumps(row) + "\n" for row in [_dispatch(1.01), _dispatch(1.1)]))
    before = trace.read_bytes(), annotation.read_bytes()
    for _ in range(2):
        assert report_main(["--run-dir", str(run), "--transparency"]) == 0
    assert (trace.read_bytes(), annotation.read_bytes()) == before
    reports = list(tmp_path.glob("b601-torque-report-*/report.json"))
    assert len(reports) == 2
    report = json.loads(reports[0].read_text())
    assert report["schema"] == "b601.transparency-assessment.v1"
    assert report["source_annotations"] == str(annotation)
    assert len(capsys.readouterr().out.splitlines()) == 4


@pytest.mark.parametrize("reading", [None, -1.2])
def test_persisted_journal_roundtrips_through_report_cli(tmp_path, monkeypatch, reading):
    run = tmp_path / "run"
    run.mkdir()
    ticks = iter((1.0, 1.2))
    monkeypatch.setattr("control_station.b601_torque_console.transparency.time.monotonic",
                        lambda: next(ticks))
    annotation = run / "b601_owner.transparency.jsonl"
    journal = TransparencyAssessment(annotation)
    try:
        journal.record(_begin(), _state())
        observation = _finish()
        if reading is not None:
            observation.update(manual_force_n=reading, force_instrument="handle gauge",
                               force_observation="steady_drag")
        journal.record(observation, _state())
    finally:
        journal.close()
    trace = run / "b601_owner.debug.jsonl"
    trace.write_text("".join(json.dumps(row) + "\n" for row in [_dispatch(1.01), _dispatch(1.1)]))
    assert report_main(["--run-dir", str(run), "--transparency"]) == 0
    report_path, = tmp_path.glob("b601-torque-report-*/report.json")
    trial = json.loads(report_path.read_text())["trials"][0]
    assert trial["record_usable_for_declared_trial"] is True
    assert trial["observation"]["manual_force_n"] == reading
    assert trial["observation"]["force_source"] == (
        "not_measured" if reading is None else "operator_entered_external_gauge")
    assert trial["hand_force_n"] is None


@pytest.mark.parametrize("bad_field,value", [("metadata", None), ("owner_snapshot", {}), ("metadata", {})])
def test_report_rejects_incomplete_annotation_files(bad_field, value):
    events = _events()
    events[0][bad_field] = value
    with pytest.raises(ValueError):
        summarize_transparency(events, [])


def test_report_never_trusts_a_forged_force_source():
    events = _events()
    events[-1]["observation"]["force_source"] = "calibrated_live_sensor"
    trial = summarize_transparency(events, [_dispatch(1.01), _dispatch(1.1)])["trials"][0]
    assert trial["observation"]["force_source"] == "not_measured"


@pytest.mark.parametrize("staged", [False, True])
def test_owner_composition_records_and_closes_annotations_without_hardware(tmp_path, monkeypatch, staged):
    from scripts.hardware import rebot_b601_rs_low_gain_probe as probe
    from scripts.hardware import rebot_b601_rs_owner as owner

    opened = []
    monkeypatch.setenv("DISPLAY", "offline-test-only")
    monkeypatch.setattr("webbrowser.open_new_tab", lambda url: opened.append(url) or True)

    def offline_probe(_args, **kwargs):
        controller = kwargs["torque_console"]
        controller.owner_ready()
        controller.step(now_sec=time.monotonic(), qpos_rad=(0.0,) * 6, qvel_rad_s=(0.0,) * 6)
        with urlopen(opened[0], timeout=2) as response:
            page = response.read().decode()
        token = page.split('name="csrf-token" content="', 1)[1].split('"', 1)[0]
        with urlopen(Request(opened[0] + "/api/assessment", data=json.dumps({"action": "context", "human_intervention": False}).encode(), headers={
            "Content-Type": "application/json", "Origin": opened[0], "X-B601-Console-Token": token,
        }), timeout=2) as response:
            assert response.status == 200
        assert controller.status()["phase"] == "inactive"
        return {"status": "OK", "mode": "hold", "hardware_claim": False}

    monkeypatch.setattr(probe, "run_probe", offline_probe)
    arguments = ["--allow-motion", "--torque-console", "--gravity-debug-mode",
                 "--torque-console-confirmation", "B601_RS_TORQUE_CONSOLE", "--debug-dynamics-trace"]
    if staged:
        arguments.append("--torque-console-staged-assist")
    args = owner._parser().parse_args(arguments)
    command = owner.build_probe_command(args, run_dir=tmp_path)
    assert probe.main(command[2:]) == 0
    result = json.loads((tmp_path / "b601_owner.json").read_text())
    assert result["hardware_claim"] is False
    assert result["transparency_assessment"]["error"] is None
    assert result["transparency_assessment"]["automatic"] is True
    assert result["transparency_assessment"]["human_intervention"] is False
    assert result["debug_dynamics_trace"]["sample_count"] == 0
    events = [json.loads(row) for row in (tmp_path / "b601_owner.transparency.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "end"
    assert any(event["event"] == "context" for event in events)


def test_automatic_windows_capture_contact_drive_target_and_full_pose(tmp_path):
    path = tmp_path / "trials.jsonl"
    journal = TransparencyAssessment(path, automatic=True)
    try:
        journal.observe(_state())
        journal.observe(_state())
        assert journal.status()["sequence"] == 1
        journal.record({"action": "context", "human_intervention": False}, _state())
        assert journal.status()["context"]["human_intervention"] is False
        journal.observe(_state(phase="armed", desired_assist_nm=0.1))
        journal.observe(_state(phase="armed", desired_assist_nm=0.1, effective_assist_nm=0.05))
        assert journal.status()["context"]["context_id"] == 3
        journal.observe(_state(phase="armed", desired_assist_nm=0.2))
        journal.record({"action": "context", "human_intervention": True},
                       _state(phase="armed", desired_assist_nm=0.2))
        journal.observe(_state(joint_name="joint4"))
    finally:
        journal.close()
    events = [json.loads(row) for row in path.read_text().splitlines()]
    report = summarize_transparency(events, [])
    assert report["schema"] == "b601.joint-diagnostics.v1"
    windows = report["trials"]
    assert len(windows) == 6
    assert windows[0]["metadata"]["human_intervention"] is None
    assert windows[2]["metadata"] == {"kind": "joint_diagnostics", "human_intervention": False, "drive_enabled": True}
    assert windows[4]["metadata"]["human_intervention"] is True
    assert windows[5]["joint_name"] == "joint4"
    assert windows[-1]["end_reason"] == "owner_closed"
    assert report["opposite_direction_pairs"] == []


@pytest.mark.parametrize("value", [None, "false", 0, 1])
def test_automatic_context_does_not_guess_human_intervention(tmp_path, value):
    journal = TransparencyAssessment(tmp_path / "trials.jsonl", automatic=True)
    try:
        with pytest.raises(ValueError):
            journal.record({"action": "context", "human_intervention": value}, _state())
        assert journal.status()["human_intervention"] is None
        assert journal.status()["sequence"] == 0
    finally:
        journal.close()


@pytest.mark.parametrize("fresh", [None, "true", 1])
def test_automatic_report_rejects_invalid_freshness_flag(tmp_path, fresh):
    path = tmp_path / "trials.jsonl"
    journal = TransparencyAssessment(path, automatic=True)
    journal.record({"action": "context", "human_intervention": False}, _state())
    journal.close()
    events = [json.loads(row) for row in path.read_text().splitlines()]
    events[1]["feedback_fresh"] = fresh
    with pytest.raises(ValueError, match="feedback_fresh"):
        summarize_transparency(events, [])


def test_automatic_report_separates_manual_motion_gravity_drift_and_drive_response(tmp_path, monkeypatch):
    ticks = iter((1.0, 1.01, 1.3, 1.31, 1.6, 1.8))
    monkeypatch.setattr("control_station.b601_torque_console.transparency.time.monotonic", lambda: next(ticks))
    path = tmp_path / "trials.jsonl"
    journal = TransparencyAssessment(path, automatic=True)
    journal.record({"action": "context", "human_intervention": True}, _state())
    journal.record({"action": "context", "human_intervention": False}, _state())
    journal.observe(_state(phase="armed", desired_assist_nm=0.1))
    journal.close()
    events = [json.loads(row) for row in path.read_text().splitlines()]
    samples = [_dispatch(1.05, q=0.0), _dispatch(1.15, q=0.01), _dispatch(1.25, q=0.01),
               _dispatch(1.35, q=0.0), _dispatch(1.45, q=0.01), _dispatch(1.55, q=0.01),
               _dispatch(1.65, phase="armed", assist=0.1), _dispatch(1.75, phase="armed", assist=0.1)]
    report = summarize_transparency(events, samples)
    observations = [row["joint_observations"][2] for row in report["trials"]]
    assert [row["response"] for row in observations] == [
        "human_force_present", "motion_without_overlay", "drive_without_resolved_motion"]
    assert all(row["friction_estimate_nm"] is None for row in observations)
    assert observations[-1]["max_abs_overlay_command_nm"] == 0.1
    assert all(row["record_usable_for_declared_trial"] for row in report["trials"])


@pytest.mark.parametrize("staged", [False, True])
def test_six_joint_receipts_are_diagnostic_copies_and_do_not_mutate_commands(staged, tmp_path):
    from robot_motion.rebot_b601_rs.torque_experiment import B601TorqueExperiment, B601TorqueExperimentLimits
    from control_station.b601_torque_console.staged_controller import B601StagedTorqueConsoleController
    from scripts.hardware import rebot_b601_rs_low_gain_probe as probe
    from scripts.hardware import rebot_b601_rs_owner as owner

    limits = B601TorqueExperimentLimits(joint_name="joint3")
    if staged:
        args = owner._parser().parse_args(["--torque-console", "--torque-console-staged-assist"])
        parsed = probe.build_parser().parse_args(owner.build_probe_command(args, run_dir=tmp_path)[2:])
        controller = B601StagedTorqueConsoleController(probe._staged_torque_console_settings_from_args(parsed))
    else:
        controller = B601TorqueExperiment(limits)
    values = {"gravity_nm": [0, 5, 6, 0, 0, 0], "overlay_nm": [0, 0, 0.1, 0, 0, 0],
              "total_nm": [0, 5, 6.1, 0, 0, 0]}
    controller.record_dispatch(joint_name="joint3", receipt_sequence=1, feedback_timestamp_sec=1,
                               overlay_after_profile_nm=0.1, gravity_nm=6, requested_total_nm=6.1,
                               conditioned_total_nm=6.1, joint_commands=values)
    values["gravity_nm"][2] = 99
    snapshot = controller.status()["last_dispatch"]["joint_commands"]
    assert snapshot["gravity_nm"][2] == 6
    assert snapshot["overlay_nm"][2] == 0.1
    assert controller.status()["phase"] == "inactive"
