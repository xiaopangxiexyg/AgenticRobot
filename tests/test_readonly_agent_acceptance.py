from __future__ import annotations

import io
import json

import pytest

from scripts.probes import run_readonly_agent_acceptance as probe
from agent_gateway.profiles.real import perception_cli, readonly_tcp_cli


@pytest.mark.parametrize("startup_fails", [False, True])
@pytest.mark.parametrize("sdk_debug", [False, True])
@pytest.mark.parametrize("scope", ["all", "main"])
@pytest.mark.parametrize("finish_status,native_success", [("completed", False), ("stuck", False), ("success", False), ("completed", True)])
@pytest.mark.parametrize("cleanup_mode", ["clean", "nonzero", "stubborn", "unreaped", "cancel_poll"])
def test_fixed_acceptance_keeps_secrets_out_of_argv_and_closes_children(tmp_path, monkeypatch, startup_fails, sdk_debug, scope, finish_status, native_success, cleanup_mode):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-ambient-secret")
    children = []
    runs = []
    handlers = {}

    def install_handler(signum, handler):
        old = handlers.get(signum, probe.signal.SIG_DFL)
        handlers[signum] = handler
        return old

    monkeypatch.setattr(probe.signal, "signal", install_handler)


    class Child:
        def __init__(self, command, **options):
            if "real-perception" in command:
                selected = perception_cli.build_parser().parse_args(command[command.index("real-perception") + 1:])
                assert selected.object_localization_backend == "sam2-grounding-dino"
                assert selected.object_localization_asset_bundle == tmp_path
            elif "real-tcp-observe" in command:
                readonly_tcp_cli.build_parser().parse_args(command[command.index("real-tcp-observe") + 1:])
            if "real-perception" in command:
                artifact_root = command[command.index("--artifact-root") + 1]
                assert probe.Path(artifact_root).is_dir()
                config_path = probe.Path(command[command.index("--camera-config") + 1])
                config = json.loads(config_path.read_text())
                assert set(config) == ({"front"} if scope == "main" else {"front", "left_wrist", "right_wrist"})
                assert config["front"] == original["front"]
            self.pid = len(children) + 100
            self.returncode = None
            self.command = command
            self.options = options
            self.terminated = False
            self.killed = False
            self.cancel_injected = False
            children.append(self)

        def poll(self):
            if cleanup_mode == "cancel_poll" and not self.cancel_injected and "real-perception" in self.command:
                self.cancel_injected = True
                handlers[probe.signal.SIGTERM](probe.signal.SIGTERM, None)
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = 0 if cleanup_mode in {"clean", "cancel_poll"} else (1 if cleanup_mode == "nonzero" else None)

        def kill(self):
            self.killed = True
            if cleanup_mode != "unreaped":
                self.returncode = -9

        def wait(self, timeout):
            if self.terminated:
                if self.returncode is None:
                    raise probe.subprocess.TimeoutExpired(self.command, timeout)
                return self.returncode
            self.returncode = 0
            if "--output-dir" in self.command:
                folder = probe.Path(self.command[self.command.index("--output-dir") + 1])
                folder.mkdir()
                (folder / "transcript_readonly_perception.json").write_text(json.dumps({
                    "finish": {"_finish": True, "status": finish_status, "summary": "fixture report"},
                    "environment_success": native_success,
                }))
            return 0

    def ready(*_args, **kwargs):
        assert kwargs["required_roles"] == (("main",) if scope == "main" else ("main", "wrist_left", "wrist_right"))
        if startup_fails:
            raise RuntimeError("startup failed")
        return {"event": "gateway_ready", "robot_backend": "absent", "current_mode": "real_observe"}

    monkeypatch.setattr(probe, "wait_ready", ready)
    monkeypatch.setattr(probe.subprocess, "Popen", Child)
    monkeypatch.setattr(probe.subprocess, "run", lambda command, **options: runs.append((command, options)))
    monkeypatch.setattr(probe.subprocess, "check_output", lambda *_args, **_options: "fixture-commit\n")
    monkeypatch.setattr(probe.sys, "stdin", io.StringIO("fixture-secret\n"))
    output = tmp_path / "run"
    camera_config = tmp_path / "cameras.json"
    original = {key: {"serial": key, "width": 640, "height": 480, "fps": 15, "enable_depth": True}
        for key in ("front", "left_wrist", "right_wrist")}
    camera_config.write_text(json.dumps(original))
    argv = []
    for name in ("repo", "python", "rpent-repo", "rpent-python", "camera-config", "camera-registry", "localization-bundle"):
        argv.extend(("--" + name, str(camera_config if name == "camera-config" else tmp_path)))
    argv.extend(("--output", str(output)))
    argv.extend(("--hf-home", str(tmp_path)))
    argv.extend(("--camera-scope", scope))
    if sdk_debug:
        argv.append("--camera-sdk-debug-log")
    report_completed = finish_status == "completed" and not native_success
    assert probe.main(argv) == (1 if startup_fails or not report_completed or cleanup_mode != "clean" else 0)
    report = json.loads((output / "acceptance.json").read_text())
    assert report["cleanup_complete"] is (cleanup_mode == "clean")
    assert children[0].killed is (cleanup_mode in {"stubborn", "unreaped"})
    expected_exit = {"clean": 0, "nonzero": 1, "stubborn": -9, "unreaped": None, "cancel_poll": 0}[cleanup_mode]
    assert report["child_exit_codes"][str(children[0].pid)] == expected_exit
    assert bool(report["cleanup_errors"]) is (cleanup_mode != "clean")
    assert all(handlers[sig] == probe.signal.SIG_DFL for sig in (probe.signal.SIGINT, probe.signal.SIGTERM))
    if cleanup_mode == "cancel_poll":
        assert report["cleanup_stop_signals"] == [probe.signal.SIGTERM]
    assert children[0].terminated
    assert "real-perception" in children[0].command and "--camera-single-start" in children[0].command
    assert ("--camera-sdk-debug-log" in children[0].command) == sdk_debug
    if sdk_debug:
        assert children[0].command[children[0].command.index("--camera-sdk-debug-log") + 1] == str(output / "librealsense.debug.log")
    assert report["camera_sdk_debug_log"] == ("librealsense.debug.log" if sdk_debug else None)
    assert "OPENAI_API_KEY" not in children[0].options["env"]
    assert children[0].options["env"]["HF_HOME"] == str(tmp_path)
    assert children[0].options["env"]["HF_HUB_OFFLINE"] == "1"
    assert report["hf_home"] == str(tmp_path)
    assert report["camera_scope"] == scope
    assert json.loads(camera_config.read_text()) == original
    if not startup_fails:
        assert report["agent_finish"]["status"] == finish_status
        assert report["environment_success"] is native_success
        assert report["status"] == ("agent_run_completed_pending_evidence_review" if report_completed else "failed")
        assert children[1].options["env"]["OPENAI_API_KEY"] == "fixture-secret"
        assert "--session-extension" in children[1].command
        instruction = children[1].command[children[1].command.index("--instruction") + 1]
        assert "finish(status='completed'" in instruction
        assert "不声明物理任务成功" in instruction
    for child in children:
        assert "fixture-secret" not in " ".join(child.command)
        assert "fixture-ambient-secret" not in " ".join(child.command)
    for path in output.rglob("*"):
        if path.is_file():
            assert "fixture-secret" not in path.read_text()
            assert "fixture-ambient-secret" not in path.read_text()


def test_check_only_filters_gateway_credential_without_reading_stdin(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-ambient-secret")
    runs = []
    monkeypatch.setattr(probe.subprocess, "run", lambda command, **options: runs.append((command, options)))
    monkeypatch.setattr(probe.sys, "stdin", io.StringIO("fixture-secret\n"))
    camera_config = tmp_path / "cameras.json"
    camera_config.write_text(json.dumps({"front": {"serial": "front"}}))
    argv = []
    for name in ("repo", "python", "rpent-repo", "rpent-python", "camera-config", "camera-registry", "localization-bundle"):
        argv.extend(("--" + name, str(camera_config if name == "camera-config" else tmp_path)))
    output = tmp_path / "run"
    argv.extend(("--output", str(output), "--check-only"))

    assert probe.main(argv) == 0
    assert len(runs) == 2
    assert "real-perception" in runs[0][0]
    assert "OPENAI_API_KEY" not in runs[0][1]["env"]
    assert probe.os.environ["OPENAI_API_KEY"] == "fixture-ambient-secret"
    assert probe.sys.stdin.read() == "fixture-secret\n"
    assert not output.exists()


@pytest.mark.parametrize("required,advertised,valid", [
    (("main",), ["main"], True),
    (("main", "wrist_left", "wrist_right"), ["main"], False),
    (("main",), ["main", "wrist_left"], False),
])
def test_ready_inventory_must_match_explicit_scope(tmp_path, required, advertised, valid):
    log = tmp_path / "ready.log"
    log.write_text(json.dumps({"event": "gateway_ready", "robot_backend": "absent",
        "current_mode": "real_observe", "rgb_roles": advertised}))
    if valid:
        assert probe.wait_ready(None, log, required_roles=required)["rgb_roles"] == advertised
    else:
        with pytest.raises(RuntimeError, match="explicitly selected scope"):
            probe.wait_ready(None, log, required_roles=required)


@pytest.mark.parametrize("startup_fails", [False, True])
def test_tcp_acceptance_exposes_state_only_and_does_not_start_cameras(tmp_path, monkeypatch, startup_fails):
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-secret")
    children, preparations = [], []
    class Child:
        def __init__(self, command, **options):
            if "real-tcp-observe" in command:
                readonly_tcp_cli.build_parser().parse_args(command[command.index("real-tcp-observe") + 1:])
            self.command, self.options = command, options
            self.returncode = None
            self.pid = 100 + len(children)
            children.append(self)
        def poll(self):
            return self.returncode
        def terminate(self):
            self.returncode = 0
        def wait(self, timeout):
            self.returncode = 0
            if "--output-dir" in self.command:
                output = probe.Path(self.command[self.command.index("--output-dir")+1])
                output.mkdir()
                (output / "transcript_readonly_tcp.json").write_text(json.dumps({
                    "finish": {"_finish": True, "status": "completed"}, "environment_success": False,
                }))
            return 0
    def ready(*args, **kwargs):
        assert kwargs["tcp"] is True and kwargs["required_roles"] == ()
        if startup_fails:
            raise RuntimeError("SDK startup failure")
        return {"event": "gateway_ready"}
    monkeypatch.setattr(probe, "wait_ready", ready)
    monkeypatch.setattr(probe.subprocess, "Popen", Child)
    monkeypatch.setattr(probe.subprocess, "run", lambda cmd, **kw: preparations.append(cmd))
    monkeypatch.setattr(probe.subprocess, "check_output", lambda *a, **kw: "fixture-commit\n")
    monkeypatch.setattr(probe.sys, "stdin", io.StringIO("fixture-secret\n"))
    argv = ["--acceptance-scope", "tcp", "--output", str(tmp_path / "out")]
    for name in ("repo", "python", "rpent-repo", "rpent-python", "arm-ip-assignment", "left-tool-profile", "right-tool-profile", "envelope-profile"):
        argv.extend(("--"+name, str(tmp_path)))
    assert probe.main(argv) == int(startup_fails)
    command = children[0].command
    assert "real-tcp-observe" in command and "--camera-config" not in command
    assert "--artifact-root" not in command and "--envelope-profile" in command
    assert "OPENAI_API_KEY" not in children[0].options["env"]
    assert children[0].poll() == 0
    if not startup_fails:
        assert preparations[0][preparations[0].index("--workflow")+1] == "state"
        assert "--include-execution" not in preparations[0]
        assert children[1].options["env"]["OPENAI_API_KEY"] == "fixture-secret"
    report = json.loads((tmp_path / "out" / "acceptance.json").read_text())
    assert report["camera_scope"] == "none" and report["cleanup_complete"]


@pytest.mark.parametrize("override", [{}, {"command_owner_count": 1}, {"motion_enabled": True}, {"component_ids": ["right_arm"]}])
def test_tcp_ready_rejects_control_owner_or_incomplete_inventory(tmp_path, override):
    event = {"event": "gateway_ready", "current_mode": "real_observe", "robot_backend": "readonly_sdk_sampler",
             "command_owner_count": 0, "motion_enabled": False, "gripper_enabled": False,
             "component_ids": ["left_arm", "right_arm"], **override}
    log = tmp_path / "ready.log"
    log.write_text(json.dumps(event))
    if override:
        with pytest.raises(RuntimeError, match="TCP-only"):
            probe.wait_ready(None, log, tcp=True)
    else:
        assert probe.wait_ready(None, log, tcp=True) == event



def test_owned_children_share_cleanup_budget_and_keep_unreaped_status(monkeypatch):
    from types import SimpleNamespace

    now = [0.0]
    waits, terminated, killed = [], [], []
    monkeypatch.setattr(probe, "time", SimpleNamespace(monotonic=lambda: now[0]))

    class StuckChild:
        def __init__(self, pid):
            self.pid = pid

        def poll(self):
            return None

        def terminate(self):
            terminated.append(self.pid)

        def kill(self):
            killed.append(self.pid)

        def wait(self, timeout):
            waits.append(timeout)
            now[0] += timeout
            raise probe.subprocess.TimeoutExpired("fixture-child", timeout)

    errors, codes = probe._close_owned_children([StuckChild(101), StuckChild(102)])
    assert set(terminated) == set(killed) == {101, 102}
    assert sum(waits) <= 80.0 and now[0] <= 80.0
    assert codes == {"101": None, "102": None}
    assert {row["pid"] for row in errors} == {101, 102}
    assert all(row["error"] == "child_did_not_exit_after_KILL" for row in errors)
