from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from control_station import __main__ as dispatcher
from control_station.sim_hil_profile import rebot_b601_rm65_application as app
from teleop_runtime.action.human_residual import HumanResidualPoseAdapter
from tests.test_control_station_rebot_b601_rm65_motion import (
    _measurement,
    _Model,
    _source,
)
from tests.test_control_station_rebot_b601_rm65_motion import (
    setup_motion as motion_fixture,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]
setup_motion = motion_fixture


def _mapping(source):
    # A synthetic calibration port, not evidence for real frame alignment.
    return SimpleNamespace(
        build_residual_adapter=lambda schema: HumanResidualPoseAdapter(
            output_action_schema=schema,
            frame_graph=source.residual_adapter.frame_graph,
        ),
    )


class _Viewer:
    def __init__(self):
        self.running = True
        self.fault = False
        self.calls = []

    def start(self):
        self.calls.append("start")

    def sync(self):
        self.calls.append("sync")
        return self.running

    def status(self):
        return {"state": "faulted" if self.fault else "closed_by_user"}

    def close(self):
        self.calls.append("close")


def _session(setup_motion):
    clock, _, runtime, source, backend, _, _ = setup_motion
    return app.assemble_b601_rm65_session(
        runtime,
        backend=backend,
        model=_Model(),
        mapping_profile=_mapping(source),
        measure_target=lambda: _measurement(runtime, clock),
    )


def test_assembly_preserves_injected_owners_and_current_pose_anchor(setup_motion):
    session = _session(setup_motion)
    _, _, runtime, _, backend, _, _ = setup_motion
    assert session.state == "inactive"
    assert session.binding.runtime is runtime
    assert session.binding.source.pose_source.backend is backend
    assert session.activate().target_measurement.value_by_component.keys() == {
        "left_arm",
        "right_arm",
    }
    session.deactivate()
    assert session.anchor is None


def test_gui_observes_existing_dispatch_and_recenter(setup_motion):
    session = _session(setup_motion)
    clock, _, _, _, backend, _, _ = setup_motion
    viewer = _Viewer()
    records = []
    commands = iter(("a", None, "c", "r", None, "q"))

    def poll(timeout):
        clock.value += max(timeout, 0.05)
        backend.q[0] += 0.002
        return next(commands)

    app.run_b601_rm65_gui(
        session,
        viewer=viewer,
        rate_hz=20,
        poll_command=poll,
        record=records.append,
        emit=lambda message: None,
    )
    assert any(event.get("accepted") for event in records), records
    assert any(event.get("status") == "dispatched" for event in records)
    assert session.state == "inactive"
    assert session.anchor is None
    # Resource lifecycles belong to the application, not this injected loop.
    assert "close" not in viewer.calls


@pytest.mark.parametrize("fault", (False, True))
def test_gui_close_or_fault_cancels_without_polling_or_master_cleanup(
    setup_motion, fault
):
    session = _session(setup_motion)
    session.activate()
    viewer = _Viewer()
    viewer.running, viewer.fault = False, fault

    def forbidden(timeout):
        pytest.fail("closed GUI must not accept another operator activation")

    if fault:
        with pytest.raises(RuntimeError, match="viewer failed"):
            app.run_b601_rm65_gui(
                session, viewer=viewer, rate_hz=20, poll_command=forbidden
            )
    else:
        app.run_b601_rm65_gui(
            session,
            viewer=viewer,
            rate_hz=20,
            poll_command=forbidden,
            emit=lambda message: None,
        )
    assert session.state == "inactive"
    assert session.anchor is None


def test_lazy_dispatcher_help_never_imports_device_or_mujoco():
    script = """
import sys
from control_station.__main__ import main
try:
    main(['b601-rm65', '--help'])
except SystemExit as error:
    assert error.code == 0
assert 'hardware.rebot_b601_rs' not in sys.modules
assert 'motorbridge' not in sys.modules
assert 'mujoco' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "--connect-read-only" in result.stdout


def _args(tmp_path, mode):
    return [
        mode,
        "--task-config",
        "task.json",
        "--mapping-profile",
        "mapping.json",
        "--config",
        "rebot.yaml",
        "--rebot-source-root",
        "vendor",
        "--log",
        str(tmp_path / "command_and_output.log"),
        "--output",
        str(tmp_path / "result.json"),
    ]


@pytest.mark.parametrize(
    "option,value",
    (("--maximum-source-age-sec", "nan"), ("--maximum-rotation-rad", "0")),
)
def test_cli_rejects_invalid_bounds_before_construction(
    tmp_path, monkeypatch, option, value
):
    monkeypatch.setattr(
        app,
        "_load_mapping",
        lambda path: pytest.fail("invalid CLI must not load resources"),
    )
    with pytest.raises(SystemExit) as error:
        app.main([*_args(tmp_path, "--validate-only"), option, value])
    assert error.value.code == 2
    assert not (tmp_path / "result.json").exists()


def test_cli_can_requires_exact_read_only_confirmation(tmp_path):
    with pytest.raises(SystemExit) as error:
        app.main(_args(tmp_path, "--connect-read-only"))
    assert error.value.code == 2


@pytest.mark.parametrize(
    "mode,interrupt",
    (
        ("--validate-only", False),
        ("--connect-read-only", False),
        ("--connect-read-only", True),
    ),
)
def test_cli_lifecycle_logs_command_and_never_sends_motor_commands(
    setup_motion, tmp_path, monkeypatch, mode, interrupt
):
    import hardware.rebot_b601_rs as hardware
    from robot_motion.rebot_b601_rs import B601PinocchioModel
    from sim.mujoco.viewer import MujocoPassiveViewerSession

    clock, _, runtime, source, backend, _, _ = setup_motion
    calls = []
    monkeypatch.setattr(
        backend, "connect", lambda: calls.append("connect"), raising=False
    )
    monkeypatch.setattr(backend, "close", lambda: calls.append("close"), raising=False)
    monkeypatch.setattr(hardware, "B601RSBackend", lambda config: backend)
    monkeypatch.setattr(
        hardware.B601RSBackendConfig, "from_rs_yaml", lambda path: object()
    )
    monkeypatch.setattr(
        B601PinocchioModel, "from_rebot_source_root", lambda path: _Model()
    )
    monkeypatch.setattr(app, "_load_mapping", lambda path: _mapping(source))
    monkeypatch.setattr(
        app,
        "build_b601_rm65_sim_target",
        lambda path: (runtime, lambda: _measurement(runtime, clock), 20),
    )
    viewer = _Viewer()
    monkeypatch.setattr(
        sys.modules[MujocoPassiveViewerSession.__module__],
        "MujocoPassiveViewerSession",
        lambda *args, **kwargs: viewer,
    )

    def run(session, **kwargs):
        session.activate()
        try:
            if interrupt:
                raise KeyboardInterrupt
        finally:
            session.deactivate()

    monkeypatch.setattr(app, "run_b601_rm65_gui", run)
    args = _args(tmp_path, mode)
    if mode == "--connect-read-only":
        args += ["--confirmation", "B601_RM65_READ_ONLY_CAN_SIM_TARGET"]
    assert dispatcher.main(["b601-rm65", *args]) == (130 if interrupt else 0)
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["real_motor_commands_sent"] is False
    assert result["target_cleanup_completed"] is True
    assert calls == ([] if mode == "--validate-only" else ["connect", "close"])
    assert viewer.calls == ([] if mode == "--validate-only" else ["start", "close"])
    events = [
        json.loads(line)
        for line in (tmp_path / "command_and_output.log").read_text().splitlines()
    ]
    assert events[0]["argv"] == [
        sys.executable,
        "-m",
        "control_station",
        "b601-rm65",
        *args,
    ]
    assert events[-1]["status"] == (
        "INTERRUPTED"
        if interrupt
        else "VALIDATED"
        if mode == "--validate-only"
        else "COMPLETED"
    )


def test_missing_canonical_mapping_fails_before_sim_or_can(tmp_path, monkeypatch):
    monkeypatch.setattr(
        app,
        "_load_mapping",
        lambda path: (_ for _ in ()).throw(RuntimeError("missing calibration")),
    )
    monkeypatch.setattr(
        app,
        "build_b601_rm65_sim_target",
        lambda path: pytest.fail("must fail before target setup"),
    )
    assert app.main(_args(tmp_path, "--validate-only")) == 1
    result = json.loads((tmp_path / "result.json").read_text())
    assert "missing calibration" in result["error"]
    assert result["target_cleanup_completed"] is False


@pytest.mark.optional
def test_actual_actuated_target_atomic_pose_measurement():
    pytest.importorskip("mujoco")
    pytest.importorskip("pinocchio")
    if not Path("assets/rm65_inspire_gripper_v3").is_dir():
        pytest.skip("Reviewed RM65 assets are not bound.")
    runtime, measure, rate = app.build_b601_rm65_sim_target(
        Path("sim/configs/tasks/rm65_right_cube_lift_rl_actuated_v1.json"),
    )
    runtime.start()
    try:
        observation = measure()
        assert rate == 15
        assert set(observation.value_by_component) == {"left_arm", "right_arm"}
        assert observation.measured_at <= observation.available_at
        assert all(
            value == "cartesian_pose"
            for value in observation.representation_by_component.values()
        )
        assert runtime.rig.config.actuation_profile is not None
        source, backend = _source(runtime, runtime.clock)
        session = app.assemble_b601_rm65_session(
            runtime,
            backend=backend,
            model=_Model(),
            mapping_profile=_mapping(source),
            measure_target=measure,
        )
        session.activate()
        receipt = session.step()
        assert receipt.accepted, receipt.message
        result = runtime.step()
        assert result.status == "dispatched", (result.status, result.resolution_failure)
        session.deactivate()
    finally:
        runtime.close()
