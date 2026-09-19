from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from control_station.composition.rebot_b601_rm65_profile import (
    load_b601_rm65_frame_mapping_profile,
)
from control_station.sim_hil_profile.rm65_b601_free_drive import (
    RM65ToB601FreeDriveSimConfig,
    build_rm65_to_b601_free_drive_bridge,
)
from control_station.sim_hil_profile.rm65_b601_free_drive_foreground import (
    run_rm65_to_b601_foreground,
)
from control_station.sources.rm65_free_drive import RM65FreeDriveStep
from robot_motion.rebot_b601_rs import (
    B601PinocchioModel,
    RM65FreeDriveSample,
)
from sim.mujoco.rebot_b601_rs import B601_RS_DEFAULT_ARM_QPOS, B601_RS_SIM_ARM_KEY

pytestmark = [pytest.mark.contract, pytest.mark.optional, pytest.mark.slow_sim]

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = REPO_ROOT / "docs/hardware/profiles/b601_rm65_mapping_150_v2.json"


def _source_root() -> Path:
    configured = os.environ.get("DEXTELEOP_REBOT_SOURCE_ROOT")
    root = Path(configured).expanduser().resolve() if configured else (
        REPO_ROOT / "hardware/reBotArm_control_py"
    )
    urdf = root / "urdf/00-arm-rs_asm-v3/urdf/00-arm-rs_asm-v3.urdf"
    if not urdf.is_file():
        pytest.skip(
            "initialize hardware/reBotArm_control_py or set "
            "DEXTELEOP_REBOT_SOURCE_ROOT to an explicit read-only source root"
        )
    return root


class _SyntheticRM65Source:
    """Deterministic RM65 sample source; it never opens a device."""

    wall_clock = staticmethod(lambda: 100.1)
    monotonic_clock = staticmethod(lambda: 200.1)

    def __init__(self, *, model, graph, profile, initial_qpos, target_qpos) -> None:
        self.model = model
        self.graph = graph
        self.profile = profile
        self.initial_qpos = np.asarray(initial_qpos, dtype=float)
        self.target_qpos = np.asarray(target_qpos, dtype=float)
        self.calls = 0
        self.rejection_at: int | None = None

    def step(self) -> RM65FreeDriveStep:
        self.calls += 1
        if self.rejection_at is not None and self.calls == self.rejection_at:
            return RM65FreeDriveStep(
                state={"synthetic": True},
                sample=None,
                rejection_reason="synthetic_receive_stale",
            )
        qpos = self.initial_qpos if self.calls == 1 else self.target_qpos
        pose = self.model.forward_kinematics(qpos)
        mapped, _ = self.graph.transform_pose_between_frames(
            pose,
            source_reference_frame_id=self.profile.source_reference_frame_id,
            source_controlled_frame_id=self.profile.source_controlled_frame_id,
            target_reference_frame_id=self.profile.target_reference_frame_id,
            target_controlled_frame_id=self.profile.target_controlled_frame_id,
        )
        sample = RM65FreeDriveSample(
            sequence=self.calls,
            source_timestamp_sec=100.0,
            receive_timestamp_sec=100.0,
            receive_timestamp_monotonic_sec=200.0,
            qpos_rad=tuple(float(value) for value in qpos),
            pose_source_reference_source_controlled=tuple(
                tuple(float(item) for item in row) for row in mapped
            ),
            source_id="synthetic-rm65-free-drive",
            source_clock_domain_id="local_wall_time",
            source_reference_frame_id="right_arm_base",
            source_controlled_frame_id="tool_tcp",
        )
        return RM65FreeDriveStep(
            state={"synthetic": True},
            sample=sample,
            rejection_reason=None,
        )


def _build_fixture():
    source_root = _source_root()
    profile = load_b601_rm65_frame_mapping_profile(PROFILE_PATH)
    model = B601PinocchioModel.from_rebot_source_root(source_root)
    initial_qpos = np.asarray(B601_RS_DEFAULT_ARM_QPOS, dtype=float)
    target_qpos = np.asarray((0.05, 1.02, 1.02, 0.03, -0.03, 0.05), dtype=float)
    source = _SyntheticRM65Source(
        model=model,
        graph=profile.build_frame_graph(),
        profile=profile,
        initial_qpos=initial_qpos,
        target_qpos=target_qpos,
    )
    bridge = build_rm65_to_b601_free_drive_bridge(
        source_root,
        source,
        profile,
        initial_qpos=initial_qpos,
        config=RM65ToB601FreeDriveSimConfig(
            translation_gain_target_reference_xyz=(1.0, 1.0, 1.0),
            rotation_gain=1.0,
            source_poll_hz=20.0,
        ),
    )
    return bridge, source, target_qpos


def test_synthetic_rm65_free_drive_reaches_b601_mujoco_target() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("pinocchio")
    bridge, source, target_qpos = _build_fixture()

    bridge.activate()
    accepted = 0
    last_tick = None
    for _ in range(2500):
        last_tick = bridge.step()
        if last_tick.retarget_result is not None and last_tick.retarget_result.accepted:
            accepted += 1

    assert last_tick is not None
    assert bridge.state == "active"
    assert source.calls >= 40
    assert accepted > 0
    assert last_tick.position_tick.driver_faulted is False
    assert last_tick.position_tick.state_within_limits is True
    np.testing.assert_allclose(
        last_tick.position_tick.measured_qpos,
        target_qpos,
        atol=5e-4,
    )


def test_synthetic_source_rejection_revokes_pending_target_in_mujoco() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("pinocchio")
    bridge, source, _target_qpos = _build_fixture()
    source.rejection_at = 3

    bridge.activate()
    rejection_tick = None
    for _ in range(200):
        tick = bridge.step()
        if tick.source_step is not None and tick.source_step.sample is None:
            rejection_tick = tick
            break

    assert rejection_tick is not None
    assert rejection_tick.state == "held"
    assert rejection_tick.position_tick.command_received is False
    assert rejection_tick.pending_command is False
    assert bridge.pending_command is None
    assert rejection_tick.position_tick.driver_faulted is False


class _ForegroundViewer:
    def __init__(self) -> None:
        self.closed = False

    def start(self):
        return {"running": True}

    def sync(self) -> bool:
        return not self.closed

    def status(self):
        return {"state": "running" if not self.closed else "closed"}

    def close(self) -> None:
        self.closed = True


class _ForegroundClock:
    def __init__(self) -> None:
        self.value = 0.0

    def now(self) -> float:
        return self.value

    def sleep(self, duration: float) -> None:
        self.value += max(float(duration), 0.0)


def test_synthetic_foreground_runner_drives_real_b601_scene_without_viewer_io() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("pinocchio")
    bridge, _source, target_qpos = _build_fixture()
    viewer = _ForegroundViewer()
    clock = _ForegroundClock()
    events: list[dict[str, object]] = []

    def poll(_timeout: float) -> str | None:
        if bridge.state == "inactive" and bridge.driver.physics_time_sec == 0.0:
            return "a"
        if bridge.driver.physics_time_sec >= 2.6:
            return "q"
        return None

    run_rm65_to_b601_foreground(
        bridge,
        viewer=viewer,
        rate_hz=1000.0,
        poll_command=poll,
        record=events.append,
        clock=clock.now,
        sleeper=clock.sleep,
    )

    assert viewer.closed is True
    assert bridge.state == "inactive"
    assert any(event["event"] == "foreground_started" for event in events)
    assert events[-1]["event"] == "foreground_stopped"
    np.testing.assert_allclose(
        bridge.driver.scene.get_arm_qpos(B601_RS_SIM_ARM_KEY),
        target_qpos,
        atol=5e-4,
    )
