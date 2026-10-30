from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from robot_motion.rm65 import (
    RM65_BIMANUAL_POSITION_SAMPLE_ALIGNED_JERK_V1,
    load_ik_tuning_config,
    with_rm65_position_ik_objective,
)
from teleop_runtime.interfaces import EffectorBackend
from teleop_runtime.skill_deploy import rm65_profile as profile_module
from teleop_runtime.skill_deploy.rm65_lower_assembly import (
    assemble_exclusive_rm65_skill_lower,
)
from teleop_runtime.skill_deploy.rm65_profile import (
    RM65_SKILL_CANONICAL_STATION_FRAME_ID_V1,
    ExclusiveRM65SkillDeploymentProfile,
    resolve_exclusive_rm65_skill_deployment_profile,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _profile() -> ExclusiveRM65SkillDeploymentProfile:
    return ExclusiveRM65SkillDeploymentProfile(
        arm_ip_assignment_path=REPO_ROOT / "README.md",
        tool_profile_path_by_side={
            "left": (
                REPO_ROOT
                / "hardware/rm65/utils/configs/calibrations/"
                "rm65_left_inspire.json"
            ),
            "right": (
                REPO_ROOT
                / "hardware/rm65/utils/configs/calibrations/"
                "rm65_right_inspire.json"
            ),
        },
        envelope_profile_path=(
            REPO_ROOT
            / "teleop_runtime/configs/skill_deploy/"
            "agent_primitive_left_relative_tight_v1.json"
        ),
    )


def _resolved(monkeypatch):
    monkeypatch.setattr(
        profile_module,
        "load_assignment",
        lambda path: {
            "assignments": {
                "left": {"arm_ip": "192.0.2.10"},
                "right": {"arm_ip": "192.0.2.11"},
            }
        },
    )
    return resolve_exclusive_rm65_skill_deployment_profile(_profile())


class _Backend(EffectorBackend):
    def __init__(self, config) -> None:
        self.config = config
        self.starts = 0

    @property
    def name(self) -> str:
        return f"{self.config.side}_arm"

    def start(self) -> None:
        self.starts += 1

    def send_command(self, command):
        raise AssertionError("assembly must not dispatch")


class _TargetBuilder:
    def __init__(self) -> None:
        self.pre_ik_target_safety_filter = None

    def set_current_qpos_pair(
        self,
        *,
        left_qpos,
        right_qpos,
        reset_tracking=True,
    ) -> None:
        raise AssertionError("assembly must not synchronize runtime state")

    def tool_tcp_world_target_from_solver_world_target(
        self,
        *,
        side,
        solver_world_target,
    ):
        assert side in {"left", "right"}
        return np.asarray(solver_world_target, dtype=float).copy()

    def tool_tcp_world_targets_for_qpos_pair(
        self,
        *,
        left_qpos,
        right_qpos,
    ):
        raise AssertionError("assembly must not read runtime state")


class _ForceGuard:
    motion_ready = False
    fault = None

    def __init__(self) -> None:
        self.starts = 0

    def start(self, *, arm_stale_watchdog=True):
        self.starts += 1
        return {"motion_ready": False}

    def close(self) -> None:
        return None


def test_lower_assembly_reuses_one_profile_and_starts_no_resources(
    monkeypatch,
) -> None:
    profile = _resolved(monkeypatch)
    backend_configs = []
    target_configs = []
    guard_inputs = []
    target_builder = _TargetBuilder()
    force_guard = _ForceGuard()

    def backend_factory(config):
        backend_configs.append(config)
        return _Backend(config)

    def target_builder_factory(config):
        target_configs.append(config)
        return target_builder

    def force_guard_factory(backends):
        guard_inputs.append(dict(backends))
        return force_guard

    assembly = assemble_exclusive_rm65_skill_lower(
        profile=profile,
        backend_factory=backend_factory,
        target_builder_factory=target_builder_factory,
        force_guard_factory=force_guard_factory,
    )

    assert len(target_configs) == 1
    target_config = target_configs[0]
    assert target_config.scene_profile == profile.collision_scene_profile
    assert target_config.solver_frequency == profile.position_profile.rates.ik_hz
    assert target_config.inactive_side_hold_enabled is True
    assert target_config.inactive_side_hold_execution_cap_enabled is True
    assert target_config.inactive_side_hold_execution_release_enabled is False
    assert target_config.output_command_type == "joint_vector"
    assert target_config.debug_profile is False
    assert target_config.debug_profile_print_every == 0

    assert [config.side for config in backend_configs] == ["left", "right"]
    for config in backend_configs:
        command = profile.position_profile.command_execution
        assert config.arm_ip == profile.arm_ip_by_side[config.side]
        assert config.connect_hardware is True
        assert config.background_streaming_enabled is True
        assert config.background_streaming_hz == (
            profile.position_profile.rates.command_hz
        )
        assert config.background_streaming_no_catch_up_after_send is True
        assert config.background_streaming_hz * config.background_streaming_warning_ratio == pytest.approx(80.)
        assert config.background_streaming_low_rate_warning_only is True
        assert config.position_command_planner == command.planner
        assert config.tool_profile == str(
            profile.tool_profile_path_by_side[config.side]
        )
        np.testing.assert_allclose(
            config.tcp_pose_flange_m_rad,
            profile.tool_tcp_pose_flange_m_rad_by_side[config.side],
        )

    assert assembly.target_builder is target_builder
    assert assembly.position_force_guard is force_guard
    assert assembly.ports.target_builder is target_builder
    assert assembly.target_safety.target_builder is target_builder
    assert assembly.target_safety.target_frame_binding is (
        assembly.target_safety.envelope_filter.target_frame_binding
    )
    assert set(
        assembly.target_safety.target_frame_binding.target_world_from_base_by_side
    ) == {"left", "right"}
    assert set(
        assembly.target_safety.target_frame_binding.reference_frame_by_side.values()
    ) == {RM65_SKILL_CANONICAL_STATION_FRAME_ID_V1}
    np.testing.assert_allclose(
        assembly.target_safety.target_frame_binding.target_world_from_solver_world,
        np.linalg.inv(profile.scene_world_from_canonical_station),
    )
    assert assembly.startup_admission.ports is assembly.ports
    assert assembly.startup_admission.target_safety is assembly.target_safety
    assert guard_inputs == [assembly.arm_backends]
    assert all(backend.starts == 0 for backend in assembly.arm_backends.values())
    assert force_guard.starts == 0


def test_lower_assembly_enables_bounded_motion_debug_without_stdout(
    monkeypatch,
) -> None:
    profile = _resolved(monkeypatch)
    target_configs = []

    assemble_exclusive_rm65_skill_lower(
        profile=profile,
        debug_motion_skill=True,
        backend_factory=lambda config: _Backend(config),
        target_builder_factory=lambda config: (
            target_configs.append(config) or _TargetBuilder()
        ),
        force_guard_factory=lambda backends: _ForceGuard(),
    )

    assert len(target_configs) == 1
    assert target_configs[0].debug_profile is True
    assert target_configs[0].debug_profile_print_every == 0


def test_lower_assembly_rejects_non_boolean_motion_debug(monkeypatch) -> None:
    with pytest.raises(TypeError, match="debug_motion_skill must be bool"):
        assemble_exclusive_rm65_skill_lower(
            profile=_resolved(monkeypatch),
            debug_motion_skill=1,
        )


def test_lower_assembly_rejects_unreviewed_policy_source_clock(
    monkeypatch,
) -> None:
    profile = _resolved(monkeypatch)
    profile = replace(
        profile,
        position_profile=RM65_BIMANUAL_POSITION_SAMPLE_ALIGNED_JERK_V1,
    )

    with pytest.raises(ValueError, match="Skill-source timing review"):
        assemble_exclusive_rm65_skill_lower(
            profile=profile,
            backend_factory=lambda config: _Backend(config),
            target_builder_factory=lambda config: _TargetBuilder(),
            force_guard_factory=lambda backends: _ForceGuard(),
        )


def test_shared_position_ik_objective_is_immutable_and_exact() -> None:
    original = load_ik_tuning_config()

    bound = with_rm65_position_ik_objective(
        original,
        position_weight=2.0,
        orientation_weight=0.75,
        mode_name="test_position_objective",
    )

    assert bound is not original
    assert original.ik_objective_mode != "test_position_objective"
    assert "test_position_objective" not in original.objective_modes
    assert bound.ik_objective_mode == "test_position_objective"
    objective = bound.objective_weights()
    assert objective.single_arm_position_weight == 2.0
    assert objective.single_arm_orientation_weight == 0.75
    assert objective.dual_arm_position_cost == 2.0
    assert objective.dual_arm_orientation_cost == 0.75


@pytest.mark.parametrize(
    ("position_weight", "orientation_weight"),
    ((-1.0, 1.0), (1.0, float("inf"))),
)
def test_shared_position_ik_objective_rejects_unsafe_weights(
    position_weight,
    orientation_weight,
) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        with_rm65_position_ik_objective(
            load_ik_tuning_config(),
            position_weight=position_weight,
            orientation_weight=orientation_weight,
        )


def test_lower_assembly_import_has_no_teleop_or_composition_subject() -> None:
    script = """
import json
import sys
import teleop_runtime.skill_deploy.rm65_lower_assembly
print(json.dumps(sorted(
    name for name in sys.modules
    if name == 'control_composition'
    or name.startswith('control_composition.')
    or name == 'teleop_runtime.real_teleop'
    or name.startswith('teleop_runtime.real_teleop.')
    or name == 'teleop_runtime.sources'
    or name.startswith('teleop_runtime.sources.')
    or name == 'teleop_runtime.collection'
    or name.startswith('teleop_runtime.collection.')
)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []


@pytest.mark.parametrize("source", ["sdk", "udp"])
def test_lower_feedback_transport_is_explicit_and_constructs_no_socket(monkeypatch, source):
    def no_socket(*args, **kwargs):
        raise AssertionError("assembly construction must not bind/open a UDP socket")
    monkeypatch.setattr("hardware.rm65.realtime_feedback.socket.socket", no_socket)
    assembly = assemble_exclusive_rm65_skill_lower(profile=_resolved(monkeypatch),
        feedback_source=source, udp_port=8090, backend_factory=_Backend,
        target_builder_factory=lambda config: _TargetBuilder(),
        force_guard_factory=lambda backends: _ForceGuard())
    assert (assembly.ports.udp_feedback is not None) == (source == "udp")
    for backend in assembly.arm_backends.values():
        assert (backend.config.udp_feedback_reader is not None) == (source == "udp")
        assert backend.starts == 0
