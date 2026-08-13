from __future__ import annotations

import importlib
from pathlib import Path


def test_legacy_position_control_tcp_envelope_is_identity_compatible() -> None:
    canonical = importlib.import_module(
        "robot_motion.target_safety.tcp_pose_envelope"
    )
    legacy = importlib.import_module(
        "robot_motion.position_control.tcp_pose_envelope"
    )

    assert legacy.__all__ == canonical.__all__
    for name in canonical.__all__:
        assert getattr(legacy, name) is getattr(canonical, name)


def test_target_safety_has_no_runtime_or_backend_reverse_dependency() -> None:
    package_root = Path(__file__).resolve().parents[1] / "robot_motion" / "target_safety"
    forbidden = (
        "control_composition",
        "control_station",
        "hardware",
        "sim",
        "teleop_runtime",
    )

    for path in package_root.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for module in forbidden:
            assert f"import {module}" not in source
            assert f"from {module}" not in source


def test_legacy_runtime_tcp_envelope_adapters_are_identity_compatible() -> None:
    module_pairs = (
        (
            "teleop_runtime.target_safety.tcp_pose_envelope_runtime",
            "teleop_runtime.position_control.tcp_pose_envelope_runtime",
        ),
        (
            "teleop_runtime.target_safety.deploy_tcp_pose_envelope",
            "teleop_runtime.position_control.deploy_tcp_pose_envelope",
        ),
    )

    for canonical_name, legacy_name in module_pairs:
        canonical = importlib.import_module(canonical_name)
        legacy = importlib.import_module(legacy_name)
        assert legacy.__all__ == canonical.__all__
        for name in canonical.__all__:
            assert getattr(legacy, name) is getattr(canonical, name)


def test_production_and_sim_callers_use_canonical_target_safety_import() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    core_callers = (
        "teleop_runtime/target_safety/tcp_pose_envelope_runtime.py",
        "teleop_runtime/target_safety/deploy_tcp_pose_envelope.py",
        "sim/mujoco/adapters/tcp_pose_envelope.py",
        "sim/sim_real_tests/mujoco/tcp_pose_envelope_closed_loop.py",
    )

    for relative_path in core_callers:
        source = (repository_root / relative_path).read_text(encoding="utf-8")
        assert "from robot_motion.target_safety.tcp_pose_envelope import" in source
        assert "robot_motion.position_control.tcp_pose_envelope" not in source

    runtime_callers = (
        "teleop_runtime/real_teleop/component_builder.py",
        "teleop_runtime/real_teleop/control_loop.py",
        "teleop_runtime/deploy/run_policy.py",
    )
    for relative_path in runtime_callers:
        source = (repository_root / relative_path).read_text(encoding="utf-8")
        assert ".target_safety." in source
        assert ".position_control.tcp_pose_envelope_runtime" not in source
        assert ".position_control.deploy_tcp_pose_envelope" not in source


def test_mujoco_target_safety_adapter_has_no_real_runtime_dependency() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    adapter_path = repository_root / "sim/mujoco/adapters/tcp_pose_envelope.py"
    source = adapter_path.read_text(encoding="utf-8")

    for module in ("control_station", "hardware", "teleop_runtime"):
        assert f"import {module}" not in source
        assert f"from {module}" not in source

    scenario_source = (
        repository_root
        / "sim/sim_real_tests/mujoco/tcp_pose_envelope_closed_loop.py"
    ).read_text(encoding="utf-8")
    assert "from sim.mujoco.adapters.tcp_pose_envelope import" in scenario_source
    assert "class _ActiveWorldTCPPoseEnvelopeFilter" not in scenario_source
