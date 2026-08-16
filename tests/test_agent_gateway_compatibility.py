from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from agent_gateway.profiles.mujoco.motion_factory import (
    build_mujoco_agent_tcp_pose_gateway_session as canonical_session_factory,
)
from agent_gateway.profiles.mujoco.skills.cube_lift.skill import (
    CubeLiftAgentSkillLifecycle as CanonicalCubeLiftLifecycle,
)
from agent_gateway.profiles.real.observation_gateway import (
    RealTcpPoseObservationProducer as CanonicalRealObservationProducer,
)
from agent_gateway.protocol.contracts import AgentSkillRequest as CanonicalRequest
from agent_gateway.protocol.motion_schemas import (
    cardinal_tcp_step_schema_entry_v1 as CompatibleMotionSchema,
)
from agent_gateway.skills.motion.schemas import (
    cardinal_tcp_step_schema_entry_v1 as CanonicalMotionSchema,
)
from control_station.agent_gateway.protocol.contracts import (
    AgentSkillRequest as CompatibleRequest,
)
from control_station.agent_gateway.protocol.motion_schemas import (
    cardinal_tcp_step_schema_entry_v1 as HistoricMotionSchema,
)
from control_station.sim_hil_profile.agent.motion_factory import (
    build_mujoco_agent_tcp_pose_gateway_session as compatible_session_factory,
)
from control_station.sim_hil_profile.tasks.cube_lift.skill import (
    CubeLiftAgentSkillLifecycle as CompatibleCubeLiftLifecycle,
)
from control_station.real_teleop.agent_observation_gateway import (
    RealTcpPoseObservationProducer as CompatibleRealObservationProducer,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
pytestmark = [pytest.mark.unit, pytest.mark.contract]


def test_historic_imports_preserve_canonical_object_identity() -> None:
    assert CompatibleRequest is CanonicalRequest
    assert compatible_session_factory is canonical_session_factory
    assert CompatibleCubeLiftLifecycle is CanonicalCubeLiftLifecycle
    assert CompatibleRealObservationProducer is CanonicalRealObservationProducer
    assert CompatibleMotionSchema is CanonicalMotionSchema
    assert HistoricMotionSchema is CanonicalMotionSchema


def test_historic_gateway_modules_are_thin_reexports() -> None:
    compatibility_roots = (
        REPO_ROOT / "control_station/agent_gateway",
        REPO_ROOT / "control_station/sim_hil_profile/agent",
        REPO_ROOT / "control_station/tasks/cube_lift",
        REPO_ROOT / "control_station/sim_hil_profile/tasks/cube_lift",
    )
    compatibility_paths = [
        path for root in compatibility_roots for path in root.rglob("*.py")
    ]
    compatibility_paths.append(
        REPO_ROOT / "control_station/real_teleop/agent_observation_gateway.py"
    )
    for path in compatibility_paths:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        assert not any(
            isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            for node in ast.walk(tree)
        ), path
        imported_roots = {
            node.module.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert imported_roots <= {"agent_gateway"}, path


def test_former_gateway_motion_schema_owner_is_a_thin_reexport() -> None:
    path = REPO_ROOT / "agent_gateway/protocol/motion_schemas.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    assert not any(
        isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        for node in ast.walk(tree)
    )
    imported_roots = {
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert imported_roots == {"agent_gateway"}


def test_top_level_gateway_is_in_package_discovery() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"agent_gateway*"' in pyproject


@pytest.mark.parametrize(
    "module, command",
    (
        ("agent_gateway", "mujoco"),
        ("control_station", "agent-gateway-mujoco"),
    ),
)
def test_canonical_and_compatibility_cli_help(module: str, command: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", module, command, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "--socket-path" in completed.stdout
