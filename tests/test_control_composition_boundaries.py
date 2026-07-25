from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

import control_composition
import pytest
import robot_motion.action as robot_action
import robot_motion.deploy as robot_deploy
from control_composition import (
    CompositionClock,
    CompositionEventEmitter,
    CompositionInformationPlan,
    CompositionProposalAdmitter,
)
from control_composition.evaluation import (
    CompositionDiagnostic,
    CompositionEvaluationRecord,
)
from control_composition.composition import (
    CompositionSpec,
    PoseMixingStrategySpec,
)
from control_composition.observability import (
    GuidanceFeedbackSpec,
    SourceDifferenceSpec,
)
from control_composition.runtime import (
    CompositeProposalSubmission,
    ControlCompositionRuntime,
)
from dataset_interface.composition_control_trace import (
    project_composition_session_spec,
)
from control_composition.policy_node import PolicyNodeOutput, PolicyNodeSpec
from control_composition.lifecycle import (
    CompositionSessionOperationSpec,
    CompositionSessionRearmReceipt,
    CompositionSessionRearmSpec,
    CompositionSessionRegistry,
    CompositionSessionState,
    CompositionSessionTransition,
    MixedControlBoundaryPolicy,
    advance_composition_session,
    initial_composition_session_state,
    validate_composition_session_rearm,
)
from control_station.composition_session import CompositionSessionController
REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "control_composition"
pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _python_files(root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    )


def _absolute_import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def test_control_composition_public_port_imports_are_side_effect_free() -> None:
    assert control_composition.CompositionClock is CompositionClock
    assert control_composition.CompositionProposalAdmitter is CompositionProposalAdmitter
    assert control_composition.CompositionEventEmitter is CompositionEventEmitter
    assert (
        control_composition.CompositionInformationPlan
        is CompositionInformationPlan
    )
    assert CompositionDiagnostic.__module__ == "control_composition.evaluation"
    assert (
        CompositionEvaluationRecord.__module__
        == "control_composition.evaluation"
    )


def test_observability_contracts_are_physically_owned_by_control_composition() -> None:
    assert SourceDifferenceSpec.__module__ == (
        "control_composition.observability.difference"
    )
    assert GuidanceFeedbackSpec.__module__ == (
        "control_composition.observability.feedback"
    )


def test_composition_contracts_are_physically_owned_by_control_composition() -> None:
    assert CompositionSpec.__module__ == "control_composition.composition.core"
    assert PoseMixingStrategySpec.__module__ == (
        "control_composition.composition.strategies"
    )
    for name in (
        "CompositionSpec",
        "PoseMixingStrategySpec",
        "SourceDifferenceSpec",
        "GuidanceFeedbackSpec",
    ):
        assert not hasattr(robot_action, name)


def test_old_composition_implementation_paths_are_removed() -> None:
    for relative_path in (
        "robot_motion/action/composition.py",
        "robot_motion/action/mixing_strategy.py",
        "robot_motion/action/source_difference.py",
        "robot_motion/action/guidance_feedback.py",
        "robot_motion/deploy/policy_node.py",
    ):
        assert not (REPO_ROOT / relative_path).exists()


def test_runtime_is_owned_only_by_control_composition() -> None:
    assert ControlCompositionRuntime.__module__ == "control_composition.runtime"
    assert CompositeProposalSubmission.__module__ == "control_composition.runtime"
    assert not (REPO_ROOT / "robot_motion/deploy/topology_runtime.py").exists()


def test_formal_runtime_does_not_depend_on_legacy_shadow_orchestration() -> None:
    source = (PACKAGE_ROOT / "runtime.py").read_text(encoding="utf-8")
    for legacy_name in (
        "PoseMixingInputSnapshot",
        "PoseMixingStrategy",
        "build_pose_mixing_input_snapshot",
        "evaluate_component_mixing_shadow",
    ):
        assert legacy_name not in source


def test_composition_policy_node_is_not_exported_by_robot_motion() -> None:
    assert PolicyNodeOutput.__module__ == "control_composition.policy_node"
    assert PolicyNodeSpec.__module__ == "control_composition.policy_node"
    assert not hasattr(robot_deploy, "PolicyNodeOutput")
    assert not hasattr(robot_deploy, "PolicyNodeSpec")


def test_station_session_path_only_owns_the_application_controller() -> None:
    import control_station.composition_session as station_session

    assert issubclass(CompositionSessionController, CompositionSessionRegistry)
    assert station_session.__all__ == ["CompositionSessionController"]
    assert not (REPO_ROOT / "control_station/handoff.py").exists()


def test_session_trace_projector_is_owned_by_dataset_adapter() -> None:
    import control_station.control_trace as station_trace

    assert project_composition_session_spec.__module__ == (
        "dataset_interface.composition_control_trace"
    )
    assert station_trace.project_composition_session_spec is (
        project_composition_session_spec
    )
    assert "project_composition_session_spec" not in station_trace.__all__


def test_station_composition_package_does_not_eagerly_load_concrete_stacks() -> None:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import control_station.composition; "
                "print(sorted(name for name in "
                "('sim','hardware','teleop_runtime','deployments') "
                "if name in sys.modules))"
            ),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert probe.stdout.strip() == "[]"


def test_control_composition_does_not_import_concrete_or_application_packages() -> None:
    forbidden = {
        "control_station",
        "deployments",
        "hardware",
        "sim",
        "teleop_runtime",
    }
    violations = {
        str(path.relative_to(REPO_ROOT)): sorted(
            _absolute_import_roots(path) & forbidden
        )
        for path in _python_files(PACKAGE_ROOT)
        if _absolute_import_roots(path) & forbidden
    }
    assert violations == {}


def test_lower_motion_and_schema_owners_do_not_reverse_depend_on_composition() -> None:
    violations: list[str] = []
    for package_name in ("robot_motion", "dataset_contracts"):
        for path in _python_files(REPO_ROOT / package_name):
            if "control_composition" in _absolute_import_roots(path):
                violations.append(str(path.relative_to(REPO_ROOT)))
    assert violations == []


def test_v24_projection_is_outside_control_composition() -> None:
    assert not (PACKAGE_ROOT / "trace.py").exists()
    for path in _python_files(PACKAGE_ROOT):
        source = path.read_text(encoding="utf-8")
        assert "dataset_contracts.control_trace" not in source


def test_control_composition_is_explicitly_packaged() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"control_composition*"' in pyproject
