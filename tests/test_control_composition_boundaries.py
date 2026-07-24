from __future__ import annotations

import ast
from pathlib import Path

import control_composition
import pytest
import robot_motion.action as robot_action
import robot_motion.deploy as robot_deploy
from control_composition import (
    CompositionClock,
    CompositionProposalAdmitter,
    CompositionTraceEmitter,
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
from control_station.composition_session import (
    CompositionSessionController,
    CompositionSessionOperationSpec as HistoricalCompositionSessionOperationSpec,
    CompositionSessionRearmReceipt as HistoricalCompositionSessionRearmReceipt,
    CompositionSessionRearmSpec as HistoricalCompositionSessionRearmSpec,
    CompositionSessionState as HistoricalCompositionSessionState,
    CompositionSessionTransition as HistoricalCompositionSessionTransition,
    advance_composition_session as historical_advance_composition_session,
    initial_composition_session_state as historical_initial_composition_session_state,
    validate_composition_session_rearm as historical_validate_composition_session_rearm,
)
from control_station.handoff import (
    MixedControlBoundaryPolicy as HistoricalMixedControlBoundaryPolicy,
)
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
    assert control_composition.CompositionTraceEmitter is CompositionTraceEmitter
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


def test_composition_policy_node_is_not_exported_by_robot_motion() -> None:
    assert PolicyNodeOutput.__module__ == "control_composition.policy_node"
    assert PolicyNodeSpec.__module__ == "control_composition.policy_node"
    assert not hasattr(robot_deploy, "PolicyNodeOutput")
    assert not hasattr(robot_deploy, "PolicyNodeSpec")


def test_station_handoff_path_reexports_canonical_lifecycle_identity() -> None:
    assert MixedControlBoundaryPolicy is HistoricalMixedControlBoundaryPolicy


def test_station_session_path_reexports_canonical_activation_identity() -> None:
    assert (
        CompositionSessionOperationSpec
        is HistoricalCompositionSessionOperationSpec
    )
    assert CompositionSessionState is HistoricalCompositionSessionState
    assert CompositionSessionTransition is HistoricalCompositionSessionTransition
    assert advance_composition_session is historical_advance_composition_session
    assert (
        initial_composition_session_state
        is historical_initial_composition_session_state
    )
    assert CompositionSessionRearmSpec is HistoricalCompositionSessionRearmSpec
    assert (
        CompositionSessionRearmReceipt
        is HistoricalCompositionSessionRearmReceipt
    )
    assert (
        validate_composition_session_rearm
        is historical_validate_composition_session_rearm
    )
    assert issubclass(CompositionSessionController, CompositionSessionRegistry)


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


def test_lower_contract_owners_do_not_reverse_depend_on_control_composition() -> None:
    violations: list[str] = []
    for package_name in ("robot_motion", "dataset_contracts", "dataset_interface"):
        for path in _python_files(REPO_ROOT / package_name):
            if "control_composition" in _absolute_import_roots(path):
                violations.append(str(path.relative_to(REPO_ROOT)))
    assert violations == []


def test_control_composition_is_explicitly_packaged() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"control_composition*"' in pyproject
