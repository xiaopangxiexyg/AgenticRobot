from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _python_files(root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for path in root.rglob("*.py")
            if "__pycache__" not in path.parts
        )
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


def _called_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


def _attribute_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }


def _annotated_class_fields(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                field.target.id
                for field in node.body
                if isinstance(field, ast.AnnAssign)
                and isinstance(field.target, ast.Name)
            }
    raise AssertionError(f"Class {class_name!r} not found in {path}.")


def _call_keyword_names_in_function(
    path: Path,
    *,
    function_name: str,
    called_name: str,
) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ),
        None,
    )
    if function is None:
        raise AssertionError(f"Function {function_name!r} not found in {path}.")
    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Name)
            and node.func.id == called_name
            or isinstance(node.func, ast.Attribute)
            and node.func.attr == called_name
        )
    ]
    if len(calls) != 1:
        raise AssertionError(
            f"Expected one {called_name!r} call in {function_name!r}; "
            f"found {len(calls)}."
        )
    return {keyword.arg for keyword in calls[0].keywords if keyword.arg is not None}


def _legacy_mujoco_composition_calls(path: Path) -> set[str]:
    legacy_methods = {
        "capture_composition_context_snapshot",
        "evaluate_composition_shadow",
        "resolve_composition_context",
        "submit_composition_context",
        "submit_composition_input",
        "submit_composition_inputs",
        "submit_composition_source_eligibility",
        "submit_residual_anchor_lifecycle",
    }
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in legacy_methods:
            continue
        receiver = node.func.value
        if not (
            isinstance(receiver, ast.Attribute)
            and receiver.attr == "composition_port"
        ):
            violations.add(node.func.attr)
    return violations


def _private_mujoco_composition_topology_calls(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    calls: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        receiver = node.func.value
        if not (
            isinstance(receiver, ast.Attribute)
            and receiver.attr == "_topology"
            and isinstance(receiver.value, ast.Attribute)
            and receiver.value.attr == "_composition_binding"
        ):
            continue
        calls.add(node.func.attr)
    return calls


def _violations(
    root: Path,
    forbidden_roots: set[str],
    *,
    excluded: frozenset[Path] = frozenset(),
) -> dict[str, list[str]]:
    return {
        str(path.relative_to(REPO_ROOT)): sorted(
            _absolute_import_roots(path) & forbidden_roots
        )
        for path in _python_files(root)
        if path not in excluded and _absolute_import_roots(path) & forbidden_roots
    }


def test_collection_frontend_does_not_absorb_deploy_or_composition() -> None:
    assert _violations(
        REPO_ROOT / "teleop_runtime" / "collection",
        {"control_composition", "deployments"},
    ) == {}


def test_general_real_deploy_does_not_absorb_composition_or_simulation() -> None:
    assert _violations(
        REPO_ROOT / "teleop_runtime" / "deploy",
        {"collection_runtime", "control_composition", "control_station", "sim"},
    ) == {}


def test_model_integrations_do_not_own_application_or_backend_runtime() -> None:
    compatibility_entrypoint = (
        REPO_ROOT / "deployments" / "openpi" / "client" / "real.py"
    )
    assert _violations(
        REPO_ROOT / "deployments",
        {
            "collection_runtime",
            "control_station",
            "hardware",
            "sim",
            "teleop_runtime",
        },
        excluded=frozenset({compatibility_entrypoint}),
    ) == {}

    tree = ast.parse(
        compatibility_entrypoint.read_text(encoding="utf-8"),
        filename=str(compatibility_entrypoint),
    )
    assert not any(
        isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        for node in ast.walk(tree)
    )
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module
    }
    assert imported_modules == {"teleop_runtime.deploy.openpi_application"}


def test_shared_deploy_and_composition_cores_remain_backend_neutral() -> None:
    assert _violations(
        REPO_ROOT / "robot_motion" / "deploy",
        {
            "collection_runtime",
            "control_composition",
            "control_station",
            "deployments",
            "hardware",
            "sim",
            "teleop_runtime",
        },
    ) == {}
    assert _violations(
        REPO_ROOT / "control_composition",
        {
            "collection_runtime",
            "control_station",
            "dataset_interface",
            "deployments",
            "hardware",
            "sim",
            "teleop_runtime",
        },
    ) == {}


def test_simulation_does_not_reverse_depend_on_real_or_application_packages() -> None:
    assert _violations(
        REPO_ROOT / "sim",
        {
            "collection_runtime",
            "control_station",
            "deployments",
            "hardware",
            "teleop_runtime",
        },
    ) == {}


def test_optional_sim_deploy_integrations_do_not_spread_into_simulation_core() -> None:
    integration_root = REPO_ROOT / "sim" / "mujoco" / "deploy"
    integration_files = frozenset(_python_files(integration_root))
    assert _violations(
        REPO_ROOT / "sim",
        {"control_composition"},
        excluded=integration_files,
    ) == {}


def test_sim_hil_strategy_control_construction_is_application_owned() -> None:
    sim_target = REPO_ROOT / "sim" / "mujoco" / "deploy" / "sim_hil_target.py"
    station_composition = (
        REPO_ROOT / "control_station" / "composition" / "mujoco_sim_hil.py"
    )
    station_factory = (
        REPO_ROOT / "control_station" / "sim_hil_profile" / "factory.py"
    )
    strategy_control_constructors = {
        "CompositionStrategyControl",
        "TimeWindowRetentionSpec",
        "build_initial_strategy_revision",
    }

    assert "control_composition" not in _absolute_import_roots(sim_target)
    assert _called_names(sim_target).isdisjoint(strategy_control_constructors)
    assert strategy_control_constructors <= _called_names(station_composition)
    assert "build_mujoco_sim_hil_strategy_control_factory" in _called_names(
        station_factory
    )


def test_optional_composition_settings_are_not_general_target_config() -> None:
    target_runtime = REPO_ROOT / "sim" / "mujoco" / "deploy" / "runtime.py"
    sim_target = REPO_ROOT / "sim" / "mujoco" / "deploy" / "sim_hil_target.py"
    composition_binding = (
        REPO_ROOT / "sim" / "mujoco" / "deploy" / "composition_binding.py"
    )
    station_composition = (
        REPO_ROOT / "control_station" / "composition" / "mujoco_sim_hil.py"
    )
    station_factory = (
        REPO_ROOT / "control_station" / "sim_hil_profile" / "factory.py"
    )

    assert _annotated_class_fields(
        target_runtime,
        "MujocoDeployConfig",
    ).isdisjoint(
        {
            "composition_information_plan",
            "enable_composition_source_eligibility",
        }
    )
    assert {
        "enable_source_eligibility",
        "information_plan",
    } <= _annotated_class_fields(
        composition_binding,
        "MujocoCompositionRuntimeConfig",
    )
    assert "MujocoCompositionRuntimeConfig" not in _called_names(sim_target)
    assert "MujocoCompositionRuntimeConfig" in _called_names(
        station_composition
    )
    assert "build_mujoco_sim_hil_composition_runtime_config" in _called_names(
        station_factory
    )


def test_general_mujoco_target_does_not_opt_into_composition_runtime() -> None:
    sim_target = REPO_ROOT / "sim" / "mujoco" / "deploy" / "sim_hil_target.py"

    sim_hil_runtime_keywords = _call_keyword_names_in_function(
        sim_target,
        function_name="build_mujoco_sim_hil_target",
        called_name="MujocoDeployRuntime",
    )
    agent_skill_runtime_keywords = _call_keyword_names_in_function(
        sim_target,
        function_name="build_mujoco_agent_skill_motion_target",
        called_name="MujocoDeployRuntime",
    )

    assert "composition_runtime_config" in sim_hil_runtime_keywords
    assert "composition_runtime_config" not in agent_skill_runtime_keywords


def test_mujoco_target_runtime_delegates_composition_runtime_construction() -> None:
    target_runtime = REPO_ROOT / "sim" / "mujoco" / "deploy" / "runtime.py"
    composition_binding = (
        REPO_ROOT / "sim" / "mujoco" / "deploy" / "composition_binding.py"
    )
    implementation_constructors = {
        "CompositionControlTraceEmitter",
        "ControlCompositionRuntime",
        "TimeWindowRetentionSpec",
    }

    assert _called_names(target_runtime).isdisjoint(implementation_constructors)
    assert implementation_constructors <= _called_names(composition_binding)

    runtime_tree = ast.parse(
        target_runtime.read_text(encoding="utf-8"),
        filename=str(target_runtime),
    )
    imported_symbols = {
        alias.name
        for node in ast.walk(runtime_tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "build_mujoco_composition_runtime_binding" in imported_symbols
    assert imported_symbols.isdisjoint(implementation_constructors)


def test_mujoco_target_runtime_delegates_pure_promotion_policy() -> None:
    target_runtime = REPO_ROOT / "sim" / "mujoco" / "deploy" / "runtime.py"
    promotion_policy = (
        REPO_ROOT / "sim" / "mujoco" / "deploy" / "composition_promotion.py"
    )

    assert "MujocoCompositionPromotionPolicy" in _called_names(target_runtime)
    assert _absolute_import_roots(promotion_policy) == {
        "__future__",
        "dataclasses",
        "robot_motion",
    }

    stateful_owner_operations = {
        "admit_evaluated_composition",
        "admit_proposal",
        "run_decision",
        "send_commands",
    }
    assert _attribute_names(promotion_policy).isdisjoint(
        stateful_owner_operations
    )
    assert stateful_owner_operations & _attribute_names(target_runtime) == {
        "admit_evaluated_composition",
        "admit_proposal",
        "run_decision",
        "send_commands",
    }
    runtime_tree = ast.parse(
        target_runtime.read_text(encoding="utf-8"),
        filename=str(target_runtime),
    )
    assert any(
        isinstance(node, ast.FunctionDef) and node.name == "step"
        for node in ast.walk(runtime_tree)
    )
    policy_source = promotion_policy.read_text(encoding="utf-8")
    runtime_source = target_runtime.read_text(encoding="utf-8")
    assert "_next_composite_proposal_sequence" not in policy_source
    assert "_composition_promotion_envelope_by_evaluation_id" not in policy_source
    assert "_next_composite_proposal_sequence" in runtime_source
    assert "_composition_promotion_envelope_by_evaluation_id" in runtime_source


def test_mujoco_runtime_uses_private_composition_topology_only_for_admission() -> None:
    target_runtime = REPO_ROOT / "sim" / "mujoco" / "deploy" / "runtime.py"

    assert _private_mujoco_composition_topology_calls(target_runtime) == {
        "admit_evaluated_composition"
    }
    assert "evaluate_composition_shadow" in _called_names(target_runtime)


def test_control_station_uses_explicit_mujoco_composition_port() -> None:
    legacy_runtime_views = {
        "_topology",
        "composition_strategy_control",
        "policy_topology",
    }
    violations = {
        str(path.relative_to(REPO_ROOT)): sorted(
            _attribute_names(path) & legacy_runtime_views
        )
        for path in _python_files(REPO_ROOT / "control_station")
        if _attribute_names(path) & legacy_runtime_views
    }

    assert violations == {}

    legacy_call_violations = {
        str(path.relative_to(REPO_ROOT)): sorted(
            _legacy_mujoco_composition_calls(path)
        )
        for path in _python_files(REPO_ROOT / "control_station")
        if _legacy_mujoco_composition_calls(path)
    }
    assert legacy_call_violations == {}
