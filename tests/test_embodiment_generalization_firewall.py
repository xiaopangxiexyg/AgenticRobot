"""Prevent new embodiment-v1 assumptions from leaking into shared control code.

This is intentionally an allowlist firewall, not a claim that RobotAction v1 is
already embodiment-neutral.  The listed files are the documented compatibility
debt for GD-ACTION-V1-001/GD-ARBITER-CORE-001/GD-SAFETY-V1-001.  Removing an
entry is always allowed; adding one requires an explicit architecture review.
"""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED_CONTROL_ROOTS = (
    REPO_ROOT / "robot_motion/action",
    REPO_ROOT / "robot_motion/runtime",
)


def _python_files(root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    )


def _absolute_imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
    return modules


def test_shared_action_and_runtime_do_not_import_device_or_frontend_packages() -> None:
    forbidden_roots = {
        "hardware",
        "sim",
        "teleop_runtime",
        "deployments",
        "rm65_inspire_tac",
    }
    violations: dict[str, list[str]] = {}
    for root in SHARED_CONTROL_ROOTS:
        for path in _python_files(root):
            imported_roots = {
                module.split(".", 1)[0]
                for module in _absolute_imported_modules(path)
            }
            forbidden = sorted(imported_roots & forbidden_roots)
            if forbidden:
                violations[str(path.relative_to(REPO_ROOT))] = forbidden
    assert violations == {}


def test_robot_action_v1_assumptions_do_not_spread_to_new_shared_files() -> None:
    markers = (
        "RM65_ARM_DOF",
        "VALID_SIDES",
        "HAND16_SLOT_DOF",
        "INSPIRE_HAND_DOF",
        "split_effector_name",
        '"left_arm"',
        '"right_arm"',
    )
    allowed_v1_surfaces = {
        "robot_motion/action/__init__.py",
        "robot_motion/action/arbiter.py",
        "robot_motion/action/arm_spec.py",
        "robot_motion/action/codec.py",
        "robot_motion/action/composition.py",
        "robot_motion/action/execution_manifest.py",
        "robot_motion/action/execution_target.py",
        "robot_motion/action/executor.py",
        "robot_motion/action/hand_spec.py",
        # Explicit bimanual-ee-pose-v1 compatibility transform.  It remains
        # left/right-coupled until the second-embodiment EG2/EG3 batch; keeping
        # it visible here prevents the exception from becoming an implicit
        # claim of payload-neutrality.
        "robot_motion/action/pose_transform.py",
        "robot_motion/action/schema.py",
        "robot_motion/action/vector_pipeline.py",
    }
    observed = {
        str(path.relative_to(REPO_ROOT))
        for root in SHARED_CONTROL_ROOTS
        for path in _python_files(root)
        if any(marker in path.read_text(encoding="utf-8") for marker in markers)
    }
    assert observed <= allowed_v1_surfaces, sorted(observed - allowed_v1_surfaces)


def test_vendor_named_shared_contracts_remain_explicit_v1_exceptions() -> None:
    vendor_markers = ("RM65", "Inspire", "DMTAC")
    allowed_vendor_named_surfaces = {
        "robot_motion/action/hand_mapping.py",
        "robot_motion/action/schema.py",
        "robot_motion/action/vector_pipeline.py",
        "robot_motion/runtime/__init__.py",
        "robot_motion/runtime/safety_operation.py",
    }
    observed = {
        str(path.relative_to(REPO_ROOT))
        for root in SHARED_CONTROL_ROOTS
        for path in _python_files(root)
        if any(marker in path.read_text(encoding="utf-8") for marker in vendor_markers)
    }
    assert observed <= allowed_vendor_named_surfaces, sorted(
        observed - allowed_vendor_named_surfaces
    )


def test_arbiter_keeps_its_single_documented_v1_name_projection_seam() -> None:
    path = REPO_ROOT / "robot_motion/action/arbiter.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    call_owners: list[str] = []
    for function in (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)):
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "split_effector_name"
            for node in ast.walk(function)
        ):
            call_owners.append(function.name)
    assert sorted(call_owners) == ["_copy_selected_component"]
