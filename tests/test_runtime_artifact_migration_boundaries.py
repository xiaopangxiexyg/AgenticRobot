from __future__ import annotations

import ast
import io
from pathlib import Path
import tokenize


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_SOURCE_ROOTS = (
    "collection_runtime",
    "control_composition",
    "control_station",
    "dataset_contracts",
    "dataset_interface",
    "deployments",
    "hardware",
    "model",
    "rm65_inspire_tac",
    "robot_motion",
    "scripts",
    "sim",
    "teleop_runtime",
    "utils",
)

# Transitional debt budget.  Entries may be removed as readers and writers are
# migrated, but new production files must use the canonical path contract or an
# explicit, reviewed legacy-reader adapter instead of expanding this set.
LEGACY_ARTIFACT_ROOT_ALLOWLIST = frozenset(
    {
        "control_station/sim_hil.py",
        "control_station/sim_hil_profile/config.py",
        "control_station/web/calibration_web_ui.py",
        "hardware/DMTAC_gripper/examples/identify_gripper_actuator.py",
        "hardware/DMTAC_gripper/portable_dmtac_gripper/cli.py",
        "hardware/DMTAC_gripper/portable_dmtac_gripper/gripper_debug.py",
        "hardware/DMTAC_gripper/portable_dmtac_gripper/tactile_assignment_gui.py",
        "hardware/DMTAC_gripper/portable_dmtac_gripper/usb_assignment.py",
        "hardware/rm65/utils/scripts/rm65_udp_actuator_trace.py",
        "hardware/teleop_hand/check_zero_pose_calibration.py",
        "hardware/teleop_hand/q6_assignment.py",
        "hardware/VIVE/mocap/vive_diagnostic_receiver.py",
        "project_paths.py",
        "scripts/identify_bimanual_gripper_position.py",
        "teleop_runtime/collection/umi/config.py",
        "teleop_runtime/configs/rm65_arm_ips.py",
        "teleop_runtime/debug/rm65_real_debug/run_rm65_idle_worker_drain_preflight.py",
        "teleop_runtime/debug/run_vive_rm65_real_teleop.py",
        "teleop_runtime/deploy/run_policy.py",
        "teleop_runtime/episodes.py",
        "teleop_runtime/position_control/config.py",
        "teleop_runtime/real_teleop/actuator_command_trace.py",
        "teleop_runtime/real_teleop/cli.py",
        "teleop_runtime/real_teleop/debug_recorders.py",
        "teleop_runtime/real_teleop/episode_runtime.py",
        "teleop_runtime/recording.py",
        "utils/dataset_postprocess/wrist_cam_adjust/geometry.py",
        "utils/hand_eye/run_calibrate_handeye.py",
        "utils/head_eye/dual_arm.py",
        "utils/head_eye/run_calibrate_handeye.py",
        "utils/umi_hand_eye/run_calibrate.py",
    }
)

CONFIGURED_LAYOUT_CLI_FILES = {
    "control_station/sim_hil.py": "--artifact-layout",
    "hardware/DMTAC_gripper/examples/identify_gripper_actuator.py": "--artifact-layout",
    "hardware/VIVE/mocap/vive_diagnostic_receiver.py": "--artifact-layout",
    "hardware/rm65/utils/scripts/rm65_udp_actuator_trace.py": "--artifact-layout",
    "scripts/identify_bimanual_gripper_position.py": "--artifact-layout",
    "teleop_runtime/debug/rm65_real_debug/run_rm65_idle_worker_drain_preflight.py": "--artifact-layout",
    "teleop_runtime/deploy/run_policy.py": "--artifact-layout",
    "teleop_runtime/real_teleop/cli.py": "--artifact-layout",
}


def _iter_production_sources() -> list[Path]:
    sources = list(REPO_ROOT.glob("*.py"))
    for relative_root in PRODUCTION_SOURCE_ROOTS:
        source_root = REPO_ROOT / relative_root
        if not source_root.is_dir():
            continue
        sources.extend(
            path
            for path in source_root.rglob("*.py")
            if "legacy" not in path.relative_to(REPO_ROOT).parts
            and "tests" not in path.relative_to(REPO_ROOT).parts
            and "__pycache__" not in path.relative_to(REPO_ROOT).parts
        )
    return sorted(set(sources))


def _string_mentions_legacy_root(value: str) -> bool:
    normalized = value.replace("\\", "/")
    if "teleop_records" in normalized:
        return True
    return "logs" in normalized.split("/")


def _source_mentions_legacy_root(path: Path) -> bool:
    tokens = tokenize.generate_tokens(
        io.StringIO(path.read_text(encoding="utf-8")).readline
    )
    for token in tokens:
        if token.type == tokenize.NAME and token.string == "teleop_records_root":
            return True
        if token.type != tokenize.STRING:
            continue
        if "teleop_records" in token.string:
            return True
        try:
            value = ast.literal_eval(token.string)
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, str) and _string_mentions_legacy_root(value):
            return True
    return False


def _literal_component_scopes(path: Path, component: str) -> set[str | None]:
    """Return function scopes containing one exact string path component."""

    scopes: set[str | None] = set()

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.function_stack: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            self.function_stack.append(node.name)
            self.generic_visit(node)
            self.function_stack.pop()

        def visit_AsyncFunctionDef(  # noqa: N802
            self,
            node: ast.AsyncFunctionDef,
        ) -> None:
            self.function_stack.append(node.name)
            self.generic_visit(node)
            self.function_stack.pop()

        def visit_Constant(self, node: ast.Constant) -> None:  # noqa: N802
            if node.value == component:
                scopes.add(self.function_stack[-1] if self.function_stack else None)

    Visitor().visit(ast.parse(path.read_text(encoding="utf-8")))
    return scopes


def test_legacy_artifact_root_reference_budget_does_not_grow() -> None:
    observed = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in _iter_production_sources()
        if _source_mentions_legacy_root(path)
    }
    unexpected = observed - LEGACY_ARTIFACT_ROOT_ALLOWLIST

    assert not unexpected, (
        "new production references to legacy artifact roots are not allowed; "
        "use runtime_artifact_paths or an explicit reviewed legacy-reader adapter: "
        f"{sorted(unexpected)}"
    )


def test_project_logs_literal_is_confined_to_archived_reader_roots() -> None:
    observed = {
        (path.relative_to(REPO_ROOT).as_posix(), scope)
        for path in _iter_production_sources()
        for scope in _literal_component_scopes(path, "logs")
    }

    assert observed == {
        (
            "hardware/DMTAC_gripper/examples/identify_gripper_actuator.py",
            "archived_logs_output_root",
        ),
        (
            "scripts/identify_bimanual_gripper_position.py",
            "archived_logs_output_root",
        ),
    }


def test_migrated_cli_layout_defaults_use_one_validated_local_switch() -> None:
    for relative_path, option in CONFIGURED_LAYOUT_CLI_FILES.items():
        tree = ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
        matches = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            first = node.args[0]
            if not isinstance(first, ast.Constant) or first.value != option:
                continue
            defaults = [
                keyword.value
                for keyword in node.keywords
                if keyword.arg == "default"
            ]
            matches.extend(defaults)
        assert len(matches) == 1, f"expected one {option} default in {relative_path}"
        default = matches[0]
        assert (
            isinstance(default, ast.Call)
            and isinstance(default.func, ast.Name)
            and default.func.id == "configured_runtime_artifact_layout"
        ), f"{relative_path} must use configured_runtime_artifact_layout()"
