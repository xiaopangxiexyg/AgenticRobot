from __future__ import annotations

import ast
from configparser import ConfigParser
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


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


def _registered_submodule_roots() -> tuple[Path, ...]:
    parser = ConfigParser()
    parser.read(REPO_ROOT / ".gitmodules", encoding="utf-8")
    return tuple(
        REPO_ROOT / parser.get(section, "path")
        for section in parser.sections()
    )


PRODUCTION_SOURCE_SUBMODULE_ROOTS = _registered_submodule_roots()

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
        "control_station/real_teleop_entrypoint.py",
        "teleop_runtime/deploy/debug_logger.py",
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
    "teleop_runtime/deploy/deploy_cli_parser.py": "--artifact-layout",
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
            and not any(
                root in path.parents
                for root in PRODUCTION_SOURCE_SUBMODULE_ROOTS
            )
        )
    return sorted(set(sources))


def _string_mentions_legacy_root(value: str) -> bool:
    normalized = value.replace("\\", "/")
    if "teleop_records" in normalized:
        return True
    return "logs" in normalized.split("/")


@dataclass(frozen=True)
class _ArtifactReferenceSummary:
    mentions_legacy_root: bool
    logs_literal_scopes: frozenset[str | None]


_IDENTIFIER_FIELDS = frozenset(
    {"arg", "asname", "attr", "id", "module", "name", "names", "rest"}
)


@lru_cache(maxsize=None)
def _artifact_reference_summary(path: Path) -> _ArtifactReferenceSummary:
    """Scan one source once for both runtime-artifact boundary assertions."""

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    has_raw_legacy_identifier = "teleop_records_root" in source
    has_raw_teleop_records = "teleop_records" in source
    source_lines = tuple(
        line.encode("utf-8") for line in source.splitlines(keepends=True)
    )

    def raw_source_segment(node: ast.AST) -> str:
        if node.end_lineno is None or node.end_col_offset is None:
            return ""
        start = node.lineno - 1
        end = node.end_lineno - 1
        if start == end:
            segment = source_lines[start][node.col_offset : node.end_col_offset]
        else:
            segment = b"".join(
                (
                    source_lines[start][node.col_offset :],
                    *source_lines[start + 1 : end],
                    source_lines[end][: node.end_col_offset],
                )
            )
        return segment.decode("utf-8")

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.function_stack: list[str] = []
            self.mentions_legacy_root = False
            self.logs_literal_scopes: set[str | None] = set()

        def generic_visit(self, node: ast.AST) -> None:
            if has_raw_legacy_identifier:
                for field_name, value in ast.iter_fields(node):
                    if field_name not in _IDENTIFIER_FIELDS:
                        continue
                    values = value if isinstance(value, list) else (value,)
                    if any(
                        isinstance(item, str)
                        and "teleop_records_root" in item.split(".")
                        for item in values
                    ):
                        self.mentions_legacy_root = True
            super().generic_visit(node)

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
            value = node.value
            if not isinstance(value, str):
                return
            if (
                (
                    has_raw_teleop_records
                    and "teleop_records" in raw_source_segment(node)
                )
                or (
                    ("logs" in value or "teleop_records" in value)
                    and _string_mentions_legacy_root(value)
                )
            ):
                self.mentions_legacy_root = True
            if value == "logs":
                self.logs_literal_scopes.add(
                    self.function_stack[-1] if self.function_stack else None
                )

    visitor = Visitor()
    visitor.visit(tree)
    return _ArtifactReferenceSummary(
        mentions_legacy_root=visitor.mentions_legacy_root,
        logs_literal_scopes=frozenset(visitor.logs_literal_scopes),
    )


def test_legacy_artifact_root_reference_budget_does_not_grow() -> None:
    observed = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in _iter_production_sources()
        if _artifact_reference_summary(path).mentions_legacy_root
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
        for scope in _artifact_reference_summary(path).logs_literal_scopes
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
