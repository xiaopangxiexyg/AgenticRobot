#!/usr/bin/env python3
"""Deterministic pytest target sharding and aggregate JUnit evidence."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import socket
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path

# These weights are deliberately sparse.  They capture the modules that
# dominate recent protected runs; unknown/new modules use a stable source-size
# estimate and are still assigned automatically.  Keeping the timing profile
# here makes the default plan useful in a clean checkout without depending on
# mutable Harness history.
FULL_WEIGHTS = {
    # Existing testcase sums from the green six-shard baseline at 47d7be6e,
    # except the two artifact I/O modules refreshed at c3c06be6 (2026-09-25).
    # These are scheduling estimates only; they do not change test selection.
    "tests/test_mujoco_temp_pipeline.py": 134.914,
    "tests/test_replay_position_admittance_active_pipeline.py": 108.657,
    "tests/test_rm65_grasp_trajectory.py": 91.046,
    "tests/test_mujoco_fast_runtime.py": 48.101,
    "tests/test_control_station_mujoco_sim_hil.py": 36.534,
    "tests/test_runtime_artifact_migration.py": 49.666,
    "tests/test_rebot_b601_real2sim.py": 23.742,
    "tests/test_collection_process_writer.py": 22.553,
    "tests/test_rebot_b601_mujoco_haptics.py": 22.540,
    "tests/test_control_station_sim_hil_allocation_profile.py": 21.878,
    "tests/test_replay_position_admittance_ik_trace.py": 17.637,
    "tests/test_sim_point_grasp_experiment.py": 16.729,
    "tests/test_agent_gateway_real_rm65_composition.py": 16.526,
    "tests/test_mujoco_trajectory_composition_publication.py": 14.943,
    "tests/test_real_shadow_acceptance.py": 14.336,
    "tests/test_camera_semantic_materialize.py": 13.292,
    "tests/test_runtime_domain_boundaries.py": 13.279,
    "tests/test_mujoco_state_bc_evaluation.py": 12.778,
    "tests/test_mujoco_cube_lift.py": 11.965,
    "tests/test_refinement_annotation_gui.py": 11.865,
    "tests/test_mujoco.py": 11.759,
    "tests/test_rebot_b601_submodule_boundary.py": 11.348,
    "tests/test_runtime_artifact_migration_boundaries.py": 11.004,
    "tests/test_position_admittance_active_shared_ik.py": 10.218,
    "tests/test_mujoco_actuated_position.py": 10.211,
    "tests/test_mujoco_state_bc_live_hitl.py": 9.473,
    "tests/test_mujoco_collision_safety.py": 8.356,
    "tests/test_runtime_artifact_resolution.py": 15.847,
    "tests/test_v25p_producer_dual_write.py": 8.031,
    "tests/test_control_station_exclusive_hitl.py": 7.844,
    "tests/test_sim_point_grasp_terminal.py": 7.738,
    "tests/test_mujoco_pose_execution_observation.py": 7.538,
    "tests/test_modality_storage.py": 7.527,
    "tests/test_mujoco_component_action_profile.py": 7.347,
    "tests/test_v25p_dual_write_shadow.py": 7.181,
    "tests/test_agent_gateway_real_station_point_grasp.py": 7.160,
    "tests/test_mujoco_physics_scene.py": 7.152,
    "tests/test_control_station_boundaries.py": 7.042,
    "tests/test_dmtac_viewer.py": 6.917,
    "tests/test_rebot_b601_mujoco_mit.py": 6.894,
    "tests/test_mujoco_cube_drop.py": 6.884,
    "tests/test_real_teleop_progress_adapter.py": 6.784,
    "tests/test_mujoco_formal_compound_authority.py": 6.681,
    "tests/test_openpi_policy_emission_source.py": 6.172,
    "tests/test_run_pytest_asset_root.py": 6.149,
    "tests/test_lazy_package_root_exports.py": 5.736,
    "tests/test_control_station_contracts.py": 5.593,
    "tests/test_policy_learning_boundaries.py": 5.103,
    # Previously unweighted modules >= 5 s in the green 298221fc full Run
    # full-20261018-baseline6j1node22 (six serial batches, Node 22).
    "tests/test_dmtac_cube_contact_probe.py": 66.941,
    "tests/test_agent_gateway_mujoco_safe_tcp_pose.py": 33.811,
    "tests/test_rm65_dmtac_sweep_probe.py": 15.684,
    "tests/test_owned_worker_cleanup.py": 12.397,
    "tests/test_mujoco_planned_tcp_constraints.py": 11.803,
    "tests/test_openpi_cpu_resources.py": 10.931,
    "tests/test_agent_gateway_mujoco_gripper_contact.py": 9.849,
    "tests/test_sam3_service.py": 9.564,
    "tests/test_agent_gateway_mujoco_back_projection.py": 8.847,
}

# Modules normally remain the isolation unit. These two long-running modules
# have independent top-level tests: replay uses function-local tmp_path logs
# and fresh runtimes, not shared rollout fixtures. Parameter matrices remain
# intact. Unlisted functions share the measured residual and stay collected.
FULL_SPLIT_NODE_WEIGHTS = {
    "tests/test_mujoco_temp_pipeline.py": {
        "test_contact_curriculum_reset_retries_recoverable_failure": 3.289,
        "test_contact_curriculum_reset_starts_from_real_bilateral_contact": 3.249,
        "test_contact_curriculum_v2_applies_lift_first_action_constraints": 3.349,
        "test_contact_curriculum_v2_repeated_reset_preserves_liftable_grasp": 7.102,
        "test_contact_curriculum_v2_scripted_gate_lifts_from_reset": 3.998,
        "test_contact_curriculum_v2_tracks_contact_loss_without_early_termination": 3.421,
        "test_contact_curriculum_v3_applies_only_nonnegative_lift_action": 3.411,
        "test_contact_curriculum_v3_projection_preserves_locked_gripper_target": 3.446,
        "test_contact_curriculum_v3_scripted_gate_lifts_from_reset": 4.143,
        "test_descend_close_task_is_feasible_without_hidden_xy_actions": 3.061,
        "test_goal_delta_actor_bootstrap_is_optional_and_nonprivileged": 1.821,
        "test_goal_delta_pick_physical_oracle_grasps_from_home": 4.075,
        "test_lift_short_v2_zero_policy_holds_and_positive_policy_lifts": 7.940,
        "test_normal_v2_scripted_physical_grasp_gate_reaches_bilateral_lift": 3.795,
        "test_pose_reset_rebuilds_solver_and_syncs_settled_actuator_state": 6.192,
        "test_pose_task_suite_has_a_scripted_feasibility_gate": 46.417,
        "test_post_contact_phase_exits_after_configured_contact_loss_grace": 2.719,
        "test_privileged_critic_checkpoint_keeps_actor_inference_self_contained": 1.626,
        "test_privileged_near_lift_v2_switches_to_lift_only_after_real_contact": 3.314,
        "test_randomized_pose_task_repeated_seed_has_identical_closed_loop_trace": 3.440,
        "test_reach_goal_delta_matches_pose_target_in_public_action_frame": 1.851,
        "test_reach_task_kernel_preserves_pose_action_lineage_and_seeded_reset": 3.779,
        "test_separate_actor_observation_critic_cannot_update_actor_backbone": 1.598,
        "test_settled_snapshot_preserves_first_actuated_policy_step": 2.652,
        "test_temp_policy_checkpoint_round_trip": 1.616,
    },
    "tests/test_replay_position_admittance_active_pipeline.py": {
        "test_larger_envelope_mujoco_abrupt_release_retains_frozen_catchup_gate": 27.246,
        "test_right_combined_formal_active_runs_through_physical_mujoco": 27.142,
        "test_half_normal_mujoco_endpoint_guard_holds_before_larger_qref_dispatch": 15.085,
        "test_named_larger_active_envelopes_close_through_shared_ik_and_release": 13.590,
        "test_raw_wrench_active_pipeline_runs_through_physical_mujoco": 13.003,
        "test_raw_wrench_active_pipeline_exercises_shared_ik": 4.441,
        "test_normal_shadow_envelope_is_replayed_under_cautious_active_caps": 4.218,
        "test_sub_hold_wrench_is_not_promoted_as_active_ik_coverage": 3.924,
    },
}

FAST_WEIGHTS = {
    "tests/test_real_shadow_acceptance.py": 18.562,
    "tests/test_runtime_domain_boundaries.py": 9.370,
    "tests/test_refinement_annotation_gui.py": 7.908,
    "tests/test_modality_storage.py": 6.843,
    "tests/test_real_teleop_progress_adapter.py": 6.224,
    "tests/test_lazy_package_root_exports.py": 5.753,
    "tests/test_control_station_contracts.py": 5.290,
    "tests/test_run_pytest_asset_root.py": 4.251,
    "tests/test_official_all_ik_governor.py": 4.179,
    "tests/test_annotation_projection.py": 3.834,
    "tests/test_agent_gateway_compatibility.py": 2.835,
    "tests/test_idle_tactile_compression.py": 2.795,
    "tests/test_control_station_boundaries.py": 2.740,
    "tests/test_agent_gateway_artifact_store.py": 2.604,
    "tests/test_position_admittance_active_shared_ik.py": 2.598,
}

FAST_INTEGRATION_WEIGHTS = {
    # Exact module sums from the unsharded integration tier at 47d7be6e.
    # Unit-heavy modules deliberately do not influence this plan.
    "tests/test_rm65_grasp_trajectory.py": 74.288,
    "tests/test_runtime_artifact_migration.py": 25.525,
    "tests/test_collection_process_writer.py": 14.611,
    "tests/test_runtime_artifact_resolution.py": 8.269,
    "tests/test_v25p_producer_dual_write.py": 6.729,
    "tests/test_agent_gateway_mujoco_long_move_trace.py": 5.382,
    "tests/test_v25p_dual_write_shadow.py": 3.672,
    "tests/test_mujoco.py": 3.477,
    "tests/test_writer_v21_import_path.py": 3.145,
    "tests/test_handeye_utils.py": 2.990,
    "tests/test_position_deploy_executor.py": 2.812,
    "tests/test_dmtac_spawn_import_path.py": 2.554,
    "tests/test_worktree_coordination.py": 1.956,
    "tests/test_run_full_acceptance.py": 1.948,
    "tests/test_collection_progress_runtime.py": 1.715,
    "tests/test_mujoco_formal_capture.py": 1.206,
    "tests/test_v25p_native_store.py": 0.923,
    "tests/test_run_fast_feedback.py": 0.886,
    "tests/test_position_force_guard.py": 0.840,
    "tests/test_progress_timeline.py": 0.820,
    "tests/test_v25p_sidecar_store.py": 0.670,
    "tests/test_control_trace_episode.py": 0.638,
    "tests/test_tactile_external_storage.py": 0.569,
    "tests/test_control_station_vive_deflection.py": 0.511,
}

WEIGHT_PROFILES = {
    "fast": FAST_WEIGHTS,
    "fast-integration": FAST_INTEGRATION_WEIGHTS,
    "full": FULL_WEIGHTS,
}


def check_required_local_sockets() -> None:
    """Verify the local transports exercised by complete acceptance."""

    with tempfile.TemporaryDirectory(prefix="dt-full-socket-") as directory:
        probes = (
            (socket.AF_UNIX, str(Path(directory) / "probe.sock"), "AF_UNIX"),
            (socket.AF_INET, ("127.0.0.1", 0), "loopback AF_INET"),
        )
        for family, address, label in probes:
            try:
                with socket.socket(family, socket.SOCK_STREAM) as server:
                    server.settimeout(1.0)
                    server.bind(address)
                    server.listen(1)
                    with socket.socket(family, socket.SOCK_STREAM) as client:
                        client.settimeout(1.0)
                        client.connect(server.getsockname())
                        connection, _ = server.accept()
                        connection.close()
            except OSError as error:
                raise OSError(
                    f"required local socket capability unavailable for {label}: {error}"
                ) from error


def _estimated_weight(path: Path, relative_path: str, profile: str) -> float:
    measured = WEIGHT_PROFILES[profile].get(relative_path)
    if measured is not None:
        return measured
    # Source size is a better fallback than file-count round robin for large
    # assertion/parameter tables, while the floor prevents tiny modules from
    # all accumulating in one shard.
    return max(0.05, path.stat().st_size / 50_000.0)


def _split_test_functions(path: Path, relative_path: str) -> list[str]:
    """Return every statically discoverable top-level test in a split module."""

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as error:
        raise ValueError(
            f"could not parse split test module {relative_path}: {error}"
        ) from error
    unsupported = []
    functions = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and (
            node.name.startswith("Test")
            or any(
                isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and child.name.startswith("test_")
                for child in node.body
            )
        ):
            unsupported.append(f"test class {node.name}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in {
                "pytest_collect_file",
                "pytest_collection_modifyitems",
                "pytest_generate_tests",
            }:
                unsupported.append(f"collection hook {node.name}")
            elif node.name.startswith("test_"):
                functions.append(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            if any(
                isinstance(target, ast.Name) and target.id.startswith("test_")
                for target in targets
            ):
                unsupported.append("dynamically assigned test")
        elif isinstance(node, (ast.Import, ast.ImportFrom)) and any(
            (alias.asname or alias.name).startswith("test_") for alias in node.names
        ):
            unsupported.append("imported test")
    nested_tests = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    } - set(functions)
    unsupported.extend(f"nested test {name}" for name in sorted(nested_tests))
    if unsupported:
        raise ValueError(
            f"split test module {relative_path} has unsupported collection: "
            + ", ".join(unsupported)
        )
    if not functions:
        raise ValueError(f"split test module has no top-level tests: {relative_path}")
    if len(functions) != len(set(functions)):
        raise ValueError(
            f"split test module has duplicate test functions: {relative_path}"
        )
    return functions


def _shard_items(
    path: Path,
    relative_path: str,
    profile: str,
) -> list[tuple[str, Path, float]]:
    node_weights = (
        FULL_SPLIT_NODE_WEIGHTS.get(relative_path) if profile == "full" else None
    )
    if node_weights is None:
        return [(relative_path, path, _estimated_weight(path, relative_path, profile))]

    functions = _split_test_functions(path, relative_path)
    unknown_weights = set(node_weights) - set(functions)
    if unknown_weights:
        raise ValueError(
            f"split timing profile names missing tests in {relative_path}: "
            f"{sorted(unknown_weights)}"
        )
    module_weight = _estimated_weight(path, relative_path, profile)
    measured_weight = sum(node_weights.values())
    unmeasured = len(functions) - len(node_weights)
    residual = module_weight - measured_weight
    if residual < 0.0 or (unmeasured == 0 and residual > 1e-6):
        raise ValueError(
            f"split timing profile does not match module weight for {relative_path}: "
            f"module={module_weight} measured={measured_weight}"
        )
    fallback = residual / unmeasured if unmeasured else 0.0
    return [
        (
            f"{relative_path}::{name}",
            Path(f"{path}::{name}"),
            node_weights.get(name, fallback),
        )
        for name in functions
    ]


def plan_shards(
    repository: Path,
    tests_root: Path,
    shard_count: int,
    profile: str,
) -> list[list[tuple[str, Path, float]]]:
    repository = repository.resolve()
    tests_root = tests_root.resolve()
    files = sorted(path for path in tests_root.rglob("test_*.py") if path.is_file())
    if not files:
        raise ValueError(f"no pytest modules found under {tests_root}")
    if not 1 <= shard_count <= len(files):
        raise ValueError(
            f"shard count must be between 1 and {len(files)}: {shard_count}"
        )

    weighted: list[tuple[str, Path, float]] = []
    expected_modules: list[str] = []
    for path in files:
        try:
            relative = path.relative_to(repository).as_posix()
        except ValueError as error:
            raise ValueError(
                f"tests root is outside repository: {tests_root}"
            ) from error
        expected_modules.append(relative)
        weighted.extend(_shard_items(path, relative, profile))

    shards: list[list[tuple[str, Path, float]]] = [[] for _ in range(shard_count)]
    totals = [0.0] * shard_count
    for item in sorted(weighted, key=lambda value: (-value[2], value[0])):
        index = min(
            range(shard_count),
            key=lambda candidate: (
                totals[candidate],
                len(shards[candidate]),
                candidate,
            ),
        )
        shards[index].append(item)
        totals[index] += item[2]

    planned = [target for shard in shards for target, _, _ in shard]
    if len(planned) != len(set(planned)):
        raise ValueError("internal shard plan contains duplicate test targets")
    planned_modules = {target.split("::", 1)[0] for target in planned}
    if planned_modules != set(expected_modules):
        missing = sorted(set(expected_modules) - planned_modules)
        extra = sorted(planned_modules - set(expected_modules))
        raise ValueError(
            f"internal shard coverage mismatch: missing={missing} extra={extra}"
        )
    return shards


def write_plan(
    repository: Path,
    tests_root: Path,
    output_dir: Path,
    shard_count: int,
    profile: str,
    prefix: str,
    manifest_name: str = "shard-plan.json",
) -> None:
    if not output_dir.is_dir():
        raise ValueError(f"output directory does not exist: {output_dir}")
    if Path(manifest_name).name != manifest_name or not manifest_name.endswith(".json"):
        raise ValueError(f"invalid shard manifest name: {manifest_name}")
    shards = plan_shards(repository, tests_root, shard_count, profile)
    manifest_shards = []
    for index, shard in enumerate(shards):
        paths_file = output_dir / f"{prefix}-{index}-paths.txt"
        paths_file.write_text(
            "".join(f"{path}\n" for _, path, _ in shard), encoding="utf-8"
        )
        test_targets = [item[0] for item in shard]
        test_modules = sorted({target.split("::", 1)[0] for target in test_targets})
        manifest_shards.append(
            {
                "index": index,
                "estimated_seconds": round(sum(item[2] for item in shard), 6),
                "file_count": len(test_modules),
                "target_count": len(shard),
                "paths_file": paths_file.name,
                "test_modules": test_modules,
                "test_targets": test_targets,
            }
        )
    manifest = {
        "schema": "dexteleop.pytest-shard-plan.v2",
        "profile": profile,
        "shard_count": shard_count,
        "file_count": len(
            {target.split("::", 1)[0] for shard in shards for target, _, _ in shard}
        ),
        "target_count": sum(len(shard) for shard in shards),
        "estimated_seconds": round(
            sum(item[2] for shard in shards for item in shard), 6
        ),
        "shards": manifest_shards,
    }
    (output_dir / manifest_name).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _test_suites(root: ET.Element, source: Path) -> Iterable[ET.Element]:
    if root.tag == "testsuite":
        return (root,)
    if root.tag == "testsuites":
        suites = tuple(child for child in root if child.tag == "testsuite")
        if suites:
            return suites
        raise ValueError(f"JUnit shard has no <testsuite>: {source}")
    raise ValueError(f"JUnit shard root is not <testsuite> or <testsuites>: {source}")


def merge_junit(destination: Path, sources: list[Path]) -> None:
    if not sources:
        raise ValueError(f"no JUnit reports were supplied for {destination}")
    roots: list[ET.Element] = []
    for source in sources:
        if not source.is_file():
            raise ValueError(f"missing JUnit shard report: {source}")
        roots.extend(_test_suites(ET.parse(source).getroot(), source))

    merged = ET.Element("testsuite", dict(roots[0].attrib))
    for name in ("tests", "errors", "failures", "skipped"):
        merged.set(name, str(sum(int(root.attrib.get(name, "0")) for root in roots)))
    merged.set(
        "time",
        format(sum(float(root.attrib.get("time", "0")) for root in roots), ".6f"),
    )

    seen_cases: set[tuple[str, str]] = set()
    for root in roots:
        for child in root:
            if child.tag == "testcase":
                identity = (
                    child.attrib.get("classname", ""),
                    child.attrib.get("name", ""),
                )
                if identity in seen_cases:
                    raise ValueError(
                        "duplicate testcase in JUnit shards: "
                        f"{identity[0]}::{identity[1]}"
                    )
                seen_cases.add(identity)
            merged.append(copy.deepcopy(child))

    ET.indent(merged, space="  ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(merged).write(destination, encoding="utf-8", xml_declaration=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="write deterministic shard path files")
    plan.add_argument("--repository", type=Path, required=True)
    plan.add_argument("--tests-root", type=Path, required=True)
    plan.add_argument("--output-dir", type=Path, required=True)
    plan.add_argument("--shards", type=int, required=True)
    plan.add_argument("--profile", choices=sorted(WEIGHT_PROFILES), required=True)
    plan.add_argument("--prefix", default="shard")
    plan.add_argument("--manifest-name", default="shard-plan.json")
    merge = commands.add_parser("merge-junit", help="merge and validate shard JUnit")
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("sources", type=Path, nargs="+")
    commands.add_parser(
        "check-sockets",
        help="verify loopback TCP and Unix sockets before complete acceptance",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "plan":
            write_plan(
                arguments.repository,
                arguments.tests_root,
                arguments.output_dir,
                arguments.shards,
                arguments.profile,
                arguments.prefix,
                arguments.manifest_name,
            )
        elif arguments.command == "merge-junit":
            merge_junit(arguments.output, arguments.sources)
        else:
            check_required_local_sockets()
    except (OSError, ValueError, ET.ParseError) as error:
        print(f"pytest_sharding: error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
