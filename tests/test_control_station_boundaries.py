from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.support.repository_imports import (
    absolute_import_roots as _absolute_import_roots,
)
from tests.support.repository_imports import (
    python_files as _python_files,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STATION_ROOT = REPO_ROOT / "control_station"
GATEWAY_ROOT = REPO_ROOT / "agent_gateway"
pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def _violations(
    paths: tuple[Path, ...],
    forbidden_roots: set[str],
) -> dict[str, list[str]]:
    violations: dict[str, list[str]] = {}
    for path in paths:
        forbidden = _absolute_import_roots(path) & forbidden_roots
        if forbidden:
            violations[_relative(path)] = sorted(forbidden)
    return violations


def test_station_application_core_does_not_import_concrete_data_planes() -> None:
    core_paths = tuple(
        STATION_ROOT / relative_path
        for relative_path in (
            "__init__.py",
            "adapters.py",
            "composition_session.py",
            "contracts.py",
            "events.py",
            "operation_coordinator.py",
            "publication_barrier.py",
            "runtime.py",
            "state_machine.py",
            "status.py",
            "web/__init__.py",
            "web/api.py",
            "web/server.py",
        )
    )
    assert (
        _violations(
            core_paths,
            {
                "deployments",
                "agent_gateway",
                "hardware",
                "numpy",
                "scipy",
                "sim",
                "teleop_runtime",
            },
        )
        == {}
    )


def test_agent_gateway_does_not_import_upper_framework_packages() -> None:
    gateway_paths = _python_files(GATEWAY_ROOT)
    assert (
        _violations(
            gateway_paths,
            {"rpent", "rpent_dexteleop"},
        )
        == {}
    )


def test_agent_gateway_core_is_application_and_backend_neutral() -> None:
    gateway_core_paths = (
        *_python_files(GATEWAY_ROOT / "protocol"),
        *_python_files(GATEWAY_ROOT / "runtime"),
        *_python_files(GATEWAY_ROOT / "transport"),
    )
    assert (
        _violations(
            tuple(gateway_core_paths),
            {
                "control_station",
                "deployments",
                "hardware",
                "numpy",
                "scipy",
                "sim",
                "teleop_runtime",
            },
        )
        == {}
    )


def test_agent_gateway_skill_definitions_are_target_and_backend_neutral() -> None:
    assert (
        _violations(
            _python_files(GATEWAY_ROOT / "skills"),
            {
                "control_station",
                "deployments",
                "hardware",
                "numpy",
                "robot_motion",
                "scipy",
                "sim",
                "teleop_runtime",
            },
        )
        == {}
    )


def test_hitl_core_is_backend_model_topology_and_trace_neutral() -> None:
    hitl_paths = _python_files(STATION_ROOT / "hitl")
    assert (
        _violations(
            hitl_paths,
            {
                "deployments",
                "hardware",
                "sim",
                "teleop_runtime",
            },
        )
        == {}
    )

    source = "\n".join(path.read_text(encoding="utf-8") for path in hitl_paths).lower()
    for concrete_name in (
        "left_arm",
        "right_arm",
        "sim_hil",
        "mujoco",
        "openpi",
        "rm65",
        "vive",
    ):
        assert concrete_name not in source
    assert "dataset_contracts.control_trace" not in source


def test_real_teleop_and_umi_ui_packages_do_not_import_entity_data_planes() -> None:
    ui_paths = (
        *_python_files(STATION_ROOT / "real_teleop"),
        *_python_files(STATION_ROOT / "umi"),
        STATION_ROOT / "web/calibration_web_ui.py",
        STATION_ROOT / "web/startup_progress.py",
    )

    assert (
        _violations(
            tuple(ui_paths),
            {
                "collection_runtime",
                "dataset_interface",
                "hardware",
                "robot_motion",
                "sim",
                "teleop_runtime",
            },
        )
        == {}
    )


def test_station_dataset_projection_debt_is_explicit_and_cannot_spread() -> None:
    assert _violations(
        _python_files(STATION_ROOT),
        {"dataset_interface"},
    ) == {
        "control_station/real_teleop_entrypoint.py": ["dataset_interface"],
        # Concrete UMI application leaf loads the dataset/model layout contract;
        # device derivation and collection lifecycle remain lower-owned.
        "control_station/umi_application.py": ["dataset_interface"],
        "control_station/composition/mujoco_sim_hil.py": ["dataset_interface"],
        "control_station/control_trace.py": ["dataset_interface"],
        # Concrete post-capture application binding. Presentation stays in
        # Station; discovery/resolution/indexing remain lower-owned.
        "control_station/refinement_annotation/application.py": ["dataset_interface"],
        "control_station/refinement_annotation/cli.py": ["dataset_interface"],
        "control_station/refinement_annotation/detail_annotation.py": [
            "dataset_interface"
        ],
    }


def test_station_model_specific_bindings_are_explicit_and_cannot_spread() -> None:
    assert _violations(
        _python_files(STATION_ROOT),
        {"deployments"},
    ) == {
        "control_station/preflight/openpi_rtc.py": ["deployments"],
        # Concrete Sim-HIL application binding. Model transport and artifact
        # semantics remain deployments-owned; Station only connects the
        # provider to the selected simulated observation/target profile.
        "control_station/sim_hil_profile/openpi_policy_source.py": ["deployments"],
    }
    assert not (STATION_ROOT / "composition/openpi_rtc_source_preflight.py").exists()


def test_station_sim_dependencies_are_confined_to_concrete_profiles() -> None:
    assert _violations(
        _python_files(STATION_ROOT),
        {"sim"},
    ) == {
        "control_station/composition/mujoco_allocation_control_cycle.py": ["sim"],
        "control_station/composition/mujoco_allocation_publication.py": ["sim"],
        "control_station/composition/mujoco_allocation_shadow.py": ["sim"],
        "control_station/composition/mujoco_policy_human_allocation_binding.py": [
            "sim"
        ],
        "control_station/composition/mujoco_human.py": ["sim"],
        "control_station/composition/mujoco_sim_hil.py": ["sim"],
        "control_station/sim_hil.py": ["sim"],
        "control_station/real_teleop_entrypoint.py": ["sim"],
        "control_station/rm65_calibration_debug_application.py": ["sim"],
        "control_station/sim_hil_profile/factory.py": ["sim"],
        "control_station/sim_hil_profile/openpi_policy_source.py": ["sim"],
        # Concrete B601-to-RM65 GUI assembly; shared control remains sim-neutral.
        "control_station/sim_hil_profile/rebot_b601_rm65_application.py": ["sim"],
        # Concrete B601 gripper-to-RM65 MuJoCo adapter; shared mapping stays neutral.
        "control_station/sim_hil_profile/rebot_b601_rm65_gripper.py": ["sim"],
        # Concrete RM65 free-drive to B601 MuJoCo reverse assembly.
        "control_station/sim_hil_profile/rm65_b601_free_drive.py": ["sim"],
        # Concrete RM65 free-drive to B601 MuJoCo application leaf.
        "control_station/sim_hil_profile/rm65_b601_free_drive_application.py": [
            "sim"
        ],
        # Concrete RM65 gripper-to-B601 MuJoCo adapter; shared mapping stays neutral.
        "control_station/sim_hil_profile/rm65_b601_gripper.py": ["sim"],
        # Concrete state-BC Sim-HIL binding; the shared runtime remains neutral.
        "control_station/sim_hil_profile/state_bc_policy_source.py": ["sim"],
    }


def test_gateway_sim_dependencies_are_confined_to_mujoco_profile() -> None:
    assert _violations(
        _python_files(GATEWAY_ROOT),
        {"sim"},
    ) == {
        "agent_gateway/profiles/mujoco/gateway_cli.py": ["sim"],
        # Native grasp preview and target-bound contact stay in concrete leaves.
        "agent_gateway/profiles/mujoco/grasp_candidate_selection.py": ["sim"],
        # Frozen capture and exact paired execution are native profile edges.
        "agent_gateway/profiles/mujoco/grasp_deployment.py": ["sim"],
        "agent_gateway/profiles/mujoco/grasp_fast_plan.py": ["sim"],
        "agent_gateway/profiles/mujoco/grasp_observation.py": ["sim"],
        "agent_gateway/profiles/mujoco/gripper_skill.py": ["sim"],
        "agent_gateway/profiles/mujoco/held_box.py": ["sim"],
        "agent_gateway/profiles/mujoco/support_box.py": ["sim"],
        "agent_gateway/profiles/mujoco/motion_application.py": ["sim"],
        "agent_gateway/profiles/mujoco/motion_deployment.py": ["sim"],
        "agent_gateway/profiles/mujoco/motion_factory.py": ["sim"],
        # The concrete assembly names its lower measured-object observer type.
        "agent_gateway/profiles/mujoco/motion_gateway_session.py": ["sim"],
        # Sensor-shaped planning is a concrete simulator leaf.
        "agent_gateway/profiles/mujoco/motion_observed_scene.py": ["sim"],
        # Anchored motion reuses the lower planner's typed rejection in this leaf.
        "agent_gateway/profiles/mujoco/observed_point_motion.py": ["sim"],
        # The frozen query binds only this concrete simulator's private planner.
        "agent_gateway/profiles/mujoco/object_goal_query.py": ["sim"],
        # Plan-only tool hovers bind the same private native planner and its
        # typed rejections in this leaf; nothing is dispatched.
        "agent_gateway/profiles/mujoco/tool_entity.py": ["sim"],
        # Release-pose queries bind the private payload planner and payload
        # binding in this leaf; nothing is dispatched.
        "agent_gateway/profiles/mujoco/support_release_query.py": ["sim"],
        # Grasp-approach queries bind the same private planner in this leaf.
        "agent_gateway/profiles/mujoco/grasp_approach_query.py": ["sim"],
        # The concrete safe-TCP lifecycle binds its lower simulation planner.
        "agent_gateway/profiles/mujoco/motion_skill.py": ["sim"],
        "agent_gateway/profiles/mujoco/observation.py": ["sim"],
        # Carried-object composition binds the existing lower payload planner.
        "agent_gateway/profiles/mujoco/object_motion.py": ["sim"],
        # Native object measurements resolve simulation frames in this leaf.
        "agent_gateway/profiles/mujoco/object_state.py": ["sim"],
        "agent_gateway/profiles/mujoco/policy_application.py": ["sim"],
        "agent_gateway/profiles/mujoco/policy_factory.py": ["sim"],
        "agent_gateway/profiles/mujoco/skills/cube_lift/deployment.py": ["sim"],
        "agent_gateway/profiles/mujoco/visual_observation.py": ["sim"],
    }


def test_gateway_real_observation_profile_is_io_and_control_neutral() -> None:
    assert (
        _violations(
            (GATEWAY_ROOT / "profiles/real/observation_gateway.py",),
            {
                "control_station",
                "hardware",
                "robot_motion",
                "sim",
                "teleop_runtime",
            },
        )
        == {}
    )


def test_station_numerical_source_debt_is_explicit_and_cannot_spread() -> None:
    assert _violations(
        _python_files(STATION_ROOT),
        {"numpy", "scipy"},
    ) == {
        "control_station/real_teleop_entrypoint.py": ["numpy", "scipy"],
        "control_station/rm65_calibration_debug_application.py": [
            "numpy",
            "scipy",
        ],
    }


def test_gateway_numerical_dependencies_are_confined_to_concrete_profiles() -> None:
    assert _violations(
        _python_files(GATEWAY_ROOT),
        {"numpy", "scipy"},
    ) == {
        # RGB-D decoding, projection, and mask geometry stay in concrete
        # perception profiles; Gateway core remains numerical-neutral.
        "agent_gateway/profiles/back_projection.py": ["numpy"],
        "agent_gateway/profiles/mujoco/back_projection.py": ["numpy"],
        "agent_gateway/profiles/mujoco/held_box.py": ["numpy"],
        "agent_gateway/profiles/mujoco/support_box.py": ["numpy"],
        "agent_gateway/profiles/mujoco/motion_observed_scene.py": ["numpy"],
        # Candidate actions are constructed only in this concrete motion leaf.
        "agent_gateway/profiles/mujoco/observed_point_motion.py": ["numpy"],
        "agent_gateway/profiles/mujoco/object_motion.py": ["numpy"],
        "agent_gateway/profiles/mujoco/object_state.py": ["numpy"],
        "agent_gateway/profiles/placement.py": ["numpy"],
        "agent_gateway/profiles/paired_rgbd.py": ["numpy"],
        "agent_gateway/profiles/real/back_projection.py": ["numpy"],
        "agent_gateway/profiles/segmentation.py": ["numpy"],
        "agent_gateway/profiles/segmentation_batch.py": ["numpy", "scipy"],
        # Concrete anchor mask geometry shares the same numerical boundary.
        "agent_gateway/profiles/surface_anchor_selection.py": ["numpy", "scipy"],
        # Tool-entity relation and hover executors transform measured points,
        # tool poses and hover bounds between Station and geometry frames in
        # the shared profile layer and its two concrete leaves.
        "agent_gateway/profiles/tool_entity.py": ["numpy"],
        "agent_gateway/profiles/mujoco/tool_entity.py": ["numpy"],
        # Depth-only support top measurement (plane fit, coverage grid) and the
        # MuJoCo release query's candidate filtering share this boundary.
        "agent_gateway/profiles/measured_support.py": ["numpy"],
        "agent_gateway/profiles/mujoco/support_release_query.py": ["numpy"],
        "agent_gateway/profiles/mujoco/grasp_approach_query.py": ["numpy"],
        "agent_gateway/profiles/real/tool_entity.py": ["numpy"],
        "agent_gateway/profiles/real/held_object.py": ["numpy"],
        "agent_gateway/profiles/real/object_motion.py": ["numpy"],
        # The concrete RM65 assembly builds the tool-entity public frame and
        # the request-owned FK inputs at this profile edge.
        "agent_gateway/profiles/real/rm65_composition.py": ["numpy"],
        "agent_gateway/profiles/mujoco/object_goal_query.py": ["numpy"],
        "agent_gateway/profiles/mujoco/motion_skill.py": ["numpy"],
        "agent_gateway/profiles/mujoco/grasp_candidate_selection.py": ["numpy"],
        "agent_gateway/profiles/mujoco/grasp_fast_plan.py": ["numpy"],
        "agent_gateway/profiles/mujoco/grasp_observation.py": ["numpy"],
        # Object goals project the measured relation only in the concrete leaf.
        "agent_gateway/profiles/mujoco/object_motion.py": ["numpy"],
        # Private-copy body Jacobians and frame projection stay in this leaf.
        "agent_gateway/profiles/mujoco/object_state.py": ["numpy"],
        # The concrete real-RM65 CLI validates reviewed camera/tool geometry
        # while assembling its profile; Gateway core remains numerical-neutral.
        "agent_gateway/profiles/real/gateway_cli.py": ["numpy"],
        # The concrete RM65 approach profile decodes RGBD and binds its
        # retained visible scene; numerical work stays outside Gateway core.
        "agent_gateway/profiles/real/grasp_approach_plan.py": ["numpy"],
        "agent_gateway/profiles/real/grasp_fast_plan.py": ["numpy"],
        "agent_gateway/profiles/real/grasp_first_path.py": ["numpy"],
        # Frozen RGB-D/FK decoding and held-object/tool transforms belong to
        # these concrete real-profile leaves, not to Gateway core.
        "agent_gateway/profiles/real/grasp_observation.py": ["numpy"],
        # RGB-D obstacle projection and retained diagnostics are concrete profiles.
        "agent_gateway/profiles/real/motion_observed_scene.py": ["numpy"],
        "agent_gateway/profiles/real/motion_observed_scene_debug.py": ["numpy"],
        "agent_gateway/profiles/real/object_transport.py": ["numpy"],
        "agent_gateway/profiles/real/perception_probe.py": ["numpy"],
        # Offline v1 and deployment-bound live v2 point-grasp stay in one
        # concrete profile leaf: NumPy decodes/projects only at this boundary.
        "agent_gateway/profiles/real/point_grasp_gateway.py": ["numpy"],
        # Optional single-candidate rendering resolves Station/camera poses
        # only in the concrete real-perception profile boundary.
        "agent_gateway/profiles/real/point_grasp_render_gateway.py": ["numpy"],
        "agent_gateway/profiles/real/realsense_perception.py": ["numpy"],
        # Combined real-RM65 deployment resolves the reviewed v4 fixed camera
        # chains and measured wrist-camera FK only at the concrete profile edge.
        "agent_gateway/profiles/real/rm65_camera_geometry.py": ["numpy"],
        # Optional grasp reachability converts Station candidates and checks
        # finite endpoint-IK residuals only at the concrete RM65 profile edge.
        "agent_gateway/profiles/real/rm65_grasp_reachability.py": ["numpy"],
        # Concrete perception frame binding uses NumPy only to apply the
        # reviewed scene-to-station transform.  Gateway core remains neutral.
        "agent_gateway/profiles/real/perception_frames.py": ["numpy"],
        # The real release-pose query fits support planes to RGB-D and plans
        # the two legs only in this concrete RM65 profile leaf; nothing is dispatched.
        "agent_gateway/profiles/real/support_release_query.py": ["numpy"],
    }


def test_generic_passive_teleop_source_does_not_construct_vive_q6() -> None:
    source_text = (STATION_ROOT / "sources" / "teleop_pose.py").read_text(
        encoding="utf-8"
    )
    assert "passive_vive_q6" not in source_text
    assert "build_passive_vive_q6_session" not in source_text


def test_lower_packages_do_not_reverse_depend_on_control_station() -> None:
    violations: dict[str, list[str]] = {}
    for package_name in (
        "collection_runtime",
        "control_composition",
        "dataset_contracts",
        "dataset_interface",
        "deployments",
        "hardware",
        "robot_motion",
        "sim",
        "teleop_runtime",
    ):
        package_root = REPO_ROOT / package_name
        if not package_root.exists():
            continue
        violations.update(_violations(_python_files(package_root), {"control_station"}))
    # Historic command shims only. Device, motion, safety, and collection
    # ownership remain below the Station leaf, and no implementation or API
    # facade may join this allowlist.
    assert violations == {
        "teleop_runtime/collection/umi/app.py": ["control_station"],
        "teleop_runtime/debug/run_vive_rm65_calibration_debug.py": ["control_station"],
        "teleop_runtime/debug/run_vive_rm65_real_teleop.py": ["control_station"],
    }


def test_lower_packages_do_not_reverse_depend_on_agent_gateway() -> None:
    violations: dict[str, list[str]] = {}
    for package_name in (
        "collection_runtime",
        "control_composition",
        "dataset_contracts",
        "dataset_interface",
        "deployments",
        "hardware",
        "robot_motion",
        "sim",
        "teleop_runtime",
    ):
        package_root = REPO_ROOT / package_name
        if not package_root.exists():
            continue
        violations.update(_violations(_python_files(package_root), {"agent_gateway"}))
    assert violations == {}


def test_historic_command_shims_are_thin_and_hold_no_runtime_owner() -> None:
    shim_paths = (
        REPO_ROOT / "teleop_runtime/collection/umi/app.py",
        REPO_ROOT / "teleop_runtime/debug/run_vive_rm65_calibration_debug.py",
        REPO_ROOT / "teleop_runtime/debug/run_vive_rm65_real_teleop.py",
    )
    allowed_import_roots = {"__future__", "control_station", "sys"}
    for path in shim_paths:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        assert len(source.splitlines()) <= 20
        assert not any(
            isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            for node in ast.walk(tree)
        )
        assert _absolute_import_roots(path) <= allowed_import_roots
