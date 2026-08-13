"""Project-wide pytest tier classification.

The repository keeps its tests in one directory while the runtime is being
split into smaller packages.  Classifying by module name here gives developers
stable selection commands without a risky bulk file move.  A test may carry a
contract marker in addition to its execution tier.

New tests are unit tests by default.  Add their module to an explicit set below
only when it needs optional software or performs multi-component/file-format
integration work; do not classify mocked hardware behavior as ``hardware``.

Tests marked ``hardware`` are skipped unless the operator explicitly supplies
``--run-hardware``.  The opt-in only permits those tests to run; callers should
also select them with ``-m hardware`` and complete the device-specific safety
preflight.  Test-module import must remain free of device I/O because marker
handling happens after collection.
"""

from __future__ import annotations

from pathlib import Path
import socket

import pytest


OPTIONAL_MODULES = frozenset(
    {
        "test_dataset_media_integrity.py",
        "test_dmtac_viewer.py",
        "test_mujoco_fast_runtime.py",
        "test_mujoco_rgb_distortion.py",
        "test_mujoco_umi_replay.py",
        "test_multi_view_real_sim_compare.py",
        "test_multi_view_video_player.py",
        "test_openpi05_deployment.py",
        "test_parquet_schema_migration.py",
        "test_workspace_near_boundary_postprocess.py",
        "test_wrist_rgb_mask_viewer.py",
    }
)

INTEGRATION_MODULES = frozenset(
    {
        "test_collection_background_finalize.py",
        "test_collection_pose_frame_storage.py",
        "test_collection_process_writer.py",
        "test_collection_storage_queue.py",
        "test_control_station_mujoco_human_station.py",
        "test_control_station_mujoco_sim_hil.py",
        "test_control_station_sim_hil_gui.py",
        "test_control_station_vive_calibration.py",
        "test_control_trace_episode.py",
        "test_dmtac_spawn_import_path.py",
        "test_dataset_interface.py",
        "test_dataset_postprocess.py",
        "test_dataset_v22_deploy.py",
        "test_hardlink_dedupe.py",
        "test_logical_dataset_manifest.py",
        "test_mujoco_formal_capture.py",
        "test_mujoco_temp_pipeline.py",
        "test_parquet_schema_migration.py",
        "test_position_deploy_executor.py",
        "test_relocate_mask_metadata.py",
        "test_rm65_worker_drain_integration.py",
        "test_tactile_external_storage.py",
        "test_umi_next_state_arm_action.py",
        "test_umi_v25p_episode_identity.py",
        "test_v25p_dual_write_shadow.py",
        "test_v25p_episode_maintenance_guard.py",
        "test_v25p_sidecar_store.py",
        "test_writer_video_stats.py",
        "test_wrist_depth_visualize.py",
        "test_wrist_final_mask.py",
        "test_wrist_rgb_mask_viewer.py",
    }
)

CONTRACT_MODULES = frozenset(
    {
        "test_camera_registry.py",
        "test_collection_pose_frame_storage.py",
        "test_component_target_capability.py",
        "test_dataset_arm_control_mode_contract.py",
        "test_dataset_dependency_boundaries.py",
        "test_dataset_v22_deploy.py",
        "test_dataset_v22_layout.py",
        "test_embodiment_generalization_firewall.py",
        "test_model_continuation_data_contract.py",
        "test_openpi_model_rtc_artifact_binding.py",
        "test_package_migrations.py",
        "test_parquet_schema_migration.py",
        "test_robot_motion_action_contracts.py",
        "test_robot_motion_deploy_contracts.py",
        "test_robot_motion_replay_contracts.py",
        "test_rm65_joint6_limits.py",
        "test_rm65_target_runtime_provider.py",
        "test_rtc_causal_conditioning_binding.py",
        "test_rtc_conditioning_bundle.py",
        "test_sim_real2sim_artifact.py",
        "test_sim_contracts.py",
        "test_target_runtime_contracts.py",
        "test_target_runtime_preparation.py",
        "test_target_runtime_profile_selection.py",
        "test_teleop_action_contract.py",
        "test_teleop_collection_topology.py",
        "test_teleop_runtime_lazy_imports.py",
        "test_v25p_capture_profiles.py",
        "test_v25p_contracts.py",
    }
)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the explicit real-hardware execution gate."""

    group = parser.getgroup("dexteleop")
    group.addoption(
        "--run-hardware",
        action="store_true",
        default=False,
        help=(
            "Permit tests marked 'hardware' to run. Also select them explicitly "
            "with '-m hardware' and complete the device safety preflight."
        ),
    )


@pytest.fixture
def require_udp_loopback() -> None:
    """Skip a real-loopback integration test when UDP sockets are unavailable."""

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind(("127.0.0.1", 0))
        finally:
            sock.close()
    except OSError as exc:
        pytest.skip(f"loopback UDP is unavailable in this environment: {exc}")


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Apply tiers and keep real-hardware tests opt-in."""

    for item in items:
        module_name = Path(str(item.path)).name
        has_optional_marker = item.get_closest_marker("optional") is not None
        has_integration_marker = item.get_closest_marker("integration") is not None
        has_unit_marker = item.get_closest_marker("unit") is not None
        optional = module_name in OPTIONAL_MODULES or has_optional_marker
        integration = (
            module_name in INTEGRATION_MODULES
            or optional
            or has_integration_marker
        )

        if optional and not has_optional_marker:
            item.add_marker(pytest.mark.optional)
        if integration and not has_integration_marker:
            item.add_marker(pytest.mark.integration)
        elif not integration and not has_unit_marker:
            item.add_marker(pytest.mark.unit)
        if module_name in CONTRACT_MODULES:
            item.add_marker(pytest.mark.contract)

    if config.getoption("--run-hardware"):
        return

    skip_hardware = pytest.mark.skip(
        reason=(
            "real-hardware test requires explicit --run-hardware permission, "
            "'-m hardware' selection, and the device-specific safety preflight"
        )
    )
    for item in items:
        if item.get_closest_marker("hardware") is not None:
            item.add_marker(skip_hardware)
