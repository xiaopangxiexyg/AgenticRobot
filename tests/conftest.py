"""Project-wide pytest tier classification.

The repository keeps its tests in one directory while the runtime is being
split into smaller packages.  Classifying by module name here gives developers
stable selection commands without a risky bulk file move.  A test may carry a
contract marker in addition to its execution tier.

New tests are unit tests by default.  Add their module to an explicit set below
only when it needs optional software or performs multi-component/file-format
integration work; do not classify mocked hardware behavior as ``hardware``.
"""

from __future__ import annotations

from pathlib import Path
import socket

import pytest


OPTIONAL_MODULES = frozenset(
    {
        "test_mujoco.py",
        "test_mujoco_canonical_ik.py",
        "test_mujoco_collision.py",
        "test_mujoco_cube_drop.py",
        "test_mujoco_cube_lift.py",
        "test_mujoco_dataset_replay.py",
        "test_mujoco_dataset_v22.py",
        "test_mujoco_direct_qpos.py",
        "test_mujoco_fast_runtime.py",
        "test_mujoco_physics_scene.py",
        "test_mujoco_pointcloud.py",
        "test_mujoco_rgb_distortion.py",
        "test_multi_view_real_sim_compare.py",
        "test_multi_view_video_player.py",
        "test_openpi05_deployment.py",
        "test_sapien_renderer_viewer_close.py",
        "test_state_augmentation.py",
    }
)

INTEGRATION_MODULES = frozenset(
    {
        "test_collection_background_finalize.py",
        "test_collection_storage_queue.py",
        "test_dataset_interface.py",
        "test_dataset_postprocess.py",
        "test_dataset_v22_deploy.py",
        "test_dataset_v22_layout.py",
        "test_parquet_schema_migration.py",
        "test_real_undistort.py",
        "test_replay_gui_dataset.py",
        "test_tactile_external_storage.py",
        "test_writer_video_stats.py",
        "test_wrist_depth_storage.py",
        "test_wrist_depth_visualize.py",
        "test_wrist_final_mask.py",
        "test_wrist_rgb_mask_viewer.py",
    }
)

CONTRACT_MODULES = frozenset(
    {
        "test_camera_registry.py",
        "test_dataset_arm_control_mode_contract.py",
        "test_dataset_dependency_boundaries.py",
        "test_dataset_v22_deploy.py",
        "test_dataset_v22_layout.py",
        "test_package_migrations.py",
        "test_parquet_schema_migration.py",
        "test_robot_motion_action_contracts.py",
        "test_robot_motion_deploy_contracts.py",
        "test_robot_motion_replay_contracts.py",
        "test_sim_contracts.py",
        "test_teleop_action_contract.py",
        "test_teleop_collection_topology.py",
    }
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


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Apply execution-tier markers without changing existing test node IDs."""

    for item in items:
        module_name = Path(str(item.path)).name
        optional = module_name in OPTIONAL_MODULES
        integration = module_name in INTEGRATION_MODULES or optional

        if optional:
            item.add_marker(pytest.mark.optional)
        if integration:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)
        if module_name in CONTRACT_MODULES:
            item.add_marker(pytest.mark.contract)
