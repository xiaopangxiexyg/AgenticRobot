from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from agent_gateway.profiles.real.gateway_cli import (
    DEFAULT_RM65_CAMERA_MOUNT_URDF,
    DEFAULT_RM65_CAMERA_REGISTRY_V4,
)
from agent_gateway.profiles.real.realsense_perception import (
    build_real_perception_source_binding,
)
from agent_gateway.profiles.real.rm65_camera_geometry import (
    build_rm65_rgb_camera_geometry,
)
from robot_motion.rm65 import (
    build_rm65_dual_arm_scene_profile,
    load_world_from_arm_base_transforms,
    load_world_from_fixed_link_transform,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CAMERA_CONFIG = (
    REPO_ROOT / "teleop_runtime/configs/devices/agent_gateway_realsense.json"
)


def _scene_geometry():
    scene = build_rm65_dual_arm_scene_profile("dual_arm_hand_proxy_open")
    return (
        load_world_from_arm_base_transforms(scene),
        load_world_from_fixed_link_transform(
            scene,
            link_name="workspace_near_boundary_frame",
        ),
    )


def _source_binding(registry_path: Path):
    arm_bases, _station = _scene_geometry()
    return build_real_perception_source_binding(
        camera_config_path=CAMERA_CONFIG,
        camera_registry_path=registry_path,
        scene_world_from_arm_base_by_side=arm_bases,
    )


def _build_geometry(
    *,
    registry_path: Path,
    station_override=None,
    target_builder=None,
    camera_mount_urdf_path: Path = DEFAULT_RM65_CAMERA_MOUNT_URDF,
    camera_asset_root: Path = REPO_ROOT / "assets",
):
    arm_bases, station = _scene_geometry()
    return build_rm65_rgb_camera_geometry(
        source_binding=_source_binding(registry_path),
        target_builder=(
            SimpleNamespace(solver_world_targets_for_qpos_pair=lambda **_kwargs: None)
            if target_builder is None
            else target_builder
        ),
        camera_registry_path=registry_path,
        camera_mount_urdf_path=camera_mount_urdf_path,
        camera_asset_root=camera_asset_root,
        scene_world_from_arm_base_by_side=arm_bases,
        scene_world_from_canonical_station=(
            station if station_override is None else station_override
        ),
    )


def test_v4_default_geometry_closes_over_all_three_deployed_camera_roles() -> None:
    assert DEFAULT_RM65_CAMERA_MOUNT_URDF == (
        REPO_ROOT / "assets/rm65_inspire_gripper_v4/"
        "rm65_dual_arm_wrist_cameras_v4_head_camera_4b2_cad_with_legacy_housing.urdf"
    )
    assert DEFAULT_RM65_CAMERA_REGISTRY_V4 == (
        REPO_ROOT / "robot_motion/configs/rm65_camera_registry_v4.json"
    )

    geometry = _build_geometry(registry_path=DEFAULT_RM65_CAMERA_REGISTRY_V4)

    assert tuple(
        (item["source_id"], item["role"]) for item in geometry.source_catalog()
    ) == (
        ("front", "main"),
        ("left_wrist", "wrist_left"),
        ("right_wrist", "wrist_right"),
    )


def test_v4_geometry_accepts_selected_urdf_from_isolated_asset_root(tmp_path) -> None:
    asset_root = tmp_path / "contract-assets"
    selected = asset_root / DEFAULT_RM65_CAMERA_MOUNT_URDF.relative_to(
        REPO_ROOT / "assets"
    )
    selected.parent.mkdir(parents=True)
    selected.write_bytes(DEFAULT_RM65_CAMERA_MOUNT_URDF.read_bytes())

    geometry = _build_geometry(
        registry_path=DEFAULT_RM65_CAMERA_REGISTRY_V4,
        camera_mount_urdf_path=selected,
        camera_asset_root=asset_root,
    )

    assert tuple(item["role"] for item in geometry.source_catalog()) == (
        "main",
        "wrist_left",
        "wrist_right",
    )


def test_v4_geometry_accepts_asset_root_selected_through_worktree_link(tmp_path) -> None:
    asset_root = tmp_path / "read-only-input"
    selected = asset_root / DEFAULT_RM65_CAMERA_MOUNT_URDF.relative_to(
        REPO_ROOT / "assets"
    )
    selected.parent.mkdir(parents=True)
    selected.write_bytes(DEFAULT_RM65_CAMERA_MOUNT_URDF.read_bytes())
    worktree_assets = tmp_path / "worktree" / "assets"
    worktree_assets.parent.mkdir()
    worktree_assets.symlink_to(asset_root, target_is_directory=True)

    geometry = _build_geometry(
        registry_path=DEFAULT_RM65_CAMERA_REGISTRY_V4,
        camera_mount_urdf_path=(worktree_assets / selected.relative_to(asset_root)),
        camera_asset_root=worktree_assets,
    )

    assert len(geometry.source_catalog()) == 3


def test_v4_geometry_rejects_same_urdf_bytes_outside_selected_asset_root(
    tmp_path,
) -> None:
    selected_root = tmp_path / "selected-assets"
    other_root = tmp_path / "other-assets"
    relative = DEFAULT_RM65_CAMERA_MOUNT_URDF.relative_to(REPO_ROOT / "assets")
    selected = selected_root / relative
    other = other_root / relative
    selected.parent.mkdir(parents=True)
    other.parent.mkdir(parents=True)
    payload = DEFAULT_RM65_CAMERA_MOUNT_URDF.read_bytes()
    selected.write_bytes(payload)
    other.write_bytes(payload)

    with pytest.raises(ValueError, match="does not bind 'front' once"):
        _build_geometry(
            registry_path=DEFAULT_RM65_CAMERA_REGISTRY_V4,
            camera_mount_urdf_path=other,
            camera_asset_root=selected_root,
        )


def test_v4_geometry_rejects_same_basename_at_another_asset_path(tmp_path) -> None:
    asset_root = tmp_path / "assets"
    relative = DEFAULT_RM65_CAMERA_MOUNT_URDF.relative_to(REPO_ROOT / "assets")
    declared = asset_root / relative
    selected = asset_root / "another-bundle" / relative.name
    declared.parent.mkdir(parents=True)
    selected.parent.mkdir(parents=True)
    payload = DEFAULT_RM65_CAMERA_MOUNT_URDF.read_bytes()
    declared.write_bytes(payload)
    selected.write_bytes(payload)

    with pytest.raises(ValueError, match="does not bind 'front' once"):
        _build_geometry(
            registry_path=DEFAULT_RM65_CAMERA_REGISTRY_V4,
            camera_mount_urdf_path=selected,
            camera_asset_root=asset_root,
        )


def test_v4_geometry_rejects_registry_asset_path_escape(tmp_path) -> None:
    registry = json.loads(DEFAULT_RM65_CAMERA_REGISTRY_V4.read_text(encoding="utf-8"))
    placements = registry["cameras"]["front"]["extrinsics"]["frames"][
        "head_camera_frame"
    ]["placements"]
    for placement in placements:
        placement["urdf"] = "assets/../outside/" + Path(placement["urdf"]).name
    changed = tmp_path / "escaping_registry.json"
    changed.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ValueError, match="does not bind 'front' once"):
        _build_geometry(registry_path=changed)


def test_v4_geometry_rejects_registry_symlink_escape_from_asset_root(
    tmp_path,
) -> None:
    asset_root = tmp_path / "assets"
    outside = tmp_path / "outside"
    relative = DEFAULT_RM65_CAMERA_MOUNT_URDF.relative_to(REPO_ROOT / "assets")
    outside_selected = outside / relative
    outside_selected.parent.mkdir(parents=True)
    outside_selected.write_bytes(DEFAULT_RM65_CAMERA_MOUNT_URDF.read_bytes())
    asset_root.mkdir()
    (asset_root / relative.parts[0]).symlink_to(
        outside / relative.parts[0],
        target_is_directory=True,
    )

    with pytest.raises(ValueError, match="does not bind 'front' once"):
        _build_geometry(
            registry_path=DEFAULT_RM65_CAMERA_REGISTRY_V4,
            camera_mount_urdf_path=(asset_root / relative),
            camera_asset_root=asset_root,
        )


def test_v4_geometry_rejects_duplicate_selected_registry_placement(tmp_path) -> None:
    registry = json.loads(DEFAULT_RM65_CAMERA_REGISTRY_V4.read_text(encoding="utf-8"))
    placements = registry["cameras"]["front"]["extrinsics"]["frames"][
        "head_camera_frame"
    ]["placements"]
    selected = next(
        item
        for item in placements
        if item["urdf"] == "assets/" + str(
            DEFAULT_RM65_CAMERA_MOUNT_URDF.relative_to(REPO_ROOT / "assets")
        )
    )
    placements.append(dict(selected))
    changed = tmp_path / "duplicate_registry.json"
    changed.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ValueError, match="does not bind 'front' once"):
        _build_geometry(registry_path=changed)


def test_v4_geometry_keeps_absolute_registry_paths_strict(tmp_path) -> None:
    registry = json.loads(DEFAULT_RM65_CAMERA_REGISTRY_V4.read_text(encoding="utf-8"))
    logical_selected = "assets/" + str(
        DEFAULT_RM65_CAMERA_MOUNT_URDF.relative_to(REPO_ROOT / "assets")
    )
    for camera in registry["cameras"].values():
        for frame in camera["extrinsics"]["frames"].values():
            for placement in frame.get("placements", ()):
                if placement["urdf"] == logical_selected:
                    placement["urdf"] = str(DEFAULT_RM65_CAMERA_MOUNT_URDF.resolve())
    changed = tmp_path / "absolute_registry.json"
    changed.write_text(json.dumps(registry), encoding="utf-8")

    geometry = _build_geometry(registry_path=changed)

    assert len(geometry.source_catalog()) == 3


def test_v4_geometry_rejects_registry_pose_that_differs_from_selected_urdf(
    tmp_path,
) -> None:
    registry = json.loads(DEFAULT_RM65_CAMERA_REGISTRY_V4.read_text(encoding="utf-8"))
    matrix = registry["cameras"]["front"]["extrinsics"]["frames"]
    matrix["head_camera_frame"]["T_camera_to_parent"][2][3] -= 0.05
    changed = tmp_path / "changed_registry.json"
    changed.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ValueError, match="selected URDF and runtime"):
        _build_geometry(registry_path=changed)


def test_v4_geometry_rejects_station_chain_that_differs_from_motion_profile() -> None:
    _arm_bases, station = _scene_geometry()
    changed = np.asarray(station, dtype=float).copy()
    changed[0, 3] += 0.01

    with pytest.raises(ValueError, match="world_from_canonical_station"):
        _build_geometry(
            registry_path=DEFAULT_RM65_CAMERA_REGISTRY_V4,
            station_override=changed,
        )


def test_asset_local_v4_registry_is_not_silently_used_with_deployed_wrist_roles() -> (
    None
):
    asset_registry = (
        REPO_ROOT / "assets/rm65_inspire_gripper_v4/cameras/realsense_registry.json"
    )

    with pytest.raises(ValueError, match="serial differs"):
        _source_binding(asset_registry)


def test_dynamic_wrist_pose_is_unavailable_before_fk_projection_finishes() -> None:
    geometry = _build_geometry(
        registry_path=DEFAULT_RM65_CAMERA_REGISTRY_V4,
        target_builder=SimpleNamespace(
            solver_world_targets_for_qpos_pair=lambda **_kwargs: {
                "left": np.eye(4),
                "right": np.eye(4),
            }
        ),
    )
    geometry.publish_measured_qpos(
        qpos_by_side={"left": np.zeros(6), "right": np.zeros(6)},
        measured_at=10.0,
        available_at=10.2,
        evidence_refs=("rm65-hardware-qpos-cut:test",),
    )

    with pytest.raises(RuntimeError, match="pose_from_future"):
        geometry.pose_cut(
            source_id="left_wrist",
            frame_acquired_at=10.0,
            now=10.19,
        )

    cut = geometry.pose_cut(
        source_id="left_wrist",
        frame_acquired_at=10.0,
        now=10.2,
    )
    assert cut.measured_at == 10.0
    assert cut.available_at == 10.2
