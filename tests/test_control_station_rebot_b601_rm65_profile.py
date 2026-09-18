from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from control_station.composition.rebot_b601_rm65_profile import (
    B601_RM65_FRAME_MAPPING_PROFILE_OFF,
    B601RM65FrameMappingProfile,
    load_b601_rm65_frame_mapping_profile,
)
from robot_motion.action import (
    ActionComponentCatalogV2,
    ActionPayloadSchema,
    ComponentActionSchema,
    ControlComponentSpecV2,
)
from teleop_runtime.action.human_residual import HumanPoseSample


def _schema(
    *,
    reference_frame: str = "right_arm_base",
    controlled_frame: str = "tool_tcp",
    value_semantics: str = "relative_delta",
) -> ComponentActionSchema:
    payload = ActionPayloadSchema(
        schema_id="rm65-right-pose-residual",
        schema_version="development",
        representation_family="cartesian_pose",
        dtype="float64",
        shape=(7,),
        coordinate_names=("x", "y", "z", "qx", "qy", "qz", "qw"),
        units=("m", "m", "m", "1", "1", "1", "1"),
        value_semantics=value_semantics,
        reference_frame=reference_frame,
        controlled_frame=controlled_frame,
        semantic_parameters={
            "translation_application": "reference_frame_additive",
            "rotation_application": "baseline_local_right_multiply",
        },
    )
    catalog = ActionComponentCatalogV2(
        catalog_id="b601-rm65-frame-profile-test",
        catalog_version="development",
        components=(
            ControlComponentSpecV2(
                component_id="right_arm",
                controlled_entity_id="rm65-right",
                command_channel_id="rm65-right-arm",
                selection_group_id="arms",
                resource_claims=("resource:rm65-right-arm",),
                accepted_payload_schema_hashes=(payload.semantic_content_hash,),
            ),
        ),
    )
    return ComponentActionSchema(
        schema_id="rm65-right-pose-residual",
        schema_version="development",
        component_catalog=catalog,
        payload_schema_by_component={"right_arm": payload},
    )


def _profile(**overrides: object) -> B601RM65FrameMappingProfile:
    reference = np.eye(4, dtype=np.float64)
    reference[:3, :3] = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()
    values: dict[str, object] = {
        "profile_id": "b601-rm65-lab-calibration-test-v1",
        "enabled": True,
        "calibration_source_id": "synthetic-offline-calibration-fixture",
        "calibration_source_sha256": "a" * 64,
        "matrix_target_reference_source_reference": reference,
        "matrix_source_controlled_target_controlled": np.eye(4),
        "translation_gain_target_reference_xyz": (2.0, 0.5, 1.0),
        "rotation_gain": 1.0,
    }
    values.update(overrides)
    return B601RM65FrameMappingProfile(**values)


def _sample(
    sample_id: str,
    measured_at: float,
    x: float,
    *,
    rotation: np.ndarray | None = None,
) -> HumanPoseSample:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = x
    if rotation is not None:
        pose[:3, :3] = rotation
    return HumanPoseSample(
        component_id="right_arm",
        sample_id=sample_id,
        measured_at=measured_at,
        measured_clock_domain_id="test.monotonic",
        source_reference_frame_id="base_link",
        source_controlled_frame_id="gripper_end",
        source_coordinate_binding_id="synthetic-b601-pose",
        source_coordinate_binding_hash="synthetic-b601-pose-hash",
        pose_source_reference_source_controlled=pose,
    )


def test_default_profile_is_not_a_runnable_identity_calibration() -> None:
    profile = B601RM65FrameMappingProfile()

    assert profile.profile_id == B601_RM65_FRAME_MAPPING_PROFILE_OFF
    assert not profile.enabled
    assert profile.matrix_target_reference_source_reference is None
    assert profile.matrix_source_controlled_target_controlled is None
    with pytest.raises(RuntimeError, match="profile is disabled"):
        profile.build_frame_graph()
    with pytest.raises(RuntimeError, match="profile is disabled"):
        profile.build_residual_adapter(_schema())


def test_enabled_profile_reuses_canonical_frame_and_residual_math() -> None:
    profile = _profile()
    adapter = profile.build_residual_adapter(_schema())
    anchor = adapter.capture_anchor(
        (_sample("anchor", 1.0, 0.0),),
        anchor_generation=3,
        captured_at=1.0,
        captured_clock_domain_id="test.monotonic",
    )

    result = adapter.adapt(
        (_sample("current", 1.1, 0.1),),
        anchor=anchor,
        adapted_at=1.1,
        adapted_clock_domain_id="test.monotonic",
    )

    value = np.asarray(result.relative_action.value("right_arm").values)
    np.testing.assert_allclose(value[:3], (0.0, 0.05, 0.0), atol=1e-12)
    np.testing.assert_allclose(value[3:], (0.0, 0.0, 0.0, 1.0), atol=1e-12)
    assert result.receipt.frame_graph_hash == adapter.frame_graph.semantic_content_hash
    assert result.receipt.translation_gain_by_component["right_arm"] == (
        2.0,
        0.5,
        1.0,
    )


def test_controlled_frame_transform_maps_source_rotation_axis() -> None:
    controlled = np.eye(4, dtype=np.float64)
    controlled[:3, :3] = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()
    profile = _profile(
        matrix_target_reference_source_reference=np.eye(4),
        matrix_source_controlled_target_controlled=controlled,
        translation_gain_target_reference_xyz=(1.0, 1.0, 1.0),
    )
    adapter = profile.build_residual_adapter(_schema())
    anchor = adapter.capture_anchor(
        (_sample("anchor", 1.0, 0.0),),
        anchor_generation=1,
        captured_at=1.0,
        captured_clock_domain_id="test.monotonic",
    )
    source_rotation = Rotation.from_rotvec((0.1, 0.0, 0.0)).as_matrix()

    result = adapter.adapt(
        (_sample("current", 1.1, 0.0, rotation=source_rotation),),
        anchor=anchor,
        adapted_at=1.1,
        adapted_clock_domain_id="test.monotonic",
    )

    value = np.asarray(result.relative_action.value("right_arm").values)
    np.testing.assert_allclose(
        Rotation.from_quat(value[3:]).as_rotvec(),
        (0.0, -0.1, 0.0),
        atol=1e-12,
    )


def test_tracked_v3_tool_tcp_profile_maps_b601_tool_axes_to_rm65_tool_axes() -> None:
    profile_path = (
        Path(__file__).resolve().parents[1]
        / "docs/hardware/profiles/b601_rm65_mapping_150_v2.json"
    )
    profile = load_b601_rm65_frame_mapping_profile(profile_path)
    assert profile.profile_id == "b601-rm65-provisional-lab-150-v3-tool-tcp"
    expected_controlled = np.array(
        (
            (0.0, -1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    np.testing.assert_array_equal(
        profile.matrix_source_controlled_target_controlled,
        expected_controlled,
    )
    assert (
        profile.semantic_content_hash
        == "f602cf4f7fa3b4a3ff3f46f85dec5592ae6e95fef98c4c2f810f88120b56dc4a"
    )
    assert profile.translation_gain_target_reference_xyz == (1.5, 1.5, 1.5)

    adapter = profile.build_residual_adapter(_schema())
    anchor = adapter.capture_anchor(
        (_sample("anchor-v3", 1.0, 0.0),),
        anchor_generation=1,
        captured_at=1.0,
        captured_clock_domain_id="test.monotonic",
    )
    expected_target_rotvec_by_source_axis = {
        (0.1, 0.0, 0.0): (0.0, -0.1, 0.0),
        (0.0, 0.1, 0.0): (0.1, 0.0, 0.0),
        (0.0, 0.0, 0.1): (0.0, 0.0, 0.1),
    }

    for index, (source_rotvec, expected_target_rotvec) in enumerate(
        expected_target_rotvec_by_source_axis.items(),
        start=1,
    ):
        result = adapter.adapt(
            (
                _sample(
                    f"current-v3-{index}",
                    1.0 + index / 10.0,
                    0.0,
                    rotation=Rotation.from_rotvec(source_rotvec).as_matrix(),
                ),
            ),
            anchor=anchor,
            adapted_at=1.0 + index / 10.0,
            adapted_clock_domain_id="test.monotonic",
        )
        value = np.asarray(result.relative_action.value("right_arm").values)
        np.testing.assert_allclose(
            Rotation.from_quat(value[3:]).as_rotvec(),
            expected_target_rotvec,
            atol=1e-12,
        )


def test_v3_profile_matches_rm65_mujoco_tool_tcp_zero_pose() -> None:
    """Derive the canonical tool_tcp mapping from the checked-in MuJoCo model.

    The visible ``right_realman`` assembly is intentionally not used as the
    controlled frame. This regression keeps the profile tied to the same
    ``right_arm_link6``/``tool_tcp`` body that the target projector and IK
    runtime measure.
    """
    pytest.importorskip("mujoco")
    urdf_path = (
        Path(__file__).resolve().parents[1]
        / "assets/rm65_inspire_gripper_v3"
        / "rm65_dual_arm_wrist_cameras_v3_head_camera.urdf"
    )
    if not urdf_path.is_file():
        pytest.skip("RM65 MuJoCo URDF is not available in this checkout")

    from sim.mujoco.scene import MujocoScene

    scene = MujocoScene.from_urdf(
        urdf_path,
        camera_names=(),
        timestep=1.0 / 120.0,
        gravity=(0.0, 0.0, 0.0),
    )
    profile = load_b601_rm65_frame_mapping_profile(
        Path(__file__).resolve().parents[1]
        / "docs/hardware/profiles/b601_rm65_mapping_150_v2.json"
    )

    target_base = scene.body_pose("right_arm_base_link")
    tool_pose = np.linalg.inv(target_base) @ scene.body_pose("right_arm_link6")
    visible_pose = np.linalg.inv(target_base) @ scene.body_pose("right_realman")
    reference_rotation = np.asarray(
        profile.matrix_target_reference_source_reference,
        dtype=np.float64,
    )[:3, :3]
    controlled_rotation = np.asarray(
        profile.matrix_source_controlled_target_controlled,
        dtype=np.float64,
    )[:3, :3]

    # B601 gripper_end is identity at q=0. Therefore the controlled mapping
    # must turn the source identity into the RM65 tool_tcp zero orientation.
    np.testing.assert_allclose(
        reference_rotation @ controlled_rotation,
        tool_pose[:3, :3],
        atol=1e-5,
    )
    # The visible flange is a separate Rz(-90 deg) assembly frame; it must not
    # silently replace the canonical tool_tcp/link6 orientation.
    assert not np.allclose(visible_pose[:3, :3], tool_pose[:3, :3], atol=1e-5)
    np.testing.assert_allclose(
        visible_pose[:3, :3],
        Rotation.from_euler("z", 90.0, degrees=True).as_matrix(),
        atol=1e-5,
    )


def test_payload_roundtrip_and_file_loader_verify_semantic_hash(tmp_path) -> None:
    profile = _profile()
    payload = profile.to_payload()
    path = tmp_path / "b601_rm65_mapping.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    restored = load_b601_rm65_frame_mapping_profile(path)

    assert restored.semantic_content_hash == profile.semantic_content_hash
    np.testing.assert_array_equal(
        restored.matrix_target_reference_source_reference,
        profile.matrix_target_reference_source_reference,
    )
    payload["translation_gain_target_reference_xyz"] = [1.0, 1.0, 1.0]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="semantic_content_hash mismatch"):
        load_b601_rm65_frame_mapping_profile(path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"calibration_source_sha256": "not-a-sha"}, "64-character"),
        (
            {"matrix_target_reference_source_reference": np.zeros((4, 4))},
            "numeric rigid 4x4 pose",
        ),
        (
            {"matrix_target_reference_source_reference": np.ones(7)},
            "numeric rigid 4x4 pose",
        ),
        ({"translation_gain_target_reference_xyz": (1.0, 0.0, 1.0)}, "positive"),
        ({"rotation_gain": 0.5}, "rotation_gain must be 1.0"),
    ),
)
def test_enabled_profile_rejects_ambiguous_or_unsupported_calibration(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _profile(**overrides)


def test_enabled_profile_requires_calibration_identity_and_both_transforms() -> None:
    with pytest.raises(ValueError, match="calibration source identity"):
        _profile(calibration_source_id=None)
    with pytest.raises(ValueError, match="both rigid frame transforms"):
        _profile(matrix_source_controlled_target_controlled=None)


def test_adapter_binding_rejects_wrong_rm65_schema_semantics_or_frames() -> None:
    profile = _profile()
    with pytest.raises(ValueError, match="frames do not match"):
        profile.build_residual_adapter(_schema(reference_frame="rm65_base"))
    with pytest.raises(ValueError, match="relative-delta Cartesian pose"):
        profile.build_residual_adapter(_schema(value_semantics="absolute_target"))


def test_payload_loader_rejects_unknown_or_missing_fields() -> None:
    payload = _profile().to_payload()
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="Unknown .* fields"):
        B601RM65FrameMappingProfile.from_payload(payload)

    payload = _profile().to_payload()
    del payload["calibration_source_id"]
    with pytest.raises(ValueError, match="Missing .* fields"):
        B601RM65FrameMappingProfile.from_payload(payload)

    payload = _profile().to_payload()
    payload["profile_id"] = None
    with pytest.raises(TypeError, match="profile_id must be a string"):
        B601RM65FrameMappingProfile.from_payload(payload)
