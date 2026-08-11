from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from control_station.sim_hil_profile.human_deflection import (
    HUMAN_DEFLECTION_STATUS_SCHEMA_V1,
    project_admitted_human_deflection,
)
from dataset_contracts.action_schema import ActionPayloadSchema
from robot_motion.action import (
    ActionComponentCatalogV2,
    ComponentAction,
    ComponentActionSchema,
    ControlComponentSpecV2,
)


def _relative_pose_action(
    *,
    translation: tuple[float, float, float],
    rotation_deg: float,
) -> ComponentAction:
    payload = ActionPayloadSchema(
        schema_id="right-arm-relative-pose",
        schema_version="development",
        representation_family="cartesian_pose",
        dtype="float64",
        shape=(7,),
        coordinate_names=("x", "y", "z", "qx", "qy", "qz", "qw"),
        units=("m", "m", "m", "1", "1", "1", "1"),
        value_semantics="relative_delta",
        reference_frame="right_arm_base",
        controlled_frame="right_tool_tcp",
        semantic_parameters={
            "translation_application": "reference_frame_additive",
            "rotation_application": "baseline_local_right_multiply",
        },
    )
    catalog = ActionComponentCatalogV2(
        catalog_id="human-deflection-test",
        catalog_version="development",
        components=(
            ControlComponentSpecV2(
                component_id="right_arm",
                controlled_entity_id="right_arm",
                command_channel_id="right_arm",
                selection_group_id="right_arm",
                resource_claims=("resource:right_arm",),
                accepted_payload_schema_hashes=(payload.semantic_content_hash,),
            ),
        ),
    )
    schema = ComponentActionSchema(
        schema_id="human-deflection-test-relative",
        schema_version="development",
        component_catalog=catalog,
        payload_schema_by_component={"right_arm": payload},
    )
    quaternion = Rotation.from_euler(
        "z",
        rotation_deg,
        degrees=True,
    ).as_quat()
    return ComponentAction(
        action_schema=schema,
        values_by_component={
            "right_arm": [*translation, *quaternion],
        },
    )


def _cycle(action: ComponentAction, *, status: str = "batch_admitted"):
    pair = SimpleNamespace(
        residual_action=action,
        sampled_at=10.0,
        source_sample_id="human-emission:7",
        source_sample_hash="human-emission-hash",
        anchor_receipt_id="anchor-receipt:2",
        anchor_receipt_hash="anchor-receipt-hash",
        anchor_request_generation=2,
    )
    return SimpleNamespace(
        publication=SimpleNamespace(
            input_publication=SimpleNamespace(
                status=status,
                evidence_pair=pair if status == "batch_admitted" else None,
            )
        )
    )


def test_projects_exact_admitted_relative_pose_from_active_anchor() -> None:
    action = _relative_pose_action(
        translation=(0.03, -0.04, 0.12),
        rotation_deg=90.0,
    )

    result = project_admitted_human_deflection(_cycle(action), now=10.25)

    assert result["schema"] == HUMAN_DEFLECTION_STATUS_SCHEMA_V1
    assert result["status"] == "available"
    assert result["anchor_request_generation"] == 2
    assert result["age_sec"] == pytest.approx(0.25)
    assert len(result["components"]) == 1
    component = result["components"][0]
    assert component["component_id"] == "right_arm"
    assert component["translation_xyz_m"] == pytest.approx(
        [0.03, -0.04, 0.12]
    )
    assert component["translation_distance_m"] == pytest.approx(0.13)
    assert component["translation_direction_xyz"] == pytest.approx(
        [3.0 / 13.0, -4.0 / 13.0, 12.0 / 13.0]
    )
    assert component["rotation_rotvec_rad"] == pytest.approx(
        [0.0, 0.0, math.pi / 2.0]
    )
    assert component["rotation_angle_rad"] == pytest.approx(math.pi / 2.0)
    assert component["rotation_axis_xyz"] == pytest.approx([0.0, 0.0, 1.0])
    assert component["translation_reference_frame"] == "right_arm_base"
    assert component["rotation_controlled_frame"] == "right_tool_tcp"
    assert component["rotation_application"] == (
        "baseline_local_right_multiply"
    )


def test_identity_relative_pose_renders_at_home_center() -> None:
    action = _relative_pose_action(
        translation=(0.0, 0.0, 0.0),
        rotation_deg=0.0,
    )

    component = project_admitted_human_deflection(
        _cycle(action),
        now=10.0,
    )["components"][0]

    assert component["translation_distance_m"] == 0.0
    assert component["translation_direction_xyz"] == [0.0, 0.0, 0.0]
    assert component["rotation_angle_rad"] == 0.0
    assert component["rotation_axis_xyz"] == [0.0, 0.0, 0.0]


def test_unadmitted_or_missing_relative_input_is_not_visualized() -> None:
    action = _relative_pose_action(
        translation=(0.1, 0.0, 0.0),
        rotation_deg=0.0,
    )

    rejected = project_admitted_human_deflection(
        _cycle(action, status="batch_rejected"),
        now=10.0,
    )
    missing = project_admitted_human_deflection(None, now=10.0)

    assert rejected["status"] == "waiting"
    assert rejected["components"] == []
    assert rejected["input_publication_status"] == "batch_rejected"
    assert missing["status"] == "waiting"
    assert missing["components"] == []
