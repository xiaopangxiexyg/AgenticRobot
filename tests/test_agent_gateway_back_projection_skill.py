from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from agent_gateway.profiles.back_projection import (
    LiveAlignedDepthCapture,
    LiveBackProjectionProvider,
    build_back_projection_registration,
)
from agent_gateway.skills.perception.back_projection_schemas import (
    validate_back_project_parameters_v1,
)

pytestmark = [pytest.mark.unit]


def _registration():
    return build_back_projection_registration(
        target_kind="sim",
        current_mode="sim_active",
        source_catalog=(
            {
                "source_id": "front",
                "role": "main",
                "native_width": 4,
                "native_height": 3,
            },
        ),
        reference_frame_id="canonical_station",
        reference_frame_spec_hash="frame:station:test",
        frame_graph_hash="frame-graph:test",
        runtime_session_id="session:test",
        control_domain_id="domain:test",
        target_binding_id="target:test",
        station_manifest_hash="station:test",
        execution_manifest_hash="execution:test",
    )


class _CapturePort:
    def __init__(self, depth: np.ndarray) -> None:
        self.depth = depth
        self.calls = []

    def capture_aligned_depth(self, *, role, expected_station_revision):
        self.calls.append((role, expected_station_revision))
        return LiveAlignedDepthCapture(
            source_id="front",
            role=role,
            captured_at=10.0,
            station_revision=expected_station_revision,
            capture_group_id="capture:test",
            depth_content_hash="a" * 64,
            intrinsics={
                "width": 4,
                "height": 3,
                "fx": 2.0,
                "fy": 2.0,
                "cx": 2.0,
                "cy": 1.0,
            },
            reference_from_camera=np.asarray(
                [
                    [1.0, 0.0, 0.0, 0.5],
                    [0.0, 1.0, 0.0, -0.25],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            depth_m=self.depth,
        )


def _request(parameters):
    return SimpleNamespace(parameters=parameters, expected_station_revision=7)


def test_back_project_schema_avoids_provider_one_of_and_validates_exclusive_mode() -> None:
    registration = _registration()
    schema = registration.skill.schema_entry.parameter_schema
    assert "oneOf" not in schema
    assert "anyOf" not in schema
    pixel = {
        "role": "main",
        "target_pixel_uv": [2, 1],
        "pixel_region_xyxy": None,
        "world_z_min_m": None,
        "world_z_max_m": None,
    }
    validate_back_project_parameters_v1(pixel, allowed_roles=("main",))
    with pytest.raises(ValueError, match="exactly one pixel or region"):
        validate_back_project_parameters_v1(
            {**pixel, "pixel_region_xyxy": [0, 0, 4, 3]},
            allowed_roles=("main",),
        )
    with pytest.raises(ValueError, match="supported only in region"):
        validate_back_project_parameters_v1(
            {**pixel, "world_z_min_m": 0.0},
            allowed_roles=("main",),
        )


def test_pixel_back_projection_returns_compact_station_geometry() -> None:
    registration = _registration()
    port = _CapturePort(np.full((3, 4), 2.0, dtype=np.float32))
    provider = LiveBackProjectionProvider(
        capture_port=port,
        registration=registration,
    )
    parameters = {
        "role": "main",
        "target_pixel_uv": [2, 1],
        "pixel_region_xyxy": None,
        "world_z_min_m": None,
        "world_z_max_m": None,
    }

    result = provider.project_request(_request(parameters))

    assert port.calls == [("main", 7)]
    assert result["coordinate_mode"] == "pixel"
    assert result["world_xyz_m"] == [0.5, -0.25, 3.0]
    assert result["valid_point_count"] == 1
    assert result["reference_frame_id"] == "canonical_station"
    registration.payload_validator.validate_execution_result(result)
    registration.payload_validator.validate_execution_result_for_parameters(
        parameters, result
    )


def test_region_back_projection_filters_station_z() -> None:
    registration = _registration()
    depth = np.asarray(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
        ],
        dtype=np.float32,
    )
    provider = LiveBackProjectionProvider(
        capture_port=_CapturePort(depth),
        registration=registration,
    )
    parameters = {
        "role": "main",
        "target_pixel_uv": None,
        "pixel_region_xyxy": [0, 0, 4, 3],
        "world_z_min_m": 1.9,
        "world_z_max_m": 2.1,
    }

    result = provider.project_request(_request(parameters))

    assert result["coordinate_mode"] == "region"
    assert result["valid_point_count"] == 8
    assert result["depth_m"] == 1.0
    assert result["world_xyz_m"] == pytest.approx([0.25, -0.5, 2.0])
    registration.payload_validator.validate_execution_result(result)


def test_back_project_rejects_invalid_depth_and_sparse_region() -> None:
    registration = _registration()
    provider = LiveBackProjectionProvider(
        capture_port=_CapturePort(np.zeros((3, 4), dtype=np.float32)),
        registration=registration,
    )
    with pytest.raises(RuntimeError, match="pixel_depth_invalid"):
        provider.project_request(
            _request(
                {
                    "role": "main",
                    "target_pixel_uv": [1, 1],
                    "pixel_region_xyxy": None,
                    "world_z_min_m": None,
                    "world_z_max_m": None,
                }
            )
        )
    with pytest.raises(RuntimeError, match="region_insufficient_points"):
        provider.project_request(
            _request(
                {
                    "role": "main",
                    "target_pixel_uv": None,
                    "pixel_region_xyxy": [0, 0, 4, 3],
                    "world_z_min_m": None,
                    "world_z_max_m": None,
                }
            )
        )
