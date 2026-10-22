"""Left fixed-session visual consumers use passive Skills and managed Artifacts."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from agent_gateway.profiles.mujoco.motion_factory import (
    MujocoAgentArtifactStorageConfig,
    build_mujoco_agent_tcp_pose_gateway_session,
    mujoco_agent_tcp_pose_execution_observation_plan_v1,
)
from project_paths import resolve_path
from sim.mujoco.deploy import MujocoSimHILTargetConfig
from tests.test_agent_gateway_mujoco_tcp_pose_comparison import CLIENT, TASKS, _request

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("task", TASKS)
def test_left_rgb_and_depth_artifacts_do_not_advance_physics_or_dispatch(task, tmp_path, record_property):
    facts = []
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    session = build_mujoco_agent_tcp_pose_gateway_session(
        target_config=MujocoSimHILTargetConfig(
            task_config_path=resolve_path(Path(task)), realtime_worker=False,
            runtime_session_id="test:left-visual",
            control_domain_id="test:left-visual",
            execution_observation_plan=mujoco_agent_tcp_pose_execution_observation_plan_v1(motion_side="left"),
            exposed_pose_sides=("left",), allow_scripted_task_reset=False,
        ), allowed_client_ids=(CLIENT,), motion_side="left", trace_sink=facts.append,
        artifact_storage=MujocoAgentArtifactStorageConfig(root=artifact_root),
    )
    evidence = []
    try:
        app = session.application
        assert app.motion_side == "left" and app.component_id == "left_arm"
        app.start_observation_ready()
        readonly = session.wire_adapter(authenticated_client_id=CLIENT, read_only_access=True)
        for sequence, skill, parameters, executor in (
            (1, "perception.capture_rgb", {"selected_roles": ["main"], "width": 32,
                "height": 24, "max_total_bytes": 65536}, session.rgb_executor),
            (2, "perception.capture_depth", {"role": "main"}, session.depth_executor),
        ):
            route = session.contract_bundle.route_manifest.route(skill, "2").route
            assert route.control_effect_class == "read_only" and route.privilege_tier == "t0_read_only"
            assert not route.may_move_robot and not route.may_open_publication
            qpos, qvel, sim_time, trace_count = app.scene.data.qpos.copy(), app.scene.data.qvel.copy(), float(app.scene.data.time), len(facts)
            request = _request(session, readonly, skill_id=skill, version="2", parameters=parameters, sequence=sequence)
            admitted = readonly.submit_read_only_skill_request(request.to_wire())
            assert admitted["accepted"], admitted
            assert executor is not None and executor.wait_all(timeout_sec=20)
            operation = session.service.operation_store.get(admitted["operation"]["operation_id"])
            assert operation.status == "completed" and operation.execution_outcome == "completed", operation
            assert operation.evaluation_status == "not_requested" and operation.task_outcome == "not_assessed"
            assert len(operation.output_artifact_bindings) == 1
            binding = operation.output_artifact_bindings[0]
            managed = session.artifact_store.managed_read_path(binding.artifact.artifact_id)
            assert managed.is_relative_to(artifact_root)
            payload = managed.read_bytes()
            assert payload
            if skill.endswith("rgb"):
                with Image.open(io.BytesIO(payload)) as image:
                    assert image.size == (32, 24) and image.format == "JPEG"
                    image.verify()
            else:
                depth = np.load(io.BytesIO(payload), allow_pickle=False)
                assert depth.ndim == 2 and depth.dtype == np.float32 and np.isfinite(depth).all()
                assert depth.shape == (operation.execution_result["height"], operation.execution_result["width"])
            np.testing.assert_array_equal(app.scene.data.qpos, qpos)
            np.testing.assert_array_equal(app.scene.data.qvel, qvel)
            assert float(app.scene.data.time) == sim_time and len(facts) == trace_count
            assert session.motion_executor.workers == ()
            evidence.append({"skill": skill, "operation_id": operation.operation_id,
                "artifact_id": binding.artifact.artifact_id, "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload), "physics_and_dispatch_unchanged": True})
    finally:
        assert session.close().accepted
    assert session.application.closed and session._artifact_store_closed
    record_property("left_visual_native_evidence_json", json.dumps({"task": task,
        "motion_side": "left", "artifacts": evidence, "cleanup_completed": True, "hardware_connected": False}))
