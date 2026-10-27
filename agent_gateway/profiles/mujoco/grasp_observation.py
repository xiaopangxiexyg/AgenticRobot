"""Frozen native RGB-D publication through the existing session Artifact owner."""

from __future__ import annotations

import hashlib
import io

import numpy as np
from PIL import Image

from agent_gateway.protocol.artifact_contracts import AgentArtifactBinding
from agent_gateway.runtime.artifact_producer import (
    artifact_ref_for_bytes,
    commit_artifact_bytes,
)
from agent_gateway.runtime.passive_read_executor import (
    PassiveReadResult,
    PassiveReadSkillExecutor,
)
from dataset_contracts.grasp_multiview_snapshot import (
    GRASP_MULTIVIEW_ENCODING,
    GRASP_MULTIVIEW_SCHEMA_ID,
    encode_grasp_multiview_snapshot,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.grasp_rgbd_snapshot import (
    GRASP_RGBD_SNAPSHOT_SCHEMA_ID,
    encode_grasp_rgbd_snapshot,
)
from dataset_contracts.perception_artifacts import (
    RGB_FRAME_COMPRESSION_V1,
    RGB_FRAME_ENCODING_V1,
    RGB_FRAME_SCHEMA_ID_V1,
)
from robot_motion.relative_action import xyzquat_to_pose_matrix
from sim.mujoco.deploy.target_measurement import MUJOCO_SIM_HIL_WORLD_FRAME_ID_V1


def frozen_grasp_scene_hash(state):
    """Bind the private scene without publishing simulator object coordinates."""
    if (set(state) != {"model_qpos", "model_qvel", "simulation_time_sec"}
            or not np.isfinite(float(state["simulation_time_sec"]))
            or not all(np.isfinite(np.asarray(state[key], dtype=float)).all()
                       for key in ("model_qpos", "model_qvel"))):
        raise ValueError("Frozen grasp scene state must be finite and complete.")
    return canonical_content_hash(state, domain="dexteleop.mujoco.frozen_grasp_scene.v1")


def build_mujoco_grasp_observation_executor(
    *,
    application,
    contract,
    artifact_store,
    current_station_revision,
    clock,
    max_retained_operations,
):
    """Capture once in the session's renderer; a call never starts/stops resources."""
    session_id = application.runtime.config.runtime_session_id
    profile = contract.execution_contract.motion_profile
    sources = contract.source_by_role

    def commit(data, *, slot, ordinal, schema, encoding, media, compression, operation):
        ref = artifact_ref_for_bytes(
            data,
            media_type=media,
            schema_id=schema,
            encoding=encoding,
            compression=compression,
        )
        ref = commit_artifact_bytes(
            store=artifact_store,
            data=data,
            artifact=ref,
            producer_id="agent_gateway.mujoco.grasp_observation",
            runtime_session_id=session_id,
            idempotency_key=f"grasp-observation:{operation.operation_id}:{slot}:{ordinal}",
            now=float(clock()),
        )
        return AgentArtifactBinding(
            slot_id=slot,
            ordinal=ordinal,
            artifact=ref,
            production_receipt_ref="grasp-observation:" + operation.operation_id,
        )

    def read(request, operation):
        contract.observation.payload_validator.validate_parameters(request.parameters)
        if current_station_revision() != request.expected_station_revision:
            raise RuntimeError("grasp_observation_station_changed")
        roles = tuple(request.parameters["selected_roles"])
        cut = application.capture_frozen_rgbd_cut(
            source_ids=tuple(sources[r] for r in roles)
        )
        state = {
            "model_qpos": cut.model_qpos.tolist(),
            "model_qvel": cut.model_qvel.tolist(),
            "simulation_time_sec": cut.simulation_time_sec,
        }
        # The optical conversion remains owned by scene.camera_metadata. World
        # -> public Station uses the already registered frame graph exactly once.
        views, previews = [], []
        graph = application.target_frame_profile.frame_graph
        reference_from_world = graph.resolve(
            MUJOCO_SIM_HIL_WORLD_FRAME_ID_V1, profile.reference_frame_id
        ).matrix_target_source
        for role, view in zip(roles, cut.views):
            world_from_camera = xyzquat_to_pose_matrix(
                view.optical_camera_pose_world_xyz_quat_xyzw
            )
            reference_from_camera = reference_from_world @ world_from_camera
            # The v1 codec explicitly admits 0..10 metric metres. Native sky/
            # far-plane values are invalid observations, not clamped obstacles.
            depth = view.depth_m.copy()
            invalid = (depth <= 0) | (depth > 10)
            depth[invalid] = 0
            metadata = {
                "schema_id": GRASP_RGBD_SNAPSHOT_SCHEMA_ID,
                "role": role,
                "source_id": view.view_id,
                "acquired_at": cut.captured_at,
                "station_revision": request.expected_station_revision,
                "frame_binding_hash": contract.frame_binding_hash,
                "reference_frame_id": profile.reference_frame_id,
                "camera_intrinsics": dict(view.intrinsics),
                "T_reference_from_camera": reference_from_camera.tolist(),
                "T_world_from_camera": world_from_camera.tolist(),
                "depth_valid_range_m": [0, 10],
                "invalid_depth_value_m": 0,
                "invalid_depth_pixel_count": int(np.count_nonzero(invalid)),
                "stationary_only": True,
                "target_mask_available": False,
                "capture_kind": "mujoco_atomic_native_rgbd",
            }
            views.append(
                encode_grasp_rgbd_snapshot(
                    rgb=view.rgb,
                    depth_m=depth,
                    target_mask=np.zeros(depth.shape, dtype=bool),
                    metadata=metadata,
                    qpos_by_side=cut.arm_qpos_by_side,
                )
            )
            stream = io.BytesIO()
            Image.fromarray(view.rgb).save(stream, format="JPEG", quality=95)
            previews.append((role, view, stream.getvalue()))
        observation = encode_grasp_multiview_snapshot(
            views=views,
            metadata={
                "schema_id": GRASP_MULTIVIEW_SCHEMA_ID,
                "runtime_session_id": session_id,
                "station_revision": request.expected_station_revision,
                "frame_binding_hash": contract.frame_binding_hash,
                "excluded_views": [],
                "scope": "stationary_observation_not_a_validated_plan",
                "native_scene_state_hash": frozen_grasp_scene_hash(state),
            },
        )
        if (
            current_station_revision() != request.expected_station_revision
            or application.capture_grasp_scene_state() != state
            or not application.held
            or application.active_operation_id is not None
        ):
            raise RuntimeError("grasp_observation_scene_changed")
        outputs = [
            commit(
                observation,
                slot="grasp_observation",
                ordinal=0,
                schema=GRASP_MULTIVIEW_SCHEMA_ID,
                encoding=GRASP_MULTIVIEW_ENCODING,
                media="application/zip",
                compression="zip",
                operation=operation,
            )
        ]
        frames = []
        for ordinal, (role, view, data) in enumerate(previews):
            outputs.append(
                commit(
                    data,
                    slot="rgb_frames",
                    ordinal=ordinal,
                    schema=RGB_FRAME_SCHEMA_ID_V1,
                    encoding=RGB_FRAME_ENCODING_V1,
                    media="image/jpeg",
                    compression=RGB_FRAME_COMPRESSION_V1,
                    operation=operation,
                )
            )
            frames.append(
                {
                    "role": role,
                    "source_id": view.view_id,
                    "ordinal": ordinal,
                    "width": view.rgb.shape[1],
                    "height": view.rgb.shape[0],
                    "media_type": "image/jpeg",
                    "media_content_hash": hashlib.sha256(data).hexdigest(),
                    "acquired_at": cut.captured_at,
                }
            )
        return PassiveReadResult(
            execution_result={
                "station_revision": request.expected_station_revision,
                "frames": frames,
                "excluded_views": [],
            },
            output_artifact_bindings=tuple(outputs),
            evidence_refs=("grasp-observation:" + operation.operation_id,),
            reason_code="passive_grasp_observation_retained",
        )

    return PassiveReadSkillExecutor(
        port_id="agent-gateway:mujoco-grasp-observation:" + session_id,
        executor_id=contract.observation.executor_id,
        read=read,
        current_station_revision=current_station_revision,
        clock=clock,
        max_retained_operations=max_retained_operations,
    )
