from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.embodiment import (
    CaptureProjectionComponentV2,
    CaptureProjectionProfileV2,
    EmbodimentCaptureProfile,
    EmbodimentComponentCaptureSpec,
)
from dataset_contracts.episode_identity import EpisodeCaptureIdentity
from dataset_contracts.v22 import EndEffectorSpec, PolicyIOLayout
from dataset_contracts.v23 import DEXTELEOP_DATASET_SCHEMA_V23
from dataset_contracts.v25p_native import (
    V25pComponentStreamProfile,
    V25pComponentStreamSpec,
)
from dataset_contracts.v25p_projection import V25pCompatibilityProjectionReceipt
from dataset_interface.v25p_dual_write import compare_projected_v23_row
from dataset_interface.v25p_native import load_v25p_native_episode
from dataset_interface.v25p_v23_adapter import project_native_episode_to_v23_rows
from teleop_runtime.collection.operator_labels import OperatorLabelStateMachine
from teleop_runtime.collection.sample_assembler import CollectorSample
from teleop_runtime.collection.schema_v22 import collector_sample_to_v22_row
from teleop_runtime.collection.umi.row_encoder import UmiV23RowEncoder
from teleop_runtime.collection.umi.types import UmiPoseMeasurement, UmiSample
from teleop_runtime.collection.umi.v25p_producer import UmiV25pProducer
from teleop_runtime.collection.v25p_capture_profiles import (
    V25P_REAL_TELEOP_SOURCE,
    V25P_UMI_SOURCE,
    build_policy_io_v25p_capture_profiles,
)
from collection_runtime.v25p_native_store import V25pNativeEpisodeStore
from collection_runtime.v25p_producer import V25pProducerBindings
from teleop_runtime.collection.v25p_real_teleop_producer import (
    RealTeleopV25pProducer,
)


def _hash(label: str) -> str:
    return canonical_content_hash({"label": label}, domain="test.v25p.producer.v1")


def _layout() -> PolicyIOLayout:
    return PolicyIOLayout(
        name="test_dual_scalar_gripper",
        end_effectors={
            side: EndEffectorSpec.continuous(
                side,
                adapter="test_scalar_v1",
                target_dim=1,
                target_names=("command",),
                target_source_indices=(0,),
                state_names=("measured",),
                state_source_field="gripper_measured_scalar",
                units="normalized",
            )
            for side in ("left", "right")
        },
    )


def _profiles(*, source: str, layout: PolicyIOLayout):
    component_ids = {
        f"pose.{side}": f"{source}.pose.{side}"
        for side in ("left", "right")
    } | {
        f"end_effector.{side}": f"{source}.end_effector.{side}"
        for side in ("left", "right")
    }
    action_hash = {
        key: _hash(f"{source}:{key}:action") for key in component_ids
    }
    state_hash = {
        key: _hash(f"{source}:{key}:state") for key in component_ids
    }
    embodiment = EmbodimentCaptureProfile(
        profile_id=f"{source}_test_embodiment",
        profile_version="1",
        topology_id=f"{source}_dual_pose_gripper",
        component_catalog_contract_id=f"{source}.component_catalog.test.v1",
        component_catalog_contract_hash=_hash(f"{source}:catalog"),
        target_capability_contract_id=f"{source}.target_capability.test.v1",
        target_capability_contract_hash=_hash(f"{source}:capability"),
        components=tuple(
            EmbodimentComponentCaptureSpec(
                component_id=component_id,
                component_kind=("pose_source" if key.startswith("pose") else "gripper"),
                controlled_entity_id=component_id,
                command_payload_schema_hash=action_hash[key],
                measurement_payload_schema_hash=state_hash[key],
            )
            for key, component_id in component_ids.items()
        ),
    )
    streams = []
    binding_map = {}
    projection_components = []
    for side in ("left", "right"):
        for kind, shape, names, units, representation in (
            (
                "pose",
                (7,),
                ("x", "y", "z", "qx", "qy", "qz", "qw"),
                ("m", "m", "m", "1", "1", "1", "1"),
                "xyz_quat_xyzw",
            ),
            ("end_effector", (1,), ("value",), ("normalized",), "scalar"),
        ):
            key = f"{kind}.{side}"
            component_id = component_ids[key]
            for role in ("action", "state"):
                stage = (
                    "next_accepted_observation"
                    if source == "umi" and role == "action" and kind == "pose"
                    else "accepted_reference"
                    if role == "action"
                    else "causal_measured"
                )
                stream_id = f"{source}.{role}.{kind}.{side}"
                streams.append(
                    V25pComponentStreamSpec(
                        stream_id=stream_id,
                        component_id=component_id,
                        dataset_role=role,
                        value_stage=stage,
                        payload_schema_hash=(
                            action_hash[key] if role == "action" else state_hash[key]
                        ),
                        representation_family=representation,
                        dtype="float32",
                        shape=shape,
                        coordinate_names=names,
                        units=units,
                    )
                )
                binding_map[f"{role}.{kind}.{side}"] = stream_id
                if kind == "pose":
                    field = f"{role}.{side}_arm.pose" if role == "action" else f"observation.{side}_arm.pose"
                else:
                    field = (
                        f"action.end_effector.{side}.target"
                        if role == "action"
                        else f"observation.end_effector.{side}.state"
                    )
                projection_components.append(
                    CaptureProjectionComponentV2(
                        stream_id=stream_id,
                        dataset_field=field,
                    )
                )
    stream_profile = V25pComponentStreamProfile(
        profile_id=f"{source}_component_streams",
        profile_version="1",
        embodiment_profile_hash=embodiment.semantic_content_hash,
        streams=tuple(streams),
    )
    projection = CaptureProjectionProfileV2(
        projection_id=f"{source}_v23_test",
        projection_version="1",
        temporal_alignment_contract_id=f"{source}.temporal.test.v1",
        temporal_alignment_contract_hash=_hash(f"{source}:temporal"),
        row_encoder_contract_id=f"{source}.row_encoder.test.v1",
        row_encoder_contract_hash=_hash(f"{source}:row_encoder"),
        policy_io_layout_hash=layout.layout_hash.removeprefix("sha256:"),
        pose_frame_contract_hash=_hash(f"{source}:pose_frames"),
        causal_snapshot_contract_hash=_hash(f"{source}:causal_snapshot"),
        components=tuple(projection_components),
    )
    return embodiment, stream_profile, projection, V25pProducerBindings(binding_map)


def _persist_and_project(
    tmp_path: Path,
    *,
    source: str,
    embodiment,
    stream_profile,
    projection,
    produced,
):
    identity = EpisodeCaptureIdentity(
        episode_uuid=f"{source}-episode",
        legacy_episode_index=0,
        lifecycle_owner_id=f"test.{source}.owner",
        prepared_at=0.0,
        prepared_clock_domain_id=produced.cut_clock_domain_id,
    )
    store = V25pNativeEpisodeStore(
        dataset_root=tmp_path,
        dataset_id=f"{source}-dataset",
        embodiment_profile=embodiment,
        component_stream_profile=stream_profile,
        compatibility_projection_profiles=(projection,),
    )
    store.prepare_episode(identity)
    store.bind_prepared_episode(
        identity,
        bound_at=0.0,
        bound_clock_domain_id=produced.cut_clock_domain_id,
    )
    store.append_sample(
        identity,
        sample_index=produced.sample_index,
        cut_at=produced.cut_at,
        cut_clock_domain_id=produced.cut_clock_domain_id,
        values_by_stream=produced.values_by_stream,
    )
    seal = store.seal_native_episode(
        identity,
        sealed_at=produced.cut_at + 1.0,
        sealed_clock_domain_id=produced.cut_clock_domain_id,
    )
    receipt = V25pCompatibilityProjectionReceipt(
        projection_id=f"{source}-offline-shadow",
        episode_uuid=identity.episode_uuid,
        legacy_episode_index=0,
        episode_identity_hash=identity.semantic_content_hash,
        native_capture_manifest_hash=seal.native_capture_manifest_hash,
        projection_profile_hash=projection.semantic_content_hash,
        target_dataset_schema=DEXTELEOP_DATASET_SCHEMA_V23,
        status="failed",
        completed_at=produced.cut_at + 1.1,
        completed_clock_domain_id=produced.cut_clock_domain_id,
        message="offline equivalence test does not publish a legacy episode receipt",
    )
    store.commit_episode(
        identity,
        committed_at=produced.cut_at + 1.2,
        committed_clock_domain_id=produced.cut_clock_domain_id,
        compatibility_projection_receipts=(receipt,),
    )
    loaded = load_v25p_native_episode(tmp_path, identity.episode_uuid)
    return project_native_episode_to_v23_rows(loaded, projection).rows[0]


def _real_sample(
    *, frame_index: int = 0, timestamp: float = 10.0, offset: float = 0.0
) -> CollectorSample:
    left = np.asarray(
        [0.1 + offset, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
    )
    right = np.asarray(
        [0.4 + offset, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
    )
    action = np.full(26, np.nan, dtype=np.float32)
    action[:7], action[7:14], action[14], action[20] = left, right, 0.25, 0.75
    state = np.full(38, np.nan, dtype=np.float32)
    state[:6], state[6:13] = np.arange(6), left
    state[13:19], state[19:25], state[25:32] = np.arange(6) + 10, np.arange(6) + 20, right
    return CollectorSample(
        index=frame_index,
        episode_index=0,
        frame_index=frame_index,
        timestamp=timestamp,
        session_state="RUNNING",
        control_hz=15.0,
        action=action,
        observation_state=state,
        observation_last_action=action.copy(),
        arm_state_source="causal_measured_hardware",
        active_sides=("left", "right"),
        hand_states={
            "left": {
                "gripper_measured_scalar": 0.2,
                "joint_timestamp": timestamp - 0.1,
            },
            "right": {
                "gripper_measured_scalar": 0.8,
                "joint_timestamp": timestamp - 0.1,
            },
        },
    )


def _umi_sample(*, frame_index: int = 0, timestamp: float = 100.0) -> UmiSample:
    poses = {
        side: UmiPoseMeasurement(
            side=side,
            raw_tcp_pose7=np.asarray([index, 0, 0, 0, 0, 0, 1], dtype=float),
            desk_tcp_pose7=np.asarray([index + 10, 0, 0, 0, 0, 0, 1], dtype=float),
            source_timestamp=timestamp - 0.1,
            available_timestamp=timestamp - 0.05,
            source_clock="vive_calibrated",
            available_clock="local_wall_time",
            sequence=frame_index * 100 + index,
            tracker_name=f"tracker_{index}",
            tracker_role=f"{side}_wrist",
            age_sec=0.1,
        )
        for index, side in enumerate(("left", "right"), start=1)
    }
    next_poses = {
        side: replace(
            pose,
            raw_tcp_pose7=pose.raw_tcp_pose7 + np.asarray([0.5, 0, 0, 0, 0, 0, 0]),
            source_timestamp=timestamp + 0.05,
            available_timestamp=timestamp + 0.08,
            sequence=pose.sequence + 10,
        )
        for side, pose in poses.items()
    }
    return UmiSample(
        episode_index=0,
        frame_index=frame_index,
        timestamp=timestamp,
        poses=poses,
        arm_action_poses=next_poses,
        arm_action_source_frame_index=frame_index + 1,
        arm_action_source_timestamp=timestamp + 0.1,
        gripper_commands={"left": 0.2, "right": 0.8},
        hand_states={
            "left": {
                "gripper_measured_scalar": 0.25,
                "joint_timestamp": timestamp - 0.1,
                "gripper_measurement_timestamp": timestamp - 0.1,
                "telemetry_snapshot_available_timestamp": timestamp - 0.05,
                "telemetry_snapshot_source_clock": "local_wall_time",
                "telemetry_snapshot_available_clock": "local_wall_time",
            },
            "right": {
                "gripper_measured_scalar": 0.75,
                "joint_timestamp": timestamp - 0.1,
                "gripper_measurement_timestamp": timestamp - 0.1,
                "telemetry_snapshot_available_timestamp": timestamp - 0.05,
                "telemetry_snapshot_source_clock": "local_wall_time",
                "telemetry_snapshot_available_clock": "local_wall_time",
            },
        },
        operator_labels=OperatorLabelStateMachine().snapshot(),
    )


def test_real_teleop_native_producer_matches_existing_v23_encoder(tmp_path: Path) -> None:
    layout = _layout()
    embodiment, streams, projection, bindings = _profiles(
        source="real", layout=layout
    )
    sample = _real_sample()
    legacy_row, _ = collector_sample_to_v22_row(sample, layout=layout, strict=True)
    produced = RealTeleopV25pProducer(
        layout=layout,
        stream_profile=streams,
        bindings=bindings,
    ).produce(sample)
    projected = _persist_and_project(
        tmp_path,
        source="real",
        embodiment=embodiment,
        stream_profile=streams,
        projection=projection,
        produced=produced,
    )
    report = compare_projected_v23_row(projected, legacy_row)
    assert report.equivalent, {
        key: value for key, value in report.fields.items() if not value.equivalent
    }
    assert len(report.fields) == 9


def test_producer_bindings_fail_closed_on_missing_stream_or_wrong_action_stage() -> None:
    layout = _layout()
    _embodiment, streams, _projection, bindings = _profiles(
        source="real", layout=layout
    )
    incomplete = dict(bindings.stream_id_by_value_key)
    incomplete.pop("state.pose.right")
    with pytest.raises(ValueError, match="missing"):
        RealTeleopV25pProducer(
            layout=layout,
            stream_profile=streams,
            bindings=V25pProducerBindings(incomplete),
        )
    with pytest.raises(ValueError, match="role/stage mismatch"):
        UmiV25pProducer(stream_profile=streams, bindings=bindings)


def test_umi_native_producer_matches_existing_v23_encoder_and_detects_drift(
    tmp_path: Path,
) -> None:
    layout = _layout()
    embodiment, streams, projection, bindings = _profiles(
        source="umi", layout=layout
    )
    sample = _umi_sample()
    legacy_row, _ = UmiV23RowEncoder().encode_row(
        sample,
        layout=layout,
        camera_video_paths=None,
        tactile_storage_refs=None,
        inline_tactile=False,
    )
    produced = UmiV25pProducer(
        stream_profile=streams,
        bindings=bindings,
    ).produce(sample)
    projected = _persist_and_project(
        tmp_path,
        source="umi",
        embodiment=embodiment,
        stream_profile=streams,
        projection=projection,
        produced=produced,
    )
    report = compare_projected_v23_row(projected, legacy_row)
    assert report.equivalent, {
        key: value for key, value in report.fields.items() if not value.equivalent
    }

    drifted = dict(legacy_row)
    drifted["action.end_effector.left.target"] = [0.21]
    mismatch = compare_projected_v23_row(projected, drifted)
    assert not mismatch.equivalent
    assert mismatch.fields["action.end_effector.left.target"].reason == "value_mismatch"
    assert mismatch.fields["action.end_effector.left.target"].max_absolute_error > 0.009

    wrong_sample = dict(legacy_row, frame_index=1)
    identity_mismatch = compare_projected_v23_row(projected, wrong_sample)
    assert identity_mismatch.fields["__sample_index__"].reason == "sample_index_mismatch"

    wrong_shape = dict(legacy_row)
    wrong_shape["action.end_effector.left.target"] = [0.2, 0.2]
    shape_mismatch = compare_projected_v23_row(projected, wrong_shape)
    assert shape_mismatch.fields["action.end_effector.left.target"].reason == "shape_mismatch"

    missing = dict(legacy_row)
    missing.pop("observation.right_arm.pose")
    missing_report = compare_projected_v23_row(projected, missing)
    assert missing_report.fields["observation.right_arm.pose"].reason == "legacy_field_missing"


@pytest.mark.parametrize(
    ("source", "source_id"),
    (
        ("real_v2", V25P_REAL_TELEOP_SOURCE),
        ("umi_v2", V25P_UMI_SOURCE),
    ),
)
def test_builtin_canonical_component_action_producers_round_trip_exact_v23(
    tmp_path: Path,
    source: str,
    source_id: str,
) -> None:
    """Class-like hardware test: real source objects, no device initialization."""

    layout = _layout()
    bundle = build_policy_io_v25p_capture_profiles(
        source=source_id,
        layout=layout,
        pose_frame_contract_hash=_hash(f"{source}:pose_frames"),
    )
    if source_id == V25P_REAL_TELEOP_SOURCE:
        sample = _real_sample()
        legacy_row, _ = collector_sample_to_v22_row(
            sample,
            layout=layout,
            strict=True,
        )
        produced = RealTeleopV25pProducer(
            layout=layout,
            stream_profile=bundle.streams,
            bindings=bundle.producer_bindings,
            component_action_schema=bundle.component_action_schema,
            component_action_projection=bundle.component_action_projection,
        ).produce(sample)
    else:
        sample = _umi_sample()
        legacy_row, _ = UmiV23RowEncoder().encode_row(
            sample,
            layout=layout,
            camera_video_paths=None,
            tactile_storage_refs=None,
            inline_tactile=False,
        )
        produced = UmiV25pProducer(
            stream_profile=bundle.streams,
            bindings=bundle.producer_bindings,
            component_action_schema=bundle.component_action_schema,
            component_action_projection=bundle.component_action_projection,
        ).produce(sample)
        for side in ("left", "right"):
            action_pose = produced.values_by_stream[f"umi.action.pose.{side}"]
            assert action_pose.effective_at == sample.arm_action_poses[side].source_timestamp
            assert action_pose.available_at == sample.arm_action_poses[side].available_timestamp
            assert action_pose.effective_clock_domain_id == "vive_calibrated"
            assert action_pose.available_clock_domain_id == "local_wall_time"

    projected = _persist_and_project(
        tmp_path / source,
        source=source,
        embodiment=bundle.embodiment,
        stream_profile=bundle.streams,
        projection=bundle.v23_projection,
        produced=produced,
    )
    report = compare_projected_v23_row(projected, legacy_row)
    assert report.equivalent, {
        key: value for key, value in report.fields.items() if not value.equivalent
    }
    action_stream_ids = {
        stream.stream_id
        for stream in bundle.streams.streams
        if stream.dataset_role == "action"
    }
    assert action_stream_ids == set(
        bundle.component_action_projection.stream_id_by_action_component.values()
    )


@pytest.mark.parametrize("source", ("real", "umi"))
def test_multi_frame_native_episode_remains_equivalent_to_legacy_rows(
    tmp_path: Path,
    source: str,
) -> None:
    layout = _layout()
    embodiment, streams, projection, bindings = _profiles(
        source=source, layout=layout
    )
    root = tmp_path / source
    identity = EpisodeCaptureIdentity(
        episode_uuid=f"{source}-multi-episode",
        legacy_episode_index=0,
        lifecycle_owner_id=f"test.{source}.owner",
        prepared_at=0.0,
        prepared_clock_domain_id="local_wall_time",
    )
    store = V25pNativeEpisodeStore(
        dataset_root=root,
        dataset_id=f"{source}-multi-dataset",
        embodiment_profile=embodiment,
        component_stream_profile=streams,
        compatibility_projection_profiles=(projection,),
    )
    store.prepare_episode(identity)
    store.bind_prepared_episode(
        identity,
        bound_at=0.0,
        bound_clock_domain_id="local_wall_time",
    )
    legacy_rows = []
    if source == "real":
        producer = RealTeleopV25pProducer(
            layout=layout, stream_profile=streams, bindings=bindings
        )
        encoder = None
    else:
        producer = UmiV25pProducer(stream_profile=streams, bindings=bindings)
        encoder = UmiV23RowEncoder()
    last_cut = 0.0
    for frame_index in range(24):
        if source == "real":
            sample = _real_sample(
                frame_index=frame_index,
                timestamp=10.0 + frame_index / 15.0,
                offset=frame_index * 0.001,
            )
            legacy_row, _ = collector_sample_to_v22_row(
                sample, layout=layout, strict=True
            )
        else:
            sample = _umi_sample(
                frame_index=frame_index,
                timestamp=100.0 + frame_index / 15.0,
            )
            assert encoder is not None
            legacy_row, _ = encoder.encode_row(
                sample,
                layout=layout,
                camera_video_paths=None,
                tactile_storage_refs=None,
                inline_tactile=False,
            )
        produced = producer.produce(sample)
        store.append_sample(
            identity,
            sample_index=produced.sample_index,
            cut_at=produced.cut_at,
            cut_clock_domain_id=produced.cut_clock_domain_id,
            values_by_stream=produced.values_by_stream,
        )
        legacy_rows.append(legacy_row)
        last_cut = produced.cut_at

    seal = store.seal_native_episode(
        identity,
        sealed_at=last_cut + 1.0,
        sealed_clock_domain_id="local_wall_time",
    )
    receipt = V25pCompatibilityProjectionReceipt(
        projection_id=f"{source}-multi-offline-shadow",
        episode_uuid=identity.episode_uuid,
        legacy_episode_index=0,
        episode_identity_hash=identity.semantic_content_hash,
        native_capture_manifest_hash=seal.native_capture_manifest_hash,
        projection_profile_hash=projection.semantic_content_hash,
        target_dataset_schema=DEXTELEOP_DATASET_SCHEMA_V23,
        status="failed",
        completed_at=last_cut + 1.1,
        completed_clock_domain_id="local_wall_time",
        message="offline multi-frame parity does not publish compatibility output",
    )
    store.commit_episode(
        identity,
        committed_at=last_cut + 1.2,
        committed_clock_domain_id="local_wall_time",
        compatibility_projection_receipts=(receipt,),
    )
    loaded = load_v25p_native_episode(root, identity.episode_uuid)
    projected = project_native_episode_to_v23_rows(loaded, projection)
    assert len(projected.rows) == len(legacy_rows) == 24
    reports = tuple(
        compare_projected_v23_row(native_row, legacy_row)
        for native_row, legacy_row in zip(projected.rows, legacy_rows)
    )
    assert all(report.equivalent for report in reports), tuple(
        (report.sample_index, tuple(k for k, v in report.fields.items() if not v.equivalent))
        for report in reports
        if not report.equivalent
    )
