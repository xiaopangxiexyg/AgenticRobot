"""Causal MuJoCo state sampling for deployable policy nodes."""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.runtime.control_admission import ControlAdmissionCoordinator
from robot_motion.runtime.identity import RuntimeFactIdentity
from robot_motion.runtime.observation_cut import (
    ObservationBundle,
    ObservationCutRecord,
    ObservationSampleRef,
)

from ..adapters import MujocoPositionExecutionRig
from .measurement import capture_mujoco_qpos_measurement


MUJOCO_QPOS_STREAM_ID_V1 = "robot_state.arm_qpos"
MUJOCO_QPOS_OBSERVATION_CONTRACT_HASH_V1 = canonical_content_hash(
    {
        "contract_id": "dexteleop.mujoco.qpos_observation.v1",
        "stream_id": MUJOCO_QPOS_STREAM_ID_V1,
        "payload": "arm_qpos_by_side",
        "measured_schema": "float64[side,6]",
    },
    domain="dexteleop.policy_observation_contract.v1",
)
MUJOCO_QPOS_POLICY_IO_LAYOUT_HASH_V1 = canonical_content_hash(
    {
        "layout_id": "dexteleop.mujoco.qpos_policy_io.v1",
        "input_streams": (MUJOCO_QPOS_STREAM_ID_V1,),
        "output_action_space": "bimanual_qpos_v1",
    },
    domain="dexteleop.policy_io_layout.v1",
)


class MujocoPolicyObservationAdapter:
    def __init__(
        self,
        rig: MujocoPositionExecutionRig,
        *,
        policy_node_id: str,
        policy_node_epoch: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not str(policy_node_id).strip() or int(policy_node_epoch) < 0:
            raise ValueError("policy node identity is invalid.")
        self.rig = rig
        self.policy_node_id = str(policy_node_id)
        self.policy_node_epoch = int(policy_node_epoch)
        self.clock = clock
        self._sequence = 0

    def capture(
        self,
        coordinator: ControlAdmissionCoordinator,
        *,
        trigger_reason: str = "policy_schedule",
    ) -> ObservationBundle:
        measured_at = float(self.clock())
        measurement = capture_mujoco_qpos_measurement(
            self.rig,
            sides=self.rig.config.active_sides,
            clock_domain_id=coordinator.clock_domain.clock_domain_id,
            measured_at=measured_at,
        )
        # Preserve the established policy payload representation while the
        # backend-owned measurement itself remains immutable.
        qpos_by_side = {
            side: np.asarray(qpos, dtype=float)
            for side, qpos in measurement.qpos_by_side.items()
        }
        payload_hash = canonical_content_hash(
            qpos_by_side,
            domain="dexteleop.mujoco.qpos_sample.v1",
        )
        identity = RuntimeFactIdentity(
            runtime_session_id=coordinator.clock_domain.runtime_session_id,
            control_domain_id=coordinator.control_domain_id,
            producer_id=f"{self.policy_node_id}:state",
            producer_epoch=self.policy_node_epoch,
            sequence=self._sequence,
            namespace="sensor_sample",
        )
        cutoff_at = float(self.clock())
        capture = coordinator.capture_cut(cutoff_at=cutoff_at)
        sample = ObservationSampleRef(
            stream_id=MUJOCO_QPOS_STREAM_ID_V1,
            sample_id=identity.fact_id,
            sequence=self._sequence,
            measured_at=measured_at,
            measured_clock_domain_id=coordinator.clock_domain.clock_domain_id,
            available_at=cutoff_at,
            available_clock_domain_id=coordinator.clock_domain.clock_domain_id,
            payload_object_id=f"mujoco-qpos:{payload_hash}",
            control_admission_sequence=capture.cut.cutoff_admission_sequence,
            content_metadata={
                "schema": "dexteleop.joint_measurement.v1",
                "sides": self.rig.config.active_sides,
            },
        )
        record = ObservationCutRecord.build(
            runtime_session_id=coordinator.clock_domain.runtime_session_id,
            control_cut=capture.cut,
            policy_node_id=self.policy_node_id,
            policy_node_epoch=self.policy_node_epoch,
            sequence=self._sequence,
            built_at=cutoff_at,
            built_clock_domain_id=coordinator.clock_domain.clock_domain_id,
            selected_sample_by_stream={MUJOCO_QPOS_STREAM_ID_V1: sample},
            observation_contract_hash=MUJOCO_QPOS_OBSERVATION_CONTRACT_HASH_V1,
            policy_io_layout_hash=MUJOCO_QPOS_POLICY_IO_LAYOUT_HASH_V1,
            trigger_reason=trigger_reason,
        )
        self._sequence += 1
        return ObservationBundle(
            record=record,
            payload_handles_by_stream={
                MUJOCO_QPOS_STREAM_ID_V1: {
                    "arm_qpos_by_side": qpos_by_side,
                }
            },
        )


__all__ = [
    "MUJOCO_QPOS_OBSERVATION_CONTRACT_HASH_V1",
    "MUJOCO_QPOS_POLICY_IO_LAYOUT_HASH_V1",
    "MUJOCO_QPOS_STREAM_ID_V1",
    "MujocoPolicyObservationAdapter",
]
