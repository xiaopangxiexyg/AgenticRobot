"""Causal policy observation view over existing collection-owned pollers."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from typing import Any, Callable

import numpy as np

from robot_motion.deploy import (
    DeploymentObservation,
    NormalizedAction,
    build_deployment_observation,
)
from robot_motion.deploy.schema import ACTION_DIM


class RealTeleopPolicyObservationCapture:
    """Reuse collection snapshots; never perform a synchronous device read."""

    def __init__(
        self,
        *,
        components: object,
        dataset_recorder: object | Callable[[], object | None],
        camera_sources: Mapping[str, Any],
        require_images: bool,
        pose_frame_contract: object | None,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        active_sides = tuple(str(side) for side in components.active_sides)
        recorder_provider = (
            dataset_recorder
            if callable(dataset_recorder)
            else lambda: dataset_recorder
        )
        initial_recorder = recorder_provider()
        arm_state_poller = getattr(initial_recorder, "arm_state_poller", None)
        if arm_state_poller is None:
            raise RuntimeError(
                "HITL policy inference requires the collection arm-state poller."
            )
        if not callable(getattr(arm_state_poller, "latest_by_side", None)):
            raise TypeError("collection arm-state poller lacks latest_by_side().")
        hand_backends = {
            "left": getattr(components, "left_hand_backend", None),
            "right": getattr(components, "right_hand_backend", None),
        }
        end_effector_sides = tuple(
            side
            for side in active_sides
            if hand_backends.get(side) is not None
        )
        hand_state_poller = getattr(initial_recorder, "hand_state_poller", None)
        if end_effector_sides and hand_state_poller is None:
            raise RuntimeError(
                "HITL policy inference requires the collection hand-state poller "
                "when an end effector is active."
            )
        if not callable(wall_clock):
            raise TypeError("wall_clock must be callable.")
        self.components = components
        self.dataset_recorder_provider = recorder_provider
        self.camera_sources = {
            str(key): value for key, value in dict(camera_sources).items()
        }
        self.require_images = bool(require_images)
        self.pose_frame_contract = pose_frame_contract
        self.wall_clock = wall_clock
        self.active_sides = active_sides
        self.end_effector_sides = end_effector_sides
        self._last_action = np.full((ACTION_DIM,), np.nan, dtype=np.float32)
        self._lock = threading.RLock()

    def _current_pollers(self) -> tuple[object, object | None]:
        recorder = self.dataset_recorder_provider()
        if recorder is None:
            raise RuntimeError(
                "HITL policy inference has no current formal dataset recorder."
            )
        arm_state_poller = getattr(recorder, "arm_state_poller", None)
        if arm_state_poller is None or not callable(
            getattr(arm_state_poller, "latest_by_side", None)
        ):
            raise RuntimeError(
                "Current formal dataset recorder has no causal arm-state poller."
            )
        hand_state_poller = getattr(recorder, "hand_state_poller", None)
        if self.end_effector_sides and hand_state_poller is None:
            raise RuntimeError(
                "Current formal dataset recorder has no causal hand-state poller."
            )
        return arm_state_poller, hand_state_poller

    def reset(self) -> None:
        with self._lock:
            self._last_action = np.full((ACTION_DIM,), np.nan, dtype=np.float32)

    def observe_last_action(self, action: NormalizedAction) -> None:
        if not isinstance(action, NormalizedAction):
            raise TypeError("last action must be NormalizedAction.")
        vector = np.asarray(action.vector, dtype=np.float32).reshape(-1)
        if vector.shape != (ACTION_DIM,):
            raise ValueError(
                f"last policy action must have shape {(ACTION_DIM,)}, got {vector.shape}."
            )
        with self._lock:
            self._last_action = vector.copy()

    @staticmethod
    def _deployment_arm_state(snapshot: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "current_qpos": snapshot.get("actual_qpos"),
            "actual_qpos": snapshot.get("actual_qpos"),
            "current_tcp_pose": snapshot.get("actual_tcp_pose"),
            "actual_tcp_pose": snapshot.get("actual_tcp_pose"),
            "hardware_result_code": snapshot.get("actual_hardware_result_code"),
            "hardware_measurement_timestamp": snapshot.get(
                "actual_state_timestamp"
            ),
            "actual_state_stale": snapshot.get("actual_state_stale"),
            "causal_snapshot_contract": snapshot.get(
                "causal_snapshot_contract"
            ),
        }

    def __call__(
        self,
        *,
        step_index: int | None = None,
    ) -> DeploymentObservation:
        now = float(self.wall_clock())
        arm_state_poller, hand_state_poller = self._current_pollers()
        readiness = getattr(arm_state_poller, "readiness_by_side", None)
        if callable(readiness):
            reasons = readiness(
                required_sides=self.active_sides,
                sample_timestamp=now,
            )
            blocked = {
                side: reason for side, reason in dict(reasons).items() if reason
            }
            if blocked:
                raise RuntimeError(
                    "causal arm feedback is not ready for policy inference: "
                    + ",".join(
                        f"{side}={reason}" for side, reason in sorted(blocked.items())
                    )
                )
        raw_arm_states = arm_state_poller.latest_by_side(
            sample_timestamp=now
        )
        missing = sorted(set(self.active_sides) - set(raw_arm_states))
        if missing:
            raise RuntimeError(
                "causal arm feedback missing for policy inference: "
                + ",".join(missing)
            )
        arm_states = {
            side: self._deployment_arm_state(raw_arm_states[side])
            for side in self.active_sides
        }
        hand_states = (
            {}
            if hand_state_poller is None
            else hand_state_poller.latest_by_side(sample_timestamp=now)
        )
        with self._lock:
            last_action = self._last_action.copy()
        metadata: dict[str, Any] = {
            "observation_source": "real_teleop_collection_causal_snapshot",
            "causal_snapshot_contract": (
                "source_timestamp<=T0_and_available_timestamp<=T0"
            ),
        }
        pose_contract = self.pose_frame_contract
        to_dict = getattr(pose_contract, "to_dict", None)
        if callable(to_dict):
            metadata["pose_frame_contract"] = to_dict()
        return build_deployment_observation(
            active_sides=self.active_sides,
            arms_enabled=True,
            hands_enabled=bool(self.end_effector_sides),
            end_effector_sides=self.end_effector_sides,
            step_index=0 if step_index is None else int(step_index),
            arm_states=arm_states,
            hand_states=hand_states,
            camera_sources=self.camera_sources,
            last_action=last_action,
            timestamp=now,
            require_images=self.require_images,
            hand_state_max_age_sec=(
                None
                if not self.end_effector_sides
                else float(
                    getattr(
                        getattr(hand_state_poller, "config", None),
                        "stale_after_sec",
                        0.2,
                    )
                )
            ),
            metadata=metadata,
        )


__all__ = ["RealTeleopPolicyObservationCapture"]
