"""Formal-dataset recording adapter for the real-teleop frontend.

This module owns the recorder lifecycle that bridges an already-built
``RealTeleopComponents`` aggregate to the collection writer stack.  It does
not build robot, hand, tracker, or camera devices and does not dispatch motion.
"""

from __future__ import annotations

import argparse
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dataset_contracts.capture_modalities import CaptureModalityContract
from dataset_interface.capture_modalities import bind_capture_modality_contract
from dataset_interface.gripper_force import (
    bind_gripper_force_contract,
    read_gripper_force_contract,
)
from robot_motion.action import HandActionSpec
from robot_motion.runtime import (
    LifecycleStartupError,
    collect_lifecycle_failures,
    run_best_effort_cleanup,
)

from ..collection.arm_state_polling import (
    ArmHardwareStatePoller,
    ArmHardwareStatePollerConfig,
)
from ..collection.async_writer import AsyncWriterConfig, AsyncWriterPool
from ..collection.build_collect_components import CollectorComponents
from ..collection.dataset_episode_maintenance import delete_episode_and_compact
from ..collection.hand_state_polling import (
    HandStateHistoryPoller,
    HandStateHistoryPollerConfig,
)
from ..collection.observers import CollectionWriterObserverConfig
from ..collection.policy_io_runtime import resolve_collection_policy_io_layout
from ..collection.process_writer import (
    ProcessCollectionConfig,
    ProcessCollectionWorker,
    camera_shared_slot_bytes,
)
from ..collection.sample_assembler import (
    DEFAULT_CAMERA_FRAME_MAX_AGE_SEC,
    CollectorSampleBuilder,
    CollectorSampleBuilderConfig,
)
from ..collection.writer_runtime import CollectionWriterRuntime
from ..collection.writer_v21 import LeRobotV21Writer, LeRobotV21WriterConfig
from ..control_trace.episode import EpisodeTraceRecorderSink
from ..sources.camera import RealSenseRGBSource
from .action_debug_payload import _build_action_debug_metadata
from .collection_bridge import (
    collection_backends_from_components as _collection_backends_from_components,
    collection_sources_from_components as _collection_sources_from_components,
)
from .components import RealTeleopComponents
from .control_profile import (
    arm_control_mode as _arm_control_mode_arg,
    effective_arm_control_hz as _effective_arm_control_hz,
    effective_outer_control_hz as _effective_outer_control_hz,
    effective_position_reference_hz as _effective_position_reference_hz,
)
from .dataset_finalize import RealDatasetFinalizeTask
from .debug_timing import debug_timing_metadata as _debug_timing_metadata
from .gripper_policy import (
    current_gripper_force_dataset_contract as _current_gripper_force_dataset_contract,
    gripper_force_dataset_contract_from_args as _gripper_force_dataset_contract_from_args,
    gripper_force_dataset_metadata as _gripper_force_dataset_metadata,
)
from .hand_config import dmtac_sides as _dmtac_sides
from .movement import movement_multiple_value as _movement_multiple_value
from .runtime_metadata import (
    arm_execution_profile_metadata as _arm_execution_profile_metadata,
    position_ik_iteration_metadata as _position_ik_iteration_metadata,
)
from .wrist_depth_freshness import (
    DEFAULT_WRIST_DEPTH_FPS,
    DEFAULT_WRIST_DEPTH_MAX_M,
    DEFAULT_WRIST_DEPTH_MIN_M,
    DEFAULT_WRIST_DEPTH_SAVE_FPS,
    DEFAULT_WRIST_DEPTH_STREAM_FPS,
    WRIST_DEPTH_CAMERA_KEYS,
)


def _collection_hand_action_specs(
    args: argparse.Namespace,
    active_sides: Sequence[str],
) -> Mapping[str, HandActionSpec] | None:
    dmtac_sides = _dmtac_sides(args, active_sides)
    if not dmtac_sides:
        return None
    return {side: HandActionSpec.dmtac_gripper(side) for side in dmtac_sides}


class RealTeleopDatasetRecorder:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        components: RealTeleopComponents,
        camera_sources: Mapping[str, RealSenseRGBSource],
    ) -> None:
        self.args = args
        self.components = components
        self.dataset_root = Path(args.dataset_root)
        configured_force_contract = _gripper_force_dataset_contract_from_args(args)
        persisted_force_contract = read_gripper_force_contract(self.dataset_root)
        if persisted_force_contract is not None:
            current_force_contract = _current_gripper_force_dataset_contract(
                args, components.active_sides
            )
            if current_force_contract is None or current_force_contract != persisted_force_contract:
                raise ValueError(
                    "dataset gripper force profile differs from the selected task/profile: "
                    f"existing={persisted_force_contract.to_dict()}, "
                    f"current={None if current_force_contract is None else current_force_contract.to_dict()}"
                )
            if configured_force_contract is not None and configured_force_contract != persisted_force_contract:
                raise ValueError(
                    "requested gripper force profile differs from the existing dataset profile"
                )
        self.gripper_force_contract = persisted_force_contract or configured_force_contract
        self.capture_modality_contract = CaptureModalityContract(
            ignored_camera_keys=tuple(getattr(args, "ignore_camera", ()) or ()),
            ignored_tactile_keys=tuple(getattr(args, "ignore_tactile", ()) or ()),
        )
        bind_capture_modality_contract(
            self.dataset_root,
            self.capture_modality_contract,
        )
        self.all_camera_sources = {
            str(key): source for key, source in camera_sources.items()
        }
        self.camera_sources = {
            str(key): source
            for key, source in self.all_camera_sources.items()
            if str(key) not in self.capture_modality_contract.ignored_camera_keys
        }
        self.arm_state_poller = self._build_arm_state_poller()
        self.hand_state_poller = self._build_hand_state_poller()
        policy_io_resolution = resolve_collection_policy_io_layout(
            getattr(args, "policy_io_layout", ""),
            active_sides=components.active_sides,
            hand_backends_by_side={
                "left": components.left_hand_backend,
                "right": components.right_hand_backend,
            },
            hand_retargeters_by_side={
                "left": components.left_gripper_retargeter or components.left_hand_retargeter,
                "right": components.right_gripper_retargeter or components.right_hand_retargeter,
            },
            arm_reference_frame_by_side=(
                None
                if args.pose_frame_contract is None
                else {
                    side: (
                        args.pose_frame_contract.canonical_absolute_frame
                        if args.pose_frame_contract.pose_storage_frame == "desk_absolute"
                        else args.pose_frame_contract.source.raw_frame(side)
                    )
                    for side in components.active_sides
                }
            ),
        )
        writer_config = LeRobotV21WriterConfig(
            root_dir=self.dataset_root,
            nominal_fps=float(args.hz),
            task_id=str(args.collection_task_id),
            task_description=str(args.collection_task_description),
            dataset_name=str(args.dataset_name),
            write_parquet=bool(args.write_parquet),
            write_video=bool(args.write_video and not args.no_camera),
            tactile_storage=str(args.tactile_storage),
            tactile_compression=str(getattr(args, "tactile_compression", "none")),
            allow_missing_video_frames=int(args.storage_queue_size) > 0,
            write_wrist_depth_mask=bool(getattr(args, "wrist_depth_mask", False)),
            write_wrist_depth_raw=bool(getattr(args, "wrist_depth_raw", False)),
            hand_action_specs=_collection_hand_action_specs(args, components.active_sides),
            policy_io_layout=policy_io_resolution.layout,
            policy_io_layout_provenance=policy_io_resolution.provenance,
            strict_v22_policy_io=bool(policy_io_resolution.layout is not None),
            pose_frame_contract=args.pose_frame_contract,
            wrist_depth_camera_keys=WRIST_DEPTH_CAMERA_KEYS,
            wrist_depth_min_m=float(
                getattr(args, "wrist_depth_min_m", DEFAULT_WRIST_DEPTH_MIN_M)
            ),
            wrist_depth_max_m=float(
                getattr(args, "wrist_depth_max_m", DEFAULT_WRIST_DEPTH_MAX_M)
            ),
            dataset_metadata=_gripper_force_dataset_metadata(
                args,
                components,
                contract=self.gripper_force_contract,
            ),
        )
        self.writer_storage_config = writer_config
        self.collection_worker_mode = str(
            getattr(args, "collection_worker_mode_resolved", "thread") or "thread"
        )
        self.process_worker: ProcessCollectionWorker | None = None
        self.writer_pool: AsyncWriterPool | None = None
        self.writer_runtime: CollectionWriterRuntime | None = None
        self.writer_observer: Any | None = None
        self.collector_components = CollectorComponents(
            session=components.session,
            sources=_collection_sources_from_components(components, self.camera_sources),
            retargeters={},
            backends=_collection_backends_from_components(components),
            rig=components.session.rig,
            metadata={
                "source_entrypoint": "teleop_runtime.debug.run_vive_rm65_real_teleop",
                **_arm_execution_profile_metadata(args),
                **_position_ik_iteration_metadata(args),
                "active_sides": list(components.active_sides),
                "camera_frame_alignment_policy": "latest_before_or_at_sample_timestamp",
                "hand_tactile_alignment_policy": "causal_hand_history_at_T0",
                "hand_state_alignment_policy": (
                    "source_timestamp<=T0_and_available_timestamp<=T0"
                ),
                "causal_snapshot_contract": "dexteleop_causal_snapshot_v1",
                "movement_multiple": _movement_multiple_value(args),
                "sample_hz": float(args.hz),
                "collection_worker_mode": self.collection_worker_mode,
                "collection_process_queue_size": int(
                    getattr(args, "collection_process_queue_size", 4)
                ),
                "collection_process_shared_slots": int(
                    getattr(args, "collection_process_shared_slots", 4)
                ),
                "collection_process_extra_array_mb": int(
                    getattr(args, "collection_process_extra_array_mb", 8)
                ),
                "impedance_teleop_control_hz": (
                    _effective_outer_control_hz(args)
                    if _arm_control_mode_arg(args) == "impedance"
                    else None
                ),
                "collection_action_source": (
                    "latest_dispatched_main_command"
                    if _arm_control_mode_arg(args) == "impedance"
                    else "causal_position_reference"
                ),
                "position_reference_hz": (
                    _effective_position_reference_hz(args)
                    if _arm_control_mode_arg(args) == "position"
                    else None
                ),
                "position_command_hz": float(_effective_arm_control_hz(args)),
                **_debug_timing_metadata(args),
                "record_actual_wrist_camera_pose": bool(self.arm_state_poller is not None),
                "actual_wrist_camera_pose_hz": float(
                    getattr(args, "actual_wrist_camera_pose_hz", 30.0)
                ),
                "actual_wrist_camera_pose_max_age_sec": float(
                    getattr(args, "actual_wrist_camera_pose_max_age_sec", 0.2)
                ),
                "wrist_depth_mask": bool(getattr(args, "wrist_depth_mask", False)),
                "wrist_depth_raw": bool(getattr(args, "wrist_depth_raw", False)),
                "wrist_depth_camera_keys": list(WRIST_DEPTH_CAMERA_KEYS),
                "wrist_depth_fps": DEFAULT_WRIST_DEPTH_FPS,
                "wrist_depth_save_fps": DEFAULT_WRIST_DEPTH_SAVE_FPS,
                "wrist_depth_stream_fps": int(getattr(args, "wrist_depth_stream_fps", DEFAULT_WRIST_DEPTH_STREAM_FPS)),
                "wrist_cam_stream_width": getattr(args, "wrist_cam_width", None),
                "wrist_cam_stream_height": getattr(args, "wrist_cam_height", None),
                "wrist_depth_align_to_color": bool(getattr(args, "wrist_depth_align_to_color", True)),
                "wrist_depth_min_m": float(getattr(args, "wrist_depth_min_m", DEFAULT_WRIST_DEPTH_MIN_M)),
                "wrist_depth_max_m": float(getattr(args, "wrist_depth_max_m", DEFAULT_WRIST_DEPTH_MAX_M)),
                "tactile_compression": str(getattr(args, "tactile_compression", "none")),
                "capture_modality_contract": self.capture_modality_contract.to_dict(),
                **_gripper_force_dataset_metadata(
                    args,
                    components,
                    contract=self.gripper_force_contract,
                ),
            },
        )
        # The process worker discovers persisted episode/sample indices during
        # startup.  The frontend builder owns only causal snapshot providers in
        # that mode, so zero placeholders are sufficient until start_episode.
        initial_episode_index = 0
        initial_sample_index = 0
        if self.collection_worker_mode == "thread":
            initial_writer = LeRobotV21Writer(writer_config)
            initial_episode_index = int(initial_writer.next_episode_index)
            initial_sample_index = int(initial_writer.next_sample_index)
            self.writer = initial_writer
        sample_builder_config = CollectorSampleBuilderConfig(
            active_sides=components.active_sides,
            camera_buffers={
                camera_key: source.frame_buffer()
                for camera_key, source in self.camera_sources.items()
            },
            require_camera_frames=bool(args.write_video and not args.no_camera),
            camera_frame_max_age_sec=DEFAULT_CAMERA_FRAME_MAX_AGE_SEC,
            episode_index=initial_episode_index,
            sample_index_start=initial_sample_index,
            tactile_array_snapshot_policy=(
                "always"
                if str(args.tactile_storage).strip().lower().replace("-", "_")
                == "parquet"
                else "on_change"
            ),
            position_reference_max_age_sec=float(
                getattr(args, "position_reference_max_age_sec", 0.2)
            ),
            # State semantics must not depend on the controller mode.  Both
            # force/impedance and position collection use the latest eligible
            # measured hardware snapshot at T0; commanded state remains only
            # in the explicit commanded_* sidecars.
            prefer_causal_measured_arm_state=True,
        )
        self.sample_builder = CollectorSampleBuilder(
            sample_builder_config,
            arm_state_metadata_provider=self._latest_actual_arm_state_metadata,
            hand_state_provider=(
                None
                if self.hand_state_poller is None
                else lambda _components, sample_timestamp: self._collection_hand_states(
                    sample_timestamp=float(sample_timestamp)
                )
            ),
            # The causal history returns detached containers with shared,
            # read-only arrays.  Avoid recopying an unchanged tactile frame at
            # the sample rate in both thread and process collection modes.
            hand_state_provider_returns_isolated=self.hand_state_poller is not None,
            position_reference_provider=(
                None
                if components.position_reference_publisher is None
                else components.position_reference_publisher.reference_for_sample
            ),
        )
        if self.collection_worker_mode == "process":
            self.process_worker = ProcessCollectionWorker(
                writer_config=writer_config,
                sample_builder_config=sample_builder_config,
                config=ProcessCollectionConfig(
                    queue_size=int(getattr(args, "collection_process_queue_size", 4)),
                    shared_slot_count=int(
                        getattr(args, "collection_process_shared_slots", 4)
                    ),
                    shared_slot_bytes=camera_shared_slot_bytes(
                        self.camera_sources,
                        extra_array_bytes=(
                            int(getattr(args, "collection_process_extra_array_mb", 8))
                            * 1024
                            * 1024
                        ),
                    ),
                    worker_cpus=tuple(
                        int(cpu)
                        for cpu in (
                            getattr(args, "collection_worker_cpus", ()) or ()
                        )
                    ),
                    worker_nice=int(getattr(args, "collection_worker_nice", 5)),
                ),
            )
            self.writer = self.process_worker
        else:
            assert isinstance(self.writer, LeRobotV21Writer)
            self.writer_pool = AsyncWriterPool(
                self.writer.handle_job,
                config=AsyncWriterConfig(
                    worker_count=int(args.writer_workers),
                    max_queue_size=int(args.writer_queue_size),
                    put_timeout_sec=0.25,
                    backpressure_policy="block",
                    fail_fast=True,
                ),
                name="real_teleop_dataset_writer",
            )
            self.writer_runtime = CollectionWriterRuntime(
                components=self.collector_components,
                sample_builder=self.sample_builder,
                write_planner=lambda sample, _components: self.writer.plan_sample_jobs(sample),
                writer_pool=self.writer_pool,
                config=CollectionWriterObserverConfig(
                    name="real_teleop_dataset_observer",
                    storage_queue_size=int(args.storage_queue_size),
                ),
            )
            # Keep the established attribute for diagnostics and downstream tools;
            # the shared writer runtime now owns its lifecycle.
            self.writer_observer = self.writer_runtime.writer_observer
        self.active_episode_index: int | None = None
        self.active_started_at: float | None = None
        self.control_trace_episode_sink: EpisodeTraceRecorderSink | None = None
        self._active_control_trace_preparation: Any | None = None
        self.finalize_tasks: list[RealDatasetFinalizeTask] = []
        self.saved_episodes = 0
        self.discarded_episodes = 0
        self.frequency_warning_hz = float(getattr(args, "collection_warning_hz", 15.0))
        self.frequency_window_sec = 3.0
        self.frequency_min_elapsed_sec = 2.0
        self._record_timestamps: deque[float] = deque()
        self._episode_start_monotonic: float | None = None
        self._pause_started_at: float | None = None
        self._pause_started_monotonic: float | None = None
        self._pause_reason: str | None = None
        self._pause_reasons: list[str] = []
        self._pause_started_frame_index: int | None = None
        self._pause_started_step_index: int | None = None
        self._pause_started_last_state: str | None = None
        self._pause_segments: list[dict[str, Any]] = []
        self._paused_accumulated_sec = 0.0
        # Sample assembly and write planning are CPU-heavy enough to miss a
        # teleop/IK deadline. Keep exactly one causal sample request in flight
        # for both modes: force teleop may run at 30 Hz while collection stays
        # at 15 Hz.
        self._async_sample_enabled = True
        self._sample_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="teleop_sample_assembly")
            if self._async_sample_enabled
            else None
        )
        self._sample_future: Any | None = None
        self._sample_async_skipped_busy = 0
        self._sample_process_rejected_busy = 0
        self._sample_overload_reason: str | None = None
        self._sample_modality_reason: str | None = None
        self._process_episode_submitted_count = 0
        self._collection_integrity_reasons: list[str] = []

    def _collection_hand_states(
        self,
        *,
        sample_timestamp: float,
    ) -> dict[str, dict[str, Any]]:
        if self.hand_state_poller is None:
            return {}
        states = self.hand_state_poller.latest_by_side(
            sample_timestamp=float(sample_timestamp)
        )
        ignored = set(self.capture_modality_contract.ignored_tactile_keys)
        if not ignored:
            return states
        scrubbed: dict[str, dict[str, Any]] = {}
        for side, raw_state in states.items():
            state = dict(raw_state)
            if str(side) in ignored:
                state.update(
                    {
                        "tactile": None,
                        "tactile_dict": None,
                        "tactile_timestamp": None,
                        "tactile_frame_ids": {},
                        "tactile_shapes": {},
                        "tactile_devices": [],
                        "tactile_ignored_by_dataset_contract": True,
                    }
                )
            scrubbed[str(side)] = state
        return scrubbed

    def _build_arm_state_poller(self) -> ArmHardwareStatePoller | None:
        if not bool(getattr(self.args, "record_actual_wrist_camera_pose", True)):
            return None
        backends_by_side = {
            "left": self.components.left_state_backend,
            "right": self.components.right_state_backend,
        }
        active_backends = {
            side: backend
            for side, backend in backends_by_side.items()
            if side in self.components.active_sides and backend is not None
        }
        if not active_backends:
            return None
        return ArmHardwareStatePoller(
            backends_by_side=active_backends,
            config=ArmHardwareStatePollerConfig(
                poll_hz=float(getattr(self.args, "actual_wrist_camera_pose_hz", 30.0)),
                stale_after_sec=float(
                    getattr(self.args, "actual_wrist_camera_pose_max_age_sec", 0.2)
                ),
            ),
        )

    def _build_hand_state_poller(self) -> HandStateHistoryPoller | None:
        backends_by_side = {
            "left": self.components.left_hand_backend,
            "right": self.components.right_hand_backend,
        }
        active_backends = {
            side: backend
            for side, backend in backends_by_side.items()
            if side in self.components.active_sides and backend is not None
        }
        if not active_backends:
            return None
        return HandStateHistoryPoller(
            backends_by_side=active_backends,
            config=HandStateHistoryPollerConfig(
                poll_hz=max(60.0, 2.0 * float(self.args.hz)),
            ),
        )

    def _latest_actual_arm_state_metadata(
        self,
        _components: CollectorComponents,
        sample_timestamp: float,
    ) -> Mapping[str, Mapping[str, Any]]:
        if self.arm_state_poller is None:
            return {}
        return self.arm_state_poller.latest_by_side(sample_timestamp=float(sample_timestamp))

    def reconfigure_for_current_args(self) -> None:
        desired_root = Path(self.args.dataset_root)
        desired_contract = CaptureModalityContract(
            ignored_camera_keys=tuple(getattr(self.args, "ignore_camera", ()) or ()),
            ignored_tactile_keys=tuple(getattr(self.args, "ignore_tactile", ()) or ()),
        )
        desired_force_contract = _gripper_force_dataset_contract_from_args(self.args)
        if desired_force_contract is None and desired_root == self.dataset_root:
            desired_force_contract = getattr(self, "gripper_force_contract", None)
        if desired_force_contract is not None:
            current_force_contract = _current_gripper_force_dataset_contract(
                self.args, self.components.active_sides
            )
            if current_force_contract != desired_force_contract:
                raise ValueError(
                    "selected task/profile does not match the dataset gripper force profile: "
                    f"expected={desired_force_contract.to_dict()}, "
                    f"current={None if current_force_contract is None else current_force_contract.to_dict()}"
                )
        if self.configured_for_current_args():
            self.reload_finalized_dataset_state()
            self.write_camera_device_config()
            return
        if self.is_recording:
            raise RuntimeError("Cannot reconfigure dataset recorder while recording.")
        trace_sink = getattr(self, "control_trace_episode_sink", None)
        if trace_sink is not None and desired_root != self.dataset_root:
            raise RuntimeError(
                "Formal ControlTrace capture cannot move dataset_root within one runtime "
                "session; restart the frontend for the new root."
            )
        # Validate/bind before closing the current writer.  A mismatched
        # populated dataset must leave the currently usable recorder intact.
        bind_capture_modality_contract(desired_root, desired_contract)
        if desired_force_contract is not None and desired_root != self.dataset_root:
            bind_gripper_force_contract(desired_root, desired_force_contract)
        elif desired_force_contract is not None and getattr(self, "gripper_force_contract", None) is None:
            bind_gripper_force_contract(desired_root, desired_force_contract)
        all_camera_sources = dict(self.all_camera_sources)
        self.close()
        self.__init__(
            args=self.args,
            components=self.components,
            camera_sources=all_camera_sources,
        )
        if trace_sink is not None:
            self.attach_control_trace_episode_sink(trace_sink)
        self.start()
        self.write_camera_device_config()

    def configured_for_current_args(self) -> bool:
        desired_root = Path(self.args.dataset_root)
        desired_contract = CaptureModalityContract(
            ignored_camera_keys=tuple(getattr(self.args, "ignore_camera", ()) or ()),
            ignored_tactile_keys=tuple(getattr(self.args, "ignore_tactile", ()) or ()),
        )
        desired_force_contract = _gripper_force_dataset_contract_from_args(self.args)
        if desired_force_contract is None and desired_root == self.dataset_root:
            desired_force_contract = getattr(self, "gripper_force_contract", None)
        existing_force_contract = getattr(self, "gripper_force_contract", None)
        if existing_force_contract is not None:
            current_force_contract = _current_gripper_force_dataset_contract(
                self.args, self.components.active_sides
            )
            if current_force_contract != existing_force_contract:
                return False
        return bool(
            desired_root == self.dataset_root
            and desired_contract == self.capture_modality_contract
            and desired_force_contract == existing_force_contract
            and str(self.writer_storage_config.task_id) == str(self.args.collection_task_id)
            and str(self.writer_storage_config.task_description) == str(self.args.collection_task_description)
            and str(self.writer_storage_config.dataset_name) == str(self.args.dataset_name)
        )

    def require_configured_for_current_args(self) -> None:
        if self.configured_for_current_args():
            return
        raise RuntimeError(
            "dataset task/root/modality configuration changed after warmup started; "
            "cancel and start collection again"
        )

    def reload_finalized_dataset_state(self) -> None:
        """Rescan idle dataset counters before opening the next episode."""

        if self.is_recording or self.finalizing:
            raise RuntimeError("Cannot reload dataset state while recording or finalizing.")
        writer_lifecycle = self._writer_lifecycle()
        if writer_lifecycle is None:
            raise RuntimeError("formal collection writer lifecycle is unavailable")
        if getattr(self, "collection_worker_mode", "thread") == "thread":
            self.writer_pool.join()
        self.writer.reload_finalized_dataset_state()
        if getattr(self, "collection_worker_mode", "thread") == "thread":
            self.sample_builder.set_dataset_position(
                episode_index=int(self.writer.next_episode_index),
                sample_index=int(self.writer.next_sample_index),
            )

    def delete_episode(self, episode_index: int) -> EpisodeDeleteResult:
        """Stop the writer lifecycle, compact one finalized episode, then resume."""
        if self.is_recording or self.finalizing:
            raise RuntimeError("Cannot delete an episode while recording or finalizing.")
        trace_sink = getattr(self, "control_trace_episode_sink", None)
        self.close()
        result: EpisodeDeleteResult | None = None
        maintenance_error: BaseException | None = None
        try:
            result = delete_episode_and_compact(self.dataset_root, int(episode_index))
        except BaseException as exc:  # restart the writer even when validation/deletion fails
            maintenance_error = exc
        self.__init__(
            args=self.args,
            components=self.components,
            camera_sources=self.camera_sources,
        )
        if trace_sink is not None:
            self.attach_control_trace_episode_sink(trace_sink)
        self.start()
        self.write_camera_device_config()
        if maintenance_error is not None:
            raise maintenance_error
        assert result is not None
        return result

    @property
    def is_recording(self) -> bool:
        return self.active_episode_index is not None

    def attach_control_trace_episode_sink(
        self,
        sink: EpisodeTraceRecorderSink,
    ) -> None:
        """Attach one subordinate trace sink before any episode is active."""

        if not isinstance(sink, EpisodeTraceRecorderSink):
            raise TypeError("sink must be EpisodeTraceRecorderSink.")
        if self.is_recording or self.finalizing:
            raise RuntimeError("Cannot replace the ControlTrace sink during an episode lifecycle.")
        current = getattr(self, "control_trace_episode_sink", None)
        if current is not None and current is not sink:
            raise RuntimeError("A different ControlTrace episode sink is already attached.")
        self.control_trace_episode_sink = sink

    @property
    def is_paused(self) -> bool:
        return bool(self.is_recording and getattr(self, "_pause_started_at", None) is not None)

    @property
    def can_record_samples(self) -> bool:
        return bool(self.is_recording and not self.is_paused)

    @property
    def pause_reason(self) -> str:
        return str(getattr(self, "_pause_reason", None) or "")

    @property
    def pause_reasons(self) -> tuple[str, ...]:
        return tuple(str(reason) for reason in getattr(self, "_pause_reasons", ()))

    @property
    def continuity_segment_index(self) -> int:
        return int(getattr(self, "_continuity_segment_index", 0) or 0)

    @property
    def frame_count(self) -> int:
        if getattr(self, "collection_worker_mode", "thread") == "process":
            return int(self._process_episode_submitted_count)
        return int(self.sample_builder.frame_index)

    @property
    def finalizing(self) -> bool:
        return any(not task.done for task in self.finalize_tasks)

    def _writer_lifecycle(self) -> Any:
        """Return the shared writer runtime while preserving old observer seams."""

        if getattr(self, "collection_worker_mode", "thread") == "process":
            return self.process_worker
        runtime = getattr(self, "writer_runtime", None)
        observer = getattr(self, "writer_observer", None)
        if runtime is not None and observer is getattr(runtime, "writer_observer", None):
            return runtime
        # Existing tests and downstream diagnostics may replace the established
        # writer_observer attribute with a fake.  Keep that seam functional
        # without making the shared runtime depend on the GUI entrypoint.
        return observer

    def start(self) -> None:
        writer_lifecycle = self._writer_lifecycle()
        if writer_lifecycle is None:
            raise RuntimeError("formal collection writer lifecycle is unavailable")
        rollback_steps = [("writer_runtime", lambda: writer_lifecycle.close(flush=False))]
        try:
            writer_lifecycle.start()
            if self.arm_state_poller is not None:
                rollback_steps.append(("arm_state_poller", self.arm_state_poller.close))
                self.arm_state_poller.start()
            if self.hand_state_poller is not None:
                rollback_steps.append(("hand_state_poller", self.hand_state_poller.close))
                self.hand_state_poller.start()
        except BaseException as start_error:  # noqa: BLE001 - rollback before propagating.
            rollback_failures = collect_lifecycle_failures(reversed(rollback_steps))
            if rollback_failures:
                raise LifecycleStartupError(
                    "real_teleop_dataset_recorder.start",
                    start_error,
                    rollback_failures,
                ) from start_error
            raise

    def write_camera_device_config(self) -> None:
        if not self.camera_sources:
            return
        self.writer.write_device_config(
            "realsense",
            {
                camera_key: source.device_info()
                for camera_key, source in self.camera_sources.items()
            },
        )

    def start_episode(self, metadata: Mapping[str, Any], *, now: float) -> int:
        if self.is_recording:
            raise RuntimeError("dataset episode is already recording")
        if self.finalizing:
            raise RuntimeError("previous dataset episode is still finalizing")
        ready, message = self.actual_feedback_warmup_status(now)
        if not ready:
            raise RuntimeError(f"actual TCP feedback is not ready: {message}")
        trace_sink = getattr(self, "control_trace_episode_sink", None)
        trace_preparation = None
        metadata_payload = dict(metadata)
        force_contract = getattr(self, "gripper_force_contract", None)
        if force_contract is not None:
            metadata_payload["gripper_force_profile"] = force_contract.to_dict()
        if trace_sink is not None:
            planned_episode_index = int(self.writer.next_episode_index)
            trace_preparation = trace_sink.prepare_episode(
                legacy_episode_index=planned_episode_index,
                prepared_at=trace_sink.control_now(),
            )
            trace_manifest_prefix = (
                "meta/v24/control_trace/episodes/"
                f"{trace_preparation.episode_uuid}"
            )
            metadata_payload["control_trace_episode"] = {
                "schema": "dexteleop_control_trace_episode_binding_v1",
                "episode_uuid": trace_preparation.episode_uuid,
                "trace_session_id": trace_sink.runtime.session.trace_session_id,
                "profile_id": trace_sink.runtime.profile.profile_id,
                "profile_scope": trace_sink.runtime.profile.profile_scope,
                "seal_assessment_ref": (
                    f"{trace_manifest_prefix}/seal_assessment.cjson"
                ),
                "capture_commit_ref": (
                    f"{trace_manifest_prefix}/capture_commit.cjson"
                ),
            }
        writer_start_kwargs: dict[str, Any] = {"metadata": metadata_payload}
        if (
            trace_preparation is not None
            and getattr(self, "collection_worker_mode", "thread") == "process"
            and bool(getattr(self.process_worker, "native_stream_contracts", {}))
        ):
            writer_start_kwargs["episode_uuid"] = trace_preparation.episode_uuid
        try:
            episode_index = self.writer.start_episode(**writer_start_kwargs)
        except BaseException:
            if trace_sink is not None and trace_preparation is not None:
                trace_sink.cancel_prepared_episode(trace_preparation)
            raise
        if trace_preparation is not None and int(episode_index) != trace_preparation.legacy_episode_index:
            self.writer.discard_episode(int(episode_index))
            trace_sink.cancel_prepared_episode(trace_preparation)
            raise RuntimeError(
                "Legacy dataset episode index changed after ControlTrace preparation: "
                f"planned={trace_preparation.legacy_episode_index} actual={int(episode_index)}"
            )
        if trace_sink is not None and trace_preparation is not None:
            try:
                trace_sink.bind_prepared_episode(
                    trace_preparation,
                    bound_at=trace_sink.control_now(),
                )
            except BaseException:
                self.writer.discard_episode(int(episode_index))
                trace_sink.cancel_prepared_episode(trace_preparation)
                raise
        if getattr(self, "collection_worker_mode", "thread") == "thread":
            self.sample_builder.set_episode(episode_index)
        self._process_episode_submitted_count = 0
        self._sample_overload_reason = None
        self._sample_modality_reason = None
        self._collection_integrity_reasons = []
        self.active_episode_index = int(episode_index)
        self.active_started_at = float(now)
        self._active_control_trace_preparation = trace_preparation
        self._record_timestamps.clear()
        self._episode_start_monotonic = time.monotonic()
        self._pause_started_at = None
        self._pause_started_monotonic = None
        self._pause_reason = None
        self._pause_reasons = []
        self._pause_started_frame_index = None
        self._pause_started_step_index = None
        self._pause_started_last_state = None
        self._pause_segments = []
        self._paused_accumulated_sec = 0.0
        self._continuity_segment_index = 0
        return int(episode_index)

    def actual_feedback_warmup_status(self, now: float) -> tuple[bool, str]:
        """Return the feedback-start gate without issuing a synchronous hardware read."""

        if self.arm_state_poller is None:
            return True, "actual TCP feedback recording is disabled"
        readiness = self.arm_state_poller.readiness_by_side(
            required_sides=tuple(self.components.active_sides),
            sample_timestamp=float(now),
        )
        pending = {side: reason for side, reason in readiness.items() if reason is not None}
        if not pending:
            return True, "actual TCP feedback ready"
        details = ", ".join(f"{side}={reason}" for side, reason in sorted(pending.items()))
        return False, f"等待 actual TCP feedback：{details}。"

    def _note_recorded_sample(self) -> None:
        current_time = time.monotonic()
        self._record_timestamps.append(current_time)
        cutoff = current_time - self.frequency_window_sec
        while self._record_timestamps and self._record_timestamps[0] < cutoff:
            self._record_timestamps.popleft()

    def _mark_collection_integrity_issue(self, reason: str) -> None:
        """Persist any causal sample loss instead of silently accepting it."""

        normalized = str(reason).strip()
        if not normalized:
            return
        reasons = getattr(self, "_collection_integrity_reasons", None)
        if reasons is None:
            reasons = []
            self._collection_integrity_reasons = reasons
        if normalized not in reasons:
            reasons.append(normalized)

    def _current_pause_elapsed_sec(self, *, current_time: float | None = None) -> float:
        pause_started = getattr(self, "_pause_started_monotonic", None)
        if pause_started is None:
            return 0.0
        now_monotonic = time.monotonic() if current_time is None else float(current_time)
        return max(0.0, now_monotonic - float(pause_started))

    def _total_paused_sec(self, *, current_time: float | None = None) -> float:
        accumulated = max(0.0, float(getattr(self, "_paused_accumulated_sec", 0.0) or 0.0))
        return accumulated + self._current_pause_elapsed_sec(current_time=current_time)

    def _active_elapsed_sec(self, *, current_time: float | None = None) -> tuple[float, float]:
        start_time = getattr(self, "_episode_start_monotonic", None)
        now_monotonic = time.monotonic() if current_time is None else float(current_time)
        wall_elapsed_sec = (
            max(0.0, now_monotonic - float(start_time))
            if start_time is not None
            else 0.0
        )
        paused_sec = min(wall_elapsed_sec, self._total_paused_sec(current_time=now_monotonic))
        return max(0.0, wall_elapsed_sec - paused_sec), wall_elapsed_sec

    def pause_episode(
        self,
        *,
        reason: str,
        now: float,
        step_index: int,
        last_state: str | None,
    ) -> bool:
        if not self.is_recording:
            return False
        if self.is_paused:
            normalized_reason = str(reason)
            if normalized_reason and normalized_reason not in self._pause_reasons:
                self._pause_reasons.append(normalized_reason)
            return False
        # Process mode pauses must never wait on the snapshot thread from the
        # 60 Hz control loop.  At most the already accepted causal sample may
        # complete after this boundary; stop/finalize still performs a drain.
        if getattr(self, "collection_worker_mode", "thread") == "thread":
            self._drain_async_sample()
        if not hasattr(self, "_pause_segments"):
            self._pause_segments = []
        self._pause_started_at = float(now)
        self._pause_started_monotonic = time.monotonic()
        self._pause_reason = str(reason)
        self._pause_reasons = [str(reason)]
        self._pause_started_frame_index = int(self.frame_count)
        self._pause_started_step_index = int(step_index)
        self._pause_started_last_state = last_state
        self._record_timestamps.clear()
        return True

    def _close_pause_segment(
        self,
        *,
        resume_reason: str,
        now: float,
        step_index: int,
    ) -> dict[str, Any] | None:
        if not self.is_paused:
            return None
        pause_started_at = self._pause_started_at
        pause_started_monotonic = self._pause_started_monotonic
        duration_sec = (
            max(0.0, time.monotonic() - float(pause_started_monotonic))
            if pause_started_monotonic is not None
            else max(0.0, float(now) - float(pause_started_at or now))
        )
        segment = {
            "reason": str(self._pause_reason or ""),
            "reasons": list(self.pause_reasons),
            "resume_reason": str(resume_reason),
            "start_timestamp": float(pause_started_at or now),
            "end_timestamp": float(now),
            "duration_sec": float(duration_sec),
            "start_frame_index": self._pause_started_frame_index,
            "end_frame_index": int(self.frame_count),
            "start_step_index": self._pause_started_step_index,
            "end_step_index": int(step_index),
            "last_state_at_pause": self._pause_started_last_state,
        }
        if not hasattr(self, "_pause_segments"):
            self._pause_segments = []
        self._pause_segments.append(segment)
        self._paused_accumulated_sec = max(
            0.0,
            float(getattr(self, "_paused_accumulated_sec", 0.0) or 0.0) + float(duration_sec),
        )
        self._pause_started_at = None
        self._pause_started_monotonic = None
        self._pause_reason = None
        self._pause_reasons = []
        self._pause_started_frame_index = None
        self._pause_started_step_index = None
        self._pause_started_last_state = None
        self._record_timestamps.clear()
        return segment

    def resume_episode(
        self,
        *,
        reason: str,
        now: float,
        step_index: int,
    ) -> bool:
        if not self.is_paused:
            return False
        # Reset model-temporal state while the recorder is still paused.  If
        # the process worker cannot acknowledge the ordered reset, fail closed
        # and leave the episode paused instead of reopening a mixed segment.
        if getattr(self, "collection_worker_mode", "thread") == "thread":
            reset_history = getattr(
                self.sample_builder,
                "reset_temporal_history",
                None,
            )
            if callable(reset_history):
                reset_history()
        else:
            process_worker = getattr(self, "process_worker", None)
            reset_history = getattr(process_worker, "reset_temporal_history", None)
            if callable(reset_history):
                reset_history()
        resumed = self._close_pause_segment(
            resume_reason=reason,
            now=now,
            step_index=step_index,
        ) is not None
        if resumed:
            # A modality rejection is a one-shot result from the previous
            # sample attempt.  Do not carry it across a successful resume:
            # otherwise the control loop can immediately pause the reopened
            # episode again before the first new sample has completed, which
            # produces a pause/resume storm with the same stale-camera reason.
            self._sample_modality_reason = None
            self._sample_overload_reason = None
            self._continuity_segment_index = self.continuity_segment_index + 1
        return resumed

    def pause_summary(self) -> dict[str, Any]:
        active_elapsed_sec, wall_elapsed_sec = self._active_elapsed_sec()
        return {
            "recording_pause_schema": "dexteleop_recording_pause_v1",
            "recording_paused": bool(self.is_paused),
            "recording_pause_reason": self.pause_reason,
            "recording_pause_reasons": list(self.pause_reasons),
            "recording_pause_segments": [dict(segment) for segment in getattr(self, "_pause_segments", [])],
            "recording_pause_count": len(getattr(self, "_pause_segments", [])) + (1 if self.is_paused else 0),
            "recording_paused_duration_sec": self._total_paused_sec(),
            "collection_continuity_segment_count": self.continuity_segment_index + 1,
            "recording_active_duration_sec": active_elapsed_sec,
            "recording_wall_duration_sec": wall_elapsed_sec,
        }

    def collection_frequency_status(self) -> dict[str, Any]:
        frame_count = int(self.frame_count)
        current_time = time.monotonic()
        elapsed_sec, wall_elapsed_sec = self._active_elapsed_sec(current_time=current_time)
        average_hz = (float(frame_count) / elapsed_sec) if elapsed_sec > 0.0 else None
        rolling_hz = None
        if len(self._record_timestamps) >= 2:
            span = float(self._record_timestamps[-1] - self._record_timestamps[0])
            if span > 0.0:
                rolling_hz = float(len(self._record_timestamps) - 1) / span
        sample_hz = rolling_hz if rolling_hz is not None else average_hz
        enough_data = (
            self.can_record_samples
            and frame_count >= 4
            and elapsed_sec >= self.frequency_min_elapsed_sec
            and sample_hz is not None
        )
        threshold_hz = float(self.frequency_warning_hz)
        warning_active = bool(enough_data and float(sample_hz) < threshold_hz)
        warning = {
            "active": warning_active,
            "threshold_hz": threshold_hz,
            "message": "",
        }
        if warning_active:
            warning["message"] = (
                f"WARNING: 数采实际频率 {float(sample_hz):.1f} Hz，"
                f"低于目标 {threshold_hz:.1f} Hz；请检查相机、写盘或 CPU 负载。"
            )
        tactile_frame_counts: dict[str, int] = {}
        tactile_write_hz: dict[str, float] = {}
        if self.active_episode_index is not None:
            try:
                tactile_frame_counts = self.writer.tactile_frame_counts(self.active_episode_index)
            except Exception:  # noqa: BLE001
                tactile_frame_counts = {}
        if elapsed_sec > 0.0:
            tactile_write_hz = {
                side: float(count) / elapsed_sec
                for side, count in tactile_frame_counts.items()
            }
        tactile_write_hz_mean = (
            sum(tactile_write_hz.values()) / float(len(tactile_write_hz))
            if tactile_write_hz
            else None
        )
        return {
            "sample_hz": sample_hz,
            "average_hz": average_hz,
            "rolling_hz": rolling_hz,
            "target_hz": threshold_hz,
            "elapsed_sec": elapsed_sec,
            "wall_elapsed_sec": wall_elapsed_sec,
            "paused": bool(self.is_paused),
            "pause_reason": self.pause_reason,
            "paused_duration_sec": self._total_paused_sec(current_time=current_time),
            "frame_count": frame_count,
            "camera_frame_max_age_sec": DEFAULT_CAMERA_FRAME_MAX_AGE_SEC,
            "camera_frame_rejection": self.sample_builder.last_camera_rejection,
            "camera_frame_age_sec_by_key": dict(
                self.sample_builder.last_camera_frame_age_sec_by_key
            ),
            "tactile_frame_counts": tactile_frame_counts,
            "tactile_write_hz": tactile_write_hz,
            "tactile_write_hz_mean": tactile_write_hz_mean,
            "window_sec": float(self.frequency_window_sec),
            "sample_assembly_async": bool(
                getattr(self, "_async_sample_enabled", False)
            ),
            "sample_assembly_skipped_busy": int(
                getattr(self, "_sample_async_skipped_busy", 0)
            ),
            "collection_worker_mode": getattr(
                self, "collection_worker_mode", "thread"
            ),
            "collection_process_rejected_busy": int(
                getattr(self, "_sample_process_rejected_busy", 0)
            ),
            "collection_overload_reason": self.collection_overload_reason,
            "collection_integrity_valid": not bool(
                getattr(self, "_collection_integrity_reasons", [])
            ),
            "collection_integrity_reasons": list(
                getattr(self, "_collection_integrity_reasons", [])
            ),
            "collection_process": (
                {}
                if getattr(self, "process_worker", None) is None
                else self.process_worker.quick_status()
            ),
            "warning": warning,
        }

    def _action_debug_metadata(
        self,
        step_result: SessionStepResult,
        *,
        record_wall_time: float,
    ) -> dict[str, Any]:
        return _build_action_debug_metadata(
            step_result,
            components=getattr(self, "components", None),
            record_wall_time=record_wall_time,
        )

    def record_step(
        self,
        step_result: SessionStepResult,
        *,
        include_debug_metadata: bool = False,
        include_action_debug_metadata: bool = False,
        phase_recorder: Any | None = None,
        metadata_extra: Mapping[str, Any] | None = None,
    ) -> Any | None:
        if not self.can_record_samples:
            return None
        if include_action_debug_metadata:
            metadata_with_action_debug = dict(metadata_extra or {})
            record_wall_time = time.time()
            try:
                metadata_with_action_debug["action_debug"] = self._action_debug_metadata(
                    step_result,
                    record_wall_time=record_wall_time,
                )
            except Exception as exc:  # noqa: BLE001
                metadata_with_action_debug["action_debug"] = {
                    "schema": "dexteleop_action_debug_v1",
                    "enabled": True,
                    "error": f"{type(exc).__name__}: {exc}",
                    "record_wall_time": float(record_wall_time),
                }
            metadata_extra = metadata_with_action_debug
        previous_include_debug_metadata = bool(self.sample_builder.include_debug_metadata)
        previous_include_action_debug_metadata = bool(
            getattr(self.sample_builder, "include_action_debug_metadata", False)
        )
        self.sample_builder.include_debug_metadata = bool(include_debug_metadata)
        self.sample_builder.include_action_debug_metadata = bool(include_action_debug_metadata)
        try:
            if getattr(self, "collection_worker_mode", "thread") == "process":
                if self.process_worker is None:
                    raise RuntimeError("formal collection process worker is unavailable")
                self._sample_overload_reason = None
                snapshot = self.sample_builder.snapshot_inputs(
                    step_result,
                    self.collector_components,
                )
                if snapshot is None:
                    result = None
                elif self.process_worker.submit_snapshot(
                    snapshot,
                    metadata_extra=metadata_extra,
                ):
                    self._process_episode_submitted_count += 1
                    result = True
                else:
                    self._sample_process_rejected_busy += 1
                    self._sample_overload_reason = (
                        "collection_process_queue_or_shared_slots_busy"
                    )
                    self._mark_collection_integrity_issue(self._sample_overload_reason)
                    result = None
            else:
                result = self._writer_lifecycle().handle_step(
                    step_result,
                    phase_recorder=phase_recorder,
                    metadata_extra=metadata_extra,
                )
        finally:
            self.sample_builder.include_debug_metadata = previous_include_debug_metadata
            self.sample_builder.include_action_debug_metadata = previous_include_action_debug_metadata
        if result is None and self.sample_builder.last_camera_rejection:
            self._sample_modality_reason = str(self.sample_builder.last_camera_rejection)
            self._mark_collection_integrity_issue(self._sample_modality_reason)
        if result is not None:
            self._sample_modality_reason = None
            self._note_recorded_sample()
        return result

    def _drain_async_sample(self) -> Any | None:
        future = getattr(self, "_sample_future", None)
        if future is None:
            return None
        self._sample_future = None
        return future.result()

    def record_step_async(
        self,
        step_result: SessionStepResult,
        *,
        include_debug_metadata: bool = False,
        include_action_debug_metadata: bool = False,
        metadata_extra: Mapping[str, Any] | None = None,
    ) -> bool:
        """Queue one causal dataset sample without blocking the control/IK loop."""

        if not bool(getattr(self, "_async_sample_enabled", False)):
            result = self.record_step(
                step_result,
                include_debug_metadata=include_debug_metadata,
                include_action_debug_metadata=include_action_debug_metadata,
                metadata_extra=metadata_extra,
            )
            return result is not None
        if not self.can_record_samples:
            return False
        previous = getattr(self, "_sample_future", None)
        if previous is not None:
            if not previous.done():
                # At 15 Hz a healthy assembly (~8-20 ms) finishes well before
                # the next request.  Never stall motion if storage is unhealthy.
                self._sample_async_skipped_busy += 1
                self._sample_overload_reason = "collection_snapshot_thread_busy"
                self._mark_collection_integrity_issue(self._sample_overload_reason)
                return False
            self._sample_future = None
            try:
                previous_result = previous.result()
            except Exception as exc:
                error_kind = (
                    "collection_process_error"
                    if getattr(self, "collection_worker_mode", "thread") == "process"
                    else "collection_writer_error"
                )
                self._sample_overload_reason = (
                    f"{error_kind}:{type(exc).__name__}:{exc}"
                )
                self._mark_collection_integrity_issue(self._sample_overload_reason)
                return False
            if previous_result is None and self._sample_modality_reason:
                # A completed asynchronous sample was rejected by the
                # modality contract (most commonly a camera frame just over
                # the freshness limit).  Do not enqueue a second sample while
                # the first rejection is still awaiting the control-loop
                # pause boundary; otherwise the same one-shot reason can be
                # replayed after resume and create a pause/resume storm.
                return False
            if (
                getattr(self, "collection_worker_mode", "thread") == "process"
                and previous_result is None
                and self._sample_overload_reason is not None
            ):
                return False
        executor = getattr(self, "_sample_executor", None)
        if executor is None:
            raise RuntimeError("teleop sample executor is not available")
        self._sample_future = executor.submit(
            self.record_step,
            step_result,
            include_debug_metadata=include_debug_metadata,
            include_action_debug_metadata=include_action_debug_metadata,
            phase_recorder=None,
            metadata_extra=None if metadata_extra is None else dict(metadata_extra),
        )
        return True

    @property
    def collection_overload_reason(self) -> str:
        return str(getattr(self, "_sample_overload_reason", None) or "")

    def poll_collection_overload(self) -> str:
        """Nonblocking 60 Hz poll for a rejected/failed collection sample."""

        future = getattr(self, "_sample_future", None)
        if future is not None and future.done():
            self._sample_future = None
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - isolate collection from control.
                error_kind = (
                    "collection_process_error"
                    if getattr(self, "collection_worker_mode", "thread") == "process"
                    else "collection_writer_error"
                )
                self._sample_overload_reason = (
                    f"{error_kind}:{type(exc).__name__}:{exc}"
                )
                self._mark_collection_integrity_issue(self._sample_overload_reason)
            else:
                if result is None and self._sample_overload_reason is None:
                    # Camera/reference rejection is accounted by the normal
                    # sample-frequency diagnostics, not process overload.
                    return ""
        return self.collection_overload_reason

    @property
    def collection_modality_reason(self) -> str:
        return str(getattr(self, "_sample_modality_reason", None) or "")

    def collection_worker_ready_status(self) -> tuple[bool, str]:
        if getattr(self, "process_worker", None) is None:
            return True, "thread collection writer ready"
        status = self.process_worker.quick_status()
        error = status.get("worker_error")
        if error:
            return False, f"数采子进程失败：{error}"
        if not bool(status.get("worker_alive")):
            return False, "数采子进程已退出。"
        if int(status.get("slots_available", 0) or 0) <= 0:
            return False, "等待数采子进程释放共享内存槽。"
        return True, "数采子进程队列可用。"

    def _writer_stats_payload(self) -> dict[str, Any]:
        if getattr(self, "process_worker", None) is not None:
            try:
                return {
                    "mode": "process",
                    **asdict(self.process_worker.stats()),
                    "process_status": dict(self.process_worker.process_status),
                }
            except Exception as exc:  # noqa: BLE001
                return {"mode": "process", "error": f"{type(exc).__name__}: {exc}"}
        try:
            if getattr(self, "writer_pool", None) is None:
                raise RuntimeError("dataset writer pool is unavailable")
            return asdict(self.writer_pool.stats())
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _storage_queue_stats_payload(self) -> dict[str, Any]:
        if getattr(self, "process_worker", None) is not None:
            return {
                "enabled": False,
                "mode": "process_whole_sample_queue",
                "storage_queue_size": 0,
                "routes": {},
            }
        try:
            if getattr(self, "writer_observer", None) is None:
                raise RuntimeError("dataset writer observer is unavailable")
            return dict(self.writer_observer.storage_stats())
        except Exception as exc:  # noqa: BLE001
            return {"enabled": False, "error": f"{type(exc).__name__}: {exc}", "routes": {}}

    def _finalize_summary(
        self,
        *,
        reason: str,
        step_index: int,
        last_state: str | None,
        frame_count: int,
        queued_at: float,
        frequency_status: Mapping[str, Any] | None = None,
        pause_summary: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        pause_payload = self.pause_summary() if pause_summary is None else dict(pause_summary)
        return {
            "stop_reason": str(reason),
            "last_step_index": int(step_index),
            "last_state": last_state,
            "episode_frame_count": int(frame_count),
            "dataset_root": str(self.dataset_root),
            "camera_keys": sorted(self.camera_sources),
            "writer_pool_stats_at_stop": self._writer_stats_payload(),
            "storage_queue_stats_at_stop": self._storage_queue_stats_payload(),
            "collection_frequency": {} if frequency_status is None else dict(frequency_status),
            "collection_integrity_valid": not bool(
                getattr(self, "_collection_integrity_reasons", [])
            ),
            "collection_integrity_reasons": list(
                getattr(self, "_collection_integrity_reasons", [])
            ),
            **pause_payload,
            "background_finalize": True,
            "finalize_queued_at": float(queued_at),
        }

    def _run_finalize_task(
        self,
        task: RealDatasetFinalizeTask,
    ) -> None:
        summary_payload = dict(task.summary)

        def run_stage(
            name: str,
            callback: Callable[[], Any],
        ) -> Any:
            started_at = time.perf_counter()
            try:
                return callback()
            finally:
                task.record_stage_timing(name, time.perf_counter() - started_at)

        try:
            if task.before_finalize is not None:
                audit_result = run_stage("position_audit_close", task.before_finalize)
                if audit_result is not None:
                    summary_payload["position_audit_finalize"] = dict(audit_result)
            run_stage("writer_flush", self._writer_lifecycle().flush)
            if bool(
                getattr(
                    getattr(self, "process_worker", None),
                    "native_external_stream_bindings",
                    {},
                )
            ):
                trace_sink = getattr(self, "control_trace_episode_sink", None)
                if trace_sink is None or task.control_trace_release is None:
                    raise RuntimeError(
                        "External native streams require an episode trace "
                        "release before legacy finalize."
                    )
                task.control_trace_release = run_stage(
                    "native_stream_release_freeze",
                    lambda: trace_sink.resolve_release(
                        task.control_trace_release
                    ),
                )
                if task.control_trace_release is None:
                    raise RuntimeError(
                        "External native stream episode release resolved empty."
                    )
            summary_payload["writer_pool_stats_after_flush"] = self._writer_stats_payload()
            storage_queue_stats = self._storage_queue_stats_payload()
            summary_payload["storage_queue_stats"] = storage_queue_stats
            dropped_routes = {
                str(route_key): int(route_stats.get("dropped", 0) or 0)
                for route_key, route_stats in dict(
                    storage_queue_stats.get("routes", {}) or {}
                ).items()
                if isinstance(route_stats, Mapping)
                and int(route_stats.get("dropped", 0) or 0) > 0
            }
            if dropped_routes:
                drop_reason = "storage_queue_dropped:" + ",".join(
                    f"{key}={count}" for key, count in sorted(dropped_routes.items())
                )
                integrity_reasons = list(
                    summary_payload.get("collection_integrity_reasons", ()) or ()
                )
                if drop_reason not in integrity_reasons:
                    integrity_reasons.append(drop_reason)
                summary_payload["collection_integrity_valid"] = False
                summary_payload["collection_integrity_reasons"] = integrity_reasons
            # This is persisted with the episode record; the final writer stage
            # itself is measured on task completion and emitted in the Dataset
            # completion log because it cannot be known before the record is
            # atomically finalized.
            summary_payload["finalize_prewrite_timing_ms"] = dict(task.stage_timings_ms)
            summary_payload["finalize_started_at"] = float(
                task.started_at if task.started_at is not None else time.time()
            )
            legacy_finalize_result = run_stage(
                "writer_finalize_episode",
                lambda: self.writer.finalize_episode(task.episode_index, summary=summary_payload),
            )
        except BaseException as exc:  # noqa: BLE001
            trace_sink = getattr(self, "control_trace_episode_sink", None)
            if trace_sink is not None:
                try:
                    task.control_trace_finalize_report = trace_sink.legacy_finalize_failed(
                        task.control_trace_release,
                        completed_at=trace_sink.commit_now(),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                except BaseException as trace_exc:  # preserve legacy failure as primary
                    task.control_trace_error = (
                        f"legacy_finalize_failure_reporting:{type(trace_exc).__name__}: {trace_exc}"
                    )
            task.mark_done(success=False, error=f"{type(exc).__name__}: {exc}")
            return
        trace_sink = getattr(self, "control_trace_episode_sink", None)
        if trace_sink is not None:
            native_stream_artifact_receipts = tuple(
                getattr(
                    legacy_finalize_result,
                    "native_stream_artifact_receipts",
                    (),
                )
                or ()
            )
            artifact_receipt_error = str(
                getattr(legacy_finalize_result, "artifact_receipt_error", "")
                or ""
            )
            if artifact_receipt_error:
                task.control_trace_error = (
                    "native_artifact_receipt:"
                    f"{artifact_receipt_error}"
                )
            try:
                def complete_control_trace():
                    kwargs: dict[str, Any] = {
                        "completed_at": trace_sink.commit_now(),
                    }
                    if native_stream_artifact_receipts:
                        kwargs["native_stream_artifact_receipts"] = (
                            native_stream_artifact_receipts
                        )
                    return trace_sink.legacy_finalize_completed(
                        task.control_trace_release,
                        **kwargs,
                    )

                task.control_trace_finalize_report = run_stage(
                    "control_trace_post_finalize",
                    complete_control_trace,
                )
            except BaseException as trace_exc:  # legacy finalize remains successful
                post_finalize_error = (
                    f"post_finalize:{type(trace_exc).__name__}: {trace_exc}"
                )
                task.control_trace_error = ";".join(
                    value
                    for value in (
                        task.control_trace_error,
                        post_finalize_error,
                    )
                    if value
                )
        task.mark_done(success=True)

    def _start_finalize_task(self, task: RealDatasetFinalizeTask) -> bool:
        """Start one previously queued finalizer without blocking control."""

        if task.done or task.thread is not None:
            return False
        task.mark_started()
        thread = threading.Thread(
            target=self._run_finalize_task,
            name=f"real_dataset_finalize_{task.episode_index}",
            args=(task,),
            daemon=True,
        )
        task.thread = thread
        try:
            thread.start()
        except BaseException as exc:  # noqa: BLE001 - preserve task completion semantics.
            task.mark_done(success=False, error=f"thread_start {type(exc).__name__}: {exc}")
            return False
        return True

    def start_pending_finalize_tasks(self, *, max_tasks: int = 1) -> list[RealDatasetFinalizeTask]:
        """Launch at most ``max_tasks`` queued finalizers in FIFO order.

        Real teleoperation currently admits one active recording at a time.  The
        explicit bound nevertheless prevents a future caller from starting
        multiple CPU/IO-heavy finalizers together at an episode boundary.
        """

        started: list[RealDatasetFinalizeTask] = []
        for task in self.finalize_tasks:
            if len(started) >= max(0, int(max_tasks)):
                break
            if task.pending and self._start_finalize_task(task):
                started.append(task)
        return started

    def stop_episode(
        self,
        *,
        reason: str,
        step_index: int,
        last_state: str | None,
        summary: Mapping[str, Any] | None = None,
        defer_finalize_start: bool = False,
        before_finalize: Callable[[], Mapping[str, Any] | None] | None = None,
    ) -> RealDatasetFinalizeTask | None:
        if self.active_episode_index is None:
            return None
        self._drain_async_sample()
        stop_time = time.time()
        if self.is_paused:
            self._close_pause_segment(
                resume_reason=f"stop:{reason}",
                now=stop_time,
                step_index=step_index,
            )
        episode_index = int(self.active_episode_index)
        frame_count = int(self.frame_count)
        frequency_status = self.collection_frequency_status()
        pause_summary = self.pause_summary()
        trace_sink = getattr(self, "control_trace_episode_sink", None)
        trace_preparation = getattr(self, "_active_control_trace_preparation", None)
        trace_release = None
        trace_release_error = ""
        if trace_sink is not None and trace_preparation is not None:
            try:
                trace_release = trace_sink.release_episode(
                    trace_preparation,
                    unbound_at=trace_sink.control_now(),
                    reason=f"legacy_episode_stop:{reason}",
                )
            except BaseException as trace_exc:  # legacy finalize must remain available
                trace_release_error = (
                    f"release:{type(trace_exc).__name__}: {trace_exc}"
                )
                self._mark_collection_integrity_issue(
                    f"control_trace_episode_{trace_release_error}"
                )
        self._active_control_trace_preparation = None
        self.active_episode_index = None
        self.active_started_at = None
        self._episode_start_monotonic = None
        self._pause_started_at = None
        self._pause_started_monotonic = None
        self._pause_reason = None
        self._pause_reasons = []
        self._pause_started_frame_index = None
        self._pause_started_step_index = None
        self._pause_started_last_state = None
        if frame_count <= 0:
            self._writer_lifecycle().flush()
            self.writer.discard_episode(episode_index)
            if trace_sink is not None:
                try:
                    trace_sink.defer_legacy_episode_discarded(
                        trace_release,
                        completed_at=trace_sink.commit_now(),
                    )
                except BaseException as trace_exc:
                    self._mark_collection_integrity_issue(
                        "control_trace_episode_discard_reporting:"
                        f"{type(trace_exc).__name__}:{trace_exc}"
                    )
            self.discarded_episodes += 1
            self._record_timestamps.clear()
            self._pause_segments = []
            self._paused_accumulated_sec = 0.0
            return None
        task = RealDatasetFinalizeTask(
            episode_index=episode_index,
            frame_count=frame_count,
            reason=str(reason),
            queued_at=time.time(),
            before_finalize=before_finalize,
            control_trace_release=trace_release,
            control_trace_error=trace_release_error,
        )
        task.summary = self._finalize_summary(
            reason=reason,
            step_index=step_index,
            last_state=last_state,
            frame_count=frame_count,
            queued_at=task.queued_at,
            frequency_status=frequency_status,
            pause_summary=pause_summary,
        )
        if summary is not None:
            task.summary.update(dict(summary))
        self.finalize_tasks.append(task)
        if not defer_finalize_start:
            self._start_finalize_task(task)
        return task

    def poll_finalize_tasks(self) -> list[RealDatasetFinalizeTask]:
        completed: list[RealDatasetFinalizeTask] = []
        remaining: list[RealDatasetFinalizeTask] = []
        for task in self.finalize_tasks:
            if not task.done:
                remaining.append(task)
                continue
            if task.thread is not None:
                task.thread.join(timeout=0.0)
            if task.success:
                self.saved_episodes += 1
            completed.append(task)
        self.finalize_tasks = remaining
        return completed

    def wait_finalize_tasks(self) -> None:
        while self.finalizing:
            self.start_pending_finalize_tasks()
            for task in list(self.finalize_tasks):
                if task.thread is not None:
                    task.thread.join(timeout=0.1)
            self.poll_finalize_tasks()

    def close(self) -> None:
        steps = []
        if getattr(self, "_sample_executor", None) is not None:
            steps.extend(
                [
                    ("sample_assembly_drain", self._drain_async_sample),
                    (
                        "sample_assembly_executor",
                        lambda: self._sample_executor.shutdown(wait=True, cancel_futures=False),
                    ),
                ]
            )
        if self.arm_state_poller is not None:
            steps.append(("arm_state_poller", self.arm_state_poller.close))
        if self.hand_state_poller is not None:
            steps.append(("hand_state_poller", self.hand_state_poller.close))
        steps.extend(
            [
                ("finalize_tasks", self.wait_finalize_tasks),
                ("writer_runtime", lambda: self._writer_lifecycle().close(flush=True)),
            ]
        )
        run_best_effort_cleanup("real_teleop_dataset_recorder.close", steps)



__all__ = [
    "RealTeleopDatasetRecorder",
    "_collection_hand_action_specs",
]
