"""Episode lifecycle around the shared LeRobot-style writer."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
import time
from typing import Any, Mapping

from robot_motion.action.arm_spec import ArmActionSpec
from robot_motion.action.hand_spec import HandActionSpec
from dataset_interface.capture_modalities import bind_capture_modality_contract

from ..async_writer import AsyncWriterConfig, AsyncWriterPool
from ..dataset_episode_maintenance import EpisodeDeleteResult, delete_episode_and_compact
from ..writer_v21 import LeRobotV21Writer, LeRobotV21WriterConfig
from .calibration import UmiCalibrationBinding
from .config import UmiCollectionConfig
from .row_encoder import UmiV23RowEncoder
from .types import UmiSample


class UmiDatasetRecorder:
    def __init__(
        self,
        *,
        config: UmiCollectionConfig,
        calibration: UmiCalibrationBinding,
        policy_io_layout: Any,
        T_tcp_to_tracker_by_side: Mapping[str, Any] | None = None,
        calibration_by_side: Mapping[str, UmiCalibrationBinding] | None = None,
    ) -> None:
        self.config = config
        self.capture_modality_contract = config.capture_modality_contract
        bind_capture_modality_contract(
            config.dataset_root,
            self.capture_modality_contract,
        )
        tcp_transforms = {
            side: (
                config.side(side).tcp_transform
                if T_tcp_to_tracker_by_side is None
                else T_tcp_to_tracker_by_side[side]
            )
            for side in ("left", "right")
        }
        self.encoder = UmiV23RowEncoder()
        self.writer = LeRobotV21Writer(
            LeRobotV21WriterConfig(
                root_dir=config.dataset_root,
                nominal_fps=config.timing.sample_hz,
                task_id=config.task_id,
                task_description=config.task_description,
                write_parquet=True,
                write_video=True,
                tactile_storage="external",
                tactile_compression=config.tactile_compression,
                allow_missing_video_frames=bool(
                    self.capture_modality_contract.ignored_camera_keys
                ),
                policy_io_layout=policy_io_layout,
                policy_io_layout_provenance={
                    "source": str(config.policy_io_layout),
                    "embodiment": "umi_dual_wrist",
                },
                strict_v22_policy_io=True,
                pose_frame_contract=calibration.pose_frame_contract(),
                row_encoder=self.encoder,
                dataset_metadata={
                    "embodiment": "umi_dual_wrist",
                    "embodiment_has_arm": False,
                    "calibration_result_path": str(calibration.path),
                    "calibration_sha256": calibration.sha256,
                    "calibration_created_at_utc": calibration.created_at.isoformat(),
                    "calibration_expires_at_utc": calibration.expires_at.isoformat(),
                    "tcp_extrinsic_by_side": {
                        side: {
                            "T_tcp_to_tracker": tcp_transforms[side].tolist(),
                            "status": config.side(side).tcp_extrinsic_status,
                            "tcp_profile": config.side(side).tcp_profile,
                            "hand_eye_result_path": (
                                None
                                if calibration_by_side is None or side not in calibration_by_side
                                else str(calibration_by_side[side].path)
                            ),
                            "hand_eye_result_sha256": (
                                None
                                if calibration_by_side is None or side not in calibration_by_side
                                else calibration_by_side[side].sha256
                            ),
                        }
                        for side in ("left", "right")
                    },
                    "operator_label_semantics": "annotation_only_no_arm_control",
                    "dmtac_gripper_timing": {
                        "command_hz": float(config.dmtac_gripper_command_hz),
                        "telemetry_hz": float(config.dmtac_gripper_telemetry_hz),
                        "telemetry_command_grace_s": float(
                            config.dmtac_gripper_telemetry_command_grace_s
                        ),
                        "measurement_timestamp_field": "joint_timestamp",
                    },
                    "umi_arm_action_alignment": {
                        "schema": "dexteleop.umi_arm_action_alignment.v1",
                        "arm_action_t": "observation.arm.pose[t+1]",
                        "source_step": "next_accepted_observation",
                        "gripper_action_t": "angle_trigger_final_command[t]",
                        "terminal_policy": "drop_last_source_sample",
                        "first_last_action_arm": "observation.arm.pose[t]",
                    },
                },
                arm_action_specs={
                    side: ArmActionSpec.tool_tcp(side, reference_frame="desk")
                    for side in ("left", "right")
                },
                hand_action_specs={
                    side: HandActionSpec.dmtac_gripper(side)
                    for side in ("left", "right")
                },
            )
        )
        self.pool = self._new_writer_pool()
        self.episode_index: int | None = None
        self.frame_count = 0
        self.accepted_sample_count = 0
        self._pending_sample: UmiSample | None = None
        self.action_continuity_break_count = 0
        self.continuity_dropped_source_sample_count = 0
        self.action_continuity_break_reasons: Counter[str] = Counter()
        self.action_alignment_max_gap_sec = max(
            0.2,
            3.0 / float(config.timing.sample_hz),
        )
        self.sample_attempt_count = 0
        self.rejection_counts: Counter[str] = Counter()
        self.episode_started_at: float | None = None
        self.last_episode_index: int | None = None
        self.last_frame_count: int | None = None
        self.last_accepted_sample_count: int | None = None
        self.last_attempt_count: int | None = None
        self.last_rejection_counts: dict[str, int] = {}
        self.last_outcome = "ready"
        self.last_message = ""

    def _new_writer_pool(self) -> AsyncWriterPool:
        return AsyncWriterPool(
            self.writer.handle_job,
            config=AsyncWriterConfig(
                worker_count=1,
                max_queue_size=256,
                put_timeout_sec=0.05,
                backpressure_policy="raise",
                fail_fast=True,
            ),
            name="umi_dataset_writer",
        )

    @property
    def recording(self) -> bool:
        return self.episode_index is not None

    @property
    def next_frame_index(self) -> int:
        # ``accepted_sample_count`` counts source observations, including a
        # pending source observation discarded at a continuity break.  Output
        # frame indices must remain contiguous for the dataset writer.
        return int(self.frame_count + (1 if self._pending_sample is not None else 0))

    def start_episode(self, metadata: Mapping[str, Any] | None = None) -> int:
        if self.recording:
            raise RuntimeError("UMI episode is already recording")
        self.reload_finalized_dataset_state()
        index = self.writer.start_episode(metadata=metadata)
        self.encoder.reset_episode(index)
        self.episode_index = index
        self.frame_count = 0
        self.accepted_sample_count = 0
        self._pending_sample = None
        self.action_continuity_break_count = 0
        self.continuity_dropped_source_sample_count = 0
        self.action_continuity_break_reasons.clear()
        self.sample_attempt_count = 0
        self.rejection_counts.clear()
        self.episode_started_at = time.time()
        self.last_outcome = "recording"
        self.last_message = ""
        self.pool.start()
        return index

    def reload_finalized_dataset_state(self) -> None:
        if self.recording:
            raise RuntimeError("Cannot reload UMI dataset state while recording.")
        self.pool.join()
        self.writer.reload_finalized_dataset_state()

    def reconfigure_dataset_root(self, dataset_root: str | Path) -> None:
        """Switch an idle UMI recorder to a GUI-selected dataset root."""

        self.reconfigure_config(
            replace(self.config, dataset_root=Path(dataset_root).expanduser())
        )

    def reconfigure_config(self, next_config: UmiCollectionConfig) -> None:
        """Atomically rebind an idle recorder to one dataset-level policy."""

        if self.recording:
            raise RuntimeError("Cannot change UMI dataset configuration while recording.")
        resolved_root = Path(next_config.dataset_root).expanduser()
        next_contract = next_config.capture_modality_contract
        if (
            resolved_root == self.writer.root_dir
            and next_contract == self.capture_modality_contract
        ):
            self.reload_finalized_dataset_state()
            return
        self.pool.join()
        bind_capture_modality_contract(
            resolved_root,
            next_contract,
        )
        next_writer = LeRobotV21Writer(
            replace(
                self.writer.config,
                root_dir=resolved_root,
                allow_missing_video_frames=bool(next_contract.ignored_camera_keys),
            )
        )
        self.pool.close(wait=True)
        self.config = next_config
        self.capture_modality_contract = next_contract
        self.writer = next_writer
        self.pool = self._new_writer_pool()
        self.last_episode_index = None
        self.last_frame_count = None
        self.last_outcome = "ready"
        self.last_message = f"dataset root changed to {resolved_root}"

    def note_snapshot_attempt(self, rejection_reasons: tuple[str, ...] = ()) -> None:
        if not self.recording:
            return
        self.sample_attempt_count += 1
        self.rejection_counts.update(str(reason) for reason in rejection_reasons)

    def break_action_continuity(self, *, reason: str) -> bool:
        """Prevent next-state arm actions from crossing an observation gap.

        A pending UMI source sample can only be aligned with the next accepted
        observation when the stream stayed continuous.  Once a snapshot or
        gripper update is rejected, discard that pending source sample and let
        the next two accepted samples establish a fresh adjacent pair.
        """

        if not self.recording or self._pending_sample is None:
            return False
        self._pending_sample = None
        self.action_continuity_break_count += 1
        self.continuity_dropped_source_sample_count += 1
        self.action_continuity_break_reasons.update((str(reason),))
        return True

    def submit(self, sample: UmiSample) -> None:
        if self.episode_index is None or sample.episode_index != self.episode_index:
            raise RuntimeError("UMI sample does not belong to the active episode")
        # Direct recorder users may not call note_snapshot_attempt(); keep the
        # accounting internally consistent without double-counting controller calls.
        if int(sample.frame_index) != self.next_frame_index:
            raise RuntimeError(
                "UMI accepted sample frame_index is not contiguous: "
                f"got {sample.frame_index}, expected {self.next_frame_index}"
            )
        previous = self._pending_sample
        if (
            previous is not None
            and float(sample.timestamp) - float(previous.timestamp)
            > self.action_alignment_max_gap_sec
        ):
            self.break_action_continuity(reason="accepted_sample_gap_exceeded")
            # The assembler reserved the index while the old pending source
            # still existed.  Reuse the now-free output index after dropping it.
            sample = replace(sample, frame_index=self.next_frame_index)
        if self.sample_attempt_count <= self.accepted_sample_count:
            self.sample_attempt_count += 1
        self.accepted_sample_count += 1
        previous = self._pending_sample
        self._pending_sample = sample
        if previous is None:
            return
        aligned = replace(
            previous,
            arm_action_poses=sample.poses,
            arm_action_source_frame_index=int(sample.frame_index),
            arm_action_source_timestamp=float(sample.timestamp),
        )
        for job in self.writer.plan_sample_jobs(aligned):
            self.pool.submit(job)
        self.frame_count += 1

    def stop_episode(self, *, reason: str, discard: bool = False) -> int | None:
        if self.episode_index is None:
            return None
        index = self.episode_index
        frame_count = int(self.frame_count)
        accepted_sample_count = int(self.accepted_sample_count)
        terminal_sample_dropped = self._pending_sample is not None
        self._pending_sample = None
        self.last_outcome = "finalizing"
        try:
            self.pool.join()
            should_discard = bool(discard or frame_count <= 0)
            if should_discard:
                self.writer.discard_episode(index)
                outcome = "discarded"
                message = (
                    f"episode {index} discarded: {reason}"
                    if discard
                    else f"episode {index} discarded: no valid samples"
                )
            else:
                self.writer.finalize_episode(
                    index,
                    summary={
                        "end_reason": str(reason),
                        "frame_count": frame_count,
                        "source_sample_count": accepted_sample_count,
                        "terminal_source_sample_dropped": terminal_sample_dropped,
                        "arm_action_alignment": "next_accepted_observation",
                        "action_continuity_break_count": int(
                            self.action_continuity_break_count
                        ),
                        "continuity_dropped_source_sample_count": int(
                            self.continuity_dropped_source_sample_count
                        ),
                        "action_continuity_break_reasons": dict(
                            sorted(self.action_continuity_break_reasons.items())
                        ),
                        "action_alignment_max_gap_sec": float(
                            self.action_alignment_max_gap_sec
                        ),
                        "sample_attempt_count": int(self.sample_attempt_count),
                        "sample_rejection_count": int(
                            self.sample_attempt_count - accepted_sample_count
                        ),
                        "sample_rejection_counts_by_reason": dict(
                            sorted(self.rejection_counts.items())
                        ),
                    },
                )
                outcome = "saved"
                message = f"episode {index} saved with {frame_count} frames"
        except BaseException as exc:  # noqa: BLE001
            self.last_outcome = "error"
            self.last_message = f"{type(exc).__name__}: {exc}"
            raise
        self.episode_index = None
        self.episode_started_at = None
        self.last_episode_index = index
        self.last_frame_count = frame_count
        self.last_accepted_sample_count = accepted_sample_count
        self.last_attempt_count = int(self.sample_attempt_count)
        self.last_rejection_counts = dict(sorted(self.rejection_counts.items()))
        self.last_outcome = outcome
        self.last_message = message
        return index

    def effective_sample_hz(self, *, now: float | None = None) -> float | None:
        if self.episode_started_at is None:
            return None
        elapsed = (time.time() if now is None else float(now)) - self.episode_started_at
        if elapsed <= 0.0:
            return None
        return float(self.accepted_sample_count) / elapsed

    def stats(self) -> dict[str, Any]:
        value = self.pool.stats()
        return {
            "queued": value.queued,
            "submitted": value.submitted,
            "completed": value.completed,
            "dropped": value.dropped,
            "failures": value.failures,
            "queue_high_water": value.queue_high_water,
            "sample_attempt_count": int(self.sample_attempt_count),
            "sample_accepted_count": int(self.accepted_sample_count),
            "sample_written_row_count": int(self.frame_count),
            "terminal_sample_pending": self._pending_sample is not None,
            "action_continuity_break_count": int(self.action_continuity_break_count),
            "continuity_dropped_source_sample_count": int(
                self.continuity_dropped_source_sample_count
            ),
            "action_continuity_break_reasons": dict(
                sorted(self.action_continuity_break_reasons.items())
            ),
            "action_alignment_max_gap_sec": float(
                self.action_alignment_max_gap_sec
            ),
            "sample_rejection_count": int(
                self.sample_attempt_count - self.accepted_sample_count
            ),
            "sample_rejection_counts_by_reason": dict(
                sorted(self.rejection_counts.items())
            ),
        }

    def delete_episode(self, episode_index: int) -> EpisodeDeleteResult:
        if self.recording:
            raise RuntimeError("Cannot delete a finalized UMI episode while recording.")
        self.pool.join()
        result = delete_episode_and_compact(self.writer.root_dir, int(episode_index))
        self.writer.reload_finalized_dataset_state()
        self.last_episode_index = None
        self.last_frame_count = None
        self.last_accepted_sample_count = None
        self.last_attempt_count = None
        self.last_rejection_counts = {}
        self.last_outcome = "deleted"
        self.last_message = (
            f"episode {result.deleted_episode_index} deleted; "
            f"{result.episode_count} episodes remain, next={result.next_episode_index}"
        )
        return result

    def close(self) -> None:
        if self.episode_index is not None:
            self.stop_episode(reason="runtime_close")
        self.pool.close(wait=True)


__all__ = ["UmiDatasetRecorder"]
