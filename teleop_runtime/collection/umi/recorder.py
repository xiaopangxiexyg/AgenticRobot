"""Episode lifecycle around the shared LeRobot-style writer."""

from __future__ import annotations

from typing import Any, Mapping

from robot_motion.action.arm_spec import ArmActionSpec
from robot_motion.action.hand_spec import HandActionSpec

from ..async_writer import AsyncWriterConfig, AsyncWriterPool
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
    ) -> None:
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
                allow_missing_video_frames=False,
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
                            "T_tcp_to_tracker": config.side(side).tcp_transform.tolist(),
                            "status": config.side(side).tcp_extrinsic_status,
                        }
                        for side in ("left", "right")
                    },
                    "operator_label_semantics": "annotation_only_no_arm_control",
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
        self.pool = AsyncWriterPool(
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
        self.episode_index: int | None = None
        self.frame_count = 0

    @property
    def recording(self) -> bool:
        return self.episode_index is not None

    def start_episode(self, metadata: Mapping[str, Any] | None = None) -> int:
        if self.recording:
            raise RuntimeError("UMI episode is already recording")
        index = self.writer.start_episode(metadata=metadata)
        self.encoder.reset_episode(index)
        self.episode_index = index
        self.frame_count = 0
        self.pool.start()
        return index

    def submit(self, sample: UmiSample) -> None:
        if self.episode_index is None or sample.episode_index != self.episode_index:
            raise RuntimeError("UMI sample does not belong to the active episode")
        for job in self.writer.plan_sample_jobs(sample):
            self.pool.submit(job)
        self.frame_count += 1

    def stop_episode(self, *, reason: str) -> int | None:
        if self.episode_index is None:
            return None
        index = self.episode_index
        self.pool.join()
        self.writer.finalize_episode(
            index,
            summary={"end_reason": str(reason), "frame_count": int(self.frame_count)},
        )
        self.episode_index = None
        return index

    def stats(self) -> dict[str, Any]:
        value = self.pool.stats()
        return {
            "queued": value.queued,
            "submitted": value.submitted,
            "completed": value.completed,
            "dropped": value.dropped,
            "failures": value.failures,
            "queue_high_water": value.queue_high_water,
        }

    def close(self) -> None:
        if self.episode_index is not None:
            self.stop_episode(reason="runtime_close")
        self.pool.close(wait=True)


__all__ = ["UmiDatasetRecorder"]
