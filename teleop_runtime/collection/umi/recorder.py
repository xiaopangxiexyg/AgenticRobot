"""Episode lifecycle around the shared LeRobot-style writer."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping
import uuid

from dataset_contracts.episode_identity import EpisodeCaptureIdentity
from robot_motion.action.arm_spec import ArmActionSpec
from robot_motion.action.hand_spec import HandActionSpec
from dataset_interface.capture_modalities import (
    bind_capture_modality_contract,
    dataset_is_blank,
)
from dataset_interface.modality_artifacts import (
    assert_dataset_capture_integrity_compatible,
)

from collection_runtime.async_writer import AsyncWriterConfig, AsyncWriterPool
from collection_runtime.dataset_episode_maintenance import (
    EpisodeDeleteResult,
    delete_episode_and_compact,
)
from collection_runtime.dataset_capture_recovery import (
    prepare_dataset_capture_writable,
)
from collection_runtime.progress.profiles import inspect_dataset_progress_profile
from ..schema_v22 import (
    StrictV22SampleRejection,
    validate_strict_v22_measured_state,
)
from ..writer_v21 import LeRobotV21Writer, LeRobotV21WriterConfig
from ..v25p_dual_write_shadow import (
    V25pDualWriteShadow,
    append_shadow_from_legacy_jobs,
    finalize_shadow_after_legacy,
)
from collection_runtime.compatibility import CompatibilityEpisodeSidecarStore
from .calibration import UmiCalibrationBinding
from .config import UmiCollectionConfig
from .row_encoder import UmiV23RowEncoder
from .topology import SIDES
from .types import UmiSample


def _validate_umi_dataset_admission(
    dataset_root: str | Path,
    *,
    task_id: str,
) -> None:
    """Reject targets whose dataset-wide semantics do not match UMI capture."""

    root = Path(dataset_root).expanduser()
    progress = inspect_dataset_progress_profile(root, requested_profile=None)
    if progress.profile is not None:
        raise ValueError(
            "Target dataset is already bound to a collection progress profile; "
            "the UMI collector cannot append unprofiled Episodes to it."
        )
    if dataset_is_blank(root):
        return
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Cannot validate target dataset metadata {info_path}: {exc}"
        ) from exc
    if not isinstance(info, Mapping):
        raise ValueError(f"Target dataset metadata is not an object: {info_path}")
    existing_task_id = str(info.get("task_id", "") or "").strip()
    if existing_task_id and existing_task_id != str(task_id).strip():
        raise ValueError(
            "Target dataset task binding differs from the UMI runtime config: "
            f"existing={existing_task_id!r}, requested={str(task_id).strip()!r}."
        )


class UmiDatasetRecorder:
    def __init__(
        self,
        *,
        config: UmiCollectionConfig,
        calibration: UmiCalibrationBinding,
        policy_io_layout: Any,
        T_tcp_to_tracker_by_side: Mapping[str, Any] | None = None,
        calibration_by_side: Mapping[str, UmiCalibrationBinding] | None = None,
        v25p_sidecar_store: CompatibilityEpisodeSidecarStore | None = None,
        v25p_native_shadow: V25pDualWriteShadow | None = None,
        episode_uuid_factory: Callable[[], object] = uuid.uuid4,
    ) -> None:
        self.config = config
        self.policy_io_layout = policy_io_layout
        self.capture_modality_contract = config.capture_modality_contract
        prepare_dataset_capture_writable(config.dataset_root)
        if config.tactile_artifact_integrity:
            assert_dataset_capture_integrity_compatible(config.dataset_root)
        _validate_umi_dataset_admission(
            config.dataset_root,
            task_id=config.task_id,
        )
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
            for side in SIDES
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
                tactile_artifact_integrity=config.tactile_artifact_integrity,
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
                        for side in SIDES
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
                    for side in SIDES
                },
                hand_action_specs={
                    side: HandActionSpec.dmtac_gripper(side)
                    for side in SIDES
                },
            )
        )
        self.pool = self._new_writer_pool()
        if (
            v25p_sidecar_store is not None
            and v25p_sidecar_store.dataset_root
            != Path(config.dataset_root).expanduser().resolve()
        ):
            raise ValueError("v2.5-p store dataset_root does not match the UMI recorder.")
        self.v25p_sidecar_store = v25p_sidecar_store
        if (
            v25p_native_shadow is not None
            and v25p_native_shadow.store.dataset_root
            != Path(config.dataset_root).expanduser().resolve()
        ):
            raise ValueError(
                "v2.5-p native shadow dataset_root does not match the UMI recorder."
            )
        self.v25p_native_shadow = v25p_native_shadow
        self._v25p_native_shadow_active = False
        self._v25p_native_shadow_error = ""
        self._episode_uuid_factory = episode_uuid_factory
        self._active_episode_identity: EpisodeCaptureIdentity | None = None
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
        self._externally_noted_pending_samples = 0
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
    def active_episode_identity(self) -> EpisodeCaptureIdentity | None:
        return self._active_episode_identity

    def begin_subtask_attempt(
        self,
        *,
        track_id: str,
        attempt_id: str,
        subtask_id: str,
        actor_id: str,
        timestamp: float | None = None,
    ) -> Any:
        store = self.v25p_sidecar_store
        identity = self.active_episode_identity
        if store is None or identity is None:
            raise RuntimeError("No active UMI v2.5-p episode is available.")
        event_time = time.time() if timestamp is None else float(timestamp)
        return store.begin_subtask(
            identity,
            track_id=track_id,
            attempt_id=attempt_id,
            subtask_id=subtask_id,
            effective_at=event_time,
            available_at=event_time,
            effective_clock_domain_id="local_wall_clock",
            available_clock_domain_id="local_wall_clock",
            actor_id=actor_id,
            frame_index=int(self.frame_count),
        )

    def finish_subtask_attempt(
        self,
        *,
        track_id: str,
        attempt_id: str,
        subtask_id: str,
        outcome: str,
        actor_id: str,
        timestamp: float | None = None,
    ) -> Any:
        store = self.v25p_sidecar_store
        identity = self.active_episode_identity
        if store is None or identity is None:
            raise RuntimeError("No active UMI v2.5-p episode is available.")
        event_time = time.time() if timestamp is None else float(timestamp)
        return store.finish_subtask(
            identity,
            track_id=track_id,
            attempt_id=attempt_id,
            subtask_id=subtask_id,
            outcome=outcome,
            effective_at=event_time,
            available_at=event_time,
            effective_clock_domain_id="local_wall_clock",
            available_clock_domain_id="local_wall_clock",
            actor_id=actor_id,
            frame_index=int(self.frame_count),
        )

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
        identity = EpisodeCaptureIdentity(
            episode_uuid=str(self._episode_uuid_factory()),
            legacy_episode_index=int(self.writer.next_episode_index),
            lifecycle_owner_id="teleop_runtime.collection.umi.recorder",
            prepared_at=time.monotonic(),
            prepared_clock_domain_id="python_monotonic",
        )
        metadata_payload = dict(metadata or {})
        store = self.v25p_sidecar_store
        native_shadow = getattr(self, "v25p_native_shadow", None)
        sidecar_prepared = False
        native_prepared = False
        try:
            if store is not None:
                store.prepare_episode(identity)
                sidecar_prepared = True
                prefix = f"meta/v25p/episodes/{identity.episode_uuid}"
                metadata_payload["v25p_episode"] = {
                    "schema": "dexteleop_v25p_episode_binding_v1",
                    "episode_uuid": identity.episode_uuid,
                    "episode_identity_hash": identity.semantic_content_hash,
                    "identity_ref": f"{prefix}/identity.cjson",
                    "embodiment_binding_ref": f"{prefix}/embodiment_binding.cjson",
                    "episode_manifest_ref": f"{prefix}/episode_manifest.cjson",
                }
            if native_shadow is not None:
                native_shadow.prepare_episode(identity)
                native_prepared = True
                native_prefix = f"meta/v25p/native/episodes/{identity.episode_uuid}"
                metadata_payload["v25p_native_episode"] = {
                    "schema": "dexteleop.v25p.native_episode_binding.v1",
                    "episode_uuid": identity.episode_uuid,
                    "episode_identity_hash": identity.semantic_content_hash,
                    "episode_manifest_ref": f"{native_prefix}/episode_manifest.cjson",
                }
        except BaseException:
            if native_shadow is not None and native_prepared:
                native_shadow.discard_episode(identity)
            if store is not None and sidecar_prepared:
                store.discard_episode(identity)
            raise
        index: int | None = None
        legacy_episode_open = False
        try:
            index = self.writer.start_episode(metadata=metadata_payload)
            legacy_episode_open = True
            if int(index) != identity.legacy_episode_index:
                self.writer.discard_episode(int(index))
                legacy_episode_open = False
                raise RuntimeError(
                    "UMI legacy episode index changed after owner identity preparation."
                )
            if store is not None:
                store.bind_prepared_episode(
                    identity,
                    bound_at=time.monotonic(),
                    bound_clock_domain_id="python_monotonic",
                )
            if native_shadow is not None:
                native_shadow.bind_prepared_episode(
                    identity,
                    bound_at=time.monotonic(),
                    bound_clock_domain_id="python_monotonic",
                )
            self.encoder.reset_episode(index)
            self.pool.start()
        except BaseException:
            if legacy_episode_open and index is not None:
                self.writer.discard_episode(int(index))
            if store is not None and sidecar_prepared:
                store.discard_episode(identity)
            if native_shadow is not None and native_prepared:
                native_shadow.discard_episode(identity)
            raise
        assert index is not None
        self.episode_index = index
        self._active_episode_identity = identity
        self._v25p_native_shadow_active = native_shadow is not None
        self._v25p_native_shadow_error = ""
        self.frame_count = 0
        self.accepted_sample_count = 0
        self._pending_sample = None
        self.action_continuity_break_count = 0
        self.continuity_dropped_source_sample_count = 0
        self.action_continuity_break_reasons.clear()
        self.sample_attempt_count = 0
        self.rejection_counts.clear()
        self._externally_noted_pending_samples = 0
        self.episode_started_at = time.time()
        self.last_outcome = "recording"
        self.last_message = ""
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
        if (
            self.v25p_sidecar_store is not None
            and resolved_root.resolve() != self.writer.root_dir.resolve()
        ):
            raise RuntimeError(
                "A UMI recorder with v2.5-p sidecars cannot change dataset_root; "
                "construct a new recorder/store pair."
            )
        if (
            self.v25p_native_shadow is not None
            and resolved_root.resolve() != self.writer.root_dir.resolve()
        ):
            raise RuntimeError(
                "A UMI recorder with a v2.5-p native shadow cannot change "
                "dataset_root; construct a new recorder/shadow pair."
            )
        prepare_dataset_capture_writable(resolved_root)
        if next_config.tactile_artifact_integrity:
            assert_dataset_capture_integrity_compatible(resolved_root)
        _validate_umi_dataset_admission(
            resolved_root,
            task_id=next_config.task_id,
        )
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
                tactile_compression=next_config.tactile_compression,
                tactile_artifact_integrity=next_config.tactile_artifact_integrity,
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
        if not rejection_reasons:
            self._externally_noted_pending_samples += 1

    @staticmethod
    def _sample_rejection_key(exc: StrictV22SampleRejection) -> str:
        if exc.reason == "stale_measured_state" and exc.age_sec is not None:
            if exc.age_sec < 0.0:
                return f"{exc.side}:future_gripper_measurement:{exc.age_sec:.3f}"
            return f"{exc.side}:stale_gripper_measurement:{exc.age_sec:.3f}"
        if exc.reason == "missing_measured_field":
            return f"{exc.side}:missing_gripper_measurement"
        if exc.reason == "missing_measured_timestamp":
            return f"{exc.side}:missing_gripper_measurement_timestamp"
        return f"{exc.side}:{exc.reason}"

    def _reject_sample_contract(self, exc: StrictV22SampleRejection) -> bool:
        """Account and isolate a sample-level v2.2 contract rejection."""

        externally_noted = int(self._externally_noted_pending_samples)
        if externally_noted > 0:
            self._externally_noted_pending_samples = externally_noted - 1
        else:
            self.sample_attempt_count += 1
            self.rejection_counts.update((self._sample_rejection_key(exc),))
        self.break_action_continuity(
            reason=f"sample_rejected:{exc.side}:{exc.reason}"
        )
        self.last_outcome = "recording"
        self.last_message = f"sample rejected: {exc}"
        return False

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

    def submit(self, sample: UmiSample) -> bool:
        if self.episode_index is None or sample.episode_index != self.episode_index:
            raise RuntimeError("UMI sample does not belong to the active episode")
        # Direct recorder users may not call note_snapshot_attempt(); keep the
        # accounting internally consistent without double-counting controller calls.
        if int(sample.frame_index) != self.next_frame_index:
            raise RuntimeError(
                "UMI accepted sample frame_index is not contiguous: "
                f"got {sample.frame_index}, expected {self.next_frame_index}"
            )
        try:
            validate_strict_v22_measured_state(
                sample.hand_states,
                layout=self.policy_io_layout,
                sample_timestamp=float(sample.timestamp),
                # Match UmiSnapshotAssembler's causal future-measurement
                # boundary for direct recorder callers.
                future_tolerance_sec=1e-3,
            )
        except StrictV22SampleRejection as exc:
            return self._reject_sample_contract(exc)
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
        if self._externally_noted_pending_samples > 0:
            self._externally_noted_pending_samples -= 1
        else:
            self.sample_attempt_count += 1
        self.accepted_sample_count += 1
        previous = self._pending_sample
        self._pending_sample = sample
        if previous is None:
            return True
        aligned = replace(
            previous,
            arm_action_poses=sample.poses,
            arm_action_source_frame_index=int(sample.frame_index),
            arm_action_source_timestamp=float(sample.timestamp),
        )
        jobs = self.writer.plan_sample_jobs(aligned)
        shadow = getattr(self, "v25p_native_shadow", None)
        identity = self._active_episode_identity
        if (
            shadow is not None
            and identity is not None
            and self._v25p_native_shadow_active
        ):
            error = append_shadow_from_legacy_jobs(
                shadow, identity, source_sample=aligned, jobs=jobs,
                record_error=lambda error: setattr(self, "_v25p_native_shadow_error", error),
            )
            if error:
                self._v25p_native_shadow_active = False
        for job in jobs:
            self.pool.submit(job)
        self.frame_count += 1
        return True

    def stop_episode(self, *, reason: str, discard: bool = False) -> int | None:
        if self.episode_index is None:
            return None
        index = self.episode_index
        identity = self._active_episode_identity
        native_shadow = getattr(self, "v25p_native_shadow", None)
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
                if self.v25p_sidecar_store is not None and identity is not None:
                    self.v25p_sidecar_store.discard_episode(identity)
                if (
                    native_shadow is not None
                    and identity is not None
                    and getattr(self, "_v25p_native_shadow_active", False)
                ):
                    native_shadow.discard_episode(identity)
                    self._v25p_native_shadow_active = False
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
            if self.v25p_sidecar_store is not None and identity is not None:
                try:
                    self.v25p_sidecar_store.close_unsealed(identity)
                except BaseException:
                    pass
            if (
                native_shadow is not None
                and identity is not None
                and getattr(self, "_v25p_native_shadow_active", False)
            ):
                try:
                    native_shadow.close_unsealed(identity)
                except BaseException:
                    pass
                self._v25p_native_shadow_active = False
            self.last_outcome = "error"
            self.last_message = f"{type(exc).__name__}: {exc}"
            raise
        if (
            outcome == "saved"
            and self.v25p_sidecar_store is not None
            and identity is not None
        ):
            try:
                self.v25p_sidecar_store.finalize_after_legacy_episode(
                    identity,
                    sealed_at=time.time(),
                    sealed_clock_domain_id="local_wall_clock",
                )
            except BaseException as sidecar_exc:
                self.last_message = (
                    f"{message}; v2.5-p sidecar incomplete: "
                    f"{type(sidecar_exc).__name__}: {sidecar_exc}"
                )
                try:
                    self.v25p_sidecar_store.close_unsealed(identity)
                except BaseException:
                    pass
        if (
            outcome == "saved"
            and native_shadow is not None
            and identity is not None
            and getattr(self, "_v25p_native_shadow_active", False)
        ):
            try:
                finalize_shadow_after_legacy(
                    native_shadow,
                    identity,
                    dataset_root=self.writer.root_dir,
                    episode_index=index,
                    incomplete_message="",
                    record_error=lambda error: setattr(self, "_v25p_native_shadow_error", error),
                )
            finally:
                self._v25p_native_shadow_active = False
        self.episode_index = None
        self._active_episode_identity = None
        self.episode_started_at = None
        self.last_episode_index = index
        self.last_frame_count = frame_count
        self.last_accepted_sample_count = accepted_sample_count
        self.last_attempt_count = int(self.sample_attempt_count)
        self.last_rejection_counts = dict(sorted(self.rejection_counts.items()))
        self.last_outcome = outcome
        if not self.last_message.startswith(f"{message}; v2.5-p sidecar incomplete:"):
            self.last_message = message
        if getattr(self, "_v25p_native_shadow_error", ""):
            self.last_message += (
                "; v2.5-p native shadow incomplete: "
                f"{self._v25p_native_shadow_error}"
            )
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
