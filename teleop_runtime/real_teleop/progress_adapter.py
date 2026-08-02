"""Default-off RealTeleop adapter for collection-progress Episodes.

The integration is explicitly enabled by a progress profile and remains
LINEAR-first for the real-robot application.  It binds the generic
``collection_runtime.progress`` core to the existing formal-dataset Episode
owner without moving identity or commit ownership.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
import shutil
from typing import Any

from collection_runtime.immutable_canonical_store import fsync_directory
from collection_runtime.progress import (
    AbortEpisodeProgressRequest,
    AdvanceProgressStepRequest,
    CompleteEpisodeProgressRequest,
    ProgressController,
    ProgressEffectRequest,
    ProgressFactStore,
    ProgressRequestContext,
    ReportAttemptFailureRequest,
    ResumeFailedAttemptRequest,
    StartEpisodeProgressRequest,
    UndoLastProgressAdvanceRequest,
)
from collection_runtime.progress.profiles import (
    ProgressProfileCatalogError,
    inspect_dataset_progress_profile,
    list_managed_task_profiles,
    load_managed_task_profile,
    resolve_task_progress_profile_reference,
    resolve_task_stage_profile_reference,
)
from collection_runtime.sample_acceptance import AcceptedSampleReceipt
from control_station.real_teleop.progress import (
    ProgressControlAvailability,
    ProgressProfileOption,
    build_real_teleop_progress_ui_payload,
)
from dataset_contracts.causal_snapshot import LOCAL_WALL_CLOCK
from dataset_contracts.episode_identity import EpisodeCaptureIdentity
from dataset_contracts.progress_facts import (
    HoldPauseEffectReceiptV1,
    ProgressAdvanceEventV1,
    ProgressOperationReceiptV1,
    ResumeAfterFailureReceiptV1,
)
from dataset_contracts.progress_profile import (
    ProgressEpisodeBindingV1,
    TaskProgressProfileV1,
)
from dataset_contracts.progress_stages import (
    ProgressStageEntryBindingV1,
    TaskStageProfileV1,
    validate_progress_stage_entry_binding,
    validate_task_stage_profile,
)


REAL_TELEOP_PROGRESS_CLOCK_DOMAIN = LOCAL_WALL_CLOCK
REAL_TELEOP_PROGRESS_RELATIVE_ROOT = Path("meta/progress/episodes")
REAL_TELEOP_PROGRESS_REFERENCE_SCHEMA_V1 = (
    "dexteleop.real_teleop_progress_episode_reference.v1"
)

_NORMAL_COMPLETION_REASONS = frozenset({"manual_key_stop"})
_SAFETY_STOP_REASONS = frozenset(
    {
        "force_collision",
        "joint_hard_limit",
        "position_force_guard",
        "runtime_fault",
        "runtime_estop",
        "safety_stop",
    }
)
_MANUAL_ABORT_REASONS = frozenset(
    {"manual_key_stop", "quit_requested", "shutdown"}
)


def load_real_teleop_progress_profile(reference: str) -> TaskProgressProfileV1:
    """Load an explicit path or ``catalog:<profile_id>`` reference."""

    try:
        profile = resolve_task_progress_profile_reference(reference)
    except ProgressProfileCatalogError as exc:
        raise ValueError(f"Invalid collection progress profile: {exc}") from exc
    validate_real_teleop_initial_progress_profile(profile)
    return profile


def validate_real_teleop_initial_progress_profile(
    profile: TaskProgressProfileV1,
    *,
    collection_task_id: str | None = None,
) -> None:
    """Reject progress capabilities not implemented by the RealTeleop adapter.

    ``collection_task_id`` is retained as a compatibility argument for callers
    that already pass it, but progress-profile identity is intentionally
    independent from the formal dataset task/name.  Dataset-wide profile
    consistency is enforced separately by the immutable profile binding and
    semantic-content hash admission.
    """

    if not isinstance(profile, TaskProgressProfileV1):
        raise TypeError("collection progress profile must be TaskProgressProfileV1.")
    if profile.topology_kind != "LINEAR":
        raise ValueError(
            "The RealTeleop progress adapter only accepts a LINEAR profile."
        )
    for step in profile.steps:
        requirement = step.arm_control
        if requirement is not None and not requirement.teleop_sides:
            raise ValueError(
                "RealTeleop progress arm_control must leave at least one arm in "
                f"teleop; step={step.progress_step_id!r}."
            )
    del collection_task_id


def resolve_real_teleop_progress_profile(
    value: object,
    *,
    collection_task_id: str | None = None,
) -> TaskProgressProfileV1 | None:
    """Normalize configuration without binding progress to dataset naming.

    The optional task argument remains in the call surface for compatibility;
    the selected profile owns its own task semantics and is not rewritten or
    rejected because the formal dataset uses another task/name.
    """

    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    profile = (
        value
        if isinstance(value, TaskProgressProfileV1)
        else load_real_teleop_progress_profile(str(value))
    )
    validate_real_teleop_initial_progress_profile(profile)
    return profile


def resolve_real_teleop_task_stage_profile(
    progress_profile: TaskProgressProfileV1 | None,
    value: object = None,
) -> TaskStageProfileV1 | None:
    """Resolve the exact ordered-stage profile owned by a coarse profile."""

    if progress_profile is None:
        if value is not None and str(value).strip():
            raise ValueError(
                "--collection-stage-profile requires a collection progress "
                "profile."
            )
        return None
    if isinstance(value, TaskStageProfileV1):
        result = value
    elif value is not None and str(value).strip():
        try:
            result = resolve_task_stage_profile_reference(
                str(value),
                progress_profile=progress_profile,
            )
        except ProgressProfileCatalogError as exc:
            raise ValueError(
                f"Invalid collection stage profile: {exc}"
            ) from exc
    else:
        matches = tuple(
            bundle.stage_profile
            for bundle in list_managed_task_profiles(
                topology_kind="LINEAR"
            )
            if (
                bundle.progress_profile.semantic_content_hash
                == progress_profile.semantic_content_hash
                and bundle.stage_profile is not None
            )
        )
        result = matches[0] if len(matches) == 1 else None
    if result is not None:
        validate_task_stage_profile(
            progress_profile=progress_profile,
            stage_profile=result,
        )
    return result


def resolve_real_teleop_progress_entry(
    *,
    progress_profile: TaskProgressProfileV1,
    stage_profile: TaskStageProfileV1 | None,
    entry_step_id: object = "",
    entry_stage_id: object = "",
    entry_condition: object = "NOMINAL",
) -> tuple[str, str | None, str]:
    """Resolve one capture entry without silently guessing recovery state."""

    step_id = str(entry_step_id or "").strip() or progress_profile.initial_step_id
    if step_id not in progress_profile.step_by_id:
        raise ValueError(f"Unknown progress entry step {step_id!r}.")
    condition = str(entry_condition or "NOMINAL").strip().upper()
    if stage_profile is None:
        if str(entry_stage_id or "").strip():
            raise ValueError(
                "A fine-stage entry requires a task stage profile."
            )
        if condition != "NOMINAL":
            raise ValueError(
                "RECOVERY_FROM_FAILURE requires a task stage profile."
            )
        return step_id, None, condition
    stages = stage_profile.stages_for_step(step_id)
    if not stages:
        raise ValueError(
            f"No fine stages are defined for progress step {step_id!r}."
        )
    stage_id = str(entry_stage_id or "").strip() or stages[0].stage_id
    probe = ProgressStageEntryBindingV1(
        episode_capture_identity_hash="0" * 64,
        task_stage_profile_hash=stage_profile.semantic_content_hash,
        entry_stage_id=stage_id,
        entry_condition=condition,
    )
    validate_progress_stage_entry_binding(
        progress_profile=progress_profile,
        progress_entry_step_id=step_id,
        stage_profile=stage_profile,
        stage_binding=probe,
    )
    return step_id, stage_id, condition


def list_real_teleop_managed_progress_profiles(
    *,
    collection_task_id: str | None = None,
) -> tuple[TaskProgressProfileV1, ...]:
    """List all GUI-selectable managed LINEAR profiles.

    ``collection_task_id`` is intentionally not a filter.  A formal dataset
    task/name describes the capture dataset; a progress profile is a separate
    operator-selected long-horizon semantic contract.
    """

    del collection_task_id
    profiles = tuple(
        bundle.progress_profile
        for bundle in list_managed_task_profiles(topology_kind="LINEAR")
    )
    for profile in profiles:
        validate_real_teleop_initial_progress_profile(profile)
    return tuple(sorted(profiles, key=lambda item: item.profile_id))


@lru_cache(maxsize=32)
def list_real_teleop_managed_progress_profile_options(
    *,
    collection_task_id: str | None = None,
) -> tuple[ProgressProfileOption, ...]:
    """Build immutable GUI options for one formal task.

    Managed catalog assets do not change during one process lifetime. Caching
    prevents the startup/Home polling loop from reopening profile resources at
    50 Hz. The retained task argument is not a semantic filter.
    """

    return tuple(
        ProgressProfileOption(
            profile_id=profile.profile_id,
            profile_version=profile.profile_version,
            task_id=profile.task_id,
            display_name=profile.profile_id,
            step_count=len(profile.steps),
        )
        for profile in list_real_teleop_managed_progress_profiles(
            collection_task_id=collection_task_id,
        )
    )


def load_real_teleop_managed_progress_profile(
    profile_id: str,
    *,
    collection_task_id: str | None = None,
) -> TaskProgressProfileV1:
    """Resolve one GUI selection without changing formal task identity."""

    del collection_task_id

    try:
        profile = load_managed_task_profile(profile_id).progress_profile
    except ProgressProfileCatalogError as exc:
        raise ValueError(f"Invalid managed progress profile: {exc}") from exc
    validate_real_teleop_initial_progress_profile(profile)
    return profile


def _dataset_recorder_matches_progress_selection(
    *,
    args: argparse.Namespace,
    dataset_recorder: object | None,
) -> bool:
    if dataset_recorder is None:
        return False
    recorder_root = getattr(dataset_recorder, "dataset_root", None)
    storage_config = getattr(dataset_recorder, "writer_storage_config", None)
    recorder_task_id = getattr(storage_config, "task_id", None)
    if recorder_root is None or recorder_task_id is None:
        return False
    return bool(
        Path(recorder_root).expanduser().resolve()
        == Path(args.dataset_root).expanduser().resolve()
        and str(recorder_task_id) == str(args.collection_task_id)
    )


def select_real_teleop_managed_progress_profile(
    *,
    args: argparse.Namespace,
    dataset_recorder: object | None,
    profile_id: str,
) -> TaskProgressProfileV1:
    """Apply one admitted GUI selection without starting an Episode."""

    if not bool(getattr(args, "collection_progress_gui", False)):
        raise RuntimeError("collection progress GUI selection is disabled")
    selected = load_real_teleop_managed_progress_profile(
        profile_id,
        collection_task_id=str(args.collection_task_id),
    )
    recorder_matches = _dataset_recorder_matches_progress_selection(
        args=args,
        dataset_recorder=dataset_recorder,
    )
    if recorder_matches:
        selector = getattr(dataset_recorder, "select_progress_profile", None)
        if not callable(selector):
            raise RuntimeError(
                "The active dataset recorder cannot accept progress profile selection."
            )
        selector(selected)
        args.collection_progress_profile = selected
    else:
        inspect_dataset_progress_profile(
            Path(args.dataset_root),
            requested_profile=selected,
        )
        args.collection_progress_profile = selected
    return selected


def build_real_teleop_startup_progress_ui_payload(
    *,
    args: argparse.Namespace,
    dataset_recorder: object | None,
    profile_selection_enabled: bool = True,
) -> dict[str, object]:
    """Build the pre-Home projection used before the main control loop exists."""

    recorder_matches = _dataset_recorder_matches_progress_selection(
        args=args,
        dataset_recorder=dataset_recorder,
    )
    if recorder_matches:
        profile = getattr(dataset_recorder, "progress_profile", None)
        profile_selection_locked = bool(
            getattr(
                dataset_recorder,
                "progress_profile_selection_locked",
                False,
            )
        )
        profile_binding_source = str(
            getattr(
                dataset_recorder,
                "progress_profile_binding_source",
                "",
            )
        )
    else:
        admission = inspect_dataset_progress_profile(
            Path(args.dataset_root),
            requested_profile=getattr(
                args,
                "collection_progress_profile",
                None,
            ),
        )
        profile = admission.profile
        profile_selection_locked = admission.locked
        profile_binding_source = admission.source
    available_profiles = list_real_teleop_managed_progress_profile_options(
        collection_task_id=str(args.collection_task_id),
    )
    payload = build_real_teleop_progress_ui_payload(
        profile=profile,
        episode=None,
        availability=ProgressControlAvailability(),
        gui_enabled=bool(getattr(args, "collection_progress_gui", False)),
        available_profiles=available_profiles,
        profile_selection_enabled=profile_selection_enabled,
        profile_selection_locked=profile_selection_locked,
        profile_binding_source=profile_binding_source,
        arm_control_mode=str(
            getattr(
                args,
                "collection_progress_arm_control_mode",
                "profile",
            )
        ),
        arm_control_mode_selection_enabled=profile_selection_enabled,
    )
    if (
        bool(getattr(args, "collection_progress_gui", False))
        and profile is None
        and not profile_selection_locked
        and not available_profiles
    ):
        payload["message"] = (
            "当前没有可选的 managed LINEAR progress profile；"
            "请检查 progress profile catalog。"
        )
    if (
        bool(getattr(args, "collection_progress_gui", False))
        and not profile_selection_enabled
        and not profile_selection_locked
        and available_profiles
    ):
        payload["message"] = (
            "设备仍在启动；完成连接检查后即可在 runtime Home 前选择进度 profile。"
        )
    return payload


def _request_context(
    *,
    request_id: str,
    effective_at: float,
    origin: str,
    available_at: float | None = None,
    actor_id: str = "teleop_runtime.real_teleop.dataset_recorder",
) -> ProgressRequestContext:
    return ProgressRequestContext(
        request_id=request_id,
        effective_at=float(effective_at),
        effective_clock_domain_id=REAL_TELEOP_PROGRESS_CLOCK_DOMAIN,
        available_at=float(
            effective_at if available_at is None else available_at
        ),
        available_clock_domain_id=REAL_TELEOP_PROGRESS_CLOCK_DOMAIN,
        actor_id=actor_id,
        origin=origin,
    )


def _termination_cause(reason: str) -> str:
    normalized = str(reason).strip().lower()
    if normalized in _SAFETY_STOP_REASONS:
        return "SAFETY_STOP"
    if normalized in _MANUAL_ABORT_REASONS:
        return "MANUAL_STOP"
    return "ABORTED"


class RealTeleopProgressEpisode:
    """One subordinate progress participant prepared by the formal Episode owner."""

    def __init__(
        self,
        *,
        dataset_root: str | Path,
        identity: EpisodeCaptureIdentity,
        profile: TaskProgressProfileV1,
        stage_profile: TaskStageProfileV1 | None = None,
        entry_step_id: str = "",
        entry_stage_id: str = "",
        entry_condition: str = "NOMINAL",
    ) -> None:
        if not isinstance(identity, EpisodeCaptureIdentity):
            raise TypeError("identity must be EpisodeCaptureIdentity.")
        validate_real_teleop_initial_progress_profile(profile)
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.identity = identity
        self.profile = profile
        self.stage_profile = stage_profile
        (
            resolved_step_id,
            resolved_stage_id,
            resolved_condition,
        ) = resolve_real_teleop_progress_entry(
            progress_profile=profile,
            stage_profile=stage_profile,
            entry_step_id=entry_step_id,
            entry_stage_id=entry_stage_id,
            entry_condition=entry_condition,
        )
        self.relative_dir = (
            REAL_TELEOP_PROGRESS_RELATIVE_ROOT / identity.episode_uuid
        )
        self.episode_dir = (self.dataset_root / self.relative_dir).resolve()
        expected_parent = (
            self.dataset_root / REAL_TELEOP_PROGRESS_RELATIVE_ROOT
        ).resolve()
        if self.episode_dir.parent != expected_parent:
            raise ValueError("Progress Episode path escaped its owned root.")
        default_entry = bool(
            resolved_step_id == profile.initial_step_id
            and resolved_condition == "NOMINAL"
            and (
                stage_profile is None
                or resolved_stage_id
                == stage_profile.stages_for_step(resolved_step_id)[0].stage_id
            )
        )
        self.binding = ProgressEpisodeBindingV1(
            episode_capture_identity_hash=identity.semantic_content_hash,
            task_progress_profile_hash=profile.semantic_content_hash,
            bound_clock_domain_id=REAL_TELEOP_PROGRESS_CLOCK_DOMAIN,
            recording_scope=("FULL_TASK" if default_entry else "EVENT_WINDOW"),
            entry_step_id=resolved_step_id,
            entry_mode=(
                "PROFILE_INITIAL"
                if default_entry
                else (
                    "SCENARIO_SETUP"
                    if resolved_condition == "RECOVERY_FROM_FAILURE"
                    else "ACTUAL_TRUNCATED_HISTORY"
                )
            ),
            history_complete=default_entry,
            assumed_completed_path_step_ids=(
                profile.ancestor_path_to(resolved_step_id)
                if default_entry
                else ()
            ),
            truncated_at_start=not default_entry,
        )
        self.stage_entry_binding = (
            None
            if stage_profile is None or resolved_stage_id is None
            else ProgressStageEntryBindingV1(
                episode_capture_identity_hash=identity.semantic_content_hash,
                task_stage_profile_hash=stage_profile.semantic_content_hash,
                entry_stage_id=resolved_stage_id,
                entry_condition=resolved_condition,
            )
        )
        self.store = ProgressFactStore(
            episode_dir=self.episode_dir,
            identity=identity,
            profile=profile,
            binding=self.binding,
            stage_profile=stage_profile,
            stage_entry_binding=self.stage_entry_binding,
        )
        self.controller = ProgressController(store=self.store)
        self._bound = False
        self._closed = False

    @property
    def metadata_reference(self) -> dict[str, Any]:
        prefix = self.relative_dir.as_posix()
        payload = {
            "schema": REAL_TELEOP_PROGRESS_REFERENCE_SCHEMA_V1,
            "episode_uuid": self.identity.episode_uuid,
            "episode_identity_hash": self.identity.semantic_content_hash,
            "task_progress_profile_hash": self.profile.semantic_content_hash,
            "progress_episode_binding_hash": self.binding.semantic_content_hash,
            "dataset_profile_ref": "meta/progress/profile.cjson",
            "profile_ref": f"{prefix}/profile.cjson",
            "binding_ref": f"{prefix}/binding.cjson",
            "manifest_ref": f"{prefix}/manifest.cjson",
        }
        if (
            self.stage_profile is not None
            and self.stage_entry_binding is not None
        ):
            payload.update(
                {
                    "task_stage_profile_hash": (
                        self.stage_profile.semantic_content_hash
                    ),
                    "stage_entry_binding_hash": (
                        self.stage_entry_binding.semantic_content_hash
                    ),
                    "dataset_stage_profile_ref": (
                        "meta/progress/stages.cjson"
                    ),
                    "stage_profile_ref": f"{prefix}/stages.cjson",
                    "stage_entry_ref": f"{prefix}/stage_entry.cjson",
                }
            )
        return payload

    @property
    def bound(self) -> bool:
        return self._bound

    def bind(self, *, started_at: float) -> None:
        if self._closed:
            raise RuntimeError("Progress Episode is closed.")
        result = self.controller.start(
            StartEpisodeProgressRequest(
                attempt_id=f"attempt:{self.identity.episode_uuid}:1",
            ),
            context=_request_context(
                request_id=f"progress:start:{self.identity.episode_uuid}",
                effective_at=started_at,
                origin="episode_owner_start",
            ),
        )
        if result.state.phase != "ACTIVE_ATTEMPT":
            raise RuntimeError("Progress Episode did not enter ACTIVE_ATTEMPT.")
        self._bound = True

    @property
    def state(self):
        return self.controller.state

    @property
    def allowed_next_step_ids(self) -> tuple[str, ...]:
        step_id = self.state.current_progress_step_id
        if step_id is None or self.state.phase != "ACTIVE_ATTEMPT":
            return ()
        return self.profile.allowed_next_step_ids(step_id)

    @property
    def can_undo_last_advance(self) -> bool:
        return bool(
            self.state.phase == "ACTIVE_ATTEMPT"
            and self.state.last_effective_advance_fact_id is not None
        )

    def preview_advance_target(
        self,
        target_progress_step_id: str | None = None,
    ) -> str:
        allowed = self.allowed_next_step_ids
        if not allowed:
            raise RuntimeError("The current progress step is terminal.")
        if target_progress_step_id is None:
            if len(allowed) != 1:
                raise RuntimeError(
                    "A branch point requires target_progress_step_id."
                )
            return allowed[0]
        target = str(target_progress_step_id).strip()
        if target not in allowed:
            raise RuntimeError("Requested progress target is not legal.")
        return target

    def preview_undo_target(self) -> str:
        advance_fact_id = self.state.last_effective_advance_fact_id
        if advance_fact_id is None:
            raise RuntimeError("No fresh progress Advance is available to undo.")
        fact = self.store.fact_by_id(advance_fact_id)
        if fact is None or not isinstance(fact.payload, ProgressAdvanceEventV1):
            raise RuntimeError("The fresh progress Advance fact is unavailable.")
        return fact.payload.from_progress_step_id

    def advance(
        self,
        *,
        intent_id: str,
        effective_at: float,
        target_progress_step_id: str | None = None,
    ):
        return self.controller.advance(
            AdvanceProgressStepRequest(
                to_attempt_id=(
                    f"attempt:{self.identity.episode_uuid}:advance:{intent_id}"
                ),
                target_progress_step_id=target_progress_step_id,
            ),
            context=_request_context(
                request_id=f"progress:advance:{intent_id}",
                effective_at=effective_at,
                origin="online_operator",
                actor_id="control_station.real_teleop",
            ),
        )

    def undo_last_advance(
        self,
        *,
        intent_id: str,
        effective_at: float,
    ):
        advance_fact_id = self.state.last_effective_advance_fact_id
        if advance_fact_id is None:
            raise RuntimeError("No fresh progress Advance is available to undo.")
        return self.controller.undo_last_advance(
            UndoLastProgressAdvanceRequest(
                advance_fact_id=advance_fact_id,
            ),
            context=_request_context(
                request_id=f"progress:undo:{intent_id}",
                effective_at=effective_at,
                origin="online_operator",
                actor_id="control_station.real_teleop",
            ),
        )

    def report_failure(
        self,
        *,
        intent_id: str,
        effective_at: float,
        available_at: float | None = None,
        timing_semantics: str = "OPERATOR_REPORT_TIME",
        capture_failure_trigger: bool = True,
    ) -> ProgressEffectRequest:
        result = self.controller.report_failure(
            ReportAttemptFailureRequest(
                timing_semantics=timing_semantics,
                capture_failure_trigger=capture_failure_trigger,
            ),
            context=_request_context(
                request_id=f"progress:failure:{intent_id}",
                effective_at=effective_at,
                available_at=available_at,
                origin="online_operator",
                actor_id="control_station.real_teleop",
            ),
        )
        if len(result.effect_requests) != 1:
            raise RuntimeError(
                "Accepted progress Failure did not publish one Hold/Pause effect."
            )
        return result.effect_requests[0]

    def record_applied_hold_pause(
        self,
        *,
        effect_request: ProgressEffectRequest,
        requested_at: float,
        application_hold_effective_at: float,
        formal_capture_pause_effective_at: float,
        available_at: float,
    ):
        request_id = f"progress:hold_pause:{effect_request.idempotency_key}"
        receipt = HoldPauseEffectReceiptV1(
            source_failure_fact_id=effect_request.source_failure_fact_id,
            application_hold=ProgressOperationReceiptV1(
                operation="APPLICATION_HOLD",
                status="APPLIED",
                requested_at=requested_at,
                effective_at=application_hold_effective_at,
                clock_domain_id=REAL_TELEOP_PROGRESS_CLOCK_DOMAIN,
            ),
            formal_capture_pause=ProgressOperationReceiptV1(
                operation="FORMAL_CAPTURE_PAUSE",
                status="APPLIED",
                requested_at=requested_at,
                effective_at=formal_capture_pause_effective_at,
                clock_domain_id=REAL_TELEOP_PROGRESS_CLOCK_DOMAIN,
            ),
        )
        return self.controller.record_hold_pause_receipt(
            receipt,
            context=_request_context(
                request_id=request_id,
                effective_at=max(
                    application_hold_effective_at,
                    formal_capture_pause_effective_at,
                ),
                available_at=available_at,
                origin="real_teleop_effect_receipt",
            ),
        )

    def record_applied_resume(
        self,
        *,
        resume_operation_id: str,
        requested_at: float,
        application_resume_effective_at: float,
        formal_capture_resume_effective_at: float,
        available_at: float,
    ):
        failure_fact_id = self.state.pending_failure_fact_id
        hold_pause_fact_id = self.state.hold_pause_receipt_fact_id
        if failure_fact_id is None or hold_pause_fact_id is None:
            raise RuntimeError("No applied failed Attempt is available to resume.")
        receipt = ResumeAfterFailureReceiptV1(
            source_failure_fact_id=failure_fact_id,
            hold_pause_receipt_fact_id=hold_pause_fact_id,
            resume_operation_id=resume_operation_id,
            application_resume=ProgressOperationReceiptV1(
                operation="APPLICATION_RESUME",
                status="APPLIED",
                requested_at=requested_at,
                effective_at=application_resume_effective_at,
                clock_domain_id=REAL_TELEOP_PROGRESS_CLOCK_DOMAIN,
            ),
            formal_capture_resume=ProgressOperationReceiptV1(
                operation="FORMAL_CAPTURE_RESUME",
                status="APPLIED",
                requested_at=requested_at,
                effective_at=formal_capture_resume_effective_at,
                clock_domain_id=REAL_TELEOP_PROGRESS_CLOCK_DOMAIN,
            ),
        )
        return self.controller.record_resume_receipt(
            receipt,
            context=_request_context(
                request_id=f"progress:resume:{resume_operation_id}",
                effective_at=max(
                    application_resume_effective_at,
                    formal_capture_resume_effective_at,
                ),
                available_at=available_at,
                origin="real_teleop_effect_receipt",
            ),
        )

    def retry_on_accepted_sample(
        self,
        receipt: AcceptedSampleReceipt,
        *,
        available_at: float,
    ):
        if not isinstance(receipt, AcceptedSampleReceipt):
            raise TypeError("receipt must be AcceptedSampleReceipt.")
        if receipt.episode_index != self.identity.legacy_episode_index:
            raise RuntimeError(
                "Accepted sample receipt does not belong to this progress Episode."
            )
        resume_fact_id = self.state.resume_receipt_fact_id
        if self.state.phase != "RESUME_RECEIVED" or resume_fact_id is None:
            return None
        return self.controller.resume_failed_attempt(
            ResumeFailedAttemptRequest(
                new_attempt_id=(
                    f"attempt:{self.identity.episode_uuid}:retry:"
                    f"{receipt.sample_index}"
                ),
            ),
            context=_request_context(
                request_id=f"progress:retry:{resume_fact_id}",
                effective_at=receipt.sample_timestamp,
                available_at=available_at,
                origin="writer_accepted_sample",
            ),
        )

    def terminate(self, *, reason: str, effective_at: float) -> None:
        if not self._bound:
            raise RuntimeError("Progress Episode was not bound.")
        normalized = str(reason).strip().lower()
        context = _request_context(
            request_id=f"progress:terminate:{self.identity.episode_uuid}",
            effective_at=effective_at,
            origin="episode_owner_stop",
        )
        normal_completion = bool(
            normalized in _NORMAL_COMPLETION_REASONS
            and self.state.phase == "ACTIVE_ATTEMPT"
            and self.state.current_progress_step_id
            in self.profile.terminal_step_ids
        )
        if normal_completion:
            self.controller.complete(
                CompleteEpisodeProgressRequest(),
                context=context,
            )
        else:
            self.controller.abort(
                AbortEpisodeProgressRequest(
                    termination_cause=_termination_cause(normalized),
                ),
                context=context,
            )

    def seal(self, *, sealed_at: float):
        return self.controller.seal(
            sealed_at=sealed_at,
            sealed_clock_domain_id=REAL_TELEOP_PROGRESS_CLOCK_DOMAIN,
        )

    def close_unsealed(self) -> None:
        if self._closed:
            return
        self.store.close()
        self._closed = True

    def discard(self) -> None:
        """Discard only this exact unsealed subordinate directory."""

        if self.store.sealed:
            raise RuntimeError("A sealed progress Episode cannot be discarded.")
        self.close_unsealed()
        if self.episode_dir.exists():
            shutil.rmtree(self.episode_dir)
            if self.episode_dir.parent.exists():
                fsync_directory(self.episode_dir.parent)


__all__ = [
    "REAL_TELEOP_PROGRESS_CLOCK_DOMAIN",
    "REAL_TELEOP_PROGRESS_REFERENCE_SCHEMA_V1",
    "REAL_TELEOP_PROGRESS_RELATIVE_ROOT",
    "RealTeleopProgressEpisode",
    "build_real_teleop_startup_progress_ui_payload",
    "list_real_teleop_managed_progress_profile_options",
    "load_real_teleop_progress_profile",
    "resolve_real_teleop_progress_profile",
    "select_real_teleop_managed_progress_profile",
    "validate_real_teleop_initial_progress_profile",
]
