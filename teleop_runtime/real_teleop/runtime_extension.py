"""Dependency-neutral application extension seam for the RealTeleop owner loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from collection_runtime.sample_acceptance import SampleTerminalEvent

from .components import RealTeleopComponents


@dataclass(frozen=True)
class RealTeleopDatasetSampleAdmission:
    """One extension's narrow gate in front of the existing formal writer."""

    allowed: bool = True
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))


def runtime_extension_dataset_sample_admission(
    extension: object | None,
    *,
    step_result: object,
    now: float,
) -> RealTeleopDatasetSampleAdmission:
    """Call the optional sample gate while preserving the default path."""

    if extension is None:
        return RealTeleopDatasetSampleAdmission()
    hook = getattr(extension, "dataset_sample_admission", None)
    if not callable(hook):
        return RealTeleopDatasetSampleAdmission()
    result = hook(step_result=step_result, now=float(now))
    if not isinstance(result, RealTeleopDatasetSampleAdmission):
        raise TypeError(
            "runtime extension dataset_sample_admission() must return "
            "RealTeleopDatasetSampleAdmission."
        )
    return result


def runtime_extension_after_dataset_sample_attempt(
    extension: object | None,
    *,
    admission: RealTeleopDatasetSampleAdmission,
    accepted: bool,
    now: float,
) -> None:
    """Report the writer's authoritative result to an interested extension."""

    if extension is None:
        return
    hook = getattr(extension, "after_dataset_sample_attempt", None)
    if callable(hook):
        hook(admission=admission, accepted=bool(accepted), now=float(now))


def runtime_extension_after_dataset_sample_terminal(
    extension: object | None,
    *,
    event: SampleTerminalEvent,
    now: float,
) -> None:
    """Deliver one writer-terminal sample outcome to an extension.

    A queue admission is intentionally not sufficient for rule/progress state
    transitions.  This adapter reconstructs the original admission metadata
    carried by the recorder and reuses the historical callback so extensions
    written before terminal-event support remain source-compatible.
    """

    if extension is None:
        return
    if not isinstance(event, SampleTerminalEvent):
        raise TypeError("dataset sample terminal event has an invalid type.")
    runtime_extension_after_dataset_sample_attempt(
        extension,
        admission=RealTeleopDatasetSampleAdmission(
            allowed=True,
            metadata={
                **dict(event.metadata),
                "_terminal_kind": event.terminal_kind,
                "_terminal_reason": event.reason,
                **(
                    {}
                    if event.receipt is None
                    else {
                        "_terminal_episode_index": event.receipt.episode_index,
                        "_terminal_sample_index": event.receipt.sample_index,
                        "_terminal_episode_frame_index": event.receipt.episode_frame_index,
                        "_terminal_sample_timestamp": event.receipt.sample_timestamp,
                    }
                ),
            },
        ),
        accepted=bool(event.accepted),
        now=float(now),
    )


def runtime_extension_after_session_error(
    extension: object | None,
    *,
    error: BaseException,
    now: float,
) -> None:
    """Synchronize an extension when Session.run_step() raises."""

    if extension is None:
        return
    hook = getattr(extension, "after_session_error", None)
    if callable(hook):
        hook(error=error, now=float(now))


def runtime_extension_refresh_status(
    extension: object | None,
    *,
    force: bool = False,
) -> None:
    """Refresh an extension's observational UI projection after a lifecycle edge.

    Startup admission and Hold/recovery transitions can happen outside the
    normal ``after_control_input_poll`` callback.  A dedicated application
    (for example the rule-writing UI) therefore needs a small explicit seam to
    republish its current status immediately.  This helper never changes
    authority, safety, or dispatch state; extensions that do not expose the
    optional hook retain the historical no-op behavior.
    """

    if extension is None:
        return
    hook = getattr(extension, "refresh_status", None)
    if not callable(hook):
        return
    try:
        hook(force=bool(force))
    except TypeError:
        # Compatibility with a thin extension that provides the older
        # zero-argument hook.  Only retry for the explicitly requested force
        # path; a TypeError from the hook body must not be silently retried on
        # every steady-state frame.
        if not force:
            raise
        hook()


def runtime_extension_collection_startup_blockers(
    extension: object | None,
) -> dict[str, str]:
    """Read optional extension-owned blockers before opening a formal writer.

    This is an observational seam only.  It lets an exclusive runtime source
    expose the same readiness facts it uses for command admission without
    moving writer ownership or weakening the generic device gate.
    """

    if extension is None:
        return {}
    hook = getattr(extension, "collection_startup_blockers", None)
    if not callable(hook):
        return {}
    result = hook()
    if result is None:
        return {}
    if not isinstance(result, Mapping):
        raise TypeError(
            "runtime extension collection_startup_blockers() must return a mapping."
        )
    return {
        str(key): str(value)
        for key, value in result.items()
        if str(value).strip()
    }


def runtime_extension_service_startup_control_plane(
    extension: object | None,
    *,
    now: float,
) -> bool:
    """Service an extension's command-inert UI work before runtime Home.

    An early-bound application can remain in the startup wait loop while a
    Rig/backend fault is repaired.  The realtime control loop, and therefore
    its normal ``after_control_input_poll`` callback, has not started yet in
    that state.  This optional seam lets the application finish purely
    control-plane work such as offline rule-plan validation without consuming
    motion, gripper, Episode, or Hold intents.
    """

    if extension is None:
        return False
    hook = getattr(extension, "service_startup_control_plane", None)
    if not callable(hook):
        return False
    return bool(hook(now=float(now)))


def runtime_extension_discards_episode_on_stop(
    extension: object | None,
    *,
    reason: str,
) -> bool:
    """Ask an exclusive source whether this stop cancels its whole Episode.

    Ordinary real teleoperation keeps its historical finalize-on-stop behavior.
    A rule source may instead define one Episode as one indivisible execution
    attempt, in which case every non-completion stop must discard that attempt.
    """

    if extension is None:
        return False
    hook = getattr(extension, "discards_episode_on_stop", None)
    return bool(callable(hook) and hook(reason=str(reason)))


def runtime_extension_restarts_after_hold(
    extension: object | None,
    *,
    reason: str,
) -> bool:
    """Return whether Hold cancels the active attempt and requires fresh Home."""

    if extension is None:
        return False
    hook = getattr(extension, "restarts_after_hold", None)
    return bool(callable(hook) and hook(reason=str(reason)))


def runtime_extension_after_runtime_hold(
    extension: object | None,
    *,
    reason: str,
    hardware_hold_confirmed: bool,
) -> None:
    """Notify an extension after the authoritative Session Hold boundary."""

    if extension is None:
        return
    hook = getattr(extension, "after_runtime_hold", None)
    if callable(hook):
        hook(
            reason=str(reason),
            hardware_hold_confirmed=bool(hardware_hold_confirmed),
        )


def runtime_extension_after_runtime_home(
    extension: object | None,
    *,
    now: float,
) -> None:
    """Notify an extension only after runtime Home and arm admission succeed."""

    if extension is None:
        return
    hook = getattr(extension, "after_runtime_home", None)
    if callable(hook):
        hook(now=float(now))


def runtime_extension_bind_runtime_home_snapshot(
    extension: object | None,
    *,
    qpos_by_side: Mapping[str, Any] | None,
) -> None:
    """Pass the accepted atomic Home snapshot to an interested extension.

    This is a lifecycle handoff only.  It does not arm hardware or authorize
    dispatch; it gives an exclusive source the same Home epoch used by the
    canonical runtime so it can build its own command-inert first waypoint.
    Extensions that do not own a source-specific Home anchor are unchanged.
    """

    if extension is None:
        return
    hook = getattr(extension, "bind_runtime_home_snapshot", None)
    if callable(hook):
        hook(qpos_by_side=qpos_by_side)


def runtime_extension_allows_pre_episode_armed_warmup(
    extension: object | None,
) -> bool:
    """Admit an extension-owned ARMED start without weakening sample gates.

    This opt-in is consulted only by the Episode-start countdown.  Once an
    Episode is active, the separate active-bootstrap hook owns any bounded
    ARMED transition; sample admission remains independently strict.
    """

    if extension is None:
        return False
    hook = getattr(extension, "allows_pre_episode_armed_warmup", None)
    return bool(callable(hook) and hook())


def runtime_extension_allows_active_episode_bootstrap_without_dispatch(
    extension: object | None,
) -> bool:
    """Identify a command-inert active-Episode bootstrap owned by an extension.

    This hook does not authorize motion or sample admission.  It only tells the
    owner loop that an ARMED extension is still preparing its first arm command.
    All observation, modality, force, collision, TCP-envelope, Session,
    backend-receipt, and writer sample gates remain authoritative.
    """

    if extension is None:
        return False
    hook = getattr(
        extension,
        "allows_active_episode_bootstrap_without_dispatch",
        None,
    )
    return bool(callable(hook) and hook())


def runtime_extension_defers_progress_entry_arm_control(
    extension: object | None,
    *,
    target_progress_step_id: str,
    current_fixed_sides: tuple[str, ...],
    target_fixed_sides: tuple[str, ...],
    fixed_scope: str,
) -> bool:
    """Let an exclusive source defer one exact progress-entry Fix transition.

    Deferral changes only when the profile's Fix authority is established.  It
    does not authorize motion, bypass safety, or relax dataset sample gates.
    The extension remains responsible for establishing the declared Fix from
    a command that passed the normal Session filter/guard/dispatch path.
    """

    if extension is None:
        return False
    hook = getattr(extension, "defers_progress_entry_arm_control", None)
    return bool(
        callable(hook)
        and hook(
            target_progress_step_id=str(target_progress_step_id),
            current_fixed_sides=tuple(current_fixed_sides),
            target_fixed_sides=tuple(target_fixed_sides),
            fixed_scope=str(fixed_scope),
        )
    )


class RealTeleopRuntimeExtension(Protocol):
    """Application-leaf hooks around one existing RealTeleop runtime.

    ``attach_components`` must be side-effect free: it may validate and return
    a replacement immutable aggregate, but it must not start a thread, socket,
    device, or motion.  ``start`` runs only inside the normal runtime cleanup
    boundary.  ``close_before_session`` must establish any extension-owned
    lower HOLD before the Session and hardware resources close.
    """

    def attach_components(
        self,
        components: RealTeleopComponents,
    ) -> RealTeleopComponents: ...

    def start(self, *, components: RealTeleopComponents) -> None: ...

    def bind_runtime(
        self,
        *,
        components: RealTeleopComponents,
        clutch_input: Any,
        camera_sources: Mapping[str, Any],
        dataset_recorder: object | None,
        dataset_recorder_provider: Callable[[], object | None],
        episode_recorder: object,
        control_ui: object | None,
        recording_active: Callable[[], bool],
        pause_recording: Callable[..., bool],
        collection_transition_active: Callable[[], bool] | None = None,
    ) -> None: ...

    def after_control_input_poll(self, *, now: float) -> None: ...

    def after_session_step(self, *, step_result: object, now: float) -> None: ...

    def after_session_error(self, *, error: BaseException, now: float) -> None: ...

    # Optional observational hook.  The runtime calls it only at lifecycle
    # edges (or when a dedicated UI needs an immediate refresh); it is not an
    # authority or dispatch callback.
    def refresh_status(self, *, force: bool = False) -> None: ...

    # Optional observational hook consulted before a formal writer is opened.
    # It must not move hardware or mutate lifecycle state.
    def collection_startup_blockers(self) -> Mapping[str, str]: ...

    # Optional command-inert hook used only while startup admission is waiting.
    # It must not consume motion, gripper, Episode, or Hold intents.
    def service_startup_control_plane(self, *, now: float) -> bool: ...

    def discards_episode_on_stop(self, *, reason: str) -> bool: ...

    def restarts_after_hold(self, *, reason: str) -> bool: ...

    def after_runtime_hold(
        self,
        *,
        reason: str,
        hardware_hold_confirmed: bool,
    ) -> None: ...

    def after_runtime_home(self, *, now: float) -> None: ...

    # Optional lifecycle handoff.  The hook must only copy/validate the
    # accepted Home snapshot; it must not move hardware or dispatch commands.
    def bind_runtime_home_snapshot(
        self,
        *,
        qpos_by_side: Mapping[str, Any] | None,
    ) -> None: ...

    def allows_pre_episode_armed_warmup(self) -> bool: ...

    def allows_active_episode_bootstrap_without_dispatch(self) -> bool: ...

    def defers_progress_entry_arm_control(
        self,
        *,
        target_progress_step_id: str,
        current_fixed_sides: tuple[str, ...],
        target_fixed_sides: tuple[str, ...],
        fixed_scope: str,
    ) -> bool: ...

    def after_dataset_sample_attempt(
        self,
        *,
        admission: RealTeleopDatasetSampleAdmission,
        accepted: bool,
        now: float,
    ) -> None: ...

    def close_before_session(
        self,
        *,
        components: RealTeleopComponents,
        terminal_error: BaseException | None,
    ) -> None: ...


__all__ = [
    "RealTeleopDatasetSampleAdmission",
    "RealTeleopRuntimeExtension",
    "runtime_extension_dataset_sample_admission",
    "runtime_extension_after_dataset_sample_attempt",
    "runtime_extension_after_dataset_sample_terminal",
    "runtime_extension_after_session_error",
    "runtime_extension_after_runtime_hold",
    "runtime_extension_after_runtime_home",
    "runtime_extension_bind_runtime_home_snapshot",
    "runtime_extension_collection_startup_blockers",
    "runtime_extension_discards_episode_on_stop",
    "runtime_extension_refresh_status",
    "runtime_extension_service_startup_control_plane",
    "runtime_extension_restarts_after_hold",
    "runtime_extension_allows_active_episode_bootstrap_without_dispatch",
    "runtime_extension_defers_progress_entry_arm_control",
    "runtime_extension_allows_pre_episode_armed_warmup",
]
