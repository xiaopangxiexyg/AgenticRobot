"""Foreground operator driving for an already assembled B601/RM65 session.

This application leaf advances the injected target owner, not a second rig or
servo worker. Device construction and master gravity compensation stay outside
this loop. No module import or operator command enables/disables real motors.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping

from control_station.sources.rebot_b601_rm65_motion import (
    B601RM65MotionSession,
    B601RM65TargetExecutionGateError,
)
from hardware.rebot_b601_rs import B601HardwareDependencyError
from utils.terminal_commands import ImmediateTerminalCommands


class B601RM65TerminalCommands(ImmediateTerminalCommands):
    """Per-session nonblocking operator input.

    A real controlling terminal uses cbreak mode, so ``a``/``c``/``r``/``s``
    and ``q`` take effect immediately without requiring Enter or a shell
    command re-run.  A pipe or injected file descriptor retains the line
    protocol used by tests and by scripted diagnostics.  The reader owns the
    terminal mode only when it enabled cbreak mode itself.
    """


def _print_status(message: str) -> None:
    print(message, flush=True)


def run_b601_rm65_foreground(
    session: B601RM65MotionSession,
    *,
    advance_target: Callable[[], object],
    rate_hz: float,
    poll_command: Callable[[float], str | None] | None = None,
    auto_activate: bool = False,
    emit: Callable[[str], None] = _print_status,
    record: Callable[[Mapping[str, object]], None] | None = None,
) -> None:
    """Run in the calling foreground process until q/EOF/exception/Ctrl-C.

    a activates at the two measured current poses, c clutches, r releases and
    recenters, s deactivates, and q exits. ``auto_activate`` is an explicit
    receive-only simulation convenience: it performs the same fresh-anchor
    transaction once at startup, without enabling or commanding the master.
    On a real terminal these are immediate single-key transitions; no Enter or
    shell history re-run is needed. The target's existing runtime step belongs
    in advance_target.
    poll_command waits at most its supplied timeout; it must not run a blocking
    prompt or launch a background input thread.
    record receives events for an application-owned JSON/log sink. It must not
    issue commands. Failures propagate after withdrawing target following.
    """
    if not isinstance(session, B601RM65MotionSession):
        raise TypeError("session must be B601RM65MotionSession.")
    if not math.isfinite(rate_hz) or rate_hz <= 0:
        raise ValueError("rate_hz must be finite and positive.")
    if not isinstance(auto_activate, bool):
        raise TypeError("auto_activate must be bool.")
    if not all(callable(value) for value in (advance_target, emit)):
        raise TypeError("Target, terminal and status callbacks must be callable.")
    if poll_command is not None and not callable(poll_command):
        raise TypeError("poll_command must be callable or None.")
    if record is not None and not callable(record):
        raise TypeError("record must be callable or None.")
    clock = session.binding.runtime.clock
    period = 1.0 / rate_hz

    def event(name: str, **details: object) -> None:
        if record is not None:
            record({"event": name, "at": clock(), "state": session.state, **details})

    def instructions() -> None:
        emit(
            "[B601→RM65] Keys (no Enter): a=activate/retry c=clutch r=recenter s=stop q=quit ?=help"
        )
        emit("[B601→RM65] Start inactive; reposition/settle first, then a. c before recentering.")
        emit(
            "[B601→RM65] Feedback-capability or target-gate errors stay held; repair/rearm the source and press a again."
        )
        emit("[B601→RM65] Ctrl-C/EOF: stop RM65 following; master gravity/damping is unchanged.")

    def recoverable_feedback_error(command: str, error: B601HardwareDependencyError) -> None:
        """Keep the receive-only UI alive while the host feedback port recovers.

        The backend remains fail-closed: no old observation is reused and no
        target proposal is admitted.  This handler only converts a known
        dependency/feedback-capability failure into an operator-visible retry
        state.  Other exceptions still propagate to the application boundary.
        """

        message = str(error)
        emit(f"[B601→RM65] waiting: {type(error).__name__}: {message}")
        emit(
            "[B601→RM65] no RM65 following; fix the feedback runtime/stream, then press a to retry."
        )
        event(
            "feedback_capability_unavailable",
            command=command,
            error_type=type(error).__name__,
            error=message,
            retryable=True,
        )

    def recoverable_target_gate_error(
        command: str, error: B601RM65TargetExecutionGateError
    ) -> None:
        """Keep the receive-only UI open for a lower-owner gate transition.

        A target fault/transition is not permission to publish an old anchor.
        The snapshot is reported verbatim and the operator must repair or
        re-arm the lower owner before retrying ``a``/``r``.
        """

        snapshot = error.snapshot
        emit(
            "[B601→RM65] target gate held: "
            f"status={snapshot.status} reason={snapshot.reason!r} "
            f"gate_epoch={snapshot.gate_epoch}; no RM65 following."
        )
        emit(
            "[B601→RM65] repair/re-arm the target gate, then press a to retry; "
            "no stale anchor is reused."
        )
        event(
            "target_execution_gate_unavailable",
            command=command,
            gate_status=snapshot.status,
            gate_reason=snapshot.reason,
            gate_epoch=snapshot.gate_epoch,
            error=str(error),
            retryable=True,
        )

    def recoverable_activation_value_error(command: str, error: ValueError) -> None:
        """Keep the receive-only UI alive when a fresh activation is unavailable."""

        emit(f"[B601→RM65] activation held: {type(error).__name__}: {error}")
        emit(
            "[B601→RM65] no RM65 following; repair the current source, then "
            "press a to retry."
        )
        event(
            "activation_rejected",
            command=command,
            error_type=type(error).__name__,
            error=str(error),
            retryable=True,
        )

    def activate(command: str) -> bool:
        """Capture a fresh anchor and report retryable activation failures."""

        try:
            session.activate()
        except B601HardwareDependencyError as error:
            recoverable_feedback_error(command, error)
            return False
        except B601RM65TargetExecutionGateError as error:
            recoverable_target_gate_error(command, error)
            return False
        except ValueError as error:
            # Freshness/source/anchor rejection is a normal retry state for a
            # receive-only simulation. Never reuse the previous anchor.
            recoverable_activation_value_error(command, error)
            return False
        emit(
            f"[B601→RM65] active ({'auto' if command == 'auto_activate' else 'manual'})"
        )
        event("operator_command", command=command)
        return True

    owns_terminal_reader = poll_command is None
    try:
        if poll_command is None:
            poll_command = B601RM65TerminalCommands()
        # Never inherit a previously retained reference when launching the UI.
        session.deactivate()
        instructions()
        event("foreground_started")
        next_tick = float(clock()) + period
        if auto_activate:
            emit(
                "[B601→RM65] auto-activation requested; waiting for one fresh "
                "source snapshot (no key required)."
            )
            if activate("auto_activate"):
                next_tick = float(clock()) + period
        while True:
            command = poll_command(max(0.0, next_tick - float(clock())))
            if command is not None:
                command = command.strip().lower()
                if command == "q":
                    event("operator_quit")
                    return
                if command in {"", "?"}:
                    instructions()
                elif command in {"a", "c", "r", "s"}:
                    # Invalid state transitions are operator mistakes, not
                    # permission to re-open a fault or publish an old anchor.
                    allowed = {
                        "a": {"inactive", "held"},
                        "c": {"active", "clutched"},
                        "r": {"clutched"},
                        "s": {"inactive", "active", "held", "clutched"},
                    }
                    if session.state not in allowed[command]:
                        emit(f"[B601→RM65] {command} rejected in {session.state}")
                        event("operator_command_rejected", command=command)
                    else:
                        if command == "a":
                            if not activate(command):
                                next_tick = float(clock()) + period
                                continue
                        else:
                            try:
                                {
                                    "c": session.engage_clutch,
                                    "r": session.release_clutch,
                                    "s": session.deactivate,
                                }[command]()
                            except B601HardwareDependencyError as error:
                                # A missing host observation capability is
                                # recoverable for this receive-only application.
                                recoverable_feedback_error(command, error)
                                next_tick = float(clock()) + period
                                continue
                            except B601RM65TargetExecutionGateError as error:
                                # A lower-owner fault/transition is retryable, but
                                # never an implicit re-arm or permission to reuse
                                # a previous target anchor.
                                recoverable_target_gate_error(command, error)
                                next_tick = float(clock()) + period
                                continue
                        if command in {"a", "r"}:
                            # Activation/recenter consumes one complete master
                            # snapshot as the anchor. Do not immediately poll
                            # the same vendor cache frame again in this tick.
                            next_tick = float(clock()) + period
                        if command != "a":
                            emit(f"[B601→RM65] {session.state}")
                            event("operator_command", command=command)
                else:
                    emit("[B601→RM65] unknown command; ? for help")
                    event("operator_command_rejected", command=command)
            now = float(clock())
            if now < next_tick:
                continue
            proposal_receipt = None
            source_diagnostics: dict[str, object] = {}
            session_step_started = time.perf_counter()
            if session.state == "active":
                try:
                    proposal_receipt = session.step()
                except B601HardwareDependencyError as error:
                    # ``step`` has moved the session to held and cleared its
                    # anchor.  Keep advancing the lower target once below so
                    # the cancellation is observed without retaining a stale
                    # RM65 proposal.
                    recoverable_feedback_error("step", error)
                else:
                    source_diagnostics = session.last_step_diagnostics
                    event(
                        "proposal_admission",
                        accepted=proposal_receipt.accepted,
                        rejection_code=proposal_receipt.rejection_code,
                        rejection_message=proposal_receipt.message,
                        source_diagnostics=source_diagnostics,
                    )
                    if not proposal_receipt.accepted:
                        emit(
                            f"[B601→RM65] held: {proposal_receipt.rejection_code}; inspect before a"
                        )
            session_step_elapsed = max(
                0.0, time.perf_counter() - session_step_started
            )
            target_details = advance_target()
            if not isinstance(target_details, Mapping):
                target_details = {}
            source_diagnostics = (
                source_diagnostics or session.last_step_diagnostics
            )
            bridge_step = {
                "event": "bridge_step",
                "state": session.state,
                "source_admitted": source_diagnostics.get(
                    "source_admitted", False
                ),
                "source_reused": bool(
                    source_diagnostics.get("source_reused", False)
                ),
                "source_sequence": source_diagnostics.get("source_sequence"),
                "source_receive_timestamp_sec": source_diagnostics.get(
                    "source_receive_timestamp_sec"
                ),
                "source_observation_sample_id": source_diagnostics.get(
                    "source_observation_sample_id"
                ),
                "source_poll_started_monotonic_sec": source_diagnostics.get(
                    "source_poll_started_monotonic_sec"
                ),
                "source_poll_completed_monotonic_sec": source_diagnostics.get(
                    "source_poll_completed_monotonic_sec"
                ),
                "source_poll_elapsed_sec": source_diagnostics.get(
                    "source_poll_elapsed_sec"
                ),
                "feedback_poll_elapsed_sec": source_diagnostics.get(
                    "feedback_poll_elapsed_sec"
                ),
                "fk_elapsed_sec": source_diagnostics.get("fk_elapsed_sec"),
                "emission_elapsed_sec": source_diagnostics.get(
                    "emission_elapsed_sec"
                ),
                "source_adaptation_elapsed_sec": source_diagnostics.get(
                    "source_adaptation_elapsed_sec"
                ),
                "proposal_submission_elapsed_sec": source_diagnostics.get(
                    "proposal_submission_elapsed_sec"
                ),
                "session_step_elapsed_sec": (
                    source_diagnostics.get("session_step_elapsed_sec")
                    if source_diagnostics.get("session_step_elapsed_sec") is not None
                    else session_step_elapsed
                ),
                "proposal_accepted": (
                    None
                    if proposal_receipt is None
                    else bool(proposal_receipt.accepted)
                ),
                "proposal_rejection_code": (
                    None
                    if proposal_receipt is None
                    else proposal_receipt.rejection_code
                ),
                "target_dispatch_accepted": target_details.get(
                    "target_dispatch_accepted", False
                ),
                "target_dispatch_monotonic_sec": target_details.get(
                    "target_dispatch_monotonic_sec"
                ),
                "target_step_elapsed_sec": target_details.get(
                    "target_step_elapsed_sec"
                ),
                "target_resolution_invoked": target_details.get(
                    "target_resolution_invoked", False
                ),
                "target_resolution_elapsed_sec": target_details.get(
                    "target_resolution_elapsed_sec"
                ),
                "target_pose_transform_elapsed_sec": (
                    target_details.get("target_resolution", {})
                    if isinstance(target_details.get("target_resolution"), Mapping)
                    else {}
                ).get("pose_transform_elapsed_sec"),
                "target_pose_smoothing_elapsed_sec": (
                    target_details.get("target_resolution", {})
                    if isinstance(target_details.get("target_resolution"), Mapping)
                    else {}
                ).get("pose_smoothing_elapsed_sec"),
                "target_pose_safety_filter_elapsed_sec": (
                    target_details.get("target_resolution", {})
                    if isinstance(target_details.get("target_resolution"), Mapping)
                    else {}
                ).get("pose_safety_filter_elapsed_sec"),
                "target_pre_ik_elapsed_sec": (
                    target_details.get("target_resolution", {})
                    if isinstance(target_details.get("target_resolution"), Mapping)
                    else {}
                ).get("pre_ik_elapsed_sec"),
                "target_ik_elapsed_sec": target_details.get(
                    "target_ik_elapsed_sec"
                ),
                "target_post_ik_elapsed_sec": (
                    target_details.get("target_resolution", {})
                    if isinstance(target_details.get("target_resolution"), Mapping)
                    else {}
                ).get("post_ik_elapsed_sec"),
                "target_resolution_build_elapsed_sec": (
                    target_details.get("target_resolution", {})
                    if isinstance(target_details.get("target_resolution"), Mapping)
                    else {}
                ).get("resolution_build_elapsed_sec"),
                "target_command_resolve_elapsed_sec": (
                    target_details.get("target_resolution", {})
                    if isinstance(target_details.get("target_resolution"), Mapping)
                    else {}
                ).get("command_resolve_elapsed_sec"),
                "target_command_metadata_elapsed_sec": (
                    target_details.get("target_resolution", {})
                    if isinstance(target_details.get("target_resolution"), Mapping)
                    else {}
                ).get("command_metadata_elapsed_sec"),
                "target_resolution_cache_insert_elapsed_sec": (
                    target_details.get("target_resolution", {})
                    if isinstance(target_details.get("target_resolution"), Mapping)
                    else {}
                ).get("resolution_cache_insert_elapsed_sec"),
                "target_resolution_cache_release_elapsed_sec": (
                    target_details.get("target_resolution", {})
                    if isinstance(target_details.get("target_resolution"), Mapping)
                    else {}
                ).get("resolution_cache_release_elapsed_sec"),
                "target_resolution_cache_size_before": (
                    target_details.get("target_resolution", {})
                    if isinstance(target_details.get("target_resolution"), Mapping)
                    else {}
                ).get("resolution_cache_size_before"),
                "target_resolution_cache_size_after": (
                    target_details.get("target_resolution", {})
                    if isinstance(target_details.get("target_resolution"), Mapping)
                    else {}
                ).get("resolution_cache_size_after"),
                "target_resolution_cache_size_after_release": (
                    target_details.get("target_resolution", {})
                    if isinstance(target_details.get("target_resolution"), Mapping)
                    else {}
                ).get("resolution_cache_size_after_release"),
                "target_gc_count_before": (
                    target_details.get("target_resolution", {})
                    if isinstance(target_details.get("target_resolution"), Mapping)
                    else {}
                ).get("gc_count_before"),
                "target_gc_count_after": (
                    target_details.get("target_resolution", {})
                    if isinstance(target_details.get("target_resolution"), Mapping)
                    else {}
                ).get("gc_count_after"),
                "target_post_resolution_elapsed_sec": target_details.get(
                    "target_post_resolution_elapsed_sec"
                ),
                "target_resolution_status": (
                    target_details.get("target_resolution", {})
                    if isinstance(target_details.get("target_resolution"), Mapping)
                    else {}
                ).get("resolve_status"),
                "target_ik_status": (
                    target_details.get("target_resolution", {})
                    if isinstance(target_details.get("target_resolution"), Mapping)
                    else {}
                ).get("ik_status"),
                "target_ik_solver": (
                    target_details.get("target_resolution", {})
                    if isinstance(target_details.get("target_resolution"), Mapping)
                    else {}
                ).get("ik_solver"),
                "target_status": target_details.get("status"),
                "target_resolution_failure": target_details.get(
                    "resolution_failure"
                ),
                "target_runtime": target_details.get("target_runtime"),
            }
            source_receive = bridge_step["source_receive_timestamp_sec"]
            target_dispatch = bridge_step["target_dispatch_monotonic_sec"]
            if (
                isinstance(source_receive, (int, float))
                and not isinstance(source_receive, bool)
                and isinstance(target_dispatch, (int, float))
                and not isinstance(target_dispatch, bool)
                and math.isfinite(float(source_receive))
                and math.isfinite(float(target_dispatch))
                and float(target_dispatch) >= float(source_receive)
            ):
                bridge_step["source_to_target_dispatch_latency_sec"] = (
                    float(target_dispatch) - float(source_receive)
                )
            else:
                bridge_step["source_to_target_dispatch_latency_sec"] = None
            event(
                "bridge_step",
                **{
                    key: value
                    for key, value in bridge_step.items()
                    if key not in {"event", "state"}
                },
            )
            # Do not burst stale catch-up proposals after slow polling/IK.
            next_tick += period
            completed_at = float(clock())
            if next_tick <= completed_at:
                next_tick = completed_at + period
    finally:
        if owns_terminal_reader and isinstance(
            poll_command, B601RM65TerminalCommands
        ):
            poll_command.close()
        # This cleanup never touches the independently owned master backend.
        # Lower cancellation errors must escape, not masquerade as success.
        session.deactivate()
        event("foreground_stopped")
        emit("[B601→RM65] following stopped; master state unchanged")
