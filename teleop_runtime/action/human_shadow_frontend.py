"""Default-off canonical-GUI wiring for passive human proposal shadowing.

The frontend owns a runtime trace session, but never owns a dataset episode,
Rig, dispatcher, or dispatch token. It is built from an explicit embodiment
profile so out-of-session control branches cannot disappear silently.
"""

from __future__ import annotations

import argparse
import math
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

from dataset_contracts.control_trace_episode import (
    CONTROL_TRACE_SESSION_RELATIVE_ROOT_V1,
)
from robot_motion.action import (
    ActionComponentCatalog,
    ActionComponentSpec,
    ArmActionSpec,
    BIMANUAL_EE_POSE_V1,
    BIMANUAL_QPOS_V1,
    HandActionSpec,
    build_robot_action_v1_runtime_schema,
)

from teleop_runtime.control_trace import (
    ControlTraceRuntime,
    ControlTraceSpool,
    ControlTraceSpoolSink,
    ControlTraceSpoolWorker,
    ControlTraceWriterHealthLedger,
    EpisodeCaptureManifestStore,
    EpisodeTraceRecorderSink,
    PendingEpisodeTraceRelease,
    RuntimeTraceSession,
)
from teleop_runtime.deploy.arbiter_shadow import (
    PassivePolicyArbiterShadow,
    PolicyArbiterShadowConfig,
)

from .adapters import TeleopCommandActionAdapterConfig
from .human_proposal import HumanProposalAdapter, HumanProposalAdapterConfig
from .human_proposal import HumanCommandEmission, HumanCommandFilterObservation
from .human_shadow import HumanProposalShadowRuntime
from teleop_runtime.real_teleop.temporary_dual_arm_safety import (
    TemporaryDualArmSafetyEvaluation,
)


HUMAN_ACTION_ARBITER_MODES = ("off", "shadow")
HUMAN_SHADOW_TRACE_PROFILE_V1 = "human_shadow_dry_run_v1"
HUMAN_FORMAL_SHADOW_TRACE_PROFILE_V1 = "formal_real_human_shadow_v1"
HUMAN_SHADOW_CLOCK_DOMAIN_V1 = "local_monotonic"
DEFAULT_HUMAN_SHADOW_TRACE_CAPACITY = 262_144
DEFAULT_HUMAN_SHADOW_WAL_MAX_BYTES = 128 * 1024 * 1024
DEFAULT_HUMAN_SHADOW_OBSERVER_QUEUE_CAPACITY = 4096


def _positive_rate(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0 or parsed > 60.0:
        raise argparse.ArgumentTypeError("value must be in (0, 60]")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def add_human_shadow_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("passive human ActionArbiter shadow")
    group.add_argument(
        "--human-action-arbiter-mode",
        choices=HUMAN_ACTION_ARBITER_MODES,
        default="off",
        help=(
            "Default-off passive human proposal shadow. It records would-select facts "
            "but never owns or changes the existing GUI dispatch path."
        ),
    )
    group.add_argument(
        "--human-action-arbiter-hz",
        type=_positive_rate,
        default=60.0,
        help="Human shadow decision ceiling in (0, 60]; does not limit other loops.",
    )
    group.add_argument(
        "--human-action-shadow-control-domain-id",
        default="",
        help="Optional explicit station control-domain identity for the shadow trace.",
    )
    group.add_argument(
        "--human-action-shadow-embodiment-instance-id",
        default="",
        help="Optional explicit embodiment instance identity for the shadow trace.",
    )
    group.add_argument(
        "--human-action-shadow-trace-capacity",
        type=_positive_int,
        default=DEFAULT_HUMAN_SHADOW_TRACE_CAPACITY,
        help=(
            "Bounded in-memory ControlTrace fact capacity. Exhaustion marks debug "
            "evidence incomplete and never blocks control."
        ),
    )
    group.add_argument(
        "--human-action-shadow-observer-queue-size",
        type=_positive_int,
        default=DEFAULT_HUMAN_SHADOW_OBSERVER_QUEUE_CAPACITY,
        help=(
            "Bounded non-blocking GUI observer queue. Overflow marks the runtime "
            "trace incomplete and never blocks dispatch."
        ),
    )
    group.add_argument(
        "--human-action-shadow-trace-root",
        type=Path,
        default=None,
        help=(
            "Optional scoped debug WAL root. This remains a RuntimeTraceSession and "
            "is not a formal dataset episode or capture seal."
        ),
    )
    group.add_argument(
        "--human-action-shadow-wal-max-mb",
        type=_positive_int,
        default=DEFAULT_HUMAN_SHADOW_WAL_MAX_BYTES // (1024 * 1024),
        help="Preallocated scoped debug WAL capacity in MiB.",
    )


def validate_human_shadow_arguments(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    enabled = str(getattr(args, "human_action_arbiter_mode", "off")) == "shadow"
    trace_root = getattr(args, "human_action_shadow_trace_root", None)
    if not enabled and trace_root is not None:
        parser.error(
            "--human-action-shadow-trace-root requires "
            "--human-action-arbiter-mode shadow."
        )
    if trace_root is not None and bool(getattr(args, "no_debug_disk_write", False)):
        parser.error(
            "--human-action-shadow-trace-root conflicts with --no-debug-disk-write."
        )
    if enabled:
        active_sides = tuple(getattr(args, "active_sides", ("left", "right")))
        base_backend = str(getattr(args, "hand_backend", "inspire") or "inspire")
        out_of_session = tuple(
            f"{side}_hand"
            for side in active_sides
            if str(
                getattr(args, f"{side}_hand_backend", None) or base_backend
            )
            == "dmtac-gripper"
        )
        if out_of_session:
            parser.error(
                "human ActionArbiter shadow does not yet cover the canonical GUI's "
                "out-of-session DMTAC dispatch branch: "
                + ", ".join(out_of_session)
            )


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


@dataclass(frozen=True)
class RealTeleopHumanShadowProfile:
    mode: str
    pose_source_id: str
    active_sides: tuple[str, ...]
    arm_reference_frame_by_side: Mapping[str, str]
    hand_backend_by_side: Mapping[str, str] = field(default_factory=dict)
    session_hand_sides: tuple[str, ...] = ()
    out_of_session_effectors: tuple[str, ...] = ()
    gripper_slot_by_side: Mapping[str, int] = field(default_factory=dict)
    action_space: str = BIMANUAL_EE_POSE_V1
    arm_joint_policy: str = "source_pose_when_available"
    decision_hz: float = 60.0
    trace_capacity: int = DEFAULT_HUMAN_SHADOW_TRACE_CAPACITY
    observer_queue_capacity: int = DEFAULT_HUMAN_SHADOW_OBSERVER_QUEUE_CAPACITY
    trace_root: Path | None = None
    wal_max_bytes: int = DEFAULT_HUMAN_SHADOW_WAL_MAX_BYTES
    control_domain_id: str = ""
    embodiment_instance_id: str = ""
    trace_profile_id: str = HUMAN_SHADOW_TRACE_PROFILE_V1

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        if mode not in HUMAN_ACTION_ARBITER_MODES:
            raise ValueError(f"Unsupported human ActionArbiter mode {mode!r}.")
        sides = tuple(
            dict.fromkeys(str(side).strip().lower() for side in self.active_sides)
        )
        if not sides or any(side not in {"left", "right"} for side in sides):
            raise ValueError("active_sides must contain left and/or right.")
        frames = {
            str(side): _required(frame, name=f"{side} arm reference frame")
            for side, frame in dict(self.arm_reference_frame_by_side).items()
        }
        if set(frames) != set(sides):
            raise ValueError("Every active arm side requires one explicit reference frame.")
        session_hands = tuple(
            dict.fromkeys(
                str(side).strip().lower() for side in self.session_hand_sides
            )
        )
        if any(side not in sides for side in session_hands):
            raise ValueError("session_hand_sides must be a subset of active_sides.")
        rate = float(self.decision_hz)
        if not math.isfinite(rate) or rate <= 0.0 or rate > 60.0:
            raise ValueError("decision_hz must be in (0, 60].")
        if (
            int(self.trace_capacity) <= 0
            or int(self.observer_queue_capacity) <= 0
            or int(self.wal_max_bytes) < 4096
        ):
            raise ValueError("Observer, trace, and WAL capacities must be positive.")
        if self.action_space not in {BIMANUAL_EE_POSE_V1, BIMANUAL_QPOS_V1}:
            raise ValueError("GUI human shadow supports absolute pose or absolute qpos.")
        if self.arm_joint_policy not in {"qpos", "source_pose_when_available"}:
            raise ValueError("Unsupported human shadow arm_joint_policy.")
        trace_profile_id = str(self.trace_profile_id).strip()
        if trace_profile_id not in {
            HUMAN_SHADOW_TRACE_PROFILE_V1,
            HUMAN_FORMAL_SHADOW_TRACE_PROFILE_V1,
        }:
            raise ValueError("Unsupported human shadow trace_profile_id.")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(
            self, "pose_source_id", _required(self.pose_source_id, name="pose_source_id")
        )
        object.__setattr__(self, "active_sides", sides)
        object.__setattr__(self, "arm_reference_frame_by_side", frames)
        object.__setattr__(
            self,
            "hand_backend_by_side",
            {str(k): str(v) for k, v in dict(self.hand_backend_by_side).items()},
        )
        object.__setattr__(self, "session_hand_sides", session_hands)
        object.__setattr__(
            self,
            "out_of_session_effectors",
            tuple(str(value) for value in self.out_of_session_effectors),
        )
        object.__setattr__(
            self,
            "gripper_slot_by_side",
            {str(k): int(v) for k, v in dict(self.gripper_slot_by_side).items()},
        )
        object.__setattr__(self, "decision_hz", rate)
        object.__setattr__(self, "trace_capacity", int(self.trace_capacity))
        object.__setattr__(
            self, "observer_queue_capacity", int(self.observer_queue_capacity)
        )
        object.__setattr__(self, "wal_max_bytes", int(self.wal_max_bytes))
        object.__setattr__(self, "trace_profile_id", trace_profile_id)
        if self.trace_root is not None:
            object.__setattr__(self, "trace_root", Path(self.trace_root))


@dataclass(frozen=True)
class HumanShadowFrontendCloseReport:
    trace_session_id: str
    in_memory_complete: bool
    durable_complete: bool
    trace_rejection_count: int
    observer_drop_count: int
    processing_error_count: int
    wal_directory: Path | None


@dataclass(frozen=True)
class _ObserverJob:
    kind: str
    payload: object | None


class HumanShadowFrontend:
    """Lifecycle holder for one non-authoritative canonical-GUI shadow."""

    def __init__(
        self,
        *,
        profile: RealTeleopHumanShadowProfile,
        runtime: HumanProposalShadowRuntime,
        trace_runtime: ControlTraceRuntime,
        wal_directory: Path | None,
    ) -> None:
        self.profile = profile
        self.runtime = runtime
        self.trace_runtime = trace_runtime
        self.wal_directory = wal_directory
        self._spool: ControlTraceSpool | None = None
        self._health: ControlTraceWriterHealthLedger | None = None
        self._worker: ControlTraceSpoolWorker | None = None
        self._observer_queue: queue.Queue[_ObserverJob] = queue.Queue(
            maxsize=profile.observer_queue_capacity
        )
        self._observer_thread: threading.Thread | None = None
        self._started = False
        self._closed = False
        self.observer_drop_count = 0
        self.processing_error_count = 0
        self.last_processing_error: BaseException | None = None
        self.close_report: HumanShadowFrontendCloseReport | None = None
        self._episode_trace_sink: EpisodeTraceRecorderSink | None = None

    @property
    def runtime_session_id(self) -> str:
        return self.runtime.shadow.config.runtime_session_id

    @property
    def control_domain_id(self) -> str:
        return self.runtime.shadow.config.control_domain_id

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("Human shadow frontend is closed.")
        if self._started:
            return
        if self.wal_directory is not None:
            self.wal_directory.mkdir(parents=True, exist_ok=False)
            trace_id = self.trace_runtime.session.trace_session_id
            self._spool = ControlTraceSpool(
                self.wal_directory / "control_trace.wal",
                trace_session_id=trace_id,
                max_bytes=self.profile.wal_max_bytes,
            )
            self._health = ControlTraceWriterHealthLedger(
                self.wal_directory / "writer_health.ledger",
                trace_session_id=trace_id,
            )
            sink = ControlTraceSpoolSink(
                sink_id=f"human-shadow-spool:{trace_id}",
                runtime=self.trace_runtime,
                spool=self._spool,
                health_ledger=self._health,
            )
            self._worker = ControlTraceSpoolWorker(sink)
            self._worker.start()
        self.runtime.start()
        self._observer_thread = threading.Thread(
            target=self._run_observer,
            name="human-action-shadow-observer",
            daemon=True,
        )
        self._observer_thread.start()
        self._started = True

    def build_formal_episode_trace_sink(
        self,
        *,
        dataset_root: str | Path,
    ) -> EpisodeTraceRecorderSink:
        """Build the single subordinate formal sink; does not start an episode."""

        if self.profile.trace_profile_id != HUMAN_FORMAL_SHADOW_TRACE_PROFILE_V1:
            raise RuntimeError("Human shadow frontend is not configured for formal capture.")
        if self.wal_directory is None:
            raise RuntimeError("Formal human shadow capture requires a dataset-local WAL.")
        root = Path(dataset_root)
        expected_wal_directory = (
            root
            / CONTROL_TRACE_SESSION_RELATIVE_ROOT_V1
            / self.trace_runtime.session.trace_session_id
        )
        if self.wal_directory.resolve() != expected_wal_directory.resolve():
            raise ValueError(
                "Formal human shadow WAL must use the canonical dataset session path: "
                f"{expected_wal_directory}"
            )
        if self._episode_trace_sink is not None:
            return self._episode_trace_sink

        def flush_to_durable():
            worker = self._worker
            if worker is None:
                raise RuntimeError("Human shadow spool worker is not started.")
            return worker.flush_once()

        def session_dependencies(runtime: ControlTraceRuntime):
            return tuple(
                entry.journal_sequence
                for entry in runtime.journal_entries()
                if entry.episode_membership is None
                and entry.fact.event_type
                in {"producer_epoch_activation", "authority_event", "execution_gate_event"}
            )

        self._episode_trace_sink = EpisodeTraceRecorderSink(
            self.trace_runtime,
            lifecycle_owner_id="canonical_gui_real_teleop_dataset_recorder",
            flush_to_durable=flush_to_durable,
            dependency_sequences=session_dependencies,
            control_clock=self.runtime.shadow.now,
            manifest_store=EpisodeCaptureManifestStore(root),
            release_scheduler=self._schedule_episode_release,
        )
        return self._episode_trace_sink

    def _submit(self, job: _ObserverJob) -> bool:
        if self._closed:
            self.observer_drop_count += 1
            return False
        if not self._started:
            self.start()
        try:
            self._observer_queue.put_nowait(job)
        except queue.Full:
            self.observer_drop_count += 1
            return False
        return True

    def observe_emission(self, emission) -> bool:
        return self._submit(_ObserverJob("emission", emission))

    def observe_legacy_execution(self, observation) -> bool:
        return self._submit(_ObserverJob("legacy_execution", observation))

    def observe_filtered_command(
        self,
        emission: HumanCommandEmission,
        command,
        observed_at: float,
        limiter_evaluation: TemporaryDualArmSafetyEvaluation | None,
    ) -> bool:
        if limiter_evaluation is None:
            self.processing_error_count += 1
            self.last_processing_error = RuntimeError(
                "Human filtered-command observation lacks limiter evidence."
            )
            return False
        observation = HumanCommandFilterObservation.capture(
            emission=emission,
            observed_at=observed_at,
            command=command,
        )
        return self._submit(
            _ObserverJob("filtered_command", (observation, limiter_evaluation))
        )

    def _run_observer(self) -> None:
        while True:
            job = self._observer_queue.get()
            try:
                if job.kind == "stop":
                    return
                if job.kind == "emission":
                    self.runtime.observe_emission(job.payload)
                elif job.kind == "filtered_command":
                    observation, limiter_evaluation = job.payload
                    self.runtime.observe_filtered_command(
                        observation,
                        limiter_evaluation,
                    )
                elif job.kind == "legacy_execution":
                    self.runtime.observe_legacy_execution(job.payload)
                elif job.kind == "episode_release_barrier":
                    callback, pending = job.payload
                    try:
                        pending.set_result(callback())
                    except BaseException as exc:
                        pending.set_error(exc)
                else:
                    raise ValueError(f"Unsupported human shadow observer job {job.kind!r}.")
                if self._worker is not None:
                    self._worker.notify()
            except BaseException as exc:  # observe-only errors never stop dispatch
                self.processing_error_count += 1
                self.last_processing_error = exc
            finally:
                self._observer_queue.task_done()

    def _stop_observer(self, *, timeout_sec: float) -> None:
        thread = self._observer_thread
        if thread is None:
            return
        self._observer_queue.put(_ObserverJob("stop", None), timeout=timeout_sec)
        thread.join(timeout=timeout_sec)
        if thread.is_alive():
            raise TimeoutError("Human shadow observer did not stop before deadline.")

    def observe_emission_sync_for_test(self, emission):
        """Direct fixture seam; production TeleopSession uses the queued callback."""

        self.start()
        observed = self.runtime.observe_emission(emission)
        if self._worker is not None:
            self._worker.notify()
        return observed

    def observe_legacy_execution_sync_for_test(self, observation):
        """Direct fixture seam paired with ``observe_emission_sync_for_test``."""

        self.start()
        observed = self.runtime.observe_legacy_execution(observation)
        if self._worker is not None:
            self._worker.notify()
        return observed

    def observe_filtered_command_sync_for_test(
        self,
        emission: HumanCommandEmission,
        command,
        observed_at: float,
        limiter_evaluation: TemporaryDualArmSafetyEvaluation,
    ):
        """Direct fixture seam for exact legacy-filter evidence."""

        self.start()
        observation = HumanCommandFilterObservation.capture(
            emission=emission,
            observed_at=observed_at,
            command=command,
        )
        return self.runtime.observe_filtered_command(
            observation,
            limiter_evaluation,
        )

    def _schedule_episode_release(
        self,
        callback: Callable[[], object],
    ) -> PendingEpisodeTraceRelease:
        pending = PendingEpisodeTraceRelease()
        if not self._submit(
            _ObserverJob("episode_release_barrier", (callback, pending))
        ):
            pending.set_error(RuntimeError("Human shadow observer queue rejected episode barrier."))
        return pending

    def close(self, *, reason: str = "canonical_gui_shutdown") -> None:
        if self._closed:
            return
        failures: list[BaseException] = []
        try:
            self._stop_observer(timeout_sec=5.0)
            self.runtime.close(reason=reason)
            if self._worker is not None:
                self._worker.notify()
                self._worker.stop(timeout_sec=5.0)
        except BaseException as exc:  # preserve cleanup of local durable handles
            failures.append(exc)
        for resource in (self._health, self._spool):
            if resource is None:
                continue
            try:
                resource.close()
            except BaseException as exc:
                failures.append(exc)
        report = self.trace_runtime.completeness()
        observer_complete = bool(
            self.observer_drop_count == 0
            and self.processing_error_count == 0
            and self.runtime.trace_rejection_count == 0
        )
        self.close_report = HumanShadowFrontendCloseReport(
            trace_session_id=self.trace_runtime.session.trace_session_id,
            in_memory_complete=bool(report.in_memory_complete and observer_complete),
            durable_complete=bool(report.durable_complete and observer_complete),
            trace_rejection_count=self.runtime.trace_rejection_count,
            observer_drop_count=self.observer_drop_count,
            processing_error_count=self.processing_error_count,
            wal_directory=self.wal_directory,
        )
        self._closed = True
        if failures:
            raise RuntimeError(
                "Human shadow frontend cleanup failed: "
                + "; ".join(type(exc).__name__ for exc in failures)
            ) from failures[0]


def _hand_specs(profile: RealTeleopHumanShadowProfile) -> dict[str, HandActionSpec]:
    result: dict[str, HandActionSpec] = {}
    for side in profile.session_hand_sides:
        backend = str(profile.hand_backend_by_side.get(side, "")).strip()
        if backend == "inspire":
            result[side] = HandActionSpec.inspire_tac(side)
        elif backend == "dmtac-gripper":
            result[side] = HandActionSpec.dmtac_gripper(
                side,
                gripper_slot=int(profile.gripper_slot_by_side.get(side, 0)),
            )
        else:
            raise ValueError(
                f"No explicit HandActionSpec mapping for backend {backend!r} on {side}."
            )
    return result


def build_real_teleop_human_shadow_frontend(
    profile: RealTeleopHumanShadowProfile,
    *,
    clock: Callable[[], float] = time.monotonic,
    uuid_factory: Callable[[], uuid.UUID] = uuid.uuid4,
) -> HumanShadowFrontend | None:
    if profile.mode == "off":
        return None
    if profile.out_of_session_effectors:
        raise ValueError(
            "Human shadow cannot start while canonical GUI controls effectors outside "
            "TeleopSession: " + ", ".join(profile.out_of_session_effectors)
        )

    runtime_uuid = uuid_factory().hex
    runtime_session_id = f"human-gui-runtime-{runtime_uuid}"
    trace_session_id = f"human-gui-trace-{runtime_uuid}"
    side_token = "-".join(profile.active_sides)
    hand_token = "-".join(
        f"{side}-{profile.hand_backend_by_side.get(side, 'none')}"
        for side in profile.session_hand_sides
    ) or "arms-only"
    control_domain_id = str(profile.control_domain_id).strip() or (
        f"real-control:{profile.pose_source_id}:{side_token}"
    )
    embodiment_instance_id = str(profile.embodiment_instance_id).strip() or (
        f"rm65:{side_token}:{hand_token}"
    )
    producer_id = "human:vive"
    components = tuple(
        ActionComponentSpec(
            component_id=f"{side}_arm",
            selection_group_id=f"{side}_arm",
            effector_name=f"{side}_arm",
        )
        for side in profile.active_sides
    ) + tuple(
        ActionComponentSpec(
            component_id=f"{side}_hand",
            selection_group_id=f"{side}_hand",
            effector_name=f"{side}_hand",
        )
        for side in profile.session_hand_sides
    )
    catalog = ActionComponentCatalog(
        catalog_id=f"human-gui:{profile.pose_source_id}:{side_token}:{hand_token}:v1",
        action_space=profile.action_space,
        components=components,
    )
    action_adapter = TeleopCommandActionAdapterConfig(
        arm_joint_policy=profile.arm_joint_policy,
        arm_specs={
            side: ArmActionSpec.tool_tcp(
                side,
                reference_frame=profile.arm_reference_frame_by_side[side],
            )
            for side in profile.active_sides
        },
        hand_specs=_hand_specs(profile),
        include_command_metadata=False,
        source="human:vive:canonical_gui",
    )
    action_schema = build_robot_action_v1_runtime_schema(
        component_catalog=catalog,
        arm_action_specs_by_side=dict(action_adapter.arm_specs or {}),
        hand_action_specs_by_side=dict(action_adapter.hand_specs or {}),
    )
    adapter = HumanProposalAdapter(
        HumanProposalAdapterConfig(
            runtime_session_id=runtime_session_id,
            control_domain_id=control_domain_id,
            embodiment_instance_id=embodiment_instance_id,
            producer_id=producer_id,
            producer_epoch=0,
            produced_clock_domain_id=HUMAN_SHADOW_CLOCK_DOMAIN_V1,
            effective_clock_domain_id=HUMAN_SHADOW_CLOCK_DOMAIN_V1,
            component_catalog=catalog,
            action_schema=action_schema,
            action_adapter=action_adapter,
        )
    )
    shadow = PassivePolicyArbiterShadow(
        PolicyArbiterShadowConfig(
            runtime_session_id=runtime_session_id,
            control_domain_id=control_domain_id,
            embodiment_instance_id=embodiment_instance_id,
            producer_id=producer_id,
            producer_epoch=0,
            active_effectors=tuple(item.effector_name for item in components),
            action_space=profile.action_space,
            decision_hz=profile.decision_hz,
            component_catalog=catalog,
            action_schema=action_schema,
        ),
        clock=clock,
    )
    started_at = float(clock())
    trace_runtime = ControlTraceRuntime(
        RuntimeTraceSession(
            trace_session_id=trace_session_id,
            runtime_session_id=runtime_session_id,
            control_domain_id=control_domain_id,
            profile_id=profile.trace_profile_id,
            started_at=started_at,
            clock_domain_id=HUMAN_SHADOW_CLOCK_DOMAIN_V1,
        ),
        capacity=profile.trace_capacity,
    )
    runtime = HumanProposalShadowRuntime(
        adapter=adapter,
        shadow=shadow,
        trace_runtime=trace_runtime,
    )
    wal_directory = (
        None
        if profile.trace_root is None
        else Path(profile.trace_root) / trace_session_id
    )
    return HumanShadowFrontend(
        profile=profile,
        runtime=runtime,
        trace_runtime=trace_runtime,
        wal_directory=wal_directory,
    )


__all__ = [
    "DEFAULT_HUMAN_SHADOW_TRACE_CAPACITY",
    "DEFAULT_HUMAN_SHADOW_WAL_MAX_BYTES",
    "DEFAULT_HUMAN_SHADOW_OBSERVER_QUEUE_CAPACITY",
    "HUMAN_ACTION_ARBITER_MODES",
    "HUMAN_SHADOW_CLOCK_DOMAIN_V1",
    "HUMAN_FORMAL_SHADOW_TRACE_PROFILE_V1",
    "HUMAN_SHADOW_TRACE_PROFILE_V1",
    "HumanShadowFrontend",
    "HumanShadowFrontendCloseReport",
    "RealTeleopHumanShadowProfile",
    "add_human_shadow_arguments",
    "build_real_teleop_human_shadow_frontend",
    "validate_human_shadow_arguments",
]
