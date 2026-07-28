"""Default-off passive shadow consumer for the canonical replay GUI.

The consumer reads the immutable source-frame tap owned by the existing replay
executor.  It never reads a dataset cursor, resamples actions, acquires a
dispatch token, or calls a backend.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from robot_motion.action import (
    ActionComponentCatalog,
    ActionComponentSpec,
    project_robot_action_v1_to_component_action,
)
from robot_motion.replay.proposal_source import ReplaySourceFrameEmission
from teleop_runtime.control_trace import (
    ControlTraceRuntime,
    ControlTraceSpool,
    ControlTraceSpoolSink,
    ControlTraceSpoolWorker,
    ControlTraceWriterHealthLedger,
    RuntimeTraceSession,
)
from teleop_runtime.control_trace.projection import (
    make_runtime_trace_terminal,
    project_replay_binding_evidence,
    project_replay_proposal_adapter_result,
    project_replay_source_frame_emission,
    project_shadow_startup,
    project_shadow_step,
)
from teleop_runtime.deploy.arbiter_shadow import (
    PassivePolicyArbiterShadow,
    PolicyArbiterShadowConfig,
)

from .replay_proposal import (
    DatasetReplayProposalShadowAdapter,
    DatasetReplayProposalShadowConfig,
)


REPLAY_GUI_SHADOW_CLOCK_DOMAIN_V1 = "local_monotonic"
REPLAY_GUI_SHADOW_TRACE_PROFILE_V1 = "replay_shadow_dry_run_v1"
DEFAULT_REPLAY_GUI_SHADOW_QUEUE_CAPACITY = 4096
DEFAULT_REPLAY_GUI_SHADOW_TRACE_CAPACITY = 65536
DEFAULT_REPLAY_GUI_SHADOW_WAL_MAX_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class ReplayGuiShadowProfile:
    queue_capacity: int = DEFAULT_REPLAY_GUI_SHADOW_QUEUE_CAPACITY
    trace_capacity: int = DEFAULT_REPLAY_GUI_SHADOW_TRACE_CAPACITY
    trace_root: Path | None = None
    wal_max_bytes: int = DEFAULT_REPLAY_GUI_SHADOW_WAL_MAX_BYTES
    decision_hz: float = 60.0

    def __post_init__(self) -> None:
        if int(self.queue_capacity) <= 0:
            raise ValueError("Replay GUI shadow queue_capacity must be positive.")
        if int(self.trace_capacity) <= 0:
            raise ValueError("Replay GUI shadow trace_capacity must be positive.")
        if int(self.wal_max_bytes) <= 0:
            raise ValueError("Replay GUI shadow wal_max_bytes must be positive.")
        if not 0.0 < float(self.decision_hz) <= 60.0:
            raise ValueError("Replay GUI shadow decision_hz must be in (0, 60].")
        object.__setattr__(self, "queue_capacity", int(self.queue_capacity))
        object.__setattr__(self, "trace_capacity", int(self.trace_capacity))
        object.__setattr__(self, "wal_max_bytes", int(self.wal_max_bytes))
        object.__setattr__(self, "decision_hz", float(self.decision_hz))
        if self.trace_root is not None:
            object.__setattr__(self, "trace_root", Path(self.trace_root).expanduser())


@dataclass(frozen=True)
class ReplayGuiShadowBinding:
    source_path: str
    episode_id: str
    action_field: str
    source_hz: float
    dataset_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("source_path", "episode_id", "action_field"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty.")
        if float(self.source_hz) <= 0.0:
            raise ValueError("Replay GUI shadow source_hz must be positive.")
        object.__setattr__(self, "source_path", str(Path(self.source_path).resolve()))
        object.__setattr__(self, "episode_id", str(self.episode_id).strip())
        object.__setattr__(self, "action_field", str(self.action_field).strip())
        object.__setattr__(self, "source_hz", float(self.source_hz))
        object.__setattr__(self, "dataset_metadata", dict(self.dataset_metadata))


@dataclass(frozen=True)
class ReplayGuiShadowCloseReport:
    trace_session_id: str | None
    in_memory_complete: bool
    durable_complete: bool
    processing_error_count: int
    trace_rejection_count: int
    bound_generation: int
    wal_directory: Path | None


@dataclass
class _BoundRuntime:
    binding: ReplayGuiShadowBinding
    generation: int
    adapter: DatasetReplayProposalShadowAdapter
    shadow: PassivePolicyArbiterShadow
    trace: ControlTraceRuntime
    wal_directory: Path | None
    spool: ControlTraceSpool | None = None
    health: ControlTraceWriterHealthLedger | None = None
    worker: ControlTraceSpoolWorker | None = None
    evidence_hashes: set[str] | None = None
    trace_rejection_count: int = 0


class ReplayGuiShadowConsumer:
    """Single passive consumer for one canonical GUI replay source queue."""

    def __init__(
        self,
        *,
        profile: ReplayGuiShadowProfile,
        clock: Callable[[], float] = time.monotonic,
        uuid_factory: Callable[[], uuid.UUID] = uuid.uuid4,
    ) -> None:
        self.profile = profile
        self._clock = clock
        self._uuid_factory = uuid_factory
        self.emission_queue: queue.Queue[ReplaySourceFrameEmission | None] = queue.Queue(
            maxsize=profile.queue_capacity
        )
        self._lock = threading.RLock()
        self._processing_lock = threading.RLock()
        self._binding: ReplayGuiShadowBinding | None = None
        self._binding_generation = 0
        self._runtime: _BoundRuntime | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="replay-gui-passive-shadow",
            daemon=True,
        )
        self.processing_error_count = 0
        self.last_processing_error = ""
        self.close_report: ReplayGuiShadowCloseReport | None = None
        self._thread.start()

    def bind(self, binding: ReplayGuiShadowBinding) -> None:
        """Replace passive evidence context; never changes replay execution."""

        if not isinstance(binding, ReplayGuiShadowBinding):
            raise TypeError("binding must be ReplayGuiShadowBinding.")
        with self._processing_lock:
            old_runtime: _BoundRuntime | None
            with self._lock:
                if self._closed:
                    return
                old_runtime = self._runtime
                self._runtime = None
                self._binding = binding
                self._binding_generation += 1
            if old_runtime is not None:
                self._close_runtime(
                    old_runtime,
                    status="aborted",
                    reason="binding_replaced",
                )

    def clear_binding(self, *, reason: str) -> None:
        with self._processing_lock:
            old_runtime: _BoundRuntime | None
            with self._lock:
                old_runtime = self._runtime
                self._runtime = None
                self._binding = None
                self._binding_generation += 1
            if old_runtime is not None:
                self._close_runtime(old_runtime, status="aborted", reason=reason)

    def _create_runtime(
        self,
        binding: ReplayGuiShadowBinding,
        generation: int,
        emission: ReplaySourceFrameEmission,
    ) -> _BoundRuntime:
        components = tuple(
            ActionComponentSpec(
                component_id=component,
                selection_group_id=component,
                effector_name=component,
            )
            for component in emission.action.active_effectors
        )
        if not components:
            raise ValueError("Replay shadow emission has no active components.")
        runtime_uuid = self._uuid_factory().hex
        runtime_session_id = f"replay-gui-runtime-{runtime_uuid}"
        trace_session_id = f"replay-gui-trace-{runtime_uuid}"
        control_domain_id = "replay-gui-control-0"
        embodiment_instance_id = "replay-gui-target-0"
        producer_id = "replay:canonical-gui"
        catalog = ActionComponentCatalog(
            catalog_id=(
                f"replay-gui:{emission.action.action_space}:"
                f"{'-'.join(emission.action.active_effectors)}:v1"
            ),
            action_space=emission.action.action_space,
            components=components,
        )
        action_schema = project_robot_action_v1_to_component_action(
            emission.action
        ).action_schema
        shadow = PassivePolicyArbiterShadow(
            PolicyArbiterShadowConfig(
                runtime_session_id=runtime_session_id,
                control_domain_id=control_domain_id,
                embodiment_instance_id=embodiment_instance_id,
                producer_id=producer_id,
                producer_epoch=0,
                active_effectors=emission.action.active_effectors,
                action_space=emission.action.action_space,
                decision_hz=self.profile.decision_hz,
                component_catalog=catalog,
                action_schema=action_schema,
                clock_domain_id=REPLAY_GUI_SHADOW_CLOCK_DOMAIN_V1,
                boot_id=runtime_session_id,
            ),
            clock=self._clock,
        )
        adapter = DatasetReplayProposalShadowAdapter(
            DatasetReplayProposalShadowConfig(
                runtime_session_id=runtime_session_id,
                control_domain_id=control_domain_id,
                embodiment_instance_id=embodiment_instance_id,
                producer_id=producer_id,
                producer_epoch=0,
                action_schema=action_schema,
                component_schema_hash=catalog.semantic_content_hash,
                component_coverage=emission.action.active_effectors,
                source_path=binding.source_path,
                episode_id=binding.episode_id,
                action_field=binding.action_field,
                source_action_space=emission.action.action_space,
                source_hz=binding.source_hz,
                source_start_index=0,
                dataset_metadata=binding.dataset_metadata,
                # The current GUI has no immutable target pose-frame or
                # execution-capability contract.  Keep the relation explicitly
                # indeterminate; same-emission passive evaluation is still legal.
                target_frame_contract_hash=None,
                target_execution_capability_profile_hash=None,
            )
        )
        trace = ControlTraceRuntime(
            RuntimeTraceSession(
                trace_session_id=trace_session_id,
                runtime_session_id=runtime_session_id,
                control_domain_id=control_domain_id,
                profile_id=REPLAY_GUI_SHADOW_TRACE_PROFILE_V1,
                started_at=float(self._clock()),
                clock_domain_id=REPLAY_GUI_SHADOW_CLOCK_DOMAIN_V1,
            ),
            capacity=self.profile.trace_capacity,
        )
        runtime = _BoundRuntime(
            binding=binding,
            generation=generation,
            adapter=adapter,
            shadow=shadow,
            trace=trace,
            wal_directory=(
                None
                if self.profile.trace_root is None
                else self.profile.trace_root / trace_session_id
            ),
            evidence_hashes=set(),
        )
        if runtime.wal_directory is not None:
            runtime.wal_directory.mkdir(parents=True, exist_ok=False)
            runtime.spool = ControlTraceSpool(
                runtime.wal_directory / "control_trace.wal",
                trace_session_id=trace_session_id,
                max_bytes=self.profile.wal_max_bytes,
            )
            runtime.health = ControlTraceWriterHealthLedger(
                runtime.wal_directory / "writer_health.ledger",
                trace_session_id=trace_session_id,
            )
            sink = ControlTraceSpoolSink(
                sink_id=f"replay-gui-shadow-spool:{trace_session_id}",
                runtime=trace,
                spool=runtime.spool,
                health_ledger=runtime.health,
            )
            runtime.worker = ControlTraceSpoolWorker(sink)
            runtime.worker.start()
        self._issue_many(runtime, project_shadow_startup(shadow))
        return runtime

    def _issue_many(self, runtime: _BoundRuntime, facts) -> None:
        for fact in facts:
            if not runtime.trace.issue(fact).accepted:
                runtime.trace_rejection_count += 1
        if runtime.worker is not None:
            runtime.worker.notify()

    def _observe(self, emission: ReplaySourceFrameEmission) -> None:
        with self._processing_lock:
            self._observe_serialized(emission)

    def _observe_serialized(self, emission: ReplaySourceFrameEmission) -> None:
        with self._lock:
            binding = self._binding
            generation = self._binding_generation
            runtime = self._runtime
        if binding is None:
            return
        if str(Path(emission.source_path).resolve()) != binding.source_path:
            raise ValueError("Replay shadow emission does not match the bound source path.")
        if runtime is None or runtime.generation != generation:
            candidate = self._create_runtime(binding, generation, emission)
            with self._lock:
                if self._closed or self._binding_generation != generation:
                    keep = False
                elif self._runtime is None:
                    self._runtime = candidate
                    runtime = candidate
                    keep = True
                else:
                    runtime = self._runtime
                    keep = False
            if not keep:
                self._close_runtime(candidate, status="aborted", reason="binding_race")
        assert runtime is not None
        adapted = runtime.adapter.adapt_source_frame_emission(emission)
        evidence_hash = adapted.binding_assessment.semantic_content_hash
        assert runtime.evidence_hashes is not None
        if evidence_hash not in runtime.evidence_hashes:
            self._issue_many(
                runtime,
                project_replay_binding_evidence(
                    adapted,
                    runtime_session_id=runtime.shadow.config.runtime_session_id,
                    control_domain_id=runtime.shadow.config.control_domain_id,
                    embodiment_instance_id=runtime.shadow.config.embodiment_instance_id,
                    occurred_at=emission.observed_at,
                    clock_domain_id=emission.clock_domain_id,
                ),
            )
            runtime.evidence_hashes.add(evidence_hash)
        self._issue_many(
            runtime,
            (
                project_replay_source_frame_emission(
                    emission,
                    runtime_session_id=runtime.shadow.config.runtime_session_id,
                    control_domain_id=runtime.shadow.config.control_domain_id,
                    embodiment_instance_id=runtime.shadow.config.embodiment_instance_id,
                ),
                *project_replay_proposal_adapter_result(
                    adapted,
                    runtime_session_id=runtime.shadow.config.runtime_session_id,
                    control_domain_id=runtime.shadow.config.control_domain_id,
                    embodiment_instance_id=runtime.shadow.config.embodiment_instance_id,
                    occurred_at=emission.observed_at,
                    clock_domain_id=emission.clock_domain_id,
                ),
            ),
        )
        if adapted.proposal_draft is None:
            return
        step = runtime.shadow.evaluate_proposal_draft(
            adapted.proposal_draft,
            source_event_id=adapted.source_adapter_receipt.representation_trace_id,
        )
        self._issue_many(runtime, project_shadow_step(runtime.shadow, step))

    def _run(self) -> None:
        while True:
            emission = self.emission_queue.get()
            try:
                if emission is None:
                    return
                self._observe(emission)
            except BaseException as exc:  # passive evidence must never stop replay
                with self._lock:
                    self.processing_error_count += 1
                    self.last_processing_error = f"{type(exc).__name__}: {exc}"
            finally:
                self.emission_queue.task_done()

    def _close_runtime(self, runtime: _BoundRuntime, *, status: str, reason: str) -> None:
        self._issue_many(
            runtime,
            (
                make_runtime_trace_terminal(
                    runtime.shadow,
                    terminal_status=status,
                    reason=reason,
                    occurred_at=float(self._clock()),
                ),
            ),
        )
        if runtime.worker is not None:
            runtime.worker.notify()
            runtime.worker.stop(timeout_sec=5.0)
        if runtime.health is not None:
            runtime.health.close()
        if runtime.spool is not None:
            runtime.spool.close()

    def status(self) -> dict[str, Any]:
        with self._lock:
            runtime = self._runtime
            binding = self._binding
            return {
                "enabled": True,
                "authoritative": False,
                "dispatch_owner_token": False,
                "bound": binding is not None,
                "binding_generation": self._binding_generation,
                "source_path": "" if binding is None else binding.source_path,
                "trace_session_id": (
                    None if runtime is None else runtime.trace.session.trace_session_id
                ),
                "binding_verdict": (
                    None
                    if runtime is None
                    else runtime.adapter.binding_assessment.verdict
                ),
                "queue_size": self.emission_queue.qsize(),
                "queue_capacity": self.profile.queue_capacity,
                "processing_error_count": self.processing_error_count,
                "last_processing_error": self.last_processing_error,
                "trace_rejection_count": (
                    0 if runtime is None else runtime.trace_rejection_count
                ),
                "wal_directory": (
                    ""
                    if runtime is None or runtime.wal_directory is None
                    else str(runtime.wal_directory)
                ),
            }

    def trace_facts_for_test(self):
        """Return immutable admitted facts for hardware-free contract fixtures."""

        with self._lock:
            runtime = self._runtime
            if runtime is None:
                return ()
            return tuple(entry.fact for entry in runtime.trace.records())

    def trace_records_for_test(self):
        """Return logical envelopes for read-side materialization fixtures."""

        with self._lock:
            runtime = self._runtime
            if runtime is None:
                return ()
            return runtime.trace.records()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.emission_queue.put(None, timeout=5.0)
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise TimeoutError("Replay GUI shadow consumer did not stop before deadline.")
        with self._lock:
            runtime = self._runtime
            self._runtime = None
        if runtime is not None:
            with self._processing_lock:
                self._close_runtime(
                    runtime,
                    status="completed",
                    reason="gui_shutdown",
                )
            completeness = runtime.trace.completeness()
            self.close_report = ReplayGuiShadowCloseReport(
                trace_session_id=runtime.trace.session.trace_session_id,
                in_memory_complete=bool(
                    completeness.in_memory_complete
                    and self.processing_error_count == 0
                    and runtime.trace_rejection_count == 0
                ),
                durable_complete=bool(
                    completeness.durable_complete
                    and self.processing_error_count == 0
                    and runtime.trace_rejection_count == 0
                ),
                processing_error_count=self.processing_error_count,
                trace_rejection_count=runtime.trace_rejection_count,
                bound_generation=self._binding_generation,
                wal_directory=runtime.wal_directory,
            )
        else:
            self.close_report = ReplayGuiShadowCloseReport(
                trace_session_id=None,
                in_memory_complete=self.processing_error_count == 0,
                durable_complete=False,
                processing_error_count=self.processing_error_count,
                trace_rejection_count=0,
                bound_generation=self._binding_generation,
                wal_directory=None,
            )


__all__ = [
    "DEFAULT_REPLAY_GUI_SHADOW_QUEUE_CAPACITY",
    "DEFAULT_REPLAY_GUI_SHADOW_TRACE_CAPACITY",
    "DEFAULT_REPLAY_GUI_SHADOW_WAL_MAX_BYTES",
    "REPLAY_GUI_SHADOW_CLOCK_DOMAIN_V1",
    "REPLAY_GUI_SHADOW_TRACE_PROFILE_V1",
    "ReplayGuiShadowBinding",
    "ReplayGuiShadowCloseReport",
    "ReplayGuiShadowConsumer",
    "ReplayGuiShadowProfile",
]
