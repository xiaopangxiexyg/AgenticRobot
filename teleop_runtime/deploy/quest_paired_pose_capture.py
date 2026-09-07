"""Application-leaf binding for passive Quest/RM65 paired-pose capture.

This module joins the unique Quest controller-pose runtime, the existing
policy-deploy robot pose journal, and the lower-owned paired-pose reducer.  It
does not refresh a robot, issue commands, choose a controller or robot side,
persist evidence, fit or accept calibration, install frames, or render a
trajectory.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from dataset_contracts.canonical_encoding import canonical_content_hash
from hardware.quest_dev.features.controller_pose.runtime import (
    ControllerPoseAcceptedEvent,
    ControllerPoseEventCut,
    ControllerPoseInvalidationEvent,
    ControllerPoseObserverCloseCut,
    ControllerPoseSubscription,
    ControllerPoseTelemetryRuntime,
)
from hardware.quest_dev.features.visualization.paired_pose_capture import (
    CaptureTerminalError,
    ManualStaticPairedPoseCapture,
    PairedPoseCaptureBinding,
    PairedPoseCaptureEvidence,
    QuestControllerCaptureBinding,
    StaticWindowCaptureProfile,
    StaticWindowReceipt,
    StaticWindowRejectedError,
)
from hardware.quest_dev.features.visualization.protocol import (
    calibration_preview_message,
    capture_status_message,
)
from hardware.quest_dev.features.visualization.transforms import matrix_to_column_major
from hardware.quest_dev.server import (
    QuestDevHost,
    QuestDevLaunchInfo,
    build_quest_dev_host,
    load_feature_runtime,
)
from robot_motion.runtime.identity import ClockDomainDescriptor
from robot_motion.runtime.pose_observation import (
    PoseCoordinateBinding,
    PoseMeasurementAcceptedEvent,
    PoseMeasurementInvalidationEvent,
    PoseMeasurementJournalCut,
    PoseMeasurementJournalEvent,
)

from .arm_state_reader import ArmStateReader
from .rm65_pose_observation import RM65PoseObservationAssembly

CONTROLLER_POSE_FEATURE_NAME = "controller_pose"
QUEST_CONTROLLER_CANDIDATE_SCHEMA = "dexteleop.quest_controller_capture_candidate.v1"
QUEST_CONTROLLER_CANDIDATE_HASH_DOMAIN = (
    "dexteleop.quest_controller_capture_candidate.v1"
)
DEFAULT_STATIC_WINDOW_DURATION_S = 3.0
DEFAULT_CURRENT_TRACKING_MAX_AGE_S = 0.5

QuestPoseEvent = ControllerPoseAcceptedEvent | ControllerPoseInvalidationEvent


class QuestPairedPoseCaptureTerminalError(RuntimeError):
    """The application can no longer produce paired-pose evidence."""


class QuestPairedPoseCaptureCleanupError(RuntimeError):
    """The passive observer or its uniquely owned host did not close cleanly."""


class QuestPairedPoseCaptureStartupError(RuntimeError):
    """The unique Quest host/application failed before becoming caller-owned."""


class QuestControllerSelectionUnavailableError(RuntimeError):
    """A retryable controller discovery/tracking condition blocks selection."""


class QuestControllerTrackingUnavailableError(
    QuestControllerSelectionUnavailableError
):
    """The discovered controller has no current real grip pose."""


def _required_text(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    return normalized


def _non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_timeout(value: object, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return normalized


@dataclass(frozen=True)
class PassiveRobotPoseJournalSource:
    """Narrow, no-refresh view of the one policy-deploy pose journal."""

    receiver_clock: ClockDomainDescriptor
    monotonic_ns_clock: Callable[[], int] = field(repr=False, compare=False)
    coordinate_binding_by_side: Mapping[str, PoseCoordinateBinding]
    _atomic_cut: Callable[[], PoseMeasurementJournalCut | None] = field(
        repr=False,
        compare=False,
    )
    _events_after: Callable[[int], tuple[PoseMeasurementJournalEvent, ...]] = field(
        repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.receiver_clock, ClockDomainDescriptor):
            raise TypeError("receiver_clock must be a ClockDomainDescriptor")
        if self.receiver_clock.monotonic is not True:
            raise ValueError("receiver_clock.monotonic must be exactly True")
        if not callable(self.monotonic_ns_clock):
            raise TypeError("monotonic_ns_clock must be callable")
        if not callable(self._atomic_cut) or not callable(self._events_after):
            raise TypeError("pose journal cut/event readers must be callable")
        bindings = dict(self.coordinate_binding_by_side)
        if not bindings:
            raise ValueError("coordinate_binding_by_side must not be empty")
        if set(bindings) - {"left", "right"}:
            raise ValueError("coordinate_binding_by_side contains an unknown side")
        if any(
            not isinstance(value, PoseCoordinateBinding) for value in bindings.values()
        ):
            raise TypeError("coordinate bindings must be PoseCoordinateBinding values")
        mismatches = sorted(
            side
            for side, binding in bindings.items()
            if binding.receipt_clock_domain_id != self.receiver_clock.clock_domain_id
        )
        if mismatches:
            raise ValueError(
                f"robot pose bindings differ from the receiver clock: {mismatches}"
            )
        object.__setattr__(
            self,
            "coordinate_binding_by_side",
            MappingProxyType(bindings),
        )

    def atomic_cut(self) -> PoseMeasurementJournalCut:
        """Read a typed journal cut without refreshing or querying a backend."""

        cut = self._atomic_cut()
        if not isinstance(cut, PoseMeasurementJournalCut):
            raise TypeError("robot pose journal returned an invalid cut")
        if cut.receiver_clock != self.receiver_clock:
            raise RuntimeError("robot pose journal receiver clock drifted")
        return cut

    def events_after(
        self,
        event_sequence: int,
    ) -> tuple[PoseMeasurementJournalEvent, ...]:
        """Read retained global events after one inclusive cursor."""

        cursor = _non_negative_int(event_sequence, name="event_sequence")
        events = tuple(self._events_after(cursor))
        if any(
            not isinstance(
                event,
                (PoseMeasurementAcceptedEvent, PoseMeasurementInvalidationEvent),
            )
            for event in events
        ):
            raise TypeError("robot pose journal returned an untyped event")
        return events


def bind_passive_robot_pose_journal_source(
    *,
    arm_reader: ArmStateReader,
    pose_observation: RM65PoseObservationAssembly,
) -> PassiveRobotPoseJournalSource:
    """Remove refresh/backend authority before handing the source to Quest."""

    if not isinstance(arm_reader, ArmStateReader):
        raise TypeError("arm_reader must be the policy deployment ArmStateReader")
    if not isinstance(pose_observation, RM65PoseObservationAssembly):
        raise TypeError("pose_observation must be an RM65PoseObservationAssembly")
    if arm_reader.receipt_clock != pose_observation.receipt_clock:
        raise ValueError("arm reader and pose assembly receiver clocks differ")
    if (
        arm_reader.receipt_monotonic_ns_clock
        is not pose_observation.receipt_monotonic_ns_clock
    ):
        raise ValueError("arm reader and pose assembly clock callables differ")
    expected = {
        side: projection.coordinate_binding
        for side, projection in pose_observation.projection_by_side.items()
    }
    actual = {
        side: projection.coordinate_binding
        for side, projection in arm_reader.pose_projection_by_side.items()
    }
    if actual != expected:
        raise ValueError("arm reader pose projections differ from the assembly")
    return PassiveRobotPoseJournalSource(
        receiver_clock=pose_observation.receipt_clock,
        monotonic_ns_clock=pose_observation.receipt_monotonic_ns_clock,
        coordinate_binding_by_side=expected,
        _atomic_cut=arm_reader.pose_measurement_journal_cut,
        _events_after=arm_reader.pose_measurement_events_after,
    )


@dataclass(frozen=True)
class QuestControllerCandidate:
    """Pose-redacted controller identity observed after the subscription cut."""

    server_instance_id: str
    client_id: int
    client_instance_id: str
    xr_session_id: str
    reference_space_type: str
    reference_space_generation: int
    controller_source_session_id: str
    handedness: str
    target_ray_mode: str
    profiles: tuple[str, ...]
    receipt_clock_domain_id: str
    first_event_sequence: int
    first_sample_sequence: int
    candidate_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "server_instance_id",
            "client_instance_id",
            "xr_session_id",
            "reference_space_type",
            "controller_source_session_id",
            "handedness",
            "target_ray_mode",
            "receipt_clock_domain_id",
        ):
            object.__setattr__(
                self,
                name,
                _required_text(getattr(self, name), name=name),
            )
        client_id = _non_negative_int(self.client_id, name="client_id")
        generation = _non_negative_int(
            self.reference_space_generation,
            name="reference_space_generation",
        )
        first_event = _non_negative_int(
            self.first_event_sequence,
            name="first_event_sequence",
        )
        first_sample = _non_negative_int(
            self.first_sample_sequence,
            name="first_sample_sequence",
        )
        profiles = tuple(
            _required_text(value, name="profile") for value in self.profiles
        )
        if len(profiles) != len(set(profiles)):
            raise ValueError("profiles must be unique")
        if self.target_ray_mode != "tracked-pointer":
            raise ValueError("capture candidates must use tracked-pointer poses")
        payload = {
            "schema": QUEST_CONTROLLER_CANDIDATE_SCHEMA,
            "server_instance_id": self.server_instance_id,
            "client_id": client_id,
            "client_instance_id": self.client_instance_id,
            "xr_session_id": self.xr_session_id,
            "reference_space_type": self.reference_space_type,
            "reference_space_generation": generation,
            "controller_source_session_id": self.controller_source_session_id,
            "handedness": self.handedness,
            "target_ray_mode": self.target_ray_mode,
            "profiles": list(profiles),
            "receipt_clock_domain_id": self.receipt_clock_domain_id,
            "first_event_sequence": first_event,
            "first_sample_sequence": first_sample,
        }
        object.__setattr__(self, "client_id", client_id)
        object.__setattr__(self, "reference_space_generation", generation)
        object.__setattr__(self, "first_event_sequence", first_event)
        object.__setattr__(self, "first_sample_sequence", first_sample)
        object.__setattr__(self, "profiles", profiles)
        object.__setattr__(
            self,
            "candidate_hash",
            canonical_content_hash(
                payload,
                domain=QUEST_CONTROLLER_CANDIDATE_HASH_DOMAIN,
            ),
        )

    @property
    def identity_key(self) -> tuple[object, ...]:
        # ``profiles`` is retained in the candidate/evidence payload as
        # diagnostic metadata, but Quest Browser may change it while the same
        # XRInputSource becomes more specific. It must not create a second
        # candidate for the same session-local source.
        return (
            self.server_instance_id,
            self.client_id,
            self.client_instance_id,
            self.xr_session_id,
            self.reference_space_type,
            self.reference_space_generation,
            self.controller_source_session_id,
            self.handedness,
            self.target_ray_mode,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": QUEST_CONTROLLER_CANDIDATE_SCHEMA,
            "server_instance_id": self.server_instance_id,
            "client_id": self.client_id,
            "client_instance_id": self.client_instance_id,
            "xr_session_id": self.xr_session_id,
            "reference_space_type": self.reference_space_type,
            "reference_space_generation": self.reference_space_generation,
            "controller_source_session_id": self.controller_source_session_id,
            "handedness": self.handedness,
            "target_ray_mode": self.target_ray_mode,
            "profiles": list(self.profiles),
            "receipt_clock_domain_id": self.receipt_clock_domain_id,
            "first_event_sequence": self.first_event_sequence,
            "first_sample_sequence": self.first_sample_sequence,
            "candidate_hash": self.candidate_hash,
        }


@dataclass(frozen=True)
class QuestCaptureSelectionIntent:
    candidate_hash: str
    robot_side: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_hash",
            _required_text(self.candidate_hash, name="candidate_hash"),
        )
        side = _required_text(self.robot_side, name="robot_side")
        if side not in {"left", "right"}:
            raise ValueError("robot_side must be 'left' or 'right'")
        object.__setattr__(self, "robot_side", side)


@dataclass(frozen=True)
class QuestCaptureFrameFacts:
    candidate_hash: str
    reference_frame_id: str
    moving_frame_id: str
    xr_frame_clock_domain_id: str

    def __post_init__(self) -> None:
        for name in (
            "candidate_hash",
            "reference_frame_id",
            "moving_frame_id",
            "xr_frame_clock_domain_id",
        ):
            object.__setattr__(
                self,
                name,
                _required_text(getattr(self, name), name=name),
            )


def build_quest_controller_runtime_loader(
    *,
    receiver_clock: ClockDomainDescriptor,
    monotonic_ns_clock: Callable[[], int],
    fallback_loader: Callable[..., Any] = load_feature_runtime,
) -> Callable[..., Any]:
    """Inject the exact receiver clock into the unique controller runtime."""

    if not isinstance(receiver_clock, ClockDomainDescriptor):
        raise TypeError("receiver_clock must be a ClockDomainDescriptor")
    if receiver_clock.monotonic is not True:
        raise ValueError("receiver_clock.monotonic must be exactly True")
    if not callable(monotonic_ns_clock):
        raise TypeError("monotonic_ns_clock must be callable")
    if not callable(fallback_loader):
        raise TypeError("fallback_loader must be callable")

    def runtime_loader(name, server, config, stop_event):
        feature_config = dict(config)
        if str(name) != CONTROLLER_POSE_FEATURE_NAME:
            return fallback_loader(name, server, feature_config, stop_event)
        configured_clock = feature_config.get("receipt_clock_domain_id")
        expected_clock = receiver_clock.clock_domain_id
        if configured_clock is not None and configured_clock != expected_clock:
            raise ValueError(
                "controller-pose receipt_clock_domain_id conflicts with the "
                "policy-deploy receiver clock"
            )
        feature_config["receipt_clock_domain_id"] = expected_clock
        return ControllerPoseTelemetryRuntime(
            server,
            feature_config,
            stop_event,
            monotonic_ns_clock=monotonic_ns_clock,
        )

    return runtime_loader


@dataclass(frozen=True)
class QuestPairedPoseCaptureLaunchConfig:
    config_path: Path
    force_http: bool
    profile: StaticWindowCaptureProfile
    capture_id: str
    startup_timeout_s: float
    callback_drain_timeout_s: float
    host_join_timeout_s: float
    realman_left_ip: str | None = None
    realman_left_urdf: Path | None = None
    controller_handedness: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "config_path", Path(self.config_path))
        if not isinstance(self.force_http, bool):
            raise TypeError("force_http must be boolean")
        if not isinstance(self.profile, StaticWindowCaptureProfile):
            raise TypeError("profile must be a StaticWindowCaptureProfile")
        object.__setattr__(
            self,
            "capture_id",
            _required_text(self.capture_id, name="capture_id"),
        )
        left_ip = self.realman_left_ip
        left_urdf = self.realman_left_urdf
        if (left_ip is None) != (left_urdf is None):
            raise ValueError(
                "realman_left_ip and realman_left_urdf must be supplied together"
            )
        if left_ip is not None:
            object.__setattr__(
                self,
                "realman_left_ip",
                _required_text(left_ip, name="realman_left_ip"),
            )
            object.__setattr__(self, "realman_left_urdf", Path(left_urdf))
        handedness = self.controller_handedness
        if handedness in (None, "", "any"):
            handedness = None
        else:
            handedness = _required_text(
                handedness,
                name="controller_handedness",
            ).lower()
            if handedness not in {"left", "right"}:
                raise ValueError(
                    "controller_handedness must be left, right, any, or None"
                )
        object.__setattr__(self, "controller_handedness", handedness)
        for name in (
            "startup_timeout_s",
            "callback_drain_timeout_s",
            "host_join_timeout_s",
        ):
            object.__setattr__(
                self,
                name,
                _positive_timeout(getattr(self, name), name=name),
            )


@runtime_checkable
class QuestPairedPoseCaptureApplicationPort(Protocol):
    """Bounded passive lifecycle used by the policy application root."""

    passive_journal_only: bool
    bounded_nonblocking_poll: bool
    robot_pose_journal: PassiveRobotPoseJournalSource

    def poll(self) -> Mapping[str, Any]: ...

    def publish_capture_status(
        self,
        *,
        terminal_message: str | None = None,
    ) -> Mapping[str, Any]: ...

    def publish_diagnostic_preview(self, diagnostic_fit: Any) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


def require_quest_paired_pose_capture_application(
    value: Any,
    *,
    robot_pose_journal: PassiveRobotPoseJournalSource,
) -> QuestPairedPoseCaptureApplicationPort:
    if not isinstance(robot_pose_journal, PassiveRobotPoseJournalSource):
        raise TypeError("robot_pose_journal must be PassiveRobotPoseJournalSource")
    if not isinstance(value, QuestPairedPoseCaptureApplicationPort):
        raise TypeError("Quest paired-pose application does not implement its port")
    if not isinstance(value, QuestPairedPoseCaptureApplication):
        raise TypeError(
            "Quest paired-pose factory must return the concrete application"
        )
    if getattr(value, "passive_journal_only", None) is not True:
        raise ValueError("Quest paired-pose application must be journal-only")
    if getattr(value, "bounded_nonblocking_poll", None) is not True:
        raise ValueError("Quest paired-pose application poll must be bounded")
    if value.robot_pose_journal is not robot_pose_journal:
        raise ValueError(
            "Quest paired-pose application bound a different robot journal"
        )
    return value


class QuestPairedPoseCaptureApplication:
    """Bounded bridge from source events into one explicit R2 capture."""

    passive_journal_only = True
    bounded_nonblocking_poll = True

    def __init__(
        self,
        *,
        runtime: ControllerPoseTelemetryRuntime,
        robot_pose_journal: PassiveRobotPoseJournalSource,
        profile: StaticWindowCaptureProfile,
        capture_id: str,
        callback_drain_timeout_s: float,
        owned_host: QuestDevHost | None = None,
        owned_host_thread: threading.Thread | None = None,
        host_join_timeout_s: float | None = None,
        host_error_box: dict[str, BaseException] | None = None,
    ) -> None:
        if not isinstance(runtime, ControllerPoseTelemetryRuntime):
            raise TypeError("runtime must be ControllerPoseTelemetryRuntime")
        if not isinstance(robot_pose_journal, PassiveRobotPoseJournalSource):
            raise TypeError("robot_pose_journal must be a passive journal source")
        if not isinstance(profile, StaticWindowCaptureProfile):
            raise TypeError("profile must be a StaticWindowCaptureProfile")
        if runtime.receipt_clock_domain_id != (
            robot_pose_journal.receiver_clock.clock_domain_id
        ):
            raise ValueError("Quest and robot receipt clock domains differ")
        if runtime.monotonic_ns_clock is not robot_pose_journal.monotonic_ns_clock:
            raise ValueError("Quest and robot receipt clock callables differ")
        if getattr(runtime.server, "monotonic_ns_clock", None) is not (
            robot_pose_journal.monotonic_ns_clock
        ):
            raise ValueError("Quest HTTP and robot receipt clock callables differ")
        if owned_host is None and owned_host_thread is not None:
            raise ValueError("owned_host_thread requires owned_host")
        if owned_host is not None:
            if owned_host_thread is None:
                raise ValueError("owned_host requires its serving thread")
            if host_join_timeout_s is None:
                raise ValueError("owned_host requires host_join_timeout_s")

        self.runtime = runtime
        self.robot_pose_journal = robot_pose_journal
        self.profile = profile
        self.capture_id = _required_text(capture_id, name="capture_id")
        self.callback_drain_timeout_s = _positive_timeout(
            callback_drain_timeout_s,
            name="callback_drain_timeout_s",
        )
        self._owned_host = owned_host
        self._owned_host_thread = owned_host_thread
        self._host_join_timeout_s = (
            None
            if host_join_timeout_s is None
            else _positive_timeout(host_join_timeout_s, name="host_join_timeout_s")
        )
        self._host_error_box = host_error_box if host_error_box is not None else {}
        self._condition = threading.Condition(threading.RLock())
        self._ingress_generation = 0
        self._subscription: ControllerPoseSubscription | None = None
        self._subscription_active = False
        self._initial_robot_cut = robot_pose_journal.atomic_cut()
        self._robot_cursor = self._initial_robot_cut.event_sequence
        self._staged_quest_events: dict[int, QuestPoseEvent] = {}
        self._capture_handed_off_sequences: set[int] = set()
        self._candidate_by_identity: dict[
            tuple[object, ...], QuestControllerCandidate
        ] = {}
        self._candidate_by_hash: dict[str, QuestControllerCandidate] = {}
        self._candidate_cursor: int | None = None
        self._selected_candidate_hash: str | None = None
        self._selected_robot_side: str | None = None
        self._selected_binding: PairedPoseCaptureBinding | None = None
        self._capture: ManualStaticPairedPoseCapture | None = None
        self._pending_window_end: tuple[int, int, int] | None = None
        self._close_cut: ControllerPoseObserverCloseCut | None = None
        self._final_robot_cut: PoseMeasurementJournalCut | None = None
        self._evidence: PairedPoseCaptureEvidence | None = None
        self._terminal_reason: str | None = None
        self._sealing = False
        self._closing = False
        self._closed = False
        self._close_attempt = 0
        self._last_close_error: str | None = None
        self._host_cleanup_failure: str | None = None
        self._discard_callbacks = False

        subscription: ControllerPoseSubscription | None = None
        try:
            ingress_generation = self._ingress_generation
            subscription = runtime.subscribe_with_cut(
                on_sample=lambda event: self._on_quest_event(
                    event,
                    ingress_generation=ingress_generation,
                ),
                on_invalidation=lambda event: self._on_quest_event(
                    event,
                    ingress_generation=ingress_generation,
                ),
            )
            with self._condition:
                self._validate_subscription_locked(subscription)
                self._subscription = subscription
                self._subscription_active = True
                self._candidate_cursor = subscription.next_event_sequence
                for event in self._staged_quest_events.values():
                    if event.event_sequence < subscription.next_event_sequence:
                        raise RuntimeError(
                            "Quest callback predates the atomic subscription cut"
                        )
                self._project_candidates_locked()
                if self._terminal_reason is not None:
                    raise QuestPairedPoseCaptureTerminalError(self._terminal_reason)
        except Exception:
            with self._condition:
                self._discard_callbacks = True
            if subscription is not None:
                try:
                    runtime.unsubscribe_with_cut(subscription.token)
                except Exception as cleanup_error:
                    raise QuestPairedPoseCaptureCleanupError(
                        "Quest subscription setup rollback failed"
                    ) from cleanup_error
            raise

    @property
    def evidence(self) -> PairedPoseCaptureEvidence | None:
        with self._condition:
            return self._evidence

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def candidates(self) -> tuple[QuestControllerCandidate, ...]:
        with self._condition:
            return tuple(
                sorted(
                    self._candidate_by_hash.values(),
                    key=lambda value: (
                        value.first_event_sequence,
                        value.client_id,
                        value.controller_source_session_id,
                    ),
                )
            )

    def select_unique_candidate(
        self,
        *,
        robot_side: str,
        preferred_handedness: str | None = None,
    ) -> PairedPoseCaptureBinding:
        """Select one deterministic controller without guessing between peers.

        The terminal path may optionally constrain handedness.  If the
        constraint still leaves zero or multiple candidates, the operator must
        resolve the ambiguity; this method never silently picks the first
        controller.
        """

        side = _required_text(robot_side, name="robot_side").lower()
        if side not in {"left", "right"}:
            raise ValueError("robot_side must be 'left' or 'right'")
        handedness = (
            None
            if preferred_handedness in (None, "", "any")
            else _required_text(
                preferred_handedness,
                name="preferred_handedness",
            ).lower()
        )
        if handedness is not None and handedness not in {"left", "right"}:
            raise ValueError("preferred_handedness must be left, right, any, or None")
        candidates = self.candidates()
        if handedness is not None:
            candidates = tuple(
                candidate
                for candidate in candidates
                if candidate.handedness == handedness
            )
        if not candidates:
            qualifier = "" if handedness is None else f" with handedness={handedness!r}"
            raise QuestControllerSelectionUnavailableError(
                "no active Quest tracked-pointer candidate"
                f"{qualifier}; pair the Quest page and enter MR/VR first"
            )
        if len(candidates) != 1:
            summary = ", ".join(
                f"{candidate.handedness}:{candidate.candidate_hash[:12]}"
                for candidate in candidates
            )
            raise RuntimeError(
                "Quest controller selection is ambiguous; keep exactly one "
                f"candidate or pass a handedness filter (candidates: {summary})"
            )
        candidate = candidates[0]
        frame_facts = QuestCaptureFrameFacts(
            candidate_hash=candidate.candidate_hash,
            reference_frame_id=(
                f"quest-{candidate.reference_space_type}-generation-"
                f"{candidate.reference_space_generation}"
            ),
            moving_frame_id=(
                f"quest-controller-{candidate.handedness}-"
                f"{candidate.controller_source_session_id}"
            ),
            xr_frame_clock_domain_id="webxr-frame-clock",
        )
        return self.select(
            intent=QuestCaptureSelectionIntent(
                candidate_hash=candidate.candidate_hash,
                robot_side=side,
            ),
            frame_facts=frame_facts,
        )

    def _validate_subscription_locked(
        self,
        subscription: ControllerPoseSubscription,
    ) -> None:
        if not isinstance(subscription, ControllerPoseSubscription):
            raise TypeError("controller runtime returned an invalid subscription")
        if subscription.server_instance_id != self.runtime.server_instance_id:
            raise ValueError("Quest subscription server instance drifted")
        if subscription.receipt_clock_domain_id != (
            self.robot_pose_journal.receiver_clock.clock_domain_id
        ):
            raise ValueError("Quest subscription receipt clock differs")

    def _terminal_locked(self, reason: str) -> None:
        if self._terminal_reason is None:
            self._terminal_reason = _required_text(reason, name="terminal reason")

    def _sync_host_error_locked(self) -> None:
        error = self._host_error_box.get("error")
        if error is not None and not self._closing:
            self._terminal_locked(f"quest_host_failed:{type(error).__name__}:{error}")

    def _on_quest_event(
        self,
        event: QuestPoseEvent,
        *,
        ingress_generation: int | None = None,
    ) -> None:
        """Bounded callback: stage only; never invoke the reducer here."""

        with self._condition:
            generation = (
                self._ingress_generation
                if ingress_generation is None
                else ingress_generation
            )
            if generation != self._ingress_generation:
                self._condition.notify_all()
                return
            if self._discard_callbacks:
                self._condition.notify_all()
                return
            if not isinstance(
                event,
                (ControllerPoseAcceptedEvent, ControllerPoseInvalidationEvent),
            ):
                self._terminal_locked("Quest callback delivered an untyped event")
                self._condition.notify_all()
                return
            if event.server_instance_id != self.runtime.server_instance_id:
                self._terminal_locked("Quest callback server instance drifted")
                self._condition.notify_all()
                return
            if event.receipt_clock_domain_id != (
                self.robot_pose_journal.receiver_clock.clock_domain_id
            ):
                self._terminal_locked("Quest callback receipt clock drifted")
                self._condition.notify_all()
                return
            subscription = self._subscription
            if (
                subscription is not None
                and event.event_sequence < subscription.next_event_sequence
            ):
                self._terminal_locked("Quest callback predates subscription cut")
                self._condition.notify_all()
                return
            existing = self._staged_quest_events.get(event.event_sequence)
            if existing is not None:
                if existing != event:
                    self._terminal_locked(
                        "Quest event sequence was reused with different content"
                    )
                self._condition.notify_all()
                return
            if len(self._staged_quest_events) >= self.profile.max_event_history:
                self._terminal_locked("Quest callback staging overflow")
                self._condition.notify_all()
                return
            self._staged_quest_events[event.event_sequence] = event
            self._project_candidates_locked()
            self._condition.notify_all()

    def _project_candidates_locked(self) -> None:
        """Project pose-redacted candidates only in global event order."""

        cursor = self._candidate_cursor
        if cursor is None:
            return
        while cursor in self._staged_quest_events:
            event = self._staged_quest_events[cursor]
            if isinstance(event, ControllerPoseAcceptedEvent):
                self._record_candidates_locked(event)
            else:
                self._remove_invalidated_candidates_locked(event)
            cursor += 1
        self._candidate_cursor = cursor
        self._prune_staged_quest_events_locked()

    def _remove_invalidated_candidates_locked(
        self,
        event: ControllerPoseInvalidationEvent,
    ) -> None:
        invalidated = tuple(
            (identity, candidate)
            for identity, candidate in self._candidate_by_identity.items()
            if candidate.server_instance_id == event.server_instance_id
            and candidate.client_id == event.client_id
            and candidate.client_instance_id == event.client_instance_id
            and candidate.xr_session_id == event.xr_session_id
            and candidate.reference_space_type == event.reference_space_type
            and candidate.reference_space_generation
            == event.previous_reference_space_generation
        )
        for identity, candidate in invalidated:
            self._candidate_by_identity.pop(identity, None)
            self._candidate_by_hash.pop(candidate.candidate_hash, None)

    def _record_candidates_locked(self, event: ControllerPoseAcceptedEvent) -> None:
        sample = event.sample
        for controller in sample.controllers:
            if controller.emulated_position is not False:
                continue
            if controller.target_ray_mode != "tracked-pointer":
                continue
            if len(controller.profiles) != len(set(controller.profiles)):
                continue
            candidate = QuestControllerCandidate(
                server_instance_id=event.server_instance_id,
                client_id=event.client_id,
                client_instance_id=sample.client_instance_id,
                xr_session_id=sample.xr_session_id,
                reference_space_type=sample.reference_space_type,
                reference_space_generation=sample.reference_space_generation,
                controller_source_session_id=controller.source_session_id,
                handedness=controller.handedness,
                target_ray_mode=controller.target_ray_mode,
                profiles=controller.profiles,
                receipt_clock_domain_id=event.receipt_clock_domain_id or "",
                first_event_sequence=event.event_sequence,
                first_sample_sequence=sample.sample_sequence,
            )
            if candidate.identity_key in self._candidate_by_identity:
                continue
            if len(self._candidate_by_identity) >= self.profile.max_event_history:
                self._terminal_locked("Quest controller candidate history overflow")
                return
            self._candidate_by_identity[candidate.identity_key] = candidate
            self._candidate_by_hash[candidate.candidate_hash] = candidate

    def select(
        self,
        *,
        intent: QuestCaptureSelectionIntent,
        frame_facts: QuestCaptureFrameFacts,
    ) -> PairedPoseCaptureBinding:
        if not isinstance(intent, QuestCaptureSelectionIntent):
            raise TypeError("intent must be QuestCaptureSelectionIntent")
        if not isinstance(frame_facts, QuestCaptureFrameFacts):
            raise TypeError("frame_facts must be QuestCaptureFrameFacts")
        with self._condition:
            self._assert_operator_action_available_locked()
            if self._capture is not None:
                raise RuntimeError("a Quest controller is already selected")
            if frame_facts.candidate_hash != intent.candidate_hash:
                raise ValueError("selection and frame facts name different candidates")
            try:
                candidate = self._candidate_by_hash[intent.candidate_hash]
            except KeyError:
                raise KeyError("selected Quest candidate was not observed") from None
            try:
                robot_binding = self.robot_pose_journal.coordinate_binding_by_side[
                    intent.robot_side
                ]
            except KeyError:
                raise KeyError("selected robot side is not bound") from None
            tracking_rejection = self._candidate_tracking_rejection_locked(candidate)
            if tracking_rejection is not None:
                raise QuestControllerTrackingUnavailableError(
                    "selected Quest controller "
                    f"{candidate.controller_source_session_id} is not currently "
                    f"usable: {tracking_rejection}. A visible system controller "
                    "model or short-lived cached axis is not calibration evidence"
                )
            subscription = self._rotate_subscription_for_capture_locked(candidate)
            baseline = -1
            matches = tuple(
                session
                for session in subscription.active_sessions
                if session.client_id == candidate.client_id
                and session.client_instance_id == candidate.client_instance_id
                and session.xr_session_id == candidate.xr_session_id
                and session.reference_space_type == candidate.reference_space_type
                and session.reference_space_generation
                == candidate.reference_space_generation
            )
            if len(matches) > 1:
                raise RuntimeError("subscription cut repeated one Quest session")
            if matches and matches[0].last_accepted_sample_sequence is not None:
                baseline = int(matches[0].last_accepted_sample_sequence)
            robot_selection_cut = self.robot_pose_journal.atomic_cut()
            self._validate_robot_selection_cut_locked(
                robot_selection_cut,
                robot_binding,
            )
            self._robot_cursor = robot_selection_cut.event_sequence
            quest_binding = QuestControllerCaptureBinding(
                server_instance_id=subscription.server_instance_id,
                subscription_token=subscription.token,
                client_id=candidate.client_id,
                client_instance_id=candidate.client_instance_id,
                xr_session_id=candidate.xr_session_id,
                reference_space_type=candidate.reference_space_type,
                reference_space_generation=candidate.reference_space_generation,
                controller_source_session_id=(candidate.controller_source_session_id),
                handedness=candidate.handedness,
                target_ray_mode=candidate.target_ray_mode,
                profiles=candidate.profiles,
                reference_frame_id=frame_facts.reference_frame_id,
                moving_frame_id=frame_facts.moving_frame_id,
                xr_frame_clock_domain_id=(frame_facts.xr_frame_clock_domain_id),
                receipt_clock_domain_id=candidate.receipt_clock_domain_id,
                next_event_sequence=subscription.next_event_sequence,
                baseline_sample_sequence=baseline,
            )
            binding = PairedPoseCaptureBinding(
                capture_id=self.capture_id,
                receiver_clock=self.robot_pose_journal.receiver_clock,
                robot=robot_binding,
                robot_next_event_sequence=robot_selection_cut.event_sequence + 1,
                quest=quest_binding,
            )
            self._capture = ManualStaticPairedPoseCapture(
                profile=self.profile,
                binding=binding,
            )
            self._selected_candidate_hash = candidate.candidate_hash
            self._selected_robot_side = intent.robot_side
            self._selected_binding = binding
            self._drain_quest_events_locked()
            if self._terminal_reason is not None:
                raise QuestPairedPoseCaptureTerminalError(self._terminal_reason)
            return binding

    def _candidate_tracking_rejection_locked(
        self,
        candidate: QuestControllerCandidate,
    ) -> str | None:
        """Explain why a historical candidate lacks a current real grip pose.

        Candidate projection is intentionally historical: it records that a
        source was once observed after the discovery cut.  A later
        ``getPose() == null`` event is accepted as an ordered empty sample, so
        the historical candidate must not by itself authorize selection.  The
        runtime's latest sample and pose-redacted diagnostic form the narrow
        freshness check. The capture reducer remains fail-closed if tracking
        is lost immediately after this check.
        """

        latest = self.runtime.latest_sample(client_id=candidate.client_id)
        if latest is None:
            return "no accepted controller-pose sample exists for this XR client"
        now_ns = int(self.runtime.monotonic_ns_clock())
        age_ns = max(0, now_ns - int(latest.received_monotonic_ns))
        maximum_age_ns = round(DEFAULT_CURRENT_TRACKING_MAX_AGE_S * 1_000_000_000)
        if age_ns > maximum_age_ns:
            return (
                f"latest controller telemetry is stale ({age_ns / 1e9:.3f}s old; "
                f"limit={DEFAULT_CURRENT_TRACKING_MAX_AGE_S:.3f}s)"
            )
        if (
            latest.client_instance_id != candidate.client_instance_id
            or latest.xr_session_id != candidate.xr_session_id
            or latest.reference_space_type != candidate.reference_space_type
            or latest.reference_space_generation
            != candidate.reference_space_generation
        ):
            return "XR session or reference-space generation changed"
        matches = tuple(
            controller
            for controller in latest.controllers
            if controller.source_session_id == candidate.controller_source_session_id
        )
        if len(matches) > 1:
            return "latest sample repeats the selected source identity"
        if matches:
            controller = matches[0]
            if controller.handedness != candidate.handedness:
                return "controller handedness changed"
            if controller.target_ray_mode != candidate.target_ray_mode:
                return "controller target-ray mode changed"
            if controller.emulated_position is True:
                return (
                    "emulated_position (WebXR is estimating rather than measuring "
                    "the grip position)"
                )
            return None

        diagnostics = tuple(
            diagnostic
            for diagnostic in latest.controller_diagnostics
            if diagnostic.source_session_id
            == candidate.controller_source_session_id
        )
        if len(diagnostics) > 1:
            return "latest sample repeats the selected tracking diagnostic"
        if diagnostics:
            diagnostic = diagnostics[0]
            details = {
                "missing_grip_space": "XRInputSource.gripSpace is absent",
                "pose_unavailable": (
                    "WebXR getPose(gripSpace, referenceSpace) returned null"
                ),
                "pose_transform_unavailable": (
                    "WebXR grip pose has no 4x4 transform"
                ),
                "pose_matrix_invalid": "WebXR grip pose matrix is not finite",
                "pose_error": "WebXR getPose raised an exception",
            }.get(diagnostic.reason, "WebXR did not provide a real grip pose")
            error_suffix = (
                ""
                if diagnostic.error_name is None
                else f"; error={diagnostic.error_name}"
            )
            return f"{diagnostic.reason} ({details}{error_suffix})"
        active = ",".join(
            f"{item.source_session_id}:{item.handedness}:{item.reason}"
            for item in latest.controller_diagnostics
        )
        return (
            "selected source is absent from the latest XR input-source set"
            + ("" if not active else f"; current sources={active}")
        )

    def _validate_robot_selection_cut_locked(
        self,
        cut: PoseMeasurementJournalCut,
        binding: PoseCoordinateBinding,
    ) -> None:
        stream_id = binding.stream_id
        binding_hash = binding.semantic_content_hash
        active_hash = cut.binding_hash_by_stream.get(stream_id)
        if active_hash is not None and active_hash != binding_hash:
            self._terminal_locked(
                "selected robot pose binding drifted before selection"
            )
            raise QuestPairedPoseCaptureTerminalError(self._terminal_reason)
        if stream_id in cut.invalidated_stream_ids:
            expected_hash = cut.expected_next_binding_hash_by_stream.get(stream_id)
            if expected_hash != binding_hash:
                self._terminal_locked(
                    "selected robot pose stream is terminal-invalidated"
                )
                raise QuestPairedPoseCaptureTerminalError(self._terminal_reason)

    def _rotate_subscription_for_capture_locked(
        self,
        candidate: QuestControllerCandidate,
    ) -> ControllerPoseSubscription:
        """Replace discovery ingress with a fresh capture fence."""

        previous = self._require_subscription_locked()
        try:
            previous_close = self.runtime.unsubscribe_with_cut(previous.token)
        except Exception as exc:
            self._terminal_locked(
                f"Quest discovery unsubscribe failed:{type(exc).__name__}:{exc}"
            )
            raise QuestPairedPoseCaptureTerminalError(self._terminal_reason) from exc
        self._subscription_active = False
        self._subscription = None
        self._ingress_generation += 1
        if (
            not isinstance(previous_close, ControllerPoseObserverCloseCut)
            or previous_close.subscription_token != previous.token
            or previous_close.server_instance_id != previous.server_instance_id
            or previous_close.receipt_clock_domain_id
            != previous.receipt_clock_domain_id
            or previous_close.stopped
        ):
            self._terminal_locked("Quest discovery close cut drifted")
            raise QuestPairedPoseCaptureTerminalError(self._terminal_reason)
        ingress_generation = self._ingress_generation
        self._candidate_cursor = None
        self._staged_quest_events.clear()
        self._capture_handed_off_sequences.clear()
        try:
            subscription = self.runtime.subscribe_with_cut(
                on_sample=lambda event: self._on_quest_event(
                    event,
                    ingress_generation=ingress_generation,
                ),
                on_invalidation=lambda event: self._on_quest_event(
                    event,
                    ingress_generation=ingress_generation,
                ),
            )
        except Exception as exc:
            self._terminal_locked(
                f"Quest capture subscribe failed:{type(exc).__name__}:{exc}"
            )
            raise QuestPairedPoseCaptureTerminalError(self._terminal_reason) from exc
        self._subscription = subscription
        self._subscription_active = True
        try:
            self._validate_subscription_locked(subscription)
        except Exception as exc:
            self._terminal_locked(
                f"Quest capture subscription cut failed:{type(exc).__name__}:{exc}"
            )
            try:
                self.runtime.unsubscribe_with_cut(subscription.token)
            except Exception as cleanup_exc:
                raise QuestPairedPoseCaptureCleanupError(
                    "Quest malformed capture subscription rollback failed"
                ) from cleanup_exc
            self._subscription = None
            self._subscription_active = False
            raise QuestPairedPoseCaptureTerminalError(self._terminal_reason) from exc
        self._candidate_cursor = subscription.next_event_sequence
        self._project_candidates_locked()
        if subscription.next_event_sequence < previous_close.next_event_sequence:
            self._terminal_locked("Quest capture event cut regressed across rotation")
            raise QuestPairedPoseCaptureTerminalError(self._terminal_reason)
        active_matches = tuple(
            session
            for session in subscription.active_sessions
            if session.client_id == candidate.client_id
            and session.client_instance_id == candidate.client_instance_id
            and session.xr_session_id == candidate.xr_session_id
            and session.reference_space_type == candidate.reference_space_type
            and session.reference_space_generation
            == candidate.reference_space_generation
        )
        if len(active_matches) != 1:
            self._terminal_locked(
                "selected Quest candidate epoch is not active at the capture cut"
            )
            raise QuestPairedPoseCaptureTerminalError(self._terminal_reason)
        last_sample_sequence = active_matches[0].last_accepted_sample_sequence
        if (
            last_sample_sequence is None
            or last_sample_sequence < candidate.first_sample_sequence
        ):
            self._terminal_locked(
                "Quest capture sample baseline regressed across rotation"
            )
            raise QuestPairedPoseCaptureTerminalError(self._terminal_reason)
        return subscription

    def _require_subscription_locked(self) -> ControllerPoseSubscription:
        subscription = self._subscription
        if subscription is None:
            raise RuntimeError("Quest subscription is unavailable")
        return subscription

    def _assert_operator_action_available_locked(self) -> None:
        self._sync_host_error_locked()
        if self._terminal_reason is not None:
            raise QuestPairedPoseCaptureTerminalError(self._terminal_reason)
        if self._closing or self._closed:
            raise RuntimeError("Quest paired-pose application is closing")

    def _require_capture_locked(self) -> ManualStaticPairedPoseCapture:
        self._assert_operator_action_available_locked()
        if self._capture is None:
            raise RuntimeError("no Quest controller candidate is selected")
        return self._capture

    def _drain_quest_events_locked(self) -> None:
        capture = self._capture
        if capture is None or self._terminal_reason is not None:
            return
        for sequence in sorted(self._staged_quest_events):
            if sequence in self._capture_handed_off_sequences:
                continue
            event = self._staged_quest_events[sequence]
            try:
                capture.ingest_quest_event(event)
            except CaptureTerminalError as exc:
                self._terminal_locked(str(exc))
                return
            except Exception as exc:  # noqa: BLE001 - preserve passive terminal fact.
                self._terminal_locked(
                    f"Quest reducer failure:{type(exc).__name__}:{exc}"
                )
                return
            self._capture_handed_off_sequences.add(sequence)
        self._prune_staged_quest_events_locked()

    def _prune_staged_quest_events_locked(self) -> None:
        candidate_cursor = self._candidate_cursor
        if candidate_cursor is None:
            return
        capture = self._capture
        prune_before = candidate_cursor
        if capture is not None:
            capture_cursor = int(capture.snapshot_status()["quest_next_event_sequence"])
            prune_before = min(candidate_cursor, capture_cursor)
        for sequence in tuple(self._staged_quest_events):
            if sequence >= prune_before:
                continue
            self._staged_quest_events.pop(sequence, None)
            self._capture_handed_off_sequences.discard(sequence)

    def _drain_robot_events_locked(self) -> None:
        capture = self._capture
        if capture is None or self._terminal_reason is not None:
            return
        try:
            events = self.robot_pose_journal.events_after(self._robot_cursor)
        except Exception as exc:  # noqa: BLE001 - journal gaps are terminal evidence.
            self._terminal_locked(
                f"robot pose journal read failed:{type(exc).__name__}:{exc}"
            )
            return
        for event in events:
            expected = self._robot_cursor + 1
            if event.event_sequence != expected:
                self._terminal_locked("robot pose journal event gap")
                return
            try:
                capture.ingest_robot_event(event)
            except CaptureTerminalError as exc:
                self._terminal_locked(str(exc))
                return
            except Exception as exc:  # noqa: BLE001 - preserve passive terminal fact.
                self._terminal_locked(
                    f"robot reducer failure:{type(exc).__name__}:{exc}"
                )
                return
            self._robot_cursor = event.event_sequence

    def poll(self) -> Mapping[str, Any]:
        """Drain bounded passive journals; never refresh or query a backend."""

        with self._condition:
            self._sync_host_error_locked()
            if (
                not self._closed
                and self._terminal_reason is None
                and self._evidence is None
            ):
                self._drain_quest_events_locked()
                self._drain_robot_events_locked()
            status = self._snapshot_status_locked()
            if self._terminal_reason is not None and not self._closed:
                raise QuestPairedPoseCaptureTerminalError(self._terminal_reason)
            return status

    def publish_capture_status(
        self,
        *,
        terminal_message: str | None = None,
    ) -> Mapping[str, Any]:
        """Publish the current pose-redacted capture state to Quest."""

        status = dict(self.snapshot_status())
        publisher = getattr(self.runtime.server, "publish_feature_message", None)
        if callable(publisher):
            publisher(
                "visualization",
                capture_status_message(
                    status,
                    terminal_message=terminal_message,
                ),
            )
        return status

    def publish_diagnostic_preview(self, diagnostic_fit: Any) -> Mapping[str, Any]:
        """Publish one candidate-only Quest-space robot-base preview."""

        evidence = self.evidence
        if evidence is None:
            raise RuntimeError("cannot publish a preview before capture is sealed")
        fit = getattr(diagnostic_fit, "fit", None)
        if fit is None:
            raise TypeError("diagnostic_fit must expose fit")
        binding = evidence.binding
        preview_tcp_matrix = None
        samples = getattr(evidence.dataset, "samples", ())
        if samples:
            preview_tcp_matrix = matrix_to_column_major(
                samples[-1].first.matrix_reference_from_moving
            )
        payload = calibration_preview_message(
            robot=f"rm65_{self._selected_robot_side or 'unknown'}",
            capture_id=self.capture_id,
            evidence_semantic_content_hash=evidence.semantic_content_hash,
            diagnostic_fit_semantic_content_hash=str(
                getattr(diagnostic_fit, "semantic_content_hash", "")
            ),
            quest_session_id=binding.quest.xr_session_id,
            reference_space_type=binding.quest.reference_space_type,
            reference_space_generation=binding.quest.reference_space_generation,
            controller_source_session_id=binding.quest.controller_source_session_id,
            matrix_quest_from_robot_base=matrix_to_column_major(
                fit.second_reference_from_first_reference
            ),
            robot_frame=binding.robot.reference_frame_id,
            tcp_frame=binding.robot.moving_frame_id,
            matrix_robot_from_tcp=preview_tcp_matrix,
        )
        publisher = getattr(self.runtime.server, "publish_feature_message", None)
        if callable(publisher):
            publisher("visualization", payload)
        return payload

    def _validate_quest_cut_locked(self, cut: ControllerPoseEventCut) -> None:
        subscription = self._require_subscription_locked()
        if not isinstance(cut, ControllerPoseEventCut):
            raise TypeError("Quest runtime returned an invalid event cut")
        if cut.server_instance_id != subscription.server_instance_id:
            raise ValueError("Quest event cut server instance drifted")
        if cut.receipt_clock_domain_id != subscription.receipt_clock_domain_id:
            raise ValueError("Quest event cut receipt clock drifted")
        if cut.stopped:
            raise RuntimeError("Quest runtime stopped before capture boundary")
        if cut.next_event_sequence < subscription.next_event_sequence:
            raise ValueError("Quest event cut predates the subscription")

    def begin_window(self, *, window_id: str) -> None:
        """Pin an explicit half-open start boundary using the shared clock."""

        self.poll()
        with self._condition:
            capture = self._require_capture_locked()
            start_ns = int(self.robot_pose_journal.monotonic_ns_clock())
            robot_cut = self.robot_pose_journal.atomic_cut()
            quest_cut = self.runtime.atomic_event_cut()
            self._validate_quest_cut_locked(quest_cut)
            capture.begin_window(
                window_id=_required_text(window_id, name="window_id"),
                start_received_monotonic_ns=start_ns,
                robot_start_next_event_sequence=robot_cut.event_sequence + 1,
                quest_start_next_event_sequence=quest_cut.next_event_sequence,
            )

    def complete_window(self) -> StaticWindowReceipt | None:
        """Pin one immutable end tuple and retry only its asynchronous drain."""

        self.poll()
        with self._condition:
            capture = self._require_capture_locked()
            if self._pending_window_end is None:
                robot_cut = self.robot_pose_journal.atomic_cut()
                quest_cut = self.runtime.atomic_event_cut()
                self._validate_quest_cut_locked(quest_cut)
                end_ns = int(self.robot_pose_journal.monotonic_ns_clock())
                self._pending_window_end = (
                    end_ns,
                    robot_cut.event_sequence + 1,
                    quest_cut.next_event_sequence,
                )
            self._drain_quest_events_locked()
            self._drain_robot_events_locked()
            self._assert_operator_action_available_locked()
            end_ns, robot_next, quest_next = self._pending_window_end
            try:
                receipt = capture.complete_window(
                    end_received_monotonic_ns=end_ns,
                    robot_end_next_event_sequence=robot_next,
                    quest_end_next_event_sequence=quest_next,
                )
            except StaticWindowRejectedError:
                self._pending_window_end = None
                raise
            if receipt is not None:
                self._pending_window_end = None
            return receipt

    def _unsubscribe_locked(self) -> ControllerPoseObserverCloseCut | None:
        subscription = self._subscription
        if not self._subscription_active or subscription is None:
            return self._close_cut
        try:
            self._close_cut = self.runtime.unsubscribe_with_cut(subscription.token)
        except Exception as exc:  # noqa: BLE001 - no fabricated close cut.
            self._terminal_locked(
                f"Quest atomic unsubscribe failed:{type(exc).__name__}:{exc}"
            )
        else:
            self._subscription_active = False
        return self._close_cut

    def request_seal(self) -> PairedPoseCaptureEvidence:
        """Explicitly pin source close cuts and seal accepted static windows.

        Normal application shutdown never calls this method implicitly.  An
        operator/application coordinator must first complete every intended
        window and then request sealing as a separate state transition.
        """

        self.poll()
        with self._condition:
            if self._evidence is not None:
                return self._evidence
            self._assert_operator_action_available_locked()
            if self._sealing:
                raise RuntimeError("paired-pose sealing is already in progress")
            self._sealing = True
        try:
            with self._condition:
                capture = self._require_capture_locked()
                capture_status = capture.snapshot_status()
                if capture_status["active_window_id"] is not None:
                    raise RuntimeError("cannot seal while a static window is active")
                if capture_status["completed_window_count"] <= 0:
                    raise RuntimeError(
                        "at least one accepted static window is required"
                    )
                close_cut = self._unsubscribe_locked()
                if close_cut is None or self._terminal_reason is not None:
                    reason = self._terminal_reason or "Quest close cut is unavailable"
                    raise QuestPairedPoseCaptureTerminalError(reason)
                self._final_robot_cut = self.robot_pose_journal.atomic_cut()
                try:
                    evidence = capture.seal_capture(
                        robot_cut=self._final_robot_cut,
                        quest_cut=close_cut,
                    )
                except Exception as exc:
                    self._terminal_locked(
                        f"paired-pose seal failed:{type(exc).__name__}:{exc}"
                    )
                    self._discard_callbacks = True
                    raise QuestPairedPoseCaptureTerminalError(
                        self._terminal_reason or "paired-pose seal failed"
                    ) from exc
                if evidence is not None:
                    self._evidence = evidence
                    self._discard_callbacks = True
                    return evidence
                deadline = time.monotonic() + self.callback_drain_timeout_s
                while True:
                    self._drain_quest_events_locked()
                    self._assert_operator_action_available_locked()
                    quest_next = int(
                        capture.snapshot_status()["quest_next_event_sequence"]
                    )
                    if quest_next >= close_cut.next_event_sequence:
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        self._terminal_locked("Quest close callback drain timed out")
                        self._discard_callbacks = True
                        raise QuestPairedPoseCaptureTerminalError(
                            self._terminal_reason or "Quest callback drain timed out"
                        )
                    self._condition.wait(timeout=remaining)
                self._drain_robot_events_locked()
                self._assert_operator_action_available_locked()
                assert self._final_robot_cut is not None
                try:
                    evidence = capture.seal_capture(
                        robot_cut=self._final_robot_cut,
                        quest_cut=close_cut,
                    )
                except Exception as exc:
                    self._terminal_locked(
                        f"paired-pose seal failed:{type(exc).__name__}:{exc}"
                    )
                    self._discard_callbacks = True
                    raise QuestPairedPoseCaptureTerminalError(
                        self._terminal_reason or "paired-pose seal failed"
                    ) from exc
                if evidence is None:
                    self._terminal_locked("paired-pose finalization did not drain")
                    self._discard_callbacks = True
                    raise QuestPairedPoseCaptureTerminalError(self._terminal_reason)
                self._evidence = evidence
                self._discard_callbacks = True
                return evidence
        finally:
            with self._condition:
                self._sealing = False
                self._condition.notify_all()

    def _cancel_capture(self) -> None:
        """Atomically unsubscribe without manufacturing capture evidence."""

        with self._condition:
            self._unsubscribe_locked()
            self._discard_callbacks = True
            if self._subscription_active:
                raise QuestPairedPoseCaptureCleanupError(
                    self._terminal_reason or "Quest observer unsubscribe failed"
                )

    def _close_owned_host(self) -> None:
        host = self._owned_host
        thread = self._owned_host_thread
        if host is None or thread is None:
            return
        if self._host_cleanup_failure is not None:
            raise QuestPairedPoseCaptureCleanupError(self._host_cleanup_failure)
        errors: list[str] = []
        try:
            host.request_shutdown()
        except Exception as exc:  # noqa: BLE001 - aggregate bounded teardown.
            errors.append(
                f"Quest host shutdown request failed:{type(exc).__name__}:{exc}"
            )
        if thread is threading.current_thread():
            errors.append("Quest host cannot join its serving thread")
        else:
            assert self._host_join_timeout_s is not None
            thread.join(timeout=self._host_join_timeout_s)
            if thread.is_alive():
                errors.append("Quest host serving thread did not stop")
        if not thread.is_alive() and thread is not threading.current_thread():
            serving_error = self._host_error_box.get("error")
            if serving_error is not None:
                self._host_cleanup_failure = (
                    "Quest host serving thread failed:"
                    f"{type(serving_error).__name__}:{serving_error}"
                )
                errors.append(self._host_cleanup_failure)
            try:
                host.close()
            except BaseException as exc:
                self._host_cleanup_failure = (
                    f"Quest host close failed:{type(exc).__name__}:{exc}"
                )
                if not isinstance(exc, Exception):
                    raise
                errors.append(self._host_cleanup_failure)
            else:
                cleanup_proven = self._host_cleanup_failure is None
                if cleanup_proven:
                    with self._condition:
                        self._owned_host = None
                        self._owned_host_thread = None
                        self._subscription_active = False
        if errors:
            with self._condition:
                self._terminal_locked("; ".join(errors))
            raise QuestPairedPoseCaptureCleanupError("; ".join(errors))

    def close(self) -> None:
        """Unsubscribe/finalize before stopping the uniquely owned Quest host."""

        wait_timeout_s = self.callback_drain_timeout_s + (
            0.0 if self._host_join_timeout_s is None else self._host_join_timeout_s
        )
        deadline = time.monotonic() + wait_timeout_s
        with self._condition:
            waited_for_attempt = False
            while self._closing:
                waited_for_attempt = True
                self._condition.wait()
            if self._closed:
                return
            if waited_for_attempt and self._last_close_error is not None:
                raise QuestPairedPoseCaptureCleanupError(self._last_close_error)
            self._close_attempt += 1
            self._last_close_error = None
            self._closing = True
            if self._sealing:
                self._terminal_locked("paired-pose sealing cancelled by close")
                self._discard_callbacks = True
                self._condition.notify_all()
                while self._sealing:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        self._closing = False
                        self._condition.notify_all()
                        raise QuestPairedPoseCaptureCleanupError(
                            "paired-pose sealing did not stop before close"
                        )
                    self._condition.wait(timeout=remaining)
        cleanup_errors: list[Exception] = []
        cleanup_complete = False
        try:
            try:
                self._cancel_capture()
            except Exception as exc:  # noqa: BLE001 - continue owned-host cleanup.
                cleanup_errors.append(exc)
            try:
                self._close_owned_host()
            except Exception as exc:  # noqa: BLE001 - aggregate cleanup evidence.
                cleanup_errors.append(exc)
            cleanup_complete = True
        finally:
            with self._condition:
                self._discard_callbacks = True
                self._closed = cleanup_complete and not cleanup_errors
                self._closing = False
                if cleanup_errors:
                    self._last_close_error = "; ".join(
                        f"{type(error).__name__}:{error}" for error in cleanup_errors
                    )
                self._condition.notify_all()
        if cleanup_errors:
            details = "; ".join(
                f"{type(error).__name__}:{error}" for error in cleanup_errors
            )
            raise QuestPairedPoseCaptureCleanupError(
                f"Quest paired-pose cleanup failed: {details}"
            ) from cleanup_errors[0]

    def snapshot_status(self) -> Mapping[str, Any]:
        with self._condition:
            return self._snapshot_status_locked()

    def _snapshot_status_locked(self) -> Mapping[str, Any]:
        self._sync_host_error_locked()
        capture_status = (
            None if self._capture is None else self._capture.snapshot_status()
        )
        if self._closed:
            state = "closed"
        elif self._terminal_reason is not None:
            state = "terminal"
        elif self._capture is None:
            state = "waiting_for_selection"
        elif self._evidence is not None:
            state = "sealed"
        else:
            state = "bound"
        return {
            "mode": "passive_quest_robot_paired_pose_application",
            "state": state,
            "capture_id": self.capture_id,
            "profile_id": self.profile.profile_id,
            "candidate_count": len(self._candidate_by_hash),
            "candidates": [candidate.as_dict() for candidate in self.candidates()],
            "selected_candidate_hash": self._selected_candidate_hash,
            "selected_robot_side": self._selected_robot_side,
            "selected_binding": (
                None
                if self._selected_binding is None
                else self._selected_binding.as_dict()
            ),
            "subscription_next_event_sequence": (
                None
                if self._subscription is None
                else self._subscription.next_event_sequence
            ),
            "staged_quest_event_count": len(self._staged_quest_events),
            "initial_robot_next_event_sequence": (
                self._initial_robot_cut.event_sequence + 1
            ),
            "terminal_reason": self._terminal_reason,
            "capture": capture_status,
            "evidence_hash": (
                None if self._evidence is None else self._evidence.semantic_content_hash
            ),
            "closed": self._closed,
        }


class QuestTerminalCaptureController:
    """Terminal ``r``/``c`` coordinator for one passive Quest capture.

    The controller deliberately reuses the application methods that the
    desktop operator page used previously.  It owns no keyboard reader; the
    deployment loop passes it keys from its existing ``NonBlockingKeyboard``
    instance, so safety/re-arm and passive-capture input cannot race over
    stdin.  Each ``r`` starts one fixed-duration window.  Completion is
    scheduled from :meth:`poll`, so the existing deployment loop remains the
    only owner of terminal input and application progress.
    """

    def __init__(
        self,
        *,
        application: QuestPairedPoseCaptureApplication,
        robot_side: str,
        preferred_handedness: str | None = None,
        window_id_prefix: str = "static-window",
        window_duration_s: float = DEFAULT_STATIC_WINDOW_DURATION_S,
        monotonic_ns_clock: Callable[[], int] | None = None,
        emit: Callable[[str], None] = print,
        fit_function: Callable[[PairedPoseCaptureEvidence], Any] | None = None,
    ) -> None:
        if not isinstance(application, QuestPairedPoseCaptureApplication):
            raise TypeError("application must be QuestPairedPoseCaptureApplication")
        side = _required_text(robot_side, name="robot_side").lower()
        if side not in {"left", "right"}:
            raise ValueError("robot_side must be 'left' or 'right'")
        handedness = (
            None
            if preferred_handedness in (None, "", "any")
            else _required_text(
                preferred_handedness,
                name="preferred_handedness",
            ).lower()
        )
        if handedness is not None and handedness not in {"left", "right"}:
            raise ValueError(
                "preferred_handedness must be left, right, any, or None"
            )
        if not callable(emit):
            raise TypeError("emit must be callable")
        if fit_function is not None and not callable(fit_function):
            raise TypeError("fit_function must be callable or None")
        self.application = application
        self.robot_side = side
        self.preferred_handedness = handedness
        self.window_id_prefix = _required_text(
            window_id_prefix,
            name="window_id_prefix",
        )
        duration_s = _positive_timeout(window_duration_s, name="window_duration_s")
        clock = monotonic_ns_clock
        if clock is None:
            journal = getattr(application, "robot_pose_journal", None)
            clock = getattr(journal, "monotonic_ns_clock", None)
            if clock is None:
                clock = time.monotonic_ns
        if not callable(clock):
            raise TypeError("monotonic_ns_clock must be callable")
        self.emit = emit
        self.fit_function = fit_function
        self.window_duration_s = duration_s
        self._window_duration_ns = round(duration_s * 1_000_000_000.0)
        if self._window_duration_ns <= 0:
            raise ValueError("window_duration_s is too small for a nanosecond clock")
        self._monotonic_ns_clock = clock
        self._window_index = 0
        self._pending_action: str | None = None
        self._window_deadline_ns: int | None = None
        self._terminal_error: str | None = None
        self._last_status_key: tuple[Any, ...] | None = None
        self._diagnostic_fit: Any | None = None
        self._waiting_selection_reason: str | None = None

    @property
    def diagnostic_fit(self) -> Any | None:
        return self._diagnostic_fit

    @property
    def terminal_error(self) -> str | None:
        return self._terminal_error

    def _status_key(self, status: Mapping[str, Any]) -> tuple[Any, ...]:
        capture = status.get("capture")
        capture = capture if isinstance(capture, Mapping) else {}
        return (
            status.get("state"),
            status.get("candidate_count"),
            status.get("selected_candidate_hash"),
            status.get("selected_robot_side"),
            capture.get("active_window_id"),
            capture.get("completed_window_count"),
            capture.get("draining"),
            capture.get("sealed"),
            status.get("evidence_hash"),
            status.get("terminal_reason"),
            self._pending_action,
        )

    def _publish_status(
        self,
        *,
        terminal_message: str | None = None,
        force: bool = False,
    ) -> Mapping[str, Any]:
        status = self.application.snapshot_status()
        key = self._status_key(status)
        if force or key != self._last_status_key or terminal_message is not None:
            self.application.publish_capture_status(
                terminal_message=terminal_message,
            )
            self._last_status_key = key
        return status

    def _emit_error(self, prefix: str, error: Exception) -> None:
        message = f"{prefix}: {type(error).__name__}: {error}"
        self.emit(f"[Quest] {message}")
        self._publish_status(terminal_message=message, force=True)

    def _ensure_selected(self) -> None:
        status = self.application.snapshot_status()
        if status.get("selected_candidate_hash") is not None:
            return
        binding = self.application.select_unique_candidate(
            robot_side=self.robot_side,
            preferred_handedness=self.preferred_handedness,
        )
        self.emit(
            "[Quest] selected controller candidate "
            f"{binding.quest.controller_source_session_id} "
            f"({binding.quest.handedness}); robot_side={self.robot_side}"
        )
        self._publish_status(force=True)

    def _next_window_id(self) -> str:
        self._window_index += 1
        return f"{self.window_id_prefix}-{self._window_index:03d}"

    def _begin_next_window(self) -> None:
        window_id = self._next_window_id()
        self.application.begin_window(window_id=window_id)
        self._window_deadline_ns = (
            self._now_monotonic_ns() + self._window_duration_ns
        )
        self.emit(
            "[Quest] window started: "
            f"{window_id}; hold the arm/controller still for "
            f"{self.window_duration_s:g}s; it will finish automatically"
        )
        self._publish_status(force=True)

    def _now_monotonic_ns(self) -> int:
        value = int(self._monotonic_ns_clock())
        if value < 0:
            raise ValueError("monotonic_ns_clock returned a negative value")
        return value

    @staticmethod
    def _capture_from_status(status: Mapping[str, Any]) -> Mapping[str, Any]:
        capture = status.get("capture")
        return capture if isinstance(capture, Mapping) else {}

    def _window_is_due(self) -> bool:
        deadline_ns = self._window_deadline_ns
        if deadline_ns is None:
            return False
        return self._now_monotonic_ns() >= deadline_ns

    def _schedule_due_window_completion(self) -> None:
        """Turn an elapsed fixed window into the existing bounded drain action."""

        if self._pending_action is not None or self._window_deadline_ns is None:
            return
        status = self.application.snapshot_status()
        capture = self._capture_from_status(status)
        if capture.get("active_window_id") is None:
            self._window_deadline_ns = None
            return
        if not self._window_is_due():
            return
        self._window_deadline_ns = None
        self._pending_action = "complete"
        self.emit(
            f"[Quest] {self.window_duration_s:g}s static window elapsed; "
            "completing automatically and "
            "draining Quest/RealMan samples..."
        )
        self._publish_status(force=True)

    def _complete_pending_window(self, *, seal_after: bool) -> bool:
        try:
            receipt = self.application.complete_window()
        except StaticWindowRejectedError as error:
            self._pending_action = None
            self._window_deadline_ns = None
            self._emit_error("static window rejected", error)
            return False
        except QuestPairedPoseCaptureTerminalError as error:
            self._pending_action = None
            self._window_deadline_ns = None
            self._terminal_error = str(error)
            self._emit_error("capture became terminal", error)
            return False
        except Exception as error:  # noqa: BLE001 - operator-visible passive error.
            self._pending_action = None
            self._window_deadline_ns = None
            self._emit_error("window completion failed", error)
            return False
        if receipt is None:
            self._window_deadline_ns = None
            self._pending_action = (
                "seal_after_complete" if seal_after else "complete"
            )
            self.emit("[Quest] window end pinned; draining Quest/RealMan samples...")
            self._publish_status(force=True)
            return False
        self._pending_action = None
        self._window_deadline_ns = None
        self.emit(
            "[Quest] window completed: "
            f"{receipt.window_id}; samples="
            f"robot={receipt.metrics.get('robot_sample_count', '?')},"
            f"quest={receipt.metrics.get('quest_sample_count', '?')}"
        )
        self._publish_status(force=True)
        if seal_after:
            self._seal_and_solve()
        return True

    def _seal_and_solve(self) -> None:
        status = self.application.snapshot_status()
        capture = status.get("capture")
        capture = capture if isinstance(capture, Mapping) else {}
        completed = int(capture.get("completed_window_count", 0) or 0)
        if completed < 3:
            self._pending_action = None
            self.emit(
                "[Quest] c requires at least 3 accepted static windows for AX=XB; "
                f"current={completed}. Press r to collect another window."
            )
            self._publish_status(force=True)
            return
        try:
            evidence = self.application.request_seal()
            fit_function = self.fit_function
            if fit_function is None:
                from .quest_paired_pose_evidence_store import (
                    fit_quest_paired_pose_evidence,
                )

                fit_function = fit_quest_paired_pose_evidence
            diagnostic_fit = fit_function(evidence)
            self._diagnostic_fit = diagnostic_fit
            fit = diagnostic_fit.fit
            self.emit(
                "[Quest] diagnostic fit: "
                f"samples={fit.sample_count}, "
                f"translation_rms_m={fit.translation_rms_m:.6g}, "
                f"rotation_rms_deg={fit.rotation_rms_deg:.6g}"
            )
            self.application.publish_diagnostic_preview(diagnostic_fit)
            self.emit(
                "[Quest] preview only; accepted frame not installed. "
                "Quest will draw the left-arm base/TCP when fresh read-only "
                "feedback is available."
            )
            self._publish_status(
                terminal_message=(
                    "diagnostic fit ready; preview only, accepted frame not installed"
                ),
                force=True,
            )
        except QuestPairedPoseCaptureTerminalError as error:
            self._terminal_error = str(error)
            self._emit_error("capture sealing became terminal", error)
        except Exception as error:  # noqa: BLE001 - diagnostic failure is not motion.
            self._pending_action = None
            self._emit_error("diagnostic fit failed", error)

    def _try_pending_action(self) -> None:
        pending = self._pending_action
        if pending == "select_and_begin":
            try:
                self._ensure_selected()
                self._begin_next_window()
            except QuestControllerSelectionUnavailableError as error:
                reason = str(error)
                if reason != self._waiting_selection_reason:
                    self.emit(f"[Quest] waiting for real grip tracking: {reason}")
                    self._waiting_selection_reason = reason
                    self._publish_status(
                        terminal_message=f"waiting for real grip tracking: {reason}",
                        force=True,
                    )
                return
            except QuestPairedPoseCaptureTerminalError as error:
                self._pending_action = None
                self._terminal_error = str(error)
                self._emit_error("capture became terminal", error)
                return
            except Exception as error:  # noqa: BLE001 - keep main loop alive.
                self._pending_action = None
                self._emit_error("deferred capture start failed", error)
                return
            self._pending_action = None
            self._waiting_selection_reason = None
            self._publish_status(force=True)
        elif pending == "complete":
            self._complete_pending_window(seal_after=False)
        elif pending == "seal_after_complete" and self._complete_pending_window(
            seal_after=True
        ):
            return

    def poll(self) -> Mapping[str, Any]:
        """Drain the passive application and progress any bounded ``c`` drain."""

        if self._terminal_error is not None:
            return self._publish_status()
        try:
            status = self.application.poll()
        except QuestPairedPoseCaptureTerminalError as error:
            self._terminal_error = str(error)
            self._emit_error("passive capture became terminal", error)
            return self._publish_status()
        self._publish_status()
        self._schedule_due_window_completion()
        self._try_pending_action()
        return status

    def handle_key(self, key: str) -> bool:
        """Handle one already-polled terminal key; return whether it was used."""

        normalized = str(key).lower()
        if normalized not in {"r", "c"}:
            return False
        if self._terminal_error is not None:
            self.emit(
                "[Quest] key ignored: passive capture is terminal; "
                "restart the observation-only session"
            )
            return True
        if self._pending_action is not None:
            self.emit(
                "[Quest] key ignored: previous window operation is still draining"
            )
            return True
        try:
            self._ensure_selected()
            status = self.application.snapshot_status()
            capture = self._capture_from_status(status)
            active_window = capture.get("active_window_id")
            if normalized == "r":
                if active_window is not None and self._window_is_due():
                    self._pending_action = "complete"
                    self._window_deadline_ns = None
                    self._complete_pending_window(seal_after=False)
                    status = self.application.snapshot_status()
                    capture = self._capture_from_status(status)
                    active_window = capture.get("active_window_id")
                    if self._pending_action is not None or active_window is not None:
                        return True
                if active_window is None:
                    self._begin_next_window()
                else:
                    self.emit(
                        "[Quest] r ignored: current static window is still "
                        f"recording for {self.window_duration_s:g}s; wait for "
                        "automatic completion"
                    )
                return True
            if active_window is not None:
                if not self._window_is_due():
                    self.emit(
                        "[Quest] c ignored: current static window has not reached "
                        f"{self.window_duration_s:g}s; wait for automatic completion"
                    )
                else:
                    self._pending_action = "seal_after_complete"
                    self._window_deadline_ns = None
                    self._complete_pending_window(seal_after=True)
            else:
                self._seal_and_solve()
            return True
        except QuestPairedPoseCaptureTerminalError as error:
            self._terminal_error = str(error)
            self._emit_error("terminal capture operation failed", error)
            return True
        except QuestControllerSelectionUnavailableError as error:
            if normalized == "r":
                self._pending_action = "select_and_begin"
                self._waiting_selection_reason = str(error)
                self.emit(
                    "[Quest] r armed; waiting for a current real grip pose. "
                    "The 3s window will start automatically when tracking returns: "
                    f"{error}"
                )
                self._publish_status(
                    terminal_message=f"waiting for real grip tracking: {error}",
                    force=True,
                )
            else:
                self._emit_error("capture operation rejected", error)
            return True
        except (KeyError, RuntimeError, ValueError) as error:
            # Candidate ambiguity, a temporarily lost grip pose, and an
            # operator boundary rejection are retryable.  They must not be
            # reported as a terminal capture failure or latch the session.
            self._emit_error("capture operation rejected", error)
            return True
        except Exception as error:  # noqa: BLE001 - keep deployment loop alive.
            self._emit_error("capture operation failed", error)
            return True


class QuestPairedPoseCaptureFactory:
    """Single-use programmatic owner for one Quest host/application resource."""

    def __init__(
        self,
        config: QuestPairedPoseCaptureLaunchConfig,
        *,
        host_builder: Callable[..., QuestDevHost] = build_quest_dev_host,
    ) -> None:
        if not isinstance(config, QuestPairedPoseCaptureLaunchConfig):
            raise TypeError("config must be QuestPairedPoseCaptureLaunchConfig")
        if not callable(host_builder):
            raise TypeError("host_builder must be callable")
        self.config = config
        self.host_builder = host_builder
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._started = False
        self._starting = False
        self._cancel_startup = False
        self._application: QuestPairedPoseCaptureApplication | None = None
        self._launch_info: QuestDevLaunchInfo | None = None
        self._startup_error: BaseException | None = None
        self._startup_failure_pending = False
        self._owned_host: QuestDevHost | None = None
        self._owned_host_thread: threading.Thread | None = None
        self._thread_started = False
        self._host_error_box: dict[str, BaseException] = {}
        self._host_cleanup_failure: str | None = None
        self._closing = False
        self._closed = False
        self._last_close_error: str | None = None

    @property
    def application(self) -> QuestPairedPoseCaptureApplication | None:
        with self._lock:
            return self._application

    @property
    def launch_info(self) -> QuestDevLaunchInfo | None:
        with self._lock:
            return self._launch_info

    @property
    def cleanup_pending(self) -> bool:
        with self._lock:
            return self._startup_failure_pending or (
                self._startup_error is not None and not self._closed
            )

    def wait_for_application(
        self,
        *,
        timeout_s: float,
    ) -> QuestPairedPoseCaptureApplication:
        """Wait for programmatic operator access without exposing host internals."""

        timeout = _positive_timeout(timeout_s, name="timeout_s")
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._application is None and (
                self._startup_failure_pending
                or (self._startup_error is None and not self._closed)
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    if self._startup_failure_pending:
                        raise RuntimeError(
                            "Quest paired-pose startup cleanup is still pending"
                        ) from self._startup_error
                    raise TimeoutError("Quest paired-pose application was not ready")
                self._condition.wait(timeout=remaining)
            if self._startup_error is not None:
                raise RuntimeError(
                    "Quest paired-pose application startup failed"
                ) from (self._startup_error)
            if self._closed:
                raise RuntimeError("Quest paired-pose factory is closed")
            assert self._application is not None
            return self._application

    def _raise_startup_failure(self, error: BaseException) -> None:
        with self._condition:
            self._startup_error = error
            self._startup_failure_pending = True
            self._starting = False
            self._condition.notify_all()
        try:
            self.close()
        except BaseException as cleanup_error:
            combined = QuestPairedPoseCaptureStartupError(
                "Quest paired-pose startup failed and cleanup remains unproven: "
                f"{type(cleanup_error).__name__}:{cleanup_error}"
            )
            with self._condition:
                self._startup_error = combined
                self._startup_failure_pending = False
                self._condition.notify_all()
            if not isinstance(error, Exception):
                raise error from cleanup_error
            if not isinstance(cleanup_error, Exception):
                raise cleanup_error from error
            raise combined from error
        if not isinstance(error, Exception):
            with self._condition:
                self._startup_failure_pending = False
                self._condition.notify_all()
            raise error
        wrapped = QuestPairedPoseCaptureStartupError(
            "Quest paired-pose application startup failed"
        )
        with self._condition:
            self._startup_error = wrapped
            self._startup_failure_pending = False
            self._condition.notify_all()
        raise wrapped from error

    def close(self) -> None:
        """Close an application or retain failed-start ownership for retry."""

        with self._condition:
            waited_for_attempt = False
            while self._starting:
                self._cancel_startup = True
                waited_for_attempt = True
                self._condition.wait()
            while self._closing:
                waited_for_attempt = True
                self._condition.wait()
            if self._closed:
                return
            if waited_for_attempt and self._last_close_error is not None:
                raise QuestPairedPoseCaptureCleanupError(self._last_close_error)
            self._closing = True
            self._last_close_error = None
            application = self._application
            host = self._owned_host
            thread = self._owned_host_thread
            thread_started = self._thread_started or (
                thread is not None and thread.ident is not None
            )
            if not self._started and application is None and host is None:
                self._closed = True
                self._closing = False
                self._condition.notify_all()
                return

        errors: list[Exception] = []
        resources_closed = host is None
        try:
            if application is not None:
                try:
                    application.close()
                except Exception as exc:  # noqa: BLE001 - preserve retry owner.
                    errors.append(exc)
                else:
                    resources_closed = True
            elif host is not None:
                try:
                    host.request_shutdown()
                except Exception as exc:  # noqa: BLE001 - continue bounded cleanup.
                    errors.append(exc)
                thread_alive = False
                if thread is not None and thread_started:
                    if thread is threading.current_thread():
                        errors.append(
                            QuestPairedPoseCaptureCleanupError(
                                "factory cannot join its serving thread"
                            )
                        )
                        thread_alive = True
                    else:
                        thread.join(timeout=self.config.host_join_timeout_s)
                        thread_alive = thread.is_alive()
                        if thread_alive:
                            errors.append(
                                QuestPairedPoseCaptureCleanupError(
                                    "Quest host serving thread did not stop"
                                )
                            )
                if not thread_alive:
                    if self._host_cleanup_failure is not None:
                        errors.append(
                            QuestPairedPoseCaptureCleanupError(
                                self._host_cleanup_failure
                            )
                        )
                    else:
                        try:
                            host.close()
                        except BaseException as exc:
                            self._host_cleanup_failure = (
                                f"Quest host close failed:{type(exc).__name__}:{exc}"
                            )
                            if not isinstance(exc, Exception):
                                raise
                            errors.append(
                                QuestPairedPoseCaptureCleanupError(
                                    self._host_cleanup_failure
                                )
                            )
                        else:
                            resources_closed = True
        finally:
            with self._condition:
                if resources_closed:
                    self._owned_host = None
                    self._owned_host_thread = None
                    self._thread_started = False
                self._closed = resources_closed and not errors
                self._closing = False
                if errors:
                    self._last_close_error = "; ".join(
                        f"{type(error).__name__}:{error}" for error in errors
                    )
                self._condition.notify_all()
        if errors:
            assert self._last_close_error is not None
            raise QuestPairedPoseCaptureCleanupError(
                f"Quest paired-pose factory cleanup failed: {self._last_close_error}"
            ) from errors[0]

    def __call__(
        self,
        *,
        robot_pose_journal: PassiveRobotPoseJournalSource,
    ) -> QuestPairedPoseCaptureApplication:
        if not isinstance(robot_pose_journal, PassiveRobotPoseJournalSource):
            raise TypeError("robot_pose_journal must be PassiveRobotPoseJournalSource")
        with self._lock:
            if self._closed or self._closing:
                raise RuntimeError("Quest paired-pose factory is closed")
            if self._started:
                raise RuntimeError("Quest paired-pose factory is single-use")
            self._started = True
            self._starting = True

        try:
            return self._start_application(robot_pose_journal=robot_pose_journal)
        except BaseException as exc:  # noqa: BLE001 - transaction rollback boundary.
            self._raise_startup_failure(exc)
            raise AssertionError("unreachable")

    def _start_application(
        self,
        *,
        robot_pose_journal: PassiveRobotPoseJournalSource,
    ) -> QuestPairedPoseCaptureApplication:

        clock = robot_pose_journal.monotonic_ns_clock
        runtime_loader = build_quest_controller_runtime_loader(
            receiver_clock=robot_pose_journal.receiver_clock,
            monotonic_ns_clock=clock,
        )
        host_kwargs: dict[str, Any] = {
            "force_http": self.config.force_http,
            "monotonic_ns_clock": clock,
            "runtime_loader": runtime_loader,
        }
        if self.config.realman_left_ip is not None:
            host_kwargs.update(
                {
                    "realman_left_ip": self.config.realman_left_ip,
                    "realman_left_urdf": self.config.realman_left_urdf,
                }
            )
        host = self.host_builder(self.config.config_path, **host_kwargs)
        with self._condition:
            self._owned_host = host
            self._launch_info = host.launch_info
            self._host_error_box = {}
            startup_cancelled = self._cancel_startup
        if startup_cancelled:
            raise RuntimeError("Quest paired-pose startup was cancelled")
        ready = threading.Event()

        def on_ready() -> None:
            ready.set()

        def serve() -> None:
            try:
                host.serve_forever(ready_callback=on_ready)
            except BaseException as exc:  # noqa: BLE001 - startup rollback boundary.
                self._host_error_box.setdefault("error", exc)
                ready.set()

        thread = threading.Thread(
            target=serve,
            name="quest-paired-pose-host",
            daemon=False,
        )
        with self._condition:
            self._owned_host_thread = thread
            startup_cancelled = self._cancel_startup
        if startup_cancelled:
            raise RuntimeError("Quest paired-pose startup was cancelled")
        thread.start()
        with self._condition:
            self._thread_started = True
            startup_cancelled = self._cancel_startup
        if startup_cancelled:
            raise RuntimeError("Quest paired-pose startup was cancelled")
        if not ready.wait(timeout=self.config.startup_timeout_s):
            raise TimeoutError("Quest paired-pose host startup timed out")
        error = self._host_error_box.get("error")
        if error is not None:
            raise error
        runtime = host.require_running_feature_runtime(
            CONTROLLER_POSE_FEATURE_NAME,
            ControllerPoseTelemetryRuntime,
        )
        application = QuestPairedPoseCaptureApplication(
            runtime=runtime,
            robot_pose_journal=robot_pose_journal,
            profile=self.config.profile,
            capture_id=self.config.capture_id,
            callback_drain_timeout_s=self.config.callback_drain_timeout_s,
            owned_host=host,
            owned_host_thread=thread,
            host_join_timeout_s=self.config.host_join_timeout_s,
            host_error_box=self._host_error_box,
        )
        with self._condition:
            startup_cancelled = self._cancel_startup
            if not startup_cancelled:
                self._application = application
                self._starting = False
            self._condition.notify_all()
        if startup_cancelled:
            with self._condition:
                self._application = application
            raise RuntimeError("Quest paired-pose startup was cancelled")
        return application


__all__ = [
    "PassiveRobotPoseJournalSource",
    "QuestCaptureFrameFacts",
    "QuestCaptureSelectionIntent",
    "QuestControllerCandidate",
    "QuestControllerSelectionUnavailableError",
    "QuestControllerTrackingUnavailableError",
    "QuestPairedPoseCaptureApplication",
    "QuestPairedPoseCaptureApplicationPort",
    "QuestPairedPoseCaptureCleanupError",
    "QuestPairedPoseCaptureFactory",
    "QuestPairedPoseCaptureLaunchConfig",
    "QuestPairedPoseCaptureStartupError",
    "QuestPairedPoseCaptureTerminalError",
    "QuestTerminalCaptureController",
    "bind_passive_robot_pose_journal_source",
    "build_quest_controller_runtime_loader",
    "require_quest_paired_pose_capture_application",
]
