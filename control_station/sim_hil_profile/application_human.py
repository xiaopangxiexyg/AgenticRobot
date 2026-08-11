"""Human-source attachment, acceptance, and runtime-Home lifecycle.

This mixin coordinates an already-selected passive human source. Concrete
device production remains in teleop_runtime and target execution remains in
the lower runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .human import _SimHILHumanAttachment
from .status import _vive_receiver_status
from .validation import _finite


class _SimHILApplicationHumanMixin:
    def prepare_human_attachment(
        self,
        *,
        coordinate_profile_path: Path,
        source_role: str,
    ) -> _SimHILHumanAttachment:
        """Build a passive human source without mutating the live runtime."""

        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot attach human input to a closed runtime.")
            if self.human_input_attached:
                raise RuntimeError("Human input is already attached.")
            factory = self._human_attachment_factory
            if factory is None:
                raise RuntimeError("Human input attachment is not configured.")
        return factory(
            Path(coordinate_profile_path).expanduser(),
            str(source_role),
        )

    def activate_human_attachment(
        self,
        attachment: _SimHILHumanAttachment,
    ) -> dict[str, object]:
        """Commit a prepared passive source while the execution gate is held."""

        if not isinstance(attachment, _SimHILHumanAttachment):
            raise TypeError("attachment must be a _SimHILHumanAttachment.")
        with self._lock:
            if not self._started or not self._held:
                raise RuntimeError(
                    "Human input attachment requires a started held runtime."
                )
            receipt = self.human_source.attach(attachment.source)
            self.gripper_input_selection = (
                attachment.gripper_input_selection
            )
            self.calibration_binding = dict(
                attachment.calibration_binding
            )
            self._runtime_home_pose_desk_by_side_capture = (
                attachment.runtime_home_pose_desk_by_side_capture
            )
            self.readiness_provider.calibration_ready = True
            self.readiness_provider.home_ready = False
            return {
                **dict(receipt),
                "calibration_binding": dict(self.calibration_binding),
            }

    def swap_vive_left_right_sources(self) -> dict[str, object]:
        """Swap upstream VIVE wrist roles before composition starts."""

        with self._lock:
            if self._started:
                raise RuntimeError(
                    "VIVE source roles cannot be swapped after Sim-HIL "
                    "composition has started."
                )
            if self._human_input_check_active:
                raise RuntimeError(
                    "VIVE source roles cannot be swapped while passive input "
                    "acceptance is running."
                )
            session = getattr(self.human_source, "session", None)
            if session is None:
                raise RuntimeError(
                    "Passive human source does not expose its teleop session."
                )
            from teleop_runtime.sources.pose.vive import VivePoseSource

            sources = [
                source
                for source in tuple(
                    getattr(session, "sources", ()) or ()
                )
                if isinstance(source, VivePoseSource)
            ]
            if len(sources) != 1:
                raise RuntimeError(
                    "Expected exactly one VivePoseSource in the passive "
                    f"human session; found {len(sources)}."
                )
            source = sources[0]
            receipt = dict(source.swap_left_right_source_roles())
            self._vive_source_swap_count += 1
            self._last_human_input_check_step = None
            self._human_input_check_report = None
            self._last_error = None
            receipt.update(
                {
                    "swapped_at": _finite(
                        self.clock(),
                        name="vive_source_swap.swapped_at",
                    ),
                    "swap_count": self._vive_source_swap_count,
                }
            )
            return receipt

    def run_human_input_acceptance_check(
        self,
        *,
        timeout_sec: float | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> dict[str, object]:
        """Start only the passive human source and wait for one valid emission.

        This is an input-device acceptance step, not target authorization or
        simulator dispatch. It may run before startup or while a started
        runtime is explicitly held for deferred human attachment.
        """

        with self._lock:
            if self._closed:
                raise RuntimeError(
                    "Human input acceptance requires a newly built Sim-HIL "
                    "application; this instance is closed."
                )
            if self._started and not self._held:
                raise RuntimeError(
                    "Hold a started Sim-HIL runtime before checking human input."
                )
            if not self.human_input_attached:
                raise RuntimeError(
                    "Human input acceptance requires an attached human source."
                )
            if self._human_input_check_active:
                raise RuntimeError(
                    "Human input acceptance is already running."
                )
            timeout = (
                self.config.warmup_timeout_sec
                if timeout_sec is None
                else _finite(timeout_sec, name="input_check.timeout_sec")
            )
            if timeout <= 0.0:
                raise ValueError("input acceptance timeout must be positive.")
            self._human_input_check_active = True

        try:
            self.human_source.start()
            # Preserve cleanup ownership once lower resources are live.
            with self._lock:
                self._sources_started = True
            self.human_source.set_calibration_ready()
            deadline = self.clock() + timeout
            while True:
                self._raise_if_cancelled(
                    cancel_requested,
                    phase="human_input_readiness",
                )
                now = _finite(self.clock(), name="input_check.now")
                step = self.human_source.step(now=now)
                with self._lock:
                    self._last_human_input_check_step = step
                emission = getattr(step, "emission", None)
                emission_valid = bool(
                    getattr(step, "emission_valid", False)
                )
                if bool(getattr(step, "command_dispatched", False)):
                    raise RuntimeError(
                        "Passive human input check attempted dispatch."
                    )
                if bool(getattr(step, "authoritative", False)):
                    raise RuntimeError(
                        "Passive human input check became authoritative."
                    )
                if emission_valid and emission is not None:
                    session_step = getattr(step, "session_step", None)
                    channels = dict(
                        getattr(session_step, "channels", {}) or {}
                    )
                    report = {
                        "status": "ready",
                        "checked_at": now,
                        "emission_id": getattr(
                            emission,
                            "emission_id",
                            None,
                        ),
                        "emission_hash": getattr(
                            emission,
                            "semantic_content_hash",
                            None,
                        ),
                        "channel_names": sorted(channels),
                        "resolved_gripper_input": getattr(
                            self.gripper_input_selection,
                            "resolved_source",
                            None,
                        ),
                        "q6_port": getattr(
                            self.gripper_input_selection,
                            "q6_port",
                            None,
                        ),
                        "fallback_reason": getattr(
                            self.gripper_input_selection,
                            "fallback_reason",
                            None,
                        ),
                        "rig_absent": bool(
                            getattr(
                                getattr(
                                    self.human_source,
                                    "binding_receipt",
                                    None,
                                ),
                                "rig_absent",
                                False,
                            )
                        ),
                        "command_dispatched": False,
                        "authoritative": False,
                    }
                    with self._lock:
                        self._human_input_check_report = report
                        self._last_error = None
                    return dict(report)
                if now >= deadline:
                    rejection_reason = getattr(
                        step,
                        "rejection_reason",
                        None,
                    )
                    session_step = getattr(step, "session_step", None)
                    channels = dict(
                        getattr(session_step, "channels", {}) or {}
                    )
                    expected_channels: set[str] = set()
                    session = getattr(self.human_source, "session", None)
                    for source in tuple(
                        getattr(session, "sources", ()) or ()
                    ):
                        expected = getattr(
                            source,
                            "expected_channel_names",
                            None,
                        )
                        if callable(expected):
                            expected_channels.update(expected())
                    missing_channels = sorted(
                        expected_channels - set(channels)
                    )
                    invalid_channels = sorted(
                        name
                        for name, channel in channels.items()
                        if not bool(getattr(channel, "valid", False))
                    )
                    fault_codes = sorted(
                        {
                            str(getattr(fault, "code", "unknown"))
                            for fault in tuple(
                                getattr(session_step, "faults", ()) or ()
                            )
                        }
                    )
                    vive_status = _vive_receiver_status(session)
                    vive_trackers = dict(
                        vive_status.get("trackers") or {}
                    )
                    vive_role_facts = {
                        side: {
                            "status": tracker.get("status_level"),
                            "valid": tracker.get("valid"),
                            "age_sec": tracker.get("age_sec"),
                            "seq": tracker.get("seq"),
                            "error": tracker.get("sender_error"),
                            "selected_outputs": tracker.get(
                                "selected_for_output_channels"
                            ),
                        }
                        for side, tracker in vive_trackers.items()
                    }
                    raise TimeoutError(
                        "Passive human input acceptance timed out: "
                        f"rejection={rejection_reason or 'none'}; "
                        f"missing_channels={missing_channels}; "
                        f"invalid_channels={invalid_channels}; "
                        f"available_channels={sorted(channels)}; "
                        f"fault_codes={fault_codes}; "
                        f"vive_receiver={vive_status.get('status')}; "
                        f"vive_packet_count={vive_status.get('packet_count')}; "
                        f"vive_packet_age_sec={vive_status.get('packet_age_sec')}; "
                        f"vive_roles={vive_role_facts}"
                    )
                self.sleep(0.005)
        except Exception as error:
            with self._lock:
                self._last_error = f"{type(error).__name__}:{error}"
            raise
        finally:
            with self._lock:
                self._human_input_check_active = False

    @staticmethod
    def _channel_pose_in_desk(
        channel: Any,
        calibration: Any,
    ) -> Any:
        from teleop_runtime.transforms import as_pose_matrix

        pose = as_pose_matrix(channel.value)
        frame_name = str(getattr(channel, "frame_name", "") or "")
        metadata = dict(getattr(channel, "metadata", {}) or {})
        tracker_role = metadata.get("tracker_role")
        if frame_name in {"vive_world", "world"}:
            return calibration.world_to_desk(
                pose,
                tracker_role=(
                    None if tracker_role is None else str(tracker_role)
                ),
            )
        if frame_name.endswith("_operator") or frame_name == "operator":
            return calibration.operator_to_desk(pose)
        if frame_name in {calibration.desk_frame.name, "desk"}:
            return pose
        raise ValueError(
            f"Unsupported runtime-Home tracker frame {frame_name or '<none>'!r}."
        )

    def capture_runtime_home(
        self,
        *,
        samples_required: int = 5,
        timeout_sec: float = 1.0,
    ) -> dict[str, object]:
        """Re-anchor live human pose to current MuJoCo pose without dispatch."""

        sample_count = int(samples_required)
        timeout = _finite(timeout_sec, name="runtime_home.timeout_sec")
        if sample_count <= 0:
            raise ValueError("samples_required must be positive.")
        if timeout <= 0.0:
            raise ValueError("runtime Home timeout must be positive.")
        with self._lock:
            if self._closed:
                raise RuntimeError("Runtime Home requires an open application.")
            if not self._sources_started:
                raise RuntimeError(
                    "Runtime Home requires passive human input acceptance first."
                )
            if self._episode_ended:
                raise RuntimeError(
                    "Runtime Home cannot mutate an ended Sim-HIL episode."
                )
            if self._started and not self._held:
                raise RuntimeError(
                    "Hold Sim-HIL before recapturing runtime Home."
                )
            capture_robot_home = (
                self._runtime_home_pose_desk_by_side_capture
            )
            if capture_robot_home is None:
                raise RuntimeError(
                    "Runtime Home robot-pose capture is not bound."
                )
            session = getattr(self.human_source, "session", None)
            calibration = getattr(session, "calibration", None)
            retargeters = tuple(
                getattr(session, "retargeters", ()) or ()
            )
            if session is None or calibration is None:
                raise RuntimeError(
                    "Passive human source does not expose its calibration session."
                )

        pose_retargeters: dict[str, Any] = {}
        for retargeter in retargeters:
            config = getattr(retargeter, "config", None)
            side = str(getattr(config, "side", "") or "").strip()
            if (
                side
                and callable(getattr(retargeter, "set_home_anchors", None))
                and str(
                    getattr(config, "input_channel_name", "") or ""
                ).strip()
            ):
                if side in pose_retargeters:
                    raise RuntimeError(
                        f"Multiple pose retargeters expose side {side!r}."
                    )
                pose_retargeters[side] = retargeter
        active_sides = tuple(sorted(pose_retargeters))
        if not active_sides:
            raise RuntimeError(
                "Runtime Home has no pose retargeter."
            )

        samples: dict[str, list[Any]] = {
            side: [] for side in active_sides
        }
        deadline = self.clock() + timeout
        last_step = None
        while any(len(values) < sample_count for values in samples.values()):
            now = _finite(self.clock(), name="runtime_home.now")
            if now >= deadline:
                counts = {
                    side: len(values) for side, values in samples.items()
                }
                raise TimeoutError(
                    "Timed out waiting for runtime-Home VIVE samples: "
                    f"{counts}"
                )
            step = self.human_source.step(now=now)
            if bool(getattr(step, "command_dispatched", False)) or bool(
                getattr(step, "authoritative", False)
            ):
                raise RuntimeError(
                    "Runtime Home passive input unexpectedly gained dispatch "
                    "or authority."
                )
            last_step = step
            channels = dict(
                getattr(getattr(step, "session_step", None), "channels", {})
                or {}
            )
            for side in active_sides:
                if len(samples[side]) >= sample_count:
                    continue
                retargeter = pose_retargeters[side]
                channel_name = str(
                    retargeter.config.input_channel_name
                )
                channel = channels.get(channel_name)
                if channel is None or not bool(
                    getattr(channel, "valid", False)
                ):
                    continue
                pose_desk = self._channel_pose_in_desk(
                    channel,
                    calibration,
                )
                # ``set_home_anchors`` owns the tracker-local translation and
                # rotation adjustment.  Keep capture samples in the raw desk
                # frame so that adjustment is applied exactly once.
                samples[side].append(pose_desk)
            self.sleep(0.005)

        from teleop_runtime.calibration_flow import average_pose

        human_home_by_side = {
            side: average_pose(samples[side]) for side in active_sides
        }
        # Capture the robot pose after the sample interval so a preceding HOLD
        # has time to reach the physics owner. The lower rig serializes this
        # read against mj_step.
        robot_home_by_side = {
            str(side): pose
            for side, pose in dict(capture_robot_home()).items()
        }
        missing_robot_sides = set(active_sides) - set(robot_home_by_side)
        if missing_robot_sides:
            raise RuntimeError(
                "Runtime Home robot-pose capture is missing sides: "
                f"{sorted(missing_robot_sides)}"
            )
        for side in active_sides:
            pose_retargeters[side].set_home_anchors(
                human_pose_desk=human_home_by_side[side],
                robot_pose_desk=robot_home_by_side[side],
            )
        session.mark_home_recalibrated(
            reason="sim_hil_runtime_home_captured"
        )
        captured_at = _finite(
            self.clock(),
            name="runtime_home.captured_at",
        )
        receipt: dict[str, object] = {
            "capture_id": f"sim-hil-runtime-home:{captured_at:.9f}",
            "captured_at": captured_at,
            "clock_domain_id": getattr(
                getattr(self.runtime, "config", None),
                "clock_domain_id",
                None,
            ),
            "sample_count_by_side": {
                side: len(samples[side]) for side in active_sides
            },
            "human_home_pose_desk_by_side": {
                side: (
                    human_home_by_side[side].tolist()
                    if callable(
                        getattr(human_home_by_side[side], "tolist", None)
                    )
                    else human_home_by_side[side]
                )
                for side in active_sides
            },
            "robot_home_pose_desk_by_side": {
                side: (
                    robot_home_by_side[side].tolist()
                    if callable(
                        getattr(robot_home_by_side[side], "tolist", None)
                    )
                    else robot_home_by_side[side]
                )
                for side in active_sides
            },
            "command_dispatched": False,
            "authoritative": False,
        }
        with self._lock:
            self._last_human_input_check_step = last_step
            self._runtime_home_capture = receipt
            self.readiness_provider.home_ready = True
            self._last_passive_observation_error = None
        return dict(receipt)

    def poll_passive_human_input(
        self,
        *,
        now: float | None = None,
    ) -> Any:
        """Refresh human observation while HOLD without composing or dispatching."""

        with self._lock:
            if (
                self._closed
                or not self._sources_started
                or not self._started
                or not self._held
                or self._episode_ended
            ):
                return None
            timestamp = _finite(
                self.clock() if now is None else now,
                name="passive_observation.now",
            )
            try:
                step = self.human_source.step(now=timestamp)
                if bool(getattr(step, "command_dispatched", False)) or bool(
                    getattr(step, "authoritative", False)
                ):
                    raise RuntimeError(
                        "Held passive observation unexpectedly gained dispatch "
                        "or authority."
                    )
                self._last_human_input_check_step = step
                self._last_passive_observation_error = None
                return step
            except Exception as error:
                self._last_passive_observation_error = (
                    f"{type(error).__name__}:{error}"
                )
                raise
