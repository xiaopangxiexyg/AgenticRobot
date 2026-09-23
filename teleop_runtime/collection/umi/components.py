"""Backend-neutral runtime bundle for the arm-free UMI collector.

Concrete VIVE, RealSense, AngleTrigger and DMTac construction lives in
``default_components``.  Keeping this lifecycle bundle dependent only on
runtime ports lets another hardware factory provide the same UMI sample
semantics without changing the recorder or persisted dataset contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Mapping, Protocol

from robot_motion.runtime.lifecycle import run_best_effort_cleanup

from ..sample_assembler import CameraFrameMatch
from .config import UmiCollectionConfig
from .topology import DEFAULT_COMMAND_CHANNEL_BY_SIDE


class UmiStartupProgress(Protocol):
    def begin(
        self,
        key: str,
        *,
        label: str,
        message: str,
        requirement: str = "feature_dependent",
    ) -> None: ...

    def succeed(self, key: str, *, message: str = "连接完成") -> None: ...

    def fail(self, key: str, error: str, *, message: str = "连接失败") -> None: ...


class UmiComponentBuilder(Protocol):
    """Build source/control ports without owning dataset or GUI state."""

    def __call__(
        self,
        config: UmiCollectionConfig,
        *,
        prepared_hardware: Any = None,
    ) -> "UmiComponents": ...


class UmiHardwarePreparer(Protocol):
    """Resolve external device bindings without starting runtime resources."""

    def __call__(self, config: UmiCollectionConfig) -> Any: ...


@dataclass(frozen=True)
class UmiHardwarePresentation:
    """Default text/status projection for one UMI hardware family."""

    preflight_key: str
    preflight_label: str
    preflight_message: str
    preflight_success_message: str
    component_build_message: str
    component_control_hint: str
    recovery_error_formatter: Callable[[Exception], str] = str

    def __post_init__(self) -> None:
        for name in (
            "preflight_key",
            "preflight_label",
            "preflight_message",
            "preflight_success_message",
            "component_build_message",
            "component_control_hint",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"UMI hardware presentation {name} must be non-empty")

    def recovery_message(self, error: Exception) -> str:
        return str(self.recovery_error_formatter(error))


@dataclass(frozen=True)
class UmiStartupResource:
    """One ordered UMI runtime resource and its presentation contract."""

    key: str
    component: Any
    label: str
    requirement: str
    strict_start_message: str
    capability_success_message: str = "连接完成"
    strict_success_message: str = "连接完成"
    cleanup_name: str = ""
    require_end_effector_readiness: bool = False
    start_action: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        for name in (
            "key",
            "label",
            "requirement",
            "strict_start_message",
            "capability_success_message",
            "strict_success_message",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"UMI startup resource {name} must be non-empty")
        if not callable(getattr(self.component, "start", None)):
            raise TypeError(f"UMI startup resource {self.key!r} has no start()")
        if not callable(getattr(self.component, "close", None)):
            raise TypeError(f"UMI startup resource {self.key!r} has no close()")
        if self.start_action is not None and not callable(self.start_action):
            raise TypeError(f"UMI startup resource {self.key!r} start_action must be callable")
        if not self.cleanup_name:
            object.__setattr__(self, "cleanup_name", self.key)


class UmiEndEffectorDiagnosticsProvider(Protocol):
    """Project one concrete end effector into GUI-neutral diagnostic fields."""

    def __call__(
        self,
        side: str,
        backend: Any,
        config: UmiCollectionConfig,
    ) -> Mapping[str, Any]: ...


@dataclass
class UmiComponents:
    # Compatibility field names are retained because existing UMI tests and
    # external launch glue may still construct this bundle directly.  New
    # callers use the backend-neutral properties below.
    vive: Any
    cameras: dict[str, Any]
    angle_trigger: Any
    grippers: dict[str, Any]
    rig: Any
    retargeters: dict[str, Any]
    hand_state_poller: Any
    foot_pad: Any | None = None
    foot_pad_error: str = ""
    strict_startup: bool = False
    startup_errors: dict[str, str] = field(default_factory=dict)
    end_effector_readiness_check: Callable[[], None] | None = None
    startup_resources: tuple[UmiStartupResource, ...] = ()
    end_effector_diagnostics_provider: UmiEndEffectorDiagnosticsProvider | None = None
    command_channel_by_side: Mapping[str, str] = field(
        default_factory=lambda: dict(DEFAULT_COMMAND_CHANNEL_BY_SIDE)
    )
    command_input_name: str = "command_input"
    _started: bool = False

    @property
    def pose_source(self) -> Any:
        return self.vive

    @property
    def camera_sources(self) -> Mapping[str, Any]:
        return self.cameras

    @property
    def command_input(self) -> Any:
        return self.angle_trigger

    @property
    def end_effectors(self) -> Mapping[str, Any]:
        return self.grippers

    @property
    def command_rig(self) -> Any:
        return self.rig

    @property
    def command_retargeters(self) -> Mapping[str, Any]:
        return self.retargeters

    @property
    def state_history(self) -> Any:
        return self.hand_state_poller

    def _require_backends_ready(self, *, label: str) -> None:
        unavailable = []
        for side, backend in sorted(self.grippers.items()):
            ready = getattr(backend, "is_ready", None)
            if not callable(ready):
                continue
            if ready():
                continue
            get_state = getattr(backend, "get_state", None)
            state: Mapping[str, Any] = {}
            if callable(get_state):
                try:
                    state = get_state(include_tactile_arrays=False)
                except TypeError:
                    state = get_state()
            detail = str(state.get("telemetry_error") or "backend is not ready")
            unavailable.append(f"{side}={detail}")
        if unavailable:
            raise RuntimeError(f"{label} startup incomplete: " + "; ".join(unavailable))

    def _require_dmtac_ready(self) -> None:
        """Compatibility readiness check for the old private test seam."""

        self._require_backends_ready(label="DMTAC")

    def _require_end_effectors_ready(self) -> None:
        if self.end_effector_readiness_check is not None:
            self.end_effector_readiness_check()
            return
        self._require_backends_ready(label="end-effector")

    def _default_startup_resources(self) -> tuple[UmiStartupResource, ...]:
        return (
            UmiStartupResource(
                key="end_effector.rig",
                component=self.rig,
                label="末端执行器 Rig",
                requirement="collection_core",
                strict_start_message="正在启动末端执行器 Rig",
                cleanup_name="rig",
                require_end_effector_readiness=True,
            ),
            UmiStartupResource(
                key="end_effector.state_history",
                component=self.hand_state_poller,
                label="末端执行器状态采样",
                requirement="collection_core",
                strict_start_message="正在启动状态采样",
                cleanup_name="hand_state_poller",
            ),
            UmiStartupResource(
                key="pose_source",
                component=self.vive,
                label="位姿输入",
                requirement="collection_core",
                strict_start_message="正在启动位姿输入",
                cleanup_name="vive",
            ),
            *(
                UmiStartupResource(
                    key=f"camera.{key}",
                    component=camera,
                    label=f"腕部相机 {key}",
                    requirement="collection",
                    strict_start_message="正在启动视频流",
                    strict_success_message="视频流已启动",
                    cleanup_name=f"camera[{key}]",
                )
                for key, camera in self.cameras.items()
            ),
            UmiStartupResource(
                key="command_input",
                component=self.angle_trigger,
                label="末端命令输入",
                requirement="collection_core",
                strict_start_message="正在启动命令输入",
                cleanup_name="angle_trigger",
            ),
        )

    def ordered_startup_resources(self) -> tuple[UmiStartupResource, ...]:
        resources = (
            tuple(self.startup_resources)
            if self.startup_resources
            else self._default_startup_resources()
        )
        keys = tuple(resource.key for resource in resources)
        if len(keys) != len(set(keys)):
            raise ValueError("UMI startup resource keys must be unique")
        return resources

    def hardware_status(self) -> dict[str, dict[str, Any]]:
        errors = dict(self.startup_errors)
        return {
            resource.key: {
                "online": resource.key not in errors,
                "health": "online" if resource.key not in errors else "error",
                "status_level": "ok" if resource.key not in errors else "error",
                "status_label": "在线" if resource.key not in errors else "连接失败",
                "requirement": resource.requirement,
                "error": errors.get(resource.key, ""),
            }
            for resource in self.ordered_startup_resources()
        }

    def end_effector_diagnostics(
        self,
        side: str,
        config: UmiCollectionConfig,
    ) -> dict[str, Any]:
        backend = self.grippers[str(side)]
        if self.end_effector_diagnostics_provider is not None:
            return dict(
                self.end_effector_diagnostics_provider(str(side), backend, config)
            )
        ready = getattr(backend, "is_ready", None)
        return {
            "configured": True,
            "is_connected": bool(ready()) if callable(ready) else True,
        }

    def command_channel_name(self, side: str) -> str:
        try:
            name = str(self.command_channel_by_side[str(side)]).strip()
        except KeyError as exc:
            raise KeyError(f"UMI command channel missing for side {side!r}") from exc
        if not name:
            raise ValueError(f"UMI command channel for {side!r} must be non-empty")
        return name

    def start(self, *, startup_progress: UmiStartupProgress | None = None) -> None:
        if self._started:
            return
        resources = self.ordered_startup_resources()
        if not self.strict_startup:
            self.startup_errors.clear()
            for resource in resources:
                if startup_progress is not None:
                    startup_progress.begin(
                        resource.key,
                        label=resource.label,
                        message="正在调用设备 start()",
                        requirement=resource.requirement,
                    )
                try:
                    (resource.start_action or resource.component.start)()
                    if resource.require_end_effector_readiness:
                        self._require_end_effectors_ready()
                except Exception as exc:  # noqa: BLE001 - report each independent device.
                    detail = f"{type(exc).__name__}: {exc}"
                    self.startup_errors[resource.key] = detail
                    if startup_progress is not None:
                        startup_progress.fail(resource.key, detail)
                    if resource.key.startswith("camera.") and isinstance(exc, TimeoutError):
                        # A native camera start may still hold the process-wide
                        # SDK lifecycle lock. Do not open the next device.
                        raise
                else:
                    if startup_progress is not None:
                        startup_progress.succeed(
                            resource.key,
                            message=resource.capability_success_message,
                        )
            self._started = True
            return
        started: list[UmiStartupResource] = []
        current_name = resources[0].key if resources else "umi.components"
        try:
            for resource in resources:
                current_name = resource.key
                if startup_progress is not None:
                    startup_progress.begin(
                        resource.key,
                        label=resource.label,
                        message=resource.strict_start_message,
                        requirement=resource.requirement,
                    )
                (resource.start_action or resource.component.start)()
                started.append(resource)
                if resource.require_end_effector_readiness:
                    self._require_end_effectors_ready()
                if startup_progress is not None:
                    startup_progress.succeed(
                        resource.key,
                        message=resource.strict_success_message,
                    )
        except BaseException as exc:
            if startup_progress is not None:
                startup_progress.fail(
                    current_name,
                    f"{type(exc).__name__}: {exc}",
                )
            run_best_effort_cleanup(
                "umi_components.start.rollback",
                (
                    (resource.cleanup_name, resource.component.close)
                    for resource in reversed(started)
                ),
            )
            raise
        self._started = True

    def close(self) -> None:
        steps = [
            (resource.cleanup_name, resource.component.close)
            for resource in reversed(self.ordered_startup_resources())
        ]
        if self.foot_pad is not None:
            steps.append(("foot_pad", self.foot_pad.close))
        run_best_effort_cleanup("umi_components.close", steps)
        self._started = False

    def camera_selectors(self) -> dict[str, Any]:
        def select(camera: Any, timestamp: float) -> CameraFrameMatch | None:
            frame = camera.frame_buffer().get_latest_before_or_at(timestamp)
            if frame is None:
                return None
            return CameraFrameMatch(
                camera_key=camera.camera_key,
                frame=frame,
                selected_timestamp=float(frame.source_timestamp),
                age_sec=float(timestamp - frame.source_timestamp),
            )

        return {
            key: partial(select, camera)
            for key, camera in self.cameras.items()
        }


def build_umi_components(
    config: UmiCollectionConfig,
    *,
    dmtac_assignment: dict[str, Any] | None = None,
) -> UmiComponents:
    """Compatibility wrapper for the default concrete hardware factory."""

    from .default_components import build_default_umi_components

    return build_default_umi_components(
        config,
        dmtac_assignment=dmtac_assignment,
    )


def _configure_dmtac_backend_configs(
    config: UmiCollectionConfig,
    backend_configs: dict[str, Any],
) -> None:
    """Compatibility wrapper for the old private import path."""

    from .default_components import configure_dmtac_backend_configs

    configure_dmtac_backend_configs(config, backend_configs)


__all__ = [
    "UmiComponentBuilder",
    "UmiComponents",
    "UmiEndEffectorDiagnosticsProvider",
    "UmiHardwarePreparer",
    "UmiHardwarePresentation",
    "UmiStartupResource",
    "build_umi_components",
]
