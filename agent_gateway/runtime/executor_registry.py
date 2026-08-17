"""Exact Route-to-executor bindings for one Agent Gateway session.

The registry is intentionally a selector only.  It owns no queue, safety
policy, HOLD authority, Station lifecycle, or lower dispatch.  Every
advertised Route must bind exactly once, including its immutable Definition,
Route, and deployment-binding hashes.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..protocol.contracts import AgentSkillRequest
from ..protocol.skill_catalog import SkillRouteManifest, SkillRouteManifestEntry
from .operation_port import AgentSkillExecutionPort


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot contain surrounding whitespace.")
    return value


@dataclass(frozen=True)
class SkillExecutorBinding:
    """One exact advertised Route/deployment binding and its execution port."""

    runtime_session_id: str
    skill_id: str
    skill_version: str
    skill_definition_hash: str
    route_id: str
    route_version: str
    route_hash: str
    deployment_binding_hash: str
    control_effect_class: str
    execution_port: AgentSkillExecutionPort

    def __post_init__(self) -> None:
        for field_name in (
            "runtime_session_id",
            "skill_id",
            "skill_version",
            "skill_definition_hash",
            "route_id",
            "route_version",
            "route_hash",
            "deployment_binding_hash",
            "control_effect_class",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        no_io = getattr(self.execution_port, "no_io", None)
        if not isinstance(no_io, bool):
            raise TypeError("execution_port must declare boolean no_io.")
        execution_effect = getattr(
            self.execution_port,
            "control_effect_class",
            None,
        )
        if not no_io and execution_effect != self.control_effect_class:
            raise ValueError(
                "Authoritative execution port control_effect_class does not "
                "match its Route."
            )
        if no_io and execution_effect not in {None, self.control_effect_class}:
            raise ValueError(
                "No-I/O execution port control_effect_class contradicts its Route."
            )
        _required(
            getattr(self.execution_port, "port_id", None),
            field_name="execution_port.port_id",
        )
        for method_name in ("activate", "cancel"):
            if not callable(getattr(self.execution_port, method_name, None)):
                raise TypeError(
                    "execution_port does not implement the typed execution "
                    f"contract: missing {method_name}()."
                )

    @classmethod
    def from_manifest_entry(
        cls,
        *,
        manifest: SkillRouteManifest,
        entry: SkillRouteManifestEntry,
        execution_port: AgentSkillExecutionPort,
    ) -> SkillExecutorBinding:
        if not isinstance(manifest, SkillRouteManifest):
            raise TypeError("manifest must be a SkillRouteManifest.")
        if not isinstance(entry, SkillRouteManifestEntry):
            raise TypeError("entry must be a SkillRouteManifestEntry.")
        route = entry.route
        return cls(
            runtime_session_id=manifest.runtime_session_id,
            skill_id=route.skill_id,
            skill_version=route.skill_version,
            skill_definition_hash=route.skill_definition_hash,
            route_id=route.route_id,
            route_version=route.route_version,
            route_hash=route.semantic_content_hash,
            deployment_binding_hash=(
                entry.deployment_binding.deployment_binding_hash
            ),
            control_effect_class=route.control_effect_class,
            execution_port=execution_port,
        )

    @property
    def request_key(self) -> tuple[str, ...]:
        return (
            self.runtime_session_id,
            self.skill_id,
            self.skill_version,
            self.skill_definition_hash,
            self.route_id,
            self.route_version,
            self.route_hash,
            self.deployment_binding_hash,
        )

    def validate_request(self, request: AgentSkillRequest) -> None:
        if not isinstance(request, AgentSkillRequest):
            raise TypeError("request must be an AgentSkillRequest.")
        if _request_key(request) != self.request_key:
            raise ValueError(
                "Agent Skill request does not match the exact executor binding."
            )


def _request_key(request: AgentSkillRequest) -> tuple[str, ...]:
    return (
        request.runtime_session_id,
        request.skill_id,
        request.skill_version,
        request.skill_definition_hash,
        request.route_id,
        request.route_version,
        request.route_hash,
        request.deployment_binding_hash,
    )


class SkillExecutorRegistry:
    """Fail-closed exact executor selector for all advertised Routes."""

    def __init__(
        self,
        *,
        manifest: SkillRouteManifest,
        bindings: tuple[SkillExecutorBinding, ...],
    ) -> None:
        if not isinstance(manifest, SkillRouteManifest):
            raise TypeError("manifest must be a SkillRouteManifest.")
        normalized = tuple(bindings)
        if any(not isinstance(item, SkillExecutorBinding) for item in normalized):
            raise TypeError("bindings must contain SkillExecutorBinding values.")

        expected: dict[tuple[str, ...], SkillRouteManifestEntry] = {}
        for entry in manifest.skill_routes:
            route = entry.route
            key = (
                manifest.runtime_session_id,
                route.skill_id,
                route.skill_version,
                route.skill_definition_hash,
                route.route_id,
                route.route_version,
                route.semantic_content_hash,
                entry.deployment_binding.deployment_binding_hash,
            )
            expected[key] = entry

        selected: dict[tuple[str, ...], SkillExecutorBinding] = {}
        for binding in normalized:
            entry = expected.get(binding.request_key)
            if entry is None:
                raise ValueError(
                    "Executor binding does not close over an advertised exact Route."
                )
            if binding.control_effect_class != entry.route.control_effect_class:
                raise ValueError(
                    "Executor binding control_effect_class does not match Route."
                )
            if binding.request_key in selected:
                raise ValueError("Advertised exact Route has multiple executors.")
            selected[binding.request_key] = binding
        missing = tuple(key for key in expected if key not in selected)
        if missing:
            raise ValueError("Every advertised exact Route requires one executor.")

        self.manifest = manifest
        self._bindings = normalized
        self._by_request_key = selected

    @classmethod
    def bind_all_routes_to_port(
        cls,
        *,
        manifest: SkillRouteManifest,
        execution_port: AgentSkillExecutionPort,
    ) -> SkillExecutorRegistry:
        """Compatibility adapter for the legacy one-port Gateway constructor."""

        return cls(
            manifest=manifest,
            bindings=tuple(
                SkillExecutorBinding.from_manifest_entry(
                    manifest=manifest,
                    entry=entry,
                    execution_port=execution_port,
                )
                for entry in manifest.skill_routes
            ),
        )

    @property
    def bindings(self) -> tuple[SkillExecutorBinding, ...]:
        return self._bindings

    @property
    def all_no_io(self) -> bool:
        return all(binding.execution_port.no_io for binding in self._bindings)

    @property
    def has_authoritative_execution(self) -> bool:
        return any(not binding.execution_port.no_io for binding in self._bindings)

    @property
    def single_execution_port(self) -> AgentSkillExecutionPort | None:
        ports = {id(binding.execution_port): binding.execution_port for binding in self._bindings}
        if len(ports) != 1:
            return None
        return next(iter(ports.values()))

    def binding_for_request(self, request: AgentSkillRequest) -> SkillExecutorBinding:
        if not isinstance(request, AgentSkillRequest):
            raise TypeError("request must be an AgentSkillRequest.")
        try:
            binding = self._by_request_key[_request_key(request)]
        except KeyError as error:
            raise KeyError(
                "No executor is bound to the request's exact Route/deployment."
            ) from error
        binding.validate_request(request)
        return binding

    def execution_port_for_request(
        self,
        request: AgentSkillRequest,
    ) -> AgentSkillExecutionPort:
        return self.binding_for_request(request).execution_port


__all__ = ["SkillExecutorBinding", "SkillExecutorRegistry"]
