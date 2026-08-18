"""Project profile bindings and the existing active operation as read-only views."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..protocol._contract_validation import (
    enum_value,
    identifier,
    unique_sorted_text,
)
from ..protocol.hardware_views import (
    HARDWARE_HEALTH_STATES_V1,
    HardwareResourceView,
    HardwareSkillRef,
    HardwareStatusView,
)
from ..protocol.skill_catalog import SkillExecutionRoute, SkillRouteManifest
from .operation_store import BoundedInMemoryAgentSkillOperationStore


@dataclass(frozen=True)
class HardwareResourceBinding:
    """Trusted profile mapping; it owns no device, lock, lease, or admission."""

    resource_id: str
    resource_kind: str
    route_hashes: tuple[str, ...]
    health_provider: Callable[[], str]
    occupied_provider: Callable[[], bool]
    occupies_on_read: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resource_id",
            identifier(self.resource_id, field_name="resource_id"),
        )
        object.__setattr__(
            self,
            "resource_kind",
            identifier(self.resource_kind, field_name="resource_kind"),
        )
        object.__setattr__(
            self,
            "route_hashes",
            unique_sorted_text(
                self.route_hashes,
                field_name="route_hashes",
                allow_empty=False,
            ),
        )
        if not callable(self.health_provider):
            raise TypeError("health_provider must be callable.")
        if not callable(self.occupied_provider):
            raise TypeError("occupied_provider must be callable.")
        if not isinstance(self.occupies_on_read, bool):
            raise TypeError("occupies_on_read must be a boolean.")


def _skill_ref(route: SkillExecutionRoute) -> HardwareSkillRef:
    return HardwareSkillRef(
        route.skill_id,
        route.skill_version,
    )


class HardwareViewProvider:
    def __init__(
        self,
        *,
        manifest: SkillRouteManifest,
        operation_store: BoundedInMemoryAgentSkillOperationStore,
        bindings: tuple[HardwareResourceBinding, ...],
    ) -> None:
        if not isinstance(manifest, SkillRouteManifest):
            raise TypeError("manifest must be a SkillRouteManifest.")
        if not isinstance(operation_store, BoundedInMemoryAgentSkillOperationStore):
            raise TypeError("operation_store must be typed.")
        bindings = tuple(bindings)
        if not bindings or any(
            not isinstance(item, HardwareResourceBinding) for item in bindings
        ):
            raise ValueError("bindings must contain HardwareResourceBinding values.")
        ids = tuple(item.resource_id for item in bindings)
        if len(set(ids)) != len(ids):
            raise ValueError("Hardware resource IDs must be unique.")

        routes = {
            entry.route.semantic_content_hash: entry.route
            for entry in manifest.skill_routes
        }
        for binding in bindings:
            if set(binding.route_hashes) - set(routes):
                raise ValueError(
                    f"Hardware binding {binding.resource_id!r} has unknown Routes."
                )
            if not any(
                routes[value].control_effect_class in {"read_only", "motion"}
                for value in binding.route_hashes
            ):
                raise ValueError("Lifecycle-only resources are not hardware views.")

        self.manifest = manifest
        self.operation_store = operation_store
        self.bindings = tuple(sorted(bindings, key=lambda item: item.resource_id))
        self._routes = routes

    def resource_views(self) -> tuple[HardwareResourceView, ...]:
        views = []
        for binding in self.bindings:
            routes = tuple(self._routes[value] for value in binding.route_hashes)
            views.append(
                HardwareResourceView(
                    resource_id=binding.resource_id,
                    resource_kind=binding.resource_kind,
                    target_kind=self.manifest.target_kind,
                    read_skill_refs=tuple(
                        _skill_ref(route)
                        for route in routes
                        if route.control_effect_class == "read_only"
                    ),
                    drive_skill_refs=tuple(
                        _skill_ref(route)
                        for route in routes
                        if route.control_effect_class == "motion"
                    ),
                )
            )
        return tuple(views)

    def status_views(
        self,
        *,
        station_revision: int,
    ) -> tuple[HardwareStatusView, ...]:
        operation = self.operation_store.active_operation()
        request = (
            None
            if operation is None
            else self.operation_store.request_for_operation(operation.operation_id)
        )
        route = None if request is None else self._routes.get(request.route_hash)
        unknown_active = operation is not None and route is None
        views = []

        for binding in self.bindings:
            try:
                health = enum_value(
                    binding.health_provider(),
                    field_name="hardware health",
                    allowed=HARDWARE_HEALTH_STATES_V1,
                )
                lower_occupied = binding.occupied_provider()
                if not isinstance(lower_occupied, bool):
                    raise TypeError("occupied_provider must return a boolean.")
            except Exception:  # noqa: BLE001 - public view fails closed.
                views.append(
                    HardwareStatusView(
                        binding.resource_id,
                        "occupied",
                        "unavailable",
                        station_revision,
                        "status_provider_unavailable_fail_closed",
                    )
                )
                continue

            if unknown_active:
                views.append(
                    HardwareStatusView(
                        binding.resource_id,
                        "occupied",
                        "unavailable",
                        station_revision,
                        "unknown_active_operation_fail_closed",
                    )
                )
                continue

            bound = request is not None and request.route_hash in binding.route_hashes
            if not bound or route is None:
                views.append(
                    HardwareStatusView(
                        binding.resource_id,
                        "occupied" if lower_occupied else "free",
                        health,
                        station_revision,
                        (
                            "lower_resource_occupied"
                            if lower_occupied
                            else "no_active_bound_operation"
                        ),
                    )
                )
                continue

            occupies = (
                lower_occupied
                or route.control_effect_class != "read_only"
                or binding.occupies_on_read
            )
            if lower_occupied:
                reason_code = "lower_resource_occupied"
            elif occupies:
                reason_code = "active_exclusive_operation"
            else:
                reason_code = "active_shared_read_operation"
            views.append(
                HardwareStatusView(
                    binding.resource_id,
                    "occupied" if occupies else "free",
                    health,
                    station_revision,
                    reason_code,
                    active_operation_id=operation.operation_id,
                    active_skill_ref=_skill_ref(route),
                )
            )
        return tuple(views)


__all__ = ["HardwareResourceBinding", "HardwareViewProvider"]
