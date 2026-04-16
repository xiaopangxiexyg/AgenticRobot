from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar


T = TypeVar("T")


class ComponentRegistry(Generic[T]):
    def __init__(self, component_kind: str):
        self.component_kind = component_kind
        self._factories: dict[str, Callable[..., T]] = {}

    def register(
        self, name: str, factory: Callable[..., T] | None = None, *, override: bool = False
    ):
        if factory is None:
            def decorator(inner: Callable[..., T]) -> Callable[..., T]:
                self.register(name, inner, override=override)
                return inner

            return decorator

        if not override and name in self._factories:
            raise KeyError(f"{self.component_kind} '{name}' is already registered.")
        self._factories[name] = factory
        return factory

    def create(self, name: str, *args, **kwargs) -> T:
        if name not in self._factories:
            raise KeyError(
                f"Unknown {self.component_kind} '{name}'. Registered: {sorted(self._factories)}"
            )
        return self._factories[name](*args, **kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


source_registry = ComponentRegistry("source")
retargeter_registry = ComponentRegistry("retargeter")
backend_registry = ComponentRegistry("backend")
rig_registry = ComponentRegistry("rig")

register_source = source_registry.register
register_retargeter = retargeter_registry.register
register_backend = backend_registry.register
register_rig = rig_registry.register
