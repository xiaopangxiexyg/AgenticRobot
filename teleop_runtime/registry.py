from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Generic, TypeVar


T = TypeVar("T")

_BUILTIN_MODULES_BY_KIND = {
    "source": ("teleop_runtime.sources",),
    "retargeter": ("teleop_runtime.retargeters",),
    "backend": (
        "teleop_runtime.backends.arms.rm65",
        "teleop_runtime.backends.arms.rm65_impedance",
        "teleop_runtime.backends.hands.inspire_tac",
        "teleop_runtime.backends.grippers.dmtac",
    ),
    "rig": ("teleop_runtime.rigs",),
}


class ComponentRegistry(Generic[T]):
    def __init__(self, component_kind: str):
        self.component_kind = component_kind
        self._factories: dict[str, Callable[..., T]] = {}
        self._loading_builtins = False
        self._builtins_loaded = False

    def _ensure_builtins_loaded(self) -> None:
        module_names = _BUILTIN_MODULES_BY_KIND.get(self.component_kind)
        if (
            module_names is None
            or self._builtins_loaded
            or self._loading_builtins
        ):
            return
        self._loading_builtins = True
        try:
            for module_name in module_names:
                import_module(module_name)
            self._builtins_loaded = True
        finally:
            self._loading_builtins = False

    def register(
        self, name: str, factory: Callable[..., T] | None = None, *, override: bool = False
    ):
        self._ensure_builtins_loaded()
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
        self._ensure_builtins_loaded()
        if name not in self._factories:
            raise KeyError(
                f"Unknown {self.component_kind} '{name}'. Registered: {sorted(self._factories)}"
            )
        return self._factories[name](*args, **kwargs)

    def names(self) -> tuple[str, ...]:
        self._ensure_builtins_loaded()
        return tuple(sorted(self._factories))


source_registry = ComponentRegistry("source")
retargeter_registry = ComponentRegistry("retargeter")
backend_registry = ComponentRegistry("backend")
rig_registry = ComponentRegistry("rig")

register_source = source_registry.register
register_retargeter = retargeter_registry.register
register_backend = backend_registry.register
register_rig = rig_registry.register
