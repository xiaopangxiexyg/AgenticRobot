from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, TypeVar

from ..interfaces import EffectorBackend, ObservationSource, Retargeter, Rig
from ..session import TeleopSession

_NamedT = TypeVar("_NamedT")


def _coerce_named_mapping(
    values: Mapping[str, _NamedT] | Iterable[_NamedT] | None,
) -> dict[str, _NamedT]:
    if values is None:
        return {}
    if isinstance(values, Mapping):
        return {str(name): value for name, value in values.items()}
    resolved: dict[str, _NamedT] = {}
    for value in values:
        name = getattr(value, "name", None)
        if not name:
            raise ValueError(f"Collector component {value!r} is missing a usable 'name'.")
        resolved[str(name)] = value
    return resolved


@dataclass(frozen=True)
class CollectorComponents:
    session: TeleopSession
    sources: dict[str, ObservationSource] = field(default_factory=dict)
    retargeters: dict[str, Retargeter] = field(default_factory=dict)
    backends: dict[str, EffectorBackend] = field(default_factory=dict)
    rig: Rig | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", dict(self.sources))
        object.__setattr__(self, "retargeters", dict(self.retargeters))
        object.__setattr__(self, "backends", dict(self.backends))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_iterables(
        cls,
        *,
        session: TeleopSession,
        sources: Mapping[str, ObservationSource] | Iterable[ObservationSource] | None = None,
        retargeters: Mapping[str, Retargeter] | Iterable[Retargeter] | None = None,
        backends: Mapping[str, EffectorBackend] | Iterable[EffectorBackend] | None = None,
        rig: Rig | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "CollectorComponents":
        return cls(
            session=session,
            sources=_coerce_named_mapping(sources),
            retargeters=_coerce_named_mapping(retargeters),
            backends=_coerce_named_mapping(backends),
            rig=rig,
            metadata={} if metadata is None else dict(metadata),
        )

    def require_source(self, name: str) -> ObservationSource:
        try:
            return self.sources[str(name)]
        except KeyError as exc:
            raise KeyError(f"Missing collector source '{name}'.") from exc

    def require_backend(self, name: str) -> EffectorBackend:
        try:
            return self.backends[str(name)]
        except KeyError as exc:
            raise KeyError(f"Missing collector backend '{name}'.") from exc

    def require_retargeter(self, name: str) -> Retargeter:
        try:
            return self.retargeters[str(name)]
        except KeyError as exc:
            raise KeyError(f"Missing collector retargeter '{name}'.") from exc
