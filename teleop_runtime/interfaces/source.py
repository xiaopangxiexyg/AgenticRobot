from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..schemas import ObservationChannel


class ObservationSource(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    def start(self) -> None:
        """Optional resource allocation hook."""

    @abstractmethod
    def get_latest_channels(self) -> dict[str, ObservationChannel]:
        raise NotImplementedError

    def expected_channel_names(self) -> tuple[str, ...]:
        return ()

    def close(self) -> None:
        """Optional resource release hook."""

    def __enter__(self) -> "ObservationSource":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
