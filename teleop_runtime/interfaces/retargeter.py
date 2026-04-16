from __future__ import annotations

from abc import ABC, abstractmethod

from ..calibration import TeleopCalibration
from ..schemas import TeleopCommand, TeleopObservation


class Retargeter(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def retarget(
        self, observation: TeleopObservation, calibration: TeleopCalibration | None = None
    ) -> TeleopCommand:
        raise NotImplementedError
