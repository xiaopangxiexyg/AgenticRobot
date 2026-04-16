from __future__ import annotations

import time
from dataclasses import dataclass

from ...interfaces import ObservationSource
from ...schemas import ObservationChannel


@dataclass
class _ButtonState:
    side: str | None
    value: bool
    timestamp: float
    critical: bool


class KeyboardButtonSource(ObservationSource):
    def __init__(self, *, name: str = "keyboard_buttons"):
        self._name = name
        self._buttons: dict[str, _ButtonState] = {}

    @property
    def name(self) -> str:
        return self._name

    def set_button(
        self, button_name: str, pressed: bool, *, side: str | None = None, critical: bool = False
    ) -> None:
        self._buttons[button_name] = _ButtonState(
            side=side,
            value=bool(pressed),
            timestamp=time.time(),
            critical=critical,
        )

    def expected_channel_names(self) -> tuple[str, ...]:
        return tuple(self._buttons)

    def get_latest_channels(self) -> dict[str, ObservationChannel]:
        return {
            button_name: ObservationChannel(
                channel_name=button_name,
                capability="bool",
                side=state.side,
                value=state.value,
                frame_name=None,
                timestamp=state.timestamp,
                source_name=self.name,
                valid=True,
                critical=state.critical,
                metadata={},
            )
            for button_name, state in self._buttons.items()
        }
