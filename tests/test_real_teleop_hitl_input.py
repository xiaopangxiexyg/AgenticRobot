from __future__ import annotations

import pytest

from teleop_runtime.real_teleop.hitl_terminal import HITLTerminalClutchInput


class _Keyboard:
    def __init__(self) -> None:
        self.buttons = {}

    def set_button(self, name, pressed, **kwargs):
        del kwargs
        self.buttons[name] = bool(pressed)


def test_hitl_terminal_preserves_existing_keys_and_emits_edges() -> None:
    keyboard = _Keyboard()
    terminal = HITLTerminalClutchInput(
        keyboard,
        takeover_key="u",
        policy_resume_key="i",
    )
    terminal.enabled = False

    assert terminal._handle_character("z") is True
    assert keyboard.buttons["left_clutch_button"] is True
    assert terminal._handle_character("t") is True
    assert terminal.consume_episode_start_request() is True

    terminal._handle_character("u")
    terminal._handle_character("i")
    assert terminal.consume_hitl_source_requests() == ("human", "policy")
    assert terminal.consume_hitl_source_requests() == ()


def test_hitl_terminal_keys_cannot_shadow_safety_or_existing_controls() -> None:
    with pytest.raises(ValueError, match="conflict"):
        HITLTerminalClutchInput(
            _Keyboard(),
            takeover_key="q",
            policy_resume_key="i",
        )
    with pytest.raises(ValueError, match="must differ"):
        HITLTerminalClutchInput(
            _Keyboard(),
            takeover_key="u",
            policy_resume_key="u",
        )


def test_quit_stops_same_terminal_batch_before_later_hitl_edge() -> None:
    terminal = HITLTerminalClutchInput(_Keyboard())
    assert terminal._handle_character("q") is False
    assert terminal.should_quit is True
    assert terminal.consume_hitl_source_requests() == ()
