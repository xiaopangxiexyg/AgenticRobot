from __future__ import annotations

from types import SimpleNamespace

import pytest

from teleop_runtime.collection import run_collect, terminal_keys


def test_run_collect_keeps_terminal_controller_compatibility_alias() -> None:
    assert run_collect.TerminalKeyController is terminal_keys.TerminalKeyController


def test_disabled_controller_is_safe_to_poll_and_close() -> None:
    controller = terminal_keys.TerminalKeyController()

    assert controller.enabled is False
    assert controller.poll() == []
    controller.close()
    controller.close()
    assert controller.enabled is False


def test_start_disables_cleanly_when_no_terminal_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_open(*_args: object, **_kwargs: object) -> int:
        raise OSError("no tty")

    monkeypatch.setattr(
        terminal_keys.sys, "stdin", SimpleNamespace(isatty=lambda: False)
    )
    monkeypatch.setattr(terminal_keys.os, "open", fail_open)

    controller = terminal_keys.TerminalKeyController()
    controller.start()

    assert controller.enabled is False
    assert controller.input_source == "disabled"
    assert controller.disabled_reason == "OSError: no tty"
    assert controller.poll() == []


def test_poll_decodes_available_bytes_and_lowercases_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = terminal_keys.TerminalKeyController()
    controller.enabled = True
    controller._fd = 7
    chunks = iter((b"Ab", b""))
    readiness = iter(([7], []))
    monkeypatch.setattr(
        terminal_keys.select, "select", lambda *_args: (next(readiness), [], [])
    )
    monkeypatch.setattr(terminal_keys.os, "read", lambda *_args: next(chunks))

    assert controller.poll() == ["a", "b"]


def test_poll_propagates_ctrl_c_as_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = terminal_keys.TerminalKeyController()
    controller.enabled = True
    controller._fd = 7
    monkeypatch.setattr(terminal_keys.select, "select", lambda *_args: ([7], [], []))
    monkeypatch.setattr(terminal_keys.os, "read", lambda *_args: b"\x03")

    with pytest.raises(KeyboardInterrupt):
        controller.poll()
