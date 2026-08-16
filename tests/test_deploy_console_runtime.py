from __future__ import annotations

import pytest

from teleop_runtime.deploy import console_runtime, run_policy


class _FakeStdin:
    def __init__(self, *, is_tty: bool, text: str = "") -> None:
        self.is_tty = is_tty
        self.text = text

    def isatty(self) -> bool:
        return self.is_tty

    def fileno(self) -> int:
        return 7

    def read(self, size: int) -> str:
        assert size == 1
        return self.text[:1]


def test_run_policy_preserves_console_runtime_compatibility_exports() -> None:
    assert run_policy.NonBlockingKeyboard is console_runtime.NonBlockingKeyboard
    assert run_policy.LoopRate is console_runtime.LoopRate


def test_loop_rate_preserves_smoothed_frequency_estimate() -> None:
    rate = console_runtime.LoopRate()

    assert rate.tick(10.0) == 0.0
    assert rate.tick(10.5) == pytest.approx(2.0)
    assert rate.tick(11.0) == pytest.approx(2.0)
    assert rate.tick(11.25) == pytest.approx(2.4)


def test_non_tty_keyboard_is_a_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdin = _FakeStdin(is_tty=False)
    monkeypatch.setattr(console_runtime.sys, "stdin", stdin)

    with console_runtime.NonBlockingKeyboard() as keyboard:
        assert keyboard.poll() is None


def test_tty_keyboard_sets_polls_and_restores_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    stdin = _FakeStdin(is_tty=True, text="g")
    old_settings = ["old-settings"]
    monkeypatch.setattr(console_runtime.sys, "stdin", stdin)
    monkeypatch.setattr(
        console_runtime.termios,
        "tcgetattr",
        lambda fd: events.append(("get", fd)) or old_settings,
    )
    monkeypatch.setattr(
        console_runtime.tty,
        "setcbreak",
        lambda fd: events.append(("cbreak", fd)),
    )
    monkeypatch.setattr(
        console_runtime.select,
        "select",
        lambda readers, writers, errors, timeout: (
            events.append(("poll", readers, writers, errors, timeout))
            or ([stdin], [], [])
        ),
    )
    monkeypatch.setattr(
        console_runtime.termios,
        "tcsetattr",
        lambda fd, when, settings: events.append(
            ("restore", fd, when, settings)
        ),
    )

    with console_runtime.NonBlockingKeyboard() as keyboard:
        assert keyboard.poll() == "g"

    assert events == [
        ("get", 7),
        ("cbreak", 7),
        ("poll", [stdin], [], [], 0.0),
        ("restore", 7, console_runtime.termios.TCSADRAIN, old_settings),
    ]
