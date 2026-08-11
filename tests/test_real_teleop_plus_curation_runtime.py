from __future__ import annotations

from pathlib import Path
from threading import Event
from types import SimpleNamespace

from teleop_runtime.real_teleop.plus_curation_runtime import (
    RealTeleopPlusCurationRuntime,
)


class _FakeSession:
    def __init__(self) -> None:
        self.enabled = True
        self.status = "ready"
        self.message = ""
        self.pending_episode_index = 4
        self.pending_dataset_root = Path("/tmp/plus-source")
        self.pending_token = "token-4"
        self.variant_id = "variant-a"
        self.review: tuple[str, str] | None = None
        self.save_impl = lambda **_kwargs: SimpleNamespace(
            annotation_episode_id="annotation:4"
        )

    def matches_pending(self, *, episode_index: int, pending_token: str) -> bool:
        return bool(
            self.enabled
            and self.pending_episode_index == episode_index
            and self.pending_token == pending_token
        )

    def set_review(self, *, disposition: str, reason_code: str) -> None:
        if disposition == "invalid":
            raise ValueError("invalid disposition")
        self.review = (disposition, reason_code)

    def save(self, **kwargs):
        return self.save_impl(**kwargs)


def test_plus_curation_request_is_decoded_and_consumed_once() -> None:
    session = _FakeSession()
    runtime = RealTeleopPlusCurationRuntime(session)  # type: ignore[arg-type]
    state_changes: list[str] = []

    assert runtime.submit_request_text(
        '{"episode_index":4,"pending_token":" token-4 ",'
        '"disposition":"include","reason_code":"success"}',
        on_state_changed=lambda: state_changes.append(session.status),
    )
    assert runtime.consume_request() == {
        "episode_index": 4,
        "pending_token": "token-4",
        "disposition": "include",
        "reason_code": "success",
    }
    assert runtime.consume_request() is None
    assert state_changes == ["saving"]

    assert not runtime.submit_request_text(
        "[]",
        on_state_changed=lambda: state_changes.append(session.status),
    )
    assert runtime.consume_request() is None
    assert session.status == "error"
    assert "payload must be an object" in session.message


def test_plus_curation_runtime_serializes_one_background_save() -> None:
    session = _FakeSession()
    runtime = RealTeleopPlusCurationRuntime(session)  # type: ignore[arg-type]
    entered = Event()
    release = Event()
    saved_kwargs: list[dict[str, object]] = []
    successes: list[str] = []
    failures: list[str] = []
    states: list[str] = []

    def save_impl(**kwargs):
        saved_kwargs.append(dict(kwargs))
        entered.set()
        assert release.wait(timeout=2.0)
        session.status = "saved"
        return SimpleNamespace(annotation_episode_id="annotation:4")

    session.save_impl = save_impl
    thread = runtime.start_save(
        episode_index=4,
        pending_token="token-4",
        disposition="include",
        reason_code="success",
        on_state_changed=lambda: states.append(session.status),
        on_failure=failures.append,
        on_success=lambda record: successes.append(record.annotation_episode_id),
    )

    assert thread is not None
    assert entered.wait(timeout=2.0)
    assert runtime.save_in_flight()
    assert runtime.start_save(
        episode_index=4,
        pending_token="token-4",
        disposition="include",
        reason_code="success",
        on_state_changed=lambda: states.append(session.status),
        on_failure=failures.append,
        on_success=lambda _record: None,
    ) is None
    release.set()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert saved_kwargs == [
        {
            "disposition": "include",
            "reason_code": "success",
            "pending_token": "token-4",
            "variant_id": "variant-a",
            "source_episode_index": 4,
            "source_dataset_root": Path("/tmp/plus-source"),
        }
    ]
    assert successes == ["annotation:4"]
    assert failures == []
    assert states[0] == "saving"
    assert states[-1] == "saved"


def test_plus_curation_failure_does_not_overwrite_a_newer_pending_episode() -> None:
    session = _FakeSession()
    runtime = RealTeleopPlusCurationRuntime(session)  # type: ignore[arg-type]
    failures: list[str] = []

    def save_impl(**_kwargs):
        session.pending_episode_index = 5
        session.pending_dataset_root = Path("/tmp/new-source")
        session.pending_token = "token-5"
        session.status = "ready"
        session.message = "new episode pending"
        raise OSError("simulated sidecar failure")

    session.save_impl = save_impl
    thread = runtime.start_save(
        episode_index=4,
        pending_token="token-4",
        disposition="include",
        reason_code="success",
        on_state_changed=lambda: None,
        on_failure=failures.append,
        on_success=lambda _record: None,
    )
    assert thread is not None
    thread.join(timeout=2.0)

    assert session.status == "ready"
    assert session.message == "new episode pending"
    assert failures == ["Plus sidecar 写入失败: OSError: simulated sidecar failure"]


def test_plus_curation_stale_token_is_rejected_without_starting_a_thread() -> None:
    session = _FakeSession()
    runtime = RealTeleopPlusCurationRuntime(session)  # type: ignore[arg-type]
    states: list[str] = []

    assert runtime.start_save(
        episode_index=4,
        pending_token="stale-token",
        disposition="include",
        reason_code="success",
        on_state_changed=lambda: states.append(session.status),
        on_failure=lambda _message: None,
        on_success=lambda _record: None,
    ) is None
    assert session.status == "error"
    assert "已过期" in session.message
    assert states == ["error"]
