from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from collection_runtime.modality_storage import ModalityStorageError
from scripts.probes import detach_verified_realman_tactile as probe


def test_wait_for_global_collection_quiescence_waits_until_empty() -> None:
    active_results = iter(
        [
            ("pid=10 python -m teleop_runtime.collection",),
            ("pid=10 python -m teleop_runtime.collection",),
            (),
        ]
    )
    now = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    probe._wait_for_global_collection_quiescence(
        deadline=30.0,
        poll_sec=2.0,
        active_probe=lambda: next(active_results),
        monotonic=lambda: now[0],
        sleep=sleep,
    )

    assert sleeps == [2.0, 2.0]


def test_wait_for_global_collection_quiescence_times_out() -> None:
    now = [0.0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    with pytest.raises(TimeoutError, match="formal collection"):
        probe._wait_for_global_collection_quiescence(
            deadline=3.0,
            poll_sec=2.0,
            active_probe=lambda: ("pid=10 active",),
            monotonic=lambda: now[0],
            sleep=sleep,
        )


def test_wait_for_predecessor_uses_process_start_identity() -> None:
    identities = iter(
        [
            (100, "python detach_verified_realman_tactile.py"),
            (100, "python detach_verified_realman_tactile.py"),
            (101, "python unrelated.py"),
        ]
    )
    now = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    probe._wait_for_predecessor(
        123,
        timeout_sec=30.0,
        poll_sec=2.0,
        identity_probe=lambda pid: next(identities),
        monotonic=lambda: now[0],
        sleep=sleep,
    )

    assert sleeps == [2.0]


def test_wait_for_predecessor_rejects_unrelated_process() -> None:
    with pytest.raises(RuntimeError, match="unrelated predecessor"):
        probe._wait_for_predecessor(
            123,
            timeout_sec=30.0,
            poll_sec=2.0,
            identity_probe=lambda pid: (100, "python unrelated.py"),
        )


def test_apply_replans_when_collection_restarts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = tmp_path / "dataset"
    cold_root = tmp_path / "cold"
    dataset_root.mkdir()
    cold_root.mkdir()
    plan_path = tmp_path / "relocation.json"
    plan_path.write_text(
        json.dumps({"entries": [{"target": str(dataset_root)}]}),
        encoding="utf-8",
    )
    plans = [
        SimpleNamespace(
            ready=True,
            errors=(),
            plan_hash="plan-1",
            dataset_fingerprint="fingerprint-1",
        ),
        SimpleNamespace(
            ready=True,
            errors=(),
            plan_hash="plan-2",
            dataset_fingerprint="fingerprint-2",
        ),
    ]
    plan_calls: list[tuple[object, ...]] = []
    detach_calls: list[str] = []
    wait_calls: list[tuple[float, float]] = []

    def plan(*args: object, **kwargs: object) -> object:
        plan_calls.append(args)
        return plans[len(plan_calls) - 1]

    def detach(*args: object, **kwargs: object) -> object:
        plan_hash = str(kwargs["expected_plan_hash"])
        detach_calls.append(plan_hash)
        if len(detach_calls) == 1:
            raise ModalityStorageError("formal collection is active: pid=10 active")
        return SimpleNamespace(to_dict=lambda: {"status": "complete"})

    monkeypatch.setattr(probe, "plan_modality_detach", plan)
    monkeypatch.setattr(probe, "detach_modality", detach)
    monkeypatch.setattr(
        probe,
        "_wait_for_global_collection_quiescence",
        lambda *, deadline, poll_sec: wait_calls.append((deadline, poll_sec)),
    )
    monkeypatch.setattr(
        probe.sys,
        "argv",
        [
            "detach_verified_realman_tactile.py",
            "--plan",
            str(plan_path),
            "--entry-index",
            "0",
            "--cold-root",
            str(cold_root),
            "--apply",
            "--confirm-legacy-hardlink-detach",
            "--wait-for-no-active-collection",
            "--poll-sec",
            "1",
        ],
    )

    assert probe.main() == 0
    assert len(wait_calls) == 2
    assert len(plan_calls) == 2
    assert detach_calls == ["plan-1", "plan-2"]


def test_apply_does_not_retry_unrelated_storage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = tmp_path / "dataset"
    cold_root = tmp_path / "cold"
    dataset_root.mkdir()
    cold_root.mkdir()
    plan_path = tmp_path / "relocation.json"
    plan_path.write_text(
        json.dumps({"entries": [{"target": str(dataset_root)}]}),
        encoding="utf-8",
    )
    proposal = SimpleNamespace(
        ready=True,
        errors=(),
        plan_hash="plan",
        dataset_fingerprint="fingerprint",
    )
    monkeypatch.setattr(probe, "plan_modality_detach", lambda *args, **kwargs: proposal)
    monkeypatch.setattr(
        probe,
        "detach_modality",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ModalityStorageError("cold artifact verification failed")
        ),
    )
    monkeypatch.setattr(
        probe,
        "_wait_for_global_collection_quiescence",
        lambda *, deadline, poll_sec: None,
    )
    monkeypatch.setattr(
        probe.sys,
        "argv",
        [
            "detach_verified_realman_tactile.py",
            "--plan",
            str(plan_path),
            "--entry-index",
            "0",
            "--cold-root",
            str(cold_root),
            "--apply",
            "--confirm-legacy-hardlink-detach",
            "--wait-for-no-active-collection",
        ],
    )

    with pytest.raises(ModalityStorageError, match="verification failed"):
        probe.main()


def test_apply_can_delegate_dynamic_disjoint_collection_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = tmp_path / "archived_dataset"
    cold_root = tmp_path / "cold"
    dataset_root.mkdir()
    cold_root.mkdir()
    plan_path = tmp_path / "relocation.json"
    plan_path.write_text(
        json.dumps({"entries": [{"target": str(dataset_root)}]}),
        encoding="utf-8",
    )
    proposal = SimpleNamespace(
        ready=True,
        errors=(),
        plan_hash="plan",
        dataset_fingerprint="fingerprint",
    )
    detach_kwargs: dict[str, object] = {}

    def detach(*args: object, **kwargs: object) -> object:
        detach_kwargs.update(kwargs)
        return SimpleNamespace(to_dict=lambda: {"status": "complete"})

    monkeypatch.setattr(probe, "plan_modality_detach", lambda *args, **kwargs: proposal)
    monkeypatch.setattr(probe, "detach_modality", detach)
    monkeypatch.setattr(
        probe.sys,
        "argv",
        [
            "detach_verified_realman_tactile.py",
            "--plan",
            str(plan_path),
            "--entry-index",
            "0",
            "--cold-root",
            str(cold_root),
            "--apply",
            "--confirm-legacy-hardlink-detach",
            "--allow-disjoint-active-collection",
        ],
    )

    assert probe.main() == 0
    assert detach_kwargs["allow_disjoint_active_collection"] is True
    assert detach_kwargs["active_collection_dataset_roots"] == ()


def test_sampled_revalidation_is_forwarded_to_plan_and_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = tmp_path / "dataset"
    cold_root = tmp_path / "cold"
    dataset_root.mkdir()
    cold_root.mkdir()
    plan_path = tmp_path / "relocation.json"
    plan_path.write_text(
        json.dumps({"entries": [{"target": str(dataset_root)}]}),
        encoding="utf-8",
    )
    proposal = SimpleNamespace(
        ready=True,
        errors=(),
        plan_hash="plan",
        dataset_fingerprint="fingerprint",
    )
    plan_kwargs: dict[str, object] = {}
    detach_kwargs: dict[str, object] = {}

    def plan(*args: object, **kwargs: object) -> object:
        plan_kwargs.update(kwargs)
        return proposal

    def detach(*args: object, **kwargs: object) -> object:
        detach_kwargs.update(kwargs)
        return SimpleNamespace(to_dict=lambda: {"status": "complete"})

    monkeypatch.setattr(probe, "plan_modality_detach", plan)
    monkeypatch.setattr(probe, "detach_modality", detach)
    monkeypatch.setattr(
        probe.sys,
        "argv",
        [
            "detach_verified_realman_tactile.py",
            "--plan",
            str(plan_path),
            "--entry-index",
            "0",
            "--cold-root",
            str(cold_root),
            "--sampled-revalidation",
            "--apply",
            "--confirm-legacy-hardlink-detach",
            "--allow-disjoint-active-collection",
        ],
    )

    assert probe.main() == 0
    assert plan_kwargs["revalidation_profile"] == "sampled_v1"
    assert detach_kwargs["revalidation_profile"] == "sampled_v1"
