from __future__ import annotations

import threading

import pytest

from robot_motion.runtime import thread_affinity


def test_cpu_thread_sibling_ids_parses_linux_cpu_lists(monkeypatch) -> None:
    monkeypatch.setattr(
        thread_affinity.Path,
        "read_text",
        lambda _path, *, encoding: "2-3,7\n",
    )

    assert thread_affinity.cpu_thread_sibling_ids(3) == (2, 3, 7)


def test_thread_affinity_none_is_a_platform_independent_noop(monkeypatch) -> None:
    monkeypatch.delattr(thread_affinity.os, "sched_getaffinity", raising=False)
    monkeypatch.delattr(thread_affinity.os, "sched_setaffinity", raising=False)

    receipt = thread_affinity.bind_current_thread_to_cpu(
        None,
        role="rollback_worker",
    )

    assert receipt["requested"] is False
    assert receipt["applied"] is False
    assert receipt["native_thread_id"] == threading.get_native_id()
    assert receipt["effective_cpu_ids"] is None


def test_thread_affinity_applies_and_verifies_exact_native_thread_mask(
    monkeypatch,
) -> None:
    masks = {0: {2, 3}}
    calls: list[tuple[int, set[int]]] = []

    def fake_get(pid: int):
        return masks.get(int(pid), {2, 3})

    def fake_set(pid: int, cpus: set[int]) -> None:
        calls.append((int(pid), set(cpus)))
        masks[int(pid)] = set(cpus)

    monkeypatch.setattr(thread_affinity.os, "sched_getaffinity", fake_get)
    monkeypatch.setattr(thread_affinity.os, "sched_setaffinity", fake_set)

    receipt = thread_affinity.bind_current_thread_to_cpu(3, role="c2_worker")

    assert calls == [(threading.get_native_id(), {3})]
    assert receipt["applied"] is True
    assert receipt["effective_cpu_ids"] == [3]


def test_thread_affinity_rejects_cpu_outside_inherited_mask(monkeypatch) -> None:
    monkeypatch.setattr(
        thread_affinity.os,
        "sched_getaffinity",
        lambda _pid: {4, 5},
    )

    with pytest.raises(RuntimeError, match="outside.*affinity mask"):
        thread_affinity.bind_current_thread_to_cpu(6, role="reference_worker")


def test_thread_affinity_binds_a_real_child_thread_on_linux() -> None:
    if not hasattr(thread_affinity.os, "sched_getaffinity"):
        pytest.skip("Linux sched affinity is unavailable")
    allowed = thread_affinity.available_cpu_ids()
    if not allowed:
        pytest.skip("no CPU is available to the test process")
    result: dict[str, object] = {}

    def bind_child() -> None:
        result.update(
            thread_affinity.bind_current_thread_to_cpu(
                allowed[0],
                role="unit_child_worker",
            )
        )

    worker = threading.Thread(target=bind_child)
    worker.start()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert result["effective_cpu_ids"] == [allowed[0]]
    assert result["applied"] is True


def test_process_cpu_exclusion_migrates_non_exempt_native_threads(
    monkeypatch,
) -> None:
    masks = {
        101: {0, 1, 14, 15},
        102: {1, 14, 15},
        103: {14},
    }
    calls: list[tuple[int, set[int]]] = []

    def fake_get(native_thread_id: int):
        if int(native_thread_id) == 0:
            return set(masks[101])
        return set(masks[int(native_thread_id)])

    def fake_set(native_thread_id: int, cpus: set[int]) -> None:
        calls.append((int(native_thread_id), set(cpus)))
        masks[int(native_thread_id)] = set(cpus)

    monkeypatch.setattr(
        thread_affinity,
        "_native_thread_ids",
        lambda: (101, 102, 103),
    )
    monkeypatch.setattr(thread_affinity.os, "sched_getaffinity", fake_get)
    monkeypatch.setattr(thread_affinity.os, "sched_setaffinity", fake_set)

    receipt = thread_affinity.exclude_cpu_ids_from_current_process(
        (14, 15),
        role="deployment_client_ordinary_threads",
        exempt_native_thread_ids=(103,),
    )

    assert calls == [(101, {0, 1}), (102, {1})]
    assert masks[103] == {14}
    assert receipt["applied"] is True
    assert receipt["excluded_cpu_ids"] == [14, 15]
    assert receipt["exempt_native_thread_ids"] == [103]
    assert receipt["native_thread_count"] == 3
    assert receipt["native_threads_migrated"] == 2


def test_process_cpu_exclusion_does_not_broaden_a_pinned_thread(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        thread_affinity,
        "_native_thread_ids",
        lambda: (201,),
    )
    monkeypatch.setattr(
        thread_affinity.os,
        "sched_getaffinity",
        lambda native_thread_id: (
            {0, 14} if int(native_thread_id) == 0 else {14}
        ),
    )
    monkeypatch.setattr(
        thread_affinity.os,
        "sched_setaffinity",
        lambda _native_thread_id, _cpus: None,
    )

    with pytest.raises(RuntimeError, match="without broadening"):
        thread_affinity.exclude_cpu_ids_from_current_process(
            (14,),
            role="deployment_client_ordinary_threads",
        )
