from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path

import pytest

from deployments.openpi import cpu_resources as resources

pytestmark = [pytest.mark.unit, pytest.mark.contract]
ROOT = Path(__file__).resolve().parents[1]
TRAINING_MODULES = (
    "deployments.openpi.training.launch",
    "deployments.openpi.training.prepared_skill_launch",
    "deployments.openpi.corrected_train",
    "deployments.openpi.training.prepared_writing_sequence",
    "deployments.openpi.training.prepared_writing_state_sequence",
)


def run_python(args, *, extra_path=None, environment=None, timeout=30):
    paths = [str(ROOT)]
    if extra_path is not None:
        paths.insert(0, str(extra_path))
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(paths), **(environment or {})},
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def test_resources_import_without_native_or_control_dependencies():
    result = run_python(
        [
            "-c",
            """
import importlib.abc, sys
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('numpy', 'torch', 'jax', 'hardware', 'teleop_runtime') or fullname.startswith('robot_motion.runtime'):
            raise ImportError('unexpected dependency: ' + fullname)
sys.meta_path.insert(0, Blocker())
import deployments.openpi.cpu_resources
assert 'robot_motion.runtime' not in sys.modules
assert 'numpy' not in sys.modules
""",
        ]
    )
    assert result.returncode == 0, result.stderr


def test_default_bootstrap_does_not_read_or_mutate_resources(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["launch", "train", "--profile", "existing.json"])
    monkeypatch.setattr(
        resources.os, "sched_getaffinity", lambda *a: pytest.fail("no query")
    )
    monkeypatch.setattr(
        resources.os, "sched_setaffinity", lambda *a: pytest.fail("no bind")
    )
    monkeypatch.setattr(resources.os, "execve", lambda *a: pytest.fail("no exec"))
    before = dict(os.environ)
    resources.bootstrap_training_cpu_resources(TRAINING_MODULES[1])
    assert dict(os.environ) == before


def test_bootstrap_preserves_downstream_arguments_and_caller_environment(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "corrected",
            "--exclude-cpu",
            "6",
            "--native-thread-count",
            "1",
            "--",
            "my-config",
            "--native-thread-count",
            "9",
            "--exp-name",
            "unchanged",
        ],
    )
    mask, calls = {6, 7, 22, 23}, []
    monkeypatch.setattr(resources.os, "sched_getaffinity", lambda pid: set(mask))

    def bind(pid, cpus):
        mask.clear()
        mask.update(cpus)

    monkeypatch.setattr(resources.os, "sched_setaffinity", bind)

    class ExecBoundary(Exception):
        pass

    def execute(executable, args, environment):
        calls.append((executable, args, environment))
        raise ExecBoundary

    monkeypatch.setattr(resources.os, "execve", execute)
    before = dict(os.environ)
    with pytest.raises(ExecBoundary):
        resources.bootstrap_training_cpu_resources(TRAINING_MODULES[2])
    assert mask == {7, 22, 23}
    executable, args, environment = calls[0]
    assert executable == sys.executable
    assert args == [
        sys.executable,
        "-m",
        TRAINING_MODULES[2],
        "--",
        "my-config",
        "--native-thread-count",
        "9",
        "--exp-name",
        "unchanged",
    ]
    for key in resources.NATIVE_THREAD_ENVIRONMENT_VARIABLES:
        assert environment[key] == "1"
    assert dict(os.environ) == before


def test_bootstrap_detects_affinity_change_before_exec(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["launch", "--exclude-cpu", "6"])
    masks = iter(({6, 7}, {6, 8}))
    monkeypatch.setattr(resources.os, "sched_getaffinity", lambda pid: next(masks))
    monkeypatch.setattr(
        resources.os, "sched_setaffinity", lambda *a: pytest.fail("no bind")
    )
    monkeypatch.setattr(resources.os, "execve", lambda *a: pytest.fail("no exec"))
    with pytest.raises(RuntimeError, match="affinity changed"):
        resources.bootstrap_training_cpu_resources(TRAINING_MODULES[0])


def test_exclusion_verifies_effective_mask(monkeypatch):
    monkeypatch.setattr(resources.os, "sched_getaffinity", lambda pid: {6, 7})
    monkeypatch.setattr(resources.os, "sched_setaffinity", lambda *a: None)
    with pytest.raises(RuntimeError, match="failed to verify"):
        resources.exclude_cpus_before_exec([6])


def test_exec_failure_does_not_retry_without_resource_options(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["launch", "--native-thread-count", "1", "train"])
    calls = []

    def execute(*args):
        calls.append(args)
        raise OSError("synthetic exec failure")

    monkeypatch.setattr(resources.os, "execve", execute)
    with pytest.raises(OSError, match="synthetic exec failure"):
        resources.bootstrap_training_cpu_resources(TRAINING_MODULES[1])
    assert len(calls) == 1


@pytest.mark.parametrize("module", TRAINING_MODULES)
def test_cli_llc_check_matches_inherited_mask_without_binding(module):
    if not hasattr(os, "sched_getaffinity"):
        pytest.skip("Linux affinity unavailable")
    before = set(os.sched_getaffinity(0))
    cpu = min(before)
    shared = set(resources.cpu_last_level_cache_ids(cpu))
    target = before - shared
    result = run_python(
        ["-m", module, "--exclude-llc-of-cpu", str(cpu), "--cpu-resource-check-only"]
    )
    if not target:
        assert result.returncode == 2
        assert "no runnable CPU" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout)
        plan = report["cpu_resource_plan"]
        assert plan["effective_cpu_ids"] == sorted(target)
        assert plan["excluded_cpu_ids"] == sorted(before & shared)
        assert plan["last_level_cache_groups"] == [
            {"control_cpu": cpu, "shared_cpu_ids": sorted(shared)}
        ]
        assert report["affinity_applied"] is False
    assert set(os.sched_getaffinity(0)) == before


@pytest.mark.parametrize("cpu", [-1, True, 1.5])
def test_plan_rejects_invalid_cpu_values(cpu):
    with pytest.raises(ValueError, match="non-negative integer"):
        resources.cpu_exclusion_plan([cpu], [])
    with pytest.raises(ValueError, match="non-negative integer"):
        resources.exclude_cpus_before_exec([cpu])


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_native_pool_count_is_positive_integer(count):
    with pytest.raises(ValueError, match="positive integer"):
        resources.native_thread_environment(count)


@pytest.mark.parametrize("module", TRAINING_MODULES)
def test_actual_cli_check_only_precedes_heavy_imports(module, tmp_path):
    (tmp_path / "sitecustomize.py").write_text(
        """
import importlib.abc, sys
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('numpy', 'torch', 'jax', 'hardware', 'teleop_runtime', 'dataset_interface') or fullname.startswith('robot_motion.runtime'):
            raise ImportError('check-only imported runtime: ' + fullname)
sys.meta_path.insert(0, Blocker())
""",
        encoding="utf-8",
    )
    result = run_python(
        ["-m", module, "--native-thread-count", "1", "--cpu-resource-check-only"],
        extra_path=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["affinity_applied"] is False and report["model_started"] is False
    assert report["cpu_resource_plan"] is None
    assert set(report["native_thread_environment"].values()) == {"1"}


@pytest.mark.parametrize(
    "flag,value,message",
    [
        ("--native-thread-count", "0", "positive"),
        ("--native-thread-count", "wrong", "integer"),
        ("--exclude-cpu", "-1", "non-negative"),
    ],
)
def test_cli_invalid_resources_fail_before_profile_access(flag, value, message):
    result = run_python(
        ["-m", TRAINING_MODULES[1], "train", "--profile", "/missing.json", flag, value]
    )
    assert result.returncode == 2
    assert message in result.stderr and "/missing.json" not in result.stderr


@pytest.mark.parametrize("module", TRAINING_MODULES)
def test_existing_help_after_real_reexec_has_only_restricted_native_threads(
    module, tmp_path
):
    if not hasattr(os, "sched_getaffinity"):
        pytest.skip("Linux affinity unavailable")
    before = set(os.sched_getaffinity(0))
    if len(before) < 2:
        pytest.skip("requires two allowed CPUs")
    excluded = min(before)
    target = before - {excluded}
    receipt = tmp_path / "import_receipt.json"
    # This runs in both pre-exec and post-exec interpreters. The first has no
    # heavy imports and exec does not run atexit; only the clean child writes.
    (tmp_path / "sitecustomize.py").write_text(
        """
import atexit, importlib.abc, json, os, pathlib, sys
expected = set(json.loads(os.environ['EXPECTED_CPUS']))
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('numpy', 'torch', 'jax'):
            assert set(os.sched_getaffinity(0)) == expected, 'native import before CPU isolation'
            assert os.environ.get('OMP_NUM_THREADS') == '1', 'native import before pool cap'
sys.meta_path.insert(0, Guard())
def receipt():
    threads = {p.name: sorted(os.sched_getaffinity(int(p.name))) for p in pathlib.Path('/proc/self/task').iterdir()}
    pathlib.Path(os.environ['IMPORT_RECEIPT']).write_text(json.dumps({'threads': threads, 'numpy_imported': 'numpy' in sys.modules}))
atexit.register(receipt)
""",
        encoding="utf-8",
    )
    # corrected_train preserves its existing wrapper/OpenPI separator contract.
    help_args = (
        ["--help", "--", "unused-corrected-v1"]
        if module.endswith(".corrected_train")
        else ["--help"]
    )
    result = run_python(
        [
            "-m",
            module,
            "--exclude-cpu",
            str(excluded),
            "--native-thread-count",
            "1",
            *help_args,
        ],
        extra_path=tmp_path,
        environment={
            "EXPECTED_CPUS": json.dumps(sorted(target)),
            "IMPORT_RECEIPT": str(receipt),
        },
    )
    assert result.returncode == 0, result.stderr
    assert "--exclude-llc-of-cpu" in result.stdout
    assert result.stderr.count("[OpenPICPU] READY:") == 1
    report = json.loads(receipt.read_text())
    assert report["numpy_imported"]
    assert report["threads"] and all(
        set(mask) == target for mask in report["threads"].values()
    )
    assert set(os.sched_getaffinity(0)) == before


@pytest.mark.parametrize("module", TRAINING_MODULES)
def test_inprocess_main_cannot_claim_preimport_resource_isolation(module, monkeypatch):
    import importlib

    launcher = importlib.import_module(module)
    args = ["--native-thread-count", "1"]
    if module.endswith(".launch"):
        args += [
            "--base-config",
            "base",
            "--base-checkpoint-dir",
            "/unused",
            "--artifact-template-json",
            "/unused",
            "--sampling-spec-json",
            "/unused",
            "--source-dataset-root",
            "/unused",
            "--camera-key",
            "front",
            "--output-config-name",
            "output",
            "--exp-name",
            "test",
        ]
    elif module.endswith(".prepared_skill_launch"):
        args += ["train", "--profile", "/unused"]
        monkeypatch.setattr(
            launcher, "_load_profile", lambda *a: pytest.fail("no profile read")
        )
    elif module.endswith(".corrected_train"):
        args += ["--", "unused-corrected-v1"]
        monkeypatch.setattr(
            launcher,
            "load_camera_role_binding",
            lambda *a: pytest.fail("no config read"),
        )
    else:
        monkeypatch.setattr(
            launcher, "run_sequence", lambda *a, **k: pytest.fail("no training")
        )
    with pytest.raises(SystemExit) as error:
        launcher.main(args)
    assert error.value.code == 2


@pytest.mark.integration
@pytest.mark.parametrize("method", ["spawn", "fork", "forkserver"])
def test_real_torch_dataloader_inherits_bootstrapped_mask_and_pool_caps(
    method, tmp_path
):
    pytest.importorskip("torch")
    if (
        not hasattr(os, "sched_getaffinity")
        or method not in multiprocessing.get_all_start_methods()
    ):
        pytest.skip("process start method/affinity unavailable")
    before = set(os.sched_getaffinity(0))
    if len(before) < 2:
        pytest.skip("requires two allowed CPUs")
    target = set(sorted(before)[:2])
    (tmp_path / "loader_fixture.py").write_text(
        """
import json, os, pathlib, sys
from deployments.openpi.cpu_resources import bootstrap_training_cpu_resources, NATIVE_THREAD_ENVIRONMENT_VARIABLES
if __name__ == '__main__':
    bootstrap_training_cpu_resources('loader_fixture')
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, get_worker_info
def snapshot():
    np.linalg.svd(np.eye(4))
    return {'pid': os.getpid(), 'cpus': sorted(os.sched_getaffinity(0)),
            'threads': {p.name: sorted(os.sched_getaffinity(int(p.name))) for p in pathlib.Path('/proc/self/task').iterdir()},
            'native': {key: os.environ.get(key) for key in NATIVE_THREAD_ENVIRONMENT_VARIABLES},
            'torch_threads': torch.get_num_threads(), 'is_worker': get_worker_info() is not None}
class Samples(Dataset):
    def __len__(self): return 2
    def __getitem__(self, index): return snapshot()
if __name__ == '__main__':
    loader = DataLoader(Samples(), batch_size=None, num_workers=1, prefetch_factor=1,
                        multiprocessing_context=sys.argv[1], timeout=10)
    workers = list(loader)
    print(json.dumps({'parent': snapshot(), 'workers': workers}))
""",
        encoding="utf-8",
    )
    exclusions = [
        arg for cpu in sorted(before - target) for arg in ("--exclude-cpu", str(cpu))
    ]
    result = run_python(
        ["-m", "loader_fixture", method, *exclusions, "--native-thread-count", "1"],
        extra_path=tmp_path,
        environment={"CUDA_VISIBLE_DEVICES": ""},
        timeout=45,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert len(report["workers"]) == 2
    assert not report["parent"]["is_worker"]
    for sample in [report["parent"], *report["workers"]]:
        assert set(sample["cpus"]) == target
        assert sample["threads"] and all(
            set(mask) == target for mask in sample["threads"].values()
        )
        assert set(sample["native"].values()) == {"1"}
        assert sample["torch_threads"] == 1
    assert all(
        s["is_worker"] and s["pid"] != report["parent"]["pid"]
        for s in report["workers"]
    )
    assert set(os.sched_getaffinity(0)) == before
