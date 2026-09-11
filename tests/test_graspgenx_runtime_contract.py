from __future__ import annotations

import json
from pathlib import Path

import pytest

from deployments.perception.graspgenx_runtime_contract import (
    GRASPGENX_INPROCESS_RUNTIME_CONTRACT_ID,
    GRASPGENX_REVIEWED_PYTHON_VERSIONS,
    GraspGenXRuntimeCheck,
    default_graspgenx_source_root,
    verify_graspgenx_inprocess_runtime,
)
from deployments.perception.sam2_graspgenx_point_grasp import (
    GRASPGENX_CHECKPOINT_REVIEWED_REVISION,
    GRASPGENX_REVIEWED_REVISION,
    GraspGenXPointGraspModelSpec,
)


class _FakeCuda:
    def __init__(self, *, available: bool, device_count: int) -> None:
        self._available = available
        self._device_count = device_count

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._device_count


class _FakeTorch:
    def __init__(
        self,
        version: str = "2.6.0",
        *,
        cuda_available: bool = True,
        device_count: int = 1,
    ) -> None:
        self.__version__ = version
        self.cuda = _FakeCuda(available=cuda_available, device_count=device_count)


def _importer(torch_module: object | None):
    def import_module(name: str):
        if name == "torch" and torch_module is not None:
            return torch_module
        raise ModuleNotFoundError(f"No module named {name!r}")

    return import_module


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _asset_bundle(tmp_path: Path) -> Path:
    root = tmp_path / "graspgenx"
    paths = {
        "generator_config": "checkpoints/release/gen/config.yaml",
        "generator_checkpoint": "checkpoints/release/gen/epoch_736.pth",
        "discriminator_config": "checkpoints/release/dis/config.yaml",
        "discriminator_checkpoint": "checkpoints/release/dis/epoch_1056.pth",
    }
    files = {}
    for index, (name, relative) in enumerate(paths.items()):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            path.unlink()
        path.write_bytes(f"test-{index}".encode())
        files[name] = {"relative_path": relative, "sha256": _sha256(path)}
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "dexteleop.perception_graspgenx_point_grasp_assets.v1",
                "model_profile_id": "graspgenx-test",
                "source": {
                    "git_url": "https://github.com/NVlabs/GraspGenX.git",
                    "revision": GRASPGENX_REVIEWED_REVISION,
                    "license": "Apache-2.0",
                },
                "checkpoint_repository": {
                    "git_url": "https://huggingface.co/adithyamurali/GraspGenXModel",
                    "revision": GRASPGENX_CHECKPOINT_REVIEWED_REVISION,
                    "license": "NVIDIA-Open-Model-License",
                },
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    return root


def _source_checkout(tmp_path: Path) -> Path:
    root = tmp_path / "graspgenx-source"
    package = root / "graspgenx"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    return root


def _verify(
    tmp_path: Path,
    *,
    torch_module: object | None = None,
    version_info: tuple[int, int, int] = (3, 10, 14),
    device: str = "cuda",
    git_revision: str | None = GRASPGENX_REVIEWED_REVISION,
):
    return verify_graspgenx_inprocess_runtime(
        model_spec=GraspGenXPointGraspModelSpec.from_asset_bundle(
            _asset_bundle(tmp_path)
        ),
        source_root=_source_checkout(tmp_path),
        device=device,
        import_module=_importer(torch_module),
        git_head_revision=lambda _root: git_revision,
        version_info=version_info,
    )


def test_receipt_passes_for_green_environment(tmp_path: Path) -> None:
    receipt = _verify(tmp_path, torch_module=_FakeTorch())

    assert receipt.passed
    assert receipt.contract_id == GRASPGENX_INPROCESS_RUNTIME_CONTRACT_ID
    assert [check.name for check in receipt.checks] == [
        "python_version",
        "torch_import",
        "torch_version",
        "cuda_available",
        "cuda_device_index",
        "source_root",
        "source_revision",
        "source_package",
        "asset_bundle",
        "checkpoint_layout",
    ]
    assert receipt.failed_checks == ()


def test_python_version_outside_reviewed_range_fails(tmp_path: Path) -> None:
    for version_info in ((3, 9, 18), (3, 12, 1)):
        receipt = _verify(tmp_path, torch_module=_FakeTorch(), version_info=version_info)
        assert not receipt.passed
        assert not receipt.checks[0].status


def test_torch_version_bounds(tmp_path: Path) -> None:
    accepted = ("2.1.0", "2.6.9")
    rejected = ("2.0.2", "2.7.0", "25.0.1")
    for version in accepted:
        receipt = _verify(tmp_path, torch_module=_FakeTorch(version))
        assert dict(
            (check.name, check.status) for check in receipt.checks
        )["torch_version"]
    for version in rejected:
        receipt = _verify(tmp_path, torch_module=_FakeTorch(version))
        assert not dict(
            (check.name, check.status) for check in receipt.checks
        )["torch_version"]
        assert not receipt.passed


@pytest.mark.parametrize(
    "major,minor", GRASPGENX_REVIEWED_PYTHON_VERSIONS
)
def test_reviewed_python_versions_pass(
    tmp_path: Path, major: int, minor: int
) -> None:
    receipt = _verify(
        tmp_path, torch_module=_FakeTorch(), version_info=(major, minor, 0)
    )
    assert receipt.checks[0].status


def test_missing_torch_fails_closed_with_diagnostics(tmp_path: Path) -> None:
    receipt = _verify(tmp_path, torch_module=None)

    assert not receipt.passed
    statuses = {check.name: check.status for check in receipt.checks}
    assert not statuses["torch_import"]
    assert not statuses["torch_version"]
    assert not statuses["cuda_available"]
    assert not statuses["cuda_device_index"]
    assert "ModuleNotFoundError" in receipt.checks[1].detail


def test_unavailable_cuda_fails(tmp_path: Path) -> None:
    receipt = _verify(
        tmp_path, torch_module=_FakeTorch(cuda_available=False, device_count=0)
    )

    assert not receipt.passed
    statuses = {check.name: check.status for check in receipt.checks}
    assert not statuses["cuda_available"]
    assert not statuses["cuda_device_index"]


def test_cuda_device_index_out_of_range_fails(tmp_path: Path) -> None:
    receipt = _verify(
        tmp_path,
        torch_module=_FakeTorch(device_count=1),
        device="cuda:1",
    )
    assert not receipt.passed
    assert not receipt.checks[4].status

    in_range = _verify(
        tmp_path,
        torch_module=_FakeTorch(device_count=2),
        device="cuda:1",
    )
    assert in_range.passed


def test_non_cuda_device_is_rejected_before_checks(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="CUDA device"):
        _verify(tmp_path, torch_module=_FakeTorch(), device="cpu")


def test_missing_source_root_fails(tmp_path: Path) -> None:
    receipt = verify_graspgenx_inprocess_runtime(
        model_spec=GraspGenXPointGraspModelSpec.from_asset_bundle(
            _asset_bundle(tmp_path)
        ),
        source_root=tmp_path / "missing-source",
        import_module=_importer(_FakeTorch()),
        git_head_revision=lambda _root: GRASPGENX_REVIEWED_REVISION,
        version_info=(3, 10, 14),
    )

    assert not receipt.passed
    statuses = {check.name: check.status for check in receipt.checks}
    assert not statuses["source_root"]
    assert not statuses["source_revision"]


def test_source_revision_mismatch_fails(tmp_path: Path) -> None:
    receipt = _verify(tmp_path, git_revision="0" * 40)

    assert not receipt.passed
    assert not receipt.checks[6].status
    assert receipt.checks[6].expected == GRASPGENX_REVIEWED_REVISION
    assert receipt.checks[6].observed == "0" * 40


def test_git_failure_is_recorded_not_raised(tmp_path: Path) -> None:
    def _failing_git(_root: Path) -> str:
        raise OSError("not a git repository")

    receipt = verify_graspgenx_inprocess_runtime(
        model_spec=GraspGenXPointGraspModelSpec.from_asset_bundle(
            _asset_bundle(tmp_path)
        ),
        source_root=_source_checkout(tmp_path),
        import_module=_importer(_FakeTorch()),
        git_head_revision=_failing_git,
        version_info=(3, 10, 14),
    )

    assert not receipt.passed
    assert not receipt.checks[6].status
    assert "OSError" in receipt.checks[6].detail


def test_missing_source_package_layout_fails(tmp_path: Path) -> None:
    empty_root = tmp_path / "empty-source"
    empty_root.mkdir()
    receipt = verify_graspgenx_inprocess_runtime(
        model_spec=GraspGenXPointGraspModelSpec.from_asset_bundle(
            _asset_bundle(tmp_path)
        ),
        source_root=empty_root,
        import_module=_importer(_FakeTorch()),
        git_head_revision=lambda _root: GRASPGENX_REVIEWED_REVISION,
        version_info=(3, 10, 14),
    )

    assert not receipt.passed
    assert not receipt.checks[7].status


def test_missing_checkpoint_layout_fails(tmp_path: Path) -> None:
    bundle = _asset_bundle(tmp_path)
    spec = GraspGenXPointGraspModelSpec.from_asset_bundle(bundle)
    green = verify_graspgenx_inprocess_runtime(
        model_spec=spec,
        source_root=_source_checkout(tmp_path),
        import_module=_importer(_FakeTorch()),
        git_head_revision=lambda _root: GRASPGENX_REVIEWED_REVISION,
        version_info=(3, 10, 14),
    )
    assert green.passed

    removed = bundle / "checkpoints" / "release" / "gen"
    removed.rename(tmp_path / "removed-gen")
    rerun = verify_graspgenx_inprocess_runtime(
        model_spec=spec,
        source_root=_source_checkout(tmp_path),
        import_module=_importer(_FakeTorch()),
        git_head_revision=lambda _root: GRASPGENX_REVIEWED_REVISION,
        version_info=(3, 10, 14),
    )

    assert not rerun.passed
    assert not rerun.checks[9].status
    assert "gen" in rerun.checks[9].detail


def test_receipt_mapping_is_json_serializable_and_deterministic(
    tmp_path: Path,
) -> None:
    first = _verify(tmp_path, torch_module=_FakeTorch()).to_mapping()
    second = _verify(tmp_path, torch_module=_FakeTorch()).to_mapping()

    assert first == second
    encoded = json.dumps(first)
    assert json.loads(encoded) == first
    assert first["schema"] == "dexteleop.perception.graspgenx_runtime_receipt.v1"
    assert first["status"] == "passed"
    assert first["contract_id"] == GRASPGENX_INPROCESS_RUNTIME_CONTRACT_ID

    failed = _verify(tmp_path, torch_module=None).to_mapping()
    assert failed["status"] == "failed"
    assert any(check["status"] == "failed" for check in failed["checks"])


def test_failure_summary_lists_failed_checks(tmp_path: Path) -> None:
    receipt = _verify(tmp_path, torch_module=None)

    summary = receipt.failure_summary()
    assert "torch_import" in summary
    assert "cuda_available" in summary


def test_default_source_root_points_at_pinned_submodule() -> None:
    root = default_graspgenx_source_root()

    assert root.name == "graspgenx"
    assert root.parent.name == "perception"
    assert root.parent.parent.name == "model"


def test_single_passed_check_does_not_make_empty_receipt_pass() -> None:
    from deployments.perception.graspgenx_runtime_contract import (
        GraspGenXRuntimeReceipt,
    )

    receipt = GraspGenXRuntimeReceipt(
        contract_id=GRASPGENX_INPROCESS_RUNTIME_CONTRACT_ID,
        device="cuda",
        source_root="/tmp/graspgenx",
        model_spec_hash="0" * 64,
        checks=(
            GraspGenXRuntimeCheck(
                name="stub",
                status=True,
                expected="e",
                observed="o",
            ),
        ),
    )
    assert receipt.passed

    empty = GraspGenXRuntimeReceipt(
        contract_id=GRASPGENX_INPROCESS_RUNTIME_CONTRACT_ID,
        device="cuda",
        source_root="/tmp/graspgenx",
        model_spec_hash="0" * 64,
    )
    assert not empty.passed
