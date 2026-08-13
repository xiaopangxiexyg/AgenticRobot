from __future__ import annotations

import json
from pathlib import Path
from dataclasses import replace
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from dataset_contracts.camera_semantics import (
    CAMERA_SEMANTIC_MATERIALIZATION_HASH_DOMAIN_V1,
    CAMERA_SEMANTIC_MATERIALIZATION_SCHEMA_V1,
    CameraRole,
    CameraRoleBinding,
    CameraSemanticContractError,
    CameraSemanticResolution,
    ResolvedCameraRole,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.policy_artifact import (
    PolicyArtifactContract,
    PolicyCameraInput,
    PolicyCameraInputContract,
)
from dataset_contracts.v22 import EndEffectorSpec, PolicyIOLayout
from dataset_interface.camera_semantics import (
    CAMERA_ROLE_REGISTRY_ENV,
    CAMERA_SEMANTIC_MATERIALIZATION_RELATIVE_PATH,
    DatasetCameraSemanticResolver,
    load_camera_role_binding,
    resolve_camera_snapshot,
)
from dataset_interface.config import DatasetInterfaceConfig, MaskConfig
from dataset_interface.pipeline import MaskedFramePipeline
from dataset_interface.sample import FrameRequest
from deployments.openpi.camera_semantics import (
    LEGACY_SWAPPED_WRIST_CAMERA_SEMANTICS_V1,
    legacy_swapped_wrist_artifact,
    validate_declared_camera_semantics,
    validate_legacy_swapped_wrist_artifact,
)
from deployments.openpi.corrected_train import (
    _camera_audits,
    _dataset_roots,
    _split_args,
    _uses_wrist_cameras,
)
from deployments.openpi import corrected_train
from deployments.client import load_policy_artifact
from deployments.openpi.server import build_runtime_registry_entry
from deployments.openpi.training.adapter import (
    OPENPI_MODEL_RTC_SAMPLE_IDENTITY_KEY_V1,
    OpenPICameraSemanticTransform,
)
from deployments.openpi.training.launch import (
    _validate_checkpoint_camera_compatibility,
)
from utils.dataset_postprocess.wrist_cam_adjust.depth_mask.core import (
    WRIST_DEPTH_MASK_FORMAT,
    pack_binary_mask,
)
from utils.random_background import (
    PhotometricAugmentConfig,
    RandomBackgroundAugmentConfig,
)


_LEFT_SERIAL = "260522278055"
_RIGHT_SERIAL = "261122273924"


def _binding() -> CameraRoleBinding:
    return CameraRoleBinding(
        roles=(
            CameraRole("left_wrist", _LEFT_SERIAL),
            CameraRole("right_wrist", _RIGHT_SERIAL),
        )
    )


def _snapshot(*, corrected: bool) -> dict[str, object]:
    left_serial = _LEFT_SERIAL if corrected else _RIGHT_SERIAL
    right_serial = _RIGHT_SERIAL if corrected else _LEFT_SERIAL
    return {
        "left_wrist": {
            "camera_key": "left_wrist",
            "serial": left_serial,
            "serial_number": left_serial,
        },
        "right_wrist": {
            "camera_key": "right_wrist",
            "serial": right_serial,
            "serial_number": right_serial,
        },
    }


def _write_direct_dataset(root: Path, *, corrected: bool) -> None:
    devices = root / "meta" / "devices"
    devices.mkdir(parents=True)
    (devices / "realsense.json").write_text(
        json.dumps(_snapshot(corrected=corrected)),
        encoding="utf-8",
    )


def _artifact() -> PolicyArtifactContract:
    layout = PolicyIOLayout(
        name="camera_semantics_fixture",
        end_effectors={
            "left": EndEffectorSpec.none("left"),
            "right": EndEffectorSpec.none("right"),
        },
    )
    return PolicyArtifactContract(
        policy_io_layout=layout,
        model_id="camera-semantics-fixture",
        checkpoint_step=10,
        camera_input=PolicyCameraInputContract(
            inputs=(
                PolicyCameraInput(
                    source_key="left_wrist",
                    model_input_key="observation.images.left_wrist",
                    server_camera_slot="left_wrist_0_rgb",
                ),
                PolicyCameraInput(
                    source_key="right_wrist",
                    model_input_key="observation.images.right_wrist",
                    server_camera_slot="right_wrist_0_rgb",
                ),
            )
        ),
    )


def _write_video(path: Path, value: int) -> None:
    cv2 = pytest.importorskip("cv2")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        15.0,
        (8, 6),
    )
    assert writer.isOpened()
    try:
        writer.write(np.full((6, 8, 3), value, dtype=np.uint8))
    finally:
        writer.release()


def _write_mask(path: Path, value: int) -> None:
    mask = np.full((6, 8), value, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.parent / "final_keep.bitpack.bin"
    stream.write_bytes(pack_binary_mask(mask))
    path.write_text(
        json.dumps(
            {
                "format": WRIST_DEPTH_MASK_FORMAT,
                "camera_key": path.parent.name,
                "stream": {
                    "path": stream.name,
                    "encoding": "np.packbits",
                    "dtype": "uint1",
                    "shape": [6, 8],
                    "packed_bytes_per_frame": 6,
                    "frame_count": 1,
                },
            }
        ),
        encoding="utf-8",
    )


def _write_frame_index_episode(
    root: Path,
    *,
    left: list[int],
    right: list[int],
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    data = root / "data" / "episode_000000.parquet"
    data.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "observation.images.left_wrist.video_frame_index": left,
                "observation.images.right_wrist.video_frame_index": right,
            }
        ),
        data,
    )
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "episodes.jsonl").write_text(
        json.dumps(
            {
                "episode_index": 0,
                "frame_count": len(left),
                "parquet_path": str(data.relative_to(root)),
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_historical_and_corrected_snapshots_resolve_by_serial() -> None:
    historical = resolve_camera_snapshot(
        _snapshot(corrected=False),
        _binding(),
        source_id="historical",
    )
    corrected = resolve_camera_snapshot(
        _snapshot(corrected=True),
        _binding(),
        source_id="corrected",
    )

    assert historical.status == "swapped"
    assert historical.canonical_to_stored_key == {
        "left_wrist": "right_wrist",
        "right_wrist": "left_wrist",
    }
    assert corrected.status == "identity"
    assert corrected.canonical_to_stored_key == {
        "left_wrist": "left_wrist",
        "right_wrist": "right_wrist",
    }


def test_materialization_receipt_exposes_identity_and_bypasses_old_provenance(
    tmp_path: Path,
) -> None:
    root = tmp_path / "materialized"
    _write_direct_dataset(root, corrected=True)
    _write_frame_index_episode(root, left=[0, 1], right=[0, 1])
    (root / "meta" / "provenance.jsonl").write_text(
        json.dumps(
            {
                "output_episode_index": 0,
                "source_dataset_root": str(tmp_path / "unavailable_old_source"),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    snapshot_resolution = resolve_camera_snapshot(
        _snapshot(corrected=True),
        _binding(),
        source_id=root.name,
    )
    receipt = {
        "schema": CAMERA_SEMANTIC_MATERIALIZATION_SCHEMA_V1,
        "complete": True,
        "binding_hash": _binding().contract_hash,
        "output_snapshot_hash": snapshot_resolution.source_device_snapshot_hash,
        "episode_indices": [0],
    }
    receipt["receipt_hash"] = canonical_content_hash(
        receipt,
        domain=CAMERA_SEMANTIC_MATERIALIZATION_HASH_DOMAIN_V1,
    )
    receipt_path = root / CAMERA_SEMANTIC_MATERIALIZATION_RELATIVE_PATH
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    resolver = DatasetCameraSemanticResolver(root, _binding())

    assert resolver.resolution_for_episode(0).status == "identity"
    assert resolver.resolution_for_episode(0).canonical_to_stored_key == {
        "left_wrist": "left_wrist",
        "right_wrist": "right_wrist",
    }

    receipt["episode_indices"] = [0, 1]
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(CameraSemanticContractError, match="receipt hash mismatch"):
        DatasetCameraSemanticResolver(root, _binding())


def test_camera_role_binding_hash_is_independent_of_json_key_order() -> None:
    reversed_binding = CameraRoleBinding.from_dict(
        {
            "schema": "dexteleop.camera_role_binding.v1",
            "roles": {
                "right_wrist": {"serial": _RIGHT_SERIAL},
                "left_wrist": {"serial": _LEFT_SERIAL},
            },
        }
    )

    assert reversed_binding.roles == _binding().roles
    assert reversed_binding.contract_hash == _binding().contract_hash


def test_camera_contract_rejects_none_and_inconsistent_status() -> None:
    with pytest.raises(CameraSemanticContractError, match="non-empty"):
        CameraRole(None, _LEFT_SERIAL)  # type: ignore[arg-type]

    with pytest.raises(CameraSemanticContractError, match="disagrees"):
        CameraSemanticResolution(
            source_id="bad-status",
            source_device_snapshot_hash="snapshot",
            binding_hash="binding",
            roles=(
                ResolvedCameraRole("left_wrist", "right_wrist", _LEFT_SERIAL),
                ResolvedCameraRole("right_wrist", "left_wrist", _RIGHT_SERIAL),
            ),
            status="identity",
        )


def test_snapshot_rejects_missing_and_duplicate_required_serials() -> None:
    missing = _snapshot(corrected=True)
    missing.pop("right_wrist")
    with pytest.raises(CameraSemanticContractError, match="does not contain serial"):
        resolve_camera_snapshot(missing, _binding(), source_id="missing")

    duplicate = _snapshot(corrected=True)
    duplicate["duplicate"] = {
        "serial": _LEFT_SERIAL,
        "serial_number": _LEFT_SERIAL,
    }
    with pytest.raises(CameraSemanticContractError, match="appears under both"):
        resolve_camera_snapshot(duplicate, _binding(), source_id="duplicate")


def test_runtime_snapshot_serial_mismatch_fails_closed() -> None:
    snapshot = _snapshot(corrected=True)
    assert isinstance(snapshot["left_wrist"], dict)
    snapshot["left_wrist"]["serial_number"] = "unexpected"

    with pytest.raises(CameraSemanticContractError, match="differs from observed"):
        resolve_camera_snapshot(snapshot, _binding(), source_id="bad")


def test_direct_dataset_resolver_is_read_only(tmp_path: Path) -> None:
    dataset = tmp_path / "historical"
    _write_direct_dataset(dataset, corrected=False)
    (dataset / "meta" / "episodes.jsonl").write_text(
        json.dumps(
            {
                "episode_index": 0,
                "frame_count": 1,
                "video_paths": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    before = {
        path.relative_to(dataset): path.read_bytes()
        for path in dataset.rglob("*")
        if path.is_file()
    }

    resolver = DatasetCameraSemanticResolver(dataset, _binding())
    assert resolver.stored_key_for(123, "left_wrist") == "right_wrist"
    assert resolver.audit_summary()["resolutions"][0]["status"] == "swapped"

    after = {
        path.relative_to(dataset): path.read_bytes()
        for path in dataset.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_pipeline_environment_opt_in_is_lazy_and_has_explicit_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "historical"
    _write_direct_dataset(dataset, corrected=False)
    (dataset / "meta" / "episodes.jsonl").write_text(
        json.dumps(
            {"episode_index": 0, "frame_count": 1, "video_paths": {}}
        )
        + "\n",
        encoding="utf-8",
    )
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "cameras": {
                    "left_wrist": {"serial": _LEFT_SERIAL},
                    "right_wrist": {"serial": _RIGHT_SERIAL},
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv(CAMERA_ROLE_REGISTRY_ENV, str(registry))
    config = DatasetInterfaceConfig(
        dataset_root=dataset,
        mask=MaskConfig(final_mask_root=tmp_path / "unused"),
        augmentation=RandomBackgroundAugmentConfig(
            provider="black",
            edge_blur_px=0,
            photometric=PhotometricAugmentConfig(enabled=False),
        ),
    )
    pipeline = MaskedFramePipeline(config)
    try:
        assert pipeline.camera_semantics is None
        assert pipeline._stored_camera_key(0, "front") == "front"
        assert pipeline.camera_semantics is None
        assert pipeline._stored_camera_key(0, "left_wrist") == "right_wrist"
        assert pipeline.camera_semantics is not None
    finally:
        pipeline.close()

    monkeypatch.delenv(CAMERA_ROLE_REGISTRY_ENV)
    rollback = MaskedFramePipeline(config)
    try:
        assert rollback._stored_camera_key(0, "left_wrist") == "left_wrist"
        assert rollback.camera_semantics is None
    finally:
        rollback.close()


def test_masked_frame_pipeline_reads_video_and_mask_through_stored_key(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "historical"
    left_video = dataset / "videos/left.mp4"
    right_video = dataset / "videos/right.mp4"
    left_mask = dataset / "masks/left_wrist/manifest.json"
    right_mask = dataset / "masks/right_wrist/manifest.json"
    _write_video(left_video, 25)
    _write_video(right_video, 210)
    _write_mask(left_mask, 0)
    _write_mask(right_mask, 1)
    meta = dataset / "meta"
    meta.mkdir(parents=True)
    (meta / "episodes.jsonl").write_text(
        json.dumps(
            {
                "episode_index": 0,
                "frame_count": 1,
                "video_paths": {
                    "left_wrist": str(left_video.relative_to(dataset)),
                    "right_wrist": str(right_video.relative_to(dataset)),
                },
                "final_mask_paths": {
                    "left_wrist": str(left_mask.relative_to(dataset)),
                    "right_wrist": str(right_mask.relative_to(dataset)),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_direct_dataset(dataset, corrected=False)
    config = DatasetInterfaceConfig(
        dataset_root=dataset,
        mask=MaskConfig(final_mask_root=tmp_path / "unused"),
        camera_role_binding=_binding(),
        augmentation=RandomBackgroundAugmentConfig(
            provider="black",
            edge_blur_px=0,
            photometric=PhotometricAugmentConfig(enabled=False),
        ),
    )

    with MaskedFramePipeline(config) as pipeline:
        image, mask, video_path, mask_path = pipeline.read_raw(
            FrameRequest(
                episode_index=0,
                camera_key="left_wrist",
                frame_index=0,
            )
        )

    assert video_path == right_video.resolve()
    assert mask_path == right_mask.resolve()
    assert float(image.mean()) > 150.0
    assert np.all(mask == 1)


def test_organized_dataset_resolves_each_episode_source(tmp_path: Path) -> None:
    dataset = tmp_path / "organized"
    meta = dataset / "meta"
    old_devices = meta / "source_devices" / "old_source"
    new_devices = meta / "source_devices" / "new_source"
    old_devices.mkdir(parents=True)
    new_devices.mkdir(parents=True)
    (old_devices / "realsense.json").write_text(
        json.dumps(_snapshot(corrected=False)),
        encoding="utf-8",
    )
    (new_devices / "realsense.json").write_text(
        json.dumps(_snapshot(corrected=True)),
        encoding="utf-8",
    )
    (meta / "provenance.jsonl").write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {
                    "output_episode_index": 0,
                    "source_dataset_root": "/offline/old_source",
                },
                {
                    "output_episode_index": 1,
                    "source_dataset_root": "/offline/new_source",
                },
            )
        ),
        encoding="utf-8",
    )

    resolver = DatasetCameraSemanticResolver(dataset, _binding())

    assert resolver.resolution_for_episode(0).status == "swapped"
    assert resolver.stored_key_for(0, "left_wrist") == "right_wrist"
    assert resolver.resolution_for_episode(1).status == "identity"
    assert resolver.stored_key_for(1, "left_wrist") == "left_wrist"
    audit = resolver.audit_summary()
    assert audit["episode_count"] == 2
    assert {row["status"] for row in audit["resolutions"]} == {
        "identity",
        "swapped",
    }


def test_remapped_frame_index_alignment_fails_closed(tmp_path: Path) -> None:
    dataset = tmp_path / "old"
    _write_direct_dataset(dataset, corrected=False)
    _write_frame_index_episode(dataset, left=[0, 1], right=[0, 2])

    resolver = DatasetCameraSemanticResolver(dataset, _binding())
    with pytest.raises(CameraSemanticContractError, match="not row-aligned"):
        resolver.require_frame_index_alignment()

    _write_frame_index_episode(dataset, left=[0, 1], right=[0, 1])
    audit = resolver.require_frame_index_alignment()
    assert audit["aligned"] is True
    assert audit["remapped_episode_count"] == 1
    assert audit["remapped_row_count"] == 2


def test_legacy_artifact_preserves_checkpoint_native_model_slots(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "historical"
    _write_direct_dataset(dataset, corrected=False)
    base = _artifact()
    base_payload = base.to_dict()

    compat = legacy_swapped_wrist_artifact(base, _binding())

    assert base.to_dict() == base_payload
    assert compat.metadata["camera_semantics"] == (
        LEGACY_SWAPPED_WRIST_CAMERA_SEMANTICS_V1
    )
    assert compat.camera_input is not None
    assert compat.camera_input.image_key_map == {
        "right_wrist": "observation.images.left_wrist",
        "left_wrist": "observation.images.right_wrist",
    }
    assert compat.camera_input.server_camera_slots == (
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    )


@pytest.mark.parametrize(
    ("corrected", "expected_legacy_left", "expected_canonical_left"),
    (
        (False, 11, 22),
        (True, 22, 11),
    ),
)
def test_openpi_training_projection_handles_future_mixed_data(
    tmp_path: Path,
    corrected: bool,
    expected_legacy_left: int,
    expected_canonical_left: int,
) -> None:
    dataset = tmp_path / ("new" if corrected else "old")
    _write_direct_dataset(dataset, corrected=corrected)
    resolver = DatasetCameraSemanticResolver(dataset, _binding())
    stored_left = np.full((2, 3, 3), 11, dtype=np.uint8)
    stored_right = np.full((2, 3, 3), 22, dtype=np.uint8)
    sample = {
        OPENPI_MODEL_RTC_SAMPLE_IDENTITY_KEY_V1: np.asarray([0, 0]),
        "observation.images.left_wrist": stored_left,
        "observation.images.right_wrist": stored_right,
    }

    legacy = OpenPICameraSemanticTransform(
        artifact=legacy_swapped_wrist_artifact(_artifact(), _binding()),
        resolver=resolver,
    )(sample)
    canonical = OpenPICameraSemanticTransform(
        artifact=_artifact(),
        resolver=resolver,
    )(sample)

    assert np.all(
        legacy["observation.images.left_wrist"] == expected_legacy_left
    )
    assert np.all(
        canonical["observation.images.left_wrist"]
        == expected_canonical_left
    )
    assert np.all(
        legacy["observation.images.right_wrist"]
        == (11 if corrected else 22)
    )
    assert np.all(sample["observation.images.left_wrist"] == 11)
    assert np.all(sample["observation.images.right_wrist"] == 22)


def test_openpi_training_projection_rejects_missing_identity_and_media(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "old"
    _write_direct_dataset(dataset, corrected=False)
    transform = OpenPICameraSemanticTransform(
        artifact=_artifact(),
        resolver=DatasetCameraSemanticResolver(dataset, _binding()),
    )
    with pytest.raises(ValueError, match="augmentation_key"):
        transform({})
    with pytest.raises(ValueError, match="stored camera input is missing"):
        transform(
            {
                OPENPI_MODEL_RTC_SAMPLE_IDENTITY_KEY_V1: np.asarray([0, 0]),
                "observation.images.left_wrist": np.zeros((2, 2, 3)),
            }
        )


def test_checked_in_camera_configs_match_physical_serial_truth() -> None:
    root = Path(__file__).resolve().parents[1]
    binding = load_camera_role_binding(
        root / "robot_motion/configs/rm65_camera_registry.json"
    )
    realsense = json.loads(
        (root / "teleop_runtime/configs/devices/realsense.json").read_text(
            encoding="utf-8"
        )
    )
    umi = json.loads(
        (root / "teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert binding.role_to_serial == {
        "left_wrist": _LEFT_SERIAL,
        "right_wrist": _RIGHT_SERIAL,
    }
    assert realsense["left_wrist"]["serial"] == _LEFT_SERIAL
    assert realsense["right_wrist"]["serial"] == _RIGHT_SERIAL
    assert umi["sides"]["left"]["camera_serial"] == _LEFT_SERIAL
    assert umi["sides"]["right"]["camera_serial"] == _RIGHT_SERIAL


def test_checked_in_legacy_compat_artifact_is_bound_to_base_checkpoint() -> None:
    root = Path(__file__).resolve().parents[1]
    base = load_policy_artifact(
        root
        / "deployments/openpi/configs/model_rtc/0712-wrist-cube-92599.policy_artifact.json"
    )
    compat = load_policy_artifact(
        root
        / "deployments/openpi/configs/model_rtc/0712-wrist-cube-92599-legacy-wrist-compat.policy_artifact.json"
    )
    expected = legacy_swapped_wrist_artifact(base, _binding())

    assert compat.to_dict() == expected.to_dict()


def test_server_accepts_only_compat_artifact_bound_to_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "92599"
    checkpoint.mkdir()
    base = _artifact()
    base_path = checkpoint / "policy_artifact.json"
    base_path.write_text(json.dumps(base.to_dict()), encoding="utf-8")
    compat = legacy_swapped_wrist_artifact(base, _binding())
    compat_path = tmp_path / "compat.json"
    compat_path.write_text(json.dumps(compat.to_dict()), encoding="utf-8")

    model_id, entry = build_runtime_registry_entry(
        checkpoint,
        policy_artifact_json=compat_path,
    )

    assert model_id == base.model_id
    assert entry["checkpoint_dir"] == str(checkpoint)
    assert entry["camera_slots"] == [
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    ]

    unrelated = _artifact()
    unrelated_path = tmp_path / "unrelated.json"
    unrelated_path.write_text(json.dumps(unrelated.to_dict()), encoding="utf-8")
    with pytest.raises(ValueError, match="Legacy wrist compatibility route"):
        build_runtime_registry_entry(
            checkpoint,
            policy_artifact_json=unrelated_path,
        )


def test_compat_artifact_rejects_double_wrap_and_tampering() -> None:
    base = _artifact()
    compat = legacy_swapped_wrist_artifact(base, _binding())
    with pytest.raises(ValueError, match="already declares"):
        legacy_swapped_wrist_artifact(compat, _binding())

    assert compat.camera_input is not None
    tampered_metadata = replace(
        compat,
        metadata={
            **compat.metadata,
            "camera_input_routing_hash": "tampered",
        },
    )
    with pytest.raises(ValueError, match="routing_hash"):
        validate_declared_camera_semantics(tampered_metadata)

    tampered_model = replace(compat, model_id="wrong-model")
    with pytest.raises(ValueError, match="outside the declared"):
        validate_legacy_swapped_wrist_artifact(base, tampered_model)


def test_model_rtc_template_requires_checkpoint_bound_legacy_route() -> None:
    base = _artifact()
    compat = legacy_swapped_wrist_artifact(base, _binding())
    template = replace(
        compat,
        model_id="continued-model",
        checkpoint_step=None,
    )

    _validate_checkpoint_camera_compatibility(base, template)

    tampered = replace(
        template,
        metadata={
            **template.metadata,
            "base_policy_artifact_hash": "wrong-base",
        },
    )
    with pytest.raises(ValueError, match="checkpoint-compatible"):
        _validate_checkpoint_camera_compatibility(base, tampered)


def test_corrected_training_entrypoint_registers_future_dataclass_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "openpi"
    package_path = checkout / "src" / "openpi" / "__init__.py"
    package_path.parent.mkdir(parents=True)
    package_path.write_text("", encoding="utf-8")
    train_path = checkout / "scripts" / "train.py"
    train_path.parent.mkdir()
    train_path.write_text(
        """from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class Args:
    value: int = 19


received = None


def main(config):
    global received
    received = config
""",
        encoding="utf-8",
    )
    openpi_module = ModuleType("openpi")
    openpi_module.__file__ = str(package_path)
    monkeypatch.setitem(sys.modules, "openpi", openpi_module)
    module_name = "_dexteleop_openpi_corrected_train"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    config = object()

    corrected_train._run_openpi_train(config)

    loaded = sys.modules[module_name]
    assert loaded.Args().value == 19
    assert loaded.received is config
    assert Path.cwd() == caller


def test_corrected_training_helpers_audit_each_source(tmp_path: Path) -> None:
    old = tmp_path / "old"
    new = tmp_path / "new"
    _write_direct_dataset(old, corrected=False)
    _write_direct_dataset(new, corrected=True)
    _write_frame_index_episode(old, left=[0, 1], right=[0, 1])
    _write_frame_index_episode(new, left=[0, 1], right=[0, 1])
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "cameras": {
                    "left_wrist": {"serial": _LEFT_SERIAL},
                    "right_wrist": {"serial": _RIGHT_SERIAL},
                }
            }
        ),
        encoding="utf-8",
    )
    factory = SimpleNamespace(
        camera_slots=("left_wrist_0_rgb", "right_wrist_0_rgb"),
        dataset_sources=(
            SimpleNamespace(dataset_root=old),
            SimpleNamespace(dataset_root=new),
        ),
    )

    assert _split_args(["--dry-run", "--", "config"])[1] == ["config"]
    with pytest.raises(ValueError, match="Separate"):
        _split_args(["config"])
    assert _uses_wrist_cameras(factory)
    assert _dataset_roots(factory) == (old.resolve(), new.resolve())
    audits = _camera_audits(factory, registry)
    assert [
        row["camera_semantics"]["resolutions"][0]["status"]
        for row in audits
    ] == [
        "swapped",
        "identity",
    ]
    assert audits[0]["frame_index_alignment"]["remapped_row_count"] == 2
    assert audits[1]["frame_index_alignment"]["remapped_row_count"] == 0
