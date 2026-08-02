from __future__ import annotations

import importlib

import pytest


MIGRATED_MODULES = (
    "async_writer",
    "dataset_episode_maintenance",
    "immutable_canonical_store",
    "stream_sink",
    "v25p_component_action",
    "v25p_native_store",
    "v25p_parquet_store",
    "v25p_producer",
    "v25p_sidecar",
)
pytestmark = [pytest.mark.unit, pytest.mark.contract]


@pytest.mark.parametrize("module_name", MIGRATED_MODULES)
def test_legacy_collection_api_reexports_canonical_objects(module_name: str) -> None:
    canonical = importlib.import_module(f"collection_runtime.{module_name}")
    legacy = importlib.import_module(f"teleop_runtime.collection.{module_name}")

    canonical_all = getattr(canonical, "__all__", None)
    if canonical_all is None:
        assert not hasattr(legacy, "__all__")
        public_names = tuple(
            name for name in vars(canonical) if not name.startswith("_")
        )
    else:
        assert tuple(legacy.__all__) == tuple(canonical_all)
        public_names = tuple(canonical_all)
    for public_name in public_names:
        assert getattr(legacy, public_name) is getattr(canonical, public_name)


def test_legacy_real_teleop_dataset_layout_reexports_canonical_objects() -> None:
    canonical = importlib.import_module("collection_runtime.dataset_layout")
    legacy = importlib.import_module("teleop_runtime.real_teleop.dataset_layout")

    assert tuple(legacy.__all__) == tuple(canonical.__all__)
    for public_name in canonical.__all__:
        assert getattr(legacy, public_name) is getattr(canonical, public_name)


def test_versioned_component_capture_shims_preserve_object_identity() -> None:
    canonical = importlib.import_module("collection_runtime.component_capture")
    old_action = importlib.import_module("collection_runtime.v25p_component_action")
    old_native = importlib.import_module("collection_runtime.v25p_native_store")
    old_parquet = importlib.import_module("collection_runtime.v25p_parquet_store")
    old_producer = importlib.import_module("collection_runtime.v25p_producer")

    assert (
        old_action.V25pActionComponentTiming
        is canonical.ComponentActionTiming
    )
    assert old_native.V25pComponentInput is canonical.ComponentStreamInput
    assert (
        old_native.V25pNativeEpisodeStore
        is canonical.ComponentCaptureEpisodeStore
    )
    assert (
        old_native.V25pNativeSealReport
        is canonical.ComponentCaptureSealReport
    )
    assert (
        old_native.V25pNativeCommitReport
        is canonical.ComponentCaptureCommitReport
    )
    assert (
        old_native.validate_native_stream_profile
        is canonical.validate_component_stream_profile
    )
    assert (
        old_parquet.V25pParquetChunkEpisodeWriter
        is canonical.ParquetChunkEpisodeWriter
    )
    assert old_producer.V25pProducerBindings is canonical.ComponentStreamBindings
    assert old_producer.V25pProducedSample is canonical.ProducedComponentSample


def test_versioned_sidecar_shim_preserves_object_identity() -> None:
    canonical = importlib.import_module("collection_runtime.compatibility")
    old = importlib.import_module("collection_runtime.v25p_sidecar")

    assert (
        old.V25pEpisodeSidecarFinalizeReport
        is canonical.CompatibilityEpisodeSidecarFinalizeReport
    )
    assert (
        old.V25pEpisodeSidecarStore
        is canonical.CompatibilityEpisodeSidecarStore
    )
    assert (
        old.V25pV23CompatibilitySidecarStore
        is canonical.CompatibilityEpisodeSidecarStore
    )


def test_capability_named_packages_do_not_export_versioned_runtime_names() -> None:
    component_capture = importlib.import_module(
        "collection_runtime.component_capture"
    )
    compatibility = importlib.import_module("collection_runtime.compatibility")

    assert all("V25p" not in name for name in component_capture.__all__)
    assert all("V25p" not in name for name in compatibility.__all__)


def test_umi_snapshot_rejection_reexports_collection_runtime_contract() -> None:
    canonical = importlib.import_module("collection_runtime.source_cut")
    umi_types = importlib.import_module("teleop_runtime.collection.umi.types")

    assert umi_types.SnapshotRejection is canonical.SnapshotRejection
