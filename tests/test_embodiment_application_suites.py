from __future__ import annotations

from dataclasses import replace
import json

import pytest

from teleop_runtime.target_runtime import (
    EMBODIMENT_APPLICATION_SUITE_SPEC_SCHEMA_V1,
    EmbodimentApplicationSurfaceSpec,
    EmbodimentApplicationSuiteBinding,
    EmbodimentApplicationSuiteCatalog,
    EmbodimentApplicationSuiteSpec,
    LoadedEmbodimentApplicationSurface,
)


class _OpaqueApplicationImplementation:
    def __init__(self, name: str) -> None:
        self.name = name
        self.invocation_count = 0

    def invoke(self) -> None:
        self.invocation_count += 1
        raise AssertionError("The suite catalog must not invoke applications.")


def _surface(
    surface_id: str,
    *,
    surface_kind: str,
    target_profile_id: str,
    target_profile_hash: str,
    required_assets: tuple[str, ...],
) -> EmbodimentApplicationSurfaceSpec:
    return EmbodimentApplicationSurfaceSpec(
        surface_id=surface_id,
        surface_kind=surface_kind,
        application_contract_id=f"contract:{surface_id}:v1",
        application_contract_hash=f"contract-hash:{surface_id}:v1",
        target_runtime_profile_id=target_profile_id,
        target_runtime_profile_hash=target_profile_hash,
        required_asset_bundle_ids=required_assets,
    )


def _suite() -> EmbodimentApplicationSuiteSpec:
    return EmbodimentApplicationSuiteSpec(
        suite_id="fixture-mobile-manipulator-suite-v1",
        suite_version="1",
        embodiment_definition_id="fixture-mobile-manipulator-v1",
        embodiment_definition_hash="fixture-mobile-manipulator-definition-hash-v1",
        integration_package_name="dexteleop-fixture-mobile-manipulator",
        integration_package_version="2.1.0",
        suite_contract_hash="fixture-suite-contract-hash-v1",
        surfaces=(
            _surface(
                "fixture.policy-deployment",
                surface_kind="policy_deployment",
                target_profile_id="fixture-policy-target-v1",
                target_profile_hash="fixture-policy-target-hash-v1",
                required_assets=("controller-map", "robot-description"),
            ),
            _surface(
                "fixture.gui-teleop-collection",
                surface_kind="gui_teleop_collection",
                target_profile_id="fixture-human-target-v1",
                target_profile_hash="fixture-human-target-hash-v1",
                required_assets=("robot-description", "teleop-calibration"),
            ),
        ),
        asset_bundle_hash_by_id={
            "robot-description": "fixture-robot-description-hash-v1",
            "controller-map": "fixture-controller-map-hash-v1",
            "teleop-calibration": "fixture-teleop-calibration-schema-hash-v1",
        },
    )


def _loaded(
    suite: EmbodimentApplicationSuiteSpec,
    surface_id: str,
    implementation: object,
) -> LoadedEmbodimentApplicationSurface:
    surface = suite.surface_by_id[surface_id]
    return LoadedEmbodimentApplicationSurface(
        suite_id=suite.suite_id,
        suite_spec_hash=suite.semantic_content_hash,
        surface_id=surface.surface_id,
        surface_spec_hash=surface.semantic_content_hash,
        application_contract_hash=surface.application_contract_hash,
        implementation=implementation,
    )


def test_suite_spec_round_trips_without_loader_or_module_path() -> None:
    suite = _suite()

    payload = json.loads(json.dumps(suite.to_payload()))
    restored = EmbodimentApplicationSuiteSpec.from_payload(payload)

    assert restored == suite
    assert restored.schema == EMBODIMENT_APPLICATION_SUITE_SPEC_SCHEMA_V1
    assert tuple(restored.surface_by_id) == (
        "fixture.gui-teleop-collection",
        "fixture.policy-deployment",
    )
    assert tuple(restored.asset_bundle_hash_by_id) == (
        "controller-map",
        "robot-description",
        "teleop-calibration",
    )
    encoded = json.dumps(payload, sort_keys=True)
    assert "loader" not in encoded
    assert "module_path" not in encoded
    assert "implementation" not in encoded


def test_suite_hash_is_independent_of_surface_and_asset_input_order() -> None:
    suite = _suite()
    reordered = replace(
        suite,
        surfaces=tuple(reversed(suite.surfaces)),
        asset_bundle_hash_by_id=dict(
            reversed(tuple(suite.asset_bundle_hash_by_id.items()))
        ),
    )

    assert reordered == suite
    assert reordered.semantic_content_hash == suite.semantic_content_hash


def test_catalog_lists_and_selects_without_loading_any_surface() -> None:
    suite = _suite()
    load_count = {surface_id: 0 for surface_id in suite.surface_by_id}

    def loader(surface_id: str):
        def load() -> LoadedEmbodimentApplicationSurface:
            load_count[surface_id] += 1
            return _loaded(
                suite,
                surface_id,
                _OpaqueApplicationImplementation(surface_id),
            )

        return load

    binding = EmbodimentApplicationSuiteBinding(
        suite_spec=suite,
        surface_loader_by_id={
            surface_id: loader(surface_id) for surface_id in suite.surface_by_id
        },
    )
    catalog = EmbodimentApplicationSuiteCatalog((binding,))

    assert catalog.suite_ids() == (suite.suite_id,)
    assert catalog.suite_specs() == (suite,)
    assert catalog.select(suite.suite_id) is binding
    assert load_count == {
        "fixture.gui-teleop-collection": 0,
        "fixture.policy-deployment": 0,
    }


def test_catalog_loads_only_the_selected_complete_surface_without_invoking_it() -> None:
    suite = _suite()
    gui_implementation = _OpaqueApplicationImplementation("gui")
    policy_implementation = _OpaqueApplicationImplementation("policy")
    load_count = {"gui": 0, "policy": 0}

    def load_gui() -> LoadedEmbodimentApplicationSurface:
        load_count["gui"] += 1
        return _loaded(
            suite,
            "fixture.gui-teleop-collection",
            gui_implementation,
        )

    def load_policy() -> LoadedEmbodimentApplicationSurface:
        load_count["policy"] += 1
        return _loaded(
            suite,
            "fixture.policy-deployment",
            policy_implementation,
        )

    catalog = EmbodimentApplicationSuiteCatalog(
        (
            EmbodimentApplicationSuiteBinding(
                suite_spec=suite,
                surface_loader_by_id={
                    "fixture.gui-teleop-collection": load_gui,
                    "fixture.policy-deployment": load_policy,
                },
            ),
        )
    )

    loaded = catalog.load_surface(
        suite.suite_id,
        "fixture.gui-teleop-collection",
    )

    assert loaded.implementation is gui_implementation
    assert load_count == {"gui": 1, "policy": 0}
    assert gui_implementation.invocation_count == 0
    assert policy_implementation.invocation_count == 0


def test_binding_requires_an_exact_loader_set_before_any_load() -> None:
    suite = _suite()
    surface_id = "fixture.gui-teleop-collection"

    with pytest.raises(ValueError, match="exactly match"):
        EmbodimentApplicationSuiteBinding(
            suite_spec=suite,
            surface_loader_by_id={
                surface_id: lambda: _loaded(suite, surface_id, object()),
            },
        )


def test_binding_rejects_normalized_loader_id_collisions() -> None:
    suite = _suite()
    gui_id = "fixture.gui-teleop-collection"
    policy_id = "fixture.policy-deployment"

    with pytest.raises(ValueError, match="unique after normalization"):
        EmbodimentApplicationSuiteBinding(
            suite_spec=suite,
            surface_loader_by_id={
                gui_id: lambda: _loaded(suite, gui_id, object()),
                f" {gui_id} ": lambda: _loaded(suite, gui_id, object()),
                policy_id: lambda: _loaded(suite, policy_id, object()),
            },
        )


def test_direct_surface_constructor_rejects_string_asset_collection() -> None:
    with pytest.raises(TypeError, match="iterable of identifiers"):
        _surface(
            "fixture.invalid-assets",
            surface_kind="gui_teleop_collection",
            target_profile_id="fixture-target-v1",
            target_profile_hash="fixture-target-hash-v1",
            required_assets="robot-description",  # type: ignore[arg-type]
        )


def test_loaded_surface_identity_drift_fails_closed() -> None:
    suite = _suite()
    surface_id = "fixture.gui-teleop-collection"
    correct = _loaded(suite, surface_id, object())
    drifted = replace(
        correct,
        application_contract_hash="another-application-contract-hash",
    )
    binding = EmbodimentApplicationSuiteBinding(
        suite_spec=suite,
        surface_loader_by_id={
            surface.surface_id: (
                (lambda value=drifted: value)
                if surface.surface_id == surface_id
                else (
                    lambda selected=surface.surface_id: _loaded(
                        suite,
                        selected,
                        object(),
                    )
                )
            )
            for surface in suite.surfaces
        },
    )

    with pytest.raises(ValueError, match="application_contract_hash"):
        EmbodimentApplicationSuiteCatalog((binding,)).load_surface(
            suite.suite_id,
            surface_id,
        )


def test_catalog_rejects_invalid_loader_result() -> None:
    suite = _suite()
    selected_id = "fixture.gui-teleop-collection"
    binding = EmbodimentApplicationSuiteBinding(
        suite_spec=suite,
        surface_loader_by_id={
            surface.surface_id: (
                (lambda: object())
                if surface.surface_id == selected_id
                else (
                    lambda surface_id=surface.surface_id: _loaded(
                        suite,
                        surface_id,
                        object(),
                    )
                )
            )
            for surface in suite.surfaces
        },
    )

    with pytest.raises(TypeError, match="must return"):
        EmbodimentApplicationSuiteCatalog((binding,)).load_surface(
            suite.suite_id,
            selected_id,
        )


def test_suite_rejects_undeclared_assets_duplicate_surfaces_and_tampering() -> None:
    suite = _suite()
    unknown_asset_surface = replace(
        suite.surfaces[0],
        required_asset_bundle_ids=("not-declared",),
    )
    with pytest.raises(ValueError, match="undeclared asset"):
        replace(suite, surfaces=(unknown_asset_surface, suite.surfaces[1]))
    with pytest.raises(ValueError, match="surface IDs must be unique"):
        replace(suite, surfaces=(suite.surfaces[0], suite.surfaces[0]))

    payload = json.loads(json.dumps(suite.to_payload()))
    payload["integration_package_version"] = "9.9.9"
    with pytest.raises(ValueError, match="semantic hash mismatch"):
        EmbodimentApplicationSuiteSpec.from_payload(payload)

    extra_field = json.loads(json.dumps(suite.to_payload()))
    extra_field["module_path"] = "fixture.unsupported.dynamic_import"
    with pytest.raises(ValueError, match="extra=.*module_path"):
        EmbodimentApplicationSuiteSpec.from_payload(extra_field)


def test_catalog_rejects_unknown_suite_or_surface_without_loading() -> None:
    suite = _suite()
    load_count = 0

    def load(surface_id: str):
        def selected() -> LoadedEmbodimentApplicationSurface:
            nonlocal load_count
            load_count += 1
            return _loaded(suite, surface_id, object())

        return selected

    binding = EmbodimentApplicationSuiteBinding(
        suite_spec=suite,
        surface_loader_by_id={
            surface_id: load(surface_id) for surface_id in suite.surface_by_id
        },
    )
    catalog = EmbodimentApplicationSuiteCatalog((binding,))

    with pytest.raises(KeyError, match="Unknown embodiment application suite"):
        catalog.load_surface("not-registered", "any-surface")
    with pytest.raises(KeyError, match="has no surface"):
        catalog.load_surface(suite.suite_id, "not-registered")
    assert load_count == 0
