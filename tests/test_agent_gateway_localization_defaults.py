"""Default selection is SAM3; capability enablement and ownership stay explicit."""
from types import SimpleNamespace

import pytest

from agent_gateway.profiles.mujoco import gateway_cli as sim
from agent_gateway.profiles.real import gateway_cli as real, perception_cli as camera

pytestmark = [pytest.mark.unit, pytest.mark.contract]


@pytest.mark.parametrize("module", [sim, real, camera])
def test_default_is_sam3_but_existing_capabilities_are_still_off(module):
    parser = module.build_parser()
    assert parser.get_default("object_localization_backend") == "sam3-service"
    assert not parser.get_default("sam3_multi_instance_text")
    if module is sim:
        # None distinguishes an explicitly requested endpoint (which also enables
        # segmentation) from the effective default, resolved only when enabled.
        assert parser.get_default("sam3_endpoint") is None
        assert not parser.get_default("enable_segmentation")
        assert not parser.get_default("enable_fast_grasp")
    elif module is real:
        assert not parser.get_default("enable_point_grasp")
    else:
        assert not parser.get_default("enable_object_localization")


def sim_args(*flags):
    parser = sim.build_parser()
    return parser, parser.parse_args([
        "--socket-path", "/unused.sock", "--task-config", "/unused-task.json", *flags,
    ])


@pytest.mark.parametrize("flag", ["--enable-segmentation", "--enable-object-localization",
                                "--enable-segmentation-batch-v2", "--enable-segmentation-geometry-v3"])
def test_sim_enabled_default_constructs_sam3_without_starting_or_loading_sam2(monkeypatch, flag):
    from deployments.perception import sam2_grounding_dino, sam3_service
    monkeypatch.setattr(sam2_grounding_dino.Sam2GroundingDinoModelSpec, "from_asset_bundle",
                        lambda *_: pytest.fail("SAM2 must not be used by default"))
    monkeypatch.setattr(sam3_service.Sam3ServiceLocalizer, "_call",
                        lambda *_: pytest.fail("CLI construction must not contact service"))
    parser, args = sim_args(flag, "--artifact-root", "/unused")
    assert sim._segmentation_requested(args)
    localizer = sim._localizer_from_args(parser, args, bundle=None, device="cuda")
    assert isinstance(localizer, sam3_service.Sam3ServiceLocalizer)
    assert localizer.model_spec.endpoint == sam3_service.DEFAULT_SAM3_ENDPOINT
    assert not localizer.started
    assert not localizer.multi_instance_text
    localizer.close()


def test_sim_multi_instance_requires_and_reuses_existing_localization_route():
    parser, args = sim_args("--enable-object-localization", "--sam3-multi-instance-text")
    localizer = sim._localizer_from_args(parser, args, bundle=None, device="cuda")
    assert localizer.multi_instance_text
    assert not localizer.started
    localizer.close()

    parser, args = sim_args("--sam3-multi-instance-text")
    with pytest.raises(SystemExit):
        sim._localizer_from_args(parser, args, bundle=None, device="cuda")
    parser, args = sim_args("--enable-object-localization", "--sam3-multi-instance-text",
                            "--object-localization-backend", "sam2-grounding-dino")
    with pytest.raises(SystemExit):
        sim._localizer_from_args(parser, args, bundle=None, device="cuda")


def test_sim_explicit_sam2_keeps_original_bundle_and_device(monkeypatch, tmp_path):
    from deployments.perception import sam2_grounding_dino, sam3_service
    spec = object()
    monkeypatch.setattr(sam3_service, "Sam3ServiceLocalizer",
                        lambda **_: pytest.fail("Explicit SAM2 must not construct SAM3"))
    monkeypatch.setattr(sam2_grounding_dino.Sam2GroundingDinoModelSpec, "from_asset_bundle",
                        lambda path: spec if path == tmp_path else pytest.fail("bundle differs"))
    monkeypatch.setattr(sam2_grounding_dino, "Sam2GroundingDinoLocalizer", lambda **kw: kw)
    parser, args = sim_args("--object-localization-backend", "sam2-grounding-dino")
    assert sim._localizer_from_args(parser, args, bundle=tmp_path, device="cpu") == {
        "model_spec": spec, "device": "cpu"}


def test_sim_default_does_not_silently_accept_sam2_assets(tmp_path):
    parser, args = sim_args()
    with pytest.raises(SystemExit) as error:
        sim._localizer_from_args(parser, args, bundle=tmp_path, device="cuda")
    assert error.value.code == 2


def test_sim_default_service_failure_does_not_fallback(monkeypatch):
    from deployments.perception import sam3_service
    def unavailable(*_):
        raise RuntimeError("test service unavailable")
    monkeypatch.setattr(sam3_service.Sam3ServiceLocalizer, "_call", unavailable)
    parser, args = sim_args()
    localizer = sim._localizer_from_args(parser, args, bundle=None, device="cuda")
    with pytest.raises(RuntimeError, match="test service unavailable"):
        localizer.start()
    localizer.close()


@pytest.mark.parametrize("mode", ["in-process", "service"])
@pytest.mark.parametrize("generator", ["graspmoe", "diffusion"])
def test_sim_fast_grasp_default_reuses_service_masks_without_sam2_bundle(monkeypatch, tmp_path, mode, generator):
    from deployments.perception import graspgenx_client, sam2_graspgenx_point_grasp, sam3_service
    spec = object()
    monkeypatch.setattr(sam2_graspgenx_point_grasp.GraspGenXPointGraspModelSpec,
                        "from_asset_bundle", lambda _: spec)
    monkeypatch.setattr(graspgenx_client, "GraspGenXInProcessBackend", lambda **kw: kw)
    monkeypatch.setattr(graspgenx_client, "GraspGenXClient", lambda **kw: kw)
    monkeypatch.setattr(sam2_graspgenx_point_grasp, "Sam2GraspGenXPointGraspPlanner",
                        lambda **kw: SimpleNamespace(**kw))
    source_flags = ["--graspgenx-source-root", str(tmp_path)] if mode == "in-process" else []
    parser, args = sim_args("--enable-fast-grasp", "--enable-grasp-approach", "--enable-grippers",
        "--grasp-contact-target-body", "debug_cube", "--no-realtime-worker",
        "--artifact-root", str(tmp_path), "--graspgenx-point-grasp-bundle", str(tmp_path),
        "--graspgenx-mode", mode, "--graspgenx-planner", generator, *source_flags)
    planner = sim._fast_grasp_model_from_args(parser, args)
    assert isinstance(planner.point_segmenter, sam3_service.Sam3ServiceLocalizer)
    assert not planner.point_segmenter.started
    assert planner.planner == generator
    if mode == "in-process":
        assert planner.client["source_root"] == tmp_path
    else:
        assert planner.client["port"] == 5556
    planner.point_segmenter.close()


def test_sim_default_localization_has_no_sam2_bundle_dependency(monkeypatch, tmp_path):
    def after_dependencies(*_):
        raise RuntimeError("passed model dependency checks")
    monkeypatch.setattr(sim, "_absolute_existing_file", after_dependencies)
    with pytest.raises(RuntimeError, match="passed model dependency checks"):
        sim.main(["--socket-path", "/unused.sock", "--task-config", "/unused-task.json",
                  "--enable-object-localization", "--artifact-root", str(tmp_path)])


def test_explicit_sam2_segment_requires_bundle_before_any_path_or_model_read(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sim, "_absolute_existing_file",
                        lambda *_: pytest.fail("Missing model dependency must fail before task read"))
    with pytest.raises(SystemExit) as error:
        sim.main(["--socket-path", "/unused.sock", "--task-config", "/unused-task.json",
                  "--enable-segmentation", "--object-localization-backend", "sam2-grounding-dino",
                  "--artifact-root", str(tmp_path)])
    assert error.value.code == 2
    assert "SAM2 segmentation requires --segmentation-asset-bundle" in capsys.readouterr().err
