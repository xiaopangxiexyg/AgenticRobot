import json
import subprocess
import sys
from pathlib import Path

from types import SimpleNamespace

import pytest

from scripts import smoke_sam2_grounding_dino as probe


@pytest.mark.parametrize("filename", ["model.safetensors", "pytorch_model.bin"])
def test_weights_must_be_an_existing_local_file(tmp_path, filename):
    weights = tmp_path / filename
    weights.write_bytes(b"weights")
    calls = []
    def lookup(model, name, *, cache_dir):
        calls.append((model, name, cache_dir))
        return str(weights) if name == filename else object()
    assert probe._cached_weights("encoder", str(tmp_path), lookup) == str(weights)
    assert all(call[0] == "encoder" and call[2] == str(tmp_path) for call in calls)


@pytest.mark.parametrize("missing", [None, object(), "/nonexistent/weights"])
def test_missing_cache_never_downloads_or_substitutes_weights(tmp_path, missing):
    with pytest.raises(FileNotFoundError, match="no local model weights"):
        probe._cached_weights("encoder", str(tmp_path), lambda *a, **k: missing)


def test_cuda_preflight_uses_native_operator_and_checks_finiteness():
    class Output:
        shape = (1, 2, 4)
    calls = []
    tensor = object()
    torch = SimpleNamespace(rand=lambda *a, **k: tensor, tensor=lambda *a, **k: tensor,
        ones=lambda *a, **k: tensor, int64=object(),
        isfinite=lambda output: SimpleNamespace(all=lambda: True))
    extension = SimpleNamespace(ms_deform_attn_forward=lambda *a: (calls.append(a) or Output()))
    probe._cuda_operator(torch, extension, "cuda")
    assert len(calls) == 1 and calls[0][-1] == 1
    torch.isfinite = lambda output: SimpleNamespace(all=lambda: False)
    with pytest.raises(RuntimeError, match="operator preflight failed"):
        probe._cuda_operator(torch, extension, "cuda")


def test_failure_receipt_preserves_phase_before_any_device_or_model_start(monkeypatch, tmp_path):
    from deployments.perception import sam2_grounding_dino as owner
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    def missing(*args, **kwargs):
        raise FileNotFoundError("missing_manifest")
    monkeypatch.setattr(owner.Sam2GroundingDinoModelSpec, "from_asset_bundle", missing)
    receipt = probe.run(SimpleNamespace(asset_bundle=tmp_path, device="cuda", verify_only=True))
    assert receipt["status"] == "failed" and receipt["phase"] == "asset_manifest"
    assert receipt["error"] == {"type": "FileNotFoundError", "message": "missing_manifest"}
    assert not receipt["devices_started"] and not receipt["model_loaded"]


def test_documented_direct_script_resolves_worktree_without_pythonpath(tmp_path):
    root = Path(__file__).resolve().parents[1]
    missing = tmp_path / "missing-model-bundle"
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(root / "scripts/smoke_sam2_grounding_dino.py"),
         "--asset-bundle", str(missing), "--verify-only", "--device", "cpu"],
        cwd=tmp_path, capture_output=True, text=True, timeout=30., check=False,
    )
    assert result.returncode == 1
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "failed" and receipt["phase"] == "asset_manifest"
    assert receipt["error"]["type"] == "FileNotFoundError"
    assert missing.name in receipt["error"]["message"]
    assert not receipt["devices_started"] and not receipt["model_loaded"]
