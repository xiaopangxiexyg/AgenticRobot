#!/usr/bin/env python3
"""Preflight reviewed localization models offline, without starting devices."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

# Direct script execution must resolve this worktree without an editable install.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _cached_weights(model_id, cache_dir, lookup):
    for filename in ("model.safetensors", "pytorch_model.bin"):
        result = lookup(model_id, filename, cache_dir=cache_dir)
        if isinstance(result, str) and Path(result).is_file():
            return result
    raise FileNotFoundError(
        f"Text encoder {model_id!r} has no local model weights in {cache_dir}. "
        "Bind the reviewed HF cache before starting the Gateway; no download was attempted."
    )


def _cuda_operator(torch, extension, device):
    value = torch.rand(1, 4, 1, 4, device=device)
    shapes = torch.tensor([[2, 2]], dtype=torch.int64, device=device)
    starts = torch.tensor([0], dtype=torch.int64, device=device)
    locations = torch.rand(1, 2, 1, 1, 1, 2, device=device)
    weights = torch.ones(1, 2, 1, 1, 1, device=device)
    output = extension.ms_deform_attn_forward(value, shapes, starts, locations, weights, 1)
    if tuple(output.shape) != (1, 2, 4) or not bool(torch.isfinite(output).all()):
        raise RuntimeError("Grounding DINO CUDA operator preflight failed.")


def run(args):
    # Set policy before importing Transformers/Hub, which cache these settings.
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    receipt = {"schema": "agenticrobot.localization_preflight.v1", "status": "failed",
               "phase": "asset_manifest", "devices_started": False, "model_loaded": False}
    localizer = None
    try:
        from deployments.perception.sam2_grounding_dino import (
            Sam2GroundingDinoLocalizer, Sam2GroundingDinoModelSpec, _installed_git_revision,
        )
        spec = Sam2GroundingDinoModelSpec.from_asset_bundle(args.asset_bundle)
        receipt["phase"] = "reviewed_sources"
        import groundingdino
        import sam2
        for package, name, revision in (
            (groundingdino, "groundingdino", spec.grounding_dino_source_revision),
            (sam2, "sam2", spec.sam2_source_revision),
        ):
            actual = _installed_git_revision(package.__file__, package_name=name)
            if actual != revision:
                raise RuntimeError(f"{name} revision differs from the reviewed asset manifest.")
        from groundingdino.util.slconfig import SLConfig
        config = Path(spec.grounding_dino_model_config)
        if not config.is_file():
            config = Path(groundingdino.__file__).resolve().parent / "config" / config.name
        encoder = SLConfig.fromfile(str(config)).text_encoder_type
        receipt["phase"] = "offline_text_encoder"
        from transformers import AutoConfig, AutoTokenizer
        from transformers.utils.hub import TRANSFORMERS_CACHE
        from huggingface_hub import try_to_load_from_cache
        AutoConfig.from_pretrained(encoder, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(encoder, local_files_only=True)
        if not tokenizer("preflight")["input_ids"]:
            raise RuntimeError("Text encoder tokenizer returned no token IDs.")
        receipt["text_encoder"] = encoder
        receipt["cache_dir"] = str(TRANSFORMERS_CACHE)
        receipt["text_encoder_weights"] = _cached_weights(
            encoder, TRANSFORMERS_CACHE, try_to_load_from_cache)
        receipt["phase"] = "detector_operator"
        import torch
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; no automatic CPU fallback.")
            extension = importlib.import_module("groundingdino._C")
            _cuda_operator(torch, extension, device)
            receipt["detector_extension"] = str(extension.__file__)
        receipt["torch"] = torch.__version__
        receipt["device"] = str(device)
        receipt["cxx11_abi"] = bool(torch._C._GLIBCXX_USE_CXX11_ABI)
        if not args.verify_only:
            receipt["phase"] = "load_models"
            localizer = Sam2GroundingDinoLocalizer(model_spec=spec, device=args.device)
            localizer.start()
            receipt["model_loaded"] = True
        receipt.update(status="passed", phase="complete")
    except Exception as exc:
        receipt["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if localizer is not None:
            localizer.close()
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-bundle", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verify-only", action="store_true",
        help="Check weights, offline text encoder, source identity and native operator without loading detector models")
    args = parser.parse_args(argv)
    receipt = run(args)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
