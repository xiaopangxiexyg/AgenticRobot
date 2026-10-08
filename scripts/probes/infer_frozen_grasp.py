"""Offline GPU work only: point segmentation or GraspGenX on retained arrays.

No camera, Gateway, robot SDK or execution imports. This separate probe lets
CPU planning replay share exactly the same inference, without another capture.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True)
    p.add_argument("--mode", choices=("segment", "infer"), required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--bundle", required=True)
    p.add_argument("--source")
    p.add_argument("--pixel", nargs=2, type=int)
    args = p.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONNOUSERSITE="1")
    sys.path.insert(0, args.repo)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(17)
    np.random.seed(17)
    if args.mode == "segment":
        from deployments.perception.sam2_grounding_dino import Sam2GroundingDinoModelSpec
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from PIL import Image
        spec = Sam2GroundingDinoModelSpec.from_asset_bundle(args.bundle)
        predictor = SAM2ImagePredictor(build_sam2(spec.sam2_model_config, str(spec.sam2_checkpoint_path), device="cuda"))
        with np.load(args.input, allow_pickle=False) as data:
            rgb = data["rgb"]
        if args.pixel is None:
            raise ValueError("Point segmentation needs --pixel")
        predictor.set_image(rgb)
        masks, scores, _ = predictor.predict(point_coords=np.array([args.pixel], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32), multimask_output=True)
        selected = np.asarray(masks[int(np.argmax(scores))], dtype=bool)
        np.save(out / "mask.npy", selected, allow_pickle=False)
        np.save(out / "rgb.npy", rgb, allow_pickle=False)
        overlay = rgb.copy()
        overlay[selected] = (overlay[selected]*.6 + np.array([255, 0, 255])*.4).astype(np.uint8)
        Image.fromarray(overlay).save(out / "mask.jpg")
        result = {"mode": "segment", "pixel_uv": args.pixel, "mask_pixels": int(selected.sum()), "scores": np.asarray(scores).tolist()}
    else:
        from deployments.perception.graspgenx_client import GraspGenXInProcessBackend
        from deployments.perception.sam2_graspgenx_point_grasp import GraspGenXPointGraspModelSpec, DMTAC_M_GRASPGENX_SWEEP_VOLUME_V1
        if args.source is None:
            raise ValueError("Inference needs --source")
        spec = GraspGenXPointGraspModelSpec.from_asset_bundle(args.bundle)
        backend = GraspGenXInProcessBackend(model_spec=spec, source_root=args.source, device="cuda")
        with np.load(args.input, allow_pickle=False) as data:
            points = np.asarray(data["points"], dtype=np.float32)
        backend.start()
        try:
            poses, scores = backend.infer_object(points, sweep_volume_params=DMTAC_M_GRASPGENX_SWEEP_VOLUME_V1,
                planner="graspmoe", num_grasps=200, score_threshold=-1.0, top_k=-1)
            np.savez_compressed(out / "inference.npz", points=points, poses=poses, scores=scores)
            result = {"mode": "infer", "point_count": len(points), "raw_grasp_count": len(poses)}
        finally:
            backend.close()
    result.update(elapsed_sec=time.perf_counter()-started, hardware_connected=False)
    (out / "result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
