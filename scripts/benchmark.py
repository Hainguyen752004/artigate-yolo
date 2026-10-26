"""Parameters, GFLOPs and forward latency for model YAMLs or checkpoints (same settings for every row).

    python scripts/benchmark.py --models configs/models/*.yaml --device 0
    python scripts/benchmark.py --models runs/segment/A4_s0/weights/best.pt --device 0 --half
    python scripts/benchmark.py --baselines unet malunet egeunet --device 0 --half
    python scripts/benchmark.py --nnunet-model-dir $nnUNet_results/Dataset501_ISIC2018Canvas/nnUNetTrainer__nnUNetPlans__2d --device 0
Forward pass only (no NMS or mask decoding), batch 1, fused, after warm-up. YOLO GFLOPs come from
Ultralytics' get_flops; the baselines are profiled with the same counter (thop, conv/linear layers) on a
640x640 input. nnU-Net: params and GFLOPs of its network at 640x640, and the time of one sliding-window
prediction of a 640x640 canvas (predict_single_npy_array, fold 0, checkpoint_final, no mirroring/TTA,
nnU-Net's own mixed precision), which includes its preprocessing and export.
"""
import argparse
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("YOLO_VERBOSE", "false")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from artigate.modules import register_modules


def build(path, nc):
    from ultralytics.nn.tasks import SegmentationModel, load_checkpoint

    if str(path).endswith(".pt"):
        return load_checkpoint(path)[0].float()
    return SegmentationModel(str(path), ch=3, nc=nc, verbose=False)


def baseline_flops(model, imgsz):
    """GFLOPs (2 x MACs, as Ultralytics) of a baseline network at imgsz x imgsz."""
    from copy import deepcopy

    import thop

    p = next(model.parameters())
    return thop.profile(deepcopy(model), inputs=[torch.zeros(1, 3, imgsz, imgsz, device=p.device)], verbose=False)[0] / 1e9 * 2


def time_forward(model, x, warmup, runs, device):
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - start) / runs * 1000


def nnunet_row(model_dir, imgsz, warmup, runs, device):
    """(params M, GFLOPs, ms) of nnU-Net v2: one sliding-window prediction of a canvas-sized image."""
    import numpy as np
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                                perform_everything_on_device=device.type == "cuda", device=device,
                                verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(model_dir), use_folds=(0,), checkpoint_name="checkpoint_final.pth")
    network = predictor.network
    params = sum(p.numel() for p in network.parameters()) / 1e6
    try:
        gflops = baseline_flops(network.to(device).eval(), imgsz)
    except Exception:  # noqa: BLE001 - a plan whose strides do not divide imgsz
        gflops = float("nan")
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (3, 1, imgsz, imgsz)).astype(np.float32)  # NaturalImage2DIO layout (c, 1, h, w)
    properties = {"spacing": (999, 1, 1)}
    for _ in range(warmup):
        predictor.predict_single_npy_array(image, properties, None, None, False)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        predictor.predict_single_npy_array(image, properties, None, None, False)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return params, gflops, (time.perf_counter() - start) / runs * 1000


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--baselines", nargs="*", default=[], choices=("unet", "malunet", "egeunet"))
    parser.add_argument("--nnunet-model-dir", type=Path, default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--nc", type=int, default=1)
    args = parser.parse_args(argv)

    register_modules()
    from ultralytics.utils.torch_utils import get_flops

    if args.device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(f"cuda:{args.device}" if str(args.device).isdigit() else args.device)
    print(f"{'model':<45}{'params(M)':>10}{'GFLOPs':>9}{'ms/img':>9}{'FPS':>8}")
    for path in args.models:
        model = build(path, args.nc).eval()
        model = model.fuse(verbose=False) if hasattr(model, "fuse") else model
        params = sum(p.numel() for p in model.parameters()) / 1e6
        gflops = get_flops(model, args.imgsz)
        model = model.to(device)
        x = torch.zeros(1, 3, args.imgsz, args.imgsz, device=device)
        if args.half and device.type == "cuda":
            model, x = model.half(), x.half()
        with torch.inference_mode():
            for _ in range(args.warmup):
                model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(args.runs):
                model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
        ms = (time.perf_counter() - start) / args.runs * 1000
        print(f"{Path(path).name:<45}{params:>10.3f}{gflops:>9.2f}{ms:>9.2f}{1000 / ms:>8.1f}")
    for name in args.baselines:
        from artigate.baselines.models import build_model
        from artigate.baselines.recipe import load_recipe

        model = build_model(name, load_recipe(name)[0]["model"]["kwargs"]).eval().to(device)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        gflops = baseline_flops(model, args.imgsz)
        x = torch.zeros(1, 3, args.imgsz, args.imgsz, device=device)
        if args.half and device.type == "cuda":
            model, x = model.half(), x.half()
        ms = time_forward(model, x, args.warmup, args.runs, device)
        print(f"{name:<45}{params:>10.3f}{gflops:>9.2f}{ms:>9.2f}{1000 / ms:>8.1f}")
    if args.nnunet_model_dir is not None:
        params, gflops, ms = nnunet_row(args.nnunet_model_dir, args.imgsz, args.warmup, args.runs, device)
        print(f"{'nnU-Net 2D (sliding window, no TTA)':<45}{params:>10.3f}{gflops:>9.2f}{ms:>9.2f}{1000 / ms:>8.1f}")
    print(f"batch 1, {args.imgsz}x{args.imgsz}, {'FP16' if args.half else 'FP32'}, fused, {device}")


if __name__ == "__main__":
    main()
