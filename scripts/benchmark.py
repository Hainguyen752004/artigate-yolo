"""Parameters, GFLOPs and forward latency for model YAMLs or checkpoints (same settings for every row).

    python scripts/benchmark.py --models configs/models/*.yaml --device 0
    python scripts/benchmark.py --models runs/segment/A4_s0/weights/best.pt --device 0 --half
Forward pass only (no NMS or mask decoding), batch 1, fused, after warm-up.
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True)
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
    print(f"batch 1, {args.imgsz}x{args.imgsz}, {'FP16' if args.half else 'FP32'}, fused, {device}")


if __name__ == "__main__":
    main()
