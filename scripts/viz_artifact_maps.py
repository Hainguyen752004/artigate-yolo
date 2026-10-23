"""Artifact maps predicted by the ArtifactHead next to the lesion prediction (paper figures).

Each output: [image | max_k artifact probability | per-type colours where p > 0.5 | top-1 lesion
outline]. Pick images by a fixed rule (e.g. the first N of each artifact stratum), not by hand.

    python scripts/viz_artifact_maps.py --weights runs/segment/A4_s0/weights/best.pt --split test --n 24
"""
import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("YOLO_VERBOSE", "false")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
import torch

from artigate.artifact_loss import artifact_logits
from artigate.data.dataset_yaml import load_dataset_yaml, split_dir
from artigate.modules import register_modules

COLOURS = np.array([(0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 0, 0)], np.float32)  # hair, ruler, marker, corner


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default=str(ROOT / "configs" / "data" / "isic2018.yaml"))
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--images", help="folder of images instead of --data/--split (e.g. a severity set)")
    parser.add_argument("--out", default=str(ROOT / "runs" / "viz" / "artifact_maps"))
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    register_modules()
    from ultralytics import YOLO

    if args.images:
        image_dir = Path(args.images)
    else:
        config, root = load_dataset_yaml(args.data)
        image_dir = split_dir(root, config, args.split, "images")
    files = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[: args.n]
    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")
    model = YOLO(args.weights, task="segment")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in files:
        image = cv2.imread(str(path))
        height, width = image.shape[:2]
        result = model.predict(str(path), imgsz=args.imgsz, conf=0.001, device=device, verbose=False, retina_masks=True)[0]
        logits = artifact_logits(model.predictor.model.model)
        if logits is None:
            raise SystemExit("this model has no ArtifactHead")
        prob = torch.sigmoid(logits[0].float()).cpu().numpy()
        # undo Ultralytics' letterbox (minimal padding to a multiple of the stride, centred)
        scale = args.imgsz / max(height, width)
        new_h, new_w = round(height * scale), round(width * scale)
        padded_h, padded_w = prob.shape[1] * 4, prob.shape[2] * 4
        top, left = (padded_h - new_h) // 2, (padded_w - new_w) // 2
        prob = np.stack([cv2.resize(cv2.resize(p, (padded_w, padded_h))[top:top + new_h, left:left + new_w], (width, height))
                         for p in prob])
        peak = prob.max(0)
        heat = cv2.applyColorMap((peak * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        types = image.astype(np.float32).copy()
        on = peak > 0.5
        types[on] = 0.4 * types[on] + 0.6 * COLOURS[prob.argmax(0)[on]]
        lesion = image.copy()
        if result.masks is not None and len(result.masks):
            best = int(result.boxes.conf.argmax())
            mask = cv2.resize((result.masks.data[best].cpu().numpy() > 0.5).astype(np.uint8), (width, height),
                              interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(lesion, contours, -1, (0, 255, 0), max(2, width // 300))
        cv2.imwrite(str(out / f"{path.stem}.jpg"), np.concatenate([image, heat, types.astype(np.uint8), lesion], 1))
    print(f"saved {len(files)} figures to {out}")


if __name__ == "__main__":
    main()
