"""How well does the ArtifactHead find artifacts? Held-out synthetic artifacts with fixed seeds.

For each exported image (640 canvas) artifacts are composited inside the content area with a fixed
per-image seed (different from training), the model runs once, and each predicted map (p > --thr)
is compared with the exact synthetic mask; anti-aliased edges are ignored. Per type it reports IoU
(summed intersection / summed union), the mean per-image Dice and the share of pixels predicted
positive on the untouched image ("activation on clean"). Real hair or rulers already in the photo
count as false positives, so IoU is a lower bound: use it to compare models.

    python scripts/eval_artifact_head.py --weights runs/segment/A4_s0/weights/best.pt --split val --n 100
"""
import argparse
import csv
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
from artigate.data.artifact_dataset import content_boxes
from artigate.data.dataset_yaml import load_dataset_yaml, split_dir
from artigate.data.masks import read_label_mask
from artigate.data.synth_artifacts import TYPES, composite
from artigate.modules import register_modules

DEFAULT_DATA = ROOT / "configs" / "data" / "isic2018.yaml"


@torch.inference_mode()
def artifact_probabilities(model, image, device):
    """Artifact probabilities (K, H, W) for a square canvas image (BGR uint8)."""
    x = torch.from_numpy(np.ascontiguousarray(image[..., ::-1].transpose(2, 0, 1))).to(device).float()[None] / 255
    model(x)
    probabilities = torch.sigmoid(artifact_logits(model)[0].float())
    return torch.nn.functional.interpolate(probabilities[None], size=image.shape[:2], mode="bilinear",
                                           align_corners=False)[0].cpu().numpy()


def lesion_box(label_path, size):
    if not label_path.is_file():
        return None
    mask = read_label_mask(label_path, size, size)
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def pick_device(value):
    if value is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(f"cuda:{value}" if str(value).isdigit() else value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--severity", nargs=2, type=int, default=[1, 4])
    parser.add_argument("--thr", type=float, default=0.5)
    parser.add_argument("--seed-base", type=int, default=20000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", help="optional per-image CSV")
    args = parser.parse_args(argv)

    register_modules()
    from ultralytics.nn.tasks import load_checkpoint

    device = pick_device(args.device)
    model = load_checkpoint(args.weights)[0].float().eval().to(device)
    config, root = load_dataset_yaml(args.data)
    image_dir, label_dir = split_dir(root, config, args.split, "images"), split_dir(root, config, args.split, "labels")
    content = content_boxes(image_dir)
    files = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[: args.n]
    k = len(TYPES)
    inter, union, count = np.zeros(k), np.zeros(k), np.zeros(k, int)
    dice, clean = [[] for _ in TYPES], [[] for _ in TYPES]
    rows = []
    for index, path in enumerate(files):
        image = cv2.imread(str(path))
        size = image.shape[0]
        for t, p in enumerate(artifact_probabilities(model, image, device) > args.thr):
            clean[t].append(p.mean())
        (x1, y1, x2, y2), _ = content.get(path.stem, ((0, 0, size, size), size))
        box = lesion_box(label_dir / f"{path.stem}.txt", size)
        local = None if box is None else (box[0] - x1, box[1] - y1, box[2] - x1, box[3] - y1)
        synthetic = image.copy()
        masks = np.zeros(image.shape[:2] + (k,), bool)
        touched = np.zeros_like(masks)
        m, tch, applied = composite(synthetic[y1:y2, x1:x2], np.random.default_rng(args.seed_base + index), local,
                                    severity=tuple(args.severity))
        masks[y1:y2, x1:x2], touched[y1:y2, x1:x2] = m, tch
        predicted = artifact_probabilities(model, synthetic, device) > args.thr
        row = {"image_id": path.stem, "applied": "+".join(applied)}
        for t, name in enumerate(TYPES):
            if name not in applied:
                continue
            valid = ~(touched[..., t] & ~masks[..., t])
            truth, pred = masks[..., t] & valid, predicted[t] & valid
            i, u = (truth & pred).sum(), (truth | pred).sum()
            inter[t] += i
            union[t] += u
            count[t] += 1
            d = 2 * i / max(truth.sum() + pred.sum(), 1)
            dice[t].append(d)
            row[f"dice_{name}"] = round(float(d), 4)
        rows.append(row)
    print(f"{'type':<12}{'images':>7}{'IoU':>8}{'Dice':>8}{'activation on clean':>21}")
    for t, name in enumerate(TYPES):
        iou = inter[t] / union[t] if union[t] else float("nan")
        mean_dice = float(np.mean(dice[t])) if dice[t] else float("nan")
        print(f"{name:<12}{count[t]:>7}{iou:>8.3f}{mean_dice:>8.3f}{100 * np.mean(clean[t]):>20.2f}%")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["image_id", "applied", *[f"dice_{n}" for n in TYPES]])
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
