"""Artifact severity-sweep copies of an exported split (stress test; the key claims need real artifacts).

For each artifact type (and "mix") and severity 1-4 every image gets artifacts composited inside
its letterbox content area with a fixed per-image seed, then is saved again as JPEG. Labels, masks
and metadata are copied unchanged (only hair may cover the lesion). Severity 0 is the original split.
Seeds start at --seed-base, far from anything used in training.

Each output folder is a dataset root with its own data.yaml, so scripts/eval.py can score it:
    python scripts/make_severity_sets.py --data configs/data/isic2018.yaml --split test --out data/yolo/isic2018_sev
    python scripts/eval.py --weights <best.pt> --data data/yolo/isic2018_sev/hair_s3/data.yaml --split test
"""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault("YOLO_VERBOSE", "false")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
import yaml

from artigate.data.artifact_dataset import content_boxes
from artigate.data.dataset_yaml import load_dataset_yaml, split_dir
from artigate.data.masks import read_label_mask
from artigate.data.synth_artifacts import TYPES, composite


def seed_for(stem, kind, severity, base):
    return base + int(hashlib.md5(f"{stem}|{kind}|{severity}".encode()).hexdigest()[:8], 16)


def lesion_box(label_path, size):
    if not label_path.is_file():
        return None
    mask = read_label_mask(label_path, size, size)
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(ROOT / "configs" / "data" / "isic2018.yaml"))
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--types", nargs="+", default=[*TYPES, "mix"])
    parser.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--seed-base", type=int, default=10000)
    parser.add_argument("--quality", type=int, default=95)
    args = parser.parse_args(argv)

    config, root = load_dataset_yaml(args.data)
    image_dir = split_dir(root, config, args.split, "images")
    label_dir = split_dir(root, config, args.split, "labels")
    mask_dir = split_dir(root, config, args.split, "masks")
    metadata = root / "metadata" / f"{args.split}.jsonl"
    content = content_boxes(image_dir)
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    for kind in args.types:
        if kind != "mix" and kind not in TYPES:
            raise SystemExit(f"unknown type {kind}; choose from {TYPES} or mix")
        for severity in args.severities:
            out = Path(args.out) / f"{kind}_s{severity}"
            for sub in ("images", "labels", "masks"):
                (out / sub / args.split).mkdir(parents=True, exist_ok=True)
            for path in images:
                image = cv2.imread(str(path))
                size = image.shape[0]
                (x1, y1, x2, y2), _ = content.get(path.stem, ((0, 0, image.shape[1], size), size))
                box = lesion_box(label_dir / f"{path.stem}.txt", size)
                local = None if box is None else (box[0] - x1, box[1] - y1, box[2] - x1, box[3] - y1)
                rng = np.random.default_rng(seed_for(path.stem, kind, severity, args.seed_base))
                crop = image[y1:y2, x1:x2]
                if kind == "mix":
                    composite(crop, rng, local, probs=(0.6, 0.4, 0.4, 0.4), severity=severity)
                else:
                    composite(crop, rng, local, severity=severity, force=kind)
                cv2.imwrite(str(out / "images" / args.split / path.name), image, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                for directory, suffix, target in ((label_dir, ".txt", "labels"), (mask_dir, ".png", "masks")):
                    source = directory / f"{path.stem}{suffix}"
                    if source.is_file():
                        shutil.copy(source, out / target / args.split / source.name)
            if metadata.is_file():
                (out / "metadata").mkdir(exist_ok=True)
                shutil.copy(metadata, out / "metadata" / metadata.name)
            (out / "data.yaml").write_text(yaml.safe_dump(
                {"path": ".", args.split: f"images/{args.split}", "names": {0: "lesion"}}, sort_keys=False), encoding="utf-8")
            print(f"{out.name}: {len(images)} images")


if __name__ == "__main__":
    main()
