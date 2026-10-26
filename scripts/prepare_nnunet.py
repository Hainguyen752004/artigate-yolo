"""nnU-Net v2 dataset from the exported canvases, with one fold = the official train/val split.

    export nnUNet_raw=... nnUNet_preprocessed=... nnUNet_results=...
    python scripts/prepare_nnunet.py dataset --data configs/data/isic2018.yaml
    nnUNetv2_plan_and_preprocess -d 501 -c 2d --verify_dataset_integrity
    nnUNetv2_train 501 2d 0                                  # default trainer, fold 0 = splits_final.json
    nnUNetv2_predict -i $nnUNet_raw/Dataset501_ISIC2018Canvas/imagesVal -o <pred>/val -d 501 -c 2d -f 0 \
        --disable_tta -chk checkpoint_final.pth
    python scripts/eval_seg_baseline.py --pred-dir <pred>/val --run nnunet_s0 --split val \
        --checkpoint $nnUNet_results/Dataset501_ISIC2018Canvas/nnUNetTrainer__nnUNetPlans__2d/fold_0/checkpoint_final.pth

"dataset" writes Dataset<id>_<name> under $nnUNet_raw: imagesTr (train + val canvases as RGB PNG,
<stem>_0000.png), labelsTr (0/1 PNG), imagesTs (test) and imagesVal (val only, for predicting the val
split), dataset.json (channels R, G, B; labels background/lesion), and splits_final.json under
$nnUNet_preprocessed with one fold: train = the official training images, val = the official validation
images (CLAUDE.md, "Baseline ngoài YOLO"). nnU-Net creates splits only when that file is missing, and
plan_and_preprocess leaves it in place.

"images" writes the images of any exported split (external sets, severity sets) as nnU-Net input:
    python scripts/prepare_nnunet.py images --data configs/data/ph2.yaml --split test --out <dir>
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image

from artigate.baselines.data import canvas_items
from artigate.data.masks import read_mask

DEFAULT_DATA = ROOT / "configs" / "data" / "isic2018.yaml"


def _project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def write_images(items, folder):
    """Canvas images as RGB PNG named <stem>_0000.png (one file holds the three channels)."""
    folder.mkdir(parents=True, exist_ok=True)
    for image_path, _ in items:
        Image.open(image_path).convert("RGB").save(folder / f"{image_path.stem}_0000.png")
    return [p.stem for p, _ in items]


def write_labels(items, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for image_path, mask_path in items:
        Image.fromarray(read_mask(mask_path).astype(np.uint8)).save(folder / f"{image_path.stem}.png")


def dataset(args):
    raw, preprocessed = args.raw or os.environ.get("nnUNet_raw"), args.preprocessed or os.environ.get("nnUNet_preprocessed")
    if not raw or not preprocessed:
        raise SystemExit("set nnUNet_raw and nnUNet_preprocessed (or pass --raw / --preprocessed)")
    raw, preprocessed = Path(raw), Path(preprocessed)
    name = f"Dataset{args.id:03d}_{args.name}"
    root = raw / name
    train, val, test = (canvas_items(args.data, split) for split in ("train", "val", "test"))
    train_ids = write_images(train, root / "imagesTr")
    val_ids = write_images(val, root / "imagesTr")
    write_labels(train + val, root / "labelsTr")
    write_images(val, root / "imagesVal")
    write_images(test, root / "imagesTs")
    (root / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "R", "1": "G", "2": "B"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": len(train_ids) + len(val_ids),
        "file_ending": ".png",
        "source": f"canvases of {args.data}",
    }, indent=2), encoding="utf-8")
    splits = preprocessed / name / "splits_final.json"
    splits.parent.mkdir(parents=True, exist_ok=True)
    splits.write_text(json.dumps([{"train": train_ids, "val": val_ids}], indent=2), encoding="utf-8")
    print(f"{root}: {len(train_ids)} train + {len(val_ids)} val in imagesTr, {len(test)} test; splits in {splits}")
    return root, splits


def images(args):
    ids = write_images(canvas_items(args.data, args.split), args.out)
    print(f"{len(ids)} images in {args.out}")
    return ids


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("dataset", help="nnU-Net raw dataset and splits_final.json")
    make.add_argument("--data", type=_project_path, default=DEFAULT_DATA)
    make.add_argument("--id", type=int, default=501)
    make.add_argument("--name", default="ISIC2018Canvas")
    make.add_argument("--raw", default=None, help="default: $nnUNet_raw")
    make.add_argument("--preprocessed", default=None, help="default: $nnUNet_preprocessed")
    export = sub.add_parser("images", help="images of one exported split as nnU-Net input")
    export.add_argument("--data", type=_project_path, required=True)
    export.add_argument("--split", required=True, choices=("train", "val", "test"))
    export.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    return dataset(args) if args.command == "dataset" else images(args)


if __name__ == "__main__":
    main()
