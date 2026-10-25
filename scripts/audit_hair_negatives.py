"""How many real hair pixels did training label "absent"? Runs on the "seen" images of the real-hair set.

During training the hair channel of an ISIC 2018 image is "absent" where auto_negatives finds no thin
dark structure, and everywhere if the image-level hair flag is 0. No real hair pixel is ever labelled
"present". This script maps each 512x512 hair mask back onto the training canvas (stretch back to the
original size, then the same letterbox as the export) and counts the hair pixels that auto_negatives
marks absent, and the images whose flag would mark all of them absent.

The mapping is checked per image: the 512x512 image, mapped the same way, must correlate with the
exported training image (--min-corr); images below it are reported and left out.

    python scripts/audit_hair_negatives.py --out runs/audit/hair_negatives.csv
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from artigate.data.artifact_dataset import auto_negatives, load_flags
from artigate.data.export import read_manifest
from artigate.data.masks import read_mask
from artigate.data.views import letterbox


def _project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def to_canvas(image, source_size, canvas_size, interpolation):
    """Undo the 512x512 stretch, then apply the export letterbox."""
    restored = cv2.resize(image, tuple(source_size), interpolation=interpolation)
    color = 0 if image.ndim == 2 else (114, 114, 114)
    return letterbox(restored, canvas_size, color=color, interpolation=interpolation)[0]


def correlation(a, b):
    a, b = (cv2.resize(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() for x in (a, b))
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))


def audit_image(hossain_image, hair_mask, train_image, source_size, flag=None):
    """Per-image counts on the training canvas of size train_image.shape."""
    canvas = train_image.shape[0]
    mapped_image = to_canvas(hossain_image, source_size, canvas, cv2.INTER_LINEAR)
    hair = to_canvas(hair_mask.astype(np.uint8) * 255, source_size, canvas, cv2.INTER_NEAREST) > 127
    absent = auto_negatives(train_image)[..., 0]
    return {
        "corr": correlation(mapped_image, train_image),
        "hair_pixels": int(hair.sum()),
        "absent_auto": int((hair & absent).sum()),
        "flag_hair": "" if flag is None else int(flag),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hair-manifest", type=_project_path, default=ROOT / "data" / "splits" / "hair_hossain.csv")
    parser.add_argument("--raw-root", type=_project_path, default=ROOT / "data" / "raw")
    parser.add_argument("--train-export", type=_project_path, default=ROOT / "data" / "yolo" / "isic2018")
    parser.add_argument("--flags-csv", type=_project_path, default=ROOT / "data" / "splits" / "isic2018_train_flags.csv")
    parser.add_argument("--min-corr", type=float, default=0.95)
    parser.add_argument("--out", type=_project_path, default=ROOT / "runs" / "audit" / "hair_negatives.csv")
    args = parser.parse_args(argv)

    sizes = {}
    for line in (args.train_export / "metadata" / "train.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        sizes[record["sample_id"]] = record["source_size"]
    flags = load_flags(args.flags_csv) if args.flags_csv.is_file() else {}
    rows = []
    for item in read_manifest(args.hair_manifest):
        if item["subset"] != "seen":
            continue
        sample_id = item["sample_id"]
        train_image = cv2.imread(str(args.train_export / "images" / "train" / f"{sample_id}_v1.jpg"))
        flag = flags[sample_id][0] if sample_id in flags else None
        counts = audit_image(cv2.imread(str(args.raw_root / item["image"])), read_mask(args.raw_root / item["mask"]),
                             train_image, sizes[sample_id], flag)
        rows.append({"sample_id": sample_id, **counts})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    kept = [row for row in rows if row["corr"] >= args.min_corr and row["hair_pixels"] > 0]
    hair = np.array([row["hair_pixels"] for row in kept], float)
    absent = np.array([row["absent_auto"] for row in kept], float)
    per_image = absent / hair
    print(f"seen images {len(rows)}; aligned (corr >= {args.min_corr}) with hair {len(kept)}; "
          f"left out {len(rows) - len(kept)}: {[r['sample_id'] for r in rows if r not in kept][:10]}")
    print(f"auto_negatives: {absent.sum() / hair.sum():.2%} of all real hair pixels labelled absent; "
          f"per image median {np.median(per_image):.2%}, p90 {np.percentile(per_image, 90):.2%}, max {per_image.max():.2%}")
    if flags:
        zero = [row for row in kept if row["flag_hair"] == 0]
        print(f"image-level flags: {len(zero)} of {len(kept)} images have hair = 0, so all their hair pixels are absent")
        # A flag of 0 marks the whole hair channel absent, so the union counts every hair pixel of those images.
        union = sum(row["hair_pixels"] if row["flag_hair"] == 0 else row["absent_auto"] for row in kept)
        print(f"rules or flags: {union / hair.sum():.2%} of all real hair pixels labelled absent")
    else:
        print(f"image-level flags: not computed, {args.flags_csv} does not exist")
    print(f"[INFO] Wrote per-image counts to {args.out}")
    return rows


if __name__ == "__main__":
    main()
