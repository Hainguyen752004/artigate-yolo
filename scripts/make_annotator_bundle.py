"""Folder of images for an annotator who labels on their own computer, without the repository.

The images are the views label_artifacts.py shows with --manifest: the original scaled so its longer
side is 900 px (INTER_AREA down, INTER_LINEAR up), saved as JPEG quality 95; the tool shows them unchanged.
    test_<n>/     the agreement subset (data/splits/isic2018_test_kappa_subset.csv), or with --all-test
                  all 1,000 test images (for an annotator 1 who works without the repository)
    practice_20/  training images for practice: 4 flagged with each artifact type and 4 with none
                  (Bissoto flags, seed 20261026), listed in practice_ids.csv
plus a copy of label_artifacts.py. The instructions and launchers are added by hand.

    python scripts/make_annotator_bundle.py --out ../handoff/artigate_annotator2
    python scripts/make_annotator_bundle.py --all-test --out ../handoff/artigate_annotator1
"""
import argparse
import csv
from pathlib import Path
import random
import shutil

import cv2

ROOT = Path(__file__).resolve().parents[1]
SIZE = 900
SEED = 20261026
TYPES = ("hair", "ruler", "marker", "dark_corner")


def view(path, size=SIZE):
    img = cv2.imread(str(path))
    r = size / max(img.shape[:2])
    return cv2.resize(img, None, fx=r, fy=r, interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)


def practice_ids(flags, seed=SEED, per_type=4):
    rng = random.Random(seed)
    chosen = []
    for t in TYPES:
        chosen += rng.sample(sorted(r["image_id"] for r in flags if r[t] == "1" and r["image_id"] not in chosen), per_type)
    clean = sorted(r["image_id"] for r in flags if all(r[t] == "0" for t in TYPES))
    return sorted(chosen + rng.sample(clean, per_type))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--all-test", action="store_true", help="every test image instead of the agreement subset")
    args = parser.parse_args(argv)
    splits = ROOT / "data" / "splits"
    with (splits / "isic2018.csv").open(encoding="utf-8") as stream:
        manifest = {r["sample_id"]: r for r in csv.DictReader(stream)}
    if args.all_test:
        subset = sorted(i for i, r in manifest.items() if r["split"] == "test")
    else:
        with (splits / "isic2018_test_kappa_subset.csv").open(encoding="utf-8") as stream:
            subset = [r["image_id"] for r in csv.DictReader(stream)]
    with (splits / "isic2018_train_flags.csv").open(encoding="utf-8") as stream:
        practice = practice_ids(list(csv.DictReader(stream)))
    for folder, ids, split in ((f"test_{len(subset)}", subset, "test"), ("practice_20", practice, "train")):
        target = args.out / "images" / folder
        target.mkdir(parents=True, exist_ok=True)
        for image_id in ids:
            row = manifest[image_id]
            if row["split"] != split:
                raise SystemExit(f"{image_id} is in {row['split']}, expected {split}")
            cv2.imwrite(str(target / f"{image_id}.jpg"), view(args.raw_root / row["image"]), [cv2.IMWRITE_JPEG_QUALITY, 95])
    (args.out / "tool").mkdir(exist_ok=True)
    shutil.copy2(ROOT / "scripts" / "label_artifacts.py", args.out / "tool" / "label_artifacts.py")
    (args.out / "ket_qua").mkdir(exist_ok=True)
    with (args.out / "practice_ids.csv").open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows([["image_id"], *[[i] for i in practice]])
    print(f"{len(subset)} test and {len(practice)} practice images in {args.out}")


if __name__ == "__main__":
    main()
