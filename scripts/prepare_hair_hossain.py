"""Write the manifest of the real-hair test set (data/splits/hair_hossain.csv).

The hair-mask dataset ("A skin lesion hair mask dataset with fine-grained annotations") has 500 ISIC
images, each stretched to 512x512 (verified: correlation >= 0.97 with the stretched original for all
493 images that are in ISIC 2018), and a hair mask drawn on that 512x512 image. The model is run on
these 512x512 images so that each mask matches its own image. Used only to evaluate the artifact head.

Extra columns: isic_split is the image's ISIC 2018 split (train, val, test or none); subset is "seen"
for images in the ISIC 2018 training set and "unseen" otherwise. Expects the dataset under
<raw-root>/hair_hossain/<release folder>/{dermoscopic_image,hair_mask}/ISIC_xxxxxxx.png.
"""
import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from artigate.data.export import check_manifest, read_manifest, write_manifest
from artigate.data.split import isic_id

RELEASE = "A skin lesion hair mask dataset with fine-grained annotations"
EXPECTED_COUNT = 500


def _project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def find_rows(raw_root, dataset_dir, isic_splits):
    rows, problems = [], []
    for image in sorted((dataset_dir / "dermoscopic_image").glob("*.png")):
        mask = dataset_dir / "hair_mask" / image.name
        sample_id = isic_id(image.name)
        if sample_id is None or not mask.is_file():
            problems.append(image.name)
            continue
        split = isic_splits.get(sample_id, "none")
        rows.append({
            "sample_id": sample_id,
            "split": "test",
            "image": image.relative_to(raw_root).as_posix(),
            "mask": mask.relative_to(raw_root).as_posix(),
            "group": sample_id,
            "subset": "seen" if split == "train" else "unseen",
            "isic_split": split,
        })
    return rows, problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", type=_project_path, default=ROOT / "data" / "raw")
    parser.add_argument("--dataset-dir", type=_project_path, default=None, help="default: <raw-root>/hair_hossain/<release>")
    parser.add_argument("--isic-manifest", type=_project_path, default=ROOT / "data" / "splits" / "isic2018.csv")
    parser.add_argument("--out", type=_project_path, default=ROOT / "data" / "splits" / "hair_hossain.csv")
    parser.add_argument("--no-count-check", action="store_true", help="skip the 500-image check")
    args = parser.parse_args(argv)

    isic_splits = {row["sample_id"]: row["split"] for row in read_manifest(args.isic_manifest)}
    dataset_dir = args.dataset_dir or args.raw_root / "hair_hossain" / RELEASE
    rows, problems = find_rows(args.raw_root, dataset_dir, isic_splits)
    if problems:
        raise ValueError(f"{len(problems)} images without an ISIC id or a hair mask, e.g. {problems[:5]}")
    if not args.no_count_check and len(rows) != EXPECTED_COUNT:
        raise ValueError(f"found {len(rows)} images, the release has {EXPECTED_COUNT}")
    check_manifest(rows)
    write_manifest(args.out, rows)
    print(f"[INFO] Wrote {len(rows)} rows to {args.out}: subsets {dict(Counter(r['subset'] for r in rows))}, "
          f"ISIC 2018 splits {dict(Counter(r['isic_split'] for r in rows))}")
    return rows


if __name__ == "__main__":
    main()
