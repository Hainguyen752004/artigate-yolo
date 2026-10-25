"""Write the HAM10000 manifest (data/splits/ham10000.csv): external test set, one image per lesion.

HAM10000 has several images of many lesions, so each lesion_id contributes only its image with the
smallest image_id. Masks are the Tschandl et al. lesion segmentations. Every row goes to the test
split; the extra column dx is the diagnosis from the metadata, for per-class reports.

Expects under <raw-root>/ham10000/: HAM10000_images_part_1/ and HAM10000_images_part_2/ (ISIC_*.jpg),
HAM10000_segmentations_lesion_tschandl/ (ISIC_*_segmentation.png) and HAM10000_metadata.csv.
HAM10000 is CC BY-NC 4.0; only the manifest goes into git.
"""
import argparse
from collections import Counter
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from artigate.data.export import check_manifest, write_manifest

IMAGE_DIRS = ("HAM10000_images_part_1", "HAM10000_images_part_2")
MASK_DIR = "HAM10000_segmentations_lesion_tschandl"


def _project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def first_image_per_lesion(metadata_csv):
    """lesion_id -> metadata row of its image with the smallest image_id."""
    with Path(metadata_csv).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    chosen = {}
    for row in sorted(rows, key=lambda r: r["image_id"]):
        chosen.setdefault(row["lesion_id"], row)
    return chosen, len(rows)


def find_rows(raw_root, dataset_dir):
    images = {path.stem: path for name in IMAGE_DIRS for path in (dataset_dir / name).glob("*.jpg")}
    chosen, n_images = first_image_per_lesion(dataset_dir / "HAM10000_metadata.csv")
    rows, problems = [], []
    for lesion_id, meta in sorted(chosen.items(), key=lambda item: item[1]["image_id"]):
        image_id = meta["image_id"]
        mask = dataset_dir / MASK_DIR / f"{image_id}_segmentation.png"
        if image_id not in images or not mask.is_file():
            problems.append(image_id)
            continue
        rows.append({
            "sample_id": image_id,
            "split": "test",
            "image": images[image_id].relative_to(raw_root).as_posix(),
            "mask": mask.relative_to(raw_root).as_posix(),
            "group": lesion_id,
            "dx": meta["dx"],
        })
    return rows, problems, n_images


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", type=_project_path, default=ROOT / "data" / "raw")
    parser.add_argument("--dataset-dir", type=_project_path, default=None, help="default: <raw-root>/ham10000")
    parser.add_argument("--out", type=_project_path, default=ROOT / "data" / "splits" / "ham10000.csv")
    args = parser.parse_args(argv)

    rows, problems, n_images = find_rows(args.raw_root, args.dataset_dir or args.raw_root / "ham10000")
    if problems:
        raise ValueError(f"{len(problems)} chosen images have no image file or mask, e.g. {problems[:5]}")
    check_manifest(rows)
    write_manifest(args.out, rows)
    print(f"[INFO] {n_images} images of {len(rows)} lesions in the metadata; wrote one image per lesion to {args.out}")
    print(f"[INFO] diagnoses: {dict(sorted(Counter(row['dx'] for row in rows).items()))}")
    return rows


if __name__ == "__main__":
    main()
