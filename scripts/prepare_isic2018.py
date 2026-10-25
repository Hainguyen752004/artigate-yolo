"""Write the ISIC 2018 Task 1 split manifest (data/splits/isic2018.csv) from the official release.

Expects the six official zips extracted under data/raw/isic2018/, one folder per zip:
ISIC2018_Task1-2_<Split>_Input/ISIC_xxxxxxx.jpg and ISIC2018_Task1_<Split>_GroundTruth/
ISIC_xxxxxxx_segmentation.png for <Split> in Training, Validation, Test. The official split is
kept as released (2,594 / 100 / 1,000). ISIC 2018 has no lesion IDs, so each image is its own group.
"""
import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from artigate.data.export import check_manifest, write_manifest
from artigate.data.split import isic_id

SPLITS = {"Training": "train", "Validation": "val", "Test": "test"}
EXPECTED_COUNTS = {"train": 2594, "val": 100, "test": 1000}


def _project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _by_id(directory, pattern):
    return {isic_id(path.name): path for path in directory.glob(pattern) if isic_id(path.name)}


def find_pairs(raw_root, dataset_dir):
    """Pair every image with its mask per official split; returns (rows, problems)."""
    rows, problems = [], []
    for official, split in SPLITS.items():
        image_dir = dataset_dir / f"ISIC2018_Task1-2_{official}_Input"
        mask_dir = dataset_dir / f"ISIC2018_Task1_{official}_GroundTruth"
        missing = [str(directory) for directory in (image_dir, mask_dir) if not directory.is_dir()]
        if missing:
            raise FileNotFoundError("ISIC 2018 folders are missing: " + ", ".join(missing))
        images = _by_id(image_dir, "*.jpg")
        masks = _by_id(mask_dir, "*_segmentation.png")
        problems += [f"{split}: no mask for {sample_id}" for sample_id in sorted(images.keys() - masks.keys())]
        problems += [f"{split}: no image for {sample_id}" for sample_id in sorted(masks.keys() - images.keys())]
        for sample_id in sorted(images.keys() & masks.keys()):
            rows.append({
                "sample_id": sample_id,
                "split": split,
                "image": images[sample_id].relative_to(raw_root).as_posix(),
                "mask": masks[sample_id].relative_to(raw_root).as_posix(),
                "group": sample_id,
            })
    return rows, problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", type=_project_path, default=ROOT / "data" / "raw")
    parser.add_argument("--dataset-dir", type=_project_path, default=None, help="default: <raw-root>/isic2018")
    parser.add_argument("--out", type=_project_path, default=ROOT / "data" / "splits" / "isic2018.csv")
    parser.add_argument("--no-count-check", action="store_true", help="skip the official 2,594 / 100 / 1,000 check")
    args = parser.parse_args(argv)

    dataset_dir = args.dataset_dir or args.raw_root / "isic2018"
    rows, problems = find_pairs(args.raw_root, dataset_dir)
    if problems:
        raise ValueError(f"{len(problems)} unpaired files, e.g. {problems[:5]}")
    counts = Counter(row["split"] for row in rows)
    if not args.no_count_check and dict(counts) != EXPECTED_COUNTS:
        raise ValueError(f"split sizes {dict(counts)} differ from the official release {EXPECTED_COUNTS}")
    check_manifest(rows)
    write_manifest(args.out, rows)
    print(f"[INFO] Wrote {len(rows)} rows to {args.out}: {dict(counts)}")
    return rows


if __name__ == "__main__":
    main()
