"""Write the PH2 manifest (data/splits/ph2.csv): 200 dermoscopy images, external test set only.

Expects the PH2 release under data/raw/ph2/: "PH2 Dataset images/IMDxxx/IMDxxx_Dermoscopic_Image/
IMDxxx.bmp" with its lesion mask "IMDxxx_lesion/IMDxxx_lesion.bmp". PH2 may not be redistributed,
so only the manifest goes into git.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from artigate.data.export import check_manifest, write_manifest

EXPECTED_COUNT = 200


def _project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def find_rows(raw_root, dataset_dir):
    """One test row per PH2 case; returns (rows, problems)."""
    image_root = dataset_dir / "PH2 Dataset images"
    if not image_root.is_dir():
        raise FileNotFoundError(f"PH2 image folder is missing: {image_root}")
    rows, problems = [], []
    for case in sorted(path for path in image_root.iterdir() if path.is_dir()):
        image = case / f"{case.name}_Dermoscopic_Image" / f"{case.name}.bmp"
        mask = case / f"{case.name}_lesion" / f"{case.name}_lesion.bmp"
        missing = [path.name for path in (image, mask) if not path.is_file()]
        if missing:
            problems.append(f"{case.name}: missing {missing}")
            continue
        rows.append({
            "sample_id": case.name,
            "split": "test",
            "image": image.relative_to(raw_root).as_posix(),
            "mask": mask.relative_to(raw_root).as_posix(),
            "group": case.name,
        })
    return rows, problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", type=_project_path, default=ROOT / "data" / "raw")
    parser.add_argument("--dataset-dir", type=_project_path, default=None, help="default: <raw-root>/ph2")
    parser.add_argument("--out", type=_project_path, default=ROOT / "data" / "splits" / "ph2.csv")
    parser.add_argument("--no-count-check", action="store_true", help="skip the 200-case check")
    args = parser.parse_args(argv)

    rows, problems = find_rows(args.raw_root, args.dataset_dir or args.raw_root / "ph2")
    if problems:
        raise ValueError(f"{len(problems)} incomplete cases, e.g. {problems[:5]}")
    if not args.no_count_check and len(rows) != EXPECTED_COUNT:
        raise ValueError(f"found {len(rows)} cases, the PH2 release has {EXPECTED_COUNT}")
    check_manifest(rows)
    write_manifest(args.out, rows)
    print(f"[INFO] Wrote {len(rows)} test rows to {args.out}")
    return rows


if __name__ == "__main__":
    main()
