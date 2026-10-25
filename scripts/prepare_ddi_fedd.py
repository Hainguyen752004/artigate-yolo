"""Write the DDI + FEDD manifest (data/splits/ddi_fedd.csv): external test set only.

FEDD (Carrion et al., MICCAI 2023) annotates 333 DDI images with label maps (0 background, 1 lesion,
2 marker, 3 ruler, 4 skin) saved as 256x256 .npy files, some of them in several split folders.
FEDD resized every image to 256x256 with PIL without keeping the aspect ratio, so each label map is
resized back to its image size with nearest-neighbour interpolation. The lesion, marker and ruler
masks are written as PNGs under <raw-root>/ddi/fedd_masks/, next to the images, because DDI may not
be redistributed: only the manifests go into git.

Also writes data/splits/ddi_fedd_artifacts.csv (image_id,marker,ruler with 1/0, usable with
--strata) and data/splits/ddi_fedd_skin_tone.csv (image_id,skin_tone from the DDI metadata).
Expected layout: <raw-root>/ddi/DDI_images/*.png, ddi_metadata.csv, fedd/ddi_labels/**/*.npy.
"""
import argparse
import csv
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from artigate.data.export import check_manifest, write_manifest
from artigate.data.masks import write_mask

FEDD_CLASSES = {"lesion": 1, "marker": 2, "ruler": 3}
EXPECTED_COUNT = 333


def _project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_label_maps(label_dir):
    """FEDD label maps by DDI file stem; copies in several split folders must be identical."""
    maps, conflicts = {}, []
    for path in sorted(label_dir.rglob("*.npy")):
        label = np.load(path, allow_pickle=False)
        if path.stem in maps and not np.array_equal(maps[path.stem], label):
            conflicts.append(path.stem)
        maps.setdefault(path.stem, label)
    if conflicts:
        raise ValueError(f"FEDD label maps differ between split folders for {sorted(set(conflicts))[:5]}")
    return maps


def read_skin_tones(metadata_csv):
    with metadata_csv.open(newline="", encoding="utf-8") as stream:
        return {Path(row["DDI_file"]).stem: row["skin_tone"] for row in csv.DictReader(stream)}


def build(raw_root, dataset_dir):
    """Write the full-size masks and return (manifest rows, artifact flags, skin tones)."""
    image_dir = dataset_dir / "DDI_images"
    mask_root = dataset_dir / "fedd_masks"
    skin_tones = read_skin_tones(image_dir / "ddi_metadata.csv")
    rows, flags, tones = [], [], []
    for stem, label in sorted(load_label_maps(dataset_dir / "fedd" / "ddi_labels").items()):
        image_path = image_dir / f"{stem}.png"
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"DDI image for FEDD mask {stem} is missing: {image_path}")
        height, width = image.shape[:2]
        full = cv2.resize(label.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        for name, value in FEDD_CLASSES.items():
            (mask_root / name).mkdir(parents=True, exist_ok=True)
            write_mask(mask_root / name / f"{stem}.png", full == value)
        sample_id = f"DDI_{stem}"
        rows.append({
            "sample_id": sample_id,
            "split": "test",
            "image": image_path.relative_to(raw_root).as_posix(),
            "mask": (mask_root / "lesion" / f"{stem}.png").relative_to(raw_root).as_posix(),
            "group": sample_id,
        })
        flags.append({"image_id": sample_id, "marker": int((full == 2).any()), "ruler": int((full == 3).any())})
        tones.append({"image_id": sample_id, "skin_tone": skin_tones[stem]})
    return rows, flags, tones


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", type=_project_path, default=ROOT / "data" / "raw")
    parser.add_argument("--dataset-dir", type=_project_path, default=None, help="default: <raw-root>/ddi")
    parser.add_argument("--splits-dir", type=_project_path, default=ROOT / "data" / "splits")
    parser.add_argument("--no-count-check", action="store_true", help="skip the 333-image check")
    args = parser.parse_args(argv)

    rows, flags, tones = build(args.raw_root, args.dataset_dir or args.raw_root / "ddi")
    if not args.no_count_check and len(rows) != EXPECTED_COUNT:
        raise ValueError(f"found {len(rows)} FEDD-annotated images, the FEDD release has {EXPECTED_COUNT}")
    check_manifest(rows)
    write_manifest(args.splits_dir / "ddi_fedd.csv", rows)
    _write_csv(args.splits_dir / "ddi_fedd_artifacts.csv", flags)
    _write_csv(args.splits_dir / "ddi_fedd_skin_tone.csv", tones)
    print(f"[INFO] Wrote {len(rows)} test rows to {args.splits_dir / 'ddi_fedd.csv'}: "
          f"{sum(f['marker'] for f in flags)} with marker, {sum(f['ruler'] for f in flags)} with ruler")
    return rows


if __name__ == "__main__":
    main()
