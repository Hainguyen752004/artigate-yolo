"""Export (image, mask) pairs listed in a split manifest to a YOLO-seg dataset.

A manifest is a CSV with columns sample_id, split, image, mask, group. Image and mask paths are
relative to the raw-data root; group is the leakage unit (lesion_id for HAM10000) and defaults to
sample_id. Each exported sample gets images/<split>/<stem>.jpg, labels/<split>/<stem>.txt,
masks/<split>/<stem>.png (the ground truth used for evaluation) and one metadata line holding
the view transform.
"""
import csv
import json
from pathlib import Path

import cv2
from tqdm import tqdm

from artigate.data.masks import mask_to_polygons, polygons_to_label_lines, read_mask, write_mask
from artigate.data.split import group_leaks, id_overlaps
from artigate.data.views import make_view

MANIFEST_FIELDS = ("sample_id", "split", "image", "mask", "group")
KINDS = ("images", "labels", "masks")


def read_manifest(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = set(MANIFEST_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"manifest {path} is missing columns: {sorted(missing)}")
        rows = [dict(row) for row in reader]
    for row in rows:
        row["group"] = row["group"] or row["sample_id"]
    return rows


def write_manifest(path, rows):
    """Write the manifest columns, followed by any extra per-row columns (e.g. a subset label)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extras = [key for key in (rows[0] if rows else {}) if key not in MANIFEST_FIELDS]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*MANIFEST_FIELDS, *extras])
        writer.writeheader()
        writer.writerows(rows)


def check_manifest(rows):
    """Fail on duplicate sample IDs, groups spanning splits, or ISIC IDs shared between splits."""
    counts = {}
    for row in rows:
        counts[row["sample_id"]] = counts.get(row["sample_id"], 0) + 1
    duplicates = sorted(sample for sample, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate sample_id values: {duplicates[:10]}")
    leaks = group_leaks([row["group"] for row in rows], [row["split"] for row in rows])
    if leaks:
        raise ValueError(f"{len(leaks)} groups span several splits, e.g. {list(leaks.items())[:5]}")
    ids_by_split = {}
    for row in rows:
        ids_by_split.setdefault(row["split"], []).append(row["sample_id"])
    overlaps = id_overlaps(ids_by_split)
    if overlaps:
        examples = {pair: ids[:5] for pair, ids in overlaps.items()}
        raise ValueError(f"ISIC IDs shared between splits: {examples}")


def split_paths(out_root, split):
    out_root = Path(out_root)
    return [out_root / kind / split for kind in KINDS] + [out_root / "metadata" / f"{split}.jsonl"]


def export_split(rows, raw_root, out_root, split, views, size=640):
    """Write every view of every row; returns counts for the export summary."""
    raw_root, out_root = Path(raw_root), Path(out_root)
    image_dir, label_dir, mask_dir, metadata_path = split_paths(out_root, split)
    for directory in (image_dir, label_dir, mask_dir, metadata_path.parent):
        directory.mkdir(parents=True, exist_ok=True)
    summary = {"samples": len(rows), "images": 0, "empty_labels": [], "instances": 0}
    with metadata_path.open("w", encoding="utf-8", newline="\n") as metadata:
        for row in tqdm(rows, desc=f"export {split}"):
            image = cv2.imread(str(raw_root / row["image"]))
            if image is None:
                raise ValueError(f"unreadable image: {raw_root / row['image']}")
            mask = read_mask(raw_root / row["mask"])
            for view in views:
                view_image, view_mask, transform = make_view(image, mask, view, size)
                stem = f"{row['sample_id']}_{view}"
                polygons = mask_to_polygons(view_mask)
                if not cv2.imwrite(str(image_dir / f"{stem}.jpg"), view_image):
                    raise OSError(f"failed to write image: {image_dir / f'{stem}.jpg'}")
                (label_dir / f"{stem}.txt").write_text(
                    "\n".join(polygons_to_label_lines(polygons, size, size)), encoding="utf-8"
                )
                write_mask(mask_dir / f"{stem}.png", view_mask)
                record = {"stem": stem, "sample_id": row["sample_id"], "group": row["group"], "split": split,
                          "instances": len(polygons), **transform}
                metadata.write(json.dumps(record, sort_keys=True) + "\n")
                summary["images"] += 1
                summary["instances"] += len(polygons)
                if not polygons:
                    summary["empty_labels"].append(stem)
    return summary
