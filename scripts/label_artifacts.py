#!/usr/bin/env python
"""Fast image-level artifact tagging (runs on your laptop, needs a display).

Shows one image at a time, scaled to fit --size. Keys:
    h = hair   r = ruler   m = marker/ink   d = dark corner   (toggle)
    Enter / Space = save and next      b = back      q = quit (progress is saved)
About 3-5 s per image, so the 1,000 ISIC 2018 test images take roughly 1-1.5 h.
Definitions and the procedure are in docs/annotation_test_artifacts.md: read them first.

Annotator 1 labels every test image from the original files (better for faint hair):
    python scripts/label_artifacts.py --manifest data/splits/isic2018.csv --split test \\
        --out data/splits/isic2018_test_artifacts_a1.csv
Annotator 2 labels only the agreement subset (200 images, fixed in advance):
    python scripts/label_artifacts.py --manifest data/splits/isic2018.csv --split test \\
        --ids data/splits/isic2018_test_kappa_subset.csv --out data/splits/isic2018_test_artifacts_a2.csv
--images <folder> still works for exported canvases. The image_id written is the sample id (or the
file stem); view suffixes are ignored when the flags are read, so one label file serves every view.
"""

import argparse
import csv
from pathlib import Path
import re

import cv2

TYPES = ("hair", "ruler", "marker", "dark_corner")
KEYS = {ord(c): k for k, c in enumerate("hrmd")}
KEYS.update({ord(c.upper()): k for k, c in enumerate("hrmd")})  # Caps Lock on
BAR = 34  # status bar height, drawn above the image so it hides nothing


ROOT = Path(__file__).resolve().parents[1]


def _base_id(name):
    return re.sub(r"_v\d+$", "", name)


def render(img, cur, i, n):
    """The image with the status bar added on top (the image pixels are left untouched)."""
    vis = cv2.copyMakeBorder(img, BAR, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    txt = "  ".join(f"[{'x' if cur[k] else ' '}] {t}" for k, t in enumerate(TYPES))
    cv2.putText(vis, f"{i + 1}/{n}  {txt}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return vis


def image_list(images=None, manifest=None, split="test", raw_root="data/raw", ids=None):
    """[(image_id, path)] sorted by id, optionally restricted to the ids listed in a CSV."""
    if manifest:
        with open(manifest, newline="", encoding="utf-8") as f:
            items = [(r["sample_id"], Path(raw_root) / r["image"]) for r in csv.DictReader(f) if r["split"] == split]
    else:
        items = [(p.stem, p) for p in Path(images).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    if ids:
        with open(ids, newline="", encoding="utf-8") as f:
            wanted = {_base_id(r["image_id"]) for r in csv.DictReader(f)}
        items = [(i, p) for i, p in items if _base_id(i) in wanted]
    return sorted(items)


def unused_types(labels, min_images=20):
    """Types never marked present after at least ``min_images`` labels: often a key the input method swallowed."""
    if len(labels) < min_images:
        return []
    return [t for k, t in enumerate(TYPES) if not any(v[k] for v in labels.values())]


def warn_unused(labels):
    for t in unused_types(labels):
        print(f"WARNING: no image marked '{t}' in {len(labels)} labels. If that is unexpected, turn off the "
              f"Vietnamese input method (Unikey/EVKey -> E) and check that its key shows [x] in the status bar.\n"
              f"CANH BAO: chua co anh nao duoc danh dau '{t}'. Kiem tra bo go tieng Viet da tat chua.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", help="folder of images (e.g. an exported canvas split)")
    ap.add_argument("--manifest", help="split manifest; shows the original images of --split")
    ap.add_argument("--split", default="test")
    ap.add_argument("--raw-root", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--ids", help="CSV with an image_id column: label only these images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=900)
    a = ap.parse_args()
    if bool(a.images) == bool(a.manifest):
        ap.error("give exactly one of --images and --manifest")

    files = image_list(a.images, a.manifest, a.split, a.raw_root, a.ids)
    if not files:
        raise SystemExit("no images to label")
    out = Path(a.out)
    labels = {}
    if out.exists():
        with open(out, newline="") as f:
            labels = {r["image_id"]: [int(r[t]) for t in TYPES] for r in csv.DictReader(f)}

    def save():
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_id", *TYPES])
            for k in sorted(labels):
                w.writerow([k, *labels[k]])

    i = next((j for j, (k, _) in enumerate(files) if k not in labels), len(files))
    while 0 <= i < len(files):
        key_id, p = files[i]
        cur = list(labels.get(key_id, [0, 0, 0, 0]))
        img = cv2.imread(str(p))
        if img is None:
            raise SystemExit(f"cannot read {p}")
        r = a.size / max(img.shape[:2])
        img = cv2.resize(img, None, fx=r, fy=r, interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
        while True:
            cv2.imshow("label_artifacts  h r m d | Enter next | b back | q quit", render(img, cur, i, len(files)))
            key = cv2.waitKey(0) & 0xFF
            if key in KEYS:
                cur[KEYS[key]] ^= 1
            elif key in (13, 10, 32):
                labels[key_id] = cur
                save()
                i += 1
                break
            elif key in (ord("b"), ord("B")):
                i = max(0, i - 1)
                break
            elif key in (ord("q"), ord("Q")):
                save()
                cv2.destroyAllWindows()
                print(f"saved {len(labels)} labels to {out}")
                warn_unused(labels)
                return
    save()
    cv2.destroyAllWindows()
    print(f"done: {len(labels)} labels in {out}")
    warn_unused(labels)


if __name__ == "__main__":
    main()
