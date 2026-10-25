#!/usr/bin/env python
"""Fast image-level artifact tagging (runs on your laptop, needs a display).

Shows one image at a time. Keys:
    h = hair   r = ruler   m = marker/ink   d = dark corner   (toggle)
    Enter / Space = save and next      b = back      q = quit (progress is saved)
About 3-5 s per image, so the 1,000 ISIC 2018 test images take roughly 1-1.5 h.
Two people labelling the same 200 images gives an agreement (Cohen's kappa) for the paper.

    python scripts/label_artifacts.py --images data/yolo/isic2018/images/test \
        --out data/splits/isic2018_test_artifacts_hai.csv
The image_id written is the file stem (e.g. ISIC_0012345_v1); the view suffix is ignored when the
flags are read (artigate.data.artifact_dataset.sample_id), so one label file serves every view.
"""

import argparse
import csv
from pathlib import Path

import cv2

TYPES = ("hair", "ruler", "marker", "dark_corner")
KEYS = {ord("h"): 0, ord("r"): 1, ord("m"): 2, ord("d"): 3}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=900)
    a = ap.parse_args()

    files = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
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

    i = next((j for j, p in enumerate(files) if p.stem not in labels), len(files))
    while 0 <= i < len(files):
        p = files[i]
        cur = list(labels.get(p.stem, [0, 0, 0, 0]))
        img = cv2.imread(str(p))
        r = a.size / max(img.shape[:2])
        img = cv2.resize(img, None, fx=r, fy=r) if r < 1 else img
        while True:
            vis = img.copy()
            txt = "  ".join(f"[{'x' if cur[k] else ' '}] {t}" for k, t in enumerate(TYPES))
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 34), (0, 0, 0), -1)
            cv2.putText(vis, f"{i + 1}/{len(files)}  {txt}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("label_artifacts  h r m d | Enter next | b back | q quit", vis)
            key = cv2.waitKey(0) & 0xFF
            if key in KEYS:
                cur[KEYS[key]] ^= 1
            elif key in (13, 10, 32):
                labels[p.stem] = cur
                save()
                i += 1
                break
            elif key == ord("b"):
                i = max(0, i - 1)
                break
            elif key == ord("q"):
                save()
                cv2.destroyAllWindows()
                print(f"saved {len(labels)} labels to {out}")
                return
    save()
    cv2.destroyAllWindows()
    print(f"done: {len(labels)} labels in {out}")


if __name__ == "__main__":
    main()
